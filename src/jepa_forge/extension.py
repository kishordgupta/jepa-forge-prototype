"""Repeated-split comparisons with independent selection and a global test gate."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import f_classif

from .compiler import compile_development_task, compile_task, _array_hash
from .datasets import load_dataset, make_splits
from .evaluation import _metrics
from .experiments import source_inventory
from .extension_models import train_checkpoints, infer_neural, neural_prediction, restore_neural
from .model import TrainConfig
from .raw_sensors import load_raw_sensor
from .schema import TaskSpec
from .selection import development_data, propose_candidates, fit_validation_probe, file_hash


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_extension_dataset(name, seed):
    return load_raw_sensor(name, seed) if name.startswith(("mhealth_", "pamap2_")) else load_dataset(name, seed)


def score(metrics, classification):
    return metrics["macro_f1"] if classification else -metrics["rmse"]


def fit_tree_probe(X, y, splits, classification, kind, seed, iterations=100):
    if set(splits) != {"train", "val"}:
        raise ValueError("Probe selection must not receive test partitions")
    tr, va = splits["train"], splits["val"]
    history, winner, best_score = [], None, -np.inf
    for depth in ((8, 16) if kind == "extra_trees" else (4, 6)):
        if kind == "extra_trees":
            cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
            model = cls(n_estimators=100, max_depth=depth, min_samples_leaf=2, n_jobs=2, random_state=seed)
        else:
            from catboost import CatBoostClassifier
            if not classification:
                raise ValueError("CatBoost in this protocol is classification only")
            model = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=0.1,
                                       random_seed=seed, thread_count=2, verbose=False,
                                       allow_writing_files=False, task_type="CPU")
        model.fit(X[tr], y[tr])
        prediction = np.asarray(model.predict(X[va])).reshape(-1) if classification else model.predict(X[va])
        metrics = _metrics(y[va], prediction, classification)
        item = {"depth": depth, "validation": metrics}
        history.append(item)
        if score(metrics, classification) > best_score:
            winner, best_score, parameters = model, score(metrics, classification), {"depth": depth}
    return winner, {"score": best_score, "parameters": parameters, "history": history,
                    "validation": _metrics(y[va], np.asarray(winner.predict(X[va])).reshape(-1) if classification else winner.predict(X[va]), classification)}


def feature_ranking(X, y, classification, method, seed):
    """Train-only rankings. In forecasting X must exclude all future coordinates."""
    if method == "filter":
        if classification:
            with np.errstate(divide="ignore", invalid="ignore"):
                importance = f_classif(X, y)[0]
        else:
            a, b = X - X.mean(0), y - y.mean(0)
            norm = np.sqrt(np.square(a).sum(0)[:, None] * np.square(b).sum(0)[None, :])
            importance = np.mean(np.square((a.T @ b) / np.maximum(norm, 1e-12)), axis=1)
    elif method == "embedded":
        cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
        selector = cls(n_estimators=100, max_depth=8, min_samples_leaf=2, n_jobs=2, random_state=seed)
        selector.fit(X, y)
        importance = selector.feature_importances_
    else:
        raise ValueError(method)
    importance = np.nan_to_num(importance, nan=-np.inf, posinf=np.finfo(float).max)
    return np.argsort(-importance, kind="stable")


def _fingerprint(dataset, splits):
    return {"X_sha256": _array_hash(dataset.X), "y_sha256": None if dataset.y is None else _array_hash(dataset.y),
            "groups_sha256": None if dataset.groups is None else _array_hash(dataset.groups),
            "split_sha256": {k: _array_hash(v) for k, v in splits.items()}}


def select_extension(dataset, splits, config, output):
    """All model/mask choices are made here, on copies with no test rows."""
    output = Path(output)
    lock_path = output / "lock.json"
    full_fingerprint = _fingerprint(dataset, splits)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["config"] != config or lock["data"] != full_fingerprint:
            raise ValueError("Cannot resume changed data/configuration")
        return lock
    output.mkdir(parents=True, exist_ok=True)
    dev = development_data(dataset, splits)
    classification = dev.labels is not None
    tasks = propose_candidates(dev.dataset)
    candidates = []
    seed = config["model_seed"]
    base_epochs = config["epochs_per_candidate"]
    params = dict(epochs=base_epochs, batch_size=config["batch_size"], learning_rate=0.001,
                  latent_dim=32, ema=0.99, variance_weight=1., seed=seed, device=config["device"])
    start = time.perf_counter()

    def save_estimator(estimator, path):
        joblib.dump(estimator, path)
        return file_hash(path)

    for task_index, task in enumerate(tasks):
        compiled = compile_development_task(dev.dataset, task, dev.splits)
        target = dev.labels if classification else dev.dataset.X[:, task.target]
        context = compiled.X[:, task.context]
        for kind in config["neural_methods"]:
            directory = output / kind / task.name
            directory.mkdir(parents=True, exist_ok=True)
            def callback(values, mean, std, record, kind=kind, directory=directory):
                if kind == "jepa":
                    estimator, probe = fit_validation_probe(values, target, dev.splits, classification, seed)
                    probe_path = directory / f"probe_{record['epoch']}.joblib"
                    record["probe_sha256"] = save_estimator(estimator, probe_path)
                    record["probe"] = probe
                    record["probe_path"] = str(probe_path.relative_to(output))
                    record["validation"] = probe["validation"]
                else:
                    prediction = neural_prediction(values, kind, classification, mean, std)
                    record["validation"] = _metrics(target[dev.splits["val"]], prediction[dev.splits["val"]], classification)
                record.update(method=kind, task=asdict(task), task_index=task_index,
                              checkpoint_path=str((directory / record["checkpoint"]).relative_to(output)))
                candidates.append(record)
            train_checkpoints(compiled, target, classification, kind, TrainConfig(**params), [base_epochs], directory, callback)
        for kind in config["classical_methods"]:
            if kind == "catboost" and not classification:
                continue
            directory = output / kind / task.name
            directory.mkdir(parents=True, exist_ok=True)
            begin = time.perf_counter()
            estimator, record = (fit_validation_probe(context, target, dev.splits, classification, seed) if kind == "linear"
                                  else fit_tree_probe(context, target, dev.splits, classification, kind, seed, config["catboost_iterations"]))
            path = directory / "probe.joblib"
            record.update(method=kind, task=asdict(task), task_index=task_index, probe_path=str(path.relative_to(output)),
                          probe_sha256=save_estimator(estimator, path), fitting_seconds=time.perf_counter()-begin)
            candidates.append(record)

    # A fixed mask spends exactly K*E epochs on one trajectory. It may select
    # among K checkpoints using the same K*5 validation probe fits as search.
    task = tasks[0]
    compiled = compile_development_task(dev.dataset, task, dev.splits)
    target = dev.labels if classification else dev.dataset.X[:, task.target]
    directory = output / "jepa_fixed_budget" / task.name
    directory.mkdir(parents=True, exist_ok=True)
    def fixed_callback(values, mean, std, record):
        estimator, probe = fit_validation_probe(values, target, dev.splits, classification, seed)
        probe_path = directory / f"probe_{record['epoch']}.joblib"
        record.update(method="jepa_fixed_budget", task=asdict(task), task_index=0,
                      validation=probe["validation"], probe=probe,
                      probe_path=str(probe_path.relative_to(output)), probe_sha256=save_estimator(estimator, probe_path),
                      checkpoint_path=str((directory / record["checkpoint"]).relative_to(output)))
        candidates.append(record)
    train_checkpoints(compiled, target, classification, "jepa", TrainConfig(**{**params, "epochs": base_epochs*len(tasks)}),
                      list(range(base_epochs, base_epochs*(len(tasks)+1), base_epochs)), directory, fixed_callback)

    # Independent filter and embedded-feature selectors use training labels.
    # They select exactly the same number of scalar observations as JEPA.
    eligible = np.arange(dataset.X.shape[1]) if classification else np.arange(min(tasks[0].target))
    budget = len(tasks[0].context)
    for selection in ("filter", "embedded"):
        begin = time.perf_counter()
        ranking = feature_ranking(dev.dataset.X[np.ix_(dev.splits["train"], eligible)], target[dev.splits["train"]], classification, selection, seed)
        columns = sorted(eligible[ranking[:budget]].tolist())
        target_columns = sorted(set(range(dataset.X.shape[1]))-set(columns)) if classification else tasks[0].target
        feature_task = TaskSpec(f"{selection}_features", columns, target_columns, "Training-only feature ranking with a fixed observation count")
        compiled_feature = compile_development_task(dev.dataset, feature_task, dev.splits)
        for kind in ("linear", "extra_trees"):
            name = f"{selection}_{kind}"
            directory = output / name
            directory.mkdir(parents=True, exist_ok=True)
            context = compiled_feature.X[:, columns]
            estimator, record = (fit_validation_probe(context, target, dev.splits, classification, seed) if kind == "linear"
                                  else fit_tree_probe(context, target, dev.splits, classification, kind, seed))
            path = directory / "probe.joblib"
            record.update(method=name, task=asdict(feature_task), task_index=0, probe_path=str(path.relative_to(output)),
                          probe_sha256=save_estimator(estimator, path), fitting_seconds=time.perf_counter()-begin,
                          ranking_fit_partition="train", ranking_eligible_columns=eligible.tolist())
            candidates.append(record)

    selected = {}
    for method in dict.fromkeys(r["method"] for r in candidates):
        indices = [i for i, r in enumerate(candidates) if r["method"] == method]
        selected[method] = max(indices, key=lambda i: score(candidates[i]["validation"], classification))
    # Cheap fixed-mask diagnostic is separate from the budget-matched control.
    selected["jepa_fixed_short"] = next(i for i, r in enumerate(candidates) if r["method"] == "jepa" and r["task_index"] == 0)
    updates_search = sum(r["updates"] for r in candidates if r["method"] == "jepa")
    updates_fixed = max(r["updates"] for r in candidates if r["method"] == "jepa_fixed_budget")
    if updates_search != updates_fixed:
        raise AssertionError("Policy training update budgets differ")
    lock = {"schema_version": 1, "dataset": dataset.name, "config": config, "data": full_fingerprint,
            "development_data": _fingerprint(dev.dataset, dev.splits),
            "development_label_sha256": None if dev.labels is None else _array_hash(dev.labels),
            "classification": classification, "candidates": candidates, "selected": selected,
            "budget": {"context_features": budget, "candidate_masks": len(tasks),
                       "search_updates": updates_search, "fixed_updates": updates_fixed,
                       "search_probe_fits": len(tasks)*5, "fixed_probe_fits": len(tasks)*5},
            "elapsed_seconds": time.perf_counter()-start, "locked_at": datetime.now(timezone.utc).isoformat(),
            "test_used_for_selection": False,
            "split_sizes": {k: len(v) for k,v in splits.items()},
            "split_groups": None if dataset.groups is None else {k: np.unique(dataset.groups[v]).tolist() for k,v in splits.items()}}
    write_json(lock_path, lock)
    return lock


def evaluate_extension(dataset, splits, lock, directory, expected_hash, device):
    directory = Path(directory)
    if file_hash(directory / "lock.json") != expected_hash or _fingerprint(dataset, splits) != lock["data"]:
        raise ValueError("Data or lock changed before final test")
    classification = lock["classification"]
    results = []
    for method, idx in lock["selected"].items():
        row = lock["candidates"][idx]
        family = row["method"]
        task = TaskSpec(**row["task"])
        compiled = compile_task(dataset, task, splits)
        context = compiled.X[:, task.context]
        truth = dataset.y if classification else dataset.X[:, task.target]
        if family in {"jepa", "jepa_fixed_budget", "supervised", "tabm"}:
            model, saved = restore_neural(compiled, directory / row["checkpoint_path"], row["checkpoint_sha256"], device)
            values = infer_neural(model, saved["kind"], context)
            del model
            if saved["kind"] == "jepa":
                if file_hash(directory / row["probe_path"]) != row["probe_sha256"]:
                    raise ValueError("Frozen probe changed")
                estimator = joblib.load(directory / row["probe_path"])
                prediction = estimator.predict(values)
            else:
                prediction = neural_prediction(values, saved["kind"], classification, np.asarray(saved["mean"]), np.asarray(saved["std"]))
        else:
            if file_hash(directory / row["probe_path"]) != row["probe_sha256"]:
                raise ValueError("Frozen probe changed")
            estimator = joblib.load(directory / row["probe_path"])
            prediction = np.asarray(estimator.predict(context))
            if classification:
                prediction = prediction.reshape(-1)
        replay = _metrics(truth[splits["val"]], prediction[splits["val"]], classification)
        for k, v in row["validation"].items():
            if not np.isclose(replay[k], v, atol=1e-5, rtol=1e-5):
                raise AssertionError(f"Validation replay mismatch for {dataset.name}/{method}/{k}: {v} vs {replay[k]}")
        test = _metrics(truth[splits["test"]], prediction[splits["test"]], classification)
        np.savez_compressed(directory / f"test_{method}.npz", indices=splits["test"], truth=truth[splits["test"]], prediction=prediction[splits["test"]])
        results.append({"method": method, "selected_candidate": idx, "task": task.name,
                        "context": task.context, "target": task.target,
                        "epoch": row.get("epoch"), "validation": replay, "test": test,
                        "prediction_sha256": file_hash(directory / f"test_{method}.npz")})
    return {"dataset": dataset.name, "split_seed": lock["config"]["split_seed"],
            "classification": classification, "lock_sha256": expected_hash, "methods": results,
            "split_sizes": lock["split_sizes"], "split_groups": lock["split_groups"],
            "validation_replayed": True, "test_evaluated_after_global_lock": True}


def run_extension(config_path, output_path, phase="all"):
    import torch
    from importlib.metadata import version
    torch.set_num_threads(2)
    config = json.loads(Path(config_path).read_text())
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    # Freeze the executed module closure; unrelated paper builders and the
    # separate vision-transfer runner may be authored while experiments run.
    inventory_hashes = source_inventory(Path(config_path).resolve().parents[1])
    module_names = {"compiler", "datasets", "evaluation", "experiments", "extension", "extension_models",
                    "model", "public_datasets", "raw_sensors", "schema", "selection", "synthetic_datasets", "__init__"}
    executed = {k:v for k,v in inventory_hashes.items()
                if k.startswith("src/jepa_forge/") and Path(k).stem in module_names}
    protocol = {"config": config, "source_sha256": executed,
                "versions": {p:version(p) for p in ["numpy", "scipy", "scikit-learn", "torch", "tabm", "catboost"]}}
    protocol_path = output / "protocol_lock.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Source/config changed since this run started; use a fresh directory")
    write_json(protocol_path, protocol)
    records, inventory = [], []
    for name in config["datasets"]:
        dataset = load_extension_dataset(name, config["data_seed"])
        inventory.append({"dataset": name, "rows": len(dataset.X), "features": dataset.X.shape[1],
                          "groups": None if dataset.groups is None else len(np.unique(dataset.groups)),
                          "metadata": dataset.metadata, "X_sha256": _array_hash(dataset.X)})
        for split_seed in config["split_seeds"]:
            splits = make_splits(dataset, split_seed)
            options = {k:v for k,v in config.items() if k not in {"datasets", "split_seeds"}}
            options["split_seed"] = split_seed
            directory = output / name / str(split_seed)
            if phase == "evaluate" and not (directory / "lock.json").exists():
                raise ValueError("Missing development selection lock")
            lock = select_extension(dataset, splits, options, directory)
            records.append({"dataset": name, "split_seed": split_seed,
                            "lock_sha256": file_hash(directory / "lock.json"), "lock": lock})
            print(f"LOCKED {name} split={split_seed} candidates={len(lock['candidates'])}", flush=True)
    global_path = output / "all_selections_locked.json"
    global_lock = {"protocol_sha256": file_hash(protocol_path),
                   "locks": [{k:v for k,v in row.items() if k != "lock"} for row in records],
                   "test_evaluation_started": False}
    if global_path.exists() and json.loads(global_path.read_text()) != global_lock:
        raise ValueError("Global lock changed")
    write_json(global_path, global_lock)
    if phase == "select":
        return
    final = []
    for row in records:
        name, split_seed = row["dataset"], row["split_seed"]
        dataset = load_extension_dataset(name, config["data_seed"])
        directory = output / name / str(split_seed)
        result = evaluate_extension(dataset, make_splits(dataset, split_seed), row["lock"], directory, row["lock_sha256"], config["device"])
        write_json(directory / "final_evaluation.json", result)
        final.append(result)
        print(f"TESTED {name} split={split_seed}", flush=True)
    result = {"schema_version": 1, "status": "complete", "config": config, "protocol": protocol,
              "global_lock_sha256": file_hash(global_path), "global_lock": global_lock,
              "inventory": inventory, "selections": records, "final_evaluations": final,
              "completed_at": datetime.now(timezone.utc).isoformat()}
    with gzip.GzipFile(output / "extension_benchmark.json.gz", "wb", mtime=0) as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode())
    return result
