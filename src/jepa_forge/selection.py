"""Budget-matched task search, using development data only.

This is supervised task selection for classification: the encoder remains
label-free, but training labels fit probes and validation labels rank tasks.
Templates propose feature sets; they do not discover causal relevance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compiler import _array_hash, compile_development_task
from .evaluation import _metrics
from .model import TrainConfig, load_model, train_jepa
from .schema import RawDataset, TaskSpec


@dataclass
class DevelopmentData:
    dataset: RawDataset
    splits: dict[str, np.ndarray]
    labels: np.ndarray | None


def development_data(dataset, splits):
    """Copy only train/validation rows. No test values enter the returned object."""
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Expected train, val, test row partitions")
    arrays = {k: np.asarray(v) for k, v in splits.items()}
    for values in arrays.values():
        if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
            raise ValueError("Partitions must be nonempty integer row arrays")
        if values.min() < 0 or values.max() >= len(dataset.X):
            raise ValueError("Row outside dataset")
    all_rows = np.concatenate(list(arrays.values()))
    if len(np.unique(all_rows)) != len(all_rows):
        raise ValueError("Row partitions overlap or contain duplicates")
    rows = np.concatenate([arrays["train"], arrays["val"]])
    n_train = len(arrays["train"])
    # A whitelist prevents arbitrary metadata from carrying full-dataset values.
    schema_keys = {"input_shape", "feature_time_indices", "task_type", "temporal_direction",
                   "view_a_columns", "view_b_columns", "spatial_spectral_shape", "flatten_order"}
    subset = RawDataset(
        dataset.name, dataset.modality, np.array(dataset.X[rows], copy=True),
        list(dataset.feature_names), y=None,
        groups=None if dataset.groups is None else np.array(dataset.groups[rows], copy=True),
        intervals=None if dataset.intervals is None else np.array(dataset.intervals[rows], copy=True),
        roles=dict(dataset.roles), metadata=deepcopy({k: v for k, v in dataset.metadata.items() if k in schema_keys}),
    )
    labels = None if dataset.y is None else np.array(dataset.y[rows], copy=True)
    return DevelopmentData(subset, {"train": np.arange(n_train), "val": np.arange(n_train, len(rows))}, labels)


def propose_candidates(dataset):
    """Deterministic schema-only templates with equal context sizes.

    Four masks are used unless genuine structural duplication leaves fewer.
    Forecasts share one fixed final-quarter target and observe half the past.
    Classification masks predict the complementary features. Tabular order
    templates are generic hypotheses, requiring domain review before deployment.
    """
    d = len(dataset.feature_names)
    shape = dataset.metadata.get("input_shape", [d])
    forecasting = dataset.metadata.get("task_type") == "forecasting"
    definitions = []
    fixed_target = None
    if forecasting:
        length, channels = shape
        horizon = max(1, length // 4)
        past = length - horizon
        budget = past // 2
        fixed_target = list(range(past * channels, d))
        times = [("recent_past", range(past-budget, past)), ("early_past", range(budget)),
                 ("spaced_past", range(0, 2*budget, 2)),
                 ("middle_past", range((past-budget)//2, (past-budget)//2+budget))]
        definitions = [(name, [t*channels+c for t in ts for c in range(channels)]) for name, ts in times]
    elif dataset.modality == "image":
        height, width = shape
        if height != width or height % 2:
            raise ValueError("Image templates currently require an even square grid")
        grid = np.arange(d).reshape(height, width)
        definitions = [("left_half", grid[:, :width//2].ravel().tolist()),
                       ("right_half", grid[:, width//2:].ravel().tolist()),
                       ("top_half", grid[:height//2].ravel().tolist()),
                       ("bottom_half", grid[height//2:].ravel().tolist())]
    elif dataset.modality in {"sequence", "time_series"}:
        length, channels = shape
        budget = length // 2
        times = [("first_tokens", range(budget)), ("last_tokens", range(length-budget, length)),
                 ("alternating_tokens", range(0, 2*budget, 2)),
                 ("middle_tokens", range((length-budget)//2, (length-budget)//2+budget))]
        definitions = [(name, [t*channels+c for t in ts for c in range(channels)]) for name, ts in times]
    elif dataset.name == "breast_cancer" and d == 30:
        # The public loader fixes these three documented families of measurements.
        definitions = [(name, list(range(start, start+10))) for name, start in
                       (("mean_measurements", 0), ("standard_error_measurements", 10), ("worst_measurements", 20))]
    elif dataset.metadata.get("spatial_spectral_shape") == [3, 3, 4]:
        definitions = [(f"bands_{a}_{b}", [4*p+c for p in range(9) for c in (a, b)])
                       for a, b in ((0, 1), (2, 3), (0, 2), (1, 3))]
    else:
        budget = d // 2
        definitions = [("first_columns", list(range(budget))),
                       ("last_columns", list(range(d-budget, d))),
                       ("alternating_columns", list(range(0, 2*budget, 2))),
                       ("middle_columns", list(range((d-budget)//2, (d-budget)//2+budget)))]
    tasks, seen = [], set()
    for name, context in definitions:
        target = fixed_target if fixed_target is not None else sorted(set(range(d))-set(context))
        key = tuple(context), tuple(target)
        if key not in seen:
            tasks.append(TaskSpec(name, context, list(target),
                "Schema template; equal context budget; " + ("fixed future target." if forecasting else "complementary target features.")))
            seen.add(key)
    validate_candidate_budget(tasks, forecasting)
    return tasks


def validate_candidate_budget(tasks, forecasting):
    if len(tasks) < 2 or len({t.name for t in tasks}) != len(tasks):
        raise ValueError("At least two uniquely named candidates are required")
    if len({len(t.context) for t in tasks}) != 1:
        raise ValueError("Candidate context budgets differ")
    if forecasting and len({tuple(t.target) for t in tasks}) != 1:
        raise ValueError("Forecast candidates must share identical targets")


def fit_validation_probe(features, target, splits, classification, seed=7):
    """No test argument or test partition is accepted by the selection probe."""
    if set(splits) != {"train", "val"}:
        raise ValueError("Selection probes accept development partitions only")
    tr, va = splits["train"], splits["val"]
    if len(features) != len(tr)+len(va) or len(target) != len(features):
        raise ValueError("Development arrays must contain exactly train and validation rows")
    history, best_model, best_score, best_params = [], None, -np.inf, None
    for strength in (0.01, 0.1, 1.0, 10.0, 100.0):
        params = {"C" if classification else "alpha": strength}
        estimator = LogisticRegression(C=strength, max_iter=2000, random_state=seed) if classification else Ridge(alpha=strength)
        model = make_pipeline(StandardScaler(), estimator).fit(features[tr], target[tr])
        metrics = _metrics(target[va], model.predict(features[va]), classification)
        score = metrics["macro_f1"] if classification else -metrics["rmse"]
        history.append({"parameters": params, "validation": metrics})
        if score > best_score:
            best_model, best_score, best_params = model, score, params
    return best_model, {"score": float(best_score), "parameters": best_params,
                        "validation": _metrics(target[va], best_model.predict(features[va]), classification),
                        "history": history, "fit_partition": "train", "selection_partition": "val"}


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_task(development, configs, output_dir, candidates=None):
    """Train all candidates, rank validation scores, then write a durable lock.

    The interface has no full dataset or test-array parameter. All candidates and
    seeds finish before selecting the largest mean validation score. Stable ties
    use declaration order. Failed candidates stop the run instead of disappearing.
    """
    if not isinstance(development, DevelopmentData) or set(development.splits) != {"train", "val"}:
        raise ValueError("A physically separated DevelopmentData object is required")
    if development.dataset.y is not None:
        raise ValueError("Encoder dataset must not contain supervised labels")
    if not configs or len({c.seed for c in configs}) != len(configs):
        raise ValueError("Unique training seeds are required")
    common = [{k: v for k, v in asdict(c).items() if k != "seed"} for c in configs]
    if any(c != common[0] for c in common):
        raise ValueError("All candidates must use the same training budget")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"selection_lock.json").exists():
        raise FileExistsError("Selection is already locked; use a new output directory")
    classification = development.labels is not None
    candidates = list(candidates or propose_candidates(development.dataset))
    validate_candidate_budget(candidates, not classification)
    records = []
    for task in candidates:
        compiled = compile_development_task(development.dataset, task, development.splits)
        target = development.labels if classification else development.dataset.X[:, task.target]
        record = {"task": asdict(task), "audit": compiled.report, "runs": []}
        for config in configs:
            run_dir = output/task.name/str(config.seed)
            model, training = train_jepa(compiled, config, run_dir)
            z = model.encode_context(compiled.X[:, task.context])
            probe, score = fit_validation_probe(z, target, compiled.splits, classification, config.seed)
            restored = load_model(compiled, run_dir/"checkpoint.pt", device=config.device)
            replay = restored.encode_context(compiled.X[:, task.context])
            np.testing.assert_allclose(replay, z, atol=1e-5, rtol=1e-5)
            np.savez_compressed(run_dir/"development_embeddings.npz", embeddings=z,
                               val_predictions=probe.predict(z[compiled.splits["val"]]))
            training["checkpoint"] = str(Path(task.name)/str(config.seed)/"checkpoint.pt")
            training["checkpoint_sha256"] = file_hash(run_dir/"checkpoint.pt")
            training["checkpoint_replay_max_error"] = float(np.max(np.abs(replay-z)))
            entry = {"seed": config.seed, "probe": score, "training": training,
                     "embeddings_sha256": file_hash(run_dir/"development_embeddings.npz")}
            _write(run_dir/"candidate.json", entry)
            record["runs"].append(entry)
            del model, restored, probe
        scores = [r["probe"]["score"] for r in record["runs"]]
        record["mean_validation_score"] = float(np.mean(scores))
        record["std_validation_score"] = float(np.std(scores, ddof=1)) if len(scores)>1 else 0.0
        records.append(record)
        _write(output/"candidate_progress.json", records)
        print(f"{development.dataset.name}: {task.name} validation={record['mean_validation_score']:.6f}", flush=True)
    winner = max(range(len(records)), key=lambda i: records[i]["mean_validation_score"])
    lock = {"schema_version": 1, "dataset": development.dataset.name,
            "selected_task": asdict(candidates[winner]), "selected_index": winner,
            "ranking_rule": "largest mean validation score across predeclared seeds; first declared candidate wins exact ties",
            "metric": "macro_f1" if classification else "negative_rmse",
            "selection_uses_labels": classification, "encoder_uses_labels": False,
            "test_data_available_to_selector": False, "test_evaluation_started": False,
            "training_configs": [asdict(c) for c in configs],
            "development_X_sha256": _array_hash(development.dataset.X),
            "development_labels_sha256": None if development.labels is None else _array_hash(development.labels),
            "development_split_sha256": {k: _array_hash(v) for k,v in development.splits.items()},
            "candidates": records}
    _write(output/"selection_lock.json", lock)
    return lock
