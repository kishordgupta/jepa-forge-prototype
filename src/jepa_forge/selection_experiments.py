"""Two-phase selection experiment: lock every dataset, then open test partitions."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .compiler import _array_hash, compile_task, export_task, load_export
from .datasets import load_dataset, make_splits
from .evaluation import evaluate_representations
from .experiments import source_inventory, write_json
from .model import TrainConfig, build_model, load_model
from .schema import TaskSpec
from .selection import development_data, file_hash, select_task


def evaluate_selected(dataset, splits, directory, expected_lock_hash, device):
    directory = Path(directory)
    lock_path = directory/"selection_lock.json"
    if file_hash(lock_path) != expected_lock_hash:
        raise ValueError("Selection lock changed before test evaluation")
    lock = json.loads(lock_path.read_text())
    development = development_data(dataset, splits)
    if dataset.name != lock["dataset"] or _array_hash(development.dataset.X) != lock["development_X_sha256"]:
        raise ValueError("Development data differ from the locked selection")
    label_hash = None if development.labels is None else _array_hash(development.labels)
    if label_hash != lock["development_labels_sha256"]:
        raise ValueError("Development labels changed")
    if {k: _array_hash(v) for k,v in development.splits.items()} != lock["development_split_sha256"]:
        raise ValueError("Development partitions changed")
    winner = max(range(len(lock["candidates"])), key=lambda i: lock["candidates"][i]["mean_validation_score"])
    record = lock["candidates"][winner]
    if winner != lock["selected_index"] or record["task"] != lock["selected_task"]:
        raise ValueError("Selected task is inconsistent with validation ranking")
    task = TaskSpec(**lock["selected_task"])
    compiled = compile_task(dataset, task, splits)
    export_task(compiled, directory/"selected_export")
    restored = load_export(directory/"selected_export")
    np.testing.assert_array_equal(compiled.X, restored.X)
    results = []
    for run in record["runs"]:
        checkpoint = directory/task.name/str(run["seed"])/"checkpoint.pt"
        if file_hash(checkpoint) != run["training"]["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        model = load_model(compiled, checkpoint, device=device)
        embeddings = model.encode_context(compiled.X[:, task.context])
        random = build_model(compiled, TrainConfig(**{**run["training"]["config"], "device": device}))
        random_embeddings = random.encode_context(compiled.X[:, task.context])
        evaluations = evaluate_representations(compiled, embeddings, random_embeddings, run["seed"])
        jepa = next(row for row in evaluations if row["method"] == "jepa_context_linear")
        if jepa["selected_parameters"] != run["probe"]["parameters"]:
            raise ValueError("Final probe parameters differ from locked validation choice")
        for metric, value in run["probe"]["validation"].items():
            if not np.isclose(jepa["validation"][metric], value, atol=1e-5, rtol=1e-5):
                raise ValueError("Final validation metric differs from locked selection")
        np.savez_compressed(directory/task.name/str(run["seed"])/"final_embeddings.npz", context=embeddings, untrained=random_embeddings)
        results.append({"seed": run["seed"], "evaluation": evaluations})
        del model, random
    result = {"dataset": dataset.name, "task": asdict(task), "selection_lock_sha256": expected_lock_hash,
              "test_evaluated_after_lock": True, "test_used_for_ranking": False,
              "selected_export_verified": True, "audit": compiled.report,
              "split_sizes": {k: len(v) for k,v in splits.items()}, "runs": results}
    write_json(directory/"final_evaluation.json", result)
    return result


def run_selection_benchmark(config_path, output_dir):
    config = json.loads(Path(config_path).read_text())
    if len(set(config["datasets"])) != len(config["datasets"]) or not config["datasets"]:
        raise ValueError("Unique dataset names are required")
    output = Path(output_dir)
    if (output/"results.json").exists():
        raise FileExistsError("Use a fresh output directory; completed selections are immutable")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    fields = ("epochs", "batch_size", "learning_rate", "latent_dim", "ema", "variance_weight", "device")
    configs = [TrainConfig(**{k: config[k] for k in fields}, seed=seed) for seed in config["model_seeds"]]
    result = {"schema_version": 1, "status": "selecting", "config": config,
              "source_sha256": source_inventory(Path(config_path).resolve().parents[1]),
              "config_sha256": file_hash(config_path), "selections": [], "final_evaluations": [],
              "scope": "Schema-constrained task search; not causal feature discovery. Classification selection uses development labels.",
              "gpu_scope": "JEPA training and encoding on configured GPU; preprocessing and sklearn probes on CPU."}
    start = time.perf_counter()
    write_json(output/"results.json", result)
    try:
        # Complete and lock ALL validation searches before any final test evaluation.
        for name in config["datasets"]:
            dataset = load_dataset(name, seed=config["data_seed"])
            splits = make_splits(dataset, seed=config["split_seed"])
            directory = output/name
            lock = select_task(development_data(dataset, splits), configs, directory)
            result["selections"].append({"dataset": name, "lock": lock,
                                         "lock_sha256": file_hash(directory/"selection_lock.json")})
            write_json(output/"results.json", result)
        global_lock = {"config_sha256": result["config_sha256"],
                       "dataset_locks": {s["dataset"]: s["lock_sha256"] for s in result["selections"]},
                       "test_evaluation_started": False}
        write_json(output/"all_selections_locked.json", global_lock)
        result["all_selections_lock_sha256"] = file_hash(output/"all_selections_locked.json")
        result["status"] = "evaluating_selected_tasks"
        write_json(output/"results.json", result)
        for selected in result["selections"]:
            name = selected["dataset"]
            dataset = load_dataset(name, seed=config["data_seed"])
            splits = make_splits(dataset, seed=config["split_seed"])
            final = evaluate_selected(dataset, splits, output/name, selected["lock_sha256"], config["device"])
            result["final_evaluations"].append(final)
            write_json(output/"results.json", result)
            print(f"{name}: locked task {final['task']['name']} evaluated on test", flush=True)
        result["status"] = "complete"
        result["elapsed_wall_seconds"] = time.perf_counter()-start
        write_json(output/"results.json", result)
    except Exception as error:
        result["status"] = "failed"
        result["failure_type"] = type(error).__name__
        write_json(output/"results.json", result)
        raise
    return result
