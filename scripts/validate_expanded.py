#!/usr/bin/env python3
"""Independently validate expanded experiment artifacts without GPU execution.

The original benchmark is a historical snapshot and is not read or compared to
current source files here. This validator applies current-runtime integrity
checks only to the result file explicitly passed on the command line.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEVICE_FIELDS = (
    "online_parameter_device", "target_parameter_device", "context_batch_device",
    "target_batch_device", "loss_device", "online_gradient_device",
)
BASE_METHODS = {
    "raw_context_linear", "raw_context_pca", "raw_context_extra_trees",
    "random_encoder_linear", "jepa_context_linear",
}
TRAIN_CONFIG_FIELDS = (
    "epochs", "batch_size", "learning_rate", "latent_dim", "ema",
    "variance_weight", "device",
)


class ValidationError(ValueError):
    """A specific integrity condition failed."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(value, path="result"):
    """Reject non-finite JSON numbers, including histories and diagnostics."""
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        require(math.isfinite(value), f"Non-finite number at {path}")


def unique(items, description):
    repeated = [key for key, count in Counter(items).items() if count > 1]
    require(not repeated, f"Duplicate {description}: {repeated}")


def registry_plan(config):
    """Resolve configured names and both policy order and definitions live."""
    from jepa_forge.datasets import load_dataset, make_splits, propose_tasks

    names, seeds = config["datasets"], config["model_seeds"]
    require(names and all(isinstance(name, str) for name in names), "Invalid dataset names")
    require(seeds and all(type(seed) is int for seed in seeds), "Model seeds must be integers")
    unique(names, "configured datasets")
    unique(seeds, "configured model seeds")
    plan = {}
    for name in names:
        dataset = load_dataset(name, seed=config["split_seed"])
        require(dataset.name == name, f"Registry name mismatch: {name}")
        tasks = propose_tasks(dataset)
        require(tasks, f"Registry supplies no tasks for {name}")
        unique([task.name for task in tasks], f"registry tasks for {name}")
        splits = make_splits(dataset, seed=config["split_seed"])
        for index, task in enumerate(tasks):
            plan[(name, task.name)] = {
                "dataset": dataset, "task": task, "splits": splits,
                "primary": index == 0, "classification": dataset.y is not None,
            }
    return plan


def validate_metrics(metrics, classification, path):
    expected = {"accuracy", "macro_f1"} if classification else {"rmse", "mae"}
    require(set(metrics) == expected, f"Metric fields differ at {path}")
    for name, value in metrics.items():
        require(type(value) in (int, float) and math.isfinite(value), f"Invalid metric at {path}.{name}")
        require(0 <= value <= 1 if classification else value >= 0, f"Metric outside range at {path}.{name}")


def validate_evaluation(rows, classification, path):
    methods = [row["method"] for row in rows]
    unique(methods, f"methods at {path}")
    special = "full_raw_linear_reference" if classification else "persistence"
    require(set(methods) == BASE_METHODS | {special}, f"Missing/unexpected methods at {path}")
    for row in rows:
        where = f"{path}/{row['method']}"
        validate_metrics(row["validation"], classification, where + "/validation")
        validate_metrics(row["test"], classification, where + "/test")
        budget = "full_information_reference" if row["method"] == "full_raw_linear_reference" else "context_only"
        require(row["information_budget"] == budget, f"Information budget mismatch: {where}")
        require(row["evaluation_partition"] == "test", f"Wrong evaluation partition: {where}")
        if row["method"] == "persistence":
            require(row["fit_partition"] == row["selection_partition"] == "none", f"Persistence was fitted: {where}")
            continue
        require(row["fit_partition"] == "train" and row["selection_partition"] == "val", f"Probe partition mismatch: {where}")
        history = row["selection_history"]
        require(history, f"No selection history: {where}")
        unique([json.dumps(item["parameters"], sort_keys=True) for item in history], f"probe candidates at {where}")
        for item in history:
            validate_metrics(item["validation"], classification, where + "/selection_history")
        metric = "macro_f1" if classification else "rmse"
        best = (max if classification else min)(history, key=lambda item: item["validation"][metric])
        require(row["selected_parameters"] == best["parameters"], f"Selection is not validation-optimal: {where}")
        for name, value in row["validation"].items():
            require(math.isclose(value, best["validation"][name], rel_tol=1e-10, abs_tol=1e-10), f"Selected validation score mismatch: {where}")


def validate_payload(config, data, plan):
    """Validate completeness and recorded claims before reading binary files."""
    finite_tree(config, "config")
    finite_tree(data)
    require(data["status"] == "complete", "Experiment status is not complete")
    require(data["config"] == config, "Recorded configuration differs from requested configuration")
    require(config["device"] == "mps", "Expanded GPU validation requires configured MPS")
    unique(config["datasets"], "configured datasets")
    unique(config["model_seeds"], "configured model seeds")
    require(set(config["datasets"]) == {key[0] for key in plan}, "Registry dataset set differs from configuration")
    expected = {(name, task, seed) for name, task in plan for seed in config["model_seeds"]}
    actual = [(run["dataset"], run["task"], run["seed"]) for run in data["runs"]]
    unique(actual, "dataset/task/seed runs")
    require(set(actual) == expected, f"Run matrix mismatch: missing={sorted(expected-set(actual))}, unexpected={sorted(set(actual)-expected)}")
    audits = [(item["dataset"], item["task"]) for item in data["task_audits"]]
    unique(audits, "task audits")
    require(set(audits) == set(plan), "Task audit matrix differs from registry")
    for item in data["task_audits"]:
        key = (item["dataset"], item["task"])
        require(item["primary_a_priori"] is plan[key]["primary"], f"Primary policy mismatch: {key}")
        require(item["report"]["status"] == "passed" and not item["report"]["errors"], f"Task audit failed: {key}")
        require(item["export_roundtrip_passed"] is True, f"No export round-trip evidence: {key}")
    for run in data["runs"]:
        key = (run["dataset"], run["task"])
        where = f"{run['dataset']}/{run['task']}/{run['seed']}"
        require(run["primary_a_priori"] is plan[key]["primary"], f"Primary policy mismatch: {where}")
        training = run["training"]
        expected_config = {field: config[field] for field in TRAIN_CONFIG_FIELDS}
        expected_config["seed"] = run["seed"]
        require(training["config"] == expected_config, f"Training configuration mismatch: {where}")
        evidence = training["device_evidence"]
        require(evidence["requested_device"] == "mps", f"Wrong requested training device: {where}")
        for field in DEVICE_FIELDS:
            require(evidence[field] == "mps:0", f"Non-MPS evidence {field}: {where}")
        for field in ("target_requires_grad", "target_has_gradient", "gpu_fallback_enabled"):
            require(evidence[field] is False, f"Invalid teacher/fallback evidence {field}: {where}")
        require(training["labels_used_for_training"] is False, f"Training used labels: {where}")
        require(training["test_rows_used_for_training_or_selection"] is False, f"Training used test rows: {where}")
        require(training["selected_epoch"] == config["epochs"], f"Wrong selected epoch: {where}")
        require([item["epoch"] for item in training["training_history"]] == list(range(1, config["epochs"]+1)), f"Incomplete training history: {where}")
        require(training["checkpoint_roundtrip"]["passed"] is True, f"Checkpoint round-trip failed: {where}")
        diagnostic = training.get("validation_pairing_diagnostic", {})
        require(diagnostic.get("used_for_model_selection") is False and diagnostic.get("shuffle_fixed_points") == 0, f"Invalid validation pairing diagnostic: {where}")
        validate_evaluation(run["evaluation"], plan[key]["classification"], where)


def validate_sources(root, config_path, recorded):
    """Require every current runtime source/config and reject missing entries."""
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    require(config_path.is_relative_to(root), "Configuration must be inside the repository")
    paths = list((root / "src").rglob("*.py")) + list((root / "configs").rglob("*.json"))
    paths += [root / "scripts/run_experiments.py"]
    expected = {str(path.relative_to(root)): path for path in paths if "__pycache__" not in path.parts}
    require(str(config_path.relative_to(root)) in expected, "Configuration is not included in runtime source inventory")
    actual = {name: digest for name, digest in recorded.items()
              if name.startswith("src/") or name.startswith("configs/") or name == "scripts/run_experiments.py"}
    require(set(actual) == set(expected), f"Runtime source inventory mismatch: missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}")
    for name, path in expected.items():
        require(path.is_file() and sha256(path) == actual[name], f"Current runtime source hash mismatch: {name}")
    return actual


def summarize(data):
    """Recompute primary-metric aggregates in the original summary schema."""
    groups = {}
    for run in data["runs"]:
        for row in run["evaluation"]:
            metric = "macro_f1" if "macro_f1" in row["test"] else "rmse"
            key = (run["dataset"], run["task"], row["method"], metric, run["primary_a_priori"])
            groups.setdefault(key, []).append(row["test"][metric])
    return [{"dataset": key[0], "task": key[1], "method": key[2], "metric": key[3],
             "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values)>1 else 0.0,
             "n": len(values), "primary": key[4]}
            for key, values in groups.items()]


def validate_binaries(directory, data, plan):
    """Reload exports and replay checkpoints on CPU training samples only."""
    from jepa_forge.compiler import load_export
    from jepa_forge.model import load_model
    import torch

    directory = Path(directory)
    tasks = {}
    for audit in data["task_audits"]:
        key = (audit["dataset"], audit["task"])
        spec = plan[key]
        compiled = load_export(directory / "compiled" / key[0] / key[1])
        require(compiled.task == spec["task"], f"Export task differs from current registry: {key}")
        require(compiled.dataset.name == key[0], f"Export dataset name differs: {key}")
        require(np.array_equal(compiled.dataset.X, spec["dataset"].X), f"Export raw data differs from current seeded loader: {key}")
        for field in ("y", "groups", "intervals"):
            require(np.array_equal(getattr(compiled.dataset, field), getattr(spec["dataset"], field)), f"Export {field} differs from registry: {key}")
        for split, indices in spec["splits"].items():
            require(np.array_equal(compiled.splits[split], indices), f"Export split differs from current seeded split: {key}/{split}")
        require(compiled.manifest == audit["manifest"], f"Recorded export manifest differs: {key}")
        require(compiled.report == audit["report"], f"Recorded export audit differs: {key}")
        require(audit["split_sizes"] == {name: len(idx) for name, idx in compiled.splits.items()}, f"Recorded split sizes differ: {key}")
        tasks[key] = compiled
    max_error, replay_rows = 0.0, 0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for run in data["runs"]:
            key = (run["dataset"], run["task"])
            where = f"{key[0]}/{key[1]}/{run['seed']}"
            run_dir = directory / "runs" / key[0] / key[1] / str(run["seed"])
            checkpoint = run_dir / "checkpoint.pt"
            require(sha256(checkpoint) == run["training"]["checkpoint_roundtrip"]["sha256"], f"Checkpoint hash differs: {where}")
            require(read_json(run_dir / "evaluation.json") == run, f"Per-run evaluation differs from aggregate: {where}")
            require(read_json(run_dir / "history.json") == run["training"]["training_history"], f"Per-run history differs from aggregate: {where}")
            compiled = tasks[key]
            with np.load(run_dir / "embeddings.npz", allow_pickle=False) as embeddings:
                expected_shape = (len(compiled.X), run["training"]["config"]["latent_dim"])
                for field in ("context", "untrained"):
                    require(embeddings[field].shape == expected_shape and np.isfinite(embeddings[field]).all(), f"Invalid saved {field} embeddings: {where}")
                sample = compiled.splits["train"][:32]
                reference = embeddings["context"][sample].copy()
            model = load_model(compiled, checkpoint, device="cpu")
            replay = model.encode_context(compiled.X[np.ix_(sample, compiled.task.context)])
            require(np.allclose(replay, reference, atol=1e-4, rtol=1e-4), f"CPU checkpoint replay differs: {where}")
            max_error = max(max_error, float(np.max(np.abs(replay-reference))))
            replay_rows += len(sample)
            del model
    finally:
        torch.set_num_threads(previous_threads)
    return {"export_reload_checks": len(tasks), "checkpoint_hash_checks": len(data["runs"]),
            "checkpoint_cpu_replay_checks": len(data["runs"]), "checkpoint_cpu_replay_rows": replay_rows,
            "checkpoint_cpu_replay_max_absolute_error": max_error,
            "checkpoint_cpu_replay_atol": 1e-4, "checkpoint_cpu_replay_rtol": 1e-4}


def validate(config_path, results_path, root=ROOT, expected_scope=(20, 2, 3)):
    config, data = read_json(config_path), read_json(results_path)
    plan = registry_plan(config)
    if expected_scope is not None:
        dataset_count, policies_per_dataset, seed_count = expected_scope
        require(len(config["datasets"]) == dataset_count, f"Expanded scope requires {dataset_count} datasets")
        require(len(config["model_seeds"]) == seed_count, f"Expanded scope requires {seed_count} model seeds")
        policy_counts = Counter(name for name, task in plan)
        require(all(count == policies_per_dataset for count in policy_counts.values()), f"Expanded scope requires {policies_per_dataset} policies for every dataset")
    validate_payload(config, data, plan)
    runtime = validate_sources(root, config_path, data["source_sha256"])
    binary_checks = validate_binaries(Path(results_path).parent, data, plan)
    summary = summarize(data)
    report = {
        "status": "passed", "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(config_path), "results_sha256": sha256(results_path),
        "dataset_count": len(config["datasets"]), "model_seeds": config["model_seeds"],
        "completed_gpu_runs": len(data["runs"]), "audited_tasks": len(plan),
        "expected_run_matrix_size": len(plan)*len(config["model_seeds"]),
        "actual_device": "mps:0", "runtime_source_hashes_match": True,
        "verified_runtime_source_sha256": runtime, "all_numeric_evidence_finite": True,
        "summary_rows": len(summary), "summary_standard_deviation_ddof": 1,
        "low_rank_warning_runs": sum(bool(run["training"]["final_train_context_diagnostics"]["heuristic_collapse_flag"]) for run in data["runs"]),
        "scope": "Current expanded results only; original benchmark remains a separate historical snapshot.",
        **binary_checks,
    }
    return report, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expanded.json")
    parser.add_argument("--results", default="artifacts/expanded/results.json")
    parser.add_argument("--output", default="results/expanded_validation.json")
    parser.add_argument("--summary", default="results/expanded_summary.json")
    parser.add_argument("--expected-datasets", type=int, default=20)
    parser.add_argument("--expected-policies", type=int, default=2)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report, summary = validate(args.config, args.results, expected_scope=(args.expected_datasets, args.expected_policies, args.expected_seeds))
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        report["summary_sha256"] = sha256(summary_path)
    except (ValidationError, KeyError, OSError, ValueError, TypeError) as error:
        report = {"status": "failed", "recorded_utc": datetime.now(timezone.utc).isoformat(),
                  "error_type": type(error).__name__, "error": str(error), "summary_written": False}
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, indent=2))
        return 1
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
