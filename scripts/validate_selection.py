#!/usr/bin/env python3
"""Verify selection locks, ranking, GPU evidence, exports and CPU replay."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from jepa_forge.compiler import compile_development_task, load_export
from jepa_forge.datasets import load_dataset, make_splits
from jepa_forge.model import load_model
from jepa_forge.selection import development_data, file_hash, fit_validation_probe, propose_candidates, validate_candidate_budget
from jepa_forge.schema import TaskSpec
from validate_expanded import DEVICE_FIELDS, finite_tree, require, unique, validate_evaluation


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def validate_payload(data, config):
    finite_tree(data)
    require(data["status"] == "complete", "Selection experiment is incomplete")
    require(data["config"] == config, "Configuration mismatch")
    names = config["datasets"]
    unique(names, "configured datasets")
    unique(config["model_seeds"], "model seeds")
    unique([s["dataset"] for s in data["selections"]], "selected datasets")
    unique([s["dataset"] for s in data["final_evaluations"]], "final datasets")
    require(set(names) == {s["dataset"] for s in data["selections"]} == {s["dataset"] for s in data["final_evaluations"]}, "Dataset coverage differs")
    require(config["device"] in {"mps", "cuda"}, "GPU configuration required")
    for selection in data["selections"]:
        lock = selection["lock"]
        require(lock["dataset"] == selection["dataset"], "Lock dataset mismatch")
        require(lock["test_data_available_to_selector"] is False and lock["test_evaluation_started"] is False, "Test data used during selection")
        require(lock["encoder_uses_labels"] is False, "Encoder used labels")
        classification = lock["metric"] == "macro_f1"
        require(lock["metric"] in {"macro_f1", "negative_rmse"}, "Unknown selection metric")
        require(lock["selection_uses_labels"] is classification, "Label role mismatch")
        candidates = lock["candidates"]
        validate_candidate_budget([TaskSpec(**c["task"]) for c in candidates], not classification)
        for candidate in candidates:
            require(candidate["audit"]["status"] == "passed", "Candidate audit failed")
            require(set(candidate["audit"]["splits"]) == {"train", "val"}, "Candidate audit received test rows")
            seeds = [r["seed"] for r in candidate["runs"]]
            require(seeds == config["model_seeds"], "Incomplete candidate seed matrix")
            scores = []
            for run in candidate["runs"]:
                training, probe = run["training"], run["probe"]
                for field in ("epochs", "batch_size", "learning_rate", "latent_dim", "ema", "variance_weight", "device"):
                    require(training["config"][field] == config[field], "Training budget changed")
                require(training["config"]["seed"] == run["seed"], "Training seed changed")
                evidence = training["device_evidence"]
                require(all(evidence[field] == config["device"]+":0" for field in DEVICE_FIELDS), "Actual GPU evidence missing")
                require(not evidence["gpu_fallback_enabled"] and not evidence["target_requires_grad"] and not evidence["target_has_gradient"], "Invalid GPU/teacher evidence")
                require(training["labels_used_for_training"] is False and training["test_rows_used_for_training_or_selection"] is False, "Training leakage claim failed")
                require([e["epoch"] for e in training["training_history"]] == list(range(1, config["epochs"]+1)), "Incomplete training history")
                require(training["selected_epoch"] == config["epochs"], "Wrong checkpoint epoch")
                metric = "macro_f1" if classification else "rmse"
                best = (max if classification else min)(probe["history"], key=lambda h: h["validation"][metric])
                require(probe["parameters"] == best["parameters"] and probe["validation"] == best["validation"], "Probe choice is not validation-optimal")
                expected_score = probe["validation"][metric] * (1 if classification else -1)
                require(np.isclose(probe["score"], expected_score, atol=1e-12), "Wrong ranking score")
                scores.append(expected_score)
            require(np.isclose(candidate["mean_validation_score"], np.mean(scores), atol=1e-12), "Candidate average differs")
            std = np.std(scores, ddof=1) if len(scores)>1 else 0.0
            require(np.isclose(candidate["std_validation_score"], std, atol=1e-12), "Candidate deviation differs")
        winner = max(range(len(candidates)), key=lambda i: candidates[i]["mean_validation_score"])
        require(lock["selected_index"] == winner and lock["selected_task"] == candidates[winner]["task"], "Selected task is not validation-optimal")
        final = next(f for f in data["final_evaluations"] if f["dataset"] == selection["dataset"])
        require(final["task"] == lock["selected_task"] and final["selection_lock_sha256"] == selection["lock_sha256"], "Final task differs from lock")
        require(final["test_evaluated_after_lock"] is True and final["test_used_for_ranking"] is False, "Invalid test boundary")
        require([r["seed"] for r in final["runs"]] == config["model_seeds"], "Missing final seeds")
        for run in final["runs"]:
            validate_evaluation(run["evaluation"], classification, selection["dataset"])


def validate(config_path, results_path):
    config, data = read(config_path), read(results_path)
    validate_payload(data, config)
    require(file_hash(config_path) == data["config_sha256"], "Configuration fingerprint differs")
    runtime = {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT/"src").rglob("*.py")}
    runtime.update({str(Path(config_path).resolve().relative_to(ROOT)): file_hash(config_path),
                    "scripts/run_selection.py": file_hash(ROOT/"scripts/run_selection.py")})
    require(all(data["source_sha256"].get(k) == v for k,v in runtime.items()), "Executed source/config fingerprint differs")
    directory = Path(results_path).parent
    require(file_hash(directory/"all_selections_locked.json") == data["all_selections_lock_sha256"], "Global lock changed")
    global_lock = read(directory/"all_selections_locked.json")
    require(global_lock["dataset_locks"] == {s["dataset"]: s["lock_sha256"] for s in data["selections"]}, "Global lock does not cover every dataset")
    require(global_lock["test_evaluation_started"] is False, "Global lock occurred after test evaluation")
    torch.set_num_threads(1)
    replay_count, max_error, exports, candidate_count, low_rank = 0, 0.0, 0, 0, 0
    for selection in data["selections"]:
        name, lock = selection["dataset"], selection["lock"]
        task_dir = directory/name
        require(file_hash(task_dir/"selection_lock.json") == selection["lock_sha256"] and read(task_dir/"selection_lock.json") == lock, "Dataset lock changed")
        dataset = load_dataset(name, seed=config["data_seed"])
        splits = make_splits(dataset, config["split_seed"])
        dev = development_data(dataset, splits)
        require([asdict(t) for t in propose_candidates(dev.dataset)] == [c["task"] for c in lock["candidates"]], "Candidate generator differs from measured candidates")
        for candidate in lock["candidates"]:
            candidate_count += 1
            task = TaskSpec(**candidate["task"])
            compiled = compile_development_task(dev.dataset, task, dev.splits)
            require(compiled.report == candidate["audit"], "Development audit changed")
            target = dev.labels if dev.labels is not None else dev.dataset.X[:, task.target]
            for run in candidate["runs"]:
                run_dir = task_dir/task.name/str(run["seed"])
                require(file_hash(run_dir/"checkpoint.pt") == run["training"]["checkpoint_sha256"], "Checkpoint changed")
                require(file_hash(run_dir/"development_embeddings.npz") == run["embeddings_sha256"], "Development embeddings changed")
                require(read(run_dir/"candidate.json") == run, "Per-run record differs from lock")
                with np.load(run_dir/"development_embeddings.npz", allow_pickle=False) as stored:
                    z, predictions = stored["embeddings"], stored["val_predictions"]
                require(z.shape == (len(dev.dataset.X), config["latent_dim"]) and np.isfinite(z).all(), "Embedding shape/finiteness failed")
                probe, score = fit_validation_probe(z, target, dev.splits, dev.labels is not None, run["seed"])
                require(score == run["probe"], "Validation probe replay differs")
                np.testing.assert_allclose(probe.predict(z[dev.splits["val"]]), predictions)
                model = load_model(compiled, run_dir/"checkpoint.pt", "cpu")
                sample = dev.splits["train"][:32]
                replay = model.encode_context(compiled.X[np.ix_(sample, task.context)])
                np.testing.assert_allclose(replay, z[sample], atol=1e-4, rtol=1e-4)
                max_error = max(max_error, float(np.max(np.abs(replay-z[sample]))))
                replay_count += 1
                low_rank += bool(run["training"]["final_train_context_diagnostics"]["heuristic_collapse_flag"])
        exported = load_export(task_dir/"selected_export")
        np.testing.assert_array_equal(exported.dataset.X, dataset.X)
        require(asdict(exported.task) == lock["selected_task"], "Selected export changed")
        for split, indices in splits.items():
            np.testing.assert_array_equal(exported.splits[split], indices)
        final = next(f for f in data["final_evaluations"] if f["dataset"] == name)
        require(read(task_dir/"final_evaluation.json") == final, "Final evaluation file differs")
        exports += 1
    return {"status": "passed", "datasets": len(config["datasets"]), "candidates": candidate_count,
            "gpu_training_runs": replay_count, "checkpoint_cpu_replays": replay_count,
            "validation_probe_replays": replay_count, "checkpoint_max_absolute_error": max_error,
            "selected_export_reloads": exports, "final_selected_seed_evaluations": len(config["datasets"])*len(config["model_seeds"]),
            "all_test_evaluations_after_global_lock": True, "low_rank_warning_runs": low_rank,
            "results_sha256": file_hash(results_path), "config_sha256": file_hash(config_path),
            "verified_executed_source_sha256": runtime}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection.json")
    parser.add_argument("--results", default="artifacts/selection/results.json")
    parser.add_argument("--output", default="results/selection_validation.json")
    args = parser.parse_args()
    report = validate(args.config, args.results)
    Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
