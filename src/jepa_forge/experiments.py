"""Reproducible public/synthetic GPU experiment orchestration."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch

from .compiler import compile_task, export_task, load_export
from .datasets import load_dataset, make_splits, propose_tasks
from .evaluation import evaluate_representations, predictability_diagnostic
from .model import TrainConfig, build_model, train_jepa, load_model


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def source_inventory(root):
    files = []
    for folder in ("scripts", "configs"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".json"):
                files.append(path)
    files += [root / "pyproject.toml"]
    inventory = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}
    # Hash the actual imported package, including when the CLI is installed.
    # An editable source checkout and a wheel use the same portable keys.
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" not in path.parts:
            inventory[str(Path("src/jepa_forge") / path.relative_to(package))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return inventory


def _run_benchmark(config_path, output_dir):
    config = json.loads(Path(config_path).read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if config["device"] == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for a verified GPU experiment.")
    started = time.perf_counter()
    result = {"schema_version": "1.0", "status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "packages": {"python": platform.python_version(), "torch": torch.__version__,
                                             "numpy": np.__version__, "scikit_learn": sklearn.__version__},
              "source_sha256": source_inventory(Path(config_path).resolve().parents[1]), "runs": [], "task_audits": [],
              "experiment_scope": "Fresh prototype measurements; not a reproduction of proposal result table.",
              "comparison_rule": "Compare methods within one dataset-policy only; policies may differ in context and horizon.",
              "gpu_scope": "JEPA training and encoding on requested device; sklearn probes, baselines and preprocessing on CPU."}
    write_json(output / "results.json", result)
    for dataset_name in config["datasets"]:
        dataset = load_dataset(dataset_name, seed=config["split_seed"])
        splits = make_splits(dataset, seed=config["split_seed"])
        for policy_index, task in enumerate(propose_tasks(dataset)):
            compiled = compile_task(dataset, task, splits)
            task_dir = output / "compiled" / dataset_name / task.name
            export_task(compiled, task_dir)
            reloaded = load_export(task_dir)
            if not np.array_equal(reloaded.X, compiled.X):
                raise AssertionError("Export round-trip changed prepared features.")
            task_evidence = {"dataset": dataset_name, "task": task.name,
                             "primary_a_priori": policy_index == 0, "report": compiled.report,
                             "manifest": compiled.manifest, "predictability": predictability_diagnostic(compiled),
                             "export_roundtrip_passed": True,
                             "split_sizes": {k: len(v) for k, v in splits.items()}}
            result["task_audits"].append(task_evidence)
            for seed in config["model_seeds"]:
                train_config = TrainConfig(**{k: config[k] for k in
                    ("epochs", "batch_size", "learning_rate", "latent_dim", "ema", "variance_weight", "device")}, seed=seed)
                run_dir = output / "runs" / dataset_name / task.name / str(seed)
                untrained = build_model(compiled, train_config)
                context_values = compiled.X[:, task.context]
                random_embeddings = untrained.encode_context(context_values)
                del untrained
                model, evidence = train_jepa(compiled, train_config, run_dir)
                embeddings = model.encode_context(context_values)
                restored_model = load_model(compiled, run_dir / "checkpoint.pt", device=config["device"])
                restored_embeddings = restored_model.encode_context(context_values)
                np.testing.assert_allclose(restored_embeddings, embeddings, atol=1e-5, rtol=1e-5)
                evidence["checkpoint_roundtrip"] = {
                    "passed": True, "max_absolute_error": float(np.abs(restored_embeddings - embeddings).max()),
                    "sha256": hashlib.sha256((run_dir / "checkpoint.pt").read_bytes()).hexdigest()}
                del restored_model
                rows = evaluate_representations(compiled, embeddings, random_embeddings, seed=seed)
                np.savez_compressed(run_dir / "embeddings.npz", context=embeddings, untrained=random_embeddings)
                run = {"dataset": dataset_name, "task": task.name, "primary_a_priori": policy_index == 0,
                       "seed": seed, "training": evidence, "evaluation": rows}
                result["runs"].append(run)
                write_json(run_dir / "evaluation.json", run)
                result["elapsed_seconds"] = time.perf_counter() - started
                write_json(output / "results.json", result)
                main = next(r for r in rows if r["method"] == "jepa_context_linear")
                print(json.dumps({"dataset": dataset_name, "task": task.name, "seed": seed,
                                  "device": config["device"], "test": main["test"]}), flush=True)
                del model
                if config["device"] == "mps":
                    torch.mps.empty_cache()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = "complete"
    write_json(output / "results.json", result)
    return result


def run_benchmark(config_path, output_dir):
    try:
        return _run_benchmark(config_path, output_dir)
    except Exception as error:
        path = Path(output_dir) / "results.json"
        partial = json.loads(path.read_text()) if path.exists() else {"runs": []}
        partial.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        write_json(path, partial)
        raise
