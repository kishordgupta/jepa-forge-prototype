"""Human-reviewed task choice and reproducible compilation."""
import argparse
import json
from pathlib import Path

from .compiler import audit_task, compile_task, export_task, load_export
from .datasets import DATASET_NAMES, load_dataset, make_splits, propose_tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect, validate, compile and benchmark JEPA-style tasks.")
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect", help="Show metadata, candidate policies and diagnostics.")
    inspect.add_argument("dataset", choices=DATASET_NAMES)
    inspect.add_argument("--seed", type=int, default=2026)
    compile_parser = subs.add_parser("compile", help="Compile one explicitly selected policy.")
    compile_parser.add_argument("dataset", choices=DATASET_NAMES)
    compile_parser.add_argument("--task", required=True)
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument("--seed", type=int, default=2026)
    verify = subs.add_parser("verify", help="Check export hashes and structural validity.")
    verify.add_argument("directory")
    benchmark = subs.add_parser("benchmark", help="Run fixed, documented experiments.")
    benchmark.add_argument("--config", default="configs/benchmark.json")
    benchmark.add_argument("--output", default="artifacts/benchmark")
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        from .experiments import run_benchmark
        run_benchmark(args.config, args.output)
        return
    if args.command == "verify":
        compiled = load_export(args.directory)
        print(json.dumps({"verified": True, "dataset": compiled.dataset.name, "task": compiled.task.name}, indent=2))
        return
    dataset = load_dataset(args.dataset, seed=args.seed)
    splits = make_splits(dataset, seed=args.seed)
    tasks = propose_tasks(dataset)
    if args.command == "inspect":
        print(json.dumps({"name": dataset.name, "modality": dataset.modality,
                          "shape": list(dataset.X.shape), "metadata": dataset.metadata,
                          "split_sizes": {k: len(v) for k, v in splits.items()},
                          "candidates": [{"name": t.name, "description": t.description,
                                          "context": t.context, "target": t.target,
                                          "audit": audit_task(dataset, t, splits)} for t in tasks]}, indent=2))
    else:
        selected = [t for t in tasks if t.name == args.task]
        if not selected:
            parser.error("Unknown task. Run inspect to see available task names.")
        result = export_task(compile_task(dataset, selected[0], splits), Path(args.output))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
