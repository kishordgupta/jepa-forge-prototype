#!/usr/bin/env python3
"""Run the documented experiment matrix. Request GPU access when sandboxed."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jepa_forge.experiments import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.json")
    parser.add_argument("--output", default="artifacts/benchmark")
    args = parser.parse_args()
    run_benchmark(args.config, args.output)

