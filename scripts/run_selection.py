#!/usr/bin/env python3
"""Run the predeclared automatic task-selection study."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from jepa_forge.selection_experiments import run_selection_benchmark

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/selection.json")
    p.add_argument("--output", default="artifacts/selection")
    args = p.parse_args()
    run_selection_benchmark(args.config, args.output)
