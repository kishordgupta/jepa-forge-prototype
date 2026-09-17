"""Run the frozen extension protocol; all neural execution requires its device."""
import argparse
from jepa_forge.extension import run_extension

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/extension.json")
p.add_argument("--output", default="artifacts/extension/full")
p.add_argument("--phase", choices=["all", "select", "evaluate"], default="all")
args = p.parse_args()
run_extension(args.config, args.output, args.phase)
