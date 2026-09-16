#!/usr/bin/env python3
"""One-epoch GPU contract check of every expanded dataset-policy pair."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from jepa_forge.datasets import load_dataset, make_splits, propose_tasks
from jepa_forge.compiler import compile_task
from jepa_forge.model import TrainConfig, train_jepa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/expanded.json')
    parser.add_argument('--output', default='artifacts/expanded_smoke')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in config['datasets']:
        data = load_dataset(name, seed=config['split_seed'])
        splits = make_splits(data, seed=config['split_seed'])
        for task in propose_tasks(data):
            compiled = compile_task(data, task, splits)
            # Deliberately tiny execution check; not used to select protocol.
            train_config = TrainConfig(epochs=1, seed=7, device=config['device'])
            model, evidence = train_jepa(compiled, train_config, output/name/task.name)
            row = {'dataset': name, 'task': task.name, 'shape': list(data.X.shape),
                   'split_sizes': {k: len(v) for k,v in splits.items()},
                   'device_evidence': evidence['device_evidence']}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model
            if config['device'] == 'mps':
                torch.mps.empty_cache()
    (output/'checks.json').write_text(json.dumps({'status':'passed', 'checks': rows},indent=2)+'\n')


if __name__ == '__main__':
    main()
