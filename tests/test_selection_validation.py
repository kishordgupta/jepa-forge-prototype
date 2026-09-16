"""Recorded evidence cannot silently change the ranking or test boundary."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
SPEC = importlib.util.spec_from_file_location("validate_selection", ROOT/"scripts/validate_selection.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


@pytest.fixture(scope="module")
def recorded_subset():
    with gzip.open(ROOT/"results/selection_benchmark.json.gz", "rt") as stream:
        data = json.load(stream)
    data["selections"] = data["selections"][:1]
    name = data["selections"][0]["dataset"]
    data["final_evaluations"] = [f for f in data["final_evaluations"] if f["dataset"] == name]
    data["config"]["datasets"] = [name]
    return data


def test_recorded_gpu_selection_evidence_is_consistent(recorded_subset):
    validator.validate_payload(recorded_subset, recorded_subset["config"])


@pytest.mark.parametrize("mutation,message", [
    (lambda d: d.update(status="selecting"), "incomplete"),
    (lambda d: d["selections"][0]["lock"].update(test_data_available_to_selector=True), "Test data"),
    (lambda d: d["selections"][0]["lock"].update(selected_index=99), "validation-optimal"),
    (lambda d: d["selections"][0]["lock"]["candidates"][0]["runs"].pop(), "seed matrix"),
    (lambda d: d["selections"][0]["lock"]["candidates"][0]["runs"][0]["training"]["device_evidence"].update(loss_device="cpu"), "GPU evidence"),
    (lambda d: d["selections"][0]["lock"]["candidates"][0].update(mean_validation_score=123), "average differs"),
    (lambda d: d["final_evaluations"][0].update(test_used_for_ranking=True), "test boundary"),
    (lambda d: d["final_evaluations"].clear(), "coverage differs"),
])
def test_corrupted_evidence_is_rejected(recorded_subset, mutation, message):
    data = deepcopy(recorded_subset)
    mutation(data)
    with pytest.raises(ValueError, match=message):
        validator.validate_payload(data, data["config"])
