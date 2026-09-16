"""Test separation, meaningful constraints, and locked final evaluation."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_forge.compiler import compile_development_task, export_task
from jepa_forge.datasets import DATASET_NAMES, load_dataset, make_splits
from jepa_forge.public_datasets import PUBLIC_DATASET_NAMES, _cache_dir
from jepa_forge.model import TrainConfig
from jepa_forge.selection import development_data, file_hash, fit_validation_probe, propose_candidates, select_task, validate_candidate_budget
from jepa_forge.selection_experiments import evaluate_selected


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_all_registry_candidates_respect_budget_and_development_audit(name):
    if name in PUBLIC_DATASET_NAMES and not (_cache_dir()/f"{name}.zip").exists():
        pytest.skip("Public archive not cached; CI does not download datasets")
    dataset = load_dataset(name)
    dev = development_data(dataset, make_splits(dataset, 3026))
    candidates = propose_candidates(dev.dataset)
    assert len(candidates) >= 3
    assert len({len(t.context) for t in candidates}) == 1
    assert len({(tuple(t.context), tuple(t.target)) for t in candidates}) == len(candidates)
    for task in candidates:
        compiled = compile_development_task(dev.dataset, task, dev.splits)
        assert compiled.report["status"] == "passed"
        assert set(compiled.splits) == {"train", "val"}
    if dataset.y is None:
        times = np.asarray(dataset.metadata["feature_time_indices"])
        assert len({tuple(t.target) for t in candidates}) == 1
        assert all(times[t.context].max() < times[t.target].min() for t in candidates)


def test_test_poisoning_does_not_change_actual_selection(tmp_path):
    torch.set_num_threads(1)
    original = load_dataset("wine")
    splits = make_splits(original, 3026)
    poisoned = deepcopy(original)
    poisoned.X[splits["test"]] = np.nan
    poisoned.y[splits["test"]] = 999
    poisoned.metadata["test_values"] = poisoned.X[splits["test"]]
    before, after = development_data(original, splits), development_data(poisoned, splits)
    assert before.dataset.y is after.dataset.y is None
    assert "test_values" not in after.dataset.metadata
    assert not np.shares_memory(before.dataset.X, original.X)
    configs = [TrainConfig(epochs=1, batch_size=64, latent_dim=4, device="cpu", seed=7)]
    a = select_task(before, configs, tmp_path/"a")
    b = select_task(after, configs, tmp_path/"b")
    assert a["selected_task"] == b["selected_task"]
    assert a["development_X_sha256"] == b["development_X_sha256"]
    assert [c["mean_validation_score"] for c in a["candidates"]] == [c["mean_validation_score"] for c in b["candidates"]]
    for ca, cb in zip(a["candidates"], b["candidates"]):
        assert ca["runs"][0]["probe"] == cb["runs"][0]["probe"]
    expected = file_hash(tmp_path/"a/selection_lock.json")
    result = evaluate_selected(original, splits, tmp_path/"a", expected, "cpu")
    assert result["task"] == a["selected_task"]
    assert result["test_evaluated_after_lock"]
    with pytest.raises(FileExistsError, match="locked"):
        select_task(before, configs, tmp_path/"a")
    with pytest.raises(ValueError, match="lock changed"):
        evaluate_selected(original, splits, tmp_path/"a", "0"*64, "cpu")
    changed = deepcopy(original)
    changed.X[splits["train"][0], 0] += 1
    with pytest.raises(ValueError, match="Development data differ"):
        evaluate_selected(changed, splits, tmp_path/"a", expected, "cpu")


def test_development_scaling_rejects_group_and_row_leakage(tmp_path):
    data = load_dataset("wine")
    splits = make_splits(data, 3026)
    dev = development_data(data, splits)
    task = propose_candidates(dev.dataset)[0]
    compiled = compile_development_task(dev.dataset, task, dev.splits)
    np.testing.assert_allclose(compiled.mean, data.X[splits["train"]].mean(0))
    with pytest.raises(ValueError, match="finalized"):
        export_task(compiled, tmp_path)
    dev.dataset.groups = np.zeros(len(dev.dataset.X), dtype=int)
    with pytest.raises(ValueError, match="group_overlap"):
        compile_development_task(dev.dataset, task, dev.splits)
    dev.dataset.groups = None
    dev.dataset.X[dev.splits["val"][0]] = dev.dataset.X[dev.splits["train"][0]]
    with pytest.raises(ValueError, match="duplicate_rows_across_splits"):
        compile_development_task(dev.dataset, task, dev.splits)


def test_incomparable_candidates_and_test_probe_inputs_are_rejected():
    data = load_dataset("synthetic_sensors")
    tasks = propose_candidates(data)
    changed = deepcopy(tasks)
    changed[1].context.pop()
    with pytest.raises(ValueError, match="budgets"):
        validate_candidate_budget(changed, True)
    changed = deepcopy(tasks)
    changed[1].target.pop()
    with pytest.raises(ValueError, match="identical targets"):
        validate_candidate_budget(changed, True)
    with pytest.raises(ValueError, match="development partitions only"):
        fit_validation_probe(data.X, data.X, make_splits(data), False)


def test_tie_break_is_declared_order_not_test_score(tmp_path, monkeypatch):
    # Replace only the probe, retaining actual training/checkpoint execution.
    import jepa_forge.selection as module
    torch.set_num_threads(1)
    data = load_dataset("wine")
    dev = development_data(data, make_splits(data))
    original = module.fit_validation_probe
    def tied(*args, **kwargs):
        model, record = original(*args, **kwargs)
        record["score"] = 0.5
        return model, record
    monkeypatch.setattr(module, "fit_validation_probe", tied)
    result = select_task(dev, [TrainConfig(epochs=1, device="cpu")], tmp_path)
    assert result["selected_index"] == 0
