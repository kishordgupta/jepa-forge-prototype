"""Mechanism diversity, deterministic generation, and real compiler integration."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from jepa_forge.compiler import compile_task
from jepa_forge.datasets import make_splits
from jepa_forge.model import TrainConfig, build_model, train_jepa
from jepa_forge.synthetic_datasets import (
    SYNTHETIC_DATASET_NAMES, load_synthetic_dataset, propose_synthetic_tasks,
)


@pytest.fixture(scope="module")
def generated():
    return {name: load_synthetic_dataset(name, seed=2026) for name in SYNTHETIC_DATASET_NAMES}


def test_exactly_ten_distinct_mechanisms(generated):
    assert len(SYNTHETIC_DATASET_NAMES) == len(set(SYNTHETIC_DATASET_NAMES)) == 10
    assert len({d.metadata["generator_family"] for d in generated.values()}) == 10
    assert len({d.metadata["generator"] for d in generated.values()}) == 10
    assert sum(d.modality == "time_series" for d in generated.values()) == 6
    assert sum(d.modality == "tabular" for d in generated.values()) == 4
    # Matching feature shapes must still correspond to distinct generated arrays.
    datasets = list(generated.values())
    for i, left in enumerate(datasets):
        for right in datasets[i + 1:]:
            if left.X.shape == right.X.shape:
                assert not np.array_equal(left.X, right.X)


@pytest.mark.parametrize("name", SYNTHETIC_DATASET_NAMES)
def test_deterministic_finite_nonduplicate_and_all_tasks_compile(name, generated):
    data = generated[name]
    repeat = load_synthetic_dataset(name, seed=2026)
    changed = load_synthetic_dataset(name, seed=2027)
    np.testing.assert_array_equal(data.X, repeat.X)
    assert data.metadata == repeat.metadata
    assert not np.array_equal(data.X, changed.X)
    assert data.X.shape[0] == 2000
    assert 64 <= data.X.shape[1] <= 512
    assert np.isfinite(data.X).all()
    assert np.unique(data.X, axis=0).shape[0] == len(data.X)
    assert len(data.feature_names) == len(set(data.feature_names)) == data.X.shape[1]
    assert all(role == "measurement" for role in data.roles.values())
    splits = make_splits(data, seed=2026)
    assert [len(splits[s]) for s in ("train", "val", "test")] == [1200, 400, 400]
    tasks = propose_synthetic_tasks(data)
    assert len(tasks) == 2
    for task in tasks:
        assert set(task.context).isdisjoint(task.target)
        compiled = compile_task(data, task, splits)
        assert compiled.report["status"] == "passed"
        assert np.isfinite(compiled.X).all()
        if data.modality == "time_series":
            times = np.asarray(data.metadata["feature_time_indices"])
            assert times[task.context].max() < times[task.target].min()


@pytest.mark.parametrize("name", SYNTHETIC_DATASET_NAMES[:6])
def test_entity_splits_preserve_overlapping_windows(name, generated):
    data = generated[name]
    assert data.y is None
    assert data.groups.shape == (2000,)
    assert data.intervals.shape == (2000, 2)
    assert len(np.unique(data.groups)) == 100
    np.testing.assert_array_equal(data.intervals[:, 1] - data.intervals[:, 0], np.full(2000, 31))
    np.testing.assert_array_equal(data.groups[:20], np.zeros(20, dtype=int))
    assert data.intervals[1, 0] <= data.intervals[0, 1]
    splits = make_splits(data, seed=2026)
    groups = {split: set(data.groups[rows]) for split, rows in splits.items()}
    assert [len(groups[s]) for s in ("train", "val", "test")] == [60, 20, 20]
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


@pytest.mark.parametrize("name", SYNTHETIC_DATASET_NAMES[6:])
def test_labels_are_separate_and_do_not_select_tasks_or_splits(name, generated):
    data = generated[name]
    assert data.y.shape == (2000,)
    assert data.groups is None
    assert data.metadata["label_role"] == "evaluation_only"
    assert len(np.unique(data.y)) >= 2
    changed = replace(data, y=np.zeros_like(data.y))
    assert propose_synthetic_tasks(data) == propose_synthetic_tasks(changed)
    for split, indices in make_splits(data).items():
        np.testing.assert_array_equal(indices, make_splits(changed)[split])
    for column in data.X.T:
        assert not np.array_equal(column, data.y)


@pytest.mark.parametrize("name", ["synthetic_lorenz", "synthetic_nonlinear_multiview"])
def test_representative_cpu_model_smoke(name, generated, tmp_path):
    data = generated[name]
    # Representative small subsets keep this a contract test, not a benchmark.
    indices = np.r_[0:20, 20:40, 40:60] if data.groups is not None else np.arange(60)
    small = replace(data, X=data.X[indices], y=None if data.y is None else data.y[indices],
                    groups=None if data.groups is None else data.groups[indices],
                    intervals=None if data.intervals is None else data.intervals[indices])
    splits = {"train": np.arange(0, 20), "val": np.arange(20, 40), "test": np.arange(40, 60)}
    compiled = compile_task(small, propose_synthetic_tasks(small)[0], splits)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, info = train_jepa(compiled, TrainConfig(epochs=1, batch_size=10, device="cpu"), tmp_path / name)
        z = model.encode_context(compiled.X[:, compiled.task.context])
        assert z.shape == (60, 32)
        assert np.isfinite(z).all()
        assert info["labels_used_for_training"] is False
        assert info["test_rows_used_for_training_or_selection"] is False
    finally:
        torch.set_num_threads(old_threads)


def test_unknown_name_rejected():
    with pytest.raises(ValueError, match="Unknown expanded"):
        load_synthetic_dataset("synthetic_unregistered")
