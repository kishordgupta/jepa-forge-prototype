from copy import deepcopy
import json

import numpy as np
import pytest

from jepa_forge.datasets import make_splits
from jepa_forge.extension import select_extension, evaluate_extension, feature_ranking
from jepa_forge.raw_sensors import sensor_windows
from jepa_forge.schema import RawDataset
from jepa_forge.selection import file_hash


def test_sensor_window_provenance_excludes_gaps_nulls_transitions_and_missing():
    values = np.ones((64*6, 3))
    labels = np.ones(len(values), dtype=int)
    time = np.arange(len(values)) * .01
    labels[64:128] = 0
    labels[140] = 2
    values[200, 1] = np.nan
    time[260:] += .1
    windows = sensor_windows(values, labels, time, step=1)
    assert windows == [(0, 64, 1), (320, 384, 1)]


def test_independent_filter_only_ranks_supplied_training_features():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(100, 10))
    y = (X[:, 3] > 0).astype(int)
    assert feature_ranking(X, y, True, "filter", 7)[0] == 3
    continuous = X[:, [8]] * 2
    assert feature_ranking(X, continuous, False, "filter", 7)[0] == 8


def small_data():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(90, 8))
    names = [f"x{i}" for i in range(8)]
    return RawDataset("extension_test", "tabular", X, names, y=(X[:, 2] > 0).astype(int),
                      roles={n:"measurement" for n in names}, metadata={"input_shape":[8],"task_type":"classification"})


def test_extension_policy_budget_and_test_poisoning(tmp_path):
    d = small_data()
    splits = make_splits(d, 41)
    cfg = {"model_seed":7,"epochs_per_candidate":1,"batch_size":128,"device":"cpu",
           "neural_methods":["jepa", "supervised"],"classical_methods":["linear"],"catboost_iterations":2,"split_seed":41}
    lock = select_extension(d, splits, cfg, tmp_path / "clean")
    poisoned = deepcopy(d)
    poisoned.X[splits["test"]] = 999.
    poisoned.y[splits["test"]] = 1 - poisoned.y[splits["test"]]
    other = select_extension(poisoned, splits, cfg, tmp_path / "poison")
    assert lock["selected"] == other["selected"]
    for a,b in zip(lock["candidates"], other["candidates"]):
        assert a["validation"] == b["validation"]
        assert a["task"] == b["task"]
    assert lock["budget"]["search_updates"] == lock["budget"]["fixed_updates"]
    assert lock["budget"]["search_probe_fits"] == lock["budget"]["fixed_probe_fits"] == 20
    result = evaluate_extension(d, splits, lock, tmp_path / "clean", file_hash(tmp_path / "clean/lock.json"), "cpu")
    assert result["validation_replayed"]
    path = tmp_path / "clean/lock.json"
    original_hash = file_hash(path)
    path.write_text(json.dumps({**lock,"selected":{}}))
    with pytest.raises(ValueError, match="lock changed"):
        evaluate_extension(d, splits, lock, tmp_path / "clean", original_hash, "cpu")
