"""Compiler invariants and export integrity, including intentionally bad inputs."""

from dataclasses import replace
import numpy as np
import pytest

from jepa_forge.compiler import audit_task, compile_task, export_task, load_export
from jepa_forge.datasets import load_dataset, make_splits, propose_tasks
from jepa_forge.schema import RawDataset, TaskSpec


@pytest.fixture
def fixture():
    data = RawDataset("fixture", "tabular", np.array([[1., 2.], [3., 4.], [101., 110.], [202., 230.]]), ["a", "b"])
    task = TaskSpec("test", [0], [1])
    splits = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([3])}
    return data, task, splits


def codes(data, task, splits):
    return {item["code"] for item in audit_task(data, task, splits)["errors"]}


def test_train_only_normalization(fixture):
    data, task, splits = fixture
    compiled = compile_task(data, task, splits)
    np.testing.assert_array_equal(compiled.mean, [2., 3.])
    np.testing.assert_array_equal(compiled.std, [1., 1.])
    np.testing.assert_array_equal(compiled.X[:2], [[-1., -1.], [1., 1.]])
    np.testing.assert_array_equal(compiled.X[2], [99., 107.])
    assert compiled.X.dtype == np.float32
    changed = replace(data, X=data.X.copy())
    changed.X[2:] *= 100
    altered = compile_task(changed, task, splits)
    np.testing.assert_array_equal(altered.mean, compiled.mean)
    np.testing.assert_array_equal(altered.std, compiled.std)


@pytest.mark.parametrize("mutation,expected", [
    (lambda d, t, s: (d, replace(t, target=[0]), s), "context_target_overlap"),
    (lambda d, t, s: (d, replace(t, target=[2]), s), "feature_index_out_of_bounds"),
    (lambda d, t, s: (replace(d, roles={"a": "identifier"}), t, s), "forbidden_feature_role"),
    (lambda d, t, s: (replace(d, roles={"b": "label"}), t, s), "forbidden_feature_role"),
    (lambda d, t, s: (d, t, {**s, "test": np.array([1])}), "split_row_overlap"),
    (lambda d, t, s: (d, t, {**s, "test": np.array([], dtype=int)}), "invalid_split"),
    (lambda d, t, s: (d, t, {**s, "test": np.array([4])}), "row_index_out_of_bounds"),
    (lambda d, t, s: (replace(d, X=np.array([[1., np.nan], [2., 3.], [4., 5.], [6., 7.]])), t, s), "nonfinite_features"),
    (lambda d, t, s: (replace(d, X=np.array([[1., 2.], [3., 4.], [1., 2.], [7., 8.]])), t, s), "duplicate_rows_across_splits"),
    (lambda d, t, s: (replace(d, groups=np.array([0, 0, 0, 1])), t, s), "group_overlap"),
    (lambda d, t, s: (replace(d, intervals=np.array([[0, 2], [2, 4], [4, 6], [7, 9]])), t, s), "temporal_interval_overlap"),
    (lambda d, t, s: (replace(d, metadata={"input_shape": ["two", 1], "task_type": "forecasting"}), t, s), "invalid_input_shape"),
])
def test_hard_diagnostics(fixture, mutation, expected):
    data, task, splits = mutation(*fixture)
    assert expected in codes(data, task, splits)
    with pytest.raises(ValueError, match=expected):
        compile_task(data, task, splits)


def test_temporal_direction_and_group_specific_intervals(fixture):
    data, task, splits = fixture
    data = replace(data, groups=np.array([0, 0, 1, 2]),
                   intervals=np.array([[0, 9], [4, 13], [0, 9], [0, 9]]),
                   metadata={"task_type": "forecasting", "input_shape": [2, 1]})
    assert not codes(data, task, splits)
    assert "temporal_direction" in codes(data, replace(task, context=[1], target=[0]), splits)


def test_future_window_overlap_even_with_distinct_feature_rows(fixture):
    data, task, splits = fixture
    data = replace(data, intervals=np.array([[0, 23], [24, 47], [40, 63], [70, 93]]))
    assert "temporal_interval_overlap" in codes(data, task, splits)


def test_zero_pixels_are_valid_and_constants_warn(fixture):
    data, task, splits = fixture
    data = replace(data, X=np.array([[0., 1.], [0., 2.], [0., 3.], [0., 4.]]))
    compiled = compile_task(data, task, splits)
    assert compiled.std[0] == 1
    assert "constant_context" in {item["code"] for item in compiled.report["warnings"]}


@pytest.mark.parametrize("name", ["wine", "digits", "synthetic_sensors"])
def test_bundled_datasets_compile_deterministically(name):
    data = load_dataset(name, seed=17)
    copy = load_dataset(name, seed=17)
    np.testing.assert_array_equal(data.X, copy.X)
    splits = make_splits(data, seed=17)
    for task in propose_tasks(data):
        compiled = compile_task(data, task, splits)
        assert compiled.report["status"] == "passed"
    if data.groups is not None:
        assert data.X.shape == (1600, 48)
        assert data.y is None
        assert [len(splits[s]) for s in ("train", "val", "test")] == [960, 320, 320]
        assert set(data.groups[splits["train"]]).isdisjoint(data.groups[splits["test"]])
        # Windows overlap inside one entity, making row-random splits unsafe.
        assert data.intervals[1, 0] <= data.intervals[0, 1]


def test_splits_do_not_depend_on_labels():
    data = load_dataset("wine")
    initial = make_splits(data, 29)
    changed = make_splits(replace(data, y=np.zeros_like(data.y)), 29)
    for name in initial:
        np.testing.assert_array_equal(initial[name], changed[name])


def test_export_roundtrip_without_pickle(fixture, tmp_path):
    data, task, splits = fixture
    data = replace(data, y=np.array(["a", "b", "a", "b"]), groups=np.array([0, 0, 1, 2]))
    compiled = compile_task(data, task, splits)
    exported = export_task(compiled, tmp_path)
    with np.load(exported["archive"], allow_pickle=False) as archive:
        assert all(not archive[key].dtype.hasobject for key in archive.files)
    restored = load_export(tmp_path)
    np.testing.assert_array_equal(restored.X, compiled.X)
    np.testing.assert_array_equal(restored.dataset.y, data.y)
    np.testing.assert_array_equal(restored.dataset.groups, data.groups)
    assert restored.task == task
    assert restored.manifest == compiled.manifest


@pytest.mark.parametrize("filename", ["dataset.npz", "manifest.json"])
def test_export_tamper_is_detected(fixture, tmp_path, filename):
    export_task(compile_task(*fixture), tmp_path)
    path = tmp_path / filename
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_export(tmp_path)


def test_object_payload_is_never_pickled(fixture, tmp_path):
    data, task, splits = fixture
    data = replace(data, y=np.array([{"payload": i} for i in range(4)], dtype=object))
    with pytest.raises(ValueError, match="pickle-based export is prohibited"):
        export_task(compile_task(data, task, splits), tmp_path)


def test_copy_warning_preserves_signed_zero_and_training_only_comparison():
    # Equal observed train columns can diverge in held-out rows without changing
    # the train-only warning. Constant copies are reported separately.
    X = np.array([[0., 0., -0., 0.], [2., 0., 2., 0.],
                  [7., 0., 11., 0.], [9., 0., 13., 0.]])
    data = RawDataset("copies", "tabular", X, ["a", "zero_a", "b", "zero_b"])
    task = TaskSpec("copy", [0, 1], [2, 3])
    splits = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([3])}
    report = audit_task(data, task, splits)
    copies = next(w for w in report["warnings"] if w["code"] == "trivial_target_copy")
    assert copies["column_pairs"] == [[0, 2]]
