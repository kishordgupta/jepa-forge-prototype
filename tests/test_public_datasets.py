"""Offline unit tests plus integration checks only for already cached archives."""
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from jepa_forge.compiler import compile_task
from jepa_forge.datasets import make_splits
from jepa_forge import public_datasets as public


def _zip(path, member, text):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, text)
    return path


def test_cached_corruption_fails_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("JEPA_FORGE_DATA_HOME", str(tmp_path))
    (tmp_path / "sonar.zip").write_bytes(b"corrupt archive")
    monkeypatch.setattr(public, "urlopen", lambda *a, **kw: pytest.fail("Network must not be used for a corrupted cache"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        public.load_public_dataset("sonar")


def test_download_must_match_pinned_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("JEPA_FORGE_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(public, "urlopen", lambda *a, **kw: io.BytesIO(b"unexpected upstream content"))
    with pytest.raises(ValueError, match="differs from the reviewed"):
        public._archive_path("sonar")
    assert not (tmp_path / "sonar.zip").exists()


def test_cache_default_uses_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("JEPA_FORGE_DATA_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert public._cache_dir() == tmp_path / ".cache" / "public_datasets"


def test_breast_cancer_parser_removes_identifier_and_label(tmp_path):
    features = ",".join(str(v) for v in range(30))
    path = _zip(tmp_path / "tiny.zip", "wdbc.data", f"patientA,M,{features}\npatientB,B,{features}\n")
    X, y, names, groups, extra = public._parse_dataset("breast_cancer", path)
    assert X.shape == (2, 30)
    assert y.tolist() == ["M", "B"]
    assert groups.tolist() == ["patientA", "patientB"]
    assert len(names) == len(set(names)) == 30
    assert extra["removed_input_fields"] == ["ID", "Diagnosis"]


def test_malformed_one_hot_digit_label_is_rejected(tmp_path):
    text = " ".join(["0"] * 256 + ["1", "1"] + ["0"] * 8) + "\n"
    # Two records preserve the loader's matrix rank; both have invalid labels.
    path = _zip(tmp_path / "tiny.zip", "semeion.data", text * 2)
    with pytest.raises(ValueError, match="Malformed Semeion"):
        public._parse_dataset("semeion", path)


def test_sampling_deduplicates_before_split_and_is_reproducible():
    X = np.arange(8000, dtype=float).reshape(4000, 2)
    X[3000:] = X[:1000]
    first, metadata = public._select_rows(X, None, seed=9, max_rows=1000)
    second, _ = public._select_rows(X, None, seed=9, max_rows=1000)
    changed_seed, _ = public._select_rows(X, None, seed=10, max_rows=1000)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, changed_seed)
    assert len(first) == len(np.unique(X[first], axis=0)) == 1000
    assert first.max() < 3000
    assert metadata["duplicate_rows_removed"] == 1000


def test_group_sampling_retains_each_group_with_bounded_equal_budget():
    X = np.arange(12000, dtype=float).reshape(6000, 2)
    groups = np.repeat(np.arange(30), 200)
    selected, _ = public._select_rows(X, groups, seed=2026)
    values, counts = np.unique(groups[selected], return_counts=True)
    assert len(selected) == 3000
    np.testing.assert_array_equal(values, np.arange(30))
    np.testing.assert_array_equal(counts, np.full(30, 100))


def test_labels_cannot_change_loader_sampling_or_task_masks(tmp_path, monkeypatch):
    path = tmp_path / "source.zip"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(public, "_archive_path", lambda name: path)
    fixture_X = np.arange(6000 * 16, dtype=float).reshape(6000, 16)
    labels = np.arange(6000) % 3
    spec = list(public._SOURCES["letter"])
    spec[3] = 6000
    monkeypatch.setitem(public._SOURCES, "letter", tuple(spec))
    def parser(name, archive):
        return fixture_X, labels.copy(), [f"feature_{i}" for i in range(16)], None, {"modality": "tabular"}
    monkeypatch.setattr(public, "_parse_dataset", parser)
    before = public.load_public_dataset("letter", seed=8)
    labels[:] = np.random.default_rng(91).permutation(labels)
    after = public.load_public_dataset("letter", seed=8)
    np.testing.assert_array_equal(before.X, after.X)
    assert before.metadata["source_row_indices"] == after.metadata["source_row_indices"]
    assert public.propose_public_tasks(before) == public.propose_public_tasks(after)
    assert not np.array_equal(before.y, after.y)


@pytest.mark.parametrize("name", public.PUBLIC_DATASET_NAMES)
def test_cached_official_dataset_compiles_without_network(name, monkeypatch):
    path = public._cache_dir() / f"{name}.zip"
    if not path.exists():
        pytest.skip("Optional integration check: official source archive is not already cached")
    monkeypatch.setattr(public, "urlopen", lambda *a, **kw: pytest.fail("Offline tests must never download"))
    dataset = public.load_public_dataset(name, seed=2026)
    assert 100 <= len(dataset.X) <= 3000
    assert len(np.unique(dataset.X, axis=0)) == len(dataset.X)
    assert dataset.metadata["source_archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dataset.metadata["label_role"] == "evaluation_only"
    splits = make_splits(dataset, seed=2026)
    for task in public.propose_public_tasks(dataset):
        assert compile_task(dataset, task, splits).report["status"] == "passed"
    if dataset.groups is not None:
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            assert set(dataset.groups[splits[first]]).isdisjoint(dataset.groups[splits[second]])
