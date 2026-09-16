"""Held-out data must not affect probe fitting or validation selection."""

import numpy as np
import pytest

from jepa_forge.evaluation import _probe


@pytest.mark.parametrize("kind", ["linear", "pca", "trees"])
@pytest.mark.parametrize("classification", [True, False], ids=["classification", "regression"])
def test_test_partition_cannot_change_probe_selection(kind, classification):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(90, 12))
    splits = {
        "train": np.arange(54),
        "val": np.arange(54, 72),
        "test": np.arange(72, 90),
    }
    labels = (
        np.argmax(features[:, :3], axis=1)
        if classification
        else np.column_stack((
            features[:, 0] + 0.2 * features[:, 2],
            features[:, 1] - 0.3 * features[:, 3],
        ))
    )
    changed_features = features.copy()
    changed_labels = labels.copy()
    test = splits["test"]
    changed_features[test] = rng.normal(100, 5, size=(len(test), features.shape[1]))
    changed_labels[test] = (
        (labels[test] + 1) % 3 if classification else labels[test] + 100
    )

    original = _probe(features, labels, splits, classification, kind, seed=7)
    changed = _probe(changed_features, changed_labels, splits, classification, kind, seed=7)

    # Changing both held-out inputs and labels must leave every selection
    # candidate's validation result unchanged, not just the winning parameter.
    assert len(original["selection_history"]) == len(changed["selection_history"])
    for before, after in zip(original["selection_history"], changed["selection_history"]):
        assert before["parameters"] == after["parameters"]
        # Parallel tree prediction may change the last floating-point bit of
        # a reduction while producing the same fitted candidate and outcome.
        assert before["validation"] == pytest.approx(after["validation"], rel=1e-12, abs=1e-12)
    assert original["selected_parameters"] == changed["selected_parameters"]
    assert original["validation"] == pytest.approx(changed["validation"], rel=1e-12, abs=1e-12)
    # A distinct test result confirms that the perturbed partition was used
    # for evaluation rather than silently ignored or replaced by validation.
    assert original["test"] != changed["test"]
    assert original["fit_partition"] == changed["fit_partition"] == "train"
    assert original["selection_partition"] == changed["selection_partition"] == "val"
    assert original["evaluation_partition"] == changed["evaluation_partition"] == "test"
