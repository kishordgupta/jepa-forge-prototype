"""Train/validation-selected probes with an untouched final test partition.

The JEPA backend never receives labels. These supervised probes are a separate
evaluation stage. All preprocessing lives inside training-fitted pipelines.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _metrics(y, prediction, classification):
    if classification:
        return {"accuracy": float(accuracy_score(y, prediction)),
                "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0))}
    return {"rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction))}


def _probe(features, labels, splits, classification, kind="linear", seed=7):
    """Choose hyperparameters on validation, then evaluate once on test.

    Do not refit on train+val: keeping the training data fixed makes all methods
    directly comparable and preserves the exact fitted feature statistics.
    """
    train, val, test = (splits[k] for k in ("train", "val", "test"))
    candidates = []
    if kind == "trees":
        for depth in (8, None):
            cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
            candidates.append(({"max_depth": depth, "n_estimators": 100},
                               cls(n_estimators=100, max_depth=depth, random_state=seed, n_jobs=2)))
    else:
        for strength in (0.01, 0.1, 1.0, 10.0, 100.0):
            estimator = (LogisticRegression(C=strength, max_iter=2000, random_state=seed)
                         if classification else Ridge(alpha=strength))
            transforms = [StandardScaler()]
            if kind == "pca":
                n_components = min(8, features.shape[1], len(train) - 1)
                transforms.append(PCA(n_components=n_components, svd_solver="full"))
            candidates.append(({"C" if classification else "alpha": strength},
                               make_pipeline(*transforms, estimator)))
    best_model, best_params, best_score = None, None, -np.inf
    selection_history = []
    for params, model in candidates:
        model.fit(features[train], labels[train])
        val_metrics = _metrics(labels[val], model.predict(features[val]), classification)
        score = val_metrics["macro_f1"] if classification else -val_metrics["rmse"]
        selection_history.append({"parameters": params, "validation": val_metrics})
        if score > best_score:
            best_model, best_params, best_score = model, params, score
    return {"validation": _metrics(labels[val], best_model.predict(features[val]), classification),
            "test": _metrics(labels[test], best_model.predict(features[test]), classification),
            "selected_parameters": best_params, "selection_history": selection_history,
            "fit_partition": "train", "selection_partition": "val", "evaluation_partition": "test"}


def evaluate_representations(compiled, trained_embeddings, random_embeddings, seed=7):
    classification = compiled.dataset.y is not None
    target = (compiled.dataset.y if classification
              else compiled.dataset.X[:, compiled.task.target])
    context = compiled.X[:, compiled.task.context]
    methods = [
        ("raw_context_linear", context, "linear", "context_only"),
        ("raw_context_pca", context, "pca", "context_only"),
        ("raw_context_extra_trees", context, "trees", "context_only"),
        ("random_encoder_linear", random_embeddings, "linear", "context_only"),
        ("jepa_context_linear", trained_embeddings, "linear", "context_only"),
    ]
    if classification:
        methods.append(("full_raw_linear_reference", compiled.X, "linear", "full_information_reference"))
    rows = []
    for name, features, kind, budget in methods:
        result = _probe(features, target, compiled.splits, classification, kind, seed)
        rows.append({"method": name, "information_budget": budget,
                     "feature_dimensions": int(features.shape[1]), **result})
    if not classification:
        channels = int(compiled.dataset.metadata["input_shape"][1])
        raw_context = compiled.dataset.X[:, compiled.task.context]
        last = raw_context[:, -channels:]
        prediction = np.tile(last, (1, len(compiled.task.target) // channels))
        rows.append({"method": "persistence", "information_budget": "context_only",
                     "feature_dimensions": channels, "selected_parameters": {},
                     "validation": _metrics(target[compiled.splits["val"]], prediction[compiled.splits["val"]], False),
                     "test": _metrics(target[compiled.splits["test"]], prediction[compiled.splits["test"]], False),
                     "fit_partition": "none", "selection_partition": "none", "evaluation_partition": "test"})
    return rows


def predictability_diagnostic(compiled):
    """Validation-only raw linear prediction of targets; a heuristic, not certification."""
    tr, va = compiled.splits["train"], compiled.splits["val"]
    x, y = compiled.X[:, compiled.task.context], compiled.X[:, compiled.task.target]
    model = Ridge(alpha=10).fit(x[tr], y[tr])
    error = float(mean_squared_error(y[va], model.predict(x[va])))
    mean_error = float(mean_squared_error(y[va], np.broadcast_to(y[tr].mean(0), y[va].shape)))
    return {"partition": "val", "target_scale": "training_standardized",
            "ridge_mse": error, "training_mean_mse": mean_error,
            "relative_mse_reduction": 1 - error / max(mean_error, 1e-12),
            "interpretation": "Validation heuristic only; scientific suitability needs domain review."}

