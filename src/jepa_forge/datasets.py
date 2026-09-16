"""Public datasets and documented deterministic synthetic generators."""

import numpy as np
from sklearn.datasets import load_digits, load_wine

from .schema import RawDataset, TaskSpec
from .public_datasets import PUBLIC_DATASET_NAMES, load_public_dataset, propose_public_tasks
from .synthetic_datasets import SYNTHETIC_DATASET_NAMES, load_synthetic_dataset, propose_synthetic_tasks

DATASET_NAMES = ("wine", "digits", "synthetic_sensors", *PUBLIC_DATASET_NAMES, *SYNTHETIC_DATASET_NAMES)


def load_dataset(name: str, seed: int = 2026) -> RawDataset:
    """Load a bundled public dataset or generate independent sensor entities."""
    if name == "wine":
        data = load_wine()
        return RawDataset(
            name="wine", modality="tabular", X=np.asarray(data.data, dtype=np.float64),
            feature_names=list(data.feature_names), y=np.asarray(data.target),
            roles={str(column): "measurement" for column in data.feature_names},
            metadata={
                "source": "UCI Wine via sklearn.datasets.load_wine",
                "source_url": "https://archive.ics.uci.edu/dataset/109/wine",
                "loader_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html",
                "license": "CC BY 4.0 (UCI dataset page)",
                "citation": "Aeberhard, S. & Forina, M. (1992). Wine. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J",
                "task_type": "classification", "label_names": list(data.target_names),
                "label_role": "evaluation_only", "input_shape": [13],
                "split_policy": "random rows, independent of labels",
            },
        )
    if name == "digits":
        data = load_digits()
        feature_names = [f"pixel_r{r}_c{c}" for r in range(8) for c in range(8)]
        return RawDataset(
            name="digits", modality="image", X=np.asarray(data.data, dtype=np.float64),
            feature_names=feature_names, y=np.asarray(data.target),
            roles={column: "pixel" for column in feature_names},
            metadata={
                "source": "UCI Optical Recognition of Handwritten Digits via sklearn.datasets.load_digits",
                "source_url": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
                "loader_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
                "license": "CC BY 4.0 (UCI dataset page)",
                "citation": "Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49",
                "subset_note": "The sklearn loader provides 1797 rows from the UCI collection, not its full 5620 rows.",
                "evaluation_limitation": "Writer identifiers are unavailable in the bundled loader; random row splits do not establish writer-independent generalization.",
                "task_type": "classification", "label_role": "evaluation_only",
                "input_shape": [8, 8], "flatten_order": "row-major",
                "pixel_range": [0, 16], "zero_is_valid": True,
                "split_policy": "random rows, independent of labels",
            },
        )
    if name == "synthetic_sensors":
        return _synthetic_sensors(seed)
    if name in PUBLIC_DATASET_NAMES:
        return load_public_dataset(name, seed=seed)
    if name in SYNTHETIC_DATASET_NAMES:
        return load_synthetic_dataset(name, seed=seed)
    raise ValueError(f"Unknown dataset {name!r}; choose one of {DATASET_NAMES}")


def _synthetic_sensors(seed: int) -> RawDataset:
    rng = np.random.default_rng(seed)
    n_groups, windows_per_group, window, stride, channels = 100, 16, 24, 8, 2
    length = window + stride * (windows_per_group - 1)
    rows, groups, intervals = [], [], []
    for group in range(n_groups):
        frequency = rng.uniform(0.035, 0.085)
        amplitude = rng.uniform(0.7, 1.6)
        phase = rng.uniform(-np.pi, np.pi)
        offset = rng.normal(0.0, 0.25, channels)
        autoregressive = np.zeros(length)
        autoregressive[0] = rng.normal(0.0, 0.08)
        for t in range(1, length):
            autoregressive[t] = 0.82 * autoregressive[t - 1] + rng.normal(0.0, 0.055)
        t = np.arange(length)
        angle = 2 * np.pi * frequency * t + phase
        sequence = np.column_stack((
            amplitude * np.sin(angle) + autoregressive + offset[0],
            0.75 * amplitude * np.sin(angle + 0.45) + 0.6 * autoregressive + offset[1],
        ))
        sequence += rng.normal(0.0, 0.035, sequence.shape)
        for w in range(windows_per_group):
            start = w * stride
            rows.append(sequence[start:start + window].reshape(-1))
            groups.append(group)
            intervals.append([start, start + window - 1])
    feature_names = [f"t{t:02d}_sensor{c}" for t in range(window) for c in range(channels)]
    return RawDataset(
        name="synthetic_sensors", modality="time_series", X=np.asarray(rows, dtype=np.float64),
        feature_names=feature_names, y=None, groups=np.asarray(groups, dtype=np.int64),
        intervals=np.asarray(intervals, dtype=np.int64),
        roles={column: "measurement" for column in feature_names},
        metadata={
            "source": "Locally generated synthetic correlated oscillatory sensors",
            "source_url": None, "license": "CC0-1.0", "generator_seed": int(seed),
            "generator": "sinusoidal two-channel signals with random entity frequency/amplitude/phase/offset, shared AR(1) noise and measurement noise",
            "generator_parameters": {
                "groups": n_groups, "windows_per_group": windows_per_group,
                "window_timesteps": window, "stride": stride, "channels": channels,
                "frequency_range": [0.035, 0.085], "amplitude_range": [0.7, 1.6],
                "ar_coefficient": 0.82, "ar_innovation_std": 0.055,
                "measurement_noise_std": 0.035,
            },
            "input_shape": [window, channels], "flatten_order": "time-major",
            "feature_time_indices": np.repeat(np.arange(window), channels).tolist(),
            "task_type": "forecasting", "temporal_direction": "past_to_future",
            "interval_semantics": "inclusive raw timestep coverage within entity",
            "split_policy": "held-out independent entities; overlapping windows stay in one split",
            "group_role": "split_metadata_only",
        },
    )


def propose_tasks(dataset: RawDataset) -> list[TaskSpec]:
    """Return transparent templates; no labels or held-out values select masks."""
    if dataset.name == "wine":
        d = dataset.X.shape[1]
        return [
            TaskSpec("alternating", list(range(0, d, 2)), list(range(1, d, 2)),
                     "Predict odd-indexed chemical measurements from even-indexed measurements."),
            TaskSpec("first_half", list(range(d // 2)), list(range(d // 2, d)),
                     "Predict the latter measurements from the first half in documented column order."),
        ]
    if dataset.name == "digits":
        center = [r * 8 + c for r in range(2, 6) for c in range(2, 6)]
        right = [r * 8 + c for r in range(8) for c in range(4, 8)]
        return [
            TaskSpec("center", [i for i in range(64) if i not in center], center,
                     "Predict the central 4x4 image region from the outer pixels."),
            TaskSpec("right_half", [r * 8 + c for r in range(8) for c in range(4)], right,
                     "Predict the right image half from the left half."),
        ]
    if dataset.name == "synthetic_sensors":
        return [
            TaskSpec("past16_future8", list(range(32)), list(range(32, 48)),
                     "Use the first 16 timesteps of both channels to predict the final 8."),
            TaskSpec("past12_future12", list(range(24)), list(range(24, 48)),
                     "Use the first 12 timesteps of both channels to predict the final 12."),
        ]
    if dataset.name in PUBLIC_DATASET_NAMES:
        return propose_public_tasks(dataset)
    if dataset.name in SYNTHETIC_DATASET_NAMES:
        return propose_synthetic_tasks(dataset)
    raise ValueError(f"No task templates registered for {dataset.name!r}")


def make_splits(dataset: RawDataset, seed: int = 2026) -> dict[str, np.ndarray]:
    """Make label-independent 60/20/20 splits, respecting supplied entities."""
    rng = np.random.default_rng(seed)
    if dataset.groups is None:
        units = rng.permutation(len(dataset.X))
        a, b = int(0.6 * len(units)), int(0.8 * len(units))
        return {name: np.sort(indices).astype(np.int64) for name, indices in
                zip(("train", "val", "test"), (units[:a], units[a:b], units[b:]))}
    units = rng.permutation(np.unique(dataset.groups))
    a, b = int(0.6 * len(units)), int(0.8 * len(units))
    return {name: np.flatnonzero(np.isin(dataset.groups, selected)).astype(np.int64)
            for name, selected in zip(("train", "val", "test"), (units[:a], units[a:b], units[b:]))}
