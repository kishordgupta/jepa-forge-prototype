"""Audit and compile explicit context/target tasks with verifiable exports."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import CompiledTask, RawDataset, TaskSpec

_SPLITS = ("train", "val", "test")
_FORBIDDEN_ROLES = {"label", "labels", "target_label", "id", "identifier", "group_id", "subject_id"}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    description = json.dumps({"shape": list(array.shape), "dtype": array.dtype.str}, sort_keys=True)
    return hashlib.sha256(description.encode() + array.tobytes()).hexdigest()


def audit_task(dataset: RawDataset, task: TaskSpec, splits: dict[str, np.ndarray]) -> dict:
    """Return machine-readable errors/warnings without fitting on held-out rows."""
    report = {"errors": [], "warnings": [], "dataset": dataset.name, "task": task.name}

    def add(level, code, message, **details):
        report[level].append({"code": code, "message": message, **_jsonable(details)})

    X = np.asarray(dataset.X)
    if X.ndim != 2 or not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        add("errors", "invalid_feature_matrix", "X must be a real numerical [rows, features] array.")
        report["status"] = "blocked"
        return report
    n, d = X.shape
    report["dimensions"] = {"rows": n, "features": d, "context": len(task.context), "target": len(task.target)}
    report["schema"] = {"modality": dataset.modality, "feature_names": dataset.feature_names,
                        "roles": dataset.roles, "labels_separate": dataset.y is not None,
                        "groups_separate": dataset.groups is not None,
                        "input_shape": dataset.metadata.get("input_shape")}
    if n == 0 or d == 0:
        add("errors", "empty_dataset", "The feature matrix must have rows and columns.")
    if not np.isfinite(X).all():
        add("errors", "nonfinite_features", "X contains NaN or infinite values; provide an explicit train-fitted imputation policy.")
    if len(dataset.feature_names) != d or len(set(dataset.feature_names)) != len(dataset.feature_names):
        add("errors", "invalid_feature_names", "Feature names must be unique and match the number of columns.")
    shape = dataset.metadata.get("input_shape")
    shape_valid = False
    if shape is not None:
        shape_valid = isinstance(shape, (list, tuple)) and len(shape) > 0 and all(isinstance(v, (int, np.integer)) and v > 0 for v in shape) and int(np.prod(shape)) == d
        if not shape_valid:
            add("errors", "invalid_input_shape", "Declared input_shape must contain positive dimensions whose product equals the feature count.")
    selected = {}
    for role, values in (("context", task.context), ("target", task.target)):
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "iu":
            add("errors", "invalid_task_indices", f"{role} must be a nonempty one-dimensional integer index list.", view=role)
            continue
        if (arr < 0).any() or (arr >= d).any():
            add("errors", "feature_index_out_of_bounds", f"{role} contains an out-of-bounds column index.", view=role)
            continue
        if len(np.unique(arr)) != len(arr):
            add("errors", "duplicate_task_indices", f"{role} repeats a column.", view=role)
        selected[role] = arr.astype(np.int64)
    if len(selected) == 2:
        overlap = np.intersect1d(selected["context"], selected["target"])
        if overlap.size:
            add("errors", "context_target_overlap", "Context and target columns must be disjoint.", columns=overlap)
        if len(dataset.feature_names) == d:
            forbidden = [dataset.feature_names[i] for i in np.union1d(*selected.values())
                         if dataset.roles.get(dataset.feature_names[i], "").lower() in _FORBIDDEN_ROLES]
            if forbidden:
                add("errors", "forbidden_feature_role", "Labels and identifiers cannot be model inputs or prediction targets.", columns=forbidden)
        if dataset.metadata.get("task_type") == "forecasting" or dataset.metadata.get("temporal_direction") == "past_to_future":
            times = dataset.metadata.get("feature_time_indices")
            if times is None and shape_valid and len(shape) == 2:
                times = np.repeat(np.arange(shape[0]), shape[1])
            if times is None or np.asarray(times).shape != (d,) or not np.issubdtype(np.asarray(times).dtype, np.number) or not np.isfinite(times).all():
                add("errors", "missing_feature_times", "Forecasting requires a finite timestep for each feature.")
            elif np.max(np.asarray(times)[selected["context"]]) >= np.min(np.asarray(times)[selected["target"]]):
                add("errors", "temporal_direction", "Every context timestep must precede every target timestep.")
    valid_splits = {}
    for name in _SPLITS:
        if name not in splits:
            add("errors", "missing_split", f"Missing {name} split.", split=name)
            continue
        arr = np.asarray(splits[name])
        if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "iu":
            add("errors", "invalid_split", f"{name} must be a nonempty one-dimensional integer array.", split=name)
            continue
        if (arr < 0).any() or (arr >= n).any():
            add("errors", "row_index_out_of_bounds", f"{name} contains an out-of-bounds row index.", split=name)
            continue
        if len(np.unique(arr)) != len(arr):
            add("errors", "duplicate_split_indices", f"{name} repeats a row.", split=name)
        valid_splits[name] = arr.astype(np.int64)
    if set(splits) - set(_SPLITS):
        add("errors", "unexpected_split", "Only train, val and test splits are supported.")
    report["splits"] = {name: len(idx) for name, idx in valid_splits.items()}
    if len(valid_splits) == 3:
        all_indices = np.concatenate(list(valid_splits.values()))
        if len(np.unique(all_indices)) != len(all_indices):
            add("errors", "split_row_overlap", "A row occurs in more than one split.")
        if len(np.unique(all_indices)) < n:
            add("warnings", "unassigned_rows", "Some rows are unused by all three splits.", count=n - len(np.unique(all_indices)))
        seen = {}
        duplicate = None
        # Normalize signed zero so +0 and -0 do not conceal exact duplicate observations.
        canonical = np.array(X, dtype=np.float64, copy=True)
        canonical[canonical == 0] = 0
        for name, indices in valid_splits.items():
            for idx in indices:
                key = canonical[idx].tobytes()
                if key in seen and seen[key][0] != name:
                    duplicate = [seen[key], (name, int(idx))]
                    break
                seen[key] = (name, int(idx))
            if duplicate:
                break
        if duplicate:
            add("errors", "duplicate_rows_across_splits", "Identical raw feature rows occur across splits.", example=duplicate)
    groups = None
    if dataset.groups is not None:
        groups = np.asarray(dataset.groups)
        if groups.shape != (n,) or groups.dtype.kind not in "biufUS" or (groups.dtype.kind in "f" and not np.isfinite(groups).all()):
            add("errors", "invalid_groups", "groups must contain one finite numerical or string identifier per row.")
            groups = None
        else:
            group_sets = {name: set(groups[idx].tolist()) for name, idx in valid_splits.items()}
            report["group_counts"] = {name: len(values) for name, values in group_sets.items()}
            for i, first in enumerate(group_sets):
                for second in list(group_sets)[i + 1:]:
                    if group_sets[first] & group_sets[second]:
                        add("errors", "group_overlap", "An entity occurs in multiple splits.", splits=[first, second])
    if dataset.y is not None and (np.asarray(dataset.y).ndim == 0 or len(dataset.y) != n):
        add("errors", "invalid_labels", "Separate labels must have one first-axis entry per row.")
    if dataset.intervals is not None:
        intervals = np.asarray(dataset.intervals)
        if intervals.shape != (n, 2) or not np.issubdtype(intervals.dtype, np.number) or not np.isfinite(intervals).all() or (intervals[:, 0] > intervals[:, 1]).any():
            add("errors", "invalid_intervals", "intervals must be finite [n, 2] inclusive start/end ranges with start <= end.")
        else:
            by_group = {}
            for split, idxs in valid_splits.items():
                for idx in idxs:
                    group = groups[idx].item() if groups is not None else "__global__"
                    by_group.setdefault(group, []).append((float(intervals[idx, 0]), float(intervals[idx, 1]), split, int(idx)))
            collision = None
            for group, rows in by_group.items():
                latest = {}
                for start, end, split, idx in sorted(rows):
                    for previous_split, (previous_end, previous_idx) in latest.items():
                        if previous_split != split and start <= previous_end:
                            collision = {"group": group, "rows": [previous_idx, idx], "splits": [previous_split, split]}
                            break
                    if collision:
                        break
                    if split not in latest or end > latest[split][0]:
                        latest[split] = (end, idx)
                if collision:
                    break
            if collision:
                add("errors", "temporal_interval_overlap", "Inclusive raw time coverage overlaps across splits within an entity.", **collision)
    if "train" in valid_splits and len(selected) == 2 and np.isfinite(X).all():
        train = X[valid_splits["train"]]
        column_std = np.std(train, axis=0)
        for view, cols in selected.items():
            constants = cols[column_std[cols] < 1e-12]
            if len(constants):
                add("warnings", f"constant_{view}", f"Some {view} columns are constant on training rows; zero-valued pixels remain valid.", columns=constants)
        # Compare canonical column bytes once per column rather than rescanning
        # every context-target pair. Signed zeros must retain array_equal semantics.
        canonical_columns = np.array(train, copy=True)
        canonical_columns[canonical_columns == 0] = 0
        target_columns = {}
        for target in selected["target"]:
            if column_std[target] >= 1e-12:
                target_columns.setdefault(canonical_columns[:, target].tobytes(), []).append(int(target))
        copies = [(int(context), target) for context in selected["context"]
                  for target in target_columns.get(canonical_columns[:, context].tobytes(), [])]
        if copies:
            add("warnings", "trivial_target_copy", "Some nonconstant target columns exactly duplicate context columns on training rows.", column_pairs=copies)
    report["normalization"] = {"fit_split": "train", "method": "column mean and population standard deviation",
                               "constant_columns": "scale replaced by 1", "output_dtype": "float32"}
    report["status"] = "blocked" if report["errors"] else "passed"
    return report


def compile_task(dataset: RawDataset, task: TaskSpec, splits: dict[str, np.ndarray]) -> CompiledTask:
    report = audit_task(dataset, task, splits)
    if report["errors"]:
        codes = ", ".join(item["code"] for item in report["errors"])
        raise ValueError(f"Task audit failed: {codes}")
    split_arrays = {name: np.asarray(splits[name], dtype=np.int64).copy() for name in _SPLITS}
    raw = np.asarray(dataset.X, dtype=np.float64)
    mean = raw[split_arrays["train"]].mean(axis=0)
    std = raw[split_arrays["train"]].std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X = ((raw - mean) / std).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("float32_overflow: standardized features exceed finite float32 range")
    manifest = {
        "format": "jepa-forge", "format_version": 1,
        "dataset": {"name": dataset.name, "modality": dataset.modality,
                    "feature_names": list(dataset.feature_names), "roles": dict(dataset.roles),
                    "metadata": _jsonable(dataset.metadata), "raw_X_sha256": _array_hash(np.asarray(dataset.X))},
        "task": _jsonable(asdict(task)),
        "splits": {name: {"rows": len(idx), "indices_sha256": _array_hash(idx)} for name, idx in split_arrays.items()},
        "normalization": report["normalization"],
        "inference_contract": "Context-only features are X[:, task.context]. Target columns and labels are excluded from context-only inference.",
    }
    return CompiledTask(dataset, task, X, split_arrays, mean, std, report, manifest)


def _safe_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        if all(isinstance(v, str) for v in array.flat):
            array = array.astype(str)
        else:
            raise ValueError(f"{name} contains object data; pickle-based export is prohibited")
    return array


def export_task(compiled: CompiledTask, output_dir) -> dict:
    """Write a pickle-free archive and hashed manifest; return artifact paths/hashes."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"raw_X": _safe_array(compiled.dataset.X, "raw_X"), "X": compiled.X,
              "mean": compiled.mean, "std": compiled.std,
              **{f"split_{name}": idx for name, idx in compiled.splits.items()}}
    for field in ("y", "groups", "intervals"):
        value = getattr(compiled.dataset, field)
        if value is not None:
            arrays[field] = _safe_array(value, field)
    arrays = {key: _safe_array(value, key) for key, value in arrays.items()}
    archive = output / "dataset.npz"
    np.savez_compressed(archive, **arrays)
    manifest = {**compiled.manifest, "report": compiled.report,
                "archive": {"filename": archive.name, "sha256": _sha256(archive),
                            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}}}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    checksums = {"dataset.npz": _sha256(archive), "manifest.json": _sha256(manifest_path)}
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"directory": str(output), "archive": str(archive), "manifest": str(manifest_path), "sha256": checksums}


def load_export(output_dir) -> CompiledTask:
    """Validate checksums and compiler invariants before loading an export."""
    output = Path(output_dir)
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    for name in ("dataset.npz", "manifest.json"):
        if checksums.get(name) != _sha256(output / name):
            raise ValueError(f"Checksum mismatch for {name}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "jepa-forge" or manifest.get("format_version") != 1:
        raise ValueError("Unsupported export format")
    if manifest["archive"]["sha256"] != checksums["dataset.npz"]:
        raise ValueError("Archive checksum differs from manifest")
    with np.load(output / "dataset.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    description = manifest["dataset"]
    raw = RawDataset(description["name"], description["modality"], arrays["raw_X"],
                     description["feature_names"], arrays.get("y"), arrays.get("groups"),
                     arrays.get("intervals"), description["roles"], description["metadata"])
    if _array_hash(raw.X) != description["raw_X_sha256"]:
        raise ValueError("Raw data checksum differs from manifest")
    task = TaskSpec(**manifest["task"])
    splits = {name: arrays[f"split_{name}"] for name in _SPLITS}
    for name, idx in splits.items():
        if _array_hash(idx) != manifest["splits"][name]["indices_sha256"]:
            raise ValueError(f"Split checksum mismatch: {name}")
    compiled = compile_task(raw, task, splits)
    for key in ("X", "mean", "std"):
        if not np.array_equal(getattr(compiled, key), arrays[key]):
            raise ValueError(f"Stored {key} differs from recomputed train-only normalization")
    return compiled
