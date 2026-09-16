"""Pinned, real UCI datasets for the expanded prototype experiment.

Downloads are cached, checked against a reviewed SHA-256, and parsed without
extracting archive paths. Sampling and exact-feature deduplication do not use
class labels. The original benchmark partitions are pooled and re-split for
this pilot; no claim of reproducing the official benchmark is made.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from scipy.io import arff

from .schema import RawDataset, TaskSpec

PUBLIC_DATASET_NAMES = (
    "breast_cancer", "ionosphere", "sonar", "semeion", "letter", "pendigits",
    "satimage", "har", "dry_bean", "isolet",
)

# Archive hashes were measured from official UCI downloads on 2026-09-16.
# A changed upstream archive must be reviewed, not silently trusted.
_SOURCES = {
    "breast_cancer": (17, "breast+cancer+wisconsin+diagnostic", "bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af", 569, 30,
                      "Wolberg, W., Mangasarian, O., Street, N. & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic).", "C5DW2B"),
    "ionosphere": (52, "ionosphere", "4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860", 351, 34,
                   "Sigillito, V., Wing, S., Hutton, L. & Baker, K. (1989). Ionosphere.", "C5W01B"),
    "sonar": (151, "connectionist+bench+sonar+mines+vs+rocks", "088b0b6813fb5ab84736bc2a1e3d6bb886e317520d2f2c04152e2ed65af7a18d", 208, 60,
              "Sejnowski, T. & Gorman, R. (1988). Connectionist Bench (Sonar, Mines vs. Rocks).", "C5T01Q"),
    "semeion": (178, "semeion+handwritten+digit", "6fb091394714cddda5751d4e1c2781ab094e7cf15de07917fb40e581f19efc75", 1593, 256,
                "Semeion Handwritten Digit (1998). UCI Machine Learning Repository.", "C5SC8V"),
    "letter": (59, "letter+recognition", "3b5f07a334697b6cace4fbae22940393a18fee596e73f68d97ce5973d52dc60f", 20000, 16,
               "Slate, D. (1991). Letter Recognition.", "C5ZP40"),
    "pendigits": (81, "pen+based+recognition+of+handwritten+digits", "1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64", 10992, 16,
                  "Alpaydin, E. & Alimoglu, F. (1996). Pen-Based Recognition of Handwritten Digits.", "C5MG6K"),
    "satimage": (146, "statlog+landsat+satellite", "7c54e0e11c872a1b0b647da370d596dcb06746159cce4121d92ccd70b7d7ce3c", 6435, 36,
                 "Srinivasan, A. (1993). Statlog (Landsat Satellite).", "C55887"),
    "har": (240, "human+activity+recognition+using+smartphones", "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031", 10299, 561,
            "Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L. & Parra, X. (2013). Human Activity Recognition Using Smartphones.", "C54S4K"),
    "dry_bean": (602, "dry+bean+dataset", "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f", 13611, 16,
                 "Dry Bean (2020). UCI Machine Learning Repository. Koklu, M. & Ozkan, I. A. (2020), Computers and Electronics in Agriculture 174, 105507.", "C50S4B"),
    "isolet": (54, "isolet", "fe2e0d45f1057d112e051d309070992fc178e7f6922cc7b4e8b2bd310303a053", 7797, 617,
               "Cole, R. & Fanty, M. (1991). ISOLET.", "C51G69"),
}

_COMPLEXITY = {
    "breast_cancer": "Thirty correlated morphology measurements across mean, standard-error and worst-value summaries; small real biomedical classification set.",
    "ionosphere": "Seventeen complex-valued radar pulse pairs, mixed constant and informative channels, and nonlinear signal structure.",
    "sonar": "Sixty ordered spectral energy bands with only 208 observations; a high-feature-to-sample real sensing problem.",
    "semeion": "256 binary image pixels and ten classes at 16x16 resolution, larger spatial masks than the original 8x8 Digits pilot.",
    "letter": "Twenty-six classes with geometric moments and edge summaries extracted from distorted font images.",
    "pendigits": "Eight ordered two-coordinate trajectory points; ten classes and preserved source grouping across 44 writer-like groups.",
    "satimage": "Nine spatial neighbors times four spectral bands; six land-cover classes and interdependent spatial/spectral features.",
    "har": "561 engineered inertial time/frequency features with six activities and explicit subject-held-out evaluation.",
    "dry_bean": "Seven varieties with correlated geometric and shape features, imbalance and redundant derived attributes.",
    "isolet": "617 acoustic summary features with 26 spoken-letter classes; substantially higher dimensional than the initial benchmark.",
}


def _cache_dir() -> Path:
    configured = os.environ.get("JEPA_FORGE_DATA_HOME")
    return Path(configured) if configured else Path.cwd() / ".cache" / "public_datasets"


def _archive_path(name: str) -> Path:
    identifier, slug, expected, *_ = _SOURCES[name]
    root = _cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.zip"
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Cached {name} archive SHA-256 mismatch; remove/review the corrupted archive explicitly")
        return path
    url = f"https://archive.ics.uci.edu/static/public/{identifier}/{slug}.zip"
    request = Request(url, headers={"User-Agent": "JEPA-FORGE research prototype (public dataset download)"})
    with urlopen(request, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Downloaded {name} archive SHA-256 differs from the reviewed official source")
    # Avoid exposing partial downloads to another loader process.
    with tempfile.NamedTemporaryFile(dir=root, suffix=".download", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _uncompress(payload: bytes) -> bytes:
    """Read legacy Unix-compress members through gzip's non-executing decoder."""
    executable = shutil.which("gzip")
    if not executable:
        raise RuntimeError("The official ISOLET/PenDigits .Z members require the standard gzip command; install gzip")
    return subprocess.run([executable, "-cd"], input=payload, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True, timeout=60).stdout


def _numeric(z: zipfile.ZipFile, member: str, delimiter=None) -> np.ndarray:
    payload = z.read(member)
    if member.endswith(".Z"):
        payload = _uncompress(payload)
    return np.loadtxt(io.BytesIO(payload), delimiter=delimiter, dtype=np.float64)


def _mixed(z: zipfile.ZipFile, member: str) -> np.ndarray:
    return np.loadtxt(io.BytesIO(z.read(member)), delimiter=",", dtype=str)


def _parse_dataset(name: str, path: Path):
    """Return X, labels, feature names, optional source groups, parser metadata."""
    groups = None
    extra = {"modality": "tabular", "limitations": []}
    with zipfile.ZipFile(path) as z:
        if name == "breast_cancer":
            rows = _mixed(z, "wdbc.data")
            X, labels, groups = rows[:, 2:].astype(float), rows[:, 1], rows[:, 0]
            measures = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
                        "concavity", "concave_points", "symmetry", "fractal_dimension"]
            names = [f"{kind}_{measure}" for kind in ("mean", "standard_error", "worst") for measure in measures]
            extra.update(group_role="source_record_id_only", removed_input_fields=["ID", "Diagnosis"])
            extra["limitations"] = ["Public historical feature dataset; not clinical validation or a diagnostic tool.",
                                    "Unique source record IDs are retained for grouping; no repeated-patient history is provided."]
        elif name in {"ionosphere", "sonar"}:
            rows = _mixed(z, "ionosphere.data" if name == "ionosphere" else "sonar.all-data")
            X, labels = rows[:, :-1].astype(float), rows[:, -1]
            if name == "ionosphere":
                names = [f"pulse{p:02d}_{component}" for p in range(17) for component in ("real", "imag")]
                extra.update(modality="sequence", input_shape=[17, 2], axis_semantics="pulse index; real/imaginary autocorrelation components")
            else:
                names = [f"spectral_band_{i:02d}" for i in range(60)]
                extra.update(modality="sequence", input_shape=[60, 1], axis_semantics="ordered spectral energy bands, not equal time steps")
            extra["limitations"] = ["Source acquisition/session identifiers are absent; random record splitting cannot establish independent-session generalization.",
                                    "The reconstruction mask uses the documented feature order; downstream task is classification, not forecasting."]
        elif name == "semeion":
            rows = _numeric(z, "semeion.data")
            X, one_hot = rows[:, :256], rows[:, 256:]
            if one_hot.shape[1] != 10 or not np.all(one_hot.sum(axis=1) == 1) or not np.isin(one_hot, [0, 1]).all():
                raise ValueError("Malformed Semeion one-hot labels")
            labels = one_hot.argmax(axis=1)
            names = [f"pixel_r{r}_c{c}" for r in range(16) for c in range(16)]
            extra.update(modality="image", input_shape=[16, 16], flatten_order="row-major", zero_is_valid=True)
            extra["limitations"] = ["Writer IDs are not present in the numeric records; random record splits do not demonstrate writer-independent performance."]
        elif name == "letter":
            rows = _mixed(z, "letter-recognition.data")
            X, labels = rows[:, 1:].astype(float), rows[:, 0]
            names = ["x_box", "y_box", "width", "height", "on_pixels", "x_mean", "y_mean", "x_variance", "y_variance",
                     "xy_correlation", "xxy", "xyy", "x_edges", "x_edge_y", "y_edges", "y_edge_x"]
            extra["limitations"] = ["Source font identifiers are absent; this is a new bounded row split, not the original 16000/4000 benchmark or font-independent test."]
        elif name == "pendigits":
            matrices, identities = [], []
            for split, count in (("tra", 30), ("tes", 14)):
                matrix = _numeric(z, f"pendigits.{split}", ",")
                original = _uncompress(z.read(f"pendigits-orig.{split}.Z")).decode("ascii")
                comments = np.asarray([[int(v) for v in line.split()[1:]] for line in original.splitlines() if line.startswith(".COMMENT")])
                if comments.shape != (len(matrix), 3) or not np.array_equal(comments[:, 0], matrix[:, -1]) or len(np.unique(comments[:, 1])) != count:
                    raise ValueError("PenDigits original comment groups are not aligned with normalized records")
                matrices.append(matrix)
                identities.extend(f"{split}:{group}" for group in comments[:, 1])
            rows = np.concatenate(matrices)
            X, labels, groups = rows[:, :-1], rows[:, -1].astype(int), np.asarray(identities)
            names = [f"point{p}_{axis}" for p in range(8) for axis in ("x", "y")]
            extra.update(modality="sequence", input_shape=[8, 2], flatten_order="point-major",
                         group_role="namespaced second field of aligned original .COMMENT records",
                         source_partition_counts={"tra": 7494, "tes": 3498},
                         group_interpretation="Source groups match the documented 30/14 writer populations and approximately 250 records/group; writer identity interpretation is inferred, not independently validated.")
            extra["limitations"] = ["The source comment grouping is preserved conservatively; source numeric records do not themselves name writers.",
                                    "Original partitions are pooled and new source-group splits are made; this is not the published 7494/3498 benchmark."]
        elif name == "satimage":
            rows = np.concatenate([_numeric(z, "sat.trn"), _numeric(z, "sat.tst")])
            X, labels = rows[:, :-1], rows[:, -1].astype(int)
            names = [f"pixel_r{r}_c{c}_band{b}" for r in range(3) for c in range(3) for b in range(4)]
            extra.update(spatial_spectral_shape=[3, 3, 4], source_partition_counts={"train": 4435, "test": 2000})
            extra["limitations"] = ["Geographic IDs are unavailable and neighboring records may overlap; row splits do not establish spatially independent generalization.",
                                    "Physical pixel/band masks are retained, but the existing pilot uses a tabular encoder, not a multispectral CNN.",
                                    "Original training/test files are pooled for a new bounded split."]
        elif name == "har":
            with zipfile.ZipFile(io.BytesIO(z.read("UCI HAR Dataset.zip"))) as h:
                prefix = "UCI HAR Dataset/"
                X = np.concatenate([_numeric(h, f"{prefix}{split}/X_{split}.txt") for split in ("train", "test")])
                labels = np.concatenate([_numeric(h, f"{prefix}{split}/y_{split}.txt") for split in ("train", "test")]).astype(int)
                groups = np.concatenate([_numeric(h, f"{prefix}{split}/subject_{split}.txt") for split in ("train", "test")]).astype(int)
                source_names = [line.split(maxsplit=1)[1] for line in h.read(prefix + "features.txt").decode().splitlines()]
            names = [f"f{i:03d}_{label}" for i, label in enumerate(source_names)]
            extra.update(group_role="explicit original subject identifier", source_feature_names=source_names,
                         source_partition_counts={"train": 7352, "test": 2947}, window_description="128 readings at 50 Hz, 50 percent overlap; 561 source-engineered features")
            extra["limitations"] = ["Source-provided engineered features are already normalized; this pilot cannot reconstruct or audit every original preprocessing step.",
                                    "Explicit subjects are held out; exact recording intervals are not available with the feature vectors.",
                                    "The original subject split is replaced by a declared new group split; no original benchmark score is claimed."]
        elif name == "dry_bean":
            table, _ = arff.loadarff(io.StringIO(z.read("DryBeanDataset/Dry_Bean_Dataset.arff").decode("utf-8")))
            names = list(table.dtype.names[:-1])
            X = np.column_stack([table[column] for column in names]).astype(float)
            labels = table[table.dtype.names[-1]].astype(str)
            extra["limitations"] = ["Several shape features are algebraically related; predictable context-target views need not imply semantic representation quality.",
                                    "No acquisition/batch identifier is available; row splits do not establish new-batch generalization."]
        elif name == "isolet":
            rows = np.concatenate([_numeric(z, member, ",") for member in ("isolet1+2+3+4.data.Z", "isolet5.data.Z")])
            X, labels = rows[:, :-1], rows[:, -1].astype(int)
            names = [f"acoustic_feature_{i:03d}" for i in range(617)]
            extra.update(source_partition_counts={"isolet1_2_3_4": 6238, "isolet5": 1559})
            extra["limitations"] = ["Explicit speaker IDs are not in numeric records; the source reports missing recordings, so identities are not guessed from fixed row blocks.",
                                    "The published speaker-separated partitions are pooled for the bounded row-split pilot; results do not establish speaker-independent performance.",
                                    "The source says exact acoustic feature order is unknown; masks are numerical partitions, not semantic frequency groups."]
        else:
            raise ValueError(f"Unknown public dataset: {name}")
    return X, labels, names, groups, extra


def _select_rows(X: np.ndarray, groups: np.ndarray | None, seed: int, max_rows: int = 3000):
    """First occurrence of each feature row, then seeded label-blind sampling."""
    canonical = np.array(X, dtype=np.float64, copy=True)
    canonical[canonical == 0] = 0
    _, first = np.unique(canonical, axis=0, return_index=True)
    candidates = np.sort(first)
    rng = np.random.default_rng(seed)
    if groups is not None and len(candidates) > max_rows:
        units = np.unique(groups[candidates])
        cap = max_rows // len(units)
        if cap < 1:
            raise ValueError("Dataset has more source groups than the requested sample bound")
        selected = np.concatenate([rng.choice(candidates[groups[candidates] == group],
                                              size=min(cap, np.sum(groups[candidates] == group)), replace=False)
                                   for group in units])
        method = f"seeded uniform sampling up to {cap} rows per source group; labels unused"
    elif len(candidates) > max_rows:
        selected = rng.choice(candidates, size=max_rows, replace=False)
        method = f"seeded uniform sample of {max_rows} unique feature rows; labels unused"
    else:
        selected = candidates
        method = "all unique feature rows; labels unused"
    return np.sort(selected).astype(np.int64), {"method": method, "seed": int(seed), "max_rows": max_rows,
        "deduplication": "retain first source occurrence of identical feature vectors, independent of labels and before splitting",
        "duplicate_rows_removed": int(len(X) - len(candidates)), "unique_source_rows": int(len(candidates))}


def load_public_dataset(name: str, seed: int = 2026) -> RawDataset:
    if name not in _SOURCES:
        raise ValueError(f"Unknown public dataset {name!r}; choose one of {PUBLIC_DATASET_NAMES}")
    path = _archive_path(name)
    X, labels, names, groups, extra = _parse_dataset(name, path)
    identifier, slug, digest, full_rows, dimensions, citation, doi = _SOURCES[name]
    X = np.asarray(X, dtype=np.float64)
    if X.shape != (full_rows, dimensions) or len(labels) != full_rows or not np.isfinite(X).all():
        raise ValueError(f"Unexpected source dimensions or nonfinite data for {name}: {X.shape}")
    indices, selection = _select_rows(X, groups, seed)
    label_names, encoded = np.unique(labels.astype(str), return_inverse=True)
    modality = extra.pop("modality")
    metadata = {
        "source": f"Official UCI dataset {identifier}",
        "source_url": f"https://archive.ics.uci.edu/dataset/{identifier}/{slug}",
        "download_url": f"https://archive.ics.uci.edu/static/public/{identifier}/{slug}.zip",
        "source_archive_sha256": digest, "source_archive_bytes": path.stat().st_size,
        "source_verified_date": "2026-09-16", "source_full_rows": full_rows,
        "source_full_features": dimensions, "sample_rows": len(indices),
        "citation": citation + f" https://doi.org/10.24432/{doi}",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "task_type": "classification", "label_role": "evaluation_only",
        "label_names": label_names.tolist(), "input_shape": [dimensions],
        "sample_selection": selection, "source_row_indices": indices.tolist(),
        "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "complexity_rationale": _COMPLEXITY[name],
        "split_policy": "held-out source groups" if groups is not None else "label-independent random record split",
        "scientific_claim_boundary": "Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.",
        **extra,
    }
    return RawDataset(name, modality, X[indices].copy(), names, encoded[indices].astype(np.int64),
                      None if groups is None else groups[indices].copy(), roles={column: "measurement" for column in names},
                      metadata=metadata)


def propose_public_tasks(dataset: RawDataset) -> list[TaskSpec]:
    """Two a priori masks per registered dataset; no labels/statistical search."""
    name, d = dataset.name, dataset.X.shape[1]
    if name not in _SOURCES:
        raise ValueError(f"No public task templates for {name!r}")
    all_indices = list(range(d))
    if name == "semeion":
        target = [r * 16 + c for r in range(4, 12) for c in range(4, 12)]
        return [TaskSpec("center", [i for i in all_indices if i not in target], target, "Outer pixels predict the central 8x8 region."),
                TaskSpec("right_half", [r * 16 + c for r in range(16) for c in range(8)],
                         [r * 16 + c for r in range(16) for c in range(8, 16)], "Left eight columns predict right eight columns.")]
    if name == "breast_cancer":
        return [TaskSpec("mean_to_worst", list(range(10)), list(range(20, 30)), "Mean morphology predicts worst-value morphology; IDs and diagnosis remain excluded."),
                TaskSpec("mean_to_error", list(range(10)), list(range(10, 20)), "Mean morphology predicts standard-error morphology.")]
    if name in {"ionosphere", "pendigits"}:
        points = d // 2
        even = [2 * p + c for p in range(0, points, 2) for c in range(2)]
        odd = [i for i in all_indices if i not in even]
        split = 2 * (points // 2)
        return [TaskSpec("alternating_pairs", even, odd, "Alternate ordered pulse/trajectory pairs predict the withheld pairs."),
                TaskSpec("first_half", list(range(split)), list(range(split, d)), "Earlier ordered pairs predict later pairs; downstream labels are classification labels.")]
    if name == "satimage":
        center = list(range(16, 20))
        first_bands = [4 * pixel + band for pixel in range(9) for band in (0, 1)]
        return [TaskSpec("neighbor_to_center", [i for i in all_indices if i not in center], center, "Eight neighbors predict all four central-pixel bands."),
                TaskSpec("spectral_halves", first_bands, [i for i in all_indices if i not in first_bands], "First two bands predict last two bands at each of nine pixels.")]
    if name == "har":
        source_names = dataset.metadata["source_feature_names"]
        context = [i for i, feature in enumerate(source_names) if feature.startswith("t")]
        target = [i for i, feature in enumerate(source_names) if feature.startswith("f")]
        return [TaskSpec("time_to_frequency", context, target, "Source-defined time-domain features predict frequency-domain features; subjects remain held out."),
                TaskSpec("alternating", all_indices[::2], all_indices[1::2], "Alternate named inertial summary columns predict withheld columns.")]
    return [TaskSpec("alternating", all_indices[::2], all_indices[1::2], "Fixed alternating numerical features; no label-driven feature choice."),
            TaskSpec("first_half", all_indices[:d // 2], all_indices[d // 2:], "Fixed first-half to second-half numerical features; no semantic ordering claim.")]
