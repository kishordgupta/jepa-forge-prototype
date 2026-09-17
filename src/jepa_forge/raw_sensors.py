"""Checksummed public sensor streams with original file and sample provenance.

Subject IDs are public, pseudonymous source identifiers, never predictive inputs.
No source data are bundled. Windows never cross files, gaps, or activity changes.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import urlopen
import zipfile

import numpy as np

from .schema import RawDataset


SOURCES = {
    "mhealth": {
        "url": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
        "sha256": "16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0",
        "doi": "10.24432/C5TW22", "hz": 50, "step": 1,
        "columns": [0, 1, 2, 5, 6, 7, 14, 15, 16],
        "locations": ["chest", "left_ankle", "right_lower_arm"],
    },
    "pamap2": {
        "url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "sha256": "76b3580bd5a804121f507717cb498c66a312ef5c774f4a78ac470e6558add352",
        "doi": "10.24432/C5NW2H", "hz": 100, "step": 2,
        "columns": [4, 5, 6, 21, 22, 23, 38, 39, 40],
        "locations": ["hand", "chest", "ankle"],
    },
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sensor_windows(values, labels, timestamps, step, length=64):
    """Return valid, disjoint source intervals; no interpolation or label voting."""
    span = length * step
    rows = []
    for start in range(0, len(values) - span + 1, span):
        stop = start + span
        label = labels[start]
        if label <= 0 or np.any(labels[start:stop] != label):
            continue
        if not np.isfinite(values[start:stop]).all():
            continue
        if timestamps is not None:
            delta = np.diff(timestamps[start:stop])
            # PAMAP2 samples every 0.01 s. Gaps invalidate the entire window.
            if np.any(np.abs(delta - 0.01) > 0.002):
                continue
        rows.append((start, stop, int(label)))
    return rows


def load_raw_sensor(name, seed=2026, per_subject=160, cache_dir=".cache/public_datasets"):
    corpus, task = name.rsplit("_", 1)
    if corpus not in SOURCES or task not in {"activity", "forecast"}:
        raise ValueError("Expected mhealth/pamap2 with _activity or _forecast")
    spec = SOURCES[corpus]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{corpus}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with urlopen(spec["url"], timeout=120) as inp, partial.open("wb") as out:
            while block := inp.read(8 * 1024 * 1024):
                out.write(block)
        if sha256(partial) != spec["sha256"]:
            raise ValueError("Sensor source checksum mismatch")
        partial.replace(archive)
    if sha256(archive) != spec["sha256"]:
        raise ValueError("Cached sensor source checksum mismatch")
    derived = cache / f"{corpus}_raw_v1_s{seed}_n{per_subject}.npz"
    metadata_path = derived.with_suffix(".json")
    if not derived.exists():
        rows, labels, groups, intervals, provenance, counts = [], [], [], [], [], []
        with zipfile.ZipFile(archive) as outer:
            payload = outer if corpus == "mhealth" else zipfile.ZipFile(io.BytesIO(outer.read("PAMAP2_Dataset.zip")))
            pattern = r"MHEALTHDATASET/mHealth_subject(\d+)\.log" if corpus == "mhealth" else r"PAMAP2_Dataset/Protocol/subject(\d+)\.dat"
            for member in sorted(payload.namelist()):
                match = re.fullmatch(pattern, member)
                if not match:
                    continue
                subject = int(match[1])
                # Subject 109 has only an optional rope-jumping recording in the
                # protocol directory. Exclusion is source-defined, not metric-based.
                if corpus == "pamap2" and subject == 109:
                    continue
                raw_bytes = payload.read(member)
                raw = np.loadtxt(io.BytesIO(raw_bytes))
                values = raw[:, spec["columns"]]
                y = raw[:, -1 if corpus == "mhealth" else 1].astype(int)
                timestamps = None if corpus == "mhealth" else raw[:, 0]
                valid = sensor_windows(values, y, timestamps, spec["step"])
                rng = np.random.default_rng(seed + subject)
                chosen = np.sort(rng.choice(len(valid), min(per_subject, len(valid)), replace=False))
                counts.append({"subject": subject, "valid_windows": len(valid), "retained": len(chosen)})
                member_hash = hashlib.sha256(raw_bytes).hexdigest()
                for j in chosen:
                    start, stop, label = valid[j]
                    rows.append(values[start:stop:spec["step"]].reshape(-1))
                    labels.append(label)
                    groups.append(subject)
                    intervals.append([start, stop - 1])
                    provenance.append({"member": member, "member_sha256": member_hash,
                                       "source_row_start_zero_based": start,
                                       "source_row_stop_exclusive": stop,
                                       "sample_step": spec["step"],
                                       "start_seconds": float(timestamps[start]) if timestamps is not None else start / spec["hz"],
                                       "end_seconds_inclusive": float(timestamps[stop-1]) if timestamps is not None else (stop-1) / spec["hz"]})
            if payload is not outer:
                payload.close()
        X = np.asarray(rows, dtype=np.float64)
        np.savez_compressed(derived, X=X, labels=labels, groups=groups, intervals=intervals)
        metadata_path.write_text(json.dumps({"provenance": provenance, "counts": counts,
                                             "derived_sha256": sha256(derived)}, sort_keys=True))
    meta = json.loads(metadata_path.read_text())
    if sha256(derived) != meta["derived_sha256"]:
        raise ValueError("Derived sensor cache checksum mismatch")
    with np.load(derived, allow_pickle=False) as data:
        X, labels, groups, intervals = (data[k].copy() for k in ("X", "labels", "groups", "intervals"))
    feature_names = [f"t{t:02d}_{location}_acc_{axis}" for t in range(64)
                     for location in spec["locations"] for axis in "xyz"]
    return RawDataset(name, "sequence", X, feature_names,
                      y=np.unique(labels, return_inverse=True)[1] if task == "activity" else None,
                      groups=groups, intervals=intervals,
                      roles={n: "measurement" for n in feature_names},
                      metadata={"source": corpus, "source_url": spec["url"],
                                "archive_sha256": spec["sha256"], "citation_doi": spec["doi"],
                                "license": "CC BY 4.0", "input_shape": [64, 9],
                                "feature_time_indices": np.repeat(np.arange(64), 9).tolist(),
                                "task_type": "classification" if task == "activity" else "forecasting",
                                "temporal_direction": "past_to_future" if task == "forecast" else "bidirectional_context", "flatten_order": "time-major",
                                "source_hz": spec["hz"], "sample_step": spec["step"],
                                "effective_hz": 50, "units": "m/s^2", "subject_counts": meta["counts"],
                                "source_label_values": np.unique(labels).tolist(),
                                "interval_semantics": "inclusive original file sample rows; whole subjects stay in one partition",
                                "window_provenance": meta["provenance"], "generator_seed": seed,
                                "filter": "finite accelerometer channels, contiguous samples, single nonzero activity per window",
                                "sampling": "up to 160 nonoverlapping windows per subject, label-independent random sampling after transition/null filtering",
                                "limitations": ["Activity labels remove transition and null windows in both tasks; this is segmented-activity forecasting, not unconstrained streaming.",
                                                "PAMAP2 is decimated by two without antialias filtering; original sample indices are retained.",
                                                "Eight PAMAP2 protocol subjects are used; source-limited subject 109 is excluded."]})
