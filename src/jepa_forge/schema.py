"""Explicit data contracts for the small JEPA-FORGE compiler."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RawDataset:
    name: str
    modality: str
    X: np.ndarray
    feature_names: list[str]
    y: np.ndarray | None = None
    groups: np.ndarray | None = None
    intervals: np.ndarray | None = None
    roles: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    name: str
    context: list[int]
    target: list[int]
    description: str = ""


@dataclass
class CompiledTask:
    dataset: RawDataset
    task: TaskSpec
    X: np.ndarray
    splits: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    report: dict[str, Any]
    manifest: dict[str, Any]
