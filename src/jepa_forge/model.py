"""Small, structure-aware JEPA-style pilot models.

The online encoder sees context values and an observation mask. A frozen EMA
encoder sees target values and its mask, and a predictor matches its latent
representation with stop-gradient MSE. A context variance penalty is an explicit
pilot adaptation to discourage collapse; this is not an implementation of a
particular published JEPA architecture. Labels are never consumed here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 0.001
    latent_dim: int = 32
    ema: float = 0.99
    variance_weight: float = 1.0
    seed: int = 7
    device: str = "mps"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 2 or self.latent_dim < 2:
            raise ValueError("epochs >= 1, batch_size >= 2 and latent_dim >= 2 are required")
        if self.learning_rate <= 0 or not 0 <= self.ema < 1 or self.variance_weight < 0:
            raise ValueError("Invalid learning rate, EMA decay, or variance weight")


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS GPU requested but unavailable; CPU fallback is disabled")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
            raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK to ensure GPU execution")
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU requested but unavailable; CPU fallback is disabled")
    elif device.type != "cpu":
        raise ValueError("Supported devices are cpu, mps, and cuda")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_spec(compiled: Any) -> dict[str, Any]:
    X = np.asarray(compiled.X)
    if X.ndim != 2:
        raise ValueError("Compiled X must be a standardized [sample, feature] array")
    modality = str(compiled.dataset.modality).lower()
    # The compiler may call the same two structured modalities by these names.
    modality = {"tabular": "tabular", "image": "image", "images": "image",
                "sequence": "sequence", "time_series": "sequence",
                "timeseries": "sequence", "sensor": "sequence"}.get(modality, modality)
    if modality not in {"tabular", "image", "sequence"}:
        raise ValueError(f"Unsupported modality: {modality}")
    input_shape = [int(v) for v in compiled.dataset.metadata.get("input_shape", [X.shape[1]])]
    if math.prod(input_shape) != X.shape[1]:
        raise ValueError("input_shape does not match the flattened feature count")
    if modality in {"image", "sequence"} and len(input_shape) != 2:
        raise ValueError("Images need [height, width]; sequences need [time, channels]")
    context = [int(v) for v in compiled.task.context]
    target = [int(v) for v in compiled.task.target]
    if not context or not target or len(set(context)) != len(context) or len(set(target)) != len(target):
        raise ValueError("Context and target must have nonempty, unique feature indices")
    if set(context) & set(target):
        raise ValueError("Context and target features must be disjoint")
    if min(context + target) < 0 or max(context + target) >= X.shape[1]:
        raise ValueError("Task index is outside the feature range")
    return {"modality": modality, "input_shape": input_shape, "input_dim": int(X.shape[1]),
            "context": context, "target": target}


def _encoder(spec: dict[str, Any], latent_dim: int) -> nn.Module:
    hidden = 64
    if spec["modality"] == "tabular":
        return nn.Sequential(nn.Linear(2 * spec["input_dim"], hidden), nn.GELU(),
                             nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent_dim))
    if spec["modality"] == "image":
        height, width = spec["input_shape"]
        return nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.GELU(),
                             nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.Flatten(),
                             nn.Linear(32 * height * width, hidden), nn.GELU(),
                             nn.Linear(hidden, latent_dim))
    length, channels = spec["input_shape"]
    return nn.Sequential(nn.Conv1d(2 * channels, 32, 5, padding=2), nn.GELU(),
                         nn.Conv1d(32, 32, 3, padding=1), nn.GELU(), nn.Flatten(),
                         nn.Linear(32 * length, hidden), nn.GELU(), nn.Linear(hidden, latent_dim))


class JEPAEncoder(nn.Module):
    """Context-only inference interface over a separately masked target teacher.

    Public inference inputs are already standardized *context columns*, in the
    exact order compiled.task.context. Full rows and target values are rejected.
    ``encode_target`` is a separate teacher diagnostic, not a probe input.
    """

    def __init__(self, spec: dict[str, Any], config: TrainConfig):
        super().__init__()
        self.spec = deepcopy(spec)
        self.config = config
        self.online_encoder = _encoder(spec, config.latent_dim)
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(config.latent_dim, 64), nn.GELU(),
                                       nn.Linear(64, config.latent_dim))
        self.register_buffer("context_indices", torch.tensor(spec["context"], dtype=torch.long))
        self.register_buffer("target_indices", torch.tensor(spec["target"], dtype=torch.long))
        self.target_encoder.eval()

    @property
    def device(self) -> torch.device:
        return next(self.online_encoder.parameters()).device

    def train(self, mode: bool = True) -> "JEPAEncoder":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _pack(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != len(indices):
            raise ValueError(f"Expected [sample, {len(indices)}] observed values")
        full = values.new_zeros((values.shape[0], self.spec["input_dim"]))
        full[:, indices] = values
        mask = values.new_zeros(full.shape)
        mask[:, indices] = 1.0
        if self.spec["modality"] == "tabular":
            return torch.cat((full, mask), dim=1)
        if self.spec["modality"] == "image":
            shape = (values.shape[0], *self.spec["input_shape"])
            return torch.stack((full.reshape(shape), mask.reshape(shape)), dim=1)
        shape = (values.shape[0], *self.spec["input_shape"])
        # Raw flattened sequences are time-major: [sample, time, channels].
        return torch.cat((full.reshape(shape), mask.reshape(shape)), dim=2).transpose(1, 2)

    def forward_context(self, values: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(self._pack(values, self.context_indices))

    @torch.no_grad()
    def forward_target(self, values: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(self._pack(values, self.target_indices))

    @torch.no_grad()
    def update_target(self) -> None:
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.mul_(self.config.ema).add_(online, alpha=1.0 - self.config.ema)

    def _encode(self, values: np.ndarray, target: bool, batch_size: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        expected = len(self.spec["target" if target else "context"])
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(f"Expected [sample, {expected}] {'target' if target else 'context'} columns only")
        if not np.isfinite(values).all() or batch_size < 1:
            raise ValueError("Inputs must be finite and batch_size positive")
        if not len(values):
            return np.empty((0, self.config.latent_dim), dtype=np.float32)
        was_training = self.training
        self.eval()
        chunks = []
        try:
            with torch.inference_mode():
                for start in range(0, len(values), batch_size):
                    batch = torch.as_tensor(values[start:start + batch_size], device=self.device)
                    z = self.forward_target(batch) if target else self.forward_context(batch)
                    chunks.append(z.cpu().numpy())
        finally:
            self.train(was_training)
        return np.concatenate(chunks)

    def encode_context(self, context_values: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return self._encode(context_values, target=False, batch_size=batch_size)

    def encode_target(self, target_values: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return self._encode(target_values, target=True, batch_size=batch_size)


def build_model(compiled: Any, config: TrainConfig | None = None) -> JEPAEncoder:
    """Construct a seeded model; also supplies the paired untrained baseline."""
    config = config or TrainConfig()
    device = _resolve_device(config.device)
    # Initialize on CPU before moving to the requested device for matched weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = JEPAEncoder(_model_spec(compiled), config)
    return model.to(device)


def representation_stats(embeddings: np.ndarray) -> dict[str, float | int | bool]:
    """CPU float64 covariance diagnostics, not a proof of useful representations."""
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("At least two finite embeddings are needed for diagnostics")
    centered = z - z.mean(axis=0)
    std = z.std(axis=0, ddof=0)
    eigenvalues = np.clip(np.linalg.eigvalsh(centered.T @ centered / len(z)), 0, None)
    total = eigenvalues.sum()
    if total > 0:
        weights = eigenvalues[eigenvalues > 0] / total
        rank = float(np.exp(-(weights * np.log(weights)).sum()))
    else:
        rank = 0.0
    return {"n_samples": len(z), "latent_dim": z.shape[1], "mean_std": float(std.mean()),
            "min_std": float(std.min()), "effective_rank": rank,
            "normalized_effective_rank": rank / min(z.shape[1], len(z) - 1),
            "heuristic_collapse_flag": bool(std.mean() < 0.01 or rank < 2.0)}


def _loss(model: JEPAEncoder, context: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
    z_context = model.forward_context(context)
    z_target = model.forward_target(target)
    prediction = model.predictor(z_context)
    prediction_loss = F.mse_loss(prediction, z_target)
    variance_loss = F.relu(1.0 - torch.sqrt(z_context.var(dim=0, unbiased=False) + 1e-4)).mean()
    return prediction_loss + model.config.variance_weight * variance_loss, prediction_loss, variance_loss


def _batches(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    # Balanced splits prevent singleton last batches from defeating the std penalty.
    # With batch_size=2 and an odd sample count, one batch contains three rows.
    count = max(1, min(math.ceil(len(indices) / batch_size), len(indices) // 2))
    return list(np.array_split(indices, count))


def _pairing_diagnostic(model: JEPAEncoder, context: torch.Tensor, target: torch.Tensor,
                        seed: int, batch_size: int) -> dict[str, Any]:
    """Compare matched and shuffled validation targets after all optimization."""
    model.eval()
    predictions, teacher_latents = [], []
    with torch.no_grad():
        for rows in _batches(np.arange(len(context)), batch_size):
            index = torch.as_tensor(rows, dtype=torch.long, device=model.device)
            predictions.append(model.predictor(model.forward_context(context[index])).cpu().numpy())
            teacher_latents.append(model.forward_target(target[index]).cpu().numpy())
    predicted = np.concatenate(predictions).astype(np.float64)
    teacher = np.concatenate(teacher_latents).astype(np.float64)
    # A seeded random cycle is a reproducible permutation without fixed points.
    order = np.random.default_rng(seed).permutation(len(teacher))
    shuffled_indices = np.empty(len(teacher), dtype=np.int64)
    shuffled_indices[order] = np.roll(order, 1)
    true_mse = float(np.square(predicted - teacher).mean())
    shuffled_mse = float(np.square(predicted - teacher[shuffled_indices]).mean())
    return {
        "split": "validation", "n_rows": len(teacher), "shuffle_seed": seed,
        "shuffle_fixed_points": int(np.sum(shuffled_indices == np.arange(len(teacher)))),
        "true_target_latent_mse": true_mse,
        "shuffled_target_latent_mse": shuffled_mse,
        "shuffled_minus_true_mse": shuffled_mse - true_mse,
        "shuffled_to_true_mse_ratio": shuffled_mse / true_mse if true_mse > 0 else None,
        "teacher_frozen_during_diagnostic": True,
        "used_for_model_selection": False,
        "interpretation": "Higher shuffled loss is heuristic evidence of context-target correspondence. A single shuffle is not a significance test, an independent utility result, or proof against collapse.",
    }


def train_jepa(compiled: Any, config: TrainConfig, output_dir: str | Path) -> tuple[JEPAEncoder, dict[str, Any]]:
    """Train for exactly the predeclared epochs; never select using test outcomes."""
    model = build_model(compiled, config)
    X = np.asarray(compiled.X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Training data must be finite")
    train_indices = np.asarray(compiled.splits["train"], dtype=np.int64)
    val_indices = np.asarray(compiled.splits["val"], dtype=np.int64)
    if len(train_indices) < 2 or len(val_indices) < 2:
        raise ValueError("At least two training and validation rows are required")
    for indices in (train_indices, val_indices):
        if len(set(indices.tolist())) != len(indices) or indices.min() < 0 or indices.max() >= len(X):
            raise ValueError("Invalid split indices")
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Training and validation splits overlap")
    context_columns = np.asarray(model.spec["context"])
    target_columns = np.asarray(model.spec["target"])
    # Only selected rows are copied to the GPU; held-out test rows remain unused.
    train_context = torch.as_tensor(X[np.ix_(train_indices, context_columns)], device=model.device)
    train_target = torch.as_tensor(X[np.ix_(train_indices, target_columns)], device=model.device)
    val_context = torch.as_tensor(X[np.ix_(val_indices, context_columns)], device=model.device)
    val_target = torch.as_tensor(X[np.ix_(val_indices, target_columns)], device=model.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    monitor = X[np.ix_(train_indices[:min(512, len(train_indices))], context_columns)]
    initial_stats = representation_stats(model.encode_context(monitor))
    history: list[dict[str, Any]] = []
    device_evidence: dict[str, Any] = {}
    _synchronize(model.device)
    start_time = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        accumulated = np.zeros(3, dtype=np.float64)
        _synchronize(model.device)
        train_start = time.perf_counter()
        for rows in _batches(rng.permutation(len(train_indices)), config.batch_size):
            index = torch.as_tensor(rows, dtype=torch.long, device=model.device)
            context, target = train_context[index], train_target[index]
            optimizer.zero_grad(set_to_none=True)
            losses = _loss(model, context, target)
            if not bool(torch.isfinite(losses[0]).item()):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            losses[0].backward()
            if not device_evidence:
                device_evidence = {
                    "requested_device": config.device,
                    "online_parameter_device": str(next(model.online_encoder.parameters()).device),
                    "target_parameter_device": str(next(model.target_encoder.parameters()).device),
                    "context_batch_device": str(context.device), "target_batch_device": str(target.device),
                    "loss_device": str(losses[0].device),
                    "online_gradient_device": str(next(model.online_encoder.parameters()).grad.device),
                    "target_requires_grad": any(p.requires_grad for p in model.target_encoder.parameters()),
                    "target_has_gradient": any(p.grad is not None for p in model.target_encoder.parameters()),
                    "gpu_fallback_enabled": False,
                }
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            optimizer.step()
            model.update_target()
            accumulated += np.asarray([v.detach().item() for v in losses]) * len(rows)
        _synchronize(model.device)
        train_seconds = time.perf_counter() - train_start
        model.eval()
        val_losses = np.zeros(3, dtype=np.float64)
        with torch.no_grad():
            for rows in _batches(np.arange(len(val_indices)), config.batch_size):
                index = torch.as_tensor(rows, dtype=torch.long, device=model.device)
                losses = _loss(model, val_context[index], val_target[index])
                val_losses += np.asarray([v.item() for v in losses]) * len(rows)
        train_losses = accumulated / len(train_indices)
        val_losses /= len(val_indices)
        stats = representation_stats(model.encode_context(monitor))
        entry = {"epoch": epoch, "train_total_loss": float(train_losses[0]),
                 "train_prediction_loss": float(train_losses[1]), "train_variance_loss": float(train_losses[2]),
                 "val_total_loss": float(val_losses[0]), "val_prediction_loss": float(val_losses[1]),
                 "val_variance_loss": float(val_losses[2]), "train_seconds": train_seconds,
                 "train_context_diagnostics": stats}
        history.append(entry)
    pairing_diagnostic = _pairing_diagnostic(model, val_context, val_target,
                                             seed=config.seed + 2026, batch_size=config.batch_size)
    _synchronize(model.device)
    elapsed = time.perf_counter() - start_time
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    # Tensor-only state dict; no pickled application object or optimizer state.
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
    config_document = {"train_config": asdict(config), "model_spec": model.spec,
                       "input_contract": "Already standardized context-only columns, in task.context order"}
    (output_dir / "config.json").write_text(json.dumps(config_document, indent=2) + "\n")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    run_info = {
        "config": asdict(config), "model_spec": model.spec,
        "method": "JEPA-style context-to-target EMA latent prediction with context variance regularization",
        "loss": "mean_squared_error(prediction, stop_gradient(EMA_target)) + variance_weight * mean(relu(1 - sqrt(var(context_latents) + 1e-4)))",
        "selection_criterion": "Final epoch from predeclared fixed epoch count; validation logged only",
        "selected_epoch": config.epochs, "labels_used_for_training": False,
        "test_rows_used_for_training_or_selection": False,
        "n_train": len(train_indices), "n_val": len(val_indices),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "frozen_target_parameters": sum(p.numel() for p in model.target_encoder.parameters()),
        "device_evidence": device_evidence,
        "synchronized_training_and_monitoring_seconds": elapsed,
        "synchronized_training_seconds": sum(entry["train_seconds"] for entry in history),
        "timing_note": "Wall time includes batching, optimizer, device synchronization; total additionally includes validation, CPU covariance diagnostics, and final validation target-shuffle diagnostic. Not a hardware benchmark.",
        "initial_train_context_diagnostics": initial_stats,
        "final_train_context_diagnostics": history[-1]["train_context_diagnostics"],
        "collapse_heuristic": "mean coordinate std < 0.01 OR covariance effective rank < 2; heuristic only",
        "normalized_effective_rank_definition": "covariance effective rank / min(latent_dim, n_samples - 1)",
        "validation_pairing_diagnostic": pairing_diagnostic,
        "training_history": history,
        "checkpoint": str(checkpoint),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    model.eval()
    return model, run_info


def load_model(compiled: Any, checkpoint_path: str | Path, device: str = "cpu") -> JEPAEncoder:
    """Load tensor weights after verifying the supplied task's feature contract.

    Scaling remains the compiler's responsibility. Persist compiler artifacts
    alongside this checkpoint to reproduce inputs, as the experiment runner does.
    """
    checkpoint_path = Path(checkpoint_path)
    saved = json.loads((checkpoint_path.parent / "config.json").read_text())
    if saved["model_spec"] != _model_spec(compiled):
        raise ValueError("Checkpoint context, target, or shape differs from the compiled task")
    config = TrainConfig(**{**saved["train_config"], "device": device})
    model = build_model(compiled, config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model
