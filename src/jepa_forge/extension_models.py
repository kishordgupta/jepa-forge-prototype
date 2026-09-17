"""Predeclared neural baselines and checkpoint-budget controls for extension study."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import TrainConfig, build_model, _batches, _loss, _synchronize, representation_stats
from .selection import file_hash


def make_neural(kind, compiled, config, d_out):
    torch.manual_seed(config.seed)
    if kind in {"jepa", "supervised"}:
        model = build_model(compiled, config)
        if kind == "supervised":
            model.predictor = nn.Linear(config.latent_dim, d_out).to(model.device)
        return model
    if kind == "tabm":
        from tabm import TabM
        return TabM.make(n_num_features=len(compiled.task.context), d_out=d_out,
                         n_blocks=2, d_block=128, dropout=0.1, k=8).to(config.device)
    raise ValueError(kind)


def infer_neural(model, kind, context, batch_size=256):
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(context), batch_size):
            x = torch.as_tensor(np.asarray(context[start:start+batch_size], dtype=np.float32), device=device)
            if kind == "jepa":
                z = model.forward_context(x)
            elif kind == "supervised":
                z = model.predictor(model.forward_context(x))
            else:
                z = model(x)
            outputs.append(z.cpu().numpy())
    return np.concatenate(outputs)


def neural_prediction(outputs, kind, classification, mean, std):
    if kind == "tabm":
        if classification:
            # Average class probabilities, not logits, as prescribed by TabM.
            shifted = outputs - outputs.max(axis=-1, keepdims=True)
            probabilities = np.exp(shifted)
            outputs = (probabilities / probabilities.sum(axis=-1, keepdims=True)).mean(axis=1)
        else:
            outputs = outputs.mean(axis=1)
    return outputs.argmax(axis=1) if classification else outputs * std + mean


def train_checkpoints(compiled, target, classification, kind, config, checkpoints, directory, callback):
    """Train exactly max(checkpoints) epochs, invoking validation-only callbacks.

    The compiled object must contain only physically separated development rows.
    Labels for JEPA are used only by the callback after training updates.
    """
    if set(compiled.splits) != {"train", "val"}:
        raise ValueError("Neural extension training accepts development rows only")
    if sorted(set(checkpoints)) != list(checkpoints) or checkpoints[-1] != config.epochs:
        raise ValueError("Checkpoints must end at the total budget")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tr = compiled.splits["train"]
    d_out = int(np.max(target)) + 1 if classification else target.shape[1]
    model = make_neural(kind, compiled, config, d_out)
    device = next(model.parameters()).device
    context = compiled.X[:, compiled.task.context].astype(np.float32)
    x = torch.as_tensor(context[tr], device=device)
    future = torch.as_tensor(compiled.X[np.ix_(tr, compiled.task.target)].astype(np.float32), device=device)
    mean = np.zeros(1) if classification else target[tr].mean(axis=0)
    std = np.ones(1) if classification else np.maximum(target[tr].std(axis=0), 1e-6)
    y = torch.as_tensor(target[tr] if classification else (target[tr] - mean) / std,
                        dtype=torch.long if classification else torch.float32, device=device)
    if kind == "jepa":
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif kind == "supervised":
        parameters = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
    else:
        parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    records, evidence, epoch_losses = [], {}, []
    elapsed_training = 0.0
    total_updates = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        _synchronize(device)
        before = time.perf_counter()
        loss_sum = 0.
        for rows in _batches(rng.permutation(len(tr)), config.batch_size):
            idx = torch.as_tensor(rows, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if kind == "jepa":
                loss = _loss(model, x[idx], future[idx])[0]
            else:
                pred = model.predictor(model.forward_context(x[idx])) if kind == "supervised" else model(x[idx])
                if kind == "tabm":
                    truth = y[idx].unsqueeze(1).expand((-1, pred.shape[1]) if classification else (-1, pred.shape[1], -1))
                    loss = F.cross_entropy(pred.flatten(0, 1), truth.flatten(0, 1)) if classification else F.mse_loss(pred, truth)
                else:
                    loss = F.cross_entropy(pred, y[idx]) if classification else F.mse_loss(pred, y[idx])
            loss.backward()
            if not evidence:
                evidence = {"requested": str(config.device), "parameter": str(parameters[0].device),
                            "input": str(x.device), "loss": str(loss.device),
                            "gradient": str(parameters[0].grad.device), "cpu_fallback": False}
            nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            if kind == "jepa":
                model.update_target()
            loss_sum += float(loss.detach().cpu()) * len(rows)
            total_updates += 1
        _synchronize(device)
        elapsed_training += time.perf_counter() - before
        epoch_losses.append(loss_sum / len(tr))
        if not np.isfinite(epoch_losses[-1]):
            raise RuntimeError("Nonfinite neural loss; stop instead of dropping a candidate")
        if epoch in checkpoints:
            path = directory / f"epoch_{epoch}.pt"
            torch.save({"model": model.state_dict(), "config": asdict(config), "kind": kind,
                        "d_out": d_out, "mean": mean.tolist(), "std": std.tolist()}, path)
            outputs = infer_neural(model, kind, context)
            if not np.isfinite(outputs).all():
                raise RuntimeError("Nonfinite neural outputs")
            record = {"epoch": epoch, "checkpoint": path.name, "checkpoint_sha256": file_hash(path),
                      "updates": total_updates, "training_seconds": elapsed_training,
                      "device_evidence": evidence, "trainable_parameters": sum(p.numel() for p in parameters),
                      "training_loss": epoch_losses.copy(), "seed": config.seed}
            if kind == "jepa":
                record["representation"] = representation_stats(outputs[tr])
            callback(outputs, mean, std, record)
            records.append(record)
    del model
    return records


def restore_neural(compiled, checkpoint, expected_hash, device):
    if file_hash(checkpoint) != expected_hash:
        raise ValueError("Extension checkpoint changed after selection lock")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = TrainConfig(**{**saved["config"], "device": device})
    model = make_neural(saved["kind"], compiled, config, saved["d_out"])
    model.load_state_dict(saved["model"], strict=True)
    return model, saved
