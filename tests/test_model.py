"""Fast CPU tests for the context boundary, EMA update, and checkpoint contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jepa_forge.model import TrainConfig, _batches, build_model, load_model, representation_stats, train_jepa


class LabelGuardDataset:
    def __init__(self, modality, shape):
        self.name = "test_fixture"
        self.modality = modality
        self.metadata = {"input_shape": shape}

    @property
    def y(self):
        raise AssertionError("Self-supervised model must never read labels")


def fixture_task(modality="tabular", shape=(6,)):
    X = np.random.default_rng(13).normal(size=(24, np.prod(shape))).astype(np.float32)
    middle = X.shape[1] // 2
    return SimpleNamespace(
        X=X, dataset=LabelGuardDataset(modality, list(shape)),
        task=SimpleNamespace(context=list(range(middle)), target=list(range(middle, X.shape[1]))),
        splits={"train": np.arange(16), "val": np.arange(16, 20), "test": np.arange(20, 24)},
    )


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("modality,shape", [("tabular", (6,)), ("image", (8, 8)), ("sequence", (24, 2))])
def test_context_only_boundary_and_seeded_initialization(modality, shape):
    compiled = fixture_task(modality, shape)
    config = TrainConfig(epochs=2, device="cpu")
    model = build_model(compiled, config)
    paired = build_model(compiled, config)
    context = compiled.X[:, compiled.task.context].copy()
    before = model.encode_context(context)
    np.testing.assert_array_equal(before, paired.encode_context(context))
    # Mutating every withheld value cannot change the public inference input.
    compiled.X[:, compiled.task.target] = 1e6
    np.testing.assert_array_equal(before, model.encode_context(compiled.X[:, compiled.task.context]))
    with pytest.raises(ValueError, match="context columns only"):
        model.encode_context(compiled.X)
    assert before.shape == (24, config.latent_dim)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())


def test_training_updates_ema_without_gradients_and_round_trips(tmp_path):
    compiled = fixture_task()
    config = TrainConfig(epochs=2, batch_size=7, device="cpu", ema=0.9)
    initial = build_model(compiled, config)
    initial_teacher = [p.clone() for p in initial.target_encoder.parameters()]
    trained, info = train_jepa(compiled, config, tmp_path)
    assert any(not torch.equal(before, after) for before, after in zip(initial_teacher, trained.target_encoder.parameters()))
    assert all(not p.requires_grad and p.grad is None for p in trained.target_encoder.parameters())
    assert info["selected_epoch"] == 2
    assert info["labels_used_for_training"] is False
    assert info["test_rows_used_for_training_or_selection"] is False
    assert info["device_evidence"]["loss_device"] == "cpu"
    assert len(info["training_history"]) == 2
    assert np.isfinite(info["training_history"][-1]["train_total_loss"])
    diagnostic = info["validation_pairing_diagnostic"]
    assert diagnostic["shuffle_fixed_points"] == 0
    assert diagnostic["used_for_model_selection"] is False
    assert diagnostic["true_target_latent_mse"] >= 0
    assert diagnostic["shuffled_target_latent_mse"] >= 0
    assert 0 <= info["final_train_context_diagnostics"]["normalized_effective_rank"] <= 1.00001
    restored = load_model(compiled, tmp_path / "checkpoint.pt", device="cpu")
    np.testing.assert_array_equal(trained.encode_context(compiled.X[:, compiled.task.context]),
                                  restored.encode_context(compiled.X[:, compiled.task.context]))
    compiled.task.context = compiled.task.context[::-1]
    with pytest.raises(ValueError, match="differs"):
        load_model(compiled, tmp_path / "checkpoint.pt")


def test_ema_is_exact_convex_update():
    model = build_model(fixture_task(), TrainConfig(device="cpu", ema=0.75))
    old_teacher = [p.clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.online_encoder.parameters():
            p.add_(2.0)
    model.update_target()
    for old, actual in zip(old_teacher, model.target_encoder.parameters()):
        torch.testing.assert_close(actual, old + 0.5)


def test_gpu_request_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        build_model(fixture_task(), TrainConfig(device="mps"))


def test_degenerate_embeddings_trigger_diagnostic():
    stats = representation_stats(np.ones((12, 4)))
    assert stats["mean_std"] == 0
    assert stats["effective_rank"] == 0
    assert stats["heuristic_collapse_flag"] is True


def test_overlap_is_rejected():
    compiled = fixture_task()
    compiled.task.target[0] = compiled.task.context[0]
    with pytest.raises(ValueError, match="disjoint"):
        build_model(compiled, TrainConfig(device="cpu"))


@pytest.mark.parametrize("n_rows,batch_size", [(3, 2), (5, 2), (7, 3), (129, 128), (257, 128)])
def test_balanced_batches_never_leave_singleton(n_rows, batch_size):
    indices = np.arange(n_rows)
    batches = _batches(indices, batch_size)
    assert all(len(batch) >= 2 for batch in batches)
    np.testing.assert_array_equal(np.concatenate(batches), indices)


def test_normalized_rank_accounts_for_sample_count():
    z = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    stats = representation_stats(z)
    assert stats["effective_rank"] == pytest.approx(2)
    assert stats["normalized_effective_rank"] == pytest.approx(1)
