"""Expanded evidence must be complete, finite, GPU-backed, and reproducible."""
from copy import deepcopy
import importlib.util
import json
import gzip
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "validate_expanded", Path(__file__).resolve().parents[1] / "scripts/validate_expanded.py"
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


@pytest.fixture
def evidence():
    config = {
        "datasets": ["fixture_class", "fixture_forecast"], "model_seeds": [7, 17, 27],
        "split_seed": 2026, "epochs": 2, "batch_size": 8, "learning_rate": 0.001,
        "latent_dim": 4, "ema": 0.99, "variance_weight": 1.0, "device": "mps",
    }
    plan, audits, runs = {}, [], []
    for dataset in config["datasets"]:
        classification = dataset == "fixture_class"
        for task_index, task in enumerate(["primary", "secondary"]):
            primary = task_index == 0
            plan[(dataset, task)] = {"classification": classification, "primary": primary}
            audits.append({"dataset": dataset, "task": task, "primary_a_priori": primary,
                           "report": {"status": "passed", "errors": []}, "export_roundtrip_passed": True})
            for seed_index, seed in enumerate(config["model_seeds"]):
                training = {
                    "config": {**{key: config[key] for key in validator.TRAIN_CONFIG_FIELDS}, "seed": seed},
                    "device_evidence": {**{key: "mps:0" for key in validator.DEVICE_FIELDS},
                                        "requested_device": "mps", "target_requires_grad": False,
                                        "target_has_gradient": False, "gpu_fallback_enabled": False},
                    "labels_used_for_training": False, "test_rows_used_for_training_or_selection": False,
                    "selected_epoch": 2, "training_history": [{"epoch": 1}, {"epoch": 2}],
                    "checkpoint_roundtrip": {"passed": True},
                    "validation_pairing_diagnostic": {"used_for_model_selection": False, "shuffle_fixed_points": 0},
                }
                validation = {"accuracy": 0.7, "macro_f1": 0.7} if classification else {"rmse": 0.3, "mae": 0.2}
                worse = {"accuracy": 0.6, "macro_f1": 0.6} if classification else {"rmse": 0.4, "mae": 0.3}
                test = {key: 0.5 + seed_index * 0.1 for key in validation}
                special = "full_raw_linear_reference" if classification else "persistence"
                rows = []
                for method in sorted(validator.BASE_METHODS | {special}):
                    row = {
                        "method": method, "information_budget": "full_information_reference" if method == "full_raw_linear_reference" else "context_only",
                        "validation": dict(validation), "test": dict(test),
                        "fit_partition": "none" if method == "persistence" else "train",
                        "selection_partition": "none" if method == "persistence" else "val", "evaluation_partition": "test",
                        "selected_parameters": {"strength": 0.1},
                        "selection_history": [{"parameters": {"strength": 0.1}, "validation": dict(validation)},
                                              {"parameters": {"strength": 1}, "validation": dict(worse)}],
                    }
                    rows.append(row)
                runs.append({"dataset": dataset, "task": task, "seed": seed,
                             "primary_a_priori": primary, "training": training, "evaluation": rows})
    return config, {"status": "complete", "config": deepcopy(config), "task_audits": audits, "runs": runs}, plan


def test_complete_recorded_matrix_passes(evidence):
    validator.validate_payload(*evidence)


@pytest.mark.parametrize("mutation,message", [
    (lambda c, d, p: d.update(status="running"), "not complete"),
    (lambda c, d, p: d["runs"].pop(), "Run matrix mismatch"),
    (lambda c, d, p: d["runs"].append(deepcopy(d["runs"][0])), "Duplicate.*runs"),
    (lambda c, d, p: d["task_audits"].append(deepcopy(d["task_audits"][0])), "Duplicate task audits"),
    (lambda c, d, p: d["task_audits"].pop(), "Task audit matrix"),
    (lambda c, d, p: d["runs"][0].update(seed=99), "Run matrix mismatch"),
    (lambda c, d, p: d["runs"][0].update(primary_a_priori=False), "Primary policy mismatch"),
    (lambda c, d, p: d["config"].update(epochs=3), "Recorded configuration"),
    (lambda c, d, p: d["runs"][0]["training"]["device_evidence"].update(online_gradient_device="cpu"), "Non-MPS evidence"),
    (lambda c, d, p: d["runs"][0]["training"]["device_evidence"].update(target_has_gradient=True), "teacher/fallback"),
    (lambda c, d, p: d["runs"][0]["training"]["device_evidence"].update(gpu_fallback_enabled=True), "teacher/fallback"),
    (lambda c, d, p: d["runs"][0]["training"].update(test_rows_used_for_training_or_selection=True), "Training used test"),
    (lambda c, d, p: d["runs"][0]["training"]["training_history"].pop(), "Incomplete training history"),
    (lambda c, d, p: d["runs"][0]["training"]["checkpoint_roundtrip"].update(passed=False), "Checkpoint round-trip"),
    (lambda c, d, p: d["runs"][0]["evaluation"][0]["test"].update(macro_f1=float("nan")), "Non-finite"),
    (lambda c, d, p: d["runs"][0]["evaluation"][0]["test"].update(macro_f1=1.1), "outside range"),
    (lambda c, d, p: d["runs"][0]["evaluation"][0].update(fit_partition="train+val"), "Probe partition"),
    (lambda c, d, p: d["runs"][0]["evaluation"][0].update(selected_parameters={"strength": 1}), "not validation-optimal"),
    (lambda c, d, p: d["runs"][0]["evaluation"].pop(), "Missing/unexpected methods"),
])
def test_invalid_evidence_is_rejected(evidence, mutation, message):
    mutation(*evidence)
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate_payload(*evidence)


@pytest.mark.parametrize("field", ["datasets", "model_seeds"])
def test_duplicate_config_entries_are_rejected(evidence, field):
    config, data, plan = evidence
    config[field].append(config[field][0])
    data["config"] = deepcopy(config)
    with pytest.raises(validator.ValidationError, match="Duplicate configured"):
        validator.validate_payload(config, data, plan)


def test_summary_matches_original_schema_and_uses_sample_standard_deviation(evidence):
    config, data, plan = evidence
    summary = validator.summarize(data)
    assert len(summary) == 24
    for row in summary:
        assert set(row) == {"dataset", "task", "method", "metric", "mean", "std", "n", "primary"}
        assert row["n"] == 3
        assert row["mean"] == pytest.approx(0.6)
        assert row["std"] == pytest.approx(0.1)
        assert row["metric"] == ("macro_f1" if row["dataset"] == "fixture_class" else "rmse")
        assert row["primary"] is (row["task"] == "primary")


@pytest.fixture
def runtime(tmp_path):
    names = ["src/jepa_forge/implementation.py", "configs/expanded.json", "configs/benchmark.json", "scripts/run_experiments.py"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# test runtime\n")
    hashes = {name: validator.sha256(tmp_path / name) for name in names}
    return tmp_path, tmp_path / "configs/expanded.json", hashes


def test_current_runtime_inventory_is_exact_and_excludes_report_code(runtime):
    root, config, hashes = runtime
    hashes["scripts/build_report.py"] = "historical report hash is not runtime"
    verified = validator.validate_sources(root, config, hashes)
    assert len(verified) == 4


@pytest.mark.parametrize("change", ["modified", "missing_hash", "new_source", "obsolete_source"])
def test_runtime_changes_or_incomplete_inventory_fail(runtime, change):
    root, config, hashes = runtime
    if change == "modified":
        (root / "src/jepa_forge/implementation.py").write_text("# changed\n")
    elif change == "missing_hash":
        hashes.pop("src/jepa_forge/implementation.py")
    elif change == "new_source":
        (root / "src/jepa_forge/new.py").write_text("# new\n")
    else:
        hashes["src/jepa_forge/deleted.py"] = "obsolete"
    with pytest.raises(validator.ValidationError, match="source"):
        validator.validate_sources(root, config, hashes)


def test_cli_writes_failed_status_without_success_summary(tmp_path):
    output, summary = tmp_path / "validation.json", tmp_path / "summary.json"
    assert validator.main(["--config", str(tmp_path / "missing.json"), "--output", str(output), "--summary", str(summary)]) == 1
    assert json.loads(output.read_text())["status"] == "failed"
    assert not summary.exists()


def report_module():
    spec = importlib.util.spec_from_file_location(
        "expanded_report", Path(__file__).resolve().parents[1] / "scripts/build_expanded_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_result_gzip_is_deterministic_and_preserves_exact_json_bytes(tmp_path):
    report = report_module()
    raw = b'{"status": "complete", "unicode": "\\u03b1"}\n'
    source, published = tmp_path / "raw.json", tmp_path / "results.json.gz"
    source.write_bytes(raw)
    first = report.publish_evidence(source, published)
    compressed = published.read_bytes()
    assert compressed[4:8] == b"\x00\x00\x00\x00"
    assert gzip.decompress(compressed) == raw
    assert report.result_bytes(published) == raw
    # Rebuilding from the published file itself must preserve its byte identity.
    second = report.publish_evidence(published, published)
    assert published.read_bytes() == compressed
    assert first == second
    assert first["uncompressed_sha256"] == validator.sha256(source)


@pytest.mark.parametrize("issue", ["partial", "missing_run", "cpu"])
def test_report_rejects_partial_incomplete_or_cpu_evidence(evidence, issue):
    config, data, plan = evidence
    report = report_module()
    if issue == "partial":
        data["status"] = "running"
    elif issue == "missing_run":
        data["runs"].pop()
    else:
        data["runs"][0]["training"]["device_evidence"]["loss_device"] = "cpu"
    with pytest.raises(ValueError):
        report.check_matrix(data, 2, 12)
