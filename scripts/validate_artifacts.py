#!/usr/bin/env python3
"""Validate the completed benchmark's evidence without retraining."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jepa_forge.compiler import load_export

def main():
    directory = ROOT / "artifacts/benchmark"
    data = json.loads((directory / "results.json").read_text())
    assert data["status"] == "complete"
    assert len(data["runs"]) == 18 and len(data["task_audits"]) == 6
    expected = {(r["dataset"], r["task"], s) for r in data["task_audits"] for s in (7,17,27)}
    assert {(r["dataset"],r["task"],r["seed"]) for r in data["runs"]} == expected
    for task in data["task_audits"]:
        assert not task["report"]["errors"]
        load_export(directory / "compiled" / task["dataset"] / task["task"])
    for run in data["runs"]:
        evidence = run["training"]["device_evidence"]
        for key in ("online_parameter_device", "target_parameter_device", "context_batch_device", "target_batch_device", "loss_device", "online_gradient_device"):
            assert evidence[key] == "mps:0", (run["dataset"], key)
        assert not evidence["target_has_gradient"] and not evidence["target_requires_grad"]
        assert run["training"]["checkpoint_roundtrip"]["passed"]
        assert len(run["training"]["training_history"]) == 60
    # Source integrity for code which actually produced the measured experiments.
    runtime = {k:v for k,v in data["source_sha256"].items() if k.startswith("src/") or k in ("scripts/run_experiments.py", "configs/benchmark.json")}
    current_match = all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest() == v for k,v in runtime.items())
    snapshot_match = False
    snapshot = ROOT / "results/original_runtime.zip"
    if not current_match and snapshot.exists():
        with zipfile.ZipFile(snapshot) as archive:
            snapshot_match = all(hashlib.sha256(archive.read(k)).hexdigest() == v for k,v in runtime.items())
    assert current_match or snapshot_match, "Neither current source nor archived original runtime matches original measurements"
    junit = ET.parse(ROOT / "artifacts/pytest.xml").getroot()
    tests = sum(int(s.get("tests",0)) for s in junit.findall("testsuite"))
    failures = sum(int(s.get("failures",0)) + int(s.get("errors",0)) for s in junit.findall("testsuite"))
    assert failures == 0 and tests >= 44
    summary = {"status": "passed", "tests_passed": tests, "test_failures": failures,
               "completed_gpu_runs": 18, "audited_tasks": 6, "export_reload_checks": 6,
               "checkpoint_reload_checks": 18, "actual_device": "mps:0", "runtime_source_hashes_match": current_match,
               "archived_original_runtime_hashes_match": snapshot_match,
               "low_rank_warning_runs": sum(r["training"]["final_train_context_diagnostics"]["heuristic_collapse_flag"] for r in data["runs"])}
    (ROOT / "artifacts/validation.json").write_text(json.dumps(summary, indent=2)+"\n")
    index = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.name != "results.json":
            index[str(p.relative_to(ROOT))] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/artifact_index.json").write_text(json.dumps(index,indent=2)+"\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
