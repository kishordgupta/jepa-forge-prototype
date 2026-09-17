"""Independent selection/budget checks, saved prediction replay, and CPU/GPU replay."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jepa_forge.compiler import compile_development_task
from jepa_forge.datasets import make_splits
from jepa_forge.evaluation import _metrics
from jepa_forge.extension import load_extension_dataset, score, write_json
from jepa_forge.extension_models import restore_neural, infer_neural
from jepa_forge.raw_sensors import sha256
from jepa_forge.schema import TaskSpec
from jepa_forge.selection import development_data


def validate_payload(payload, vision=False):
    assert payload["status"] == "complete"
    records = payload["selections"]
    expected = len(payload["protocol"]["config"]["datasets"])*len(payload["protocol"]["config"]["split_seeds"])
    assert len(records) == len(payload["final_evaluations"]) == expected
    seen, neural_checkpoints = set(),0
    counts = {"task_splits":len(records),"gpu_checkpoint_records":0,"matched_policy_budgets":0,"selected_evaluations":0}
    for record in records:
        lock = record["scratch"] if vision else record["lock"]
        key = record["dataset"],record["split_seed"]
        assert key not in seen
        seen.add(key)
        assert lock["test_used_for_selection"] is False
        budgets=lock["budget"]
        assert budgets["search_updates"] == budgets["fixed_updates"]
        assert budgets["search_probe_fits"] == budgets["fixed_probe_fits"]
        counts["matched_policy_budgets"] += 1
        if lock["split_groups"] is not None:
            groups=[set(v) for v in lock["split_groups"].values()]
            assert all(not(a&b) for i,a in enumerate(groups) for b in groups[i+1:])
        candidates=lock["candidates"]
        for method,index in lock["selected"].items():
            if method == "jepa_fixed_short":
                expected_index=next(i for i,r in enumerate(candidates) if r["method"]=="jepa" and r["task_index"]==0)
            else:
                eligible=[i for i,r in enumerate(candidates) if r["method"]==method]
                expected_index=max(eligible,key=lambda i:score(candidates[i]["validation"],lock["classification"]))
            assert index==expected_index
            counts["selected_evaluations"]+=1
        for candidate in candidates:
            assert len(candidate["task"]["context"])==budgets["context_features"]
            assert not(set(candidate["task"]["context"]) & set(candidate["task"]["target"]))
            if not lock["classification"]:
                assert candidate["task"]["target"]==candidates[0]["task"]["target"]
                assert max(candidate["task"]["context"]) < min(candidate["task"]["target"])
            if "checkpoint_path" in candidate:
                evidence=candidate["device_evidence"]
                assert all(evidence[k].startswith("mps") for k in ["requested","parameter","input","loss","gradient"])
                assert not evidence["cpu_fallback"]
                counts["gpu_checkpoint_records"] += 1
        final=next(r for r in payload["final_evaluations"] if (r["dataset"],r["split_seed"])==key)
        assert final["validation_replayed"] and final["test_evaluated_after_global_lock"]
        assert set(lock["selected"]) <= {r["method"] for r in final["methods"]}
        if vision:
            official=record["official"]
            assert not official["test_used_for_selection"]
            assert official["selected"]==max(range(len(official["candidates"])),key=lambda i:official["candidates"][i]["probe"]["score"])
    return counts


def validate(directory, vision=False, replay_device="mps"):
    directory=Path(directory)
    path=directory/("vision_transfer.json.gz" if vision else "extension_benchmark.json.gz")
    raw=gzip.decompress(path.read_bytes()); payload=json.loads(raw)
    counts=validate_payload(payload,vision)
    source_checks=0
    for source,expected in payload["protocol"]["source_sha256"].items():
        assert sha256(source)==expected, f"Executed source changed: {source}"
        source_checks+=1
    maximum_error, replayed, predictions = 0.,0,0
    torch.set_num_threads(2)
    for record in payload["selections"]:
        name,seed=record["dataset"],record["split_seed"]
        lock=record["scratch"] if vision else record["lock"]
        sub=directory/name/str(seed)
        assert sha256(sub/"lock.json")==record["scratch_lock_sha256" if vision else "lock_sha256"]
        if vision:
            from jepa_forge.vision_transfer import vision_dataset
            dataset=vision_dataset(name,payload["protocol"]["config"]["image_sample_count"],payload["protocol"]["config"]["data_seed"])
        else:
            dataset=load_extension_dataset(name,payload["config"]["data_seed"])
        dev=development_data(dataset,make_splits(dataset,seed))
        for candidate in lock["candidates"]:
            if "probe_path" in candidate:
                assert sha256(sub/candidate["probe_path"])==candidate["probe_sha256"]
            if "checkpoint_path" not in candidate:
                continue
            compiled=compile_development_task(dev.dataset,TaskSpec(**candidate["task"]),dev.splits)
            context=compiled.X[np.ix_(dev.splits["train"][:8],candidate["task"]["context"])]
            cpu,saved=restore_neural(compiled,sub/candidate["checkpoint_path"],candidate["checkpoint_sha256"],"cpu")
            a=infer_neural(cpu,saved["kind"],context)
            del cpu
            gpu,_=restore_neural(compiled,sub/candidate["checkpoint_path"],candidate["checkpoint_sha256"],replay_device)
            b=infer_neural(gpu,saved["kind"],context)
            del gpu
            np.testing.assert_allclose(a,b,atol=5e-4,rtol=5e-4)
            maximum_error=max(maximum_error,float(np.max(np.abs(a-b))))
            replayed+=1
        final=next(r for r in payload["final_evaluations"] if r["dataset"]==name and r["split_seed"]==seed)
        for method in final["methods"]:
            if method["method"].endswith("_official"):
                path=sub/"official_test_predictions.npz"
            else:
                path=sub/f"test_{method['method']}.npz"
                assert sha256(path)==method["prediction_sha256"]
            with np.load(path,allow_pickle=False) as arrays:
                metrics=_metrics(arrays["truth"],arrays["prediction"],lock["classification"])
            assert metrics==method["test"]
            predictions+=1
        print(f"REPLAYED {name} split={seed}",flush=True)
    report={"status":"passed","result_uncompressed_sha256":hashlib.sha256(raw).hexdigest(),"result_compressed_sha256":sha256(directory/("vision_transfer.json.gz" if vision else "extension_benchmark.json.gz")),
            **counts,"source_files_verified":source_checks,"cpu_gpu_checkpoint_replays":replayed,
            "cpu_gpu_max_absolute_error":maximum_error,"cpu_gpu_atol":5e-4,"cpu_gpu_rtol":5e-4,
            "replay_samples_per_checkpoint":8,"saved_test_predictions_recomputed":predictions,
            "validation_metrics_replayed_in_final_runner":True,
            "scope":"Actual local checkpoints and prediction files verified. The lightweight published archive retains results/hashes and this report, not the large checkpoint collection."}
    write_json(directory/"validation.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--directory",required=True);p.add_argument("--vision",action="store_true")
    a=p.parse_args();validate(a.directory,a.vision)
