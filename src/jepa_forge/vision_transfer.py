"""Frozen official-model transfer on observed pixels, separate from scratch compute."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gzip
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.request import urlopen
import zipfile

import joblib
import numpy as np
import torch
from torch.nn import functional as F

from .compiler import _array_hash
from .datasets import load_dataset, make_splits
from .evaluation import _metrics
from .extension import write_json, select_extension, evaluate_extension
from .model import _resolve_device, _synchronize
from .raw_sensors import sha256
from .schema import RawDataset
from .selection import propose_candidates, fit_validation_probe, file_hash


IJEPA = {"repository": "facebook/ijepa_vith14_1k", "revision": "f157467ea509bc356ff9f61fd3c0d840eec5e04e",
         "checkpoint_sha256":"ad6c1f6e318ace5633bea4496bc5c76248f6dc59c9a0d8a010cf31ed9dee2da8",
         "pretraining": "ImageNet-1K, 300 epochs", "family": "I-JEPA ViT-H/14", "license": "CC-BY-NC-4.0"}
VJEPA = {"repository": "facebookresearch/jepa", "revision": "51c59d518fc63c08464af6de585f78ac0c7ed4d5",
         "checkpoint_sha256":"0cd3bfce97d04891a31eac70d573ef94f423d858a6b0dc0b3ddf5f8d00b33fe4",
         "checkpoint_url": "https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar",
         "pretraining": "VideoMix2M, 90000 iterations", "family": "V-JEPA ViT-L/16", "license": "CC-BY-NC-4.0"}


def moving_shapes(seed=9026, groups=60, clips_per_group=4):
    """Independent simulated entities with four motion laws; no natural-video claim.

    Entity identity controls appearance and initial position; each entity has a
    fixed law. Clips are consecutive segments of its trajectory and must stay
    together in train/validation/test. Pixel values are the only model inputs.
    """
    rng = np.random.default_rng(seed)
    videos, labels, ids, intervals, entities = [], [], [], [], []
    T, H = 8, 16
    yy, xx = np.mgrid[:H, :H]
    for group in range(groups):
        label = group % 4
        phase, speed = rng.uniform(0, 2*np.pi), rng.uniform(.07, .15)
        base_y, base_x = rng.uniform(5, 11, size=2)
        radius, intensity = rng.uniform(1.2, 2.2), rng.uniform(.6, 1.)
        entity_frames = []
        for t in range(T * clips_per_group):
            if label == 0:
                x, y = 2 + ((base_x + speed*t*4) % 12), base_y
            elif label == 1:
                x, y = base_x, 2 + ((base_y + speed*t*4) % 12)
            elif label == 2:
                x, y = 8 + 4*np.cos(phase + speed*t), 8 + 4*np.sin(phase + speed*t)
            else:
                x, y = 8 + 4*np.sin(phase + speed*t), base_y
            frame = intensity * np.exp(-((xx-x)**2 + (yy-y)**2) / (2*radius**2))
            frame += rng.normal(0, .015, (H,H))
            entity_frames.append(np.round(np.clip(frame, 0, 1) * 255).astype(np.uint8))
        for clip in range(clips_per_group):
            videos.append(np.asarray(entity_frames[clip*T:(clip+1)*T]))
            labels.append(label); ids.append(group); intervals.append([clip*T,(clip+1)*T-1])
        entities.append({"group":group,"law":label,"phase":phase,"speed":speed,"radius":radius,"intensity":intensity})
    values = np.asarray(videos)
    names = [f"t{t}_r{r}_c{c}" for t in range(T) for r in range(H) for c in range(H)]
    return RawDataset("synthetic_motion_video", "sequence", values.reshape(len(values), -1).astype(float)/255,
                      names, np.asarray(labels), np.asarray(ids), np.asarray(intervals),
                      {n:"pixel" for n in names},
                      {"source":"Deterministic synthetic moving Gaussian shapes", "license":"CC0-1.0",
                       "task_type":"classification", "input_shape":[T,H*H], "video_shape":[T,H,H],
                       "generator_seed":seed, "entity_parameters":entities,
                       "label_names":["horizontal periodic wrap", "vertical periodic wrap", "circular", "horizontal sinusoid"],
                       "split_policy":"whole simulated entities", "limitations":["Toy motion-law discrimination; no real-world video or control claim.",
                       "Horizontal/vertical laws wrap at image boundaries; this is a kinematic generator, not a physical simulator."]})


def vision_dataset(name, sample_count=600, seed=9026):
    if name == "synthetic_motion_video":
        return moving_shapes(seed)
    data = deepcopy(load_dataset(name, seed=2026))
    indices = np.sort(np.random.default_rng(seed).choice(len(data.X), sample_count, replace=False))
    data.X, data.y = data.X[indices], data.y[indices]
    data.metadata.update(transfer_subset_source_rows=indices.tolist(), transfer_sampling_seed=seed)
    return data


def visible_pixels(dataset, task):
    """Mask BEFORE resizing so interpolation cannot leak hidden source pixels."""
    masked = np.zeros_like(dataset.X)
    masked[:, task.context] = dataset.X[:, task.context]
    if dataset.name == "digits":
        masked = masked / 16.
    shape = dataset.metadata.get("video_shape", dataset.metadata["input_shape"])
    return np.round(np.clip(masked.reshape(len(masked), *shape), 0, 1) * 255).astype(np.uint8)


@contextmanager
def isolated_import(source):
    saved = {k:v for k,v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(source))
    try:
        yield
    finally:
        sys.path.remove(str(source))
        for key in list(sys.modules):
            if key == "src" or key.startswith("src."):
                del sys.modules[key]
        sys.modules.update(saved)


def _download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(".partial")
        with urlopen(url, timeout=120) as inp, partial.open("wb") as out:
            shutil.copyfileobj(inp, out, length=8*1024*1024)
        partial.replace(path)
    return path


def load_official(name, device="mps", image_cache=".cache/pretrained", video_cache=".cache/vjepa"):
    _resolve_device(device)
    if name == "ijepa":
        from transformers import AutoModel, AutoImageProcessor
        from huggingface_hub import hf_hub_download
        options = {"revision":IJEPA["revision"],"cache_dir":image_cache,"trust_remote_code":False}
        weight = Path(hf_hub_download(IJEPA["repository"], "model.safetensors", **{k:v for k,v in options.items() if k != "trust_remote_code"}))
        if sha256(weight) != IJEPA["checkpoint_sha256"]:
            raise ValueError("I-JEPA checkpoint checksum mismatch")
        model = AutoModel.from_pretrained(IJEPA["repository"], **options).eval().requires_grad_(False).to(device)
        processor = AutoImageProcessor.from_pretrained(IJEPA["repository"], **options, use_fast=False)
        provenance = {**IJEPA,"implementation":"transformers.IJepaModel with official Meta-distributed weights and processor",
                      "processor":processor.to_dict()}
    else:
        cache = Path(video_cache)
        revision = VJEPA["revision"]
        archive = _download(f"https://codeload.github.com/facebookresearch/jepa/zip/{revision}", cache / "sources" / f"jepa-{revision}.zip")
        source = archive.with_suffix("")
        if not source.exists():
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    relative = Path(member.filename)
                    if relative.is_absolute() or ".." in relative.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("Unsafe official source archive path")
                z.extractall(source.parent)
        with isolated_import(source):
            module = importlib.import_module("src.models.vision_transformer")
            model = module.vit_large(img_size=224, patch_size=16, num_frames=16, tubelet_size=2, uniform_power=True)
        weight = _download(VJEPA["checkpoint_url"], cache / "checkpoints" / "vitl16.pth.tar")
        if sha256(weight) != VJEPA["checkpoint_sha256"]:
            raise ValueError("V-JEPA checkpoint checksum mismatch")
        checkpoint = torch.load(weight, map_location="cpu", weights_only=True, mmap=True)
        state = {k.removeprefix("module.").removeprefix("backbone."):v for k,v in checkpoint["target_encoder"].items()}
        model.load_state_dict(state, strict=True)
        del checkpoint, state
        model = model.eval().requires_grad_(False).to(device)
        processor = None
        provenance = {**VJEPA, "implementation":"native official source; target_encoder; strict state loading",
                      "source_archive_sha256":sha256(archive), "inference_frames":8,
                      "preprocessing":"masked grayscale replicated to RGB; bilinear resize to 224 without cropping; ImageNet channel normalization",
                      "temporal_note":"8-frame inference with official positional interpolation; pretrained configuration uses 16 frames"}
    provenance.update(checkpoint_sha256=sha256(weight), checkpoint_bytes=weight.stat().st_size,
                      device=str(next(model.parameters()).device), precision="float32",
                      frozen=True, pooling="mean of all output patch tokens", parameters=sum(p.numel() for p in model.parameters()))
    return model, processor, provenance


def encode_official(model, processor, pixels, name, batch_size):
    device = next(model.parameters()).device
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            batch = pixels[start:start+batch_size]
            if name == "ijepa":
                rgb = np.repeat(batch[...,None],3,axis=-1)
                x = processor(images=list(rgb), return_tensors="pt")["pixel_values"].to(device)
                tokens = model(pixel_values=x).last_hidden_state
            else:
                n,t,h,w = batch.shape
                x = torch.as_tensor(batch.copy(),dtype=torch.float32,device=device).reshape(n*t,1,h,w)/255
                x = F.interpolate(x, size=(224,224),mode="bilinear",align_corners=False,antialias=True).expand(-1,3,-1,-1)
                mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
                std = x.new_tensor([.229,.224,.225])[None,:,None,None]
                x = ((x-mean)/std).reshape(n,t,3,224,224).permute(0,2,1,3,4).contiguous()
                tokens = model(x)
            outputs.append(tokens.mean(dim=1).float().cpu().numpy())
    z = np.concatenate(outputs)
    if not np.isfinite(z).all():
        raise ValueError("Official encoder produced nonfinite embeddings")
    return z, {"parameter":str(next(model.parameters()).device),"input":str(x.device),"output":str(tokens.device),
               "input_shape":list(x.shape), "token_shape":list(tokens.shape),"cpu_fallback":False}


def run_vision(config_path, output_path, image_cache=".cache/pretrained", video_cache=".cache/vjepa"):
    torch.set_num_threads(2)
    cfg = json.loads(Path(config_path).read_text())
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version
    protocol = {"config":cfg,"source_sha256":{str(Path("src/jepa_forge") / p.name):sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "versions":{k:version(k) for k in ["torch","torchvision","transformers","numpy","scikit-learn"]}}
    if (output/"protocol.json").exists() and json.loads((output/"protocol.json").read_text()) != protocol:
        raise ValueError("Vision source/config changed; use a new directory")
    write_json(output/"protocol.json",protocol)
    records, model_provenance, inventories = [], {}, []
    for name in cfg["datasets"]:
        dataset = vision_dataset(name,cfg["image_sample_count"],cfg["data_seed"])
        tasks = propose_candidates(dataset)
        family = "vjepa" if name == "synthetic_motion_video" else "ijepa"
        directory = output / name
        directory.mkdir(exist_ok=True)
        embeddings = {}
        missing = [t for t in tasks if not (directory/f"{family}_{t.name}.npz").exists()]
        if missing:
            model, processor, provenance = load_official(family,cfg["device"],image_cache,video_cache)
            for task in missing:
                pixels = visible_pixels(dataset,task)
                start = time.perf_counter()
                z,evidence = encode_official(model,processor,pixels,family, cfg["image_batch_size"] if family=="ijepa" else cfg["video_batch_size"])
                _synchronize(next(model.parameters()).device)
                path = directory/f"{family}_{task.name}.npz"
                np.savez_compressed(path,embeddings=z)
                write_json(path.with_suffix(".json"),{"pixel_sha256":_array_hash(pixels),"embedding_sha256":_array_hash(z),
                           "file_sha256":sha256(path),"seconds":time.perf_counter()-start,"device_evidence":evidence,"provenance":provenance})
                print(f"ENCODED {name} {task.name}: {len(z)} official {family} embeddings",flush=True)
            del model, processor
            torch.mps.empty_cache()
        for task in tasks:
            path = directory/f"{family}_{task.name}.npz"
            meta = json.loads(path.with_suffix(".json").read_text())
            if sha256(path) != meta["file_sha256"] or _array_hash(visible_pixels(dataset,task)) != meta["pixel_sha256"]:
                raise ValueError("Official embedding cache mismatch")
            with np.load(path) as z: embeddings[task.name] = z["embeddings"].copy()
            model_provenance[f"{name}/{task.name}"] = meta
        inventories.append({"dataset":name,"rows":len(dataset.X),"X_sha256":_array_hash(dataset.X),"metadata":dataset.metadata})
        for split_seed in cfg["split_seeds"]:
            splits = make_splits(dataset,split_seed)
            opts = {"model_seed":7,"epochs_per_candidate":cfg["epochs_per_candidate"],"batch_size":128,"device":cfg["device"],
                    "neural_methods":["jepa","supervised"],"classical_methods":["linear","extra_trees"],"catboost_iterations":100,"split_seed":split_seed}
            sub = directory/str(split_seed)
            scratch = select_extension(dataset,splits,opts,sub)
            rows = np.concatenate([splits["train"],splits["val"]])
            dev_splits = {"train":np.arange(len(splits["train"])),"val":np.arange(len(splits["train"]),len(rows))}
            candidates=[]
            for task in tasks:
                estimator, probe = fit_validation_probe(embeddings[task.name][rows],dataset.y[rows],dev_splits,True,7)
                path = sub/f"{family}_{task.name}.joblib"
                joblib.dump(estimator,path)
                candidates.append({"task":task.name,"probe":probe,"probe_path":path.name,"probe_sha256":sha256(path)})
            chosen = max(range(len(candidates)),key=lambda i:candidates[i]["probe"]["score"])
            official = {"family":family,"candidates":candidates,"selected":chosen,"test_used_for_selection":False}
            write_json(sub/"official_lock.json",official)
            records.append({"dataset":name,"split_seed":split_seed,"scratch":scratch,"official":official,
                            "scratch_lock_sha256":sha256(sub/"lock.json"),"official_lock_sha256":sha256(sub/"official_lock.json")})
            print(f"LOCKED vision {name} split={split_seed}",flush=True)
    global_lock = [{k:v for k,v in r.items() if k not in {"scratch","official"}} for r in records]
    write_json(output/"all_selections_locked.json",global_lock)
    final=[]
    for record in records:
        name, split_seed = record["dataset"],record["split_seed"]
        d = vision_dataset(name,cfg["image_sample_count"],cfg["data_seed"])
        splits=make_splits(d,split_seed); sub=output/name/str(split_seed)
        result=evaluate_extension(d,splits,record["scratch"],sub,record["scratch_lock_sha256"],cfg["device"])
        if sha256(sub/"official_lock.json") != record["official_lock_sha256"]:
            raise ValueError("Official selection changed")
        official=record["official"]; row=official["candidates"][official["selected"]]
        if sha256(sub/row["probe_path"]) != row["probe_sha256"]:
            raise ValueError("Official probe changed")
        with np.load(output/name/f"{official['family']}_{row['task']}.npz") as z:
            prediction=joblib.load(sub/row["probe_path"]).predict(z["embeddings"])
        validation=_metrics(d.y[splits["val"]],prediction[splits["val"]],True)
        if validation != row["probe"]["validation"]:
            raise ValueError("Official validation replay mismatch")
        result["methods"].append({"method":official["family"]+"_official", "task":row["task"],"validation":validation,
                                  "test":_metrics(d.y[splits["test"]],prediction[splits["test"]],True)})
        np.savez_compressed(sub/"official_test_predictions.npz",indices=splits["test"],truth=d.y[splits["test"]],prediction=prediction[splits["test"]])
        final.append(result)
        print(f"TESTED vision {name} split={split_seed}",flush=True)
    payload={"schema_version":1,"status":"complete","protocol":protocol,"inventory":inventories,"model_provenance":model_provenance,
             "selections":records,"global_lock":global_lock,"global_lock_sha256":sha256(output/"all_selections_locked.json"),"final_evaluations":final}
    with gzip.GzipFile(output/"vision_transfer.json.gz","wb",mtime=0) as f: f.write(json.dumps(payload,sort_keys=True,allow_nan=False).encode())
    return payload
