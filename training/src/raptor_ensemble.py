"""Passive, strict Raptor ensemble with one input preparation per study.

Checkpoint members must share the complete serving input contract. The module
does not change runtime settings, import training/acquisition entry points, or
open files at import. Explicit inference applies the requested runtime policy. Inference
returns each member's raw probability and their float32 arithmetic mean: no
ranking, model omission, neutral filling, or automatic precision retry.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

ENSEMBLE_VERSION = "raptor-shared-input-ensemble-v1"


@dataclass
class EnsembleMember:
    name: str
    model: object
    metadata: dict
    checkpoint_sha256: str | None = None
    checkpoint_path: str | None = None


class EnsembleInferenceError(RuntimeError):
    """Failure diagnostics are available without exposing partial predictions."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    """Validate the explicit manifest and every file hash before loading a model.

    Relative member paths are resolved against the manifest directory. Manifest
    data contains no executable source or runtime overrides.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ensemble manifest requires a nonempty members list")
    resolved, names, paths = [], set(), set()
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("each ensemble manifest member must be an object")
        name, digest = item.get("name"), item.get("sha256")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
            raise ValueError(f"invalid manifest member name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate manifest member name: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"member {name} requires an expected SHA256")
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"member {name} requires a checkpoint path")
        checkpoint = Path(raw_path)
        checkpoint = (path.parent / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
        if checkpoint in paths:
            raise ValueError(f"checkpoint path repeated in ensemble: {checkpoint}")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        observed = file_sha256(checkpoint)
        if observed != digest.lower():
            raise ValueError(f"checkpoint SHA256 mismatch for {name}: {observed} != {digest.lower()}")
        names.add(name); paths.add(checkpoint)
        resolved.append({"name": name, "path": str(checkpoint), "sha256": observed})
    return {"members": resolved, "manifest_sha256": file_sha256(path), "manifest_path": str(path)}


def load_members(manifest_path):
    """Explicitly load all verified models on CPU; no GPU allocation or execution."""
    from raptor_common import load_checkpoint
    manifest = read_manifest(manifest_path)
    members = []
    for item in manifest["members"]:
        model, metadata = load_checkpoint(Path(item["path"]))
        # Detect file changes during loading, rather than issuing a stale hash receipt.
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"checkpoint changed while loading: {item['name']}")
        model.eval().to("cpu")
        members.append(EnsembleMember(item["name"], model, metadata, item["sha256"], item["path"]))
    validate_members(members)
    return members, manifest


def serving_contract(metadata):
    from raptor_common import LAB
    from raptor_runtime import checkpoint_norm
    if not isinstance(metadata.get("contract"), str) or not metadata["contract"]:
        raise ValueError("each ensemble checkpoint must identify its corpus contract")
    if "lab" not in metadata or list(metadata["lab"]) != list(LAB):
        raise ValueError("each ensemble checkpoint must declare the exact ordered 12 target labels")
    for field in ("res", "k_eval"):
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"checkpoint requires positive integer {field}")
    norm, norm_source = checkpoint_norm(metadata)
    policy = metadata.get("window_policy", "legacy")
    if policy not in {"legacy", "global-valid", "within-slot-valid"}:
        raise ValueError(f"unknown checkpoint window policy: {policy!r}")
    return {"contract": metadata["contract"], "targets": list(LAB), "res": int(metadata["res"]),
            "k_eval": int(metadata["k_eval"]), "norm": norm, "window_policy": policy}, norm_source


def validate_members(members, corpus=None):
    """Refuse contract differences before pixel reads or model-device changes."""
    if not members:
        raise ValueError("ensemble requires at least one checkpoint")
    names = [m.name for m in members]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("ensemble requires nonempty unique member names")
    contract, _ = serving_contract(members[0].metadata)
    for member in members[1:]:
        actual, _ = serving_contract(member.metadata)
        if actual != contract:
            changed = [k for k in contract if actual[k] != contract[k]]
            raise ValueError(f"member {member.name} has a different input contract: {changed}")
    if corpus is not None and corpus.contract != contract["contract"]:
        raise ValueError(f"corpus contract {corpus.contract} != ensemble {contract['contract']}")
    return contract


def predict_ensemble(members, corpus, ids=None, *, amp="fp16", device="cuda", batch=1,
                     residency="sequential", runtime_policy="strict", log=print):
    """Return (mean[N,12], raw[M,N,12], receipt), sharing decoded inputs.

    resident keeps every model on the selected device during inference; callers
    must budget sum(weights) plus one model's activations. sequential holds all
    models on CPU and transfers one model per input batch, bounding device model
    residency to one at the cost of transfers. Both keep at most `batch` model
    inputs in host memory and release device model residency at the end. Input
    tensors and their order exactly follow raptor_runtime.prepare_input/predict.
    Omitted ids means corpus.ids. Runtime policy is applied before device placement;
    configure strict CUBLAS settings before any earlier CUDA initialization.
    """
    import torch
    from raptor_runtime import configure_runtime, prepare_input, tensor_sha256

    members = list(members)
    ids = list(map(str, corpus.ids if ids is None else ids))
    contract = validate_members(members, corpus)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("ensemble requires nonempty unique study IDs")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    if residency not in {"resident", "sequential"}:
        raise ValueError("residency must be resident or sequential")
    if amp not in {"fp16", "bf16", "none"}:
        raise ValueError(f"unknown AMP mode: {amp}")
    device = str(device)
    device_type = device.split(":", 1)[0]
    if device_type not in {"cpu", "cuda"} or (device_type == "cpu" and amp != "none"):
        raise ValueError("explicit cuda or CPU --amp none is required")
    if any(uid not in corpus for uid in ids):
        raise ValueError("requested study missing from the explicit corpus ID list")
    runtime = configure_runtime(amp, device, runtime_policy)
    device = runtime["device"]
    device_type = device.split(":", 1)[0]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": torch.float32}[amp]
    raw = np.empty((len(members), len(ids), len(contract["targets"])), dtype=np.float32)
    aggregate = hashlib.sha256()
    receipt = {"status": "RUNNING", "version": ENSEMBLE_VERSION, "residency": residency,
               "requested_amp": amp, "effective_amp": amp, "device": device, "batch_size": batch,
               "runtime": runtime, "window_policy": contract["window_policy"], "norm": contract["norm"],
               "input_contract": contract, "studies": len(ids), "member_count": len(members),
               "reduction": "numpy.mean(raw_probabilities,axis=0,dtype=float32)",
               "input_hash_scheme": "normalized-float32-shape-v1", "input_tensor_hashes": [],
               "input_preparations": 0, "completed_batches": 0, "failures": [],
               "fp32_fallbacks": 0, "neutral_fills": 0,
               "member_results": [{"name": member.name, "checkpoint_sha256": member.checkpoint_sha256,
                                   "checkpoint_path": member.checkpoint_path,
                                   "norm_source": serving_contract(member.metadata)[1],
                                   "completed_studies": 0, "status": "NOT_STARTED"} for member in members]}
    started = time.perf_counter()
    stage, active_member, active_ids = "model_placement", None, []
    try:
        # Ensure sequential mode does not inherit several already-resident GPU models.
        for member in members:
            active_member = member.name
            member.model.eval().to(device if residency == "resident" else "cpu")
        with torch.inference_mode():
            for start in range(0, len(ids), batch):
                active_ids = ids[start:start + batch]
                stage, active_member = "input_preparation", None
                rows = []
                for uid in active_ids:
                    x = prepare_input(corpus, uid, contract["k_eval"], contract["res"],
                                      contract["norm"], contract["window_policy"])
                    expected = (contract["k_eval"], 3, contract["res"], contract["res"])
                    if x.shape != expected or x.dtype != np.float32 or not np.isfinite(x).all():
                        raise ValueError(f"invalid prepared input for {uid}: {x.shape}, {x.dtype}")
                    record = {"uid": uid, "sha256": tensor_sha256(x)}
                    receipt["input_tensor_hashes"].append(record)
                    aggregate.update((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
                    receipt["input_preparations"] += 1
                    rows.append(x)
                # All members receive exactly the same input tensor, with no mutation.
                # Normal tensor version tracking detects in-place member mutations.
                # The stack is a copy: corpus buffers and host inputs stay immutable.
                with torch.inference_mode(False):
                    xt = torch.from_numpy(np.stack(rows)).to(device)
                for index, member in enumerate(members):
                    active_member = member.name
                    stage = "member_inference"
                    member_receipt = receipt["member_results"][index]
                    member_receipt["status"] = "RUNNING"
                    if residency == "sequential":
                        member.model.to(device)
                    try:
                        input_version = xt._version
                        with torch.autocast(device_type, dtype=dtype, enabled=amp != "none"):
                            logits = member.model(xt)
                        if xt._version != input_version:
                            raise RuntimeError("member mutated its shared input tensor")
                        if tuple(logits.shape) != (len(active_ids), len(contract["targets"])):
                            raise RuntimeError(f"invalid member logits shape: {tuple(logits.shape)}")
                        if not torch.isfinite(logits).all():
                            raise RuntimeError("nonfinite member logits")
                        p = torch.sigmoid(logits.float()).cpu().numpy()
                        if not np.isfinite(p).all():
                            raise RuntimeError("nonfinite member probabilities")
                        raw[index, start:start + len(active_ids)] = p
                        member_receipt["completed_studies"] += len(active_ids)
                        member_receipt["status"] = "COMPLETE" if member_receipt["completed_studies"] == len(ids) else "RUNNING"
                        del logits, p
                    finally:
                        if residency == "sequential":
                            member.model.to("cpu")
                receipt["completed_batches"] += 1
                del xt, rows
                if log is not None:
                    stop = start + len(active_ids)
                    if stop == len(ids) or start // 50 != stop // 50:
                        log(f"ensemble {stop}/{len(ids)} studies x {len(members)} members, {time.perf_counter()-started:.1f}s")
        stage, active_member = "aggregation", None
        if not np.isfinite(raw).all():
            raise RuntimeError("ensemble contains nonfinite member probabilities")
        mean = np.mean(raw, axis=0, dtype=np.float32)
        if not np.isfinite(mean).all():
            raise RuntimeError("nonfinite ensemble probability mean")
        input_digest = aggregate.hexdigest()
        for index, mr in enumerate(receipt["member_results"]):
            mr.update(ordered_inputs_sha256=input_digest,
                      input_tensor_hashes_ref="inference.input_tensor_hashes",
                      probabilities_sha256=tensor_sha256(raw[index]))
        receipt.update(status="COMPLETE", ordered_inputs_sha256=input_digest, all_finite=True,
                       seconds=round(time.perf_counter()-started, 3),
                       ensemble_probabilities_sha256=tensor_sha256(mean),
                       max_resident_models=len(members) if residency == "resident" else 1)
        return mean, raw, receipt
    except Exception as exc:
        failure = {"stage": stage, "member": active_member, "uids": active_ids,
                   "error": f"{type(exc).__name__}: {exc}"}
        if hasattr(exc, "receipt"):
            failure["source_receipt"] = exc.receipt
        receipt["failures"].append(failure)
        if active_member is not None:
            next(r for r in receipt["member_results"] if r["name"] == active_member)["status"] = "FAILED"
        receipt.update(status="FAILED", all_finite=False, seconds=round(time.perf_counter()-started, 3),
                       ordered_inputs_sha256=aggregate.hexdigest())
        raise EnsembleInferenceError(f"ensemble failed at {stage}: {exc}", receipt) from exc
    finally:
        # Preserve the primary failure if device cleanup itself cannot complete.
        cleanup_failures = []
        for member in members:
            try:
                member.model.to("cpu")
            except Exception as exc:
                cleanup_failures.append({"member": member.name, "error": f"{type(exc).__name__}: {exc}"})
        if cleanup_failures:
            receipt["cleanup_failures"] = cleanup_failures
