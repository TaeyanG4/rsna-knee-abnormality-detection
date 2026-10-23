"""Explicit execution and input receipts for our Raptor models.

This module has no training, downloads, file writes or backend changes at import.
Equal receipts make a device comparison interpretable; they do not promise
bitwise equality between different CUDA architectures or library versions.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def native_bf16_supported(torch_module, device="cuda"):
    """Reject emulated BF16 on T4; older Torch falls back to native CUDA capability."""
    if not torch_module.cuda.is_available():
        return False
    with torch_module.cuda.device(device):
        try:
            return bool(torch_module.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:  # Torch versions without the including_emulation parameter.
            major, _ = torch_module.cuda.get_device_capability(device)
            cuda_version = torch_module.version.cuda or "0"
            return major >= 8 and int(cuda_version.split(".")[0]) >= 11


def runtime_snapshot(torch_module=None):
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    out = {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    for name in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        if hasattr(torch.backends.cuda.matmul, name):
            out[name] = bool(getattr(torch.backends.cuda.matmul, name))
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        if hasattr(torch.backends.cuda, name):
            out[name] = bool(getattr(torch.backends.cuda, name)())
    return out


def configure_runtime(amp, device, policy="strict", torch_module=None):
    """Apply and return the requested/effective policy before moving a model to CUDA.

    Strict disables algorithm benchmarking and rejects nondeterministic operations.
    Throughput permits kernel selection but still disables TF32 explicitly. Neither
    policy changes the requested AMP type or retries failures in another precision.
    CPU checks require --amp none, rather than silently converting a CUDA request.
    """
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if amp not in {"fp16", "bf16", "none"}:
        raise ValueError(f"unknown AMP mode: {amp}")
    if policy not in {"strict", "throughput"}:
        raise ValueError(f"unknown runtime policy: {policy}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)
    cuda = device.split(":", 1)[0] == "cuda"
    if not cuda and device != "cpu":
        raise ValueError(f"unsupported device: {device}")
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not cuda and amp != "none":
        raise ValueError("CPU reference inference requires explicit --amp none")
    if policy == "strict":
        allowed = {":4096:8", ":16:8"}
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in allowed:
            if cuda and torch.cuda.is_initialized():
                raise RuntimeError("configure strict runtime before initializing CUDA, or set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if cuda and amp == "bf16" and not native_bf16_supported(torch, device):
        raise ValueError("requested BF16 is not natively supported on this GPU; select --amp fp16 or none explicitly")
    torch.backends.cudnn.benchmark = policy == "throughput"
    torch.backends.cudnn.deterministic = policy == "strict"
    torch.use_deterministic_algorithms(policy == "strict")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = {"requested_amp": amp, "effective_amp": amp, "device": device, "policy": policy,
           "precision_fallback_allowed": False, "backends": runtime_snapshot(torch)}
    if cuda:
        out.update(compute_capability=list(torch.cuda.get_device_capability(device)),
                   native_bf16_supported=native_bf16_supported(torch, device))
    return out


def checkpoint_norm(meta):
    """Old public checkpoints omit norm; preserve their ImageNet contract explicitly."""
    norm = meta.get("norm", "imagenet")
    if norm not in {"imagenet", "none"}:
        raise ValueError(f"unsupported checkpoint normalization: {norm!r}")
    return norm, "checkpoint" if "norm" in meta else "legacy_imagenet_assumption"


def normalize_input(x, norm):
    if norm not in {"imagenet", "none"}:
        raise ValueError(f"unsupported normalization: {norm!r}")
    x = np.asarray(x, dtype=np.float32)
    if norm == "imagenet":
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite model input")
    return x


def tensor_sha256(x):
    """Hash shape, little-endian float32 dtype and contiguous bytes, before transfer."""
    arr = np.ascontiguousarray(x, dtype="<f4")
    header = json.dumps({"shape": list(arr.shape), "dtype": "<f4"}, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(header + b"\n")
    digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def prepare_input(corpus, uid, k_eval, res, norm, window_policy="legacy"):
    # Import a passive helper only when called, to avoid a common/runtime import cycle.
    from raptor_common import select_centers, windows_from
    mask = corpus.mask(uid)
    centers = select_centers(mask, corpus.depth, k_eval, corpus.spec, policy=window_policy)
    return normalize_input(windows_from(corpus.volume(uid), centers, res), norm)


def predict(model, corpus, ids, k_eval, res, amp, device, norm="imagenet", batch=1, log=print,
            window_policy="legacy"):
    """Strict batched inference; exceptions/nonfinite logits stop before any CSV write.

    Call configure_runtime first. Input hashes cover each actual normalized model
    tensor. Their aggregate includes ordered study IDs, so missing/reordered studies
    cannot accidentally compare as the same input set.
    """
    import torch
    from raptor_common import LAB
    if batch < 1:
        raise ValueError("batch must be positive")
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("inference requires a nonempty, unique study ID list")
    if amp not in {"fp16", "bf16", "none"}:
        raise ValueError(f"unknown AMP mode: {amp}")
    device_type = str(device).split(":", 1)[0]
    if device_type == "cpu" and amp != "none":
        raise ValueError("CPU reference inference requires explicit --amp none")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": torch.float32}[amp]
    model.eval().to(device)
    probs = np.zeros((len(ids), len(LAB)), np.float32)
    input_hashes = []
    aggregate = hashlib.sha256()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(ids), batch):
            uids = ids[start:start + batch]
            rows = []
            for uid in uids:
                try:
                    x = prepare_input(corpus, uid, k_eval, res, norm, window_policy)
                except Exception as exc:
                    raise RuntimeError(f"input preparation failed for {uid}: {exc}") from exc
                digest = tensor_sha256(x)
                record = {"uid": str(uid), "sha256": digest}
                input_hashes.append(record)
                aggregate.update((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
                rows.append(x)
            xt = torch.from_numpy(np.stack(rows)).to(device)
            try:
                with torch.autocast(device_type, dtype=dtype, enabled=amp != "none"):
                    logits = model(xt)
                if tuple(logits.shape) != (len(uids), len(LAB)):
                    raise RuntimeError(f"expected {(len(uids), len(LAB))} logits; got {tuple(logits.shape)}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite logits")
                p = torch.sigmoid(logits.float()).cpu().numpy()
                if not np.isfinite(p).all():
                    raise RuntimeError("nonfinite probabilities")
            except RuntimeError as exc:
                raise RuntimeError(f"inference failed for batch {uids} with amp={amp}; no precision fallback: {exc}") from exc
            stop = start + len(uids)
            probs[start:stop] = p
            if stop == len(ids) or start // 50 != stop // 50:
                log(f"  {stop}/{len(ids)} studies, {time.perf_counter() - t0:.0f}s")
    seconds = time.perf_counter() - t0
    return probs, {"seconds": round(seconds, 3), "per_study_s": round(seconds / len(ids), 4),
                   "fp32_fallbacks": 0, "all_finite": True, "batch_size": batch,
                   "window_policy": window_policy, "norm": norm,
                   "input_hash_scheme": "normalized-float32-shape-v1",
                   "ordered_inputs_sha256": aggregate.hexdigest(), "input_tensor_hashes": input_hashes}


# Serving adapters may use the descriptive name; keep predict for existing callers.
predict_batches = predict
