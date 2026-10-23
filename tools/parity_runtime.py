"""Cross-environment numerical parity helpers for PyTorch.

Import this module (or paste it into the first notebook cell) BEFORE anything touches CUDA,
so CUBLAS_WORKSPACE_CONFIG takes effect.

    from parity_runtime import configure_parity, runtime_info, dump_activations
    info = configure_parity(amp="fp16")          # same call in every environment
    json.dump(info, open("runtime.json", "w"), indent=2)

CLI:
    python parity_runtime.py --out runtime.json                       # snapshot only
    python parity_runtime.py --configure --amp fp16 --out runtime.json
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import platform
import sys
import warnings

_LIBS = ("numpy", "scipy", "cv2", "PIL", "pydicom", "pylibjpeg", "gdcm", "openjpeg",
         "SimpleITK", "nibabel", "timm", "transformers", "albumentations", "monai",
         "torchvision", "pandas", "sklearn")
_ENV = ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "TIMM_FUSED_ATTN", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "NVIDIA_TF32_OVERRIDE")


def _capability(torch):
    if not torch.cuda.is_available():
        return None
    return list(torch.cuda.get_device_capability(0))


def _set_fp32_ieee(torch):
    """Disable TF32 with whichever API this torch version has; never mix the two APIs."""
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    if hasattr(matmul, "fp32_precision"):          # newer PyTorch API
        matmul.fp32_precision = "ieee"
        cudnn.fp32_precision = "ieee"
        return "fp32_precision=ieee"
    matmul.allow_tf32 = False                      # legacy API (default False since 1.12)
    cudnn.allow_tf32 = False                       # legacy API (default True: convs use TF32 on sm80+)
    torch.set_float32_matmul_precision("highest")
    return "allow_tf32=False"


def configure_parity(amp: str = "fp16", sdpa: str = "auto", strict: bool = True,
                     threads: int | None = None, seed: int | None = None) -> dict:
    """Apply one numerical policy. Call with identical arguments in every environment compared.

    amp:    "fp16" | "bf16" | "none" (fp32). bf16 is refused below compute capability 8.0
            because sm<80 has no native bf16 and torch may silently emulate it.
    sdpa:   "auto" -> math for fp32, mem-efficient for fp16/bf16 (flash off: flash needs sm80+,
            so leaving it on makes sm80+ and sm75 GPUs run different attention kernels).
            "math" | "efficient" | "flash" | "keep" (do not touch).
    strict: cudnn.deterministic + use_deterministic_algorithms(warn_only=True), benchmark off.
            This gives run-to-run repeatability on ONE GPU; it does not make two GPU models equal.
    threads: pin torch / OpenCV CPU threads (CPU interpolation results can depend on thread count).
    """
    import torch

    if amp not in {"fp16", "bf16", "none"}:
        raise ValueError(f"amp must be fp16|bf16|none, got {amp!r}")
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        warnings.warn("CUDA already initialized without CUBLAS_WORKSPACE_CONFIG; cuBLAS determinism not guaranteed")
    cap = _capability(torch)
    if amp == "bf16" and cap is not None and cap[0] < 8:
        raise ValueError(f"bf16 requested on sm{cap[0]}{cap[1]} (no native bf16); use fp16 or none")

    tf32 = _set_fp32_ieee(torch)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = strict
    torch.use_deterministic_algorithms(strict, warn_only=True)

    if sdpa == "auto":
        sdpa = "math" if amp == "none" else "efficient"
    if sdpa != "keep" and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(sdpa == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(sdpa in {"efficient", "flash"})
        torch.backends.cuda.enable_math_sdp(True)   # always keep a fallback

    if threads is not None:
        torch.set_num_threads(threads)
        try:
            import cv2
            cv2.setNumThreads(threads)
        except ImportError:
            pass

    if seed is not None:
        import random
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        torch.manual_seed(seed)

    info = runtime_info()
    info["policy"] = {"amp": amp, "sdpa": sdpa, "strict": strict, "threads": threads,
                      "seed": seed, "tf32_api": tf32}
    return info


def runtime_info() -> dict:
    """Snapshot of everything that can change numerics. Diff two snapshots before comparing outputs."""
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "env": {k: os.environ.get(k) for k in _ENV}}
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info
    b = torch.backends
    info.update(torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=b.cudnn.version() if b.cudnn.is_available() else None,
                cuda_available=torch.cuda.is_available(), torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
                cudnn_benchmark=b.cudnn.benchmark, cudnn_deterministic=b.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                float32_matmul_precision=torch.get_float32_matmul_precision())
    for key, getter in (("matmul_tf32", lambda: b.cuda.matmul.allow_tf32),
                        ("cudnn_tf32", lambda: b.cudnn.allow_tf32),
                        ("matmul_fp32_precision", lambda: b.cuda.matmul.fp32_precision),
                        ("cudnn_fp32_precision", lambda: b.cudnn.fp32_precision),
                        ("fp16_reduced_precision_reduction",
                         lambda: b.cuda.matmul.allow_fp16_reduced_precision_reduction),
                        ("flash_sdp", b.cuda.flash_sdp_enabled),
                        ("mem_efficient_sdp", b.cuda.mem_efficient_sdp_enabled),
                        ("math_sdp", b.cuda.math_sdp_enabled)):
        try:
            info[key] = getter()
        except Exception:
            pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                    sm_count=p.multi_processor_count, gpu_mem_gb=round(p.total_memory / 2**30, 2),
                    gpu_count=torch.cuda.device_count(), arch_list=torch.cuda.get_arch_list())
    libs = {}
    for name in _LIBS:
        try:
            mod = __import__(name)
            libs[name] = str(getattr(mod, "__version__", getattr(mod, "GDCM_VERSION", "present")))
        except Exception:
            pass
    try:
        import cv2
        libs["cv2_threads"] = cv2.getNumThreads()
    except Exception:
        pass
    info["libs"] = libs
    return info


def diff_runtime(a: dict, b: dict, ignore=("gpu", "capability", "sm_count", "gpu_mem_gb", "platform",
                                           "processor", "cpu_count", "arch_list")) -> dict:
    """Keys whose values differ between two runtime_info() snapshots (hardware identity ignored)."""
    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{prefix}{k}."))
            else:
                out[f"{prefix}{k}"] = v
        return out
    fa, fb = flat(a), flat(b)
    keys = sorted(set(fa) | set(fb))
    return {k: [fa.get(k), fb.get(k)] for k in keys
            if k.split(".")[0] not in ignore and fa.get(k) != fb.get(k)}


def dump_activations(model, inputs, path, names=None, depth=2):
    """Run one forward pass and save module outputs to an .npz for first-divergence search.

    names: exact module names to capture; default = modules whose name has < `depth` dots.
    Compare two dumps with:  python compare_arrays.py a.npz b.npz
    Keys are prefixed with the call order so the first diverging layer is easy to read.
    """
    import numpy as np
    import torch

    captured, hooks, order = {}, [], [0]

    def hook(name):
        def fn(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, dict):
                t = next((v for v in t.values() if torch.is_tensor(v)), None)
            if torch.is_tensor(t):
                captured[f"{order[0]:04d}:{name}"] = t.detach().float().cpu().numpy()
                order[0] += 1
        return fn

    for name, module in model.named_modules():
        if not name:
            continue
        if (names is not None and name in names) or (names is None and name.count(".") < depth):
            hooks.append(module.register_forward_hook(hook(name)))
    try:
        with torch.inference_mode():
            out = model(*inputs) if isinstance(inputs, (tuple, list)) else model(inputs)
    finally:
        for h in hooks:
            h.remove()
    t = out[0] if isinstance(out, (tuple, list)) else out
    if torch.is_tensor(t):
        captured[f"{order[0]:04d}:__output__"] = t.detach().float().cpu().numpy()
    np.savez_compressed(path, **captured)
    return list(captured)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configure", action="store_true", help="apply configure_parity before snapshot")
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"])
    ap.add_argument("--sdpa", default="auto", choices=["auto", "math", "efficient", "flash", "keep"])
    ap.add_argument("--no-strict", action="store_true")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--diff", nargs=2, metavar=("A_JSON", "B_JSON"), help="diff two saved snapshots")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.diff:
        a, b = (json.load(open(p, encoding="utf-8")) for p in args.diff)
        result = diff_runtime(a, b)
    elif args.configure:
        result = configure_parity(args.amp, args.sdpa, not args.no_strict, args.threads)
    else:
        result = runtime_info()
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    if args.diff and result:
        sys.exit(1)


if __name__ == "__main__":
    main()
