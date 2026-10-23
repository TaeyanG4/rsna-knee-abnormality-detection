"""Shared MIL-S arm-hook CLI v2.0.0 (2026-10-13). Offline; two CPU threads.

  python -B -m training.runners.mil_screen_hooks_v2 {validate,smoke,train}
      --arm package.module:ARM --spec S --stage D --corpus C
      [--device cpu|cuda] [--workers 2] [--out O] [--note N --evidence-root E] [--resume R]

The arm module must be a bound source of the spec (except the identity arm).
Full CUDA fits still require an independently reviewed PASS note.
"""
import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import cv2
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    from training.arms import mil_screen_hooks_v2 as mil
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("validate", "smoke", "train"))
    for key in ("arm", "spec", "stage", "corpus"):
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for key in ("out", "resume", "note", "evidence-root"):
        ap.add_argument("--" + key)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    spec = mil.life.ArmSpec(Path(a.spec).read_bytes())
    arm = mil.load_arm(a.arm, spec)
    if a.command == "validate":
        mil.life.validate_stage(spec, a.stage)
        _, train, val, _, _ = mil.validate_data(spec, a.stage, a.corpus, engineering=True)
        result = dict(status="VALIDATED_NO_EXECUTION", train=len(train), val=len(val), spec_sha256=spec.sha256)
    else:
        mil.need(a.out is not None, "output required")
        result = mil.run(spec, a.stage, a.corpus, a.out, arm=arm, mode="full" if a.command == "train" else "smoke",
            device=a.device, workers=a.workers, resume=a.resume,
            note=mil.read(a.note) if a.note else None, evidence_root=a.evidence_root)
        if a.command == "smoke":
            mil.record_smoke(spec, a.out, result, [sys.executable, "-B", "-m", "training.runners.mil_screen_hooks_v2",
                *(sys.argv[1:] if argv is None else argv)])
        result = {k: result[k] for k in ("status", "spec_sha256", "results", "elapsed_seconds")}
    print(mil.life.canonical_bytes(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
