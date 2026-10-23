"""H-B-105 Small MIL CLI v1.0.0 (2026-10-11). No legacy guard chain or time gates."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    from training.arms import lifecycle as life
    from training.arms import mil_small as mil
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("validate", "smoke", "train"))
    ap.add_argument("--spec", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--resume")
    ap.add_argument("--note")
    ap.add_argument("--evidence-root")
    a = ap.parse_args(argv)
    try:
        spec = life.ArmSpec(Path(a.spec).read_bytes())
        if a.command == "validate":
            life.validate_stage(spec, a.stage)
            _, train, val, _, _ = mil.validate_data(spec, a.stage, a.corpus)
            result = {"status": "VALIDATED_NO_EXECUTION", "train": len(train), "val": len(val), "spec_sha256": spec.sha256}
        else:
            mil.need(a.out is not None, "output required")
            result = mil.run(spec, a.stage, a.corpus, a.out, mode="full" if a.command == "train" else "smoke",
                             device=a.device, workers=a.workers, resume=a.resume,
                             note=mil.read(a.note) if a.note else None, evidence_root=a.evidence_root)
            if a.command == "smoke":
                command = [sys.executable, "-B", "-m", "training.runners.mil_small_arms", *(sys.argv[1:] if argv is None else argv)]
                mil.record_smoke(spec, a.out, result, command)
            result = {k: result[k] for k in ("status", "spec_sha256", "results", "elapsed_seconds")}
        print(life.canonical_bytes(result).decode())
        return 0
    except (life.ContractError, OSError, ValueError) as error:
        print(life.canonical_bytes({"status": "REJECTED", "error_type": type(error).__name__}).decode())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
