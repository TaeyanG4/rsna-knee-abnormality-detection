"""H-B-106 coverage CLI v1.0.0: separate preset, no shared runner mutation."""
import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    from training.arms import lifecycle as life, mil, mil_coverage as coverage
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("validate", "smoke", "train"))
    for name in ("spec", "stage", "corpus"):
        ap.add_argument("--"+name, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    for name in ("out", "note", "evidence-root"):
        ap.add_argument("--"+name)
    a = ap.parse_args(argv)
    spec = life.ArmSpec(Path(a.spec).read_bytes())
    if a.command == "validate":
        life.validate_stage(spec, a.stage)
        _, train, val, _, _ = coverage.validate_data(spec, a.stage, a.corpus)
        result = dict(status="VALIDATED_NO_EXECUTION", train=len(train), val=len(val), spec_sha256=spec.sha256)
    else:
        mil.need(a.out is not None, "output required")
        result = coverage.run(spec, a.stage, a.corpus, a.out, mode="full" if a.command=="train" else "smoke",
            device=a.device, workers=a.workers, note=mil.read(a.note) if a.note else None, evidence_root=a.evidence_root)
        if a.command == "smoke":
            mil.record_smoke(spec, a.out, result, [sys.executable,"-B","-m","training.runners.mil_coverage",*(sys.argv[1:] if argv is None else argv)])
        result = {k:result[k] for k in ("status","spec_sha256","results","elapsed_seconds")}
    print(life.canonical_bytes(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
