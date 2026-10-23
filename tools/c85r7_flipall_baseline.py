"""Authenticated CPU-only H-B-153 timing baseline, v2.0.0.

Changelog 2.0.0: bind v1.1.0 N3/N32 receipts and ref 57133175 ledger
records; derive hidden and R7 wall-time estimates without manual constants.
New file; there is no previous version to back up. Never calls Kaggle or CUDA.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import math
from pathlib import Path

VERSION = "2.0.0"
ROOT = Path(__file__).resolve().parents[1]
REF = "57133175"
IDENTITY = "EXP169-C85R7CODEC-1.1.0"
MANIFEST = "c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6"
RECEIPTS = {
    "visible_n3": (".local/scratch/slot-b/knee-exp169-c85r7codec-v1/c85r7-receipt.json",
                   "e0eb466f3b0d545a3f17142c75c9a6742e3e90c2aba5916267ec7909f3caa3f3", 3),
    "stress_n32": (".local/scratch/slot-b/knee-exp169-c85r7codec-v1-stress-n32/c85r7-receipt.json",
                   "da965da52265af350011932d62b95422710b6cbaf71a3ec2fefff72330f919c9", 32),
}
SUBMIT = "artifacts/experiments/SUBMISSION_SCREEN/submit_ledger.jsonl"
READINGS = "artifacts/evidence_ledger/readings.csv"
SUBMIT_RECORD_SHA256 = "54484278c1f7c89e023ee184b1d58130cd8e33531242a0aad9a136c2872c9c9a"
READING_RECORD_SHA256 = "e45be5e89498ef117195c6db062ddca5b42f7c1819f77d42f6eee4987e8aa68b"


def _need(condition, message):
    if not condition:
        raise ValueError("DO_NOT_SUBMIT: authenticated baseline: " + message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _validate_receipt(receipt, count):
    _need(receipt.get("version") == "1.1.0" and receipt.get("status") == "COMPLETE_C85R7"
          and receipt.get("release") == "C85R7+codecs", "older, partial or wrong release receipt")
    _need(receipt.get("studies") == count and receipt.get("findings") == 12, "wrong receipt cohort/target count")
    members = {"R", "S_R7", "EXP143v2", "EXP155", "Q_A5q"}
    _need(set(receipt.get("executed_members", [])) == members
          and set(receipt.get("intended_members", [])) == members, "incomplete members")
    _need(receipt.get("fallbacks") and all(v == 0 for v in receipt["fallbacks"].values()), "prediction fallback")
    for identity in (receipt.get("r7_identity", {}), receipt.get("pr7", {}).get("r7_identity", {})):
        features = identity.get("features", {})
        _need(identity.get("manifest_sha256") == MANIFEST and identity.get("manifest_pinned") is True
              and identity.get("stand_in") is None, "R7 identity is not authenticated")
        _need(features.get("flip_tta") is False and features.get("compile") is True
              and features.get("trainer_md5") == "8622009e3db0414124d5614ed252d7a2"
              and features.get("views") == "v4" and features.get("crop") == 352
              and features.get("size") == 384 and features.get("eval_stride") == 1,
              "baseline R7 configuration mismatch")
    for name in ("a172", "pr7"):
        branch = receipt.get(name, {})
        _need(branch.get("studies") == count, "branch cohort mismatch")
        child = branch.get("child", {})
        _need(child.get("exit_code") == 0 and _positive(child.get("seconds")), "incomplete child timing")
        _need(all(not v for v in branch.get("failures", {}).values()), "branch failure")
    pr7 = receipt["pr7"]
    _need(pr7.get("source_version") == "C85R7CODEC 1.1.0"
          and pr7.get("worker_exit_codes") == {"0": 0, "1": 0}, "PR7 worker lineage/failure")
    phase = pr7.get("phases", {}).get("S_R7", {})
    _need(phase.get("exit_code") == 0 and _positive(phase.get("seconds")), "missing complete R7 phase timing")
    _need(phase["seconds"] <= pr7["child"]["seconds"], "R7 phase exceeds enclosing child")
    codecs = receipt.get("codecs", {})
    _need(codecs.get("status") == "ACTIVE" and codecs.get("s_r7_gate") == "ACTIVE"
          and _positive(codecs.get("install_seconds")), "codec baseline incomplete")
    _need(receipt.get("telemetry", {}).get("p_branch", {}).get("rows") == count,
          "receipt output cohort mismatch")


def _linear(n3, n32):
    _need(_positive(n3) and _positive(n32) and n32 > n3, "nonpositive or declining N32 timing")
    slope = (n32 - n3) / 29.0
    fixed = max(0.0, n3 - 3.0 * slope)
    return {"n3_seconds": n3, "n32_seconds": n32, "seconds_per_study": slope,
            "fixed_seconds": fixed, "hidden_studies": 1300,
            "projected_seconds": 1300.0 * slope + fixed}


def baseline(root=ROOT):
    """Fail closed on identity drift; ledger appends preserve pinned record identity."""
    root = Path(root)
    receipts, bindings = {}, {}
    for name, (relative, expected, count) in RECEIPTS.items():
        raw = (root / relative).read_bytes()
        _need(_sha(raw) == expected, f"receipt hash mismatch: {name}")
        receipt = json.loads(raw)
        _validate_receipt(receipt, count)
        receipts[name] = receipt
        bindings[name] = {"path": relative, "sha256": expected, "studies": count}

    raw = (root / SUBMIT).read_bytes()
    submissions = [(line, json.loads(line)) for line in raw.splitlines() if line.strip()]
    submissions = [(line, row) for line, row in submissions if str(row.get("ref")) == REF]
    _need(len(submissions) == 1, "submission ref missing or duplicated")
    line, submitted = submissions[0]
    _need(_sha(line) == SUBMIT_RECORD_SHA256, "submission record hash mismatch")
    _need(submitted.get("id") == IDENTITY and submitted.get("version") == "1"
          and submitted.get("kernel") == "taeyangg4/knee-exp169-c85r7codec-v1"
          and submitted.get("outcome") == "SUBMITTED" and submitted.get("cli_exit") == 0,
          "submission identity or status mismatch")
    bindings["submission"] = {"path": SUBMIT, "file_sha256": _sha(raw),
                              "record_sha256": SUBMIT_RECORD_SHA256, "ref": REF}

    raw = (root / READINGS).read_bytes()
    lines = raw.splitlines()
    _need(bool(lines), "missing reading header")
    readings = []
    for line in lines[1:]:
        if not line.strip():
            continue
        row = next(csv.DictReader(io.StringIO((lines[0] + b"\n" + line).decode("utf-8-sig"))))
        if row.get("submission_id") == REF:
            readings.append((line, row))
    _need(len(readings) == 1, "COMPLETE reading missing or duplicated")
    line, reading = readings[0]
    _need(_sha(line) == READING_RECORD_SHA256, "COMPLETE reading record hash mismatch")
    _need(reading.get("key") == IDENTITY and "COMPLETE" in reading.get("note", ""),
          "reading does not attest COMPLETE for this identity")
    bindings["completion"] = {"path": READINGS, "file_sha256": _sha(raw),
                              "record_sha256": READING_RECORD_SHA256, "ref": REF}
    start = datetime.fromisoformat(submitted["time"])
    end = datetime.fromisoformat(reading["read_at"]).replace(tzinfo=timezone(timedelta(hours=9)))
    observed = (end - start).total_seconds()
    _need(_positive(observed), "invalid submit-to-COMPLETE bound")

    r3, r32 = receipts["visible_n3"], receipts["stress_n32"]
    total = _linear(*(sum(r[branch]["child"]["seconds"] for branch in ("a172", "pr7")) for r in (r3, r32)))
    codec_fixed = max(r["codecs"]["install_seconds"] for r in (r3, r32))
    total["codec_fixed_seconds"] = codec_fixed
    total["projected_seconds"] += codec_fixed
    r7 = _linear(*(r["pr7"]["phases"]["S_R7"]["seconds"] for r in (r3, r32)))
    hidden = max(total["projected_seconds"], observed)
    # If the hidden bound dominates, allocate the entire unexplained excess to
    # R7. This deliberately overestimates the incremental cost of flip TTA.
    r7_excess = max(0.0, observed - total["projected_seconds"])
    s = min(hidden, r7["projected_seconds"] + r7_excess)
    return {"version": VERSION, "source_version": "1.1.0", "submission_ref": REF,
            "hidden_seconds": hidden, "r7_seconds": s, "r7_share": s / hidden,
            "bindings": bindings,
            "derivation": {"total": total, "r7": r7, "r7_unexplained_excess_seconds": r7_excess,
                           "submitted_at": start.isoformat(), "complete_observed_at": end.isoformat(),
                           "submit_to_complete_bound_seconds": observed,
                           "formula": "slope=(N32-N3)/29; fixed=max(0,N3-3*slope); estimate=1300*slope+fixed; H=max(total+codec_fixed,observed_bound)"},
            "assumptions": [
                "N32 stress and N3 visible are extrapolation anchors, not hidden-cohort measurements; assume 1300 hidden studies with comparable study cost.",
                "A172 then PR7 child durations are serial wall times; sum them once. Receipt R7 phase is parallel-worker elapsed wall time: never divide it again by two GPUs or multiply by worker count.",
                "Clamp a negative fitted intercept to zero; codec install is a fixed max(N3,N32) cost. Other uninstrumented startup is covered only to the extent of the observed submit-to-COMPLETE bound.",
                "The COMPLETE reading at 20:01 KST is used, rather than approximate 20:00 in the note. Submission-to-reading includes queue and observation delay, so is a conservative observed runtime bound.",
                "Any observed-bound excess over the slope total is charged wholly to R7. Apply the measured flip/plain factor to the entire R7 share, including fixed overhead, with no speedup credit.",
                "Transfer the same-device local ratio to T4 as a projection assumption, not a measured T4 timing guarantee. Gates do not authorize submission.",
                "Receipt bytes and exact relevant ledger lines are pinned; whole current ledger hashes are recorded, while unrelated append-only additions remain acceptable."
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(baseline(args.root), indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"DO_NOT_SUBMIT: {exc}\n")


if __name__ == "__main__":
    main()
