"""v1.0.0: explicit intended-cohort identity checks; never infer full size from N>=30.

Release integration is PLAN ONLY. EXP176 uses this helper in its own wrapper.
Only UID columns are accepted as cohort evidence; no labels/reports are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

VERSION = "1.0.0"
UID = "StudyInstanceUID"


def identity(ids):
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids) or any(
        not isinstance(x, str) or not x or any(c in x for c in "\r\n/\\") or x in (".", "..")
        for x in ids
    ):
        raise ValueError("Expected nonempty unique safe study IDs")
    def digest(values):
        return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()
    return {"studies": len(ids), "ordered_sha256": digest(ids),
            "membership_sha256": digest(sorted(ids)),
            "hash_scheme": "utf8-uid-lines-final-lf-v1"}


def check(ids, intended):
    actual = identity(ids)
    if any(actual.get(k) != intended.get(k) for k in actual):
        raise ValueError("Intended cohort size/hash/order mismatch")
    return {"status": "MATCH_INTENDED_COHORT", **actual,
            "scope": intended.get("scope", "UNSPECIFIED"),
            "full_hidden_qualification": False}


def read_ids(path):
    # Caller must supply a test/benchmark UID CSV, never train.csv or label files.
    path = Path(path)
    if path.name not in ("test.csv", "submission.csv", "benchmark-predictions.csv", "cohort.csv"):
        raise ValueError("Only explicit cohort/output CSV filenames are allowed")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if UID not in (reader.fieldnames or []) or "Report" in reader.fieldnames:
            raise ValueError("Invalid cohort CSV schema")
        ids = [r[UID] for r in reader]
    identity(ids)
    return ids


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--intended", required=True)
    args = p.parse_args(argv)
    print(json.dumps(check(read_ids(args.csv), json.loads(Path(args.intended).read_text(encoding="utf-8")))))


if __name__ == "__main__":
    main()
