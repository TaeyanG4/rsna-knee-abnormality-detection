"""Recompute label_hash / split_hash of finished runs with the platform-independent scheme.

Until 2026-10-02 the hashes were taken over pandas to_csv output, which ends lines with
CRLF on Windows and LF on Linux, so a local run and a Kaggle/Colab run on identical labels
looked incomparable. raptor_common now hashes with LF everywhere ("lf-v2"). This rewrites
the two hashes in summary.json and keeps the old values under legacy_hashes.

  python tools/rehash_summary.py EXP060_RAPTOR_F0 [TAG ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
from raptor_common import EXPERIMENTS, load_folds, load_label_csv  # noqa: E402


def rehash(tag: str) -> dict:
    path = EXPERIMENTS / tag / "summary.json"
    s = json.loads(path.read_text(encoding="utf-8"))
    if s.get("hash_scheme") == "lf-v2":
        return {"tag": tag, "changed": False}
    args = s.get("args") or {}
    _, label_hash = load_label_csv(Path(args["labels"]))
    _, split_hash = load_folds(Path(args["fold_file"]))
    s["legacy_hashes"] = {"label_hash": s.get("label_hash"), "split_hash": s.get("split_hash")}
    s.update(label_hash=label_hash, split_hash=split_hash, hash_scheme="lf-v2")
    path.write_text(json.dumps(s, indent=1), encoding="utf-8")
    return {"tag": tag, "changed": True, "label_hash": label_hash[:12], "split_hash": split_hash[:12]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for t in sys.argv[1:]:
        print(json.dumps(rehash(t)))
