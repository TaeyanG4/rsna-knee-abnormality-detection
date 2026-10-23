"""Stitch per-fold OOF predictions into one file and score it the way ADR 0004 requires.

  python training/src/raptor_report.py --prefix EXP061_R1 --which best
  python training/src/raptor_report.py --prefix EXP061_R1 --compare EXP062_MIX   # paired bootstrap A vs B
  python training/src/raptor_report.py --prefix EXP061_R1 --labels storage/labels/x.csv  # re-score on another label set

Outputs artifacts/predictions/<PREFIX>_oof_<which>.csv (+ .json report) with: weak OOF macro
AUC on the chosen labels, per-target AUC, per-scanner-group AUC (from study_meta.csv when
present), Gold58 overlap (always empty for clean OOF), and, with --compare, the study-paired
bootstrap interval of the macro-AUC difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raptor_common import EXPERIMENTS, LAB, ROOT, UID, gold_labels, load_label_csv, macro_auc  # noqa: E402

META = ROOT / "storage" / "labels" / "derived_meta" / "study_meta.csv"


def stitch(prefix, which):
    dirs = sorted(d for d in EXPERIMENTS.glob(prefix + "*") if (d / "summary.json").exists())
    frames, folds, label_sources, missing = [], [], set(), []
    for d in dirs:
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        f = d / f"oof_{which}.csv"
        if not f.exists():
            missing.append(d.name); continue
        df = pd.read_csv(f, dtype={UID: str}); df["fold"] = s.get("fold"); frames.append(df)
        folds.append(s.get("fold")); label_sources.add(s.get("args", {}).get("weak_labels"))
    if not frames:
        raise SystemExit(f"no OOF files for {prefix} (missing: {missing})")
    oof = pd.concat(frames, ignore_index=True)
    if oof[UID].duplicated().any():
        raise SystemExit("studies appear in more than one fold; folds overlap")
    return oof, sorted(folds), label_sources, missing


def score(oof, labels):
    ids = [u for u in oof[UID] if u in labels]
    y = np.stack([labels[u] for u in ids]); p = oof.set_index(UID).loc[ids, LAB].to_numpy()
    return ids, y, p


def group_breakdown(ids, y, p):
    if not META.exists():
        return None
    meta = pd.read_csv(META, dtype={UID: str}).set_index(UID)
    groups = meta["scanner_group"].reindex(ids).fillna("unknown").to_numpy()
    out = {}
    for g in sorted(set(groups)):
        m = groups == g
        if m.sum() < 40:
            continue
        macro, per = macro_auc(y[m], p[m])
        out[g] = {"studies": int(m.sum()), "macro": macro, "computable_targets": len(per)}
    return out


def bootstrap_delta(y, pa, pb, reps=1000, seed=20261002):
    rng = np.random.default_rng(seed); n = len(y); deltas = []
    for _ in range(reps):
        pick = rng.integers(0, n, n)
        a, _ = macro_auc(y[pick], pa[pick]); b, _ = macro_auc(y[pick], pb[pick])
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(b - a)
    d = np.array(deltas)
    return {"mean": float(d.mean()), "low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5)),
            "fraction_positive": float((d > 0).mean()), "reps": int(len(d)), "unit": "study"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--which", default="best", choices=["best", "swa"])
    ap.add_argument("--labels", default=None, help="label CSV to score against (default: the labels the folds trained on)")
    ap.add_argument("--compare", default=None, help="second experiment prefix (B); reports B − A with a paired bootstrap")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    oof, folds, sources, missing = stitch(a.prefix, a.which)
    label_path = Path(a.labels) if a.labels else ROOT / next(iter(sources))
    labels, label_hash = load_label_csv(label_path)
    out = Path(a.out) if a.out else ROOT / "artifacts" / "predictions" / f"{a.prefix}_oof_{a.which}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    oof.drop(columns=["fold"]).to_csv(out, index=False)
    ids, y, p = score(oof, labels)
    macro, per = macro_auc(y, p)
    report = {"prefix": a.prefix, "which": a.which, "folds": folds, "studies": int(len(oof)), "scored_studies": len(ids),
              "labels": str(label_path), "label_hash": label_hash, "weak_oof_macro_auc": macro, "per_target": per,
              "scanner_groups": group_breakdown(ids, y, p), "missing_folds": missing, "csv": str(out)}
    print(f"{a.prefix} {a.which}: folds {folds} | {len(oof)} studies | labels {label_path.name}")
    if missing:
        print("missing OOF:", missing)
    print(f"weak OOF macro AUC {macro:.4f}")
    for k, v in per.items():
        print(f"   {k:18s} {v:.3f}")
    if report["scanner_groups"]:
        print("by scanner group:", {g: round(v["macro"], 4) for g, v in report["scanner_groups"].items()})
        vals = [v["macro"] for v in report["scanner_groups"].values() if np.isfinite(v["macro"])]
        report["scanner_group_spread"] = float(max(vals) - min(vals)) if vals else None
    gold = gold_labels(); gid = [u for u in oof[UID] if u in gold]
    if gid:
        report["gold_overlap"] = len(gid)
        print(f"WARNING: {len(gid)} Gold58 studies inside the OOF; this OOF is not clean")
    if a.compare:
        oof_b, folds_b, _, _ = stitch(a.compare, a.which)
        common = [u for u in ids if u in set(oof_b[UID])]
        ya = np.stack([labels[u] for u in common])
        pa = oof.set_index(UID).loc[common, LAB].to_numpy(); pb = oof_b.set_index(UID).loc[common, LAB].to_numpy()
        ma, _ = macro_auc(ya, pa); mb, _ = macro_auc(ya, pb)
        report["compare"] = {"b": a.compare, "common_studies": len(common), "macro_a": ma, "macro_b": mb, "delta_b_minus_a": mb - ma,
                             "bootstrap": bootstrap_delta(ya, pa, pb)}
        bs = report["compare"]["bootstrap"]
        print(f"compare {a.compare} − {a.prefix}: {mb - ma:+.4f} (95% {bs['low']:+.4f} ~ {bs['high']:+.4f}, positive {bs['fraction_positive']:.0%}) on {len(common)} studies")
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
