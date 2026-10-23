"""Judge label/recipe experiments fold by fold, the way ADR 0004 asks.

Every run is re-scored on the SAME reference labels (default: the original report labels), so a
model trained on mixed labels is not rewarded for echoing its own teacher. For each candidate fold
the baseline run of the same fold is paired study by study and a bootstrap interval of the macro
AUC difference is reported. Gold58 is printed as a diagnostic only.

  python tools/fold_compare.py --baseline EXP061_R1 --candidate EXP073_R2PF EXP072_R2MIX_R34
  python tools/fold_compare.py --baseline EXP061_R1 --candidate EXP073_R2PF --which swa --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
from raptor_common import EXPERIMENTS, LAB, ROOT, UID, load_label_csv, macro_auc  # noqa: E402
from raptor_report import bootstrap_delta, score  # noqa: E402

DEFAULT_LABELS = ROOT / "storage" / "labels" / "dreaddevelopment__rsna-knee-labels" / "labels_llm_soft.csv"


def load_oof(tag: str, which: str):
    f = EXPERIMENTS / tag / f"oof_{which}.csv"
    if not f.exists():
        f = EXPERIMENTS / tag / "oof_best.csv"
    return pd.read_csv(f, dtype={UID: str}) if f.exists() else None


def fold_runs(prefix: str):
    """{fold: tag} for tags named <prefix>_F<k>."""
    out = {}
    for d in EXPERIMENTS.glob(f"{prefix}_F*"):
        if (d / "summary.json").exists():
            try:
                out[int(d.name.rsplit("_F", 1)[1])] = d.name
            except ValueError:
                pass
    return out


def gold_of(tag: str):
    s = json.loads((EXPERIMENTS / tag / "summary.json").read_text(encoding="utf-8"))
    return s.get("gold_once_after_weak_selected_epoch")


def compare(baseline: str, candidate: str, labels: dict, which: str, reps: int):
    base, cand = fold_runs(baseline), fold_runs(candidate)
    rows = []
    for fold in sorted(cand):
        if fold not in base:
            rows.append({"fold": fold, "candidate": cand[fold], "note": "no baseline fold"}); continue
        a, b = load_oof(base[fold], which), load_oof(cand[fold], which)
        if a is None or b is None:
            rows.append({"fold": fold, "candidate": cand[fold], "note": "oof missing"}); continue
        ids_a, y, pa = score(a, labels)
        ids_b, _, pb = score(b, labels)
        common = sorted(set(ids_a) & set(ids_b))
        ia = {u: i for i, u in enumerate(ids_a)}; ib = {u: i for i, u in enumerate(ids_b)}
        y = np.stack([labels[u] for u in common]); pa = pa[[ia[u] for u in common]]; pb = pb[[ib[u] for u in common]]
        ma, per_a = macro_auc(y, pa); mb, per_b = macro_auc(y, pb)
        boot = bootstrap_delta(y, pa, pb, reps=reps)
        rows.append({"fold": fold, "baseline": base[fold], "candidate": cand[fold], "n": len(common),
                     "baseline_auc": round(ma, 4), "candidate_auc": round(mb, 4), "delta": round(mb - ma, 4),
                     "ci_low": round(boot["low"], 4), "ci_high": round(boot["high"], 4),
                     "gold_baseline": gold_of(base[fold]), "gold_candidate": gold_of(cand[fold]),
                     "per_target_delta": {k: round(per_b[k] - per_a[k], 3) for k in per_a if k in per_b}})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="prefix of the reference runs, e.g. EXP061_R1")
    ap.add_argument("--candidate", nargs="+", required=True, help="prefixes to judge, e.g. EXP073_R2PF")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS), help="reference labels every run is scored on")
    ap.add_argument("--which", default="best", choices=["best", "swa"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    labels, label_hash = load_label_csv(Path(a.labels))
    report = {"reference_labels": a.labels, "label_hash": label_hash, "which": a.which, "comparisons": {}}
    for cand in a.candidate:
        rows = compare(a.baseline, cand, labels, a.which, a.reps)
        report["comparisons"][cand] = rows
        print(f"\n== {cand} vs {a.baseline} (scored on reference labels, {a.which})")
        for r in rows:
            if "note" in r:
                print(f"  fold {r['fold']}: {r['note']}"); continue
            verdict = "better" if r["ci_low"] > 0 else ("worse" if r["ci_high"] < 0 else "no difference")
            print(f"  fold {r['fold']}: {r['baseline_auc']:.4f} -> {r['candidate_auc']:.4f}  delta {r['delta']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] {verdict} | gold {r['gold_baseline']:.3f} -> {r['gold_candidate']:.3f}")
            worst = sorted(r["per_target_delta"].items(), key=lambda kv: kv[1])[:2]
            best = sorted(r["per_target_delta"].items(), key=lambda kv: -kv[1])[:2]
            print(f"     up: {best}  down: {worst}")
        deltas = [r["delta"] for r in rows if "delta" in r]
        if len(deltas) > 1:
            print(f"  mean delta over {len(deltas)} folds: {np.mean(deltas):+.4f}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
