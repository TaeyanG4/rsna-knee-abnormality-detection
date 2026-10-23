"""CPU-only paired weak-ranking audit of frozen R1/ResNet best OOF predictions.

No model imports, training, Gold predictions, blending, ratio search or routing.
The paired model comparison is a recipe comparison, not an encoder-only effect.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PREFIXES = {"R1": "EXP061_R1", "ResNet34": "EXP071_R34_LR"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def pair_table(y, left, right):
    """Exact +/- pair counts. Status0=correct,1=tied,2=wrong; ties half error."""
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both weak classes required")
    matrix = np.zeros((3, 3), dtype=np.int64)
    for begin in range(0, len(positive), 128):
        ids = positive[begin:begin + 128]
        a = np.where(left[ids, None] > left[negative], 0, np.where(left[ids, None] == left[negative], 1, 2))
        b = np.where(right[ids, None] > right[negative], 0, np.where(right[ids, None] == right[negative], 1, 2))
        matrix += np.bincount((a * 3 + b).ravel(), minlength=9).reshape(3, 3)
    pairs = len(positive) * len(negative)
    if matrix.sum() != pairs:
        raise ValueError("Pair partition mismatch")
    left_error = float(np.dot(np.array([0, .5, 1]), matrix.sum(axis=1)))
    right_error = float(np.dot(np.array([0, .5, 1]), matrix.sum(axis=0)))
    left_auc, right_auc = roc_auc_score(y, left), roc_auc_score(y, right)
    if abs(left_auc - (1 - left_error / pairs)) > 1e-12 or abs(right_auc - (1 - right_error / pairs)) > 1e-12:
        raise ValueError("Pair-derived AUC disagrees with sklearn")
    wrong_left = int(matrix[2].sum())
    wrong_right = int(matrix[:, 2].sum())
    ranks_left, ranks_right = pd.Series(left).rank().to_numpy(), pd.Series(right).rank().to_numpy()
    return {"studies": len(y), "P": len(positive), "N": len(negative), "pairs": pairs,
            "R1_auc": float(left_auc), "ResNet_auc": float(right_auc), "delta": float(right_auc - left_auc),
            "spearman": float(np.corrcoef(ranks_left, ranks_right)[0, 1]),
            "spearman_positive": float(pd.Series(left[positive]).corr(pd.Series(right[positive]), method="spearman")),
            "spearman_negative": float(pd.Series(left[negative]).corr(pd.Series(right[negative]), method="spearman")),
            "both_correct": int(matrix[0, 0]), "both_wrong": int(matrix[2, 2]),
            "R1_wrong_ResNet_correct": int(matrix[2, 0]), "R1_correct_ResNet_wrong": int(matrix[0, 2]),
            "R1_wrong_total": wrong_left, "ResNet_wrong_total": wrong_right,
            "fraction_R1_wrong_reversed_by_ResNet": int(matrix[2, 0]) / wrong_left if wrong_left else None,
            "fraction_R1_wrong_shared": int(matrix[2, 2]) / wrong_left if wrong_left else None,
            "opposite_strict_order_rate": (int(matrix[2, 0]) + int(matrix[0, 2])) / pairs,
            "tie_in_either_pairs": int(matrix[1].sum() + matrix[:, 1].sum() - matrix[1, 1]),
            "R1_error_equivalents": left_error, "ResNet_error_equivalents": right_error,
            "pair_matrix_R1_rows_ResNet_columns": matrix.tolist()}


def run(root, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError("Preserve prior audits; use new output")
    start = time.perf_counter()
    prior_path = root / "docs/handoffs/evidence/2026-10/2026-10-03-error-audit/audit.json"
    prior = read_json(prior_path)
    prior_hashes = {x["uri"].replace("workspace://", ""): x["sha256"] for x in prior["source_manifest"]}
    def check_prior(path):
        relative = str(path.relative_to(root)).replace("\\", "/")
        if relative not in prior_hashes or sha(path) != prior_hashes[relative]:
            raise ValueError(f"Input differs from prior audited snapshot: {relative}")
    out.mkdir(parents=True)
    write_json(out / "analysis_plan.json", {"created_at": datetime.now().astimezone().isoformat(),
        "status": "Exploratory secondary audit; prior PFOA AUC direction already known",
        "scope": "Same4345 studies, original Dread float32/rint labels, best-only10 OOF files; no Gold/weights/volumes/Report read",
        "metrics_fixed_before_pair_results": ["12-target pooled and fivefold AUC", "exact3x3 paired ranking outcomes", "Spearman overall/within weak class", "4343 healthy PFOA sensitivity"],
        "forbidden": ["blend or ratio search", "target routing", "Gold/LB selection", "architecture-only causal claim"]})
    label_path = root / "storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv"
    folds_path = root / "artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv"
    excluded_path = root / "artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/excluded.csv"
    check_prior(label_path); check_prior(folds_path)
    labels = pd.read_csv(label_path, usecols=[UID, *TARGETS], dtype={UID: str}).set_index(UID)
    folds = pd.read_csv(folds_path, dtype={UID: str}).set_index(UID)
    if not labels.index.is_unique or not folds.index.is_unique:
        raise ValueError("Duplicate source identity")
    ids = sorted(folds.index[folds.fold >= 0])
    if len(ids) != 4345 or not set(ids).issubset(labels.index):
        raise ValueError("Expected original4345 cohort")
    if folds.loc[ids, "is_gold"].any() or folds.loc[ids, "quarantined_gold_report_duplicate"].any():
        raise ValueError("Gold/quarantine overlap")
    if folds.groupby("report_group_sha256").fold.nunique().max() > 1:
        raise ValueError("Split report group")
    ysoft = labels.loc[ids, TARGETS].to_numpy(np.float32)
    if not np.isfinite(ysoft).all() or not ((ysoft >= 0) & (ysoft <= 1)).all():
        raise ValueError("Invalid reference labels")
    y = np.rint(ysoft).astype(int)
    fold_numbers = folds.loc[ids, "fold"].to_numpy(int)
    sources = [prior_path, label_path, folds_path, excluded_path]
    predictions, recipes = {}, {}
    for name, prefix in PREFIXES.items():
        parts, recipe = [], []
        for fold in range(5):
            folder = root / "artifacts/experiments" / f"{prefix}_F{fold}"
            summary_path, oof_path = folder / "summary.json", folder / "oof_best.csv"
            check_prior(summary_path); check_prior(oof_path)
            summary = read_json(summary_path)
            if summary["fold"] != fold or summary["label_hash"] != prior["reference"]["canonical_label_hash"] or summary["split_hash"] != prior["reference"]["split_hash"] or summary["args"]["select"] != "oof":
                raise ValueError("Saved recipe/reference contract differs")
            frame = pd.read_csv(oof_path, dtype={UID: str})
            if list(frame.columns) != [UID, *TARGETS] or not frame[UID].is_unique:
                raise ValueError("OOF identity/columns differ")
            frame = frame.set_index(UID)
            if set(frame.index) != set(folds.index[folds.fold == fold]):
                raise ValueError("OOF UID set differs from fold")
            parts.append(frame)
            args = summary["args"]
            recipe.append({"tag": f"{prefix}_F{fold}", "fold": fold, "best_epoch": summary["best_epoch"],
                "historical_selected_checkpoint": summary["selected_checkpoint"], "used_here": "best",
                "total_seconds": summary["total_seconds"], "train_rows": summary["train"],
                "args": {key: args[key] for key in ["arch", "res", "k", "k_eval", "epochs", "bs", "accum", "bb_lr", "head_lr", "wd", "workers", "grad_ckpt", "amp", "norm", "oof_subsample", "seed"]},
                "environment": summary["environment"]})
            sources.extend([summary_path, oof_path])
        pred = pd.concat(parts).loc[ids, TARGETS].to_numpy(float)
        if not np.isfinite(pred).all() or not ((pred >= 0) & (pred <= 1)).all():
            raise ValueError("Invalid saved probabilities")
        predictions[name] = pred; recipes[name] = recipe
    records = []
    for fold in [None, 0, 1, 2, 3, 4]:
        use = np.ones(len(ids), dtype=bool) if fold is None else fold_numbers == fold
        for j, target in enumerate(TARGETS):
            metric = pair_table(y[use, j], predictions["R1"][use, j], predictions["ResNet34"][use, j])
            records.append({"scope": "pooled" if fold is None else f"fold{fold}", "target": target, **metric})
    excluded = set(pd.read_csv(excluded_path, usecols=[UID], dtype={UID: str})[UID])
    healthy = np.array([uid not in excluded for uid in ids]); j = TARGETS.index("PF OA")
    healthy_pf = pair_table(y[healthy, j], predictions["R1"][healthy, j], predictions["ResNet34"][healthy, j])
    table = pd.DataFrame([{key: val for key, val in row.items() if not isinstance(val, list)} for row in records])
    table.to_csv(out / "paired_targets.csv", index=False)
    table[table.target == "PF OA"].to_csv(out / "pfoa_pairs.csv", index=False)
    target_stability, scope_macros = [], []
    for target in TARGETS:
        part = table[(table.target == target) & (table.scope != "pooled")]
        pooled = table[(table.target == target) & (table.scope == "pooled")].iloc[0]
        target_stability.append({"target": target, "fold_delta_mean": float(part.delta.mean()),
            "positive_folds": int((part.delta > 0).sum()), "negative_folds": int((part.delta < 0).sum()),
            "fold_delta_min": float(part.delta.min()), "fold_delta_max": float(part.delta.max()),
            "pooled_delta": float(pooled.delta), "pooled_spearman": float(pooled.spearman),
            "fold_mean_rescue_fraction": float(part.fraction_R1_wrong_reversed_by_ResNet.mean())})
    for scope, part in table.groupby("scope"):
        scope_macros.append({"scope": scope, "R1_macro": float(part.R1_auc.mean()), "ResNet_macro": float(part.ResNet_auc.mean()),
            "delta_macro": float(part.delta.mean()), "delta_other11": float(part.loc[part.target != "PF OA", "delta"].mean())})
    pd.DataFrame(target_stability).to_csv(out / "target_stability.csv", index=False)
    pd.DataFrame(scope_macros).to_csv(out / "scope_macros.csv", index=False)
    # Check same raw inputs against earlier manifest, without opening any Gold file.
    source_records = []
    for path in sources:
        relative = str(path.relative_to(root)).replace("\\", "/")
        source_records.append({"path": relative, "sha256": sha(path)})
    summary = {"observed_at": datetime.now().astimezone().isoformat(), "cpu_seconds": time.perf_counter() - start,
        "reference": {**prior["reference"], "Gold_labels_or_predictions_used_here": False},
        "source_recipes": recipes, "metrics": records, "target_stability": target_stability, "scope_macros": scope_macros,
        "source_files_matched_prior_audit": 22,
        "healthy4343_pfoa_sensitivity": healthy_pf,
        "existing_pfoa_conditional_delta_ci": next(t["delta_ci_vs_R1"] for t in prior["models"]["ResNet34"]["targets"] if t["target"] == "PF OA"),
        "sources": source_records, "tool_sha256": sha(__file__),
        "limitations": ["Weak-reference development pairs, not clinical errors", "Best epochs selected on exposed folds, with historical200-row OOF subsampling", "Recipe differences include architecture/pretraining,lr,epochs,precision,framework and device", "Pair reversals are descriptive, not achievable ensemble gain; no oracle or blend evaluated", "Pooled pairs include cross-fold score calibration; per-fold rows retained", "Gold and publicLB were not scored or used for selection"]}
    write_json(out / "audit.json", summary)
    return {"cpu_seconds": summary["cpu_seconds"], "pfoa": [r for r in records if r["target"] == "PF OA"],
            "scope_macros": scope_macros, "target_stability": target_stability, "healthy4343_pfoa": healthy_pf}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT)); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.root, args.out)
    print(json.dumps({"cpu_seconds": result["cpu_seconds"], "pfoa": [
        {key: row[key] for key in ["scope", "delta", "R1_wrong_ResNet_correct", "R1_correct_ResNet_wrong", "fraction_R1_wrong_reversed_by_ResNet"]}
        for row in result["pfoa"]], "out": args.out}, ensure_ascii=False, indent=2))
