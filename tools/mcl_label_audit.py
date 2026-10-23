"""Read-only train-report MCL definition audit; writes analysis, never training labels.

Text patterns are overlapping review queues, not clinical adjudication. No existing
training/acquisition scripts, external report code, models or GPU are executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

UID = "StudyInstanceUID"
ROOT = Path(__file__).resolve().parents[1]
TOPIC = "docs/handoffs/evidence/2026-10/2026-10-03-continuous-research/topic_733343.json"
LABEL_FILES = {
    "dread": "storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv",
    "qwen": "storage/labels/laymond__rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_raw.csv",
    "flight": "storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv",
    "pilkwang": "storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", str(text).lower()) if not unicodedata.combining(c))


MENTION = re.compile(r"\bmcl\b|\blcm\b|\blci\b|\blli\b|medial\w* collat|collat\w* mediall?|collat\w* intern|colat\w* mediall?|colat\w* intern|medial\w* kollat|medijaln\w* kolat|innenband|медиал\w* колатерал|εσω πλαγι|内侧副韧带", re.I)
FLAGS = {
    "explicit_high_grade_or_complete_cue": re.compile(r"high[- ]?grade|full[- ]thickness|complete (?:acute )?(?:tear|rupture)|complete rupture|komplett\w* ruptur|ruptur\w* komplett|ruptura total|rupture complete|grade\s*(?:iii|3)\b|тотална руптура|πληρη\w* ρηξ"),
    "explicit_low_grade_cue": re.compile(r"low[- ]?grade|bas grade|bajo grado|grade\s*(?:i|1)\b(?!\s*[-/]\s*(?:ii|2))|\bi\s+stupnja|grade[- ]?one"),
    "chronic_old_healed_cue": re.compile(r"\b(?:old|chronic|remote|healed|healing|ancien\w*|chronique|sequel\w*|antigu\w*|cronico|chronisch\w*)\b"),
    "grade_ii_cue": re.compile(r"grade\s*(?:ii|2)\b(?!\s*[-/]\s*(?:iii|3))|grado\s*(?:ii|2)\b|\bii\s+stupnja|moyen grade"),
    "partial_tear_cue": re.compile(r"partial\w* (?:tear|ruptur)|(?:tear|ruptur)\w* partial|dechirure.{0,60}partiel|partiel.{0,60}dechirure|partielle\w* ruptur|partiell.{0,40}ruptur|teilruptur"),
    "sprain_cue": re.compile(r"sprain|entorse|distorsion|distension|esguince"),
    "explicit_no_rupture_cue": re.compile(r"no evidence of (?:rupture|tear)|without (?:rupture|tear|discontinuity)|no (?:rupture|tear)|bez znakova rupture|sans.{0,20}(?:rupture|dechirure)"),
}


def report_features(report):
    # Keep clauses, not arbitrary windows spanning adjacent ACL/LCL descriptions.
    # This is still deliberately a retrieval heuristic, especially outside English.
    text = str(report).replace("I. stupnja", "I stupnja").replace("II. stupnja", "II stupnja")
    clauses = [clause.strip() for clause in re.split(r"[.\n;]+", text) if MENTION.search(normalize(clause))]
    lowered = normalize(" | ".join(clauses))
    flags = {name: bool(pattern.search(lowered)) for name, pattern in FLAGS.items()}
    flags.update(mcl_clause_found=bool(clauses),
                 high_and_exclusion_cues=flags["explicit_high_grade_or_complete_cue"] and
                 (flags["explicit_low_grade_cue"] or flags["chronic_old_healed_cue"] or flags["explicit_no_rupture_cue"]))
    return flags, " | ".join(clauses)


def _read(path, columns):
    frame = pd.read_csv(path, usecols=columns, dtype={UID: str})
    if frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError(f"invalid study identity: {path}")
    return frame.set_index(UID)


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"preserve prior analysis; use a new output: {output}")
    sources = []
    train_path = root / "storage/competition/train.csv"; sources.append(train_path)
    train = _read(train_path, [UID, "Report", "MCL"])
    gold_ids = set(train.index[train.MCL.notna()])
    if len(gold_ids) != 58:
        raise ValueError("unexpected Gold exclusion set")
    folds_path = root / "artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv"; sources.append(folds_path)
    folds = _read(folds_path, [UID, "fold", "report_group_sha256", "is_gold", "quarantined_gold_report_duplicate"])
    ids = sorted(set(folds.index[folds.fold >= 0]) & set(train.index) - gold_ids)
    if len(ids) != 4345 or set(ids) & gold_ids:
        raise ValueError("audit requires the existing 4345-study development cohort with no Gold")
    cohort = train.loc[ids, ["Report"]].join(folds[["fold", "report_group_sha256"]])
    for name, relative in LABEL_FILES.items():
        path = root / relative; sources.append(path)
        columns = [UID, "MCL"] if name != "qwen" else [UID, "Report", "MCL__confidence", "MCL__label", "MCL__mentioned", "MCL__negated", "status"]
        label = _read(path, columns)
        if name == "qwen":
            label = label.rename(columns={"MCL__confidence": name, "Report": "qwen_report", "MCL__label": "qwen_hard",
                                           "MCL__mentioned": "qwen_mentioned", "MCL__negated": "qwen_negated", "status": "qwen_status"})
        else:
            label = label.rename(columns={"MCL": name})
        cohort = cohort.join(label)
        metadata = path.parent / "dataset-metadata.json"
        if metadata.exists():
            sources.append(metadata)
    for model, prefix in (("R1", "EXP061_R1"), ("R2", "EXP072_R2MIX_R34")):
        parts = []
        for fold in range(5):
            path = root / f"artifacts/experiments/{prefix}_F{fold}/oof_best.csv"; sources.append(path)
            frame = _read(path, [UID, "MCL"])
            if set(frame.index) != set(cohort.index[cohort.fold == fold]):
                raise ValueError(f"OOF fold identity differs: {path}")
            parts.append(frame)
        prediction = pd.concat(parts).rename(columns={"MCL": f"{model}_mcl_prediction"})
        cohort = cohort.join(prediction)
    meta_path = root / "storage/labels/derived_meta/study_meta.csv"; sources.append(meta_path)
    cohort = cohort.join(_read(meta_path, [UID, "manufacturer", "field", "scanner_group"]))
    extracted = [report_features(text) for text in cohort.Report]
    features = pd.DataFrame([value[0] for value in extracted], index=cohort.index)
    cohort = cohort.join(features)
    cohort["mcl_clauses"] = [value[1] for value in extracted]
    binary = {name: np.rint(cohort[name].to_numpy(dtype=np.float32)) for name in LABEL_FILES}
    label_summary = {}
    for name in LABEL_FILES:
        values = cohort[name].to_numpy(dtype=np.float32); finite = np.isfinite(values)
        if np.isinf(values).any() or np.any(values[finite] < 0) or np.any(values[finite] > 1):
            raise ValueError(f"invalid probabilities: {name}")
        label_summary[name] = {"available": int(finite.sum()), "missing": int((~finite).sum()),
                               "binary_positive": int((binary[name] == 1).sum()), "exact_half": int((values == .5).sum()),
                               "soft_value_counts": {str(k): int(v) for k, v in cohort[name].value_counts(dropna=False).sort_index().items()}}
    pairs = []
    for left, right in itertools.combinations(LABEL_FILES, 2):
        a, b = binary[left], binary[right]; valid = np.isfinite(a) & np.isfinite(b)
        pairs.append({"left": left, "right": right, "available": int(valid.sum()),
                      "both_positive": int((valid & (a == 1) & (b == 1)).sum()),
                      "left_only_positive": int((valid & (a == 1) & (b == 0)).sum()),
                      "right_only_positive": int((valid & (a == 0) & (b == 1)).sum()),
                      "both_negative": int((valid & (a == 0) & (b == 0)).sum()),
                      "mean_absolute_soft_difference": float(np.abs(cohort.loc[valid, left] - cohort.loc[valid, right]).mean())})
    strata = []
    for flag in features.columns:
        mask = cohort[flag].to_numpy(bool)
        strata.append({"review_queue": flag, "studies": int(mask.sum()),
                       **{name + "_positive": int((mask & (binary[name] == 1)).sum()) for name in LABEL_FILES},
                       "R1_prediction_median": float(cohort.loc[mask, "R1_mcl_prediction"].median()) if mask.any() else None})
    error_path = root / "docs/handoffs/evidence/2026-10/2026-10-03-error-audit/weak_rank_error_cases.csv"; sources.append(error_path)
    errors = pd.read_csv(error_path, dtype={UID: str})
    errors = errors[(errors.model == "R1") & (errors.target == "MCL")]
    top = cohort.loc[errors[UID]].copy()
    top["allocated_macro_deficit"] = errors.set_index(UID).loc[top.index, "allocated_macro_deficit"]
    output.mkdir(parents=True)
    # No export has the original training-label schema; these are analysis tables only.
    table_columns = ["fold", "report_group_sha256", "dread", "qwen", "flight", "pilkwang", "qwen_hard",
                     "qwen_mentioned", "qwen_negated", "R1_mcl_prediction", "R2_mcl_prediction", "manufacturer", "field", *features.columns]
    cohort[table_columns].to_csv(output / "cohort_features.csv", index=True)
    pd.DataFrame(strata).to_csv(output / "review_queue_counts.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "label_disagreements.csv", index=False)
    top[[*table_columns, "mcl_clauses", "allocated_macro_deficit"]].to_csv(output / "r1_top10_weak_errors.csv", index=True)
    selected = []
    for flag in FLAGS:
        subset = cohort[cohort[flag] & ((cohort.dread > .5) | (cohort.qwen > .5))]
        selected.extend(subset.index[:8])
    selected = list(dict.fromkeys([*top.index, *selected]))
    cohort.loc[selected, [*table_columns, "mcl_clauses"]].to_csv(output / "review_cases.csv", index=True)
    topic_path = root / TOPIC; sources.append(topic_path)
    topic = json.loads(topic_path.read_text(encoding="utf-8"))
    topic_lines = topic["revision"]["content"].splitlines()
    quote = next(line for line in topic_lines if line.startswith("MCL tear:"))
    summary = {"observed_at": datetime.now().astimezone().isoformat(),
               "scope": "4345 train-only exposed development studies; no Gold58 labels scored; no label edits/training/GPU",
               "cohort": {"train_rows": len(train), "gold_excluded": len(gold_ids), "gold_overlap": len(set(ids) & gold_ids),
                          "extra_quarantined_excluded": len(set(train.index) - gold_ids - set(ids)), "studies": len(cohort),
                          "report_groups": int(cohort.report_group_sha256.nunique())},
               "host_definition": {"url": topic["url"], "revision": topic["current_revision"], "source_observed_at": topic["observed_at"],
                                   "source_authority_field": topic.get("authority"), "quote": quote,
                                   "line": topic_lines.index(quote) + 1,
                                   "authority_caveat": "Stored source authority says participant; cited as competition overview by named Po-Hao Howard Chen, not independently reauthenticated in this audit."},
               "binary_contract": "float32 probabilities -> numpy.rint; exact0.5 rounds0; Qwen hard label shown separately",
               "labels": label_summary, "pairs": pairs, "overlapping_text_review_queues": strata,
               "qwen": {"reports_exactly_equal_to_train": int((cohort.Report == cohort.qwen_report).sum()),
                        "status_counts": {str(k): int(v) for k, v in cohort.qwen_status.value_counts(dropna=False).items()},
                        "mentioned_negated_counts": {str(k): int(v) for k, v in cohort.groupby(["qwen_mentioned", "qwen_negated"], dropna=False).size().items()},
                        "unmentioned_positive": int(((cohort.qwen_mentioned == 0) & (binary["qwen"] == 1)).sum()),
                        "negated_positive": int(((cohort.qwen_negated == 1) & (binary["qwen"] == 1)).sum()),
                        "hard_vs_rounded_probability_disagreements": int((cohort.qwen_hard != binary["qwen"]).sum())},
               "r1_top10": {"studies": len(top), "review_cues": {flag: int(top[flag].sum()) for flag in features.columns}},
               "limitations": ["Reports are not independent image rereads or clinical ground truth", "Text patterns are overlapping retrieval cues, not labels; severity association/negation/languages require manual review", "R1 checkpoints were selected on exposed weak folds; R2 also has indirect teacher-fold leakage and mixed-label epoch selection", "No public/private performance gain is established", "No existing labels, predictions, checkpoints or training processes were modified"],
               "sources": [{"path": str(path.relative_to(root)).replace('\\', '/'), "sha256": sha(path)} for path in sources]}
    (output / "audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.root, args.out)
    print(json.dumps({"cohort": result["cohort"], "labels": {key: {k: v for k, v in value.items() if k != 'soft_value_counts'} for key, value in result['labels'].items()}, "pairs": result['pairs'], "review_queues": result['overlapping_text_review_queues'], "qwen": result['qwen'], "r1_top10": result['r1_top10']}, indent=2, ensure_ascii=False))
