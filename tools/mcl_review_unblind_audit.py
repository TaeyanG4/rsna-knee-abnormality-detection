"""Post-freeze MCL weak-label comparison; no clinical scoring or label export.

Verifies frozen annotations before first sealed access. Original annotations,
training labels, predictions and models remain read-only. Output is create-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "docs/handoffs/evidence/2026-10/2026-10-03-mcl-review"
UID = "StudyInstanceUID"
SOURCES = {
    "dread": ("storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv", "MCL"),
    "qwen_raw_v2": ("storage/labels/laymond__rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_raw.csv", "MCL__confidence"),
    "flight_hybrid": ("storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv", "MCL"),
    "pilkwang_v2": ("storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv", "MCL"),
}


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Frozen hash mismatch: {path}")
    return actual


def run(root, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError("Preserve prior audits; choose new output directory")
    base = root / BASE
    freeze = read_json(base / "annotations120_freeze_v1.json")
    manifest = read_json(base / "manifest.json")
    # No sealed path is opened until every annotation/protocol lock passes.
    locks = {base / relative: expected for relative, expected in freeze["source_batches"].items()}
    locks.update({base / "annotations120_v1.jsonl": freeze["combined120_sha256"],
                  base / "continuation/annotations_remaining100_v1.jsonl": freeze["remaining100_sha256"],
                  base / "packet.jsonl": freeze["packet_sha256"],
                  base / "annotation_schema.json": manifest["files_sha256"]["annotation_schema.json"],
                  root / "docs/research/mcl-review-protocol-2026-10-03.md": freeze["protocol_sha256"],
                  root / "tools/mcl_review_packet.py": manifest["tool_sha256"]})
    lock_hashes = {str(path.relative_to(root)).replace("\\", "/"): checked(path, expected) for path, expected in locks.items()}
    out.mkdir(parents=True)
    write_json(out / "pre_access_lock.json", {"verified_at": now(), "status": "PASS", "sealed_access_started": False,
               "authorization": "Root requested separate local post-freeze unblinding audit after120 annotations frozen; no edits/relabel/training",
               "freeze_manifest_sha256": sha(base / "annotations120_freeze_v1.json"), "verified_hashes": lock_hashes})

    first_sealed_access = now()
    sealed_hashes = {name: checked(base / name, manifest["files_sha256"][name])
                     for name in ("sealed/mapping.jsonl", "sealed/selection_manifest.json")}
    mapping = pd.DataFrame(read_lines(base / "sealed/mapping.jsonl"))
    selection = read_json(base / "sealed/selection_manifest.json")
    annotations = read_lines(base / "annotations120_v1.jsonl")
    anno = pd.DataFrame([{"case_id": row["case_id"], "decision": row["decision"], "negative_basis": row["negative_basis"],
                           "language": row["language"], "contradiction": row["contradiction"],
                           "high_grade_without_acute": any(c["anatomy"] != "other_or_uncertain" and c["assertion"] == "affirmed"
                              and c["severity"] in {"high_grade_partial", "complete"} and c["temporality"] != "acute" for c in row["components"])}
                         for row in annotations])
    if len(anno) != 120 or not anno.case_id.is_unique or not mapping.case_id.is_unique or not mapping[UID].is_unique:
        raise ValueError("Invalid review identity/size")
    if set(anno.case_id) != set(mapping.case_id):
        raise ValueError("Blinded/sealed case coverage mismatch")
    frame = anno.merge(mapping, on="case_id", validate="one_to_one").sort_values("case_id").set_index(UID)
    old_audit = read_json(root / "docs/handoffs/evidence/2026-10/2026-10-03-mcl-definition/audit.json")
    old_hashes = {entry["path"]: entry["sha256"] for entry in old_audit["sources"]}
    source_hashes = {}
    for name, (relative, column) in SOURCES.items():
        path = root / relative
        source_hashes[relative] = checked(path, old_hashes[relative])
        values = pd.read_csv(path, usecols=[UID, column], dtype={UID: str}).set_index(UID)
        if not values.index.is_unique or not set(frame.index).issubset(values.index):
            raise ValueError(f"Invalid label-source identity: {name}")
        raw = values.loc[frame.index, column].to_numpy(float)
        if not np.isfinite(raw).all() or not ((raw >= 0) & (raw <= 1)).all():
            raise ValueError(f"Invalid source values: {name}")
        frame[name + "_raw_soft"] = raw
        frame[name + "_weak_binary"] = np.rint(raw.astype(np.float32)).astype(int)
        frame[name + "_exact_half_float32"] = raw.astype(np.float32) == .5
    if not np.array_equal(frame.dread.to_numpy(float), frame.dread_raw_soft.to_numpy(float)):
        raise ValueError("Sealed Dread values differ from original source")
    if not np.array_equal(frame.qwen.to_numpy(float), frame.qwen_raw_v2_raw_soft.to_numpy(float)):
        raise ValueError("Sealed Qwen values differ from original source")
    frame["review_phase"] = np.where(frame.case_id <= "MCL-020", "pilot20", "continuation100")

    def table(subset, group_columns):
        records = []
        grouped = [((), subset)] if not group_columns else subset.groupby(group_columns, dropna=False, sort=True)
        for key, part in grouped:
            keys = key if isinstance(key, tuple) else (key,)
            grouping = dict(zip(group_columns, keys))
            for source in SOURCES:
                binary = part[source + "_weak_binary"]
                soft = part[source + "_raw_soft"]
                records.append({**grouping, "source": source, "studies": len(part),
                    "weak_positive": int(binary.sum()), "weak_negative": int((binary == 0).sum()),
                    "exact_half_float32": int(part[source + "_exact_half_float32"].sum()),
                    "raw_soft_min": float(soft.min()), "raw_soft_median": float(soft.median()), "raw_soft_max": float(soft.max())})
        return records

    negative = frame[frame.decision == "report_negative_candidate"]
    unknown = frame[frame.decision == "unknown"]
    high = frame[frame.high_grade_without_acute]
    dread_disagreement = negative[negative.dread_weak_binary == 1]
    outputs = {
        "by_decision.csv": table(frame, ["decision"]),
        "negative_by_basis.csv": table(negative, ["negative_basis"]),
        "negative_by_stratum.csv": table(negative, ["selection_stratum"]),
        "stratum_decision_source.csv": table(frame, ["selection_stratum", "decision"]),
        "by_review_phase.csv": table(frame, ["review_phase", "decision"]),
    }
    for filename, rows in outputs.items():
        if any(row["weak_positive"] + row["weak_negative"] != row["studies"] for row in rows):
            raise ValueError("Source-count partition mismatch")
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    # Distinct audit schema: no MCL training-target column and no model prediction.
    frame.drop(columns=["dread", "qwen"]).to_csv(out / "case_comparison.csv", index=True)
    dread_disagreement.drop(columns=["dread", "qwen"]).to_csv(out / "dread_positive_report_negative_cases.csv", index=True)
    summary = {"observed_at": now(), "first_sealed_access_at": first_sealed_access,
        "scope": "Post-freeze local weak-label comparison of120 purposively selected train Reports; not clinical accuracy/prevalence/model validation",
        "binary_contract": "Read raw soft values unchanged; convert float32 then numpy.rint; exact0.5 becomes0",
        "cases": len(frame), "negative_candidates": len(negative), "unknown": len(unknown),
        "unmentioned": int((frame.decision == "not_mentioned").sum()), "positive_candidates": int((frame.decision == "report_positive_candidate").sum()),
        "negative_candidates_by_source": table(negative, []), "unknown_by_source": table(unknown, []),
        "high_grade_without_acute_cases": len(high), "high_grade_without_acute_by_source": table(high, []),
        "negative_basis_by_source": outputs["negative_by_basis.csv"],
        "stratum_decision_counts": [{"selection_stratum": key[0], "decision": key[1], "studies": len(part)}
                                    for key, part in frame.groupby(["selection_stratum", "decision"], sort=True)],
        "dread_positive_report_negative": {"count": len(dread_disagreement), "case_ids": list(dread_disagreement.case_id),
            "raw_soft_values": {row.case_id: row.dread_raw_soft for _, row in dread_disagreement.iterrows()}},
        "exact_half_counts_all120": {source: int(frame[source + "_exact_half_float32"].sum()) for source in SOURCES},
        "raw_soft_counts_negative_candidates": {source: {str(k): int(v) for k,v in negative[source + "_raw_soft"].value_counts().sort_index().items()} for source in SOURCES},
        "selection": {"seed": selection["seed"], "predefined_quotas": selection["stratum_precedence_and_quotas"],
                       "eligible_studies": selection["eligible_studies"], "prior_seen_excluded": selection["conservatively_excluded_previous_seen"]},
        "source_sha256": source_hashes, "sealed_sha256": sealed_hashes,
        "annotations_sha256": freeze["combined120_sha256"], "tool_sha256": sha(__file__),
        "definition_source": old_audit["host_definition"],
        "limitations": ["Source-dependent strata and purposive sampling preclude population error-rate inference",
                        "Unknown/not-mentioned are never coded as negative in this audit",
                        "No positive text candidate; binary sensitivity/AUC cannot be evaluated",
                        "Weak soft values are preserved, not calibrated against this text review",
                        "No annotation, existing label, model, prediction or training process changed"]}
    write_json(out / "audit.json", summary)
    for path, expected in locks.items():
        checked(path, expected)
    for relative, expected in source_hashes.items():
        checked(root / relative, expected)
    write_json(out / "verification.json", {"verified_at": now(), "status": "PASS", "annotation_protocol_locks_unchanged": len(locks),
        "source_label_files_unchanged": len(source_hashes), "sealed_to_source_match": "Dread/Qwen exact",
        "all_case_ids_unique": 120, "count_partitions": "PASS", "annotations_after_unblind_edited": False,
        "output_sha256": {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()}})
    return {"negative_candidates_by_source": summary["negative_candidates_by_source"],
            "dread_positive_report_negative": summary["dread_positive_report_negative"],
            "unknown_by_source": summary["unknown_by_source"], "high_grade_without_acute_by_source": summary["high_grade_without_acute_by_source"],
            "out": str(out)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT)); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.out), ensure_ascii=False, indent=2))
