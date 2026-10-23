"""Create a blinded report-only MCL review packet; never export training labels.

Build reads existing numeric retrieval cues, never model prediction columns.
Annotation validation reads only the blinded packet/schema, never sealed mapping.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UID = "StudyInstanceUID"
SEED = "mcl-report-review-2026-10-03-v1"
PRIOR = "docs/handoffs/evidence/2026-10/2026-10-03-mcl-definition"
QUOTAS = [("high_grade_or_complete", 12), ("low_grade_or_remote", 30),
          ("unspecified_grade_ii_or_partial", 20), ("source_disagreement", 30),
          ("explicit_negation_control", 14), ("unmentioned_control", 14)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_lines(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def enum(values):
    return {"type": "string", "enum": values.split("|")}


def schema():
    span = {"type": "object", "additionalProperties": False,
            "required": ["start", "end", "quote"], "properties": {
                "start": {"type": "integer", "minimum": 0}, "end": {"type": "integer", "minimum": 1},
                "quote": {"type": "string", "minLength": 1}}}
    properties = {
        "anatomy": enum("MCL_unspecified|superficial_MCL|deep_MCL|other_or_uncertain"),
        "location": {"type": "string"}, "finding": {"type": "string", "minLength": 1},
        "assertion": enum("affirmed|negated|uncertain|historical|conditional"),
        "severity": enum("normal|low_grade|high_grade_partial|complete|partial_unspecified|grade_II_unspecified|unspecified"),
        "temporality": enum("acute|chronic_remote_healed|acute_on_chronic|unspecified"),
        "fiber_disruption": enum("present|absent|uncertain|not_reported"),
        "edema": enum("present|absent|uncertain|not_reported"),
        "evidence": {"type": "array", "minItems": 1, "items": span},
        "interpretation_note": {"type": "string"}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "MCL report-interpretation v1",
            "type": "object", "additionalProperties": False,
            "required": ["case_id", "schema_version", "reviewer", "language", "language_confidence", "decision",
                         "negative_basis", "components", "contradiction", "needs_language_review", "rationale"],
            "properties": {
                "case_id": {"type": "string", "pattern": "^MCL-[0-9]{3}$"},
                "schema_version": {"const": "1.0"}, "reviewer": {"type": "string", "minLength": 1},
                "language": {"type": "string", "minLength": 1}, "language_confidence": enum("high|medium|low"),
                "decision": enum("report_positive_candidate|report_negative_candidate|unknown|not_mentioned|uninterpretable"),
                "negative_basis": enum("not_applicable|explicit_normal_or_intact|explicit_low_grade|chronic_remote_only|multiple_explicit_exclusions"),
                "components": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                    "required": list(properties), "properties": properties}},
                "contradiction": {"type": "boolean"}, "needs_language_review": {"type": "boolean"},
                "rationale": {"type": "string", "minLength": 1}}}


def build(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; preserve frozen packets")
    started = datetime.now().astimezone().isoformat()
    paths = {"train": root / "storage/competition/train.csv",
             "folds": root / "artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv",
             "features": root / PRIOR / "cohort_features.csv",
             "manual": root / PRIOR / "manual_review.json",
             "top10": root / PRIOR / "r1_top10_weak_errors.csv",
             "prior_queue": root / PRIOR / "review_cases.csv",
             "corruption": root / "artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/excluded.csv"}
    train = pd.read_csv(paths["train"], usecols=[UID, "Report", "MCL"], dtype={UID: str}).set_index(UID)
    folds = pd.read_csv(paths["folds"], dtype={UID: str}).set_index(UID)
    columns = [UID, "report_group_sha256", "dread", "qwen", "qwen_negated", "qwen_mentioned",
               "explicit_high_grade_or_complete_cue", "explicit_low_grade_cue", "chronic_old_healed_cue",
               "grade_ii_cue", "partial_tear_cue", "mcl_clause_found"]
    features = pd.read_csv(paths["features"], usecols=columns, dtype={UID: str}).set_index(UID)
    for name, frame in (("train", train), ("folds", folds), ("features", features)):
        if not frame.index.is_unique or frame.index.isna().any():
            raise ValueError(f"Invalid study identity in {name}")
    gold = set(train.index[train.MCL.notna()])
    if len(gold) != 58 or len(features) != 4345:
        raise ValueError("Unexpected source cohort")
    prior_manual = json.loads(paths["manual"].read_text(encoding="utf-8"))
    seen = {case[UID] for case in prior_manual["cases"]}
    manual_ids = set(seen)
    for name in ("top10", "prior_queue"):
        seen.update(pd.read_csv(paths[name], usecols=[UID], dtype={UID: str})[UID])
    corrupt = set(pd.read_csv(paths["corruption"], usecols=[UID], dtype={UID: str})[UID])
    gold_groups = set(folds.loc[folds.index.intersection(gold), "report_group_sha256"])
    prior_groups = set(folds.loc[folds.index.intersection(seen), "report_group_sha256"])
    exclusion = gold | corrupt | seen | set(folds.index[folds.is_gold | folds.quarantined_gold_report_duplicate | (folds.fold < 0)])
    eligible = features.loc[~features.index.isin(exclusion) & ~features.report_group_sha256.isin(gold_groups | prior_groups)].copy()
    eligible = eligible.join(train[["Report"]])
    if eligible.Report.isna().any() or not (eligible.report_group_sha256 == folds.loc[eligible.index, "report_group_sha256"]).all():
        raise ValueError("Missing report or split group mismatch")
    for column in ("dread", "qwen"):
        values = eligible[column].to_numpy(float)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Invalid source probabilities")
    # Keep the original rint contract, including exact-half negatives.
    disagreement = np.rint(eligible.dread.to_numpy(np.float32)) != np.rint(eligible.qwen.to_numpy(np.float32))
    conditions = {
        "high_grade_or_complete": eligible.explicit_high_grade_or_complete_cue,
        "low_grade_or_remote": eligible.explicit_low_grade_cue | eligible.chronic_old_healed_cue,
        "unspecified_grade_ii_or_partial": eligible.grade_ii_cue | eligible.partial_tear_cue,
        "source_disagreement": pd.Series(disagreement, index=eligible.index),
        "explicit_negation_control": eligible.qwen_negated == 1,
        "unmentioned_control": (eligible.qwen_mentioned == 0) & ~eligible.mcl_clause_found}
    assigned, selected, selected_groups, selected_texts, populations = set(), [], set(), set(), {}
    for stratum, quota in QUOTAS:
        candidates = [uid for uid in eligible.index[conditions[stratum]] if uid not in assigned]
        assigned.update(candidates)  # Mutually exclusive strata fixed by precedence, not successful draws.
        populations[stratum] = {"studies": len(candidates), "groups": int(eligible.loc[candidates, "report_group_sha256"].nunique()), "quota": quota}
        candidates.sort(key=lambda uid: digest(f"{SEED}|draw|{stratum}|{uid}"))
        draws = []
        for uid in candidates:
            group = eligible.loc[uid, "report_group_sha256"]
            text_hash = digest(" ".join(eligible.loc[uid, "Report"].split()))
            if group in selected_groups or text_hash in selected_texts:
                continue
            draws.append((uid, stratum)); selected_groups.add(group); selected_texts.add(text_hash)
            if len(draws) == quota:
                break
        if len(draws) != quota:
            raise ValueError(f"Insufficient distinct reports in {stratum}; do not silently refill")
        selected.extend(draws)
    selected.sort(key=lambda item: digest(f"{SEED}|blind-order|{item[0]}"))
    packets, mapping = [], []
    for i, (uid, stratum) in enumerate(selected, 1):
        case_id = f"MCL-{i:03d}"; row = eligible.loc[uid]
        packets.append({"case_id": case_id, "report": row.Report})
        mapping.append({"case_id": case_id, UID: uid, "report_group_sha256": row.report_group_sha256,
                        "report_sha256": digest(row.Report), "selection_stratum": stratum,
                        "dread": float(row.dread), "qwen": float(row.qwen), "fold": int(folds.loc[uid, "fold"])})
    if len(packets) != 120 or set(item[UID] for item in mapping) & exclusion:
        raise ValueError("Packet size/exclusion failure")
    output.mkdir(parents=True); (output / "sealed").mkdir()
    write_lines(output / "packet.jsonl", packets)
    write_lines(output / "pilot20.jsonl", packets[:20])
    write_json(output / "annotation_schema.json", schema())
    write_lines(output / "sealed/mapping.jsonl", mapping)
    write_json(output / "sealed/selection_manifest.json", {
        "seed": SEED, "stratum_precedence_and_quotas": QUOTAS, "populations": populations,
        "eligible_studies": len(eligible), "eligible_report_groups": int(eligible.report_group_sha256.nunique()),
        "previous_manual_cases": len(manual_ids), "conservatively_excluded_previous_seen": len(seen),
        "prior_report_groups": len(prior_groups), "selected_groups": len(selected_groups),
        "selected_normalized_reports": len(selected_texts), "exclusion_uids": sorted(exclusion),
        "exclusion_gold_groups": sorted(gold_groups), "exclusion_prior_groups": sorted(prior_groups),
        "limits": ["Purpose-stratified development text audit, not population accuracy/prevalence", "Existing folds/model exposure remain", "Report grouping is not patient independence"]})
    hashes = {str(path.relative_to(output)).replace("\\", "/"): sha(path) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "manifest.json", {
        "created_at": started, "schema_version": "1.0", "seed": SEED,
        "purpose": "Blinded report interpretation development packet; no clinical adjudication, independent model validation, or training-label export",
        "reviewer_access": ["packet.jsonl", "pilot20.jsonl", "annotation_schema.json"],
        "do_not_read_during_annotation": "sealed/", "sealed_is_access_convention_not_encryption": True,
        "packet_cases": 120, "pilot_cases": 20, "pilot_rule": "First20 from independently hashed blinded order; frozen before opening Reports",
        "selection_note": "Predefined text/source-disagreement strata, one case per report group; per-case selection layer hidden",
        "exclusion_checks": {"gold_uid_overlap": 0, "gold_report_group_overlap": 0,
            "prior_manual_uid_overlap": 0, "prior_seen_report_group_overlap": 0, "corrupt_uid_overlap": 0,
            "duplicate_selected_report_groups": 0, "duplicate_selected_normalized_reports": 0},
        "input_sha256": {str(path.relative_to(root)).replace("\\", "/"): sha(path) for path in paths.values()},
        "tool_sha256": sha(__file__), "files_sha256": hashes})
    return {"output": str(output), "cases": 120, "pilot": 20, "eligible_studies": len(eligible),
            "prior_seen_excluded": len(seen), "gold_overlap": 0, "group_duplicates": 0, "schema_version": "1.0"}


def validate_schema(value, definition, path="annotation"):
    kind = definition.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}
    if kind and (not isinstance(value, types[kind]) or (kind == "integer" and isinstance(value, bool))):
        raise ValueError(f"{path}: expected {kind}")
    if "const" in definition and value != definition["const"]:
        raise ValueError(f"{path}: const mismatch")
    if "enum" in definition and value not in definition["enum"]:
        raise ValueError(f"{path}: unknown enum")
    if kind == "object":
        if set(definition.get("required", [])) - set(value):
            raise ValueError(f"{path}: missing required keys")
        properties = definition.get("properties", {})
        if definition.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path}: unexpected keys")
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], f"{path}.{key}")
    elif kind == "array":
        if len(value) < definition.get("minItems", 0):
            raise ValueError(f"{path}: too few items")
        for i, item in enumerate(value):
            validate_schema(item, definition["items"], f"{path}[{i}]")
    elif kind == "string":
        if len(value) < definition.get("minLength", 0) or ("pattern" in definition and not re.search(definition["pattern"], value)):
            raise ValueError(f"{path}: invalid string")
    elif kind == "integer" and value < definition.get("minimum", value):
        raise ValueError(f"{path}: below minimum")


def validate_annotations(packet_path, schema_path, annotations_path, require_count):
    # Deliberately cannot resolve a sealed path or use identity/source labels.
    packets = read_lines(packet_path)
    if any(set(row) != {"case_id", "report"} for row in packets):
        raise ValueError("Blinded packet leaks fields")
    reports = {row["case_id"]: row["report"] for row in packets}
    if len(reports) != len(packets):
        raise ValueError("Duplicate case IDs")
    rows = read_lines(annotations_path); definition = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if len(rows) != require_count or len({row.get("case_id") for row in rows}) != len(rows):
        raise ValueError("Annotation count or uniqueness mismatch")
    decisions, total_spans = {}, 0
    for row in rows:
        validate_schema(row, definition)
        if row["case_id"] not in reports:
            raise ValueError("Annotation case outside packet")
        report = reports[row["case_id"]]
        for component in row["components"]:
            for span in component["evidence"]:
                if not (0 <= span["start"] < span["end"] <= len(report)) or report[span["start"]:span["end"]] != span["quote"]:
                    raise ValueError(f"{row['case_id']}: evidence mismatch")
                total_spans += 1
        decision = row["decision"]
        if decision == "report_positive_candidate":
            qualifying = [c for c in row["components"] if c["anatomy"] != "other_or_uncertain" and c["assertion"] == "affirmed"
                          and c["severity"] in {"high_grade_partial", "complete"} and c["temporality"] == "acute"
                          and c["fiber_disruption"] != "absent" and c["edema"] != "absent"]
            if not qualifying or row["contradiction"] or row["needs_language_review"]:
                raise ValueError(f"{row['case_id']}: positive candidate lacks explicit qualifying component")
        if decision == "report_negative_candidate":
            if row["negative_basis"] == "not_applicable" or not row["components"] or row["contradiction"] or row["needs_language_review"]:
                raise ValueError(f"{row['case_id']}: negative lacks resolved evidence")
            unresolved = [c for c in row["components"] if c["anatomy"] != "other_or_uncertain" and
                          (c["assertion"] in {"uncertain", "conditional"} or
                           (c["severity"] in {"grade_II_unspecified", "partial_unspecified", "high_grade_partial", "complete"}
                            and c["temporality"] != "chronic_remote_healed" and c["assertion"] != "negated"))]
            if unresolved:
                raise ValueError(f"{row['case_id']}: unresolved potentially qualifying MCL finding")
        if decision != "report_negative_candidate" and row["negative_basis"] != "not_applicable":
            raise ValueError("Negative basis on nonnegative decision")
        if decision == "not_mentioned" and row["components"]:
            raise ValueError("Not mentioned case has assertions")
        decisions[decision] = decisions.get(decision, 0) + 1
    return {"observed_at": datetime.now().astimezone().isoformat(), "status": "PASS", "cases": len(rows),
            "exact_evidence_spans": total_spans, "decisions": decisions, "sealed_mapping_read": False,
            "packet_sha256": sha(packet_path), "schema_sha256": sha(schema_path), "annotations_sha256": sha(annotations_path),
            "tool_sha256": sha(__file__), "limitation": "Mechanical checks, not proof of correct language or clinical interpretation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("build"); create.add_argument("--root", default=str(ROOT)); create.add_argument("--out", required=True)
    check = sub.add_parser("validate"); check.add_argument("--packet", required=True); check.add_argument("--schema", required=True)
    check.add_argument("--annotations", required=True); check.add_argument("--count", type=int, required=True); check.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.root, args.out)
    else:
        if Path(args.receipt).exists():
            raise FileExistsError("Preserve prior validation receipt")
        result = validate_annotations(args.packet, args.schema, args.annotations, args.count)
        write_json(args.receipt, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
