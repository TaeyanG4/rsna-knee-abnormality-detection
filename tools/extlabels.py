"""EXP136 offline external label conversion. Standard library only; no image IO.

Mapping contract: artifacts/experiments/EXP136_EXTLABELS/PROTOCOL.md.
An empty target cell means unassessable, NOT a negative. Existing outputs are
never replaced. This module has no import-time actions and never trains.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture")
PROTOCOL = Path("artifacts/experiments/EXP136_EXTLABELS/PROTOCOL.md")
MOAKS_ORDER = ("22", "30", "65", "63A", "63B", "63C", "63D", "63E", "63F", "61")
XR_ORDER = ("15", "37", "42", "43")
VISIT_MONTH = {0: "V00", 12: "V01", 24: "V03", 36: "V05", 48: "V06",
               60: "V07", 72: "V08", 84: "V09", 96: "V10", 108: "V11", 120: "V12"}
CART_REGIONS = {"Medial OA": ("FMC", "FMP", "TMA", "TMC", "TMP"),
                "Lateral OA": ("FLC", "FLP", "TLA", "TLC", "TLP"),
                "PF OA": ("PM", "PL", "FMA", "FLA")}
OST_REGIONS = {"Medial OA": ("FMC", "FMP", "TM"),
               "Lateral OA": ("FLC", "FLP", "TL"),
               "PF OA": ("PS", "PI", "PM", "PL", "FMA", "FLA")}
POMA_ALIASES = {
    "MACLTR": "OTHERACLTEAR", "MACLRP": "OTHERACLREPAIR",
    "MACLBML": "OTHERACLBML", "MEFFWK": "OTHEREFFUSIONWHOLE",
    "MSYIC": "OTHERSYNOVITISINTER", "MCMPM": "CARTPATELLAMED",
    "MCMPL": "CARTPATELLALAT", "MCMFMA": "CARTFEMURANTMED",
    "MCMFLA": "CARTFEMURANTLAT", "MCMFMC": "CARTTIBIOFEMURCENTMED",
    "MCMFLC": "CARTTIBIOFEMURCENTLAT", "MCMFMP": "CARTTIBIOFEMURPOSTMED",
    "MCMFLP": "CARTTIBIOFEMURPOSTLAT",
}
for _short, _long in (("A", "ANT"), ("C", "CENT"), ("P", "POST")):
    for _side, _word in (("M", "MED"), ("L", "LAT")):
        POMA_ALIASES[f"MCMT{_side}{_short}"] = f"CARTTIBIOTIBIA{_long}{_word}"
for _short, _long in (("A", "ANT"), ("B", "BODY"), ("P", "POST")):
    for _side, _word in (("M", "MED"), ("L", "LAT")):
        POMA_ALIASES[f"MMT{_side}{_short}"] = f"MENMOR{_long}{_word}"
for _side, _word in (("M", "MED"), ("L", "LAT")):
    POMA_ALIASES[f"MMRT{_side}"] = f"MENROOTPOST{_word}"
    POMA_ALIASES[f"MMRTA{_side}"] = f"MENROOTANT{_word}"

FMRI_LABELS = {
    "Cartilage - Partial Thickness loss/defect": "cartilage_partial",
    "Cartilage - Full Thickness loss/defect": "cartilage_full",
    "Bone- Subchondral edema": "subchondral_edema",
    "Ligament - MCL Low-Mod Grade Sprain": "mcl_low_mod",
    "Ligament - MCL High Grade sprain": "mcl_high",
    "Meniscus Tear": "meniscus_tear", "Periarticular cysts": "periarticular_cysts",
    "Joint Effusion": "effusion", "Displaced Meniscal Tissue": "displaced_meniscal_tissue",
    "Ligament - ACL Low Grade sprain": "acl_low",
    "Ligament - ACL High Grade Sprain": "acl_high",
    "LCL Complex - Low-Mod Grade Sprain": "lcl_low_mod",
    "LCL Complex- High Grade Sprain": "lcl_high",
    "Ligament - PCL Low-Mod grade sprain": "pcl_low_mod",
    "Ligament - PCL High Grade": "pcl_high",
    "Bone-Fracture/Contusion/dislocation": "fracture_contusion_dislocation",
    "Muscle Strain": "muscle_strain", "Bone - Lesion": "bone_lesion",
    "Soft Tissue Lesion": "soft_tissue_lesion", "Joint Bodies": "joint_bodies",
    "Patellar Retinaculum - High grade sprain": "patellar_retinaculum_high",
    "artifact": "artifact",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code(value: Any) -> float | None:
    """Decode a SAS formatted grade without treating missing/sentinel as disease."""
    if value is None:
        return None
    text = str(value).strip().split(":", 1)[0].strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None
    number = float(text)
    return number if math.isfinite(number) and number >= 0 else None


def grade(value: Any, allowed: Iterable[float]) -> float | None:
    number = code(value)
    return number if number in allowed else None


def threshold(value: Any, allowed: Iterable[float], minimum: float) -> int | None:
    number = grade(value, allowed)
    return None if number is None else int(number >= minimum)


def any_assessed(values: Iterable[int | None]) -> int | None:
    values = list(values)
    if 1 in values:
        return 1
    return 0 if values and all(v == 0 for v in values) else None


def empty_targets() -> dict[str, int | None]:
    return dict.fromkeys(TARGETS)


def oai_visit(value: Any) -> str:
    value = str(value).strip().upper().split(":", 1)[0].strip()
    if value == "BL":
        return "V00"
    if re.fullmatch(r"M\d+", value):
        if int(value[1:]) not in VISIT_MONTH:
            raise ValueError("unrecognized OAI month")
        return VISIT_MONTH[int(value[1:])]
    if re.fullmatch(r"V\d{2}", value):
        return value
    if re.fullmatch(r"\d{1,2}(?:\.0)?", value):
        return f"V{int(float(value)):02d}"
    raise ValueError("unrecognized actual OAI visit (never use iwave/tmpt)")


def oai_key(participant: Any, side: Any, visit: Any) -> str:
    participant = str(participant).strip()
    if not re.fullmatch(r"\d+", participant):
        raise ValueError("invalid OAI participant key")
    side = str(side).strip().upper().split(":", 1)[0].strip()
    side = {"1": "R", "1.0": "R", "RIGHT": "R", "2": "L", "2.0": "L", "LEFT": "L"}.get(side, side)
    if side not in ("R", "L"):
        raise ValueError("invalid OAI side")
    return f"OAI_{participant}_{side}_{oai_visit(visit)}"


def cartilage_parts(value: Any) -> tuple[int, int] | None:
    number = code(value)
    if number is None:
        return None
    scaled = number * 10
    if abs(scaled - round(scaled)) > 1e-7:
        return None
    area, full = divmod(round(scaled), 10)
    return (area, full) if 0 <= full <= area <= 3 else None


def moaks_oa(scores: dict[str, Any], target: str) -> int | None:
    cartilage = [cartilage_parts(scores.get("MCM" + r)) for r in CART_REGIONS[target]]
    osteophytes = [grade(scores.get("MOS" + r), range(4)) for r in OST_REGIONS[target]]
    if any(p is not None and p[1] > 0 for p in cartilage) and any(
            v is not None and v >= 1 for v in osteophytes):
        return 1
    if all(p == (0, 0) for p in cartilage) and all(v == 0 for v in osteophytes):
        return 0
    return None


def map_moaks(scores: dict[str, Any], *, poma: bool = False) -> dict[str, int | None]:
    result = empty_targets()
    result["ACL"] = threshold(scores.get("MACLTR"), range(3), 1)
    result["Effusion"] = threshold(scores.get("MEFFWK"), range(4), 2)
    result["Synovitis"] = threshold(scores.get("MSYIC"), range(4), 2)
    result["Baker's"] = threshold(scores.get("MPOPCYS"), range(2), 1)
    for side, name in (("M", "Medial Meniscus"), ("L", "Lateral Meniscus")):
        findings = [threshold(scores.get(f"MMT{side}{part}"), range(9), 2) for part in ("A", "B", "P")]
        findings.append(threshold(scores.get(f"MMRT{side}"), range(2), 1))
        if poma:
            findings.append(threshold(scores.get(f"MMRTA{side}"), range(2), 1))
        result[name] = any_assessed(findings)
    for name in CART_REGIONS:
        result[name] = moaks_oa(scores, name)
    return result


def map_xray(scores: dict[str, Any]) -> dict[str, int | None]:
    result = empty_targets()
    kl = grade(scores.get("XRKL"), range(5))
    # Fractional JSN is retained natively; only >=1 supports a localized positive.
    medial, lateral = code(scores.get("XRJSM")), code(scores.get("XRJSL"))
    medial = medial if medial is not None and medial <= 3 else None
    lateral = lateral if lateral is not None and lateral <= 3 else None
    for target, jsn in (("Medial OA", medial), ("Lateral OA", lateral)):
        if kl is not None and kl >= 2 and jsn is not None and jsn >= 1:
            result[target] = 1
        elif kl == 0 and medial == 0 and lateral == 0:
            result[target] = 0
    return result


def map_worms(scores: dict[str, Any]) -> dict[str, int | None]:
    result = empty_targets()
    for target in ("ACL", "MCL"):
        value = grade(scores.get("MW" + target), range(5))
        result[target] = 0 if value == 0 else 1 if value in (3, 4) else None
    for side, target in (("M", "Medial Meniscus"), ("L", "Lateral Meniscus")):
        result[target] = any_assessed(threshold(scores.get(f"MWMT{side}{part}"), range(5), 2)
                                      for part in ("A", "B", "P"))
    result["Effusion"] = threshold(scores.get("MWEFFWK"), range(4), 2)
    result["Baker's"] = threshold(scores.get("MWPOPCY"), range(4), 2)
    return result


def merge_oa(mri: int | None, xray: int | None) -> tuple[int | None, int]:
    if mri is not None and xray is not None and mri != xray:
        return None, 1
    return (mri if mri is not None else xray), 0


@dataclass
class Table:
    source: str
    labels: list[dict[str, Any]] = field(default_factory=list)
    native: list[dict[str, Any]] = field(default_factory=list)
    inputs: list[Path] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reading:
    study_key: str
    scores: dict[str, Any]
    native: dict[str, Any]
    project: str
    version: str
    filename: str
    reader: str = ""


def read_rows(path: Path, *, delimiter: str = ",", encoding: str = "utf-8-sig"):
    with Path(path).open("r", encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"missing/duplicate columns: {path.name}")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"wrong row width: {path.name}")
            yield {k.strip(): v.strip() for k, v in row.items()}


def read_oai(path: Path, kind: str) -> list[Reading]:
    records = []
    match = re.search(r"(\d{2})\.txt$", path.name, re.I)
    for original in read_rows(path, delimiter="|", encoding="latin-1"):
        row = {k.upper(): v for k, v in original.items()}
        if kind == "poma":
            visit = oai_visit(row["VISIT"] if "VISIT" in row else row.get("WAVE", ""))
            native = {k: code(v) for k, v in row.items() if k.startswith(("CART", "BML", "MEN", "OTHER"))}
            scores = {short: native.get(long) for short, long in POMA_ALIASES.items()}
            project = "INCOA" if "incoa" in path.name.lower() else "TKR"
        else:
            if match is None:
                raise ValueError("OAI visit-specific input filename must end in two digits")
            visit = "V" + match.group(1)
            native = {k[3:]: code(v) for k, v in row.items()
                      if k.startswith(visit) and not any(x in k for x in ("CMNT", "COMMENT", "TECH", "BARCD"))}
            scores = native
            project = row.get("READPRJ", "")
            if not native:
                raise ValueError(f"visit columns absent: {path.name}")
        records.append(Reading(oai_key(row.get("ID", ""), row.get("SIDE", ""), visit), scores,
                               native, project, row.get("VERSION", ""), path.name, row.get("READER", "")))
    return records


def reading_rank(record: Reading, order: tuple[str, ...]) -> tuple:
    rank = order.index(record.project) if record.project in order else len(order)
    version = code(record.version)
    canonical = hashlib.sha256(json.dumps(record.native, sort_keys=True, allow_nan=False).encode()).hexdigest()
    reader_rank = (0 if record.reader == "R1" else 1 if record.reader == "R2" else 2)
    return rank, record.project, reader_rank, record.reader, -(version if version is not None else -1), record.filename, canonical


def select_readings(records: list[Reading], order: tuple[str, ...]) -> tuple[dict[str, Reading], dict[str, int]]:
    selected, counts = {}, {}
    for record in records:
        key = record.study_key
        counts[key] = counts.get(key, 0) + 1
        if key not in selected or reading_rank(record, order) < reading_rank(selected[key], order):
            selected[key] = record
    return selected, counts


def build_oai(source: Path) -> Table:
    mr = source / "clinical/extracted/mr_assess/Semi-Quant Scoring_ASCII"
    xr = source / "clinical/extracted/xray/X-Ray Image Assessments_ASCII/XR Knee Semi-Quant"
    buckets: dict[str, list[Reading]] = {k: [] for k in ("moaks", "poma", "xray", "worms", "bicl", "bloksbml")}
    paths = []
    patterns = ((mr, r"kmri_sq_moaks_bicl\d{2}\.txt", "moaks"),
                (mr, r"kmri_poma_(incoa|tkr)_moaks_bicl\.txt", "poma"),
                (xr, r"kxr_sq_bu\d{2}\.txt", "xray"),
                (mr, r"kmri_sq_worms_link\d{2}\.txt", "worms"),
                (mr, r"kmri_sq_bicl\d{2}\.txt", "bicl"),
                (mr, r"kmri_sq_blksbml_bicl\d{2}\.txt", "bloksbml"))
    for directory, pattern, kind in patterns:
        for path in sorted(directory.glob("*.txt")):
            if re.fullmatch(pattern, path.name, re.I):
                buckets[kind].extend(read_oai(path, kind))
                paths.append(path)
    if not paths:
        raise FileNotFoundError("no OAI assessment tables present")
    chosen, counts = {}, {}
    for kind, records in buckets.items():
        order = MOAKS_ORDER if kind == "moaks" else XR_ORDER if kind == "xray" else ("INCOA", "TKR") if kind == "poma" else ()
        chosen[kind], counts[kind] = select_readings(records, order)
    table = Table("OAI", inputs=paths)
    lesion_records: dict[str, list[Reading]] = {}
    for record in buckets["bloksbml"]:
        lesion_records.setdefault(record.study_key, []).append(record)
    all_keys = sorted(set().union(*(set(x) for x in chosen.values())))
    for key in all_keys:
        native: dict[str, Any] = {"study_key": key}
        for kind in buckets:
            if kind == "bloksbml" and key in lesion_records:
                # This table has one row per lesion, not competing whole-knee reads.
                # Retain ALL lesion assessments; a reader-selection maximum would
                # silently throw away the remaining lesion features.
                records = lesion_records[key]
                native["bloksbml_lesion_record_count"] = len(records)
                native["bloksbml_lesion_records_json"] = json.dumps([
                    {"scores": r.native, "readprj": r.project, "version": r.version,
                     "source_file": r.filename} for r in sorted(records, key=lambda r: reading_rank(r, ()))
                ], sort_keys=True, allow_nan=False, separators=(",", ":"))
                continue
            if key in chosen[kind]:
                record = chosen[kind][key]
                native.update({f"{kind}_{name}": value for name, value in record.native.items()})
                native.update({f"{kind}_readprj": record.project, f"{kind}_version": record.version,
                               f"{kind}_reader": record.reader, f"{kind}_source_file": record.filename,
                               f"{kind}_reading_count": counts[kind][key]})
        if key in chosen["moaks"]:
            result = map_moaks(chosen["moaks"][key].scores)
            native["mapped_mri_source"] = "moaks"
        elif key in chosen["poma"]:
            result = map_moaks(chosen["poma"][key].scores, poma=True)
            native["mapped_mri_source"] = "poma"
        else:
            result = empty_targets()
            native["mapped_mri_source"] = "none"
        if key in chosen["worms"]:
            supplemental = map_worms(chosen["worms"][key].scores)
            for target in TARGETS:
                if result[target] is None and supplemental[target] is not None:
                    result[target] = supplemental[target]
                    native["worms_supplied_" + target] = 1
        if key in chosen["xray"]:
            radiographic = map_xray(chosen["xray"][key].scores)
            for target in ("Medial OA", "Lateral OA"):
                native["mri_" + target] = result[target]
                native["xray_" + target] = radiographic[target]
                result[target], native["discordant_" + target] = merge_oa(result[target], radiographic[target])
        table.labels.append({"study_key": key, **result})
        table.native.append(native)
    table.notes = {"input_readings": {k: len(v) for k, v in buckets.items()},
                   "selected_readings": {k: len(v) for k, v in chosen.items()},
                   "mri_oa_xray_disagreements": {t: sum(r.get("discordant_" + t, 0) for r in table.native)
                                                for t in ("Medial OA", "Lateral OA")},
                   "image_availability": "unfiltered metadata; join validated EXP135 cache by study_key"}
    return table


def fmri_stem(value: str) -> str:
    value = value.strip()
    if value.lower().endswith(".h5"):
        value = value[:-3]
    if not re.fullmatch(r"file\d+", value):
        raise ValueError("invalid fastMRI filename (bare fileNNN or fileNNN.h5 required)")
    return value


def build_fastmri(source: Path) -> Table:
    annotations, filelist = source / "Annotations/knee.csv", source / "Annotations/knee_file_list.csv"
    files = set()
    with filelist.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream) if row and any(cell.strip() for cell in row)]
        # The upstream knee list ends with the literal one-cell footer 'Knee data'.
        # It has no header: file1000000 on the first line is a reviewed volume.
        if rows and len(rows[-1]) == 1 and rows[-1][0].strip() == "Knee data":
            rows.pop()
        for number, row in enumerate(rows):
            if len(row) != 1:
                raise ValueError("fastMRI reviewed file list must have one column")
            if number == 0 and row[0].strip().lower() in ("file", "filename"):
                continue
            stem = fmri_stem(row[0])
            if stem in files:
                raise ValueError("duplicate file in reviewed fastMRI file list")
            files.add(stem)
    if not files:
        raise ValueError("empty fastMRI reviewed population")
    lookup = {" ".join(k.lower().split()): v for k, v in FMRI_LABELS.items()}
    counts: dict[str, dict[str, int]] = {name: {} for name in files}
    for row in read_rows(annotations):
        stem = fmri_stem(row["file"])
        if stem not in files:
            raise ValueError("fastMRI annotation outside reviewed file list")
        label = lookup.get(" ".join(row["label"].lower().split()))
        if label is None:
            raise ValueError("unregistered fastMRI category; review protocol before export")
        counts[stem][label] = counts[stem].get(label, 0) + 1
    table = Table("FASTMRI_PLUS", inputs=[annotations, filelist])
    for stem in sorted(files):
        boxes = counts[stem]
        native = {"study_key": "FMRI_SC_" + stem, **{x: int(boxes.get(x, 0) > 0) for x in FMRI_LABELS.values()}}
        native.update({x + "_annotation_count": boxes.get(x, 0) for x in FMRI_LABELS.values()})
        native.update({"acl_any": int(native["acl_low"] or native["acl_high"]),
                       "mcl_any": int(native["mcl_low_mod"] or native["mcl_high"]),
                       "any_meniscus": int(native["meniscus_tear"] or native["displaced_meniscal_tissue"]),
                       "any_cartilage": int(native["cartilage_partial"] or native["cartilage_full"]),
                       "reviewed": 1, "no_findings": int(not boxes), "bakers_proxy_approximate": 1})
        result = empty_targets()
        result.update({"ACL": native["acl_any"], "MCL": native["mcl_any"], "Effusion": native["effusion"],
                       "Baker's": native["periarticular_cysts"]})
        table.labels.append({"study_key": native["study_key"], **result})
        table.native.append(native)
    table.notes = {"negatives": "no annotation in reviewed volume; not exhaustive clinical negatives",
                   "no_findings_volumes": sum(x["no_findings"] for x in table.native),
                   "weak_proxies": ["ACL/MCL any grade", "Effusion any size", "Baker's any periarticular cyst"]}
    return table


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9-]+", value):
        raise ValueError("unsafe source study identifier")
    return value


def build_kneemri(source: Path) -> Table:
    path = source / "metadata.csv"
    table = Table("KNEEMRI", inputs=[path])
    present = {p.name for p in source.glob("*.pck")} | {p.name for p in source.glob("vol*/*.pck")}
    for row in read_rows(path):
        value = grade(row["aclDiagnosis"], range(3))
        if value is None:
            raise ValueError("unrecognized KneeMRI ACL grade")
        key = f"KMRI_{_identifier(row['examId'])}_{_identifier(row['seriesNo'])}"
        relative = Path(row["volumeFilename"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative) or relative.suffix.lower() != ".pck":
            raise ValueError("unsafe KneeMRI relative volume path")
        available = (source / relative).is_file() if len(relative.parts) > 1 else relative.name in present
        native = {"study_key": key, "acl_grade": int(value), "acl_any": int(value > 0),
                  "acl_complete": int(value == 2), "knee_side": row.get("kneeLR", ""),
                  "image_file_present": int(available), "volume_filename": relative.as_posix()}
        table.native.append(native)
        table.labels.append({"study_key": key, **empty_targets(), "ACL": int(value > 0)})
    table.notes = {"image_files_present": sum(r["image_file_present"] for r in table.native),
                   "image_availability": "all metadata rows retained; no pickle read",
                   "weak_proxies": ["ACL includes partial tears; native acl_complete available"]}
    return table


def read_mrnet(path: Path) -> dict[str, int]:
    values = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 2 or not re.fullmatch(r"\d{1,4}", row[0].strip()):
                raise ValueError("MRNet expects headerless two-column case,binary label")
            key = f"{int(row[0]):04d}"
            label = grade(row[1], range(2))
            if label is None or key in values:
                raise ValueError("invalid/duplicate MRNet label")
            values[key] = int(label)
    if not values:
        raise ValueError("empty MRNet label file")
    return values


def build_mrnet(source: Path) -> Table:
    table = Table("MRNET")
    for split in ("train", "valid"):
        paths = {name: source / f"{split}-{name}.csv" for name in ("abnormal", "acl", "meniscus")}
        exists = [p.is_file() for p in paths.values()]
        if not any(exists):
            continue
        if not all(exists):
            raise ValueError(f"partial MRNet label set for {split}")
        labels = {name: read_mrnet(path) for name, path in paths.items()}
        if any(set(v) != set(labels["acl"]) for v in labels.values()):
            raise ValueError("MRNet source heads do not cover identical cases")
        table.inputs.extend(paths.values())
        for case in sorted(labels["acl"]):
            key = f"MRN_{split}_{case}"
            table.native.append({"study_key": key, "split": split, "abnormal": labels["abnormal"][case],
                                 "acl": labels["acl"][case], "any_meniscus": labels["meniscus"][case]})
            table.labels.append({"study_key": key, **empty_targets(), "ACL": labels["acl"][case]})
    if not table.inputs:
        raise FileNotFoundError("MRNet label csvs absent")
    table.notes = {"meniscus_compartment": "unavailable; keep native any_meniscus",
                   "splits_present": sorted({r["split"] for r in table.native})}
    return table


def validate_table(table: Table) -> None:
    keys = [r["study_key"] for r in table.labels]
    native_keys = [r["study_key"] for r in table.native]
    if not keys or len(set(keys)) != len(keys) or len(set(native_keys)) != len(native_keys) or set(keys) != set(native_keys):
        raise ValueError("empty, duplicate, or misaligned label/native study keys")
    for row in table.labels:
        if set(row) != {"study_key", *TARGETS} or any(row[t] not in (None, 0, 1) for t in TARGETS):
            raise ValueError("external table violates 12-target masked binary contract")


def label_counts(table: Table) -> dict[str, dict[str, int]]:
    validate_table(table)
    return {t: {"positive": sum(r[t] == 1 for r in table.labels),
                "negative": sum(r[t] == 0 for r in table.labels),
                "nan": sum(r[t] is None for r in table.labels)} for t in TARGETS}


def export_table(table: Table, output_dir: Path, protocol: Path) -> dict[str, Any]:
    validate_table(table)
    if not protocol.is_file():
        raise FileNotFoundError("registered EXP136 protocol required before export")
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / f"{table.source}_{suffix}" for suffix in ("12.csv", "native.csv", "receipt.json")]
    if any(p.exists() for p in destinations):
        raise FileExistsError("external output already exists; choose a new version directory")
    # Exclusive creation prevents overwriting a concurrent builder. Receipt is written last.
    for dest, rows, columns in ((destinations[0], table.labels, ["study_key", *TARGETS]),
                                (destinations[1], table.native, ["study_key", *sorted(set().union(*(set(r) for r in table.native)) - {"study_key"})])):
        with dest.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["study_key"]))
    receipt = {"source": table.source, "contract": "EXP136-v1", "rows": len(table.labels),
               "target_order": list(TARGETS), "counts": label_counts(table), "notes": table.notes,
               "protocol_sha256": sha256(protocol), "tool_sha256": sha256(Path(__file__)),
               "inputs_sha256": {str(p): sha256(p) for p in sorted(set(table.inputs))},
               "outputs_sha256": {p.name: sha256(p) for p in destinations[:2]}}
    with destinations[2].open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def check_target_order(root: Path) -> None:
    sample = root / "storage/competition/sample_submission.csv"
    if not sample.is_file():
        raise FileNotFoundError("competition sample_submission header required")
    with sample.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if tuple(header[1:]) != TARGETS:
        raise ValueError("competition target order differs from registered EXP136 order")


BUILDERS = {"oai": (build_oai, "oai"), "fastmri_plus": (build_fastmri, "fastmri_plus/repo"),
            "kneemri": (build_kneemri, "kneemri"), "mrnet": (build_mrnet, "mrnet")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["all", *BUILDERS], default="all")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, help="default: ROOT/storage/labels/external; never overwrite")
    parser.add_argument("--counts-only", action="store_true", help="validate and print aggregates without writing files")
    args = parser.parse_args(argv)
    check_target_order(args.root)
    output = args.output_dir or args.root / "storage/labels/external"
    protocol = args.root / PROTOCOL
    if not protocol.is_file():
        raise FileNotFoundError("registered protocol absent")
    summaries = {}
    for source in BUILDERS if args.source == "all" else (args.source,):
        builder, relative = BUILDERS[source]
        try:
            table = builder(args.root / "storage/external_data" / relative)
        except FileNotFoundError:
            if args.source != "all":
                raise
            summaries[source] = {"status": "absent; no table generated"}
            continue
        if args.counts_only:
            summaries[source] = {"rows": len(table.labels), "counts": label_counts(table), "notes": table.notes}
        else:
            receipt = export_table(table, output, protocol)
            summaries[source] = {k: receipt[k] for k in ("rows", "counts", "notes", "outputs_sha256")}
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
