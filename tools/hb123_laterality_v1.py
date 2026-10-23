"""Header-only side-map builder/validator v1.0.0. Prints aggregates only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.arms import laterality_v1 as lat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ".local/scratch/sol-5/hb123/side-map.json"
ARCHIVE = "storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip"
CORPUS = "storage/cache/derived/RAPTOR64_336_C140_S02"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lat.canonical(value) + b"\n")


def header_dict(dataset):
    result = {}
    for key in lat.TAGS:
        value = dataset.get(key)
        if value is None:
            continue
        if key in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing"):
            result[key] = [float(x) for x in value]
        elif key in ("Rows", "Columns"):
            result[key] = int(value)
        elif key == "PatientOrientation":
            result[key] = list(map(str, value)) if not isinstance(value, str) else value.split("\\")
        else:
            result[key] = str(value)
    return result


def metadata(paths):
    rows, seen = [], set()
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {lat.UID, "SeriesInstanceUID", "Anatomical_Plane"} <= set(reader.fieldnames or []):
                raise ValueError("series metadata schema differs")
            for row in reader:
                key = row[lat.UID], row["SeriesInstanceUID"]
                if key in seen:
                    raise ValueError("duplicate series metadata")
                seen.add(key)
                rows.append({k: row[k] for k in (lat.UID, "SeriesInstanceUID", "Anatomical_Plane")})
    return rows


def build(args):
    import numpy as np
    import pydicom
    started = time.perf_counter()
    rows = metadata(args.metadata)
    ids = np.load(Path(args.corpus) / "ids.npy", allow_pickle=True).astype(str).tolist()
    if args.limit:
        ids = ids[:args.limit]
    wanted = set(ids)
    rows = [r for r in rows if r[lat.UID] in wanted]
    keys = {(r[lat.UID], r["SeriesInstanceUID"]) for r in rows}
    by_series = defaultdict(list)
    studies = defaultdict(list)
    qa = Counter()
    source_digest = hashlib.sha256()
    with zipfile.ZipFile(args.archive) as archive:
        for entry in archive.infolist():
            parts = entry.filename.split("/")
            if len(parts) >= 3 and parts[-1].endswith(".dcm") and (parts[-3], parts[-2]) in keys:
                by_series[parts[-3], parts[-2]].append(entry)
        for i, row in enumerate(rows):
            entries = sorted(by_series[(row[lat.UID], row["SeriesInstanceUID"])], key=lambda x: x.filename)
            headers, errors = [], 0
            picked = sorted({0, len(entries) // 2, len(entries) - 1}) if entries else []
            qa["dicom_instances_indexed"] += len(entries)
            for j in picked:
                entry = entries[j]
                source_digest.update(lat.canonical([entry.filename, entry.CRC, entry.file_size]))
                try:
                    with archive.open(entry) as stream:
                        dataset = pydicom.dcmread(stream, stop_before_pixels=True, specific_tags=lat.TAGS)
                    headers.append(header_dict(dataset))
                except Exception:
                    errors += 1
            summary = lat.summarize_series(headers, row["Anatomical_Plane"], missing=not entries, errors=errors)
            studies[row[lat.UID]].append(summary)
            qa["series"] += 1
            qa["headers_read"] += len(headers)
            qa["header_errors"] += errors
            qa["missing_series"] += not bool(entries)
            qa["unsafe_orientation_series"] += not summary["orientation_safe"]
            qa["patient_orientation_conflicts"] += summary["patient_orientation_conflicts"]
            for tag, count in summary["tag_fields"].items():
                qa["present_" + tag] += count
            if (i + 1) % 2000 == 0:
                print(json.dumps({"series_processed": i + 1, "headers_read": qa["headers_read"]}), flush=True)
    resolved = [{lat.UID: uid, **lat.resolve_study(studies[uid])} for uid in ids]
    for row in resolved:
        qa["side_" + row["side"]] += 1
        qa["reason_" + row["reason"]] += 1
        for key in ("tag_geometry_comparable", "tag_geometry_agree", "tag_geometry_disagree",
                    "direct_tag_conflict", "geometry_conflict"):
            qa[key] += row[key]
    inputs = {str(p): lat.sha(p) for p in [*args.metadata, str(Path(args.corpus) / "ids.npy"),
                                                        str(Path(args.corpus) / "manifest.json")]}
    doc = {"schema": "hb123-side-map/v1", "policy": lat.POLICY, "inputs": inputs,
           "archive": {"path": str(args.archive), "bytes": Path(args.archive).stat().st_size,
                       "sampled_member_directory_sha256": source_digest.hexdigest()},
           "studies": resolved, "scope": "partial-smoke" if args.limit else "all-corpus-ids"}
    write(args.out, doc)
    evidence = {"schema": "hb123-side-map-qa/v1", "policy": lat.POLICY,
                "counts": {"studies": len(ids), **dict(qa)}, "side_map_sha256": lat.sha(args.out),
                "sources": {p: lat.sha(ROOT / p) for p in ("training/arms/laterality_v1.py", "tools/hb123_laterality_v1.py")},
                "elapsed_seconds": time.perf_counter() - started,
                "limitations": ["3 sampled headers per series, not every slice",
                                "patient-coordinate geometry is a header proxy, not anatomical ground truth",
                                "conflicts and unknowns remain unchanged; no labels/reports consulted"]}
    write(Path(args.out).with_suffix(".qa.json"), evidence)
    return {"status": "BUILT", **evidence}


def validate(args):
    import numpy as np
    qa = json.loads(Path(args.side_map).with_suffix(".qa.json").read_bytes())
    sides = lat.SideMap(args.side_map, qa["side_map_sha256"])
    doc = json.loads(Path(args.side_map).read_bytes())
    for path, digest in doc["inputs"].items():
        if lat.sha(path) != digest:
            raise ValueError("side-map input changed")
    for path, digest in qa["sources"].items():
        if lat.sha(ROOT / path) != digest:
            raise ValueError("side-map source changed")
    sides.require(np.load(Path(args.corpus) / "ids.npy", allow_pickle=True).astype(str))
    return {"status": "VALIDATED", "side_map_sha256": sides.sha256, "counts": qa["counts"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--metadata", nargs="+", default=["storage/competition/train_series.csv", "storage/competition/test_series.csv"])
    p.add_argument("--archive", default=ARCHIVE)
    p.add_argument("--corpus", default=CORPUS)
    p.add_argument("--out", default=DEFAULT)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("validate")
    p.add_argument("--side-map", default=DEFAULT)
    p.add_argument("--corpus", default=CORPUS)
    args = ap.parse_args(argv)
    print(json.dumps(build(args) if args.command == "build" else validate(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
