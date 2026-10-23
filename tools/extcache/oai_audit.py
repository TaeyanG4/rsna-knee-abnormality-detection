"""H-B-88 header-only OAI investigation; never writes GENERAL/PRODUCTION.

Run as ``python -m tools.extcache.oai_audit``. This wrapper leaves EXP155's
hash-bound sources/common modules intact and reuses their selection helpers.
See tools/extcache/OAI_AUDIT.md for the execution/review contract.
"""
from __future__ import annotations

import os

# Set BEFORE importing numpy/legacy helpers, even if the caller set larger values.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import tarfile

import numpy as np

from . import common, sources
from raptor_corpus import pick_series
from exp143_own0949 import SPEC as SPEC96

VERSION = "oai-header-audit-v1.0.1"
ROOT = common.ROOT
INDEX = "storage/cache/external/OAI/gen/index.csv"
SAMPLE = "tools/extcache/oai_hb88_sample.json"
BOUND_FILES = (
    "tools/extcache/oai_audit.py", SAMPLE, "tools/extcache/OAI_AUDIT.md",
    "tests/test_extcache_oai_audit.py", "tools/extcache/sources.py",
    "tools/extcache/common.py", "training/src/raptor_corpus.py",
    "training/src/raptor_common.py", "training/src/exp143_own0949.py",
)
POLICY = {
    "SAG_IW_TSE": ("Sagittal", 0, "IW_TSE"),
    "COR_IW_TSE": ("Coronal", 0, "IW_TSE"),
    "COR_MPR": ("Coronal", 1, "DESS_MPR"),
    "AX_MPR": ("Axial", -1, "DESS_MPR"),
}
POLICY_REASON = (
    "SAG_IW_TSE->Sag0 and COR_IW_TSE->Cor0 are the requested H-B-88 policy; "
    "COR_MPR->Cor1 is a DESS source-contract proxy, not PD/T2 equivalence. "
    "AX_MPR retains fluid=-1 (DESS contrast not classified as binary PD/T2 FS); "
    "the axial slot accepts any contrast. No cross-plane/contrast fallback."
)
CONTRACTS = {"raptor64_336_c140_s02": common.CONTRACTS[common.CONTRACT]["slots"],
             "exp143_raptor96_384_c140_s02": SPEC96["slots"]}
MAX_HEADER_MEMBER = sources.MAX_MEMBER_BYTES
MAX_FRAMES = 20000
MAX_MEMBERS = 20000
AXIS_COS = math.cos(math.radians(20))
# Whitelist prevents private/report text and pixels from being read into outputs.
TAGS = ["SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID",
        "ImageOrientationPatient", "ImagePositionPatient", "NumberOfFrames",
        "EchoNumbers", "EchoTime", "EffectiveEchoTime", "ImageLaterality", "Laterality",
        "Rows", "Columns", "PixelSpacing", "SeriesDescription", "ImageType",
        "AcquisitionNumber", "TemporalPositionIdentifier", "FrameOfReferenceUID",
        "PerFrameFunctionalGroupsSequence", "SharedFunctionalGroupsSequence"]


def json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _sha(path):
    return common.sha256(Path(path))


def fingerprint(root=ROOT):
    return {p: _sha(Path(root) / p) for p in BOUND_FILES}


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _nested(obj, sequence, field):
    seq = _get(obj, sequence, []) or []
    if len(seq) > 1:
        raise ValueError(f"ambiguous {sequence}")
    return _get(seq[0], field) if seq else None


def _value(per, shared, ds, field, sequence=None):
    for container in (per, shared):
        v = _nested(container, sequence, field) if sequence else _get(container, field)
        if v is not None:
            return v
    return _get(ds, field)


def _vector(value, length):
    try:
        v = [float(x) for x in value]
        return v if len(v) == length and all(math.isfinite(x) for x in v) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _scalar(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _text(value):
    return str(value).strip() if value is not None else ""


def _multi_text(value):
    if value is None:
        return []
    if isinstance(value, str):
        return value.split("\\")
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]


def _side(value):
    return {"L": "L", "LEFT": "L", "R": "R", "RIGHT": "R"}.get(_text(value).upper(), "")


def sequence_name(description):
    d = _text(description).upper()
    return next((p for p in POLICY if re.match(r"^" + p + r"(?:$|[_\s-])", d)), "")


def geometry(iop):
    """Strict geometry in front of the existing plane helper; no name fallback."""
    v = _vector(iop, 6)
    if v is None:
        return "Unknown", None
    r, c = np.asarray(v[:3]), np.asarray(v[3:])
    if abs(np.linalg.norm(r) - 1) > .01 or abs(np.linalg.norm(c) - 1) > .01 or abs(np.dot(r, c)) > .01:
        return "Unknown", None
    n = np.cross(r, c)
    n /= np.linalg.norm(n)
    if float(np.max(np.abs(n))) < AXIS_COS:
        return "Unknown", n.tolist()
    return sources.plane_from_iop(v), n.tolist()


def header_frames(ds, member):
    """Extract classic or enhanced MR frames from whitelisted DICOM-like headers.

    Functional-group attributes override shared attributes, then top-level tags.
    Output never contains pixels, PatientName, PatientID, reports or private tags.
    """
    raw_n = _get(ds, "NumberOfFrames", 1)
    n = int(raw_n)
    if not 1 <= n <= MAX_FRAMES or float(raw_n) != n:
        raise ValueError("invalid NumberOfFrames")
    per = _get(ds, "PerFrameFunctionalGroupsSequence", []) or []
    shared = _get(ds, "SharedFunctionalGroupsSequence", []) or []
    if (per and len(per) != n) or (n > 1 and len(per) != n) or len(shared) > 1:
        raise ValueError("NumberOfFrames/functional-group length mismatch")
    shared = shared[0] if shared else {}
    result = []
    for i in range(n):
        pf = per[i] if per else {}
        iop = _vector(_value(pf, shared, ds, "ImageOrientationPatient", "PlaneOrientationSequence"), 6)
        ipp = _vector(_value(pf, shared, ds, "ImagePositionPatient", "PlanePositionSequence"), 3)
        spacing = _vector(_value(pf, shared, ds, "PixelSpacing", "PixelMeasuresSequence"), 2)
        frame_side = _text(_value(pf, shared, ds, "FrameLaterality", "FrameAnatomySequence"))
        image_side = _text(_get(ds, "ImageLaterality"))
        series_side = _text(_get(ds, "Laterality"))
        side_values = [x for x in (frame_side, image_side, series_side) if x]
        normalized_sides = {_side(x) for x in side_values}
        issues = []
        if "" in normalized_sides or len(normalized_sides) > 1:
            issues.append("invalid_or_conflicting_laterality")
        plane, normal = geometry(iop)
        if plane == "Unknown":
            issues.append("missing_invalid_or_oblique_iop")
        if ipp is None:
            issues.append("missing_or_invalid_ipp")
        if spacing is None or min(spacing) <= 0:
            issues.append("missing_or_invalid_pixel_spacing")
        sop, series = _text(_get(ds, "SOPInstanceUID")), _text(_get(ds, "SeriesInstanceUID"))
        study = _text(_get(ds, "StudyInstanceUID"))
        if not sop or not series or not study:
            issues.append("missing_uid")
        rows, cols = _scalar(_get(ds, "Rows")), _scalar(_get(ds, "Columns"))
        if not rows or not cols or rows < 1 or cols < 1 or rows != int(rows) or cols != int(cols):
            issues.append("invalid_matrix")
        raw_effective_te = _value(pf, shared, ds, "EffectiveEchoTime", "MREchoSequence")
        raw_echo_time = _get(ds, "EchoTime")
        effective_te = _scalar(raw_effective_te)
        echo_time = _scalar(raw_echo_time)
        for raw_te, te in ((raw_effective_te, effective_te), (raw_echo_time, echo_time)):
            if raw_te is not None and (te is None or te < 0):
                issues.append("invalid_echo_time")
        image_type = _multi_text(_value(pf, shared, ds, "FrameType", "MRImageFrameTypeSequence"))
        if not image_type:
            image_type = _multi_text(_get(ds, "ImageType"))
        if any(marker in " ".join(image_type).upper() for marker in ("LOCALIZER", "SCOUT", "SURVEY")):
            issues.append("localizer")
        result.append(dict(
            member=member, frame_number=i + 1, number_of_frames=n,
            sop_instance_uid=sop, series_instance_uid=series, study_instance_uid=study,
            frame_of_reference_uid=_text(_get(ds, "FrameOfReferenceUID")),
            image_orientation_patient=iop, image_position_patient=ipp, normal=normal,
            plane=plane, echo_numbers=_multi_text(_get(ds, "EchoNumbers")),
            echo_time_ms=echo_time, effective_echo_time_ms=effective_te,
            frame_laterality=frame_side, image_laterality=image_side, laterality=series_side,
            resolved_laterality=next(iter(normalized_sides)) if len(normalized_sides) == 1 else "",
            pixel_spacing=spacing, rows=rows, columns=cols,
            acquisition_number=_text(_get(ds, "AcquisitionNumber")),
            temporal_position=_text(_value(pf, shared, ds, "TemporalPositionIndex", "FrameContentSequence")
                                    or _get(ds, "TemporalPositionIdentifier")),
            stack_id=_text(_value(pf, shared, ds, "StackID", "FrameContentSequence")),
            dimension_index_values=_multi_text(_value(pf, shared, ds, "DimensionIndexValues", "FrameContentSequence")),
            dicom_series_description=_text(_get(ds, "SeriesDescription")),
            image_type=image_type, issues=issues))
    return result


def _rounded(values, digits=4):
    return tuple(round(x, digits) for x in values) if values is not None else None


def group_frames(frames):
    """Split BEFORE sorting; never project unrelated frames using the first IOP."""
    groups = defaultdict(list)
    identities = Counter((f["sop_instance_uid"], f["frame_number"]) for f in frames)
    for f in frames:
        key = (f["study_instance_uid"], f["series_instance_uid"], f["frame_of_reference_uid"],
               _rounded(f["image_orientation_patient"]), tuple(f["echo_numbers"]),
               f["echo_time_ms"], f["effective_echo_time_ms"], f["resolved_laterality"],
               f["rows"], f["columns"], _rounded(f["pixel_spacing"], 6),
               f["acquisition_number"], f["temporal_position"], f["stack_id"],
               tuple(f["image_type"]), f["dicom_series_description"])
        groups[key].append(f)
    out = []
    for key, fs in sorted(groups.items(), key=lambda kv: json_text(kv[0])):
        first = fs[0]
        problems = set(x for f in fs for x in f["issues"])
        if any(identities[(f["sop_instance_uid"], f["frame_number"])] != 1 for f in fs):
            problems.add("duplicate_sop_frame")
        normal = first["normal"]
        valid_positions = normal is not None and all(f["image_position_patient"] is not None for f in fs)
        positions = []
        if valid_positions:
            fs = sorted(fs, key=lambda f: (float(np.dot(f["image_position_patient"], normal)),
                                          f["sop_instance_uid"], f["frame_number"], f["member"]))
            positions = [float(np.dot(f["image_position_patient"], normal)) for f in fs]
        else:
            fs = sorted(fs, key=lambda f: (f["member"], f["frame_number"]))
        delta = np.diff(positions)
        if len(delta) and np.any(delta <= 1e-3):
            problems.add("duplicate_or_near_duplicate_positions")
        if len(fs) < 2:
            problems.add("single_frame_not_a_volume")
        # IOP/IPP consistency: an axis-normal stack must not walk across the plane.
        drift = None
        if valid_positions and len(fs) > 1:
            offsets = np.asarray([f["image_position_patient"] for f in fs]) - fs[0]["image_position_patient"]
            residuals = offsets - np.outer(offsets @ np.asarray(normal), normal)
            drift = float(np.linalg.norm(residuals, axis=1).max())
            if drift > 1.0:
                problems.add("in_plane_position_drift_gt_1mm")
        gid = hashlib.sha256(json_text(key).encode()).hexdigest()[:20]
        for f in fs:
            f["group_id"] = gid
        out.append(dict(
            group_id=gid, frame_count=len(fs), plane=first["plane"],
            study_instance_uid=first["study_instance_uid"], series_instance_uid=first["series_instance_uid"],
            frame_of_reference_uid=first["frame_of_reference_uid"],
            iop=first["image_orientation_patient"], pixel_spacing=first["pixel_spacing"],
            rows=first["rows"], columns=first["columns"],
            echo_numbers=first["echo_numbers"], echo_time_ms=first["echo_time_ms"],
            effective_echo_time_ms=first["effective_echo_time_ms"],
            resolved_laterality=first["resolved_laterality"],
            dicom_series_description=first["dicom_series_description"],
            position_range_mm=[positions[0], positions[-1]] if positions else None,
            median_step_mm=float(np.median(delta)) if len(delta) else None,
            max_step_mm=float(max(delta)) if len(delta) else None, in_plane_drift_mm=drift,
            frame_refs=[[f["member"], f["frame_number"], f["sop_instance_uid"]] for f in fs],
            issues=sorted(problems)))
    return out


def row_issues(row):
    issues = []
    name = sequence_name(row.get("series_description"))
    if not name:
        issues.append("unsupported_source_sequence")
    expected = f"OAI_{row.get('subject_id', '')}_{row.get('knee_side', '')}_{row.get('visit', '')}"
    if row.get("study_key") != expected or row.get("knee_side") not in ("L", "R") or not row.get("visit"):
        issues.append("index_subject_side_visit_join_mismatch")
    desc = row.get("series_description", "").upper()
    sides = {"R" if s == "RIGHT" else "L" for s in re.findall(r"(?:^|[_\s-])(LEFT|RIGHT)(?=$|[_\s-])", desc)}
    if sides and sides != {row.get("knee_side")}:
        issues.append("index_description_side_conflict")
    return issues


def route_group(group, row):
    issues = set(group["issues"]) | set(row_issues(row))
    name = sequence_name(row.get("series_description"))
    header_name = sequence_name(group["dicom_series_description"])
    if header_name and header_name != name:
        issues.add("header_vs_index_sequence_conflict")
    side = group["resolved_laterality"]
    if side and side != row.get("knee_side"):
        issues.add("header_vs_index_side_conflict")
    # A description itself may carry side even when DICOM laterality is absent.
    hd = group["dicom_series_description"].upper()
    hs = {"R" if s == "RIGHT" else "L" for s in re.findall(r"(?:^|[_\s-])(LEFT|RIGHT)(?=$|[_\s-])", hd)}
    if hs and hs != {row.get("knee_side")}:
        issues.add("header_description_side_conflict")
    plane, fluid, contrast = POLICY.get(name, ("Unknown", -1, "Unknown"))
    if group["plane"] != plane:
        issues.add("geometry_vs_source_plane_conflict")
    return dict(group, route_eligible=not issues, routing_issues=sorted(issues),
                source_sequence=name, proposed_plane=plane, proposed_fluid=fluid,
                contrast_family=contrast,
                laterality_basis="DICOM+index" if side else "index_only_unconfirmed_by_DICOM")


def recorded_archive(row, root=ROOT):
    """Use ONLY the recorded path. No basename searches or relocation guesses."""
    recorded = row.get("source_archive_member", "").split("::", 1)[0]
    if not recorded:
        raise ValueError("missing_recorded_raw_path")
    win = PureWindowsPath(recorded)
    p = Path(recorded)
    if win.is_absolute() and os.name != "nt":
        raise ValueError("recorded_windows_path_unavailable_on_this_host")
    if not p.is_absolute():
        p = Path(root) / p
    p = p.resolve()
    base = (Path(root) / "storage/external_data/oai").resolve()
    if not p.is_relative_to(base):
        raise ValueError("recorded_raw_path_outside_oai_registry")
    if not p.name.lower().endswith((".tar.gz", ".tgz", ".tar")):
        raise ValueError("recorded_raw_path_not_supported_archive")
    return p


def audit_archive(row, root=ROOT):
    record = dict(series_key=row["series_key"], study_key=row["study_key"],
                  index_plane=row["plane"], index_fluid=row["fluid_sensitive"],
                  source_description=row["series_description"],
                  recorded_archive=row["source_archive_member"].split("::", 1)[0],
                  status="UNREAD", groups=[], frames=[], errors=[])
    try:
        path = recorded_archive(row, root)
    except ValueError as e:
        record.update(status="INVALID_RECORDED_PATH", errors=[str(e)])
        return record
    if not path.is_file():
        record["status"] = "MISSING_RAW"
        return record
    import pydicom
    frames, errors, seen = [], [], set()
    before = path.stat()
    try:
        # pydicom rewinds while parsing file meta and stopping before pixels.
        # Stream mode (r|*) cannot provide seekable member files on Python 3.12.
        with tarfile.open(path, "r:*") as tar:
            for m in tar:
                if m.isdir():
                    continue
                sources._safe_member(m)
                w = PureWindowsPath(m.name)
                name = PurePosixPath(m.name.replace("\\", "/")).as_posix()
                if w.drive or w.root or name in seen or m.size > MAX_HEADER_MEMBER:
                    raise ValueError("unsafe_or_duplicate_archive_member")
                seen.add(name)
                if len(seen) > MAX_MEMBERS:
                    raise ValueError("archive_member_limit")
                try:
                    with tar.extractfile(m) as stream:
                        ds = pydicom.dcmread(stream, stop_before_pixels=True, specific_tags=TAGS)
                    fs = header_frames(ds, name)
                    if len(frames) + len(fs) > MAX_FRAMES:
                        raise ValueError("archive_frame_limit")
                    frames.extend(fs)
                except Exception as e:
                    # Never dump a DICOM Dataset / arbitrary exception text.
                    errors.append(dict(member=name, error_type=type(e).__name__))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            errors.append(dict(error_type="ArchiveChangedDuringRead"))
    except Exception as e:
        errors.append(dict(error_type=type(e).__name__))
    groups = group_frames(frames)
    if len({f["study_instance_uid"] for f in frames}) > 1:
        errors.append(dict(error_type="MultipleStudyInstanceUIDs"))
    if not frames:
        errors.append(dict(error_type="NoFrames"))
    for group in groups:
        if errors:
            group["issues"].append("archive_incomplete_or_invalid")
    record.update(status="READ_ERROR" if errors else "READ_OK", errors=errors, frames=frames,
                  groups=[route_group(g, row) for g in groups],
                  member_count=len(seen), frame_count=len(frames),
                  raw_size=before.st_size, raw_mtime_ns=before.st_mtime_ns,
                  header_sha256=hashlib.sha256(json_text(frames).encode()).hexdigest())
    return record


def read_oai_index(root=ROOT):
    rows = common.read_index("oai", root)
    if not rows:
        raise ValueError("empty OAI index")
    keys = [r["series_key"] for r in rows]
    archives = [r["source_archive_member"].split("::", 1)[0].replace("\\", "/") for r in rows]
    if len(set(keys)) != len(keys) or len(set(archives)) != len(archives):
        raise ValueError("duplicate index series_key or recorded archive")
    if any(r.get("source", "").lower() != "oai" for r in rows):
        raise ValueError("non-OAI index row")
    return rows


def fixed_sample(rows, manifest, index_sha):
    if manifest["index_sha256"] != index_sha:
        raise ValueError("fixed sample index hash changed; re-review required")
    lookup = {r["series_key"]: r for r in rows}
    keys = manifest["series_keys"]
    if len(keys) != 18 or len(set(keys)) != 18 or not set(keys) <= set(lookup):
        raise ValueError("fixed sample must contain 18 unique indexed keys")
    selected = [lookup[k] for k in keys]
    counts = Counter(sequence_name(r["series_description"]) for r in selected)
    if counts != Counter(AX_MPR=6, COR_MPR=6, SAG_IW_TSE=3, COR_IW_TSE=3):
        raise ValueError("fixed sample stratum counts changed")
    for row in selected:
        name = sequence_name(row["series_description"])
        expected = "Sagittal" if name in ("AX_MPR", "COR_MPR") else POLICY[name][0]
        if row["plane"] != expected:
            raise ValueError("fixed sample no longer matches discordant/control definition")
    return selected


def _pick_rows(rows):
    return [dict(r, Anatomical_Plane=r["plane"], Fluid_Sensitive=r["fluid_sensitive"],
                 SeriesInstanceUID=r["series_key"]) for r in rows]


def select_slots(rows, slots, strict=False):
    used, picks = set(), []
    candidates = _pick_rows(rows)
    known = bool(rows) and all(str(r.get("rule_hit", "")).startswith("known contrast") for r in rows)
    picker = common.pick_strict if strict or known else pick_series
    for plane, fluid, _ in slots:
        selected = picker(candidates, plane, fluid, used)
        if selected is not None:
            used.add(selected["series_key"])
        picks.append(selected)
    return picks


def policy_projection(rows):
    """Declared-source counterfactual only; NEVER an assertion about raw geometry."""
    result = []
    for r in rows:
        name = sequence_name(r["series_description"])
        if name and not row_issues(r):
            plane, fluid, _ = POLICY[name]
            result.append(dict(r, plane=plane, fluid_sensitive=fluid))
    return result


def header_candidates(record, row):
    result = []
    if record["status"] != "READ_OK":
        return result
    for g in record["groups"]:
        if g["route_eligible"]:
            result.append(dict(row, series_key=f"{row['series_key']}::{g['group_id']}",
                               parent_series_key=row["series_key"],
                               dicom_series_uid=g["series_instance_uid"],
                               plane=g["plane"], fluid_sensitive=g["proposed_fluid"],
                               frame_count=g["frame_count"],
                               coverage_mm=g["position_range_mm"][1] - g["position_range_mm"][0],
                               rule_hit="known contrast: H-B-88 strict proposal"))
    return result


def preview_counts(rows, records):
    """Full-index denominator; missing/error raw rows do not become AFTER frames."""
    by_study, after = defaultdict(list), defaultdict(list)
    statuses, group_issues = Counter(), Counter()
    total_groups = eligible_groups = dicom_side_unconfirmed = 0
    for row, record in zip(rows, records, strict=True):
        if row["series_key"] != record["series_key"]:
            raise ValueError("audit record order/identity mismatch")
        by_study[row["study_key"]].append(row)
        statuses[record["status"]] += 1
        for g in record["groups"]:
            total_groups += 1
            eligible_groups += int(g["route_eligible"])
            dicom_side_unconfirmed += int(not g["resolved_laterality"])
            group_issues.update(g["routing_issues"])
        after[row["study_key"]].extend(header_candidates(record, row))
    duplicate_series = 0
    for uid, candidates in after.items():
        parents = defaultdict(set)
        for c in candidates:
            parents[c["dicom_series_uid"]].add(c["parent_series_key"])
        duplicate = {k for k, v in parents.items() if len(v) > 1}
        duplicate_series += len(duplicate)
        # Do not count one DICOM series transported by two archives twice.
        after[uid] = sorted((c for c in candidates if c["dicom_series_uid"] not in duplicate),
                            key=lambda c: (-c["coverage_mm"], -c["frame_count"], c["series_key"]))
    comparisons = {}
    for contract, slots in CONTRACTS.items():
        stats = []
        for plane, fluid, k in slots:
            stats.append(dict(plane=plane, fluid=fluid, slices_per_selection=k,
                              BEFORE=0, AFTER_HEADER_ELIGIBLE=0, AFTER_DECLARED_ONLY=0,
                              BEFORE_source_sequences=Counter(), AFTER_source_sequences=Counter()))
        for uid, group in by_study.items():
            selections = {
                "BEFORE": select_slots(group, slots),
                "AFTER_HEADER_ELIGIBLE": select_slots(after[uid], slots, strict=True),
                "AFTER_DECLARED_ONLY": select_slots(policy_projection(group), slots, strict=True),
            }
            for mode, picks in selections.items():
                for stat, pick in zip(stats, picks):
                    if pick is not None:
                        stat[mode] += 1
                        if mode != "AFTER_DECLARED_ONLY":
                            key = "BEFORE_source_sequences" if mode == "BEFORE" else "AFTER_source_sequences"
                            stat[key][sequence_name(pick["series_description"]) or "OTHER"] += 1
        for stat in stats:
            stat["planned_slice_positions"] = {
                m: stat[m] * stat["slices_per_selection"]
                for m in ("BEFORE", "AFTER_HEADER_ELIGIBLE", "AFTER_DECLARED_ONLY")}
        comparisons[contract] = stats
    return dict(index_series=len(rows), index_studies=len(by_study), raw_statuses=dict(statuses),
                geometry_coverage_complete=statuses.get("READ_OK", 0) == len(rows),
                groups_read=total_groups, groups_route_eligible_before_cross_archive_check=eligible_groups,
                groups_without_dicom_laterality=dicom_side_unconfirmed,
                group_routing_issues=dict(group_issues),
                duplicate_dicom_series_across_archives_quarantined=duplicate_series,
                counts=comparisons,
                interpretation="Selected-study counts per slot, full index denominator. AFTER_HEADER_ELIGIBLE "
                "uses only eligible header groups; unavailable raw is unresolved, not a predicted loss. "
                "AFTER_DECLARED_ONLY is a counterfactual from source names, never verified geometry. "
                "Planned slice positions are NOT pixel-valid masks or unique source frames.")


def verification(path, root=ROOT):
    note = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if note.get("verdict") != "PASS" or note.get("tool_version") != VERSION:
        raise ValueError("separate astra PASS verification required")
    if note.get("code_sha256") != fingerprint(root):
        raise ValueError("verification code hashes differ; re-verification required")
    if note.get("reviewer") != "astra" or note.get("tests_passed") is not True or not note.get("test_command"):
        raise ValueError("verification must name astra and B's passed tests/command")
    return dict(verification_note=str(Path(path).resolve()), verification_sha256=_sha(path))


def _output_dir(path, root=ROOT):
    p, r = Path(path).resolve(), Path(root).resolve()
    if not any(p.is_relative_to(r / parent) and p != r / parent for parent in (".local", "artifacts")):
        raise ValueError("output must be a new child under .local/ or artifacts/")
    p.mkdir(parents=True, exist_ok=False)
    return p


def run(command, output, note, root=ROOT):
    verified = verification(note, root)
    index_sha = _sha(Path(root) / INDEX)
    sample = json.loads((Path(root) / SAMPLE).read_text(encoding="utf-8-sig"))
    if index_sha != sample["index_sha256"]:
        raise ValueError("investigation index hash changed; re-review required")
    rows = read_oai_index(root)
    meta = dict(tool_version=VERSION, command=command, **verified,
                index_sha256=index_sha, code_sha256=fingerprint(root),
                cpu_workers=1, library_threads=1, pixels_read=False, cache_writes=False,
                policy=POLICY, policy_reason=POLICY_REASON)
    if command == "audit":
        rows = fixed_sample(rows, sample, index_sha)
        meta["sample_sha256"] = _sha(Path(root) / SAMPLE)
    out = _output_dir(output, root)
    records = []
    stamp = dict(tool_version=VERSION, verification_note=verified["verification_note"])
    with (out / "archives.jsonl").open("x", encoding="utf-8") as archives, \
            (out / "frames.jsonl").open("x", encoding="utf-8") as frames_out:
        for i, row in enumerate(rows):
            record = audit_archive(row, root)
            frames = record.pop("frames")
            if command == "audit":
                for frame in frames:
                    frames_out.write(json_text(dict(frame, series_key=row["series_key"], **stamp)) + "\n")
            archives.write(json_text(dict(record, **stamp)) + "\n")
            # Frame identity is retained in archives.jsonl; do not retain it in RAM.
            light = dict(record, groups=[{k: v for k, v in g.items() if k != "frame_refs"}
                                         for g in record["groups"]])
            records.append(light)
            if (i + 1) % 100 == 0:
                print(f"{command}: {i + 1}/{len(rows)} archives", file=sys.stderr, flush=True)
    if _sha(Path(root) / INDEX) != index_sha or fingerprint(root) != meta["code_sha256"]:
        raise ValueError("code/index changed during run; output has no completion receipt")
    summary = dict(meta, execution_complete=True,
                   raw_statuses=dict(Counter(r["status"] for r in records)),
                   missing_raw_series=[r["series_key"] for r in records if r["status"] == "MISSING_RAW"],
                   selected_series=[r["series_key"] for r in rows] if command == "audit" else None,
                   missing_raw=[r["recorded_archive"] for r in records if r["status"] == "MISSING_RAW"],
                   output_sha256={p.name: _sha(p) for p in (out / "archives.jsonl", out / "frames.jsonl")})
    if command == "preview":
        summary.update(preview_counts(rows, records))
    else:
        summary["geometry_coverage_complete"] = all(r["status"] == "READ_OK" for r in records)
    summary["status"] = "COMPLETE_HEADERS" if summary["geometry_coverage_complete"] else "INCOMPLETE_RAW_EVIDENCE"
    with (out / "summary.json").open("x", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{VERSION} {summary['status']}; summary={out / 'summary.json'}")
    print(json_text(summary["raw_statuses"]))
    for contract, counts in summary.get("counts", {}).items():
        print(contract + ": slot | BEFORE | AFTER_HEADER_ELIGIBLE | AFTER_DECLARED_ONLY")
        for s in counts:
            print(f"{s['plane']} fluid{s['fluid']} | {s['BEFORE']} | "
                  f"{s['AFTER_HEADER_ELIGIBLE']} | {s['AFTER_DECLARED_ONLY']}")
    # Incomplete evidence is never disguised as a successful admission.
    return 0 if summary["geometry_coverage_complete"] else 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("fingerprint", help="print code hashes for the independent reviewer; no data read")
    for name in ("audit", "preview"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True, help="new directory under .local or artifacts")
        p.add_argument("--verification-note", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.command == "fingerprint":
        print(json.dumps(dict(tool_version=VERSION, code_sha256=fingerprint(),
                              verification_note=None, purpose="review-input-only"), indent=2))
        return 0
    return run(a.command, a.out, a.verification_note)


if __name__ == "__main__":
    raise SystemExit(main())
