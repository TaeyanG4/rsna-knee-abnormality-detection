"""Full H-B-88 header audit; versioned extension of the frozen v1 audit.

No pixels are decoded or cached. Run ``python -m tools.extcache.oai_audit_v2``.

Changelog v2.0.3 (Claude opus for B, 2026-10-10): D4 fix - IPP spacing/order
checks and homogeneity checks use the same canonical reference frame (normal)
as the recorded iop/position_range/drift, not the first spatial frame.
Changelog v2.0.4 (Claude opus for B, 2026-10-10): the verification note may name
any reviewer in REVIEWERS ("astra", "claude-opus"); all other checks unchanged;
the reviewer is stamped on every output record.
"""
from __future__ import annotations

from . import oai_audit as old  # also limits numerical-library threads to one

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import tarfile
import time

import numpy as np

VERSION = "oai-header-audit-v2.0.4"
# Separate read-only reviewers whose PASS note may authorize audit-all.
REVIEWERS = ("astra", "claude-opus")
EXPECTED_ARCHIVES = 9641
ROOT = old.ROOT
PROTOCOL = "artifacts/experiments/HB88_OAI_HEADER_AUDIT_V2/PROTOCOL.md"
OLD_RAW = "storage/external_data/oai/raw/V00"
NEW_RAW = "storage/external_data/oai/raw/V00-redl-20261010"
SELECTION = "storage/external_data/oai/selection/V00_moaks_4series.csv"
IMAGE03 = "storage/external_data/oai/image03_meta/image03.txt"
MOAKS = "storage/external_data/oai/clinical/extracted/mr_assess/Semi-Quant Scoring_ASCII/kMRI_SQ_MOAKS_BICL00.txt"
INPUTS = (old.INDEX, old.SAMPLE, SELECTION, IMAGE03, MOAKS)
BOUND_FILES = tuple(dict.fromkeys((*old.BOUND_FILES,
    "tools/extcache/oai_audit_v2.py", "tests/test_extcache_oai_audit_v2.py",
    "tools/extcache/OAI_AUDIT_V2.md", PROTOCOL)))
POLICY = {"SAG_IW_TSE": ("Sagittal", 1, "IW_TSE_FS"),
          "COR_IW_TSE": ("Coronal", 0, "IW_TSE_NON_FS"),
          "COR_MPR": ("Coronal", 1, "DESS_WE"),
          "AX_MPR": ("Axial", -1, "DESS_WE")}
TAGS = list(dict.fromkeys(old.TAGS + ["ScanOptions", "SequenceName", "ScanningSequence",
    "SequenceVariant", "MRAcquisitionType", "RepetitionTime", "FlipAngle", "InstanceNumber",
    "DimensionIndexSequence", "InPlanePhaseEncodingDirection", "SpectrallySelectedSuppression",
    "FrameLaterality", "TemporalPositionIndex", "StackID", "DimensionIndexValues",
    "InStackPositionNumber", "FrameType", "FrameAcquisitionNumber"]))
MR_CLASSES = {"1.2.840.10008.5.1.4.1.1.4", "1.2.840.10008.5.1.4.1.1.4.1",
              "1.2.840.10008.5.1.4.1.1.4.4"}
# Dimension pointers whose values are spatial positions, not separate acquisitions.
SPATIAL_DIMENSIONS = {0x00209057, 0x00200032}
NONSPATIAL_DIMENSIONS = {0x00209128, 0x00209056, 0x00180086, 0x00189082, 0x00200012,
                        0x00200100, 0x00209156}
sha = old._sha
dump = old.json_text


def fingerprint(root=ROOT):
    return {p: sha(Path(root) / p) for p in BOUND_FILES}


def input_hashes(root=ROOT):
    return {p: sha(Path(root) / p) for p in INPUTS}


def tokens(value):
    return [x.strip().upper() for x in old._multi_text(value)]


def header_frames(ds, member):
    frames = old.header_frames(ds, member)
    shared = old._get(ds, "SharedFunctionalGroupsSequence", []) or [{}]
    per = old._get(ds, "PerFrameFunctionalGroupsSequence", []) or [{}] * len(frames)
    dimensions = old._get(ds, "DimensionIndexSequence", []) or []
    pointers = [old._get(d, "DimensionIndexPointer") for d in dimensions]
    for f, pf in zip(frames, per, strict=True):
        # Use both frame-specific and top-level types: a top-level projection must
        # not disappear when Enhanced MR supplies an otherwise normal FrameType.
        types = tokens(f["image_type"])
        all_types = " ".join(types + tokens(old._get(ds, "ImageType")))
        f["image_type"] = types
        f["excluded_reason"] = ("projection_image" if "PROJECTION" in all_types else
            "localizer" if any(x in all_types for x in ("LOCALIZER", "SCOUT", "SURVEY")) else "")
        f["scan_options"] = tokens(old._value(pf, shared[0], ds, "ScanOptions"))
        f["sequence_name"] = old._text(old._value(pf, shared[0], ds, "SequenceName"))
        f["scanning_sequence"] = tokens(old._value(pf, shared[0], ds, "ScanningSequence"))
        f["sequence_variant"] = tokens(old._value(pf, shared[0], ds, "SequenceVariant"))
        f["mr_acquisition_type"] = old._text(old._value(pf, shared[0], ds, "MRAcquisitionType"))
        f["instance_number"] = old._scalar(old._get(ds, "InstanceNumber"))
        f["repetition_time_ms"] = old._scalar(old._value(pf, shared[0], ds,
            "RepetitionTime", "MRTimingAndRelatedParametersSequence"))
        f["flip_angle"] = old._scalar(old._value(pf, shared[0], ds,
            "FlipAngle", "MRTimingAndRelatedParametersSequence"))
        f["spectrally_selected_suppression"] = old._text(old._value(pf, shared[0], ds,
            "SpectrallySelectedSuppression", "MRModifierSequence")).upper()
        f["in_stack_position"] = old._scalar(old._value(pf, shared[0], ds,
            "InStackPositionNumber", "FrameContentSequence"))
        # Enhanced and classic acquisition numbers are distinct attributes.
        # Keep both: replacing the classic key can merge different acquisitions.
        # Missing/blank stays unknown, distinct from an explicit number (even 0).
        f["frame_acquisition_number"] = old._text(old._value(pf, shared[0], ds,
            "FrameAcquisitionNumber", "FrameContentSequence"))
        f["nonspatial_dimensions"] = []
        vals = f["dimension_index_values"]
        if vals or pointers:
            try:
                if len(vals) != len(pointers):
                    raise ValueError("dimension length mismatch")
                for p, val in zip(pointers, vals, strict=True):
                    p = int(p)
                    if p not in SPATIAL_DIMENSIONS | NONSPATIAL_DIMENSIONS:
                        raise ValueError("unknown dimension pointer semantics")
                    if p not in SPATIAL_DIMENSIONS:
                        f["nonspatial_dimensions"].append([p, val])
            except (TypeError, ValueError):
                f["issues"].append("unresolved_dimension_semantics")
        if old._text(old._get(ds, "SOPClassUID")) not in MR_CLASSES:
            f["issues"].append("unsupported_sop_class")
    return frames


CONTRAST_FIELDS = ("scan_options", "sequence_name", "scanning_sequence", "sequence_variant",
    "mr_acquisition_type", "repetition_time_ms", "flip_angle", "spectrally_selected_suppression",
    "nonspatial_dimensions")
PARTITION_FIELDS = (*CONTRAST_FIELDS, "frame_acquisition_number")


def order_state(positions):
    if len(positions) < 2:
        return "insufficient"
    d = np.diff(positions)
    return "ascending" if np.all(d > .001) else "descending" if np.all(d < -.001) else "unordered_or_duplicate"


def frame_sort_key(f):
    """Canonical header order, including a content tie-break for duplicate IDs.

    v1 chooses the first frame's normal for IPP sorting and drift. Sort before
    entering that helper so a within-tolerance IOP difference cannot change the
    reference normal or geometry eligibility with archive enumeration order.
    """
    return (tuple(f["image_orientation_patient"] or ()), f["sop_instance_uid"],
        f["frame_number"], f["member"],
        dump({k: v for k, v in f.items() if k != "group_id"}))


def group_frames(frames):
    """Partition contrast/Enhanced acquisitions, then reuse v1 geometry checks."""
    ids = Counter((f["sop_instance_uid"], f["frame_number"]) for f in frames)
    partitions = defaultdict(list)
    for f in frames:
        if not f["excluded_reason"]:
            partitions[dump([f[k] for k in PARTITION_FIELDS])].append(f)
    groups = []
    for partition, partition_frames in sorted(partitions.items()):
        fs = sorted(partition_frames, key=frame_sort_key)
        lookup = {(f["member"], f["frame_number"]): f for f in fs}
        for g in old.group_frames(fs):
            members = [lookup[(ref[0], ref[1])] for ref in g["frame_refs"]]
            # D4: one reference for every geometry check. v1 records iop,
            # pixel_spacing, position range and drift from its first INPUT frame,
            # i.e. the canonical (frame_sort_key) minimum, not the first spatial
            # frame. Using members[0] here let a within-rounding IOP tilt on the
            # most-inferior slice flip irregular_ipp_spacing either way.
            first = min(members, key=frame_sort_key)
            gid = hashlib.sha256((g["group_id"] + partition).encode()).hexdigest()[:24]
            for f in members:
                f["group_id"] = gid
            g.update(group_id=gid, image_type=first["image_type"],
                     **{k: first[k] for k in PARTITION_FIELDS})
            issues = set(g["issues"])
            if (first["image_orientation_patient"], first["pixel_spacing"]) != (g["iop"], g["pixel_spacing"]):
                issues.add("geometry_reference_frame_mismatch")  # fail closed if v1 ever changes
            if any(ids[(f["sop_instance_uid"], f["frame_number"])] != 1 for f in members):
                issues.add("duplicate_sop_frame")
            for key, atol in (("image_orientation_patient", 1e-4), ("pixel_spacing", 1e-6)):
                if first[key] is not None and any(f[key] is None or
                        not np.allclose(f[key], first[key], atol=atol, rtol=0) for f in members):
                    issues.add("inhomogeneous_" + key)
            n = first["normal"]
            if n is not None and all(f["image_position_patient"] is not None for f in members):
                position = lambda f: float(np.dot(f["image_position_patient"], n))
                # Member/frame names are stable archive identities; tar iteration
                # order is not. Keep this diagnostic invariant under enumeration.
                raw_order = sorted((f for f in fs if f.get("group_id") == gid),
                    key=lambda f: (f["member"], f["frame_number"], frame_sort_key(f)))
                g["archive_ipp_order"] = order_state([position(f) for f in raw_order])
                instance = [f["instance_number"] for f in members]
                g["instance_ipp_order"] = "missing_or_nonunique"
                if all(x is not None for x in instance) and len(set(instance)) == len(instance):
                    g["instance_ipp_order"] = order_state([position(f) for f in sorted(members,
                        key=lambda f: f["instance_number"])])
                positions = [position(f) for f in members]
                g["ordered_positions_mm"] = positions
                steps = np.diff(positions)
                median = float(np.median(steps)) if len(steps) else 0.
                if len(steps) and np.any(np.abs(steps - median) > max(.05, .05 * abs(median))):
                    issues.add("irregular_ipp_spacing")
            g["issues"] = sorted(issues)
            groups.append(g)
    return sorted(groups, key=lambda g: g["group_id"])


def contrast_evidence(g, source):
    options = set(g["scan_options"])
    seq = g["sequence_name"].lower()
    suppression = g["spectrally_selected_suppression"]
    image = " ".join(g["image_type"]).upper()
    explicit_fs = bool(options & {"FS", "FATSAT", "SPAIR", "SPIR", "STIR"}) or bool(
        re.search(r"(?:^|\s)(?:FS|FATSAT|FAT SAT|FAT SATURATION)(?:$|\s)", image))
    fs = explicit_fs or suppression == "FAT"
    we = "WE" in options or "WATER EXCITATION" in image
    problems = []
    if source in ("SAG_IW_TSE", "COR_IW_TSE"):
        if "tse" not in seq:
            problems.append("missing_or_conflicting_tse_sequence_evidence")
        if suppression not in ("", "NONE", "FAT"):
            problems.append("conflicting_tse_suppression_method")
        if source == "SAG_IW_TSE" and (not fs or we):
            problems.append("missing_or_conflicting_fat_suppression_evidence")
        if source == "COR_IW_TSE" and (not options or fs or we or suppression not in ("", "NONE")):
            problems.append("missing_or_conflicting_non_fs_evidence")
    elif source in ("COR_MPR", "AX_MPR"):
        if "de3d" not in seq or not we or g["mr_acquisition_type"].upper() != "3D":
            problems.append("missing_or_conflicting_dess_we_evidence")
        if explicit_fs or suppression not in ("", "NONE", "FAT"):
            problems.append("conflicting_dess_suppression_method")
    else:
        problems.append("unsupported_source_sequence")
    return dict(contrast_verified=not problems, contrast_issues=problems,
        fat_suppression_method="WATER_EXCITATION" if we else "FAT_SAT" if fs else "NONE" if not problems else "UNKNOWN",
        contrast_certainty="HEADER_SUPPORTED_SOURCE_PROTOCOL" if not problems else "UNRESOLVED")


def archive_tail(recorded, root=ROOT):
    value = str(recorded).split("::", 1)[0].replace("\\", "/")
    p = PurePosixPath(value)
    if ".." in p.parts:
        raise ValueError("raw_path_traversal")
    prefixes = (str(Path(root).resolve()).replace("\\", "/") + "/" + OLD_RAW + "/", OLD_RAW + "/")
    prefix = next((x for x in prefixes if value.casefold().startswith(x.casefold())), None)
    if prefix is None:
        raise ValueError("recorded_path_does_not_match_exact_old_root")
    tail = value[len(prefix):]
    if not tail.startswith("image03/00m/") or not tail.endswith(".tar.gz"):
        raise ValueError("invalid_baseline_archive_tail")
    return tail


def relocated_archive(row, root=ROOT):
    tail = archive_tail(row["source_archive_member"], root)
    base = (Path(root) / NEW_RAW).resolve()
    path = (base / tail).resolve()
    if not path.is_relative_to(base):
        raise ValueError("relocated_path_outside_registered_root")
    return path


def uri_tail(uri):
    # Bind exact OAI URI suffix; no basename join or access to the URI/network.
    parts = str(uri).replace("\\", "/").split("/00m/")
    if len(parts) != 2 or ".." in PurePosixPath(parts[1]).parts or not parts[1].endswith(".tar.gz"):
        raise ValueError("invalid_image03_baseline_uri")
    return "image03/00m/" + parts[1]


def description_sides(text):
    """Preserve absent, single and contradictory side evidence separately."""
    return {"R" if s == "RIGHT" else "L" for s in re.findall(
        r"(?:^|[_\s-])(LEFT|RIGHT)(?=$|[_\s-])", str(text).upper())}


def side_from_description(text):
    # Identity joins require one side; routing additionally inspects the set.
    sides = description_sides(text)
    return next(iter(sides)) if len(sides) == 1 else ""


def read_columns(path, fields, delimiter=",", skip_dictionary=False, encoding="utf-8-sig"):
    """Parse only named identity columns; do not retain/output clinical scores."""
    with Path(path).open(newline="", encoding=encoding) as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        positions = [header.index(k) for k in fields]
        if skip_dictionary:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError("malformed_identity_table")
            yield {k: row[i].strip() for k, i in zip(fields, positions, strict=True)}


def build_joins(rows, root=ROOT):
    chosen, metadata, moaks = defaultdict(list), defaultdict(list), defaultdict(list)
    for x in read_columns(Path(root) / SELECTION, ("ID", "side", "series", "image_file")):
        chosen[uri_tail(x["image_file"])].append(x)
    wanted = {x["image_file"] for values in chosen.values() for x in values}
    for x in read_columns(Path(root) / IMAGE03,
            ("src_subject_id", "image_file", "image_description", "visit"), "\t", True):
        if x["image_file"] in wanted:
            metadata[x["image_file"]].append(x)
    for x in read_columns(Path(root) / MOAKS, ("ID", "SIDE", "READPRJ", "VERSION"), "|", encoding="latin-1"):
        side = {"1": "R", "2": "L", "1: Right": "R", "2: Left": "L"}.get(x["SIDE"], "")
        if not side:
            raise ValueError("unknown_MOAKS_SIDE_code")
        moaks[(x["ID"], side)].append((x["READPRJ"], x["VERSION"]))
    joins = {}
    for row in rows:
        problems = set(old.row_issues(row))
        tail = archive_tail(row["source_archive_member"], root)
        sels = chosen.get(tail, [])
        meta = []
        image_side = ""
        if len(sels) != 1:
            problems.add("selection_uri_cardinality_not_one")
        else:
            sel = sels[0]
            if (sel["ID"], old._side(sel["side"]), old.sequence_name(sel["series"])) != (
                    row["subject_id"], row["knee_side"], old.sequence_name(row["series_description"])):
                problems.add("selection_vs_index_identity_conflict")
            meta = metadata.get(sel["image_file"], [])
            if len(meta) != 1:
                problems.add("image03_uri_cardinality_not_one")
            else:
                m = meta[0]
                image_side = side_from_description(m["image_description"])
                if (m["src_subject_id"], image_side, old.sequence_name(m["image_description"])) != (
                        row["subject_id"], row["knee_side"], old.sequence_name(row["series_description"])):
                    problems.add("image03_vs_index_identity_conflict")
                if m["visit"] != "V00":
                    problems.add("image03_vs_baseline_visit_conflict")
        if row["visit"] != "V00":
            problems.add("nonbaseline_visit")
        projects = moaks.get((row["subject_id"], row["knee_side"]), [])
        if not projects:
            problems.add("missing_MOAKS_participant_side")
        duplicate_read = len(set(projects)) != len(projects)
        if duplicate_read:
            problems.add("duplicate_MOAKS_reading_identity")
        joins[row["series_key"]] = dict(issues=sorted(problems), join_verified=not problems,
            participant=row["subject_id"], knee_side=row["knee_side"], visit=row["visit"],
            selection_rows=len(sels), image03_rows=len(meta), image03_side=image_side,
            image03_visit=meta[0]["visit"] if len(meta) == 1 else "",
            moaks_read_count=len(projects), moaks_read_identities=[list(x) for x in sorted(set(projects))],
            moaks_visit_basis="BICL00 baseline file and /00m/ image URI",
            clinical_values_checked=False, conflicting_clinical_reads="NOT_CHECKED_IDENTITY_ONLY")
    return joins


def route_group(g, row, join):
    source = old.sequence_name(row["series_description"])
    plane, fluid, family = POLICY.get(source, ("Unknown", -1, "UNKNOWN"))
    issues = set(g["issues"]) | set(old.row_issues(row)) | set(join["issues"])
    if old.sequence_name(g["dicom_series_description"]) != source:
        issues.add("header_vs_index_sequence_conflict")
    side = g["resolved_laterality"]
    header_sides = description_sides(g["dicom_series_description"])
    if side and side != row["knee_side"]:
        issues.add("header_vs_index_side_conflict")
    if header_sides and header_sides != {row["knee_side"]}:
        issues.add("header_description_side_conflict")
    basis = "DICOM+image03+selection+MOAKS" if side else "image03+selection+MOAKS_DICOM_absent"
    if not side and (not join["join_verified"] or join["image03_side"] != row["knee_side"]):
        issues.add("missing_independent_laterality")
    if g["plane"] != plane:
        issues.add("geometry_vs_source_plane_conflict")
    evidence = contrast_evidence(g, source)
    issues.update(evidence["contrast_issues"])
    return dict(g, source_sequence=source, proposed_plane=plane, proposed_fluid=fluid,
        contrast_family=family, laterality_basis=basis, header_description_sides=sorted(header_sides), **evidence,
        route_eligible=not issues, routing_issues=sorted(issues))


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("registered_CPU_wall_cap_no_completion_receipt")


def audit_archive(row, join, root=ROOT, deadline=None):
    record = dict(series_key=row["series_key"], study_key=row["study_key"],
        index_plane=row["plane"], index_fluid=row["fluid_sensitive"],
        source_description=row["series_description"], join=join,
        recorded_archive=row["source_archive_member"].split("::", 1)[0],
        status="UNREAD", groups=[], excluded_frames=[], errors=[])
    try:
        path = relocated_archive(row, root)
    except ValueError as e:
        return dict(record, status="INVALID_RECORDED_PATH", errors=[str(e)])
    record["relocated_archive"] = str(path)
    if not path.is_file():
        return dict(record, status="MISSING_RAW")
    import pydicom
    frames, errors, seen = [], [], set()
    stat = path.stat()
    try:
        # Seekable tar member required by pydicom; never extract to disk.
        with tarfile.open(path, "r:*") as tar:
            for member in tar:
                check_deadline(deadline)
                if member.isdir():
                    continue
                old.sources._safe_member(member)
                win = PureWindowsPath(member.name)
                name = PurePosixPath(member.name.replace("\\", "/")).as_posix()
                if win.drive or win.root or name in seen or member.size > old.MAX_HEADER_MEMBER:
                    raise ValueError("unsafe_or_duplicate_archive_member")
                seen.add(name)
                if len(seen) > old.MAX_MEMBERS:
                    raise ValueError("archive_member_limit")
                try:
                    with tar.extractfile(member) as stream:
                        ds = pydicom.dcmread(stream, stop_before_pixels=True, specific_tags=TAGS)
                    check_deadline(deadline)
                    fs = header_frames(ds, name)
                    if len(frames) + len(fs) > old.MAX_FRAMES:
                        raise ValueError("archive_frame_limit")
                    frames.extend(fs)
                except TimeoutError:
                    raise
                except Exception as e:
                    errors.append(dict(member=name, error_type=type(e).__name__))
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            errors.append(dict(error_type="ArchiveChangedDuringRead"))
    except TimeoutError:
        raise
    except Exception as e:
        errors.append(dict(error_type=type(e).__name__))
    if not frames:
        errors.append(dict(error_type="NoFrames"))
    if len({f["study_instance_uid"] for f in frames}) > 1:
        errors.append(dict(error_type="MultipleStudyInstanceUIDs"))
    frames.sort(key=frame_sort_key)
    groups = group_frames(frames)
    for g in groups:
        if errors:
            g["issues"].append("archive_incomplete_or_invalid")
    exclusions = [dict(member=f["member"], frame_number=f["frame_number"],
        sop_instance_uid=f["sop_instance_uid"], series_instance_uid=f["series_instance_uid"],
        image_type=f["image_type"], plane=f["plane"], reason=f["excluded_reason"])
        for f in frames if f["excluded_reason"]]
    return dict(record, status="READ_ERROR" if errors else "READ_OK", errors=errors,
        groups=[route_group(g, row, join) for g in groups], excluded_frames=exclusions,
        member_count=len(seen), frame_count=len(frames), retained_frame_count=len(frames)-len(exclusions),
        raw_size=stat.st_size, raw_mtime_ns=stat.st_mtime_ns,
        header_sha256=hashlib.sha256(dump(frames).encode()).hexdigest())


def cross_archive_issues(records):
    series, identities = defaultdict(set), defaultdict(set)
    for r in records:
        for g in r["groups"]:
            series[g["series_instance_uid"]].add(r["series_key"])
            for _, frame, sop in g["frame_refs"]:
                identities[(sop, frame)].add(r["series_key"])
        for f in r["excluded_frames"]:
            series[f["series_instance_uid"]].add(r["series_key"])
            identities[(f["sop_instance_uid"], f["frame_number"])].add(r["series_key"])
    bad_series = {k for k, v in series.items() if len(v) > 1}
    bad_ids = {k for k, v in identities.items() if len(v) > 1}
    for r in records:
        for g in r["groups"]:
            problems = set(g["routing_issues"])
            if g["series_instance_uid"] in bad_series:
                problems.add("cross_archive_series_uid")
            if any((sop, frame) in bad_ids for _, frame, sop in g["frame_refs"]):
                problems.add("cross_archive_sop_frame")
            g.update(routing_issues=sorted(problems), route_eligible=not problems)
    return dict(duplicate_series_uids=len(bad_series), duplicate_sop_frames=len(bad_ids))


def routing_tables(rows, records):
    by_study, after, groups = defaultdict(list), defaultdict(list), {}
    for r, rec in zip(rows, records, strict=True):
        if r["series_key"] != rec["series_key"]:
            raise ValueError("row_receipt_identity_mismatch")
        by_study[r["study_key"]].append(r)
        for g in rec["groups"]:
            key = r["series_key"] + "::" + g["group_id"]
            groups[key] = g
        after[r["study_key"]].extend(old.header_candidates(rec, r))
    comparisons, selections = {}, []
    for contract, slots in old.CONTRACTS.items():
        counts = [dict(plane=p, fluid=f, slices_per_selection=k, BEFORE=0, AFTER=0,
            BEFORE_source_sequences=Counter(), AFTER_source_sequences=Counter()) for p, f, k in slots]
        for study, original in by_study.items():
            candidates = sorted(after[study], key=lambda c: (-c["coverage_mm"], -c["frame_count"], c["series_key"]))
            before = old.select_slots(original, slots)
            corrected = old.select_slots(candidates, slots, strict=True)
            for stat, b, a in zip(counts, before, corrected, strict=True):
                for label, picked in (("BEFORE", b), ("AFTER", a)):
                    if picked:
                        stat[label] += 1
                        stat[label + "_source_sequences"][old.sequence_name(picked["series_description"])] += 1
                selections.append(dict(contract=contract, study_key=study, plane=stat["plane"], fluid=stat["fluid"],
                    before_series_key=b["series_key"] if b else None,
                    after_group_key=a["series_key"] if a else None,
                    contrast_family=groups[a["series_key"]]["contrast_family"] if a else None))
        for stat in counts:
            stat["planned_slice_positions"] = {m: stat[m] * stat["slices_per_selection"] for m in ("BEFORE", "AFTER")}
        comparisons[contract] = counts
    return comparisons, selections


def verification(note_path, root=ROOT):
    note = json.loads(Path(note_path).read_text(encoding="utf-8-sig"))
    # v2.0.4: the recorded independent reviewer may be any allow-listed one.
    if note.get("verdict") != "PASS" or note.get("tool_version") != VERSION or (
            not isinstance(note.get("reviewer"), str) or note["reviewer"] not in REVIEWERS):
        raise ValueError("independent_reviewer_PASS_required")
    if note.get("tests_passed") is not True or not note.get("test_command") or not note.get("review_report"):
        raise ValueError("test_receipt_and_review_report_required")
    if note.get("code_sha256") != fingerprint(root) or note.get("input_sha256") != input_hashes(root):
        raise ValueError("code_or_input_hash_changed_reverify")
    return dict(verification_note=str(Path(note_path).resolve()), verification_sha256=sha(note_path),
                reviewer=note["reviewer"])


def run(output, note_path, root=ROOT, max_seconds=10800):
    started, cpu_start = time.monotonic(), time.process_time()
    deadline = started + max_seconds
    verified = verification(note_path, root)
    inputs, codes = input_hashes(root), fingerprint(root)
    sample = json.loads((Path(root) / old.SAMPLE).read_text(encoding="utf-8-sig"))
    if inputs[old.INDEX] != sample["index_sha256"]:
        raise ValueError("pinned_index_changed")
    rows = old.read_oai_index(root)
    if len(rows) != EXPECTED_ARCHIVES:
        raise ValueError("expected_exactly_9641_indexed_archives")
    joins = build_joins(rows, root)
    paths = [relocated_archive(r, root) for r in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("relocation_collision")
    raw_base = (Path(root) / NEW_RAW).resolve()
    inventory = sorted(p.resolve() for p in raw_base.rglob("*.tar.gz"))
    extra = sorted(str(p.relative_to(raw_base)) for p in set(inventory) - set(paths))
    out = old._output_dir(output, root)
    stamp = dict(tool_version=VERSION, **verified)
    records = []
    # Per-archive receipts remain inspectable if interrupted, but never imply
    # completion. Cross-archive decisions are in final archives/routing outputs.
    with (out / "archive-progress.jsonl").open("x", encoding="utf-8") as progress:
        for i, row in enumerate(rows):
            check_deadline(deadline)
            rec = audit_archive(row, joins[row["series_key"]], root, deadline)
            progress.write(dump(dict(rec, **stamp)) + "\n")
            progress.flush()
            records.append(rec)
            if (i + 1) % 100 == 0:
                print(f"audit-all {i+1}/{len(rows)} archives; wall {time.monotonic()-started:.1f}s", flush=True)
    duplicates = cross_archive_issues(records)
    counts, selections = routing_tables(rows, records)
    outputs = []
    def write_jsonl(name, values):
        p = out / name
        with p.open("x", encoding="utf-8") as f:
            for value in values:
                f.write(dump(dict(value, **stamp)) + "\n")
        outputs.append(p)
    write_jsonl("archives.jsonl", records)
    write_jsonl("routing.jsonl", (dict(series_key=r["series_key"], study_key=r["study_key"],
        relocated_archive=r.get("relocated_archive"), group_key=r["series_key"]+"::"+g["group_id"], **g)
        for r in records for g in r["groups"]))
    write_jsonl("slot-selections.jsonl", selections)
    outputs.append(out / "archive-progress.jsonl")
    if input_hashes(root) != inputs or fingerprint(root) != codes or sha(note_path) != verified["verification_sha256"]:
        raise ValueError("inputs_code_or_review_changed_no_completion_receipt")
    for r in records:
        if "raw_size" in r:
            st = Path(r["relocated_archive"]).stat()
            if (st.st_size, st.st_mtime_ns) != (r["raw_size"], r["raw_mtime_ns"]):
                raise ValueError("raw_archive_changed_no_completion_receipt")
    check_deadline(deadline)
    complete = all(r["status"] == "READ_OK" for r in records)
    groups = [g for r in records for g in r["groups"]]
    summary = dict(**stamp, execution_complete=True,
        status="COMPLETE_HEADERS" if complete else "INCOMPLETE_RAW_EVIDENCE",
        input_sha256=inputs, code_sha256=codes, protocol=PROTOCOL,
        index_archives=len(rows), index_studies=len({r["study_key"] for r in rows}),
        inventory_archives=len(inventory), unindexed_archives=extra,
        raw_statuses=dict(Counter(r["status"] for r in records)),
        excluded_frames=dict(Counter(f["reason"] for r in records for f in r["excluded_frames"])),
        groups=len(groups), eligible_groups=sum(g["route_eligible"] for g in groups),
        routing_issues=dict(Counter(p for g in groups for p in g["routing_issues"])),
        join_issues=dict(Counter(p for j in joins.values() for p in j["issues"])),
        laterality_basis=dict(Counter(g["laterality_basis"] for g in groups)),
        cross_archive_duplicates=duplicates, counts=counts,
        cpu_workers=1, library_threads=1, wall_hours=(time.monotonic()-started)/3600,
        cpu_process_hours=(time.process_time()-cpu_start)/3600, gpu_hours=0,
        pixel_decoding=False, pixel_cache_writes=False, raw_byte_sha256=False,
        output_sha256={p.name: sha(p) for p in outputs},
        limitations="Header eligibility only; raw size/mtime and parsed-header hashes, not raw byte authentication. "
            "MOAKS identity joins only: clinical read conflicts/targets not certified. No pixel/cache/quality/refit admission.")
    # Hashing large JSONL outputs is real CPU/I/O work and also belongs to the cap.
    check_deadline(deadline)
    summary["wall_hours"] = (time.monotonic()-started)/3600
    summary["cpu_process_hours"] = (time.process_time()-cpu_start)/3600
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"{VERSION} {summary['status']} -> {out / 'summary.json'}")
    for contract, stats in counts.items():
        print(contract + " | slot | BEFORE | AFTER")
        for s in stats:
            print(f"{s['plane']} fluid{s['fluid']} | {s['BEFORE']} | {s['AFTER']}")
    return 0 if complete else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fingerprint")
    audit = sub.add_parser("audit-all")
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--verification-note", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fingerprint":
        print(json.dumps(dict(tool_version=VERSION, code_sha256=fingerprint(), input_sha256=input_hashes()), indent=2))
        return 0
    return run(args.out, args.verification_note)


if __name__ == "__main__":
    raise SystemExit(main())
