"""H-B-123 right-knee rendering v1.0.0 (2026-10-13).

Header evidence only. No image classifiers, report access or label access.
Changelog: conservative side resolution, bound map, pre-augmentation corpus view.
Unknown/conflicting studies are retained unchanged; absent map IDs are fatal.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

VERSION = "1.0.0"
UID = "StudyInstanceUID"
MIN_OFFSET_MM = 20.0
TAGS = ["ImageLaterality", "Laterality", "BodyPartExamined", "PatientOrientation",
        "ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "Rows", "Columns"]
POLICY = {
    "version": VERSION, "canonical_side": "R", "flip_side": "L",
    "flip_planes": ["Coronal", "Axial"], "flip_axis": "last",
    "sagittal": "unchanged (including slice order)",
    "unknown": "retain unchanged", "absent_id": "reject",
    "geometry": "LPS image-center x; abs(series median x)>=20mm; >=2 agreeing series",
    "conflict": "any direct-side conflict or tag/geometry disagreement => U",
    "orientation": "coronal/axial increasing image-column direction must have LPS x>=0.8",
    "sampling": "lexical first/middle/last DICOM of every metadata series",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def direct_sides(header):
    result = set()
    for key in ("ImageLaterality", "Laterality"):
        value = str(header.get(key, "")).strip().upper()
        if value in ("L", "R"):
            result.add(value)
        elif value in ("B", "BOTH", "BILATERAL"):
            result.update(("L", "R"))
    # KNEE alone and patient orientation do not encode the studied knee's side.
    words = re.findall(r"[A-Z]+", str(header.get("BodyPartExamined", "")).upper())
    compact = "".join(words)
    if "LEFT" in words or compact in ("LKNEE", "KNEEL", "LEFTKNEE", "KNEELEFT"):
        result.add("L")
    if "RIGHT" in words or compact in ("RKNEE", "KNEER", "RIGHTKNEE", "KNEERIGHT"):
        result.add("R")
    return result


def geometry(header, plane):
    """IOP gives directions, not laterality. IPP + IOP + FOV give center x."""
    try:
        iop = np.asarray(header["ImageOrientationPatient"], dtype=float)
        ipp = np.asarray(header["ImagePositionPatient"], dtype=float)
        spacing = np.asarray(header["PixelSpacing"], dtype=float)
        rows, cols = int(header["Rows"]), int(header["Columns"])
        if (iop.shape != (6,) or ipp.shape != (3,) or spacing.shape != (2,)
                or not np.isfinite(np.r_[iop, ipp, spacing]).all()
                or min(rows, cols) <= 0 or min(spacing) <= 0
                or abs(np.linalg.norm(iop[:3]) - 1) > .02
                or abs(np.linalg.norm(iop[3:]) - 1) > .02
                or abs(np.dot(iop[:3], iop[3:])) > .02):
            return None, False, False
        normal = np.abs(np.cross(iop[:3], iop[3:]))
        axis = {"Sagittal": 0, "Coronal": 1, "Axial": 2}.get(plane)
        plane_ok = axis is not None and int(normal.argmax()) == axis and normal[axis] >= .8
        orientation_ok = plane_ok and (plane == "Sagittal" or iop[0] >= .8)
        # When present, PatientOrientation's first value describes column direction.
        po = header.get("PatientOrientation", [])
        if isinstance(po, str):
            po = po.split("\\")
        po_ok = not po or not str(po[0]) or plane == "Sagittal" or "L" in str(po[0]).upper()
        center = ipp + iop[:3] * spacing[1] * (cols - 1) / 2 + iop[3:] * spacing[0] * (rows - 1) / 2
        return float(center[0]), bool(orientation_ok and po_ok), not po_ok
    except (KeyError, TypeError, ValueError):
        return None, False, False


def summarize_series(headers, plane, *, missing=False, errors=0):
    tags, centers = set(), []
    safe, po_conflicts = True, 0
    tag_fields = {key: 0 for key in TAGS[:4]}
    digest = hashlib.sha256()
    for header in headers:
        tags.update(direct_sides(header))
        for key in tag_fields:
            tag_fields[key] += bool(header.get(key))
        center, orientation_ok, po_conflict = geometry(header, plane)
        if center is not None:
            centers.append(center)
        # Only planes affected by the new transform gate its orientation.
        if plane in ("Coronal", "Axial"):
            safe &= orientation_ok
        po_conflicts += po_conflict
        digest.update(canonical(header))
    median = float(np.median(centers)) if centers else None
    side = "U" if median is None or abs(median) < MIN_OFFSET_MM else ("L" if median > 0 else "R")
    # Conflicting strongly offset samples indicate a bilateral/localizer acquisition.
    strong = {"L" if x > 0 else "R" for x in centers if abs(x) >= MIN_OFFSET_MM}
    return {"tags": sorted(tags), "geometry_side": side, "sample_geometry_conflict": len(strong) > 1,
            "orientation_safe": bool(safe and headers and not missing and not errors),
            "headers": len(headers), "missing": bool(missing), "errors": errors,
            "patient_orientation_conflicts": po_conflicts, "tag_fields": tag_fields,
            "header_evidence_sha256": digest.hexdigest()}


def resolve_study(series):
    tags = set().union(*(set(s["tags"]) for s in series)) if series else set()
    geo = {s["geometry_side"] for s in series} - {"U"}
    strong = sum(s["geometry_side"] != "U" for s in series)
    comparable = len(tags) == 1 and len(geo) == 1
    disagree = comparable and tags != geo
    if not series or any(s["missing"] or s["errors"] for s in series):
        side, reason = "U", "missing_or_unreadable_headers"
    elif len(tags) > 1:
        side, reason = "U", "direct_tag_conflict"
    elif len(geo) > 1 or any(s["sample_geometry_conflict"] for s in series):
        side, reason = "U", "geometry_conflict"
    elif disagree:
        side, reason = "U", "tag_geometry_disagreement"
    elif not all(s["orientation_safe"] for s in series):
        side, reason = "U", "unsafe_orientation"
    elif tags:
        side, reason = next(iter(tags)), "direct_tags"
    elif len(geo) == 1 and strong >= 2:
        side, reason = next(iter(geo)), "geometry_consensus"
    else:
        side, reason = "U", "insufficient_evidence"
    return {"side": side, "reason": reason, "series": len(series),
            "headers": sum(s["headers"] for s in series),
            "tag_geometry_comparable": bool(comparable),
            "tag_geometry_agree": bool(comparable and not disagree),
            "tag_geometry_disagree": bool(disagree),
            "direct_tag_conflict": len(tags) > 1,
            "geometry_conflict": len(geo) > 1 or any(s["sample_geometry_conflict"] for s in series),
            "evidence_sha256": hashlib.sha256(canonical(series)).hexdigest()}


class SideMap:
    def __init__(self, path, expected_sha256=None):
        self.path = str(Path(path).resolve())
        self.sha256 = sha(path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("side map digest differs")
        doc = json.loads(Path(path).read_bytes())
        if doc.get("schema") != "hb123-side-map/v1" or doc.get("policy") != POLICY:
            raise ValueError("side map policy differs")
        self.sides = {r[UID]: r["side"] for r in doc["studies"]}
        if len(self.sides) != len(doc["studies"]) or not self.sides or set(self.sides.values()) - {"L", "R", "U"}:
            raise ValueError("malformed side map")

    def require(self, ids):
        if set(map(str, ids)) - self.sides.keys():
            raise ValueError("side map lacks required study IDs")

    def side(self, uid):
        try:
            return self.sides[str(uid)]
        except KeyError as error:
            raise ValueError("side map lacks study ID") from error


def canonical_volume(volume, side, slots):
    """Pre-resize/pre-augmentation flip; no RNG, slice reorder or in-place writes."""
    if side not in ("L", "R", "U"):
        raise ValueError("invalid side")
    if volume.ndim != 3 or sum(int(s[2]) for s in slots) != len(volume):
        raise ValueError("volume/slot layout differs")
    if any(s[0] not in ("Sagittal", "Coronal", "Axial") for s in slots):
        raise ValueError("unknown plane")
    if side != "L":
        return volume
    output = volume.copy()
    offset = 0
    for plane, _, count in slots:
        end = offset + int(count)
        if plane in ("Coronal", "Axial"):
            output[offset:end] = volume[offset:end, :, ::-1]
        offset = end
    return output


class CanonicalCorpus:
    """Same corpus interface and mask, usable by both training and inference."""
    def __init__(self, corpus, side_map):
        self.base, self.side_map = corpus, side_map
        side_map.require(corpus.ids)

    def __getattr__(self, key):
        if key in ("base", "side_map"):
            raise AttributeError(key)
        return getattr(self.base, key)

    def volume(self, uid):
        return canonical_volume(self.base.volume(uid), self.side_map.side(uid), self.base.spec["slots"])

    def mask(self, uid):
        return self.base.mask(uid)


class CanonicalDataset:
    """Spawn-safe decorator for the shared MIL dataset, preserving its RNG calls.

    The shared factory constructs its normal dataset; this decorator supplies a
    corpus view before __getitem__ selects windows, resizes or augments. It does
    not copy the shared runner's sampling, augmentation or inference code.
    """
    def __init__(self, dataset, side_map):
        if not all(hasattr(dataset, k) for k in ("corpus", "corpus_dir", "ids", "set_epoch", "k")):
            raise ValueError("shared MIL dataset interface differs")
        side_map.require(dataset.ids)
        self.dataset, self.side_map = dataset, side_map

    def __len__(self):
        return len(self.dataset)

    @property
    def k(self):
        return self.dataset.k

    def set_epoch(self, epoch):
        return self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        from training.src.raptor_common import Corpus
        if self.dataset.corpus is None:
            self.dataset.corpus = CanonicalCorpus(Corpus(self.dataset.corpus_dir), self.side_map)
        elif not isinstance(self.dataset.corpus, CanonicalCorpus):
            self.dataset.corpus = CanonicalCorpus(self.dataset.corpus, self.side_map)
        return self.dataset[index]
