"""EXP081: CPU-only, frozen-weight coronal-fluid input preparation.

No training, model loading, label access, inference, remote job or submission.
Original sources/cache are immutable; each run requires a new output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UID, SERIES = "StudyInstanceUID", "SeriesInstanceUID"
EXPERIMENT = "EXP081_CORONAL_SPAN_FIXED_F2"
TRANSFORM = "exp081-coronal-fluid-max-center-span-v1"
START, STOP = 32, 44
MAX_PATCH_BYTES = 85_349_376
EXPECTED_IDS = "6632df20158b6d5f0f756242ac220d3d65b0321109228288ef84d368576a2ede"
DRAFT = "docs/handoffs/evidence/2026-10/2026-10-03-series-inventory/v5/protocol-draft.json"
DRAFT_SHA = "3434148f93078dda5dce27e6fa40b4d5aed3de40fe2fa492589aae204fdf7655"
CORPUS = "storage/cache/derived/RAPTOR64_336_C140_S02"
FOLDS = "artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv"
ARCHIVE = "storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip"
INDEX = ".local/kneeops/archive-index/9175e113e4a00fe679c4088b.sqlite"
BASE = "artifacts/experiments/EXP080_INPUT_B_F2"
LOG = "artifacts/logs/EXP080_INPUT_B_F2"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def uid_digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def dump(path, value, *, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def uri(path):
    return "workspace://" + Path(path).relative_to(ROOT).as_posix()


def flag(value):
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return int(value.lower() == "true")
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number in (0, 1) else None
    except (ValueError, TypeError):
        return None


class HeaderBudgetExceeded(RuntimeError):
    pass


class HeaderBudget:
    def __init__(self, max_bytes=256 * 1024**2, max_seconds=600):
        self.max_bytes = int(max_bytes)
        self.max_seconds = float(max_seconds)
        self.started = time.perf_counter()
        self.bytes_read = 0
        self.files = 0
        self.max_high_water = 0

    def check(self, requested=0):
        if time.perf_counter() - self.started >= self.max_seconds:
            raise HeaderBudgetExceeded("header wall-time cap reached")
        if requested < 0 or self.bytes_read + requested > self.max_bytes:
            raise HeaderBudgetExceeded("total header API byte cap reached")

    def receipt(self):
        return dict(bytes_read=self.bytes_read, files=self.files,
                    max_high_water=self.max_high_water,
                    seconds=time.perf_counter() - self.started,
                    max_bytes=self.max_bytes, max_seconds=self.max_seconds,
                    byte_scope="Bytes returned to parser, not compressed/block I/O; ZIP can prefetch.")


class BoundedHeader:
    def __init__(self, handle, budget, member_limit=2 * 1024**2):
        self.handle, self.budget = handle, budget
        self.member_limit = int(member_limit)
        self.read_bytes = self.high_water = 0

    def tell(self):
        return self.handle.tell()

    def seek(self, offset, whence=0):
        self.budget.check()
        if whence not in (0, 1):
            raise HeaderBudgetExceeded("seek-to-end denied")
        target = offset if whence == 0 else self.tell() + offset
        if target < 0 or target > self.member_limit:
            raise HeaderBudgetExceeded("seek beyond member header cap")
        position = self.handle.seek(offset, whence)
        self.high_water = max(self.high_water, position)
        self.budget.max_high_water = max(self.budget.max_high_water, self.high_water)
        return position

    def read(self, size=-1):
        if size is None or size < 0:
            raise HeaderBudgetExceeded("unbounded/deflated payload read denied")
        self.budget.check(size)
        if self.tell() + size > self.member_limit or self.read_bytes + size > self.member_limit:
            raise HeaderBudgetExceeded("member header byte cap reached")
        value = self.handle.read(size)
        self.read_bytes += len(value)
        self.budget.bytes_read += len(value)
        self.high_water = max(self.high_water, self.tell())
        self.budget.max_high_water = max(self.budget.max_high_water, self.high_water)
        return value

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)


def _vector(value, length):
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
        return result if len(result) == length and np.isfinite(result).all() else None
    except (ValueError, TypeError):
        return None


def _angle(first, second):
    # Signed dot deliberately rejects 180-degree flips.
    first, second = first / np.linalg.norm(first), second / np.linalg.norm(second)
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1, 1))))


def geometry(headers, uid, series):
    """Reject incomplete geometry deterministically; never infer anatomical coverage."""
    def reject(reason):
        return {"eligible": False, "reason": reason, "headers": len(headers)}
    if not headers:
        return reject("no_headers")
    normals, axes, positions, frames = [], [], [], []
    for header in headers:
        if str(header.get(UID, "")) != uid or str(header.get(SERIES, "")) != series:
            return reject("uid_mismatch")
        orient = _vector(header.get("ImageOrientationPatient"), 6)
        point = _vector(header.get("ImagePositionPatient"), 3)
        spacing = _vector(header.get("PixelSpacing"), 2)
        if orient is None or point is None or spacing is None:
            return reject("missing_or_nonfinite_geometry")
        try:
            if int(header.get("Rows", 0)) <= 0 or int(header.get("Columns", 0)) <= 0 or min(spacing) <= 0:
                return reject("invalid_matrix_or_spacing")
            if int(header.get("NumberOfFrames", 1)) != 1:
                return reject("multiframe")
        except (TypeError, ValueError, OverflowError):
            return reject("invalid_matrix_or_frames")
        image_type = header.get("ImageType", [])
        if isinstance(image_type, str):
            image_type = image_type.split("\\")
        if any("MOSAIC" in str(value).upper() for value in image_type):
            return reject("mosaic")
        first, second = orient[:3], orient[3:]
        if max(abs(np.linalg.norm(first) - 1), abs(np.linalg.norm(second) - 1), abs(np.dot(first, second))) > .001:
            return reject("invalid_iop")
        normal = np.cross(first, second)
        normal /= np.linalg.norm(normal)
        normals.append(normal); axes.append((first, second)); positions.append(point)
        frames.append(str(header.get("FrameOfReferenceUID") or ""))
    if not all(frames) or len(set(frames)) != 1:
        return reject("missing_or_inconsistent_frame")
    if int(np.argmax(np.abs(normals[0]))) != 1:
        return reject("not_coronal_dominant_normal")
    normal_drift = max(_angle(normals[0], n) for n in normals)
    axis_drift = max(max(_angle(axes[0][0], a), _angle(axes[0][1], b)) for a, b in axes)
    if normal_drift > 1.0 or axis_drift > 1.0:
        return reject("orientation_drift")
    projections = np.sort(np.asarray(positions) @ normals[0])
    gaps = np.diff(projections)
    if len(projections) < 3:
        return reject("fewer_than_three_positions")
    if np.any(gaps <= .001):
        return reject("duplicate_projected_positions")
    ratio = float(gaps.max() / np.median(gaps))
    if ratio > 1.5:
        return reject("irregular_gaps")
    extent = float(projections[-1] - projections[0])
    return dict(eligible=True, reason="eligible", headers=len(headers), extent_mm=extent,
                extent_bucket=int(round(extent / .01)), normal=normals[0].tolist(),
                normal_drift_degrees=normal_drift, axis_drift_degrees=axis_drift,
                gap_ratio=ratio, frame_uid=frames[0],
                interpretation="Slice-center acquisition length; not anatomical coverage or image quality.")


def choose_series(current, candidates, geometry_by_series, used_other):
    current_geometry = geometry_by_series.get(current, {})
    if not current_geometry.get("eligible"):
        return current
    eligible = [s for s in set(candidates) | {current}
                if (s == current or s not in used_other) and geometry_by_series.get(s, {}).get("eligible")]
    maximum = max(int(round(geometry_by_series[s]["extent_mm"] / .01)) for s in eligible)
    best = [s for s in eligible if int(round(geometry_by_series[s]["extent_mm"] / .01)) == maximum]
    return current if current in best else min(best)


def verify_overlay(original, original_mask, changed, changed_mask):
    for volume in (original, changed):
        if volume.shape != (64, 336, 336) or volume.dtype != np.uint8:
            raise ValueError("overlay requires uint8[64,336,336]")
    for mask in (original_mask, changed_mask):
        if mask.shape != (64,) or mask.dtype != np.uint8 or not np.isin(mask, [0, 1]).all():
            raise ValueError("overlay requires uint8 binary mask[64]")
    for region in (slice(0, START), slice(STOP, None)):
        if not np.array_equal(original[region], changed[region]) or not np.array_equal(original_mask[region], changed_mask[region]):
            raise ValueError("untouched slot bytes changed")
    expected = (changed[START:STOP].reshape(12, -1).sum(1) > 0).astype(np.uint8)
    if not np.array_equal(expected, changed_mask[START:STOP]):
        raise ValueError("patch mask disagrees with positive pixel sum")
    return {"untouched_bytes_equal": True, "patch_mask_valid": True}


class OverlayCorpus:
    """Explicit changed-input contract; never impersonates the original contract."""
    def __init__(self, baseline, patch_ids, patches, masks):
        self.baseline, self.ids = baseline, baseline.ids
        self.spec, self.depth = baseline.spec, baseline.depth
        self.contract = TRANSFORM
        self.row = {uid: i for i, uid in enumerate(patch_ids)}
        if len(self.row) != len(patch_ids):
            raise ValueError("duplicate patch UID")
        if any(uid not in set(map(str, baseline.ids)) for uid in patch_ids):
            raise ValueError("patch UID absent from baseline")
        self.patches, self.masks = patches, masks
        if patches.shape != (len(patch_ids), 12, 336, 336) or patches.dtype != np.uint8:
            raise ValueError("invalid patch tensor")
        if masks.shape != (len(patch_ids), 12) or masks.dtype != np.uint8:
            raise ValueError("invalid patch masks")
        if not np.isin(masks, [0, 1]).all() or not np.array_equal(
                masks, (patches.reshape(len(patch_ids), 12, 336 * 336).sum(2) > 0).astype(np.uint8)):
            raise ValueError("patch masks disagree with positive pixel sum")
        self.manifest = {"transform": TRANSFORM, "parent_contract": baseline.contract,
                         "shape_compatibility_only": True, "requires_explicit_comparison_adapter": True}

    def volume(self, uid):
        base = self.baseline.volume(uid)
        if uid not in self.row:
            return base
        result = base.copy(); result[START:STOP] = self.patches[self.row[uid]]
        result.setflags(write=False)
        return result

    def mask(self, uid):
        base = self.baseline.mask(uid)
        if uid not in self.row:
            return base
        result = base.copy(); result[START:STOP] = self.masks[self.row[uid]]
        result.setflags(write=False)
        return result


def _adapters():
    sys.path.insert(0, str(ROOT / "training/src"))
    import raptor_dicom
    import raptor_common
    return raptor_dicom, raptor_common


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _header_dict(ds):
    result = {UID: str(getattr(ds, UID, "")), SERIES: str(getattr(ds, SERIES, ""))}
    for name, width in (("ImageOrientationPatient", 6), ("ImagePositionPatient", 3), ("PixelSpacing", 2)):
        value = _vector(getattr(ds, name, None), width)
        result[name] = None if value is None else value.tolist()
    for key in ("Rows", "Columns", "NumberOfFrames"):
        result[key] = int(getattr(ds, key, 1 if key == "NumberOfFrames" else 0))
    result["FrameOfReferenceUID"] = str(getattr(ds, "FrameOfReferenceUID", "") or "")
    image_type = getattr(ds, "ImageType", []) or []
    if isinstance(image_type, str):
        image_type = image_type.split("\\")
    result["ImageType"] = [str(value) for value in image_type]
    for key in ("EchoTime", "RepetitionTime", "SliceThickness", "SpacingBetweenSlices"):
        vector = _vector(getattr(ds, key, None), 1)
        result[key] = None if vector is None else float(vector[0])
    return result


class ScreenSource:
    def __init__(self, source, budget):
        self.source, self.budget = source, budget
        self.headers, self.header_records = {}, {}
        self.pixel_reads = self.pixel_bytes = 0
        self.pixel_enabled = False
        self.active_member = None

    def members(self, *args):
        return self.source.members(*args)

    def read_header(self, member):
        if member.path in self.headers:
            return self.headers[member.path]
        import pydicom
        self.active_member = member.path
        self.budget.check()
        entry = zipfile.ZipInfo(member.path)
        info = json.loads(member.info)
        for key, value in info.items():
            setattr(entry, key, value)
        with self.source.archive.open(entry, "r") as stream:
            reader = BoundedHeader(stream, self.budget)
            ds = pydicom.dcmread(reader, stop_before_pixels=True)
        if "PixelData" in ds:
            raise RuntimeError("pixel data parsed during header-only stage")
        self.budget.files += 1
        self.headers[member.path] = ds
        self.header_records[member.path] = dict(path=member.path, header=_header_dict(ds),
            api_bytes=reader.read_bytes, high_water=reader.high_water,
            zip_crc32=info.get("CRC"), zip_offset=info.get("header_offset"),
            zip_member_bytes=info.get("file_size"), zip_compressed_bytes=info.get("compress_size"))
        return ds

    def read(self, member):
        if not self.pixel_enabled:
            raise RuntimeError("pixels forbidden until geometry and map are frozen")
        value = self.source.read(member)
        self.pixel_reads += 1; self.pixel_bytes += len(value)
        return value


def metadata_plan(draft, corpus, rd):
    folds = _csv(ROOT / FOLDS)
    excluded = set(draft["cohort"]["excluded_uids"])
    selected = [r for r in folds if r["fold"] == "2" and r[UID] not in excluded]
    if any(flag(r["is_gold"]) != 0 or flag(r["quarantined_gold_report_duplicate"]) != 0 for r in selected):
        raise ValueError("unexpected exposed Gold/quarantine flag in F2")
    ids = sorted(r[UID] for r in selected)
    if len(ids) != 868 or len(set(ids)) != 868 or uid_digest(ids) != EXPECTED_IDS:
        raise ValueError("F2 UID population mismatch")
    if len(set(r["report_group_sha256"] for r in selected)) != 840 or not all(u in corpus for u in ids):
        raise ValueError("F2 groups/corpus mismatch")
    rows = {uid: [] for uid in ids}
    for row in _csv(ROOT / "storage/competition/train_series.csv"):
        if row[UID] in rows:
            rows[row[UID]].append(row)
    picks = {}
    with (ROOT / CORPUS / "picks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["uid"] in rows:
                if row["uid"] in picks:
                    raise ValueError("duplicate baseline pick UID")
                picks[row["uid"]] = row
    plans, screen = [], {}
    for uid in ids:
        original = [r.get("series") if r else None for r in picks[uid]["slots"]]
        replay, used = [], set()
        for plane, fluid, _ in corpus.spec["slots"]:
            chosen = rd.pick_series(rows[uid], plane, fluid, used)
            series = None if chosen is None else chosen[SERIES]
            replay.append(series)
            if series:
                used.add(series)
        if replay != original:
            raise ValueError(f"original five-slot selection replay mismatch: {uid}")
        current = original[2]
        mapping = {r[SERIES]: r for r in rows[uid]}
        if len(mapping) != len(rows[uid]):
            raise ValueError("duplicate series metadata")
        other = sorted(s for i, s in enumerate(original) if i != 2 and s)
        current_row = mapping.get(current, {})
        eligible = (current_row.get("Anatomical_Plane") == "Coronal" and
                    flag(current_row.get("Fluid_Sensitive")) == 1 and
                    flag(current_row.get("Fat_Suppression")) in (0, 1))
        alternatives = sorted(r[SERIES] for r in rows[uid]
            if eligible and r[SERIES] != current and r[SERIES] not in other and
            r["Anatomical_Plane"] == "Coronal" and flag(r.get("Fluid_Sensitive")) == 1 and
            flag(r.get("Fat_Suppression")) == flag(current_row.get("Fat_Suppression")))
        reason = "screen_pending" if alternatives else ("no_comparable_alternative" if eligible else "current_fallback_or_unknown")
        plans.append(dict(uid=uid, original_series=original, current=current, alternatives=alternatives,
                          other_slots=other, selected=current, reason=reason, changed=False))
        if alternatives:
            for series in [current, *alternatives]:
                screen[(uid, series)] = mapping[series]
    if len(screen) != 129 or sum(bool(p["alternatives"]) for p in plans) != 63:
        raise ValueError("metadata screen scope differs from registered129/63")
    return ids, plans, screen


def prepare(output):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    progress = {"experiment": EXPERIMENT, "pid": os.getpid(), "started_at": now(), "stage": "registration",
                "gpu_authorized": False, "cpu_only": True}
    dump(output / "progress.json", progress)
    budget = None
    geometry_rows, source = [], None
    try:
        if sha(ROOT / DRAFT) != DRAFT_SHA:
            raise ValueError("approved draft changed")
        draft = json.loads((ROOT / DRAFT).read_text(encoding="utf-8"))
        bindings = {DRAFT: DRAFT_SHA}
        # Bind metadata/reference sources only; do not open any labels/results.
        for location, expected in draft["source_hashes"].items():
            relative = location.removeprefix("workspace://")
            actual = sha(ROOT / relative)
            if actual != expected:
                raise ValueError(f"frozen source changed: {relative}")
            bindings[relative] = actual
        for relative in (f"{CORPUS}/ids.npy", f"{CORPUS}/masks.npy", f"{BASE}/model_best.pt",
                         f"{BASE}/provenance.json", f"{LOG}/job.json", f"{LOG}/SUCCESS",
                         "artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL/exp080_queue.json"):
            bindings[relative] = sha(ROOT / relative)
        for relative, expected in {
            f"{CORPUS}/ids.npy": "48cf9dcbd34276c898dabfd09d901d5288b272522328016076da0b9189a3246a",
            f"{CORPUS}/masks.npy": "011696644468b5a624c9e7862be5a853444e1c98640a77118ba996fd6b437ea2",
            "artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL/exp080_queue.json": "eea5394ab34d0f5250b852c63cdf00d188f98bcb24bbe7d191881492def4e774",
        }.items():
            if bindings[relative] != expected:
                raise ValueError(f"frozen population/cache/queue hash mismatch: {relative}")
        provenance = json.loads((ROOT / BASE / "provenance.json").read_text(encoding="utf-8"))
        job = json.loads((ROOT / LOG / "job.json").read_text(encoding="utf-8"))
        command = job["command"]
        for argument, expected in (("--fixed-epoch", "8"), ("--epochs", "8"), ("--fold", "2"),
                                   ("--window-policy", "within-slot-valid"), ("--bs", "8")):
            if command[command.index(argument) + 1] != expected:
                raise ValueError(f"baseline run argument mismatch: {argument}")
        if (provenance["checkpoint_selection"] != "fixed-final" or
            provenance["eval_ids_sha256"] != EXPECTED_IDS or
            provenance["initial_state_sha256"] != draft["baseline"]["original_initial_state_sha256"] or
            provenance["train_ids_sha256"] != draft["cohort"]["training_population_reference"]["train_ids_sha256"]):
            raise ValueError("baseline initialization/selection/population mismatch")
        if not (ROOT / LOG / "SUCCESS").read_text().startswith("exit 0"):
            raise ValueError("baseline did not finish successfully")
        archive_stat = (ROOT / ARCHIVE).stat()
        signature = [archive_stat.st_size, archive_stat.st_mtime_ns]
        if signature != [265018939620, 1789814663000000000]:
            raise ValueError("archive stat changed")
        registration = dict(schema="exp081-cpu-registered-v1", experiment=EXPERIMENT, created_at=now(),
            status="CPU_PREPARATION_AUTHORIZED_GPU_NOT_AUTHORIZED", transform=TRANSFORM,
            decision_at="2026-10-03T18:59:00+09:00",
            decision_context="Selected after root inspected EXP080 F2 results; geometry motivation predates those results. Not a demonstrated Fracture fix or untouched holdout.",
            checkpoint_semantics="Raw file SHA plus fixed-final8 launch/provenance/SUCCESS; checkpoint not deserialized and no embedded metrics read.",
            checkpoint_sha256=bindings[f"{BASE}/model_best.pt"], initialization_sha256=provenance["initial_state_sha256"],
            source_bindings={"workspace://" + p: h for p, h in bindings.items()},
            preparation_tool_sha256=sha(Path(__file__)), approved_protocol=draft,
            baseline_runtime=provenance["eval_runtime"], baseline_environment=job["environment"],
            baseline_training_eval_batch_size=4, future_paired_inference_batch_size=8,
            future_runtime="Both newly recomputed868 passes: same weights/order/batch8/strictFP16/backend settings. No savedOOF parity assumption. T4 parity remains a separate measured gate, not implied by deterministic flags.",
            archive_signature=signature, geometry_budget=dict(worker_count=1, headers=3305, series=129, max_seconds=600, max_bytes=256*1024**2),
            pixel_budget=dict(max_patch_payload_bytes=MAX_PATCH_BYTES, no_op_controls_max=63, candidate_studies_max=63),
            prohibited=["label/result reads", "GPU inference", "training", "remote execution", "submission", "source/cache edits"])
        dump(output / "registration.json", registration)
        rd, common = _adapters()
        corpus = common.Corpus(ROOT / CORPUS)
        corpus.masks.setflags(write=False)
        ids, plans, screen = metadata_plan(draft, corpus, rd)
        dump(output / "cohort.json", dict(ids=ids, ids_sha256=EXPECTED_IDS, report_groups=840, label_reads=0))
        dump(output / "metadata-map.json", dict(plans=plans, series=[dict(uid=u, series=s, metadata=r) for (u, s), r in sorted(screen.items())]))
        progress.update(stage="header_screen", updated_at=now()); dump(output / "progress.json", progress, exclusive=False)
        budget = HeaderBudget()
        with rd.IndexedZipDicomSource(ROOT / ARCHIVE, ROOT / INDEX) as raw:
            index_rows = raw.connection.execute("SELECT payload FROM metadata").fetchall()
            if len(index_rows) != 1:
                raise ValueError("archive index metadata is not unique")
            index_metadata = json.loads(index_rows[0][0])
            if (index_metadata.get("signature") != signature or
                index_metadata.get("directory_sha256") != "18bc75eba31d03a12f35e646a337980db06c3cb48b974402396fec9dfe5d7412"):
                raise ValueError("archive index signature/directory digest mismatch")
            dump(output / "archive-index-receipt.json", dict(metadata=index_metadata, index=uri(ROOT / INDEX),
                 read_only=True, central_directory_rescanned=False, full_archive_crc_checked=False))
            source = ScreenSource(raw, budget)
            members = {(u, s): source.members("train", u, s) for u, s in screen}
            if sum(map(len, members.values())) != 3305 or any(not value for value in members.values()):
                raise ValueError("index member count mismatch")
            for number, ((uid, series), row) in enumerate(sorted(screen.items()), 1):
                headers, records = [], []
                for member in members[(uid, series)]:
                    ds = source.read_header(member)
                    if str(getattr(ds, UID, "")) != uid or str(getattr(ds, SERIES, "")) != series:
                        raise ValueError("acquired DICOM UID mismatch")
                    records.append(source.header_records[member.path]); headers.append(records[-1]["header"])
                geometry_rows.append(dict(uid=uid, series=series, metadata=row, geometry=geometry(headers, uid, series), headers=records))
                if number % 16 == 0:
                    progress.update(series_completed=number, header_io=budget.receipt(), updated_at=now())
                    dump(output / "progress.json", progress, exclusive=False)
            screen_receipt = budget.receipt()
            dump(output / "geometry.json", dict(created_at=now(), records=geometry_rows, header_io=screen_receipt, pixel_reads=source.pixel_reads))
            geometry_hash = sha(output / "geometry.json")
            lookup = {(row["uid"], row["series"]): row["geometry"] for row in geometry_rows}
            for plan in plans:
                if not plan["alternatives"]:
                    continue
                current = plan["current"]
                geometries = {s: lookup[(plan["uid"], s)] for s in [current, *plan["alternatives"]]}
                selected = choose_series(current, plan["alternatives"], geometries, set(plan["other_slots"]))
                plan.update(selected=selected, changed=selected != current,
                            reason="greater_center_span" if selected != current else
                            ("current_geometry_ineligible" if not geometries[current]["eligible"] else "current_max_bucket_or_no_eligible_larger"))
            changed = [p for p in plans if p["changed"]]
            if len(changed) * 12 * 336 * 336 > MAX_PATCH_BYTES:
                raise ValueError("patch byte bound exceeded")
            dump(output / "selection-map.json", dict(created_at=now(), geometry_sha256=geometry_hash,
                 metadata_sha256=sha(output / "metadata-map.json"), registration_sha256=sha(output / "registration.json"),
                 transform=TRANSFORM, plans=plans, changed=len(changed), pixel_reads_at_freeze=source.pixel_reads))
            map_hash = sha(output / "selection-map.json")
            progress.update(stage="frozen_noop_replay", changed=len(changed), map_sha256=map_hash, updated_at=now())
            dump(output / "progress.json", progress, exclusive=False)
            # The exclusive on-disk map and geometry are hashed before enabling pixels.
            if source.pixel_reads != 0:
                raise RuntimeError("pixel read before selection freeze")
            source.pixel_enabled = True
            spec = dict(corpus.spec, depth=12, slots=[["Coronal", 1, 12]])
            noops = []
            for plan in plans:
                if not plan["alternatives"]:
                    continue
                uid, current = plan["uid"], plan["current"]
                original = corpus.volume(uid); original_mask = corpus.mask(uid)
                volume, mask, receipt = rd.prepare_study(uid, [screen[(uid, current)]], source, spec, "train")
                equal = np.array_equal(volume, original[START:STOP]) and np.array_equal(mask, original_mask[START:STOP])
                noops.append(dict(uid=uid, series=current, equal=bool(equal), receipt=receipt,
                                  cached_slot_sha256=array_sha(original[START:STOP]), cached_mask_sha256=array_sha(original_mask[START:STOP])))
                if not equal:
                    dump(output / "noop-failure.json", noops[-1])
                    raise ValueError(f"original-slot no-op differs from immutable cache: {uid}")
            dump(output / "noop-replay.json", dict(controls=noops, all_equal=True, controls_count=len(noops)))
            patch_dir = output / "overlay"; patch_dir.mkdir()
            patches = np.zeros((len(changed), 12, 336, 336), dtype=np.uint8)
            masks = np.zeros((len(changed), 12), dtype=np.uint8)
            receipts = []
            progress.update(stage="render_overlay", updated_at=now()); dump(output / "progress.json", progress, exclusive=False)
            for index, plan in enumerate(changed):
                uid, selected = plan["uid"], plan["selected"]
                original, original_mask = corpus.volume(uid), corpus.mask(uid)
                before = array_sha(original); before_mask = array_sha(original_mask)
                patch, mask, receipt = rd.prepare_study(uid, [screen[(uid, selected)]], source, spec, "train")
                candidate, candidate_mask = original.copy(), original_mask.copy()
                candidate[START:STOP], candidate_mask[START:STOP] = patch, mask
                verified = verify_overlay(original, original_mask, candidate, candidate_mask)
                if array_sha(original) != before or array_sha(original_mask) != before_mask:
                    raise ValueError("original corpus changed while patching")
                patches[index], masks[index] = patch, mask
                receipts.append(dict(uid=uid, row=index, series=selected, render=receipt, verification=verified,
                    base_volume_sha256=before, base_mask_sha256=before_mask, candidate_volume_sha256=array_sha(candidate),
                    mask_before=original_mask.tolist(), mask_after=candidate_mask.tolist(),
                    centers_before=common.select_centers(original_mask, 64, 24, corpus.spec, "within-slot-valid"),
                    centers_after=common.select_centers(candidate_mask, 64, 24, corpus.spec, "within-slot-valid")))
            patches.setflags(write=False); masks.setflags(write=False)
            np.save(patch_dir / "patches.npy", patches, allow_pickle=False)
            np.save(patch_dir / "masks.npy", masks, allow_pickle=False)
            patch_ids = [plan["uid"] for plan in changed]
            dump(patch_dir / "ids.json", patch_ids)
            overlay = OverlayCorpus(corpus, patch_ids, patches, masks)
            unchanged = []
            for uid in ids:
                if uid in overlay.row:
                    continue
                base_volume, base_mask = corpus.volume(uid), corpus.mask(uid)
                candidate_volume, candidate_mask = overlay.volume(uid), overlay.mask(uid)
                if not np.shares_memory(base_volume, candidate_volume) or not np.shares_memory(base_mask, candidate_mask):
                    raise ValueError("unchanged UID does not reuse baseline bytes")
                unchanged.append(dict(uid=uid, volume_sha256=array_sha(base_volume), mask_sha256=array_sha(base_mask),
                                      candidate_is_same_readonly_memory=True))
            dump(output / "overlay-receipts.json", dict(changed=receipts, unchanged=unchanged,
                unchanged_scope="Both arms reference the identical read-only original memory; hashes computed once, no redundant6GiB copy.",
                pixel_reads=source.pixel_reads, pixel_bytes=source.pixel_bytes))
        if sha(output / "selection-map.json") != map_hash or sha(output / "geometry.json") != geometry_hash:
            raise ValueError("frozen selection artifacts changed")
        if sha(Path(__file__)) != registration["preparation_tool_sha256"]:
            raise ValueError("preparation implementation changed during execution")
        for relative, expected in bindings.items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f"bound source/cache changed: {relative}")
        final_stat = (ROOT / ARCHIVE).stat()
        if [final_stat.st_size, final_stat.st_mtime_ns] != signature:
            raise ValueError("archive changed during preparation")
        files = {uri(path): sha(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "progress.json"}
        result = dict(status="CPU_PREPARATION_COMPLETE_GPU_NOT_RUN", experiment=EXPERIMENT, completed_at=now(),
            seconds=time.perf_counter() - started, selection_map_sha256=map_hash, geometry_sha256=geometry_hash,
            changed=len(changed), unchanged=len(ids)-len(changed), no_op_controls=len(noops),
            all_no_op_equal=True, patch_payload_bytes=int(patches.nbytes), patch_limit_bytes=MAX_PATCH_BYTES,
            header_io=screen_receipt, pixel_reads=source.pixel_reads, pixel_bytes=source.pixel_bytes,
            reason_counts=dict(Counter(p["reason"] for p in plans)), artifacts=files,
            source_bindings_rechecked=True, inference_authorized=False, labels_opened=False,
            limitations=["Exposed F2; hypothesis selected after EXP080 results", "Maximum span is not anatomical quality", "T4/local numerical parity not established by this CPU input artifact"])
        dump(output / "receipt.json", result)
        progress.update(stage="complete", completed_at=now(), seconds=result["seconds"])
        dump(output / "progress.json", progress, exclusive=False)
        print(json.dumps({key: result[key] for key in ("status", "seconds", "changed", "unchanged", "patch_payload_bytes")}), flush=True)
        return result
    except Exception as exc:
        failure = dict(status="FAILED_STOP_NO_GPU", stage=progress["stage"], error=f"{type(exc).__name__}: {exc}",
                       failed_at=now(), seconds=time.perf_counter()-started, pid=os.getpid(),
                       header_io=None if budget is None else budget.receipt())
        if progress["stage"] == "header_screen":
            dump(output / "geometry-partial.json", dict(completed_series=geometry_rows,
                 completed_headers=[] if source is None else list(source.header_records.values()),
                 active_member=None if source is None else source.active_member,
                 failure=failure, resumable_only_under_new_explicit_run=True))
        dump(output / "failure.json", failure)
        progress.update(stage="failed", error=failure["error"], updated_at=now())
        dump(output / "progress.json", progress, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments" / EXPERIMENT)
    args = parser.parse_args()
    output = args.output.absolute()
    if not output.is_relative_to(ROOT / "artifacts/experiments"):
        parser.error("output must be a new experiment directory under artifacts/experiments")
    prepare(output)


if __name__ == "__main__":
    main()
