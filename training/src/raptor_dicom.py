"""Passive DICOM input adapters for our fixed-slot Raptor checkpoints.

Healthy-input math is a manual refactor of raptor_corpus.py v1 (2026-10-02),
not of the public anchor notebook's later dense-96 sampler. There is no training,
acquisition, backend configuration, or file access at import. Construction reads
explicit metadata; pixel reads happen only on volume()/mask(). Original corpora
are never modified. The legacy mask means positive post-normalization pixel sum,
not confirmed acquisition presence; receipts keep acquisition status separately.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UID = "StudyInstanceUID"
SERIES = "SeriesInstanceUID"
ADAPTER_VERSION = "raptor-dicom-fixed-slot-v1"
HEADER_READER_VERSION = "directory-header-only-v1"


@dataclass(frozen=True)
class DicomMember:
    path: str
    info: str | None = None


class DicomPreparationError(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def _component(value):
    value = str(value)
    if not value or value in {".", ".."} or any(c in value for c in ("/", "\\", ":", "\x00")):
        raise ValueError(f"invalid DICOM directory identifier: {value!r}")
    return value


class _CountedHeaderReader:
    """Count bytes returned to pydicom, not filesystem/block/cache traffic."""

    def __init__(self, handle):
        self.handle = handle
        self.bytes_returned = 0

    def read(self, size=-1):
        value = self.handle.read(size)
        self.bytes_returned += len(value)
        return value

    def __getattr__(self, name):
        return getattr(self.handle, name)


class DirectoryDicomSource:
    """Read a directory containing <study>/<series>/*.dcm, with no extraction."""

    def __init__(self, directory, *, header_only=False):
        if not isinstance(header_only, bool):
            raise TypeError("header_only must be an explicit bool")
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        self.descriptor = {"kind": "directory", "path": str(self.directory)}
        self.header_only = header_only
        self.header_io = {"files_attempted": 0, "files_completed": 0, "logical_bytes_read": 0,
                          "seconds": 0., "scope": "header reads only; bytes returned to pydicom, not physical I/O"}
        if header_only:
            self.descriptor["header_reader"] = HEADER_READER_VERSION

    def members(self, split, uid, series):
        directory = self.directory / _component(uid) / _component(series)
        if not directory.resolve().is_relative_to(self.directory):
            raise ValueError("DICOM directory escapes its explicit source")
        if not directory.is_dir():
            return []
        paths = sorted(p for p in directory.iterdir() if p.is_file() and p.name.endswith(".dcm"))
        if any(not p.resolve().is_relative_to(self.directory) for p in paths):
            raise ValueError("DICOM file escapes its explicit source")
        return [DicomMember(p.relative_to(self.directory).as_posix()) for p in paths]

    def _member_path(self, member):
        path = (self.directory / member.path).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("DICOM member escapes its explicit source")
        return path

    def read(self, member):
        return self._member_path(member).read_bytes()

    def read_header(self, member):
        """Opt-in streaming header parse; selected pixels still use full read().

        The default deliberately delegates through read() to preserve existing
        instrumentation and behavior. No specific_tags filter is used: all
        pre-pixel metadata and pydicom errors remain visible. Deflated datasets
        may still require the full compressed payload. This is not a CRC audit.
        """
        import pydicom
        if not self.header_only:
            return pydicom.dcmread(io.BytesIO(self.read(member)), stop_before_pixels=True)
        started, reader = time.perf_counter(), None
        self.header_io["files_attempted"] += 1
        try:
            with self._member_path(member).open("rb") as handle:
                reader = _CountedHeaderReader(handle)
                header = pydicom.dcmread(reader, stop_before_pixels=True)
            self.header_io["files_completed"] += 1
            return header
        finally:
            self.header_io["seconds"] += time.perf_counter() - started
            if reader is not None:
                self.header_io["logical_bytes_read"] += reader.bytes_returned

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _IndexedZip(zipfile.ZipFile):
    """Existing index supplies ZipInfo; do not scan the large central directory."""

    def _RealGetContents(self):
        self.start_dir = os.fstat(self.fp.fileno()).st_size


class IndexedZipDicomSource:
    """Read only an explicitly named ZIP and KneeOps files index (SQLite mode=ro).

    No application DB, DataManager, corpus builder, or acquisition script is used.
    Use one instance per process; sqlite/ZipFile are intentionally not shared workers.
    """

    def __init__(self, archive, index):
        self.archive_path, self.index_path = Path(archive).resolve(), Path(index).resolve()
        if not self.archive_path.is_file() or not self.index_path.is_file():
            raise FileNotFoundError(f"ZIP/index absent: {self.archive_path}, {self.index_path}")
        self.connection = sqlite3.connect(self.index_path.as_uri() + "?mode=ro", uri=True)
        self.connection.execute("PRAGMA query_only=ON")
        try:
            self.archive = _IndexedZip(self.archive_path, mode="r")
        except Exception:
            self.connection.close()
            raise
        self.descriptor = {"kind": "indexed_zip", "archive": str(self.archive_path),
                           "index": str(self.index_path), "archive_bytes": self.archive_path.stat().st_size,
                           "sqlite_mode": "ro", "index_used_without_rebuild": True}

    def members(self, split, uid, series):
        rows = self.connection.execute(
            "SELECT path,info FROM files WHERE split=? AND uid=? AND series=? ORDER BY path",
            (str(split), str(uid), str(series))).fetchall()
        return [DicomMember(str(path), info) for path, info in rows]

    def read(self, member):
        entry = zipfile.ZipInfo(member.path)
        for key, value in json.loads(member.info).items():
            setattr(entry, key, value)
        with self.archive.open(entry, "r") as handle:
            return handle.read()

    def close(self):
        self.archive.close()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _flag(value):
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y"}:
            return 1
        if text in {"false", "no", "n"}:
            return 0
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return int(number) if np.isfinite(number) and number in (0.0, 1.0) else None


def pick_series(rows, plane, fluid, used):
    """Same first-row preference as corpus v1 on valid 0/1 metadata.

    Unknown flags are neither true nor false; only plane fallback is allowed.
    This is an explicit robustness change for malformed/unknown metadata.
    """
    candidates = [r for r in rows if r["Anatomical_Plane"] == plane and r[SERIES] not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates if _flag(r.get("Fluid_Sensitive")) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def _contract_spec(contract):
    # Import only a passive helper on explicit construction; no workspace read at import.
    from raptor_common import CONTRACTS
    if contract not in CONTRACTS:
        raise ValueError(f"unsupported corpus contract: {contract!r}")
    c = CONTRACTS[contract]
    return {"slots": [list(s) for s in c["slots"]], "depth": sum(s[2] for s in c["slots"]),
            "img": c["img"], "crop_mm": c["crop_mm"], "span": list(c["span"]),
            "normalization": "series percentile 2-98 over picked slices, clip, uint8",
            "mask_semantics": "positive normalized pixel sum (legacy corpus contract)"}


def _ordered_members(source, members, slot_receipt):
    import pydicom
    records, spacings = [], []
    for member in members:  # Source order is lexical; stable position ties preserve it.
        try:
            read_header = getattr(source, "read_header", None)
            header = (read_header(member) if callable(read_header) else
                      pydicom.dcmread(io.BytesIO(source.read(member)), stop_before_pixels=True))
            iop = getattr(header, "ImageOrientationPatient", None)
            ipp = getattr(header, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                iop, ipp = np.asarray(iop, float), np.asarray(ipp, float)
                position = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
            else:
                position = float(getattr(header, "InstanceNumber", 0) or 0)
                slot_receipt["instance_order_fallbacks"] += 1
            spacing = getattr(header, "PixelSpacing", None)
            spacing = float(spacing[0]) if spacing is not None else 0.5
            if not np.isfinite(position) or not np.isfinite(spacing):
                raise ValueError("nonfinite position or pixel spacing")
        except Exception as exc:
            slot_receipt.update(status="header_failed", failed_source=member.path)
            raise RuntimeError(f"DICOM header failed: {member.path}: {exc}") from exc
        records.append((position, member, spacing)); spacings.append(spacing)
    records.sort(key=lambda r: r[0])
    order = [[r[1].path, r[0], r[2]] for r in records]
    slot_receipt["ordered_sources_sha256"] = hashlib.sha256(
        json.dumps(order, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return [(member, spacing) for _, member, spacing in records], float(np.median(spacings))


def _decode(source, member):
    import pydicom
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    raw = source.read(member)
    dataset = pydicom.dcmread(io.BytesIO(raw))
    pixels = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError("expected finite 2-D DICOM pixels")
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        pixels = pixels.max() - pixels
    return pixels, hashlib.sha256(raw).hexdigest()


def prepare_study(uid, rows, source, spec, split="test"):
    """Render one fixed-slot stack, raising with a receipt on acquired-input failure.

    Percentiles include repeated selected indices exactly as corpus v1 did. No
    laterality normalization is added. Healthy output must match corpus bytes.
    """
    import cv2
    depth, img = int(spec["depth"]), int(spec["img"])
    volume = np.zeros((depth, img, img), np.uint8)
    receipt = {"uid": str(uid), "split": str(split), "adapter": ADAPTER_VERSION,
               "status": "preparing", "slots": [], "acquired_slices": 0,
               "metadata_rows": len(rows), "mask_semantics": "positive normalized pixel sum"}
    offset, used = 0, set()
    try:
        for slot_index, (plane, fluid, count) in enumerate(spec["slots"]):
            sr = {"slot": slot_index, "plane": plane, "fluid": fluid, "width": count,
                  "offset": offset, "status": "metadata_absent", "instance_order_fallbacks": 0}
            receipt["slots"].append(sr)
            record = pick_series(rows, plane, fluid, used)
            if record is None:
                offset += count
                continue
            series = str(record[SERIES]); used.add(series)
            sr.update(series=series, status="acquired", protocol_flag=_flag(record.get("Fluid_Sensitive")))
            members = source.members(split, str(uid), series)
            if not members:
                sr["status"] = "acquired_files_missing"
                raise RuntimeError(f"metadata names acquired series without DICOM files: {uid}/{series}")
            files, median_spacing = _ordered_members(source, members, sr)
            n = len(files)
            low = int(n * spec["span"][0]); high = max(int(n * spec["span"][1]) - 1, low)
            selected = np.linspace(low, high, count).round().astype(int) if n > 1 else np.zeros(count, int)
            arrays, source_hashes = {}, {}
            for index in sorted(set(map(int, selected))):
                member = files[min(index, n - 1)][0]
                try:
                    arrays[index], digest = _decode(source, member)
                except Exception as exc:
                    sr.update(status="decode_failed", failed_source=member.path)
                    raise RuntimeError(f"DICOM pixels failed: {member.path}: {exc}") from exc
                source_hashes[index] = {"path": member.path, "sha256": digest}
            # Intentionally include each repeated sample in the percentile population.
            loq, hiq = np.percentile(np.concatenate([arrays[int(i)].ravel() for i in selected]), [2.0, 98.0])
            sr.update(files=n, slices=selected.tolist(), normalization=[float(loq), float(hiq)],
                      selected_sources=[source_hashes[i] for i in sorted(source_hashes)],
                      median_spacing=median_spacing)
            for index in selected:
                pixels = arrays[int(index)]
                spacing = files[min(int(index), n - 1)][1]
                spacing = spacing if spacing > 0 else median_spacing
                normalized = np.clip((pixels - loq) / (hiq - loq + 1e-6), 0, 1)
                height, width = normalized.shape
                crop = min(int(round(float(spec["crop_mm"]) / max(spacing, 1e-3))), min(height, width))
                if crop <= 0:
                    sr["status"] = "render_failed"
                    raise ValueError(f"empty physical crop for {uid}/{series}")
                y0, x0 = (height - crop) // 2, (width - crop) // 2
                resized = cv2.resize(normalized[y0:y0 + crop, x0:x0 + crop], (img, img), interpolation=cv2.INTER_AREA)
                volume[offset] = (resized * 255).astype(np.uint8)
                offset += 1
            sr.update(status="rendered", rendered=count)
            receipt["acquired_slices"] += count
        if offset != depth:
            raise ValueError(f"slot depth {offset} does not match {depth}")
        mask = (volume.reshape(depth, -1).sum(1) > 0).astype(np.uint8)
        receipt.update(status="metadata_absent" if not used else "rendered",
                       acquired_slots=len(used), positive_mask_slices=int(mask.sum()),
                       volume_sha256=hashlib.sha256(memoryview(volume).cast("B")).hexdigest(),
                       mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
        volume.setflags(write=False); mask.setflags(write=False)
        return volume, mask, receipt
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise DicomPreparationError(f"study {uid}: {exc}", receipt) from exc


def _csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class DicomCorpus:
    """Corpus interface backed by unseen DICOM UIDs, retaining at most one study.

    receipts contain metadata and hashes, not pixel arrays. from_records is useful
    for explicit ZIP parity on selected training UIDs without reading train labels.
    Source lifetime belongs to the caller (usually a source context manager).
    """

    def __init__(self, test_csv, series_csv, source, contract="raptor64_336_c140_s02", split="test"):
        tests, rows = _csv_rows(test_csv), _csv_rows(series_csv)
        if not tests or UID not in tests[0]:
            raise ValueError("test CSV requires nonempty StudyInstanceUID rows")
        self._initialize([r[UID] for r in tests], rows, source, contract, split)
        self.manifest["metadata"] = {"test_csv_sha256": hashlib.sha256(Path(test_csv).read_bytes()).hexdigest(),
                                     "series_csv_sha256": hashlib.sha256(Path(series_csv).read_bytes()).hexdigest()}

    @classmethod
    def from_records(cls, ids, rows, source, contract="raptor64_336_c140_s02", split="test"):
        obj = cls.__new__(cls)
        obj._initialize(ids, rows, source, contract, split)
        return obj

    def _initialize(self, ids, rows, source, contract, split):
        self.ids = list(map(str, ids))
        if not self.ids or any(not u for u in self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError("expected nonempty unique study IDs")
        self.source, self.contract, self.split = source, contract, str(split)
        self.spec = _contract_spec(contract); self.depth = self.spec["depth"]
        self.rows = {uid: [] for uid in self.ids}
        seen = set()
        for row in rows:
            if any(key not in row for key in (UID, SERIES, "Anatomical_Plane")):
                raise ValueError("series metadata missing UID, series UID, or plane")
            uid = str(row[UID])
            if uid not in self.rows:
                continue
            series = str(row[SERIES])
            if not series or (uid, series) in seen:
                raise ValueError(f"empty/duplicate series metadata for {uid}/{series}")
            seen.add((uid, series))
            # Do not retain arbitrary metadata columns, labels, or Reports.
            self.rows[uid].append({UID: uid, SERIES: series, "Anatomical_Plane": row["Anatomical_Plane"],
                                   "Fluid_Sensitive": row.get("Fluid_Sensitive")})
        self.receipts = {}; self._cached_uid = None; self._cached = None
        self.manifest = {"adapter": ADAPTER_VERSION, "contract": contract, "spec": self.spec,
                         "source": dict(source.descriptor), "split": self.split,
                         "ids_sha256": hashlib.sha256("\n".join(self.ids).encode()).hexdigest(),
                         "cache_max_studies": 1, "acquired_failure_policy": "raise"}

    def __len__(self):
        return len(self.ids)

    def __contains__(self, uid):
        return str(uid) in self.rows

    def _ensure(self, uid):
        uid = str(uid)
        if uid not in self.rows:
            raise KeyError(uid)
        if self._cached_uid != uid:
            self._cached_uid = None; self._cached = None
            try:
                volume, mask, receipt = prepare_study(uid, self.rows[uid], self.source, self.spec, self.split)
            except DicomPreparationError as exc:
                self.receipts[uid] = exc.receipt
                raise
            self.receipts[uid] = receipt
            self._cached_uid = uid; self._cached = (volume, mask)
        return self._cached

    def volume(self, uid):
        return self._ensure(uid)[0]

    def mask(self, uid):
        return self._ensure(uid)[1]
