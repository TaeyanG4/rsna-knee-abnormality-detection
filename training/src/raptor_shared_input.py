"""Shared-decode input adapter for our fixed-slot Raptor checkpoints (H-B-15, D-B-018).

The 0.943 anchor's maxspan-v5 Raptor arm already decodes every DICOM plane that the
raptor64_336_c140_s02 contract needs: it decodes the union of its own dense picks and
of the 64 contract picks (anchor_source.py A:3979-3981). This module

1. renders our contract from such already-decoded float32 planes plus the caller's
   header order (``render_study`` / ``render_anchor_capture``),
2. keeps the result in a uint8 memmap store with per-study status and sha256
   (``SharedInputStore``), and
3. serves the store to ``raptor_runtime.predict`` through ``StoreCorpus``, which has the
   DicomCorpus interface and falls back to the unchanged DicomCorpus for every row
   whose status is not OK (or whose bytes, picks or audit do not check out).

The contract math is the inline block of ``raptor_dicom.prepare_study`` (span linspace,
p2/p98 over the picks including repeats, clip, centred crop, cv2.INTER_AREA, uint8
truncation, positive-sum mask, zero-padded missing slots). It cannot be imported as a
function, so it is mirrored line for line here and pinned: rendering refuses to run
(status SOURCE_DRIFT, i.e. our DICOM fallback) when the source of
``raptor_dicom.prepare_study`` or ``raptor_dicom._decode`` differs from the pinned
hashes. Series choice, flag parsing, the contract table and the fallback reader are
imported, not copied. No file access, threads or backend changes happen at import.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

import raptor_dicom as _rd
from raptor_dicom import SERIES, DicomCorpus, _contract_spec, pick_series

SHARED_INPUT_VERSION = "raptor-shared-input-v1"
STORE_SCHEMA = "raptor-shared-input-store-v1"
DEFAULT_CONTRACT = "raptor64_336_c140_s02"

# sha256 of inspect.getsource(...) for the raptor_dicom functions whose semantics the
# render mirrors (identical in the shipped EXP089 copy and the current repository file).
PINNED_SOURCES = {
    "prepare_study": "3f8496f0b45de6b5b55e9a5c0c2d0f6af8ac1b49bac4de6e3955acfa39dad3bc",
    "_decode": "9018c2d52644e9b84b4d6de8fcb64ec2afebf3afa3fce4f02ad8be87abb1d529",
}

OK = "OK"
STATUSES = (
    OK,
    "ANCHOR_RAISED",       # the anchor's own preparation raised for this study
    "CAPTURE_INCOMPLETE",  # a contract series or plane was not ordered/decoded by the caller
    "ORDER_MISMATCH",      # caller's file list/spacing cannot equal our header order
    "ORDER_FALLBACK",      # caller reported an InstanceNumber/index/unreadable-header fallback
    "RENDER_FAILED",
    "SOURCE_DRIFT",        # pinned raptor_dicom sources changed
    "DROPPED",             # render queue full; the anchor never waits for us
    "HOOK_ERROR",
)
# Anchor events (anchor_source.py A:3789-3800) after which its order may differ from ours.
ANCHOR_ORDER_EVENTS = frozenset({"raptor_order_index_fallback", "raptor_order_nonfinite_fallback",
                                 "raptor_order_unreadable_skipped"})


class SharedInputError(RuntimeError):
    """A study that must be served by the DICOM fallback; ``code`` is a store status."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def contract_spec(contract=DEFAULT_CONTRACT):
    return _contract_spec(contract)


_PINNED_RESULT = {}


def source_hashes():
    return {name: hashlib.sha256(inspect.getsource(getattr(_rd, name)).encode()).hexdigest()
            for name in PINNED_SOURCES}


def check_pinned_sources():
    """Raise SOURCE_DRIFT unless raptor_dicom still has the mirrored math (cached)."""
    if "value" not in _PINNED_RESULT:
        try:
            actual = source_hashes()
            _PINNED_RESULT["value"] = (actual == PINNED_SOURCES, actual)
        except Exception as exc:  # source unavailable: cannot prove identity
            _PINNED_RESULT["value"] = (False, {"error": f"{type(exc).__name__}: {exc}"})
    ok, actual = _PINNED_RESULT["value"]
    if not ok:
        raise SharedInputError("SOURCE_DRIFT", f"raptor_dicom sources differ from the pinned math: {actual}")
    return dict(actual)


def series_dir(series_root, uid, series):
    """The anchor's series directory key (A:3952-3953)."""
    return f"{series_root}/{uid}/{series}"


def dicom_names(directory):
    """Same member rule as DirectoryDicomSource.members: regular files ending in .dcm."""
    with os.scandir(directory) as entries:
        return sorted(e.name for e in entries if e.name.endswith(".dcm") and e.is_file())


def contract_picks(rows, spec):
    """Series UID (or None) per slot, by raptor_dicom.pick_series."""
    picks, used = [], set()
    for plane, fluid, _ in spec["slots"]:
        record = pick_series(rows, plane, fluid, used)
        if record is None:
            picks.append(None)
            continue
        used.add(str(record[SERIES]))
        picks.append(str(record[SERIES]))
    return picks


def contract_plane_keys(uid, rows, series_root, orders_by_dir, spec):
    """Keys of the planes our contract selects from an anchor capture (to release the rest early).

    Only a memory optimisation: if this ever disagreed with render_study, rendering would
    stop with CAPTURE_INCOMPLETE and the study would use the DICOM fallback.
    """
    keys = set()
    for series, (_, _, count) in zip(contract_picks(rows, spec), spec["slots"]):
        entry = orders_by_dir.get(series_dir(series_root, uid, series)) if series is not None else None
        if not entry or not entry[0]:
            continue
        files, n = entry[0], len(entry[0])
        low = int(n * spec["span"][0]); high = max(int(n * spec["span"][1]) - 1, low)
        selected = np.linspace(low, high, count).round().astype(int) if n > 1 else np.zeros(count, int)
        keys.update(str(files[min(int(i), n - 1)][0]) for i in set(selected.tolist()))
    return keys


def _volume_sha256(volume):
    return hashlib.sha256(memoryview(np.ascontiguousarray(volume)).cast("B")).hexdigest()


def _mask_sha256(mask):
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def render_study(uid, rows, orders, pixels, spec):
    """Render one fixed-slot stack from already-decoded planes.

    orders: {series_uid: (files, median_spacing)} with files = [(key, pixel_spacing), ...]
    in header order (IOP/IPP position, ties by path) for every acquired contract series.
    pixels: {key: 2-D finite float32 array after modality LUT and MONOCHROME1 inversion}
    for at least every selected plane. Returns (volume, mask, receipt) like
    raptor_dicom.prepare_study; raises SharedInputError when an input is missing.
    """
    import cv2
    check_pinned_sources()
    depth, img = int(spec["depth"]), int(spec["img"])
    volume = np.zeros((depth, img, img), np.uint8)
    receipt = {"uid": str(uid), "adapter": SHARED_INPUT_VERSION, "mirrors": _rd.ADAPTER_VERSION,
               "slots": [], "picks": [], "mask_semantics": "positive normalized pixel sum"}
    offset, used = 0, set()
    for slot_index, (plane, fluid, count) in enumerate(spec["slots"]):
        record = pick_series(rows, plane, fluid, used)
        if record is None:
            receipt["picks"].append(None)
            receipt["slots"].append({"slot": slot_index, "status": "metadata_absent"})
            offset += count
            continue
        series = str(record[SERIES]); used.add(series)
        receipt["picks"].append(series)
        if series not in orders:
            raise SharedInputError("CAPTURE_INCOMPLETE", f"{uid}/{series}: no header order supplied")
        files, median_spacing = orders[series]
        n = len(files)
        if n == 0:
            raise SharedInputError("CAPTURE_INCOMPLETE", f"{uid}/{series}: acquired series without files")
        # --- mirrored from raptor_dicom.prepare_study (pinned by PINNED_SOURCES) ---
        low = int(n * spec["span"][0]); high = max(int(n * spec["span"][1]) - 1, low)
        selected = np.linspace(low, high, count).round().astype(int) if n > 1 else np.zeros(count, int)
        arrays = {}
        for index in sorted(set(map(int, selected))):
            key = files[min(index, n - 1)][0]
            pixel = pixels.get(key)
            if pixel is None:
                raise SharedInputError("CAPTURE_INCOMPLETE", f"{uid}/{series}: plane {index} not decoded")
            if (not isinstance(pixel, np.ndarray) or pixel.dtype != np.float32 or pixel.ndim != 2
                    or not np.isfinite(pixel).all()):
                raise SharedInputError("CAPTURE_INCOMPLETE", f"{uid}/{series}: plane {index} is not finite 2-D float32")
            arrays[index] = pixel
        # Intentionally include each repeated sample in the percentile population.
        loq, hiq = np.percentile(np.concatenate([arrays[int(i)].ravel() for i in selected]), [2.0, 98.0])
        for index in selected:
            pixel = arrays[int(index)]
            spacing = files[min(int(index), n - 1)][1]
            spacing = spacing if spacing > 0 else median_spacing
            normalized = np.clip((pixel - loq) / (hiq - loq + 1e-6), 0, 1)
            height, width = normalized.shape
            crop = min(int(round(float(spec["crop_mm"]) / max(spacing, 1e-3))), min(height, width))
            if crop <= 0:
                raise SharedInputError("RENDER_FAILED", f"empty physical crop for {uid}/{series}")
            y0, x0 = (height - crop) // 2, (width - crop) // 2
            resized = cv2.resize(normalized[y0:y0 + crop, x0:x0 + crop], (img, img), interpolation=cv2.INTER_AREA)
            volume[offset] = (resized * 255).astype(np.uint8)
            offset += 1
        # --- end of mirrored block ---
        receipt["slots"].append({"slot": slot_index, "status": "rendered", "series": series, "files": n,
                                 "slices": selected.tolist(), "normalization": [float(loq), float(hiq)],
                                 "median_spacing": float(median_spacing)})
    if offset != depth:
        raise SharedInputError("RENDER_FAILED", f"slot depth {offset} does not match {depth}")
    mask = (volume.reshape(depth, -1).sum(1) > 0).astype(np.uint8)
    receipt.update(volume_sha256=_volume_sha256(volume), mask_sha256=_mask_sha256(mask),
                   positive_mask_slices=int(mask.sum()), acquired_slots=len(used))
    volume.setflags(write=False); mask.setflags(write=False)
    return volume, mask, receipt


def _norm(path):
    return os.path.normcase(os.path.normpath(str(path)))


def render_anchor_capture(uid, rows, series_root, orders_by_dir, pixels, spec, flagged_files=(), listing=dicom_names):
    """Render from what the anchor's maxspan-v5 reader ordered and decoded for one study.

    orders_by_dir: {series directory: (files, median_spacing)} as returned by the anchor's
    order_and_meta; pixels: {path: array} as returned by its read_px. Before rendering,
    prove that the anchor's order is the one our reader would produce: the same .dcm
    member set (no unreadable-header skip), finite spacings, the same median spacing and no
    anchor order-fallback event for files of these series. Any doubt -> SharedInputError.
    """
    flagged_dirs = {_norm(os.path.dirname(str(f))) for f in flagged_files}
    orders = {}
    for series in contract_picks(rows, spec):
        if series is None:
            continue
        key = series_dir(series_root, uid, series)
        if key not in orders_by_dir:
            raise SharedInputError("CAPTURE_INCOMPLETE", f"{uid}/{series}: series not ordered by the anchor")
        files, median_spacing = orders_by_dir[key]
        files = [(str(path), spacing) for path, spacing in files]
        names = [os.path.basename(path) for path, _ in files]
        if any(_norm(os.path.dirname(path)) != _norm(key) for path, _ in files):
            raise SharedInputError("ORDER_MISMATCH", f"{uid}/{series}: file outside its series directory")
        if len(set(names)) != len(names) or sorted(names) != listing(key):
            raise SharedInputError("ORDER_MISMATCH", f"{uid}/{series}: anchor file list differs from the directory")
        spacings = [float(spacing) for _, spacing in files]
        if not files or not np.isfinite(spacings).all():
            raise SharedInputError("ORDER_MISMATCH", f"{uid}/{series}: empty series or nonfinite spacing")
        if float(np.median(spacings)) != float(median_spacing):
            raise SharedInputError("ORDER_MISMATCH", f"{uid}/{series}: median spacing differs")
        if _norm(key) in flagged_dirs:
            raise SharedInputError("ORDER_FALLBACK", f"{uid}/{series}: anchor order used a fallback")
        orders[series] = (files, float(np.median(spacings)))
    return render_study(uid, rows, orders, pixels, spec)


class SharedInputStore:
    """uint8 memmap store: volumes.npy, masks.npy, manifest.json, records.jsonl.

    One writer (thread-safe within a process). A study is served only from its last
    record, which is appended after its rows are written; records carry status, reason,
    volume/mask sha256 and the slot picks used for rendering.
    """
    MANIFEST, VOLUMES, MASKS, RECORDS = "manifest.json", "volumes.npy", "masks.npy", "records.jsonl"

    def __init__(self, directory, manifest, mode):
        self.directory, self.manifest, self.mode = Path(directory), manifest, mode
        self.ids = list(manifest["ids"])
        self.row = {uid: i for i, uid in enumerate(self.ids)}
        self.depth, self.img = int(manifest["spec"]["depth"]), int(manifest["spec"]["img"])
        self.volumes = np.load(self.directory / self.VOLUMES, mmap_mode=mode)
        self.masks = np.load(self.directory / self.MASKS, mmap_mode=mode)
        if self.volumes.shape != (len(self.ids), self.depth, self.img, self.img) or self.volumes.dtype != np.uint8:
            raise ValueError("store volume shape/dtype differs from its manifest")
        if self.masks.shape != (len(self.ids), self.depth) or self.masks.dtype != np.uint8:
            raise ValueError("store mask shape/dtype differs from its manifest")
        self._lock = threading.Lock()

    @classmethod
    def create(cls, directory, ids, contract=DEFAULT_CONTRACT, extra=None):
        directory = Path(directory)
        ids = [str(u) for u in ids]
        if not ids or len(set(ids)) != len(ids) or any(not u for u in ids):
            raise ValueError("store requires nonempty unique study IDs")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"refuse to reuse a nonempty store directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        spec = _contract_spec(contract)
        depth, img = int(spec["depth"]), int(spec["img"])
        for name, shape in ((cls.VOLUMES, (len(ids), depth, img, img)), (cls.MASKS, (len(ids), depth))):
            array = np.lib.format.open_memmap(directory / name, mode="w+", dtype=np.uint8, shape=shape)
            del array
        manifest = {"schema": STORE_SCHEMA, "adapter": SHARED_INPUT_VERSION, "contract": contract, "spec": spec,
                    "ids": ids, "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {})}
        tmp = directory / (cls.MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=1, allow_nan=False), encoding="utf-8")
        os.replace(tmp, directory / cls.MANIFEST)
        (directory / cls.RECORDS).touch()
        return cls(directory, manifest, "r+")

    @classmethod
    def open(cls, directory, mode="r"):
        directory = Path(directory)
        manifest = json.loads((directory / cls.MANIFEST).read_text(encoding="utf-8"))
        if manifest.get("schema") != STORE_SCHEMA:
            raise ValueError(f"unsupported store schema: {manifest.get('schema')!r}")
        return cls(directory, manifest, mode)

    def _append(self, record):
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with (self.directory / self.RECORDS).open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

    def write(self, uid, volume, mask, info=None):
        uid = str(uid)
        if self.mode == "r":
            raise PermissionError("store opened read-only")
        volume, mask = np.asarray(volume), np.asarray(mask)
        if volume.shape != (self.depth, self.img, self.img) or volume.dtype != np.uint8:
            raise ValueError(f"volume shape/dtype {volume.shape}/{volume.dtype} differs from the store")
        if mask.shape != (self.depth,) or mask.dtype != np.uint8:
            raise ValueError(f"mask shape/dtype {mask.shape}/{mask.dtype} differs from the store")
        record = {"uid": uid, "row": self.row[uid], "status": OK, "reason": None,
                  "volume_sha256": _volume_sha256(volume), "mask_sha256": _mask_sha256(mask),
                  "written_at": time.time(), **(info or {})}
        with self._lock:
            self.volumes[record["row"]] = volume
            self.masks[record["row"]] = mask
            self._append(record)
        return record

    def mark(self, uid, status, reason=None, info=None):
        uid = str(uid)
        if status not in STATUSES or status == OK:
            raise ValueError(f"mark needs a non-OK store status, got {status!r}")
        record = {"uid": uid, "row": self.row[uid], "status": status,
                  "reason": None if reason is None else str(reason)[:1000], "written_at": time.time(), **(info or {})}
        with self._lock:
            self._append(record)
        return record

    def flush(self):
        for array in (self.volumes, self.masks):
            if isinstance(array, np.memmap) and self.mode != "r":
                array.flush()

    def records(self):
        """Last record per study; a torn final line (interrupted writer) is ignored."""
        out = {}
        path = self.directory / self.RECORDS
        if not path.is_file():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("uid") in self.row:
                out[record["uid"]] = record
        return out

    def read(self, uid, record=None, verify=True):
        """In-memory read-only copies of an OK row; digest mismatch raises STORE_DIGEST."""
        uid = str(uid)
        record = record if record is not None else self.records().get(uid)
        if record is None or record.get("status") != OK:
            raise SharedInputError("EMPTY" if record is None else record.get("status"), f"{uid}: no OK record")
        row = self.row[uid]
        volume, mask = np.array(self.volumes[row]), np.array(self.masks[row])
        if verify and (_volume_sha256(volume) != record.get("volume_sha256") or _mask_sha256(mask) != record.get("mask_sha256")):
            raise SharedInputError("STORE_DIGEST", f"{uid}: stored bytes differ from their sha256")
        volume.setflags(write=False); mask.setflags(write=False)
        return volume, mask

    def summary(self):
        records = self.records()
        counts = {}
        for record in records.values():
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        counts["EMPTY"] = len(self.ids) - len(records)
        return {"studies": len(self.ids), "status_counts": counts, "directory": str(self.directory),
                "bytes_allocated": int(self.volumes.nbytes + self.masks.nbytes)}


class StoreCorpus:
    """DicomCorpus interface over a SharedInputStore; every doubt goes to DicomCorpus.

    A row is served from the store only if its last record is OK, its recorded picks
    equal raptor_dicom.pick_series on this corpus's own metadata, and its bytes match the
    recorded sha256. ``audit`` > 0 additionally decodes that many store-served studies
    through the fallback and compares bytes; one mismatch distrusts the whole store.
    A missing or unreadable store means every study uses the fallback.
    """

    def __init__(self, store_dir, ids, rows, source, contract=DEFAULT_CONTRACT, split="test", audit=0, verify=True):
        self.fallback = DicomCorpus.from_records(ids, rows, source, contract, split)
        self.ids, self.contract, self.split = self.fallback.ids, contract, self.fallback.split
        self.spec, self.depth, self.rows = self.fallback.spec, self.fallback.depth, self.fallback.rows
        self.verify, self.audit_remaining, self.trusted = bool(verify), int(audit), True
        self.store, self.store_error, self._records = None, None, {}
        try:
            store = SharedInputStore.open(store_dir, mode="r")
            if store.manifest.get("contract") != contract or store.manifest.get("spec") != self.spec:
                raise ValueError("store contract/spec differs from the requested contract")
            self.store, self._records = store, store.records()
        except Exception as exc:
            self.store_error = f"{type(exc).__name__}: {exc}"
        self.receipts, self.audits, self.served = {}, [], {}
        self._cached_uid, self._cached = None, None
        self.manifest = {"adapter": SHARED_INPUT_VERSION, "contract": contract, "spec": self.spec,
                         "store": str(store_dir), "store_error": self.store_error, "split": self.split,
                         "fallback": self.fallback.manifest, "audit": int(audit), "verify": self.verify}

    def __len__(self):
        return len(self.ids)

    def __contains__(self, uid):
        return str(uid) in self.rows

    def _store_reason(self, uid):
        if self.store is None:
            return "STORE_ABSENT", None
        if not self.trusted:
            return "STORE_UNTRUSTED", None
        record = self._records.get(uid)
        if record is None:
            return "EMPTY", None
        if record.get("status") != OK:
            return str(record.get("status")), None
        if record.get("picks") != contract_picks(self.rows[uid], self.spec):
            return "PICKS_DIFFER", None
        try:
            return None, self.store.read(uid, record, verify=self.verify)
        except SharedInputError as exc:
            return exc.code, None

    def _ensure(self, uid):
        uid = str(uid)
        if uid not in self.rows:
            raise KeyError(uid)
        if self._cached_uid == uid:
            return self._cached
        self._cached_uid, self._cached = None, None
        reason, arrays = self._store_reason(uid)
        if reason is None and self.audit_remaining > 0:
            self.audit_remaining -= 1
            reference = (self.fallback.volume(uid), self.fallback.mask(uid))
            equal = bool(np.array_equal(arrays[0], reference[0]) and np.array_equal(arrays[1], reference[1]))
            self.audits.append({"uid": uid, "equal": equal})
            if not equal:
                self.trusted, reason, arrays = False, "AUDIT_MISMATCH", None
        if reason is None:
            record = self._records[uid]
            self.receipts[uid] = {"uid": uid, "source": "store", "status": "rendered",
                                  "volume_sha256": record["volume_sha256"], "mask_sha256": record["mask_sha256"]}
        else:
            arrays = (self.fallback.volume(uid), self.fallback.mask(uid))
            self.receipts[uid] = dict(self.fallback.receipts[uid], source="dicom_fallback", store_status=reason)
        self.served[uid] = "store" if reason is None else reason
        self._cached_uid, self._cached = uid, arrays
        return arrays

    def volume(self, uid):
        return self._ensure(uid)[0]

    def mask(self, uid):
        return self._ensure(uid)[1]

    def stats(self):
        counts = {}
        for value in self.served.values():
            counts[value] = counts.get(value, 0) + 1
        return {"served": counts, "store_error": self.store_error, "trusted": self.trusted, "audits": list(self.audits)}
