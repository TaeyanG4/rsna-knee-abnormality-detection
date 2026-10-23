"""CPU-only, resumable HC15 cache producer storage; no decoder or dispatch.

The future reviewed DICOM producer supplies one complete NPY payload at a time.
This module commits validated series, preserves interrupted writes, binds resume
to the exact source contract, and emits the runner's hc15-cache-v1 metadata only
after every frozen series and study is accounted for. It does not authorize a
whole-cache build, make an execution admission, load torch, or alter the runner.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid

GEOMETRY = {"size": 384, "mm_per_pixel": .4, "max_slices": 64,
            "dtype": "uint8", "slots": 6}


class CacheError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CacheError(message)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def check_hash(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256")


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path != root, "Path escapes cache root")
    return path


def frozen_contract(root, manifest_path, expected_sha, adapter_path, decoder_path, *, index_sha256=None):
    """Bind existing V2 metadata without images, model imports, or cache writes."""
    root = Path(root).resolve()
    check_hash(expected_sha)
    require(digest_file(manifest_path) == expected_sha, "Preparation manifest drift")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    require(manifest.get("source_plan") == "H-C-15", "Wrong plan")
    require(manifest.get("schema") == "hc15-open-recipe-preparation-v1", "Wrong manifest schema")
    files = {}
    for name in ("cohort.json", "series.csv"):
        matches = [x for x in manifest["generated"] if Path(x["path"]).name == name]
        require(len(matches) == 1, "Ambiguous generated input")
        binding = matches[0]
        path = checked_path(root, binding["path"])
        require(path.stat().st_size == binding["bytes"] and digest_file(path) == binding["sha256"], "Frozen input drift")
        files[name] = path
    cohort = json.loads(files["cohort.json"].read_text(encoding="utf-8"))
    train, val = cohort["train_uids"], cohort["val_uids"]
    require(len(train) == 3455 and len(val) == 888 and len(set(train + val)) == 4343, "Cohort membership/count changed")
    for ids, key in ((train, "train_uid_sha256"), (val, "validation_uid_sha256")):
        require(hashlib.sha256("\n".join(ids).encode()).hexdigest() == manifest["cohort"][key], "Cohort order changed")
    with files["series.csv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == manifest["cache"]["series"], "Series count changed")
    require({r["StudyInstanceUID"] for r in rows} == set(train + val), "Series/study coverage differs")
    return {
        "schema": "hc15-cache-producer-contract-v1",
        "source_manifest_sha256": expected_sha,
        "adapter_sha256": digest_file(adapter_path),
        "decoder_sha256": digest_file(decoder_path),
        "producer_sha256": digest_file(__file__),
        "source_archive": manifest["archive"],
        # None denotes metadata inspection only; CacheStore refuses execution
        # until a future allocation phase actually binds the index bytes.
        "archive_index_sha256": index_sha256,
        "generated_sha256": {k: digest_file(v) for k, v in files.items()},
        "geometry": GEOMETRY,
        "study_uids": train + val,
        "series": rows,  # Keep source CSV order; evaluation ties depend on it.
    }


def inspect_npy(stream, file_size, expected_n):
    """Bounded header parsing; reject pickle/object/float/Fortran payloads."""
    require(stream.read(6) == b"\x93NUMPY", "Invalid NPY magic")
    version = stream.read(2)
    require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "Unsupported NPY version")
    width = 2 if version[0] == 1 else 4
    raw_length = stream.read(width)
    require(len(raw_length) == width, "Truncated NPY length")
    length = int.from_bytes(raw_length, "little")
    require(0 < length <= 65536, "Unbounded NPY header")
    raw = stream.read(length)
    require(len(raw) == length, "Truncated NPY header")
    header = ast.literal_eval(raw.decode("utf-8" if version[0] == 3 else "latin1"))
    require(type(header) is dict and type(header.get("shape")) is tuple
            and len(header["shape"]) == 3
            and all(type(x) is int for x in header["shape"])
            and type(header.get("fortran_order")) is bool
            and type(header.get("descr")) is str, "Invalid NPY header field types")
    require(header == {"descr": "|u1", "fortran_order": False, "shape": (expected_n, 384, 384)}, "NPY geometry/dtype differs")
    require(stream.tell() + expected_n * 384 * 384 == file_size, "NPY payload length differs")


class CacheStore:
    """One lightweight writer; explicit session chunks under one total budget.

    Budget is a *future allocation contract*, not authorization to run. Wall time
    includes resume integrity checks. An uncleanly terminated session retains its
    lock and needs owner intervention; it cannot silently reset elapsed time or
    retry a bad series. Partial outputs are retained in abandoned/ on retry.
    """
    def __init__(self, output, contract, *, max_bytes, max_total_seconds, reserve_bytes):
        self.output = Path(output).resolve()
        self.contract = json.loads(canonical(contract))
        contract = self.contract
        require(contract.get("geometry") == GEOMETRY, "Geometry contract differs")
        for key in ("source_manifest_sha256", "adapter_sha256", "producer_sha256", "decoder_sha256", "archive_index_sha256"):
            check_hash(contract.get(key))
        require(isinstance(contract.get("source_archive"), dict)
                and all(k in contract["source_archive"] for k in ("path", "index", "bytes")), "Missing source archive identity")
        rows = contract["series"]
        ids = [r["SeriesInstanceUID"] for r in rows]
        require(len(ids) == len(set(ids)) > 0, "Duplicate/empty series list")
        require(all(isinstance(s, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", s) for s in ids), "Unsafe series UID")
        studies = contract["study_uids"]
        require(len(studies) == len(set(studies)) > 0, "Duplicate/empty study list")
        require({r["StudyInstanceUID"] for r in rows} == set(studies), "Study coverage differs")
        require(type(max_bytes) is int and max_bytes > 0 and type(reserve_bytes) is int and reserve_bytes >= 0, "Invalid byte budget")
        require(type(max_total_seconds) in (int, float) and math.isfinite(max_total_seconds) and max_total_seconds > 0, "Invalid wall budget")
        self.rows = {r["SeriesInstanceUID"]: r for r in rows}
        self.budget = {"max_bytes": max_bytes, "max_total_seconds": max_total_seconds, "reserve_bytes": reserve_bytes}
        self.identity = {"contract": self.contract, "budget": self.budget}
        self.records = {}
        self.active = False
        self.ledger_open = False

    def _json(self, path, value):
        payload = canonical(value)
        previous_bytes = path.stat().st_size if path.exists() else 0
        temp = path.with_name(path.name + ".part-" + uuid.uuid4().hex)
        with temp.open("xb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
        if hasattr(self, "used_bytes"):
            self.used_bytes += len(payload) - previous_bytes

    def _used(self):
        # Include retained interrupted output, metadata and temporary files.
        return sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())

    def _gate(self, added=0):
        require(self.active, "Writer is not active")
        require(self.spent + time.monotonic() - self.started < self.budget["max_total_seconds"], "Total time budget exhausted")
        require(self.used_bytes + added + 65536 <= self.budget["max_bytes"], "Byte budget exhausted")
        require(shutil.disk_usage(self.output).free >= added + self.budget["reserve_bytes"], "Disk reserve would be consumed")

    def __enter__(self):
        # Refuse unrelated existing directories. A clean prior chunk can resume.
        if self.output.exists():
            require((self.output / "identity.json").is_file(), "Existing directory is not this cache")
            require(json.loads((self.output / "identity.json").read_text()) == self.identity, "Source/budget identity drift")
        else:
            initial_bytes = len(canonical(self.identity)) + 65536 + 512
            require(initial_bytes <= self.budget["max_bytes"], "Byte budget cannot hold identity/ledger")
            ancestor = self.output.parent
            while not ancestor.exists():
                ancestor = ancestor.parent
            require(shutil.disk_usage(ancestor).free >= initial_bytes + self.budget["reserve_bytes"],
                    "Disk reserve cannot hold identity/ledger")
            self.output.mkdir(parents=True, exist_ok=False)
            self._json(self.output / "identity.json", self.identity)
        try:
            with (self.output / "WRITER.lock").open("x") as stream:
                stream.write(str(os.getpid()))
        except FileExistsError as exc:
            raise CacheError("Writer lock exists; active/stale ownership must be resolved explicitly") from exc
        self.started = time.monotonic()
        self.active = True
        try:
            state_path = self.output / "state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {"spent_seconds": 0, "active": False}
            require(state.get("active") is False, "Unclean previous session needs explicit budget reconciliation")
            self.spent = state["spent_seconds"]
            require(type(self.spent) in (int, float) and math.isfinite(self.spent) and self.spent >= 0, "Invalid elapsed ledger")
            self.used_bytes = self._used()
            self._gate()
            self._json(state_path, {"spent_seconds": self.spent, "active": True})
            self.ledger_open = True
            records = self.output / "records"
            if records.exists():
                for path in records.glob("*.json"):
                    self._gate()
                    record = json.loads(path.read_text())
                    sid = path.stem
                    self._validate_record(sid, record)
                    self.records[sid] = record
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_):
        if self.active:
            # On ordinary exceptions preserve completed records and bill time.
            try:
                if self.ledger_open:
                    spent = self.spent + time.monotonic() - self.started
                    self._json(self.output / "state.json", {"spent_seconds": spent, "active": False})
            finally:
                self.active = False
                self.ledger_open = False
                (self.output / "WRITER.lock").unlink()

    def _validate_record(self, sid, record):
        require(sid in self.rows, "Foreign series in cache")
        require(record.get("study") == self.rows[sid]["StudyInstanceUID"] and record.get("series") == sid, "Series/study identity differs")
        n, decoded = record["n"], record["decoded_count"]
        require(type(n) is int and type(decoded) is int and decoded >= 0 and n == min(decoded, 64), "Retained count differs")
        require(record.get("source_members_sha256") and record.get("decoder_sha256"), "Decode provenance missing")
        check_hash(record["source_members_sha256"]); check_hash(record["decoder_sha256"])
        require(record["decoder_sha256"] == self.contract["decoder_sha256"], "Decoder identity differs")
        if n < 3:
            require(record.get("ineligible") is True and record.get("reason") == "fewer_than_three_decoded_frames" and "path" not in record, "Invalid ineligible-series record")
            return
        require(record.get("ineligible") is False, "Eligible series marked ineligible")
        expected = f"arrays/{sid}/volume.npy"
        require(record["path"] == expected, "Unexpected array path")
        path = checked_path(self.output, expected)
        require(path.stat().st_size == record["bytes"] and digest_file(path) == record["sha256"], "Array changed")
        with path.open("rb") as stream:
            inspect_npy(stream, record["bytes"], n)

    def contains(self, sid):
        return sid in self.records

    def commit(self, sid, payload, *, decoded_count, source_members_sha256, decoder_sha256):
        """Commit decoded data once; decode errors must raise, never become n=0."""
        self._gate()
        require(not (self.output / "cache.json").exists(), "Final cache is immutable")
        require(sid in self.rows and sid not in self.records, "Foreign/already committed series")
        require(type(decoded_count) is int and decoded_count >= 0, "Invalid decoded count")
        check_hash(source_members_sha256); check_hash(decoder_sha256)
        require(decoder_sha256 == self.contract["decoder_sha256"], "Decoder identity differs")
        n = min(decoded_count, 64)
        record = {"study": self.rows[sid]["StudyInstanceUID"], "series": sid, "n": n,
                  "decoded_count": decoded_count, "source_members_sha256": source_members_sha256,
                  "decoder_sha256": decoder_sha256, "ineligible": n < 3}
        if n < 3:
            require(payload is None, "Ineligible series must not carry pixels")
            record["reason"] = "fewer_than_three_decoded_frames"
            self._gate(len(canonical(record)) + 256)
        else:
            require(isinstance(payload, bytes), "Expected complete NPY bytes")
            inspect_npy(io.BytesIO(payload), len(payload), n)
            self._gate(len(payload) + 4096)
            folder = checked_path(self.output, f"arrays/{sid}")
            if folder.exists():
                # An interrupted pre-receipt write is never silently overwritten.
                abandoned = checked_path(self.output, "abandoned/" + sid + "-" + uuid.uuid4().hex)
                abandoned.parent.mkdir(exist_ok=True)
                folder.rename(abandoned)
            folder.mkdir(parents=True, exist_ok=False)
            dest = folder / "volume.npy"
            with dest.open("xb") as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            self.used_bytes += len(payload)
            record.update(path=dest.relative_to(self.output).as_posix(), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self._gate(len(canonical(record)) + 256)
        (self.output / "records").mkdir(exist_ok=True)
        self._json(self.output / "records" / (sid + ".json"), record)
        self.records[sid] = record
        return record

    def finalize(self, *, deadline=None):
        def time_gate():
            if deadline is not None:
                require(type(deadline) in (int, float) and math.isfinite(deadline)
                        and time.monotonic() < deadline, "Finalize deadline exhausted/invalid")
        time_gate()
        self._gate()
        require(set(self.records) == set(self.rows), "Not all frozen series completed")
        valid = {r["study"] for r in self.records.values() if r["n"] >= 3}
        require(valid == set(self.contract["study_uids"]), "A study has no eligible decoded series")
        for sid, record in self.records.items():
            time_gate()
            self._gate(); self._validate_record(sid, record)
        time_gate()
        result = {"schema": "hc15-cache-v1", "complete": True,
                  "source_manifest_sha256": self.contract["source_manifest_sha256"],
                  "adapter_sha256": self.contract["adapter_sha256"],
                  "producer_sha256": self.contract["producer_sha256"],
                  "contract": GEOMETRY, "series": self.records,
                  "gpu_dispatch_allowed": False}
        path = self.output / "cache.json"
        if path.exists():
            require(json.loads(path.read_text()) == result, "Final cache differs")
        else:
            self._gate(len(canonical(result)) + 256)
            self._json(path, result)
        return path
