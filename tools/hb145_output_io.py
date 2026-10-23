"""H-B-145 v1.1.3 output I/O; one bounded Windows sharing retry policy.

Changelog v1.1.3: journals fail closed; derived exports/receipts may be deferred.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

REPLACE_ATTEMPTS = 40
REPLACE_DELAYS = tuple(min(.05 * (i + 1), 1.) for i in range(REPLACE_ATTEMPTS - 1))


def retry(operation, *args, **kwargs):
    """40 total attempts, 29.5 seconds maximum backoff; other errors propagate."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            return operation(*args, **kwargs)
        except OSError as error:
            if not (isinstance(error, PermissionError) or getattr(error, "winerror", None) in (5, 32)):
                raise
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_DELAYS[attempt])


def replace(source, target):
    return retry(os.replace, source, target)


def rename(source, target):
    return retry(os.rename, source, target)


def unlink(path):
    return retry(os.unlink, path)


def rmtree(path):
    """Only caller-validated owned scratch; every unlink/rmdir uses retry."""
    path = Path(path)
    for child in path.iterdir():
        if child.is_symlink() or not child.is_dir():
            unlink(child)
        else:
            rmtree(child)
    retry(os.rmdir, path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def write():
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    retry(write)
    replace(temporary, path)


def read_json(path):
    return retry(lambda: json.loads(Path(path).read_text(encoding="utf-8")))


def read_receipt(path):
    """Optional diagnostics must not invalidate authoritative study journals."""
    for candidate in (Path(path), Path(path).with_suffix(".deferred.json")):
        try:
            return read_json(candidate)
        except (OSError, ValueError):
            continue
    return {}


class DerivedStream:
    """Best-effort native diagnostic CSV; role CSVs are rebuilt from journals."""
    def __init__(self, path, **kwargs):
        self.path, self.stream = Path(path), None
        try:
            self.stream = retry(lambda: self.path.open("w", **kwargs))
        except OSError as error:
            self._deferred(error)

    def _deferred(self, error):
        print(f"EXPORT_DEFERRED {self.path.name}: {type(error).__name__}; journals authoritative",
              file=sys.stderr, flush=True)

    def _call(self, method, *args):
        if self.stream is not None:
            try:
                return getattr(self.stream, method)(*args)
            except OSError as error:
                self._deferred(error)
                try:
                    self.stream.close()
                except OSError:
                    pass
                self.stream = None

    def write(self, value):
        self._call("write", value)
        return len(value)

    def flush(self):
        self._call("flush")

    def close(self):
        self._call("close")


def receipt(path, value):
    """A locked receipt cannot invalidate committed predictions or stop inference."""
    path = Path(path)
    try:
        atomic_json(path, value)
        return True
    except OSError as error:
        print(f"RECEIPT_DEFERRED {path.name}: {type(error).__name__}", file=sys.stderr, flush=True)
        fallback = path.with_suffix(".deferred.json")
        try:
            atomic_json(fallback, {**value, "publication_deferred": path.name,
                                   "publication_error": type(error).__name__})
        except OSError:
            print("RECEIPT_UNAVAILABLE: journals remain authoritative", file=sys.stderr, flush=True)
        return False


class JournalCommitError(BaseException):
    """Must escape serving code's ordinary per-study Exception fallback."""


def journal(path, value, out):
    try:
        atomic_json(path, value)
    except OSError as error:
        receipt(Path(out) / "journal_failure_receipt.json", {
            "version": "1.1.3", "state": "JOURNAL_ERROR", "identity": value["identity"],
            "journal": Path(path).name, "error": type(error).__name__,
            "winerror": getattr(error, "winerror", None), "committed": False})
        raise JournalCommitError("persistent journal write failure; resume from committed journals") from error
