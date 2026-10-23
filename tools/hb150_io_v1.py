"""H-B-150 Windows output IO v1.0.0 (2026-10-14).

Changelog: bounded sharing-error retries; nonfatal exports/heartbeats and
fatal persistent journal errors with best-effort independent failure receipts.
"""
import json
import os
from pathlib import Path
import time

VERSION = "1.0.0"
ATTEMPTS = 40
RETRY_SECONDS = .05


class JournalWriteError(OSError):
    pass


def _retry(operation):
    for index in range(ATTEMPTS):
        try:
            return operation()
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)):
                raise
            if index == ATTEMPTS - 1:
                raise
            time.sleep(RETRY_SECONDS)


def _failure(path, exc, critical):
    if not critical:
        return False
    receipt = Path(str(path) + ".io-failure.json")
    value = dict(schema="hb150-io-failure/v1", status="FAILED", path=str(path),
                 error=type(exc).__name__, reason="persistent_journal_failure")
    try:
        # Independent publication path also uses bounded retries. Never recurse.
        _retry(lambda: receipt.write_text(json.dumps(value), encoding="utf-8"))
    except OSError:
        # stdout remains an independent receipt channel if the filesystem fails.
        print(json.dumps(value), flush=True)
    raise JournalWriteError(f"persistent journal write failed: {path}") from exc


def safe_replace(source, destination, *, critical=True):
    try:
        _retry(lambda: os.replace(source, destination))
        return True
    except OSError as exc:
        return _failure(destination, exc, critical)


def safe_unlink(path, *, critical=False):
    try:
        _retry(lambda: Path(path).unlink(missing_ok=True))
        return True
    except OSError as exc:
        return _failure(path, exc, critical)


def atomic_json(path, value, *, critical=True):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, sort_keys=True, allow_nan=False)
        _retry(lambda: temporary.write_text(payload, encoding="utf-8"))
        return safe_replace(temporary, path, critical=critical)
    except JournalWriteError:
        raise
    except OSError as exc:
        return _failure(path, exc, critical)
    finally:
        safe_unlink(temporary, critical=False)
