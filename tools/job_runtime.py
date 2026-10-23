"""Identity/receipt primitives, v1.0.3 (2026-10-12).

Changelog: atomic claims/publication and bounded log tails; no job deadlines.
v1.0.1: distinguish a released sharing lock from denied replacement rights.
v1.0.2: dead legacy jobs accept any terminal marker; FAILED wins if both exist.
v1.0.3: Windows JSON readers share delete access during atomic publication.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import errno
import json
import os
from pathlib import Path
import re
import time
import uuid
import psutil

VERSION = "1.0.3"
CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("KNEE_WORKSPACE_ROOT", CODE_ROOT)).resolve()
LOGS = ROOT / "artifacts" / "logs"


def tag_name(tag):
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("tag must be a single safe filename")
    return tag


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def command_digest(command, cwd):
    return digest({"command": command, "cwd": str(Path(cwd).resolve())})


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    replace_file(temp, path)


def replace_file(source, destination):
    """Wait only for a Windows sharing lock, never mask a permissions failure."""
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if os.name != 'nt' or (not sharing_locked(destination) and
                not (getattr(exc, 'winerror', None) in (5,32,33) and replacement_permitted(source,destination))):
                raise
            time.sleep(.01)


def replacement_permitted(source, destination):
    """A short reader may have closed before the sharing-lock probe ran.

    Retry only if file deletion and destination-directory creation are allowed;
    readonly files and ACL denials must fail rather than wait forever.
    """
    import ctypes as c
    from ctypes import wintypes as w
    import stat
    kernel = c.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [w.LPCWSTR,w.DWORD,w.DWORD,w.LPVOID,w.DWORD,w.DWORD,w.HANDLE]
    create.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.CloseHandle.restype = w.BOOL
    def probe(path,access,flags=0):
        handle=create(str(Path(path).resolve()),access,7,None,3,flags,None)
        if handle == c.c_void_p(-1).value: return c.get_last_error()
        kernel.CloseHandle(handle)
        return 0
    for file in (Path(source),Path(destination)):
        try:
            if file.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY: return False
        except FileNotFoundError:
            if file == Path(source): return False
    return (probe(source,0x10000) in (0,32,33) and probe(destination,0x10000) in (0,2,32,33) and
            probe(Path(destination).parent,2,0x02000000) == 0)


def sharing_locked(path):
    # Rename can report ERROR_ACCESS_DENIED for a reader without SHARE_DELETE.
    # Probe DELETE access to distinguish that case from an actual ACL denial.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(Path(path).resolve()), 0x10000, 7, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        return ctypes.get_last_error() in (32, 33)
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    close(handle)
    return False


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _shared_read_fd(path, flags):
    """Open an existing Windows file sharing delete access with publishers.

    The CRT reader omits FILE_SHARE_DELETE. Use a non-inheritable Win32 handle
    sharing read/write/delete, then transfer ownership to Python's file object.
    Open errors propagate unchanged; no retry or liveness deadline is inferred.
    """
    import ctypes as c
    from ctypes import wintypes as w
    import msvcrt
    kernel = c.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
    create.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.CloseHandle.restype = w.BOOL
    handle = create(os.fsdecode(path), 0x80000000, 7, None, 3, 0, None)
    if handle == c.c_void_p(-1).value:
        error = c.WinError(c.get_last_error())
        error.filename = os.fsdecode(path)
        raise error
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
    except BaseException:
        kernel.CloseHandle(handle)
        raise


def read_json(path):
    opener = _shared_read_fd if os.name == 'nt' else None
    with open(path, encoding="utf-8-sig", opener=opener) as stream:
        return json.load(stream)


def identity(pid=None):
    process = psutil.Process(os.getpid() if pid is None else pid)
    return {"pid": process.pid, "create_time": process.create_time()}


def liveness(record):
    """alive/dead/unknown; a reused PID is dead for the recorded identity."""
    if not isinstance(record, dict) or type(record.get("pid")) is not int or record["pid"] <= 0:
        return "unknown"
    try:
        process = psutil.Process(record["pid"])
        if process.status() == psutil.STATUS_ZOMBIE:
            return "dead"
        if "create_time" not in record:
            return "unknown"
        return "alive" if abs(process.create_time() - float(record["create_time"])) < 0.01 else "dead"
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return "dead"
    except (psutil.AccessDenied, ValueError, TypeError):
        return "unknown"


@contextmanager
def file_lock(path, *, blocking=True):
    """Kernel-released mutex; never delete or replace its persistent file."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if not blocking:
                    raise BlockingIOError("orchestration claim is busy")
                time.sleep(0.1)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def bounded_tail(path, lines=5, limit=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2); size = stream.tell(); stream.seek(max(0, size - limit))
            text = stream.read(limit).decode("utf-8", errors="replace")
        return text.splitlines()[-lines:]
    except FileNotFoundError:
        return []


def job_status(logs, tag, *, tail=False, expected_command=None):
    directory = Path(logs) / tag_name(tag)
    result = {"tag": tag, "state": "absent", "pid": None, "started": None, "command": None, "tail": []}
    if not (directory / "job.json").exists():
        if any((directory / name).exists() for name in ("SUCCESS", "FAILED")):
            result.update(state="unknown", reason="terminal marker without job metadata")
        return result
    try:
        job = read_json(directory / "job.json")
        if not isinstance(job, dict) or not isinstance(job.get("command"), list):
            raise ValueError("invalid job metadata")
        result.update({key: job.get(key) for key in ("pid", "started", "command")})
        if tail:
            result["tail"] = bounded_tail(directory / "run.log")
        if expected_command is not None and job["command"] != expected_command:
            raise ValueError("existing tag command differs")
        modern = job.get("schema") == "gpu-job/v2"
        if modern and job.get("command_sha256") != command_digest(job["command"], job["cwd"]):
            raise ValueError("job command identity differs")
        wrapper = liveness({"pid": job.get("pid"), **({"create_time": job["create_time"]} if "create_time" in job else {})})
        child = liveness(job["child_identity"]) if job.get("child_identity") else "dead"
        markers = [name for name in ("SUCCESS", "FAILED") if (directory / name).exists()]
        if not modern and wrapper == "dead" and markers:
            result.update(state="FAILED" if "FAILED" in markers else "SUCCESS",
                          receipt_kind="legacy-marker")
            return result
        if len(markers) > 1:
            raise ValueError("conflicting terminal markers")
        if markers:
            name = markers[0]
            marker = (directory / name).read_text(encoding="utf-8").strip()
            match = re.fullmatch(r"exit (-?\d+) at \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", marker)
            if not match or (int(match[1]) == 0) != (name == "SUCCESS"):
                raise ValueError("invalid terminal marker")
            if modern:
                receipt = read_json(directory / "receipt.json")
                expected_hash = command_digest(job["command"], job["cwd"])
                if (receipt.get("schema") != "gpu-job-receipt/v1" or
                    receipt.get("run_id") != job.get("run_id") or receipt.get("tag") != tag or
                    receipt.get("command_sha256") != expected_hash or job.get("command_sha256") != expected_hash or
                    receipt.get("exit_code") != int(match[1]) or receipt.get("state") != name or
                    receipt.get("wrapper_identity") != {"pid": job["pid"], "create_time": job["create_time"]} or
                    receipt.get("child_identity") != job.get("child_identity")):
                    raise ValueError("terminal receipt identity/command differs")
                if child != "dead":
                    raise ValueError("terminal receipt with live or unobservable child")
            result.update(state=name, receipt_kind="bound" if modern else "legacy-marker")
            return result
        if child == "alive" and wrapper == "dead":
            result.update(state="orphaned", reason="child alive after wrapper disappeared")
        elif wrapper == "alive":
            result["state"] = "running"
        elif not modern and wrapper == "unknown" and type(job.get("pid")) is int and psutil.pid_exists(job["pid"]):
            result.update(state="running", receipt_kind="legacy-pid", reason="legacy PID has no start identity; wait only")
        elif wrapper == "unknown" or child == "unknown" or (modern and job.get("phase") in {"dispatching", "spawning"}):
            result.update(state="unknown", reason="process identity or child handoff unconfirmed")
        else:
            result.update(state="stale", reason="process exited without terminal receipt")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.update(state="unknown", reason=str(exc))
    return result


def active_jobs(logs, exclude=None):
    results = []
    for path in sorted(Path(logs).glob("*/job.json")):
        if path.parent.name == exclude:
            continue
        item = job_status(logs, path.parent.name)
        if item["state"] in {"running", "orphaned", "unknown"}:
            results.append(item)
    return results
