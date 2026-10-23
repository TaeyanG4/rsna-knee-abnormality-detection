"""Release insurance v1.0.0 (H-B-129, 2026-10-13).

Scheduling/recovery only: no model, ranking, sentinel or CSV arithmetic lives here.
Bootstrap Deadline.start() in the FIRST notebook cell, before unpack/install/imports.
The coordinator must keep native work in Supervisor children, then call the existing
authenticated collector AFTER leaving the Supervisor context. Identity errors remain
fatal. Missing complete study/member rows go to the release's existing fallback.

POSIX tree contract: only the outer supervisor creates a session; ALL nested launchers
must use this module or inherit its process group (no setsid/setpgid/daemon escape).
Windows uses suspended creation + a Job Object, including orphaned grandchildren.
Never use PIPE with Supervisor: redirect output to a file to avoid pipe backpressure.
WatchdogPool owns its Pool: do not put a raw Pool context around guarded iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import multiprocessing as mp
from multiprocessing.pool import Pool as _Pool
from pathlib import Path
import os
import queue
import signal
import subprocess
import threading
import time

VERSION = "1.0.0"
DEFAULT_DEADLINE_SECONDS = 8 * 3600 + 40 * 60
DEFAULT_STALL_SECONDS = 20 * 60
START_ENV = "KNEE_RELEASE_START_MONOTONIC_V1"
BUDGET_ENV = "KNEE_RELEASE_DEADLINE_SECONDS_V1"
STALL_ENV = "KNEE_RELEASE_POOL_STALL_SECONDS_V1"
TREE_ENV = "KNEE_RELEASE_TREE_V1"
FAULT_ENV = "KNEE_RELEASE_FAULT_V1"


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class Deadline:
    started: float
    seconds: float = DEFAULT_DEADLINE_SECONDS

    def __post_init__(self):
        if not math.isfinite(self.started) or self.started < 0:
            raise ValueError("invalid monotonic kernel start")
        _positive(self.seconds, "deadline")

    @classmethod
    def start(cls, seconds=None, *, env=None):
        """First cell only. Repeated calls inherit the original start AND budget."""
        env = os.environ if env is None else env
        if START_ENV in env:
            deadline = cls.from_env(env)
            if seconds is not None and float(seconds) != deadline.seconds:
                raise ValueError("cannot reset an inherited deadline")
            return deadline
        deadline = cls(time.monotonic(), _positive(
            env.get(BUDGET_ENV, DEFAULT_DEADLINE_SECONDS) if seconds is None else seconds, "deadline"))
        env.update(deadline.environment())
        return deadline

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        # A missing bootstrap is a wiring error, never a fresh eight-hour allowance.
        return cls(float(env[START_ENV]), _positive(env[BUDGET_ENV], "deadline"))

    def environment(self, base=None):
        return dict({} if base is None else base,
                    **{START_ENV: repr(self.started), BUDGET_ENV: repr(self.seconds)})

    @property
    def remaining(self):
        return max(0., self.started + self.seconds - time.monotonic())

    @property
    def expired(self):
        return self.remaining <= 0


class _WindowsJob:
    """No child executes until assigned; closing the job kills its entire tree."""
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        class Basic(c.Structure):
            _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64),
                        ("flags", w.DWORD), ("min_ws", c.c_size_t), ("max_ws", c.c_size_t),
                        ("active", w.DWORD), ("affinity", c.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IO(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in ("r", "w", "o", "rb", "wb", "ob")]
        class Limits(c.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", c.c_size_t),
                        ("job_memory", c.c_size_t), ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]
        self.c = c
        self.k = c.WinDLL("kernel32", use_last_error=True)
        self.k.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        self.k.CreateJobObjectW.restype = w.HANDLE
        self.k.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        self.k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.k.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.k.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = Limits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.k.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            self.close()
            raise error

    def assign_and_resume(self, proc):
        if not self.k.AssignProcessToJobObject(self.handle, int(proc._handle)):
            raise self.c.WinError(self.c.get_last_error())
        nt = self.c.WinDLL("ntdll")
        nt.NtResumeProcess.argtypes = [self.c.c_void_p]
        nt.NtResumeProcess.restype = self.c.c_long
        if nt.NtResumeProcess(int(proc._handle)) != 0:
            raise OSError("NtResumeProcess failed")

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None


@dataclass
class Child:
    label: str
    process: object = None
    job: object = None
    group_owner: bool = False
    returncode: int | None = None
    reason: str | None = None
    cleanup_error: str | None = None


class Supervisor:
    """One absolute deadline for concurrent/sequential branches, with an active timer.

    Use start()/wait() for shards or run() for sequential phases. Expired starts return
    a skipped Child (124) without launching anything. After cleanup, run the existing
    collector even for nonzero return codes. Never catch authentication errors there.
    """
    def __init__(self, deadline, *, poll_seconds=.1, cleanup_seconds=2.):
        self.deadline = deadline
        self.poll = _positive(poll_seconds, "poll")
        self.cleanup_seconds = _positive(cleanup_seconds, "cleanup")
        self.children = []
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True, name="release-deadline")
        self._thread.start()

    def start(self, args, *, label, env=None, **kwargs):
        if any(kwargs.get(k) == subprocess.PIPE for k in ("stdout", "stderr", "stdin")):
            raise ValueError("use files, not PIPE, for supervised children")
        if any(k in kwargs for k in ("start_new_session", "process_group", "creationflags", "preexec_fn")):
            raise ValueError("Supervisor owns process-tree creation")
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("supervisor closed")
            child = Child(label)
            self.children.append(child)
            if self.deadline.expired:
                child.returncode, child.reason = 124, "deadline_before_start"
                return child
            inherited = os.environ if env is None else env
            child_env = self.deadline.environment(inherited)
            child_env[TREE_ENV] = "1"
            # Nested POSIX launches inherit the OUTER group, including fork pools.
            child.group_owner = os.name != "nt" and os.environ.get(TREE_ENV) != "1"
            options = dict(creationflags=0x4) if os.name == "nt" else dict(start_new_session=child.group_owner)
            try:
                if os.name == "nt":
                    child.job = _WindowsJob()
                child.process = subprocess.Popen(args, env=child_env, **options, **kwargs)
                if child.job:
                    child.job.assign_and_resume(child.process)
            except BaseException:
                self._finish(child, "launch_error")
                raise
            return child

    def _finish(self, child, reason):
        if child.returncode is not None:
            return
        proc = child.process
        code = proc.poll() if proc else None
        # Even a normally exited parent may have orphaned children. Clean the tree.
        try:
            if child.job:
                child.job.close()
            elif proc and child.group_owner:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if proc and proc.poll() is None:
                proc.kill()
            if proc:
                proc.wait(timeout=self.cleanup_seconds)
        except (OSError, subprocess.SubprocessError) as exc:
            child.cleanup_error = f"{type(exc).__name__}: {exc}"
        child.reason = reason
        child.returncode = (code if code is not None else 124 if reason.startswith("deadline") else 1)

    def _monitor(self):
        while not self._closed.wait(self.poll):
            with self._lock:
                for child in self.children:
                    if child.returncode is not None:
                        continue
                    if self.deadline.expired:
                        self._finish(child, "deadline")
                    elif child.process and child.process.poll() is not None:
                        self._finish(child, "completed" if child.process.returncode == 0 else "child_exit")

    def wait(self, child):
        while child.returncode is None:
            self._closed.wait(min(self.poll, max(.001, self.deadline.remaining)))
        return child

    def run(self, args, **kwargs):
        return self.wait(self.start(args, **kwargs))

    def close(self):
        with self._lock:
            for child in self.children:
                self._finish(child, "deadline" if self.deadline.expired else "coordinator_exit")
            self._closed.set()
        self._thread.join(timeout=self.cleanup_seconds)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PoolAborted(RuntimeError):
    """Operational branch failure; preserve only completed inference journal rows."""


class _TrackedPool(_Pool):
    """Retain every worker BEFORE start, including fast initializer crashes."""
    def __init__(self, *args, **kwargs):
        self.workers_ever = []
        super().__init__(*args, **kwargs)

    def Process(self, ctx, *args, **kwargs):
        process = ctx.Process(*args, **kwargs)
        self.workers_ever.append(process)
        return process


class WatchdogPool:
    """Owned Pool with bounded cleanup and imap_unordered progress/death monitoring.

    Works with chunksize > 1 (CPython otherwise returns a generator WITHOUT next's
    timeout argument). A pump keeps exactly one buffered item; waiting for the
    consumer is backpressure, not a decoder stall. A worker retirement/replacement
    is a failure: maxtasksperchild is deliberately unsupported. Defaults to fork
    on POSIX, spawn on Windows. No changes to task chunks or successful results.
    """
    def __init__(self, processes=None, initializer=None, initargs=(), *, deadline=None,
                 stall_seconds=None, poll_seconds=.1, cleanup_seconds=2., context=None):
        self.deadline = Deadline.from_env() if deadline is None else deadline
        self.stall = _positive(os.environ.get(STALL_ENV, DEFAULT_STALL_SECONDS)
                               if stall_seconds is None else stall_seconds, "stall")
        self.poll = _positive(poll_seconds, "poll")
        self.cleanup = _positive(cleanup_seconds, "cleanup")
        if self.deadline.expired:
            raise PoolAborted("deadline_before_pool")
        ctx = mp.get_context(context or ("spawn" if os.name == "nt" else "fork"))
        self.pool = _TrackedPool(processes, initializer, initargs, context=ctx)
        self._workers = self.pool.workers_ever
        self._iterators = []
        self._stopping = threading.Event()
        self._cleanup_thread = None
        self._stop_lock = threading.Lock()

    def _dead_worker(self):
        return next((w for w in list(self._workers) if w.exitcode is not None), None)

    def terminate(self):
        with self._stop_lock:
            if self._stopping.is_set():
                return
            self._stopping.set()
            # Stop replacements before killing workers. Pool.terminate can itself
            # hang on CPython internal locks after a native crash: never await it
            # without a bound. Root Supervisor remains the final process-tree guard.
            self.pool._worker_handler._state = "TERMINATE"
            self.pool._change_notifier.put(None)
            for worker in list(self.pool._pool) + self._workers:
                try:
                    if worker.is_alive():
                        worker.kill()
                except (OSError, ValueError):
                    pass
            self._cleanup_thread = threading.Thread(target=self.pool.terminate, daemon=True,
                                                    name="release-pool-cleanup")
            self._cleanup_thread.start()

    def join(self):
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=self.cleanup)

    def imap_unordered(self, function, iterable, chunksize=1):
        if self._stopping.is_set():
            raise PoolAborted("pool_already_stopped")
        if self._iterators and not self._iterators[-1].done.is_set():
            raise ValueError("only one active imap per WatchdogPool")
        iterator = _WatchedResults(self, self.pool.imap_unordered(function, iterable, chunksize))
        self._iterators.append(iterator)
        return iterator

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for iterator in self._iterators:
            iterator.done.set()
        self.terminate()
        self.join()


class _WatchedResults:
    def __init__(self, owner, iterator):
        self.owner, self.iterator = owner, iterator
        self.items = queue.Queue(maxsize=1)
        self.done = threading.Event()
        self.abort = None
        self.last_result = time.monotonic()
        self.backpressure = threading.Event()
        self.pump_done = threading.Event()
        self._pump_thread = threading.Thread(target=self._pump, daemon=True, name="release-pool-pump")
        self._watch_thread = threading.Thread(target=self._watch, daemon=True, name="release-pool-watchdog")
        self._pump_thread.start()
        self._watch_thread.start()

    def _pump(self):
        try:
            while not self.done.is_set():
                try:
                    value = next(self.iterator)
                except StopIteration:
                    return
                self.last_result = time.monotonic()
                self.backpressure.set()
                while not self.done.is_set():
                    try:
                        self.items.put(value, timeout=self.owner.poll)
                        break
                    except queue.Full:
                        pass
                self.last_result = time.monotonic()
                self.backpressure.clear()
        except Exception as exc:
            self._fail(f"pool_task_error: {type(exc).__name__}: {exc}")
        finally:
            self.pump_done.set()

    def _fail(self, reason):
        if not self.done.is_set():
            self.abort = reason
            self.done.set()
            self.owner.terminate()

    def _watch(self):
        while not self.done.wait(self.owner.poll):
            dead = self.owner._dead_worker()
            if self.owner.deadline.expired:
                self._fail("deadline")
            elif dead:
                self._fail(f"pool_worker_exit: pid={dead.pid}, code={dead.exitcode}")
            elif (not self.pump_done.is_set() and not self.backpressure.is_set() and self.items.empty()
                  and time.monotonic() - self.last_result >= self.owner.stall):
                self._fail("pool_no_result")

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.abort:
                raise PoolAborted(self.abort)
            try:
                value = self.items.get(timeout=self.owner.poll)
                if self.abort:
                    raise PoolAborted(self.abort)
                return value
            except queue.Empty:
                if self.pump_done.is_set():
                    self.done.set()
                    if self.abort:
                        raise PoolAborted(self.abort)
                    raise StopIteration
                if self.done.is_set():
                    raise PoolAborted(self.abort or "pool_stopped")


class InjectedDecodeFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class FaultInjection:
    """N32-only opt-in. Call before decode INSIDE its existing study error handler.

    Run four separate stress cases so kill/hang cannot mask one another. `all` is
    expanded by cases(); each case gets a fresh output, process tree and deadline.
    Production entrypoints MUST parse this with stress=False to reject activation.
    """
    mode: str = "off"
    uid: str | None = None
    early_seconds: float = 5.

    @classmethod
    def cases(cls, ids, *, stress, env=None, early_seconds=5.):
        mode = (os.environ if env is None else env).get(FAULT_ENV, "off")
        if mode == "off":
            return [cls()]
        if not stress or len(ids) != 32 or len(set(ids)) != 32:
            raise ValueError("fault injection requires a diagnostic N32 cohort")
        if mode not in {"decode", "kill", "hang", "deadline", "all"}:
            raise ValueError("unknown release fault mode")
        seconds = _positive(early_seconds, "early deadline")
        return [cls(m, ids[1], seconds) for m in
                (("decode", "kill", "hang", "deadline") if mode == "all" else (mode,))]

    def deadline(self, inherited):
        # Never grant more time than the kernel has left.
        shortened = time.monotonic() - inherited.started + self.early_seconds
        return Deadline(inherited.started, min(inherited.seconds, shortened)) if self.mode == "deadline" else inherited

    def assert_csv(self, ids, actual, expected, *, marker=None, failed_ids=()):
        """Stress oracle: expected bytes MUST come from the existing survivor logic.

        Call after cleanup/publication. Checks all 32 full rows, the actual fault
        marker, target failure, and byte identity to the explicitly failed-member
        replay. No new blend or sentinel policy is introduced here.
        """
        import csv
        import io
        if len(ids) != 32 or len(set(ids)) != 32:
            raise AssertionError("stress assertion requires N32")
        rows = list(csv.reader(io.StringIO(actual.decode("utf-8"))))
        if (len(rows) != 33 or any(len(row) != 13 for row in rows)
                or [row[0] for row in rows[1:]] != list(ids)):
            raise AssertionError("incomplete stress CSV cohort/schema")
        if not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for row in rows[1:] for v in row[1:]):
            raise AssertionError("invalid stress CSV probability")
        if self.mode != "off":
            if marker is None or Path(marker).read_text(encoding="utf-8").strip() != self.mode:
                raise AssertionError("fault was not reached")
            if self.uid not in failed_ids:
                raise AssertionError("injected study must use failed-member fallback")
        if actual != expected:
            raise AssertionError("CSV differs from existing survivor fallback")

    def before_study(self, uid, *, marker=None):
        if self.mode == "off" or uid != self.uid:
            return
        if marker is not None:
            path = Path(marker)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.mode + "\n", encoding="utf-8")
        if self.mode == "decode":
            raise InjectedDecodeFailure("H-B-129 injected study decode failure")
        if self.mode == "kill":
            os._exit(86)  # native worker death: no Python exception/finally
        # Both hang cases require an external watchdog, not a cooperative timeout.
        while True:
            time.sleep(60)


class CompleteRows:
    """Small CPU/stress journal; production can retain its authenticated journals.

    Atomic 12-target study/member commits, preserving dtype and every bit. Failed
    rows have no committed file. Temporary files after a kill are ignored. One
    writer per study/member is required. Identity/committed corruption is fatal.
    No neutral values enter the native blend: use recovered()'s NaNs there; for
    C85 use c85_inputs(), which pairs .5 with explicit per-member failure maps.
    """
    def __init__(self, directory, ids, members, identity, *, create=False):
        self.directory = Path(directory)
        self.ids, self.members = list(ids), list(members)
        if not self.ids or len(set(ids)) != len(ids) or not members or len(set(members)) != len(members):
            raise ValueError("nonempty unique cohort/member roster required")
        self.header = dict(version=VERSION, ids=self.ids, members=self.members, identity=identity)
        if create:
            self.directory.mkdir(parents=True, exist_ok=False)
            (self.directory / "identity.json").write_text(json.dumps(self.header, sort_keys=True), encoding="utf-8")
        if json.loads((self.directory / "identity.json").read_text(encoding="utf-8")) != self.header:
            raise ValueError("journal identity/cohort/member mismatch")

    def _path(self, uid, member):
        return self.directory / f"m{self.members.index(member):03d}-s{self.ids.index(uid):06d}.npy"

    @staticmethod
    def _row(value):
        import numpy as np
        a = np.asarray(value)
        if (a.shape != (12,) or a.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or not np.isfinite(a).all() or not ((a >= 0) & (a <= 1)).all()):
            raise ValueError("only complete finite 12-target probability rows can commit")
        return a

    def commit(self, uid, member, values):
        import numpy as np
        row = self._row(values)
        path = self._path(uid, member)
        if path.exists():
            raise ValueError("duplicate study/member commit")
        temporary = path.with_suffix(".part")
        with temporary.open("wb") as stream:
            np.save(stream, row, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def recovered(self):
        import numpy as np
        raw, failures = {}, {}
        for member in self.members:
            rows = {}
            for uid in self.ids:
                path = self._path(uid, member)
                if path.exists():
                    rows[uid] = self._row(np.load(path, allow_pickle=False))
            dtype = np.result_type(*[x.dtype for x in rows.values()]) if rows else np.dtype('float64')
            raw[member] = np.full((len(self.ids), 12), np.nan, dtype=dtype)
            failures[member] = {}
            for i, uid in enumerate(self.ids):
                if uid in rows:
                    raw[member][i] = rows[uid]
                else:
                    failures[member][uid] = "MissingPrediction: child did not finish study"
        return raw, failures

    def c85_inputs(self):
        raw, failures = self.recovered()
        for member, rows in raw.items():
            for i, uid in enumerate(self.ids):
                if uid in failures[member]:
                    rows[i] = .5
        return raw, failures
