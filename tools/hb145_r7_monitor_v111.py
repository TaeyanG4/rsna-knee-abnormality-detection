"""v1.1.3 unattended, resume-safe owned-process monitor.

Changelog v1.1.3: shared output retries; status publication loss is nonfatal
while journals progress. Genuine no-commit stalls and RAM enforcement remain.

Changelog v1.1.2: bounded Windows replace retries in an isolated status worker;
heartbeat failures are recorded and prolonged publication loss triggers STALL.

Telemetry runs in a single daemon worker; the RAM guard never waits for it.
Only atomic journal file metadata is inspected, never prediction/label rows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from tools import hb145_output_io as output_io

RAM_LIMIT_BYTES = 38 * 10**9
TELEMETRY_TIMEOUT_SECONDS = 2.0
CLEANUP_TIMEOUT_SECONDS = 2.0
REPLACE_ATTEMPTS = output_io.REPLACE_ATTEMPTS
REPLACE_DELAYS = output_io.REPLACE_DELAYS
STATUS_CLOSE_SECONDS = 4 * sum(REPLACE_DELAYS) + 2
EXPOSURE = {
    "exposure": "EXPOSED", "output_kind": "EXPOSED", "clean_oof": False,
    "pooled_target_priors": True, "pooled_state_priors": True,
    "fastmri_plus_auxiliary": True, "anchor_ancestry": "unresolved",
    "side_ancestry": "unresolved",
    "exposure_reasons": ["pooled target + state priors", "fastMRI+ auxiliary",
                         "anchor/side ancestry unresolved"],
}


def _utc(timestamp=None):
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp,
                                  timezone.utc).isoformat()


def _atomic_json(path, value):
    return output_io.atomic_json(path, value)


class _StatusWriter:
    """Single writer with a coalesced pending snapshot; never blocks the guard."""
    def __init__(self, path):
        self.path = path
        self.condition = threading.Condition()
        self.pending = None
        self.stopping = False
        self.started = time.perf_counter()
        self.health = {"error_count": 0, "consecutive_failures": 0,
                       "last_error": None, "last_error_utc": None,
                       "last_success_utc": None, "last_success_monotonic": None,
                       "successful_writes": 0}
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="hb145-v112-status")

    def start(self):
        self.thread.start()

    def submit(self, value):
        with self.condition:
            self.pending = value
            self.condition.notify()

    def snapshot(self):
        with self.condition:
            result = dict(self.health)
        last = result["last_success_monotonic"]
        result["publication_idle_seconds"] = time.perf_counter() - (
            self.started if last is None else last)
        return result

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.stopping)
                if self.pending is None:
                    return
                value, self.pending = self.pending, None
            try:
                _atomic_json(self.path, value)
            except Exception as error:
                with self.condition:
                    self.health["error_count"] += 1
                    self.health["consecutive_failures"] += 1
                    self.health["last_error"] = f"{type(error).__name__}: {error}"
                    self.health["last_error_utc"] = _utc()
            else:
                with self.condition:
                    self.health["consecutive_failures"] = 0
                    self.health["last_success_utc"] = _utc()
                    self.health["last_success_monotonic"] = time.perf_counter()
                    self.health["successful_writes"] += 1

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.thread.join(timeout=STATUS_CLOSE_SECONDS)
        return not self.thread.is_alive()


def _journal_snapshot(directory):
    if directory is None:
        return {}
    result = {}
    for path in Path(directory).glob("*.json"):
        try:
            stat = path.stat()
            result[path.name] = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:  # concurrent atomic replacement
            continue
    return result


def _gpu_sample(env):
    if env.get("CUDA_VISIBLE_DEVICES") == "-1":
        return [0, 0]
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=TELEMETRY_TIMEOUT_SECONDS, check=True)
    values = [int(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
    if len(values) != 2:
        raise ValueError("unexpected nvidia-smi columns")
    return values


class _Telemetry:
    """One daemon worker, bounded subprocess and a one-item result queue."""
    def __init__(self, env, interval):
        self.results = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.env = env
        self.interval = max(interval, 0.05)
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="hb145-v111-telemetry")

    def _run(self):
        while not self.stop.is_set():
            try:
                result = (_gpu_sample(self.env), None)
            except Exception as error:
                result = (None, f"{type(error).__name__}: {error}")
            try:
                self.results.put_nowait(result)
            except queue.Full:
                pass
            self.stop.wait(self.interval)

    def start(self):
        self.thread.start()

    def take(self):
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.stop.set()
        # Subprocess timeout bounds real calls; even an unexpected stuck provider
        # cannot block either the guard or owned-child cleanup.
        self.thread.join(timeout=TELEMETRY_TIMEOUT_SECONDS + 0.5)
        return not self.thread.is_alive()


class _WindowsJob:
    """Kill-on-close job owns even descendants missed by process polling."""
    def __init__(self, child):
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", w.DWORD), ("minimum", ctypes.c_size_t),
                        ("maximum", ctypes.c_size_t), ("active", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD),
                        ("scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read", "write", "other")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.kernel = kernel
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel.CloseHandle(handle)
            raise error
        if not kernel.AssignProcessToJobObject(handle, w.HANDLE(int(child._handle))):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel.CloseHandle(handle)
            raise error
        self.handle = handle

    def close(self):
        if self.handle:
            self.kernel.TerminateJobObject(self.handle, 1)
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _cleanup(child, tracked, job, psutil):
    errors = []
    if child is not None:
        try:
            root = tracked.get(child.pid)
            if root is None:
                root = psutil.Process(child.pid)
            for process in root.children(recursive=True):
                tracked.setdefault(process.pid, process)
            tracked.setdefault(root.pid, root)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    owned = list(reversed(list(tracked.values())))
    for process in owned:
        try:
            process.terminate()  # psutil validates PID identity before signalling
        except psutil.NoSuchProcess:
            pass
        except Exception as error:
            errors.append(f"terminate {process.pid}: {type(error).__name__}")
    if job is not None:
        job.close()
    elif child is not None and os.name != "nt":
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    _, alive = psutil.wait_procs(owned, timeout=CLEANUP_TIMEOUT_SECONDS)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
        except Exception as error:
            errors.append(f"kill {process.pid}: {type(error).__name__}")
    if child is not None and os.name != "nt":
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _, alive = psutil.wait_procs(alive, timeout=CLEANUP_TIMEOUT_SECONDS)
    errors.extend(f"still alive {process.pid}" for process in alive)
    if child is not None:
        try:
            child.wait(timeout=CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            child.kill()
            try:
                child.wait(timeout=CLEANUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                errors.append("root wait timed out after kill")
    return errors


def monitored_run(command, env, log, receipt_path, on_tick=None,
                  progress_dir=None, stall_seconds=900, poll_seconds=1):
    import psutil
    if stall_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("stall_seconds and poll_seconds must be positive")
    receipt_path = Path(receipt_path)
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    status_path = receipt_path.parent / "status.json"
    hook_path = receipt_path.parent / "hook.json"
    hook = json.loads(hook_path.read_text(encoding="utf-8")) if hook_path.exists() else {}
    provenance = {role + "_" + kind: {**EXPOSURE, "stackers_cross_fitted":
                  "not_applied" if kind == "raw" else False}
                  for role in hook.get("models", "heldout,f0").split(",")
                  for kind in ("raw", "post")}
    progress_dir = Path(progress_dir) if progress_dir is not None else receipt_path.parent / "journal"
    initial = _journal_snapshot(progress_dir)
    observed = initial
    started = time.perf_counter()
    last_progress = started
    last_commit = max((stamp[0] / 1e9 for stamp in initial.values()), default=None)
    samples, tracked, telemetry_errors, cleanup_errors = [], {}, [], []
    child = job = None
    telemetry = _Telemetry(env, poll_seconds)
    gpu, baseline = [0, 0], None
    code, state, failure = 1, "STARTING", None
    stopped = stalled = False
    heartbeat = 0
    status_writer = _StatusWriter(status_path)
    logged_status_errors = 0
    stall_reason = None

    def log_status_errors(stream):
        nonlocal logged_status_errors
        health = status_writer.snapshot()
        if health["error_count"] > logged_status_errors:
            stream.write(f"\nHEARTBEAT_ERROR: {health['error_count']} exhausted writes; "
                         f"{health['last_error']}\n")
            stream.flush()
            logged_status_errors = health["error_count"]
        return health

    def save_status(final=False):
        status_writer.submit({"version": "1.1.3", **EXPOSURE,
            "identity": hook.get("identity"), "role_provenance": provenance,
            "state": state, "completed_count": len(observed),
            "initial_completed_count": len(initial),
            "new_completed_count": len(set(observed) - set(initial)),
            "last_commit_utc": _utc(last_commit) if last_commit is not None else None,
            "last_commit_unix": last_commit,
            "last_commit_monotonic": last_progress if observed != initial else None,
            "idle_seconds": time.perf_counter() - last_progress,
            "heartbeat_utc": _utc(), "heartbeat_count": heartbeat,
            "idle_heartbeat": heartbeat, "stall_seconds": stall_seconds,
            "ram_guard_stopped": stopped, "stalled": stalled,
            "stall_reason": stall_reason, "heartbeat_health": status_writer.snapshot(),
            "telemetry_error_count": len(telemetry_errors),
            "cleanup_complete": final and not cleanup_errors,
            "exit_code": code if final else None})

    with log.open("w", encoding="utf-8") as stream:
        try:
            status_writer.start()
            save_status()
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=(os.name != "nt"))
            root = psutil.Process(child.pid)
            tracked[root.pid] = root
            job = _WindowsJob(child)
            telemetry.start()
            state = "RUNNING"
            while child.poll() is None:
                heartbeat += 1
                try:
                    processes = [root] + root.children(recursive=True)
                except psutil.NoSuchProcess:
                    processes = list(tracked.values())
                rss = private = cpu = 0
                for process in processes:
                    try:
                        tracked.setdefault(process.pid, process)
                        info = process.memory_info()
                        rss += info.rss
                        private += getattr(info, "private", info.rss)
                        try:
                            cpu += process.cpu_percent(interval=None)
                        except Exception as error:
                            telemetry_errors.append(f"process CPU telemetry: {type(error).__name__}")
                    except psutil.NoSuchProcess:
                        pass
                # RAM decision precedes telemetry, exports, and filesystem scans.
                if max(rss, private) >= RAM_LIMIT_BYTES:
                    samples.append([time.perf_counter(), rss, private, cpu, 0, *gpu])
                    stopped, state, code = True, "RAM_GUARD", 2
                    stream.write("\nRAM_GUARD: owned process tree exceeded memory limit\n")
                    stream.flush()
                    break
                telemetry_result = telemetry.take()
                if telemetry_result is not None:
                    value, error = telemetry_result
                    if error:
                        telemetry_errors.append(error)
                    else:
                        gpu = value
                        if baseline is None:
                            baseline = value[0]
                try:
                    system_cpu = psutil.cpu_percent(interval=None)
                except Exception as error:
                    telemetry_errors.append(f"cpu telemetry: {type(error).__name__}")
                    system_cpu = 0
                samples.append([time.perf_counter(), rss, private, cpu, system_cpu, *gpu])
                current = _journal_snapshot(progress_dir)
                changed = [key for key, stamp in current.items() if observed.get(key) != stamp]
                if changed:
                    last_progress = time.perf_counter()
                    last_commit = max(current[key][0] / 1e9 for key in changed)
                observed = current
                health = log_status_errors(stream)
                if time.perf_counter() - last_progress >= stall_seconds:
                    stall_reason = "no study commits"
                if stall_reason:
                    stalled, state, code = True, "STALL", 3
                    stream.write(f"\nSTALL: {stall_reason} for >= {stall_seconds:g} seconds; "
                                 f"heartbeat_errors={health['error_count']}; cleanup then nonzero exit\n")
                    stream.flush()
                    save_status()
                    break
                save_status()
                if on_tick:
                    on_tick()
                time.sleep(poll_seconds)
            if state == "RUNNING":
                code = child.wait(timeout=CLEANUP_TIMEOUT_SECONDS)
                state = "COMPLETED" if code == 0 else "CHILD_FAILED"
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            state, code = "MONITOR_ERROR", 130 if isinstance(error, KeyboardInterrupt) else 1
            stream.write(f"\nMONITOR_ERROR: {failure}\n")
            stream.flush()
        finally:
            try:
                cleanup_errors = _cleanup(child, tracked, job, psutil)
            except BaseException as error:
                cleanup_errors.append(f"cleanup: {type(error).__name__}: {error}")
                if job is not None:
                    job.close()
            telemetry_stopped = True
            if telemetry.thread.ident is not None:
                telemetry_stopped = telemetry.close()
                if not telemetry_stopped:
                    telemetry_errors.append("telemetry worker exceeded bounded shutdown timeout")
            pending_telemetry = telemetry.take()
            if pending_telemetry and pending_telemetry[1]:
                telemetry_errors.append(pending_telemetry[1])
            if cleanup_errors:
                code = code or 1
            current = _journal_snapshot(progress_dir)
            changed = [key for key, stamp in current.items() if observed.get(key) != stamp]
            if changed:
                last_progress = time.perf_counter()
                last_commit = max(current[key][0] / 1e9 for key in changed)
            observed = current
            try:
                logical_cpus = psutil.cpu_count()
            except Exception as error:
                telemetry_errors.append(f"CPU count telemetry: {type(error).__name__}")
                logical_cpus = None
            # Child cleanup always precedes waiting for bounded status retries.
            # A failed final heartbeat must not suppress the independent receipt.
            save_status(final=True)
            status_stopped = status_writer.close() if status_writer.thread.ident is not None else True
            health = log_status_errors(stream)
            receipt = {"version": "1.1.3", **EXPOSURE, "exit_code": code,
                "identity": hook.get("identity"), "role_provenance": provenance,
                "state": state, "seconds": time.perf_counter() - started,
                "ram_guard_stopped": stopped, "ram_guard_bytes": RAM_LIMIT_BYTES,
                "stalled": stalled, "stall_seconds": stall_seconds,
                "stall_reason": stall_reason, "heartbeat_health": health,
                "status_worker_stopped": status_stopped,
                "monitor_error": failure, "telemetry_errors": telemetry_errors,
                "telemetry_worker_stopped": telemetry_stopped,
                "cleanup_errors": cleanup_errors, "cleanup_complete": not cleanup_errors,
                "completed_count": len(observed), "initial_completed_count": len(initial),
                "last_commit_utc": _utc(last_commit) if last_commit is not None else None,
                "status_path": str(status_path), "baseline_gpu_mib": baseline,
                "logical_cpus": logical_cpus,
                "peak_rss_bytes": max((row[1] for row in samples), default=0),
                "peak_private_bytes": max((row[2] for row in samples), default=0),
                "peak_total_gpu_mib": max((row[5] for row in samples), default=0),
                "samples_columns": ["monotonic", "rss", "private", "tree_cpu_percent_one_core_100",
                                    "system_cpu_percent", "total_gpu_mib", "gpu_util_percent"],
                "samples": samples}
            output_io.receipt(receipt_path, receipt)
    return code, receipt
