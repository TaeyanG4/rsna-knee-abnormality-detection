"""H-B-150 independent supervisor v2.0.0 (2026-10-14).

Changelog v2: separate launch-v115-bound stage; scientific behavior unchanged.

Changelog: owned-tree termination, completed-work deadlines, bounded telemetry,
immutable named attempts and a crash-conservative shared 12-hour pilot ledger.
Fork of H-B-147 v1.0.0: bounded output IO and phase-normalized 30-minute
throughput-collapse aborts against measured healthy CUDA-smoke rates.
Run this INSIDE gpu_job.py; other gpu_job callers are unchanged.
"""
import argparse
from collections import deque
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil
from tools.job_runtime import file_lock, read_json
from tools.hb150_io_v1 import atomic_json

VERSION = "2.0.0"
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local/scratch/astra-b3/H-B-147"
WORK150 = ROOT / ".local/scratch/astra-b3/H-B-150"


def stall_signature(sample):
    try:
        return (sample["util"] >= 98 and 0 <= sample["power"] < 90
                and sample["used_mib"] >= min(.94 * sample["total_mib"], .95 * 11264)
                and sample["total_mib"] > 0)
    except (KeyError, TypeError):
        return False


class Policy:
    def __init__(self, start, *, idle=900., stall=600., total=43200.,
                 healthy_rates=None, collapse=1800., fraction=.25):
        self.start = self.last = start
        self.idle, self.stall, self.total = idle, stall, total
        self.sequence = 0
        self.stall_since = None
        self.phase = "startup"
        self.rates = dict(healthy_rates or {})
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in self.rates.values()):
            raise ValueError("healthy rates must be finite positive units per second")
        self.collapse, self.fraction = collapse, fraction
        self.kind = None
        self.previous_check = start
        self.counts = {}
        self.active_time = 0.
        self.work = deque()
        self.ratio = None

    def check(self, now, heartbeat=None, telemetry=None):
        # Clock is measured only in comparable phases. Switches never manufacture
        # work or reset accumulated slow time; checkpoint/startup pause the clock.
        if self.kind in self.rates:
            self.active_time += max(0., now - self.previous_check)
        self.previous_check = now
        if heartbeat:
            self.phase = heartbeat.get("phase", self.phase)
            detail = heartbeat.get("detail") or {}
            kind = detail.get("progress_kind") if isinstance(detail, dict) else None
            self.kind = kind or {"training_step": "train_step", "validation": "validation",
                                "panel_inference": "panel"}.get(self.phase)
            stamp, seq = heartbeat.get("completed_at"), heartbeat.get("sequence", 0)
            if (isinstance(seq, int) and seq > self.sequence and isinstance(stamp, (int, float))
                    and self.last <= stamp <= now and heartbeat.get("event") != "enter"):
                self.sequence, self.last = seq, stamp
                units = detail.get("cumulative_units") if isinstance(detail, dict) else None
                if self.kind in self.rates and isinstance(units, (float, int)) and math.isfinite(units):
                    prior = self.counts.get(self.kind, 0.)
                    if units >= prior:
                        self.work.append((self.active_time, (units-prior)/self.rates[self.kind]))
                        self.counts[self.kind] = units
        while self.work and self.work[0][0] <= self.active_time - self.collapse:
            self.work.popleft()
        if self.active_time >= self.collapse:
            self.ratio = sum(credit for _, credit in self.work) / self.collapse
        if stall_signature(telemetry):
            if self.stall_since is None:
                self.stall_since = now
        else:
            self.stall_since = None
        if now - self.start >= self.total:
            return "total_cap"
        if now - self.last >= self.idle:
            return "no_heartbeat_15m"
        if (self.stall_since is not None and now - self.stall_since >= self.stall
                and now - self.last >= self.stall):
            return "wddm_no_heartbeat_10m"
        if self.ratio is not None and self.ratio < self.fraction:
            return "throughput_below_25pct_30m"
        return None


def telemetry():
    try:
        result = subprocess.run(["nvidia-smi", "--id=0",
            "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            return {"error": "nvidia-smi exit " + str(result.returncode)}
        values = [float(v.strip()) for v in result.stdout.strip().split(",")]
        return dict(zip(("util", "power", "used_mib", "total_mib"), values))
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def collect_owned(root, owned):
    """psutil Process objects retain PID+creation identity; never PID-name kills."""
    try:
        if root.is_running():
            for proc in root.children(recursive=True):
                try:
                    owned[(proc.pid, proc.create_time())] = proc
                except psutil.NoSuchProcess:
                    pass
    except psutil.NoSuchProcess:
        pass  # Normal race: child exits between is_running() and children().


def terminate_owned(root, owned):
    try:
        collect_owned(root, owned)
    except psutil.NoSuchProcess:
        pass
    # Stop the parent first to prevent new descendants; retain previously seen
    # children even if they have already become orphaned.
    victims = [root] + list(owned.values())
    for proc in victims:
        try:
            if proc.is_running():
                proc.kill()  # psutil checks identity before signalling.
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(victims, timeout=5)
    return [{"pid": p.pid, "create_time": p.create_time()} for p in alive]


def supervise(command, attempt, *, total=43200., cpu=False, idle=900., stall=600., poll=5.,
              healthy_rates=None):
    """Small thresholds injectable for CPU tests only; production CLI fixes them."""
    attempt = Path(attempt)
    attempt.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    env = {**os.environ, "HB147_HEARTBEAT": str((attempt / "heartbeat.json").resolve()),
           "HB147_TOKEN": token}
    started = time.monotonic()
    policy = Policy(started, idle=idle, stall=stall, total=total, healthy_rates=healthy_rates)
    owned, child, root, reason, code, survivors = {}, None, None, None, None, []
    atomic_json(attempt / "command.json", dict(command=command, token=token, version=VERSION,
                idle_seconds=idle, stall_seconds=stall, remaining_total_seconds=total,
                healthy_rates=healthy_rates, collapse_seconds=1800, collapse_fraction=.25))
    try:
        with ExitStack() as stack:
            try:
                log = stack.enter_context((attempt / "child.log").open("xb", buffering=0))
            except OSError:
                log = subprocess.DEVNULL
            try:
                samples = stack.enter_context((attempt / "telemetry.jsonl").open("x", encoding="utf-8"))
            except OSError:
                samples = None
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            root = psutil.Process(child.pid)
            atomic_json(attempt / "identity.json", dict(pid=root.pid, create_time=root.create_time()))
            while True:
                collect_owned(root, owned)
                if child.poll() is not None:
                    code = child.returncode
                    reason = "completed" if code == 0 else "child_failed"
                    break
                sample = {"disabled": "CPU"} if cpu else telemetry()
                now = time.monotonic()
                heartbeat = None
                try:
                    record = read_json(attempt / "heartbeat.json")
                    if record.get("token") == token and record.get("pid") == child.pid:
                        heartbeat = record
                except (OSError, ValueError):
                    pass
                reason = policy.check(now, heartbeat, sample)
                try:
                    if samples is not None:
                        samples.write(json.dumps(dict(elapsed=now-started, idle=now-policy.last,
                            phase=policy.phase, sequence=policy.sequence, telemetry=sample,
                            normalized_throughput=policy.ratio, decision=reason)) + "\n")
                        samples.flush()
                except OSError:
                    pass  # Telemetry is an export, never a training journal.
                if reason:
                    code = 124
                    break
                time.sleep(min(poll, max(.001, total - (time.monotonic()-started))))
    except BaseException:
        reason, code = "supervisor_error", 125
        raise
    finally:
        if root is not None:
            survivors = terminate_owned(root, owned)
        if child is not None:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                code, reason = 125, "termination_unconfirmed"
        if survivors:
            code, reason = 125, "termination_unconfirmed"
        atomic_json(attempt / "receipt.json", dict(version=VERSION, reason=reason, exit_code=code,
            elapsed_seconds=time.monotonic()-started, last_phase=policy.phase,
            completed_events=policy.sequence, survivors=survivors, partial_attempt=str(attempt)))
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--budget", required=True, help="Same ledger for all four fits and panel jobs")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--healthy-rates", help="JSON file: measured train_step/validation/panel units per second")
    ap.add_argument("--measurement", action="store_true", help="Only the bounded H-B-150 CUDA full-shape smoke")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    attempt, budget = Path(args.attempt).resolve(), Path(args.budget).resolve()
    if not all(any(path.is_relative_to(root) for root in (WORK, WORK150)) for path in (attempt, budget)):
        ap.error("attempt and budget must stay inside H-B-147 or H-B-150 scratch")
    rates = json.loads(Path(args.healthy_rates).read_text()) if args.healthy_rates else None
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.measurement:
        if (args.cpu or len(command) < 5 or command[1:4] != ["-B", "-m", "tools.hb150_fit_v2"]
                or command[4] != "smoke" or "--device" not in command
                or command[command.index("--device")+1:] == []
                or command[command.index("--device")+1] != "cuda"):
            ap.error("measurement is restricted to python -B -m tools.hb150_fit_v2 smoke --device cuda")
    if not args.cpu and not args.measurement and (not isinstance(rates, dict) or not {"train_step", "validation"} <= rates.keys()):
        ap.error("GPU fits require measured healthy rates for train_step and validation")
    Policy(0., healthy_rates=rates)  # Fail before creating an attempt or claiming its budget.
    if not command:
        ap.error("command required after --")
    with file_lock(budget.with_suffix(".lock"), blocking=False):
        state = json.loads(budget.read_text()) if budget.exists() else {"used_seconds": 0.}
        now = time.time()
        if state.get("active_since") is not None:
            state["used_seconds"] += max(0., now - state["active_since"])
        remaining = 43200. - state["used_seconds"]
        if args.measurement:
            remaining = min(remaining, 900.)
        if remaining <= 0:
            raise RuntimeError("shared 12-hour cap exhausted; no launch")
        if attempt.exists():
            raise FileExistsError("partial attempt preserved; choose an explicitly named new attempt")
        state.update(active_since=now, attempt=str(attempt))
        atomic_json(budget, state)
        start = time.monotonic()
        try:
            return supervise(command, attempt, total=remaining, cpu=args.cpu, healthy_rates=rates)
        finally:
            state.update(used_seconds=state["used_seconds"] + time.monotonic()-start, active_since=None)
            atomic_json(budget, state)


if __name__ == "__main__":
    raise SystemExit(main())
