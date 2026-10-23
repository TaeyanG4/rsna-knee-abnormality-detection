"""H-B-147 independent supervisor v1.0.0 (2026-10-13).

Changelog: owned-tree termination, completed-work deadlines, bounded telemetry,
immutable named attempts and a crash-conservative shared 12-hour pilot ledger.
Run this INSIDE gpu_job.py; other gpu_job callers are unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil
from tools.job_runtime import file_lock, read_json
from tools.hb147_progress_v1 import atomic_json

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local/scratch/astra-b3/H-B-147"


def stall_signature(sample):
    try:
        return (sample["util"] >= 98 and 0 <= sample["power"] < 90
                and sample["used_mib"] >= min(.94 * sample["total_mib"], .95 * 11264)
                and sample["total_mib"] > 0)
    except (KeyError, TypeError):
        return False


class Policy:
    def __init__(self, start, *, idle=900., stall=600., total=43200.):
        self.start = self.last = start
        self.idle, self.stall, self.total = idle, stall, total
        self.sequence = 0
        self.stall_since = None
        self.phase = "startup"

    def check(self, now, heartbeat=None, telemetry=None):
        if heartbeat:
            self.phase = heartbeat.get("phase", self.phase)
            stamp, seq = heartbeat.get("completed_at"), heartbeat.get("sequence", 0)
            if (isinstance(seq, int) and seq > self.sequence and isinstance(stamp, (int, float))
                    and self.last <= stamp <= now):
                self.sequence, self.last = seq, stamp
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


def supervise(command, attempt, *, total=43200., cpu=False, idle=900., stall=600., poll=5.):
    """Small thresholds injectable for CPU tests only; production CLI fixes them."""
    attempt = Path(attempt)
    attempt.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    env = {**os.environ, "HB147_HEARTBEAT": str((attempt / "heartbeat.json").resolve()),
           "HB147_TOKEN": token}
    started = time.monotonic()
    policy = Policy(started, idle=idle, stall=stall, total=total)
    owned, child, root, reason, code, survivors = {}, None, None, None, None, []
    atomic_json(attempt / "command.json", dict(command=command, token=token, version=VERSION,
                idle_seconds=idle, stall_seconds=stall, remaining_total_seconds=total))
    try:
        with (attempt / "child.log").open("xb", buffering=0) as log, (attempt / "telemetry.jsonl").open("x", encoding="utf-8") as samples:
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
                samples.write(json.dumps(dict(elapsed=now-started, idle=now-policy.last,
                    phase=policy.phase, sequence=policy.sequence, telemetry=sample, decision=reason)) + "\n")
                samples.flush()
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
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    attempt, budget = Path(args.attempt).resolve(), Path(args.budget).resolve()
    if not attempt.is_relative_to(WORK) or not budget.is_relative_to(WORK):
        ap.error("attempt and budget must stay inside H-B-147 scratch")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("command required after --")
    with file_lock(budget.with_suffix(".lock"), blocking=False):
        state = json.loads(budget.read_text()) if budget.exists() else {"used_seconds": 0.}
        now = time.time()
        if state.get("active_since") is not None:
            state["used_seconds"] += max(0., now - state["active_since"])
        remaining = 43200. - state["used_seconds"]
        if remaining <= 0:
            raise RuntimeError("shared 12-hour cap exhausted; no launch")
        if attempt.exists():
            raise FileExistsError("partial attempt preserved; choose an explicitly named new attempt")
        state.update(active_since=now, attempt=str(attempt))
        atomic_json(budget, state)
        start = time.monotonic()
        try:
            return supervise(command, attempt, total=remaining, cpu=args.cpu)
        finally:
            state.update(used_seconds=state["used_seconds"] + time.monotonic()-start, active_since=None)
            atomic_json(budget, state)


if __name__ == "__main__":
    raise SystemExit(main())
