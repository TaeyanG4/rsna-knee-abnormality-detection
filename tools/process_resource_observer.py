"""Untimed owned-process observer, v1.0.0.

Changelog 2026-10-12: duration is telemetry, never admission or cancellation.
The historical hash-bound supervisor remains immutable. Only failure cleanup
uses bounded terminate/kill escalation; a healthy child has no wall deadline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from tools import process_resource_receipt as helpers
from tools.process_resource_receipt import OwnedTree, resolve_cgroup, cgroup_snapshot, utc, INTERVAL

VERSION = "1.0.0"
SCHEMA = "process-resource-observer/v1"

def helper_hash():
    return hashlib.sha256(Path(helpers.__file__).read_bytes()).hexdigest()

def supervise(argv, out, cwd=None):
    # Validate dependencies/arguments before creating output or launching a child.
    import psutil
    if not argv:
        raise ValueError("nonempty argv required")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "process-resource-observer/v1", "started_at": utc(),
               "argv": list(argv), "cwd": str(Path(cwd or os.getcwd()).resolve()),
               "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "platform": sys.platform, "psutil_version": psutil.__version__,
               "helper_sha256": helper_hash(), "interval_seconds": INTERVAL, "retries": 0,
               "status": "FAILED", "failure": None, "child_exit_code": None,
               "sample_count": 0, "sample_attempts": 0, "missed_schedule_slots": 0,
               "max_sample_gap_seconds": 0.0, "sampled_tree_rss_max_bytes": None,
               "cgroup_scope": "session context, possibly shared; never a producer-exclusive peak; peak never reset",
               "limitations": ["100ms sampling can miss instantaneous RSS peaks and short-lived or detached descendants",
                               "Summed process RSS can double-count shared memory; it is not unique physical RAM",
                               "Only observed PID+creation-time identities can be cleaned; no process-group/global kill",
                               "Supervisor hard kill or machine shutdown can prevent cleanup/receipt finalization"]}
    tree, child, location = OwnedTree(psutil), None, None
    start = time.monotonic()
    try:
        if sys.platform.startswith("linux"):
            location = resolve_cgroup(Path("/proc/self/cgroup").read_text(), Path("/proc/self/mountinfo").read_text())
            receipt["cgroup_start"] = cgroup_snapshot(location)
        receipt["system_ram_start"] = {"available": psutil.virtual_memory().available, "total": psutil.virtual_memory().total}
        with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr, (out / "samples.jsonl").open("x", encoding="utf-8") as samples:
            child = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     shell=False, start_new_session=os.name == "posix")
            try:
                identity = tree.record(psutil.Process(child.pid))
                receipt["child_identity"] = {"pid": identity[0], "create_time": identity[1]}
            except psutil.NoSuchProcess:
                tree.gone_observations += 1
                raise RuntimeError("child vanished before its creation-time identity could be recorded")
            if location:
                try:
                    receipt["child_cgroup_at_start"] = Path(f"/proc/{child.pid}/cgroup").read_text()
                except OSError as exc:
                    receipt["child_cgroup_read_error"] = str(exc)
            next_sample, previous = time.monotonic(), None
            while True:
                now = time.monotonic()
                if previous is not None:
                    receipt["max_sample_gap_seconds"] = max(receipt["max_sample_gap_seconds"], now - previous)
                previous = now
                receipt["sample_attempts"] += 1
                sample = tree.sample()
                sample["elapsed_seconds"] = now - start
                samples.write(json.dumps(sample) + "\n"); samples.flush()
                receipt["sample_count"] += 1
                receipt["sampled_tree_rss_max_bytes"] = max(receipt["sampled_tree_rss_max_bytes"] or 0, sample["rss_bytes"])
                code = child.poll()
                if code is not None:
                    receipt["child_exit_code"] = code
                    if code != 0:
                        raise RuntimeError(f"child exited {code}")
                    survivors = [identity for identity, _ in tree.live() if identity[0] != child.pid]
                    if survivors:
                        raise RuntimeError(f"descendants survived child exit: {survivors}")
                    receipt["status"] = "COMPLETE"
                    break
                next_sample += INTERVAL
                remaining = next_sample - time.monotonic()
                if remaining < 0:
                    missed = int(-remaining // INTERVAL) + 1
                    receipt["missed_schedule_slots"] += missed
                    next_sample += missed * INTERVAL
                time.sleep(max(0, next_sample - time.monotonic()))
    except BaseException as exc:
        receipt["status"] = "FAILED"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, psutil.AccessDenied):
            tree.errors.append(receipt["failure"])
    finally:
        if child is not None:
            receipt["child_exit_code_before_cleanup"] = child.poll()
            receipt["cleanup"] = tree.cleanup()
            if "child_identity" not in receipt and child.poll() is None:
                # Popen still owns the unreaped child / Windows process handle;
                # this does not look up or signal an arbitrary PID.
                try:
                    child.kill()
                    receipt["cleanup"]["actions"].append({"pid": child.pid, "action": "kill_owned_popen_handle", "creation_time_unavailable": True})
                except OSError as exc:
                    receipt["cleanup"]["errors"].append(f"owned root handle cleanup: {exc}")
            try:
                receipt["child_exit_code"] = child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                receipt["cleanup"]["errors"].append("root Popen did not exit after owned cleanup")
            if receipt["cleanup"]["errors"] or receipt["cleanup"]["survivors"]:
                receipt["status"] = "FAILED"
                receipt["failure"] = receipt["failure"] or {"type": "CleanupFailure", "message": "owned process cleanup incomplete"}
        receipt["cgroup_end"] = cgroup_snapshot(location)
        receipt["system_ram_end"] = {"available": psutil.virtual_memory().available, "total": psutil.virtual_memory().total}
        receipt.update(ended_at=utc(), elapsed_seconds=time.monotonic() - start,
                       observed_identities=[{"pid": p, "create_time": c} for p, c in sorted(tree.known)],
                       access_failures=tree.errors, gone_observations=tree.gone_observations,
                       reused_pid_observations=tree.reused_pid_observations)
        receipt["access_failure_count"] = len(tree.errors) + sum("AccessDenied" in error for error in receipt.get("cleanup", {}).get("errors", []))
        receipt["failed_sample_attempts"] = receipt["sample_attempts"] - receipt["sample_count"]
        receipt["cgroup_telemetry_complete"] = None if location is None else all(
            not receipt[key]["errors"] for key in ("cgroup_start", "cgroup_end"))
        receipt["wrapper_exit_code"] = 0 if receipt["status"] == "COMPLETE" else 1
        with (out / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
    return receipt


def resource_gate(receipt, argv):
    """Validate exact execution/identity/telemetry, with no elapsed ceiling."""
    def need(ok, message):
        if not ok:
            raise ValueError(message)
    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    need(receipt.get("schema") == SCHEMA and "timeout_seconds" not in receipt,
         "untimed resource schema required")
    need(receipt.get("argv") == list(argv) and receipt.get("retries") == 0 and
         receipt.get("supervisor_sha256") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest() and
         receipt.get("helper_sha256") == helper_hash(), "resource command/source differs")
    need(receipt.get("status") == "COMPLETE" and receipt.get("failure") is None and
         all(type(receipt.get(k)) is int and receipt[k] == 0 for k in
             ("child_exit_code", "wrapper_exit_code", "child_exit_code_before_cleanup")),
         "resource execution failed")
    child = receipt.get("child_identity", {})
    need(type(child.get("pid")) is int and child["pid"] > 0 and positive(child.get("create_time")) and
         child in receipt.get("observed_identities", []), "owned worker identity missing")
    need(type(receipt.get("sample_count")) is int and receipt["sample_count"] > 0 and
         receipt.get("sample_attempts") == receipt["sample_count"] and
         positive(receipt.get("sampled_tree_rss_max_bytes")) and
         receipt.get("access_failures") == [] and
         receipt.get("access_failure_count") == receipt.get("failed_sample_attempts") == 0 and
         all(receipt.get("cleanup", {}).get(k) == [] for k in ("actions", "errors", "survivors")),
         "resource telemetry/cleanup failed")
    need(positive(receipt.get("elapsed_seconds")), "invalid resource duration")
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cwd")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = supervise(argv, args.out, args.cwd)
    print(json.dumps({"status": result["status"], "receipt": str(Path(args.out).resolve() / "receipt.json")}))
    return result["wrapper_exit_code"]

if __name__ == "__main__":
    raise SystemExit(main())
