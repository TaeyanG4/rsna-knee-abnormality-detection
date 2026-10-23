"""Run one exact argv with sampled owned-tree RSS; never retry or reset cgroups.

CPU-only supervisor. Example: python process_resource_receipt.py --out NEW_DIR
--timeout 1800 -- python producer.py run ...
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

INTERVAL = 0.1
CGROUP_FILES = ("memory.current", "memory.peak", "memory.events", "memory.max")


def utc():
    return datetime.now(timezone.utc).isoformat()


def mount_unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def resolve_cgroup(cgroup_text, mountinfo_text):
    """Resolve membership against mount root (including namespace-root mounts)."""
    memberships = []
    for line in cgroup_text.splitlines():
        _, controllers, path = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            memberships.append((2 if not controllers else 1, path))
    candidates = []
    for line in mountinfo_text.splitlines():
        before, after = line.split(" - ", 1)
        fields, tail = before.split(), after.split()
        version = 2 if tail[0] == "cgroup2" else 1 if tail[0] == "cgroup" and "memory" in tail[2].split(",") else None
        for member_version, membership in memberships:
            if version != member_version:
                continue
            root, mount = map(mount_unescape, (fields[3], fields[4]))
            # Reject traversal, rather than inventing a path outside the mount.
            if ".." in PurePosixPath(membership).parts:
                continue
            if membership == "/":
                relative = ""  # cgroup namespace reports its own root as '/'.
            elif membership == root:
                relative = ""
            elif root == "/":
                relative = membership.lstrip("/")
            elif membership.startswith(root.rstrip("/") + "/"):
                relative = membership[len(root):].lstrip("/")
            else:
                continue
            candidates.append({"version": version, "membership": membership,
                               "mount_root": root, "mount_point": mount,
                               "path": str(PurePosixPath(mount) / relative)})
    if not candidates:
        raise RuntimeError("actual memory cgroup could not be resolved")
    return sorted(candidates, key=lambda x: (x["version"], len(x["mount_root"])), reverse=True)[0]


def cgroup_snapshot(location):
    if location is None:
        return None
    result = {"observed_at": utc(), "location": location, "values": {}, "errors": {}}
    mapping = dict(zip(CGROUP_FILES, CGROUP_FILES)) if location["version"] == 2 else {
        "memory.current": "memory.usage_in_bytes", "memory.peak": "memory.max_usage_in_bytes",
        "memory.max": "memory.limit_in_bytes", "memory.failcnt": "memory.failcnt"}
    for key, filename in mapping.items():
        try:
            raw = (Path(location["path"]) / filename).read_text().strip()
            result["values"][key] = dict((k, int(v)) for k, v in (row.split() for row in raw.splitlines())) if key == "memory.events" else int(raw) if raw.isdigit() else raw
        except (OSError, ValueError) as exc:
            result["errors"][key] = f"{type(exc).__name__}: {exc}"
    if location["version"] == 1:
        result["errors"]["memory.events"] = "cgroup v1 has no memory.events; memory.failcnt recorded separately"
    return result


class OwnedTree:
    def __init__(self, psutil):
        self.psutil = psutil
        self.known = set()
        self.errors = []
        self.gone_observations = 0
        self.reused_pid_observations = 0

    def record(self, proc):
        identity = (proc.pid, proc.create_time())
        self.known.add(identity)
        return identity

    def matched(self, identity):
        try:
            proc = self.psutil.Process(identity[0])
            if proc.create_time() != identity[1]:
                self.reused_pid_observations += 1
                return None
            if proc.status() == self.psutil.STATUS_ZOMBIE:
                return None
            return proc
        except self.psutil.NoSuchProcess:
            self.gone_observations += 1
            return None

    def discover(self):
        # Known descendants remain owned even after their parent exits/reparents.
        for identity in list(self.known):
            proc = self.matched(identity)
            if proc is None:
                continue
            try:
                children = proc.children(recursive=True)
                for child in children:
                    try:
                        self.record(child)
                    except self.psutil.NoSuchProcess:
                        self.gone_observations += 1
            except self.psutil.NoSuchProcess:
                self.gone_observations += 1

    def live(self):
        return [(identity, proc) for identity in list(self.known)
                if (proc := self.matched(identity)) is not None]

    def sample(self):
        self.discover()
        rss, processes = 0, []
        for identity, proc in self.live():
            try:
                value = proc.memory_info().rss
                # psutil memory_info is not itself identity-reuse guarded.
                if self.matched(identity) is None:
                    continue
                rss += value
                processes.append({"pid": identity[0], "create_time": identity[1], "rss": value})
            except self.psutil.NoSuchProcess:
                self.gone_observations += 1
        return {"rss_bytes": rss, "processes": processes}

    def cleanup(self):
        result = {"actions": [], "errors": [], "survivors": []}
        try:
            self.discover()
        except self.psutil.Error as exc:
            result["errors"].append(f"discovery: {type(exc).__name__}: {exc}")
        # Bounded escalation of the same execution, not a producer retry.
        for method in ("terminate", "kill"):
            for identity in sorted(self.known, key=lambda x: x[1], reverse=True):
                try:
                    proc = self.matched(identity)
                    if proc is not None:
                        getattr(proc, method)()
                        result["actions"].append({"pid": identity[0], "create_time": identity[1], "action": method})
                except self.psutil.Error as exc:
                    result["errors"].append(f"{identity}: {type(exc).__name__}: {exc}")
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    if not self.live():
                        break
                except self.psutil.Error:
                    break
                time.sleep(0.05)
        for identity in self.known:
            try:
                if self.matched(identity) is not None:
                    result["survivors"].append({"pid": identity[0], "create_time": identity[1]})
            except self.psutil.Error as exc:
                result["errors"].append(f"final identity check: {type(exc).__name__}: {exc}")
        return result


def supervise(argv, out, timeout, cwd=None):
    # Validate dependencies/arguments before creating output or launching a child.
    import psutil
    if not argv or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("nonempty argv and finite positive timeout required")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "process-resource-receipt-v1", "started_at": utc(),
               "argv": list(argv), "cwd": str(Path(cwd or os.getcwd()).resolve()),
               "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "platform": sys.platform, "psutil_version": psutil.__version__,
               "timeout_seconds": timeout, "interval_seconds": INTERVAL, "retries": 0,
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
                if now - start >= timeout:
                    raise TimeoutError(f"child exceeded {timeout} seconds")
                next_sample += INTERVAL
                remaining = next_sample - time.monotonic()
                if remaining < 0:
                    missed = int(-remaining // INTERVAL) + 1
                    receipt["missed_schedule_slots"] += missed
                    next_sample += missed * INTERVAL
                time.sleep(max(0, min(next_sample - time.monotonic(), timeout - (time.monotonic() - start))))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--cwd")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = supervise(argv, args.out, args.timeout, args.cwd)
    print(json.dumps({"status": result["status"], "receipt": str(Path(args.out).resolve() / "receipt.json")}))
    return result["wrapper_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
