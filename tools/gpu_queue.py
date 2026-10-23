"""Sequential GPU queue with explicit dependencies, v2.0.0 (2026-10-12).

Changelog: PID/start/receipt liveness, DAG failure propagation and independent
branches, command-bound reuse, validation before launch. No waiting deadline.
CLI: gpu_queue.py plan.json [--foreground]. Plan entries retain tag, command,
after (tag, or list of tags), continue_on_failure. Log path is unchanged.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.job_runtime import (ROOT, LOGS, CODE_ROOT, atomic_json, digest,
    job_status, read_json, tag_name)

VERSION = "2.0.0"


def state(tag):
    return job_status(LOGS, tag)["state"]


def wait_for(tag, poll=30):
    while True:
        value = state(tag)
        if value != "running":
            return value
        time.sleep(poll)


def dependencies(job):
    after = job.get("after", [])
    return [after] if isinstance(after, str) else after


def validate_plan(plan):
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must be a nonempty list")
    indexed = {}
    for job in plan:
        if not isinstance(job, dict):
            raise ValueError("job must be an object")
        tag = tag_name(job.get("tag"))
        cmd = job.get("command")
        if tag in indexed or not isinstance(cmd, list) or not cmd or any(not isinstance(x, str) or not x or '\x00' in x for x in cmd):
            raise ValueError("duplicate tag or invalid command")
        if "continue_on_failure" in job and type(job["continue_on_failure"]) is not bool:
            raise ValueError("continue_on_failure must be boolean")
        deps = dependencies(job)
        if not isinstance(deps, list) or len(set(deps)) != len(deps):
            raise ValueError("invalid dependencies")
        for dep in deps:
            tag_name(dep)
        indexed[tag] = job
    done, visiting = set(), set()
    def visit(tag):
        if tag in visiting:
            raise ValueError("dependency cycle")
        if tag in done or tag not in indexed:
            return
        visiting.add(tag)
        for dep in dependencies(indexed[tag]):
            visit(dep)
        visiting.remove(tag); done.add(tag)
    for tag in indexed:
        visit(tag)
    return plan


def command_matches(job):
    path = LOGS / job["tag"] / "job.json"
    if not path.exists():
        return True
    return job_status(LOGS, job["tag"], expected_command=job["command"])["state"] != "unknown"


def run_plan(plan_path):
    plan = validate_plan(read_json(plan_path))
    log = LOGS / "queue" / (Path(plan_path).stem + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    def note(msg):
        line = time.strftime("%Y-%m-%dT%H:%M:%S ") + msg
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    pending = {job["tag"]: job for job in plan}
    results, tolerated = {}, set()
    while pending:
        for tag, job in list(pending.items()):
            deps = dependencies(job)
            if any(dep in pending for dep in deps):
                continue
            failed = []
            for dep in deps:
                predecessor = results.get(dep)
                if predecessor is None:
                    predecessor = wait_for(dep)
                if predecessor != "SUCCESS":
                    failed.append(dep)
            if failed and not job.get("continue_on_failure"):
                results[tag] = "BLOCKED"
                note(f"plan stopped before {tag}: dependency failed/unavailable: {', '.join(failed)}")
            elif not command_matches(job):
                results[tag] = "CONFLICT"
                note(f"{tag}: existing command or receipt differs; use a new tag")
            else:
                existing = state(tag)
                if existing == "running":
                    note(f"{tag}: already running, waiting without launching a duplicate")
                    existing = wait_for(tag)
                if existing == "SUCCESS":
                    results[tag] = "SUCCESS"
                    note(f"{tag}: already SUCCESS, skipping")
                elif existing in {"unknown", "orphaned", "stale", "FAILED"}:
                    results[tag] = existing
                    note(f"{tag}: {existing}; no implicit retry")
                else:
                    note(f"{tag}: starting")
                    result = subprocess.run([sys.executable, "-B", str(CODE_ROOT / "tools" / "gpu_job.py"),
                        "--tag", tag, "--wait", "--", *job["command"]], cwd=ROOT,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    final = state(tag)
                    results[tag] = "SUCCESS" if result.returncode == 0 and final == "SUCCESS" else "FAILED"
                    note(f"{tag}: {final} (launcher exit {result.returncode})")
                if results[tag] == "SUCCESS" and job.get("continue_on_failure"):
                    tolerated.update(failed)
            del pending[tag]
    failed = [tag for tag, result in results.items() if result != "SUCCESS" and tag not in tolerated]
    code = int(bool(failed))
    atomic_json(log.with_name(log.stem + ".receipt-" + uuid.uuid4().hex + ".json"), {
        "schema": "gpu-queue-receipt/v1", "plan_sha256": digest(plan), "results": results,
        "tolerated_failures": sorted(tolerated), "exit_code": code})
    note("plan finished" if not code else "plan stopped: failed or blocked jobs remain")
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--foreground", action="store_true")
    a = ap.parse_args(argv)
    try:
        validate_plan(read_json(a.plan))
        if a.foreground:
            return run_plan(a.plan)
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
        env = {**os.environ, "PYTHONUTF8": "1", "KNEE_WORKSPACE_ROOT": str(ROOT)}
        proc = subprocess.Popen([sys.executable, "-B", __file__, str(Path(a.plan).resolve()), "--foreground"],
            cwd=ROOT, env=env, creationflags=flags, start_new_session=os.name != "nt",
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"queue runner started pid {proc.pid}; log {LOGS / 'queue' / (Path(a.plan).stem + '.log')}")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
