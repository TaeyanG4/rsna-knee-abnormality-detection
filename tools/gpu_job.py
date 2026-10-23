"""Logged detached local jobs, v2.0.0 (2026-10-12).

Changelog: atomic single-lane claim, PID/start identities, command-bound terminal
receipts and bounded status reads. Public CLI and legacy marker text preserved.
  python tools/gpu_job.py --tag TAG [--wait] -- command args...
  python tools/gpu_job.py --status TAG
  python tools/gpu_job.py --list
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
from tools.job_runtime import (ROOT, LOGS, CODE_ROOT, active_jobs, atomic_json,
    atomic_text, command_digest, file_lock, identity, job_status, read_json, tag_name)

VERSION = "2.0.0"


def status(tag, *, tail=True):
    return job_status(LOGS, tag, tail=tail)


def finish(directory, job, code, error=None):
    ended = time.strftime("%Y-%m-%dT%H:%M:%S")
    state = "SUCCESS" if code == 0 else "FAILED"
    atomic_json(directory / "receipt.json", {
        "schema": "gpu-job-receipt/v1", "run_id": job["run_id"], "tag": job["tag"],
        "command_sha256": job["command_sha256"], "state": state, "exit_code": code,
        "wrapper_identity": {k: job[k] for k in ("pid", "create_time")},
        "child_identity": job.get("child_identity"), "ended": ended,
        "error_type": error, "elapsed_seconds": time.time() - job["started_epoch"],
    })
    atomic_text(directory / state, f"exit {code} at {ended}")


def run_job(tag, run_id):
    directory = LOGS / tag_name(tag)
    with file_lock(LOGS / ".gpu-claim.lock"):
        job = read_json(directory / "job.json")
        if job.get("run_id") != run_id or job["command_sha256"] != command_digest(job["command"], ROOT):
            return 2
        job.update(identity(), phase="starting")
        atomic_json(directory / "job.json", job)
        try:
            # Keep the original environment fields; listing/status never import torch.
            sys.path.insert(0, str(CODE_ROOT / "training" / "src"))
            from raptor_common import environment
            job["environment"] = environment()
            job["phase"] = "spawning"
            atomic_json(directory / "job.json", job)
            with (directory / "run.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(job["command"], cwd=ROOT, stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            try:
                job["child_identity"] = identity(child.pid)
            except Exception:
                # A very short command can already have exited. Reap it before publishing.
                code = child.poll()
                if code is None:
                    raise
                job["child_identity"] = None
            job["phase"] = "running"
            atomic_json(directory / "job.json", job)
        except Exception as exc:
            # If a child might exist, retain the spawning reservation (fail closed).
            if "child" in locals() and child.poll() is None:
                raise
            finish(directory, job, 1, type(exc).__name__)
            return 1
    code = child.wait()
    finish(directory, job, code)
    return code


def launch(tag, command, wait=False):
    tag_name(tag)
    directory = LOGS / tag
    with file_lock(LOGS / ".gpu-claim.lock"):
        busy = active_jobs(LOGS)
        if busy:
            raise ValueError("GPU lane occupied or unconfirmed: " + ", ".join(x["tag"] for x in busy))
        directory.mkdir(parents=True, exist_ok=True)
        old = [directory / n for n in ("job.json", "run.log", "SUCCESS", "FAILED", "receipt.json") if (directory / n).exists()]
        if old:
            archive = directory / "attempts" / uuid.uuid4().hex
            archive.mkdir(parents=True)
            for path in old:
                path.replace(archive / path.name)
        job = {"schema": "gpu-job/v2", "tag": tag, "run_id": uuid.uuid4().hex,
               **identity(), "command": command, "cwd": str(ROOT),
               "command_sha256": command_digest(command, ROOT), "phase": "dispatching",
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "started_epoch": time.time(),
               "environment": {}, "child_identity": None}
        atomic_json(directory / "job.json", job)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "KNEE_WORKSPACE_ROOT": str(ROOT)}
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
        try:
            proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--_run-job", tag, job["run_id"]],
                cwd=ROOT, env=env, creationflags=flags, start_new_session=os.name != "nt",
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            job.update(identity(proc.pid))
            atomic_json(directory / "job.json", job)
        except Exception as exc:
            if "proc" in locals():
                raise  # wrapper may exist; never fabricate a terminal result
            finish(directory, job, 1, type(exc).__name__)
            return 1
    print(f"started {tag} pid {proc.pid}; log {directory / 'run.log'}", flush=True)
    if wait:
        code = proc.wait()
        result = status(tag)
        print(json.dumps(result, indent=1, ensure_ascii=False))
        return code or (0 if result["state"] == "SUCCESS" else 1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag")
    ap.add_argument("--status")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--_run-job", nargs=2, help=argparse.SUPPRESS)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    try:
        if a._run_job:
            return run_job(*a._run_job)
        if a.list:
            for path in sorted(LOGS.glob("*/job.json")):
                s = status(path.parent.name, tail=False)
                print(f"{s['state']:8s} {s['tag']:40s} {s['started'] or ''}")
            return 0
        if a.status:
            print(json.dumps(status(a.status), indent=1, ensure_ascii=False))
            return 0
        command = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not a.tag or not command:
            ap.error("need --tag and a command after --")
        return launch(a.tag, command, a.wait)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
