"""Launch a fresh, bounded Codex/Claude CLI task and collect its result locally.

No model override, shell command interpolation, credentials, or approval bypass.
One owned child per workspace. CLI completion is not independent validation.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {"starting", "running"}
TERMINAL = {"completed", "failed", "needs_attention", "timed_out", "cancelled"}
CORE = ("AGENTS.md", "CLAUDE.md", "HANDOFF.md", "plan.md", "discoveries.md")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        for attempt in range(8):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                # Windows readers can briefly deny replacing an open file.
                if attempt == 7:
                    raise
                time.sleep(min(0.02 * 2 ** attempt, 0.25))
    finally:
        temp.unlink(missing_ok=True)


def home(root):
    return root / ".local" / "agent-sessions"


def job_path(root, job_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", job_id):
        raise ValueError("Invalid job ID")
    base = home(root).resolve()
    path = (base / job_id).resolve()
    if path.parent != base:
        raise ValueError("Job directory escaped agent-sessions")
    return path


def input_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Task file must exist inside this workspace")
    if path.stat().st_size > 64 * 1024:
        raise ValueError("Task file exceeds 64 KiB; link to evidence instead")
    return path


def executable(provider):
    result = shutil.which(provider)
    if not result:
        raise ValueError(f"{provider} CLI not found on PATH; no automatic installation")
    if os.name == "nt" and Path(result).suffix.lower() not in {".exe", ".com"}:
        raise ValueError("Use a native CLI executable on Windows, not a shell shim")
    return str(Path(result).resolve())


def command(provider, exe, root, role, write=False):
    if write and role != "implementer":
        raise ValueError("Only an implementer can request --write")
    if provider == "codex":
        return [exe, "exec", "--json", "--color", "never", "--skip-git-repo-check",
                "--sandbox", "workspace-write" if write else "read-only",
                "-c", 'approval_policy="never"', "-C", str(root), "-"]
    tools = "Read,Glob,Grep,Edit,Write" if write else "Read,Glob,Grep"
    return [exe, "--print", "--output-format", "stream-json", "--verbose",
            "--permission-mode", "dontAsk", "--tools", tools, "--allowedTools", tools,
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--max-turns", "16"]


def prompt(root, role, task, write):
    return (
        f"You are a delegated {role} in {root}. This is a fresh, bounded child session.\n"
        "First read AGENTS.md and HANDOFF.md, then the relevant plan.md entries and "
        "the summary/index of discoveries.md. Read only relevant lesson IDs and linked "
        "evidence. Follow AGENTS.md and the task scope below; do not resume unrelated research.\n"
        "The parent owns central records. Do not edit AGENTS.md, CLAUDE.md, HANDOFF.md, "
        "plan.md, discoveries.md, STATE, handoff indexes or other agents' files. Return "
        "proposed central updates in your final answer; the parent records them.\n"
        "Do not spawn agents, invoke another CLI session, submit/upload to Kaggle, "
        "start training/downloads, modify credentials/settings, or make paid/public actions. "
        "Do not inspect credential files or print secrets. Do not restart the application. "
        "Reuse completed evidence.\n"
        + ("Implementation: edit only files explicitly assigned by the task. Command access "
           "depends on the provider permissions; report denied work without bypassing them.\n"
           if write else "Read-only task: do not modify any workspace files or run experiments.\n")
        + "Your final answer must name files/evidence, actual checks, findings and unverified "
        "items. Do not claim independent reproduction without performing it.\n\n"
        "TASK FROM THE PARENT:\n" + task
    )


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a safe existence probe on Windows.
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied is not proof of death.
        try:
            code = ctypes.c_uint32()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def release(root, job_id):
    lock = home(root) / "active.lock"
    if lock.exists() and read_json(lock).get("job_id") == job_id:
        lock.unlink()


def parse_events(provider, path):
    final, session_id, terminal, denials = "", None, None, []
    malformed = 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(event, dict):
            malformed += 1
            continue
        if provider == "codex":
            if event.get("type") == "thread.started":
                session_id = event.get("thread_id")
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if not isinstance(item, dict):
                    malformed += 1
                elif item.get("type") == "agent_message":
                    if isinstance(item.get("text"), str):
                        final = item["text"]
                    else:
                        malformed += 1
            if event.get("type") in {"turn.completed", "turn.failed"}:
                terminal = event["type"]
        else:
            session_id = event.get("session_id") or session_id
            # Preserve useful partial findings on timeout/turn-limit failures.
            if event.get("type") == "assistant":
                message = event.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    texts = [x["text"] for x in content if isinstance(x, dict)
                             and x.get("type") == "text" and isinstance(x.get("text"), str)]
                    if texts:
                        final = "\n".join(texts)
            if event.get("type") == "result":
                if isinstance(event.get("result"), str):
                    final = event["result"] if event.get("subtype") == "success" else event["result"] or final
                elif event.get("subtype") == "success":
                    malformed += 1
                denials = event.get("permission_denials") or []
                terminal = "failed" if event.get("is_error") or event.get("subtype") != "success" else "completed"
    success = terminal == ("turn.completed" if provider == "codex" else "completed")
    return {"text": final, "native_session_id": session_id, "terminal_event": terminal,
            "success": success and bool(final.strip()) and not malformed,
            "permission_denials": denials, "malformed_lines": malformed}


def stop_child(child):
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        import signal
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            child.kill()
        else:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=5)


def core_hashes(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in CORE if (root / name).is_file()}


def worker(root, job_id):
    folder = job_path(root, job_id)
    record = read_json(folder / "job.json")
    if record["state"] != "starting":
        raise ValueError("Job is not in starting state")
    child = None
    record.update(state="running", worker_pid=os.getpid(), started_at=now())
    try:
        write_json(folder / "job.json", record)
        env = dict(os.environ, KNEE_AGENT_CHILD="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with (folder / "prompt.txt").open("rb") as inp, (folder / "events.jsonl").open("wb") as out, (folder / "stderr.log").open("wb") as err:
            child = subprocess.Popen(record["command"], cwd=root, env=env, stdin=inp,
                                     stdout=out, stderr=err, shell=False, creationflags=flags,
                                     start_new_session=os.name != "nt")
            record["child_pid"] = child.pid
            write_json(folder / "job.json", record)
            deadline = time.monotonic() + record["timeout_seconds"]
            forced = None
            while child.poll() is None:
                if (folder / "cancel.request").exists():
                    forced = "cancelled"
                elif time.monotonic() >= deadline:
                    forced = "timed_out"
                if forced:
                    stop_child(child)
                    break
                time.sleep(0.5)
        parsed = parse_events(record["provider"], folder / "events.jsonl")
        (folder / "result.txt").write_text(parsed.pop("text"), encoding="utf-8")
        state = forced or ("completed" if child.returncode == 0 and parsed["success"] else "failed")
        after_hashes = core_hashes(root)
        changed = [key for key in record["core_hashes"].keys() | after_hashes.keys()
                   if record["core_hashes"].get(key) != after_hashes.get(key)]
        if state == "completed" and (parsed["permission_denials"] or changed):
            state = "needs_attention"
        record.update(state=state, exit_code=child.returncode, **parsed,
                      central_files_changed_during_run=changed)
    except Exception as exc:
        record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        if child is not None:
            try:
                stop_child(child)
            except Exception as cleanup_error:
                record["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
    finally:
        record["finished_at"] = now()
        try:
            write_json(folder / "job.json", record)
        finally:
            # A persistence failure must not strand the lock after child exit;
            # a live child must keep the lock even if cleanup itself failed.
            if child is None or child.poll() is not None:
                release(root, job_id)
    return record


def start(root, provider, role, task_file, write=False, timeout=600, dry_run=False):
    if os.environ.get("KNEE_AGENT_CHILD"):
        raise ValueError("Nested dispatch is disabled; return findings to the parent")
    if not 10 <= timeout <= 3600:
        raise ValueError("Timeout must be 10..3600 seconds")
    path = input_path(root, task_file)
    missing = [name for name in CORE if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing workspace entry documents: {missing}")
    argv = command(provider, executable(provider), root, role, write)
    if dry_run:
        return {"dry_run": True, "command": argv, "cwd": str(root), "role": role,
                "task_file": str(path), "timeout_seconds": timeout, "model": "provider-configured"}
    base = home(root)
    base.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    lock = base / "active.lock"
    try:
        with lock.open("x", encoding="utf-8") as f:
            json.dump({"job_id": job_id, "dispatcher_pid": os.getpid(), "created_at": now()}, f)
    except FileExistsError:
        raise ValueError("An owned session is active or interrupted; inspect status/recover first") from None
    folder = job_path(root, job_id)
    proc = None
    try:
        folder.mkdir()
        task = path.read_text(encoding="utf-8-sig")
        full_prompt = prompt(root, role, task, write)
        (folder / "prompt.txt").write_text(full_prompt, encoding="utf-8")
        record = {"id": job_id, "provider": provider, "role": role, "write": write,
                  "root": str(root), "state": "starting", "created_at": now(),
                  "task_file": path.relative_to(root).as_posix(), "task_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "prompt_sha256": hashlib.sha256(full_prompt.encode()).hexdigest(),
                  "command": argv, "timeout_seconds": timeout, "core_hashes": core_hashes(root),
                  "dispatcher_pid": os.getpid(), "model": "provider-configured"}
        write_json(folder / "job.json", record)
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with (folder / "worker.log").open("wb") as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                                     "_worker", job_id], cwd=root, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=log, creationflags=flags,
                                    start_new_session=os.name != "nt")
        # Separate immutable launch file avoids racing worker updates to job.json.
        write_json(folder / "launch.json", {"worker_pid": proc.pid, "launched_at": now()})
        return {"id": job_id, "state": "starting", "worker_pid": proc.pid,
                "folder": str(folder), "timeout_seconds": timeout}
    except Exception as exc:
        if proc is None:
            release(root, job_id)
            raise
        raise ValueError(f"Worker launched for {job_id} (PID {proc.pid}), but launch metadata "
                         f"failed: {exc}. Lock retained; inspect status before recovery.") from exc


def status(root, job_id):
    folder = job_path(root, job_id)
    record = read_json(folder / "job.json")
    if record["state"] in ACTIVE:
        launch = read_json(folder / "launch.json") if (folder / "launch.json").exists() else {}
        pid = record.get("worker_pid") or launch.get("worker_pid")
        record["worker_alive"] = alive(pid) if pid else None
        if pid and not record["worker_alive"]:
            record["observed_state"] = "interrupted; verify child before recover"
    record["folder"] = str(folder)
    return record


def recover(root, job_id):
    folder = job_path(root, job_id)
    record = status(root, job_id)
    lock = home(root) / "active.lock"
    if not lock.exists() or read_json(lock).get("job_id") != job_id:
        raise ValueError("No matching owned lock")
    launch = read_json(folder / "launch.json") if (folder / "launch.json").exists() else {}
    pids = [record.get("worker_pid"), launch.get("worker_pid"), record.get("child_pid"), record.get("dispatcher_pid")]
    if any(alive(pid) for pid in pids if pid):
        raise ValueError("Recorded process still exists; cannot safely recover this lock")
    record.update(state="failed", finished_at=now(), recovery="interrupted job; no automatic retry")
    write_json(folder / "job.json", record)
    release(root, job_id)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor")
    sub.add_parser("list")
    p = sub.add_parser("start")
    p.add_argument("--provider", choices=["codex", "claude"], required=True)
    p.add_argument("--role", choices=["researcher", "reviewer", "implementer"], required=True)
    p.add_argument("--task-file", required=True)
    p.add_argument("--write", action="store_true")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--dry-run", action="store_true")
    for name in ["status", "result", "cancel", "recover", "_worker"]:
        p = sub.add_parser(name)
        p.add_argument("id")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "doctor":
        result = {"root": str(root), "nested_dispatch": bool(os.environ.get("KNEE_AGENT_CHILD")),
                  "claude_parent_context": bool(os.environ.get("CLAUDECODE")), "providers": {}}
        for provider in ["codex", "claude"]:
            try:
                exe = executable(provider)
                r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                result["providers"][provider] = {"path": exe, "version": r.stdout.strip(), "exit_code": r.returncode}
            except (ValueError, subprocess.TimeoutExpired) as exc:
                result["providers"][provider] = {"error": str(exc)}
    elif args.action == "start":
        result = start(root, args.provider, args.role, args.task_file, args.write, args.timeout, args.dry_run)
    elif args.action == "list":
        result = [status(root, p.parent.name) for p in sorted(home(root).glob("*/job.json"))]
    elif args.action == "result":
        record = status(root, args.id)
        if record["state"] in ACTIVE:
            raise ValueError("Job has no terminal status yet")
        path = job_path(root, args.id) / "result.txt"
        print(path.read_text(encoding="utf-8") if path.exists() else "No final response; inspect stderr.log and job.json")
        return 0 if record["state"] == "completed" else 1
    elif args.action == "cancel":
        record = status(root, args.id)
        if record["state"] not in ACTIVE:
            raise ValueError("Job is already terminal")
        (job_path(root, args.id) / "cancel.request").write_text(now(), encoding="utf-8")
        result = {"id": args.id, "cancel_requested": True, "note": "worker stops only its own child"}
    elif args.action == "recover":
        result = recover(root, args.id)
    elif args.action == "_worker":
        result = worker(root, args.id)
    else:
        result = status(root, args.id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)
