"""Subprocess failure/cancellation/permission boundaries without calling paid CLIs."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("agent_session", Path(__file__).resolve().parents[1] / "tools/agent_session.py")
agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent)


def events(tmp_path, rows):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows), encoding="utf-8")
    return path


def seed_job(root, code, timeout=10):
    for name in agent.CORE:
        (root / name).write_text("keep", encoding="utf-8")
    job_id = "test-job"
    folder = agent.job_path(root, job_id)
    folder.mkdir(parents=True)
    (folder / "prompt.txt").write_text("task", encoding="utf-8")
    script = root / "fake_cli.py"
    script.write_text(code, encoding="utf-8")
    agent.write_json(folder / "job.json", {"id": job_id, "state": "starting", "provider": "codex",
        "command": [sys.executable, str(script)], "timeout_seconds": timeout, "core_hashes": agent.core_hashes(root)})
    agent.write_json(agent.home(root) / "active.lock", {"job_id": job_id})
    return job_id, folder


GOOD = '''import json
print(json.dumps({"type":"thread.started","thread_id":"fake-session"}))
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"checked"}}))
print(json.dumps({"type":"turn.completed"}))
'''


def test_codex_exit_zero_without_terminal_is_not_success(tmp_path):
    path = events(tmp_path, [{"type": "item.completed", "item": {"type": "agent_message", "text": "looks done"}}])
    assert not agent.parse_events("codex", path)["success"]


def test_claude_permission_denial_preserved(tmp_path):
    path = events(tmp_path, [{"type": "result", "subtype": "success", "is_error": False,
        "result": "could not inspect", "session_id": "abc", "permission_denials": [{"tool_name": "Read"}]}])
    parsed = agent.parse_events("claude", path)
    assert parsed["permission_denials"] and parsed["native_session_id"] == "abc"


def test_bad_json_and_failed_turn_not_success(tmp_path):
    path = events(tmp_path, ["not json", {"type": "turn.completed"}, {"type": "turn.failed"}])
    parsed = agent.parse_events("codex", path)
    assert not parsed["success"] and parsed["malformed_lines"] == 1


def test_real_process_result_and_lock_release(tmp_path):
    job_id, folder = seed_job(tmp_path, GOOD)
    result = agent.worker(tmp_path, job_id)
    assert result["state"] == "completed" and result["native_session_id"] == "fake-session"
    assert (folder / "result.txt").read_text() == "checked"
    assert not (agent.home(tmp_path) / "active.lock").exists()


def test_exit_zero_incomplete_process_is_failed(tmp_path):
    job_id, _ = seed_job(tmp_path, "print('no result')")
    assert agent.worker(tmp_path, job_id)["state"] == "failed"


def test_changed_or_deleted_central_file_needs_attention(tmp_path):
    job_id, _ = seed_job(tmp_path, "from pathlib import Path\nPath('HANDOFF.md').unlink()\n" + GOOD)
    result = agent.worker(tmp_path, job_id)
    assert result["state"] == "needs_attention"
    assert result["central_files_changed_during_run"] == ["HANDOFF.md"]


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_stop_owned_child(tmp_path, cancel):
    job_id, folder = seed_job(tmp_path, "import time\ntime.sleep(60)", timeout=0.05)
    if cancel:
        (folder / "cancel.request").write_text("cancel")
    result = agent.worker(tmp_path, job_id)
    assert result["state"] == ("cancelled" if cancel else "timed_out")
    assert not agent.alive(result["child_pid"])
    assert not (agent.home(tmp_path) / "active.lock").exists()


def test_reviewer_cannot_enable_writes(tmp_path):
    with pytest.raises(ValueError, match="implementer"):
        agent.command("codex", "codex", tmp_path, "reviewer", True)
    codex = agent.command("codex", "codex", tmp_path, "reviewer")
    claude = agent.command("claude", "claude", tmp_path, "reviewer")
    assert "read-only" in codex and not any("bypass" in x for x in codex)
    assert claude[claude.index("--tools") + 1] == "Read,Glob,Grep"
    assert "--model" not in codex + claude


def test_escape_and_nested_dispatch_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        agent.job_path(tmp_path, "../outside")
    with pytest.raises(ValueError):
        agent.input_path(tmp_path, "../missing")
    monkeypatch.setenv("KNEE_AGENT_CHILD", "1")
    with pytest.raises(ValueError, match="Nested"):
        agent.start(tmp_path, "codex", "reviewer", "missing")


def test_live_lock_not_recovered(tmp_path):
    job_id, folder = seed_job(tmp_path, GOOD)
    record = agent.read_json(folder / "job.json")
    record["worker_pid"] = os.getpid()
    agent.write_json(folder / "job.json", record)
    assert agent.alive(os.getpid())
    with pytest.raises(ValueError, match="still exists"):
        agent.recover(tmp_path, job_id)


def test_second_start_rejected_before_provider_launch(tmp_path, monkeypatch):
    seed_job(tmp_path, GOOD)
    (tmp_path / "task.txt").write_text("check")
    monkeypatch.delenv("KNEE_AGENT_CHILD", raising=False)
    monkeypatch.setattr(agent, "executable", lambda _: sys.executable)
    with pytest.raises(ValueError, match="active or interrupted"):
        agent.start(tmp_path, "codex", "reviewer", "task.txt")


def test_dry_run_has_no_state_and_uses_argument_list(tmp_path, monkeypatch):
    for name in agent.CORE:
        (tmp_path / name).write_text("keep")
    task = tmp_path / "task with spaces.txt"
    task.write_text("literal $(do-not-run) and `quotes`")
    monkeypatch.delenv("KNEE_AGENT_CHILD", raising=False)
    monkeypatch.setattr(agent, "executable", lambda _: sys.executable)
    result = agent.start(tmp_path, "codex", "reviewer", task.name, dry_run=True)
    assert result["command"][-1] == "-"
    assert not agent.home(tmp_path).exists()


def test_null_provider_fields_fail_without_losing_metadata(tmp_path):
    path = events(tmp_path, [{"type": "thread.started", "thread_id": "retained"},
        {"type": "item.completed", "item": None},
        {"type": "item.completed", "item": {"type": "agent_message", "text": None}},
        {"type": "turn.completed"}])
    parsed = agent.parse_events("codex", path)
    assert not parsed["success"] and parsed["native_session_id"] == "retained"
    path = events(tmp_path, [{"type": "result", "subtype": "success", "result": None,
        "session_id": "retained", "permission_denials": [{"tool_name": "Read"}]}])
    parsed = agent.parse_events("claude", path)
    assert not parsed["success"] and parsed["permission_denials"]


def test_claude_turn_limit_preserves_partial_response(tmp_path):
    path = events(tmp_path, [{"type": "assistant", "message": {"content": [
        {"type": "text", "text": "partial finding"}]}},
        {"type": "result", "subtype": "error_max_turns", "is_error": True}])
    parsed = agent.parse_events("claude", path)
    assert not parsed["success"] and parsed["text"] == "partial finding"


def test_claude_worker_permissions_and_prompt_delivery(tmp_path):
    code = '''import json, os, sys
assert sys.stdin.read() == "task"
assert os.environ["KNEE_AGENT_CHILD"] == "1"
print(json.dumps({"type":"result","subtype":"success","result":"reviewed",
                  "session_id":"claude-fake","permission_denials":[{"tool_name":"Read"}]}))
'''
    job_id, folder = seed_job(tmp_path, code)
    record = agent.read_json(folder / "job.json")
    record["provider"] = "claude"
    agent.write_json(folder / "job.json", record)
    result = agent.worker(tmp_path, job_id)
    assert result["state"] == "needs_attention" and result["native_session_id"] == "claude-fake"
    assert (folder / "result.txt").read_text() == "reviewed"


def test_missing_final_message_is_not_success(tmp_path):
    path = events(tmp_path, [{"type": "turn.completed"}])
    assert not agent.parse_events("codex", path)["success"]
    path = events(tmp_path, [{"type": "assistant", "message": {"content": [
        {"type": "text", "text": "working on it"}]}},
        {"type": "result", "subtype": "success", "result": ""}])
    assert not agent.parse_events("claude", path)["success"]


@pytest.mark.skipif(os.name != "nt", reason="Windows open file replacement semantics")
def test_windows_open_reader_does_not_strand_atomic_write(tmp_path):
    import threading
    import time
    path = tmp_path / "job.json"
    agent.write_json(path, {"state": "starting"})
    reader = path.open("rb")
    # Reproduce the OS rejection first, then verify the bounded retry survives it.
    other = tmp_path / "probe.json"
    other.write_text("{}")
    with pytest.raises(PermissionError):
        os.replace(other, path)
    def close_reader():
        time.sleep(0.08)
        reader.close()
    thread = threading.Thread(target=close_reader)
    thread.start()
    try:
        agent.write_json(path, {"state": "completed"})
    finally:
        thread.join()
    assert agent.read_json(path)["state"] == "completed"


@pytest.mark.parametrize("fail_call", [1, 3])
def test_worker_persistence_failure_releases_finished_lock(tmp_path, monkeypatch, fail_call):
    job_id, folder = seed_job(tmp_path, GOOD)
    original = agent.write_json
    calls = 0
    def fail_write(path, obj):
        nonlocal calls
        calls += 1
        if calls == fail_call:
            raise PermissionError("injected persistence failure")
        return original(path, obj)
    monkeypatch.setattr(agent, "write_json", fail_write)
    if fail_call == 3:
        with pytest.raises(PermissionError):
            agent.worker(tmp_path, job_id)
    else:
        assert agent.worker(tmp_path, job_id)["state"] == "failed"
    assert not (agent.home(tmp_path) / "active.lock").exists()


def test_launch_metadata_failure_keeps_running_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace
    for name in agent.CORE:
        (tmp_path / name).write_text("keep")
    (tmp_path / "task.txt").write_text("check")
    monkeypatch.delenv("KNEE_AGENT_CHILD", raising=False)
    monkeypatch.setattr(agent, "executable", lambda _: sys.executable)
    monkeypatch.setattr(agent.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=os.getpid()))
    original = agent.write_json
    def fail_launch(path, obj):
        if path.name == "launch.json":
            raise PermissionError("injected launch failure")
        return original(path, obj)
    monkeypatch.setattr(agent, "write_json", fail_launch)
    with pytest.raises(ValueError, match="Lock retained"):
        agent.start(tmp_path, "codex", "reviewer", "task.txt")
    assert (agent.home(tmp_path) / "active.lock").exists()


def test_recover_dead_job_releases_lock_without_retry(tmp_path):
    job_id, folder = seed_job(tmp_path, GOOD)
    result = agent.recover(tmp_path, job_id)
    assert result["state"] == "failed" and "no automatic retry" in result["recovery"]
    assert not (agent.home(tmp_path) / "active.lock").exists()
