"""CPU subprocess checks for the unattended v1.1.2 monitor; no real data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil
import pytest

from tools import hb145_r7_monitor_v111 as monitor


def run(tmp_path, script="import time; time.sleep(30)", **kwargs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    return monitor.monitored_run([sys.executable, "-c", script], env,
        tmp_path / "monitor.log", tmp_path / "resources.json",
        poll_seconds=0.02, **kwargs)


def status(tmp_path):
    return json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))


def assert_dead(pid):
    try:
        assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_gpu_telemetry_timeout_is_explicit_and_caught(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == monitor.TELEMETRY_TIMEOUT_SECONDS
        assert kwargs["check"] is True
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(monitor.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        monitor._gpu_sample({})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    code, receipt = monitor.monitored_run(
        [sys.executable, "-c", "import time; time.sleep(.3)"], env,
        tmp_path / "monitor.log", tmp_path / "resources.json", poll_seconds=.02)
    assert code == 0
    assert any("TimeoutExpired" in error for error in receipt["telemetry_errors"])
    assert receipt["cleanup_complete"]


@pytest.mark.parametrize("exception", [RuntimeError("injected tick"), KeyboardInterrupt()])
def test_tick_exception_always_cleans_owned_tree(monkeypatch, tmp_path, exception):
    pid_file = tmp_path / "descendants.json"
    script = ("import subprocess,sys,time,json,os; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
              f"open({str(pid_file)!r},'w').write(json.dumps([os.getpid(),p.pid])); "
              "time.sleep(30)")
    def tick():
        if pid_file.exists():
            raise exception
    code, receipt = run(tmp_path, script, on_tick=tick)
    assert code != 0
    assert receipt["state"] == "MONITOR_ERROR"
    assert receipt["cleanup_complete"], receipt["cleanup_errors"]
    for pid in json.loads(pid_file.read_text()):
        assert_dead(pid)
    assert status(tmp_path)["cleanup_complete"]


def test_ram_guard_does_not_wait_for_stuck_telemetry(monkeypatch, tmp_path):
    entered, unblock = threading.Event(), threading.Event()
    def stuck(env):
        entered.set()
        unblock.wait(8)
        return [0, 0]
    monkeypatch.setattr(monitor, "_gpu_sample", stuck)
    monkeypatch.setattr(monitor, "RAM_LIMIT_BYTES", 1)
    started = time.perf_counter()
    try:
        code, receipt = run(tmp_path)
        assert time.perf_counter() - started < 4
        assert code != 0 and receipt["ram_guard_stopped"]
        assert receipt["peak_rss_bytes"] > 0
        assert receipt["cleanup_complete"]
    finally:
        unblock.set()


def test_resume_stalls_without_treating_old_journals_as_progress(tmp_path):
    journal = tmp_path / "journal"
    journal.mkdir()
    marker = journal / "old.json"
    marker.write_text("{}")
    old = time.time() - 3600
    os.utime(marker, (old, old))
    code, receipt = run(tmp_path, stall_seconds=.15)
    state = status(tmp_path)
    assert code != 0 and receipt["state"] == "STALL"
    assert state["completed_count"] == state["initial_completed_count"] == 1
    assert state["new_completed_count"] == 0
    assert state["last_commit_unix"] == pytest.approx(old)
    assert state["last_commit_monotonic"] is None
    assert state["heartbeat_count"] > 1 and state["idle_seconds"] >= .15
    assert "STALL:" in (tmp_path / "monitor.log").read_text()
    assert receipt["cleanup_complete"] and marker.exists()


def test_atomic_commits_reset_idle_timer_and_final_count(tmp_path):
    journal = tmp_path / "journal"
    journal.mkdir()
    script = ("import pathlib,time,os; "
              f"p=pathlib.Path({str(journal)!r}); "
              "time.sleep(.08); (p/'one.tmp').write_text('{}'); "
              "os.replace(p/'one.tmp',p/'one.json'); time.sleep(.08); "
              "(p/'two.json').write_text('{}')")
    code, receipt = run(tmp_path, script, stall_seconds=1)
    assert code == 0 and receipt["completed_count"] == 2
    assert status(tmp_path)["new_completed_count"] == 2
    assert status(tmp_path)["last_commit_utc"]
    assert receipt["cleanup_complete"]
    assert not list(tmp_path.glob("*.tmp"))


def test_successful_root_exit_cleans_lingering_grandchild(tmp_path):
    pid_file = tmp_path / "grandchild.json"
    script = ("import subprocess,sys,time,json; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
              f"open({str(pid_file)!r},'w').write(json.dumps(p.pid)); time.sleep(.2)")
    code, receipt = run(tmp_path, script)
    assert code == 0 and receipt["cleanup_complete"], receipt
    assert_dead(json.loads(pid_file.read_text()))


def test_status_and_receipt_exposure_identity_propagation(tmp_path):
    (tmp_path / "hook.json").write_text(json.dumps({"identity": "test-identity", "models": "heldout,f0"}))
    code, receipt = run(tmp_path, "pass")
    assert code == 0
    for record in (receipt, status(tmp_path)):
        assert record["identity"] == "test-identity"
        assert record["clean_oof"] is False and record["exposure"] == "EXPOSED"
        assert set(record["role_provenance"]) == {"heldout_raw", "heldout_post", "f0_raw", "f0_post"}
        assert all(role["exposure"] == "EXPOSED" for role in record["role_provenance"].values())


def test_telemetry_provider_exception_is_recorded(monkeypatch, tmp_path):
    def broken(env):
        raise OSError("telemetry unavailable")
    monkeypatch.setattr(monitor, "_gpu_sample", broken)
    code, receipt = run(tmp_path, "import time; time.sleep(.15)")
    assert code == 0
    assert any("telemetry unavailable" in error for error in receipt["telemetry_errors"])


@pytest.mark.parametrize("failures", [3, 40])
def test_replace_permission_retry_is_bounded(monkeypatch, tmp_path, failures):
    original = monitor.os.replace
    attempts, delays = [], []
    def replace(source, target):
        attempts.append(1)
        if len(attempts) <= failures:
            raise PermissionError(13, "reader holds destination")
        original(source, target)
    monkeypatch.setattr(monitor.os, "replace", replace)
    monkeypatch.setattr(monitor.time, "sleep", delays.append)
    if failures == 40:
        with pytest.raises(PermissionError):
            monitor._atomic_json(tmp_path / "status.json", {"ok": True})
        assert len(attempts) == 40 and len(delays) == 39
    else:
        monitor._atomic_json(tmp_path / "status.json", {"ok": True})
        assert len(attempts) == 4 and status(tmp_path)["ok"]
    assert delays == list(monitor.REPLACE_DELAYS[:len(delays)])


@pytest.mark.skipif(os.name != "nt", reason="Windows reader/replace sharing violation")
def test_windows_status_reader_10ms_writer_50ms_for_30_seconds(tmp_path):
    path = tmp_path / "status.json"
    monitor._atomic_json(path, {"sequence": 0})
    stop, held = threading.Event(), threading.Event()
    errors, reads = [], []
    def reader():
        try:
            # Force the original failure before sustained concurrent readers.
            with path.open(encoding="utf-8") as stream:
                held.set()
                time.sleep(.15)
                json.load(stream)
            while not stop.is_set():
                tick = time.perf_counter()
                try:
                    with path.open(encoding="utf-8") as stream:
                        value = json.load(stream)
                        time.sleep(.002)
                    reads.append(value["sequence"])
                except PermissionError:
                    # Windows may also deny a new read during the rename itself.
                    # Retry at the next 10ms reader tick; malformed JSON still fails.
                    pass
                stop.wait(max(0, .01 - (time.perf_counter() - tick)))
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=reader)
    thread.start()
    assert held.wait(2)
    started, sequence = time.perf_counter(), 0
    try:
        while time.perf_counter() - started < 30:
            tick = time.perf_counter()
            sequence += 1
            monitor._atomic_json(path, {"sequence": sequence})
            time.sleep(max(0, .05 - (time.perf_counter() - tick)))
    finally:
        stop.set()
        thread.join(2)
    assert not thread.is_alive() and not errors
    assert sequence >= 100 and len(reads) >= 1000
    assert status(tmp_path)["sequence"] == sequence


@pytest.mark.parametrize("mode", ["persistent", "final_only", "transient"])
def test_heartbeat_failure_never_suppresses_child_or_receipt(monkeypatch, tmp_path, mode):
    original = monitor._atomic_json
    attempts = []
    def write(path, value):
        if Path(path).name == "status.json":
            attempts.append(value["state"])
            if (mode == "persistent" or
                mode == "final_only" and value["exit_code"] is not None or
                mode == "transient" and len(attempts) == 1):
                raise PermissionError(13, "injected status publication failure")
        original(path, value)
    monkeypatch.setattr(monitor, "_atomic_json", write)
    marker = tmp_path / "child_completed"
    code, receipt = run(tmp_path, "import time,pathlib; time.sleep(.25); "
                        f"pathlib.Path({str(marker)!r}).touch()", stall_seconds=2)
    persisted = json.loads((tmp_path / "resources.json").read_text())
    assert code == 0 and marker.exists() and receipt == persisted
    assert receipt["state"] == "COMPLETED" and receipt["cleanup_complete"]
    assert receipt["status_worker_stopped"]
    health = receipt["heartbeat_health"]
    assert health["error_count"] >= 1 and "PermissionError" in health["last_error"]
    assert "HEARTBEAT_ERROR:" in (tmp_path / "monitor.log").read_text()
    if mode == "transient":
        assert health["consecutive_failures"] == 0 and health["successful_writes"] > 0
    else:
        assert health["consecutive_failures"] > 0


@pytest.mark.parametrize("commits", [False, True])
def test_persistent_heartbeat_failure_surfaces_in_stall(monkeypatch, tmp_path, commits):
    original = monitor._atomic_json
    def write(path, value):
        if Path(path).name == "status.json":
            raise PermissionError(13, "persistent reader")
        original(path, value)
    monkeypatch.setattr(monitor, "_atomic_json", write)
    journal = tmp_path / "journal"
    journal.mkdir()
    def tick():
        if commits:
            (journal / "progress.json").write_text(str(time.time_ns()))
    code, receipt = run(tmp_path, "import time; time.sleep(.5)", on_tick=tick, stall_seconds=.2)
    assert code == (0 if commits else 3) and receipt["cleanup_complete"]
    assert receipt["state"] == ("COMPLETED" if commits else "STALL")
    assert receipt["heartbeat_health"]["error_count"] > 0
    assert receipt["stall_reason"] == (
        None if commits else "no study commits")
    assert ("STALL:" in (tmp_path / "monitor.log").read_text()) is not commits
    assert (tmp_path / "resources.json").exists()


def test_status_retries_do_not_block_ram_guard(monkeypatch, tmp_path):
    original = monitor._atomic_json
    entered, unblock = threading.Event(), threading.Event()
    cleanup_times = []
    cleanup = monitor._cleanup
    def write(path, value):
        if Path(path).name == "status.json":
            entered.set()
            unblock.wait(5)
        original(path, value)
    def clean(*args):
        cleanup_times.append(time.perf_counter())
        unblock.set()
        return cleanup(*args)
    monkeypatch.setattr(monitor, "_atomic_json", write)
    monkeypatch.setattr(monitor, "_cleanup", clean)
    monkeypatch.setattr(monitor, "RAM_LIMIT_BYTES", 1)
    started = time.perf_counter()
    try:
        code, receipt = run(tmp_path)
        assert entered.is_set() and code == 2 and receipt["ram_guard_stopped"]
        assert cleanup_times[0] - started < 2
        assert receipt["cleanup_complete"]
    finally:
        unblock.set()
