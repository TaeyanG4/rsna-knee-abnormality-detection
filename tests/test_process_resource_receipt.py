"""CPU synthetic lifecycle tests; no producer/model imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import psutil
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "process_resource_receipt.py"
spec = importlib.util.spec_from_file_location("resource_receipt", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def run(tmp_path, code, *, timeout=5, args=()):
    argv = [sys.executable, "-c", code, *args]
    result = module.supervise(argv, tmp_path / "run", timeout)
    assert json.loads((tmp_path / "run" / "receipt.json").read_text()) == result
    assert result["argv"] == argv
    assert result["retries"] == 0
    assert result["cleanup"]["survivors"] == []
    return result


def test_success_exact_argv_and_no_overwrite(tmp_path):
    args = ["space value", "--", 'literal"quote', "a\\b", ""]
    result = run(tmp_path, "import sys,json,time; print(json.dumps(sys.argv[1:])); time.sleep(.22)", args=args)
    assert result["status"] == "COMPLETE"
    assert result["child_exit_code"] == result["wrapper_exit_code"] == 0
    assert result["sample_count"] >= 2
    assert result["sampled_tree_rss_max_bytes"] > 0
    assert json.loads((tmp_path / "run" / "stdout.log").read_text()) == args
    receipt = (tmp_path / "run" / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.supervise([sys.executable, "-c", "raise Exception('must not run')"], tmp_path / "run", 5)
    assert (tmp_path / "run" / "receipt.json").read_bytes() == receipt


def test_nonzero_exit(tmp_path):
    result = run(tmp_path, "import time,sys; time.sleep(.15); sys.exit(7)")
    assert result["status"] == "FAILED"
    assert result["child_exit_code"] == 7
    assert result["wrapper_exit_code"] == 1
    assert result["failure"]["message"] == "child exited 7"


def test_timeout_owns_only_child(tmp_path):
    # An unrelated process must not be touched by owned-tree cleanup.
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = run(tmp_path, "import time; time.sleep(30)", timeout=.25)
        assert result["status"] == "FAILED"
        assert result["failure"]["type"] == "TimeoutError"
        assert result["cleanup"]["actions"]
        assert sentinel.poll() is None
        assert sentinel.pid not in [x["pid"] for x in result["observed_identities"]]
    finally:
        sentinel.terminate(); sentinel.wait(timeout=3)


@pytest.mark.parametrize("parent_exit", [0, 8])
def test_surviving_descendant_is_reaped(tmp_path, parent_exit):
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(.4); sys.exit(%d)" % parent_exit
    result = run(tmp_path, code)
    child_pid = int((tmp_path / "run" / "stdout.log").read_text())
    assert result["status"] == "FAILED"
    assert child_pid in [x["pid"] for x in result["observed_identities"]]
    assert child_pid in [x["pid"] for x in result["cleanup"]["actions"]]
    assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE


def test_reused_pid_is_never_signalled(monkeypatch):
    class FakeProcess:
        pid = 13
        def create_time(self): return 999.0
        def terminate(self): raise AssertionError("reused PID signalled")
        def kill(self): raise AssertionError("reused PID signalled")
    monkeypatch.setattr(module, "INTERVAL", .1)
    monkeypatch.setattr(psutil, "Process", lambda pid: FakeProcess())
    tree = module.OwnedTree(psutil)
    tree.known.add((13, 1.0))
    result = tree.cleanup()
    assert result["actions"] == result["survivors"] == result["errors"] == []
    assert tree.reused_pid_observations > 0


@pytest.mark.parametrize("membership,root,mount,expected", [
    ("/a/b", "/", "/sys/fs/cgroup", "/sys/fs/cgroup/a/b"),
    ("/a/b", "/a", "/sys/fs/cgroup", "/sys/fs/cgroup/b"),
    ("/", "/host/container", "/sys/fs/cgroup", "/sys/fs/cgroup"),
    ("/a", "/", "/cgroup\\040space", "/cgroup space/a"),
])
def test_cgroup_v2_resolution(membership, root, mount, expected):
    result = module.resolve_cgroup(f"0::{membership}\n", f"12 10 0:1 {root} {mount} rw - cgroup2 cgroup rw\n")
    assert result["path"] == expected
    assert result["version"] == 2


def test_cgroup_v1_resolution_and_traversal_rejected():
    mount = "12 10 0:1 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
    assert module.resolve_cgroup("2:memory:/foo", mount)["path"] == "/sys/fs/cgroup/memory/foo"
    with pytest.raises(RuntimeError):
        module.resolve_cgroup("2:memory:/../elsewhere", mount)


def test_cgroup_snapshot_reads_never_resets(tmp_path):
    values = {"memory.current": "123", "memory.peak": "456", "memory.max": "max", "memory.events": "oom 0\noom_kill 0\n"}
    for name, value in values.items(): (tmp_path / name).write_text(value)
    snap = module.cgroup_snapshot({"path": str(tmp_path), "version": 2})
    assert snap["values"] == {"memory.current": 123, "memory.peak": 456, "memory.max": "max", "memory.events": {"oom": 0, "oom_kill": 0}}
    assert snap["errors"] == {}
    for name, value in values.items(): assert (tmp_path / name).read_text() == value


def test_access_failure_is_fail_fast_and_receipted(tmp_path, monkeypatch):
    def fail(self): raise psutil.AccessDenied(123)
    monkeypatch.setattr(module.OwnedTree, "sample", fail)
    result = run(tmp_path, "import time; time.sleep(30)")
    assert result["status"] == "FAILED"
    assert result["sample_attempts"] == 1
    assert result["sample_count"] == 0
    assert len(result["access_failures"]) == 1


def test_cli_keeps_literal_separator(tmp_path):
    out = tmp_path / "cli"
    command = [sys.executable, str(SOURCE), "--out", str(out), "--timeout", "5", "--", sys.executable, "-c", "import sys,json; print(json.dumps(sys.argv[1:]))", "--", "x y"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=8)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((out / "stdout.log").read_text()) == ["--", "x y"]


def test_timeout_cleans_recursive_grandchild(tmp_path):
    middle = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)"
    code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{middle!r}]); time.sleep(30)"
    result = run(tmp_path, code, timeout=.7)
    assert result["failure"]["type"] == "TimeoutError"
    assert len(result["observed_identities"]) == 3
    samples = [json.loads(line) for line in (tmp_path / "run" / "samples.jsonl").read_text().splitlines()]
    assert max(len(s["processes"]) for s in samples) == 3
    assert all(s["rss_bytes"] == sum(p["rss"] for p in s["processes"]) for s in samples)


def test_missing_cgroup_files_explicit(tmp_path):
    result = module.cgroup_snapshot({"path": str(tmp_path), "version": 2})
    assert set(result["errors"]) == set(module.CGROUP_FILES)
    assert result["values"] == {}


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_launches_nothing(tmp_path, timeout):
    with pytest.raises(ValueError):
        module.supervise([sys.executable, "-c", "raise Exception('must not run')"], tmp_path / "run", timeout)
    assert not (tmp_path / "run").exists()
