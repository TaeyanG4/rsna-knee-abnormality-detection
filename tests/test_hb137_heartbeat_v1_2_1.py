"""Windows sharing races, exhausted retries, and concurrent real file traffic."""
import json
import os
import threading
import time
from pathlib import Path

import pytest

from tools import hb137_safety_v1_2_1 as s


@pytest.mark.parametrize("failures", [1, 4, 7])
def test_replace_permission_backoff_is_bounded(tmp_path, monkeypatch, failures):
    path = tmp_path / "progress.json"
    s.atomic(path, {"progress_ns": 1})
    original = os.replace
    calls, sleeps = [], []
    def replace(src, dst):
        calls.append(1)
        if len(calls) <= failures:
            raise PermissionError("reader owns file")
        return original(src, dst)
    monkeypatch.setattr(s.os, "replace", replace)
    monkeypatch.setattr(s.time, "sleep", sleeps.append)
    if failures == 7:
        with pytest.raises(PermissionError):
            s.atomic(path, {"progress_ns": 2})
        assert s.read_heartbeat(path)["progress_ns"] == 1
    else:
        s.atomic(path, {"progress_ns": 2})
        assert s.read_heartbeat(path)["progress_ns"] == 2
    assert sleeps == list(s.REPLACE_DELAYS[:min(failures, 6)])
    assert len(calls) == min(failures + 1, 7)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("error", [PermissionError, OSError, TypeError])
def test_failed_heartbeat_warns_and_training_io_continues(tmp_path, monkeypatch, caplog, error):
    import io
    path = tmp_path / "progress.json"
    monkeypatch.setenv("HB137_PROGRESS", str(path))
    def fail(*a, **kw):
        raise error("publication failed")
    monkeypatch.setattr(s, "atomic", fail)
    stream = io.BytesIO()
    assert s.ProgressIO(stream, "checkpoint_write").write(b"completed") == 9
    assert stream.getvalue() == b"completed"
    assert "Heartbeat write failed; continuing" in caplog.text
    assert not path.exists()


@pytest.mark.parametrize("failure", ["permission", "json"])
@pytest.mark.parametrize("recover", [True, False])
def test_read_retries_without_fabricating_progress(tmp_path, monkeypatch, failure, recover):
    path = tmp_path / "progress.json"
    calls, sleeps = [], []
    def read(*a, **kw):
        calls.append(1)
        if recover and len(calls) == 3:
            return '{"progress_ns": 2}'
        if failure == "permission":
            raise PermissionError("replace in flight")
        return '{"progress_ns":'
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(s.time, "sleep", sleeps.append)
    note = s.read_heartbeat(path)
    assert note == ({"progress_ns": 2} if recover else {})
    assert len(calls) == (3 if recover else 5)
    assert sleeps == list(s.READ_DELAYS[:2] if recover else s.READ_DELAYS)
    w = s.Watchdog(0)
    assert w.check(900, note.get("progress_ns")) == (None if recover else "NO_STEP_PROGRESS")


def test_reader_closes_before_decode_and_retry(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text("{}")
    opened = []
    original_open, original_loads = Path.open, json.loads
    def tracked_open(*a, **kw):
        stream = original_open(*a, **kw)
        opened.append(stream)
        return stream
    def decode(text):
        assert opened and all(f.closed for f in opened)
        if len(opened) < 3:
            raise json.JSONDecodeError("partial", text, 0)
        return original_loads(text)
    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(s.json, "loads", decode)
    assert s.read_heartbeat(path) == {}
    assert len(opened) == 3


@pytest.mark.skipif(os.name != "nt", reason="30-second Windows sharing regression")
def test_windows_writer_50ms_reader_10ms_for_30_seconds(tmp_path, monkeypatch, caplog):
    path = tmp_path / "progress.json"
    monkeypatch.setenv("HB137_PROGRESS", str(path))
    s.heartbeat("start")
    barrier = threading.Barrier(3)
    errors, reads, writes, observed = [], [], [], []
    def worker(interval, writer):
        try:
            barrier.wait()
            start = time.monotonic()
            deadline = start + 30
            due = start
            while time.monotonic() < deadline:
                if writer:
                    s.heartbeat("training_step", completed=len(writes))
                    writes.append(time.monotonic())
                else:
                    note = s.read_heartbeat(path)
                    reads.append(time.monotonic())
                    if note:
                        observed.append(note["progress_ns"])
                due += interval
                time.sleep(max(0, due - time.monotonic()))
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=worker, args=(.05, True)),
               threading.Thread(target=worker, args=(.01, False))]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(35)
    assert all(not t.is_alive() for t in threads)
    assert not errors
    assert len(writes) >= 550 and len(reads) >= 2700
    assert len(set(observed)) >= 500 and observed == sorted(observed)
    assert writes[-1] - writes[0] >= 29.5
    assert "Heartbeat write failed" not in caplog.text
    assert "Heartbeat read unavailable" not in caplog.text
    assert not list(tmp_path.glob("*.tmp"))


def test_watchdog_survives_unreadable_heartbeat_until_idle(tmp_path, monkeypatch):
    import sys
    from tools.hb137_screen_v1 import life
    original = Path.read_text
    def denied(path, *args, **kwargs):
        if path.name == "progress.json":
            raise PermissionError("monitor race")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", denied)
    ledger = tmp_path / "budget.json"
    with pytest.raises(life.ContractError, match="child failed"):
        s.run([sys.executable, "-c", "import time; time.sleep(30)"], ledger,
              idle=.3, poll=.05, sample=lambda: None)
    doc = json.loads(ledger.read_text())
    assert not doc["running"]
    assert doc["attempts"][-1]["reason"] == "NO_STEP_PROGRESS"
