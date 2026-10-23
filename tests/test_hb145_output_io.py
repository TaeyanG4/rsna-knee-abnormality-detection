"""v1.1.3 CPU output contention, fatal journal and carry-over contracts."""
import ast
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest
from tools import hb145_output_io as io
from tools import hb145_r7_heldout_v1 as h
from tools import hb145_r7_pipeline_v111 as pipeline
from tools import hb145_r7_monitor_v111 as monitor
from tools import hb145_r7_carryover_v113 as migration


def record(out, uid="test", identity="test"):
    item = {h.UID: uid, "identity": identity, "fold_k": h.fold_k(uid),
            "status": "ok", "raw": [.2] * 12, "post": [.3] * 12}
    io.journal(h.record_path(out, uid), item, out)
    return item


@pytest.mark.parametrize("operation", ["replace", "rename", "unlink"])
@pytest.mark.parametrize("kind", ["permission", 5, 32])
def test_all_mutations_share_bounded_policy(monkeypatch, operation, kind):
    calls, sleeps = [], []
    def fail(*args):
        calls.append(args)
        error = PermissionError("reader") if kind == "permission" else OSError("reader")
        if kind != "permission":
            error.winerror = kind
        raise error
    monkeypatch.setattr(io.os, operation, fail)
    monkeypatch.setattr(io.time, "sleep", sleeps.append)
    with pytest.raises(OSError):
        getattr(io, operation)(*(["a"] if operation == "unlink" else ["a", "b"]))
    assert len(calls) == 40 and sleeps == list(io.REPLACE_DELAYS)
    assert sum(sleeps) == 29.5


def test_nonsharing_errors_do_not_retry(monkeypatch):
    calls = []
    def fail():
        calls.append(1)
        raise OSError(28, "disk full")
    with pytest.raises(OSError):
        io.retry(fail)
    assert len(calls) == 1


def test_exhausted_export_is_nonfatal_and_recovers(monkeypatch, tmp_path):
    record(tmp_path)
    original = io.os.replace
    calls = []
    def replace(source, target):
        if Path(target).suffix == ".csv":
            calls.append(1)
            raise PermissionError("held export")
        original(source, target)
    with monkeypatch.context() as m:
        m.setattr(io.os, "replace", replace)
        m.setattr(io.time, "sleep", lambda _: None)
        assert len(h.export_csv(tmp_path, ["test"], "test")) == 1
    assert len(calls) == 40
    assert json.loads((tmp_path / "export_health.json").read_text())["deferred"]
    h.export_csv(tmp_path, ["test"], "test")
    assert not json.loads((tmp_path / "export_health.json").read_text())["deferred"]
    assert (tmp_path / "post_stacker_NOT_CROSSFITTED.csv").exists()


def test_persistent_journal_failure_escapes_serving_catch_with_receipt(monkeypatch, tmp_path):
    original = io.os.replace
    def replace(source, target):
        if Path(target).parent.name == "journal":
            raise PermissionError("journal locked")
        original(source, target)
    monkeypatch.setattr(io.os, "replace", replace)
    monkeypatch.setattr(io.time, "sleep", lambda _: None)
    with pytest.raises(io.JournalCommitError):
        try:
            record(tmp_path)
        except Exception:
            pytest.fail("must escape serving per-study fallback")
    receipt = json.loads((tmp_path / "journal_failure_receipt.json").read_text())
    assert receipt["state"] == "JOURNAL_ERROR" and not receipt["committed"]
    assert not h.load_done(tmp_path, ["test"], "test")


def test_persistent_receipt_failure_uses_independent_fallback(monkeypatch, tmp_path):
    original = io.os.replace
    def replace(source, target):
        if Path(target).name == "receipt.json":
            raise PermissionError("receipt locked")
        original(source, target)
    monkeypatch.setattr(io.os, "replace", replace)
    monkeypatch.setattr(io.time, "sleep", lambda _: None)
    assert not io.receipt(tmp_path / "receipt.json", {"ok": True})
    assert json.loads((tmp_path / "receipt.deferred.json").read_text())["ok"]


def test_checkpoint_exports_count_commits_and_retry_next_checkpoint(monkeypatch, tmp_path):
    exports = []
    monkeypatch.setattr(h, "export_csv", lambda *args: exports.append(args))
    manager = pipeline.CheckpointExports(tmp_path, [], "test", 300)
    directory = tmp_path / "journal"
    directory.mkdir()
    for i in range(399):
        (directory / f"{i}.json").touch()
    for _ in range(20):
        manager.tick()
    assert not exports
    (directory / "399.json").touch()
    manager.tick()
    manager.close()
    for _ in range(20):
        manager.tick()
    assert len(exports) == 1
    for i in range(400, 500):
        (directory / f"{i}.json").touch()
    manager.tick()
    manager.close()
    assert len(exports) == 2


@pytest.mark.skipif(os.name != "nt", reason="requires real Windows sharing denial")
def test_reader_holds_export_three_seconds_run_continues(tmp_path):
    record(tmp_path)
    h.export_csv(tmp_path, ["test"], "test")
    path = tmp_path / "post_stacker_NOT_CROSSFITTED.csv"
    held = threading.Event()
    def reader():
        with path.open("rb"):
            held.set()
            time.sleep(3)
    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    assert held.wait(2)
    exports = pipeline.CheckpointExports(tmp_path, ["test"], "test", 0, every=1)
    started = time.perf_counter()
    code, receipt = monitor.monitored_run([sys.executable, "-c", "import time; time.sleep(3.5)"],
        dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), tmp_path / "run.log",
        tmp_path / "resources.json", exports.tick, poll_seconds=.03, stall_seconds=10)
    exports.close()
    reader_thread.join()
    assert time.perf_counter() - started >= 3 and code == 0
    assert receipt["state"] == "COMPLETED" and len(receipt["samples"]) > 20
    assert not json.loads((tmp_path / "export_health.json").read_text())["deferred"]


def test_generated_output_mutations_all_use_helper(tmp_path):
    runtime = h.runtime_source(tmp_path, dual=True) / "compact_infer.py"
    tree = ast.parse(runtime.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ("replace", "rename", "unlink", "rmtree") or not (
                isinstance(node.func.value, ast.Name) and node.func.value.id in ("os", "shutil"))


def test_export_journal_read_sharing_error_is_deferred(monkeypatch, tmp_path):
    record(tmp_path)
    def locked(*args):
        raise PermissionError("journal temporarily unreadable")
    monkeypatch.setattr(h, "load_done", locked)
    manager = pipeline.CheckpointExports(tmp_path, ["test"], "test", 0, every=1)
    manager.tick()
    manager.close()
    assert json.loads((tmp_path / "export_health.json").read_text())["deferred"]


def test_final_journal_read_failure_stops_with_receipt(monkeypatch, tmp_path):
    record(tmp_path)
    monkeypatch.setattr(h, "load_done", lambda *args: (_ for _ in ()).throw(PermissionError("read locked")))
    with pytest.raises(io.JournalCommitError, match="validation incomplete"):
        h.export_csv(tmp_path, ["test"], "test")
    receipt = io.read_json(tmp_path / "journal_failure_receipt.json")
    assert receipt["state"] == "JOURNAL_ERROR" and not receipt["validation_complete"]


def test_derived_stream_open_write_and_close_failures_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setattr(io.time, "sleep", lambda _: None)
    original = Path.open
    def locked(path, *args, **kwargs):
        if path.suffix == ".csv":
            raise PermissionError("locked derived csv")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, "open", locked)
        stream = io.DerivedStream(tmp_path / "predictions.csv", encoding="utf-8")
        assert stream.write("test") == 4
        stream.flush()
        stream.close()
    class BrokenStream:
        def write(self, value):
            raise PermissionError("write failed")
        def close(self):
            raise PermissionError("close failed")
    stream.stream = BrokenStream()
    assert stream.write("test") == 4 and stream.stream is None


def test_optional_receipt_read_locked_or_invalid_is_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setattr(io.time, "sleep", lambda _: None)
    assert io.read_receipt(tmp_path / "receipt.json") == {}
    (tmp_path / "receipt.json").write_text("broken json")
    io.atomic_json(tmp_path / "receipt.deferred.json", {"ok": True})
    assert io.read_receipt(tmp_path / "receipt.json") == {"ok": True}


def test_carry_over_verified_idempotent_payload_and_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(h, "SCRATCH", tmp_path)
    source, out = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    out.mkdir()
    old_runtime = source / "stage/r7_runtime/compact_infer.py"
    old_runtime.parent.mkdir(parents=True)
    old_runtime.write_text('(out / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")')
    new_runtime = out / "runtime.py"
    new_runtime.write_text(h.output_runtime_patch(old_runtime.read_text()))
    old = {"version": "1.1.2", "uids": [str(i) for i in range(302)], "device": "cuda",
           "runtime_sha256": h.r7.s6.sha256(old_runtime)}
    old_id = h.fingerprint(old)
    monkeypatch.setattr(migration, "V112_IDENTITY", old_id)
    h.atomic_json(source / "identity.json", old)
    h.atomic_json(source / "status.json", {"identity": old_id, "state": "MONITOR_ERROR", "cleanup_complete": True})
    for i in range(300):
        record(source, str(i), old_id)
    new = {**old, "version": "1.1.3", "runtime_sha256": h.r7.s6.sha256(new_runtime)}
    original_journal = h.output_io.journal
    copies = []
    def interrupted(*args):
        copies.append(1)
        if len(copies) == 4:
            raise KeyboardInterrupt("interrupted carry-over")
        return original_journal(*args)
    with monkeypatch.context() as m:
        m.setattr(h.output_io, "journal", interrupted)
        with pytest.raises(KeyboardInterrupt):
            migration.carry_over(source, out, new, new_runtime)
    assert len(list((out / "journal").glob("*.json"))) == 3
    with pytest.raises(ValueError, match="incomplete carry-over"):
        migration.validate_ready(out, h.fingerprint(new))
    manifest = migration.carry_over(source, out, new, new_runtime)
    assert manifest["committed_carried"] == 300 and manifest["remaining"] == 2
    assert migration.carry_over(source, out, new, new_runtime) == manifest
    done = h.load_done(out, new["uids"], h.fingerprint(new))
    assert len(done) == 300 and [u for u in new["uids"] if u not in done] == ["300", "301"]
    assert all(r["raw"] == [.2] * 12 for r in done.values())
    with pytest.raises(ValueError, match="semantic identity"):
        migration.carry_over(source, out, {**new, "device": "cpu"}, new_runtime)
    changed = next(iter(done.values()))
    changed["raw"] = [.9] * 12
    h.atomic_json(h.record_path(out, changed[h.UID]), changed)
    with pytest.raises(ValueError, match="committed journal changed"):
        migration.validate_ready(out, h.fingerprint(new))
    with pytest.raises(ValueError, match="collision"):
        migration.carry_over(source, out, new, new_runtime)
