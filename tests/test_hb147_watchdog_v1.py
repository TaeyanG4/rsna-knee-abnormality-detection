"""CPU-only watchdog regression and real-native-pixel readout/checkpoint tests."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest
from tools import hb147_watchdog_v1 as w
from tools.hb147_progress_v1 import Progress, ProgressWriter

STALL = dict(util=100, power=70, used_mib=15900, total_mib=16384)


def beat(seq, stamp, phase="training_step"):
    return dict(sequence=seq, completed_at=stamp, phase=phase)


def test_healthy_heartbeats_override_stall_telemetry():
    p = w.Policy(0)
    for t in range(0, 3601, 30):
        assert p.check(t, beat(t+1, t), STALL) is None


def test_true_stall_and_generic_idle_boundaries():
    p = w.Policy(0)
    assert p.check(0, None, STALL) is None
    assert p.check(599.99, None, STALL) is None
    assert p.check(600, None, STALL) == "wddm_no_heartbeat_10m"
    p = w.Policy(0)
    assert p.check(899.99) is None
    assert p.check(900) == "no_heartbeat_15m"


def test_interrupted_signature_needs_continuous_ten_minutes():
    p = w.Policy(0)
    p.check(0, None, STALL)
    assert p.check(500, None, {}) is None
    assert p.check(600, None, STALL) is None
    assert p.check(899, None, STALL) is None
    assert p.check(900, None, STALL) == "no_heartbeat_15m"


def test_long_checkpoint_progress_and_total_cap():
    p = w.Policy(0, total=43200)
    for t in range(0, 42001, 60):
        assert p.check(t, beat(t+1, t, "checkpoint_write"), STALL) is None
    assert p.check(43200, beat(43201, 43200), STALL) == "total_cap"


def test_stale_future_or_phase_only_events_do_not_extend_deadline():
    p = w.Policy(0)
    p.check(50, beat(1, 50))
    assert p.check(100, beat(2, 200)) is None
    assert p.check(949, beat(1, 949, "checkpoint_write")) is None
    assert p.check(950, dict(sequence=1, completed_at=50, phase="panel_inference")) == "no_heartbeat_15m"


def test_chunked_checkpoint_serialization_and_no_timer(tmp_path):
    import torch
    progress = Progress(tmp_path / "heartbeat.json", "test")
    payload = {"tensor": torch.arange(2400000, dtype=torch.float32)}
    buffer = io.BytesIO()
    writer = ProgressWriter(buffer, progress)
    torch.save(payload, writer)
    writer.flush()
    record = json.loads(progress.path.read_text())
    assert record["phase"] == "checkpoint_write" and record["detail"]["bytes"] > 9_000_000
    buffer.seek(0)
    torch.testing.assert_close(torch.load(buffer, weights_only=True)["tensor"], payload["tensor"])
    progress.phase_start("validation")
    assert json.loads(progress.path.read_text())["sequence"] == record["sequence"]


def test_long_actual_writes_emit_healthy_heartbeats(tmp_path, monkeypatch):
    # Simulated 20-minute write; each returned chunk advances the clock 60s.
    import tools.hb147_progress_v1 as module
    clock = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    progress = Progress(tmp_path / "heartbeat.json", "test")
    policy = w.Policy(0)
    class Sink:
        def write(self, data):
            clock[0] += 60
            return len(data)
        def flush(self):
            pass
    writer = ProgressWriter(Sink(), progress)
    for _ in range(20):
        writer.write(b"real completed write")
        assert policy.check(clock[0], json.loads(progress.path.read_text()), STALL) is None
    assert clock[0] == 1200


@pytest.mark.parametrize("phase", ["training_step", "validation", "panel_inference", "checkpoint_write"])
def test_healthy_subprocess_with_stall_like_telemetry(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(w, "telemetry", lambda: STALL)
    code = ("from tools.hb147_progress_v1 import PROGRESS; import time; "
            f"[(PROGRESS.completed('{phase}', i), time.sleep(.05)) for i in range(12)]")
    rc = w.supervise([sys.executable, "-B", "-c", code], tmp_path/"attempt",
                     idle=1.5, stall=.3, total=5, poll=.03)
    assert rc == 0
    receipt = json.loads((tmp_path/"attempt/receipt.json").read_text())
    assert receipt["last_phase"] == phase and receipt["completed_events"] >= 2
    assert receipt["survivors"] == []


def test_true_stall_terminates_only_owned_tree_preserves_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "telemetry", lambda: STALL)
    partial, descendant = tmp_path/"partial.pt", tmp_path/"descendant.json"
    code = ("import subprocess,sys,time,json,psutil; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            f"Path({str(descendant)!r}).write_text(json.dumps(dict(pid=p.pid, created=psutil.Process(p.pid).create_time()))); "
            f"Path({str(partial)!r}).write_bytes(b'partial-checkpoint'); time.sleep(60)")
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        rc = w.supervise([sys.executable,"-B","-c",code], tmp_path/"attempt",
                         idle=3, stall=1, total=5, poll=.03)
        assert rc == 124 and partial.read_bytes() == b"partial-checkpoint"
        receipt = json.loads((tmp_path/"attempt/receipt.json").read_text())
        assert receipt["reason"] == "wddm_no_heartbeat_10m" and not receipt["survivors"]
        owned = json.loads(descendant.read_text())
        assert not psutil.pid_exists(owned["pid"]) or psutil.Process(owned["pid"]).create_time() != owned["created"]
        assert unrelated.poll() is None
        with pytest.raises(FileExistsError):
            w.supervise([sys.executable,"-c","pass"],tmp_path/"attempt",cpu=True)
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_cpu_total_deadline_and_nonzero_exit(tmp_path):
    rc = w.supervise([sys.executable,"-c","import time; time.sleep(60)"],tmp_path/"total",
                     cpu=True, total=.2, poll=.02)
    assert rc == 124
    assert json.loads((tmp_path/"total/receipt.json").read_text())["reason"] == "total_cap"
    assert w.supervise([sys.executable,"-c","raise SystemExit(7)"],tmp_path/"exit",cpu=True,poll=.02) == 7


def test_shared_budget_exhaustion_and_crash_conservative_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "WORK", tmp_path)
    ledger = tmp_path/"budget.json"
    ledger.write_text(json.dumps(dict(used_seconds=43199, active_since=time.time()-10)))
    with pytest.raises(RuntimeError, match="cap exhausted"):
        w.main(["--attempt",str(tmp_path/"blocked"),"--budget",str(ledger),"--cpu","--",sys.executable,"-c","pass"])
    assert not (tmp_path/"blocked").exists()


def test_real_native_pixel_panel_and_validation_progress(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from training.arms import native_resolution_v2 as n
    from training.arms import mil_screen_hooks_v3 as runner
    progress = Progress(tmp_path/"heartbeat.json", "test")
    monkeypatch.setattr(runner, "PROGRESS", progress)
    corpus = n.h.common.Corpus(n.h.ROOT/n.CORPUS)
    uid = corpus.ids.tolist()[0]
    ds = n.NativeWindows(n.h.MILWindows(n.h.ROOT/n.CORPUS, [uid],
                         {uid: np.full(12,.5,np.float32)},24,False,42),320)
    loader = torch.utils.data.DataLoader(ds,batch_size=1,collate_fn=n.ARM.collate_fn)
    class PixelModel(torch.nn.Module):
        def forward(self,x):
            return x.mean(dim=(1,2,3,4))[:,None].expand(-1,12)
    for phase in ("validation", "panel_inference"):
        values, ids, batches = runner.evaluate_with(n.ARM, PixelModel(), loader, "cpu", "none", corpus, tta=True, phase=phase)
        assert values.shape == (1,12) and np.isfinite(values).all() and batches == 1
        record = json.loads(progress.path.read_text())
        assert record["phase"] == phase and record["event"] == "completed"


def test_new_sources_and_scientific_configs_unchanged():
    from tools import hb147_build_v1 as old, hb147_build_v2 as new
    if not (new.WORK/"configs/native320-s42.json").exists():
        pytest.skip("run configure first")
    for r in (224,320):
        for s in (42,43):
            a,b = old.spec_for(r,s),new.spec_for(r,s)
            assert a.document["config"] == b.document["config"]
            new.native.ARM.validate_recipe(b)
            for path,digest in b.document["inputs"]["sources"].items():
                assert new.h.sha(new.ROOT/path) == digest
            assert "tools/hb147_watchdog_v1.py" in b.document["inputs"]["sources"]
