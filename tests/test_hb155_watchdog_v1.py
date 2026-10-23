"""H-B-155 paired crop CPU regression: sharing races, throughput, owned-tree cleanup."""
import json
import os
import subprocess
import sys

import psutil
import pytest

from tools import hb150_io_v1 as io
from tools import hb155_watchdog_v1 as w

STALL = dict(util=100, power=70, used_mib=15900, total_mib=16376)


def beat(seq, now, units, kind="train_step", event="completed"):
    return dict(sequence=seq, completed_at=now, phase=kind, event=event,
                detail=dict(progress_kind=kind, cumulative_units=units))


def test_sharing_replace_retries_and_nonfatal_export(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "RETRY_SECONDS", 0)
    calls = []
    replace = io.os.replace
    def locked_twice(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("reader holds destination")
        return replace(src, dst)
    monkeypatch.setattr(io.os, "replace", locked_twice)
    assert io.atomic_json(tmp_path/"heartbeat.json", {"sequence": 1}, critical=False)
    assert len(calls) == 3
    monkeypatch.setattr(io.os, "replace", lambda *_: (_ for _ in ()).throw(PermissionError("locked")))
    assert not io.atomic_json(tmp_path/"export.json", {}, critical=False)
    with pytest.raises(io.JournalWriteError):
        io.atomic_json(tmp_path/"journal.json", {})
    assert json.loads((tmp_path/"journal.json.io-failure.json").read_text())["status"] == "FAILED"


def test_winerror_32_retry_and_nonsharing_error_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "RETRY_SECONDS", 0)
    calls = []
    def error(*_):
        calls.append(1)
        exc = OSError("sharing")
        exc.winerror = 32
        raise exc
    monkeypatch.setattr(io.os, "replace", error)
    assert not io.safe_replace(tmp_path/"a", tmp_path/"b", critical=False)
    assert len(calls) == io.ATTEMPTS
    calls.clear()
    def different(*_):
        calls.append(1)
        raise OSError("disk failure")
    monkeypatch.setattr(io.os, "replace", different)
    assert not io.safe_replace(tmp_path/"a", tmp_path/"b", critical=False)
    assert len(calls) == 1


def test_zero_progress_and_wddm_deadlines():
    p = w.Policy(0)
    assert p.check(899.9) is None
    assert p.check(900) == "no_heartbeat_15m"
    p = w.Policy(0)
    p.check(0, telemetry=STALL)
    assert p.check(599, telemetry=STALL) is None
    assert p.check(600, telemetry=STALL) == "wddm_no_heartbeat_10m"


@pytest.mark.parametrize("rate, expected", [(1., None), (.25, None), (.2, "throughput_below_25pct_30m")])
def test_continuous_progress_can_still_be_throughput_collapse(rate, expected):
    p = w.Policy(0, healthy_rates={"train_step": 1.})
    p.check(0, beat(1, 0, 0))
    for t in range(60, 1800, 60):
        assert p.check(t, beat(t+1, t, rate*t), STALL) is None
    assert p.check(1800, beat(1801, 1800, rate*1800), STALL) == expected


def test_phase_switches_use_comparable_units_and_do_not_reset_slow_window():
    p = w.Policy(0, healthy_rates={"train_step": 2., "validation": .5})
    totals = dict(train_step=0., validation=0.)
    kind = "train_step"
    p.check(0, beat(1, 0, 0., kind))
    for t in range(60, 1801, 60):
        totals[kind] += p.rates[kind] * 60 * .2
        verdict = p.check(t, beat(t+1, t, totals[kind], kind))
        assert verdict == ("throughput_below_25pct_30m" if t == 1800 else None)
        kind = "validation" if kind == "train_step" else "train_step"
        # A phase transition with an inflated sequence/time is not completed work.
        p.check(t, beat(t+9999, t, totals[kind], kind, event="enter"))
    assert p.ratio == pytest.approx(.2)


def test_phase_only_forged_or_future_heartbeat_never_counts_work():
    p = w.Policy(0, healthy_rates={"train_step": 1.})
    p.check(0, beat(1, 0, 0))
    p.check(100, beat(2, 200, 1000))
    assert p.sequence == 1
    p.check(500, beat(999, 500, 500, event="enter"))
    assert p.sequence == 1
    assert p.check(900) == "no_heartbeat_15m"


def test_checkpoint_time_pauses_throughput_without_resetting_window():
    p = w.Policy(0, healthy_rates={"train_step": 1.})
    p.check(0, beat(1, 0, 0))
    p.check(600, beat(2, 600, 120))
    p.check(600, beat(2, 600, 0, "checkpoint", "enter"))
    for t in range(660, 2401, 60):
        assert p.check(t, beat(t+1, t, t, "checkpoint")) is None
    assert p.active_time == 600
    p.check(2400, beat(2401, 2400, 120, "train_step", "enter"))
    for t in range(2460, 3600, 60):
        assert p.check(t, beat(t+1, t, 120+.2*(t-2400))) is None
    assert p.check(3600, beat(3601, 3600, 360)) == "throughput_below_25pct_30m"


def test_supervisor_cpu_smoke_and_owned_tree_cleanup(tmp_path):
    code = ("import os,time; from tools.hb150_io_v1 import atomic_json; "
            "[atomic_json(os.environ['HB155_HEARTBEAT'],dict(token=os.environ['HB155_TOKEN'],"
            "pid=os.getpid(),sequence=i+1,completed_at=time.monotonic(),phase='train_step',"
            "event='completed',detail=dict(progress_kind='train_step',cumulative_units=i+1)),"
            "critical=False) or time.sleep(.01) for i in range(4)]")
    assert w.supervise([sys.executable, "-B", "-c", code], tmp_path/"healthy", cpu=True, poll=.02) == 0
    assert json.loads((tmp_path/"healthy/receipt.json").read_text())["reason"] == "completed"
    descendant = tmp_path/"descendant.json"
    code = ("import subprocess,sys,time,json,psutil;from pathlib import Path;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            f"Path({str(descendant)!r}).write_text(json.dumps(dict(pid=p.pid,created=psutil.Process(p.pid).create_time())));"
            "time.sleep(60)")
    unrelated = subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])
    try:
        assert w.supervise([sys.executable,"-c",code],tmp_path/"stalled",cpu=True,idle=1,total=5,poll=.02) == 124
        record = json.loads(descendant.read_text())
        assert not psutil.pid_exists(record["pid"]) or psutil.Process(record["pid"]).create_time() != record["created"]
        assert unrelated.poll() is None
        assert (tmp_path/"stalled/child.log").exists()
        with pytest.raises(FileExistsError):
            w.supervise([sys.executable,"-c","pass"],tmp_path/"stalled",cpu=True)
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_cli_requires_measured_rates_before_gpu_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "WORK", tmp_path)
    with pytest.raises(SystemExit):
        w.main(["--attempt",str(tmp_path/"attempt"),"--budget",str(tmp_path/"ledger.json"),"--",sys.executable,"-c","pass"])
    assert not (tmp_path/"attempt").exists()
    assert w.main(["--cpu","--attempt",str(tmp_path/"cpu"),"--budget",str(tmp_path/"ledger.json"),"--",sys.executable,"-c","pass"]) == 0
    assert json.loads((tmp_path/"ledger.json").read_text())["active_since"] is None


def test_supervisor_export_open_failure_is_nonfatal(tmp_path, monkeypatch):
    from pathlib import Path
    original = Path.open
    def locked_export(path, *args, **kwargs):
        if path.name in ("child.log", "telemetry.jsonl"):
            raise PermissionError("diagnostic export blocked")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", locked_export)
    assert w.supervise([sys.executable, "-c", "pass"],tmp_path/"attempt",cpu=True,poll=.02) == 0
    assert json.loads((tmp_path/"attempt/receipt.json").read_text())["reason"] == "completed"


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), "1"])
def test_bad_measured_rate_rejected(rate):
    with pytest.raises(ValueError):
        w.Policy(0, healthy_rates={"train_step": rate})


def test_fork_preserves_watchdog_mechanics_and_isolates_namespace():
    import ast
    parent = ast.parse((w.ROOT/'tools/hb147_confirm_watchdog_v1.py').read_text(encoding='utf-8'))
    current = ast.parse((w.ROOT/'tools/hb155_watchdog_v1.py').read_text(encoding='utf-8'))
    for name in ('stall_signature','telemetry','collect_owned','terminate_owned'):
        a = next(n for n in parent.body if getattr(n, 'name', None) == name)
        b = next(n for n in current.body if getattr(n, 'name', None) == name)
        assert ast.dump(a,include_attributes=False) == ast.dump(b,include_attributes=False)
    assert w.WORK == w.ROOT/'.local/scratch/slot-b/hb155/pilot-v120'
    source = (w.ROOT/'tools/hb155_watchdog_v1.py').read_text(encoding='utf-8')
    assert 'HB147_TOKEN' not in source and 'HB147_HEARTBEAT' not in source
    assert 'HB155_TOKEN' in source and 'HB155_HEARTBEAT' in source


def test_no_artificial_training_wall_cap():
    p=w.Policy(0)
    assert p.check(50000,beat(1,50000,1)) is None
