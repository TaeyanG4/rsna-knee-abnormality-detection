"""Synthetic state machines plus real CPU subprocess lifetime checks."""
import json
import sys

import pytest

from tools import hb137_safety_v1_2_1 as s
from tools import hb137_screen_v1 as old


def test_progress_only_resets_on_new_completed_step():
    w = s.Watchdog(0, idle=10)
    assert w.check(9, 1) is None
    assert w.check(18, 1) is None
    assert w.check(19, 1) == "NO_STEP_PROGRESS"
    assert w.check(20, 2) is None


@pytest.mark.parametrize("field,value", [("util",97), ("watts",90), ("used_mib",15000)])
def test_fallback_requires_all_conditions_and_continuity(field, value):
    g = dict(util=100, watts=70, used_mib=15900, total_mib=16376)
    w = s.Watchdog(0, idle=900, stall=120)
    assert w.check(0, None, g) is None
    assert w.check(119, None, {**g, field:value}) is None
    assert w.check(120, None, g) is None
    assert w.check(239, None, g) is None
    assert w.check(240, 1, g) is None
    assert w.check(359, 1, g) is None
    assert w.check(360, 1, g) == "WDDM_FALLBACK_STALL"


@pytest.mark.parametrize("idle,stall", [(0,120),(900,float('nan'))])
def test_invalid_thresholds(idle,stall):
    with pytest.raises(ValueError): s.Watchdog(0,idle,stall)


@pytest.mark.parametrize("failure", ["idle","stall","total","exit"])
def test_cpu_child_terminated_and_partial_receipt(tmp_path, failure):
    ledger = tmp_path / "budget.json"
    cmd = [sys.executable,"-c","raise SystemExit(7)" if failure == "exit" else "import time;time.sleep(30)"]
    with pytest.raises(old.life.ContractError,match="child failed"):
        s.run(cmd, ledger, limit=.2 if failure == "total" else 10,
              idle=.2 if failure == "idle" else 900, stall=.1, poll=.05,
              sample=lambda: dict(util=100,watts=70,used_mib=15900,total_mib=16376) if failure == "stall" else None)
    r = json.loads(ledger.read_text())
    assert not r["running"] and r["attempts"][-1]["partial_not_scoreable"]
    assert r["attempts"][-1]["reason"] == dict(idle="NO_STEP_PROGRESS",stall="WDDM_FALLBACK_STALL",total="TOTAL_RUNTIME_CAP",exit=None)[failure]


def test_success_and_cumulative_exhaustion(tmp_path):
    ledger=tmp_path/"budget.json"
    assert s.run([sys.executable,"-c","pass"], ledger, sample=lambda:None)["status"] == "BUDGETED_COMMAND_PASS"
    with pytest.raises(old.life.ContractError,match="budget exhausted"):
        s.run([sys.executable,"-c","pass"], ledger, limit=.00001)


def test_progressing_real_child_does_not_timeout(tmp_path):
    script = "import os,json,time;from pathlib import Path;p=Path(os.environ['HB137_PROGRESS']);[(p.write_text(json.dumps({'progress_ns':i})),time.sleep(.1)) for i in range(8)]"
    # Atomic heartbeat helper is separately tested; generous startup allowance.
    assert s.run([sys.executable,"-c",script], tmp_path/"b.json", idle=.5,poll=.1,sample=lambda:None)["status"] == "BUDGETED_COMMAND_PASS"


def test_atomic_heartbeat(tmp_path,monkeypatch):
    p=tmp_path/"progress.json"
    monkeypatch.setenv("HB137_PROGRESS",str(p))
    s.heartbeat(); first=json.loads(p.read_text())
    s.heartbeat(); second=json.loads(p.read_text())
    assert second["progress_ns"] > first["progress_ns"]


def test_interrupted_ledger_refuses_launch(tmp_path):
    p=tmp_path/"b.json"
    p.write_text(json.dumps(dict(version=s.VERSION,running=True,seconds=0,attempts=[])))
    with pytest.raises(old.life.ContractError,match="interrupted"):
        s.run([sys.executable,"-c","pass"],p)


def test_admission_missing_or_incomplete_fails_closed(tmp_path,monkeypatch):
    p=tmp_path/"receipt.json"
    with pytest.raises(FileNotFoundError): s.admission(p)
    monkeypatch.setattr(s,"bindings",lambda:{"synthetic":"hash"})
    p.write_text(json.dumps(dict(status="PASS",device="cuda",limit_bytes=s.LIMIT_BYTES,sources=s.bindings(),measurements=[])))
    with pytest.raises(old.life.ContractError,match="incomplete"):
        s.admission(p)


def test_new_arm_preserves_recipe_and_progress(tmp_path,monkeypatch):
    import torch
    from training.arms import hb137_spatial_adapter_v1_2_1 as a
    from training.arms import hb137_spatial_adapter_v1 as b
    for new, previous in ((a.HEAD,b.HEAD),(a.ADAPTER,b.ADAPTER)):
        c=new.contract(42); c.pop("execution_safety_version")
        assert c == previous.contract(42)
    p=tmp_path/"heartbeat.json"
    monkeypatch.setenv("HB137_PROGRESS",str(p))
    assert a.HEAD.name.endswith("V121")


@pytest.mark.parametrize("bad", [None,"over","zero","nan","allocated","shape","steps","source","device","status"])
def test_complete_measured_gate_boundary(tmp_path,monkeypatch,bad):
    from tools import mil_screen_arm_build_v1 as b
    monkeypatch.setattr(s,"bindings",lambda:{"synthetic":"hash"})
    monkeypatch.setattr(b.life,"check_target_observation",lambda *args:None)
    monkeypatch.setattr(b.hooks,"control_spec",lambda seed:None)
    measurements=[dict(kind=k,phase=p,steps=3 if p=="train" else 2,
                       shape=[8,12,3,224,224] if p=="train" else [4,36,3,224,224],
                       allocated_bytes=100,reserved_bytes=s.LIMIT_BYTES)
                  for k in ("head","adapter") for p in ("train","validation","panel")]
    measurements.append({**measurements[-1],"kind":"parent"})
    r=dict(status="PASS",device="cuda",limit_bytes=s.LIMIT_BYTES,sources=s.bindings(),measurements=measurements,observation={})
    if bad=="over": measurements[0]["reserved_bytes"]+=1
    if bad=="zero": measurements[0]["allocated_bytes"]=0
    if bad=="nan": measurements[0]["reserved_bytes"]=float("nan")
    if bad=="allocated": measurements[0]["allocated_bytes"]=s.LIMIT_BYTES+1
    if bad=="shape": measurements[0]["shape"][0]=4
    if bad=="steps": measurements[0]["steps"]=1
    if bad=="source": r["sources"]={}
    if bad=="device": r["device"]="cpu"
    if bad=="status": r["status"]="FAIL"
    p=tmp_path/"receipt.json";p.write_text(json.dumps(r))
    if bad is None: assert s.admission(p)["status"]=="PASS"
    else:
        with pytest.raises((old.life.ContractError,ValueError)): s.admission(p)


def test_versioned_card_uses_gate_and_isolated_attempt():
    from tools import hb137_screen_v1_2_1 as screen
    assert screen.VERSION == "1.2.1"
    for phase in ("smoke","fit","panel"):
        for cmd in screen.commands()["phases"][phase]:
            if "tools/gpu_job.py" in cmd:
                if "memory-smoke" in cmd:
                    assert "--wait" in cmd
                    continue
                assert "tools.hb137_screen_v1_2_1" in cmd and "budget-run" in cmd
                assert "--wait" in cmd
                assert any("attempt-v1.2.1" in x for x in cmd)
