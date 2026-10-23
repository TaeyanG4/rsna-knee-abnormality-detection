"""CPU-only production memory/phase/throughput/owned-attempt regressions."""
import json
import sys

import pytest

from tools import pl_production_watchdog_v1 as w


@pytest.fixture
def evidence(tmp_path):
    go, source = tmp_path/"go.json", tmp_path/"completed-screen.json"
    go.write_text('{"registered_go":{"selected_arm":"global"}}')
    source.write_text('{"completed":true}')
    doc = dict(schema="pl-production-memory/v1", device_name="NVIDIA GeForce RTX 4070 Ti SUPER",
               total_memory_mib=16376, selected_arm="global", go_sha256=w.sha(go),
               full_shape=dict(batch_size=8,k_train=12,k_eval=36,resolution=224,tta=True),
               measurements=[dict(phase=p,peak_reserved_bytes=9*1024**3,full_shape=True)
                             for p in ("fit","tta")],
               healthy_rates={"train_step":1.},
               source_pins=[dict(path=str(source),sha256=w.sha(source))])
    path = tmp_path/"memory.json"
    path.write_text(json.dumps(doc))
    return path, go, source, doc


def test_memory_gate_bound_to_go_and_measurement(evidence):
    path, go, source, doc = evidence
    assert w.validate_memory(path,go=go,arm="global") == doc
    with pytest.raises(ValueError,match="arm differs"):
        w.validate_memory(path,go=go,arm="fill")
    source.write_text("changed")
    with pytest.raises(ValueError,match="receipt changed"):
        w.validate_memory(path,go=go)


@pytest.mark.parametrize("mutation,message", [
    (lambda d:d.update(device_name="CPU"),"4070"),
    (lambda d:d["full_shape"].update(tta=False),"full production"),
    (lambda d:d["full_shape"].update(batch_size=1),"full production"),
    (lambda d:d["full_shape"].update(k_eval=24),"full production"),
    (lambda d:d["measurements"].pop(),"TTA measurements"),
    (lambda d:d["measurements"][0].update(full_shape=False),"not full shape"),
    (lambda d:d["measurements"][0].update(peak_reserved_bytes=10*1024**3+1),"10 GiB"),
    (lambda d:d.update(healthy_rates={"train_step":0}),"healthy"),
    (lambda d:d.update(source_pins=[]),"pins required"),
    (lambda d:d.update(go_sha256="0"*64),"GO binding changed"),
])
def test_refuse_unmeasured_or_wrong_memory(evidence, mutation, message):
    path, go, _, doc = evidence
    mutation(doc)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match=message):
        w.validate_memory(path,go=go)


def beat(seq, now, units, phase="training_step"):
    return dict(sequence=seq,completed_at=now,phase=phase,event="completed",
                detail=dict(progress_kind="train_step",cumulative_units=units))


@pytest.mark.parametrize("rate,expected", [(.2,"throughput_below_25pct_30m"),(.25,None),(1.,None)])
def test_throughput_boundaries(rate, expected):
    policy = w.Policy(0,healthy_rates={"train_step":1.})
    policy.check(0,beat(1,0,0))
    for now in range(60,1800,60):
        assert policy.check(now,beat(now+1,now,rate*now)) is None
    assert policy.check(1800,beat(1801,1800,rate*1800)) == expected


def test_phase_deadlines_keep_hb147_policy():
    policy = w.Policy(0,healthy_rates={"train_step":1.})
    assert policy.check(899) is None
    assert policy.check(900) == "no_heartbeat_15m"
    policy = w.Policy(0)
    stall = dict(util=100,power=70,used_mib=15900,total_mib=16376)
    assert policy.check(0,telemetry=stall) is None
    assert policy.check(600,telemetry=stall) == "wddm_no_heartbeat_10m"


def test_progress_counts_commits_and_nonfatal_exports(tmp_path,monkeypatch):
    path=tmp_path/"heartbeat.json"
    progress=w.ProductionProgress(path,"token")
    progress.completed("training_step",dict(update=10,skips=2))
    doc=json.loads(path.read_text())
    assert doc["detail"]["cumulative_units"] == 8
    sequence=doc["sequence"]
    progress.phase_start("checkpoint_write")
    assert json.loads(path.read_text())["sequence"] == sequence
    calls=[]
    monkeypatch.setattr(w,"atomic_json",lambda *a,**kw:calls.append(kw) or False)
    progress.completed("checkpoint_write",dict(bytes=4096))
    assert calls == [{"critical":False}]


def test_cpu_supervisor_and_immutable_attempt(tmp_path):
    child=("import time; from tools.pl_production_watchdog_v1 import PROGRESS; "
           "PROGRESS.completed('training_step',dict(update=1,skips=0)); time.sleep(.1)")
    attempt=tmp_path/"attempt"
    assert w.supervise([sys.executable,"-B","-c",child],attempt,cpu=True,poll=.02,
                       healthy_rates={"train_step":1.}) == 0
    receipt=json.loads((attempt/"receipt.json").read_text())
    assert receipt["reason"] == "completed" and receipt["survivors"] == []
    with pytest.raises(FileExistsError):
        w.supervise([sys.executable,"-c","pass"],attempt,cpu=True)


def test_cpu_cli_uses_measured_memory_and_preserves_attempt(evidence,tmp_path,monkeypatch):
    path,go,_,_=evidence
    monkeypatch.setattr(w,"PRODUCTION_ROOTS",(tmp_path,))
    args=["--attempt",str(tmp_path/"attempt"),"--budget",str(tmp_path/"budget.json"),
          "--memory-evidence",str(path),"--go",str(go),"--arm","global","--cpu",
          "--",sys.executable,"-c","pass"]
    assert w.main(args) == 0
    state=json.loads((tmp_path/"budget.json").read_text())
    assert state["active_since"] is None and state["used_seconds"] > 0
    with pytest.raises(FileExistsError):
        w.main(args)


def test_cpu_budget_exhaustion_refuses_before_child(evidence,tmp_path,monkeypatch):
    path,go,_,_=evidence
    monkeypatch.setattr(w,"PRODUCTION_ROOTS",(tmp_path,))
    (tmp_path/"budget.json").write_text('{"used_seconds":43200}')
    with pytest.raises(ValueError,match="cap exhausted"):
        w.main(["--attempt",str(tmp_path/"attempt"),"--budget",str(tmp_path/"budget.json"),
                "--memory-evidence",str(path),"--go",str(go),"--arm","global","--cpu",
                "--",sys.executable,"-c","pass"])
    assert not (tmp_path/"attempt").exists()
