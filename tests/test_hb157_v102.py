"""v1.0.2 supervision/carryover regressions; CPU only."""
import copy
import sys
import pytest
from tools import hb157_rkd_v1 as b
from tools import hb150_watchdog_v1 as w
from tools.pl_production_watchdog_v1 import ProductionProgress


def beat(seq, now, units, kind='train_step', event='completed'):
    return dict(sequence=seq, completed_at=now, phase=kind, event=event,
                detail=dict(progress_kind=kind, cumulative_units=units))


def test_normal_progress_beyond_four_hours_and_resume_offset():
    p=w.Policy(0,total=None,idle_requires_gpu_inactive=True,
               healthy_rates={'train_step':1/5.422,'bn_replay':1/.875})
    p.check(0,beat(1,0,272))
    for t in range(6,20001,6):
        assert p.check(t,beat(t+1,t,272+t/6),{'util':100}) is None
    assert p.ratio > .8
    # BN and checkpoint transitions retain independent unit counters.
    assert p.check(20000,beat(20001,20000,0,'checkpoint','enter'),{'util':0}) is None
    assert p.check(20100,beat(20101,20100,1,'bn_replay'),{'util':50}) is None


@pytest.mark.parametrize('sample',[None,{'error':'unavailable'},{'util':float('nan')},
                                  {'util':float('inf')},{'util':-1},{'util':1}])
def test_idle_requires_continuous_measured_inactivity(sample):
    p=w.Policy(0,total=None,idle_requires_gpu_inactive=True)
    p.check(0,telemetry={'util':0})
    assert p.check(899,telemetry=sample) is None
    assert p.check(900,telemetry={'util':0}) is None
    assert p.check(1799,telemetry={'util':0}) is None
    assert p.check(1800,telemetry={'util':0})=='no_heartbeat_15m'


def test_completed_progress_resets_inactive_interval_but_enter_does_not():
    p=w.Policy(0,total=None,idle_requires_gpu_inactive=True)
    p.check(0,telemetry={'util':0})
    p.check(899,beat(1,899,1),{'util':0})
    p.check(900,telemetry={'util':0})
    assert p.check(1799,beat(100,1799,1,event='enter'),{'util':0}) is None
    assert p.check(1800,telemetry={'util':0})=='no_heartbeat_15m'


def test_wddm_and_measured_throughput_remain_active():
    sample=dict(util=100,power=70,used_mib=15900,total_mib=16376)
    p=w.Policy(0,total=None,idle_requires_gpu_inactive=True)
    p.check(0,telemetry=sample)
    assert p.check(600,telemetry=sample)=='wddm_no_heartbeat_10m'
    for fraction in [.2,.5]:
        p=w.Policy(0,total=None,idle_requires_gpu_inactive=True,healthy_rates={'train_step':1/5.422})
        p.check(0,beat(1,0,0))
        for t in range(60,1801,60):
            result=p.check(t,beat(t+1,t,t/5.422*fraction),{'util':90})
            assert result==('throughput_below_25pct_30m' if t==1800 and fraction==.2 else None)


def test_real_progress_adapter_units(tmp_path):
    path=tmp_path/'heartbeat.json';p=ProductionProgress(path,'fixture')
    p.completed('training_step',{'update':273})
    assert b.read(path)['detail']==dict(update=273,progress_kind='train_step',cumulative_units=273)


def test_unlimited_supervisor_real_cpu_process_and_log_receipt(tmp_path):
    log='artifacts/logs/HB157_RKD_V102'
    assert w.supervise([sys.executable,'-B','-c','print("CPU only")'],tmp_path/'attempt',
                       cpu=True,total=None,idle_requires_gpu_inactive=True,poll=.02,gpu_job_log_dir=log)==0
    r=b.read(tmp_path/'attempt/receipt.json')
    assert r['reason']=='completed' and r['survivors']==[] and r['gpu_job_log_dir']==log
    assert b.read(tmp_path/'attempt/command.json')['remaining_total_seconds'] is None


def test_idle_owned_cleanup_without_total(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'telemetry',lambda:{'util':0})
    assert w.supervise([sys.executable,'-B','-c','import time;time.sleep(60)'],tmp_path/'idle',
                       total=None,idle=.3,idle_requires_gpu_inactive=True,poll=.02)==124
    r=b.read(tmp_path/'idle/receipt.json')
    assert r['reason']=='no_heartbeat_15m' and not r['survivors']


@pytest.mark.parametrize('field',['config','workers','decisions','resume','train_roster_sha256','clean_oof','pins'])
def test_reuse_rejects_scientific_and_training_changes(field):
    p=copy.deepcopy(b.read(b.PROTOCOL))
    if field=='pins':p[field]['training/arms/hb157_rkd_fit_v1.py']='0'*64
    else:p[field]='drift'
    with pytest.raises(ValueError,match='reuse'):b.reuse_descriptor(p)


@pytest.mark.parametrize('name',['protocol.json','cuda-smoke.json','smoke-attempt/receipt.json',
                                'smoke-attempt/identity.json','smoke-attempt/heartbeat.json',
                                'pre-gpu-review.json','cpu-smoke.json'])
def test_reuse_rejects_evidence_tamper(name,monkeypatch):
    original=b.sha
    monkeypatch.setattr(b,'sha',lambda p:'0'*64 if p==b.PRIOR/name else original(p))
    with pytest.raises(ValueError):b.validate()


def test_fit_admission_binds_new_and_original_evidence(monkeypatch):
    seen=[]
    monkeypatch.setattr(b,'require_pre_review',lambda:None)
    original=b.read
    monkeypatch.setattr(b,'read',lambda p:{} if p==b.WORK/'fit-review.json' else original(p))
    original_check=b.check_review
    def check(r,bindings):
        if 'source_protocol_sha256' in bindings:seen.append(bindings)
        else:original_check(r,bindings)
    monkeypatch.setattr(b,'check_review',check)
    assert b.require_fit()['status']=='PASS'
    assert seen[0]==dict(protocol_sha256=b.PROTOCOL,cpu_smoke_sha256=b.WORK/'cpu-smoke.json',
                         cuda_smoke_sha256=b.PRIOR/'cuda-smoke.json',
                         supervisor_receipt_sha256=b.PRIOR/'smoke-attempt/receipt.json',
                         source_protocol_sha256=b.PRIOR/'protocol.json')


@pytest.mark.parametrize('mode,tag',[('fit','HB157_RKD_V102'),('smoke','HB157_RKD_SMOKE_V102')])
def test_launch_versioned_tags_and_windows_interpreter(mode,tag,monkeypatch):
    commands=[]
    monkeypatch.setattr(b.subprocess,'call',lambda command,**kw:commands.append(command) or 0)
    b.launch(mode)
    assert commands[0][3:5]==['--tag',tag]
    assert '/' not in commands[0][0] and commands[0][-1]==mode


def test_fit_accounting_no_longer_caps_normal_or_resumed_run(tmp_path,monkeypatch):
    from tools import hb151_build_v3 as parent
    smoke=b.read(b.PRIOR/'cuda-smoke.json');calls=[]
    monkeypatch.setattr(b,'WORK',tmp_path)
    monkeypatch.setattr(b,'RUN',tmp_path/'fit')
    monkeypatch.setattr(b,'require_gpu_job',lambda *a,**kw:None)
    monkeypatch.setattr(b,'validate',lambda:None)
    monkeypatch.setattr(b,'require_pre_review',lambda:None)
    monkeypatch.setattr(b,'require_fit',lambda:smoke)
    monkeypatch.setattr(parent,'desktop_memory_snapshot',lambda:smoke['desktop_before_cuda'])
    monkeypatch.setattr(w,'supervise',lambda *a,**kw:calls.append(kw) or 0)
    b.atomic_json(tmp_path/'budget.json',dict(attempt=3,charged_seconds=20000,running=False))
    assert b.supervise('fit',True)==0
    assert calls[0]['total'] is None and calls[0]['idle_requires_gpu_inactive'] is True
    assert calls[0]['gpu_job_log_dir']=='artifacts/logs/HB157_RKD_V102'
    assert calls[0]['healthy_rates']['train_step']==1/smoke['measurements']['train']['seconds_per_unit']
    assert b.read(tmp_path/'budget.json')['charged_seconds']>=20000
