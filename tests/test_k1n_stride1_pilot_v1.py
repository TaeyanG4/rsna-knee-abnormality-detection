"""CPU-only admission bypass, exact rule boundaries, roster and resource gates."""
import copy
import json
import pytest
from tools import k1n_stride1_pilot_v1 as p
from tools import l1_rco_k1_eval_v1 as evaluator
from tools import l1_rco_runtime_v120 as safety


def metrics():
    names=['jev_hybrid','aquila_v1','dread','consensus']
    return dict(rulers={n:dict(delta=0.) for n in names},
                per_target=[dict(ruler=n,target=t,delta=0.) for n in names for t in evaluator.TARGETS])


@pytest.mark.parametrize('macro_losses,target_delta,passed',[(0,0,True),(1,-.005,True),(2,0,False),(0,-.005001,False),(0,float('nan'),False)])
def test_registered_rule(macro_losses,target_delta,passed):
    m=metrics()
    for n in list(m['rulers'])[:macro_losses]:m['rulers'][n]['delta']=-.000001
    m['per_target'][0]['delta']=target_delta
    assert p.go_rule(m) is passed


def test_missing_or_duplicate_target_rejected():
    m=metrics(); m['per_target'][-1]=dict(m['per_target'][0])
    assert not p.go_rule(m)
    m=metrics(); m['per_target'].pop(); assert not p.go_rule(m)


def test_direct_run_cannot_bypass_review(monkeypatch):
    monkeypatch.setattr(p,'admit',lambda:dict(status='BLOCKED'))
    monkeypatch.setattr(p,'owner_check',lambda:pytest.fail('GPU ownership accessed before review'))
    with pytest.raises(ValueError,match='review'):p.run()


@pytest.mark.parametrize('fault',['missing','self','bindings','report'])
def test_review_artifact_fails_closed(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(p,'OUT',tmp_path); monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'review_bindings',lambda:{'smoke':'bound'})
    if fault!='missing':
        (tmp_path/'review.md').write_text('independent evidence')
        r=dict(verdict='PASS',reviewer='astra-5',bindings={'smoke':'bound'},report='review.md',report_sha256=p.sha(tmp_path/'review.md'))
        if fault=='self':r['reviewer']='astra-b2'
        if fault=='bindings':r['bindings']={}
        if fault=='report':r['report_sha256']='bad'
        (tmp_path/'REVIEW.json').write_text(json.dumps(r))
    with pytest.raises((ValueError,OSError)):p.require_review()


def test_worker_no_owner_fails_before_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'read',lambda _:dict(supervisor={'pid':0,'create_time':0}))
    with pytest.raises(ValueError,match='owned worker'):p.worker()


def test_gpu_owner_wrong_command_fails(monkeypatch):
    monkeypatch.setattr(p,'read',lambda _:dict(command=['wrong']))
    with pytest.raises(ValueError,match='command identity'):p.owner_check()


def test_real_desktop_compute_parser(monkeypatch):
    sample='123, C:\\Windows\\explorer.exe\n456, C:\\Python312\\python.exe\n789, C:\\Chrome\\chrome.exe\n'
    monkeypatch.setattr(safety.subprocess,'check_output',lambda *a,**kw:sample)
    assert len(safety.active_ml_processes())==1 and not safety.active_ml_processes(456)


def test_memory_and_watchdog_boundaries():
    gib=2**30
    assert safety.memory_ok(12*gib,1*gib,16*gib)
    assert not safety.memory_ok(12*gib+1,1*gib,16*gib)
    assert not safety.memory_ok(12*gib,3*gib,16*gib)
    assert safety.stale_reason({'time':0},900)
    assert safety.stale_reason({'time':0},600,dict(util=99,power=80,desktop_bytes=15*gib,total_bytes=16*gib))


def test_exact_roster_is_977():
    assert len(p.cohort())==977
    assert p.RULE['evaluation_n']==977 and p.ORDER==(('v2',2),('s0',2),('s1',2),('s1',1))
