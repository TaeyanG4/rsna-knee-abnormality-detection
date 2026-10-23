"""CPU-only regression tests for the single registered crop pilot v1.0.0."""
import copy
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from tools import k1_tta_pilot_v1 as p
from tools import l1_rco_k1_eval_v1 as evaluator
from tools import l1_rco_runtime_v120 as safety


def metrics():
    names=['jev_hybrid','aquila_v1','dread','consensus']
    return dict(rulers={n:dict(delta=.003) for n in names},
        per_target=[dict(ruler=n,target=t,delta=.003) for n in names for t in evaluator.TARGETS])


@pytest.mark.parametrize('values,passed',[
    ([.003]*4,True),([.002999]*4,False),([.004]*3+[0],True),
    ([.006,.006,0,0],True),([.01,.01,-.001,-.001],False),
    ([float('nan'),.01,.01,.01],False)])
def test_macro_mean_and_vote_boundaries(values,passed):
    m=metrics()
    for n,d in zip(m['rulers'],values):m['rulers'][n]['delta']=d
    assert p.go_rule(m) is passed


@pytest.mark.parametrize('values,passed',[
    ([-.005]*4,True),([-.005001]*4,False),
    ([-.02,0,0,0],True),([-.020001,0,0,0],False),
    ([float('inf'),0,0,0],False)])
def test_target_mean_not_single_ruler_floor(values,passed):
    m=metrics()
    rows=[r for r in m['per_target'] if r['target']==evaluator.TARGETS[0]]
    for r,d in zip(rows,values):r['delta']=d
    assert p.go_rule(m) is passed


@pytest.mark.parametrize('fault',['missing','duplicate','ruler'])
def test_incomplete_metrics_fail(fault):
    m=metrics()
    if fault=='missing':m['per_target'].pop()
    elif fault=='duplicate':m['per_target'][-1]=dict(m['per_target'][0])
    else:m['rulers'].pop('dread')
    assert not p.go_rule(m)


def test_probability_pair_before_ranks():
    from tools.c85r7_rco_overlay_v1 import ranks
    rng=np.random.default_rng(281)
    a={m:rng.random((977,12)).astype(np.float32) for m,_ in p.ORDER}
    z={m:rng.random((977,12)).astype(np.float32) for m,_ in p.ORDER}
    candidate,control=p.paired_trios(a,z)
    np.testing.assert_array_equal(candidate,ranks(np.mean([ranks((a[m]+z[m])/2) for m in a],axis=0)))
    np.testing.assert_array_equal(control,ranks(np.mean([ranks(a[m]) for m in a],axis=0)))
    wrong=ranks(np.mean([(ranks(a[m])+ranks(z[m]))/2 for m in a],axis=0))
    assert not np.array_equal(candidate,wrong)


@pytest.mark.parametrize('fault',['shape','nan','range','member'])
def test_pair_input_validation(fault):
    a={m:np.ones((977,12),np.float32)*.5 for m,_ in p.ORDER}
    z=copy.deepcopy(a)
    if fault=='shape':z['s0']=z['s0'][:-1]
    elif fault=='nan':z['s0'][0,0]=np.nan
    elif fault=='range':z['s0'][0,0]=1.01
    else:z.pop('s1')
    with pytest.raises(ValueError):p.paired_trios(a,z)


def test_crop_preserves_order_axes_and_metadata():
    from tools import k1sr_runtime_v11 as runtime
    x=(torch.arange(2*3*384*384).reshape(2,3,384,384)%251).to(torch.uint8)
    meta=torch.arange(14).reshape(2,7).float()
    a=p.transfer_view(x,meta,'cpu',False); z=p.transfer_view(x,meta,'cpu')
    assert all(torch.equal(u,v) for u,v in zip(a,runtime.transfer_inputs(x,meta,'cpu')))
    assert torch.equal(z[0],x[None,...,8:360,16:368])
    assert torch.equal(a[1],z[1]) and torch.equal(a[2],z[2])
    assert not torch.equal(a[0],z[0])
    with pytest.raises(ValueError):p.transfer_view(x.float(),meta,'cpu')


def test_direct_run_cannot_bypass_review(monkeypatch):
    monkeypatch.setattr(p,'admit',lambda:dict(status='BLOCKED'))
    monkeypatch.setattr(p,'owner_check',lambda:pytest.fail('ownership before review'))
    with pytest.raises(ValueError,match='review'):p.run()


@pytest.mark.parametrize('fault',['missing','self','bindings','report'])
def test_review_fails_closed(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(p,'OUT',tmp_path);monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'review_bindings',lambda:{'smoke':'bound'})
    if fault!='missing':
        (tmp_path/'review.md').write_text('independent evidence')
        r=dict(verdict='PASS',reviewer='astra-verifier',bindings={'smoke':'bound'},report='review.md',report_sha256=p.sha(tmp_path/'review.md'))
        if fault=='self':r['reviewer']='astra-b1'
        if fault=='bindings':r['bindings']={}
        if fault=='report':r['report_sha256']='bad'
        (tmp_path/'REVIEW.json').write_text(json.dumps(r))
    with pytest.raises((ValueError,OSError)):p.require_review()


def test_worker_no_owner_before_cuda(monkeypatch):
    monkeypatch.setattr(p,'read',lambda _:dict(supervisor={'pid':0,'create_time':0}))
    with pytest.raises(ValueError,match='owned worker'):p.worker()


def test_wrong_gpu_job_identity(monkeypatch):
    monkeypatch.setattr(p,'read',lambda _:dict(command=['wrong']))
    with pytest.raises(ValueError,match='command identity'):p.owner_check()


@pytest.mark.parametrize('util,processes,passed',[
    (59,'123, C:\\Windows\\explorer.exe\n',True),
    (60,'',False),(0,'456, C:\\Python312\\python.exe\n',False),
    (0,'456, C:\\unknown.exe\n',False),(0,'42, C:\\Python312\\python.exe\n',True),
    (0,'bad row',False)])
def test_idle_resource_gate(monkeypatch,util,processes,passed):
    monkeypatch.setattr(safety,'gpu_snapshot',lambda:dict(util=util))
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:processes)
    if passed:assert p.idle_snapshot(42)['util']==util
    else:
        with pytest.raises(ValueError):p.idle_snapshot(42)


@pytest.mark.parametrize('name,passed',[('dwm.exe',True),('python.exe',False)])
def test_hidden_wddm_name_resolved_or_blocked(monkeypatch,name,passed):
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(safety,'gpu_snapshot',lambda:dict(util=20))
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:'123, [Insufficient Permissions]\n')
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(name=lambda:name))
    if passed:p.idle_snapshot(42)
    else:
        with pytest.raises(ValueError):p.idle_snapshot(42)


def test_memory_watchdog_no_total_timeout():
    gib=2**30
    assert safety.memory_ok(12*gib,2*gib,16*gib)
    assert not safety.memory_ok(12*gib+1,2*gib,16*gib)
    assert not safety.memory_ok(12*gib,2*gib+1,16*gib)
    assert safety.stale_reason({'time':0},900)
    assert safety.stale_reason({'time':0},600,dict(util=99,power=80,desktop_bytes=15*gib,total_bytes=16*gib))
    assert p.RULE['gpu_seconds_limit'] is None
    assert '7200' not in inspect.getsource(p.run)


@pytest.mark.parametrize('dest',['preds','pilot-receipt.json','gpu-admission.json','pilot-status.json','pilot-progress.json','PILOT-RESULT.json','pilot-terminal.json'])
def test_existing_destination_retained(tmp_path,monkeypatch,dest):
    monkeypatch.setattr(p,'OUT',tmp_path)
    f=tmp_path/dest;f.write_text('preserve')
    with pytest.raises(ValueError,match='existing destination'):p.assert_fresh()
    assert f.read_text()=='preserve'


@pytest.mark.parametrize('fault',['receipt','incomplete','registration','cohort','preprocessing','checkpoint','csv','stride'])
def test_control_authentication(monkeypatch,fault):
    from tools import h216_pair_eval_v1 as loader
    ids=list(range(977))
    r=dict(status='COMPLETE',preregistration_sha256='ok',outputs={
        m+'_s2':dict(eval_stride=2,checkpoint_sha256=p.CHECKPOINTS[m][1],csv_sha256='ok') for m,_ in p.ORDER})
    reg=dict(inputs=dict(n=977,id_list_sha256=p.digest(ids)),preprocessing=dict(size=384,crop=[16,368],slices=3,phys_mm=3.5,encoder_chunk=32,fp16=True,tf32=False,extent=False))
    if fault=='incomplete':r['status']='PARTIAL'
    elif fault=='registration':r['preregistration_sha256']='bad'
    elif fault=='cohort':reg['inputs']['n']=976
    elif fault=='preprocessing':reg['preprocessing']['fp16']=False
    elif fault=='checkpoint':r['outputs']['v2_s2']['checkpoint_sha256']='bad'
    elif fault=='csv':r['outputs']['v2_s2']['csv_sha256']='bad'
    elif fault=='stride':r['outputs']['v2_s2']['eval_stride']=1
    monkeypatch.setattr(p,'cohort',lambda:ids)
    monkeypatch.setattr(p,'sha',lambda path:('bad' if fault=='receipt' else p.CONTROL_RECEIPT_SHA) if Path(path).name=='pilot-receipt.json' else 'ok')
    monkeypatch.setattr(p,'read',lambda path:r if Path(path).name=='pilot-receipt.json' else reg)
    monkeypatch.setattr(loader,'load',lambda *a:pytest.fail('tampered controls reached load'))
    with pytest.raises(ValueError):p.controls()


def test_actual_training_and_exact_roster():
    assert len(p.cohort())==977
    assert p.ORDER==(('v2',2),('s0',2),('s1',2))
    evidence=p.training_evidence()
    assert all(v['augmentation']=='crop' and v['offset_support']==[0,32] for v in evidence.values())
