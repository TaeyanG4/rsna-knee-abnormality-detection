"""H-B-147 lens debt v1.0.0: decision, binding and controller regressions."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import hb147_lenses_v1 as b


def test_q_rule_three_rulers_and_twofold_cell_floor():
    d=np.full((2,4,12),.001)
    d[:,3]=0
    assert b.q_guard(d)['PASS']
    d[:,2]=0
    assert not b.q_guard(d)['PASS']
    d[:]=.003
    d[:,:,0]=-.003
    assert b.q_guard(d)['PASS']
    d[1,0,0]=-.00301
    result=b.q_guard(d)
    assert result['qualifying_rulers']==[4,4]
    assert not result['harm_guard_pass'] and not result['PASS']


@pytest.mark.parametrize('shape,value',[((1,4,12),0),((2,3,12),0),((2,4,12),np.nan),((2,4,12),np.inf)])
def test_q_malformed_closed(shape,value):
    with pytest.raises(ValueError): b.q_guard(np.full(shape,value))


def test_replacement_uses_all_folds_unchanged_baseline_and_no_mutation():
    from scipy.stats import rankdata
    raw={f:{k:np.full((f+2,12),.1*(f+1),np.float32) for k in ('S42','S43','R42')} for f in range(5)}
    rosters=[{'val':[f'f{f}-{i}' for i in range(f+2)]} for f in range(5)]
    before=copy.deepcopy(raw); calls=[]
    def variants(x):
        calls.append(copy.deepcopy(x))
        s=np.concatenate([(x[f]['S42']+x[f]['S43'])/np.float32(2) for f in range(5)])
        r=np.concatenate([x[f]['R42'] for f in range(5)])
        return {'replace':rankdata((rankdata(s,axis=0)+rankdata(r,axis=0))/2,axis=0)}
    original=variants(raw)['replace']; arms={s:np.full((3,12),.9) for s in ('42','43')}
    controls={s:np.full((3,12),.05) for s in ('42','43')}
    qa,qc,qb=b.q_replacements(SimpleNamespace(variants=variants),raw,rosters,1,rosters[1]['val'],arms,controls,original)
    assert np.array_equal(qb,original[2:5]) and not np.array_equal(qa,qc)
    for call in calls[1:]:
        for f in range(5):
            for k in raw[f]:
                assert np.array_equal(raw[f][k],before[f][k])
                if f!=1 or k=='R42': assert np.array_equal(call[f][k],before[f][k])
        assert call[1]['S42'].dtype==np.float32
    with pytest.raises(ValueError):
        b.q_replacements(SimpleNamespace(variants=variants),raw,rosters,1,list(reversed(rosters[1]['val'])),arms,controls,original)


def test_fullshape_all_eight_same_panel_pair():
    for phase,n in [('cpu',1),('calibration',4)]:
        tasks=b.task_plan(phase)
        assert len(tasks)==16 and all(t['n']==n for t in tasks)
    full=b.task_plan('full')
    assert len(set(b.task_name(t) for t in full))==16
    assert sum(t['n'] for t in full)==8*(58+557)
    assert {t['key'] for t in full}=={b.key(f,r,s) for f in (0,1) for r in (224,320) for s in (42,43)}
    with pytest.raises(ValueError): b.task_plan('other')


def test_idle_not_wall_cap():
    assert b.idle_reason(100000,99999) is None
    assert b.idle_reason(899,0) is None
    assert b.idle_reason(900,0)
    assert b.idle_reason(600,0,(15000,100,80),16376)
    assert b.idle_reason(600,0,(15000,100,150),16376) is None


def test_memory_12gib_and_desktop_headroom():
    from tools.c85r7_flipall_timing import memory_admission
    desktop=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',total_mib=16376,used_mib=1800)
    assert memory_admission(desktop,[12*2**30])==12*2**30
    with pytest.raises(ValueError): memory_admission(desktop,[12*2**30+1])
    with pytest.raises(ValueError): memory_admission({**desktop,'used_mib':3000},[12*2**30])


def test_preexecution_review_and_worker_bypass(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'WORK',tmp_path)
    monkeypatch.setattr(b,'registered',lambda:{})
    with pytest.raises(FileNotFoundError): b.l6()
    monkeypatch.delenv('HB147_LENSES_CONTROLLER',raising=False)
    with pytest.raises(ValueError): b.require_controller('calibration')


def test_review_binding_and_stale_evidence(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'WORK',tmp_path); monkeypatch.setattr(b,'ROOT',tmp_path)
    (tmp_path/'registration.json').write_text('{}')
    (tmp_path/'review.md').write_text('Independent review evidence')
    r=dict(verdict='PASS',independent=True,reviewer='reviewer1',scope=b.TAG,
           registration_sha256=b.sha(tmp_path/'registration.json'),evidence_path='review.md',
           evidence_sha256=b.sha(tmp_path/'review.md'))
    b.review_check(r,b.TAG)
    for changes in ({'reviewer':'root'},{'verdict':'FAIL'},{'registration_sha256':'stale'},{'evidence_sha256':'stale'}):
        with pytest.raises(ValueError): b.review_check({**r,**changes},b.TAG)


def test_partial_fold0_stops_before_output_read(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'BASE',tmp_path)
    calls=[]
    def read(p):
        calls.append(str(p))
        if str(p).endswith('receipt.json'): return dict(reason='running',exit_code=None,survivors=[])
        return {}
    monkeypatch.setattr(b,'read',read)
    with pytest.raises(ValueError,match='not terminal'): b.member(0,320,42)
    assert len(calls)==2 and not any('outputs' in p for p in calls)


def test_journal_identity_finite_and_serving(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'WORK',tmp_path); (tmp_path/'registration.json').write_text('{}')
    t=dict(key='f0-native320-s42',panel='gold58',n=1)
    reg=dict(members={t['key']:dict(checkpoint_sha256='ck',resolution=320)},panels={'gold58':{'ids':['u']}})
    row=dict(identity={**{k:t[k] for k in ('key','panel')},'phase':'cpu'},key=t['key'],panel=t['panel'],ids=['u'],
        checkpoint_sha256='ck',resolution=320,registration_sha256=b.sha(tmp_path/'registration.json'),
        k=36,tta=True,workers=2,batch=1,device='cpu',runtime=dict(policy='strict',precision_fallback_allowed=False),
        probabilities=[[.5]*12])
    b.validate_journal(row,t,'cpu',reg)
    for change in ({'ids':['other']},{'checkpoint_sha256':'foreign'},{'resolution':224},{'k':24},
                   {'probabilities':[[float('nan')]*12]},{'batch':4}):
        with pytest.raises(ValueError): b.validate_journal({**row,**change},t,'cpu',reg)
