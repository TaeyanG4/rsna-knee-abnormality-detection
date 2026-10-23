"""v1.0.0 fixed formulas, exact977 integrity and four-ruler selection tests."""
import copy
import numpy as np
import pytest
from tools import k1_pooling_v1 as p


def metric_fixture(deltas=(.002,.002,.002,.002)):
    return dict(rulers={n:dict(delta=d) for n,d in zip(p.RULERS,deltas)},
        per_target=[dict(ruler=n,target=t,delta=.002) for n in p.RULERS for t in p.TARGETS])


@pytest.mark.parametrize('deltas,yes',[
    ((.002,.002,.002,.002),True),((.001999,)*4,False),
    ((.003,.003,.003,-.001),True),((.005,.005,-.001,-.001),False)])
def test_exact_macro_boundaries(deltas,yes):
    assert p.go_rule(metric_fixture(deltas)) is yes


@pytest.mark.parametrize('delta,yes',[(-.005,True),(-.0050001,False),(float('nan'),False),(float('inf'),False)])
def test_every_target_floor(delta,yes):
    m=metric_fixture(); m['per_target'][0]['delta']=delta
    assert p.go_rule(m) is yes


def test_no_missing_duplicate_ruler_target():
    m=metric_fixture(); m['per_target'][-1]=m['per_target'][0]; assert not p.go_rule(m)
    m=metric_fixture(); m['rulers'].pop('dread'); assert not p.go_rule(m)


def test_fixed_pooling_per_target_and_top_min3():
    pooled=np.arange(12,dtype=np.float32)
    windows=np.arange(60,dtype=np.float32).reshape(5,12)
    actual=p.candidate(pooled,windows)
    for j in range(12):
        expected=.5*pooled[j]+.5*(windows[-3:,j].mean() if j in p.FOCAL else windows[:,j].mean())
        assert actual[j]==expected
    for n in (1,2,3):
        np.testing.assert_allclose(p.candidate(pooled,windows[:n]),.5*pooled+.5*windows[:n].mean(0))
    with pytest.raises(ValueError):p.candidate(pooled,windows[:0])
    windows[0,0]=np.nan
    with pytest.raises(ValueError):p.candidate(pooled,windows)


def dump_fixture(tmp_path):
    ids=[str(i) for i in range(977)]
    pooled=np.zeros((977,12),np.float32)
    win=np.zeros((1954,12),np.float32)
    data=dict(ids=np.asarray(ids),pooled=pooled,windows=win,offsets=np.arange(978,dtype=np.int64)*2)
    path=tmp_path/'dump.npz'; np.savez(path,**data)
    return path,ids,data


@pytest.mark.parametrize('fault',['hash','roster','offset','nan','dtype','partial','counts'])
def test_dump_tamper_and_partial_rejected(tmp_path,fault):
    path,ids,data=dump_fixture(tmp_path); counts=[2]*977
    if fault=='roster':data['ids'][0]='bad'
    if fault=='offset':data['offsets'][50]-=1
    if fault=='nan':data['windows'][0,0]=np.nan
    if fault=='dtype':data['pooled']=data['pooled'].astype(np.float64)
    if fault=='partial':data['pooled']=data['pooled'][:-1]
    if fault=='counts':counts[0]=1
    np.savez(path,**data)
    with pytest.raises(ValueError):p.load_dump(path,'bad' if fault=='hash' else p.sha(path),ids,counts)


def test_dump_and_evaluator_self_control(tmp_path):
    path,ids,data=dump_fixture(tmp_path)
    a,b=p.load_dump(path,p.sha(path),ids,[2]*977)
    refs={n:(np.tile((np.arange(977)%2)[:,None],(1,12)),np.ones((977,12),bool),1) for n in p.RULERS}
    m=p.metrics(a,b,refs)
    assert not p.go_rule(m)
    assert all(x['delta']==0 for x in m['rulers'].values())
    assert all(v==0 for d in m['diagnostics'].values() for v in d.values())
    with pytest.raises(ValueError):p.metrics(a[:-1],b[:-1],refs)
    refs['dread'][0][:]=0
    with pytest.raises(ValueError,match='undefined'):p.metrics(a,b,refs)


def test_rank_trio_uses_member_ranks():
    # torch import follows DLL pin; CPU only.
    p.base.import_trainer()
    from tools.c85r7_rco_overlay_v1 import ranks
    a=np.tile(np.linspace(-2,2,977,dtype=np.float32)[:,None],(1,12))
    np.testing.assert_array_equal(p.trio([a,a*2,a*3]),ranks(a))


def test_exact_registered977():
    assert len(p.cohort())==977
    assert set(p.FOCAL).isdisjoint(p.EXTENT) and set(p.FOCAL+p.EXTENT)==set(range(12))
    assert p.RULE['eval_stride']==2 and p.RULE['gpu_seconds_limit']==5400
