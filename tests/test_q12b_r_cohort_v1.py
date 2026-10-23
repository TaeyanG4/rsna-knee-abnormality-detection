"""Q12b v1.0.1 CPU contracts; no real label rows are printed."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
from pathlib import Path
import types
import numpy as np
import pytest
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from tools import q12b_r_cohort_v1 as q


def synthetic():
    rng = np.random.default_rng(81)
    n = 80
    compact = rng.random((n,12))
    arrays = {m:rng.random((n,12)) for m in q.MEMBERS}
    refs = {}
    for r,name in enumerate(('jev_hybrid','aquila_v1','dread','consensus')):
        y = rng.integers(0,2,(n,12)); mask = np.ones_like(y,dtype=bool)
        if name=='consensus': mask[::4] = False
        refs[name] = (y,mask,2 if name=='consensus' else 1)
    return compact,arrays,refs


def test_auc_oracle_ties_and_degenerate():
    y = np.array([0,1,0,1,1,0]); scores = np.array([.1,.3,.3,.3,.8,.8])
    assert q.auc(y,scores) == pytest.approx(roc_auc_score(y,scores))
    assert np.isnan(q.auc(np.zeros(4),np.arange(4)))
    assert np.isnan(q.auc([],[]))


def test_shared_bootstrap_against_independent_sklearn_oracle():
    compact,arrays,refs = synthetic(); draws=12; seed=13
    actual=q.summarize(compact,arrays,refs,draws,seed)
    n=len(compact)
    blend=.65*rankdata(arrays['R384'],axis=0)/n+.35*rankdata(arrays['R224'],axis=0)/n
    values=[]; other=[]; rng=np.random.default_rng(seed)
    for d in range(draws):
        idx=rng.integers(0,n,n); cells=[]; context=[]
        for y,mask,_ in refs.values():
            take=idx[mask[idx,5]]
            cells.append(roc_auc_score(y[take,5],compact[take,5])-roc_auc_score(y[take,5],blend[take,5]))
            for j in q.OTHER5:
                take=idx[mask[idx,j]]
                context.append(roc_auc_score(y[take,j],compact[take,j])-roc_auc_score(y[take,j],blend[take,j]))
        values.append(np.mean(cells)); other.append(np.mean(context))
    out=actual['R_blend']
    assert out['four_ruler_mean']['Lateral OA']['ci95']==pytest.approx(np.quantile(values,[.025,.975]))
    assert out['other5_macro']['ci95']==pytest.approx(np.quantile(other,[.025,.975]))
    for name,(y,mask,_) in refs.items():
        m=mask[:,5]
        assert out['per_ruler'][name]['Lateral OA']['source_auc']==pytest.approx(roc_auc_score(y[m,5],blend[m,5]))
    assert actual==q.summarize(compact,arrays,refs,draws,seed)


def test_evaluation_rejects_nonfinite_and_invalid_masks():
    c,a,r=synthetic(); a['R384'][0,0]=np.nan
    with pytest.raises(ValueError,match='predictions'): q.summarize(c,a,r,2)
    c,a,r=synthetic(); y,m,w=r['consensus']; r['consensus']=(y,m.astype(int),w)
    with pytest.raises(ValueError,match='mask'): q.summarize(c,a,r,2)


@pytest.mark.parametrize('bad_draws',[1,2])
def test_undefined_draws_and_aggregate_requires_every_cell(monkeypatch,bad_draws):
    y=np.tile(np.array([0,1,0,1,0,1])[:,None],(1,12)); mask=np.ones((6,12),bool)
    c=np.tile(np.linspace(.1,.9,6)[:,None],(1,12)); a={m:c.copy() for m in q.MEMBERS}
    refs={name:(y.copy(),mask.copy(),1) for name in ('jev_hybrid','aquila_v1','dread','consensus')}
    refs['consensus'][0][:,5]=[0,0,0,1,0,1]
    class Draws:
        def __init__(self): self.i=0
        def integers(self,*args):
            self.i+=1
            return np.array([0,1,0,1,0,1]) if self.i<=bad_draws else np.arange(6)
    monkeypatch.setattr(np.random,'default_rng',lambda *args:Draws())
    if bad_draws==2:
        with pytest.raises(ValueError,match='too few'): q.summarize(c,a,refs,10)
    else:
        result=q.summarize(c,a,refs,10)
        for source in result.values():
            assert source['per_ruler']['consensus']['Lateral OA']['undefined_draws']==1
            assert source['per_ruler']['jev_hybrid']['Lateral OA']['undefined_draws']==0
            assert source['four_ruler_mean']['Lateral OA']['undefined_draws']==1
            assert source['other5_macro']['undefined_draws']==0
            assert source['all12_macro']['undefined_draws']==1


def test_real_frozen_cohort_and_source_identity():
    ids,compact,refs,fragment=q.fable_context()
    assert len(ids)==len(set(ids))==976 and compact.shape==(976,12)
    assert len(fragment)==64 and set(refs)=={'jev_hybrid','aquila_v1','dread','consensus'}
    files,checkpoints=q.stage_sources()
    assert len(files)>4 and set(checkpoints)==set(q.MEMBERS)


@pytest.mark.parametrize('defect',['reverse','nan','constant','fallback','target','missing'])
def test_prediction_archive_rejections(tmp_path,defect):
    c,a,_=synthetic(); ids=[str(i) for i in range(len(c))]; targets=q.LABELS.copy()
    if defect=='nan': a['R384'][0,0]=np.nan
    if defect=='constant': a['R384'][:,0]=.2
    if defect=='fallback': a['R384'][0]=.5
    if defect=='target': targets.reverse()
    if defect=='missing': del a['R224']
    path=tmp_path/'predictions.npz'
    np.savez(path,ids=ids[::-1] if defect=='reverse' else ids,targets=targets,**a)
    with pytest.raises(ValueError): q.check_predictions(path,{'ids':ids})


def test_valid_archive_and_plan(tmp_path):
    c,a,_=synthetic(); ids=[str(i) for i in range(len(c))]
    path=tmp_path/'predictions.npz'; np.savez(path,ids=ids,targets=q.LABELS,**a)
    assert set(q.check_predictions(path,{'ids':ids}))==set(q.MEMBERS)
    from tools.gpu_queue import validate_plan
    p=validate_plan(q.plan()); assert len(p)==1 and p[0]['after']=='L1_RCO_K0_V2' and p[0]['continue_on_failure'] is True
    assert Path(p[0]['command'][0]).is_absolute() and Path(p[0]['command'][2]).is_absolute()


@pytest.mark.parametrize('change',[{'verdict':'FAIL'},{'reviewer':q.BUILDER},{'binding':{}},{'evaluation_checked':False}])
def test_review_fail_closed(change):
    binding={'test':'digest'}
    review=dict(schema='Q12b-review/v1',verdict='PASS',builder=q.BUILDER,reviewer='independent',binding=binding,
                member_identity_checked=True,cpu_parity_checked=True,evaluation_checked=True,memory_watchdog_checked=True)
    review.update(change)
    with pytest.raises(ValueError): q.check_review(review,binding)


@pytest.mark.parametrize('state',['unknown','running','STALE','BLOCKED'])
def test_dependency_cannot_be_skipped(monkeypatch,state):
    import psutil
    from tools import job_runtime
    monkeypatch.setattr(psutil,'Process',lambda:types.SimpleNamespace(parents=lambda:[types.SimpleNamespace(pid=42)]))
    monkeypatch.setattr(job_runtime,'job_status',lambda logs,tag,**kw:dict(state='running',pid=42) if tag==q.TAG else dict(state=state))
    with pytest.raises(ValueError,match='predecessor'): q.execution_gate()


def test_command_lock_bypass_rejected(monkeypatch):
    from tools import job_runtime
    monkeypatch.setattr(job_runtime,'job_status',lambda *a,**kw:dict(state='unknown'))
    with pytest.raises(ValueError,match='lock'): q.execution_gate()


def test_throughput_boundaries():
    history=list(range(0,110,10))
    q.throughput_guard(history,1899)
    with pytest.raises(ValueError,match='throughput'): q.throughput_guard(history,1900)
    q.throughput_guard(list(range(0,2010,10)),2000)


def test_indexed_glob_preserves_order():
    paths=['archive/train_series/a/b/2.dcm','archive/train_series/a/b/1.dcm','archive/train_series/a/c/9.dcm']
    assert q.indexed_glob(q.series_index(paths),'archive/train_series/a/b/*.dcm')==paths[:2]
    with pytest.raises(ValueError): q.indexed_glob(q.series_index(paths),'archive/train_series/*/b/*.dcm')


def test_reading_rule_and_no_result_go():
    assert 'uninterpretable' in q.READING_RULE and 'EXPOSED' in q.READING_RULE
    assert q.RULES['other5']==[7,9,10,8,11] and q.RULES['draws']==1000
