"""Q23 functional contracts; never load labels, checkpoints or CUDA."""
import copy
import numpy as np
import pytest
from scipy.stats import rankdata
from tools import q23_exp138_gold58_v1 as q


def test_native_head_rank_then_fold_mean_with_ties():
    raw=np.random.default_rng(23).integers(0,5,(5,9,4,12))/4
    oracle=np.mean([np.mean([rankdata(raw[f,:,h,:],axis=0,method='average')/9
        for h in range(4)],axis=0) for f in range(5)],axis=0)
    assert np.array_equal(q.aggregate(raw),oracle)
    assert not np.array_equal(q.aggregate(raw),np.mean(raw,axis=(0,2)))
    assert np.all(q.aggregate(np.ones((5,1,4,12)))==1)


@pytest.mark.parametrize('kind',['nan','range','heads','folds','empty'])
def test_aggregation_rejects_invalid(kind):
    raw=np.ones((5,9,4,12))*.5
    if kind=='nan': raw[0,0,0,0]=np.nan
    if kind=='range': raw[0,0,0,0]=1.1
    if kind=='heads': raw=raw[:,:,:3]
    if kind=='folds': raw=raw[:4]
    if kind=='empty': raw=raw[:,:0]
    with pytest.raises((ValueError,RuntimeError)): q.aggregate(raw)


def fixture_values():
    rng=np.random.default_rng(24)
    common={n:rng.random((58,12)) for n in q.d.COMMON}
    failed={n:np.zeros(58,dtype=bool) for n in common}
    return [str(i) for i in range(58)],common,failed,rng.random((58,12)),rng.random((58,12)),rng.random((58,12))


def test_final_column_replacement_exact_formulas_and_identity():
    args=fixture_values(); v=q.outputs(*args); candidate=args[-1]
    for base in ('HB167','HB174'):
        for suffix,route in [('EXP138',q.r.rank(candidate)),('halfO',q.r.rank(.5*v['O']+.5*candidate))]:
            actual=v[base+'_'+suffix]
            assert actual[:,10].tobytes()==route[:,10].tobytes()
            assert np.delete(actual,10,1).tobytes()==np.delete(v[base],10,1).tobytes()
    assert len(q.PAIRS)==7
    # Synthetic tie/order shape distinguishes native-score mixing from extra rerank.
    assert not np.array_equal(q.r.rank(.5*v['O']+.5*candidate),q.r.rank(.5*v['O']+.5*q.r.rank(candidate)))


def test_missing_or_nonfinite_baseline_fails():
    args=fixture_values(); args[4][0,0]=np.nan
    with pytest.raises((ValueError,RuntimeError)): q.outputs(*args)


def test_bootstrap_identity_pair_and_strict_zero_floor():
    y=np.tile((np.arange(58)%2)[:,None],(1,12)); p=np.tile(np.arange(58)[:,None]/58,(1,12))
    v={'HB167_EXP138':p,'HB167':p.copy()}
    result=q.paired_metrics(y,v,[('HB167_EXP138','HB167')])
    contrast=result['comparisons']['HB167_EXP138-minus-HB167']
    assert result['bootstrap']['valid_draws']==2000
    assert contrast['contusion']['ci95']==[0.,0.]
    assert contrast['macro12']['delta']==0
    assert contrast['other_targets_identical'] and contrast['operational']=='FAIL'
    assert not result['qualified_for_submission']


def test_bootstrap_positive_target_macro_identity_and_mutation_rejected():
    y=np.tile((np.arange(58)%2)[:,None],(1,12)); p=np.tile(np.arange(58)[:,None]/58,(1,12))
    improved=p.copy(); improved[:,10]=y[:,10]
    v={'HB167_EXP138':improved,'HB167':p}
    contrast=q.paired_metrics(y,v,[('HB167_EXP138','HB167')])['comparisons']['HB167_EXP138-minus-HB167']
    assert contrast['operational']=='PASS'
    assert contrast['macro12']['delta']==pytest.approx(contrast['contusion']['delta']/12)
    improved[:,0]=y[:,0]
    contrast=q.paired_metrics(y,v,[('HB167_EXP138','HB167')])['comparisons']['HB167_EXP138-minus-HB167']
    assert contrast['operational']=='FAIL'


def test_projection_threshold_and_no_gpu_division():
    cal={'scale':69.40676358664376,'assumption':'cross-model'}
    limit=2.27*3600*58/32/cal['scale']
    a=q.projection(limit,cal); b=q.projection(limit+1,cal)
    assert a['incremental_hours']==pytest.approx(2.27)
    assert a['projected_total_hours']==pytest.approx(9.)
    assert not b['fits_historical_headroom']
    assert a['implied_same_count_t4_factor']==pytest.approx(1.7084741805943078)
    with pytest.raises((ValueError,RuntimeError)): q.projection(float('nan'),cal)


def test_cpu_environment_present_empty(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    with pytest.raises((ValueError,RuntimeError)): q.c.cpu_only()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',''); q.c.cpu_only()


def test_launch_no_gpu_and_exact_command():
    assert q.launch_command()==[str(q.t.GPU_PYTHON),'-B',str(q.SCRIPT),'run']
    plan=q.gpu_queue.validate_plan([dict(tag=q.TAG,command=q.launch_command())])
    assert plan is not None


def test_prereview_rejects_self_review_and_binding_drift(monkeypatch):
    review=dict(verdict='PASS',builder=q.BUILDER,reviewer='independent',binding_sha256='abc',scope='CPU_IMPLEMENTATION_TESTS_PARITY')
    monkeypatch.setattr(q.io,'read_json',lambda path:review)
    q.prereview({'binding_sha256':'abc'})
    review['reviewer']=q.BUILDER
    with pytest.raises((ValueError,RuntimeError)): q.prereview({'binding_sha256':'abc'})
    review['reviewer']='independent'
    with pytest.raises((ValueError,RuntimeError)): q.prereview({'binding_sha256':'changed'})


@pytest.mark.parametrize('fault',[None,'missing_review','self_review','binding','validation','receipt','attempt'])
def test_gpu_review_admission_fail_closed(monkeypatch,tmp_path,fault):
    reg={'binding_sha256':'bound'}
    valid=dict(state='CPU_VALIDATED',binding_sha256='bound',evidence={'cpu-smoke':'receipt-hash'})
    review=dict(verdict='PASS',builder=q.BUILDER,reviewer='independent',
        execution_authority='B',tag=q.TAG,binding_sha256='bound',validation_sha256='validate-hash')
    if fault=='self_review': review['reviewer']=q.BUILDER
    if fault=='binding': review['binding_sha256']='wrong'
    if fault=='validation': review['validation_sha256']='wrong'
    if fault=='attempt': (tmp_path/'attempt.json').write_text('{}')
    def read(path):
        if path.name=='gpu-review.json':
            if fault=='missing_review': raise FileNotFoundError(path)
            return review
        assert path.name=='validate.json'
        return valid
    def hash_file(path):
        if path.name=='validate.json': return 'validate-hash'
        assert path.name=='cpu-smoke.json'
        return 'changed' if fault=='receipt' else 'receipt-hash'
    monkeypatch.setattr(q,'WORK',tmp_path)
    monkeypatch.setattr(q,'current',lambda:reg)
    monkeypatch.setattr(q,'prereview',lambda _:None)
    monkeypatch.setattr(q.io,'read_json',read)
    monkeypatch.setattr(q,'sha',hash_file)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    if fault is None:
        assert q.launch_check()['state']=='REVIEW_BOUND_LAUNCH_CHECK_PASS_NO_ADMISSION'
    else:
        with pytest.raises((ValueError,RuntimeError,FileNotFoundError)): q.launch_check()


def test_standalone_readout_is_contusion_only():
    y=np.tile((np.arange(58)%2)[:,None],(1,12)); p=np.tile(np.arange(58)[:,None]/58,(1,12))
    result=q.paired_metrics(y,{'EXP138':p,'O':p.copy()},[('EXP138','O')])
    assert set(result['comparisons']['EXP138-minus-O'])=={'contusion'}
    assert 'auc' not in result and set(result['contusion_auc'])=={'EXP138','O'}
