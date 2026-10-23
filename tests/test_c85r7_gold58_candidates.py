"""v1.0.7 candidate harness: math, provenance, fail-closed launch and real replay."""
import copy
import os
import numpy as np
import pytest
from tools import c85r7_gold58_candidates as c


def synthetic():
    rng=np.random.default_rng(71)
    labels=np.column_stack([rng.permutation(np.arange(58)%2) for _ in range(12)])
    values={n:rng.choice(np.linspace(0,1,13),size=(58,12)) for n in ('plain','HB167','HB168','HB170','HB172')}
    return labels,values


def test_cpu_empty_environment():
    assert 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES']==''
    c.cpu_only()


@pytest.mark.parametrize('env',[None,'-1','0'])
def test_cpu_guard_rejects_missing_or_nonempty(monkeypatch,env):
    if env is None: monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    else: monkeypatch.setenv('CUDA_VISIBLE_DEVICES',env)
    with pytest.raises(ValueError,match='present-empty'): c.cpu_only()


def test_bootstrap_matches_independent_sklearn_tie_aware_oracle(monkeypatch):
    from sklearn.metrics import roc_auc_score
    y,outputs=synthetic()
    monkeypatch.setitem(c.RULES,'draws',17)
    actual=c.paired_metrics(y,outputs)
    rng=np.random.default_rng(c.RULES['seed']); draws=[]
    while len(draws)<17:
        idx=rng.integers(0,58,58)
        if any(len(np.unique(y[idx,j]))<2 for j in range(12)): continue
        draws.append([roc_auc_score(y[idx,j],outputs['HB168'][idx,j])-
                      roc_auc_score(y[idx,j],outputs['HB167'][idx,j]) for j in range(12)])
    a=np.asarray(draws); got=actual['comparisons']['HB168-minus-HB167']
    assert np.allclose(got['macro_ci95'],np.quantile(a.mean(1),[.025,.975]))
    assert np.allclose(got['weak_four_ci95'],np.quantile(a[:,list(c.d.gold.WEAK)].mean(1),[.025,.975]))


def test_fixed_2000_deterministic_and_identity():
    y,outputs=synthetic(); outputs['HB170']=outputs['HB167'].copy()
    a=c.paired_metrics(y,outputs); b=c.paired_metrics(y,outputs)
    assert a==b and a['bootstrap']['valid_draws']==2000 and c.RULES['seed']==167170172
    pair=a['comparisons']['HB170-minus-HB167']
    assert pair['macro_delta']==pair['weak_four_delta']==0
    assert pair['macro_ci95']==pair['weak_four_ci95']==[0.,0.]


def test_harm_floor_blocks_go(monkeypatch):
    y,_=synthetic(); monkeypatch.setitem(c.RULES,'draws',8)
    result=c.paired_metrics(y,dict(plain=y.copy(),HB167=y.copy(),HB168=1-y))
    for v in result['comparisons'].values():
        assert v['macro_delta']==v['weak_four_delta']==-1 and not v['no_harm_pass']
    assert not result['qualified_for_submission']


def test_degenerate_draws_rejected_and_reported(monkeypatch):
    y,p=synthetic(); y[:]=0; y[0,:]=1
    monkeypatch.setitem(c.RULES,'draws',10)
    r=c.paired_metrics(y,p)['bootstrap']
    assert r['rejected']>0 and r['attempts']==r['valid_draws']+r['rejected']


@pytest.mark.parametrize('damage',['nan','outofbounds','shape','labels','degenerate'])
def test_invalid_metric_inputs(damage):
    y,p=synthetic()
    if damage=='nan': p['HB168'][0,0]=np.nan
    elif damage=='outofbounds': p['HB168'][0,0]=2
    elif damage=='shape': p['HB168']=p['HB168'][:-1]
    elif damage=='labels': y[0,0]=2
    else: y[:,0]=0
    with pytest.raises(ValueError): c.paired_metrics(y,p)


def review_fixture():
    reg=dict(binding_sha256='binding',binding=dict(differences=['explicit']))
    review=dict(verdict='PASS',builder=c.BUILDER,reviewer='astra-4',execution_authority='B',
                binding_sha256='binding',validation_sha256='validation',tag=c.TAG,approved_differences=['explicit'])
    return reg,review


def test_independent_bound_review_accepts():
    reg,r=review_fixture(); c.check_review(r,reg,'validation')


@pytest.mark.parametrize('field,value',[('reviewer','astra-b1'),('reviewer',''),('reviewer',None),
    ('verdict','FAIL'),('builder','other'),('execution_authority','builder'),('binding_sha256','old'),
    ('validation_sha256','old'),('tag','old'),('approved_differences',[])])
def test_review_bypass_rejected(field,value):
    reg,r=review_fixture(); r[field]=value
    with pytest.raises(ValueError,match='independent'): c.check_review(r,reg,'validation')


def test_admit_dry_run_does_not_create_admission(monkeypatch,tmp_path):
    reg,r=review_fixture(); monkeypatch.setattr(c,'WORK',tmp_path)
    monkeypatch.setattr(c,'current',lambda:reg); monkeypatch.setattr(c,'validation_current',lambda _: 'validation')
    p=tmp_path/'synthetic-review.json'; c.io.atomic_json(p,r)
    value=c.admit(p,True)
    assert value['state']=='DRY_RUN_PASS' and not (tmp_path/'admission.json').exists()


def test_admit_dry_run_no_review_blocks(monkeypatch):
    reg,_=review_fixture(); monkeypatch.setattr(c,'current',lambda:reg)
    monkeypatch.setattr(c,'validation_current',lambda _: 'validation')
    with pytest.raises(ValueError,match='even for dry-run'): c.admit(None,True)


def measured_fixture():
    ids=[str(i) for i in range(58)]
    rows=[dict(study=i,probabilities=[.6]*12,windows=94,resolution=224,failed_study_fallback=False,
               actual_pair=True,retain_until_replacement=True,forward_precision=[['float16'],['float16']],
               seconds=1,peak_allocated_bytes=2**30,peak_reserved_bytes=2*2**30) for i in ids]
    value=c.stamp(state='COMPLETE',done=58,binding_sha256='b',rows=rows,seconds=60,device_uuid='gpu')
    desktop=dict(uuid='gpu',device_name='NVIDIA GeForce RTX 4070 Ti SUPER',used_mib=1024,total_mib=16376)
    telemetry=dict(memory_failure=None,peak_gpu_mib=3072)
    return value,ids,desktop,telemetry


def test_measured_pair_validation():
    v,ids,desktop,t=measured_fixture(); c.validate_measurement(v,ids,'b',desktop,t)


@pytest.mark.parametrize('damage',['duplicate','order','nan','fallback','shape','unpaired','precision','memory',
    'headroom','binding','time','device','telemetry','incomplete'])
def test_measurement_rejects_corrupt_pair(damage):
    v,ids,desktop,t=measured_fixture(); r=v['rows'][0]
    if damage=='duplicate': r['study']=ids[1]
    elif damage=='order': v['rows'].reverse()
    elif damage=='nan': r['probabilities'][0]=float('nan')
    elif damage=='fallback': r['failed_study_fallback']=True
    elif damage=='shape': r['windows']=93
    elif damage=='unpaired': r['actual_pair']=False
    elif damage=='precision': r['forward_precision']=[['float16']]
    elif damage=='memory': r['peak_reserved_bytes']=13*2**30
    elif damage=='headroom': desktop['used_mib']=15000
    elif damage=='binding': v['binding_sha256']='old'
    elif damage=='time': v['seconds']=1
    elif damage=='device': v['device_uuid']='other'
    elif damage=='telemetry': t['memory_failure']='memory limit'
    else: v['state']='INFERENCE'
    with pytest.raises(ValueError): c.validate_measurement(v,ids,'b',desktop,t)


def test_worker_rejects_unsupervised_before_gpu(monkeypatch):
    monkeypatch.delenv('HB164_CONTROLLER_PID',raising=False)
    monkeypatch.setattr(c,'admitted',lambda:pytest.fail('admission reached'))
    with pytest.raises(ValueError,match='supervised'): c.worker()


def test_run_no_admission_never_reaches_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'WORK',tmp_path); monkeypatch.setattr(c,'current',lambda:dict())
    monkeypatch.setattr(c,'install_supervisor',lambda:pytest.fail('GPU reached'))
    with pytest.raises(OSError): c.run()


def test_no_dryrun_gpu_bypass(monkeypatch):
    monkeypatch.setattr(c,'run',lambda:pytest.fail('GPU reached'))
    assert c.main(['run','--dry-run'])==2


def test_main_delegates_new_actions_without_changing_old(monkeypatch):
    monkeypatch.setattr(c,'main',lambda args:args)
    assert c.d.main(['candidates-evaluate','--part','a'])==['evaluate','--part','a']
    assert c.d.command('gold')[-2:]==['--mode','gold']


def test_real_admitted_sources_and_actual_preserved_blend_paths():
    reg,arrays,failed,arms=c.context(); files,proof=c.authenticated_sources()
    ids=reg['binding']['ids']
    out=c.blends(ids,arrays,failed,arms,files,arrays['R224'])
    assert np.array_equal(out['HB167'],out['HB170']) and np.array_equal(out['HB168'],out['HB172'])
    assert len(proof['embedded']['fluid'])==len(proof['embedded']['mirror'])==31
    assert set(np.where(np.any(out['HB168']!=out['HB167'],axis=0))[0])<=set([7,9])
    broken=copy.deepcopy(failed); broken['R224']=broken['R224'].astype(int)
    with pytest.raises(ValueError,match='boolean'): c.blends(ids,arrays,broken,arms,files)


def test_actual_admitted_namespace_no_cuda():
    ns=c.namespace()
    assert ns['K_EVAL']==94 and len(ns['LAB'])==12
    assert not ns['torch'].cuda.is_initialized()


def test_real_common_measurement_schema_control():
    reg,arrays,_,_=c.context()
    result=c.replay_measurement(reg,arrays)
    assert result['state']=='PASS_SCHEMA_CONTROL' and result['actual_mirror'] is False
