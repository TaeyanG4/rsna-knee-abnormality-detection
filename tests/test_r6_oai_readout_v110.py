"""Confirmation partition, fixed statistics and authenticated cache gates."""
import ast
import copy
from pathlib import Path
import numpy as np
import pytest
from tools import r6_oai_readout_v110 as r


def fixture(n=96):
    rng=np.random.default_rng(189)
    y=np.full((n,12),np.nan)
    for j in r.MAPPED:
        y[:,j]=rng.permutation(np.arange(n)%2).astype(float)
        y[rng.choice(n,n//10,replace=False),j]=np.nan
    return y,{k:rng.choice(np.linspace(0,1,9),(n,12)) for k in r.ARMS}


def test_frozen_v1_and_execution_functions_identical():
    assert r.sha(Path(r.pilot.__file__))==r.PILOT_TOOL_SHA
    old=ast.parse(Path(r.pilot.__file__).read_text(encoding='utf-8'))
    new=ast.parse(Path(r.__file__).read_text(encoding='utf-8'))
    funcs=lambda tree:{x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    a,b=funcs(old),funcs(new)
    for name in ('infer','serving','series_probe','verify_series','configure','worker','run',
                 'check_measurement','admit_memory','original_group','contract_files','export_contracts',
                 'bootstrap','outputs','check_review'):
        assert a[name]==b[name],name


def test_complement_exact_and_order_independent():
    ids=[f'OAI_{i}_L' for i in range(557)]
    old=r.pilot.select_ids(ids); new=r.select_ids(ids)
    assert len(new)==461 and not set(old)&set(new)
    assert sorted(old+new)==sorted(ids)
    assert new==r.select_ids(ids[::-1])
    with pytest.raises(ValueError): r.select_ids(ids[:-1])
    with pytest.raises(ValueError): r.select_ids(ids[:-1]+[ids[0]])
    with pytest.raises(ValueError,match='participant'): r.select_ids(ids[:-1]+['OAI_0_R'])


@pytest.mark.parametrize('n',[96,461,557])
def test_masked_bootstrap_expanded_sklearn_oracle(n):
    from sklearn.metrics import roc_auc_score
    y,v=fixture(n); eligible=r.label_contract(y)['eligible']
    boot,w,_=r.bootstrap(y,list(v.values()),eligible,draws=5)
    for k,value in enumerate(v.values()):
        for b,weights in enumerate(w):
            ix=np.repeat(np.arange(n),weights.astype(int))
            for q,j in enumerate(eligible):
                chosen=ix[np.isfinite(y[ix,j])]
                assert boot[k,b,q]==pytest.approx(roc_auc_score(y[chosen,j],value[chosen,j]))
    assert np.array_equal(w,r.bootstrap(y,list(v.values()),eligible,draws=5)[1])


def test_fixed_macro7_excludes_extra_eligible_targets():
    y,_=fixture(); good=np.nan_to_num(y,nan=.5)
    a=good.copy(); a[:,[0,4]]=1-a[:,[0,4]]
    v={'R7flip':good,'R6flip':a}
    got=r.metrics(y,v,r.label_contract(y),draws=10)
    c=got['comparisons']['R6flip-minus-R7flip']
    assert got['eligible']==r.MAPPED and got['macro_targets']==r.MACRO7
    assert c['macro_available']['ci95']==[0,0]
    assert c['weak_four']['ci95']==[0,0]
    assert c['per_target_delta']['0']==-1 and c['per_target_delta']['4']==-1


def test_optional_targets_cannot_change_primary_2000_draws():
    y,v=fixture(30)
    y[:,[0,4]]=np.nan
    base=r.metrics(y,v,r.label_contract(y))
    y[:10,0]=[0]*5+[1]*5
    y[10:20,4]=[0]*5+[1]*5
    extra=r.metrics(y,v,r.label_contract(y))
    repeat=r.metrics(y,v,r.label_contract(y))
    for key in ('macro_available','weak_four','point_floors'):
        assert base['comparisons']['R6flip-minus-R7flip'][key]==extra['comparisons']['R6flip-minus-R7flip'][key]
    assert base['attempts']==extra['attempts']
    assert base['valid_draws']==extra['valid_draws']==2000
    assert extra==repeat


@pytest.mark.parametrize('missing',[2,8])
def test_missing_primary_is_incomplete_never_reduced_macro(missing):
    y,v=fixture(); y[:,missing]=np.nan
    v['R6flip']=v['R7flip'].copy()
    got=r.metrics(y,v,r.label_contract(y),draws=5)['comparisons']['R6flip-minus-R7flip']
    assert got['macro_available'] is None and got['point_floors']=='INCOMPLETE'
    assert (got['weak_four'] is None)==(missing==8)


def test_original_floors_and_harm():
    y,_=fixture(); good=np.nan_to_num(y,nan=.5)
    c=r.metrics(y,{'R7flip':good,'R6flip':1-good},r.label_contract(y),draws=5)['comparisons']['R6flip-minus-R7flip']
    assert c['point_floors']=='FAIL' and c['macro_available']['statistical']=='HARM'
    assert r.RULES['macro_floor']==-.001 and r.RULES['weak_four_floor']==-.002
    assert r.RULES['draws']==2000 and r.RULES['seed']==174175


@pytest.mark.parametrize('damage',['nan','inf','range','shape','label','unmapped'])
def test_invalid_mask_or_prediction_rejected(damage):
    y,v=fixture()
    if damage=='nan': v['R6flip'][0,0]=np.nan
    if damage=='inf': v['R6flip'][0,0]=np.inf
    if damage=='range': v['R6flip'][0,0]=2
    if damage=='shape': v['R6flip']=v['R6flip'][:-1]
    if damage=='label': y[0,0]=2
    if damage=='unmapped': y[0,1]=0
    with pytest.raises(ValueError): r.metrics(y,v,r.label_contract(y),draws=2)


def test_pilot_registration_hash_before_deserialization(monkeypatch):
    monkeypatch.setattr(r,'sha',lambda _: 'wrong')
    monkeypatch.setattr(r.io,'read_json',lambda _:pytest.fail('must not read unsealed pilot'))
    with pytest.raises(ValueError,match='frozen'): r.pilot_registration()


@pytest.mark.parametrize('tampered',['gpu-review','validate','cpu-smoke','prepare','replay'])
def test_pilot_cpu_oracle_chain_rejects_pre_registration_tamper(monkeypatch,tampered):
    reg=r.pilot_registration()
    original=r.sha
    def altered(path):
        return 'tampered' if Path(path)==r.PILOT/(tampered+'.json') else original(path)
    monkeypatch.setattr(r,'sha',altered)
    with pytest.raises(ValueError): r.pilot_registration()


def test_pilot_empty_cpu_file_manifest_rejected_before_read(monkeypatch):
    original_sha=r.sha; original_read=r.io.read_json
    def altered_hash(path):
        return 'empty-manifest-hash' if Path(path)==r.PILOT/'cpu-smoke.json' else original_sha(path)
    def guarded_read(path):
        assert Path(path)!=r.PILOT/'cpu-smoke.json','must reject tampered receipt before deserializing'
        return original_read(path)
    monkeypatch.setattr(r,'sha',altered_hash); monkeypatch.setattr(r.io,'read_json',guarded_read)
    with pytest.raises(ValueError,match='CPU evidence chain'): r.pilot_registration()


def test_frozen_pilot_exact_reproduction_gate(monkeypatch):
    comparison={k:{'value':1} for k in ('macro_available','weak_four','point_floors','per_target_delta')}
    result={'comparisons':{'R6flip-minus-R7flip':comparison}}
    frozen={'measurement_sha256':'sealed','result':copy.deepcopy(result)}
    monkeypatch.setattr(r.io,'read_json',lambda _:frozen)
    monkeypatch.setattr(r,'sha',lambda _:'sealed')
    assert r.verify_frozen_pilot(result)==frozen
    for key in comparison:
        bad=copy.deepcopy(result); bad['comparisons']['R6flip-minus-R7flip'][key]={'value':2}
        with pytest.raises(ValueError,match='frozen pilot'): r.verify_frozen_pilot(bad)


def test_pilot_artifact_seal_before_prediction_read(tmp_path,monkeypatch):
    from tools import job_runtime
    monkeypatch.setattr(r,'PILOT',tmp_path)
    monkeypatch.setattr(r,'pilot_registration',lambda:{})
    monkeypatch.setattr(job_runtime,'job_status',lambda *a,**k:dict(state='SUCCESS'))
    monkeypatch.setattr(r,'sha',lambda _: 'actual')
    def read(path):
        assert path.name=='measurement.json','prediction read before seal validation'
        return dict(attempt_sha256='wrong',review_sha256='actual')
    monkeypatch.setattr(r.io,'read_json',read)
    with pytest.raises(ValueError,match='seal'): r.authenticated_pilot({})


def test_bound_independent_review():
    reg={'binding_sha256':'b'}
    rev=dict(verdict='PASS',reviewer='independent',builder=r.BUILDER,tag=r.TAG,execution_authority='B',binding_sha256='b',validation_sha256='v')
    r.check_review(rev,reg,'v')
    for key,value in [('reviewer',r.BUILDER),('verdict','HOLD'),('binding_sha256','x'),('validation_sha256','x'),('tag',r.pilot.TAG)]:
        with pytest.raises(ValueError): r.check_review(dict(rev,**{key:value}),reg,'v')


def test_cohort_alignment_and_stack_fail_closed(monkeypatch):
    pids=['a','c']; cids=['b','d']; allids=['a','b','c','d']
    reg={'binding':dict(pilot_ids=pids,ids=cids,pooled_ids=allids)}
    def arm(ids,base):
        return dict(ids=ids,predictions=np.full((2,12),base).tolist(),desktop={'uuid':'same'},runtime={'recipe':1},torch='same')
    old={'arms':{a:arm(pids,.2) for a in r.ARMS}}
    new={'arms':{a:arm(cids,.8) for a in r.ARMS}}
    monkeypatch.setattr(r.panel,'outcomes',lambda ids:ids)
    def metric(ids,pred,contract,draws=None):
        for a in r.ARMS:
            assert np.asarray(pred[a])[:,0].tolist()==[.2 if u in pids else .8 for u in ids]
        return ids
    monkeypatch.setattr(r,'metrics',metric)
    assert r.cohort_results(reg,old,new,dict(pilot={},confirmation={},pooled={}))['pooled']==allids
    bad=copy.deepcopy(new); bad['arms']['R6flip']['desktop']['uuid']='other'
    with pytest.raises(ValueError,match='software'): r.cohort_results(reg,old,bad,{})
    bad=copy.deepcopy(new); bad['arms']['R6flip']['ids']=cids[::-1]
    with pytest.raises(ValueError,match='order'): r.cohort_results(reg,old,bad,{})
    with pytest.raises(ValueError,match='union'): r.cohort_results({'binding':dict(pilot_ids=pids,ids=pids,pooled_ids=allids)},old,new,{})


def test_no_gpu_worker_without_controller(monkeypatch):
    monkeypatch.delenv('HB164_CONTROLLER_PID',raising=False)
    with pytest.raises(ValueError,match='supervised'): r.worker()


def test_launch_rejects_missing_review(monkeypatch):
    monkeypatch.setattr(r,'review_ready',lambda:(_ for _ in ()).throw(ValueError('no review')))
    with pytest.raises(ValueError,match='review'): r.launch_check()
