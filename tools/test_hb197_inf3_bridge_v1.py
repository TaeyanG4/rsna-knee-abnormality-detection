"""CPU contract tests for one fixed INF3 bridge; no model/data-row output."""
import copy
from fractions import Fraction
import numpy as np
import pandas as pd
import pytest
from tools import hb197_inf3_bridge_v1 as b


def fixture():
    rng=np.random.default_rng(197)
    leaves={f:rng.uniform(.01,.99,(10 if f=='l2' else 5,58,12)).astype(np.float32) for f in b.FAMILIES}
    costs={f:rng.uniform(1,8,(len(a),12)) for f,a in leaves.items()}
    return leaves,costs


def test_inverse_fraction_oracle_endpoints_identity():
    q=np.tile(np.array([0.,.25,.5,.75,1.])[:,None],(1,12))
    w=np.arange(1,13,dtype=float)
    actual=b.inverse_cost(q,w)
    expected=np.asarray([[float(Fraction(str(x))/(int(y)-(int(y)-1)*Fraction(str(x)))) for y in w] for x in q[:,0]])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)
    assert np.array_equal(actual[0],np.zeros(12)) and np.array_equal(actual[-1],np.ones(12))
    assert np.array_equal(b.inverse_cost(q,np.ones(12)),q)
    p=np.linspace(0,1,101)[:,None]*np.ones((1,12))
    weighted=w*p/(1-p+w*p)
    np.testing.assert_allclose(b.inverse_cost(weighted,w),p,atol=1e-14)


@pytest.mark.parametrize('w',[np.zeros(12),np.full(12,np.nan),np.ones(11),np.full(12,.9)])
def test_bad_weights(w):
    with pytest.raises(ValueError): b.inverse_cost(np.ones((3,12))*.5,w)


def test_transform_precedes_mean_and_family_probability_pool():
    a=np.full((4,12),.2); c=np.full((4,12),.8); w=np.full(12,4.)
    assert not np.allclose((b.inverse_cost(a,w)+b.inverse_cost(c,w))/2,b.inverse_cost((a+c)/2,w))
    leaves,costs=fixture()
    transformed={f:np.stack([np.asarray([[float(Fraction(str(float(q)))/(Fraction(str(float(w)))-(Fraction(str(float(w)))-1)*Fraction(str(float(q))))) for q,w in zip(row,cost)] for row in a]) for a,cost in zip(leaves[f],costs[f])]) for f in b.FAMILIES}
    means=[]
    for f,a in transformed.items():
        means.append(((a[:5]+a[5:])/2).mean(0) if f=='l2' else a.mean(0))
    expected=pd.DataFrame(sum(means)/3).rank(method='average',pct=True).to_numpy()
    assert np.array_equal(b.serving_q3(leaves,costs),expected)
    wrong=b.ml.rank(sum(b.ml.rank(a) for a in means)/3)
    assert not np.array_equal(expected,wrong)


def test_rank_ties_permutation_full_roster():
    leaves,costs=fixture()
    for f in leaves: leaves[f][:,1]=leaves[f][:,0]
    out=b.serving_q3(leaves,costs)
    assert np.array_equal(out[0],out[1])
    order=np.random.default_rng(3).permutation(58)
    perm=b.serving_q3({f:a[:,order] for f,a in leaves.items()},costs)
    assert np.array_equal(perm,out[order])
    chunks=np.concatenate([b.serving_q3({f:a[:,s] for f,a in leaves.items()},costs) for s in (slice(0,29),slice(29,58))])
    assert not np.array_equal(chunks,out)


@pytest.mark.parametrize('mode',['missing','extra','count','nan','range','cohort'])
def test_invalid_serving_sources(mode):
    leaves,costs=fixture()
    if mode=='missing': del leaves['dino']
    if mode=='extra': leaves['other']=leaves['dino']
    if mode=='count': leaves['l2']=leaves['l2'][:-1]
    if mode=='nan': leaves['dino'][0,0,0]=np.nan
    if mode=='range': leaves['convnext'][0,0,0]=1.1
    if mode=='cohort': leaves['dino']=leaves['dino'][:,:57]
    with pytest.raises(ValueError): b.serving_q3(leaves,costs)


def entries():
    return [dict(family=f,seed=s,fold=k,source_sha256=f'{i:064x}',sha256=f'{i+100:064x}') for i,(f,s,k) in enumerate(b.EXPECTED)]


@pytest.mark.parametrize('mode',['missing','duplicate','order','fold','seed','source'])
def test_exact_identity_matrix(mode):
    e=entries(); b.validate_entries(e)
    if mode=='missing': e.pop()
    if mode=='duplicate': e[1]=e[0].copy()
    if mode=='order': e[0],e[1]=e[1],e[0]
    if mode=='fold': e[0]['fold']=4
    if mode=='seed': e[0]['seed']=44
    if mode=='source': e[1]['source_sha256']=e[0]['source_sha256']
    with pytest.raises(ValueError): b.validate_entries(e)


@pytest.mark.parametrize('ids,wanted,a',[(['a','a'],['a','b'],np.ones((2,12))),(['a','c'],['a','b'],np.ones((2,12))),(['a','b'],['a','a'],np.ones((2,12))),(['a','b'],['a','b'],np.full((2,12),np.nan))])
def test_bad_alignment(ids,wanted,a):
    with pytest.raises(ValueError): b.align(ids,a,wanted)


def test_alignment_and_target_shape():
    a=np.arange(24).reshape(2,12)/24
    assert np.array_equal(b.align(['b','a'],a,['a','b']),a[::-1])
    with pytest.raises(ValueError): b.probabilities(np.zeros((3,11)))


def test_gate_boundaries_no_ci_requirement():
    d=np.full((4,12),.0005); assert b.gate(d)
    d[3]=-.003; assert b.gate(d)
    d[3,0]=-.00300001; assert not b.gate(d)
    d[:]=.000499; assert not b.gate(d)
    d[0,0]=np.nan
    with pytest.raises(ValueError): b.gate(d)


def test_oof_heldout_fold_aggregation():
    rng=np.random.default_rng(33)
    raw={f:{k:rng.random((6,12)).astype(np.float32) for k in ('R42','R43','DINO','CNX')} for f in range(5)}
    costs={f:{k:np.ones(12) for k in raw[f]} for f in raw}
    q0,q3=b.oof_values(raw,costs)
    expected=[]
    for f in range(5): expected.append(((raw[f]['R42'].astype(float)+raw[f]['R43'].astype(float))/2+raw[f]['DINO']+raw[f]['CNX'])/3)
    assert np.array_equal(q3,b.ml.rank(np.concatenate(expected)))
    assert not np.array_equal(q0,q3)
    del raw[2]['DINO']
    with pytest.raises(ValueError): b.oof_values(raw,costs)


def test_parent_noop_and_mask_oracle():
    files=b.unpack_parent(); ns=b.parent.math_functions(files)
    rng=np.random.default_rng(197); arrays=[rng.random((58,12)) for _ in range(6)]; ids=[f'id{i}' for i in range(58)]
    base=ns['blend'](ids,*arrays)
    assert np.array_equal(base,ns['reference'](*arrays))
    assert ns['p'].csv_bytes(ids,base)==ns['p'].csv_bytes(ids,ns['blend'](ids,*[x.copy() for x in arrays]))
    for scenario in ((),(0,),(1,),(2,),(3,4,5),(0,1,2),(0,1,2,3,4,5)):
        masks=[np.zeros(58,bool) for _ in range(6)]
        for i in scenario: masks[i][0]=True
        failed={name:{ids[i]:'fixture' for i in np.flatnonzero(mask)} for name,mask in zip(b.NAMES,masks)}
        actual=ns['blend'](ids,*arrays,failed)
        assert np.array_equal(actual,b.parent.oracle(arrays,masks))
    with pytest.raises(ValueError): ns['blend'](ids,*arrays,{'R224':dict.fromkeys(ids[:2],'fixture')})


def test_immutable_registration_and_output_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'OUT',tmp_path)
    proof={'members':[1], 'bindings':{'fixed':'abc'}}
    monkeypatch.setattr(b,'authenticate',lambda:(copy.deepcopy(proof),None))
    b.register(); b.register(); b.validate()
    proof['bindings']['fixed']='def'
    with pytest.raises(ValueError): b.validate()
    with pytest.raises(ValueError): b.register()
    with pytest.raises(ValueError): b.save('../escape.json',{})


def test_cpu_model_loading_tripwire():
    import torch
    assert not torch.cuda._initialized
    with pytest.raises(RuntimeError,match='FORBIDDEN'): torch.load('unused.pt')
