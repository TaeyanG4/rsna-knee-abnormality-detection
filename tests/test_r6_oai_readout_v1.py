"""Masked OAI paired contrast, exact formula and fail-closed provenance gates."""
import copy
import hashlib
import json
import types
import numpy as np
import pytest
from tools import r6_oai_readout_v1 as r


def fixture():
    rng=np.random.default_rng(189)
    y=np.full((96,12),np.nan)
    for j in r.MAPPED:
        y[:,j]=rng.permutation(np.arange(96)%2).astype(float)
        y[rng.choice(96,9,replace=False),j]=np.nan
    return y,{k:rng.choice(np.linspace(0,1,9),(96,12)) for k in r.ARMS}


def test_selection_independent_of_order_and_unique_participants():
    ids=[f'OAI_{i}_L' for i in range(557)]
    assert r.select_ids(ids)==r.select_ids(ids[::-1])
    assert len(r.select_ids(ids))==96
    with pytest.raises(ValueError): r.select_ids(ids+[ids[0]])


def test_masked_paired_bootstrap_sklearn_oracle():
    from sklearn.metrics import roc_auc_score
    y,v=fixture(); eligible=r.label_contract(y)['eligible']
    actual,w,_=r.bootstrap(y,list(v.values()),eligible,draws=13)
    for k,value in enumerate(v.values()):
        for b,weights in enumerate(w):
            idx=np.repeat(np.arange(96),weights.astype(int))
            for q,j in enumerate(eligible):
                chosen=idx[np.isfinite(y[idx,j])]
                assert actual[k,b,q]==pytest.approx(roc_auc_score(y[chosen,j],value[chosen,j]))
    again=r.bootstrap(y,list(v.values()),eligible,draws=13)
    assert np.array_equal(w,again[1])
    assert np.array_equal(actual,again[0])


def test_eligibility_fixed_missing_weak_and_identity():
    y,v=fixture(); y[:,8]=np.nan; y[:4,8]=1; y[4:8,8]=0
    contract=r.label_contract(y)
    assert 8 not in contract['eligible']
    v['R6flip']=v['R7flip'].copy()
    value=r.metrics(y,v,contract,draws=20)
    result=value['comparisons']['R6flip-minus-R7flip']
    assert result['macro_available']['ci95']==[0,0]
    assert result['weak_four'] is None and result['point_floors']=='INCOMPLETE'
    assert value['decision']=='HOLD_PENDING_INDEPENDENT_RESULT_REVIEW'
    y[9,8]=1
    with pytest.raises(ValueError,match='drift'): r.metrics(y,v,contract,draws=20)


@pytest.mark.parametrize('damage',['nan_prediction','inf_prediction','bounds','shape','bad_label','posinf_label','neginf_label','unmapped','no_eligible'])
def test_invalid_inputs(damage):
    y,v=fixture()
    if damage=='nan_prediction': v['R6flip'][0,0]=np.nan
    if damage=='inf_prediction': v['R6flip'][0,0]=np.inf
    if damage=='bounds': v['R6flip'][0,0]=1.1
    if damage=='shape': v['R6flip']=v['R6flip'][:-1]
    if damage=='bad_label': y[0,0]=2
    if damage=='posinf_label': y[0,0]=np.inf
    if damage=='neginf_label': y[0,0]=-np.inf
    if damage=='unmapped': y[0,1]=1
    if damage=='no_eligible': y[:]=np.nan
    with pytest.raises(ValueError): r.metrics(y,v,r.label_contract(y),draws=2)


def test_bootstrap_cap(monkeypatch):
    y,v=fixture(); y[:,0]=0; y[0,0]=1
    monkeypatch.setitem(r.RULES,'max_attempts',1)
    with pytest.raises(ValueError,match='cap'): r.bootstrap(y,list(v.values()),[0],draws=2)


def test_exact_full_blend_and_missing_members():
    _,v=fixture(); rng=np.random.default_rng(1)
    common={k:rng.random((96,12)) for k in r.COMMON}
    values=r.outputs(common,v['R7flip'],v['R6flip'])
    for name,slot in [('HB167','R7flip'),('replacement','R6flip')]:
        oracle=r.d.formula.blend([str(i) for i in range(96)],common['R'],common['R224'],v[slot],common['EXP143v2'],common['EXP155'],common['Q_A5q'])
        assert np.array_equal(values[name],oracle)
    assert set(r.outputs(None,v['R7flip'],v['R6flip']))==set(r.ARMS)
    common.pop('R')
    with pytest.raises(ValueError): r.outputs(common,v['R7flip'],v['R6flip'])


def test_d_b_201_floors_and_harm():
    y,v=fixture(); good=np.nan_to_num(y,nan=.5)
    v=dict(R7flip=good,R6flip=1-good)
    got=r.metrics(y,v,r.label_contract(y),draws=10)['comparisons']['R6flip-minus-R7flip']
    assert got['point_floors']=='FAIL'
    assert got['macro_available']['statistical']=='HARM'
    assert r.RULES['macro_floor']==-.001 and r.RULES['weak_four_floor']==-.002


def test_review_bound_and_builder_distinct():
    reg={'binding_sha256':'b'}
    review=dict(verdict='PASS',reviewer='independent',builder=r.BUILDER,tag=r.TAG,execution_authority='B',binding_sha256='b',validation_sha256='v')
    r.check_review(review,reg,'v')
    for k,value in [('reviewer',r.BUILDER),('verdict','HOLD'),('binding_sha256','x'),('validation_sha256','x'),('execution_authority','builder')]:
        wrong=dict(review,**{k:value})
        with pytest.raises(ValueError): r.check_review(wrong,reg,'v')


def test_series_probe_fail_closed(tmp_path):
    source='def probe(self, uid):\n    if True:\n        if True:\n            item = self.builder.build(uid)\n'
    from pathlib import Path
    ns=dict(Path=Path,json=json,hashlib=hashlib)
    exec(r.series_probe(source,{'u':['a','b','c','d']},tmp_path/'series-proof'),ns)
    builder=types.SimpleNamespace(build=lambda _:dict(blocks=[dict(series=x) for x in 'abcd']))
    ns['probe'](types.SimpleNamespace(builder=builder),'u')
    r.verify_series(tmp_path,{'u':['a','b','c','d']})
    builder.build=lambda _:dict(blocks=[dict(series=x) for x in 'abc'])
    with pytest.raises(ValueError,match='missing'): ns['probe'](types.SimpleNamespace(builder=builder),'u')
    with pytest.raises(ValueError,match='drift'): r.series_probe('no marker',{},tmp_path)


def test_serving_scoped_on_error(monkeypatch):
    old=(r.g.WORK,r.g.BUNDLE,r.g.MANIFEST_SHA,r.g.port_module,r.d.gold.GOLD_DICOM)
    with pytest.raises(RuntimeError):
        with r.serving('R7flip'): raise RuntimeError('test')
    assert old==(r.g.WORK,r.g.BUNDLE,r.g.MANIFEST_SHA,r.g.port_module,r.d.gold.GOLD_DICOM)


def test_gpu_worker_requires_ownership(monkeypatch):
    monkeypatch.delenv('HB164_CONTROLLER_PID',raising=False)
    with pytest.raises(ValueError,match='supervised'): r.worker()


def test_actual_contract_identity():
    files=r.contract_files()
    for arm in r.ARMS:
        assert files[arm]['exp169_r7_port.py']==r.d.stage_files(r.STAGES[arm])['exp169_r7_port.py']
        assert r.sha(r.BUNDLES[arm]/'manifest.json')==r.MANIFESTS[arm]


def test_real_measurement_adapter_supervisor_seconds(tmp_path):
    import time
    cuda=types.SimpleNamespace(synchronize=lambda:None,max_memory_reserved=lambda:1024,max_memory_allocated=lambda:512,get_device_name=lambda:'fixture4070')
    rows=[]; path=tmp_path/'status.json'
    for uid in ('a','b'):
        r.d.commit(types.SimpleNamespace(cuda=cuda),path,rows,uid,96,192,time.perf_counter()-.1,0,0,0)
    status=r.io.read_json(path)
    assert status['done']==2
    assert sum(x['seconds'] for x in status['rows'])>0
    assert all(x['seconds']==x['study_seconds'] for x in status['rows'])
    files=r.contract_files()['R6flip']
    from tools import exp169_r7_port as port
    src=tmp_path/'src'; src.mkdir()
    for n,b in files.items():
        if n.startswith('exp169_r7_src/'):
            (src/n.split('/')[-1]).write_bytes(b)
    runtime=port.diagnostic_source(src,tmp_path/'runtime')
    instrumented=r.d.instrument((runtime/'compact_infer.py').read_text(encoding='utf-8'))
    assert 'from tools import c85r7_r224flip_diagnostics as hb153' in instrumented
    compile(instrumented,'<GPU diagnostic>','exec')


def test_current_rejects_source_drift(tmp_path,monkeypatch):
    source=tmp_path/'source.py'; source.write_text('before')
    ids=[f'OAI_{i}_L' for i in range(557)]
    binding=dict(rules=r.RULES,ids=r.select_ids(ids),sources={'source.py':r.sha(source)})
    monkeypatch.setattr(r,'ROOT',tmp_path)
    monkeypatch.setattr(r.raw,'ids_only',lambda:ids)
    monkeypatch.setattr(r,'read',lambda _:dict(binding=binding,binding_sha256=r.digest(binding)))
    source.write_text('after')
    with pytest.raises(ValueError,match='source drift'): r.current()


def test_prepared_rejects_raw_bytes_drift(monkeypatch):
    reg=dict(binding_sha256='r')
    monkeypatch.setattr(r,'current',lambda:reg)
    monkeypatch.setattr(r,'read',lambda _:dict(binding_sha256='r',data={'series':'old'}))
    monkeypatch.setattr(r,'data_pins',lambda:{'series':'changed'})
    with pytest.raises(ValueError,match='data drift'): r.verify_prepared()


def test_historical_memory_gate_live_headroom_and_uuid():
    reg={'binding':{'memory':{'R6flip':dict(uuid='physical',peak_bytes=11*2**30)}}}
    desktop=dict(uuid='physical',device_name='NVIDIA RTX 4070 Ti SUPER',used_mib=1500,total_mib=16376)
    assert r.admit_memory(reg,'R6flip',desktop)==11*2**30
    with pytest.raises(ValueError,match='physical'): r.admit_memory(reg,'R6flip',dict(desktop,uuid='other'))
    with pytest.raises(ValueError,match='headroom'): r.admit_memory(reg,'R6flip',dict(desktop,used_mib=4000))
    reg['binding']['memory']['R6flip']['peak_bytes']=13*2**30
    with pytest.raises(ValueError,match='12 GiB'): r.admit_memory(reg,'R6flip',desktop)
