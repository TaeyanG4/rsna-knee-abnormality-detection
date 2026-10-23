"""U embedded finish and independently replayed admission; CPU-only fixtures."""
import ast
import copy
import json
import sys
from types import ModuleType
from functools import lru_cache
import numpy as np
import pytest
from tools import u_union_stage_v1 as b
from tools import u_union_integration_v1 as u
from tools import u_union_admit_v1 as a

def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(u.util.encoded(v))

@lru_cache(None)
def package(diag=False):
    outputs,reg=b.artifacts(diag)
    code=''.join(json.loads(outputs['kernel/release.ipynb'])['cells'][2]['source'])
    files=b.archive.parent.archive(code)[2]
    call=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='finish')
    return outputs,reg,files,ast.literal_eval(call.args[2])

def make_case(root,monkeypatch,diag=False,fault=None):
    outputs,reg,files,cfg=package(diag)
    payload=root/'payload';payload.mkdir(parents=True)
    for name,data in files.items():
        p=payload/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    monkeypatch.syspath_prepend(str(payload))
    import exp169_r224 as r224
    r=ModuleType('u_embedded_runtime');monkeypatch.setitem(sys.modules,r.__name__,r)
    exec(compile(files[b.RUNTIME],b.RUNTIME,'exec'),r.__dict__)
    n=32 if diag else 3;ids=list(map(str,range(n)));rng=np.random.default_rng(802)
    arrays=[rng.integers(1,9,(n,12))/10 for _ in range(6)]
    public,own,pf,of=u.split(r224,ids,*arrays,{})
    parent=r224.p.csv_bytes(ids,u.outer(public,own,pf,of,r224.c85.scored_rank))
    d=root/'output';d.mkdir()
    snap=u.snapshot(r224,d,ids,arrays,{},parent)
    p=dict(status='DIAGNOSTIC_C85R7' if diag else 'COMPLETE_C85R7',version=reg['release_version'],studies=n,findings=12,
        fallbacks=dict(a172_prediction=0,r_prediction=0,s_r7_prediction=0,r224_prediction=0),codecs=dict(status='ACTIVE',s_r7_gate='ACTIVE'),
        submission_sha256=None if diag else u.digest(parent),intended_members=sorted(a.MEMBERS),executed_members=sorted(a.MEMBERS),skipped_members=[],drop_o=False,**snap)
    put(d/'c85r7-receipt.json',p)
    (d/('diagnostic-c85r7.csv' if diag else 'submission.csv')).write_bytes(parent)
    predictions=[rng.integers(0,10,(n,12))/10 for _ in range(3)]
    rows=[dict(uid=uid,ok=True,config_sha256=r.CONFIG_SHA,k1_models=list(r.MODEL_IDS),finished_elapsed_seconds=10.,
        predictions={m:v[i].tolist() for m,v in zip(r.MODEL_IDS,predictions)},classifier_seconds=100/n,extra_render_seconds=2/n,load_seconds=60/n,
        per_model_seconds={m:dict(load=20/n,classify=100/(n*3)) for m in r.MODEL_IDS}) for i,uid in enumerate(ids)]
    if fault in ('one_failed','too_many'): rows[0].update(ok=False,predictions={})
    if fault=='systemic':rows[0]['systemic_failure']=True
    if fault=='missing':rows.pop()
    if fault=='snapshot':put(d/'u-inputs.json',{})
    run=root/'run';path=run/'pr7/pr7-run/s_r7/controller/worker0/k1-studies.jsonl'
    path.parent.mkdir(parents=True);path.write_text(''.join(json.dumps(v)+'\n' for v in rows))
    if fault in ('too_many','systemic','missing','snapshot'):
        with pytest.raises(RuntimeError,match='fail-loud'):r.finish(d,run,cfg,1000.)
        assert json.loads((d/'k1sr-receipt.json').read_bytes())['fail_loud_triggered']
        return
    k=r.finish(d,run,cfg,1000.)
    a.chain(a.Evidence(),d,reg,k,p,diag)
    return d,reg,k,p

@pytest.mark.parametrize('diag',[False,True])
def test_embedded_finish_independent_full_rank_replay(tmp_path,monkeypatch,diag):
    d,reg,k,p=make_case(tmp_path,monkeypatch,diag)
    assert k['k1_eval_stride']==2 and k['contract']['eval_stride']==2
    assert not any('a5' in v.name for v in d.iterdir())

def test_allowed_single_failure_n32(tmp_path,monkeypatch):
    make_case(tmp_path,monkeypatch,True,'one_failed')

@pytest.mark.parametrize('fault',['too_many','systemic','missing','snapshot'])
def test_embedded_fail_loud(tmp_path,monkeypatch,fault):make_case(tmp_path,monkeypatch,False,fault)

@pytest.mark.parametrize('diag',[False,True])
@pytest.mark.parametrize('fault',['EXP143v2','EXP155','Q_A5q','stride','weights','a5','fraction','model','math'])
def test_admission_negative(tmp_path,monkeypatch,diag,fault):
    d,reg,k,p=make_case(tmp_path,monkeypatch,diag)
    reg,k,p=copy.deepcopy((reg,k,p))
    if fault in a.MEMBERS:p['executed_members'].remove(fault)
    elif fault=='stride':k['k1_eval_stride']=3
    elif fault=='weights':reg['u_contract']['y']=[.8,.2]
    elif fault=='a5':k['a5_executed']=True
    elif fault=='fraction':k['failed_study_fraction']=.06
    elif fault=='model':k['k1_models']=k['k1_models'][:2]
    elif fault=='math':
        ids,vals=u.util.table((d/'u-final.csv').read_bytes());wrong=u.csv_bytes(ids,np.full_like(vals,.123));(d/'u-final.csv').write_bytes(wrong)
        rec=json.loads((d/'u-receipt.json').read_bytes());rec['csv_sha256']['u-final.csv']=u.digest(wrong);put(d/'u-receipt.json',rec)
        k['u_receipt_sha256']=u.util.sha(d/'u-receipt.json')
    with pytest.raises((a.Refuse,ValueError)):a.chain(a.Evidence(),d,reg,k,p,diag)

def test_payload_scope():
    for diag in (False,True):
        outputs,reg,files,cfg=package(diag);scope=json.loads(outputs['payload-scope.json'])
        assert scope['runtime_functions_changed']==['finish']
        assert len(scope['removed'])==4 and len(scope['added'])==1
        assert not any('p25_a5' in n or 'x3_a5' in n for n in files)
        assert cfg['diagnostic'] is diag and cfg['never_submit'] is diag
