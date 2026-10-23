"""X3 A5 nesting, authentic payload identity and fail-closed finish coverage."""
import ast
import json
import sys
from functools import lru_cache
from types import ModuleType
import numpy as np
import pytest
from tools import x3_a5_stage_v1 as b
from tools import x3_a5_integration_v1 as x
from tools import c85r7_r6w958k1x3sr_v1162 as parent
from test_k1n_v7 import records,replay

@lru_cache(None)
def artifacts(diagnostic=False): return b.artifacts(diagnostic)

@lru_cache(None)
def payload(diagnostic=False):
    outputs,reg=artifacts(diagnostic)
    nb=json.loads(outputs['kernel/release.ipynb'])
    return parent.parent.archive(''.join(nb['cells'][2]['source']))[2],reg

@pytest.fixture
def r224(monkeypatch):
    path=b.SCRATCH/'parent-payload'
    nb=json.loads((b.PARENT/'kernel/release.ipynb').read_bytes())
    files=parent.parent.archive(''.join(nb['cells'][2]['source']))[2]
    for name,data in files.items():
        target=path/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): assert target.read_bytes()==data
        else: target.write_bytes(data)
    monkeypatch.syspath_prepend(str(path))
    import exp169_r224
    return exp169_r224


def test_snapshot_immutable_replay(tmp_path,r224):
    rng=np.random.default_rng(619); ids=['a','b','c']
    arrays=[rng.random((3,12)) for _ in range(6)]
    original=r224.p.csv_bytes(ids,r224.blend(ids,*arrays,{}))
    first=x.snapshot(r224,tmp_path,ids,arrays,{},original)
    assert x.snapshot(r224,tmp_path,ids,arrays,{},original)==first
    with pytest.raises(ValueError,match='immutable X3 snapshot'):
        x.snapshot(r224,tmp_path,ids,arrays,{},original+b' ')

@pytest.mark.parametrize('diagnostic',[False,True])
def test_scope(diagnostic):
    outputs,reg=artifacts(diagnostic)
    scope=json.loads(outputs['payload-scope.json'])
    files,_=payload(diagnostic)
    anb=json.loads((b.A5_STAGE/'kernel/release.ipynb').read_bytes())
    af=parent.parent.archive(''.join(anb['cells'][2]['source']))[2]
    assert all(files[n]==af[n] for n in b.MODULES)
    assert scope['changed']==sorted(['exp169_r7_port.py','exp169_release_c85r7.py',b.CODEC])
    assert scope['runtime_functions_identical_to_reviewed_a5_except_finish']
    assert len(scope['untouched'])==33
    c=json.loads(outputs['config.json'])
    assert c['alpha']==.2 and not c['drop_o'] and c['k1_eval_stride']==3
    assert c.get('r6_eval_stride',1)==1 and c['k1_guard_hours']==8.8
    assert reg['never_submit']==diagnostic
    assert len(c['models'])==3 and reg['r224_weight']==.35
    assert 'CAPTURE_TIMEOUT_SECONDS = 900' in files[b.RUNTIME].decode()
    assert 'CALL_TIMEOUT_SECONDS = 300' in files[b.RUNTIME].decode()

@pytest.mark.parametrize('failed',[False,True])
def test_exact_parent_branch_split(r224,failed):
    rng=np.random.default_rng(190)
    ids=list(map(str,range(101)))
    arrays=[rng.random((101,12)) for _ in range(6)]
    failures={}
    if failed:
        for i,n in enumerate(['R','R224','S_R7','EXP143v2','EXP155','Q_A5q']):
            failures[n]={str(i):'test'}
        arrays[0][0]=.5; arrays[1][1]=.5
        for n in failures: failures[n]['100']='test'
    public,own,pf,of=x.split(r224,ids,*arrays,failures)
    np.testing.assert_array_equal(x.outer(public,own,pf,of,r224.c85.scored_rank),r224.blend(ids,*arrays,failures))


def setup_chain(path,r224,diagnostic=False,raw=None):
    rng=np.random.default_rng(109)
    ids=list(map(str,range(23)))
    arrays=[rng.random((23,12)) for _ in range(6)]
    original=r224.p.csv_bytes(ids,r224.blend(ids,*arrays,{}))
    (path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(original)
    c=x.snapshot(r224,path,ids,arrays,{},original)
    (path/'c85r7-receipt.json').write_bytes(b.util.encoded(c))
    p=(path/'x3-a5-original-p.csv').read_bytes()
    raw=rng.random((5,23,12)) if raw is None else np.broadcast_to(raw,(5,23,12))
    prime=b.a5.blend(p,raw,np.ones(23,bool))
    a=dict(status='COMPLETE_A5',original_p_sha256=b.digest(p),pprime_sha256=b.digest(prime))
    (path/'a5-receipt.json').write_bytes(b.util.encoded(a))
    (path/'a5-pprime.csv').write_bytes(prime)
    composed=x.compose(path,prime,diagnostic)
    assert composed!=b.a5.blend(original,raw,np.ones(23,bool))
    return original,composed,rng

@pytest.mark.parametrize('diagnostic',[False,True])
@pytest.mark.parametrize('fault',['none','override','snapshot','pprime','a5_receipt','composed','original'])
def test_chain_and_finish(tmp_path,r224,diagnostic,fault):
    original,composed,rng=setup_chain(tmp_path,r224,diagnostic)
    files,reg=payload(diagnostic)
    nb=json.loads(artifacts(diagnostic)[0]['kernel/release.ipynb'])
    tree=ast.parse(''.join(nb['cells'][2]['source']))
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='finish')
    reg=ast.literal_eval(call.args[2])
    r=ModuleType('x3a5_baked');exec(compile(files[b.RUNTIME],'<x3a5>','exec'),r.__dict__)
    rows=records(r,[rng.random((23,12)) for _ in range(3)])
    worker=tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0';worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    if fault=='override': composed=original
    if fault=='snapshot': (tmp_path/'x3-a5-inputs.json').write_text('{}')
    if fault=='pprime': (tmp_path/'a5-pprime.csv').write_bytes(original)
    if fault=='a5_receipt': (tmp_path/'a5-receipt.json').write_text('{}')
    if fault=='composed': (tmp_path/'x3-a5-inf3.csv').write_bytes(original)
    if fault=='original': (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(composed)
    if fault!='none':
        with pytest.raises((ValueError,KeyError)):
            r.finish(tmp_path,tmp_path/'run',reg,15.,parent_override=composed)
        assert not (tmp_path/'k1sr-parent.csv').exists()
    else:
        rec=r.finish(tmp_path,tmp_path/'run',reg,15.,parent_override=composed)
        assert rec['status']=='COMPLETE_SHARED_K1N' and rec['parent_mode']=='INF3_A5_PPRIME'
        assert (tmp_path/'k1sr-parent.csv').read_bytes()==composed
        assert (tmp_path/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')).read_bytes()==r.blend(composed,rows)
        if diagnostic: assert not (tmp_path/'submission.csv').exists()
