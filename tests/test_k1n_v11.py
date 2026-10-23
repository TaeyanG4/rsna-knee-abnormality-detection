"""Runtime11 stride identity, exact original windows, legacy bytes and deadlines."""
import base64
import contextlib
import copy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from tools import k1n_stage_v11 as b
from tools import k1_shared_render_v3 as s
from tools import k1sr_runtime_v11 as source
import test_k1n_stride_v710 as oracle
from test_k1n_v7 import records, replay


def runtime():
    r = ModuleType('baked_runtime11_test')
    code = Path(source.__file__).read_text().replace('REGISTERED = config_schema.current()',
                                                   'REGISTERED = '+repr(b.registered()))
    exec(compile(code, '<runtime11>', 'exec'), r.__dict__)
    return r


@pytest.mark.parametrize('value', [None, {}, {'k1_v2':2},
    {'k1_v2':2,'k1_v3g_seed0':2,'k1_v3g_seed1':True},
    {'k1_v2':2,'k1_v3g_seed0':2,'k1_v3g_seed1':1.0},
    {'k1_v2':2,'k1_v3g_seed0':2,'k1_v3g_seed1':4}])
def test_strict_complete_map(value):
    c = b.registered(); c['k1_member_eval_strides'] = value
    with pytest.raises(ValueError): b.schema.validate(c)
    with pytest.raises(ValueError): oracle.s.stride_check(1)
    assert b.schema.validate(b.registered())['k1_member_eval_strides'] == {
        'k1_v2':2,'k1_v3g_seed0':2,'k1_v3g_seed1':1}


@pytest.mark.parametrize('stride', [1,2])
@pytest.mark.parametrize('n,spacing', [(1,3.5),(2,3.5),(10,3.5),(13,.7),(27,1.2)])
def test_original_windows(tmp_path,monkeypatch,stride,n,spacing):
    monkeypatch.setattr(oracle,'s',s)
    oracle.test_original_cstudies_method_exact_windows(tmp_path,stride,n,spacing)


@pytest.mark.parametrize('flipped,fail', [(False,False),(True,False),(True,True)])
def test_both_routes_restore(monkeypatch,flipped,fail):
    ci=SimpleNamespace(FORCED_SIDE={'uid':'L','other':'R'}); sentinel=object()
    class DS:
        uids=['uid']; native=True; side_net=sentinel; eval_stride=1; extent_meta=True; fluid=True; window_rows=None
        def __init__(self): self.builder=SimpleNamespace(build=lambda uid:{})
        def _native_and_geometry(self,*a):
            if flipped: ci.FORCED_SIDE['uid']='R'
            return {'flipped':flipped}
        def __getitem__(self,i):
            if self.side_net is None:
                assert not self.extent_meta and not self.fluid and 'uid' not in ci.FORCED_SIDE
                if fail: raise RuntimeError('injected')
                return {'uid':'uid','route':'original','stride':self.eval_stride}
            self._native_and_geometry(); self.builder.build('uid')
            return {'uid':'uid','error':None,'anchor':None}
    monkeypatch.setattr(s,'from_rows',lambda raw,uid,fn,stride:dict(uid=uid,route='shared',stride=stride))
    monkeypatch.setattr(s.w,'validate_item',lambda *a:None)
    ds=DS(); build,native=ds.builder.build,ds._native_and_geometry
    a=s.SharedR6.__new__(s.SharedR6); a.ds,a.module,a.active,a.eval_strides=ds,ci,False,(2,1)
    if fail:
        with pytest.raises(RuntimeError,match='injected'): a[0]
    else:
        _,items,detail=a[0]
        assert detail['k1_eval_strides']==[2,1] and set(items)=={1,2}
        assert all(v['stride']==k and v['route']==('original' if flipped else 'shared') for k,v in items.items())
    assert ds.eval_stride==1 and ds.side_net is sentinel and ds.extent_meta and ds.fluid
    assert ds.builder.build==build and ds._native_and_geometry==native and not a.active
    assert ci.FORCED_SIDE=={'uid':'L','other':'R'}


@pytest.mark.parametrize('cost,ok', [(50,True),(100,False)])
def test_owner_group_transfer_cumulative_deadline(tmp_path,monkeypatch,cost,ok):
    r=runtime(); clock=[1000.]; transfers=[]; forwards=[]
    monkeypatch.setenv('K1SR_STARTED','1000')
    monkeypatch.setattr(r,'time',SimpleNamespace(time=lambda:clock[0],perf_counter=lambda:clock[0]))
    monkeypatch.setattr(r,'deadline',lambda seconds:contextlib.nullcontext())
    monkeypatch.setattr(r,'precision',contextlib.nullcontext)
    monkeypatch.setattr(r.w,'offline',contextlib.nullcontext)
    monkeypatch.setattr(r.w,'validate_item',lambda item,uid:(np.zeros((item['n'],1)),np.zeros((item['n'],1))))
    def load(dev,m): clock[0]+=500; return m
    def transfer(x,meta,dev): transfers.append(len(x)); clock[0]+=1; return len(x)
    def predict(model,prepared,**kw): forwards.append((model,prepared)); clock[0]+=cost; return np.full(12,.4)
    monkeypatch.setattr(r,'load_classifier',load); monkeypatch.setattr(r,'transfer_inputs',transfer)
    monkeypatch.setattr(r,'predict_prepared',predict)
    owner=r.Owner(tmp_path,'cuda')
    owner.handle(dict(uid='0',_k1sr=dict(item={2:{'n':4},1:{'n':9}},detail={})),lambda i:None)
    rec=json.loads((tmp_path/'k1-studies.jsonl').read_text())
    assert rec['ok'] is ok and transfers==[4,9]
    assert forwards==list(zip(r.MODEL_IDS,[4,4,9]))
    assert rec['load_seconds']==1500 and rec['classifier_seconds']==3*cost+2
    assert rec['k1_member_eval_strides']==r.MEMBER_STRIDES
    assert rec['per_member_windows']==dict(zip(r.MODEL_IDS,[4,4,9]))
    assert not owner.failed and not torch.cuda.is_initialized()


@pytest.mark.parametrize('diagnostic',[False,True])
def test_finish_receipts_and_tamper(tmp_path,diagnostic):
    r=runtime(); cfg=b.package(diagnostic)[-2]
    payload=replay.csv_bytes(['0','1'],np.full((2,12),.5))
    rows=records(r,[np.full((2,12),.4)]*3)
    for row in rows: row['k1_member_eval_strides']=dict(r.MEMBER_STRIDES)
    assert r.blend(payload,rows)
    bad=copy.deepcopy(rows); bad[0]['k1_member_eval_strides']['k1_v3g_seed1']=2
    with pytest.raises(ValueError,match='identity'): r.blend(payload,bad)
    (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(payload)
    worker=tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'; worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    rec=r.finish(tmp_path,tmp_path/'run',cfg,1.)
    assert rec['k1_executed'] and rec['k1_member_eval_strides']==r.MEMBER_STRIDES
    assert rec['never_submit'] is diagnostic
    if diagnostic: assert rec['submission_sha256'] is None and not (tmp_path/'submission.csv').exists()


@pytest.mark.parametrize('diagnostic',[False,True])
def test_payload_scope_inverse_and_legacy(diagnostic):
    nb,meta,old,new,on,om,cfg,oreg=b.package(diagnostic)
    assert len(old)==39 and len(new)==42
    for name in ('tools/k1sr_runtime_v10.py','tools/k1_shared_render_v2.py','tools/k1n_config_v1.py'):
        assert old[name]==new[name]
    engine=b.prior.replay.load(); normalized=copy.deepcopy(nb)
    code=''.join(nb['cells'][2]['source']); orig=''.join(on['cells'][2]['source'])
    blob,pins,_=engine.before.parent.archive(code); ob,op,_=engine.before.parent.archive(orig)
    pc=json.loads((b.PARENT/'config.json').read_bytes())
    for a,z in [(repr(base64.b64encode(blob).decode()),repr(base64.b64encode(ob).decode())),
        ('PINS = '+repr(pins),'PINS = '+repr(op)),(repr(engine.digest(blob)),repr(engine.digest(ob))),
        ('from tools import k1sr_runtime_v11 as _k1sr','from tools import k1sr_runtime_v10 as _k1sr'),
        (repr(cfg),repr(b.prior.config(pc,diagnostic)))]: code=b.exact(code,a,z)
    normalized['cells'][2]['source']=code.splitlines(True); normalized['cells'][1]=on['cells'][1]
    assert normalized==on
    assert meta==dict(om,id=cfg['kernel_id'],title=cfg['kernel_id'].split('/')[1])
    outputs,reg=b.artifacts(diagnostic); scope=json.loads(outputs['payload-scope.json'])
    assert len(scope['untouched'])==36 and reg['never_submit'] is diagnostic
    assert scope['config_diff']['k1_member_eval_strides']['after']=={'k1_v2':2,'k1_v3g_seed0':2,'k1_v3g_seed1':1}
    c=b.registered()
    assert c['alpha']==.25 and c['drop_o'] and c['k1_guard_hours']==8.8


def test_original_backups_unchanged():
    backup=b.ROOT/'.local/backups/astra-b2-k1s1stride1-v1'
    for name in ('k1_shared_render_v2.py','k1sr_runtime_v10.py','k1n_stage_v10.py','k1n_config_v1.py'):
        assert (backup/name).read_bytes()==(b.ROOT/'tools'/name).read_bytes()
