"""Registered stride contracts, using the original CStudies method as oracle."""
import ast
import hashlib
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pytest
import torch

from tools import k1_shared_render_v2 as s
from tools import k1_shared_render_v1 as old
from tools import k1_shared_render_harness_v1 as h
from test_k1_shared_render_v1 import windows
from test_k1n_v7 import configuration, runtime, records, replay
from tools import k1n_config_v1 as schema
from tools import k1n_stage_v1 as b


@pytest.mark.parametrize('stride', [2, 3])
@pytest.mark.parametrize('n,spacing', [(1,3.5),(2,3.5),(4,3.5),(10,3.5),(13,.7),(27,1.2)])
def test_original_cstudies_method_exact_windows(tmp_path, stride, n, spacing):
    source = h.ASSETS/'source/compact_infer.py'
    assert s.w.sha(source) == s.K1_INFER_SHA
    cls = next(x for x in ast.parse(source.read_text(encoding='utf-8')).body
               if isinstance(x, ast.ClassDef) and x.name == 'CStudies')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '__getitem__')
    ns = dict(time=time, Path=Path, hashlib=hashlib, np=np, torch=torch, shutil=shutil)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
    px = torch.arange(2*n, dtype=torch.uint8)[:,None,None].expand(-1,384,384).clone()
    meta = torch.arange(2*n*12, dtype=torch.float32).reshape(2*n,12)
    meta[:,5] = math.log1p(spacing)
    raw = dict(px=px, meta=meta, blocks=[dict(n_slices=n, views=[(0,0),(1,n)])])
    fn = windows(h.ASSETS/'source/compact_train.py')['window_rows']
    ds = SimpleNamespace(uids=['synthetic'], tmp=tmp_path, native=True, bank=None,
        builder=SimpleNamespace(build=lambda uid:raw), _native_and_geometry=lambda *a:None,
        slices=3, eval_stride=stride, phys_mm=3.5, extent_meta=False, size=384, window_rows=fn)
    expected = ns['__getitem__'](ds,0)
    assert expected['error'] is None
    actual = s.from_rows(raw,'synthetic',fn,stride)
    assert torch.equal(actual['images'],expected['images'])
    assert torch.equal(actual['meta'],expected['meta'])
    if stride == 3:
        legacy = old.from_rows(raw,'synthetic',fn)
        assert torch.equal(actual['images'],legacy['images'])
        assert torch.equal(actual['meta'],legacy['meta'])


@pytest.mark.parametrize('value', [0,1,4,-2,True,False,'2',2.,None])
def test_config_and_adapter_reject_stride(value):
    c = configuration(); c['k1_eval_stride'] = value
    with pytest.raises(ValueError, match='stride'): schema.validate(c)
    with pytest.raises(ValueError, match='stride'): s.stride_check(value)


@pytest.mark.parametrize('stride', [2,3])
@pytest.mark.parametrize('flipped,fail', [(False,False),(True,False),(True,True)])
def test_selected_stride_routes_and_restoration(monkeypatch,stride,flipped,fail):
    ci = SimpleNamespace(FORCED_SIDE={'uid':'L','other':'R'})
    sentinel = object()
    class DS:
        uids=['uid']; native=True; side_net=sentinel; eval_stride=1; extent_meta=True; fluid=True; window_rows=None
        def __init__(self): self.builder=SimpleNamespace(build=lambda uid:{})
        def _native_and_geometry(self,*args):
            if flipped: ci.FORCED_SIDE['uid']='R'
            return {'flipped':flipped}
        def __getitem__(self,i):
            if self.eval_stride != 1:
                assert self.eval_stride == stride and self.side_net is None
                assert not self.extent_meta and not self.fluid and 'uid' not in ci.FORCED_SIDE
                if fail: raise RuntimeError('injected')
                return {'uid':'uid','route':'original'}
            self._native_and_geometry()
            self.builder.build('uid')
            return {'uid':'uid','error':None,'anchor':{'method':'test'}}
    def captured(raw,uid,fn,selected):
        assert selected == stride
        return {'uid':uid,'route':'shared'}
    monkeypatch.setattr(s,'from_rows',captured)
    monkeypatch.setattr(s.w,'validate_item',lambda *a:None)
    ds = DS(); build, native = ds.builder.build, ds._native_and_geometry
    adapter = s.SharedR6.__new__(s.SharedR6)
    adapter.ds,adapter.module,adapter.active,adapter.eval_stride=ds,ci,False,stride
    if fail:
        with pytest.raises(RuntimeError, match='injected'): adapter[0]
    else:
        _,item,detail = adapter[0]
        assert detail['k1_eval_stride'] == stride
        assert item['route'] == ('original' if flipped else 'shared')
    assert ds.eval_stride == 1 and ds.side_net is sentinel and ds.extent_meta and ds.fluid
    assert ds.builder.build == build and ds._native_and_geometry == native and not adapter.active
    assert ci.FORCED_SIDE == {'uid':'L','other':'R'}


@pytest.mark.parametrize('diagnostic', [False,True])
def test_stride_baked_registration_receipt_and_never_submit(tmp_path,diagnostic):
    c = schema.validate(s.w.read(b.ROOT/'configs/k1n/x3-stride2-drop-o.json'))
    assert c['k1_eval_stride'] == 2 and c['drop_o'] and c['alpha'] == .25
    cfg = b.config(c,diagnostic)
    assert cfg['k1_eval_stride'] == cfg['contract']['eval_stride'] == 2
    files = b.package(c,diagnostic)[3]
    assert files['tools/k1_shared_render_v2.py'] == Path(s.__file__).read_bytes()
    r = runtime(c)
    assert r.K1_EVAL_STRIDE == 2
    parent = replay.csv_bytes(['0','1'], np.full((2,12),.3))
    (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(parent)
    worker = tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    import json
    rows = records(r,[np.full((2,12),.4)]*3)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    receipt = r.finish(tmp_path,tmp_path/'run',cfg,1.)
    assert receipt['k1_eval_stride'] == 2 and receipt['k1_executed']
    assert receipt['contract']['eval_stride'] == 2
    if diagnostic:
        assert receipt['never_submit'] and receipt['submission_sha256'] is None
        assert not (tmp_path/'submission.csv').exists()


def test_default_and_frozen_stage_identity():
    assert schema.validate(schema.current())['k1_eval_stride'] == 3
    pins = s.w.read(b.ROOT/'.local/scratch/astra-b1/k1n-stride-v710/frozen-stage-hashes.json')
    for path, pin in pins.items(): assert s.w.sha(path) == pin
    assert not torch.cuda.is_initialized()
