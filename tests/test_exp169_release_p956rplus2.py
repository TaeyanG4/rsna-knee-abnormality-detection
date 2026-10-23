"""CPU release gates for probability-mean R2; parent S6 and cell-7 unchanged."""
import ast
import gc
import hashlib
import importlib.util
import inspect
import os
import queue
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import exp169_release_p956rplus as parent
from tools import exp169_release_p956rplus2 as p
from tools import exp169_p956rplus2_cpu as cpu

spec=importlib.util.spec_from_file_location('p956_cases_for_r2',Path(__file__).with_name('test_exp169_release_p956.py'))
cases=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cases)
visible=cases.visible


def test_unchanged_parent_math():
    for name in ('blend', 'failure_masks', 'public_formula', 'commit', 'validate_s6',
                 's6_failure_evidence', 'anatomy_anchors'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(p, name)))) == ast.dump(
            ast.parse(inspect.getsource(getattr(parent, name)))), name
    assert p.PINS == parent.PINS and p.DATASETS[1:] == parent.DATASETS[1:]
    source = (p.VISIBLE / 'exp155-assets/public-serving.py').read_text(encoding='utf-8')
    old, new = ast.parse(parent.r_runtime_source(source)), ast.parse(p.r_runtime_source(source))
    before = {n.name: ast.dump(n) for n in old.body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in new.body if isinstance(n, ast.FunctionDef)}
    assert {name for name in before if before[name] != after[name]} == {'main'}


@pytest.mark.parametrize('fault', [None, 'first', 'second', 'both', 'nan', 'range', 'shape'])
def test_member_mean_and_failures(tmp_path, fault):
    binding = p.RBinding(types.SimpleNamespace(checked_model=None, input_record=None), tmp_path)
    def infer(model, *args):
        if fault == 'both' or (fault == 'first' and model == 0) or (fault == 'second' and model == 1):
            raise RuntimeError('injected')
        if model == 0:
            if fault == 'nan': return np.full(12, np.nan)
            if fault == 'range': return np.full(12, 1.1)
            if fault == 'shape': return np.full(11, .2)
        return np.full(12, .2 if model == 0 else .8, dtype=np.float32)
    if fault == 'both':
        with pytest.raises(ValueError, match='both R2 checkpoints failed'):
            binding.infer_members('study', [(0,320),(1,320)], None, None, 'cpu', infer)
    else:
        result = binding.infer_members('study', [(0,320),(1,320)], None, None, 'cpu', infer)
        np.testing.assert_array_equal(result, np.full(12, .5 if fault is None else
            .2 if fault == 'second' else .8, dtype=np.float32))


@pytest.mark.parametrize('fault', [None, 'first', 'second', 'both', 'decode'])
def test_actual_r_consumer_and_receipts(tmp_path, monkeypatch, fault):
    import multiprocessing as mp
    # Real authenticated prediction values; only model executor/decode transport mocked.
    ids, real = p.v5q._read_csv(parent.OUT / 'cpu-replay/rplus.csv')
    comp, output = tmp_path / 'comp', tmp_path / 'r'
    comp.mkdir(); output.mkdir()
    (comp/'test.csv').write_text(p.UID+'\n'+'\n'.join(ids)+'\n')
    pd.DataFrame({p.UID: ids, 'SeriesInstanceUID': ids}).to_csv(comp/'test_series.csv', index=False)
    (comp/'sample_submission.csv').write_bytes(p.csv_bytes(ids, real))
    source = p.r_runtime_source((p.VISIBLE/'exp155-assets/public-serving.py').read_text(encoding='utf-8'))
    selected = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and
                n.name in {'_pre_one','_pre_init','_exp152_bounded_preprocess','main'}]
    bound = p.RBinding(types.SimpleNamespace(checked_model=None,input_record=lambda *a: {}), output)
    decoded, forwards = [], []
    def reader(uid, *args):
        decoded.append(uid)
        if fault == 'decode' and uid == ids[0]: raise OSError('decode injected')
        return np.array([ids.index(uid)]), np.array([1])
    def predict(model, plain, mirror, device):
        i = int(plain[0]); forwards.append((i, model))
        if i == 0 and (fault == 'both' or fault == model): raise RuntimeError('member injected')
        return real[i] if model == 'first' else real[i] * .8
    class Pool:
        def __init__(self, n, initializer): initializer()
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def imap_unordered(self, fn, batch, chunksize): return map(fn, reversed(batch))
    monkeypatch.setattr(mp,'get_context',lambda kind: types.SimpleNamespace(Pool=Pool))
    monkeypatch.setitem(sys.modules,'torch_xla',None)
    ns=dict(np=np, os=os,time=time,queue=queue,threading=threading,gc=gc,
        torch=types.SimpleNamespace(__version__='CPU',cuda=types.SimpleNamespace(is_available=lambda:False,device_count=lambda:0)),
        LAB=list(p.LABELS),ARMS=[dict(file=n,arch='pinned',res=384,w=.5) for n in ('first','second')],
        find_test_root=lambda:str(comp),find_weight_file=lambda n:n,load_model=lambda name,*a:(name,320),
        gpu_windows=lambda vol,*a,**kw:vol,infer_pair_probs=predict,K_EVAL=94,INPUT_RES=320,NORM=None,
        rankpct=None,QUEUE_MAX=4,FASTREAD_VARIANT='pinned',FR_HDR=1,FR_PX=1,IMG=1,MAXS=1,SLOTS=1,
        _load_fastread=lambda:types.SimpleNamespace(make_builder=lambda *a,**kw:reader),
        _E152=bound,_E152_WORK=output,_E152_VARIANT='R',_E152_MANIFEST={})
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<R2 CPU main>','exec'),ns)
    ns['main']()
    assert sorted(decoded)==sorted(ids) and len(decoded)==len(ids)
    assert len(forwards)==2*(len(ids)-int(fault=='decode'))
    expected = .5*(real.astype(np.float32)+real.astype(np.float32)*.8)
    if fault in ('both','decode'): expected[0]=.5
    if fault=='first': expected[0]=real[0]*.8
    if fault=='second': expected[0]=real[0]
    with np.load(output/'raw.npz') as raw:
        np.testing.assert_allclose(raw['public'],expected,rtol=1e-6)
    receipt=p.read_json(output/'receipt.json')
    assert receipt['failure_count']==int(fault in ('both','decode'))
    assert bool(receipt['member_failures'])==(fault in ('first','second','both'))


def test_staged_archive_and_pending_guard(tmp_path):
    assert p.validate()['status']=='PASS_CPU_VALIDATE'
    meta=p.read_json(p.OUT/'kernel/kernel-metadata.json')
    assert meta['id']=='taeyangg4/knee-exp169-p956rplus2-v1' and meta['is_private']
    assert p.read_json(p.OUT/'kernel/release.ipynb')['metadata']['kernelspec']['name']=='python3'
    blob,pins=p.package()
    runtime=tmp_path/'runtime'
    p.unpack(blob,hashlib.sha256(blob).hexdigest(),runtime,pins)
    result=subprocess.run([sys.executable,'-B','-c',
        "import exp169_release_p956rplus2 as p; p.runtime_pins(p.Path('.')); print(p.VERSION)"],
        cwd=runtime,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1'),
        text=True,capture_output=True,check=True)
    assert result.stdout.strip()=='1.1.0'
    if p.validate_dataset()['pending_seed44']:
        with pytest.raises(ValueError,match='seed44 pending'): p.weight_manifest(p.OUT/p.DATASET_DIR)
        assert not (p.OUT/p.DATASET_DIR/p.R44_NAME).exists()


def test_real_cpu_replay():
    report=p.read_json(cpu.SCRATCH/'cpu-replay/replay.json')
    for name,digest in report['files'].items(): p.check_pin(cpu.SCRATCH/'cpu-replay'/name,digest)
    assert report['strict_load'] and report['state_tensors']==[486,486]
    assert report['decode_calls']==report['studies']==2
    assert report['one_failure_exact'] and report['both_fail_raise'] and report['finite']
    assert report['s6_exact'] and report['public_formula_exact']
    assert report['native_resolution']==320 and report['k_eval']==94


def test_checkpoint_hash_drift_rejected(tmp_path):
    manifest={'checkpoints':{p.R_NAME:{'sha256':p.R_SHA},p.R44_NAME:{'sha256':'0'*64}}}
    (tmp_path/'manifest.json').write_bytes(p.json_bytes(manifest))
    (tmp_path/p.R_NAME).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA256 mismatch'): p.weight_manifest(tmp_path)


@pytest.mark.parametrize('fault',[None,'R failure','S6 failure','ID mismatch','receipt failure','GPU shape'])
def test_r2_full_orchestration_real_outputs(tmp_path, visible, monkeypatch, fault):
    # Inherit the real-output lifecycle test; only the two-member binding API changes.
    source=inspect.getsource(cases.test_real_data_cpu_orchestration_smoke)
    source=source.replace('assert len(namespace["ARMS"]) == 1','assert len(namespace["ARMS"]) == 2')
    source=source.replace('p.run_r(runtime, weights, comp,','p.run_r(runtime, weights.parent, comp,')
    source=source.replace('p.run_release(runtime, output)','p.run_release(runtime, output, expected_manifest)')
    manifest={'checkpoints':{name:{'sha256':p.R_SHA} for name in (p.R_NAME,p.R44_NAME)}}
    monkeypatch.setattr(p,'weight_manifest',lambda *a,**kw:manifest)
    namespace=dict(cases.__dict__,p=p,expected_manifest=manifest)
    exec(compile(source,'<R2 inherited real-output orchestration>','exec'),namespace)
    namespace['test_real_data_cpu_orchestration_smoke'](tmp_path,visible,monkeypatch,fault)


def test_pending_finalization_cpu_fixture(tmp_path, monkeypatch):
    # Only the temporary test stage receives a seed43-as-seed44 fixture.
    monkeypatch.setattr(p, 'R44_SOURCE', tmp_path/'not-yet-trained.pt')
    stage=tmp_path/'stage'
    p.prepare(stage)
    p.register(stage)
    assert p.finalize(stage)['status']=='BLOCKED_SEED44'
    monkeypatch.setattr(p, 'R44_SOURCE', p.R_SOURCE)
    p.finalize(stage)
    assert not p.validate_dataset(stage)['pending_seed44']
    assert p.weight_manifest(stage/p.DATASET_DIR)['checkpoints'][p.R44_NAME]['sha256']==p.R_SHA
    assert p.validate(stage)['status']=='PASS_CPU_VALIDATE'
    p.register(stage)
    with pytest.raises(ValueError, match='fresh CPU replay of actual seed44 required'):
        p.admit(stage)
    with pytest.raises(ValueError, match='only pending'):
        p.finalize(stage)
