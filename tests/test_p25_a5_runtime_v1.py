"""CPU regressions for the fixed A5 member; synthetic predictions, no labels."""
import contextlib
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tools import p25_a5_runtime_v1 as a
from tools import k1n_cpu_replay_v1 as replay


@pytest.mark.parametrize('n,missing', [(1,0),(1,1),(17,0),(17,3),(17,17)])
def test_formula_full_roster_ties_singleton_absent(n, missing):
    rng = np.random.default_rng(619)
    parent = rng.integers(0,9,(n,12)).astype(np.float32)/8
    raw = rng.integers(0,7,(5,n,12)).astype(np.float32)/6
    mask = np.arange(n) >= missing
    raw[:,~mask] = np.nan
    payload = replay.csv_bytes(list(map(str,range(n))),parent)
    base = pd.DataFrame(parent).rank(pct=True).to_numpy()
    mixed = base.copy()
    if mask.any():
        count = mask.sum()
        ranked = [(pd.DataFrame(f[mask]).rank().to_numpy()-1)/(count-1)
                  if count > 1 else np.full((1,12),.5) for f in raw]
        mean = np.mean(ranked,axis=0).astype(np.float32)
        mixed[mask] = .85*base[mask]+.15*pd.DataFrame(mean).rank(pct=True).to_numpy()
    expected = pd.DataFrame(mixed).rank(pct=True).to_numpy().astype(np.float32)
    ids, actual = a.util.table(a.blend(payload,raw,mask))
    assert ids == list(map(str,range(n)))
    np.testing.assert_array_equal(actual,expected)
    if n > 1 and not missing:
        order = rng.permutation(n)
        shuffled = a.blend(replay.csv_bytes([str(i) for i in order],parent[order]),raw[:,order],mask)
        np.testing.assert_array_equal(a.util.table(shuffled,ids)[1],actual)


@pytest.mark.parametrize('fault',['shape','nan','range','absent_not_nan'])
def test_blend_failures(fault):
    raw=np.full((5,2,12),.5)
    mask=np.array([True,True])
    if fault=='shape': raw=raw[:4]
    if fault=='nan': raw[0,0,0]=np.nan
    if fault=='range': raw[0,0,0]=1.1
    if fault=='absent_not_nan': mask[0]=False
    with pytest.raises(ValueError):
        a.blend(replay.csv_bytes(['a','b'],np.full((2,12),.2)),raw,mask)


def checkpoint():
    return dict(fold=0,cfg=copy.deepcopy(a.CFG),labels=a.util.LABELS,
                state_dict={str(i):None for i in range(180)})


@pytest.mark.parametrize('fault',['fold','img','missing_backbone','labels','keys'])
def test_checkpoint_contract(fault):
    z=checkpoint()
    a.check_checkpoint(z,0)
    if fault=='fold': z['fold']=1
    if fault=='img': z['cfg']['img']=224
    if fault=='missing_backbone': del z['cfg']['backbone']
    if fault=='labels': z['labels']=list(reversed(z['labels']))
    if fault=='keys': z['state_dict'].pop('0')
    with pytest.raises((KeyError,ValueError)): a.check_checkpoint(z,0)


def test_hash_before_pickle_and_strict_load(tmp_path,monkeypatch):
    path=tmp_path/'m_f0.pt'
    path.write_bytes(b'synthetic checkpoint, not an anchor fold')
    calls=[]
    def load(*args,**kwargs):
        calls.append(kwargs)
        return checkpoint()
    monkeypatch.setattr(torch,'load',load)
    with pytest.raises(ValueError,match='hash/file'): a.load_fold(path,0)
    assert not calls
    monkeypatch.setattr(a,'HASHES',(a.util.sha(path),*a.HASHES[1:]))
    class Net:
        def load_state_dict(self,state,strict): assert strict and len(state)==180
        def eval(self): return self
    monkeypatch.setattr(a,'model',Net)
    assert isinstance(a.load_fold(path,0),Net)
    assert calls==[dict(map_location='cpu',weights_only=False)]


def test_micro_active_slots_and_failure():
    images=np.zeros((3,6,16,2,2),np.uint8)
    masks=np.zeros((3,6),np.uint8)
    masks[0,[0,2]]=16; masks[2,5]=16
    images[0,[0,2]]=255; images[2,5]=128
    class Net:
        def __call__(self,im,sl,sm,si,b,vm):
            assert sl.tolist()==[1,3,6] and si.tolist()==[0,0,1] and b==2
            assert sm.shape==(3,0) and vm.all() and im.shape==(3,16,2,2)
            return torch.zeros(2,12)
    result=a.micro(images,masks,dev='cpu',model_group=[Net()]*5)
    assert result.shape==(5,3,12) and np.isnan(result[:,1]).all()
    assert (result[:,[0,2]]==.5).all()
    class Bad:
        def __call__(self,*args,**kwargs): return torch.full((2,12),float('nan'))
    with pytest.raises(ValueError,match='neutral fallback'):
        a.micro(images,masks,dev='cpu',model_group=[Bad()]*5)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('n',[1,7,8,9,32,48,49])
def test_micro_ranges(n):
    ranges=a.ranges(n)
    flat=sorted(x for group in ranges for x in group)
    assert flat==[(s,min(s+8,n)) for s in range(0,n,8)]
    assert all((s//8)%2==d for d,group in enumerate(ranges) for s,e in group)


def test_codec_identity(tmp_path,monkeypatch):
    path=tmp_path/'knee_hb126_hook.py'; path.write_text('# fixture')
    receipt=dict(status='ACTIVE',hook_dir=str(tmp_path),hook_sha256=a.util.sha(path))
    hook=SimpleNamespace(__file__=str(path),KNEE_HB126_STATE=dict(patched=True,errors=[]))
    monkeypatch.setitem(sys.modules,'knee_hb126_hook',hook)
    assert a.activate_codec(receipt)
    with pytest.raises(ValueError,match='hook hash'):
        a.activate_codec(dict(receipt,hook_sha256='bad'))
    with pytest.raises(ValueError,match='active inherited'):
        a.activate_codec(dict(receipt,status='ABSENT'))
    hook.KNEE_HB126_STATE['patched']=False
    with pytest.raises(ValueError,match='activation'): a.activate_codec(receipt)


@pytest.mark.parametrize('diagnostic',[False,True])
@pytest.mark.parametrize('fail',[False,True])
def test_run_cpu_orchestration(tmp_path,monkeypatch,diagnostic,fail):
    from tools import p25_a5_render_v1 as render
    from tools import l1_rco_k2_worker_v102 as w
    # Exercise two blocks/prefetch/receipt flow with tiny CPU tensors.
    # CUDA transport/inference is replaced explicitly; this is not GPU qualification.
    comp=tmp_path/'competition'; (comp/'test_series').mkdir(parents=True)
    n=49
    payload=replay.csv_bytes(list(map(str,range(n))),np.full((n,12),.4))
    (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(payload)
    pd.DataFrame({a.util.UID:list(map(str,range(n))),'SeriesInstanceUID':['series']*n}).to_csv(comp/'test_series.csv',index=False)
    monkeypatch.setitem(sys.modules,'tools.k1sr_runtime_v10',SimpleNamespace(phase_deadline=lambda *x:contextlib.nullcontext()))
    monkeypatch.setitem(sys.modules,'timm',SimpleNamespace(__version__='1.0.25'))
    monkeypatch.setattr(a,'activate_codec',lambda x:'cpu-fixture')
    monkeypatch.setattr(a,'checkpoint_root',lambda:tmp_path)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'device',lambda d:contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    class Model:
        def to(self,*args): return self
        def eval(self): return self
    loaded=[]
    def load(p,f): loaded.append(f); return Model()
    monkeypatch.setattr(a,'load_fold',load)
    def build(args):
        i,u,rows=args
        if fail: raise RuntimeError('injected acquired decode failure')
        return i,np.zeros((6,16,336,336),np.uint8),np.array([16,0,0,0,0,0],np.uint8)
    monkeypatch.setattr(render,'build_study',build)
    calls=[]
    def predict(images,masks,groups,executor):
        calls.append(len(masks)); return np.full((5,len(masks),12),.5,np.float32)
    monkeypatch.setattr(a,'predict',predict)
    if fail:
        with pytest.raises(RuntimeError,match='injected acquired'):
            a.run(tmp_path,comp,{},diagnostic)
    else:
        output=a.run(tmp_path,comp,{},diagnostic)
        assert calls==[48,1] and loaded==list(range(5))
        assert a.util.table(output)[0]==list(map(str,range(n)))
    receipt=w.read(tmp_path/'a5-receipt.json')
    assert receipt['status']==('FAILED_A5' if fail else 'COMPLETE_A5')
    assert receipt['never_submit']==diagnostic and not receipt['qualified_for_submission']
    if diagnostic: assert not (tmp_path/'submission.csv').exists()
    assert not torch.cuda.is_initialized()
