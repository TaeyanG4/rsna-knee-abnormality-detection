"""H-B-180: all failure combinations, exact archive scope and embedded collect."""
import ast
import itertools
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tools import c85r7_r224flipoaq_v1133 as b


@pytest.fixture(scope='module')
def candidate():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    return b.math_functions(b.payload()[2])


def inputs(n=1200):
    rng=np.random.default_rng(180)
    return [f's{i}' for i in range(n)], [rng.integers(1,49,(n,12))/50 for _ in range(6)]


def failures(ids,masks):
    return {name:dict.fromkeys([u for u,f in zip(ids,m) if f],'fixture') for name,m in zip(b.MEMBERS,masks)}


@pytest.mark.parametrize('case',['healthy','all64','constant','own_failed','r384_failed','compact_failed'])
def test_independent_oracle_protected_ten_and_outer_failures(candidate,case):
    ids,arrays=inputs()
    masks=[np.zeros(len(ids),bool) for _ in arrays]
    if case=='all64':
        for row,bits in enumerate(itertools.product((False,True),repeat=6)):
            for mask,bit in zip(masks,bits): mask[row]=bit
    elif case=='constant':
        for a in arrays: a[:]=.2
    elif case=='own_failed':
        for m in masks[3:]: m[:]=True
    elif case=='r384_failed': masks[0][:]=True
    elif case=='compact_failed': masks[2][:]=True
    actual=candidate['blend'](ids,*arrays,failures(ids,masks))
    expected=b.oracle(arrays,masks)
    assert actual.tobytes()==expected.tobytes()
    prior=b.math_functions(b.parent()[1])['blend'](ids,*arrays,failures(ids,masks))
    assert np.ascontiguousarray(actual[:,b.PROTECTED]).tobytes()==np.ascontiguousarray(prior[:,b.PROTECTED]).tobytes()
    assert np.isfinite(actual).all() and np.all((actual>=0)&(actual<=1))
    if case in ('healthy','constant'):
        np.testing.assert_array_equal(actual,candidate['reference'](*arrays))
    if case=='all64':
        assert np.all(actual[63]==.5)  # every member failed
        # Own weights renormalize independently; all own failures use public alone.
        assert np.sum(masks[1])==32 and 32/len(ids)<.03


def test_eight_public_truth_rows_and_success_only_ranks(candidate):
    ids,arrays=inputs(400)
    masks=[np.zeros(len(ids),bool) for _ in range(3)]
    bits=list(itertools.product((False,True),repeat=3))
    for row,combination in enumerate(bits):
        for mask,bit in zip(masks,combination): mask[row]=bit
    def rank(a,m):
        z=np.zeros(a.shape)
        z[~m]=pd.DataFrame(a[~m]).rank(method='average',pct=True).to_numpy()
        return z
    ar,br,sr=[rank(a,m) for a,m in zip(arrays,masks)]
    public=candidate['p_branch'](*arrays[:3],*masks)
    for row,(rf,f224,sf) in enumerate(bits):
        if not sf: expected=sr[row]
        elif rf and f224: expected=np.full(12,.5)
        elif rf: expected=br[row]
        elif f224: expected=ar[row]
        else: expected=.65*ar[row]+.35*br[row]
        np.testing.assert_array_equal(public[row,[5,6]],expected[[5,6]])
    # Failed compact rows must not change any healthy compact percentile.
    changed=[a.copy() for a in arrays]
    changed[2][masks[2]]=1
    np.testing.assert_array_equal(public[:,[5,6]],candidate['p_branch'](*changed[:3],*masks)[:,[5,6]])


def test_oa_ignores_r_when_compact_healthy_and_keeps_own(candidate):
    ids,arrays=inputs(100)
    actual=candidate['blend'](ids,*arrays)
    changed=[a.copy() for a in arrays]
    changed[0]=changed[0][::-1].copy(); changed[1]=changed[1][::-1].copy()
    alternate=candidate['blend'](ids,*changed,{'R':dict.fromkeys(ids,'fixture')})
    np.testing.assert_array_equal(actual[:,[5,6]],alternate[:,[5,6]])
    rank=lambda a:pd.DataFrame(a).rank(method='average',pct=True).to_numpy()
    own=rank((17*rank(arrays[3])+17*rank(arrays[4])+12*rank(arrays[5]))/46)
    np.testing.assert_array_equal(actual[:,[5,6]],(.85*rank(arrays[2])+.15*own)[:,[5,6]])
    assert np.max(np.abs(actual[:,[5,6]]-rank(arrays[2])[:,[5,6]]))>.01


def test_r224_three_percent_abort_unchanged(candidate):
    ids,arrays=inputs(100)
    arrays[1][:3]=.5
    candidate['blend'](ids,*arrays)
    arrays[1][3]=.5
    with pytest.raises(ValueError,match='more than 3%'): candidate['blend'](ids,*arrays)
    with pytest.raises(ValueError): candidate['failure_gate'](np.array([],bool))


@pytest.mark.parametrize('bad',['nan','inf','negative','over_one','shape','duplicate'])
def test_invalid_rejected(candidate,bad):
    ids,arrays=inputs(100)
    if bad=='nan': arrays[2][0,5]=np.nan
    if bad=='inf': arrays[2][0,6]=np.inf
    if bad=='negative': arrays[2][0,5]=-.1
    if bad=='over_one': arrays[2][0,6]=1.1
    if bad=='shape': arrays[2]=arrays[2][:,:11]
    if bad=='duplicate': ids[0]=ids[1]
    with pytest.raises(ValueError): candidate['blend'](ids,*arrays)


def test_all_embedded_files_and_function_scope():
    nb,meta,files,blob,manifest,full=b.artifacts()
    pnb,original,_,_=b.parent()
    b.scope(nb,meta)
    assert len(manifest['unchanged'])==29 and len(manifest['changed'])==2
    assert not manifest['added'] and not manifest['removed']
    assert set(files)==set(original)
    for item in manifest['unchanged']: assert files[item['name']]==original[item['name']]
    assert nb['cells'][:-1]==pnb['cells'][:-1]
    for name in (b.HELPER,b.WRAPPER):
        before,after=ast.parse(original[name]),ast.parse(files[name])
        def functions(tree):
            return {n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        a,z=functions(before),functions(after)
        assert a.keys()==z.keys()
        assert {k for k in a if a[k]!=z[k]}==({'p_branch','reference'} if name==b.HELPER else {'collect','runtime_pins'})
        assert [ast.dump(n) for n in before.body if isinstance(n,(ast.Import,ast.ImportFrom))]==[ast.dump(n) for n in after.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    def function_bytes(raw,name):
        return raw.split(('def '+name+'(').encode(),1)[1].split(b'\n\ndef ',1)[0]
    assert function_bytes(original[b.HELPER],'blend')==function_bytes(files[b.HELPER],'blend')
    assert b.old.replace(original[b.WRAPPER],b.old.digest(original[b.HELPER]),b.old.digest(files[b.HELPER])).decode().split('def runtime_pins',1)[1].split('\ndef ',1)[0]==files[b.WRAPPER].decode().split('def runtime_pins',1)[1].split('\ndef ',1)[0]
    pm=b.old.read(b.BASE/'kernel/kernel-metadata.json')
    assert {k:v for k,v in pm.items() if k not in ('id','title')}=={k:v for k,v in meta.items() if k not in ('id','title')}
    assert meta['id']==b.SLUG and meta['title']=='knee-lb-r224flipoaq-v1'
    assert all(k.encode() in full for k in files)
    assert b.payload()[3]==blob


@pytest.mark.parametrize('kind',['bootstrap','metadata'])
def test_scope_mutation_fails_closed(kind):
    nb,meta,*_=b.payload()
    if kind=='bootstrap': nb['cells'][-1]['source'].append('\nprint("extra")\n')
    else: meta['enable_internet']=not meta.get('enable_internet',False)
    with pytest.raises(ValueError,match='forbidden'): b.scope(nb,meta)


@pytest.mark.parametrize('value,result',[(.956,'DROP'),(.957,'UNRESOLVED_RESERVE'),(.958,'RETAIN')])
def test_registered_decision(value,result):
    assert b.decision(value)==result


@pytest.mark.parametrize('bad',[None,float('nan'),float('inf'),True,-.1,1.1])
def test_decision_rejects_invalid(bad):
    with pytest.raises(ValueError): b.decision(bad)


def test_registered_stage_identity(candidate):
    reg=b.old.read(b.OUT/'registration.json')
    assert reg==b.registration() and reg['routed_columns']==[5,6]
    assert reg['exposure']=='EXPOSED' and not reg['clean_oof'] and not reg['qualified_for_submission']
    assert reg['public_reading']['reference_value']==.957
    assert b.validate()['identical']==29
    assert list(candidate['p'].LABELS[5:7])==['Lateral OA','PF OA']


def test_isolated_embedded_collect(tmp_path):
    """Smoke the unchanged collect wiring; R224 here is an explicit surrogate.

    Separate cpu-replay uses authenticated real R224/R7-flip/own data for two
    actual studies. This test checks collection/reference plumbing only.
    """
    runtime=b.SCRATCH/'candidate-runtime'
    script=r'''
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
import pandas as pd
runtime=Path(sys.argv[1]); tmp=Path(sys.argv[2]); sys.path.insert(0,str(runtime))
import exp169_release_c85r7codec as c
import exp169_r224 as h
assert Path(c.__file__).resolve().parent==runtime.resolve() and c.r224 is h
assert all(not n.startswith('tools.') for n in sys.modules)
assert c.VERSION=='1.1.33-r224flipoaq'
c.p.REPLAY=Path(sys.argv[3])
staged=c.stage_replay_input(tmp/'pr7')
ids,r,ours,_=c.authenticated_pr7(staged)
work=staged/'pr7-run'; csvdir=staged/'csv'; csvdir.mkdir(exist_ok=True)
for name in (h.RAW,h.SUB):
    frame=pd.DataFrame(r,columns=c.p.LABELS); frame.insert(0,'StudyInstanceUID',ids)
    frame.to_csv(csvdir/name,index=False)
h.record(work,csvdir)
(staged/'R224-exit.json').write_text('{"exit_code":0}')
receipt,payload=c.collect(Path(sys.argv[4]),staged,tmp/'out')
assert receipt['status']=='COMPLETE_C85R7' and receipt['telemetry']['healthy_reference_exact']
assert receipt['fallbacks']['r224_prediction']==0 and receipt['version']=='1.1.33-r224flipoaq'
assert (tmp/'out/submission.csv').read_bytes()==payload
import torch
assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
print(json.dumps({'status':'PASS','studies':len(ids),'embedded_collect':True,'R224':'explicit R384 surrogate','R7':'recorded S6 stand-in'}))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    from tools import exp169_release_c85r7codec as original
    run=subprocess.run([sys.executable,'-B','-I','-c',script,str(runtime),str(tmp_path),str(original.p.REPLAY),str(original.c85.A5Q)],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert run.returncode==0,run.stderr[-7000:]
    assert '"embedded_collect": true' in run.stdout


def test_admit_rejects_stale_smoke(monkeypatch):
    original=b.old.read
    def read(path):
        if path==b.SCRATCH/'replay.json': return {'status':'PASS','kernel_sha256':'stale'}
        return original(path)
    monkeypatch.setattr(b.old,'read',read)
    with pytest.raises(ValueError,match='missing/stale'): b.admit()
