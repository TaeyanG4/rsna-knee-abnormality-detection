"""H-B-168 strict archive scope and independent fluid/combined-R CPU oracle."""
import ast
import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tools import c85r7_stackq_v1125 as b


@pytest.fixture(scope='module')
def candidate():
    return b.math_functions(b.payload()[2])


def inputs(n=1200):
    rng=np.random.default_rng(168)
    return [f's{i}' for i in range(n)], [rng.integers(1,49,(n,12))/50 for _ in range(6)]


def oracle(arrays,masks):
    """Column-wise independent oracle: fluid skips R7 including its failure mask."""
    def rank(values,failed):
        out=np.full(len(values),np.nan)
        out[~failed]=pd.Series(values[~failed]).rank(method='average',pct=True)
        return out
    out=np.zeros_like(arrays[0])
    rf,f224,sf,*ofs=masks
    for col in range(12):
        a,bb,s,*owns=[rank(v[:,col],f) for v,f in zip(arrays,masks)]
        rr=np.where(rf,bb,np.where(f224,a,.65*a+.35*bb))
        failed_r=rf & f224
        rr[failed_r]=.5
        if col in (7,9):
            public,pf=rr,failed_r
        else:
            public=.5*rr+.5*s
            public[failed_r]=s[failed_r]
            public[sf]=rr[sf]
            pf=failed_r & sf
            public[pf]=.5
        den=sum(w*(~f).astype(int) for w,f in zip((17,17,12),ofs))
        own=sum(w*np.nan_to_num(v) for w,v in zip((17,17,12),owns))/np.maximum(den,1)
        pr,oo=rank(public,pf),rank(own,den==0)
        result=.85*pr+.15*oo
        result[den==0]=pr[den==0]
        result[pf]=oo[pf]
        result[pf & (den==0)]=.5
        out[:,col]=result
    return out


def failures(ids,masks):
    return {name:{uid:'fixture' for uid,failed in zip(ids,mask) if failed}
            for name,mask in zip(('R','R224','S_R7','EXP143v2','EXP155','Q_A5q'),masks)}


@pytest.mark.parametrize('case',['healthy','all64','constant','own_failed','r384_failed','r7_failed'])
def test_oracle_and_unchanged_ten(candidate,case):
    ids,arrays=inputs()
    masks=[np.zeros(len(ids),bool) for _ in range(6)]
    if case=='all64':
        for row,bits in enumerate(itertools.product((False,True),repeat=6)):
            for mask,bit in zip(masks,bits): mask[row]=bit
    if case=='constant':
        for a in arrays: a[:]=.2
    if case=='own_failed':
        for m in masks[3:]: m[:]=True
    if case=='r384_failed': masks[0][:]=True
    if case=='r7_failed': masks[2][:]=True
    failed=failures(ids,masks)
    actual=candidate['blend'](ids,*arrays,failed)
    np.testing.assert_array_equal(actual,oracle(arrays,masks))
    prior=b.math_functions(b.parent()[1])['blend'](ids,*arrays,failed)
    keep=[i for i in range(12) if i not in (7,9)]
    np.testing.assert_array_equal(actual[:,keep],prior[:,keep])
    assert np.isfinite(actual).all() and (actual>=0).all() and (actual<=1).all()
    if case in ('healthy','constant'):
        np.testing.assert_array_equal(actual,candidate['reference'](*arrays))


def test_fluid_independent_of_r7_and_uses_reranked_combined_r(candidate):
    ids,arrays=inputs(100)
    baseline=candidate['blend'](ids,*arrays)
    arrays[2]=arrays[2][::-1].copy()
    changed=candidate['blend'](ids,*arrays,{'S_R7':dict.fromkeys(ids,'failed')})
    np.testing.assert_array_equal(baseline[:,[7,9]],changed[:,[7,9]])
    rank=lambda a:pd.DataFrame(a).rank(pct=True).to_numpy()
    rr=.65*rank(arrays[0])+.35*rank(arrays[1])
    oo=rank((17*rank(arrays[3])+17*rank(arrays[4])+12*rank(arrays[5]))/46)
    assert np.max(abs(baseline[:,[7,9]]-(.85*rr+.15*oo)[:,[7,9]]))>.01
    assert np.max(abs(baseline[:,[7,9]]-(.85*rank(arrays[0])+.15*oo)[:,[7,9]]))>.01


def test_three_percent_and_sentinel(candidate):
    ids,arrays=inputs(100)
    arrays[1][:3]=.5
    candidate['blend'](ids,*arrays)
    arrays[1][3]=.5
    with pytest.raises(ValueError,match='more than 3%'): candidate['blend'](ids,*arrays)


@pytest.mark.parametrize('bad',['nan','inf','negative','shape','duplicate'])
def test_invalid_input(candidate,bad):
    ids,arrays=inputs(100)
    if bad=='nan': arrays[1][0,0]=np.nan
    if bad=='inf': arrays[1][0,0]=np.inf
    if bad=='negative': arrays[1][0,0]=-.1
    if bad=='shape': arrays[1]=arrays[1][:,:11]
    if bad=='duplicate': ids[0]=ids[1]
    with pytest.raises(ValueError): candidate['blend'](ids,*arrays)


def test_exact_archive_bootstrap_and_mechanics():
    nb,meta,files,blob,manifest,full=b.artifacts()
    pnb,original,_,_=b.parent()
    b.scope(nb,meta)
    assert len(manifest['unchanged'])==29 and len(manifest['changed'])==2
    assert not manifest['added'] and not manifest['removed']
    assert set(files)==set(original)
    assert b.old.ROUTE.encode() in files[b.HELPER]
    for item in manifest['unchanged']: assert files[item['name']]==original[item['name']]
    assert nb['cells'][:-1]==pnb['cells'][:-1]
    # Unchanged function ASTs prove no process, import, pool, thread or timing edits.
    for name in (b.HELPER,b.WRAPPER):
        before=ast.parse(original[name]); after=ast.parse(files[name])
        def select(tree):
            return {n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        a,z=select(before),select(after)
        changed={k for k in a if a[k]!=z[k]}
        assert changed==({'p_branch','blend','reference'} if name==b.HELPER else {'collect','runtime_pins'})
        if name==b.WRAPPER:
            # The pin function changes only the chained digest literal.
            assert b.old.replace(original[name],b.old.digest(original[b.HELPER]),b.old.digest(files[b.HELPER])).decode().split('def runtime_pins',1)[1].split('\ndef ',1)[0]==files[name].decode().split('def runtime_pins',1)[1].split('\ndef ',1)[0]
        assert [ast.dump(n) for n in before.body if isinstance(n,(ast.Import,ast.ImportFrom))]==[ast.dump(n) for n in after.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    pm=b.old.read(b.BASE/'kernel/kernel-metadata.json')
    assert {k:v for k,v in pm.items() if k not in ('id','title')}=={k:v for k,v in meta.items() if k not in ('id','title')}
    assert meta['id']==b.SLUG and meta['title']=='knee-lb-stackq-v1'
    assert all(k.encode() in full for k in files)
    assert b.payload()[3]==blob


@pytest.mark.parametrize('kind',['bootstrap','metadata'])
def test_reject_scope_mutations(kind):
    nb,meta,*_=b.payload()
    if kind=='bootstrap': nb['cells'][-1]['source'].append('\nprint("extra")\n')
    else: meta['enable_internet']=not meta.get('enable_internet',False)
    with pytest.raises(ValueError,match='forbidden'): b.scope(nb,meta)


@pytest.mark.parametrize('ps,cs,hours,expected',[(.957,.956,9,True),(.957,.95599,8,False),(.957,.958,9.001,False)])
def test_registered_decision(ps,cs,hours,expected):
    assert b.decision(ps,cs,hours)==expected


@pytest.mark.parametrize('values',[(None,.957,8),(.957,float('nan'),8),(.957,.957,0),(.957,2,8)])
def test_decision_reject_unmeasured(values):
    with pytest.raises(ValueError): b.decision(*values)


def test_mapping_and_stage_identity():
    reg=b.old.read(b.OUT/'registration.json')
    assert reg==b.registration()
    assert 'OMITTED' in reg['compo_mapping']['status']
    assert reg['purpose']=='final-pair candidate'
    assert reg['public_reading']['reference_value'] is None
    assert not reg['qualified_for_submission']
    assert b.validate()['identical']==29


def test_isolated_embedded_import_and_real_collect(tmp_path):
    """Actual embedded imports, with cwd outside repository and no tools fallback."""
    runtime=b.SCRATCH/'candidate-runtime'
    script='''
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
runtime=Path(sys.argv[1]); root=Path(sys.argv[2]); tmp=Path(sys.argv[3])
sys.path.insert(0,str(runtime))
import exp169_release_c85r7codec as c
import exp169_r224 as h
assert Path(c.__file__).resolve().parent==runtime.resolve()
assert Path(h.__file__).resolve().parent==runtime.resolve()
assert c.VERSION=='1.1.25-stackq'
assert c.r224 is h
assert all(not n.startswith('tools.') for n in sys.modules)
c.p.REPLAY=Path(sys.argv[4])
staged=c.stage_replay_input(tmp/'pr7')
ids,r,ours,_=c.authenticated_pr7(staged)
work=staged/'pr7-run'; csvdir=staged/'csv'; csvdir.mkdir(exist_ok=True)
for name in (h.RAW,h.SUB):
    frame=pd.DataFrame(r,columns=c.p.LABELS); frame.insert(0,'StudyInstanceUID',ids)
    frame.to_csv(csvdir/name,index=False)
h.record(work,csvdir)
(staged/'R224-exit.json').write_text('{"exit_code":0}')
receipt,payload=c.collect(Path(sys.argv[5]),staged,tmp/'out')
assert receipt['status']=='COMPLETE_C85R7'
assert receipt['telemetry']['healthy_reference_exact']
assert receipt['fallbacks']['r224_prediction']==0
assert receipt['version']=='1.1.25-stackq'
assert (tmp/'out/submission.csv').read_bytes()==payload
print(json.dumps({'status':'PASS','studies':len(ids),'targets':12,'r224':'explicit synthetic real-R384 surrogate','r7':'recorded S6 stand-in','actual_embedded_imports':True}))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    from tools import exp169_release_c85r7codec as original
    run=subprocess.run([sys.executable,'-I','-c',script,str(runtime),str(b.ROOT),str(tmp_path),str(original.p.REPLAY),str(original.c85.A5Q)],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert run.returncode==0,run.stderr[-7000:]
    assert '"actual_embedded_imports": true' in run.stdout


def test_admit_rejects_stale_smoke(monkeypatch):
    original=b.old.read
    def read(path):
        if path==b.SCRATCH/'replay.json': return {'status':'PASS','kernel_sha256':'stale'}
        return original(path)
    monkeypatch.setattr(b.old,'read',read)
    with pytest.raises(ValueError,match='missing/stale'): b.admit()
