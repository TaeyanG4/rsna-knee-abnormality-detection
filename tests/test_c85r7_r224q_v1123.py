"""H-B-164 CPU gates: exact archive scope, public failure oracle and entrypoints."""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import c85r7_r224q_v1123 as b
from tools import exp169_r224 as h
from tools import exp169_release_c85r7codec as parent


@pytest.fixture
def candidate():
    mod = types.ModuleType('hb164_candidate')
    mod.__file__ = str(b.SCRATCH / 'candidate-runtime' / b.WRAPPER)
    exec(compile(b.patched(b.old.parent()[1])[b.WRAPPER], b.WRAPPER, 'exec'), mod.__dict__)
    return mod


def samples(n=100):
    rng = np.random.default_rng(164)
    return [f's{i}' for i in range(n)], [rng.integers(1, 49, (n, 12)) / 50 for _ in range(6)]


def oracle(arrays, masks):
    """Independent per-target public np.where logic plus frozen outer policy."""
    def rank(v, failed):
        result = np.full(v.shape, np.nan)
        result[~failed] = pd.Series(v[~failed]).rank(pct=True).to_numpy()
        return result
    out = np.zeros_like(arrays[0])
    for j in range(12):
        rr, r224, ours, e143, e155, q = [rank(v[:, j], f) for v, f in zip(arrays, masks)]
        rf, f224, sf, *of = masks
        rhalf = np.where(rf, r224, np.where(f224, rr, .65 * rr + .35 * r224))
        pf = rf & f224 & sf
        pp = .5 * rhalf + .5 * ours
        pp[rf & f224] = ours[rf & f224]
        pp[sf] = rhalf[sf]
        pp[pf] = .5
        den = sum(w * (~f).astype(int) for w, f in zip((17,17,12), of))
        own = sum(w * np.nan_to_num(v) for w, v in zip((17,17,12), (e143,e155,q))) / np.maximum(den,1)
        pr, oo = rank(pp, pf), rank(own, den == 0)
        result = .85 * pr + .15 * oo
        result[den == 0] = pr[den == 0]
        result[pf] = oo[pf]
        result[pf & (den == 0)] = .5
        out[:,j] = result
    return out


@pytest.mark.parametrize('mode', ['healthy','failures','all_own_fail','all_r384_fail','all_r7_fail'])
def test_independent_pandas_oracle(mode):
    ids, arrays = samples()
    masks = [np.zeros(100, bool) for _ in range(6)]
    if mode == 'failures':
        for m, rows in zip(masks, ([0,1,3],[0,1,2],[0,4],[0,2],[0,5],[0,6])):
            m[rows] = True
    elif mode == 'all_own_fail':
        for m in masks[3:]: m[:] = True
    elif mode == 'all_r384_fail': masks[0][:] = True
    elif mode == 'all_r7_fail': masks[2][:] = True
    failures = {name:{uid:'synthetic failure' for uid,f in zip(ids,mask) if f}
                for name,mask in zip(['R','R224','S_R7','EXP143v2','EXP155','Q_A5q'], masks)}
    actual = h.blend(ids,*arrays,failures)
    np.testing.assert_array_equal(actual,oracle(arrays,masks))
    if mode == 'healthy': np.testing.assert_array_equal(actual,h.reference(*arrays))


def test_sentinel_and_exact_three_percent_gate():
    ids, arrays = samples()
    arrays[1][:3] = .5
    h.blend(ids,*arrays)
    arrays[1][3] = .5
    with pytest.raises(ValueError,match='more than 3%'): h.blend(ids,*arrays)
    with pytest.raises(ValueError): h.failure_gate(np.array([],bool))


@pytest.mark.parametrize('bad', ['nan','inf','negative','shape','duplicate'])
def test_invalid_predictions_reject(bad):
    ids, arrays = samples()
    if bad == 'nan': arrays[1][0,0] = np.nan
    if bad == 'inf': arrays[1][0,0] = np.inf
    if bad == 'negative': arrays[1][0,0] = -.1
    if bad == 'shape': arrays[1] = arrays[1][:,:11]
    if bad == 'duplicate': ids[0] = ids[1]
    with pytest.raises(ValueError): h.blend(ids,*arrays)


def r224_files(tmp, ids, raw, sub=None):
    work = tmp/'pr7-run'
    work.mkdir(parents=True,exist_ok=True)
    csvdir = tmp/'csv'
    csvdir.mkdir(exist_ok=True)
    for name, values in [(h.RAW,raw),(h.SUB,raw if sub is None else sub)]:
        frame=pd.DataFrame(values,columns=h.p.LABELS)
        frame.insert(0,'StudyInstanceUID',ids)
        frame.to_csv(csvdir/name,index=False)
    h.record(work,csvdir)
    (tmp/'R224-exit.json').write_text(json.dumps({'exit_code':0}))


@pytest.mark.parametrize('fault', ['none','tamper','exit','identity','cohort','duplicate','rank','failures'])
def test_output_authentication(tmp_path,fault):
    ids, arrays = samples()
    raw = arrays[1]
    sub = 1-raw if fault=='rank' else raw.copy()
    saved_ids = ids.copy()
    if fault=='failures': raw[:4] = .5
    if fault=='cohort': saved_ids[0]='alien'
    if fault=='duplicate': saved_ids[0]=saved_ids[1]
    r224_files(tmp_path,saved_ids,raw,sub)
    if fault=='tamper': (tmp_path/'csv'/h.RAW).write_bytes(b'tampered')
    if fault=='exit': (tmp_path/'R224-exit.json').write_text('{"exit_code":1}')
    if fault=='identity':
        path=tmp_path/'pr7-run/r224-receipt.json'
        value=json.loads(path.read_text()); value['checkpoint_sha256']='0'*64
        path.write_text(json.dumps(value))
    if fault=='none':
        values,failures,_=h.read(tmp_path,list(reversed(ids)))
        np.testing.assert_allclose(values,raw[::-1],rtol=0,atol=3e-8)
        assert not failures
    else:
        with pytest.raises(ValueError): h.read(tmp_path,ids)


def test_exact_scope_and_source():
    nb,meta,files,blob,manifest,full=b.artifacts()
    original=b.old.parent()[1]
    assert len(manifest['unchanged'])==28
    assert [x['name'] for x in manifest['changed']]==[b.WRAPPER]
    assert set(files)-set(original)=={'exp169_r224.py','raptor224_run.py'}
    for item in manifest['unchanged']: assert files[item['name']]==original[item['name']]
    assert b.old.digest(files[h.SOURCE])=='bc29b23a7aac8ace281619b71c8b49d8c4ac83e645d4e29e457e7e83f477fe2c'
    assert nb['cells'][0]==b.old.read(b.fast.OUT/'kernel/release.ipynb')['cells'][0]
    assert b.old.unpack(b.fastq.restored(nb))[0]==files
    parent_meta=b.old.read(b.old.BASE/'kernel/kernel-metadata.json')
    assert {k:v for k,v in meta.items() if k not in ['id','title']}=={k:v for k,v in parent_meta.items() if k not in ['id','title']}
    assert meta['id']==b.SLUG and meta['title']=='knee-lb-r224q-v1'
    assert b.registration()['flip_tta'] is False
    assert b.payload()[3]==blob
    assert all(k.encode() in full for k in files)


@pytest.mark.parametrize('score,state',[(.957,'ALIVE'),(.955,'DEAD'),(.956,'HOLD'),(.9573,'ALIVE')])
def test_registered_decisions(score,state):
    assert b.reading(score)==state
    assert b.registration()['timing']['max_projected_hours']==8.5


@pytest.mark.parametrize('fails',[False,True])
def test_collect_real_visible_predictions_with_explicit_synthetic_r224(tmp_path,candidate,fails):
    staged=parent.stage_replay_input(tmp_path/'pr7')
    ids,r,ours,_=parent.authenticated_pr7(staged)
    r224_files(staged,ids,r)
    if fails: (staged/'R224-exit.json').write_text('{"exit_code":1}')
    out=tmp_path/'out'
    if fails:
        with pytest.raises(ValueError): candidate.collect(candidate.c85.A5Q,staged,out)
        assert not (out/'submission.csv').exists()
        assert candidate.read_json(out/candidate.RECEIPT)['status']=='FAILED_EXECUTION'
    else:
        receipt,payload=candidate.collect(candidate.c85.A5Q,staged,out)
        assert receipt['status']=='COMPLETE_C85R7' and 'R224' in receipt['executed_members']
        assert receipt['fallbacks']['r224_prediction']==0
        assert (out/'submission.csv').read_bytes()==payload
        assert receipt['telemetry']['healthy_reference_exact']


@pytest.mark.parametrize('fail224',[False,True])
def test_new_subprocess_preserves_existing_commands(tmp_path,monkeypatch,candidate,fail224):
    runtime=tmp_path/'runtime'
    out=tmp_path/'run/pr7'; work=out/'pr7-run'; work.mkdir(parents=True)
    (out/'ids.json').write_text('["a"]')
    (work/'r7-identity.json').write_text('{"manifest_sha256":"identity"}')
    (work/'code-paths.json').write_text(json.dumps({'code':str(work/'code'),'anatomy':str(work/'anatomy.pt')}))
    monkeypatch.setattr(candidate,'runtime_pins',lambda *a:None)
    monkeypatch.setattr(candidate,'competition_root',lambda:tmp_path/'competition')
    monkeypatch.setattr(candidate.v5q,'_mount',lambda *a:tmp_path/'mount')
    monkeypatch.setattr(candidate.p,'check_pin',lambda *a:None)
    monkeypatch.setattr(candidate.s6,'install_wheels',lambda *a:None)
    events=[]; recorded=[]
    def launch(args,*,label,env):
        events.append((label,args,env))
        if fail224 and label=='C85R7 R224': raise RuntimeError('injected R224 exit')
    monkeypatch.setattr(candidate.v5q,'run_process_group',launch)
    monkeypatch.setattr(candidate.r7,'infer_studies',lambda *a,**k:events.append(('R7',a,k)))
    monkeypatch.setattr(h,'record',lambda *a:recorded.append(a))
    candidate.run_public(runtime,out)
    assert [x[0] for x in events]==['C85R7 R','C85R7 R224','R7']
    assert events[0][1]==[sys.executable,'-u',str(runtime/'exp169_release_c85.py'),'r-worker','--runtime',str(runtime),'--weights',str(tmp_path/'mount'/candidate.p.R_NAME),'--competition',str(tmp_path/'competition'),'--output',str(work/'r')]
    assert events[1][1]==[sys.executable,'-u',str(runtime/h.SOURCE)]
    assert events[0][2]==events[1][2]
    assert events[2][2]['workers']==2 and events[2][2]['device']=='cuda'
    assert len(recorded)==int(not fail224)
    assert candidate.read_json(out/'R224-exit.json')['exit_code']==int(fail224)


def test_fast_full_fixture_matrix(tmp_path,monkeypatch):
    from tools import c85r7_own25q3_cpu_fixtures as fixtures
    monkeypatch.setattr(fixtures,'b',types.SimpleNamespace(ROOT=b.ROOT,OUT=b.OUT,old=b.old,
        VISIBLE_IDS=b.fast.VISIBLE_IDS,reference=b.fast.reference))
    results=[fixtures.run_case(tmp_path/name,**kw) for name,kw in fixtures.case_specs()]
    assert len(results)==77 and all(x['status']=='PASS' for x in results)
