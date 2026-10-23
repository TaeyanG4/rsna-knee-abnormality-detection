"""H-B-175: actual archive integration, independent rank oracle and failure gates."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import c85r7_r224r76fq_v1131 as b
from tools.c85r7_r76_cpu_smoke import runtime


@pytest.fixture
def candidate():
    return runtime()


def samples(n=100):
    rng=np.random.default_rng(175)
    return [str(i) for i in range(n)], [rng.integers(1,49,(n,12))/50 for _ in range(7)]


def oracle(arrays,masks):
    def rank(x,f):
        out=np.zeros_like(x)
        out[~f]=pd.DataFrame(x[~f]).rank(pct=True).to_numpy()
        return out
    rr,r224,r7,e143,e155,q,r6=[rank(x,f) for x,f in zip(arrays,masks)]
    rf,f224,f7,f143,f155,fq,f6=masks
    rh=np.where(rf[:,None],r224,np.where(f224[:,None],rr,.65*rr+.35*r224))
    sf=f7&f6
    slot=np.where(f7[:,None],r6,np.where(f6[:,None],r7,.5*r7+.5*r6))
    slot[sf]=.5
    rfail=rf&f224
    pp=np.where(rfail[:,None],slot,np.where(sf[:,None],rh,.5*rh+.5*slot))
    pf=rfail&sf
    pp[pf]=.5
    den=17*(~f143).astype(int)+17*(~f155).astype(int)+12*(~fq).astype(int)
    own=(17*e143+17*e155+12*q)/np.maximum(den,1)[:,None]
    pr,oo=rank(pp,pf),rank(own,den==0)
    out=np.where(pf[:,None],oo,np.where((den==0)[:,None],pr,.85*pr+.15*oo))
    out[pf&(den==0)]=.5
    return out


@pytest.mark.parametrize('mode',['healthy','mixed','r7','r6','both','own','r384'])
def test_ranks_survivors_and_no_extra_slot_rank(candidate,mode):
    ids,arrays=samples()
    masks=[np.zeros(100,bool) for _ in arrays]
    if mode=='mixed':
        for mask,rows in zip(masks,([0,1,3],[0,1,2],[0,4],[0,2],[0,5],[0,6],[0,7])): mask[rows]=True
    for idx in {'r7':[2],'r6':[6],'both':[2,6],'own':[3,4,5],'r384':[0]}.get(mode,[]): masks[idx][:]=True
    failures={key:{u:'test' for u,f in zip(ids,mask) if f} for key,mask in
              zip(['R','R224','S_R7','EXP143v2','EXP155','Q_A5q','S_R6'],masks)}
    actual=candidate.r224.blend(ids,*arrays[:6],failures,r6=arrays[6])
    np.testing.assert_array_equal(actual,oracle(arrays,masks))
    if mode=='healthy': np.testing.assert_array_equal(actual,candidate.r224.reference(*arrays[:6],r6=arrays[6]))


@pytest.mark.parametrize('bad',['nan','inf','negative','above_one','shape','duplicate','r224_4percent'])
def test_rejection(candidate,bad):
    ids,arrays=samples()
    if bad=='duplicate': ids[0]=ids[1]
    elif bad=='shape': arrays[6]=arrays[6][:,:11]
    elif bad=='r224_4percent': arrays[1][:4]=.5
    else: arrays[6][0,0]={'nan':np.nan,'inf':np.inf,'negative':-.1,'above_one':1.1}[bad]
    with pytest.raises(ValueError): candidate.r224.blend(ids,*arrays[:6],r6=arrays[6])


def test_r224_boundary(candidate):
    ids,arrays=samples(); arrays[1][:3]=.5
    candidate.r224.blend(ids,*arrays[:6],r6=arrays[6])


def test_exact_scope():
    nb,meta,files,_,manifest,full=b.artifacts()
    original=b.parent()[1]
    assert len(manifest['unchanged'])==27 and len(manifest['changed'])==4
    assert set(files)==set(original)
    for name in manifest['unchanged']: assert files[name]==original[name]
    def source(raw,name):
        tree=ast.parse(raw)
        return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name))
    for name in ('worker_args','diagnostic_source'):
        assert source(files[b.PORT],name)==source(original[b.PORT],name)
    assert nb['cells'][:-1]==b.parent()[0]['cells'][:-1]
    parent_meta=b.old.read(b.BASE/'kernel/kernel-metadata.json')
    assert {k:v for k,v in meta.items() if k not in ('id','title','dataset_sources')}=={k:v for k,v in parent_meta.items() if k not in ('id','title','dataset_sources')}
    assert meta['dataset_sources']==parent_meta['dataset_sources']+[b.R6_PRIVATE]
    assert meta['id']==b.SLUG and meta['title']==b.SLUG.split('/')[1]
    assert all(name.encode() in full for name in files)


def r6_fixture(candidate,tmp,ids,values):
    work=tmp/'pr7-run'; worker=work/'s_r6/controller/worker0'
    worker.mkdir(parents=True)
    manifest=b.old.read(b.R6_LOCAL/'manifest.json')
    identity=dict(dataset_slug=b.R6_PRIVATE,bundle_id=b.R6_ID,manifest_sha256=b.R6_SHA,
                  file_sha256={'manifest.json':b.R6_SHA,**manifest['sha256']})
    (work/'r6-identity.json').write_text(json.dumps(identity))
    provenance=dict(identity,source_sha256=candidate.r7.SOURCE_HASHES,anchor_weights_sha256=candidate.s6.ANATOMY_SHA256)
    (work/'s_r6/provenance.json').write_text(json.dumps(provenance))
    (worker/'ids.json').write_text(json.dumps(ids))
    (worker/'receipt.json').write_text('{"fallback":{}}')
    (worker/'predictions.csv').write_bytes(candidate.p.csv_bytes(ids,values))
    return worker


@pytest.mark.parametrize('fault',['none','pin','bundle','slug','duplicate','missing'])
def test_authenticated_r6_uid_and_failures(tmp_path,candidate,fault):
    ids,arrays=samples(3)
    (tmp_path/'ids.json').write_text(json.dumps(ids[::-1]))
    worker=r6_fixture(candidate,tmp_path,ids,arrays[6])
    if fault in ('pin','bundle','slug'):
        path=worker.parent.parent/'provenance.json'; value=json.loads(path.read_text())
        value[{'pin':'manifest_sha256','bundle':'bundle_id','slug':'dataset_slug'}[fault]]='bad'
        path.write_text(json.dumps(value))
    if fault=='duplicate': (worker/'ids.json').write_text(json.dumps([ids[0]]*3))
    if fault=='missing': (worker/'predictions.csv').unlink()
    def read(): return candidate.authenticated_pr7(tmp_path,member='S_R6',bundle_id=b.R6_ID,dataset_slug=b.R6_PRIVATE,manifest_pin=b.R6_SHA)
    if fault in ('none','missing'):
        got,_,value,receipt=read(); assert got==ids[::-1]
        if fault=='none':
            np.testing.assert_allclose(value,arrays[6][::-1],atol=3e-8,rtol=0)
            assert not receipt['failures']['S_R6']
        else: assert set(receipt['failures']['S_R6'])==set(ids)
    else:
        with pytest.raises(ValueError): read()


@pytest.mark.parametrize('fail_r7',[False,True])
def test_synchronous_launch_and_survivor(tmp_path,monkeypatch,candidate,fail_r7):
    out=tmp_path/'run/pr7'; work=out/'pr7-run'; work.mkdir(parents=True)
    (out/'ids.json').write_text('["a"]')
    (work/'r7-identity.json').write_text('{"manifest_sha256":"test"}')
    (work/'r6-identity.json').write_text(json.dumps(dict(dataset_slug=b.R6_PRIVATE)))
    (work/'code-paths.json').write_text(json.dumps(dict(code=str(work/'code'),anatomy=str(work/'anatomy.pt'))))
    monkeypatch.setattr(candidate,'runtime_pins',lambda *a:None)
    monkeypatch.setattr(candidate,'competition_root',lambda:tmp_path/'competition')
    monkeypatch.setattr(candidate.v5q,'_mount',lambda *a:tmp_path/'mount')
    monkeypatch.setattr(candidate.base,'r6_mount',lambda:(b.R6_LOCAL,b.R6_PRIVATE))
    monkeypatch.setattr(candidate.p,'check_pin',lambda *a:None)
    monkeypatch.setattr(candidate.s6,'install_wheels',lambda *a:None)
    monkeypatch.setattr(candidate.r224,'record',lambda *a:None)
    events=[]
    def process(args,*,label,env): events.append(label)
    def infer(*a,**kw):
        member='R6' if kw.get('bundle_id')==b.R6_ID else 'R7'
        events.append(member)
        assert kw['workers']==2 and kw['device']=='cuda'
        if member=='R7' and fail_r7: raise RuntimeError('injected R7 failure')
    monkeypatch.setattr(candidate.v5q,'run_process_group',process)
    monkeypatch.setattr(candidate.r7,'infer_studies',infer)
    candidate.run_public(tmp_path/'runtime',out)
    assert events==['C85R7 R','C85R7 R224','R7','R6']
    assert json.loads((out/'S_R7-exit.json').read_text())['exit_code']==int(fail_r7)
    assert json.loads((out/'S_R6-exit.json').read_text())['exit_code']==0


def test_collect_actual_visible_r_own_with_explicit_r6_fixture(tmp_path,candidate):
    from tools import exp169_release_c85r7codec as historical
    staged=historical.stage_replay_input(tmp_path/'pr7')
    ids,r,ours,_=historical.authenticated_pr7(staged)
    csvdir=staged/'csv'; csvdir.mkdir(exist_ok=True)
    for name in (candidate.r224.RAW,candidate.r224.SUB):
        (csvdir/name).write_bytes(candidate.p.csv_bytes(ids,r))
    candidate.r224.record(staged/'pr7-run',csvdir)
    (staged/'R224-exit.json').write_text('{"exit_code":0}')
    r6_fixture(candidate,staged,ids[::-1],(1-ours)[::-1])
    receipt,payload=candidate.collect(historical.c85.A5Q,staged,tmp_path/'out')
    assert receipt['status']=='COMPLETE_C85R7'
    assert set(receipt['executed_members'])==set(candidate.MEMBERS+['R224','S_R6'])
    assert receipt['fallbacks']['s_r6_prediction']==0
    # Historical Own output records failures; the healthy-only diagnostic is
    # correctly false. Compare the actual survivor policy independently.
    oid,e143,e155,q,a=candidate.c85.authenticated_a172(historical.c85.A5Q)
    finalids,actual=candidate.v5q._read_csv(tmp_path/'out/submission.csv')
    arrays=[r,r,ours,e143,e155,q,1-ours]
    assert oid==ids==finalids
    masks=[np.array([uid in a['failures'].get(name,{}) for uid in ids]) for name in
           ['R','R224','S_R7','EXP143v2','EXP155','Q_A5q','S_R6']]
    # CSV serialization uses seven decimal places (maximum rounding 5e-8).
    np.testing.assert_allclose(actual,oracle(arrays,masks),atol=5.1e-8,rtol=0)
    assert (tmp_path/'out/submission.csv').read_bytes()==payload


def test_r6_manifest_override_rejects_before_read(candidate):
    for kw in (dict(expected_manifest_sha256='wrong'),dict(dataset_slug='other/other')):
        kwargs=dict(bundle_id=b.R6_ID,dataset_slug=b.R6_PRIVATE); kwargs.update(kw)
        with pytest.raises(ValueError): candidate.r7.verify_bundle(Path('missing'),**kwargs)
