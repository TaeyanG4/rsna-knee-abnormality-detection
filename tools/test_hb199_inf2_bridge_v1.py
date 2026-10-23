"""Focused synthetic tests for the single H-B-199 INF2 CPU bridge."""
import copy
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata
from tools import hb199_inf2_bridge_v1 as b


def raw_fixture(n=11):
    rng = np.random.default_rng(199)
    return {f: {k: rng.integers(1, 20, (n, 12)).astype(np.float32)/20 for k in b.RAW_KEYS}
            for f in range(5)}


def test_target_mask_and_original_arithmetic():
    raw = raw_fixture()
    p, c = b.q_pair(raw)
    ranks = {k: b.ml.rank(np.concatenate([
        np.stack([raw[f][x] for x in keys]).mean(0, dtype=np.float32) for f in range(5)]))
        for k, keys in dict(R=('R42', 'R43'), D=('DINO',), C=('CNX',), S=('S42', 'S43')).items()}
    hybrid = np.where(np.arange(12)[None, :] >= 8, ranks['D'], ranks['S'])
    np.testing.assert_array_equal(p, b.ml.rank((ranks['R']+ranks['D']+ranks['C'])/3))
    np.testing.assert_array_equal(c, b.ml.rank((ranks['R']+hybrid+ranks['C'])/3))
    np.testing.assert_array_equal(p[:, b.KEEP], c[:, b.KEEP])
    assert np.any(p[:, :8] != c[:, :8])


@pytest.mark.parametrize('rank', [b.ml.rank, b.serving.rank])
def test_noop_small_equal_dino(rank):
    raw = raw_fixture()
    for fold in raw.values(): fold['S42'] = fold['S43'] = fold['DINO'].copy()
    p, c = b.q_pair(raw, rank)
    np.testing.assert_array_equal(p, c)


def test_parent_q_rank_ties_permutation_and_full_roster():
    values = np.tile(np.array([.1,.5,.5,.9])[:,None], (1,12))
    rank = b.serving.rank(values)
    np.testing.assert_array_equal(rank[:,0], [0,.5,.5,1])
    order = [2,0,3,1]
    np.testing.assert_array_equal(b.serving.rank(values[order]), rank[order])
    assert not np.array_equal(rank[:2], b.serving.rank(values[:2]))
    np.testing.assert_array_equal(b.serving.rank(values[:1]), np.zeros((1,12)))


@pytest.mark.parametrize('kind', ['missing','extra','nan','range','dtype','shape','fold'])
def test_reject_invalid_members(kind):
    raw = raw_fixture()
    if kind == 'missing': del raw[1]['S42']
    if kind == 'extra': raw[1]['S44'] = raw[1]['S42']
    if kind == 'nan': raw[1]['S42'][0,0] = np.nan
    if kind == 'range': raw[1]['S42'][0,0] = -1
    if kind == 'dtype': raw[1]['S42'] = raw[1]['S42'].astype(np.float64)
    if kind == 'shape': raw[1]['S42'] = raw[1]['S42'][:2]
    if kind == 'fold': del raw[4]
    with pytest.raises(ValueError): b.q_pair(raw)


def test_small_seed_fold_source_authentication():
    auth = b.read(b.ROOT/b.aligned.AUTH)
    members = [e for e in auth['members'] if e['family']=='S']
    b.check_roster(members,True)
    for field, value in [('seed',44),('fold',5),('path','wrong/model_best.pt'),('arch','convnext_tiny.fb_in22k_ft_in1k')]:
        bad=copy.deepcopy(members);bad[0][field]=value
        with pytest.raises(ValueError):b.check_roster(bad,True)
    with pytest.raises(ValueError): b.check_roster(members[:-1]+[members[0]],True)


def test_parent_identity_roster():
    entries = b.read(b.ROOT/b.BASE_IDENTITIES)['q_checkpoints']
    b.check_roster(entries)
    with pytest.raises(ValueError): b.check_roster(entries[:-1]+[entries[0]])
    bad=copy.deepcopy(entries);bad[0]['family']='convnext'
    with pytest.raises(ValueError):b.check_roster(bad)


def test_pin_alteration_missing_and_outside(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'ROOT',tmp_path)
    path=tmp_path/'input.txt';path.write_text('original')
    bindings={};b.bind(bindings,path);b.verify_pins(bindings)
    path.write_text('altered')
    with pytest.raises(ValueError,match='pin'):b.verify_pins(bindings)
    with pytest.raises(FileNotFoundError):b.bind({},tmp_path/'missing')
    with pytest.raises(ValueError,match='outside'):b.rel(tmp_path.parent/'outside')


def test_csv_alignment_duplicate_target_order_nan(tmp_path):
    header=['StudyInstanceUID',*b.TARGETS]
    def put(ids,columns=header,values=None):
        path=tmp_path/'predictions.csv'
        rows=[','.join(columns)]
        rows += [','.join([uid,*(['0.5']*12 if values is None else values)]) for uid in ids]
        path.write_text('\n'.join(rows)+'\n')
        return path
    p=put(['a','b']);assert b.aligned.load_csv(p,['a','b']).shape==(2,12)
    for ids in (['b','a'],['a','a'],['a']):
        with pytest.raises(ValueError):b.aligned.load_csv(put(ids),['a','b'])
    with pytest.raises(ValueError):b.aligned.load_csv(put(['a','b'],header[:-1]+['WRONG']),['a','b'])
    with pytest.raises(ValueError):b.aligned.load_csv(put(['a','b'],values=['nan']*12),['a','b'])


def test_original_pooled_gate_boundaries_and_no_fold_gate():
    d=np.full((4,12),.0005)
    assert b.selection(d)
    d[3]=-.003;assert b.selection(d)
    d[3,0]=-.003001;assert not b.selection(d)
    d=np.full((4,12),.000499);assert not b.selection(d)
    d[:3]=.0005;assert b.selection(d)
    d[0,0]=np.nan;assert not b.selection(d)
    with pytest.raises(ValueError):b.selection(np.zeros((5,12)))
    # Adverse fold diagnostic does not enter the registered pooled gate.
    adverse=np.full((4,12),.001);adverse[0,0]=-.006848
    assert not b.selection(adverse)
    assert b.selection(np.full((4,12),.001))


def outer_reference(arrays, failures, ids):
    # Independent implementation of the exact documented survivor policy.
    r,r224,s,e143,e155,q=arrays
    names=['R','R224','S_R7','EXP143v2','EXP155','Q_A5q']
    masks=[np.array([uid in failures.get(name,{}) for uid in ids]) for name in names]
    masks[0] |= (r==.5).all(1);masks[1] |= (r224==.5).all(1)
    def rank(x,failed):
        a=np.zeros_like(x,dtype=float)
        a[~failed]=rankdata(x[~failed],axis=0,method='average')/sum(~failed) if (~failed).any() else 0
        return a
    rr,rr224,ss,a,bq,c=[rank(x,f) for x,f in zip(arrays,masks)]
    rf,f224,sf,f143,f155,fq=masks
    mix=.65*rr+.35*rr224
    for i in range(len(ids)):
        if rf[i] and f224[i]:mix[i]=.5
        elif rf[i]:mix[i]=rr224[i]
        elif f224[i]:mix[i]=rr[i]
    w=np.array([.5,.5,.5,.5,.5,.2,.6,.5,.8,.35,.5,.5])
    p=w*mix+(1-w)*ss
    for i in range(len(ids)):
        if rf[i] and f224[i] and sf[i]:p[i]=.5
        elif rf[i] and f224[i]:p[i]=ss[i]
        elif sf[i]:p[i]=mix[i]
    denom=17*(~f143).astype(int)+17*(~f155).astype(int)+12*(~fq).astype(int)
    own=(17*a+17*bq+12*c)/np.maximum(denom,1)[:,None]
    pf=rf&f224&sf;of=denom==0;pr=rank(p,pf);oo=rank(own,of)
    out=.85*pr+.15*oo
    for i in range(len(ids)):
        if pf[i] and of[i]:out[i]=.5
        elif pf[i]:out[i]=oo[i]
        elif of[i]:out[i]=pr[i]
    return out


@pytest.mark.parametrize('case',['healthy','partial','both_public','own_missing','all_missing','sentinel'])
def test_pinned_parent_outer_survivor_parity(case):
    functions=b.parent_math();rng=np.random.default_rng(20261018)
    ids=[str(i) for i in range(100)]
    arrays=[rng.integers(1,18,(100,12))/20 for _ in range(6)]
    failures={}
    if case=='partial':failures={'R':{'0':1},'R224':{'1':1},'S_R7':{'2':1},'EXP143v2':{'3':1},'Q_A5q':{'4':1}}
    if case in ('both_public','all_missing'):failures={k:{'0':1} for k in ['R','R224','S_R7']}
    if case in ('own_missing','all_missing'):failures.update({k:{'0':1} for k in ['EXP143v2','EXP155','Q_A5q']})
    if case=='sentinel':arrays[0][0]=.5;arrays[1][1]=.5
    actual=functions['blend'](ids,*arrays,failures)
    expected=outer_reference(arrays,failures,ids)
    np.testing.assert_array_equal(actual,expected)
    if case=='healthy':np.testing.assert_array_equal(actual,functions['reference'](*arrays))
    assert actual.shape==(100,12) and np.isfinite(actual).all()
    # H189 serializes only the final float32 values. Keep that boundary visible.
    np.testing.assert_array_equal(actual.astype(np.float32),expected.astype(np.float32))
    payload=functions['csv_bytes'](ids,actual)
    assert payload==functions['csv_bytes'](ids,expected)
    rows=list(csv.reader(io.StringIO(payload.decode())))
    assert rows[0]==['StudyInstanceUID',*b.TARGETS] and [r[0] for r in rows[1:]]==ids
    np.testing.assert_array_equal(np.asarray([r[1:] for r in rows[1:]],np.float32),actual.astype(np.float32))
    order=rng.permutation(100)
    np.testing.assert_array_equal(functions['blend']([ids[i] for i in order],*[a[order] for a in arrays],failures),actual[order])


def test_pinned_parent_failure_gate_and_bad_inputs():
    functions=b.parent_math();ids=[str(i) for i in range(100)];arrays=[np.full((100,12),.4) for _ in range(6)]
    functions['blend'](ids,*arrays,{'R224':{str(i):1 for i in range(3)}})
    with pytest.raises(ValueError,match='3%'):functions['blend'](ids,*arrays,{'R224':{str(i):1 for i in range(4)}})
    with pytest.raises(ValueError,match='duplicate'):functions['blend'](['0']*100,*arrays)
    arrays[0][0,0]=np.nan
    with pytest.raises(ValueError,match='probabilities'):functions['blend'](ids,*arrays)


def test_atomic_create_only_and_windows_retry(tmp_path,monkeypatch):
    target=tmp_path/'artifact.json';original=b.os.replace;calls=[]
    def intermittent(source,dest):
        calls.append(1)
        if len(calls)<3:raise PermissionError('Windows sharing violation')
        return original(source,dest)
    monkeypatch.setattr(b.os,'replace',intermittent)
    monkeypatch.setattr(b.time,'sleep',lambda _:None)
    b.write(target,{'x':1});assert json.loads(target.read_text())=={'x':1};assert len(calls)==3
    with pytest.raises(ValueError,match='immutable'):b.write(target,{'x':2})
    assert json.loads(target.read_text())=={'x':1}


def test_registration_seal_detects_edit(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'OUT',tmp_path)
    b.write(tmp_path/'registration.json',{'version':'bogus'})
    (tmp_path/'registration.sha256').write_text('0'*64)
    with pytest.raises(ValueError,match='seal'):b.validate()


def test_inventory_payload_flag_is_not_digest(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'ROOT',tmp_path)
    p=tmp_path/'metadata.json';p.write_text('{}')
    expected={**b.pin(p),'payload':True}
    b.bind({},p,expected)
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='pin'):b.bind({},p,expected)
