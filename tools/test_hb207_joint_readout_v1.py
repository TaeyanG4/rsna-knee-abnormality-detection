"""Synthetic-only tests for H207; never read real labels/probabilities/weights."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from tools import hb207_joint_readout_v1 as b


@pytest.fixture(scope='module')
def op():
    return b.operators(b.authenticated_registration())


def raw_fixture(n=8):
    rng = np.random.default_rng(207)
    return {f: {k: rng.integers(0, 32, (n, 12)).astype(np.float32) / 32
                for k in sorted(b.KEYS)} for f in range(5)}


def context(n=40):
    ids = ['synthetic-' + str(i) for i in range(n)]
    labels = ((np.arange(n)[:, None] + np.arange(12)[None, :]) % 3 == 0).astype(float)
    refs = {r: (labels.copy(), np.ones((n, 12), bool), minimum)
            for r, minimum in b.MIN_CLASS.items()}
    refs['consensus'][1][::7, 1::3] = False
    components = [dict(component_id='component-' + str(i), members=ids[i:i+2])
                  for i in range(0, n, 2)]
    return b.ml.Context(ids, refs, components)


def independent_rank(p):
    return (rankdata(np.asarray(p, np.float64), axis=0, method='average') - 1) / max(1, len(p)-1)


def test_fixed_arms_exact_order_and_full_roster(op):
    raw = raw_fixture()
    actual = b.four_arms(raw, op)
    families = {k: independent_rank(np.concatenate([
        (raw[f][names[0]] + raw[f][names[1]]) / 2 if len(names) == 2 else raw[f][names[0]]
        for f in range(5)])) for k, names in
        {'R': ('R42','R43'), 'D': ('DINO',), 'C': ('CNX',), 'C4': ('C4',), 'S': ('S42','S43')}.items()}
    hybrid = families['D'].copy(); hybrid[:, :8] = families['S'][:, :8]
    for name, d, c in [('Q0', families['D'], families['C']), ('Q4', families['D'], families['C4']),
                       ('Q2', hybrid, families['C']), ('Q24', hybrid, families['C4'])]:
        expected = independent_rank((independent_rank(families['R']) + independent_rank(d)
                                     + independent_rank(c)) / 3)
        np.testing.assert_array_equal(actual[name], expected)
    assert tuple(actual) == b.ARMS
    np.testing.assert_array_equal(actual['Q0'][:, 8:], actual['Q2'][:, 8:])
    np.testing.assert_array_equal(actual['Q4'][:, 8:], actual['Q24'][:, 8:])
    # Ranking fold0 alone must not silently replace fixed pooled ranks.
    assert not np.array_equal(actual['Q0'][:8], independent_rank(actual['Q0'][:8]))


def test_actual_serving_tail_and_parent_identity(op):
    reg = b.authenticated_registration()
    data = (b.WORK / 'candidate-runtime/exp169_partner_bundle_release.py').read_text()
    env = b.inf2.extracted_functions(data, ('rank','aggregate'), dict(need=b.need))
    rng = np.random.default_rng(910)
    scores = {k: rng.random((10 if k in ('l2','small') else 5, 30, 12), dtype=np.float32)
              for k in ('l2','dino','convnext','small')}
    full, families = env['aggregate'](scores)
    np.testing.assert_array_equal(full, op['bundle'](families))
    # Exact serving Small pair-fold mean, not flat ten mean, is preserved in the source.
    small = ((scores['small'][:5] + scores['small'][5:]) / 2).mean(0)
    assert np.any(small != scores['small'].mean(0))
    np.testing.assert_array_equal(families['dino'][:, :8], op['rank'](small)[:, :8])
    parent = b.inf2.parent_files()['exp169_partner_bundle_release.py'].decode()
    parent_rank = b.inf2.extracted_functions(parent, ('rank',), dict(need=b.need))['rank']
    values = np.repeat(np.array([.1,.5,.5,.9])[:, None], 12, axis=1)
    np.testing.assert_array_equal(op['rank'](values), parent_rank(values))
    np.testing.assert_array_equal(op['rank'](values)[:,0], [0,.5,.5,1])
    np.testing.assert_array_equal(op['rank'](values[:1]), np.zeros((1,12)))
    assert reg['rank_conventions']['historical_oof'] == 'average_rank/N'


def test_noop_and_probability_first_precision(op):
    raw = raw_fixture()
    for row in raw.values():
        row['C4'] = row['CNX'].copy()
        row['S42'] = row['S43'] = row['DINO'].copy()
    result = b.four_arms(raw, op)
    assert all(np.array_equal(a, result['Q0']) for a in result.values())
    raw = raw_fixture()
    # A FP64 pair would distinguish the final two rows; FP32 rounds them to one tie.
    a = np.array([.5, np.nextafter(np.float32(.5), np.float32(1))], np.float32)
    raw[0]['R42'][:2] = a[:,None]
    raw[0]['R43'][:2] = np.float32(.5)
    captured = []
    def rank(p):
        captured.append(p.copy())
        return independent_rank(p)
    b.four_arms(raw, {**op, 'rank': rank})
    assert captured[0].dtype == np.float32 and np.array_equal(captured[0][0], captured[0][1])
    fp64 = (raw[0]['R42'].astype(float) + raw[0]['R43'].astype(float)) / 2
    assert fp64[0,0] != fp64[1,0]
    mean = captured[0]
    bad = (independent_rank(np.concatenate([raw[f]['R42'] for f in range(5)])) +
           independent_rank(np.concatenate([raw[f]['R43'] for f in range(5)]))) / 2
    assert not np.array_equal(independent_rank(mean), independent_rank(bad))


@pytest.mark.parametrize('defect', ['missing','extra','nan','inf','negative','above_one','dtype','shape','fold'])
def test_invalid_raw_rejected(defect, op):
    raw = raw_fixture()
    if defect == 'missing': del raw[1]['S42']
    if defect == 'extra': raw[1]['S44'] = raw[1]['S42']
    if defect == 'nan': raw[1]['C4'][0,0] = np.nan
    if defect == 'inf': raw[1]['C4'][0,0] = np.inf
    if defect == 'negative': raw[1]['C4'][0,0] = -.1
    if defect == 'above_one': raw[1]['C4'][0,0] = 1.1
    if defect == 'dtype': raw[1]['S42'] = raw[1]['S42'].astype(float)
    if defect == 'shape': raw[1]['S42'] = raw[1]['S42'][:3]
    if defect == 'fold': del raw[4]
    with pytest.raises(ValueError): b.four_arms(raw, op)


def test_overlap_and_roster_identity():
    raw = raw_fixture()
    rosters = [{'val': ['synthetic-'+str(f)+'-'+str(i) for i in range(8)], 'train': []} for f in range(5)]
    cc = {f: {k:a for k,a in row.items() if k not in ('S42','S43')} for f,row in raw.items()}
    ss = {f: {k:a for k,a in row.items() if k != 'C4'} for f,row in raw.items()}
    merged = b.merge_inputs(cc, rosters, ss, rosters)
    assert all(np.array_equal(merged[f][k], raw[f][k]) for f in range(5) for k in b.KEYS)
    bad = copy.deepcopy(ss); bad[0]['DINO'][0,0] += .001
    with pytest.raises(ValueError, match='overlap'): b.merge_inputs(cc, rosters, bad, rosters)
    bad = copy.deepcopy(rosters); bad[0]['val'].reverse()
    with pytest.raises(ValueError, match='roster'): b.merge_inputs(cc, rosters, ss, bad)
    with pytest.raises(ValueError, match='row count'):
        b.merge_inputs(cc, [{'val':['one']}]*5, ss, [{'val':['one']}]*5)


def test_full_synthetic_roster_hash_reorder_duplicate_and_leakage():
    ids = ['synthetic-'+str(i) for i in range(4343)]
    sizes = [888,864,864,864,863]; rosters=[]; offset=0
    for n in sizes:
        val=ids[offset:offset+n]; offset+=n
        rosters.append(dict(val=val, train=[u for u in ids if u not in set(val)]))
    expected=dict(rows=4343, ordered_uid_sha256=b.aligned.serving.uid_hash(ids),
        folds=[dict(fold=f,rows=len(r['val']),ordered_uid_sha256=b.aligned.serving.uid_hash(r['val']))
               for f,r in enumerate(rosters)])
    assert b.check_rosters(rosters, expected) == ids
    for defect in ('reorder','duplicate','leakage','hash','missing'):
        rr,ee=copy.deepcopy(rosters),copy.deepcopy(expected)
        if defect=='reorder':rr[0]['val'].reverse()
        if defect=='duplicate':rr[0]['val'][1]=rr[0]['val'][0]
        if defect=='leakage':rr[0]['train'][0]=rr[0]['val'][0]
        if defect=='hash':ee['folds'][2]['ordered_uid_sha256']='bad'
        if defect=='missing':rr.pop()
        with pytest.raises(ValueError):b.check_rosters(rr,ee)


def test_auc_paired_component_expansion_ci_and_slice(op):
    ctx = context()
    arms = b.four_arms(raw_fixture(), op)
    out = b.readout(ctx, arms, op)
    for key, values in arms.items():
        for i,r in enumerate(b.RULERS):
            y,mask,minimum = ctx.refs[r]
            for j,t in enumerate(b.ml.TARGETS):
                assert out['arms'][key][r]['targets'][t]['value'] == pytest.approx(
                    roc_auc_score(y[mask[:,j],j], values[mask[:,j],j]), abs=1e-14)
    for contrast in b.CONTRASTS:
        ref = contrast.split('-')[1]
        d = ctx.score(arms['Q24']) - ctx.score(arms[ref])
        assert out['contrasts'][contrast]['jev']['macro'] == b.interval(d[:,0].mean(1))
        assert out['contrasts'][contrast]['jev']['weak4'] == b.interval(d[:,0][:,b.ml.WEAK].mean(1))
    # Check a non-unit connected-component resample against physical expansion.
    draw = 13
    index = np.repeat(np.arange(40), ctx.w[draw].astype(int))
    assert np.any(ctx.w[draw] != 1)
    for key in ('Q0','Q24'):
        y,mask,_=ctx.refs['jev']; idx=index[mask[index,0]]
        assert ctx.score(arms[key])[draw,0,0] == pytest.approx(roc_auc_score(y[idx,0],arms[key][idx,0]),abs=1e-14)
    assert out['support']['consensus']['mask_sha256'] == hashlib.sha256(ctx.refs['consensus'][1].tobytes()).hexdigest()
    # Same context/draws deterministically replay; fold readout receives slices, not re-ranked values.
    assert b.readout(context(), arms, op) == out
    fc=context(8)
    sl={k:a[:8] for k,a in arms.items()}
    assert b.readout(fc,sl,op)['rows']==8
    assert all('PASS' not in json.dumps(v) for v in out['contrasts'].values())


def test_missing_classes_mask_nan_and_undefined_draws(op):
    arms=b.four_arms(raw_fixture(),op);ctx=context()
    ctx.refs['jev'][0][:,0]=0
    result=b.readout(ctx,arms,op)
    assert result['registered_point_rule'].startswith('HOLD')
    assert result['arms']['Q0']['jev']['targets']['ACL']['value'] is None
    assert result['arms']['Q0']['jev']['targets']['ACL']['ci95'] is None
    ctx=context();ctx.refs['jev'][0][0,0]=np.nan
    with pytest.raises(ValueError,match='masked labels'):b.readout(ctx,arms,op)
    ctx.refs['jev'][1][0,0]=False
    b.readout(ctx,arms,op)  # An unavailable label outside the fixed mask is permissible.
    ctx=context();ctx.w[2,0]=.5
    with pytest.raises(ValueError,match='weights'):b.readout(ctx,arms,op)
    value=b.interval(np.array([.1, np.nan,.2,.3]))
    assert value['valid_draws']==2 and value['ci95']==pytest.approx([.2025,.2975])


def test_registered_dominance_policy_no_new_ci_floor(op):
    points={k:np.zeros((4,2)) for k in b.ARMS}
    contrasts={k:np.zeros((4,12)) for k in b.CONTRASTS}
    rule=op['decision_readout']
    assert rule(points,contrasts)=='REPORT_FOR_B_HOLISTIC_JUDGMENT'
    points['Q4'][0,0]=.00000001
    assert rule(points,contrasts)=='NO_DEMONSTRATED_INCREMENTAL_ROLE'
    points['Q24'][1,1]=.00000001
    assert rule(points,contrasts)=='REPORT_FOR_B_HOLISTIC_JUDGMENT'
    del contrasts['Q24-Q2']
    assert rule(points,contrasts).startswith('HOLD')
    contrasts['Q24-Q2']=np.full((4,12),np.nan)
    assert rule(points,contrasts).startswith('HOLD')


def test_static_prepare_never_loads_numeric_payloads(monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        p=path.as_posix().lower()
        assert not any(part in p for part in ('/storage/labels/','/storage/cache/','/.local/relabel/'))
        assert path.suffix.lower() not in ('.pt','.pth','.ckpt','.npz','.npy','.csv','.dcm')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    def forbidden(*args,**kwargs): raise AssertionError('numeric loader called before review')
    monkeypatch.setattr(b.c4,'authenticate_oof',forbidden)
    monkeypatch.setattr(b.inf2,'load_inputs',forbidden)
    monkeypatch.setattr(b.ml,'make_context',forbidden)
    captured={}
    monkeypatch.setattr(b.inf2,'write',lambda path, value:captured.update(value))
    actual_pin=b.pin
    monkeypatch.setattr(b,'pin',lambda path: dict(bytes=0,sha256='synthetic-static-fixture')
        if Path(path).name in ('TEST-RECEIPT.json','CONTRACT.json') else actual_pin(path))
    assert b.prepare()['status']=='SOURCE_READY_REVIEW_REQUIRED'
    assert captured['arms']==list(b.ARMS) and captured['exact_F'].startswith('NOT_IMPLEMENTED')


def fake_review(tmp_path, monkeypatch):
    monkeypatch.setattr(b,'OUT',tmp_path)
    (tmp_path/'CONTRACT.json').write_text('{}')
    (tmp_path/'TEST-RECEIPT.json').write_text('{}')
    evidence=tmp_path/'CHECKLIST.md';evidence.write_text('Synthetic fixture only. No real review.')
    payload=dict(schema='h207-source-review/v1',verdict='PASS',reviewer='synthetic-independent',
        builder=b.BUILDER,scope='fixed4_exposed_heldout_oof_only',
        contract_sha256=b.pin(tmp_path/'CONTRACT.json')['sha256'],joint_registration_sha256=b.REG_SHA,
        test_receipt_sha256=b.pin(tmp_path/'TEST-RECEIPT.json')['sha256'],
        evidence_path=b.rel(evidence),evidence_sha256=b.pin(evidence)['sha256'])
    path=tmp_path/'SYNTHETIC-REVIEW.json'
    path.write_text(json.dumps(payload))
    return path,payload


@pytest.mark.parametrize('defect',['none','missing','self','fail','scope','contract','tests','evidence','hash'])
def test_source_review_gate_before_any_payload(tmp_path,monkeypatch,defect):
    path,payload=fake_review(tmp_path,monkeypatch)
    if defect=='self':payload['reviewer']=b.BUILDER
    if defect=='fail':payload['verdict']='FAIL'
    if defect=='scope':payload['scope']='GPU'
    if defect=='contract':payload['contract_sha256']='wrong'
    if defect=='tests':payload['test_receipt_sha256']='wrong'
    if defect=='evidence':payload['evidence_sha256']='wrong'
    path.write_text(json.dumps(payload));digest=b.pin(path)['sha256']
    if defect=='hash':digest='wrong'
    if defect=='none':
        assert b.review_gate(path,digest)['reviewer']=='synthetic-independent'
        return
    def forbidden(*a,**kw):raise AssertionError('payload load before gate')
    monkeypatch.setattr(b,'validate_contract',forbidden)
    monkeypatch.setattr(b.c4,'authenticate_oof',forbidden)
    monkeypatch.setattr(b.inf2,'load_inputs',forbidden)
    with pytest.raises(ValueError):b.run(None if defect=='missing' else path,digest)


def test_pin_drift_checkpoint_stat_and_immutable_publication(tmp_path,monkeypatch):
    source=tmp_path/'input.json';source.write_text('{}');p=b.rel(source)
    b.verify_closure({p:b.pin(source)})
    old=b.pin(source);source.write_text('{"changed":1}')
    with pytest.raises(ValueError,match='drift'):b.verify_closure({p:old})
    checkpoint=tmp_path/'synthetic.pt';checkpoint.write_bytes(b'no checkpoint content is read')
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path!=checkpoint
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    b.verify_closure({b.rel(checkpoint):dict(bytes=checkpoint.stat().st_size,sha256='inherited')})
    dest=tmp_path/'published.json';b.inf2.write(dest,{'aggregate':1})
    with pytest.raises(ValueError,match='immutable'):b.inf2.write(dest,{'aggregate':2})
    assert json.loads(dest.read_text())=={'aggregate':1}


def test_windows_bounded_publication_retry(monkeypatch):
    calls=[]
    monkeypatch.setattr(b.inf2.time,'sleep',lambda _:None)
    def flaky():
        calls.append(1)
        if len(calls)<3:raise PermissionError('synthetic sharing violation')
        return 'done'
    assert b.inf2.retry(flaky)=='done' and len(calls)==3
    calls.clear()
    def broken():
        calls.append(1);raise PermissionError('persistent synthetic lock')
    with pytest.raises(PermissionError):b.inf2.retry(broken)
    assert len(calls)==40


@pytest.mark.parametrize('post_drift',[False,True])
def test_synthetic_full_run_and_historical_drift(tmp_path,monkeypatch,op,post_drift):
    raw=raw_fixture();ctx=context()
    rr=[dict(train=[],val=ctx.ids[f*8:(f+1)*8]) for f in range(5)]
    cc={f:{k:a for k,a in row.items() if k not in ('S42','S43')} for f,row in raw.items()}
    ss={f:{k:a for k,a in row.items() if k!='C4'} for f,row in raw.items()}
    cq0,cq4=b.c4.oof_values(cc)
    hq0,hq2=b.inf2.q_pair(ss,b.ml.rank);eq0,eq2=b.inf2.q_pair(ss,op['rank'])
    histories={'c4':dict(primary=b.ml.clean(b.c4.report(ctx,cq4,cq0))),
               'inf2':dict(primary=b.inf2.describe(ctx,hq2,hq0),
                           exact_parent_q_arithmetic=b.inf2.describe(ctx,eq2,eq0))}
    arms=b.four_arms(raw,op)
    summary=b.historical_checks(ctx,cc,ss,arms,histories,op)
    assert summary['old_gates_used_for_joint'] is False
    bad=copy.deepcopy(histories);bad['c4']['primary']['primary_macro']['value']+=.1
    with pytest.raises(ValueError,match='old C4'):b.historical_checks(ctx,cc,ss,arms,bad,op)
    path,payload=fake_review(tmp_path,monkeypatch);review_sha=b.pin(path)['sha256']
    cr=dict(candidate='R6W958-C4-V1',bindings={})
    sr=dict(candidate='R6W958-INF2-V1',bindings={},members=[])
    reg=dict(input_registration_refs={'synthetic-c4':{},'synthetic-inf2':{}},
             ruler_bindings={},roster={},bootstrap=dict(draws=300,seed=20261003))
    contract=dict(source_pins={})
    checks=[]
    def validate():
        checks.append(1)
        if post_drift and len(checks)>1:raise ValueError('synthetic post-execution source drift')
        return reg,contract
    monkeypatch.setattr(b,'validate_contract',validate)
    monkeypatch.setattr(b,'operators',lambda _:op)
    monkeypatch.setattr(b,'check_rosters',lambda *_:ctx.ids)
    monkeypatch.setattr(b.c4,'authenticate_oof',lambda:(cc,rr,dict(bindings={},ruler_bindings={})))
    monkeypatch.setattr(b.inf2,'load_inputs',lambda _: (ss,rr,dict(ruler_bindings={}),{},[],{}))
    original_read=b.read
    def read(path):
        if str(path).endswith('synthetic-c4'):return cr
        if str(path).endswith('synthetic-inf2'):return sr
        for k,p in b.HISTORIES.items():
            if str(path).replace('\\','/').endswith(p):return histories[k]
        return original_read(path)
    monkeypatch.setattr(b,'read',read)
    def make_context(ids,_):
        index=[ctx.ids.index(u) for u in ids]
        refs={r:(y[index],mask[index],minimum) for r,(y,mask,minimum) in ctx.refs.items()}
        comps=[dict(component_id='component-'+str(i),members=ctx.ids[i:i+2]) for i in range(0,40,2)]
        out=b.ml.Context(ids,refs,comps);out.bindings={};return out
    monkeypatch.setattr(b.ml,'make_context',make_context)
    if post_drift:
        with pytest.raises(ValueError,match='post-execution source drift'):b.run(path,review_sha)
        assert not (tmp_path/'RESULT.json').exists()
        return
    result=b.run(path,review_sha)
    assert result['status'].startswith('COMPLETE_EXPOSED_Q_PROXY')
    stored=json.loads((tmp_path/'RESULT.json').read_text())
    assert len(stored['folds'])==5 and stored['qualified_for_submission'] is False
    assert stored['clean_oof'] is False and 'pending' in stored['lenses']['L6']
    assert stored['pooled']['registered_point_rule'] in ('NO_DEMONSTRATED_INCREMENTAL_ROLE','REPORT_FOR_B_HOLISTIC_JUDGMENT')
    encoded=json.dumps(stored)
    assert not any(uid in encoded for uid in ctx.ids)
    with pytest.raises(ValueError,match='immutable'):b.run(path,review_sha)
