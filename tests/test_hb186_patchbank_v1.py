"""CPU-only contract tests for the one H-B-186 tool; no label-row output."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
from pathlib import Path
import copy
import json
import numpy as np
import pytest
from tools import hb186_patchbank_v1 as m


def fixture():
    rng = np.random.default_rng(123)
    bank = [f'b{i}' for i in range(8)]
    dev = [f'd{i}' for i in range(32)]
    confirm = [f'c{i}' for i in range(24)]
    ids = bank + dev + confirm
    labels = {u: (0. if u[0] == 'b' else float(int(u[1:]) % 2)) for u in ids}
    groups = {u: 'patient-' + u for u in ids}
    records = {}
    for u in ids:
        idx = int(u[1:])
        meta, x = [], []
        for plane, view in [(0, 1), (1, 3)]:
            for slice_ in range(6):
                for cell in range(9):
                    meta.append([plane, 1, 1, view, 1, cell, plane, slice_, idx % 2, 1])
                    # Anatomically structured nuisance; disease is a separate direction.
                    vector = rng.normal(0, .02, 32)
                    vector[cell] += .5
                    vector[-1] += labels[u] * .3
                    x.append(vector)
        records[u] = dict(x=np.asarray(x, 'float32'), meta=np.asarray(meta, 'int32'),
                          logit=np.asarray(rng.normal()+.5*labels[u], 'float32'))
    r = dict(train=bank, bank=bank, development=dev, confirmation=confirm,
             heldout=dev+confirm, groups=groups)
    return r, records, labels


def test_encoder_bank_head_patient_disjointness():
    r, _, _ = fixture()
    assert m.assert_exclusion(r['train'], r['bank'], r['development'], r['confirmation'], r['groups'])
    bad = dict(r['groups']); bad[r['confirmation'][0]] = bad[r['bank'][0]]
    with pytest.raises(ValueError, match='encoder/held-out'):
        m.assert_exclusion(r['train'], r['bank'], r['development'], r['confirmation'], bad)
    bad = dict(r['groups']); bad[r['confirmation'][0]] = bad[r['development'][0]]
    with pytest.raises(ValueError, match='inner exclusion'):
        m.assert_exclusion(r['train'], r['bank'], r['development'], r['confirmation'], bad)
    with pytest.raises(ValueError, match='training-only'):
        m.assert_exclusion(r['train'], [r['confirmation'][0]], r['development'], r['confirmation'], r['groups'])


def test_shuffle_preserves_vectors_size_and_exact_stratum_counts():
    from collections import Counter
    r, records, labels = fixture()
    bank = m.build_bank(records, r['bank'], labels)
    control = m.shuffle_bank(bank)
    assert np.array_equal(control['x'], bank['x'])
    assert np.array_equal(control['owners'], bank['owners'])
    assert np.array_equal(control['meta'][:, :3], bank['meta'][:, :3])
    assert Counter(m.strata(control['meta'])) == Counter(m.strata(bank['meta']))
    assert not np.array_equal(control['meta'][:, 3:6], bank['meta'][:, 3:6])
    again = m.shuffle_bank(bank)
    assert np.array_equal(again['meta'], control['meta'])


def test_persistence_requires_adjacent_slices_and_matches_protocol():
    r, records, labels = fixture()
    model = m.Distance(m.build_bank(records, r['bank'], labels))
    query = copy.deepcopy(records[r['confirmation'][0]])
    value, coverage = model.score(query)
    assert np.isfinite(value) and coverage == 1
    query['meta'][:, 7] *= 2
    assert np.isnan(model.score(query)[0])
    query['meta'][:, 1] = 0
    value, coverage = model.score(query)
    assert np.isnan(value) and coverage == 0


def test_fusion_cannot_read_confirmation_labels_or_fit_confirmation_scaler():
    r, records, y = fixture()
    z = {u: float(v['logit']) for u, v in records.items()}
    d = {u: float(i % 5) for i, u in enumerate(records)}
    pred = m.fit_head(r['development'], r['confirmation'], r['groups'], z, d, y)
    y2 = {u: v for u, v in y.items() if u not in r['confirmation']}
    assert np.array_equal(pred, m.fit_head(r['development'], r['confirmation'], r['groups'], z, d, y2))
    bad = dict(r['groups']); bad[r['confirmation'][0]] = bad[r['development'][0]]
    with pytest.raises(ValueError, match='head patient leakage'):
        m.fit_head(r['development'], r['confirmation'], bad, z, d, y)


def test_synthetic_end_to_end_and_single_class_ruler():
    r, records, y = fixture()
    result = m.analyze(r, records, y, tiny=True)
    assert set(result['primary']['auc']) == {'classifier', 'matched', 'shuffled'}
    assert result['primary']['valid_bootstraps'] == 20
    assert result['strict_bank_sensitivity']['bank_vectors'] > 0
    assert result['decision']['deployment_allowed'] is False
    assert result['clean_oof'] is False
    assert sum(result['top50_protocol_mix'].values()) == len(r['confirmation'])
    out = m.evaluate(np.zeros(6), {k: np.arange(6) for k in ('classifier','matched','shuffled')}, 5)
    assert out['auc']['matched'] is None
    assert out['ci95']['matched'] is None


@pytest.mark.parametrize('point,lopo,nuisance,reason', [
    ({'classifier': .7, 'matched': .8, 'shuffled': .8}, {'a': {'delta': .1}}, 0, 'matched <= shuffled'),
    ({'classifier': .7, 'matched': .8, 'shuffled': .75}, {'a': {'delta': -.01}}, 0, 'LOPO reverses benefit'),
    ({'classifier': .7, 'matched': .8, 'shuffled': .75}, {'a': {'delta': .1}}, 26, 'top deviations mainly protocol/FOV'),
])
def test_each_registered_rejection_is_binding(point, lopo, nuisance, reason):
    ids = [str(i) for i in range(50)]
    qa = dict(reviewer='independent', bank_retrieval_position_pass=True,
              top50={u: i < nuisance for i, u in enumerate(ids)})
    result = m.rejection(point, lopo, qa, ids)
    assert result['verdict'] == 'REJECT' and reason in result['rejects']


def test_missing_qa_is_hold_and_exact_50_required():
    p = dict(classifier=.7, matched=.8, shuffled=.75)
    assert m.rejection(p, {'a': {'delta': .1}}, None, [])['verdict'] == 'HOLD'


def test_hook_pure_observer_byte_identity_cpu():
    import torch
    from types import SimpleNamespace
    torch.manual_seed(1)
    model = SimpleNamespace(enc=torch.nn.Conv2d(3, 12, 3, padding=1).eval())
    x = torch.rand(4, 3, 11, 11)
    with torch.inference_mode():
        before = model.enc(x).numpy()
        hook = m.DenseHook(model)
        after = model.enc(x).numpy()
        hook.close()
    assert before.tobytes() == after.tobytes()
    assert hook.maps[0].shape == (4, 9, 32)
    assert not torch.cuda.is_available()


def test_shard_corruption_and_wrong_registration_fail(tmp_path):
    path = tmp_path / 'shard.npz'
    m.atomic_npz(path, x=np.zeros((5,32)), meta=np.zeros((5,10)), uid=np.asarray('uid'), logit=np.asarray(1.))
    m.safeio.atomic_json(path.with_suffix('.json'), dict(sha256=m.sha(path), registration_sha256='identity'))
    assert m.load_shard(path, 'uid', 'identity')['x'].shape == (5,32)
    with pytest.raises(ValueError, match='integrity'):
        m.load_shard(path, 'uid', 'wrong')
    with path.open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        m.load_shard(path, 'uid', 'identity')


def test_review_gate_rejects_absence_and_forged_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        m.review_gate()
    m.safeio.atomic_json(tmp_path/'admission.json', dict(verdict='PASS', builder='a', reviewer='a'))
    with pytest.raises(ValueError, match='independent'):
        m.review_gate()
    for name in ['registration.json','cpu-smoke.json','tests.txt','REVIEW.md']:
        (tmp_path/name).write_text('{"pass":true}')
    a = dict(verdict='PASS', builder='a', reviewer='b', registration_sha256='false')
    m.safeio.atomic_json(tmp_path/'admission.json', a)
    with pytest.raises(ValueError, match='evidence mismatch'):
        m.review_gate()


def test_memory_gate_and_progress_throughput_watchdogs(tmp_path):
    gib = 1024**3
    m.memory_gate(1*gib, 12*gib, 16*gib)
    with pytest.raises(ValueError):
        m.memory_gate(3*gib, 12*gib, 16*gib)
    with pytest.raises(ValueError):
        m.memory_gate(0, 13*gib, 24*gib)
    w = m.Watchdog(tmp_path)
    w.last = 100
    assert '15 minutes' in w.failed(1000)
    w.last = 2500
    w.commits = [0, 60, 120, 180, 240, 300, 2300]
    assert '25%' in w.failed(2500)


def test_patch_rows_keep_native_slice_and_contrast():
    raw = np.zeros((5, 12), 'float32')
    raw[:, 2] = [0, 1, 1, 2, 0]
    raw[:, 3] = [1, 3, 2, 1, 0]
    raw[:, 8] = 1
    raw[:, 9] = [7, 8, 9, 10, 0]
    x, meta = m.patch_rows(np.ones((5,9,32), 'float32'), raw)
    assert x.shape == (18,32)
    assert set(meta[:,7]) == {7,8}
    assert (meta[:,2] == 1).all()


@pytest.mark.skipif(not (m.OUT/'cpu-smoke.json').exists(), reason='real CPU receipt not yet generated')
def test_real_two_study_identity_receipt_is_bound_to_current_registration():
    receipt = m.read(m.OUT/'cpu-smoke.json')
    assert receipt['pass'] and receipt['hook_byte_identity'] and receipt['complete_forward']
    assert receipt['real_studies'] == 2
    assert receipt['registration_sha256'] == m.sha(m.OUT/'registration.json')
    assert len(receipt['logits_sha256']) == 2


def test_full_fov_not_local_crop_and_phys_step():
    raw = np.zeros((4,12), 'float32')
    raw[:,0] = [0,0,1,1]
    raw[:,3] = [0,1,0,1]
    raw[:,6] = np.array([90,100,180,100])/130
    raw[:,5] = np.log1p(2.)
    x, meta = m.patch_rows(np.ones((4,9,32)), raw)
    assert set(meta[:9,8]) == {0} and set(meta[9:,8]) == {2}
    assert set(meta[:,9]) == {2}


def test_phys_step_two_adjacent_and_retrieval_evidence():
    r, records, labels = fixture()
    model = m.Distance(m.build_bank(records, r['bank'], labels))
    query = copy.deepcopy(records[r['confirmation'][0]])
    query['meta'][:,7] *= 2
    query['meta'][:,9] = 2
    score, coverage, detail = model.score(query, evidence=True)
    assert np.isfinite(score) and coverage == 1
    assert len(detail['patches']) == 3
    assert all(p['nearest_bank_uid'] in r['bank'] for p in detail['patches'])


@pytest.mark.parametrize('spacing,expected', [(1.,3), (1.4,3)])
def test_phys_step_matches_frozen_float32_metadata_boundary(spacing, expected):
    raw = np.zeros((2,12), 'float32')
    raw[:,3] = [0,1]
    raw[:,5] = np.log1p(spacing)
    _, meta = m.patch_rows(np.ones((2,9,32)), raw)
    assert set(meta[:,9]) == {expected}


def test_missing_distance_forces_hold_and_empty_ruler():
    r, records, labels = fixture()
    for u in r['heldout']:
        records[u]['meta'][:,7] *= 2
    result = m.analyze(r, records, labels, tiny=True)
    assert 'missing study distance or patch coverage below 0.5' in result['decision']['holds']
    assert result['decision']['verdict'] == 'HOLD'
    assert m.evaluate(np.array([], int), {k:np.array([]) for k in ['classifier','matched','shuffled']})['valid_bootstraps'] == 0


def test_wddm_watchdog(tmp_path):
    w = m.Watchdog(tmp_path); w.last = 0
    assert 'WDDM' in w.failed(601, (100, 80, 15000, 16000))
    assert w.failed(601, (100, 180, 15000, 16000)) is None


def dicom(**changes):
    from pydicom.dataset import Dataset
    d = Dataset()
    for k,v in dict(EchoTime=20, RepetitionTime=2500, ScanningSequence='SE',
                    MRAcquisitionType='2D', ScanOptions='FS', **{}).items():
        setattr(d,k,v)
    for k,v in changes.items():
        if v is None:
            delattr(d,k)
        else:
            setattr(d,k,v)
    return d


@pytest.mark.parametrize('changes,expected', [
    ({},(1,1)), ({'ScanOptions':''},(1,0)), ({'EchoTime':70},(2,1)),
    ({'RepetitionTime':600,'EchoTime':11},(0,1)),
    ({'ScanOptions':None},(-1,-1)), ({'ScanOptions':'WE'},(-1,-1)),
    ({'EchoTime':float('nan')},(-1,-1)), ({'RepetitionTime':1200},(-1,-1)),
    ({'ScanningSequence':['SE','IR']},(-1,-1)), ({'ScanningSequence':'GR'},(-1,-1)),
    ({'MRAcquisitionType':'3D'},(-1,-1)), ({'InversionTime':100},(-1,-1)),
])
def test_contrast_whitelist(changes,expected):
    assert m.dicom_contrast(dicom(**changes)) == expected


def test_series_missing_or_conflicting_slice_is_unknown(tmp_path, monkeypatch):
    import pydicom
    folder=tmp_path/'series'; folder.mkdir()
    for name in ['a.dcm','b.dcm']:
        (folder/name).write_bytes(b'test')
    def fake(p, **kw):
        assert kw == dict(stop_before_pixels=True,specific_tags=m.CONTRAST_TAGS)
        return dicom(ScanOptions='' if p.name=='a.dcm' else 'FS')
    monkeypatch.setattr(pydicom,'dcmread',fake)
    codes,audit=m.series_contrasts(tmp_path)
    assert codes['series']==(-1,-1) and not audit['series']['consistent']


def test_unknown_never_matches_or_enters_bank_and_cross_contrast_fails():
    r,records,labels=fixture()
    bank=m.build_bank(records,r['bank'],labels)
    query=copy.deepcopy(records[r['heldout'][0]])
    for code in [-1,0,2]:
        query['meta'][:,1]=code
        score,coverage=m.Distance(bank).score(query)
        assert np.isnan(score) and coverage==0
    bank['meta'][:,1]=-1
    with pytest.raises(ValueError,match='unknown bank contrast'):
        m.Distance(bank)
    for u in r['bank']:
        records[u]['meta'][:,1]=-1
    with pytest.raises(ValueError,match='insufficient reference bank'):
        m.build_bank(records,r['bank'],labels)


def test_matching_lookup_does_not_mutate_frozen_raw_and_checks_bijection():
    raw=np.zeros((4,12),'float32'); raw[:,0]=[0,0,1,1]; raw[:,3]=[0,1,0,1]
    raw[:,7:9]=-1; original=raw.tobytes()
    assert m.series_lookup([{'series':'a'},{'series':'b'}],raw)=={0:'a',1:'b'}
    with pytest.raises(ValueError,match='duplicate series'):
        m.series_lookup([{'series':'a'},{'series':'a'}],raw)
    with pytest.raises(ValueError,match='mapping mismatch'):
        m.series_lookup([{'series':'a'}],raw)
    _,meta=m.patch_rows(np.ones((4,9,32)),raw,{0:(0,0),1:(2,1)})
    assert raw.tobytes()==original
    assert set(meta[:9,1])=={0} and set(meta[9:,1])=={2}


@pytest.mark.skipif(not (m.OUT/'cpu-smoke.json').exists(), reason='real CPU receipt not yet generated')
def test_real_retrieval_uses_known_equal_contrast_and_audited_series():
    receipt=m.read(m.OUT/'cpu-smoke.json')
    assert receipt['registration_sha256']==m.sha(m.OUT/'registration.json')
    assert receipt['contrast_source'].startswith('DICOM per-slice')
    assert all(c>0 for c in receipt['real_query_coverage'])
    retrieval=m.read(m.OUT/'cpu-smoke/retrieval.json')
    assert retrieval['metadata_columns'][1:3]==['contrast','fatsat']
    for p in retrieval['evidence']['patches']:
        q,b=p['query_meta'],p['nearest_bank_meta']
        assert q[:6]==b[:6]
        assert q[1] in (0,1,2) and q[2] in (0,1)
    r=m.read(m.OUT/'registration.json')
    for uid in [r['bank'][0],r['heldout'][0]]:
        with np.load(m.shard_path(m.OUT/'cpu-smoke',uid),allow_pickle=False) as record:
            audit=json.loads(str(record['contrast_audit']))
            assert audit and all(a['slices']>0 for a in audit.values())
            assert any(a['code'][0]>=0 and a['consistent'] for a in audit.values())
