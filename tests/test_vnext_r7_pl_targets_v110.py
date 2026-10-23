"""CPU exact-target equivalence, heldout authentication and leakage contracts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tools import vnext_r7_pl_targets_v110 as t


def fixture_cohort():
    ids = [f'study-{i}' for i in range(50)]
    rows = pd.DataFrame({t.pl.UID: ids, 'component_id': ids, 'fold': [str(i % 5) for i in range(50)]})
    return t.build_cohort(rows, [], ids), ids


def numerical(ids):
    rng = np.random.default_rng(22)
    return ({u: rng.uniform(.1, .9, 12).astype(np.float32) for u in ids},
            {u: rng.uniform(.01, .99, 12).astype(np.float32) for u in ids})


@pytest.mark.parametrize('arm', ['global', 'fill'])
@pytest.mark.parametrize('mode,fold', [('fivefold', 0), ('all-data', 'all')])
def test_exact_screen_math_and_materialized_roundtrip(tmp_path, arm, mode, fold):
    cohort, ids = fixture_cohort()
    l2, teacher = numerical(ids)
    train, val = t.selected_rosters(cohort, mode, fold)
    mask = np.random.default_rng(9).uniform(size=(len(train), 12)) > .5
    s = t.materialize_targets(tmp_path, cohort, l2, teacher, None, arm, mode, fold, 42, 'a'*64, mask, engineering=True)
    _, got = t.load_materialized(tmp_path)
    # Screen build applies these imported functions to full ids after fitting on train.
    scaled, _ = t.pl.scaled_teacher(l2, teacher, train, train)
    want = t.pl.arm_targets(t.pl.aligned(l2, train), scaled, mask)[arm]
    np.testing.assert_array_equal(t.pl.aligned(got, train), want)
    np.testing.assert_array_equal(np.asarray(s['weights'], dtype=np.float32), t.pl.frozen_weights(l2, train))
    assert not set(got) & set(val)


def test_validation_gold_values_cannot_affect_training_targets(tmp_path):
    cohort, ids = fixture_cohort()
    l2, teacher = numerical(ids)
    train, val = t.selected_rosters(cohort, 'fivefold', 0)
    mask = np.zeros((len(train), 12), dtype=bool)
    t.materialize_targets(tmp_path/'a', cohort, l2, teacher, None, 'global', 'fivefold', 0, 42, 'a'*64, mask, engineering=True)
    for u in val:
        l2[u] = np.full(12, np.nan)
        teacher[u] = np.full(12, np.nan)
    t.materialize_targets(tmp_path/'b', cohort, l2, teacher, None, 'global', 'fivefold', 0, 42, 'a'*64, mask, engineering=True)
    a, av = t.load_materialized(tmp_path/'a')
    b, bv = t.load_materialized(tmp_path/'b')
    np.testing.assert_array_equal(t.pl.aligned(av, train), t.pl.aligned(bv, train))
    assert a['weights'] == b['weights']


def test_component_and_gold_exclusion():
    cohort, ids = fixture_cohort()
    rows = pd.DataFrame({t.pl.UID: ids, 'component_id': ids, 'fold': [str(i % 5) for i in range(50)]})
    gold = ids[:3]
    cohort = t.build_cohort(rows, gold, ids, ids+['author-only'])
    train, val = t.selected_rosters(cohort, 'fivefold', 0)
    assert not set(gold) & set(train+val)
    assert cohort['reconciliation']['author_only'] == ['author-only']
    rows.loc[1, 'component_id'] = ids[0]
    with pytest.raises(ValueError, match='component crosses'):
        t.build_cohort(rows, gold, ids)


def test_reject_target_and_spec_tampering(tmp_path):
    cohort, ids = fixture_cohort()
    l2, teacher = numerical(ids)
    t.materialize_targets(tmp_path, cohort, l2, teacher, None, 'global', 'all-data', 'all', 42,
                          'a'*64, np.zeros((len(ids), 12), dtype=bool), engineering=True)
    with (tmp_path/'targets.npz').open('ab') as f:
        f.write(b'tamper')
    with pytest.raises(ValueError, match='targets changed'):
        t.load_materialized(tmp_path)


def test_production_rejects_substituted_l2_before_reading_labels(tmp_path, monkeypatch):
    cohort, _ = fixture_cohort()
    l2 = tmp_path/'substitute.csv'
    l2.write_bytes(b'synthetic substitute; must not parse')
    def forbidden(*args, **kwargs):
        raise AssertionError('label table opened before pin validation')
    monkeypatch.setattr(t.pl, 'table', forbidden)
    with pytest.raises(ValueError, match='L2 asset differs'):
        t.materialize_targets(tmp_path/'out', cohort, l2, tmp_path/'teacher.csv',
            tmp_path/'seal.json', 'global', 'all-data', 'all', 42, 'a'*64)


def test_loading_production_bundle_revalidates_teacher_seal(tmp_path, monkeypatch):
    cohort, ids = fixture_cohort()
    l2, teacher = numerical(ids)
    s = t.materialize_targets(tmp_path, cohort, l2, teacher, None, 'global', 'all-data', 'all', 42,
        'a'*64, np.zeros((len(ids), 12), dtype=bool), engineering=True)
    seal = tmp_path/'teacher-seal.json'
    seal.write_text('{}')
    s.update(engineering_only=False, teacher_seal_path=str(seal), teacher_sha256=t.sha(seal))
    s['spec_sha256'] = t.digest({k:v for k,v in s.items() if k != 'spec_sha256'})
    t.pl.write(tmp_path/'target-spec.json', s)
    def fail(path):
        raise ValueError('nested producer journal changed')
    monkeypatch.setattr(t, 'validate_teacher_seal', fail)
    with pytest.raises(ValueError, match='nested producer journal changed'):
        t.load_materialized(tmp_path)


def producer_fixture(tmp_path):
    h = t.producer
    ids = ['study-1', 'study-2']
    runtime = tmp_path/'stage/r7_runtime/compact_infer.py'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('# fixture')
    identity = dict(**h.EXPOSURE, role_provenance=h.role_provenance('heldout,f0'), version='1.1.2',
        auxiliary_device='cpu', device='cpu', models='heldout,f0', uids=ids,
        runner_sha256=t.sha(h.__file__), pipeline_sha256=t.sha(t.pipeline.__file__),
        monitor_sha256=t.sha(t.monitor.__file__), runtime_sha256=t.sha(runtime),
        bundle={'manifest_sha256':h.MANIFEST_SHA, 'manifest_pinned':True, 'file_sha256':{m:'a'*64 for m in h.r7.MODELS}},
        code={'tree_sha256':'a'*64, 'r7_closure_sha256':{'source':'b'*64}}, deps={'tree_sha256':h.DEPS_SHA})
    fingerprint = h.fingerprint(identity)
    common = dict(**h.EXPOSURE, role_provenance=h.role_provenance('heldout,f0'), version='1.1.2', identity=fingerprint, auxiliary_device='cpu')
    t.pl.write(tmp_path/'identity.json', identity)
    t.pl.write(tmp_path/'run_receipt.json', dict(common, models='heldout,f0', selected=2, completed=2, failures=0, exit_code=0))
    t.pl.write(tmp_path/'receipt_dual.json', dict(common, models=h.r7.MODELS, status='PASS', device='cpu', studies=2, scored=3, fallback={}))
    records = {}
    for uid in ids:
        k = h.fold_k(uid)
        rec = dict(common, StudyInstanceUID=uid, fold_k=k, models='heldout,f0',
            role_folds={'heldout': k, 'f0':0}, roles_coincide=k == 0, status='ok', pooled_prior_bias=True,
            predictions={str(f): {'raw':[.3]*12, 'post':[.4]*12} for f in {k,0}})
        t.pl.write(h.record_path(tmp_path, uid), rec)
        records[uid] = rec
    t.pipeline.export_roles(tmp_path, ids, records)
    return ids


def test_producer_roundtrip_and_pinned_journal(tmp_path):
    ids = producer_fixture(tmp_path)
    seal = tmp_path/'sealed.json'
    s = t.seal_teacher(tmp_path, seal, ids)
    assert s['role'] == 'heldout' and s['producer_exposure'] == 'EXPOSED'
    t.validate_teacher_seal(seal)
    p = t.producer.record_path(tmp_path, ids[0])
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError, match='input hash changed'):
        t.validate_teacher_seal(seal)


@pytest.mark.parametrize('column,value', [('model_role','f0'), ('output_kind','post_stacker'),
    ('prediction_reference','wrong'), ('model_fold','99'), ('exposure','CLEAN')])
def test_producer_wrong_role_or_reference_rejected(tmp_path, column, value):
    producer_fixture(tmp_path)
    path = tmp_path/'heldout_raw.csv'
    frame = pd.read_csv(path)
    frame[column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        t.seal_teacher(tmp_path, tmp_path/'seal.json')


def test_incomplete_and_wrong_order_rejected(tmp_path):
    ids = producer_fixture(tmp_path)
    with pytest.raises(ValueError, match='ordered roster'):
        t.seal_teacher(tmp_path, tmp_path/'seal.json', ids[::-1])
    p = tmp_path/'run_receipt.json'
    run = t.read(p)
    run['completed'] = 1
    t.pl.write(p, run)
    with pytest.raises(ValueError, match='incomplete'):
        t.seal_teacher(tmp_path, tmp_path/'seal.json')
