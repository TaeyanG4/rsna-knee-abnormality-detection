"""Synthetic v2 contracts and shipped code bindings; no production data loaded."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import oof_error_audit_v2 as audit


@pytest.fixture
def synthetic():
    variants = []
    rng = np.random.default_rng(813)
    ids = {f: [f'f{f}u{i}' for i in range(8)] for f in range(5)}
    raw = {}
    for name, seeds in [('l2', [42, 43]), ('dino', [42]), ('conv', [42]), ('h1', [42])]:
        members = []
        for seed in seeds:
            for fold in range(5):
                members.append(dict(seed=seed, fold=fold, path=f'{name}/S{seed}_F{fold}.csv', sha256='a'*64))
                # Different fold calibration plus ties expose accidental global-first ranking.
                raw[name, seed, fold] = (np.round(rng.uniform(size=(8, 12)), 1) + fold)/6
        variants.append(dict(id=name, family=name, label_recipe='synthetic', enabled=True,
                             prediction_policy='plain K36', expected_seeds=seeds,
                             path_regex=rf'{name}/S(?:42|43)_F[0-4]\.csv', members=members))
    registry = dict(schema='oof-error-audit-registry-v2', version='2.0.1', expected_studies=40,
                    folds=list(range(5)), variants=variants,
                    bundles=[dict(id='bundle', members=dict(l2='l2', dino='dino', convnext='conv'))])
    return registry, ids, raw


def references(y, mask=None, minimum=1):
    y = np.asarray(y, float)
    mask = np.ones(y.shape, bool) if mask is None else mask
    return {k: (y.copy(), mask.copy(), minimum) for k in ('jev', 'aquila', 'dread', 'consensus')}


def test_aggregation_exactly_matches_registered_helpers(synthetic):
    registry, ids, raw = synthetic
    actual = audit.aggregate_variants(registry, ids, raw)
    pooled = {}
    for v in registry['variants']:
        pooled[v['id']] = np.concatenate([
            np.mean([audit.evaluation.pct_rank(raw[v['id'], s, f]) for s in v['expected_seeds']], axis=0)
            for f in range(5)])
        np.testing.assert_array_equal(actual[v['id']], audit.bundle_helper.global_rank(pooled[v['id']]))
    expected, _ = audit.bundle_helper.bundle_of({k: pooled[v] for k, v in registry['bundles'][0]['members'].items()})
    np.testing.assert_array_equal(actual['bundle'], expected)
    wrong = audit.bundle_helper.global_rank(np.concatenate([raw['l2', 42, f] for f in range(5)]))
    assert not np.array_equal(actual['l2'], wrong)


def test_new_family_arm_is_config_only(synthetic):
    registry, ids, raw = synthetic
    variant = copy.deepcopy(registry['variants'][1])
    variant.update(id='coverage_arm', family='dino_coverage', path_regex=r'coverage/S42_F[0-4]\.csv')
    for m in variant['members']:
        m['path'] = f"coverage/S42_F{m['fold']}.csv"
        raw['coverage_arm', 42, m['fold']] = raw['dino', 42, m['fold']].copy()
    registry['variants'].append(variant)
    registry['bundles'].append(dict(id='bundle_coverage', members=dict(l2='l2', dino='coverage_arm', convnext='conv')))
    out = audit.aggregate_variants(registry, ids, raw)
    np.testing.assert_array_equal(out['bundle'], out['bundle_coverage'])


def test_explicit_common_fold_variant_audit_is_config_only(synthetic):
    registry, ids, raw = synthetic
    registry['folds'] = [0]
    registry['expected_studies'] = len(ids[0])
    for v in registry['variants']:
        v['members'] = [m for m in v['members'] if m['fold'] == 0]
    result = audit.aggregate_variants(registry, ids, raw)
    assert all(p.shape == (8, 12) for p in result.values())


def test_wrong_variant_identity_is_rejected(synthetic):
    registry, _, _ = synthetic
    conv = registry['variants'][2]
    conv['path_regex'] = r'artifacts/EXP170/S42_F[0-4]/oof_k36_flip_tta\.csv'
    for m in conv['members']:
        m['path'] = f"artifacts/EXP170/S42_F{m['fold']}/oof_k36_flip_tta.csv"
    audit.validate_registry(registry)
    conv['members'][0]['path'] = 'artifacts/EXP166/S42_F0/oof_k36_flip_tta.csv'
    with pytest.raises(ValueError, match='identity'):
        audit.validate_registry(registry)


def test_missing_or_duplicate_seed_fold_fails(synthetic):
    registry, _, _ = synthetic
    registry['variants'][0]['members'][1] = registry['variants'][0]['members'][0].copy()
    with pytest.raises(ValueError, match='grid'):
        audit.validate_registry(registry)


def test_optional_unavailable_variant_never_enters_bundle(synthetic):
    registry, _, _ = synthetic
    registry['variants'].append(dict(id='exp172_pending', enabled=False, members=[dict(sha256=None)]))
    audit.validate_registry(registry)
    registry['bundles'][0]['members']['dino'] = 'exp172_pending'
    with pytest.raises(ValueError, match='enabled'):
        audit.validate_registry(registry)


def test_prediction_hash_binding_rejects_changed_bytes(tmp_path):
    p = tmp_path/'oof.csv'
    p.write_text('synthetic-only', encoding='utf-8')
    entry = dict(path='oof.csv', sha256=audit.sha(p))
    audit.verify_entry(tmp_path, entry)
    p.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        audit.verify_entry(tmp_path, entry)


def test_shipped_registry_helper_bindings_are_complete_and_current():
    """Defect 1: exercise the checked-in pins, not only synthetic verify_entry."""
    registry = audit.read(audit.ROOT/audit.REGISTRY)
    audit.validate_registry(registry)
    helpers = [entry for entry in registry['bindings'] if entry.get('role') == 'helper']
    own_sources = {'tools/oof_error_audit_v2.py', 'tools/oof_error_audit_v2.md',
                   'tests/test_oof_error_audit_v2.py'}
    expected = set(audit.SOURCE_FILES) - own_sources
    assert len(helpers) == len(expected)
    assert {entry['path'] for entry in helpers} == expected
    for entry in helpers:
        audit.verify_entry(audit.ROOT, entry)


def test_stale_helper_pin_fails_before_loading_data(synthetic, monkeypatch, tmp_path):
    """Defect 1 must remain fail-closed; never silently refresh a pinned hash."""
    registry, _, _ = synthetic
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    registry['roster_manifest'] = dict(path=manifest.name, sha256=audit.sha(manifest))
    helper = tmp_path/'helper.py'
    helper.write_text('# reviewed helper\n', encoding='utf-8')
    entry = dict(path=helper.name, sha256=audit.sha(helper), role='helper')
    registry['bindings'] = [entry]
    audit.verify_entry(tmp_path, entry)
    helper.write_text('# changed since review\n', encoding='utf-8')

    def unexpected_read(*args, **kwargs):
        pytest.fail('Stale helper must fail before any manifest or data is loaded')

    monkeypatch.setattr(audit, 'read', unexpected_read)
    monkeypatch.setattr(audit.bundle_helper, 'load_oof', unexpected_read)
    with pytest.raises(ValueError, match=r'Input hash mismatch: helper\.py'):
        audit.load_inputs(tmp_path, registry, audit.aggregation.Budget(60))
    assert entry['sha256'] != audit.sha(helper)


def test_pooled_and_saved_oof_duplicate_ids_fail(synthetic, tmp_path):
    registry, ids, raw = synthetic
    ids[1][0] = ids[0][0]
    with pytest.raises(ValueError, match='Duplicate'):
        audit.aggregate_variants(registry, ids, raw)
    path = tmp_path/'oof.csv'
    frame = pd.DataFrame(np.full((2, 12), .5), columns=audit.TARGETS)
    frame.insert(0, audit.UID, ['u0', 'u0'])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='IDs differ'):
        audit.bundle_helper.load_oof(path, ['u0', 'u1'])


def test_registered_reference_drop_duplicates_is_prevented(tmp_path):
    path = tmp_path/'labels.csv'
    pd.DataFrame({audit.UID: ['s0', 's0'], 'ACL': [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        audit.assert_reference_ids(path)


def test_reference_preflight_rejects_non_label_columns_without_reading_them(tmp_path):
    path = tmp_path/'labels.csv'
    path.write_text(audit.UID + ',Report\nu0,SYNTHETIC_NOT_A_REPORT\n', encoding='utf-8')
    with pytest.raises(ValueError, match='non-label'):
        audit.assert_reference_ids(path)


def test_threshold_boundaries_are_strict_and_missing_is_not_negative():
    y = np.tile(np.array([1, 1, 0, 0, np.nan])[:, None], (1, 12))
    mask = np.isfinite(y)
    # Known prevalence .5 -> FP threshold .75. Equality is neither FP nor FN.
    p = np.tile(np.array([.49, .5, .75, .751, .999])[:, None], (1, 12))
    refs = references(y, mask)
    refs['aquila'][1][0, :] = False
    strata, rows, threshold = audit.error_strata(p, refs)
    np.testing.assert_array_equal(threshold, np.full(12, .75))
    assert strata['FN'][:, 0].tolist() == [True, False, False, False, False]
    assert strata['FP'][:, 0].tolist() == [False, False, False, True, False]
    assert strata['unknown_agreement_FN'][0, 0]
    assert not strata['disputed_FN'][0, 0]
    assert not strata['agreed_pos_FN'][0, 0]
    assert strata['agreed_neg_middle'][2, 0]
    assert rows[0]['jev_known'] == 4
    assert rows[0]['FN_disputed_share'] is None


def test_invalid_known_label_and_non_boolean_mask_fail():
    y = np.zeros((4, 12)); mask = np.ones(y.shape, bool)
    y[0, 0] = np.nan
    with pytest.raises(ValueError, match='known reference'):
        audit.checked_reference((y, mask, 1), 4)
    mask[0, 0] = False
    clean, _, _ = audit.checked_reference((y, mask, 1), 4)
    assert clean[0, 0] == 0
    with pytest.raises(ValueError, match='mask'):
        audit.checked_reference((clean, mask.astype(int), 1), 4)


def test_masked_auc_counts_deficits_and_minimum_classes():
    y = np.tile(np.array([0, 1, 0, 1, 0, 1])[:, None], (1, 12)).astype(float)
    p = y*.8 + .1
    mask = np.ones(y.shape, bool); mask[1:, 0] = False
    weights = np.ones((10, 6))
    rows, pairs, contributions = audit.metric_tables(
        {'a': p, 'b': p.copy()}, {'consensus': (y, mask, 2)}, np.zeros(6, int), weights,
        audit.aggregation.Budget(60))
    acl = next(r for r in rows if r['family'] == 'a' and r['target'] == 'ACL' and r['fold'] == 'all')
    assert acl['n'] == 1 and acl['excluded'] == 5 and acl['pos'] == 0 and acl['neg'] == 1
    assert acl['auc'] is None and acl['status'] == 'INSUFFICIENT_CLASSES'
    assert np.isnan(contributions['a', 'consensus', 'ACL']).all()
    mcl = next(r for r in pairs if r['family_A'] == 'a' and r['target'] == 'MCL')
    assert mcl['delta_auc'] == 0 and mcl['ci95'] == [0., 0.] and mcl['valid_draws'] == 10
    macro = next(r for r in pairs if r['family_A'] == 'a' and r['target'] == 'macro12')
    assert macro['delta_auc'] is None and macro['valid_draws'] == 0


def test_bootstrap_reuses_v1_statistic_and_legacy_conditional_intervals():
    rng = np.random.default_rng(59)
    y = np.tile((np.arange(40) % 2)[:, None], (1, 12))
    p, q = rng.random((40, 12)), rng.random((40, 12))
    groups = [np.array([i, i+1]) for i in range(0, 40, 2)]
    draw_ids = audit.evaluation.bootstrap_index(groups, 30, audit.SEED)
    weights = np.stack([np.bincount(ii, minlength=40) for ii in draw_ids])
    ref = (y, np.ones(y.shape, bool), 1)
    a = audit.aggregation.weighted_auc(p, ref, weights)
    b = audit.aggregation.weighted_auc(q, ref, weights)
    for k, w in enumerate(weights):
        for j in range(12):
            assert a[k, j] == pytest.approx(audit.legacy.weighted_auc(y[:, j], p[:, j], w))
    old = audit.legacy.conditional_intervals(y, {'R1': p, 'candidate': q}, np.repeat(np.arange(20), 2), 30, audit.SEED)
    for j, target in enumerate(audit.TARGETS):
        ci = old['models']['candidate']['paired_delta_vs_R1'][target]
        np.testing.assert_allclose(np.quantile(b[:, j]-a[:, j], [.025, .975]), [ci['low'], ci['high']])


def test_target_confusion_includes_middle_and_uses_other_label_mask():
    y = np.zeros((6, 12)); y[0, :] = 1
    p = np.tile(np.array([.2, .2, .6, .7, .95, .99])[:, None], (1, 12))
    refs = references(y)
    contusion = audit.TARGETS.index('Contusion')
    refs['jev'][1][2, contusion] = False
    strata = audit.error_strata(p, refs)[0]
    table = audit.target_confusion(p, refs, strata)
    row = next(r for r in table if r['target'] == 'MCL' and r['other_target'] == 'Contusion' and r['stratum'] == 'agreed_neg_middle')
    assert row['stratum_n'] == 2 and row['other_known'] == 1 and row['other_unknown'] == 1
    assert {r['target'] for r in table} == {'MCL', 'Fracture', 'Lateral OA', 'PF OA'}


def test_mcl_controls_exact_match_without_replacement_and_unmatched_retained():
    ids = [f'u{i}' for i in range(7)]
    y = np.zeros((7, 12)); y[6, :] = 1
    for t in ('Contusion', 'Effusion'):
        y[:5, audit.TARGETS.index(t)] = [1, 1, 1, 1, 0]
    refs = references(y)
    strata = {k: np.zeros(y.shape, bool) for k in ('agreed_neg_FP', 'agreed_neg_low')}
    j = audit.TARGETS.index('MCL')
    strata['agreed_neg_FP'][:3, j] = True
    strata['agreed_neg_low'][3:6, j] = True
    matches = audit.match_mcl_controls(ids, np.zeros(7, int), refs, strata)
    matched = [m for m in matches if m['status'] == 'MATCHED']
    assert len(matched) == 1 and matched[0]['control_id'] == 'u3'
    assert len(matches) == 3


class FakeCorpus:
    spec = dict(depth=64, slots=[['Sagittal', 1, 18], ['Sagittal', 0, 14],
                                ['Coronal', 1, 12], ['Coronal', 0, 8], ['Axial', -1, 12]])

    def __init__(self):
        self.data = np.broadcast_to(np.arange(64, dtype=np.uint8)[:, None, None], (64, 8, 8))
        self.valid = np.ones(64, np.uint8)

    def volume(self, uid):
        return self.data

    def __contains__(self, uid):
        return True

    def mask(self, uid):
        return self.valid


def test_quantile_slots_use_filled_positions_and_handle_missing():
    corpus = FakeCorpus()
    corpus.valid[:18] = 0
    corpus.valid[[1, 4, 7, 13, 16]] = 1
    assert audit.slot_indices(corpus, 'synthetic', 'Sagittal_fluid1') == [4, 4, 7, 13, 13]
    corpus.valid[18:32] = 0
    assert audit.slot_indices(corpus, 'synthetic', 'Sagittal_fluid0') == [None]*5


def test_montage_blinding_same_pixels_despite_role_changes(tmp_path):
    corpus = FakeCorpus()
    case = dict(code='C0001', anatomy='meniscus', study_id='synthetic_private_uid', strata=['agreed_pos_FN'])
    a, b = tmp_path/'panel_0001.png', tmp_path/'panel_0002.png'
    public = audit.render_blind_panel(corpus, case, a)
    case['strata'] = ['agreed_pos_TP_controls']
    audit.render_blind_panel(corpus, case, b)
    assert a.read_bytes() == b.read_bytes()
    text = ' '.join(public['rendered_text'])
    for forbidden in ('synthetic_private_uid', 'FN', 'TP', 'control', 'strat', 'agreed'):
        assert forbidden not in text
    assert len(public['slot_indices']) == 4
    assert all(len(v) == 5 for v in public['slot_indices'].values())
    with Image.open(a) as im:
        assert not im.info
    case['code'] = 'FN-C0001'
    with pytest.raises(ValueError, match='neutral'):
        audit.render_blind_panel(corpus, case, a)


def test_mcl_panel_has_all_three_groups_and_five_fractions(tmp_path):
    row = audit.render_blind_panel(FakeCorpus(), dict(code='C0002', anatomy='mcl', study_id='u1'), tmp_path/'panel.png')
    assert set(row['slot_indices']) == {'Coronal_fluid1', 'Coronal_fluid0', 'Axial'}
    assert len(row['rendered_text']) == 16


@pytest.mark.parametrize('path', ['../escape.csv', '.local/relabel/anything.csv',
                                  'storage/competition/train.csv', '.local/token.json', 'data/*.csv'])
def test_protected_or_implicit_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        audit.safe_path(tmp_path, path)


def test_verification_receipt_requires_current_separate_pass(monkeypatch, tmp_path):
    registry = tmp_path/'registry.json'; registry.write_text('{}', encoding='utf-8')
    note_path = tmp_path/'verification.json'
    monkeypatch.setattr(audit, 'source_hashes', lambda root: {'tool.py': 'a'*64})
    note = dict(status='PASS', tool_version='2.0.1', reviewer='gpt-6-astra', separate_pass=True,
                source_sha256={'tool.py': 'a'*64}, registry_sha256=audit.sha(registry))
    note_path.write_text(json.dumps(note), encoding='utf-8')
    audit.verify_note(tmp_path, note_path, registry)
    note['source_sha256']['tool.py'] = 'b'*64
    note_path.write_text(json.dumps(note), encoding='utf-8')
    with pytest.raises(ValueError, match='stale'):
        audit.verify_note(tmp_path, note_path, registry)


def test_rebinding_helper_invalidates_existing_verification(monkeypatch, tmp_path):
    """Updating a stale registry pin cannot reuse the previous PASS receipt."""
    registry_path = tmp_path/'registry.json'
    registry = dict(bindings=[dict(path='helper.py', sha256='a'*64, role='helper')])
    registry_path.write_text(json.dumps(registry), encoding='utf-8')
    monkeypatch.setattr(audit, 'source_hashes', lambda root: {'tool.py': 'b'*64})
    note_path = tmp_path/'verification.json'
    note = dict(status='PASS', tool_version=audit.VERSION, reviewer='gpt-6-astra',
                separate_pass=True, source_sha256={'tool.py': 'b'*64},
                registry_sha256=audit.sha(registry_path))
    note_path.write_text(json.dumps(note), encoding='utf-8')
    audit.verify_note(tmp_path, note_path, registry_path)
    registry['bindings'][0]['sha256'] = 'c'*64
    registry_path.write_text(json.dumps(registry), encoding='utf-8')
    with pytest.raises(ValueError, match='Verification is stale'):
        audit.verify_note(tmp_path, note_path, registry_path)


def test_synthetic_package_end_to_end(synthetic, monkeypatch, tmp_path):
    """Exercise serialization, counts, panels and final receipt with no real inputs."""
    registry, by_fold, raw = synthetic
    ids = [u for f in range(5) for u in by_fold[f]]
    folds = np.repeat(np.arange(5), 8)
    families = audit.aggregate_variants(registry, by_fold, raw)
    y = np.tile((np.arange(40) % 2)[:, None], (1, 12))
    refs = references(y)
    fake_bootstrap = dict(unit='synthetic component', draws=300, seed=audit.SEED)
    monkeypatch.setattr(audit, 'load_inputs', lambda *args: (
        ids, folds, families, refs, np.ones((300, 40)), fake_bootstrap, {}))
    monkeypatch.setattr(audit, 'source_hashes', lambda root: {'synthetic': 'a'*64})
    from tools import exp175_image_domain_audit as images
    monkeypatch.setattr(images, 'open_corpus', lambda path: FakeCorpus())
    cache = tmp_path/audit.CORPUS; cache.mkdir(parents=True)
    for name in ('manifest.json', 'ids.npy', 'masks.npy'):
        (cache/name).write_bytes(b'synthetic placeholder; open_corpus is mocked')
    registry_path = tmp_path/'registry.json'
    registry_path.write_text(json.dumps(registry), encoding='utf-8')
    verification = tmp_path/'verification.json'
    verification.write_text(json.dumps(dict(status='PASS', tool_version=audit.VERSION,
        reviewer='gpt-6-astra', separate_pass=True, source_sha256={'synthetic': 'a'*64},
        registry_sha256=audit.sha(registry_path))), encoding='utf-8')
    result = audit.run(tmp_path, registry_path, tmp_path/'output', verification, seconds=60, per_stratum=2)
    assert result['status'] == 'COMPLETE'
    out = tmp_path/'output'
    assert not (out/'FAILED.json').exists()
    report = json.loads((out/'result.json').read_text())
    assert report['studies'] == 40 and report['tool_version'] == audit.VERSION
    metrics = pd.read_csv(out/'auc_family_reference_target_fold.csv')
    assert len(metrics) == 5*4*12*6
    key = json.loads((out/'montage_key.json').read_text())
    index = json.loads((out/'blind/index.json').read_text())
    assert len(key['cases']) == len(index['panels']) > 0
    assert 'study_id' not in json.dumps(index) and 'strata' not in json.dumps(index)
    with pytest.raises(ValueError, match='NEW'):
        audit.run(tmp_path, registry_path, out, verification, seconds=60)
