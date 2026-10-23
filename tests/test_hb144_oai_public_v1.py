"""CPU-only contract/regression tests; all outcomes below are synthetic."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import hb144_oai_public_v1 as h


def test_ranks_ties_and_fixed_blends():
    a = np.tile(np.array([.1, .1, .4, .8])[:, None], (1, 12))
    b = a[::-1].copy()
    np.testing.assert_allclose(h.rank(a)[:, 0], [.375, .375, .75, 1])
    actual = h.blends({'R': a, 'R7': b, 'S6': a, 'R2': b})
    for key, w in [('P_R7', .5), ('R_R7_40_60', .4), ('R_R7_30_70', .3)]:
        np.testing.assert_allclose(actual[key], w*h.rank(a)+(1-w)*h.rank(b))
    np.testing.assert_array_equal(actual['P956'], h.rank(a))
    np.testing.assert_array_equal(actual['R2_in_P_R7'], h.rank(b))
    assert set(h.blends({'R': a})) == {'R'}  # no invented missing member


def synthetic():
    y = np.tile(np.array([0.] * 30 + [1.] * 30)[:, None], (1, 12))
    y[:, [1, 10, 11]] = np.nan
    p = np.tile(np.linspace(.05, .95, 60)[:, None], (1, 12))
    return y, p


def test_fixed_nine_bootstrap_missing_targets_and_correlations():
    y, p = synthetic()
    r = h.statistics(y, {'good': p, 'reverse': 1-p, 'constant': p*0+.5}, draws=20)
    assert r['members']['good']['macro9']['auc'] == 1
    assert r['members']['good']['macro9']['ci95'] == [1, 1]
    assert r['members']['reverse']['weak4']['auc'] == 0
    assert r['members']['constant']['macro9']['auc'] == .5
    assert r['members']['good']['per_target']['MCL']['auc'] is None
    assert r['members']['good']['per_target']['MCL']['valid_draws'] == 0
    assert r['spearman']['good:constant']['ACL'] is None
    assert r['spearman']['good:reverse']['ACL'] == -1
    json.dumps(r, allow_nan=False)


def test_macro_never_drops_undefined_target():
    y, p = synthetic()
    y[:, 0] = np.nan
    r = h.statistics(y, {'R': p}, draws=10)
    assert r['members']['R']['macro9']['auc'] is None
    assert r['members']['R']['macro9']['ci95'] is None
    assert r['members']['R']['weak4']['auc'] == 1


def test_one_class_draws_and_small_class_ci_withheld():
    y = np.tile([0., 1.], (12, 1)).T
    p = y*.8+.1
    r = h.statistics(y, {'R': p}, draws=100)
    target = r['members']['R']['per_target']['ACL']
    assert 0 < target['valid_draws'] < 100
    assert target['ci95'] is None and not target['eligible']


def test_paired_shared_bootstrap_reproducible():
    y, p = synthetic()
    members = {'R': p, 'S6': p, 'R7': p, 'R2': p}
    a = h.statistics(y, members, draws=12)
    b = h.statistics(y, members, draws=12)
    assert a == b
    for comparison in a['paired_comparisons'].values():
        assert comparison['weak_four']['delta'] == 0
        assert comparison['weak_four']['ci95'] == [0, 0]


@pytest.mark.parametrize('value', [np.zeros((2, 11)), np.full((2, 12), np.nan),
                                   np.full((2, 12), 1.01), np.full((2, 12), -.1)])
def test_invalid_predictions_rejected(value):
    with pytest.raises(ValueError, match='Invalid predictions'):
        h.probs(value, 2)


def test_frozen_route_never_reuses_sagittal():
    mod = h.runtime()
    rows = [dict(SeriesInstanceUID='s', Anatomical_Plane='Sagittal', Fluid_Sensitive=1, group_key='s'),
            dict(SeriesInstanceUID='c', Anatomical_Plane='Coronal', Fluid_Sensitive=0, group_key='c')]
    routes = h.routes(mod, rows)
    assert [r['series'] for r in routes] == ['s', None, 'c', None, None]
    assert [r['slices'] for r in routes] == [26, 22, 18, 12, 18]


def test_unknown_native_mapping_rejected():
    group = dict(study_key='test', eligible=True, group_key='g', series_key='s',
                 source_sequence='UNKNOWN', plane='Sagittal', fluid=1,
                 contrast_family='IW_TSE_FS', contrast_verified=True)
    with pytest.raises(ValueError, match='mapping'):
        h.records(SimpleNamespace(groups={'g': group}), 'test')


def test_review_is_bound_to_source_registration_scope_and_members(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'validate', lambda: {})
    monkeypatch.setattr(h, 'sha', lambda p: 'digest')
    monkeypatch.setattr(h, 'smoke_binding', lambda: 'smoke-digest')
    assert h.admit()['status'].startswith('HOLD')
    v = dict(verdict='PASS', independent=True, reviewer='separate-review-session',
             scope='DESCRIPTIVE_PUBLIC_OAI_ANCESTRY_NOT_OAI_FREE',
             registration_sha256='digest', tool_sha256='digest', members=list(h.MEMBERS), cuda_smoke_sha256='smoke-digest')
    p = tmp_path / 'review.json'
    p.write_text(json.dumps(v))
    assert h.admit(p)['status'] == 'PASS_AVAILABLE_MEMBERS_ONLY'
    for key, wrong in [('independent', False), ('tool_sha256', 'changed'),
                       ('registration_sha256', 'changed'), ('scope', 'OAI_FREE'), ('members', ['R']), ('cuda_smoke_sha256', 'old')]:
        p.write_text(json.dumps({**v, key: wrong}))
        with pytest.raises(ValueError, match='Independent review'):
            h.admit(p)
    monkeypatch.setattr(h, 'smoke_binding', lambda: None)
    with pytest.raises(ValueError, match='CUDA smoke required'):
        h.admit(p)


def test_prepared_volume_tamper_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'SCRATCH', tmp_path)
    (tmp_path / 'prepared').mkdir()
    registration = tmp_path / 'registration.json'
    registration.write_text('{}')
    p = tmp_path / 'prepared/id.npz'
    np.savez(p, vol=np.zeros((2, 2), np.uint8), mask=np.ones(2))
    h.write(str(p)+'.json', dict(status='PASS', uid='id', registration_sha256=h.sha(registration), sha256=h.sha(p)))
    h.volume('id')
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='no longer authentic'):
        h.volume('id')


def test_partial_predictions_rejected_before_score(tmp_path):
    p = tmp_path / 'R.csv'
    h.write(str(p)+'.json', dict(full_panel=False))
    with pytest.raises(ValueError, match='Partial/changed'):
        h.read_prediction(p, ['one', 'two'])


def test_no_output_outside_new_task_folders(tmp_path):
    with pytest.raises(ValueError, match='Output outside'):
        h.output_path(tmp_path / 'outside.csv')
    assert h.output_path(h.SCRATCH / 'test.csv').parent == h.SCRATCH


def test_prediction_roundtrip_full_score_and_immutable_output(tmp_path, monkeypatch):
    # End-to-end artifact checks use synthetic studies/probabilities/outcomes only.
    monkeypatch.setattr(h, 'SCRATCH', tmp_path / 'scratch')
    monkeypatch.setattr(h, 'PANEL', tmp_path / 'panel')
    monkeypatch.setattr(h, 'FULL', tmp_path / 'panel/full/new')
    ids = [f'synthetic-{i:04}' for i in range(557)]
    monkeypatch.setattr(h, 'ids_only', lambda: ids)
    monkeypatch.setattr(h, 'validate', lambda: {'ids': ids})
    for p in (h.SCRATCH / 'registration.json', h.PANEL / 'registration.json', h.PANEL / 'PROTOCOL.md'):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
    bindings = {}
    for uid in ids:
        p = h.SCRATCH / 'prepared' / (uid + '.npz.json')
        h.write(p, {'synthetic': True})
        bindings[uid] = h.sha(p)
    values = np.tile(np.linspace(.01, .99, len(ids))[:, None], (1, 12))
    y = np.tile((np.arange(len(ids)) >= 280).astype(float)[:, None], (1, 12))
    y[:, [1, 10, 11]] = np.nan
    scorer = h.panel_module()
    monkeypatch.setattr(scorer, 'outcomes', lambda _: y)
    for name, (_, digest) in h.MEMBERS.items():
        extra = dict(device='cuda', checkpoint_sha256=digest, input_receipts=bindings)
        h.write_predictions(h.FULL, name, ids, values, extra)
        np.testing.assert_array_equal(h.read_prediction(h.FULL / (name + '.csv'), ids), values)
        with pytest.raises(ValueError, match='full output exists'):
            h.write_predictions(h.FULL, name, ids, values, extra)
    assert h.score()['studies'] == 557
    analysis = h.read(h.FULL / 'analysis.json')
    assert analysis['members']['R']['macro9']['auc'] == 1
    with pytest.raises(ValueError, match='Analysis exists'):
        h.score()
    p = h.FULL / 'R.csv.json'
    r = h.read(p)
    r['checkpoint_sha256'] = 'wrong'
    h.write(p, r)
    with pytest.raises(ValueError, match='member identity'):
        h.read_prediction(h.FULL / 'R.csv', ids)


def test_smoke_admission_rejects_missing_or_blocked_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'SCRATCH', tmp_path)
    assert h.smoke_binding() is None
    h.write(tmp_path / 'cuda-two/receipt.json', {'status': 'BLOCKED_GPU_NOT_IDLE'})
    assert h.smoke_binding() is None


def test_selected_background_slice_uses_frozen_content_mask():
    vol = np.ones((96, 384, 384), dtype=np.uint8)
    vol[7] = 0
    vol[26:48] = 0
    mask = (vol.reshape(96, -1).sum(1) > 0).astype(np.uint8)
    routes = [dict(series=s, slices=n) for s, n in zip(('s', None, 'c1', 'c0', 'a'), (26, 22, 18, 12, 18))]
    assert h.validate_volume(vol, mask, routes) == 1
    mask[7] = 1
    with pytest.raises(ValueError, match='content rule'):
        h.validate_volume(vol, mask, routes)


@pytest.mark.parametrize('missing_slot_has_pixels', [False, True])
def test_selected_empty_or_absent_nonzero_slot_rejected(missing_slot_has_pixels):
    vol = np.ones((96, 384, 384), dtype=np.uint8)
    if not missing_slot_has_pixels:
        vol[:48] = 0
    mask = (vol.reshape(96, -1).sum(1) > 0).astype(np.uint8)
    routes = [dict(series=s, slices=n) for s, n in zip(('s', None, 'c1', 'c0', 'a'), (26, 22, 18, 12, 18))]
    with pytest.raises(ValueError, match='decoded/absent'):
        h.validate_volume(vol, mask, routes)
