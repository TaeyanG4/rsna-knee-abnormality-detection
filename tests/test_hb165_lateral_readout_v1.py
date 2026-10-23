"""Synthetic CPU coverage of the fixed Lateral-only follow-up v1.0.0."""
import numpy as np
import pytest
from tools import hb165_lateral_readout_v1 as r


def test_route_preserves_medial_and_ten_other_columns():
    rng = np.random.default_rng(9)
    c, x = rng.random((20, 12)), rng.random((20, 12))
    out = r.route(c, x)
    assert np.array_equal(out[:, r.LAT], x[:, r.LAT])
    assert np.array_equal(out[:, r.OTHER], c[:, r.OTHER])
    assert np.array_equal(out[:, r.TARGETS.index('Medial OA')], c[:, r.TARGETS.index('Medial OA')])
    assert not np.shares_memory(out, c)


@pytest.mark.parametrize('bad', [np.zeros((3, 11)), np.full((3, 12), np.nan),
                                     np.full((3, 12), 1.01), np.full((3, 12), -.01)])
def test_route_rejects_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        r.route(np.zeros((3, 12)), bad)


def test_structural_zeros_allow_only_identical_nan_masks():
    c = np.full((8, 4, 12), .5); c[2, 0, 0] = np.nan
    a = c.copy(); a[..., r.LAT] += .012
    d = r.structural(a, c)
    assert np.all(d[..., r.OTHER] == 0)
    assert r.summarize_delta(d)['macro_mean']['delta'] == pytest.approx(.001)
    a[2, 0, 0] = .5
    with pytest.raises(ValueError, match='unchanged AUC'):
        r.structural(a, c)


def test_ci_never_drops_invalid_draws():
    value = r.stats([.01, .02, np.nan, .03])
    assert value['ci95'] is None and value['invalid_draws'] == 1
    assert r.stats([.01])['ci95'] is None
    assert r.stats([.01, .02, .03])['ci95'] == pytest.approx([.02025, .02975])


@pytest.mark.parametrize('change', ['fold', 'target', 'train_duplicate', 'val_duplicate', 'overlap'])
def test_config_identity_and_disjointness(change):
    c = dict(fold=2, recipe=dict(targets=list(r.TARGETS)), train=['a', 'b'], val=['c', 'd'])
    r.check_config(c, 2)
    if change == 'fold': c['fold'] = 3
    if change == 'target': c['recipe']['targets'] = list(reversed(r.TARGETS))
    if change == 'train_duplicate': c['train'] = ['a', 'a']
    if change == 'val_duplicate': c['val'] = ['c', 'c']
    if change == 'overlap': c['train'] = ['a', 'c']
    with pytest.raises(ValueError): r.check_config(c, 2)


@pytest.mark.parametrize('change', ['verdict', 'independent', 'reviewer', 'bindings'])
def test_review_fails_closed(monkeypatch, change):
    expected = {'tool': '123'}
    obj = dict(verdict='PASS', independent=True, reviewer='independent-lane', bindings=expected)
    monkeypatch.setattr(r, 'bindings', lambda: expected)
    monkeypatch.setattr(r.old.b, 'read', lambda _: obj)
    r.require_review('synthetic')
    obj[change] = dict(verdict='FAIL', independent=False, reviewer='/root', bindings={'tool': 'old'})[change]
    with pytest.raises(ValueError, match='pre-review'): r.require_review('synthetic')


def test_routed_context_rejects_nonlateral_prediction_change():
    class Context:
        def score(self, p): return np.broadcast_to(p.mean(0), (3, 4, 12)).copy()
    c = np.full((5, 12), .4); a = c.copy(); a[:, r.LAT] = .7
    ctx = r.RoutedContext(Context())
    assert np.all(ctx.delta(a, c)[..., r.OTHER] == 0)
    a[0, r.TARGETS.index('Medial OA')] += .1
    with pytest.raises(ValueError, match='non-Lateral'): ctx.delta(a, c)


def test_fold2_l6_reuses_pooled_replace_recipe(monkeypatch):
    called = []
    def aligned(fold, ids, arms, controls, variant):
        called.append((fold, ids, arms, controls, variant))
        return ('a', 'c'), {'bindings': {'source': 'hash'}}
    monkeypatch.setattr(r.completion, 'aligned_bundle', aligned)
    bundle, prov = r.make_bundle(2, 'ctx', ['id'], {'42': 1}, {'42': 2})
    assert called == [(2, ['id'], {'42': 1}, {'42': 2}, 'replace')]
    assert bundle['arm'] == 'a' and bundle['context'] == 'ctx'
    assert prov == {'source': 'hash'}


def test_fold3_l6_reuses_original_partner_bundle(monkeypatch):
    monkeypatch.setattr(r.old, 'load_partners', lambda ids, root: ('partners', {'source': 'hash'}))
    seen = []
    def bundle(ctx, partners, arm, control):
        seen.append((ctx, partners, arm, control)); return {'original': True}
    monkeypatch.setattr(r.old, 'bundle', bundle)
    out, prov = r.make_bundle(3, 'ctx', ['id'], {'44': np.array([.2]), '45': np.array([.6])},
                            {'44': np.array([.1]), '45': np.array([.3])})
    assert out == {'original': True} and prov == {'source': 'hash'}
    assert seen[0][:2] == ('ctx', 'partners')
    assert seen[0][2] == pytest.approx([.4]) and seen[0][3] == pytest.approx([.2])
