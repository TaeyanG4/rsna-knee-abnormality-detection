"""Synthetic CPU checks; no real labels, checkpoints or model forwards."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import rankdata

from tools import hb121_state_stackers_v1 as h


def fixture():
    # Two rows/component, mixed classes within every component. None cross CV.
    n = 80
    components = tuple('c'+str(i//2) for i in range(n))
    rng = np.random.default_rng(121)
    p = rng.uniform(.01, .99, (n, 3, 12)).astype(np.float32)
    y = np.tile(np.arange(n)[:, None] % 2, (1, 12)).astype(float)
    mask = np.ones_like(y, bool)
    refs = {r: (y.copy(), mask.copy(), 1) for r in h.readout.RULERS}
    source = h.Ancestry(frozenset({'external-training-component'}), True, 'a'*64, True)
    def block(rows):
        return h.FeatureBlock(tuple(int(i) for i in rows), p[list(rows)].copy(),
                              ('synthetic',)*len(rows), {'synthetic': source})
    folds = []
    all_rows = np.arange(n)
    for outer in range(5):
        train = all_rows[(all_rows//2) % 5 != outer]
        test = all_rows[(all_rows//2) % 5 == outer]
        inner = []
        for k in range(2):
            a = train[(train//2) % 2 != k]
            b = train[(train//2) % 2 == k]
            inner.append(h.Split(block(a), block(b)))
        folds.append(h.OuterFold(block(train), block(test), tuple(inner)))
    return folds, components, refs


def changed_source(block, **kwargs):
    return replace(block, ancestry={k: replace(v, **kwargs) for k, v in block.ancestry.items()})


def test_probability_first_family_mean_and_rank_ties():
    rng = np.random.default_rng(51)
    m = {k: rng.integers(0, 4, (2 if k != 'C4' else 1, 10, 12)).astype(np.float32)/4
         for k in h.FAMILIES}
    out = h.probability_means(m)
    np.testing.assert_array_equal(out, np.stack([m[k].mean(0, dtype=np.float32) for k in h.FAMILIES], 1))
    groups = ('fold0',)*4 + ('fold1',)*6
    expected = np.concatenate([(rankdata(a, axis=0)-1)/(len(a)-1) for a in (out[:4], out[4:])])
    np.testing.assert_array_equal(h.same_fold_rank(out, groups), expected)
    assert not np.array_equal(expected, (rankdata(out, axis=0)-1)/9)
    changed = out.copy(); changed[4:] = 1
    np.testing.assert_array_equal(h.same_fold_rank(changed, groups)[:4], expected[:4])
    np.testing.assert_array_equal(h.same_fold_rank(out[:1], ('one',)), np.full((1, 3, 12), .5))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.01, 1.01])
def test_bad_probabilities(bad):
    folds, c, _ = fixture()
    folds[0].train.probabilities[0, 0, 0] = bad
    with pytest.raises(ValueError, match='probabilities'):
        h.validate_nested(folds, c)


@pytest.mark.parametrize('level', ['outer_train', 'outer_test', 'inner_train', 'inner_test'])
def test_ancestral_leak_before_any_fit(level, monkeypatch):
    folds, c, refs = fixture()
    f = folds[0]
    if level.startswith('outer'):
        side = level.split('_')[1]
        source = c[f.test.rows[0]]
        folds[0] = replace(f, **{side: changed_source(getattr(f, side), exposed_components=frozenset({source}))})
    else:
        side = level.split('_')[1]
        inner = f.inner[0]
        source = c[inner.test.rows[0]]
        split = replace(inner, **{side: changed_source(getattr(inner, side), exposed_components=frozenset({source}))})
        folds[0] = replace(f, inner=(split, f.inner[1]))
    def forbidden(*args, **kwargs):
        pytest.fail('ancestry gate must run before optimizer')
    monkeypatch.setattr(h, 'fit_state', forbidden)
    with pytest.raises(ValueError, match='ancestral component exposure'):
        h.nested_crossfit(folds, c, refs)


def test_non_nested_stitched_oof_rejected_despite_each_row_heldout():
    folds, components, _ = fixture()
    f = folds[0]
    # Standard OOF: each source excludes its own fold but sees outer fold0.
    groups = tuple(str((i//2) % 5) for i in f.train.rows)
    sources = {g: h.Ancestry(frozenset(c for i,c in enumerate(components)
                                     if str((i//2) % 5) != g), True, 'b'*64)
               for g in set(groups)}
    block = replace(f.train, source_groups=groups, ancestry=sources)
    h.block_check(block, components)  # ordinary row-level OOF exclusion passes
    with pytest.raises(ValueError, match='ancestral'):
        h.block_check(block, components, {components[i] for i in f.test.rows})


@pytest.mark.parametrize('kwargs', [{'complete': False}, {'evidence_sha256': 'unknown'}])
def test_unknown_ancestry_fails_closed(kwargs):
    f, c, _ = fixture()
    f[0] = replace(f[0], train=changed_source(f[0].train, **kwargs))
    with pytest.raises(ValueError, match='incomplete transitive ancestry'):
        h.validate_nested(f, c)


def test_component_crossing_missing_duplicate_or_inner_coverage():
    f, c, _ = fixture()
    cc = list(c); cc[f[0].train.rows[0]] = cc[f[0].test.rows[0]]
    with pytest.raises(ValueError, match='component crosses'):
        h.validate_nested(f, cc)
    for bad in (f[:4], f[:4]+[f[0]]):
        with pytest.raises(ValueError):
            h.validate_nested(bad, c)
    f[0] = replace(f[0], inner=(f[0].inner[0], f[0].inner[0]))
    with pytest.raises(ValueError, match='inner test coverage'):
        h.validate_nested(f, c)


def test_weight_grid_simplex_equal_first_and_small_maps():
    grid = h.weight_grid()
    np.testing.assert_array_equal(grid[0], np.full(3, 1/3))
    np.testing.assert_allclose(grid.sum(1), 1)
    assert len(grid) == len(np.unique(np.round(grid, 14), axis=0)) == 25
    assert (grid > 0).all()
    for features in h.STATES.values():
        assert len(features)+1 <= 4


def test_logistic_fit_uses_only_mask_and_finite_coefficients():
    rng = np.random.default_rng(2)
    x = rng.random((40, 3)); y = (x[:, 0] > .5).astype(float)
    mask = np.ones(40, bool); mask[:4] = False
    beta = h.fit_state(x, y, mask, 1.)
    y[:4] = np.nan
    np.testing.assert_array_equal(beta, h.fit_state(x, y, mask, 1.))
    assert len(beta) == 4 and np.isfinite(beta).all()
    with pytest.raises(ValueError, match='both classes'):
        h.fit_state(x, np.ones(40), mask, 1.)


@pytest.fixture(scope='module')
def evaluated():
    folds, c, refs = fixture()
    return folds, c, refs, h.nested_crossfit(folds, c, refs)


def test_actual_nested_api_finite_deterministic_unchanged_targets(evaluated):
    folds, c, refs, out = evaluated
    again = h.nested_crossfit(folds, c, refs)
    assert out['outer_constants'] == again['outer_constants']
    assert out['ship_constants'] is None
    for name, a in out['arms'].items():
        np.testing.assert_array_equal(a, again['arms'][name])
        assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
        affected = {'Synovitis': [8], 'Lateral OA': [5], 'state_both': [5, 8], 'family_weights': range(12)}[name]
        ix = [j for j in range(12) if j not in affected]
        np.testing.assert_array_equal(a[:, ix], out['control'][:, ix])
    for constants in out['outer_constants']:
        assert constants['state']['Synovitis']['parameters'] == 3
        assert constants['state']['Lateral OA']['parameters'] == 4
    for f in folds:
        np.testing.assert_array_equal(out['control'][list(f.test.rows)], h.features(f.test)[1])


def test_outer_labels_never_select_outer_constants(evaluated):
    folds, c, refs, out = evaluated
    refs = {r: (y.copy(), m.copy(), n) for r, (y,m,n) in refs.items()}
    for y, _, _ in refs.values():
        y[list(folds[0].test.rows)] = 1 - y[list(folds[0].test.rows)]
    other = h.nested_crossfit(folds, c, refs)
    assert other['outer_constants'][0] == out['outer_constants'][0]
    for name in out['arms']:
        np.testing.assert_array_equal(other['arms'][name][list(folds[0].test.rows)],
                                      out['arms'][name][list(folds[0].test.rows)])


def test_missing_class_blocks_and_nonfinite_masked_data_ignored():
    f, c, refs = fixture()
    refs['jev'][1][:, 8] = False
    with pytest.raises(ValueError, match='fixed class support'):
        h.nested_crossfit(f, c, refs)


def test_seven_lenses_uses_corrected_api_and_keeps_conditional_cis(evaluated):
    _, c, refs, out = evaluated
    ids = ['synthetic-'+str(i) for i in range(len(c))]
    comp = [dict(component_id=g, members=[ids[i] for i,v in enumerate(c) if v==g]) for g in dict.fromkeys(c)]
    scores = h.seven_lenses(out, ids, refs, comp)
    ctx = h.base.Context(ids, refs, comp)
    for name in out['arms']:
        exact = h.readout.evaluate(ctx, {'nested_5x2': out['arms'][name]}, {'nested_5x2': out['control']})
        assert scores[name] == exact
        assert exact['version'] == '2.0.0' and set(exact['lenses']) == {'L'+str(i) for i in range(1,8)}
        assert not exact['lenses']['L5']['eligible']  # folds are not independent seeds
        assert not exact['lenses']['L6']['eligible']
        assert not exact['lenses']['L7']['eligible']
        assert exact['lenses']['L2']['valid_draws'] == 300
        assert exact['lenses']['L4']['legacy_telemetry_note'].find('NOT') >= 0


@pytest.mark.parametrize('complete,oai', [(True,None), (True,False), (False,True)])
def test_oai_unknown_ancestry_rejected_gold_always_diagnostic(complete,oai):
    ids=['a','b']; p=np.zeros((2,12)); y=p.copy(); m=np.ones_like(y,bool)
    ancestry=[h.Ancestry(frozenset(),complete,'d'*64,oai)]
    components=[dict(component_id=u,members=[u]) for u in ids]
    with pytest.raises(ValueError,match='OAI-free'):
        h.image_read_entry(ids,p,p,y,m,components,ancestry)
    assert not h.image_read_entry(ids,p,p,y,m,components,ancestry,gold58=True)['eligible']


def test_proven_oai_entry_eligible():
    ids=['a','b']; p=np.zeros((2,12)); m=np.ones_like(p,bool)
    components=[dict(component_id=u,members=[u]) for u in ids]
    source=h.Ancestry(frozenset(),True,'d'*64,True)
    assert h.image_read_entry(ids,p,p,p,m,components,[source])['eligible']


def test_oai_exposed_component_rejected_despite_free_claim():
    ids=['a','b']; p=np.zeros((2,12)); m=np.ones_like(p,bool)
    components=[dict(component_id='patient-'+u,members=[u]) for u in ids]
    source=h.Ancestry(frozenset({'patient-a'}),True,'d'*64,True)
    with pytest.raises(ValueError,match='OAI-free'):
        h.image_read_entry(ids,p,p,p,m,components,[source])
    assert not h.image_read_entry(ids,p,p,p,m,components,[source],gold58=True)['eligible']


@pytest.mark.parametrize('components', [[], [{'component_id':'a','members':['a']}],
    [{'component_id':'a','members':['a','b']}, {'component_id':'b','members':['b']}]])
def test_image_read_component_coverage_required_even_for_diagnostic(components):
    ids=['a','b']; p=np.zeros((2,12)); m=np.ones_like(p,bool)
    source=h.Ancestry(frozenset(),True,'d'*64,True)
    with pytest.raises(ValueError,match='component coverage'):
        h.image_read_entry(ids,p,p,p,m,components,[source],gold58=True)


def test_proven_direct_exposure_precedes_unknown_ancestry():
    folds,c,_=fixture()
    f=folds[0]
    block=changed_source(f.train,complete=False,exposed_components=frozenset({c[f.test.rows[0]]}))
    with pytest.raises(ValueError,match='ancestral component exposure'):
        h.block_check(block,c,{c[i] for i in f.test.rows})
