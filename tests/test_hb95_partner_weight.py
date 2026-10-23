"""H-B-95 v1.2.0: fine-grid, pooled-guard, numerical and isolation tests."""
import csv
import numpy as np
import pytest
from tools import hb95_partner_weight as h
from tools import exp115_vlm_eval as ev


def fixture_data(n=120):
    rng = np.random.default_rng(17)
    raw = {name: rng.random((n, 12)) for name in h.FILES}
    y = rng.integers(0, 2, (n, 12)).astype(float)
    refs = {name: (y.copy(), np.ones_like(y, bool), 1) for name in h.REFS}
    return raw, refs, [np.array([i, i+1]) for i in range(0, n, 2)]


def test_weighted_auc_matches_expanded_existing_evaluator_with_ties_and_masks():
    rng = np.random.default_rng(20)
    y = rng.integers(0, 2, 100).astype(float)
    s = rng.integers(0, 8, 100).astype(float)
    mask = rng.random(100) > .3
    weights = rng.integers(0, 4, (8, 100))
    got = h.weighted_auc(y, s, mask, 2, weights)
    expected = []
    for w in weights:
        ix = np.repeat(np.arange(100), w)
        expected.append(ev.auc_targets(y[ix, None], s[ix, None], mask[ix, None], 2)[0])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)


def test_auc_minimum_and_no_labels():
    w = np.ones((2, 4))
    w[1, 0] = 0
    got = h.weighted_auc(np.array([1, 0, 0, 0]), np.arange(4), np.ones(4, bool), 1, w)
    assert got[0] == 0 and np.isnan(got[1])
    assert np.isnan(h.weighted_auc(np.zeros(4), np.arange(4), np.zeros(4, bool), 1, w)).all()


def test_nested_components_disjoint_and_every_row_tested_once_per_repeat():
    _, _, groups = fixture_data()
    splits = h.nested_splits(groups)
    assert len(splits) == 10
    for rep in range(5):
        both = [s for s in splits if s[0] == rep]
        assert sorted(np.concatenate([s[3] for s in both])) == list(range(120))
    for _, _, inner, outer in splits:
        sets = [set(ix) for ix in [*inner, outer]]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert set.union(*sets) == set(range(120))
        for group in groups:
            assert sum(set(group).issubset(part) for part in sets) == 1


def test_bootstrap_keeps_components_together_and_reproduces_screen_draws():
    _, _, groups = fixture_data()
    weights, digest = h.component_weights(groups, 120, draws=8)
    for g in groups:
        np.testing.assert_array_equal(weights[:, g[0]], weights[:, g[1]])
    for b, draw in enumerate(ev.bootstrap_index(groups, 8, h.SEED)):
        np.testing.assert_array_equal(weights[b+1], np.bincount(draw, minlength=120))
    assert digest == h.component_weights(groups, 120, draws=8)[1]
    with pytest.raises(ValueError, match='partition'):
        h.component_weights(groups[:-1], 120)


def test_rank_formula_and_no_outside_partition_influence():
    raw, _, _ = fixture_data()
    ix = np.arange(50)
    actual = h.predictions(raw, ix)
    r = h.rank((ev.pct_rank(raw['R42'][ix])+ev.pct_rank(raw['R43'][ix]))/2)
    q = h.rank((r+h.rank(raw['C4'][ix])+h.rank(raw['D172'][ix]))/3)
    np.testing.assert_allclose(actual['C4'][2], .6*h.rank(raw['anchor'][ix])+.4*q)
    changed = {k: v.copy() for k, v in raw.items()}
    for a in changed.values():
        a[50:] = 0
    np.testing.assert_array_equal(h.predictions(changed, ix)['C4'][2], actual['C4'][2])


def test_partition_family_composition_matches_direct_auc_all_candidates():
    raw, refs, groups = fixture_data()
    ww, _ = h.component_weights(groups, 120, draws=2)
    ix = np.arange(60)
    score = h.score_partition(raw, refs, ix, ww)
    pred = h.predictions(raw, ix)
    for ci, c in enumerate(h.CANDIDATES):
        p = pred[c['q']][h.WEIGHTS.index(c['other'])].copy()
        p[:, h.WEAK] = pred[c['q']][h.WEIGHTS.index(c['weak'])][:, h.WEAK]
        for ri, ref in enumerate(h.REFS):
            y, m, minimum = refs[ref]
            expected = ev.auc_targets(y[ix], p, m[ix], minimum)
            np.testing.assert_allclose(score[0, ci, ri], expected, atol=1e-14)


def test_guards_target_specific_and_undefined_fail_closed():
    d = np.zeros((4, 4, 12))
    d[1, :2, h.OTHER[0]] = -.0101
    d[2, 2] = -.0031
    d[3, 0, 0] = np.nan
    assert h.measures(d)[3].tolist() == [True, False, False, False]
    d = np.zeros((4, 12))
    d[:2, h.WEAK] = -.002
    assert h.measures(d)[3]  # No unregistered -0.010 weak-target guard.
    d[:2] = -.0011
    assert not h.measures(d)[3]


def test_inner_selection_recomputed_per_draw_and_ties_prefer_baseline():
    base = h.baseline('C4')
    alternative = next(i for i, c in enumerate(h.CANDIDATES)
                       if c['q'] == 'C4' and c['weak'] == c['other'] == .4)
    inner = np.full((3, len(h.CANDIDATES), 4, 12), .7)
    inner[1, alternative, :2, h.WEAK] += .01
    inner[2, alternative, :2, h.WEAK] += .01
    inner[2, alternative, 2] -= .004
    assert h.select(inner, [base, alternative], base).tolist() == [base, alternative, base]
    inner[:] = np.nan
    assert h.select(inner, [base, alternative], base).tolist() == [base]*3


def test_grid_and_policy_contract():
    assert h.FINE_WEIGHTS == (.35, .40, .45, .50, .55, .60, .65, .70)
    assert len(h.CANDIDATES) == 18
    assert all(c['global_weight'] for c in h.CANDIDATES)
    assert set(h.policies()) == {'C4_global', 'MIL_global', 'joint_global'}
    for name, (allowed, base) in h.policies().items():
        assert base in allowed
        assert len(allowed) == (18 if name == 'joint_global' else 9)


def test_end_to_end_identity_is_zero_and_no_adaptive_gain():
    raw, refs, groups = fixture_data()
    for k in raw:
        raw[k] = raw['anchor'].copy()
    r = h.analyze(raw, refs, groups, draws=3)
    for row in r['policies'].values():
        assert row['primary_weak_four']['delta'] == 0
        assert row['primary_weak_four']['ci95'] == [0, 0]
        assert row['harm_guard']['passed']
    assert r['recommendation']['release_v6'].startswith('keep 0.15')
    assert not r['recommendation']['nested_numeric_support']
    assert not r['recommendation']['single_anchor_advance']
    for q in ('C4', 'MIL'):
        for metric in r['flat_regions'][q]['unconstrained'].values():
            assert metric['flat_weights'] == list(h.FINE_WEIGHTS)


def test_oof_loader_rejects_wrong_roster_without_printing_rows(tmp_path):
    p = tmp_path/'pred.csv'
    with p.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['StudyInstanceUID', *h.TARGETS])
        writer.writerow(['fake', *([.5]*12)])
    with pytest.raises(ValueError, match='held-out IDs differ'):
        h.load_oof(p, ['different'])


def test_create_only_output_guard_and_required_protocol(tmp_path):
    with pytest.raises(ValueError, match='PROTOCOL'):
        h.main(['--root', str(tmp_path)])
    out = tmp_path/h.OUT
    out.mkdir(parents=True)
    (out/'PROTOCOL.md').write_text('protocol')
    (out/'result.json').write_text('{}')
    with pytest.raises(ValueError, match='Create-only'):
        h.main(['--root', str(tmp_path)])


def test_pooled_guard_repairs_missing_mcl_without_changing_objective_or_supported_cells():
    shape = (2, len(h.CANDIDATES), 4, 12)
    a, b, pooled = [np.full(shape, .7) for _ in range(3)]
    base = h.baseline('C4')
    alt = next(i for i, c in enumerate(h.CANDIDATES)
               if c['q'] == 'MIL' and c['weak'] == c['other'] == .65)
    a[:, alt, :2, h.WEAK] += .008
    b[:, alt, :2, h.WEAK] += .008
    a[:, :, 1, h.TARGETS.index('MCL')] = np.nan
    pooled[:, :, 1, h.TARGETS.index('MCL')] = .85
    pooled[:, alt, :2, h.WEAK] = .1  # must NOT become the weak objective
    inner, guard, audit = h.repair_inner_guards([a, b], pooled)
    assert h.select(inner, [base, alt], base).tolist() == [base, base]
    assert h.select(inner, [base, alt], base, guard).tolist() == [alt, alt]
    np.testing.assert_array_equal(inner, (a+b)/2)
    np.testing.assert_array_equal(guard[:, :, 0], inner[:, :, 0])
    assert audit['point_repaired_cells'] == ['consensus:MCL']
    assert audit['point_unresolved_cells'] == 0
    pooled[1, :, 1, h.TARGETS.index('MCL')] = np.nan
    inner, guard, audit = h.repair_inner_guards([a, b], pooled)
    assert h.select(inner, [base, alt], base, guard).tolist() == [alt, base]
    assert audit['bootstrap_unresolved_cells'] == 1


def test_missing_weak_objective_stays_fail_closed_after_guard_repair():
    parts = [np.full((1, len(h.CANDIDATES), 4, 12), .7) for _ in range(2)]
    pooled = parts[0].copy()
    parts[0][:, :, 0, h.WEAK[0]] = np.nan
    inner, guard, _ = h.repair_inner_guards(parts, pooled)
    base = h.baseline('C4')
    assert not np.isfinite(h.selection_utility(inner, range(len(h.CANDIDATES)), base, guard)[1]).any()
    assert h.select(inner, range(len(h.CANDIDATES)), base, guard).tolist() == [base]


def test_pooled_inner_scores_ignore_outer_predictions_and_labels():
    raw, refs, groups = fixture_data()
    _, _, inner_indices, outer = h.nested_splits(groups)[0]
    ix = np.concatenate(inner_indices)
    ww, _ = h.component_weights(groups, 120, draws=2)
    before = h.score_partition(raw, refs, ix, ww)
    for values in raw.values():
        values[outer] = 100
    for y, mask, _ in refs.values():
        y[outer] = 1-y[outer]
        mask[outer] = False
    np.testing.assert_array_equal(h.score_partition(raw, refs, ix, ww), before)


def test_half_curve_matches_direct_scoring_and_artifact_contract(tmp_path):
    raw, refs, groups = fixture_data()
    result = h.analyze(raw, refs, groups, draws=2)
    weights, _ = h.component_weights(groups, 120, draws=2)
    for half, split in zip(result['robustness']['halves'], h.nested_splits(groups)[:2]):
        ix = split[3]
        assert half['studies'] == len(ix)
        assert half['components'] == len(ix)//2
        assert half['indices_sha256'] == h.digest_array(ix.astype('<i8'))
        scores = h.score_partition(raw, refs, ix, weights)
        expected = h.grid_summary(scores-scores[:, h.baseline('C4'), None])
        for got, want in zip(half['fixed_grid'], expected):
            for contrast in ('versus_C4_015', 'versus_own_015'):
                for target in h.TARGETS:
                    assert got[contrast]['mean_jev_consensus_targets'][target]['delta'] == pytest.approx(
                        want[contrast]['mean_jev_consensus_targets'][target]['delta'])
    assert sum(v['studies'] for v in result['robustness']['halves']) == 120
    h.write_curve_artifacts(result, tmp_path)
    with (tmp_path/'grid.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 54
    assert {row['view'] for row in rows} == {'split_average', 'half_0', 'half_1'}
    assert sum(row['reference_only'] == 'False' for row in rows) == 48
    for contrast in ('versus_C4_015', 'versus_own_015'):
        for t in h.TARGETS:
            assert f'{contrast}_mean_jev_consensus_{t}' in rows[0]
        assert f'{contrast}_harm_guard_passed' in rows[0]
    assert (tmp_path/'curve.png').stat().st_size > 1000


def test_flat_regions_include_boundary_exclude_baseline_and_apply_guards():
    grid = []
    for q in ('C4', 'MIL'):
        for w, v in ((.15, 1.), (.35, .01), (.40, .0097), (.45, .00969), (.50, .01)):
            grid.append(dict(q=q, weak=w, versus_C4_015={
                'primary_weak_four': {'delta': v},
                'macro_mean_jev_consensus': {'delta': v},
                'harm_guard': {'passed': w != .35}}))
    r = h.flat_regions(grid)
    for q in ('C4', 'MIL'):
        for metric in ('primary_weak_four', 'macro_mean_jev_consensus'):
            assert r[q]['unconstrained'][metric]['peak_weight'] == .35
            assert r[q]['unconstrained'][metric]['flat_weights'] == [.35, .40, .50]
            assert r[q]['guarded'][metric]['peak_weight'] == .50
            assert r[q]['guarded'][metric]['flat_weights'] == [.40, .50]
    for row in grid:
        row['versus_C4_015']['harm_guard']['passed'] = False
    assert h.flat_regions(grid)['MIL']['guarded']['primary_weak_four']['flat_weights'] == []


def test_split_average_matches_direct_outer_evaluations():
    raw, refs, groups = fixture_data()
    result = h.analyze(raw, refs, groups, draws=1)
    weights, _ = h.component_weights(groups, 120, draws=1)
    deltas = []
    for _, _, _, ix in h.nested_splits(groups):
        scores = h.score_partition(raw, refs, ix, weights)
        deltas.append(scores-scores[:, h.baseline('C4'), None])
    expected = h.grid_summary(np.mean(deltas, axis=0))
    for got, want in zip(result['fixed_grid'], expected):
        for metric in ('primary_weak_four', 'macro_mean_jev_consensus', 'dread_macro'):
            np.testing.assert_allclose(got['versus_C4_015'][metric]['delta'],
                                       want['versus_C4_015'][metric]['delta'])
