"""Synthetic-only checks; no project labels, checkpoints or prediction files."""
import copy
import json

import numpy as np
import pytest
from scipy.stats import rankdata

from tools import exp115_vlm_eval as ev
from tools import exp179_screen as screen


def fixture_data(n=24):
    rng = np.random.default_rng(617)
    y = np.tile(np.arange(n)[:, None] % 2, (1, 12)).astype(float)
    masks = np.ones_like(y, dtype=bool)
    refs = {name: (y.copy(), masks.copy(), 1) for name in screen.REFERENCES}
    # Independent reference directions exercise averaging across reference draws.
    refs['consensus'][0][4:8] = 1 - refs['consensus'][0][4:8]
    controls = {seed: rng.uniform(.1, .9, y.shape) for seed in screen.SEEDS}
    candidates = {seed: np.clip(p + (.4*y-.2), 0, 1)
                  for seed, p in controls.items()}
    rad = {seed: rng.uniform(0, 1, y.shape) for seed in screen.SEEDS}
    dino = rng.uniform(0, 1, y.shape)
    # Two-row components preserve both classes on every draw.
    groups = [np.array([i, i+1]) for i in range(0, n, 2)]
    draws = ev.bootstrap_index(groups, 7, 51)
    return candidates, controls, refs, draws, ev, rad, dino


def direct_delta(candidate, control, ref, index):
    y, mask, minimum = ref
    return (ev.macro(ev.auc_targets(y[index], candidate[index], mask[index], minimum)) -
            ev.macro(ev.auc_targets(y[index], control[index], mask[index], minimum)))


def test_seed_mean_auc_and_joint_draw_ci_match_explicit_calculation():
    args = fixture_data()
    candidates, controls, refs, draws, _, _, _ = args
    result = screen.score_pairs(*args)
    for ref in screen.REFERENCES:
        point = np.mean([direct_delta(candidates[s], controls[s], refs[ref],
                                     np.arange(len(candidates[s]))) for s in screen.SEEDS])
        delta_draws = [np.mean([direct_delta(candidates[s], controls[s], refs[ref], ix)
                               for s in screen.SEEDS]) for ix in draws]
        assert result['direct']['references'][ref]['delta'] == pytest.approx(point)
        assert result['direct']['references'][ref]['ci95'] == pytest.approx(
            np.percentile(delta_draws, [2.5, 97.5]))
        assert set(result['direct']['references'][ref]['per_target_delta']) == set(screen.TARGETS)
    joint = [np.mean([direct_delta(candidates[s], controls[s], refs[ref], ix)
                     for s in screen.SEEDS for ref in ('jev', 'consensus')]) for ix in draws]
    assert result['direct']['primary_mean_ci95'] == pytest.approx(np.percentile(joint, [2.5, 97.5]))
    assert result['bootstrap']['draws'] == len(draws)
    assert 'not AUC of seed-mean' in result['bootstrap']['estimator']
    json.dumps(result, allow_nan=False)


def test_does_not_score_average_seed_predictions():
    y = np.tile(np.array([0, 0, 1, 1.])[:, None], (1, 12))
    refs = {name: (y, np.ones_like(y, dtype=bool), 1) for name in screen.REFERENCES}
    candidates = {42: np.tile(np.array([.1, .2, .8, .9])[:, None], (1, 12)),
                  43: np.tile(np.array([1., .9, 0., .1])[:, None], (1, 12))}
    controls = {s: np.full_like(y, .5) for s in screen.SEEDS}
    result = screen.score_pairs(candidates, controls, refs, [np.arange(4)], ev,
                                controls, controls[42])
    assert result['direct']['primary_mean_delta'] == pytest.approx(0.)
    wrong = direct_delta((candidates[42]+candidates[43])/2,
                         controls[42], refs['jev'], np.arange(4))
    assert wrong == pytest.approx(-.5)
    assert result['direct']['primary_mean_delta'] != wrong


def test_fixed_bundle_matches_registered_exp172_family_rank_recipe():
    args = fixture_data()
    candidates, controls, refs, draws, _, rad, dino = args
    result = screen.score_pairs(*args)
    within = lambda p: rankdata(p, axis=0, method='average') / len(p)
    global_rank = lambda p: (rankdata(p, axis=0, method='average')-1) / (len(p)-1)
    radial = (within(rad[42]) + within(rad[43])) / 2
    bundle = lambda p: global_rank((global_rank(radial) + global_rank(within(dino)) +
                                   global_rank(within(p))) / 3)
    for ref in ('jev', 'aquila', 'dread', 'consensus'):
        expected = np.mean([direct_delta(bundle(candidates[s]), bundle(controls[s]),
                            refs[ref], np.arange(len(dino))) for s in screen.SEEDS])
        assert result['bundle_replacement']['references'][ref]['delta'] == pytest.approx(expected)
    joint = [np.mean([direct_delta(bundle(candidates[s]), bundle(controls[s]), refs[r], ix)
                     for s in screen.SEEDS for r in ('jev', 'consensus')]) for ix in draws]
    assert result['bundle_replacement']['primary_mean_ci95'] == pytest.approx(
        np.percentile(joint, [2.5, 97.5]))


def gate_pair(mean=.002, lower=.0001, dread=-.003):
    return {'primary_mean_delta': mean, 'primary_mean_ci95': [lower, .004],
            'references': {name: {'delta': dread if name == 'dread' else mean,
                'per_target_delta': dict.fromkeys(screen.TARGETS, mean)}
                for name in screen.REFERENCES}}


@pytest.mark.parametrize('mean,lower,dread,bundle,status', [
    (.002, .0001, -.003, .00001, 'ADVANCE'),
    (.001999999, .0001, -.003, .00001, 'HOLD'),
    (.002, 0., -.003, .00001, 'HOLD'),
    (.002, .0001, -.003000001, .00001, 'HOLD'),
    (.002, .0001, -.003, 0., 'HOLD'),
])
def test_frozen_gate_boundaries(mean, lower, dread, bundle, status):
    # Bundle interval crosses zero deliberately; positivity of its point alone
    # is the registered condition, not a secretly stronger interval threshold.
    result = screen.admission(gate_pair(mean, lower, dread), gate_pair(bundle, -.01))
    assert result['status'] == status


def test_missing_primary_target_prevents_advance():
    pair = gate_pair()
    pair['references']['jev']['per_target_delta']['Fracture'] = None
    assert screen.admission(pair, gate_pair())['status'] == 'HOLD'


def test_low_validity_jev_loss_flags_review_without_rewriting_numeric_gate():
    pair = gate_pair()
    pair['references']['jev']['per_target_delta']['Synovitis'] = -.02
    flags = screen._ruler_flags(pair, None)
    assert flags['status'] == 'REVIEW'
    assert flags['low_validity_jev_losses'] == [
        {'target': 'Synovitis', 'jev_delta': -.02, 'consensus_delta': .002}]
    assert screen.admission(pair, gate_pair())['status'] == 'ADVANCE'
    ruler_map = {'schema': 'evidence-rulers-v1', 'rows': [
        {'target': t, 'reference': 'jev_hybrid',
         'validity': 'LOW' if t == 'ACL' else 'HIGH'} for t in screen.TARGETS]}
    assert screen._ruler_flags(pair, ruler_map)['status'] == 'NO_LOW_VALIDITY_LOSS_FLAG'
    pair['references']['jev']['per_target_delta']['ACL'] = -.01
    assert screen._ruler_flags(pair, ruler_map)['status'] == 'REVIEW'


def test_low_validity_status_is_review_for_whole_result():
    args = list(fixture_data())
    controls = args[1]
    args[0] = {s: p.copy() for s, p in controls.items()}
    column = screen.TARGETS.index('Synovitis')
    y = args[2]['jev'][0]
    for seed in screen.SEEDS:
        args[0][seed][:, column] = 1-y[:, column]
        controls[seed][:, column] = y[:, column]
    result = screen.score_pairs(*args)
    assert result['status'] == 'REVIEW'
    assert result['registered_gate']['status'] == 'HOLD'
    assert result['ruler_advisory']['direct']['status'] == 'REVIEW'


def test_aquila_is_report_only_and_masked_nan_stays_unknown():
    args = list(fixture_data())
    y, mask, minimum = args[2]['aquila']
    y[:] = np.nan
    mask[:] = False
    result = screen.score_pairs(*args)
    assert result['direct']['references']['aquila']['delta'] is None
    assert all(x is None for x in result['direct']['references']['aquila']['per_target_delta'].values())
    assert result['aquila_policy'].endswith('never a gate')
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('defect', ['missing_seed', 'extra_seed', 'nan_prediction',
    'out_of_range', 'wrong_shape', 'duplicate_seed_type', 'nonbinary_label',
    'integer_mask', 'negative_draw', 'float_draw', 'empty_draw', 'empty_draws',
    'missing_reference'])
def test_malformed_inputs_fail_closed(defect):
    args = list(fixture_data())
    if defect == 'missing_seed':
        del args[0][43]
    elif defect == 'extra_seed':
        args[0][44] = args[0][42]
    elif defect == 'nan_prediction':
        args[0][42][0, 0] = np.nan
    elif defect == 'out_of_range':
        args[0][42][0, 0] = 1.01
    elif defect == 'wrong_shape':
        args[1][42] = args[1][42][:-1]
    elif defect == 'duplicate_seed_type':
        args[0]['42'] = args[0].pop(42)
    elif defect == 'nonbinary_label':
        args[2]['jev'][0][0, 0] = .2
    elif defect == 'integer_mask':
        y, mask, minimum = args[2]['jev']
        args[2]['jev'] = y, mask.astype(int), minimum
    elif defect == 'negative_draw':
        args[3] = [np.array([-1, 0])]
    elif defect == 'float_draw':
        args[3] = [np.array([0., 1.])]
    elif defect == 'empty_draw':
        args[3] = [np.array([], dtype=int)]
    elif defect == 'empty_draws':
        args[3] = []
    elif defect == 'missing_reference':
        del args[2]['consensus']
    with pytest.raises(ValueError):
        screen.score_pairs(*args)


def test_empty_primary_reference_is_not_silently_dropped():
    args = list(fixture_data())
    args[2]['consensus'][1][:] = False
    with pytest.raises(ValueError, match='primary reference AUC/bootstrap undefined'):
        screen.score_pairs(*args)


def test_array_api_has_no_mutation_and_preserves_verification_note():
    args = fixture_data()
    before = {i: copy.deepcopy(args[i]) for i in (0, 1, 5)}
    result = screen.score_pairs(*args, verification_note='verification/exp179-PASS.md')
    for collection_index in (0, 1, 5):
        for key in args[collection_index]:
            np.testing.assert_array_equal(args[collection_index][key], before[collection_index][key])
    assert result['verification_note'] == 'verification/exp179-PASS.md'
    assert result['tool_version'] == screen.VERSION
