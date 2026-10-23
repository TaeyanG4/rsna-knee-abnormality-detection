"""Tiny CPU screen contracts; no CUDA, training, or network."""
import numpy as np
import pytest
from tools import exp157_l2_screen as exp


def metrics(j=.001, low=-.0009, aquila=0., dread=-.003):
    return {'jev_hybrid': {'delta': j, 'ci95': [low, .002]},
            'aquila_v1': {'delta': aquila}, 'dread': {'delta': dread}}


def test_preregistered_strict_and_inclusive_boundaries():
    assert exp.decision(metrics())['status'] == 'ADOPT'
    for bad in (metrics(j=0), metrics(low=-.001), metrics(aquila=-1e-12), metrics(dread=-.0030001)):
        assert exp.decision(bad)['status'] == 'HOLD'


def test_each_seed_ranked_before_mean_with_average_ties():
    a = np.tile([[0.], [0.], [1.]], (1, 12))
    b = np.tile([[100.], [10.], [0.]], (1, 12))
    expected = np.tile([[.75], [7/12], [2/3]], (1, 12))
    np.testing.assert_allclose(exp.seed_mean([a, b]), expected)
    np.testing.assert_allclose(exp.seed_mean([a, b * 100]), expected)


def test_invalid_seed_predictions_fail_closed():
    for values in ([np.zeros((3, 12))], [np.zeros((3, 12)), np.zeros((2, 12))],
                   [np.zeros((3, 12)), np.full((3, 12), np.nan)]):
        with pytest.raises(ValueError):
            exp.seed_mean(values)


def test_control_tags_cover_matched_seeds_and_folds():
    assert exp.control_tag(43, 0) == 'EXP127_AUG1_S43_F0'
    assert exp.control_tag(42, 4) == 'EXP134_AUG1_S42_F4'
