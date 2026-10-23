import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('abdiag', Path(__file__).resolve().parents[1]/'tools/input_ab_diagnostics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_tie_aware_auc_macro_contribution_and_target_population():
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 12))
    a = np.full((4, 12), .5)
    b = np.tile(np.array([0, .5, .5, 1])[:, None], (1, 12))
    rows = diag.target_pair(y, a, b)
    for row in rows:
        assert row['P'] == row['N'] == 2
        assert row['auc_a'] == .5 and row['auc_b'] == .875
        assert row['delta'] == .375
        assert row['macro_contribution'] == pytest.approx(.375/12)


def test_exact_half_weak_labels_round_to_even_and_one_class_remains_unknown():
    y = np.tile(np.array([0, .5, .5])[:, None], (1, 12))
    rows = diag.target_pair(y, np.zeros((3, 12)), np.ones((3, 12)))
    assert all(r['P'] == 0 and r['N'] == 3 and r['delta'] is None for r in rows)


def test_paired_group_bootstrap_identical_predictions_zero_swap_reverses_bounds():
    y = np.tile(np.array([0, 0, 1, 1, 0, 1])[:, None], (1, 12))
    groups = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    a = np.full(y.shape, .5); b = y*.5+.25
    zero = diag.target_intervals([(y, a, a, groups)]*2, reps=100)
    assert all(r['low'] == r['high'] == 0 and r['valid_reps'] > 0 for r in zero.values())
    forward = diag.target_intervals([(y, a, b, groups)]*2, reps=100)
    reverse = diag.target_intervals([(y, b, a, groups)]*2, reps=100)
    for t in diag.LAB:
        assert forward[t]['low'] == -reverse[t]['high']
        assert forward[t]['high'] == -reverse[t]['low']
