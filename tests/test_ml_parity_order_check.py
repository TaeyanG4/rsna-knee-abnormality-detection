"""Synthetic CPU only: no real cohort, CUDA, model, remote or DICOM reads."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ml_parity_order_check', ROOT / 'tools/ml_parity_order_check.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_identical_predictions_have_no_order_changes():
    p = np.random.default_rng(0).random((20, 12))
    rep = m.compare(p, p.copy())
    assert rep['max_abs'] == 0 and rep['max_order_change_fraction'] == 0
    assert rep['pairs_per_target'] == 190


def test_single_swap_counts_one_pair_and_moves_auc_only_when_labels_split_it():
    p = np.tile(np.linspace(0.1, 0.9, 6)[:, None], (1, 12))
    q = p.copy(); q[[2, 3], 0] = q[[3, 2], 0]          # swap adjacent studies in ACL only
    changes, pairs = m.order_changes(p, q)
    assert changes == [1] + [0] * 11 and pairs == 15
    y = np.zeros((6, 12)); y[3:, :] = 1                 # boundary between rows 2 and 3 -> AUC must drop
    rep = m.compare(p, q, y)
    assert rep['macro_auc_ref'] == 1.0 and rep['macro_auc_cand'] < 1.0
    y2 = np.zeros((6, 12)); y2[4:, :] = 1               # swap inside the negatives -> AUC unchanged
    assert m.compare(p, q, y2)['macro_auc_delta'] == 0.0


def test_unlabelled_rows_are_skipped():
    p = np.random.default_rng(1).random((10, 12))
    y = np.full((10, 12), np.nan)
    y[:6] = np.where(np.arange(6)[:, None] % 2 == 0, 0.0, 1.0) * np.ones((6, 12))
    rep = m.compare(p, p, y)
    assert rep['targets_scored'] == 12 and rep['macro_auc_delta'] == 0.0
