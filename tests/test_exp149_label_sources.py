"""Small CPU-only contract tests for EXP149; no CUDA or training imports."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from tools import exp149_label_sources as exp
from training.runners.exp149_label_sources import move_batch


def _aquila_csv(root, ids, *, bad=None):
    file = root / exp.AQUILA
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([exp.UID, *exp.T])
        for uid in ids:
            values = [.25] * 12
            if uid == bad:
                values[0] = .01
            writer.writerow([uid, *values])


def test_nested_teacher_excludes_outer_and_own_fold():
    folds = {0: (1, 2, 3, 4), 1: (0, 2, 3, 4)}
    for outer, training in folds.items():
        for study_fold in training:
            teacher = exp.teacher_for(outer, study_fold)
            train, held_out = exp.TEACHERS[teacher]
            assert outer not in train
            assert study_fold not in train
            assert study_fold in held_out
    with pytest.raises(ValueError):
        exp.teacher_for(0, 0)


def test_ordered_roster_preserves_canonical_and_disjointness():
    ids = ['a', 'b', 'c', 'd', 'e']
    fold_of = dict(zip(ids, range(5)))
    assert exp.ordered_for(ids, fold_of, (2, 3, 4), (0, 1)) == {
        'train': ['c', 'd', 'e'], 'val': ['a', 'b']}
    with pytest.raises(ValueError):
        exp.ordered_for(ids, fold_of, (0, 2), (0, 1))


def test_l3_requires_a_teacher_value_for_every_training_study():
    ids = ['f0', 'f1', 'f2', 'f3', 'f4']
    fold_of = dict(zip(ids, range(5)))
    dread = {uid: np.full(12, .4, np.float32) for uid in ids}
    aquila = {uid: np.full(12, .8, np.float32) for uid in ids}
    teachers = {'T234': {'f1': .91}, 'T134': {'f2': .92},
                'T124': {'f3': .93}, 'T123': {'f4': .94}}
    rows = exp.l3_rows(ids, fold_of, dread, aquila, teachers, 0)
    assert rows['f0'][exp.EFFUSION] == pytest.approx(.5)
    assert rows['f1'][exp.EFFUSION] == pytest.approx(.91)
    assert rows['f4'][exp.EFFUSION] == pytest.approx(.94)
    del teachers['T124']['f3']
    with pytest.raises(ValueError, match='missing'):
        exp.l3_rows(ids, fold_of, dread, aquila, teachers, 0)


def test_aquila_fails_closed_on_coverage_clip_and_gold(tmp_path):
    _aquila_csv(tmp_path, ['u0', 'u1', 'extra'])
    labels, rec = exp.strict_aquila(tmp_path, ['u0', 'u1'], [])
    assert set(labels) == {'u0', 'u1', 'extra'}
    assert rec['sha256']
    with pytest.raises(ValueError, match='Gold58'):
        exp.strict_aquila(tmp_path, ['u0', 'u1'], ['u1'])
    with pytest.raises(ValueError, match='cover every'):
        exp.strict_aquila(tmp_path, ['u0', 'u1', 'u2'], [])
    (tmp_path / exp.AQUILA).unlink()
    _aquila_csv(tmp_path, ['u0', 'u1'], bad='u1')
    with pytest.raises(ValueError, match='clipped'):
        exp.strict_aquila(tmp_path, ['u0', 'u1'], [])


def test_paired_bootstrap_is_shared_and_reports_target_delta():
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 12))
    good = np.tile(np.array([.1, .2, .8, .9])[:, None], (1, 12))
    bad = 1 - good
    ref = (y, np.ones_like(y, dtype=bool), 1)
    draws = [np.arange(4)] * 300
    result = exp.paired_metric(good, bad, ref, draws)
    assert result['delta'] == pytest.approx(1.0)
    assert result['ci95'] == pytest.approx([1.0, 1.0])
    assert result['per_target_delta']['Effusion'] == pytest.approx(1.0)
    assert result['bootstrap_draws'] == 300


class _FakeTensor:
    def __init__(self):
        self.moved = False

    def to(self, device, non_blocking=False):
        assert device == 'cpu' and non_blocking
        self.moved = True
        return self


@pytest.mark.parametrize('container', [list, tuple])
def test_batch_transfer_accepts_pin_memory_list_or_tuple(container):
    x, y = _FakeTensor(), _FakeTensor()
    got = move_batch(container([x, y, ['uid']]), 'cpu')
    assert got == (x, y, ['uid'])
    assert x.moved and y.moved
