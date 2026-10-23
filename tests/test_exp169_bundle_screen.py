"""CPU contract checks for the preregistered EXP169 screen."""
import csv
from pathlib import Path

import numpy as np
import pytest

from tools import exp169_bundle_screen as screen


def test_readiness_fails_until_exp167_collected(tmp_path):
    protocol = tmp_path / screen.PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_text('registered\n', encoding='utf-8')
    with pytest.raises(ValueError, match='EXP167 collection'):
        screen.readiness(tmp_path)


def test_exact_fold_paths_and_oof_policies():
    root = Path('workspace')
    first = screen.paths(root, 0)
    last = screen.paths(root, 4)
    assert 'EXP149_L2_S42_F0/oof_k36.csv' in first['l2_s42'].as_posix()
    assert 'EXP157_L2_S43_F4/oof_k36.csv' in last['l2_s43'].as_posix()
    assert '.local/exp160-collected-v1/BASE/oof_best.csv' in first['dino'].as_posix()
    assert 'EXP164_DINO_BASE_S42_F4/oof_best.csv' in last['dino'].as_posix()
    assert 'EXP166_CONVNEXT_T_S42_F0/oof_k36_flip_tta.csv' in first['convnext'].as_posix()
    assert 'EXP167_CONVNEXT_T_S42_F4/oof_k36_flip_tta.csv' in last['convnext'].as_posix()


def test_load_oof_reorders_only_complete_finite_target_table(tmp_path):
    from tools.exp162_dino_folds_kaggle import TARGETS, UID
    path = tmp_path / 'oof.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([UID, *TARGETS])
        writer.writerow(['b', *([0.8] * 12)])
        writer.writerow(['a', *([0.2] * 12)])
    value = screen.load_oof(path, ['a', 'b'])
    assert value.shape == (2, 12) and value[0, 0] == .2
    with pytest.raises(ValueError, match='held-out IDs'):
        screen.load_oof(path, ['a', 'c'])


def test_bundle_equal_thirds_and_rerank():
    a = np.tile(np.arange(1, 7)[:, None], (1, 12))
    b = a[::-1]
    bundle, ranked = screen.bundle_of({'l2': a, 'dino': a, 'convnext': b})
    assert list(ranked) == list(screen.FAMILIES)
    assert bundle.shape == (6, 12)
    assert bundle[0, 0] == 0 and bundle[-1, 0] == 1
    with pytest.raises(ValueError, match='Exactly three'):
        screen.bundle_of({'l2': a, 'dino': b})


def test_admission_uses_primary_paired_ci_and_dread_floor():
    pair = {'primary_mean_delta': .001, 'primary_mean_ci95': [.0001, .002],
            'references': {'dread': {'delta': -.003}}}
    assert screen.admission(pair)['status'] == 'ADMIT'
    pair['primary_mean_ci95'][0] = 0
    assert screen.admission(pair)['status'] == 'HOLD'
    pair['primary_mean_ci95'][0] = .0001
    pair['references']['dread']['delta'] = -.00301
    assert screen.admission(pair)['status'] == 'HOLD'
