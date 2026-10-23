"""CPU-only EXP122 protocol checks. No test creates a CUDA context."""
from pathlib import Path

import pytest

from tools import exp122_multihead_seeds as ex


def test_runner_ordered_equivalence():
    """Every byte of the copied EXP117 runner, in order, except its guard import."""
    a = (ex.ROOT / 'training/runners/exp117_multihead.py').read_bytes()
    b = (ex.ROOT / ex.RUNNER).read_bytes()
    old = b'from tools import exp117_multihead as guard  # EXP117: own guard'
    new = b'from tools import exp122_multihead_seeds as guard  # EXP122: seed-replicate guard'
    assert a.count(old) == b.count(new) == 1
    assert a.replace(old, new) == b


@pytest.mark.parametrize('seed', ex.SEEDS)
def test_five_independent_fold_entries(seed, monkeypatch, tmp_path):
    p = {'seed': seed, 'common_args': {}, 'paths': {'corpus': 'c', 'labels': 'l', 'eval_labels': 'l'}}
    monkeypatch.setattr(ex, 'protocol', lambda s, root=ex.ROOT: p)
    monkeypatch.setattr(ex, 'train_python', lambda: 'python')
    monkeypatch.setattr(ex.e117.e113, 'weights_path', lambda root: Path(root) / 'weights')
    entries = ex.queue_entries(seed, 'PRIOR', root=tmp_path)
    assert [x['tag'] for x in entries] == [f'EXP122_MULTIHEAD_S{seed}_F{k}' for k in range(5)]
    assert [x['after'] for x in entries] == ['PRIOR'] + [x['tag'] for x in entries[:-1]]
    assert all(x['continue_on_failure'] is True for x in entries)
    assert all(x['command'][1].endswith('exp122_multihead_seeds.py') for x in entries)


def test_initial_state_requires_identical_backbone_and_changed_head():
    base = {'backbone.w': ['a'], 'head.w': ['b']}
    ex.assert_initial_state(43, {'backbone.w': ['a'], 'head.w': ['c']}, base)
    with pytest.raises(ValueError, match='backbone'):
        ex.assert_initial_state(43, {'backbone.w': ['z'], 'head.w': ['c']}, base)
    with pytest.raises(ValueError, match='head'):
        ex.assert_initial_state(43, base, base)
    with pytest.raises(ValueError, match='EXP118'):
        ex.assert_initial_state(43, {'backbone.w': ['a'], 'head.w': ['c']}, base,
                                {'backbone.w': ['a'], 'head.w': ['d']})


def test_seed_specific_paths_and_tags():
    assert ex.paths(43)['protocol'].endswith('EXP122_MULTIHEAD_S43_PROTOCOL_V1')
    assert ex.tag_for(4, seed=44) == 'EXP122_MULTIHEAD_S44_F4'
    with pytest.raises(ValueError):
        ex.paths(42)
