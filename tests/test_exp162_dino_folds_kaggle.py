"""CPU-only wiring and admission tests for EXP162; no CUDA or network access."""
import json
from pathlib import Path
import zipfile

import pytest

from tools import exp162_dino_folds_kaggle as e


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_fold_mapping_uses_rosters_not_runner_fold_argument(tmp_path, monkeypatch):
    parent = tmp_path / 'parent.json'
    baseline = {'train': ['b', 'c', 'd', 'e'], 'val': ['a']}
    _json(tmp_path / 'baseline.json', baseline)
    _json(parent, {'ordered_ids': {'path': 'baseline.json'}})
    monkeypatch.setattr(e, 'PARENT', 'parent.json')
    monkeypatch.setattr(e, 'PARENT_SHA', e.sha(parent))
    monkeypatch.setattr(e, 'FOLD_COUNTS', {i: 1 for i in range(5)})
    monkeypatch.setattr(e, 'STUDIES', 5)
    bindings = {}
    for fold, uid in enumerate('abcde'):
        path = f'fold{fold}.json'
        _json(tmp_path / path, {'train': sorted(set('abcde') - {uid}), 'val': [uid]})
        bindings[str(fold)] = {'ordered_ids': {'path': path, 'sha256': e.sha(tmp_path / path)}}
    _json(tmp_path / 'artifacts/experiments/EXP134_AUG1_KFOLD_PROTOCOL_V1/s42/manifest.json',
          {'folds': bindings})
    rosters = e.roster(tmp_path)
    assert [rosters[i][1]['val'][0] for i in range(5)] == list('abcde')
    _json(tmp_path / 'fold0.json', {'train': ['b', 'c', 'd', 'e'], 'val': ['z']})
    with pytest.raises(ValueError):
        e.roster(tmp_path)


def test_package_archive_has_unique_literal_members(tmp_path):
    target = tmp_path / 'assets.zip.bin'
    digest = e.pack(target, {'a.txt': b'abc', 'nested/b.txt': b'def'})
    assert digest == e.sha(target)
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ['a.txt', 'nested/b.txt']
    with pytest.raises(ValueError, match='unsafe'):
        e.safe('../secrets.txt')
    with pytest.raises(ValueError, match='fresh'):
        e.pack(target, {'x': b'x'})


def test_notebook_sets_cuda_environment_before_any_query():
    nb = e.notebook('0' * 64)
    source = nb['cells'][1]['source']
    assert source.index("os.environ['CUBLAS_WORKSPACE_CONFIG']") < source.index('subprocess.run')
    assert "os.environ['CUDA_VISIBLE_DEVICES'] = ''" in source
    assert 'exp162-package.zip.bin' in source
    compile(source, 'run.ipynb', 'exec')


@pytest.mark.parametrize('seconds,expected', [(6900, True), (6899, False), (0, False),
                                               (float('nan'), False)])
def test_fold_deadline(seconds, expected):
    assert e.admit_fold(seconds) is expected


def test_runner_command_pins_fixed_recipe_without_importing_torch(tmp_path):
    protocol = tmp_path / 'fold1.json'
    _json(protocol, {'tag': e.tag(1), 'expected_args': {'seed': 42, 'fold': 1,
          'epochs': 20, 'k': 12, 'k_eval': 36, 'amp': 'fp16', 'grad_ckpt': True},
          'paths': {'labels': 'labels.csv'}})
    args = e.runner_argv(protocol, tmp_path)
    assert args[args.index('--fold')+1] == '1'
    assert args[args.index('--amp')+1] == 'fp16'
    assert args[args.index('--labels')+1] == str(tmp_path / 'labels.csv')
    assert '--grad-ckpt' in args
    assert not any('torch' in value for value in args)


def test_frozen_runner_is_wrapped_with_one_import_change():
    original = (e.ROOT / e.SOURCE_RUNNER).read_text(encoding='utf-8')
    adapted = e.runner_source()
    assert adapted == original.replace('from tools import exp158_dino_strong as guard',
                                       'from tools import exp162_dino_folds_kaggle as guard')
    assert adapted.count('from tools import exp162_dino_folds_kaggle as guard') == 1


def test_registered_paired_ci_requires_positive_lower_bound():
    import numpy as np

    class FakeEvaluator:
        @staticmethod
        def auc_targets(_y, score, _mask, _minimum):
            return np.full(12, score[0, 0])

        @staticmethod
        def macro(values):
            return float(np.mean(values))

    refs = {name: (np.zeros((4, 12)), np.ones((4, 12), bool), 1)
            for name in ('jev', 'aquila', 'dread', 'consensus')}
    draws = [np.arange(4)] * 300
    base = np.full((4, 12), .6)
    assert e._paired(np.full((4, 12), .7), base, refs, draws, FakeEvaluator)['pass']
    assert not e._paired(base, base, refs, draws, FakeEvaluator)['pass']


def test_partial_collection_cannot_pass_screen(tmp_path):
    outputs = tmp_path / 'outputs'
    _json(outputs / 'collection.json', {'schema': e.SCHEMA, 'status': 'INCOMPLETE',
          'completed_folds': [1, 2, 3], 'incomplete_folds': {'4': 'SKIPPED_DEADLINE'}})
    result = e.screen(outputs, tmp_path / 'screen', tmp_path)
    assert result['verdict'] == 'INCOMPLETE'
    assert json.loads((tmp_path / 'screen/result.json').read_text())['incomplete_folds'] == {'4': 'SKIPPED_DEADLINE'}
