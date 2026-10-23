"""EXP167 CPU-only contract tests. Fakes use no CUDA or network."""
from __future__ import annotations

import io
import json
import tarfile

import pytest

from tools import exp167_convnext_t_folds as e


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    from tools import exp166_convnext_t as source
    from tools import exp164_dino_base_folds as dino

    root = tmp_path / 'project'
    root.mkdir()
    original = {'seed': 42, 'label_set': {'path': 'labels.csv'},
                'common_args': {'arch': 'convnext_tiny.fb_in22k_ft_in1k', 'k': 12,
                                'epochs': 20, 'fixed_epoch': 20, 'amp': 'fp16',
                                'eval_amp': 'fp16'}, 'extra_eval': {'k_eval': 36}}
    _save(root / e.SOURCE_MANIFEST, original)
    (root / e.PROTOCOL).parent.mkdir(parents=True)
    (root / e.PROTOCOL).write_text('fake registered screen', encoding='utf-8')
    (root / 'config').mkdir()
    (root / 'config/workspace.toml').write_text('schema_version = 1\n', encoding='utf-8')
    (root / 'tools').mkdir()
    (root / e.TOOL).write_text('# fake CPU tool\n', encoding='utf-8')
    reference, rosters = {'folds': {}}, {}
    for f, count in ((1, 864), (2, 864), (3, 864), (4, 863)):
        all_ids = [f'{f}-{i:04}' for i in range(4343)]
        ids = {'train': all_ids[count:], 'val': all_ids[:count]}
        rosters[f] = ('ignored', ids)
        _save(root / e.roster_path(f), ids)
        group = f'groups/fold{f}.csv'
        gp = root / group
        gp.parent.mkdir(exist_ok=True)
        gp.write_text('only metadata\n', encoding='utf-8')
        original_path = f'original/fold{f}.json'
        _save(root / original_path, ids)
        reference['folds'][str(f)] = {'fold': f, 'tag': f'old-{f}',
                                     'groups': {'path': group, 'sha256': e.sha(gp)},
                                     'ordered_ids': {'path': original_path},
                                     'n_train': 4343-count, 'n_val': count,
                                     'steps': ((4343-count)//8)*20,
                                     'pos_weight': [2.0]*12}
    _save(root / e.FOLD_SOURCE, reference)
    source_files = {name: {'sha256': 'a'*64, 'bytes': 1} for name in e.REQUIRED_MODULES}
    source_files.update({e.SOURCE_MANIFEST: {'sha256': e.sha(root / e.SOURCE_MANIFEST), 'bytes':
                                       (root / e.SOURCE_MANIFEST).stat().st_size},
                    e.SOURCE_WEIGHT: {'sha256': e.WEIGHT_SHA, 'bytes': 1},
                    'tools/exp166_convnext_t.py': {'sha256': 'a'*64, 'bytes': 1}})
    monkeypatch.setattr(source, 'registration', lambda root: (original, 'unused'))
    monkeypatch.setattr(e, '_source_files', lambda root: (source_files, 'b'*64))
    monkeypatch.setattr(e, 'runner_source', lambda root: 'from tools import exp167_convnext_t_folds as guard\n')
    monkeypatch.setattr(e, 'guard_attributes', lambda runner: ['validate_arguments'])
    monkeypatch.setattr(dino, 'roster', lambda root: rosters)
    monkeypatch.setattr(source.old, 'load_label_csv', lambda path: ({}, 'digest'))
    monkeypatch.setattr(source.old, 'pos_weight', lambda labels, ids: [2.0]*12)
    overlay = tmp_path / 'overlay'
    result = e.prepare(overlay, root)
    return root, overlay, result


def test_prepare_is_slim_serial_and_fold_local(fake_package):
    _, overlay, result = fake_package
    p = e.validate(overlay)
    assert result['status'] == 'PREPARED_NOT_LAUNCHED'
    assert p['files'][e.SOURCE_WEIGHT]['origin'] == 'exp166'
    assert not (overlay / e.SOURCE_WEIGHT).exists()
    assert (overlay / 'config/workspace.toml').is_file()
    assert [job['tag'] for job in e.queue()] == [e.tag(f) for f in range(1, 5)]
    assert [job['command'][-1] for job in e.queue()] == ['1', '2', '3', '4']
    for f in range(1, 5):
        q = e.read(overlay / e.protocol_path(f))
        assert q['folds'][str(f)]['pos_weight'] == [2.0]*12
        assert q['folds'][str(f)]['tag'] == e.tag(f)
        assert q['folds'][str(f)]['ordered_ids']['path'] == e.roster_path(f)


def test_workspace_file_is_mandatory_and_hashed(fake_package):
    _, overlay, _ = fake_package
    (overlay / 'config/workspace.toml').unlink()
    with pytest.raises(ValueError, match='package file differs'):
        e.validate(overlay)


def test_runner_guard_and_transitive_source_contract():
    from tools import exp166_convnext_t as source
    derived = e.runner_source()
    attrs = e.guard_attributes(derived)
    assert 'validate_arguments' in attrs
    assert 'build_offline_model' in attrs
    assert 'restore_epoch' in attrs
    assert 'evaluate_flip_tta' in attrs
    assert 'from tools import exp167_convnext_t_folds as guard' in derived
    assert 'oof_k36_flip_tta.csv' in derived
    files, _ = e._source_files(e.ROOT)
    assert set(source.TRANSITIVE_MODULES).issubset(files)
    assert e.REQUIRED_MODULES.issubset(files)


def test_capped_one_queue_and_exact_remote_paths():
    assert e.FULL_CAP >= 1.3 * e.FOLD0_SECONDS
    assert e.FOLD_CAP >= 1.3 * e.FOLD0_SECONDS
    plan = e.vast_plan('/workspace/exp167-overlay')
    assert 'setsid nohup python tools/gpu_queue.py queue.json --foreground' in plan
    assert 'max_gpu_seconds_per_fold=2500' in plan
    assert '/workspace/exp167' in plan and '\\' not in plan
    assert 'output-inventory.json' not in plan
    with pytest.raises(ValueError):
        e.vast_plan(r'H:\bad')


def test_incomplete_or_unsafe_export_is_rejected(fake_package, tmp_path):
    _, overlay, _ = fake_package
    export = tmp_path / 'unsafe.tar'
    with tarfile.open(export, 'w') as tar:
        data = b'bad'
        member = tarfile.TarInfo('../escape')
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    out = tmp_path / 'collected'
    with pytest.raises(ValueError, match='unsafe/incomplete'):
        e.collect(export, overlay, out)
    assert not out.exists()


def test_strict_registered_screen_gate_and_fold_local_ranks():
    import numpy as np

    class FakeEvaluator:
        @staticmethod
        def pct_rank(values):
            return (np.argsort(np.argsort(values[:, 0])) + 1)[:, None] / len(values)

    parts = {f: np.array([[f*100 + 2], [f*100 + 1]]) for f in range(5)}
    assert e.rank_per_fold(parts, FakeEvaluator)[:, 0].tolist() == [1., .5]*5
    pair = {'primary_mean_delta': .001, 'primary_mean_ci95': [-.001, .003]}
    assert not e.screen_gate(pair)
    pair['primary_mean_ci95'][0] = 0
    assert not e.screen_gate(pair)
    pair['primary_mean_ci95'][0] = .0001
    assert e.screen_gate(pair)


def test_safe_paths_and_bundle_modes(fake_package):
    _, overlay, _ = fake_package
    for name in ('../key', '/root/key', r'H:\key', 'a\\b'):
        with pytest.raises(ValueError):
            e.safe(name)
    p = e.read(overlay / 'manifest.json')
    assert p['files'][e.TOOL]['origin'] == 'overlay'
    assert p['files'][e.RUNNER]['origin'] == 'overlay'
    assert p['files']['config/workspace.toml']['origin'] == 'overlay'
    assert p['files'][e.SOURCE_MANIFEST]['origin'] == 'exp166'
