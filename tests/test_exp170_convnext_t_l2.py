"""EXP170 CPU contracts; no CUDA context, network, or Vast launch."""
from __future__ import annotations

import io
import json
import tarfile

import pytest

from tools import exp170_convnext_t_l2 as e


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    from tools import exp166_convnext_t as source
    from tools import exp164_dino_base_folds as dino

    root = tmp_path / 'project'
    root.mkdir()
    all_ids = [f'{i:04}' for i in range(4343)]
    val_counts = [888, 864, 864, 864, 863]
    offset = 0
    fold_info = {}
    for f, count in enumerate(val_counts):
        val = all_ids[offset:offset + count]
        offset += count
        val_set = set(val)
        train = [uid for uid in all_ids if uid not in val_set]
        if f == 0:
            train, val = train[::-1], val[::-1]  # EXP127 fold-0 order is retained.
        ids = {'train': train, 'val': val}
        save(root / e.roster_path(f), ids)
        original_path = f'original/fold{f}.json'
        save(root / original_path, ids)
        group = f'groups/fold{f}.csv'
        gp = root / group
        gp.parent.mkdir(exist_ok=True)
        gp.write_text('metadata only\n', encoding='utf-8')
        fold_info[str(f)] = {'fold': f, 'tag': 'old',
                             'ordered_ids': {'path': e.FOLD0_ROSTER if f == 0 else original_path,
                                             'sha256': e.sha(root / e.roster_path(f))},
                             'groups': {'path': group, 'sha256': e.sha(gp)},
                             'n_train': len(train), 'n_val': len(val),
                             'steps': (len(train) // 8) * 20,
                             'pos_weight': [3.0] * 12}
    registered = {'seed': 42, 'initial_state_sha256': 'ee83878e',
                  'paths': {'labels': 'old.csv', 'eval_labels': 'old.csv'},
                  'label_set': {'path': 'old.csv'}, 'recipe': {'changes': []},
                  'common_args': {'arch': e.__getattr__('ARCH'), 'k': 12,
                                  'epochs': 20, 'fixed_epoch': 20,
                                  'amp': 'fp16', 'eval_amp': 'fp16'},
                  'extra_eval': {'k_eval': 36}, 'folds': {'0': fold_info['0']}}
    save(root / e.SOURCE_MANIFEST, registered)
    save(root / e.FOLD_SOURCE, {'folds': {str(f): fold_info[str(f)] for f in range(1, 5)}})
    for name, content in ((e.PROTOCOL, 'registered before run'),
                          (e.TOOL, '# CPU fake tool\n'),
                          ('config/workspace.toml', 'schema_version = 1\n'),
                          (e.LABEL, 'fake,soft\n'),
                          (e.LABEL_PROVENANCE, '')):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    monkeypatch.setattr(e, 'LABEL_SHA', e.sha(root / e.LABEL))
    save(root / e.LABEL_PROVENANCE,
         {'csv_sha256': e.LABEL_SHA, 'gold58_included': False, 'rows': 4347})
    files = {name: {'sha256': 'a' * 64, 'bytes': 1}
             for name in e.REQUIRED_MODULES | set(source.TRANSITIVE_MODULES)}
    files[e.SOURCE_MANIFEST] = {'sha256': e.sha(root / e.SOURCE_MANIFEST),
                                'bytes': (root / e.SOURCE_MANIFEST).stat().st_size}
    files[e.SOURCE_WEIGHT] = {'sha256': e.WEIGHT_SHA, 'bytes': 1}
    monkeypatch.setattr(source, 'registration', lambda project: (registered, 'runner'))
    monkeypatch.setattr(e, '_source_files', lambda project: (files, 'b' * 64))
    monkeypatch.setattr(e, 'runner_source', lambda project: 'from tools import exp170_convnext_t_l2 as guard\n')
    monkeypatch.setattr(e, 'guard_attributes', lambda runner: ['validate_arguments'])
    monkeypatch.setattr(dino, 'roster', lambda project: {f: (None, None) for f in range(5)})
    monkeypatch.setattr(source.old, 'load_label_csv', lambda path: ({}, 'semantic'))
    monkeypatch.setattr(source.old, 'pos_weight', lambda labels, train: [2.0] * 12)
    overlay = tmp_path / 'overlay'
    result = e.prepare(overlay, root)
    return root, overlay, result


def test_prepare_all_five_l2_only_and_serial(fake_package):
    _, overlay, result = fake_package
    m = e.validate(overlay)
    assert result['status'] == 'PREPARED_NOT_LAUNCHED'
    assert m['serial_folds'] == list(range(5))
    assert [job['command'][-1] for job in e.queue()] == ['0', '1', '2', '3', '4']
    assert all(not job['continue_on_failure'] for job in e.queue())
    assert m['files'][e.SOURCE_WEIGHT]['origin'] == 'exp166'
    assert m['files'][e.LABEL]['origin'] == 'overlay'
    assert (overlay / 'config/workspace.toml').is_file()
    for f in range(5):
        q = e.read(overlay / e.protocol_path(f))
        assert q['paths']['labels'] == q['paths']['eval_labels'] == e.LABEL
        assert q['label_set']['semantic_sha256'] == 'semantic'
        assert q['initial_state_sha256'] == m['source_initial_state_sha256']
        assert q['folds'][str(f)]['pos_weight'] == [2.0] * 12
        assert q['folds'][str(f)]['ordered_ids']['path'] == e.roster_path(f)
    assert e.read(overlay / e.roster_path(0))['val'] != sorted(e.read(overlay / e.roster_path(0))['val'])


def test_workspace_label_and_roster_are_hashed(fake_package):
    _, overlay, _ = fake_package
    for path in ('config/workspace.toml', e.LABEL, e.roster_path(2)):
        target = overlay / path
        original = target.read_bytes()
        target.write_bytes(original + b'changed')
        with pytest.raises(ValueError, match='package file differs'):
            e.validate(overlay)
        target.write_bytes(original)


def test_runner_guard_transitive_modules_and_exact_vast_plan():
    from tools import exp166_convnext_t as source

    runner = e.runner_source()
    attrs = e.guard_attributes(runner)
    assert {'validate_arguments', 'build_offline_model', 'restore_epoch',
            'evaluate_flip_tta'}.issubset(attrs)
    assert 'from tools import exp170_convnext_t_l2 as guard' in runner
    files, _ = e._source_files(e.ROOT)
    assert set(source.TRANSITIVE_MODULES).issubset(files)
    assert e.REQUIRED_MODULES.issubset(files)
    plan = e.vast_plan('/workspace/exp170-overlay')
    assert 'setsid nohup python tools/gpu_queue.py queue.json --foreground' in plan
    assert 'serial_folds=[0,1,2,3,4]' in plan
    assert '/workspace/exp170' in plan and '\\' not in plan
    assert 'output-inventory.json' not in plan
    assert e.FULL_CAP >= 1.3 * e.MEASURED_MAX_SECONDS
    assert e.FOLD_CAP >= 1.3 * e.MEASURED_MAX_SECONDS
    with pytest.raises(ValueError):
        e.vast_plan(r'H:\bad')


def test_runner_arguments_change_both_label_flags(monkeypatch, tmp_path):
    from tools import exp166_convnext_t as source

    flags = ['python', 'old-runner.py', '--protocol', 'old.json', '--fold', '0',
             '--tag', 'old', '--labels', 'dread.csv', '--eval-labels', 'dread.csv']
    monkeypatch.setattr(source, 'runner_args',
                        lambda root, output, workers, pilot: list(flags))
    args = e.runner_args(2, tmp_path)
    assert args[args.index('--labels') + 1] == str(tmp_path / e.LABEL)
    assert args[args.index('--eval-labels') + 1] == str(tmp_path / e.LABEL)
    assert args[args.index('--protocol') + 1] == str(tmp_path / e.protocol_path(2))
    assert args[args.index('--tag') + 1] == e.tag(2)


def test_registered_gate_uses_jev_consensus_not_aquila():
    import numpy as np

    class Evaluator:
        @staticmethod
        def auc_targets(y, pred, mask, minimum):
            multiplier = {0: 1., 1: -5., 2: -.02, 3: 2.}[int(y[0])]
            return np.mean(pred, axis=0) * multiplier

        @staticmethod
        def macro(values):
            return float(np.mean(values))

    c, b = np.full((4, 12), .6), np.full((4, 12), .5)
    refs = {name: (np.full(4, marker), np.ones(4), 1) for name, marker in
            (('jev', 0), ('aquila', 1), ('dread', 2), ('consensus', 3))}
    pair = e.paired_jev_consensus(c, b, refs, [np.arange(4)] * 300, Evaluator)
    assert pair['primary_mean_delta'] == pytest.approx(.15)
    assert pair['references']['aquila']['delta'] == pytest.approx(-.5)
    assert e.admission(pair)['status'] == 'ADMIT'
    pair['primary_mean_ci95'][0] = 0
    assert e.admission(pair)['status'] == 'HOLD'


def test_unsafe_export_and_paths_rejected(fake_package, tmp_path):
    _, overlay, _ = fake_package
    export = tmp_path / 'bad.tar'
    with tarfile.open(export, 'w') as tar:
        member = tarfile.TarInfo('../escape')
        member.size = 3
        tar.addfile(member, io.BytesIO(b'bad'))
    collected = tmp_path / 'collected'
    with pytest.raises(ValueError, match='unsafe/incomplete'):
        e.collect(export, overlay, collected)
    assert not collected.exists()
    for name in ('../key', '/root/key', r'H:\key', 'a\\b'):
        with pytest.raises(ValueError):
            e.safe(name)
