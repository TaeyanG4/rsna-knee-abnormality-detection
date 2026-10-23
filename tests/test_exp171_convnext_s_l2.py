"""EXP171 CPU contracts. All weight, model and remote operations are faked."""
from __future__ import annotations

import io
import json
import tarfile

import pytest

from tools import exp171_convnext_s_l2 as e


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    from tools import exp166_convnext_t as source

    root = tmp_path / 'project'
    root.mkdir()
    ids = [f'{i:04}' for i in range(4343)]
    roster = {'train': ids[888:][::-1], 'val': ids[:888][::-1]}
    save(root / e.FOLD0_ROSTER, roster)
    group = 'groups/fold0.csv'
    (root / group).parent.mkdir(parents=True, exist_ok=True)
    (root / group).write_text('metadata only\n', encoding='utf-8')
    info = {'fold': 0, 'tag': 'old',
            'ordered_ids': {'path': e.FOLD0_ROSTER,
                            'sha256': e.sha(root / e.FOLD0_ROSTER)},
            'groups': {'path': group, 'sha256': e.sha(root / group)},
            'n_train': 3455, 'n_val': 888, 'steps': (3455 // 8) * 20,
            'pos_weight': [3.0] * 12}
    registered = {'seed': 42, 'initial_state_sha256': 'tiny-state',
                  'paths': {'labels': 'old.csv', 'eval_labels': 'old.csv'},
                  'label_set': {'path': 'old.csv'},
                  'initial_state': {'keys_file': 'tiny-keys.json'},
                  'evaluation': {'exp166_screen': {}},
                  'recipe': {'changes': []},
                  'common_args': {'arch': source.ARCH, 'k': 12,
                                  'epochs': 20, 'fixed_epoch': 20,
                                  'amp': 'fp16', 'eval_amp': 'fp16'},
                  'extra_eval': {'k_eval': 36}, 'folds': {'0': info}}
    save(root / e.SOURCE_MANIFEST, registered)
    for name, content in ((e.PROTOCOL, 'registered before run'),
                          (e.TOOL, '# CPU fake tool\n'),
                          ('config/workspace.toml', 'schema_version = 1\n'),
                          (e.LABEL, 'fake,soft\n'),
                          (e.WEIGHT, 'fake-small-weight'),
                          (e.INITIAL, 'fake-small-state')):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    save(root / e.INITIAL_KEYS, {'keys': [], 'combined_sha256': 'small-state', 'seed': 42})
    monkeypatch.setattr(e, 'LABEL_SHA', e.sha(root / e.LABEL))
    save(root / e.LABEL_PROVENANCE,
         {'csv_sha256': e.LABEL_SHA, 'gold58_included': False, 'rows': 4347})
    files = {name: {'sha256': 'a' * 64, 'bytes': 1}
             for name in e.REQUIRED_MODULES | set(source.TRANSITIVE_MODULES)}
    files[e.SOURCE_MANIFEST] = {'sha256': e.sha(root / e.SOURCE_MANIFEST),
                                'bytes': (root / e.SOURCE_MANIFEST).stat().st_size}
    files[e.SOURCE_WEIGHT] = {'sha256': 'tiny-weight', 'bytes': 1}
    files[source.INITIAL] = {'sha256': 'tiny-state', 'bytes': 1}
    files[source.INITIAL_KEYS] = {'sha256': 'tiny-keys', 'bytes': 1}
    monkeypatch.setattr(source, 'registration', lambda project: (registered, 'runner'))
    monkeypatch.setattr(e, 'register_initial', lambda project: ('small-state', {'strict_result': 'all backbone keys loaded'}))
    monkeypatch.setattr(e, '_source_files', lambda project: (files, 'b' * 64))
    monkeypatch.setattr(e, 'runner_source', lambda project: 'from tools import exp171_convnext_s_l2 as guard\n')
    monkeypatch.setattr(e, 'guard_attributes', lambda runner: ['validate_arguments'])
    monkeypatch.setattr(source.old, 'load_label_csv', lambda path: ({}, 'semantic'))
    monkeypatch.setattr(source.old, 'pos_weight', lambda labels, train: [2.0] * 12)
    overlay = tmp_path / 'overlay'
    result = e.prepare(overlay, root)
    return root, overlay, result


def test_single_fold_small_contract(fake_package):
    _, overlay, result = fake_package
    manifest = e.validate(overlay)
    q = e.read(overlay / e.protocol_path(0))
    assert result['status'] == 'PREPARED_NOT_LAUNCHED'
    assert manifest['serial_folds'] == [0] and len(e.queue()) == 1
    assert q['common_args']['arch'] == e.ARCH
    assert q['paths']['labels'] == q['paths']['eval_labels'] == e.LABEL
    assert q['initial_state_sha256'] == 'small-state'
    assert q['init_weights']['sha256'] == e.sha(overlay / e.WEIGHT)
    assert q['folds']['0']['pos_weight'] == [2.0] * 12
    assert (overlay / 'config/workspace.toml').is_file()
    assert e.SOURCE_WEIGHT not in manifest['files']
    assert all(name not in manifest['files'] for name in
               ('artifacts/experiments/EXP166_CONVNEXT_T/initial-state.pt',
                'artifacts/experiments/EXP166_CONVNEXT_T/initial-state-keys.json'))


def test_hashed_small_assets_and_roster(fake_package):
    _, overlay, _ = fake_package
    for name in ('config/workspace.toml', e.WEIGHT, e.INITIAL, e.FOLD0_ROSTER):
        path = overlay / name
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        with pytest.raises(ValueError, match='package file differs'):
            e.validate(overlay)
        path.write_bytes(original)


def test_fetch_is_explicit_token_false(tmp_path, monkeypatch):
    from types import ModuleType
    import sys
    cache = tmp_path / 'cached.safetensors'
    cache.write_bytes(b'small-weights')
    calls = []
    hub = ModuleType('huggingface_hub')
    def download(**kwargs):
        calls.append(kwargs)
        return str(cache)
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    result = e.fetch_weights(tmp_path)
    assert calls == [{'repo_id': e.WEIGHT_REPO,
                      'filename': 'model.safetensors', 'token': False}]
    assert result['sha256'] == e.sha(tmp_path / e.WEIGHT)
    with pytest.raises(ValueError, match='already exists'):
        e.fetch_weights(tmp_path)


def test_runner_arguments_and_admission(monkeypatch, tmp_path):
    from tools import exp166_convnext_t as source
    flags = ['python', 'old-runner.py', '--protocol', 'old.json', '--fold', '0',
             '--tag', 'old', '--labels', 'dread.csv', '--eval-labels', 'dread.csv',
             '--arch', source.ARCH, '--init-weights', 'tiny.safetensors']
    monkeypatch.setattr(source, 'runner_args',
                        lambda root, output, workers, pilot: list(flags))
    args = e.runner_args(0, tmp_path)
    assert args[args.index('--arch') + 1] == e.ARCH
    assert args[args.index('--init-weights') + 1] == str(tmp_path / e.WEIGHT)
    assert args[args.index('--labels') + 1] == str(tmp_path / e.LABEL)
    assert args[args.index('--eval-labels') + 1] == str(tmp_path / e.LABEL)
    pair = {'primary_mean_delta': .002,
            'primary_mean_ci95': [.00001, .004],
            'references': {'dread': {'delta': -.003},
                           'aquila': {'delta': -.1}}}
    assert e.admission(pair)['status'] == 'ADVANCE'
    pair['primary_mean_ci95'][0] = 0
    assert e.admission(pair)['status'] == 'HOLD'
    pair['primary_mean_ci95'][0] = .00001
    pair['primary_mean_delta'] = .00199
    assert e.admission(pair)['status'] == 'HOLD'


def test_derived_runner_guard_attributes_are_complete():
    runner = e.runner_source()
    attrs = e.guard_attributes(runner)
    assert 'from tools import exp171_convnext_s_l2 as guard' in runner
    assert {'validate_arguments', 'build_offline_model', 'restore_epoch',
            'evaluate_flip_tta'}.issubset(attrs)


def test_guard_stage_plan_and_unsafe_export(fake_package, tmp_path):
    _, overlay, _ = fake_package
    plan = e.vast_plan('/workspace/exp171-overlay')
    assert 'setsid nohup python tools/gpu_queue.py queue.json --foreground' in plan
    assert 'serial_folds=[0]' in plan and 'max_gpu_seconds_per_fold=5100' in plan
    assert '/workspace/exp171' in plan and '\\' not in plan
    assert 'output-inventory.json' not in plan
    with pytest.raises(ValueError):
        e.vast_plan(r'H:\bad')
    export = tmp_path / 'bad.tar'
    with tarfile.open(export, 'w') as tar:
        member = tarfile.TarInfo('../escape')
        member.size = 3
        tar.addfile(member, io.BytesIO(b'bad'))
    with pytest.raises(ValueError, match='unsafe/incomplete'):
        e.collect(export, overlay, tmp_path / 'collected')
    for name in ('../key', '/root/key', r'H:\key', 'a\\b'):
        with pytest.raises(ValueError):
            e.safe(name)
