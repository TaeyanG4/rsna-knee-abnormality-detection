"""EXP166 CPU contract tests. No CUDA context, network or account mutation."""
from __future__ import annotations

import ast
import hashlib
import pytest

from tools import exp166_convnext_t as exp


def test_runner_derivation_and_every_guard_attribute():
    source = exp.runner_source((exp.ROOT / exp.BASE_RUNNER).read_text(encoding='utf-8'))
    tree = ast.parse(source)
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == 'guard'}
    missing = sorted(name for name in used if not hasattr(exp, name))
    assert not missing, missing
    assert 'from tools import exp166_convnext_t as guard' in source
    assert 'guard.restore_epoch(' in source
    assert 'guard.save_epoch(' in source
    assert 'guard.evaluate_flip_tta(' in source
    assert 'oof_k36_flip_tta.csv' in source
    assert 'for ep in range(start_epoch, epochs):' in source


def test_guard_package_includes_all_transitive_runner_imports(monkeypatch, tmp_path):
    runner = exp.runner_source((exp.ROOT / exp.BASE_RUNNER).read_text(encoding='utf-8'))
    fake = {'schema': 'exp166-convnext-t-v1', 'runner_sha256': hashlib.sha256(runner.encode()).hexdigest()}
    monkeypatch.setattr(exp, 'registration', lambda root: (fake, runner))
    old_files = {}
    for name in exp.TRANSITIVE_MODULES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fake CPU module\n', encoding='utf-8')
        old_files[name] = {'sha256': exp.sha(path), 'bytes': path.stat().st_size}
    monkeypatch.setattr(exp.e150, 'package_files', lambda root: dict(old_files))
    for name in (exp.PROTOCOL, exp.MANIFEST, exp.WEIGHT, exp.INITIAL,
                 exp.INITIAL_KEYS, exp.TOOL):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fake')
    files, derived = exp.package_files(tmp_path)
    assert set(exp.TRANSITIVE_MODULES).issubset(files)
    assert exp.RUNNER in files and files[exp.RUNNER]['sha256'] == hashlib.sha256(derived.encode()).hexdigest()


def test_bootstrap_line_limit_and_private_stage(monkeypatch, tmp_path):
    stage = tmp_path / 'stage'
    code = stage / 'code-dataset'
    code.mkdir(parents=True)
    archive = code / exp.ARCHIVE
    archive.write_bytes(b'fake archive')
    monkeypatch.setattr(exp, 'verify_stage', lambda out: {
        'surface': 'colab', 'archive_sha256': exp.sha(archive)})
    cell = exp.bootstrap_cell(stage)
    assert len(cell.splitlines()) <= 25
    assert "'-f', 'exp166-code.zip.bin'" in cell
    assert "'remote-run'" in cell
    assert 'KAGGLE_API_TOKEN' in cell


def test_resume_metadata_rejects_mismatch(monkeypatch, tmp_path):
    meta = {'schema': 'exp166-epoch-v1', 'epoch': 0, 'checkpoint_sha256': 'bad',
            'environment': {'environment': {'torch': 'fake'},
                            'runner_sha256': 'r', 'protocol_sha256': 'p',
                            'initial_state_sha256': 'i'}}
    (tmp_path / exp.CHECKPOINT).write_bytes(b'fake')
    exp.put(tmp_path / exp.CP_META, meta)
    class NoTorch:
        def __getattr__(self, key):
            raise AssertionError('Torch should not be used after a bad hash')
    with pytest.raises(ValueError, match='Resume metadata differs'):
        exp.restore_epoch(tmp_path, NoTorch(), NoTorch(), NoTorch(), NoTorch(),
                          NoTorch(), {'environment': {'torch': 'fake'},
                                      'runner_sha256': 'r', 'protocol_sha256': 'p',
                                      'initial_state_sha256': 'i'}, 0)


def test_screen_gate_is_strict():
    baseline = {'mean_delta': .0013}
    assert exp.screen_gate({'mean_delta': .0014, 'mean_ci95': [.0001, .002]}, baseline)
    assert not exp.screen_gate({'mean_delta': .0013, 'mean_ci95': [.0001, .002]}, baseline)
    assert not exp.screen_gate({'mean_delta': .002, 'mean_ci95': [0, .003]}, baseline)
    assert not exp.screen_gate({'mean_delta': .00125, 'mean_ci95': [.0001, .002]}, {'mean_delta': .0012})


def test_safe_bundle_paths():
    for bad in ('../secret', 'C:/outside', '/root/key', 'a\\b'):
        with pytest.raises(ValueError):
            exp.safe(bad)


def test_warning_prefixed_kaggle_file_listing():
    from tools.colab.exp159_remote import parse_dataset_files
    rows = parse_dataset_files('WARNING: CLI notice\nname,size,creationDate\nepoch-state.pt,42,2026-10-10\n')
    assert rows == [('epoch-state.pt', 42)]
