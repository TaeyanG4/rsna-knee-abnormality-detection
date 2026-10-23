"""EXP159 local CPU tests: no CUDA context, credentials or network."""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import types
import zipfile

import pytest

from tools import exp159_colab_admission as x


def test_derived_runner_keeps_exp150_recipe_and_adds_resume():
    source = (x.ROOT / x.BASE_RUNNER).read_text(encoding='utf-8')
    derived = x.transform_runner(source)
    assert 'for ep in range(start_epoch, epochs):' in derived
    assert 'e159.save_epoch(' in derived
    assert 'e159.publish_epoch(out, ep)' in derived
    assert 'e159.validate_runner_args_colab' in derived
    assert 'from tools import exp150_consensus_weight as e150' in derived
    with pytest.raises(ValueError, match='anchor'):
        x.transform_runner(source.replace('for ep in range(epochs):', 'for ep in range(19):'))


def test_stage_archive_unique_private_and_bootstrap_short(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    for name in (x.BASE_RUNNER, x.TOOL, x.PROTOCOL.as_posix(),
                 'tools/colab/run_remote.py', 'tools/colab/exp159_remote.py', x.e150.MANIFEST):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / x.BASE_RUNNER).write_text((x.ROOT / x.BASE_RUNNER).read_text(encoding='utf-8'), encoding='utf-8')
    (root / x.TOOL).write_text('wrapper', encoding='utf-8')
    (root / x.PROTOCOL).write_text('registered rule', encoding='utf-8')
    (root / 'tools/colab/run_remote.py').write_text('helper', encoding='utf-8')
    (root / 'tools/colab/exp159_remote.py').write_text('exp159 helper', encoding='utf-8')
    (root / x.e150.MANIFEST).write_text(json.dumps({'source_sha256': {x.BASE_RUNNER: x.e150.sha(root / x.BASE_RUNNER)}}), encoding='utf-8')
    monkeypatch.setattr(x.e150, 'package_files', lambda _root: {
        name: dict(sha256=x.e150.sha(root / name), bytes=(root / name).stat().st_size)
        for name in (x.BASE_RUNNER, x.TOOL, x.e150.MANIFEST)
    })
    out = tmp_path / 'stage'
    x.stage(out, root)
    with zipfile.ZipFile(out / 'code-dataset' / x.ARCHIVE) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        assert x.DERIVED in names
        assert z.read(x.RECEIPT)
    assert x.e150.read(out / 'code-dataset/dataset-metadata.json')['isPrivate'] is True
    monkeypatch.setattr(x, 'verify_stage', lambda _out: x.e150.read(out / 'stage-receipt.json'))
    cell = x.bootstrap_cell(out)
    assert len(cell.splitlines()) <= 25
    assert "userdata.get('KAGGLE_API_TOKEN')" in cell
    assert '--no-deps' in cell
    assert "'-U'" in cell
    assert "['kaggle', '--version']" in cell
    assert 'z.extractall(work)' in cell
    assert 'print(token)' not in cell
    assert x.CODE_DS in cell


def test_epoch_checkpoint_restores_cpu_fake_states(tmp_path, monkeypatch):
    pytest.importorskip('numpy')
    root = tmp_path / 'root'
    for name in (x.e150.MANIFEST, x.DERIVED, x.PROTOCOL.as_posix()):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name, encoding='utf-8')
    monkeypatch.setattr(x, 'ROOT', root)
    fake_torch = types.SimpleNamespace(
        save=lambda obj, path: path.write_bytes(pickle.dumps(obj)),
        load=lambda path, **kw: pickle.loads(path.read_bytes()),
        get_rng_state=lambda: b'cpu-rng', set_rng_state=lambda value: None,
        cuda=types.SimpleNamespace(get_rng_state_all=lambda: [b'fake-rng'],
                                   set_rng_state_all=lambda value: None,
                                   is_initialized=lambda: False))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    class State:
        def __init__(self, value): self.value = value
        def state_dict(self): return {'value': self.value}
        def load_state_dict(self, value, **kw): self.value = value['value']

    class Generator:
        def __init__(self, value): self.value = value
        def get_state(self): return self.value
        def set_state(self, value): self.value = value

    model, opt, sched, scaler, gen = [State(v) for v in (3, 4, 5, 6)] + [Generator(7)]
    out = tmp_path / 'run'; out.mkdir()
    prov = {'environment': {'torch': 'fake', 'gpu': 'T4', 'driver': 'fake'}}
    x.save_epoch(out, 0, model, opt, sched, scaler, gen,
                 [{'epoch': 0, 'optimizer_updates': 431}], None, 431, 0, prov)
    assert x.resume_directory_allowed(out)
    for part in (model, opt, sched, scaler, gen): part.value = -1
    history, best, updates, skipped, start = x.restore_epoch(
        out, model, opt, sched, scaler, gen, prov, 99)
    assert (history, best, updates, skipped, start) == ([{'epoch': 0, 'optimizer_updates': 431}], None, 431, 0, 1)
    assert [v.value for v in (model, opt, sched, scaler, gen)] == [3, 4, 5, 6, 7]
    (out / x.CHECKPOINT).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='mismatch'):
        x.restore_epoch(out, model, opt, sched, scaler, gen, prov, 0)
    assert not fake_torch.cuda.is_initialized()


def test_admission_uses_each_registered_spread():
    scores = {n: {'delta': x.LIMITS[n]} for n in x.REFS}
    assert x.admission(scores)
    scores['dread']['delta'] = -x.LIMITS['dread'] - .00001
    assert not x.admission(scores)
    del scores['consensus']
    with pytest.raises(ValueError, match='four'):
        x.admission(scores)


def test_warning_prefixed_dataset_csv_and_header_variants(monkeypatch):
    from tools.colab import exp159_remote

    outputs = [
        'Warning: Looks like you are using an outdated `kaggle` version\n'
        'name,size,creationDate\nepoch-state.pt,15,2026-10-09\n',
        'Warning: update available\nFile Name,Size Bytes,Creation Date\nepoch-state.pt,15,today\n',
    ]
    for output in outputs:
        monkeypatch.setattr(exp159_remote.subprocess, 'run',
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, output, ''))
        assert exp159_remote.dataset_files('owner/results') == [('epoch-state.pt', 15)]
    with pytest.raises(ValueError, match='name,size'):
        exp159_remote.parse_dataset_files('Warning: no CSV followed')


def test_epoch_upload_uses_private_dataset_and_checks_checkpoint(tmp_path, monkeypatch):
    from tools.colab import exp159_remote
    out = tmp_path / 'run'; out.mkdir()
    (out / x.CHECKPOINT).write_bytes(b'fake-checkpoint')
    commands = []

    def fake_run(cmd, **kw):
        commands.append(cmd)
        if cmd[2] == 'status':
            return subprocess.CompletedProcess(cmd, 0, stdout='Current Dataset Status: ready', stderr='')
        return subprocess.CompletedProcess(cmd, 0, stdout='ok', stderr='')

    monkeypatch.setattr(x.subprocess, 'run', fake_run)
    monkeypatch.setattr(exp159_remote, 'dataset_files',
                        lambda ref: [(x.CHECKPOINT, (out / x.CHECKPOINT).stat().st_size)])
    assert x.publish_epoch(out, 0)
    assert x.e150.read(out / 'dataset-metadata.json')['isPrivate'] is True
    assert commands[0][:3] == ['kaggle', 'datasets', 'status']
    assert commands[1][:3] == ['kaggle', 'datasets', 'version']


def test_epoch_upload_waits_for_ready_and_retries_403(tmp_path, monkeypatch):
    from tools.colab import exp159_remote

    out = tmp_path / 'run'; out.mkdir()
    (out / x.CHECKPOINT).write_bytes(b'checkpoint')
    commands = []
    statuses = iter([(1, '', '404 Not Found'), (0, 'Current Dataset Status: pending', ''),
                     (1, '', '403 Forbidden: status not propagated'),
                     (0, 'Current Dataset Status: ready', '')])
    listings = iter([(1, '', '403 Forbidden: ListDatasetFiles'),
                     (0, 'Warning: outdated kaggle version\nname,size\nepoch-state.pt,10\n', '')])

    def fake_run(cmd, **kw):
        commands.append(cmd)
        if cmd[2] == 'status': code, stdout, stderr = next(statuses)
        elif cmd[2] == 'files': code, stdout, stderr = next(listings)
        else: code, stdout, stderr = 0, 'created', ''
        return subprocess.CompletedProcess(cmd, code, stdout, stderr)

    monkeypatch.setattr(x.subprocess, 'run', fake_run)
    monkeypatch.setattr(x.time, 'sleep', lambda _: None)
    assert x.publish_epoch(out, 0, max_wait=10, poll_interval=0)
    assert commands[1][:3] == ['kaggle', 'datasets', 'create']
    assert sum(cmd[2] == 'files' for cmd in commands) == 2


def test_epoch_upload_persistent_403_warns_and_keeps_fit_alive(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'run'; out.mkdir()
    (out / x.CHECKPOINT).write_bytes(b'checkpoint')
    commands = []

    def fake_run(cmd, **kw):
        commands.append(cmd)
        if cmd[2] == 'status':
            return subprocess.CompletedProcess(cmd, 0, 'Current Dataset Status: ready', '')
        if cmd[2] == 'files':
            return subprocess.CompletedProcess(cmd, 1, '', '403 Forbidden: ListDatasetFiles')
        return subprocess.CompletedProcess(cmd, 0, 'versioned', '')

    monkeypatch.setattr(x.subprocess, 'run', fake_run)
    monkeypatch.setattr(x.time, 'sleep', lambda _: None)
    assert x.publish_epoch(out, 0, max_wait=10, poll_interval=0) is False
    assert 'WARNING:' in capsys.readouterr().out
    assert (out / x.CHECKPOINT).read_bytes() == b'checkpoint'
    assert sum(cmd[2] == 'files' for cmd in commands) == 3
