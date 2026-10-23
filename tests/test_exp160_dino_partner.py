"""CPU-only EXP160 package, decision and fold contracts; no CUDA/network."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from tools import exp160_dino_partner as e


def test_generated_runner_and_base_adapters():
    runner = e.runner_source()
    assert 'from tools import exp160_dino_partner as guard' in runner
    assert 'tds = Aug1TripletWindows(' in runner
    assert 'guard.check_prefix_trace(trace.hexdigest())' in runner
    pos = e.adapter_source(e.ADAPTER_POS)
    patch = e.adapter_source(e.ADAPTER_PATCH)
    assert '(1, 1370, 768)' in pos
    assert '(batch * windows, 257, 768)' in patch
    assert 'num_features != 768' in patch


def test_prepare_validate_rejects_tamper(tmp_path):
    out = tmp_path / 'overlay'
    with contextlib.redirect_stdout(io.StringIO()):
        e.prepare(out)
    for arm in e.ARMS:
        p = e.validate_manifest(out / arm / 'manifest.json', out / arm)
        assert p['expected_args']['seed'] == (43 if arm == 'S43' else 42)
        assert p['expected_args']['epochs'] == 20
        assert p['expected_args']['k_eval'] == 36
        assert p['protocol_sha256'] == e.sha(out / arm / 'registered/EXP160_PROTOCOL.md')
        assert e.read(out / arm / 'queue.json') == e.queue(arm)
        assert e.sha(out / arm / 'tools/exp158_dino_strong.py') == p['files']['tools/exp158_dino_strong.py']['sha256']
        assert not (out / arm / 'admission.json').exists()
    base = out / 'BASE'
    p = e.read(base / 'manifest.json')
    assert p['initial_asset']['sha256'] is None
    assert p['files'][e.WEIGHT_BASE]['sha256'] is None
    with pytest.raises(ValueError):
        e.validate_manifest(base / 'manifest.json', base, full=True)
    (out / 'S43' / e.RUNNER).write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='file differs'):
        e.validate_manifest(out / 'S43/manifest.json', out / 'S43')


def test_preparation_never_launches_or_downloads(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('prepare must not launch or download')
    monkeypatch.setattr('subprocess.run', forbidden)
    monkeypatch.setattr(e, '_download_base', forbidden)
    with contextlib.redirect_stdout(io.StringIO()):
        e.prepare(tmp_path / 'overlay')


def test_registered_gates_require_positive_lower_ci():
    positive = {'mean_delta': .0001, 'mean_ci95': [.000001, .0003]}
    touching = {'mean_delta': .0001, 'mean_ci95': [0.0, .0003]}
    negative = {'mean_delta': -.0001, 'mean_ci95': [.000001, .0003]}
    assert e.screen_gate(positive, .005, positive) == (True, True)
    assert e.screen_gate(touching, .005, positive) == (False, True)
    assert e.screen_gate(negative, .005, positive) == (False, True)
    assert e.screen_gate(positive, .004999, positive) == (True, False)
    assert e.screen_gate(positive, .005, touching) == (True, False)


def test_vast_plan_only_prints_two_detached_lanes():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        e.vast_plan('/workspace/rsna-exp160-dino-partner-v1')
    text = out.getvalue()
    assert text.count('setsid nohup python tools/gpu_queue.py queue.json --foreground') == 2
    assert text.count('max_gpu_seconds=10800') == 2
    assert 'C:' not in text and '\\' not in text
    with pytest.raises(ValueError, match='Linux'):
        e.vast_plan('C:\\workspace\\exp160')


def test_five_fold_plan_checks_membership_and_prints_no_run(tmp_path, monkeypatch):
    root = tmp_path
    c = root / 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json'
    own = root / 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1'
    c.parent.mkdir(parents=True)
    c.write_text(json.dumps({'train': ['a'], 'val': ['v0']}), encoding='utf-8')
    for f in range(5):
        path = own / f'fold{f}/ordered-ids.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'train': ['a'], 'val': [f'v{f}']}), encoding='utf-8')
    for f in range(1, 5):
        for seed in (42, 43):
            family = 'EXP149' if seed == 42 and f == 1 else 'EXP153'
            ref = root / f'artifacts/experiments/{family}_H1_S{seed}_F{f}/oof_k36.csv'
            ref.parent.mkdir(parents=True)
            ref.write_text(f'StudyInstanceUID\nv{f}\n', encoding='utf-8')
    monkeypatch.setattr(e, 'ROOT', root)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        e.five_fold_plan('BASE')
    result = json.loads(out.getvalue())
    assert result['status'] == 'PLAN_ONLY_NO_RUN'
    assert [f['our_fold'] for f in result['remaining_folds']] == [1, 2, 3, 4]
    assert result['remaining_folds'][0]['h1_k36_seed42_43'][0].startswith('artifacts/experiments/EXP149')
    c.write_text(json.dumps({'train': ['different'], 'val': ['v0']}), encoding='utf-8')
    with pytest.raises(ValueError, match='does not map'):
        e.five_fold_plan('BASE')
