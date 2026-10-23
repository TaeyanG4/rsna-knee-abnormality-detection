"""MIL screen v1.0.0: numerical parity, immutable replay and real CPU cards.

All label/ID arrays remain in memory. CLI evidence contains aggregates only.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from training.arms import lifecycle as life
from training.arms import mil_screen as screen
from training.arms import mil_screen_replay as replay
from tools import exp115_vlm_eval as ev
from tools import exp185_finding_mil as legacy_seed
from tools import exp186_finding_mil_5fold as legacy_fold
from tools.exp162_dino_folds_kaggle import TARGETS

ROOT = Path(__file__).resolve().parents[1]


def sample(n=20):
    rng = np.random.default_rng(8371)
    y = np.tile(np.arange(n)[:, None] % 2, (1, 12)).astype(float)
    refs = {k: (y.copy(), np.ones_like(y, dtype=bool), 1)
            for k in ('jev', 'aquila', 'dread', 'consensus')}
    values = {f: {k: rng.random((n // 5, 12))
              for k in ('candidate', 'control', 'rad42', 'rad43', 'dino')}
              for f in range(5)}
    return values, refs, [np.arange(n), np.arange(n)[::-1]]


def seeds():
    _, refs, draws = sample(20)
    rng = np.random.default_rng(8388)
    pair = lambda: {s: rng.random((20, 12)) for s in (42, 43)}
    return pair(), pair(), pair(), rng.random((20, 12)), refs, draws


def test_pooled_screen_exact_legacy_parity_and_json_types():
    values, refs, draws = sample()
    actual = screen.score_folds(values, refs, draws, ev)
    assert actual == legacy_fold.score_arrays(values, refs, draws, ev)
    assert life.parse_json(life.canonical_bytes(actual)) == actual


def test_seed_primary_is_mean_auc_not_auc_of_mean_predictions():
    candidate, control, rad, dino, refs, draws = seeds()
    actual = screen.score_seeds(candidate, control, rad, dino, refs, draws, ev)
    expected = screen.validation._paired_seed_mean(candidate, control, refs, draws, ev)
    assert actual['all_targets_and_references'] == expected
    wrong = screen.paired_jev_consensus((candidate[42] + candidate[43]) / 2,
        (control[42] + control[43]) / 2, refs, draws, ev)
    assert expected['primary_mean_delta'] != wrong['primary_mean_delta']
    old_weak = legacy_seed.weak_target_readout(np.stack(list(candidate.values()), axis=2),
        np.stack(list(control.values()), axis=2), refs, draws, screen.validation._MeanSeedEvaluator(ev))
    assert actual['weak_target_readout'] == old_weak
    assert life.parse_json(life.canonical_bytes(actual)) == actual


def test_rank_boundary_is_within_fold_then_fixed_bundle(monkeypatch):
    values, refs, draws = sample()
    calls = []
    original = screen.bundle.bundle_of
    def observe(parts):
        calls.append({k: v.copy() for k, v in parts.items()})
        return original(parts)
    monkeypatch.setattr(screen.bundle, 'bundle_of', observe)
    screen.score_folds(values, refs, draws, ev)
    assert len(calls) == 2
    for index, arm in enumerate(('candidate', 'control')):
        expected = np.concatenate([ev.pct_rank(values[f][arm]) for f in range(5)])
        np.testing.assert_array_equal(calls[index]['convnext'], expected)
        wrong = ev.pct_rank(np.concatenate([values[f][arm] for f in range(5)]))
        assert not np.array_equal(expected, wrong)
    expected_rad = np.concatenate([(ev.pct_rank(values[f]['rad42']) +
                                   ev.pct_rank(values[f]['rad43'])) / 2 for f in range(5)])
    np.testing.assert_array_equal(calls[0]['l2'], expected_rad)
    np.testing.assert_array_equal(calls[0]['l2'], calls[1]['l2'])


def test_undefined_weak_bootstrap_preserves_pooled_hold_and_seed_rejection():
    values, refs, draws = sample()
    t = TARGETS.index('Synovitis')
    for y, mask, _ in refs.values():
        mask[:, t] = False
    actual = screen.score_folds(values, refs, draws, ev)
    assert actual == legacy_fold.score_arrays(values, refs, draws, ev)
    assert actual['weak_target_readout']['primary_ci95'] is None
    assert actual['status'] == 'HOLD_REVIEW'
    candidate, control, rad, dino, _, _ = seeds()
    with pytest.raises(ValueError, match='undefined weak-target'):
        screen.score_seeds(candidate, control, rad, dino, refs, draws, ev)


def test_weak_harm_guard_and_image_review_match_legacy():
    class Toy:
        @staticmethod
        def auc_targets(y, p, mask, minimum):
            return (p[0] - p[1]) / 10
        @staticmethod
        def macro(a):
            return float(np.mean(a))
    _, refs, draws = sample()
    control = np.full((20, 12), .5)
    candidate = control.copy()
    for name in ('Lateral Meniscus', 'PF OA', 'Lateral OA', 'Synovitis'):
        candidate[0, TARGETS.index(name)] += .1
    draws = [draws[0]] * 2
    actual = screen.weak_target_readout(candidate, control, refs, draws, Toy())
    assert actual == legacy_seed.weak_target_readout(candidate, control, refs, draws, Toy())
    assert actual['numeric_gate'] == 'ADVANCE' and actual['status'] == 'REVIEW_IMAGE_READ'
    candidate[1, TARGETS.index('ACL')] += .2
    harm = screen.weak_target_readout(candidate, control, refs, draws, Toy())
    assert not harm['harm_guard']['other_targets_ge_minus_0_010']
    assert harm['status'] == 'HOLD_REVIEW'


@pytest.mark.parametrize('bad', ['fold', 'family', 'shape', 'nan', 'range', 'mask', 'labels',
                                      'minimum', 'draw_float', 'draw_range', 'draw_empty'])
def test_pooled_contract_rejects_bad_inputs(bad):
    values, refs, draws = sample()
    if bad == 'fold':
        del values[4]
    elif bad == 'family':
        del values[4]['dino']
    elif bad == 'shape':
        values[4]['control'] = np.zeros((3, 12))
    elif bad in ('nan', 'range'):
        values[4]['candidate'][0, 0] = np.nan if bad == 'nan' else 1.1
    elif bad == 'mask':
        y, m, n = refs['jev']; refs['jev'] = (y, m.astype(int), n)
    elif bad == 'labels':
        refs['jev'][0][0, 0] = .25
    elif bad == 'minimum':
        y, m, n = refs['jev']; refs['jev'] = (y, m, True)
    elif bad == 'draw_float':
        draws = [np.arange(20).astype(float)]
    elif bad == 'draw_range':
        draws = [np.arange(21)]
    else:
        draws = []
    with pytest.raises(ValueError):
        screen.score_folds(values, refs, draws, ev)


@pytest.mark.parametrize('part', ['candidate', 'control', 'rad'])
def test_seed_pair_is_exact(part):
    candidate, control, rad, dino, refs, draws = seeds()
    del {'candidate': candidate, 'control': control, 'rad': rad}[part][43]
    with pytest.raises(ValueError, match='exactly integer seeds'):
        screen.score_seeds(candidate, control, rad, dino, refs, draws, ev)


@pytest.mark.parametrize('value', ['../escape.csv', 'a/../escape.csv', 'C:/foreign/a.csv',
                                  '/foreign/a.csv', r'\\server\share\a.csv', 'a:stream.csv'])
def test_portable_path_rejects_escape(value):
    with pytest.raises(ValueError):
        replay.portable_path(value, ROOT)


def test_windows_path_migration_and_link_rejection(tmp_path, monkeypatch):
    rel = 'artifacts/experiments/example.csv'
    assert replay.portable_path(rel.replace('/', '\\'), ROOT) == rel
    assert replay.portable_path(str(ROOT / rel), ROOT) == rel
    assert replay.path(ROOT, rel) == ROOT / rel
    original = Path.is_junction
    monkeypatch.setattr(Path, 'is_junction', lambda p: p.name == 'link' or original(p))
    with pytest.raises(ValueError, match='linked input'):
        replay.path(tmp_path, 'link/file.csv')


@pytest.fixture(scope='module', params=list(replay.HISTORIES))
def real_draft(request):
    kind = request.param
    return replay.draft(ROOT, kind, replay.hashed(ROOT, replay.HISTORIES[kind]))


def test_real_context_and_registration_bindings(real_draft):
    spec = real_draft
    assert spec['studies'] == (888 if spec['kind'] == 'mean-seed-f0' else 4343)
    assert len(spec['predictions']) == (7 if spec['studies'] == 888 else 25)
    assert spec['bootstrap']['draws'] == 300 and len(spec['draws_sha256']) == 64
    assert all('\\' not in row['path'] and not Path(row['path']).is_absolute()
               for row in spec['predictions'].values())
    assert spec['launch_authorized'] is False
    assert replay.validate(ROOT, spec) == spec


@pytest.mark.parametrize('field', ['sources', 'context_inputs', 'draws_sha256', 'predictions',
                                  'scientific_status', 'launch_authorized'])
def test_modified_registration_rejected(real_draft, field):
    spec = copy.deepcopy(real_draft)
    spec[field] = {} if isinstance(spec[field], dict) else 'changed'
    with pytest.raises(ValueError, match='registration differs'):
        replay.validate(ROOT, spec)


@pytest.mark.parametrize('which', ['history', 'prediction', 'source', 'context'])
def test_changed_bound_file_rejected_without_mutating_originals(real_draft, which, monkeypatch):
    spec = real_draft
    selected = {'history': spec['historical']['path'], 'source': replay.SOURCES[0],
                'prediction': next(iter(spec['predictions'].values()))['path'],
                'context': next(iter(spec['context_inputs']))}[which]
    original = replay.hashed
    monkeypatch.setattr(replay, 'hashed', lambda root, p: '0' * 64 if p == selected else original(root, p))
    with pytest.raises(ValueError):
        replay.validate(ROOT, spec)


def test_wrong_history_digest_rejected():
    with pytest.raises(ValueError, match='historical receipt differs'):
        replay.draft(ROOT, 'mean-seed-f0', '0' * 64)


def test_screen_revalidates_after_scoring_and_never_publishes_partial(real_draft, tmp_path, monkeypatch):
    historical = replay.read(ROOT / real_draft['historical']['path'])
    keys = ('status', 'weak_target_readout', 'all_targets_and_references', 'per_seed',
            'fixed_bundle_replacement_diagnostic') if real_draft['kind'] == 'mean-seed-f0' else (
            'direct', 'weak_target_readout', 'fixed_bundle_replacement_diagnostic', 'per_fold_direct', 'status')
    fn = 'score_seeds' if real_draft['kind'] == 'mean-seed-f0' else 'score_folds'
    numerical = {k: historical[k] for k in keys}
    monkeypatch.setattr(screen, fn, lambda *a: numerical)
    original = replay.validate
    calls = []
    def changed(root, spec):
        calls.append(1)
        if len(calls) == 2:
            raise life.ContractError('changed during scoring')
        return original(root, spec)
    monkeypatch.setattr(replay, 'validate', changed)
    out = tmp_path / 'result.json'
    with pytest.raises(ValueError, match='changed during scoring'):
        replay.screen(ROOT, real_draft, out)
    assert len(calls) == 2 and not out.exists()


@pytest.mark.parametrize('kind', list(replay.HISTORIES))
def test_real_cpu_cli_replay_card(kind, tmp_path):
    base = Path(os.environ.get('MIL_SCREEN_SMOKE_OUT', tmp_path)) / kind
    base.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    records = []
    def cli(args, expected, status):
        argv = [sys.executable, '-B', '-m', 'tools.mil_screen', *args, '--root', str(ROOT)]
        proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        log = base / f'command-{len(records):02d}.log'
        life.create_new(log, (proc.stdout + proc.stderr).encode())
        records.append(dict(argv=argv, returncode=proc.returncode, log=log.name, sha256=life.file_sha256(log)))
        life.create_new(base / f'command-{len(records)-1:02d}.json', life.canonical_bytes(records[-1]))
        # Only aggregate/sanitized output is retained or asserted, never input rows.
        assert proc.returncode == expected, f'CPU CLI failed; see {log}'
        output = life.parse_json(proc.stdout)
        assert output['status'] == status and not proc.stderr
        if expected == 0:
            assert output['launch_authorized'] is False
    spec = str(base / 'spec.json')
    result = str(base / 'result.json')
    digest = replay.hashed(ROOT, replay.HISTORIES[kind])
    registration = ['register-replay', '--kind', kind, '--historical-sha256', digest, '--out', spec]
    cli(registration, 0, 'REGISTERED_REPLAY')
    cli(['validate', '--spec', spec], 0, 'VALIDATED_REPLAY')
    cli(['screen', '--spec', spec, '--out', result], 0, 'PARITY_VERIFIED')
    before = {p: life.file_sha256(base / p) for p in ('spec.json', 'result.json')}
    cli(registration, 2, 'REJECTED')
    cli(['screen', '--spec', spec, '--out', result], 2, 'REJECTED')
    tampered = replay.read(spec); tampered['draws_sha256'] = '0' * 64
    life.create_new(base / 'tampered.json', life.canonical_bytes(tampered))
    cli(['validate', '--spec', str(base / 'tampered.json')], 2, 'REJECTED')
    assert before == {p: life.file_sha256(base / p) for p in before}
    output = replay.read(result)
    historical = replay.read(ROOT / replay.HISTORIES[kind])
    assert output['numerical'] == {k: historical[k] for k in output['numerical']}
    assert output['status'] == 'PARITY_VERIFIED'
    life.create_new(base / 'command-results.json', life.canonical_bytes(records))
