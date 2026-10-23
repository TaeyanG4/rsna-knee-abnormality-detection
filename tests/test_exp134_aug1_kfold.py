"""EXP134 protocol and arithmetic tests; CPU only, no CUDA initialization."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp134_aug1_kfold as g


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def make_fail():
        def fail(*args, **kwargs):
            raise AssertionError('EXP134 tests must never initialize CUDA')
        return fail
    for name in ('_lazy_init', 'init', 'get_rng_state', 'manual_seed_all'):
        monkeypatch.setattr(torch.cuda, name, make_fail())
    yield
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('seed', g.SEEDS)
def test_all_fold_contracts_equal_matched_control(seed):
    base = g.e127.base_manifest(seed)
    f0 = g.e127.load_protocol('AUG1', seed)
    p = g.derive_manifest(base, f0, seed, {}, {})
    assert p['common_args'] == base['common_args']
    assert p['variant_config'] == g.e127.CONFIG['AUG1']
    assert p['folds']['0'] == f0['folds']['0']
    for k in range(1, 5):
        got, control = p['folds'][str(k)], base['folds'][str(k)]
        assert {n: v for n, v in got.items() if n != 'tag'} == {n: v for n, v in control.items() if n != 'tag'}
        assert got['tag'] == g.tag_for(seed, k)
        assert got['steps'] == got['scheduler_total_steps']
        assert got['max_fp16_skips'] == got['steps'] // 100
        assert got['eval_batch'] == base['folds'][str(k)]['eval_batch']
        assert got['ordered_ids'] == control['ordered_ids']
        assert got['pos_weight'] == control['pos_weight']
    bad = copy.deepcopy(p)
    bad['folds']['2']['pos_weight'][0] += .01
    with pytest.raises(ValueError, match='contract'):
        g.assert_single_change(bad, base, f0)
    bad = copy.deepcopy(p)
    bad['folds']['3']['steps'] += 1
    with pytest.raises(ValueError, match='contract'):
        g.assert_single_change(bad, base, f0)


def test_queue_order_and_cpu_validation_argv(monkeypatch):
    p = g.derive_manifest(g.e127.base_manifest(42), g.e127.load_protocol('AUG1', 42), 42, {}, {})
    q = g.derive_manifest(g.e127.base_manifest(43), g.e127.load_protocol('AUG1', 43), 43, {}, {})
    monkeypatch.setattr(g, 'load_protocol', lambda seed, root=g.ROOT: p if seed == 42 else q)
    entries = g.queue_entries('PRIOR_TAIL')
    assert len(entries) == 8
    assert [x['tag'] for x in entries] == [g.tag_for(s, k) for s in g.SEEDS for k in range(1, 5)]
    assert entries[0]['after'] == 'PRIOR_TAIL'
    assert all(entries[i]['after'] == entries[i-1]['tag'] for i in range(1, 8))
    assert all(x['continue_on_failure'] and x['command'][x['command'].index('--workers') + 1] == '12' for x in entries)
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        return type('Return', (), {'returncode': 0})()
    monkeypatch.setattr(g.subprocess, 'run', fake_run)
    assert g.validate_cli(seed=42, fold=2) == 0
    assert '--validate-only' in calls[0]
    assert calls[0][calls[0].index('--device') + 1] == 'cpu'
    with pytest.raises(ValueError, match='folds 1-4'):
        g.tag_for(42, 0)
    with pytest.raises(ValueError, match='workers'):
        g.queue_entries('TAIL', workers=13)


def test_tier_a_b_boundaries_and_two_other_references():
    mean = {n: dict(delta=0., ci95=[-.01, .01], positive_draw_share=.95,
                    bootstrap_draws=300, bootstrap_finite_draws=300) for n in g.REFERENCES}
    per = {str(s): {'dread': {'delta': .002}} for s in g.SEEDS}
    mean['dread'].update(delta=.004, ci95=[.000001, .01])
    assert g.tier_verdicts(mean, per)['tier_a']['verdict'] == 'THRESHOLDS_MET'
    mean['dread'].update(delta=.001, ci95=[-.001, .004], positive_draw_share=.90)
    mean['consensus']['delta'] = .0001
    mean['flight_v4']['delta'] = .0001
    mean['relabel_v1']['delta'] = -.001
    assert g.tier_verdicts(mean, per)['tier_b']['verdict'] == 'THRESHOLDS_MET'
    assert g.tier_verdicts(mean, per)['tier_b']['adoption_authorized'] is False
    mean['flight_v4']['delta'] = 0
    assert g.tier_verdicts(mean, per)['tier_b']['verdict'] == 'THRESHOLDS_NOT_MET'
    mean['flight_v4']['delta'] = .0001
    per['43']['dread']['delta'] = 0
    assert g.tier_verdicts(mean, per)['tier_b']['verdict'] == 'THRESHOLDS_NOT_MET'


def test_paired_bootstrap_uses_identical_draws_and_two_seed_mean():
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 12))
    ref = y, np.ones_like(y, dtype=bool)
    good = {42: np.tile(np.array([.1, .3, .6, .9])[:, None], (1, 12)),
            43: np.tile(np.array([.2, .1, .7, .8])[:, None], (1, 12))}
    flat = {s: np.full((4, 12), .5) for s in g.SEEDS}
    draws = [np.arange(4), np.array([0, 1, 2, 3, 2, 3])]
    per, mean, _ = g.e127.compare_seeds(good, flat, ref, draws)
    assert mean['delta'] == pytest.approx(np.mean([per[str(s)]['delta'] for s in g.SEEDS]))
    assert mean['positive_draw_share'] == 1


def test_registered_protocol_and_runner_validate_cpu_only():
    path = ROOT / g.PROTOCOL / 'manifest.json'
    if not path.is_file():
        pytest.skip('run prepare first')
    registry = json.loads(path.read_text(encoding='utf-8'))
    assert registry['schema'] == g.SCHEMA
    for seed in g.SEEDS:
        p = g.load_protocol(seed)
        assert len(p['folds']) == 5
        assert p['derived_from']['f0_run_inventory']
