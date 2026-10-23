"""CPU-only EXP125 contract, gate and small synthetic scoring tests.

Run with system Python, OMP_NUM_THREADS=2, --basetemp .local/pytest-exp125
-p no:cacheprovider. Never rely on CUDA_VISIBLE_DEVICES. All executed tensors
and validation commands explicitly use CPU; a CUDA lazy-init tripwire is armed.
Prepared-protocol checks skip until `prepare`. Optional real-cache CPU smoke:
EXP125_CPU_SMOKE=1 (8 train studies / 2 val / 1 epoch / zero workers).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp125_radr50_k16_local as g


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    import torch
    assert not torch.cuda.is_initialized(), 'tests started after CUDA initialisation'

    def fail(*a, **kw):
        raise AssertionError('EXP125 CPU tests must never initialise CUDA')

    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)
    monkeypatch.setattr(torch.cuda, 'init', fail)
    yield
    assert not torch.cuda.is_initialized()


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def entry():
    import tools
    original_module = sys.modules['tools.exp107_kfold_radr50']
    original_attr = tools.exp107_kfold_radr50
    original_allow = g.ALLOW_PRODUCTION
    yield load_path('exp125_entry_test', ROOT / g.RUNNER)
    sys.modules['tools.exp107_kfold_radr50'] = original_module
    tools.exp107_kfold_radr50 = original_attr
    g.ALLOW_PRODUCTION = original_allow


def derived(seed):
    return g.derive_manifest(g.base_manifest(seed), seed, {}, {'test': True})


def test_registered_controls_and_only_k_diff():
    for seed in g.SEEDS:
        base, p = g.base_manifest(seed), derived(seed)
        assert p['seed'] == seed
        assert p['initial_state'] == base['initial_state']
        assert p['initial_state_sha256'] == base['initial_state_sha256']
        assert {k for k in p['common_args'] if p['common_args'][k] != base['common_args'][k]} == {'k'}
        assert (base['common_args']['k'], p['common_args']['k']) == (12, 16)
        assert p['environment'] == base['environment']
        for k, f in p['folds'].items():
            for field, val in g.fold_contract(f['n_train'], p['common_args']).items():
                assert f[field] == val
        for field in ('seed', 'initial_state_sha256', 'environment'):
            bad = copy.deepcopy(p)
            bad[field] = 'changed'
            with pytest.raises(ValueError, match='field changed'):
                g.assert_only_k_changed(bad, base)
        bad = copy.deepcopy(p)
        bad['common_args']['bs'] = 4
        with pytest.raises(ValueError, match='beyond'):
            g.assert_only_k_changed(bad, base)
        bad = copy.deepcopy(p)
        bad['folds']['0']['steps'] += 1
        with pytest.raises(ValueError, match='fold contract'):
            g.assert_only_k_changed(bad, base)


def test_unchanged_ordered_runner_and_guard_exports(entry):
    runner = entry.load_runner()
    assert Path(runner.__file__).resolve() == (ROOT / g.SHARED_RUNNER).resolve()
    base = g.base_manifest(42)
    assert g.sha256(runner.__file__) == base['source_bindings'][g.SHARED_RUNNER]
    source = Path(runner.__file__).read_text(encoding='utf-8')
    names = set(re.findall(r'guard\.([A-Za-z_]+)', source))
    assert all(hasattr(g, n) for n in names)
    for n in names - {'validate_arguments', 'LABEL_SET', '__file__'}:
        assert getattr(g, n) is getattr(g.e107, n)
    assert 'CenterSliceWindows(corpus.dir, train_ids, all_labels, a.res, a.k' in source
    for seed in g.SEEDS:
        p = derived(seed)
        a = g.runner_argv(p, python=sys.executable)
        b = g.e113.runner_argv(g.base_manifest(seed), 0, python=sys.executable)
        # Entry, protocol, tag, K are the entire ordered argv difference.
        for index in (1, a.index('--protocol') + 1, a.index('--tag') + 1, a.index('--k') + 1):
            b[index] = a[index]
        assert a == b
        allow, stripped = entry.split_options(a[2:])
        parsed = runner.build_parser().parse_args(stripped)
        assert not allow and parsed.k == 16 and parsed.seed == seed


def test_entry_options_and_smoke_cpu(entry):
    assert entry.split_options(['--label-set=dread', '--fold', '0']) == (False, ['--fold', '0'])
    for args in (['--label-set=flight_v4'], ['--label-set=dread', '--label-set=dread'],
                 ['--allow-production-after-screen-pass'] * 2):
        with pytest.raises(SystemExit):
            entry.split_options(args)
    a = g.runner_argv(derived(42), mode='smoke', workers=0, output_root='.local/exp125-smoke-test')
    for flag, value in (('--device', 'cpu'), ('--amp', 'none'), ('--eval-amp', 'none'), ('--workers', '0')):
        assert a[a.index(flag) + 1] == value


def test_default_queue_and_production_guard(monkeypatch):
    monkeypatch.setattr(g, 'load_protocol', lambda seed, root: derived(seed))
    q = g.queue_entries(after='CURRENT_TAIL')
    assert len(q) == 2 and all(e['continue_on_failure'] is True for e in q)
    assert [e['tag'] for e in q] == [g.tag_for(42), g.tag_for(43)]
    assert q[0]['after'] == 'CURRENT_TAIL' and q[1]['after'] == q[0]['tag']
    q = g.queue_entries(parallel_seeds=True)
    assert all('after' not in e for e in q)
    with pytest.raises(ValueError, match='explicit|need'):
        g.queue_entries(folds=(1,))
    calls = []
    monkeypatch.setattr(g, 'require_screen_pass', lambda root: calls.append(root))
    q = g.queue_entries(folds=(1,), allow_production=True)
    assert len(calls) == 2
    assert all('--allow-production-after-screen-pass' in e['command'] for e in q)


def test_validate_cli_forces_cpu(monkeypatch):
    monkeypatch.setattr(g, 'load_protocol', lambda seed: derived(seed))
    called = []

    def run(argv, **kw):
        called.append(argv)
        return type('Result', (), {'returncode': 0})()

    monkeypatch.setattr(g.subprocess, 'run', run)
    assert g.validate_cli(42) == 0
    a = called[0]
    assert a[a.index('--device') + 1] == 'cpu' and '--validate-only' in a


def test_four_source_consensus_exact_half_and_missing():
    soft = {n: np.tile(np.array([[0.1], [0.9], [0.5], [0.9]]), (1, 12)) for n in g.LABEL_PATHS}
    soft['steven_v4'][3, 0] = np.nan
    soft['pilkwang_v2'][1, 1] = 0.1
    refs = g.make_references(soft)
    yd, _ = refs['dread']
    yc, mask = refs['consensus']
    assert yd[2, 0] == 0 and yc[2, 0] == 1 and mask[2, 0]
    assert not mask[3, 0] and not mask[1, 1]
    # Flight is not a member of four-source consensus.
    soft['flight_v4'] = 1 - soft['flight_v4']
    assert np.array_equal(g.make_references(soft)['consensus'][1], mask)


def test_component_draws_keep_members_together():
    ids = ['a', 'b', 'c', 'd']
    comps = [dict(component_id='x', members=['a', 'b']), dict(component_id='y', members=['c', 'd'])]
    draws = g.component_draws(ids, comps, draws=20, seed=9)
    assert all(np.sum(d == 0) == np.sum(d == 1) and np.sum(d == 2) == np.sum(d == 3) for d in draws)
    assert any(len(set(d)) == 2 for d in draws)
    assert all(np.array_equal(a, b) for a, b in zip(draws, g.component_draws(ids, comps, draws=20, seed=9)))
    with pytest.raises(ValueError, match='missing'):
        g.component_draws(ids + ['e'], comps)
    with pytest.raises(ValueError, match='duplicate'):
        g.component_draws(ids, comps + [comps[0]])


def test_mean_auc_not_auc_of_mean_and_paired_bootstrap():
    # One seed gets .75, another .50; averaging probabilities would give .75.
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 12))
    ref = (y, np.ones(y.shape, bool))
    a = {42: np.tile(np.array([.1, .4, .35, .8])[:, None], (1, 12)),
         43: np.tile(np.array([.4, .3, .2, .5])[:, None], (1, 12))}
    b = {s: np.full(y.shape, .5) for s in g.SEEDS}
    draws = [np.arange(4), np.array([0, 1, 2, 3, 2, 3])]
    per, mean, spread = g.compare_seeds(a, b, ref, draws)
    assert mean['candidate'] == pytest.approx(.625)
    assert mean['delta'] == pytest.approx(.125)
    assert np.mean(g.auc_vector((a[42] + a[43]) / 2, ref)) != mean['candidate']
    expected = [np.mean([np.nanmean(g.auc_vector(a[s], ref, d)) - np.nanmean(g.auc_vector(b[s], ref, d)) for s in g.SEEDS]) for d in draws]
    assert mean['ci95'] == pytest.approx(np.percentile(expected, [2.5, 97.5]))
    assert mean['bootstrap_finite_draws'] == len(draws)
    assert spread['control'] == 0
    assert per['42']['delta'] == pytest.approx(.25)


def test_gate_boundaries_missing_and_nan():
    values = {n: dict(delta=0.0, ci95=[-0.01, 0.01]) for n in g.REFERENCES}
    values['dread'] = dict(delta=.004, ci95=[.000001, .01])
    values['flight_v4']['delta'] = values['relabel_v1']['delta'] = -.002
    assert g.gate(values)['verdict'] == 'PASS'
    for n, delta in [('dread', .0039999), ('consensus', -.000001), ('flight_v4', -.002001), ('relabel_v1', -.002001)]:
        bad = copy.deepcopy(values)
        bad[n]['delta'] = delta
        assert g.gate(bad)['verdict'] == 'FAIL'
    bad = copy.deepcopy(values)
    bad['dread']['ci95'][0] = 0
    assert g.gate(bad)['verdict'] == 'FAIL'
    bad['dread']['delta'] = float('nan')
    assert g.gate(bad)['verdict'] == 'INVALID'
    assert g.gate({})['verdict'] == 'INVALID'


def test_sparse_target_matches_exp117_nanmean():
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 12))
    y[:, 0] = 0
    ref = (y, np.ones(y.shape, bool))
    a = {s: np.full(y.shape, .5) for s in g.SEEDS}
    per, mean, _ = g.compare_seeds(a, a, ref, [np.arange(4)])
    assert mean['delta'] == 0 and mean['candidate'] == .5
    assert per['42']['scored_targets'] == 11 and mean['target_deltas'][g.T[0]] is None
    json.dumps(mean, allow_nan=False)


def test_run_verification_catches_recipe_environment_and_inventory(tmp_path):
    p = derived(42)
    p['source_bindings'][g.TOOL] = 'tool-sha'
    f = p['folds']['0']
    run = tmp_path / f['tag']
    run.mkdir()
    prov = dict(fold=0, seed=42, training_mode='full', checkpoint_selection='fixed-final',
        protocol_sha256='manifest-sha', label_set='dread', label_file_sha256=p['label_set']['sha256'],
        label_semantic_sha256=p['label_set']['semantic_sha256'], arch=p['common_args']['arch'],
        train_ids_ordered_sha256=f['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=f['ordered_ids']['val_sha256'],
        initial_state_sha256=p['initial_state_sha256'], initial_state_matches_registered=True,
        environment_matches_baseline=True, environment=p['environment'], remote_relaxed=False,
        input_policy='center3', norm='radimagenet', k_eval=24, k_eval_extra=36,
        runner_sha256=p['source_bindings'][g.SHARED_RUNNER], guard_sha256='tool-sha',
        init_weights={'weights_sha256': g.RADIMAGENET_SHA256}, actual_pos_weight=f['pos_weight'])
    summary = dict(args={**p['common_args'], 'seed': 42, 'tag': f['tag']}, mode='full',
        selected_epoch_1based=20, train=f['n_train'], val=f['n_val'], optimizer_updates=f['steps'], skipped_updates=0)
    files = ('summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv')

    def write_fixture():
        # All files here were created by this test in its temporary directory.
        for name in files:
            if name == 'summary.json':
                content = json.dumps(summary)
            elif name == 'provenance.json':
                content = json.dumps(prov)
            else:
                content = 'synthetic fixture'
            (run / name).write_text(content, encoding='utf-8')
        (run / 'output-inventory.json').write_text(json.dumps({n: {'sha256': g.sha256(run / n)} for n in files}), encoding='utf-8')

    write_fixture()
    assert len(g.verify_run(run, p, 'manifest-sha', g.TOOL, tmp_path)) == 7
    summary['args']['k'] = 12
    write_fixture()
    with pytest.raises(ValueError, match='recipe'):
        g.verify_run(run, p, 'manifest-sha', g.TOOL, tmp_path)
    summary['args']['k'] = 16
    prov['environment'] = {**p['environment'], 'torch': 'different'}
    write_fixture()
    with pytest.raises(ValueError, match='environment'):
        g.verify_run(run, p, 'manifest-sha', g.TOOL, tmp_path)
    prov['environment'] = p['environment']
    write_fixture()
    (run / 'oof_k36.csv').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory'):
        g.verify_run(run, p, 'manifest-sha', g.TOOL, tmp_path)


def test_prediction_roster_and_probability_checks(tmp_path):
    ids = ['a', 'b', 'c', 'd']
    f = tmp_path / 'pred.csv'
    d = pd.DataFrame(np.full((4, 12), .5), columns=g.T)
    d.insert(0, g.UID, ids)
    d.iloc[::-1].to_csv(f, index=False)
    assert g.load_table(f, ids, True).shape == (4, 12)
    for bad in (pd.concat([d, d.iloc[:1]]), d.iloc[:3]):
        bad.to_csv(f, index=False)
        with pytest.raises(ValueError):
            g.load_table(f, ids, True)
    d.loc[0, g.T[0]] = float('nan')
    d.to_csv(f, index=False)
    with pytest.raises(ValueError, match='probabilities'):
        g.load_table(f, ids, True)


def test_immutable_outputs_and_unpassed_production(tmp_path):
    dest = tmp_path / 'exists'
    dest.mkdir()
    with pytest.raises(SystemExit, match='existing'):
        g.prepare(dest=dest)
    report_dir = tmp_path / g.SCREEN_DIR
    report_dir.mkdir(parents=True)
    g.dump(report_dir / 'result.json', dict(schema=g.SCREEN_SCHEMA, status='INCOMPLETE', gate={'verdict': 'PASS'}))
    with pytest.raises(ValueError, match='no matching'):
        g.require_screen_pass(tmp_path)
    with pytest.raises(SystemExit, match='existing'):
        g.screen(tmp_path)


@pytest.mark.skipif(not (ROOT / g.PROTOCOL / 'manifest.json').is_file(), reason='run prepare first')
def test_prepared_guard_accepts_k16_rejects_k12_cpu_validation(entry, monkeypatch):
    for k, v in {**g.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    parser = entry.load_runner().build_parser()
    for seed in g.SEEDS:
        p = g.load_protocol(seed)
        argv = g.runner_argv(p, workers=0)
        argv = g._set_flag(argv, '--device', 'cpu') + ['--validate-only']
        _, rest = entry.split_options(argv[2:])
        args = parser.parse_args(rest)
        view, fold, ordered, out = g.validate_arguments(args, ROOT)
        assert view['common_args']['k'] == 16 and len(ordered['val']) == 888
        bad = copy.copy(args)
        bad.k = 12
        with pytest.raises(ValueError, match='frozen argument.*k'):
            g.validate_arguments(bad, ROOT)
        bad = copy.copy(args)
        bad.validate_only = False
        with pytest.raises(ValueError, match='frozen argument.*device'):
            g.validate_arguments(bad, ROOT)


@pytest.mark.skipif(os.environ.get('EXP125_CPU_SMOKE') != '1', reason='optional real-cache CPU smoke')
def test_tiny_cpu_smoke_in_process(entry, tmp_path):
    # In-process tripwire catches CUDA initialisation. No CUDA_VISIBLE_DEVICES.
    p = g.load_protocol(42)
    argv = g.runner_argv(p, mode='smoke', workers=0, output_root=tmp_path,
        smoke=dict(device='cpu', amp='none', eval_amp='none', train=8, val=2, epochs=1))
    old = dict(os.environ)
    try:
        os.environ.update({**g.THREAD_ENV, 'HF_HUB_OFFLINE': '1'})
        os.environ.pop('KNEE_REMOTE_RELAX_ENV', None)
        entry.main(argv[2:])
    finally:
        os.environ.clear()
        os.environ.update(old)
    s = g.read(tmp_path / g.tag_for(42, mode='smoke') / 'summary.json')
    assert s['args']['device'] == 'cpu' and s['args']['k'] == 16
    assert s['train'] == 8 and s['val'] == 2 and s['optimizer_updates'] == 1
    assert s['skipped_updates'] == 0
    assert s['provenance']['initial_state_matches_registered'] is True
