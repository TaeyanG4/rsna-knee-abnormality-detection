"""CPU metadata and synthetic screen tests; no full models, CUDA or real labels."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp139_aug1_variants as g


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def make_fail(name):
        def fail(*args, **kwargs):
            raise AssertionError('CUDA forbidden: ' + name)
        return fail
    for name in ('_lazy_init', 'init', 'get_rng_state', 'manual_seed_all'):
        monkeypatch.setattr(torch.cuda, name, make_fail(name))
    with torch.device('cpu'):
        yield
    assert not torch.cuda.is_initialized()


def derived(variant, seed=42):
    # Read registration JSON only, never weights, corpus or any label table.
    base = g.read(g.control_path(seed))
    return g.derive_manifest(base, variant, seed, {}, {'synthetic': True})


def entry_module():
    spec = importlib.util.spec_from_file_location('exp139_entry_test', ROOT / g.RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('variant', g.VARIANTS)
@pytest.mark.parametrize('seed', g.SEEDS)
def test_only_exp129_change_on_aug1(variant, seed):
    base, p = g.read(g.control_path(seed)), derived(variant, seed)
    assert base['folds']['0']['tag'] == f'EXP127_AUG1_S{seed}_F0'
    assert p['common_args'] == base['common_args']
    assert p['variant_config'] == g.e129.CONFIG[variant]
    assert p['augmentation_config'] == g.e127.CONFIG['AUG1'] == base['variant_config']
    assert p['derived_from']['changed_args'] == {}
    assert p['derived_from']['sha256'] == g.sha256(g.control_path(seed))
    for key in ('initial_state', 'initial_state_sha256', 'environment', 'training_code_sha256', 'label_set'):
        assert p[key] == base[key]
    assert set(p['folds']) == {'0'}
    assert p['folds']['0'] == {**base['folds']['0'], 'tag': g.tag_for(variant, seed)}
    assert p['total_steps'] == base['folds']['0']['steps']
    for field, key, value in [('common_args', 'bb_lr', .987), ('common_args', 'epochs', 40),
                               ('augmentation_config', 'gamma', [1, 1]),
                               ('adapter_initial_state', 'combined_sha256', 'tampered')]:
        bad = copy.deepcopy(p)
        bad[field][key] = value
        with pytest.raises(ValueError):
            g.assert_single_change(bad, base)
    for key in ('seed', 'initial_state_sha256', 'environment'):
        bad = copy.deepcopy(p)
        bad[key] = 'tampered'
        with pytest.raises((KeyError, ValueError)):
            g.assert_single_change(bad, base)


def section(text, start, end):
    at = text.index(start)
    return text[at:text.index(end, at)]


def test_frozen_optimizer_training_evaluation_path():
    old = (ROOT / g.e129.RUNNER).read_text(encoding='utf-8')
    new = (ROOT / g.RUNNER).read_text(encoding='utf-8')
    # The EXP129 loop is byte-for-byte reused, including LATE observer placement.
    for start, end in [('    smoke = a.mode', '    def init_record'),
                       ('    if a.grad_ckpt:', '    provenance["cuda_initialized"]')]:
        # Only provenance names in checkpoint metadata differ in the long block.
        normalized = section(new, start, end).replace('exp139_variant', 'exp129_variant')
        assert normalized == section(old, start, end)
    control = (ROOT / g.e127.RUNNER).read_text(encoding='utf-8')
    train = section(new, '    for ep in range(epochs):', '        peak =')
    train = train.replace('move_input(x, dev, non_blocking=True)', 'x.to(dev, non_blocking=True)')
    train = train.replace("            if late is not None:\n                late.update(model, successful=not (a.amp == 'fp16' and dev == 'cuda' and scaler.get_scale() < scale_before))\n", '')
    assert train == section(control, '    for ep in range(epochs):', '        peak =')
    assert new.index('supply_registered_env()  # before') < new.index('import numpy as np')
    for path in (*g.NEW_CODE, *g.TESTS):
        ast.parse((ROOT / path).read_text(encoding='utf-8'), filename=path)


def test_exact_aug1_arguments_and_thread_environment():
    entry = entry_module()
    env = dict(OMP_NUM_THREADS='2', MKL_NUM_THREADS='8', NUMEXPR_NUM_THREADS='9')
    entry.supply_registered_env(env)
    assert all(env[k] == '1' for k in g.THREAD_ENV) and env['HF_HUB_OFFLINE'] == '1'
    for variant in g.VARIANTS:
        p = derived(variant)
        args = g.runner_argv(p, python=sys.executable)
        parsed = entry.build_parser().parse_args(args[2:])
        assert (parsed.variant, parsed.seed, parsed.fold) == (variant, 42, 0)
        for key, value in p['common_args'].items():
            assert getattr(parsed, key) == value
    for dropped in ('LRH', 'LRL'):
        with pytest.raises(ValueError):
            g.tag_for(dropped, 42)


@pytest.mark.parametrize('variant', g.VARIANTS)
def test_serial_queue_and_both_seed_cpu_validate(variant, monkeypatch):
    recipes = {s: derived(variant, s) for s in g.SEEDS}
    monkeypatch.setattr(g, 'load_protocol', lambda v, s, root: recipes[s])
    items = g.queue_entries(variant, after='CURRENT_GPU_TAG')
    assert len(items) == 2 and items[0]['after'] == 'CURRENT_GPU_TAG'
    assert items[1]['after'] == items[0]['tag']
    assert [x['tag'] for x in items] == [g.tag_for(variant, s) for s in (42, 43)]
    assert all(x['command'][x['command'].index('--fold') + 1] == '0' for x in items)
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(g.subprocess, 'run', fake_run)
    assert g.validate_cli(variant) == 0
    assert [c[c.index('--seed') + 1] for c in calls] == ['42', '43']
    assert all(c[0] == sys.executable and c[c.index('--device')+1] == 'cpu' and '--validate-only' in c for c in calls)
    with pytest.raises(ValueError):
        g.queue_entries(variant, workers=13)
    with pytest.raises(ValueError):
        g.tag_for(variant, 42, fold=1)


def test_exact_verdict_and_partner_code_and_screen_registration(monkeypatch):
    for name in ('compare_seeds', 'tier_verdicts', 'partner_proxy', 'metric_contrast', 'contrast_pairs'):
        assert getattr(g, name) is getattr(g.e129, name)
    monkeypatch.setattr(g, 'sha256', lambda path: 'fake-binding')
    spec = g.screen_spec()
    assert spec['primary'] == 'k36' and spec['draws'] == 300 and spec['bootstrap_seed'] == 20261003
    assert set(g.REFERENCES) == {'dread', 'consensus', 'flight_v4', 'relabel_v1'}
    assert spec['partner']['report_only'] and spec['partner']['primary_variant_weight'] == .20
    assert set(spec['contrasts']) == set(g.VARIANTS)
    m = {n: dict(delta=.004, ci95=[.001, .006], positive_draw_share=.95,
                 bootstrap_finite_draws=300, bootstrap_draws=300) for n in g.REFERENCES}
    seeds = {str(s): copy.deepcopy(m) for s in g.SEEDS}
    verdicts = g.tier_verdicts(m, seeds)
    assert all(v['screen_verdict'] == 'THRESHOLDS_MET' and not v['adoption_authorized'] for v in verdicts.values())
    seeds['43']['dread']['delta'] = 0
    assert g.tier_verdicts(m, seeds)['tier_b']['screen_verdict'] == 'THRESHOLDS_NOT_MET'


def test_immutable_output_paths(tmp_path):
    (tmp_path / g.protocol_dir('POS')).mkdir(parents=True)
    with pytest.raises(SystemExit, match='existing'):
        g.prepare('POS', tmp_path)
    (tmp_path / g.screen_dir('POS')).mkdir()
    with pytest.raises(SystemExit, match='existing'):
        g.screen('POS', tmp_path)
    assert g.screen_all(tmp_path).count('NOT_SCREENED') == 3


@pytest.mark.parametrize('variant', g.VARIANTS)
def test_screen_routes_aug1_controls_and_late_contrasts(variant, tmp_path, monkeypatch):
    # A synthetic 4343 roster is needed for the fixed partner coverage contract;
    # metric calls are fakes and only 12-column constant arrays are allocated.
    ids = [f'u{i}' for i in range(4343)]
    folds = {}
    for i, block in enumerate([ids[:888], *[list(b) for b in np.array_split(ids[888:], 4)]]):
        path = tmp_path / f'fold{i}.json'
        path.write_text(json.dumps({'val': block}), encoding='utf-8')
        folds[str(i)] = dict(ordered_ids={'path': path.name}, tag=f'pooled{i}')
    bases, recipes = {}, {}
    registry = tmp_path / g.protocol_dir(variant) / 'manifest.json'
    registry.parent.mkdir(parents=True)
    spec = dict(bindings={}, interpretation='synthetic', averaging='seed mean', bootstrap='paired')
    registry.write_text(json.dumps({'screen': spec}), encoding='utf-8')
    for seed in g.SEEDS:
        base = dict(folds={'0': {**folds['0'], 'tag': f'EXP127_AUG1_S{seed}_F0'}}, source_bindings={})
        bases[seed] = base
        recipes[seed] = dict(folds={'0': {**folds['0'], 'tag': g.tag_for(variant, seed)}}, source_bindings={})
        for mp in (g.manifest_path(variant, seed, tmp_path), g.control_path(seed, tmp_path)):
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(g, 'base_manifest', lambda seed, root: bases[seed])
    monkeypatch.setattr(g.e127, 'base_manifest', lambda seed, root: {'folds': folds})
    monkeypatch.setattr(g, 'load_protocol', lambda v, seed, root: recipes[seed])
    checked_roles = []
    def verify(run, p, digest, root, candidate=True):
        checked_roles.append((run.name, candidate))
        return {}
    monkeypatch.setattr(g, 'verify_run', verify)
    def draws(*args, **kwargs):
        assert kwargs == dict(draws=300, seed=20261003)
        return []
    monkeypatch.setattr(g, 'component_draws', draws)
    comp = tmp_path / g.COMPONENTS
    comp.parent.mkdir(parents=True, exist_ok=True)
    comp.write_text('[]', encoding='utf-8')
    def load(path, roster, prediction=False):
        path = Path(path)
        if path == tmp_path / g.PARTNER:
            assert prediction and roster == ids
            return np.full((4343, 12), .7)
        if prediction:
            assert roster == ids[:888]
            value = .1 if path.parent.name.startswith('EXP127_AUG1') else .2
            if 'swa' in path.name: value = .3
            if 'ema' in path.name: value = .4
            if path.name.endswith('k36.csv'): value += .01
            return np.full((888, 12), value)
        return np.tile(np.arange(888) % 2, (12, 1)).T.astype(float)
    monkeypatch.setattr(g, 'load_table', load)
    captured = []
    def metrics(left, right, refs, draws, anchor, diagnostic=False):
        assert anchor.shape == (888, 12) and set(refs) == set(g.REFERENCES)
        captured.append((float(left[42][0, 0]), float(right[43][0, 0]), diagnostic))
        return dict(tiers={}, seed_mean={})
    monkeypatch.setattr(g, 'metric_contrast', metrics)
    monkeypatch.setattr(g, 'screen_markdown', lambda body: 'synthetic\n')
    result = g.screen(variant, tmp_path)
    assert checked_roles == [(g.tag_for(variant, s), True) if candidate else (f'EXP127_AUG1_S{s}_F0', False)
                             for s in g.SEEDS for candidate in (True, False)]
    mapping = dict(candidate=.2, control=.1, swa=.3, ema=.4)
    expected = [(mapping[left] + extra, mapping[right] + extra, diag)
                for _, left, right in g.contrast_pairs(variant) for extra, diag in ((.01, False), (0., True))]
    assert len(captured) == len(expected)
    for a, b in zip(captured, expected):
        assert a[:2] == pytest.approx(b[:2]) and a[2] == b[2]
    assert result['rows'] == 888 and result['gold58']['status'] == 'NOT_EVALUATED'


def test_control_verification_delegates_to_aug1(monkeypatch):
    calls = []
    monkeypatch.setattr(g.e127, 'verify_run', lambda *args: calls.append(args) or {'checked': 'AUG1'})
    assert g.verify_run(Path('EXP127_AUG1_S42_F0'), {}, 'sha', candidate=False) == {'checked': 'AUG1'}
    assert len(calls) == 1


@pytest.mark.parametrize('variant', g.VARIANTS)
def test_inventory_provenance_tamper(tmp_path, variant):
    p = derived(variant)
    p['source_bindings'].update({n: g.sha256(ROOT / n) for n in g.NEW_CODE})
    f = p['folds']['0']
    run = tmp_path / f['tag']
    run.mkdir()
    prov = dict(fold=0, seed=42, training_mode='full', checkpoint_selection='fixed-final',
        protocol_sha256='protocol-sha', label_set='dread', label_file_sha256=p['label_set']['sha256'],
        label_semantic_sha256=p['label_set']['semantic_sha256'], arch=p['common_args']['arch'],
        train_ids_ordered_sha256=f['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=f['ordered_ids']['val_sha256'],
        initial_state_sha256=p['initial_state_sha256'], initial_state_matches_registered=True,
        environment_matches_baseline=True, environment=p['environment'], remote_relaxed=False,
        input_policy='center3', norm='radimagenet', k_eval=24, k_eval_extra=36,
        runner_sha256=p['source_bindings'][g.RUNNER], guard_sha256=p['source_bindings'][g.TOOL],
        init_weights={'weights_sha256': g.RADIMAGENET_SHA256}, actual_pos_weight=f['pos_weight'],
        exp139_variant=variant, exp139_variant_config=p['variant_config'], exp139_adapter_initial_state=p['adapter_initial_state'],
        exp139_augmentation_config=p['augmentation_config'],
        exp139_code_sha256={n: p['source_bindings'][n] for n in g.NEW_CODE})
    history = [dict(epoch=i) for i in range(20)]
    summary = dict(args={**p['common_args'], 'variant': variant, 'seed': 42, 'tag': f['tag']}, mode='full',
        selected_epoch_1based=20, train=f['n_train'], val=f['n_val'], optimizer_updates=f['steps'], skipped_updates=0, history=history)
    files = {'summary.json': summary, 'provenance.json': prov, 'history.json': history,
             'model_best.pt': 'synthetic', 'oof_best.csv': 'synthetic', 'oof_k36.csv': 'synthetic'}
    if variant == 'LATE':
        files.update({n: 'synthetic' for n in g.late_files()})
        files['late.json'] = dict(epochs_1based=[18, 19, 20], ema_decay=.998,
            successful_optimizer_updates=f['steps'], bn_recomputed=False,
            swa_buffers=g.CONFIG['LATE']['swa_buffers'], ema_buffers=g.CONFIG['LATE']['ema_buffers'],
            state_sha256=dict(final='final', swa='swa', ema='ema'))
    def write_fixture():
        for name, obj in files.items():
            (run / name).write_text(json.dumps(obj), encoding='utf-8')
        (run / 'output-inventory.json').write_text(json.dumps({n: {'sha256': g.sha256(run / n)} for n in files}), encoding='utf-8')
    write_fixture()
    assert g.verify_run(run, p, 'protocol-sha', tmp_path)
    prov['exp139_adapter_initial_state'] = {}
    write_fixture()
    with pytest.raises(ValueError, match='provenance'):
        g.verify_run(run, p, 'protocol-sha', tmp_path)
    prov['exp139_adapter_initial_state'] = p['adapter_initial_state']
    prov['exp139_augmentation_config'] = {}
    write_fixture()
    with pytest.raises(ValueError, match='provenance'):
        g.verify_run(run, p, 'protocol-sha', tmp_path)
    prov['exp139_augmentation_config'] = p['augmentation_config']
    write_fixture()
    (run / 'oof_k36.csv').write_text('tamper', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory'):
        g.verify_run(run, p, 'protocol-sha', tmp_path)

