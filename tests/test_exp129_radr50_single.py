"""EXP129 CPU contract, paired-screen and immutable-output tests; no training job."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/src'))
from tools import exp129_radr50_single as g


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*args, **kwargs):
        raise AssertionError('EXP129 tests must never initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)
    monkeypatch.setattr(torch.cuda, 'init', fail)
    with torch.device('cpu'):
        yield
    assert not torch.cuda.is_initialized()


def derived(variant, seed=42):
    return g.derive_manifest(g.base_manifest(seed), variant, seed, {}, {'synthetic_spec': True})


def entry_module():
    spec = importlib.util.spec_from_file_location('exp129_entry_test', ROOT / g.RUNNER)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


@pytest.mark.parametrize('variant', g.VARIANTS)
@pytest.mark.parametrize('seed', g.SEEDS)
def test_only_registered_change(variant, seed):
    base, p = g.base_manifest(seed), derived(variant, seed)
    for key in ('initial_state', 'initial_state_sha256', 'environment', 'training_code_sha256'):
        assert p[key] == base[key]
    assert set(p['folds']) == {'0'}
    assert p['folds']['0']['steps'] == base['folds']['0']['steps'] == p['total_steps']
    changed = {k for k, v in p['common_args'].items() if v != base['common_args'][k]}
    assert changed == ({'bb_lr'} if variant in ('LRH', 'LRL') else set())
    assert p['common_args']['head_lr'] == base['common_args']['head_lr']
    if variant in ('LRH', 'LRL'):
        assert p['common_args']['bb_lr'] == base['common_args']['bb_lr'] * g.CONFIG[variant]['backbone_lr_factor']
    assert p['adapter_initial_state'] == g.adapter_registration(variant, seed)
    assert bool(p['adapter_initial_state']['keys']) == (variant in ('POS', 'CTX'))
    for key in ('initial_state_sha256', 'environment', 'seed'):
        bad = copy.deepcopy(p)
        bad[key] = 'tampered'
        with pytest.raises((ValueError, KeyError)):
            g.assert_single_change(bad, base)
    bad = copy.deepcopy(p)
    bad['common_args']['k'] = 16
    with pytest.raises(ValueError, match='single intervention'):
        g.assert_single_change(bad, base)
    bad = copy.deepcopy(p)
    bad['adapter_initial_state']['combined_sha256'] = 'tampered'
    with pytest.raises(ValueError, match='adapter'):
        g.assert_single_change(bad, base)


def section(text, start, end):
    a = text.index(start)
    return text[a:text.index(end, a)]


def test_ordered_training_path_matches_frozen_control():
    old = (ROOT / g.e125.SHARED_RUNNER).read_text(encoding='utf-8')
    new = (ROOT / g.RUNNER).read_text(encoding='utf-8')
    assert g.sha256(ROOT / g.e125.SHARED_RUNNER) == g.base_manifest(42)['source_bindings'][g.e125.SHARED_RUNNER]
    for start, end in [('    smoke = a.mode', '    def init_record'),
                       ('    if a.grad_ckpt:', '    late ='),
                       ('    runtime = configure_runtime', '    provenance =')]:
        if end == '    late =':
            assert section(new, start, end) == section(old, start, '\n    def autocast')
        else:
            assert section(new, start, end) == section(old, start, end)
    start, end = '    for ep in range(epochs):', '        peak ='
    current = section(new, start, end)
    current = current.replace('move_input(x, dev, non_blocking=True)', 'x.to(dev, non_blocking=True)')
    current = current.replace("            if late is not None:\n                late.update(model, successful=not (a.amp == 'fp16' and dev == 'cuda' and scaler.get_scale() < scale_before))\n", '')
    assert current == section(old, start, end)
    current = section(new, '    @torch.no_grad()\n    def evaluate', '    history, best')
    assert current.replace('move_input(x, dev)', 'x.to(dev)') == section(old, '    @torch.no_grad()\n    def evaluate', '    history, best')
    assert new.index('supply_registered_env()  # before') < new.index('import numpy as np')
    for path in (*g.NEW_CODE, *g.TESTS):
        ast.parse((ROOT / path).read_text(encoding='utf-8'), filename=path)


def test_no_inherited_thread_overrides_and_exact_args():
    entry = entry_module()
    env = dict(OMP_NUM_THREADS='2', MKL_NUM_THREADS='12', OPENBLAS_NUM_THREADS='8', NUMEXPR_NUM_THREADS='9')
    entry.supply_registered_env(env)
    assert all(env[k] == '1' for k in g.THREAD_ENV)
    assert env['HF_HUB_OFFLINE'] == '1'
    for variant in g.VARIANTS:
        for seed in g.SEEDS:
            p = derived(variant, seed)
            a = g.runner_argv(p, python=sys.executable)
            parsed = entry.build_parser().parse_args(a[2:])
            assert (parsed.variant, parsed.seed, parsed.fold) == (variant, seed, 0)
            b = g.e113.runner_argv(g.base_manifest(seed), 0, python=sys.executable)
            j = b.index('--label-set')
            del b[j:j+2]
            for flag in ('--protocol', '--tag', *('--' + k.replace('_', '-') for k in p['derived_from']['changed_args'])):
                b[b.index(flag) + 1] = a[a.index(flag) + 1]
            b[1] = a[1]
            assert a == b + ['--variant', variant]


@pytest.mark.parametrize('variant', g.VARIANTS)
def test_queue_is_two_serial_f0_seeds_no_launch(variant, monkeypatch):
    monkeypatch.setattr(g, 'load_protocol', lambda v, s, root: derived(v, s))
    items = g.queue_entries(variant, after='EXP127_LAST')
    assert len(items) == 2 and all(x['continue_on_failure'] for x in items)
    assert items[0]['after'] == 'EXP127_LAST'
    assert items[1]['after'] == items[0]['tag']
    assert [x['command'][x['command'].index('--seed') + 1] for x in items] == ['42', '43']
    assert all(x['command'][x['command'].index('--fold') + 1] == '0' for x in items)
    with pytest.raises(ValueError):
        g.queue_entries(variant, workers=13)
    with pytest.raises(ValueError):
        g.tag_for(variant, 42, fold=1)


def test_bootstrap_and_partner_against_independent_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(432)
    y = np.tile(np.arange(32) % 2, (12, 1)).T.astype(float)
    mask = np.ones_like(y, bool)
    refs = (y, mask)
    candidates = {s: rng.random(y.shape) for s in g.SEEDS}
    controls = {s: rng.random(y.shape) for s in g.SEEDS}
    anchor = rng.random(y.shape)
    draws = [np.arange(32), np.tile(np.arange(16), 2), np.arange(31, -1, -1)]
    report = g.partner_proxy(candidates, controls, anchor, refs, draws)
    expected = []
    for s in g.SEEDS:
        ca = .8 * rankdata(anchor, axis=0) / 32 + .2 * rankdata(candidates[s], axis=0) / 32
        co = .8 * rankdata(anchor, axis=0) / 32 + .2 * rankdata(controls[s], axis=0) / 32
        expected.append(roc_auc_score(y, ca, average='macro') - roc_auc_score(y, co, average='macro'))
    assert report['delta_partner'] == pytest.approx(np.mean(expected))
    assert report['report_only'] and set(report['comparisons']) == {'0.20', '0.15', '0.30'}
    same = g.partner_proxy(candidates, candidates, anchor, refs, draws)
    assert same['delta_partner'] == 0
    assert same['comparisons']['0.20']['seed_mean']['positive_draw_share'] == 0
    assert same['spearman']['candidate']['42']['finite_targets'] == 12
    changed = copy.deepcopy(candidates)
    changed[42][:, 0] = .5
    undefined = g.partner_proxy(changed, controls, anchor, refs, draws)
    assert undefined['spearman']['candidate']['42']['per_target'][g.T[0]] is None


def passing_metrics():
    m = {n: dict(delta=.004, ci95=[.001, .006], positive_draw_share=.95,
                 bootstrap_finite_draws=300, bootstrap_draws=300) for n in g.REFERENCES}
    s = {str(seed): copy.deepcopy(m) for seed in g.SEEDS}
    return m, s


def test_tier_b_all_safeguards_and_no_f0_adoption():
    mean, seeds = passing_metrics()
    verdicts = g.tier_verdicts(mean, seeds)
    assert all(v['screen_verdict'] == 'THRESHOLDS_MET' for v in verdicts.values())
    assert all(not v['adoption_authorized'] for v in verdicts.values())
    mean['dread'].update(delta=.001, ci95=[-.0001, .004], positive_draw_share=.90)
    mean['relabel_v1']['delta'] = -.001
    assert g.tier_verdicts(mean, seeds)['tier_b']['screen_verdict'] == 'THRESHOLDS_MET'
    for name, key, value in [('dread', 'delta', .000999), ('dread', 'positive_draw_share', .899999),
                             ('flight_v4', 'delta', -.001001), ('consensus', 'delta', 0.),
                             ('relabel_v1', 'delta', -.001001)]:
        bad = copy.deepcopy(mean)
        bad[name][key] = value
        assert g.tier_verdicts(bad, seeds)['tier_b']['screen_verdict'] == 'THRESHOLDS_NOT_MET'
    seeds['43']['dread']['delta'] = 0
    assert g.tier_verdicts(mean, seeds)['tier_b']['screen_verdict'] == 'THRESHOLDS_NOT_MET'
    mean['dread']['bootstrap_finite_draws'] = 299
    assert g.tier_verdicts(mean, seeds)['tier_b']['verdict'] == 'INVALID'


def test_late_fixed_contrasts_and_fail_closed_metadata():
    assert g.contrast_pairs('LATE') == [
        ('swa_vs_final', 'swa', 'candidate'), ('ema_vs_final', 'ema', 'candidate'),
        ('swa_vs_control', 'swa', 'control'), ('ema_vs_control', 'ema', 'control'),
        ('final_vs_control', 'candidate', 'control')]
    assert set(g.late_files()) >= {'epoch_18.pt', 'epoch_19.pt', 'epoch_20.pt', 'model_swa.pt', 'model_ema.pt'}
    summary = dict(optimizer_updates=8620, skipped_updates=2)
    meta = dict(epochs_1based=[18, 19, 20], ema_decay=.998, successful_optimizer_updates=8618,
        bn_recomputed=False, swa_buffers=g.CONFIG['LATE']['swa_buffers'], ema_buffers=g.CONFIG['LATE']['ema_buffers'],
        state_sha256=dict(final='f', swa='s', ema='e'))
    g.verify_late_metadata(meta, summary)
    for key, bad_value in [('ema_decay', .99), ('epochs_1based', [17, 18, 19]),
                           ('successful_optimizer_updates', 8620), ('bn_recomputed', True), ('state_sha256', {})]:
        bad = {**meta, key: bad_value}
        with pytest.raises(ValueError, match='LATE'):
            g.verify_late_metadata(bad, summary)


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
        exp129_variant=variant, exp129_variant_config=p['variant_config'], exp129_adapter_initial_state=p['adapter_initial_state'],
        exp129_code_sha256={n: p['source_bindings'][n] for n in g.NEW_CODE})
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
    prov['exp129_adapter_initial_state'] = {}
    write_fixture()
    with pytest.raises(ValueError, match='provenance'):
        g.verify_run(run, p, 'protocol-sha', tmp_path)
    prov['exp129_adapter_initial_state'] = p['adapter_initial_state']
    write_fixture()
    (run / 'oof_k36.csv').write_text('tamper', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory'):
        g.verify_run(run, p, 'protocol-sha', tmp_path)


def test_target_only_csv_and_uid_alignment(tmp_path):
    ids = ['a', 'b', 'c']
    frame = pd.DataFrame(np.arange(36).reshape(3, 12) / 36, columns=g.T)
    frame.insert(0, g.UID, ids)
    frame['Report'] = 'synthetic ignored field'
    path = tmp_path / 'labels.csv'
    frame.to_csv(path, index=False)
    data, _ = g.load_label_csv(path)
    assert set(data) == set(ids) and data['a'].shape == (12,)
    np.testing.assert_allclose(g.load_table(path, ids[::-1]), frame[g.T].to_numpy()[::-1])


def test_immutable_and_missing_outputs(tmp_path):
    (tmp_path / g.protocol_dir('POS')).mkdir(parents=True)
    with pytest.raises(SystemExit, match='existing'):
        g.prepare('POS', tmp_path)
    (tmp_path / g.screen_dir('POS')).mkdir()
    with pytest.raises(SystemExit, match='existing'):
        g.screen('POS', tmp_path)
    md = g.screen_all(tmp_path)
    assert md.count('NOT_SCREENED') == 5
    assert 'PENDING_POOLED_FIVE_FOLD' in md


@pytest.mark.parametrize('variant', g.VARIANTS)
def test_screen_wiring_same_run_final_and_matched_controls(variant, tmp_path, monkeypatch):
    """Synthetic arrays prove all LATE arms and both K route to fixed comparators."""
    ids = [f'u{i}' for i in range(4343)]
    folds = {}
    blocks = [ids[:888], *[list(b) for b in np.array_split(ids[888:], 4)]]
    for i, block in enumerate(blocks):
        path = tmp_path / f'fold{i}.json'
        path.write_text(json.dumps({'val': block}), encoding='utf-8')
        folds[str(i)] = dict(ordered_ids={'path': path.name}, tag='control42' if i == 0 else f'unused{i}')
    bases, recipes = {}, {}
    registry = tmp_path / g.protocol_dir(variant) / 'manifest.json'
    registry.parent.mkdir(parents=True)
    spec = dict(bindings={}, interpretation='synthetic F0 only', averaging='mean of seed metrics', bootstrap='synthetic draws')
    registry.write_text(json.dumps({'screen': spec}), encoding='utf-8')
    for seed in g.SEEDS:
        base = dict(folds=copy.deepcopy(folds), source_bindings={})
        base['folds']['0']['tag'] = f'control{seed}'
        bases[seed] = base
        recipes[seed] = dict(folds={'0': {**folds['0'], 'tag': f'candidate{seed}'}}, source_bindings={})
        for mp in (g.manifest_path(variant, seed, tmp_path), tmp_path / g.BASES[seed][0] / 'manifest.json'):
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(g, 'base_manifest', lambda seed, root: bases[seed])
    monkeypatch.setattr(g, 'load_protocol', lambda v, seed, root: recipes[seed])
    monkeypatch.setattr(g, 'verify_run', lambda *args, **kwargs: {})
    monkeypatch.setattr(g, 'component_draws', lambda *args, **kwargs: [])
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
            value = .1 if path.parent.name.startswith('control') else .2
            if 'swa' in path.name:
                value = .3
            if 'ema' in path.name:
                value = .4
            if path.name.endswith('k36.csv'):
                value += .01
            return np.full((888, 12), value)
        return np.tile(np.arange(888) % 2, (12, 1)).T.astype(float)
    monkeypatch.setattr(g, 'load_table', load)
    captured = []
    def metrics(left, right, refs, draws, anchor, diagnostic=False):
        assert anchor.shape == (888, 12) and set(refs) == set(g.REFERENCES)
        captured.append((float(left[42][0, 0]), float(right[43][0, 0]), diagnostic))
        return {'tiers': {}, 'seed_mean': {}, 'partner': {}, 'per_seed': {}}
    monkeypatch.setattr(g, 'metric_contrast', metrics)
    monkeypatch.setattr(g, 'screen_markdown', lambda body: 'synthetic report\n')
    body = g.screen(variant, tmp_path)
    assert list(body['comparisons']) == [v[0] for v in g.contrast_pairs(variant)]
    assert body['rows'] == 888 and body['gold58']['status'] == 'NOT_EVALUATED'
    mapping = {'candidate': .2, 'control': .1, 'swa': .3, 'ema': .4}
    expected = [(mapping[left] + extra, mapping[right] + extra, diagnostic)
                for _, left, right in g.contrast_pairs(variant) for extra, diagnostic in ((.01, False), (0., True))]
    assert len(captured) == len(expected)
    for actual, want in zip(captured, expected, strict=True):
        assert actual[:2] == pytest.approx(want[:2]) and actual[2] == want[2]
    assert (tmp_path / g.screen_dir(variant) / 'result.json').is_file()


@pytest.mark.parametrize('variant', g.VARIANTS)
def test_prepared_protocol_guard_cpu_only(variant, monkeypatch):
    if not (ROOT / g.protocol_dir(variant) / 'manifest.json').is_file():
        pytest.skip('run prepare first for real protocol checks')
    for key, value in {**g.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    entry = entry_module()
    for seed in g.SEEDS:
        p = g.load_protocol(variant, seed)
        argv = g._set_flag(g.runner_argv(p, workers=0), '--device', 'cpu') + ['--validate-only']
        args = entry.build_parser().parse_args(argv[2:])
        view, fold, ordered, _ = g.validate_arguments(args, ROOT)
        assert view['variant'] == variant and len(ordered['val']) == 888
        args.device = 'cuda'
        with pytest.raises(ValueError, match='explicitly use --device cpu'):
            g.validate_arguments(args, ROOT)
