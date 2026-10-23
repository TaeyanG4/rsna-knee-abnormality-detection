"""EXP127 CPU-only contract tests. No training, CUDA, network or report text.

System Python; OMP_NUM_THREADS=2; --basetemp .local/pytest-exp127
-p no:cacheprovider. Real-cache validation is the separate `validate` command.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/src'))
from tools import exp127_radr50_single as g
from raptor_common import SLOTS64, select_centers, study_rng, windows_from
from raptor_radimagenet import CenterSliceWindows
from raptor_exp127 import loader_factory, ordered_equivalence
from raptor_exp127_adj3 import adjacent_indices, adjacent3_windows_from
from raptor_exp127_aug1 import augment_windows


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*args, **kwargs):
        raise AssertionError('EXP127 tests must never initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)
    monkeypatch.setattr(torch.cuda, 'init', fail)
    with torch.device('cpu'):
        yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def corpus(tmp_path):
    """A tiny real Corpus-format fixture with all planes, holes and empty input."""
    ids = [f'synthetic-{i}' for i in range(10)]
    spec = dict(slots=SLOTS64, depth=64)
    rng = np.random.default_rng(11)
    volumes = rng.integers(0, 256, (10, 64, 12, 12), dtype=np.uint8)
    masks = np.ones((10, 64), np.uint8)
    masks[0, 17:20] = 0
    masks[1, 30:35] = 0
    masks[2] = 0
    np.save(tmp_path / 'ids.npy', np.asarray(ids))
    np.save(tmp_path / 'masks.npy', masks)
    np.save(tmp_path / 'volumes.npy', volumes)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(contract='synthetic', spec=spec,
        shards=[dict(start=0, stop=10, file='volumes.npy')])), encoding='utf-8')
    labels = {u: np.asarray([(i + j) % 2 for j in range(12)], np.float32) for i, u in enumerate(ids)}
    return tmp_path, ids, labels


def derived(variant, seed=42):
    return g.derive_manifest(g.base_manifest(seed), variant, seed, {}, {'synthetic_spec': True})


def entry_module():
    spec = importlib.util.spec_from_file_location('exp127_entry_test', ROOT / g.RUNNER)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


@pytest.mark.parametrize('variant', g.VARIANTS)
@pytest.mark.parametrize('seed', g.SEEDS)
def test_variant_off_exact_ordered_batches(variant, seed, corpus):
    path, ids, labels = corpus
    report = ordered_equivalence(variant, path, ids[:8], ids[8:], labels, seed, res=16)
    assert report['batches'] == 4 and report['exact_inputs_labels_and_order']
    assert report['device'] == 'cpu'


def test_adjacent_edges_holes_empty_and_plane_boundaries():
    spec = dict(slots=SLOTS64)
    mask = np.ones(64, np.uint8)
    centers = [0, 17, 18, 31, 32, 43, 44, 51, 52, 63, -1]
    expected = [(0, 0, 1), (16, 17, 17), (18, 18, 19), (30, 31, 31), (32, 32, 33),
                (42, 43, 43), (44, 44, 45), (50, 51, 51), (52, 52, 53), (62, 63, 63), (-1, -1, -1)]
    assert adjacent_indices(centers, mask, spec, 64) == expected
    mask[19] = 0
    assert adjacent_indices([18, 20], mask, spec, 64) == [(18, 18, 18), (20, 20, 21)]
    with pytest.raises(ValueError, match='invalid.*centre'):
        adjacent_indices([19], mask, spec, 64)
    with pytest.raises(ValueError, match='depth'):
        adjacent_indices([0], mask, dict(slots=[('x', 0, 63)]), 64)
    vol = np.broadcast_to(np.arange(64, dtype=np.uint8)[:, None, None], (64, 8, 8)).copy()
    x = adjacent3_windows_from(vol, centers, np.ones(64), spec, 8)
    assert x.dtype == np.float32 and np.array_equal(x[-1], np.zeros((3, 8, 8), np.float32))
    for j, tri in enumerate(expected[:-1]):
        assert np.array_equal(x[j, :, 0, 0], np.asarray(tri, np.float32) / 255.)


def test_adjacent_registered_centres_equal_original_triplets(corpus):
    from raptor_common import Corpus
    path, ids, _ = corpus
    c = Corpus(path)
    for uid in ids[:3]:
        for seed in g.SEEDS:
            for k, rng in ((12, study_rng(seed, 3, uid)), (24, None), (36, None)):
                centers = select_centers(c.mask(uid), c.depth, k, c.spec, 'within-slot-valid', rng)
                a = adjacent3_windows_from(c.volume(uid), centers, c.mask(uid), c.spec, 16)
                b = windows_from(c.volume(uid), centers, 16)
                assert np.array_equal(a, b)


class IdentityAffineFlipRng:
    def uniform(self, lo, hi):
        return {(-10., 10.): 0., (.9, 1.1): 1., (-.05, .05): 0., (.8, 1.25): 1.}[(lo, hi)]
    def random(self):
        return 0.  # force every eligible flip


def test_aug_plane_flip_and_no_channel_misalignment():
    plane = np.arange(64, dtype=np.float32).reshape(8, 8) / 64.
    x = np.tile(plane, (5, 3, 1, 1))
    centers = [1, 31, 32, 52, -1]
    out = augment_windows(x, centers, IdentityAffineFlipRng(), dict(slots=SLOTS64))
    assert np.allclose(out[:2], x[:2], atol=1e-6)  # both sagittal slots
    assert np.allclose(out[2:4], x[2:4, :, :, ::-1], atol=1e-6)
    assert np.array_equal(out[4], x[4])
    assert np.array_equal(out[:, 0], out[:, 1]) and np.array_equal(out[:, 1], out[:, 2])
    with pytest.raises(ValueError, match='64-slot'):
        augment_windows(x, centers, IdentityAffineFlipRng(), dict(slots=[('x', 0, 64)]))


@pytest.mark.parametrize('variant', g.VARIANTS)
def test_enabled_inputs_deterministic_epoch_seed_and_eval(variant, corpus):
    path, ids, labels = corpus
    policy = 'adjacent3' if variant == 'ADJ3' else 'center3'
    opts = dict(window_policy='within-slot-valid', input_policy=policy)
    make = loader_factory(variant)
    ds = make(path, ids, labels, 16, 12, True, 'radimagenet', 42, **opts)
    a = ds[0][0]
    assert a.device.type == 'cpu' and a.dtype == torch.float32 and torch.isfinite(a).all()
    assert a.min() >= -1 and a.max() <= 1
    random.seed(913)
    np.random.seed(912)
    assert torch.equal(a, ds[0][0])
    ds.set_epoch(1)
    assert not torch.equal(a, ds[0][0])
    other = make(path, ids, labels, 16, 12, True, 'radimagenet', 43, **opts)
    assert not torch.equal(a, other[0][0])
    evaluation = make(path, ids, labels, 16, 24, False, 'radimagenet', 42, **opts)
    e = evaluation[0][0]
    evaluation.set_epoch(9)
    assert torch.equal(e, evaluation[0][0])
    if variant != 'ADJ3':
        base = CenterSliceWindows(path, ids, labels, 16, 24, False, 'radimagenet', 42, **opts)
        assert torch.equal(e, base[0][0])
    assert torch.equal(ds[2][0], torch.full_like(ds[2][0], -1))  # all invalid


@pytest.mark.parametrize('variant', g.VARIANTS)
@pytest.mark.parametrize('seed', g.SEEDS)
def test_only_registered_intervention_and_derived_counts(variant, seed):
    base, p = g.base_manifest(seed), derived(variant, seed)
    assert p['initial_state'] == base['initial_state']
    assert p['initial_state_sha256'] == base['initial_state_sha256']
    assert p['environment'] == base['environment']
    assert set(p['folds']) == {'0'}
    changes = {k for k in p['common_args'] if p['common_args'][k] != base['common_args'][k]}
    assert changes == {'ADJ3': {'input_policy'}, 'EP40': {'epochs', 'fixed_epoch'}, 'AUG1': set()}[variant]
    f = p['folds']['0']
    factor = 2 if variant == 'EP40' else 1
    assert f['steps'] == base['folds']['0']['steps'] * factor == f['scheduler_total_steps'] == p['total_steps']
    assert f['max_fp16_skips'] == f['steps'] // 100
    for field in ('initial_state_sha256', 'environment', 'seed'):
        bad = copy.deepcopy(p)
        bad[field] = 'changed'
        with pytest.raises((ValueError, KeyError)):
            g.assert_single_change(bad, base)
    bad = copy.deepcopy(p)
    bad['common_args']['k'] = 16
    with pytest.raises(ValueError, match='single intervention'):
        g.assert_single_change(bad, base)
    bad = copy.deepcopy(p)
    bad['folds']['0']['steps'] += 1
    with pytest.raises(ValueError, match='fold contract'):
        g.assert_single_change(bad, base)


def test_ordered_runner_sections_identical_to_exp113_shared_runner():
    old = (ROOT / g.e125.SHARED_RUNNER).read_text(encoding='utf-8')
    new = (ROOT / g.RUNNER).read_text(encoding='utf-8')
    assert g.sha256(ROOT / g.e125.SHARED_RUNNER) == g.base_manifest(42)['source_bindings'][g.e125.SHARED_RUNNER]
    # Actual ordered data/shuffle/optimizer/scheduler/loss/update/eval paths,
    # rather than merely comparing a set of tokens or function names.
    for start, end in [('    smoke = a.mode', '    if a.validate_only:'),
                       ('    runtime = configure_runtime', '    provenance ='),
                       ('    if smoke:\n        provenance["smoke"]', '\n\nif __name__')]:
        ni, oi = new.index(start), old.index(start)
        assert new[ni:new.index(end, ni)] == old[oi:old.index(end, oi)]
    assert 'CenterSliceWindows = loader_factory(a.variant)' in new
    assert new.index('supply_registered_env()  # before') < new.index('import numpy as np')
    for path in (*g.NEW_CODE, g.TEST):
        ast.parse((ROOT / path).read_text(encoding='utf-8'), filename=path)


def test_runner_env_replaces_inherited_overrides_and_cpu_argv(monkeypatch):
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
            assert parsed.variant == variant and parsed.seed == seed and parsed.fold == 0
            b = g.e113.runner_argv(g.base_manifest(seed), 0, python=sys.executable)
            j = b.index('--label-set')
            del b[j:j+2]
            for flag in ('--protocol', '--tag', *('--' + k.replace('_', '-') for k in p['derived_from']['changed_args'])):
                b[b.index(flag) + 1] = a[a.index(flag) + 1]
            b[1] = a[1]
            assert a == b + ['--variant', variant]
    monkeypatch.setattr(g, 'load_protocol', lambda variant, seed: derived(variant, seed))
    called = []
    def run(argv, **kwargs):
        called.append(argv)
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(g.subprocess, 'run', run)
    assert g.validate_cli('ADJ3', 42) == 0
    assert called[0][called[0].index('--device') + 1] == 'cpu'
    assert '--validate-only' in called[0]


def test_queue_is_serial_f0_and_continue_on_failure(monkeypatch):
    monkeypatch.setattr(g, 'load_protocol', lambda variant, seed, root: derived(variant, seed))
    for variant in g.VARIANTS:
        entries = g.queue_entries(variant, after='EXISTING_TAIL')
        assert len(entries) == 2 and all(e['continue_on_failure'] is True for e in entries)
        assert entries[0]['after'] == 'EXISTING_TAIL' and entries[1]['after'] == entries[0]['tag']
        assert [e['tag'] for e in entries] == [g.tag_for(variant, s) for s in g.SEEDS]
        with pytest.raises(ValueError, match='fold 0'):
            g.tag_for(variant, 42, fold=1)
    with pytest.raises(ValueError, match='workers'):
        g.queue_entries('ADJ3', workers=13)


def test_target_only_csv_same_semantic_hash(tmp_path, monkeypatch):
    from raptor_common import load_label_csv as original
    path = tmp_path / 'synthetic_labels.csv'
    # Synthetic non-clinical marker ensures usecols actually excludes extras.
    frame = pd.DataFrame({g.UID: ['a', 'b'], **{t: [.1, .9] for t in g.T}, 'Report': ['IGNORED', 'IGNORED']})
    frame.to_csv(path, index=False)
    base, digest = original(path)
    real_read = pd.read_csv
    def checked_read(*args, **kwargs):
        assert kwargs['usecols']('Report') is False
        return real_read(*args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', checked_read)
    values, got = g.load_label_csv(path)
    assert got == digest and list(values) == list(base)
    assert all(np.array_equal(values[k], base[k]) for k in base)


def test_paired_bootstrap_seed_mean_share_and_fixed_defaults():
    y = np.tile(np.asarray([0, 0, 1, 1])[:, None], (1, 12))
    ref = (y, np.ones_like(y, dtype=bool))
    a = {42: np.tile(np.array([.1, .4, .35, .8])[:, None], (1, 12)),
         43: np.tile(np.array([.4, .3, .2, .5])[:, None], (1, 12))}
    b = {s: np.full(y.shape, .5) for s in g.SEEDS}
    draws = [np.arange(4), np.array([0, 1, 2, 3, 2, 3])]
    per, mean, _ = g.compare_seeds(a, b, ref, draws)
    old_per, old_mean, _ = g.e125.compare_seeds(a, b, ref, draws)
    assert mean['candidate'] == pytest.approx(.625) and mean['delta'] == pytest.approx(.125)
    for key in ('delta', 'candidate', 'control', 'ci95'):
        assert mean[key] == pytest.approx(old_mean[key])
        assert per['42'][key] == pytest.approx(old_per['42'][key])
    ds = np.array([np.mean([np.nanmean(g.auc_vector(a[s], ref, d)) - np.nanmean(g.auc_vector(b[s], ref, d))
                           for s in g.SEEDS]) for d in draws])
    assert mean['positive_draw_share'] == np.mean(ds > 0)
    _, ties, _ = g.compare_seeds(b, b, ref, draws)
    assert ties['positive_draw_share'] == 0
    ids = ['a', 'b', 'c', 'd']
    comps = [dict(component_id='x', members=['a', 'b']), dict(component_id='y', members=['c', 'd'])]
    boot = g.component_draws(ids, comps)
    assert len(boot) == 300 and g.BOOT_SEED == 20261003
    assert all(np.sum(d == 0) == np.sum(d == 1) and np.sum(d == 2) == np.sum(d == 3) for d in boot)
    assert all(np.array_equal(a, b) for a, b in zip(boot, g.component_draws(ids, comps)))


def test_tier_boundaries_and_f0_cannot_adopt():
    mean = {n: dict(delta=0., ci95=[-.01, .01], positive_draw_share=.95,
                   bootstrap_draws=300, bootstrap_finite_draws=300) for n in g.REFERENCES}
    mean['dread'].update(delta=.004, ci95=[.000001, .01])
    seeds = {str(s): {'dread': {'delta': .002}} for s in g.SEEDS}
    result = g.tier_verdicts(mean, seeds)
    assert all(v['screen_verdict'] == 'THRESHOLDS_MET' and not v['adoption_authorized'] for v in result.values())
    assert result['tier_b']['verdict'] == 'PENDING_POOLED_FIVE_FOLD'
    mean['dread'].update(delta=.001, ci95=[-.0001, .004], positive_draw_share=.90)
    mean['flight_v4']['delta'] = mean['consensus']['delta'] = mean['relabel_v1']['delta'] = -.001
    result = g.tier_verdicts(mean, seeds)
    assert result['tier_a']['screen_verdict'] == 'THRESHOLDS_NOT_MET'
    assert result['tier_b']['screen_verdict'] == 'THRESHOLDS_MET'
    for name, key, value in [('dread', 'delta', .000999), ('dread', 'positive_draw_share', .899999),
                             ('flight_v4', 'delta', -.001001), ('consensus', 'delta', -.001001),
                             ('relabel_v1', 'delta', -.001001)]:
        bad = copy.deepcopy(mean)
        bad[name][key] = value
        assert g.tier_verdicts(bad, seeds)['tier_b']['screen_verdict'] == 'THRESHOLDS_NOT_MET'
    seeds['43']['dread']['delta'] = 0
    assert g.tier_verdicts(mean, seeds)['tier_b']['screen_verdict'] == 'THRESHOLDS_NOT_MET'
    mean['dread']['bootstrap_finite_draws'] = 299
    assert g.tier_verdicts(mean, seeds)['tier_b']['verdict'] == 'INVALID'


@pytest.mark.parametrize('variant', g.VARIANTS)
def test_inventory_and_variant_provenance(tmp_path, variant):
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
        input_policy=p['common_args']['input_policy'], norm='radimagenet', k_eval=24, k_eval_extra=36,
        runner_sha256=p['source_bindings'][g.RUNNER], guard_sha256=p['source_bindings'][g.TOOL],
        init_weights={'weights_sha256': g.RADIMAGENET_SHA256}, actual_pos_weight=f['pos_weight'],
        exp127_variant=variant, exp127_variant_config=p['variant_config'],
        exp127_code_sha256={n: p['source_bindings'][n] for n in g.NEW_CODE})
    history = [dict(epoch=i) for i in range(p['common_args']['epochs'])]
    summary = dict(args={**p['common_args'], 'variant': variant, 'seed': 42, 'tag': f['tag']}, mode='full',
        selected_epoch_1based=p['common_args']['epochs'], train=f['n_train'], val=f['n_val'],
        optimizer_updates=f['steps'], skipped_updates=0, history=history)
    files = {'summary.json': summary, 'provenance.json': prov, 'history.json': history,
             'model_best.pt': 'synthetic', 'oof_best.csv': 'synthetic', 'oof_k36.csv': 'synthetic'}
    def write_fixture():
        for name, obj in files.items():
            (run / name).write_text(json.dumps(obj), encoding='utf-8')
        (run / 'output-inventory.json').write_text(json.dumps({n: {'sha256': g.sha256(run / n)} for n in files}), encoding='utf-8')
    write_fixture()
    assert len(g.verify_run(run, p, 'protocol-sha', tmp_path)) == 7
    prov['exp127_variant'] = 'wrong'
    write_fixture()
    with pytest.raises(ValueError, match='provenance'):
        g.verify_run(run, p, 'protocol-sha', tmp_path)
    prov['exp127_variant'] = variant
    summary['args']['k'] = 16
    write_fixture()
    with pytest.raises(ValueError, match='recipe'):
        g.verify_run(run, p, 'protocol-sha', tmp_path)
    summary['args']['k'] = 12
    write_fixture()
    (run / 'oof_k36.csv').write_text('tamper', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory'):
        g.verify_run(run, p, 'protocol-sha', tmp_path)


def test_immutable_outputs_and_missing_summary(tmp_path):
    out = tmp_path / g.protocol_dir('ADJ3')
    out.mkdir(parents=True)
    with pytest.raises(SystemExit, match='existing'):
        g.prepare('ADJ3', tmp_path)
    out = tmp_path / g.screen_dir('ADJ3')
    out.mkdir()
    with pytest.raises(SystemExit, match='existing'):
        g.screen('ADJ3', tmp_path)
    md = g.screen_all(tmp_path)
    assert md.count('NOT_SCREENED') == 3 and 'PENDING_POOLED_FIVE_FOLD' in md


@pytest.mark.parametrize('variant', g.VARIANTS)
def test_prepared_protocol_guard_cpu_only(variant, monkeypatch):
    if not (ROOT / g.protocol_dir(variant) / 'manifest.json').is_file():
        pytest.skip('prepare this variant first')
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
        args.device = 'cpu'
        args.validate_only = False
        with pytest.raises(ValueError, match='frozen argument.*device'):
            g.validate_arguments(args, ROOT)
