"""CPU-only checks for H-B-22 / EXP102: the raptor80 contract extension, the 80-slice cache, depth-scaled windows and the protocol guard.

No GPU and no training (the runner's --validate-only and smoke paths are exercised through tools/exp102_dense80.py).
Tests on the real cache / protocol are skipped when those do not exist yet.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'src'))
import raptor_contracts_ext as ext  # noqa: E402
import raptor_common  # noqa: E402
from raptor_common import CONTRACTS, Corpus, select_centers, study_rng  # noqa: E402
from tools import exp102_dense80 as m  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402

DENSE = ROOT / m.CORPUS
BASE = ROOT / m.BASE_CORPUS
REAL = ROOT / m.PROTOCOL / 'manifest.json'
def _finished(path):
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('finished') is True
    except (OSError, ValueError):
        return False


built = pytest.mark.skipif(not _finished(DENSE / 'manifest.json'), reason='RAPTOR80 cache not finished')
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP102 protocol not prepared')
SPEC80 = dict(slots=[list(s) for s in ext.SLOTS80], depth=80)


def runner_module():
    spec = importlib.util.spec_from_file_location('exp102_runner', ROOT / m.RUNNER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ contract registration
def test_raptor80_contract_registered_in_the_shared_mapping():
    c = CONTRACTS['raptor80_336_c140_s02']
    assert CONTRACTS is raptor_common.CONTRACTS
    assert [s[2] for s in c['slots']] == [22, 18, 15, 10, 15] and sum(s[2] for s in c['slots']) == 80 == ext.depth('raptor80_336_c140_s02')
    assert [(s[0], s[1]) for s in c['slots']] == [(s[0], s[1]) for s in raptor_common.SLOTS64]  # same plane/fluid order as raptor64
    base = CONTRACTS['raptor64_336_c140_s02']
    assert (c['img'], c['crop_mm'], tuple(c['span'])) == (base['img'], base['crop_mm'], tuple(base['span'])) == (336, 140.0, (0.02, 0.98))
    assert sum(s[2] for s in base['slots']) == 64  # the original contracts are untouched


def test_register_is_idempotent_and_refuses_a_conflict():
    before = copy.deepcopy(dict(CONTRACTS))
    assert ext.register() == ['raptor80_336_c140_s02'] and dict(CONTRACTS) == before
    other = {'raptor80_336_c140_s02': dict(slots=raptor_common.SLOTS64, img=336, crop_mm=140.0, span=(0.02, 0.98))}
    with pytest.raises(RuntimeError, match='different spec'):
        ext.register(other)


def test_raptor_common_and_builder_are_unchanged():
    v4 = m.read(ROOT / m.V4)
    for name, expected in v4['training_code_sha256'].items():
        assert m.sha256(ROOT / 'training/src' / name) == expected


def test_dicom_adapter_and_builder_accept_the_new_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', list(sys.argv))
    from raptor_dicom import _contract_spec
    spec = _contract_spec('raptor80_336_c140_s02')
    assert spec['depth'] == 80 and spec['img'] == 336 and spec['slots'] == SPEC80['slots']
    import raptor_corpus_ext
    with pytest.raises(SystemExit, match='no manifest'):  # argparse accepted the contract; --verify on an empty dir stops here
        raptor_corpus_ext.main(['--contract', 'raptor80_336_c140_s02', '--out', str(tmp_path / 'c'), '--verify'])


# ------------------------------------------------------------------ windows on the 80-slot layout
def _mask(missing=()):
    mk = np.ones(80, np.uint8)
    for i in missing:
        mk[i] = 0
    return mk


def test_within_slot_windows_never_cross_80_slot_boundaries():
    from raptor_windows_ext import valid_centers
    bounds = np.cumsum([0] + [s[2] for s in ext.SLOTS80])
    mk = _mask(missing=range(65, 80))  # axial series absent
    cs = valid_centers(mk, 80, SPEC80)
    assert len(cs) == 20 + 16 + 13 + 8 and all(c < 65 for c in cs)
    for c in cs:
        j = np.searchsorted(bounds, c, side='right') - 1
        assert bounds[j] < c < bounds[j + 1] - 1
    assert len(valid_centers(_mask(), 80, SPEC80)) == 70
    assert select_centers(mk, 80, len(cs), SPEC80, 'within-slot-valid') == cs  # K = n is every centre once
    tr = select_centers(mk, 80, m.K_TRAIN, SPEC80, 'within-slot-valid', study_rng(42, 0, 'u'))
    assert len(tr) == 16 and set(tr) <= set(cs) and len(set(tr)) == 16
    ev = select_centers(mk, 80, m.K_EVAL, SPEC80, 'within-slot-valid')
    assert len(ev) == 31 and set(ev) <= set(cs)
    assert select_centers(np.zeros(80, np.uint8), 80, 3, SPEC80, 'within-slot-valid') == [-1, -1, -1]


def _tiny_corpus(directory, n=3, img=8, missing=None):
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    vols = rng.integers(1, 255, size=(n, 80, img, img), dtype=np.uint8)
    masks = np.ones((n, 80), np.uint8)
    for r, idx in (missing or {}).items():
        vols[r, list(idx)] = 0; masks[r, list(idx)] = 0
    np.save(directory / 'vols_shard00.npy', vols); np.save(directory / 'masks.npy', masks)
    ids = [f'u{i}' for i in range(n)]
    np.save(directory / 'ids.npy', np.array(ids))
    manifest = dict(contract='raptor80_336_c140_s02', spec=dict(slots=SPEC80['slots'], img=img, depth=80),
                    shards=[dict(shard=0, start=0, stop=n, file='vols_shard00.npy', done=True)])
    (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return ids


def test_all_valid_windows_equal_studywindows_at_k_equal_n(tmp_path):
    from raptor_data import StudyWindows
    from raptor_windows_ext import AllValidWindows, all_position_k
    ids = _tiny_corpus(tmp_path / 'c', missing={1: range(40, 55), 2: range(80)})
    labels = {u: np.zeros(12, np.float32) for u in ids}
    corpus = Corpus(tmp_path / 'c')
    ks = [all_position_k(corpus.mask(u), corpus.depth, corpus.spec) for u in ids]
    assert ks == [70, 70 - 13, 1]  # coronal-fluid slot (40..54) gone; an empty study gets one explicit empty window
    ds = AllValidWindows(tmp_path / 'c', ids, labels, 8, 'imagenet', 42)
    for i, u in enumerate(ids):
        x, _, uid = ds[i]
        ref = StudyWindows(tmp_path / 'c', [u], labels, 8, ks[i], False, 'imagenet', 42, window_policy='within-slot-valid')[0][0]
        assert uid == u and tuple(x.shape) == (ks[i], 3, 8, 8) and np.array_equal(x.numpy(), ref.numpy())


def test_window_coverage_counts():
    masks = np.stack([_mask(), _mask(range(0, 40)), np.zeros(80, np.uint8)])
    cov = m.window_coverage(masks, SPEC80, [0, 1, 2], 16, 31)
    assert cov['studies'] == 3 and cov['max'] == 70 and cov['zero'] == 1 and cov['below_k_train'] == 1 and cov['below_k_eval'] == 1
    assert cov['min'] == 0 and abs(cov['mean'] - (70 + 34 + 0) / 3) < 1e-3


# ------------------------------------------------------------------ recipe / contract
def test_dense_args_change_only_the_window_counts():
    args = m.dense_args(dict(k99.COMMON))
    base = {k: v for k, v in k99.COMMON.items() if k != 'workers'}
    assert {k: v for k, v in args.items() if k not in ('k', 'k_eval')} == {k: v for k, v in base.items() if k not in ('k', 'k_eval')}
    assert (args['k'], args['k_eval']) == (16, 31) and 'workers' not in args
    assert args['arch'] == 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k' and args['amp'] == 'fp16' and args['amp_init_scale'] == 1024.0
    assert (args['epochs'], args['bs'], args['bb_lr'], args['head_lr'], args['wd']) == (8, 8, 3e-5, 1e-3, 0.02)
    with pytest.raises(ValueError, match='B8 recipe'):
        m.dense_args(dict(k99.COMMON, bb_lr=1e-4))


def test_run_contract_from_train_size():
    c = m.run_contract(3455, m.dense_args(dict(k99.COMMON)))
    assert (c['len_train_loader'], c['steps'], c['scheduler_total_steps'], c['max_fp16_skips'], c['eval_batch']) == (431, 3448, 3448, 34, 4)
    assert m.run_contract(3000, m.dense_args(dict(k99.COMMON)))['steps'] == 375 * 8


def test_corpus_checks_accept_only_a_slot_table_change():
    spec64 = dict(slots=[list(s) for s in raptor_common.SLOTS64], img=336, crop_mm=140.0, span=[0.02, 0.98], depth=64, normalization='n',
                  order='o', resize='r', series_selection='s')
    base = dict(contract=m.BASE_CONTRACT, spec=spec64, studies=2, shard_size=400, order='x', builder='b', archive=dict(directory_sha256='z'), finished=True,
                shards=[dict(done=True)])
    dense = copy.deepcopy(base); dense.update(contract=m.CONTRACT); dense['spec'].update(slots=SPEC80['slots'], depth=80)
    ids = np.array(['a', 'b'])
    assert m.corpus_checks(dense, base, ext.SLOTS80, ids, ids) == []
    assert m.corpus_checks(dense, base, ext.SLOTS80, ids, ids[::-1]) == ['study ids differ from the base cache']
    bad = copy.deepcopy(dense); bad['spec']['crop_mm'] = 120.0; bad['finished'] = False
    assert set(m.corpus_checks(bad, base, ext.SLOTS80, ids, ids)) == {'spec differs: crop_mm', 'dense cache not finished'}


# ------------------------------------------------------------------ the real 80-slice cache
@built
def test_real_cache_manifest_ids_and_masks():
    d, b = Corpus(DENSE), Corpus(BASE)
    assert d.contract == 'raptor80_336_c140_s02' and d.depth == 80 and d.manifest['finished'] is True
    assert len(d) == len(b) == d.manifest['studies'] and np.array_equal(d.ids, b.ids)
    assert d.masks.shape == (len(d), 80) and d.masks.dtype == np.uint8 and set(np.unique(d.masks).tolist()) <= {0, 1}
    assert m.corpus_checks(d.manifest, b.manifest, ext.SLOTS80, d.ids, b.ids) == []
    reg = d.manifest['contract_registry']
    assert reg['module_sha256'] == m.sha256(ROOT / 'training/src/raptor_contracts_ext.py')
    assert reg['raptor_common_sha256'] == m.read(ROOT / m.V4)['training_code_sha256']['raptor_common.py']
    assert d.manifest['mask_sha256'] == m.sha256(DENSE / 'masks.npy') and d.manifest['ids_sha256'] == m.sha256(DENSE / 'ids.npy')
    # a whole missing series is missing in both layouts (same series selection); 80 slots never have fewer positive slices per study
    off80, off64 = np.cumsum([0, 22, 18, 15, 10, 15]), np.cumsum([0, 18, 14, 12, 8, 12])
    for j in range(5):
        e80 = ~d.masks[:, off80[j]:off80[j + 1]].any(1); e64 = ~b.masks[:, off64[j]:off64[j + 1]].any(1)
        assert np.array_equal(e80, e64)
    assert (d.masks.sum(1) >= b.masks.sum(1)).mean() > 0.99


# ------------------------------------------------------------------ the registered protocol
@pytest.fixture()
def protocol():
    return m.read(REAL)


@pytest.fixture()
def env(monkeypatch):
    for k, v in {**m.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)


def _args(p, mode='full', output_root=None, extra=()):
    argv = m.runner_argv(p, mode, output_root=output_root)[2:] + list(extra)
    return runner_module().build_parser().parse_args(argv)


@prepared
def test_protocol_registration(protocol):
    p = protocol
    assert p['schema'] == m.SCHEMA and p['plan_item'] == 'H-B-22' and p['status'] == 'REGISTERED_NO_EXECUTION'
    assert p['initial_state_sha256'] == m.read(ROOT / m.V4)['initial_state_sha256'] and p['initial_state_sha256'].startswith('e4ea3ae6')
    assert p['common_args'] == m.dense_args(m.read(ROOT / m.V4)['common_args'])
    s = p['split']
    assert (s['n_train'], s['n_val'], s['len_train_loader'], s['steps'], s['max_fp16_skips'], s['eval_batch']) == (3455, 888, 431, 3448, 34, 4)
    assert s['pos_weight'] == m.read(ROOT / m.V4)['loss']['baseline_pos_weight']
    assert p['corpus']['manifest_sha256'] == m.sha256(DENSE / 'manifest.json') and p['corpus']['depth'] == 80
    assert p['base_corpus']['manifest_sha256'] == m.read(ROOT / m.V4)['corpus_manifest_sha256']
    for path, expected in p['source_bindings'].items():
        assert m.sha256(ROOT / path) == expected, path
    assert [e['file'] for e in p['extra_evals']] == ['oof_k24.csv', 'oof_allpos.csv'] and p['tags']['full'] == 'EXP102_B8_DENSE80_F0'


@prepared
def test_runner_guard_accepts_registered_argv_and_rejects_changes(protocol, env, tmp_path):
    p = protocol
    got, ordered, out_root = m.validate_arguments(_args(p), ROOT)
    assert out_root == (ROOT / 'artifacts/experiments').resolve() and len(ordered['train']) == 3455
    m.validate_arguments(_args(p, 'baseline-eval'), ROOT)
    m.validate_arguments(_args(p, 'smoke', output_root=tmp_path), ROOT)
    for extra, message in [(['--k', '12'], 'frozen argument changed: k'), (['--k-eval', '24'], 'frozen argument changed: k_eval'),
                           (['--tag', 'EXP102_X'], 'tag/mode'), (['--workers', '4'], 'registered worker count'), (['--seed', '43'], 'seed'),
                           (['--corpus', str(BASE)], 'corpus'), (['--output-root', str(tmp_path)], 'artifacts/experiments')]:
        with pytest.raises(ValueError, match=message):
            m.validate_arguments(_args(p, extra=extra), ROOT)
    with pytest.raises(ValueError, match='corpus'):
        m.validate_arguments(_args(p, 'baseline-eval', extra=['--corpus', str(DENSE)]), ROOT)
    with pytest.raises(ValueError, match='outside artifacts'):
        m.validate_arguments(_args(p, 'smoke', output_root=ROOT / 'artifacts' / 'tmp-smoke'), ROOT)


@prepared
def test_runner_guard_requires_the_thread_environment(protocol, env, monkeypatch):
    monkeypatch.delenv('OMP_NUM_THREADS')
    with pytest.raises(ValueError, match='OMP_NUM_THREADS'):
        m.validate_arguments(_args(protocol), ROOT)


@prepared
def test_queue_entry(protocol):
    entries = m.queue_entries(protocol, after='EXP099_B8_DREAD_F4', with_baseline=True)
    full, base = entries
    assert full['tag'] == 'EXP102_B8_DENSE80_F0' and full['after'] == 'EXP099_B8_DREAD_F4' and full['continue_on_failure'] is True
    cmd = full['command']
    assert cmd[1].endswith('exp102_dense80.py') and cmd[cmd.index('--k') + 1] == '16' and cmd[cmd.index('--k-eval') + 1] == '31'
    assert Path(cmd[cmd.index('--corpus') + 1]).resolve() == DENSE.resolve() and cmd[cmd.index('--workers') + 1] == '12'
    assert '--grad-ckpt' in cmd and cmd[cmd.index('--amp') + 1] == 'fp16' and cmd[cmd.index('--mode') + 1] == 'full'
    assert base['after'] == full['tag'] and Path(base['command'][base['command'].index('--corpus') + 1]).resolve() == BASE.resolve()
    assert m.read(ROOT / m.PROTOCOL / 'queue.json') == entries
