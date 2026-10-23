"""CPU-only checks for EXP115 (H-B-43: Qwen3.5-2B VLM + LoRA fold-0 pilot; tools/exp115_vlm_pilot.py, training/runners/exp115_vlm_pilot.py,
tools/exp115_vlm_eval.py, tools/exp115_fp16_probe.py).

Window choice equals the CoAtNet K24 evaluation (eval) / B8 sampler (train); packing follows the Qwen3-VL patchify order; the head gives
(1, 12) logits; a 2-update CPU smoke (truncated layer counts, 2 studies, 2 frame pairs) runs the runner end to end and writes predictions in the
oof_k36.csv format and weights every epoch; the evaluation reproduces the stop rule; the queue entry is a valid tools/gpu_queue.py job that the
runner guard accepts. Run with the .venv-vlm interpreter. No GPU: subprocesses get CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'training' / 'src'))
from tools import exp115_vlm_pilot as g  # noqa: E402

WEIGHTS = ROOT / g.WEIGHTS
CORPUS = ROOT / g.CORPUS
RUNNER = ROOT / g.RUNNER
MANIFEST = ROOT / g.PROTOCOL / 'manifest.json'
SIZES = [18, 14, 12, 8, 12]
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
weights = pytest.mark.skipif(not (WEIGHTS / g.WEIGHT_FILES[-1]).is_file(), reason='Qwen3.5-2B weights not present')
corpus = pytest.mark.skipif(not (CORPUS / 'manifest.json').is_file(), reason='64-slot cache not present')
prepared = pytest.mark.skipif(not MANIFEST.is_file(), reason='EXP115 protocol not prepared')
CPU_ENV = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
           'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1'}


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def r():
    return load_path('exp115_runner_test', RUNNER)


# ------------------------------------------------------------------ pairing / windows
def test_pairing_covers_every_window_once_and_keeps_slots(r):
    centers = [1, 4, 7, 10, 13, 16, 19, 22, 26, 29, 33, 36, 40, 43, 46, 51, 54, 57, 60, 61, 62, 52, 53, 55]
    centers = sorted(centers)
    pairs, cross = r.pair_indices(centers, SIZES)
    assert len(pairs) == 12 and sorted(i for p in pairs for i in p) == list(range(24))
    slots = [r.slot_of(c, SIZES) for c in centers]
    n_odd = sum(1 for s in set(slots) if slots.count(s) % 2)
    assert cross == n_odd // 2  # only odd leftovers cross slots
    assert sum(slots[a] != slots[b] for a, b in pairs) == cross
    assert [p[0] for p in pairs] == sorted(p[0] for p in pairs)


def test_pairing_empty_and_odd(r):
    pairs, cross = r.pair_indices([-1] * 4, SIZES)
    assert pairs == [(0, 1), (2, 3)] and cross == 0
    pairs, cross = r.pair_indices([2, 3, 5], SIZES)  # odd count: the leftover pairs with itself
    assert pairs == [(0, 1), (2, 2)] and cross == 0
    with pytest.raises(ValueError):
        r.pair_indices([5, 3], SIZES)
    with pytest.raises(ValueError):
        r.pair_indices([-1, 3], SIZES)


@corpus
def test_eval_windows_are_the_coatnet_k24_windows_and_train_is_the_b8_sampler(r):
    from raptor_common import Corpus, select_centers, study_rng
    c = Corpus(CORPUS)
    train, val = g.fold_ids()
    for u in val[:20]:
        assert r.study_centers(c.mask(u), c.depth, c.spec, 24, False, 42, 0, u) == select_centers(c.mask(u), c.depth, 24, c.spec, 'within-slot-valid')
    for u in train[:20]:
        b8 = select_centers(c.mask(u), c.depth, 24, c.spec, 'within-slot-valid', study_rng(42, 1, u))
        got = r.study_centers(c.mask(u), c.depth, c.spec, 24, True, 42, 1, u)
        assert got == sorted(x for x in b8 if x >= 0) + [-1] * b8.count(-1)


# ------------------------------------------------------------------ packing
def test_patchify_matches_qwen3vl_video_processor(r):
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    g_ = torch.Generator().manual_seed(0)
    frames = torch.randn(3, 2, 3, 64, 96, generator=g_)
    flat, grid = r.patchify_pairs(frames, 16, 2)
    assert flat.shape == (3 * 4 * 6, 3 * 2 * 16 * 16) and grid.tolist() == [[1, 4, 6]] * 3
    vp = Qwen3VLVideoProcessor.__new__(Qwen3VLVideoProcessor)
    for i in range(3):
        ref, gt, gh, gw = vp.patchify(frames[i:i + 1], patch_size=16, merge_size=2, temporal_patch_size=2)
        assert (gt, gh, gw) == (1, 4, 6)
        assert torch.equal(ref[0], flat[i * 24:(i + 1) * 24])


@corpus
@weights
def test_packed_study_shapes_at_384(r):
    train, val = g.fold_ids()
    mean, std, patch, merge, tps = r.processor_stats(WEIGHTS)
    assert (mean, std, patch, merge, tps) == ([.5] * 3, [.5] * 3, 16, 2, 2)
    labels = {u: np.full(12, .25, np.float32) for u in val[:2]}
    ds = r.PairStudies(CORPUS, val[:2], labels, 24, 384, False, 42, mean, std)
    s = ds[0]
    assert tuple(s['pixel_values'].shape) == (12 * 576, 1536) and s['pixel_values'].dtype == torch.float32
    assert s['grid'].tolist() == [[1, 24, 24]] * 12 and tuple(s['y'].shape) == (12,)
    assert float(s['pixel_values'].min()) >= -1.0 and float(s['pixel_values'].max()) <= 1.0
    s36 = r.PairStudies(CORPUS, val[:1], labels, 36, 384, False, 42, mean, std)[0]
    assert tuple(s36['pixel_values'].shape) == (18 * 576, 1536)
    tds = r.PairStudies(CORPUS, val[:1], labels, 24, 384, True, 42, mean, std, jitter=0.2)
    a, b = tds[0], tds[0]
    assert torch.equal(a['pixel_values'], b['pixel_values'])  # deterministic per (seed, epoch, uid)
    tds.set_epoch(1)
    assert not torch.equal(tds[0]['pixel_values'], a['pixel_values'])


@weights
def test_prompt_tokens(r):
    from transformers import AutoConfig, AutoTokenizer
    tok, cfg = AutoTokenizer.from_pretrained(WEIGHTS), AutoConfig.from_pretrained(WEIGHTS)
    ids_, types, text = r.build_prompt(tok, 12, 144, r.token_ids(cfg))
    assert ids_.count(cfg.image_token_id) == 12 * 144 == sum(types)
    assert ids_.count(cfg.vision_start_token_id) == 12 and text.endswith('<|im_start|>assistant\n')
    assert 1728 < len(ids_) < 1900


# ------------------------------------------------------------------ model
@weights
def test_head_output_shape_and_trainable_set(r):
    model, cfg, rep = r.build_model(WEIGHTS, text_layers=4, vision_layers=1, base_dtype='bfloat16', grad_ckpt=True)
    assert rep['text_layers'] == 4 and rep['vision_layers'] == 1 and rep['lora_modules'] == 3 * 8 + 7  # 3 DeltaNet layers x (5 attn + 3 mlp) + 1 full-attention layer x (4 + 3)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith('core.visual.') for n in names) and any('lora_' in n for n in names) and any(n.startswith('head.') for n in names)
    assert not any(n.startswith('core.language_model.') and 'lora_' not in n for n in names)
    from transformers import AutoTokenizer
    ids_, types, _ = r.build_prompt(AutoTokenizer.from_pretrained(WEIGHTS), 1, 4, r.token_ids(cfg))
    frames = torch.zeros(1, 2, 3, 64, 64)
    pv, grid = r.patchify_pairs(frames)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([ids_]), torch.tensor([types]), pv, grid, amp_dtype=torch.bfloat16)
    assert out.shape == (1, 12) and out.dtype == torch.float32 and torch.isfinite(out).all()
    state = r.trainable_state(model)
    r.load_trainable(model, state)
    with pytest.raises(RuntimeError):
        r.load_trainable(model, {k: v for k, v in list(state.items())[1:]})


def test_schedule_accumulation_and_gate(r):
    f = r.lr_lambda_factory(10, 100)
    assert f(0) == pytest.approx(.1) and f(9) == pytest.approx(1.0) and f(100) == pytest.approx(0.0)
    assert r.accumulation_groups(10, 4) == [4] * 8 + [2, 2]
    times = [10.0] * 16 + [1.0] * 32
    ok = r.gate_decision(times, 48, 3455, 3, 888, 24, 36, .35, 12.0, 4.0, 14.5)
    assert ok['status'] == 'GATE_PASSED' and ok['mean_micro_step_seconds'] == pytest.approx(1.0)
    assert ok['projected_total_hours'] == pytest.approx((3455 * 3 + .35 * 888 * 2.5) / 3600)
    assert r.gate_decision([2.0] * 48, 48, 3455, 3, 888, 24, 36, .35, 12.0, 4.0, 14.5)['status'] == 'GATE_FAILED'
    assert r.gate_decision([1.0] * 48, 48, 3455, 3, 888, 24, 36, .35, 15.0, 4.0, 14.5)['status'] == 'GATE_FAILED'


def test_unexpected_keys_rule(r):
    assert r.unexpected_key_ok('mtp.layers.0.mlp.up_proj.weight', 24, 24)
    assert r.unexpected_key_ok('model.language_model.layers.5.mlp.up_proj.weight', 4, 24)
    assert not r.unexpected_key_ok('model.language_model.layers.3.mlp.up_proj.weight', 4, 24)
    assert not r.unexpected_key_ok('model.visual.merger.linear_fc1.weight', 24, 24)


def test_prediction_csv_format(r, tmp_path):
    ref_header = (ROOT / g.COATNET_F0_K36).read_text(encoding='utf-8').splitlines()[0] if (ROOT / g.COATNET_F0_K36).is_file() else None
    ids_ = ['1.2.3', '1.2.4']
    p = r.write_predictions(tmp_path / 'oof_k24.csv', ids_, np.full((2, 12), .5))
    lines = p.read_text(encoding='utf-8').splitlines()
    assert lines[0].split(',') == [r.UID, *r.LAB] and len(lines) == 3
    if ref_header:
        assert lines[0] == ref_header
    for bad in (np.full((2, 12), np.nan), np.full((2, 12), 1.5), np.full((2, 11), .5)):
        with pytest.raises(ValueError):
            r.write_predictions(tmp_path / 'bad.csv', ids_, bad)


# ------------------------------------------------------------------ 2-update CPU smoke (subprocess, the real runner)
@pytest.fixture(scope='module')
def smoke(tmp_path_factory):
    if not (WEIGHTS / g.WEIGHT_FILES[-1]).is_file() or not (CORPUS / 'manifest.json').is_file():
        pytest.skip('weights or cache missing')
    out = tmp_path_factory.mktemp('exp115') / 'smoke'
    argv = [sys.executable, str(RUNNER), '--mode', 'smoke', '--tag', 'EXP115_SMOKE', '--out', str(out), '--device', 'cpu', '--text-layers', '4',
            '--vision-layers', '2', '--k-train', '4', '--k-eval', '4', '--k-eval-extra', '6', '--max-train', '2', '--max-val', '2', '--epochs', '2',
            '--accum', '2', '--workers', '0']
    proc = subprocess.run(argv, cwd=ROOT, env={**os.environ, **CPU_ENV}, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return out


def test_smoke_trains_two_updates_and_writes_outputs(smoke, r):
    s = json.loads((smoke / 'summary.json').read_text(encoding='utf-8'))
    assert s['status'] == 'COMPLETED' and s['optimizer_updates'] == 2 and s['epochs'] == 2 and s['cuda_initialized'] is False
    h = json.loads((smoke / 'history.json').read_text(encoding='utf-8'))
    assert [x['optimizer_updates'] for x in h] == [1, 2] and all(x['epoch_seconds'] > 0 and x['mean_micro_step_seconds'] > 0 for x in h)
    assert all(np.isfinite(x['loss']) for x in h)
    for e in (1, 2):
        assert (smoke / f'weights_epoch{e}.pt').is_file()
    _, val = g.fold_ids()
    for name, k in (('oof_k4.csv', 4), ('oof_k6.csv', 6)):
        df = pd.read_csv(smoke / name, dtype={r.UID: str})
        assert list(df.columns) == [r.UID, *r.LAB] and df[r.UID].tolist() == val[:2]
        P = df[r.LAB].to_numpy()
        assert np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1
    inv = json.loads((smoke / 'output-inventory.json').read_text(encoding='utf-8'))
    assert {'oof_k4.csv', 'weights_epoch2.pt', 'summary.json', 'provenance.json'} <= set(inv)


def test_smoke_weights_reload_and_training_moved_vision_and_lora(smoke, r):
    ck = torch.load(smoke / 'weights_epoch2.pt', map_location='cpu', weights_only=False)
    st = ck['trainable']
    assert ck['epoch'] == 2 and ck['args']['text_layers'] == 4 and all(v.dtype == torch.float32 for v in st.values())
    assert any(k.startswith('core.visual.') for k in st) and any('lora_B' in k for k in st)
    assert any(float(v.abs().max()) > 0 for k, v in st.items() if 'lora_B' in k)  # LoRA B starts at zero
    model, _, _ = r.build_model(WEIGHTS, text_layers=4, vision_layers=2, base_dtype='float32', grad_ckpt=False)
    base = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith('core.visual.blocks.0.mlp.linear_fc1.weight')}
    r.load_trainable(model, st)
    key = 'core.visual.blocks.0.mlp.linear_fc1.weight'
    assert not torch.equal(base[key], dict(model.named_parameters())[key].detach())


def test_fp16_probe_cpu_tiny(smoke, tmp_path):
    probe = load_path('exp115_probe_test', ROOT / 'tools' / 'exp115_fp16_probe.py')
    out = tmp_path / 'probe'
    body = probe.main(['--checkpoint', str(smoke / 'weights_epoch2.pt'), '--n-studies', '2', '--device', 'cpu', '--precisions', 'fp16,fp32',
                       '--out', str(out)])
    assert (out / 'probe.json').is_file() and (out / 'probe_predictions.csv').is_file()
    d = body['decision']
    assert d['candidate'] == 'fp16' and d['serving_precision'] in ('fp16', 'fp32') and d['fp16_nonfinite'] == (body['runs']['fp16']['nonfinite_studies'] > 0)
    assert body['runs']['fp16']['n_modules'] > 50 and body['runs']['fp16']['top_modules'] and body['t4_projection'] == {}
    assert body['parity_vs_fp32']['fp16']['max_abs_diff'] is not None


# ------------------------------------------------------------------ evaluation and stop rule
def test_eval_stop_rule_on_known_predictions(tmp_path):
    if not (ROOT / g.COATNET_F0_K36).is_file():
        pytest.skip('comparators missing')
    from tools import exp115_vlm_eval as ev
    same = ev.evaluate(ROOT / g.COATNET_F0_K36, draws=10, name='same')
    assert same['decision']['status'] == 'CONTINUE' and same['decision']['delta'] == 0.0
    assert same['scores']['same']['dread']['macro'] == pytest.approx(0.8967668520457317, abs=1e-12)  # EXP099 fold-0 K36
    cov = same['references']['consensus']['fraction']
    assert 0.6 < cov < 0.95
    _, val = g.fold_ids()
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.random((len(val), 12)), columns=ev.T); noise.insert(0, ev.UID, val)
    p = tmp_path / 'random.csv'; noise.to_csv(p, index=False)
    bad = ev.evaluate(p, draws=10, name='random')
    assert bad['decision']['status'] == 'STOP' and bad['decision']['delta'] < -0.3
    assert set(bad['deltas']) == {'random - CoAtNet_f0_K36', f"random - {'RadR50_f0_K36_local' if 'RadR50_f0_K36_local' in bad['models'] else 'RadR50_f0_K36_T4'}",
                                  'rank(CoAtNet+random) - CoAtNet_f0_K36',
                                  f"rank(CoAtNet+{'RadR50_f0_K36_local' if 'RadR50_f0_K36_local' in bad['models'] else 'RadR50_f0_K36_T4'}+random) - "
                                  f"rank(CoAtNet+{'RadR50_f0_K36_local' if 'RadR50_f0_K36_local' in bad['models'] else 'RadR50_f0_K36_T4'})"}
    with pytest.raises(ValueError):
        ev.load_pred(p, val[:-1] + ['x'])


# ------------------------------------------------------------------ protocol and queue entry
@prepared
def test_queue_entry_is_a_valid_gpu_queue_job(r):
    q = json.loads((ROOT / g.PROTOCOL / 'queue-entry.json').read_text(encoding='utf-8'))
    assert isinstance(q, list) and len(q) == 1
    e = q[0]
    assert e['tag'] == g.TAG == 'EXP115_VLM_PILOT_F0' and e['after'] == 'EXP113_RADR50_LOCAL_F4' and e['continue_on_failure'] is True
    assert Path(e['command'][0]).resolve() == (ROOT / '.venv-vlm' / 'Scripts' / 'python.exe').resolve()
    assert Path(e['command'][1]).resolve() == RUNNER.resolve()
    a = r.build_parser().parse_args(e['command'][2:])
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert a.mode == 'full' and a.tag == g.TAG and Path(a.protocol).resolve() == MANIFEST.resolve()
    assert all(getattr(a, k) == v for k, v in m['args'].items())
    assert m['args'] == g.RECIPE and m['n_train'] == 3455 and m['n_val'] == 888
    probe_q = json.loads((ROOT / g.PROTOCOL / 'queue-entry-probe.json').read_text(encoding='utf-8'))
    assert probe_q[0]['after'] == g.TAG and probe_q[0]['continue_on_failure'] is False


@prepared
def test_guard_accepts_registered_and_rejects_changed_arguments(r, monkeypatch):
    for k, v in g.REGISTERED_ENV.items():
        monkeypatch.setenv(k, v)
    e = json.loads((ROOT / g.PROTOCOL / 'queue-entry.json').read_text(encoding='utf-8'))[0]
    a = r.build_parser().parse_args(e['command'][2:])
    m, train, val = g.validate_arguments(a, ROOT, hash_weights=False)
    assert len(train) == 3455 and len(val) == 888 and g.uid_hash(val) == m['split']['ordered_ids']['val_sha256']
    for flag, value in (('--epochs', '4'), ('--lora-r', '8'), ('--k-eval', '36'), ('--text-layers', '4')):
        bad = r.build_parser().parse_args(e['command'][2:] + [flag, value])
        with pytest.raises(ValueError):
            g.validate_arguments(bad, ROOT, hash_weights=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    with pytest.raises(ValueError):
        g.validate_arguments(a, ROOT, hash_weights=False)
