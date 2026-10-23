"""EXP123 CPU checks. Every constructed model is explicitly kept on CPU."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'src'))
from tools import exp123_qwen3vl_pilot as guard  # noqa: E402

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
torch.set_default_device('cpu')


@pytest.fixture(scope='module')
def runner():
    spec = importlib.util.spec_from_file_location('exp123_test_runner', ROOT / guard.RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.set_num_threads(2)
    return mod


def test_slot_pairing_and_processor_patch_order(runner):
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    pairs, cross = runner.pair_indices([1, 3, 18, 19], [16, 16])
    assert pairs == [(0, 1), (2, 3)] and cross == 0
    frames = torch.randn(2, 2, 3, 64, 64, generator=torch.Generator(device='cpu').manual_seed(123), device='cpu')
    flat, grid = runner.patchify_pairs(frames)
    assert flat.shape == (32, 1536) and grid.tolist() == [[1, 4, 4]] * 2
    processor = Qwen3VLVideoProcessor.__new__(Qwen3VLVideoProcessor)
    ref, gt, gh, gw = processor.patchify(frames[:1], patch_size=16, merge_size=2, temporal_patch_size=2)
    assert (gt, gh, gw) == (1, 4, 4) and torch.equal(flat[:16], ref[0])
    assert not torch.cuda.is_initialized()


def tiny_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    cfg = Qwen3VLConfig()
    t = cfg.text_config
    t.vocab_size = 128
    t.hidden_size = 64
    t.intermediate_size = 128
    t.num_hidden_layers = 2
    t.num_attention_heads = 4
    t.num_key_value_heads = 2
    t.head_dim = 16
    t.max_position_embeddings = 512
    t.rope_parameters = dict(mrope_interleaved=True, mrope_section=[2, 3, 3], rope_type='default', rope_theta=5000000.0)
    v = cfg.vision_config
    v.hidden_size = 64
    v.intermediate_size = 128
    v.out_hidden_size = 64
    v.depth = 2
    v.num_heads = 4
    v.num_position_embeddings = 64
    v.deepstack_visual_indexes = [0, 1]
    cfg.image_token_id = 100
    cfg.vision_start_token_id = 101
    cfg.vision_end_token_id = 102
    cfg.video_token_id = 103
    return cfg


def test_random_tiny_model_cpu_forward_and_trainable_set(runner):
    cfg = tiny_config()
    model, _, report = runner.build_model('unused', random_config=cfg, base_dtype='float32', grad_ckpt=False)
    model.to('cpu').eval()
    assert report['random_config'] and report['missing_keys'] == [] and report['unexpected_keys'] == []
    assert report['lora_modules'] == 14 and report['params_trainable'] < report['params_total']
    assert report['params_vision'] > 0 and report['params_head'] > 0
    ids = torch.tensor([[1, 101, 100, 100, 100, 100, 102, 2]], device='cpu')
    types = (ids == 100).long()
    pv, grid = runner.patchify_pairs(torch.zeros((1, 2, 3, 64, 64), device='cpu'))
    with torch.no_grad():
        logits = model(ids, types, pv, grid)
    assert logits.shape == (1, 12) and logits.dtype == torch.float32 and torch.isfinite(logits).all()
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith('core.visual.') for n in names)
    assert any('lora_' in n for n in names)
    assert not any(n.startswith('core.language_model.') and 'lora_' not in n for n in names)
    runner.load_trainable(model, runner.trainable_state(model))
    assert not torch.cuda.is_initialized()


def test_tiny_subset_packing_cpu(runner):
    corpus = ROOT / guard.CORPUS
    if not (corpus / 'manifest.json').is_file():
        pytest.skip('corpus unavailable')
    _, val = guard.fold_ids()
    labels = {val[0]: np.zeros(12, np.float32)}
    mean, std, patch, merge, temporal = runner.processor_stats(ROOT / guard.WEIGHTS)
    assert (patch, merge, temporal) == (16, 2, 2)
    ds = runner.PairStudies(corpus, val[:1], labels, 4, 64, False, 42, mean, std, patch=patch, merge=merge)
    item = ds[0]
    assert item['pixel_values'].shape == (32, 1536) and item['grid'].tolist() == [[1, 4, 4]] * 2
    assert not torch.cuda.is_initialized()


def test_registration_and_dependency(runner):
    entries = guard.queue_entries()
    assert len(entries) == 2
    pilot, probe = entries
    assert pilot['tag'] == guard.TAG and pilot['exclusive'] and not pilot['continue_on_failure']
    assert probe['after'] == pilot['tag'] and probe['exclusive'] and not probe['continue_on_failure']
    assert Path(pilot['command'][1]).resolve() == (ROOT / guard.RUNNER).resolve()
    assert Path(probe['command'][1]).resolve() == (ROOT / guard.PROBE).resolve()
    manifest = ROOT / guard.PROTOCOL / 'manifest.json'
    if manifest.is_file():
        m = json.loads(manifest.read_text(encoding='utf-8'))
        assert m['args'] == guard.RECIPE and m['n_train'] == 3455 and m['n_val'] == 888
        assert m['model']['hf_repo'] == guard.HF_REPO
    assert runner.gate_decision([1.5] * 48, 48, 3455, 3, 888, 24, 36, .35, 10, 4, 14.5)['status'] == 'GATE_FAILED'
    assert not torch.cuda.is_initialized()
