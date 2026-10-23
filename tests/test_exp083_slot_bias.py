"""Synthetic CPU checks; no corpus pixels, baseline model, Gold or GPU phase."""
from copy import deepcopy
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import exp083_slot_bias as g


def initial_state(features=8):
    norm = torch.nn.LayerNorm(features)
    att = torch.nn.Sequential(torch.nn.Linear(features, 256), torch.nn.Tanh(),
                              torch.nn.Dropout(.2), torch.nn.Linear(256, 12))
    return {**{'norm.' + k: v for k, v in norm.state_dict().items()},
            **{'att.' + k: v for k, v in att.state_dict().items()},
            'clsW': torch.arange(12 * features, dtype=torch.float32).reshape(12, features) / 100,
            'clsb': torch.linspace(-1, 1, 12)}


def original_head(head, features):
    h = head.norm(features)
    attention = torch.softmax(head.att(h), dim=1)
    pooled = torch.einsum('bkn,bkf->bnf', attention, h)
    return (pooled * head.clsW).sum(-1) + head.clsb


def test_actual_slot_boundaries_and_repeated_centers():
    mask = np.ones(64)
    centers = [1, 16, 19, 30, 33, 42, 45, 50, 53, 62] + [33] * 14
    assert g.slots_for_centers(centers, mask) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] + [2] * 14
    assert g.centers_from_mask(mask)[0] == 1 and g.centers_from_mask(mask)[-1] == 62
    for boundary in (0, 17, 18, 31, 32, 43, 44, 51, 52, 63):
        with pytest.raises(ValueError, match='boundary'):
            g.slots_for_centers([boundary] * 24, mask)


def test_invalid_mask_and_empty_are_not_new_padding_or_missingness_rules():
    mask = np.ones(64); mask[20] = 0
    with pytest.raises(ValueError, match='zero cache mask'):
        g.slots_for_centers([20] * 24, mask)
    assert 19 not in g.centers_from_mask(mask) and 20 not in g.centers_from_mask(mask)
    assert g.centers_from_mask(np.zeros(64)) == [-1] * 24
    assert g.slots_for_centers([-1] * 24, np.zeros(64)) == [-1] * 24
    with pytest.raises(ValueError, match='Empty'):
        g.slots_for_centers([-1] * 24, np.ones(64))
    with pytest.raises(ValueError, match='Empty'):
        g.slots_for_centers([-1, 2] * 12, np.zeros(64))


@pytest.mark.parametrize('autocast', [False, True])
def test_zero_bias_exact_initial_logits_with_shared_original_head(autocast):
    state = initial_state(); a = g.make_head(state, False).eval(); b = g.make_head(state, True).eval()
    features = torch.arange(2 * 24 * 8, dtype=torch.float32).reshape(2, 24, 8) / 173
    slots = torch.tensor([[0, 1, 2, 3, 4, -1] * 4] * 2, dtype=torch.int64)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
        original = original_head(a, features)
        assert torch.equal(original, a(features, slots))
        assert torch.equal(original, b(features, slots))
    assert g.shared_head_hash(a) == g.shared_head_hash(b) == g.state_hash(state)
    assert a.slot_bias.requires_grad is False and b.slot_bias.requires_grad is True
    assert not ({p.data_ptr() for p in a.parameters()} & {p.data_ptr() for p in b.parameters()})
    assert sum(p.numel() for p in b.parameters() if p.requires_grad) - sum(p.numel() for p in a.parameters() if p.requires_grad) == 60


def test_parameter_bias_cast_cannot_promote_half_attention_and_empty_stays_zero():
    head = g.make_head(initial_state(), True)
    with torch.no_grad(): head.slot_bias.fill_(.125)
    logits = torch.arange(2 * 24 * 12, dtype=torch.float16).reshape(2, 24, 12)
    slots = torch.tensor([[-1] * 24, [2] * 24], dtype=torch.int64)
    changed = head.biased_attention(logits, slots)
    assert changed.dtype == logits.dtype == torch.float16
    assert torch.equal(changed[0], logits[0])
    torch.testing.assert_close(changed[1], logits[1] + torch.tensor(.125, dtype=torch.float16), rtol=0, atol=0)
    changed.float().sum().backward()
    assert torch.count_nonzero(head.slot_bias.grad[:2]) == 0
    assert torch.count_nonzero(head.slot_bias.grad[3:]) == 0


def test_empty_windows_keep_nonzero_encoded_features_and_zero_bias_gradient():
    head = g.make_head(initial_state(), True).eval()
    # A blank normalized image can have nonzero encoder output. Do not erase it.
    features = torch.linspace(-2, 3, 24 * 8).reshape(1, 24, 8)
    slots = torch.full((1, 24), -1, dtype=torch.int64)
    output = head(features, slots)
    assert torch.equal(output, original_head(head, features))
    output.sum().backward()
    assert torch.count_nonzero(head.slot_bias.grad) == 0


def test_each_arm_reset_and_paired_forward_rng_trajectory():
    state = initial_state(); heads = [g.make_head(state, v).train() for v in (False, True)]
    features = [torch.linspace(-2, 1, n * 24 * 8).reshape(n, 24, 8) for n in (3, 2, 1)]
    traces, outputs = [], []
    for head in heads:
        torch.manual_seed(42)
        trace, values = [], []
        for x in features:
            before = torch.get_rng_state().clone()
            value = head(x, torch.zeros(x.shape[:2], dtype=torch.int64))
            trace.append((before, torch.get_rng_state().clone()))
            values.append(value.detach())
        traces.append(trace); outputs.append(values)
        _ = torch.rand(100)  # Previous-arm work is erased by next arm's reset.
    for i in range(3):
        assert torch.equal(traces[0][i][0], traces[1][i][0])
        assert torch.equal(traces[0][i][1], traces[1][i][1])
        assert torch.equal(outputs[0][i], outputs[1][i])
    assert not torch.equal(traces[0][0][0], traces[0][0][1])
    assert not torch.cuda.is_initialized()


def test_native_fp16_feature_values_are_not_quantized_or_silently_cast():
    x = torch.linspace(-1, 1, 48, dtype=torch.float16).reshape(1, 24, 2)
    y = g.native_feature_array(x, (1, 24, 2))
    assert y.dtype == np.float16 and y.tobytes() == x.numpy().tobytes()
    with pytest.raises(TypeError, match='quantizing'):
        g.native_feature_array(x.float(), x.shape)
    with pytest.raises(ValueError):
        g.native_feature_array(x, (2, 24, 2))


def test_full_phase_requires_exact_pilot_and_explicit_root_decision(tmp_path):
    protocol = tmp_path / 'protocol.json'; protocol.write_text('{}')
    m = {'phases': {'pilot_max_seconds': 300, 'pilot_ids': ['a', 'b']}, 'encoder_state_sha256': 'encoder'}
    assert g.validate_phase_authority('pilot', protocol, m) == 300
    with pytest.raises(ValueError): g.validate_phase_authority('full', protocol, m)
    pilot = tmp_path / 'pilot.json'; decision = tmp_path / 'decision.json'
    p = dict(status='PASS', phase='pilot', protocol_sha256=g.digest(protocol), pilot_ids=['a', 'b'],
             elapsed_seconds=70., feature_dtype='float16', initial_logits_exact=True,
             initial_train_logits_exact=True, native_cache_roundtrip_exact=True,
             native_cache_logits_exact=True, extraction_split_loaders=True, disjoint_initial_head_storage=True,
             encoder_before_sha256='encoder', encoder_after_sha256='encoder')
    pilot.write_text(json.dumps(p))
    d = dict(decision='RUN_ONE_FULL_PHASE', protocol_sha256=g.digest(protocol),
             pilot_receipt_sha256=g.digest(pilot), max_total_seconds=2700, automatic_retries=0)
    decision.write_text(json.dumps(d))
    assert g.validate_phase_authority('full', protocol, m, pilot, decision) == 2630
    p['feature_dtype'] = 'float32'; pilot.write_text(json.dumps(p))
    with pytest.raises(ValueError): g.validate_phase_authority('full', protocol, m, pilot, decision)


def test_imports_do_not_load_torch_common_or_start_phase():
    script = "import importlib.util,sys;from pathlib import Path;root=Path(sys.argv[1]);sys.path.insert(0,str(root));from tools import exp083_slot_bias;spec=importlib.util.spec_from_file_location('exp083_passive',root/'training/runners/exp083_slot_bias.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);assert 'torch' not in sys.modules;assert 'raptor_common' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', script, str(g.ROOT)], capture_output=True, text=True,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr


def test_runner_has_no_scoring_gold_loading_or_half_cast():
    tree = ast.parse((g.ROOT / g.RUNNER).read_text(encoding='utf-8'))
    forbidden = {'gold_labels', 'macro_auc', 'roc_auc_score', 'evaluate_labels'}
    assert not any(isinstance(n, ast.Name) and n.id in forbidden for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'half' for n in ast.walk(tree))
    assert not any(isinstance(n, ast.ImportFrom) and n.module == 'raptor_train' for n in ast.walk(tree))
