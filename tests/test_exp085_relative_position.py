"""CPU synthetic EXP085 checks; never open real data, checkpoints or runners."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/src'))
from tools import exp085_relative_position as g


@pytest.fixture(autouse=True)
def bounded_cpu_rng():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        yield
    torch.set_num_threads(previous_threads)


class SyntheticB(nn.Module):
    """Small version of B's actual per-finding attention, not mean/max pooling."""
    def __init__(self, features=8):
        super().__init__()
        self.backbone = nn.Linear(features, features)
        self.norm = nn.LayerNorm(features)
        self.att = nn.Sequential(nn.Linear(features, 16), nn.Tanh(), nn.Dropout(.2), nn.Linear(16, 12))
        self.clsW = nn.Parameter(torch.randn(12, features) * .15)
        self.clsb = nn.Parameter(torch.randn(12) * .03)

    def encode(self, x):
        return self.backbone(x)

    def head(self, features):
        h = self.norm(features)
        a = self.att(h).softmax(dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x):
        return self.head(self.encode(x))


def paired_models():
    base = SyntheticB()
    torch.manual_seed(456)
    control = g.make_model(deepcopy(base), False)
    torch.manual_seed(456)
    treatment = g.make_model(deepcopy(base), True)
    return base, control, treatment


def example():
    x = torch.randn(2, 6, 8)
    slots = torch.tensor([[0, 0, 0, 2, 2, -1], [1, 1, 1, 3, 3, -1]])
    positions = torch.tensor([[.1, .4, .85, .15, .75, 0], [.2, .45, .8, .2, .9, 0]])
    return x, slots, positions


def activate_residual(model):
    with torch.no_grad():
        model.projection.weight.normal_(0, .08)
        model.relative[-1].weight.normal_(0, .15)
    return model.eval()


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_initial_logits_exactly_preserve_original_b(training, empty):
    base, control, treatment = paired_models()
    x, slots, positions = example()
    if empty:
        slots.fill_(-1)
        positions.zero_()
    for model in (base, control, treatment):
        model.train(training)
    state = torch.get_rng_state()
    expected = base(x)
    torch.set_rng_state(state)
    actual_control = control(x, slots, positions)
    torch.set_rng_state(state)
    actual_treatment = treatment(x, slots, positions)
    assert torch.equal(expected, actual_control)
    assert torch.equal(expected, actual_treatment)
    assert torch.isfinite(expected).all()


def test_initial_state_parameter_count_and_trainability_are_paired_without_aliasing():
    base, control, treatment = paired_models()
    cs, ts = control.state_dict(), treatment.state_dict()
    assert cs.keys() == ts.keys()
    assert all(torch.equal(cs[key], ts[key]) for key in cs)
    assert g.state_hash(cs) == g.state_hash(ts)
    assert all(torch.equal(base.state_dict()[key], control.base.state_dict()[key]) for key in base.state_dict())
    parameters = lambda m: sum(p.numel() for p in m.parameters())
    assert parameters(control) == parameters(treatment) == parameters(base) + 4 * 8 * 64 + 64
    assert all(p.requires_grad for m in (control, treatment) for p in m.parameters())
    assert not ({p.data_ptr() for p in control.parameters()} & {p.data_ptr() for p in treatment.parameters()})
    assert not control.projection.weight.any()
    assert not control.relative[-1].weight.any()
    assert control.qkv.weight.any() and control.relative[0].weight.any()


def test_masks_prevent_cross_slot_and_invalid_key_information_leakage():
    _, _, model = paired_models()
    activate_residual(model)
    features, slots, positions = example()
    expected = model.contextualize(features, slots, positions)
    changed = features.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 19
    actual = model.contextualize(changed, slots, positions)
    assert torch.equal(expected[:, :3], actual[:, :3])
    assert torch.equal(expected[:, -1], features[:, -1])
    assert torch.equal(actual[:, -1], changed[:, -1])
    assert not torch.equal(expected[:, 3:5], actual[:, 3:5])


@pytest.mark.parametrize('k', [1, 6])
def test_all_invalid_preserves_blank_encodings_and_is_finite_with_backward(k):
    _, _, model = paired_models()
    activate_residual(model)
    features = torch.randn(2, k, 8, requires_grad=True)
    slots = torch.full((2, k), -1, dtype=torch.int64)
    positions = torch.zeros(2, k)
    contextualized = model.contextualize(features, slots, positions)
    assert torch.equal(contextualized, features)
    loss = model.head(features, slots, positions).square().sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_joint_permutation_preserves_pooling_after_residual_activation():
    _, _, model = paired_models()
    activate_residual(model)
    features, slots, positions = example()
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    original = model.head(features, slots, positions)
    permuted = model.head(features[:, permutation], slots[:, permutation], positions[:, permutation])
    torch.testing.assert_close(original, permuted, rtol=2e-6, atol=2e-7)


def test_relative_position_path_learns_after_multiple_optimizer_steps():
    _, control, treatment = paired_models()
    features, slots, positions = example()
    target = torch.randn(2, 12)  # Synthetic scalar regression signal only.
    for model in (control, treatment):
        model.eval()  # No dropout noise; all parameters remain trainable.
        optimizer = torch.optim.AdamW(g.optimizer_groups(model, .02, .02), weight_decay=0)
        observed_relative_gradient = False
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(features, slots, positions) - target).square().mean()
            loss.backward()
            grad = model.relative[-1].weight.grad
            observed_relative_gradient |= bool(grad is not None and grad.abs().max() > 1e-9)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        if model is treatment:
            assert observed_relative_gradient
            assert model.relative[-1].weight.abs().max() > 1e-6
    changed_positions = positions.clone()
    changed_positions[:, :3] = positions[:, [2, 0, 1]]
    assert torch.equal(control(features, slots, positions), control(features, slots, changed_positions))
    difference = (treatment(features, slots, positions) - treatment(features, slots, changed_positions)).abs().max()
    assert difference > 1e-7


def test_optimizer_groups_use_backbone_identity_despite_base_prefix():
    _, control, _ = paired_models()
    groups = g.optimizer_groups(control, 3e-5, 1e-3)
    bb, head = [set(map(id, group['params'])) for group in groups]
    assert bb == set(map(id, control.base.backbone.parameters()))
    assert not bb & head
    assert bb | head == set(map(id, control.parameters()))
    assert len(bb) + len(head) == sum(len(group['params']) for group in groups)
    assert [group['lr'] for group in groups] == [3e-5, 1e-3]
    control.base.backbone.weight.requires_grad_(False)
    with pytest.raises(ValueError, match='full backbone'):
        g.optimizer_groups(control, 3e-5, 1e-3)


@pytest.mark.parametrize('mutation', ['slot_dtype', 'slot_range', 'position_range', 'position_nan', 'position_integer', 'shape'])
def test_malformed_model_metadata_is_rejected(mutation):
    _, _, model = paired_models()
    features, slots, positions = example()
    if mutation == 'slot_dtype': slots = slots.float()
    elif mutation == 'slot_range': slots[0, 0] = 5
    elif mutation == 'position_range': positions[0, 0] = 1.1
    elif mutation == 'position_nan': positions[0, 0] = float('nan')
    elif mutation == 'position_integer': positions = positions.long()
    else: positions = positions[:, :-1]
    with pytest.raises(ValueError):
        model.contextualize(features, slots, positions)


def test_fixed_architecture_arguments_are_enforced():
    with pytest.raises(ValueError): g.make_model(SyntheticB(), 1)
    with pytest.raises(ValueError): g.make_model(SyntheticB(), True, dimension=32)


def test_position_metadata_uses_fixed_slot_geometry_and_preserves_duplicates():
    centers = [1, 16, 19, 30, 33, 42, 45, 50, 53, 62, 1]
    slots, positions = g.position_metadata(centers, np.ones(64, dtype=np.uint8))
    np.testing.assert_array_equal(slots, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0])
    np.testing.assert_array_equal(positions, np.array([1/17, 16/17, 1/13, 12/13, 1/11, 10/11, 1/7, 6/7, 1/11, 10/11, 1/17], np.float32))
    assert slots.dtype == np.int64 and positions.dtype == np.float32
    empty_slots, empty_positions = g.position_metadata([-1] * 5, np.zeros(64))
    np.testing.assert_array_equal(empty_slots, [-1] * 5)
    np.testing.assert_array_equal(empty_positions, np.zeros(5))


@pytest.mark.parametrize('centers,mask_kind', [([0], 'ones'), ([17], 'ones'), ([18], 'ones'), ([-1], 'ones'), ([1], 'zeros'), ([1.5], 'ones'), ([1], 'gap'), ([], 'ones'), ([1], 'nan')])
def test_position_metadata_rejects_boundaries_invalid_triplets_and_bad_masks(centers, mask_kind):
    mask = np.zeros(64) if mask_kind == 'zeros' else np.ones(64)
    if mask_kind == 'gap': mask[0] = 0
    if mask_kind == 'nan': mask[0] = np.nan
    with pytest.raises(ValueError):
        g.position_metadata(centers, mask)


class SyntheticCorpus:
    depth = 64
    spec = {'slots': [('synthetic', str(i), size) for i, size in enumerate(g.SIZES)]}
    def __init__(self, _directory):
        self.volume_calls = 0
        self.pixels = (np.arange(64 * 5 * 5).reshape(64, 5, 5) % 256).astype(np.uint8)
        self.synthetic_mask = np.ones(64, np.uint8)
        self.synthetic_mask[20] = 0
    def volume(self, _uid):
        self.volume_calls += 1
        return self.pixels
    def mask(self, _uid):
        return self.synthetic_mask


@pytest.mark.parametrize('training,k', [(True, 12), (False, 24)])
def test_position_windows_preserve_original_pixels_labels_and_epoch_replay(monkeypatch, training, k):
    import raptor_data
    from raptor_common import select_centers, study_rng
    monkeypatch.setattr(raptor_data, 'Corpus', SyntheticCorpus)
    labels = {'synthetic': np.linspace(0, 1, 12, dtype=np.float32)}
    arguments = ('unused', ['synthetic'], labels, 4, k, training, 'imagenet', 42)
    original = raptor_data.StudyWindows(*arguments, window_policy='within-slot-valid')
    wrapped = g.PositionWindows(*arguments, window_policy='within-slot-valid')
    observed_centers = []
    for epoch in (0, 3):
        original.set_epoch(epoch); wrapped.set_epoch(epoch)
        expected_x, expected_y, expected_uid = original[0]
        x, slots, positions, y, uid, centers = wrapped[0]
        assert torch.equal(x, expected_x) and torch.equal(y, expected_y) and uid == expected_uid
        assert len(wrapped) == len(original) == 1
        assert int(wrapped.base.epoch.item()) == epoch
        rng = study_rng(42, epoch, uid) if training else None
        expected_centers = select_centers(wrapped.base.corpus.mask(uid), 64, k, SyntheticCorpus.spec, 'within-slot-valid', rng)
        assert centers.tolist() == expected_centers
        expected_slots, expected_positions = g.position_metadata(expected_centers, wrapped.base.corpus.mask(uid))
        np.testing.assert_array_equal(slots.numpy(), expected_slots)
        np.testing.assert_array_equal(positions.numpy(), expected_positions)
        observed_centers.append(centers.clone())
    assert wrapped.base.corpus.volume_calls == 2  # Exactly one pixel fetch per item.
    assert torch.equal(observed_centers[0], observed_centers[1]) is (not training)
    batch = g.collate([wrapped[0], wrapped[0]])
    assert batch[0].shape == (2, k, 3, 4, 4) and batch[4] == ['synthetic', 'synthetic']
    assert torch.equal(batch[1][0], batch[1][1]) and torch.equal(batch[5][0], batch[5][1])


def test_position_windows_reject_policy_change_and_epoch_race(monkeypatch):
    import raptor_data
    monkeypatch.setattr(raptor_data, 'Corpus', SyntheticCorpus)
    args = ('unused', ['synthetic'], {'synthetic': np.zeros(12, np.float32)}, 4, 12, True, 'imagenet', 42)
    with pytest.raises(ValueError, match='within-slot-valid'):
        g.PositionWindows(*args, window_policy='legacy')
    wrapped = g.PositionWindows(*args, window_policy='within-slot-valid')
    original_get = raptor_data.StudyWindows.__getitem__
    def changing_epoch(dataset, index):
        result = original_get(dataset, index)
        dataset.set_epoch(int(dataset.epoch.item()) + 1)
        return result
    monkeypatch.setattr(raptor_data.StudyWindows, '__getitem__', changing_epoch)
    with pytest.raises(ValueError, match='epoch changed'):
        wrapped[0]


def test_helper_import_is_passive(tmp_path):
    script = "import importlib.util,sys; s=importlib.util.spec_from_file_location('candidate',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert not any(k in sys.modules for k in ['torch','numpy','raptor_common','raptor_data']); assert list(__import__('pathlib').Path('.').iterdir()) == []"
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-c', script, str(ROOT / 'tools/exp085_relative_position.py')], cwd=tmp_path,
                            capture_output=True, text=True, env=env, timeout=20)
    assert result.returncode == 0, result.stderr


def protocol_fixture(tmp_path):
    ordered = {'train': ['synthetic-t'], 'val': ['synthetic-v']}
    (tmp_path / 'ids.json').write_text(json.dumps(ordered), encoding='utf-8')
    p = {'schema': g.SCHEMA, 'arm_order': list(g.ARMS), 'execution': {'total_time_limit_seconds': 7200},
         'source_bindings': {}, 'training_code_sha256': {},
         'ordered_ids': {'path': 'ids.json', **{key + '_sha256': g.ids_hash(value) for key, value in ordered.items()}}}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(p), encoding='utf-8')
    return p, path, ordered


def test_protocol_rejects_identity_source_and_ordered_id_drift(tmp_path):
    p, path, ordered = protocol_fixture(tmp_path)
    assert g.validate_protocol(path, root=tmp_path) == (p, ordered)
    for mutate in (lambda x: x.update(schema='wrong'), lambda x: x.update(arm_order=['treatment','control']),
                   lambda x: x['execution'].update(total_time_limit_seconds=7201),
                   lambda x: x['ordered_ids'].update(train_sha256='wrong'),
                   lambda x: x.update(source_bindings={'ids.json': 'wrong'})):
        bad = deepcopy(p); mutate(bad)
        path.write_text(json.dumps(bad), encoding='utf-8')
        with pytest.raises(ValueError):
            g.validate_protocol(path, root=tmp_path)


@pytest.mark.parametrize('mutation', ['status','phase','parity','arm','per_fit','total','negative','nan',
    'elapsed_negative','elapsed_nan','elapsed_zero','elapsed_bool','elapsed_over_cap','projected_bool'])
def test_pilot_cost_gate_rejects_invalid_completion_or_budget(mutation):
    p = {'execution': {'per_fit_seconds': 3600, 'total_time_limit_seconds': 7200, 'pilot_seconds': 600}}
    pilot = {'status':'PASS', 'phase':'pilot', 'paired_initialization_exact':True,
             'conservative_full_seconds_by_arm': {'control':3000.,'treatment':3000.}, 'supervised_elapsed_seconds':500.}
    g.pilot_cost_gate(pilot, p)
    if mutation in ('status','phase'): pilot[mutation] = 'wrong'
    elif mutation == 'parity': pilot['paired_initialization_exact'] = False
    elif mutation == 'arm': pilot['conservative_full_seconds_by_arm']['extra'] = 1
    elif mutation == 'per_fit': pilot['conservative_full_seconds_by_arm']['control'] = 3601
    elif mutation == 'total': pilot['conservative_full_seconds_by_arm'] = {'control': 3500., 'treatment': 3500.}
    elif mutation == 'negative': pilot['conservative_full_seconds_by_arm']['control'] = -1
    elif mutation == 'elapsed_negative': pilot['supervised_elapsed_seconds'] = -1
    elif mutation == 'elapsed_nan': pilot['supervised_elapsed_seconds'] = float('nan')
    elif mutation == 'elapsed_zero': pilot['supervised_elapsed_seconds'] = 0
    elif mutation == 'elapsed_bool': pilot['supervised_elapsed_seconds'] = True
    elif mutation == 'elapsed_over_cap': pilot['supervised_elapsed_seconds'] = 601
    elif mutation == 'projected_bool': pilot['conservative_full_seconds_by_arm']['control'] = True
    else: pilot['conservative_full_seconds_by_arm']['control'] = float('nan')
    with pytest.raises(ValueError):
        g.pilot_cost_gate(pilot, p)


@pytest.mark.parametrize('empty', [False, True])
def test_cpu_autocast_reductions_actually_run_fp32(monkeypatch, empty):
    _, _, model = paired_models()
    activate_residual(model)
    features, slots, positions = example()
    features.requires_grad_(True)
    if empty:
        slots.fill_(-1); positions.zero_()
    matmul = torch.matmul
    seen = []
    def observed_matmul(left, right, *args, **kwargs):
        result = matmul(left, right, *args, **kwargs)
        seen.append({'autocast': torch.is_autocast_enabled('cpu'), 'left': left.dtype,
                     'right': right.dtype, 'result': result.dtype})
        return result
    monkeypatch.setattr(torch, 'matmul', observed_matmul)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert torch.is_autocast_enabled('cpu')
        result = model.contextualize(features, slots, positions)
        assert torch.is_autocast_enabled('cpu')  # Island restored caller policy.
    assert len(seen) == 2
    assert all(item == {'autocast': False, 'left': torch.float32, 'right': torch.float32,
                        'result': torch.float32} for item in seen)
    dtypes = model.last_attention_dtypes
    assert all(dtypes[key] == 'torch.float32' for key in ('logits', 'attention', 'context', 'residual'))
    assert all(dtypes[key] == 'torch.bfloat16' for key in ('q', 'k', 'v', 'projection'))
    assert result.dtype == torch.float32 and torch.isfinite(result).all()
    result.square().sum().backward()
    assert torch.isfinite(features.grad).all()


class SyntheticBufferedB(SyntheticB):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.Dropout(.1))
    def encode(self, x):
        return self.backbone(x.flatten(0,1)).view(*x.shape)


@pytest.mark.parametrize('training', [False, True])
def test_actual_firstbatch_gate_checks_train_eval_rng_buffers_and_restores(training):
    model = g.make_model(SyntheticBufferedB(), True).train(training)
    x, slots, positions = example()
    rng = torch.get_rng_state().clone()
    buffers = {key: value.clone() for key, value in model.named_buffers()}
    with torch.autocast('cpu', dtype=torch.bfloat16):
        receipt = g.verify_initial_forward(model, x, slots, positions)
    assert receipt == {'eval': {'logits_exact': True, 'rng_consumption_exact': True, 'buffer_updates_exact': True},
                       'train': {'logits_exact': True, 'rng_consumption_exact': True, 'buffer_updates_exact': True},
                       'probe_rng_and_buffers_restored': True}
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value,buffers[key]) for key,value in model.named_buffers())
    assert all(module.training is training for module in model.modules())


def test_firstbatch_gate_detects_extra_rng_even_when_logits_equal(monkeypatch):
    model = g.make_model(SyntheticBufferedB(), False).train()
    x, slots, positions = example()
    original = model.contextualize
    def stochastic_probe(*args):
        torch.rand(1)
        return original(*args)
    monkeypatch.setattr(model, 'contextualize', stochastic_probe)
    rng = torch.get_rng_state().clone()
    buffers = {key: value.clone() for key, value in model.named_buffers()}
    with pytest.raises(ValueError, match='RNG'):
        g.verify_initial_forward(model, x, slots, positions)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value,buffers[key]) for key,value in model.named_buffers())
    assert model.training
