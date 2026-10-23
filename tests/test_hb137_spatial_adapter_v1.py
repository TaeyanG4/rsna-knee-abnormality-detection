"""H137 synthetic model checks and authenticated recipe checks; no label output."""
from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import torch
from torch import nn

from training.arms import mil
from training.arms import hb137_spatial_adapter_v1 as m
from tools import mil_screen_arm_build_v1 as build

torch.set_num_threads(2)


class TinyBackbone(nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.head = nn.Module()
        self.head.norm = nn.Identity()

    def forward_features(self, x):
        return self.conv(nn.functional.adaptive_avg_pool2d(x, (2, 2)))


def fixture():
    torch.manual_seed(42)
    parent = mil.FindingMIL(TinyBackbone()).eval()
    x = torch.randn(2, 3, 3, 8, 8)
    return parent, x


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize("mode", [False, True])
def test_exact_initial_parent_and_rng_parity(adapter, mode):
    parent, x = fixture()
    rng = torch.get_rng_state().clone()
    model = m.FrozenWeakMIL(deepcopy(parent), adapter).train(mode)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not model.parent.training
    assert torch.equal(parent(x), model(x))
    assert torch.equal(model(x), model(x))
    assert all(not p.requires_grad for p in model.parent.parameters())


@pytest.mark.parametrize("adapter", [False, True])
def test_finite_learning_frozen_rows_and_strict_reload(adapter):
    parent, x = fixture()
    before = deepcopy(parent.state_dict())
    baseline = parent(x).detach()
    model = m.FrozenWeakMIL(deepcopy(parent), adapter).train()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    expected = {"weak_head.att.3.weight", "weak_head.att.3.bias", "weak_head.weak_clsW", "weak_head.weak_clsb"}
    if adapter:
        expected |= {"adapter.down.weight", "adapter.spatial.weight", "adapter.out.weight"}
    assert trainable == expected
    opt, sched, scaler = mil.optimizer_stack(model, 20, "cpu", "none")
    assert len(opt.param_groups[0]["params"]) == 0  # frozen backbone; trainable head group retains deployed LR
    y = (torch.arange(24).reshape(2, 12) % 3 == 0).float()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        for n, p in model.named_parameters():
            if n in trainable:
                assert p.grad is not None and torch.isfinite(p.grad).all()
            else:
                assert p.grad is None
        if adapter:
            assert model.adapter.out.weight.grad.abs().sum() > 0
            if step == 0:
                assert model.adapter.down.weight.grad.count_nonzero() == 0
            else:
                assert model.adapter.down.weight.grad.abs().sum() > 0
                assert model.adapter.spatial.weight.grad.abs().sum() > 0
        opt.step()
        sched.step()
    assert all(torch.equal(v, model.parent.state_dict()[n]) for n, v in before.items())
    after = model(x).detach()
    assert torch.equal(after[:, m.PROTECTED], baseline[:, m.PROTECTED])
    assert not torch.equal(after[:, m.WEAK], baseline[:, m.WEAK])
    # Actual serialization and strict load, independent of full real checkpoints.
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = m.FrozenWeakMIL(deepcopy(parent), adapter).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert torch.equal(restored(x), after)
    flipped = x.flip(-1)
    for view in (x, flipped):
        assert torch.equal(restored(view)[:, m.PROTECTED], parent(view)[:, m.PROTECTED])


def test_adapter_has_spatial_effect_without_cross_window_mixing():
    torch.manual_seed(19)
    adapter = m.SpatialAdapter(4)
    tokens = torch.randn(2, 8, 4)
    assert torch.equal(adapter(tokens, 2), tokens)
    nn.init.normal_(adapter.out.weight)
    base = adapter(tokens, 2)
    changed = tokens.clone()
    changed[:, :4] += 20
    out = adapter(changed, 2)
    assert torch.equal(base[:, 4:], out[:, 4:])
    assert not torch.equal(base[:, :4], out[:, :4])


@pytest.mark.parametrize("windows", [0, -1, 3, 2.0])
def test_invalid_grid_rejected(windows):
    with pytest.raises(m.hooks.life.ContractError):
        m.SpatialAdapter(4)(torch.zeros(1, 8, 4), windows)


@pytest.mark.parametrize("adapter", [False, True])
def test_all_empty_and_partially_padded_studies_are_finite(adapter):
    parent, x = fixture()
    background = (-torch.tensor((.485, .456, .406)) / torch.tensor((.229, .224, .225))).view(3, 1, 1)
    x[0] = background
    x[1, 1] = background
    model = m.FrozenWeakMIL(deepcopy(parent), adapter)
    assert torch.isfinite(model(x)).all()
    assert torch.equal(model(x), parent(x))


@pytest.mark.parametrize("seed", [42, 43])
def test_registered_heads_and_adapter_have_identical_shared_recipe(seed):
    control = m.hooks.control_spec(seed)
    docs = []
    for kind in ("head", "adapter"):
        _, record, arm = build.state(build.ROOT / m.WORK / kind)
        spec = build.spec_for(record, seed)
        c = deepcopy(arm.validate_recipe(spec))
        assert c[arm.section].pop(arm.key) == arm.contract(seed)
        assert c == control.document["config"]
        for key in ("seed", "fold", "target_environment", "outputs"):
            assert spec.document[key] == control.document[key]
        assert spec.document["inputs"]["data"] == control.document["inputs"]["data"]
        assert spec.document["inputs"]["rosters"] == control.document["inputs"]["rosters"]
        build.life.validate_inputs(spec, build.ROOT)
        docs.append(c)
    assert docs[0] == docs[1]


@pytest.mark.parametrize("change", ["contract", "schedule", "source"])
def test_recipe_drift_fails_closed(change):
    _, record, arm = build.state(build.ROOT / m.WORK / "adapter")
    doc = build.spec_for(record, 42).document
    if change == "contract":
        doc["config"]["model"][arm.key]["adapter_rank"] = 32
    elif change == "schedule":
        doc["config"]["updates"]["extra"] = True
    else:
        doc["inputs"]["sources"].pop("training/arms/mil_screen_hooks_v1.py")
    with pytest.raises(build.life.ContractError):
        arm.validate_recipe(build.life.ArmSpec.from_document(doc))


@pytest.mark.parametrize("change", ["epoch", "device", "parent_spec", "roster"])
def test_synthetic_parent_ancestry_rejection(change):
    spec = m.hooks.control_spec(42)
    c = spec.document["config"]
    snap = {"spec_sha256": spec.sha256}
    ck = dict(epoch=19, fold=0, lab=m.hooks.TARGETS, arch=c["model"]["arch"], model_kind="finding_spatial_tokens_v1",
              provenance=dict(spec_sha256=spec.sha256, checkpoint_selection="fixed-final",
                  identity=dict(mode="full", device="cuda"), train_ids_ordered_sha256=c["roots"]["train_sha256"],
                  eval_ids_ordered_sha256=c["roots"]["val_sha256"]))
    m.check_parent_checkpoint(ck, spec, snap)
    if change == "epoch":
        ck["epoch"] = 0
    elif change == "device":
        ck["provenance"]["identity"]["device"] = "cpu"
    elif change == "parent_spec":
        ck["provenance"]["spec_sha256"] = "0" * 64
    else:
        ck["provenance"]["eval_ids_ordered_sha256"] = "0" * 64
    with pytest.raises(build.life.ContractError, match="ancestry"):
        m.check_parent_checkpoint(ck, spec, snap)
