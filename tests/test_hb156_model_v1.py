"""CPU tests for the one registered H-B-156 constructor change; no downloads."""
from __future__ import annotations

import gc
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "src"))
import exp143_own0949 as original
import hb156_model_v1 as candidate


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("H-B-156 model tests must not initialize CUDA")

    for name in ("_lazy_init", "init", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, fail)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old_threads)
    gc.collect()


def test_exact_timm_1019_registered_name():
    import timm
    assert timm.__version__ == "1.0.19"
    assert candidate.ARCH in timm.list_models(pretrained=True)
    assert timm.get_pretrained_cfg(candidate.ARCH).input_size == (3, 384, 384)


def test_default_and_explicit_coatnet_delegate(monkeypatch):
    sentinel, factory = object(), object()
    calls = []

    def delegated(res, backbone_factory=None):
        calls.append((res, backbone_factory))
        return sentinel

    monkeypatch.setattr(original, "build_model", delegated)
    assert candidate.build_model(320, factory) is sentinel
    assert candidate.build_model(384, backbone=candidate.COATNET_ARCH) is sentinel
    assert calls == [(320, factory), (384, None)]


def test_real_coatnet_state_byte_identical():
    torch.random.default_generator.manual_seed(42)
    baseline = original.build_model().eval()
    torch.random.default_generator.manual_seed(42)
    adapted = candidate.build_model().eval()
    a, b = baseline.state_dict(), adapted.state_dict()
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
        assert a[key].numpy().tobytes() == b[key].numpy().tobytes(), key


def test_convnext_real_forward_shape_and_strict_reload():
    torch.random.default_generator.manual_seed(42)
    model = candidate.build_model(backbone=candidate.ARCH).eval()
    assert model.backbone.num_features == 768
    assert model.norm.normalized_shape == (768,)
    assert model.att[0].in_features == 768 and model.att[0].out_features == 256
    assert model.att[2].p == .2
    assert tuple(model.clsW.shape) == (12, 768)
    with torch.inference_mode():
        x = torch.zeros(2, 1, 3, 384, 384)
        expected = model(x)
        assert expected.shape == (2, 12) and torch.isfinite(expected).all()
        reloaded = candidate.build_model(backbone=candidate.ARCH).eval()
        loaded = reloaded.load_state_dict(model.state_dict(), strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        assert torch.equal(expected, reloaded(x))
        assert reloaded(torch.zeros(1, 1, 3, 320, 320)).shape == (1, 12)


def test_convnext_offline_factory_has_no_coatnet_img_size(monkeypatch):
    import timm
    from torch import nn

    class Tiny(nn.Module):
        num_features = 5

        def forward(self, x):
            return x.new_zeros((len(x), self.num_features))

    calls = []

    def create(name, **kwargs):
        calls.append((name, kwargs))
        return Tiny()

    monkeypatch.setattr(timm, "create_model", create)
    model = candidate.build_model(320, backbone=candidate.ARCH).eval()
    assert calls == [(candidate.ARCH, dict(pretrained=False, num_classes=0,
                                         in_chans=3, global_pool="avg"))]
    assert model.norm.normalized_shape == (5,)
    with torch.inference_mode():
        assert model(torch.zeros(2, 12, 3, 8, 8)).shape == (2, 12)


def test_unregistered_backbone_rejected():
    with pytest.raises(ValueError, match="unregistered"):
        candidate.build_model(backbone="convnext_base")


def test_no_bn_replay_two_studies_restores_state_and_modes():
    from torch import nn
    from tools.exp112_full_swa import uid_hash

    class ReplayModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 12)
            self.register_buffer("counter", torch.tensor(0))
            self.calls = []

        def forward(self, x):
            self.calls.append((self.training, torch.is_grad_enabled(), tuple(x.shape)))
            self.counter.add_(1)
            return self.linear(x)

    model = ReplayModel().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = [(torch.zeros(1, 3), None, [uid]) for uid in ("synthetic-a", "synthetic-b")]
    receipt = candidate.recompute_bn(model, iter(loader), "cpu")
    assert model.calls == [(True, False, (1, 3))] * 2
    assert not model.training and not model.linear.training
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in model.parameters())
    assert receipt["batches"] == receipt["studies"] == receipt["unique_studies"] == 2
    assert receipt["bn_modules"] == 0 and receipt["num_batches_tracked"] == []
    assert receipt["studies_sha256"] == uid_hash(["synthetic-a", "synthetic-b"])
    assert "synthetic-a" not in repr(receipt)


def test_no_bn_empty_or_failed_replay_restores_mode():
    from torch import nn
    model = nn.Linear(3, 12).eval()
    with pytest.raises(RuntimeError, match="zero batches"):
        candidate.recompute_bn(model, [], "cpu")
    assert not model.training
    with pytest.raises(RuntimeError):
        candidate.recompute_bn(model, [(torch.zeros(1, 7), None, ["synthetic"])], "cpu")
    assert not model.training


def test_bn_replay_delegates_exactly(monkeypatch):
    from torch import nn
    from tools import exp112_full_swa as original_swa
    calls, sentinel = [], object()

    def delegate(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(original_swa, "recompute_bn", delegate)
    model, loader, autocast, update = nn.BatchNorm1d(3), object(), object(), object()
    assert candidate.recompute_bn(model, loader, "cpu", autocast, update) is sentinel
    assert calls == [(model, loader, "cpu", autocast, update)]


def test_bn_replay_results_equal_original():
    import copy
    from torch import nn
    from tools.exp112_full_swa import recompute_bn as original_recompute_bn
    model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3)).eval()
    other = copy.deepcopy(model)
    loader = [(torch.arange(6, dtype=torch.float32).reshape(2, 3), None, ["a", "b"])]
    expected = original_recompute_bn(model, loader, "cpu")
    actual = candidate.recompute_bn(other, loader, "cpu")
    expected.pop("seconds")
    actual.pop("seconds")
    assert expected == actual
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())
