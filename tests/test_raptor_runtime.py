"""CPU contract tests; import only the passive runtime/common helpers, never CLIs."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
import raptor_runtime as rt


def fake_torch(major=8, emulate_only=False, initialized=False):
    state = {"deterministic": False, "precision": "high"}
    matmul = SimpleNamespace(allow_tf32=True, allow_fp16_reduced_precision_reduction=True,
                             allow_bf16_reduced_precision_reduction=True)
    cuda = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: initialized,
                           device=lambda device: nullcontext(), get_device_capability=lambda device: (major, 0),
                           is_bf16_supported=lambda including_emulation=True: including_emulation or not emulate_only)
    return SimpleNamespace(
        cuda=cuda, version=SimpleNamespace(cuda="12.1"),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul),
                                 cudnn=SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True)),
        use_deterministic_algorithms=lambda value: state.update(deterministic=value),
        are_deterministic_algorithms_enabled=lambda: state["deterministic"],
        set_float32_matmul_precision=lambda value: state.update(precision=value),
        get_float32_matmul_precision=lambda: state["precision"],
    )


def test_strict_runtime_records_policy_and_disables_tf32(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    report = rt.configure_runtime("fp16", "cuda", torch_module=fake_torch())
    assert report["requested_amp"] == report["effective_amp"] == "fp16"
    assert report["backends"]["cudnn_benchmark"] is False
    assert report["backends"]["cudnn_deterministic"] is True
    assert report["backends"]["deterministic_algorithms"] is True
    assert report["backends"]["matmul_allow_tf32"] is False
    assert report["backends"]["cudnn_allow_tf32"] is False
    assert report["backends"]["cublas_workspace_config"] == ":4096:8"
    assert report["precision_fallback_allowed"] is False


def test_native_bf16_rejects_t4_emulation_and_old_torch(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    t4 = fake_torch(major=7, emulate_only=True)
    with pytest.raises(ValueError, match="not natively supported"):
        rt.configure_runtime("bf16", "cuda", torch_module=t4)
    t4.cuda.is_bf16_supported = lambda: True  # older signature includes emulation
    assert not rt.native_bf16_supported(t4)
    ada = fake_torch(major=8)
    ada.cuda.is_bf16_supported = lambda: True
    assert rt.native_bf16_supported(ada)


def test_cpu_amp_is_not_silently_changed_and_late_cuda_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="explicit --amp none"):
        rt.configure_runtime("fp16", "cpu", torch_module=fake_torch())
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="before initializing CUDA"):
        rt.configure_runtime("fp16", "cuda", torch_module=fake_torch(initialized=True))


def test_checkpoint_norm_is_respected_with_explicit_legacy_assumption():
    x = np.full((2, 3, 2, 2), 0.25, np.float32)
    norm, source = rt.checkpoint_norm({"norm": "none"})
    assert norm == "none" and source == "checkpoint"
    np.testing.assert_array_equal(rt.normalize_input(x, norm), x)
    assert rt.checkpoint_norm({}) == ("imagenet", "legacy_imagenet_assumption")
    assert not np.array_equal(rt.normalize_input(x, "imagenet"), x)
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        rt.checkpoint_norm({"norm": "unexpected"})


def test_tensor_hash_checks_values_shape_and_is_memory_layout_independent():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    assert rt.tensor_sha256(x) == rt.tensor_sha256(np.asfortranarray(x))
    assert rt.tensor_sha256(x) != rt.tensor_sha256(x.reshape(3, 2, 2, 2))
    changed = x.copy(); changed.flat[0] = 1
    assert rt.tensor_sha256(x) != rt.tensor_sha256(changed)
    with pytest.raises(ValueError, match="nonfinite"):
        rt.normalize_input(np.full((2, 3, 2, 2), np.nan), "none")


class TinyCorpus:
    depth = 6
    spec = {"slots": [["Sagittal", 1, 6]]}

    def volume(self, uid):
        return np.full((6, 2, 2), 30 + int(uid) * 10, np.uint8)

    def mask(self, uid):
        return np.ones(6, np.uint8)


def test_prediction_batches_keep_input_receipts_and_eval_mode():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def forward(self, x):
            assert not self.training and not torch.is_grad_enabled()
            return x.mean(dim=(1, 2, 3, 4))[:, None].repeat(1, 12)

    args = (Model(), TinyCorpus(), ["0", "1", "2"], 3, 2, "none", "cpu")
    p1, r1 = rt.predict(*args, norm="none", batch=1, log=lambda message: None)
    p2, r2 = rt.predict(*args, norm="none", batch=2, log=lambda message: None)
    np.testing.assert_array_equal(p1, p2)
    assert r1["ordered_inputs_sha256"] == r2["ordered_inputs_sha256"]
    assert r1["input_tensor_hashes"] == r2["input_tensor_hashes"]
    assert r2["fp32_fallbacks"] == 0 and r2["all_finite"]
    assert r2["batch_size"] == 2
    _, reverse = rt.predict(Model(), TinyCorpus(), ["2", "1", "0"], 3, 2, "none", "cpu", norm="none", log=lambda message: None)
    assert reverse["ordered_inputs_sha256"] != r1["ordered_inputs_sha256"]


@pytest.mark.parametrize("failure", ["runtime", "nan", "inf"])
def test_failures_stop_without_precision_retry(failure):
    torch = pytest.importorskip("torch")

    class BadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if failure == "runtime":
                raise RuntimeError("simulated kernel failure")
            return torch.full((len(x), 12), float(failure))

    model = BadModel()
    with pytest.raises(RuntimeError, match="no precision fallback"):
        rt.predict(model, TinyCorpus(), ["0"], 3, 2, "none", "cpu", norm="none")
    assert model.calls == 1


def test_prepare_input_routes_window_policy(monkeypatch):
    import raptor_common
    calls = []

    def centers(mask, depth, k, spec, **kwargs):
        calls.append({"spec": spec, **kwargs})
        return [1] * k

    monkeypatch.setattr(raptor_common, "select_centers", centers)
    pytest.importorskip("torch")
    rt.prepare_input(TinyCorpus(), "0", 3, 2, "none", window_policy="within-slot-valid")
    assert calls == [{"spec": TinyCorpus.spec, "policy": "within-slot-valid"}]
