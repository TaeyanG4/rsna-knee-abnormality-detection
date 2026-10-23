"""EXP155 CPU-only regression tests; no real corpus, CUDA, or network."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/src"))
import exp155_own0949_oai as fit


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*_args, **_kwargs):
        raise AssertionError("CPU test touched CUDA")
    for name in ("_lazy_init", "init", "manual_seed_all", "is_available", "set_device",
                 "is_bf16_supported", "synchronize", "mem_get_info"):
        monkeypatch.setattr(torch.cuda, name, fail)
    yield
    assert not torch.cuda.is_initialized()


def test_full_rsna_mask_matches_original_bce_and_gradient():
    logits = torch.linspace(-1, 1, 24).reshape(2, 12).requires_grad_()
    labels = torch.linspace(.05, .95, 24).reshape(2, 12)
    weights = torch.linspace(1, 3, 12)
    actual = fit.masked_study_bce(logits, labels, torch.ones_like(labels), weights)
    expected = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weights)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0],
                               torch.autograd.grad(expected, logits)[0])


def test_oai_only_observed_three_and_unknown_zero_gradient():
    logits = torch.zeros(2, 12, requires_grad=True)
    labels = torch.zeros(2, 12)
    labels[0, [5, 6, 8]] = torch.tensor([1., 0., 1.])
    labels[1, 6] = 1.
    # Unknown labels can contain arbitrary placeholders; the mask prevents loss.
    labels[0, 0] = 999.
    mask = torch.zeros_like(labels)
    mask[0, [5, 6, 8]] = 1
    mask[1, 6] = 1
    weights = torch.ones(12)
    loss = fit.masked_study_bce(logits, labels, mask, weights)
    expected = (F.binary_cross_entropy_with_logits(logits[0, [5, 6, 8]], labels[0, [5, 6, 8]])
                + F.binary_cross_entropy_with_logits(logits[1, 6], labels[1, 6])) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (logits.grad[mask == 0] == 0).all()
    assert (logits.grad[mask == 1] != 0).all()


def test_reject_empty_or_nonbinary_masks():
    logits = torch.zeros(1, 12)
    labels = torch.zeros(1, 12)
    for mask in (torch.zeros_like(labels), torch.full_like(labels, .5)):
        with pytest.raises(ValueError):
            fit.masked_study_bce(logits, labels, mask, torch.ones(12))


def test_disjoint_corpus_routing_and_batch(monkeypatch):
    calls = []
    class FakeDataset:
        def __init__(self, corpus, ids, labels, seed, augment=True):
            self.corpus, self.ids = corpus, list(ids)
            self.epoch = 0
        def set_epoch(self, epoch):
            self.epoch = epoch
        def __getitem__(self, index):
            uid = self.ids[index]
            calls.append((self.corpus, uid, self.epoch))
            return torch.zeros(12, 3, 4, 4), torch.zeros(12), uid
    monkeypatch.setattr(fit, "StudyDataset", FakeDataset)
    ids = ["rsna1", "OAI_12_R_V00", "rsna2"]
    labels = {uid: [0.] * 12 for uid in ids}
    masks = {uid: [1.] * 12 if uid.startswith("rsna") else [0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0.]
             for uid in ids}
    ds = fit.MixedStudyDataset("rsna-corpus", "oai-corpus", ids, labels, masks,
                               ["rsna1", "rsna2"], ["OAI_12_R_V00"], 42)
    ds.set_epoch(4)
    x, y, mask, uids = fit.unpack_batch(fit.collate([ds[0], ds[1], ds[2]]))
    assert x.shape == (3, 12, 3, 4, 4) and y.shape == mask.shape == (3, 12)
    assert uids == ids
    assert calls == [("rsna-corpus", "rsna1", 4), ("oai-corpus", "OAI_12_R_V00", 4),
                     ("rsna-corpus", "rsna2", 4)]
    assert mask[1].sum() == 3 and mask[0].sum() == 12
    with pytest.raises(ValueError, match="disjoint"):
        fit.MixedStudyDataset("r", "o", ids, labels, masks, ids, ids[:1], 42)


def test_registered_v2_helpers_and_off_only_probe_contract(tmp_path, monkeypatch):
    assert fit.make_optimization is fit.base.make_optimization
    assert fit.average_registered_epochs is fit.base.average_registered_epochs
    assert fit.build_model is fit.base.build_model
    assert fit.accumulation_group(20, 35, 16) == (False, 16)
    assert fit.accumulation_group(34, 35, 16) == (True, 3)
    monkeypatch.setattr(fit, "ROOT", tmp_path)
    probe = tmp_path / ".local/exp155-probes/first"
    assert fit.scratch_directory(probe) == probe
    with pytest.raises(ValueError):
        fit.scratch_directory(tmp_path / "artifacts/experiments/run")
    source = inspect.getsource(fit.probe)
    assert 'for mode in ("off",):' in source
    assert '"--oai-corpus"' in source
    assert 'grad_checkpointing=False' in source
    assert 'queued fit must not start' in source
    assert 'oai=True' in inspect.getsource(fit.train)


def test_tiny_validation_is_cpu_only_and_checks_mask():
    receipt = fit.tiny_validation()
    assert receipt["masked_loss_verified"] is True
    assert receipt["oai_unknown_gradient_zero"] is True
    assert receipt["rsna_bce_equivalent"] is True
