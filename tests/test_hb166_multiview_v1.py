"""CPU tests v1.0.0 for the H-B-166 multi-view arm (real caches, no label rows, no CUDA)."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # empty, never -1 (Windows cu121 backward crash)

import pytest
import torch

from tools import hb155_build_v1 as b155
from training.arms import hb155_crops_v1 as crops
from training.arms import hb166_multiview_v1 as mv
from training.arms import mil_screen_hooks_v3 as h

ROOT = b155.ROOT
FULL_DIR, CROP_DIR = ROOT / b155.CONTROL, ROOT / b155.CROP


@pytest.fixture(scope="module")
def ids():
    c = b155.read(b155.config_path("control", 42))
    return c["train"][:2] + c["val"][:1]


def labels(ids):
    return {u: h.np.full(12, .5, h.np.float32) for u in ids}


def tiny():
    torch.manual_seed(0)
    return h.FindingMIL(h.common.build_backbone("convnext_atto", pretrained=False))


@pytest.mark.parametrize("epoch", [0, 5])
def test_full_field_part_is_byte_identical_to_control_training_windows(ids, epoch):
    ds = mv.make_dataset(FULL_DIR, CROP_DIR, ids, labels(ids), 12, True, 42, 224)
    control = crops.Windows(h.MILWindows(FULL_DIR, ids, labels(ids), 12, True, 42), 224)
    ds.set_epoch(epoch)
    control.set_epoch(epoch)
    for i in range(len(ids)):
        x, y, uid, views = ds[i]
        xc, yc, uc = control[i]
        assert x.shape == (24, 3, 224, 224) and uid == uc and torch.equal(y, yc)
        assert torch.equal(x[:12], xc)
        assert views.tolist() == [mv.FULL] * 12 + [mv.CROP] * 12


def test_crop_training_streams_are_independent_but_eval_equals_crop_arm(ids):
    train = mv.make_dataset(FULL_DIR, CROP_DIR, ids, labels(ids), 12, True, 42, 224)
    arm = crops.Windows(h.MILWindows(CROP_DIR, ids, labels(ids), 12, True, 42), 224)
    train.set_epoch(1)
    arm.set_epoch(1)
    assert any(not torch.equal(train[i][0][12:], arm[i][0]) for i in range(len(ids)))
    for k in (24, 36):
        ev = mv.make_dataset(FULL_DIR, CROP_DIR, ids, labels(ids), k, False, 42, 224)
        crop_eval = crops.Windows(h.MILWindows(CROP_DIR, ids, labels(ids), k, False, 42), 224)
        full_eval = crops.Windows(h.MILWindows(FULL_DIR, ids, labels(ids), k, False, 42), 224)
        x, _, _, views = ev[0]
        assert x.shape == (2 * k, 3, 224, 224) and int((views == mv.CROP).sum()) == k
        assert torch.equal(x[k:], crop_eval[0][0]) and torch.equal(x[:k], full_eval[0][0])


def test_view_datasets_must_match():
    a = crops.Windows(h.MILWindows(FULL_DIR, ["a"], {}, 12, True, 42), 224)
    c = mv.CropWindows(h.MILWindows(CROP_DIR, ["a"], {}, 24, True, 42), 224)
    with pytest.raises(ValueError):
        mv.MultiViewWindows(a, c)
    with pytest.raises(ValueError):
        mv.MultiViewWindows(c, a)


def test_zero_embedding_equals_control_and_masks_hold():
    control = tiny().eval()
    model = mv.attach_views(tiny()).eval()
    assert torch.equal(model.view_embed, torch.zeros(2, 320))
    x = torch.randn(2, 4, 3, 64, 64)
    with torch.no_grad():
        ref = control(x)
        same = model(x, torch.zeros(2, 4, dtype=torch.int64))
        mixed = model(x, torch.tensor([[0, 0, 1, 1]] * 2))
    assert torch.allclose(ref, same, atol=1e-6) and torch.allclose(ref, mixed, atol=1e-6)
    # Background (missing-plane) crop windows stay masked even with a non-zero embedding.
    with torch.no_grad():
        model.view_embed.normal_()
        background = (torch.zeros(3, 1, 1) - torch.tensor([.485, .456, .406]).view(3, 1, 1)) / torch.tensor([.229, .224, .225]).view(3, 1, 1)
        padded = torch.cat([x[:, :2], background.expand(2, 2, 3, 64, 64)], dim=1)
        a = model(padded, torch.tensor([[0, 0, 1, 1]] * 2))
        b = model(x[:, :2], torch.tensor([[0, 0]] * 2))
    assert torch.allclose(a, b, atol=1e-5)


def test_embedding_trains_state_keys_and_head_group():
    control_keys = set(tiny().state_dict())
    model = mv.attach_views(tiny())
    assert set(model.state_dict()) == control_keys | {"view_embed"}
    opt, _, _ = h.optimizer_stack(model, 4, "cpu", "none")
    assert id(model.view_embed) in {id(p) for p in opt.param_groups[1]["params"]}
    assert id(model.view_embed) not in {id(p) for p in opt.param_groups[0]["params"]}
    out = model(torch.randn(1, 4, 3, 64, 64), torch.tensor([[0, 0, 1, 1]]))
    out.sum().backward()
    assert model.view_embed.grad is not None and float(model.view_embed.grad.abs().sum()) > 0
    with pytest.raises(TypeError):
        model(torch.randn(1, 2, 3, 64, 64))
    with pytest.raises(ValueError):
        mv.attach_views(model)


def test_inference_chunking_has_no_cross_study_effect():
    model = mv.attach_views(tiny()).eval()
    with torch.no_grad():
        model.view_embed.normal_()
        x = torch.randn(4, 4, 3, 64, 64)
        v = torch.tensor([[0, 0, 1, 1]] * 4)
        whole = mv.ARM.forward(model, x, v)
        parts = torch.cat([mv.ARM.forward(model, x[i:i + 1], v[i:i + 1]) for i in range(4)])
    assert torch.allclose(whole, parts, atol=1e-5)


def test_flip_views_uses_each_views_own_mask(ids):
    ev = mv.make_dataset(FULL_DIR, CROP_DIR, ids[:1], labels(ids[:1]), 36, False, 42, 224)
    x, _, uid, views = mv.collate([ev[0]])
    full, crop = h.common.Corpus(FULL_DIR), h.common.Corpus(CROP_DIR)
    flipped = mv.flip_views(x, views, uid, full, crop, 36)
    assert torch.equal(flipped[:, :36], h.flip_inputs(x[:, :36], uid, full, 36))
    assert torch.equal(flipped[:, 36:], h.flip_inputs(x[:, 36:], uid, crop, 36))
    with pytest.raises(ValueError):
        mv.flip_views(x, views.flip(1), uid, full, crop, 36)


def test_train_step_and_tta_evaluate_on_real_windows(ids, tmp_path):
    from tools.hb155_fit_v1 import Progress
    torch.set_num_threads(2)
    model = mv.attach_views(tiny())
    ds = mv.make_dataset(FULL_DIR, CROP_DIR, ids[:2], labels(ids[:2]), 12, True, 42, 224)
    loader = h.DataLoader(ds, batch_size=2, collate_fn=mv.ARM.collate_fn)
    opt, sched, scaler = h.optimizer_stack(model, 1, "cpu", "none")
    counters = {"updates": 0, "skips": 0}
    progress = Progress(tmp_path / "beat.json", "test")
    before = model.view_embed.detach().clone()
    record = mv.train_epoch(mv.ARM, model, loader, opt, sched, scaler, torch.ones(12), "cpu", "none", 0, counters, progress)
    assert record["batches"] == counters["updates"] == 1 and not torch.equal(before, model.view_embed.detach())
    full, crop = h.common.Corpus(FULL_DIR), h.common.Corpus(CROP_DIR)
    ev = mv.make_dataset(FULL_DIR, CROP_DIR, ids[2:], labels(ids[2:]), 36, False, 42, 224)
    eval_loader = h.DataLoader(ev, batch_size=1, collate_fn=mv.ARM.collate_fn)
    values, ordered, batches = mv.evaluate(mv.ARM, model, eval_loader, "cpu", "none", full, crop, progress, tta=True)
    assert ordered == ids[2:] and values.shape == (1, 12) and h.np.isfinite(values).all() and batches == 1
    assert not torch.cuda.is_initialized()
