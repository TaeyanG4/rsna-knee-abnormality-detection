"""EXP192 v2.0.0 geometry/checkpoint parity. Changelog: retired producer tests move to explicit frontend/history tests."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
import uuid
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch._dynamo  # lazy optimizer imports must precede CUDA tripwires

from tools import exp192_coatnet_crop_f0 as exp
import exp192_crop as crop
import exp143_own0949 as frozen
from raptor_common import study_rng, train_centers, eval_centers


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "_lazy_init", Mock(side_effect=AssertionError("CUDA forbidden")))
    yield
    torch.set_num_threads(old_threads)
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def volume():
    z, y, x = np.ogrid[:96, :384, :384]
    return ((3 * z + y // 4 + x // 2) % 256).astype(np.uint8)


def numpy_bilinear(window, resolution):
    """Independent half-pixel bilinear oracle, no torch or candidate helpers."""
    n = window.shape[-1]
    # PyTorch uses float32 source coordinates for float32 input. Matching that
    # arithmetic matters near the synthetic image's abrupt modulo boundaries.
    at = np.maximum((np.arange(resolution, dtype=np.float32) + .5) * np.float32(n / resolution) - .5, 0)
    lo = at.astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    alpha = at - lo
    x = window.astype(np.float32) / 255
    cols = x[:, :, lo] * (1 - alpha) + x[:, :, hi] * alpha
    return cols[:, lo, :] * (1 - alpha)[None, :, None] + cols[:, hi, :] * alpha[None, :, None]


@pytest.mark.parametrize("resolution", [384, 320])
@pytest.mark.parametrize("mirror", [False, True])
def test_independent_crop_resize_mirror_oracle(volume, resolution, mirror):
    # Both sagittal and coronal/axial, on both parities, including slot boundaries.
    centers = np.array([10, 10, 47, 47, 48, 48, 77, 77, 78, 78, 94, 94])
    actual = crop.scale_windows(volume, centers, resolution, mirror)
    assert actual.shape == (12, 3, resolution, resolution) and actual.dtype == np.float32
    for j, c in enumerate(centers):
        w = volume[[c - 1, c, c + 1]]
        if mirror:
            w = w[::-1] if c < 48 else w[:, :, ::-1]
        if j % 2:
            w = w[:, 55:329, 55:329]
        np.testing.assert_allclose(actual[j], numpy_bilinear(w, resolution), rtol=0, atol=2e-5)


def test_training_and_oof_share_positional_mix(volume, monkeypatch):
    mask = np.ones(96, np.uint8)
    cs = [10, 10, 48, 48, 60, 60, 78, 78, 80, 80, 90, 90]
    monkeypatch.setattr(crop, "train_centers", lambda *_: cs)
    monkeypatch.setattr(crop, "eval_centers", lambda *_: cs + [90] * 82)
    # Different output resolutions, independently verified same physical windows.
    train = crop.windows(volume, mask, True, jitter=False)
    oof = crop.windows(volume, mask)
    for j, c in enumerate(cs):
        w = volume[[c - 1, c, c + 1]]
        if j % 2:
            w = w[:, 55:329, 55:329]
        for actual, res in ((train[j], 384), (oof[j], 320)):
            expected = (numpy_bilinear(w, res) - frozen.MEAN[0]) / frozen.STD[0]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=9e-5)
    assert train.flags.c_contiguous and oof.flags.c_contiguous


@pytest.mark.parametrize("epoch", [0, 7, 15])
def test_even_train_windows_equal_control_and_intensity_stream_unchanged(volume, epoch):
    mask = np.ones(96, np.uint8)
    actual = crop.windows(volume, mask, True, 42, epoch, "synthetic")
    control = frozen.windows(volume, mask, True, 42, epoch, "synthetic")
    np.testing.assert_array_equal(actual[::2], control[::2])
    assert not np.array_equal(actual[1::2], control[1::2])
    cs = train_centers(mask, 96, 12, study_rng(42, epoch, "synthetic"))
    raw = crop.scale_windows(volume, cs, 384)
    factor = 1 + (study_rng(42, epoch, "synthetic", "intensity").random() - .5) * .20
    np.testing.assert_array_equal(actual, (np.clip(raw * factor, 0, 1) - frozen.MEAN) / frozen.STD)


@pytest.mark.parametrize("valid", [(0, 96), (48, 66), (5, 7), (90, 91)])
def test_centres_and_sparse_missing_plane_fallback_are_frozen(volume, valid, monkeypatch):
    mask = np.zeros(96, np.uint8)
    mask[slice(*valid)] = 1
    seen = []
    def scale(_vol, centers, res, mirror=False):
        seen.append(list(centers))
        return np.zeros((len(centers), 3, 1, 1), np.float32)
    monkeypatch.setattr(crop, "scale_windows", scale)
    crop.windows(volume, mask, True, 42, 3, "synthetic")
    crop.windows(volume, mask)
    assert seen == [list(train_centers(mask, 96, 12, study_rng(42, 3, "synthetic"))),
                    list(eval_centers(mask, 96, 94))]


def test_invalid_geometry_fails_closed(volume):
    for cs, res in (([1] * 11, 384), ([0] * 12, 384), ([95] * 12, 384),
                    ([1.] * 12, 384), ([1] * 12, 224)):
        with pytest.raises(ValueError):
            crop.scale_windows(volume, cs, res)
    with pytest.raises(ValueError):
        crop.scale_windows(volume[:, :320, :320], [1] * 12, 384)
    with pytest.raises(ValueError):
        crop.scale_windows(volume.astype(np.float32), [1] * 12, 384)
    with pytest.raises(ValueError):
        crop.windows(volume, np.zeros(96, np.uint8))
    with pytest.raises(ValueError):
        crop.windows(volume, np.ones(96, np.uint8), True, mirror=True)


def test_dataset_bn_epoch_shared_tensor_and_spawn_pickle(volume, tmp_path):
    uid = "SYNTHETIC"
    (tmp_path / "manifest.json").write_text(json.dumps(dict(contract=frozen.CONTRACT,
        spec=frozen.SPEC, finished=True, ids=[uid])), encoding="utf-8")
    np.save(tmp_path / "study_00000.npy", volume)
    mask = np.ones(96, np.uint8)
    (tmp_path / "study_00000.json").write_text(json.dumps(dict(uid=uid, mask=mask.tolist())), encoding="utf-8")
    ds = crop.CropStudyDataset(tmp_path, [uid], {uid: [0.] * 12}, 42, augment=False)
    ds.set_epoch(16)
    assert ds.epoch.is_shared()
    restored = pickle.loads(pickle.dumps(ds))
    x, y, key = restored[0]
    np.testing.assert_array_equal(x.numpy(), crop.windows(volume, mask, True, 42, 16, uid, jitter=False))
    assert x.shape == (12, 3, 384, 384) and y.shape == (12,) and key == uid
    # Real Windows spawn, shared epoch update observed by a persistent worker.
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1,
        persistent_workers=True, multiprocessing_context="spawn", collate_fn=frozen.collate)
    try:
        batch = next(iter(loader))
        # Worker torch uses one thread; CPU bilinear vectorization differs from
        # this two-thread parent by <=1.2e-6 after normalization.
        np.testing.assert_allclose(batch[0][0].numpy(), x.numpy(), rtol=0, atol=1.5e-6)
        ds.set_epoch(17)
        batch = next(iter(loader))
        np.testing.assert_allclose(batch[0][0].numpy(), crop.windows(volume, mask, True, 42, 17, uid, jitter=False), rtol=0, atol=1.5e-6)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_count = 0
        self.bags = []
    def backbone(self, x):
        self.enc_count += len(x)
        return x[:, :, ::32, ::32].mean((-1, -2))
    def head(self, x):
        self.bags.append(x.detach().clone())
        return x.mean(1).repeat(1, 4)


def test_inference_fixed_calls_chunk_parity_and_mirror(volume):
    mask = np.ones(96, np.uint8)
    model = RecordingModel().eval()
    first = crop.infer_pair(model, volume, mask, chunk=15)
    assert model.enc_count == 188 and [x.shape[1] for x in model.bags] == [94, 94]
    other = RecordingModel().eval()
    second = crop.infer_pair(other, volume, mask, chunk=16)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert a.shape == (12,) and np.isfinite(a).all() and ((0 <= a) & (a <= 1)).all()
    # Rebuild each full bag independently to catch parity resets at chunk boundaries.
    for mirror, actual in zip((False, True), model.bags):
        expected = torch.from_numpy(crop.windows(volume, mask, mirror=mirror))[:, :, ::32, ::32].mean((-1, -2))
        torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        crop.infer_pair(model.train(), volume, mask)
    with pytest.raises(ValueError):
        crop.infer_pair(model.eval(), volume, mask, chunk=0)
    model.head = lambda _: torch.full((1, 12), float("nan"))
    with pytest.raises(FloatingPointError):
        crop.infer_pair(model, volume, mask)








def test_crop_checkpoint_metadata_is_required(monkeypatch):
    parent_check = Mock()
    monkeypatch.setattr(exp, "BASE_CHECK", parent_check)
    exp.check_crop_checkpoint(dict(window_policy=exp.CHECKPOINT_POLICY,
                                   crop_policy=copy.deepcopy(crop.CROP_SPEC)), {}, "sha")
    parent_check.assert_called_once()
    for ck in ({}, dict(window_policy="legacy", crop_policy=crop.CROP_SPEC),
               dict(window_policy=exp.CHECKPOINT_POLICY, crop_policy={})):
        with pytest.raises(ValueError, match="mixed-crop serving policy"):
            exp.check_crop_checkpoint(ck, {}, "sha")


















