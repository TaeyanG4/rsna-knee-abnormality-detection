"""EXP143 contract tests on small synthetic arrays; never initialize CUDA."""
from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data._utils.pin_memory import pin_memory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "src"))
import exp143_own0949 as train143
from tools import exp143_own0949 as guard143


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()

    def fail(*_args, **_kwargs):
        raise AssertionError("EXP143 tests must not initialize CUDA")

    for name in ("_lazy_init", "init", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, fail)
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def tiny_geometry(monkeypatch):
    monkeypatch.setattr(train143, "IMG", 8)
    monkeypatch.setattr(train143, "TRAIN_RES", 8)
    monkeypatch.setattr(train143, "EVAL_RES", 4)
    monkeypatch.setattr(train143, "DEPTH", 8)
    monkeypatch.setattr(train143, "SLOT_SIZES", [2, 2, 2, 1, 1])
    monkeypatch.setattr(train143, "K_EVAL", 6)
    monkeypatch.setattr(train143, "K_TRAIN", 3)
    vol = np.empty((8, 8, 8), np.uint8)
    for z in range(8):
        vol[z] = np.arange(8, dtype=np.uint8)[None, :] + z * 16
    return vol, np.ones(8, dtype=np.uint8)


def test_native_geometry_and_deterministic_window_draw(tiny_geometry):
    vol, mask = tiny_geometry
    plain = train143.windows(vol, mask)
    assert plain.shape == (6, 3, 4, 4)
    assert plain.dtype == np.float32 and plain.flags.c_contiguous
    assert np.isfinite(plain).all()
    assert not np.array_equal(plain, train143.windows(vol, mask, mirror=True))
    a = train143.windows(vol, mask, train=True, seed=42, epoch=2, uid="synthetic")
    b = train143.windows(vol, mask, train=True, seed=42, epoch=2, uid="synthetic")
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3, 3, 8, 8)
    with pytest.raises(ValueError, match="inference only"):
        train143.windows(vol, mask, train=True, mirror=True)
    with pytest.raises(ValueError, match="empty study"):
        train143.windows(vol, np.zeros_like(mask))
    with pytest.raises(ValueError, match="native uint8"):
        train143.windows(vol.astype(np.float32), mask)


def test_anatomical_mirror_uses_centre_slot(tiny_geometry):
    tri = np.arange(2 * 3 * 2 * 4, dtype=np.uint8).reshape(2, 3, 2, 4)
    out = train143.mirror_triplets(tri, [1, 4])
    np.testing.assert_array_equal(out[0], tri[0, ::-1])  # sagittal: slice channels
    np.testing.assert_array_equal(out[1], tri[1, :, :, ::-1])  # coronal: image width
    np.testing.assert_array_equal(tri, np.arange(tri.size, dtype=np.uint8).reshape(tri.shape))


def test_actual_pin_memory_tuple_conversion_and_unpack():
    class Pinnable:
        def pin_memory(self):
            return self

    assert isinstance(pin_memory((Pinnable(), Pinnable(), ["u"])), list)
    batch = (torch.zeros(2, 3, 3, 4, 4), torch.zeros(2, 12), ["a", "b"])
    for value in (batch, list(batch)):
        x, y, ids = train143.unpack_batch(value)
        assert x.shape == (2, 3, 3, 4, 4) and y.shape == (2, 12)
        assert ids == ["a", "b"]
    with pytest.raises((TypeError, ValueError)):
        train143.unpack_batch((batch[0], batch[1]))
    with pytest.raises((TypeError, ValueError)):
        train143.unpack_batch((batch[0], batch[1][:1], batch[2]))


def test_real_raptor_head_with_fake_cpu_backbone():
    class TinyBackbone(nn.Module):
        num_features = 4

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, x):
            return self.conv(x).mean((-1, -2))

    model = train143.build_model(res=8, backbone_factory=TinyBackbone).eval()
    with torch.no_grad():
        out = model(torch.ones(2, 3, 3, 8, 8))
    assert out.shape == (2, 12) and torch.isfinite(out).all()
    assert not torch.cuda.is_initialized()


def test_validate_forward_and_pin_path_are_cuda_free():
    result = train143.tiny_validation()
    assert result["fake_forward"] == [2, 12]
    assert result["actual_pin_memory_tuple_returns_list"]
    assert result["cuda_calls"] == 0


def test_portable_blocker_covers_kaggle_mount(monkeypatch):
    captured = []
    monkeypatch.setattr(sys, "addaudithook", captured.append)
    train143.install_portable_gold_blocker()
    hook = captured[0]
    for p in ("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv",
              str(ROOT / "storage/competition/train.csv"), str(ROOT / "storage/gold/labels.csv")):
        with pytest.raises(PermissionError):
            hook("open", (p, "r", 0))
    hook("open", ("/kaggle/input/exp143-corpus/study_00000.npy", "rb", 0))


def test_corpus_manifest_and_receipt_binding(tmp_path):
    manifest = {"contract": train143.CONTRACT, "spec": train143.SPEC,
                "finished": True, "ids": ["synthetic"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.save(tmp_path / "study_00000.npy", np.zeros((2, 2, 2), np.uint8))
    (tmp_path / "study_00000.json").write_text(json.dumps({"uid": "wrong", "mask": [1, 1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt UID"):
        train143.Corpus(tmp_path).get("synthetic")
    manifest["contract"] = "old-64-slice"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        train143.Corpus(tmp_path)


def test_complete_competition_cohort_and_failure_cases():
    all_ids = [f"synthetic-{i:04d}" for i in range(4407)]
    gold = all_ids[::76][:58]
    selected = guard143.select_training_ids(all_ids, gold, all_ids)
    assert len(selected) == 4349 and not set(selected) & set(gold)
    assert selected == [u for u in all_ids if u not in set(gold)]
    with pytest.raises(ValueError, match="duplicate"):
        guard143.select_training_ids(all_ids + [all_ids[0]], gold, all_ids)
    with pytest.raises(ValueError, match="Gold IDs outside"):
        guard143.select_training_ids(all_ids, ["not-in-competition"], all_ids)
    with pytest.raises(ValueError, match="missing training labels"):
        guard143.select_training_ids(all_ids, gold, [u for u in all_ids if u != selected[0]])


def _write_table(path, rows, columns=None):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns or [guard143.UID, *guard143.TARGETS])
        writer.writerows(rows)


def test_strict_labels_and_prediction_identity(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_table(path, [["b", *([.25] * 12)], ["a", *([.75] * 12)]])
    assert guard143.read_predictions(path, ["a", "b"]) == [[.75] * 12, [.25] * 12]
    with pytest.raises(ValueError, match="cohort"):
        guard143.read_predictions(path, ["a", "c"])
    _write_table(tmp_path / "extra.csv", [["a", *([.5] * 12), 0]],
                 [guard143.UID, *guard143.TARGETS, "unregistered"])
    with pytest.raises(ValueError, match="columns"):
        guard143.read_label_table(tmp_path / "extra.csv")
    _write_table(tmp_path / "duplicate.csv", [["a", *([.5] * 12)], ["a", *([.7] * 12)]])
    with pytest.raises(ValueError, match="duplicate"):
        guard143.read_label_table(tmp_path / "duplicate.csv")
    _write_table(tmp_path / "nan.csv", [["a", *([float("nan")] + [.5] * 11)]])
    with pytest.raises(ValueError, match="finite"):
        guard143.read_label_table(tmp_path / "nan.csv")


def test_prespecified_cohort_rank_blends():
    public = np.tile([.1, .3, .9], (12, 1)).T
    own = public[::-1].copy()
    result = guard143.fixed_blends(public, own)
    assert set(result) == {"public0949", "own143", "primary_public50_own50",
                           "sensitivity_public60_own40"}
    np.testing.assert_array_equal(result["primary_public50_own50"], np.full((3, 12), .5))
    np.testing.assert_allclose(result["sensitivity_public60_own40"][:, 0], [.4, .5, .6])
    with pytest.raises(ValueError, match="differ"):
        guard143.fixed_blends(public, own[:2])
    with pytest.raises(ValueError, match="finite"):
        guard143.rank_matrix(np.full((3, 12), np.nan))


def test_validation_receipt_binds_registration_hash(tmp_path):
    protocol = tmp_path / "registration.json"
    protocol.write_text('{"synthetic": 1}', encoding="utf-8")
    receipt = guard143.validation_path(protocol)
    receipt.write_text(json.dumps({"status": "VALIDATED_NO_TRAINING",
        "protocol_sha256": guard143.sha256(protocol),
        "actual_pin_memory_tuple_returns_list": True,
        "full_corpus_hashes": True}), encoding="utf-8")
    guard143.require_validation(protocol)
    protocol.write_text('{"synthetic": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        guard143.require_validation(protocol)


def test_corpus_full_hash_detects_changed_volume(tmp_path, monkeypatch):
    monkeypatch.setitem(train143.SPEC, "depth", 2)
    monkeypatch.setitem(train143.SPEC, "img", 2)
    volume = tmp_path / "study_00000.npy"
    np.save(volume, np.ones((2, 2, 2), np.uint8))
    receipt = tmp_path / "study_00000.json"
    receipt.write_text(json.dumps({"uid": "synthetic", "positive_mask_slices": 2,
                                   "file_sha256": guard143.sha256(volume)}), encoding="utf-8")
    manifest = {"contract": train143.CONTRACT, "spec": train143.SPEC,
                "finished": True, "ids": ["synthetic"], "files": [{"row": 0,
                "uid": "synthetic", "sha256": guard143.sha256(volume),
                "receipt_sha256": guard143.sha256(receipt)}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert guard143.check_corpus(tmp_path, ["synthetic"], full_hash=True)["ids"] == ["synthetic"]
    with volume.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="volume bytes changed"):
        guard143.check_corpus(tmp_path, ["synthetic"], full_hash=True)
