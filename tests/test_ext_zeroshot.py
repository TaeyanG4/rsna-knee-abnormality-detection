"""EXP137 tiny CPU tests; CUDA initialization is forbidden."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from tools import ext_zeroshot as ez
from training.src.raptor_common import Corpus, SLOTS64


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()

    def make_fail():
        def fail(*_args, **_kwargs):
            raise AssertionError("EXP137 tests must never initialize CUDA")
        return fail

    # Distinct functions: torch._dynamo rejects one function registered under several rules.
    for name in ("_lazy_init", "init", "get_rng_state", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, make_fail())
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def tiny_corpus(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    ids = np.array(["case-a", "case-b", "case-c"])
    masks = np.ones((3, 64), np.uint8)
    masks[0, 18:32] = 0
    masks[2, 52:64] = 0
    volumes = np.zeros((3, 64, 8, 8), np.uint8)
    for study in range(3):
        for slot in range(64):
            volumes[study, slot] = np.arange(64, dtype=np.uint8).reshape(8, 8) + study * 10 + slot
    np.save(directory / "ids.npy", ids)
    np.save(directory / "masks.npy", masks)
    np.save(directory / "volumes.npy", volumes)
    (directory / "manifest.json").write_text(json.dumps({
        "contract": ez.CONTRACT, "spec": {"slots": SLOTS64, "depth": 64, "img": 8},
        "shards": [{"start": 0, "stop": 3, "file": "volumes.npy"}]}), encoding="utf-8")
    return Corpus(directory)


def test_corpus_statistics_and_middle_slot(tiny_corpus):
    c = tiny_corpus
    assert ez.groups(c.spec)[1] == ("Sagittal_fluid0", 18, 32)
    sampled = ez.sampled_ids(c.ids, c.masks, 18, 32, 24, 137)
    assert set(sampled) == {"case-b", "case-c"}
    assert sampled == ez.sampled_ids(c.ids, c.masks, 18, 32, 24, 137)
    image = ez.middle_image(c, "case-a", 0, 18)
    np.testing.assert_array_equal(image, c.volume("case-a")[8])
    stats = ez.corpus_statistics(c)
    assert stats["Sagittal_fluid0"]["slot_fill_rate"] == pytest.approx(2 / 3)
    assert stats["Axial"]["study_fill_rate"] == pytest.approx(2 / 3)
    assert stats["Sagittal_fluid1"]["uint8_quantiles"]["50"] > 0
    assert not ez.qa_flags(stats, stats)
    changed = json.loads(json.dumps(stats))
    changed["Coronal_fluid1"]["slot_fill_rate"] = 0
    assert any("slot-fill" in flag for flag in ez.qa_flags(changed, stats))


def test_montage_and_duplicate_label_guard(tiny_corpus, tmp_path):
    path = tmp_path / "montage.png"
    assert ez.montage(tiny_corpus, tiny_corpus, "Sagittal_fluid0", 18, 32, 2, path) == 2
    assert path.is_file() and path.stat().st_size > 100
    labels = tmp_path / "labels.csv"
    labels.write_text("study_key,ACL\ncase-a,0\ncase-a,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        ez.read_table(labels)


def test_rank_blend_auc_and_bootstrap():
    assert ez.binary_label("1.0") == 1
    assert ez.binary_label("0") == 0
    assert ez.binary_label("") is None
    with pytest.raises(ValueError, match="non-binary"):
        ez.binary_label("2")
    assert ez.rank_average([2, 2, 8, 0]).tolist() == [.625, .625, 1., .25]
    folds = {}
    for seed in ez.SEEDS:
        for fold in range(5):
            folds[(seed, fold)] = np.tile(np.array([[.1], [.5], [.9]]), (1, 12))
    np.testing.assert_allclose(ez.blend_folds(folds)[:, 0], [1 / 3, 2 / 3, 1])
    assert ez.auc([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    point, ci = ez.bootstrap_auc([0, 0, 1, 1], [0, 0, 1, 1], draws=100)
    assert point == 1 and ci == [1, 1]
    assert ez.bootstrap_auc([1, 1], [.2, .5], draws=5) == (None, None)


class FakeModel(nn.Module):
    def forward(self, x):
        # A known flip-sensitive value, returned as twelve logits.
        value = x[:, :, :, :, 0].mean(dim=(1, 2, 3))
        return value[:, None].repeat(1, 12)


def test_infer_fold_uses_plane_pair():
    x = torch.zeros((2, 36, 3, 4, 4), dtype=torch.float32)
    x[..., 0] = 2
    mirrored = torch.flip(x, (-1,))
    loader = [(x, mirrored, torch.zeros((2, 12)), ["a", "b"])]
    scores, ids = ez.infer_fold(FakeModel(), loader, "cpu")
    assert ids == ["a", "b"]
    expected = (float(torch.sigmoid(torch.tensor(2.))) + .5) / 2
    np.testing.assert_allclose(scores, expected, rtol=1e-6)


def test_native_mapping_and_gate(tmp_path):
    label_dir = tmp_path / "storage/labels/external"
    label_dir.mkdir(parents=True)
    ids = [f"case-{i}" for i in range(70)]
    mapped = label_dir / "MRNET_12.csv"
    native = label_dir / "MRNET_native.csv"
    import csv
    with mapped.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["study_key", *ez.TARGETS])
        for i, uid in enumerate(ids):
            writer.writerow([uid, int(i >= 35), *([""] * 11)])
    with native.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["study_key", "acl", "any_meniscus", "abnormal"])
        for i, uid in enumerate(ids):
            writer.writerow([uid, int(i >= 35), int(i >= 35), int(i >= 35)])
    predictions = np.tile(np.linspace(.05, .95, len(ids))[:, None], (1, 12))
    result = ez.evaluate_predictions("MRNET", ids, predictions, root=tmp_path, draws=100)
    assert result["auc_gate"]
    assert result["targets"]["native:acl <- ACL"]["ci95"] == [1., 1.]
    assert not result["targets"]["mapped:ACL"]["primary"]
    assert len(result["eligible_primary"]) == 2
    limited = ez.evaluate_predictions("MRNET", ids[:8], predictions[:8], root=tmp_path, draws=20, limited=True)
    assert limited["limited"] and not limited["auc_gate"]


def test_compare_requires_matching_snapshots(tmp_path):
    directory = tmp_path / "MRNET"
    directory.mkdir()
    ez.atomic_json(directory / "qa.json", {"source": "MRNET", "source_manifest_sha256": "one",
                                                 "flags": [], "manual_montage_review": "PASS"})
    ez.atomic_json(directory / "zeroshot_full.json", {"source": "MRNET", "cache_manifest_sha256": "two"})
    with pytest.raises(ValueError, match="snapshot"):
        ez.compare("MRNET", base=tmp_path)
