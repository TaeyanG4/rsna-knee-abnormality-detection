# v2.0.0: historical authentication and typed lifecycle replace producer/shim tests.
"""CPU-only EXP181 contract tests. Synthetic images/labels; no label rows printed."""
from __future__ import annotations

import copy
import json
import pickle
import random
import re
from pathlib import Path
import sys

import numpy as np
import pytest
# torchvision pulls in torch._dynamo, which must not see the patched CUDA stubs below.
import torchvision.transforms.functional  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "training/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from tools import exp181_coatnet_f0_bridge as g  # noqa: E402
import exp143_own0949 as frozen  # noqa: E402
import exp181_aug1 as aug  # noqa: E402


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    import torch
    assert not torch.cuda.is_initialized()
    def fail(*args, **kwargs):
        pytest.fail("CPU contract test attempted CUDA")
    for name in ("_lazy_init", "init", "is_available", "get_device_name", "set_device",
                 "manual_seed_all", "is_bf16_supported", "mem_get_info", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, fail)
    yield
    assert not torch.cuda.is_initialized()


def synthetic_study(seed=0, missing=(40, 60)):
    r = np.random.default_rng(seed)
    vol = r.integers(0, 256, size=(96, 384, 384), dtype=np.uint8)
    mask = np.ones(96, np.uint8)
    mask[[0, 1, 95]] = 0
    vol[[0, 1, 95]] = 0
    lo, hi = missing
    vol[lo:hi] = 0
    mask[lo:hi] = 0
    return vol, mask


class FixedRng:
    """Identity geometry, gamma 1, always flip."""
    def __init__(self):
        self.calls = 0
    def uniform(self, a, b):
        self.calls += 1
        return {(-10., 10.): 0., (.9, 1.1): 1., (-.05, .05): 0., (.8, 1.25): 1.}[(a, b)]
    def random(self):
        self.calls += 1
        return 0.


@pytest.mark.parametrize("epoch,uid", [(0, "u-a"), (7, "u-b"), (15, "1.2.3")])
def test_pre_augmentation_path_equals_frozen_windows(epoch, uid):
    from raptor_common import study_rng, train_centers
    vol, mask = synthetic_study()
    x01, centers, _ = aug.train_windows01(vol, mask, 42, epoch, uid)
    assert np.array_equal(aug.normalize(x01), frozen.windows(vol, mask, True, 42, epoch, uid))
    assert centers == train_centers(mask, 96, 12, study_rng(42, epoch, uid))


def test_aug1_deterministic_changes_input_and_keeps_contract():
    vol, mask = synthetic_study(1)
    a = aug.aug1_windows(vol, mask, 42, 3, "u")
    b = aug.aug1_windows(vol, mask, 42, 3, "u")
    control = frozen.windows(vol, mask, True, 42, 3, "u")
    assert np.array_equal(a, b) and a.shape == control.shape == (12, 3, 384, 384)
    assert a.dtype == np.float32 and np.isfinite(a).all() and not np.array_equal(a, control)
    assert not np.array_equal(a, aug.aug1_windows(vol, mask, 42, 4, "u"))


def test_flip_only_coronal_axial_and_shared_by_channels():
    x = np.random.default_rng(2).random((5, 3, 32, 32), dtype=np.float32)
    centers = [10, 47, 48, 77, 90]  # slots 0, 1, 2, 3, 4 by the mirror_triplets rule
    assert aug.plane_slots(centers).tolist() == [0, 1, 2, 3, 4]
    rng = FixedRng()
    out = aug.augment_triplets(x, centers, rng)
    assert rng.calls == 6 * 5
    for j in (0, 1):
        assert np.allclose(out[j], x[j], atol=1e-5)
    for j in (2, 3, 4):
        assert np.allclose(out[j], x[j][..., ::-1], atol=1e-5)


def test_missing_inputs_preserved_and_draw_order_fixed():
    x = np.random.default_rng(3).random((4, 3, 48, 48), dtype=np.float32)
    x[1] = 0
    x[2, 1] = 0
    centers = [20, 30, 50, 80]
    r1, r2 = random.Random(5), random.Random(5)
    out = aug.augment_triplets(x, centers, r1)
    assert np.array_equal(out[1], x[1]) and not out[2, 1].any()
    assert out.min() >= 0 and out.max() <= 1 and out.dtype == np.float32
    for _ in range(4):
        r2.uniform(-10., 10.); r2.uniform(.9, 1.1); r2.uniform(-.05, .05); r2.uniform(-.05, .05)
        r2.uniform(.8, 1.25); r2.random()
    assert r1.getstate() == r2.getstate()


@pytest.mark.parametrize("bad", ["dtype", "channels", "centre"])
def test_augment_rejects_invalid_inputs(bad):
    x = np.zeros((2, 3, 8, 8), np.float32)
    centers = [10, 50]
    if bad == "dtype":
        x = x.astype(np.float64)
    elif bad == "channels":
        x = np.zeros((2, 2, 8, 8), np.float32)
    else:
        centers = [10, 96]
    with pytest.raises(ValueError):
        aug.augment_triplets(x, centers, random.Random(0))


def test_aug1_spec_matches_d_b_066():
    s = aug.AUG1_SPEC
    assert (s["angle_deg"], s["scale"], s["translate_frac"], s["gamma"], s["flip_p"]) == (
        [-10.0, 10.0], [0.9, 1.1], [-0.05, 0.05], [0.8, 1.25], 0.5)
    assert s["flip_planes"] == ["Coronal", "Axial"] and s["training_only"] is True
    assert g.arm_spec("CONTROL")["augmentation"] is None and g.arm_spec("AUG1")["augmentation"] == s
    with pytest.raises(ValueError):
        g.arm_spec("TWOSCALE")


def fake_corpus(tmp_path, n=2):
    ids = [f"uid-{i}" for i in range(n)]
    for i, uid in enumerate(ids):
        vol, mask = synthetic_study(10 + i)
        np.save(tmp_path / f"study_{i:05d}.npy", vol)
        (tmp_path / f"study_{i:05d}.json").write_text(json.dumps(dict(uid=uid, mask=mask.tolist())))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(contract=frozen.CONTRACT, spec=frozen.SPEC,
                                                            finished=True, ids=ids)))
    return ids


def test_dataset_bn_path_is_frozen_training_path_is_aug1_and_pickles(tmp_path):
    import torch
    ids = fake_corpus(tmp_path)
    labels = {u: [.25] * 12 for u in ids}
    for augment in (False, True):
        a = aug.Aug1StudyDataset(tmp_path, ids, labels, 42, augment=augment)
        c = frozen.StudyDataset(tmp_path, ids, labels, 42, augment=augment)
        a.set_epoch(5); c.set_epoch(5)
        xa, ya, ua = a[1]
        xc, yc, uc = c[1]
        assert torch.equal(ya, yc) and ua == uc == ids[1]
        assert torch.equal(xa, xc) is (not augment)
    vol, mask = frozen.Corpus(tmp_path).get(ids[1])
    assert torch.equal(xa, torch.from_numpy(aug.aug1_windows(vol, mask, 42, 5, ids[1])))
    clone = pickle.loads(pickle.dumps(a))
    assert type(clone) is aug.Aug1StudyDataset and type(clone).__module__ == "exp181_aug1"


def fake_model():
    import torch
    from torch import nn
    class FakeBackbone(nn.Module):
        num_features = 4
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)
        def forward(self, x):
            return self.conv(x).mean((-1, -2))
    torch.random.default_generator.manual_seed(0)
    return frozen.build_model(backbone_factory=FakeBackbone).eval()


def test_infer_pair_equals_frozen_infer_volume():
    model = fake_model()
    vol, mask = synthetic_study(4)
    plain, mirror = g.infer_pair(model, vol, mask)
    assert np.array_equal(.5 * (plain + mirror), frozen.infer_volume(model, vol, mask))
    assert plain.shape == (12,) and not np.array_equal(plain, mirror)
    model.train()
    with pytest.raises(ValueError):
        g.infer_pair(model, vol, mask)


@pytest.fixture(scope="module")
def real_common():
    # Reads pinned structured manifests only; label values stay in memory.
    baseline = g.baseline_registration()
    return baseline, g.build_common(baseline)


def test_real_bridge_contract(real_common):
    baseline, common = real_common
    ids, val = common["train_ids"], common["val_ids"]
    assert len(ids) == len(set(ids)) == 3455 and len(val) == len(set(val)) == 888
    assert not set(ids) & set(val) and set(ids) | set(val) <= set(baseline["train_ids"])
    assert not (set(ids) | set(val)) & set(baseline["excluded_gold_ids"])
    assert len(common["excluded_non_fold_ids"]) == 4
    assert ids == [u for u in baseline["train_ids"] if u in set(ids)]
    assert common["recipe"] == baseline["recipe"] and common["seed"] == 42
    assert all(common["train_labels"][u] == baseline["train_labels"][u] for u in ids)
    assert len(common["pos_weight"]) == 12 and common["label_sha256"] == baseline["label_sha256"]
    s = g.execution_settings(common, 1, "bf16", 42, False)
    assert (s["updates_per_epoch"], s["total_steps"], s["accumulation"], s["max_hours"]) == (216, 3456, 16, 4.0)
    from tools import exp149_label_sources as e149
    canonical, fold_of = e149.fold_rosters(e149._load_manifest("student", ROOT), ROOT)
    assert set(val) == {u for u in canonical if fold_of[u] == 0}


def test_arms_share_bridge_contract_and_differ_only_in_arm_fields(real_common):
    _, common = real_common
    c, a = g.registration(common, "CONTROL", "t"), g.registration(common, "AUG1", "t")
    assert c["bridge_contract_sha256"] == a["bridge_contract_sha256"] == g.json_hash(common)
    assert {k for k in c if c[k] != a[k]} == {"arm", "dataset", "augmentation", "run_dirs", "content_sha256"}
    assert g.registered_run(c).name == "CONTROL" and g.registered_run(a).name == "AUG1"
    bad = copy.deepcopy(a)
    bad["run_dirs"] = {"local": g.run_rel("CONTROL")}
    with pytest.raises(ValueError):
        g.registered_run(bad)


@pytest.mark.parametrize("world,precision,seed,ckpt", [(2, "bf16", 42, False), (1, "fp16", 42, False),
                                                       (1, "bf16", 43, False), (1, "bf16", 42, True)])
def test_execution_settings_reject_other_lanes(real_common, world, precision, seed, ckpt):
    with pytest.raises(ValueError):
        g.execution_settings(real_common[1], world, precision, seed, ckpt)


def test_write_scope(tmp_path):
    with pytest.raises(ValueError):
        g.write_new_json(tmp_path / "x.json", {})
    with pytest.raises(ValueError):
        g.scoped_path(ROOT / "artifacts/experiments/EXP174_OWN0949_JEV/S42/x.json")


def test_check_summary(real_common):
    _, common = real_common
    p = g.registration(common, "AUG1", "t")
    settings = g.execution_settings(p, 1, "bf16", 42, False)
    good = dict(status="COMPLETE_TRAINING_NO_EVALUATION", epochs=16, updates=3456, skips=0,
                history=[dict(epoch=e) for e in range(1, 17)], model_sha256="m",
                identity=dict(protocol_sha256="p", initial_state_sha256=g.INITIAL_STATE_SHA, seed=42,
                              train_n=3455, run_dir=g.run_rel("AUG1"), execution=settings))
    g.check_summary(good, p, "p", "m")
    for key, value in (("updates", 4352), ("skips", 1), ("model_sha256", "x")):
        bad = dict(good, **{key: value})
        with pytest.raises(ValueError):
            g.check_summary(bad, p, "p", "m")
    bad = copy.deepcopy(good)
    bad["identity"]["run_dir"] = g.run_rel("CONTROL")
    with pytest.raises(ValueError):
        g.check_summary(bad, p, "p", "m")


def test_legacy_producers_are_retired():
    for name in ("prepare", "validate", "train", "predict_oof", "build_queue", "queue_json"):
        assert not hasattr(g, name)
    assert g.main(["train", "--preset-arm", "AUG1"]) == 2


def test_current_lifecycle_is_thin_dispatch(monkeypatch):
    from tools import coatnet_config
    calls = []
    monkeypatch.setattr(coatnet_config, "main", lambda argv, **kw: calls.append((argv, kw)))
    g.main(["lifecycle", "register", "--preset-arm", "AUG1", "--arm", "cpu-aug1", "--request", "new.json"])
    assert calls == [(["register", "--arm", "cpu-aug1", "--request", "new.json"], {"preset": "aug1-f0-s42"})]


def test_frozen_trainer_guard_surface_exists():
    text = (ROOT / "training/src/exp143_v2.py").read_text(encoding="utf-8")
    train = text[text.index("def train(args):"):text.index("def scratch_directory")]
    names = set(re.findall(r"guard\.(\w+)", train))
    assert names == {"load_registration", "require_validation", "execution_settings", "registered_run",
                     "write_new_json", "save_torch_atomic", "save_torch_new", "sha256", "TARGETS"}
    assert all(hasattr(g, n) for n in names)


def test_no_isolated_fit_or_global_guard_shim():
    import exp143_v2 as shared
    from tools import exp143_v2 as guard
    assert not hasattr(g, "isolated_fit") and not hasattr(g, "guard_shim")
    before = (shared.StudyDataset, guard, g.e174.base)
    g.load_registration(g.manifest_path("CONTROL"))
    assert before == (shared.StudyDataset, guard, g.e174.base)


def pair(mean, lower, dread, missing=False):
    per = {t: (None if missing and t == "ACL" else 0.) for t in g.TARGETS}
    refs = {n: dict(delta=dread if n == "dread" else mean, per_target_delta=per)
            for n in ("jev", "consensus", "dread", "aquila")}
    return dict(references=refs, primary_mean_delta=mean, primary_mean_ci95=[lower, mean + .01])


@pytest.mark.parametrize("args,status", [((.002, .0001, -.003), "ADVANCE"), ((.0019, .001, 0.), "HOLD"),
                                         ((.01, 0., 0.), "HOLD"), ((.01, .001, -.0031), "HOLD"),
                                         ((.01, .001, 0., True), "HOLD")])
def test_admission_gate(args, status):
    assert g.admission(pair(*args))["status"] == status


def test_score_on_synthetic_references():
    from tools import exp115_vlm_eval as ev
    r = np.random.default_rng(7)
    n = 300
    y = (r.random((n, 12)) < .3).astype(float)
    mask = np.ones((n, 12), bool)
    refs = {name: (y, mask, 1) for name in ("jev", "aquila", "dread", "consensus")}
    draws = [r.integers(0, n, n) for _ in range(50)]
    control = r.random((n, 12))
    same = g.score(dict(primary=control, plain=control), dict(primary=control, plain=control), refs, draws, ev)
    assert same["registered_gate"]["status"] == "HOLD" and same["primary_k94_mirror_tta"]["primary_mean_delta"] == 0
    better = np.clip(.6 * y + .4 * r.random((n, 12)), 0, 1)
    up = g.score(dict(primary=better, plain=better), dict(primary=control, plain=control), refs, draws, ev)
    assert up["status"] == "ADVANCE" and up["primary_k94_mirror_tta"]["primary_mean_ci95"][0] > 0


def test_load_oof_strict(tmp_path):
    ids = ["a", "b"]
    path = tmp_path / "oof.csv"
    path.write_text(",".join([g.base.UID, *g.TARGETS]) + "\n" + "\n".join(
        ",".join([u] + ["0.5"] * 12) for u in ids) + "\n", encoding="utf-8")
    assert g.load_oof(path, ["b", "a"]).shape == (2, 12)
    with pytest.raises(ValueError):
        g.load_oof(path, ["a", "c"])
    path.write_text(",".join([g.base.UID, *g.TARGETS[::-1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        g.load_oof(path, ids)


def test_only_pinned_historical_manifests_are_accepted(tmp_path):
    p = g.load_registration(g.manifest_path("AUG1"))
    assert p["version"] == "1.0.0" and g.VERSION == "2.0.0"
    other = tmp_path / "manifest-AUG1.json"
    other.write_bytes(g.manifest_path("AUG1").read_bytes())
    with pytest.raises(ValueError, match="root|historical manifest"):
        g.load_registration(other)


def test_spawned_dataloader_workers_load_aug1(tmp_path):
    """Windows spawn: workers must unpickle exp181_aug1 + frozen worker_init/collate."""
    import torch
    from torch.utils.data import DataLoader
    ids = fake_corpus(tmp_path)
    labels = {u: [.5] * 12 for u in ids}
    ds = aug.Aug1StudyDataset(tmp_path, ids, labels, 42)
    ds.set_epoch(2)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=frozen.collate,
                        worker_init_fn=frozen.worker_init, multiprocessing_context="spawn")
    got = [frozen.unpack_batch(b) for b in loader]
    for (x, _, uid), u in zip(got, ids):
        vol, mask = frozen.Corpus(tmp_path).get(u)
        assert uid == [u] and torch.equal(x[0], torch.from_numpy(aug.aug1_windows(vol, mask, 42, 2, u)))


def test_exp174_checker_bound_to_real_base_without_shims():
    from tools import exp143_v2 as real
    assert g.e174.base is real and g.base is real
