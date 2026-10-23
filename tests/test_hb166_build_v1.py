"""CPU tests v1.0.0 for H-B-166 registration, admission gates and fail-closed fit/panel entry points."""
import copy
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # empty, never -1 (Windows cu121 backward crash)

import pytest

from tools import hb166_build_v1 as b

REGISTERED = (b.WORK / "registration.json").exists()
needs_registration = pytest.mark.skipif(not REGISTERED, reason="stage not registered yet")


def test_closure_binds_new_and_reused_sources():
    closure = set(b.source_closure())
    for path in ("training/arms/hb166_multiview_v1.py", "tools/hb166_fit_v1.py", "tools/hb166_panel_v1.py",
                 "tools/hb166_readout_v1.py", "tools/hb166_watchdog_v1.py", "tools/hb155_fit_v1.py",
                 "training/arms/hb155_crops_v1.py", "training/arms/native_resolution_v2.py",
                 "training/src/raptor_exp185.py", "tools/multi_lens_readout_l4_v2.py"):
        assert path in closure


def test_gold58_roster_pinned_and_outside_fold():
    ids = b.gold_ids()
    assert len(ids) == 58 and b.uid_hash(ids) == b.GOLD_UID_SHA256
    c = b.read(b.b155.config_path("control", 42))
    assert not set(ids) & set(c["train"] + c["val"])


def test_control_bindings_match_pins():
    for seed in b.SEEDS:
        files = b.control_bindings(seed)
        base = b.rel(b.CONTROL_WORK / "fits" / b.control_member(seed))
        assert files[base + "/val.csv"] == b.sha(b.CONTROL_WORK / "fits" / b.control_member(seed) / "val.csv")
        assert files[base + "/final.pt"] == b.CONTROL_PINS[seed]["checkpoint"]


def test_control_pin_drift_rejected(monkeypatch):
    pins = copy.deepcopy(b.CONTROL_PINS)
    pins[42]["val_csv"] = "0" * 64
    monkeypatch.setattr(b, "CONTROL_PINS", pins)
    with pytest.raises(ValueError, match="pin"):
        b.control_bindings(42)


def test_recipe_is_the_control_recipe():
    c = b.expected(42)
    control = b.b155.validate(b.b155.config_path("control", 42))
    for key in ("recipe", "train", "val", "weights", "label_path", "label_semantic_sha256", "optimizer_steps",
                "exclusions_sha256", "side_unknown_policy", "corpus", "corpus_contract"):
        assert c[key] == control[key], key
    assert (c["batch"], c["microbatch"], c["accumulation"], c["workers"], c["epochs"], c["train_k"], c["resolution"]) == \
        (8, 2, 4, 2, 20, 12, 224)
    assert c["window_budget"]["train"] == dict(full=12, crop=12, bag=24) and c["exposure"] == "EXPOSED"
    assert c["clean_oof"] is False and c["gate"]["four_ruler_macro_min"] == .002 and c["gate"]["target_floor"] == -.005
    e = b.expected(43, engineering=True)
    assert len(e["train"]) == 2 and len(e["val"]) == 1 and len(e["gold"]) == 1 and e["engineering_only"]


@needs_registration
def test_registered_configs_validate_and_drift_is_rejected(tmp_path):
    for seed in b.SEEDS:
        for engineering in (False, True):
            b.validate(b.config_path(seed, engineering))
    assert b.registration_binding()
    c = b.read(b.config_path(42))
    c["recipe"]["updates"]["bb_lr"] = 1e-3
    path = tmp_path / "multiview-s42.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="binding differs"):
        b.validate(path)


@needs_registration
def test_registration_drift_rejected(monkeypatch):
    body = b.registration_body()
    body["gate"] = {**body["gate"], "four_ruler_macro_min": .001}
    monkeypatch.setattr(b, "registration_body", lambda: body)
    with pytest.raises(ValueError, match="registration"):
        b.registration_binding()


@needs_registration
@pytest.mark.parametrize("review", [None, {"schema": "hb166-review/v1", "verdict": "FAIL", "independent": True, "reviewer": "x"},
                                    {"schema": "hb166-review/v1", "verdict": "PASS", "independent": False, "reviewer": "x"},
                                    {"schema": "hb155-review/v1", "verdict": "PASS", "independent": True, "reviewer": "x"},
                                    {"schema": "hb166-review/v1", "verdict": "PASS", "independent": True, "reviewer": "x",
                                     "bindings": {"stale": "0"}}])
def test_fit_admission_fails_closed(tmp_path, monkeypatch, review):
    monkeypatch.setattr(b, "review_bindings", lambda: {"fresh": "1"})
    path = None
    if review is not None:
        path = tmp_path / "review.json"
        path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        b.require_fit(b.config_path(42), path)
    assert b.admit(path)["status"] == "HOLD"


@needs_registration
def test_bound_review_admits_and_engineering_cannot_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "review_bindings", lambda: {"fresh": "1"})
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"schema": "hb166-review/v1", "verdict": "PASS", "independent": True,
                                "reviewer": "test", "bindings": {"fresh": "1"}}))
    assert b.admit(path)["status"] == "ADMITTED"
    with pytest.raises(ValueError, match="engineering"):
        b.require_fit(b.config_path(42, True), path)


@needs_registration
def test_fit_and_panel_entry_points_refuse_bypass_before_cuda(tmp_path):
    from tools import hb166_fit_v1 as f
    from tools import hb166_panel_v1 as p
    with pytest.raises(ValueError, match="CUDA"):
        f.fit(b.config_path(42), tmp_path / "a", device="cpu", smoke=False)
    with pytest.raises(ValueError, match="review"):
        f.fit(b.config_path(42), tmp_path / "b", device="cuda", smoke=False)
    with pytest.raises(ValueError, match="engineering"):
        f.fit(b.config_path(42), tmp_path / "c", device="cpu", smoke=True)
    os.environ.pop("HB155_TOKEN", None)
    with pytest.raises(ValueError, match="watchdog"):
        f.fit(b.config_path(42), tmp_path / "d", device="cuda", smoke=True)
    with pytest.raises(ValueError, match="CUDA"):
        p.panel(42, tmp_path / "e", device="cpu", smoke=False)
    with pytest.raises(ValueError, match="review"):
        p.panel(42, tmp_path / "f", device="cuda", smoke=False)
    with pytest.raises(ValueError, match="watchdog"):
        p.panel(42, tmp_path / "g", device="cuda", smoke=True)
    assert not f.h.torch.cuda.is_initialized()


def snapshot():
    return b.read(b.CONTROL_WORK / "runtime" / "4090.json")


def test_runtime_pairing_rules():
    reference = snapshot()
    assert b.check_runtime(copy.deepcopy(reference), reference)
    drift = copy.deepcopy(reference)
    drift["torch_interop_threads"] = 64
    drift["libs"]["cv2_threads"] = 8
    assert b.check_runtime(drift, reference)
    for key, value in (("torch", "2.6.0+cu124"), ("cudnn_benchmark", True), ("matmul_tf32", True)):
        bad = copy.deepcopy(reference)
        bad[key] = value
        with pytest.raises(ValueError, match="runtime differs"):
            b.check_runtime(bad, reference)
    bad = copy.deepcopy(reference)
    bad["libs"]["timm"] = "1.0.20"
    with pytest.raises(ValueError, match="runtime differs"):
        b.check_runtime(bad, reference)
    bad = copy.deepcopy(reference)
    bad["gpu"] = "NVIDIA GeForce RTX 3090"
    with pytest.raises(ValueError, match="4090"):
        b.check_runtime(bad, reference)
    assert b.require_runtime(copy.deepcopy(reference))


@needs_registration
@pytest.mark.parametrize("field,value", [("peak_reserved_bytes", 13 * (1 << 30)), ("runtime_pairing", None),
                                         ("gpu_name", "NVIDIA GeForce RTX 3090"), ("optimizer_steps", 1),
                                         ("passed", False), ("config_sha256", "0" * 64)])
def test_cuda_smoke_gates(tmp_path, field, value):
    good = dict(passed=True, smoke=True, engineering_only=True, device="cuda", seed=42, version=b.VERSION,
                config_sha256=b.sha(b.config_path(42)), gpu_name=b.GPU, peak_reserved_bytes=7 << 30, full_shape=True,
                runtime_pairing=True, arm="multiview", optimizer_steps=2,
                validation_shapes=["k24", "k36", "k36flip", "gold_k36flip"], batch=8, val_batch=4,
                healthy_rates=dict(train_step=1., validation=1., panel=1.))
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(good))
    b.check_smoke("cuda", "fit", 42, path)
    path.write_text(json.dumps({**good, field: value}))
    with pytest.raises((ValueError, KeyError)):
        b.check_smoke("cuda", "fit", 42, path)


@needs_registration
def test_cpu_smoke_requires_real_model_and_no_cuda(tmp_path):
    good = dict(passed=True, smoke=True, engineering_only=True, device="cpu", seed=43, version=b.VERSION,
                config_sha256=b.sha(b.config_path(43, True)), cpu_cuda_initialized=False, arm="multiview",
                optimizer_steps=1, validation_shapes=["k24", "k36", "k36flip", "gold_k36flip"], cpu_model="real")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(good))
    b.check_smoke("cpu", "fit", 43, path)
    for field, value in (("cpu_model", "tiny"), ("cpu_cuda_initialized", True)):
        path.write_text(json.dumps({**good, field: value}))
        with pytest.raises(ValueError):
            b.check_smoke("cpu", "fit", 43, path)


def test_sequence_rejects_bad_mode_and_fit_needs_admission(monkeypatch):
    with pytest.raises(ValueError):
        b.sequence("train")
    monkeypatch.setattr(b, "disk_gate", lambda *a, **k: 1)
    monkeypatch.setattr(b, "admit", lambda review: {"status": "HOLD"})
    monkeypatch.setattr(b.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="admission"):
        b.sequence("fit", None, "t1")


@needs_registration
@pytest.mark.parametrize("mode", ["smoke", "fit"])
def test_sequence_commands_pass_the_supervisor_allowlist(tmp_path, monkeypatch, mode):
    from tools import hb166_watchdog_v1 as w
    calls = []
    monkeypatch.setattr(b, "disk_gate", lambda *a, **k: 1)
    monkeypatch.setattr(b, "admit", lambda review: {"status": "ADMITTED"})
    monkeypatch.setattr(b, "immutable", lambda *a, **k: None)
    monkeypatch.setattr(b, "read", lambda p: {"healthy_rates": {"train_step": 1., "validation": 1., "panel": 1.}})
    monkeypatch.setattr(b.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    result = b.sequence(mode, tmp_path / "review.json" if mode == "fit" else None, "t1")
    assert result["jobs"] == ["panel-control-s42", "panel-control-s43", "multiview-s42", "multiview-s43"]
    for cmd in calls:
        child = cmd[cmd.index("--") + 1:]
        module, child_mode, device = w.classify(child)
        assert child_mode == mode and device == "cuda"
        assert ("--measurement" in cmd) == (mode == "smoke") and ("--healthy-rates" in cmd) == (mode == "fit")
        assert ("--review" in child) == (mode == "fit")
        assert w.WORK in __import__("pathlib").Path(cmd[cmd.index("--attempt") + 1]).parents
