"""CPU tests v1.0.0 for H-B-166 registration, admission gates and fail-closed fit/panel entry points."""
import copy
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # empty, never -1 (Windows cu121 backward crash)

import pytest

from tools import hb166_fold3_build_v1 as b

REGISTERED = (b.WORK / "registration.json").exists()
needs_registration = pytest.mark.skipif(not REGISTERED, reason="stage not registered yet")


def test_closure_binds_new_and_reused_sources():
    closure = set(b.source_closure())
    for path in ("training/arms/hb166_multiview_v1.py", "tools/hb166_fold3_fit_v1.py", "tools/hb165_readout_v1.py",
                 "tools/hb166_fold3_readout_v1.py", "tools/hb166_fold3_watchdog_v1.py", "tools/hb155_fit_v1.py",
                 "training/arms/hb155_crops_v1.py", "training/arms/native_resolution_v2.py",
                 "training/src/raptor_exp185.py", "tools/multi_lens_readout_l4_v2.py"):
        assert path in closure


def test_gold58_roster_pinned_and_outside_fold():
    ids = b.gold_ids()
    assert len(ids) == 58 and b.uid_hash(ids) == b.GOLD_UID_SHA256
    c = b.read(b.b165.config_path("control", 44))
    assert not set(ids) & set(c["train"] + c["val"])


def test_control_bindings_match_pins():
    for seed in b.SEEDS:
        files = b.control_bindings(seed)
        base = b.rel(b.CONTROL_WORK / "fits" / b.control_member(seed))
        assert files[base + "/val.csv"] == b.sha(b.CONTROL_WORK / "fits" / b.control_member(seed) / "val.csv")
        assert files[base + "/final.pt"] == b.CONTROL_PINS[seed]["checkpoint"]


def test_control_pin_drift_rejected(monkeypatch):
    pins = copy.deepcopy(b.CONTROL_PINS)
    pins[44]["val_csv"] = "0" * 64
    monkeypatch.setattr(b, "CONTROL_PINS", pins)
    with pytest.raises(ValueError, match="pin"):
        b.control_bindings(44)


def test_recipe_is_the_control_recipe():
    c = b.expected(44)
    control = b.b165.validate(b.b165.config_path("control", 44))
    for key in ("recipe", "train", "val", "weights", "label_path", "label_semantic_sha256", "optimizer_steps",
                "exclusions_sha256", "side_unknown_policy", "corpus", "corpus_contract"):
        assert c[key] == control[key], key
    assert (c["batch"], c["microbatch"], c["accumulation"], c["workers"], c["epochs"], c["train_k"], c["resolution"]) == \
        (8, 2, 4, 2, 20, 12, 224)
    assert c["window_budget"]["train"] == dict(full=12, crop=12, bag=24) and c["exposure"] == "EXPOSED"
    assert c["clean_oof"] is False and c["gate"]["four_ruler_macro_min"] == .002 and c["gate"]["target_floor"] == -.005
    e = b.expected(45, engineering=True)
    assert len(e["train"]) == 2 and len(e["val"]) == 1 and len(e["gold"]) == 1 and e["engineering_only"]


@needs_registration
def test_registered_configs_validate_and_drift_is_rejected(tmp_path):
    for seed in b.SEEDS:
        for engineering in (False, True):
            b.validate(b.config_path(seed, engineering))
    assert b.registration_binding()
    c = b.read(b.config_path(44))
    c["recipe"]["updates"]["bb_lr"] = 1e-3
    path = tmp_path / "multiview-s44.json"
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
        b.require_fit(b.config_path(44), path)
    assert b.admit(path)["status"] == "HOLD"


@needs_registration
def test_bound_review_admits_and_engineering_cannot_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "review_bindings", lambda: {"fresh": "1"})
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"schema": "hb166-review/v1", "verdict": "PASS", "independent": True,
                                "reviewer": "test", "fold2_result_review_verdict": "PASS", "bindings": {"fresh": "1"}}))
    assert b.admit(path)["status"] == "ADMITTED"
    with pytest.raises(ValueError, match="engineering"):
        b.require_fit(b.config_path(44, True), path)




def snapshot():
    return b.read(b.b155.WORK / "runtime" / "4090.json")


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
    good = dict(passed=True, smoke=True, engineering_only=True, device="cuda", seed=44, version=b.VERSION,
                config_sha256=b.sha(b.config_path(44)), gpu_name=b.GPU, peak_reserved_bytes=7 << 30, full_shape=True,
                runtime_pairing=True, arm="multiview", optimizer_steps=2,
                validation_shapes=["k24", "k36", "k36flip", "gold_k36flip"], batch=8, val_batch=4,
                healthy_rates=dict(train_step=1., validation=1., panel=1.))
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(good))
    b.check_smoke("cuda", "fit", 44, path)
    path.write_text(json.dumps({**good, field: value}))
    with pytest.raises((ValueError, KeyError)):
        b.check_smoke("cuda", "fit", 44, path)


@needs_registration
def test_cpu_smoke_requires_real_model_and_no_cuda(tmp_path):
    good = dict(passed=True, smoke=True, engineering_only=True, device="cpu", seed=45, version=b.VERSION,
                config_sha256=b.sha(b.config_path(45, True)), cpu_cuda_initialized=False, arm="multiview",
                optimizer_steps=1, validation_shapes=["k24", "k36", "k36flip", "gold_k36flip"], cpu_model="real")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(good))
    b.check_smoke("cpu", "fit", 45, path)
    for field, value in (("cpu_model", "tiny"), ("cpu_cuda_initialized", True)):
        path.write_text(json.dumps({**good, field: value}))
        with pytest.raises(ValueError):
            b.check_smoke("cpu", "fit", 45, path)


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
    from tools import hb166_fold3_watchdog_v1 as w
    calls = []
    monkeypatch.setattr(b, "disk_gate", lambda *a, **k: 1)
    monkeypatch.setattr(b, "admit", lambda review: {"status": "ADMITTED"})
    monkeypatch.setattr(b, "immutable", lambda *a, **k: None)
    monkeypatch.setattr(b, "read", lambda p: {"healthy_rates": {"train_step": 1., "validation": 1., "panel": 1.}})
    monkeypatch.setattr(b.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    result = b.sequence(mode, tmp_path / "review.json" if mode == "fit" else None, "t1")
    assert result["jobs"] == ["multiview-s44", "multiview-s45"]
    for cmd in calls:
        child = cmd[cmd.index("--") + 1:]
        module, child_mode, device = w.classify(child)
        assert child_mode == mode and device == "cuda"
        assert ("--measurement" in cmd) == (mode == "smoke") and ("--healthy-rates" in cmd) == (mode == "fit")
        assert ("--review" in child) == (mode == "fit")
        assert w.WORK in __import__("pathlib").Path(cmd[cmd.index("--attempt") + 1]).parents


"""CPU tests v1.0.0 for the H-B-166 registered gate, L6 bundle and seven-lens readout (aggregates only)."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # empty, never -1 (Windows cu121 backward crash)

import numpy as np
import pytest

from tools import hb166_fold3_build_v1 as b
from tools import hb166_fold3_readout_v1 as r

REGISTERED = (b.WORK / "registration.json").exists()


def tensors(macro=.004, spread=.001, target_shift=None, seed_offsets=(0., 0.), draws=300):
    rng = np.random.default_rng(0)
    mean = np.full((draws + 1, 4, 12), macro) + np.r_[0, rng.normal(0, spread, draws)][:, None, None]
    if target_shift is not None:
        mean[:, :, target_shift[0]] += target_shift[1]
    seeds = {s: mean + off for s, off in zip(r.SEEDS, seed_offsets)}
    return seeds, mean


def test_gate_passes_registered_signal():
    seeds, mean = tensors()
    g = r.registered_gate(seeds, mean)
    assert g["PASS"] and g["status"] == "PASS_FOLD3_CONFIRMATION" and g["ci_positive_rulers"] == 4
    assert g["not_release_authorization"] is True and "not release authorization" in g["consequence"]


def test_gate_macro_threshold():
    seeds, mean = tensors(macro=.0019, spread=.0001)
    g = r.registered_gate(seeds, mean)
    assert not g["PASS"] and not g["checks"]["four_ruler_macro"] and g["checks"]["ci_lower_positive_rulers"]


def test_gate_requires_two_rulers_with_positive_lower_bound():
    seeds, mean = tensors(macro=.003, spread=.0001)
    mean[1:, 1:, :] = mean[1:, 1:, :] - .01   # three rulers' draws cross zero; points unchanged
    seeds = {s: mean.copy() for s in r.SEEDS}
    g = r.registered_gate(seeds, mean)
    assert g["ci_positive_rulers"] == 1 and not g["PASS"] and g["checks"]["four_ruler_macro"]
    mean[1:, 1, :] += .01
    g = r.registered_gate({s: mean.copy() for s in r.SEEDS}, mean)
    assert g["ci_positive_rulers"] == 2 and g["PASS"]


def test_gate_target_floor_blocks():
    seeds, mean = tensors(macro=.004, target_shift=(6, -.0101))   # PF OA 4-ruler mean = -0.0061
    g = r.registered_gate(seeds, mean)
    assert not g["PASS"] and not g["checks"]["target_floor"] and g["worst_target"] == "PF OA"
    seeds, mean = tensors(macro=.004, target_shift=(6, -.0089))   # -0.0049 stays above the floor
    assert r.registered_gate(seeds, mean)["checks"]["target_floor"]


def test_gate_both_seeds_positive():
    seeds, mean = tensors(macro=.004, seed_offsets=(.002, -.0045))
    g = r.registered_gate(seeds, mean)
    assert not g["PASS"] and not g["checks"]["both_seeds_positive"] and g["seed_four_ruler_macro"]["45"] < 0


@pytest.mark.parametrize("mutate", ["nan", "shape", "seed"])
def test_gate_invalid_evidence(mutate):
    seeds, mean = tensors()
    if mutate == "nan":
        mean[0, 0, 0] = np.nan
    elif mutate == "shape":
        mean = mean[:, :3]
    else:
        seeds = {"44": seeds["44"]}
    assert r.registered_gate(seeds, mean)["status"] == "INVALID_EVIDENCE"


def test_engineering_never_passes():
    seeds, mean = tensors()
    g = r.registered_gate(seeds, mean, engineering=True)
    assert g["status"] == "ENGINEERING_ONLY" and not g["PASS"]


def test_prediction_pin_and_roster_enforced(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("StudyInstanceUID," + ",".join(r.lens.TARGETS) + "\na," + ",".join(["0.5"] * 12) + "\n")
    assert r.load_csv(path, ["a"], b.sha(path)).shape == (1, 12)
    with pytest.raises(ValueError, match="pin"):
        r.load_csv(path, ["a"], "0" * 64)
    with pytest.raises(ValueError, match="roster"):
        r.load_csv(path, ["b"])


def test_partner_pins_and_bundle_on_same_rows():
    val = b.read(b.b165.config_path("control", 44))["val"]
    partners = r.partner_predictions(val)
    assert sorted(partners) == ["C4", "R50", "T"] and all(p.shape == (864, 12) for p in partners.values())
    rng = np.random.default_rng(1)
    a, c = rng.uniform(size=(864, 12)), rng.uniform(size=(864, 12))
    bundle = r.l6_bundle(object(), a, c, partners)
    assert bundle["arm"].shape == (864, 12) and not np.allclose(bundle["arm"], bundle["control"])
    same = r.l6_bundle(object(), a, a, partners)
    assert np.array_equal(same["arm"], same["control"])


def test_lens_table_reports_all_seven_with_reasons():
    lenses = {k: {"value": .001, "ci95": [0, .002], "PASS": False, "eligible": True} for k in ("L1", "L2", "L3", "L5", "L6")}
    lenses["L4"] = {"value": .004, "PASS": True, "eligible": True, "selected_targets": ["Lateral OA"]}
    lenses["L7"] = {"value": {"gold58": .01}, "PASS": False, "eligible": False}
    table = r.lens_table({"lenses": lenses})
    assert list(table) == ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    assert table["L7"]["eligible"] is False and "OAI" in table["L7"]["reason"] and "Gold58" in table["L7"]["reason"]
    lenses["L6"] = {"value": None, "PASS": False, "eligible": False, "reason": "partner missing"}
    assert r.lens_table({"lenses": lenses})["L6"]["reason"]


@pytest.mark.skipif(not REGISTERED, reason="stage not registered yet")
def test_engineering_readout_end_to_end(tmp_path):
    out = tmp_path / "readout-engineering.json"
    result = r.run(None, engineering=True, out=out)
    assert result["status"] == "ENGINEERING_ONLY"
    data = b.read(out)
    assert set(data["lens_table"]) == {"L1", "L2", "L3", "L4", "L5", "L6", "L7"}
    assert data["lenses"]["L6"]["eligible"] is True and data["lenses"]["L6"]["rows"] == 864
    assert data["lenses"]["L7"]["eligible"] is False and "OAI" in data["lenses"]["L7"]["reason"]
    assert data["lenses"]["L5"]["eligible"] is True and data["exposure"] == "EXPOSED" and data["clean_oof"] is False
    assert data["registered_gate"]["PASS"] is False and data["engineering_only"] is True


"""CPU tests v1.0.0 for the H-B-166 small transfer (no network: subprocess SSH is never reached)."""
import hashlib
import json
import subprocess
import sys

import pytest

from tools import hb166_fold3_build_v1 as b
from tools import hb166_fold3_transfer_v1 as t

REGISTERED = (b.WORK / "registration.json").exists()


def test_ssh_host_and_port_validation():
    assert t.ssh_base("root@50.175.95.210", 51626)[-1] == "root@50.175.95.210"
    assert "StrictHostKeyChecking=yes" in t.ssh_base("root@h", 22) and "BatchMode=yes" in t.ssh_base("root@h", 22)
    for host in ("-oProxyCommand=x", "root@h;rm", "a b"):
        with pytest.raises(ValueError):
            t.ssh_base(host, 22)
    with pytest.raises(ValueError):
        t.ssh_base("root@h", 0)


def run_verify(root, mode, manifest):
    return subprocess.run([sys.executable, "-c", t.VERIFY, str(root), mode], input=json.dumps(manifest).encode(),
                          capture_output=True)


def entry(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def test_remote_verifier_detects_missing_tamper_and_preexisting_new_files(tmp_path):
    (tmp_path / "req").mkdir()
    (tmp_path / "req" / "a.txt").write_bytes(b"alpha")
    manifest = dict(required={"req/a.txt": entry(b"alpha")}, files={"new/b.txt": entry(b"beta")})
    pre = run_verify(tmp_path, "preflight", manifest)
    assert pre.returncode == 0 and json.loads(pre.stdout)["status"] == "OK"
    assert json.loads(run_verify(tmp_path, "verify", manifest).stdout)["missing"] == ["new/b.txt"]
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "b.txt").write_bytes(b"BETA")
    out = run_verify(tmp_path, "preflight", manifest)
    assert out.returncode == 2 and json.loads(out.stdout)["different"] == ["new/b.txt"]
    (tmp_path / "new" / "b.txt").write_bytes(b"beta")
    (tmp_path / "req" / "a.txt").write_bytes(b"alphA")
    out = json.loads(run_verify(tmp_path, "verify", manifest).stdout)
    assert out["status"] == "HOLD" and out["different"] == ["req/a.txt"]
    bad = dict(required={"../escape.txt": entry(b"x")}, files={})
    assert run_verify(tmp_path, "preflight", bad).returncode != 0


def test_result_names_must_stay_in_stage(tmp_path):
    for name in ("tools/hb166_fit_v1.py", ".local/scratch/slot-b/hb166/pilot-fold3-v110/../x.json", "../outside"):
        with pytest.raises(ValueError):
            t.check_result_names({name: {"bytes": 1, "sha256": "0"}})


@pytest.mark.skipif(not REGISTERED, reason="stage not registered yet")
def test_new_files_are_only_hb166_and_required_never_sent():
    new = t.new_files()
    assert all("hb166" in p for p in new)
    assert not any(p.endswith((".npy", ".pt")) for p in new)
    required = t.required_files(set(new))
    assert not set(required) & set(new)
    assert any(p.endswith("control-s44/final.pt") for p in required)
    assert ".local/scratch/slot-b/hb155/crop-cache-v220/side_unknown.npy" in required


def test_plan_only_commands_never_touch_network(monkeypatch):
    monkeypatch.setattr(t.subprocess, "Popen", lambda *a, **k: pytest.fail("network used"))
    monkeypatch.setattr(t.subprocess, "run", lambda *a, **k: pytest.fail("network used"))
    assert t.pull("root@h", 22, "fits", execute=False)["status"] == "PLAN_ONLY"
    assert t.main(["preflight", "--host", "root@h", "--port", "22"]) == 0


def combined_fixture(macro=.004):
    seeds, mean = tensors(macro=macro)
    return dict(registered_gate=r.registered_gate(seeds, mean),
                lens_table={"L6": {"value": .001}}, decision_label="ADVANCE")


def test_combined_preserves_failed_fold2_but_cannot_rescue_fold3():
    f2, f3 = combined_fixture(.0012), combined_fixture(.004)
    result = r.combined(f2, f3)
    assert result["support"] and result["equal_fold_macro"] == pytest.approx(.0026)
    assert result["folds"]["2"]["registered_gate"]["status"] == "NO_PASS"
    assert not r.combined(f3, f2)["support"]
    f3["registered_gate"]["target_four_ruler_mean"][r.lens.TARGETS[0]] = -.02
    assert not r.combined(f2, f3)["support"]


def test_combined_invalid_missing_l6_and_engineering():
    f2, f3 = combined_fixture(), combined_fixture()
    f2["lens_table"]["L6"]["value"] = None
    assert r.combined(f2, f3)["equal_fold_L6_point"] is None
    f3["engineering_only"] = True
    assert r.combined(f2, f3)["status"] == "ENGINEERING_ONLY"
    f3.pop("engineering_only")
    f3["registered_gate"] = {"status": "INVALID_EVIDENCE", "PASS": False}
    assert not r.combined(f2, f3)["support"]


@needs_registration
def test_actual_control_receipts_authenticate():
    report = b.verify_controls()
    assert set(report) == {"44", "45"}
    assert all(x["status"] == "PAIRED_REUSE" and x["steps"] == 8680 and x["epochs"] == 20
               and x["k"] == 36 and x["flip"] and x["gold_rows"] == 58 for x in report.values())


@needs_registration
def test_fold3_fit_entrypoints_cannot_bypass_review_or_watchdog(tmp_path, monkeypatch):
    from tools import hb166_fold3_fit_v1 as f
    monkeypatch.delenv("HB155_TOKEN", raising=False)
    for device, smoke, message in (("cpu", False, "CUDA"), ("cuda", False, "review"),
                                    ("cpu", True, "engineering"), ("cuda", True, "watchdog")):
        with pytest.raises(ValueError, match=message):
            f.fit(b.config_path(44), tmp_path / "out", device=device, smoke=smoke)
    assert not f.h.torch.cuda.is_initialized()


@needs_registration
@pytest.mark.parametrize("reviewer,prior", [("hb166-fold3-builder", "PASS"), ("independent", "FAIL"),
                                           ("independent", None)])
def test_prior_result_fail_or_self_review_blocks(tmp_path, monkeypatch, reviewer, prior):
    monkeypatch.setattr(b, "review_bindings", lambda: {"fresh": "1"})
    path = tmp_path / "review.json"
    path.write_text(json.dumps(dict(schema="hb166-review/v1", verdict="PASS", independent=True,
                         reviewer=reviewer, fold2_result_review_verdict=prior, bindings={"fresh": "1"})))
    assert b.admit(path)["status"] == "HOLD"


def test_watchdog_only_new_fit_and_policy_unchanged():
    from tools import hb166_fold3_watchdog_v1 as w
    from tools import hb155_watchdog_v1 as parent
    assert w.base.Policy is parent.Policy and w.base.supervise is parent.supervise
    with pytest.raises(ValueError):
        w.classify(["python", "-B", "-m", "tools.hb166_panel_v1", "fit", "--device", "cuda"])


def test_review_transfer_includes_bound_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "WORK", tmp_path)
    monkeypatch.setattr(b, "ROOT", tmp_path.parent)
    for name in ("review.json", "memory-admission.json", "fold2-result-review.md"):
        (tmp_path / name).write_text(json.dumps(dict(schema="hb166-review/v1", verdict="PASS", independent=True)))
    monkeypatch.setattr(b, "admit", lambda review: {"status": "ADMITTED"})
    monkeypatch.setattr(t.subprocess, "run", lambda *a, **k: pytest.fail("network used"))
    plan = t.send_review("root@h", 22)
    assert len(plan["tar"]) == 6
    assert any(x.endswith("memory-admission.json") for x in plan["tar"])
    assert any(x.endswith("fold2-result-review.md") for x in plan["tar"])


@pytest.mark.parametrize("case", ["valid", "peak-mismatch", "over-12gib", "no-headroom", "negative-used", "no-time", "prior-drift"])
def test_measured_memory_and_prior_review_binding(tmp_path, monkeypatch, case):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "WORK", tmp_path)
    monkeypatch.setattr(b, "registration_binding", lambda: {})
    monkeypatch.setattr(b, "prepare", lambda engineering=False: {"status": "PREPARED_ENGINEERING" if engineering else "PREPARED"})
    monkeypatch.setattr(b, "smoke_paths", lambda: [])
    monkeypatch.setattr(b, "require_runtime", lambda runtime: None)
    peak = 13 * (1 << 30) if case == "over-12gib" else 8 * (1 << 30)
    for name in ("prepared.json", "cpu-prepared.json"):
        (tmp_path / name).write_text("{}")
    for seed in b.SEEDS:
        directory = tmp_path / "cuda" / b.member(seed)
        directory.mkdir(parents=True)
        (directory / "runtime.json").write_text("{}")
        (directory / "summary.json").write_text(json.dumps({"peak_reserved_bytes": peak}))
    prior = tmp_path / "fold2-result-review.md"
    prior.write_text("synthetic independent result PASS fixture")
    monkeypatch.setattr(b, "FOLD2_REVIEW_SHA256", b.sha(prior))
    m = dict(schema="hb166-fold3-memory/v1", device=b.GPU, measured_at="2026-10-15T22:00:00+09:00",
             peak_reserved_bytes=peak, used_before_cuda_bytes=1 << 30, total_bytes=11 * (1 << 30))
    if case == "peak-mismatch":
        m["peak_reserved_bytes"] -= 1
    elif case == "no-headroom":
        m["total_bytes"] -= 1
    elif case == "negative-used":
        m["used_before_cuda_bytes"] = -1
    elif case == "no-time":
        m["measured_at"] = ""
    elif case == "prior-drift":
        prior.write_text("changed prior review")
    (tmp_path / "memory-admission.json").write_text(json.dumps(m))
    if case == "valid":
        bindings = b.review_bindings()
        assert bindings["memory-admission.json"] == b.sha(tmp_path / "memory-admission.json")
        assert bindings["fold2-result-review.md"] == b.FOLD2_REVIEW_SHA256
    else:
        with pytest.raises(ValueError, match="memory|headroom|review drift"):
            b.review_bindings()
