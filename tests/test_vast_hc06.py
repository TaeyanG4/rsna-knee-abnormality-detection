"""CPU tests of the portable pair's scientific and execution boundaries."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import vast_hc06 as v


@pytest.mark.parametrize("name", ["../secret", "/root/key", "C:/secret", "C:\\secret", "//host/share", "a/../../b", "a//b", "a/./b", ""])
def test_cross_platform_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError): v.contained(tmp_path, name)


@pytest.mark.parametrize("name", ["storage/competition/train.csv", "storage/downloads/data.zip", ".ssh/id_rsa", "kaggle.json",
                                  "artifacts/experiments/EXP084_B_G1/model_best.pt", "storage/gold/gold.csv"])
def test_sensitive_or_unrelated_assets_never_allowlisted(name):
    assert not v.allowed_file(name)


def test_trainer_diff_changes_only_guard_import():
    source = (ROOT / v.TEMPLATE).read_text(encoding="utf-8")
    assert v.render_runner().replace("from tools import vast_hc06 as guard", "from tools import patch_mil as guard") == source
    assert v.FROZEN_ARGS == v.read(ROOT / v.PARENT)["arm"]["expected_args"]


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    # Copies only ~90MB small assets. No GPU, network, shards, original changes.
    directory = tmp_path_factory.mktemp("hc06") / "package"
    v.prepare(directory)
    return directory


def test_real_package_is_small_allowlisted_and_shards_are_declared_only(package):
    p, ids = v.validate_manifest(package / "manifest.json", package, allow_missing_shards=True)
    assert len(ids["train"]) == 3455 and len(ids["val"]) == 888
    assert sum(x.stat().st_size for x in package.rglob("*") if x.is_file()) < 100_000_000
    assert len([n for n in p["files"] if "vols_shard" in n]) == 12
    assert all(not (package / n).exists() for n in p["files"] if "vols_shard" in n)
    assert p["arms"]["CLS"]["expected_args"] == p["arms"]["PATCH"]["expected_args"] == v.FROZEN_ARGS
    assert "legacy_roots" not in (package / "config/workspace.toml").read_text()
    with pytest.raises(ValueError, match="missing/size"):
        v.validate_manifest(package / "manifest.json", package, full=True)


@pytest.mark.parametrize("change", ["recipe", "budget", "unlisted", "traversal", "missing"])
def test_manifest_rejects_scientific_or_inventory_drift(package, tmp_path, change):
    p = v.read(package / "manifest.json")
    if change == "recipe": p["arms"]["CLS"]["expected_args"]["seed"] = 43
    if change == "budget": p["execution"]["full_seconds"] += 1
    if change == "unlisted": p["files"]["kaggle.json"] = {"bytes": 10, "sha256": "a" * 64}
    if change == "traversal": p["files"]["../secret"] = {"bytes": 10, "sha256": "a" * 64}
    if change == "missing": p["files"].pop(v.WEIGHT)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError): v.validate_manifest(path, package, allow_missing_shards=True)


def good_pilot():
    return {"step_seconds": [.2] * 32, "eval_seconds": .2, "initial_seconds": 3.0}


def test_projection_uses_only_last24_and_measured_eval_overhead():
    pilot = good_pilot()
    pilot["step_seconds"][:8] = [1.0] * 8
    assert v.projection(pilot) == pytest.approx(1.3 * .2 * 3448 + 300)
    pilot["eval_seconds"] = 8.0
    assert v.projection(pilot) == pytest.approx(1.3 * .2 * 3448 + 1.3 * 8 * 888 / 4 + 33)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1, "0.2"])
def test_projection_rejects_invalid_measurements(bad):
    pilot = good_pilot(); pilot["step_seconds"][10] = bad
    with pytest.raises(ValueError): v.projection(pilot)


def pilot_receipt(path, arm, command):
    receipt = {**good_pilot(), "schema": v.SCHEMA, "status": "PASS", "phase": "pilot", "arm": arm["id"], "argv": command,
               "protocol_sha256": v.sha256(path), "initial_state_sha256": v.INITIAL_SHA, "initial_asset_sha256": v.WEIGHT_SHA,
               "position_policy": v.POSITION_POLICY, "aggregation_policy": arm["aggregation_policy"], "updates": 32, "skips": 0,
               "training_ids_only": True, "validation_rows": 0, "eval_train_studies": 4, "metric_access": "disabled",
               "checkpoints_saved": False, "predictions_saved": False, "peak_gpu_allocated_bytes": 2**30}
    receipt["projected_full_seconds"] = v.projection(receipt)
    return receipt


@pytest.mark.parametrize("key,value", [("skips", 1), ("updates", 31), ("validation_rows", 4), ("initial_state_sha256", "wrong"),
                                      ("aggregation_policy", v.POLICIES["PATCH"]), ("predictions_saved", True), ("peak_gpu_allocated_bytes", 24 * 2**30)])
def test_pilot_gate_rejects_unsafe_receipts(package, tmp_path, key, value):
    path = package / "manifest.json"
    p = v.read(path); arm = p["arms"]["CLS"]
    command = ["python", "test"]
    receipt = pilot_receipt(path, arm, command)
    v.validate_result(receipt, p, path, arm, "pilot", command, tmp_path)
    receipt[key] = value
    with pytest.raises(ValueError): v.validate_result(receipt, p, path, arm, "pilot", command, tmp_path)


def test_failed_second_pilot_prevents_all_full_fits_and_retry(package, tmp_path, monkeypatch):
    from tools import process_resource_receipt as supervisor
    from tools import exp086_dinov2s_v2 as base
    p = v.read(package / "manifest.json")
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(p))
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"status": "READY", "allow_execution": True, "manifest_sha256": v.sha256(path)}))
    (tmp_path / "launch.json").write_text(json.dumps({"status": "RUNNING", "manifest_sha256": v.sha256(path), "admission_sha256": v.sha256(admission)}))
    (tmp_path / "preflight.json").write_text(json.dumps({"environment": {}}))
    monkeypatch.setattr(v, "ROOT", tmp_path)
    monkeypatch.setattr(v, "validate_manifest", lambda *_a, **_k: (p, {}))
    monkeypatch.setattr(v, "check_preflight", lambda *_a: {})
    monkeypatch.setattr(v, "remote_environment", lambda: {})
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("RAM", (), {"available": 32 * 2**30})())
    monkeypatch.setattr(v.shutil, "disk_usage", lambda _p: type("Disk", (), {"free": 64 * 2**30})())
    calls = []
    def supervise(command, output, timeout, cwd):
        calls.append(command)
        output.mkdir(parents=True)
        (output / "receipt.json").write_text("{}")
        if len(calls) == 2: return {"status": "FAILED"}
        runout = output.with_name(output.name.removesuffix("_RESOURCE")); runout.mkdir()
        receipt = pilot_receipt(path, p["arms"]["CLS"], command)
        (runout / "receipt.json").write_text(json.dumps(receipt))
        return {"status": "COMPLETE"}
    monkeypatch.setattr(supervisor, "supervise", supervise)
    monkeypatch.setattr(base, "resource_gate", lambda result, *_a: v.need(result["status"] == "COMPLETE", "worker failed"))
    assert v.execute_pair(path, admission) == 1
    assert len(calls) == 2 and all(c[c.index("--phase") + 1] == "pilot" for c in calls)
    assert v.read(tmp_path / "INVALID.json")["failure"] == "ValueError: worker failed"
    assert not (tmp_path / "PAIR_VALIDATED.json").exists()
    with pytest.raises(FileExistsError): v.execute_pair(path, admission)
    assert len(calls) == 2


def test_auc_handles_ties_and_preserves_undefined_targets():
    import numpy as np
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 12))
    tied = np.ones((4, 12)) * .5
    assert v.auc12(y, tied) == [.5] * 12
    perfect = y.astype(float)
    assert v.auc12(y, perfect) == [1.0] * 12
    assert v.auc12(y, 1 - perfect) == [0.0] * 12
    y[:, 0] = 1
    assert v.auc12(y, tied)[0] is None


def test_score_refuses_without_outer_pair_record(package, monkeypatch):
    monkeypatch.setattr(v, "ROOT", package)
    monkeypatch.setattr(v, "validate_manifest", lambda *_a: (v.read(package / "manifest.json"), {}))
    with pytest.raises(FileNotFoundError): v.score(package / "manifest.json")


def test_outer_timeout_never_emits_validated_success(package, tmp_path, monkeypatch):
    from tools import process_resource_receipt as supervisor
    from tools import exp086_dinov2s_v2 as base
    p = v.read(package / "manifest.json")
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(p))
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"status": "READY", "allow_execution": True, "manifest_sha256": v.sha256(path)}))
    monkeypatch.setattr(v, "ROOT", tmp_path)
    monkeypatch.setattr(v, "validate_manifest", lambda *_a, **_k: (p, {}))
    monkeypatch.setattr(v, "check_preflight", lambda *_a: {})
    calls = []
    def timeout(command, output, limit, cwd):
        calls.append((command, limit))
        output.mkdir()
        (output / "receipt.json").write_text('{"status":"FAILED"}')
        return {"status": "FAILED"}
    monkeypatch.setattr(supervisor, "supervise", timeout)
    monkeypatch.setattr(base, "resource_gate", lambda *_a: v.need(False, "outer timeout"))
    assert v.run(path, admission) == 1
    assert calls[0][1] == 7200
    assert "_pair" in calls[0][0]
    assert not (tmp_path / "VALIDATED_SUCCESS.json").exists()
    assert v.read(tmp_path / "OUTER_INVALID.json")["valid_for_scoring"] is False
    with pytest.raises(FileExistsError): v.run(path, admission)
    assert len(calls) == 1


def test_unchanged_queue_routes_to_gpu_job_then_outer_wrapper(package, tmp_path, monkeypatch):
    # Exercise the real queue dispatch code with a no-process launcher stub.
    # Native Linux subprocess/cleanup behavior still requires remote preflight.
    import types
    fake_common = types.ModuleType("raptor_common")
    fake_common.ROOT, fake_common.LOGS = package, tmp_path / "logs"
    monkeypatch.setitem(sys.modules, "raptor_common", fake_common)
    spec = importlib.util.spec_from_file_location("hc06_queue_test", ROOT / "tools/gpu_queue.py")
    queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        marker = fake_common.LOGS / v.QUEUE[0]["tag"] / "SUCCESS"
        marker.parent.mkdir(parents=True); marker.write_text("exit0")
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue.subprocess, "run", launch)
    assert queue.run_plan(package / "queue.json") == 0
    assert len(calls) == 1 and calls[0][0][1] == str(package / "tools/gpu_job.py")
    assert calls[0][0][-len(v.QUEUE[0]["command"]):] == v.QUEUE[0]["command"]
    assert "training/runners/vast_hc06.py" not in calls[0][0]
