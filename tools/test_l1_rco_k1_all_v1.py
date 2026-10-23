"""All-data CPU regressions; synthetic completion receipts never certify a fit."""
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tools import l1_rco_k1_all_v1 as k
from tools import l1_rco_k1_allseq_v1 as q
from tools import k1_fold_dataset_v1 as assets


def test_registration_roster_recipe_and_labels():
    r = k.expected_registration()
    p = k.read(k.PARENT)
    assert r["train"] == sorted(p["train"] + p["holdout"])
    assert len(set(r["train"])) == 4167 and r["holdout"] == []
    changed = {"model_fold", "only_folds", "steps_per_epoch", "total_steps",
               "warm_steps", "bias", "unchanged"}
    assert {n: v for n, v in r["recipe"].items() if n not in changed} == {
        n: v for n, v in p["recipe"].items() if n not in changed}
    assert r["recipe"]["total_steps"] == 20830
    assert r["recipe"]["warm_steps"] == 2083
    assert r["optimizer_policy"] == p["optimizer_policy"]
    assert r["selection"]["checkpoint"] == "final EMA"
    assert not r["selection"]["early_stopping"]
    assert not r["evaluator"]["clean_oof"]
    assert len(k.training_rows(r)) == 4167
    assert len(k.guard_ids(r)) == 32
    assert k.guard_ids(r) == k.guard_ids(dict(train=list(reversed(r["train"]))))


def test_exact_scientific_source_delta():
    source = (k.ROOT / k.s.SOURCE / "compact_train.py").read_text(encoding="utf-8")
    original = k.parent.derive_trainer(source)
    actual = k.derive_trainer(source)
    restored = actual.replace(
        "assert not gold  # all labelled studies; no held-out fold",
        "assert not gold and all(pool_fold != 0)")
    restored = restored.replace("tools.l1_rco_k1_allstate_v1", "tools.l1_rco_k1_state_v2")
    restored = restored.replace("guard_constant=a.l1_guard_constant, fold='all')",
                                "guard_constant=a.l1_guard_constant)")
    restored = restored.replace("tools.l1_rco_k1_all_v1", "tools.l1_rco_k1_v3")
    restored = restored.replace("            tr = np.arange(len(pool))  # all labelled studies",
        "            tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)")
    assert restored == original
    assert "l1_state.finish(model)" in actual


@pytest.mark.parametrize("fault", ["constant", "one_constant", "nan", "range", "shape", "one"])
def test_prediction_guard_fail_closed(fault):
    values = np.linspace(.1, .9, 32)[:, None] * np.ones((1, 12))
    if fault == "constant": values[:] = .5
    if fault == "one_constant": values[:, 3] = .5
    if fault == "nan": values[0, 0] = np.nan
    if fault == "range": values[0, 0] = 1.1
    if fault == "shape": values = values[:, :11]
    if fault == "one": values = values[:1]
    with pytest.raises(ValueError):
        k.prediction_guard(values)


def synthetic_completion(root, monkeypatch):
    """Real source transform; synthetic checkpoint/receipts/predictions only."""
    monkeypatch.setattr(k, "WORK", root)
    monkeypatch.setattr(k, "OUT", root / "fit")
    out = k.OUT
    model = out / "models/E_prior_s0_fall.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"SYNTHETIC NOT A TRAINED CHECKPOINT")
    reg = dict(train=[f"synthetic-{i}" for i in range(4167)], holdout=[], model_fold="all",
               recipe=dict(model_fold="all", epochs=10, total_steps=20830),
               collapse_guard=dict(constant_loss=1.),
               selection=dict(epochs=10, early_stopping=False, holdout_selection=False),
               test_fixture=True)
    reg["prediction_guard"] = dict(ids=k.guard_ids(reg), min_target_range=1e-6)
    reg["contract_id"] = k.digest(reg)
    k.write(root / "registration.json", reg)
    training = k.derive_trainer(assets.SERVING.read_text(encoding="utf-8"))
    k.write(root / "prepare.json", dict(contract_id=reg["contract_id"],
            trainer_sha256=hashlib.sha256(training.encode()).hexdigest(), test_fixture=True))
    k.write(out / "fit-complete.json", dict(status="COMPLETE", contract_id=reg["contract_id"],
        model_fold="all", model_sha256=k.sha(model), epochs=10, optimizer_steps=20830,
        train_studies=4167, selection="final EMA", own_holdout_evaluation=False, test_fixture=True))
    k.write(out / k.EPOCHS, dict(committed_epochs=10, log=[.4]*10))
    with (out / "train-guard.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow([k.s.UID] + k.s.LABELS + ["status", "model_fold"])
        for i, uid in enumerate(reg["prediction_guard"]["ids"]):
            w.writerow([uid] + [.1 + i*.01]*12 + ["ok", "all"])
    guard = k.verify_prediction_guard(reg, out)
    k.write(out / "result.json", dict(status="COMPLETE", contract_id=reg["contract_id"],
        model_fold="all", model_sha256=k.sha(model), holdout_rows=0, training_guard_rows=32,
        training_guard_sha256=k.sha(out / "train-guard.csv"), prediction_guard=guard,
        cpu_test=False, clean_oof=False, own_holdout_evaluation=False,
        selection="final EMA", exposure="EXPOSED_EXPLORATORY", test_fixture=True))
    k.write(out / "runtime.json", dict(test_fixture=True))
    return reg


def test_export_and_asset_roundtrip(tmp_path, monkeypatch):
    reg = synthetic_completion(tmp_path / "export", monkeypatch)
    assert k.export(reg)["files"] == 7
    export = k.WORK / "EXPORT.json"
    assert assets.verify_export(export)[0]["fold"] == "all"
    dataset = tmp_path / "dataset"
    model = assets.package(export, dataset, "test-owner/knee-all-cpu-test")
    assert model["id"] == "k1_v3g_all"
    assert model == assets.verify_dataset(dataset)
    meta = k.read(dataset / "asset-manifest.json")
    assert not meta["clean_oof"] and not meta["own_holdout_evaluation"]
    assert meta["selection"] == "final EMA"


@pytest.mark.parametrize("field,value", [
    ("optimizer_steps", 20829), ("epochs", 9), ("selection", "best"),
    ("own_holdout_evaluation", True), ("train_studies", 4166)])
def test_incomplete_fit_no_export(tmp_path, monkeypatch, field, value):
    reg = synthetic_completion(tmp_path, monkeypatch)
    path = k.OUT / "fit-complete.json"
    receipt = k.read(path); receipt[field] = value; k.write(path, receipt)
    with pytest.raises(ValueError):
        k.export(reg)
    assert not (tmp_path / "EXPORT.json").exists()


@pytest.mark.parametrize("fault", ["constant", "cpu", "selection", "guard_hash"])
def test_invalid_result_cannot_export(tmp_path, monkeypatch, fault):
    reg = synthetic_completion(tmp_path, monkeypatch)
    path = k.OUT / "result.json"
    result = k.read(path)
    if fault == "constant":
        guardpath = k.OUT / "train-guard.csv"
        with guardpath.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        for row in rows[1:]: row[1:13] = [".5"] * 12
        with guardpath.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
    elif fault == "cpu": result["cpu_test"] = True
    elif fault == "selection": result["selection"] = "best"
    else: result["training_guard_sha256"] = "0"*64
    k.write(path, result)
    with pytest.raises(ValueError):
        k.export(reg)


def test_missing_review_blocks_cpu_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "WORK", tmp_path)
    monkeypatch.setattr(k, "validate", lambda r: r)
    monkeypatch.setattr(k, "complete_cache", lambda r, c: {"studies": 4167})
    assert k.admit({}, tmp_path, tmp_path / "missing")["status"] == "BLOCKED"


@pytest.mark.parametrize("state,bound", [("running", "bound"), ("FAILED", "bound"),
                                       ("SUCCESS", "legacy")])
def test_sequence_requires_bound_f2_success(monkeypatch, state, bound):
    monkeypatch.setattr(q.seq.runtime, "job_status", lambda *a, **kw: dict(state=state, receipt_kind=bound))
    with pytest.raises(ValueError, match="f2 bound SUCCESS"):
        q.preflight(q.plan())


def test_sequence_command_and_one_attempt(tmp_path, monkeypatch):
    spec = q.plan()
    assert len(spec["jobs"]) == 1
    command = spec["jobs"][0]["argv"]
    assert command[command.index("--")+1:] == k.command()
    assert spec["owner"] == str(k.BASE / "gpu0-owner")
    spec["owner"] = str(tmp_path / "owner")
    spec["ledger"] = str(tmp_path / "ledger.json")
    monkeypatch.setattr(k, "OUT", tmp_path / "fit")
    f2 = q.folds.command(fold=2, gpu=0)
    monkeypatch.setattr(q.seq.runtime, "job_status", lambda *a, **kw: dict(state="SUCCESS", receipt_kind="bound"))
    monkeypatch.setattr(q.seq.runtime, "read_json", lambda *a: dict(
        cwd=spec["owner"], command=f2,
        command_sha256=q.seq.runtime.command_digest(f2, spec["owner"])))
    monkeypatch.setattr(q.seq.runtime, "active_jobs", lambda *a: [])
    q.preflight(spec)
    Path(spec["ledger"]).write_text("{}")
    with pytest.raises(ValueError, match="no retry"):
        q.preflight(spec)
    Path(spec["ledger"]).unlink()
    k.write(k.OUT / "admission.json", {})
    with pytest.raises(ValueError, match="admission consumed"):
        q.preflight(spec)


def test_real_cpu_checkpoint_all_packaging_cli(tmp_path, monkeypatch):
    """CPU Net bytes + explicitly synthetic completion receipts, never training evidence."""
    real = k.ALL_BASE / "all/cpu/full-net/models/E_prior_s0_fall.pt"
    assert real.is_file(), "Run the real-data all CPU smoke before this local integration test"
    reg = synthetic_completion(tmp_path / "export", monkeypatch)
    target = k.OUT / "models/E_prior_s0_fall.pt"
    shutil.copyfile(real, target)
    for name in ("fit-complete.json", "result.json"):
        path = k.OUT / name
        receipt = k.read(path)
        receipt["model_sha256"] = k.sha(target)
        k.write(path, receipt)
    k.export(reg)
    export = k.WORK / "EXPORT.json"
    output = tmp_path / "dataset"
    for args in [
        ["verify-export", "--export", str(export)],
        ["package", "--export", str(export), "--output", str(output),
         "--dataset-id", "test-owner/knee-all-real-cpu-bytes"],
        ["verify-dataset", "--dataset", str(output)],
    ]:
        run = subprocess.run([sys.executable, "-B", "-m", "tools.k1_fold_dataset_v1", *args],
                             capture_output=True, text=True, cwd=k.ROOT, timeout=60)
        assert run.returncode == 0, run.stderr
    assert assets.verify_dataset(output)["checkpoint_sha256"] == k.sha(real)


def test_portable_allowlist_and_inventory_drift(tmp_path):
    from tools import l1_rco_k1_foldpack_v3 as pack
    paths = pack.files()
    assert "tools/l1_rco_k1_allseq_v1.py" in paths
    assert "tools/k1_fold_dataset_v1.py" in paths
    assert not any("/f1/" in name for name in paths)
    (tmp_path / "sample").write_bytes(b"fixed")
    k.write(tmp_path / "INVENTORY.json", dict(files=[
        dict(path="sample", bytes=5, sha256=k.sha(tmp_path / "sample"))]))
    assert pack.verify(tmp_path)["files"] == 1
    (tmp_path / "sample").write_bytes(b"drift")
    with pytest.raises(ValueError, match="drift"):
        pack.verify(tmp_path)
