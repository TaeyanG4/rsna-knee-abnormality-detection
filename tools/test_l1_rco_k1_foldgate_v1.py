"""Synthetic CPU health-gate tests; no training, real labels or network."""
import csv
import copy
import json

import numpy as np
import pytest

from tools import l1_rco_k1_foldgate_v1 as gate
from tools import exp156_k_sweep as rulers


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "rosters", lambda: {f: ["a", "b"] for f in range(4)})
    monkeypatch.setattr(gate.k, "PARENT", tmp_path / "parent.json")
    gate.k.write(gate.k.PARENT, dict(contract_id="synthetic"))
    monkeypatch.setattr(gate.k, "configure", lambda f: None)
    monkeypatch.setattr(gate.k, "expected_registration", lambda: dict(contract_id="synthetic"))
    targets = np.array([[0] * 12, [1] * 12])
    refs = {r: (targets.copy(), np.ones_like(targets, dtype=bool), 1) for r in gate.RULERS}
    monkeypatch.setattr(rulers, "references", lambda *a, **kw: refs)
    manifest = dict(schema="h224-k1-only-input/v1", folds={})
    for fold in range(4):
        path = tmp_path / f"f{fold}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([gate.k.s.UID, *gate.k.s.LABELS, "status", "model_fold"])
            writer.writerow(["a", *([.2] * 12), "ok", fold])
            writer.writerow(["b", *([.8] * 12), "ok", fold])
        receipt = tmp_path / f"f{fold}.json"
        gate.k.write(receipt, dict(status="COMPLETE", cpu_test=False, model_fold=fold,
                                  holdout_rows=2, holdout_sha256=gate.k.sha(path),
                                  contract_id="synthetic"))
        manifest["folds"][str(fold)] = dict(k1_csv=path.name, k1_result=receipt.name)
    return tmp_path, manifest, refs


@pytest.mark.parametrize("value,expected", [
    (.895, "HEALTH_PASS"), (.99, "HEALTH_PASS"), (.89499, "HOLD")])
def test_one_sided_boundary_and_better(value, expected):
    metrics = {f: {r: .90 for r in gate.RULERS} for f in range(4)}
    metrics[2]["dread"] = value
    assert gate.decide_k1(metrics)["status"] == expected


def test_partial_fold_decision():
    metrics = {f: {r: .9 for r in gate.RULERS} for f in range(3)}
    result = gate.decide_k1(metrics)
    assert result["status"] == "HEALTH_PASS"
    assert result["folds"]["3"] == "PENDING"
    assert result["pending_folds"] == [3]
    assert not result["all_folds_evaluated"]
    assert gate.decide_k1({0: metrics[0]})["status"] == "PENDING"


@pytest.mark.parametrize("mode", ["nan", "missing", "range", "no_baseline"])
def test_invalid_metrics(mode):
    metrics = {f: {r: .9 for r in gate.RULERS} for f in range(4)}
    if mode == "nan":
        metrics[1]["dread"] = float("nan")
    elif mode == "missing":
        del metrics[2]["consensus"]
    elif mode == "range":
        metrics[3]["aquila_v1"] = 1.01
    else:
        del metrics[0]
    with pytest.raises(ValueError):
        gate.decide_k1(metrics)


def test_complete_and_no_r7_reads(rig, monkeypatch):
    root, manifest, _ = rig
    original = gate.k.read
    read_paths = []
    def read(path):
        read_paths.append(str(path))
        return original(path)
    monkeypatch.setattr(gate.k, "read", read)
    result = gate.evaluate_k1(manifest)
    assert result["status"] == "HEALTH_PASS"
    assert result["all_folds_evaluated"]
    assert len(result["metrics"]) == 4
    assert all(v == 1 for m in result["metrics"].values() for v in m.values())
    assert result["r7_complement_comparison"]["status"] == "NOT_RUN"
    assert set(read_paths) == {str(root / "parent.json"),
                               *(str(root / f"f{f}.json") for f in range(4))}
    assert result["evidence"]["1"]["files"]["k1_csv"] == gate.k.sha(root / "f1.csv")


@pytest.mark.parametrize("mode", ["omitted", "absent", "partial", "missing_f0"])
def test_missing_exports(rig, mode):
    root, manifest, _ = rig
    if mode == "omitted":
        del manifest["folds"]["3"]
    elif mode == "absent":
        manifest["folds"]["3"] = dict(k1_csv="absent.csv", k1_result="absent.json")
    elif mode == "partial":
        manifest["folds"]["3"]["k1_result"] = "absent.json"
    else:
        manifest["folds"]["0"]["k1_result"] = "absent.json"
    result = gate.evaluate_k1(manifest)
    assert result["status"] == ("HEALTH_PASS" if mode in ("omitted", "absent") else "HOLD")
    if mode in ("omitted", "absent"):
        assert result["folds"]["3"] == "PENDING"
    else:
        assert result["errors"]
        assert not result["all_folds_evaluated"]


@pytest.mark.parametrize("spec", [None, {}, {"r7_csv": "forbidden.csv"}])
def test_invalid_fold_spec_is_not_pending(rig, spec):
    _, manifest, _ = rig
    manifest["folds"]["3"] = spec
    result = gate.evaluate_k1(manifest)
    assert result["status"] == "HOLD"
    assert result["folds"]["3"] == "HOLD"


@pytest.mark.parametrize("field,value", [
    ("status", "RUNNING"), ("cpu_test", True), ("cpu_test", 0),
    ("model_fold", 0), ("holdout_rows", 1), ("holdout_sha256", "wrong"),
    ("contract_id", "wrong")])
def test_provenance_fail_closed(rig, field, value):
    root, manifest, _ = rig
    path = root / "f1.json"
    receipt = gate.k.read(path)
    receipt[field] = value
    gate.k.write(path, receipt)
    result = gate.evaluate_k1(manifest)
    assert result["status"] == "HOLD"
    assert result["folds"]["1"] == "HOLD"
    assert 1 not in result["pending_folds"]


@pytest.mark.parametrize("mode", [
    "duplicate", "missing", "extra", "fold", "status", "nan", "range", "text",
    "short", "header"])
def test_invalid_predictions(rig, mode):
    root, manifest, _ = rig
    path = root / "f2.csv"
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if mode == "duplicate":
        rows[2][0] = rows[1][0]
    elif mode == "missing":
        rows.pop()
    elif mode == "extra":
        rows.append(rows[1])
    elif mode == "fold":
        rows[1][-1] = "1"
    elif mode == "status":
        rows[1][-2] = "invalid"
    elif mode == "short":
        rows[1].pop()
    elif mode == "header":
        rows[0][1] = rows[0][2]
    else:
        rows[1][1] = {"nan": "nan", "range": "1.01", "text": "PRIVATE_ROW_SENTINEL"}[mode]
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    receipt = gate.k.read(root / "f2.json")
    receipt["holdout_sha256"] = gate.k.sha(path)
    gate.k.write(root / "f2.json", receipt)
    result = gate.evaluate_k1(manifest)
    assert result["status"] == "HOLD"
    assert result["folds"]["2"] == "HOLD"
    assert "PRIVATE_ROW_SENTINEL" not in json.dumps(result)


@pytest.mark.parametrize("mode", ["one_class", "empty_mask", "nan", "shape", "mask_type"])
def test_invalid_or_undefined_ruler(rig, mode):
    _, manifest, refs = rig
    targets, mask, _ = refs["consensus"]
    if mode == "one_class":
        targets[:, 0] = 0
    elif mode == "empty_mask":
        mask[:, 0] = False
    elif mode == "nan":
        targets = targets.astype(float)
        targets[0, 0] = np.nan
    elif mode == "shape":
        targets = targets[:, :11]
    else:
        mask = mask.astype(int)
    refs["consensus"] = targets, mask, 1
    assert gate.evaluate_k1(manifest)["status"] == "HOLD"


def test_cli_default_paths_and_partial_output(rig, monkeypatch, capsys):
    root, manifest, _ = rig
    del manifest["folds"]["3"]
    monkeypatch.setattr(gate, "default_k1_manifest", lambda: manifest)
    monkeypatch.setattr(gate, "K1_OUT", root / "health.json")
    monkeypatch.setattr(gate.sys, "argv", ["gate", "--k1-only"])
    assert gate.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "HEALTH_PASS"
    assert gate.k.read(root / "health.json") == result


def test_cli_bad_manifest_retains_not_run(rig, monkeypatch, capsys):
    root, _, _ = rig
    gate.k.write(root / "bad.json", dict(schema="wrong", folds={}))
    monkeypatch.setattr(gate.sys, "argv", ["gate", "--k1-only", "--manifest",
        str(root / "bad.json"), "--output", str(root / "health.json")])
    assert gate.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["r7_complement_comparison"]["status"] == "NOT_RUN"


def test_cli_k1_dry_run_never_evaluates(rig, monkeypatch, capsys):
    root, _, _ = rig
    monkeypatch.setattr(gate, "dry_run", lambda: dict(status="REFERENCE_COVERAGE_ONLY",
                                                   gate_evaluated=False))
    monkeypatch.setattr(gate, "evaluate_k1", lambda _: pytest.fail("must not evaluate"))
    monkeypatch.setattr(gate.sys, "argv", ["gate", "--k1-only", "--dry-run",
                                         "--output", str(root / "coverage.json")])
    assert gate.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["gate_evaluated"]
    assert result["r7_complement_comparison"]["status"] == "NOT_RUN"


@pytest.fixture
def host_export(rig, monkeypatch):
    root, manifest, _ = rig
    local = dict(train=["train-a", "train-b"], holdout=["a", "b"],
                 recipe=dict(seed=0, batch=2), model_fold=2,
                 pins={"targets": "source-hash"}, code_pins={"trainer": "code-hash"},
                 collapse_guard=dict(constant_loss=.53, threshold=.528, margin=.002))
    local["contract_id"] = gate.k.digest(local)
    reg = copy.deepcopy(local)
    for field in gate.HOST_FLOAT_FIELDS:
        reg["collapse_guard"][field] += 4e-15
    base = root / "export"
    (base / "fit").mkdir(parents=True)
    (base / "fit/holdout.csv").write_bytes((root / "f2.csv").read_bytes())
    receipt = gate.k.read(root / "f2.json")
    local_argv = ["C:/python.exe", "C:/work/trainer.py", "--px", "C:/cache",
                  "--targets", "C:/work/targets.csv", "--out", "C:/work/fit",
                  "--init", "C:/weights/init.pt", "--batch", "2",
                  "--l1-guard-constant", repr(local["collapse_guard"]["constant_loss"])]
    monkeypatch.setattr(gate.k, "CACHE", gate.Path("C:/cache"))
    monkeypatch.setattr(gate.k, "trainer_argv", lambda *a: local_argv.copy())
    monkeypatch.setattr(gate.k, "expected_registration", lambda: local)
    hashes = dict(targets_sha256="prepared-target-hash", trainer_sha256="prepared-trainer-hash")
    monkeypatch.setattr(gate, "prepared_hashes", lambda _: hashes)
    argv = [v.replace("C:/", "/linux/") for v in local_argv]
    argv[-1] = repr(reg["collapse_guard"]["constant_loss"])
    prep = dict(hashes, train_count=2, holdout_training_rows=0,
                argv=argv, cache="/linux/cache")

    def seal():
        reg.pop("contract_id", None)
        reg["contract_id"] = gate.k.digest(reg)
        receipt["contract_id"] = prep["contract_id"] = reg["contract_id"]
        gate.k.write(base / "registration.json", reg)
        gate.k.write(base / "prepare.json", prep)
        gate.k.write(base / "fit/result.json", receipt)
        names = ("registration.json", "prepare.json", "fit/result.json", "fit/holdout.csv")
        gate.k.write(base / "EXPORT.json", dict(fold=2, contract_id=reg["contract_id"],
                     sha256={n: gate.k.sha(base / n) for n in names}))
    seal()
    manifest["folds"] = {n: spec for n, spec in manifest["folds"].items() if n in ("0", "2")}
    manifest["folds"]["2"] = dict(k1_csv="export/fit/holdout.csv",
                                 k1_result="export/fit/result.json")
    return root, manifest, local, reg, prep, receipt, seal


def test_host_bound_export_and_exact_baseline(host_export):
    _, manifest, _, _, _, _, _ = host_export
    result = gate.evaluate_k1(manifest)
    assert result["status"] == "HEALTH_PASS"
    assert result["pending_folds"] == [1, 3]
    evidence = result["evidence"]["2"]["contract"]
    assert evidence["mode"] == "HOST_BOUND_EXPORT"
    assert set(evidence["differing_fields"]) == {
        "registration.collapse_guard.constant_loss", "registration.collapse_guard.threshold",
        "prepare.argv.python", "prepare.argv.trainer", "prepare.argv.--px",
        "prepare.argv.--targets", "prepare.argv.--out", "prepare.argv.--init",
        "prepare.argv.--l1-guard-constant", "prepare.cache"}
    assert len(evidence["files"]) == 4
    assert result["evidence"]["0"]["contract"]["mode"] == "EXACT_LOCAL"


@pytest.mark.parametrize("change", [
    "train", "holdout", "recipe", "pins", "code_pins", "unknown", "guard_unknown",
    "constant_loss", "threshold", "trainer_hash", "targets_hash",
    "batch", "guard_argv", "argv_extra", "relative_path", "cache",
    "train_count", "holdout_training_rows", "prepare_unknown",
])
def test_host_scientific_changes_fail_even_resealed(host_export, change):
    _, manifest, _, reg, prep, _, seal = host_export
    if change in ("train", "holdout"):
        reg[change].reverse()
    elif change in ("recipe", "pins", "code_pins"):
        reg[change]["unexpected"] = "PRIVATE_ROW_SENTINEL"
    elif change == "unknown":
        reg["unexpected"] = 1
    elif change == "guard_unknown":
        reg["collapse_guard"]["margin"] += 1e-15
    elif change in gate.HOST_FLOAT_FIELDS:
        reg["collapse_guard"][change] += 2e-12
    elif change in ("trainer_hash", "targets_hash"):
        prep[change.replace("_hash", "_sha256")] = "wrong"
    elif change == "batch":
        prep["argv"][prep["argv"].index("--batch") + 1] = "4"
    elif change == "guard_argv":
        prep["argv"][-1] = "0.1"
    elif change == "argv_extra":
        prep["argv"].append("--unexpected")
    elif change == "relative_path":
        prep["argv"][0] = "python"
    elif change == "cache":
        prep["cache"] = "/different"
    elif change in ("train_count", "holdout_training_rows"):
        prep[change] += 1
    else:
        prep["unknown"] = 1
    seal()
    result = gate.evaluate_k1(manifest)
    assert result["status"] == "HOLD"
    assert result["folds"]["2"] == "HOLD"
    assert "PRIVATE_ROW_SENTINEL" not in json.dumps(result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.53"])
def test_host_collapse_invalid_types_and_nonfinite(host_export, value):
    root, manifest, _, reg, prep, receipt, _ = host_export
    reg["collapse_guard"]["constant_loss"] = value
    reg.pop("contract_id")
    reg["contract_id"] = gate.k.digest(reg)
    prep["contract_id"] = receipt["contract_id"] = reg["contract_id"]
    # Raw JSON deliberately bypasses write()'s nonfinite-output protection.
    base = root / "export"
    (base / "registration.json").write_text(json.dumps(reg), encoding="utf-8")
    gate.k.write(base / "prepare.json", prep)
    gate.k.write(base / "fit/result.json", receipt)
    export = gate.k.read(base / "EXPORT.json")
    export["contract_id"] = reg["contract_id"]
    export["sha256"] = {n: gate.k.sha(base / n) for n in export["sha256"]}
    gate.k.write(base / "EXPORT.json", export)
    assert gate.evaluate_k1(manifest)["folds"]["2"] == "HOLD"


@pytest.mark.parametrize("delta", [0, 9e-13, -9e-13])
def test_host_collapse_allowed_roundoff(host_export, delta):
    _, manifest, local, reg, prep, _, seal = host_export
    for field in gate.HOST_FLOAT_FIELDS:
        reg["collapse_guard"][field] = local["collapse_guard"][field] + delta
    # Delta=0 also exercises the unchanged exact-contract fast path.
    prep["argv"][-1] = repr(reg["collapse_guard"]["constant_loss"])
    seal()
    assert gate.evaluate_k1(manifest)["status"] == "HEALTH_PASS"


@pytest.mark.parametrize("name", [
    "registration.json", "prepare.json", "fit/result.json", "fit/holdout.csv"])
def test_host_export_hash_binding(host_export, name):
    root, manifest, *_ = host_export
    path = root / "export" / name
    path.write_bytes(path.read_bytes() + b" ")
    assert gate.evaluate_k1(manifest)["folds"]["2"] == "HOLD"


@pytest.mark.parametrize("change", ["missing_export", "missing_pin", "fold", "contract",
                                    "reg_digest", "result_contract", "csv_path"])
def test_host_export_identity_fail_closed(host_export, change):
    root, manifest, _, _, _, _, _ = host_export
    path = root / "export/EXPORT.json"
    export = gate.k.read(path)
    if change == "missing_export":
        path.unlink()
    elif change == "missing_pin":
        del export["sha256"]["registration.json"]
        gate.k.write(path, export)
    elif change in ("fold", "contract"):
        export["fold" if change == "fold" else "contract_id"] = "wrong"
        gate.k.write(path, export)
    elif change in ("reg_digest", "result_contract"):
        name = "registration.json" if change == "reg_digest" else "fit/result.json"
        value = gate.k.read(root / "export" / name)
        value["contract_id"] = "wrong"
        gate.k.write(root / "export" / name, value)
        export["sha256"][name] = gate.k.sha(root / "export" / name)
        gate.k.write(path, export)
    else:
        manifest["folds"]["2"]["k1_csv"] = "f2.csv"  # Same bytes outside the export.
    assert gate.evaluate_k1(manifest)["folds"]["2"] == "HOLD"


def test_full_gate_keeps_exact_contract_requirement(host_export):
    root, manifest, *_ = host_export
    paths = {n: root / p for n, p in manifest["folds"]["2"].items()}
    with pytest.raises(ValueError, match="contract mismatch"):
        gate.validate_k1(paths, ["a", "b"], 2)


def test_exact_local_never_reads_export(host_export, monkeypatch):
    root, manifest, local, _, _, receipt, _ = host_export
    receipt["contract_id"] = local["contract_id"]
    gate.k.write(root / "export/fit/result.json", receipt)
    monkeypatch.setattr(gate, "validate_host_export", lambda *a: pytest.fail("exact local"))
    result = gate.evaluate_k1(manifest)
    assert result["status"] == "HEALTH_PASS"
    assert result["evidence"]["2"]["contract"]["mode"] == "EXACT_LOCAL"
