"""v1.0.0 (astra, 2026-10-11): artifact lifecycle parity and real-record CPU CLI.

Synthetic PASS evidence tests integrity gates only. Real-control roundtrips stay
HOLD, never masquerade as a new runner smoke, and never print label/prediction rows.
"""
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

from training.arms import lifecycle as life


ROOT = Path(__file__).resolve().parents[1]
CONTROL = "artifacts/experiments/EXP179_MULTISCALE/outputs/M42/results/EXP179_M42_F0"
CONTROL_HASHES = {
    "summary.json": "491cd2b63c783e96b2c4c0bec17d3ff0c06ea95e4c66dc34c6e595cc9eae3f9e",
    "provenance.json": "12c51991b51fcd18014bd18c17037afda5e87e8724e1b10e0f2e2f9784423179",
}


def put(root, name, data):
    life.create_new(root / name, data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def registered(tmp_path):
    root = tmp_path / "inputs"
    runtime = {key: "fixture" for key in life.RUNTIME_KEYS}
    runtime["cudnn"] = 90100
    controls = {
        "control/summary.json": put(root, "control/summary.json", life.canonical_bytes(
            {"environment": dict(runtime, hostname="fixture-target")})),
        "control/provenance.json": put(root, "control/provenance.json", life.canonical_bytes(
            {"environment": {k: runtime[k] for k in life.PROVENANCE_RUNTIME_KEYS}})),
    }
    profile = life.environment_from_control(root / "control/summary.json", root / "control/provenance.json",
                  expected_hashes={Path(k).name: v for k, v in controls.items()})
    config = {key: {"fixture": True} for key in life.CONFIG_KEYS}
    config["targets"] = [f"t{i}" for i in range(12)]
    config["backend_policy"] = {"deterministic_algorithms": True, "amp": "fp16"}
    document = {"schema": life.SPEC_SCHEMA, "family": "convnext-mil", "arm": "fixture",
                "seed": 42, "fold": 0, "attempt": "test1", "config": config,
                "target_environment": profile,
                "inputs": {"sources": {"runner.py": put(root, "runner.py", b"fixture source\n")},
                           "data": {"data.bin": put(root, "data.bin", b"fixture data")},
                           "rosters": {"roster.json": put(root, "roster.json", b'["a","b"]')},
                           "controls": controls},
                "outputs": ["checkpoint.pt", "readout.csv"]}
    return life.ArmSpec.from_document(document), root


def note_with_evidence(spec, root, *, passed=True):
    profile = spec.document["target_environment"]
    observation = {"hostname": profile["hostname"], "runtime": profile["runtime"],
                   "backend_policy": spec.document["config"]["backend_policy"]}
    note = {"schema": life.VERIFICATION_SCHEMA, "scope": spec.scope,
            "reviewer": {"name": "synthetic-fixture", "host": "test"},
            "verdict": "PASS" if passed else "HOLD", "blockers": [] if passed else ["New target smoke pending"],
            "notes": ["Integrity fixture only; not launch evidence"], "bindings": spec.bindings,
            "checks": [], "smokes": {"cpu": None, "target": None}}
    log = {"schema": life.EVIDENCE_SCHEMA, "command": ["fixture-check"], "exit_code": 0,
           "artifacts": {}, "smoke": None}
    digest = life.content_sha256(log)
    put(root, "evidence/" + digest + ".json", life.canonical_bytes(log))
    note["checks"].append({"name": "integrity", "command": log["command"], "exit_code": 0, "log_sha256": digest})
    if passed:
        for kind in ("cpu", "target"):
            obs = observation if kind == "target" else {"hostname": "fixture-cpu", "device": "cpu"}
            smoke = {"hostname": obs["hostname"], "device": "cuda" if kind == "target" else "cpu",
                     "environment_sha256": spec.bindings["environment"] if kind == "target" else life.content_sha256(obs),
                     "bindings_sha256": life.content_sha256(spec.bindings),
                     "results": {**dict.fromkeys(life.SMOKE_COUNTS, 1), **dict.fromkeys(life.SMOKE_FLAGS, True)}}
            artifacts = {}
            for role in ("checkpoint", "readout"):
                name = f"{kind}/{role}.bin"
                artifacts[role] = {"path": name, "sha256": put(root, name, (kind + role).encode())}
            log = {"schema": life.EVIDENCE_SCHEMA, "command": ["fixture-smoke", kind], "exit_code": 0,
                   "artifacts": artifacts, "smoke": dict(smoke, observation=obs)}
            digest = life.content_sha256(log)
            put(root, "evidence/" + digest + ".json", life.canonical_bytes(log))
            note["smokes"][kind] = dict(smoke, command=log["command"], exit_code=0, log_sha256=digest)
    return note, observation


def receipt_fixture(spec, root):
    for name in spec.document["outputs"]:
        put(root, name, ("fixture-output-" + name).encode())
    return life.artifact_receipt(spec, root, verification_sha256="1" * 64, exit_code=0,
                                 elapsed_seconds=10**9, metrics={"finite_predictions": 24})


def test_spec_is_deeply_immutable_and_all_scientific_contracts_are_bound(registered):
    spec, root = registered
    doc = spec.document
    doc["config"]["updates"]["fixture"] = False
    assert spec.document["config"]["updates"]["fixture"] is True
    assert life.ArmSpec.from_document(doc).sha256 != spec.sha256
    life.validate_inputs(spec, root)
    for key in life.CONFIG_KEYS:
        doc = spec.document
        del doc["config"][key]
        with pytest.raises(life.ContractError, match="fields"):
            life.ArmSpec.from_document(doc)
    for group in life.FILE_GROUPS:
        doc = spec.document
        name = next(iter(doc["inputs"][group]))
        doc["inputs"][group][name] = "2" * 64
        changed = life.ArmSpec.from_document(doc)
        assert changed.bindings != spec.bindings
        with pytest.raises(life.ContractError):
            life.validate_inputs(changed, root)


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "a\\b", "a//b", "a/./b", "a/../b",
                                  "a:b", "x.", "x ", "nul", "COM1.txt", "x/aux.json", "x\x00y", ""])
def test_portable_unsafe_paths_rejected(name):
    with pytest.raises(life.ContractError):
        life.safe_member(name)


@pytest.mark.parametrize("change", ["case", "prefix", "duplicate", "reserved", "reserved-prefix",
                                   "output", "output-prefix", "output-type", "fold", "family"])
def test_spec_identity_and_inventory_rejections(registered, change):
    spec, _ = registered
    d = spec.document
    if change == "case":
        d["inputs"]["sources"]["RUNNER.py"] = "1" * 64
    elif change == "prefix":
        d["inputs"]["sources"]["runner.py/child"] = "1" * 64
    elif change == "duplicate":
        d["inputs"]["data"]["runner.py"] = "1" * 64
    elif change == "reserved":
        d["inputs"]["sources"]["arm-spec.json"] = "1" * 64
    elif change == "reserved-prefix":
        d["inputs"]["sources"]["arm-spec.json/child"] = "1" * 64
    elif change == "output":
        d["outputs"].append(d["outputs"][0])
    elif change == "output-prefix":
        d["outputs"].append("receipt.json/child")
    elif change == "output-type":
        d["outputs"].append({"bad": []})
    elif change == "fold":
        d.update(family="coatnet-bridge", fold=1)
    else:
        d["family"] = "dynamic-python-import"
    with pytest.raises(life.ContractError):
        life.ArmSpec.from_document(d)


def test_target_profile_must_come_from_bound_control(registered):
    spec, root = registered
    d = spec.document
    d["target_environment"]["runtime"]["gpu"] = "registration-gpu"
    with pytest.raises(life.ContractError, match="control"):
        life.validate_inputs(life.ArmSpec.from_document(d), root)


def test_fresh_staging_replacement_does_not_mutate_source(registered, tmp_path):
    spec, root = registered
    doc = spec.document
    original = (root / "runner.py").read_bytes()
    replacement = b"explicit supported runner\n"
    new_hash = put(root, "replacement.py", replacement)
    doc["inputs"]["sources"]["runner.py"] = new_hash
    revised = life.ArmSpec.from_document(doc)
    options = {"runner.py": {"source": "replacement.py", "previous_sha256": spec.inventory["runner.py"]}}
    stage = tmp_path / "stage"
    life.prepare(revised, root, stage, replacements=options)
    life.validate_stage(revised, stage)
    assert (stage / "runner.py").read_bytes() == replacement
    assert (root / "runner.py").read_bytes() == original
    before = {n: (stage / n).stat().st_mtime_ns for n in life._files(stage)}
    assert life.prepare(revised, root, stage, replacements=options) == life.validate_stage(revised, stage)
    assert before == {n: (stage / n).stat().st_mtime_ns for n in life._files(stage)}
    with pytest.raises(life.ContractError):
        life.prepare(spec, root, stage)
    options["runner.py"]["previous_sha256"] = "3" * 64
    rejected = tmp_path / "rejected"
    with pytest.raises(life.ContractError, match="source differs"):
        life.prepare(revised, root, rejected, replacements=options)
    assert not rejected.exists()


@pytest.mark.parametrize("change", ["source", "data", "receipt", "spec", "extra", "missing"])
def test_stage_tamper_rejected(registered, tmp_path, change):
    spec, root = registered
    stage = tmp_path / "stage"
    life.prepare(spec, root, stage)
    name = {"source": "runner.py", "data": "data.bin", "receipt": "stage-receipt.json",
            "spec": "arm-spec.json", "extra": "extra", "missing": "runner.py"}[change]
    if change == "missing":
        (stage / name).rename(stage / "moved.py")
    else:
        (stage / name).write_bytes(b"tampered")
    with pytest.raises(life.ContractError):
        life.validate_stage(spec, stage)
    with pytest.raises(life.ContractError):
        life.prepare(spec, root, stage)


def test_create_never_overwrites_and_registration_is_canonical(registered, tmp_path):
    spec, _ = registered
    path = tmp_path / "registered.json"
    assert life.register(spec.document, path) == spec
    with pytest.raises(FileExistsError):
        life.register(spec.document, path)
    assert path.read_bytes() == spec.encoded


def test_evidence_and_fresh_runtime_are_separate_admission_requirements(registered, tmp_path):
    spec, root = registered
    stage = tmp_path / "stage"
    life.prepare(spec, root, stage)
    note, obs = note_with_evidence(spec, tmp_path / "evidence-root")
    result = life.admission_dry_run(spec, stage, note, tmp_path / "evidence-root")
    assert result["status"] == "EVIDENCE_READY_TARGET_CHECK_REQUIRED"
    assert result["launch_authorized"] is False
    assert life.require_execution_evidence(spec, stage, note, tmp_path / "evidence-root",
                                           fresh_observation=obs) == life.content_sha256(note)
    bad = copy.deepcopy(obs)
    bad["backend_policy"]["amp"] = "bf16"
    with pytest.raises(life.ContractError, match="backend"):
        life.require_execution_evidence(spec, stage, note, tmp_path / "evidence-root", fresh_observation=bad)
    bad = copy.deepcopy(obs)
    bad["runtime"]["gpu"] = "registration-host GPU"
    with pytest.raises(life.ContractError, match="gpu"):
        life.require_execution_evidence(spec, stage, note, tmp_path / "evidence-root", fresh_observation=bad)


@pytest.mark.parametrize("change", ["log", "checkpoint", "readout", "missing-log", "command", "results", "backend", "roles"])
def test_evidence_tamper_or_inconsistent_producer_rejected(registered, tmp_path, change):
    spec, _ = registered
    evidence = tmp_path / "evidence-root"
    note, _ = note_with_evidence(spec, evidence)
    record = note["smokes"]["target"]
    path = evidence / ("evidence/" + record["log_sha256"] + ".json")
    if change in ("log", "checkpoint", "readout"):
        (path if change == "log" else evidence / ("target/" + change + ".bin")).write_bytes(b"bad")
    elif change == "missing-log":
        path.rename(path.with_suffix(".missing"))
    else:
        log = life.parse_json(path.read_bytes())
        if change == "command":
            log["command"] = ["different-command"]
        elif change == "results":
            log["smoke"]["results"]["real_data"] = False
        elif change == "backend":
            log["smoke"]["observation"]["backend_policy"]["amp"] = "bf16"
        else:
            del log["artifacts"]["checkpoint"]
        # Even a correctly re-hashed log must agree with the reviewed declarations.
        record["log_sha256"] = life.content_sha256(log)
        put(evidence, "evidence/" + record["log_sha256"] + ".json", life.canonical_bytes(log))
    with pytest.raises((life.ContractError, FileNotFoundError)):
        life.verify_evidence(spec, note, evidence)


def test_hold_dry_run_is_not_execution_approval(registered, tmp_path):
    spec, root = registered
    stage = tmp_path / "stage"
    life.prepare(spec, root, stage)
    note, obs = note_with_evidence(spec, tmp_path / "evidence-root", passed=False)
    assert life.admission_dry_run(spec, stage, note, tmp_path / "evidence-root")["status"] == "HOLD"
    with pytest.raises(life.ContractError, match="not PASS"):
        life.require_execution_evidence(spec, stage, note, tmp_path / "evidence-root", fresh_observation=obs)


def test_artifact_roundtrip_has_no_wall_gate_or_scientific_acceptance(registered, tmp_path):
    spec, _ = registered
    out = tmp_path / "outputs"
    receipt = receipt_fixture(spec, out)
    digest = life.content_sha256(receipt)
    assert receipt["elapsed_seconds"] == 10**9
    archive = tmp_path / "export.tar"
    life.export_artifacts(spec, out, receipt, archive, expected_receipt_sha256=digest)
    collected = tmp_path / "collected"
    result = life.collect_artifacts(spec, archive, collected, expected_receipt_sha256=digest)
    assert result["scientific_status"] == "UNASSESSED"
    assert life.verify_artifact_receipt(spec, collected / "outputs", receipt,
                                        expected_receipt_sha256=digest) == receipt
    with pytest.raises(FileExistsError):
        life.export_artifacts(spec, out, receipt, archive, expected_receipt_sha256=digest)
    with pytest.raises(life.ContractError, match="fresh"):
        life.collect_artifacts(spec, archive, collected, expected_receipt_sha256=digest)


def test_compatible_receipts_keep_the_producer_version(registered, tmp_path):
    spec, _ = registered
    out = tmp_path / "outputs"
    receipt = receipt_fixture(spec, out)
    receipt["tool_version"] = "1.0.0"
    assert life.verify_artifact_receipt(spec, out, receipt,
            expected_receipt_sha256=life.content_sha256(receipt))["tool_version"] == "1.0.0"
    receipt["schema"] = "unknown/v2"
    with pytest.raises(life.ContractError):
        life.verify_artifact_receipt(spec, out, receipt, expected_receipt_sha256=life.content_sha256(receipt))


def test_export_regularizes_local_hardlinks(registered, tmp_path):
    spec, _ = registered
    out = tmp_path / "outputs"
    put(out, "checkpoint.pt", b"same hardlinked output")
    os.link(out / "checkpoint.pt", out / "readout.csv")
    receipt = life.artifact_receipt(spec, out, verification_sha256="1" * 64, exit_code=0,
                                     elapsed_seconds=0, metrics={"fixture": 1})
    digest = life.content_sha256(receipt)
    archive = tmp_path / "hardlinks.tar"
    life.export_artifacts(spec, out, receipt, archive, expected_receipt_sha256=digest)
    with tarfile.open(archive) as source:
        assert all(member.isfile() for member in source.getmembers())
    life.collect_artifacts(spec, archive, tmp_path / "collected", expected_receipt_sha256=digest)


def test_export_rejects_bytes_changed_after_receipt_validation(registered, tmp_path, monkeypatch):
    spec, _ = registered
    out = tmp_path / "outputs"
    receipt = receipt_fixture(spec, out)
    addfile = tarfile.TarFile.addfile

    def mutate(self, info, fileobj=None):
        if info.name == "receipt.json":
            (out / "checkpoint.pt").write_bytes(b"concurrent change")
        return addfile(self, info, fileobj)

    monkeypatch.setattr(tarfile.TarFile, "addfile", mutate)
    with pytest.raises(life.ContractError, match="export output changed"):
        life.export_artifacts(spec, out, receipt, tmp_path / "incomplete.tar",
                               expected_receipt_sha256=life.content_sha256(receipt))


def test_linked_roots_and_members_are_rejected(registered, tmp_path, monkeypatch):
    spec, root = registered
    linked = root
    monkeypatch.setattr(life, "_linked", lambda path: path == linked)
    with pytest.raises(life.ContractError, match="linked root"):
        life.validate_inputs(spec, root)
    linked = root / "runner.py"
    with pytest.raises(life.ContractError, match="linked member"):
        life.prepare(spec, root, tmp_path / "stage")


@pytest.mark.parametrize("change", ["missing", "extra", "nonfinite", "exit", "receipt", "output", "scope"])
def test_partial_or_invalid_outputs_cannot_get_validated_receipts(registered, tmp_path, change):
    spec, _ = registered
    out = tmp_path / "outputs"
    receipt = receipt_fixture(spec, out)
    expected = life.content_sha256(receipt)
    if change == "missing":
        (out / "checkpoint.pt").rename(tmp_path / "absent.pt")
    elif change == "extra":
        put(out, "extra", b"bad")
    elif change == "nonfinite":
        receipt["metrics"]["finite_predictions"] = float("nan")
    elif change == "exit":
        receipt["exit_code"] = 1
        expected = life.content_sha256(receipt)
    elif change == "receipt":
        receipt["verification_sha256"] = "5" * 64
    elif change == "scope":
        receipt["scope"] = "another-fold"
        expected = life.content_sha256(receipt)
    else:
        (out / "readout.csv").write_bytes(b"altered")
    with pytest.raises(life.ContractError):
        life.verify_artifact_receipt(spec, out, receipt, expected_receipt_sha256=expected)


@pytest.mark.parametrize("change", ["traversal", "duplicate", "missing", "symlink", "hardlink", "directory", "tamper"])
def test_collect_rejects_unsafe_or_incomplete_archives_before_publication(registered, tmp_path, change):
    spec, _ = registered
    out = tmp_path / "outputs"
    receipt = receipt_fixture(spec, out)
    files = [("receipt.json", life.canonical_bytes(receipt))]
    files += [("outputs/" + n, (out / n).read_bytes()) for n in spec.document["outputs"]]
    if change == "traversal":
        files[1] = ("../escape", files[1][1])
    elif change == "duplicate":
        files.append(files[1])
    elif change == "missing":
        files.pop()
    elif change == "tamper":
        files[1] = (files[1][0], b"changed")
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "x") as target:
        for i, (name, raw) in enumerate(files):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            if i == 1 and change in ("symlink", "hardlink", "directory"):
                info.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "directory": tarfile.DIRTYPE}[change]
                info.linkname = "../../escape"
                info.size = 0
            target.addfile(info, io.BytesIO(raw))
    destination = tmp_path / "collected"
    with pytest.raises(life.ContractError):
        life.collect_artifacts(spec, archive, destination, expected_receipt_sha256=life.content_sha256(receipt))
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


def test_float32_csv_boundary_and_exact_alignment(tmp_path):
    values = np.asarray([[0.123456789012, .5], [.98765432109, .5 + 1e-10]], dtype=np.float64)
    path = tmp_path / "oof.csv"
    path.write_bytes(life.prediction_csv_bytes(["001", "002"], values, ["a", "b"]))
    frame = pd.DataFrame(values.astype(np.float32), columns=["a", "b"])
    frame.insert(0, "study_id", ["001", "002"])
    expected = pd.read_csv(io.StringIO(frame.to_csv(index=False)), dtype={"study_id": str})
    actual = life.load_prediction_csv(path, ["002", "001"], ["a", "b"])
    np.testing.assert_array_equal(actual, expected[["a", "b"]].to_numpy()[::-1])
    assert actual.dtype == np.float64 and actual[0, 1] == actual[1, 1]
    assert actual[0, 0] != values[1, 0]
    with pytest.raises(life.ContractError, match="roster"):
        life.load_prediction_csv(path, ["001", "003"], ["a", "b"])
    with pytest.raises(life.ContractError, match="header"):
        life.load_prediction_csv(path, ["001", "002"], ["b", "a"])


def test_real_control_cpu_cli_roundtrip(tmp_path):
    """Run every LAUNCH-card CPU command against real control/data/OOF bytes.

    This checks infrastructure only: it does not train a model or claim smoke
    coverage of a future family runner. Missing real fixtures fail, never skip.
    """
    work = Path(os.environ.get("ARM_LIFECYCLE_SMOKE_OUT", str(tmp_path / "real-smoke")))
    work.mkdir(parents=True)  # Must be fresh, preserving every earlier receipt.
    provenance = life.parse_json((ROOT / CONTROL / "provenance.json").read_bytes())
    summary = life.parse_json((ROOT / CONTROL / "summary.json").read_bytes())
    protocol = life.parse_json((ROOT / ".local/exp185-overlay/protocols/S42.json").read_bytes())
    roster_path = protocol["folds"]["0"]["ordered_ids"]["path"]
    rosters = life.parse_json((ROOT / roster_path).read_bytes())
    prediction_path = ROOT / CONTROL / "oof_k36_flip_tta.csv"
    with prediction_path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    targets = header[1:]
    profile = life.environment_from_control(ROOT / CONTROL / "summary.json", ROOT / CONTROL / "provenance.json",
                                            expected_hashes=CONTROL_HASHES)
    source_names = ["training/arms/lifecycle.py", "tools/arm_lifecycle.py"]
    label_path = protocol["paths"]["labels"]
    # Hash/copy labels opaquely: never parse or emit their rows.
    inputs = {"sources": {n: life.file_sha256(ROOT / n) for n in source_names},
              "data": {label_path: provenance["label_file_sha256"],
                       ".local/exp185-overlay/manifest.json": life.file_sha256(ROOT / ".local/exp185-overlay/manifest.json")},
              "rosters": {roster_path: protocol["folds"]["0"]["ordered_ids"]["sha256"]},
              "controls": {CONTROL + "/" + n: digest for n, digest in CONTROL_HASHES.items()}}
    config = {"model": {"arch": provenance["arch"], "kind": "historical-control-infrastructure-smoke"},
              "labels": {"file_sha256": provenance["label_file_sha256"]}, "targets": targets,
              "windows": {"sampling": provenance["sampling"], "input_policy": provenance["input_policy"]},
              "sampler": {"train_ids_ordered_sha256": provenance["train_ids_ordered_sha256"]},
              "updates": {"optimizer_updates": summary["optimizer_updates"],
                          "scheduler_total_steps": provenance["scheduler_total_steps"]},
              "bn": {"policy": "historical ConvNeXt LayerNorm; no synthetic BN claim"},
              "readout": {"primary": "oof_k36_flip_tta.csv", "serialization": "float32-pandas-csv-float64"},
              "backend_policy": provenance["runtime"],
              "roots": {"corpus": protocol["paths"]["corpus"], "scope": "CPU-artifact-smoke-only"}}
    doc = {"schema": life.SPEC_SCHEMA, "family": "convnext-mil", "arm": "real-control-records",
           "attempt": "cpu-artifacts-v1", "seed": 42, "fold": 0, "config": config, "inputs": inputs,
           "target_environment": profile, "outputs": ["oof_k36_flip_tta.csv", "summary.json", "provenance.json"]}
    spec = life.ArmSpec.from_document(doc)
    put(work, "draft.json", spec.encoded)
    results = []

    def cli(*args, expected_exit=0):
        command = [sys.executable, "-B", "-m", "tools.arm_lifecycle", *map(str, args)]
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                              env=dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=""))
        assert proc.returncode == expected_exit, proc.stdout + proc.stderr
        result = json.loads(proc.stdout)
        results.append({"command": command, "exit_code": proc.returncode, "result": result})
        return result

    spec_path, stage = work / "registered.json", work / "stage"
    cli("register", "--draft", work / "draft.json", "--root", ROOT, "--out", spec_path)
    cli("validate", "--spec", spec_path, "--root", ROOT)
    cli("prepare", "--spec", spec_path, "--root", ROOT, "--out", stage)
    before = {n: (stage / n).stat().st_mtime_ns for n in life._files(stage)}
    cli("prepare", "--spec", spec_path, "--root", ROOT, "--out", stage)
    assert before == {n: (stage / n).stat().st_mtime_ns for n in life._files(stage)}
    # Exercise explicit extracted-member replacement on real manifest bytes.
    # This is a same-content replacement; differing pre/postimages are tested above.
    member = ".local/exp185-overlay/manifest.json"
    put(work, "replacements.json", life.canonical_bytes({member: {
        "source": member, "previous_sha256": spec.inventory[member]}}))
    replacement_stage = work / "replacement-stage"
    cli("prepare", "--spec", spec_path, "--root", ROOT, "--out", replacement_stage,
        "--replacements", work / "replacements.json")
    cli("validate", "--spec", spec_path, "--root", replacement_stage, "--stage")
    cli("validate", "--spec", spec_path, "--root", stage, "--stage")
    # A genuine infrastructure check log bound to the real stage receipt.
    record = {"schema": life.EVIDENCE_SCHEMA, "command": results[-1]["command"], "exit_code": 0,
              "artifacts": {"stage": {"path": "stage/stage-receipt.json", "sha256": life.file_sha256(stage / "stage-receipt.json")}},
              "smoke": None}
    record_hash = life.content_sha256(record)
    put(work, "evidence/" + record_hash + ".json", life.canonical_bytes(record))
    note = {"schema": life.VERIFICATION_SCHEMA, "scope": spec.scope,
            "reviewer": {"name": "astra-infrastructure-builder", "host": "Codex"},
            "verdict": "HOLD", "blockers": ["Family runner real-data CPU and target-GPU smoke not implemented yet"],
            "notes": ["Historical records are infrastructure fixtures, not new model execution"],
            "bindings": spec.bindings, "checks": [{"name": "real-stage", "command": record["command"],
              "exit_code": 0, "log_sha256": record_hash}], "smokes": {"cpu": None, "target": None}}
    put(work, "verification.json", life.canonical_bytes(note))
    result = cli("admit", "--spec", spec_path, "--stage", stage, "--note", work / "verification.json",
                 "--evidence-root", work, "--dry-run")
    assert result["status"] == "HOLD" and result["launch_authorized"] is False
    # Changed data in our private copy cannot pass CPU admission or be silently
    # repaired by a repeat prepare. Original labels are never opened as text.
    staged_manifest = replacement_stage / member
    staged_manifest.write_bytes(staged_manifest.read_bytes() + b"\n")
    rejected = cli("admit", "--spec", spec_path, "--stage", replacement_stage,
                   "--note", work / "verification.json", "--evidence-root", work,
                   "--dry-run", expected_exit=2)
    assert rejected["status"] == "REJECTED"
    output = work / "outputs"
    for name in doc["outputs"]:
        put(output, name, (ROOT / CONTROL / name).read_bytes())
    # IDs may be inspected internally for alignment; only counts/hashes are emitted.
    ids = rosters["val"]
    actual = life.load_prediction_csv(output / "oof_k36_flip_tta.csv", ids, targets, id_column=header[0])
    expected = pd.read_csv(prediction_path).set_index(header[0]).loc[ids, targets].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(actual, expected)
    rewritten = work / "float32-roundtrip.csv"
    rewritten.write_bytes(life.prediction_csv_bytes(ids, actual, targets, id_column=header[0]))
    reread = life.load_prediction_csv(rewritten, ids, targets, id_column=header[0])
    historical = pd.DataFrame(actual.astype(np.float32), columns=targets)
    historical.insert(0, header[0], ids)
    np.testing.assert_array_equal(reread, pd.read_csv(io.StringIO(historical.to_csv(index=False)))[targets].to_numpy())
    put(work, "metrics.json", life.canonical_bytes({"studies": len(ids), "targets": len(targets)}))
    receipt = work / "receipt.json"
    result = cli("receipt", "--spec", spec_path, "--outputs", output, "--out", receipt,
                 "--verification-sha256", life.content_sha256(note), "--exit-code", "0",
                 "--elapsed-seconds", str(summary["total_seconds"]), "--metrics", work / "metrics.json")
    digest = result["receipt_sha256"]
    archive = work / "export.tar"
    cli("export", "--spec", spec_path, "--outputs", output, "--receipt", receipt,
        "--receipt-sha256", digest, "--out", archive)
    cli("collect", "--spec", spec_path, "--archive", archive, "--receipt-sha256", digest, "--out", work / "collected")
    put(work, "invalid.tar", b"invalid archive fixture")
    rejected = cli("collect", "--spec", spec_path, "--archive", work / "invalid.tar",
                   "--receipt-sha256", digest, "--out", work / "rejected-collection", expected_exit=2)
    assert rejected["status"] == "REJECTED" and not (work / "rejected-collection").exists()
    np.testing.assert_array_equal(life.load_prediction_csv(work / "collected/outputs/oof_k36_flip_tta.csv",
                                  ids, targets, id_column=header[0]), expected)
    put(work, "command-results.json", life.canonical_bytes({"status": "CPU_ARTIFACT_SMOKE_PASSED",
         "studies": len(ids), "targets": len(targets), "commands": results,
         "launch_authorized": False, "real_runner_smoke": False}))
