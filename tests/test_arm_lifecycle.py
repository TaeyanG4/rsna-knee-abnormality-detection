"""v1.0.0 (astra, 2026-10-11): arm contract integrity and real-record CPU smoke.

Synthetic PASS notes are schema fixtures, never launch evidence. The real-record
test deliberately stays HOLD: legacy controls are not a smoke of a new runner.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from training.arms import lifecycle as life


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/experiments/EXP179_MULTISCALE/outputs/M42/results/EXP179_M42_F0"
CONTROL_HASHES = {
    "summary.json": "491cd2b63c783e96b2c4c0bec17d3ff0c06ea95e4c66dc34c6e595cc9eae3f9e",
    "provenance.json": "12c51991b51fcd18014bd18c17037afda5e87e8724e1b10e0f2e2f9784423179",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def profile():
    return {
        "schema": life.ENVIRONMENT_SCHEMA, "hostname": "test-target",
        "runtime": {"python": "3.11.10", "driver": "610.43.02", "torch": "2.5.1+cu121",
                    "cuda": "12.1", "cudnn": 90100, "gpu": "NVIDIA GeForce RTX 3090",
                    "timm": "1.0.19", "numpy": "2.0.2", "pydicom": "3.0.2", "cv2": "4.10.0"},
        "evidence": {name: digest(name) for name in ("summary.json", "provenance.json")},
    }


def note_for(profile):
    bindings = {"sources": {"runner.py": digest("runner")}, "config": digest("config"),
                "data": {"labels": digest("labels"), "corpus": digest("corpus")},
                "rosters": {"train": digest("train"), "validation": digest("validation")},
                "controls": {"seed42/fold0": digest("control")},
                "environment": life.content_sha256(profile)}
    smokes = {}
    for kind in ("cpu", "target"):
        smokes[kind] = {
            "hostname": "test-local" if kind == "cpu" else profile["hostname"],
            "device": "cpu" if kind == "cpu" else "cuda",
            "environment_sha256": digest("test-cpu") if kind == "cpu" else bindings["environment"],
            "bindings_sha256": life.content_sha256(bindings), "command": ["test-only", "smoke", kind],
            "exit_code": 0, "log_sha256": digest("test-only " + kind),
            "results": {"real_data": True, "train_batches": 1, "validation_batches": 1,
                        "bn_batches": 1, "optimizer_steps": 1,
                        "checkpoint_reloaded": True, "finite_outputs": True},
        }
    return {"schema": life.VERIFICATION_SCHEMA, "scope": "test-only/arm42/fold0",
            "reviewer": {"name": "test-reviewer", "host": "test-host"}, "verdict": "PASS",
            "blockers": [], "notes": ["Synthetic schema fixture; not launch evidence."],
            "bindings": bindings,
            "checks": [{"name": "parity", "command": ["test-only", "parity"],
                        "exit_code": 0, "log_sha256": digest("test-parity")}], "smokes": smokes}


@pytest.fixture
def review(profile):
    note = note_for(profile)
    expected = {"expected_scope": note["scope"], "expected_bindings": copy.deepcopy(note["bindings"]),
                "target_environment": profile}
    return note, expected


def test_canonical_json_stable_without_losing_order_or_types():
    left = {"b": [2, 1], "a": "검증"}
    assert life.canonical_bytes(left) == life.canonical_bytes({"a": "검증", "b": [2, 1]})
    assert life.parse_json(life.canonical_bytes(left)) == left
    assert life.content_sha256([1, 2]) != life.content_sha256([2, 1])
    assert len({life.content_sha256(value) for value in (True, 1, 1.0)}) == 3


@pytest.mark.parametrize("payload", [b'{"a":1,"a":2}', b'{"a":{"x":1,"x":2}}',
                                    b'{"n":NaN}', b'Infinity', b'1e999', b'\xff', b'{'])
def test_ambiguous_json_rejected(payload):
    with pytest.raises(life.ContractError):
        life.parse_json(payload)


@pytest.mark.parametrize("value", [{1: "coerced-key"}, (1, 2), float("nan"), float("inf"), {1, 2}])
def test_non_json_values_rejected(value):
    with pytest.raises(life.ContractError):
        life.canonical_bytes(value)


def test_environment_roundtrip_and_exact_host(profile):
    restored = life.parse_json(life.canonical_bytes(profile))
    life.check_environment(restored, hostname="test-target", runtime=profile["runtime"])
    with pytest.raises(life.ContractError, match="hostname"):
        life.check_environment(restored, hostname="registration-host", runtime=profile["runtime"])


@pytest.mark.parametrize("key", sorted(life.RUNTIME_KEYS))
def test_each_runtime_field_enforced(profile, key):
    actual = copy.deepcopy(profile["runtime"])
    actual[key] = 1 if key == "cudnn" else "different"
    with pytest.raises(life.ContractError, match=key):
        life.check_environment(profile, hostname=profile["hostname"], runtime=actual)


@pytest.mark.parametrize("change", ["missing", "extra", "boolean", "empty", "hash"])
def test_incomplete_or_coerced_environment_rejected(profile, change):
    if change == "missing":
        del profile["runtime"]["driver"]
    elif change == "extra":
        profile["runtime"]["allow_mismatch"] = True
    elif change == "boolean":
        profile["runtime"]["cudnn"] = True
    elif change == "empty":
        profile["runtime"]["gpu"] = " "
    else:
        profile["evidence"]["summary.json"] = "not-a-hash"
    with pytest.raises(life.ContractError):
        life.validate_environment(profile)


def control_documents(tmp_path, profile):
    summary = {"environment": dict(profile["runtime"], hostname=profile["hostname"], time="recorded")}
    provenance = {"environment": {key: profile["runtime"][key] for key in life.PROVENANCE_RUNTIME_KEYS}}
    hashes = {}
    for name, document in (("summary.json", summary), ("provenance.json", provenance)):
        raw = life.canonical_bytes(document)
        (tmp_path / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def test_recorded_profile_requires_bound_matching_records(tmp_path, profile):
    hashes = control_documents(tmp_path, profile)
    result = life.environment_from_control(tmp_path / "summary.json", tmp_path / "provenance.json",
                                           expected_hashes=hashes)
    assert result == dict(profile, evidence=hashes)
    (tmp_path / "summary.json").write_bytes(b"{}")
    with pytest.raises(life.ContractError, match="hash differs"):
        life.environment_from_control(tmp_path / "summary.json", tmp_path / "provenance.json",
                                      expected_hashes=hashes)


@pytest.mark.parametrize("change", ["registration", "missing", "extra"])
def test_recorded_summary_and_provenance_must_agree(tmp_path, profile, change):
    hashes = control_documents(tmp_path, profile)
    path = tmp_path / "provenance.json"
    document = life.parse_json(path.read_bytes())
    if change == "registration":
        document["environment"]["gpu"] = "NVIDIA GeForce RTX 4070 Ti SUPER"
    elif change == "missing":
        del document["environment"]["cv2"]
    else:
        document["environment"]["relaxed"] = True
    path.write_bytes(life.canonical_bytes(document))
    hashes["provenance.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(life.ContractError):
        life.environment_from_control(tmp_path / "summary.json", path, expected_hashes=hashes)


def test_strict_pass_roundtrip_and_hold_are_distinct(review):
    note, expected = review
    restored = life.parse_json(life.canonical_bytes(note))
    assert life.require_verification_pass(restored, **expected) == life.content_sha256(note)
    note.update(verdict="HOLD", blockers=["Target smoke pending."], smokes={"cpu": None, "target": None})
    life.validate_verification_note(note, **expected)
    with pytest.raises(life.ContractError, match="not PASS"):
        life.require_verification_pass(note, **expected)
    note["verdict"] = "CHALLENGED"
    life.validate_verification_note(note, **expected)


@pytest.mark.parametrize("key", sorted(life.BINDING_KEYS))
def test_all_bindings_checked_against_independent_expectation(review, key):
    note, expected = review
    if key in ("config", "environment"):
        note["bindings"][key] = digest("changed")
    else:
        name = next(iter(note["bindings"][key]))
        note["bindings"][key][name] = digest("changed")
    with pytest.raises(life.ContractError, match="bindings differ"):
        life.require_verification_pass(note, **expected)


@pytest.mark.parametrize("change", ["scope", "schema", "verdict", "blocker", "failed-check",
                                   "boolean-exit", "empty-checks", "duplicate-check", "reviewer",
                                   "extra-field", "missing-binding", "bad-hash", "notes-string"])
def test_note_fails_closed(review, change):
    note, expected = review
    if change == "scope":
        note["scope"] = "another-fold"
    elif change == "schema":
        note["schema"] = "v0"
    elif change == "verdict":
        note["verdict"] = "PASS-WITH-NOTES"
    elif change == "blocker":
        note["blockers"] = ["Needs target smoke."]
    elif change == "failed-check":
        note["checks"][0]["exit_code"] = 1
    elif change == "boolean-exit":
        note["checks"][0]["exit_code"] = False
    elif change == "empty-checks":
        note["checks"] = []
    elif change == "duplicate-check":
        note["checks"].append(copy.deepcopy(note["checks"][0]))
    elif change == "reviewer":
        note["reviewer"]["name"] = ""
    elif change == "extra-field":
        note["relaxed"] = True
    elif change == "missing-binding":
        del note["bindings"]["rosters"]
    elif change == "bad-hash":
        note["checks"][0]["log_sha256"] = "A" * 64
    else:
        note["notes"] = "PASS"
    with pytest.raises(life.ContractError):
        life.require_verification_pass(note, **expected)


@pytest.mark.parametrize("kind", ["cpu", "target"])
@pytest.mark.parametrize("change", ["missing", "failed", "synthetic", "zero-train", "zero-val",
                                   "zero-bn", "zero-updates", "no-reload", "nonfinite",
                                   "stale", "wrong-device", "boolean-count"])
def test_pass_requires_each_bound_real_data_smoke(review, kind, change):
    note, expected = review
    smoke = note["smokes"][kind]
    if change == "missing":
        note["smokes"][kind] = None
    elif change == "failed":
        smoke["exit_code"] = 1
    elif change == "synthetic":
        smoke["results"]["real_data"] = False
    elif change.startswith("zero-"):
        field = {"train": "train_batches", "val": "validation_batches", "bn": "bn_batches",
                 "updates": "optimizer_steps"}[change[5:]]
        smoke["results"][field] = 0
    elif change == "no-reload":
        smoke["results"]["checkpoint_reloaded"] = False
    elif change == "nonfinite":
        smoke["results"]["finite_outputs"] = False
    elif change == "stale":
        smoke["bindings_sha256"] = digest("prior-code")
    elif change == "wrong-device":
        smoke["device"] = "cpu" if kind == "target" else "cuda"
    else:
        smoke["results"]["train_batches"] = True
    with pytest.raises(life.ContractError):
        life.require_verification_pass(note, **expected)


@pytest.mark.parametrize("field", ["hostname", "environment_sha256"])
def test_local_smoke_cannot_qualify_target(review, field):
    note, expected = review
    note["smokes"]["target"][field] = "registration-host" if field == "hostname" else digest("local")
    with pytest.raises(life.ContractError, match="target environment"):
        life.require_verification_pass(note, **expected)


def test_real_record_cpu_smoke(tmp_path):
    # Real frozen artifacts, no label rows read or printed. Missing fixtures fail
    # rather than skip: this workspace must have the collected control.
    profile = life.environment_from_control(CONTROL / "summary.json", CONTROL / "provenance.json",
                                           expected_hashes=CONTROL_HASHES)
    overlay = ROOT / ".local/exp185-overlay"
    manifest_raw = (overlay / "manifest.json").read_bytes()
    manifest = life.parse_json(manifest_raw)
    control = manifest["controls"]["42"]
    assert all(control["hashes"][name] == value for name, value in CONTROL_HASHES.items())
    registration_raw = (overlay / "protocols/S42.json").read_bytes()
    registration = life.parse_json(registration_raw)
    observed = dict(control["provenance"]["environment"],
                    python=control["summary"]["environment"]["python"],
                    driver=control["summary"]["environment"]["driver"])
    life.check_environment(profile, hostname=control["summary"]["environment"]["hostname"], runtime=observed)
    local = dict(observed, **registration["environment"])
    with pytest.raises(life.ContractError, match="cv2, gpu"):
        life.check_environment(profile, hostname=profile["hostname"], runtime=local)

    note = note_for(profile)
    note.update(scope="mil/real-record-contract-smoke", verdict="HOLD",
                blockers=["New runner target-host smoke has not run."],
                notes=["Real recorded control/profile integration; not training qualification."],
                smokes={"cpu": None, "target": None})
    bindings = note["bindings"]
    bindings["sources"] = {name: record["sha256"] for name, record in manifest["files"].items()
                           if name.endswith(".py")}
    bindings["config"] = hashlib.sha256(registration_raw).hexdigest()
    bindings["data"] = {key: control["provenance"][key]
                        for key in ("label_file_sha256", "corpus_manifest_sha256")}
    bindings["rosters"] = {key: control["provenance"][key]
                           for key in ("train_ids_ordered_sha256", "eval_ids_ordered_sha256")}
    bindings["controls"] = dict(CONTROL_HASHES, manifest=hashlib.sha256(manifest_raw).hexdigest())
    path = tmp_path / "real-record-verification-hold.json"
    path.write_bytes(life.canonical_bytes(note))
    expected = {"expected_scope": note["scope"], "expected_bindings": copy.deepcopy(bindings),
                "target_environment": profile}
    life.validate_verification_note(life.parse_json(path.read_bytes()), **expected)
    with pytest.raises(life.ContractError, match="not PASS"):
        life.require_verification_pass(note, **expected)
    # Independently change a recorded roster to prove the note cannot move folds.
    expected["expected_bindings"]["rosters"]["eval_ids_ordered_sha256"] = digest("different-fold")
    with pytest.raises(life.ContractError, match="bindings differ"):
        life.validate_verification_note(note, **expected)


def test_foundation_import_has_no_torch_or_search_path_side_effect():
    code = """
import sys
before = sys.path[:]
from training.arms import lifecycle
assert sys.path == before
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert not any(name.startswith('tools.exp') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
