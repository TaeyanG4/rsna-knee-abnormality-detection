"""CPU checks for the EXP174 Gold58 guard adapter; no torch, labels or GPU."""
from __future__ import annotations

import json

import pytest

from tools import exp174_gold58 as gold


def test_bound_functions_use_exp174_registration_without_mutating_exp143():
    original = gold.base.predict.__globals__["load_registration"]
    bound = gold._bound(gold.base.predict)
    assert bound.__globals__["load_registration"] is gold.e174.load_registration
    assert bound.__globals__["registered_run"] is gold.e174.registered_run
    assert gold.base.predict.__globals__["load_registration"] is original
    assert gold.base.load_registration is original


@pytest.mark.parametrize("command,frozen,gold_flag", [("predict", "predict", True), ("report", "report", None)])
def test_cli_forwards_to_frozen_exp143_paths_with_exp174_protocol(monkeypatch, command, frozen, gold_flag):
    calls = []
    monkeypatch.setattr(gold, "_bound", lambda fn: (lambda *a, **k: calls.append((fn.__name__, a, k))))
    gold.main([command, "--precision", "fp16"] if command == "predict" else [command])
    name, args, kwargs = calls[0]
    assert name == frozen and args[0].protocol == str(gold.e174.PROTOCOL) and args[0].lane == "local"
    if gold_flag:
        assert kwargs == {"gold": True} and args[0].precision == "fp16" and args[0].device == "cuda:0"


def test_unknown_arguments_are_rejected_outside_forwarded_commands():
    with pytest.raises(ValueError, match="Unknown arguments"):
        gold.main(["guard", "--precision", "fp16"])


@pytest.mark.parametrize("value,status", [(0.9300, "PASS"), (0.9287, "PASS"),
                                         (0.9151, "PASS"), (0.9149, "FAIL")])
def test_guard_threshold_is_exp143_minus_002(tmp_path, monkeypatch, value, status):
    registration = {"run_dirs": gold.e174.RUN_DIRS}
    monkeypatch.setattr(gold.e174, "load_registration", lambda protocol: registration)
    monkeypatch.setattr(gold.e174, "registered_run", lambda p: tmp_path)
    written = {}
    monkeypatch.setattr(gold.e174, "write_new_json", lambda path, value: written.update(path=path, value=value))
    report = dict(status="REPORT_ONLY_NO_SELECTION", n=58, protocol_sha256=gold.base.sha256(gold.e174.PROTOCOL),
                  run_dir=gold.e174.RUN_DIRS["local"], results={"own143": {"macro12": value}})
    (tmp_path / "gold58-report.json").write_text(json.dumps(report), encoding="utf-8")
    result = gold.guard(out=tmp_path / "guard.json")
    assert result["status"] == status and written["value"] == result
    assert abs(result["threshold"] - (result["exp143_v2_macro12"] - 0.02)) < 1e-12
    assert abs(result["exp143_v2_macro12"] - 0.9350) < 5e-4


def test_guard_rejects_a_report_from_another_run(tmp_path, monkeypatch):
    monkeypatch.setattr(gold.e174, "load_registration", lambda protocol: {})
    monkeypatch.setattr(gold.e174, "registered_run", lambda p: tmp_path)
    report = dict(status="REPORT_ONLY_NO_SELECTION", n=58, protocol_sha256="0" * 64,
                  run_dir="artifacts/experiments/EXP143_OWN0949/S42v2", results={"own143": {"macro12": .95}})
    (tmp_path / "gold58-report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="report identity"):
        gold.guard(out=tmp_path / "guard.json")


@pytest.mark.parametrize("source,value", [
    ("EXP174", 1.01), ("EXP174", -0.01), ("EXP174", None),
    ("EXP174", "NaN"), ("EXP174", True),
    ("EXP143 v2", 1.01), ("EXP143 v2", -0.01),
    ("EXP143 v2", None), ("EXP143 v2", "NaN"),
])
def test_guard_rejects_impossible_auc(tmp_path, monkeypatch, source, value):
    monkeypatch.setattr(gold.e174, "load_registration", lambda protocol: {"run_dirs": gold.e174.RUN_DIRS})
    monkeypatch.setattr(gold.e174, "registered_run", lambda p: tmp_path)
    written = []
    monkeypatch.setattr(gold.e174, "write_new_json", lambda path, value: written.append(path))
    report = dict(status="REPORT_ONLY_NO_SELECTION", n=58, protocol_sha256=gold.base.sha256(gold.e174.PROTOCOL),
                  run_dir=gold.e174.RUN_DIRS["local"],
                  results={"own143": {"macro12": value if source == "EXP174" else 0.9287}})
    (tmp_path / "gold58-report.json").write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    if source == "EXP143 v2":
        read_json = gold.base.read_json
        reference = read_json(gold.EXP143_REPORT)
        reference["results"]["own143"]["macro12"] = value
        monkeypatch.setattr(gold.base, "read_json", lambda path: reference if path == gold.EXP143_REPORT else read_json(path))
    with pytest.raises(ValueError, match=f"{source} Gold58 macro12 is not a valid AUC"):
        gold.guard(out=tmp_path / "guard.json")
    assert not written and not (tmp_path / "guard.json").exists()


@pytest.mark.parametrize("source", ["EXP174", "EXP143 v2"])
def test_guard_rejects_nonfinite_auc(tmp_path, monkeypatch, source):
    monkeypatch.setattr(gold.e174, "load_registration", lambda protocol: {})
    monkeypatch.setattr(gold.e174, "registered_run", lambda p: tmp_path)
    read_json = gold.base.read_json
    own = dict(status="REPORT_ONLY_NO_SELECTION", n=58, protocol_sha256=gold.base.sha256(gold.e174.PROTOCOL),
               run_dir=gold.e174.RUN_DIRS["local"], results={"own143": {"macro12": 0.9287}})
    reference = read_json(gold.EXP143_REPORT)
    target = own if source == "EXP174" else reference
    target["results"]["own143"]["macro12"] = float("nan")
    monkeypatch.setattr(gold.base, "read_json", lambda path: reference if path == gold.EXP143_REPORT else own)
    with pytest.raises(ValueError, match=f"{source} Gold58 macro12 is not a valid AUC"):
        gold.guard(out=tmp_path / "guard.json")
    assert not (tmp_path / "guard.json").exists()
