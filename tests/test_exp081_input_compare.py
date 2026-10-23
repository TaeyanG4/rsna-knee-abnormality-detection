"""Synthetic contracts and label gates; no acquired inputs, model weights or GPU."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import exp081_input_compare as compare


def pair():
    ids = ["a", "b", "c"]
    values = np.full((3, 12), .25, dtype=np.float32)
    timing = dict(all_finite=True, fp32_fallbacks=0, batch_size=8,
                  window_policy="within-slot-valid", norm="imagenet",
                  input_hash_scheme="normalized-float32-shape-v1",
                  input_tensor_hashes=[dict(uid=uid, sha256=uid*64) for uid in ids])
    altered = values.copy(); altered[1] = .75
    new_timing = deepcopy(timing); new_timing["input_tensor_hashes"][1]["sha256"] = "x"*64
    return ids, ["b"], values, altered, timing, new_timing


def test_changed_prediction_allowed_but_unchanged_must_be_exact():
    result = compare.validate_prediction_pair(*pair())
    assert result["passed"] and result["unchanged"] == 2
    assert result["tensor_changed_ids"] == ["b"]
    assert result["unchanged_byte_exact"] and result["unchanged_max_abs"] == 0


def test_one_ulp_unchanged_prediction_difference_blocks_labels():
    args = pair(); args[3][0, 0] = np.nextafter(args[3][0, 0], np.float32(1))
    with pytest.raises(ValueError, match="not byte-exact"):
        compare.validate_prediction_pair(*args)


def test_unchanged_tensor_difference_blocks_even_equal_predictions():
    args = pair(); args[5]["input_tensor_hashes"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="normalized tensor differs"):
        compare.validate_prediction_pair(*args)


@pytest.mark.parametrize("value", [np.nan, np.inf, -.01, 1.01])
def test_nonfinite_or_out_of_range_probability_rejected(value):
    args = pair(); args[3][1, 0] = value
    with pytest.raises(ValueError, match="probabilities"):
        compare.validate_prediction_pair(*args)


@pytest.mark.parametrize("field,value", [("all_finite", False), ("fp32_fallbacks", 1), ("batch_size", 4),
    ("window_policy", "global-valid"), ("norm", "none"), ("input_hash_scheme", "old")])
def test_runtime_input_receipt_contract_rejected(field, value):
    args = pair(); args[5][field] = value
    with pytest.raises(ValueError, match="incompatible"):
        compare.validate_prediction_pair(*args)


def test_reordered_input_receipt_rejected():
    args = pair(); args[5]["input_tensor_hashes"].reverse()
    with pytest.raises(ValueError, match="incomplete"):
        compare.validate_prediction_pair(*args)


def test_float64_probabilities_not_silently_cast():
    args = list(pair()); args[3] = args[3].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        compare.validate_prediction_pair(*args)


@pytest.mark.parametrize("gate", [{}, {"passed":False}, {"passed":True},
    {"passed":False,"label_access_authorized_after_gate":True}])
def test_label_gate_rejects_before_any_file_io(monkeypatch, gate):
    monkeypatch.setattr(compare.prep, "sha", lambda path: pytest.fail("file read before pair gate"))
    with pytest.raises(ValueError, match="sealed"):
        compare.read_reference_labels({}, gate)


def test_reference_load_keeps_exact_f2_order_and_float32(tmp_path, monkeypatch):
    path = tmp_path / "weak.csv"
    path.write_text("StudyInstanceUID,"+",".join(compare.LAB)+"\nb,"+",".join(["0.5"]*12)+
                    "\nnot_f2,"+",".join(["not a numeric label"]*12)+"\na,"+",".join(["0.2"]*12)+"\n")
    contract = dict(ids=["a","b"], reference_labels=dict(path="workspace://weak.csv",
                          file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    y = compare.read_reference_labels(contract, dict(passed=True,label_access_authorized_after_gate=True))
    assert y.dtype == np.float32 and y.shape == (2,12)
    assert y[0,0] == np.float32(.2) and y[1,0] == np.float32(.5)


def test_checkpoint_contract_checks_epoch_provenance_without_scores():
    contract = dict(checkpoint_expected=dict(epoch=7,window_policy="within-slot-valid"),
                    checkpoint_provenance=dict(checkpoint_selection="fixed-final"))
    meta = dict(epoch=7, window_policy="within-slot-valid", provenance=contract["checkpoint_provenance"],
                arbitrary_old_score=object())
    compare.validate_checkpoint(meta, contract)
    meta["epoch"] = 6
    with pytest.raises(ValueError, match="epoch"):
        compare.validate_checkpoint(meta, contract)


def test_runtime_and_hardware_changes_block_comparison():
    runtime = dict(requested_amp="fp16",effective_amp="fp16",backends={"tf32":False})
    environment = {key: "same" for key in compare.ENV_KEYS}
    compare.validate_runtime(runtime, runtime, environment, environment)
    altered = dict(environment,gpu="T4")
    with pytest.raises(ValueError, match="environment"):
        compare.validate_runtime(runtime,runtime,altered,environment)
    with pytest.raises(ValueError, match="runtime"):
        compare.validate_runtime(dict(runtime,effective_amp="bf16"),runtime,environment,environment)


def test_model_state_hash_handles_scalar_buffers_and_detects_change():
    import torch
    model = torch.nn.BatchNorm1d(2)
    model.eval()
    first = compare.state_digest(model)
    assert compare.state_digest(model) == first
    with torch.no_grad():
        model.num_batches_tracked.add_(1)
    assert compare.state_digest(model) != first
    with torch.no_grad():
        model.weight[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compare.state_digest(model)


@pytest.mark.parametrize("value", ["file:///bad", "workspace://../bad", "workspace://C:/bad"])
def test_workspace_reference_cannot_escape(value):
    with pytest.raises(ValueError):
        compare.unuri(value)


def test_import_is_passive(tmp_path):
    code = f"""
import sys
from pathlib import Path
sys.path.insert(0,{str(ROOT / 'tools')!r})
def audit(event,args):
 if event=='open' and isinstance(args[0],str):
  p=args[0].replace('\\\\','/')
  assert not any('/'+part+'/' in p for part in ('storage','artifacts','.local')),p
sys.addaudithook(audit)
import exp081_input_compare
assert 'torch' not in sys.modules
assert 'raptor_train' not in sys.modules
assert 'raptor_infer' not in sys.modules
"""
    done = subprocess.run([sys.executable,"-X","utf8","-c",code],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("failure", ["second_pass", "unchanged_difference"])
def test_run_cannot_open_labels_after_partial_or_mismatched_pair(tmp_path, monkeypatch, failure):
    ids, changed, baseline, candidate, timing, candidate_timing = pair()
    output = tmp_path / "predictions"
    checkpoint = tmp_path / "checkpoint.bin"; checkpoint.write_bytes(b"synthetic placeholder")
    contract_path = tmp_path / "contract.json"
    contract = dict(bindings={}, output="output", ids=ids, changed_ids=changed,
        report_groups=["g1","g2","g3"], diagnostic_groups={}, expected_runtime={}, expected_environment={},
        checkpoint="checkpoint", checkpoint_sha256=compare.prep.sha(checkpoint), checkpoint_expected={},
        checkpoint_provenance={}, selection_map_sha256="synthetic")
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(compare, "validate_contract", lambda value: None)
    monkeypatch.setattr(compare, "validate_bindings", lambda value: None)
    monkeypatch.setattr(compare, "unuri", lambda value: output if value == "output" else checkpoint)
    monkeypatch.setattr(compare, "collect_cpu", lambda: ({},{},ids,changed,{},{}))
    monkeypatch.setattr(compare, "load_corpora", lambda *args: (SimpleNamespace(contract="base"),SimpleNamespace(contract="overlay")))
    monkeypatch.setattr(compare, "read_metadata", lambda *args: (contract["report_groups"],{}))
    monkeypatch.setattr(compare, "state_digest", lambda model: "fixed-state")
    monkeypatch.setattr(compare, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(compare, "read_reference_labels", lambda *args: pytest.fail("labels opened despite failed pair"))
    model = SimpleNamespace(eval=lambda: None)
    calls = []
    def predict(*args, **kwargs):
        calls.append(args[1].contract)
        if len(calls) == 1:
            return baseline, timing
        if failure == "second_pass":
            raise RuntimeError("synthetic second-pass failure")
        wrong = candidate.copy(); wrong[0,0] = .5
        return wrong, candidate_timing
    runtime = SimpleNamespace(configure_runtime=lambda *args:{"backends":{}}, runtime_snapshot=lambda:{}, predict=predict)
    common = SimpleNamespace(environment=lambda:{}, load_checkpoint=lambda path:(model,{"provenance":{}}))
    monkeypatch.setattr(compare, "adapters", lambda:(common,runtime))
    with pytest.raises((ValueError,RuntimeError)):
        compare.run(contract_path, compare.prep.sha(contract_path))
    assert calls == ["base","overlay"]
    failure_receipt = json.loads((output / "FAILED.json").read_text())
    assert failure_receipt["labels_opened"] is False
    assert not (output / "report.json").exists() and not (output / "comparison-gate.json").exists()
