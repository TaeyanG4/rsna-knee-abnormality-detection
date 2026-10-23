"""D01 packaging and release checks only; no GPU, network or Kaggle writes."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("d01_package", Path(__file__).resolve().parents[1] / "tools/d01_package.py")
d01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d01)


def frame(ids, values):
    return pd.DataFrame({d01.UID: ids, **{target: values for target in d01.LABELS}})


def test_blend_aligns_uids_before_fixed_rank_mix_and_averages_ties():
    anchor = frame(["a", "b", "c", "d"], [.1, .2, .2, .4])
    r2 = frame(["d", "c", "b", "a"], [.1, .2, .3, .4])
    result = d01.blend_frames(anchor, r2, ["a", "b", "c", "d"])
    np.testing.assert_allclose(result[d01.LABELS].to_numpy(),
                               np.repeat(np.array([.325, .6375, .6125, .925])[:, None], 12, axis=1))
    assert result[d01.UID].tolist() == ["a", "b", "c", "d"]


@pytest.mark.parametrize("case", ["duplicate", "missing", "nan", "range", "wrong_order"])
def test_blend_rejects_incomplete_or_invalid_predictions(case):
    anchor = frame(["a", "b"], [.2, .8]); r2 = anchor.copy()
    if case == "duplicate":
        r2.loc[1, d01.UID] = "a"
    elif case == "missing":
        r2 = r2.iloc[:1]
    elif case == "nan":
        r2.loc[0, "ACL"] = np.nan
    elif case == "range":
        r2.loc[0, "ACL"] = 1.1
    else:
        r2 = r2[[d01.UID, *reversed(d01.LABELS)]]
    with pytest.raises(ValueError):
        d01.blend_frames(anchor, r2, ["a", "b"])


def test_preflight_separates_input_identity_from_numerical_tolerance():
    hashes = [{"uid": "a", "sha256": "input"}]
    studies = [{"uid": "a", "volume_sha256": "volume", "mask_sha256": "mask"}]
    expected = {"ids": ["a"], "input_tensor_hashes": hashes, "study_hashes": studies,
                "tolerances": {"f0_max_abs": .002, "f0_mean_abs": .0005}}
    receipt = {"inference": {"input_tensor_hashes": hashes, "status": "COMPLETE", "fp32_fallbacks": 0,
                             "neutral_fills": 0, "failures": []}, "study_receipts": studies}
    predictions = np.full((5, 1, 12), .5)
    assert d01.check_preflight(expected, receipt, predictions, ["a"], predictions[0])["status"] == "PASS"
    different = copy.deepcopy(receipt); different["inference"]["input_tensor_hashes"][0]["sha256"] = "changed"
    assert d01.check_preflight(expected, different, predictions, ["a"], predictions[0])["status"] == "FAILED"
    reference = predictions[0].copy(); reference[0, 0] += .003
    report = d01.check_preflight(expected, receipt, predictions, ["a"], reference)
    assert report["status"] == "FAILED" and report["checks"]["f0_max_abs"] is False
    predictions[0, 0, 0] = np.nan
    report = d01.check_preflight(expected, receipt, predictions, ["a"], reference)
    assert report["status"] == "FAILED"
    json.dumps(report, allow_nan=False)


def test_staging_preserves_anchor_cells_and_pins_every_asset(tmp_path):
    root = tmp_path / "source"; root.mkdir()
    anchor_dir = root / "kaggle/packages/KNEE_ANCHOR_0943"; anchor_dir.mkdir(parents=True)
    anchor = {"nbformat": 4, "nbformat_minor": 4, "metadata": {}, "cells": [d01._cell("print('anchor unchanged')\n")]}
    d01.dump(anchor_dir / "anchor.ipynb", anchor)
    d01.dump(anchor_dir / "kernel-metadata.json", {"id": "owner/anchor", "code_file": "anchor.ipynb",
             "dataset_sources": ["source/weights"], "kernel_sources": ["source/train"], "competition_sources": ["rsna-knee-abnormality-detection"]})
    for name in d01.SOURCES:
        path = root / "training/src" / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# passive fixture\n")
    for fold in range(5):
        run = root / "artifacts/experiments" / f"EXP072_R2MIX_R34_F{fold}"; run.mkdir(parents=True)
        (run / "model_best.pt").write_bytes(f"checkpoint{fold}".encode())
        d01.dump(run / "summary.json", {"fold": fold, "backbone": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "img": 224,
                 "contract": "raptor64_336_c140_s02", "args": {"norm": "imagenet", "k_eval": 24}})
    wheels = tmp_path / "wheels"; wheels.mkdir()
    for name in ("timm-1.0.19-py3-none-any.whl", "pydicom-3.0.2-py3-none-any.whl", "opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.whl"):
        (wheels / name).write_bytes(b"wheel-fixture")
    local_csv = tmp_path / "reference.csv"; frame(["a"], [.5]).to_csv(local_csv, index=False)
    local_receipt = tmp_path / "reference.json"
    d01.dump(local_receipt, {"status": "COMPLETE", "norm": "imagenet", "window_policy": "legacy", "runtime": {"effective_amp": "fp16"},
             "checkpoint_sha256": d01.sha256(root / "artifacts/experiments/EXP072_R2MIX_R34_F0/model_best.pt"), "environment": {},
             "inference": {"input_tensor_hashes": [{"uid": "a", "sha256": "tensor"}], "ordered_inputs_sha256": "all"},
             "study_receipts": [{"uid": "a", "volume_sha256": "vol", "mask_sha256": "mask"}]})
    out = tmp_path / "stage"
    summary = d01.build(root, out, wheels, local_receipt, local_csv)
    assert summary["anchor_original_unchanged"] and summary["state"] == "STAGED_ONLY_NOT_UPLOADED_OR_EXECUTED"
    candidate = json.loads((out / "candidate/d01-candidate.ipynb").read_text())
    assert candidate["cells"][1:-1] == anchor["cells"]
    manifest = d01._validate_assets(out / "assets")
    assert manifest["blend"] == {"anchor_rank": .9, "r2_rank_of_probability_mean": .1, "rank_method": "average", "final_rerank": False}
    for kind in ("candidate", "preflight"):
        metadata = json.loads((out / kind / "kernel-metadata.json").read_text())
        assert metadata["is_private"] and not metadata["enable_internet"]
        notebook = json.loads((out / kind / metadata["code_file"]).read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "notebook", "exec")
    with pytest.raises(FileExistsError):
        d01.build(root, out, wheels, local_receipt, local_csv)
    (out / "assets/weights/r2_f0_best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset changed"):
        d01._validate_assets(out / "assets")


def test_preflight_worker_uses_mean_raw_receipt_api_without_gpu(tmp_path, monkeypatch):
    """Exercise package wiring with passive fake inference; catch tuple-order drift."""
    assets = tmp_path / "assets"; assets.mkdir()
    work = tmp_path / "work"
    hashes = [{"uid": "a", "sha256": "tensor"}]
    studies = [{"uid": "a", "volume_sha256": "vol", "mask_sha256": "mask"}]
    inference = {"status": "COMPLETE", "input_tensor_hashes": hashes, "fp32_fallbacks": 0,
                 "neutral_fills": 0, "failures": []}
    manifest = {"files": {}, "members": [{"name": f"F{i}"} for i in range(5)], "contract": "contract"}
    d01.dump(assets / d01.MARKER, manifest)
    d01.dump(assets / "validation/expected_test3.json", {"ids": ["a"], "input_tensor_hashes": hashes,
             "study_hashes": studies, "tolerances": {"f0_max_abs": .002, "f0_mean_abs": .0005}})
    frame(["a"], [.2]).to_csv(assets / "validation/local_f0_test3.csv", index=False)
    raw = np.stack([np.full((1, 12), value, dtype=np.float32) for value in [.2, .3, .4, .5, .6]])
    mean = raw.mean(axis=0)

    class Source:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *exc): pass

    def write_predictions(path, ids, probabilities):
        output = pd.DataFrame(probabilities, columns=d01.LABELS)
        output.insert(0, d01.UID, ids); output.to_csv(path, index=False)

    monkeypatch.setattr(d01, "_activate_offline_environment", lambda *args: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "Tesla T4")))
    monkeypatch.setitem(sys.modules, "raptor_common", SimpleNamespace(environment=lambda: {"test": "CPU fake"}, write_predictions=write_predictions))
    monkeypatch.setitem(sys.modules, "raptor_dicom", SimpleNamespace(DirectoryDicomSource=Source,
                        DicomCorpus=lambda *args: SimpleNamespace(ids=["a"], receipts={"a": studies[0]})))
    monkeypatch.setitem(sys.modules, "raptor_ensemble", SimpleNamespace(load_members=lambda path: (["fake"] * 5, manifest),
                        predict_ensemble=lambda *args, **kwargs: (mean, raw, inference)))
    result = d01.run_r2(assets, tmp_path / "competition", work, preflight=True)
    assert result["status"] == "COMPLETE"
    np.testing.assert_allclose(pd.read_csv(work / "r2-mean.csv")[d01.LABELS].to_numpy(), mean)
    saved = np.load(work / "r2-members.npz")
    np.testing.assert_array_equal(saved["member_probabilities"], raw)
    assert json.loads((tmp_path / d01.PRECHECK).read_text())["status"] == "PASS"
