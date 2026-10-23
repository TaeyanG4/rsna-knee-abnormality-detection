"""EXP163 CPU-only contract tests with synthetic knees; no CUDA or network."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training/src"))

from tools import extlabels
from tools import exp163_oai_labels as labels
from tools import exp163_own0949_oai_ext as driver
from tools import exp163_own0949_oai_ext_release as release
import exp163_own0949_oai_ext as fit


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*_args, **_kwargs):
        raise AssertionError("CPU test touched CUDA")
    for name in ("_lazy_init", "init", "manual_seed_all", "is_available", "set_device"):
        monkeypatch.setattr(torch.cuda, name, fail)
    yield
    assert not torch.cuda.is_initialized()


def cartilage(target, value=0):
    return {"MCM" + region: value for region in extlabels.CART_REGIONS[target]}


def test_eight_target_mapping_and_four_masked():
    primary = {**cartilage("Medial OA"), **cartilage("Lateral OA"),
               **cartilage("PF OA"), "MACLTR": 2, "MEFFWK": 2,
               "MSYIC": 0, "MPOPCYS": 1}
    for side in ("M", "L"):
        primary.update({f"MMT{side}{part}": 0 for part in ("A", "B", "P")})
        primary[f"MMRT{side}"] = 0
    primary["MMTMB"] = 3
    primary["MCMFMC"] = 2.2
    primary["MOSFMC"] = 0
    primary["XRKL"] = 0
    values = labels.map_extended(primary, {"MSYIC": 3, "MACLTR": 0})
    assert values == {"ACL": 1, "Medial Meniscus": 1, "Lateral Meniscus": 0,
                      "Medial OA": 1, "Lateral OA": 0, "PF OA": 0,
                      "Effusion": 1, "Synovitis": 0}
    assert "Baker's" not in values
    assert set(labels.TARGETS_OAI).isdisjoint(labels.UNSUPERVISED)
    assert "Baker's" in labels.UNSUPERVISED


@pytest.mark.parametrize("grade, expected", [(0, 0), (1, None), (2, 1),
                                               (None, None), (9, None), ("bad", None)])
def test_acl_strict_grade_for_primary_and_poma(grade, expected):
    assert labels.map_extended({"MACLTR": grade, "MPOPCYS": 0}, None)["ACL"] == expected
    assert labels.map_extended(None, {"MACLTR": grade, "MPOPCYS": 1})["ACL"] == expected
    assert labels.map_extended({"MACLTR": grade, "MPOPCYS": 1}, None).get("Baker's") is None


def test_fallback_is_record_level_for_added_targets_and_target_level_for_synovitis():
    poma = {"MACLTR": 2, "MEFFWK": 3, "MSYIC": 2,
            **cartilage("Medial OA")}
    absent = labels.map_extended(None, poma)
    assert absent["ACL"] == 1 and absent["Effusion"] == 1
    assert absent["Medial OA"] == 0 and absent["Synovitis"] == 1
    assert "Baker's" not in absent
    partial = labels.map_extended({"MACLTR": None, "MEFFWK": 9, "MSYIC": None}, poma)
    assert partial["ACL"] is None and partial["Effusion"] is None
    assert partial["Synovitis"] == 1
    invalid = labels.map_extended({"MSYIC": 9}, poma)
    assert invalid["Synovitis"] is None


def test_aggregate_qa_all_targets_and_share_flag():
    # QA counts are over cache membership, including all-unknown exclusions.
    mapped = {str(i): {"ACL": int(i < 1200), "PF OA": None} for i in range(2000)}
    qa = labels.aggregate_qa(mapped, 2399)
    assert qa["counts"]["ACL"] == dict(positive=1200, negative=800, unknown=399,
                                      admitted_unknown=0, observed=2000)
    assert qa["counts"]["MCL"]["unknown"] == 2399
    assert qa["per_epoch_exposure"]["ACL"]["oai_positive_cells_per_epoch"] == 1200
    assert qa["per_epoch_exposure"]["ACL"]["oai_negative_cells_per_epoch"] == 800
    assert qa["per_epoch_exposure"]["ACL"]["oai_positive_fraction_of_mixed_observed_cells"] == 1200 / 6347
    assert qa["per_epoch_exposure"]["ACL"]["oai_negative_fraction_of_mixed_observed_cells"] == 800 / 6347
    assert qa["per_epoch_exposure"]["Baker's"]["oai_observed_cells_per_epoch"] == 0
    assert qa["per_epoch_exposure"]["Baker's"]["rsna_observed_cells_per_epoch"] == 4347
    assert qa["source_share_before_pos_weight"]["ACL"]["exceeds_30_percent"]
    assert "ACL" in qa["flagged_targets"]
    assert qa["source_share_before_pos_weight"]["ACL"]["oai"] == 2000 / 6347


def test_fake_label_build_aggregate_only(tmp_path, monkeypatch, capsys):
    keys = ("OAI_1_R_V00", "OAI_2_L_V00", "OAI_3_R_V00", "OAI_4_L_V00",
            "OAI_5_R_V00")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(labels, "check_cache", lambda *_a, **_kw: {"ids": list(keys)})
    monkeypatch.setattr(labels, "clinical_inputs", lambda *_a: {
        name: (tmp_path / name, "0" * 64) for name in labels.SOURCE_NAMES})
    monkeypatch.setattr(labels, "select_sources", lambda *_a: (
        {keys[0]: SimpleNamespace(scores={"MACLTR": 1, "MSYIC": 0}),
         keys[2]: SimpleNamespace(scores={"MPOPCYS": 1}),
         keys[3]: SimpleNamespace(scores={"MACLTR": 0}),
         keys[4]: SimpleNamespace(scores={"MACLTR": 1})},
        {keys[1]: SimpleNamespace(scores={"MACLTR": 2, "MSYIC": 2})}))
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    out = tmp_path / "labels.csv"
    result = labels.build_labels(SimpleNamespace(oai_cache=cache, receipt=receipt, out=out))
    assert result["admitted_oai_n"] == 3
    assert result["excluded_all_unknown_n"] == 2
    assert result["mixed_n"] == 4350
    assert result["updates_per_epoch"] == 272
    assert result["total_updates"] == 4352
    assert result["counts"]["ACL"]["positive"] == 1
    assert result["counts"]["ACL"]["negative"] == 1
    assert result["counts"]["Synovitis"]["positive"] == 1
    assert result["counts"]["Baker's"]["observed"] == 0
    assert result["counts"]["MCL"]["unknown"] == 5
    assert all(key not in capsys.readouterr().out for key in keys)
    with out.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 and all(row["mask_MCL"] == "0" and row["mask_Baker's"] == "0" for row in rows)
    assert all(row["study_key"] not in (keys[2], keys[4]) for row in rows)
    assert all(key not in json.dumps(result) for key in keys)
    _, masks = driver.read_oai_labels(out)
    assert all(mask[driver.TARGETS.index("Baker's")] == 0 for mask in masks.values())
    rows[0]["Baker's"], rows[0]["mask_Baker's"] = "1", "1"
    tampered = tmp_path / "tampered.csv"
    with tampered.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="outside the eight registered targets"):
        driver.read_oai_labels(tampered)


def test_driver_masks_and_fixed_denominator():
    assert driver.OAI_TARGETS == labels.TARGETS_OAI
    assert {driver.TARGETS.index(t) for t in driver.OAI_TARGETS} == {0, 2, 3, 4, 5, 6, 7, 8}
    assert driver.RECIPE["loss"] == "fixed-12-denominator-masked-bce-then-study-mean"
    assert driver.RECIPE["class_weights"] == "RSNA-training-only"
    assert driver.RUN_DIRS["local"].endswith("EXP163_OWN0949_OAI_EXT/S42")
    logits = torch.zeros(2, 12, requires_grad=True)
    ys = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    mask[:, [0, 4, 8]] = 1
    actual = fit.masked_study_bce(logits, ys, mask, torch.ones(12))
    expected = F.binary_cross_entropy_with_logits(logits[:, [0, 4, 8]],
                                                   ys[:, [0, 4, 8]], reduction="sum") / 24
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, logits)[0]
    assert (gradient[mask == 0] == 0).all()
    assert fit.tiny_validation()["masked_loss_verified"]


def test_queue_interface_and_module_command():
    args = driver.parser().parse_args(["queue-json", "--after", "EXP157_TAG",
                                       "--probe-report", "probe.json"])
    assert args.out.replace("\\", "/").endswith("protocol-v1/queue-s42.json")
    assert driver.VERSION == "1.1.0"
    assert labels.VERSION == driver.VERSION
    assert driver.OAI_LABELS.name == "exp163_oai_masked_v1_1.csv"
    assert set(driver.CODE) >= {"tools/exp163_oai_labels.py",
                                "training/src/exp163_own0949_oai_ext.py"}
    probe = driver.parser().parse_args(["probe-queue-json", "--after", "EXP157_TAG"])
    assert probe.out.replace("\\", "/").endswith("protocol-v1/probe-queue-s42.json")


def test_release_equal_three_rank_ties_and_paths():
    import numpy as np
    own = np.tile(np.asarray([[.1], [.3], [.3], [.9]]), (1, 12))
    public = np.tile(np.asarray([[.9], [.3], [.3], [.1]]), (1, 12))
    member = np.tile(np.asarray([[.2], [.2], [.8], [.8]]), (1, 12))
    actual = release.output_values([own, public, member], "blend", None)
    expected = (np.asarray([0, .5, .5, 1]) + np.asarray([1, .5, .5, 0])
                + np.asarray([1 / 6, 1 / 6, 5 / 6, 5 / 6])) / 3
    np.testing.assert_allclose(actual[:, 0], expected)
    assert release.PROTOCOL == driver.PROTOCOL
    assert release.RUN == driver.RUN_DIRS["local"]
    assert set(release.KERNELS) == {"own", "blend"}
    assert release.ADAPTER_NAME == "exp155_own0949_oai_release_v12.py"
    assert release.VERSION == "1.1.0"
    assert release.sha(release.adapter_path) == release.ADAPTER_SHA
    assert "release_v11.py" not in Path(release.TOOL).read_text(encoding="utf-8")
