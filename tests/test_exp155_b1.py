"""Amendment B1 CPU fakes. No production rows, network, or CUDA."""
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
from tools import exp155_oai_labels_v2 as labels
from tools import exp155_own0949_oai_v11 as driver
from tools import exp155_own0949_oai_release_v12 as release
import exp155_own0949_oai_v11 as fit


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*_args, **_kwargs):
        raise AssertionError("CPU test touched CUDA")
    for name in ("_lazy_init", "init", "manual_seed_all", "is_available", "set_device"):
        monkeypatch.setattr(torch.cuda, name, fail)
    yield
    assert not torch.cuda.is_initialized()


def cartilage(target, value):
    return {"MCM" + region: value for region in extlabels.CART_REGIONS[target]}


@pytest.mark.parametrize("target", ["PF OA", "Lateral OA"])
def test_cartilage_v2_without_osteophyte_or_xray(target):
    healthy = cartilage(target, 0)
    assert labels.oa_v2(healthy, target) == 0
    positive = dict(healthy)
    positive["MCM" + extlabels.CART_REGIONS[target][0]] = 2.2
    assert labels.oa_v2(positive, target) == 1
    positive["XRKL"] = 0
    positive["XRJSL"] = 0
    assert labels.oa_v2(positive, target) == 1
    small = dict(healthy)
    small["MCM" + extlabels.CART_REGIONS[target][0]] = 1.1
    small["MOS" + extlabels.OST_REGIONS[target][0]] = 3
    assert labels.oa_v2(small, target) is None
    partial = dict(healthy)
    partial["MCM" + extlabels.CART_REGIONS[target][0]] = 2.0
    assert labels.oa_v2(partial, target) is None
    missing = dict(healthy)
    del missing["MCM" + extlabels.CART_REGIONS[target][0]]
    assert labels.oa_v2(missing, target) is None
    invalid = dict(healthy)
    invalid["MCM" + extlabels.CART_REGIONS[target][0]] = 9.9
    assert labels.oa_v2(invalid, target) is None
    assert labels.oa_v2({"XRKL": 4, "XRJSL": 3}, target) is None


@pytest.mark.parametrize("primary,poma,expected", [
    ({"MSYIC": 0}, {"MSYIC": 3}, (0, "moaks")),
    ({"MSYIC": 2}, {"MSYIC": 0}, (1, "moaks")),
    ({"MSYIC": None}, {"MSYIC": 3}, (1, "poma")),
    ({"MSYIC": 9}, {"MSYIC": 1}, (None, "unknown")),
    ({"MSYIC": None}, {"MSYIC": None}, (None, "unknown")),
])
def test_synovitis_target_fallback(primary, poma, expected):
    assert labels.synovitis_v2(primary, poma) == expected


def test_receipt_hashes_and_reader_precedence(tmp_path):
    inputs = {}
    for name in labels.SOURCE_NAMES:
        path = tmp_path / name
        path.write_text("fake clinical input", encoding="ascii")
        inputs[str(path)] = labels.sha256(path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(dict(contract="EXP136-v1",
        tool_sha256=labels.sha256(ROOT / "tools/extlabels.py"), inputs_sha256=inputs)), encoding="utf-8")
    assert len(labels.clinical_inputs(receipt)) == 3
    next(iter(tmp_path.glob("kMRI_POMA_TKR*"))).write_text("drift", encoding="ascii")
    with pytest.raises(ValueError, match="hash changed"):
        labels.clinical_inputs(receipt)
    key = "OAI_1_R_V00"
    records = [extlabels.Reading(key, {"MSYIC": score}, {}, project, "1", file, reader)
        for project, score, file, reader in (("TKR", 3, "tkr", "R1"),
                                               ("INCOA", 0, "incoa-r2", "R2"),
                                               ("INCOA", 2, "incoa-r1", "R1"))]
    chosen, _ = extlabels.select_readings(records, ("INCOA", "TKR"))
    assert chosen[key].scores["MSYIC"] == 2


def test_build_join_masks_xray_and_aggregate_only(tmp_path, monkeypatch, capsys):
    a, b, c = "OAI_1_R_V00", "OAI_2_L_V00", "OAI_3_R_V00"
    manifest = tmp_path / "cache/manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(labels, "check_cache", lambda *_a, **_k: {"ids": [a, b, c]})
    monkeypatch.setattr(labels, "clinical_inputs", lambda *_a: {name: (tmp_path / name, "0" * 64)
                                                              for name in labels.SOURCE_NAMES})
    primary = {a: SimpleNamespace(scores={**cartilage("PF OA", 0), **cartilage("Lateral OA", 0), "MSYIC": 1}),
               b: SimpleNamespace(scores={"XRKL": 4, "XRJSL": 3, "MSYIC": None}),
               c: SimpleNamespace(scores={"MSYIC": None})}
    poma = {b: SimpleNamespace(scores={"MSYIC": 3}),
            "OAI_2_L_V01": SimpleNamespace(scores={"MSYIC": 0}),
            "OAI_4_R_V00": SimpleNamespace(scores={"MSYIC": 3})}
    monkeypatch.setattr(labels, "select_sources", lambda *_a: (primary, poma))
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    out = tmp_path / "labels/v2.csv"
    result = labels.build_labels(SimpleNamespace(oai_cache=manifest.parent, receipt=receipt, out=out))
    assert result["admitted_oai_n"] == 2 and result["excluded_all_unknown_n"] == 1
    assert result["counts"]["PF OA"] == dict(positive=0, negative=1, unknown=2, observed=1)
    assert result["counts"]["Lateral OA"]["observed"] == 1
    assert result["counts"]["Synovitis"] == dict(positive=1, negative=1, unknown=1, observed=2)
    assert result["source_share_before_pos_weight"]["Synovitis"]["oai"] == 2 / 4349
    assert result["mixed_n"] == 4349 and result["updates_per_epoch"] == 272
    assert result["total_updates"] == 4352
    assert a not in capsys.readouterr().out
    with out.open(encoding="utf-8", newline="") as stream:
        rows = {row["study_key"]: row for row in csv.DictReader(stream)}
    assert set(rows) == {a, b}
    assert rows[b]["mask_Lateral OA"] == "0" and rows[b]["mask_Synovitis"] == "1"
    assert rows[b]["Synovitis"] == "1"
    qa = json.loads(out.with_suffix(".qa.json").read_text(encoding="utf-8"))
    assert a not in json.dumps(qa) and b not in json.dumps(qa)
    monkeypatch.setattr(driver, "OAI_LABELS", out)
    monkeypatch.setattr(labels, "DEFAULT_RECEIPT", receipt)
    checked = driver.verify_v2_labels(out, 3, labels.sha256(manifest))
    assert checked[3] == qa
    out.with_suffix(".qa.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance/hash mismatch"):
        driver.verify_v2_labels(out, 3, labels.sha256(manifest))


def test_fixed_denominator_and_rsna_equivalence():
    logits = torch.linspace(-1, 1, 24).reshape(2, 12).requires_grad_()
    targets = torch.zeros_like(logits)
    weights = torch.linspace(1, 3, 12)
    mask = torch.zeros_like(logits)
    mask[:, 6] = 1
    actual = fit.masked_study_bce(logits, targets, mask, weights)
    expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights, reduction="none")[:, 6].sum() / 24
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, logits)[0]
    assert (gradient[mask == 0] == 0).all()
    full = torch.ones_like(mask)
    torch.testing.assert_close(fit.masked_study_bce(logits, targets, full, weights),
                               F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights))
    with pytest.raises(ValueError, match="at least one"):
        fit.masked_study_bce(logits, targets, torch.zeros_like(mask), weights)
    assert fit.tiny_validation()["masked_loss_verified"] is True


def test_registration_version_paths_cohort_and_weight_policy():
    assert driver.VERSION == "1.1.0"
    assert driver.PROTOCOL.name == "manifest.json" and driver.PROTOCOL.parent.name == "protocol-v1_1"
    assert driver.RUN_DIRS["local"].endswith("S42v1_1")
    assert driver.RECIPE["loss"] == "fixed-12-denominator-masked-bce-then-study-mean"
    assert driver.RECIPE["class_weights"] == "RSNA-training-only"
    assert "tools/exp155_oai_data.py" in driver.CODE
    assert "tools/exp155_oai_labels_v2.py" in driver.CODE
    assert "tools/exp155_own0949_oai_release_v11.py" in driver.CODE
    assert release.VERSION == "1.2.0" and release.RUN == driver.RUN_DIRS["local"]
    assert release.PROTOCOL == driver.PROTOCOL
    assert release.OWN_NAME.endswith("oai-bn.pt")
    queue = driver.parser().parse_args(["queue-json", "--after", "probe", "--probe-report", "probe.json"])
    assert queue.out.replace("\\", "/").endswith("protocol-v1_1/queue-s42v1_1.json")  # B: Windows separators
    rsna_ids = ["r1", "r2"]
    rsna_labels = {uid: [0.2] * 12 for uid in rsna_ids}
    oa, unknown = "OAI_1_R_V00", "OAI_2_L_V00"
    oai_labels = {oa: [0.] * 12, unknown: [0.] * 12}
    masks = {oa: [0] * 12, unknown: [0] * 12}
    masks[oa][6] = 1
    ids, _, _, admitted = driver.combine_cohorts(rsna_ids, rsna_labels, ["gold"],
        [oa, unknown], oai_labels, masks)
    assert ids == rsna_ids + [oa] and admitted == [oa]
    with pytest.raises(ValueError, match="Gold leakage"):
        driver.combine_cohorts(rsna_ids, rsna_labels, [oa], [oa, unknown], oai_labels, masks)
