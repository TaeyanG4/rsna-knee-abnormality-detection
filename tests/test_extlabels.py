"""Tiny synthetic EXP136 tests: no source data, images, network or GPU imports."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import extlabels as e


def write_csv(path, fields, rows, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("value", [None, "", ".", ".A", ".: Missing", ".D: not done",
                                    "-1", "-9: invalid", "NaN", "inf", "unknown"])
def test_missing_never_becomes_a_negative_or_positive(value):
    assert e.code(value) is None
    assert e.map_moaks({"MACLTR": value, "MEFFWK": value})["ACL"] is None


def test_grades_and_missing_subregions():
    assert e.code("2: Complete tear") == 2
    assert e.map_moaks({"MACLTR": 9})["ACL"] is None
    assert e.map_moaks({"MACLTR": 1})["ACL"] == 1
    assert e.map_moaks({"MACLRP": 1})["ACL"] is None
    result = e.map_moaks({"MMTMA": 1, "MMTMB": 0, "MMTMP": 0, "MMRTM": 0,
                          "MEFFWK": 1, "MSYIC": 2, "MPOPCYS": 1, "MBMSFMC": 3})
    assert result["Medial Meniscus"] == 0
    assert result["Lateral Meniscus"] is None
    assert result["Effusion"] == 0 and result["Synovitis"] == 1
    assert result["Baker's"] == 1
    assert result["MCL"] is None and result["Contusion"] is None and result["Fracture"] is None
    assert e.map_moaks({"MMTMA": 0, "MMTMB": 0, "MMTMP": 0})["Medial Meniscus"] is None
    assert e.map_moaks({"MMTMB": 3})["Medial Meniscus"] == 1
    assert e.map_moaks({"MMRTM": 1})["Medial Meniscus"] == 1
    assert e.map_moaks({"MEFFWK": 3})["Synovitis"] is None


def test_cartilage_decimal_and_compartment_definition():
    assert e.cartilage_parts("3.1: area, full") == (3, 1)
    assert e.cartilage_parts(3) == (3, 0)
    for value in (0.5, 1.15, 3.5, 1.2, 99, None):
        assert e.cartilage_parts(value) is None
    scores = {"MCMFMA": 2.1, "MOSFMA": 1}
    result = e.map_moaks(scores)
    assert result["PF OA"] == 1 and result["Medial OA"] is None
    scores["MCMFMA"] = 2.0
    assert e.map_moaks(scores)["PF OA"] is None
    normal = {"MCM" + r: 0 for r in e.CART_REGIONS["Medial OA"]}
    normal.update({"MOS" + r: 0 for r in e.OST_REGIONS["Medial OA"]})
    assert e.map_moaks(normal)["Medial OA"] == 0
    normal.pop("MCMTMC")
    assert e.map_moaks(normal)["Medial OA"] is None
    normal.update({"MCMFMC": 1.1, "MOSFMC": 1})
    assert e.map_moaks(normal)["Medial OA"] == 1  # known positive with another missing region


def test_xray_localization_borderline_and_mri_disagreement():
    result = e.map_xray({"XRKL": 2, "XRJSM": 1, "XRJSL": 0})
    assert result["Medial OA"] == 1 and result["Lateral OA"] is None
    assert result["PF OA"] is None
    assert e.map_xray({"XRKL": 0, "XRJSM": 0, "XRJSL": 0})["Lateral OA"] == 0
    for scores in ({"XRKL": 1, "XRJSM": 0, "XRJSL": 0},
                   {"XRKL": 2, "XRJSM": 0.5, "XRJSL": 0},
                   {"XRKL": 0, "XRJSM": 0}, {"XRKL": 99, "XRJSM": 3}):
        assert e.map_xray(scores)["Medial OA"] is None
    assert e.merge_oa(1, 0) == (None, 1)
    assert e.merge_oa(0, None) == (0, 0)
    assert e.merge_oa(None, 1) == (1, 0)


def test_worms_is_not_moaks_grade_scale():
    r = e.map_worms({"MWACL": 2, "MWMCL": 3, "MWMTMA": 2, "MWPOPCY": 1,
                     "MWEFFWK": 2, "MWBMFM": 3})
    assert r["ACL"] is None and r["MCL"] == 1 and r["Medial Meniscus"] == 1
    assert r["Baker's"] == 0 and r["Effusion"] == 1
    assert r["Synovitis"] is None and r["Contusion"] is None


def test_fixed_project_and_version_precedence_is_order_independent():
    key = "OAI_123_R_V00"
    records = [e.Reading(key, {"MACLTR": 2}, {"MACLTR": 2}, "65", "0.9", "a.txt"),
               e.Reading(key, {"MACLTR": 0}, {"MACLTR": 0}, "22", "0.1", "b.txt"),
               e.Reading(key, {"MACLTR": 1}, {"MACLTR": 1}, "22", "0.2", "b.txt")]
    for rs in (records, records[::-1]):
        selected, counts = e.select_readings(rs, e.MOAKS_ORDER)
        assert selected[key].scores["MACLTR"] == 1
        assert counts[key] == 3
    poma = [e.Reading(key, {}, {}, "INCOA", "1", "a.txt", reader=r) for r in ("", "R2", "R1")]
    assert e.select_readings(poma, ("INCOA", "TKR"))[0][key].reader == "R1"


def test_oai_integrated_poma_actual_visit_and_native_only_sources(tmp_path):
    mr = tmp_path / "clinical/extracted/mr_assess/Semi-Quant Scoring_ASCII"
    xr = tmp_path / "clinical/extracted/xray/X-Ray Image Assessments_ASCII/XR Knee Semi-Quant"
    write_csv(mr / "kMRI_SQ_MOAKS_BICL00.txt",
              ["ID", "SIDE", "READPRJ", "VERSION", "V00MACLTR", "V00MCMNTS"],
              [{"ID": "101", "SIDE": "1: Right", "READPRJ": "22", "VERSION": "0.2",
                "V00MACLTR": "0: None", "V00MCMNTS": "synthetic forbidden comment"}], "|")
    write_csv(mr / "kMRI_POMA_IncOA_MOAKS_BICL.txt",
              ["id", "side", "version", "visit", "iwave", "Reader", "OtherACLTear", "comments"],
              [{"id": "101", "side": "1", "version": "1", "visit": "0", "iwave": "6", "Reader": "R1",
                "OtherACLTear": "2", "comments": "synthetic excluded"},
               {"id": "102", "side": "2", "version": "1", "visit": "3", "iwave": "6", "Reader": "R1",
                "OtherACLTear": "1", "comments": ""}], "|")
    write_csv(xr / "kxr_sq_bu00.txt", ["ID", "SIDE", "READPRJ", "VERSION", "V00XRKL", "V00XRJSM", "V00XRJSL"],
              [{"ID": "101", "SIDE": "1", "READPRJ": "15", "VERSION": "0.9", "V00XRKL": "2",
                "V00XRJSM": "1", "V00XRJSL": "0"}], "|")
    write_csv(mr / "kMRI_SQ_WORMS_Link00.txt", ["ID", "side", "VERSION", "READPRJ", "V00MWMCL"],
              [{"ID": "101", "side": "1", "VERSION": "1", "READPRJ": "1", "V00MWMCL": "4"}], "|")
    write_csv(mr / "kmri_sq_BLKSBML_bicl00.txt", ["ID", "side", "VERSION", "READPRJ", "V00BBMLNUM", "V00BBMLST"],
              [{"ID": "101", "side": "1", "VERSION": "1", "READPRJ": "1", "V00BBMLNUM": str(i),
                "V00BBMLST": str(i)} for i in (1, 2)], "|")
    result = e.build_oai(tmp_path)
    labels = {r["study_key"]: r for r in result.labels}
    assert set(labels) == {"OAI_101_R_V00", "OAI_102_L_V03"}
    assert labels["OAI_101_R_V00"]["ACL"] == 0  # no POMA override
    assert labels["OAI_101_R_V00"]["MCL"] == 1
    assert labels["OAI_101_R_V00"]["Medial OA"] == 1
    assert labels["OAI_102_L_V03"]["ACL"] == 1
    assert "synthetic forbidden comment" not in json.dumps(result.native)
    assert "synthetic excluded" not in json.dumps(result.native)
    assert result.native[0]["poma_OTHERACLTEAR"] == 2
    assert result.native[0]["bloksbml_lesion_record_count"] == 2
    assert len(json.loads(result.native[0]["bloksbml_lesion_records_json"])) == 2
    assert e.oai_key("101", "2: Left", "M24") == "OAI_101_L_V03"
    with pytest.raises(ValueError):
        e.oai_key("101", "bilateral", "BL")


def test_poma_root_missing_is_not_negative():
    s = {"MMTMA": 0, "MMTMB": 0, "MMTMP": 0, "MMRTM": 0}
    assert e.map_moaks(s, poma=True)["Medial Meniscus"] is None
    s["MMRTAM"] = 0
    assert e.map_moaks(s, poma=True)["Medial Meniscus"] == 0


def test_fastmri_headerless_population_annotations_and_ambiguous_targets(tmp_path):
    ann = tmp_path / "Annotations"
    ann.mkdir()
    (ann / "knee_file_list.csv").write_text("file1000000\nfile1000001.h5\nKnee data\n", encoding="utf-8")
    write_csv(ann / "knee.csv", ["file", "label"],
              [{"file": "file1000001", "label": label} for label in (
                  "Meniscus Tear", "Meniscus Tear", "Ligament - ACL Low Grade sprain", "Joint Effusion",
                  "Bone- Subchondral edema", "Bone-Fracture/Contusion/dislocation", "Periarticular cysts")])
    result = e.build_fastmri(tmp_path)
    negative, positive = result.labels
    assert negative["study_key"] == "FMRI_SC_file1000000" and negative["ACL"] == 0
    assert positive["ACL"] == 1 and positive["Effusion"] == 1 and positive["Baker's"] == 1
    for target in ("Medial Meniscus", "Lateral Meniscus", "Medial OA", "Contusion", "Fracture", "Synovitis"):
        assert positive[target] is None and negative[target] is None
    assert result.native[1]["any_meniscus"] == 1
    assert result.native[1]["meniscus_tear_annotation_count"] == 2
    assert result.notes["no_findings_volumes"] == 1
    with (ann / "knee.csv").open("a", encoding="utf-8") as stream:
        stream.write("file9999,Meniscus Tear\n")
    with pytest.raises(ValueError, match="outside reviewed"):
        e.build_fastmri(tmp_path)


def test_kneemri_keeps_labelled_but_unavailable_images_without_unpickling(tmp_path):
    (tmp_path / "vol01").mkdir()
    (tmp_path / "vol01/a.pck").write_bytes(b"not a pickle; must not be loaded")
    write_csv(tmp_path / "metadata.csv", ["examId", "seriesNo", "aclDiagnosis", "kneeLR", "volumeFilename"],
              [{"examId": "1", "seriesNo": "02", "aclDiagnosis": "1", "kneeLR": "L", "volumeFilename": "a.pck"},
               {"examId": "2", "seriesNo": "03", "aclDiagnosis": "0", "kneeLR": "R", "volumeFilename": "b.pck"}])
    result = e.build_kneemri(tmp_path)
    assert result.labels[0]["study_key"] == "KMRI_1_02"
    assert [r["ACL"] for r in result.labels] == [1, 0]
    assert [r["image_file_present"] for r in result.native] == [1, 0]


def test_mrnet_padded_split_keys_and_unknown_compartment(tmp_path):
    for name, label in (("acl", 1), ("abnormal", 1), ("meniscus", 0)):
        (tmp_path / f"train-{name}.csv").write_text(f"7,{label}\n", encoding="utf-8")
    result = e.build_mrnet(tmp_path)
    assert result.labels[0]["study_key"] == "MRN_train_0007"
    assert result.labels[0]["ACL"] == 1
    assert result.labels[0]["Medial Meniscus"] is None
    (tmp_path / "valid-acl.csv").write_text("7,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="partial"):
        e.build_mrnet(tmp_path)


def test_export_blank_mask_order_receipt_and_no_overwrite(tmp_path):
    row = {"study_key": "fake", **e.empty_targets(), "ACL": 1, "MCL": 0}
    table = e.Table("FAKE", labels=[row], native=[{"study_key": "fake", "native_grade": 2}])
    protocol = tmp_path / "PROTOCOL.md"
    protocol.write_text("synthetic registered protocol", encoding="utf-8")
    out = tmp_path / "output"
    receipt = e.export_table(table, out, protocol)
    with (out / "FAKE_12.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["study_key", *e.TARGETS]
    assert rows[1][1:5] == ["1", "0", "", ""]
    assert receipt["counts"]["ACL"] == {"positive": 1, "negative": 0, "nan": 0}
    assert receipt["counts"]["Fracture"]["nan"] == 1
    for filename, digest in receipt["outputs_sha256"].items():
        assert e.sha256(out / filename) == digest
    with pytest.raises(FileExistsError):
        e.export_table(table, out, protocol)
    table.labels.append(row.copy())
    with pytest.raises(ValueError, match="duplicate"):
        e.validate_table(table)
