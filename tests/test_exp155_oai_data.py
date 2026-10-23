"""CPU-only fake OAI GENERAL volumes and EXP136 labels for EXP155."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import exp155_oai_data as data
from tools.extcache import common


def _general(root: Path, monkeypatch, study="OAI_123_R_V00", fluid=1):
    monkeypatch.setattr(common, "require_space", lambda _: None)
    pixels = np.arange(64, dtype=np.uint16).reshape(8, 8)
    volume = np.stack([pixels + i * 10 for i in range(4)])
    meta = dict(source="oai", subject_id="123", knee_side="R", visit="V00",
                study_key=study, series_key=study + "_SAG_IW_TSE", plane="Sagittal",
                fluid_sensitive=fluid, rule_hit="SAG_IW_TSE", series_description="SAG_IW_TSE",
                thickness="2", TE="", TR="", field_strength="", vendor="",
                source_archive_member="fake.tar::slice0", spacing_unknown=False)
    series = common.Series(meta, volume,
                           np.column_stack([np.arange(4) * 2, np.zeros((4, 2))]),
                           np.array([0, 1, 0, 0, 0, 1]), np.array([.5, .5]), 2.0,
                           np.full(4, .5))
    common.add_series("oai", series, root)
    return study


def _args(root, cache, resume=False):
    return SimpleNamespace(root=root, oai_cache=cache, min_free_gib=0, resume=resume)


def test_cache_contract_resume_and_source_pin(tmp_path, monkeypatch):
    uid = _general(tmp_path, monkeypatch)
    cache = tmp_path / "exp155-cache"
    result = data.build_cache(_args(tmp_path, cache))
    assert result["status"] == "COMPLETE" and result["studies"] == 1
    manifest = data.check_cache(cache, full_hash=True)
    assert manifest["contract"] == data.CONTRACT
    assert manifest["spec"] == data.SPEC
    assert manifest["ids"] == [uid]
    array = np.load(cache / "study_00000.npy", allow_pickle=False)
    assert array.shape == (96, 384, 384) and array.dtype == np.uint8
    receipt = json.loads((cache / "study_00000.json").read_text(encoding="utf-8"))
    assert receipt["positive_mask_slices"] == 26
    assert receipt["slots"][0]["series"] == uid + "_SAG_IW_TSE"
    assert all(slot is None for slot in receipt["slots"][1:])
    before = data.file_sha256(cache / "study_00000.npy")
    assert data.build_cache(_args(tmp_path, cache))["status"] == "COMPLETE"
    assert data.file_sha256(cache / "study_00000.npy") == before
    with (common.gen_dir("oai", tmp_path) / "index.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="input index/code changed"):
        data.build_cache(_args(tmp_path, cache))


def test_cache_failure_logged_and_resumed_after_input_repair(tmp_path, monkeypatch):
    uid = _general(tmp_path, monkeypatch)
    cache = tmp_path / "exp155-cache"
    original = data.render_study
    monkeypatch.setattr(data, "render_study", lambda *_: (_ for _ in ()).throw(ValueError("fake decode")))
    result = data.build_cache(_args(tmp_path, cache))
    assert result["status"] == "INCOMPLETE" and result["failed"] == 1
    assert not (cache / "manifest.json").exists()
    failure = json.loads((cache / "study_00000.failure.json").read_text(encoding="utf-8"))
    assert failure["uid"] == uid and "fake decode" in failure["error"]
    monkeypatch.setattr(data, "render_study", original)
    with pytest.raises(ValueError, match="requires --resume"):
        data.build_cache(_args(tmp_path, cache))
    assert data.build_cache(_args(tmp_path, cache, resume=True))["status"] == "COMPLETE"


def _labels(path, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["study_key", *data.TARGETS])
        writer.writeheader()
        for uid, target_values in values.items():
            writer.writerow({"study_key": uid, **target_values})
    receipt = path.with_suffix(".receipt.json")
    receipt.write_text(json.dumps({"contract": "EXP136-v1", "outputs_sha256":
                                   {path.name: data.file_sha256(path)}}), encoding="utf-8")
    return receipt


def test_labels_only_author_targets_and_unknowns_masked(tmp_path, monkeypatch):
    uid = _general(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    data.build_cache(_args(tmp_path, cache))
    source = tmp_path / "OAI_12.csv"
    receipt = _labels(source, {uid: {"ACL": "1", "Lateral OA": "0", "PF OA": "",
                                      "Synovitis": "1", "Effusion": "1"}})
    out = tmp_path / "masked.csv"
    result = data.build_labels(SimpleNamespace(oai_cache=cache, oai_labels_source=source,
                                               oai_labels_receipt=receipt, out=out))
    assert result["studies"] == 1
    with out.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["Lateral OA"] == "0" and row["mask_Lateral OA"] == "1"
    assert row["Synovitis"] == "1" and row["mask_Synovitis"] == "1"
    assert row["PF OA"] == "" and row["mask_PF OA"] == "0"
    for target in set(data.TARGETS) - set(data.ALLOWED):
        assert row[target] == "" and row["mask_" + target] == "0"
    provenance = json.loads((tmp_path / "masked.provenance.json").read_text(encoding="utf-8"))
    assert provenance["csv_sha256"] == data.file_sha256(out)


def test_label_source_hash_and_nonbinary_guard(tmp_path, monkeypatch):
    uid = _general(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    data.build_cache(_args(tmp_path, cache))
    source = tmp_path / "OAI_12.csv"
    receipt = _labels(source, {uid: {"PF OA": "0"}})
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="labels/provenance changed"):
        data.build_labels(SimpleNamespace(oai_cache=cache, oai_labels_source=source,
                                          oai_labels_receipt=receipt, out=tmp_path / "masked.csv"))
