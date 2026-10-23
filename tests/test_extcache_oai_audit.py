"""CPU-only synthetic headers/archives; never reads project images or labels."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import pytest

from tools.extcache import oai_audit as audit

IOPS = {"Sagittal": [0, 1, 0, 0, 0, 1],
        "Coronal": [1, 0, 0, 0, 0, 1],
        "Axial": [1, 0, 0, 0, 1, 0]}


def header(i=0, plane="Axial", series="1.2.3", **updates):
    ds = dict(SOPInstanceUID=f"1.2.840.{i + 1}", SeriesInstanceUID=series,
              StudyInstanceUID="1.2", ImageOrientationPatient=IOPS[plane],
              ImagePositionPatient={"Sagittal": [i * 2, 0, 0],
                                    "Coronal": [0, i * 2, 0], "Axial": [0, 0, i * 2]}[plane],
              Rows=16, Columns=16, PixelSpacing=[.5, .5], EchoTime=4.7,
              EchoNumbers=1, ImageLaterality="L", SeriesDescription="AX_MPR_LEFT",
              ImageType=["DERIVED", "PRIMARY", "MPR"])
    ds.update(updates)
    return ds


def row(key="AX", name="AX_MPR_LEFT", plane="Sagittal", fluid=-1, root=None):
    path = Path("storage/external_data/oai/raw/V00") / f"{key}.tar.gz"
    return dict(source="oai", subject_id="9000001", knee_side="L", visit="V00",
                study_key="OAI_9000001_L_V00", series_key=key,
                plane=plane, fluid_sensitive=str(fluid), series_description=name,
                source_archive_member=f"{root / path if root else path}::001",
                rule_hit=name.split("_LEFT")[0])


def frame_list(headers):
    return [f for i, h in enumerate(headers) for f in audit.header_frames(h, f"{i:03}.dcm")]


def record_for(r, headers):
    return dict(series_key=r["series_key"], status="READ_OK",
                groups=[audit.route_group(g, r) for g in audit.group_frames(frame_list(headers))])


def test_classic_whitelist_and_geometry():
    ds = header(PatientName="DO_NOT_EXPORT", PatientID="DO_NOT_EXPORT", PixelData=b"pixels")
    f = audit.header_frames(ds, "001")[0]
    assert (f["plane"], f["normal"], f["frame_number"], f["number_of_frames"]) == ("Axial", [0., 0., 1.], 1, 1)
    assert f["echo_numbers"] == ["1"] and f["echo_time_ms"] == 4.7
    assert f["resolved_laterality"] == "L"
    assert "DO_NOT_EXPORT" not in json.dumps(f) and "pixels" not in json.dumps(f)


@pytest.mark.parametrize("iop", [None, [], [1, 0, 0, 1, 0, 0],
                                     [2, 0, 0, 0, 1, 0], [float("nan")] * 6,
                                     [1, 0, 0, 0, .70710678, .70710678]])
def test_bad_or_oblique_geometry_never_uses_description(iop):
    f = audit.header_frames(header(ImageOrientationPatient=iop), "001")[0]
    assert f["plane"] == "Unknown"
    assert "missing_invalid_or_oblique_iop" in f["issues"]
    audit.json_text(f)  # non-finite values are represented by null, never NaN.


def test_localizer_first_splits_from_mpr_and_does_not_define_plane():
    hs = [header(99, "Sagittal", ImageType=["LOCALIZER"]), header(1), header(0)]
    groups = [audit.route_group(g, row()) for g in audit.group_frames(frame_list(hs))]
    assert len(groups) == 2
    good = [g for g in groups if g["route_eligible"]]
    assert len(good) == 1 and good[0]["plane"] == "Axial" and good[0]["frame_count"] == 2
    assert good[0]["position_range_mm"] == [0, 2]
    assert good[0]["frame_refs"][0][2] == "1.2.840.1"
    assert any("localizer" in g["routing_issues"] for g in groups)


def test_enhanced_per_frame_precedence_echo_side_and_frame_identity():
    hs = header(NumberOfFrames=3, ImageLaterality="", ImageOrientationPatient=IOPS["Sagittal"],
                SharedFunctionalGroupsSequence=[{
                    "PlaneOrientationSequence": [{"ImageOrientationPatient": IOPS["Coronal"]}],
                    "PixelMeasuresSequence": [{"PixelSpacing": [.4, .6]}],
                    "MREchoSequence": [{"EffectiveEchoTime": 8.0}],
                    "FrameAnatomySequence": [{"FrameLaterality": "R"}]}],
                PerFrameFunctionalGroupsSequence=[{
                    "PlaneOrientationSequence": [{"ImageOrientationPatient": IOPS["Axial"]}],
                    "PlanePositionSequence": [{"ImagePositionPatient": [0, 0, 2 * i]}],
                    "MREchoSequence": [{"EffectiveEchoTime": 5.0 if i < 2 else 10.0}],
                    "FrameAnatomySequence": [{"FrameLaterality": "L"}],
                    "FrameContentSequence": [{"TemporalPositionIndex": 1, "StackID": "A"}]}
                    for i in range(3)])
    fs = audit.header_frames(hs, "enhanced.dcm")
    assert {f["plane"] for f in fs} == {"Axial"}
    assert [f["frame_number"] for f in fs] == [1, 2, 3]
    assert len({f["sop_instance_uid"] for f in fs}) == 1
    assert all(f["pixel_spacing"] == [.4, .6] and f["resolved_laterality"] == "L" for f in fs)
    gs = audit.group_frames(fs)
    assert sorted(g["frame_count"] for g in gs) == [1, 2]
    assert all("duplicate_sop_frame" not in g["issues"] for g in gs)


@pytest.mark.parametrize("updates", [dict(NumberOfFrames=2), dict(NumberOfFrames=0),
    dict(NumberOfFrames=1.5), dict(NumberOfFrames=2, PerFrameFunctionalGroupsSequence=[{}]),
    dict(NumberOfFrames=1, SharedFunctionalGroupsSequence=[{}, {}])])
def test_reject_bad_frame_count(updates):
    with pytest.raises(ValueError):
        audit.header_frames(header(**updates), "bad")


@pytest.mark.parametrize("change", [dict(SeriesInstanceUID="9.9"), dict(EchoNumbers=2),
    dict(EchoTime=9), dict(ImageLaterality="R"), dict(Rows=32), dict(PixelSpacing=[.6, .5]),
    dict(AcquisitionNumber=2), dict(TemporalPositionIdentifier=2), dict(FrameOfReferenceUID="9.9")])
def test_homogeneity_axes_split(change):
    gs = audit.group_frames(frame_list([header(0), header(1), header(2, **change), header(3, **change)]))
    assert sorted(g["frame_count"] for g in gs) == [2, 2]


def test_iop_sign_not_merged_even_with_same_plane():
    reversed_iop = [-1, 0, 0, 0, -1, 0]
    gs = audit.group_frames(frame_list([header(0), header(1),
                                      header(2, ImageOrientationPatient=reversed_iop),
                                      header(3, ImageOrientationPatient=reversed_iop)]))
    assert len(gs) == 2 and all(g["plane"] == "Axial" for g in gs)


def test_duplicate_sop_and_positions_quarantined():
    fs = frame_list([header(0), header(0)])
    g = audit.route_group(audit.group_frames(fs)[0], row())
    assert {"duplicate_sop_frame", "duplicate_or_near_duplicate_positions"} <= set(g["routing_issues"])
    assert not g["route_eligible"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, "malformed"])
def test_invalid_echo_is_not_silently_treated_as_missing(value):
    f = audit.header_frames(header(EchoTime=value), "001")[0]
    assert "invalid_echo_time" in f["issues"]
    audit.json_text(f)


def test_position_walk_incompatible_with_normal():
    hs = [header(0), header(1, ImagePositionPatient=[2, 0, 2])]
    g = audit.route_group(audit.group_frames(frame_list(hs))[0], row())
    assert "in_plane_position_drift_gt_1mm" in g["routing_issues"]
    assert not g["route_eligible"]


@pytest.mark.parametrize("updates,reason", [
    (dict(ImageLaterality="R"), "header_vs_index_side_conflict"),
    (dict(ImageLaterality="L", Laterality="R"), "invalid_or_conflicting_laterality"),
    (dict(SeriesDescription="COR_MPR_LEFT"), "header_vs_index_sequence_conflict"),
    (dict(SeriesDescription="AX_MPR_RIGHT"), "header_description_side_conflict"),
    (dict(ImageOrientationPatient=IOPS["Sagittal"]), "geometry_vs_source_plane_conflict"),
])
def test_conflicts_do_not_force_routing(updates, reason):
    gs = audit.group_frames(frame_list([header(0, **updates), header(1, **updates)]))
    g = audit.route_group(gs[0], row())
    assert not g["route_eligible"] and reason in g["routing_issues"]


def test_laterality_missing_stays_explicit_and_index_join_must_match():
    hs = [header(0, ImageLaterality=""), header(1, ImageLaterality="")]
    g = audit.group_frames(frame_list(hs))[0]
    routed = audit.route_group(g, row())
    assert routed["route_eligible"]
    assert routed["laterality_basis"] == "index_only_unconfirmed_by_DICOM"
    bad = dict(row(), visit="V01")
    assert "index_subject_side_visit_join_mismatch" in audit.route_group(g, bad)["routing_issues"]


@pytest.mark.parametrize("name,plane,fluid", [
    ("SAG_IW_TSE", "Sagittal", 0), ("COR_IW_TSE", "Coronal", 0),
    ("COR_MPR", "Coronal", 1), ("AX_MPR", "Axial", -1)])
def test_exact_requested_policy(name, plane, fluid):
    r = row(name, name + "_LEFT")
    rec = record_for(r, [header(i, plane, SeriesDescription=name + "_LEFT") for i in range(2)])
    g = rec["groups"][0]
    assert g["route_eligible"] and (g["proposed_plane"], g["proposed_fluid"]) == (plane, fluid)
    assert audit.sequence_name(name + "_FAKE") == name
    assert audit.sequence_name(name + "TYPO") == ""


def test_before_reuses_legacy_fallback_and_order_after_is_strict():
    rows = [row("bad_ax", fluid=-1), row("sag", "SAG_IW_TSE_LEFT", fluid=1),
            row("cor", "COR_IW_TSE_LEFT", "Coronal", 0)]
    for slots in audit.CONTRACTS.values():
        before = audit.select_slots(rows, slots)
        assert [p["series_key"] if p else None for p in before] == ["sag", "bad_ax", "cor", None, None]
        after = audit.select_slots(audit.policy_projection(rows), slots, strict=True)
        assert [p["series_key"] if p else None for p in after] == [None, "sag", None, "cor", "bad_ax"]
    # First matching row wins; do not sort the old index for BEFORE.
    assert audit.select_slots([rows[1], dict(rows[1], series_key="first2")], [("Sagittal", 1, 1)])[0]["series_key"] == "sag"
    assert audit.select_slots([dict(rows[1], series_key="first2"), rows[1]], [("Sagittal", 1, 1)])[0]["series_key"] == "first2"


def test_whole_index_missing_keeps_denominator_and_projection_separate():
    r = row()
    missing = dict(series_key=r["series_key"], status="MISSING_RAW", groups=[])
    result = audit.preview_counts([r], [missing])
    assert result["index_studies"] == 1 and result["index_series"] == 1
    assert not result["geometry_coverage_complete"]
    for stats in result["counts"].values():
        assert sum(s["AFTER_HEADER_ELIGIBLE"] for s in stats) == 0
        assert stats[0]["BEFORE"] == 1  # historical fallback, even fluid=-1
        assert stats[-1]["AFTER_DECLARED_ONLY"] == 1
    with pytest.raises(ValueError):
        audit.preview_counts([r], [])


def test_both_contracts_count_selections_separately_from_slice_positions():
    r = row()
    rec = record_for(r, [header(0), header(1)])
    result = audit.preview_counts([r], [rec])
    assert result["geometry_coverage_complete"]
    counts = list(result["counts"].values())
    assert [x[-1]["AFTER_HEADER_ELIGIBLE"] for x in counts] == [1, 1]
    assert [x[-1]["planned_slice_positions"]["AFTER_HEADER_ELIGIBLE"] for x in counts] == [12, 18]


def test_duplicate_series_across_archives_not_double_counted():
    rows = [row("one"), row("two")]
    records = [record_for(r, [header(0), header(1)]) for r in rows]
    result = audit.preview_counts(rows, records)
    assert result["duplicate_dicom_series_across_archives_quarantined"] == 1
    assert all(s["AFTER_HEADER_ELIGIBLE"] == 0 for c in result["counts"].values() for s in c)


def write_tar(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def dicom_bytes(ds):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = ds["SOPInstanceUID"]
    d = FileDataset("synthetic", {}, file_meta=meta, preamble=b"\0" * 128)
    for k, v in ds.items():
        setattr(d, k, v)
    d.SOPClassUID = MRImageStorage
    d.BitsAllocated = d.BitsStored = 16
    d.HighBit = 15
    d.PixelRepresentation = 0
    d.SamplesPerPixel = 1
    d.PhotometricInterpretation = "MONOCHROME2"
    d.PixelData = b"not-a-decodable-image"  # Must never be decoded.
    d.PatientName = "DO_NOT_EXPORT"
    stream = io.BytesIO()
    d.save_as(stream)
    return stream.getvalue()


def test_archive_header_only_and_missing_raw(tmp_path):
    r = row(root=tmp_path)
    p = audit.recorded_archive(r, tmp_path)
    assert audit.audit_archive(r, tmp_path)["status"] == "MISSING_RAW"
    write_tar(p, [(f"./{i:03}", dicom_bytes(header(i))) for i in range(2)])
    original = p.read_bytes()
    rec = audit.audit_archive(r, tmp_path)
    assert rec["status"] == "READ_OK" and rec["frame_count"] == 2
    assert rec["groups"][0]["route_eligible"]
    assert "DO_NOT_EXPORT" not in audit.json_text(rec)
    assert p.read_bytes() == original
    assert not (tmp_path / "storage/cache").exists()


def test_archive_error_cannot_partially_admit_good_frames(tmp_path):
    r = row(root=tmp_path)
    p = audit.recorded_archive(r, tmp_path)
    write_tar(p, [("001", dicom_bytes(header(0))), ("002", dicom_bytes(header(1))), ("broken", b"bad")])
    rec = audit.audit_archive(r, tmp_path)
    assert rec["status"] == "READ_ERROR"
    assert rec["groups"] and not any(g["route_eligible"] for g in rec["groups"])
    assert audit.header_candidates(rec, r) == []


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:\\escape", "\\rooted"])
def test_unsafe_archive_names_fail_without_extraction(tmp_path, member):
    r = row(root=tmp_path)
    write_tar(audit.recorded_archive(r, tmp_path), [(member, dicom_bytes(header()))])
    rec = audit.audit_archive(r, tmp_path)
    assert rec["status"] == "READ_ERROR" and not rec["groups"]


def test_registry_confinement_and_no_basename_fallback(tmp_path):
    r = row(root=tmp_path)
    other = tmp_path / "storage/external_data/oai/different/AX.tar.gz"
    write_tar(other, [("001", dicom_bytes(header()))])
    assert audit.audit_archive(r, tmp_path)["status"] == "MISSING_RAW"
    r["source_archive_member"] = str(tmp_path / "outside.tar.gz") + "::001"
    assert audit.audit_archive(r, tmp_path)["status"] == "INVALID_RECORDED_PATH"


def test_fixed_list_is_pinned_and_has_required_strata():
    rows, keys = [], []
    for name, count in (("AX_MPR", 6), ("COR_MPR", 6), ("SAG_IW_TSE", 3), ("COR_IW_TSE", 3)):
        for i in range(count):
            key = f"{name}{i}"
            r = row(key, name + "_LEFT", "Coronal" if name == "COR_IW_TSE" else "Sagittal")
            rows.append(r)
            keys.append(key)
    manifest = dict(index_sha256="bound", series_keys=keys)
    assert audit.fixed_sample(rows, manifest, "bound") == rows
    with pytest.raises(ValueError):
        audit.fixed_sample(rows, manifest, "changed")
    with pytest.raises(ValueError):
        audit.fixed_sample(rows, dict(manifest, series_keys=keys[:-1] + [keys[0]]), "bound")


def test_output_cannot_touch_cache_or_reuse_existing_dir(tmp_path):
    with pytest.raises(ValueError):
        audit._output_dir(tmp_path / "storage/cache/derived/new", tmp_path)
    out = audit._output_dir(tmp_path / "artifacts/audit", tmp_path)
    assert out.is_dir()
    with pytest.raises(FileExistsError):
        audit._output_dir(out, tmp_path)


def test_review_gate_binds_code_and_requires_test_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "fingerprint", lambda root: {"code": "123"})
    note = dict(verdict="PASS", tool_version=audit.VERSION, reviewer="astra",
                tests_passed=True, test_command="B synthetic pytest", code_sha256={"code": "123"})
    path = tmp_path / "review.json"
    path.write_text(json.dumps(note))
    assert audit.verification(path, tmp_path)["verification_note"] == str(path.resolve())
    for change in (dict(verdict="HOLD"), dict(code_sha256={"code": "changed"}),
                   dict(tests_passed=False), dict(tests_passed="not run")):
        path.write_text(json.dumps(dict(note, **change)))
        with pytest.raises(ValueError):
            audit.verification(path, tmp_path)


def test_preview_command_synthetic_end_to_end(tmp_path, monkeypatch, capsys):
    r = row(root=tmp_path)
    p = audit.recorded_archive(r, tmp_path)
    write_tar(p, [(f"{i:03}", dicom_bytes(header(i))) for i in range(2)])
    index = tmp_path / audit.INDEX
    index.parent.mkdir(parents=True)
    index.write_text("synthetic index placeholder")
    sample = tmp_path / audit.SAMPLE
    sample.parent.mkdir(parents=True)
    sample.write_text(json.dumps(dict(index_sha256=audit._sha(index))))
    monkeypatch.setattr(audit, "read_oai_index", lambda root: [r])
    monkeypatch.setattr(audit, "fingerprint", lambda root: {"code": "bound"})
    monkeypatch.setattr(audit, "verification", lambda *args: dict(verification_note="synthetic",
                                                                verification_sha256="synthetic"))
    out = tmp_path / ".local/preview"
    assert audit.run("preview", out, "synthetic", tmp_path) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["execution_complete"] and summary["pixels_read"] is False
    assert summary["cache_writes"] is False and summary["index_series"] == 1
    assert "AFTER_HEADER_ELIGIBLE" in capsys.readouterr().out
    assert index.read_text() == "synthetic index placeholder"
    assert (out / "frames.jsonl").read_text() == ""  # full preview emits groups, not all frames


def test_incomplete_preview_emits_receipt_and_nonzero_exit(tmp_path, monkeypatch):
    r = row(root=tmp_path)
    index = tmp_path / audit.INDEX
    index.parent.mkdir(parents=True)
    index.write_text("synthetic")
    sample = tmp_path / audit.SAMPLE
    sample.parent.mkdir(parents=True)
    sample.write_text(json.dumps(dict(index_sha256=audit._sha(index))))
    monkeypatch.setattr(audit, "read_oai_index", lambda root: [r])
    monkeypatch.setattr(audit, "fingerprint", lambda root: {})
    monkeypatch.setattr(audit, "verification", lambda *args: dict(verification_note="synthetic",
                                                                verification_sha256="synthetic"))
    out = tmp_path / "artifacts/preview"
    assert audit.run("preview", out, "synthetic", tmp_path) == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "INCOMPLETE_RAW_EVIDENCE"
    assert summary["missing_raw"] == [r["source_archive_member"].split("::")[0]]
