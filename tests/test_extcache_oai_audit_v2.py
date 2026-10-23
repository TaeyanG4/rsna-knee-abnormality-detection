"""Synthetic-only tests for the full OAI header audit; no project raw/labels."""
import csv
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pytest

from tools.extcache import oai_audit_v2 as a
from test_extcache_oai_audit import header as old_header, row as old_row, dicom_bytes, write_tar, IOPS


def header(i=0, plane="Axial", source="AX_MPR", **kw):
    values = dict(SOPClassUID="1.2.840.10008.5.1.4.1.1.4", SequenceName="*de3d1",
                  ScanOptions=["WE"], MRAcquisitionType="3D", InstanceNumber=i+1,
                  SeriesDescription=source+"_LEFT")
    if "TSE" in source:
        values.update(SequenceName="*tse2d1", ScanOptions=["SAT2", "FS"] if source.startswith("SAG") else ["SAT1"],
                      MRAcquisitionType="2D", ImageType=["ORIGINAL", "PRIMARY"])
    values.update(kw)
    return old_header(i, plane, **values)


def row(source="AX_MPR", key="AX", root=None):
    return old_row(key, source+"_LEFT", root=root)


def join(**kw):
    return dict(dict(issues=[], join_verified=True, image03_side="L"), **kw)


def frames(hs):
    return [f for i, h in enumerate(hs) for f in a.header_frames(h, f"{i:03}")]


def routed(hs, r=None, j=None):
    return [a.route_group(g, r or row(), j or join()) for g in a.group_frames(frames(hs))]


@pytest.mark.parametrize("source,plane,fluid,family", [
    ("SAG_IW_TSE", "Sagittal", 1, "IW_TSE_FS"), ("COR_IW_TSE", "Coronal", 0, "IW_TSE_NON_FS"),
    ("COR_MPR", "Coronal", 1, "DESS_WE"), ("AX_MPR", "Axial", -1, "DESS_WE")])
def test_r2_routes_with_actual_contrast_evidence(source, plane, fluid, family):
    g = routed([header(i, plane, source) for i in range(3)], row(source))[0]
    assert g["route_eligible"]
    assert (g["proposed_plane"], g["proposed_fluid"], g["contrast_family"]) == (plane, fluid, family)


def test_projection_same_uid_is_excluded_before_stack_geometry():
    hs = [header(99, "Sagittal", ImageType=["DERIVED", "PROJECTION IMAGE"]), header(2), header(0), header(1)]
    fs = frames(hs)
    assert fs[0]["excluded_reason"] == "projection_image"
    gs = a.group_frames(fs)
    assert len(gs) == 1 and gs[0]["frame_count"] == 3 and gs[0]["plane"] == "Axial"
    assert gs[0]["ordered_positions_mm"] == [0, 2, 4]
    assert gs[0]["archive_ipp_order"] == "unordered_or_duplicate"
    assert gs[0]["instance_ipp_order"] == "ascending"
    assert a.route_group(gs[0], row(), join())["route_eligible"]
    assert all(ref[2] != "1.2.840.100" for ref in gs[0]["frame_refs"])


@pytest.mark.parametrize("marker", ["LOCALIZER", "SCOUT", "SURVEY"])
def test_localizers_are_excluded(marker):
    fs = frames([header(0, ImageType=[marker]), header(1), header(2)])
    assert a.group_frames(fs)[0]["frame_count"] == 2


@pytest.mark.parametrize("source,plane,update,problem", [
    ("SAG_IW_TSE", "Sagittal", dict(ScanOptions=["SAT2"]), "missing_or_conflicting_fat_suppression_evidence"),
    ("COR_IW_TSE", "Coronal", dict(ScanOptions=["FS"]), "missing_or_conflicting_non_fs_evidence"),
    ("COR_IW_TSE", "Coronal", dict(ScanOptions=[]), "missing_or_conflicting_non_fs_evidence"),
    ("AX_MPR", "Axial", dict(SequenceName="SE"), "missing_or_conflicting_dess_we_evidence"),
    ("COR_MPR", "Coronal", dict(ScanOptions=[]), "missing_or_conflicting_dess_we_evidence"),
    ("AX_MPR", "Axial", dict(MRAcquisitionType="2D"), "missing_or_conflicting_dess_we_evidence"),
])
def test_missing_or_conflicting_contrast_is_not_guessed(source, plane, update, problem):
    g = routed([header(i, plane, source, **update) for i in range(2)], row(source))[0]
    assert not g["route_eligible"] and problem in g["routing_issues"]


def test_scanning_sequence_does_not_override_dess_evidence():
    assert routed([header(i, ScanningSequence="SE", SequenceVariant="NONE") for i in range(2)])[0]["route_eligible"]


def test_contrast_changes_split_even_under_one_uid():
    gs = routed([header(0), header(1), header(2, ScanOptions=["FS"]), header(3, ScanOptions=["FS"])])
    assert sorted(g["frame_count"] for g in gs) == [2, 2]
    assert sum(g["route_eligible"] for g in gs) == 1
    assert len({g["group_id"] for g in gs}) == 2


def test_enhanced_precedence_projection_and_nonspatial_dimensions():
    h = header(NumberOfFrames=4, ScanOptions=["FS"], SequenceName="wrong",
        DimensionIndexSequence=[dict(DimensionIndexPointer=0x00209057), dict(DimensionIndexPointer=0x00209128)],
        SharedFunctionalGroupsSequence=[dict(ScanOptions=["WE"], SequenceName="*de3d1")],
        PerFrameFunctionalGroupsSequence=[dict(
            PlanePositionSequence=[dict(ImagePositionPatient=[0, 0, 2*(i % 2)])],
            FrameContentSequence=[dict(DimensionIndexValues=[i % 2+1, i // 2+1])],
            MRImageFrameTypeSequence=[dict(FrameType=["DERIVED", "PRIMARY", "MPR"])]) for i in range(4)])
    gs = a.group_frames(a.header_frames(h, "enhanced"))
    assert len(gs) == 2 and all(g["frame_count"] == 2 for g in gs)
    assert all(g["scan_options"] == ["WE"] and g["sequence_name"] == "*de3d1" for g in gs)
    assert len({g["group_id"] for g in gs}) == 2
    h["ImageType"] = ["PROJECTION IMAGE"]
    assert a.group_frames(a.header_frames(h, "enhanced")) == []


def serialized_header(values):
    """Real DICOM serialization + production whitelist, with no pixel decoding."""
    import io
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    def dataset(items):
        ds = Dataset()
        for key, value in items.items():
            setattr(ds, key, [dataset(x) for x in value] if key.endswith("Sequence") else value)
        return ds

    ds = dataset(values)
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.BitsAllocated = 16
    ds.PixelData = b"not-a-decodable-image"
    ds.PatientName = "DO_NOT_EXPORT"
    stream = io.BytesIO()
    pydicom.dcmwrite(stream, ds, enforce_file_format=True)
    stream.seek(0)
    parsed = pydicom.dcmread(stream, stop_before_pixels=True, specific_tags=a.TAGS)
    assert "PixelData" not in parsed and "PatientName" not in parsed
    return parsed


def test_d1_serialized_enhanced_acquisitions_do_not_merge():
    h = header(SOPClassUID="1.2.840.10008.5.1.4.1.1.4.1", NumberOfFrames=4,
        PerFrameFunctionalGroupsSequence=[dict(
            PlanePositionSequence=[dict(ImagePositionPatient=[0, 0, 2*i])],
            FrameContentSequence=[dict(FrameAcquisitionNumber=1+i//2)]) for i in range(4)])
    ds = serialized_header(h)
    assert str(ds.SOPClassUID) == h["SOPClassUID"]
    assert [p.FrameContentSequence[0].FrameAcquisitionNumber
            for p in ds.PerFrameFunctionalGroupsSequence] == [1, 1, 2, 2]
    fs = a.header_frames(ds, "enhanced")
    gs = a.group_frames(fs)
    assert sorted(g["frame_count"] for g in gs) == [2, 2]
    assert {g["frame_acquisition_number"] for g in gs} == {"1", "2"}
    assert all(a.route_group(g, row(), join())["route_eligible"] for g in gs)
    by_acquisition = {g["frame_acquisition_number"]: [r[1] for r in g["frame_refs"]] for g in gs}
    assert by_acquisition == {"1": [1, 2], "2": [3, 4]}
    reversed_groups = a.group_frames(list(reversed(a.header_frames(ds, "enhanced"))))
    assert [(g["group_id"], g["frame_refs"]) for g in gs] == [
        (g["group_id"], g["frame_refs"]) for g in reversed_groups]


def test_d3_serialized_geometry_and_eligibility_invariant_to_frame_order():
    # IOPs round to one v1 group, but choosing the second frame's normal used
    # to move the in-plane drift just above the strict 1 mm limit.
    hs = [header(0), header(1, ImageOrientationPatient=[1, 0, 0, 0, 1, .00004],
        ImagePositionPatient=[0, 1, 2]), header(2)]
    fs = [a.header_frames(serialized_header(h), f"{i:03}")[0] for i, h in enumerate(hs)]
    results = [[a.route_group(g, row(), join()) for g in a.group_frames(list(ordered))]
               for ordered in permutations(fs)]
    assert all(a.dump(result) == a.dump(results[0]) for result in results)
    assert len(results[0]) == 1
    g = results[0][0]
    assert g["route_eligible"] and g["in_plane_drift_mm"] == 1.0
    assert g["ordered_positions_mm"] == [0.0, 2.0, 4.0]
    assert g["frame_refs"] == [[f"{i:03}", 1, fs[i]["sop_instance_uid"]] for i in range(3)]


TILT = [1, 0, 0, 0, 1, .00004]  # rounds into the canonical IOP group


def d4_frames(z1, tilted, y1):
    """Three serialized slices at z=0, z1, 1.7; frame `tilted` has TILT.

    The canonical/recorded IOP is the untilted one (smallest frame_sort_key).
    Projection on the tilted normal shifts slice 1 by -0.00004*y1 mm.
    """
    ipps = [[0, 0, 0], [0, y1, z1], [0, 0, 1.7]]
    hs = [header(i, ImagePositionPatient=ipps[i],
                 **(dict(ImageOrientationPatient=TILT) if i == tilted else {})) for i in range(3)]
    return [a.header_frames(serialized_header(h), f"{i:03}")[0] for i, h in enumerate(hs)]


@pytest.mark.parametrize("tilted", [0, 1, 2])  # first, middle, last spatial slice
@pytest.mark.parametrize("z1,y1,eligible", [
    (.79998, -1, False),  # recorded-normal deviation 0.05002 mm: must quarantine
    (.80002, 1, True),    # recorded-normal deviation 0.04998 mm: must not reject
])
def test_d4_ipp_spacing_uses_recorded_reference_normal(tilted, z1, y1, eligible):
    fs = d4_frames(z1, tilted, y1)
    results = [[a.route_group(g, row(), join()) for g in a.group_frames(list(ordered))]
               for ordered in permutations(fs)]
    assert all(a.dump(result) == a.dump(results[0]) for result in results)
    assert len(results[0]) == 1
    g = results[0][0]
    assert g["iop"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert g["frame_refs"] == [[f"{i:03}", 1, fs[i]["sop_instance_uid"]] for i in range(3)]
    # Positions are the projections on the recorded IOP normal (0, 0, 1) and
    # agree with the v1 range/step fields of the same group receipt.
    assert g["ordered_positions_mm"] == pytest.approx([0.0, z1, 1.7], abs=1e-12)
    assert g["position_range_mm"] == [g["ordered_positions_mm"][0], g["ordered_positions_mm"][-1]]
    assert g["median_step_mm"] == pytest.approx(float(np.median(np.diff(g["ordered_positions_mm"]))), abs=1e-12)
    assert g["archive_ipp_order"] == g["instance_ipp_order"] == "ascending"
    assert "geometry_reference_frame_mismatch" not in g["routing_issues"]
    assert g["route_eligible"] is eligible
    assert ("irregular_ipp_spacing" in g["routing_issues"]) is (not eligible)
    if eligible:
        assert g["routing_issues"] == []
    else:
        assert g["routing_issues"] == ["irregular_ipp_spacing"]


def test_d4_reference_mismatch_fails_closed(monkeypatch):
    fs = d4_frames(.8, 1, 1)
    real = a.old.group_frames

    def tampered(frames):
        gs = real(frames)
        for g in gs:
            g["iop"] = TILT  # recorded iop no longer from the canonical frame
        return gs
    monkeypatch.setattr(a.old, "group_frames", tampered)
    g = a.route_group(a.group_frames(fs)[0], row(), join())
    assert not g["route_eligible"] and "geometry_reference_frame_mismatch" in g["routing_issues"]


@pytest.mark.parametrize("level,expected", [("per", "3"), ("shared", "2"), ("top", "1")])
def test_d1_serialized_acquisition_precedence(level, expected):
    h = header(SOPClassUID="1.2.840.10008.5.1.4.1.1.4.1", AcquisitionNumber=9,
        FrameAcquisitionNumber=1, NumberOfFrames=2,
        PerFrameFunctionalGroupsSequence=[dict(
            PlanePositionSequence=[dict(ImagePositionPatient=[0, 0, 2*i])]) for i in range(2)])
    if level in ("per", "shared"):
        h["SharedFunctionalGroupsSequence"] = [dict(FrameContentSequence=[dict(FrameAcquisitionNumber=2)])]
    if level == "per":
        for pf in h["PerFrameFunctionalGroupsSequence"]:
            pf["FrameContentSequence"] = [dict(FrameAcquisitionNumber=3)]
    fs = a.header_frames(serialized_header(h), "enhanced")
    assert {f["acquisition_number"] for f in fs} == {"9"}
    assert {f["frame_acquisition_number"] for f in fs} == {expected}
    g = a.group_frames(fs)[0]
    assert g["frame_acquisition_number"] == expected and g["frame_count"] == 2
    assert a.route_group(g, row(), join())["route_eligible"]


def test_d1_classic_and_enhanced_acquisition_keys_are_both_preserved():
    hs = [header(i, AcquisitionNumber=1+i//2, FrameAcquisitionNumber=7) for i in range(4)]
    gs = routed([serialized_header(h) for h in hs])
    assert len(gs) == 2 and all(g["frame_count"] == 2 and g["route_eligible"] for g in gs)
    assert {g["frame_acquisition_number"] for g in gs} == {"7"}


def test_d1_unknown_acquisition_does_not_merge_with_explicit_zero():
    hs = [header(i, **(dict(FrameAcquisitionNumber=0) if i < 2 else {})) for i in range(4)]
    gs = routed(hs)
    assert len(gs) == 2 and {g["frame_acquisition_number"] for g in gs} == {"", "0"}
    assert sorted(g["frame_count"] for g in gs) == [2, 2]


def test_d1_frame_acquisition_dimension_is_nonspatial():
    h = header(SOPClassUID="1.2.840.10008.5.1.4.1.1.4.1", NumberOfFrames=4,
        DimensionIndexSequence=[dict(DimensionIndexPointer=0x00209057),
                                dict(DimensionIndexPointer=0x00209156)],
        PerFrameFunctionalGroupsSequence=[dict(
            PlanePositionSequence=[dict(ImagePositionPatient=[0, 0, 2*(i % 2)])],
            FrameContentSequence=[dict(DimensionIndexValues=[i % 2+1, i // 2+1])]) for i in range(4)])
    gs = routed([serialized_header(h)])
    assert len(gs) == 2 and all(g["frame_count"] == 2 and g["route_eligible"] for g in gs)


def test_unknown_enhanced_dimensions_quarantine():
    h = header(NumberOfFrames=2, PerFrameFunctionalGroupsSequence=[dict(
        PlanePositionSequence=[dict(ImagePositionPatient=[0, 0, i*2])],
        FrameContentSequence=[dict(DimensionIndexValues=[i+1])]) for i in range(2)])
    assert "unresolved_dimension_semantics" in a.group_frames(a.header_frames(h, "e"))[0]["issues"]
    h["DimensionIndexSequence"] = [dict(DimensionIndexPointer=0x77771001)]
    assert "unresolved_dimension_semantics" in a.group_frames(a.header_frames(h, "e"))[0]["issues"]


@pytest.mark.parametrize("updates", [dict(ScanOptions=["WE", "FS"]),
    dict(SpectrallySelectedSuppression="WATER")])
def test_dess_conflicting_suppression_is_quarantined(updates):
    g = routed([header(i, **updates) for i in range(2)])[0]
    assert not g["route_eligible"] and "conflicting_dess_suppression_method" in g["routing_issues"]


def test_sagittal_fs_rejects_shared_water_suppression():
    hs = [header(i, "Sagittal", "SAG_IW_TSE", SharedFunctionalGroupsSequence=[dict(
        MRModifierSequence=[dict(SpectrallySelectedSuppression="WATER")])]) for i in range(2)]
    g = routed(hs, row("SAG_IW_TSE"))[0]
    assert not g["route_eligible"] and "conflicting_tse_suppression_method" in g["routing_issues"]


def test_whitelisted_top_level_suppression_survives_real_dicom_read():
    import io
    import pydicom
    hs = [pydicom.dcmread(io.BytesIO(dicom_bytes(header(i, SpectrallySelectedSuppression="WATER"))),
        stop_before_pixels=True, specific_tags=a.TAGS) for i in range(2)]
    g = routed(hs)[0]
    assert g["spectrally_selected_suppression"] == "WATER"
    assert not g["route_eligible"]


def test_image_type_fs_conflicts_with_coronal_non_fs():
    hs = [header(i, "Coronal", "COR_IW_TSE", ImageType=["ORIGINAL", "PRIMARY", "FS"]) for i in range(2)]
    assert not routed(hs, row("COR_IW_TSE"))[0]["route_eligible"]


@pytest.mark.parametrize("hs,issue", [
    ([header(0), header(0)], "duplicate_sop_frame"),
    ([header(0), header(1), header(3)], "irregular_ipp_spacing"),
    ([header(0), header(1, ImagePositionPatient=[2, 0, 2])], "in_plane_position_drift_gt_1mm"),
    ([header(0, SOPClassUID="1.2.3"), header(1, SOPClassUID="1.2.3")], "unsupported_sop_class"),
])
def test_bad_geometry_or_identity_cannot_route(hs, issue):
    gs = routed(hs)
    assert not any(g["route_eligible"] for g in gs)
    assert any(issue in g["routing_issues"] for g in gs)


def test_missing_dicom_side_requires_independent_join():
    hs = [header(i, ImageLaterality="") for i in range(2)]
    assert routed(hs)[0]["route_eligible"]
    bad = join(issues=["missing_MOAKS_participant_side"], join_verified=False)
    g = routed(hs, j=bad)[0]
    assert not g["route_eligible"] and "missing_independent_laterality" in g["routing_issues"]
    assert not routed([header(i, ImageLaterality="R") for i in range(2)])[0]["route_eligible"]


@pytest.mark.parametrize("description", ["AX_MPR_LEFT_RIGHT", "AX_MPR_RIGHT_LEFT"])
@pytest.mark.parametrize("dicom_side", ["", "L"])
def test_d2_serialized_contradictory_description_side_is_quarantined(description, dicom_side):
    hs = [serialized_header(header(i, SeriesDescription=description, ImageLaterality=dicom_side))
          for i in range(2)]
    g = routed(hs)[0]
    assert not g["route_eligible"]
    assert "header_description_side_conflict" in g["routing_issues"]
    assert g["header_description_sides"] == ["L", "R"]


@pytest.mark.parametrize("description,sides,eligible", [
    ("AX_MPR", [], True), ("AX_MPR_LEFT_LEFT", ["L"], True), ("AX_MPR_RIGHT", ["R"], False)])
def test_d2_absent_single_and_repeated_side_are_distinct(description, sides, eligible):
    g = routed([header(i, SeriesDescription=description, ImageLaterality="") for i in range(2)])[0]
    assert g["route_eligible"] is eligible and g["header_description_sides"] == sides
    if eligible:
        assert "header_description_side_conflict" not in g["routing_issues"]
    else:
        assert "header_description_side_conflict" in g["routing_issues"]


def realpath_row(root, source="AX_MPR", key="AX"):
    r = row(source, key, root)
    r["source_archive_member"] = str(root / a.OLD_RAW / "image03/00m/0.C.2/9000001/20050118" / (key+".tar.gz"))+"::001"
    return r


def test_exact_prefix_relocation_no_basename_or_old_fallback(tmp_path):
    r = realpath_row(tmp_path)
    new = a.relocated_archive(r, tmp_path)
    assert new == tmp_path / a.NEW_RAW / "image03/00m/0.C.2/9000001/20050118/AX.tar.gz"
    assert a.audit_archive(r, join(), tmp_path)["status"] == "MISSING_RAW"
    for name in ("storage/external_data/oai/raw/V00wrong/image03/00m/x.tar.gz", a.OLD_RAW+"/../x.tar.gz"):
        with pytest.raises(ValueError):
            a.relocated_archive(dict(r, source_archive_member=name), tmp_path)


def test_archive_receipt_header_only_and_preserves_raw(tmp_path, monkeypatch):
    import pydicom.dataset
    monkeypatch.setattr(pydicom.dataset.Dataset, "pixel_array", property(lambda _: pytest.fail("pixel decode forbidden")))
    r = realpath_row(tmp_path)
    path = a.relocated_archive(r, tmp_path)
    write_tar(path, [(str(i), dicom_bytes(h)) for i, h in enumerate([
        header(99, "Sagittal", ImageType=["PROJECTION IMAGE"]), header(1), header(0)])])
    original = path.read_bytes()
    result = a.audit_archive(r, join(), tmp_path)
    assert result["status"] == "READ_OK" and result["retained_frame_count"] == 2
    assert result["excluded_frames"][0]["reason"] == "projection_image"
    assert result["groups"][0]["route_eligible"]
    assert "DO_NOT_EXPORT" not in a.dump(result)
    assert path.read_bytes() == original and not (tmp_path / "storage/cache").exists()


def test_d3_archive_member_shuffle_preserves_geometry_receipt(tmp_path):
    hs = [header(0), header(1, ImageOrientationPatient=[1, 0, 0, 0, 1, .00004],
        ImagePositionPatient=[0, 1, 2]), header(2)]
    entries = [(f"{i:03}", dicom_bytes(h)) for i, h in enumerate(hs)]
    receipts = []
    for name, ordered in (("forward", entries), ("reverse", entries[::-1]),
                          ("shuffled", [entries[2], entries[0], entries[1]])):
        root = tmp_path/name
        r = realpath_row(root)
        write_tar(a.relocated_archive(r, root), ordered)
        rec = a.audit_archive(r, join(), root)
        assert rec["status"] == "READ_OK" and rec["groups"][0]["route_eligible"]
        receipts.append((rec["header_sha256"], rec["groups"], rec["excluded_frames"]))
    assert all(receipt == receipts[0] for receipt in receipts)


def test_partial_archive_cannot_admit_good_stack(tmp_path):
    r = realpath_row(tmp_path)
    write_tar(a.relocated_archive(r, tmp_path), [(str(i), dicom_bytes(header(i))) for i in range(2)]+[("bad", b"bad")])
    rec = a.audit_archive(r, join(), tmp_path)
    assert rec["status"] == "READ_ERROR" and not any(g["route_eligible"] for g in rec["groups"])


def write_table(root, path, columns, rows, delimiter=",", dictionary=False):
    p = root/path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(columns)
        if dictionary:
            writer.writerow(["dictionary"]*len(columns))
        writer.writerows(rows)


def join_fixture(root, *, side="2: Left", visit="V00", duplicate=False):
    r = realpath_row(root)
    uri = "s3://example/image03/00m/0.C.2/9000001/20050118/AX.tar.gz"
    write_table(root, a.SELECTION, ["ID", "side", "series", "image_file"], [["9000001", "LEFT", "AX_MPR", uri]])
    write_table(root, a.IMAGE03, ["src_subject_id", "image_file", "image_description", "visit"],
                [["9000001", uri, "AX_MPR_LEFT", visit]], "\t", True)
    values = [["9000001", side, "22", "1"], ["9000001", side, "30", "1"]]
    if duplicate:
        values.append(values[0])
    write_table(root, a.MOAKS, ["ID", "SIDE", "READPRJ", "VERSION"], values, "|")
    return r


def test_moaks_side_join_multiple_readings_are_not_collapsed(tmp_path):
    r = join_fixture(tmp_path)
    j = a.build_joins([r], tmp_path)[r["series_key"]]
    assert j["join_verified"] and j["moaks_read_count"] == 2
    assert j["clinical_values_checked"] is False


@pytest.mark.parametrize("kwargs,problem", [
    (dict(side="1: Right"), "missing_MOAKS_participant_side"),
    (dict(visit="V01"), "image03_vs_baseline_visit_conflict"),
    (dict(duplicate=True), "duplicate_MOAKS_reading_identity")])
def test_join_conflicts_fail_closed(tmp_path, kwargs, problem):
    r = join_fixture(tmp_path, **kwargs)
    j = a.build_joins([r], tmp_path)[r["series_key"]]
    assert not j["join_verified"] and problem in j["issues"]


def test_cross_archive_uid_reuse_across_knees_is_quarantined():
    records = [dict(series_key=str(i), groups=routed([header(0), header(1)]), excluded_frames=[]) for i in range(2)]
    found = a.cross_archive_issues(records)
    assert found == dict(duplicate_series_uids=1, duplicate_sop_frames=2)
    assert not any(g["route_eligible"] for r in records for g in r["groups"])


def test_excluded_frame_series_uid_still_participates_in_ownership_check():
    records = [dict(series_key="one", groups=routed([header(0), header(1)]), excluded_frames=[]),
        dict(series_key="two", groups=[], excluded_frames=[dict(series_instance_uid="1.2.3",
            sop_instance_uid="1.2.777", frame_number=1)])]
    assert a.cross_archive_issues(records)["duplicate_series_uids"] == 1
    assert not records[0]["groups"][0]["route_eligible"]


def test_before_after_counts_use_same_roster_correct_sag_and_empty_sag0():
    r = row("SAG_IW_TSE", "sag")
    r["fluid_sensitive"] = "1"
    rec = dict(series_key="sag", status="READ_OK", groups=routed(
        [header(i, "Sagittal", "SAG_IW_TSE") for i in range(2)], r))
    counts, slots = a.routing_tables([r], [rec])
    for stats in counts.values():
        assert stats[0]["BEFORE"] == stats[0]["AFTER"] == 1
        assert stats[1]["AFTER"] == 0
        assert sum(x["AFTER"] for x in stats) == 1
    assert len(slots) == 10


def test_verification_binds_inputs_code_tests_report(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "fingerprint", lambda root: {"code": "one"})
    monkeypatch.setattr(a, "input_hashes", lambda root: {"input": "two"})
    note = dict(verdict="PASS", reviewer="astra", tool_version=a.VERSION, tests_passed=True,
        test_command="synthetic B tests", review_report="independent report", code_sha256={"code": "one"},
        input_sha256={"input": "two"})
    p = tmp_path/"note.json"
    p.write_text(json.dumps(note))
    assert a.verification(p, tmp_path)["verification_sha256"] == a.sha(p)
    for change in (dict(verdict="PASS-WITH-NOTES"), dict(input_sha256={}), dict(tests_passed=False),
                   dict(review_report=""), dict(code_sha256={})):
        p.write_text(json.dumps(dict(note, **change)))
        with pytest.raises(ValueError):
            a.verification(p, tmp_path)


def test_v204_allow_listed_reviewers_accepted_others_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "fingerprint", lambda root: {"code": "one"})
    monkeypatch.setattr(a, "input_hashes", lambda root: {"input": "two"})
    note = dict(verdict="PASS", reviewer="claude-opus", tool_version=a.VERSION, tests_passed=True,
        test_command="synthetic B tests", review_report="independent report", code_sha256={"code": "one"},
        input_sha256={"input": "two"})
    p = tmp_path/"note.json"
    assert a.REVIEWERS == ("astra", "claude-opus")
    for reviewer in a.REVIEWERS:
        p.write_text(json.dumps(dict(note, reviewer=reviewer)))
        got = a.verification(p, tmp_path)
        assert got["reviewer"] == reviewer and got["verification_sha256"] == a.sha(p)
    refused = [
        (dict(reviewer="sol"), "independent_reviewer_PASS_required"),
        (dict(reviewer="Claude-Opus"), "independent_reviewer_PASS_required"),
        (dict(reviewer=""), "independent_reviewer_PASS_required"),
        (dict(reviewer=["claude-opus"]), "independent_reviewer_PASS_required"),
        (dict(verdict="FAIL"), "independent_reviewer_PASS_required"),
        (dict(tool_version="oai-header-audit-v2.0.3"), "independent_reviewer_PASS_required"),
        (dict(code_sha256={"code": "stale"}), "code_or_input_hash_changed_reverify"),
        (dict(input_sha256={"input": "stale"}), "code_or_input_hash_changed_reverify"),
        (dict(tests_passed="yes"), "test_receipt_and_review_report_required"),
        (dict(test_command=""), "test_receipt_and_review_report_required"),
    ]
    for change, message in refused:
        p.write_text(json.dumps(dict(note, **change)))
        with pytest.raises(ValueError, match=message):
            a.verification(p, tmp_path)
    missing = dict(note)
    del missing["reviewer"]
    p.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="independent_reviewer_PASS_required"):
        a.verification(p, tmp_path)


def synthetic_run_setup(tmp_path, monkeypatch, with_raw=True):
    r = join_fixture(tmp_path)
    index = tmp_path/a.old.INDEX
    index.parent.mkdir(parents=True)
    index.write_text("synthetic index")
    sample = tmp_path/a.old.SAMPLE
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(json.dumps(dict(index_sha256=a.sha(index))))
    note = tmp_path/"note.json"
    note.write_text("synthetic-only")
    monkeypatch.setattr(a.old, "read_oai_index", lambda root: [r])
    monkeypatch.setattr(a, "EXPECTED_ARCHIVES", 1)
    monkeypatch.setattr(a, "fingerprint", lambda root: {"synthetic": "code"})
    monkeypatch.setattr(a, "verification", lambda p, root: dict(
        verification_note=str(p), verification_sha256=a.sha(p), reviewer="claude-opus"))
    if with_raw:
        write_tar(a.relocated_archive(r, tmp_path), [(str(i), dicom_bytes(header(i))) for i in range(2)])
    return r, note


def test_full_command_synthetic_end_to_end_and_receipt_hashes(tmp_path, monkeypatch):
    r, note = synthetic_run_setup(tmp_path, monkeypatch)
    out = tmp_path/"artifacts/audit"
    assert a.run(out, note, tmp_path) == 0
    summary = json.loads((out/"summary.json").read_text())
    assert summary["status"] == "COMPLETE_HEADERS" and summary["eligible_groups"] == 1
    assert summary["index_archives"] == summary["inventory_archives"] == 1
    assert summary["gpu_hours"] == 0 and not summary["pixel_cache_writes"]
    for p, digest in summary["output_sha256"].items():
        assert a.sha(out/p) == digest
        for line in (out/p).read_text().splitlines():
            rec = json.loads(line)
            assert rec["tool_version"] == a.VERSION and rec["verification_note"] == str(note)
            assert rec["reviewer"] == "claude-opus"
    assert json.loads((out/"routing.jsonl").read_text())["route_eligible"]
    assert len((out/"slot-selections.jsonl").read_text().splitlines()) == 10
    with pytest.raises(FileExistsError):
        a.run(out, note, tmp_path)


def test_full_command_missing_raw_stays_incomplete_and_no_false_after(tmp_path, monkeypatch):
    _, note = synthetic_run_setup(tmp_path, monkeypatch, with_raw=False)
    out = tmp_path/".local/audit"
    assert a.run(out, note, tmp_path) == 2
    summary = json.loads((out/"summary.json").read_text())
    assert summary["status"] == "INCOMPLETE_RAW_EVIDENCE"
    assert all(s["AFTER"] == 0 for stats in summary["counts"].values() for s in stats)


def test_timeout_and_changed_raw_have_no_completion(tmp_path, monkeypatch):
    r, note = synthetic_run_setup(tmp_path, monkeypatch)
    out = tmp_path/".local/timeout"
    with pytest.raises(TimeoutError):
        a.run(out, note, tmp_path, max_seconds=-1)
    assert not (out/"summary.json").exists()


def test_changed_raw_has_no_completion(tmp_path, monkeypatch):
    r, note = synthetic_run_setup(tmp_path, monkeypatch)
    original = a.audit_archive
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with a.relocated_archive(r, tmp_path).open("ab") as f:
            f.write(b"changed")
        return result
    monkeypatch.setattr(a, "audit_archive", changed)
    out = tmp_path/".local/changed"
    with pytest.raises(ValueError, match="raw_archive_changed"):
        a.run(out, note, tmp_path)
    assert not (out/"summary.json").exists()


def test_deadline_during_archive_propagates_instead_of_read_error(tmp_path):
    r = realpath_row(tmp_path)
    write_tar(a.relocated_archive(r, tmp_path), [(str(i), dicom_bytes(header(i))) for i in range(2)])
    with pytest.raises(TimeoutError):
        a.audit_archive(r, join(), tmp_path, deadline=-1)


def test_final_archive_overrun_cannot_emit_success(tmp_path, monkeypatch):
    r, note = synthetic_run_setup(tmp_path, monkeypatch)
    original = a.audit_archive
    def slow(*args, **kwargs):
        rec = original(*args, **kwargs)
        monkeypatch.setattr(a.time, "monotonic", lambda: float("inf"))
        return rec
    monkeypatch.setattr(a, "audit_archive", slow)
    out = tmp_path/".local/final-overrun"
    with pytest.raises(TimeoutError):
        a.run(out, note, tmp_path)
    assert not (out/"summary.json").exists()
