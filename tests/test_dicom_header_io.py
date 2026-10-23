"""CPU tests of the opt-in directory header reader, including fail-closed paths."""
import sys
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
import raptor_dicom as rd
from test_raptor_serving import SPEC, _dicom, _indexed_zip, _rows


def _result(root, enabled):
    with rd.DirectoryDicomSource(root, header_only=enabled) as source:
        result = rd.prepare_study("u", _rows("u"), source, SPEC)
        return result, dict(source.header_io)


def test_default_preserves_full_reads_and_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "u/sag/a.dcm"
    _dicom(path, np.arange(4096).reshape(64, 64), 1)
    source = rd.DirectoryDicomSource(tmp_path)
    calls, original = [], source.read
    def read(member):
        calls.append(member.path)
        return original(member)
    monkeypatch.setattr(source, "read", read)
    source.read_header(rd.DicomMember("u/sag/a.dcm"))
    assert calls == ["u/sag/a.dcm"]
    assert source.descriptor == {"kind": "directory", "path": str(tmp_path.resolve())}
    assert source.header_io["files_attempted"] == 0


def test_byte_exact_volume_mask_receipt_with_ties_fallback_and_spacing(tmp_path):
    for name, position, spacing in [("z", 4, .6), ("a", 1, 1.2), ("b", 1, .8), ("c", 3, .5)]:
        path = tmp_path / f"u/sag/{name}.dcm"
        ds = _dicom(path, np.arange(256).reshape(16, 16) + position * 70, position,
                    spacing=spacing, mono="MONOCHROME1", slope=1.3, intercept=-20.)
        if name == "c":
            del ds.ImageOrientationPatient
            ds.InstanceNumber = 3
            ds.save_as(path, enforce_file_format=True)
    (av, am, ar), _ = _result(tmp_path, False)
    (bv, bm, br), stats = _result(tmp_path, True)
    np.testing.assert_array_equal(av, bv)
    np.testing.assert_array_equal(am, bm)
    assert ar == br
    assert br["slots"][0]["instance_order_fallbacks"] == 1
    assert stats["files_completed"] == stats["files_attempted"] == 4


def test_header_reads_do_not_consume_pixel_payload(tmp_path, monkeypatch):
    path = tmp_path / "u/sag/a.dcm"
    _dicom(path, np.arange(512 * 512).reshape(512, 512), 1)
    source = rd.DirectoryDicomSource(tmp_path, header_only=True)
    def forbidden(member):
        raise AssertionError("streaming header may not delegate to full read")
    monkeypatch.setattr(source, "read", forbidden)
    ds = source.read_header(rd.DicomMember("u/sag/a.dcm"))
    assert "PixelData" not in ds and ds.Rows == 512
    assert source.header_io["logical_bytes_read"] < path.stat().st_size * .01
    assert source.descriptor["header_reader"] == rd.HEADER_READER_VERSION


@pytest.mark.parametrize("failure", ["header", "pixels", "nonfinite"])
def test_failure_stage_and_receipt_are_preserved(tmp_path, failure):
    path = tmp_path / "u/sag/a.dcm"
    ds = _dicom(path, np.arange(100).reshape(10, 10), 1)
    if failure == "header":
        path.write_bytes(b"broken header")
    else:
        if failure == "pixels":
            del ds.PixelData
        else:
            ds.PixelSpacing = ["nan", "nan"]
        ds.save_as(path, enforce_file_format=True)
    errors = []
    for enabled in [False, True]:
        with pytest.raises(rd.DicomPreparationError) as exc:
            _result(tmp_path, enabled)
        errors.append(exc.value.receipt)
    assert errors[0] == errors[1]
    assert errors[0]["slots"][0]["status"] == ("decode_failed" if failure == "pixels" else "header_failed")


def test_indexed_zip_keeps_full_read_fallback(tmp_path, monkeypatch):
    directory = tmp_path / "images"
    _dicom(directory / "u/sag/a.dcm", np.arange(100).reshape(10, 10), 1)
    archive, index = tmp_path / "input.zip", tmp_path / "index.sqlite"
    _indexed_zip(directory, archive, index)
    with rd.IndexedZipDicomSource(archive, index) as source:
        calls, original = [], source.read
        def read(member):
            calls.append(member.path)
            return original(member)
        monkeypatch.setattr(source, "read", read)
        zv, zm, zr = rd.prepare_study("u", _rows("u"), source, SPEC)
    (dv, dm, dr), _ = _result(directory, True)
    assert len(calls) == 2  # one complete header read and one complete selected-pixel read
    np.testing.assert_array_equal(zv, dv)
    np.testing.assert_array_equal(zm, dm)
    assert zr["volume_sha256"] == dr["volume_sha256"]


@pytest.mark.parametrize("enabled", [False, True])
def test_path_containment_applies_before_header_read(tmp_path, enabled):
    source = rd.DirectoryDicomSource(tmp_path, header_only=enabled)
    with pytest.raises(ValueError, match="escapes"):
        source.read_header(rd.DicomMember("../outside.dcm"))


def test_opt_in_requires_bool(tmp_path):
    with pytest.raises(TypeError, match="explicit bool"):
        rd.DirectoryDicomSource(tmp_path, header_only="false")


def test_benchmark_cannot_claim_parity_with_missing_reference_uid(tmp_path):
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import dicom_header_benchmark as benchmark
    (tmp_path / "selection.json").write_text(json.dumps({"ids": ["u"]}), encoding="utf-8")
    (tmp_path / "reference.json").write_text(json.dumps({"study_receipts": []}), encoding="utf-8")
    (tmp_path / "series.csv").write_text("StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive\nu,sag,Sagittal,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="omits requested"):
        benchmark.main(["--selection", str(tmp_path / "selection.json"), "--series-csv", str(tmp_path / "series.csv"),
                        "--reference-receipt", str(tmp_path / "reference.json"), "--count", "1",
                        "--baseline-adapter", rd.__file__, "--dicom-root", str(tmp_path), "--out", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


def _operational_pair():
    import copy
    receipt = {"uid": "u", "volume_sha256": "pixels", "mask_sha256": "mask", "slots": [
        {"slot": 0, "series": "s", "ordered_sources_sha256": "fused-position-sha",
         "selected_sources": [{"path": "u/s/a.dcm", "sha256": "original-file"}], "normalization": [0., 100.]}]}
    proof = {"uid": "u", "slots": [{"slot": 0, "series": "s", "ordered_sources_sha256": "fused-position-sha",
        "ordered_path_spacing_sha256": "all-paths-and-spacing", "raw_geometry_sha256": "exact-geometry",
        "files": 30, "median_spacing": .5, "instance_order_fallbacks": 0}]}
    actual, actual_proof = copy.deepcopy(receipt), copy.deepcopy(proof)
    actual["slots"][0]["ordered_sources_sha256"] = "unfused-position-sha"
    actual_proof["slots"][0]["ordered_sources_sha256"] = "unfused-position-sha"
    return receipt, actual, proof, actual_proof


def test_v2_position_digest_difference_requires_exact_operational_proof():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from dicom_header_benchmark import operational_receipt_equal
    values = _operational_pair()
    assert operational_receipt_equal(*values)
    assert values[0] != values[1]  # v1 still fails; comparator does not mutate evidence


@pytest.mark.parametrize("mutation", ["order", "geometry", "spacing", "selected_pixel", "volume", "missing_slot", "unbound_digest", "extra_field"])
def test_v2_fails_semantic_or_provenance_differences(mutation):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from dicom_header_benchmark import operational_receipt_equal
    reference, actual, proof, actual_proof = _operational_pair()
    if mutation == "order": actual_proof["slots"][0]["ordered_path_spacing_sha256"] = "reordered"
    if mutation == "geometry": actual_proof["slots"][0]["raw_geometry_sha256"] = "changed-header"
    if mutation == "spacing": actual_proof["slots"][0]["median_spacing"] = .500000000000001
    if mutation == "selected_pixel": actual["slots"][0]["selected_sources"][0]["sha256"] = "changed-pixels"
    if mutation == "volume": actual["volume_sha256"] = "changed-volume"
    if mutation == "missing_slot": actual_proof["slots"] = []
    if mutation == "unbound_digest": actual_proof["slots"][0]["ordered_sources_sha256"] = "not-the-receipt"
    if mutation == "extra_field": actual["unexpected"] = True
    assert not operational_receipt_equal(reference, actual, proof, actual_proof)
