"""CPU synthetic-DICOM checks; never import training/acquisition entry points."""
import hashlib
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
import raptor_dicom as rd

SPEC = {"slots": [["Sagittal", 1, 4], ["Coronal", 0, 3]], "depth": 7,
        "img": 8, "crop_mm": 6., "span": [0.02, 0.98]}


def _dicom(path, pixels, position, instance=1, mono="MONOCHROME2", slope=1., intercept=0., spacing=1.):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage; dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1; dataset.PhotometricInterpretation = mono
    dataset.BitsAllocated = 16; dataset.BitsStored = 16; dataset.HighBit = 15; dataset.PixelRepresentation = 0
    dataset.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
    dataset.ImagePositionPatient = [0., 0., float(position)]
    dataset.InstanceNumber = instance; dataset.PixelSpacing = [spacing, spacing]
    dataset.RescaleSlope = slope; dataset.RescaleIntercept = intercept
    dataset.PixelData = pixels.astype("<u2").tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return dataset


def _rows(uid, series="sag", plane="Sagittal", fluid=1):
    return [{rd.UID: uid, rd.SERIES: series, "Anatomical_Plane": plane, "Fluid_Sensitive": fluid}]


def test_healthy_fixed_stack_matches_independent_contract_math(tmp_path):
    uid = "unseen-study"
    physical = []
    for index, name in [(2, "a.dcm"), (0, "z.dcm"), (1, "m.dcm")]:
        pixels = np.arange(100, dtype=np.uint16).reshape(10, 10) + index * 100
        _dicom(tmp_path / uid / "sag" / name, pixels, index, instance=100-index,
               mono="MONOCHROME1", slope=2., intercept=-5.)
        physical.append((index, (pixels.astype(np.float64) * 2. - 5.).astype(np.float32)))
    physical = [a.max() - a for _, a in sorted(physical, key=lambda r: r[0])]
    # n=3, span=(.02,.98): indices [0,0,1,1], intentionally weighted twice.
    selected = [physical[i] for i in [0, 0, 1, 1]]
    lo, hi = np.percentile(np.concatenate([a.ravel() for a in selected]), [2., 98.])
    expected = np.zeros((7, 8, 8), np.uint8)
    for index, a in enumerate(selected):
        normalized = np.clip((a-lo)/(hi-lo+1e-6), 0, 1)
        expected[index] = (cv2.resize(normalized[2:8, 2:8], (8, 8), interpolation=cv2.INTER_AREA)*255).astype(np.uint8)
    with rd.DirectoryDicomSource(tmp_path) as source:
        volume, mask, receipt = rd.prepare_study(uid, _rows(uid), source, SPEC)
    np.testing.assert_array_equal(volume, expected)
    np.testing.assert_array_equal(mask, [1, 1, 1, 1, 0, 0, 0])
    assert receipt["slots"][0]["slices"] == [0, 0, 1, 1]
    assert receipt["slots"][1]["status"] == "metadata_absent"
    assert not volume.flags.writeable


def test_acquisition_absence_is_distinct_from_files_or_decode_failure(tmp_path):
    uid = "unseen"
    source = rd.DirectoryDicomSource(tmp_path)
    volume, mask, receipt = rd.prepare_study(uid, [], source, SPEC)
    assert not mask.any() and not volume.any() and receipt["status"] == "metadata_absent"
    with pytest.raises(rd.DicomPreparationError) as missing:
        rd.prepare_study(uid, _rows(uid), source, SPEC)
    assert missing.value.receipt["slots"][0]["status"] == "acquired_files_missing"
    file = tmp_path / uid / "sag" / "a.dcm"
    dataset = _dicom(file, np.arange(100).reshape(10, 10), 0)
    del dataset.PixelData
    dataset.save_as(file, enforce_file_format=True)
    with pytest.raises(rd.DicomPreparationError) as decode:
        rd.prepare_study(uid, _rows(uid), source, SPEC)
    assert decode.value.receipt["slots"][0]["status"] == "decode_failed"


def test_black_acquired_pixels_keep_acquisition_status(tmp_path):
    uid = "black-study"
    _dicom(tmp_path / uid / "sag" / "a.dcm", np.zeros((10, 10)), 0)
    volume, mask, receipt = rd.prepare_study(uid, _rows(uid), rd.DirectoryDicomSource(tmp_path), SPEC)
    assert not mask.any() and not volume.any()
    assert receipt["status"] == "rendered" and receipt["acquired_slices"] == 4
    assert receipt["slots"][0]["status"] == "rendered"


def test_selection_keeps_csv_order_preference_and_never_reuses_series():
    rows = _rows("u", "first", fluid=0) + _rows("u", "preferred", fluid=1) + _rows("u", "later", fluid=1)
    assert rd.pick_series(rows, "Sagittal", 1, set())[rd.SERIES] == "preferred"
    assert rd.pick_series(rows, "Sagittal", 1, {"preferred"})[rd.SERIES] == "later"
    assert rd.pick_series(rows, "Sagittal", 1, {"preferred", "later"})[rd.SERIES] == "first"
    assert rd.pick_series(_rows("u", fluid="unknown"), "Sagittal", 1, set())[rd.SERIES] == "sag"


def _indexed_zip(directory, archive, index):
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for p in sorted(directory.rglob("*.dcm")):
            target.write(p, "test_series/" + p.relative_to(directory).as_posix())
    fields = ("compress_type", "comment", "extra", "create_system", "create_version", "extract_version",
              "reserved", "flag_bits", "volume", "internal_attr", "external_attr", "header_offset", "CRC",
              "compress_size", "file_size")
    with sqlite3.connect(index) as database, zipfile.ZipFile(archive) as source:
        database.execute("CREATE TABLE files (path TEXT, split TEXT, uid TEXT, series TEXT, info TEXT)")
        for info in source.infolist():
            # Production index omits binary extra/comment; empty defaults suffice here.
            attrs = {k: getattr(info, k) for k in fields if k not in {"comment", "extra"}}
            _, uid, series, _ = info.filename.split("/")
            database.execute("INSERT INTO files VALUES (?,?,?,?,?)", (info.filename, "test", uid, series, json.dumps(attrs)))


def test_directory_and_read_only_indexed_zip_are_pixel_identical(tmp_path):
    directory = tmp_path / "images"
    uid = "new-uid"
    _dicom(directory / uid / "sag" / "b.dcm", np.arange(100).reshape(10, 10), 3)
    _dicom(directory / uid / "sag" / "a.dcm", np.arange(100).reshape(10, 10) + 40, 1)
    archive, index = tmp_path / "input.zip", tmp_path / "index.sqlite"
    _indexed_zip(directory, archive, index)
    before = hashlib.sha256(index.read_bytes()).hexdigest()
    with rd.DirectoryDicomSource(directory) as a, rd.IndexedZipDicomSource(archive, index) as b:
        av, am, ar = rd.prepare_study(uid, _rows(uid), a, SPEC)
        bv, bm, br = rd.prepare_study(uid, _rows(uid), b, SPEC)
        np.testing.assert_array_equal(av, bv); np.testing.assert_array_equal(am, bm)
        assert ar["volume_sha256"] == br["volume_sha256"]
        with pytest.raises(sqlite3.OperationalError):
            b.connection.execute("DELETE FROM files")
    assert hashlib.sha256(index.read_bytes()).hexdigest() == before


def test_unseen_ids_no_labels_and_one_study_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "_contract_spec", lambda contract: dict(SPEC))
    ids = ["not-in-training-a", "not-in-training-b"]
    for uid in ids:
        _dicom(tmp_path / uid / "sag" / "a.dcm", np.arange(100).reshape(10, 10), 0)
    test = tmp_path / "test.csv"; test.write_text("StudyInstanceUID\n" + "\n".join(ids))
    series = tmp_path / "test_series.csv"
    series.write_text("StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive\n" +
                      "\n".join(f"{u},sag,Sagittal,1" for u in ids))
    source = rd.DirectoryDicomSource(tmp_path)
    calls = []
    original = source.read
    def tracked(member):
        calls.append(member.path)
        return original(member)
    monkeypatch.setattr(source, "read", tracked)
    corpus = rd.DicomCorpus(test, series, source)
    assert corpus.ids == ids
    first = corpus.volume(ids[0]); n = len(calls)
    corpus.mask(ids[0]); assert len(calls) == n
    second = corpus.volume(ids[1]); assert corpus._cached_uid == ids[1]
    assert first is not second and len(corpus.receipts) == 2
    assert corpus.manifest["cache_max_studies"] == 1
    with pytest.raises(KeyError):
        corpus.volume("not-requested")


def test_metadata_duplicate_rejected_before_pixels(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "_contract_spec", lambda contract: dict(SPEC))
    source = rd.DirectoryDicomSource(tmp_path)
    with pytest.raises(ValueError, match="duplicate series"):
        rd.DicomCorpus.from_records(["u"], _rows("u") * 2, source)
    with pytest.raises(ValueError, match="unique study"):
        rd.DicomCorpus.from_records(["u", "u"], [], source)


def test_header_failure_is_not_metadata_absence(tmp_path):
    file = tmp_path / "u" / "sag" / "broken.dcm"
    file.parent.mkdir(parents=True); file.write_bytes(b"not a DICOM")
    with pytest.raises(rd.DicomPreparationError) as failed:
        rd.prepare_study("u", _rows("u"), rd.DirectoryDicomSource(tmp_path), SPEC)
    assert failed.value.receipt["status"] == "failed"
    assert failed.value.receipt["slots"][0]["status"] == "header_failed"


def test_corpus_interface_runs_cpu_batched_inference_on_unseen_ids(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from raptor_runtime import predict_batches
    monkeypatch.setattr(rd, "_contract_spec", lambda contract: dict(SPEC))
    ids = ["unseen-a", "unseen-b"]
    for uid in ids:
        _dicom(tmp_path / uid / "sag" / "a.dcm", np.arange(100).reshape(10, 10), 0)
    corpus = rd.DicomCorpus.from_records(ids, _rows(ids[0]) + _rows(ids[1]), rd.DirectoryDicomSource(tmp_path))
    class TinyModel(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 2, 3, 4))[:, None].expand(-1, 12)
    predictions, receipt = predict_batches(TinyModel(), corpus, ids, 3, 8, "none", "cpu",
                                          norm="none", batch=2, window_policy="legacy", log=lambda _: None)
    assert predictions.shape == (2, 12) and np.isfinite(predictions).all()
    assert [r["uid"] for r in receipt["input_tensor_hashes"]] == ids
    assert receipt["fp32_fallbacks"] == 0 and corpus._cached_uid == ids[-1]


def test_cli_rejects_existing_output_before_checkpoint_load(tmp_path):
    from raptor_serve import main
    output = tmp_path / "raw.csv"; output.write_text("preserved")
    with pytest.raises(FileExistsError):
        main(["--ckpt", "missing.pt", "--test-csv", "missing.csv", "--series-csv", "missing.csv",
              "--dicom-root", str(tmp_path), "--out", str(output)])
    assert output.read_text() == "preserved"
