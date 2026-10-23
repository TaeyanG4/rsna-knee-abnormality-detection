"""Header-only builder integration: no competition labels or reports."""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
import pytest

from tools import hb123_laterality_v1 as builder
from training.arms import laterality_v1 as lat


def dicom(side, offset):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.3.4"
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Laterality = side
    ds.BodyPartExamined = "KNEE"
    ds.ImageOrientationPatient = [1, 0, 0, 0, 0, -1]
    ds.ImagePositionPatient = [offset - 49.5, 0, 0]
    ds.PixelSpacing, ds.Rows, ds.Columns = [1, 1], 100, 100
    # No pixel data, report, patient name or label fields are needed.
    out = BytesIO()
    pydicom.dcmwrite(out, ds, enforce_file_format=True)
    return out.getvalue()


def inputs(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    np.save(corpus / "ids.npy", np.asarray(["a", "b", "c"]))
    (corpus / "manifest.json").write_text("{}")
    meta = tmp_path / "series.csv"
    with meta.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([lat.UID, "SeriesInstanceUID", "Anatomical_Plane"])
        for uid in ("a", "b", "c"):
            writer.writerow([uid, "1", "Coronal"])
    archive = tmp_path / "dicoms.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for uid, side, offset in (("a", "L", 80), ("b", "R", -80)):
            for n in range(5):
                z.writestr(f"train_images/{uid}/1/{n}.dcm", dicom(side, offset))
        z.writestr("train_images/c/1/0.dcm", b"unreadable DICOM")
    return argparse.Namespace(metadata=[str(meta)], archive=str(archive), corpus=str(corpus),
                              out=str(tmp_path / "side-map.json"), limit=0)


def test_builder_real_dicom_header_only_roundtrip(tmp_path, capsys):
    args = inputs(tmp_path)
    result = builder.build(args)
    counts = result["counts"]
    assert counts["studies"] == 3 and counts["headers_read"] == 6
    assert counts["side_L"] == counts["side_R"] == counts["side_U"] == 1
    assert counts["header_errors"] == 1 and counts["missing_series"] == 0
    assert counts["tag_geometry_agree"] == 2
    checked = builder.validate(argparse.Namespace(side_map=args.out, corpus=args.corpus))
    assert checked["status"] == "VALIDATED"
    first = Path(args.out).read_bytes()
    assert lat.SideMap(args.out).side("c") == "U"
    builder.build(args)
    assert first == Path(args.out).read_bytes()
    assert capsys.readouterr().out == ""


def test_validator_rejects_input_and_source_drift(tmp_path):
    args = inputs(tmp_path)
    builder.build(args)
    check_args = argparse.Namespace(side_map=args.out, corpus=args.corpus)
    manifest = Path(args.corpus) / "manifest.json"
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="input changed"):
        builder.validate(check_args)
    manifest.write_text("{}")
    qa_path = Path(args.out).with_suffix(".qa.json")
    qa = json.loads(qa_path.read_bytes())
    qa["sources"]["training/arms/laterality_v1.py"] = "0" * 64
    builder.write(qa_path, qa)
    with pytest.raises(ValueError, match="source changed"):
        builder.validate(check_args)


def test_validator_rejects_tamper_and_partial_coverage(tmp_path):
    args = inputs(tmp_path)
    args.limit = 1
    builder.build(args)
    check_args = argparse.Namespace(side_map=args.out, corpus=args.corpus)
    with pytest.raises(ValueError, match="lacks"):
        builder.validate(check_args)
    with Path(args.out).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="digest"):
        builder.validate(check_args)


def test_metadata_refuses_duplicates_and_bad_schema(tmp_path):
    args = inputs(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        builder.metadata(args.metadata * 2)
    Path(args.metadata[0]).write_text("unrelated\nvalue\n")
    with pytest.raises(ValueError, match="schema"):
        builder.metadata(args.metadata)
