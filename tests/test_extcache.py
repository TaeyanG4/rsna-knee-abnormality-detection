"""CPU-only synthetic archive and Raptor parity checks for EXP135."""
from __future__ import annotations

import io
import json
import pickle
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from tools.extcache import common, sources


def _dicom_bytes(i, plane="Sagittal", mono1=False):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset("synthetic", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows = ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 2
    ds.InstanceNumber = i + 1
    ds.SeriesDescription = "SAG_IW_TSE_RIGHT" if plane == "Sagittal" else "CORPDFS"
    ds.ImageOrientationPatient = ([0, 1, 0, 0, 0, 1] if plane == "Sagittal" else [1, 0, 0, 0, 0, 1])
    ds.ImagePositionPatient = ([i * 2, 0, 0] if plane == "Sagittal" else [0, i * 2, 0])
    ds.PixelData = (np.arange(64, dtype=np.uint16).reshape(8, 8) + i * 7).tobytes()
    out = io.BytesIO()
    ds.save_as(out)
    return out.getvalue()


def _tar(path, members, mode):
    with tarfile.open(path, mode) as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_dicom_general_production_matches_raptor(tmp_path, monkeypatch):
    root = tmp_path
    members = [(f"s/{i:02d}.dcm", _dicom_bytes(i, mono1=True)) for i in (3, 1, 2, 0)]
    tarpath = root / "tiny.tar.gz"
    _tar(tarpath, members, "w:gz")
    task = (str(tarpath), "OAI_1_R_V00", "OAI_1_R_V00_s", "1", "R", "V00", "SAG_IW_TSE_RIGHT")
    ser = sources.read_oai_task(task)
    assert ser.meta["plane"] == "Sagittal"
    assert ser.meta["fluid_sensitive"] == 1
    assert ser.ipp[:, 0].tolist() == [0, 2, 4, 6]
    common.add_series("oai", ser, root)
    common.build_production("oai", root, shard_size=1)
    report = common.verify("oai", root)
    assert report["errors"] == []
    assert report["production_studies"] == 1

    import cv2
    import raptor_corpus
    by_name = dict(members)
    class Rows(list):  # raptor_corpus calls .fetchall() on the cursor (B fix)
        def fetchall(self):
            return list(self)
    class Connection:
        def execute(self, *_):
            return Rows((name, task[2], "{}") for name, _ in sorted(members))
    monkeypatch.setattr(raptor_corpus, "_member_bytes", lambda path, info: by_name[path])
    raptor_corpus._WORKER.update(pydicom=pydicom, cv2=cv2, con=Connection(),
                                 spec=dict(common.CONTRACTS[common.CONTRACT], depth=64),
                                 lut=pydicom.pixels.apply_modality_lut)
    row = dict(Anatomical_Plane="Sagittal", Fluid_Sensitive=1, SeriesInstanceUID=task[2])
    _, _, reference, ref_mask, _ = raptor_corpus.build_study(("synthetic", task[1], [row]))
    corpus = common.Corpus(common.prod_dir("oai", root))
    assert np.array_equal(corpus.volume(task[1]), reference)
    assert np.array_equal(corpus.mask(task[1]), ref_mask)
    assert corpus.mask(task[1]).sum() == 18
    # Reruns skip the hashed npz and preserve the shard bytes.
    before = common.sha256(common.prod_dir("oai", root) / "vols_shard0000.npy")
    common.add_series("oai", ser, root)
    common.build_production("oai", root, shard_size=1)
    assert before == common.sha256(common.prod_dir("oai", root) / "vols_shard0000.npy")


def test_fastmri_singlecoil_stream_and_spacing(tmp_path):
    h5 = tmp_path / "file100.h5"
    xml = b"""<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding><reconSpace>
      <matrixSize><x>8</x><y>8</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>160</x><y>120</y><z>4</z></fieldOfView_mm>
      </reconSpace></encoding></ismrmrdHeader>"""
    with h5py.File(h5, "w") as f:
        f.create_dataset("reconstruction_esc", data=np.arange(128, dtype=np.float32).reshape(2, 8, 8))
        f.create_dataset("ismrmrd_header", data=xml)
        f.attrs["acquisition"] = "CORPDFS_FBK"
    archive = tmp_path / "knee_singlecoil_val.tar.xz"
    _tar(archive, [("knee/file100.h5", h5.read_bytes())], "w:xz")
    with tarfile.open(archive, "r|xz") as tar:
        member = next(iter(tar))
        series = sources.fastmri_sc_series(member.name, sources._read_member(tar, member), str(archive))
    assert series.meta["fluid_sensitive"] == 1
    assert series.meta["plane"] == "Coronal"
    assert np.allclose(series.pixel_spacing, [15, 20])
    common.add_series("fastmri_sc", series, tmp_path)
    common.build_production("fastmri_sc", tmp_path, shard_size=1)
    assert common.verify("fastmri_sc", tmp_path)["errors"] == []


def test_fastmri_dicom_tar_stream(tmp_path):
    path = tmp_path / "storage/external_data/fastmri/raw/knee_DICOMs_batch1.tar.xz"
    path.parent.mkdir(parents=True)
    _tar(path, [(f"root/study1/seriesA/{i}.dcm", _dicom_bytes(i)) for i in range(2)], "w:xz")
    groups = list(sources.iter_fastmri_dcm(tmp_path))
    assert len(groups) == 1
    series = sources.fastmri_dcm_series(*groups[0])
    assert series.meta["study_key"] == "FMRI_DCM_study1"
    assert series.vol.shape == (2, 8, 8)


def test_spacing_unknown_and_restricted_pickle(tmp_path):
    from tools.exp132_external_aux import load_kneemri_pickle
    path = tmp_path / "volume.pck"
    arr = np.arange(128, dtype=np.uint8).reshape(2, 8, 8)
    path.write_bytes(pickle.dumps(arr, protocol=2))
    assert np.array_equal(load_kneemri_pickle(path), arr)
    bad = tmp_path / "bad.pck"
    bad.write_bytes(pickle.dumps(eval))
    import pytest
    with pytest.raises(pickle.UnpicklingError):
        load_kneemri_pickle(bad)
    meta = dict(source="kneemri", subject_id="1", knee_side="R", visit="",
                study_key="KMRI_1_1", series_key="KMRI_1_1", plane="Sagittal",
                fluid_sensitive=0, rule_hit="layout", series_description="", thickness="",
                TE="", TR="", field_strength="", vendor="", source_archive_member=str(path),
                spacing_unknown=True)
    series = common.Series(meta, arr, np.full((2, 3), np.nan), np.full(6, np.nan),
                           np.full(2, np.nan))
    common.add_series("kneemri", series, tmp_path)
    common.build_production("kneemri", tmp_path, shard_size=1)
    assert common.verify("kneemri", tmp_path)["errors"] == []


def test_known_contrast_uses_strict_groups():
    # B fix: MRNet coronal T1 must land in the non-FS coronal group, not the FS group via the competition fallback.
    rows = [dict(plane="Coronal", fluid_sensitive=0, series_key="cor", rule_hit="known contrast: MRNet coronal T1"),
            dict(plane="Sagittal", fluid_sensitive=1, series_key="sag", rule_hit="known contrast: MRNet sagittal T2 fat-suppressed")]
    pick_rows = [dict(r, Anatomical_Plane=r["plane"], Fluid_Sensitive=r["fluid_sensitive"], SeriesInstanceUID=r["series_key"]) for r in rows]
    assert common.pick_strict(pick_rows, "Coronal", 1, set()) is None
    assert common.pick_strict(pick_rows, "Coronal", 0, set())["series_key"] == "cor"
    assert common.pick_strict(pick_rows, "Sagittal", 0, set()) is None
    assert common.pick_strict(pick_rows, "Sagittal", 1, set())["series_key"] == "sag"
