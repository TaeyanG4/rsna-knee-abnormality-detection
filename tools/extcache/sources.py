"""Read local external archives as data; no downloaded code is executed."""
from __future__ import annotations

import csv
import io
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .common import ROOT, Series

MAX_MEMBER_BYTES = 2 * 2**30


def _safe_member(member: tarfile.TarInfo):
    p = Path(member.name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or not member.isfile() or member.size > MAX_MEMBER_BYTES:
        raise ValueError(f"unsafe archive member: {member.name}")


def _read_member(tar, member):
    _safe_member(member)
    with tar.extractfile(member) as f:
        data = f.read(MAX_MEMBER_BYTES + 1)
    if len(data) != member.size:
        raise ValueError(f"truncated archive member: {member.name}")
    return data


def plane_from_iop(iop):
    x = np.asarray(iop, float)
    if x.shape != (6,) or not np.isfinite(x).all():
        return "Unknown"
    n = np.cross(x[:3], x[3:])
    if np.linalg.norm(n) < 0.5:
        return "Unknown"
    return ("Sagittal", "Coronal", "Axial")[int(np.argmax(np.abs(n)))]


def fluid_rule(description, scan_options="", sequence_name="", *, source="dicom"):
    d = str(description).upper()
    if source == "oai":
        for prefix, fluid in (("SAG_IW_TSE", 1), ("SAG_3D_DESS", 0), ("COR_MPR", 1),
                              ("COR_IW_TSE", 0), ("AX_MPR", -1)):
            if d.startswith(prefix):
                return fluid, prefix
    if source == "fastmri_sc":
        return (1, "CORPDFS") if "CORPDFS" in d else (0, "CORPD")
    all_text = " ".join((d, str(scan_options).upper(), str(sequence_name).upper()))
    for token in ("FATSAT", "FAT SAT", "SPAIR", "SPIR", "STIR", "T2", "WATER EXCITATION", " WE ", "FS", "F/S"):
        if token in all_text:
            return 1, token.strip()
    return 0, "default fluid0"


def _float_attr(ds, name):
    try:
        return float(getattr(ds, name))
    except (AttributeError, TypeError, ValueError):
        return ""


def dicom_series(members: list[tuple[str, bytes]], *, source: str, study_key: str,
                 series_key: str, subject_id: str = "", knee_side: str = "",
                 visit: str = "", description: str = "", archive: str = "") -> Series:
    import pydicom
    from pydicom.pixels import apply_modality_lut

    recs = []
    for name, data in members:
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"multi-frame or non-2D DICOM needs a dedicated decoder: {name}")
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            arr = arr.max() - arr
        iop = np.asarray(getattr(ds, "ImageOrientationPatient", [np.nan] * 6), float)
        ipp = np.asarray(getattr(ds, "ImagePositionPatient", [np.nan] * 3), float)
        ps = np.asarray(getattr(ds, "PixelSpacing", [.5, .5]), float)
        pos = float(np.dot(ipp, np.cross(iop[:3], iop[3:]))) if np.isfinite(iop).all() and np.isfinite(ipp).all() else float(getattr(ds, "InstanceNumber", 0) or 0)
        recs.append((pos, name, arr, iop, ipp, ps, ds))
    if not recs:
        raise ValueError("empty DICOM series")
    recs.sort(key=lambda x: (x[0], x[1]))
    ds = recs[0][6]
    iop = recs[0][3]
    desc = description or str(getattr(ds, "SeriesDescription", ""))
    plane = plane_from_iop(iop)
    if plane == "Unknown":
        plane = next((p for marker, p in (("SAG", "Sagittal"), ("COR", "Coronal"), ("AX", "Axial"))
                      if marker in desc.upper()), "Unknown")
    fluid, rule = fluid_rule(desc, getattr(ds, "ScanOptions", ""), getattr(ds, "SequenceName", ""), source=source)
    if not knee_side:
        side = str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", ""))).upper()
        knee_side = "R" if side.startswith("R") else "L" if side.startswith("L") else ""
    positions = np.asarray([r[0] for r in recs], float)
    delta = np.diff(positions)
    meta = dict(source=source, subject_id=subject_id, knee_side=knee_side, visit=visit,
                study_key=study_key, series_key=series_key, plane=plane, fluid_sensitive=fluid,
                rule_hit=rule, series_description=desc,
                thickness=_float_attr(ds, "SliceThickness"), TE=_float_attr(ds, "EchoTime"),
                TR=_float_attr(ds, "RepetitionTime"), field_strength=_float_attr(ds, "MagneticFieldStrength"),
                vendor=str(getattr(ds, "Manufacturer", "")),
                source_archive_member=f"{archive}::{recs[0][1]}", spacing_unknown=False)
    return Series(meta, np.stack([r[2] for r in recs]), np.stack([r[4] for r in recs]), iop,
                  np.median(np.stack([r[5] for r in recs]), axis=0),
                  float(np.median(np.abs(delta))) if len(delta) else float("nan"),
                  np.asarray([r[5][0] for r in recs], np.float32))


def oai_tasks(root: Path = ROOT, visit: str = "V00", limit: int = 0):
    base = Path(root) / "storage/external_data/oai"
    selection = base / "selection" / f"{visit}_moaks_4series.csv"
    with selection.open(newline="", encoding="utf-8-sig") as f:
        chosen = list(csv.DictReader(f))
    meta_path = base / "image03_meta/image03.txt"
    metadata = {}
    with meta_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        next(reader, None)  # NDA dictionary description row
        for row in reader:
            if row.get("image_file"):
                metadata[row["image_file"]] = row
    tasks = []
    for row in chosen:
        uri = row["image_file"]
        tail = uri.split("/00m/", 1)
        if len(tail) != 2:
            continue
        path = base / "raw" / visit / "image03" / "00m" / tail[1]
        if not path.is_file():
            continue  # download is still in progress
        side = row["side"].upper()[0]
        sid = row["ID"]
        study = f"OAI_{sid}_{side}_{visit}"
        desc = metadata.get(uri, {}).get("image_description", row["series"])
        tasks.append((str(path), study, f"{study}_{path.stem.replace('.tar', '')}", sid, side, visit, desc))
        if limit and len(tasks) >= limit:
            break
    return tasks


def read_oai_task(task):
    path, study, key, sid, side, visit, desc = task
    members = []
    with tarfile.open(path, "r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            data = _read_member(tar, m)
            members.append((m.name, data))
    return dicom_series(members, source="oai", study_key=study, series_key=key,
                        subject_id=sid, knee_side=side, visit=visit,
                        description=desc, archive=path)


def _xml_float(root, path):
    el = root.find(path)
    return float(el.text) if el is not None and el.text else float("nan")


def fastmri_sc_series(name: str, data: bytes, archive: str) -> Series:
    import h5py
    with h5py.File(io.BytesIO(data), "r") as f:
        vol = np.asarray(f["reconstruction_esc"])
        acq = str(f.attrs.get("acquisition", ""))
        header = f["ismrmrd_header"][()]
    if isinstance(header, bytes):
        header = header.decode("utf-8")
    xml = ET.fromstring(header)
    ns = {"i": xml.tag.split("}")[0].strip("{")} if "}" in xml.tag else {}
    prefix = "i:" if ns else ""
    fov = _xml_float(xml, f"{prefix}encoding/{prefix}reconSpace/{prefix}fieldOfView_mm/{prefix}x") if not ns else _xml_float_ns(xml, "x", ns)
    fovy = _xml_float(xml, f"{prefix}encoding/{prefix}reconSpace/{prefix}fieldOfView_mm/{prefix}y") if not ns else _xml_float_ns(xml, "y", ns)
    mx = _xml_float(xml, f"{prefix}encoding/{prefix}reconSpace/{prefix}matrixSize/{prefix}x") if not ns else _xml_matrix_ns(xml, "x", ns)
    my = _xml_float(xml, f"{prefix}encoding/{prefix}reconSpace/{prefix}matrixSize/{prefix}y") if not ns else _xml_matrix_ns(xml, "y", ns)
    ps = np.asarray([fovy / my, fov / mx], np.float32)
    if not np.isfinite(ps).all() or (ps <= 0).any():
        raise ValueError("fastMRI reconSpace has invalid FOV/matrix")
    if vol.ndim != 3:
        raise ValueError("fastMRI reconstruction_esc must be slices-first 3D")
    stem = Path(name).stem
    fluid, rule = fluid_rule(acq, source="fastmri_sc")
    meta = dict(source="fastmri_sc", subject_id=stem, knee_side="", visit="",
                study_key=f"FMRI_SC_{stem}", series_key=f"FMRI_SC_{stem}", plane="Coronal",
                fluid_sensitive=fluid, rule_hit="known contrast: " + rule, series_description=acq,
                thickness="", TE="", TR="", field_strength="", vendor="",
                source_archive_member=f"{archive}::{name}", spacing_unknown=False)
    return Series(meta, vol, np.full((len(vol), 3), np.nan, np.float32),
                  np.array([1, 0, 0, 0, 0, 1], np.float32), ps)


def _xml_float_ns(xml, axis, ns):
    return _xml_float(xml, f"i:encoding/i:reconSpace/i:fieldOfView_mm/i:{axis}".replace("i:", "{" + ns["i"] + "}"))


def _xml_matrix_ns(xml, axis, ns):
    return _xml_float(xml, f"i:encoding/i:reconSpace/i:matrixSize/i:{axis}".replace("i:", "{" + ns["i"] + "}"))


def iter_fastmri_sc(root: Path = ROOT):
    base = Path(root) / "storage/external_data/fastmri/raw"
    for split in ("val", "train"):
        archive = base / f"knee_singlecoil_{split}.tar.xz"
        if not archive.is_file():
            continue
        with tarfile.open(archive, "r|xz") as tar:
            for member in tar:
                if member.isfile() and member.name.lower().endswith(".h5"):
                    yield member.name, str(archive), _read_member(tar, member)


def iter_fastmri_dcm(root: Path = ROOT):
    """Yield consecutive series groups. Repeated noncontiguous series is rejected."""
    base = Path(root) / "storage/external_data/fastmri/raw"
    for archive in sorted(base.glob("knee_DICOMs_batch*.tar.xz")):
        seen = set()
        group_key, group = None, []
        with tarfile.open(archive, "r|xz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                _safe_member(member)
                if not (Path(member.name).suffix.lower() in (".dcm", ".ima")
                        or "." not in Path(member.name).name):
                    continue
                parts = Path(member.name.replace("\\", "/")).parts
                if len(parts) < 3:
                    continue
                key = "/".join(parts[:-1])
                if key != group_key:
                    if group:
                        yield group_key, str(archive), group
                    if key in seen:
                        raise ValueError(f"noncontiguous DICOM series folder: {key}")
                    seen.add(key)
                    group_key, group = key, []
                group.append((member.name, _read_member(tar, member)))
            if group:
                yield group_key, str(archive), group


def fastmri_dcm_series(key, archive, group):
    parts = Path(key).parts
    study_folder = parts[-2] if len(parts) >= 2 else parts[-1]
    study = f"FMRI_DCM_{study_folder}"
    return dicom_series(group, source="fastmri_dcm", study_key=study,
                        series_key=f"{study}_{parts[-1]}", subject_id=study_folder, archive=archive)


def iter_kneemri(root: Path = ROOT):
    from tools.exp132_external_aux import load_kneemri_pickle
    base = Path(root) / "storage/external_data/kneemri"
    index = {p.name: p for p in base.glob("vol*/*.pck")}
    with (base / "metadata.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            path = index.get(Path(row["volumeFilename"]).name)
            if path is None:
                continue
            vol = load_kneemri_pickle(path)
            study = f"KMRI_{row['examId']}_{row['seriesNo']}"
            meta = dict(source="kneemri", subject_id=row["examId"], knee_side=row.get("kneeLR", ""),
                        visit="", study_key=study, series_key=study, plane="Sagittal", fluid_sensitive=1,
                        rule_hit="known contrast: KneeMRI sagittal PD fat-suppressed (B fix after EXP137 QA)", series_description="KneeMRI volume",
                        thickness="", TE="", TR="", field_strength="", vendor="",
                        source_archive_member=str(path), spacing_unknown=True)
            yield Series(meta, vol, np.full((len(vol), 3), np.nan, np.float32),
                         np.full(6, np.nan, np.float32), np.full(2, np.nan, np.float32))


# B fix after the EXP137 MRNet montage QA: the protocol is known (Bien et al. 2018) — sagittal T2 fat-suppressed,
# coronal T1, axial PD fat-suppressed; "known contrast" makes render_study use strict contrast-group picking.
MRNET_CONTRAST = {"sagittal": (1, "known contrast: MRNet sagittal T2 fat-suppressed"),
                  "coronal": (0, "known contrast: MRNet coronal T1"),
                  "axial": (1, "known contrast: MRNet axial PD fat-suppressed")}


def iter_mrnet(root: Path = ROOT):
    base = Path(root) / "storage/external_data/mrnet"
    for split in ("train", "valid"):
        for plane in ("sagittal", "coronal", "axial"):
            for path in sorted((base / split / plane).glob("*.npy")):
                vol = np.load(path, allow_pickle=False)
                if vol.ndim != 3 or vol.dtype != np.uint8:
                    raise ValueError(f"invalid MRNet volume: {path}")
                study = f"MRN_{split}_{path.stem}"
                meta = dict(source="mrnet", subject_id=path.stem, knee_side="", visit=split,
                            study_key=study, series_key=f"{study}_{plane}", plane=plane.capitalize(),
                            fluid_sensitive=MRNET_CONTRAST[plane][0], rule_hit=MRNET_CONTRAST[plane][1],
                            series_description=plane, thickness="", TE="", TR="", field_strength="",
                            vendor="", source_archive_member=str(path), spacing_unknown=True)
                yield Series(meta, vol, np.full((len(vol), 3), np.nan, np.float32),
                             np.full(6, np.nan, np.float32), np.full(2, np.nan, np.float32))
