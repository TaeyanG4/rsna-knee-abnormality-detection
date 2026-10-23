#!/usr/bin/env python3
"""HB126 decoder probe v1.0.0 (H-B-126 step 1, private Kaggle CPU diagnostic).

Establishes which compressed-DICOM decoders the pinned Kaggle image provides and whether
the serving readers of the C85-R7 line decode compressed DICOMs exactly like the
uncompressed originals:

  R / EXP155 / EXP143v2  public-serving.py (sha 25ef6ab9...): v2 ``read_px`` and the
                         embedded fastread ``study_lut`` path the release actually runs
  Q_A5q (A172 partner)   training/src/raptor_dicom.py ``_decode`` / ``prepare_study``
  S6 / R7                heliosli mainline ``training.dicom_geometry.pixels_from_dataset``
                         and ``training.native_cache.prepare`` (lean patch of probe_infer.py)

Phases (each one a fresh child interpreter; nothing is installed into the image):
  base        the image as is, no install                                    -> (a) + (b)
  pydicom301  the S6/R7 stage's pydicom 3.0.1 wheel via ``pip --target``      -> (a) + (b)
  codecs      every attached cp-compatible codec wheel via ``pip --target`` -> (c)

Diagnostic only: CPU, internet off, no submission file, no account operation. It never
opens train.csv, labels or report text; the competition mount is used only to confirm that
the fixture originals are byte-identical to the mounted files and to count the transfer
syntaxes of the visible test DICOM headers.

Changelog v1.0.0 (2026-10-13, Claude opus builder stand-in for B): first version.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import types
import zipfile
from pathlib import Path

VERSION = "1.0.0"
BANNER = f"HB126 decoder probe v{VERSION}"
# Filled in the kernel copy by tools/hb126_build_fixtures.py; None = do not enforce.
FIXTURES_ZIP_SHA256 = None
FIXTURES_ZIP = "hb126-fixtures-v1.zip.bin"
FIXTURES_MANIFEST = "hb126-fixtures-manifest.json"
OUT_NAME = f"hb126-decoder-probe-v{VERSION}"

TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2": "Implicit VR Little Endian",
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
    "1.2.840.10008.1.2.1.99": "Deflated Explicit VR Little Endian",
    "1.2.840.10008.1.2.4.50": "JPEG Baseline (Process 1)",
    "1.2.840.10008.1.2.4.51": "JPEG Extended (Process 2 & 4)",
    "1.2.840.10008.1.2.4.57": "JPEG Lossless (Process 14)",
    "1.2.840.10008.1.2.4.70": "JPEG Lossless SV1 (Process 14, Selection Value 1)",
    "1.2.840.10008.1.2.4.80": "JPEG-LS Lossless",
    "1.2.840.10008.1.2.4.81": "JPEG-LS Near-Lossless",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 (Lossless Only)",
    "1.2.840.10008.1.2.4.91": "JPEG 2000",
    "1.2.840.10008.1.2.5": "RLE Lossless",
}
REQUESTED = ["1.2.840.10008.1.2.4.50", "1.2.840.10008.1.2.4.51", "1.2.840.10008.1.2.4.57",
             "1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.80", "1.2.840.10008.1.2.4.81",
             "1.2.840.10008.1.2.4.90", "1.2.840.10008.1.2.4.91", "1.2.840.10008.1.2.5"]
# Named on the competition Data page ("JPEG Lossless, JPEG 2000").
CRITICAL = ["1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70",
            "1.2.840.10008.1.2.4.90", "1.2.840.10008.1.2.4.91"]

# Reader sources as served by the release line (pins from tools/exp169_release_p956.py PINS,
# the A172 partner receipt and tools/exp189_s6_port.py SOURCE_HASHES / CODE_HASHES).
READER_PINS = {
    "public-serving.py": "25ef6ab9d2e942ac291433a498246d0507481c2aefa5630cdd601d0a36852cf2",
    "raptor_dicom.py": "497a41f85a7b70cba8029c0f726ed2133653a2ce9673cd0bf8de42a85ac5634b",
    "raptor_common.py": "bc6db5b3e8cd13b52fbff8c039d3674cdf573f86c1959e22fe9b2fd94255d7dc",
    "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
}
CODE_PINS = {
    "training/dicom_geometry.py": "12aa501f9704aef15b7a3bbd560951acb6d95b89b7de0141d6cf89fc3f051ab1",
    "training/native_cache.py": "c53ce8985406b3b918d9e821d43518479e89d9f946daebd817f378e3949718b9",
}
PYDICOM301 = ("pydicom-3.0.1-py3-none-any.whl",
              "db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41")
CODEC_WHEEL_RE = re.compile(r"(gdcm|pylibjpeg|openjpeg|libjpeg|jpegls|jpeg_ls|charls|imagecodecs|rle)", re.I)
PACKAGE_RE = re.compile(r"(gdcm|pylibjpeg|openjpeg|libjpeg|jpeg|charls|pillow|imagecodecs|simpleitk|"
                        r"opencv|pydicom|dicom|numpy|rle)", re.I)
SKIP_WALK = {"train_series", "test_series", "train_images", "test_images", "__pycache__"}

# Wheels a release would need when no cp313 codec wheel is attached. Read from the PyPI JSON
# API on 2026-10-13 (metadata only, nothing downloaded); python-gdcm's cp312 entry matched our
# private copy byte-for-byte (size 13278949, sha256 512d0d90...). Re-verify sha256 at download.
REQUIRED_WHEELS = {
    "option_gdcm": [
        {"package": "python-gdcm", "version": "3.2.6", "licence": "Apache-2.0 (PyPI license_expression)",
         "file": "python_gdcm-3.2.6-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
         "url": "https://files.pythonhosted.org/packages/30/ba/52fa278ff973ae2932ffdde98e9bdeaa248abda39c1c74e39caa969d543e/python_gdcm-3.2.6-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
         "size": 13277668, "sha256": "69c987d3f1b3a2a9936288f312f1ea2192e39fa6c0be28159b06a1906b8d56ed",
         "covers": ".50 .51 .57 .70 .80 .81 .90 .91 RLE (pydicom gdcm plugin, gdcm>=3.0.10)"}],
    "option_pylibjpeg": [
        {"package": "pylibjpeg", "version": "2.1.0", "licence": "MIT",
         "file": "pylibjpeg-2.1.0-py3-none-any.whl",
         "url": "https://files.pythonhosted.org/packages/3c/31/86c111e81c113e74ac1b0afde27755088b2db394579370c893d252d9615a/pylibjpeg-2.1.0-py3-none-any.whl",
         "size": 25472, "sha256": "25df9496a69e64e98c887fddee12a1271e275b5f74ba804f9bf98a08bb80993e",
         "covers": "plugin host (pydicom needs pylibjpeg>=2.0)"},
        {"package": "pylibjpeg-libjpeg", "version": "2.4.0", "licence": "GPL-3.0",
         "file": "pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
         "url": "https://files.pythonhosted.org/packages/77/00/0a07d56c596793135bc61bcace6267ad68e5a36b9dc53e9fd7f48dac865d/pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
         "size": 3403812, "sha256": "d2c8da12c79d1f46596b89df94091be01f932d739613e8af93d1d557ac3bab74",
         "covers": ".50 .51 .57 .70 .80 .81"},
        {"package": "pylibjpeg-openjpeg", "version": "2.6.0", "licence": "MIT",
         "file": "pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
         "url": "https://files.pythonhosted.org/packages/99/db/78efc8c06522595bfa4c17e630a5ca53673b90cca97ad54f37584eea1d35/pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
         "size": 1995198, "sha256": "3e8ee9e5bd3b62766ca98fc143cd340633e60bb14450162426f32688908ad9e8",
         "covers": ".90 .91"}],
}

# Bundled pydicom test files: (compressed, twin or None, transfer syntax). No network: the
# path is built from the installed package, never via get_testdata_file.
BUNDLED = [
    ("MR_small_implicit.dcm", "MR_small.dcm", "1.2.840.10008.1.2"),
    ("MR_small_RLE.dcm", "MR_small.dcm", "1.2.840.10008.1.2.5"),
    ("MR_small_jpeg_ls_lossless.dcm", "MR_small.dcm", "1.2.840.10008.1.2.4.80"),
    ("MR_small_jp2klossless.dcm", "MR_small.dcm", "1.2.840.10008.1.2.4.90"),
    ("JPEG-lossy.dcm", None, "1.2.840.10008.1.2.4.51"),
    ("JPEG2000.dcm", None, "1.2.840.10008.1.2.4.91"),
    ("JPEGLSNearLossless_16.dcm", None, "1.2.840.10008.1.2.4.81"),
    ("SC_rgb_jpeg_gdcm.dcm", None, "1.2.840.10008.1.2.4.70"),
    ("SC_rgb_jpeg_dcmtk.dcm", None, "1.2.840.10008.1.2.4.50"),
]
SLICE_READERS = ["pydicom_pixel_array", "R_read_px_v2", "R_fastread_study_lut", "OWN_raptor_decode",
                 "S6R7_pixels_from_dataset", "S6R7_canonical"]
SERIES_READERS = ["R_study_lut", "R_build_study_v2", "OWN_prepare_study", "S6R7_native_prepare"]
OWN_CONTRACT = "raptor64_336_c140_s02"


# ---------------------------------------------------------------------------------------
# small pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_member(name: str) -> bool:
    """Zip member names allowed for extraction: relative POSIX paths without '..'."""
    if not name or name.startswith(("/", "\\")) or "\\" in name or ":" in name or "\x00" in name:
        return False
    parts = name.split("/")
    return all(p not in ("", ".", "..") for p in (parts[:-1] if name.endswith("/") else parts))


def safe_extract(archive: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    names = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if not safe_member(info.filename):
                raise ValueError(f"unsafe archive member: {info.filename!r}")
            target = (dest / info.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"archive member escapes destination: {info.filename!r}")
        for info in z.infolist():
            z.extract(info, dest)
            if not info.is_dir():
                names.append(info.filename)
    return names


def parse_wheel_name(filename: str) -> dict:
    """PEP 427 wheel name -> distribution, version and expanded (py, abi, plat) tag triples."""
    if not filename.endswith(".whl"):
        raise ValueError(f"not a wheel: {filename}")
    parts = filename[:-4].split("-")
    if len(parts) not in (5, 6):
        raise ValueError(f"malformed wheel name: {filename}")
    dist, version = parts[0], parts[1]
    py, abi, plat = parts[-3], parts[-2], parts[-1]
    tags = sorted({f"{a}-{b}-{c}" for a in py.split(".") for b in abi.split(".") for c in plat.split(".")})
    return {"file": filename, "distribution": dist, "version": version, "python": py, "abi": abi,
            "platform": plat, "tags": tags}


def supported_tags() -> list[str]:
    try:
        from packaging.tags import sys_tags
    except Exception:  # pip vendors packaging in every image that has pip
        from pip._vendor.packaging.tags import sys_tags  # type: ignore
    return [str(t) for t in sys_tags()]


def wheel_compatible(info: dict, supported) -> bool:
    supported = set(supported)
    return any(tag in supported for tag in info["tags"])


def compare_arrays(a, b) -> dict:
    """Exact comparison of two decoded arrays (shape, dtype and values)."""
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    out = {"shape_equal": a.shape == b.shape, "dtype_equal": a.dtype == b.dtype}
    out["equal"] = bool(out["shape_equal"] and out["dtype_equal"] and np.array_equal(a, b))
    if out["shape_equal"] and a.size:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        out["max_abs_diff"] = float(np.nanmax(diff)) if np.isfinite(diff).any() else None
        out["n_diff"] = int((diff > 0).sum())
    return out


def array_record(arr) -> dict:
    import numpy as np
    a = np.ascontiguousarray(arr)
    rec = {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_bytes(a.tobytes())}
    if a.size and a.dtype.kind in "fiu":
        rec.update(min=float(np.nanmin(a)), max=float(np.nanmax(a)), finite=bool(np.isfinite(a).all()))
    return rec


def classify(result: dict, twin: dict | None) -> str:
    """EQUAL / DIFF / BLANK / RAISE / TWIN_FAIL / DECODED (no twin) / SKIP."""
    status = result.get("status")
    if status in ("RAISE", "BLANK", "SKIP"):
        return status
    if twin is None:
        return "DECODED"
    if twin.get("status") != "OK":
        return "TWIN_FAIL"
    return "EQUAL" if result.get("compare", {}).get("equal") else "DIFF"


def walk_files(root: Path, name_re, max_depth: int = 6):
    """os.walk below root that never enters DICOM series trees (800k files on Kaggle)."""
    root = Path(root)
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if d not in SKIP_WALK)
        for f in sorted(filenames):
            if name_re.search(f):
                yield Path(dirpath) / f


def first_dir_with(root: Path, required, max_depth: int = 5):
    """Directories below root containing every required relative path."""
    hits = []
    root = Path(root)
    if not root.is_dir():
        return hits
    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if d not in SKIP_WALK)
        if all((Path(dirpath) / r).exists() for r in required):
            hits.append(Path(dirpath))
    return hits


# ---------------------------------------------------------------------------------------
# orchestrator (parent): mounts, fixture/code preparation, wheel inventory, phases
# ---------------------------------------------------------------------------------------
def find_competition(input_root: Path, override=None):
    if override:
        return Path(override)
    for c in (input_root / "competitions/rsna-knee-abnormality-detection",
              input_root / "rsna-knee-abnormality-detection"):
        if (c / "train_series.csv").is_file():
            return c
    hits = first_dir_with(input_root, ["train_series.csv", "test_series.csv"], max_depth=4)
    return hits[0] if hits else None


def find_fixtures(input_root: Path, override=None):
    if override:
        return Path(override)
    hits = first_dir_with(input_root, [FIXTURES_ZIP, FIXTURES_MANIFEST])
    return hits[0] if len(hits) == 1 else None


def find_code_source(input_root: Path, override=None):
    if override:
        return Path(override)
    hits = [h for h in first_dir_with(input_root, ["anatomy.pt"])
            if (h / "cloud_code.zip").is_file() or (h / "cloud_code/training/dicom_geometry.py").is_file()]
    return hits[0] if len(hits) == 1 else None


def prepare_fixtures(src, work: Path) -> dict:
    rec = {"source": str(src) if src else None, "status": "ABSENT"}
    if src is None:
        return rec
    archive, top_manifest = Path(src) / FIXTURES_ZIP, Path(src) / FIXTURES_MANIFEST
    rec["zip_sha256"] = sha256_file(archive)
    rec["pinned_zip_sha256"] = FIXTURES_ZIP_SHA256
    if FIXTURES_ZIP_SHA256 and rec["zip_sha256"] != FIXTURES_ZIP_SHA256:
        rec["status"] = "ZIP_SHA_MISMATCH"
        return rec
    dest = work / "fixtures"
    if dest.exists():
        shutil.rmtree(dest)
    names = safe_extract(archive, dest)
    manifest = json.loads((dest / FIXTURES_MANIFEST).read_text(encoding="utf-8"))
    rec["top_manifest_matches_zip"] = sha256_file(top_manifest) == sha256_file(dest / FIXTURES_MANIFEST)
    bad = [rel for rel, digest in manifest["files"].items() if sha256_file(dest / rel) != digest]
    unlisted = sorted(set(names) - set(manifest["files"]) - {FIXTURES_MANIFEST})
    rec.update(root=str(dest), members=len(names), bad_members=bad, unlisted_members=unlisted,
               manifest_version=manifest.get("version"))
    rec["status"] = "OK" if not bad and not unlisted and rec["top_manifest_matches_zip"] else "INTEGRITY_FAIL"
    return rec


def prepare_code(src, work: Path) -> dict:
    rec = {"source": str(src) if src else None, "status": "ABSENT"}
    if src is None:
        return rec
    src, code = Path(src), work / "code"
    if code.exists():
        shutil.rmtree(code)
    if (src / "cloud_code.zip").is_file():
        safe_extract(src / "cloud_code.zip", code)
        rec["layout"] = "cloud_code.zip"
    else:  # imported in place from the read-only mount (children never write bytecode)
        code = src / "cloud_code"
        rec["layout"] = "cloud_code/"
    rec["pins"] = {rel: (sha256_file(code / rel) == digest if (code / rel).is_file() else None)
                   for rel, digest in CODE_PINS.items()}
    rec["root"] = str(code)
    rec["status"] = "OK" if all(rec["pins"].values()) else "PIN_MISMATCH"
    return rec


def find_pydicom301(code_src, work: Path):
    """The S6/R7 stage's own pydicom wheel (loose or inside dependencies.zip)."""
    if code_src is None:
        return None
    src = Path(code_src)
    if (src / "dependencies.zip").is_file():
        with zipfile.ZipFile(src / "dependencies.zip") as z:
            for name in z.namelist():
                if Path(name).name == PYDICOM301[0]:
                    target = work / "wheels-pyd301" / PYDICOM301[0]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
                    return target
        return None
    hits = sorted(src.rglob(PYDICOM301[0]))
    return hits[0] if hits else None


def inventory_wheels(input_root: Path, extra_dirs=()) -> list[dict]:
    """Every codec-like wheel under the attached datasets, with tags and compatibility."""
    supported = supported_tags()
    seen, out = set(), []
    paths = list(walk_files(input_root, re.compile(r"\.whl$")))
    for d in extra_dirs:
        paths += list(walk_files(Path(d), re.compile(r"\.whl$")))
    for path in paths:
        if not CODEC_WHEEL_RE.search(path.name) or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        try:
            info = parse_wheel_name(path.name)
        except ValueError as exc:
            out.append({"file": path.name, "path": str(path), "error": str(exc)})
            continue
        info.update(path=str(path), size=path.stat().st_size, sha256=sha256_file(path),
                    compatible=wheel_compatible(info, supported))
        out.append(info)
    return out


def codec_family(filename: str) -> str:
    name = filename.lower()
    if "gdcm" in name:
        return "gdcm"
    if re.search(r"pylibjpeg|openjpeg|libjpeg|rle", name):
        return "pylibjpeg"
    return "other"


def codec_groups(compatible) -> dict:
    """Install plans: one phase per codec family, plus all together when several families exist."""
    fams = {}
    for w in compatible:
        fams.setdefault(codec_family(w["file"]), []).append(w)
    groups = {f"codecs_{f}": fams[f] for f in sorted(fams)}
    if len(fams) > 1:
        groups["codecs_all"] = list(compatible)
    return groups


def pip_target(wheels, target: Path, log: Path) -> dict:
    if target.exists():
        shutil.rmtree(target)
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
           "--disable-pip-version-check", "--target", str(target), *map(str, wheels)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write_text(proc.stdout, encoding="utf-8")
    return {"command": cmd, "returncode": proc.returncode, "target": str(target), "log": str(log)}


def run_child(phase: str, out: Path, args, *, pythonpath=None, simulate_missing=None) -> dict:
    child_json = out / f"child-{phase}.json"
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", phase, "--out", str(child_json),
           "--work", str(Path(args.work) / f"child-{phase}")]
    for flag, value in (("--fixtures-root", args.fixtures_root), ("--code-root", args.code_root),
                        ("--competition-root", args.competition_root)):
        if value:
            cmd += [flag, str(value)]
    if simulate_missing:
        cmd += ["--simulate-missing", simulate_missing]
    # Same site configuration as the release processes (no PYTHONNOUSERSITE); CPU only.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    (out / f"child-{phase}.log").write_text(proc.stdout, encoding="utf-8")
    for line in proc.stdout.splitlines():
        if line.startswith(("HB126_", "Traceback", "  File")) or "Error" in line[:40]:
            print(f"  [{phase}] {line}", flush=True)
    rec = {"phase": phase, "returncode": proc.returncode, "seconds": round(time.time() - started, 1),
           "pythonpath": str(pythonpath) if pythonpath else None, "json": str(child_json)}
    if child_json.is_file():
        rec["result"] = json.loads(child_json.read_text(encoding="utf-8"))
    return rec


def orchestrate(args) -> dict:
    input_root = Path(args.input_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    report = {"version": VERSION, "started_utc": now_utc(), "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "libc": list(platform.libc_ver()), "input_root": str(input_root)}
    comp = find_competition(input_root, args.competition_root)
    fx_src = find_fixtures(input_root, args.fixtures_dataset)
    code_src = find_code_source(input_root, args.code_source)
    report["mounts"] = {"competition": str(comp) if comp else None, "fixtures_dataset": str(fx_src) if fx_src else None,
                        "code_dataset": str(code_src) if code_src else None}
    print("HB126_MOUNTS " + json.dumps(report["mounts"]), flush=True)
    report["fixtures"] = fx = prepare_fixtures(fx_src, work)
    report["code"] = code = prepare_code(code_src, work)
    print(f"HB126_FIXTURES status={fx['status']} zip_sha256={fx.get('zip_sha256')} members={fx.get('members')}", flush=True)
    print(f"HB126_CODE status={code['status']} pins={code.get('pins')}", flush=True)
    args.fixtures_root = fx.get("root") if fx["status"] == "OK" else None
    args.code_root = code.get("root") if code["status"] in ("OK", "PIN_MISMATCH") else None
    args.competition_root = str(comp) if comp else None

    phases = report["phases"] = {}
    phases["base"] = run_child("base", out, args, simulate_missing=args.simulate_missing)

    wheel = find_pydicom301(code_src, work)
    if wheel is not None and "pydicom301" not in args.skip_phases:
        ok = sha256_file(wheel) == PYDICOM301[1]
        install = pip_target([wheel], work / "site-pydicom301", out / "pip-pydicom301.log") if ok else None
        phases["pydicom301"] = {"wheel": str(wheel), "sha256_ok": ok, "install": install}
        if ok and install["returncode"] == 0:
            phases["pydicom301"].update(run_child("pydicom301", out, args, pythonpath=work / "site-pydicom301",
                                                  simulate_missing=args.simulate_missing))
    else:
        phases["pydicom301"] = {"skipped": "no pydicom-3.0.1 wheel in the code dataset" if wheel is None else "skipped"}

    extra = [args.wheel_dir] if args.wheel_dir else []
    wheels = report["wheels"] = inventory_wheels(input_root, extra)
    compatible = [w for w in wheels if w.get("compatible")]
    for w in wheels:
        print(f"HB126_WHEEL file={w['file']} compatible={w.get('compatible')} size={w.get('size')} "
              f"sha256={w.get('sha256')}", flush=True)
    print(f"HB126_WHEELS found={len(wheels)} compatible={len(compatible)}", flush=True)
    groups = codec_groups(compatible)
    if groups and "codecs" not in args.skip_phases:
        for phase, items in groups.items():  # one family at a time, then all together
            target = work / f"site-{phase}"
            install = pip_target([w["path"] for w in items], target, out / f"pip-{phase}.log")
            phases[phase] = {"wheels": [w["file"] for w in items], "install": install}
            if install["returncode"] == 0:
                phases[phase].update(run_child(phase, out, args, pythonpath=target,
                                               simulate_missing=args.simulate_missing))
    else:
        phases["codecs"] = {"skipped": "NO_COMPATIBLE_CODEC_WHEELS" if not compatible else "skipped",
                            "required_wheels_if_approved": REQUIRED_WHEELS}
        if not compatible:
            for option, items in REQUIRED_WHEELS.items():
                for w in items:
                    print(f"HB126_NEEDED {option} {w['file']} size={w['size']} licence={w['licence']} "
                          f"sha256={w['sha256']}", flush=True)
    report["summary"] = summarize(report)
    report["finished_utc"] = now_utc()
    (out / "probe.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    lines = summary_lines(report)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report


def summarize(report: dict) -> dict:
    out = {}
    for phase, rec in report["phases"].items():
        res = rec.get("result") if isinstance(rec, dict) else None
        if not res:
            out[phase] = {"ran": False, "returncode": rec.get("returncode") if isinstance(rec, dict) else None}
            continue
        dec = res.get("decoders", {})
        missing = [u for u in REQUESTED if not dec.get(u, {}).get("available")]
        readers = {}
        for case in res.get("slices", []):
            if case.get("origin") != "dataset" or not case.get("lossless"):
                continue
            for name, r in case["readers"].items():
                readers.setdefault(name, {}).setdefault(r["class"], 0)
                readers[name][r["class"]] += 1
        series = {lab: {k: v.get("class") for k, v in rs.items()} for lab, rs in res.get("series", {}).items()}
        out[phase] = {"ran": True, "pydicom": res.get("environment", {}).get("pydicom"),
                      "missing_requested": missing, "missing_critical": [u for u in CRITICAL if u in missing],
                      "lossless_dataset_reader_classes": readers, "series": series}
    codec_phases = {k: v for k, v in out.items() if k.startswith("codecs_")}
    out["verdict"] = verdict_from(out.get("base", {}), codec_phases)
    return out


def readers_all_equal(phase: dict) -> bool | None:
    """True/False over the lossless dataset fixtures; None when no dataset fixture ran."""
    classes = phase.get("lossless_dataset_reader_classes", {})
    if not classes or not sum(sum(v.values()) for v in classes.values()):
        return None
    series_ok = all(c == "EQUAL" for rs in phase.get("series", {}).values() for c in rs.values())
    return all(set(v) <= {"EQUAL"} for v in classes.values()) and series_ok


def verdict_from(base: dict, codec_phases: dict) -> str:
    """codec_phases: {phase name: summarized phase} for the codecs_* phases that were planned."""
    if not base.get("ran"):
        return "INCONCLUSIVE_BASE_PHASE_FAILED"
    equal = readers_all_equal(base)
    if not base["missing_critical"]:
        if equal is None:
            return "DECODERS_PRESENT_READERS_UNTESTED"
        return "NO_GO_DECODERS_PRESENT_ADD_FIXTURE_GATE" if equal else "DECODERS_PRESENT_READER_MISMATCH"
    ran = {k: v for k, v in codec_phases.items() if v.get("ran")}
    fixing = sorted(k for k, v in ran.items() if not v["missing_critical"] and readers_all_equal(v))
    if fixing:
        return "GO_CANDIDATE_ATTACHED_WHEELS_FIX_CRITICAL:" + ",".join(fixing)
    if ran:
        return "ATTACHED_WHEELS_INSUFFICIENT"
    return "GO_PENDING_CP313_WHEELS_APPROVAL"


def summary_lines(report: dict) -> list[str]:
    s = report["summary"]
    lines = [BANNER, f"HB126_VERDICT {s['verdict']}"]
    for phase, rec in s.items():
        if phase == "verdict" or not isinstance(rec, dict):
            continue
        if not rec.get("ran"):
            lines.append(f"HB126_PHASE {phase} ran=False")
            continue
        lines.append(f"HB126_PHASE {phase} pydicom={rec['pydicom']} missing_requested={rec['missing_requested']} "
                     f"missing_critical={rec['missing_critical']}")
        for name, counts in sorted(rec["lossless_dataset_reader_classes"].items()):
            lines.append(f"HB126_READERS {phase} {name} {json.dumps(counts, sort_keys=True)}")
        for label, rs in sorted(rec["series"].items()):
            lines.append(f"HB126_SERIES {phase} {label} {json.dumps(rs, sort_keys=True)}")
    return lines


# ---------------------------------------------------------------------------------------
# child: everything that imports pydicom/numpy happens here, in a fresh interpreter
# ---------------------------------------------------------------------------------------
class Blank(Exception):
    """R's fastread swallowed a decode exception and returned None (a blank slice)."""


class Skip(Exception):
    """The reader does not apply to this fixture (e.g. no anatomical plane for canonicalization)."""


def block_modules(names: str) -> list[str]:
    """LOCAL SIMULATION ONLY: make these imports fail, to preview the Kaggle base image."""
    blocked = [n.strip() for n in names.split(",") if n.strip()]
    for n in blocked:
        sys.modules[n] = None  # type: ignore[assignment]
    return blocked


def environment_report() -> dict:
    import importlib
    import importlib.metadata as md
    rep = {"python": sys.version, "executable": sys.executable, "sys_path_head": sys.path[:6],
           "platform": platform.platform(), "libc": list(platform.libc_ver())}
    pkgs = set()
    for d in md.distributions():
        name = d.metadata["Name"] or ""
        if PACKAGE_RE.search(name):
            pkgs.add((name, d.version, str(getattr(d, "_path", ""))))
    rep["packages"] = [dict(name=n, version=v, location=p) for n, v, p in sorted(pkgs)]
    mods = {}
    for mod in ["pydicom", "numpy", "gdcm", "pylibjpeg", "libjpeg", "openjpeg", "rle", "jpeg_ls", "imagecodecs",
                "SimpleITK", "PIL", "cv2"]:
        try:
            m = importlib.import_module(mod)
            mods[mod] = {"version": str(getattr(m, "__version__", getattr(m, "VERSION", "?"))),
                         "file": str(getattr(m, "__file__", ""))}
        except Exception as exc:
            mods[mod] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    rep["modules"] = mods
    rep["pydicom"] = mods.get("pydicom", {}).get("version")
    # Image fingerprint to compare with the C85-R7 runtime receipt (python 3.13.15, pydicom 3.0.2,
    # torch 2.11.0+cu128, opencv 4.14.0.94). Versions from metadata only (torch is not imported).
    fp = {}
    for dist in ("torch", "timm", "numpy", "pandas", "opencv-python", "opencv-python-headless", "pillow", "pydicom"):
        with contextlib.suppress(Exception):
            fp[dist] = md.version(dist)
    with contextlib.suppress(Exception):
        fp["os_release"] = next((ln.split("=", 1)[1].strip('"') for ln in Path("/etc/os-release").read_text().splitlines()
                                 if ln.startswith("PRETTY_NAME=")), None)
    safe = {"KAGGLE_DOCKER_IMAGE", "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_CONTAINER_NAME"}
    fp["kaggle_env"] = {k: (os.environ[k] if k in safe else "<set>") for k in sorted(os.environ) if k.startswith("KAGGLE_")}
    rep["image_fingerprint"] = fp
    try:
        rep["supported_tags_head"] = supported_tags()[:8]
    except Exception as exc:
        rep["supported_tags_head"] = [repr(exc)]
    return rep


def decoder_table() -> dict:
    table = {}
    try:
        from pydicom.pixels import get_decoder
    except ImportError:
        get_decoder = None
    for uid, name in TRANSFER_SYNTAXES.items():
        row = {"name": name}
        if get_decoder is not None:
            try:
                d = get_decoder(uid)
                row.update(available=bool(d.is_available), plugins=list(d.available_plugins),
                           missing=[str(m) for m in d.missing_dependencies])
            except Exception as exc:
                row.update(available=False, error=f"{type(exc).__name__}: {exc}"[:300])
        else:  # pydicom 2 fallback
            import pydicom
            handlers = [h for h in pydicom.config.pixel_data_handlers if h.supports_transfer_syntax(uid)]
            row.update(available=any(h.is_available() for h in handlers),
                       plugins=[h.HANDLER_NAME for h in handlers if h.is_available()],
                       missing=[h.HANDLER_NAME for h in handlers if not h.is_available()])
        table[uid] = row
    return table


def alternate_decoders() -> dict:
    """Decoders outside pydicom that a wheel-free fallback could use (report only)."""
    alt = {}
    try:
        import imagecodecs as ic
        alt["imagecodecs"] = {"version": ic.__version__, "codecs": {
            n: bool(getattr(getattr(ic, n.upper(), None), "available", False))
            for n in ("jpeg8", "ljpeg", "jpegsof3", "jpegls", "jpeg2k")}}
    except Exception as exc:
        alt["imagecodecs"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        import SimpleITK as sitk
        ios = []
        with contextlib.suppress(Exception):
            ios = list(sitk.ImageFileReader().GetRegisteredImageIOs())
        alt["SimpleITK"] = {"version": sitk.Version_VersionString(), "image_ios": ios}
    except Exception as exc:
        alt["SimpleITK"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        import cv2
        info = cv2.getBuildInformation()
        alt["cv2"] = {"version": cv2.__version__,
                      "build_lines": [ln.strip() for ln in info.splitlines() if "JPEG" in ln.upper()][:12]}
    except Exception as exc:
        alt["cv2"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        import PIL
        from PIL import features
        alt["PIL"] = {"version": PIL.__version__, "jpg": features.check("jpg"), "jpg_2000": features.check("jpg_2000"),
                      "jpg_version": features.version("jpg"), "jpg_2000_version": features.version("jpg_2000")}
    except Exception as exc:
        alt["PIL"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return alt


def load_public_serving(path: Path) -> dict:
    """Top-level reader constants/functions of the pinned public-serving.py, without torch/timm."""
    import glob as _glob
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    names = {"IMG", "CROP_MM", "SLOTS", "MAXS", "HDR_THREADS", "PX_THREADS", "FASTREAD_VARIANT",
             "FR_HDR", "FR_PX", "_FASTREAD_B64"}
    funcs = {"_make_reader", "_pick_series_for_slot", "build_study"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in funcs:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            targets = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if targets and targets <= names:
                keep.append(node)
    ns = {"__name__": "hb126_public_serving", "os": os, "glob": _glob, "np": np, "ThreadPoolExecutor": ThreadPoolExecutor}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    missing = sorted((names | funcs) - set(ns))
    if missing:
        raise RuntimeError(f"public-serving.py lacks {missing}")
    return ns


def load_fastread(ns: dict):
    fast = types.ModuleType("hb126_fastread")
    fast.__file__ = "<public-serving _FASTREAD_B64>"
    exec(compile(base64.b64decode(ns["_FASTREAD_B64"]), fast.__file__, "exec"), fast.__dict__)
    if not hasattr(os, "pread"):  # Windows transport for local tests only (Kaggle is POSIX)
        fast._O_FLAGS |= getattr(os, "O_BINARY", 0)

        def _pread(fd, size, offset):
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, size)
        fast._pread = _pread
    return fast


class Readers:
    def __init__(self, readers_dir, code_root):
        self.status, self.fns = {}, {}
        self.public = self.fast = self.raptor = self.geom = self.probe_infer = None
        import pydicom
        self.fns["pydicom_pixel_array"] = lambda p, plane: pydicom.dcmread(str(p)).pixel_array
        self.status["pydicom_pixel_array"] = "OK"
        rd = Path(readers_dir) if readers_dir else None
        pins = {}
        if rd is not None:
            pins = {n: (sha256_file(rd / n) == d if (rd / n).is_file() else None) for n, d in READER_PINS.items()}
        self.status["reader_pins"] = pins
        self._load("public", lambda: self._load_public(rd))
        self._load("raptor", lambda: self._load_raptor(rd))
        self._load("geom", lambda: self._load_geom(code_root))

    def _load(self, name, fn):
        try:
            fn()
            self.status[name] = "OK"
        except Exception as exc:
            self.status[name] = f"{type(exc).__name__}: {exc}"[:300]

    def _load_public(self, rd):
        if rd is None:
            raise FileNotFoundError("no fixtures dataset (vendored public-serving.py)")
        self.public = load_public_serving(rd / "public-serving.py")
        read_px = self.public["_make_reader"]()[1]
        self.fns["R_read_px_v2"] = lambda p, plane: read_px(str(p))
        self.fast = load_fastread(self.public)
        ctx = self.fast._Ctx(self.fast.Cfg(lut=True))
        pyd_px = self.fast._make_pyd()[1]

        def r_fast(p, plane):
            data = Path(p).read_bytes()
            arr, _ = ctx.px(data, self.fast._scan_full(data))
            if arr is None:
                try:  # the swallowed exception, for the record
                    pyd_px(data)
                    raise Blank("px returned None but the pydicom fallback succeeded")
                except Blank:
                    raise
                except Exception as exc:
                    raise Blank(f"{type(exc).__name__}: {exc}") from exc
            return arr.to_float() if hasattr(arr, "to_float") else arr
        self.fns["R_fastread_study_lut"] = r_fast

    def _load_raptor(self, rd):
        if rd is None:
            raise FileNotFoundError("no fixtures dataset (vendored raptor_dicom.py)")
        if str(rd) not in sys.path:
            sys.path.insert(0, str(rd))
        import raptor_dicom
        self.raptor = raptor_dicom

        def own(p, plane):
            p = Path(p)
            source = raptor_dicom.DirectoryDicomSource(p.parent)
            return raptor_dicom._decode(source, raptor_dicom.DicomMember(p.name))[0]
        self.fns["OWN_raptor_decode"] = own

    def _load_geom(self, code_root):
        if not code_root:
            raise FileNotFoundError("no S6/R7 code dataset")
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        import pydicom
        import training.dicom_geometry as geom
        self.geom = geom

        def s6(p, plane):
            return geom.pixels_from_dataset(pydicom.dcmread(io.BytesIO(Path(p).read_bytes())), path=Path(p).name).image

        def s6c(p, plane):
            if not plane:
                raise Skip("plane unknown for this fixture; canonicalization skipped")
            payload = geom.pixels_from_dataset(pydicom.dcmread(io.BytesIO(Path(p).read_bytes())), path=Path(p).name)
            return geom.canonicalize_payload(payload, plane).image
        self.fns["S6R7_pixels_from_dataset"] = s6
        self.fns["S6R7_canonical"] = s6c


def run_reader(fn, path, plane) -> tuple[dict, object]:
    import warnings
    rec = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            arr = fn(path, plane)
            rec.update(status="OK", **array_record(arr))
        except Blank as exc:
            arr = None
            rec.update(status="BLANK", error=str(exc)[:400])
        except Skip as exc:
            arr = None
            rec.update(status="SKIP", error=str(exc)[:200])
        except Exception as exc:
            arr = None
            rec.update(status="RAISE", error=f"{type(exc).__name__}: {exc}"[:400])
    if caught:
        rec["warnings"] = sorted({str(w.message)[:160] for w in caught})[:4]
    return rec, arr


def slice_cases(fixtures_root, manifest) -> list[dict]:
    import pydicom
    cases = []
    tf = Path(pydicom.__file__).parent / "data" / "test_files"
    for name, twin, ts in BUNDLED:
        if (tf / name).is_file():
            cases.append({"origin": "bundled", "label": name, "ts": ts, "path": str(tf / name),
                          "twin": str(tf / twin) if twin and (tf / twin).is_file() else None,
                          "plane": None, "lossless": twin is not None})
    if fixtures_root and manifest:
        for src in manifest["slices"]:
            for v in src["variants"]:
                cases.append({"origin": "dataset", "source": src["source"], "label": f"{src['source']}/{v['label']}",
                              "ts": v["ts"], "path": str(Path(fixtures_root) / v["file"]),
                              "twin": str(Path(fixtures_root) / v["twin"]), "plane": src["plane"],
                              "lossless": bool(v["lossless"])})
    return cases


def run_slices(readers: Readers, cases) -> list[dict]:
    twin_cache = {}
    results = []
    for case in cases:
        row = {k: case[k] for k in ("origin", "label", "ts", "lossless")}
        row["readers"] = {}
        for name in SLICE_READERS:
            fn = readers.fns.get(name)
            if fn is None:
                row["readers"][name] = {"status": "SKIP", "class": "SKIP"}
                continue
            rec, arr = run_reader(fn, case["path"], case["plane"])
            twin = None
            if case["twin"]:
                key = (name, case["twin"])
                if key not in twin_cache:
                    twin_cache[key] = run_reader(fn, case["twin"], case["plane"])
                twin, twin_arr = twin_cache[key]
                if rec["status"] == "OK" and twin["status"] == "OK":
                    rec["compare"] = compare_arrays(arr, twin_arr)
                rec["twin_status"] = twin["status"]
            rec["class"] = classify(rec, twin)
            row["readers"][name] = rec
        results.append(row)
    return results


def alt_decode(cases) -> list[dict]:
    """Wheel-free routes (imagecodecs / cv2 / PIL / SimpleITK) on the dataset fixtures' frames."""
    import numpy as np
    import pydicom
    from pydicom.encaps import generate_frames
    mods = {}
    for m in ("imagecodecs", "cv2", "PIL.Image", "SimpleITK"):
        with contextlib.suppress(Exception):
            mods[m] = __import__(m, fromlist=["_"])
    rows = []
    for case in cases:
        if case["origin"] != "dataset" or case["ts"] in ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.5"):
            continue
        twin = pydicom.dcmread(case["twin"]).pixel_array
        ds = pydicom.dcmread(case["path"])
        frame = next(generate_frames(ds.PixelData, number_of_frames=1))
        tries = {}
        ic = mods.get("imagecodecs")
        if ic is not None:
            fnames = {"1.2.840.10008.1.2.4.80": ["jpegls_decode"], "1.2.840.10008.1.2.4.81": ["jpegls_decode"],
                      "1.2.840.10008.1.2.4.90": ["jpeg2k_decode"], "1.2.840.10008.1.2.4.91": ["jpeg2k_decode"]}.get(
                case["ts"], ["jpeg8_decode", "ljpeg_decode", "jpegsof3_decode"])
            for fname in fnames:
                tries[f"imagecodecs.{fname}"] = lambda f=fname: getattr(ic, f)(frame)
        cv2 = mods.get("cv2")
        if cv2 is not None:
            tries["cv2.imdecode"] = lambda: cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_UNCHANGED)
        image = mods.get("PIL.Image")
        if image is not None:
            tries["PIL.Image"] = lambda: np.asarray(image.open(io.BytesIO(frame)))
        sitk = mods.get("SimpleITK")
        if sitk is not None:
            tries["SimpleITK.ReadImage"] = lambda: np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(case["path"])))
        out = {}
        for name, fn in tries.items():
            try:
                arr = fn()
                if arr is None:
                    raise ValueError("decoder returned None")
                c = compare_arrays(np.asarray(arr).astype(twin.dtype) if np.asarray(arr).shape == twin.shape else arr, twin)
                out[name] = {"status": "EQUAL" if c["equal"] else "DIFF", "max_abs_diff": c.get("max_abs_diff"),
                             "shape": list(np.asarray(arr).shape), "dtype": str(np.asarray(arr).dtype)}
            except Exception as exc:
                out[name] = {"status": "RAISE", "error": f"{type(exc).__name__}: {exc}"[:200]}
        rows.append({"label": case["label"], "ts": case["ts"], "lossless": case["lossless"], "routes": out})
    return rows


def run_series(readers: Readers, fixtures_root, manifest, work: Path) -> dict:
    import numpy as np
    info = manifest["series"]
    uid, series, plane = info["study_uid"], info["series_uid"], info["plane"]
    rows_r = {uid: [{"StudyInstanceUID": uid, "SeriesInstanceUID": series, "Anatomical_Plane": plane,
                     "Fluid_Sensitive": int(info["fluid"]), "Fat_Suppression": int(info["fat"])}]}
    results = {}
    native = None
    if readers.geom is not None:
        try:
            rd = str(Path(fixtures_root) / "readers")
            if rd not in sys.path:
                sys.path.insert(0, rd)
            import probe_infer
            native = probe_infer.install_native(True)
            readers.probe_infer = probe_infer
        except Exception as exc:
            native = exc
    for v in info["variants"]:
        root = Path(fixtures_root) / v["root"]
        tsdir = root / "test_series"
        res = {}

        def guard(name, fn):
            try:
                res[name] = fn()
            except Exception as exc:
                res[name] = {"status": "RAISE", "error": f"{type(exc).__name__}: {exc}"[:400]}

        def vol_rec(vol, mask, extra=None):
            r = {"status": "OK", "vol_sha256": sha256_bytes(np.ascontiguousarray(vol).tobytes()),
                 "mask_sha256": sha256_bytes(np.ascontiguousarray(mask).tobytes()), "mask_sum": int(np.asarray(mask).sum())}
            r.update(extra or {})
            return r
        if readers.public is not None:
            ns, fast = readers.public, readers.fast

            def r_lut():
                build = fast.make_builder(ns["FASTREAD_VARIANT"], ns["IMG"], ns["SLOTS"],
                                          hdr_threads=ns["FR_HDR"], px_threads=ns["FR_PX"])
                return vol_rec(*build(uid, rows_r, str(tsdir)), {"variant": ns["FASTREAD_VARIANT"]})
            guard("R_study_lut", r_lut)
            guard("R_build_study_v2", lambda: vol_rec(*ns["build_study"](uid, rows_r, str(tsdir), ns["_make_reader"]())))
        if readers.raptor is not None:
            rp = readers.raptor

            def own():
                spec = rp._contract_spec(OWN_CONTRACT)
                try:
                    vol, mask, receipt = rp.prepare_study(uid, rows_r[uid], rp.DirectoryDicomSource(tsdir), spec, "test")
                except rp.DicomPreparationError as exc:
                    slots = [{k: s.get(k) for k in ("slot", "status", "failed_source")} for s in exc.receipt.get("slots", [])]
                    return {"status": "RAISE", "error": str(exc)[:400], "slots": slots}
                return vol_rec(vol, mask, {"receipt_status": receipt.get("status")})
            guard("OWN_prepare_study", own)
        if isinstance(native, Exception):
            res["S6R7_native_prepare"] = {"status": "RAISE", "error": f"setup {type(native).__name__}: {native}"[:400]}
        elif native is not None:
            def s6():
                out_dir = work / "native" / v["label"]
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                readers.probe_infer.OneStudySource.current = uid
                with contextlib.redirect_stdout(io.StringIO()):
                    m = native.prepare(str(root), str(out_dir), split="test", study_ids=[uid], source_directory=True)
                study = m["studies"][0]
                ser = []
                for s in study["series"]:
                    arr = np.load(s["path"], allow_pickle=False)
                    ser.append({"series": s["SeriesInstanceUID"], "array_sha256": sha256_bytes(np.ascontiguousarray(arr).tobytes()),
                                "shape": list(arr.shape), "files": len(s["files"]), "bad_files": len(s["errors"]),
                                "intensity": s["intensity"], "positions_sha256": sha256_bytes(json.dumps(s["positions"]).encode()),
                                "bad_examples": [e.get("error") for e in s["errors"][:2]]})
                return {"status": "OK" if ser else "DROPPED", "series": ser,
                        "study_errors": [str(e.get("error"))[:200] for e in study["errors"][:4]]}
            guard("S6R7_native_prepare", s6)
        results[v["label"]] = res
    ref = results.get("orig", {})
    for label, res in results.items():
        for name, r in res.items():
            r["class"] = series_class(r, ref.get(name))
    return results


def series_class(r: dict, ref: dict | None) -> str:
    if r.get("status") in ("RAISE", "DROPPED"):
        return r["status"]
    if not ref or ref.get("status") != "OK":
        return "REF_FAIL"
    if "vol_sha256" in r:
        if r["vol_sha256"] == ref["vol_sha256"] and r["mask_sha256"] == ref["mask_sha256"]:
            return "EQUAL"
        return f"DIFF(mask_sum {r['mask_sum']} vs {ref['mask_sum']})"
    if "series" in r:
        a = [(s["array_sha256"], s["files"], json.dumps(s["intensity"], sort_keys=True), s["positions_sha256"]) for s in r["series"]]
        b = [(s["array_sha256"], s["files"], json.dumps(s["intensity"], sort_keys=True), s["positions_sha256"]) for s in ref["series"]]
        if a == b:
            return "EQUAL"
        bad = sum(s["bad_files"] for s in r["series"])
        return f"DIFF(bad_files={bad})"
    return "UNKNOWN"


def competition_checks(comp: Path, manifest) -> dict:
    import pydicom
    out = {"root": str(comp)}
    if manifest:
        checks = []
        for rel, digest in manifest.get("competition_originals", {}).items():
            p = comp / rel
            checks.append({"relpath": rel, "present": p.is_file(),
                           "sha256_equal": (sha256_file(p) == digest) if p.is_file() else None})
        out["originals"] = checks
        out["originals_all_equal"] = bool(checks) and all(c["sha256_equal"] for c in checks)
    counts, errors, n = {}, 0, 0
    ts_root = comp / "test_series"
    if ts_root.is_dir():
        for study in sorted(ts_root.iterdir()):
            for series in sorted(study.iterdir()) if study.is_dir() else []:
                for f in sorted(series.glob("*.dcm")):
                    n += 1
                    try:
                        ts = str(pydicom.dcmread(str(f), stop_before_pixels=True).file_meta.TransferSyntaxUID)
                        counts[ts] = counts.get(ts, 0) + 1
                    except Exception:
                        errors += 1
    out["visible_test_transfer_syntaxes"] = counts
    out["visible_test_files"] = n
    out["visible_test_header_errors"] = errors
    return out


def child_main(args) -> int:
    started = time.time()
    result = {"phase": args.child, "version": VERSION, "started_utc": now_utc()}
    if args.simulate_missing:
        result["SIMULATED_MISSING_MODULES"] = block_modules(args.simulate_missing)
    print(f"{BANNER} child={args.child}", flush=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    result["environment"] = env = environment_report()
    print(f"HB126_PYDICOM {env.get('pydicom')} {env['modules'].get('pydicom', {}).get('file')}", flush=True)
    result["decoders"] = dec = decoder_table()
    for uid in TRANSFER_SYNTAXES:
        r = dec[uid]
        print(f"HB126_DECODER {uid} available={r.get('available')} plugins={r.get('plugins')} "
              f"missing={r.get('missing', r.get('error'))}", flush=True)
    result["alternates"] = alternate_decoders()
    manifest = None
    if args.fixtures_root:
        manifest = json.loads((Path(args.fixtures_root) / FIXTURES_MANIFEST).read_text(encoding="utf-8"))
    readers = Readers(Path(args.fixtures_root) / "readers" if args.fixtures_root else None, args.code_root)
    result["reader_load"] = readers.status
    print("HB126_READER_LOAD " + json.dumps(readers.status, sort_keys=True), flush=True)
    cases = slice_cases(args.fixtures_root, manifest)
    result["slices"] = slices = run_slices(readers, cases)
    by = {}
    for row in slices:
        for name, r in row["readers"].items():
            key = (row["origin"], name, row["ts"])
            by.setdefault(key, {}).setdefault(r["class"], 0)
            by[key][r["class"]] += 1
    for (origin, name, ts), counts in sorted(by.items()):
        print(f"HB126_SLICE origin={origin} reader={name} ts={ts} {json.dumps(counts, sort_keys=True)}", flush=True)
    try:
        result["alt_decode"] = alt = alt_decode(cases)
        counts = {}
        for row in alt:
            for route, r in row["routes"].items():
                key = (route, row["ts"], "lossless" if row["lossless"] else "lossy")
                counts.setdefault(key, {}).setdefault(r["status"], 0)
                counts[key][r["status"]] += 1
        for (route, ts, kind), c in sorted(counts.items()):
            print(f"HB126_ALT route={route} ts={ts} {kind} {json.dumps(c, sort_keys=True)}", flush=True)
    except Exception as exc:
        result["alt_decode"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    if manifest:
        try:
            result["series"] = run_series(readers, args.fixtures_root, manifest, work)
            for label, res in result["series"].items():
                print(f"HB126_SERIES_{args.child} {label} " +
                      json.dumps({k: v.get("class") for k, v in res.items()}, sort_keys=True), flush=True)
        except Exception as exc:
            result["series_error"] = traceback.format_exc()[-1500:]
            print(f"HB126_SERIES_ERROR {type(exc).__name__}: {exc}", flush=True)
    if args.competition_root and args.child == "base":
        try:
            result["competition"] = comp = competition_checks(Path(args.competition_root), manifest)
            print(f"HB126_COMPETITION originals_all_equal={comp.get('originals_all_equal')} "
                  f"visible_test_ts={comp['visible_test_transfer_syntaxes']}", flush=True)
        except Exception as exc:
            result["competition"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    result["seconds"] = round(time.time() - started, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"HB126_CHILD_DONE {args.child} seconds={result['seconds']}", flush=True)
    return 0


def default_work() -> Path:
    for c in ("/kaggle/tmp", "/tmp"):
        if os.path.isdir(c) and os.access(c, os.W_OK):
            return Path(c) / "hb126"
    return Path.cwd() / "hb126-work"


def build_parser():
    ap = argparse.ArgumentParser(description=BANNER)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--out", help="orchestrator: output dir; child: output JSON")
    ap.add_argument("--work", help="scratch dir (not saved)")
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--fixtures-dataset", help="override: folder holding the fixtures zip + manifest")
    ap.add_argument("--code-source", help="override: folder of the heliosli code dataset mount")
    ap.add_argument("--competition-root", help="override: competition mount")
    ap.add_argument("--wheel-dir", help="extra folder scanned for codec wheels")
    ap.add_argument("--fixtures-root", help=argparse.SUPPRESS)
    ap.add_argument("--code-root", help=argparse.SUPPRESS)
    ap.add_argument("--skip-phases", default="", help="comma list: pydicom301,codecs")
    ap.add_argument("--simulate-missing", help="LOCAL ONLY: comma list of modules to block (preview of a bare image)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.child:
        return child_main(args)
    print(BANNER, flush=True)
    args.out = args.out or ("/kaggle/working/" + OUT_NAME if os.path.isdir("/kaggle/working") else OUT_NAME)
    args.work = args.work or str(default_work())
    args.skip_phases = {p.strip() for p in args.skip_phases.split(",") if p.strip()}
    report = orchestrate(args)
    return 0 if report["phases"]["base"].get("returncode") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
