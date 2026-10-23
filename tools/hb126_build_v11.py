#!/usr/bin/env python3
"""HB126 v1.1.0 builder (H-B-126 step 2). LOCAL ONLY: no network, no Kaggle, no upload.

Writes artifacts/experiments/HB126_DECODER_PROBE/v1.1.0/:
  codecs-dataset/  private dataset taeyangg4/knee-hb126-codecs-cp313-v1: the pinned python-gdcm
                   3.2.6 cp313 wheel (copied from storage/external/codec-wheels-cp313-v1/, sha256
                   checked), hb126-codecs-manifest.json, dataset-metadata.json
  kernel/          new version of taeyangg4/knee-hb126-decoder-probe-v1: one launcher cell that
                   writes the sha-checked probe v1.1.0 (fixtures zip sha pinned, same v1.0.0 fixtures
                   dataset) and tools/hb126_codecs.py next to it, then runs the probe
  build-receipt.json

    python tools/hb126_build_v11.py            # (re)build both folders; idempotent

Changelog v1.1.0 (2026-10-13, Claude opus builder stand-in for B): first version.
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import hb126_codecs as codecs  # noqa: E402

VERSION = "1.1.0"
OUT = ROOT / "artifacts/experiments/HB126_DECODER_PROBE/v1.1.0"
V100 = ROOT / "artifacts/experiments/HB126_DECODER_PROBE/v1.0.0"
PROBE = ROOT / "tools/hb126_decoder_probe_v11.py"
CODECS = ROOT / "tools/hb126_codecs.py"
WHEEL_SOURCE = ROOT / "storage/external/codec-wheels-cp313-v1" / codecs.WHEEL["file"]
KERNEL_ID = "taeyangg4/knee-hb126-decoder-probe-v1"
NOTEBOOK = "hb126-decoder-probe-v1.ipynb"
FIXTURES_SLUG = "taeyangg4/knee-hb126-decoder-fixtures-v1"
CODE_SLUG = "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"
DATASETS = [FIXTURES_SLUG, codecs.DATASET_SLUG, CODE_SLUG]
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
FIXTURES_ZIP_SHA256 = "58e0f4f029edb5e426e5b7f242bef480e19d548b6892671123911ebfebbb4ed3"
SRC_DIR = "hb126-src-v1.1.0"


def need(ok, msg):
    if not ok:
        raise SystemExit("HB126 v1.1.0 build: " + msg)


def sha(path) -> str:
    return codecs.sha256_file(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def pinned_probe_source() -> str:
    src = PROBE.read_text(encoding="utf-8")
    old = "FIXTURES_ZIP_SHA256 = None\n"
    need(src.count(old) == 1, "probe pin anchor changed")
    need(f'VERSION = "{VERSION}"' in src, "probe version is not " + VERSION)
    return src.replace(old, f'FIXTURES_ZIP_SHA256 = "{FIXTURES_ZIP_SHA256}"\n')


def sources() -> dict:
    return {"hb126_decoder_probe_v11.py": pinned_probe_source(), "hb126_codecs.py": CODECS.read_text(encoding="utf-8")}


def launcher_cell(files: dict) -> str:
    pins = {name: (base64.b64encode(text.encode("utf-8")).decode("ascii"),
                   hashlib.sha256(text.encode("utf-8")).hexdigest()) for name, text in sorted(files.items())}
    return (
        "import base64, hashlib, os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"print('HB126 decoder probe v{VERSION} (launcher)', flush=True)\n"
        f"SOURCES = {pins!r}\n"
        "base = Path('/kaggle/tmp') if os.access('/kaggle/tmp', os.W_OK) else Path('/tmp')\n"
        f"src_dir = base / {SRC_DIR!r}\n"
        "src_dir.mkdir(parents=True, exist_ok=True)\n"
        "for name, (blob, digest) in SOURCES.items():\n"
        "    data = base64.b64decode(blob)\n"
        "    assert hashlib.sha256(data).hexdigest() == digest, 'probe source drift: ' + name\n"
        "    (src_dir / name).write_bytes(data)\n"
        "proc = subprocess.Popen([sys.executable, '-u', str(src_dir / 'hb126_decoder_probe_v11.py')],\n"
        "                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,\n"
        "                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))\n"
        "for line in proc.stdout:\n"
        "    print(line, end='', flush=True)\n"
        "print('HB126_EXIT', proc.wait(), flush=True)\n")


def kernel_metadata() -> dict:
    return {"id": KERNEL_ID, "title": KERNEL_ID.split("/")[1], "code_file": NOTEBOOK, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "docker_image": IMAGE, "dataset_sources": DATASETS,
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}


def notebook(files: dict) -> dict:
    md = (f"# HB126 decoder probe v{VERSION} (diagnostic only)\n\n"
          "Private CPU kernel, internet off, pinned C85-R7 image. Installs the pinned python-gdcm 3.2.6 cp313 wheel "
          "offline exactly as the C85-R7+codecs release does (private --target dir + start-up hook) and checks that "
          "the R / own / S6-R7 serving readers decode JPEG Lossless, JPEG-LS, J2K and RLE fixtures exactly like the "
          "uncompressed originals under pydicom 3.0.2 and 3.0.1, and that every array that decoded without codecs "
          "stays byte-identical. Never produces a submission.\n")
    return {"cells": [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": launcher_cell(files).splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def dataset_metadata() -> dict:
    return {"title": codecs.DATASET_SLUG.split("/")[1], "id": codecs.DATASET_SLUG, "licenses": [{"name": "other"}],
            "isPrivate": True,
            "description": ("Private H-B-126 codec dataset: the unmodified python-gdcm 3.2.6 CPython 3.13 manylinux "
                            "wheel from PyPI (Apache-2.0; bundles GDCM, BSD-style licence, LICENSE inside the wheel) "
                            "plus a sha256 manifest. Installed offline by our private kernels only.")}


def codec_manifest() -> dict:
    w = codecs.WHEEL
    return {"version": "1.0.0", "dataset": codecs.DATASET_SLUG, "purpose": "offline JPEG Lossless / JPEG-LS decoder "
            "for pydicom in the pinned Kaggle image (H-B-126, D-B-162)",
            "files": {w["file"]: {"sha256": w["sha256"], "size": w["size"], "package": w["package"],
                                  "version": w["version"], "licence": w["licence"], "source_url": w["url"],
                                  "python": "cp313", "platform": "manylinux_2_27/2_28 x86_64"}},
            "provenance": "downloaded from PyPI by B on 2026-10-13 with Taeyang's approval; sha256 verified "
                          "against the PyPI JSON API value; unmodified",
            "installed_by": "tools/hb126_codecs.py install(): pip install --no-index --no-deps --target <private "
                            "dir> (unzip fallback); never into the image's site-packages"}


def write_dataset() -> dict:
    ddir = OUT / "codecs-dataset"
    need(WHEEL_SOURCE.is_file(), f"wheel missing: {WHEEL_SOURCE}")
    need(sha(WHEEL_SOURCE) == codecs.WHEEL["sha256"] and WHEEL_SOURCE.stat().st_size == codecs.WHEEL["size"],
         "wheel sha256/size differ from the pin")
    ddir.mkdir(parents=True, exist_ok=True)
    target = ddir / codecs.WHEEL["file"]
    if not target.is_file() or sha(target) != codecs.WHEEL["sha256"]:
        shutil.copyfile(WHEEL_SOURCE, target)
    write_text(ddir / codecs.MANIFEST, json.dumps(codec_manifest(), indent=2) + "\n")
    write_text(ddir / "dataset-metadata.json", json.dumps(dataset_metadata(), indent=2) + "\n")
    extra = sorted(p.name for p in ddir.iterdir() if p.name not in (target.name, codecs.MANIFEST, "dataset-metadata.json"))
    need(not extra, f"unexpected files in the dataset folder: {extra}")
    need(sha(target) == codecs.WHEEL["sha256"], "copied wheel differs")
    return {"folder": str(ddir.relative_to(ROOT)).replace("\\", "/"), "wheel_sha256": sha(target),
            "manifest_sha256": sha(ddir / codecs.MANIFEST), "metadata_sha256": sha(ddir / "dataset-metadata.json")}


def write_kernel() -> dict:
    files = sources()
    kdir = OUT / "kernel"
    write_text(kdir / NOTEBOOK, json.dumps(notebook(files), indent=1) + "\n")
    write_text(kdir / "kernel-metadata.json", json.dumps(kernel_metadata(), indent=2) + "\n")
    extra = sorted(p.name for p in kdir.iterdir() if p.name not in (NOTEBOOK, "kernel-metadata.json"))
    need(not extra, f"unexpected files in the kernel folder: {extra}")
    return {"notebook_sha256": sha(kdir / NOTEBOOK), "metadata_sha256": sha(kdir / "kernel-metadata.json"),
            "embedded_sha256": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in files.items()},
            "tool_sha256": {"hb126_decoder_probe_v11.py": sha(PROBE), "hb126_codecs.py": sha(CODECS)},
            "fixtures_zip_sha256": FIXTURES_ZIP_SHA256}


def main(argv=None) -> int:
    print(f"HB126 v{VERSION} builder", flush=True)
    v100 = json.loads((V100 / "build-receipt.json").read_text(encoding="utf-8"))
    need(v100["dataset"]["zip_sha256"] == FIXTURES_ZIP_SHA256, "v1.0.0 fixtures zip pin differs")
    receipt = {"version": VERSION, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "dataset": write_dataset(), "kernel": write_kernel(), "codecs_module_version": codecs.VERSION,
               "fixtures": {"dataset": FIXTURES_SLUG, "zip_sha256": FIXTURES_ZIP_SHA256,
                            "built_by": "tools/hb126_build_fixtures.py v1.0.0 (unchanged)"}}
    write_text(OUT / "build-receipt.json", json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
