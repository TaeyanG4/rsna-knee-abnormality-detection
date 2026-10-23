"""HB126 decoder probe v1.0.0 + fixture builder: helper units, artifact integrity, local smoke.

CPU only. The integration test runs the probe on the built fixtures with codec modules
blocked (a local preview of a bare image); it is skipped when the artifacts are absent.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools import hb126_build_fixtures as b
from tools import hb126_decoder_probe as p

OUT = b.OUT
DATASET = OUT / "fixtures-dataset"
KERNEL = OUT / "kernel"
CODE_SOURCE = b.ROOT / ".local/exp189-cpu-v120/prepare_smoke/dataset_mount"
built = pytest.mark.skipif(not (DATASET / b.ZIP_NAME).is_file(), reason="fixtures not built")

GDCM312 = "python_gdcm-3.2.6-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
KAGGLE_TAGS = ["cp313-cp313-manylinux_2_28_x86_64", "cp313-cp313-manylinux_2_27_x86_64",
               "cp313-cp313-manylinux_2_17_x86_64", "cp313-cp313-manylinux2014_x86_64",
               "cp313-abi3-manylinux_2_17_x86_64", "py3-none-any", "py313-none-any"]


# ---------------------------------------------------------------- wheel tags
def test_parse_wheel_expands_compressed_tag_sets():
    info = p.parse_wheel_name(GDCM312)
    assert info["distribution"] == "python_gdcm" and info["version"] == "3.2.6"
    assert info["tags"] == ["cp312-cp312-manylinux_2_27_x86_64", "cp312-cp312-manylinux_2_28_x86_64"]
    assert not p.wheel_compatible(info, KAGGLE_TAGS)


@pytest.mark.parametrize("bad", ["x.tar.gz", "a-b.whl", "a-1-2-3-4-5-6.whl"])
def test_parse_wheel_rejects_malformed(bad):
    with pytest.raises(ValueError):
        p.parse_wheel_name(bad)


def test_required_wheels_fit_a_cp313_manylinux_image():
    files = [w for items in p.REQUIRED_WHEELS.values() for w in items]
    assert {w["package"] for w in files} == {"python-gdcm", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"}
    for w in files:
        info = p.parse_wheel_name(w["file"])
        assert p.wheel_compatible(info, KAGGLE_TAGS), w["file"]
        assert w["url"].endswith("/" + w["file"]) and w["url"].startswith("https://files.pythonhosted.org/")
        assert len(w["sha256"]) == 64 and int(w["size"]) > 0 and w["licence"]


def test_inventory_finds_only_codec_wheels_and_prunes_series(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / GDCM312).write_bytes(b"x")
    (tmp_path / "ds" / "pylibjpeg_fake-1.0-py3-none-any.whl").write_bytes(b"y")
    (tmp_path / "ds" / "timm-1.0.25-py3-none-any.whl").write_bytes(b"z")
    hidden = tmp_path / "comp" / "train_series" / "s"
    hidden.mkdir(parents=True)
    (hidden / "pylibjpeg_hidden-1.0-py3-none-any.whl").write_bytes(b"h")
    found = {w["file"]: w for w in p.inventory_wheels(tmp_path)}
    assert set(found) == {GDCM312, "pylibjpeg_fake-1.0-py3-none-any.whl"}
    assert found["pylibjpeg_fake-1.0-py3-none-any.whl"]["compatible"] is True
    assert found[GDCM312]["compatible"] is False  # never cp312 on this cp312-win or the cp313 image
    assert found[GDCM312]["sha256"] == hashlib.sha256(b"x").hexdigest()


def _fake_wheel(directory: Path, name="pylibjpeg_hb126fake", version="0.0.1") -> Path:
    whl = directory / f"{name}-{version}-py3-none-any.whl"
    dist = f"{name}-{version}.dist-info"
    files = {f"{name}.py": b"VALUE = 126\n",
             f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
             f"{dist}/WHEEL": b"Wheel-Version: 1.0\nGenerator: hb126-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    record = "".join(f"{k},sha256={base64.urlsafe_b64encode(hashlib.sha256(v).digest()).rstrip(b'=').decode()},{len(v)}\n"
                     for k, v in files.items()) + f"{dist}/RECORD,,\n"
    with zipfile.ZipFile(whl, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
        z.writestr(f"{dist}/RECORD", record)
    return whl


def test_pip_target_installs_offline_into_a_private_dir(tmp_path):
    whl = _fake_wheel(tmp_path)
    rec = p.pip_target([whl], tmp_path / "site", tmp_path / "pip.log")
    assert rec["returncode"] == 0, (tmp_path / "pip.log").read_text()
    assert "--no-index" in rec["command"] and "--no-deps" in rec["command"] and "--target" in rec["command"]
    out = subprocess.run([sys.executable, "-c", "import pylibjpeg_hb126fake as m; print(m.VALUE)"],
                         env=dict(os.environ, PYTHONPATH=str(tmp_path / "site")), capture_output=True, text=True)
    assert out.stdout.strip() == "126"


# ---------------------------------------------------------------- archive safety
@pytest.mark.parametrize("name,ok", [("a/b.dcm", True), ("readers/x.py", True), ("../x", False), ("/abs", False),
                                     ("a\\b", False), ("C:x", False), ("a/../b", False), ("a//b", False), ("", False)])
def test_safe_member(name, ok):
    assert p.safe_member(name) is ok


def test_safe_extract_refuses_traversal(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("ok.txt", "1")
        z.writestr("../evil.txt", "2")
    with pytest.raises(ValueError):
        p.safe_extract(bad, tmp_path / "dest")
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "dest" / "ok.txt").exists()


# ---------------------------------------------------------------- comparison / verdict
def test_compare_and_classify():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert p.compare_arrays(a, a.copy())["equal"]
    c = p.compare_arrays(a, a.astype(np.float64))
    assert not c["equal"] and c["max_abs_diff"] == 0.0 and not c["dtype_equal"]
    d = p.compare_arrays(a, a + 2)
    assert d["max_abs_diff"] == 2.0 and d["n_diff"] == 6
    assert not p.compare_arrays(a, a.ravel())["shape_equal"]
    ok = {"status": "OK"}
    assert p.classify({"status": "OK", "compare": {"equal": True}}, ok) == "EQUAL"
    assert p.classify({"status": "OK", "compare": {"equal": False}}, ok) == "DIFF"
    assert p.classify({"status": "BLANK"}, ok) == "BLANK"
    assert p.classify({"status": "RAISE"}, ok) == "RAISE"
    assert p.classify({"status": "OK"}, {"status": "RAISE"}) == "TWIN_FAIL"
    assert p.classify({"status": "OK"}, None) == "DECODED"


def _phase(missing, classes, series=None):
    return {"ran": True, "missing_critical": missing, "lossless_dataset_reader_classes": classes, "series": series or {}}


def test_verdicts():
    eq = {"R": {"EQUAL": 21}}
    assert p.verdict_from({"ran": False}, {}) == "INCONCLUSIVE_BASE_PHASE_FAILED"
    assert p.verdict_from(_phase([], eq), {}) == "NO_GO_DECODERS_PRESENT_ADD_FIXTURE_GATE"
    assert p.verdict_from(_phase([], {}), {}) == "DECODERS_PRESENT_READERS_UNTESTED"
    assert p.verdict_from(_phase([], {"R": {"EQUAL": 20, "BLANK": 1}}), {}) == "DECODERS_PRESENT_READER_MISMATCH"
    assert p.verdict_from(_phase([], eq, {"x": {"R": "RAISE"}}), {}) == "DECODERS_PRESENT_READER_MISMATCH"
    crit = _phase([".70"], {"R": {"BLANK": 9, "EQUAL": 12}})
    assert p.verdict_from(crit, {}) == "GO_PENDING_CP313_WHEELS_APPROVAL"
    assert p.verdict_from(crit, {"codecs_gdcm": {"ran": False}}) == "GO_PENDING_CP313_WHEELS_APPROVAL"
    assert p.verdict_from(crit, {"codecs_gdcm": _phase([], eq), "codecs_pylibjpeg": _phase([".70"], eq)}) == \
        "GO_CANDIDATE_ATTACHED_WHEELS_FIX_CRITICAL:codecs_gdcm"
    assert p.verdict_from(crit, {"codecs_pylibjpeg": _phase([".70"], eq)}) == "ATTACHED_WHEELS_INSUFFICIENT"


def test_codec_groups_one_phase_per_family_plus_all():
    ws = [{"file": n} for n in ("python_gdcm-3.2.6-cp313-cp313-manylinux_2_28_x86_64.whl",
                                 "pylibjpeg-2.1.0-py3-none-any.whl",
                                 "pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.whl")]
    groups = p.codec_groups(ws)
    assert list(groups) == ["codecs_gdcm", "codecs_pylibjpeg", "codecs_all"]
    assert len(groups["codecs_pylibjpeg"]) == 2 and len(groups["codecs_all"]) == 3
    assert list(p.codec_groups(ws[:1])) == ["codecs_gdcm"] and p.codec_groups([]) == {}


def test_series_class():
    ref = {"status": "OK", "vol_sha256": "a", "mask_sha256": "m", "mask_sum": 18}
    assert p.series_class(dict(ref), ref) == "EQUAL"
    assert p.series_class(dict(ref, vol_sha256="b", mask_sum=0), ref) == "DIFF(mask_sum 0 vs 18)"
    assert p.series_class({"status": "DROPPED"}, ref) == "DROPPED"
    assert p.series_class(dict(ref), {"status": "RAISE"}) == "REF_FAIL"


def test_walk_files_never_enters_series_trees(tmp_path):
    (tmp_path / "test_series" / "s").mkdir(parents=True)
    (tmp_path / "test_series" / "s" / "a.whl").write_bytes(b"")
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "b.whl").write_bytes(b"")
    import re
    assert [x.name for x in p.walk_files(tmp_path, re.compile(r"\.whl$"))] == ["b.whl"]


# ---------------------------------------------------------------- reader loading
def test_public_serving_reader_extracted_without_torch():
    src = b.READERS["public-serving.py"][0]
    assert p.sha256_file(src) == p.READER_PINS["public-serving.py"]
    ns = p.load_public_serving(src)
    assert ns["FASTREAD_VARIANT"] == "study_lut" and ns["IMG"] == 384 and ns["MAXS"] == 96
    assert "torch" not in ns and "timm" not in ns
    fast = p.load_fastread(ns)
    assert hasattr(fast, "make_builder") and hasattr(fast, "_Ctx")


def test_reader_pins_match_builder_sources():
    for name, (path, digest) in b.READERS.items():
        assert p.READER_PINS[name] == digest == p.sha256_file(path)


def test_jpeg_precision_parser():
    sof3 = bytes([0xFF, 0xD8, 0xFF, 0xC4, 0x00, 0x04, 0x00, 0x00, 0xFF, 0xC3, 0x00, 0x0B, 12, 0, 1, 0, 1, 1, 1, 0x11, 0])
    assert b.jpeg_precision(sof3) == {"marker": "0xC3", "precision": 12}
    assert b.jpeg_precision(b"\xff\xd8\xff\xd9") is None


def test_pinned_probe_source_replaces_once_and_compiles():
    src = b.pinned_probe_source("f" * 64)
    assert src.count('FIXTURES_ZIP_SHA256 = "' + "f" * 64 + '"') == 1 and "FIXTURES_ZIP_SHA256 = None" not in src
    compile(src, "probe", "exec")
    cell = b.launcher_cell(src)
    compile(cell, "cell", "exec")
    assert "HB126 decoder probe v1.0.0" in cell


@pytest.mark.parametrize("src,layout", [(CODE_SOURCE, "cloud_code.zip"),
                                        (b.ROOT / ".local/exp189-cpu-v120/input", "cloud_code/")])
def test_code_dataset_both_layouts_pass_pins(tmp_path, src, layout):
    if not (src / "anatomy.pt").is_file():
        pytest.skip("local code-dataset mirror absent")
    rec = p.prepare_code(src, tmp_path)
    assert rec["layout"] == layout and rec["status"] == "OK" and all(rec["pins"].values())
    assert (Path(rec["root"]) / "training/native_cache.py").is_file()
    wheel = p.find_pydicom301(src, tmp_path)
    if layout == "cloud_code.zip":
        assert wheel is not None and p.sha256_file(wheel) == p.PYDICOM301[1]


# ---------------------------------------------------------------- built artifacts
@built
def test_kernel_metadata_and_embedded_probe():
    meta = json.loads((KERNEL / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "taeyangg4/knee-hb126-decoder-probe-v1" and meta["title"] == "knee-hb126-decoder-probe-v1"
    assert meta["is_private"] is True and meta["enable_gpu"] is False and meta["enable_internet"] is False
    assert meta["docker_image"].endswith("sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f")
    assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert "machine_shape" not in meta and meta["kernel_type"] == "notebook" and meta["code_file"] == b.NOTEBOOK
    receipt = json.loads((OUT / "build-receipt.json").read_text(encoding="utf-8"))
    nb = json.loads((KERNEL / b.NOTEBOOK).read_text(encoding="utf-8"))
    code = "".join(nb["cells"][1]["source"])
    blob = code.split("SRC = '")[1].split("'")[0]
    embedded = base64.b64decode(blob).decode("utf-8")
    assert hashlib.sha256(embedded.encode()).hexdigest() == receipt["kernel"]["probe_source_sha256"]
    assert f'FIXTURES_ZIP_SHA256 = "{receipt["dataset"]["zip_sha256"]}"' in embedded
    # The kernel copy differs from tools/ only by the pinned zip digest.
    assert embedded == b.pinned_probe_source(receipt["dataset"]["zip_sha256"])
    assert p.sha256_file(DATASET / b.ZIP_NAME) == receipt["dataset"]["zip_sha256"]


@built
def test_dataset_metadata_private_and_zip_matches_manifest():
    meta = json.loads((DATASET / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == b.DATASET_ID and meta["isPrivate"] is True
    manifest = json.loads((DATASET / b.MANIFEST).read_text(encoding="utf-8"))
    with zipfile.ZipFile(DATASET / b.ZIP_NAME) as z:
        names = set(z.namelist())
        assert names == set(manifest["files"]) | {b.MANIFEST}
        assert all(p.safe_member(n) for n in names)
        for rel in list(manifest["files"])[:40]:
            assert hashlib.sha256(z.read(rel)).hexdigest() == manifest["files"][rel]
        assert z.read(b.MANIFEST) == (DATASET / b.MANIFEST).read_bytes()
    for s in manifest["slices"]:
        for v in s["variants"]:
            if v["lossless"]:
                assert set(v["max_abs_diff_vs_source"].values()) == {0}, (s["source"], v["label"])
            else:
                assert v["twin"].endswith(".twin.dcm")
    assert {v["ts"] for s in manifest["slices"] for v in s["variants"]} >= set(p.REQUESTED)
    assert not any("train.csv" in k for k in manifest["competition_originals"])


@built
@pytest.mark.skipif(not CODE_SOURCE.is_dir(), reason="local S6/R7 code mirror absent")
def test_local_smoke_bare_image_preview(tmp_path):
    """Codec modules blocked: JPEG lossless must show R blank / own raise / S6R7 drop, J2K via Pillow equal."""
    out, work, inp = tmp_path / "out", tmp_path / "work", tmp_path / "input" / "fake-codec-dataset"
    inp.mkdir(parents=True)
    _fake_wheel(inp)  # a compatible "codec" wheel: exercises install + codecs_pylibjpeg child plumbing
    cmd = [sys.executable, str(b.PROBE), "--input-root", str(tmp_path / "input"),
           "--fixtures-dataset", str(DATASET), "--code-source", str(CODE_SOURCE),
           "--out", str(out), "--work", str(work), "--skip-phases", "pydicom301",
           "--simulate-missing", "pylibjpeg,libjpeg,openjpeg,rle,gdcm,jpeg_ls,imagecodecs,SimpleITK"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    report = json.loads((out / "probe.json").read_text(encoding="utf-8"))
    base = report["summary"]["base"]
    assert base["missing_critical"] == ["1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70"]
    # The fake wheel installs and its phase runs, but (blocked modules) it fixes nothing.
    phase = report["phases"]["codecs_pylibjpeg"]
    assert phase["install"]["returncode"] == 0 and phase["returncode"] == 0
    assert report["summary"]["codecs_pylibjpeg"]["ran"] is True
    assert report["summary"]["verdict"] == "ATTACHED_WHEELS_INSUFFICIENT"
    assert p.verdict_from(report["summary"]["base"], {}) == "GO_PENDING_CP313_WHEELS_APPROVAL"
    assert base["lossless_dataset_reader_classes"]["R_fastread_study_lut"] == {"BLANK": 9, "EQUAL": 12}
    assert base["lossless_dataset_reader_classes"]["OWN_raptor_decode"] == {"RAISE": 9, "EQUAL": 12}
    series = base["series"]
    assert series["jpeg_lossless_sv1"]["S6R7_native_prepare"] == "DROPPED"
    assert series["jpeg_lossless_sv1"]["OWN_prepare_study"] == "RAISE"
    assert series["jpeg_lossless_sv1"]["R_study_lut"].startswith("DIFF(mask_sum 0")
    assert set(series["j2k_lossless"].values()) == {"EQUAL"} and set(series["rle_lossless"].values()) == {"EQUAL"}
    assert report["fixtures"]["status"] == "OK" and report["fixtures"]["pinned_zip_sha256"] is None
    assert report["phases"]["pydicom301"]["skipped"] == "skipped"
