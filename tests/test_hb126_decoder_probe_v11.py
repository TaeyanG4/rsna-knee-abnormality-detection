"""HB126 decoder probe v1.1.0 + builder: verdict/KEEP units, artifact integrity, local smoke (CPU only).

The smoke run uses a fake pure-python 'python-gdcm' wheel (version stub; the real manylinux wheel
cannot load on Windows) with the full local decoder set, so it exercises install -> hook -> codec
phases -> spawn child -> KEEP plumbing, not GDCM itself. Real GDCM 3.2.6 decoding of the same
fixtures is the WSL preview in VALIDATION.md; the Kaggle kernel is the decision.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import hb126_build_v11 as b
from tools import hb126_codecs as c
from tools import hb126_decoder_probe_v11 as p
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling test helper (tests/ is not a package)
from test_hb126_codecs import fake_wheel  # noqa: E402

V100 = b.V100
FIXTURES = V100 / "fixtures-dataset"
CODE_SOURCE = b.ROOT / ".local/exp189-cpu-v120/prepare_smoke/dataset_mount"
built = pytest.mark.skipif(not (FIXTURES / "hb126-fixtures-v1.zip.bin").is_file(), reason="v1.0.0 fixtures absent")


# ---------------------------------------------------------------- units

def test_series_class_ref_raise_and_readers_all_equal():
    ok = {"status": "OK", "vol_sha256": "a", "mask_sha256": "m", "mask_sum": 3}
    assert p.series_class(dict(ok), ok) == "EQUAL"
    assert p.series_class({"status": "RAISE"}, {"status": "RAISE"}) == "REF_RAISE"
    assert p.series_class({"status": "RAISE"}, ok) == "RAISE"
    assert p.series_class({"status": "DROPPED"}, ok) == "DROPPED"
    phase = {"lossless_dataset_reader_classes": {"x": {"EQUAL": 3}},
             "series": {"orig": {"OWN_prepare_study": "REF_RAISE", "R_study_lut": "EQUAL"}}}
    assert p.readers_all_equal(phase) is True
    phase["series"]["orig"]["R_study_lut"] = "REF_RAISE"
    assert p.readers_all_equal(phase) is False
    assert p.readers_all_equal({"lossless_dataset_reader_classes": {"x": {"EQUAL": 2, "SKIP": 1}}, "series": {}}) is False


def _slice(label, **readers):
    return {"label": label, "readers": {k: v for k, v in readers.items()}}


def test_keep_compare_counts_same_changed_recovered_still_failing():
    ok = lambda s: {"status": "OK", "sha256": s, "shape": [2, 2], "dtype": "uint16"}
    ref = {"slices": [_slice("a", r1=ok("x"), r2={"status": "RAISE"}, r3={"status": "BLANK"}, r4={"status": "SKIP"}),
                      _slice("b", r1=ok("y"))],
           "series": {"orig": {"R": {"status": "OK", "vol_sha256": "v", "mask_sha256": "m"},
                               "S": {"status": "OK", "series": [1]}, "O": {"status": "DROPPED"}}}}
    res = {"slices": [_slice("a", r1=ok("x"), r2=ok("z"), r3={"status": "BLANK"}, r4={"status": "SKIP"}),
                      _slice("b", r1=ok("CHANGED")), _slice("new", r1=ok("q"))],
           "series": {"orig": {"R": {"status": "OK", "vol_sha256": "v", "mask_sha256": "m"},
                               "S": {"status": "OK", "series": [2]}, "O": {"status": "OK"}}}}
    k = p.keep_compare(ref, res, "base")
    assert (k["same"], k["changed"], k["recovered"], k["still_failing"]) == (2, 2, 2, 1)
    assert k["changed_items"] == ["slice b r1", "series orig S"]
    assert "slice a r2" in k["recovered_items"] and "series orig O" in k["recovered_items"]


def _hook(**over):
    orders = {**{u: ["gdcm"] for u in c.TARGET_UIDS}, "1.2.840.10008.1.2.4.90": ["pillow", "gdcm"]}
    return {"patched": True, "errors": [], "gdcm": "3.2.6", "orders": orders, **over}


def test_hook_ok_and_spawn_ok():
    assert p.hook_ok(_hook()) is True
    assert p.hook_ok(_hook(gdcm=None)) is False
    assert p.hook_ok(_hook(errors=["x"])) is False
    assert p.hook_ok(_hook(orders={**_hook()["orders"], "1.2.840.10008.1.2.4.90": ["gdcm", "pillow"]})) is False
    assert p.hook_ok(None) is False
    assert p.spawn_ok({"hook": _hook(), "equal": True}) is True
    assert p.spawn_ok({"hook": _hook(), "equal": False}) is False
    assert p.spawn_ok({"hook": None, "equal": True}) is False


def _phase(ran=True, **over):
    rec = {"ran": ran, "missing_codec_required": [], "hook_ok": True, "spawn_ok": True,
           "lossless_dataset_reader_classes": {"r": {"EQUAL": 21}}, "series": {"orig": {"R": "EQUAL"}},
           "keep": {"changed": 0}}
    rec.update(over)
    return rec


def test_verdicts_v11():
    report = {"codec_install": {"status": "ACTIVE"}, "phases": {}}
    s = {"base": {"ran": True}, "fixtures_ok": True, "gdcm": _phase(), "gdcm_pydicom301": _phase()}
    assert p.verdict_v11(s, report) == "GO_CODECS_RELEASE_READY"
    assert p.verdict_v11(dict(s, base={"ran": False}), report) == "INCONCLUSIVE_BASE_PHASE_FAILED"
    assert p.verdict_v11(dict(s, fixtures_ok=False), report) == "INCONCLUSIVE_FIXTURES_MISSING"
    assert p.verdict_v11(s, {"codec_install": {"status": "INACTIVE_GDCM_IMPORT"}, "phases": {}}) == \
        "NO_GO_CODEC_INSTALL_INACTIVE_GDCM_IMPORT"
    assert p.verdict_v11(s, {"phases": {"codecs": {"skipped": "NO_PINNED_CODEC_WHEEL"}}}) == \
        "NO_GO_NO_CODEC_INSTALL:NO_PINNED_CODEC_WHEEL"
    bad = dict(s, gdcm=_phase(keep={"changed": 2}, spawn_ok=False), gdcm_pydicom301=_phase(ran=False))
    assert p.verdict_v11(bad, report) == "NO_GO_gdcm:KEEP_CHANGED,gdcm:SPAWN,gdcm_pydicom301:NOT_RUN"
    bad = dict(s, gdcm=_phase(lossless_dataset_reader_classes={"r": {"EQUAL": 12, "RAISE": 9}},
                              missing_codec_required=["1.2.840.10008.1.2.4.57"], hook_ok=False))
    assert p.verdict_v11(bad, report) == "NO_GO_gdcm:MISSING_DECODERS,gdcm:HOOK,gdcm:READERS"
    assert p.verdict_v11(dict(s, gdcm=_phase(keep=None)), report) == "NO_GO_gdcm:KEEP_CHANGED"


def test_compare_trees(tmp_path):
    a, b_ = tmp_path / "a", tmp_path / "b"
    for d in (a, b_):
        (d / "pkg-1.dist-info").mkdir(parents=True)
        (d / "m.py").write_text("x", encoding="utf-8")
        (d / "pkg-1.dist-info" / "METADATA").write_text("m", encoding="utf-8")
    (a / "pkg-1.dist-info" / "RECORD").write_text("pip only", encoding="utf-8")
    (a / "pkg-1.dist-info" / "INSTALLER").write_text("pip", encoding="utf-8")
    assert p.compare_trees(a, b_)["equal"] is True
    (b_ / "m.py").write_text("y", encoding="utf-8")
    r = p.compare_trees(a, b_)
    assert r["equal"] is False and r["differing"] == ["m.py"]
    assert p.compare_trees(None, b_)["equal"] is False


def test_pip_target_unzips_pure_wheels_when_pip_fails(tmp_path, monkeypatch):
    wheel, _ = fake_wheel(tmp_path / "w")
    real = subprocess.run

    class Failed:
        returncode, stdout = 1, "No module named pip"
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **k: Failed())
    rec = p.pip_target([wheel], tmp_path / "t", tmp_path / "log.txt")
    monkeypatch.setattr(p.subprocess, "run", real)
    assert rec["returncode"] == 0 and rec["method"] == "unzip" and rec["pip_returncode"] == 1
    assert (tmp_path / "t" / "gdcm.py").is_file()


def test_codec_spec_override(tmp_path):
    args = p.build_parser().parse_args([])
    assert p.codec_spec(args) == c.WHEEL
    (tmp_path / "s.json").write_text(json.dumps({"file": "x.whl", "size": 1, "sha256": "0" * 64}), encoding="utf-8")
    args = p.build_parser().parse_args(["--codec-wheel-spec", str(tmp_path / "s.json")])
    assert p.codec_spec(args)["file"] == "x.whl" and p.codec_spec(args)["version"] == c.WHEEL["version"]


# ---------------------------------------------------------------- built artifacts

def test_codec_dataset_folder_private_and_pinned():
    d = b.OUT / "codecs-dataset"
    meta = json.loads((d / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == c.DATASET_SLUG == "taeyangg4/knee-hb126-codecs-cp313-v1"
    assert meta["isPrivate"] is True and meta["licenses"] == [{"name": "other"}] and meta["title"] == meta["id"].split("/")[1]
    assert sorted(x.name for x in d.iterdir()) == sorted([c.WHEEL["file"], c.MANIFEST, "dataset-metadata.json"])
    man = json.loads((d / c.MANIFEST).read_text(encoding="utf-8"))
    entry = man["files"][c.WHEEL["file"]]
    assert entry["sha256"] == c.WHEEL["sha256"] == c.sha256_file(d / c.WHEEL["file"])
    assert entry["size"] == c.WHEEL["size"] == (d / c.WHEEL["file"]).stat().st_size


def test_kernel_metadata_and_embedded_sources():
    k = b.OUT / "kernel"
    meta = json.loads((k / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "taeyangg4/knee-hb126-decoder-probe-v1" and meta["code_file"] == b.NOTEBOOK
    assert meta["is_private"] and not meta["enable_gpu"] and not meta["enable_internet"] and "machine_shape" not in meta
    assert meta["dataset_sources"] == ["taeyangg4/knee-hb126-decoder-fixtures-v1", c.DATASET_SLUG,
                                       "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
    old = json.loads((V100 / "kernel/kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["docker_image"] == old["docker_image"] and meta["competition_sources"] == old["competition_sources"]
    nb = json.loads((k / b.NOTEBOOK).read_text(encoding="utf-8"))
    cell = "".join(nb["cells"][1]["source"])
    compile(cell, "launcher", "exec")
    tree = __import__("ast").parse(cell)
    sources = next(__import__("ast").literal_eval(n.value) for n in tree.body
                   if getattr(n, "targets", None) and getattr(n.targets[0], "id", "") == "SOURCES")
    assert sorted(sources) == ["hb126_codecs.py", "hb126_decoder_probe_v11.py"]
    codecs_src = base64.b64decode(sources["hb126_codecs.py"][0])
    assert codecs_src == b.CODECS.read_bytes() and hashlib.sha256(codecs_src).hexdigest() == sources["hb126_codecs.py"][1]
    probe_src = base64.b64decode(sources["hb126_decoder_probe_v11.py"][0]).decode()
    tool = b.PROBE.read_text(encoding="utf-8")
    assert probe_src == tool.replace("FIXTURES_ZIP_SHA256 = None\n", f'FIXTURES_ZIP_SHA256 = "{b.FIXTURES_ZIP_SHA256}"\n')
    receipt = json.loads((b.OUT / "build-receipt.json").read_text(encoding="utf-8"))
    assert receipt["kernel"]["notebook_sha256"] == c.sha256_file(k / b.NOTEBOOK)
    assert receipt["kernel"]["tool_sha256"]["hb126_decoder_probe_v11.py"] == c.sha256_file(b.PROBE)


def test_v100_probe_artifacts_unchanged():
    expected = {"tools/hb126_decoder_probe.py": "151c2f96a449ad8d9e7dc9bdf9d34c081a94a892daf5104d72c75f3cac988fc6",
                "tools/hb126_build_fixtures.py": "fe1153519c008db50f0ef0579249d0db4313207e4df3fb80face78450a7d6459",
                "tests/test_hb126_decoder_probe.py": "b2d7c647182dab36339de58b121e09ab9bc44dc9d393c7bb59cfab7717952805"}
    for rel, digest in expected.items():
        assert c.sha256_file(b.ROOT / rel) == digest, rel
    receipt = json.loads((V100 / "build-receipt.json").read_text(encoding="utf-8"))
    assert c.sha256_file(V100 / "kernel" / b.NOTEBOOK) == receipt["kernel"]["notebook_sha256"]


# ---------------------------------------------------------------- local smoke

@built
@pytest.mark.skipif(not CODE_SOURCE.is_dir(), reason="local S6/R7 code mirror absent")
def test_local_smoke_fake_gdcm_full_plumbing(tmp_path):
    """Full local decoders + fake gdcm: install, hook, codec phases, spawn, KEEP -> GO (plumbing only)."""
    wheel, spec = fake_wheel(tmp_path / "wheel")
    (tmp_path / "spec.json").write_text(json.dumps({k: spec[k] for k in ("file", "size", "sha256")}), encoding="utf-8")
    (tmp_path / "input").mkdir()
    out, work = tmp_path / "out", tmp_path / "work"
    cmd = [sys.executable, str(b.PROBE), "--input-root", str(tmp_path / "input"), "--fixtures-dataset", str(FIXTURES),
           "--code-source", str(CODE_SOURCE), "--codec-wheel", str(wheel), "--codec-wheel-spec", str(tmp_path / "spec.json"),
           "--out", str(out), "--work", str(work)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    report = json.loads((out / "probe.json").read_text(encoding="utf-8"))
    s = report["summary"]
    assert s["verdict"] == "GO_CODECS_RELEASE_READY", s["verdict"]
    assert report["codec_install"]["status"] == "ACTIVE" and report["codec_tree_compare"]["equal"] is True
    for phase, ref in p.RELEASE_PHASES.items():
        assert s[phase]["hook_ok"] and s[phase]["spawn_ok"] and s[phase]["keep"]["changed"] == 0
        assert s[phase]["keep"]["same"] > 200 and s[phase]["pydicom"] == s[ref]["pydicom"]
        assert set(s[phase]["series"]["orig"].values()) == {"EQUAL"}  # OWN_prepare_study fixed (was RAISE in v1.0.0)
    assert s["gdcm_first"]["hook_ok"] is False and s["gdcm_first"]["ran"]
    tel = report["codec_telemetry"]
    assert {"verify", "probe:gdcm", "probe:gdcm_pydicom301"} <= set(tel["branches"])
    lines = (out / "summary.txt").read_text(encoding="utf-8").splitlines()
    assert lines[1] == "HB126_VERDICT GO_CODECS_RELEASE_READY"
    assert any(x.startswith("HB126_KEEP gdcm vs=base") for x in lines)
