"""hb126_codecs 1.0.0: offline gdcm install, start-up hook, plugin order, telemetry (CPU only).

The real python-gdcm wheel is manylinux and cannot load on Windows: these tests use a fake pure-python
'python-gdcm' wheel whose gdcm module only reports a version (pydicom then registers its gdcm plugin,
which fails on use and falls through to the next plugin). The real wheel is exercised for the pin and
for the inactive path (pip refuses it here, unzip succeeds, verification fails -> INACTIVE). Real
GDCM decoding is covered by the WSL preview recorded in VALIDATION.md and by the Kaggle probe.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools import hb126_codecs as c

ROOT = Path(__file__).resolve().parents[1]
REAL_WHEEL = ROOT / "storage/external/codec-wheels-cp313-v1" / c.WHEEL["file"]
FAKE_GDCM = '''"""FAKE gdcm for tests: version stub only, cannot decode."""
class Version:
    @staticmethod
    def GetVersion():
        return "3.2.6"
'''


def fake_wheel(directory, name="python_gdcm-3.2.6-py3-none-any.whl", gdcm_py=FAKE_GDCM):
    files = {"gdcm.py": gdcm_py.encode(), "_gdcm/__init__.py": b"",
             "python_gdcm-3.2.6.dist-info/METADATA": b"Metadata-Version: 2.1\nName: python-gdcm\nVersion: 3.2.6\n",
             "python_gdcm-3.2.6.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: hb126-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    record = [f"{n},sha256={base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b'=').decode()},{len(d)}"
              for n, d in files.items()] + ["python_gdcm-3.2.6.dist-info/RECORD,,"]
    files["python_gdcm-3.2.6.dist-info/RECORD"] = ("\n".join(record) + "\n").encode()
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for n, d in files.items():
            z.writestr(zipfile.ZipInfo(n, date_time=(2026, 10, 13, 0, 0, 0)), d)
    data = path.read_bytes()
    return path, dict(c.WHEEL, file=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture(scope="module")
def active(tmp_path_factory):
    """One ACTIVE install (fake gdcm) shared by the read-only tests."""
    tmp = tmp_path_factory.mktemp("active")
    wheel, spec = fake_wheel(tmp / "wheel")
    rec = c.install(wheel, tmp / "root", wheel=spec, telemetry_dir=tmp / "tele",
                    base_env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return rec, tmp


def run_py(code, env, *args):
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env)


# ---------------------------------------------------------------- pins and self-test data

def test_pins_and_real_wheel_identity():
    assert c.WHEEL["sha256"] == "69c987d3f1b3a2a9936288f312f1ea2192e39fa6c0be28159b06a1906b8d56ed"
    assert c.WHEEL["size"] == 13277668 and c.WHEEL["file"].startswith("python_gdcm-3.2.6-cp313-cp313-manylinux")
    assert c.DATASET_SLUG == "taeyangg4/knee-hb126-codecs-cp313-v1"
    if REAL_WHEEL.is_file():
        assert c.sha256_file(REAL_WHEEL) == c.WHEEL["sha256"] and REAL_WHEEL.stat().st_size == c.WHEEL["size"]
        with zipfile.ZipFile(REAL_WHEEL) as z:
            tops = {n.split("/")[0] for n in z.namelist()}
        assert tops == {"gdcm.py", "_gdcm", "python_gdcm.libs", "python_gdcm-3.2.6.dist-info"}


def test_selftest_frames_decode_to_the_pinned_array_with_independent_decoders():
    ic = pytest.importorskip("imagecodecs")
    st = c.SELFTEST
    assert set(st["frames"]) == {"1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.80",
                                 "1.2.840.10008.1.2.4.90"}
    for uid, b64 in st["frames"].items():
        frame = base64.b64decode(b64)
        fn = ic.jpegls_decode if uid.endswith(".80") else ic.jpeg2k_decode if uid.endswith(".90") else ic.ljpeg_decode
        arr = np.asarray(fn(frame))
        assert arr.shape == (st["rows"], st["columns"]) and int(arr.max()) < 2 ** st["bits_stored"]
        assert hashlib.sha256(arr.astype("<u2").tobytes()).hexdigest() == st["array_sha256"], uid


# ---------------------------------------------------------------- install / verify

def test_install_active_with_usercustomize_entry(active):
    rec, tmp = active
    assert rec["status"] == "ACTIVE" and rec["reasons"] == [] and rec["method"] == "pip"
    assert rec["entry"] == "usercustomize" and [e["entry"] for e in rec["entries_tried"]] == ["usercustomize"]
    hook = Path(rec["hook_dir"])
    assert sorted(p.name for p in hook.iterdir() if p.suffix == ".py") == ["knee_hb126_hook.py", "usercustomize.py"]
    assert c.sha256_file(hook / "knee_hb126_hook.py") == rec["hook_sha256"]
    assert repr(str(Path(rec["site"]))) in (hook / "knee_hb126_hook.py").read_text(encoding="utf-8")
    assert Path(rec["root"]).is_absolute() and Path(rec["site"]).is_absolute()
    v = rec["verify"]
    assert v["gdcm"] == "3.2.6" and v["hook"]["patched"] and v["hook"]["entries"] == ["usercustomize"]
    assert all(t["equal"] for t in v["selftest"].values()) and len(v["selftest"]) == 4
    orders = v["orders"]
    for uid in c.TARGET_UIDS:
        assert "gdcm" in orders[uid]
    # gdcm last wherever another plugin is available; RLE (no gdcm plugin in pydicom 3) untouched
    assert all(labels[-1] == "gdcm" for labels in orders.values() if "gdcm" in labels and len(labels) > 1)
    assert "gdcm" not in orders.get("1.2.840.10008.1.2.5", [])
    tel = c.read_telemetry(tmp / "tele")
    assert tel["files"] == 1 and tel["totals"] == {"decoded_compressed_frames": 4, "plugin_failures": 0, "decode_failures": 0}
    assert set(tel["branches"]) == {"verify"} and tel["processes"][0]["gdcm"] == "3.2.6"


def test_install_falls_back_to_sitecustomize_when_user_site_disabled(tmp_path):
    wheel, spec = fake_wheel(tmp_path / "wheel")
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    rec = c.install(wheel, tmp_path / "root", wheel=spec, base_env=env, method="unzip")
    assert rec["status"] == "ACTIVE" and rec["entry"] == "sitecustomize" and rec["method"] == "unzip"
    assert [(e["entry"], e["status"]) for e in rec["entries_tried"]] == [
        ("usercustomize", "INACTIVE_HOOK_NOT_LOADED"), ("sitecustomize", "ACTIVE")]
    assert not (Path(rec["hook_dir"]) / "usercustomize.py").exists()
    assert rec["verify"]["user_site_enabled"] is False


@pytest.mark.parametrize("entry", ["usercustomize", "sitecustomize"])
def test_entry_chains_to_the_shadowed_customize_module(tmp_path, active, entry):
    rec, _ = active
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "knee_hb126_hook.py").write_text(c.hook_source(rec["site"]), encoding="utf-8")
    (hook / f"{entry}.py").write_text(c.entry_source(entry), encoding="utf-8")
    image = tmp_path / "image"
    image.mkdir()
    (image / f"{entry}.py").write_text("import os\nos.environ['IMAGE_CUSTOMIZE_RAN'] = __name__ + ':' + __file__\n",
                                       encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(hook), str(image)]), PYTHONDONTWRITEBYTECODE="1")
    code = ("import os, sys, json; m = sys.modules['knee_hb126_hook'].KNEE_HB126_STATE; "
            f"print(json.dumps([os.environ.get('IMAGE_CUSTOMIZE_RAN'), m['entries'], m['chained'], sys.modules['{entry}'].__file__]))")
    out = run_py(code, env)
    assert out.returncode == 0, out.stderr
    ran, entries, chained, module_file = json.loads(out.stdout.strip().splitlines()[-1])
    assert ran.startswith(entry + ":") and entries == [entry]
    assert len(chained) == 1 and Path(chained[0]).parent == image and Path(module_file).parent == image


def test_failing_chained_module_reports_but_hook_stays_active(tmp_path, active):
    rec, _ = active
    hook, image = tmp_path / "hook", tmp_path / "image"
    hook.mkdir(); image.mkdir()
    (hook / "knee_hb126_hook.py").write_text(c.hook_source(rec["site"]), encoding="utf-8")
    (hook / "usercustomize.py").write_text(c.entry_source("usercustomize"), encoding="utf-8")
    (image / "usercustomize.py").write_text("raise RuntimeError('image customize broken')\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(hook), str(image)]), PYTHONDONTWRITEBYTECODE="1")
    out = run_py("import sys, gdcm; print(sys.modules['knee_hb126_hook'].KNEE_HB126_STATE['armed'], gdcm.Version.GetVersion())", env)
    assert out.returncode == 0 and out.stdout.split()[-2:] == ["True", "3.2.6"]
    assert "image customize broken" in out.stderr


def test_without_hook_gdcm_is_not_importable_and_env_unchanged(active):
    rec, _ = active
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    out = run_py("import importlib.util as u; print(u.find_spec('gdcm') is None, u.find_spec('knee_hb126_hook') is None)", env)
    assert out.stdout.split() == ["True", "True"]  # the private site dir is reachable only through the hook
    assert c.child_env(env, {"status": "INACTIVE_INSTALL"}, branch="R") == env
    assert c.child_env(env, {}, branch="R") == env


def test_child_env_appends_hook_last_and_strip_env_restores(active):
    rec, tmp = active
    base = {"PYTHONPATH": "/runtime", "X": "1"}
    env = c.child_env(base, rec, branch="R", telemetry_dir=tmp / "t")
    assert env["PYTHONPATH"].split(os.pathsep) == ["/runtime", rec["hook_dir"]]
    assert env[c.BRANCH_ENV] == "R" and env[c.TELEMETRY_ENV] == str(tmp / "t") and env["X"] == "1"
    assert c.child_env(env, rec, branch="S")["PYTHONPATH"] == env["PYTHONPATH"]  # idempotent
    assert c.strip_env(env, rec) == base
    assert "PYTHONPATH" not in c.strip_env(c.child_env({}, rec, branch="R"), rec)


@pytest.mark.parametrize("fault", ["no wheel", "wrong sha", "wrong name", "root exists"])
def test_install_inactive_paths_never_raise(tmp_path, fault):
    wheel, spec = fake_wheel(tmp_path / "wheel")
    if fault == "no wheel":
        rec = c.install(None, tmp_path / "root", wheel=spec)
        assert rec["status"] == "INACTIVE_NO_WHEEL"
        return
    if fault == "wrong sha":
        spec = dict(spec, sha256="0" * 64)
    if fault == "wrong name":
        spec = dict(spec, file="other.whl")
    if fault == "root exists":
        (tmp_path / "root").mkdir()
    rec = c.install(wheel, tmp_path / "root", wheel=spec)
    expected = {"wrong sha": "INACTIVE_WHEEL_PIN", "wrong name": "INACTIVE_WHEEL_PIN", "root exists": "INACTIVE_ERROR"}
    assert rec["status"] == expected[fault] and "seconds" in rec
    assert c.child_env({"PYTHONPATH": "/r"}, rec, branch="R") == {"PYTHONPATH": "/r"}


@pytest.mark.skipif(not REAL_WHEEL.is_file() or sys.platform != "win32", reason="real manylinux wheel on Windows only")
def test_real_wheel_on_windows_is_inactive_and_leaves_an_inert_hook_dir(tmp_path):
    rec = c.install(REAL_WHEEL, tmp_path / "root", base_env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert rec["status"].startswith("INACTIVE_") and rec["entry"] is None
    assert [a["method"] for a in rec["attempts"]] == ["pip", "unzip"] and rec["attempts"][0]["ok"] is False
    assert rec["method"] == "unzip" and "GDCM_IMPORT" in rec["reasons"]
    assert not any((Path(rec["hook_dir"]) / f"{n}.py").exists() for n in c.ENTRY_NAMES)


def test_verify_detects_a_hook_that_does_not_load(tmp_path, active):
    rec, _ = active
    broken = dict(rec, hook_dir=str(tmp_path / "missing"))
    result = c.verify(broken)
    status, reasons = c.evaluate(result, broken)
    assert status == "INACTIVE_HOOK_NOT_LOADED" and "GDCM_IMPORT" in reasons


def test_evaluate_reasons():
    rec = {"hook_dir": "/h"}
    good = {"hook_file": os.path.join("/h", "knee_hb126_hook.py"), "hook": {"patched": True, "errors": []},
            "gdcm": "3.2.6", "orders": {**{u: ["gdcm"] for u in c.TARGET_UIDS},
                                         "1.2.840.10008.1.2.4.90": ["pillow", "gdcm"]},
            "selftest": {u: {"equal": True} for u in c.SELFTEST["frames"]}}
    assert c.evaluate(good, rec) == ("ACTIVE", [])
    assert c.evaluate(dict(good, orders={**good["orders"], "1.2.840.10008.1.2.4.90": ["gdcm", "pillow"]}), rec)[0] == "INACTIVE_GDCM_NOT_LAST"
    assert c.evaluate(dict(good, orders={"1.2.840.10008.1.2.4.90": ["pillow"]}), rec)[0] == "INACTIVE_GDCM_NOT_REGISTERED"
    bad = dict(good, selftest={**good["selftest"], "1.2.840.10008.1.2.4.80": {"equal": False}})
    assert c.evaluate(bad, rec)[0] == "INACTIVE_SELFTEST"
    assert c.evaluate(dict(good, hook={"patched": True, "errors": ["x"]}), rec)[0] == "INACTIVE_HOOK_NOT_PATCHED"
    assert c.evaluate(dict(good, error="boom"), rec)[0] == "INACTIVE_VERIFY_ERROR"


# ---------------------------------------------------------------- hook behaviour

HOOK_PROBE = r'''
import json, os, sys, threading
import numpy as np
import pydicom
from pydicom.pixels.decoders import base
st = sys.modules["knee_hb126_hook"].KNEE_HB126_STATE
spec = json.loads(sys.argv[1])
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
import base64
def make(uid, frame):
    ds = Dataset(); ds.file_meta = FileMetaDataset(); ds.file_meta.TransferSyntaxUID = uid
    ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = spec["rows"], spec["columns"], 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = encapsulate([frame]); ds["PixelData"].VR = "OB"; ds["PixelData"].is_undefined_length = True
    return ds
uid = "1.2.840.10008.1.2.4.70"
good = base64.b64decode(spec["frames"][uid])
os.environ["KNEE_CODEC_BRANCH"] = "A"
arrs = []
threads = [threading.Thread(target=lambda: arrs.append(make(uid, good).pixel_array)) for _ in range(6)]
[t.start() for t in threads]; [t.join() for t in threads]
os.environ["KNEE_CODEC_BRANCH"] = "B"
err = None
try:
    make(uid, b"\xff\xd8garbage").pixel_array
except Exception as exc:
    err = type(exc).__name__
plain = Dataset(); plain.file_meta = FileMetaDataset(); plain.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
plain.Rows, plain.Columns, plain.SamplesPerPixel, plain.PhotometricInterpretation = 2, 3, 1, "MONOCHROME2"
plain.BitsAllocated, plain.BitsStored, plain.HighBit, plain.PixelRepresentation = 16, 12, 11, 0
plain.PixelData = np.arange(6, dtype="<u2").tobytes()
print(json.dumps({"n": len(arrs), "equal": all(a.tobytes() == arrs[0].tobytes() for a in arrs), "err": err,
                  "plain": plain.pixel_array.tolist(), "counts": st["counts"], "final": st["final_failures"],
                  "samples": len(st["error_samples"])}))
'''


def test_hook_counts_per_branch_threads_failures_and_skips_uncompressed(tmp_path, active):
    rec, _ = active
    tele = tmp_path / "tele"
    env = c.child_env(dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), rec, branch="X", telemetry_dir=tele)
    out = run_py(HOOK_PROBE, env, json.dumps(c.SELFTEST))
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["n"] == 6 and r["equal"] and r["err"] == "RuntimeError" and r["plain"] == [[0, 1, 2], [3, 4, 5]]
    uid = "1.2.840.10008.1.2.4.70"
    labels = rec["verify"]["orders"][uid]
    assert r["counts"]["A"][uid][labels[0]] == [6, 0]          # all six decoded by the first plugin
    assert all(r["counts"]["B"][uid][label] == [0, 1] for label in labels)  # garbage: every plugin failed once
    assert r["final"] == {"B": {uid: 1}} and r["samples"] == len(labels)
    assert "1.2.840.10008.1.2.1" not in json.dumps(r["counts"])  # native syntax never reaches a plugin
    tel = c.read_telemetry(tele)
    assert tel["files"] == 1 and tel["branches"]["A"]["decoded_compressed_frames"] == 6
    assert tel["branches"]["B"]["decode_failures"] == 1 and tel["branches"]["B"]["plugin_failures"] == len(labels)
    assert tel["totals"]["decode_failures"] == 1 and len(tel["orders_variants"]) == 1


def test_hook_patches_after_spawn_too(tmp_path, active):
    rec, _ = active
    script = tmp_path / "spawner.py"
    script.write_text(
        "import multiprocessing as mp, sys, json\n"
        "def child():\n"
        "    import pydicom\n"
        "    from pydicom.pixels.decoders import base\n"
        "    st = sys.modules['knee_hb126_hook'].KNEE_HB126_STATE\n"
        "    return st['patched'], list(base.JPEGLosslessSV1Decoder._available)\n"
        "if __name__ == '__main__':\n"
        "    with mp.get_context('spawn').Pool(1) as pool:\n"
        "        print(json.dumps(pool.apply(child)))\n", encoding="utf-8")
    env = c.child_env(dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), rec, branch="S", telemetry_dir=tmp_path / "t")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    patched, labels = json.loads(out.stdout.strip().splitlines()[-1])
    assert patched is True and labels[-1] == "gdcm"
    assert c.read_telemetry(tmp_path / "t")["files"] >= 1


def test_hook_source_anchor_and_entries_compile():
    src = c.hook_source("/x/site")
    assert src.count("__HB126_SITE__ = '/x/site'") == 1
    compile(src, "hook", "exec")
    for name in c.ENTRY_NAMES:
        compile(c.entry_source(name), name, "exec")
    with pytest.raises(ValueError):
        c.entry_source("other")
    compile(c.VERIFY_SOURCE, "verify", "exec")


def test_hook_with_missing_site_is_harmless(tmp_path):
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "knee_hb126_hook.py").write_text(c.hook_source(tmp_path / "nowhere"), encoding="utf-8")
    (hook / "usercustomize.py").write_text(c.entry_source("usercustomize"), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(hook), PYTHONDONTWRITEBYTECODE="1")
    out = run_py("import pydicom, sys; from pydicom.pixels.decoders import base; "
                 "print(sys.modules['knee_hb126_hook'].KNEE_HB126_STATE['patched'], 'gdcm' in base.JPEGLosslessDecoder._available)", env)
    assert out.returncode == 0 and out.stdout.split()[-2:] == ["True", "False"]


# ---------------------------------------------------------------- helpers

def test_read_telemetry_aggregates_and_tolerates_garbage(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    for i, (branch, ok, fail, fin) in enumerate([("R", 5, 1, 0), ("R", 2, 0, 0), ("S_R7", 0, 3, 3)]):
        rec = {"pid": i, "branch": branch, "orders": {"u": ["gdcm"]}, "patched": True, "errors": [],
               "counts": {branch: {"1.2.840.10008.1.2.4.70": {"gdcm": [ok, fail]}}},
               "final_failures": {branch: {"1.2.840.10008.1.2.4.70": fin}} if fin else {}, "error_samples": []}
        (d / f"codec-{i}-ab.json").write_text(json.dumps(rec), encoding="utf-8")
    (d / "codec-9-zz.json").write_text("{not json", encoding="utf-8")
    (d / "codec-8-zz.json.tmp").write_text("{}", encoding="utf-8")
    t = c.read_telemetry(d)
    assert t["files"] == 3 and t["unreadable"] == ["codec-9-zz.json"]
    assert t["branches"]["R"]["decoded_compressed_frames"] == 7 and t["branches"]["R"]["plugin_failures"] == 1
    assert t["branches"]["S_R7"]["decode_failures"] == 3
    assert t["totals"] == {"decoded_compressed_frames": 7, "plugin_failures": 4, "decode_failures": 3}
    absent = c.read_telemetry(tmp_path / "none")
    assert absent["present"] is False and absent["totals"]["decode_failures"] == 0


def test_find_wheel_prunes_series_trees(tmp_path):
    (tmp_path / "comp/train_series/x").mkdir(parents=True)
    (tmp_path / "comp/train_series/x" / c.WHEEL["file"]).write_bytes(b"x")
    assert c.find_wheel(tmp_path) is None
    (tmp_path / "datasets/u/codecs").mkdir(parents=True)
    (tmp_path / "datasets/u/codecs" / c.WHEEL["file"]).write_bytes(b"x")
    assert c.find_wheel(tmp_path) == tmp_path / "datasets/u/codecs" / c.WHEEL["file"]
    assert c.find_wheel(tmp_path / "absent") is None


def test_safe_unzip_refuses_traversal(tmp_path):
    bad = tmp_path / "bad.whl"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.py", b"x")
    with pytest.raises(ValueError):
        c.safe_unzip(bad, tmp_path / "out")
    assert not (tmp_path / "evil.py").exists()


def test_activate_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "find_wheel", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    assert c.activate(tmp_path)["status"] == "INACTIVE_ERROR"
    monkeypatch.setattr(c, "find_wheel", lambda *a, **k: None)
    assert c.activate(tmp_path, root=tmp_path / "r")["status"] == "INACTIVE_NO_WHEEL"
