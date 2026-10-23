"""C85-R7+codecs v1.1.0 CPU gates: identical to C85-R7 v1.0.0 except the codec environment.

Process boundaries are mocked as in the v1.0.0 suite; codec installs use the fake pure-python gdcm
wheel of tests/test_hb126_codecs.py (the real manylinux wheel cannot load on Windows).
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tools import exp169_release_c85 as c85
from tools import exp169_release_c85r7 as v100
from tools import exp169_release_c85r7codec as m
from tools import hb126_codecs as codecs
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling test helpers (tests/ is not a package)
from test_exp169_release_c85r7 import A_FILES, copy_files, update_json  # noqa: E402
from test_hb126_codecs import fake_wheel  # noqa: E402


@pytest.fixture(scope="module")
def active(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("codec")
    wheel, spec = fake_wheel(tmp / "wheel")
    rec = codecs.install(wheel, tmp / "root", wheel=spec, base_env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert rec["status"] == "ACTIVE"
    return rec


@pytest.fixture
def staged(tmp_path):
    return m.stage_replay_input(tmp_path / "pr7")


# ---------------------------------------------------------------- pins and identity with v1.0.0

def test_pins_and_v100_unchanged():
    assert m.sha(m.ROOT / "tools/exp169_release_c85r7.py") == m.C85R7_SHA
    assert m.sha(m.ROOT / "tools/hb126_codecs.py") == m.CODECS_SHA
    assert v100.validate(v100.OUT)["status"] == "PASS_CPU_VALIDATE"
    for line in (v100.OUT / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            path = Path(name.strip().lstrip("*"))
            path = path if path.is_absolute() else m.ROOT / path
            if path.is_file():
                assert m.sha(path) == digest, name
    assert m.FORMULA == v100.FORMULA and m.MEMBERS == v100.MEMBERS
    assert (m.RUN, m.RECEIPT, m.STAGE, m.DIAGNOSTIC, m.P_BRANCH) == (v100.RUN, v100.RECEIPT, v100.STAGE, v100.DIAGNOSTIC, v100.P_BRANCH)


def test_cpu_replay_identical_to_v100(tmp_path):
    result = m.cpu_replay(tmp_path)
    assert result == m.cpu_replay(tmp_path)
    assert result["status"] == "PASS_CPU_ARITHMETIC_REPLAY" and result["diagnostic_sha256"] == v100.C85_REPLAY_CSV_SHA
    assert result["c85r7_v100_csv_identical"] and result["c85r7_v100_p_branch_identical"]
    assert result["receipt_diff_vs_v100"] == ["codecs", "release", "version"] and result["codecs_status"] == "ABSENT"
    receipt = m.read_json(tmp_path / m.RECEIPT)
    assert receipt["version"] == "1.1.0" and receipt["release"] == "C85R7+codecs" and receipt["pr7"]["source_version"] == "C85R7CODEC 1.1.0"
    assert receipt["codecs"]["totals"] == {"decoded_compressed_frames": 0, "plugin_failures": 0, "decode_failures": 0}


def write_telemetry(directory, branch, ok, fail, final):
    directory.mkdir(parents=True, exist_ok=True)
    rec = {"pid": len(list(directory.iterdir())) + 100, "branch": branch, "orders": {"u": ["gdcm"]}, "patched": True,
           "errors": [], "gdcm": "3.2.6", "pydicom": "3.0.2", "error_samples": [],
           "counts": {branch: {"1.2.840.10008.1.2.4.70": {"gdcm": [ok, fail]}}},
           "final_failures": {branch: {"1.2.840.10008.1.2.4.70": final}} if final else {}}
    (directory / f"codec-{rec['pid']}-aa.json").write_text(json.dumps(rec), encoding="utf-8")


def test_collect_with_codec_telemetry_keeps_csv_and_counts_per_branch(tmp_path, staged):
    plain = tmp_path / "plain"
    _, payload_plain = v100.collect(c85.A5Q, staged, plain)
    out = tmp_path / "codec"
    cdir, tele = m.codec_dirs(out / m.RUN)
    cdir.mkdir(parents=True)
    m.write_once(cdir / "install.json", m.json_bytes({"status": "ACTIVE", "method": "pip", "entry": "usercustomize",
        "hook_dir": "/h", "verify": {"gdcm": "3.2.6", "pydicom": "3.0.2", "selftest": {"u": {"equal": True}}},
        "wheel": {"sha256": codecs.WHEEL["sha256"]}, "seconds": 9.0}))
    m.write_once(cdir / "gate-s_r7.json", m.json_bytes({"status": "ACTIVE", "pydicom": "3.0.1"}))
    write_telemetry(tele, "R", 30, 0, 0)
    write_telemetry(tele, "S_R7", 28, 2, 2)
    write_telemetry(tele, "A172:own", 30, 0, 0)
    receipt, payload = m.collect(c85.A5Q, staged, out)
    assert payload == payload_plain and (out / "submission.csv").read_bytes() == (plain / "submission.csv").read_bytes()
    assert (out / m.P_BRANCH).read_bytes() == (plain / m.P_BRANCH).read_bytes()
    cd = receipt["codecs"]
    assert cd["status"] == "ACTIVE" and cd["s_r7_gate"] == "ACTIVE" and cd["s_r7_gate_pydicom"] == "3.0.1"
    assert cd["branches"]["R"]["decoded_compressed_frames"] == 30 and cd["branches"]["S_R7"]["decode_failures"] == 2
    assert cd["totals"] == {"decoded_compressed_frames": 88, "plugin_failures": 2, "decode_failures": 2}
    assert cd["telemetry_files"] == 3 and len(cd["processes"]) == 3
    assert m.read_json(out / m.RECEIPT)["codecs"]["totals"] == cd["totals"] and receipt["status"] == "COMPLETE_C85R7"


def test_codec_summary_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(m.codecs, "read_telemetry", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    assert m.codec_summary(tmp_path)["status"] == "TELEMETRY_ERROR"


def test_failed_execution_receipt_has_version_and_codecs(tmp_path, staged, monkeypatch):
    output = tmp_path / "out"
    monkeypatch.setattr(m, "write_once", lambda path, data: (_ for _ in ()).throw(OSError("injected"))
                        if path.name == "submission.csv" else v100.write_once(path, data))
    with pytest.raises(OSError):
        m.collect(c85.A5Q, staged, output)
    receipt = m.read_json(output / m.RECEIPT)
    assert receipt["status"] == "FAILED_EXECUTION" and receipt["version"] == "1.1.0" and "codecs" in receipt
    assert not (output / "submission.csv").exists()


# ---------------------------------------------------------------- runtime orchestration

def _runtime(tmp_path):
    blob, pins = m.package()
    runtime = tmp_path / "runtime"
    m.p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    return runtime


@pytest.mark.parametrize("codec_state", ["inactive", "active", "activate raises"])
def test_run_release_env_and_scripts(tmp_path, monkeypatch, active, codec_state):
    runtime, output = _runtime(tmp_path), tmp_path / "output"
    if codec_state == "active":
        monkeypatch.setattr(m.codecs, "activate", lambda *a, **k: dict(active))
    elif codec_state == "activate raises":
        monkeypatch.setattr(m.codecs, "activate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    else:
        monkeypatch.setattr(m.codecs, "activate", lambda *a, **k: {"status": "INACTIVE_NO_WHEEL"})
    events, envs = [], {}
    def executor(args, *, label, env):
        events.append(label)
        envs[label] = env
        if label == "C85R7 frozen A172":
            copy_files(c85.A5Q, output / m.RUN / "a172", A_FILES)
        else:
            assert args[2:4] == [str(runtime / "exp169_release_c85r7codec.py"), "public-worker"]
            m.stage_replay_input(output / m.RUN / "pr7")
    monkeypatch.setattr(m.v5q, "run_process_group", executor)
    monkeypatch.setattr(m, "preflight", lambda *a: c85.authenticated_a172(c85.A5Q)[0])
    result = m.run_release(runtime, output)
    assert events == ["C85R7 frozen A172", "C85R7 R + S_R7"]
    assert result["status"] == "COMPLETE_C85R7" and result["submission_sha256"] == v100.C85_REPLAY_CSV_SHA
    v100_env = m.child_base_env(runtime)
    script = (output / m.RUN / "a172.py").read_text(encoding="utf-8")
    v100_script = c85.a172_source((runtime / "a172-cell.py.txt").read_text(encoding="utf-8"), output / m.RUN / "a172")
    install = m.read_json(output / m.RUN / "codecs/install.json")
    if codec_state == "active":
        for label, branch in (("C85R7 frozen A172", "A172"), ("C85R7 R + S_R7", "PR7")):
            env = envs[label]
            assert env["PYTHONPATH"] == str(runtime) + os.pathsep + active["hook_dir"]
            assert env[codecs.BRANCH_ENV] == branch and env[codecs.TELEMETRY_ENV] == str(output / m.RUN / "codecs/telemetry")
            assert {k: v for k, v in env.items() if k not in ("PYTHONPATH", codecs.BRANCH_ENV, codecs.TELEMETRY_ENV)} == \
                {k: v for k, v in v100_env.items() if k != "PYTHONPATH"}
        assert script == m.A172_LABEL_HEADER + v100_script and install["status"] == "ACTIVE"
        assert result["codecs"]["status"] == "ACTIVE"
    else:
        assert envs["C85R7 frozen A172"] == v100_env and envs["C85R7 R + S_R7"] == v100_env
        assert script == v100_script
        assert install["status"] == ("INACTIVE_ERROR" if codec_state == "activate raises" else "INACTIVE_NO_WHEEL")


def test_install_codecs_receipt_write_failure_stays_inactive(tmp_path, monkeypatch, active):
    monkeypatch.setattr(m.codecs, "activate", lambda *a, **k: dict(active))
    monkeypatch.setattr(m, "write_once", lambda *a: (_ for _ in ()).throw(OSError("read-only")))
    rec = m.install_codecs(tmp_path / "runtime", tmp_path / "out")
    assert rec["status"] == "INACTIVE_RECEIPT_WRITE"
    assert codecs.child_env({"PYTHONPATH": "x"}, rec, branch="A172") == {"PYTHONPATH": "x"}


def _public_layout(tmp_path, codec=None):
    out = tmp_path / "run" / "pr7"
    work = out / "pr7-run"
    work.mkdir(parents=True)
    m.write_once(out / "ids.json", m.json_bytes(["a"]))
    m.write_once(work / "r7-identity.json", m.json_bytes(dict(manifest_sha256="f" * 64)))
    m.write_once(work / "code-paths.json", m.json_bytes(dict(code=str(work / "code"), anatomy=str(work / "anatomy.pt"))))
    if codec is not None:
        m.write_once(tmp_path / "run/codecs/install.json", m.json_bytes(codec))
    return out, work


@pytest.mark.parametrize("case", ["inactive", "active gate ok", "active gate fails", "R fails"])
def test_public_worker_codec_env_and_gate(tmp_path, monkeypatch, active, case):
    codec = None if case == "inactive" else active
    out, work = _public_layout(tmp_path, codec)
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(m, "runtime_pins", lambda p: None)
    monkeypatch.setattr(m, "competition_root", lambda: tmp_path / "competition")
    monkeypatch.setattr(m.v5q, "_mount", lambda *a: tmp_path / "mount")
    monkeypatch.setattr(m.p, "check_pin", lambda *a: None)
    monkeypatch.setattr(m.s6, "install_wheels", lambda *a: None)
    parent_env = codecs.child_env(dict(os.environ, PYTHONPATH=str(runtime)), codec or {}, branch="PR7",
                                  telemetry_dir=tmp_path / "run/codecs/telemetry")
    for k in list(os.environ):
        monkeypatch.delenv(k)
    for k, v in parent_env.items():
        monkeypatch.setenv(k, v)
    if case == "active gate fails":
        monkeypatch.setattr(m.codecs, "verify", lambda *a, **k: {"error": "pydicom 3.0.1 broke"})
    seen = {}
    def r(args, *, label, env):
        seen["R"] = env
        if case == "R fails": raise RuntimeError("child exit 7")
    def s(dirs, **kw):
        seen["S"] = dict(os.environ)
    monkeypatch.setattr(m.v5q, "run_process_group", r)
    monkeypatch.setattr(m.r7, "infer_studies", s)
    m.run_public(runtime, out)
    assert m.read_json(out / "R-exit.json")["exit_code"] == int(case == "R fails")
    assert m.read_json(out / "S_R7-exit.json")["exit_code"] == 0
    if case == "inactive":
        assert seen["R"]["PYTHONPATH"] == str(runtime) and codecs.BRANCH_ENV not in seen["R"]
        assert seen["S"]["PYTHONPATH"] == str(runtime) and codecs.BRANCH_ENV not in seen["S"]
        assert m.read_json(tmp_path / "run/codecs/gate-s_r7.json")["status"] == "SKIPPED_INACTIVE"
        return
    assert seen["R"]["PYTHONPATH"] == str(runtime) + os.pathsep + active["hook_dir"] and seen["R"][codecs.BRANCH_ENV] == "R"
    gate = m.read_json(tmp_path / "run/codecs/gate-s_r7.json")
    if case == "active gate fails":
        assert gate["status"].startswith("INACTIVE_")
        assert seen["S"]["PYTHONPATH"] == str(runtime) and codecs.BRANCH_ENV not in seen["S"] and codecs.TELEMETRY_ENV not in seen["S"]
    else:
        assert gate["status"] == "ACTIVE" and gate["pydicom"]
        assert seen["S"]["PYTHONPATH"] == str(runtime) + os.pathsep + active["hook_dir"] and seen["S"][codecs.BRANCH_ENV] == "S_R7"


def test_a172_label_header_relabels_phases_only():
    calls = []
    fake = types.ModuleType("exp169_release_c85")
    fake.phase = lambda output, name, action: calls.append((name, os.environ.get("KNEE_CODEC_BRANCH"), action()))
    saved = sys.modules.get("exp169_release_c85")
    sys.modules["exp169_release_c85"] = fake
    os.environ["KNEE_CODEC_BRANCH"] = "A172"
    try:
        exec(compile(m.A172_LABEL_HEADER, "<header>", "exec"), {})
        fake.phase("out", "own", lambda: os.environ["KNEE_CODEC_BRANCH"])
        fake.phase("out", "Q_A5q", lambda: os.environ["KNEE_CODEC_BRANCH"])
        after = os.environ["KNEE_CODEC_BRANCH"]
    finally:
        os.environ.pop("KNEE_CODEC_BRANCH", None)
        if saved is None: sys.modules.pop("exp169_release_c85")
        else: sys.modules["exp169_release_c85"] = saved
    assert calls == [("own", "A172:own", "A172:own"), ("Q_A5q", "A172:Q_A5q", "A172:Q_A5q")] and after == "A172"


def test_gate_with_real_verify_under_current_pydicom(tmp_path, active):
    rec = m.gate_s_r7(tmp_path / "run", active, tmp_path / "code")
    try:
        assert rec["status"] == "ACTIVE" and os.environ[codecs.BRANCH_ENV] == "S_R7"
    finally:
        os.environ.pop(codecs.BRANCH_ENV, None)


# ---------------------------------------------------------------- packaging and stages

def test_package_is_deterministic_superset_and_isolated_import_works(tmp_path):
    blob, pins = m.package()
    assert (blob, pins) == m.package() and len(pins) == 29
    old = v100.package()[1]
    assert {k: v for k, v in pins.items() if k in old} == old
    assert set(pins) - set(old) == {"exp169_release_c85r7codec.py", "hb126_codecs.py"}
    runtime = tmp_path / "runtime"
    m.p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    m.runtime_pins(runtime)
    result = subprocess.run([sys.executable, "-c",
        "import exp169_release_c85r7codec as c; c.runtime_pins(c.Path('.')); print(c.VERSION, c.base.VERSION, c.codecs.VERSION)"],
        cwd=runtime, capture_output=True, text=True, check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1"))
    assert result.stdout.strip() == "1.1.0 1.0.0 1.0.0"
    with (runtime / "hb126_codecs.py").open("ab") as f: f.write(b"\n")
    with pytest.raises(ValueError, match="SHA256"): m.runtime_pins(runtime)


def test_notebook_bootstrap_dispatch(tmp_path, monkeypatch):
    blob, pins = m.package()
    source = "".join(m.notebook_payload(blob, pins)["cells"][-1]["source"])
    runtime = tmp_path / "runtime"
    source = source.replace(f"Path({m.RUNTIME_DIR!r})", f"Path({str(runtime)!r})")
    source = source.replace("Path('/kaggle/working')", f"Path({str(tmp_path / 'output')!r})")
    events = []
    original = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "exp169_release_c85r7codec":
            def boundary(rt, out):
                m.runtime_pins(rt)
                assert rt == runtime and out == tmp_path / "output"
                events.append("dispatched")
            return types.SimpleNamespace(run_release=boundary)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    old_path = sys.path[:]
    try: exec(compile(source, "<test-bootstrap>", "exec"), {})
    finally: sys.path[:] = old_path
    assert events == ["dispatched"] and {name: m.sha(runtime / name) for name in pins} == pins


def test_stress_bootstrap_dispatches_full_release(tmp_path, monkeypatch):
    blob, pins = m.package()
    code = "".join(m.stress_notebook(blob, pins)["cells"][-1]["source"]).replace("/kaggle/working", tmp_path.as_posix())
    events = []
    def prepare(comp, out, n):
        assert n == 32 and comp == tmp_path / "competition"
        events.append("select"); return list(range(32))
    def run(rt, out):
        m.runtime_pins(rt)
        assert m.os.environ["C85_COMPETITION"] == str(tmp_path / "c85-stress-data") and out == tmp_path
        events.append("release")
    importer = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "exp169_release_c85r7codec": return types.SimpleNamespace(run_release=run, p=m.p,
            v5q=types.SimpleNamespace(_mount=lambda *a: tmp_path / "competition"))
        if name == "exp169_c85_stress": return types.SimpleNamespace(prepare=prepare)
        return importer(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("C85_COMPETITION", "")
    old_path = sys.path[:]
    try: exec(compile(code, "<stress bootstrap>", "exec"), {})
    finally: sys.path[:] = old_path
    assert events == ["select", "release"]


def test_all_cpu_cards_and_metadata(tmp_path):
    m.prepare(tmp_path)
    m.prepare(tmp_path)
    assert m.validate(tmp_path)["status"] == "PASS_CPU_VALIDATE"
    assert m.register(tmp_path)["status"] == "REGISTERED_LOCAL_ONLY"
    m.cpu_replay(tmp_path / "cpu-replay")
    result = m.admit(tmp_path)
    assert result["status"] == "HOLD_NOT_QUALIFIED" and result["cpu_preparation"] == "PASS"
    meta, stress = (m.read_json(tmp_path / f"{d}/kernel-metadata.json") for d in ("kernel", "stress-kernel"))
    old = m.read_json(v100.OUT / "kernel/kernel-metadata.json")
    assert meta["id"] == "taeyangg4/knee-exp169-c85r7codec-v1" and stress["id"] == meta["id"] + "-stress-n32"
    assert meta["dataset_sources"] == old["dataset_sources"] + ["taeyangg4/knee-hb126-codecs-cp313-v1"] == stress["dataset_sources"]
    for key in ("docker_image", "machine_shape", "is_private", "enable_internet", "enable_gpu", "competition_sources",
                "kernel_type", "language", "code_file"):
        assert meta[key] == old[key] == stress[key]
    ident = m.read_json(tmp_path / "input-identities.json")
    assert ident["codecs"]["wheel"]["sha256"] == codecs.WHEEL["sha256"] and ident["c85r7_v100"]["source_sha256"] == m.C85R7_SHA
    update_json(tmp_path / "kernel/kernel-metadata.json", lambda r: r.update(enable_internet=True))
    with pytest.raises(ValueError, match="SHA256"): m.validate(tmp_path)


def test_compare_csv(tmp_path):
    import pandas as pd
    m.cpu_replay(tmp_path / "r")
    ours = tmp_path / "r" / m.DIAGNOSTIC
    assert m.compare_csv(ours, ours)["status"] == "CSV_IDENTICAL"
    frame = pd.read_csv(ours).iloc[::-1]
    frame.iloc[0, 2] = .25
    frame.to_csv(tmp_path / "x.csv", index=False)
    r = m.compare_csv(ours, tmp_path / "x.csv")
    assert r["status"] == "CSV_DIFFERS" and r["cells_differing"] == 1
