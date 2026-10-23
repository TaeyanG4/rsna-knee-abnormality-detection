"""R+ release delta gates, reusing the parent's real-output orchestration cases."""
import ast
import builtins
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from tools import exp169_release_p956 as parent
from tools import exp169_release_p956rplus as p
from tools import exp169_p956rplus_cpu as cpu
from tools import exp152_own0949_release as helper

spec = importlib.util.spec_from_file_location("p956_parent_cases", Path(__file__).with_name("test_exp169_release_p956.py"))
cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cases)
visible = cases.visible


def test_only_checkpoint_and_release_identity_change_serving():
    names = ("failure_masks", "blend", "public_formula", "csv_bytes", "validate_s6",
             "s6_failure_evidence", "anatomy_anchors", "commit", "r_runtime_source",
             "runtime_pins", "run_r", "run_release")
    for name in names:
        before = inspect.getsource(getattr(parent, name))
        after = inspect.getsource(getattr(p, name)).replace("exp169_release_p956rplus.py", "exp169_release_p956.py")
        assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(after)), name
    assert p.PINS == parent.PINS
    assert p.s6 is parent.s6
    assert p.DATASETS[1:] == parent.DATASETS[1:]
    assert p.R_SHA != parent.R_SHA
    namespace = {}
    p.bind_r(namespace, helper, p.R_SOURCE, cpu.COMP, cpu.SCRATCH)
    assert namespace["_E152"].checked_model is helper.checked_model
    assert namespace["ARMS"] == [dict(file=p.R_NAME, arch=helper.ARCH, res=384, w=1.0)]
    assert namespace["_E152_MANIFEST"]["checkpoints"] == {p.R_NAME: {"sha256": p.R_SHA}}


@pytest.mark.parametrize("fault", [None, "R failure", "S6 failure", "ID mismatch", "receipt failure", "GPU shape"])
def test_rplus_real_output_orchestration(tmp_path, visible, monkeypatch, fault):
    monkeypatch.setattr(cases, "p", p)
    cases.test_real_data_cpu_orchestration_smoke(tmp_path, visible, monkeypatch, fault)


@pytest.mark.parametrize("path", ["healthy", "R decode", "R inference", "S6 decode", "S6 gpu",
                                  "S6 loader", "S6 dead worker", "both", "at thresholds"])
def test_rplus_keeps_public_fallback(tmp_path, visible, monkeypatch, path):
    monkeypatch.setattr(cases, "p", p)
    cases.test_real_output_failure_paths_exact_public_formula_and_receipt(tmp_path, visible, path)


def test_private_dataset_kernel_and_embedded_import(tmp_path):
    assert p.validate()["status"] == "PASS_CPU_VALIDATE"
    meta = p.read_json(p.OUT / "dataset/dataset-metadata.json")
    assert meta["id"] == "taeyangg4/knee-hb113-rplus-weights-v1"
    assert meta["title"] == "knee-hb113-rplus-weights-v1" and meta["isPrivate"] is True
    kernel = p.read_json(p.OUT / "kernel/kernel-metadata.json")
    assert kernel["id"] == "taeyangg4/knee-exp169-p956rplus-v1" and kernel["is_private"] is True
    blob, pins = p.package()
    assert p.package() == (blob, pins)
    runtime = tmp_path / "runtime"
    p.unpack(blob, hashlib.sha256(blob).hexdigest(), runtime, pins)
    result = subprocess.run([sys.executable, "-B", "-c",
        "import exp169_release_p956rplus as p; p.runtime_pins(p.Path('.')); print(p.VERSION, p.R_SHA)"],
        cwd=runtime, capture_output=True, text=True, check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1"))
    assert result.stdout.strip() == "1.0.0 " + p.R_SHA


def test_bootstrap_real_outputs(tmp_path, monkeypatch, visible):
    notebook = p.read_json(p.OUT / "kernel/release.ipynb")
    source = "".join(notebook["cells"][-1]["source"])
    runtime, output = tmp_path / "runtime", tmp_path / "output"
    source = source.replace("Path('/kaggle/working/p956-runtime')", f"Path({str(runtime)!r})")
    source = source.replace("Path('/kaggle/working')", f"Path({str(output)!r})")
    original_import = builtins.__import__
    def import_module(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if name == "exp169_release_p956rplus":
            def replay(runtime_arg, output_arg):
                module.runtime_pins(runtime_arg)
                ids, r, ours, receipt = visible
                return module.commit(output_arg, ids, r, ours, receipt, {"scope": "bootstrap CPU replay"})
            module.run_release = replay
        return module
    monkeypatch.setattr(builtins, "__import__", import_module)
    monkeypatch.syspath_prepend(str(runtime))
    try:
        exec(compile(source, "<R+ bootstrap>", "exec"), {})
        assert (output / "submission.csv").is_file()
    finally:
        if str(runtime) in sys.path:
            sys.path.remove(str(runtime))
        for name in ("exp169_release_p956rplus", "exp169_release_v5q", "exp189_s6_port"):
            sys.modules.pop(name, None)


def test_windows_positioned_reads_preserve_binary_bytes(tmp_path):
    if hasattr(os, "pread"):
        pytest.skip("Windows-only smoke transport")
    module = cpu.cpu_reader(types.SimpleNamespace(_O_FLAGS=os.O_RDONLY))
    payload = bytes(range(256)) * 10
    path = tmp_path / "binary.dcm"
    path.write_bytes(payload)
    fd = os.open(path, module._O_FLAGS)
    try:
        assert module._pread(fd, 513, 5) == payload[5:518]
        assert module._pread(fd, 900, 0) == payload[:900]
    finally:
        os.close(fd)


def test_actual_two_study_strict_load_and_cpu_evidence():
    root = p.OUT / "cpu-replay"
    report = p.read_json(root / "replay.json")
    for name, digest in report["files"].items():
        p.check_pin(root / name, digest)
    assert report["strict_load"] and report["state_tensors"] == 486
    assert report["native_resolution"] == 320 and report["k_eval"] == 94
    assert report["tta"] == ["plain", "mirror"] and report["device"] == "cpu"
    assert report["r_sha256"] == parent.R_SHA and report["rplus_sha256"] == p.R_SHA
    ids, before = p.v5q._read_csv(root / "baseline-r.csv")
    new_ids, after = p.v5q._read_csv(root / "rplus.csv")
    sids, ours = p.v5q._read_csv(root / "rplus-s6.csv")
    assert ids == new_ids == sids and len(ids) == 2
    assert np.isfinite(before).all() and np.isfinite(after).all() and np.isfinite(ours).all()
    assert np.any(before != after, axis=1).all()
    assert (root / "baseline-s6.csv").read_bytes() == (root / "rplus-s6.csv").read_bytes()
    assert (root / "submission.csv").read_bytes() == p.csv_bytes(ids, parent.public_formula(after, ours))


def test_parent_release_bytes_untouched():
    before = p.read_json(cpu.SCRATCH / "parent-before.json")
    for name, digest in before.items():
        p.check_pin(p.ROOT / name, digest)
