"""v1.2.5 scheduling amendment: CPU engineering contracts only."""
import ast
import json
from pathlib import Path

import pytest

from tools import hb149_resume_v125 as r
from tools import hb149_monitor_v125 as m


def test_runtime_math_unchanged():
    source = (r.SOURCE / "stage/r7_runtime/compact_infer.py").read_text()
    changed = r.patch_runtime(source)
    def node(code, name):
        return ast.dump(next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef) and n.name == name))
    for name in ("mirror_windows", "mirror_perm", "score_v4c", "score", "one_pass", "_forward", "_prepare", "apply_stacker", "apply_state_stacker"):
        assert node(source, name) == node(changed, name), name
    assert '.to("cpu").to(memory_format=torch.channels_last)' in changed
    assert 'hb145_models[k].to(dev)' in changed
    assert 'dev = self.device' in changed and 'sdev = self.device' in changed
    with pytest.raises(ValueError):
        r.patch_runtime(changed)


def test_residency_sequence_keeps_values():
    import torch
    models = [torch.nn.Linear(3, 2).eval() for _ in range(4)]
    before = [{k: v.clone() for k, v in model.state_dict().items()} for model in models]
    x = torch.arange(3).float()
    results = [model(x).detach().clone() for model in models]
    for k in (1, 2, 3, 2, 1):
        models[k].to("cpu")
        assert torch.equal(models[k](x), results[k])
        assert all(torch.equal(v, before[k][n]) for n, v in models[k].state_dict().items())


def test_rolling_collapse_not_reset_by_slow_commits():
    guard = m.RateGuard(8, 0)
    for t in range(60, 1800, 60):
        assert guard.update(t, t // 60) is None
    assert "throughput" in guard.update(1800, 30)


def test_rate_recovery_and_boundary():
    guard = m.RateGuard(8, 0)
    for t in range(60, 1860, 60):
        assert guard.update(t, t // 30) is None  # Exactly 25% is allowed.
    assert guard.snapshot()["observed_studies_per_minute"] == 2
    guard = m.RateGuard(8, 0)
    for t in range(60, 3601, 60):
        assert guard.update(t, t // 10) is None
    assert guard.snapshot()["observed_studies_per_minute"] == 6
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.RateGuard(value, 0)


def test_carried_hashes_and_no_recompute(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    ident = {"uids": []}
    r.io.atomic_json(source / "identity.json", ident)
    (source / "journal").mkdir()
    (source / "journal/x.json").write_text('{"synthetic": true}\n')
    reg = {"journals": {"x.json": r.sha(source / "journal/x.json")}, "execution": {"workers": 1},
           "executed_runtime_sha256": "runtime"}
    registry = tmp_path / "registry.json"
    r.io.atomic_json(registry, reg)
    monkeypatch.setattr(r, "SOURCE", source)
    monkeypatch.setattr(r, "REG", registry)
    dest = tmp_path / "dest"
    r.carry(dest, reg, ident)
    (dest / "journal/extra.json").write_text('{}')
    r.carry(dest, reg, ident)
    r.io.atomic_json(registry, {**reg, "changed": True})
    with pytest.raises(ValueError, match="original execution registration"):
        r.carry(dest, reg, ident)
    r.io.atomic_json(registry, reg)
    assert (dest / "journal/x.json").read_bytes() == (source / "journal/x.json").read_bytes()
    r.carry(dest, reg, ident)
    (dest / "journal/x.json").write_text("changed")
    with pytest.raises(ValueError, match="carried journal changed"):
        r.carry(dest, reg, ident)


def test_review_and_smoke_bypass_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "SMOKE", tmp_path)
    regpath = tmp_path / "reg.json"
    r.io.atomic_json(regpath, {"version": "1.2.5"})
    monkeypatch.setattr(r, "REG", regpath)
    proof = {"passed": False}
    r.io.atomic_json(tmp_path / "admission-v125.json", proof)
    monkeypatch.setattr(r, "smoke_result", lambda reg: proof)
    with pytest.raises(ValueError, match="admission failed"):
        r.admit({}, tmp_path / "review.json")
    proof = {"passed": True}
    r.io.atomic_json(tmp_path / "admission-v125.json", proof)
    review = {"verdict": "CLOSED", "independent": True, "reviewer": "B",
              "builder": "Codex/astra-b5", "registration_sha256": r.sha(regpath),
              "smoke_sha256": r.sha(tmp_path / "admission-v125.json")}
    r.io.atomic_json(tmp_path / "review.json", review)
    assert r.admit({}, tmp_path / "review.json")["passed"]
    for key, value in (("verdict", "HOLD"), ("independent", False), ("smoke_sha256", "stale"),
                       ("registration_sha256", "stale"), ("reviewer", "Codex/astra-b5")):
        r.io.atomic_json(tmp_path / "review.json", {**review, key: value})
        with pytest.raises(ValueError, match="independent review"):
            r.admit({}, tmp_path / "review.json")


def test_persistent_journal_failure_receipt(tmp_path, monkeypatch):
    real = r.io.atomic_json
    def locked(path, value):
        if Path(path).name == "record.json":
            raise PermissionError("locked")
        real(path, value)
    monkeypatch.setattr(r.io, "atomic_json", locked)
    with pytest.raises(r.io.JournalCommitError):
        r.io.journal(tmp_path / "record.json", {"identity": "test"}, tmp_path)
    assert r.read(tmp_path / "journal_failure_receipt.json")["committed"] is False


@pytest.mark.parametrize("fault", [None, "parity", "main_memory", "total_memory", "telemetry",
                                   "cleanup", "code_binding", "manifest_binding"])
def test_smoke_memory_parity_and_execution_binding(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(r, "SMOKE", tmp_path)
    monkeypatch.setattr(r, "REG", tmp_path / "reg.json")
    runtime = tmp_path / "stage/r7_runtime"
    runtime.mkdir(parents=True)
    (runtime / "compact_infer.py").write_text("# synthetic runtime")
    digest = r.sha(runtime / "compact_infer.py")
    reg = {"smoke_uids": list(range(5)), "executed_runtime_sha256": digest,
           "main_reserved_cap_bytes": 10e9, "total_gpu_cap_bytes": 11e9}
    r.io.atomic_json(r.REG, reg)
    r.io.atomic_json(runtime / "runtime_sha256.json", {"compact_infer.py": digest})
    r.io.atomic_json(tmp_path / "execution-runtime-v125.json", {
        "version": r.VERSION, "registration_sha256": "stale" if fault == "code_binding" else r.sha(r.REG),
        "runtime_sha256": digest, "engineering_only": True,
        "runtime_manifest_sha256": "stale" if fault == "manifest_binding" else r.sha(runtime / "runtime_sha256.json")})
    receipt = {"steady_seconds_per_study": 10, "peak_main_reserved_bytes": 10e9 + (fault == "main_memory"),
               "exit_code": 0, "failures": 0, "codec_coverage_complete": True}
    resources = {"peak_total_gpu_mib": 11000 if fault == "total_memory" else 9000,
                 "baseline_gpu_mib": 1000, "telemetry_errors": ["error"] if fault == "telemetry" else [],
                 "cleanup_complete": fault != "cleanup", "state": "COMPLETED"}
    r.io.atomic_json(tmp_path / "run_receipt.json", receipt)
    r.io.atomic_json(tmp_path / "run_resources.json", resources)
    def records(path, uids, pin):
        return {u: {"predictions": [float(fault == "parity" and path == tmp_path)]} for u in uids}
    monkeypatch.setattr(r.h, "load_done", records)
    if fault in ("code_binding", "manifest_binding"):
        with pytest.raises(ValueError, match="binding drift"):
            r.smoke_result(reg)
    else:
        assert r.smoke_result(reg)["passed"] == (fault is None)


def test_monitor_cpu_real_child_cleanup(tmp_path):
    import os
    import sys
    code, receipt = m.monitored_run([sys.executable, "-c", "print('cpu-child')"],
        dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), tmp_path / "child.log", tmp_path / "resource.json",
        poll_seconds=.05, healthy_rate=5)
    assert code == 0 and receipt["cleanup_complete"]
    assert receipt["throughput_guard"]["healthy_studies_per_minute"] == 5
