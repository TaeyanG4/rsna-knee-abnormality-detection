"""Offline f3 admission/sequencing tests; no CUDA or actual dispatch."""
import argparse
from pathlib import Path

import pytest

from tools import l1_rco_k1_meta_f3_v1 as w
from tools import l1_rco_k1_localwait_v1 as local
from tools.test_l1_rco_k1_device_plan_v1 import terminal


@pytest.fixture(autouse=True)
def restore(monkeypatch):
    for name in ("command", "admit", "require_infer_job", "BASE", "WORK", "OUT",
                 "FOLD", "GPU", "TAG", "REVIEW", "BEST", "TRAIN_STUDIES"):
        monkeypatch.setattr(w.k, name, getattr(w.k, name))
    monkeypatch.setenv("K1_FOLD", "3")
    monkeypatch.setenv("K1_GPU", "0")


def test_preserve_frozen_commands():
    before = {f: w.meta.command(f) for f in (2, "all")}
    original = w.meta.ORIGINAL_COMMANDS[2](3, review=w.review_path())
    spec = w.plan("all-run")
    argv = w.command()
    assert argv[:1] + argv[2:] == original[:1] + original[2:]
    assert argv[1] == str(w.SCRIPT)
    assert before == {f: w.meta.command(f) for f in (2, "all")}
    assert spec["predecessor"]["command"] == before["all"]
    assert [item["fold"] for item in spec["jobs"]] == [3]
    assert spec["owner"] == str(w.k.BASE / "gpu0-owner")


@pytest.mark.parametrize("fold,gpu", [(1, 0), (2, 0), ("all", 0), (3, 1)])
def test_reject_other_folds_and_gpu(fold, gpu):
    with pytest.raises(ValueError):
        w.command(fold, gpu=gpu)


@pytest.mark.parametrize("failure", [None, "parity", "mode", "binding", "old-review"])
def test_meta_admit_gate(tmp_path, monkeypatch, failure):
    calls = []
    def proof(cache):
        if failure == "parity":
            raise ValueError("px mismatch")
        return dict(mode="EXACT" if failure == "mode" else "META_EPS")
    monkeypatch.setattr(w.meta.parity, "validate_receipt", proof)
    monkeypatch.setattr(w, "evidence_binding", lambda _: dict(code="f3"))
    monkeypatch.setattr(w.k, "read", lambda _: dict(
        meta_eps_bindings=dict(code="old" if failure in ("binding", "old-review") else "f3")))
    monkeypatch.setattr(w.k, "WORK", tmp_path)
    monkeypatch.setitem(w.meta.ORIGINAL_ADMITS, 2,
                        lambda *args: calls.append(args) or dict(status="CPU_ADMISSIBLE"))
    result = w.admit({}, tmp_path, tmp_path / "REVIEW.json")
    assert result["status"] == ("BLOCKED" if failure else "CPU_ADMISSIBLE")
    assert len(calls) == (0 if failure else 1)
    if failure:
        with pytest.raises(ValueError, match="blocked"):
            w.admit({}, tmp_path, tmp_path / "REVIEW.json", False, 123)
        assert not calls


def test_binding_extends_original(monkeypatch):
    monkeypatch.setattr(w.meta, "evidence_binding",
                        lambda _: dict(parity="bound", code_pins={"old": "kept"}))
    monkeypatch.setattr(w.k, "sha", lambda p: p.name)
    proof = w.evidence_binding(Path("."))
    assert proof["code_pins"]["old"] == "kept"
    assert proof["code_pins"]["tools/l1_rco_k1_meta_f3_v1.py"] == w.SCRIPT.name
    assert "tools/gpu_job.py" in proof["code_pins"]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(w.k, "BASE", tmp_path / "folds")
    spec = w.plan("exact-all")
    pred = dict(tag=spec["predecessor"]["tag"],
                argv=["gpu_job.py", "--"] + spec["predecessor"]["command"])
    directory, job = terminal(spec, pred, "exact-all")
    monkeypatch.setattr(w.k, "validate", lambda r: r)
    w.k.write(w.k.WORK / "registration.json", {})
    monkeypatch.setattr(w, "admit", lambda *args, **kw: dict(status="CPU_ADMISSIBLE", blockers=[]))
    calls = []
    def dispatch(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["KNEE_WORKSPACE_ROOT"] == spec["owner"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        terminal(spec, spec["jobs"][0], "f3-run")
    monkeypatch.setattr(w.seq.subprocess, "run", dispatch)
    return spec, directory, job, calls


def test_success_one_dispatch_and_no_retry(rig):
    spec, _, _, calls = rig
    assert w.preflight(spec)["run_id"] == "exact-all"
    assert w.seq.run_sequence(spec)["state"] == "SUCCESS"
    assert len(calls) == 1
    with pytest.raises(ValueError, match="no retry"):
        w.preflight(spec)
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["run_id", "cwd", "command", "command_sha256", "schema", "tag"])
def test_wrong_predecessor(rig, field):
    spec, directory, job, calls = rig
    job[field] = ["wrong"] if field == "command" else "wrong"
    w.seq.runtime.atomic_json(directory / "job.json", job)
    with pytest.raises(ValueError, match="binding mismatch"):
        w.preflight(spec)
    assert not calls


@pytest.mark.parametrize("failure", ["absent", "stale", "FAILED", "receipt", "live", "legacy"])
def test_terminal_required(rig, monkeypatch, failure):
    spec, directory, job, calls = rig
    if failure == "absent":
        (directory / "job.json").unlink()
    elif failure in ("stale", "FAILED"):
        (directory / "SUCCESS").unlink()
        if failure == "FAILED":
            (directory / "FAILED").write_text("exit 1 at 2026-10-20T12:00:00")
    elif failure in ("receipt", "legacy"):
        receipt = w.k.read(directory / "receipt.json")
        receipt["run_id" if failure == "receipt" else "schema"] = "wrong"
        w.k.write(directory / "receipt.json", receipt)
    else:
        monkeypatch.setattr(w.seq.runtime, "liveness", lambda _: "alive")
        job["child_identity"] = dict(pid=123, create_time=1.)
        w.k.write(directory / "job.json", job)
    with pytest.raises((ValueError, OSError)):
        w.preflight(spec)
    assert not calls


def test_no_run_id(rig):
    spec, _, _, calls = rig
    spec["predecessor"]["run_id"] = None
    with pytest.raises(ValueError, match="explicit"):
        w.preflight(spec)
    assert not calls


@pytest.mark.parametrize("kind", ["ledger", "job", "admission", "occupied", "review"])
def test_target_blockers(rig, monkeypatch, kind):
    spec, _, _, calls = rig
    if kind == "occupied":
        monkeypatch.setattr(w.seq.runtime, "active_jobs", lambda _: [dict(tag="other")])
    elif kind == "review":
        monkeypatch.setattr(w, "admit", lambda *a, **kw: dict(status="BLOCKED", blockers=["review"]))
    else:
        path = dict(ledger=Path(spec["ledger"]),
                    job=Path(spec["owner"]) / "artifacts/logs/H224_K1_V3G_F3/job.json",
                    admission=w.k.OUT / "admission.json")[kind]
        w.k.write(path, {})
    with pytest.raises(ValueError):
        w.preflight(spec)
    assert not calls


def test_dispatch_lock(rig):
    spec, _, _, calls = rig
    with w.seq.runtime.file_lock(Path(spec["ledger"]).with_suffix(".lock"), blocking=False):
        with pytest.raises(BlockingIOError):
            w.seq.run_sequence(spec)
    assert not calls


def test_admission_cannot_be_bypassed(tmp_path, monkeypatch):
    w.install()
    monkeypatch.setattr(w.k, "OUT", tmp_path / "fit")
    monkeypatch.setattr(w.k, "WORK", tmp_path)
    monkeypatch.setattr(w.k, "require_gpu_job", lambda *_: 123)
    monkeypatch.setattr(w.k, "read", lambda _: {})
    monkeypatch.setattr(w.meta.parity, "validate_receipt",
                        lambda _: (_ for _ in ()).throw(ValueError("invalid parity")))
    monkeypatch.setattr(w.k, "supervise", lambda *_: pytest.fail("training must not run"))
    with pytest.raises(ValueError, match="admission blocked"):
        w.k.run(tmp_path, tmp_path / "REVIEW.json")


@pytest.mark.parametrize("args", [[], ["--fold=3"]])
def test_cli_defaults_f3(monkeypatch, args):
    seen = []
    def delegated():
        ap = argparse.ArgumentParser()
        ap.add_argument("--fold")
        ap.add_argument("--review")
        parsed, _ = ap.parse_known_args(w.sys.argv[1:])
        seen.append((parsed.fold, parsed.review))
    monkeypatch.setattr(w.k, "main", delegated)
    monkeypatch.setattr(w.sys, "argv", ["f3", "admit", "--dry-run"] + args)
    w.main()
    assert seen == [("3", str(w.review_path()))]


def test_cli_dry_run_does_not_launch(rig, monkeypatch, capsys):
    _, _, _, calls = rig
    monkeypatch.setattr(w.sys, "argv",
                        ["f3", "launch-after-all", "--all-run-id", "exact-all", "--dry-run"])
    w.main()
    assert '"status": "PLAN_ONLY"' in capsys.readouterr().out
    assert not calls


def test_local_wait_interruption_never_dispatches(tmp_path, monkeypatch):
    # Exercise the unchanged local waiter; no live state/process is accessed.
    monkeypatch.setattr(w.k, "ROOT", tmp_path)
    monkeypatch.setattr(w.k, "BASE", tmp_path / "folds")
    monkeypatch.setattr(local, "TASK", tmp_path / "local-waiter")
    spec = local.plan("f1-run")
    terminal(spec, spec["predecessor"], "f1-run")
    monkeypatch.setattr(local, "preflight_review", lambda: None)
    monkeypatch.setattr(w.seq, "wait_success",
                        lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(w.seq, "run_sequence", lambda _: pytest.fail("must not dispatch"))
    with pytest.raises(KeyboardInterrupt):
        local.run(spec)
    assert w.k.read(Path(spec["waiter_ledger"]))["state"] == "STOPPED"
    assert not Path(spec["ledger"]).exists()
