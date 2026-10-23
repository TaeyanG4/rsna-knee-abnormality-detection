"""Strict wrapper admission/ownership tests; no CUDA or real jobs."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import l1_rco_k1_meta_admit_v1 as w


@pytest.fixture(autouse=True)
def restore_modules(monkeypatch):
    for k in (w.folds, w.all_data):
        for name in ("command", "admit", "require_infer_job"):
            monkeypatch.setattr(k, name, getattr(k, name))


@pytest.mark.parametrize("fold", [2, "all"])
def test_command_and_install_preserve_frozen_argv(fold):
    original = w.ORIGINAL_COMMANDS[fold](fold, review=w.review_path(fold))
    actual = w.command(fold)
    assert actual[:1] + actual[2:] == original[:1] + original[2:]
    assert actual[1] == str(w.SCRIPT)
    k = w.install(fold)
    assert k.command(fold) == actual
    assert k.command(fold, action="infer")[2] == "infer"


@pytest.mark.parametrize("fold,gpu", [(1, 0), (3, 0), (2, 1), ("all", 1)])
def test_no_other_fold_or_gpu(fold, gpu):
    with pytest.raises(ValueError):
        w.command(fold, gpu=gpu)


@pytest.mark.parametrize("fold", [2, "all"])
@pytest.mark.parametrize("failure", [None, "parity", "mode", "binding"])
def test_admission_gate_before_original(tmp_path, monkeypatch, fold, failure):
    calls = []
    def validate(cache):
        if failure == "parity":
            raise ValueError("px mismatch")
        return {"mode": "EXACT" if failure == "mode" else "META_EPS"}
    monkeypatch.setattr(w.parity, "validate_receipt", validate)
    monkeypatch.setattr(w, "evidence_binding", lambda _: {"sha": "bound"})
    monkeypatch.setitem(w.ORIGINAL_ADMITS, fold,
                        lambda *args: calls.append(args) or {"status": "CPU_ADMISSIBLE"})
    k = SimpleNamespace(WORK=tmp_path, read=lambda _: {
        "meta_eps_bindings": {"sha": "bad" if failure == "binding" else "bound"}},
        write=lambda p, r: None)
    result = w.admit(k, fold, {}, tmp_path, tmp_path/"REVIEW.json")
    assert result["status"] == ("BLOCKED" if failure else "CPU_ADMISSIBLE")
    assert len(calls) == (0 if failure else 1)
    if failure:
        with pytest.raises(ValueError):
            w.admit(k, fold, {}, tmp_path, tmp_path/"REVIEW.json", False, 123)
        assert not calls


def test_sequence_f2_only_and_all_binding(monkeypatch, tmp_path):
    monkeypatch.setattr(w.folds, "BASE", tmp_path)
    f2 = w.sequence_plan(2)
    assert [j["fold"] for j in f2["jobs"]] == [2]
    argv = f2["jobs"][0]["argv"]
    assert argv[argv.index("--") + 3] == "run"
    assert str(w.SCRIPT) in f2["jobs"][0]["argv"]
    all_spec = w.sequence_plan("all")
    assert [j["fold"] for j in all_spec["jobs"]] == ["all"]
    assert str(w.SCRIPT) in all_spec["jobs"][0]["argv"]
    # allseq's predecessor lookup must now bind the wrapper, not old f2 argv.
    observed = []
    monkeypatch.setattr(w.seq.runtime, "job_status",
                        lambda logs, tag, expected_command: observed.append(expected_command)
                        or {"state": "FAILED"})
    with pytest.raises(ValueError, match="SUCCESS"):
        w.allseq.preflight(all_spec)
    assert observed == [w.command(2)]


@pytest.mark.parametrize("mutation", [None, "parent", "child", "command", "cwd", "digest", "phase"])
def test_inference_owner_binding(tmp_path, monkeypatch, mutation):
    import psutil
    from contextlib import nullcontext
    from tools import job_runtime as rt
    expected = w.command(2)
    parent = SimpleNamespace(pid=456, cmdline=lambda: expected if mutation != "parent" else ["wrong"])
    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(parent=lambda: parent))
    monkeypatch.setattr(rt, "LOGS", tmp_path)
    monkeypatch.setattr(rt, "file_lock", lambda *_: nullcontext())
    monkeypatch.setattr(rt, "identity", lambda _: {"pid": 456})
    job = dict(schema="gpu-job/v2", tag="F2", phase="running",
               child_identity={"pid": 456}, command=expected, cwd=str(tmp_path),
               command_sha256=rt.command_digest(expected, tmp_path))
    if mutation and mutation != "parent":
        field = {"child": "child_identity", "digest": "command_sha256"}.get(mutation, mutation)
        job[field] = "wrong"
    monkeypatch.setattr(rt, "read_json", lambda _: job)
    k = SimpleNamespace(TAG="F2", need=w.folds.need, run_command=lambda *_: expected,
                        owner_scope=lambda: tmp_path.resolve())
    if mutation:
        with pytest.raises(ValueError):
            w.require_infer_job(k, None, None)
    else:
        w.require_infer_job(k, None, None)


@pytest.mark.parametrize("fold", [2, "all"])
def test_run_cannot_bypass_parity_admission(tmp_path, monkeypatch, fold):
    k = w.install(fold)
    monkeypatch.setattr(k, "WORK", tmp_path)
    monkeypatch.setattr(k, "OUT", tmp_path/"fit")
    monkeypatch.setattr(k, "require_gpu_job", lambda *_: 123)
    monkeypatch.setattr(k, "read", lambda _: {})
    monkeypatch.setattr(w.parity, "validate_receipt",
                        lambda *_: (_ for _ in ()).throw(ValueError("px mismatch")))
    monkeypatch.setattr(k, "supervise", lambda *_: pytest.fail("must not launch training"))
    with pytest.raises(ValueError, match="META_EPS admission blocked"):
        k.run(tmp_path, tmp_path/"REVIEW-META-EPS.json")


@pytest.mark.parametrize("fold_args,expected", [([], "2"), (["--fold", "2"], "2"),
                                               (["--fold=all"], "all")])
def test_cli_delegation_never_inherits_f1_default(monkeypatch, fold_args, expected):
    import argparse
    seen = []
    def delegated():
        parser = argparse.ArgumentParser()
        parser.add_argument("--fold", default="1")
        args, _ = parser.parse_known_args(w.sys.argv[1:])
        seen.append(args.fold)
    k = SimpleNamespace(configure=lambda *_: None, main=delegated)
    monkeypatch.setattr(w, "install", lambda _: k)
    monkeypatch.setattr(w.sys, "argv", ["meta-admit", "admit", "--dry-run"] + fold_args)
    w.main()
    assert seen == [expected]
