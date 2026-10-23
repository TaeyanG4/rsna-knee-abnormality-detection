"""CPU-only orchestration regressions; never invoke gpu_job or CUDA."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import l1_rco_k1_foldseq_v1 as seq


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(seq.k, "BASE", tmp_path)
    spec = seq.plan()
    calls = []
    modes = {}
    # Dead, impossible process identities: exercise real job_status binding.
    def dispatch(argv, **kwargs):
        fold = (2, 3)[len(calls)]
        if fold == 3:
            ledger = seq.runtime.read_json(Path(spec["ledger"]))
            assert ledger["jobs"][0]["receipt"]["state"] == "SUCCESS"
        calls.append((argv, kwargs))
        assert argv == seq.k.launch_command(fold, gpu=0)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert kwargs["env"]["KNEE_WORKSPACE_ROOT"] == spec["owner"]
        assert kwargs["check"]
        if modes.get(fold) == "dispatch":
            raise OSError("dispatch failed")
        item = spec["jobs"][fold - 2]
        child = argv[argv.index("--") + 1:]
        job = dict(schema="gpu-job/v2", tag=item["tag"], run_id=f"new-{fold}",
                   command=child, cwd=spec["owner"],
                   command_sha256=seq.runtime.command_digest(child, spec["owner"]),
                   pid=2147483000, create_time=1., child_identity=None,
                   phase="running", started_epoch=seq.time.time())
        directory = Path(spec["owner"]) / "artifacts/logs" / item["tag"]
        seq.runtime.atomic_json(directory / "job.json", job)
        state = "FAILED" if modes.get(fold) == "failure" else "SUCCESS"
        receipt = dict(schema="gpu-job-receipt/v1", run_id=job["run_id"], tag=item["tag"],
                       command_sha256=job["command_sha256"], state=state,
                       exit_code=1 if state == "FAILED" else 0,
                       wrapper_identity={"pid": job["pid"], "create_time": job["create_time"]},
                       child_identity=None)
        mode = modes.get(fold)
        if mode in ("run_id", "command_sha256", "wrapper_identity", "child_identity", "schema"):
            receipt[mode] = "wrong"
        seq.runtime.atomic_json(directory / "receipt.json", receipt)
        (directory / state).write_text(f"exit {receipt['exit_code']} at 2026-10-20T12:00:00")
        if mode == "legacy":
            job["schema"] = "old"
            seq.runtime.atomic_json(directory / "job.json", job)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(seq.subprocess, "run", dispatch)
    return spec, calls, modes


def test_success_and_no_retry(harness):
    spec, calls, _ = harness
    result = seq.run_sequence(spec)
    assert result["state"] == "SUCCESS" and len(calls) == 2
    assert [j["gpu"] for j in result["jobs"]] == [0, 0]
    assert [j["run_id"] for j in result["jobs"]] == ["new-2", "new-3"]
    with pytest.raises(ValueError, match="no retry"):
        seq.run_sequence(spec)
    assert len(calls) == 2


@pytest.mark.parametrize("mode", [
    "failure", "dispatch", "run_id", "command_sha256",
    "wrapper_identity", "child_identity", "schema", "legacy",
])
def test_f2_failure_or_unbound_receipt_stops(harness, mode):
    spec, calls, modes = harness
    modes[2] = mode
    with pytest.raises((ValueError, OSError)):
        seq.run_sequence(spec)
    assert len(calls) == 1
    assert seq.runtime.read_json(Path(spec["ledger"]))["state"] == "STOPPED"
    with pytest.raises(ValueError, match="no retry"):
        seq.run_sequence(spec)
    assert len(calls) == 1


def test_f3_failure_stops_without_retry(harness):
    spec, calls, modes = harness
    modes[3] = "failure"
    with pytest.raises(ValueError, match="FAILED"):
        seq.run_sequence(spec)
    assert len(calls) == 2


@pytest.mark.parametrize("kind", ["job", "admission", "ledger"])
def test_existing_attempt_blocks_both_folds(harness, kind):
    spec, calls, _ = harness
    if kind == "job":
        path = Path(spec["owner"]) / "artifacts/logs" / spec["jobs"][1]["tag"] / "SUCCESS"
    elif kind == "admission":
        path = seq.k.BASE / "f3/fit/admission.json"
    else:
        path = Path(spec["ledger"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="no retry"):
        seq.run_sequence(spec)
    assert not calls


def test_wait_blocks_f3_and_checks_immutable_binding(harness, monkeypatch):
    spec, calls, _ = harness
    original = seq.runtime.job_status
    polls = []
    def status(logs, tag, **kwargs):
        if tag.endswith("F2") and not polls:
            polls.append(tag)
            return {"state": "running"}
        return original(logs, tag, **kwargs)
    monkeypatch.setattr(seq.runtime, "job_status", status)
    monkeypatch.setattr(seq.time, "sleep", lambda seconds: (
        pytest.fail("f3 launched before f2 completed") if len(calls) != 1 else None))
    assert seq.run_sequence(spec)["state"] == "SUCCESS"
    assert polls


def test_replaced_job_is_rejected(harness, monkeypatch):
    spec, calls, _ = harness
    original = seq.wait_success
    def replaced(logs, item, bound):
        wrong = copy.deepcopy(bound)
        wrong["run_id"] = "replacement"
        seq.runtime.atomic_json(logs / item["tag"] / "job.json", wrong)
        return original(logs, item, bound)
    monkeypatch.setattr(seq, "wait_success", replaced)
    with pytest.raises(ValueError, match="job binding changed"):
        seq.run_sequence(spec)
    assert len(calls) == 1


def test_dry_plan_does_not_dispatch(harness):
    spec, calls, _ = harness
    seq.check_fresh(spec)
    assert not calls and not Path(spec["ledger"]).exists()
    assert seq.command()[-3:] == ["launch-single", "--fold", "2"]
    assert [j["gpu"] for j in spec["jobs"]] == [0, 0]


def test_second_orchestrator_cannot_enter(harness):
    spec, calls, _ = harness
    with seq.runtime.file_lock(Path(spec["ledger"]).with_suffix(".lock"), blocking=False):
        with pytest.raises(BlockingIOError):
            seq.run_sequence(spec)
    assert not calls


def test_versioned_package_keeps_training_inputs():
    from tools import l1_rco_k1_foldpack_v1 as old
    from tools import l1_rco_k1_foldpack_v2 as new
    previous = set(old.files())
    current = set(new.files())
    assert previous <= current
    assert current - previous == {
        "tools/l1_rco_k1_foldseq_v1.py",
        "tools/l1_rco_k1_foldpack_v2.py",
        "tools/l1_rco_k1_folds_v2.md",
    }
    assert new.ARCHIVE.name == "h224-folds-v2.tar"
    assert new.PACKAGE != old.PACKAGE
