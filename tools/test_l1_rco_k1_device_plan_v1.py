"""Offline orchestration tests: genuine-shaped bound receipts, no GPU dispatch."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import l1_rco_k1_foldseq_v1 as seq
from tools import l1_rco_k1_localwait_v1 as local


def terminal(spec, item, run_id, state="SUCCESS"):
    child = item["argv"][item["argv"].index("--") + 1:]
    job = dict(schema="gpu-job/v2", tag=item["tag"], run_id=run_id,
               command=child, cwd=spec["owner"],
               command_sha256=seq.runtime.command_digest(child, spec["owner"]),
               pid=2147483000, create_time=1., child_identity=None,
               phase="running", started_epoch=seq.time.time())
    directory = Path(spec["owner"]) / "artifacts/logs" / item["tag"]
    seq.runtime.atomic_json(directory / "job.json", job)
    receipt = dict(schema="gpu-job-receipt/v1", tag=item["tag"], run_id=run_id,
                   command_sha256=job["command_sha256"], state=state,
                   exit_code=0 if state == "SUCCESS" else 1,
                   wrapper_identity=dict(pid=job["pid"], create_time=job["create_time"]),
                   child_identity=None)
    seq.runtime.atomic_json(directory / "receipt.json", receipt)
    (directory / state).write_text(
        f"exit {receipt['exit_code']} at 2026-10-20T12:00:00")
    return directory, job


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(seq.k, "BASE", tmp_path / "folds")
    monkeypatch.setattr(seq.k, "ROOT", tmp_path / "project")
    monkeypatch.setattr(local, "TASK", tmp_path / "waiter")
    spec = local.plan("exact-f1")
    directory, job = terminal(spec, spec["predecessor"], "exact-f1")
    reviews = []
    monkeypatch.setattr(local, "preflight_review", lambda: reviews.append(True))
    calls = []

    def dispatch(argv, **kwargs):
        calls.append(argv)
        assert argv == spec["jobs"][0]["argv"]
        assert kwargs["env"]["KNEE_WORKSPACE_ROOT"] == spec["owner"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        terminal(spec, spec["jobs"][0], "exact-f3")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(seq.subprocess, "run", dispatch)
    return spec, directory, job, calls, reviews


def test_local_success_one_dispatch_and_no_retry(rig):
    spec, _, _, calls, reviews = rig
    assert local.run(spec)["state"] == "SUCCESS"
    assert len(calls) == 1 and len(reviews) == 2
    with pytest.raises(ValueError, match="no retry"):
        local.run(spec)
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["run_id", "cwd", "command_sha256", "schema", "command"])
def test_predecessor_binding(rig, field):
    spec, directory, job, calls, _ = rig
    job[field] = ["wrong"] if field == "command" else "wrong"
    seq.runtime.atomic_json(directory / "job.json", job)
    with pytest.raises(ValueError, match="binding mismatch"):
        local.run(spec)
    assert not calls


@pytest.mark.parametrize("mode", ["FAILED", "stale", "legacy", "receipt", "replaced"])
def test_predecessor_not_success(rig, monkeypatch, mode):
    spec, directory, job, calls, _ = rig
    if mode in ("FAILED", "stale"):
        (directory / "SUCCESS").unlink()
        if mode == "FAILED":
            terminal(spec, spec["predecessor"], "exact-f1", state="FAILED")
    elif mode == "legacy":
        receipt = seq.runtime.read_json(directory / "receipt.json")
        receipt["schema"] = "legacy"
        seq.runtime.atomic_json(directory / "receipt.json", receipt)
    elif mode == "receipt":
        receipt = seq.runtime.read_json(directory / "receipt.json")
        receipt["run_id"] = "wrong"
        seq.runtime.atomic_json(directory / "receipt.json", receipt)
    else:
        original = seq.wait_success
        def changed(logs, item, bound):
            changed_job = dict(bound, run_id="replaced")
            seq.runtime.atomic_json(directory / "job.json", changed_job)
            return original(logs, item, bound)
        monkeypatch.setattr(seq, "wait_success", changed)
    with pytest.raises(ValueError):
        local.run(spec)
    assert not calls
    assert seq.runtime.read_json(Path(spec["waiter_ledger"]))["state"] == "STOPPED"


def test_wait_does_not_dispatch_early(rig, monkeypatch):
    spec, _, _, calls, _ = rig
    original = seq.runtime.job_status
    polls = []
    def status(logs, tag, **kwargs):
        if tag.endswith("F1") and not polls:
            polls.append(tag)
            return dict(state="running")
        return original(logs, tag, **kwargs)
    monkeypatch.setattr(seq.runtime, "job_status", status)
    def sleep(seconds):
        assert not calls
    monkeypatch.setattr(seq.time, "sleep", sleep)
    assert local.run(spec)["state"] == "SUCCESS"
    assert len(calls) == 1


def test_review_blocks_before_wait(rig, monkeypatch):
    spec, _, _, calls, _ = rig
    def blocked():
        raise ValueError("independent review missing")
    monkeypatch.setattr(local, "preflight_review", blocked)
    with pytest.raises(ValueError, match="review"):
        local.run(spec)
    assert not calls


@pytest.mark.parametrize("kind", ["job", "admission", "dispatch", "waiter"])
def test_consumed_f3_blocks(rig, kind):
    spec, _, _, calls, _ = rig
    paths = dict(job=Path(spec["owner"]) / "artifacts/logs/H224_K1_V3G_F3/job.json",
                 admission=seq.k.BASE / "f3/fit/admission.json",
                 dispatch=Path(spec["ledger"]), waiter=Path(spec["waiter_ledger"]))
    seq.runtime.atomic_json(paths[kind], {})
    with pytest.raises(ValueError, match="no retry"):
        local.run(spec)
    assert not calls


def test_occupied_gpu_stops_after_success(rig, monkeypatch):
    spec, _, _, calls, _ = rig
    monkeypatch.setattr(seq.runtime, "active_jobs", lambda logs: [dict(tag="other")])
    with pytest.raises(ValueError, match="occupied"):
        local.run(spec)
    assert not calls


def test_waiter_lock_blocks_duplicate(rig):
    spec, _, _, calls, _ = rig
    with seq.runtime.file_lock(Path(spec["waiter_ledger"]).with_suffix(".lock"), blocking=False):
        with pytest.raises(BlockingIOError):
            local.run(spec)
    assert not calls


def test_dry_run_never_reads_predecessor_or_dispatches(monkeypatch, capsys):
    monkeypatch.setattr(local.sys, "argv", ["waiter", "launch-after-f1", "--dry-run"])
    def denied(*args, **kwargs):
        pytest.fail("dry-run must not inspect predecessor or launch")
    monkeypatch.setattr(seq.runtime, "read_json", denied)
    monkeypatch.setattr(seq.runtime, "active_jobs", denied)
    monkeypatch.setattr(seq.subprocess, "run", denied)
    local.main()
    assert '"predecessor_checked": false' in capsys.readouterr().out


def test_single_vast_f2_only(tmp_path, monkeypatch):
    monkeypatch.setattr(seq.k, "BASE", tmp_path)
    spec = seq.single_plan(2)
    assert len(spec["jobs"]) == 1 and spec["jobs"][0]["fold"] == 2
    calls = []
    def dispatch(argv, **kwargs):
        calls.append(argv)
        terminal(spec, spec["jobs"][0], "f2")
    monkeypatch.setattr(seq.subprocess, "run", dispatch)
    assert seq.run_sequence(spec)["state"] == "SUCCESS"
    assert len(calls) == 1
    with pytest.raises(ValueError, match="no retry"):
        seq.run_sequence(spec)
    with pytest.raises(ValueError, match="only fold 2"):
        seq.single_plan(3)


def test_old_vast_cli_is_retired(monkeypatch):
    monkeypatch.setattr(seq.sys, "argv", ["seq", "launch-sequential"])
    with pytest.raises(ValueError, match="retired"):
        seq.main()


def gate_metrics():
    from tools import l1_rco_k1_foldgate_v1 as gate
    return {f: {model: {r: value for r in gate.RULERS}
                for model, value in (("k1", .90), ("r7", .89))} for f in range(4)}


def test_falsifier_accepts_exact_boundary_and_three_folds():
    from tools import l1_rco_k1_foldgate_v1 as gate
    metrics = gate_metrics()
    for r in gate.RULERS:
        metrics[1]["k1"][r] = .895
        metrics[0]["r7"][r] = .91
    assert gate.decide(metrics)["status"] == "SCREEN_PASS_PENDING_REVIEW"


@pytest.mark.parametrize("mode", ["low", "high", "two_losses", "nan", "missing_ruler", "missing_fold"])
def test_falsifier_blocks(mode):
    from tools import l1_rco_k1_foldgate_v1 as gate
    metrics = gate_metrics()
    ruler = gate.RULERS[0]
    if mode in ("low", "high"):
        metrics[2]["k1"][ruler] = .8949 if mode == "low" else .9051
    elif mode == "two_losses":
        for fold in (1, 3):
            metrics[fold]["r7"][ruler] = .91
    elif mode == "nan":
        metrics[2]["k1"][ruler] = float("nan")
    elif mode == "missing_ruler":
        del metrics[2]["r7"][ruler]
    else:
        del metrics[3]
    if mode in ("nan", "missing_ruler", "missing_fold"):
        with pytest.raises(ValueError):
            gate.decide(metrics)
    else:
        assert gate.decide(metrics)["status"] == "HOLD"


def test_gate_end_to_end_and_provenance(tmp_path, monkeypatch):
    import csv
    import numpy as np
    from tools import l1_rco_k1_foldgate_v1 as gate
    from tools import exp156_k_sweep as rulers
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "rosters", lambda: {f: ["a", "b"] for f in range(4)})
    monkeypatch.setattr(gate.k, "PARENT", tmp_path / "parent.json")
    gate.k.write(gate.k.PARENT, dict(contract_id="synthetic"))
    monkeypatch.setattr(gate.k, "configure", lambda f: None)
    monkeypatch.setattr(gate.k, "expected_registration", lambda: dict(contract_id="synthetic"))
    target = np.array([[0] * 12, [1] * 12])
    monkeypatch.setattr(rulers, "references", lambda *a, **kw: {
        r: (target, np.ones_like(target, dtype=bool), 1) for r in gate.RULERS})
    manifest = dict(schema="h224-own-holdout-input/v1", folds={})
    for fold in range(4):
        path = tmp_path / f"synthetic-f{fold}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([gate.k.s.UID, *gate.k.s.LABELS, "status", "model_fold"])
            writer.writerow(["a", *([.2] * 12), "ok", fold])
            writer.writerow(["b", *([.8] * 12), "ok", fold])
        result = tmp_path / f"k1-f{fold}.json"
        gate.k.write(result, dict(status="COMPLETE", cpu_test=False, model_fold=fold,
                                 holdout_rows=2, holdout_sha256=gate.k.sha(path),
                                 contract_id="synthetic"))
        receipt = tmp_path / f"r7-f{fold}.json"
        gate.k.write(receipt, dict(complete=True, mode=f"screen_f{fold}",
                                  model_selection=f"f{fold}_only", output_kind="pre_stacker",
                                  clean_oof=False, csv_sha256=gate.k.sha(path)))
        manifest["folds"][str(fold)] = dict(k1_csv=path.name, r7_csv=path.name,
                                          k1_result=result.name, r7_receipt=receipt.name)
    assert gate.evaluate(manifest)["status"] == "SCREEN_PASS_PENDING_REVIEW"
    receipt = tmp_path / "r7-f1.json"
    wrong = gate.k.read(receipt)
    wrong["model_selection"] = "all_folds"
    gate.k.write(receipt, wrong)
    with pytest.raises(ValueError, match="same-fold"):
        gate.evaluate(manifest)


@pytest.mark.parametrize("mode", ["missing", "wrong_fold", "bad_status", "nonfinite"])
def test_prediction_roster_is_fail_closed(tmp_path, mode):
    import csv
    from tools import l1_rco_k1_foldgate_v1 as gate
    path = tmp_path / "synthetic.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([gate.k.s.UID, *gate.k.s.LABELS, "status", "model_fold"])
        writer.writerow(["a", *([float("nan") if mode == "nonfinite" else .5] * 12),
                         "bad" if mode == "bad_status" else "ok", 2 if mode == "wrong_fold" else 1])
    with pytest.raises(ValueError):
        gate.predictions(path, ["a", "b"] if mode == "missing" else ["a"], 1)


def test_package_v4_adds_only_device_tools():
    from tools import l1_rco_k1_foldpack_v3 as old
    from tools import l1_rco_k1_foldpack_v4 as new
    assert set(new.files()) - set(old.files()) == {
        "tools/l1_rco_k1_foldpack_v4.py", "tools/l1_rco_k1_localwait_v1.py",
        "tools/l1_rco_k1_foldgate_v1.py", "tools/test_l1_rco_k1_device_plan_v1.py",
        "tools/l1_rco_k1_device_plan_v1.md",
        "tools/test_l1_rco_k1_foldseq_v1.py",
        "tools/exp156_k_sweep.py", "tools/exp115_vlm_eval.py", "tools/exp115_vlm_pilot.py",
    }
    assert new.ARCHIVE != old.ARCHIVE


def test_package_preserves_v3_config_instead_of_concurrent_changes(tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile
    from tools import l1_rco_k1_foldpack_v4 as pack
    name = "tools/k1n_config_v1.py"
    data = b"synthetic archived dependency\n"
    archive = tmp_path / "previous.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo(pack.previous.PREFIX + "/" + name)
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    monkeypatch.setattr(pack.previous, "ARCHIVE", archive)
    monkeypatch.setattr(pack, "PRESERVED_V3", {name: hashlib.sha256(data).hexdigest()})
    assert pack.preserved_bytes(name) == data
    monkeypatch.setattr(pack, "PRESERVED_V3", {name: "0" * 64})
    with pytest.raises(ValueError, match="dependency drift"):
        pack.preserved_bytes(name)
