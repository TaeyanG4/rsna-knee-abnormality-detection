"""v2.0.0: queue dependency, command and failure tests; no training or GPU use."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("testable_gpu_queue", Path(__file__).resolve().parents[1] / "tools" / "gpu_queue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    def unexpected_sleep(seconds):
        pytest.fail("queue attempted to wait for a job that can never finish")

    monkeypatch.setattr(module.time, "sleep", unexpected_sleep)
    states, started = {}, []
    monkeypatch.setattr(module, "state", lambda tag: states.get(tag, "absent"))

    def run(command, **kwargs):
        tag = command[command.index("--tag") + 1]
        started.append(tag)
        states[tag] = "SUCCESS"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    return module, states, started


def plan_file(tmp_path, jobs):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(jobs), encoding="utf-8")
    return path


def job(tag, after=None, **kwargs):
    result = {"tag": tag, "command": ["python", "unused.py"], **kwargs}
    if after:
        result["after"] = after
    return result


def test_external_failure_stops_entire_chain_without_waiting_for_skipped_job(queue, tmp_path):
    module, states, started = queue
    states["A"] = "FAILED"
    path = plan_file(tmp_path, [job("B", "A"), job("C", "B"), job("D", "C")])
    assert module.run_plan(path) == 1
    assert started == []
    log = (module.LOGS / "queue" / "plan.log").read_text()
    assert "plan stopped before B" in log and "plan finished" not in log


@pytest.mark.parametrize("marker,code", [("FAILED", 1), ("absent", 1), ("absent", 0), ("running", 1)])
def test_launched_failure_or_missing_success_stops_before_next_wait(queue, tmp_path, monkeypatch, marker, code):
    module, states, started = queue

    def fail(command, **kwargs):
        tag = command[command.index("--tag") + 1]
        started.append(tag)
        states[tag] = marker
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(module.subprocess, "run", fail)
    path = plan_file(tmp_path, [job("A"), job("B", "A"), job("C", "B")])
    assert module.run_plan(path) == 1
    assert started == ["A"]


def test_success_chain_and_finished_jobs_are_not_relaunched(queue, tmp_path):
    module, states, started = queue
    states["A"] = "SUCCESS"
    path = plan_file(tmp_path, [job("A"), job("B", "A"), job("C", "B")])
    assert module.run_plan(path) == 0
    assert started == ["B", "C"]


def test_explicit_continuation_handles_launcher_failure_without_marker(queue, tmp_path, monkeypatch):
    module, states, started = queue

    def run(command, **kwargs):
        tag = command[command.index("--tag") + 1]
        started.append(tag)
        if tag == "A":
            return SimpleNamespace(returncode=2)
        states[tag] = "SUCCESS"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    path = plan_file(tmp_path, [job("A"), job("B", "A", continue_on_failure=True), job("C", "B")])
    assert module.run_plan(path) == 0
    assert started == ["A", "B", "C"]


def test_running_job_is_awaited_before_an_unlinked_next_job(queue, tmp_path, monkeypatch):
    module, states, started = queue
    states["A"] = "running"
    waited = []

    def wait(tag, poll=30):
        assert started == []
        waited.append(tag)
        states[tag] = "SUCCESS"
        return "SUCCESS"

    monkeypatch.setattr(module, "wait_for", wait)
    path = plan_file(tmp_path, [job("A"), job("B")])
    assert module.run_plan(path) == 0
    assert waited == ["A"] and started == ["B"]


def test_last_failure_returns_nonzero(queue, tmp_path, monkeypatch):
    module, states, _ = queue
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    assert module.run_plan(plan_file(tmp_path, [job("A")])) == 1


def test_failed_branch_does_not_block_independent_work(queue,tmp_path):
    module,states,started=queue
    states['A']='FAILED'
    assert module.run_plan(plan_file(tmp_path,[job('B','A'),job('C','B'),job('D')]))==1
    assert started==['D']


def test_out_of_order_dependencies(queue,tmp_path):
    module,states,started=queue
    assert module.run_plan(plan_file(tmp_path,[job('C','B'),job('B','A'),job('A')]))==0
    assert started==['A','B','C']


@pytest.mark.parametrize('jobs',[[job('A','A')],[job('A','B'),job('B','A')],[job('A'),job('A')],
    [job('../bad')],[job('A',continue_on_failure='true')]])
def test_invalid_plan_rejected_before_launch(queue,tmp_path,jobs):
    module,states,started=queue
    with pytest.raises(ValueError): module.run_plan(plan_file(tmp_path,jobs))
    assert not started


@pytest.mark.parametrize('value',['absent','stale','unknown','orphaned'])
def test_unsatisfiable_dependency_never_waits(queue,tmp_path,value):
    module,states,started=queue
    states['A']=value
    assert module.run_plan(plan_file(tmp_path,[job('B','A'),job('C')]))==1
    assert started==['C']


def test_completed_command_conflict(queue,tmp_path):
    module,states,started=queue
    path=module.LOGS/'A/job.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(command=['wrong'],pid=1)))
    states['A']='SUCCESS'
    assert module.run_plan(plan_file(tmp_path,[job('A')]))==1
    assert not started
