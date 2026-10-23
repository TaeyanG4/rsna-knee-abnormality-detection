"""Regression reproductions for astra-2 R1/R2; virtual minutes, real CPU I/O."""
import io
import json

import pytest
import torch

from tools import hb137_safety_v1_2_1 as s
from training.arms import mil_screen_hooks_v1_2_1 as h

STALL = dict(util=100, watts=70, used_mib=15900, total_mib=16376)


def test_healthy_five_second_progress_under_stall_telemetry_never_aborts():
    w = s.Watchdog(0)
    for now in range(0, 3601, 5):
        assert w.check(now, now + 1, STALL) is None
    assert w.check(4199, 3601, STALL) is None
    assert w.check(4200, 3601, STALL) == "WDDM_FALLBACK_STALL"


def test_true_stall_requires_ten_minutes_and_recovers_on_progress():
    w = s.Watchdog(0)
    assert w.check(0, 1, STALL) is None
    for now in (120, 599):
        assert w.check(now, 1, STALL) is None
    assert w.check(600, 1, STALL) == "WDDM_FALLBACK_STALL"
    assert w.check(605, 2, STALL) is None
    assert w.check(1204, 2, STALL) is None
    assert w.check(1205, 2, STALL) == "WDDM_FALLBACK_STALL"


def test_telemetry_recovery_restarts_continuous_interval():
    w = s.Watchdog(0, idle=1800)
    assert w.check(0, None, STALL) is None
    assert w.check(599, None, None) is None
    assert w.check(600, None, STALL) is None
    assert w.check(1199, None, STALL) is None
    assert w.check(1200, None, STALL) == "WDDM_FALLBACK_STALL"


@pytest.mark.parametrize("phase", ["training_step", "validation", "panel_inference", "checkpoint_write", "checkpoint_read"])
def test_any_phase_resets_idle_but_replayed_heartbeat_does_not(tmp_path, monkeypatch, phase):
    p = tmp_path / "progress.json"
    monkeypatch.setenv("HB137_PROGRESS", str(p))
    w = s.Watchdog(0)
    for now in range(0, 3601, 100):
        s.heartbeat(phase)
        note = json.loads(p.read_text())
        assert note["phase"] == phase
        assert w.check(now, note["progress_ns"]) is None
    assert w.check(4499, note["progress_ns"]) is None
    assert w.check(4500, note["progress_ns"]) == "NO_STEP_PROGRESS"


@pytest.mark.parametrize("operation", ["write", "read", "readinto"])
def test_long_checkpoint_io_reports_actual_progress_not_a_timer(monkeypatch, operation):
    clock, events, w = [0], [], s.Watchdog(0)

    class SlowFile(io.BytesIO):
        def write(self, data):
            clock[0] += 300
            return super().write(data)

        def read(self, size=-1):
            clock[0] += 300
            return super().read(size)

        def readinto(self, data):
            clock[0] += 300
            return super().readinto(data)

    def beat(phase, **details):
        events.append((phase, details["bytes_completed"]))
        assert w.check(clock[0], len(events), STALL) is None

    monkeypatch.setattr(s, "heartbeat", beat)
    stream = SlowFile() if operation == "write" else SlowFile(b"a" * 64)
    progress = s.ProgressIO(stream, "checkpoint_" + operation)
    progress.CHUNK = 8
    if operation == "write":
        assert progress.write(b"a" * 64) == 64
    elif operation == "read":
        assert progress.read(64) == b"a" * 64
    else:
        result = bytearray(64)
        assert progress.readinto(result) == 64
        assert result == b"a" * 64
    assert clock[0] == 2400 and len(events) == 8
    assert [n for _, n in events] == list(range(8, 65, 8))
    # Once the I/O stops, the same phase heartbeat cannot hide a true hang.
    assert w.check(2999, len(events), STALL) is None
    assert w.check(3000, len(events), STALL) == "WDDM_FALLBACK_STALL"
    assert w.check(3300, len(events)) == "NO_STEP_PROGRESS"


def test_blocked_write_does_not_emit_heartbeat(monkeypatch):
    events = []
    monkeypatch.setattr(s, "heartbeat", lambda *a, **k: events.append(a))

    class Blocked:
        def write(self, data):
            assert events == []
            raise TimeoutError("simulated blocked I/O")

    with pytest.raises(TimeoutError):
        s.ProgressIO(Blocked(), "checkpoint_write").write(b"data")
    assert events == []


def test_real_checkpoint_roundtrip_reports_write_read_verify(tmp_path, monkeypatch):
    phases = []
    monkeypatch.setattr(s, "heartbeat", lambda phase, **k: phases.append(phase))
    p = tmp_path / "checkpoint.pt"
    payload = {"weights": torch.arange(1024)}
    with p.open("xb") as stream:
        s.save_checkpoint(payload, stream)
    got = s.load_checkpoint(p, weights_only=False, map_location="cpu")
    assert torch.equal(got["weights"], payload["weights"])
    import hashlib
    assert s.checkpoint_sha(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    assert {"checkpoint_write", "checkpoint_read", "checkpoint_verify"} <= set(phases)


@pytest.mark.parametrize("phase", ["validation", "panel_inference"])
def test_identity_parent_evaluation_and_flip_emit_phase_progress(monkeypatch, phase):
    phases = []
    monkeypatch.setattr(h, "heartbeat", lambda phase, **k: phases.append((phase, k["part"])))
    monkeypatch.setattr(h, "flip_inputs", lambda x, *a: x)
    x = torch.ones(2, 12)
    class Loader(list):
        dataset = type("Dataset", (), {"k": 36})()
    loader = Loader([(x, x, ["a", "b"])])
    values, ids, count = h.evaluate_with(h.IDENTITY, torch.nn.Identity(), loader,
                                         "cpu", "none", None, tta=True, phase=phase)
    assert phases == [(phase, "forward"), (phase, "flip")]
    assert values.shape == (2, 12) and count == 1 and ids == ["a", "b"]


def test_training_progress_includes_completed_optimizer_step(monkeypatch):
    events = []
    monkeypatch.setattr(h, "heartbeat", lambda phase, **k: events.append((phase, k)))
    model = torch.nn.Linear(4, 12)
    opt, sched, scaler = h.optimizer_stack(model, 3, "cpu", "none")
    loader = [(torch.ones(2, 4), torch.zeros(2, 12), ["a", "b"])] * 3
    counters = dict(updates=0, skips=0)
    h.train_epoch_with(h.IDENTITY, model, loader, opt, sched, scaler, torch.ones(12),
                       "cpu", "none", 0, counters)
    assert [v["completed"] for phase, v in events if phase == "training_step"] == [1, 2, 3]


def test_total_cap_is_not_extended_by_healthy_phase_heartbeats(tmp_path):
    import sys
    code = "from tools.hb137_safety_v1_2_1 import heartbeat; import time\nwhile True:\n heartbeat('checkpoint_write'); time.sleep(.02)"
    ledger = tmp_path / "budget.json"
    with pytest.raises(h.life.ContractError, match="child failed"):
        s.run([sys.executable, "-B", "-c", code], ledger, limit=.7,
              idle=900, poll=.05, sample=lambda: STALL)
    assert json.loads(ledger.read_text())["attempts"][-1]["reason"] == "TOTAL_RUNTIME_CAP"


def test_versioned_hooks_change_only_progress_and_checkpoint_io():
    import ast
    import inspect
    from training.arms import mil_screen_hooks_v1 as old

    class StripProgress(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == "heartbeat":
                    return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node) == "if device == 'cuda':\n    torch.cuda.synchronize()":
                return None
            return self.generic_visit(node)

    for name in ("run", "train_epoch_with", "evaluate_with"):
        source = inspect.getsource(getattr(h, name))
        source = source.replace('save_checkpoint(', 'torch.save(').replace('load_checkpoint(', 'torch.load(')
        source = source.replace(', phase="validation"', '')
        got = StripProgress().visit(ast.parse(source))
        assert ast.dump(got) == ast.dump(ast.parse(inspect.getsource(getattr(old, name))))
