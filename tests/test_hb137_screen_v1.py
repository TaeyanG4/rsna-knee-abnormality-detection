"""Synthetic scoring, command ordering and CPU-only watchdog checks for H137."""
from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from tools import hb137_screen_v1 as s


def synthetic():
    rng = np.random.default_rng(11)
    n = 80
    ids = [f"synthetic-{i}" for i in range(n)]
    y = (np.arange(n)[:, None] + np.arange(12)[None, :]) % 2
    refs = {r: (y, np.ones_like(y, bool), 1) for r in s.screen.ml.RULERS}
    ctx = s.screen.ml.Context(ids, refs, [dict(component_id=u, members=[u]) for u in ids])
    parents = {seed: rng.uniform(.1, .9, (n, 12)).astype(np.float32) for seed in ("42", "43")}
    arrays = {kind: deepcopy(parents) for kind in ("parent", "head", "adapter")}
    for seed in parents:
        for kind, gain in (("head", .05), ("adapter", .4)):
            arrays[kind][seed][:, s.spatial.WEAK] = np.clip(
                parents[seed][:, s.spatial.WEAK] + gain * (2 * y[:, s.spatial.WEAK] - 1), 0, 1)
    outcomes = y.astype(float)
    outcomes[:, [1, 10, 11]] = np.nan
    return ctx, arrays, ids, outcomes


def test_three_contrasts_seven_lenses_and_paired_mapped_macro_ci():
    ctx, arrays, ids, y = synthetic()
    result = s.three_way(ctx, arrays, arrays, ids, y,
        lambda a, b: (np.mean(list(a.values()), axis=0), np.mean(list(b.values()), axis=0)))
    assert set(result["contrasts"]) == {"head-parent", "adapter-parent", "adapter-head"}
    assert result["decision"]["diagnostic"] == "NECESSARY_CHECKS_PASS"
    assert result["decision"]["admission"] == "HOLD"
    for key, r in result["contrasts"].items():
        assert set(r["lenses"]) == {f"L{i}" for i in range(1, 8)}
        assert r["version"] == "2.0.0"
        assert r["bootstrap"]["draws"] == 300
        assert r["lenses"]["L4"]["targets"][s.hooks.TARGETS[3]]["rulers_ge_0_003"] == 4
        a, b = key.split("-")
        item = s.screen.image_item(ids, arrays[a], arrays[b], y)
        w = s.screen.ml.weights(ids, item["components"])
        delta = s.screen.ml.auc_draws(item["arm"], item["labels"], item["mask"], w) - s.screen.ml.auc_draws(
            item["control"], item["labels"], item["mask"], w)
        expected = s.screen.ml.stat(delta[:, s.screen.PANEL_TARGETS].mean(1))
        macro = r["lenses"]["L7"]["entries"]["oai_moaks"]["overall_macro"]
        assert macro["value"] == expected["value"] and macro["ci95"] == expected["ci95"]
        assert macro["valid_draws"] == 300 and not macro["eligible"]
        assert r["lenses"]["L6"]["final_proxy"]["value"] is None


@pytest.mark.parametrize("bad", ["seed", "protected", "nan", "probability", "shape"])
def test_invalid_or_incomplete_pairs_rejected(bad):
    _, arrays, _, _ = synthetic()
    if bad == "seed":
        del arrays["adapter"]["43"]
    elif bad == "protected":
        arrays["head"]["42"][0, 0] = 1
    elif bad == "nan":
        arrays["adapter"]["42"][0, 3] = np.nan
    elif bad == "probability":
        arrays["adapter"]["42"][0, 3] = 2
    else:
        arrays["head"]["42"] = arrays["head"]["42"][:-1]
    with pytest.raises(s.life.ContractError):
        s.protected_equal(arrays)


def decisions():
    def r(v):
        return dict(lenses=dict(L5=dict(weak_four=dict(value=v), sign_agreement=dict(L1=True)),
                                L6=dict(value=.0001)), harm_guard=dict(PASS=True))
    return {"head-parent": r(0), "adapter-parent": r(.001), "adapter-head": r(.0001)}


@pytest.mark.parametrize("bad", ["weak", "head", "tie", "sign", "blend", "guard", "missing"])
def test_null_failed_control_and_missing_metrics_stop(bad):
    r = decisions()
    assert s.decision(r)["diagnostic"] == "NECESSARY_CHECKS_PASS"
    assert s.head_checks(r["head-parent"])
    if bad == "weak":
        r["adapter-parent"]["lenses"]["L5"]["weak_four"]["value"] = .00099
    elif bad == "head":
        r["head-parent"]["lenses"]["L5"]["weak_four"]["value"] = -.00001
        assert not s.head_checks(r["head-parent"])
    elif bad == "tie":
        r["adapter-head"]["lenses"]["L5"]["weak_four"]["value"] = 0
    elif bad == "sign":
        r["adapter-head"]["lenses"]["L5"]["sign_agreement"]["L1"] = False
    elif bad == "blend":
        r["adapter-parent"]["lenses"]["L6"]["value"] = -.00001
    elif bad == "guard":
        r["head-parent"]["harm_guard"]["PASS"] = False
    else:
        r["head-parent"]["lenses"]["L5"]["weak_four"]["value"] = None
    assert s.decision(r)["diagnostic"] == "STOP"
    assert s.decision(r)["admission"] == "HOLD"


@pytest.mark.parametrize("key", ["sources", "cache_manifest_sha256"])
def test_panel_source_cache_drift_rejected(key):
    contract = dict(sources={"source": "sha"}, cache_manifest_sha256="cache")
    s.panel_binding(contract, contract)
    bad = deepcopy(contract)
    bad[key] = {}
    with pytest.raises(s.life.ContractError, match="sources/cache"):
        s.panel_binding(bad, contract)


def test_launch_order_queue_watchdog_and_futility_before_adapter():
    phases = s.commands()["phases"]
    assert len(phases["smoke"]) == 4 and len(phases["panel"]) == 6
    assert len(phases["review"]) == 8 and len(phases["fit"]) == 9
    assert phases["fit"][4][-1] == "head-gate"
    for p in ("smoke", "fit", "panel"):
        for cmd in phases[p]:
            if "tools/gpu_job.py" in cmd:
                assert cmd.count("--") == 2 and "--wait" in cmd
                assert "budget-run" in cmd
    fit_tags = [c[c.index("--tag") + 1] for c in phases["fit"] if "--tag" in c]
    assert fit_tags == [f"hb137_{k}-s{seed}-f0-v1-fit" for k in ("head", "adapter") for seed in (42, 43)]


def test_budget_cpu_success_failure_timeout_and_exhaustion(tmp_path):
    ledger = tmp_path / "budget.json"
    result = s.budget_run([sys.executable, "-c", "pass"], ledger, 20)
    assert result["status"] == "BUDGETED_COMMAND_PASS"
    with pytest.raises(s.life.ContractError, match="child failed"):
        s.budget_run([sys.executable, "-c", "raise SystemExit(7)"], ledger, 20)
    before = json.loads(ledger.read_text())
    assert before["attempts"][-1]["code"] == 7
    with pytest.raises(s.life.ContractError, match="child failed"):
        s.budget_run([sys.executable, "-c", "import time; time.sleep(10)"], ledger, before["seconds"] + .1)
    after = json.loads(ledger.read_text())
    assert after["attempts"][-1]["timed_out"] and not after["running"]
    with pytest.raises(s.life.ContractError, match="budget exhausted"):
        s.budget_run([sys.executable, "-c", "pass"], ledger, .01)


def test_interrupted_budget_fails_closed(tmp_path):
    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps(dict(version=s.VERSION, seconds=0, running=True, attempts=[])))
    with pytest.raises(s.life.ContractError, match="interrupted"):
        s.budget_run([sys.executable, "-c", "pass"], ledger)


def test_uncertain_child_exit_keeps_blocked_ledger(tmp_path, monkeypatch):
    class InterruptedChild:
        def wait(self, timeout):
            raise KeyboardInterrupt
        def poll(self):
            return None
    monkeypatch.setattr(s.subprocess, "Popen", lambda *a, **kw: InterruptedChild())
    ledger = tmp_path / "budget.json"
    with pytest.raises(KeyboardInterrupt):
        s.budget_run(["synthetic-child"], ledger)
    assert json.loads(ledger.read_text())["running"]


def test_readout_and_head_gate_fail_closed_before_unrun_fits(monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError("synthetic missing member")
    def forbidden(*a, **kw):
        raise AssertionError("missing fit must fail before prediction loading or result writing")
    monkeypatch.setattr(s.build, "arm_snapshot", missing)
    monkeypatch.setattr(s.screen, "load_predictions", forbidden)
    monkeypatch.setattr(s.build, "same_or_new", forbidden)
    for method in (s.readout, s.head_gate):
        with pytest.raises((FileNotFoundError, s.life.ContractError)):
            method()
