"""Frozen exploratory gate and arm selection regression checks."""
import numpy as np
import pytest
from tools import hb146_readout_v110 as readout


def test_report_branch_does_not_need_oai():
    got = readout.gate([.002, .002, .002, -.1], [.002, .002, .002, -.1], None, None, False)
    assert got["GO"] and got["report_branch"] and not got["image_branch"]


def test_report_metrics_must_pass_on_same_three_rulers():
    got = readout.gate([.003, .003, .003, 0], [0, .003, .003, .003], None, None, False)
    assert not got["GO"]
    assert len(got["qualifying_report_rulers"]) == 2


@pytest.mark.parametrize("macro,weak,gold,oai,eligible", [
    ([.002]*3+[np.nan], [.002]*4, None, None, False),
    ([0]*4, [0]*4, .00299, .003, True),
    ([0]*4, [0]*4, .003, .00299, True),
    ([0]*4, [0]*4, .003, .003, False),
    ([-.00301,0,0,0], [0]*4, .003, .003, True),
    ([np.nan,0,0,0], [0]*4, .003, .003, True),
])
def test_fail_closed_branches(macro, weak, gold, oai, eligible):
    assert not readout.gate(macro, weak, gold, oai, eligible)["GO"]


def test_image_branch_boundary_and_independent_weak_scores():
    got = readout.gate([-.003,0,0,0], [np.nan]*4, .003, .003, True)
    assert got["GO"] and got["image_branch"] and not got["report_branch"]


def test_engineering_never_go():
    gate = readout.gate([.01]*4, [.01]*4, .01, .01, True, engineering=True)
    assert gate["status"] == "ENGINEERING_ONLY" and not gate["GO"]
    assert readout.choose_arm({"global": gate, "fill": gate}, engineering=True)["selected_arm"] is None


def test_selection_uses_only_passing_arms_and_global_tie():
    low = readout.gate([.003]*4, [.003]*4, None, None, False)
    high = readout.gate([.004]*4, [.003]*4, None, None, False)
    assert readout.choose_arm({"global": low, "fill": high})["selected_arm"] == "fill"
    assert readout.choose_arm({"global": low, "fill": low})["selected_arm"] == "global"
    high["GO"] = False
    assert readout.choose_arm({"global": low, "fill": high})["selected_arm"] == "global"
    low["GO"] = False
    assert readout.choose_arm({"global": low, "fill": high})["status"] == "NO_GO"


def test_cli_records_explicit_exploratory_choice(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(readout, "run", lambda *a: calls.append(a) or {"status": "NO_GO"})
    readout.main(["--work", "synthetic", "--exposed-exploratory"])
    assert calls[-1][3] is True
    readout.main(["--work", "synthetic"])
    assert calls[-1][3] is False
