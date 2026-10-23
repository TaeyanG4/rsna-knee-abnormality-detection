"""CPU-only negative gates and real lightweight child lifecycle checks."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import d01_r2_shard_benchmark as bench


def _parts():
    ids = [f"u{i:03}" for i in range(100)]
    raw = np.random.default_rng(23).random((5, 100, 12), dtype=np.float32)
    return ids, raw, [(ids[::2], raw[:, ::2]), (ids[1::2], raw[:, 1::2])]


def test_interleaved_shards_merge_to_exact_original_order():
    ids, raw, parts = _parts()
    assert bench.split_ids(ids) == [ids[::2], ids[1::2]]
    np.testing.assert_array_equal(bench.merge_arrays(ids, list(reversed(parts))), raw)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "unexpected", "nan", "shape", "out_of_range"])
def test_merge_fails_invalid_coverage_or_predictions(problem):
    ids, _, parts = _parts()
    parts = [(list(i), p.copy()) for i, p in parts]
    if problem == "duplicate": parts[1][0][0] = parts[0][0][0]
    if problem == "missing": parts[1] = (parts[1][0][:-1], parts[1][1][:, :-1])
    if problem == "unexpected": parts[1][0][0] = "not-requested"
    if problem == "nan": parts[1][1][0, 0, 0] = np.nan
    if problem == "shape": parts[1] = (parts[1][0], parts[1][1][:4])
    if problem == "out_of_range": parts[1][1][0, 0, 0] = 1.1
    with pytest.raises(ValueError): bench.merge_arrays(ids, parts)


def test_probability_gate_enforces_max_and_each_member_mean():
    ref = np.full((5, 100, 12), .5, dtype=np.float32)
    actual = ref.copy(); actual[2, 4, 8] += .0021
    gate = bench.probability_gate(actual, ref)
    assert not gate["pass"] and not gate["members"][2]["pass"]
    actual = ref.copy(); actual[4] += .0006
    gate = bench.probability_gate(actual, ref)
    assert gate["mean_abs"] < .0005 and not gate["pass"]  # aggregate would conceal one bad member
    assert bench.probability_gate(ref.copy(), ref)["pass"]


def test_build_run_separation_rejects_local_gpu_run(tmp_path):
    with pytest.raises(RuntimeError, match="Kaggle Linux"):
        bench.run(tmp_path, tmp_path, tmp_path, tmp_path / "codec", tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()


def test_supervisor_reaps_timed_out_children(tmp_path):
    pytest.importorskip("psutil")
    commands = [[sys.executable, "-c", "import time; time.sleep(30)"]] * 2
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        bench.supervise(commands, [dict(os.environ)] * 2, tmp_path, .3)
    assert time.monotonic() - started < 8
    progress = json.loads((tmp_path / "supervisor-progress.json").read_text())
    assert progress["all_children_reaped"] and progress["status"] == "FAILED_OR_INTERRUPTED"


def test_supervisor_reaps_sibling_when_one_child_fails(tmp_path):
    pytest.importorskip("psutil")
    commands = [[sys.executable, "-c", "raise SystemExit(7)"],
                [sys.executable, "-c", "import time; time.sleep(30)"]]
    with pytest.raises(RuntimeError, match="shard failed"):
        bench.supervise(commands, [dict(os.environ)] * 2, tmp_path, 5)
    progress = json.loads((tmp_path / "supervisor-progress.json").read_text())
    assert progress["all_children_reaped"]
    assert progress["exit_codes"][0] == 7
