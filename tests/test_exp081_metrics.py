"""Synthetic CPU tests; no clinical labels, results, models or DICOM inputs."""
from copy import deepcopy
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp081_metrics as metrics


def inputs():
    labels = np.tile(np.asarray([0, 1] * 4)[:, None], (1, 12))
    baseline = .9 - .8 * labels
    candidate = .1 + .8 * labels
    groups = np.repeat(["a", "b", "c", "d"], 2)
    changed = np.asarray([True, True, False, False, True, True, False, False])
    return [labels, baseline, candidate, groups, changed]


def evaluate(data=None, **kwargs):
    return metrics.evaluate(*(inputs() if data is None else data), reps=kwargs.pop("reps", 9), **kwargs)


def test_explicit_perfect_reversal_all_targets_and_macro_contributions():
    report = evaluate()
    assert report["targets_order"] == ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
    for row in report["full"]["targets"]:
        assert (row["P"], row["N"]) == (4, 4)
        assert row["auc"] == {"baseline": 0., "candidate": 1.}
        assert row["delta"] == 1.
        assert row["macro_contribution"] == pytest.approx(1 / 12)
    for name in ("macro12", "PFOA", "other11", "Fracture"):
        assert report["full"]["aggregates"]["delta"][name] == 1.
        assert report["conditional_intervals"][name] == dict(requested_reps=9, valid_reps=9, low=1., high=1.)
    assert report["conditional_intervals"]["groups"] == 4
    assert all(r["studies"] == 4 for r in report["subsets"].values())
    json.dumps(report, allow_nan=False)


def test_identical_predictions_have_zero_paired_intervals():
    data = inputs()
    data[2] = data[1].copy()
    report = evaluate(data)
    for interval in report["conditional_intervals"]["targets"].values():
        assert interval["low"] == interval["high"] == 0.


def test_one_class_all12_stays_null_json_safe_and_records_zero_valid_reps():
    data = inputs()
    data[0][:] = 0
    report = evaluate(data)
    assert report["full"]["aggregates"]["delta"] == dict(macro12=None, PFOA=None, other11=None, Fracture=None)
    for row in report["full"]["targets"]:
        assert row["P"] == 0 and row["N"] == 8
        assert row["auc"] == dict(baseline=None, candidate=None)
        assert row["delta"] is row["macro_contribution"] is None
    for interval in report["conditional_intervals"]["targets"].values():
        assert interval == dict(requested_reps=9, valid_reps=0, low=None, high=None)
    json.dumps(report, allow_nan=False)


def test_one_class_pfoa_does_not_make_partial_macro_but_other11_remains_defined():
    data = inputs()
    data[0][:, metrics.PFOA] = 0
    report = evaluate(data)
    aggregates = report["full"]["aggregates"]["delta"]
    assert aggregates == dict(macro12=None, PFOA=None, other11=1., Fracture=1.)
    assert report["conditional_intervals"]["macro12"]["valid_reps"] == 0
    assert report["conditional_intervals"]["other11"]["valid_reps"] == 9


def test_float32_rounding_occurs_before_numpy_half_rounding():
    data = inputs()
    soft = np.asarray([.5, .50000001, .5000001, .49, .51, .9, 0., 1.])
    data[0] = np.tile(soft[:, None], (1, 12))
    report = evaluate(data)
    assert all(r["P"] == 4 and r["N"] == 4 for r in report["full"]["targets"])
    assert np.rint(soft).sum() == 5  # float64-first rounding would disagree.


def test_empty_subset_and_one_class_changed_subset_are_reported_without_partial_auc():
    data = inputs()
    data[4][:] = True
    report = evaluate(data)
    empty = report["subsets"]["unchanged"]
    assert empty["studies"] == 0 and empty["aggregates"]["delta"]["macro12"] is None
    assert all(r["P"] == r["N"] == 0 for r in empty["targets"])
    data[4] = data[0][:, 0].astype(bool)
    report = evaluate(data)
    assert report["subsets"]["changed"]["aggregates"]["delta"]["macro12"] is None
    assert report["full"]["aggregates"]["delta"]["macro12"] == 1.


def pairwise_auc(y, p, weights):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    denominator = weights[pos].sum() * weights[neg].sum()
    if denominator == 0:
        return np.nan
    numerator = sum(weights[i] * weights[j] * (float(p[i] > p[j]) + .5 * float(p[i] == p[j]))
                    for i in pos for j in neg)
    return numerator / denominator


def test_group_bootstrap_matches_independent_pairwise_oracle_with_invalid_draws():
    labels = np.asarray([0, 0, 1, 1, 1, 0])
    baseline = np.asarray([.1, .7, .8, .8, .3, .7])
    candidate = np.asarray([.2, .2, .3, .1, .9, .8])
    groups = np.asarray(["a", "a", "b", "b", "b", "c"])
    data = [np.tile(v[:, None], (1, 12)) for v in (labels, baseline, candidate)]
    data += [groups, np.zeros(6, dtype=bool)]
    reps, seed = 63, 21
    report = evaluate(data, reps=reps, seed=seed)
    unique, codes = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(reps):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[codes]
        deltas.append(pairwise_auc(labels, candidate, weights) - pairwise_auc(labels, baseline, weights))
    good = np.asarray(deltas)[np.isfinite(deltas)]
    actual = report["conditional_intervals"]["macro12"]
    assert 0 < len(good) < reps
    assert actual["valid_reps"] == len(good)
    assert [actual["low"], actual["high"]] == pytest.approx(np.quantile(good, [.025, .975]))
    point = report["full"]["aggregates"]["delta"]["macro12"]
    assert point == pytest.approx(pairwise_auc(labels, candidate, np.ones(6)) - pairwise_auc(labels, baseline, np.ones(6)))


def test_paired_group_results_are_invariant_to_joint_row_permutation():
    data = inputs()
    rng = np.random.default_rng(33)
    data[1] = rng.random((8, 12))
    data[2] = rng.random((8, 12))
    permutation = np.asarray([7, 0, 2, 5, 4, 3, 6, 1])
    before = evaluate(data, reps=37)
    after = evaluate([v[permutation] for v in data], reps=37)
    assert before == after


def test_existing_metadata_groups_are_descriptive_and_unknown_fields_rejected():
    groups = {"scanner_group": ["x", "y"] * 4, "field": [1.5] * 8,
              "manufacturer": ["known"] * 4 + ["unknown"] * 4,
              "cached_mask": ["all_nonzero"] * 8}
    report = evaluate(diagnostic_groups=groups)
    scans = [r for r in report["diagnostic_groups"] if r["field"] == "scanner_group"]
    assert len(scans) == 2 and sum(r["studies"] for r in scans) == 8
    assert all(r["aggregates"]["delta"]["macro12"] is None for r in scans)
    with pytest.raises(ValueError):
        evaluate(diagnostic_groups={"new_outcome_group": ["a"] * 8})


def test_raw_probabilities_are_not_replaced_by_ranks(monkeypatch):
    data = inputs()
    data[1] = np.tile(np.linspace(.117, .893, 8)[:, None], (1, 12))
    data[2] = np.tile(np.linspace(.036, .954, 8)[::-1, None], (1, 12))
    data[4][:] = True
    helper = metrics.weighted_auc
    seen = []
    def audited(y, p, weights):
        assert any(np.array_equal(p, original[:, 0]) for original in data[1:3])
        seen.append(len(p))
        return helper(y, p, weights)
    monkeypatch.setattr(metrics, "weighted_auc", audited)
    evaluate(data)
    assert seen


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("value", [np.nan, np.inf, -.001, 1.001])
def test_nonfinite_or_out_of_range_arrays_rejected(index, value):
    data = inputs()
    data[index] = data[index].astype(float)
    data[index][0, 0] = value
    with pytest.raises(ValueError):
        evaluate(data)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("shape", [(0, 12), (8, 11), (8,), (7, 12), (8, 12, 1)])
def test_empty_invalid_or_mismatched_array_shapes_rejected(index, shape):
    data = inputs()
    data[index] = np.zeros(shape)
    with pytest.raises(ValueError):
        evaluate(data)


@pytest.mark.parametrize("value", [["a"] * 7, [["a"]] * 8, [None] * 8,
                                  [" "] * 8, [np.nan] * 8, [1, "1"] * 4])
def test_invalid_report_groups_rejected(value):
    data = inputs(); data[3] = value
    with pytest.raises(ValueError):
        evaluate(data)


@pytest.mark.parametrize("value", [[1] * 7, [[1]] * 8, [np.nan] * 8, [.5] * 8, [2] * 8, ["1"] * 8])
def test_invalid_changed_mask_rejected(value):
    data = inputs(); data[4] = value
    with pytest.raises(ValueError):
        evaluate(data)


@pytest.mark.parametrize("kwargs", [{"reps": 0}, {"reps": -1}, {"reps": 1.5}, {"reps": True},
                                    {"seed": -1}, {"seed": 1.5}, {"seed": True},
                                    {"diagnostic_groups": []}, {"diagnostic_groups": {"field": [None] * 8}}])
def test_invalid_bootstrap_and_diagnostics_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate(**kwargs)


def test_numeric_integer_report_group_ids_keep_distinct_large_integer_identity():
    data = inputs()
    data[3] = np.asarray([2**60 + i // 2 for i in range(8)], dtype=np.int64)
    assert evaluate(data)["conditional_intervals"]["groups"] == 4


def test_mixed_numeric_groups_do_not_merge_large_integer_ids_via_float_conversion():
    data = inputs()
    data[3] = [2**60, 2**60, 2**60 + 1, 2**60 + 1, 1.5, 1.5, 2.5, 2.5]
    assert evaluate(data)["conditional_intervals"]["groups"] == 4


def test_default_protocol_uses_500_draws_and_frozen_seed():
    report = metrics.evaluate(*inputs())
    assert report["conditional_intervals"]["seed"] == 20261003
    assert report["conditional_intervals"]["requested_reps"] == 500
    assert report["conditional_intervals"]["macro12"]["valid_reps"] == 500


def test_eval_is_immutable_and_does_not_open_files(monkeypatch):
    data = inputs()
    before = deepcopy(data)
    for value in data:
        value.flags.writeable = False
    def forbidden(*args, **kwargs):
        raise AssertionError("Metric evaluation attempted file access")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    evaluate(data)
    assert all(np.array_equal(a, b) for a, b in zip(data, before))


def test_import_is_passive_without_data_model_or_command_access(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), EXP081_TEST_ROOT=str(ROOT))
    code = """
import os, sys
from pathlib import Path
root = Path(os.environ['EXP081_TEST_ROOT']).resolve()
blocked = tuple(root / name for name in ('storage', 'artifacts', '.local'))
def audit(event, args):
    assert event not in ('subprocess.Popen', 'os.system', 'socket.connect'), event
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        assert not any(path.is_relative_to(parent) for parent in blocked), path
sys.addaudithook(audit)
import tools.exp081_metrics
assert 'torch' not in sys.modules
assert 'raptor_train' not in sys.modules
assert 'raptor_infer' not in sys.modules
"""
    run = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert not list(tmp_path.iterdir())


def test_cli_style_import_without_repository_root_on_sys_path(tmp_path):
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'tools')!r}); "
            "import exp081_input_compare; import exp081_metrics; "
            "assert 'torch' not in sys.modules")
    run = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code], cwd=tmp_path,
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
