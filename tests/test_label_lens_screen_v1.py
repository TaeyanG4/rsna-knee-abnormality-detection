"""Synthetic CPU tests; no real labels, data rows or external code execution."""
import json
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from tools import label_lens_screen_v1 as m


def synthetic(n=30):
    rng = np.random.default_rng(812)
    y = np.tile(np.arange(n) % 2, (11, 1)).T
    a, b = rng.integers(0, 5, (2, n, 11))
    masks = np.ones_like(y, dtype=bool)
    masks[:4, 1:4] = False
    return a, b, y, masks


def test_paired_jackknife_matches_bruteforce_masked_sklearn():
    a, b, y, masks = synthetic()
    result = m.paired(a, b, y, masks)
    points, leaves = [], []
    for j in range(11):
        mm = masks[:, j]
        def delta(mask):
            return (roc_auc_score(y[mask, j], a[mask, j])
                    - roc_auc_score(y[mask, j], b[mask, j]))
        d = delta(mm)
        loo = []
        for i in range(len(y)):
            selected = mm.copy()
            selected[i] = False
            loo.append(delta(selected))
        points.append(d)
        leaves.append(loo)
        assert result["targets"][j]["delta"] == pytest.approx(d, abs=1e-14)
        sd = np.sqrt((len(y)-1)/len(y)*np.sum((loo-np.mean(loo))**2))
        assert result["targets"][j]["jackknife_se"] == pytest.approx(sd, abs=1e-14)
    point, loo = np.mean(points), np.mean(leaves, axis=0)
    se = np.sqrt((len(y)-1)/len(y)*np.sum((loo-np.mean(loo))**2))
    assert result["delta"] == pytest.approx(point, abs=1e-14)
    assert result["jackknife_se"] == pytest.approx(se, abs=1e-14)
    assert result["ci95"] == pytest.approx([point-1.959963984540054*se,
                                           point+1.959963984540054*se])
    assert result["p_delta_positive_normal"] == pytest.approx(NormalDist().cdf(point/se))


def test_zero_variance_ties_and_undefined_rejection():
    a, _, y, masks = synthetic()
    result = m.paired(a, a, y, masks)
    assert result["delta"] == 0
    assert result["jackknife_se"] == 0
    assert result["p_delta_positive_normal"] == .5
    for j in range(11):
        assert result["targets"][j]["delta"] == 0
    y[:, 0] = 0
    y[0, 0] = 1
    with pytest.raises(ValueError, match="Insufficient"):
        m.paired(a, a, y, masks)


def test_macro_is_eleven_and_shape_is_enforced():
    a, b, y, masks = synthetic()
    result = m.paired(a, b, y, masks)
    assert len(result["targets"]) == 11
    assert result["delta"] == pytest.approx(np.mean([r["delta"] for r in result["targets"]]))
    with pytest.raises(ValueError, match="eleven"):
        m.paired(a[:, :10], b[:, :10], y[:, :10], masks[:, :10])
    assert len(m.TARGETS) == 11 and "Synovitis" not in m.TARGETS
    assert m.INDICES == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]


def test_fixed_blends_preserve_nested_rank_and_vector():
    rng = np.random.default_rng(182)
    r7 = rng.integers(0, 7, (40, 12))
    members = {f"{a}_{s}": rng.integers(0, 7, r7.shape)
               for a in ("v2", "s0", "s1") for s in ("s3", "s2")}
    got = m.candidates(r7, members)
    r = lambda x: rankdata(x, axis=0, method="average")/len(x)
    assert m.SHARE == [.1, .2, .2, .5, .35, .5, .35, .35, .5, .2, .15, .35]
    assert list(got) == m.CANDIDATES
    for group in ("v2_s3", "x3_s3", "x3_s2"):
        means = (r(members["v2_s3"]) if group == "v2_s3" else
                 np.mean([r(members[a + "_" + group[-2:]]) for a in ("v2", "s0", "s1")], axis=0))
        for suffix, share in (("flat20", .2), ("shareV2", np.array(m.SHARE))):
            expected = r((1-share)*r(r7)+share*r(means))
            np.testing.assert_array_equal(got[group + "_" + suffix], expected)
    unranked = np.mean([r(members[a+"_s3"]) for a in ("v2", "s0", "s1")], axis=0)
    assert not np.array_equal(got["x3_s3_flat20"], r(.8*r(r7)+.2*unranked))


def make_csv(tmp_path):
    frame = pd.DataFrame({m.UID: ["c", "a", "b", "d"]})
    for target in m.TARGETS:
        frame[target] = [.4999, .5, .8, .1]
    frame["unused_text"] = ["ignored"]*4
    path = tmp_path / "labels.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_csv_threshold_alignment_and_no_external_execution(tmp_path):
    path, _ = make_csv(tmp_path)
    (tmp_path / "benchmark_core.py").write_text("raise RuntimeError('DO NOT EXECUTE')", encoding="utf-8")
    (y, mask), counts = m.read_lens(path, ["a", "b", "c"])
    np.testing.assert_array_equal(y[:, 0], [1, 1, 0])
    assert mask.all() and y.shape == (3, 11)
    assert counts["source_rows"] == 4
    assert counts["source_rows_outside_panel"] == 1
    assert all(t["exactly_half"] == 1 for t in counts["targets"])
    assert "a" not in counts and m.UID not in json.dumps(counts)
    with pytest.raises(ValueError, match="Only CSV"):
        m.read_lens(tmp_path / "benchmark_core.py", ["a"])


@pytest.mark.parametrize("fault", ["duplicate", "missing", "nonfinite", "range", "text", "schema", "null"])
def test_bad_csv_rejected_without_row_leak(tmp_path, fault):
    path, frame = make_csv(tmp_path)
    if fault == "duplicate":
        frame.loc[1, m.UID] = "c"
    elif fault == "missing":
        frame.loc[1, m.UID] = "private-id"
    elif fault == "nonfinite":
        frame.loc[1, m.TARGETS[0]] = np.inf
    elif fault == "range":
        frame.loc[1, m.TARGETS[0]] = 1.01
    elif fault == "text":
        frame[m.TARGETS[0]] = "private-label"
    elif fault == "schema":
        frame = frame.drop(columns=m.TARGETS[0])
    else:
        frame.loc[1, m.UID] = None
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError) as err:
        m.read_lens(path, ["a", "b", "c"])
    assert "private-id" not in str(err.value) and "private-label" not in str(err.value)


@pytest.mark.parametrize("d,p,expected", [
    (.001, .8, True), (.001, .799999, False), (0., .9, False),
    (-.001, .9, False), (.001, None, False), (.001, float("nan"), False),
    (float("nan"), .9, False), (float("inf"), .9, False)])
def test_track_rule_boundary_and_both_required(d, p, expected):
    pairs = dict(pair1=dict(delta=.002, p_delta_positive_normal=.9),
                 pair2=dict(delta=d, p_delta_positive_normal=p))
    assert m.tracks(pairs) == expected
    assert not m.tracks(dict(pair1=pairs["pair1"]))
    assert not m.tracks(dict(pairs, extra=pairs["pair1"]))


def test_correlations_ties_constants_and_known_order():
    c = m.correlations([1, 2, 3], [3, 2, 1])
    assert c["pearson"] == pytest.approx(-1)
    assert c["spearman"] == pytest.approx(-1)
    c = m.correlations([1, 1, 2], [1, 2, 3])
    assert c["spearman"] == pytest.approx(np.sqrt(.75))
    assert m.correlations([1, 1, 1], [1, 2, 3])["pearson"] is None
    with pytest.raises(ValueError):
        m.correlations([1, 2, np.nan], [1, 2, 3])


def test_registration_is_immutable_and_never_contains_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    clean = dict(spec=m.SPEC, pins={"synthetic.csv": "abc"}, runtime={"python": "test"})
    monkeypatch.setattr(m, "registration", lambda: clean.copy())
    assert m.register()["status"] == "REGISTERED"
    raw = (tmp_path / "REGISTRATION.json").read_text(encoding="utf-8")
    assert m.UID not in raw
    assert m.register()["status"] == "REGISTERED_IDENTICAL"
    clean["pins"] = {"synthetic.csv": "changed"}
    with pytest.raises(ValueError, match="differs"):
        m.register()
    with pytest.raises(ValueError, match="changed"):
        m.validate_registration()


@pytest.mark.parametrize("x", [np.array([np.nan]), np.empty((0, 12)), np.zeros((2, 3, 4))])
def test_rank_invalid_input(x):
    with pytest.raises(ValueError):
        m.rank(x)
