"""Synthetic regressions for the fixed CPU post-processing screen."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score
from tools import postproc_screen_v1 as s


def test_rank_matches_pandas_and_rejects_nonfinite():
    x = np.array([[1, 2], [1, 1], [3, 3], [4, 2.]])
    np.testing.assert_array_equal(s.rank(x), pd.DataFrame(x).rank(pct=True).values)
    with pytest.raises(ValueError):
        s.rank([1, np.nan])


def test_gauss_fixed_weighted_reference_and_tail_finite():
    a = np.arange(60).reshape(5, 12)
    b = a[::-1]
    expected = .8 * ndtri(np.clip(s.rank(a), .1, .9)) + .2 * ndtri(np.clip(s.rank(b), .1, .9))
    np.testing.assert_allclose(s.aggregate([a, b], [.8, .2], "gauss"), expected)
    assert np.isfinite(s.aggregate([np.ones((1, 12))] * 2, [.8, .2], "gauss")).all()


def test_trim_three_is_median_and_two_rejected():
    rng = np.random.default_rng(23)
    members = [rng.integers(0, 5, (19, 12)) for _ in range(3)]
    expected = np.median([s.rank(m) for m in members], axis=0)
    for weights in ([1, 1, 1], [17, 17, 12]):
        np.testing.assert_allclose(s.aggregate(members, weights, "trimmed"), expected)
    with pytest.raises(ValueError):
        s.aggregate(members[:2], [1, 1], "trimmed")
    np.testing.assert_allclose(s.aggregate([members[0]] * 3, [17, 17, 12], "trimmed"), s.rank(members[0]))


def test_prior_training_only_and_unknown_fallback():
    ids = [str(i) for i in range(50) if s.stage0.fold(str(i)) != 0][:4]
    labels = pd.DataFrame(np.array([[0]*12, [1]*12, [0]*12, [0]*12, [1]*12]), index=ids+["eval"], columns=s.TARGETS)
    meta = pd.Series(["M", "M", "F", "O", "F"], index=labels.index)
    prior, counts = s.fit_prior(ids, labels, meta, ["eval"])
    labels.loc["eval"] = np.nan
    other, _ = s.fit_prior(ids, labels, meta, ["eval"])
    for key in prior:
        np.testing.assert_array_equal(prior[key], other[key])
    out, _ = s.predict_prior(["eval", ids[-1], "missing"], meta, prior)
    np.testing.assert_array_equal(out[0], np.zeros(12))
    np.testing.assert_array_equal(out[1:], np.full((2, 12), .25))
    assert counts["train_n"] == 4
    with pytest.raises(ValueError, match="overlap"):
        s.fit_prior(ids, labels, meta, [ids[0]])
    fold0 = next(str(i) for i in range(100) if s.stage0.fold(str(i)) == 0)
    with pytest.raises(ValueError, match="Fold0"):
        s.fit_prior([fold0], labels, meta, [])


def test_grid_pairs_are_separate_and_only_one_target_changes():
    rng = np.random.default_rng(9)
    base = s.rank(rng.normal(size=(20, 12)))
    pred = s.postprocess(base, rng.integers(0, 2, base.shape))
    assert len(pred) == 11
    for a, b in s.PAIRS:
        for lam in s.LAMBDAS:
            p = pred[f"pair_{a}from{b}_l{lam:.1f}"]
            np.testing.assert_array_equal(np.delete(p, a, 1), np.delete(base, a, 1))
            np.testing.assert_array_equal(p[:, a], s.rank((1-lam)*s.rank(base[:, a])+lam*s.rank(base[:, b])))


@pytest.mark.parametrize("change,p,expected", [
    ({}, .8, True), ({"dread": 0}, 1, False), ({"jev_hybrid": -.000001}, 1, False),
    ({}, .79999, False), ({}, None, False), ({"consensus": np.nan}, 1, False)])
def test_strict_pass_rule(change, p, expected):
    ds = dict.fromkeys(s.RULERS, .001)
    ds.update(change)
    assert s.passes(ds, p) is expected


def test_jackknife_matches_brute_force_with_ties_and_masks():
    rng = np.random.default_rng(2)
    y = np.tile(np.array([0, 1]*8)[:, None], (1, 12))
    a, b = [rng.integers(0, 5, y.shape).astype(float) for _ in range(2)]
    mask = np.ones_like(y, bool)
    mask[[0, 1], :4] = False
    refs = {r: (y, mask, 1) for r in s.RULERS}
    got = s.paired_panel(a, b, refs)["consensus"]
    leaves = []
    for i in range(len(y)):
        dd = []
        for j in range(12):
            keep = mask[:, j].copy()
            keep[i] = False
            dd.append(roc_auc_score(y[keep, j], a[keep, j])-roc_auc_score(y[keep, j], b[keep, j]))
        leaves.append(np.mean(dd))
    expected = s.ev.normal_summary(got["delta"], leaves)
    np.testing.assert_allclose(got["ci95"], expected["ci95"], atol=1e-14)


def test_gold_paired_bootstrap_ties_and_invalid_draws():
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 12))
    a = np.tile(np.array([.2, .4, .4, .9])[:, None], (1, 12))
    b = a[::-1].copy()
    w = np.array([[1, 1, 1, 1], [2, 0, 1, 1], [4, 0, 0, 0]])
    scores = s.gold.auc_table({"a": a, "b": b}, y, w)
    for name, pred in (("a", a), ("b", b)):
        for draw in range(2):
            idx = np.repeat(np.arange(4), w[draw])
            assert scores[name][draw, 0] == roc_auc_score(y[idx, 0], pred[idx, 0])
    out = s.summarize_gold(scores["a"]-scores["b"])["macro12"]
    assert out["valid_draws"] == 1 and out["requested_draws"] == 2
    assert out["share_positive"] == 1
    zero = s.summarize_gold(scores["a"]-scores["a"])["macro12"]
    assert zero["share_positive"] == 0


def test_header_scan_official_paths_and_series_filter(tmp_path, monkeypatch):
    import io
    import zipfile
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    csv_path, zip_path = tmp_path / "test_series.csv", tmp_path / "data.zip"
    pd.DataFrame({s.UID: ["1", "2", "3"],
                  "SeriesInstanceUID": ["11", "22", "33"]}).to_csv(csv_path, index=False)

    def header(sex):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        if sex is not None:
            ds.PatientSex = sex
        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ds, enforce_file_format=True)
        return buf.getvalue()

    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("test_series/1/11/a.dcm", header("M"))
        z.writestr("test_series/1/11/b.dcm", header("M"))
        z.writestr("test_images/2/22/a.dcm", header(None))
        z.writestr("test_series/3/33/a.dcm", header(""))
        z.writestr("train_series/1/11/a.dcm", header("F"))
        z.writestr("test_series/4/44/a.dcm", header("F"))
    monkeypatch.setattr(s, "TEST_SERIES", csv_path)
    monkeypatch.setattr(s, "RAW_ZIP", zip_path)
    counts = s.test_header_counts()
    assert counts["readable_studies"] == counts["readable_series"] == 3
    assert counts["header_files_read"] == 4
    assert counts["studies_with_PatientSex_tag"] == 2
    assert counts["studies_with_nonempty_PatientSex"] == 1
    assert counts["header_read_failures"] == 0
