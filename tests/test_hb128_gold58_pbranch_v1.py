"""CPU tests for tools/hb128_gold58_pbranch_v1.py (no GPU, no label rows emitted)."""
import json
import os
import zipfile

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from tools import hb128_gold58_pbranch_v1 as h

L = len(h.LABELS)


def members(n=58, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    out = {m: rng.random((n, L)) for m in h.MEMBERS}
    if ties:
        for m in h.MEMBERS:
            out[m][:6] = np.round(out[m][:6], 1)  # ties exercise average ranks
    return out


def test_two_member_blends_equal_public_cell7_formula():
    m = members(40, 1)
    for name, other in (("P956", "S6"), ("P_R7", "R7")):
        ours = h.rank_blend(m, h.BLENDS[name])
        assert np.array_equal(ours, h.p956.public_formula(m["R"], m[other]))
    # failures: R failure rows are all-0.5 sentinels in the public rule; S failures come from the controller
    m["R"][[2, 5]] = .5
    s_failed = np.zeros(40, bool)
    s_failed[[5, 9]] = True
    failed = {"R": np.all(m["R"] == .5, axis=1), "S6": s_failed}
    ours = h.rank_blend(m, h.BLENDS["P956"], failed)
    assert np.array_equal(ours, h.p956.public_formula(m["R"], m["S6"], s_failed))
    assert np.all(ours[5] == .5)


def test_three_member_blend_weights_and_survivor_renormalisation():
    m = members(30, 2, ties=False)
    p3 = h.rank_blend(m, h.BLENDS["P3"]).astype(float)
    manual = .5 * np.column_stack([h.pct(m["R"][:, j]) for j in range(L)]) + \
        .25 * np.column_stack([h.pct(m["S6"][:, j]) for j in range(L)]) + \
        .25 * np.column_stack([h.pct(m["R7"][:, j]) for j in range(L)])
    assert np.allclose(p3, manual.astype(np.float32), atol=0)
    failed = {k: np.zeros(30, bool) for k in h.MEMBERS}
    failed["R7"][4] = True
    failed["R"][7] = failed["S6"][7] = failed["R7"][7] = True
    out = h.rank_blend(m, h.BLENDS["P3"], failed).astype(float)
    ok = ~failed["R7"]
    r = h.pct(m["R"][~failed["R"], 0])
    s = h.pct(m["S6"][~failed["S6"], 0])
    # study 4: R7 failed -> (.5 r(R) + .25 r(S6)) / .75, ranks over each member's own scored rows
    ri = np.flatnonzero(~failed["R"]).tolist().index(4)
    si = np.flatnonzero(~failed["S6"]).tolist().index(4)
    assert out[4, 0] == pytest.approx((.5 * r[ri] + .25 * s[si]) / .75, abs=1e-7)
    assert np.all(out[7] == .5) and ok.sum() == 28
    with pytest.raises(ValueError, match="weights"):
        h.rank_blend(m, {"R": .5, "S6": .4})


def test_pct_matches_pandas_average_rank():
    import pandas as pd
    v = np.array([.3, .1, .3, .9, .1, .5])
    assert np.array_equal(h.pct(v), pd.Series(v).rank(method="average", pct=True).to_numpy())
    with pytest.raises(ValueError):
        h.pct(np.array([.1, np.nan]))


def test_score_arrays_auc_bootstrap_spearman_and_readouts():
    rng = np.random.default_rng(5)
    ids = [f"s{i:02d}" for i in range(58)]
    y = rng.integers(0, 2, (58, L)).astype(float)
    m = members(58, 6)
    m["R7"] = m["S6"].copy()
    stats, preds = h.score_arrays(ids, y, m)
    for name in ("R", "S6", "P3"):
        for j, t in enumerate(h.LABELS):
            assert stats["candidates"][name]["targets"][t]["observed"] == pytest.approx(roc_auc_score(y[:, j], preds[name][:, j]))
    c = stats["contrasts"]["R7-S6"]
    assert c["macro12"]["observed"] == 0 and c["weak_four"]["ci95"] == [0, 0]
    assert stats["candidates"]["R"]["macro12"]["requested_draws"] == 300
    rho = stats["spearman"]["S6_vs_R7"]
    assert rho["weak_four_mean"] == pytest.approx(1) and rho["median12"] == pytest.approx(1)
    r = stats["readouts"]
    assert r["I3_rule"]["readout"] == "P3_AVERAGE"
    assert r["HB128_corr_gate"]["decorrelated"] is False
    assert r["E7_gate"]["met"] is False
    assert np.array_equal(preds["P3"], h.rank_blend(m, h.BLENDS["P3"]).astype(float))
    text = h.markdown(h.clean(dict(stats, caveat=h.CAVEAT)))
    assert "P_R7" in text and "rho S6-R7" in text


def test_macro_draw_invalid_when_a_target_lacks_a_class():
    rng = np.random.default_rng(8)
    ids = [str(i) for i in range(58)]
    y = rng.integers(0, 2, (58, L)).astype(float)
    y[:, 0] = 0
    stats, _ = h.score_arrays(ids, y, members(58, 9))
    assert stats["candidates"]["R"]["macro12"]["observed"] is None
    assert stats["candidates"]["R"]["weak_four"]["observed"] is not None
    with pytest.raises(ValueError):
        h.score_arrays(ids[:57] + ids[:1], y, members(58, 9))


@pytest.mark.parametrize("w4,mac,expected", [(.006, .001, "R7_ONLY"), (-.006, .0, "KEEP_S6"), (.004, -.01, "P3_AVERAGE"),
                                             (.006, -.001, "UNSPECIFIED_R7_WEAK_FOUR_UP_MACRO_DOWN"), (None, .0, "UNAVAILABLE")])
def test_i3_readout_branches(w4, mac, expected):
    stats = {"contrasts": {"R7-S6": {"weak_four": {"observed": w4}, "macro12": {"observed": mac}}},
             "candidates": {"P3": {"macro12": {"observed": .9}}, "P_R7": {"macro12": {"observed": .89}}}}
    corr = {"S6_vs_R7": {"weak_four_mean": .9, "median12": .92}}
    out = h.readouts(stats, corr)
    assert out["I3_rule"]["readout"] == expected
    assert out["HB128_corr_gate"]["decorrelated"] is True and out["E7_gate"]["met"] is True


def refs_from(m):
    return {"R_raw": m["R"].copy(), "S6_raw": m["S6"].copy(), "R7_raw": m["R7"].copy(),
            "P956_final": h.rank_blend(m, h.BLENDS["P956"]).astype(float),
            "P_R7_final": h.rank_blend(m, h.BLENDS["P_R7"]).astype(float)}


def test_parity_statuses():
    ids = ["a" * 20, "b" * 20, "c" * 20]
    m = {k: np.array([[.1] * L, [.5] * L, [.9] * L]) + i * .01 for i, k in enumerate(h.MEMBERS)}
    none = {k: np.zeros(3, bool) for k in h.MEMBERS}
    refs = refs_from(m)
    assert h.compare(ids, m, none, refs)["status"] == "PASS"
    drift = dict(refs, R7_raw=refs["R7_raw"] + .01)  # raw drift, same ranks
    out = h.compare(ids, m, none, drift)
    assert out["status"] == "FINAL_MATCH_RAW_DRIFT"
    assert out["comparisons"]["R7_raw"]["rank_order_identical_targets"] == L
    flipped = dict(refs, P_R7_final=refs["P_R7_final"][::-1].copy())
    assert h.compare(ids, m, none, flipped)["status"] == "FAIL"
    bad = dict(none, S6=np.array([False, True, False]))
    with pytest.raises(ValueError, match="fallbacks"):
        h.compare(ids, m, bad, refs)


def test_stage_dicom_extracts_only_requested_studies(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "SCRATCH", tmp_path / "scratch")
    ids = ["1.2.3", "1.2.4"]
    archive = tmp_path / "comp.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for uid in ids + ["9.9.9"]:
            for k in range(2):
                z.writestr(f"train_series/{uid}/77{k}/{k}.dcm", f"{uid}-{k}".encode())
        z.writestr("train_series/1.2.3/770/readme.txt", b"x")
        z.writestr("test_series/1.2.3/770/0.dcm", b"y")
    series = tmp_path / "train_series.csv"
    series.write_text("StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n"
                      "1.2.3,770,1,1,Sagittal\n9.9.9,1,0,0,Axial\n1.2.4,771,0,0,Coronal\n", encoding="utf-8")
    dest = tmp_path / "scratch" / "dicom"
    out = h.stage_dicom(archive, dest, ids, series)
    assert out["status"] == "DICOM_STAGED" and out["files"] == 4 and out["studies"] == 2
    assert sorted(p.name for p in (dest / "train_series").iterdir()) == ids
    assert "9.9.9" not in (dest / "train_series.csv").read_text(encoding="utf-8")
    assert h.stage_dicom(archive, dest, ids, series) == out  # idempotent re-verify
    (dest / "train_series/1.2.3/770/0.dcm").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs"):
        h.verify_dicom(dest, ids)
    with pytest.raises(ValueError, match="below"):
        h.stage_dicom(archive, tmp_path / "elsewhere", ids, series)
    with pytest.raises(ValueError, match="lacks"):
        h.stage_dicom(archive, tmp_path / "scratch" / "other", ["1.2.3", "5.5.5"], series)


def test_unsafe_zip_members_rejected():
    with pytest.raises(ValueError, match="unsafe"):
        h._safe_member("train_series/1.2.3/../x.dcm", {"1.2.3"})
    assert h._safe_member("train_series/1.2.3/7/x.txt", {"1.2.3"}) is None
    assert h._safe_member("train_series/1.2.3/7/x.dcm", {"1.2.3"}) == ["train_series", "1.2.3", "7", "x.dcm"]


def test_wheel_extraction_rejects_traversal(tmp_path):
    wheel = tmp_path / "evil.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("../evil.py", b"x")
    with pytest.raises(ValueError, match="unsafe"):
        h._extract_wheels([wheel], tmp_path / "deps")
    good = tmp_path / "good.whl"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("pkg/__init__.py", b"")
    h._extract_wheels([good], tmp_path / "deps2")
    assert (tmp_path / "deps2/pkg/__init__.py").is_file()


def test_scratch_guard_and_pythonpath_restore(tmp_path):
    with pytest.raises(ValueError, match="below"):
        h.scratch(h.ROOT / "artifacts" / "x")
    with pytest.raises(ValueError, match="below"):
        h.scratch(h.SCRATCH)
    before = os.environ.get("PYTHONPATH")
    with h.worker_pythonpath(tmp_path):
        assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)
    assert os.environ.get("PYTHONPATH") == before


def test_visible_set_and_reference_pins():
    root, split, ids = h.study_set("visible")
    assert split == "test_series" and len(ids) == 3 and all((root / split / u).is_dir() for u in ids)
    refs = h.references(ids)
    assert set(refs) == set(h.REFERENCES) and all(v.shape == (3, L) for v in refs.values())
    # the two Kaggle final CSVs are the public cell-7 blends of the Kaggle raw outputs
    kaggle = {"R": refs["R_raw"], "S6": refs["S6_raw"], "R7": refs["R7_raw"]}
    assert np.abs(h.rank_blend(kaggle, h.BLENDS["P_R7"]) - refs["P_R7_final"]).max() <= h.FINAL_ATOL
    assert np.abs(h.rank_blend(kaggle, h.BLENDS["P956"]) - refs["P956_final"]).max() <= h.FINAL_ATOL


def test_r_worker_glue_with_fake_serving(tmp_path, monkeypatch):
    """CPU check of the R child's bookkeeping (served math itself is P956/H-B-113's, exercised on CUDA by B)."""
    import types
    import torch
    root = tmp_path / "data"
    (root / "test_series").mkdir(parents=True)
    ids = ["1.1", "1.2"]
    (root / "test_series.csv").write_text("StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n"
                                          "1.1,5,1,1,Sagittal\n1.2,6,0,0,Axial\n", encoding="utf-8")
    monkeypatch.setattr(h, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(h, "study_set", lambda name: (root, "test_series", ids))
    monkeypatch.setattr(h.p956, "check_pin", lambda path, digest: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "FAKE")
    seen = []

    def make_builder(*a, **k):
        def reader(uid, records, input_root):
            seen.append((uid, records[uid][0]["SeriesInstanceUID"], input_root))
            return np.full((96, 384, 384), 7, np.uint8), np.ones(96, np.uint8)
        return reader
    fake = types.SimpleNamespace(FASTREAD_VARIANT="v", IMG=384, SLOTS=[], FR_HDR=1, FR_PX=1,
                                 _eval_centers=lambda mask, d, k: list(range(k)))
    fake.load_model = lambda path, arch, res, device: ("model", 320)
    fake.gpu_windows = lambda vol, mask, k, res, dev, mirror=False: torch.full((k, 3, 4, 4), 1. if mirror else 0.)
    fake.infer_probs = lambda m, x, d: np.full(12, .25 if float(x.mean()) else .75, np.float32)
    fake.infer_pair_probs = lambda m, p, q, d: .5 * (fake.infer_probs(m, p, d) + fake.infer_probs(m, q, d))
    monkeypatch.setattr(h.hb113, "runtime", lambda: fake)
    monkeypatch.setattr(h.hb113, "fastread", lambda module: types.SimpleNamespace(make_builder=make_builder))
    out = h.r_worker("visible", tmp_path / "scratch" / "run" / "r", None)
    assert out["status"] == "PASS_R" and out["studies"] == 2
    assert [s[:2] for s in seen] == [("1.1", "5"), ("1.2", "6")] and seen[0][2] == str(root / "test_series")
    with np.load(tmp_path / "scratch/run/r/raw.npz") as raw:
        assert raw["ids"].tolist() == ids and np.allclose(raw["R"], .5) and np.allclose(raw["R_mirror"], .25)
    receipt = json.loads((tmp_path / "scratch/run/r/receipt.json").read_text())
    assert receipt["raw_sha256"] == h.sha(tmp_path / "scratch/run/r/raw.npz") and len(receipt["inputs"]) == 2
    with pytest.raises(FileExistsError):
        h.r_worker("visible", tmp_path / "scratch" / "run" / "r", None)


def test_registration_pins_when_present():
    if not (h.STAGE / "registration.json").exists():
        pytest.skip("registration not written yet")
    binding = h.validate()
    assert binding["gold_count"] == 58 and binding["blends"] == h.BLENDS
    assert json.loads((h.STAGE / "registration.json").read_text())["serving"]["amp"] == "fp16"
