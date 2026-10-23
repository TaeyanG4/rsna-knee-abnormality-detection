"""Native448 OAI source/mask identity and resumable output contract tests."""
import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from tools import hb147_oai_native448_v1 as m


@pytest.mark.parametrize("bad", [[], ["a", "a"], ["other"], ["b", "a"]])
def test_roster_rejects(bad):
    with pytest.raises(ValueError):
        m.selected_ids(bad, ["a", "b"])


def test_canonical_subset():
    assert m.selected_ids(["a", "c"], ["a", "b", "c"]) == ["a", "c"]
    assert m.selected_ids(None, ["a", "b"]) == ["a", "b"]


def test_native_renders_original_pixels_and_keeps_shared_contract(monkeypatch, tmp_path):
    row = dict(plane="Sagittal", fluid_sensitive="1", series_key="synthetic",
               rule_hit="known contrast synthetic", spacing_unknown="false")
    yy, xx = np.indices((512, 512))
    plane = ((xx % 7) * 100 + (yy % 11) * 50 + (xx * yy) % 53).astype(np.float32)
    vol = np.stack([plane + i * 10 for i in range(4)])
    series = m.common.Series(row, vol, np.zeros((4, 3)), np.zeros((4, 6)),
                             np.full(2, .4), 1., np.full(4, .4))
    monkeypatch.setattr(m.common, "load_series", lambda *args: series)
    original_contracts = copy.deepcopy(m.common.CONTRACTS)
    old, old_mask, picks = m.common.render_study([row], "oai", tmp_path)
    native, native_mask, native_picks = m.render448([row], tmp_path)
    assert native.shape == (64, 448, 448) and native.dtype == np.uint8
    assert m.common.CONTRACTS == original_contracts
    assert np.array_equal(native_mask, old_mask) and native_picks == picks
    assert not np.array_equal(native[0], cv2.resize(old[0], (448, 448), interpolation=cv2.INTER_LINEAR))


@pytest.fixture
def fake_build(monkeypatch, tmp_path):
    ids = ["synthetic"]
    masks = np.ones((1, 64), np.uint8)
    binding = {"studies": 1, "version": m.VERSION, "general_npz": [], "source_shards": []}
    ctx = dict(ids=ids, masks=masks, binding=binding, spec={"depth": 64, "img": 448},
               picks={ids[0]: [{"series": "synthetic"}]})
    def fake_context(requested=None):
        m.selected_ids(requested, ids)
        return copy.deepcopy(ctx)
    monkeypatch.setattr(m, "OUT", tmp_path / "oai448")
    monkeypatch.setattr(m, "MIN_FREE", 0)
    monkeypatch.setattr(m, "panel_ids", lambda: ids)
    monkeypatch.setattr(m, "context", fake_context)
    monkeypatch.setattr(m, "authenticate_inputs", lambda ctx, **kwargs: None)
    evidence = dict(replay336_exact=True, picks_identical=True, masks_identical=True,
                    picks_sha256=m.digest(ctx["picks"][ids[0]]),
                    mask_sha256=hashlib.sha256(masks[0].tobytes()).hexdigest())
    monkeypatch.setattr(m, "render_checked", lambda *args: (np.ones((64, 448, 448), np.uint8), evidence))
    return ctx


def test_prepare_verify_resume_is_immutable(fake_build, monkeypatch):
    out = m.prepare()
    assert m.verify(out)["studies"] == 1
    before = m.sha(out / "manifest.json")
    monkeypatch.setattr(m, "render_checked", lambda *args: pytest.fail("finished corpus rerendered"))
    assert m.prepare() == out
    assert m.sha(out / "manifest.json") == before
    from training.src.raptor_common import Corpus
    c = Corpus(out)
    assert c.volume("synthetic").shape == (64, 448, 448)
    assert np.array_equal(c.mask("synthetic"), fake_build["masks"][0])


def test_subset_has_fixed_smoke_path(fake_build):
    out = m.prepare(fake_build["ids"])
    assert out == m.OUT / "smoke"
    assert m.verify(out)["status"] == "VERIFIED"


@pytest.mark.parametrize("kind", ["pixels", "masks", "registration", "bounds", "picks", "incomplete"])
def test_output_corruption_fails_closed(fake_build, kind):
    out = m.prepare()
    manifest = m.io.read_json(out / "manifest.json")
    if kind == "pixels":
        with (out / "vols_shard0000.npy").open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x02")
    elif kind == "masks":
        np.save(out / "masks.npy", np.zeros((1, 64), np.uint8), allow_pickle=False)
    elif kind == "registration":
        m.io.atomic_json(out / "registration.json", {})
    elif kind == "bounds":
        manifest["shards"][0]["start"] = 1
    elif kind == "picks":
        manifest["shards"][0]["picks_sha256"] = "bad"
    elif kind == "incomplete":
        manifest["finished"] = False
    m.io.atomic_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError):
        m.verify(out)


def test_output_root_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="fixed roots"):
        m.verify(tmp_path)


def test_source_npz_corruption_fails(monkeypatch, tmp_path):
    (tmp_path / "s.npz").write_bytes(b"changed")
    monkeypatch.setattr(m.oai, "gen_dir", lambda root: tmp_path)
    with pytest.raises(ValueError, match="NPZ drift"):
        m.authenticate_inputs({"binding": {"general_npz": [{"file": "s.npz", "sha256": "wrong"}],
                                          "source_shards": []}})


def test_source_shard_corruption_fails(monkeypatch, tmp_path):
    (tmp_path / "s.npy").write_bytes(b"changed")
    monkeypatch.setattr(m, "SOURCE", tmp_path)
    with pytest.raises(ValueError, match="source shard drift"):
        m.authenticate_inputs({"binding": {"general_npz": [],
                                          "source_shards": [{"file": "s.npy", "sha256": "wrong"}]}})


def test_progress_reports_verified_files_not_failed_authentication(monkeypatch, tmp_path):
    (tmp_path / "s.npz").write_bytes(b"source")
    (tmp_path / "s.npy").write_bytes(b"shard")
    monkeypatch.setattr(m.oai, "gen_dir", lambda root: tmp_path)
    monkeypatch.setattr(m, "SOURCE", tmp_path)
    ctx = {"binding": {"general_npz": [{"file": "s.npz", "sha256": m.sha(tmp_path / "s.npz")}],
                       "source_shards": [{"file": "s.npy", "sha256": m.sha(tmp_path / "s.npy")}]}}
    ticks = []
    m.authenticate_inputs(ctx, progress=lambda: ticks.append(True))
    assert len(ticks) == 2
    ctx["binding"]["source_shards"][0]["sha256"] = "wrong"
    ticks.clear()
    with pytest.raises(ValueError, match="source shard drift"):
        m.authenticate_inputs(ctx, progress=lambda: ticks.append(True))
    assert len(ticks) == 1


def test_prepare_verify_progress_preserves_immutable_output(fake_build):
    ticks = []
    callback = lambda: ticks.append(True)
    out = m.prepare(progress=callback)
    assert len(ticks) == 3  # metadata commit, pixel commit, final shard verify
    before = m.sha(out / "manifest.json")
    ticks.clear()
    assert m.prepare(progress=callback) == out
    assert len(ticks) == 2  # resumed shard check plus final verify
    assert m.sha(out / "manifest.json") == before
    ticks.clear()
    assert m.verify(out, progress=callback)["status"] == "VERIFIED"
    assert len(ticks) == 1
    with (out / "vols_shard0000.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x02")
    ticks.clear()
    with pytest.raises(ValueError, match="native shard drift"):
        m.verify(out, progress=callback)
    assert ticks == []


@pytest.mark.parametrize("kind", ["pixels", "mask", "picks", "native_mask"])
def test_replay_drift_rejected(monkeypatch, tmp_path, kind):
    old = np.zeros((64, 336, 336), np.uint8)
    np.save(tmp_path / "source.npy", old[None], allow_pickle=False)
    mask = np.ones(64, np.uint8)
    picks = [{"series": "same"}]
    ctx = dict(plans={"a": []}, rows={}, audit=type("Audit", (), {"groups": {}})(),
               positions={"a": 0}, source={"shards": [{"start": 0, "stop": 1, "file": "source.npy"}]},
               ids=["a"], masks=mask[None], picks={"a": picks})
    returned_old, returned_mask, returned_picks = old.copy(), mask.copy(), copy.deepcopy(picks)
    if kind == "pixels":
        returned_old[0, 0, 0] = 1
    elif kind == "mask":
        returned_mask[0] = 0
    elif kind == "picks":
        returned_picks[0]["series"] = "different"
    monkeypatch.setattr(m, "SOURCE", tmp_path)
    monkeypatch.setattr(m.oai, "render", lambda *args: (returned_old, returned_mask, returned_picks))
    native_mask = mask.copy()
    if kind == "native_mask":
        native_mask[0] = 0
    monkeypatch.setattr(m, "render448", lambda *args: (np.zeros((64, 448, 448), np.uint8), native_mask, picks))
    monkeypatch.setattr(m.oai, "plan_rows", lambda *args: [])
    monkeypatch.setattr(m.oai, "_check_picks", lambda *args: None)
    with pytest.raises(ValueError, match="differs|drift"):
        m.render_checked(ctx, "a")
