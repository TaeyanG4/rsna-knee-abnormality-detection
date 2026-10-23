import copy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from training.arms import laterality_v1 as lat


def header(side="L", x=80., plane="Coronal", po=None):
    iop = {"Coronal": [1, 0, 0, 0, 0, -1], "Axial": [1, 0, 0, 0, 1, 0],
           "Sagittal": [0, 1, 0, 0, 0, -1]}[plane]
    result = {"Laterality": side, "ImageOrientationPatient": iop,
              "ImagePositionPatient": [x - iop[0] * 49.5, 0, 0],
              "PixelSpacing": [1, 1], "Rows": 100, "Columns": 100}
    if po is not None:
        result["PatientOrientation"] = po
    return result


def series(side="L", x=80., plane="Coronal", **kw):
    return lat.summarize_series([header(side, x, plane)], plane, **kw)


@pytest.mark.parametrize("side,x", [("L", 80), ("R", -80)])
def test_tags_agree_geometry(side, x):
    out = lat.resolve_study([series(side, x)])
    assert out["side"] == side and out["tag_geometry_agree"]


def test_direct_tag_sources_and_body_part():
    assert lat.direct_sides({"ImageLaterality": "L", "Laterality": "R"}) == {"L", "R"}
    assert lat.direct_sides({"BodyPartExamined": "RIGHT KNEE"}) == {"R"}
    assert lat.direct_sides({"BodyPartExamined": "KNEE", "PatientOrientation": ["L", "F"]}) == set()
    assert lat.direct_sides({"Laterality": "B"}) == {"L", "R"}


def test_geometry_requires_position_and_valid_frame():
    h = header()
    assert lat.geometry(h, "Coronal") == (80., True, False)
    del h["ImagePositionPatient"]
    assert lat.geometry(h, "Coronal") == (None, False, False)
    h = header(); h["ImageOrientationPatient"] = [0] * 6
    assert lat.geometry(h, "Coronal")[0] is None


def test_geometry_only_needs_two_series():
    assert lat.resolve_study([series("", 80.)])["side"] == "U"
    result = lat.resolve_study([series("", 80.), series("", 82., "Axial")])
    assert result["side"] == "L" and result["reason"] == "geometry_consensus"


@pytest.mark.parametrize("items,reason", [
    ([series("L"), series("R")], "direct_tag_conflict"),
    ([series("", 80), series("", -80)], "geometry_conflict"),
    ([series("L", -80)], "tag_geometry_disagreement"),
    ([series("", 0), series("", 19.9)], "insufficient_evidence"),
    ([series(errors=1)], "missing_or_unreadable_headers"),
    ([], "missing_or_unreadable_headers"),
])
def test_unknown_fail_closed(items, reason):
    out = lat.resolve_study(items)
    assert out["side"] == "U" and out["reason"] == reason


def test_orientation_qa():
    h = header(); h["ImageOrientationPatient"] = [-1, 0, 0, 0, 0, -1]
    assert not lat.geometry(h, "Coronal")[1]
    h = header(po=["R", "F"])
    assert lat.geometry(h, "Coronal")[1:] == (False, True)
    assert not lat.geometry(header(), "Axial")[1]
    out = lat.resolve_study([lat.summarize_series([h], "Coronal")])
    assert out["side"] == "U" and out["reason"] == "unsafe_orientation"


def test_sample_conflict():
    s = lat.summarize_series([header("", 80), header("", -80)], "Coronal")
    assert s["sample_geometry_conflict"]
    assert lat.resolve_study([s])["side"] == "U"


def make_map(tmp_path, records=None):
    p = tmp_path / "sides.json"
    p.write_bytes(lat.canonical({"schema": "hb123-side-map/v1", "policy": lat.POLICY,
                              "studies": records or [{lat.UID: "a", "side": "L"}]}))
    return lat.SideMap(p, lat.sha(p))


def test_map_binding_coverage_and_pickle(tmp_path):
    sides = make_map(tmp_path)
    assert pickle.loads(pickle.dumps(sides)).side("a") == "L"
    with pytest.raises(ValueError, match="digest"):
        lat.SideMap(sides.path, "0" * 64)
    with pytest.raises(ValueError, match="lacks"):
        sides.require(["absent"])
    with pytest.raises(ValueError, match="lacks"):
        sides.side("absent")


def test_map_reject_duplicate_unknown_policy(tmp_path):
    with pytest.raises(ValueError, match="malformed"):
        make_map(tmp_path, [{lat.UID: "a", "side": "L"}] * 2)
    sides = make_map(tmp_path)
    doc = json.loads(open(sides.path).read()); doc["policy"]["canonical_side"] = "L"
    open(sides.path, "w").write(json.dumps(doc))
    with pytest.raises(ValueError, match="policy"):
        lat.SideMap(sides.path)


SLOTS = [("Sagittal", 1, 2), ("Coronal", 1, 2), ("Axial", -1, 2)]


@pytest.mark.parametrize("side", ["L", "R", "U"])
def test_flip_exact_no_mutation_rng_or_slice_reversal(side):
    volume = np.arange(6 * 3 * 4, dtype=np.uint8).reshape(6, 3, 4)
    original = volume.copy()
    volume.flags.writeable = False
    state = copy.deepcopy(np.random.get_state())
    out = lat.canonical_volume(volume, side, SLOTS)
    assert np.array_equal(out[:2], original[:2])
    assert np.array_equal(out[2:], original[2:, :, ::-1] if side == "L" else original[2:])
    assert np.array_equal(volume, original)
    assert np.array_equal(np.random.get_state()[1], state[1])
    assert np.array_equal(lat.canonical_volume(out, side, SLOTS), original)


def test_invalid_volume_or_side():
    with pytest.raises(ValueError):
        lat.canonical_volume(np.zeros((6, 3, 4)), "?", SLOTS)
    with pytest.raises(ValueError):
        lat.canonical_volume(np.zeros((5, 3, 4)), "L", SLOTS)


class FakeCorpus:
    ids = ["a"]
    spec = {"slots": SLOTS}
    def volume(self, uid):
        return np.arange(72, dtype=np.uint8).reshape(6, 3, 4)
    def mask(self, uid):
        return np.ones(6, dtype=bool)


class FakeDataset:
    corpus_dir, ids, k = "unused", ["a"], 36
    def __init__(self):
        self.corpus, self.epoch = FakeCorpus(), 0
    def __len__(self): return 1
    def set_epoch(self, epoch): self.epoch = epoch
    def __getitem__(self, index): return self.corpus.volume("a"), self.epoch


def test_dataset_decorator_before_shared_item_and_spawn(tmp_path):
    sides = make_map(tmp_path)
    ds = lat.CanonicalDataset(FakeDataset(), sides)
    ds = pickle.loads(pickle.dumps(ds))
    ds.set_epoch(3)
    assert len(ds) == 1 and ds.k == 36
    first, epoch = ds[0]
    assert epoch == 3
    assert np.array_equal(first, lat.canonical_volume(FakeCorpus().volume("a"), "L", SLOTS))
    assert np.array_equal(ds[0][0], first)  # no repeated/double flip
    assert np.array_equal(ds.dataset.corpus.mask("a"), FakeCorpus().mask("a"))


def test_dataset_interface_rejected(tmp_path):
    with pytest.raises(ValueError, match="interface"):
        lat.CanonicalDataset(SimpleNamespace(), make_map(tmp_path))
