"""CPU-only strict META exception and immutable-reference regression tests."""
import copy
import json
import numpy as np
import pytest
from tools import l1_rco_k1_foldcache_v2 as k


@pytest.fixture
def panel(tmp_path, monkeypatch):
    ids = ["study-%02d" % i for i in range(32)]
    arrays = {u: np.zeros((2, 12), dtype=np.float32) for u in ids}
    ref = dict(contract_id="test", ids=ids, renderer_pins={}, source_receipt_sha256="source",
               studies={u: dict(px="pixel", meta=k.old.array_hash(arrays[u]),
                               blocks="blocks", rows=2) for u in ids})
    reference = tmp_path/"local32.json"
    reference.write_text(json.dumps(ref))
    (tmp_path/"receipt.json").write_text("{}")
    archive = tmp_path/"local32-meta.npz"
    np.savez_compressed(archive, **arrays)
    actual = copy.deepcopy(ref["studies"])
    monkeypatch.setattr(k, "identity", lambda _: (ref, {"contract_id": "test"}, ids))
    monkeypatch.setattr(k.old, "study_hashes", lambda *_: actual)
    monkeypatch.setattr(k, "meta_arrays", lambda *_: arrays)
    return tmp_path, reference, archive, arrays, actual, ref, ids


def test_exact_unchanged_and_explicit_exception(panel):
    cache, ref, archive, arrays, actual, _, ids = panel
    assert k.assess(cache, ref)["mode"] == "EXACT"
    arrays[ids[0]][0, 4] = np.float32(2.22e-16)
    actual[ids[0]]["meta"] = k.old.array_hash(arrays[ids[0]])
    with pytest.raises(ValueError, match="parity mismatch"):
        k.assess(cache, ref)
    proof = k.parity(cache, ref, archive, 1e-12)
    assert proof["mode"] == "META_EPS"
    assert proof["changed_cells"] == proof["changed_studies"] == 1
    assert proof["per_study"][ids[0]]["changed_columns"] == [4]
    assert k.validate_receipt(cache, ref, archive) == proof
    assert k.parity(cache, ref, archive, 1e-12) == proof


@pytest.mark.parametrize("field", ["px", "blocks", "rows"])
def test_nonmeta_never_relaxed(panel, field):
    cache, ref, archive, _, actual, _, ids = panel
    actual[ids[0]][field] = "changed"
    with pytest.raises(ValueError, match="EXACT"):
        k.assess(cache, ref, archive, 1e-12)


@pytest.mark.parametrize("value", [0, -1e-12, 1.01e-12, float("inf"), float("nan")])
def test_tolerance_cap(panel, value):
    cache, ref, archive, *_ = panel
    with pytest.raises(ValueError, match="tolerance"):
        k.assess(cache, ref, archive, value)


@pytest.mark.parametrize("mode", ["above", "nan", "dtype", "shape"])
def test_meta_invalid(panel, mode):
    cache, ref, archive, arrays, actual, _, ids = panel
    uid = ids[0]
    if mode == "above":
        arrays[uid][0, 4] = 2e-12
    elif mode == "nan":
        arrays[uid][0, 4] = np.nan
    elif mode == "dtype":
        arrays[uid] = arrays[uid].astype("float64")
    else:
        arrays[uid] = arrays[uid][:1]
    actual[uid]["meta"] = k.old.array_hash(arrays[uid])
    with pytest.raises(ValueError):
        k.assess(cache, ref, archive, 1e-12)


@pytest.mark.parametrize("mode", ["missing", "tamper", "duplicate"])
def test_reference_authentication(panel, mode):
    cache, ref, archive, arrays, _, _, ids = panel
    arrays = copy.deepcopy(arrays)
    if mode == "missing":
        del arrays[ids[0]]
    elif mode == "tamper":
        arrays[ids[0]][0, 0] = 1e-16
    else:
        arrays["extra"] = arrays[ids[0]]
    np.savez_compressed(archive, **arrays)
    with pytest.raises(ValueError):
        k.assess(cache, ref, archive, 1e-12)


@pytest.mark.parametrize("field,value", [
    ("changed_cells", 9), ("max_abs_diff", 1e-15), ("mode", "RELAXED"), ("mode", "EXACT"),
    ("meta_reference_sha256", "bad"), ("actual_studies_sha256", "bad"),
    ("studies", 31), ("smoke", True), ("tolerance", 1e-11),
    ("cache_receipt_sha256", "bad"), ("reference_sha256", "bad"),
])
def test_receipt_tamper(panel, field, value):
    cache, ref, archive, *_ = panel
    proof = k.parity(cache, ref, archive, 1e-12)
    proof[field] = value
    (cache/"H224-PARITY.json").write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        k.validate_receipt(cache, ref, archive)


def test_legacy_exact_and_no_overwrite(panel):
    cache, ref, archive, *_ = panel
    proof = k.assess(cache, ref)
    del proof["mode"]
    (cache/"H224-PARITY.json").write_text(json.dumps(proof))
    assert k.validate_receipt(cache, ref, archive) == proof
    with pytest.raises(ValueError, match="preserve"):
        k.parity(cache, ref, archive, 1e-12)


def test_export_authenticated_and_idempotent(panel, monkeypatch):
    from tools import l1_rco_k1_v1 as base
    cache, refpath, archive, arrays, _, ref, ids = panel
    reg = dict(train=ids, holdout=[], contract_id="test")
    ref["source_receipt_sha256"] = k.f.sha(cache/"receipt.json")
    monkeypatch.setattr(k, "identity", lambda _: (ref, reg, ids))
    monkeypatch.setattr(base, "complete_cache", lambda *_: {})
    output = cache/"export.npz"
    first = k.export_meta(cache, refpath, output)
    assert k.export_meta(cache, refpath, output) == first
    with np.load(output) as saved:
        assert set(saved.files) == set(ids)
    arrays[ids[0]][0, 1] = 1
    with pytest.raises(ValueError, match="hash mismatch"):
        k.export_meta(cache, refpath, output)
