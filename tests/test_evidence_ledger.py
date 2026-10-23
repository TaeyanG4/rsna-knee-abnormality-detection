import pytest

from tools import evidence_ledger as L


def test_record_reading_report(tmp_path):
    for key, d, f, pub in [("a", .90, .85, .930), ("b", .91, .84, .935), ("c", .92, .87, .940)]:
        L.record(key, "OOF", [f"dread={d}", f"flight={f}"], root=tmp_path)
        L.reading(key, "1", pub, root=tmp_path)
    out = {d["source"]: d for d in L.report(root=tmp_path)}
    assert out["dread"]["spearman"] == 1.0 and out["dread"]["sign_agreement"] == "3/3"
    assert out["flight"]["sign_agreement"] == "2/3"
    with pytest.raises(ValueError, match="append-only"):
        L.record("a", "OOF", ["dread=0.5"], root=tmp_path)
    with pytest.raises(ValueError, match="already"):
        L.reading("a", "1", .9, root=tmp_path)


def test_post_reading_measurement_is_excluded(tmp_path):
    L.record("a", "OOF", ["dread=0.9"], root=tmp_path); L.reading("a", "1", .93, root=tmp_path)
    L.record("b", "OOF", ["dread=0.95"], root=tmp_path); L.reading("b", "2", .94, root=tmp_path)
    L.record("a", "OOF", ["late=0.1"], root=tmp_path)  # after the reading -> flagged, not scored
    assert "late" not in {d["source"] for d in L.report(root=tmp_path)}
