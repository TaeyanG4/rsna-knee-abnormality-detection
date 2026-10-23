"""CPU real-receipt authentication and synthetic rejection tests for H-B-153."""
import json

import pytest

from tools import c85r7_flipall_baseline as m


@pytest.fixture
def sources(tmp_path):
    for relative in [v[0] for v in m.RECEIPTS.values()] + [m.SUBMIT, m.READINGS]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((m.ROOT / relative).read_bytes())
    return tmp_path


def test_real_authenticated_baseline():
    result = m.baseline()
    assert result["hidden_seconds"] == pytest.approx(19357.222869978385)
    assert result["r7_seconds"] == pytest.approx(13048.471554289028)
    assert result["derivation"]["submit_to_complete_bound_seconds"] == 16882
    assert result["derivation"]["total"]["seconds_per_study"] == pytest.approx(14.496410509827589)
    assert result["r7_share"] < 1
    assert len(result["bindings"]) == 4
    assert all(result["bindings"][key]["record_sha256"] for key in ("submission", "completion"))


@pytest.mark.parametrize("name", list(m.RECEIPTS))
def test_tampered_receipt_fails_closed(sources, name):
    path = sources / m.RECEIPTS[name][0]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="DO_NOT_SUBMIT.*receipt hash mismatch"):
        m.baseline(sources)


@pytest.mark.parametrize("field,value", [("version", "1.0.0"), ("status", "PARTIAL"), ("studies", 31)])
def test_old_partial_wrong_count_rejected(field, value):
    receipt = json.loads((m.ROOT / m.RECEIPTS["stress_n32"][0]).read_bytes())
    receipt[field] = value
    with pytest.raises(ValueError, match="DO_NOT_SUBMIT"):
        m._validate_receipt(receipt, 32)


@pytest.mark.parametrize("relative", [m.SUBMIT, m.READINGS])
def test_mutated_bound_record_rejected(sources, relative):
    path = sources / relative
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    lines = [line.replace(b"1.1.0", b"1.0.0") if b"57133175" in line else line for line in lines]
    path.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="record hash mismatch"):
        m.baseline(sources)


@pytest.mark.parametrize("relative", [m.SUBMIT, m.READINGS])
def test_duplicate_bound_record_rejected(sources, relative):
    path = sources / relative
    raw = path.read_bytes()
    matching = next(line for line in raw.splitlines() if b"57133175" in line)
    path.write_bytes(raw.rstrip(b"\r\n") + b"\n" + matching + b"\n")
    with pytest.raises(ValueError, match="missing or duplicated"):
        m.baseline(sources)


def test_unrelated_append_does_not_invalidate_record_pins(sources):
    old = m.baseline(sources)
    path = sources / m.SUBMIT
    with path.open("ab") as stream:
        stream.write(b'\n{"ref":"unrelated","status":"test"}\n')
    path = sources / m.READINGS
    with path.open("ab") as stream:
        stream.write(b"\nunrelated,unrelated,,2026-10-15 00:00:00,append test\n")
    new = m.baseline(sources)
    assert new["hidden_seconds"] == old["hidden_seconds"]
    assert new["bindings"]["submission"]["file_sha256"] != old["bindings"]["submission"]["file_sha256"]


@pytest.mark.parametrize("n3,n32", [(float("nan"), 10), (10, float("inf")), (10, 10), (10, 9), (0, 1)])
def test_invalid_slopes_rejected(n3, n32):
    with pytest.raises(ValueError, match="DO_NOT_SUBMIT"):
        m._linear(n3, n32)


def test_negative_fixed_is_not_runtime_credit():
    result = m._linear(1, 100)
    assert result["fixed_seconds"] == 0
    assert result["projected_seconds"] == pytest.approx(1300 * 99 / 29)
