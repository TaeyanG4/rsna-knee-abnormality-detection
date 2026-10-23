"""CPU-only v1.2.3 postrun contracts; never access running full outputs."""
import csv
import importlib.util
import io

import pytest
from tools import hb149_postrun_v123 as p

# Changelog 2026-10-14 (v1.2.4 build): superseded. Pairing v1.2.4 changed the code
# hash/version bound by postrun-registration-v123.json, so postrun v1.2.3 now fails
# closed by design (asserted in tests/test_hb149_postrun_v124.py). Original file:
# .local/backups/astra-b5-H-B-149/v1.2.3/test_hb149_postrun_v123.py
pytestmark = pytest.mark.skip(reason="superseded by postrun v1.2.4; v1.2.3 fails closed on pairing v1.2.4")


def result():
    variants = {}
    for name in p.pair.VARIANTS:
        variants[name] = dict(draws=2000, seed=p.pair.SEED,
            macro={"delta_auc": {"value": .002, "ci95": [.001, .003], "valid_bootstraps": 2000}},
            weak_four={"delta_auc": {"value": -.0001}},
            per_target=[dict(target=t, delta_auc={"value": .002}) for t in p.pair.base.LABELS],
            runtime=p.pair.runtime_estimate(1300, name), go_gate="HOLD" if name == "full_r7_flip" else "GO")
    return dict(**p.pair.base.EXPOSURE, release_authorized=False, variants=variants)


def test_registered_card_and_engineering_never_advances():
    value = result()
    assert p.decision_lines(value)[-1] == "SUBMIT_CANDIDATE=FLIP_F0F2"
    assert p.decision_lines(value, True)[-1] == "SUBMIT_CANDIDATE=NONE"
    assert "runtime_h=8.886365" in p.decision_lines(value)[0]
    assert "weak_four_delta=-0.000100" in p.decision_lines(value)[1]  # descriptive, not a new gate


@pytest.mark.parametrize("failure", ["floor", "macro", "ci", "draws", "undefined"])
def test_violations_block_candidate(failure):
    value = result()
    s = value["variants"]["r7_flip_f0_f2"]
    if failure == "floor": s["per_target"][0]["delta_auc"]["value"] = -.005001
    if failure == "undefined": s["per_target"][0]["delta_auc"]["value"] = None
    if failure == "macro": s["macro"]["delta_auc"]["value"] = .00099
    if failure == "ci": s["macro"]["delta_auc"]["ci95"][0] = 0
    if failure == "draws": s["macro"]["delta_auc"]["valid_bootstraps"] = 1899
    with pytest.raises(ValueError, match="disagrees"):
        p.decision_lines(value)  # forged/stale GO string cannot bypass the floor
    s["go_gate"] = "HOLD"
    assert p.decision_lines(value)[-1] == "SUBMIT_CANDIDATE=NONE"


def test_floor_equality_passes():
    value = result()
    value["variants"]["r7_flip_f0_f2"]["per_target"][0]["delta_auc"]["value"] = -.005
    assert p.decision_lines(value)[-1] == "SUBMIT_CANDIDATE=FLIP_F0F2"


def test_registration_tamper_blocks(tmp_path, monkeypatch):
    r = p.pair.read_json(p.REGISTRATION)
    r["per_target_delta_min"] = -.01
    path = tmp_path / "registration.json"
    p.output_io.atomic_json(path, r)
    monkeypatch.setattr(p, "REGISTRATION", path)
    with pytest.raises(ValueError, match="registered GO"):
        p.decision_lines(result())


@pytest.mark.parametrize("field", ["historical_registration_sha256", "source_identity", "production_identity", "code_sha256"])
def test_amendment_tamper_blocks(tmp_path, monkeypatch, field):
    amendment = p.pair.read_json(p.AMENDMENT)
    if field == "code_sha256":
        amendment[field]["tools/hb149_r7_pairing.py"] = "0" * 64
    else:
        amendment[field] = "0" * 64
    path = tmp_path / "amendment.json"
    p.output_io.atomic_json(path, amendment)
    monkeypatch.setattr(p, "AMENDMENT", path)
    with pytest.raises(ValueError, match="differs"):
        p.decision_lines(result())


@pytest.fixture
def numeric(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    monkeypatch.setattr(p.pair, "SCRATCH", tmp_path)
    source = tmp_path / p.SOURCE
    source.parent.mkdir(parents=True)
    def put(rows, fields=None):
        buf = io.StringIO(newline="")
        wr = csv.writer(buf)
        wr.writerow(fields or [p.pair.base.UID, *p.pair.base.LABELS])
        wr.writerows(rows)
        source.write_text(buf.getvalue(), encoding="utf-8")
        monkeypatch.setattr(p, "SOURCE_SHA", p.sha(source))
        config = tmp_path / "config.json"
        p.output_io.atomic_json(config, {"config": {"labels": {"path": p.SOURCE}},
                                       "inputs": {"data": {p.SOURCE: p.SOURCE_SHA}}})
        monkeypatch.setattr(p, "CONFIG", config)
    put([["a", *([.5] * 12)]])
    selection = tmp_path / "selection.json"
    p.output_io.atomic_json(selection, {"uids": ["a"]})
    return selection, tmp_path / "output", put, source


def test_export_hash_coverage_and_retry(numeric, monkeypatch):
    selection, out, _, _ = numeric
    replace = p.output_io.os.replace
    attempts = []
    def locked_once(*args):
        attempts.append(1)
        if len(attempts) == 1: raise PermissionError("locked")
        return replace(*args)
    monkeypatch.setattr(p.output_io.os, "replace", locked_once)
    monkeypatch.setattr(p.output_io.time, "sleep", lambda _: None)
    dest, receipt = p.export_l2(selection, out)
    assert receipt["export_sha256"] == p.sha(dest)
    assert receipt["missing_studies"] == 0 and len(attempts) >= 3
    assert "uids" not in receipt


@pytest.mark.parametrize("bad", ["missing", "duplicate", "text", "range", "hash", "schema"])
def test_export_fail_closed(numeric, bad):
    selection, out, put, source = numeric
    if bad == "missing": p.output_io.atomic_json(selection, {"uids": ["a", "b"]})
    if bad == "duplicate": put([["a", *([.5] * 12)]] * 2)
    if bad == "text": put([["a", "CONFIDENTIAL_SENTINEL", *([.5] * 11)]])
    if bad == "range": put([["a", *([1.1] * 12)]])
    if bad == "hash": source.write_text("changed")
    if bad == "schema": put([["a", *([.5] * 12), "CONFIDENTIAL_SENTINEL"]], [p.pair.base.UID, *p.pair.base.LABELS, "text"])
    with pytest.raises(ValueError) as exc:
        p.export_l2(selection, out)
    assert "CONFIDENTIAL_SENTINEL" not in str(exc.value)
    assert not (out / "l2-numeric.csv").exists()


def test_real_registered_export_cpu_smoke(tmp_path, monkeypatch):
    # Internal UID-only selection; no label rows printed or inspected by the agent.
    monkeypatch.setattr(p.pair, "SCRATCH", tmp_path)
    with (p.ROOT / p.SOURCE).open(encoding="utf-8-sig", newline="") as f:
        uids = [r[p.pair.base.UID] for r in csv.DictReader(f)][:20]
    selection = tmp_path / "selection.json"
    p.output_io.atomic_json(selection, {"uids": uids})
    _, receipt = p.export_l2(selection, tmp_path / "export")
    assert receipt["covered_studies"] == 20 and receipt["source_sha256"] == p.SOURCE_SHA


def test_real_twenty_study_postrun_cpu_smoke(monkeypatch):
    # Real end-to-end readout exercises a transient CSV sharing failure too.
    path_class = type(p.WORK)
    original_open = path_class.open
    failures = []
    def locked_once(path, *args, **kwargs):
        if path.name == "paired-readout-v123.pending.csv" and args and args[0] == "w" and not failures:
            failures.append(1)
            raise PermissionError("sharing violation")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(path_class, "open", locked_once)
    monkeypatch.setattr(p.output_io.time, "sleep", lambda _: None)
    path = p.WORK / "postrun_smoke_v123.py"
    spec = importlib.util.spec_from_file_location("postrun_smoke_v123", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = mod.smoke()
    assert actual["engineering_only"] is True and actual["draws"] == 2000
    assert failures == [1]
    output = p.WORK / "postrun-smoke-v123/output"
    receipt = p.pair.read_json(output / "postrun-receipt-v123.json")
    assert receipt["aggregate_csv_sha256"] == p.sha(output / "paired-readout-v123.csv")
