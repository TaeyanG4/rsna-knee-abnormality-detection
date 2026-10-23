"""CPU-only v1.2.4 postrun contracts (2026-10-14); never access running full outputs.

Labelled-intersection export/readout, unchanged GO card, registration binding of
the label-availability and D-B-177 data exclusions, v1.2.3 supersession.
"""
import csv
import importlib.util
import io

import pytest
from tools import hb149_postrun_v124 as p


def result(evaluated=989):
    variants = {}
    for name in p.pair.VARIANTS:
        variants[name] = dict(draws=2000, seed=p.pair.SEED,
            macro={"delta_auc": {"value": .002, "ci95": [.001, .003], "valid_bootstraps": 2000}},
            weak_four={"delta_auc": {"value": -.0001}},
            per_target=[dict(target=t, delta_auc={"value": .002}) for t in p.pair.base.LABELS],
            runtime=p.pair.runtime_estimate(1300, name), go_gate="HOLD" if name == "full_r7_flip" else "GO")
    return dict(**p.pair.base.EXPOSURE, release_authorized=False, variants=variants,
                evaluated_studies=evaluated, selected_studies=1000)


def test_registered_card_and_engineering_never_advances():
    value = result()
    assert p.decision_lines(value)[-1] == "SUBMIT_CANDIDATE=FLIP_F0F2"
    assert p.decision_lines(value, True)[-1] == "SUBMIT_CANDIDATE=NONE"
    assert "runtime_h=8.886365" in p.decision_lines(value)[0]
    assert "evaluated_studies=989/1000" in p.decision_lines(value)[1]
    assert "weak_four_delta=-0.000100" in p.decision_lines(value)[1]  # descriptive, not a new gate


@pytest.mark.parametrize("failure", ["floor", "macro", "ci", "draws", "undefined", "runtime"])
def test_violations_block_candidate(failure):
    value = result()
    s = value["variants"]["r7_flip_f0_f2"]
    if failure == "floor": s["per_target"][0]["delta_auc"]["value"] = -.005001
    if failure == "undefined": s["per_target"][0]["delta_auc"]["value"] = None
    if failure == "macro": s["macro"]["delta_auc"]["value"] = .00099
    if failure == "ci": s["macro"]["delta_auc"]["ci95"][0] = 0
    if failure == "draws": s["macro"]["delta_auc"]["valid_bootstraps"] = 1899
    if failure == "runtime":
        s["runtime"] = dict(s["runtime"], within_go_budget=False)
        with pytest.raises(ValueError, match="runtime"):
            p.decision_lines(value)
        return
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


@pytest.mark.parametrize("field", ["historical_registration_sha256", "source_identity", "production_identity",
                                   "code_sha256", "label_exclusion_sha256", "data_exclusion_sha256",
                                   "flip_output_present_at_registration", "version", "decision_policy_unchanged"])
def test_amendment_tamper_blocks(tmp_path, monkeypatch, field):
    amendment = p.pair.read_json(p.AMENDMENT)
    if field == "code_sha256":
        amendment[field]["tools/hb145_data_exclusions_v1.py"] = "0" * 64
    elif field == "flip_output_present_at_registration":
        amendment[field] = True
    elif field == "decision_policy_unchanged":
        amendment[field] = False
    elif field == "version":
        amendment[field] = "1.2.3"
    else:
        amendment[field] = "0" * 64
    path = tmp_path / "amendment.json"
    p.output_io.atomic_json(path, amendment)
    monkeypatch.setattr(p, "AMENDMENT", path)
    with pytest.raises(ValueError, match="differs"):
        p.decision_lines(result())


def test_label_exclusion_record_change_blocks(tmp_path, monkeypatch):
    record = p.pair.read_json(p.LABEL_EXCLUSION)
    record["uids"] = record["uids"][:-1]
    path = tmp_path / "label-exclusion.json"
    p.output_io.atomic_json(path, record)
    monkeypatch.setattr(p, "LABEL_EXCLUSION", path)
    with pytest.raises(ValueError, match="registration lineage"):
        p.registration()


def test_v123_postrun_now_fails_closed():
    # Superseded entry point: its registered pairing version/hash no longer match.
    from tools import hb149_postrun_v123 as v123
    with pytest.raises(ValueError, match="differs"):
        v123.registration()


@pytest.fixture
def numeric(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    monkeypatch.setattr(p.pair, "SCRATCH", tmp_path)
    source = tmp_path / p.SOURCE
    source.parent.mkdir(parents=True)
    selection = tmp_path / "selection.json"
    p.output_io.atomic_json(selection, {"uids": ["a", "b", "c"]})
    exclusion = tmp_path / "label-exclusion.json"

    def register(uids, **changes):
        value = dict(schema=p.pair.LABEL_EXCLUSION_SCHEMA, version=p.pair.LABEL_EXCLUSION_VERSION,
                     selection_sha256=p.sha(selection), selected_studies=3, labelled_studies=3 - len(uids),
                     count=len(uids), uids=list(uids), decided_before_flip_result=True,
                     gate_policy_unchanged=True, source_sha256=p.SOURCE_SHA, reason="fixture")
        value.update(changes)
        p.output_io.atomic_json(exclusion, value)

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
        register(["c"])
    put([["a", *([.5] * 12)], ["b", *([.25] * 12)], ["z", *([.5] * 12)]])
    return selection, tmp_path / "output", put, source, exclusion, register


def test_export_labelled_intersection_and_retry(numeric, monkeypatch):
    selection, out, _, _, exclusion, _ = numeric
    replace = p.output_io.os.replace
    attempts = []
    def locked_once(*args):
        attempts.append(1)
        if len(attempts) == 1: raise PermissionError("locked")
        return replace(*args)
    monkeypatch.setattr(p.output_io.os, "replace", locked_once)
    monkeypatch.setattr(p.output_io.time, "sleep", lambda _: None)
    dest, receipt = p.export_l2(selection, out, exclusion)
    assert receipt["export_sha256"] == p.sha(dest) and receipt["status"] == "READY_LABELLED_INTERSECTION"
    assert (receipt["selected_studies"], receipt["covered_studies"], receipt["missing_studies"],
            receipt["registered_label_exclusions"]) == (3, 2, 1, 1)
    assert receipt["label_exclusion_sha256"] == p.sha(exclusion)
    assert "uids" not in receipt and len(attempts) >= 3
    with dest.open(newline="") as f:
        assert [r[p.pair.base.UID] for r in csv.DictReader(f)] == ["a", "b"]


@pytest.mark.parametrize("bad", ["unregistered-missing", "stale-exclusion", "other-source", "duplicate", "text",
                                 "range", "hash", "schema", "not-before-flip"])
def test_export_fail_closed(numeric, bad):
    selection, out, put, source, exclusion, register = numeric
    if bad == "unregistered-missing": put([["a", *([.5] * 12)]])  # b now also missing; only c registered
    if bad == "stale-exclusion": put([[u, *([.5] * 12)] for u in "abc"])  # c labelled yet excluded
    if bad == "other-source": register(["c"], source_sha256="0" * 64)
    if bad == "duplicate": put([["a", *([.5] * 12)]] * 2)
    if bad == "text": put([["a", "CONFIDENTIAL_SENTINEL", *([.5] * 11)]])
    if bad == "range": put([["a", *([1.1] * 12)]])
    if bad == "hash": source.write_text("changed")
    if bad == "schema": put([["a", *([.5] * 12), "CONFIDENTIAL_SENTINEL"]], [p.pair.base.UID, *p.pair.base.LABELS, "text"])
    if bad == "not-before-flip": register(["c"], decided_before_flip_result=False)
    with pytest.raises(ValueError) as exc:
        p.export_l2(selection, out, exclusion)
    assert "CONFIDENTIAL_SENTINEL" not in str(exc.value)
    assert not (out / "l2-numeric.csv").exists()
    if bad in ("unregistered-missing", "stale-exclusion"):
        assert p.pair.read_json(out / "l2-export-receipt.json")["status"] == "COVERAGE_BLOCKED"


def test_real_registered_intersection_export_cpu(tmp_path, monkeypatch):
    # Real registered source, selection and exclusion; aggregate counts only.
    monkeypatch.setattr(p.pair, "SCRATCH", tmp_path)
    _, receipt = p.export_l2(p.pair.SELECTION, tmp_path / "export", p.LABEL_EXCLUSION)
    assert (receipt["selected_studies"], receipt["covered_studies"], receipt["missing_studies"]) == (1000, 989, 11)
    assert receipt["status"] == "READY_LABELLED_INTERSECTION" and receipt["source_sha256"] == p.SOURCE_SHA
    assert receipt["selection_sha256"] == p.SELECTION_SHA


def test_real_twenty_study_postrun_cpu_smoke(monkeypatch):
    # Real end-to-end readout on the intersection; exercises a transient CSV sharing failure too.
    path_class = type(p.WORK)
    original_open = path_class.open
    failures = []
    def locked_once(path, *args, **kwargs):
        if path.name == "paired-readout-v124.pending.csv" and args and args[0] == "w" and not failures:
            failures.append(1)
            raise PermissionError("sharing violation")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(path_class, "open", locked_once)
    monkeypatch.setattr(p.output_io.time, "sleep", lambda _: None)
    path = p.WORK / "postrun_smoke_v124.py"
    spec = importlib.util.spec_from_file_location("postrun_smoke_v124", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = mod.smoke()
    assert actual["engineering_only"] is True and actual["draws"] == 2000
    assert actual["evaluated_studies"] == actual["selected_studies"] - mod.UNLABELLED
    assert failures == [1]
    output = p.WORK / "postrun-smoke-v124/output"
    receipt = p.pair.read_json(output / "postrun-receipt-v124.json")
    assert receipt["aggregate_csv_sha256"] == p.sha(output / "paired-readout-v124.csv")
    assert receipt["label_availability_exclusion"]["count"] == mod.UNLABELLED
