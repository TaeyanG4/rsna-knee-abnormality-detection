"""v1.2.4 (2026-10-14): registered data-exclusion baseline and labelled-intersection readout.

Synthetic fixtures only; never reads the registered full runs or real label rows.
"""
import csv

import numpy as np
import pytest
from test_hb149_r7_flip_v120 import paired_artifacts
from hb145_v113_fixture import migrate_pair
from hb145_exclusion_fixture import complete_receipt, fail_journal, prediction_count, register
from tools import hb149_r7_pairing as pair
from tools import hb145_data_exclusions_v1 as ex
from tools import hb145_r7_pipeline_v111 as oldpipe

old = pair.base


def complete_roster(baseline):
    """The shared fixture journals only the selection; commit the rest as ok."""
    for i, uid in enumerate(pair.read_json(baseline / "identity.json")["uids"]):
        if old.record_path(baseline, uid).exists():
            continue
        item = {"uid": uid}
        for k in oldpipe.required_folds(uid):
            item["hb145_active_fold"] = k
            oldpipe.capture(uid, item, [.2 + .6 * (i % 2)] * 12, [.3] * 12)
        oldpipe.commit(item)


def outside_and_inside(selection, baseline):
    """Latest roster entries first: fixture migration carries the first 16 journals,
    so mutating these never trips the (separately tested) carry-over contract."""
    selected = pair.read_json(selection)["uids"]
    roster = pair.read_json(baseline / "identity.json")["uids"]
    return [u for u in roster if u not in set(selected)][::-1], selected[::-1]


def build(artifacts, monkeypatch, failed_selected=0, error=ex.ERROR, register_uids=None):
    selection, baseline, *_ = artifacts
    complete_roster(baseline)
    outside, inside = outside_and_inside(selection, baseline)
    failed = outside[:2 - failed_selected] + inside[:failed_selected]
    for uid in failed:
        fail_journal(baseline, uid, error if uid == failed[0] else ex.ERROR)
    complete_receipt(baseline, len(failed))
    migrate_pair(artifacts, monkeypatch)
    path = artifacts[0].parent / "data-exclusions-fixture.json"
    if error == ex.ERROR:
        register(baseline, register_uids or failed, path, monkeypatch)
    return failed, path


@pytest.fixture
def excluded(paired_artifacts, monkeypatch):
    failed, path = build(paired_artifacts, monkeypatch)
    return paired_artifacts, failed, path


def entry(name, artifacts):
    selection, baseline, *_ = artifacts
    if name == "validate":
        return pair.main(["validate", "--selection", str(selection), "--baseline", str(baseline)])
    if name == "readout":
        return pair.readout(*artifacts, draws=100)
    return pair.selected_uids(selection, baseline, pair.read_json(baseline / "identity.json")["uids"],
                              require_complete=True)


@pytest.mark.parametrize("name", ["validate", "readout", "inference"])
def test_exact_registered_exclusions_accepted_outside_selection(excluded, name):
    artifacts, failed, path = excluded
    selection, baseline, *_ = artifacts
    assert pair.read_json(baseline / "run_receipt.json")["failures"] == 2
    record, _, status = pair.baseline_status(selection, baseline)
    assert status["data_exclusions"] == {"accepted": True, "count": 2, "error": "BadZipFile",
                                         "reason": ex.REASON, "sha256": ex.sha(path), "in_frozen_selection": 0}
    assert not set(failed) & set(record["uids"])
    result = entry(name, artifacts)
    if name == "readout":
        assert result["baseline_data_exclusions"]["count"] == 2
        assert result["evaluated_studies"] == result["selected_studies"] == 48
        assert result["label_availability_exclusion"] is None
    elif name == "validate":
        assert result == 0


def mutate(artifacts, failed, path, monkeypatch, mutation):
    selection, baseline, *_ = artifacts
    receipt_path = baseline / "run_receipt.json"
    if mutation == "unregistered-record":
        record = pair.read_json(path)
        record["reason"] = "other"
        old.atomic_json(path, record)  # hash no longer the registered one
    elif mutation == "missing-record":
        path.unlink()
    elif mutation == "other-error-type":
        jp = old.record_path(baseline, failed[0])
        rec = pair.read_json(jp)
        rec["error"] = "RuntimeError"
        old.atomic_json(jp, rec)
        record = pair.read_json(path)
        record["exclusions"] = [dict(e, journal_sha256=ex.sha(jp)) if e["uid"] == failed[0] else e
                                for e in record["exclusions"]]
        register(baseline, failed, path, monkeypatch, record)
    elif mutation == "journal-hash":
        jp = old.record_path(baseline, failed[0])
        rec = pair.read_json(jp)
        rec["dicom_files"] = 1
        old.atomic_json(jp, rec)
    elif mutation in ("extra-failure", "extra-failure-registered-count"):
        outside, _ = outside_and_inside(selection, baseline)
        extra = next(u for u in outside if u not in failed)
        fail_journal(baseline, extra)
        receipt = pair.read_json(receipt_path)
        receipt["failures"] = 3 if mutation == "extra-failure" else 2
        receipt["unique_model_predictions"] = prediction_count(
            baseline, pair.read_json(baseline / "identity.json")["uids"])
        old.atomic_json(receipt_path, receipt)
        register(baseline, failed, path, monkeypatch, ex.build_record(baseline, failed, "t", "fixture-only"))
    elif mutation == "listed-not-failed":
        record = pair.read_json(path)
        outside, _ = outside_and_inside(selection, baseline)
        ok = next(u for u in outside if u not in failed)
        record["exclusions"].append(dict(record["exclusions"][0], uid=ok, journal=old.record_path(baseline, ok).name))
        record["count"] = 3
        register(baseline, failed, path, monkeypatch, record)
    else:
        receipt = pair.read_json(receipt_path)
        if mutation == "receipt-count":
            receipt["failures"] = 1
        elif mutation == "not-completed":
            receipt["resources"]["state"] = "STALLED"
        elif mutation == "decode-failures":
            receipt["decode_failures"] = 1
        elif mutation == "exit-code":
            receipt["exit_code"] = 1
        elif mutation == "prediction-count":
            receipt["unique_model_predictions"] += 1
        elif mutation == "bool-failures":
            receipt["failures"] = True
        old.atomic_json(receipt_path, receipt)
        # Re-register so only the intended receipt check can refuse.
        register(baseline, failed, path, monkeypatch, ex.build_record(baseline, failed, "t", "fixture-only"))


@pytest.mark.parametrize("name", ["validate", "readout", "inference"])
@pytest.mark.parametrize("mutation", ["unregistered-record", "missing-record", "other-error-type", "journal-hash",
                                      "extra-failure", "extra-failure-registered-count", "listed-not-failed",
                                      "receipt-count", "not-completed", "decode-failures", "exit-code",
                                      "prediction-count", "bool-failures"])
def test_any_other_failure_still_rejects(excluded, monkeypatch, name, mutation):
    artifacts, failed, path = excluded
    mutate(artifacts, failed, path, monkeypatch, mutation)
    with pytest.raises(ValueError, match="full must complete"):
        entry(name, artifacts)


REASONS = {"unregistered-record": "differs from registration", "missing-record": "differs from registration",
           "other-error-type": "exclusion journal differs", "journal-hash": "exclusion journal differs",
           "extra-failure": "only failures are the registered", "extra-failure-registered-count": "run failures differ",
           "listed-not-failed": "only failures are the registered", "receipt-count": "only failures are the registered",
           "not-completed": "only failures are the registered", "decode-failures": "only failures are the registered",
           "exit-code": "only failures are the registered", "prediction-count": "prediction count differs",
           "bool-failures": "only failures are the registered"}


@pytest.mark.parametrize("mutation", sorted(REASONS))
def test_each_rejection_hits_its_intended_check(excluded, monkeypatch, mutation):
    artifacts, failed, path = excluded
    mutate(artifacts, failed, path, monkeypatch, mutation)
    baseline = artifacts[1]
    identity, receipt = (pair.read_json(baseline / n) for n in ("identity.json", "run_receipt.json"))
    with pytest.raises(ValueError, match=REASONS[mutation]):
        ex.verify_run(baseline, identity, receipt)


def test_other_failure_type_never_registrable(paired_artifacts, monkeypatch):
    # A non-BadZipFile failure cannot even be described as a registered exclusion.
    failed, path = build(paired_artifacts, monkeypatch, error="RuntimeError")
    with pytest.raises(ValueError, match="registered-type"):
        ex.build_record(paired_artifacts[1], failed, "t", "fixture-only")
    with pytest.raises(ValueError, match="full must complete"):
        pair.baseline_ready(*paired_artifacts[:2])


@pytest.mark.parametrize("name", ["validate", "readout", "inference"])
def test_registered_exclusion_inside_selection_rejected(paired_artifacts, monkeypatch, name):
    failed, _ = build(paired_artifacts, monkeypatch, failed_selected=1)
    selection = pair.read_json(paired_artifacts[0])["uids"]
    assert sum(u in selection for u in failed) == 1
    with pytest.raises(ValueError, match="intersects the frozen selection"):
        entry(name, paired_artifacts)


def test_zero_failure_baseline_unchanged(paired_artifacts, monkeypatch):
    complete_roster(paired_artifacts[1])
    complete_receipt(paired_artifacts[1], 0)
    migrate_pair(paired_artifacts, monkeypatch)
    _, _, status = pair.baseline_status(*paired_artifacts[:2])
    assert status["producer_failures"] == 0 and status["data_exclusions"] is None


# ---- labelled intersection -------------------------------------------------

def drop_labels(artifacts, count):
    selection, _, _, labels, _ = artifacts
    with labels.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    drop = sorted(r[old.UID] for r in rows)[:count]
    with labels.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in rows if r[old.UID] not in drop)
    return drop, rows


def exclusion_record(artifacts, uids, **changes):
    selection = artifacts[0]
    selected = pair.read_json(selection)["uids"]
    value = dict(schema=pair.LABEL_EXCLUSION_SCHEMA, version=pair.LABEL_EXCLUSION_VERSION,
                 selection_sha256=old.r7.s6.sha256(selection), selected_studies=len(selected),
                 labelled_studies=len(selected) - len(uids), count=len(uids), uids=list(uids),
                 decided_before_flip_result=True, gate_policy_unchanged=True, source_sha256="fixture",
                 reason="fixture")
    value.update(changes)
    path = artifacts[0].parent / "label-exclusion.json"
    old.atomic_json(path, value)
    return path


@pytest.fixture
def ready(paired_artifacts, monkeypatch):
    return migrate_pair(paired_artifacts, monkeypatch)


def test_labelled_intersection_readout_uses_unchanged_rules(ready):
    drop, rows = drop_labels(ready, 3)
    path = exclusion_record(ready, drop)
    result = pair.readout(*ready, draws=100, label_exclusion=path)
    assert result["selected_studies"] == 48 and result["evaluated_studies"] == 45 and result["studies"] == 45
    assert result["label_availability_exclusion"]["count"] == 3
    assert result["label_availability_exclusion"]["sha256"] == old.r7.s6.sha256(path)
    assert "uids" not in result and "uids" not in result["label_availability_exclusion"]
    # Identical to the unchanged registered policy applied to the intersection directly.
    selected = pair.read_json(ready[0])["uids"]
    keep = [u for u in selected if u not in drop]
    plain = pair.predictions(ready[1] / "heldout_raw.csv", selected, "heldout")
    flip = pair.predictions(ready[2] / "heldout_flip_raw.csv", selected, "heldout_flip")
    labels = {r[old.UID]: [float(r[t]) for t in old.LABELS] for r in rows}
    direct = pair.variant_readout([labels[u] for u in keep], [plain[u] for u in keep], [flip[u] for u in keep],
                                  [old.fold_k(u) for u in keep], draws=100)
    for name in pair.VARIANTS:
        assert result["variants"][name]["macro"] == direct["variants"][name]["macro"]
        assert result["variants"][name]["go_gate"] == direct["variants"][name]["go_gate"]


def test_missing_labels_without_registered_exclusion_still_blocks(ready):
    drop_labels(ready, 3)
    with pytest.raises(ValueError, match="missing paired L2"):
        pair.readout(*ready, draws=100)


@pytest.mark.parametrize("mutation", ["stale", "too-few", "outside-selection", "selection-hash", "not-before-flip",
                                      "policy-changed", "count", "duplicate", "version", "all-studies"])
def test_label_exclusion_fail_closed(ready, mutation):
    drop, _ = drop_labels(ready, 3)
    uids, changes = list(drop), {}
    if mutation == "stale":
        uids.append(next(u for u in pair.read_json(ready[0])["uids"] if u not in drop))
        changes = dict(count=4, labelled_studies=44)
    elif mutation == "too-few":
        uids, changes = uids[:2], dict(count=2, labelled_studies=46)
    elif mutation == "outside-selection":
        uids[0] = "not-selected"
    elif mutation == "selection-hash":
        changes = dict(selection_sha256="0" * 64)
    elif mutation == "not-before-flip":
        changes = dict(decided_before_flip_result=False)
    elif mutation == "policy-changed":
        changes = dict(gate_policy_unchanged=False)
    elif mutation == "count":
        changes = dict(count=2)
    elif mutation == "duplicate":
        uids[1] = uids[0]
    elif mutation == "version":
        changes = dict(version="1.2.3")
    else:
        uids = pair.read_json(ready[0])["uids"]
        changes = dict(count=48, labelled_studies=0)
    path = exclusion_record(ready, uids, **changes)
    with pytest.raises(ValueError):
        pair.readout(*ready, draws=100, label_exclusion=path)


def test_intersection_floor_violation_blocks_go():
    # Registered floor stays enforced on whatever labelled intersection is scored.
    y = np.tile(np.arange(80)[:, None] % 2, (1, 12)).astype(float)
    plain, flip = 1 - y, y.copy()
    plain[:, 0], flip[:, 0] = y[:, 0], 1 - y[:, 0]
    result = pair.variant_readout(y[:70], plain[:70], flip[:70], np.full(70, 2), draws=100)
    assert result["go_variants"] == []
    assert result["variants"]["r7_flip_f0_f2"]["target_floor"]["passed"] is False
