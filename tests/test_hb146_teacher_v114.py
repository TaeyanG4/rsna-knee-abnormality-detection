"""Seal v1.1.4 (2026-10-14): synthetic dual-role contracts with registered D-B-177 exclusions.

No real label rows, no registered full run. The v1.1.4 launch adapter's PL module
is loaded under a unique name so it cannot collide with the v1.1.3 adapter.
"""
import csv
import importlib.util
from pathlib import Path

import pytest
from tools import hb146_teacher_v114 as teacher
from tools import hb145_data_exclusions_v1 as ex
from hb145_exclusion_fixture import register

hb, pipeline = teacher.hb, teacher.pipeline
ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / ".local/scratch/astra-b4/H-B-146/launch-adapter-v114/launch_pl_labels_v111.py"
_spec = importlib.util.spec_from_file_location("hb146_launch_pl_labels_adapter_v114", ADAPTER)
apl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apl)
UIDS = ["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d"]


def make_source(tmp_path, monkeypatch, failed=(), errors=None):
    monkeypatch.setattr(hb, "fold_k", apl.hash_fold)
    monkeypatch.setattr(apl, "WORK", tmp_path / "work")
    source = tmp_path / "teacher"
    runtime = source / "stage/r7_runtime/compact_infer.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("# synthetic runtime\n")
    provenance = dict(**hb.EXPOSURE, role_provenance=hb.role_provenance("heldout,f0"))
    identity = dict(**provenance, version="1.1.1", models="heldout,f0", uids=UIDS,
                    auxiliary_device="cuda", device="cuda", monitor_sha256=apl.sha(teacher.monitor.__file__),
                    runner_sha256=apl.sha(hb.__file__), pipeline_sha256=apl.sha(pipeline.__file__),
                    runtime_sha256=apl.sha(runtime), bundle="b", code="c", deps="d")
    apl.write(source / "identity.json", identity)
    records = {}
    for i, uid in enumerate(UIDS):
        k = apl.hash_fold(uid)
        raw = [.2 + .15 * i] * 12
        rec = {**hb.EXPOSURE, apl.UID: uid, "fold_k": k, "identity": hb.fingerprint(identity), "status": "ok",
               "pooled_prior_bias": True, "models": "heldout,f0", "auxiliary_device": "cuda",
               "role_folds": {"heldout": k, "f0": 0}, "roles_coincide": k == 0, "error": "",
               "predictions": {str(f): {"raw": raw, "post": raw} for f in pipeline.required_folds(uid)}}
        if uid in failed:
            rec.update(status="failure", error=(errors or {}).get(uid, ex.ERROR), predictions={})
        records[uid] = rec
        apl.write(hb.record_path(source, uid), rec)
    pipeline.export_roles(source, UIDS, records)
    meta = dict(**provenance, version="1.1.1", identity=hb.fingerprint(identity), auxiliary_device="cuda")
    apl.write(source / "run_receipt.json", dict(**meta, completed=4, selected=4, failures=len(failed), exit_code=0,
              decode_failures=0, models="heldout,f0", resources={"state": "COMPLETED"},
              unique_model_predictions=sum(len(r["predictions"]) for r in records.values())))
    apl.write(source / "receipt_dual.json", dict(**meta, models=hb.r7.MODELS))
    from hb145_v113_fixture import migrate
    migrate(source, monkeypatch)
    return source


def seal(source, out=None):
    out = out or apl.WORK / "teacher-f0.json"
    return teacher.seal(source, out, exposed_exploratory=True, pl_module=apl)


def lines(path):
    return Path(path).read_bytes().splitlines(keepends=True)


def test_zero_failures_seal_is_byte_copy(tmp_path, monkeypatch):
    source = make_source(tmp_path, monkeypatch)
    result = seal(source)
    receipt = apl.read(apl.WORK / "teacher-f0.json")
    assert result["rows"] == 4 and result["excluded"] == 0
    assert (apl.WORK / "teacher-f0.csv").read_bytes() == (source / "f0_raw.csv").read_bytes()
    assert receipt["seal_version"] == "1.1.4" and receipt["data_exclusions"] is None
    assert receipt["producer_failures"] == 0 and receipt["failures"] == 0 and receipt["complete"] is True
    values, sealed = apl.validate_teacher(apl.WORK / "teacher-f0.csv", apl.WORK / "teacher-f0.json", UIDS)
    assert len(values) == 4 and apl.teacher_excluded(sealed) == set()


def test_registered_exclusion_sealed_without_teacher_target(tmp_path, monkeypatch):
    failed = [UIDS[1]]
    source = make_source(tmp_path, monkeypatch, failed)
    register(source, failed, tmp_path / "excl.json", monkeypatch)
    result = seal(source)
    assert result["rows"] == 3 and result["excluded"] == 1
    receipt = apl.read(apl.WORK / "teacher-f0.json")
    assert receipt["data_exclusions"]["uids"] == failed and receipt["data_exclusions"]["count"] == 1
    assert receipt["producer_failures"] == 1 and receipt["failures"] == 0 and receipt["teacher_rows"] == 3
    assert receipt["csv_sha256"] == apl.sha(apl.WORK / "teacher-f0.csv")
    assert receipt["source_csv_sha256"] == apl.sha(source / "f0_raw.csv")
    original, sealed = lines(source / "f0_raw.csv"), lines(apl.WORK / "teacher-f0.csv")
    assert len(sealed) == len(original) - 1 and all(line in original for line in sealed)
    assert not any(failed[0].encode() in line for line in sealed)
    values, r = apl.validate_teacher(apl.WORK / "teacher-f0.csv", apl.WORK / "teacher-f0.json", UIDS)
    assert set(values) == set(UIDS) - set(failed) and apl.teacher_excluded(r) == set(failed)
    assert str((tmp_path / "excl.json").resolve()) in receipt["pins"]
    seal(source)  # deterministic re-seal is idempotent


@pytest.mark.parametrize("case", ["unregistered", "other-error", "extra-failure", "listed-ok-row", "csv-row-tampered",
                                  "sealed-csv-differs", "receipt-count"])
def test_any_other_failure_rejected(tmp_path, monkeypatch, case):
    failed = [UIDS[1], UIDS[2]] if case == "extra-failure" else [UIDS[1]]
    errors = {UIDS[1]: "RuntimeError"} if case == "other-error" else None
    source = make_source(tmp_path, monkeypatch, failed, errors)
    if case == "extra-failure":
        register(source, failed[:1], tmp_path / "excl.json", monkeypatch,
                 ex.build_record(source, failed[:1], "t", "fixture-only"))
    elif case == "receipt-count":
        receipt = apl.read(source / "run_receipt.json")
        receipt["failures"] = 2
        apl.write(source / "run_receipt.json", receipt)
        register(source, failed, tmp_path / "excl.json", monkeypatch)
    elif case in ("listed-ok-row", "csv-row-tampered", "sealed-csv-differs"):
        register(source, failed, tmp_path / "excl.json", monkeypatch)
    if case in ("listed-ok-row", "csv-row-tampered"):
        path = source / "f0_raw.csv"
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            columns, rows = reader.fieldnames, list(reader)
        for row in rows:
            if row[apl.UID] == failed[0]:
                if case == "listed-ok-row":
                    row["status"] = "ok"
                else:
                    row["ACL"] = "0.5"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    if case == "sealed-csv-differs":
        (apl.WORK).mkdir(parents=True, exist_ok=True)
        (apl.WORK / "teacher-f0.csv").write_text("changed\n")
    reason = {"listed-ok-row": "empty failure row", "csv-row-tampered": "empty failure row",
              "sealed-csv-differs": "sealed teacher CSV differs"}.get(case, "only failures are the registered")
    with pytest.raises(ValueError, match=reason):
        seal(source)


def test_explicit_out_and_opt_in(tmp_path, monkeypatch):
    source = make_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicit"):
        teacher.seal(source, apl.WORK / "teacher-f0.json", pl_module=apl)
    with pytest.raises(ValueError, match="--out"):
        teacher.seal(source, None, exposed_exploratory=True, pl_module=apl)
    with pytest.raises(ValueError, match="scratch"):
        teacher.seal(source, tmp_path / "elsewhere.json", exposed_exploratory=True, pl_module=apl)


@pytest.mark.parametrize("layer", ["identity.json", "run_receipt.json", "receipt_dual.json", "journal", "csv"])
@pytest.mark.parametrize("mutation", ["clean", "true-clean"])
def test_exposure_fail_closed(tmp_path, monkeypatch, layer, mutation):
    source = make_source(tmp_path, monkeypatch)
    if layer == "csv":
        path = source / "f0_raw.csv"
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            columns, rows = reader.fieldnames, list(reader)
        rows[0]["exposure" if mutation == "clean" else "clean_oof"] = "CLEAN" if mutation == "clean" else "True"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    else:
        path = hb.record_path(source, UIDS[0]) if layer == "journal" else source / layer
        record = apl.read(path)
        record["exposure" if mutation == "clean" else "clean_oof"] = "CLEAN" if mutation == "clean" else True
        apl.write(path, record)
    with pytest.raises(ValueError):
        seal(source)


@pytest.mark.parametrize("mutation", ["seal-version", "exclusion-uids", "unregistered-producer-failures",
                                      "exclusion-sha", "csv-hash"])
def test_adapter_validate_teacher_fail_closed(tmp_path, monkeypatch, mutation):
    failed = [UIDS[1]]
    source = make_source(tmp_path, monkeypatch, failed)
    register(source, failed, tmp_path / "excl.json", monkeypatch)
    seal(source)
    receipt_path, csv_path = apl.WORK / "teacher-f0.json", apl.WORK / "teacher-f0.csv"
    r = apl.read(receipt_path)
    if mutation == "seal-version":
        r["seal_version"] = "1.1.3"
    elif mutation == "exclusion-uids":
        r["data_exclusions"]["uids"] = [UIDS[2]]
    elif mutation == "unregistered-producer-failures":
        r["data_exclusions"] = None
    elif mutation == "exclusion-sha":
        r["data_exclusions"]["sha256"] = "0" * 64
    else:
        csv_path.write_text(csv_path.read_text() + "\n")
    apl.write(receipt_path, r)
    with pytest.raises(ValueError):
        apl.validate_teacher(csv_path, receipt_path, UIDS)


def test_journal_tamper_and_device_mismatch(tmp_path, monkeypatch):
    source = make_source(tmp_path, monkeypatch)
    rec = apl.read(hb.record_path(source, UIDS[0]))
    rec["predictions"]["0"]["raw"][0] = .99
    apl.write(hb.record_path(source, UIDS[0]), rec)
    with pytest.raises(ValueError):
        seal(source)
