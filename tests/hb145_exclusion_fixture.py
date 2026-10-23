"""Test-only registration of synthetic D-B-177-style data exclusions (v1.0.0, 2026-10-14).

Never reads the registered full runs or real labels. Production constants are
replaced only through scoped monkeypatches.
"""
from tools import hb145_data_exclusions_v1 as ex

hb = ex.hb


def fail_journal(run_dir, uid, error=ex.ERROR):
    path = hb.record_path(run_dir, uid)
    rec = hb.output_io.read_json(path)
    rec.update(status="failure", error=error, predictions={})
    hb.atomic_json(path, rec)
    return path


def prediction_count(run_dir, uids):
    return sum(len(hb.output_io.read_json(hb.record_path(run_dir, u)).get("predictions") or {}) for u in uids)


def complete_receipt(run_dir, failures):
    """Make the fixture receipt look like the real COMPLETED dual-role receipt."""
    identity = hb.output_io.read_json(run_dir / "identity.json")
    path = run_dir / "run_receipt.json"
    receipt = hb.output_io.read_json(path)
    receipt.update(failures=failures, models="heldout,f0", resources={"state": "COMPLETED"},
                   unique_model_predictions=prediction_count(run_dir, identity["uids"]),
                   decode_failures=0, exit_code=0)
    hb.atomic_json(path, receipt)
    return receipt


def register(run_dir, uids, path, monkeypatch, record=None):
    record = record or ex.build_record(run_dir, uids, "2026-10-14T00:00:00+00:00", "fixture-only")
    hb.atomic_json(path, record)
    monkeypatch.setattr(ex, "PATH", path)
    monkeypatch.setattr(ex, "REGISTERED_SHA256", ex.sha(path))
    monkeypatch.setattr(ex, "REGISTERED_UIDS", tuple(sorted(e["uid"] for e in record["exclusions"])))
    return record
