"""H-B-146 1.1.1 CPU seal for H-B-145 dual-role f0 outputs.

Changelog 2026-10-13 v1.1.1: final producer schema, release-device auxiliaries,
strict EXPOSED provenance at every layer; bind monitor and receipt identities.
The v1.0 teacher adapter is preserved. No inference is performed here.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb146_pl_labels_v111 as pl
from tools import hb145_r7_heldout_v1 as hb
from tools import hb145_r7_pipeline_v111 as pipeline
from tools import hb145_r7_monitor_v111 as monitor


def require_roles(value):
    roles = value.get("role_provenance", {})
    pl.need(set(roles) == {"heldout_raw", "heldout_post", "f0_raw", "f0_post"},
            "teacher role provenance incomplete")
    for name, entry in roles.items():
        pl.require_teacher_exposure(entry)
        expected = "not_applied" if name.endswith("_raw") else False
        actual = entry.get("stackers_cross_fitted")
        pl.need(type(actual) is type(expected) and actual == expected, "role stacker provenance differs")


def seal(source, out=None, *, exposed_exploratory=False):
    pl.need(exposed_exploratory, "explicit --exposed-exploratory is required")
    source = Path(source)
    out = Path(out) if out else pl.WORK / "teacher-f0.json"
    pl.need(out.resolve().is_relative_to(pl.WORK.resolve()), "receipt must remain in H-B-146 scratch")
    identity = pl.read(source / "identity.json")
    run = pl.read(source / "run_receipt.json")
    worker = pl.read(source / "receipt_dual.json")
    for record in (identity, run, worker):
        pl.require_teacher_exposure(record)
        require_roles(record)
        pl.need(record.get("version") == "1.1.1", "H-B-145 v1.1.1 required")
    fingerprint = hb.fingerprint(identity)
    device = identity.get("auxiliary_device")
    pl.need(device in ("cpu", "cuda") and identity.get("device") == device
            and all(r.get("identity") == fingerprint and r.get("auxiliary_device") == device
                    for r in (run, worker)), "teacher identity/device differs")
    csv_path = source / "f0_raw.csv"
    values = pl.table(csv_path)
    pl.need(identity.get("version") == "1.1.1" and identity.get("models") == "heldout,f0",
            "H-B-145 v1.1.1 dual-role identity required")
    pl.need(set(values) == set(identity["uids"]) and len(values) == len(identity["uids"]), "teacher roster differs")
    pl.need(run.get("completed") == run.get("selected") == len(values) and len(values) > 0
            and run.get("failures") == 0 and run.get("exit_code") == 0
            and run.get("models") == "heldout,f0", "teacher run incomplete")
    pl.need(worker.get("models") == hb.r7.MODELS, "dual model receipt differs")
    runtime = source / "stage/r7_runtime/compact_infer.py"
    pl.need(identity.get("runner_sha256") == pl.sha(hb.__file__)
            and identity.get("pipeline_sha256") == pl.sha(pipeline.__file__)
            and identity.get("monitor_sha256") == pl.sha(monitor.__file__)
            and identity.get("runtime_sha256") == pl.sha(runtime)
            and all(identity.get(k) for k in ("bundle", "code", "deps")), "teacher code identity differs")
    columns = [pl.UID, "fold_k", "status", "model_role", "model_fold", "output_kind",
               "pooled_prior_bias", "stackers_cross_fitted", "prediction_reference", "roles_coincide",
               "auxiliary_device", *pl.TEACHER_EXPOSURE]
    meta = pl.pd.read_csv(csv_path, usecols=columns, dtype={pl.UID: str}).set_index(pl.UID)
    exposure = pl.pd.read_csv(csv_path, usecols=[pl.UID, *pl.TEACHER_EXPOSURE],
                             dtype=str, keep_default_na=False).set_index(pl.UID)
    pl.need(meta.index.is_unique and set(meta.index) == set(values), "metadata roster differs")
    paths = [csv_path, source / "identity.json", source / "run_receipt.json",
             source / "receipt_dual.json", Path(hb.__file__), Path(pipeline.__file__),
             Path(monitor.__file__), runtime, Path(__file__), Path(pl.__file__)]
    for uid, value in values.items():
        path = hb.record_path(source, uid)
        rec = pl.read(path)
        pl.require_teacher_exposure(rec)
        pl.require_teacher_exposure(exposure.loc[uid].to_dict(), csv=True)
        pipeline.validate_record(rec)
        row = meta.loc[uid]
        pl.need(rec.get(pl.UID) == uid and rec.get("identity") == hb.fingerprint(identity)
                and rec.get("fold_k") == pl.hash_fold(uid) and row["fold_k"] == pl.hash_fold(uid)
                and rec.get("models") == "heldout,f0" and rec.get("auxiliary_device") == device
                and row["auxiliary_device"] == device
                and rec.get("status") in ("ok", "dropped series") and row["status"] == rec["status"]
                and rec.get("pooled_prior_bias") is True and isinstance(row["pooled_prior_bias"], (bool, pl.np.bool_))
                and row["pooled_prior_bias"] == True
                and row["model_role"] == "f0" and row["model_fold"] == 0
                and row["output_kind"] == "pre_stacker" and row["stackers_cross_fitted"] == "not_applied"
                and isinstance(row["roles_coincide"], (bool, pl.np.bool_))
                and row["roles_coincide"] == (pl.hash_fold(uid) == 0)
                and row["prediction_reference"] == f"{path.name}#predictions/0", "teacher journal metadata differs")
        raw = pl.np.asarray(rec["predictions"]["0"]["raw"], dtype=pl.np.float32)
        pl.need(pl.np.array_equal(raw, value), "teacher journal probabilities differ")
        paths.append(path)
    receipt = dict(version=pl.VERSION, mode="screen_f0", model_selection="f0_only", output_kind="pre_stacker",
                   **pl.TEACHER_EXPOSURE, producer_version="1.1.1", producer_identity=fingerprint,
                   auxiliary_device=device, role_provenance=identity["role_provenance"],
                   csv_sha256=pl.sha(csv_path), complete=True, failures=0,
                   source_pool_membership="manifest_contract_unverified_membership",
                   validation_label_exposure="pooled_prior_not_clean", auxiliary_label_exposure="unknown",
                   oai_ancestry="teacher ancestry evidence partial: ImageNet init, fastMRI+, SKM-TEA segmenter",
                   status=pl.STATUS, limitations=pl.LIMITATIONS, exposed_exploratory=True, pins=pl.pins(paths))
    pl.write(out, receipt)
    return {"sealed": True, "rows": len(values), "status": pl.STATUS, "receipt": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seal"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exposed-exploratory", action="store_true")
    args = parser.parse_args(argv)
    print(seal(args.source, args.out, exposed_exploratory=args.exposed_exploratory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
