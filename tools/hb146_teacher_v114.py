"""H-B-146 1.1.4 CPU seal for H-B-145 dual-role f0 outputs.

Changelog 2026-10-14 v1.1.4: accept the COMPLETED H-B-145 v1.1.3 run whose ONLY
failures are exactly the registered D-B-177 BadZipFile studies
(tools/hb145_data_exclusions_v1.py); any other failure still rejects. Excluded
studies get no teacher target: their failure rows are removed from a sealed
teacher CSV written next to the receipt (all other rows byte-identical), and the
receipt records the exclusion. EXPOSED f0-only contract otherwise unchanged;
tools/hb146_teacher_v113.py preserved.
Changelog 2026-10-14 v1.1.3: registered v113 carry-over and output-code pins;
EXPOSED f0-only scientific contract unchanged; v111 preserved.
Changelog 2026-10-13 v1.1.1: final producer schema, release-device auxiliaries,
strict EXPOSED provenance at every layer; bind monitor and receipt identities.
No inference is performed here.
"""
from __future__ import annotations
import argparse
import csv
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb146_pl_labels_v113 as pl
from tools import hb145_baseline_contract_v113 as contract
from tools import hb145_data_exclusions_v1 as exclusions
from tools import hb145_r7_heldout_v1 as hb
from tools import hb145_r7_pipeline_v111 as pipeline
from tools import hb145_r7_monitor_v111 as monitor

SEAL_VERSION = "1.1.4"
PRODUCER_VERSION = "1.1.3"


def require_roles(value, pl=pl):
    roles = value.get("role_provenance", {})
    pl.need(set(roles) == {"heldout_raw", "heldout_post", "f0_raw", "f0_post"},
            "teacher role provenance incomplete")
    for name, entry in roles.items():
        pl.require_teacher_exposure(entry)
        expected = "not_applied" if name.endswith("_raw") else False
        actual = entry.get("stackers_cross_fitted")
        pl.need(type(actual) is type(expected) and actual == expected, "role stacker provenance differs")


def sealed_teacher_csv(source_csv, dest, excluded, pl=pl):
    """Copy f0_raw.csv minus the excluded failure rows; kept lines byte-identical."""
    text = hb.output_io.retry(lambda: Path(source_csv).read_bytes()).decode("utf-8")  # no newline translation
    lines = text.splitlines(keepends=True)
    records = list(csv.reader(io.StringIO(text, newline="")))
    pl.need(len(lines) == len(records) and records, "teacher CSV has embedded line breaks or no header")
    header = records[0]
    pl.need(pl.UID in header and "status" in header, "teacher CSV schema differs")
    uid_at, status_at = header.index(pl.UID), header.index("status")
    targets = [header.index(t) for t in pl.TARGETS]
    kept, removed = [lines[0]], set()
    for line, row in zip(lines[1:], records[1:]):
        uid = row[uid_at]
        if uid in excluded:
            pl.need(uid not in removed and row[status_at] == "failure"
                    and all(row[i] == "" for i in targets), "excluded teacher row is not an empty failure row")
            removed.add(uid)
        else:
            kept.append(line)
    pl.need(removed == set(excluded), "registered exclusion rows missing from teacher CSV")
    content = "".join(kept)
    dest = Path(dest)
    if dest.exists():
        # Immutable once sealed; Windows readers are retried, never overwritten.
        current = hb.output_io.retry(lambda: dest.read_bytes()).decode("utf-8")
        pl.need(current == content, "sealed teacher CSV differs; version required")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        hb.output_io.retry(lambda: tmp.write_text(content, encoding="utf-8", newline=""))
        hb.output_io.replace(tmp, dest)  # bounded 40-attempt sharing-violation retry
    return dest


def seal(source, out=None, *, exposed_exploratory=False, pl_module=None):
    pl = pl_module if pl_module is not None else globals()["pl"]
    pl.need(exposed_exploratory, "explicit --exposed-exploratory is required")
    pl.need(out is not None, "explicit --out receipt path is required (v1.1.4 has no default stage)")
    source, out = Path(source), Path(out)
    pl.need(out.resolve().is_relative_to(pl.WORK.resolve()) and out.suffix == ".json",
            "receipt must remain in H-B-146 scratch")
    identity = pl.read(source / "identity.json")
    run = pl.read(source / "run_receipt.json")
    worker = pl.read(source / "receipt_dual.json")
    for record in (identity, run, worker):
        pl.require_teacher_exposure(record)
        require_roles(record, pl)
        pl.need(record.get("version") == PRODUCER_VERSION, "H-B-145 v1.1.3 required")
    lineage = contract.validate(source, identity)
    fingerprint = hb.fingerprint(identity)
    device = identity.get("auxiliary_device")
    pl.need(device in ("cpu", "cuda") and identity.get("device") == device
            and all(r.get("identity") == fingerprint and r.get("auxiliary_device") == device
                    for r in (run, worker)), "teacher identity/device differs")
    pl.need(identity.get("version") == PRODUCER_VERSION and identity.get("models") == "heldout,f0",
            "H-B-145 v1.1.3 dual-role identity required")
    roster = identity["uids"]
    pl.need(run.get("completed") == run.get("selected") == len(roster) and len(roster) > 0
            and run.get("exit_code") == 0 and run.get("models") == "heldout,f0", "teacher run incomplete")
    excluded, accepted = set(), None
    if run.get("failures") != 0:
        # Only the registered BadZipFile studies; anything else raises here.
        accepted = exclusions.verify_run(source, identity, run)
        excluded = set(accepted["uids"])
    source_csv = source / "f0_raw.csv"
    csv_path = sealed_teacher_csv(source_csv, out.with_suffix(".csv"), excluded, pl)
    values = pl.table(csv_path)
    expected = set(roster) - excluded
    pl.need(set(values) == expected and len(values) == len(roster) - len(excluded), "teacher roster differs")
    pl.need(worker.get("models") == hb.r7.MODELS, "dual model receipt differs")
    runtime = source / "stage/r7_runtime/compact_infer.py"
    pl.need(identity.get("runner_sha256") == pl.sha(hb.__file__)
            and identity.get("pipeline_sha256") == pl.sha(pipeline.__file__)
            and identity.get("monitor_sha256") == pl.sha(monitor.__file__)
            and identity.get("runtime_sha256") == pl.sha(runtime)
            and all(identity.get(k) for k in ("bundle", "code", "deps")), "teacher code identity differs")
    pl.need(identity.get("output_io_sha256") == pl.sha(hb.output_io.__file__)
            and identity.get("carryover_sha256") == pl.sha(contract.carry.__file__), "teacher output code identity differs")
    columns = [pl.UID, "fold_k", "status", "model_role", "model_fold", "output_kind",
               "pooled_prior_bias", "stackers_cross_fitted", "prediction_reference", "roles_coincide",
               "auxiliary_device", *pl.TEACHER_EXPOSURE]
    meta = pl.pd.read_csv(csv_path, usecols=columns, dtype={pl.UID: str}).set_index(pl.UID)
    exposure = pl.pd.read_csv(csv_path, usecols=[pl.UID, *pl.TEACHER_EXPOSURE],
                             dtype=str, keep_default_na=False).set_index(pl.UID)
    pl.need(meta.index.is_unique and set(meta.index) == set(values), "metadata roster differs")
    paths = [source_csv, csv_path, source / "identity.json", source / "run_receipt.json",
             source / "receipt_dual.json", Path(hb.__file__), Path(pipeline.__file__),
             Path(monitor.__file__), runtime, Path(__file__), Path(pl.__file__)]
    paths += [source / "carry_over_intent.json", source / "carry_over_manifest.json",
              Path(contract.__file__), Path(contract.carry.__file__), Path(hb.output_io.__file__)]
    if accepted:
        paths += [exclusions.PATH, Path(exclusions.__file__)] + [hb.record_path(source, u) for u in sorted(excluded)]
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
    data_exclusions = None if accepted is None else dict(
        sha256=accepted["exclusions_sha256"], count=accepted["count"], uids=accepted["uids"],
        error=accepted["error"], reason=accepted["reason"], reference=accepted["reference"],
        rule="no teacher target: training-split rows use L2 only (control-identical); "
             "validation-split rows keep L2 labels only")
    receipt = dict(version=pl.VERSION, mode="screen_f0", model_selection="f0_only", output_kind="pre_stacker",
                   **pl.TEACHER_EXPOSURE, producer_version=PRODUCER_VERSION, producer_identity=fingerprint,
                   seal_version=SEAL_VERSION, carry_over=lineage,
                   auxiliary_device=device, role_provenance=identity["role_provenance"],
                   csv_sha256=pl.sha(csv_path), source_csv_sha256=pl.sha(source_csv),
                   teacher_csv=str(csv_path.resolve()), teacher_rows=len(values),
                   producer_studies=len(roster), producer_failures=run.get("failures"),
                   complete=True, failures=0, data_exclusions=data_exclusions,
                   source_pool_membership="manifest_contract_unverified_membership",
                   validation_label_exposure="pooled_prior_not_clean", auxiliary_label_exposure="unknown",
                   oai_ancestry="teacher ancestry evidence partial: ImageNet init, fastMRI+, SKM-TEA segmenter",
                   status=pl.STATUS, limitations=pl.LIMITATIONS, exposed_exploratory=True, pins=pl.pins(paths))
    pl.write(out, receipt)
    return {"sealed": True, "rows": len(values), "excluded": len(excluded), "status": pl.STATUS,
            "receipt": str(out), "teacher_csv": str(csv_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seal"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exposed-exploratory", action="store_true")
    args = parser.parse_args(argv)
    print(seal(args.source, args.out, exposed_exploratory=args.exposed_exploratory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
