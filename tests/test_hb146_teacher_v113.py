"""Synthetic numeric dual-role sealing contracts, no real label rows."""
import csv
from pathlib import Path
import pytest
from tools import hb146_teacher_v113 as teacher

pl, hb, pipeline = teacher.pl, teacher.hb, teacher.pipeline


@pytest.fixture
def dual(tmp_path, monkeypatch):
    # v1 wrapper tests mutate hb.fold_k globally; restore the dual-role contract locally.
    monkeypatch.setattr(hb, "fold_k", pl.hash_fold)
    monkeypatch.setattr(pl, "WORK", tmp_path)
    source = tmp_path / "teacher"
    runtime = source / "stage/r7_runtime/compact_infer.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("# synthetic runtime\n")
    uid = "synthetic-study"
    provenance = dict(**hb.EXPOSURE, role_provenance=hb.role_provenance("heldout,f0"))
    identity = dict(**provenance, version="1.1.1", models="heldout,f0", uids=[uid],
                    auxiliary_device="cuda", device="cuda", monitor_sha256=pl.sha(teacher.monitor.__file__),
                    runner_sha256=pl.sha(hb.__file__), pipeline_sha256=pl.sha(pipeline.__file__),
                    runtime_sha256=pl.sha(runtime), bundle="b", code="c", deps="d")
    pl.write(source / "identity.json", identity)
    receipt_meta = dict(**provenance, version="1.1.1", identity=hb.fingerprint(identity), auxiliary_device="cuda")
    pl.write(source / "run_receipt.json", dict(**receipt_meta, completed=1, selected=1, failures=0, exit_code=0, models="heldout,f0"))
    pl.write(source / "receipt_dual.json", dict(**receipt_meta, models=hb.r7.MODELS))
    k = pl.hash_fold(uid)
    raw = [.25] * 12
    rec = {**hb.EXPOSURE, pl.UID: uid, "fold_k": k, "identity": hb.fingerprint(identity), "status": "ok",
           "pooled_prior_bias": True, "models": "heldout,f0", "auxiliary_device": "cuda",
           "role_folds": {"heldout": k, "f0": 0}, "roles_coincide": k == 0,
           "predictions": {str(f): {"raw": raw, "post": raw} for f in pipeline.required_folds(uid)}}
    path = hb.record_path(source, uid)
    pl.write(path, rec)
    pipeline.export_roles(source, [uid], {uid: rec})
    from hb145_v113_fixture import migrate
    migrate(source, monkeypatch)
    return source, uid, path


def test_seal_dual_role(dual):
    source, uid, path = dual
    result = teacher.seal(source, exposed_exploratory=True)
    receipt = pl.read(pl.WORK / "teacher-f0.json")
    assert result["rows"] == 1 and receipt["status"] == "EXPOSED_EXPLORATORY"
    assert str(path.resolve()) in receipt["pins"]
    assert receipt["limitations"] == pl.LIMITATIONS
    assert receipt["csv_sha256"] == pl.sha(source / "f0_raw.csv")


def test_explicit_opt_in(dual):
    with pytest.raises(ValueError, match="explicit"):
        teacher.seal(dual[0])


@pytest.mark.parametrize("layer", ["identity.json", "run_receipt.json", "receipt_dual.json", "journal", "csv",
                                    "identity-role", "run-role", "worker-role"])
@pytest.mark.parametrize("mutation", ["missing-exposure", "clean", "missing-clean", "true-clean", "string-clean"])
def test_exposure_fail_closed_every_layer(dual, layer, mutation):
    source, uid, journal = dual
    paths = {"identity-role": "identity.json", "run-role": "run_receipt.json", "worker-role": "receipt_dual.json"}
    path = journal if layer == "journal" else source / paths.get(layer, layer)
    if layer == "csv":
        path = source / "f0_raw.csv"
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            columns, rows = reader.fieldnames, list(reader)
        entry = rows[0]
    else:
        record = pl.read(path)
        entry = record["role_provenance"]["f0_raw"] if layer.endswith("-role") else record
    if mutation == "missing-exposure":
        entry.pop("exposure")
    elif mutation == "clean":
        entry["exposure"] = "CLEAN"
    elif mutation == "missing-clean":
        entry.pop("clean_oof")
    elif mutation == "true-clean":
        entry["clean_oof"] = "True" if layer == "csv" else True
    else:
        entry["clean_oof"] = "false" if layer == "csv" else "False"
    if layer == "csv":
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    else:
        pl.write(path, record)
    with pytest.raises(ValueError):
        teacher.seal(source, exposed_exploratory=True)


def test_sealed_provenance_remains_required(dual):
    source, uid, _ = dual
    teacher.seal(source, exposed_exploratory=True)
    receipt = pl.WORK / "teacher-f0.json"
    values, sealed = pl.validate_teacher(source / "f0_raw.csv", receipt, [uid])
    assert len(values) == 1 and sealed["clean_oof"] is False
    sealed.pop("clean_oof")
    pl.write(receipt, sealed)
    with pytest.raises(ValueError, match="EXPOSED"):
        pl.validate_teacher(source / "f0_raw.csv", receipt, [uid])


def test_device_identity_mismatch_rejected(dual):
    source, _, path = dual
    rec = pl.read(path)
    rec["auxiliary_device"] = "cpu"
    pl.write(path, rec)
    with pytest.raises(ValueError):
        teacher.seal(source, exposed_exploratory=True)


@pytest.mark.parametrize("mutation", ["incomplete", "wrong-model", "tamper-journal", "tamper-runtime", "wrong-role"])
def test_fail_closed(dual, mutation):
    source, uid, path = dual
    if mutation == "incomplete":
        pl.write(source / "run_receipt.json", dict(completed=1, selected=2, failures=0, exit_code=0, models="heldout,f0"))
    elif mutation == "wrong-model":
        pl.write(source / "receipt_dual.json", dict(models=[hb.r7.MODELS[0]]))
    elif mutation == "tamper-journal":
        rec = pl.read(path)
        rec["predictions"]["0"]["raw"][0] = .7
        pl.write(path, rec)
    elif mutation == "tamper-runtime":
        (source / "stage/r7_runtime/compact_infer.py").write_text("# changed\n")
    else:
        p = source / "f0_raw.csv"
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            columns, rows = reader.fieldnames, list(reader)
        rows[0]["model_role"] = "heldout"
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ValueError):
        teacher.seal(source, exposed_exploratory=True)

