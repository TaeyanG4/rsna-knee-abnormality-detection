"""Synthetic numeric dual-role sealing contracts, no real label rows."""
import csv
from pathlib import Path
import pytest
from tools import hb146_teacher_v110 as teacher

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
    identity = dict(version="1.1.0", models="heldout,f0", uids=[uid],
                    runner_sha256=pl.sha(hb.__file__), pipeline_sha256=pl.sha(pipeline.__file__),
                    runtime_sha256=pl.sha(runtime), bundle="b", code="c", deps="d")
    pl.write(source / "identity.json", identity)
    pl.write(source / "run_receipt.json", dict(completed=1, selected=1, failures=0, exit_code=0, models="heldout,f0"))
    pl.write(source / "receipt_dual.json", dict(models=hb.r7.MODELS))
    k = pl.hash_fold(uid)
    raw = [.25] * 12
    rec = {pl.UID: uid, "fold_k": k, "identity": hb.fingerprint(identity), "status": "ok",
           "pooled_prior_bias": True, "models": "heldout,f0", "auxiliary_device": "cpu",
           "role_folds": {"heldout": k, "f0": 0}, "roles_coincide": k == 0,
           "predictions": {str(f): {"raw": raw, "post": raw} for f in pipeline.required_folds(uid)}}
    path = hb.record_path(source, uid)
    pl.write(path, rec)
    pipeline.export_roles(source, [uid], {uid: rec})
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

