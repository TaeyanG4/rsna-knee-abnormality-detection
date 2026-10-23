"""v1.2.6 CPU fixtures only; never inspect the running production directory."""
import json
import pytest
from tools import hb149_postrun_v126 as p


@pytest.fixture
def completed(tmp_path, monkeypatch):
    out = tmp_path / "full-v125"
    monkeypatch.setattr(p, "FLIP_DIR", out)
    # Isolated fixture creation has no concurrent producer; avoid 18,000 fsyncs.
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    uids = [f"fixture-{i}" for i in range(1600) if p.flip.fold_k(f"fixture-{i}") != 0][:1000]
    assert len(uids) == 1000
    ident = dict(uids=uids, engineering_only=True)
    pin = p.flip.fingerprint(ident)
    monkeypatch.setattr(p, "IDENTITY", pin)
    write(out / "identity.json", ident)
    for uid in uids:
        k = p.flip.fold_k(uid)
        write(p.flip.record_path(out, uid), dict(
            **{p.flip.UID: uid}, **p.flip.EXPOSURE, identity=pin, fold_k=k,
            models="heldout_flip", status="ok", decode_failures=0, decoded_files=1, dicom_files=1,
            role_folds={"heldout_flip": k}, roles_coincide=False, flip_tta=True,
            predictions={str(k): dict(raw=[.5]*12, post=[.5]*12)}))
    runtime = out / "stage/r7_runtime"
    runtime.mkdir(parents=True)
    (runtime / "compact_infer.py").write_text("# synthetic engineering runtime\n")
    digest = p.sha(runtime / "compact_infer.py")
    write(runtime / "runtime_sha256.json", {"compact_infer.py": digest})
    hashes = {p.flip.record_path(out, u).name: p.sha(p.flip.record_path(out, u)) for u in uids[:551]}
    reg = dict(version="1.2.5", scientific_identity=pin, journals=hashes,
               executed_runtime_sha256=digest, execution={"workers": 1, "chunk": 32, "amp": "fp16"})
    reg_path = tmp_path / "resume-registration-v125.json"
    write(reg_path, reg)
    monkeypatch.setattr(p, "RESUME_REGISTRATION", reg_path)
    monkeypatch.setattr(p, "RESUME_SHA", p.sha(reg_path))
    write(out / "execution-amendment-v125.json", dict(version="1.2.5", scientific_identity=pin,
          registration_sha256=p.RESUME_SHA, carried=551, journal_hashes=hashes,
          executed_runtime_sha256=digest, execution=reg["execution"]))
    write(out / "execution-runtime-v125.json", dict(version="1.2.5", registration_sha256=p.RESUME_SHA,
          runtime_sha256=digest, runtime_manifest_sha256=p.sha(runtime / "runtime_sha256.json"), engineering_only=False))
    resource = dict(version="1.2.5", identity=pin, state="COMPLETED", exit_code=0, completed_count=1000,
                    cleanup_complete=True, stalled=False, ram_guard_stopped=False, samples=[])
    write(out / "run_resources.json", resource)
    write(out / "run_receipt.json", dict(version="1.2.0", identity=pin, models="heldout_flip",
          selected=1000, completed=1000, failures=0, decode_failures=0, exit_code=0,
          codec_coverage_complete=True, resources={k:v for k,v in resource.items() if k != "samples"},
          **p.flip.EXPOSURE))
    return out, uids, write


def test_completed_exact_carryover_and_unchanged_readout(completed, monkeypatch, tmp_path):
    out, _, _ = completed
    proof = p.validate_flip(out)
    assert proof["journals"] == 1000 and proof["carried_byte_identical"] == 551
    calls = []
    monkeypatch.setattr(p.legacy, "run", lambda *a: calls.append(a) or "unchanged")
    dest = tmp_path / "output"
    assert p.run("selection", "baseline", out, dest, "exclusion") == "unchanged"
    assert calls == [("selection", "baseline", out, dest, "exclusion")]
    assert p.read(dest / "flip-validation-v126.json") == proof


@pytest.mark.parametrize("dirname", ["full-v122", "other", "partial"])
def test_wrong_directory_rejected_without_reading(dirname, monkeypatch, tmp_path):
    monkeypatch.setattr(p, "read", lambda _: pytest.fail("must reject before reading"))
    monkeypatch.setattr(p, "sha", lambda _: pytest.fail("must reject before hashing"))
    with pytest.raises(ValueError, match="only full-v125"):
        p.validate_flip(tmp_path / dirname)


@pytest.mark.parametrize("fault", ["partial", "running", "failure", "stalled", "cleanup", "identity",
    "receipt", "manifest", "carried", "missing", "extra", "temporary", "new-journal", "runtime", "registration"])
def test_incomplete_or_tampered_resume_blocks(completed, fault, monkeypatch):
    out, uids, write = completed
    if fault in {"partial", "running", "failure", "stalled", "cleanup"}:
        path = out / "run_resources.json"
        value = p.read(path)
        key, val = {"partial": ("completed_count", 551), "running": ("state", "RUNNING"),
            "failure": ("exit_code", 1), "stalled": ("stalled", True), "cleanup": ("cleanup_complete", False)}[fault]
        value[key] = val
        write(path, value)
    elif fault == "identity": write(out / "identity.json", {"uids": uids[:-1]})
    elif fault == "receipt":
        value = p.read(out / "run_receipt.json")
        value["completed"] = 551
        write(out / "run_receipt.json", value)
    elif fault == "manifest":
        value = p.read(out / "execution-amendment-v125.json")
        value["journal_hashes"] = {}
        write(out / "execution-amendment-v125.json", value)
    elif fault == "carried":
        path = p.flip.record_path(out, uids[0])
        path.write_bytes(path.read_bytes() + b" ")
    elif fault == "missing": p.flip.record_path(out, uids[-1]).unlink()
    elif fault == "extra": write(out / "journal/extra.json", {})
    elif fault == "temporary": (out / "journal/uncommitted.tmp").touch()
    elif fault == "new-journal":
        path = p.flip.record_path(out, uids[-1])
        value = p.read(path)
        value["flip_tta"] = False
        write(path, value)
    elif fault == "runtime": (out / "stage/r7_runtime/compact_infer.py").write_text("changed")
    elif fault == "registration": p.RESUME_REGISTRATION.write_bytes(b"{}")
    monkeypatch.setattr(p.legacy, "run", lambda *a: pytest.fail("readout must not start"))
    with pytest.raises(ValueError):
        p.run(None, None, out, out / "output", None)


def test_missing_receipt_fails_closed(completed):
    out, _, _ = completed
    (out / "run_receipt.json").unlink()
    with pytest.raises(FileNotFoundError): p.validate_flip(out)


def test_production_defaults_and_registration():
    assert p.FLIP_DIR == p.WORK / "full-v125"
    assert p.sha(p.RESUME_REGISTRATION) == p.RESUME_SHA
    assert p.legacy.VERSION == "1.2.4"
    p.legacy.registration()


def test_main_failure_no_confidential_values(monkeypatch, capsys):
    monkeypatch.setattr(p, "run", lambda *a: (_ for _ in ()).throw(ValueError("SECRET_SENTINEL")))
    assert p.main([]) == 1
    output = capsys.readouterr()
    assert output.out.strip() == "SUBMIT_CANDIDATE=NONE"
    assert "SECRET_SENTINEL" not in output.err


def test_export_only_never_opens_flip(monkeypatch):
    monkeypatch.setattr(p, "validate_flip", lambda *a: pytest.fail("no flip access for export"))
    monkeypatch.setattr(p.legacy, "main", lambda a: 0 if "--export-only" in a else 1)
    assert p.main(["--export-only"]) == 0
