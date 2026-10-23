"""Test-only migration; never reads the registered full runs or real labels."""
from tools import hb145_r7_heldout_v1 as hb
from tools import hb145_baseline_contract_v113 as contract


def migrate(source, monkeypatch, carried=None):
    identity = hb.output_io.read_json(source / "identity.json")
    old_identity = hb.fingerprint(identity)
    identity.update(version="1.1.3", output_io_sha256=hb.r7.s6.sha256(hb.output_io.__file__),
                    carryover_sha256=hb.r7.s6.sha256(contract.carry.__file__))
    new_identity = hb.fingerprint(identity)
    hb.atomic_json(source / "identity.json", identity)
    for name in ("run_receipt.json", "receipt_dual.json"):
        path = source / name
        if path.exists():
            receipt = hb.output_io.read_json(path)
            receipt.update(version="1.1.3", identity=new_identity)
            hb.atomic_json(path, receipt)
    entries = []
    for uid in identity["uids"]:
        path = hb.record_path(source, uid)
        if not path.exists():
            continue
        record = hb.output_io.read_json(path)
        digest = hb.r7.s6.sha256(path)
        record["identity"] = new_identity
        is_carried = carried is None or len(entries) < carried
        if is_carried:
            record["carry_over"] = dict(source_identity=old_identity, source_journal_sha256=digest,
                                        source_version="1.1.2", destination_version="1.1.3")
        hb.atomic_json(path, record)
        if is_carried:
            entries.append(dict(journal=path.name, source_sha256=digest, destination_sha256=hb.r7.s6.sha256(path)))
    monkeypatch.setattr(contract, "SOURCE_IDENTITY", old_identity)
    monkeypatch.setattr(contract, "IDENTITY", new_identity)
    monkeypatch.setattr(contract, "TOTAL", len(identity["uids"]))
    monkeypatch.setattr(contract, "CARRIED", len(entries))
    hb.atomic_json(source / "carry_over_intent.json", dict(source_identity=old_identity,
                   identity=new_identity, expected_commits=len(entries)))
    hb.atomic_json(source / "carry_over_manifest.json", dict(version="1.1.3", source="fixture-only",
        source_identity=old_identity, identity=new_identity, committed_carried=len(entries),
        remaining=len(identity["uids"]) - len(entries), entries=entries,
        full_runtime_equivalent_except_output_io=True, prediction_and_all_original_metadata_equal=True,
        source_journals_unchanged=True))
    return identity


def migrate_pair(artifacts, monkeypatch):
    from tools import hb149_r7_flip_v120 as flip
    selection, baseline, flip_dir, _, _ = artifacts
    frozen = selection.read_bytes()
    identity = migrate(baseline, monkeypatch, carried=16)
    fid = hb.output_io.read_json(flip_dir / "identity.json")
    fid["baseline_identity"] = hb.fingerprint(identity)
    flip.atomic_json(flip_dir / "identity.json", fid)
    receipt = hb.output_io.read_json(flip_dir / "run_receipt.json")
    receipt["identity"] = flip.fingerprint(fid)
    flip.atomic_json(flip_dir / "run_receipt.json", receipt)
    for uid in fid["uids"]:
        path = flip.record_path(flip_dir, uid)
        record = hb.output_io.read_json(path)
        record["identity"] = flip.fingerprint(fid)
        flip.atomic_json(path, record)
    assert selection.read_bytes() == frozen
    return artifacts
