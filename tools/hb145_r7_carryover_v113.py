"""v1.1.3 one-time, pinned v112 prediction carry-over; never recompute commits."""
from __future__ import annotations

import json
from pathlib import Path

from tools import hb145_r7_heldout_v1 as h

V112_IDENTITY = "72b11bb15e2ea9ca94081d2f0c69899abc59b688b68203ebadf0da3754e67637"
IO_FIELDS = {"version", "runner_sha256", "monitor_sha256", "pipeline_sha256",
             "runtime_sha256", "output_io_sha256", "carryover_sha256"}


def validate_ready(out, identity):
    """An interrupted import may only continue through prepare --carry-over."""
    out = Path(out)
    intent = out / "carry_over_intent.json"
    if not intent.exists():
        return
    expected = h.output_io.read_json(intent)
    manifest_path = out / "carry_over_manifest.json"
    if not manifest_path.exists():
        raise ValueError("incomplete carry-over; repeat prepare --carry-over before run")
    manifest = h.output_io.read_json(manifest_path)
    if (expected.get("identity") != identity or manifest.get("identity") != identity
            or manifest.get("source_identity") != expected.get("source_identity")
            or manifest.get("committed_carried") != 300 or len(manifest.get("entries", [])) != 300):
        raise ValueError("carry-over completion manifest mismatch")
    for entry in manifest["entries"]:
        path = out / "journal" / entry["journal"]
        if path.parent != out / "journal" or h.r7.s6.sha256(path) != entry["destination_sha256"]:
            raise ValueError("carry-over committed journal changed")


def validate_identity(old, new, old_runtime, new_runtime):
    if h.fingerprint(old) != V112_IDENTITY or old["version"] != "1.1.2" or new["version"] != "1.1.3":
        raise ValueError("carry-over requires the registered v112 source and v113 destination")
    if {k: v for k, v in old.items() if k not in IO_FIELDS} != {
            k: v for k, v in new.items() if k not in IO_FIELDS}:
        raise ValueError("carry-over semantic identity differs")
    for record, runtime in ((old, old_runtime), (new, new_runtime)):
        if h.r7.s6.sha256(runtime) != record["runtime_sha256"]:
            raise ValueError("carry-over runtime pin mismatch")
    # Full generated program equality after ONLY the declared output substitutions.
    if h.output_runtime_patch(old_runtime.read_text(encoding="utf-8")) != new_runtime.read_text(encoding="utf-8"):
        raise ValueError("carry-over runtime differs beyond output I/O")


def carry_over(source, out, identity_record, new_runtime):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or not source.is_relative_to(h.SCRATCH.resolve()) or source == h.SCRATCH.resolve():
        raise ValueError("carry-over source must be a separate owned task run")
    with h.exclusive_run(source):
        old = json.loads((source / "identity.json").read_text(encoding="utf-8"))
        validate_identity(old, identity_record, source / "stage/r7_runtime/compact_infer.py", Path(new_runtime))
        state = json.loads((source / "status.json").read_text(encoding="utf-8"))
        if state.get("identity") != V112_IDENTITY or not state.get("cleanup_complete") or state.get("state") != "MONITOR_ERROR":
            raise ValueError("source must be the stopped, cleaned-up v112 run")
        records = h.load_done(source, old["uids"], V112_IDENTITY)
        if len(records) != 300 or len(list((source / "journal").glob("*.json"))) != 300:
            raise ValueError("expected exactly 300 registered source commits")
        identity = h.fingerprint(identity_record)
        entries = []
        # Validate ALL records/collisions before copying any; idempotent after interruption.
        pending = []
        for uid, record in records.items():
            old_path, new_path = h.record_path(source, uid), h.record_path(out, uid)
            digest = h.r7.s6.sha256(old_path)
            carried = {**record, "identity": identity, "carry_over": {
                "source_identity": V112_IDENTITY, "source_journal_sha256": digest,
                "source_version": "1.1.2", "destination_version": "1.1.3"}}
            if new_path.exists() and json.loads(new_path.read_text(encoding="utf-8")) != carried:
                raise ValueError("carry-over destination collision")
            pending.append((old_path, new_path, carried, digest))
        h.atomic_json(out / "carry_over_intent.json", {
            "source_identity": V112_IDENTITY, "identity": identity, "expected_commits": 300})
        for old_path, new_path, carried, digest in pending:
            if not new_path.exists():
                h.output_io.journal(new_path, carried, out)
            actual = json.loads(new_path.read_text(encoding="utf-8"))
            restored = {k: v for k, v in actual.items() if k != "carry_over"}
            restored["identity"] = V112_IDENTITY
            if restored != json.loads(old_path.read_text(encoding="utf-8")) or h.r7.s6.sha256(old_path) != digest:
                raise ValueError("carry-over payload/source changed")
            entries.append({"journal": old_path.name, "source_sha256": digest,
                            "destination_sha256": h.r7.s6.sha256(new_path)})
        manifest = {"version": "1.1.3", "source": str(source), "source_identity": V112_IDENTITY,
                    "identity": identity, "committed_carried": len(records),
                    "remaining": len(old["uids"]) - len(records),
                    "full_runtime_equivalent_except_output_io": True,
                    "prediction_and_all_original_metadata_equal": True,
                    "source_journals_unchanged": True, "entries": entries}
        h.atomic_json(out / "carry_over_manifest.json", manifest)
        validate_ready(out, identity)
        print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}))
        return manifest
