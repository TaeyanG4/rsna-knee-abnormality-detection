"""Downstream contract v1.1.3 (2026-10-14): authenticate the registered carry-over.

Only destination files are read. The stopped v112 source is never reopened.
Identity pins come from H-B-145 REPORT.md; prediction math is unchanged.
"""
from pathlib import Path
import re

from tools import hb145_r7_heldout_v1 as hb
from tools import hb145_r7_carryover_v113 as carry

VERSION = "1.1.3"
SOURCE_IDENTITY = carry.V112_IDENTITY
IDENTITY = "f46bbdc29c71c7b5196987b2ea43a4d5c256f4b0c02b0e4fcde1ee2fb48480aa"
TOTAL = 4407
CARRIED = 300


def validate(source, identity):
    """Fail closed on absent, incomplete, altered or unrelated migration evidence."""
    source = Path(source)
    fingerprint = hb.fingerprint(identity)
    if (fingerprint != IDENTITY or identity.get("version") != VERSION
            or identity.get("models") != "heldout,f0"
            or identity.get("exposure") != "EXPOSED" or identity.get("clean_oof") is not False
            or identity.get("device") != "cuda"
            or len(identity.get("uids", [])) != TOTAL or len(set(identity["uids"])) != TOTAL):
        raise ValueError("unregistered v113 baseline identity")
    intent_path, manifest_path = (source / n for n in ("carry_over_intent.json", "carry_over_manifest.json"))
    intent, manifest = (hb.output_io.read_json(p) for p in (intent_path, manifest_path))
    entries = manifest.get("entries", [])
    if (intent != dict(source_identity=SOURCE_IDENTITY, identity=fingerprint, expected_commits=CARRIED)
            or manifest.get("version") != VERSION or manifest.get("identity") != fingerprint
            or manifest.get("source_identity") != SOURCE_IDENTITY
            or manifest.get("committed_carried") != CARRIED or len(entries) != CARRIED
            or manifest.get("remaining") != TOTAL - CARRIED
            or any(manifest.get(k) is not True for k in (
                "full_runtime_equivalent_except_output_io", "prediction_and_all_original_metadata_equal",
                "source_journals_unchanged"))):
        raise ValueError("carry-over manifest/intent contract differs")
    roster = {hb.record_path(source, uid).name: uid for uid in identity["uids"]}
    seen = set()
    for entry in entries:
        name = entry.get("journal", "")
        if name not in roster or name in seen or not re.fullmatch(r"[0-9a-f]{64}\.json", name):
            raise ValueError("carry-over journal roster differs")
        seen.add(name)
        path = source / "journal" / name
        digest = entry.get("source_sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or hb.r7.s6.sha256(path) != entry.get("destination_sha256"):
            raise ValueError("carry-over journal hash differs")
        record = hb.output_io.read_json(path)
        if (record.get("identity") != fingerprint or record.get(hb.UID) != roster[name]
                or record.get("carry_over") != dict(source_identity=SOURCE_IDENTITY,
                    source_journal_sha256=digest, source_version="1.1.2", destination_version=VERSION)):
            raise ValueError("carry-over journal lineage differs")
    return dict(source_identity=SOURCE_IDENTITY, identity=fingerprint, committed_carried=CARRIED,
                manifest_sha256=hb.r7.s6.sha256(manifest_path), intent_sha256=hb.r7.s6.sha256(intent_path))
