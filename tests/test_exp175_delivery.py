"""EXP175 delivery regressions; no audit, data, GPU or approval generation."""
import hashlib
from pathlib import Path
import re
import shlex

import pytest

from tools import exp175_image_domain_audit as audit
from tools import exp175_behavior as behavior


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = "artifacts/experiments/EXP175_IMAGE_DOMAIN_AUDIT"
VERSION = "1.1.1"
EXPECTED_FILES = {
    "tools/exp175_image_domain_audit.py",
    "tools/exp175_behavior.py",
    "tests/test_exp175_image_domain_audit.py",
    "tests/test_exp175_behavior.py",
    "tests/test_exp175_delivery.py",
    f"{DELIVERY}/PROTOCOL.md",
    f"{DELIVERY}/RUNBOOK.md",
}
RUNBOOK = (ROOT / DELIVERY / "RUNBOOK.md").read_text(encoding="utf-8")
COMMANDS = re.findall(r"^& \$py tools/exp175_image_domain_audit\.py (.+)$", RUNBOOK, re.M)


def test_every_documented_audit_command_is_covered():
    assert len(COMMANDS) == 3, "pilot, full and behavior-only commands must all be checked"
    assert len(set(COMMANDS)) == 3


@pytest.mark.parametrize("command", COMMANDS, ids=["pilot", "full", "behavior-only"])
def test_documented_audit_commands_parse_with_required_review_note(command):
    # Parse exactly the delivered argument list. Argparse accepts the variable
    # spelling as a string; no real note is needed or invented for this check.
    args = audit.parser().parse_args(shlex.split(command))
    assert args.verification_note == "$verificationNote"
    assert args.out.startswith(f"{DELIVERY}/")
    assert args.out.endswith(f"-v{VERSION}")
    assert args.workers == 4
    if args.behavior_only:
        assert args.behavior_only == f"{DELIVERY}/run-v{VERSION}/metrics.csv"
    # Removing the delivered flag reproduces the original execution defect.
    tokens = shlex.split(command)
    i = tokens.index("--verification-note")
    del tokens[i:i + 2]
    with pytest.raises(SystemExit) as error:
        audit.parser().parse_args(tokens)
    assert error.value.code == 2


def test_runbook_documents_external_exact_source_review_and_all_tests():
    assert "external independent reviewer's PASS" in RUNBOOK
    assert '"verified_tool_version": "1.1.1"' in RUNBOOK
    assert '"exp175_image_domain_audit.py"' in RUNBOOK
    assert '"exp175_behavior.py"' in RUNBOOK
    assert "not approval" in RUNBOOK
    assert "synthetic" in RUNBOOK and "new review and receipt" in RUNBOOK
    assignment = re.search(r"^\$verificationNote = '([^']+)'$", RUNBOOK, re.M)
    assert assignment and assignment.group(1).endswith(f"-v{VERSION}.json")
    assert "Test-Path -LiteralPath $verificationNote -PathType Leaf" in RUNBOOK
    pytest_commands = re.findall(r"^& \$py -m pytest (.+)$", RUNBOOK, re.M)
    assert len(pytest_commands) == 1
    tokens = shlex.split(pytest_commands[0])
    assert {name for name in tokens if name.endswith(".py")} == {
        name for name in EXPECTED_FILES if name.startswith("tests/")
    }


def test_historical_document_versions_and_current_tool_pair():
    # Old delivery stays immutable; current sources require a new exact-source review.
    assert audit.VERSION == behavior.VERSION == "1.2.0"
    assert audit.VERSION != VERSION
    for name in ("RUNBOOK.md", "PROTOCOL.md"):
        text = (ROOT / DELIVERY / name).read_text(encoding="utf-8")
        assert re.search(rf"\bv{re.escape(VERSION)}\b", text.splitlines()[0]), name
        if name == "RUNBOOK.md":
            assert f"Changelog v{VERSION}:" in text


def test_historical_delivery_manifest_has_exact_inventory_and_preserved_documents():
    manifest = ROOT / DELIVERY / "SHA256SUMS.txt"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == "41127f4d716256f48f6b1a1712707a30cb2a0264b2d62276795bbfc59a0ddfbb"
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        assert match, f"invalid delivery hash row: {line!r}"
        rows.append(match.groups())
    names = [name for _, name in rows]
    assert len(names) == len(set(names)), "duplicate delivery path"
    assert set(names) == EXPECTED_FILES
    for expected_hash, name in rows:
        if not name.startswith(DELIVERY + "/"):
            continue  # Source hashes identify the historical delivery, not today's version.
        path = ROOT / name
        assert path.is_file(), name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"stale delivery SHA256: {name}"
