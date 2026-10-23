"""CPU lifecycle CLI v1.0.0 (astra, 2026-10-11).

Changelog: explicit registration/staging/admission/artifact commands. No launcher,
dynamic imports, GPU probes or time limits. Invoke with python -m tools.arm_lifecycle.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tarfile

from training.arms import lifecycle as life


def read(path):
    return life.parse_json(Path(path).read_bytes())


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--draft", required=True)
    register.add_argument("--root", required=True)
    register.add_argument("--out", required=True)
    for name in ("validate", "prepare", "admit", "receipt", "export", "collect"):
        sub = commands.add_parser(name)
        sub.add_argument("--spec", required=True)
        if name in ("validate", "prepare"):
            sub.add_argument("--root", required=True)
        if name == "validate":
            sub.add_argument("--stage", action="store_true")
        if name in ("prepare", "receipt", "export", "collect"):
            sub.add_argument("--out", required=True)
        if name == "prepare":
            sub.add_argument("--replacements")
        if name == "admit":
            sub.add_argument("--stage", required=True)
            sub.add_argument("--note", required=True)
            sub.add_argument("--evidence-root", required=True)
            sub.add_argument("--dry-run", action="store_true", required=True)
        if name in ("receipt", "export"):
            sub.add_argument("--outputs", required=True)
        if name == "receipt":
            sub.add_argument("--verification-sha256", required=True)
            sub.add_argument("--exit-code", required=True, type=int)
            sub.add_argument("--elapsed-seconds", required=True, type=float)
            sub.add_argument("--metrics", required=True)
        if name == "export":
            sub.add_argument("--receipt", required=True)
        if name in ("export", "collect"):
            sub.add_argument("--receipt-sha256", required=True)
        if name == "collect":
            sub.add_argument("--archive", required=True)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.command == "register":
            spec = life.ArmSpec.from_document(read(a.draft))
            life.validate_inputs(spec, a.root)
            life.register(spec.document, a.out)
            result = {"status": "REGISTERED_NO_EXECUTION", "spec_sha256": spec.sha256}
        else:
            spec = life.ArmSpec(Path(a.spec).read_bytes())
            if a.command == "validate":
                (life.validate_stage if a.stage else life.validate_inputs)(spec, a.root)
                result = {"status": "VALIDATED_NO_EXECUTION", "spec_sha256": spec.sha256}
            elif a.command == "prepare":
                result = life.prepare(spec, a.root, a.out,
                                      replacements=read(a.replacements) if a.replacements else None)
            elif a.command == "admit":
                result = life.admission_dry_run(spec, a.stage, read(a.note), a.evidence_root)
            elif a.command == "receipt":
                receipt = life.artifact_receipt(spec, a.outputs, verification_sha256=a.verification_sha256,
                                               exit_code=a.exit_code, elapsed_seconds=a.elapsed_seconds,
                                               metrics=read(a.metrics))
                life.create_new(a.out, life.canonical_bytes(receipt))
                result = {"status": receipt["status"], "receipt_sha256": life.content_sha256(receipt),
                          "scientific_status": receipt["scientific_status"]}
            elif a.command == "export":
                digest = life.export_artifacts(spec, a.outputs, read(a.receipt), a.out,
                                               expected_receipt_sha256=a.receipt_sha256)
                result = {"status": "EXPORTED_ARTIFACTS", "archive_sha256": digest}
            else:
                result = life.collect_artifacts(spec, a.archive, a.out,
                                                expected_receipt_sha256=a.receipt_sha256)
        # Only identities/status, never input rows or the complete scientific config.
        if a.command == "prepare":
            result = {key: result[key] for key in ("status", "spec_sha256")}
        print(life.canonical_bytes(result).decode("utf-8"))
        return 0
    except (life.ContractError, OSError, ValueError, tarfile.TarError, EOFError) as error:
        # Do not echo arbitrary paths or JSON/CSV payloads in CLI failure logs.
        print(life.canonical_bytes({"status": "REJECTED", "error_type": type(error).__name__}).decode())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
