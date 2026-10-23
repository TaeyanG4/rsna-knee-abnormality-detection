"""H-B-105 package/evidence/readout operations v1.0.0 (2026-10-11).

Changelog: Small backbone-only experiment, immutable specs, offline package,
reviewed evidence merge, integrity export and paired multi-lens screen specs.
No training launcher, network, GPU probe or wall-time limits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import tarfile

from training.arms import lifecycle as life, mil_small as mil

ROOT = Path(__file__).resolve().parents[1]
EXP = "artifacts/experiments/HB105_CONVNEXT_S_MIL"
WEIGHT = "storage/weights/exp171/convnext_small_fb_in22k_ft_in1k.safetensors"
INITIAL = "artifacts/experiments/EXP171_CONVNEXT_S_L2/initial-state.pt"
READOUTS = ("oof_best.csv", "oof_k36.csv", "oof_k36_flip_tta.csv")
read, sha, save = mil.read, mil.sha, mil.write_new


def config_path(seed):
    mil.need(seed in (42, 43), "unsupported seed")
    return f"training/configs/hb105-small-s{seed}-f0-v1.json"


def control_path(seed):
    return f"artifacts/experiments/EXP185_FINDING_MIL/outputs/S{seed}/results/EXP185_FINDING_MIL_F0_S{seed}"


def spec_for(root, seed):
    return life.ArmSpec((Path(root) / config_path(seed)).read_bytes())


def configure(root):
    """Derive the one registered contrast; never rewrite existing specs."""
    import torch
    root = Path(root)
    semantic = mil.state_hash(torch.load(root / INITIAL, map_location="cpu", weights_only=True))
    results = []
    for seed in (42, 43):
        doc = life.ArmSpec((root / f"training/configs/mil-s{seed}-f0-v3.json").read_bytes()).document
        doc.update(arm=f"HB105_SMALL_S{seed}_F0", attempt="v1")
        old = doc["config"]["model"]
        for name in (old["weights"], old["initial_state"]):
            del doc["inputs"]["data"][name]
        doc["config"]["model"] = dict(arch=mil.ARCH, kind=old["kind"], weights=WEIGHT,
            weights_sha256=sha(root / WEIGHT), initial_state=INITIAL,
            initial_file_sha256=sha(root / INITIAL), initial_semantic_sha256=semantic)
        doc["inputs"]["data"].update({p: sha(root / p) for p in (WEIGHT, INITIAL)})
        doc["inputs"]["sources"] = {p: sha(root / p) for p in mil.SOURCE_FILES}
        control = control_path(seed)
        hashes = {n: sha(root / control / n) for n in ("summary.json", "provenance.json")}
        doc["inputs"]["controls"] = {control + "/" + n: sha(root / control / n)
                                    for n in (*hashes, *READOUTS)}
        doc["target_environment"] = life.environment_from_control(root / control / "summary.json",
            root / control / "provenance.json", expected_hashes=hashes)
        spec = life.ArmSpec.from_document(doc)
        mil.validate_data(spec, root, root / doc["config"]["roots"]["corpus"])
        life.register(doc, root / config_path(seed))
        results.append({"seed": seed, "spec_sha256": spec.sha256})
    return {"status": "CONFIGURED_NO_EXECUTION", "specs": results}


def package(root, out, archive, cpu_root=None):
    root, out = Path(root), Path(out)
    files = set()
    for seed in (42, 43):
        spec = spec_for(root, seed)
        life.validate_inputs(spec, root)
        files.update(spec.inventory)
        files.add(config_path(seed))
    files.update([EXP + "/PROTOCOL.md", EXP + "/LAUNCH.md"])
    out.mkdir(parents=True, exist_ok=False)
    for name in sorted(files):
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
    if cpu_root is not None:
        for seed in (42, 43):
            source = Path(cpu_root) / f"cpu-s{seed}"
            note = read(source / "verification.json")
            life.verify_evidence(spec_for(root, seed), note, source)
            mil.need(note["verdict"] == "HOLD" and note["smokes"]["cpu"] is not None, "CPU evidence required")
            # Resume optimizer states are unnecessary for admission/transfer.
            for p in source.rglob("*"):
                if p.is_file() and "epochs" not in p.relative_to(source).parts:
                    name = f"cpu-s{seed}/" + p.relative_to(source).as_posix()
                    dest = out / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dest)
                    files.add(name)
    save(out / "package.json", {"schema": "hb105-package/v1", "seeds": [42, 43],
        "files": {p: sha(out / p) for p in sorted(files)}})
    with tarfile.open(archive, "x") as tar:
        for p in sorted(files | {"package.json"}):
            tar.add(out / p, arcname=p, recursive=False)
    return {**verify_package(out), "archive_sha256": sha(archive), "bytes": Path(archive).stat().st_size}


def verify_package(root):
    root = Path(root)
    manifest = read(root / "package.json")
    mil.need(manifest["schema"] == "hb105-package/v1" and manifest["seeds"] == [42, 43], "package schema differs")
    life._verify_files(root, manifest["files"])
    for seed in (42, 43):
        life.validate_inputs(spec_for(root, seed), root)
    return {"status": "PACKAGE_VERIFIED", "files": len(manifest["files"])}


def combine(spec, cpu, target, dest, reviewer_name, reviewer_host):
    """Explicit independent-review attestation over two actual HOLD smokes.

    B invokes after inspecting both smokes. The CPU builder never issues PASS.
    Hardlinks avoid duplicate evidence checkpoint storage on the same volume.
    """
    original = {}
    for kind, root in (("cpu", cpu), ("target", target)):
        note = read(root / "verification.json")
        life.verify_evidence(spec, note, root)
        mil.need(note["verdict"] == "HOLD" and note["smokes"][kind] is not None
                 and note["smokes"]["target" if kind == "cpu" else "cpu"] is None, "two original smokes required")
        original[kind] = note
    mil.need(bool(reviewer_name.strip()) and bool(reviewer_host.strip()), "reviewer identity required")
    dest.mkdir(parents=True, exist_ok=False)
    checks, smokes = [], {}
    for kind, root in (("cpu", cpu), ("target", target)):
        note = original[kind]
        for record, role in ((note["checks"][0], "check"), (note["smokes"][kind], "smoke")):
            log = read(root / "evidence" / (record["log_sha256"] + ".json"))
            for artifact in log["artifacts"].values():
                member = kind + "/" + artifact["path"]
                destination = dest / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    try:
                        os.link(root / artifact["path"], destination)
                    except OSError:
                        shutil.copyfile(root / artifact["path"], destination)
                mil.need(sha(destination) == artifact["sha256"], "evidence copy differs")
                artifact["path"] = member
            digest = life.content_sha256(log)
            save(dest / "evidence" / (digest + ".json"), log)
            updated = dict(record, log_sha256=digest)
            if role == "check":
                updated["name"] = kind + "-executed-family-smoke"
                checks.append(updated)
            else:
                smokes[kind] = updated
    merged = {"schema": life.VERIFICATION_SCHEMA, "scope": spec.scope,
        "reviewer": {"name": reviewer_name, "host": reviewer_host}, "verdict": "PASS", "blockers": [],
        "notes": ["Independent reviewer attests inspection of actual CPU and target evidence."],
        "bindings": spec.bindings, "checks": checks, "smokes": smokes}
    life.verify_evidence(spec, merged, dest)
    save(dest / "verification-pass.json", merged)
    return {"status": "PASS_EVIDENCE_MERGED", "verification_sha256": life.content_sha256(merged)}


def check_outputs(spec, output, *, smoke=False):
    summary, provenance, history = [read(output / n) for n in ("summary.json", "provenance.json", "history.json")]
    c = spec.document["config"]
    mil.need(summary["spec_sha256"] == provenance["spec_sha256"] == spec.sha256, "output spec differs")
    mil.need(summary["status"] == ("ENGINEERING_SMOKE" if smoke else "FIXED_FINAL_COMPLETE"), "output mode differs")
    mil.need(summary["epochs"] == (1 if smoke else 20) and len(history) == summary["epochs"], "epochs differ")
    mil.need(summary["attempted_updates"] == (1 if smoke else c["updates"]["steps"])
        and summary["skipped_updates"] <= (0 if smoke else c["updates"]["max_skips"]), "updates differ")
    if not smoke:
        mil.need(summary["cpu_cuda_initialized"] is None and summary["val"] == 888
            and summary["train"] == 3455 and provenance["verification_sha256"] is not None, "full output incomplete")
    inventory = read(output / "output-inventory.json")
    mil.need(set(inventory) == set(mil.OUTPUTS) - {"output-inventory.json"}, "inventory differs")
    for p, r in inventory.items():
        mil.need(sha(output / p) == r["sha256"] and (output / p).stat().st_size == r["bytes"], "output hash differs")
    return summary


def finish(spec, output, note_path, archive, receipt_path, *, smoke=False):
    note = read(note_path)
    life.verify_evidence(spec, note, note_path.parent)
    if not smoke:
        life.require_verification_pass(note, **spec.verification_args)
    summary = check_outputs(spec, output, smoke=smoke)
    if not smoke:
        mil.need(read(output / "provenance.json")["verification_sha256"] == life.content_sha256(note), "run evidence differs")
    receipt = life.artifact_receipt(spec, output, verification_sha256=life.content_sha256(note), exit_code=0,
        elapsed_seconds=summary["elapsed_seconds"], metrics={"attempted_updates": summary["attempted_updates"],
            "val_rows": summary["val"], "engineering_smoke": int(smoke)})
    save(receipt_path, receipt)
    digest = life.content_sha256(receipt)
    archive_digest = life.export_artifacts(spec, output, receipt, archive, expected_receipt_sha256=digest)
    return {"status": "EXPORTED_ENGINEERING_SMOKE" if smoke else "EXPORTED_ARTIFACTS",
            "receipt_sha256": digest, "archive_sha256": archive_digest}


def screen_specs(root, collected, out, *, replay=False):
    """Pin three paired views; flip-TTA primary, others sensitivity checks."""
    root, out = Path(root), Path(out)
    documents = []
    for filename in READOUTS:
        pairs = {}
        for seed in (42, 43):
            spec = spec_for(root, seed)
            control = control_path(seed) + "/" + filename
            mil.need(sha(root / control) == spec.document["inputs"]["controls"][control], "control readout differs")
            arm = root / control if replay else Path(collected) / f"s{seed}" / "outputs" / filename
            if not replay:
                directory = Path(collected) / f"s{seed}"
                receipt = read(directory / "receipt.json")
                mil.need(receipt["metrics"]["engineering_smoke"] == 0, "smoke is not scientific output")
                life.verify_artifact_receipt(spec, directory / "outputs", receipt,
                    expected_receipt_sha256=life.content_sha256(receipt))
                check_outputs(spec, directory / "outputs")
                roster = read(root / spec.document["config"]["roots"]["ordered"])["val"]
                life.load_prediction_csv(arm, roster, mil.TARGETS, id_column=mil.UID)
            pairs[str(seed)] = {"arm": [{"path": arm.resolve().as_posix(), "sha256": sha(arm)}],
                               "control": [{"path": control, "sha256": sha(root / control)}]}
        name = ("CONTROL_REPLAY_" if replay else "HB105_") + filename.removesuffix(".csv")
        doc = {"name": name, "family": "convnext", "label_change": False, "seeds": pairs}
        path = out / (filename.removesuffix(".csv") + ".json")
        save(path, doc)
        documents.append(str(path))
    return {"status": "SCREEN_SPECS_PINNED", "engineering_replay": replay, "specs": documents}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("configure")
    sub.add_parser("verify-package")
    p = sub.add_parser("package")
    for n in ("out", "archive"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--cpu-root", type=Path)
    p = sub.add_parser("combine")
    p.add_argument("--seed", type=int, required=True)
    for n in ("cpu", "target", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--reviewer-name", required=True)
    p.add_argument("--reviewer-host", required=True)
    p = sub.add_parser("finish")
    p.add_argument("--seed", type=int, required=True)
    for n in ("outputs", "note", "archive", "receipt"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--engineering-smoke", action="store_true")
    p = sub.add_parser("screen-specs")
    p.add_argument("--collected", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--control-replay", action="store_true")
    a = ap.parse_args()
    if a.command == "configure":
        result = configure(a.root)
    elif a.command == "package":
        result = package(a.root, a.out, a.archive, a.cpu_root)
    elif a.command == "verify-package":
        result = verify_package(a.root)
    elif a.command == "combine":
        result = combine(spec_for(a.root, a.seed), a.cpu, a.target, a.out, a.reviewer_name, a.reviewer_host)
    elif a.command == "finish":
        result = finish(spec_for(a.root, a.seed), a.outputs, a.note, a.archive, a.receipt, smoke=a.engineering_smoke)
    else:
        mil.need(a.control_replay or a.collected is not None, "collected directory required")
        result = screen_specs(a.root, a.collected, a.out, replay=a.control_replay)
    print(life.canonical_bytes(result).decode())


if __name__ == "__main__":
    main()
