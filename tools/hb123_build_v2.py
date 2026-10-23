"""H-B-123 lifecycle v2.0.0. Changelog: workers2 CPU evidence in fresh v2 stages. Derive from published shared controls; no fits of controls."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"

from training.arms import lifecycle as life, mil_laterality_v2 as mil
from tools import hb105_mil_small as old

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local/scratch/sol-5/hb123/v2"
CORPUS = ROOT / "storage/cache/derived/RAPTOR64_336_C140_S02"


def spec_path(seed):
    mil.control_path(seed)
    return ROOT / f"training/configs/hb123-right-s{seed}-f0-v2.json"


def spec_for(seed):
    return life.ArmSpec(spec_path(seed).read_bytes())


def same_or_new(path, value):
    path = Path(path)
    raw = life.canonical_bytes(value)
    if path.exists():
        mil.need(path.read_bytes() == raw, "immutable artifact differs; new version required")
    else:
        life.create_new(path, raw)


def configure():
    results = []
    # All dependencies must be published before any new spec is written.
    for seed in (42, 43):
        control = life.ArmSpec((ROOT / mil.control_path(seed)).read_bytes())
        mil.shared.validate_data(control, ROOT, CORPUS)
    for seed in (42, 43):
        control = life.ArmSpec((ROOT / mil.control_path(seed)).read_bytes())
        doc = control.document
        doc["config"]["windows"]["laterality_v1"] = mil.rendering_contract(seed)
        recipe = ROOT / f"training/configs/hb123-right-recipe-s{seed}-f0-v2.json"
        # Lifecycle recipe evidence is a baseline scientific spec, never executable.
        original = life.ArmSpec((ROOT / f"training/configs/hb105-small-s{seed}-f0-v1.json").read_bytes()).document
        original.update(arm=f"HB123_RIGHT_RECIPE_S{seed}_F0", attempt="v2", config=doc["config"])
        same_or_new(recipe, life.ArmSpec.from_document(original).document)
        target_digest = doc["target_environment"]["evidence"]["declaration.json"]
        target = next(p for p, h in doc["inputs"]["evidence"].items() if h == target_digest)
        doc.update(arm=f"HB123_RIGHT_S{seed}_F0", attempt="v2")
        sources = dict(doc["inputs"]["sources"])
        sources.update({p: mil.sha(ROOT / p) for p in mil.EXTRA_SOURCES + [mil.control_path(seed)]})
        doc["inputs"]["sources"] = sources
        doc["inputs"]["evidence"] = {recipe.relative_to(ROOT).as_posix(): mil.sha(recipe), target: target_digest}
        spec = life.ArmSpec.from_document(doc)
        mil.validate_data(spec, ROOT, CORPUS)
        same_or_new(spec_path(seed), spec.document)
        results.append(dict(seed=seed, spec_sha256=spec.sha256, control_spec_sha256=control.sha256))
    return dict(status="TWO_ARM_SPECS_REGISTERED", specs=results)


def prepare(seed):
    spec = spec_for(seed)
    run = WORK / f"s{seed}"
    life.validate_inputs(spec, ROOT)
    same_or_new(run / "registered.json", spec.document)
    life.prepare(spec, ROOT, run / "stage")
    life.validate_stage(spec, run / "stage")
    _, train, val, _, _ = mil.validate_data(spec, run / "stage", CORPUS)
    mil.side_map_for(spec, run / "stage").require(train + val)
    return dict(status="REGISTER_VALIDATE_PREPARE_PASS", seed=seed)


def admit(seed, evidence="cpu"):
    run = WORK / f"s{seed}"
    name = "verification-pass.json" if evidence == "pass" else "verification.json"
    return life.admission_dry_run(spec_for(seed), run / "stage", mil.read(run / evidence / name), run / evidence)


def finish(seed, smoke=False):
    spec = spec_for(seed)
    run = WORK / f"s{seed}"
    mode = "cpu" if smoke else "full"
    note = run / ("cpu/verification.json" if smoke else "pass/verification-pass.json")
    folder = run / ("cpu-export" if smoke else "export")
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / "export.json"
    if record.exists():
        result = mil.read(record)
        mil.need(mil.sha(folder / "artifacts.tar") == result["archive_sha256"], "export archive changed")
    else:
        result = old.finish(spec, run / mode / "outputs", note, folder / "artifacts.tar", folder / "receipt.json", smoke=smoke)
        same_or_new(record, result)
    dest = run / ("cpu-collected" if smoke else "collected")
    if not dest.exists():
        life.collect_artifacts(spec, folder / "artifacts.tar", dest, expected_receipt_sha256=result["receipt_sha256"])
    life.verify_artifact_receipt(spec, dest / "outputs", mil.read(dest / "receipt.json"),
                                expected_receipt_sha256=result["receipt_sha256"])
    old.check_outputs(spec, dest / "outputs", smoke=smoke)
    return dict(status="EXPORTED_COLLECTED", seed=seed, engineering_smoke=smoke)


def cpu_card(seed):
    prepare(seed)
    run = WORK / f"s{seed}"
    if not (run / "cpu").exists():
        subprocess.run([sys.executable, "-B", "-m", "training.runners.mil_laterality_v2", "smoke",
            "--spec", str(spec_path(seed)), "--stage", str(run / "stage"), "--corpus", str(CORPUS),
            "--device", "cpu", "--workers", "2", "--out", str(run / "cpu")], cwd=ROOT, check=True)
    provenance = mil.read(run / "cpu/outputs/provenance.json")
    mil.need(provenance["identity"]["workers"] == 2, "CPU loader differs from control")
    result = admit(seed)
    mil.need(result["status"] == "HOLD", "CPU evidence cannot grant target admission")
    finish(seed, smoke=True)
    summary = mil.read(run / "cpu/outputs/summary.json")
    mil.need(summary["results"]["optimizer_steps"] == 1 and summary["cpu_cuda_initialized"] is False,
             "CPU smoke incomplete")
    # Independently reload the arm checkpoint for every registered readout.
    ids = mil.read(run / "stage" / spec_for(seed).document["config"]["roots"]["ordered"])["val"][:1]
    same_or_new(run / "infer-ids.json", ids)
    parity = {}
    for k, tta, filename, served in ((24, False, "oof_best.csv", "cpu-infer-k24.csv"),
                                    (36, False, "oof_k36.csv", "cpu-infer-k36.csv"),
                                    (36, True, "oof_k36_flip_tta.csv", "cpu-infer.csv")):
        out = run / served
        if not out.exists():
            command = [sys.executable, "-B", "-m", "training.runners.mil_laterality_v2", "infer",
                "--spec", str(spec_path(seed)), "--stage", str(run / "stage"), "--corpus", str(CORPUS),
                "--checkpoint", str(run / "cpu-collected/outputs/model_best.pt"), "--ids", str(run / "infer-ids.json"),
                "--out", str(out), "--k", str(k)]
            if not tta:
                command.append("--no-tta")
            subprocess.run(command, cwd=ROOT, check=True)
        mil.need(out.read_bytes() == (run / "cpu/outputs" / filename).read_bytes(), "serving parity differs")
        parity[filename] = dict(byte_equal=True, sha256=mil.sha(out))
    card = dict(status="CPU_CARD_PASS", seed=seed, register=True, validate=True, prepare=True,
                admission="HOLD", export_collect=True, serving_byte_parity=True, readout_parity=parity, summary=summary)
    same_or_new(run / "cpu-card.json", card)
    return {k: card[k] for k in ("status", "seed", "admission", "serving_byte_parity")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("configure", "prepare", "cpu-card", "admit", "combine", "finish"))
    ap.add_argument("--seed", choices=(42, 43), type=int, default=42)
    ap.add_argument("--evidence", choices=("cpu", "pass"), default="cpu")
    ap.add_argument("--reviewer-name")
    ap.add_argument("--reviewer-host")
    a = ap.parse_args(argv)
    if a.command == "configure":
        result = configure()
    elif a.command == "combine":
        mil.need(a.reviewer_name and a.reviewer_host, "independent reviewer required")
        run = WORK / f"s{a.seed}"
        result = old.combine(spec_for(a.seed), run / "cpu", run / "target", run / "pass",
                             a.reviewer_name, a.reviewer_host)
    elif a.command == "admit":
        result = admit(a.seed, a.evidence)
    else:
        result = {"prepare": prepare, "cpu-card": cpu_card, "finish": finish}[a.command](a.seed)
    print(life.canonical_bytes(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
