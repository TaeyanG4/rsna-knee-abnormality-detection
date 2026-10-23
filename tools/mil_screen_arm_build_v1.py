"""Generic shared-control arm builder/screen v1.0.0 (2026-10-13), no GPU launcher.

For mil_screen_hooks_v1 arms (H-B-123/H-B-124) paired with the fresh shared
HB122 control fits. Every command prints canonical JSON; artifacts are
create-only (a differing rerun needs a new version). Run from project root:

  configure   --arm M:ARM --work W [--config-dir training/configs]
  cpu-card    --work W --seed S      register/validate/prepare, real CPU smoke
                                     (workers=2), HOLD admission, engineering export
  admit       --work W --seed S --evidence cpu|pass
  combine     --work W --seed S --reviewer-name N --reviewer-host H
  finish      --work W --seed S [--engineering-smoke]
  panel-infer --work W --seed S --side arm|control [--engineering-smoke]
  readout     --work W [--engineering-smoke]
  waiting     --work W

CUDA smoke/fit commands are printed by `commands` and dispatched by B via
tools/gpu_job.py. L7 uses ONE hooked inference path (evaluate_with, K36,
flip mean, batch 4) for both the arm and the shared control checkpoints.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"

from training.arms import mil_screen_hooks_v1 as hooks
from tools import hb105_mil_small as old
from tools import hb122_build_v1 as control_build

life, need, sha, read = hooks.life, hooks.need, hooks.sha, hooks.read
ROOT, CORPUS, WORKERS = control_build.ROOT, control_build.CORPUS, control_build.WORKERS
VERSION = "1.0.0"
PY = "C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe"
PANEL_CACHE = "storage/cache/derived/EXT_OAI_RAPTOR64_336_C140_v2"
PANEL_PATH = "hooked evaluate_with: MILWindows K36 eval centers, batch4, flip-mean, strict runtime"


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def same_or_new(path, value):
    control_build.same_or_new(path, value)


def import_arm(reference):
    import importlib
    module_name, _, attr = reference.partition(":")
    module = importlib.import_module(module_name)
    arm = getattr(module, attr)
    need(isinstance(arm, hooks.Arm) and arm.name and rel(module.__file__) in arm.sources,
         "arm must be a named hooks.Arm whose module is one of its bound sources")
    return arm


def state(work):
    work = Path(work).resolve()
    record = read(work / "arm-build.json")
    need(record["version"] == VERSION and record["work"] == rel(work), "arm build record differs")
    arm = import_arm(record["arm"])
    need(arm.name == record["arm_name"], "arm identity drift")
    return work, record, arm


def spec_for(record, seed):
    return life.ArmSpec((ROOT / record["specs"][str(seed)]).read_bytes())


def configure(reference, work, config_dir):
    arm = import_arm(reference)
    work, config_dir = Path(work).resolve(), Path(config_dir).resolve()
    need(work.is_relative_to(ROOT / ".local/scratch"), "work must live under .local/scratch")
    slug = arm.name.lower().replace("_", "-")
    specs = {}
    for seed in (42, 43):
        control = hooks.control_spec(seed)
        cdoc = control.document
        declaration = cdoc["target_environment"]["evidence"]["declaration.json"]
        target = [p for p, h in cdoc["inputs"]["evidence"].items() if h == declaration]
        need(len(target) == 1, "control target declaration binding differs")
        recipe_control = next(p for p in cdoc["inputs"]["evidence"] if p != target[0])
        rdoc = life.ArmSpec((ROOT / recipe_control).read_bytes()).document
        rdoc["config"][arm.section][arm.key] = arm.contract(seed)
        rdoc["arm"] = f"{arm.name}_RECIPE_S{seed}_F0"
        recipe = config_dir / f"{slug}-recipe-s{seed}-f0-v1.json"
        same_or_new(recipe, life.ArmSpec.from_document(rdoc).document)
        doc = cdoc
        doc["config"][arm.section][arm.key] = arm.contract(seed)
        doc["arm"] = f"{arm.name}_S{seed}_F0"
        doc["inputs"]["sources"] = {p: sha(ROOT / p) for p in sorted(hooks.expected_sources(arm, control))}
        doc["inputs"]["evidence"] = {rel(recipe): sha(recipe), target[0]: declaration}
        spec = life.ArmSpec.from_document(doc)
        hooks.validate_data_with(arm, spec, ROOT, CORPUS)
        path = config_dir / f"{slug}-s{seed}-f0-v1.json"
        same_or_new(path, spec.document)
        specs[str(seed)] = rel(path)
    record = dict(version=VERSION, arm=reference, arm_name=arm.name, work=rel(work), specs=specs,
                  workers=WORKERS, control_specs={str(s): hooks.control_path(s) for s in (42, 43)})
    same_or_new(work / "arm-build.json", record)
    return dict(status="ARM_SPECS_REGISTERED", specs={s: dict(path=p, sha256=sha(ROOT / p)) for s, p in specs.items()})


def prepare(work, seed):
    work, record, arm = state(work)
    spec = spec_for(record, seed)
    run = work / f"s{seed}"
    life.validate_inputs(spec, ROOT)
    same_or_new(run / "registered.json", spec.document)
    if not (run / "stage").exists():
        life.prepare(spec, ROOT, run / "stage")
    life.validate_stage(spec, run / "stage")
    hooks.validate_data_with(arm, spec, run / "stage", CORPUS)
    return dict(status="REGISTER_VALIDATE_PREPARE_PASS", arm=arm.name, seed=seed)


def admit(work, seed, evidence="cpu"):
    work, record, _ = state(work)
    run = work / f"s{seed}"
    name = "verification-pass.json" if evidence == "pass" else "verification.json"
    return life.admission_dry_run(spec_for(record, seed), run / "stage", read(run / evidence / name), run / evidence)


def finish(work, seed, smoke=False):
    work, record, _ = state(work)
    spec = spec_for(record, seed)
    run = work / f"s{seed}"
    note = run / ("cpu/verification.json" if smoke else "pass/verification-pass.json")
    folder = run / ("cpu-export" if smoke else "export")
    folder.mkdir(parents=True, exist_ok=True)
    rec = folder / "export.json"
    if rec.exists():
        result = read(rec)
        need(sha(folder / "artifacts.tar") == result["archive_sha256"], "export archive changed")
    else:
        result = old.finish(spec, run / ("cpu" if smoke else "full") / "outputs", note, folder / "artifacts.tar",
                            folder / "receipt.json", smoke=smoke)
        same_or_new(rec, result)
    dest = run / ("cpu-collected" if smoke else "collected")
    if not dest.exists():
        life.collect_artifacts(spec, folder / "artifacts.tar", dest, expected_receipt_sha256=result["receipt_sha256"])
    life.verify_artifact_receipt(spec, dest / "outputs", read(dest / "receipt.json"),
                                 expected_receipt_sha256=result["receipt_sha256"])
    old.check_outputs(spec, dest / "outputs", smoke=smoke)
    return dict(status="EXPORTED_COLLECTED", seed=seed, engineering_smoke=smoke)


def runner_command(record, seed, command, device, out, extra=()):
    run = f'{record["work"]}/s{seed}'
    return [PY, "-B", "-m", "training.runners.mil_screen_hooks_v1", command, "--arm", record["arm"],
            "--spec", record["specs"][str(seed)], "--stage", f"{run}/stage", "--corpus", rel(CORPUS),
            "--device", device, "--workers", str(WORKERS), "--out", out, *extra]


def cpu_card(work, seed):
    prepare(work, seed)
    work, record, arm = state(work)
    run = work / f"s{seed}"
    if not (run / "cpu").exists():
        command = runner_command(record, seed, "smoke", "cpu", rel(run / "cpu"))
        subprocess.run([sys.executable, *command[1:]], cwd=ROOT, check=True)
    result = admit(work, seed)
    need(result["status"] == "HOLD", "CPU evidence cannot grant target admission")
    finish(work, seed, smoke=True)
    summary = read(run / "cpu/outputs/summary.json")
    need(summary["results"]["optimizer_steps"] == 1 and summary["cpu_cuda_initialized"] is False, "CPU smoke incomplete")
    card = dict(status="CPU_CARD_PASS", arm=arm.name, seed=seed, register=True, validate=True, prepare=True,
                admission="HOLD", export_collect=True, summary=summary)
    same_or_new(run / "cpu-card.json", card)
    return dict(status=card["status"], arm=arm.name, seed=seed, admission="HOLD", results=summary["results"])


def commands(work):
    """Exact B commands (CUDA smoke, reviewed PASS merge, fit, export) per seed."""
    work, record, arm = state(work)
    me = [PY, "-B", "-m", "tools.mil_screen_arm_build_v1"]
    w = ["--work", record["work"]]
    lines = {}
    for seed in (42, 43):
        run, tag = f'{record["work"]}/s{seed}', f'{arm.name.lower()}-s{seed}-f0-v1'
        lines[str(seed)] = [
            [PY, "-B", "tools/gpu_job.py", "--tag", f"{tag}-smoke", "--wait", "--",
             *runner_command(record, seed, "smoke", "cuda", f"{run}/target")],
            [*me, "combine", *w, "--seed", str(seed), "--reviewer-name", "<B>", "--reviewer-host", "<host>"],
            [*me, "admit", *w, "--seed", str(seed), "--evidence", "pass"],
            [PY, "-B", "tools/gpu_job.py", "--tag", f"{tag}-fit", "--wait", "--",
             *runner_command(record, seed, "train", "cuda", f"{run}/full",
                             ("--note", f"{run}/pass/verification-pass.json", "--evidence-root", f"{run}/pass"))],
            [*me, "finish", *w, "--seed", str(seed)],
            [PY, "-B", "tools/gpu_job.py", "--tag", f"{tag}-panel", "--wait", "--",
             *me, "panel-infer", *w, "--seed", str(seed), "--side", "arm"],
            [PY, "-B", "tools/gpu_job.py", "--tag", f"{tag}-ctlpanel", "--wait", "--",
             *me, "panel-infer", *w, "--seed", str(seed), "--side", "control"]]
    return dict(status="COMMANDS", arm=arm.name, commands=lines, readout=[*me, "readout", *w])


def combine(work, seed, name, host):
    work, record, _ = state(work)
    run = work / f"s{seed}"
    need(name and host and not name.startswith("<"), "independent reviewer identity required")
    return old.combine(spec_for(record, seed), run / "cpu", run / "target", run / "pass", name, host)


# ---- seven-lens screen against the fresh shared control ----

def arm_snapshot(work, record, seed, smoke=False):
    from tools import hb122_screen_v1 as screen
    spec = spec_for(record, seed)
    arm = import_arm(record["arm"])
    arm.validate_recipe(spec)
    life.validate_inputs(spec, ROOT)
    run = work / f"s{seed}"
    base = run / ("cpu-collected" if smoke else "collected")
    out = base / "outputs"
    receipt, collected = read(base / "receipt.json"), read(base / "collection.json")
    export = read(run / ("cpu-export" if smoke else "export") / "export.json")
    life.verify_artifact_receipt(spec, out, receipt, expected_receipt_sha256=export["receipt_sha256"])
    need(collected["status"] == "COLLECTED_ARTIFACTS_VERIFIED" and collected["spec_sha256"] == spec.sha256
         and collected["receipt_sha256"] == export["receipt_sha256"], "collection binding differs")
    summary = old.check_outputs(spec, out, smoke=smoke)
    prov = read(out / "provenance.json")
    identity = prov["identity"]
    need(receipt["metrics"]["engineering_smoke"] == int(smoke) and identity["workers"] == WORKERS
         and identity["mode"] == ("smoke" if smoke else "full") and identity["device"] == ("cpu" if smoke else "cuda")
         and identity["batch"] == (2 if smoke else 8) and prov["screen"] == arm.screen, "arm fit identity differs")
    note_root = run / ("cpu" if smoke else "pass")
    note = read(note_root / ("verification.json" if smoke else "verification-pass.json"))
    life.verify_evidence(spec, note, note_root)
    need(life.content_sha256(note) == receipt["verification_sha256"], "receipt evidence differs")
    if not smoke:
        life.require_verification_pass(note, **spec.verification_args)
        life.check_target_observation(spec, summary["observation"])
        r = spec.document["config"]["roots"]
        need(prov["train_ids_ordered_sha256"] == r["train_sha256"] and prov["eval_ids_ordered_sha256"] == r["val_sha256"]
             and prov["checkpoint_selection"] == "fixed-final" and prov["verification_sha256"] == receipt["verification_sha256"],
             "heldout ancestry differs")
    files = (base / "receipt.json", base / "collection.json", out / "model_best.pt", out / screen.PRIMARY,
             out / "provenance.json", out / "summary.json", out / "output-inventory.json", ROOT / record["specs"][str(seed)])
    return dict(key=f"{arm.name.lower()}-s{seed}", seed=seed, spec_sha256=spec.sha256, engineering_smoke=smoke,
                runtime_sha256=life.content_sha256(identity["observation"]["runtime"]),
                checkpoint=rel(out / "model_best.pt"), prediction=rel(out / screen.PRIMARY),
                bindings={rel(p): sha(p) for p in files})


def panel_paths(work, seed, side, smoke):
    folder = work / ("panel-smoke" if smoke else "panel-hooked-v1")
    path = folder / f"{side}-s{seed}.csv"
    return path, Path(str(path) + ".json")


def panel_infer(work, seed, side, smoke=False):
    """Same hooked inference path for arm and shared-control checkpoints."""
    import torch
    import cv2
    from tools import hb122_screen_v1 as screen
    from tools import oai_moaks_panel as p
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    work, record, arm = state(work)
    need(side in ("arm", "control"), "panel side differs")
    path, receipt_path = panel_paths(work, seed, side, smoke)
    need(not path.exists() and not receipt_path.exists(), "panel output exists; authenticate or version it")
    if side == "arm":
        snap, spec, use_arm = arm_snapshot(work, record, seed, smoke), spec_for(record, seed), arm
    else:
        snap, spec, use_arm = screen.snapshot("control", seed, smoke), control_build.spec_for("control", seed), hooks.IDENTITY
    ids = p.validate()
    use_ids = ids[:2] if smoke else ids
    need(smoke or len(use_ids) == 557, "full panel size differs")
    device = "cpu" if smoke else "cuda"
    amp = "none" if smoke else "fp16"
    hooks.configure_runtime(amp, device, policy="strict")
    observed = hooks.observation(device)
    if not smoke:
        life.check_target_observation(spec, observed)
    corpus = hooks.common.Corpus(ROOT / PANEL_CACHE)
    need(p.CACHES["64"] == PANEL_CACHE and p.ROOT == ROOT, "registered panel cache differs")
    hooks.verify_shards(corpus, use_ids)
    checkpoint = torch.load(ROOT / snap["checkpoint"], map_location="cpu", weights_only=False)
    need(checkpoint["provenance"]["spec_sha256"] == spec.sha256 and checkpoint["epoch"] == (0 if smoke else 19)
         and sha(ROOT / snap["checkpoint"]) == snap["bindings"][snap["checkpoint"]], "panel checkpoint differs")
    c = spec.document["config"]
    model = use_arm.attach(hooks.build_model(c["model"], ROOT), spec, ROOT)
    hooks.strict_load(model, checkpoint["model"])
    del checkpoint
    model = model.to(device)
    zeros = {u: hooks.np.zeros(12, hooks.np.float32) for u in use_ids}
    ds = use_arm.dataset(hooks.MILWindows(ROOT / PANEL_CACHE, use_ids, zeros, 36, False, spec.document["seed"]),
                         spec, ROOT, train=False)
    loader = hooks.DataLoader(ds, batch_size=4, shuffle=False, num_workers=min(WORKERS, 2),
                              collate_fn=use_arm.collate_fn, worker_init_fn=hooks.worker_init)
    start = time.monotonic()
    values, ordered, _ = hooks.evaluate_with(use_arm, model, loader, device, amp, corpus, tta=True)
    need(ordered == use_ids and values.shape == (len(use_ids), 12) and hooks.np.isfinite(values).all()
         and ((values >= 0) & (values <= 1)).all(), "invalid panel forward")
    life.create_new(path, life.prediction_csv_bytes(use_ids, values, hooks.TARGETS, id_column="study_key"))
    receipt = dict(version=VERSION, side=side, seed=seed, arm=record["arm"] if side == "arm" else "identity",
                   snapshot=snap, path=PANEL_PATH, cache_manifest_sha256=sha(ROOT / PANEL_CACHE / "manifest.json"),
                   predictions_sha256=sha(path), rows=len(use_ids), ids_sha256=hooks.uid_hash(use_ids),
                   engineering_smoke=smoke, device=device, observation=observed,
                   seconds=time.monotonic() - start, sources=screen.source_pins())
    same_or_new(receipt_path, receipt)
    return dict(status="PANEL_INFERRED", side=side, seed=seed, rows=len(use_ids), seconds=receipt["seconds"])


def panel_prediction(work, record, seed, side, smoke, snap):
    from tools import hb122_screen_v1 as screen
    from tools import oai_moaks_panel as p
    path, receipt_path = panel_paths(work, seed, side, smoke)
    receipt = read(receipt_path)
    ids = p.validate()
    ids = ids[:2] if smoke else ids
    need(receipt["version"] == VERSION and receipt["side"] == side and receipt["seed"] == seed
         and receipt["snapshot"] == snap and receipt["path"] == PANEL_PATH
         and receipt["predictions_sha256"] == sha(path) and receipt["rows"] == len(ids)
         and receipt["ids_sha256"] == hooks.uid_hash(ids) and receipt["engineering_smoke"] is smoke
         and receipt["device"] == ("cpu" if smoke else "cuda")
         and receipt["arm"] == (record["arm"] if side == "arm" else "identity"), "panel binding differs")
    return ids, screen.load_predictions(path, ids, "study_key"), receipt


def readout(work, smoke=False):
    from tools import hb122_screen_v1 as screen
    work, record, arm = state(work)
    controls = {str(s): screen.snapshot("control", s, smoke) for s in (42, 43)}
    arms = {str(s): arm_snapshot(work, record, s, smoke) for s in (42, 43)}
    need(len({v["runtime_sha256"] for v in (*controls.values(), *arms.values())}) == 1, "fits ran on different runtimes")
    panels, receipts, panel_ids = {"arm": {}, "control": {}}, {}, None
    for side, snaps in (("arm", arms), ("control", controls)):
        for seed, snap in snaps.items():
            ids, values, receipt = panel_prediction(work, record, int(seed), side, smoke, snap)
            need(panel_ids is None or ids == panel_ids, "panel paired roster differs")
            panel_ids = ids
            panels[side][seed] = values
            if not smoke:
                life.check_target_observation(spec_for(record, int(seed)) if side == "arm"
                                              else control_build.spec_for("control", int(seed)), receipt["observation"])
            receipts[f"{side}-s{seed}"] = sha(panel_paths(work, int(seed), side, smoke)[1])
    from tools import oai_moaks_panel as p
    y = p.outcomes(panel_ids)
    from tools import hb105_aligned_oof_v1 as aligned
    raw, rosters, auth = aligned.authenticate()
    ids = rosters[0]["val"]
    arrays = {side: {seed: (raw[0]["S" + seed].copy() if smoke else
                            screen.load_predictions(ROOT / snap["prediction"], ids)) for seed, snap in snaps.items()}
              for side, snaps in (("arm", arms), ("control", controls))}
    arm_bundle, control_bundle = screen.fixed_bundle(raw, arrays["arm"], arrays["control"], aligned.variants)
    ctx = screen.ml.make_context(ids, ROOT)
    need(ctx.bindings == auth["ruler_bindings"], "ruler binding changed")
    bundle = dict(context=ctx, arm=arm_bundle, control=control_bundle, recipe=dict(scope="ALIGNED_OOF_SURROGATE_ONLY",
        change="only fold0 S42/S43 replaced by arm vs fresh control; folds1-4 fixed",
        arithmetic=aligned.POLICY["arithmetic"], final_proxy=screen.FINAL_PROXY))
    result = screen.evaluate(ctx, arrays["arm"], arrays["control"], bundle,
                             screen.image_item(panel_ids, panels["arm"], panels["control"], y))
    result.update(version=VERSION, arm=arm.name, status="ENGINEERING_CONTROL_REPLAY" if smoke else "COMPLETE_FOLD0_SCREEN",
                  fitted_members=dict(arm=arms, control=controls), panel_receipts=receipts, panel_path=PANEL_PATH,
                  aligned_authentication=auth, ruler_bindings=ctx.bindings, rows=len(ids), panel_rows=len(panel_ids),
                  engineering_smoke=smoke, sources=screen.source_pins())
    for snap in (*arms.values(), *controls.values()):
        for path, digest in snap["bindings"].items():
            need(sha(ROOT / path) == digest, "member changed during scoring")
    output = work / ("screen-cpu-smoke-v1.json" if smoke else "screen-seven-lens-v1.json")
    same_or_new(output, result)
    return dict(status=result["status"], output=rel(output), rows=len(ids), panel_rows=len(panel_ids),
                lenses=list(result["lenses"]))


def waiting(work):
    work, record, _ = state(work)
    missing = []
    for seed in (42, 43):
        for path in (work / f"s{seed}/collected/receipt.json", ROOT / f".local/scratch/claude-hb122/control-s{seed}/collected/receipt.json",
                     *panel_paths(work, seed, "arm", False), *panel_paths(work, seed, "control", False)):
            if not path.is_file():
                missing.append(rel(path))
    return dict(status="WAITING" if missing else "READY_FOR_READOUT", missing=missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("configure", "prepare", "cpu-card", "admit", "combine", "finish", "commands",
                                        "panel-infer", "readout", "waiting"))
    ap.add_argument("--arm")
    ap.add_argument("--work", required=True)
    ap.add_argument("--config-dir", default="training/configs")
    ap.add_argument("--seed", type=int, choices=(42, 43))
    ap.add_argument("--evidence", choices=("cpu", "pass"), default="cpu")
    ap.add_argument("--side", choices=("arm", "control"))
    ap.add_argument("--engineering-smoke", action="store_true")
    ap.add_argument("--reviewer-name")
    ap.add_argument("--reviewer-host")
    a = ap.parse_args()
    if a.command == "configure":
        need(a.arm is not None, "--arm required")
        result = configure(a.arm, a.work, a.config_dir)
    elif a.command in ("commands", "readout", "waiting"):
        result = commands(a.work) if a.command == "commands" else (
            readout(a.work, a.engineering_smoke) if a.command == "readout" else waiting(a.work))
    else:
        need(a.seed is not None, "--seed required")
        if a.command == "prepare":
            result = prepare(a.work, a.seed)
        elif a.command == "cpu-card":
            result = cpu_card(a.work, a.seed)
        elif a.command == "admit":
            result = admit(a.work, a.seed, a.evidence)
        elif a.command == "combine":
            result = combine(a.work, a.seed, a.reviewer_name or "", a.reviewer_host or "")
        elif a.command == "finish":
            result = finish(a.work, a.seed, a.engineering_smoke)
        else:
            need(a.side is not None, "--side required")
            result = panel_infer(a.work, a.seed, a.side, a.engineering_smoke)
    print(life.canonical_bytes(result).decode())


if __name__ == "__main__":
    main()
