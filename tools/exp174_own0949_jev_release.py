"""EXP174 v1.0.0: private, local-only release builder and future B plans.

Reuse SHA-pinned EXP152 averaging, serving, notebook and parity arithmetic.
No CLI command runs CUDA, uploads, submits, reads credentials or downloads.
public3-plan creates a runner for B to execute later on a claimed GPU; it does
not execute that runner. Kaggle notebook runs and account commands are plans.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

VERSION = "1.0.0"
TOOL = Path(__file__).resolve()
ROOT = TOOL.parents[1]
BASE = ROOT / "artifacts/experiments/EXP174_OWN0949_JEV"
PROTOCOL = BASE / "protocol-v1/manifest.json"
RUN = "artifacts/experiments/EXP174_OWN0949_JEV/S42"
OWN_NAME = "exp174-jev-e14-e11-e12-bn.pt"
ASSET_DATASET = "taeyangg4/knee-exp174-own0949-jev-v1-assets"
KERNELS = {"own": "knee-exp174-own0949-jev-v1"}
BASE_NAME = "exp152_own0949_release.py"
BASE_SHA256 = "ebb8e06734b72f40f090389b23d8767137c8eb6c92dee547367c1f18188f94e2"
BASE_PATH = TOOL.with_name(BASE_NAME)


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Isolated module: adaptations cannot change an independently imported EXP152.
# The authentic original is packaged beside this adapter in the private archive.
if _file_sha(BASE_PATH) != BASE_SHA256:
    raise ValueError("Pinned EXP152 release implementation has changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_shared_exp152", BASE_PATH)
_shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _shared
_spec.loader.exec_module(_shared)
for _key in ("VERSION", "TOOL", "ROOT", "PROTOCOL", "RUN", "OWN_NAME", "ASSET_DATASET", "KERNELS"):
    setattr(_shared, _key, globals()[_key])

ARCH, CONTRACT, TARGETS, EPOCHS = _shared.ARCH, _shared.CONTRACT, _shared.TARGETS, _shared.EPOCHS
PUBLIC_NAME, PUBLIC_SHA, PUBLIC_DATASET = _shared.PUBLIC_NAME, _shared.PUBLIC_SHA, _shared.PUBLIC_DATASET
SOURCE_SHA, TOL, POLICY = _shared.SOURCE_SHA, _shared.TOL, _shared.POLICY
need, sha, read_json = _shared.need, _shared.sha, _shared.read_json
write_text, fresh_dir, safe_load = _shared.write_text, _shared.fresh_dir, _shared.safe_load
state_schema = _shared.state_schema
serving_source, public_source = _shared.serving_source, _shared.public_source
_original_load_module = _shared.load_module
_original_own_meta = _shared.own_meta
_original_notebook = _shared.notebook
_original_write_json = _shared.write_json
_original_verify_assets = _shared.verify_assets
_original_projection = _shared.projection
_original_bind = _shared.bind
_qualification_active = False


def load_module(name, path):
    path = Path(path)
    if path.resolve() == (ROOT / "tools/exp143_v2.py").resolve():
        path = ROOT / "tools/exp174_own0949_jev.py"
    return _original_load_module(name, path)


def own_meta(checkpoint, protocol_sha, run_dir=RUN):
    _original_own_meta(checkpoint, protocol_sha, run_dir)


def write_json(path, value):
    if value.get("schema") == "exp152-average-v1":
        value = dict(value, schema="exp174-average-v1", shared_serving_sha256=BASE_SHA256)
    elif value.get("schema") == "exp152-parity-plan-v1":
        value = dict(value, schema="exp174-parity-plan-v1", registered_release_variants=["own"],
                     sequence=["Local Public3 verifies decoded uint8/mask/centers and normalized windows against the frozen local predictor",
                               "Local own raw probabilities and own final CSV must match predictor-derived references; public probabilities are checked internally for unchanged EXP152 qualification",
                               "Push only the private own kernel after local PASS; B downloads its full outputs",
                               "compare-parity gates exact own input hashes, finite raw max2e-3/mean5e-4 and own CSV bytes; retain runtime differences"])
    return _original_write_json(path, value)


def average(args):
    """CPU-only exact EXP152 fixed [14,11,12] SWA + authenticated trainer BN."""
    guard = load_module("exp174_release_completion_guard", ROOT / "tools/exp143_v2.py")
    guard.require_completed(args.protocol)
    return _shared.average(args)


def notebook(variant, dataset, archive_sha, manifest_sha, module_sha, original_markdown):
    need(variant == "own", "EXP174 release registers own only; future blends need separate registration")
    nb, _, _ = _original_notebook(variant, dataset, archive_sha, manifest_sha,
                                  module_sha, original_markdown)
    code = nb["cells"][-1]["source"]
    code = code.replace("exp152-assets", "exp174-assets")
    code = _shared.replace_once(code, "'exp152_own0949_release.py', 'public-serving.py'",
                               "'exp174_own0949_jev_release.py', 'exp152_own0949_release.py', 'public-serving.py'")
    code = _shared.replace_once(code, "_hash(_assets/'exp152_own0949_release.py')",
                               "_hash(_assets/'exp174_own0949_jev_release.py')")
    code = _shared.replace_once(code, "import exp152_own0949_release as _e152",
                               "import exp174_own0949_jev_release as _e152")
    code = code.replace("private EXP152", "private EXP174")
    compile(code, "exp174-notebook", "exec")
    nb["cells"][-1]["source"] = code
    title = "EXP174 Dread75/JEV25 OAI-free own checkpoint"
    description = ("EXP143 v2 OAI-free recipe, with only Dread75/JEV25 labels replacing Dread75/Aquila25. "
                   "Fixed epochs14/11/12, registered train-only BN, seed42 and Gold58 excluded. "
                   "Identical EXP152 own serving: K94, native320 crop, anatomical mirror, fp16 chunk16. "
                   "Private dataset and notebook; internet off; no training labels, source study IDs, "
                   "reports or raw study data. Every submission requires Taeyang's chat instruction. ")
    description += "Original author notices retained below; executable source is SHA-pinned."
    nb["cells"][0]["source"] = "# " + title + "\n\n" + description
    return nb, title, description


def verify_assets(path, manifest):
    _original_verify_assets(path, manifest)
    need(manifest.get("schema") == "exp174-release-v1", "EXP174 release schema mismatch")
    need(manifest["checkpoints"][OWN_NAME].get("oai") is False, "EXP174 must be OAI-free")
    need(set(manifest["asset_sha256"]) == {OWN_NAME, TOOL.name, BASE_NAME, "public-serving.py"},
         "Unexpected release payload")
    need(manifest["asset_sha256"][BASE_NAME] == BASE_SHA256, "Shared serving source mismatch")
    need(manifest.get("registered_release_variants") == ["own"] and set(manifest["ranks"]) == {"own"},
         "Only the registered own release is permitted")


def private_payload(checkpoint):
    """Allow only inference fields and hashes; strip all training metadata."""
    identity = checkpoint["provenance"]
    need(set(identity) <= {"protocol_sha256", "initial_state_sha256", "seed", "precision", "world_size",
         "contract", "train_n", "torch", "run_dir", "execution"}, "Unreviewed provenance fields")
    for key in ("protocol_sha256", "initial_state_sha256"):
        need(isinstance(identity.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", identity[key]),
             "Missing or malformed provenance hash: " + key)
    need(identity.get("run_dir") == RUN and identity.get("train_n") == 4347
         and identity.get("seed") == 42 and identity.get("world_size") == 1,
         "Wrong registered training identity")
    execution = identity.get("execution", {})
    need(isinstance(execution, dict) and execution.get("grad_checkpointing") is False
         and execution.get("batch") == 1 and execution.get("effective_batch") == 16,
         "Wrong EXP174 execution")
    allowed = {"model", "arch", "res", "inference_res", "lab", "contract", "k_eval", "norm",
               "window_policy", "oai", "checkpoint_kind", "swa_epochs_1based", "seed"}
    return {**{key: checkpoint[key] for key in allowed},
            "provenance": {"protocol_sha256": identity["protocol_sha256"], "run_dir": RUN,
                           "initial_state_sha256": identity["initial_state_sha256"]}}


def stage(args):
    receipt = read_json(Path(args.average) / "average-receipt.json")
    need(receipt.get("schema") == "exp174-average-v1" and receipt.get("oai") is False
         and receipt.get("status") == "VERIFIED_FIXED_SWA_BN" and receipt.get("epochs_1based") == EPOCHS
         and receipt.get("run_dir") == RUN and receipt.get("shared_serving_sha256") == BASE_SHA256,
         "EXP174 average receipt invalid")
    need(receipt["protocol_sha256"] == sha(PROTOCOL), "Registered protocol SHA mismatch")
    guard = load_module("exp174_release_stage_guard", ROOT / "tools/exp143_v2.py")
    protocol = guard.load_registration(PROTOCOL)
    need(len(protocol["train_ids"]) == 4347 and protocol["recipe"]["oai"] is False,
         "Registered cohort mismatch")
    checkpoint = Path(args.average) / OWN_NAME
    need(sha(checkpoint) == receipt["checkpoint_sha256"], "Average checkpoint SHA mismatch")
    ck = safe_load(checkpoint)
    own_meta(ck, receipt["protocol_sha256"])
    need(state_schema(ck["model"]) == receipt["state_schema"], "Average schema mismatch")
    payload = private_payload(ck)
    need(re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+", args.dataset), "Invalid dataset ID")
    source, credits = public_source()
    out = fresh_dir(args.out)
    assets = fresh_dir(out / "assets")
    import torch
    with (assets / OWN_NAME).open("xb") as stream:
        torch.save(payload, stream)
    shutil.copyfile(TOOL, assets / TOOL.name)
    shutil.copyfile(BASE_PATH, assets / BASE_NAME)
    write_text(assets / "public-serving.py", serving_source(source))
    manifest = dict(schema="exp174-release-v1", version=VERSION,
        protocol_sha256=receipt["protocol_sha256"], epochs_1based=EPOCHS,
        average_receipt_sha256=sha(Path(args.average) / "average-receipt.json"),
        source_final_sha256=receipt["source_final_sha256"], source_average_sha256=receipt["checkpoint_sha256"],
        public_source_sha256=SOURCE_SHA, serving_sha256=sha(assets / "public-serving.py"),
        # Public checkpoint is qualification-only; never packaged or attached
        # to the own notebook. Its digest authenticates the unchanged EXP152
        # local public-probability reference check.
        checkpoints={OWN_NAME: dict(sha256=sha(assets / OWN_NAME), oai=False),
                     PUBLIC_NAME: dict(sha256=PUBLIC_SHA, oai=True)},
        own_state_schema=receipt["state_schema"],
        asset_sha256={name: sha(assets / name) for name in (OWN_NAME, TOOL.name, BASE_NAME, "public-serving.py")},
        ranks=dict(own="public numpy double argsort / (N-1)"),
        policy=POLICY, tolerance=TOL, dataset=args.dataset,
        source_bindings_sha256=_shared.jhash(protocol["source_bindings"]),
        approval_required={"own": "each submission requires Taeyang chat instruction"},
        public_reference_role="local EXP152 Public3 qualification only; no public release arm",
        registered_release_variants=["own"])
    write_json(assets / "release-manifest.json", manifest)
    verify_assets(assets, manifest)
    dataset = fresh_dir(out / "dataset")
    archive = dataset / "exp174-assets.zip.bin"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(assets.iterdir()):
            bundle.write(path, path.name)
    write_json(dataset / "dataset-metadata.json", dict(title=args.dataset.split("/")[1], id=args.dataset,
               licenses=[{"name": "other"}], description="Private EXP174 OAI-free model weights and EXP152 serving code. No source data, labels, study IDs or reports."))
    kernels = {}
    for variant in KERNELS:
        dest = fresh_dir(out / variant)
        nb, title, description = notebook(variant, args.dataset, sha(archive), sha(assets / "release-manifest.json"),
                                          sha(assets / TOOL.name), credits)
        write_json(dest / "release.ipynb", nb)
        write_json(dest / "kernel-metadata.json", kernel_metadata(variant, args.dataset))
        write_text(dest / "DESCRIPTION.md", "# " + title + "\n\n" + description + "\n")
        kernels[variant] = {name: sha(dest / name) for name in ("release.ipynb", "kernel-metadata.json", "DESCRIPTION.md")}
    report = dict(status="STAGED_NOT_UPLOADED", version=VERSION, private_required=True,
                  archive_sha256=sha(archive), manifest_sha256=sha(assets / "release-manifest.json"),
                  dataset_metadata_sha256=sha(dataset / "dataset-metadata.json"),
                  kernels=kernels, projection=projection(), dataset=args.dataset)
    write_json(out / "build-receipt.json", report)
    return dict(status=report["status"], out=str(out), archive_sha256=report["archive_sha256"])


def checked_stage(stage_dir):
    path = Path(stage_dir).resolve()
    report = read_json(path / "build-receipt.json")
    need(report.get("version") == VERSION, "Stage version mismatch")
    for file, key in (("assets/release-manifest.json", "manifest_sha256"),
                      ("dataset/exp174-assets.zip.bin", "archive_sha256"),
                      ("dataset/dataset-metadata.json", "dataset_metadata_sha256")):
        need(sha(path / file) == report[key], "Stage hash mismatch: " + file)
    manifest = read_json(path / "assets/release-manifest.json")
    verify_assets(path / "assets", manifest)
    need(sha(TOOL) == manifest["asset_sha256"][TOOL.name], "Executing tool differs from staged tool")
    for variant, files in report["kernels"].items():
        for name, digest in files.items():
            need(sha(path / variant / name) == digest, "Kernel build hash mismatch")
    return path, report, manifest


def projection():
    result = _original_projection()
    result.pop("blend_seconds", None)
    result.pop("conservative_blend_seconds", None)
    result["caveat"] = result["caveat"].replace("EXP152", "EXP174")
    result["basis"] = ("D-B-065 corrected timing60: 88s/58 ~=1.52s/study, K94+mirror 2xT4. "
                       "Own setup180s and 1.5x+300s are allowances, not measurements.")
    result["checkpoint_effect"] = "Identical EXP152 own graph/decode; label swap changes weights only. Projection, not measured."
    return result


def bind(ns, manifest, files, variant, work):
    """Own serving, plus an internal qualification-only two-model binding.

    EXP152 Public3 independently validates own and public raw probabilities and
    starts by binding both models. Temporarily admit that binding only while the
    explicit future qualification runner is active; release variants stay own.
    """
    if variant != "blend":
        return _original_bind(ns, manifest, files, variant, work)
    need(_qualification_active, "Blend is not registered; internal qualification only")
    original_kernels = _shared.KERNELS
    _shared.KERNELS = {**original_kernels, "blend": "qualification-reference-only"}
    try:
        return _original_bind(ns, manifest, files, variant, work)
    finally:
        _shared.KERNELS = original_kernels


def _qualification_public3(args):
    """Called only by B's separately generated runner, never by this CLI."""
    global _qualification_active
    need(not _qualification_active, "Public3 qualification already active")
    _qualification_active = True
    try:
        return _shared.public3(args)
    finally:
        _qualification_active = False


# No public3 symbol is exposed as a CLI command. Its sole execution path is the
# separately generated B runner; all GPU/account actions remain explicit later.
for _name in ("load_module", "own_meta", "write_json", "notebook", "verify_assets", "checked_stage", "projection", "bind"):
    setattr(_shared, _name, globals()[_name])
def kernel_metadata(variant, dataset=ASSET_DATASET):
    # Frozen function defaults retain EXP152's dataset unless explicitly passed.
    return _shared.kernel_metadata(variant, dataset)


checked_model = _shared.checked_model
output_values, percentile_average = _shared.output_values, _shared.percentile_average
save_raw, finish_receipt = _shared.save_raw, _shared.finish_receipt
parity_plan, compare_parity = _shared.parity_plan, _shared.compare_parity


def public3_plan(args):
    stage_dir, report, _ = checked_stage(args.stage)
    _shared.verify_plan(args.plan, report)
    need(sha(args.protocol) == read_json(stage_dir / "assets/release-manifest.json")["protocol_sha256"],
         "Public3 protocol mismatch")
    # Serialize structured arguments: never interpolate them into Python code.
    out = fresh_dir(args.out)
    config = dict(stage=str(stage_dir), plan=str(Path(args.plan).resolve()), protocol=str(Path(args.protocol).resolve()),
                  public_checkpoint=str(Path(args.public_checkpoint).resolve()),
                  competition=str(Path(args.competition).resolve()), dicom_root=args.dicom_root,
                  device=args.device, out=str(Path(args.result_out).resolve()))
    write_json(out / "runner-arguments.json", config)
    runner = ("# B-only future GPU execution. Generated by a plan-only CPU command.\n"
              "import hashlib, importlib.util, json, sys\n"
              "from pathlib import Path\nfrom types import SimpleNamespace\n"
              f"tool = Path({str(TOOL)!r})\n"
              f"expected = {sha(TOOL)!r}\n"
              "with tool.open('rb') as f:\n    actual = hashlib.file_digest(f, 'sha256').hexdigest()\n"
              "if actual != expected: raise ValueError('Adapter changed')\n"
              "spec = importlib.util.spec_from_file_location('exp174_b_public3', tool)\n"
              "module = importlib.util.module_from_spec(spec)\nsys.modules[spec.name] = module\nspec.loader.exec_module(module)\n"
              "config = Path(__file__).with_name('runner-arguments.json')\n"
              "with config.open('rb') as f:\n    actual = hashlib.file_digest(f, 'sha256').hexdigest()\n"
              f"if actual != {sha(out / 'runner-arguments.json')!r}: raise ValueError('Runner arguments changed')\n"
              "args = SimpleNamespace(**json.loads(config.read_text(encoding='utf-8')))\n"
              "print(json.dumps(module._qualification_public3(args), indent=2, allow_nan=False))\n")
    compile(runner, "exp174-b-public3", "exec")
    write_text(out / "run-public3-b-only.py", runner)
    python = args.python
    command = "& " + _shared.ps_quote(python) + " " + _shared.ps_quote(out / "run-public3-b-only.py")
    write_text(out / "PLAN.ps1", "# PLAN ONLY: B must claim the local GPU after training ends. No uploads.\n" + command + "\n")
    return dict(status="PLAN_ONLY_NOT_EXECUTED", out=str(out), command=command,
                runner_sha256=sha(out / "run-public3-b-only.py"),
                note="Future B runner initializes CUDA. Plan generation does not run it. HOLD until CPU average and local/T4 parity PASS.")


def push_plan(args):
    stage_dir, report, _ = checked_stage(args.stage)
    path = Path(args.out)
    venv = ROOT / ".venv/Scripts/python.exe"
    quote = _shared.ps_quote
    parity_command = ("& " + quote(args.python) + " " + quote(TOOL) + " compare-parity"
                      + " --stage " + quote(stage_dir)
                      + " --plan " + quote(args.parity_plan)
                      + " --local " + quote(args.local_public3)
                      + " --own " + quote(stage_dir / "own-t4-output")
                      + " --out " + quote(args.parity_out))
    commands = ["# PLAN ONLY: private assets + own-kernel qualification; no submission commands.",
                "# B checks verified CPU average, local Public3 PASS, quota and reserved lane before execution.",
                "Push-Location -LiteralPath " + quote(stage_dir / "dataset"),
                "try {", "  & " + quote(venv) + " -m kaggle datasets create -p .",
                "  if ($LASTEXITCODE -ne 0) { throw 'Dataset create failed; inspect CLI output' }",
                "} finally { Pop-Location }",
                "# Private is the create default; verify private dataset readiness on the site before push.",
                "# Direct CLI preserves frozen metadata; own notebook has no public dataset attachment.",
                "& " + quote(venv) + " -m kaggle kernels push -p " + quote(stage_dir / "own") + " --accelerator NvidiaTeslaT4",
                "# Wait for verified COMPLETE, then download full qualification outputs and receipts.",
                "& " + quote(venv) + " -m kaggle kernels output taeyangg4/" + KERNELS["own"] + " -p " + quote(stage_dir / "own-t4-output"),
                "# Exact future own-only parity comparison:", parity_command,
                "# HOLD unless verified CPU average receipt and final Public3 parity PASS.",
                "# Future submission needs Taeyang's chat instruction; blends need separate registration."]
    write_text(path, "\n".join(commands) + "\n")
    return dict(status="PLAN_ONLY_NOT_EXECUTED", out=str(path), sha256=sha(path), dataset=report["dataset"])


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", action="version", version=VERSION)
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("average")
    a.add_argument("--protocol", default=str(PROTOCOL)); a.add_argument("--checkpoint")
    a.add_argument("--out", default=str(BASE / "average-v1"))
    s = sub.add_parser("stage")
    s.add_argument("--average", default=str(BASE / "average-v1")); s.add_argument("--out", required=True)
    s.add_argument("--dataset", default=ASSET_DATASET)
    p = sub.add_parser("parity-plan")
    p.add_argument("--stage", required=True); p.add_argument("--out", required=True)
    p.add_argument("--test-csv", default=str(ROOT / "storage/competition/test.csv"))
    p.add_argument("--series-csv", default=str(ROOT / "storage/competition/test_series.csv"))
    p = sub.add_parser("public3-plan")
    p.add_argument("--stage", required=True); p.add_argument("--plan", required=True)
    p.add_argument("--protocol", default=str(PROTOCOL)); p.add_argument("--public-checkpoint", required=True)
    p.add_argument("--competition", default=str(ROOT / "storage/competition")); p.add_argument("--dicom-root")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--out", required=True)
    p.add_argument("--result-out", required=True)
    p.add_argument("--python", default=r"C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe")
    p = sub.add_parser("compare-parity")
    for name in ("stage", "plan", "local", "own", "out"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("push-plan")
    p.add_argument("--stage", required=True); p.add_argument("--out", required=True)
    p.add_argument("--parity-plan", required=True); p.add_argument("--local-public3", required=True)
    p.add_argument("--parity-out", required=True)
    p.add_argument("--python", default=r"C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = globals()[args.command.replace("-", "_")](args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
