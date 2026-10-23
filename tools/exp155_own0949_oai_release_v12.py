"""EXP155 v1.2.0: local-only release adapter for Amendment B1 weights.

Shares the SHA-pinned EXP152 decoder, model, parity logic and notebook runner.
Only the explicit public3 command may initialize CUDA. No command uploads,
downloads, submits, accesses credentials, or packages external study data.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
import zipfile

VERSION = "1.2.0"
TOOL = Path(__file__).resolve()
ROOT = TOOL.parents[1]
PROTOCOL = ROOT / "artifacts/experiments/EXP155_OWN0949_OAI/protocol-v1_1/manifest.json"
RUN = "artifacts/experiments/EXP155_OWN0949_OAI/S42v1_1"
OWN_NAME = "exp155-v1_1-e14-e11-e12-oai-bn.pt"
MEMBER_NAME = "exp143-v2-e14-e11-e12-bn.pt"
ASSET_DATASET = "taeyangg4/knee-exp155-own0949-oai-v1-2-assets"
KERNELS = {"own": "knee-exp155-own0949-oai-v1-2",
           "blend": "knee-exp155-own0949-oai-blend-v1-2",
           "blend3": "knee-exp155-own0949-oai-blend3-v1-2"}
BASE_NAME = "exp152_own0949_release.py"
BASE_SHA256 = "ebb8e06734b72f40f090389b23d8767137c8eb6c92dee547367c1f18188f94e2"
BASE_PATH = TOOL.with_name(BASE_NAME)


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# A private module instance makes adaptation local to EXP155. Importing this
# adapter never mutates an independently imported EXP152 module or initializes
# torch/CUDA. The unchanged source is included and authenticated in the archive.
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
need, sha, read_json, write_json = _shared.need, _shared.sha, _shared.read_json, _shared.write_json
write_text, fresh_dir, safe_load = _shared.write_text, _shared.fresh_dir, _shared.safe_load
state_schema, average_states, merge_replayed_bn = _shared.state_schema, _shared.average_states, _shared.merge_replayed_bn
serving_source, public_source = _shared.serving_source, _shared.public_source
_original_load_module = _shared.load_module
_original_notebook = _shared.notebook
_original_verify_assets = _shared.verify_assets
_original_projection = _shared.projection
_original_output_values = _shared.output_values
_original_bind = _shared.bind
_original_checked_model = _shared.checked_model
_original_save_raw = _shared.save_raw


def load_module(name, path):
    """Redirect only the frozen v2 guard to the registered EXP155 guard."""
    path = Path(path)
    if path.resolve() == (ROOT / "tools/exp143_v2.py").resolve():
        path = ROOT / "tools/exp155_own0949_oai_v11.py"
    return _original_load_module(name, path)


def own_meta(ck, protocol_sha, run_dir=RUN):
    need(ck.get("oai") is True, "Own checkpoint metadata mismatch: oai")
    expected = dict(arch=ARCH, res=384, inference_res=320, lab=TARGETS, contract=CONTRACT,
                    k_eval=94, norm="imagenet", window_policy="legacy", oai=True,
                    checkpoint_kind="fixed-swa-bn", swa_epochs_1based=EPOCHS, seed=42)
    for key, value in expected.items():
        need(key in ck and ck[key] == value, "Own checkpoint metadata mismatch: " + key)
    need(ck.get("provenance", {}).get("protocol_sha256") == protocol_sha,
         "Own checkpoint protocol SHA mismatch")
    need(ck["provenance"].get("run_dir") == run_dir, "Own checkpoint run mismatch")
    need(isinstance(ck.get("model"), dict) and ck["model"], "Missing model state")


def average(args):
    """Reconstruct fixed SWA, preserving authenticated mixed-cohort BN buffers."""
    import torch
    guard = load_module("exp155_release_guard", ROOT / "tools/exp155_own0949_oai_v11.py")
    protocol = guard.load_registration(args.protocol)
    need(protocol["recipe"]["swa_1based"] == EPOCHS and protocol["run_dirs"]["local"] == RUN,
         "Only EXP155 S42v1_1 fixed epochs [14,11,12] is supported")
    run = guard.registered_run(protocol, "local")
    finished = run / "model_best.pt"
    if args.checkpoint:
        need(Path(args.checkpoint).resolve() == finished.resolve(), "Checkpoint must be EXP155 S42v1_1/model_best.pt")
    summary = read_json(run / "summary.json")
    need(summary.get("status") == "COMPLETE_TRAINING_NO_EVALUATION" and summary.get("epochs") == 16,
         "Training / BN replay not complete")
    need(summary.get("model_sha256") == sha(finished), "Final checkpoint SHA mismatch")
    checkpoint = safe_load(finished)
    own_meta(checkpoint, sha(args.protocol))
    need(summary.get("identity") == checkpoint["provenance"], "Summary identity differs")
    # EXP155 train_ids is the complete registered RSNA + OAI cohort, never Gold58.
    train_ids = protocol["train_ids"]
    need(len(train_ids) == len(set(train_ids)), "Duplicate mixed training IDs")
    n, bn = len(train_ids), summary["bn"]
    batches = math.ceil(n / protocol["recipe"]["bn_batch"])
    need(bn.get("studies") == bn.get("unique_studies") == n and bn.get("batches") == batches
         and bn.get("num_batches_tracked") == [batches] and bn.get("mode") == "train"
         and bn.get("gradients") == "none", "Incomplete mixed train-only BN receipt")
    expected_ids = hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest()
    need(bn.get("studies_sha256") == expected_ids, "BN replay cohort differs")
    states, members = [], {}
    for epoch in EPOCHS:
        path = run / f"epoch_{epoch:02d}.pt"
        members[str(epoch)] = sha(path)
        need(summary.get("swa_member_sha256", {}).get(str(epoch)) == members[str(epoch)], "Epoch SHA mismatch")
        item = safe_load(path)
        need(item.get("epoch") == epoch and set(item) == {"model", "epoch"}, "Epoch payload mismatch")
        states.append(item["model"])
    averaged = average_states(states)
    del states
    train = load_module("exp155_release_training_base", ROOT / "training/src/exp143_own0949.py")
    model = train.build_model(320)  # CPU architecture construction only.
    model.load_state_dict(checkpoint["model"], strict=True)
    from torch.nn.modules.batchnorm import _BatchNorm
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, _BatchNorm)]
    bn_keys = {name + "." + key for name, module in modules for key in
               ("running_mean", "running_var", "num_batches_tracked") if getattr(module, key) is not None}
    need(len(modules) == bn.get("bn_modules"), "BN module count mismatch")
    reconstructed = merge_replayed_bn(averaged, checkpoint["model"], bn_keys, batches)
    model.load_state_dict(reconstructed, strict=True)
    del model
    out = fresh_dir(args.out)
    with (out / OWN_NAME).open("xb") as stream:
        torch.save(dict(checkpoint, model=reconstructed), stream)
    receipt = dict(schema="exp155-average-v1.1", version=VERSION, status="VERIFIED_FIXED_SWA_BN",
                   checkpoint=OWN_NAME, checkpoint_sha256=sha(out / OWN_NAME),
                   source_final_sha256=sha(finished), summary_sha256=sha(run / "summary.json"),
                   protocol_sha256=sha(args.protocol), run_dir=RUN, epochs_1based=EPOCHS,
                   epoch_sha256=members, state_schema=state_schema(reconstructed),
                   bn_buffer_keys=sorted(bn_keys), bn=bn, oai=True,
                   bn_handling="mixed RSNA/OAI train-only authenticated trainer replay; exact final-state equality",
                   source_bindings_sha256=_shared.jhash(protocol["source_bindings"]),
                   shared_serving_sha256=BASE_SHA256)
    write_json(out / "average-receipt.json", receipt)
    return dict(status=receipt["status"], out=str(out), checkpoint_sha256=receipt["checkpoint_sha256"])


def notebook(variant, dataset, archive_sha, manifest_sha, module_sha, original_markdown):
    # Reuse the exact EXP152 executable setup and public serving pipeline. Only
    # packaging paths/import and prose change; _E152 names remain its runtime ABI.
    need(variant in KERNELS, "Unknown variant")
    nb, _, _ = _original_notebook("blend" if variant == "blend3" else variant,
                                  dataset, archive_sha, manifest_sha, module_sha, original_markdown)
    code = nb["cells"][-1]["source"]
    code = code.replace("exp152-assets", "exp155-assets")
    code = _shared.replace_once(code, "'exp152_own0949_release.py', 'public-serving.py'",
                               repr(MEMBER_NAME) + ", 'exp155_own0949_oai_release_v12.py', 'exp152_own0949_release.py', 'public-serving.py'")
    code = _shared.replace_once(code, "_hash(_assets/'exp152_own0949_release.py')", "_hash(_assets/'exp155_own0949_oai_release_v12.py')")
    code = _shared.replace_once(code, "import exp152_own0949_release as _e152", "import exp155_own0949_oai_release_v12 as _e152")
    if variant == "blend3":
        code = _shared.replace_once(code, "_work = Path(",
                                    "_files[" + repr(MEMBER_NAME) + "] = _assets/" + repr(MEMBER_NAME) + "\n_work = Path(")
        code = _shared.replace_once(code, "_e152.bind(globals(), _manifest, _files, 'blend', _work)",
                                    "_e152.bind(globals(), _manifest, _files, 'blend3', _work)")
    code = code.replace("private EXP152", "private EXP155")
    compile(code, "exp155-notebook", "exec")
    nb["cells"][-1]["source"] = code
    title = "EXP155 own0949 OAI v1.2" + (" fixed 50/50 public blend" if variant == "blend" else
                                        " fixed equal three-member rank blend" if variant == "blend3" else " own checkpoint")
    description = ("OAI-derived weights: EXP143-v2 recipe plus masked PF OA, Lateral OA and Synovitis supervision. "
                   "Fixed epochs 14/11/12 and registered mixed train-only BN; Gold58 excluded from training. "
                   "Unchanged EXP152 serving: K94, native320 crop, anatomical mirror, fp16 chunk16. "
                   "Private, internet off. Only model weights, serving code and nonidentifying lineage hashes; "
                   "no raw OAI data, participant IDs or labels. Every submission requires Taeyang's instruction. ")
    if variant == "blend":
        description += "Fixed own/public=0.5/0.5 average-tie percentile ranks, no final re-ranking or LB tuning. "
    if variant == "blend3":
        description += "Public 0.949, EXP143 v2 and EXP155 each have weight 1/3 in average-tie percentile ranks; no final re-ranking. "
    description += "Original author notices retained below; executable source is SHA-pinned."
    nb["cells"][0]["source"] = "# " + title + "\n\n" + description
    return nb, title, description


def verify_assets(path, manifest):
    _original_verify_assets(path, manifest)
    need(manifest.get("schema") == "exp155-release-v1.2", "EXP155 release schema mismatch")
    need(manifest["checkpoints"][OWN_NAME].get("oai") is True, "EXP155 OAI provenance missing")
    expected = {OWN_NAME, MEMBER_NAME, TOOL.name, BASE_NAME, "public-serving.py"}
    need(set(manifest["asset_sha256"]) == expected, "Unexpected release payload")
    need(manifest["asset_sha256"][BASE_NAME] == BASE_SHA256, "Shared serving source mismatch")
    need(manifest["checkpoints"][MEMBER_NAME].get("oai") is False,
         "EXP143 member must be OAI-free")


def stage(args):
    receipt = read_json(Path(args.average) / "average-receipt.json")
    need(receipt.get("schema") == "exp155-average-v1.1" and receipt.get("oai") is True and
         receipt.get("status") == "VERIFIED_FIXED_SWA_BN" and receipt.get("epochs_1based") == EPOCHS,
         "EXP155 average receipt invalid")
    need(receipt.get("shared_serving_sha256") == BASE_SHA256, "Average serving binding mismatch")
    checkpoint = Path(args.average) / OWN_NAME
    need(sha(checkpoint) == receipt["checkpoint_sha256"], "Average checkpoint SHA mismatch")
    ck = safe_load(checkpoint)
    own_meta(ck, receipt["protocol_sha256"])
    need(state_schema(ck["model"]) == receipt["state_schema"], "Average schema mismatch")
    # Training identity must not become an accidental OAI participant export.
    assert_private_checkpoint(ck)
    member = Path(args.exp143_checkpoint)
    need(member.name == MEMBER_NAME, "Expected authenticated EXP143 v2 average")
    member_receipt = read_json(member.parent / "average-receipt.json")
    member_protocol = ROOT / "artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json"
    member_protocol_sha = sha(member_protocol)
    need(member_receipt.get("status") == "VERIFIED_FIXED_SWA_BN" and
         member_receipt.get("checkpoint_sha256") == sha(member) and
         member_receipt.get("protocol_sha256") == member_protocol_sha and
         member_receipt.get("epochs_1based") == EPOCHS,
         "EXP143 average receipt mismatch")
    mck = safe_load(member)
    need(mck.get("oai") is False and mck.get("checkpoint_kind") == "fixed-swa-bn" and
         mck.get("swa_epochs_1based") == EPOCHS and
         mck.get("provenance", {}).get("protocol_sha256") == member_protocol_sha and
         mck.get("arch") == ARCH and mck.get("res") == 384 and
         state_schema(mck["model"]) == receipt["state_schema"],
         "EXP143 member metadata/schema mismatch")
    need(re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+", args.dataset), "Invalid dataset ID")
    source, credits = public_source()
    out = fresh_dir(args.out)
    assets = fresh_dir(out / "assets")
    shutil.copyfile(checkpoint, assets / OWN_NAME)
    # Package inference fields only; no EXP143 training identifiers leave this machine.
    permitted = {"model", "arch", "res", "inference_res", "lab", "contract", "k_eval", "norm",
                 "window_policy", "oai", "checkpoint_kind", "swa_epochs_1based", "seed"}
    import torch
    with (assets / MEMBER_NAME).open("xb") as stream:
        torch.save({**{k: v for k, v in mck.items() if k in permitted},
                    "provenance": {"protocol_sha256": member_protocol_sha}}, stream)
    shutil.copyfile(TOOL, assets / TOOL.name)
    shutil.copyfile(BASE_PATH, assets / BASE_NAME)
    write_text(assets / "public-serving.py", serving_source(source))
    manifest = dict(schema="exp155-release-v1.2", version=VERSION,
                    protocol_sha256=receipt["protocol_sha256"], epochs_1based=EPOCHS,
                    average_receipt_sha256=sha(Path(args.average) / "average-receipt.json"),
                    source_final_sha256=receipt["source_final_sha256"],
                    public_source_sha256=SOURCE_SHA, serving_sha256=sha(assets / "public-serving.py"),
                    checkpoints={OWN_NAME: dict(sha256=receipt["checkpoint_sha256"], oai=True),
                                 PUBLIC_NAME: dict(sha256=PUBLIC_SHA, oai=True),
                                 MEMBER_NAME: dict(sha256=sha(assets / MEMBER_NAME), oai=False)},
                    own_state_schema=receipt["state_schema"],
                    asset_sha256={n: sha(assets / n) for n in (OWN_NAME, MEMBER_NAME, TOOL.name, BASE_NAME, "public-serving.py")},
                    ranks=dict(own="public numpy double argsort / (N-1)",
                               blend="average ties (rank-1)/(N-1), .5 own + .5 public; no final re-rank",
                               blend3="EXP155, public 0.949, EXP143 v2: average-tie percentile ranks, 1/3 each; no final re-rank"),
                    policy=POLICY, tolerance=TOL, dataset=args.dataset,
                    member_protocol_sha256=member_protocol_sha,
                    source_member_sha256=sha(member),
                    source_member_receipt_sha256=sha(member.parent / "average-receipt.json"),
                    source_bindings_sha256=receipt["source_bindings_sha256"],
                    approval_required={v: "OAI-derived; Taeyang instruction for each submission" for v in KERNELS})
    write_json(assets / "release-manifest.json", manifest)
    verify_assets(assets, manifest)
    dataset = fresh_dir(out / "dataset")
    archive = dataset / "exp155-assets.zip.bin"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(assets.iterdir()):
            bundle.write(path, path.name)
    write_json(dataset / "dataset-metadata.json", dict(title=args.dataset.split("/")[1], id=args.dataset,
               licenses=[{"name": "other"}], description="Private EXP155 OAI-derived weights and EXP152 serving code. Fixed epochs14/11/12; no external study data, participant IDs or labels."))
    kernels = {}
    for variant in KERNELS:
        dest = fresh_dir(out / variant)
        nb, title, description = notebook(variant, args.dataset, sha(archive), sha(assets / "release-manifest.json"),
                                          sha(assets / TOOL.name), credits)
        write_json(dest / "release.ipynb", nb)
        write_json(dest / "kernel-metadata.json", kernel_metadata(variant, args.dataset))
        write_text(dest / "DESCRIPTION.md", "# " + title + "\n\n" + description + "\n")
        kernels[variant] = {n: sha(dest / n) for n in ("release.ipynb", "kernel-metadata.json", "DESCRIPTION.md")}
    report = dict(status="STAGED_NOT_UPLOADED", version=VERSION, private_required=True,
                  archive_sha256=sha(archive), manifest_sha256=sha(assets / "release-manifest.json"),
                  dataset_metadata_sha256=sha(dataset / "dataset-metadata.json"),
                  kernels=kernels, projection=projection(), dataset=args.dataset)
    write_json(out / "build-receipt.json", report)
    return dict(status=report["status"], out=str(out), archive_sha256=report["archive_sha256"])


def assert_private_checkpoint(checkpoint):
    """Fail closed on identity metadata beyond the frozen nonidentifying schema.

    Do not silently strip or rewrite the trainer checkpoint. New trainer identity
    fields require deliberate review and an explicit allowlist addition.
    """
    permitted = {"protocol_sha256", "run_dir", "torch", "seed", "precision",
                 "initial_state_sha256", "world_size", "contract", "train_n",
                 "rsna_train_n", "oai_train_n", "execution"}
    identity = checkpoint["provenance"]
    need(set(identity) <= permitted, "Unreviewed checkpoint provenance fields: " + repr(sorted(set(identity) - permitted)))
    need(all(isinstance(value, (str, int, float, bool, type(None)))
             for key, value in identity.items() if key != "execution"),
         "Nested checkpoint provenance could contain study identifiers")
    for key in ("protocol_sha256", "initial_state_sha256"):
        if key in identity:
            need(isinstance(identity[key], str) and re.fullmatch(r"[0-9a-f]{64}", identity[key]),
                 "Invalid nonidentifying checkpoint hash: " + key)
    need(all(type(identity[key]) is int and identity[key] >= 0
             for key in ("train_n", "rsna_train_n", "oai_train_n") if key in identity),
         "Checkpoint population counts must be nonidentifying integers")
    execution = identity.get("execution", {})
    need(isinstance(execution, dict) and set(execution) <= {"lane", "batch", "accumulation",
         "effective_batch", "grad_checkpointing", "microbatches_per_epoch", "updates_per_epoch",
         "total_steps", "max_hours", "reserve_seconds"}, "Unreviewed checkpoint execution fields")
    need(all(isinstance(value, (int, float, bool)) for key, value in execution.items() if key != "lane")
         and execution.get("lane", "local") == "local", "Unreviewed checkpoint execution values")
    # Model tensors and these inference fields are the entire allowed payload.
    need(set(checkpoint) <= {"model", "arch", "res", "inference_res", "lab", "contract", "k_eval",
         "norm", "window_policy", "oai", "checkpoint_kind", "swa_epochs_1based", "seed", "provenance",
         "epoch", "tag", "amp_train", "amp_eval", "fold", "label_set"},
         "Unreviewed checkpoint payload fields")
    need(all(isinstance(checkpoint[key], (str, int, float, bool))
             for key in ("epoch", "tag", "amp_train", "amp_eval", "fold", "label_set") if key in checkpoint),
         "Unexpected structured checkpoint metadata")
    for key, value in {"epoch": 15, "tag": "S42v1_1", "amp_train": "bf16", "amp_eval": "fp16",
                       "fold": -1, "label_set": "l2-dread75-aquila25-plus-oai-three-target-b1"}.items():
        need(key not in checkpoint or checkpoint[key] == value, "Unregistered checkpoint metadata: " + key)


def checked_stage(stage_dir):
    path = Path(stage_dir).resolve()
    report = read_json(path / "build-receipt.json")
    need(report.get("version") == VERSION, "Stage version mismatch")
    for file, key in (("assets/release-manifest.json", "manifest_sha256"),
                      ("dataset/exp155-assets.zip.bin", "archive_sha256"),
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
    result["caveat"] = result["caveat"].replace("EXP152", "EXP155")
    result["checkpoint_effect"] = "Identical EXP152 graph/decode; OAI changes weights only. Timing remains a projection."
    return result


def output_values(arms, variant, public_rank):
    if variant != "blend3":
        return _original_output_values(arms, variant, public_rank)
    import numpy as np
    need(len(arms) == 3, "Three rank arms required")
    shape = np.asarray(arms[0]).shape
    need(len(shape) == 2 and shape[0] > 0 and shape[1] == 12 and
         all(np.asarray(a).shape == shape and np.isfinite(a).all() and
             (np.asarray(a) >= 0).all() and (np.asarray(a) <= 1).all() for a in arms),
         "Invalid three-arm probabilities")
    ranks = [np.asarray([percentile_average(np.asarray(a)[:, j].tolist())
                         for j in range(12)]).T for a in arms]
    return sum(ranks) / 3.0


def bind(ns, manifest, files, variant, work):
    _original_bind(ns, manifest, files, variant, work)
    if variant == "blend3":
        need(set(files) == {OWN_NAME, PUBLIC_NAME, MEMBER_NAME}, "Incomplete three-arm paths")
        ns["ARMS"] = [dict(file=n, arch=ARCH, res=384, w=1.0 / 3.0)
                      for n in (OWN_NAME, PUBLIC_NAME, MEMBER_NAME)]


def checked_model(ns, path, arch, res, device):
    if Path(path).name == MEMBER_NAME:
        ck = safe_load(path)
        need(ck.get("oai") is False and ck.get("checkpoint_kind") == "fixed-swa-bn" and
             ck.get("swa_epochs_1based") == EPOCHS and
             ck.get("provenance", {}).get("protocol_sha256") ==
             ns["_E152_MANIFEST"]["member_protocol_sha256"],
             "EXP143 member metadata differs")
    return _original_checked_model(ns, path, arch, res, device)


def save_raw(work, ids, arms, inputs, manifest, variant):
    if variant != "blend3":
        return _original_save_raw(work, ids, arms, inputs, manifest, variant)
    import numpy as np
    need(len(ids) == len(set(ids)) and set(ids) == set(inputs), "Incomplete input diagnostics")
    output_values(arms, variant, lambda a: a)
    with (Path(work) / "raw.npz").open("xb") as stream:
        np.savez(stream, ids=np.asarray(ids), targets=np.asarray(TARGETS),
                 own=arms[0], public=arms[1], exp143=arms[2])
    write_json(Path(work) / "inputs.json", inputs)
    write_json(Path(work) / "runtime.json", _shared.runtime_info())
    write_json(Path(work) / "binding.json", dict(manifest_sha256=_shared.jhash(manifest),
               variant=variant, checkpoint_sha256={n: manifest["checkpoints"][n]["sha256"]
               for n in (OWN_NAME, PUBLIC_NAME, MEMBER_NAME)}, source_sha256=SOURCE_SHA,
               serving_sha256=manifest["serving_sha256"], tolerance=TOL))


def push_plan(args):
    """Reuse local plan writer, then replace only EXP152 names and policy prose."""
    # Base function uses adapted checked_stage/KERNELS and never calls an account API.
    result = _shared.push_plan(args)
    path = Path(args.out)
    text = path.read_text(encoding="utf-8")
    text = text.replace("H-B-63-exp152-", "EXP155-v1-2-").replace("EXP152 ", "EXP155 ")
    text = text.replace(" fixed A2; user approved", " fixed OAI-v1.2; user approved")
    text = text.replace("OAI warning for blend", "OAI lineage for all variants")
    text = text.replace("blend additionally has OAI-derived lineage", "all variants have OAI-derived lineage")
    text = text.replace("Run compare-parity against both downloaded output folders.",
                        "For blend3, review three-arm raw.npz parity (own/public/exp143) separately; two-arm compare-parity is unsupported.")
    path.write_text(text, encoding="utf-8", newline="\n")
    result["sha256"] = sha(path)
    return result


# Bind local-only adapters on the isolated shared module. Parity, input hashing,
# raw probability handling, checked model construction and decoder stay identical.
for _name in ("load_module", "own_meta", "notebook", "verify_assets", "checked_stage", "projection",
              "output_values", "bind", "checked_model", "save_raw"):
    setattr(_shared, _name, globals()[_name])
parity_plan, public3, compare_parity = _shared.parity_plan, _shared.public3, _shared.compare_parity
percentile_average = _shared.percentile_average


def kernel_metadata(variant, dataset=ASSET_DATASET):
    result = _shared.kernel_metadata("blend" if variant == "blend3" else variant, dataset)
    if variant == "blend3":
        result["id"] = "taeyangg4/" + KERNELS[variant]
        result["title"] = KERNELS[variant]
    return result


def parser():
    result = _shared.parser()
    result.description = __doc__
    for action in result._subparsers._group_actions:
        if "stage" in action.choices:
            action.choices["stage"].add_argument("--exp143-checkpoint", required=True)
    return result


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
