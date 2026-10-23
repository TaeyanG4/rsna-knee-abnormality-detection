"""EXP190: corrected OAI supervision on the EXP181 fold-0 CoAtNet bridge.



CPU commands register/validate/queue-json/collect/screen never launch CUDA. The

two GPU commands require gpu_job. This is a single, fresh OAI attempt; CONTROL

is authenticated historical EXP181 evidence. v1.4.0 rebinds changed legacy
timing dependencies in a new EXP190G namespace. v1.3.0 removed bridge-global

mutation and binds the original CONTROL source archive; EXP190G is a fresh

engineering registration. Collected screening uses the holistic multi-lens readout.

"""

from __future__ import annotations



import argparse

import copy

import hashlib

import json

import math

import os

from pathlib import Path

import re

import sys

import time

import types

from functools import lru_cache



ROOT = Path(__file__).resolve().parents[1]

for _path in (ROOT, ROOT / "training/src"):

    if str(_path) not in sys.path:

        sys.path.insert(0, str(_path))

from tools import exp181_coatnet_f0_bridge as bridge

from tools import exp155_own0949_oai_v11 as old_oai

from tools import exp155_oai_labels_v2 as mapping

from training.arms import oai_exclusions



# v1.4.0: fresh EXP190G binds untimed legacy dependencies; F remains immutable.
# v1.3.0: explicit historical verification and multi-lens collection, EXP190F.

# v1.2.0: panel-excluded training/BN, immutable EXP190E registration.

# v1.1.0: direct EXP155 API, scientific identities, untimed training/readout.

VERSION = "1.4.0"

ARM = "OAI"

EXP = ROOT / "artifacts/experiments/EXP190G_COATNET_OAI_F0"

PROTOCOL = EXP / "PROTOCOL.md"

REG = EXP / "protocol-v1"

MANIFEST = REG / "manifest-OAI.json"

VALIDATION = REG / "validation.json"

QUEUE = REG / "queue-exp190.json"

LABELS = REG / "oai-masked.csv"

LABEL_PROVENANCE = REG / "oai-masked.provenance.json"

LABEL_QA = REG / "oai-masked.qa.json"

COLLECTION = EXP / "collection.json"

SCREEN = EXP / "screen/result-OAI-vs-CONTROL.json"

GOLD_CTRL = EXP / "gold58/control.csv"

GOLD_OAI = EXP / "gold58/oai.csv"

GOLD_RECEIPT = EXP / "gold58/prediction-receipt.json"

GOLD_REPORT = EXP / "gold58/diagnostic.json"

GOLD_QUEUE = REG / "queue-gold58.json"

ATTEMPT = EXP / "attempt.json"

CACHE = ROOT / "storage/cache/derived/EXP155_OAI_RAPTOR96_384_C140_S02_v2"

CACHE_MANIFEST_SHA = "da3509c79039e101573dc58cea7aa9e5e5f762b66478d36fd1b52df1cdbeed21"

CACHE_BUILD_SHA = "b95022ef07f7f637ed3319b42e15a2679986016bfd25f6aa818bc366bfedecec"

OAI_TARGETS = ("PF OA", "Lateral OA", "Synovitis")

CONTROL_GRAD_CHECKPOINTING = False  # EXP181 CONTROL summary/provenance execution

WEAK = ("Lateral Meniscus", *OAI_TARGETS)

RUN_REL = "artifacts/experiments/EXP190G_COATNET_OAI_F0/OAI"

TRAIN_TAG, OOF_TAG, GOLD_TAG = "EXP190G_OAI_F0", "EXP190G_OAI_F0_OOF", "EXP190G_GOLD58_F0"

SOURCE = ("tools/exp190_coatnet_oai_f0.py", "tools/exp155_own0949_oai_v11.py",

          "tools/exp155_oai_labels_v2.py", "tools/extcache/oai_cache_v2.py",

          "training/src/exp155_own0949_oai_v11.py",

          "training/arms/oai_smoke.py", "training/arms/oai_exclusions.py", "training/arms/coatnet_fit.py",

          "training/arms/coatnet_history.py", "training/arms/coatnet_screen.py",

          "training/arms/collected_screen.py", "tools/multi_lens_readout.py",

          ".local/backups/refactor-step8-history-0f8ad6311feb4bbd86ffc0193b0d5042/index.json",

          "training/arms/coatnet.py", "training/arms/coatnet_lifecycle.py", "training/arms/lifecycle.py",

          "artifacts/experiments/EXP190G_COATNET_OAI_F0/PROTOCOL.md")

HASH = bridge.sha256

EXP155_TRAINER = ROOT / "training/src/exp155_own0949_oai_v11.py"

EXP155_TRAINER_SHA256 = "f53e16eef250cc7d4e5b504e8a9da6c073bcc7e5cb3e823c677ce80a4b49954e"





def write_new(path, value):

    path = Path(path).resolve()

    if EXP.resolve() not in path.parents:

        raise ValueError("EXP190 output must stay below its experiment directory")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8", newline="\n") as stream:

        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)

        stream.write("\n")





def cache_binding(cache=CACHE, full_hash=False):

    """Reject the old cache, stale metadata/code, missing receipts and shard drift."""

    cache = Path(cache).resolve()

    if cache != CACHE.resolve():

        raise ValueError("EXP190 requires the corrected production v2 cache path")

    if HASH(cache / "manifest.json") != CACHE_MANIFEST_SHA or HASH(cache / "build-registration.json") != CACHE_BUILD_SHA:

        raise ValueError("corrected OAI v2 cache manifest/build hash differs")

    man = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))

    reg = json.loads((cache / "build-registration.json").read_text(encoding="utf-8"))

    if (man.get("schema") != "hb89-oai-exp143-96x384-v2"

            or man.get("contract") != "exp143_raptor96_384_c140_s02"

            or man.get("build_registration_sha256") != CACHE_BUILD_SHA

            or man.get("tool_version") != "oai-cache-v2-1.0.1"

            or man.get("audit_summary_sha256") != reg.get("audit", {}).get("summary_sha256")

            or man.get("tool_sha256") != HASH(ROOT / "tools/extcache/oai_cache_v2.py")

            or reg.get("schema") != man["schema"] or reg.get("contract") != man["contract"]

            or reg.get("ids") != man.get("ids") or reg.get("code_sha256") is None

            or reg.get("sag0_fallback_mode") != "competition_then_copy"

            or man.get("sag0_fallback_mode") != reg["sag0_fallback_mode"]):

        raise ValueError("corrected OAI v2 audit/renderer contract differs")

    for rel, expected in reg["code_sha256"].items():

        if HASH(ROOT / rel) != expected:

            raise ValueError("OAI production renderer/source hash differs: " + rel)

    ids = man["ids"]

    if len(ids) != len(set(ids)) or len(ids) != 2387 or len(man.get("files", [])) != len(ids):

        raise ValueError("OAI production roster/receipt count differs")

    for i, row in enumerate(man["files"]):

        if row.get("row") != i or row.get("uid") != ids[i]:

            raise ValueError("OAI production file order differs")

        volume = cache / f"study_{i:05d}.npy"

        receipt = cache / f"study_{i:05d}.json"

        if not volume.is_file() or not receipt.is_file():

            raise FileNotFoundError("OAI production volume/receipt missing")

        if full_hash and (HASH(volume) != row["sha256"] or HASH(receipt) != row["receipt_sha256"]):

            raise ValueError("OAI production volume/receipt hash differs")

    return man





def label_values(cache_ids):

    if any(not re.fullmatch(r"OAI_[0-9]+_[RL]_V00", uid) for uid in cache_ids):

        raise ValueError("corrected cache has a nonbaseline OAI identity")

    inputs = mapping.clinical_inputs(mapping.DEFAULT_RECEIPT)

    primary, poma = mapping.select_sources(inputs)

    rows = {}

    for uid in cache_ids:

        m = primary[uid].scores if uid in primary else None

        f = poma[uid].scores if uid in poma else None

        vals = {target: mapping.oa_v2(m, target) for target in OAI_TARGETS[:2]}

        vals["Synovitis"], _ = mapping.synovitis_v2(m, f)

        if any(x is not None for x in vals.values()):

            rows[uid] = vals

    if not rows:

        raise ValueError("no observed OAI labels")

    return rows, inputs





def build_label_artifacts(cache_ids):

    import csv

    if any(p.exists() for p in (LABELS, LABEL_PROVENANCE, LABEL_QA)):

        raise FileExistsError("EXP190 labels already exist; no overwrite")

    values, inputs = label_values(cache_ids)

    fields = ["study_key", *bridge.TARGETS, *("mask_" + t for t in bridge.TARGETS)]

    LABELS.parent.mkdir(parents=True, exist_ok=True)

    with LABELS.open("x", encoding="utf-8", newline="") as stream:

        writer = csv.DictWriter(stream, fieldnames=fields)

        writer.writeheader()

        for uid in cache_ids:

            if uid not in values:

                continue

            row = {"study_key": uid, **{t: "" for t in bridge.TARGETS},

                   **{"mask_" + t: 0 for t in bridge.TARGETS}}

            for target, value in values[uid].items():

                if value is not None:

                    row[target] = value

                    row["mask_" + target] = 1

            writer.writerow(row)

    counts = {t: {"positive": sum(r[t] == 1 for r in values.values()),

                  "negative": sum(r[t] == 0 for r in values.values())} for t in OAI_TARGETS}

    for t in OAI_TARGETS:

        counts[t]["observed"] = counts[t]["positive"] + counts[t]["negative"]

        counts[t]["unknown"] = len(cache_ids) - counts[t]["observed"]

    qa = dict(schema="exp190-oai-label-qa-v1", cache_n=len(cache_ids), admitted_n=len(values),

              excluded_all_unknown_n=len(cache_ids) - len(values), counts=counts,

              rsna_fold_train_n=bridge.N_TRAIN, other_targets_masked=True)

    write_new(LABEL_QA, qa)

    write_new(LABEL_PROVENANCE, dict(schema="exp190-oai-labels-v1", targets=list(OAI_TARGETS),

        mapping="EXP155 B1: MOAKS cartilage v2 plus target-level POMA Synovitis",

        cache_manifest_sha256=CACHE_MANIFEST_SHA, cache_build_sha256=CACHE_BUILD_SHA,

        clinical_receipt_sha256=HASH(mapping.DEFAULT_RECEIPT),

        clinical_inputs_sha256={name: digest for name, (_, digest) in inputs.items()},

        mapping_source_sha256=HASH(ROOT / "tools/exp155_oai_labels_v2.py"),

        csv_sha256=HASH(LABELS), qa_sha256=HASH(LABEL_QA), rows=len(values)))

    return qa





def verified_labels(cache_ids):

    from tools.exp155_oai_labels_v2 import clinical_inputs

    prov = json.loads(LABEL_PROVENANCE.read_text(encoding="utf-8"))

    qa = json.loads(LABEL_QA.read_text(encoding="utf-8"))

    if (prov.get("cache_manifest_sha256") != CACHE_MANIFEST_SHA

            or prov.get("cache_build_sha256") != CACHE_BUILD_SHA

            or prov.get("clinical_receipt_sha256") != HASH(mapping.DEFAULT_RECEIPT)

            or prov.get("mapping_source_sha256") != HASH(ROOT / "tools/exp155_oai_labels_v2.py")

            or prov.get("clinical_inputs_sha256") != {n: d for n, (_, d) in clinical_inputs(mapping.DEFAULT_RECEIPT).items()}

            or prov.get("csv_sha256") != HASH(LABELS) or prov.get("qa_sha256") != HASH(LABEL_QA)

            or qa.get("cache_n") != len(cache_ids)):

        raise ValueError("EXP190 OAI label provenance changed")

    labels, masks = old_oai.read_oai_labels(LABELS)

    expected = set(cache_ids)

    if not set(labels) <= expected or len(labels) != qa["admitted_n"] or not labels:

        raise ValueError("OAI label/cache membership differs")

    counts = {t: {"positive": 0, "negative": 0, "unknown": len(cache_ids), "observed": 0} for t in OAI_TARGETS}

    for uid in labels:

        for t in OAI_TARGETS:

            i = bridge.TARGETS.index(t)

            if masks[uid][i]:

                counts[t]["observed"] += 1

                counts[t]["unknown"] -= 1

                counts[t]["positive" if labels[uid][i] == 1 else "negative"] += 1

    if qa["counts"] != counts:

        raise ValueError("OAI aggregate QA differs from CSV")

    return labels, masks, qa





def source_bindings():

    return {name: HASH(ROOT / name) for name in dict.fromkeys([*bridge.CODE, *SOURCE])}





def fixed_draw_counts(rsna_n, oai_n):

    if rsna_n != bridge.N_TRAIN or oai_n <= 0:

        raise ValueError("invalid mixed cohort")

    oai = round(rsna_n * oai_n / (rsna_n + oai_n))

    if not 0 < oai < rsna_n:

        raise ValueError("invalid OAI draw allocation")

    return dict(per_epoch=rsna_n, rsna=rsna_n - oai, oai=oai,

                updates_per_epoch=216, total_updates=bridge.UPDATES)





def control_recipe_with_oai(control):

    """EXP155's three-target loss with every CONTROL model/optimizer knob fixed."""

    recipe = copy.deepcopy(old_oai.RECIPE)

    ctrl = control["recipe"]

    scientific = {"arch", "res", "inference_res", "train_k", "eval_k", "epochs", "swa_1based",

        "batch", "effective_batch", "bn_batch", "bb_lr", "head_lr", "wd", "pct_start", "clip", "norm",

        "window_policy", "primary_blend_public_own", "sensitivity_blend_public_own", "seed_local",

        "seed_kaggle", "validation", "checkpoint_selection"}

    if any(key not in ctrl or recipe[key] != ctrl[key] for key in scientific):

        raise ValueError("EXP155 changed a CONTROL scientific recipe knob")

    # Metadata is outside the recipe identity; only supported scientific keys execute.

    supported = scientific | {"labels", "oai", "grad_checkpointing_default", "grad_checkpointing_allowed",

        "oai_targets", "oai_mix", "loss", "class_weights", "bn_cohort"}

    recipe = {key: recipe[key] for key in supported}

    if (ctrl.get("epochs") != bridge.EPOCHS or recipe.get("swa_1based") != ctrl.get("swa_1based")

            or recipe.get("eval_k") != ctrl.get("eval_k")

            or control.get("grad_checkpointing") is not CONTROL_GRAD_CHECKPOINTING

            or CONTROL_GRAD_CHECKPOINTING not in ctrl.get("grad_checkpointing_allowed", [])

            or recipe.get("grad_checkpointing_default") is not CONTROL_GRAD_CHECKPOINTING

            or recipe.get("grad_checkpointing_allowed") != [CONTROL_GRAD_CHECKPOINTING]):

        raise ValueError("CONTROL epochs/SWA/serving changed")

    # Fixed-step replacement is the sole departure from EXP155's old, longer

    # one-copy epoch. All inherited EXP181 model and optimizer knobs stay exact.

    recipe["oai_mix"] = "deterministic-fixed-step-replacement-per-epoch"

    return recipe





def register(args):

    exclusion = oai_exclusions.bind(getattr(args, "oai_exclusion_list", None))

    if not PROTOCOL.is_file() or MANIFEST.exists() or (ROOT / RUN_REL).exists():

        raise FileExistsError("EXP190 protocol missing or registration/run exists")

    control = bridge.load_registration(bridge.manifest_path("CONTROL"), args.corpus, args.init_weights)

    bridge.require_validation(bridge.manifest_path("CONTROL"))

    man = cache_binding(args.oai_cache)

    prospective, _ = label_values(man["ids"])

    if set(prospective) & (set(control["train_ids"]) | set(control["val_ids"]) |

                           set(control["excluded_gold_ids"])):

        raise ValueError("OAI/RSNA or Gold cohort overlap")

    recipe = control_recipe_with_oai(control)

    qa = build_label_artifacts(man["ids"])

    labels, masks, _ = verified_labels(man["ids"])

    allowed = oai_exclusions.filter_ids(man["ids"], exclusion)

    oai_ids = [uid for uid in allowed if uid in labels]

    rsna = control["train_ids"]

    all_ids, all_labels, all_masks, admitted = old_oai.combine_cohorts(

        rsna, control["train_labels"], control["excluded_gold_ids"], allowed, labels, masks)

    if admitted != oai_ids:

        raise ValueError("OAI admission order changed")

    p = copy.deepcopy(control)

    p.pop("content_sha256")

    p.pop("budgets", None)

    p.pop("oof_max_seconds", None)

    p.update(schema="exp190-coatnet-oai-f0-v1", version=VERSION, arm=ARM,

        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), run_dirs={"local": RUN_REL},

        dataset="EXP155 masked mixed dataset, deterministic fixed-step draw adapter",

        augmentation=None, recipe=recipe, train_ids=all_ids,

        rsna_train_ids=rsna, oai_train_ids=oai_ids, train_labels=all_labels,

        train_masks=all_masks, pos_weight=bridge.base.pos_weights(control["train_labels"], rsna),

        oai_corpus_manifest_sha256=CACHE_MANIFEST_SHA, oai_corpus_build_sha256=CACHE_BUILD_SHA,

        oai_label_sha256=HASH(LABELS), oai_label_provenance_sha256=HASH(LABEL_PROVENANCE),

        oai_label_qa_sha256=HASH(LABEL_QA), oai_label_qa=qa,

        oai_exclusion=exclusion,

        oai_membership=dict(cache_n=len(man["ids"]), labels_n=len(labels), admitted_n=len(oai_ids),

                            cache_without_observed_label=len(man["ids"]) - len(labels),

                            withheld_labelled_n=len(labels) - len(oai_ids),

                            labelled_without_cache=0),

        control_protocol_sha256=HASH(bridge.manifest_path("CONTROL")),

        sampling=fixed_draw_counts(len(rsna), len(oai_ids)),

        source_bindings=source_bindings(), screen_source_bindings=bridge.source_bindings(bridge.SCREEN_CODE))

    p["content_sha256"] = bridge.json_hash(p)

    write_new(MANIFEST, p)

    print(json.dumps(dict(status="REGISTERED_NO_TRAINING", manifest_sha256=HASH(MANIFEST),

                          rsna=len(rsna), oai=len(oai_ids), draw_counts=p["sampling"], qa=qa)))





def load_registration(protocol=MANIFEST, corpus=None, init_weights=None, oai_corpus=CACHE):

    if Path(protocol).resolve() != MANIFEST.resolve():

        raise ValueError("EXP190 manifest path differs")

    p = json.loads(MANIFEST.read_text(encoding="utf-8"))

    digest = p.pop("content_sha256", None)

    if bridge.json_hash(p) != digest or p.get("schema") != "exp190-coatnet-oai-f0-v1" or p.get("version") != VERSION:

        raise ValueError("EXP190 registration content changed")

    p["content_sha256"] = digest

    oai_exclusions.validate_training(p)

    ctrl = bridge.load_registration(bridge.manifest_path("CONTROL"), corpus, init_weights)

    scientific = {"seed", "precision", "grad_checkpointing", "world_size", "val_ids",

        "excluded_gold_ids", "excluded_non_fold_ids", "label_sha256", "label_provenance_sha256",

        "init_sha256", "init_provenance", "corpus_manifest_sha256", "corpus_build_sha256", "fold",

        "base_protocol_sha256", "base_provenance_sha256", "expected_initial_state_sha256",

        "evaluation", "screen", "bridge_contract_sha256"}

    if any(p.get(key) != ctrl.get(key) for key in scientific):

        raise ValueError("an EXP181 CONTROL scientific field changed outside the OAI arm")

    cache = cache_binding(oai_corpus)

    labels, masks, qa = verified_labels(cache["ids"])

    ids = [u for u in oai_exclusions.filter_ids(cache["ids"], p["oai_exclusion"]) if u in labels]

    if (p["control_protocol_sha256"] != HASH(bridge.manifest_path("CONTROL"))

            or p["bridge_contract_sha256"] != ctrl["bridge_contract_sha256"]

            or p["val_ids"] != ctrl["val_ids"] or p["rsna_train_ids"] != ctrl["train_ids"]

            or p["excluded_gold_ids"] != ctrl["excluded_gold_ids"]

            or p["init_sha256"] != ctrl["init_sha256"] or p["label_sha256"] != ctrl["label_sha256"]

            or p["recipe"] != control_recipe_with_oai(ctrl) or p["seed"] != 42 or p["precision"] != "bf16"

            or p["run_dirs"] != {"local": RUN_REL}

            or p["sampling"] != fixed_draw_counts(bridge.N_TRAIN, len(p["oai_train_ids"]))

            or p["source_bindings"] != source_bindings()

            or p["screen_source_bindings"] != bridge.source_bindings(bridge.SCREEN_CODE)

            or p["oai_label_sha256"] != HASH(LABELS)

            or p["oai_label_provenance_sha256"] != HASH(LABEL_PROVENANCE)

            or p["oai_label_qa_sha256"] != HASH(LABEL_QA) or p["oai_label_qa"] != qa

            or p["oai_membership"] != dict(cache_n=len(cache["ids"]), labels_n=len(labels),

                admitted_n=len(ids), cache_without_observed_label=len(cache["ids"]) - len(labels),

                withheld_labelled_n=len(labels) - len(ids),

                labelled_without_cache=0)):

        raise ValueError("EXP190 control/recipe/source binding changed")

    if p["oai_train_ids"] != ids or p["train_ids"] != ctrl["train_ids"] + ids:

        raise ValueError("EXP190 mixed cohort order changed")

    if any(p["train_labels"][u] != ctrl["train_labels"][u] or p["train_masks"][u] != [1] * 12

           for u in ctrl["train_ids"]):

        raise ValueError("EXP190 RSNA labels/masks differ from CONTROL")

    if any(p["train_labels"][u] != labels[u] or p["train_masks"][u] != masks[u] for u in ids):

        raise ValueError("EXP190 OAI labels/masks differ")

    if p["pos_weight"] != ctrl["pos_weight"]:

        raise ValueError("EXP190 class weights differ from CONTROL")

    if corpus is not None and HASH(Path(corpus) / "manifest.json") != ctrl["corpus_manifest_sha256"]:

        raise ValueError("RSNA corpus changed")

    if init_weights is not None and HASH(init_weights) != ctrl["init_sha256"]:

        raise ValueError("initial weights changed")

    return p





def require_validation(_protocol=MANIFEST):

    v = json.loads(VALIDATION.read_text(encoding="utf-8"))

    if (v.get("status") != "VALIDATED_NO_TRAINING" or v.get("manifest_sha256") != HASH(MANIFEST)

            or not v.get("full_oai_hashes") or not v.get("full_rsna_hashes")

            or not v.get("fixed_step_adapter_verified") or not v.get("masked_loss_verified")

            or v.get("cache_manifest_sha256") != CACHE_MANIFEST_SHA

            or v.get("label_sha256") != HASH(LABELS)):

        raise ValueError("EXP190 full CPU validation receipt missing/stale")

    return v





def validate(args):

    if VALIDATION.exists():

        raise FileExistsError("EXP190 validation already exists")

    p = load_registration(MANIFEST, args.corpus, args.init_weights, args.oai_cache)

    bridge.require_validation(bridge.manifest_path("CONTROL"))

    cache_binding(args.oai_cache, full_hash=True)

    bridge.base.check_corpus(args.corpus, full_hash=True)

    import torch

    import exp143_v2 as fit

    from tools.exp112_full_swa import state_dict_sha256

    if str(torch.__version__) != bridge.TORCH_VERSION:

        raise ValueError("torch stack differs from CONTROL")

    fit._cpu_seed(42)

    model = fit.build_model()

    fit.load_imagenet(model, args.init_weights)

    initial = state_dict_sha256(model.state_dict())

    del model

    if initial != bridge.INITIAL_STATE_SHA:

        raise ValueError("initial state differs from CONTROL")

    # The observed denominator remains 12; masked cells contribute zero.

    import exp155_own0949_oai_v11 as mixed_fit

    trainer_contract(mixed_fit)

    checks = mixed_fit.tiny_validation()

    if not checks.get("masked_loss_verified"):

        raise ValueError("EXP155 masked-loss CPU check failed")

    fixed_step_checks()

    result = dict(status="VALIDATED_NO_TRAINING", manifest_sha256=HASH(MANIFEST),

        control_manifest_sha256=p["control_protocol_sha256"], full_oai_hashes=True,

        full_rsna_hashes=True, cache_manifest_sha256=CACHE_MANIFEST_SHA,

        label_sha256=HASH(LABELS), initial_state_sha256=initial,

        masked_loss_verified=True, fixed_step_adapter_verified=True, torch=str(torch.__version__))

    write_new(VALIDATION, result)

    print(json.dumps(result))





def execution_settings(p, world, precision, seed, checkpointing=None):

    if world != 1 or precision != "bf16" or seed != 42 or checkpointing not in (None, CONTROL_GRAD_CHECKPOINTING):

        raise ValueError("EXP190 requires CONTROL device/precision/seed")

    control = bridge.load_registration(bridge.manifest_path("CONTROL"))

    settings = bridge.execution_settings(control, 1, "bf16", 42, CONTROL_GRAD_CHECKPOINTING)

    if (control["grad_checkpointing"] is not CONTROL_GRAD_CHECKPOINTING

            or settings["grad_checkpointing"] is not CONTROL_GRAD_CHECKPOINTING

            or settings["total_steps"] != bridge.UPDATES):

        raise ValueError("CONTROL optimizer schedule drift")

    return {k:v for k,v in settings.items() if k not in ("max_hours", "reserve_seconds")}





def validate_membership(p):

    """EXP155's loss contract on the EXP181 fold, excluding all held-out RSNA."""

    oai_exclusions.validate_training(p)

    rsna, oai = p["rsna_train_ids"], p["oai_train_ids"]

    if (len(rsna) != bridge.N_TRAIN or not oai or p["train_ids"] != rsna + oai

            or len(set(p["train_ids"])) != len(p["train_ids"])

            or set(p["train_ids"]) & (set(p["val_ids"]) | set(p["excluded_gold_ids"])

                                      | set(p["excluded_non_fold_ids"]))

            or set(p["train_labels"]) != set(p["train_ids"])

            or set(p["train_masks"]) != set(p["train_ids"])):

        raise ValueError("EXP190 mixed membership/fold exclusion differs")

    allowed = {bridge.TARGETS.index(t) for t in OAI_TARGETS}

    for uid in p["train_ids"]:

        ys, ms = p["train_labels"][uid], p["train_masks"][uid]

        if len(ys) != 12 or len(ms) != 12 or not any(ms):

            raise ValueError("EXP190 invalid mixed label row")

        if uid in rsna:

            if ms != [1] * 12 or any(not math.isfinite(y) or not 0 <= y <= 1 for y in ys):

                raise ValueError("EXP190 RSNA label/mask changed")

        elif any(m not in (0, 1) or (m and (i not in allowed or y not in (0, 1)))

                 or (not m and y != 0) for i, (y, m) in enumerate(zip(ys, ms))):

            raise ValueError("EXP190 OAI observed/masked cells changed")

    if p["pos_weight"] != bridge.base.pos_weights(p["train_labels"], rsna):

        raise ValueError("EXP190 class weights must use fold training RSNA only")





class FixedStepMixedDataset:

    """Deterministically rotate OAI/RSNA exposures without extending an epoch."""

    def __init__(self, inner):

        self.inner = inner

        self.ids = self.inner.ids

        # EXP155 keeps these on both EXP143 children, not on MixedStudyDataset.

        rsna_child, oai_child = self.inner.rsna, self.inner.oai

        if (not isinstance(rsna_child.augment, bool)

                or not isinstance(oai_child.augment, bool)

                or rsna_child.augment != oai_child.augment):

            raise ValueError("EXP155 child dataset augmentation modes differ")

        self.augment = rsna_child.augment

        self.rsna = [i for i, u in enumerate(self.ids) if u in self.inner.rsna_ids]

        self.oai = [i for i, u in enumerate(self.ids) if u in self.inner.oai_ids]

        if (len(self.rsna) != bridge.N_TRAIN or not self.oai

                or len(self.rsna) + len(self.oai) != len(self.ids)):

            raise ValueError("mixed dataset membership differs")

        self.draws = fixed_draw_counts(len(self.rsna), len(self.oai))



    def __getattr__(self, name):

        # Spawn workers may ask for attributes before pickle restores `inner`.

        inner = self.__dict__.get("inner")

        if inner is None:

            raise AttributeError(name)

        return getattr(inner, name)



    def _epoch(self):

        rsna_epoch = int(self.inner.rsna.epoch.item())

        oai_epoch = int(self.inner.oai.epoch.item())

        if rsna_epoch != oai_epoch:

            raise ValueError("EXP155 child dataset epochs differ")

        return rsna_epoch



    def set_epoch(self, epoch):

        self.inner.set_epoch(epoch)

        if self._epoch() != epoch:

            raise ValueError("EXP155 epoch propagation failed")



    @lru_cache(maxsize=32)

    def selection(self, epoch):

        import random

        if not self.augment:

            return tuple(self.rsna)

        rsna, oai = self.rsna.copy(), self.oai.copy()

        random.Random(42 + 1009 * epoch).shuffle(rsna)

        random.Random(43 + 1009 * epoch).shuffle(oai)

        # Rotate before selecting; one study is never drawn twice in an epoch.

        r = self.draws["rsna"]

        o = self.draws["oai"]

        rsna = rsna[(epoch * r) % len(rsna):] + rsna[:(epoch * r) % len(rsna)]

        oai = oai[(epoch * o) % len(oai):] + oai[:(epoch * o) % len(oai)]

        selected = rsna[:r] + oai[:o]

        random.Random(44 + 1009 * epoch).shuffle(selected)

        if len(selected) != bridge.N_TRAIN or len(set(selected)) != bridge.N_TRAIN:

            raise ValueError("EXP190 epoch selection changed/repeated a study")

        return tuple(selected)



    def __len__(self):

        # The EXP155 BN replay must see every mixed training study, once.

        return bridge.N_TRAIN if self.augment else len(self.inner)



    def __getitem__(self, i):

        if not self.augment:

            return self.inner[i]

        return self.inner[self.selection(self._epoch())[i]]





class EXP190MixedStudyDataset(FixedStepMixedDataset):

    """Importable class so Windows spawned DataLoader workers can unpickle it."""





def fixed_step_checks():

    """CPU check of the adapter's update count, mask-independent draw, and BN replay."""

    import torch

    class Fake:

        def __init__(self, augment):

            self.ids = [f"R{i}" for i in range(bridge.N_TRAIN)] + [f"OAI_{i}_R_V00" for i in range(2000)]

            self.rsna_ids = set(self.ids[:bridge.N_TRAIN])

            self.oai_ids = set(self.ids[bridge.N_TRAIN:])

            self.rsna = types.SimpleNamespace(augment=augment, epoch=torch.zeros(1, dtype=torch.int64))

            self.oai = types.SimpleNamespace(augment=augment, epoch=torch.zeros(1, dtype=torch.int64))

        def __len__(self):

            return len(self.ids)

        def set_epoch(self, epoch):

            self.rsna.epoch.fill_(epoch)

            self.oai.epoch.fill_(epoch)

        def __getitem__(self, i):

            return self.ids[i]

    a = FixedStepMixedDataset(Fake(True))

    b = FixedStepMixedDataset(Fake(False))

    if (len(a) != bridge.N_TRAIN or len(b) != bridge.N_TRAIN + 2000

            or a.draws["updates_per_epoch"] * bridge.EPOCHS != bridge.UPDATES):

        raise ValueError("fixed-step/BN adapter length differs")

    for epoch in (0, 1, bridge.EPOCHS - 1):

        a.set_epoch(epoch)

        selected = a.selection(epoch)

        if (selected != a.selection(epoch) or len(selected) != bridge.N_TRAIN

                or len(set(selected)) != bridge.N_TRAIN

                or sum(i >= bridge.N_TRAIN for i in selected) != a.draws["oai"]

                or a._epoch() != epoch or a[0] != a.inner.ids[selected[0]]):

            raise ValueError("fixed-step draw changed")

    if b[bridge.N_TRAIN] != b.inner.ids[bridge.N_TRAIN]:

        raise ValueError("BN replay dropped the OAI cohort")





def trainer_contract(fit):

    """Versioned public API plus exact source binding; no bytecode inspection."""

    if (Path(getattr(fit, "__file__", "")).resolve() != EXP155_TRAINER.resolve()

            or HASH(EXP155_TRAINER) != EXP155_TRAINER_SHA256

            or getattr(fit, "TRAINER_API_VERSION", None) != "exp155-mixed-v2"

            or not all(callable(getattr(fit, name, None)) for name in

                       ("train", "train_epoch", "mixed_dataset", "tiny_validation"))):

        raise ValueError("EXP155 explicit trainer API/source binding changed")

    return fit.TRAINER_API_VERSION





def dataset_factory(args, p, seed, augment=True):

    import exp155_own0949_oai_v11 as fit

    return EXP190MixedStudyDataset(fit.mixed_dataset(args, p, seed, augment))





def trainer_guard():

    names = ("TARGETS", "sha256", "save_torch_new", "save_torch_atomic", "read_json")

    return types.SimpleNamespace(**{name: getattr(old_oai, name) for name in names},

        load_registration=load_registration, require_validation=require_validation,

        registered_run=run_path, execution_settings=execution_settings,

        validate_membership=validate_membership, source_bindings=source_bindings, write_new_json=write_new)





def run_path(_p=None, lane="local"):

    if lane != "local":

        raise ValueError("EXP190 is local only")

    return EXP / ARM





def check_summary(summary, p, protocol_sha, model_sha):

    ident = summary.get("identity", {})

    if (summary.get("status") != "COMPLETE_TRAINING_NO_EVALUATION"

            or summary.get("epochs") != bridge.EPOCHS or summary.get("updates") != bridge.UPDATES

            or summary.get("skips") != 0 or len(summary.get("history", [])) != bridge.EPOCHS

            or [h.get("epoch") for h in summary["history"]] != list(range(1, bridge.EPOCHS + 1))

            or summary.get("model_sha256") != model_sha

            or ident.get("protocol_sha256") != protocol_sha

            or ident.get("initial_state_sha256") != bridge.INITIAL_STATE_SHA

            or ident.get("seed") != 42 or ident.get("run_dir") != RUN_REL

            or ident.get("execution") != execution_settings(p, 1, "bf16", 42, CONTROL_GRAD_CHECKPOINTING)

            or ident.get("train_n") != len(p["train_ids"])):

        raise ValueError("EXP190 fit summary incomplete or changed")

    return summary





def check_checkpoint(ck, p, protocol_sha):

    if (ck.get("oai") is not True or ck.get("arch") != p["recipe"]["arch"]

            or ck.get("contract") != "exp143_raptor96_384_c140_s02"

            or ck.get("swa_epochs_1based") != p["recipe"]["swa_1based"]

            or ck.get("checkpoint_kind") != "fixed-swa-bn" or ck.get("seed") != 42

            or ck.get("lab") != bridge.TARGETS

            or ck.get("provenance", {}).get("protocol_sha256") != protocol_sha

            or ck["provenance"].get("run_dir") != RUN_REL):

        raise ValueError("EXP190 checkpoint differs from registered fit")









def smoke(args):

    from training.arms.oai_smoke import run

    result = run(args)

    print(json.dumps({k: result[k] for k in ('status', 'device', 'verdict', 'launch_authorized', 'results')}))





def admit(args):

    """CPU admission for this engineering registration; never invent CONTROL runtime."""

    load_registration(MANIFEST, args.corpus, args.init_weights, args.oai_cache)

    require_validation()

    result = dict(status='HOLD', launch_authorized=False, manifest_sha256=HASH(MANIFEST),

        blockers=['EXP190G is engineering-only: authentic corresponding-CONTROL runtime evidence is absent',

                  'New versioned qualification with CPU/4070 smoke and independent PASS is required'])

    print(json.dumps(result))

    return result





def train(args):

    # This version records engineering parity only. Historical CONTROL lacks

    # runtime evidence; current-host observations cannot fill historical fields.

    if admit(args)['status'] != 'PASS':

        raise ValueError('EXP190G engineering registration is not launch-qualified')

    bridge.require_gpu_job(TRAIN_TAG)

    p = load_registration(MANIFEST, args.corpus, args.init_weights, args.oai_cache)

    require_validation()

    if run_path().exists() or ATTEMPT.exists() or not 0 <= args.workers <= 12:

        raise FileExistsError("EXP190 single fit/pilot attempt already exists or workers invalid")

    # Recheck every shard and receipt immediately before the sole attempt.

    cache_binding(args.oai_cache, full_hash=True)

    import exp155_own0949_oai_v11 as fit

    trainer_contract(fit)

    bridge.require_local_device()

    write_new(ATTEMPT, dict(status="STARTED_ONE_ATTEMPT", tag=TRAIN_TAG,

        manifest_sha256=HASH(MANIFEST), cache_manifest_sha256=CACHE_MANIFEST_SHA,

        started=time.strftime("%Y-%m-%dT%H:%M:%S%z")))

    args.protocol, args.device, args.precision, args.seed = str(MANIFEST), "cuda", "bf16", 42

    args.oai_corpus = str(CACHE)

    args.grad_checkpointing, args.resume = CONTROL_GRAD_CHECKPOINTING, False

    return fit.train(args, guard=trainer_guard(), dataset_factory=dataset_factory)





def infer_readout(model, corpus, ids, device, precision, chunk):

    """Shared untimed K94 readout, used by production and real-data qualification."""

    import numpy as np

    pairs = [bridge.infer_pair(model, *corpus.get(uid), device=device,

                              precision=precision, chunk=chunk) for uid in ids]

    return tuple(np.asarray(values) for values in zip(*pairs))





def predict_oof(args):

    import numpy as np

    p = load_registration(MANIFEST, args.corpus, args.init_weights, args.oai_cache)

    require_validation()

    bridge.require_gpu_job(OOF_TAG)

    run = run_path()

    protocol_sha = HASH(MANIFEST)

    check_summary(bridge.read_json(run / "summary.json"), p, protocol_sha, bridge.sha256(run / "model_best.pt"))

    for name in (bridge.OOF_PLAIN, bridge.OOF_PRIMARY, bridge.OOF_RECEIPT, bridge.INVENTORY):

        if (run / name).exists():

            raise FileExistsError("OOF output exists; no overwrite: " + name)

    device_name = bridge.require_local_device()

    from tools.exp113_radr50_local import install_gold_blocker

    from exp143_own0949 import Corpus, build_model, install_portable_gold_blocker

    install_gold_blocker(ROOT)

    install_portable_gold_blocker()

    import torch

    start = time.monotonic()

    ck = torch.load(run / "model_best.pt", map_location="cpu", weights_only=False)

    check_checkpoint(ck, p, protocol_sha)

    model = build_model(p["recipe"]["inference_res"])

    model.load_state_dict(ck["model"], strict=True)

    model.eval().to("cuda:0")

    corpus = Corpus(args.corpus)

    plain, mirror = infer_readout(model, corpus, p['val_ids'], 'cuda:0', 'fp16', 16)

    served = .5 * (plain + mirror)

    bridge.base.write_predictions(run / bridge.OOF_PLAIN, p["val_ids"], plain)

    bridge.base.write_predictions(run / bridge.OOF_PRIMARY, p["val_ids"], served)

    receipt = dict(status="PREDICTED", schema=bridge.SCHEMA, version=VERSION, arm=p["arm"], fold=0,

        rows=len(p["val_ids"]), cohort_sha256=bridge.uid_hash(p["val_ids"]), protocol_sha256=protocol_sha,

        bridge_contract_sha256=p["bridge_contract_sha256"], checkpoint_sha256=bridge.sha256(run / "model_best.pt"),

        summary_sha256=bridge.sha256(run / "summary.json"), initial_state_sha256=bridge.INITIAL_STATE_SHA,

        outputs={n: bridge.sha256(run / n) for n in (bridge.OOF_PLAIN, bridge.OOF_PRIMARY)}, primary=bridge.OOF_PRIMARY,

        k_eval=p["recipe"]["eval_k"], inference_res=p["recipe"]["inference_res"], precision="fp16",

        device_name=device_name, torch=str(torch.__version__), seconds=time.monotonic() - start)

    write_new(run / bridge.OOF_RECEIPT, receipt)

    write_new(run / bridge.INVENTORY, bridge.inventory(run, ["summary.json", "provenance.json", "model_best.pt",

                                                    bridge.OOF_PLAIN, bridge.OOF_PRIMARY, bridge.OOF_RECEIPT]))

    print(json.dumps({k: v for k, v in receipt.items() if k != "outputs"}))





def predict_gold58(args):

    """Separate, exposed diagnostic on both authenticated fold-0 checkpoints."""

    bridge.require_gpu_job(GOLD_TAG)

    if any(path.exists() for path in (GOLD_CTRL, GOLD_OAI, GOLD_RECEIPT)):

        raise FileExistsError("Gold58 prediction already attempted")

    if not SCREEN.is_file():

        raise FileNotFoundError("screen before exposed Gold58 diagnostic")

    load_registration(MANIFEST, args.corpus)

    ctrl, cand = bridge.verified_arm("CONTROL"), verified_arm()

    if ctrl["p"]["excluded_gold_ids"] != cand["p"]["excluded_gold_ids"]:

        raise ValueError("Gold exclusion changed")

    bridge.require_local_device()

    from exp143_own0949 import Corpus, build_model

    import numpy as np

    import torch

    _, ids, y = bridge.base.official_population(args.competition)

    if len(ids) != 58 or set(ids) != set(ctrl["p"]["excluded_gold_ids"]):

        raise ValueError("Gold58 cohort differs")

    corpus = Corpus(args.corpus)

    outputs = {}

    for name, rec, protocol, path, checker in (

        ("CONTROL", ctrl, bridge.manifest_path("CONTROL"), GOLD_CTRL, bridge.check_checkpoint),

        (ARM, cand, MANIFEST, GOLD_OAI, check_checkpoint)):

        ck = torch.load(rec["run"] / "model_best.pt", map_location="cpu", weights_only=False)

        checker(ck, rec["p"], HASH(protocol))

        model = build_model(rec["p"]["recipe"]["inference_res"])

        model.load_state_dict(ck["model"], strict=True)

        model.eval().to("cuda:0")

        predictions = []

        for uid in ids:

            vol, mask = corpus.get(uid)

            a, b = bridge.infer_pair(model, vol, mask, "cuda:0", "fp16")

            predictions.append(.5 * (a + b))

        path.parent.mkdir(parents=True, exist_ok=True)

        bridge.base.write_predictions(path, ids, np.asarray(predictions))

        outputs[name] = dict(path=str(path.relative_to(ROOT)), sha256=HASH(path),

                             checkpoint_sha256=HASH(rec["run"] / "model_best.pt"),

                             protocol_sha256=rec["protocol_sha256"])

        del model, ck

        torch.cuda.empty_cache()

    write_new(GOLD_RECEIPT, dict(status="PREDICTED_EXPOSED_REPORT_ONLY", rows=58,

        cohort_sha256=bridge.uid_hash(ids), official_labels_sha256=bridge.json_hash(y.astype(int).tolist()),

        serving="K94 mirror TTA, fp16", outputs=outputs))





def gold58_report(args):

    """CPU only: score the separately predicted Gold58, never select on it."""

    from sklearn.metrics import roc_auc_score

    import numpy as np

    if GOLD_REPORT.exists():

        raise FileExistsError("Gold58 diagnostic already exists")

    ctrl, cand = bridge.verified_arm("CONTROL"), verified_arm()

    _, ids, y = bridge.base.official_population(args.competition)

    r = json.loads(GOLD_RECEIPT.read_text(encoding="utf-8"))

    if (r.get("status") != "PREDICTED_EXPOSED_REPORT_ONLY" or r.get("rows") != 58

            or r.get("cohort_sha256") != bridge.uid_hash(ids)

            or r.get("official_labels_sha256") != bridge.json_hash(y.astype(int).tolist())):

        raise ValueError("Gold58 diagnostic receipt differs")

    scores = {}

    for name, rec, path in (("CONTROL", ctrl, GOLD_CTRL), (ARM, cand, GOLD_OAI)):

        bound = r["outputs"][name]

        if (bound["path"] != str(path.relative_to(ROOT)) or bound["sha256"] != HASH(path)

                or bound["checkpoint_sha256"] != HASH(rec["run"] / "model_best.pt")

                or bound["protocol_sha256"] != rec["protocol_sha256"]):

            raise ValueError("Gold58 predictions lack checkpoint/manifest binding")

        pred = bridge.load_oof(path, ids)

        per = {target: float(roc_auc_score(y[:, i], pred[:, i]))

               for i, target in enumerate(bridge.TARGETS)}

        scores[name] = dict(macro12=float(np.mean(list(per.values()))), per_target=per)

    delta = {target: scores[ARM]["per_target"][target] - scores["CONTROL"]["per_target"][target]

             for target in bridge.TARGETS}

    write_new(GOLD_REPORT, dict(status="REPORT_ONLY_EXPOSED", studies=58, targets=list(bridge.TARGETS),

        scores=scores, macro12_delta=scores[ARM]["macro12"] - scores["CONTROL"]["macro12"],

        per_target_delta=delta, inputs={name: bound["sha256"] for name, bound in r["outputs"].items()},

        labels_sha256=r["official_labels_sha256"],

        caveat="Exposed Gold58 image reads; diagnostic only, never a selection gate."))





def queue_json(args):

    load_registration(MANIFEST, args.corpus, args.init_weights, args.oai_cache)

    require_validation()

    if run_path().exists() or ATTEMPT.exists() or QUEUE.exists() or not 0 <= args.workers <= 12:

        raise FileExistsError("EXP190 run/queue exists or workers invalid")

    common = ["--corpus", str(Path(args.corpus).resolve()), "--oai-cache", str(CACHE.resolve()),

              "--init-weights", str(Path(args.init_weights).resolve()), "--workers", str(args.workers)]

    command = [args.python, "-m", "tools.exp190_coatnet_oai_f0"]

    jobs = [dict(tag=TRAIN_TAG, after=args.after, continue_on_failure=False,

                 command=[*command, "train", *common]),

            dict(tag=OOF_TAG, after=TRAIN_TAG, continue_on_failure=False,

                 command=[*command, "predict-oof", *common])]

    write_new(QUEUE, jobs)

    print(json.dumps(dict(status="QUEUE_JSON_ONLY_NOT_LAUNCHED", path=str(QUEUE))))





def queue_gold58_json(args):

    if not SCREEN.is_file() or GOLD_QUEUE.exists() or any(p.exists() for p in (GOLD_CTRL, GOLD_OAI, GOLD_RECEIPT)):

        raise FileExistsError("screen missing or Gold58 diagnostic already queued/attempted")

    bridge.verified_arm("CONTROL")

    verified_arm()

    load_registration(MANIFEST, args.corpus)

    command = [args.python, "-m", "tools.exp190_coatnet_oai_f0", "predict-gold58",

               "--corpus", str(Path(args.corpus).resolve()),

               "--competition", str(Path(args.competition).resolve())]

    write_new(GOLD_QUEUE, [dict(tag=GOLD_TAG, after=OOF_TAG, continue_on_failure=False, command=command)])

    print(json.dumps(dict(status="QUEUE_JSON_ONLY_NOT_LAUNCHED", path=str(GOLD_QUEUE))))





def verified_arm():

    """Explicit OAI identity check; never replace another module's globals."""

    from training.arms import coatnet_history

    p = load_registration()

    require_validation()

    record = coatnet_history.verify_record(p, run_path(), HASH(MANIFEST),

        summary_checker=check_summary, checkpoint_checker=check_checkpoint)

    record["historical_only"] = False

    return record





def collect(args):

    from training.arms.coatnet_screen import collect_records

    result = collect_records(bridge.verified_arm("CONTROL"), verified_arm(), candidate=ARM,

        out=getattr(args, "out", None) or COLLECTION, verification_note=args.verification_note)

    print(json.dumps(dict(status=result["status"], launch_authorized=False)))

    return result





def screen_decision(primary):

    if any(primary["references"][r]["per_target_delta"][t] is None

           for r in ("jev", "consensus") for t in bridge.TARGETS):

        raise ValueError("undefined primary target AUC")

    delta = {t: .5 * sum(primary["references"][r]["per_target_delta"][t]

                         for r in ("jev", "consensus")) for t in bridge.TARGETS}

    weak = sum(delta[t] for t in WEAK) / len(WEAK)

    dread = primary["references"]["dread"]["delta"]

    checks = dict(macro12=primary["primary_mean_delta"] >= -.001,

        other_targets=all(delta[t] >= -.010 for t in bridge.TARGETS if t not in WEAK),

        dread=dread is not None and dread >= -.003)

    status = "HOLD" if weak <= 0 or not all(checks.values()) else "REVIEW"

    return status, weak, delta, dread, checks





def oai_records_score(control, candidate, *, root=ROOT):
    """Preserve registered OAI numerical diagnostics beside the holistic readout."""
    from training.arms.coatnet_screen import score_records
    pair = score_records(control, candidate, root=root)
    primary = pair["primary_k94_mirror_tta"]
    status, weak, delta, dread, checks = screen_decision(primary)
    pair.update(historical_weak_gate=status, weak_target_mean_delta=weak,
        per_target_mean_delta=delta, oai_supervised_targets={t: delta[t] for t in OAI_TARGETS},
        gold58="exposed diagnostic only; no selection on Gold58",
        synovitis_image_read="required before adoption; compare OAI MOAKS image panel")
    return pair




def screen(args):

    from training.arms.coatnet_screen import screen_records

    result = screen_records(bridge.verified_arm("CONTROL"), verified_arm(), candidate=ARM,

        collection=getattr(args, "collection", None) or COLLECTION,

        out=getattr(args, "out", None) or SCREEN, verification_note=args.verification_note,

        readout=getattr(args, "readout", None), score_fn=oai_records_score)
    print(json.dumps(dict(status=result["status"], launch_authorized=False)))

    return result





def parser():

    ap = argparse.ArgumentParser(description=__doc__)

    sub = ap.add_subparsers(dest="command", required=True)

    for name in ("register", "validate", "queue-json", "train", "predict-oof", "smoke", "admit"):

        p = sub.add_parser(name)

        p.add_argument("--corpus", default=str(bridge.CORPUS))

        p.add_argument("--oai-cache", default=str(CACHE))

        p.add_argument("--init-weights", default=str(bridge.INIT))

        if name == "register":

            p.add_argument("--oai-exclusion-list", help="ID-only CSV/text; default is the MOAKS panel")

        if name in ("queue-json", "train", "predict-oof", "smoke"):

            p.add_argument("--workers", type=int, default=12)

        if name == "smoke":

            p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

            p.add_argument("--out", required=True)

        if name == "queue-json":

            p.add_argument("--python", default=bridge.base.DEFAULT_PYTHON)

            p.add_argument("--after", default="EXP188_COVERAGE_F0_OOF")

    for name in ("collect", "screen"):

        p = sub.add_parser(name)

        p.add_argument("--verification-note", required=True)

        p.add_argument("--out")

        if name == "screen":
            p.add_argument("--collection")
            p.add_argument("--readout", help="Optional partner, anchor and image-read evidence JSON")
    p = sub.add_parser("queue-gold58-json")

    p.add_argument("--python", default=bridge.base.DEFAULT_PYTHON)

    p.add_argument("--corpus", default=str(bridge.CORPUS))

    p.add_argument("--competition", default=str(ROOT / "storage/competition"))

    p = sub.add_parser("predict-gold58")

    p.add_argument("--corpus", default=str(bridge.CORPUS))

    p.add_argument("--competition", default=str(ROOT / "storage/competition"))

    p = sub.add_parser("gold58-report")

    p.add_argument("--competition", default=str(ROOT / "storage/competition"))

    return ap





def main(argv=None):

    args = parser().parse_args(argv)

    return globals()[args.command.replace("-", "_")](args)





if __name__ == "__main__":

    main()

