"""H-B-166 multi-view paired pilot: registration, configs and fail-closed admission v1.0.0 (2026-10-15).

One registered change versus the EXISTING H-B-155 control fits (pilot-v120,
fold 2, seeds 42/43, Vast RTX 4090): the MIL bag gets compartment-crop windows
(crop-cache-v220) in addition to the unchanged full-field native448 windows,
with a learned zero-initialised view-type embedding (training/arms/
hb166_multiview_v1.py). Everything else is the control recipe, authenticated
through tools/hb155_build_v1.validate(). The controls are NOT refit: their
predictions/checkpoints are pinned by SHA256 below. Admission = registration +
prepared caches + CPU/CUDA smoke receipts + runtime pairing + an independent
review bound to all of them. No GPU/remote action happens in this module unless
B runs the smoke/fit sequences.
"""
import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tools import hb155_build_v1 as b155
from tools.hb150_io_v1 import atomic_json

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local/scratch/slot-b/hb166/pilot-v100"
VERSION = "1.0.0"
EXPERIMENT = "H-B-166"
SEEDS = (42, 43)
GPU = b155.GPU
CONTROL_WORK = b155.WORK
CONTROL_CORPUS = b155.CONTROL
CROP_CORPUS = b155.CROP
MEMORY_GATE_BYTES = 12 * (1 << 30)   # BUILDER-CHECKLIST v1.0.6 item 3
GOLD_LABELS = "storage/gold/gold_labels.csv"
GOLD_LABELS_SHA256 = "990dd72929224e76391c93bd24e861dc75eaa69bc6f14ed71463abc9911e3b66"
GOLD_UID_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
GOLD_IDS = WORK / "gold58-ids.json"
# Existing H-B-155 control evidence (verified 2026-10-15 against readout.json
# prediction_bindings and result-inventory.json). Never refit, never replaced.
CONTROL_PINS = {
    42: dict(config="948ed73b5f389414c134d5a1a00706233e5def6170b8e943311699e9c41924d5",
             val_csv="ca0d3d7bfda6d488b16c065c0617f297074d2e3e1cf84445dd4f38e0b278b66d",
             checkpoint="58d9adc24ed72ceb9827b01bba8b08baa1e2eb37c31c327e38c85a2d157d719a"),
    43: dict(config="b81370c2e91bdd3ae354440eb0c658f4a2a55f8f84b04fc40c384c22334e92a4",
             val_csv="b4e1650c1ac06ed9d3b20a7c6b3ff6e2f47124dc0588396485b2ea31b0e05fc4",
             checkpoint="369cdb96f5c6d909ed162876180c350d30fc8b486cb6d5b89069cbaa58ccaa9a")}
CONTROL_RUNTIME_SHA256 = "c8959a6f292fb9fa110d008a94174bd6b06676b73280f1f496e467f5cb5a9459"
CONTROL_READOUT_SHA256 = "acaa97493028be3fa3afbd268735d5ae44ab6d0cf04e8dceb913ac1a19765d73"
CONTROL_INVENTORY_SHA256 = "639b14a270be4cd8962695eddec8c658998fc62ee463f27a624fd92e10de63e5"
CONTROL_FIT_FILES = ("heartbeat.json", "summary.json", "runtime.json", "final.pt", "val.csv", "val.receipt.json",
                     "val_k24.csv", "val_k24.receipt.json", "val_k36.csv", "val_k36.receipt.json")
# Host-dependent, numerically irrelevant runtime fields that may differ after a
# Vast restart on the same machine; every other snapshot key must be identical.
RUNTIME_DRIFT_ALLOWED = ("torch_interop_threads", "libs.cv2_threads")
L6_PARTNERS = {  # HB105 aligned held-out fold-2 OOF (artifacts/experiments/HB105_ALIGNED_OOF_V1 pins)
    "R50": {"42": ("artifacts/experiments/EXP169_RELEASE_V2/pre-reading/l2-flip-oof/EXP157_L2_S42_F2/oof_k36_flip_tta.csv",
                   "722d429262ea4b7ebce144befc34728ac1010e9017947756fa1cbe9f8c3160a2"),
            "43": ("artifacts/experiments/EXP169_RELEASE_V2/pre-reading/l2-flip-oof/EXP157_L2_S43_F2/oof_k36_flip_tta.csv",
                   "0bf4b8721162671fc4943cb84443445e96a9c84cd1e9367556c7fb6cd8db8abe")},
    "C4": {"42": ("artifacts/experiments/EXP183_MULTISCALE_5FOLD/outputs/fold2/results/EXP183_M42_F2/oof_k36_flip_tta.csv",
                  "6e853ca4574a2c3269823e10947666cee10d7a9b6e7498d0240d444ab1bbff8e")},
    "T": {"42": ("artifacts/experiments/EXP186_FINDING_MIL_5FOLD/outputs-b/fold2/results/EXP186B_FINDING_MIL_S42_F2/oof_k36_flip_tta.csv",
                 "eac7994db0326797b72278f2d56b9b3df07c3e2b76b2d57a8c73642628541955"),
          "43": (".local/refactor/mil-s43-members-collected-v1/fold2/outputs/oof_k36_flip_tta.csv",
                 "3891cb8ddf626b70fe5c384e287c1e68e6193e67ea9a50347a95262483c4af5d")}}
HB105_MANIFEST = ("artifacts/experiments/HB105_ALIGNED_OOF_V1/input-source-manifest.json")
WINDOW_BUDGET = dict(
    rule="K windows PER VIEW; bag = K full-field + K compartment-crop windows",
    train=dict(full=12, crop=12, bag=24), val_k24=dict(full=24, crop=24, bag=48),
    val_k36=dict(full=36, crop=36, bag=72), val_k36flip=dict(full=36, crop=36, bag=72, passes=2),
    gold_k36flip=dict(full=36, crop=36, bag=72, passes=2),
    backbone_compute_vs_control=2.0,
    full_field="byte-identical to the H-B-155 control windows (same class hb155_crops_v1.Windows on native448, same RNG streams)",
    crop="H-B-155 crop-arm windowing on crop-cache-v220 (side-v2 flag authenticated) with independent RNG streams "
         "hb166-crop-windows / hb166-crop-intensity; evaluation windows identical to the crop arm")
VIEW_EMBEDDING = dict(types=["full_field_native448", "compartment_crop_v220"], dim=768, parameters=1536,
    init="zeros (no RNG draw; at init the full-field-only forward equals the control model)",
    placement="added to every spatial token of a window before the shared finding LayerNorm (model.norm)",
    pooling="unchanged twelve per-target attention heads over all valid tokens of both views",
    optimizer_group="head (lr 1e-3, weight decay 0.02), same AdamW/OneCycle as control")
GATE = dict(
    name="H-B-166 registered pass signal (fixed 2026-10-15 before any fit)",
    aggregate="AUC of arithmetic seed-mean probabilities: mean(multiview s42,s43) vs mean(control s42,s43), fixed epoch 20, K36 flip",
    four_ruler_macro_min=.002,
    ci="paired connected-component bootstrap, 300 draws, seed 20261003 (tools/multi_lens_readout.py component contract)",
    ci_rule="95% lower bound of the 12-target macro delta > 0 on >= 2 of the 4 rulers (jev, consensus, dread, aquila)",
    ci_positive_rulers_min=2,
    target_floor=-.005,
    target_rule="no target with 4-ruler mean delta (seed-mean predictions) < -0.005",
    both_seeds_positive="for each seed s: 4-ruler mean 12-target macro of AUC(multiview_s) - AUC(control_s) > 0",
    on_pass="PASS_FOLD3_CONFIRMATION_REQUIRED (fold 2 informed the idea; a PASS triggers a fold-3 confirmation, not a release)",
    on_fail="NO_PASS (all seven lenses still reported; B judges holistically)")
BIAS_STATEMENT = ("Fold 2 informed the idea: H-B-155 on this same fold (D-B-192) showed crop gains on Medial/Lateral OA and "
                  "whole-joint losses, which motivated keeping the full field and adding crop views. This fold-2 pilot "
                  "is therefore selection-exposed (EXPOSED, clean_oof=false). A PASS triggers a fold-3 confirmation "
                  "with fresh controls; it never authorises a release.")
LENS_PLAN = dict(
    L1_L5="tools/multi_lens_readout_l4_v2.evaluate on the 864 fold-2 rows; L5 uses BOTH same-recipe seed pairs (42/42, 43/43)",
    L6="partner bundle on the same 864 rows: equal-family percentile-rank mean of HB105 held-out fold-2 R50 (seed-mean 42/43), "
       "C4 (s42), T (seed-mean 42/43) plus the own ConvNeXt-S slot = multiview seed-mean vs control seed-mean "
       "(deployed S members replaced; HB105 'add' structure); lens threshold +0.0005 on the jev+consensus macro",
    L7="Gold58 (58 studies outside the fold-2 roster; exposed/descriptive, never eligible by the lens rule) from K36-flip "
       "inference of all four checkpoints on the same 4090 (multiview inside the fit, controls via tools/hb166_panel_v1.py). "
       "OAI MOAKS ineligible: no native448 or compartment-crop OAI cache exists (only legacy 336/384 OAI caches); building "
       "both OAI caches with geometry/laterality QA is a separate data job and the H-B-155 controls were never run on OAI.")
SOURCES = ["training/arms/hb166_multiview_v1.py", "tools/hb166_build_v1.py", "tools/hb166_fit_v1.py",
           "tools/hb166_panel_v1.py", "tools/hb166_watchdog_v1.py", "tools/hb166_readout_v1.py",
           "tools/hb155_build_v1.py", "tools/hb155_fit_v1.py", "tools/hb155_watchdog_v1.py",
           "training/arms/hb155_crops_v1.py", "tools/parity_runtime.py", "config/workspace.toml",
           "tools/gpu_job.py", "tools/gpu_queue.py", "tools/job_runtime.py",
           "tools/multi_lens_readout.py", "tools/multi_lens_readout_l4_v2.py"]


def need(ok, why):
    if not ok:
        raise ValueError(why)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def uid_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def immutable(p, value):
    p = Path(p)
    if p.exists():
        need(read(p) == value, "immutable binding changed: " + rel(p))
    else:
        atomic_json(p, value)


def member(seed):
    need(seed in SEEDS, "unregistered seed")
    return f"multiview-s{seed}"


def control_member(seed):
    need(seed in SEEDS, "unregistered seed")
    return f"control-s{seed}"


def config_path(seed, engineering=False):
    return WORK / ("cpu-configs" if engineering else "configs") / (member(seed) + ".json")


def source_closure():
    """Explicit local Python dependency closure (same algorithm as hb155_build_v1)."""
    found = set(SOURCES)
    todo = [p for p in found if p.endswith(".py")]
    while todo:
        path = todo.pop()
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    package = path.split("/")[:-node.level]
                    base = ".".join(package + ([base] if base else []))
                names = [base] + [base + "." + a.name for a in node.names if a.name != "*"]
            for name in names:
                stem = name.replace(".", "/")
                for prefix in ("", "training/src/", "tools/"):
                    for suffix in (".py", "/__init__.py"):
                        candidate = prefix + stem + suffix
                        if candidate not in found and (ROOT / candidate).is_file():
                            found.add(candidate)
                            todo.append(candidate)
    return sorted(found)


def gold_ids():
    """Gold58 UIDs (ids only, never label values), frozen in the stage."""
    if not GOLD_IDS.exists():
        import pandas as pd
        need(sha(ROOT / GOLD_LABELS) == GOLD_LABELS_SHA256, "Gold58 label file differs from pin")
        ids = sorted(pd.read_csv(ROOT / GOLD_LABELS, usecols=["StudyInstanceUID"], dtype=str)["StudyInstanceUID"])
        atomic_json(GOLD_IDS, dict(schema="hb166-gold58-ids/v1", source=GOLD_LABELS,
                                   source_sha256=GOLD_LABELS_SHA256, ids=ids, uid_sha256=uid_hash(ids)))
    g = read(GOLD_IDS)
    need(g.get("schema") == "hb166-gold58-ids/v1" and len(g["ids"]) == 58 and len(set(g["ids"])) == 58
         and g["ids"] == sorted(g["ids"]) and uid_hash(g["ids"]) == g["uid_sha256"] == GOLD_UID_SHA256,
         "Gold58 roster differs from registered pin")
    return g["ids"]


def control_bindings(seed):
    """Existing H-B-155 control evidence from its pinned result inventory."""
    inventory_path = CONTROL_WORK / "result-inventory.json"
    need(sha(inventory_path) == CONTROL_INVENTORY_SHA256, "H-B-155 result inventory differs from pin")
    files = read(inventory_path)["files"]
    pins = CONTROL_PINS[seed]
    base = rel(CONTROL_WORK / "fits" / control_member(seed))
    out = {}
    for name in CONTROL_FIT_FILES:
        key = base + "/" + name
        need(key in files, "control result missing from inventory: " + key)
        out[key] = files[key]["sha256"]
    need(out[base + "/val.csv"] == pins["val_csv"] and out[base + "/final.pt"] == pins["checkpoint"]
         and out[base + "/runtime.json"] == CONTROL_RUNTIME_SHA256, "control prediction/checkpoint/runtime pin differs")
    return out


def expected(seed, engineering=False):
    control_path = b155.config_path("control", seed)
    need(sha(control_path) == CONTROL_PINS[seed]["config"], "H-B-155 control config differs from pin")
    c = copy.deepcopy(b155.validate(control_path))  # authenticates the full control recipe/source closure
    need(c["arm"] == "control" and c["seed"] == seed and c["fold"] == 2 and not c["engineering_only"],
         "control identity differs")
    gold = gold_ids()
    need(not set(gold) & set(c["train"] + c["val"]), "Gold58 overlaps the fold-2 roster")
    train, val = c["train"], c["val"]
    if engineering:
        train, val, gold = train[:2], val[:1], gold[:1]
    inputs = {path: sha(ROOT / path) for path in source_closure()}
    inputs[rel(control_path)] = CONTROL_PINS[seed]["config"]
    inputs[rel(GOLD_IDS)] = sha(GOLD_IDS)
    for directory in (CONTROL_CORPUS, CROP_CORPUS):
        inputs[directory + "/manifest.json"] = sha(ROOT / directory / "manifest.json")
    return dict(schema="hb166-config/v1", version=VERSION, experiment=EXPERIMENT, arm="multiview", seed=seed, fold=2,
        control_config=rel(control_path), control_config_sha256=CONTROL_PINS[seed]["config"],
        control_member=control_member(seed), control_fits=control_bindings(seed),
        recipe=c["recipe"], train=train, val=val, gold=gold, weights=c["weights"],
        label_path=c["label_path"], label_semantic_sha256=c["label_semantic_sha256"],
        corpus=CONTROL_CORPUS, corpus_contract=c["corpus_contract"],
        crop_corpus=CROP_CORPUS, crop_corpus_contract="hb155_compartment64_448_side_v2",
        resolution=224, epochs=20, batch=8, microbatch=2, accumulation=4, train_k=12, workers=2,
        optimizer_steps=c["optimizer_steps"], selection="fixed_epoch20", gpu_name=GPU,
        window_budget=WINDOW_BUDGET, view_embedding=VIEW_EMBEDDING, memory_gate_bytes=MEMORY_GATE_BYTES,
        exclusions_sha256=c["exclusions_sha256"], side_unknown_policy=c["side_unknown_policy"],
        gate=GATE, bias_statement=BIAS_STATEMENT, exposure="EXPOSED", clean_oof=False,
        estimand="paired multiview-minus-control delta on exposed fold 2 against the existing H-B-155 control fits",
        engineering_only=engineering, inputs=dict(sorted(inputs.items())), review_required=True)


def validate(path):
    c = read(path)
    need(c.get("schema") == "hb166-config/v1", "foreign config")
    need(c == expected(c["seed"], c["engineering_only"]), "config/source binding differs")
    return c


def registration_body():
    configs = {rel(config_path(seed, engineering)): sha(config_path(seed, engineering))
               for seed in SEEDS for engineering in (False, True)}
    controls = {control_member(seed): dict(config_sha256=CONTROL_PINS[seed]["config"], files=control_bindings(seed))
                for seed in SEEDS}
    partners = {family: {seed: dict(path=p, sha256=d) for seed, (p, d) in seeds.items()}
                for family, seeds in L6_PARTNERS.items()}
    return dict(schema="hb166-registration/v1", version=VERSION, experiment=EXPERIMENT, plan_item="H-B-166",
        title="Multi-view own model: unchanged full-field native448 windows + compartment crop windows, learned view embedding",
        fold=2, seeds=list(SEEDS), members=[member(s) for s in SEEDS], configs=configs,
        controls=dict(source="existing H-B-155 control fits (pilot-v120), not refit", members=controls,
                      readout_sha256=CONTROL_READOUT_SHA256, result_inventory_sha256=CONTROL_INVENTORY_SHA256,
                      runtime_sha256=CONTROL_RUNTIME_SHA256),
        window_budget=WINDOW_BUDGET, view_embedding=VIEW_EMBEDDING, gate=GATE, lens_plan=LENS_PLAN,
        l6_partners=dict(manifest=HB105_MANIFEST, files=partners,
                         exposure="held-out fold-2 OOF of production families trained on the fold-2 complement; fold 2 is selection-exposed"),
        gold58=dict(ids_file=rel(GOLD_IDS), ids_sha256=sha(GOLD_IDS), uid_sha256=GOLD_UID_SHA256,
                    label_file=GOLD_LABELS, label_sha256=GOLD_LABELS_SHA256, status="EXPOSED_DESCRIPTIVE"),
        bias_statement=BIAS_STATEMENT, exposure="EXPOSED", clean_oof=False,
        pairing="same GPU type (RTX 4090 sm89) and identical software stack as the control fits; runtime snapshot diff "
                "vs the control runtime.json must be empty except " + ", ".join(RUNTIME_DRIFT_ALLOWED),
        recipe="identical to H-B-155 control: batch 8 / microbatch 2 / accumulation 4, workers 2, 224 px, 20 epochs, "
               "K12 train (per view), K24/K36/K36flip validation (per view), fixed final epoch, L2 labels, fp16 strict runtime",
        memory_gate_bytes=MEMORY_GATE_BYTES, not_release_authorization=True)


def register():
    for seed in SEEDS:
        for engineering in (False, True):
            immutable(config_path(seed, engineering), expected(seed, engineering))
    body = registration_body()
    path = WORK / "registration.json"
    if path.exists():
        old = read(path)
        need({k: v for k, v in old.items() if k != "registered_at"} == body, "registration changed after registering")
    else:
        stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        atomic_json(path, {**body, "registered_at": stamp})
    return dict(status="REGISTERED", members=len(SEEDS), engineering_members=len(SEEDS),
                registration_sha256=sha(path), exposure="EXPOSED", clean_oof=False)


def registration_binding():
    path = WORK / "registration.json"
    need(path.is_file(), "registration missing; run register before any smoke or fit")
    old = read(path)
    need({k: v for k, v in old.items() if k != "registered_at"} == registration_body() and old.get("registered_at"),
         "registration differs from code/configs")
    return {rel(path): sha(path)}


def cache_bindings(c):
    import numpy as np
    out = {}
    for directory, contract, names in ((c["corpus"], c["corpus_contract"], ("ids", "masks")),
                                       (c["crop_corpus"], c["crop_corpus_contract"], ("ids", "masks", "side_unknown"))):
        d = ROOT / directory
        m = read(d / "manifest.json")
        need(m.get("finished") is True and m["contract"] == contract, "cache incomplete/foreign: " + directory)
        for name in names:
            need(sha(d / (name + ".npy")) == m[name + "_sha256"], "cache index changed: " + directory)
        ids = np.load(d / "ids.npy", allow_pickle=False).astype(str).tolist()
        need(set(c["train"] + c["val"] + c["gold"]) <= set(ids), "cache roster missing: " + directory)
        need(m["shape_total"] == [len(ids), 64, 448, 448], "cache shape differs")
        end = 0
        for s in m["shards"]:
            need(s["start"] == end and s["stop"] > end and Path(s["file"]).name == s["file"], "invalid shard roster")
            need((d / s["file"]).stat().st_size == s["bytes"], "cache shard missing/truncated")
            end = s["stop"]
        need(end == len(ids), "incomplete shard roster")
        out[rel(d / "manifest.json")] = sha(d / "manifest.json")
    return out


def prepare(engineering=False):
    registration = registration_binding()
    bindings = dict(registration)
    for seed in SEEDS:
        path = config_path(seed, engineering)
        c = validate(path)
        bindings[rel(path)] = sha(path)
        bindings.update(cache_bindings(c))
    immutable(WORK / ("cpu-prepared.json" if engineering else "prepared.json"),
              dict(schema="hb166-prepared/v1", bindings=bindings))
    return dict(status="PREPARED_ENGINEERING" if engineering else "PREPARED", members=len(SEEDS))


def check_runtime(snapshot, reference):
    from tools.parity_runtime import diff_runtime
    drift = {k: v for k, v in diff_runtime(reference, snapshot).items() if k not in RUNTIME_DRIFT_ALLOWED}
    need(snapshot.get("gpu") == GPU and snapshot.get("capability") == [8, 9], "paired runs require an RTX 4090 sm89")
    need(not drift, "runtime differs from the control fits: " + ", ".join(sorted(drift)))
    return True


def require_runtime(snapshot):
    """Same reviewed 4090 stack as the control fits (hb155 runtime-review PASS)."""
    reference = CONTROL_WORK / "runtime" / "4090.json"
    need(sha(reference) == CONTROL_RUNTIME_SHA256, "control runtime reference differs")
    review = read(CONTROL_WORK / "runtime-review.json")
    need(review.get("verdict") == "PASS" and review.get("independent") is True
         and review.get("bindings", {}).get(rel(reference)) == CONTROL_RUNTIME_SHA256,
         "control runtime review absent/stale")
    return check_runtime(snapshot, read(reference))


def smoke_paths():
    paths = []
    for seed in SEEDS:
        paths.append(("cpu", "fit", seed, WORK / "cpu" / member(seed) / "summary.json"))
        paths.append(("cuda", "fit", seed, WORK / "cuda" / member(seed) / "summary.json"))
        paths.append(("cpu", "panel", seed, WORK / "cpu" / ("panel-" + control_member(seed)) / "summary.json"))
        paths.append(("cuda", "panel", seed, WORK / "cuda" / ("panel-" + control_member(seed)) / "summary.json"))
    return paths


def check_smoke(device, kind, seed, path):
    s = read(path)
    need(s.get("passed") is True and s.get("smoke") is True and s.get("engineering_only") is True
         and s.get("device") == device and s.get("seed") == seed and s.get("version") == VERSION, "smoke identity differs: " + rel(path))
    cp = config_path(seed, device == "cpu")
    need(s.get("config_sha256") == sha(cp), "smoke config binding differs: " + rel(path))
    if device == "cpu":
        need(s.get("cpu_cuda_initialized") is False, "CPU smoke initialised CUDA")
    else:
        need(s.get("gpu_name") == GPU and 0 < s.get("peak_reserved_bytes", 0) <= MEMORY_GATE_BYTES, "CUDA memory gate")
        need(s.get("full_shape") is True and s.get("runtime_pairing") is True, "CUDA smoke shape/runtime pairing")
    if kind == "fit":
        need(s.get("arm") == "multiview" and s["optimizer_steps"] == (1 if device == "cpu" else 2), "smoke steps differ")
        need(s.get("validation_shapes") == ["k24", "k36", "k36flip", "gold_k36flip"], "smoke shapes differ")
        if device == "cpu":
            need(s.get("cpu_model") == "real", "CPU smoke must run the real ConvNeXt-S multiview model")
        if device == "cuda":
            need(s["batch"] == 8 and s["val_batch"] == 4, "smoke batch differs")
            need(all(s["healthy_rates"].get(k, 0) > 0 for k in ("train_step", "validation", "panel")), "rates missing")
    else:
        need(s.get("arm") == "control-panel" and s.get("checkpoint_sha256") == CONTROL_PINS[seed]["checkpoint"],
             "panel smoke checkpoint differs")
        if device == "cuda":
            need(s.get("healthy_rates", {}).get("panel", 0) > 0, "panel rate missing")
    return cp


def review_bindings():
    bindings = registration_binding()
    for name in ("prepared.json", "cpu-prepared.json"):
        p = WORK / name
        need(p.is_file(), name + " missing; run prepare")
        bindings[rel(p)] = sha(p)
    need(prepare()["status"] == "PREPARED" and prepare(True)["status"] == "PREPARED_ENGINEERING", "prepare drift")
    for device, kind, seed, path in smoke_paths():
        cp = check_smoke(device, kind, seed, path)
        bindings[rel(cp)] = sha(cp)
        bindings[rel(path)] = sha(path)
    for seed in SEEDS:
        runtime = WORK / "cuda" / member(seed) / "runtime.json"
        require_runtime(read(runtime))
        bindings[rel(runtime)] = sha(runtime)
    return bindings


def require_fit(path, review=None):
    c = validate(path)
    need(not c["engineering_only"], "engineering config cannot fit")
    need(c.get("exposure") == "EXPOSED" and c.get("clean_oof") is False and c.get("fold") == 2, "exposure label differs")
    need(review is not None, "independent review required")
    r = read(review)
    need(r.get("schema") == "hb166-review/v1" and r.get("verdict") == "PASS" and r.get("independent") is True
         and bool(r.get("reviewer")) and r.get("bindings") == review_bindings(),
         "independent smoke-bound review required (schema hb166-review/v1, verdict PASS, exact review-bindings)")
    return c


def admit(review=None):
    try:
        for seed in SEEDS:
            require_fit(config_path(seed), review)
        return dict(status="ADMITTED", exposure="EXPOSED", clean_oof=False)
    except (ValueError, OSError, KeyError) as exc:
        return dict(status="HOLD", reason=str(exc))


def disk_gate(path, additional_bytes):
    import shutil
    free = shutil.disk_usage(path).free
    need(free >= additional_bytes, f"insufficient disk: need {additional_bytes} free bytes; have {free}")
    return free


def watchdog(attempt, *, measurement=False, rates=None):
    cmd = [sys.executable, "-B", "-m", "tools.hb166_watchdog_v1", "--attempt", str(attempt),
           "--budget", str(WORK / "gpu-budget.json")]
    return cmd + (["--measurement"] if measurement else ["--healthy-rates", str(rates)])


def sequence(mode, review=None, attempt="v1"):
    """smoke: 2 control Gold58 panel smokes + 2 multiview CUDA smokes; fit: 2 control panels (fast) + 2 fits."""
    need(mode in ("smoke", "fit") and attempt.replace("-", "").isalnum(), "invalid sequence")
    disk_gate(WORK, 30 * (1 << 30))
    if mode == "fit":
        need(admit(review)["status"] == "ADMITTED", "admission HOLD")
    jobs = []
    for seed in SEEDS:
        smoke_name = "panel-" + control_member(seed)
        out = WORK / "cuda" / smoke_name if mode == "smoke" else WORK / "panel" / control_member(seed)
        jobs.append((smoke_name, out, ["tools.hb166_panel_v1", mode, "--seed", str(seed)]))
    for seed in SEEDS:
        name = member(seed)
        out = WORK / ("cuda" if mode == "smoke" else "fits") / name
        jobs.append((name, out, ["tools.hb166_fit_v1", mode, "--config", str(config_path(seed))]))
    for name, out, args in jobs:
        att = WORK / "attempts" / f"{mode}-{name}-{attempt}"
        if mode == "smoke":
            cmd = watchdog(att, measurement=True)
        else:
            rates = WORK / "cuda" / name / "healthy-rates.json"
            immutable(rates, read(WORK / "cuda" / name / "summary.json")["healthy_rates"])
            cmd = watchdog(att, rates=rates)
        cmd += ["--", sys.executable, "-B", "-m", *args, "--out", str(out), "--device", "cuda"]
        if review:
            cmd += ["--review", str(review)]
        subprocess.run(cmd, cwd=ROOT, check=True)
    return dict(status="SEQUENCE_" + mode.upper() + "_COMPLETE", jobs=[j[0] for j in jobs])


def snapshot_runtime(out, device="cpu"):
    if device == "cpu":
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # empty, never -1 (see tools/hb166_fit_v1.hide_gpus_for_cpu)
    from tools import hb166_fit_v1 as f
    f.h.torch.set_num_threads(2)
    f.h.configure_runtime("none" if device == "cpu" else "fp16", device, policy="strict")
    snapshot = f.runtime_info()
    atomic_json(out, snapshot)
    if device == "cuda":
        require_runtime(snapshot)
    return dict(status="SNAPSHOT", device=device, path=str(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("register", "validate", "prepare", "admit", "review-bindings",
                                        "smoke-sequence", "fit-sequence", "snapshot", "closure"))
    ap.add_argument("--engineering", action="store_true")
    ap.add_argument("--review", type=Path)
    ap.add_argument("--attempt", default="v1")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = ap.parse_args(argv)
    try:
        if a.command == "register":
            result = register()
        elif a.command == "validate":
            for seed in SEEDS:
                validate(config_path(seed, a.engineering))
            result = dict(status="VALID", members=len(SEEDS), registration=registration_binding())
        elif a.command == "prepare":
            result = prepare(a.engineering)
        elif a.command == "admit":
            result = admit(a.review)
        elif a.command == "review-bindings":
            result = review_bindings()
        elif a.command == "snapshot":
            need(a.out is not None, "snapshot needs --out")
            result = snapshot_runtime(a.out, a.device)
        elif a.command == "closure":
            result = dict(status="CLOSURE", files=source_closure())
        else:
            result = sequence(a.command.split("-")[0], a.review, a.attempt)
    except (ValueError, OSError, KeyError) as exc:
        if a.command in ("smoke-sequence", "fit-sequence"):
            raise
        result = dict(status="HOLD", reason=str(exc))
    print(json.dumps(result))
    return 2 if result.get("status") == "HOLD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
