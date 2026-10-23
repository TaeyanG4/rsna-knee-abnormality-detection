"""H-B-166 fold-3 confirmation v1.1.0: seeds44/45, reused H-B-165 controls.
CPU build only; B admits all GPU execution. See registration and CHANGELOG.
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
from tools import hb165_build_v1 as b165
from tools import hb165_readout_v1 as r165
from tools.hb150_io_v1 import atomic_json

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local/scratch/slot-b/hb166/pilot-fold3-v110"
VERSION = "1.1.0"
EXPERIMENT = "H-B-166"
SEEDS = (44, 45)
GPU = b155.GPU
CONTROL_WORK = b165.WORK
CONTROL_CORPUS = b155.CONTROL
CROP_CORPUS = b155.CROP
MEMORY_GATE_BYTES = 12 * (1 << 30)   # BUILDER-CHECKLIST v1.0.6 item 3
GOLD_LABELS = "storage/gold/gold_labels.csv"
GOLD_LABELS_SHA256 = "990dd72929224e76391c93bd24e861dc75eaa69bc6f14ed71463abc9911e3b66"
GOLD_UID_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
GOLD_IDS = WORK / "gold58-ids.json"
# Existing H-B-165 fold-3 control evidence (authenticated complete receipts).
# Never refit or replace unless pairing validation fails.
CONTROL_PINS = {44: {'config': '08093d35981c303db1b5366e18d43cfd631becc834ce602eed2710f96ef0a27c', 'val_csv': '1a7ec61a64123bbbc41f8d35ea65b199914e8e266a1103d0ff057c9ba7a24413', 'checkpoint': '7ed35b89c87b51cb95496e331b5aaf899df6157ba740161e4a4e97dbca712b1b'}, 45: {'config': '0614bad6a6c44b2845b831561a4a259bd2147ac31f580ea478f21521692f7eff', 'val_csv': '2a916a2e05effb155c1592d827e059ed6c21a93dbd6360cbe282ff7e05c8cd0b', 'checkpoint': '841982b8097d464d12608fbf5d255da2a18e1c1e0a855ec7585e4264bb7ecdbe'}}
CONTROL_RUNTIME_SHA256 = "c8959a6f292fb9fa110d008a94174bd6b06676b73280f1f496e467f5cb5a9459"
CONTROL_FIT_FILES = ('heartbeat.json', 'summary.json', 'runtime.json', 'final.pt', 'val.csv', 'val.receipt.json', 'val_k24.csv', 'val_k24.receipt.json', 'val_k36.csv', 'val_k36.receipt.json', 'gold.csv', 'gold.receipt.json')
CONTROL_FILE_PINS = {'.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/heartbeat.json': '00ac0f02f251eb7570f5cbec8acc727965a1e1c7c81b303be9e485f122394cd3', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/summary.json': '4b24854693a30bcfd8673e97287ee71209098514e749147e9c6e81d00f4b5461', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/runtime.json': 'c8959a6f292fb9fa110d008a94174bd6b06676b73280f1f496e467f5cb5a9459', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/final.pt': '7ed35b89c87b51cb95496e331b5aaf899df6157ba740161e4a4e97dbca712b1b', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/val.csv': '1a7ec61a64123bbbc41f8d35ea65b199914e8e266a1103d0ff057c9ba7a24413', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/val.receipt.json': 'd0710582515a993afbe50e72ae508725ac101e9eb28926ed95185c507cf60281', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/val_k24.csv': '5e4606c3564ef5fd6ed2e5abab462550b15bb52d268c025c6658f83d4f0c602b', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/val_k24.receipt.json': '90c2a5992bf8ce01aaedae012817ac0a398dad44f1ff8f181134bdccb23e5432', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/val_k36.csv': '7f7d25a14896d30be626693666290cfe4d4ab384b69e36795854c947e9fddbaa', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/val_k36.receipt.json': '4db7207ad08222ed43b687e70b8eeab924764e020ffde63029f2fef200676a90', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/gold.csv': '089e88ba82712f19c6993d40c89e86463581b6da4e1cfdcc709a4f10eb2fe088', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s44/gold.receipt.json': '76d1e8f71bf92179242dd06e98d2722c0f9d44ac5dd1583eaae974c63fbf15c3', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/heartbeat.json': '00ac0f02f251eb7570f5cbec8acc727965a1e1c7c81b303be9e485f122394cd3', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/summary.json': '619e35f6f6182ca3b0a3124998a2b67d23285df1de395c61ea8ecb3ebe2ae8ca', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/runtime.json': 'c8959a6f292fb9fa110d008a94174bd6b06676b73280f1f496e467f5cb5a9459', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/final.pt': '841982b8097d464d12608fbf5d255da2a18e1c1e0a855ec7585e4264bb7ecdbe', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/val.csv': '2a916a2e05effb155c1592d827e059ed6c21a93dbd6360cbe282ff7e05c8cd0b', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/val.receipt.json': '7b293dc1ce2f62e181123d440566ad4c1fe8ed5db8dae730423e6c6cafa3c511', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/val_k24.csv': 'e4fc0b95a4874e29dd95df55440416ea5fde191c03e1f4ea378ba0bac1ffbe8d', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/val_k24.receipt.json': '3c7179ea784836773c24b1c403cddd6101706c546469abcc7337c1f81e533018', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/val_k36.csv': '252da8b4649219bafd6ff9b134e72d5c88b75cc9a03091dceda2a9e27d3dda18', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/val_k36.receipt.json': 'ac5386b5f8495b5d22ec4590bcb5c0aa1547275998c0249b0d16d6e4de866e72', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/gold.csv': 'ccc66d63b079b6f7c025fb20a357aeec3dc47178d88dd217f62b728b74237d21', '.local/scratch/slot-b/hb165/pilot-fold3-v100/fits/control-s45/gold.receipt.json': '3642de69ccda0cb74f13b37068ac77f0d08c1fba56fef1cfa4542ce51b5f5d55'}
# Host-dependent, numerically irrelevant runtime fields that may differ after a
# Vast restart on the same machine; every other snapshot key must be identical.
RUNTIME_DRIFT_ALLOWED = ("torch_interop_threads", "libs.cv2_threads")
L6_PARTNERS = {'C4': {'42': ('artifacts/experiments/EXP183_MULTISCALE_5FOLD/outputs/fold3/results/EXP183_M42_F3/oof_k36_flip_tta.csv', 'e15ce3ab3d295e5181ec77c15d24e209433c3edb3802aedd68763bbe07893357')}, 'R50': {'42': ('artifacts/experiments/EXP169_RELEASE_V2/pre-reading/l2-flip-oof/EXP157_L2_S42_F3/oof_k36_flip_tta.csv', 'f5e6511fd16c922cb616f0e06a9f2ec46d6554fc676772ec98660060177d6efa'), '43': ('artifacts/experiments/EXP169_RELEASE_V2/pre-reading/l2-flip-oof/EXP157_L2_S43_F3/oof_k36_flip_tta.csv', '5d2b9d28d13d623de353068f65a6d1c6ed8e09a8275ba4d11939546f89e65231')}, 'T': {'42': ('artifacts/experiments/EXP186_FINDING_MIL_5FOLD/outputs-b/fold3/results/EXP186B_FINDING_MIL_S42_F3/oof_k36_flip_tta.csv', '736f7daff87e863232fa1d84aa2db3a88ea6982b1032cab393b5ca32f3ffdd5f'), '43': ('.local/refactor/mil-s43-members-collected-v1/fold3/outputs/oof_k36_flip_tta.csv', 'f242a046670594e623f609e5977422274890d4846ed4e7993022e57dc687ab0e')}}
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
GATE = {'name': 'H-B-166 fold-3 confirmation (fixed before fits)', 'aggregate': 'AUC of arithmetic seed-mean probabilities: mean(multiview s44,s45) vs mean(control s44,s45), fixed epoch 20, K36 flip', 'four_ruler_macro_min': 0.002, 'ci': 'paired connected-component bootstrap, 300 draws, seed 20261003 (tools/multi_lens_readout.py component contract)', 'ci_rule': '95% lower bound of the 12-target macro delta > 0 on >= 2 of the 4 rulers (jev, consensus, dread, aquila)', 'ci_positive_rulers_min': 2, 'target_floor': -0.005, 'target_rule': 'no target with 4-ruler mean delta (seed-mean predictions) < -0.005', 'both_seeds_positive': 'for each seed s: 4-ruler mean 12-target macro of AUC(multiview_s) - AUC(control_s) > 0', 'on_pass': 'PASS_FOLD3_CONFIRMATION (not release authorization)', 'on_fail': 'NO_PASS (all seven lenses still reported; B judges holistically)'}
BIAS_STATEMENT = 'EXPOSED, clean_oof=false. Fold 2 motivated this confirmation; fold 3 previously informed the shared recipe. New seeds 44/45 are paired with existing full-field controls. No clean holdout or release claim.'
LENS_PLAN = {'L1_L5': 'tools/multi_lens_readout_l4_v2.evaluate on the 864 fold-3 rows; L5 uses BOTH same-recipe seed pairs (44/44, 45/45)', 'L6': "partner bundle on the same 864 rows: equal-family percentile-rank mean of HB105 held-out fold-3 R50 (seed-mean 42/43), C4 (s42), T (seed-mean 42/43) plus the own ConvNeXt-S slot = multiview seed-mean vs control seed-mean (deployed S members replaced; HB105 'add' structure); lens threshold +0.0005 on the jev+consensus macro", 'L7': 'Gold58: same 58 exposed descriptive studies, multiview exports during fit; reuse authenticated H-B-165 control gold.csv K36 flip. Never eligible. OAI MOAKS ineligible: native448 and compartment-crop OAI caches do not exist; separate geometry/laterality-qualified data job required.'}
COMBINED_RULE = {'aggregation': 'Equal fold weights (0.5 each) of fold-specific AUC deltas; never pool probabilities across folds. Preserve each fold gate and seven-lens decision separately.', 'macro_min': 0.002, 'target_floor': -0.005, 'rule': 'COMBINED_SUPPORT only if fold-3 gate PASS, equal-fold four-ruler macro >= 0.002 and every equal-fold four-ruler target delta >= -0.005; otherwise NO_COMBINED_SUPPORT. Engineering evidence never supports. This cannot rescue a failed fold-3 gate or revise fold-2 NO_PASS.', 'uncertainty': 'No combined CI or independence claim: training complements overlap and fold 2 selected the hypothesis. Show fold-specific component CIs and all four seed deltas.', 'holistic': 'Report L1-L7 and native decision_label for BOTH folds plus equal-fold L6 point delta when available; no post-hoc gate overrides. B alone decides follow-up after independent result review.', 'not_release_authorization': True}
FOLD2_READOUT = ".local/scratch/slot-b/hb166/pilot-v100/readout.json"
FOLD2_READOUT_SHA256 = '7d067d71105e225db62075add15c109b0ebf037062d739b6379379b47496eb58'
FOLD2_REVIEW_SHA256 = '9579bc51b0cda74996bdb9ad7eef3a5013e7e95ab0999ee1df8be05f5e3fab3f'
SOURCES = ["training/arms/hb166_multiview_v1.py", "tools/hb166_fold3_build_v1.py", "tools/hb166_fold3_fit_v1.py",
           "tools/hb166_fold3_watchdog_v1.py", "tools/hb166_fold3_readout_v1.py",
           "tools/hb155_build_v1.py", "tools/hb155_fit_v1.py", "tools/hb155_watchdog_v1.py",
           "training/arms/hb155_crops_v1.py", "tools/parity_runtime.py", "config/workspace.toml",
           "tools/gpu_job.py", "tools/gpu_queue.py", "tools/job_runtime.py",
           "tools/hb166_fold3_transfer_v1.py", "tools/multi_lens_readout.py", "tools/multi_lens_readout_l4_v2.py"]


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
    base = rel(CONTROL_WORK / "fits" / control_member(seed))
    need(read(ROOT / base / "heartbeat.json").get("status") == "complete", "control incomplete")
    out = {base + "/" + n: CONTROL_FILE_PINS[base + "/" + n] for n in CONTROL_FIT_FILES}
    need(out[base + "/val.csv"] == CONTROL_PINS[seed]["val_csv"]
         and out[base + "/final.pt"] == CONTROL_PINS[seed]["checkpoint"], "control pin inconsistency")
    return out


def verify_controls():
    """Authenticate completed control bytes, receipts and recipe against the fold-2 multiview parent."""
    report = {}
    for seed in SEEDS:
        cp = b165.config_path("control", seed)
        c = b165.validate(cp)
        need(sha(cp) == CONTROL_PINS[seed]["config"], "control config drift")
        d = CONTROL_WORK / "fits" / control_member(seed)
        for p, digest in control_bindings(seed).items():
            need(sha(ROOT / p) == digest, "control file drift: " + p)
        summary = read(d / "summary.json")
        require_runtime(read(d / "runtime.json"))
        for name, ids in (("val", c["val"]), ("gold", c["gold"])):
            r165.validate_records(c, summary, read(d / (name + ".receipt.json")), ids,
                config_hash=sha(cp), csv_hash=sha(d / (name + ".csv")),
                checkpoint_hash=CONTROL_PINS[seed]["checkpoint"], prepare_hash=sha(CONTROL_WORK / "prepared.json"),
                runtime_hash=sha(d / "runtime.json"))
        parent = read(b155.WORK / "configs" / ("control-s" + str(seed-2) + ".json"))
        normalized = copy.deepcopy(c["recipe"])
        normalized["roots"] = parent["recipe"]["roots"]
        need(normalized == parent["recipe"], "recipe beyond fold population differs")
        for k in ("weights", "label_path", "label_semantic_sha256", "corpus_contract", "resolution", "epochs", "batch", "microbatch", "accumulation", "train_k", "workers", "optimizer_steps", "exclusions_sha256", "side_unknown_policy"):
            need(c[k] == parent[k], "control recipe differs: " + k)
        need(c["gold"] == gold_ids(), "Gold58 roster differs")
        report[str(seed)] = dict(status="PAIRED_REUSE", config=sha(cp), val_uid_sha256=uid_hash(c["val"]),
            train_uid_sha256=uid_hash(c["train"]), epochs=c["epochs"], steps=c["optimizer_steps"],
            k=36, flip=True, gold_rows=len(c["gold"]), runtime_sha256=sha(d / "runtime.json"))
    return report


def expected(seed, engineering=False):
    control_path = b165.config_path("control", seed)
    need(sha(control_path) == CONTROL_PINS[seed]["config"], "H-B-165 control config differs from pin")
    c = copy.deepcopy(b165.validate(control_path))  # authenticates the full control recipe/source closure
    need(c["arm"] == "control" and c["seed"] == seed and c["fold"] == 3 and not c["engineering_only"],
         "control identity differs")
    gold = gold_ids()
    need(not set(gold) & set(c["train"] + c["val"]), "Gold58 overlaps the fold-3 roster")
    train, val = c["train"], c["val"]
    if engineering:
        train, val, gold = train[:2], val[:1], gold[:1]
    inputs = {path: sha(ROOT / path) for path in source_closure()}
    inputs[rel(control_path)] = CONTROL_PINS[seed]["config"]
    inputs[rel(GOLD_IDS)] = sha(GOLD_IDS)
    for directory in (CONTROL_CORPUS, CROP_CORPUS):
        inputs[directory + "/manifest.json"] = sha(ROOT / directory / "manifest.json")
    return dict(schema="hb166-config/v1", version=VERSION, experiment=EXPERIMENT, arm="multiview", seed=seed, fold=3,
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
        estimand="paired multiview-minus-control delta on exposed fold 3 against the existing H-B-165 control fits",
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
        fold=3, seeds=list(SEEDS), members=[member(s) for s in SEEDS], configs=configs,
        controls=dict(source="existing H-B-165 fold-3 control fits (pilot-fold3-v100), not refit", members=controls,
                      file_pins=CONTROL_FILE_PINS,
                      runtime_sha256=CONTROL_RUNTIME_SHA256),
        window_budget=WINDOW_BUDGET, view_embedding=VIEW_EMBEDDING, gate=GATE, lens_plan=LENS_PLAN,
        combined_rule=COMBINED_RULE, fold2_readout=dict(path=FOLD2_READOUT, sha256=FOLD2_READOUT_SHA256),
        fold2_result_review=dict(path=rel(WORK / "fold2-result-review.md"), sha256=FOLD2_REVIEW_SHA256,
                                 verdict="PASS", reviewer="astra-4"),
        l6_partners=dict(manifest=HB105_MANIFEST, files=partners,
                         exposure="held-out fold-3 OOF of production families; confirmation is EXPOSED"),
        gold58=dict(ids_file=rel(GOLD_IDS), ids_sha256=sha(GOLD_IDS), uid_sha256=GOLD_UID_SHA256,
                    label_file=GOLD_LABELS, label_sha256=GOLD_LABELS_SHA256, status="EXPOSED_DESCRIPTIVE"),
        bias_statement=BIAS_STATEMENT, exposure="EXPOSED", clean_oof=False,
        pairing="same GPU type (RTX 4090 sm89) and identical software stack as the control fits; runtime snapshot diff "
                "vs the control runtime.json must be empty except " + ", ".join(RUNTIME_DRIFT_ALLOWED),
        recipe="identical to H-B-155 control: batch 8 / microbatch 2 / accumulation 4, workers 2, 224 px, 20 epochs, "
               "K12 train (per view), K24/K36/K36flip validation (per view), fixed final epoch, L2 labels, fp16 strict runtime",
        memory_gate_bytes=MEMORY_GATE_BYTES, not_release_authorization=True)


def register():
    need(sha(WORK / "fold2-result-review.md") == FOLD2_REVIEW_SHA256, "fold-2 review drift")
    verify_controls()
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
    verify_controls()
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
    reference = b155.WORK / "runtime" / "4090.json"
    need(sha(reference) == CONTROL_RUNTIME_SHA256, "control runtime reference differs")
    review = read(b155.WORK / "runtime-review.json")
    need(review.get("verdict") == "PASS" and review.get("independent") is True
         and review.get("bindings", {}).get(rel(reference)) == CONTROL_RUNTIME_SHA256,
         "control runtime review absent/stale")
    return check_runtime(snapshot, read(reference))


def smoke_paths():
    paths = []
    for seed in SEEDS:
        paths.append(("cpu", "fit", seed, WORK / "cpu" / member(seed) / "summary.json"))
        paths.append(("cuda", "fit", seed, WORK / "cuda" / member(seed) / "summary.json"))
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
    memory = WORK / "memory-admission.json"
    m = read(memory)
    need(m.get("schema") == "hb166-fold3-memory/v1" and m.get("device") == GPU,
         "measured memory admission required")
    need(isinstance(m.get("measured_at"), str) and bool(m["measured_at"]), "memory timestamp required")
    peak = max(read(WORK / "cuda" / member(s) / "summary.json")["peak_reserved_bytes"] for s in SEEDS)
    need(m.get("peak_reserved_bytes") == peak and 0 <= m.get("used_before_cuda_bytes", -1)
         and peak <= MEMORY_GATE_BYTES and m["used_before_cuda_bytes"] + peak + 2*(1<<30) <= m.get("total_bytes", 0),
         "measured 12 GiB + 2 GiB headroom gate")
    bindings[rel(memory)] = sha(memory)
    prior_review = WORK / "fold2-result-review.md"
    need(prior_review.is_file(), "independent fold-2 result review artifact missing")
    need(sha(prior_review) == FOLD2_REVIEW_SHA256, "independent fold-2 result review drift")
    bindings[rel(prior_review)] = sha(prior_review)
    return bindings


def require_fit(path, review=None):
    c = validate(path)
    need(not c["engineering_only"], "engineering config cannot fit")
    need(c.get("exposure") == "EXPOSED" and c.get("clean_oof") is False and c.get("fold") == 3, "exposure label differs")
    need(review is not None, "independent review required")
    r = read(review)
    need(r.get("schema") == "hb166-review/v1" and r.get("verdict") == "PASS" and r.get("independent") is True
         and bool(r.get("reviewer")) and r.get("reviewer") != "hb166-fold3-builder"
         and r.get("fold2_result_review_verdict") == "PASS" and r.get("bindings") == review_bindings(),
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
    cmd = [sys.executable, "-B", "-m", "tools.hb166_fold3_watchdog_v1", "--attempt", str(attempt),
           "--budget", str(WORK / "gpu-budget.json")]
    return cmd + (["--measurement"] if measurement else ["--healthy-rates", str(rates)])


def sequence(mode, review=None, attempt="v1"):
    """smoke: 2 multiview CUDA smokes; fit: 2 multiview fits. Existing control/Gold58 outputs reused."""
    need(mode in ("smoke", "fit") and attempt.replace("-", "").isalnum(), "invalid sequence")
    disk_gate(WORK, 30 * (1 << 30))
    if mode == "fit":
        need(admit(review)["status"] == "ADMITTED", "admission HOLD")
    jobs = []
    for seed in SEEDS:
        name = member(seed)
        out = WORK / ("cuda" if mode == "smoke" else "fits") / name
        jobs.append((name, out, ["tools.hb166_fold3_fit_v1", mode, "--config", str(config_path(seed))]))
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
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # empty, never -1 (see tools/hb166_fold3_fit_v1.hide_gpus_for_cpu)
    from tools import hb166_fold3_fit_v1 as f
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
