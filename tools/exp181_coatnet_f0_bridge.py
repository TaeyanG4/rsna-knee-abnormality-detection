"""EXP181 v2.0.0: fold-0 CoAtNet bridge on the EXP143 v2 recipe (H-B-87).

Changelog v2.0.0: retire cloned fit/guard mutation and legacy producers. New
CONTROL/AUG1 work routes through the typed CoAtNet configuration lifecycle.
Scientific helpers remain for parity and authenticated historical evidence;
old registrations never authorize a new launch.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "training/src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from tools import exp143_v2 as base
# Static module binding keeps EXP174's `base` attached to the real EXP143 v2 module.
from tools import exp174_own0949_jev as e174

VERSION, HISTORICAL_VERSION = "2.0.0", "1.0.0"
SCHEMA = "exp181-coatnet-f0-bridge-v1"
EXP = ROOT / "artifacts/experiments/EXP181_COATNET_F0_BRIDGE"
PROTOCOL_DOC = EXP / "PROTOCOL.md"
PROTOCOL_DIR = EXP / "protocol-v1"
VALIDATION = PROTOCOL_DIR / "validation.json"
QUEUE = PROTOCOL_DIR / "queue-exp181.json"
SCREEN_DIR = EXP / "screen"
BASE_PROTOCOL = ROOT / "artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json"
BASE_PROVENANCE = ROOT / "artifacts/experiments/EXP143_OWN0949/S42v2/provenance.json"
BASE_SHA = "b2aaca64289249595e532b74d361175ed78b1170109eee302ec29068e0e7af11"
INITIAL_STATE_SHA = "8064b3a56c602e5e52abd3a10b77f235300dd7008f56fc4d7d47fca75e5d3c43"
FOLD_MANIFEST = ROOT / "artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/manifest.json"
FOLD_MANIFEST_SHA = "707205dbf69ffcaf1c8fdd3df84aa90a2cd63464408d4cfa3a1bdce6e8aee53d"
FOLD0_IDS = ROOT / "artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold0/ordered-ids.json"
FOLD0_SHA = "0a1b2f3d67ba9588855d671cc7119d168fbfb8d0b4499357ad68249cbcc2f473"
INIT = ROOT / "storage/weights/exp143-imagenet-init/coatnet384-imagenet.pt"
CORPUS = base.CORPUS
TORCH_VERSION = "2.5.1+cu121"
GPU_NAME = "4070 Ti SUPER"
SEED, PRECISION = 42, "bf16"
N_TRAIN, N_VAL, EPOCHS, UPDATES = 3455, 888, 16, 3456
BUDGETS = dict(local=4.0, reserve_seconds=600)
OOF_MAX_SECONDS = 1800
OOF_PLAIN, OOF_PRIMARY = "oof_k94.csv", "oof_k94_mirror_tta.csv"
OOF_RECEIPT, INVENTORY = "oof-receipt.json", "output-inventory.json"
ARM_ORDER = ("CONTROL", "AUG1")
CODE = list(dict.fromkeys([*base.CODE, base.PROTOCOL, "tools/exp174_own0949_jev.py",
    "tools/exp181_coatnet_f0_bridge.py", "training/src/exp181_aug1.py",
    "tools/gpu_queue.py", "tools/gpu_job.py",
    "artifacts/experiments/EXP181_COATNET_F0_BRIDGE/PROTOCOL.md"]))
SCREEN_CODE = ["tools/exp179_screen.py", "tools/exp170_convnext_t_l2.py", "tools/exp115_vlm_eval.py",
               "tools/exp156_k_sweep.py", "tools/exp149_label_sources.py",
               "tools/exp162_dino_folds_kaggle.py"]
SCREEN = dict(primary=OOF_PRIMARY, diagnostic=OOF_PLAIN, fold=0, studies=N_VAL,
              references=["jev", "aquila", "dread", "consensus"], primary_references=["jev", "consensus"],
              reference_loader="tools.exp179_screen.load_context -> tools.exp156_k_sweep.references",
              bootstrap="validation components, 300 draws, seed 20261003, shared by both arms",
              gate=dict(primary_mean_min=0.002, primary_ci95_lower_gt=0.0, dread_min=-0.003,
                        all_primary_point_targets_defined=True),
              aquila="L2 training self-reference; report only",
              ruler_advisory="exp179_screen._ruler_flags caution list; REVIEW never auto-promotes")
TARGETS = base.TARGETS
sha256, json_hash, read_json = base.sha256, base.json_hash, base.read_json
save_torch_new, save_torch_atomic = base.save_torch_new, base.save_torch_atomic


def _aug1_spec():
    from exp181_aug1 import AUG1_SPEC
    return copy.deepcopy(AUG1_SPEC)


def arm_spec(arm):
    if arm == "CONTROL":
        return dict(dataset="exp143_own0949.StudyDataset (frozen)", augmentation=None)
    if arm == "AUG1":
        return dict(dataset="exp181_aug1.Aug1StudyDataset", augmentation=_aug1_spec())
    raise ValueError("unregistered EXP181 arm: " + str(arm))


def manifest_path(arm):
    arm_spec(arm)
    return PROTOCOL_DIR / f"manifest-{arm}.json"


def run_rel(arm):
    arm_spec(arm)
    return f"artifacts/experiments/EXP181_COATNET_F0_BRIDGE/{arm}"


def train_tag(arm):
    return f"EXP181_{arm}_F0"


def oof_tag(arm):
    return f"EXP181_{arm}_F0_OOF"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def uid_hash(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def scoped_path(path, root=None):
    path, root = Path(path).resolve(), Path(root or EXP).resolve()
    if root not in path.parents:
        raise ValueError("output must be a child of the EXP181 subtree")
    return path


def write_new_json(path, value):
    path = scoped_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_bindings(names=CODE):
    return {name: sha256(ROOT / name) for name in names}


def baseline_registration(corpus=None, init_weights=None):
    """Verified EXP143 v2 S42v2 control contract (EXP174's frozen checker)."""
    if e174.BASE_SHA != BASE_SHA or e174.INITIAL_STATE_SHA != INITIAL_STATE_SHA:
        raise ValueError("EXP174 frozen control pins changed")
    return e174.baseline_registration(corpus, init_weights)


def fold0_roster():
    if sha256(FOLD_MANIFEST) != FOLD_MANIFEST_SHA or sha256(FOLD0_IDS) != FOLD0_SHA:
        raise ValueError("EXP099 fold protocol changed")
    m = read_json(FOLD_MANIFEST)["folds"]["0"]
    o = read_json(FOLD0_IDS)
    if (Path(m["ordered_ids"]["path"]) != FOLD0_IDS.relative_to(ROOT)
            or m["ordered_ids"]["sha256"] != FOLD0_SHA
            or uid_hash(o["train"]) != m["ordered_ids"]["train_sha256"]
            or uid_hash(o["val"]) != m["ordered_ids"]["val_sha256"]
            or not len(o["train"]) == m["n_train"] == N_TRAIN
            or not len(o["val"]) == m["n_val"] == N_VAL
            or len(set(o["train"])) != N_TRAIN or len(set(o["val"])) != N_VAL
            or set(o["train"]) & set(o["val"])):
        raise ValueError("EXP099 fold-0 roster differs from its manifest")
    return o["train"], o["val"]


def build_common(baseline):
    """Arm-independent bridge contract; labels come only from the pinned control manifest."""
    train0, val0 = fold0_roster()
    universe = set(baseline["train_ids"])
    if not set(train0) <= universe or not set(val0) <= universe:
        raise ValueError("fold-0 studies outside the EXP143 v2 non-Gold cohort")
    if (set(train0) | set(val0)) & set(baseline["excluded_gold_ids"]):
        raise ValueError("Gold58 leaked into the bridge")
    keep = set(train0)
    ids = [u for u in baseline["train_ids"] if u in keep]
    excluded = [u for u in baseline["train_ids"] if u not in keep and u not in set(val0)]
    if len(ids) != N_TRAIN or set(ids) != keep or len(excluded) != len(universe) - N_TRAIN - N_VAL:
        raise ValueError("bridge training roster differs")
    labels = {u: list(baseline["train_labels"][u]) for u in ids}
    common = dict(schema=SCHEMA, version=HISTORICAL_VERSION, status="REGISTERED_NO_TRAINING",
        recipe=copy.deepcopy(baseline["recipe"]), seed=SEED, precision=PRECISION,
        grad_checkpointing=False, world_size=1, budgets=dict(BUDGETS), oof_max_seconds=OOF_MAX_SECONDS,
        train_ids=ids, val_ids=list(val0), excluded_gold_ids=list(baseline["excluded_gold_ids"]),
        excluded_non_fold_ids=excluded, train_labels=labels,
        pos_weight=base.pos_weights(labels, ids),
        label_sha256=baseline["label_sha256"], label_file_name=baseline["label_file_name"],
        label_provenance=copy.deepcopy(baseline["label_provenance"]),
        label_provenance_sha256=baseline["label_provenance_sha256"],
        init_sha256=baseline["init_sha256"], init_provenance=copy.deepcopy(baseline["init_provenance"]),
        corpus_manifest_sha256=baseline["corpus_manifest_sha256"],
        corpus_build_sha256=baseline["corpus_build_sha256"],
        fold=dict(source="EXP099_KFOLD_PROTOCOL_V1", fold=0, manifest_sha256=FOLD_MANIFEST_SHA,
                  ordered_ids=FOLD0_IDS.relative_to(ROOT).as_posix(), ordered_ids_sha256=FOLD0_SHA,
                  train_sha256=uid_hash(train0), val_sha256=uid_hash(val0),
                  train_order="EXP143 v2 registered order restricted to EXP099 fold-0 train (folds 1-4)",
                  bridge_train_sha256=uid_hash(ids), n_train=N_TRAIN, n_val=N_VAL),
        base_protocol_sha256=BASE_SHA, base_provenance_sha256=sha256(BASE_PROVENANCE),
        expected_initial_state_sha256=INITIAL_STATE_SHA,
        evaluation=dict(k_eval=baseline["recipe"]["eval_k"], inference_res=baseline["recipe"]["inference_res"],
                        tta="frozen infer_volume mirror (sagittal channel reverse, coronal/axial horizontal)",
                        precision="fp16", primary=OOF_PRIMARY, diagnostic=OOF_PLAIN),
        screen=copy.deepcopy(SCREEN))
    if common["recipe"] != dict(base.RECIPE, labels="registered-replacement-soft"):
        raise ValueError("control recipe changed")
    return common


ARM_FIELDS = {"arm", "dataset", "augmentation", "run_dirs", "created", "source_bindings",
              "screen_source_bindings", "bridge_contract_sha256", "content_sha256"}


def registration(common, arm, created=None):
    p = copy.deepcopy(common)
    p.update(arm=arm, **arm_spec(arm), run_dirs={"local": run_rel(arm)},
             created=created or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             source_bindings=source_bindings(), screen_source_bindings=source_bindings(SCREEN_CODE),
             bridge_contract_sha256=json_hash(common))
    p["content_sha256"] = json_hash(p)
    return p


def execution_settings(p, world, precision, seed, checkpointing=None):
    if (world != 1 or precision != PRECISION or seed != SEED
            or (checkpointing is not None and checkpointing is not False)):
        raise ValueError("EXP181 requires local BF16 seed42, checkpointing OFF")
    settings = base.execution_settings(p, world, precision, seed, False)
    if (settings["updates_per_epoch"] * EPOCHS != UPDATES or settings["total_steps"] != UPDATES
            or settings["accumulation"] != 16 or settings["batch"] != 1):
        raise ValueError("EXP181 update schedule differs from registration")
    return settings


def registered_run(p, lane="local"):
    if lane != "local" or p.get("run_dirs") != {"local": run_rel(p.get("arm"))}:
        raise ValueError("EXP181 permits only its registered fresh local arm run")
    return scoped_path(ROOT / p["run_dirs"]["local"])




def load_registration(protocol, corpus=None, init_weights=None):
    """Authenticate immutable EXP181 evidence; no legacy execution qualification."""
    from training.arms import coatnet_history
    p = coatnet_history.read_registration(protocol)
    need(p["arm"] in ARM_ORDER and Path(protocol).resolve() == manifest_path(p["arm"]).resolve(),
         "EXP181 historical manifest path differs")
    if corpus is not None:
        need(sha256(Path(corpus) / "manifest.json") == p["corpus_manifest_sha256"], "corpus changed")
    if init_weights is not None:
        need(sha256(init_weights) == p["init_sha256"], "initial weights changed")
    return p


def require_validation(protocol):
    from training.arms import coatnet_history
    load_registration(protocol)
    return coatnet_history.require_validation(protocol)


def aug1_real_checks(corpus, ids):
    """CPU, aggregate-only: frozen equivalence, determinism and range on real studies."""
    import numpy as np
    import exp143_own0949 as frozen
    import exp181_aug1 as aug
    c = frozen.Corpus(corpus)
    checks = []
    for uid in ids:
        vol, mask = c.get(uid)
        for epoch in (0, 15):
            x01, centers, _ = aug.train_windows01(vol, mask, SEED, epoch, uid)
            if not np.array_equal(aug.normalize(x01), frozen.windows(vol, mask, True, SEED, epoch, uid)):
                raise ValueError("AUG1 pre-augmentation path differs from frozen windows")
            a, b = aug.aug1_windows(vol, mask, SEED, epoch, uid), aug.aug1_windows(vol, mask, SEED, epoch, uid)
            if not np.array_equal(a, b) or a.shape != (12, 3, 384, 384) or a.dtype != np.float32:
                raise ValueError("AUG1 is not deterministic / wrong shape")
            if np.array_equal(a, aug.normalize(x01)):
                raise ValueError("AUG1 changed nothing")
            checks.append(dict(epoch=epoch, centers=len(centers)))
    return len(checks)










def arm_dataset(arm):
    if arm == "CONTROL":
        return None
    if arm == "AUG1":
        from exp181_aug1 import Aug1StudyDataset
        return Aug1StudyDataset
    raise ValueError("unregistered arm")




def require_gpu_job(tag):
    path = ROOT / "artifacts/logs" / tag / "job.json"
    deadline = time.monotonic() + 30
    while True:
        try:
            job = read_json(path)
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise ValueError("gpu_job supervision receipt missing") from None
            time.sleep(.25)
    if os.getppid() != job.get("pid") or job.get("tag") != tag:
        raise ValueError("explicit GPU work must run through gpu_queue/gpu_job")


def require_local_device():
    import torch
    if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or str(torch.__version__) != TORCH_VERSION
            or GPU_NAME not in torch.cuda.get_device_name(0)):
        raise ValueError("EXP181 must run on the local 4070 Ti SUPER / torch 2.5.1 stack")
    return torch.cuda.get_device_name(0)




def check_summary(summary, p, protocol_sha, model_sha):
    identity = summary.get("identity", {})
    if (summary.get("status") != "COMPLETE_TRAINING_NO_EVALUATION"
            or summary.get("epochs") != EPOCHS or summary.get("updates") != UPDATES
            or summary.get("skips") != 0 or len(summary.get("history", [])) != EPOCHS
            or [h.get("epoch") for h in summary["history"]] != list(range(1, EPOCHS + 1))
            or summary.get("model_sha256") != model_sha
            or identity.get("protocol_sha256") != protocol_sha
            or identity.get("initial_state_sha256") != INITIAL_STATE_SHA
            or identity.get("seed") != SEED or identity.get("train_n") != N_TRAIN
            or identity.get("run_dir") != p["run_dirs"]["local"]
            or identity.get("execution") != execution_settings(p, 1, PRECISION, SEED, False)):
        raise ValueError("EXP181 fit incomplete or identity changed")
    return summary


def check_checkpoint(ck, p, protocol_sha):
    r = p["recipe"]
    if (ck.get("arch") != r["arch"] or ck.get("contract") != "exp143_raptor96_384_c140_s02"
            or ck.get("oai") is not False or ck.get("swa_epochs_1based") != r["swa_1based"]
            or ck.get("lab") != TARGETS or ck.get("checkpoint_kind") != "fixed-swa-bn"
            or ck.get("seed") != SEED or ck.get("provenance", {}).get("protocol_sha256") != protocol_sha
            or ck["provenance"].get("run_dir") != p["run_dirs"]["local"]):
        raise ValueError("checkpoint differs from the registered EXP181 arm")


def infer_pair(model, vol, mask, device="cpu", precision="none", chunk=16):
    """Frozen infer_volume, returning (plain, mirror); served = .5 * (plain + mirror)."""
    import torch
    from exp143_own0949 import amp_context, windows
    if any(m.training for m in model.modules()):
        raise ValueError("inference requires eval mode")
    probs = []
    with torch.inference_mode():
        for mirror in (False, True):
            x = torch.from_numpy(windows(vol, mask, mirror=mirror))
            with amp_context(device, precision):
                feats = torch.cat([model.backbone(z.to(device)) for z in x.split(chunk)], 0)
                logits = model.head(feats.unsqueeze(0)).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite prediction; no fallback")
            probs.append(logits.sigmoid()[0].cpu().numpy())
    return probs[0], probs[1]


def inventory(run, names):
    return {n: dict(sha256=sha256(run / n), bytes=(run / n).stat().st_size) for n in names}




def load_oof(path, ids):
    import numpy as np
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [base.UID, *TARGETS], "OOF columns differ: " + str(path))
        rows = list(reader)
    by_id = {r[base.UID]: r for r in rows}
    need(len(by_id) == len(rows) == len(ids) and set(by_id) == set(ids), "OOF IDs differ: " + str(path))
    values = np.asarray([[float(by_id[u][t]) for t in TARGETS] for u in ids], np.float64)
    need(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "invalid OOF: " + str(path))
    return values


def verified_arm(arm):
    from training.arms import coatnet_history
    return coatnet_history.verified_arm(manifest_path(arm))


def admission(pair):
    dread = pair["references"]["dread"]["delta"]
    complete = all(pair["references"][n]["per_target_delta"][t] is not None
                   for n in ("jev", "consensus") for t in TARGETS)
    g = SCREEN["gate"]
    checks = {"primary_mean_at_least_0_002": pair["primary_mean_delta"] >= g["primary_mean_min"],
              "primary_ci_lower_positive": pair["primary_mean_ci95"][0] > g["primary_ci95_lower_gt"],
              "dread_at_least_minus_0_003": dread is not None and dread >= g["dread_min"],
              "all_primary_point_targets_defined": complete}
    return {"status": "ADVANCE" if all(checks.values()) else "HOLD", **checks,
            "rule": "mean(JEV,consensus) delta >= +0.002; paired CI95 lower > 0; Dread delta >= -0.003"}


def score(candidate, control, refs, draws, evaluator):
    from tools.exp170_convnext_t_l2 import paired_jev_consensus
    from tools.exp179_screen import _ruler_flags
    primary = paired_jev_consensus(candidate["primary"], control["primary"], refs, draws, evaluator)
    plain = paired_jev_consensus(candidate["plain"], control["plain"], refs, draws, evaluator)
    gate = admission(primary)
    advisory = _ruler_flags(primary, None)
    return dict(status="REVIEW" if advisory["status"] == "REVIEW" else gate["status"],
                registered_gate=gate, ruler_advisory=advisory, primary_k94_mirror_tta=primary,
                diagnostic_k94_plain=plain)








def dispatch(argv=None):
    """Thin current lifecycle entry point with explicit historical readouts."""
    argv = list(sys.argv[1:] if argv is None else argv)
    arm_parser = argparse.ArgumentParser(add_help=False)
    arm_parser.add_argument("--preset-arm", choices=ARM_ORDER, default="CONTROL")
    selected, remaining = arm_parser.parse_known_args(argv)
    if remaining and remaining[0] in ("collect", "screen"):
        from training.arms import coatnet_screen
        ap = argparse.ArgumentParser(description="Read-only authenticated EXP181 historical pair")
        ap.add_argument("command", choices=("collect", "screen"))
        ap.add_argument("--out", required=True)
        ap.add_argument("--verification-note", required=True)
        ap.add_argument("--collection")
        ap.add_argument("--readout")
        args = ap.parse_args(remaining)
        need(selected.preset_arm != "CONTROL", "historical candidate must differ from CONTROL")
        ctrl, cand = verified_arm("CONTROL"), verified_arm(selected.preset_arm)
        if args.command == "collect":
            return coatnet_screen.collect_records(ctrl, cand, candidate=selected.preset_arm,
                out=args.out, verification_note=args.verification_note)
        need(args.collection, "historical screen requires --collection")
        return coatnet_screen.screen_records(ctrl, cand, candidate=selected.preset_arm,
            collection=args.collection, out=args.out, verification_note=args.verification_note,
            readout=args.readout)
    from tools.coatnet_config import main as config_main
    if not remaining or remaining[0] != "lifecycle":
        raise ValueError("legacy EXP181 producers retired; use lifecycle <command> --preset-arm CONTROL|AUG1")
    return config_main(remaining[1:], preset=selected.preset_arm.lower() + "-f0-s42")


def main(argv=None):
    try:
        result = dispatch(argv)
        if isinstance(result, dict):
            print(json.dumps(dict(status=result["status"], launch_authorized=False)))
            return 0
        return result
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps(dict(status="REJECTED", launch_authorized=False, reason=str(error))))
        return 2


if __name__ == "__main__":
    sys.exit(main())
