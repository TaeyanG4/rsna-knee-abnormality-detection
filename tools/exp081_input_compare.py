"""One frozen EXP081 baseline/candidate comparison; registration is CPU-only.

The run command performs exactly two full F2 inference passes, gates their
identity/finite/runtime receipts, and only then opens the fixed weak labels.
It never imports training/acquisition scripts, reads Gold, or launches jobs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp081_coronal_span as prep

CPU = ROOT / "artifacts/experiments/EXP081_CORONAL_SPAN_FIXED_F2"
PROTOCOL = ROOT / "artifacts/experiments/EXP081_CORONAL_SPAN_GPU_PROTOCOL/launch-contract.json"
OUTPUT = ROOT / "artifacts/predictions/EXP081_CORONAL_SPAN_FIXED_F2"
META = ROOT / "storage/labels/derived_meta/study_meta.csv"
SCHEMA = "exp081-two-pass-contract-v1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ENV_KEYS = ("python", "torch", "cuda", "cudnn", "gpu", "driver", "timm", "numpy", "pydicom", "cv2")
REFERENCE_SHA = "8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1"
CONSTANTS = dict(studies=868, changed=34, unchanged=834, batch=8, k_eval=24, res=224,
                 amp="fp16", policy="strict", device="cuda", norm="imagenet",
                 window_policy="within-slot-valid", bootstrap_reps=500, bootstrap_seed=20261003)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unuri(value):
    if not isinstance(value, str) or not value.startswith("workspace://"):
        raise ValueError("expected workspace URI")
    relative = Path(value.removeprefix("workspace://"))
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("workspace URI escapes root")
    return ROOT / relative


def adapters():
    sys.path.insert(0, str(ROOT / "training/src"))
    import raptor_common
    import raptor_runtime
    return raptor_common, raptor_runtime


def validate_bindings(bindings):
    for name, expected in bindings.items():
        if prep.sha(unuri(name)) != expected:
            raise ValueError("bound input/source changed: " + name)


def collect_cpu():
    """Read receipts and immutable CPU inputs; no labels or model deserialization."""
    registration, receipt = read(CPU / "registration.json"), read(CPU / "receipt.json")
    if (receipt.get("status") != "CPU_PREPARATION_COMPLETE_GPU_NOT_RUN" or
        receipt.get("changed") != 34 or receipt.get("unchanged") != 834 or
        receipt.get("no_op_controls") != 63 or not receipt.get("all_no_op_equal") or
        receipt.get("labels_opened") is not False or not receipt.get("source_bindings_rechecked")):
        raise ValueError("CPU preparation receipt contract mismatch")
    if prep.sha(ROOT / "tools/exp081_coronal_span.py") != registration["preparation_tool_sha256"]:
        raise ValueError("CPU implementation changed since preparation")
    bindings = dict(registration["source_bindings"])
    bindings.update(receipt["artifacts"])
    for path in (CPU / "receipt.json", ROOT / "tools/exp081_coronal_span.py", META):
        bindings[prep.uri(path)] = prep.sha(path)
    validate_bindings(bindings)
    cohort, selection = read(CPU / "cohort.json"), read(CPU / "selection-map.json")
    ids = cohort["ids"]
    if (len(ids) != 868 or ids != sorted(set(ids)) or prep.uid_digest(ids) != prep.EXPECTED_IDS or
        cohort.get("label_reads") != 0 or selection.get("pixel_reads_at_freeze") != 0 or
        prep.sha(CPU / "selection-map.json") != receipt["selection_map_sha256"]):
        raise ValueError("CPU UID/selection freeze mismatch")
    plans = selection["plans"]
    if [row["uid"] for row in plans] != ids:
        raise ValueError("CPU plan order mismatch")
    changed_ids = [row["uid"] for row in plans if row["changed"]]
    if len(changed_ids) != 34 or read(CPU / "overlay/ids.json") != changed_ids:
        raise ValueError("CPU changed UID/patch order mismatch")
    noops, checks = read(CPU / "noop-replay.json"), read(CPU / "overlay-receipts.json")
    if len(noops["controls"]) != 63 or not all(row.get("equal") is True for row in noops["controls"]):
        raise ValueError("CPU original-slot no-op controls failed")
    if ([row["uid"] for row in checks["changed"]] != changed_ids or
        [row["uid"] for row in checks["unchanged"]] != [uid for uid in ids if uid not in set(changed_ids)]):
        raise ValueError("CPU byte-check population mismatch")
    return registration, receipt, ids, changed_ids, checks, bindings


def load_corpora(ids, changed_ids, checks, *, verify_bytes=True):
    common, _ = adapters()
    baseline = common.Corpus(ROOT / prep.CORPUS)
    baseline.masks.setflags(write=False)
    patches = np.load(CPU / "overlay/patches.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(CPU / "overlay/masks.npy", mmap_mode="r", allow_pickle=False)
    candidate = prep.OverlayCorpus(baseline, changed_ids, patches, masks)
    if candidate.contract == baseline.contract or candidate.contract != prep.TRANSFORM:
        raise ValueError("changed inputs must have an explicit new transform contract")
    if verify_bytes:
        for row in checks["changed"]:
            uid = row["uid"]
            original, mask = baseline.volume(uid), baseline.mask(uid)
            altered, altered_mask = candidate.volume(uid), candidate.mask(uid)
            prep.verify_overlay(original, mask, altered, altered_mask)
            if (prep.array_sha(original) != row["base_volume_sha256"] or
                prep.array_sha(mask) != row["base_mask_sha256"] or
                prep.array_sha(altered) != row["candidate_volume_sha256"]):
                raise ValueError("changed study bytes differ from CPU receipt: " + uid)
            if (common.select_centers(mask, 64, 24, baseline.spec, "within-slot-valid") != row["centers_before"] or
                common.select_centers(altered_mask, 64, 24, baseline.spec, "within-slot-valid") != row["centers_after"]):
                raise ValueError("frozen center lists differ: " + uid)
        for row in checks["unchanged"]:
            uid = row["uid"]
            volume, mask = baseline.volume(uid), baseline.mask(uid)
            if (prep.array_sha(volume) != row["volume_sha256"] or prep.array_sha(mask) != row["mask_sha256"] or
                not np.shares_memory(volume, candidate.volume(uid)) or
                not np.shares_memory(mask, candidate.mask(uid))):
                raise ValueError("untouched study bytes differ from CPU receipt: " + uid)
    return baseline, candidate


def read_metadata(ids, baseline):
    with (ROOT / prep.FOLDS).open(encoding="utf-8-sig", newline="") as stream:
        fold_rows = {row[prep.UID]: row for row in csv.DictReader(stream)}
    for uid in ids:
        row = fold_rows[uid]
        if row["fold"] != "2" or prep.flag(row["is_gold"]) != 0 or prep.flag(row["quarantined_gold_report_duplicate"]) != 0:
            raise ValueError("F2 includes unexpected Gold/quarantine/split flag")
    report_groups = [fold_rows[uid]["report_group_sha256"] for uid in ids]
    if len(set(report_groups)) != 840 or any(not value for value in report_groups):
        raise ValueError("report group population mismatch")
    metadata = {}
    with META.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            uid = row[prep.UID]
            if uid in metadata:
                raise ValueError("duplicate metadata UID")
            metadata[uid] = row
    groups = {key: [metadata[uid].get(key) or "unknown" for uid in ids]
              for key in ("scanner_group", "field", "manufacturer")}
    groups["cached_mask"] = ["some_zero_mask" if (baseline.mask(uid) == 0).any() else "all_nonzero" for uid in ids]
    return report_groups, groups


def register(path):
    path = Path(path).absolute()
    if path.exists() or not path.is_relative_to(ROOT / "artifacts/experiments"):
        raise ValueError("registration requires a new contract file under artifacts/experiments")
    registration, receipt, ids, changed_ids, checks, bindings = collect_cpu()
    baseline, _ = load_corpora(ids, changed_ids, checks)
    report_groups, diagnostic_groups = read_metadata(ids, baseline)
    sources = ("tools/exp081_input_compare.py", "tools/exp081_metrics.py", "tools/oof_error_audit.py",
               "training/src/raptor_common.py", "training/src/raptor_runtime.py")
    for relative in sources:
        bindings["workspace://" + relative] = prep.sha(ROOT / relative)
    contract = dict(schema=SCHEMA, created_at=prep.now(), experiment=prep.EXPERIMENT,
        status="REGISTERED_NOT_RUN_ROOT_LAUNCH_REQUIRED", constants=CONSTANTS, targets=LAB,
        authorized_decision_at="2026-10-03T19:28:00+09:00",
        authorization_scope="Root decided one local fixed-weight two-pass GPU comparison under existing user authorization. Prior CPU registration records its historical CPU-only authorization and remains unchanged. No remote job/newfit/submission authorized here.",
        cpu_directory=prep.uri(CPU), output=prep.uri(OUTPUT),
        checkpoint=f"workspace://{prep.BASE}/model_best.pt", checkpoint_sha256=registration["checkpoint_sha256"],
        bindings=bindings, ids=ids, ids_sha256=prep.EXPECTED_IDS, changed_ids=changed_ids,
        report_groups=report_groups, diagnostic_groups=diagnostic_groups,
        checkpoint_expected=dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", res=224, lab=LAB,
             contract="raptor64_336_c140_s02", k_eval=24, norm="imagenet", window_policy="within-slot-valid",
             epoch=7, tag="EXP080_INPUT_B_F2", amp_train="fp16", amp_eval="fp16"),
        checkpoint_provenance=read(ROOT / prep.BASE / "provenance.json"),
        expected_runtime=registration["baseline_runtime"],
        expected_environment={key: registration["baseline_environment"][key] for key in ENV_KEYS},
        reference_labels=registration["approved_protocol"]["evaluation_plan"]["reference_labels"],
        unchanged_prediction_gate=dict(atol=0, rtol=0, byte_exact=True, failure_action="stop before labels; no tolerance relaxation",
             justification="Same GPU/runtime/model.eval weights, identical batch shapes/order/positions, no cross-study mixing; dropout disabled and BN uses saved statistics. Identical normalized input hashes required. Numerical portability is not inferred across devices."),
        before_labels_requirements=["both full868 passes complete", "finite probabilities in[0,1]", "checkpoint/model/source/input receipts unchanged",
             "runtime matches registered local baseline and is unchanged between passes", "834 unchanged normalized tensors and predictions byte-exact"],
        mask_mechanism=[{key: row[key] for key in ("uid", "mask_before", "mask_after", "centers_before", "centers_after")}
             for row in checks["changed"] if row["mask_before"] != row["mask_after"] or row["centers_before"] != row["centers_after"]],
        limitations=["Hypothesis selected after EXP080 F2 results; exposed development fold, not untouched clinical/private evidence",
             "Original training OOF batch4 is not an adoption or parity gate; both registered passes use batch8",
             "No minimum positive AUC/target-all-positive gate; inspect magnitude, paired conditional CI and all12 losses",
             "Subgroup diagnostics are descriptive; no routing, ratio/target threshold search or newfit",
             "This local comparison does not establish T4/4070 numerical parity"],
        cpu_receipt_sha256=prep.sha(CPU / "receipt.json"), selection_map_sha256=receipt["selection_map_sha256"],
        source_preflight="No model loaded, GPU used, label file opened or output prediction produced during registration.")
    if contract["reference_labels"]["file_sha256"] != REFERENCE_SHA:
        raise ValueError("fixed reference label file hash changed")
    validate_bindings(bindings)
    path.parent.mkdir(parents=True, exist_ok=True)
    prep.dump(path, contract)
    print(json.dumps(dict(status="REGISTERED_CPU_ONLY", contract=prep.uri(path), sha256=prep.sha(path))), flush=True)
    return contract


def validate_contract(contract):
    if (contract.get("schema") != SCHEMA or contract.get("constants") != CONSTANTS or contract.get("targets") != LAB or
        contract.get("ids_sha256") != prep.EXPECTED_IDS or len(contract.get("ids", [])) != 868 or
        contract["ids"] != sorted(set(contract["ids"])) or prep.uid_digest(contract["ids"]) != prep.EXPECTED_IDS or
        len(contract.get("changed_ids", [])) != 34 or len(set(contract["changed_ids"])) != 34 or
        not set(contract["changed_ids"]) <= set(contract["ids"])):
        raise ValueError("changed frozen GPU comparison contract")
    gate = contract.get("unchanged_prediction_gate", {})
    if gate.get("atol") != 0 or gate.get("rtol") != 0 or gate.get("byte_exact") is not True:
        raise ValueError("unchanged predictions require preregistered byte-exact equality")
    if contract["reference_labels"]["file_sha256"] != REFERENCE_SHA:
        raise ValueError("changed reference label source")
    if (contract.get("cpu_directory") != prep.uri(CPU) or contract.get("output") != prep.uri(OUTPUT) or
        contract.get("checkpoint") != f"workspace://{prep.BASE}/model_best.pt"):
        raise ValueError("changed fixed experiment input/output paths")


def validate_checkpoint(meta, contract):
    if any(meta.get(key) != value for key, value in contract["checkpoint_expected"].items()):
        raise ValueError("checkpoint architecture/input/fixed-final epoch metadata mismatch")
    if meta.get("provenance") != contract["checkpoint_provenance"]:
        raise ValueError("checkpoint provenance mismatch")


def state_digest(model):
    """Hash every parameter/buffer and reject nonfinite weights without logging values."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("nonfinite model state: " + name)
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_runtime(actual, expected, environment, expected_environment):
    if actual != expected:
        raise ValueError("runtime receipt differs from registered same-device strictFP16 baseline")
    if {key: environment.get(key) for key in ENV_KEYS} != expected_environment:
        raise ValueError("GPU/library/driver environment differs from registered local baseline")


def validate_prediction_pair(ids, changed_ids, baseline, candidate, baseline_timing, candidate_timing):
    for values in (baseline, candidate):
        if values.shape != (len(ids), 12) or values.dtype != np.float32 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("invalid full-cohort float32 probabilities")
    changed_set = set(changed_ids)
    unchanged = np.array([uid not in changed_set for uid in ids])
    for timing in (baseline_timing, candidate_timing):
        if (timing.get("all_finite") is not True or timing.get("fp32_fallbacks") != 0 or
            timing.get("batch_size") != 8 or timing.get("window_policy") != "within-slot-valid" or
            timing.get("norm") != "imagenet" or timing.get("input_hash_scheme") != "normalized-float32-shape-v1" or
            [row["uid"] for row in timing.get("input_tensor_hashes", [])] != ids):
            raise ValueError("incomplete or incompatible inference timing/input receipt")
    ah, bh = baseline_timing["input_tensor_hashes"], candidate_timing["input_tensor_hashes"]
    changed_inputs = []
    for uid, a, b in zip(ids, ah, bh):
        if uid not in changed_set and a["sha256"] != b["sha256"]:
            raise ValueError("unchanged UID normalized tensor differs: " + uid)
        if a["sha256"] != b["sha256"]:
            changed_inputs.append(uid)
    if baseline[unchanged].tobytes() != candidate[unchanged].tobytes():
        difference = float(np.max(np.abs(baseline[unchanged] - candidate[unchanged])))
        raise ValueError(f"unchanged predictions not byte-exact (max_abs={difference}); stop before labels")
    return dict(passed=True, finite=True, unchanged=int(unchanged.sum()), unchanged_byte_exact=True,
                unchanged_max_abs=0.0, tensor_changed_ids=changed_inputs,
                tensor_changed_count=len(changed_inputs), label_access_authorized_after_gate=True)


def read_reference_labels(contract, gate):
    """Called only after both passes and all identity/finite checks pass."""
    if gate.get("passed") is not True or gate.get("label_access_authorized_after_gate") is not True:
        raise ValueError("labels remain sealed until completed pair passes its gate")
    reference = contract["reference_labels"]
    path = unuri(reference["path"])
    if prep.sha(path) != reference["file_sha256"]:
        raise ValueError("reference label file changed")
    wanted, values = set(contract["ids"]), {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = [name if name in reader.fieldnames else name + "_soft" for name in LAB]
        if any(column not in reader.fieldnames for column in columns):
            raise ValueError("reference target columns missing")
        for row in reader:
            uid = row[prep.UID]
            if uid not in wanted:
                continue
            if uid in values:
                raise ValueError("duplicate reference F2 UID")
            values[uid] = [float(row[column]) for column in columns]
    if set(values) != wanted:
        raise ValueError("reference labels do not cover exact F2 population")
    y = np.asarray([values[uid] for uid in contract["ids"]], dtype=np.float32)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("invalid fixed F2 reference labels")
    return y


def run(path, expected_hash):
    started = time.perf_counter()
    if prep.sha(path) != expected_hash:
        raise ValueError("launch contract SHA differs from explicit command")
    contract = read(path); validate_contract(contract); validate_bindings(contract["bindings"])
    output = unuri(contract["output"])
    output.mkdir(parents=True, exist_ok=False)
    progress = dict(stage="input_preflight", pid=os.getpid(), started_at=prep.now(), labels_opened=False,
                    contract_sha256=expected_hash)
    prep.dump(output / "progress.json", progress)
    try:
        registration, receipt, ids, changed_ids, checks, cpu_bindings = collect_cpu()
        if ids != contract["ids"] or changed_ids != contract["changed_ids"]:
            raise ValueError("launch/CPU population mismatch")
        baseline, candidate = load_corpora(ids, changed_ids, checks)
        groups, diagnostics = read_metadata(ids, baseline)
        if groups != contract["report_groups"] or diagnostics != contract["diagnostic_groups"]:
            raise ValueError("frozen report/diagnostic group mismatch")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        common, runtime = adapters()
        actual_runtime = runtime.configure_runtime("fp16", "cuda", "strict")
        actual_environment = common.environment()
        validate_runtime(actual_runtime, contract["expected_runtime"], actual_environment, contract["expected_environment"])
        checkpoint = unuri(contract["checkpoint"])
        if prep.sha(checkpoint) != contract["checkpoint_sha256"]:
            raise ValueError("fixed checkpoint changed")
        model, meta = common.load_checkpoint(checkpoint)
        validate_checkpoint(meta, contract)
        del meta  # Embedded historical metrics are neither inspected nor reported.
        model.eval()
        initial_state = state_digest(model)
        prep.dump(output / "runtime-before.json", dict(runtime=actual_runtime, environment=actual_environment,
            model_state_sha256=initial_state, checkpoint_sha256=contract["checkpoint_sha256"],
            explicit_transform=dict(weights_contract=baseline.contract, candidate_contract=candidate.contract,
                                    change="Only frozen slot2 patches; intentional registered input intervention")))
        arrays, timings = {}, {}
        for name, corpus in (("baseline", baseline), ("candidate", candidate)):
            progress.update(stage=name + "_inference", updated_at=prep.now())
            prep.dump(output / "progress.json", progress, exclusive=False)
            if runtime.runtime_snapshot() != actual_runtime["backends"]:
                raise ValueError("runtime changed before inference pass")
            arrays[name], timings[name] = runtime.predict(model, corpus, ids, 24, 224, "fp16", "cuda",
                 norm="imagenet", batch=8, window_policy="within-slot-valid", log=lambda message: print(name + message, flush=True))
            if state_digest(model) != initial_state or runtime.runtime_snapshot() != actual_runtime["backends"]:
                raise ValueError("model state/runtime changed during inference")
            if not np.isfinite(arrays[name]).all():
                raise ValueError("nonfinite inference output")
            np.save(output / f"{name}.npy", arrays[name], allow_pickle=False)
            prep.dump(output / f"{name}-receipt.json", dict(complete=True, completed_at=prep.now(),
                      ids_sha256=prep.EXPECTED_IDS, timing=timings[name], model_state_sha256=initial_state,
                      predictions_sha256=prep.sha(output / f"{name}.npy")))
        progress.update(stage="paired_identity_gate", updated_at=prep.now())
        prep.dump(output / "progress.json", progress, exclusive=False)
        validate_bindings(contract["bindings"])
        # Recheck the selected baseline bytes and immutable patches after both
        # passes; manifest hashes alone do not detect externally changed shards.
        load_corpora(ids, changed_ids, checks)
        if prep.sha(path) != expected_hash:
            raise ValueError("contract changed during inference")
        validate_runtime(actual_runtime, contract["expected_runtime"], common.environment(), contract["expected_environment"])
        gate = validate_prediction_pair(ids, changed_ids, arrays["baseline"], arrays["candidate"], timings["baseline"], timings["candidate"])
        gate.update(completed_at=prep.now(), contract_sha256=expected_hash,
             predictions={name: prep.sha(output / f"{name}.npy") for name in arrays},
             source_bindings_rechecked=True, model_state_unchanged=True, runtime_unchanged=True)
        prep.dump(output / "comparison-gate.json", gate)
        # First allowed reference-label file open in this command occurs here.
        progress.update(stage="fixed_label_evaluation", labels_opened=True, updated_at=prep.now())
        prep.dump(output / "progress.json", progress, exclusive=False)
        labels = read_reference_labels(contract, gate)
        import exp081_metrics
        changed = np.array([uid in set(changed_ids) for uid in ids], dtype=bool)
        report = exp081_metrics.evaluate(labels, arrays["baseline"], arrays["candidate"], np.array(groups), changed,
                                         diagnostic_groups=diagnostics, reps=500, seed=20261003)
        report.update(experiment=prep.EXPERIMENT, created_at=prep.now(), contract_sha256=expected_hash,
            selection_map_sha256=contract["selection_map_sha256"], checkpoint_sha256=contract["checkpoint_sha256"],
            reference_file_sha256=contract["reference_labels"]["file_sha256"],
            label_scope="Only exact frozen F2 weak label values used; train.csv/Gold labels never opened. float32 then np.rint.",
            reference_canonical_hash_from_frozen_provenance=contract["reference_labels"]["canonical_sha256"],
            runtime=actual_runtime, environment=actual_environment, timings=timings,
            mask_mechanism=contract["mask_mechanism"], limitations=contract["limitations"], gate=gate,
            gpu_inference_passes=2, new_training_fits=0, total_seconds=time.perf_counter()-started)
        for name in arrays:
            common.write_predictions(output / f"{name}.csv", ids, arrays[name])
        prep.dump(output / "report.json", report)
        validate_bindings(contract["bindings"])
        prep.dump(output / "SUCCESS.json", dict(status="TWO_PASSES_AND_FIXED_EVALUATION_COMPLETE", completed_at=prep.now(),
             contract_sha256=expected_hash, report_sha256=prep.sha(output / "report.json"),
             seconds=time.perf_counter()-started, labels_opened_after_gate=True))
        progress.update(stage="complete", updated_at=prep.now()); prep.dump(output / "progress.json", progress, exclusive=False)
        print(json.dumps(dict(status="COMPLETE", report=prep.uri(output / "report.json"), seconds=time.perf_counter()-started)), flush=True)
        return report
    except Exception as exc:
        prep.dump(output / "FAILED.json", dict(status="STOP_NO_AUTOMATIC_RETRY", stage=progress["stage"],
             labels_opened=progress["labels_opened"], error=f"{type(exc).__name__}: {exc}", failed_at=prep.now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register", help="CPU only; verify inputs and write immutable launch contract")
    registration.add_argument("--out", type=Path, default=PROTOCOL)
    inference = commands.add_parser("run", help="Exactly two GPU passes and gated fixed evaluation; root launches separately")
    inference.add_argument("--contract", type=Path, default=PROTOCOL)
    inference.add_argument("--contract-sha256", required=True)
    args = parser.parse_args()
    if args.command == "register":
        register(args.out)
    else:
        run(args.contract, args.contract_sha256)


if __name__ == "__main__":
    main()
