"""H224 fold dataset packager v1.4.0, local CPU/file operations only.

Changelog v1.4.0: explicit local seed1 producer and Vast seed2; preserve historical exports.
Changelog v1.3.0: fullbag-all/standard-all-seed1, exact HOST_BOUND_EXPORT and actual seed metadata.
Changelog v1.2.0: accept explicit D-A-052 fullbag fold0 exports and model ID.
Changelog v1.1.0: accept fixed-final-EMA all-data exports with training-only guard.
v1.0.0: verify complete exports, reconstruct hash-bound training source,
and emit the four-file private asset layout consumed by k1sr_runtime_v7.
No checkpoint deserialization, labels, account access, or upload.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil

from tools import k1n_config_v1 as schema

VERSION = "1.4.0"
ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / ".local/scratch/astra-b1/k1-shared-render/dataset-k1-v3gs1/source/compact_train.py"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def member(root, name):
    """Reject traversal and linked ancestors before opening any input."""
    need(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_./-]+", name),
         "unsafe member name")
    parts = PurePosixPath(name).parts
    need(parts and not PurePosixPath(name).is_absolute() and ".." not in parts
         and str(PurePosixPath(name)) == name, "unsafe member path")
    root = Path(root).absolute()
    path = root / name
    for item in (path, *path.parents):
        need(not item.is_symlink() and not item.is_junction(), "linked input/output path")
    need(path.resolve().is_relative_to(root.resolve()), "member escapes root")
    return path


def next_backend(variant, seed, producer_mode=None):
    if producer_mode in ('local_all_seed1', 'all_seed2'):
        from tools.l1_rco_k1_next_v2 import backend
        need(variant == 'standard' and seed == (1 if producer_mode == 'local_all_seed1' else 2), 'producer seed/variant mismatch')
        return backend(producer_mode)
    need(producer_mode is None and ((variant == 'fullbag' and seed == 0) or (variant == 'standard' and seed == 1)), 'unsupported next identity')
    from tools.l1_rco_k1_next_v1 import backend
    return backend('fullbag_all' if variant == 'fullbag' else 'all_seed1')


def derive_training_source(serving, fold, variant=None, seed=0, producer_mode=None):
    if fold == "all" and (variant == "fullbag" or seed in (1, 2)):
        return next_backend(variant, seed, producer_mode).derive_trainer(serving.decode("utf-8")).encode("utf-8")
    # Reuse the exact source transform, never execute/import the trainer itself.
    if fold == 0:
        from tools import l1_rco_k1_fullbag_v1 as producer
        return producer.derive_trainer(serving.decode("utf-8")).encode("utf-8")
    if fold == "all":
        from tools import l1_rco_k1_all_v1 as producer
        return producer.derive_trainer(serving.decode("utf-8")).encode("utf-8")
    environment = {key: os.environ.get(key) for key in ("K1_FOLD", "K1_GPU")}
    try:
        from tools import l1_rco_k1_folds_v1 as producer
        previous = producer.FOLD
        try:
            producer.FOLD = fold
            return producer.derive_trainer(serving.decode("utf-8")).encode("utf-8")
        finally:
            producer.FOLD = previous
    finally:
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def verify_export(export, serving_source=SERVING):
    export = Path(export).absolute()
    need(export.name == "EXPORT.json", "expected EXPORT.json")
    root = export.parent
    info = read(member(root, "EXPORT.json"))
    fold = info.get("fold")
    is_all = fold == "all"
    need(is_all or (type(fold) is int and fold in (0, 1, 2, 3)), "export fold must be 0..3 or all")
    need(info.get("version") == "1.0.0", "unsupported producer export version")
    checkpoint = f"fit/models/E_prior_s0_f{fold}.pt"
    predictions = "fit/train-guard.csv" if is_all else "fit/holdout.csv"
    expected = {checkpoint, predictions, "fit/result.json", "fit/fit-complete.json",
                "fit/runtime.json", "registration.json", "prepare.json"}
    pins = info.get("sha256")
    need(isinstance(pins, dict) and set(pins) == expected, "export inventory mismatch")
    sums = {}
    for line in member(root, "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_./-]+)", line)
        need(match is not None, "invalid SHA256SUMS line")
        pin, name = match.groups()
        need(name not in sums, "duplicate SHA256SUMS entry")
        sums[name] = pin
    need(sums == pins, "SHA256SUMS/export mismatch")
    for name, pin in pins.items():
        need(schema.valid_hash(pin), "invalid export hash")
        need(sha(member(root, name)) == pin, "export hash mismatch: " + name)
    reg = read(member(root, "registration.json"))
    if fold == 0:
        need(reg.get("variant") == "fullbag" and len(reg["holdout"]) == 977
             and reg["recipe"].get("windows") == 640
             and reg["recipe"].get("encoder_microbatch") == 32,
             "fold0 requires explicit fullbag registration")
    contract = info.get("contract_id")
    need(schema.valid_hash(contract) and reg.get("contract_id") == contract
         and digest({k: v for k, v in reg.items() if k != "contract_id"}) == contract,
         "registration contract mismatch")
    need(reg.get("model_fold") == fold and reg["recipe"]["model_fold"] == fold,
         "registration fold mismatch")
    result = read(member(root, "fit/result.json"))
    complete = read(member(root, "fit/fit-complete.json"))
    prep = read(member(root, "prepare.json"))
    for receipt in (result, complete, prep):
        need(receipt.get("contract_id") == contract, "receipt contract mismatch")
    for receipt in (result, complete):
        need(receipt.get("status") == "COMPLETE" and receipt.get("model_fold") == fold
             and receipt.get("model_sha256") == pins[checkpoint], "incomplete/wrong fold model")
    need(result.get("cpu_test") is False and result.get("clean_oof") is False
         and result.get("exposure") == "EXPOSED_EXPLORATORY", "production exposed result required")
    if is_all:
        need(reg["holdout"] == [] and result.get("holdout_rows") == 0
             and info.get("clean_oof") is False
             and info.get("own_holdout_evaluation") is False
             and info.get("selection") == "final EMA"
             and result.get("own_holdout_evaluation") is False
             and result.get("selection") == complete.get("selection") == "final EMA"
             and complete.get("own_holdout_evaluation") is False,
             "all-data final EMA/no-holdout metadata required")
        from tools import l1_rco_k1_all_v1 as producer
        need(reg["prediction_guard"]["ids"] == producer.guard_ids(reg)
             and len(reg["prediction_guard"]["ids"]) == 32
             and reg["prediction_guard"]["min_target_range"] == 1e-6,
             "all-data guard registration mismatch")
        guard = producer.verify_prediction_guard(reg, root / "fit")
        need(result.get("training_guard_sha256") == pins[predictions]
             and result.get("training_guard_rows") == 32
             and result.get("prediction_guard") == guard,
             "all-data prediction collapse guard mismatch")
        need(reg["selection"]["epochs"] == 10
             and reg["selection"]["early_stopping"] is False
             and reg["selection"]["holdout_selection"] is False
             and reg["recipe"]["total_steps"] == 20830,
             "all-data fixed training recipe required")
    else:
        need(result.get("holdout_sha256") == pins["fit/holdout.csv"]
             and result.get("holdout_rows") == len(reg["holdout"]) > 0, "holdout result mismatch")
    need(complete.get("epochs") == reg["recipe"]["epochs"] == 10
         and complete.get("optimizer_steps") == reg["recipe"]["total_steps"]
         and complete.get("train_studies") == len(reg["train"]) == (4167 if is_all else (3190, 3073, 3100, 3138)[fold]),
         "incomplete training receipt")
    serving_path = Path(serving_source).absolute()
    serving_path = member(serving_path.parent, serving_path.name)
    need(sha(serving_path) == schema.TRAINER_SHA, "frozen serving source hash mismatch")
    serving = serving_path.read_bytes()
    variant, seed = reg.get("variant"), reg.get("model_seed", 0)
    training = derive_training_source(serving, fold, variant, seed, reg.get("producer_mode"))
    if is_all and (variant == "fullbag" or seed in (1, 2)):
        from tools.l1_rco_k1_nexthost_v1 import validate_host_export
        producer = next_backend(variant, seed, reg.get("producer_mode"))
        validate_host_export(dict(k1_result=root/"fit/result.json", k1_csv=root/"fit/train-guard.csv"),
                             result, producer.expected_registration(), "all", backend=producer)
    need(hashlib.sha256(training).hexdigest() == prep.get("trainer_sha256"),
         "reconstructed training source hash mismatch")
    for name in ("Net", "OFLEncoder"):
        def definition(source):
            nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name]
            need(len(nodes) == 1, "architecture definition missing")
            return ast.dump(nodes[0], include_attributes=False)
        need(definition(serving) == definition(training), "serving architecture drift")
    return info, serving_path, prep["trainer_sha256"]


def verify_dataset(directory, model_id=None):
    root = Path(directory).absolute()
    manifest = read(member(root, "asset-manifest.json"))
    metadata = read(member(root, "dataset-metadata.json"))
    need(metadata.get("isPrivate") is True and schema.valid_slug(metadata.get("id")),
         "private real dataset identity required")
    need(metadata.get("title") == metadata["id"].split("/")[1], "dataset title must equal slug")
    files = manifest.get("files")
    need(isinstance(files, dict) and len(files) == 2, "dataset file inventory")
    checkpoints = [n for n in files if schema.safe_path(n, "weights", ".pt")]
    need(len(checkpoints) == 1 and "source/compact_train.py" in files, "dataset layout")
    actual = set()
    for path in root.rglob("*"):
        safe = member(root, path.relative_to(root).as_posix())
        if safe.is_file():
            actual.add(safe.relative_to(root).as_posix())
    need(actual == set(files) | {"asset-manifest.json", "dataset-metadata.json"}, "extra dataset files")
    need(manifest.get("clean_oof") is False and manifest.get("exposure") == "EXPOSED_EXPLORATORY",
         "exposure metadata required")
    for name, pin in files.items():
        need(schema.valid_hash(pin) and sha(member(root, name)) == pin, "dataset file hash mismatch")
    need(files["source/compact_train.py"] == schema.TRAINER_SHA, "serving source identity")
    next_identity = manifest["model_fold"] == "all" and (manifest.get("variant") == "fullbag" or manifest["model_seed"] in (1, 2))
    if next_identity:
        need((manifest.get("variant"), manifest["model_seed"]) in (("fullbag", 0), ("standard", 1), ("standard", 2)), "unsupported all-data variant/seed")
        expected_slug = "taeyangg4/knee-k1-v3gfball-v1-assets" if manifest["variant"] == "fullbag" else f"taeyangg4/knee-k1-v3gall-s{manifest['model_seed']}-v1-assets"
        need(metadata["id"] == expected_slug, "next dataset identity mismatch")
    inferred_id = ("k1_v3g_fullbag_all" if next_identity and manifest["variant"] == "fullbag" else
                   f"k1_v3g_all_s{manifest['model_seed']}" if next_identity else "k1_v3g_fullbag" if manifest.get("variant") == "fullbag" else
                   "k1_v3g_all" if manifest["model_fold"] == "all" else
                   f"k1_v3g_seed{manifest['model_seed']}" if manifest["model_fold"] == 0
                   else f"k1_v3g_fold{manifest['model_fold']}")
    return dict(id=model_id or inferred_id,
                dataset_slug=metadata["id"], checkpoint=checkpoints[0],
                checkpoint_sha256=files[checkpoints[0]], trainer_source="source/compact_train.py",
                trainer_sha256=files["source/compact_train.py"],
                manifest_sha256=sha(root / "asset-manifest.json"))


def package(export, output, dataset_id, serving_source=SERVING):
    need(schema.valid_slug(dataset_id), "real dataset slug required")
    info, source, training_pin = verify_export(export, serving_source)
    fold = info["fold"]
    reg = read(Path(export).parent / "registration.json")
    checkpoint = f"weights/E_prior_s0_f{fold}.pt"
    original = f"fit/models/E_prior_s0_f{fold}.pt"
    files = {checkpoint: info["sha256"][original], "source/compact_train.py": schema.TRAINER_SHA}
    manifest = dict(schema="k1-v3g-fold-assets/v1", version=VERSION, files=files,
                    model_fold=fold, model_seed=reg.get("model_seed", 0), clean_oof=False, exposure="EXPOSED_EXPLORATORY",
                    training_contract=info["contract_id"], training_source_sha256=training_pin,
                    export_sha256=sha(export), export_files=info["sha256"],
                    architecture_evidence="Net and OFLEncoder AST identical to hash-bound reconstructed fold trainer",
                    serving_architecture="convnext_small; spatial=True; extent=False; crop=352; in_chans=3; state_spec=None")
    if fold == "all":
        manifest.update(selection="final EMA", own_holdout_evaluation=False,
                        evaluation_note="All labelled studies used in training; own-holdout evaluation is impossible.")
    elif fold == 0:
        manifest.update(variant="fullbag", own_holdout_evaluation=True)
    if fold == "all" and (reg.get("variant") == "fullbag" or reg.get("model_seed") in (1, 2)):
        manifest.update(variant=reg["variant"], storage_note=reg["storage_note"])
        expected_slug = "taeyangg4/knee-k1-v3gfball-v1-assets" if reg["variant"] == "fullbag" else f"taeyangg4/knee-k1-v3gall-s{reg['model_seed']}-v1-assets"
        need(dataset_id == expected_slug, "next dataset identity mismatch")
    metadata = dict(id=dataset_id, title=dataset_id.split("/")[1], isPrivate=True,
                    licenses=[dict(name="other")],
                    description=f"Private H224 K1 v3-G fold{fold} checkpoint and frozen serving source. "
                                "No data, labels or predictions. EXPOSED_EXPLORATORY; clean_oof=false.")
    output = Path(output).absolute()
    member(output, "dataset-metadata.json")  # Reject linked destination ancestors.
    need(not output.exists(), "output exists; use verify-dataset or a new version")
    output.mkdir(parents=True)
    for name, src in ((checkpoint, Path(export).parent / original), ("source/compact_train.py", source)):
        dst = member(output, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        need(sha(dst) == files[name], "copy hash mismatch")
    (output / "asset-manifest.json").write_bytes(encoded(manifest))
    # Metadata is the completion marker; a partial copy cannot be upload-ready.
    (output / "dataset-metadata.json").write_bytes(encoded(metadata))
    return verify_dataset(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("verify-export", "package"):
        sub = commands.add_parser(action)
        sub.add_argument("--export", type=Path, required=True)
        sub.add_argument("--serving-source", type=Path, default=SERVING)
        if action == "package":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--dataset-id", required=True)
    sub = commands.add_parser("verify-dataset")
    sub.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "verify-export":
            info, _, _ = verify_export(args.export, args.serving_source)
            result = dict(status="PASS_EXPORT", fold=info["fold"], files=len(info["sha256"]))
        elif args.action == "package":
            result = dict(status="PREPARED_LOCAL_ONLY", model=package(
                args.export, args.output, args.dataset_id, args.serving_source))
        else:
            result = dict(status="PASS_DATASET", model=verify_dataset(args.dataset))
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, "HOLD: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
