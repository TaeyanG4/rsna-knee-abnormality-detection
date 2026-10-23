"""MIL registration v1.1.0 (2026-10-11).

Changelog: production members use a hash-pinned recipe and target declaration,
without reading or requiring paired controls. Experimental migration unchanged.

Convert hash-pinned legacy recipes into new immutable specs. Normalize relative
Windows paths only here; signed specs stay strictly POSIX. No generated source,
legacy guard imports, time limits, label output or launch side effects.
"""
from pathlib import Path

import numpy as np

from training.arms import lifecycle as life

VERSION = "1.1.0"
AUGMENTATION = dict(rotation_degrees=[-10., 10.], scale=[.9, 1.1],
    translation_fraction=[-.05, .05], gamma=[.8, 1.25], horizontal_flip_probability=.5,
    horizontal_flip_slots=[32, 63], sagittal_flip=False, affine_interpolation="bilinear",
    affine_fill=0., transform_order=["brightness", "affine", "gamma", "flip", "normalise"])
CROP = dict(crop_policy="mixed", nominal_mm=[140, 100], detail_slice=[48, 288],
            allocation="even=context,odd=detail", shared_frozen_initial_state=True)


def scientific_subset(record, expected, message):
    life._need(all(k in record and life.canonical_bytes(record[k]) == life.canonical_bytes(v)
                   for k, v in expected.items()), message)


def portable_member(value):
    life._need(type(value) is str, "member must be text")
    return life.safe_member(value.replace("\\", "/"))


def control_records(spec, root):
    life._need(spec.run_kind == "experiment", "control records apply only to experiments")
    records = {}
    for name, digest in spec.document["target_environment"]["evidence"].items():
        paths = [p for p, h in spec.document["inputs"]["controls"].items() if h == digest]
        life._need(len(paths) == 1, "one bound control record required")
        records[name] = life.parse_json(life._path(root, paths[0]).read_bytes())
    return records["summary.json"], records["provenance.json"]


def validate_control(spec, summary, provenance):
    """Compare scientific keys, not incidental paths, tags, workers or wall time.

The supported contrast is the MIL head. Labels, roster, seed, fold and all
other scientific settings must match the recorded multiscale control.
"""
    from training.arms import mil
    life._need(spec.run_kind == "experiment", "paired control applies only to experiments")
    c, d = spec.document["config"], spec.document
    r, u, w = c["roots"], c["updates"], c["windows"]
    args = summary["args"]
    expected = dict(seed=d["seed"], fold=d["fold"], arch=c["model"]["arch"],
                    variant="AUG1", mode="full", res=w["res"], k=w["train_k"],
                    k_eval=w["eval_k"], epochs=u["epochs"], fixed_epoch=u["fixed_epoch"],
                    bs=c["sampler"]["batch"], accum=u["accum"], bb_lr=u["bb_lr"],
                    head_lr=u["head_lr"], wd=u["weight_decay"], grad_ckpt=False,
                    amp=u["amp_train"], amp_init_scale=u["amp_init_scale"],
                    eval_amp=u["amp_eval"], norm=w["norm"], input_policy=w["input"],
                    window_policy=w["policy"], runtime_policy="strict", device="cuda")
    scientific_subset(args, expected, "control scientific args differ")
    expected = dict(seed=d["seed"], fold=d["fold"], arch=c["model"]["arch"],
                    train_ids_ordered_sha256=r["train_sha256"],
                    eval_ids_ordered_sha256=r["val_sha256"],
                    label_semantic_sha256=c["labels"]["semantic_sha256"],
                    label_file_sha256=d["inputs"]["data"][c["labels"]["path"]],
                    initial_state_sha256=c["model"]["initial_semantic_sha256"],
                    corpus_manifest_sha256=d["inputs"]["data"][r["manifest"]],
                    len_train_loader=u["loader_batches"], scheduler_total_steps=u["steps"],
                    max_fp16_skips=u["max_skips"], k_eval=w["eval_k"], k_eval_extra=w["extra_k"],
                    sampling=c["sampler"]["stream"], checkpoint_selection="fixed-final",
                    loss_policy=u["loss"], flip_tta=c["readout"]["tta"],
                    initial_state_matches_registered=True, initial_state_mismatched_keys=[],
                    training_mode="full", input_policy=w["input"], norm=w["norm"],
                    gold_access="disabled", exp127_variant="AUG1")
    scientific_subset(provenance, expected, "control scientific identity differs")
    scientific_subset(provenance["exp127_variant_config"], AUGMENTATION, "control augmentation differs")
    for key in ("actual_pos_weight", "computed_pos_weight"):
        mil.need(np.array_equal(np.asarray(provenance[key], np.float32),
                                np.asarray(c["labels"]["pos_weight"], np.float32)),
                 "control positive weights differ")
    for record in (summary, provenance):
        scientific_subset(record["exp179"], CROP, "control crop differs")
    for key in ("runtime", "eval_runtime"):
        runtime = provenance[key]
        mil.need(runtime["effective_amp"] == "fp16" and runtime["device"] == "cuda"
                 and runtime["precision_fallback_allowed"] is False
                 and runtime["backends"] == c["backend_policy"], "control runtime differs")
    mil.need(summary["selected_epoch_1based"] == u["fixed_epoch"]
             and summary["train"] == r["n_train"] and summary["val"] == r["n_val"]
             and 0 <= summary["skipped_updates"] <= u["max_skips"]
             and summary["optimizer_updates"] + summary["skipped_updates"] == u["steps"]
             and provenance["fp16_skipped_updates"] == summary["skipped_updates"],
             "control completion differs")


def register_member(root, *, recipe, recipe_sha256, target, target_sha256, arm, attempt):
    """Use an existing scientific recipe, retaining its seed/fold/input hashes.

    Prior sources, controls and target qualification are not inherited. Bind
    the recipe as provenance, the current sources and an explicit target. Only
    recipe/data/roster bytes and the target declaration are needed on this path.
    """
    from training.arms import mil
    root = Path(root)
    recipe, target = portable_member(recipe), portable_member(target)
    life._verify_files(root, {recipe: recipe_sha256, target: target_sha256})
    doc = life.ArmSpec(life._path(root, recipe).read_bytes()).document
    profile, declaration = life.environment_from_declaration(
        life._path(root, target), expected_sha256=target_sha256)
    doc.update(schema=life.MEMBER_SPEC_SCHEMA, run_kind="member", arm=arm, attempt=attempt,
               target_environment=profile)
    doc["inputs"] = dict(
        sources={p: life.file_sha256(life._path(root, p)) for p in mil.SOURCE_FILES},
        data=doc["inputs"]["data"], rosters=doc["inputs"]["rosters"], controls={},
        evidence={recipe: recipe_sha256, target: target_sha256})
    spec = life.ArmSpec.from_document(doc)
    life.check_target_observation(spec, declaration)
    mil.validate_data(spec, root, root / doc["config"]["roots"]["corpus"])
    return spec


def migrate(root, *, protocol, protocol_sha256, control, control_hashes, arm, attempt):
    """Return a verified spec without writing. Caller supplies trusted digests.

Historical text, timing caps and registration-host fields are not executable
config. Bind the original protocol bytes as provenance; never rewrite them.
"""
    from training.arms import mil
    root = Path(root)
    protocol, control = portable_member(protocol), portable_member(control)
    life._verify_files(root, {protocol: protocol_sha256})
    old = mil.read(life._path(root, protocol))
    mil.need(set(control_hashes) == {"summary.json", "provenance.json"}, "control digest pair required")
    profile = life.environment_from_control(
        life._path(root, control + "/summary.json"), life._path(root, control + "/provenance.json"),
        expected_hashes=control_hashes)
    seed, fold = old["seed"], old["fold"]
    mil.need(type(seed) is int and type(fold) is int and seed in (42, 43) and fold in range(5),
             "unsupported MIL member")
    mil.need(set(old["folds"]) == {str(fold)}, "single fold protocol required")
    f = old["folds"][str(fold)]
    summary = mil.read(root / control / "summary.json")
    provenance = mil.read(root / control / "provenance.json")
    keys = ("arch", "res", "k", "k_eval", "epochs", "fixed_epoch", "bs", "accum",
            "grad_ckpt", "amp", "amp_init_scale", "eval_amp", "runtime_policy", "norm",
            "bb_lr", "head_lr", "wd", "pretrained", "window_policy", "device", "input_policy")
    scientific_subset(old["common_args"], {k: summary["args"][k] for k in keys}, "legacy scientific args differ")
    scientific_subset(old["variant_config"], AUGMENTATION, "legacy augmentation differs")
    scientific_subset(old["exp179"], CROP, "legacy crop differs")
    mil.need(old["extra_eval"]["k_eval"] == mil.WINDOWS["extra_k"] and old["variant"] == "AUG1"
             and old["recipe"]["loss_policy"] == mil.UPDATES["loss"]
             and old["exp179"]["crop_policy"] == "mixed"
             and old["exp185"]["model_kind"] == "finding_spatial_tokens_v1",
             "legacy recipe differs")
    corpus = portable_member(old["paths"]["corpus"])
    model = dict(arch=mil.ARCH, kind="finding_spatial_tokens_v1",
                 weights=portable_member(old["init_weights"]["local"]),
                 weights_sha256=old["init_weights"]["sha256"],
                 initial_state="artifacts/experiments/EXP166_CONVNEXT_T/initial-state.pt",
                 initial_file_sha256=old["initial_state_file_sha256"],
                 initial_semantic_sha256=old["initial_state_sha256"])
    labels = {k: old["label_set"][k] for k in ("path", "name", "semantic_sha256")}
    labels.update(path=portable_member(labels["path"]), pos_weight=f["pos_weight"])
    config = dict(model=model, labels=labels, targets=mil.TARGETS, windows=mil.WINDOWS,
                  sampler=mil.SAMPLER, updates={**mil.UPDATES, "loader_batches": f["len_train_loader"],
                    "steps": f["steps"], "max_skips": f["max_fp16_skips"]},
                  bn={"policy": "none", "module_count": 0}, readout=mil.READOUT,
                  backend_policy=provenance["runtime"]["backends"],
                  roots=dict(corpus=corpus, manifest=corpus + "/manifest.json",
                    ordered=portable_member(f["ordered_ids"]["path"]), groups=portable_member(f["groups"]["path"]),
                    canonical=portable_member(old["paths"]["base_ordered_ids"]),
                    order_policy="base-canonical" if fold == 0 else "sorted-uid",
                    n_train=f["n_train"], n_val=f["n_val"],
                    train_sha256=f["ordered_ids"]["train_sha256"], val_sha256=f["ordered_ids"]["val_sha256"]))
    roots = config["roots"]
    expected_files = {roots["manifest"]: old["corpus_manifest_sha256"],
                      labels["path"]: old["label_set"]["sha256"],
                      model["weights"]: model["weights_sha256"],
                      model["initial_state"]: model["initial_file_sha256"],
                      roots["ordered"]: f["ordered_ids"]["sha256"], roots["groups"]: f["groups"]["sha256"]}
    life._verify_files(root, expected_files)
    inputs = {group: {p: life.file_sha256(life._path(root, p)) for p in paths} for group, paths in (
        ("sources", mil.SOURCE_FILES), ("data", [roots["manifest"], labels["path"], model["weights"], model["initial_state"]]),
        ("rosters", [roots[k] for k in ("ordered", "groups", "canonical")]))}
    inputs["controls"] = {control + "/" + name: digest for name, digest in control_hashes.items()}
    inputs["controls"][protocol] = protocol_sha256
    spec = life.ArmSpec.from_document(dict(schema=life.SPEC_SCHEMA, family="convnext-mil", arm=arm,
        seed=seed, fold=fold, attempt=attempt, config=config, inputs=inputs, target_environment=profile, outputs=mil.OUTPUTS))
    mil.validate_data(spec, root, root / corpus)
    return spec
