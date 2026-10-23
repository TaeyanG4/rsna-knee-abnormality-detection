"""EXP132 local external-corpus conversion, frozen queue plans, and F0 screens.

No command here launches a GPU job or uploads external data.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "src"))
from raptor_common import Corpus, SLOTS64  # noqa: E402
from tools import exp127_radr50_single as control  # noqa: E402
from tools import exp129_radr50_single as screen_math  # noqa: E402

SOURCE_NAMES = {"kneemri": "KMRI", "mrnet": "MRN"}
FULL_SOURCE = {"KMRI": "KNEEMRI", "MRN": "MRNET"}
VARIANT_SOURCES = {"KMRI": ("KMRI",), "MRN": ("MRN",), "BOTH": ("KMRI", "MRN")}
LABELS = {"KMRI": ("kmri_acl_any", "kmri_acl_full"),
          "MRN": ("mrn_abnormal", "mrn_acl", "mrn_meniscus")}
LICENCES = {"KMRI": "KneeMRI CC BY-NC-ND 4.0; local non-commercial research; no redistribution",
            "MRN": "MRNet Stanford Research Agreement; non-commercial research, no redistribution, no re-identification"}
EXP = ROOT / "artifacts" / "experiments" / "EXP132_EXTERNAL_AUX"
RATIO, LOSS_WEIGHT = .25, .5
SEEDS = (42, 43)
CODE_NAMES = control.CODE_NAMES
ENV_KEYS = control.ENV_KEYS
LABEL_SET = control.LABEL_SET
worker_init = control.worker_init
read_rows = control.read_rows
read = control.read
canonical_order = control.canonical_order
validate_fold_membership = control.validate_fold_membership
pos_weight = control.pos_weight
fold_contract = control.fold_contract
state_hashes = control.state_hashes
mismatched_keys = control.mismatched_keys
uid_hash = control.uid_hash
load_label_csv = control.load_label_csv
install_gold_blocker = control.install_gold_blocker
forbid_cuda_init = control.forbid_cuda_init


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_dir(source, root=ROOT):
    return Path(root) / "storage" / "cache" / "derived" / f"EXT132_{FULL_SOURCE[source]}_336"


def source_dir(source, root=ROOT):
    return Path(root) / "storage" / "external_data" / FULL_SOURCE[source].lower()


def labels_filename(source):
    return f"EXT132_{FULL_SOURCE[source]}_labels.csv"


class RestrictedNumpyUnpickler(pickle.Unpickler):
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"),
               ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"),
               # protocol-2 pickles store array bytes as _codecs.encode(text, 'latin1'): a pure str->bytes call
               ("_codecs", "encode")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")
        return super().find_class(module, name)


def load_kneemri_pickle(path):
    with Path(path).open("rb") as stream:
        arr = RestrictedNumpyUnpickler(stream).load()
        if stream.read(1):
            raise ValueError("trailing bytes in KneeMRI pickle")
    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.dtype.kind not in "uif":
        raise ValueError("KneeMRI volume must be numeric slices-first ndarray")
    if not np.isfinite(arr).all() or min(arr.shape) < 2:
        raise ValueError("invalid KneeMRI volume")
    return arr


def _label(x):
    value = int(x)
    if value not in (0, 1):
        raise ValueError("external label must be binary")
    return value


def kneemri_records(source):
    meta = source / "metadata.csv"
    if not meta.is_file():
        raise FileNotFoundError(meta)
    records, missing = [], 0
    # The distributed layout keeps volumes in vol01..volNN subfolders while metadata.csv names bare files.
    index = {}
    for p in sorted(source.glob("*.pck")) + sorted(source.glob("vol*/*.pck")):
        if index.setdefault(p.name, p) != p:
            raise ValueError(f"duplicate KneeMRI volume file: {p.name}")
    with meta.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            grade = int(row["aclDiagnosis"])
            if grade not in (0, 1, 2):
                raise ValueError("invalid KneeMRI ACL grade")
            rel = Path(row["volumeFilename"].replace("\\", "/"))
            if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() != ".pck":
                raise ValueError("unsafe KneeMRI volume filename")
            path = source / rel if len(rel.parts) > 1 else index.get(rel.name, source / rel)
            if not path.is_file():
                missing += 1
                continue
            uid = f"KMRI_{row['examId']}_{row['seriesNo']}"
            records.append((uid, {"Sagittal": path}, (int(grade >= 1), int(grade == 2))))
    return records, missing


def _read_mrn_labels(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    values = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) != 2:
                raise ValueError(f"MRNet label row width invalid: {path.name}")
            case = int(row[0])
            if case in values:
                raise ValueError(f"duplicate MRNet case in {path.name}")
            values[case] = _label(row[1])
    return values


def mrnet_records(source):
    if not source.is_dir():
        raise FileNotFoundError(f"MRNet directory absent: {source}")
    records = []
    for split in ("train", "valid"):
        labels = [_read_mrn_labels(source / f"{split}-{name}.csv")
                  for name in ("abnormal", "acl", "meniscus")]
        if not labels[0] or any(set(x) != set(labels[0]) for x in labels[1:]):
            raise ValueError(f"MRNet {split} labels do not align")
        for case in sorted(labels[0]):
            planes = {p: source / split / p.lower() / f"{case:04d}.npy"
                      for p in ("Sagittal", "Coronal", "Axial")}
            absent = [p for p in planes.values() if not p.is_file()]
            if absent:
                raise FileNotFoundError(f"MRNet plane absent: {absent[0]}")
            records.append((f"MRN_{split}_{case:04d}", planes, tuple(x[case] for x in labels)))
    return records, 0


def _render_plane(arr, count):
    import cv2
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.dtype.kind not in "uif" or not np.isfinite(arr).all():
        raise ValueError("external volume must be finite numeric slices-first array")
    if arr.shape[0] < 2:
        raise ValueError("external volume needs >=2 slices")
    chosen = np.linspace(0, arr.shape[0] - 1, count).round().astype(int)
    picked = arr[chosen].astype(np.float32)
    lo, hi = np.percentile(picked, [2., 98.])
    picked = np.clip((picked - lo) / (hi - lo + 1e-6), 0, 1)
    return np.stack([np.asarray(cv2.resize(a, (336, 336), interpolation=cv2.INTER_AREA) * 255,
                                 dtype=np.uint8) for a in picked])


def _load_plane(path, source):
    if source == "KMRI":
        return load_kneemri_pickle(path)
    a = np.load(path, allow_pickle=False)
    if a.dtype != np.uint8:
        raise ValueError(f"MRNet volume must be uint8: {path}")
    return a


# v2 (B, 2026-10-08 after the EXP137 MRNet montage QA): a series fills ONLY the slot group of its own contrast,
# as in the competition contract SLOTS64 = [Sag fluid1 x18, Sag fluid0 x14, Cor fluid1 x12, Cor fluid0 x8, Axial x12].
# v1 filled every slot of the plane, putting MRNet coronal T1 into the fat-suppressed coronal group and the
# KneeMRI/MRNet fat-suppressed sagittal series into the non-FS sagittal group.
SLOT_GROUPS = {("Sagittal", 1): (0, 18), ("Sagittal", 0): (18, 14), ("Coronal", 1): (32, 12),
               ("Coronal", 0): (44, 8), ("Axial", -1): (52, 12)}
SOURCE_CONTRAST = {"KMRI": {"Sagittal": 1},                              # KneeMRI: sagittal PD fat-suppressed
                   "MRN": {"Sagittal": 1, "Coronal": 0, "Axial": -1}}   # MRNet: sag T2 FS, cor T1, axial PD FS


def _volume(planes, source):
    vol = np.zeros((64, 336, 336), dtype=np.uint8)
    mask = np.zeros(64, dtype=np.uint8)  # filled slots, not nonzero pixels: dark slices after the percentile clip stay valid
    for plane in ("Sagittal", "Coronal", "Axial"):
        if plane in planes:
            start, count = SLOT_GROUPS[(plane, SOURCE_CONTRAST[source][plane])]
            vol[start:start + count] = _render_plane(_load_plane(planes[plane], source), count)
            mask[start:start + count] = 1
    if not mask.any() or not vol.any():
        raise ValueError("external volume has no nonzero pixels")
    return vol, mask


def convert(source, root=ROOT, shard_size=32):
    source = SOURCE_NAMES.get(source.lower(), source.upper())
    if source not in LABELS or shard_size < 1:
        raise ValueError("unknown source or invalid shard size")
    src, out = source_dir(source, root), corpus_dir(source, root)
    if not src.is_dir():
        raise FileNotFoundError(f"{'MRNet' if source == 'MRN' else 'KneeMRI'} directory absent: {src}")
    provenance_path = src / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"external provenance receipt absent: {provenance_path}")
    if out.exists():
        raise FileExistsError(f"refusing existing external corpus: {out}")
    records, missing = (kneemri_records(src) if source == "KMRI" else mrnet_records(src))
    if not records or len({r[0] for r in records}) != len(records):
        raise ValueError("empty or duplicate external study IDs")
    out.mkdir(parents=True)
    ids, masks, label_rows, shards = [], [], [], []
    for start in range(0, len(records), shard_size):
        batch = records[start:start + shard_size]
        vols = np.lib.format.open_memmap(out / f"vols_shard{len(shards):02d}.npy", mode="w+", dtype=np.uint8,
                                         shape=(len(batch), 64, 336, 336))
        for j, (uid, planes, labels) in enumerate(batch):
            vol, mask = _volume(planes, source)
            vols[j] = vol
            ids.append(uid); masks.append(mask); label_rows.append((uid, *labels))
        vols.flush(); del vols
        name = f"vols_shard{len(shards):02d}.npy"
        shards.append(dict(shard=len(shards), start=start, stop=start + len(batch), file=name,
                           sha256=digest(out / name), done=True))
    np.save(out / "ids.npy", np.asarray(ids, dtype=f"<U{max(map(len, ids))}"))
    np.save(out / "masks.npy", np.stack(masks))
    with (out / labels_filename(source)).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(("StudyInstanceUID", *LABELS[source])); writer.writerows(label_rows)
    spec = dict(slots=SLOTS64, depth=64, img=336, crop_mm=None, span=[0., 1.],
                normalization="per-volume selected slices percentile 2-98, clip, uint8",
                resize="cv2.INTER_AREA to 336 square; physical spacing unavailable",
                orientation="left/right not normalised; competition corpus has no laterality convention")
    manifest = dict(contract=f"ext132_{source.lower()}_336", spec=spec, builder="EXP132 v1", studies=len(ids),
                    dtype="uint8", shape_total=[len(ids), 64, 336, 336], shards=shards,
                    ids_sha256=digest(out / "ids.npy"), masks_sha256=digest(out / "masks.npy"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs = ["ids.npy", "masks.npy", "manifest.json", labels_filename(source), *(s["file"] for s in shards)]
    receipt = dict(source=source, studies=len(ids), missing_files_skipped=missing,
                   source_provenance_sha256=digest(provenance_path), licence=LICENCES[source],
                   output_sha256={name: digest(out / name) for name in outputs},
                   orientation="not normalised", physical_spacing="unavailable")
    (out / "conversion_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    # Exercise the unchanged Corpus reader before reporting success.
    corpus = Corpus(out)
    if len(corpus) != len(ids) or corpus.volume(ids[0]).shape != (64, 336, 336):
        raise RuntimeError("external Corpus contract verification failed")
    return receipt


def tag_for(variant, seed):
    if variant not in VARIANT_SOURCES or seed not in SEEDS:
        raise ValueError("unregistered EXP132 variant/seed")
    return f"EXP132_{variant}_S{seed}_F0"


def protocol_path(variant, seed, root=ROOT):
    return Path(root) / "artifacts/experiments/EXP132_EXTERNAL_AUX" / "protocol" / variant / f"s{seed}.json"


def external_bindings(variant, root=ROOT):
    result = {}
    for source in VARIANT_SOURCES[variant]:
        d = corpus_dir(source, root)
        receipt = d / "conversion_receipt.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"convert {source} before queuing: {receipt}")
        rec = json.loads(receipt.read_text(encoding="utf-8"))
        for name, expected in rec["output_sha256"].items():
            if digest(d / name) != expected:
                raise ValueError(f"external corpus changed: {name}")
        result[source] = dict(corpus=str(d), receipt_sha256=digest(receipt),
                              manifest_sha256=digest(d / "manifest.json"),
                              labels_sha256=digest(d / labels_filename(source)),
                              studies=rec["studies"], label_names=list(LABELS[source]))
    return result


def make_protocol(variant, seed, root=ROOT):
    base = control.base_manifest(seed, root)
    control.check_control_assets(base, seed, root)
    ext = external_bindings(variant, root)
    code = {p: digest(Path(root) / p) for p in
            ("tools/exp132_external_aux.py", "training/src/raptor_exp132.py",
             "tests/test_exp132_external_aux.py", "artifacts/experiments/EXP132_EXTERNAL_AUX/PROTOCOL.md")}
    return dict(schema="exp132-external-aux-v1", variant=variant, seed=seed, fold=0,
                tag=tag_for(variant, seed), base_protocol_path=control.BASES[seed][0] + "/manifest.json",
                base_protocol_sha256=control.BASES[seed][1], base_tag=base["folds"]["0"]["tag"],
                common_args=copy.deepcopy(base["common_args"]), fold_contract=copy.deepcopy(base["folds"]["0"]),
                external=ext, external_batches_per_competition_step=RATIO,
                auxiliary_bce_weight=LOSS_WEIGHT, external_schedule="one external batch before every fourth competition optimizer step; one shared update",
                code_sha256=code, screen=control.screen_spec(root),
                partner_proxy="EXP129 fixed 0.80 CoAtNet5 + 0.20 candidate, report only",
                frozen_scope="fold 0, seeds 42/43; validation competition OOF only; no serving auxiliary parameters")


def load_protocol(variant, seed, root=ROOT):
    path = protocol_path(variant, seed, root)
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != make_protocol(variant, seed, root):
        raise ValueError("EXP132 protocol/input/code binding changed")
    return stored


def validate_arguments(args, root=ROOT):
    root = Path(root)
    p = load_protocol(args.variant, args.seed, root)
    base = control.base_manifest(args.seed, root)
    control.check_control_assets(base, args.seed, root)
    if args.fold != 0 or Path(args.protocol).resolve() != protocol_path(args.variant, args.seed, root).resolve():
        raise ValueError("EXP132 protocol/fold mismatch")
    if os.environ.get("KNEE_REMOTE_RELAX_ENV"):
        raise ValueError("EXP132 local protocol refuses remote relaxation")
    smoke = args.mode == "smoke"
    for key, expected in p["common_args"].items():
        got = getattr(args, key, None)
        if got != expected and not (args.validate_only and key == "device" and got == "cpu") and not (
                smoke and got in base["smoke_allowed"].get(key, [])):
            raise ValueError("frozen control argument changed: " + key)
    if args.validate_only and args.device != "cpu":
        raise ValueError("validation must be CPU")
    if not 0 <= args.workers <= 12:
        raise ValueError("workers outside 0..12")
    if args.tag != p["tag"] + ("_SMOKE" if smoke else ""):
        raise ValueError("EXP132 tag mismatch")
    for key in ("labels", "eval_labels", "corpus"):
        if Path(getattr(args, key)).resolve() != (root / base["paths"][key]).resolve():
            raise ValueError("control input path changed: " + key)
    if Path(args.init_weights).resolve() != control.weights_path(root).resolve():
        raise ValueError("RadImageNet weight path changed")
    for key, value in {**control.THREAD_ENV, "HF_HUB_OFFLINE": "1"}.items():
        if os.environ.get(key) != value:
            raise ValueError("registered environment missing: " + key)
    default = (root / "artifacts/experiments").resolve()
    out = Path(args.output_root).resolve() if args.output_root else default
    if smoke:
        if not args.output_root or control.within(out, root / "artifacts"):
            raise ValueError("smoke output must be outside artifacts")
    elif out != default:
        raise ValueError("full output root changed")
    return base, base["folds"]["0"], control.read(root / base["folds"]["0"]["ordered_ids"]["path"]), out


def register(variant, root=ROOT):
    if variant not in VARIANT_SOURCES:
        raise ValueError("unknown variant")
    protocols = {seed: make_protocol(variant, seed, root) for seed in SEEDS}
    for seed, p in protocols.items():
        path = protocol_path(variant, seed, root)
        if path.exists():
            load_protocol(variant, seed, root)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as f:
                json.dump(p, f, indent=2)
    return protocols


def runner_argv(variant, seed, root=ROOT, workers=12):
    from tools import exp113_radr50_local as e113
    if not 0 <= workers <= 12:
        raise ValueError("workers outside 0..12")
    p = load_protocol(variant, seed, root)
    base = control.base_manifest(seed, root)
    argv = e113.runner_argv(base, 0, "full", root=root, workers=workers, label_set="dread")
    argv[1] = str(Path(root) / "training/src/raptor_exp132.py")
    i = argv.index("--label-set")
    del argv[i:i + 2]
    argv = control._set_flag(argv, "--protocol", str(protocol_path(variant, seed, root)))
    argv = control._set_flag(argv, "--tag", p["tag"])
    return argv + ["--variant", variant]


def queue(variant, after, out, root=ROOT, workers=12):
    register(variant, root)
    entries, previous = [], None if after.lower() == "none" else after
    for seed in SEEDS:
        item = dict(tag=tag_for(variant, seed), command=runner_argv(variant, seed, root, workers),
                    continue_on_failure=True)
        if previous:
            item["after"] = previous
        entries.append(item); previous = item["tag"]
    if out != "-":
        path = Path(out)
        if path.exists():
            raise FileExistsError(f"refusing existing queue plan: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    print(json.dumps(entries, indent=2))
    return entries


def validate_run(run, protocol, root=ROOT):
    run = Path(run)
    required = ("summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv", "oof_k36.csv")
    inv = json.loads((run / "output-inventory.json").read_text(encoding="utf-8"))
    if set(inv) != set(required):
        raise ValueError("EXP132 output inventory incomplete")
    for name in required:
        if inv[name]["sha256"] != digest(run / name):
            raise ValueError(f"EXP132 output changed: {name}")
    prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    expected = dict(protocol_sha256=digest(protocol_path(protocol["variant"], protocol["seed"], root)),
                    exp132_variant=protocol["variant"], exp132_external=protocol["external"],
                    exp132_ratio=RATIO, exp132_loss_weight=LOSS_WEIGHT,
                    seed=protocol["seed"], fold=0, training_mode="full", gold_access="disabled",
                    initial_state_matches_registered=True, environment_matches_baseline=True,
                    initial_state_sha256=control.base_manifest(protocol["seed"], root)["initial_state_sha256"],
                    runner_sha256=protocol["code_sha256"]["training/src/raptor_exp132.py"],
                    guard_sha256=protocol["code_sha256"]["tools/exp132_external_aux.py"],
                    exp132_code_sha256=protocol["code_sha256"],
                    train_ids_ordered_sha256=protocol["fold_contract"]["ordered_ids"]["train_sha256"],
                    eval_ids_ordered_sha256=protocol["fold_contract"]["ordered_ids"]["val_sha256"])
    for key, value in expected.items():
        if prov.get(key) != value:
            raise ValueError(f"EXP132 provenance mismatch: {key}")
    if (summary.get("selected_epoch_1based") != protocol["common_args"]["epochs"]
            or summary.get("optimizer_updates") != protocol["fold_contract"]["steps"]
            or summary.get("val") != protocol["fold_contract"]["n_val"]
            or summary.get("auxiliary_parameters_exported") is not False
            or summary.get("auxiliary_batches") != (protocol["fold_contract"]["len_train_loader"] // 4) * protocol["common_args"]["epochs"]
            or not 0 <= summary.get("skipped_updates", -1) <= protocol["fold_contract"]["max_fp16_skips"]):
        raise ValueError("EXP132 result does not match frozen control schedule/export")
    return {str(run / name): digest(run / name) for name in (*required, "output-inventory.json")}


def screen(variant, root=ROOT):
    root = Path(root)
    out = root / "artifacts/experiments/EXP132_EXTERNAL_AUX" / "screen" / variant
    if out.exists():
        raise FileExistsError(f"refusing existing screen: {out}")
    pp = {k: {"candidate": {}, "control": {}} for k in ("k36", "k24")}
    bindings, ids = {}, None
    for seed in SEEDS:
        p = load_protocol(variant, seed, root)
        base = control.base_manifest(seed, root)
        val = control.read(root / base["folds"]["0"]["ordered_ids"]["path"])["val"]
        if len(val) != 888 or len(set(val)) != 888 or ids is not None and ids != val:
            raise ValueError("screen requires exact shared competition V888 roster")
        ids = val
        candidate = root / "artifacts/experiments" / p["tag"]
        control_run = root / "artifacts/experiments" / p["base_tag"]
        bindings.update(validate_run(candidate, p, root))
        bindings.update(control.verify_run(control_run, base, control.BASES[seed][1], root, candidate=False))
        bindings[str(protocol_path(variant, seed, root))] = digest(protocol_path(variant, seed, root))
        for k, filename in (("k36", "oof_k36.csv"), ("k24", "oof_best.csv")):
            pp[k]["candidate"][seed] = control.load_table(candidate / filename, ids, prediction=True)
            pp[k]["control"][seed] = control.load_table(control_run / filename, ids, prediction=True)
    refs = control.make_references({n: control.load_table(root / path, ids) for n, path in control.LABEL_PATHS.items()})
    draws = control.component_draws(ids, control.read(root / control.COMPONENTS), draws=control.DRAWS, seed=control.BOOT_SEED)
    pooled = [u for f in control.base_manifest(42, root)["folds"].values()
              for u in control.read(root / f["ordered_ids"]["path"])["val"]]
    if len(pooled) != 4343 or len(set(pooled)) != 4343:
        raise ValueError("partner anchor roster invalid")
    anchor_all = control.load_table(root / screen_math.PARTNER, pooled, prediction=True)
    ix = {u: i for i, u in enumerate(pooled)}
    anchor = anchor_all[[ix[u] for u in ids]]
    metrics = {k: screen_math.metric_contrast(v["candidate"], v["control"], refs, draws, anchor, k != "k36")
               for k, v in pp.items()}
    body = dict(schema="exp132-screen-v1", variant=variant, rows=len(ids), scope="F0 only",
                metrics=metrics, tiers=metrics["k36"]["tiers"], input_bindings=bindings,
                external_validation_rows=0, adoption_authorized=False,
                caveat="Tier A provisional F0; Tier B requires pooled five-fold two-seed evidence; partner report only")
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    lines = [f"# EXP132 {variant} competition-only F0 screen", "", body["caveat"], "",
             "| K | reference | candidate | control | delta | CI95 |", "|---|---|---:|---:|---:|---|"]
    for k, metric in metrics.items():
        for name, value in metric["seed_mean"].items():
            lines.append(f"| {k} | {name} | {value['candidate']:.6f} | {value['control']:.6f} | {value['delta']:+.6f} | {value['ci95']} |")
    lines.extend(["", "Tier verdicts: " + json.dumps(body["tiers"], sort_keys=True),
                  "Partner proxy: " + json.dumps(metrics["k36"]["partner"], sort_keys=True)])
    (out / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(out), tiers=body["tiers"], partner_delta=metrics["k36"]["partner"]["delta_partner"]), indent=2))
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("convert")
    p.add_argument("--source", choices=tuple(SOURCE_NAMES), required=True)
    p.add_argument("--shard-size", type=int, default=32)
    p = sub.add_parser("queue")
    p.add_argument("--variant", choices=tuple(VARIANT_SOURCES), required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, help="emit this seed only; protocol still registers both")
    p.add_argument("--after", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=12)
    p = sub.add_parser("screen")
    p.add_argument("--variant", choices=tuple(VARIANT_SOURCES), required=True)
    args = ap.parse_args(argv)
    if args.command == "convert":
        print(json.dumps(convert(args.source, shard_size=args.shard_size), indent=2))
    elif args.command == "queue":
        if args.seed is None:
            queue(args.variant, args.after, args.out, workers=args.workers)
        else:
            register(args.variant)
            item = dict(tag=tag_for(args.variant, args.seed), command=runner_argv(args.variant, args.seed, workers=args.workers),
                        continue_on_failure=True)
            if args.after.lower() != "none":
                item["after"] = args.after
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("x", encoding="utf-8") as f:
                json.dump([item], f, indent=2)
            print(json.dumps([item], indent=2))
    else:
        screen(args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
