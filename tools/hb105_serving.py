"""H-B-105/107 serving identity and MIL component timing, v1.0.0.

New wrapper only. No fitting, metric reading, downloads, account operations or
data transfer. CUDA is reachable only through the explicit `time` subcommand.
The timing experiment is not a complete release/runtime qualification.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
REG = "artifacts/experiments/EXP169_RELEASE_V6/v1.3.0/build-01/registration.json"
READOUT = ".local/hb105m-collected-v1/fivefold-q7-with-s-v1.json"
CORPUS = "storage/cache/derived/RAPTOR64_336_C140_S02"
ARCH = {"T": "convnext_tiny.fb_in22k_ft_in1k", "S": "convnext_small.fb_in22k_ft_in1k"}
SOURCES = ["tools/hb105_serving.py", "tools/exp169_release_v6.py",
           "tools/exp152_own0949_release.py", "training/arms/mil.py",
           "training/arms/lifecycle.py", "training/src/raptor_common.py",
           "training/src/raptor_exp185.py", "training/src/raptor_runtime.py",
           "config/workspace.toml"]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pin(p):
    p = Path(p)
    return dict(bytes=p.stat().st_size, sha256=sha(p))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def uid_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def validate_rosters(folds):
    need(len(folds) == 5, "five fold rosters required")
    universe = set().union(*(set(d["val"]) for d in folds))
    need(len(universe) == 4343 and sum(len(d["val"]) for d in folds) == 4343, "heldout partition")
    for d in folds:
        need(not set(d["train"]) & set(d["val"]) and set(d["train"]) | set(d["val"]) == universe
             and len(d["train"]) + len(d["val"]) == 4343, "train/val overlap, duplicate or coverage")
    return universe


def imports():
    # No availability probes, seeding, backend calls or CUDA APIs on CPU paths.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "2"
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    from training.arms import mil
    from tools import exp169_release_v6 as serving
    return np, torch, mil, serving


def relative(p, root):
    return Path(str(p).replace("\\", "/")).resolve().relative_to(root.resolve()).as_posix()


def meta(ck, entry, serving):
    need(ck.get("arch") == entry["arch"], "architecture differs: " + entry["id"])
    # v6 validates Tiny only; reuse every other metadata gate without editing it.
    serving.mil_meta({**ck, "arch": ARCH["T"]}, entry)


def load_model(path, entry, device, modules):
    _, torch, mil, serving = modules
    need(pin(path) == {k: entry[k] for k in ("bytes", "sha256")}, "checkpoint changed")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta(ck, entry, serving)
    model = mil.FindingMIL(mil.common.build_backbone(entry["arch"], pretrained=False))
    mil.strict_load(model, ck["model"])
    need(model.pool_policy == "finding_spatial_tokens_v1", "pooling differs")
    return model.eval().to(device)


def rank(values):
    import numpy as np
    from tools.exp152_own0949_release import percentile_average
    a = np.asarray(values, np.float64)
    need(a.ndim == 2 and a.shape[1] == 12 and len(a) and np.isfinite(a).all(), "rank input")
    return np.asarray([percentile_average(a[:, j].tolist()) for j in range(12)]).T


def alternatives(scores):
    """Healthy serving arithmetic, no fallback or new outer blend weights."""
    import numpy as np
    need(set(scores) == {"R50", "C4", "T", "S"}, "family set")
    need(scores["R50"].ndim == 3, "R50 dimensions")
    n = scores["R50"].shape[1]
    for k, count in (("R50", 10), ("C4", 5), ("T", 10), ("S", 10)):
        a = scores[k]
        need(a.dtype == np.float32 and a.shape == (count, n, 12) and np.isfinite(a).all()
             and ((a >= 0) & (a <= 1)).all(), "invalid family: " + k)
    r = scores["R50"]
    f = {"R50": rank(((r[:5] + r[5:]) / 2).mean(0)),
         **{k: rank(scores[k].mean(0)) for k in ("C4", "T", "S")}}
    return {k: rank(sum(rank(f[x]) for x in names) / len(names)) for k, names in {
        "Q7": ("R50", "C4", "T"), "Q7_add_S": ("R50", "C4", "T", "S"),
        "Q7_replace_T_with_S": ("R50", "C4", "S")}.items()}


def own_alternatives(s42, s43, scores, weight):
    """Existing own-only B6 menu; leave alternative/weight selection to B."""
    import numpy as np
    need(type(weight) is float and weight in (.30, .50), "unregistered outer weight")
    need(s42.shape == s43.shape and s42.ndim == 2 and s42.shape[1] == 12
         and all(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() for a in (s42, s43)), "own anchors")
    own = rank((s42.astype(np.float64) + s43.astype(np.float64)) / 2)
    bundles = alternatives(scores)
    need(all(q.shape == own.shape for q in bundles.values()), "anchor/cohort dimensions")
    return {name: (1 - weight) * own + weight * rank(q) for name, q in bundles.items()}


def build(out):
    need(not Path(out).exists(), "manifest destination exists")
    np, torch, mil, serving = imports()
    root = ROOT
    bindings = {}

    def bind(path):
        path = Path(path)
        bindings[path.relative_to(root).as_posix()] = pin(path)
        return read(path)

    reg = bind(root / REG)
    saved = bind(root / READOUT)
    need(saved["status"] == "COMPLETE_OOF_READOUT", "readout incomplete")
    # Bind saved metadata and sources; do not load labels or prediction rows.
    checked_readout = 0
    for path, digest in saved["bindings"].items():
        if path.endswith(".json"):
            need(sha(root / path) == digest, "saved readout binding changed: " + path)
            checked_readout += 1
    folds = [bind(root / f"artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold{f}/ordered-ids.json")
             for f in range(5)]
    validate_rosters(folds)
    members = []

    def member(family, seed, fold, path, spec_path=None, deployed=None):
        p = root / path
        inventory = bind(p.parent / "output-inventory.json")
        actual = pin(p)
        need(actual == inventory["model_best.pt"], "inventory checkpoint mismatch: " + path)
        prov = bind(p.parent / "provenance.json")
        need(sha(p.parent / "provenance.json") == inventory["provenance.json"]["sha256"], "provenance inventory")
        train_hash = prov["train_ids_ordered_sha256"]
        need(train_hash in [uid_hash(folds[fold]["train"]), uid_hash(sorted(folds[fold]["train"]))]
             and prov["eval_ids_ordered_sha256"] == uid_hash(folds[fold]["val"]), "ancestry mismatch")
        entry = dict(id=f"{family}-s{seed}-f{fold}", family=family, seed=seed, fold=fold,
                     path=path, **actual, train_uid_sha256=train_hash,
                     heldout_uid_sha256=prov["eval_ids_ordered_sha256"],
                     train_count=len(folds[fold]["train"]), heldout_count=len(folds[fold]["val"]))
        ck = torch.load(p, map_location="cpu", weights_only=False)
        entry["arch"] = ck["arch"]
        need(ck["fold"] == fold, "checkpoint fold")
        if spec_path:
            sp = root / spec_path
            spec = bind(sp)
            rc = bind(p.parent.parent / "receipt.json")
            coll = bind(p.parent.parent / "collection.json")
            summary = bind(p.parent / "summary.json")
            need(spec["seed"] == seed and spec["fold"] == fold, "spec roster")
            need(rc["status"] == "ARTIFACTS_VERIFIED" and rc["exit_code"] == 0
                 and rc["metrics"].get("engineering_smoke", 0) == 0, "incomplete/scientific receipt")
            need(coll["status"] == "COLLECTED_ARTIFACTS_VERIFIED"
                 and coll["receipt_sha256"] == sha(p.parent.parent / "receipt.json"), "collection receipt")
            need(rc["spec_sha256"] == coll["spec_sha256"] == prov["spec_sha256"] == sha(sp)
                 and rc["outputs"]["model_best.pt"] == actual["sha256"]
                 and rc["verification_sha256"] == prov["verification_sha256"], "receipt binding")
            for name in ("summary.json", "provenance.json", "output-inventory.json"):
                need(rc["outputs"][name] == sha(p.parent / name), "receipt metadata changed: " + name)
            need(summary["status"] == "FIXED_FINAL_COMPLETE" and summary["epochs"] == 20
                 and summary["attempted_updates"] == spec["config"]["updates"]["steps"]
                 and summary["skipped_updates"] <= spec["config"]["updates"]["max_skips"]
                 and summary["spec_sha256"] == sha(sp) and prov["checkpoint_selection"] == "fixed-final"
                 and (summary["train"], summary["val"]) == (len(folds[fold]["train"]), len(folds[fold]["val"])), "fit completion")
            need(train_hash == spec["config"]["roots"]["train_sha256"]
                 and prov["eval_ids_ordered_sha256"] == spec["config"]["roots"]["val_sha256"], "spec ancestry")
            entry["spec_sha256"] = sha(sp)
        else:
            need(ck.get("seed", ck.get("provenance", {}).get("seed")) == seed, "checkpoint seed")
        if family in ARCH:
            need(entry["arch"] == ARCH[family], "wrong family architecture")
            meta(ck, entry, serving)
        if deployed:
            entry["deployed_export"] = deployed
            need(actual["sha256"] == deployed["source_sha256"], "release source binding")
        members.append(entry)
        return entry

    partner_path = "artifacts/releases/EXP169_B172_v2_0_2_stage/assets/release-manifest.json"
    c4_path = "artifacts/releases/EXP169_V4_A4_B4_s43_stage_v413/c4/release-manifest.json"
    partner, c4 = bind(root / partner_path), bind(root / c4_path)
    need(sha(root / partner_path) == reg["assets"]["partner"]["manifest_sha256"]
         and sha(root / c4_path) == reg["assets"]["c4"]["manifest_sha256"], "release asset manifest")
    for family, entries in (("R50", [e for e in partner["entries"] if e["family"] == "l2"]),
                            ("C4", c4["entries"])):
        for e in entries:
            member(family, e["seed"], e["fold"], e["source"].replace("\\", "/"), deployed=e)
    for e in reg["mil"]:
        path = relative(e["path"], root)
        sp = relative(e["spec_pin"]["path"], root) if e.get("spec_pin") else None
        m = member("T", e["seed"], e["fold"], path, sp)
        need(m["sha256"] == e["sha256"] and m["bytes"] == e["bytes"], "deployed Tiny identity")
    for seed in (42, 43):
        for fold in range(5):
            folder = f".local/hb105m-collected-v1/s{seed}-f{fold}" if fold else f".local/hb105-collected-v1/s{seed}"
            spec = f"training/configs/hb105-member-s{seed}-f{fold}-v1.json" if fold else f"training/configs/hb105-small-s{seed}-f0-v1.json"
            member("S", seed, fold, folder + "/outputs/model_best.pt", spec)
    need(len(members) == 35 and len({m["sha256"] for m in members}) == 35, "member coverage")
    for family, seeds in (("R50", (42, 43)), ("C4", (42,)), ("T", (42, 43)), ("S", (42, 43))):
        need({(e["seed"], e["fold"]) for e in members if e["family"] == family}
             == {(s, f) for s in seeds for f in range(5)}, "seed/fold roster: " + family)
    members.sort(key=lambda e: (e["family"], e["seed"], e["fold"]))
    old_path = "artifacts/experiments/EXP185_FINDING_MIL/outputs/S43/results/EXP185_FINDING_MIL_F0_S43/model_best.pt"
    old = pin(root / old_path)
    need(old == bind(root / Path(old_path).parent / "output-inventory.json")["model_best.pt"], "old T43F0 checkpoint")
    new = next(e for e in members if e["id"] == "T-s43-f0")
    corpus = mil.common.Corpus(root / CORPUS)
    cm = bind(root / CORPUS / "manifest.json")
    for name in ("ids", "masks"):
        bindings[CORPUS + f"/{name}.npy"] = pin(root / CORPUS / f"{name}.npy")
        need(bindings[CORPUS + f"/{name}.npy"]["sha256"] == cm[name + "_sha256"], "corpus metadata")
    cohort = []
    for fold, doc in enumerate(folds):
        eligible = [u for u in doc["val"] if np.count_nonzero(corpus.mask(u)) >= 3]
        for direction in (1, -1):
            uid = min((u for u in eligible if u not in [c["uid"] for c in cohort]), key=lambda u:
                      (direction * int(np.count_nonzero(corpus.mask(u))), hashlib.sha256(u.encode()).hexdigest()))
            volume = corpus.volume(uid)
            need(np.any(volume), "selected metadata cohort has empty volume; do not silently replace")
            centers = mil.common.select_centers(corpus.mask(uid), 64, 36, corpus.spec, "within-slot-valid")
            x = serving.mil_windows(volume, centers)
            cohort.append(dict(uid=uid, fold=fold, valid_slices=int(np.count_nonzero(corpus.mask(uid))),
                               stratum="min_coverage" if direction == 1 else "max_coverage",
                               volume_sha256=hashlib.sha256(volume.tobytes()).hexdigest(),
                               windows_sha256=hashlib.sha256(x.tobytes()).hexdigest(), centers=list(map(int, centers)),
                               trained_members=[m["id"] for m in members if uid in folds[m["fold"]]["train"]]))
    for source in SOURCES:
        bindings[source] = pin(root / source)
    result = dict(version=VERSION, status="INPUTS_BOUND_RELEASE_HOLD", sources={p: bindings[p] for p in SOURCES},
                  members=members, bindings=bindings, corpus=dict(path=CORPUS,
                  metadata={n: pin(root / CORPUS / n) for n in ("manifest.json", "ids.npy", "masks.npy")}),
                  cohort=cohort, cohort_policy="Each heldout fold: min/max valid-mask coverage; SHA256(UID) tie-break; no labels/predictions",
                  alternatives={"Q7_add_S": ["R50", "C4", "T", "S"], "Q7_replace_T_with_S": ["R50", "C4", "S"]},
                  outer_blend=dict(status="B selection unbound; anchor bytes not rehashed by this component audit",
                  anchors={k: reg["assets"][k] for k in ("s42", "s43")},
                  formula="(1-w)*rank((own_s42.float64+own_s43.float64)/2)+w*rank(bundle); no final rerank",
                  existing_own_only_menu={k: v for k, v in reg["candidates"].items() if v["family"] == "B6"}),
                  audit=dict(actual_checkpoint_bytes_hashed=35, metadata_readout_bindings_checked=checked_readout,
                  disjoint_within_fold=True, complete_partition=4343, timing_cohort_is_holdout=False,
                  old_T_s43_f0=dict(path=old_path, **old), deployed_T_s43_f0=dict(sha256=new["sha256"], bytes=new["bytes"]),
                  T_s43_f0_byte_identical=old["sha256"] == new["sha256"],
                  saved_oof_order=saved["rank_policy"], deployed_order="FP32 R50 seed pairs then fold mean; C4/T/S direct member mean; family rank; equal family mean; bundle rank",
                  blockers=["Saved OOF Tiny S43/F0 differs from deployed v6 member" if old["sha256"] != new["sha256"] else "",
                            "Saved R50 OOF single-view; deployed R50 plane-flip mean probability",
                            "Saved OOF seed ranks differ from serving probability-before-rank nesting",
                            "No independent image/OAI panel exclusion evidence; no L7 or metric-independent holdout claim",
                            "Outer own-heavy candidate unselected; no new weights",
                            "MIL component timings exclude DICOM decode, R50/C4/anchor graph and release fallback; full candidate capacity/runtime unqualified",
                            "3090 measurements cannot certify 2xT4 <=8h target / 9h platform limit"]))
    write(out, result)
    return dict(status=result["status"], members=35, cohort=len(cohort), T_s43_f0_byte_identical=result["audit"]["T_s43_f0_byte_identical"])


def verify_sources(manifest):
    for path, expected in manifest["sources"].items():
        need(pin(ROOT / path) == expected, "source changed: " + path)


def verify_corpus(manifest, directory, modules):
    np, _, mil, serving = modules
    for name, expected in manifest["corpus"]["metadata"].items():
        need(pin(Path(directory) / name) == expected, "corpus metadata changed: " + name)
    corpus = mil.common.Corpus(directory)
    for c in manifest["cohort"]:
        need(c["uid"] in corpus, "cohort missing UID")
        vol = corpus.volume(c["uid"])
        need(hashlib.sha256(vol.tobytes()).hexdigest() == c["volume_sha256"], "cohort volume changed")
        centers = mil.common.select_centers(corpus.mask(c["uid"]), 64, 36, corpus.spec, "within-slot-valid")
        need(list(centers) == c["centers"], "center selection changed")
        need(hashlib.sha256(serving.mil_windows(vol, centers).tobytes()).hexdigest() == c["windows_sha256"], "windows changed")
    return corpus


def discover(manifest, roots):
    """Read-only, bounded discovery. Metadata names only; no imports of torch."""
    wanted = [e for e in manifest["members"] if e["family"] in ARCH]
    sizes = {e["bytes"] for e in wanted}
    by_digest, corpora = {}, []
    seen = set()
    for root in map(Path, roots):
        need(root.is_dir(), "unverified/missing search root: " + str(root))
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {".git", ".ssh", ".kaggle", ".codex", "api-keys", ".relabel"}
                             and not (Path(directory) / d).is_symlink())
            for name in sorted(files):
                p = Path(directory) / name
                if p.is_symlink() or str(p.resolve()) in seen:
                    continue
                seen.add(str(p.resolve()))
                if name == "manifest.json" and p.parent.name == "RAPTOR64_336_C140_S02":
                    if pin(p) == manifest["corpus"]["metadata"]["manifest.json"]:
                        corpora.append(str(p.parent.resolve()))
                elif (name == "model_best.pt" or name.startswith("mil-")) and p.suffix == ".pt" and p.stat().st_size in sizes:
                    digest = sha(p)
                    by_digest.setdefault(digest, []).append(str(p.resolve()))
    missing = [e["id"] for e in wanted if e["sha256"] not in by_digest]
    paths = {e["id"]: sorted(by_digest[e["sha256"]])[0] for e in wanted if e["sha256"] in by_digest}
    # Identical duplicate bytes are safe aliases; preserve every match for review.
    return dict(status="MATCHED_METADATA_ONLY" if not missing and corpora else "HOLD_MISSING_INPUTS",
                missing_members=missing, corpus_candidates=sorted(corpora),
                corpus=sorted(corpora)[0] if corpora else None, members=paths,
                identical_aliases={e["id"]: sorted(by_digest.get(e["sha256"], [])) for e in wanted})


def preflight(manifest_path, roots):
    m = read(manifest_path)
    verify_sources(m)
    return dict(manifest_sha256=sha(manifest_path), **discover(m, roots))


def prediction(model, x, uid, corpus, device, modules):
    """Instrument exact v6 transfer/FP16/FP32 sigmoid/flip probability order."""
    np, torch, mil, serving = modules
    sync = (lambda: torch.cuda.synchronize(device)) if device == "cuda:0" else (lambda: None)
    def timed(call):
        sync(); start = time.perf_counter(); v = call(); sync()
        return v, time.perf_counter() - start
    flipped, flip_s = timed(lambda: mil.flip_inputs(torch.from_numpy(x)[None], [uid], corpus, 36)[0])
    out, measures = [], dict(flip_cpu_s=flip_s, transfer_s=0., forward_s=0., sigmoid_d2h_s=0.)
    for view in (x, flipped):
        tensor, seconds = timed(lambda: (view if torch.is_tensor(view) else torch.from_numpy(np.ascontiguousarray(view))).to(device))
        measures["transfer_s"] += seconds
        amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda:0" else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            logits, seconds = timed(lambda: model(tensor[None]))
        measures["forward_s"] += seconds
        need(tuple(logits.shape) == (1, 12) and bool(torch.isfinite(logits).all()), "nonfinite/shape")
        prob, seconds = timed(lambda: torch.sigmoid(logits.float())[0].cpu().numpy())
        measures["sigmoid_d2h_s"] += seconds
        out.append(serving.probability(prob, (12,), "member"))
    return (out[0] + out[1]) * .5, measures


def budget(credit, hourly, hours):
    need(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (credit, hourly, hours)), "finite positive budget inputs required")
    need(credit - hourly * hours >= 5, "reserve below $5; no top-up")
    return dict(credit_usd=credit, hourly_usd=hourly, approved_hours=hours,
                estimated_max_cost_usd=hourly * hours, reserve_usd=5,
                note="B supplies fresh account values incl. storage/idle charges; estimate, not measured billing")


def run(manifest_path, out, *, resolution=None, family=None, credit=None, hourly=None, hours=None):
    m = read(manifest_path)
    verify_sources(m)
    cuda = resolution is not None
    b = budget(credit, hourly, hours) if cuda else None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(version=VERSION, manifest_sha256=sha(manifest_path), status="STARTED",
                   device="cuda:0" if cuda else "cpu", missing=0, failures=0, fallback_count=0,
                   budget=b, scope="MIL component only; not complete candidate", measurements=[],
                   cold_definition="New process/imports/model load and first forward; OS disk cache uncontrolled; cohort verification prewarms volumes",
                   warm_definition="Repeats 1 and 2 after all cohort/model combinations have run; repeat 0 excluded from warm throughput")
    write(out / "started.json", receipt)
    start = time.perf_counter()
    try:
        modules = imports()
        np, torch, mil, serving = modules
        receipt["python_start_to_imports_s"] = time.perf_counter() - START
        receipt["runtime"] = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__)
        import timm
        receipt["runtime"]["timm"] = timm.__version__
        if cuda:
            resolved = read(resolution)
            need(resolved["status"] == "MATCHED_METADATA_ONLY" and resolved["manifest_sha256"] == sha(manifest_path), "preflight not matched")
            receipt["resolution_sha256"] = sha(resolution)
            need(family in ARCH, "one family per process")
            # B-run-only boundary. CPU paths never execute these calls.
            init = time.perf_counter()
            need(torch.cuda.device_count() == 1, "exactly one visible GPU required")
            props = torch.cuda.get_device_properties(0)
            need("3090" in props.name, "this card is for retained RTX3090 only")
            serving.backend(strict=True)
            torch.cuda.synchronize()
            receipt["cuda_startup_s"] = time.perf_counter() - init
            receipt["gpu"] = dict(name=props.name, total_bytes=props.total_memory)
            torch.cuda.reset_peak_memory_stats()
            corpus_path = resolved["corpus"]
            paths = resolved["members"]
            entries = [e for e in m["members"] if e["family"] == family]
            need(len(entries) == 10, "incomplete family")
            cohort, passes = m["cohort"], 3
        else:
            corpus_path = ROOT / m["corpus"]["path"]
            entries = [e for e in m["members"] if e["id"] in ("T-s42-f0", "S-s42-f0")]
            paths = {e["id"]: ROOT / e["path"] for e in entries}
            cohort, passes = m["cohort"][:1], 1
        t = time.perf_counter()
        corpus = verify_corpus(m, corpus_path, modules)
        receipt["corpus_and_all_cohort_verification_s"] = time.perf_counter() - t
        models = []
        for entry in entries:
            need(not cuda or time.perf_counter() - start < hours * 3600, "B-approved time exhausted before load")
            t = time.perf_counter()
            model = load_model(paths[entry["id"]], entry, receipt["device"], modules)
            if cuda:
                torch.cuda.synchronize()
            receipt["measurements"].append(dict(event="load", member=entry["id"],
                hash_deserialize_construct_strict_load_device_s=time.perf_counter() - t))
            models.append(model)
        receipt["resident_models"] = len(models)
        if cuda:
            receipt["resident_allocated_bytes"] = torch.cuda.memory_allocated()
        predictions = np.empty((passes, len(entries), len(cohort), 12), np.float32)
        max_error = 0.
        events = out / "events.jsonl"
        with events.open("x", encoding="utf-8") as log:
            for repeat in range(passes):
                for i, c in enumerate(cohort):
                    need(not cuda or time.perf_counter() - start < hours * 3600, "B-approved compute estimate exhausted; preserve evidence")
                    t = time.perf_counter()
                    volume = corpus.volume(c["uid"])
                    # Touch mmap bytes so I/O does not hide inside the window timer.
                    volume = np.array(volume, copy=True)
                    read_s = time.perf_counter() - t
                    t = time.perf_counter()
                    centers = mil.common.select_centers(corpus.mask(c["uid"]), 64, 36, corpus.spec, "within-slot-valid")
                    x = serving.mil_windows(volume, centers)
                    preprocess_s = time.perf_counter() - t
                    for j, (entry, model) in enumerate(zip(entries, models)):
                        need(not cuda or time.perf_counter() - start < hours * 3600, "B-approved time exhausted before forward")
                        t = time.perf_counter()
                        p, measure = prediction(model, x, c["uid"], corpus, receipt["device"], modules)
                        wall = time.perf_counter() - t
                        predictions[repeat, j, i] = p
                        # First invocation of each model: same-data reference parity.
                        # Reference work is measured separately, excluded from throughput.
                        reference_s = None
                        if repeat == 0 and i == 0:
                            t = time.perf_counter()
                            reference = serving.mil_prediction(model, x, c["uid"], corpus, receipt["device"])
                            reference_s = time.perf_counter() - t
                            err = float(np.max(np.abs(reference - p)))
                            max_error = max(max_error, err)
                            need(err <= 1e-6, "same-process serving parity >1e-6")
                        row = dict(member=entry["id"], cohort_index=i, repeat=repeat,
                                   first_model_forward=repeat == 0 and i == 0, **measure,
                                   mil_tta_wall_s=wall, reference_parity_s=reference_s,
                                   cache_read_s=read_s if j == 0 else 0.,
                                   preprocess_s=preprocess_s if j == 0 else 0.)
                        log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
        receipt["max_reference_abs_error"] = max_error
        receipt["repeat_max_abs_error"] = float(np.max(np.abs(predictions - predictions[:1])))
        need(receipt["repeat_max_abs_error"] <= 1e-6, "repeat parity >1e-6")
        if cuda:
            receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            receipt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        t = time.perf_counter()
        with (out / "probabilities.npz").open("xb") as f:
            np.savez(f, probabilities=predictions, member_ids=[e["id"] for e in entries], ids=[c["uid"] for c in cohort])
        receipt["output_write_s"] = time.perf_counter() - t
        receipt["total_s"] = time.perf_counter() - start
        if cuda:
            receipt["estimated_elapsed_cost_usd"] = receipt["total_s"] / 3600 * hourly
        verify_sources(m)
        receipt["status"] = "COMPONENT_PARITY_COMPLETE" if cuda else "CPU_REAL_DATA_SMOKE_COMPLETE"
        receipt["files"] = {p.name: pin(p) for p in out.iterdir() if p.is_file()}
        write(out / "receipt.json", receipt)
        return {k: receipt[k] for k in ("status", "device", "resident_models", "max_reference_abs_error", "total_s")}
    except BaseException as exc:
        receipt.update(status="FAILED_PRESERVED", failures=1, error=str(exc), traceback=traceback.format_exc())
        write(out / "failure.json", receipt)
        raise


def compare(manifest, tiny, small, out):
    """CPU-only accounting of completed component receipts, never an admission."""
    import numpy as np
    m = read(manifest)
    verify_sources(m)
    reports, arrays = {}, {}
    for family, folder in (("T", tiny), ("S", small)):
        folder = Path(folder)
        r = read(folder / "receipt.json")
        need(r["manifest_sha256"] == sha(manifest) and r["status"] == "COMPONENT_PARITY_COMPLETE"
             and r["device"] == "cuda:0" and r["resident_models"] == 10, "incomplete/wrong timing receipt")
        need(r["failures"] == r["missing"] == r["fallback_count"] == 0
             and r["max_reference_abs_error"] <= 1e-6 and r["repeat_max_abs_error"] <= 1e-6, "failed parity")
        for name, expected in r["files"].items():
            need(Path(name).name == name and pin(folder / name) == expected, "timing evidence changed")
        with np.load(folder / "probabilities.npz", allow_pickle=False) as z:
            need(z["ids"].tolist() == [c["uid"] for c in m["cohort"]]
                 and z["member_ids"].tolist() == [e["id"] for e in m["members"] if e["family"] == family], "timing roster/order")
            a = z["probabilities"]
            need(a.shape == (3, 10, len(m["cohort"]), 12) and a.dtype == np.float32
                 and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), "timing probability array")
            arrays[family] = a
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        expected_keys = {(rep, i, e["id"]) for rep in range(3) for i in range(len(m["cohort"]))
                         for e in m["members"] if e["family"] == family}
        need(len(events) == len(expected_keys) and
             {(e["repeat"], e["cohort_index"], e["member"]) for e in events} == expected_keys, "timing events incomplete")
        warm = [e for e in events if e["repeat"] > 0]
        fields = ("cache_read_s", "preprocess_s", "flip_cpu_s", "transfer_s", "forward_s", "sigmoid_d2h_s", "mil_tta_wall_s")
        need(all(math.isfinite(e[k]) and e[k] >= 0 for e in events for k in fields), "invalid timing values")
        reports[family] = dict(receipt_sha256=sha(folder / "receipt.json"), gpu=r["gpu"], runtime=r["runtime"],
            cold_definition=r["cold_definition"], python_start_to_imports_s=r["python_start_to_imports_s"],
            cuda_startup_s=r["cuda_startup_s"], model_load_s=sum(e["hash_deserialize_construct_strict_load_device_s"] for e in r["measurements"]),
            first_forward_s=sum(e["mil_tta_wall_s"] for e in events if e["first_model_forward"]),
            warm_per_study={k: sum(e[k] for e in warm) / (2 * len(m["cohort"])) for k in fields},
            resident_allocated_bytes=r["resident_allocated_bytes"], peak_allocated_bytes=r["peak_allocated_bytes"],
            peak_reserved_bytes=r["peak_reserved_bytes"], total_measured_s=r["total_s"],
            estimated_elapsed_cost_usd=r["estimated_elapsed_cost_usd"])
    need(reports["T"]["gpu"] == reports["S"]["gpu"] and reports["T"]["runtime"] == reports["S"]["runtime"], "runtime/device mismatch")
    result = dict(status="COMPONENT_COMPARISON_ONLY_FULL_RELEASE_HOLD", manifest_sha256=sha(manifest), families=reports,
        warm_replacement_mil_delta_s_per_study=reports["S"]["warm_per_study"]["mil_tta_wall_s"]-reports["T"]["warm_per_study"]["mil_tta_wall_s"],
        estimated_addition_mil_increment_s_per_study=reports["S"]["warm_per_study"]["mil_tta_wall_s"],
        estimated_combined_mil_weight_residency_bytes=sum(r["resident_allocated_bytes"] for r in reports.values()),
        S_T_mean_abs_probability_difference=float(np.abs(arrays["S"][0].mean(0)-arrays["T"][0].mean(0)).mean()),
        limitation="Isolated 3090 MIL families. Addition estimate assumes shared windows and ignores co-residency effects. No full candidate/2xT4 certificate, no metric or independent-holdout claim, no measured billing.")
    write(out, result)
    return result


def package(manifest, out):
    m = read(manifest)
    verify_sources(m)
    files = {p: ROOT / p for p in SOURCES}
    files["input-manifest.json"] = Path(manifest)
    folder = ROOT / "artifacts/experiments/HB105_SERVING_V1"
    for name in ("LAUNCH.md", "REPORT.md"):
        files[name] = folder / name
    need(sum(p.stat().st_size for p in files.values()) < 2_000_000, "package unexpectedly large")
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, p in sorted(files.items()):
            z.writestr(name, p.read_bytes())
        for name in ("tools/__init__.py", "training/__init__.py", "training/arms/__init__.py", "training/src/__init__.py"):
            z.writestr(name, b"")
    return pin(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", required=True)
    v = sub.add_parser("preflight"); v.add_argument("--manifest", required=True); v.add_argument("--roots", nargs="+", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--manifest", required=True); s.add_argument("--out", required=True)
    t = sub.add_parser("time")
    for arg in ("manifest", "resolution", "out", "family"):
        t.add_argument("--" + arg, required=True)
    for arg in ("credit", "hourly", "hours"):
        t.add_argument("--" + arg, type=float, required=True)
    k = sub.add_parser("package"); k.add_argument("--manifest", required=True); k.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    for arg in ("manifest", "tiny", "small", "out"):
        c.add_argument("--" + arg, required=True)
    bp = sub.add_parser("budget")
    for arg in ("credit", "hourly", "hours"):
        bp.add_argument("--" + arg, type=float, required=True)
    a = p.parse_args()
    if a.cmd == "build":
        result = build(a.out)
    elif a.cmd == "preflight":
        result = preflight(a.manifest, a.roots)
    elif a.cmd == "smoke":
        result = run(a.manifest, a.out)
    elif a.cmd == "time":
        result = run(a.manifest, a.out, resolution=a.resolution, family=a.family,
                     credit=a.credit, hourly=a.hourly, hours=a.hours)
    elif a.cmd == "compare":
        result = compare(a.manifest, a.tiny, a.small, a.out)
    elif a.cmd == "budget":
        result = budget(a.credit, a.hourly, a.hours)
    else:
        result = package(a.manifest, a.out)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("status") == "HOLD_MISSING_INPUTS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
