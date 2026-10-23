"""EXP130 Gold58 diagnostic for the OAI 0.949 checkpoint and local RadR50.

Build changes only the fidelity notebook's study source and output contract. Its
reader, model, K94 windowing and anatomical mirror implementation stay intact.
The notebook is untrusted third-party code: build/inspect locally, run only on
Kaggle. Gold58 is exposed and selected against by the OAI author; report only.

  python tools/exp130_gold58_compare.py build
  python tools/exp130_gold58_compare.py predict --device cpu
  python tools/exp130_gold58_compare.py compare --oai .local/exp130-gold58/oai-output/gold58_oai.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".local/exp130-gold58"
FIDELITY = ROOT / ".local/exp128-oai-light/fidelity"
UID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FIDELITY_SHA256 = "ba3755221fbf4541ab503d25387320031f1188dd5aef284679b7ea512394a6aa"
KERNEL_ID = "taeyangg4/knee-exp130-oai-gold58"
SEEDS = {42: ("EXP113_RADR50_LOCAL_PROTOCOL_V1", "EXP113_RADR50_LOCAL_F"),
         43: ("EXP118_RADR50_LOCAL_S43_PROTOCOL_V1", "EXP118_RADR50_LOCAL_S43_F")}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(body)


def official_gold(root=ROOT):
    """Return ordered UIDs and binary labels. Never emit official CSV rows."""
    with (Path(root) / "storage/competition/train.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames is not None and set([UID, *TARGETS]) <= set(reader.fieldnames),
             "official train.csv columns differ")
        rows = [row for row in reader if (row["ACL"] or "").strip()]
    ids = [row[UID] for row in rows]
    need(len(ids) == 58 and len(set(ids)) == 58 and all(ids), "expected 58 unique Gold UIDs")
    labels = []
    for row in rows:
        vals = [float(row[t]) for t in TARGETS]
        need(all(v in (0.0, 1.0) for v in vals), "Gold labels must be binary and complete")
        labels.append([int(v) for v in vals])
    return ids, labels


def replace_once(source, old, new):
    need(source.count(old) == 1, f"fidelity source drift at {old[:70]!r}")
    return source.replace(old, new, 1)


def gold_kernel_source(source, ids):
    """Surgical main-body retarget; all original preprocessing/model functions unchanged."""
    need(len(ids) == 58 and len(set(ids)) == 58, "Gold UID constant must have 58 unique IDs")
    prefix, sep, main = source.partition("def main():\n")
    need(bool(sep) and main.rstrip().endswith("main()"), "fidelity main entry changed")
    uid_constant = "GOLD58_UIDS = " + repr(tuple(ids)) + "\n\n"
    main = replace_once(main, 'tsdir = ROOT + "/test_series"', 'tsdir = ROOT + "/train_series"')
    main = replace_once(main, 'tsdir = ROOT + "/test_images"', 'tsdir = ROOT + "/train_images"')
    main = replace_once(main,
        'test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)\n'
        '    test_ids = test["StudyInstanceUID"].tolist()',
        'test_ids = list(GOLD58_UIDS)\n'
        '    assert len(test_ids) == 58 and len(set(test_ids)) == 58')
    main = replace_once(main, 'pd.read_csv(ROOT + "/test_series.csv")',
                        'pd.read_csv(ROOT + "/train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})')
    main = replace_once(main, 'SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}',
                        'tser = tser[tser["StudyInstanceUID"].isin(test_ids)]\n'
                        '    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}')
    main = replace_once(main, 'print("test root:", ROOT, "| series dir:", tsdir, flush=True)',
                        'print("train image root:", ROOT, "| series dir:", tsdir, flush=True)')
    main = replace_once(main, 'print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)',
                        'print(f"Gold studies {len(test_ids)} | Gold series {len(tser)}", flush=True)')
    main = replace_once(main, 'np.full((N, len(LAB)), 0.5, np.float32)',
                        'np.full((N, len(LAB)), np.nan, np.float32)')
    main = replace_once(main, 'models = {}\n',
        'assert ngpu == 2 and all("T4" in torch.cuda.get_device_name(g) for g in range(ngpu)), "requires 2 x T4"\n'
        '    models = {}\n')
    main = replace_once(main, 'q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()',
                        't_loaded = time.time()\n'
                        '    q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; failures = []; lock = threading.Lock()')
    main = replace_once(main, 'sid, vol, mask = item; i = pos[sid]\n            try:',
                        'sid, vol, mask = item; i = pos[sid]\n            try:\n'
                        '                if not mask.any(): raise RuntimeError("empty or failed DICOM preparation")')
    main = replace_once(main,
        'print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)',
        'with lock: failures.append(i)\n'
        '                print(f"  [gpu {d}] study {i} FAILED ({type(e).__name__}: {e})", flush=True)')
    main = replace_once(main, 'for d in devs:\n        for m, _ in models[d]: del m',
                        't_gpu_done = time.time()\n'
                        '    if failures or not np.isfinite(arm_probs[0]).all():\n'
                        '        raise RuntimeError(f"Gold58 inference failed on {len(failures)} studies")\n'
                        '    for d in devs:\n        for m, _ in models[d]: del m')
    main = replace_once(main, 'print(f"[pre] done: {N} studies in {time.time()-t_pre:.0f}s',
                        't_pre_done = time.time()\n'
                        '    print(f"[pre] done: {N} studies in {t_pre_done-t_pre:.0f}s')
    begin = main.index('    # WEIGHTED rank-mean blend across the test set')
    end = main.index('\n\nif __name__ == "__main__":', begin)
    main = main[:begin] + '''    # Preserve pre-rank sigmoid probabilities for a 58-study AUC and later rank blends.
    assert A == 1 and arm_probs[0].shape == (58, len(LAB))
    probs = arm_probs[0]
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    out = "/kaggle/working/gold58_oai.csv"
    sub = pd.DataFrame(probs, columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    assert sub["StudyInstanceUID"].tolist() == test_ids
    sub.to_csv(out, index=False)
    elapsed = time.time() - t0
    timing = dict(schema="exp130-gold58-timing-v1", studies=N, k_eval=K_EVAL,
                  mirror=True, gpus=[torch.cuda.get_device_name(g) for g in range(ngpu)],
                  seconds_total=elapsed, seconds_per_study=elapsed/N,
                  projected_1300_seconds_at_observed_rate=elapsed/N*1300,
                  seconds_load=t_loaded-t0, seconds_preparation=t_pre_done-t_pre,
                  seconds_until_gpu_done=t_gpu_done-t0,
                  output_sha256=__import__("hashlib").sha256(open(out, "rb").read()).hexdigest(),
                  failures=len(failures), probabilities="sigmoid pair mean before cohort ranks")
    with open("/kaggle/working/gold58_timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    print("Gold58 predictions", N, "timing seconds/study", timing["seconds_per_study"], flush=True)
''' + main[end:]
    return prefix + uid_constant + 'def main():\n' + main


def build(root=ROOT, dest=BASE / "oai_gold58"):
    root, dest = Path(root), Path(dest)
    source_path = root / ".local/exp128-oai-light/fidelity/release.ipynb"
    need(sha256(source_path) == FIDELITY_SHA256, "fidelity notebook hash differs")
    nb = json.loads(source_path.read_text(encoding="utf-8"))
    need(len(nb["cells"]) == 9, "fidelity notebook cell count differs")
    ids, _labels = official_gold(root)
    code = copy.deepcopy(nb["cells"][6])
    code["source"] = gold_kernel_source("".join(code["source"]), ids).splitlines(keepends=True)
    code["outputs"] = []
    code["execution_count"] = None
    nb["cells"] = [*copy.deepcopy(nb["cells"][:6]), code]
    nb["cells"][1]["source"] = ["# EXP130 Gold58 diagnostic\n\n",
        "Source and OAI acknowledgement from the EXP128 fidelity credit are preserved below. ",
        "Gold58 is exposed and the OAI checkpoint used it for epoch selection. ",
        "This run is report only and never a selection gate.\n\n",
        *nb["cells"][1]["source"]]
    meta = json.loads((root / ".local/exp128-oai-light/fidelity/kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = "knee-exp130-oai-gold58"
    meta["machine_shape"] = "NvidiaTeslaT4"
    need(meta["is_private"] is True and meta["enable_gpu"] is True and
         meta["enable_internet"] is False and len(meta["dataset_sources"]) == 1 and
         len(meta["competition_sources"]) == 1 and bool(meta["docker_image"]),
         "fidelity metadata drift")
    need(not dest.exists(), "kernel directory already exists; refusing overwrite")
    write_new(dest / "release.ipynb", json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    write_new(dest / "kernel-metadata.json", json.dumps(meta, indent=2) + "\n")
    receipt = {"schema": "exp130-gold58-build-v1", "status": "UNPUSHED", "gold_uids": len(ids),
               "uid_source": "official train.csv rows with non-null ACL; UIDs only embedded",
               "source_fidelity_sha256": FIDELITY_SHA256,
               "notebook_sha256": sha256(dest / "release.ipynb"),
               "metadata_sha256": sha256(dest / "kernel-metadata.json"),
               "core_reader_model_windows_mirror": "unchanged fidelity code before def main()"}
    write_new(dest / "build-receipt.json", json.dumps(receipt, indent=2) + "\n")
    return receipt


def prediction_csv(path, ids):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames == [UID, *TARGETS], f"prediction columns differ: {path}")
        rows = list(reader)
    got = [row[UID] for row in rows]
    need(got == ids, f"prediction UID order/cohort differs: {path}")
    values = [[float(row[t]) for t in TARGETS] for row in rows]
    need(all(math.isfinite(x) and 0 <= x <= 1 for row in values for x in row),
         f"prediction probabilities invalid: {path}")
    return values


def write_predictions(path, ids, values):
    need(len(ids) == len(values) == 58 and all(len(row) == len(TARGETS) for row in values),
         "prediction shape differs")
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([UID, *TARGETS])
        writer.writerows([[uid, *map(float, row)] for uid, row in zip(ids, values)])


def predict(root=ROOT, out=BASE / "local", device="cpu", workers=0):
    """Ten trusted own checkpoints; import torch only when inference is requested."""
    import sys
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    root, out = Path(root), Path(out)
    need(device in ("cpu", "cuda") and workers >= 0, "invalid device/workers")
    need(not out.exists(), "local output directory already exists; refusing overwrite")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "training/src"))
    from tools import exp124_flip_tta as flip
    from training.src.raptor_common import Corpus, build_model
    from training.src.raptor_radimagenet import CenterSliceWindows
    from training.src.raptor_runtime import configure_runtime
    ids, _labels = official_gold(root)
    configure_runtime("none" if device == "cpu" else "fp16", device, policy="strict")
    outputs, provenance = {}, {}
    for seed, (protocol_name, prefix) in SEEDS.items():
        ppath = root / "artifacts/experiments" / protocol_name / "manifest.json"
        p = json.loads(ppath.read_text(encoding="utf-8"))
        corpus = Corpus(root / p["paths"]["corpus"])
        need(corpus.contract == "raptor64_336_c140_s02" and
             sha256(corpus.dir / "manifest.json") == p["corpus_manifest_sha256"],
             "corpus contract/manifest differs")
        need(set(ids) <= set(corpus.ids), "Gold studies missing from local corpus")
        args = p["common_args"]
        need(args["window_policy"] == "within-slot-valid" and args["input_policy"] == "center3" and
             args["norm"] == "radimagenet" and args["res"] == 224 and p["seed"] == seed,
             "RadR50 evaluation recipe differs")
        ranges = flip.plane_ranges(corpus.spec)
        labels = {uid: np.zeros(len(TARGETS), np.float32) for uid in ids}
        ds = CenterSliceWindows(corpus.dir, ids, labels, 224, 36, False, args["norm"], seed,
                                window_policy=args["window_policy"], input_policy=args["input_policy"])
        loader = DataLoader(flip.PlaneAwareWindows(ds, corpus, ranges), batch_size=4, shuffle=False,
                            num_workers=workers, collate_fn=flip.collate_pair)
        plain_folds, tta_folds, hashes = [], [], []
        for fold in range(5):
            fold_ids_path = root / p["folds"][str(fold)]["ordered_ids"]["path"]
            fold_ids = json.loads(fold_ids_path.read_text(encoding="utf-8"))
            need(not (set(ids) & (set(fold_ids["train"]) | set(fold_ids["val"]))),
                 "Gold58 overlaps a RadR50 training/evaluation fold")
            ckpt_path = root / "artifacts/experiments" / f"{prefix}{fold}" / "model_best.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            need(ckpt.get("tag") == f"{prefix}{fold}" and ckpt.get("fold") == fold,
                 "checkpoint tag/fold differs")
            prov = ckpt.get("provenance") or {}
            need(prov.get("protocol_sha256") == sha256(ppath), "checkpoint protocol differs")
            model = build_model(args["arch"], pretrained=False)
            model.load_state_dict(ckpt["model"], strict=True)
            model = model.to(device).eval()
            a, b, got = flip.infer_pair(model, loader, device, "exp113")
            need(got == ids and a.shape == b.shape == (58, len(TARGETS)), "inference cohort/shape differs")
            plain_folds.append(a)
            tta_folds.append((a + b) / 2)
            hashes.append(sha256(ckpt_path))
            del model, ckpt
        outputs[f"s{seed}_plain"] = np.mean(plain_folds, axis=0).tolist()
        outputs[f"s{seed}_tta"] = np.mean(tta_folds, axis=0).tolist()
        provenance[str(seed)] = {"protocol_sha256": sha256(ppath), "checkpoints_sha256": hashes,
                                 "fold_mean": "probability arithmetic mean", "k_eval": 36,
                                 "tta": "EXP124 coronal/axial horizontal flip; sagittal unchanged"}
    out.mkdir(parents=True)
    for name, values in outputs.items():
        write_predictions(out / f"gold58_{name}.csv", ids, values)
    write_new(out / "receipt.json", json.dumps({"schema": "exp130-local-v1", "studies": 58,
              "device": device, "models": provenance,
              "outputs": {name: sha256(out / f"gold58_{name}.csv") for name in outputs}}, indent=2) + "\n")
    return {"studies": 58, "outputs": list(outputs)}


def rank_average(values):
    """Average-tie percentile ranks, as in EXP128's blend code."""
    need(bool(values) and all(math.isfinite(x) for x in values), "invalid ranks")
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    at = 0
    while at < len(order):
        end = at + 1
        while end < len(order) and values[order[end]] == values[order[at]]:
            end += 1
        rank = ((at + end - 1) / 2 + 1) / len(values)
        for i in order[at:end]:
            result[i] = rank
        at = end
    return result


def rank_matrix(rows):
    return [list(row) for row in zip(*(rank_average([r[t] for r in rows]) for t in range(len(TARGETS))))]


def mix(a, b, wa):
    need(len(a) == len(b), "blend rows differ")
    return [[wa*x + (1-wa)*y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def auc(y, score):
    need(len(y) == len(score), "AUC lengths differ")
    pos = sum(y)
    neg = len(y) - pos
    if not pos or not neg:
        return None
    ranks = rank_average(score)
    # rank_average is rank/n; multiply by n to get standard 1-based midranks.
    n = len(y)
    return (sum(r*n for r, label in zip(ranks, y) if label) - pos*(pos+1)/2) / (pos*neg)


def scores(labels, candidates, eligible, indices=None):
    indices = list(range(len(labels))) if indices is None else indices
    result = {}
    for name, values in candidates.items():
        per = {}
        for t, target in enumerate(TARGETS):
            y = [labels[i][t] for i in indices]
            x = [values[i][t] for i in indices]
            per[target] = auc(y, x)
        result[name] = {"per_target": per,
                        "macro": sum(per[TARGETS[t]] for t in eligible) / len(eligible)}
    return result


def bootstrap(labels, candidates, eligible, pairs, draws=1000, seed=20261008):
    rng = random.Random(seed)
    n = len(labels)
    diffs = {f"{a} - {b}": [] for a, b in pairs}
    rejected = 0
    while len(next(iter(diffs.values()))) < draws:
        indices = [rng.randrange(n) for _ in range(n)]
        if any(sum(labels[i][t] for i in indices) < 3 or
               sum(labels[i][t] for i in indices) > n-3 for t in eligible):
            rejected += 1
            continue
        sampled = scores(labels, candidates, eligible, indices)
        for a, b in pairs:
            diffs[f"{a} - {b}"].append(sampled[a]["macro"] - sampled[b]["macro"])
    def quantile(x, q):
        s = sorted(x)
        pos = q * (len(s)-1)
        lo = int(pos)
        return s[lo] + (s[min(lo+1, len(s)-1)]-s[lo])*(pos-lo)
    return {key: {"ci95": [quantile(vals, .025), quantile(vals, .975)],
                  "share_positive": sum(v > 0 for v in vals)/len(vals)} for key, vals in diffs.items()}, rejected


def compare(oai, local=BASE / "local", root=ROOT, out=BASE / "comparison", draws=1000):
    ids, labels = official_gold(root)
    local = Path(local)
    raw = {"oai_0949": prediction_csv(oai, ids)}
    for seed in SEEDS:
        for view in ("plain", "tta"):
            raw[f"s{seed}_{view}"] = prediction_csv(local / f"gold58_s{seed}_{view}.csv", ids)
    rank = {name: rank_matrix(rows) for name, rows in raw.items()}
    candidates = {"oai_0949": raw["oai_0949"],
                  "radr50_s42": raw["s42_plain"],
                  "radr50_s42_tta": raw["s42_tta"],
                  "radr50_2seed_rank": mix(rank["s42_plain"], rank["s43_plain"], .5),
                  "radr50_2seed_rank_tta": mix(rank["s42_tta"], rank["s43_tta"], .5)}
    partner_rank = rank_matrix(candidates["radr50_2seed_rank_tta"])
    candidates["B2_090_010"] = mix(rank["oai_0949"], partner_rank, .9)
    candidates["B2_sensitivity_080_020"] = mix(rank["oai_0949"], partner_rank, .8)
    counts = {TARGETS[t]: {"positive": sum(row[t] for row in labels),
                           "negative": len(labels)-sum(row[t] for row in labels)}
              for t in range(len(TARGETS))}
    eligible = [t for t in range(len(TARGETS)) if min(counts[TARGETS[t]].values()) >= 3]
    need(bool(eligible), "no eligible Gold targets")
    result = scores(labels, candidates, eligible)
    pairs = [("radr50_s42", "oai_0949"),
             ("radr50_2seed_rank_tta", "radr50_s42"),
             ("radr50_2seed_rank_tta", "radr50_2seed_rank"),
             ("B2_090_010", "oai_0949"),
             ("B2_sensitivity_080_020", "oai_0949"),
             ("B2_090_010", "radr50_2seed_rank_tta")]
    cis, rejected = bootstrap(labels, candidates, eligible, pairs, draws=draws)
    differences = {f"{a} - {b}": {"delta": result[a]["macro"] - result[b]["macro"], **cis[f"{a} - {b}"]}
                   for a, b in pairs}
    body = {"schema": "exp130-gold58-comparison-v1", "studies": 58,
            "labels": "official storage/competition/train.csv, rows with non-null ACL",
            "counts": counts, "macro_targets": [TARGETS[t] for t in eligible],
            "excluded_from_macro": [TARGETS[t] for t in range(len(TARGETS)) if t not in eligible],
            "models": result, "paired_study_bootstrap": {"draws": draws, "seed": 20261008,
                "rejected_draws_under_3_per_class": rejected, "differences": differences},
            "inputs_sha256": {"oai": sha256(oai), **{f"s{seed}_{view}": sha256(local / f"gold58_s{seed}_{view}.csv")
                            for seed in SEEDS for view in ("plain", "tta")}},
            "interpretation": "Report only, never a gate. Gold58 has 58 enriched studies; per-target counts are small. "
                "The OAI 0.949 checkpoint did not train on Gold58 but selected epochs on it, making its Gold AUC optimistic. "
                "Our G1 fold models did not train on Gold58. This is not the 4,343-study weak-label OOF ruler "
                "and does not predict private leaderboard lift."}
    out = Path(out)
    need(not out.exists(), "comparison output directory already exists; refusing overwrite")
    out.mkdir(parents=True)
    write_new(out / "comparison.json", json.dumps(body, indent=2, allow_nan=False) + "\n")
    lines = ["# EXP130 Gold58 comparison", "", body["interpretation"], "",
             "| Target | P | N | Macro? | " + " | ".join(candidates) + " |",
             "|---|---:|---:|:---:|" + "---:|"*len(candidates)]
    for target in TARGETS:
        c = counts[target]
        lines.append(f"| {target} | {c['positive']} | {c['negative']} | " +
                     ("yes" if target in body["macro_targets"] else "no") + " | " +
                     " | ".join("undefined" if result[name]["per_target"][target] is None else
                                f"{result[name]['per_target'][target]:.4f}" for name in candidates) + " |")
    lines += ["", "| Macro AUC | | | | " + " | ".join(f"{result[name]['macro']:.4f}" for name in candidates) + " |",
              "", "Paired study bootstrap (" + str(draws) + " accepted draws; 95% percentile intervals):", "",
              "| Difference | Delta | 95% CI | P(delta > 0) |", "|---|---:|---:|---:|"]
    for name, d in differences.items():
        lines.append(f"| {name} | {d['delta']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {d['share_positive']:.3f} |")
    write_new(out / "comparison.md", "\n".join(lines) + "\n")
    return {"studies": 58, "macro_targets": len(eligible),
            "macros": {name: value["macro"] for name, value in result.items()},
            "output": str(out)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", default=str(BASE / "oai_gold58"))
    p = sub.add_parser("predict")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", default=str(BASE / "local"))
    p = sub.add_parser("compare")
    p.add_argument("--oai", required=True)
    p.add_argument("--local", default=str(BASE / "local"))
    p.add_argument("--out", default=str(BASE / "comparison"))
    p.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.action == "build":
        result = build(dest=args.out)
    elif args.action == "predict":
        result = predict(out=args.out, device=args.device, workers=args.workers)
    else:
        result = compare(args.oai, local=args.local, out=args.out, draws=args.draws)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
