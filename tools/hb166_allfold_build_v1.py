"""H-B-166 all-fold completion v1.2.2: folds0/1/4 seeds46..51; no new controls.
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
from tools import hb166_fold3_build_v1 as parent
from tools import hb166_allfold_spec_v1 as spec

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local/scratch/slot-b/hb166/allfold-v122"
VERSION = "1.2.2"
EXPERIMENT = "H-B-166"
SEEDS = (46, 47, 48, 49, 50, 51)
SEED_FOLD = {46: 0, 47: 0, 48: 1, 49: 1, 50: 4, 51: 4}
PARENT_WORK = ROOT / ".local/scratch/slot-b/hb166/pilot-fold3-v110"
PARENT_REG_SHA = "00684e5c55cf432513932f2eb0c255cb5f1702201ee64fabcdcb44d3f5d10373"
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
CONTROL_RUNTIME_SHA256 = "c8959a6f292fb9fa110d008a94174bd6b06676b73280f1f496e467f5cb5a9459"
# Host-dependent, numerically irrelevant runtime fields that may differ after a
# Vast restart on the same machine; every other snapshot key must be identical.
RUNTIME_DRIFT_ALLOWED = ("torch_interop_threads", "libs.cv2_threads")
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
SOURCES = ["training/arms/hb166_multiview_v1.py", "tools/hb166_allfold_build_v1.py", "tools/hb166_allfold_fit_v1.py",
           "tools/hb166_allfold_watchdog_v1.py", "tools/hb166_fold3_readout_v1.py",
           "tools/hb155_build_v1.py", "tools/hb155_fit_v1.py", "tools/hb155_watchdog_v1.py",
           "training/arms/hb155_crops_v1.py", "tools/parity_runtime.py", "config/workspace.toml",
           "tools/gpu_job.py", "tools/gpu_queue.py", "tools/job_runtime.py",
           "tools/hb166_allfold_transfer_v1.py", "tools/hb166_allfold_spec_v1.py", "tools/multi_lens_readout.py", "tools/multi_lens_readout_l4_v2.py"]


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
    return "UNPAIRED_HB105_S42_S43_f" + str(SEED_FOLD[seed])


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
    return {}


def verify_controls():
    return spec.control_inventory()


def expected(seed, engineering=False):
    need(seed in SEEDS, "unregistered seed")
    need(sha(PARENT_WORK / "registration.json") == PARENT_REG_SHA, "parent registration drift")
    cp = PARENT_WORK / "configs" / ("multiview-s" + str(44 + seed % 2) + ".json")
    preg = read(PARENT_WORK / "registration.json")
    need(sha(cp) == preg["configs"][rel(cp)], "sealed parent config drift")
    base = read(cp)
    for p, digest in base["inputs"].items():
        need(sha(ROOT / p) == digest, "parent dependency drift: " + p)
    fold = SEED_FOLD[seed]
    train, val = roster(fold)
    gold = gold_ids()
    need(not set(gold) & set(train+val), "Gold58 overlaps training/validation")
    c = b165.derive(base, seed=seed, train=train, val=val, gold=gold,
                   fold_root=rel(WORK / ("fold"+str(fold))), engineering=engineering)
    inputs = {p: sha(ROOT/p) for p in source_closure()}
    # Preserve every parent dependency, including initial weights, labels and cache QA.
    inputs.update(base["inputs"])
    inputs.update(read(b155.WORK / "configs/control-s42.json")["inputs"])
    for p in (cp, PARENT_WORK/"registration.json", GOLD_IDS, WORK/"readout-spec.json",
              WORK/"control-inventory.json", WORK/("fold"+str(fold))/"ordered-ids.json",
              WORK/("fold"+str(fold))/"group-metadata.csv", WORK/"fold-bindings.json"):
        inputs[rel(p)] = sha(p)
    c.update(version=VERSION, fold=fold, control_config=None, control_config_sha256=None,
             control_member=control_member(seed), control_fits={}, parent_config=rel(cp), parent_config_sha256=sha(cp),
             gate=spec.RULE, bias_statement=spec.BIAS, estimand=spec.ESTIMAND,
             inputs=dict(sorted(inputs.items())), exposure="EXPOSED", clean_oof=False)
    if engineering:
        c.update(train=train[:2], val=val[:1], gold=gold[:1])
    return c


def validate(path):
    c = read(path)
    need(c.get("schema") == "hb166-config/v1", "foreign config")
    need(c == expected(c["seed"], c["engineering_only"]), "config/source binding differs")
    return c


def registration_body():
    return dict(schema="hb166-registration/v1", version=VERSION, experiment=EXPERIMENT,
        folds=[0,1,4], seed_fold={str(s):f for s,f in SEED_FOLD.items()}, seeds=list(SEEDS), members=[member(s) for s in SEEDS],
        configs={rel(config_path(s,e)): sha(config_path(s,e)) for s in SEEDS for e in (False,True)},
        parent_registration_sha256=PARENT_REG_SHA, admission_purpose='measurement',
        hypotheses_sha256=spec.hypotheses_sha256(), historical_fold3_readout_sha256=spec.HISTORICAL_READOUT_SHA256,
        readout_spec={rel(WORK/"readout-spec.json"): sha(WORK/"readout-spec.json")},
        control_inventory={rel(WORK/"control-inventory.json"): sha(WORK/"control-inventory.json")},
        fold_bindings={rel(WORK/"fold-bindings.json"): sha(WORK/"fold-bindings.json")},
        gpu_dependency=spec.DEPENDENCY, recipe="Unchanged v1.1.0: workers2, batch8/micro2/accum4, epoch20, K36 flip; only fold roots, population-implied updates and new seeds differ. Fixed parent class weights retained.",
        exposure="EXPOSED", clean_oof=False, not_release_authorization=True)


def register():
    if (WORK / "registration.json").exists():
        registration_binding()
        for seed in SEEDS:
            for engineering in (False, True):
                validate(config_path(seed, engineering))
        return dict(status="REGISTERED", members=len(SEEDS), engineering_members=len(SEEDS),
                    registration_sha256=sha(WORK/"registration.json"), exposure="EXPOSED", clean_oof=False)
    register_rosters()
    gold_ids()
    immutable(WORK / "control-inventory.json", spec.control_inventory())
    immutable(WORK / "readout-spec.json", spec.make_spec())
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
        raise ValueError("unregistered smoke kind")
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
    need(m.get("schema") == "hb166-allfold-memory/v1" and m.get("device") == GPU,
         "measured memory admission required")
    need(isinstance(m.get("measured_at"), str) and bool(m["measured_at"]), "memory timestamp required")
    peak = max(read(WORK / "cuda" / member(s) / "summary.json")["peak_reserved_bytes"] for s in SEEDS)
    need(m.get("peak_reserved_bytes") == peak and 0 <= m.get("used_before_cuda_bytes", -1)
         and peak <= MEMORY_GATE_BYTES and m["used_before_cuda_bytes"] + peak + 2*(1<<30) <= m.get("total_bytes", 0),
         "measured 12 GiB + 2 GiB headroom gate")
    bindings[rel(memory)] = sha(memory)
    bindings.update(require_gpu())
    return bindings


def require_fit(path, review=None):
    c = validate(path)
    need(not c["engineering_only"] and c["fold"] == SEED_FOLD[c["seed"]], "scientific fold/config mismatch")
    require_gpu()
    need(review is not None, "independent smoke-bound review required")
    r = read(review)
    need(r.get("schema") == "hb166-allfold-fit-review/v1" and r.get("verdict") == "PASS"
         and r.get("independent") is True and bool(r.get("reviewer"))
         and r.get("reviewer") != "hb166-allfold-builder" and r.get("bindings") == review_bindings(),
         "independent smoke-bound review required")
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
    cmd = [sys.executable, "-B", "-m", "tools.hb166_allfold_watchdog_v1", "--attempt", str(attempt),
           "--budget", str(WORK / "gpu-budget.json")]
    return cmd + (["--measurement"] if measurement else ["--healthy-rates", str(rates)])


def sequence(mode, review=None, attempt="v1"):
    """Six serial multiview smokes or fits; no new controls."""
    need(mode in ("smoke", "fit") and attempt.replace("-", "").isalnum(), "invalid sequence")
    require_gpu()
    disk_gate(WORK, 30 * (1 << 30))
    if mode == "fit":
        need(admit(review)["status"] == "ADMITTED", "admission HOLD")
    jobs = []
    for seed in SEEDS:
        name = member(seed)
        out = WORK / ("cuda" if mode == "smoke" else "fits") / name
        jobs.append((name, out, ["tools.hb166_allfold_fit_v1", mode, "--config", str(config_path(seed))]))
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
    if device == "cuda":
        require_gpu()
    if device == "cpu":
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # empty, never -1 (see tools/hb166_allfold_fit_v1.hide_gpus_for_cpu)
    from tools import hb166_allfold_fit_v1 as f
    f.h.torch.set_num_threads(2)
    f.h.configure_runtime("none" if device == "cpu" else "fp16", device, policy="strict")
    snapshot = f.runtime_info()
    atomic_json(out, snapshot)
    if device == "cuda":
        require_runtime(snapshot)
    return dict(status="SNAPSHOT", device=device, path=str(out))



def register_rosters():
    import csv
    import io
    b165.exclusions.load()
    bindings = {}
    for fold in (0,1,4):
        source = ROOT / f"artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold{fold}"
        raw = read(source / "ordered-ids.json")
        data = {k: b165.exclusions.filtered(raw[k]) for k in ("train", "val")}
        dest = WORK / f"fold{fold}"
        immutable(dest / "ordered-ids.json", data)
        with (source / "group-metadata.csv").open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f); fields = reader.fieldnames
            rows = [r for r in reader if r["StudyInstanceUID"] not in b165.exclusions.UIDS]
        buf = io.StringIO(newline=""); writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        target = dest / "group-metadata.csv"
        if target.exists():
            need(target.read_bytes() == buf.getvalue().encode(), "group file drift")
        else:
            target.write_bytes(buf.getvalue().encode())
        for directory in (source,dest):
            for name in ("ordered-ids.json", "group-metadata.csv"):
                bindings[rel(directory/name)] = sha(directory/name)
    immutable(WORK/"fold-bindings.json", bindings)
    for fold in (0,1,4):
        roster(fold)


def roster(fold):
    import csv
    for p,d in read(WORK/"fold-bindings.json").items():
        need(sha(ROOT/p) == d, "roster binding drift: " + p)
    raw = read(ROOT/f"artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold{fold}/ordered-ids.json")
    cur = read(WORK/f"fold{fold}/ordered-ids.json")
    need(cur == {k:b165.exclusions.filtered(raw[k]) for k in ("train","val")}, "asymmetric exclusions")
    a,v = cur["train"],cur["val"]
    need(not set(a)&set(v) and len(set(a+v)) == len(a+v), "duplicate or leaking roster")
    with (WORK/f"fold{fold}/group-metadata.csv").open(newline="",encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    need(len(rows) == len(a+v), "group rows mismatch")
    for split, ids in (("train",a),("val",v)):
        need({r["StudyInstanceUID"] for r in rows if r["split"]==split} == set(ids), "group membership mismatch")
    need(not {r["component_id"] for r in rows if r["split"]=="train"} &
         {r["component_id"] for r in rows if r["split"]=="val"}, "component crosses fold")
    canonical=read(ROOT/b165.CANONICAL); pos={u:i for i,u in enumerate(canonical["train"]+canonical["val"])}
    need(all(all(pos[y]>pos[x] for x,y in zip(ids,ids[1:])) for ids in (a,v)), "noncanonical order")
    return a,v


def pre_bindings():
    bindings = registration_binding()
    prepare(); prepare(True)
    for name in ("prepared.json", "cpu-prepared.json"):
        bindings[rel(WORK/name)] = sha(WORK/name)
    for seed in SEEDS:
        p = WORK/"cpu"/member(seed)/"summary.json"
        check_smoke("cpu", "fit", seed, p)
        bindings[rel(p)] = sha(p)
    return bindings


def dependency_bindings():
    """B-supplied copies of completed evidence. Never reads running fold3 outputs."""
    rp, vp, dp = (WORK/n for n in ("fold3-readout.json", "fold3-result-review.json", "b-admission.json"))
    result, review, decision = read(rp), read(vp), read(dp)
    need(sha(rp) == spec.HISTORICAL_READOUT_SHA256, 'historical fold3 bytes changed')
    spec.require_measurement_history(result)
    need(review.get("schema") == "hb166-fold3-result-review/v1" and review.get("verdict") == "PASS"
         and review.get("independent") is True and bool(review.get("reviewer"))
         and review.get("reviewer") not in ("hb166-allfold-builder", "hb166-fold3-builder")
         and review.get("readout_sha256") == sha(rp)
         and review.get("registration_sha256") == PARENT_REG_SHA, "fold3 independent result review absent/stale")
    need(decision.get("schema") == "hb166-allfold-b-admission/v1" and decision.get("coordinator") == "B"
         and decision.get("decision") == "ADMIT_FRESH_FOLD_MEASUREMENT"
         and decision.get("purpose") == "measurement"
         and decision.get("hypotheses_sha256") == spec.hypotheses_sha256() and bool(decision.get("recorded_at"))
         and decision.get("registration_sha256") == sha(WORK/"registration.json")
         and decision.get("fold3_readout_sha256") == sha(rp)
         and decision.get("fold3_result_review_sha256") == sha(vp), "explicit B admission absent/stale")
    return {rel(p):sha(p) for p in (rp,vp,dp)}


def require_gpu():
    bindings = dependency_bindings()
    p = WORK/"pre-review.json"; r=read(p)
    need(r.get("schema") == "hb166-allfold-pre-review/v1" and r.get("verdict") == "PASS"
         and r.get("independent") is True and bool(r.get("reviewer"))
         and r.get("reviewer") != "hb166-allfold-builder" and r.get("bindings") == pre_bindings(),
         "independent CPU-bound pre-execution review required")
    bindings[rel(p)] = sha(p)
    return bindings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("register", "validate", "prepare", "admit", "review-bindings", "pre-bindings", "admit-smoke",
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
        elif a.command == "pre-bindings":
            result = pre_bindings()
        elif a.command == "admit-smoke":
            result = dict(status="ADMITTED_SMOKE", bindings=require_gpu())
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
