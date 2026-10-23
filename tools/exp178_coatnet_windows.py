"""EXP178: authenticated, serving-only EXP143-v2 within-slot diagnostic.

Wraps frozen EXP143/EXP152 code. CPU tests -> separate verification -> prepare
-> gpu_job supervised-run -> report. No account operations or submissions.

Changelog v1.1.0: remove CPU/inference wall deadlines and subprocess timeouts;
wait for terminal child status, retaining actual failure, receipt and resource gates.
Changelog v1.0.3 (Claude opus for B, 2026-10-10): GPU admission (supervisor and
child) uses the median utilization of >=5 nvidia-smi samples over >=5 s instead
of one sample; compute-app, free-memory and device checks kept; samples recorded.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT))
import numpy as np
from tools import exp152_own0949_release as release
import exp143_own0949 as training
import raptor_common as common

VERSION = "1.1.0"
UID, TARGETS = release.UID, release.TARGETS
EXP = ROOT / "artifacts/experiments/EXP178_COATNET_WINDOWS_DIAG"
PROTOCOL = EXP / "PROTOCOL.md"
ASSETS = ROOT / ".local/exp152-own0949-v1/stage/assets"
SERVING = ASSETS / "public-serving.py"
CHECKPOINT = ASSETS / release.OWN_NAME
TRAIN_PROTOCOL = release.PROTOCOL
CORPUS = ROOT / "storage/cache/derived/EXP143_RAPTOR96_384_C140_S02"
GOLD = ROOT / "artifacts/experiments/EXP165_GOLD_TRAIN/gold-labels.csv"
TEST = ROOT / "storage/competition/test.csv"
SERIES = ROOT / "storage/competition/test_series.csv"
DICOM = ROOT / "storage/downloads/rsna-knee-abnormality-detection/prepared/9175e113e4a00fe679c4088b/test_series"
AUTOCHAIN = ROOT / ".local/exp174-autochain.log"
PREDECESSOR_SUCCESS = ROOT / "artifacts/logs/EXP174_OWN0949_JEV_S42/SUCCESS"
PINNED = {
    SERVING: "25ef6ab9d2e942ac291433a498246d0507481c2aefa5630cdd601d0a36852cf2",
    CHECKPOINT: "a163f3f8fa7ae57a32ea5237ce36501f74331d7fcec826bc50eb904a8e2f3295",
    ASSETS / "release-manifest.json": "d85946bd08c6087ef170e705bf695260c5f398e57479bd30b48ba65e0838b295",
    TRAIN_PROTOCOL: "b2aaca64289249595e532b74d361175ed78b1170109eee302ec29068e0e7af11",
    CORPUS / "manifest.json": "9c521caa39c51cee0834176c980be63ffeaf4d22b072215dd74dc6e0c97c8b4c",
    GOLD: "62a26087d3ade2e788ff3dcbf5bba9cf9b371a0f76ae6906ffc271e99b06c616",
}
CODE = [Path(__file__), ROOT / "tests/test_exp178_coatnet_windows.py", PROTOCOL, EXP / "RUNBOOK.md",
        EXP / "CHANGELOG.md",
        ROOT / "tools/exp152_own0949_release.py", ROOT / "tools/exp143_v2.py",
        ROOT / "training/src/exp143_own0949.py",
        ROOT / "training/src/raptor_common.py", ROOT / "training/src/raptor_dicom.py",
        ROOT / "tools/gpu_job.py"]
_SELECTOR_LOCK = threading.RLock()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return release.sha(Path(path))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def output_path(path):
    path = Path(path).resolve()
    need(path.is_relative_to(EXP.resolve()), "Outputs must remain inside the EXP178 directory")
    return path


def array_sha(x):
    return release.array_sha(x)


def code_hashes():
    return {relative(p): sha(p) for p in CODE}


def verify_review(attestation):
    """B writes this small attestation AFTER CPU tests and independent PASS.

    It is not created by the experiment tool and cannot self-authorize a run.
    """
    a = read(attestation)
    need(a.get("verdict") == "PASS", "Separate verification PASS required")
    need(a.get("version") == VERSION and a.get("code_sha256") == code_hashes(),
         "Reviewed code/protocol/test hashes no longer match")
    note = ROOT / a["verification_note"]
    tests = ROOT / a["cpu_test_log"]
    need(sha(note) == a["verification_sha256"] and sha(tests) == a["cpu_test_log_sha256"],
         "Review note or CPU-test evidence changed")
    need(a.get("b_spot_check") and a.get("cpu_tests_passed") is True,
         "B must confirm tests and spot-check the independent PASS")
    return dict(verification_note=relative(note), verification_sha256=sha(note),
                attestation=relative(attestation), attestation_sha256=sha(attestation))


def base_receipt(review):
    return dict(version=VERSION, protocol_sha256=sha(PROTOCOL), **review)


def within_centers(mask, depth, k):
    mask = np.asarray(mask)
    need(depth == 96 and mask.shape == (96,) and k > 0, "Bad EXP143 centre contract")
    need(np.isin(mask, [0, 1]).all(), "Mask must be binary")
    cs = common.select_centers(mask, depth, k, training.SPEC, "within-slot-valid")
    need(all(c >= 1 for c in cs), "No within-slot valid triplet; no fallback")
    return cs


def centers(mask, policy="current"):
    need(policy in ("current", "within"), "Unknown window policy")
    mask = np.asarray(mask)
    need(mask.shape == (96,) and np.isin(mask, [0, 1]).all() and mask.any(), "Bad study mask")
    return (common.eval_centers(mask, 96, 94) if policy == "current"
            else within_centers(mask, 96, 94))


def training_windows(vol, mask, policy, mirror=False):
    """Restore the original imported selector even when frozen windows raises.

    This scoped patch is only used in this single-threaded diagnostic process.
    Frozen code and files remain unchanged; training is never called.
    """
    need(policy in ("current", "within"), "Unknown window policy")
    centers(mask, policy)
    with _SELECTOR_LOCK:
        original = common.eval_centers
        try:
            if policy == "within":
                common.eval_centers = within_centers
            return training.windows(vol, mask, mirror=mirror)
        finally:
            common.eval_centers = original


def public_module(source, *, for_inference=False):
    """Keep frozen definitions intact; CPU loading omits only GPU startup.

    timm imports torchvision/Torch Dynamo even when no model is built. That is
    unnecessary for CPU windows/decoding and conflicts with CUDA-call guards.
    The two global backend writes also belong only to the inference process.
    Match these three exact top-level statements, not names or line numbers;
    never transform function/class bodies or normalization constants.
    """
    tree = ast.parse(source, filename=str(SERVING))
    startup = (
        "import timm",
        "torch.backends.cudnn.benchmark = True",
        "torch.backends.cuda.matmul.allow_tf32 = True",
    )
    expected = {ast.dump(ast.parse(s).body[0]): s for s in startup}
    found = Counter()
    cpu_body = []
    for node in tree.body:
        signature = ast.dump(node)
        if signature in expected:
            found[signature] += 1
        else:
            cpu_body.append(node)
    need(found == Counter({s: 1 for s in expected}), "Frozen inference startup changed")
    if not for_inference:
        tree.body = cpu_body
    return tree


def load_public_namespace(*, for_inference=False):
    """CPU preprocessing by default; the supervised GPU child opts into timm.

    Inference executes the entire authenticated source, including its original
    backend settings. CPU loading cannot construct a timm backbone.
    """
    source = SERVING.read_bytes()
    need(hashlib.sha256(source).hexdigest() == PINNED[SERVING], "Frozen serving source changed")
    ns = {"__name__": "exp178_frozen_public", "__file__": str(SERVING)}
    tree = public_module(source.decode("utf-8"), for_inference=for_inference)
    exec(compile(tree, str(SERVING), "exec"), ns)
    need(ns["LAB"] == TARGETS and ns["SLOTS"] == [tuple(s) for s in training.SPEC["slots"]],
         "Frozen target/slot identity changed")
    return ns


def public_windows(ns, vol, mask, policy, mirror=False, device=None):
    need(policy in ("current", "within"), "Unknown window policy")
    expected = centers(mask, policy)
    original = ns["_eval_centers"]
    try:
        if policy == "within":
            ns["_eval_centers"] = within_centers
        need(ns["_eval_centers"](mask, 96, 94) == expected, "Centre parity failed")
        if device is None:
            return ns["eval_windows"](vol, mask, 94, 320, mirror=mirror)
        return ns["gpu_windows"](vol, mask, 94, 320, device, mirror=mirror)
    finally:
        ns["_eval_centers"] = original


def portable_pread(fd, n, offset):
    """Windows positional binary read, preserving descriptor position.

    A process-local lock serializes seek/read/restore across helper threads.
    No os module or frozen file is patched.
    """
    need(n >= 0 and offset >= 0, "Invalid positional read")
    with _SELECTOR_LOCK:
        previous = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, n)
        finally:
            os.lseek(fd, previous, os.SEEK_SET)


def install_portable_reader(module, force=False):
    """Adapt embedded POSIX-only I/O in memory; rendering math stays frozen."""
    if force or not hasattr(os, "pread"):
        module._O_FLAGS |= getattr(os, "O_BINARY", 0)
        def read_at(fd, n, offset):
            data = portable_pread(fd, n, offset)
            if module.ACCOUNT:
                module.STATS.add(reads=1, nbytes=len(data))
            return data
        module._pread = read_at
        return "seek-read-restore-binary-v1"
    return "native-pread"


def make_reader(ns, work, force_portable=False):
    # The frozen loader writes _fastread.py beside __file__, in new output only.
    ns["__file__"] = str(Path(work) / "frozen-public-context.py")
    module = ns["_load_fastread"]()
    adapter = install_portable_reader(module, force=force_portable)
    return module.make_builder(ns["FASTREAD_VARIANT"], 384, ns["SLOTS"],
                               hdr_threads=2, px_threads=2), adapter


def window_audit(mask, current, within, slots=None):
    group = np.repeat(np.arange(5), training.SLOT_SIZES)
    def one(cs):
        n_cross = n_invalid = n_mix = 0
        for c in cs:
            idx = np.asarray([c - 1, c, c + 1])
            n_cross += len(set(group[idx])) > 1
            n_invalid += not np.all(np.asarray(mask)[idx] > 0)
            if slots is not None:
                sources = {slots[int(group[j])].get("series") for j in idx if mask[j] > 0}
                sources.discard(None)
                n_mix += len(sources) > 1
        return dict(cross_slot=int(n_cross), invalid_mask=int(n_invalid),
                    valid_series_mix=int(n_mix) if slots is not None else None,
                    duplicate_centers=len(cs) - len(set(cs)))
    need(len(current) == len(within) == 94, "Audit needs paired K94")
    a, b = one(current), one(within)
    need(b["cross_slot"] == b["invalid_mask"] == 0, "Treatment window is unsafe")
    return dict(windows=94, positional_changed=int(np.sum(np.asarray(current) != within)),
                multiset_replaced=94 - sum((Counter(current) & Counter(within)).values()),
                current=a, within=b, centers_current=current, centers_within=within)


def correlations(a, b):
    from scipy.stats import rankdata
    a, b = np.asarray(a), np.asarray(b)
    need(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 12,
         "Prediction target shape mismatch")
    need(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite probabilities")
    def corr(x, y):
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])
    per = {}
    for j, target in enumerate(TARGETS):
        per[target] = dict(pearson=corr(a[:, j], b[:, j]),
            spearman=corr(rankdata(a[:, j]), rankdata(b[:, j])),
            mean_abs=float(np.abs(b[:, j] - a[:, j]).mean()),
            max_abs=float(np.abs(b[:, j] - a[:, j]).max()))
    def mean(key):
        values = [v[key] for v in per.values() if v[key] is not None]
        return float(np.mean(values)) if values else None
    return dict(per_target=per, mean_pearson=mean("pearson"), mean_spearman=mean("spearman"),
                mean_abs=float(np.abs(b - a).mean()), max_abs=float(np.abs(b - a).max()))


def paired_auc(y, current, within, draws=2000, seed=20261010):
    """Paired study bootstrap, weighted pair-count AUC equivalent to resampling.

    Extends raptor_report.bootstrap_delta with target intervals and fixed12
    macro eligibility. Pair ties receive 0.5. No predictions are re-ranked by
    target weights and the same study multiplicities apply to every target.
    """
    y, a, b = map(np.asarray, (y, current, within))
    need(y.shape == a.shape == b.shape and y.ndim == 2 and y.shape[1] == 12,
         "AUC shape mismatch")
    need(np.isin(y, [0, 1]).all() and np.isfinite(a).all() and np.isfinite(b).all(),
         "AUC requires complete binary labels and finite predictions")
    need(((a >= 0) & (a <= 1)).all() and ((b >= 0) & (b <= 1)).all(),
         "AUC diagnostic requires raw probabilities in [0,1]")
    need(draws > 0 and all(len(np.unique(y[:, j])) == 2 for j in range(12)),
         "All twelve targets require two classes")
    ma, pa = common.macro_auc(y, a)
    mb, pb = common.macro_auc(y, b)
    rng = np.random.default_rng(seed)
    n = len(y)
    picks = rng.integers(0, n, (draws, n))
    counts = np.asarray([np.bincount(p, minlength=n) for p in picks], dtype=np.float64)
    delta = np.full((draws, 12), np.nan)
    def interval(d):
        d = np.asarray(d); d = d[np.isfinite(d)]
        return dict(valid_draws=len(d), low=float(np.quantile(d, .025)) if len(d) else None,
                    high=float(np.quantile(d, .975)) if len(d) else None)
    per = {}
    for j, target in enumerate(TARGETS):
        pos, neg = y[:, j] == 1, y[:, j] == 0
        cp, cn = counts[:, pos], counts[:, neg]
        denom = cp.sum(1) * cn.sum(1)
        def pair_matrix(p):
            d = p[pos, j, None] - p[neg, j][None, :]
            return (d > 0).astype(float) + .5 * (d == 0)
        numerator = np.einsum("bi,ij,bj->b", cp, pair_matrix(b) - pair_matrix(a), cn)
        np.divide(numerator, denom, out=delta[:, j], where=denom > 0)
        per[target] = dict(current=float(pa[target]), within=float(pb[target]),
            delta=float(pb[target] - pa[target]), positives=int(pos.sum()), negatives=int(neg.sum()),
            **interval(delta[:, j]))
    valid_macro = np.isfinite(delta).all(1)
    macro_draws = delta[valid_macro].mean(1) if valid_macro.any() else np.empty(0)
    return dict(per_target=per, macro12=dict(current=float(ma), within=float(mb),
        delta=float(mb - ma), **interval(macro_draws)),
        bootstrap=dict(unit="study", draws=draws, seed=seed, paired=True,
                       macro_rule="all12 targets have both classes", interval="percentile95, unadjusted"))


def predecessor_state(text, success):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    completed = [i for i, line in enumerate(lines)
                 if re.search(r"(?:^|\s)EXP174_CHAIN_COMPLETE(?:\s|$)", line)]
    exits = [(i, line.rsplit("EXP174_CHAIN_EXIT=", 1)[1].strip())
             for i, line in enumerate(lines) if "EXP174_CHAIN_EXIT=" in line]
    if exits and exits[-1][1] != "0":
        return "FAILED"
    if (completed and exits and completed[-1] < exits[-1][0]
            and exits[-1][0] == len(lines) - 1 and success):
        return "READY"
    return "WAIT"


def require_predecessor():
    text = AUTOCHAIN.read_text(encoding="utf-8") if AUTOCHAIN.exists() else ""
    state = predecessor_state(text, PREDECESSOR_SUCCESS.exists())
    need(state == "READY", "EXP174 autochain not complete: " + state)
    return dict(autochain_sha256=sha(AUTOCHAIN), success_sha256=sha(PREDECESSOR_SUCCESS))


_DESKTOP_GPU_PROCESSES = frozenset({
    "explorer.exe", "dwm.exe", "textinputhost.exe", "searchapp.exe",
    "shellexperiencehost.exe", "nvidia overlay.exe", "chrome.exe",
    "msedge.exe", "msedgewebview2.exe", "whale.exe", "telegram.exe",
    "discord.exe", "chatgpt.exe", "claude.exe",
})
_MIN_FREE_GPU_MIB = 10240


def _windows_process_names():
    """Resolve WDDM's '[Insufficient Permissions]' names without opening CUDA."""
    output = subprocess.check_output([
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        "Get-Process | Select-Object Id,ProcessName | ConvertTo-Csv -NoTypeInformation",
    ], text=True)
    rows = csv.DictReader(output.splitlines())
    need(rows.fieldnames == ["Id", "ProcessName"], "Windows process inventory unavailable")
    return {int(row["Id"]): row["ProcessName"].lower() + ".exe"
            for row in rows if row["Id"].isdigit() and row["ProcessName"]}


_BUSY = "Local GPU is busy, lacks free memory, or is not the registered device"
_MAX_MEDIAN_UTIL = 10      # percent, applied to the median of the window below
_UTIL_SAMPLES = 5          # at least this many nvidia-smi readings ...
_UTIL_WINDOW_SECONDS = 5.0  # ... whose first-to-last span is at least this long
_UTIL_MAX_SAMPLES = 12     # fail closed if the sleep/clock never reaches the window
_now, _sleep = time.monotonic, time.sleep


def _gpu_sample():
    line = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip()
    rows = list(csv.reader(line.splitlines()))
    need(len(rows) == 1 and len(rows[0]) == 4, "Expected exactly one local GPU")
    name, util_text, used_text, free_text = (value.strip() for value in rows[0])
    return dict(name=name, utilization=int(util_text), memory_mib=int(used_text),
                memory_free_mib=int(free_text))


def _gpu_window():
    """v1.0.3: >=5 readings over >=5 s. One nvidia-smi utilization reading is
    intermittently >10% from WDDM desktop rendering alone (two refused launches
    on 2026-10-10), so the 10% rule applies to the median of the window."""
    start, samples, times = _now(), [], []
    while True:
        sample = _gpu_sample()
        times.append(_now() - start)
        need("RTX 4070 Ti SUPER" in sample["name"] and
             all(s["name"] == sample["name"] for s in samples), _BUSY)
        samples.append(dict(sample, t_seconds=round(times[-1], 3)))
        if len(samples) >= _UTIL_SAMPLES and times[-1] - times[0] >= _UTIL_WINDOW_SECONDS:
            return samples
        need(len(samples) < _UTIL_MAX_SAMPLES, "GPU utilization window not reached")
        _sleep(_UTIL_WINDOW_SECONDS / (_UTIL_SAMPLES - 1))


def require_idle_gpu(owned_pid=None):
    """Admit quiet WDDM desktop contexts, never another research process."""
    samples = _gpu_window()
    name = samples[0]["name"]
    utils = [s["utilization"] for s in samples]
    util = float(np.median(utils))
    used = max(s["memory_mib"] for s in samples)
    free = min(s["memory_free_mib"] for s in samples)
    need(util <= _MAX_MEDIAN_UTIL and free >= _MIN_FREE_GPU_MIB,
         f"{_BUSY} (median utilization {util:g}% of {utils}; minimum free {free} MiB)")

    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name",
        "--format=csv,noheader"], text=True)
    listed = list(csv.reader(apps.splitlines()))
    need(all(len(row) == 2 and row[0].strip().isdigit() for row in listed),
         "GPU process inventory unavailable")
    hidden = [int(row[0]) for row in listed
              if int(row[0]) != owned_pid and row[1].strip() == "[Insufficient Permissions]"]
    resolved = _windows_process_names() if hidden else {}
    desktop_count = 0
    for pid_text, path in listed:
        pid = int(pid_text)
        if pid == owned_pid:
            continue
        executable = (resolved.get(pid) if pid in hidden else ntpath.basename(path.strip().strip('"')).lower())
        need(executable in _DESKTOP_GPU_PROCESSES,
             f"Competing or unidentified GPU process: PID {pid}")
        desktop_count += 1
    return dict(name=name, utilization=util, memory_mib=used,
                memory_free_mib=free, desktop_processes=desktop_count,
                utilization_rule=f"median of >={_UTIL_SAMPLES} samples over >={_UTIL_WINDOW_SECONDS:g} s "
                                 f"<= {_MAX_MEDIAN_UTIL}%; free memory = minimum, used = maximum",
                samples=samples)


def read_gold():
    # Structured-only artifact: deliberately never opens train.csv or reports.
    with GOLD.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        need(r.fieldnames == [UID, *TARGETS], "Gold target order changed")
        rows = list(r)
    ids = [r[UID] for r in rows]
    y = np.asarray([[float(r[t]) for t in TARGETS] for r in rows])
    need(len(ids) == len(set(ids)) == 58 and np.isin(y, [0, 1]).all(), "Invalid Gold58")
    need(all(len(np.unique(y[:, j])) == 2 for j in range(12)), "Gold requires both classes per target")
    p = read(TRAIN_PROTOCOL)
    need(set(ids) == set(p["excluded_gold_ids"]) and not set(ids) & set(p["train_ids"]),
         "Gold training leakage or roster drift")
    return ids, y


def test_ids():
    with TEST.open(encoding="utf-8-sig", newline="") as f:
        ids = [r[UID] for r in csv.DictReader(f)]
    need(len(ids) == len(set(ids)) == 3, "Visible cohort is not three unique studies")
    return ids


def data_files(gold_ids, visible_ids):
    corpus = training.Corpus(CORPUS)
    files = [*PINNED, TEST, SERIES]
    for uid in gold_ids:
        row = corpus.rows[uid]
        files.extend([CORPUS / f"study_{row:05d}.npy", CORPUS / f"study_{row:05d}.json"])
    for uid in visible_ids:
        folder = DICOM / uid
        need(folder.is_dir(), "Visible DICOM study directory missing")
        found = sorted(folder.rglob("*.dcm"))
        need(found, "No visible DICOM files")
        files.extend(found)
    return sorted(set(files))


def prepare(args):
    wall, cpu = time.monotonic(), time.process_time()
    review = verify_review(args.verification)
    for p, digest in PINNED.items():
        need(sha(p) == digest, "Pinned input changed: " + relative(p))
    ids, _ = read_gold()
    visible = test_ids()
    need(not set(ids) & set(visible), "Gold/test cohorts overlap")
    files = data_files(ids, visible)
    hashes = {relative(p): sha(p) for p in files}
    result = dict(**base_receipt(review), status="REGISTERED_NOT_RUN", targets=TARGETS,
        gold_ids=ids, test_ids=visible, code_sha256=code_hashes(), inputs_sha256=hashes,
        wall_seconds=time.monotonic()-wall, cpu_seconds=time.process_time()-cpu,
        gpu_hours=0, dicom_root=relative(DICOM))
    write_new(output_path(args.registration), result)
    print(json.dumps({k: result[k] for k in ("status", "version", "wall_seconds", "cpu_seconds", "gpu_hours")}))


def checked_registration(args):
    review = verify_review(args.verification)
    r = read(args.registration)
    need(r["status"] == "REGISTERED_NOT_RUN" and r["code_sha256"] == code_hashes(), "Registration drift")
    need(r["attestation_sha256"] == review["attestation_sha256"] and r["targets"] == TARGETS,
         "Registration verification/target mismatch")
    ids, _ = read_gold()
    visible = test_ids()
    need(ids == r["gold_ids"] and visible == r["test_ids"], "Registered roster drift")
    files = data_files(ids, visible)
    need(set(r["inputs_sha256"]) == {relative(p) for p in files}, "Input inventory drift")
    for name, digest in r["inputs_sha256"].items():
        need(sha(ROOT / name) == digest, "Registered input drift: " + name)
    return r, review


def supervised_run(args):
    """CPU supervisor; wait for the owned child and validate its terminal receipt."""
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
         "Launch gpu_job with CUDA_VISIBLE_DEVICES=-1; only the owned inference child enables device0")
    cpu = time.process_time()
    registration, review = checked_registration(args)
    predecessor = require_predecessor()
    device = require_idle_gpu()
    out = output_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "tools.exp178_coatnet_windows", "_infer",
        "--registration", str(Path(args.registration).resolve()), "--verification",
        str(Path(args.verification).resolve()), "--out", str(out)]
    start = time.monotonic()
    with (out / "inference.log").open("xb", buffering=0) as log:
        p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                 "EXP178_SUPERVISOR_PID": str(os.getpid()), "CUDA_VISIBLE_DEVICES": "0"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        rc = p.wait()
        status = "CHILD_FINISHED" if rc == 0 else "FAILED"
    elapsed = time.monotonic() - start
    result = dict(**base_receipt(review), status=status, exit_code=rc, elapsed_seconds=elapsed,
                  gpu_hours_upper_bound=elapsed/3600, predecessor=predecessor, gpu_before=device,
                  cpu_seconds=time.process_time()-cpu,
                  child_cpu_seconds=None,
                  child_cpu_accounting="Read successful inference.json; unavailable after child failure",
                  registration_sha256=sha(args.registration))
    write_new(out / "supervisor.json", result)
    need(status == "CHILD_FINISHED", "GPU child failed")
    inference = read(out / "inference.json")
    need(inference["status"] == "COMPLETE_PAIRED_INFERENCE" and inference["studies"] == 61,
         "Exit zero is not paired-inference validation")
    print(json.dumps(result))


def infer(args):
    need(os.environ.get("EXP178_SUPERVISOR_PID") == str(os.getppid()),
         "Inference must be a child of supervised-run; direct invocation forbidden")
    import torch
    import timm
    import scipy
    import sklearn
    import cv2
    wall, cpu = time.monotonic(), time.process_time()
    registration, review = checked_registration(args)
    require_predecessor()
    before = require_idle_gpu(owned_pid=os.getpid())
    torch.set_num_threads(2)
    torch.manual_seed(20261010)
    ns = load_public_namespace(for_inference=True)
    manifest = read(ASSETS / "release-manifest.json")
    out = output_path(args.out)
    need(out.is_dir() and not (out / "inference.json").exists(), "Invalid inference output")
    release.bind(ns, manifest, {release.OWN_NAME: CHECKPOINT}, "own", out)
    model, _ = release.checked_model(ns, CHECKPOINT, release.ARCH, 384, "cuda:0")
    need("RTX 4070 Ti SUPER" in torch.cuda.get_device_name(0), "Wrong CUDA device")
    torch.cuda.reset_peak_memory_stats()
    corpus = training.Corpus(CORPUS)
    from tools.exp143_v2 import series_records
    from raptor_dicom import DirectoryDicomSource, prepare_study
    records = series_records(SERIES)
    source = DirectoryDicomSource(DICOM, header_only=True)
    reader, io_adapter = make_reader(ns, out)
    inputs, probabilities = [], {"current": [], "within": []}
    ids = registration["gold_ids"] + registration["test_ids"]
    try:
        for index, uid in enumerate(ids):
            cohort = "gold58" if index < 58 else "visible3"
            if index < 58:
                vol, mask = corpus.get(uid)
                receipt = read(CORPUS / f"study_{corpus.rows[uid]:05d}.json")
            else:
                vol, mask = reader(uid, records, str(DICOM))
                ref_vol, ref_mask, receipt = prepare_study(uid, records.get(uid, []), source, training.SPEC, "test")
                need(np.array_equal(vol, ref_vol) and np.array_equal(mask, ref_mask), "Visible decode parity failed")
            audit = window_audit(mask, centers(mask), centers(mask, "within"), receipt["slots"])
            row = dict(uid=uid, cohort=cohort, volume_sha256=array_sha(vol), mask_sha256=array_sha(mask),
                       audit=audit, window_sha256={}, repeat_max_abs=None,
                       centers_current_sha256=array_sha(np.asarray(audit["centers_current"], np.int64)),
                       centers_within_sha256=array_sha(np.asarray(audit["centers_within"], np.int64)))
            pred = {}
            for policy in (("current", "within") if index % 2 == 0 else ("within", "current")):
                windows = []
                for mirror in (False, True):
                    x = public_windows(ns, vol, mask, policy, mirror, "cuda:0")
                    reference = training_windows(vol, mask, policy, mirror)
                    public_cpu = public_windows(ns, vol, mask, policy, mirror)
                    need(np.array_equal(reference, public_cpu.numpy()) and
                         np.array_equal(reference, x.cpu().numpy()), "Exact normalized window parity failed")
                    row["window_sha256"][policy + ("_mirror" if mirror else "_plain")] = array_sha(reference)
                    windows.append(x)
                p = ns["infer_pair_probs"](model, *windows, "cuda:0")
                need(p.shape == (12,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
                     "Invalid raw probability")
                pred[policy] = p
                if index in (0, 58) and policy == "current":
                    repeat = ns["infer_pair_probs"](model, *windows, "cuda:0")
                    row["repeat_max_abs"] = float(np.max(np.abs(repeat-p)))
                    need(row["repeat_max_abs"] <= 2e-4, "Same-session repeat drift exceeds gate")
                del windows, x, reference, public_cpu
            for policy in probabilities:
                probabilities[policy].append(pred[policy])
            inputs.append(row)
            print(json.dumps(dict(done=index+1, total=61, cohort=cohort,
                                  elapsed_seconds=time.monotonic()-wall)), flush=True)
    finally:
        source.close()
    torch.cuda.synchronize()
    np.savez_compressed(out / "predictions.npz", ids=np.asarray(ids),
        targets=np.asarray(TARGETS), current=np.asarray(probabilities["current"]),
        within=np.asarray(probabilities["within"]), version=np.asarray(VERSION),
        verification_note=np.asarray(review["verification_note"]),
        verification_sha256=np.asarray(review["verification_sha256"]),
        protocol_sha256=np.asarray(sha(PROTOCOL)))
    write_new(out / "inputs.json", dict(**base_receipt(review), studies=inputs))
    result = dict(**base_receipt(review), status="COMPLETE_PAIRED_INFERENCE", studies=61,
        registration_sha256=sha(args.registration), checkpoint_sha256=sha(CHECKPOINT),
        predictions_sha256=sha(out / "predictions.npz"), inputs_sha256=sha(out / "inputs.json"),
        elapsed_seconds=time.monotonic()-wall, cpu_seconds=time.process_time()-cpu,
        gpu_name=before["name"], gpu_before=before, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        decoder_io_adapter=io_adapter,
        timm_version=timm.__version__, numpy_version=np.__version__, scipy_version=scipy.__version__,
        sklearn_version=sklearn.__version__, opencv_version=cv2.__version__, python_version=sys.version,
        cudnn_version=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    write_new(out / "inference.json", result)


def report(args):
    wall, cpu = time.monotonic(), time.process_time()
    registration, review = checked_registration(args)
    out = output_path(args.out)
    r, supervisor = read(out / "inference.json"), read(out / "supervisor.json")
    need(r["status"] == "COMPLETE_PAIRED_INFERENCE" and r.get("studies") == 61
         and supervisor["status"] == "CHILD_FINISHED" and supervisor.get("exit_code") == 0,
         "Incomplete or failed run")
    need(r["registration_sha256"] == supervisor["registration_sha256"] == sha(args.registration),
         "Inference registration mismatch")
    need(r["attestation_sha256"] == review["attestation_sha256"], "Inference review mismatch")
    need(sha(out / "predictions.npz") == r["predictions_sha256"]
         and sha(out / "inputs.json") == r["inputs_sha256"], "Inference artifact drift")
    with np.load(out / "predictions.npz", allow_pickle=False) as z:
        ids, a, b = z["ids"].tolist(), z["current"], z["within"]
        need(z["targets"].tolist() == TARGETS and a.shape == b.shape == (61, 12), "Output identity mismatch")
        need(str(z["version"]) == VERSION and str(z["verification_sha256"]) == review["verification_sha256"],
             "Raw prediction version/review mismatch")
    need(ids == registration["gold_ids"] + registration["test_ids"], "Prediction alignment mismatch")
    need(np.isfinite(a).all() and np.isfinite(b).all() and
         ((a >= 0) & (a <= 1)).all() and ((b >= 0) & (b <= 1)).all(), "Bad predictions")
    gold_ids, y = read_gold()
    need(ids[:58] == gold_ids, "Label/prediction alignment mismatch")
    inputs = read(out / "inputs.json")["studies"]
    need([v["uid"] for v in inputs] == ids, "Window audit alignment mismatch")
    cohorts = {}
    for name, lo, hi in (("gold58", 0, 58), ("visible3", 58, 61)):
        rows = inputs[lo:hi]
        n = len(rows) * 94
        counts = {key: sum(v["audit"][key] for v in rows)
                  for key in ("positional_changed", "multiset_replaced")}
        audit = dict(windows=n, **counts,
            positional_changed_fraction=counts["positional_changed"]/n,
            multiset_replaced_fraction=counts["multiset_replaced"]/n)
        for policy in ("current", "within"):
            audit[policy] = {key: sum(v["audit"][policy][key] for v in rows)
                for key in ("cross_slot", "invalid_mask", "valid_series_mix", "duplicate_centers")}
        cohorts[name] = dict(studies=hi-lo, windows=audit, predictions=correlations(a[lo:hi], b[lo:hi]))
    auc = paired_auc(y, a[:58], b[:58])
    c = cohorts["gold58"]["predictions"]
    material = ((c["mean_spearman"] is not None and c["mean_spearman"] < .995)
                or any(v["mean_abs"] >= .01 for v in c["per_target"].values()))
    result = dict(**base_receipt(review), status="REPORT_ONLY_NO_SELECTION", cohorts=cohorts,
        gold58_auc=auc, material_prediction_movement=material,
        next_test="H-B-87 separately registered training comparison" if material else "No automatic follow-up",
        inference_sha256=sha(out / "inference.json"), supervisor_sha256=sha(out / "supervisor.json"),
        gpu_hours_upper_bound=supervisor["gpu_hours_upper_bound"],
        cpu_seconds_supervisor=supervisor["cpu_seconds"],
        cpu_seconds_inference=r["cpu_seconds"], cpu_seconds_report=time.process_time()-cpu,
        cpu_seconds_prepare=registration["cpu_seconds"], report_wall_seconds=time.monotonic()-wall,
        risks=["Gold58 exposed, untrained-on; diagnostic only", "No private-LB or causal training gain claim",
               "Serving-only shift on a legacy-trained model", "Unadjusted per-target study-bootstrap intervals",
               "Cached adjacency and valid masks are not native anatomical validity",
               "Visible3 correlations have only three observations"])
    write_new(out / "report.json", result)
    lines = ["# EXP178 serving-only diagnostic", "", "REPORT_ONLY_NO_SELECTION; Gold58 exposed, untrained-on.",
        f"Version {VERSION}; verification: {review['verification_note']} ({review['verification_sha256']}).", "",
        "| Cohort | Mean Pearson | Mean Spearman | Positional change | Multiset replacement |",
        "|---|---:|---:|---:|---:|"]
    for name, v in cohorts.items():
        c, w = v["predictions"], v["windows"]
        lines.append(f"| {name} | {c['mean_pearson']} | {c['mean_spearman']} | "
                     f"{w['positional_changed_fraction']:.4%} | {w['multiset_replaced_fraction']:.4%} |")
    lines += ["",
        "| Target | Current AUC | Within AUC | Delta | Paired study 95% CI | Valid draws |",
        "|---|---:|---:|---:|---|---:|"]
    for target, v in [*auc["per_target"].items(), ("Macro12", auc["macro12"])]:
        ci = "undefined" if v["low"] is None else f'[{v["low"]:+.6f}, {v["high"]:+.6f}]'
        lines.append(f'| {target} | {v["current"]:.6f} | {v["within"]:.6f} | {v["delta"]:+.6f} | {ci} | {v["valid_draws"]} |')
    lines += ["", "Full correlations, window counts, hashes, timing and caveats: report.json.",
              "No production swap or submission is authorized by this diagnostic."]
    with (out / "report.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(dict(status=result["status"], macro12=auc["macro12"],
                         gpu_hours_upper_bound=result["gpu_hours_upper_bound"])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "supervised-run", "_infer", "report", "predecessor-status", "hashes"])
    p.add_argument("--verification", type=Path)
    p.add_argument("--registration", type=Path, default=EXP / "registration-v1.json")
    p.add_argument("--out", type=Path, default=EXP / "run-v1")
    args = p.parse_args(argv)
    if args.command == "hashes":
        print(json.dumps(dict(version=VERSION, code_sha256=code_hashes()), indent=2)); return
    if args.command == "predecessor-status":
        text = AUTOCHAIN.read_text(encoding="utf-8") if AUTOCHAIN.exists() else ""
        print(predecessor_state(text, PREDECESSOR_SUCCESS.exists())); return
    need(args.verification is not None, "--verification is required")
    {"prepare": prepare, "supervised-run": supervised_run, "_infer": infer, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
