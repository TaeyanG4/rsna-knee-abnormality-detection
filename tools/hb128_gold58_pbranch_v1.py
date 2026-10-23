"""H-B-128 Gold58 P-branch diagnostic v1.0.0: R, S6 and R7 served as on Kaggle, plus P956 / P_R7 / P3.

Members (all served exactly like their Kaggle releases; model/window/TTA/stacker math untouched):
- R  : nartaa public checkpoint 7e5315da..., P956's R serving (public-serving.py via exp169_release_p956
       .r_runtime_source, executed in memory as H-B-113 does): native 320 centre crop, K94, plain + anatomical
       mirror probability mean, CUDA autocast fp16. Runs in its own child process (`r-worker`).
- S6 : heliosli compact bundle 209a32c8b50a6b1c through tools/exp189_s6_port.py (pinned, unchanged).
- R7 : heliosli compact bundle 8c7b666e27dc0e8f (manifest c4bb0f73...) through tools/exp169_r7_port.py
       (pinned, unchanged): state spec, side classifier, SKM-TEA anatomy anchors, fluid + state stackers.
Blends (percentile ranks over the scored cohort, public cell 7 failure rule generalised to renormalised
survivor weights; equal to exp169_release_p956.public_formula for the two-member blends):
  P956 = .5 r(R) + .5 r(S6);  P_R7 = .5 r(R) + .5 r(R7);  P3 = .5 r(R) + .25 r(S6) + .25 r(R7).

Local-serving choices (recorded in every receipt): amp fp16 for S6/R7 (Kaggle T4 'auto' resolves to fp16;
'auto' on the local Ada GPU would pick bf16); loader workers 2 as served; one local GPU -> one S worker
(Kaggle: two T4 workers; per-study math is unchanged); the dependency wheels pinned by exp189 (pydicom
3.0.1, timm 1.0.25) are extracted into a scratch dir on the S workers' PYTHONPATH (no pip install into the
system interpreter); R runs on the system interpreter like P956's R child, which runs before the wheel
install. R7's manifest asks for torch.compile; its own gate falls back to eager when compile is unavailable
(the fallback reason is recorded). No deadlines, no network, no Kaggle.

Scope: Gold58 is EXPOSED and tiny. R was trained on all data including Gold58 (report labels); S6/R7
manifests state gold_in_training 0 (author-attributed, unaudited); public-LB selection touches all three.
Everything here is descriptive: no release, submission or weight decision is made by this tool.

Commands (B runs .local/run-scripts/hb128-gold58-v1.ps1):
  stage-dicom   CPU: extract the 58 Gold studies from the competition ZIP (CRC-checked) into scratch
  prepare       CPU: extract + pin-check the public code dataset copy and the two wheels into scratch
  register      CPU: write-once registration (roster, formulas, rules, asset/source/reference pins)
  validate      CPU: re-check registration pins
  infer         GPU: --set visible|gold, members R/S6/R7 -> predictions.npz + receipt.json
  parity        CPU: visible-3 local vs Kaggle raw outputs and final P956 / P_R7 CSVs
  score         CPU: Gold58 AUCs, paired bootstrap CIs, Spearman S6 vs R7, rule readouts

Changelog
- v1.0.0 (Claude opus builder stand-in for B, 2026-10-13): initial build.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import zlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp169_release_p956 as p956
from tools import exp189_s6_port as s6
from tools import exp169_r7_port as r7
from tools import hb113_gold58 as hb113

VERSION = "1.0.0"
NAME = "H_B_128_GOLD58_PBRANCH"
STAGE = ROOT / "artifacts/experiments/H_B_128_GOLD58_PBRANCH/v1.0.0"
SCRATCH = ROOT / ".local/scratch/claude-hb128"
GOLD_DICOM = SCRATCH / "dicom"
RUNTIME = SCRATCH / "runtime"
ZIP = ROOT / "storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip"
SERIES = ROOT / "storage/competition/train_series.csv"
LABELS_CSV = ROOT / "storage/competition/train.csv"
VISIBLE_DATA = ROOT / ".local/exp189-cpu-v120/input/visible_data"
VISIBLE_SERIES_SHA = "d8b0402ccd4d1383bf8b154b781e53f156a5438778fadca04fca3c64592f4f2d"
CODE_SOURCE = ROOT / ".local/exp189-cpu-v120/prepare_smoke/dataset_mount"
S6_BUNDLE = ROOT / "storage/weights/external/heliosli_rsna-knee-compact-v1_v10-equivalent"
R7_BUNDLE = ROOT / "storage/external/heliosli-r7cnxfsss-v1"
R7_MANIFEST_SHA = "c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6"
R_WEIGHTS = hb113.PARENT
UID, LABELS = s6.UID, tuple(s6.LABELS)
WEAK = (3, 5, 6, 8)  # Lateral Meniscus, Lateral OA, PF OA, Synovitis (multi_lens_readout.WEAK)
MEMBERS = ("R", "S6", "R7")
BLENDS = {"P956": {"R": .5, "S6": .5}, "P_R7": {"R": .5, "R7": .5}, "P3": {"R": .5, "S6": .25, "R7": .25}}
CANDIDATES = MEMBERS + tuple(BLENDS)
CONTRASTS = (("R7", "S6"), ("P_R7", "P956"), ("P3", "P_R7"), ("P3", "P956"), ("S6", "R"), ("R7", "R"))
CORRELATIONS = (("S6", "R7"), ("R", "S6"), ("R", "R7"))
AMP = "fp16"
LOADER_WORKERS = 2
RAW_ATOL = 5e-3        # registered fp16 cross-GPU tolerance on raw member probabilities
FINAL_ATOL = 1e-6      # final rank CSVs must match the Kaggle files (float32 rounding only)
W4_MARGIN = .005       # I-3 decision rule margin
CORR_GATE = .93        # H-B-128 / E7 decorrelation gate
REFERENCES = {
    "R_raw": (".local/kout-p956-v1/p956-run/r/raw.npz",
              "4f7fda7d673722d486a32e2eab404c7779914b2416c75ad9d479fbc546a0d600"),
    "S6_raw": (".local/kout-p956-v1/p956-run/s6/predictions.csv", p956.S6_VISIBLE_SHA),
    "P956_final": (".local/kout-p956-v1/submission.csv",
                   "5c4828369e39d80a6e67ece4e784206e99323120aff975216005755f7f7053dd"),
    "R7_raw": (".local/scratch/slot-b/c85r7-visible/c85r7-run/pr7/pr7-run/s_r7/predictions.csv",
               "fa4d89746bef047af6b14866c3272d9818aa2af433dfe33c4fe4ed0593ac5ab3"),
    "P_R7_final": (".local/external/heliosli-r7-20261013/output/submission.csv",
                   "1199c2d767b5a53219d777b6041593cc392f56eff09228bda88d68c90d5f3cc8"),
}
CAVEAT = ("Gold58 is exposed and tiny: R was trained on all data including Gold58 with report labels; S6/R7 "
          "manifests state gold_in_training 0 (author-attributed, unaudited); public-LB selection touches all "
          "members. Fixed-cohort rank blends; paired study bootstrap CIs are conditional diagnostics. "
          "Descriptive only: no release, submission or weight decision follows mechanically.")
need, sha = p956.need, p956.sha


# ----------------------------------------------------------------------------- small helpers
def jwrite(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def scratch(path, root: Path | None = None) -> Path:
    root = (SCRATCH if root is None else Path(root)).resolve()
    path = Path(path).resolve()
    need(path.is_relative_to(root) and path != root, "output must be below " + str(root))
    return path


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def tree_digest(lines: list[str]) -> str:
    return hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()


def dir_tree(root: Path) -> dict:
    """sha256 of every file below root (no symlinks) and one digest over 'relpath\\0sha\\n' lines."""
    manifest = s6.code_tree_manifest(root)
    return {"tree_sha256": manifest["tree_sha256"], "files": manifest["files"]}


def gold():
    """Official Gold58 roster (sorted UIDs) and labels; labels never leave this process except as aggregates."""
    return hb113.gold()


# ----------------------------------------------------------------------------- rank blends and statistics
def pct(values) -> np.ndarray:
    """pandas Series.rank(method='average', pct=True)."""
    from scipy.stats import rankdata
    values = np.asarray(values, dtype=np.float64)
    need(values.ndim == 1 and len(values) and np.isfinite(values).all(), "rank inputs must be finite and nonempty")
    return rankdata(values, method="average") / len(values)


def rank_blend(members: dict, weights: dict, failed: dict | None = None) -> np.ndarray:
    """Weighted percentile-rank blend; each member ranks only its scored rows; per-study survivor weights are
    renormalised (public cell 7 for two members); all members failed -> 0.5. Returns float32 like the CSVs."""
    need(weights and abs(sum(weights.values()) - 1) < 1e-12 and all(w > 0 for w in weights.values()), "blend weights")
    arrays = {m: np.asarray(members[m], dtype=np.float64) for m in weights}
    n = len(next(iter(arrays.values())))
    need(all(a.shape == (n, len(LABELS)) for a in arrays.values()), "blend member shapes differ")
    masks = {m: np.zeros(n, bool) if not failed or m not in failed else np.asarray(failed[m], bool) for m in weights}
    out = np.empty((n, len(LABELS)))
    for j in range(len(LABELS)):
        num, den = np.zeros(n), np.zeros(n)
        for m, w in weights.items():
            ok = ~masks[m]
            ranks = np.zeros(n)
            if ok.any():
                ranks[ok] = pct(arrays[m][ok, j])
            num = num + np.where(ok, w * ranks, 0.)
            den = den + np.where(ok, w, 0.)
        column = np.full(n, .5)
        alive = den > 0
        column[alive] = num[alive] / den[alive]
        out[:, j] = column
    return out.astype(np.float32)


def candidates(members: dict, failed: dict | None = None) -> dict:
    result = {m: np.asarray(members[m], dtype=np.float64) for m in MEMBERS}
    result.update({name: rank_blend(members, w, failed).astype(np.float64) for name, w in BLENDS.items()})
    return result


def summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    out = hb113.summary(values)
    valid = values[1:][np.isfinite(values[1:])]
    out["share_positive"] = float((valid > 0).mean()) if len(valid) else None
    return out


def bootstrap_weights(ids):
    from tools import multi_lens_readout as lens
    return lens.weights(list(ids), [{"component_id": u, "members": [u]} for u in ids])


def auc_table(preds: dict, y: np.ndarray, w: np.ndarray) -> dict:
    from tools import multi_lens_readout as lens
    mask = np.isfinite(y)
    return {name: lens.auc_draws(np.asarray(p, dtype=np.float64), y, mask, w) for name, p in preds.items()}


def block(draws: np.ndarray) -> dict:
    """Per-target, macro12 and weak-four summaries of a (draws+1, 12) AUC/delta matrix. np.mean (not nanmean):
    a draw lacking a class on any fixed target is invalid for that macro, so target sets never shrink."""
    return {"targets": {t: summary(draws[:, i]) for i, t in enumerate(LABELS)},
            "macro12": summary(draws.mean(1)), "weak_four": summary(draws[:, list(WEAK)].mean(1))}


def spearman_table(preds: dict) -> dict:
    from scipy.stats import spearmanr
    out = {}
    for a, b in CORRELATIONS:
        per = {}
        for i, t in enumerate(LABELS):
            rho = spearmanr(preds[a][:, i], preds[b][:, i]).statistic
            per[t] = float(rho) if np.isfinite(rho) else None
        values = np.array([np.nan if v is None else v for v in per.values()])
        weak = values[list(WEAK)]
        out[f"{a}_vs_{b}"] = {"targets": per,
                              "weak_four_mean": float(weak.mean()) if np.isfinite(weak).all() else None,
                              "macro12_mean": float(values.mean()) if np.isfinite(values).all() else None,
                              "median12": float(np.median(values)) if np.isfinite(values).all() else None,
                              "min12": float(values.min()) if np.isfinite(values).all() else None}
    return out


def readouts(stats: dict, corr: dict) -> dict:
    """Registered mechanical readouts (plan.md H-B-128, PROPOSALS I-3/I-4, E7). Inputs to B's decision only."""
    def obs(path):
        node = stats
        for key in path:
            node = node[key]
        return node["observed"]
    d_w4, d_mac = obs(("contrasts", "R7-S6", "weak_four")), obs(("contrasts", "R7-S6", "macro12"))
    if d_w4 is None or d_mac is None:
        i3 = "UNAVAILABLE"
    elif d_w4 >= W4_MARGIN and d_mac >= 0:
        i3 = "R7_ONLY"
    elif d_w4 <= -W4_MARGIN:
        i3 = "KEEP_S6"
    elif abs(d_w4) < W4_MARGIN:
        i3 = "P3_AVERAGE"
    else:
        i3 = "UNSPECIFIED_R7_WEAK_FOUR_UP_MACRO_DOWN"
    rho = corr["S6_vs_R7"]
    p3, pr7 = obs(("candidates", "P3", "macro12")), obs(("candidates", "P_R7", "macro12"))
    return {
        "I3_rule": {"rule": "R7-S6 weak-four >= +0.005 and macro >= 0 -> R7_ONLY; <= -0.005 -> KEEP_S6; |d| < 0.005 -> P3_AVERAGE",
                    "lens": "Gold58 (exposed; the OAI panel is not used because R is OAI-trained)",
                    "delta_weak_four": d_w4, "delta_macro12": d_mac, "readout": i3},
        "HB128_corr_gate": {"rule": "build P3 only if mean Spearman(S6, R7) over the weak four < 0.93 (runtime gate separate)",
                            "weak_four_mean": rho["weak_four_mean"],
                            "decorrelated": None if rho["weak_four_mean"] is None else rho["weak_four_mean"] < CORR_GATE},
        "E7_gate": {"rule": "median per-target Spearman(S6, R7) < 0.93 and P3 macro >= P_R7 macro",
                    "median12": rho["median12"], "p3_macro12": p3, "p_r7_macro12": pr7,
                    "met": None if None in (rho["median12"], p3, pr7) else bool(rho["median12"] < CORR_GATE and p3 >= pr7)},
        "note": "Descriptive readouts on an exposed 58-study set; B decides with the runtime and public evidence.",
    }


def score_arrays(ids, y, members: dict, failed: dict | None = None) -> tuple[dict, dict]:
    need(len(ids) == len(set(ids)) and y.shape == (len(ids), len(LABELS)), "score roster/labels shape")
    preds = candidates(members, failed)
    w = bootstrap_weights(ids)
    table = auc_table(preds, y, w)
    stats = {"candidates": {name: block(table[name]) for name in CANDIDATES},
             "contrasts": {f"{a}-{b}": block(table[a] - table[b]) for a, b in CONTRASTS}}
    corr = spearman_table(preds)
    return dict(stats, spearman=corr, readouts=readouts(stats, corr)), preds


# ----------------------------------------------------------------------------- DICOM staging (CPU)
def _safe_member(name: str, idset: set) -> list[str] | None:
    parts = name.split("/")
    if len(parts) != 4 or parts[0] != "train_series" or parts[1] not in idset or not parts[3].endswith(".dcm"):
        return None
    need(all(p and p not in (".", "..") and ":" not in p and "\\" not in p for p in parts), "unsafe ZIP member " + name)
    return parts


def _crc32(path: Path) -> int:
    crc = 0
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def stage_dicom(zip_path: Path = ZIP, dest: Path | None = None, ids=None, series_csv: Path = SERIES) -> dict:
    """Extract only the requested train studies (CRC-checked reads) plus their series metadata rows."""
    ids = list(gold()[0] if ids is None else ids)
    dest = scratch(GOLD_DICOM if dest is None else dest)
    if (dest / "manifest.json").exists():
        return verify_dicom(dest, ids)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(scratch(partial))  # our own interrupted attempt, never source data
    idset, files = set(ids), []
    with zipfile.ZipFile(zip_path) as archive:
        members = [(info, parts) for info in archive.infolist() if (parts := _safe_member(info.filename, idset))]
        need({parts[1] for _, parts in members} == idset, "ZIP lacks requested Gold studies")
        need(len({info.filename for info, _ in members}) == len(members), "duplicate ZIP members")
        for info, parts in members:
            target = partial.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(info)  # zipfile verifies the stored CRC32
            need(len(data) == info.file_size and (zlib.crc32(data) & 0xFFFFFFFF) == info.CRC, "ZIP member CRC/size")
            with target.open("xb") as f:
                f.write(data)
            files.append(("/".join(parts), info.file_size, info.CRC))
    frame = pd.read_csv(series_csv, dtype=str, keep_default_na=False)
    need(list(frame.columns[:2]) == [UID, "SeriesInstanceUID"] and "Anatomical_Plane" in frame.columns, "series CSV schema")
    subset = frame.loc[frame[UID].isin(idset)]
    need(set(subset[UID]) == idset, "series metadata lacks Gold studies")
    subset.to_csv(partial / "train_series.csv", index=False, lineterminator="\n")
    manifest = {"version": VERSION, "zip": rel(zip_path), "zip_size": Path(zip_path).stat().st_size,
                "series_csv_source_sha256": sha(series_csv), "series_csv_sha256": sha(partial / "train_series.csv"),
                "studies": len(ids), "ids": ids, "files": len(files), "bytes": int(sum(f[1] for f in files)),
                "tree_sha256": tree_digest([f"{p}\0{n}\0{c:08x}\n" for p, n, c in files]),
                "entries": sorted([p, n, f"{c:08x}"] for p, n, c in files)}
    jwrite(partial / "manifest.json", manifest)
    partial.rename(dest)
    return verify_dicom(dest, ids)


def verify_dicom(dest: Path = GOLD_DICOM, ids=None) -> dict:
    dest = Path(dest)
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    if ids is not None:
        need(manifest["ids"] == list(ids), "staged Gold roster differs")
    need(sha(dest / "train_series.csv") == manifest["series_csv_sha256"], "staged series CSV drift")
    seen = []
    for path in sorted((dest / "train_series").rglob("*")):
        if path.is_file():
            seen.append([path.relative_to(dest).as_posix(), path.stat().st_size, f"{_crc32(path):08x}"])
    need(seen == manifest["entries"], "staged DICOM tree differs from its manifest")
    need({p.name for p in (dest / "train_series").iterdir()} == set(manifest["ids"]), "staged study directories differ")
    return {"status": "DICOM_STAGED", "studies": manifest["studies"], "files": manifest["files"],
            "bytes": manifest["bytes"], "tree_sha256": manifest["tree_sha256"],
            "manifest_sha256": sha(dest / "manifest.json")}


# ----------------------------------------------------------------------------- runtime (code dataset + wheels)
PROBE = ("import importlib.metadata as m, json, pydicom, timm; print(json.dumps({'pydicom': m.version('pydicom'), "
         "'timm': m.version('timm'), 'pydicom_module': pydicom.__version__, 'timm_module': timm.__version__, "
         "'pydicom_file': pydicom.__file__, 'timm_file': timm.__file__}))")


def _extract_wheels(wheels, deps: Path) -> None:
    deps.mkdir(parents=True, exist_ok=False)
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                target = (deps / name).resolve()
                need(target.is_relative_to(deps.resolve()) and ":" not in name, "unsafe wheel member " + name)
            archive.extractall(deps)


def probe_deps(deps: Path, code: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=str(deps) + os.pathsep + str(code), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True).stdout
    versions = json.loads(out.strip().splitlines()[-1])
    expected = {**s6.WHEEL_VERSIONS, **{k + "_module": v for k, v in s6.WHEEL_VERSIONS.items()}}
    need({k: versions[k] for k in expected} == expected, "dependency versions differ: " + json.dumps(versions))
    need(all(Path(versions[k]).resolve().is_relative_to(deps.resolve()) for k in ("pydicom_file", "timm_file")),
         "dependencies did not resolve inside the extracted wheels")
    return {k: versions[k] for k in expected}


def prepare_runtime(runtime: Path = RUNTIME, code_source: Path = CODE_SOURCE) -> dict:
    """Public cell 4's code preparation (exp189 prepare_code: tree/module/anatomy/wheel pins) without pip."""
    runtime = scratch(runtime)
    record_path = runtime / "prepare.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        code, deps = Path(record["code"]), Path(record["deps"])
        s6.verify_code(code, record["anatomy"])
        need(dir_tree(deps)["tree_sha256"] == record["deps_tree_sha256"], "extracted dependency tree drift")
        need(probe_deps(deps, code) == record["versions"], "dependency probe drift")
        return record
    if runtime.exists():
        shutil.rmtree(runtime)  # our own incomplete scratch attempt (no prepare.json)
    code, anatomy = s6.prepare_code(code_source, runtime / "code", install_wheels=False)
    wheels = sorted((runtime / "code" / "wheels").glob("*.whl"))
    s6.verify_wheels(wheels)
    deps = runtime / "deps"
    _extract_wheels(wheels, deps)
    record = {"version": VERSION, "status": "RUNTIME_READY", "code": str(code), "anatomy": str(anatomy),
              "deps": str(deps), "code_tree_sha256": s6.verify_code(code, anatomy)["tree_sha256"],
              "anatomy_sha256": s6.ANATOMY_SHA256, "wheel_sha256": s6.verify_wheels(wheels),
              "deps_tree_sha256": dir_tree(deps)["tree_sha256"], "versions": probe_deps(deps, code),
              "code_source": rel(code_source)}
    jwrite(record_path, record)
    return record


@contextlib.contextmanager
def worker_pythonpath(deps: Path):
    """The ports build each worker's PYTHONPATH from os.environ; prepend the pinned wheels only for them."""
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(deps) + (os.pathsep + old if old else "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old


# ----------------------------------------------------------------------------- registration
def study_set(name: str):
    """(data root, split dir name, ordered study ids) for the visible-3 parity set or the Gold58 set."""
    if name == "visible":
        need(sha(VISIBLE_DATA / "test_series.csv") == VISIBLE_SERIES_SHA, "visible series CSV drift")
        ids = sorted(pd.read_csv(VISIBLE_DATA / "test_series.csv", dtype=str)[UID].unique().tolist())
        need(len(ids) == 3, "visible set must have 3 studies")
        return VISIBLE_DATA, "test_series", ids
    need(name == "gold", "unknown study set")
    return GOLD_DICOM, "train_series", gold()[0]


def source_files() -> dict:
    files = {rel(Path(__file__)): Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(ROOT / "tools") and path.suffix == ".py":
                files[rel(path)] = path
    for folder in (s6.SOURCE, r7.SOURCE):
        for path in sorted(folder.glob("*.py")):
            files[rel(path)] = path
    files[rel(hb113.SERVING)] = hb113.SERVING
    return dict(sorted(files.items()))


def asset_pins() -> dict:
    s6.verify_bundle(S6_BUNDLE)
    _, identity = r7.verify_bundle(R7_BUNDLE, R7_MANIFEST_SHA)
    p956.check_pin(R_WEIGHTS, p956.R_SHA)
    files = {rel(R_WEIGHTS): p956.R_SHA}
    files.update({rel(S6_BUNDLE / n): d for n, d in s6.HASHES.items()})
    files.update({rel(S6_BUNDLE / "manifest.json"): s6.MANIFEST_SHA256, rel(S6_BUNDLE / "README.md"): s6.README_SHA256})
    files.update({rel(R7_BUNDLE / n): d for n, d in identity["file_sha256"].items()})
    for path in sorted(CODE_SOURCE.iterdir()):
        files[rel(path)] = sha(path)
    for path in (LABELS_CSV, SERIES, VISIBLE_DATA / "test_series.csv"):
        files[rel(path)] = sha(path)
    return dict(sorted(files.items()))


def register() -> dict:
    ids, y = gold()
    dicom = verify_dicom(GOLD_DICOM, ids)
    prepared = prepare_runtime()
    binding = {
        "version": VERSION, "name": NAME, "caveat": CAVEAT, "gold_count": len(ids), "ids": ids,
        "targets": list(LABELS), "weak_indices": list(WEAK), "members": list(MEMBERS),
        "blends": BLENDS, "contrasts": [f"{a}-{b}" for a, b in CONTRASTS], "correlations": [f"{a}_vs_{b}" for a, b in CORRELATIONS],
        "rank": "percentile average rank over the scored cohort; renormalised survivor weights; float32",
        "bootstrap": {"draws": 300, "seed": 20261003, "unit": "Gold study singleton", "ci": "percentile 95%",
                      "fixed_ranks": "blend ranks computed once on the full cohort, then resampled"},
        "serving": {"R": "P956 R native320 / K94 / plain+anatomical mirror mean / autocast fp16 (H-B-113 path)",
                    "S6": f"exp189_s6_port {s6.VERSION} bundle {s6.BUNDLE_ID}",
                    "R7": f"exp169_r7_port {r7.VERSION} bundle {r7.BUNDLE_ID} manifest {R7_MANIFEST_SHA}",
                    "amp": AMP, "loader_workers": LOADER_WORKERS, "max_fallback_fraction": 1.0},
        "parity": {"raw_atol": RAW_ATOL, "final_atol": FINAL_ATOL, "references": {k: {"path": p, "sha256": d} for k, (p, d) in REFERENCES.items()},
                   "gold_gate": "infer --set gold requires parity status PASS unless --accept-parity names the observed status"},
        "rules": {"W4_MARGIN": W4_MARGIN, "CORR_GATE": CORR_GATE,
                  "I3": "R7-S6 weak-four >= +0.005 and macro >= 0 -> R7_ONLY; <= -0.005 -> KEEP_S6; |d| < 0.005 -> P3_AVERAGE",
                  "HB128": "P3 only if mean weak-four Spearman(S6,R7) < 0.93", "E7": "median12 Spearman < 0.93 and P3 macro >= P_R7 macro"},
        "dicom": dicom, "runtime": {k: prepared[k] for k in ("code_tree_sha256", "anatomy_sha256", "wheel_sha256",
                                                             "deps_tree_sha256", "versions")},
        "visible_tree": dir_tree(VISIBLE_DATA / "test_series"),
        "assets": asset_pins(), "sources": {name: sha(path) for name, path in source_files().items()},
    }
    p956.write_once(STAGE / "registration.json", p956.json_bytes(clean(binding)))
    return {"status": "REGISTERED", "studies": len(ids), "binding_sha256": sha(STAGE / "registration.json")}


def validate(assets: bool = False) -> dict:
    binding = json.loads((STAGE / "registration.json").read_text(encoding="utf-8"))
    need(binding["version"] == VERSION and binding["caveat"] == CAVEAT, "binding policy drift")
    need(binding["targets"] == list(LABELS) and binding["blends"] == BLENDS, "target/blend drift")
    for name, digest in binding["sources"].items():
        p956.check_pin(ROOT / name, digest)
    for key, (path, digest) in REFERENCES.items():
        need(binding["parity"]["references"][key] == {"path": path, "sha256": digest}, "reference pin drift")
        p956.check_pin(ROOT / path, digest)
    need(gold()[0] == binding["ids"], "Gold roster drift")
    if assets:
        for name, digest in binding["assets"].items():
            p956.check_pin(ROOT / name, digest)
        need(dir_tree(VISIBLE_DATA / "test_series") == binding["visible_tree"], "visible DICOM tree drift")
        need(verify_dicom(GOLD_DICOM, binding["ids"]) == binding["dicom"], "staged Gold DICOM drift")
    return binding


# ----------------------------------------------------------------------------- inference (GPU)
def r_worker(set_name: str, output: Path, limit: int | None) -> dict:
    """Child process: P956 R served on the requested studies (H-B-113's in-memory serving path)."""
    import torch
    need(torch.cuda.is_available(), "R worker requires CUDA (P956 R asserts CUDA/fp16)")
    output = scratch(output)
    output.mkdir(parents=True, exist_ok=False)
    root, split, ids = study_set(set_name)
    ids = ids[:limit] if limit else ids
    torch.set_num_threads(4)
    p956.check_pin(R_WEIGHTS, p956.R_SHA)
    module = hb113.runtime()
    module._E152_FILES = {"R": R_WEIGHTS}
    module._E152_MANIFEST = {"checkpoints": {"R": {"sha256": p956.R_SHA}}}
    native_infer = module.infer_probs
    metadata = pd.read_csv(root / (split + ".csv"), dtype={UID: str, "SeriesInstanceUID": str})
    records = {u: f.to_dict("records") for u, f in metadata.loc[metadata[UID].isin(ids)].groupby(UID)}
    need(set(records) == set(ids), "missing series metadata")
    decoder = hb113.fastread(module)
    reader = decoder.make_builder(module.FASTREAD_VARIANT, module.IMG, module.SLOTS,
                                  hdr_threads=module.FR_HDR, px_threads=module.FR_PX)
    model = module.load_model(str(R_WEIGHTS), hb113.helper.ARCH, 384, "cuda")[0]
    arrays, inputs, times = {"R": [], "R_plain": [], "R_mirror": []}, [], []
    started = time.perf_counter()
    for index, uid in enumerate(ids):
        tick = time.perf_counter()
        vol, mask = reader(uid, records, str(root / split))
        need(vol.shape == (96, 384, 384) and vol.dtype == np.uint8 and mask.any(), "empty or invalid P956 decode")
        plain = module.gpu_windows(vol, mask, 94, 320, "cuda")
        mirror = module.gpu_windows(vol, mask, 94, 320, "cuda", mirror=True)
        inputs.append(hb113.helper.input_record(module.__dict__, vol, mask, plain.cpu().numpy(), mirror.cpu().numpy()))
        views = []

        def capture(m, x, d):
            p = native_infer(m, x, d)
            views.append(p)
            return p
        module.infer_probs = capture
        mean = module.infer_pair_probs(model, plain, mirror, "cuda")
        arrays["R"].append(mean)
        arrays["R_plain"].append(views[0])
        arrays["R_mirror"].append(views[1])
        times.append(time.perf_counter() - tick)
        print(json.dumps({"member": "R", "completed": index + 1, "total": len(ids), "seconds": times[-1]}), flush=True)
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    need(np.array_equal(arrays["R"], .5 * (arrays["R_plain"] + arrays["R_mirror"])), "R TTA mean mismatch")
    need(np.isfinite(arrays["R"]).all() and ((arrays["R"] >= 0) & (arrays["R"] <= 1)).all(), "invalid R probabilities")
    np.savez(output / "raw.npz", ids=np.asarray(ids), targets=np.asarray(LABELS), **arrays)
    receipt = {"status": "PASS_R", "version": VERSION, "studies": len(ids), "raw_sha256": sha(output / "raw.npz"),
               "seconds": time.perf_counter() - started, "seconds_per_study": times, "inputs": inputs,
               "device": torch.cuda.get_device_name(), "torch": torch.__version__,
               "precision": "P956 CUDA autocast fp16", "decode": "P956 embedded study_lut, DICOM directory"}
    jwrite(output / "receipt.json", receipt)
    return {k: receipt[k] for k in ("status", "studies", "seconds", "raw_sha256")}


def _controller_summary(directory: Path) -> dict:
    controller = json.loads((directory / "controller" / "submission_receipt.json").read_text(encoding="utf-8"))
    workers = []
    for w in controller.get("workers", []):
        c = w.get("compile") or {}
        workers.append({k: w.get(k) for k in ("status", "amp", "gpu", "torch", "timm", "scored", "anchor_methods", "stats")}
                       | {"compile_active": c.get("active"), "compile_decision": c.get("decision"),
                          "compile_fallback": c.get("fallback")})
    try:
        anchors = s6.anatomy_fallbacks(controller)
    except Exception as exc:  # telemetry only, as in the ports
        anchors = {"unavailable": repr(exc)[:200]}
    return {"status": controller.get("status"), "rows": controller.get("rows"), "scored": controller.get("scored"),
            "fallback_studies": controller.get("fallback_studies"),
            "fallback_reasons": controller.get("fallback_reasons", {}), "worker_exit_codes": controller.get("worker_exit_codes"),
            "gpus": controller.get("gpus"), "seconds": controller.get("seconds"), "anchors": anchors, "workers": workers}


def parity_gate(accept: str | None) -> dict:
    path = SCRATCH / "runs" / "visible" / "parity.json"
    need(path.exists(), "visible parity has not been run")
    parity = json.loads(path.read_text(encoding="utf-8"))
    need(parity["status"] == "PASS" or (accept and parity["status"] == accept),
         f"visible parity status {parity['status']}; rerun or pass --accept-parity {parity['status']} after review")
    return {"status": parity["status"], "accepted_override": parity["status"] != "PASS", "sha256": sha(path)}


def infer(set_name: str, output: Path, members=MEMBERS, limit: int | None = None,
          loader_workers: int = LOADER_WORKERS, accept_parity: str | None = None) -> dict:
    binding = validate(assets=True)
    output = scratch(output)
    need(not output.exists(), "run output is immutable; choose a fresh directory")
    need(set(members) <= set(MEMBERS) and members, "unknown members")
    full = limit is None and tuple(members) == MEMBERS
    gate = parity_gate(accept_parity) if set_name == "gold" and full else None
    root, split, ids = study_set(set_name)
    ids = ids[:limit] if limit else ids
    prepared = prepare_runtime()
    need(all(prepared[k] == binding["runtime"][k] for k in binding["runtime"]), "prepared runtime differs from registration")
    code, anatomy, deps = Path(prepared["code"]), Path(prepared["anatomy"]), Path(prepared["deps"])
    dirs = [root / split / uid for uid in ids]
    output.mkdir(parents=True)
    jwrite(output / "receipt.json", {"status": "RUNNING", "set": set_name, "members": list(members)})
    seconds, arrays, failed, details = {}, {}, {}, {}
    started = time.perf_counter()
    if "R" in members:
        tick = time.perf_counter()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        args = [sys.executable, "-B", str(Path(__file__).resolve()), "r-worker", "--set", set_name, "--output", str(output / "r")]
        s6._run_group(args + (["--limit", str(limit)] if limit else []), env, label="HB128 R")
        receipt = json.loads((output / "r" / "receipt.json").read_text(encoding="utf-8"))
        p956.check_pin(output / "r" / "raw.npz", receipt["raw_sha256"])
        with np.load(output / "r" / "raw.npz", allow_pickle=False) as raw:
            need(raw["ids"].tolist() == ids and raw["targets"].tolist() == list(LABELS), "R roster/targets")
            for key in ("R", "R_plain", "R_mirror"):
                arrays[key] = raw[key].astype(np.float64)
        failed["R"] = np.zeros(len(ids), bool)
        details["R"] = {k: receipt[k] for k in ("status", "device", "torch", "precision", "seconds")}
        seconds["R"] = time.perf_counter() - tick
    for member in ("S6", "R7"):
        if member not in members:
            continue
        tick = time.perf_counter()
        with worker_pythonpath(deps):
            if member == "S6":
                rows = s6.infer_studies(dirs, bundle_dir=S6_BUNDLE, code_dir=code, anatomy_weights=anatomy,
                                        output_dir=output / "s6", workers=loader_workers, amp=AMP, device="cuda",
                                        max_fallback_fraction=1.0)
            else:
                rows = r7.infer_studies(dirs, bundle_dir=R7_BUNDLE, code_dir=code, anatomy_weights=anatomy,
                                        output_dir=output / "r7", workers=loader_workers, amp=AMP, device="cuda",
                                        max_fallback_fraction=1.0, expected_manifest_sha256=R7_MANIFEST_SHA)
        need([row[UID] for row in rows] == ids, member + " row order")
        arrays[member] = np.asarray([[row[t] for t in LABELS] for row in rows], dtype=np.float64)
        summary_ = _controller_summary(output / member.lower())
        failed[member] = np.asarray([uid in summary_["fallback_reasons"] for uid in ids])
        details[member] = summary_
        seconds[member] = time.perf_counter() - tick
    np.savez(output / "predictions.npz", ids=np.asarray(ids), targets=np.asarray(LABELS),
             **arrays, **{"failed_" + m: failed[m] for m in failed})
    import torch  # version telemetry only; the parent never initialises CUDA
    receipt = {"status": "INFERENCE_COMPLETE" if full else "PARTIAL_SMOKE", "version": VERSION, "set": set_name,
               "members": list(members), "studies": len(ids), "limit": limit, "loader_workers": loader_workers,
               "amp": AMP, "binding_sha256": sha(STAGE / "registration.json"), "parity_gate": gate,
               "prediction_sha256": sha(output / "predictions.npz"), "fallback_counts": {m: int(failed[m].sum()) for m in failed},
               "seconds": seconds, "total_seconds": time.perf_counter() - started, "members_detail": details,
               "torch": torch.__version__, "runtime_versions": prepared["versions"], "caveat": CAVEAT}
    jwrite(output / "receipt.json", receipt)
    return {k: receipt[k] for k in ("status", "set", "studies", "fallback_counts", "total_seconds", "prediction_sha256")}


def load_run(run: Path, set_name: str) -> tuple[list, dict, dict, dict]:
    run = scratch(run)
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    need(receipt["status"] == "INFERENCE_COMPLETE" and receipt["set"] == set_name, "complete run required")
    need(receipt["binding_sha256"] == sha(STAGE / "registration.json"), "run binding drift")
    p956.check_pin(run / "predictions.npz", receipt["prediction_sha256"])
    with np.load(run / "predictions.npz", allow_pickle=False) as data:
        need(data["targets"].tolist() == list(LABELS), "target order")
        ids = data["ids"].tolist()
        members = {m: data[m].astype(np.float64) for m in MEMBERS}
        failed = {m: data["failed_" + m].astype(bool) for m in MEMBERS}
    for m in MEMBERS:
        need(members[m].shape == (len(ids), len(LABELS)) and np.isfinite(members[m]).all()
             and ((members[m] >= 0) & (members[m] <= 1)).all(), "invalid member probabilities: " + m)
    return ids, members, failed, receipt


# ----------------------------------------------------------------------------- parity (CPU)
def _csv(path: Path, ids) -> np.ndarray:
    frame = pd.read_csv(path, dtype={UID: str}, float_precision="round_trip")
    need(list(frame.columns) == [UID, *LABELS] and sorted(frame[UID]) == sorted(ids), "reference CSV schema/cohort")
    return frame.set_index(UID).loc[list(ids), list(LABELS)].to_numpy(np.float64)


def references(ids) -> dict:
    out = {}
    for key, (path, digest) in REFERENCES.items():
        p956.check_pin(ROOT / path, digest)
        if key == "R_raw":
            with np.load(ROOT / path, allow_pickle=False) as raw:
                rid = raw["ids"].tolist()
                need(sorted(rid) == sorted(ids) and raw["targets"].tolist() == list(LABELS), "R reference cohort")
                out[key] = raw["public"][[rid.index(u) for u in ids]].astype(np.float64)
        else:
            out[key] = _csv(ROOT / path, ids)
    return out


def compare(ids, members: dict, failed: dict, refs: dict) -> dict:
    need(not any(f.any() for f in failed.values()), "parity requires zero fallbacks")
    local = {"R_raw": members["R"], "S6_raw": members["S6"], "R7_raw": members["R7"],
             "P956_final": rank_blend(members, BLENDS["P956"]).astype(np.float64),
             "P_R7_final": rank_blend(members, BLENDS["P_R7"]).astype(np.float64)}
    rows = {}
    for key, value in local.items():
        diff = np.abs(value - refs[key])
        atol = RAW_ATOL if key.endswith("_raw") else FINAL_ATOL
        rows[key] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()), "atol": atol,
                     "within": bool(diff.max() <= atol), "argmax": [ids[int(diff.argmax() // len(LABELS))][-12:],
                                                                   LABELS[int(diff.argmax() % len(LABELS))]]}
        if key.endswith("_raw"):
            same = [bool((np.argsort(np.argsort(value[:, j])) == np.argsort(np.argsort(refs[key][:, j]))).all())
                    for j in range(len(LABELS))]
            rows[key]["rank_order_identical_targets"] = int(sum(same))
    finals = rows["P956_final"]["within"] and rows["P_R7_final"]["within"]
    raws = all(rows[k]["within"] for k in ("R_raw", "S6_raw", "R7_raw"))
    status = "PASS" if finals and raws else ("FINAL_MATCH_RAW_DRIFT" if finals else "FAIL")
    return {"status": status, "comparisons": rows}


def parity(run: Path) -> dict:
    validate()
    ids, members, failed, receipt = load_run(run, "visible")
    result = compare(ids, members, failed, references(ids))
    p3 = rank_blend(members, BLENDS["P3"])
    report = {"version": VERSION, "studies": len(ids), "run_receipt_sha256": sha(scratch(run) / "receipt.json"),
              "p3_local_sha256": hashlib.sha256(p3.tobytes()).hexdigest(),
              "anchors": {m: receipt["members_detail"][m].get("anchors") for m in ("S6", "R7")},
              "compile": [w.get("compile_fallback") for w in receipt["members_detail"]["R7"]["workers"]],
              "scope": "visible-3 local CUDA vs Kaggle T4 outputs; ranks over 3 studies are coarse", **result}
    jwrite(SCRATCH / "runs" / "visible" / "parity.json", report)
    return {"status": report["status"], **{k: v["max_abs_diff"] for k, v in report["comparisons"].items()}}


# ----------------------------------------------------------------------------- scoring (CPU)
def markdown(result: dict) -> str:
    def f(x, d=4):
        return "n/a" if x is None else f"{x:+.{d}f}" if d == 5 else f"{x:.{d}f}"
    lines = [f"# H-B-128 Gold58 P-branch diagnostic v{VERSION}", "", result["caveat"], "",
             "| candidate | macro12 [95% CI] | weak-four [95% CI] |", "|---|---|---|"]
    for name in CANDIDATES:
        c = result["candidates"][name]
        lines.append(f"| {name} | {f(c['macro12']['observed'])} [{', '.join(f(x) for x in c['macro12']['ci95'] or [None, None])}] | "
                     f"{f(c['weak_four']['observed'])} [{', '.join(f(x) for x in c['weak_four']['ci95'] or [None, None])}] |")
    lines += ["", "| contrast | d macro12 [95% CI] | d weak-four [95% CI] | P(d>0) weak-four |", "|---|---|---|---|"]
    for name, c in result["contrasts"].items():
        lines.append(f"| {name} | {f(c['macro12']['observed'], 5)} [{', '.join(f(x, 5) for x in c['macro12']['ci95'] or [None, None])}] | "
                     f"{f(c['weak_four']['observed'], 5)} [{', '.join(f(x, 5) for x in c['weak_four']['ci95'] or [None, None])}] | "
                     f"{f(c['weak_four']['share_positive'], 3)} |")
    lines += ["", "| target | " + " | ".join(CANDIDATES) + " | rho S6-R7 |", "|---" * (len(CANDIDATES) + 2) + "|"]
    for t in LABELS:
        lines.append(f"| {t} | " + " | ".join(f(result["candidates"][n]["targets"][t]["observed"]) for n in CANDIDATES)
                     + f" | {f(result['spearman']['S6_vs_R7']['targets'][t], 3)} |")
    rho = result["spearman"]["S6_vs_R7"]
    lines += ["", f"Spearman S6 vs R7: weak-four mean {f(rho['weak_four_mean'], 3)}, macro12 mean {f(rho['macro12_mean'], 3)}, "
                  f"median12 {f(rho['median12'], 3)}", "", "Readouts: " + json.dumps(
                      {k: v.get("readout", v.get("decorrelated", v.get("met"))) for k, v in result["readouts"].items() if isinstance(v, dict)})]
    return "\n".join(lines) + "\n"


def score(run: Path, allow_fallbacks: bool = False) -> dict:
    binding = validate()
    ids, members, failed, receipt = load_run(run, "gold")
    gold_ids, y = gold()
    need(ids == gold_ids == binding["ids"], "Gold roster mismatch")
    counts = {m: int(failed[m].sum()) for m in MEMBERS}
    need(allow_fallbacks or not any(counts.values()), f"member fallbacks {counts}; rerun or pass --allow-fallbacks")
    stats, preds = score_arrays(ids, y, members, failed)
    run = scratch(run)
    np.savez(run / "candidates.npz", ids=np.asarray(ids), targets=np.asarray(LABELS),
             **{name: np.asarray(p, dtype=np.float64) for name, p in preds.items()})
    result = {"version": VERSION, "status": "GOLD58_EXPOSED_PBRANCH_DIAGNOSTIC" if not any(counts.values())
              else "GOLD58_DIAGNOSTIC_WITH_FALLBACKS", "caveat": CAVEAT, "studies": len(ids),
              "positives_per_target": {t: int(np.nansum(y[:, i])) for i, t in enumerate(LABELS)},
              "fallback_counts": counts, "receipt_sha256": sha(run / "receipt.json"),
              "binding_sha256": sha(STAGE / "registration.json"), "candidates_sha256": sha(run / "candidates.npz"),
              "parity_gate": receipt.get("parity_gate"), "bootstrap": binding["bootstrap"], **stats}
    jwrite(run / "score.json", result)
    (run / "score.md").write_text(markdown(clean(result)), encoding="utf-8")
    return {"status": result["status"], "score_sha256": sha(run / "score.json"),
            "readouts": {k: v.get("readout", v.get("decorrelated", v.get("met")))
                         for k, v in result["readouts"].items() if isinstance(v, dict)}}


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stage-dicom")
    p.add_argument("--zip", type=Path, default=ZIP)
    sub.add_parser("prepare")
    sub.add_parser("register")
    p = sub.add_parser("validate")
    p.add_argument("--assets", action="store_true")
    p = sub.add_parser("infer")
    p.add_argument("--set", choices=("visible", "gold"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--members", default=",".join(MEMBERS))
    p.add_argument("--limit", type=int)
    p.add_argument("--loader-workers", type=int, default=LOADER_WORKERS)
    p.add_argument("--accept-parity")
    p = sub.add_parser("r-worker")
    p.add_argument("--set", choices=("visible", "gold"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("parity")
    p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--allow-fallbacks", action="store_true")
    a = parser.parse_args(argv)
    if a.command == "stage-dicom":
        result = stage_dicom(a.zip)
    elif a.command == "prepare":
        result = {k: v for k, v in prepare_runtime().items() if k != "wheel_sha256"}
    elif a.command == "register":
        result = register()
    elif a.command == "validate":
        validate(a.assets)
        result = {"status": "VALID", "assets_checked": a.assets}
    elif a.command == "infer":
        members = tuple(m for m in MEMBERS if m in a.members.split(","))
        need(len(members) == len(a.members.split(",")), "unknown member name")
        result = infer(a.set, a.output, members, a.limit, a.loader_workers, a.accept_parity)
    elif a.command == "r-worker":
        result = r_worker(a.set, a.output, a.limit)
    elif a.command == "parity":
        result = parity(a.run)
    else:
        result = score(a.run, a.allow_fallbacks)
    print(json.dumps(clean(result)), flush=True)
    if a.command == "parity" and result["status"] != "PASS":
        raise SystemExit(3)  # report written; the Gold run refuses until B reviews (see --accept-parity)


if __name__ == "__main__":
    main()
