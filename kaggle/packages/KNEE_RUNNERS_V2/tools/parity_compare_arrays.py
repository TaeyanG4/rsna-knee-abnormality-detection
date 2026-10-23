"""Compare outputs from two environments against tolerances fixed BEFORE looking at the results.

    python compare_arrays.py ref.npz cand.npz --tol tol.json --out report.json
    python compare_arrays.py ref.csv cand.csv --id-col id --tol-max 2e-3 --tol-mean 5e-4
    python compare_arrays.py actA.npz actB.npz                # report only (first-divergence search)

Inputs: .npy, .npz (every shared key is compared), .csv (numeric columns, rows aligned by --id-col),
.pt/.pth (tensor or dict of tensors). Integer/bool arrays must match exactly; floats use tolerances.

tol.json example (write it and record its hash before the candidate run):
    {"max_abs": 0.002, "p99_abs": 0.002, "mean_abs": 0.0005, "min_spearman": 0.999}
    Per-key overrides: {"default": {...}, "keys": {"logits": {...}}}

Exit code: 0 pass (or report-only), 1 tolerance failure, 2 structural error (shape/key/NaN mismatch).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

METRICS = ("max_abs", "p99_abs", "mean_abs", "max_rel")


def load(path, id_col=None, key=None):
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".npy":
        return {"array": np.load(p, allow_pickle=False)}
    if suf == ".npz":
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files if key is None or k == key}
    if suf in {".pt", ".pth"}:
        import torch
        obj = torch.load(p, map_location="cpu", weights_only=True)
        if torch.is_tensor(obj):
            return {"tensor": obj.float().numpy() if obj.is_floating_point() else obj.numpy()}
        return {k: (v.float().numpy() if v.is_floating_point() else v.numpy())
                for k, v in obj.items() if torch.is_tensor(v)}
    if suf in {".csv", ".tsv", ".parquet"}:
        import pandas as pd
        df = pd.read_parquet(p) if suf == ".parquet" else pd.read_csv(p, sep="\t" if suf == ".tsv" else ",")
        if id_col:
            if df[id_col].duplicated().any():
                raise ValueError(f"{path}: duplicate ids in {id_col}")
            df = df.sort_values(id_col).reset_index(drop=True)
        num = df.select_dtypes("number")
        if id_col and id_col in num:
            num = num.drop(columns=id_col)
        out = {c: num[c].to_numpy() for c in num.columns}
        if id_col:
            out["__ids__"] = df[id_col].astype(str).to_numpy()
        return out
    raise ValueError(f"unsupported file type: {path}")


def _rank(x):
    try:
        from scipy.stats import rankdata
        return rankdata(x)
    except ImportError:
        r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x)); return r


def _spearman(a, b):
    if a.size < 10 or np.ptp(a) == 0 or np.ptp(b) == 0:  # rank correlation is noise on tiny samples
        return None
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def compare(a, b, tol):
    rep = {"shape": list(a.shape), "dtype": [str(a.dtype), str(b.dtype)]}
    if a.shape != b.shape:
        return {**rep, "error": f"shape mismatch {a.shape} vs {b.shape}"}
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        eq = bool(np.array_equal(a, b))
        return {**rep, "exact_equal": eq, "pass": eq, "rule": "non-numeric: exact"}
    if a.dtype.kind in "biu" and b.dtype.kind in "biu":
        eq = bool(np.array_equal(a, b))
        rep.update(exact_equal=eq, mismatches=int((a != b).sum()), pass_=eq, rule="integer: exact")
        rep["pass"] = rep.pop("pass_")
        return rep
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    fa, fb = np.isfinite(a64), np.isfinite(b64)
    if not np.array_equal(fa, fb) or not np.array_equal(a64[~fa], b64[~fb], equal_nan=True):
        return {**rep, "error": f"non-finite pattern differs (ref {int((~fa).sum())}, cand {int((~fb).sum())})"}
    d = np.abs(a64[fa] - b64[fb])
    if d.size == 0:
        return {**rep, "exact_equal": True, "pass": True}
    denom = np.maximum(np.abs(a64[fa]), 1e-12)
    rep.update(exact_equal=bool(np.array_equal(a, b)), n=int(d.size), max_abs=float(d.max()),
               p99_abs=float(np.quantile(d, .99)), mean_abs=float(d.mean()),
               max_rel=float((d / denom).max()), argmax_flat=int(np.argmax(np.where(fa, np.abs(a64 - b64), -1))))
    if a.ndim == 2 and a.shape[1] <= 512:
        cols = [_spearman(a64[:, j], b64[:, j]) for j in range(a.shape[1])]
        cols = [c for c in cols if c is not None]
        rep["min_spearman"] = min(cols) if cols else None
    elif a.ndim == 1:
        rep["min_spearman"] = _spearman(a64, b64)
    if tol:
        fails = [m for m in METRICS if m in tol and rep[m] > tol[m]]
        if "min_spearman" in tol and rep.get("min_spearman") is not None and rep["min_spearman"] < tol["min_spearman"]:
            fails.append("min_spearman")
        rep.update(pass_=not fails, failed=fails, tolerance=tol)
        rep["pass"] = rep.pop("pass_")
    else:
        rep["pass"] = None
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref"); ap.add_argument("cand")
    ap.add_argument("--id-col"); ap.add_argument("--key", help="compare only this npz key")
    ap.add_argument("--tol", help="tolerance JSON file (preferred: frozen before the run)")
    for m in METRICS:
        ap.add_argument(f"--tol-{m.replace('_abs', '').replace('_', '-')}", dest=m, type=float)
    ap.add_argument("--min-spearman", type=float)
    ap.add_argument("--out")
    ap.add_argument("--top", type=int, default=10, help="print the N worst keys")
    args = ap.parse_args()

    tol_cfg, tol_sha = {}, None
    if args.tol:
        raw = Path(args.tol).read_bytes(); tol_sha = hashlib.sha256(raw).hexdigest(); tol_cfg = json.loads(raw)
    cli = {m: getattr(args, m) for m in METRICS if getattr(args, m) is not None}
    if args.min_spearman is not None:
        cli["min_spearman"] = args.min_spearman
    default_tol = {**tol_cfg.get("default", {k: v for k, v in tol_cfg.items() if k != "keys"}), **cli}

    try:
        A, B = load(args.ref, args.id_col, args.key), load(args.cand, args.id_col, args.key)
    except Exception as e:  # unreadable/missing input is a structural error, not a tolerance failure
        print(f"status=ERROR load failed: {type(e).__name__}: {e}")
        sys.exit(2)
    report = {"ref": args.ref, "cand": args.cand, "tolerance_file_sha256": tol_sha, "keys": {}}
    structural = []
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a or only_b:
        structural.append(f"key mismatch: only_ref={only_a} only_cand={only_b}")
    for k in sorted(set(A) & set(B)):
        tol = {**default_tol, **tol_cfg.get("keys", {}).get(k, {})}
        r = compare(np.asarray(A[k]), np.asarray(B[k]), tol)
        report["keys"][k] = r
        if "error" in r:
            structural.append(f"{k}: {r['error']}")
    fails = [k for k, r in report["keys"].items() if r.get("pass") is False]
    report.update(structural_errors=structural, failed_keys=fails,
                  status="ERROR" if structural else "FAIL" if fails else
                  "PASS" if any(r.get("pass") for r in report["keys"].values()) else "REPORT_ONLY")

    worst = sorted(report["keys"].items(), key=lambda kv: -(kv[1].get("max_abs") or (0 if kv[1].get("exact_equal", True) else np.inf)))
    print(f"status={report['status']} keys={len(report['keys'])} failed={len(fails)} structural={len(structural)}")
    for e in structural:
        print("  ERROR", e)
    first = next((k for k in sorted(report["keys"]) if not report["keys"][k].get("exact_equal", False)), None)
    if first:
        print(f"  first non-identical key (sorted order): {first}")
    for k, r in worst[:args.top]:
        stats = " ".join(f"{m}={r[m]:.3g}" for m in ("max_abs", "p99_abs", "mean_abs") if m in r)
        sp = f" spearman_min={r['min_spearman']:.6f}" if r.get("min_spearman") is not None else ""
        if "mismatches" in r:
            stats += f"mismatches={r['mismatches']}"
        print(f"  {k}: exact={r.get('exact_equal')} pass={r.get('pass')} {stats}{sp}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(2 if structural else 1 if fails else 0)


if __name__ == "__main__":
    main()
