"""1.0.0 thin f0-on-all wrapper around H-B-145; no source-file edits.

Only scheduling and the journal's selected-fold field change. The generated
worker imports this wrapper so its bookkeeping agrees with the parent. Serving
math, rendering, f0 weights and preprocessing remain H-B-145's. No averaging or
post-stacker output is admitted. Initial version: 2026-10-13.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb146_pl_labels_v1 as pl
from tools import hb145_r7_heldout_v1 as hb

# Process-local extension only; never rewrite another lane's module.
hb.fold_k = lambda uid: 0
hb.SCRATCH = pl.WORK
install_dataset = hb.install_dataset
checkpoint = hb.checkpoint
_original_runtime = hb.runtime_source


def runtime_source(stage):
    runtime = _original_runtime(stage)
    path = runtime / "compact_infer.py"
    code = path.read_text(encoding="utf-8")
    old = "from tools import hb145_r7_heldout_v1 as HB"
    pl.need(code.count(old) == 1, "H-B-145 hook changed; review adapter")
    code = code.replace(old, "from tools import hb146_teacher_v1 as HB")
    code += "\n# HB146 wrapper sha256: " + pl.sha(__file__) + "\n"
    compile(code, str(path), "exec")
    path.write_text(code, encoding="utf-8", newline="\n")
    pin_path = runtime / "runtime_sha256.json"
    if pin_path.exists():
        hashes = pl.read(pin_path)
        hashes["compact_infer.py"] = pl.sha(path)
        pl.write(pin_path, hashes)
    return runtime


def journal_pins(out, identity, values, mode):
    """Bind raw CSV to atomic study records without printing target values."""
    meta = pl.pd.read_csv(Path(out) / "raw.csv", usecols=[pl.UID, "fold_k", "status", "output_kind"],
                          dtype={pl.UID: str}).set_index(pl.UID)
    pl.need(meta.index.is_unique and set(meta.index) == set(values), "journal CSV roster differs")
    paths = []
    for uid, value in values.items():
        path = hb.record_path(out, uid)
        rec = pl.read(path)
        expected = 0 if mode == "screen_f0" else pl.hash_fold(uid)
        pl.need(rec.get(pl.UID) == uid and rec.get("identity") == hb.fingerprint(identity)
                and rec.get("fold_k") == expected and meta.loc[uid, "fold_k"] == expected
                and rec.get("status") in ("ok", "dropped series")
                and rec["status"] == meta.loc[uid, "status"]
                and meta.loc[uid, "output_kind"] == "pre_stacker"
                and rec.get("pooled_prior_bias") is True, "journal identity/status differs")
        raw = pl.np.asarray(rec.get("raw"), dtype=pl.np.float32)
        pl.need(raw.shape == (12,) and pl.np.array_equal(raw, value), "journal raw probabilities differ")
        paths.append(path)
    return paths


def seal(out, mode="screen_f0", receipt_out=None):
    out = Path(out)
    identity = pl.read(out / "identity.json")
    run = pl.read(out / "run_receipt.json")
    csv_path = out / "raw.csv"
    values = pl.table(csv_path)
    pl.need(set(values) == set(identity["uids"]) and run["completed"] == run["selected"] == len(values)
            and run["failures"] == 0, "teacher incomplete; failures are not pseudo-labels")
    used = [0] if mode == "screen_f0" else list(range(4))
    runtime = out / "stage/r7_runtime/compact_infer.py"
    pl.need(identity.get("code") is not None and identity["runtime_sha256"] == pl.sha(runtime),
            "unverified code/runtime identity")
    if mode == "screen_f0":
        pl.need("# HB146 wrapper sha256: " + pl.sha(__file__) in runtime.read_text(encoding="utf-8"),
                "teacher wrapper changed")
    paths = [out / "identity.json", out / "run_receipt.json", Path(hb.__file__), Path(__file__), runtime]
    paths += journal_pins(out, identity, values, mode)
    for k in used:
        path = out / f"receipt_f{k}.json"
        r = pl.read(path)
        pl.need(r["models"] == [hb.r7.MODELS[k]], "wrong teacher model")
        paths.append(path)
    pl.need(identity["runner_sha256"] == pl.sha(hb.__file__), "teacher runner changed")
    for k in range(4):
        path = out / f"receipt_f{k}.json"
        pl.need(mode != "screen_f0" or k == 0 or not path.exists(), "mixed teacher run directory")
    receipt = {"version": "1.0.0", "mode": mode, "output_kind": "pre_stacker",
               "model_selection": "f0_only" if mode == "screen_f0" else "sha1_first8_mod4",
               "csv_sha256": pl.sha(csv_path), "complete": True, "failures": 0,
               "source_pool_membership": "manifest_contract_unverified_membership",
               "validation_label_exposure": "pooled_prior_not_clean", "auxiliary_label_exposure": "unknown",
               "oai_ancestry": "unknown", "pins": pl.pins(paths)}
    receipt_path = Path(receipt_out) if receipt_out else (pl.WORK / "teacher-heldout.json" if mode == "production_heldout" else out / "teacher.json")
    pl.need(receipt_path.resolve().is_relative_to(pl.WORK.resolve()), "receipt must stay in H-B-146 scratch")
    pl.write(receipt_path, receipt)
    pl.validate_teacher(csv_path, receipt_path, list(values), mode)
    return {"sealed": True, "mode": mode, "rows": len(values)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "run", "export", "seal", "seal-heldout"])
    ap.add_argument("--out", type=Path, default=pl.WORK / "teacher-f0")
    ap.add_argument("--receipt-out", type=Path)
    a, rest = ap.parse_known_args(argv)
    if a.command.startswith("seal"):
        pl.need(not rest, "unexpected seal arguments")
        print(seal(a.out, "production_heldout" if a.command == "seal-heldout" else "screen_f0", a.receipt_out))
        return 0
    pl.need(a.receipt_out is None, "receipt-out is only for sealing")
    hb.runtime_source = runtime_source
    return hb.main([a.command, "--out", str(a.out), *rest])


if __name__ == "__main__":
    raise SystemExit(main())
