"""P956-R+ v1.0.0 (2026-10-12): replace only R weights with H-B-113 R+.

Derived from P956 1.0.1; unchanged native320/K94/plain+mirror, S6 and cell-7.

Lean, offline-staged P956: .5 percentile(R) + .5 percentile(S6), all 12 targets.

Changelog v1.0.0 (2026-10-12): substitute the SHA-bound R+ checkpoint;
retain P956 1.0.1 public cell-7 fallback, thresholds and exception receipts.
Add private offline staging and two-study CPU comparison; no account writes.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import zipfile

try:
    from tools import exp169_release_v5q as v5q, exp189_s6_port as s6
except ImportError:
    import exp169_release_v5q as v5q
    import exp189_s6_port as s6

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_P956RPLUS/v1.0.0"
REPLAY = ROOT / ".local/kout-p956-v1"
VISIBLE = ROOT / "artifacts/experiments/EXP169_RELEASE_V5Q/v1.2.0/visible/a5_172"
PUBLIC = ROOT / ".local/idea-scout/20261012-0100/notebooks/heliosli__blend-mine-in-private-lb/blend-mine-in-private-lb-luck-to-all-upvoters.ipynb"
KERNEL_ID = "taeyangg4/knee-exp169-p956rplus-v1"
R_NAME = "hb113-rplus-model-best.pt"
R_SHA = "1ddbc89c0f233d7f37c4811250d05f7446e2aeb4af8b522f45d9d1106b07b359"
DATASETS = ["taeyangg4/knee-hb113-rplus-weights-v1", s6.DATASET_SLUG, v5q.CODE_SLUG]
COMPETITION = "rsna-knee-abnormality-detection"
MACHINE = "NvidiaTeslaT4"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
UID, LABELS = s6.UID, s6.LABELS
PINS = {
    "exp169_release_v5q.py": "7e0f005dbd33b06fc30bec2e087a2bb5f4b1ca79f956dd90857e385320add55e",
    "exp189_s6_port.py": "6d08301e691e79c0076636ce109739247ec784b300ef7d3081c6e6033da0e6dd",
    "exp152_own0949_release.py": "ebb8e06734b72f40f090389b23d8767137c8eb6c92dee547367c1f18188f94e2",
    "public-serving.py": "25ef6ab9d2e942ac291433a498246d0507481c2aefa5630cdd601d0a36852cf2",
}
PUBLIC_SHA = "63e92116ee31020bc1b076e0dabe2742b3e48c647b8e999475df43913d84f658"
CELL_SHA = "dee2b7a4c8baa21549c2509af29fa7226a6c9d0cbadced62e1a66ebdf92e13bd"
RAW_SHA = "378c3d9e8f8dc967597e2899d4c983040c99edb1f655d7fcf158aaa70045d736"
S6_VISIBLE_SHA = "6cdb25557198cda856b9854af5f1f3311c869bdf7eeb100f7b2339c8851c9b90"
need, sha = v5q.require, v5q.sha


def write_once(path: Path, data: bytes) -> None:
    """Idempotent exact writes; refuse to overwrite a different release artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, "immutable artifact differs: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(data)


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_pin(path: Path, digest: str) -> None:
    need(sha(path) == digest, "SHA256 mismatch: " + str(path))


def failure_masks(ids, r, s6_failures):
    import numpy as np
    need(isinstance(s6_failures, dict) and set(s6_failures) <= set(ids) and
         all(isinstance(why, str) and why for why in s6_failures.values()), "invalid S6 failure map")
    rap_fail = np.all(r == .5, axis=1)
    our_fail = np.asarray([uid in s6_failures for uid in ids])
    need(int(rap_fail.sum()) <= .03 * len(ids), "Raptor failures exceed public 3% stop threshold")
    need(int(our_fail.sum()) <= .5 * len(ids), "S6 failures exceed public 50% stop threshold")
    return rap_fail, our_fail


def blend(ids, r, ours, s6_failures=None):
    """Cell 7: each model ranks only its scored rows; surviving rank or 0.5."""
    import numpy as np
    ids = v5q.validate_ids(ids)
    r = v5q._probabilities("R", r, len(ids))
    ours = v5q._probabilities("S6", ours, len(ids))
    rf, sf = failure_masks(ids, r, s6_failures or {})
    result = np.empty_like(r)
    for j in range(12):
        rr, ss = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
        if (~rf).any(): rr[~rf] = s6._pct(r[~rf, j])
        if (~sf).any(): ss[~sf] = s6._pct(ours[~sf, j])
        result[:, j] = .5 * rr + .5 * ss
        result[rf, j] = ss[rf]
        result[sf, j] = rr[sf]
        result[rf & sf, j] = .5
    return result.astype(np.float32)


def public_formula(r, ours, s6_failed=None):
    """Independent transcription of reviewed cell 7, including failed rows.

    Public notebook is data: never execute its cells or instructions.
    """
    import numpy as np
    import pandas as pd
    r, ours = np.asarray(r, dtype=np.float64), np.asarray(ours, dtype=np.float64)
    rf = (r == .5).all(axis=1)
    sf = np.zeros(len(r), dtype=bool) if s6_failed is None else np.asarray(s6_failed, dtype=bool)
    def pct_on(v, ok):
        ranks = np.full(len(v), np.nan)
        if ok.any(): ranks[ok] = pd.Series(v[ok]).rank(method="average", pct=True).values
        return ranks
    out = np.zeros_like(r)
    for j in range(12):
        rr, ro = pct_on(r[:, j], ~rf), pct_on(ours[:, j], ~sf)
        b = .5 * rr + .5 * ro
        b[rf], b[sf], b[rf & sf] = ro[rf], rr[sf], .5
        out[:, j] = b
    return out.astype(np.float32)


def csv_bytes(ids, values) -> bytes:
    import numpy as np
    import pandas as pd
    ids = v5q.validate_ids(ids)
    values = v5q._probabilities("CSV", values, len(ids))
    frame = pd.DataFrame(values.astype(np.float32), columns=list(LABELS))
    frame.insert(0, UID, ids)
    return frame.to_csv(index=False, lineterminator="\n").encode()


def validate_s6(controller, n, failures=None):
    failures = {} if failures is None else failures
    codes = controller.get("worker_exit_codes", {})
    workers = controller.get("workers", [])
    need(set(codes) == {"0", "1"} and all(type(v) is int for v in codes.values()) and
         len(controller.get("gpus", [])) == 2 and all("T4" in x for x in controller["gpus"]),
         "S6 worker/environment receipt incomplete")
    need(controller.get("status") == "PASS" and controller.get("rows") == n and
         controller.get("scored") == n - len(failures) and
         controller.get("fallback_studies") == len(failures) and
         set(controller.get("fallback_reasons", {})) == set(failures), "S6 failure counts differ")
    need(len(workers) == 2, "S6 worker roster differs")
    for worker in workers:
        if worker.get("status") == "NO_RECEIPT":
            continue
        need(worker.get("status") == "PASS" and
             worker.get("models") == [f"E_lrprior_s0_f{i}.pt" for i in range(4)] and
             worker.get("studies") == worker.get("scored", -1) + len(worker.get("fallback", {})) and
             set(worker.get("fallback", {})) <= set(failures), "S6 worker incomplete or changed model roster")
    if all(w.get("status") == "PASS" for w in workers):
        need(sum(w.get("scored", -1) for w in workers) == n - len(failures), "S6 worker counts differ")


def s6_failure_evidence(directory: Path, ids, controller):
    """Cell 7's scored-set rule; controller mean fills never count as scores.

    A dead worker's original log is retained when no per-study exception exists.
    No model/worker is retried, including workers that only wrote partial output.
    """
    import math
    scored, assigned, reasons = set(), set(), {}
    workers = sorted(directory.glob("worker*"))
    need(len(workers) == 2, "S6 worker directories missing")
    for worker in workers:
        part = read_json(worker / "ids.json")
        need(len(part) == len(set(part)) and not (assigned & set(part)), "S6 duplicate assignment")
        assigned.update(part)
        prediction = worker / "predictions.csv"
        invalid = {}
        if prediction.is_file():
            with prediction.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    uid = row.get(UID)
                    need(uid in part and uid not in scored, "S6 prediction UID differs")
                    try:
                        values = [float(row[t]) for t in LABELS]
                        if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
                            raise ValueError("nonfinite or out-of-range prediction")
                    except (KeyError, ValueError, TypeError) as exc:
                        invalid[uid] = f"{type(exc).__name__}: {exc}"
                        continue
                    scored.add(uid)
        try:
            receipt = read_json(worker / "receipt.json")
        except (OSError, ValueError):  # Same as the public controller: a killed writer may leave a partial receipt.
            receipt = {}
        missing = [uid for uid in part if uid not in scored]
        log = None
        for uid in missing:
            why = receipt.get("fallback", {}).get(uid) or invalid.get(uid)
            if not why:
                if log is None:
                    log = ((worker / "log.txt").read_text(encoding="utf-8", errors="replace")
                           if (worker / "log.txt").is_file() else "worker log unavailable")
                why = "MissingPrediction: worker did not score study; " + log
            reasons[uid] = why
    need(assigned == set(ids), "S6 assignment cohort differs")
    need(set(reasons) == set(controller.get("fallback_reasons", {})), "S6 scored-set/receipt mismatch")
    validate_s6(controller, len(ids), reasons)
    return reasons


def anatomy_anchors(controller):
    # A dead worker has no anchor counters, but its absence must not stop fallback.
    workers = [w for w in controller["workers"] if w.get("status") == "PASS"]
    if workers:
        return s6.anatomy_fallbacks(dict(controller, workers=workers))
    return dict(scored=controller["scored"], model_anchor=0, fallback=controller["scored"], methods={})


def commit(output: Path, ids, r, ours, controller, provenance, *, r_failures=None, s6_failures=None):
    """Receipt and all validation precede the only final submission commit."""
    import numpy as np
    output.mkdir(parents=True, exist_ok=True)
    names = ("submission.csv", "p956-submission.tmp", "p956-receipt.json")
    need(not any((output / name).exists() for name in names), "stale P956 output")
    ids = v5q.validate_ids(ids)
    r_failures = {} if r_failures is None else r_failures
    s6_failures = {} if s6_failures is None else s6_failures
    validate_s6(controller, len(ids), s6_failures)
    r = v5q._probabilities("R", r, len(ids))
    rf, sf = failure_masks(ids, r, s6_failures)
    need(set(r_failures) == {uid for uid, failed in zip(ids, rf) if failed} and
         all(isinstance(why, str) and why for why in r_failures.values()), "R failures lack exception evidence")
    values = blend(ids, r, ours, s6_failures)
    need(np.array_equal(values, public_formula(r, ours, sf)), "public formula mismatch")
    payload = csv_bytes(ids, values)
    temporary = output / names[1]
    receipt = dict(status="PASS_P956_OUTPUT", version=VERSION, studies=len(ids),
                   findings=12, members=["R", "S6"], weights=[.5, .5],
                   rank="average / own non-failed N", failure_policy="public cell 7: surviving rank; both failed 0.5",
                   stop_thresholds=dict(R=.03, S6=.5), r_failures=len(r_failures),
                   s6_failures=len(s6_failures), both_failed=int((rf & sf).sum()),
                   replacements=[dict(uid=uid, replacement="0.5" if rf[i] and sf[i] else
                       "S6 rank" if rf[i] else "R rank", exceptions={
                           model: reasons[uid] for model, reasons in (("R", r_failures), ("S6", s6_failures))
                           if uid in reasons}) for i, uid in enumerate(ids) if rf[i] or sf[i]],
                   timing_policy="unlimited", submission_sha256=hashlib.sha256(payload).hexdigest(),
                   s6_anatomy_anchors=anatomy_anchors(controller), provenance=provenance)
    try:
        write_once(temporary, payload)
        saved_ids, saved = v5q._read_csv(temporary)
        v5q.validate_ids(ids, saved_ids)
        need(np.array_equal(saved.astype(np.float32), values), "CSV float32 round trip")
        write_once(output / names[2], json_bytes(receipt))
        os.replace(temporary, output / names[0])
    except BaseException:
        # Only our newly-created temporary, never a pre-existing user output.
        temporary.unlink(missing_ok=True)
        raise
    return receipt


def r_runtime_source(source: str) -> str:
    need(hashlib.sha256(source.encode()).hexdigest() == PINS["public-serving.py"], "R source drift")
    # Scheduling, exception transport and output policy only; model/window math intact.
    edits = [
        ('    nproc = max(4, min(96, (os.cpu_count() or 8) - 8))',
         '    nproc = min(12, max(1, (os.cpu_count() or 8) - 2))'),
        ('    if not np.isfinite(ranks).all():\n        ranks[~np.isfinite(ranks)] = 0.5',
         '    _E152.need(np.isfinite(ranks).all(), "Nonfinite R output; no fallback")'),
        ('def _pre_one(sid):\n    try:', 'def _pre_one(sid):\n    error = None\n    try:'),
        ('        print(f"  [pre] {sid[:16]} FAILED ({type(e).__name__}: {e})", flush=True)',
         '        error = f"{type(e).__name__}: {e}"\n'
         '        print(f"  [pre] {sid[:16]} FAILED ({error})", flush=True)'),
        ('    return sid, vol, mask', '    return sid, vol, mask, error'),
        ('            sid, vol, mask = item; i = pos[sid]',
         '            sid, vol, mask, decode_error = item; i = pos[sid]'),
        ('                assert mask.any(), "Empty decoded study"',
         '                if decode_error: raise RuntimeError(decode_error)\n'
         '                assert mask.any(), "Empty decoded study"'),
        ('                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)',
         '                    arm_probs[a][i] = _E152.prediction(infer_pair_probs(model, xw, xm, d))'),
        ('                _exp152_errors.append((i, type(e).__name__, str(e)))',
         '                _exp152_errors.append((i, type(e).__name__, str(e)))\n'
         '                for arm in arm_probs: arm[i] = 0.5'),
        ('for i, (sid, vol, mask) in enumerate(_exp152_bounded_preprocess(pool, test_ids, _pre_one, QUEUE_MAX)):',
         'for i, (sid, vol, mask, error) in enumerate(_exp152_bounded_preprocess(pool, test_ids, _pre_one, QUEUE_MAX)):'),
        ('            q.put((sid, vol, mask))', '            q.put((sid, vol, mask, error))'),
        ('    _E152.need(not _exp152_errors, "Inference failed; no submission: " + repr(_exp152_errors))',
         '    _E152.capture_failures(test_ids, _exp152_errors)'),
    ]
    for old, new in edits:
        need(source.count(old) == 1, "R adapter anchor changed")
        source = source.replace(old, new)
    compile(source, "<p956-R>", "exec")
    return source


class RBinding:
    """Replace only EXP152's output callbacks; load exactly the pinned H-B-113 R+ arm."""
    need = staticmethod(need)

    def __init__(self, helper, work):
        self.checked_model = helper.checked_model
        self.input_record = helper.input_record
        self.work = Path(work)
        self.failures = {}

    @staticmethod
    def prediction(values):
        import numpy as np
        value = np.asarray(values)
        need(value.shape == (12,) and np.isfinite(value).all() and
             ((value >= 0) & (value <= 1)).all(), "invalid R model prediction")
        return values

    def capture_failures(self, ids, errors):
        need(len(errors) == len({item[0] for item in errors}), "duplicate R failure")
        self.failures = {ids[i]: f"{kind}: {message}" for i, kind, message in errors}

    def save_raw(self, work, ids, arms, inputs, manifest, variant):
        import numpy as np
        need(variant == "R" and len(arms) == 1 and set(inputs) <= set(ids), "R incomplete")
        values = v5q._probabilities("R", arms[0], len(v5q.validate_ids(ids)))
        sentinels = {uid for uid, failed in zip(ids, np.all(values == .5, axis=1)) if failed}
        need(set(self.failures) <= sentinels and set(ids) - set(inputs) <= sentinels, "R incomplete")
        for uid in sentinels - set(self.failures):
            self.failures[uid] = "FailedStudySentinel: all 12 raw values equal 0.5 (public cell 7); no exception raised"
        need(not (self.work / "raw.npz").exists(), "stale R raw output")
        np.savez(self.work / "raw.npz", ids=np.asarray(ids), targets=np.asarray(LABELS), public=arms[0])

    def output_values(self, arms, variant, public_rank):
        need(len(arms) == 1 and variant == "R", "R-only binding violated")
        # Intermediate R CSV is raw, never used as a ranked blend input.
        return v5q._probabilities("R", arms[0], len(arms[0]))

    def finish_receipt(self, work, seconds):
        write_once(self.work / "receipt.json", json_bytes(dict(status="PASS_R", members=["R"],
            raw_sha256=sha(self.work / "raw.npz"), seconds=seconds,
            failure_count=len(self.failures), failures=self.failures)))


def bind_r(namespace, helper, weights: Path, competition: Path, output: Path):
    # The pinned helper accepts the manifest-bound R+ checkpoint with strict=True.
    need(helper.ARCH == "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "R architecture differs")
    namespace.update(_E152=RBinding(helper, output), _E152_WORK=output, _E152_VARIANT="R",
        _E152_FILES={R_NAME: weights},
        _E152_MANIFEST={"checkpoints": {R_NAME: {"sha256": R_SHA}}},
        ARMS=[dict(file=R_NAME, arch=helper.ARCH, res=384, w=1.0)],
        find_test_root=lambda: str(competition))


def runtime_pins(runtime: Path):
    for name, digest in PINS.items():
        check_pin(runtime / name, digest)
    for name, digest in s6.SOURCE_HASHES.items():
        check_pin(runtime / "exp189_s6_src" / name, digest)


def run_r(runtime: Path, weights: Path, competition: Path, output: Path):
    runtime_pins(runtime)
    check_pin(weights, R_SHA)
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("p956_r_helper", runtime / "exp152_own0949_release.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # A real module is necessary for the inherited fork Pool's pickle references.
    module = types.ModuleType("p956_r_serving")
    module.__file__ = str(runtime / "public-serving.py")
    sys.modules[module.__name__] = module
    exec(compile(r_runtime_source((runtime / "public-serving.py").read_text(encoding="utf-8")),
                 module.__file__, "exec"), module.__dict__)
    bind_r(module.__dict__, helper, weights, competition, output)
    module.main()


def run_release(runtime: Path, output: Path):
    """Kaggle only. R subprocess exits before S6 wheel install/model load."""
    import numpy as np
    import subprocess
    runtime_pins(runtime)
    need(not (output / "submission.csv").exists() and not (output / "p956-run").exists(), "stale release output")
    work = output / "p956-run"
    work.mkdir(parents=True, exist_ok=False)
    gpu_names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    need(len(gpu_names) == 2 and all("T4" in name for name in gpu_names), "requires 2xT4")
    comp = v5q._mount(COMPETITION, "test.csv")
    ids, _ = v5q._read_csv(comp / "sample_submission.csv")
    with (comp / "test.csv").open(encoding="utf-8", newline="") as stream:
        test_ids = v5q.validate_ids([row[UID] for row in csv.DictReader(stream)])
    need(set(ids) == set(test_ids), "test/sample cohort mismatch")
    weights = v5q._mount(DATASETS[0], R_NAME) / R_NAME
    check_pin(weights, R_SHA)
    bundle = v5q._mount(s6.DATASET_SLUG, "manifest.json")
    code_source = v5q._mount(v5q.CODE_SLUG, "anatomy.pt")
    s6.verify_bundle(bundle)
    s6.verify_source(runtime / "exp189_s6_src")
    code, anatomy = s6.prepare_code(code_source, work / "s6-code", install_wheels=False)
    env = dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    v5q.run_process_group([sys.executable, "-u", str(runtime / "exp169_release_p956rplus.py"), "r-worker",
        "--runtime", str(runtime), "--weights", str(weights), "--competition", str(comp),
        "--output", str(work / "r")], label="P956 R", env=env)
    rr = read_json(work / "r/receipt.json")
    need(rr.get("status") == "PASS_R" and rr.get("failure_count") == len(rr.get("failures", {})), "R did not finish")
    check_pin(work / "r/raw.npz", rr["raw_sha256"])
    with np.load(work / "r/raw.npz", allow_pickle=False) as raw:
        need(set(raw.files) == {"ids", "targets", "public"} and raw["targets"].tolist() == list(LABELS), "R schema")
        raw_ids = v5q.validate_ids(test_ids, raw["ids"].tolist())
        order = [raw_ids.index(uid) for uid in ids]
        r = raw["public"][order].copy()
    s6.install_wheels(sorted((work / "s6-code/wheels").glob("*.whl")), code)
    rows = s6.infer_studies([comp / "test_series" / uid for uid in ids], bundle_dir=bundle,
        code_dir=code, anatomy_weights=anatomy, output_dir=work / "s6", workers=2,
        amp="auto", device="cuda", max_fallback_fraction=.5, source_dir=runtime / "exp189_s6_src")
    v5q.validate_ids(ids, [row[UID] for row in rows])
    sids, ours = v5q._read_csv(work / "s6/predictions.csv")
    v5q.validate_ids(ids, sids)
    need(np.array_equal(ours, np.asarray([[row[t] for t in LABELS] for row in rows])), "S6 CSV differs")
    controller = read_json(work / "s6/controller/submission_receipt.json")
    failures = s6_failure_evidence(work / "s6/controller", ids, controller)
    return commit(output, ids, r, ours, controller, dict(runtime_sha256=sha(runtime / "exp169_release_p956rplus.py"),
        r_raw_sha256=sha(work / "r/raw.npz"), s6_raw_sha256=sha(work / "s6/predictions.csv"),
        s6_provenance_sha256=sha(work / "s6/provenance.json"), source_pins=PINS),
        r_failures=rr.get("failures", {}), s6_failures=failures)


def public_spec():
    check_pin(PUBLIC, PUBLIC_SHA)
    notebook = read_json(PUBLIC)
    cell = "".join(notebook["cells"][7]["source"])
    need(hashlib.sha256(cell.encode()).hexdigest() == CELL_SHA, "public blend cell drift")
    tree = ast.parse(cell)
    assignments = {ast.unparse(node.targets[0]): node.value for node in tree.body if isinstance(node, ast.Assign)}
    need(ast.literal_eval(assignments["(W_RAPTOR, W_OURS)"]) == (.5, .5), "public global weights changed")
    need(set(ast.literal_eval(assignments["W_RAPTOR_C"]).values()) == {.5}, "public OA weights changed")
    return notebook


def source_files():
    files = {name: ROOT / "tools" / name for name in PINS if name != "public-serving.py"}
    files["public-serving.py"] = VISIBLE / "exp155-assets/public-serving.py"
    files["exp169_release_p956rplus.py"] = Path(__file__).resolve()
    files["exp169_p956rplus_cpu.py"] = ROOT / "tools/exp169_p956rplus_cpu.py"
    files["exp169_release_p956.py"] = ROOT / "tools/exp169_release_p956.py"
    files.update({"exp189_s6_src/" + name: s6.SOURCE / name for name in s6.SOURCE_HASHES})
    for name, digest in PINS.items():
        check_pin(files[name], digest)
    s6.verify_source()
    return files


def package():
    files = source_files()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return payload.getvalue(), {name: sha(path) for name, path in sorted(files.items())}


def unpack(blob: bytes, digest: str, destination: Path, pins: dict):
    need(hashlib.sha256(blob).hexdigest() == digest, "runtime archive hash mismatch")
    need(not destination.exists(), "runtime destination must be fresh")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        need(len(archive.namelist()) == len(pins) and set(archive.namelist()) == set(pins), "runtime roster differs")
        for name in archive.namelist():
            need(not name.startswith(("/", "\\")) and "\\" not in name and ":" not in name and
                 ".." not in Path(name).parts, "unsafe runtime path")
            need(hashlib.sha256(archive.read(name)).hexdigest() == pins[name], "runtime member differs")
        destination.mkdir(parents=True)
        archive.extractall(destination)


R_SOURCE = ROOT / "artifacts/experiments/H_B_109_RPLUS_PUBLIC/v1.0.0/member/outputs/model_best.pt"


def prepare_dataset(out=OUT):
    import shutil
    check_pin(R_SOURCE, R_SHA)
    dest = out / "dataset" / R_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(R_SOURCE, dest)
    check_pin(dest, R_SHA)
    metadata = dict(title=DATASETS[0].split("/")[1], id=DATASETS[0], isPrivate=True,
        licenses=[{"name": "CC0-1.0"}])
    write_once(dest.parent / "dataset-metadata.json", json_bytes(metadata))
    return validate_dataset(out)


def validate_dataset(out=OUT):
    check_pin(out / "dataset" / R_NAME, R_SHA)
    meta = read_json(out / "dataset/dataset-metadata.json")
    need(meta["id"] == DATASETS[0] and meta["title"] == DATASETS[0].split("/")[1]
         and meta["isPrivate"] is True, "private R+ dataset metadata differs")
    return dict(status="PASS_DATASET_LOCAL", checkpoint_sha256=R_SHA)


def prepare(out=OUT):
    public = public_spec()
    prepare_dataset(out)
    blob, files = package()
    digest = hashlib.sha256(blob).hexdigest()
    # Bootstrap has no imported notebook code; authenticate before extraction/import.
    bootstrap = "\n".join([
        "import base64, hashlib, io, json, sys, zipfile", "from pathlib import Path",
        f"BLOB = {base64.b64encode(blob).decode()!r}", f"PINS = {files!r}",
        "blob = base64.b64decode(BLOB)", f"assert hashlib.sha256(blob).hexdigest() == {digest!r}",
        "runtime = Path('/kaggle/working/p956-runtime')", "assert not runtime.exists(), 'stale runtime'",
        "with zipfile.ZipFile(io.BytesIO(blob)) as archive:",
        "    assert len(archive.namelist()) == len(PINS) and set(archive.namelist()) == set(PINS)",
        "    for name in archive.namelist():",
        "        assert not name.startswith(('/', '\\\\')) and '\\\\' not in name and ':' not in name and '..' not in Path(name).parts",
        "        assert hashlib.sha256(archive.read(name)).hexdigest() == PINS[name]",
        "    runtime.mkdir()", "    archive.extractall(runtime)", "sys.path.insert(0, str(runtime))",
        "import exp169_release_p956rplus as release",
        "result = release.run_release(runtime, Path('/kaggle/working'))",
        "try: print(json.dumps(result, indent=2), flush=True)", "except Exception: pass", "",
    ])
    compile(bootstrap, "<p956-notebook>", "exec")
    credit = "".join(public["cells"][1]["source"])
    credit = credit.replace("These are our retrained checkpoints, rather than those baseline weights.",
        "This release uses the H-B-113 R+ continuation of nartaa's public R checkpoint and heliosli's unchanged S6 bundle.")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.13"}}, "cells": [
        {"cell_type": "markdown", "id": "p956-protocol", "metadata": {}, "source": [
            "# P956-R+ v1.0.0\nR+/S6 = 0.5/0.5 average percentile ranks for all 12 findings. "
            "Public cell 7 failed-study fallback; R >3% or S6 >50% stops; no retries or execution deadline. "
            "CPU verified; fresh Kaggle run remains required.\n"]},
        {"cell_type": "markdown", "id": "credits", "metadata": {}, "source": credit.splitlines(True)},
        {"cell_type": "code", "id": "p956-run", "metadata": {}, "execution_count": None,
         "outputs": [], "source": bootstrap.splitlines(True)}]}
    metadata = dict(id=KERNEL_ID, title=KERNEL_ID.split("/")[1], code_file="release.ipynb",
        language="python", kernel_type="notebook", is_private=True, enable_gpu=True,
        enable_tpu=False, enable_internet=False, machine_shape=MACHINE, docker_image=IMAGE,
        dataset_sources=DATASETS, competition_sources=[COMPETITION], kernel_sources=[], model_sources=[])
    write_once(out / "kernel/release.ipynb", json_bytes(notebook))
    write_once(out / "kernel/kernel-metadata.json", json_bytes(metadata))
    record = dict(status="PREPARED_LOCAL_ONLY", version=VERSION, kernel_id=KERNEL_ID,
        archive_sha256=digest, files=files, public_notebook_sha256=PUBLIC_SHA, public_blend_cell_sha256=CELL_SHA,
        notebook_sha256=sha(out / "kernel/release.ipynb"), metadata_sha256=sha(out / "kernel/kernel-metadata.json"))
    write_once(out / "build-receipt.json", json_bytes(record))
    return record


def validate(out=OUT):
    validate_dataset(out)
    manifest = read_json(out / "build-receipt.json")
    check_pin(out / "kernel/release.ipynb", manifest["notebook_sha256"])
    check_pin(out / "kernel/kernel-metadata.json", manifest["metadata_sha256"])
    notebook = read_json(out / "kernel/release.ipynb")
    meta = read_json(out / "kernel/kernel-metadata.json")
    need(notebook["metadata"]["kernelspec"]["name"] == "python3", "kernelspec python3 required")
    need(meta["id"] == KERNEL_ID and meta["title"] == KERNEL_ID.split("/")[1], "kernel id/title mismatch")
    need(meta["dataset_sources"] == DATASETS and meta["competition_sources"] == [COMPETITION]
         and meta["kernel_sources"] == [] and meta["enable_internet"] is False and meta["is_private"] is True
         and meta["enable_gpu"] is True and meta["enable_tpu"] is False and meta["model_sources"] == []
         and meta["machine_shape"] == MACHINE and meta["docker_image"] == IMAGE, "lean offline metadata differs")
    source = "".join(notebook["cells"][-1]["source"])
    compile(source, "<validated-p956>", "exec")
    literal = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(source).body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id in {"BLOB", "PINS"}}
    blob, current_pins = package()
    need(base64.b64decode(literal["BLOB"]) == blob and literal["PINS"] == current_pins == manifest["files"], "embedded source drift")
    need(hashlib.sha256(blob).hexdigest() == manifest["archive_sha256"], "archive drift")
    public_spec()
    return dict(status="PASS_CPU_VALIDATE", files=len(current_pins), **{
        k: manifest[k] for k in ("notebook_sha256", "metadata_sha256", "archive_sha256")})


def cpu_replay(output: Path):
    """Fresh two-study R/R+ CPU-float32 smoke; replay the unchanged S6 arm."""
    try:
        from tools.exp169_p956rplus_cpu import replay
    except ImportError:
        from exp169_p956rplus_cpu import replay
    return replay(output)


def register(out=OUT):
    validation = validate(out)
    record = dict(version=VERSION, kernel_id=KERNEL_ID,
        formula="0.5 r(R+) + 0.5 r(S6), average pct rank, all 12", **validation)
    record["status"] = "REGISTERED_LOCAL_ONLY"
    write_once(out / "registration.json", json_bytes(record))
    return record


def admit(out=OUT):
    validation = validate(out)
    record = read_json(out / "registration.json")
    need(record["notebook_sha256"] == validation["notebook_sha256"], "registration drift")
    replay = read_json(out / "cpu-replay/replay.json")
    need(replay["status"] == "PASS_CPU_REPLAY" and replay["public_formula_exact"] and replay["s6_exact"] and replay["r_changed_each_study"] and replay["strict_load"], "CPU replay required")
    check_pin(out / "cpu-replay/submission.csv", replay["submission_sha256"])
    return dict(status="PASS_ADMIT_DRY_RUN", kernel_id=KERNEL_ID,
        notebook_sha256=validation["notebook_sha256"], qualified_for_submission=False,
        next="B: fresh Kaggle 2xT4 visible run and review; no push/upload/submit performed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "register", "admit"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=OUT)
        if name == "admit":
            p.add_argument("--dry-run", action="store_true", required=True)
    p = sub.add_parser("cpu-replay")
    p.add_argument("--output", type=Path, default=OUT / "cpu-replay")
    p = sub.add_parser("r-worker")
    for name in ("runtime", "weights", "competition", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "r-worker":
        run_r(args.runtime, args.weights, args.competition, args.output)
        return
    if args.command == "run":
        result = run_release(args.runtime, args.output)
    elif args.command == "cpu-replay":
        result = cpu_replay(args.output)
    else:
        result = globals()[args.command](args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
