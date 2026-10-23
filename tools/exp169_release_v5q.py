"""EXP169 v5q: isolated A5-172/A5q-172 overlay on qualified A172 v2.0.2.

Only stages local kernels. Scored A172 files and uploaded datasets stay immutable;
the derived runtime removes timing policy only. No Kaggle command is issued here.

Changelog
- v1.2.0 (astra, 2026-10-11): no timing limits or projection gates, including the
  inherited A172 runtime; A172 completes before S6 wheel installation; versioned
  staging; original parent math, authentication and atomic output commit retained.
- v1.0.0 (sol): initial overlay, receipt exit-code fix.
- v1.1.0 (Claude opus for B, 2026-10-11), fixes for astra relv5q-thorough FAIL:
  * blend: the trio is the parent's EXP155 v1.2 output_values sequence (sum of
    per-member percentile_average ranks / member count) -> float32 pandas CSV ->
    float64 parse -> exp169_release_v2.final_values(..., Q, "A172"). A5 puts P954 in
    R's slot of [EXP155, R, EXP143]; A5q appends S6 to the parent's three members.
    With S6 constant both reproduce A172 exactly (also N=1300 ties).
  * lifecycle: A172's submission.csv is moved aside first; every check, hash and the
    receipt precede one atomic os.replace commit; cleanup is commit-aware and never
    deletes a committed output; post-success logging is best effort.
  * A172 runs in its own session/process group, killed as a group on timeout.
  * S6 preflight (all pins, no install) runs before A172; port v1.1.0 pins the code
    tree and wheels and bounds the wheel install.
  * timing projection includes the per-study anatomy-anchor time; anatomy fallbacks
    are counted and logged (never rejected). Hidden runs reject only on the clock.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

VERSION = "1.2.0"
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/releases/EXP169_A172_v2_0_2_stage"
OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_V5Q/v1.2.0"
EXP189 = ROOT / "artifacts/experiments/EXP189_S6_PORT/v1.2.0/kernels/parity"
WORK = Path("/kaggle/working")
UID = "StudyInstanceUID"
LABELS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
TARGETS = (UID, *LABELS)
A172_NOTEBOOK_SHA = "12c1c6b44489f7d58281b1599f23351c9be266bc56618dc2bd778b7b26061acc"
A172_ARCHIVE_SHA = "b755a132fd3186b703f9cbcf1b124b57406b11ec87ea34c358f6a297762e8a09"
A172_SOURCE_SHA = "f526eda7a7e5192c51ecdcaa35b1eff9b4c9f598d4e7bf2d2a5a31c58918d3d4"
A172_CELL_SHA = "0d0f7a7aaaa94f3182c9cb15500ebe02e641656465261b717278bd60e1559d5d"
A172_VERIFICATION_SHA = "0fe66742360cf2e13155c7038f4a572359cf1613a93a91d03dee7f8d200208ab"
S6_PORT_SHA = "6d08301e691e79c0076636ce109739247ec784b300ef7d3081c6e6033da0e6dd"
CODE_SLUG = "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"
BASE_LINEAR_SECONDS = 2.5 * 3600
BASE_CONSERVATIVE_SECONDS = 3.9 * 3600
COHORT_STUDIES = 1300
S6_ATOL = 2e-6
PARTNER_ATOL = 2e-5
ANCHOR_MAX_ATOL = .002
ANCHOR_MEAN_ATOL = .0005
SERVICE_ALLOWANCE_SECONDS = .003
CANDIDATES = {"A5-172": "a5_172", "A5q-172": "a5q_172"}
# Trio members in exp155 output_values order; the parent A172 trio is [EXP155, R, EXP143].
TRIO_MEMBERS = {"A5-172": ("EXP155", "P954", "EXP143"), "A5q-172": ("EXP155", "R", "EXP143", "S6")}
PASS_STATUS = "PASS_V5Q_VISIBLE_UNQUALIFIED"
TOLERANCE_SCHEMA = "exp169-v5q-visible-tolerance-v3"
TEMPORARY_NAMES = ("v5q-submission.tmp", "v5q-receipt.tmp")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_ids(*sequences) -> list[str]:
    require(bool(sequences), "no ID sequences")
    ids = [str(x) for x in sequences[0]]
    require(bool(ids) and len(ids) == len(set(ids)), "empty or duplicate IDs")
    for other in sequences[1:]:
        require(ids == [str(x) for x in other], "study ID/order mismatch")
    return ids


def timing_projection(s6_seconds_per_study: float, studies: int = COHORT_STUDIES,
                      s6_startup_seconds: float = 0.0) -> dict:
    rate = float(s6_seconds_per_study)
    startup = float(s6_startup_seconds)
    require(math.isfinite(rate) and rate > 0 and studies > 0 and
            math.isfinite(startup) and startup >= 0, "invalid S6 timing input")
    linear = BASE_LINEAR_SECONDS + startup + studies * rate
    conservative = BASE_CONSERVATIVE_SECONDS + startup + studies * rate
    return {"s6_seconds_per_study": rate, "s6_startup_seconds": startup, "studies": studies,
            "a172_linear_seconds": BASE_LINEAR_SECONDS,
            "a172_conservative_seconds": BASE_CONSERVATIVE_SECONDS,
            "projected_linear_seconds": linear,
            "projected_conservative_seconds": conservative, "informational": True}


def s6_service_rate(service: dict) -> float:
    """decode + render + anatomy anchor + GPU seconds per study, plus a fixed allowance,
    without any two-GPU speedup credit."""
    return float(service["service_seconds_per_study"]) + SERVICE_ALLOWANCE_SECONDS


def _probabilities(name: str, arr, n: int):
    import numpy as np
    a = np.asarray(arr, dtype=np.float64)
    require(a.shape == (n, 12) and np.isfinite(a).all() and ((0 <= a) & (a <= 1)).all(),
            "invalid probabilities: " + name)
    return a


def _percentile_average():
    """The registered EXP143/EXP152 rank used by both exp155 output_values and v2.rank."""
    try:
        from tools.exp152_own0949_release import percentile_average
    except ImportError:
        from exp152_own0949_release import percentile_average
    return percentile_average


def _a172_module():
    try:
        from tools import exp169_release_v2 as a172
    except ImportError:
        import exp169_release_v2 as a172
    return a172


def trio_output_values(arms) -> "np.ndarray":
    """EXP155 v1.2.0 output_values('blend3') generalised to 3 or 4 members: per-member,
    per-target percentile_average ranks, summed in member order, divided by the count."""
    import numpy as np
    percentile_average = _percentile_average()
    require(len(arms) in (3, 4), "three or four trio members required")
    shape = np.asarray(arms[0]).shape
    require(len(shape) == 2 and shape[0] > 0 and shape[1] == 12 and
            all(np.asarray(a).shape == shape and np.isfinite(a).all() and
                (np.asarray(a) >= 0).all() and (np.asarray(a) <= 1).all() for a in arms),
            "Invalid trio member probabilities")
    ranks = [np.asarray([percentile_average(np.asarray(a)[:, j].tolist())
                         for j in range(12)]).T for a in arms]
    return sum(ranks) / float(len(arms))


def trio_round_trip(ids, trio, path: str | Path | None = None) -> "np.ndarray":
    """The parent's serialization boundary: float32 DataFrame -> pandas to_csv ->
    the float64 parse of base.validate_csv. With a path the CSV is kept for audit."""
    import numpy as np
    import pandas as pd
    ids = validate_ids(ids)
    sub = pd.DataFrame(np.asarray(trio).astype(np.float32), columns=list(LABELS))
    sub.insert(0, UID, ids)
    sub = sub[list(TARGETS)]
    if path is None:
        buffer = io.StringIO()
        sub.to_csv(buffer, index=False)
        stream = io.StringIO(buffer.getvalue(), newline="")
    else:
        require(not Path(path).exists(), "trio CSV already exists: " + str(path))
        sub.to_csv(path, index=False)
        stream = Path(path).open("r", encoding="utf-8-sig", newline="")
    with stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(TARGETS), "trio CSV columns differ")
        rows = list(reader)
    validate_ids(ids, [row[UID] for row in rows])
    values = np.asarray([[float(row[t]) for t in LABELS] for row in rows], np.float64)
    require(values.shape == (len(ids), 12) and np.isfinite(values).all() and
            ((values >= 0) & (values <= 1)).all(), "trio CSV values invalid")
    return values


def trio_members(candidate: str, ids, r, s6, exp143, exp155, *, s6_port=None) -> list:
    import numpy as np
    require(candidate in CANDIDATES, "unknown candidate")
    n = len(validate_ids(ids))
    r, s6, exp143, exp155 = (_probabilities(name, arr, n) for name, arr in
                             (("R", r), ("S6", s6), ("EXP143", exp143), ("EXP155", exp155)))
    if candidate == "A5q-172":
        return [exp155, r, exp143, s6]
    if s6_port is None:
        from tools import exp189_s6_port as s6_port
    rows_r = [{UID: uid, **dict(zip(LABELS, row))} for uid, row in zip(ids, r)]
    rows_s = [{UID: uid, **dict(zip(LABELS, row))} for uid, row in zip(ids, s6)]
    p954_rows = s6_port.p954(rows_r, rows_s)
    validate_ids(ids, [row[UID] for row in p954_rows])
    p954 = np.asarray([[row[label] for label in LABELS] for row in p954_rows], dtype=np.float64)
    return [exp155, p954, exp143]


def trio_values(candidate: str, ids, r, s6, exp143, exp155, *, s6_port=None, trio_csv=None):
    return trio_round_trip(ids, trio_output_values(
        trio_members(candidate, ids, r, s6, exp143, exp155, s6_port=s6_port)), trio_csv)


def blend_arrays(candidate: str, ids, r, s6, exp143, exp155, q, *, a172=None, s6_port=None,
                 trio_csv=None):
    """Registered A5/A5q formula through the parent's exact trio/final sequence."""
    import numpy as np
    n = len(validate_ids(ids))
    q = _probabilities("Q", q, n)
    trio = trio_values(candidate, ids, r, s6, exp143, exp155, s6_port=s6_port, trio_csv=trio_csv)
    if a172 is None:
        a172 = _a172_module()
    return np.asarray(a172.final_values(trio, q, "A172"), dtype=np.float64)


def _read_csv(path: Path) -> tuple[list[str], "np.ndarray"]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == TARGETS, "CSV target schema/order mismatch: " + str(path))
        rows = list(reader)
    ids = validate_ids([row[UID] for row in rows])
    return ids, _probabilities(str(path), [[float(row[label]) for label in LABELS] for row in rows], len(ids))


def _write_csv(path: Path, ids, values) -> None:
    a = _probabilities("final", values, len(ids))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(TARGETS)
        for uid, row in zip(ids, a):
            writer.writerow([uid, *[format(float(x), ".10g") for x in row]])


def _mount(slug: str, required: str) -> Path:
    name = slug.split("/")[-1]
    root = Path("/kaggle/input")
    candidates = [root / name, root / "competitions" / name,
                  root / "datasets" / slug]
    candidates.extend(root.glob("datasets/*/" + name))
    candidates.extend(root.glob("datasets/*/" + name + "/*"))
    candidates.extend(root.glob(name + "/*"))
    matches = list(dict.fromkeys(p for p in candidates if (p / required).is_file()))
    require(len(matches) == 1, "dataset mount absent or ambiguous: " + slug)
    return matches[0]


def validate_s6_receipt(controller: dict, expected_rows: int) -> None:
    worker_codes = controller.get("worker_exit_codes")
    require(isinstance(worker_codes, dict) and set(worker_codes) == {"0", "1"} and
            all(type(code) is int and code == 0 for code in worker_codes.values()),
            "S6 worker exit-code receipt incomplete")
    require(controller.get("status") == "PASS" and
            controller.get("rows") == controller.get("scored") == expected_rows and
            controller.get("fallback_studies") == 0 and
            len(controller.get("gpus", [])) == 2 and
            all("T4" in name for name in controller["gpus"]),
            "S6 timing receipt incomplete")


# ---------------------------------------------------------------- output lifecycle

def _log(*parts, stream=None) -> None:
    """Best-effort logging: a closed or broken stdout/stderr never changes the outcome."""
    try:
        print(*parts, file=stream or sys.stdout, flush=True)
    except Exception:
        pass


def log_result(result: dict) -> None:
    _log(json.dumps(result, indent=2))


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name("v5q-receipt.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def committed(root: str | Path | None = None) -> bool:
    """True only after the final commit: PASS receipt whose hash names submission.csv."""
    root = WORK if root is None else Path(root)
    try:
        receipt = json.loads((root / "v5q-receipt.json").read_text(encoding="utf-8"))
        return (receipt.get("status") == PASS_STATUS and (root / "submission.csv").is_file() and
                sha(root / "submission.csv") == receipt.get("submission_sha256"))
    except Exception:
        return False


def cleanup_uncommitted(root: str | Path | None = None, *, candidate: str = "UNKNOWN",
                        error: BaseException | None = None) -> bool:
    """Before success only: remove submission.csv (A172's, partial or uncommitted) and this
    invocation's temporaries, then record the failure. A committed output is never touched."""
    root = WORK if root is None else Path(root)
    if committed(root):
        return False
    (root / "submission.csv").unlink(missing_ok=True)
    for name in TEMPORARY_NAMES:
        (root / name).unlink(missing_ok=True)
    if error is not None:
        failure = {"status": "FAILED_V5Q", "candidate": candidate, "generator_version": VERSION,
                   "error": type(error).__name__ + ": " + str(error)}
        try:
            _atomic_write_text(root / "v5q-receipt.json", json.dumps(failure, indent=2) + "\n")
        except Exception:
            pass
        _log("FAILED_V5Q: " + failure["error"], stream=sys.stderr)
    return True


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            try:
                os.killpg(process.pid, getattr(signal, "SIGKILL", 9))
            except ProcessLookupError:
                pass
    finally:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        process.wait()


def run_process_group(args: list[str], *, label: str, env=None) -> None:
    """Wait for completion without a time limit; an interruption kills the process group."""
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    process = subprocess.Popen(args, env=env, **options)
    try:
        code = process.wait()
    except BaseException:
        _kill_process_group(process)
        raise
    if code != 0:
        raise RuntimeError(f"{label} failed with exit code {code}")


def _replace_once(source: str, old: str, new: str) -> str:
    require(source.count(old) == 1, "A172 timing adapter anchor changed: " + old[:90])
    return source.replace(old, new)


def a172_runtime_source(source: str) -> str:
    """Exact authenticated cell edits: delete its alarm, add the runtime policy adapter.

    All extraction, archive/source authentication, model loading, inference, and
    failure cleanup statements in the parent cell are retained verbatim.
    """
    require(hashlib.sha256(source.encode()).hexdigest() == A172_CELL_SHA, "A172 cell drift")
    changes = (
        ("import os, signal as _exp169_signal, time as _exp169_time", "import os"),
        ("    _exp169_started = _exp169_time.monotonic()\n", ""),
        ("    def _exp169_deadline(_signum, _frame): raise TimeoutError('EXP169 seven-hour deadline guard')\n", ""),
        ("    _exp169_signal.signal(_exp169_signal.SIGALRM, _exp169_deadline)\n", ""),
        ("    _exp169_signal.alarm(24300)\n", ""),
        ("    finally:\n        _exp169_signal.alarm(0)\n", ""),
        ("        _e169.verify_assets(_partner, _e169.read_json(_partner/'release-manifest.json'))\n",
         "        _e169.verify_assets(_partner, _e169.read_json(_partner/'release-manifest.json'))\n"
         "        import exp169_release_v5q as _v5q\n"
         "        _runtime_policy = _v5q.install_a172_runtime(_e169)\n"
         "        _e169.write_json(Path('/kaggle/working/a172-runtime-functions.json'), _runtime_policy)\n"),
    )
    for old, new in changes:
        source = _replace_once(source, old, new)
    compile(source, "<a172-v5q-unlimited>", "exec")
    return source


def a172_function_source(name: str, source: str) -> str:
    """Remove only the explicitly enumerated timing policies from three parent functions.

    Exact anchors make upstream drift an error. Memory, cohort, asset, score,
    parity and output checks survive unchanged. A failed worker already aborts
    the loading barrier, so waiting without a timer preserves error propagation.
    """
    edits = {
        "partner_scores": (
            ("def partner_scores(work, competition, ids, manifest, deadline):",
             "def partner_scores(work, competition, ids, manifest, deadline=None):"),
            ("                need(time.monotonic() < deadline, 'Partner producer deadline exceeded')\n", ""),
            ("            loaded_barrier.wait(timeout=600)", "            loaded_barrier.wait()"),
            ("                need(started < deadline, 'Partner GPU deadline exceeded')\n", ""),
        ),
        "pilot": (
            ("partner_scores(work, competition, selected, manifest, started + 3600)",
             "partner_scores(work, competition, selected, manifest, None)"),
        ),
        "serve": (
            ("    remaining = LIMIT_SECONDS - trio_seconds - p.get('elapsed_seconds', math.inf) - RESERVE_SECONDS\n", ""),
            ("         p.get('projected_full_seconds', math.inf) <= remaining and\n", ""),
            ("'Measured partner pilot does not fit the seven-hour/memory gate'",
             "'Measured partner pilot coverage or memory gate failed'"),
            ("    deadline = start + remaining\n", ""),
            ("partner(work, competition, ids, manifest, deadline)",
             "partner(work, competition, ids, manifest, None)"),
            ("        need(clock() < deadline, 'Partner/global rank exceeded deadline')\n", ""),
            ("        need(trio_seconds + p['elapsed_seconds'] + clock() - start <= LIMIT_SECONDS - RESERVE_SECONDS,\n"
             "             'Full-cohort seven-hour deadline exceeded')\n", ""),
        ),
    }
    require(name in edits, "unknown A172 timing adapter function")
    for old, new in edits[name]:
        source = _replace_once(source, old, new)
    compile(source, "<a172-v5q-" + name + ">", "exec")
    return source


def install_a172_runtime(a172) -> dict:
    """In-memory derivative only; never rewrite any frozen parent code or receipt.

    Private base globals and a source cache preserve the parent's inspect-based
    serve adapter. Model functions and mathematical operations are the originals.
    """
    import inspect
    import linecache
    import types
    require(sha(a172.__file__) == A172_SOURCE_SHA and
            sha(a172.base.__file__) == a172.BASE_SHA, "A172 runtime source drift")
    base = types.ModuleType("exp169_partner_bundle_release_v5q_unlimited")
    base.__dict__.update(a172.base.__dict__)
    sources = {}
    for name in ("partner_scores", "pilot", "serve"):
        source = a172_function_source(name, inspect.getsource(getattr(a172.base, name)))
        filename = "<exp169-v5q-1.2.0-" + name + ">"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        exec(compile(source, filename, "exec"), base.__dict__)
        sources[name] = hashlib.sha256(source.encode()).hexdigest()
    partner = a172.clone(a172.partner_scores, base=base)
    a172.pilot = a172.clone(a172.pilot, base=base, partner_scores=partner)
    original_serve = a172.serve
    a172.serve = a172.clone(original_serve, base=base, partner_scores=partner)
    a172.serve.__kwdefaults__ = original_serve.__kwdefaults__.copy()
    a172.partner_scores = partner
    return {"version": VERSION, "timing_policy": "unlimited", "functions": sources}


def run_a172(a172_source: str, *, started_epoch: float, root: str | Path | None = None) -> None:
    """Run the authenticated A172 derivative with timing policy removed."""
    root = WORK if root is None else Path(root)
    script = root / "a172-v2.0.2-v5q-1.2.0.py"
    source = a172_runtime_source(a172_source)
    script.write_text(source, encoding="utf-8")
    (root / "a172-runtime-adapter.json").write_text(json.dumps({
        "version": VERSION, "parent_cell_sha256": A172_CELL_SHA,
        "runtime_cell_sha256": sha(script), "timing_policy": "unlimited",
        "prediction_policy": "unchanged A172 v2.0.2"}, indent=2) + "\n", encoding="utf-8")
    run_process_group([sys.executable, "-u", str(script)], label="A172 v2.0.2 / v5q 1.2.0")


def preflight_s6(*, started_epoch: float, root: str | Path | None = None) -> dict:
    """Before A172: every S6 pin (port, sources, bundle, code tree/closure, anatomy, wheels).
    Hash checks only; nothing is installed or executed from the code dataset."""
    import exp189_s6_port as s6
    root = WORK if root is None else Path(root)
    require(sha(Path(s6.__file__)) == S6_PORT_SHA, "S6 port hash mismatch")
    s6.verify_source()
    bundle = _mount(s6.DATASET_SLUG, "manifest.json")
    s6.verify_bundle(bundle)
    code_src = _mount(CODE_SLUG, "anatomy.pt")
    target = root / "s6-preflight"
    try:
        s6.prepare_code(code_src, target, install_wheels=False)
    finally:
        shutil.rmtree(target, ignore_errors=True)
    record = {"status": "PASS_S6_PREFLIGHT", "generator_version": VERSION, "port_version": s6.VERSION,
              "code_tree_sha256": s6.CODE_TREE_SHA256, "wheel_sha256": s6.WHEEL_HASHES,
              "seconds_after_start": time.time() - float(started_epoch)}
    (root / "v5q-preflight.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _log("V5Q_S6_PREFLIGHT", json.dumps(record))
    return record


def run_after_a172(candidate: str, *, started_epoch: float) -> dict:
    """Called inside a candidate Kaggle kernel after the A172 derivative subprocess.

    No submission.csv exists between the first statement and the final atomic commit;
    the commit is the last operation, after every check, hash and the PASS receipt.
    """
    import gc
    import time
    import torch
    import numpy as np
    root = WORK
    sys.path.insert(0, str(root / "exp155-assets"))
    sys.path.insert(0, str(root / "exp169-partner" / "code"))
    sys.path.insert(0, str(root / "exp189_s6_port" / "tools"))
    try:
        require(candidate in CANDIDATES, "unknown candidate")
        require(not (root / "a172-submission.csv").exists() and not (root / "v5q-receipt.json").exists(),
                "stale v5q outputs")
        require((root / "submission.csv").is_file(), "A172 submission absent")
        os.replace(root / "submission.csv", root / "a172-submission.csv")
        import exp189_s6_port as s6
        import exp169_release_v2 as a172
        comp = _mount("rsna-knee-abnormality-detection", "sample_submission.csv")
        data = _mount("rsna-knee-abnormality-detection", "test.csv")
        require(sha(Path(a172.__file__)) == A172_SOURCE_SHA, "A172 runtime hash mismatch")
        require(sha(Path(s6.__file__)) == S6_PORT_SHA, "S6 port hash mismatch")
        manifest = json.loads((root / "exp169-partner/release-manifest.json").read_text())
        a172.verify_assets(root / "exp169-partner", manifest)
        base_receipt = json.loads((root / "exp169-executed-candidate.json").read_text())
        require(base_receipt.get("status") == "PASS_BUNDLE" and base_receipt.get("candidate") == "A172",
                "A172 did not complete")
        with np.load(root / "exp169-trio/raw.npz", allow_pickle=False) as anchor:
            require(set(anchor.files) == {"ids", "targets", "own", "public", "exp143"}, "A172 trio member schema")
            ids = validate_ids(anchor["ids"].tolist())
            require(anchor["targets"].tolist() == list(LABELS), "A172 target order mismatch")
            r, exp143, exp155 = (anchor[key].copy() for key in ("public", "exp143", "own"))
        with np.load(root / "partner-raw.npz", allow_pickle=False) as partner:
            require(set(partner.files) == {"ids", "l2", "dino", "convnext"}, "A172 partner member schema")
            validate_ids(ids, partner["ids"].tolist())
            q, _ = a172.aggregate({key: partner[key] for key in ("l2", "dino", "convnext")})
        sample_ids, _ = _read_csv(comp / "sample_submission.csv")
        validate_ids(sample_ids, ids)
        base_ids, _ = _read_csv(root / "a172-submission.csv")
        validate_ids(ids, base_ids)
        bundle = _mount(s6.DATASET_SLUG, "manifest.json")
        code_src = _mount(CODE_SLUG, "anatomy.pt")
        s6.verify_bundle(bundle)
        s6.verify_source()
        s6_phase_tic = time.monotonic()
        code, anatomy = s6.prepare_code(code_src, root / "s6-code", install_wheels=True)
        torch.cuda.empty_cache()
        gc.collect()
        rows = s6.infer_studies([data / "test_series" / uid for uid in ids], bundle_dir=bundle,
                                code_dir=code, anatomy_weights=anatomy, output_dir=root / "s6-port",
                                workers=2, amp="auto",
                                max_fallback_fraction=.03)
        s6_elapsed = time.monotonic() - s6_phase_tic
        s6_ids = validate_ids(ids, [row[UID] for row in rows])
        s6_values = _probabilities("S6", [[row[label] for label in LABELS] for row in rows], len(ids))
        shutil.copyfile(root / "s6-port/predictions.csv", root / "s6-port-predictions.csv")
        csv_ids, _ = _read_csv(root / "s6-port-predictions.csv")
        validate_ids(ids, s6_ids, csv_ids)
        controller = json.loads((root / "s6-port/controller/submission_receipt.json").read_text())
        validate_s6_receipt(controller, len(ids))
        service = s6.service_components(controller)
        anchors = s6.anatomy_fallbacks(controller)
        _log("V5Q_S6_ANATOMY_ANCHORS", json.dumps(anchors))
        # Decode, render, anatomy anchor and GPU seconds are summed without a 2-GPU
        # speedup credit. Startup includes code extraction, hash checks and launch.
        # Both observed times and projections are informational.
        service_rate = s6_service_rate(service)
        startup = max(0.0, s6_elapsed - len(ids) * service_rate)
        timing = timing_projection(service_rate, s6_startup_seconds=startup)
        timing["service_per_study"] = service["per_study"]
        timing["observed_a172_and_setup_seconds"] = time.time() - started_epoch - s6_elapsed
        timing["projected_from_observed_seconds"] = (timing["observed_a172_and_setup_seconds"] +
                                                    startup + COHORT_STUDIES * service_rate)
        final = blend_arrays(candidate, ids, r, s6_values, exp143, exp155, q,
                             a172=a172, s6_port=s6, trio_csv=root / "v5q-trio.csv")
        temporary = root / "v5q-submission.tmp"
        _write_csv(temporary, ids, final)
        written_ids, written = _read_csv(temporary)
        validate_ids(sample_ids, written_ids)
        require(np.array_equal(written, np.asarray([[float(format(float(v), ".10g")) for v in row]
                                                    for row in final])), "final CSV round trip differs")
        result = {"status": PASS_STATUS, "candidate": candidate, "generator_version": VERSION,
                  "port_version": s6.VERSION, "studies": len(ids),
                  "trio_members": list(TRIO_MEMBERS[candidate]),
                  "a172_receipt_sha256": sha(root / "exp169-executed-candidate.json"),
                  "a172_submission_sha256": sha(root / "a172-submission.csv"),
                  "s6_raw_sha256": sha(root / "s6-port-predictions.csv"),
                  "trio_csv_sha256": sha(root / "v5q-trio.csv"),
                  "submission_sha256": sha(temporary),
                  "s6_provenance_sha256": sha(root / "s6-port/provenance.json"),
                  "s6_anatomy_anchors": anchors,
                  "s6_elapsed_seconds": s6_elapsed, "timing_projection": timing,
                  "started_epoch": started_epoch, "timing_policy": "unlimited",
                  "source_sha256": S6_PORT_SHA, "a172_source_sha256": A172_SOURCE_SHA}
        _atomic_write_text(root / "v5q-receipt.json", json.dumps(result, indent=2) + "\n")
        os.replace(temporary, root / "submission.csv")  # commit: nothing after this can delete it
    except BaseException as exc:
        cleanup_uncommitted(root, candidate=candidate, error=exc)
        raise
    return result


# ---------------------------------------------------------------- local staging

def _cell(source: str, kind: str = "code") -> dict:
    return {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if kind == "code" else {})}


def _guarded(body: str, candidate: str) -> str:
    return ("try:\n" + textwrap.indent(body, "    ") +
            "except BaseException as _v5q_exc:\n"
            f"    v5q.cleanup_uncommitted(candidate={candidate!r}, error=_v5q_exc)\n"
            "    raise\n")


def stage() -> dict:
    """Generate two local, unpushed kernels and one unchanged public S6 sibling."""
    require(sha(STAGE / "kernel/release.ipynb") == A172_NOTEBOOK_SHA, "qualified A172 notebook drift")
    require(sha(STAGE / "dataset/exp169-assets.zip.bin") == A172_ARCHIVE_SHA, "A172 upload archive drift")
    require(sha(ROOT / "tools/exp169_release_v2.py") == A172_SOURCE_SHA, "A172 source drift")
    require(sha(ROOT / "artifacts/experiments/EXP169_RELEASE_V2/VERIFICATION-v2.0.2.md") ==
            A172_VERIFICATION_SHA, "A172 verification note drift")
    require(sha(ROOT / "tools/exp189_s6_port.py") == S6_PORT_SHA, "S6 source drift")
    receipt = json.loads((STAGE / "build-receipt.json").read_text())
    require(receipt["version"] == "2.0.2" and receipt["candidate"] == "A172" and
            receipt["notebook_sha256"] == A172_NOTEBOOK_SHA and
            receipt["archive_sha256"] == A172_ARCHIVE_SHA and
            receipt["verification_note"]["sha256"] == A172_VERIFICATION_SHA,
            "A172 stage binding drift")
    base_nb = json.loads((STAGE / "kernel/release.ipynb").read_text(encoding="utf-8"))
    require(len(base_nb["cells"]) == 5, "A172 notebook layout changed")
    a172_source = "".join(base_nb["cells"][4]["source"])
    parity_nb = json.loads((EXP189 / "exp189_parity.ipynb").read_text(encoding="utf-8"))
    port_cell = parity_nb["cells"][0]
    require("import exp189_s6_port as s6" in "".join(port_cell["source"]), "S6 port archive cell changed")
    port_source = "".join(port_cell["source"])
    import ast
    assignments = {node.targets[0].id: ast.literal_eval(node.value)
                   for node in ast.parse(port_source).body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1 and
                   isinstance(node.targets[0], ast.Name) and
                   node.targets[0].id in {"PORT_ARCHIVE_B64", "PORT_ARCHIVE_SHA256"}}
    blob = base64.b64decode(assignments["PORT_ARCHIVE_B64"])
    require(hashlib.sha256(blob).hexdigest() == assignments["PORT_ARCHIVE_SHA256"],
            "embedded S6 archive hash differs")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        require(hashlib.sha256(archive.read("tools/exp189_s6_port.py")).hexdigest() == S6_PORT_SHA,
                "embedded S6 port source differs")
        port_text = archive.read("tools/exp189_s6_port.py").decode("utf-8")
    code_tree_pinned = 'CODE_TREE_SHA256: str | None = None\n' not in port_text
    base_meta = json.loads((STAGE / "kernel/kernel-metadata.json").read_text())
    must_sources = base_meta["dataset_sources"]
    require(must_sources == ["taeyangg4/knee-exp169-a172-v2-assets",
                             "taeyangg4/knee-exp155-own0949-oai-v1-2-assets",
                             "nartaa/rsna-knee-publication-swa-weights-20261007"], "A172 sources changed")
    OUT.mkdir(parents=True, exist_ok=True)
    source_bytes = (ROOT / "tools/exp169_release_v5q.py").read_bytes()
    require(b"\r" not in source_bytes, "v5q source must use LF line endings")
    source = source_bytes.decode("utf-8")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    a172_hash = hashlib.sha256(a172_source.encode()).hexdigest()
    files = {}
    for candidate, folder in CANDIDATES.items():
        kernel_dir = OUT / "kernels" / folder
        kernel_dir.mkdir(parents=True, exist_ok=True)
        bootstrap = (f"# EXP169 v5q {VERSION} {candidate}: observed timing, 2xT4 check, pinned v5q source.\n"
                     "import hashlib, subprocess, sys, time\nfrom pathlib import Path\n"
                     "V5Q_STARTED_EPOCH = time.time()\n"
                     f"print('EXP169 v5q {VERSION} {candidate}', flush=True)\n"
                     "GPU_NAMES = subprocess.check_output(['nvidia-smi', '--query-gpu=name',\n"
                     "    '--format=csv,noheader'], text=True).splitlines()\n"
                     "assert len(GPU_NAMES) == 2 and all('T4' in name for name in GPU_NAMES), GPU_NAMES\n"
                     f"V5Q_SOURCE = {source!r}\n"
                     f"assert hashlib.sha256(V5Q_SOURCE.encode()).hexdigest() == '{source_hash}'\n"
                     "Path('/kaggle/working/exp169_release_v5q.py').write_text(V5Q_SOURCE, encoding='utf-8')\n"
                     "sys.path.insert(0, '/kaggle/working')\n"
                     "import exp169_release_v5q as v5q\n"
                     f"assert v5q.VERSION == '{VERSION}'\n"
                     "del V5Q_SOURCE\n")
        guarded_port = _cell(_guarded(port_source, candidate))
        a172_run = ("# S6 preflight (pins only, no install), then A172 v2.0.2 with timing policy removed.\n"
                    "# A172 finishes in its own process before S6 can install wheels.\n"
                    "import hashlib\n"
                    f"A172_SOURCE = {a172_source!r}\n"
                    f"assert hashlib.sha256(A172_SOURCE.encode()).hexdigest() == '{a172_hash}'\n"
                    + _guarded("v5q.preflight_s6(started_epoch=V5Q_STARTED_EPOCH)\n"
                               "v5q.run_a172(A172_SOURCE, started_epoch=V5Q_STARTED_EPOCH)\n", candidate)
                    + "del A172_SOURCE\n")
        v5_run = ("# Pinned v5q overlay; the only write of submission.csv is its final atomic commit.\n"
                  + _guarded(f"V5Q_RESULT = v5q.run_after_a172({candidate!r}, started_epoch=V5Q_STARTED_EPOCH)\n",
                             candidate)
                  + "v5q.log_result(V5Q_RESULT)  # best effort; never deletes or raises\n")
        notebook = {"cells": [base_nb["cells"][0], _cell(bootstrap), guarded_port, _cell(a172_run), _cell(v5_run)],
                    "metadata": base_nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}
        for cell in notebook["cells"][1:]:
            compile("".join(cell["source"]), f"<v5q-{folder}>", "exec")
        nb_path = kernel_dir / "release.ipynb"
        nb_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        meta = dict(base_meta)
        meta.update(id=f"taeyangg4/knee-exp169-{folder.replace('_', '-')}-v5q",
                    title=f"knee-exp169-{folder.replace('_', '-')}-v5q")
        meta["dataset_sources"] = [*must_sources, "heliosli/rsna-knee-compact-v1", CODE_SLUG]
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        files[folder] = {name: sha(kernel_dir / name) for name in ("release.ipynb", "kernel-metadata.json")}
    sibling = OUT / "kernels/s6_public_parity"
    sibling.mkdir(parents=True, exist_ok=True)
    parity_copy = json.loads((EXP189 / "exp189_parity.ipynb").read_text(encoding="utf-8"))
    (sibling / "release.ipynb").write_text(json.dumps(parity_copy, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    parity_meta = json.loads((EXP189 / "kernel-metadata.json").read_text())
    parity_meta.update(id="taeyangg4/knee-exp169-v5q-s6-parity", title="knee-exp169-v5q-s6-parity",
                       code_file="release.ipynb", machine_shape=base_meta["machine_shape"],
                       docker_image=base_meta["docker_image"])
    (sibling / "kernel-metadata.json").write_text(json.dumps(parity_meta, indent=2) + "\n", encoding="utf-8")
    files["s6_public_parity"] = {name: sha(sibling / name) for name in ("release.ipynb", "kernel-metadata.json")}
    tolerance = {"schema": TOLERANCE_SCHEMA, "study_ids": "exact_order",
                 "targets": list(TARGETS), "s6_probability_atol": S6_ATOL,
                 "s6_probability_rtol": 0, "s6_bad_cells": 0,
                 "a172_anchor_max_abs": ANCHOR_MAX_ATOL,
                 "a172_anchor_mean_abs": ANCHOR_MEAN_ATOL,
                 "a172_partner_max_abs": PARTNER_ATOL,
                 "a172_inputs": "exact", "a172_binding": "exact",
                 "timing_policy": "informational; no self-imposed limits",
                 "a172_conservative_seconds": BASE_CONSERVATIVE_SECONDS,
                 "cohort_studies": COHORT_STUDIES,
                 "s6_projection_terms": ["decode_s", "render_s", "anchor_s", "gpu_s"],
                 "s6_service_allowance_seconds": SERVICE_ALLOWANCE_SECONDS,
                 "trio_members": {k: list(v) for k, v in TRIO_MEMBERS.items()},
                 "trio_boundary": "exp155 v1.2 output_values -> float32 pandas CSV -> float64 parse -> "
                                  "exp169_release_v2.final_values(A172)",
                 "final_csv": "exact after .10g serialization",
                 "stage_notebook_sha256": A172_NOTEBOOK_SHA,
                 "stage_archive_sha256": A172_ARCHIVE_SHA,
                 "port_source_sha256": S6_PORT_SHA, "generator_version": VERSION}
    (OUT / "visible-tolerance.json").write_text(json.dumps(tolerance, indent=2) + "\n", encoding="utf-8")
    files["visible-tolerance.json"] = sha(OUT / "visible-tolerance.json")
    build = {"schema": "exp169-v5q-build-v1", "generator_version": VERSION,
             "generator_sha256": source_hash, "a172_cell_sha256": a172_hash,
             "a172_runtime_cell_sha256": hashlib.sha256(a172_runtime_source(a172_source).encode()).hexdigest(),
             "timing_policy": "unlimited",
             "port_source_sha256": S6_PORT_SHA, "port_archive_sha256": assignments["PORT_ARCHIVE_SHA256"],
             "s6_code_tree_pinned": code_tree_pinned,
             "status": ("STAGED_NOT_PUSHED" if code_tree_pinned else
                        "STAGED_BLOCKED_S6_CODE_TREE_UNPINNED (candidate preflight fails closed)"),
             "files": files}
    (OUT / "build-receipt.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
    return {**files, "build-receipt.json": sha(OUT / "build-receipt.json"),
            "s6_code_tree_pinned": code_tree_pinned}


def rebind_s6() -> dict:
    """After a reviewed S6 port change (e.g. `pin-code --write` plus exp189_make_kernels):
    replace the S6_PORT_SHA line of this source. Run `stage` afterwards in a new process."""
    path = ROOT / "tools/exp169_release_v5q.py"
    text = path.read_text(encoding="utf-8")
    old_line = f'S6_PORT_SHA = "{S6_PORT_SHA}"\n'
    new_sha = sha(ROOT / "tools/exp189_s6_port.py")
    require(text.count(old_line) == 1, "S6_PORT_SHA line not found")
    path.write_text(text.replace(old_line, f'S6_PORT_SHA = "{new_sha}"\n'), encoding="utf-8", newline="\n")
    return {"old_port_sha256": S6_PORT_SHA, "new_port_sha256": new_sha, "v5q_sha256": sha(path),
            "next": "python tools/exp169_release_v5q.py stage"}


def compare_visible(candidate: str, output: Path, sibling: Path, result: Path) -> dict:
    """Local CLI compatibility: one comparator implementation for all entry points."""
    try:
        from tools.exp169_release_v5q_compare import compare_visible as compare
    except ImportError:
        from exp169_release_v5q_compare import compare_visible as compare
    return compare(candidate, output, sibling, result)


def cpu_replay(s6_predictions: Path, output: Path) -> dict:
    """Replay both formulas on qualified visible A172 raw arrays and real S6 outputs.

    No model, GPU, label table, install, network call or public score is involved.
    This checks the blend/serialization only; it is not a fresh A172 inference.
    """
    import numpy as np
    output, s6_predictions = Path(output), Path(s6_predictions)
    require(not output.exists(), "fresh CPU replay output required")
    parent = STAGE / "kernel-output"
    receipt = json.loads((parent / "exp169-executed-candidate.json").read_text())
    trio_receipt = json.loads((parent / 'exp169-trio/receipt.json').read_text())
    require(receipt.get("status") == "PASS_BUNDLE" and receipt.get("candidate") == "A172" and
            sha(parent / "submission.csv") == receipt["submission_sha256"] and
            sha(parent / "partner-raw.npz") == receipt["partner_raw_sha256"],
            "qualified A172 visible outputs differ")
    require(trio_receipt.get('status') == 'COMPLETE_INFERENCE' and
            sha(parent / 'exp169-trio/raw.npz') == trio_receipt['files']['raw.npz'] and
            sha(parent / 'exp169-trio/submission.csv') ==
            trio_receipt['files']['submission.csv'] == receipt['trio_sha256'],
            'qualified A172 trio outputs differ')
    parent_ids, parent_csv = _read_csv(parent / "submission.csv")
    s6_ids, s6_values = _read_csv(s6_predictions)
    with np.load(parent / "exp169-trio/raw.npz", allow_pickle=False) as anchor, \
         np.load(parent / "partner-raw.npz", allow_pickle=False) as partner:
        ids = validate_ids(parent_ids, s6_ids, anchor["ids"].tolist(), partner["ids"].tolist())
        require(anchor["targets"].tolist() == list(LABELS), "A172 target order mismatch")
        members = [anchor[key].copy() for key in ("public", "exp143", "own")]
        q, _ = _a172_module().aggregate({key: partner[key] for key in ("l2", "dino", "convnext")})
    _probabilities("S6", s6_values, len(ids))
    try:
        from tools import exp189_s6_port as s6
    except ImportError:
        import exp189_s6_port as s6
    output.mkdir(parents=True)
    candidates = {}
    for candidate, folder in CANDIDATES.items():
        neutral = blend_arrays(candidate, ids, members[0], np.full_like(s6_values, .5),
                               members[1], members[2], q, s6_port=s6)
        rounded_neutral = np.asarray([[float(format(float(v), '.10g')) for v in row] for row in neutral])
        require(np.array_equal(rounded_neutral, parent_csv), "neutral S6 differs from qualified A172")
        values = blend_arrays(candidate, ids, members[0], s6_values, members[1], members[2], q,
                              s6_port=s6, trio_csv=output / (folder + '-trio.csv'))
        path = output / (folder + '.csv')
        _write_csv(path, ids, values)
        saved_ids, saved = _read_csv(path)
        validate_ids(ids, saved_ids)
        expected = np.asarray([[float(format(float(v), '.10g')) for v in row] for row in values])
        require(np.array_equal(saved, expected), "CPU replay final CSV differs")
        candidates[candidate] = {"neutral_parent_exact": True, "rows": len(ids),
                                 "submission_sha256": sha(path),
                                 "trio_sha256": sha(output / (folder + '-trio.csv'))}
    report = {"status": "PASS_CPU_BLEND_REPLAY", "generator_version": VERSION,
              "scope": "qualified visible A172 arrays + supplied real S6 predictions; no fresh A172 inference",
              "s6_predictions_sha256": sha(s6_predictions),
              "parent_receipt_sha256": sha(parent / 'exp169-executed-candidate.json'),
              "parent_trio_raw_sha256": sha(parent / 'exp169-trio/raw.npz'),
              "parent_partner_raw_sha256": sha(parent / 'partner-raw.npz'),
              "candidates": candidates}
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage")
    sub.add_parser("rebind-s6")
    replay = sub.add_parser("cpu-replay")
    replay.add_argument("--s6-predictions", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    comp = sub.add_parser("compare-visible")
    comp.add_argument("--candidate", choices=CANDIDATES, required=True)
    comp.add_argument("--output", type=Path, required=True)
    comp.add_argument("--sibling-output", type=Path, required=True)
    comp.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        print(json.dumps(stage(), indent=2))
    elif args.command == "rebind-s6":
        print(json.dumps(rebind_s6(), indent=2))
    elif args.command == "cpu-replay":
        print(json.dumps(cpu_replay(args.s6_predictions, args.output), indent=2))
    else:
        print(json.dumps(compare_visible(args.candidate, args.output, args.sibling_output, args.result), indent=2))


if __name__ == "__main__":
    main()
