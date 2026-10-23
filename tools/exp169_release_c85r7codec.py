"""C85-R7+codecs v1.1.0: C85-R7 v1.0.0 with offline python-gdcm in every child (H-B-126 step 2).

2026-10-13 (Claude opus builder stand-in for B): new wrapper only. exp169_release_c85r7.py v1.0.0
(SHA256 38cd9e84...), C85 v1.0.4, P956, A172/v5q, EXP189 and the R7 port are imported unchanged
and hash-pinned; the formula, members, ranks, survivor substitution, P-branch diagnostic,
publication and failure handling are v1.0.0's.

The only differences (D-B-162: the pinned image cannot decode JPEG Lossless .4.57/.4.70 or JPEG-LS
.4.80/.4.81; R blanks such series, S6/R7 drop them, our own reader raises):
- run_release: after v1.0.0's preflight, hb126_codecs.activate() installs the pinned python-gdcm
  3.2.6 cp313 wheel (private dataset taeyangg4/knee-hb126-codecs-cp313-v1) into a fresh private
  /kaggle/tmp|/tmp directory (pip --no-index --no-deps --target; unzip fallback) and verifies it in
  a fresh interpreter. ACTIVE -> the hook directory is appended LAST to PYTHONPATH of the A172
  (own + Q) child and the public worker; the public worker passes it to its R r-worker and, through
  os.environ, to the S_R7 stage and its grandchildren. Any other status -> exactly v1.0.0's
  environment (telemetry only; never an abort).
- The hook makes gdcm importable at interpreter start, keeps every plugin that existed before first
  (gdcm is only a last resort there, so Pillow J2K, native RLE and uncompressed decoding are
  untouched) and counts compressed-frame decodes per branch.
- S_R7: after the stage pip-installs its pydicom 3.0.1, the hook is verified again under that
  pydicom; a failure strips the hook for S_R7 only (v1.0.0 behaviour there).
- The A172 child script gets a 12-line header that only labels the telemetry branch per phase
  (A172:own, A172:Q_A5q); without ACTIVE codecs the script is byte-identical to v1.0.0's.
- Receipt: version 1.1.0, release "C85R7+codecs" and a `codecs` field (install/gate status, per
  branch decoded_compressed_frames / plugin_failures / decode_failures, per-process hook records).
CPU commands never infer, install into the image, submit, query scores, or touch a Kaggle account.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

try:
    from tools import exp169_release_c85r7 as base, hb126_codecs as codecs
except ImportError:
    import exp169_release_c85r7 as base
    import hb126_codecs as codecs

c85, p, r7, v5q, s6 = base.c85, base.p, base.r7, base.v5q, base.s6
need, sha, read_json, write_once, json_bytes = base.need, base.sha, base.read_json, base.write_once, base.json_bytes
telemetry, optional_json, competition_root = base.telemetry, base.optional_json, base.competition_root
ROOT = base.ROOT
VERSION = "1.1.0"
OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0"
KERNEL_ID = "taeyangg4/knee-exp169-c85r7codec-v1"
RELEASE = "C85R7+codecs"
FORMULA = base.FORMULA
C85R7_SHA = "38cd9e84349b6fdb93cada8b657844ba78c63749ba5e23f5027be9d2d5ce7091"
CODECS_SHA = "bd463c800196081f2396c8706fa735cdcd5e173eebad8be269169e1bda1dc919"
C85R7_STAGE = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.0.0"
DATASETS = [*base.DATASETS, codecs.DATASET_SLUG]
MEMBERS = base.MEMBERS
RUN, RECEIPT, STAGE, DIAGNOSTIC, P_BRANCH = base.RUN, base.RECEIPT, base.STAGE, base.DIAGNOSTIC, base.P_BRANCH
CODEC_DIR = "codecs"  # RUN/codecs: install.json, gate-s_r7.json, telemetry/
RUNTIME_DIR = "/kaggle/working/c85r7codec-runtime"
A172_LABEL_HEADER = (
    "# C85R7+codecs v1.1.0: label the codec telemetry branch per A172 phase; nothing else changes.\n"
    "import os as _hb126_os\n"
    "import exp169_release_c85 as _hb126_c85\n"
    "_hb126_phase = _hb126_c85.phase\n"
    "def _hb126_labelled_phase(output, name, action):\n"
    "    _hb126_os.environ['KNEE_CODEC_BRANCH'] = 'A172:' + str(name)\n"
    "    try:\n"
    "        return _hb126_phase(output, name, action)\n"
    "    finally:\n"
    "        _hb126_os.environ['KNEE_CODEC_BRANCH'] = 'A172'\n"
    "_hb126_c85.phase = _hb126_labelled_phase\n")

blend, reference, p_branch, authenticated_pr7 = base.blend, base.reference, base.p_branch, base.authenticated_pr7
preflight, compare_p, stage_replay_input = base.preflight, base.compare_p, base.stage_replay_input


# ---------------------------------------------------------------------------------------- codecs
def codec_dirs(run_dir):
    work = Path(run_dir) / CODEC_DIR
    return work, work / "telemetry"


def child_base_env(runtime):
    """v1.0.0's child environment, unchanged."""
    return dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1",
                HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")


def install_codecs(runtime, output, *, input_root="/kaggle/input", root=None):
    """Parent, before any child: never raises; anything but ACTIVE keeps v1.0.0's environment."""
    work, tele = codec_dirs(output / RUN)
    try:
        rec = codecs.activate(input_root, root=root, telemetry_dir=tele, base_env=child_base_env(runtime))
    except Exception as exc:  # activate() never raises; belt and braces
        rec = dict(status="INACTIVE_ERROR", error=f"{type(exc).__name__}: {exc}"[:300])
    try:
        write_once(work / "install.json", json_bytes(rec))
    except Exception as exc:  # children read this file: without it, stay inactive everywhere
        rec = dict(rec, status="INACTIVE_RECEIPT_WRITE", receipt_error=f"{type(exc).__name__}: {exc}"[:300])
    print("C85R7CODEC_CODECS_INSTALL", json.dumps({k: rec.get(k) for k in ("status", "method", "entry", "reasons",
                                                                           "seconds")}), flush=True)
    return rec


def codec_receipt(run_dir):
    rec = optional_json(Path(run_dir) / CODEC_DIR / "install.json")
    return rec if rec.get("status") == "ACTIVE" and rec.get("hook_dir") else {"status": rec.get("status", "ABSENT")}


def gate_s_r7(run_dir, codec, code_dir):
    """Public worker, after s6.install_wheels: verify the hook under the S stage's pydicom 3.0.1.

    ACTIVE -> branch label S_R7 for the stage; anything else strips the hook from os.environ, so the
    S_R7 stage runs exactly as in v1.0.0. Never raises.
    """
    work, tele = codec_dirs(run_dir)
    record = {"status": "SKIPPED_INACTIVE"}
    try:
        if codec.get("status") == "ACTIVE":
            env = dict(os.environ, PYTHONPATH=str(code_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""))
            result = codecs.verify(codec, base_env=env, branch="verify:S_R7", telemetry_dir=tele)
            status, reasons = codecs.evaluate(result, codec)
            record = dict(status=status, reasons=reasons, pydicom=result.get("pydicom"), gdcm=result.get("gdcm"),
                          selftest={u: t.get("equal") for u, t in (result.get("selftest") or {}).items()},
                          verify=result)
    except Exception as exc:
        record = dict(status="INACTIVE_ERROR", error=f"{type(exc).__name__}: {exc}"[:300])
    try:
        if record["status"] == "ACTIVE":
            os.environ[codecs.BRANCH_ENV] = "S_R7"
        elif codec.get("status") == "ACTIVE":
            stripped = codecs.strip_env(os.environ, codec)
            for key in (codecs.BRANCH_ENV, codecs.TELEMETRY_ENV, "PYTHONPATH"):
                os.environ.pop(key, None)
            os.environ.update({k: v for k, v in stripped.items() if k in ("PYTHONPATH",)})
    except Exception as exc:
        record["env_error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        write_once(work / "gate-s_r7.json", json_bytes(record))
    except Exception:
        pass
    print("C85R7CODEC_CODECS_GATE_S_R7", json.dumps({k: record.get(k) for k in ("status", "reasons", "pydicom")}),
          flush=True)
    return record


def codec_summary(output):
    """Receipt field: install/gate status, per-branch decode counts, per-process hook records."""
    work, tele = codec_dirs(output / RUN)
    try:
        install = optional_json(work / "install.json")
        gate = optional_json(work / "gate-s_r7.json")
        found = codecs.read_telemetry(tele)
        v = install.get("verify") or {}
        return dict(status=install.get("status", "ABSENT"), method=install.get("method"), entry=install.get("entry"),
                    reasons=install.get("reasons"), gdcm=v.get("gdcm"), pydicom=v.get("pydicom"),
                    selftest={u: t.get("equal") for u, t in (v.get("selftest") or {}).items()},
                    wheel_sha256=(install.get("wheel") or {}).get("sha256"), install_seconds=install.get("seconds"),
                    s_r7_gate=gate.get("status", "ABSENT"), s_r7_gate_pydicom=gate.get("pydicom"),
                    branches=found["branches"], totals=found["totals"], telemetry_files=found["files"],
                    processes=found["processes"], orders_variants=found["orders_variants"],
                    hook_errors=found["hook_errors"], error_samples=found["error_samples"],
                    unreadable=found["unreadable"])
    except Exception as exc:
        return dict(status="TELEMETRY_ERROR", error=f"{type(exc).__name__}: {exc}"[:300])


def a172_script(cell, adir, codec):
    source = c85.a172_source(cell, adir)
    return A172_LABEL_HEADER + source if codec.get("status") == "ACTIVE" else source


# ---------------------------------------------------------------------------------------- collect
def candidate_receipt(ids, a, b, payload, codec):
    receipt = base.candidate_receipt(ids, a, b, payload)
    receipt.update(version=VERSION, release=RELEASE, codecs=codec)
    return receipt


def failed_execution(output, error, completed, a172_dir, pr7_dir):
    """v1.0.0's failed_execution with version 1.1.0 and the codec summary."""
    for name in ("submission.csv", DIAGNOSTIC, RECEIPT):
        (output / name).unlink(missing_ok=True)
    children = {}
    for label, path in (("A172", a172_dir / "exp169-executed-candidate.json"),
                        ("R", pr7_dir / "pr7-run/r/receipt.json"),
                        ("S_R7", pr7_dir / "pr7-run/s_r7/controller/submission_receipt.json")):
        if path.is_file():
            children[label] = dict(path=str(path), sha256=sha(path))
    payload = json_bytes(dict(status="FAILED_EXECUTION", version=VERSION, release=RELEASE, intended_members=MEMBERS,
        authenticated_completed_components=completed, child_receipts=children,
        error=f"{type(error).__name__}: {error}", submission_sha256=None, qualified_for_submission=False,
        codecs=codec_summary(output)))
    staged = output / STAGE / "failed-receipt.json"
    write_once(staged, payload)
    c85.publish(staged, output / RECEIPT, payload)


def collect(a172_dir, pr7_dir, output, *, diagnostic=False):
    """v1.0.0's collect; differences: receipt version/release/codecs and the PR7 source version."""
    import numpy as np
    record, stage = output / RECEIPT, output / STAGE
    repeat = diagnostic and record.is_file() and (output / DIAGNOSTIC).is_file()
    need(not (output / "submission.csv").exists(), "stale C85R7 submission")
    need(repeat or not any(path.exists() for path in (record, stage, output / DIAGNOSTIC, output / P_BRANCH)),
         "stale C85R7 output")
    if not repeat:
        stage.mkdir(parents=True)  # exclusive claim; never reused after failure
    completed = []
    try:
        ids, exp143, exp155, q, a = c85.authenticated_a172(a172_dir)
        completed.append("A172")
        pids, r, ours, b = authenticated_pr7(pr7_dir)
        b["source_version"] = "C85R7CODEC " + VERSION
        completed.append("PR7")
        if not ids:
            ids = pids
            exp143, exp155, q = [np.full((len(ids), 12), .5) for _ in range(3)]
            a["failures"] = {name: dict.fromkeys(ids, "A172 child aborted") for name in MEMBERS[2:]}
            a["prediction_fallbacks"] = len(ids)
        if not pids:
            pids = ids
            r, ours = [np.full((len(ids), 12), .5) for _ in range(2)]
            b["failures"] = {name: dict.fromkeys(ids, "PR7 child aborted") for name in MEMBERS[:2]}
            b["r_prediction_fallbacks"] = b["s_r7_prediction_fallbacks"] = len(ids)
        need(set(ids) == set(pids), "C85R7 cohort mismatch")
        order = [pids.index(uid) for uid in ids]
        r, ours = r[order], ours[order]
        values = blend(ids, r, ours, exp143, exp155, q, {**a["failures"], **b["failures"]})
        expected = reference(r, ours, exp143, exp155, q)
        payload = p.csv_bytes(ids, values)
        receipt = candidate_receipt(ids, a, b, payload, codec_summary(output))
        receipt["telemetry"] = dict(healthy_reference_exact=bool(np.array_equal(values, expected)),
                                    uid_order_aligned=ids != pids, csv_float32_roundtrip=True)
        branch = {}
        def p_payload():
            failed = [uid in b["failures"]["S_R7"] for uid in ids]
            branch["payload"] = p.csv_bytes(ids, p_branch(r, ours, failed))
            return dict(file=P_BRANCH, sha256=hashlib.sha256(branch["payload"]).hexdigest(), rows=len(ids),
                        formula="public cell 7: .5 r(R)+.5 r(S_R7), each over own non-failed rows; float32",
                        r_failed=int(np.all(r == .5, axis=1).sum()), s_r7_failed=int(sum(failed)))
        telemetry(receipt["telemetry"], "p_branch", p_payload)
        def p_write():
            if repeat:
                need((output / P_BRANCH).read_bytes() == branch["payload"], "P-branch replay differs")
                return True
            staged = stage / P_BRANCH
            write_once(staged, branch["payload"])
            c85.publish(staged, output / P_BRANCH, branch["payload"])
            return True
        if "payload" in branch:  # diagnostic file only: any error is telemetry
            telemetry(receipt["telemetry"], "p_branch_written", p_write)
        if diagnostic and receipt["status"] == "COMPLETE_C85R7":
            receipt.update(status="DIAGNOSTIC_C85R7", submission_sha256=None)
        receipt_payload = json_bytes(receipt)
        if repeat:
            need(record.read_bytes() == receipt_payload and (output / DIAGNOSTIC).read_bytes() == payload,
                 "immutable diagnostic replay differs")
            return receipt, payload
        if diagnostic or receipt["status"] == "COMPLETE_C85R7":
            name = DIAGNOSTIC if diagnostic else "submission.csv"
            staged = stage / name
            write_once(staged, payload)
            saved_ids, saved = v5q._read_csv(staged)
            v5q.validate_ids(ids, saved_ids)
            v5q._probabilities("final CSV", saved, len(ids))
            receipt["telemetry"]["csv_float32_roundtrip"] = bool(np.array_equal(saved.astype(np.float32), values.astype(np.float32)))
            c85.publish(staged, output / name, payload)
        # Success is materialized only after final CSV publication and verification.
        receipt_payload = json_bytes(receipt)
        staged = stage / RECEIPT
        write_once(staged, receipt_payload)
        c85.publish(staged, record, receipt_payload)
        return receipt, payload
    except BaseException as error:
        if not repeat:
            failed_execution(output, error, completed, a172_dir, pr7_dir)
        raise


# ---------------------------------------------------------------------------------------- runtime
def run_public(runtime, output):
    """Child: v1.0.0's single R pass, then the R7 S stage; R gets the hook env, S_R7 is gated."""
    runtime_pins(runtime)
    comp = competition_root()
    ids = read_json(output / "ids.json")
    work = output / "pr7-run"
    work.mkdir(exist_ok=True)
    weights = v5q._mount(p.DATASETS[0], p.R_NAME) / p.R_NAME
    p.check_pin(weights, p.R_SHA)
    identity = read_json(work / "r7-identity.json")
    paths = read_json(work / "code-paths.json")
    code, anatomy = Path(paths["code"]), Path(paths["anatomy"])
    run_dir = output.parent
    codec = codec_receipt(run_dir)
    _, tele = codec_dirs(run_dir)
    env = codecs.child_env(dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1"), codec,
                           branch="R", telemetry_dir=tele)
    c85.phase(output, "R", lambda: v5q.run_process_group([sys.executable, "-u", str(runtime / "exp169_release_c85.py"),
        "r-worker", "--runtime", str(runtime), "--weights", str(weights), "--competition", str(comp),
        "--output", str(work / "r")], label="C85R7 R", env=env))
    def infer_s():
        bundle = v5q._mount(r7.DATASET_SLUG, "manifest.json")
        s6.install_wheels(sorted((work / "s-code/wheels").glob("*.whl")), code)
        gate_s_r7(run_dir, codec, code)
        r7.infer_studies([comp / "test_series" / uid for uid in ids], bundle_dir=bundle, code_dir=code,
            anatomy_weights=anatomy, output_dir=work / "s_r7", workers=2, amp="auto", device="cuda",
            max_fallback_fraction=1.0, source_dir=runtime / "exp169_r7_src",
            expected_manifest_sha256=identity["manifest_sha256"])
    c85.phase(output, "S_R7", infer_s)


def runtime_pins(runtime):
    base.runtime_pins(runtime)
    for name, digest in (("exp169_release_c85r7.py", C85R7_SHA), ("hb126_codecs.py", CODECS_SHA)):
        p.check_pin(runtime / name, digest)


def run_release(runtime, output, *, input_root="/kaggle/input"):
    """Target-only orchestration; CPU tests replace the process execution boundaries."""
    runtime_pins(runtime)
    need(not any((output / x).exists() for x in ("submission.csv", RUN, RECEIPT, STAGE, DIAGNOSTIC, P_BRANCH)),
         "stale C85R7 output")
    work = output / RUN
    work.mkdir(parents=True)
    adir, pdir = work / "a172", work / "pr7"
    adir.mkdir(); pdir.mkdir()
    env = child_base_env(runtime)
    ids = preflight(runtime, output)
    codec = install_codecs(runtime, output, input_root=input_root)
    _, tele = codec_dirs(work)
    for directory in (adir, pdir): write_once(directory / "ids.json", json_bytes(ids))
    write_once(adir / "q-a5q-manifest.json", (runtime / "q-a5q-manifest.json").read_bytes())
    script = work / "a172.py"
    write_once(script, a172_script((runtime / "a172-cell.py.txt").read_text(encoding="utf-8"), adir, codec).encode())
    c85.phase(adir, "child", lambda: v5q.run_process_group([sys.executable, "-u", str(script)],
        label="C85R7 frozen A172", env=codecs.child_env(env, codec, branch="A172", telemetry_dir=tele)))
    c85.phase(pdir, "child", lambda: v5q.run_process_group([sys.executable, "-u", str(runtime / "exp169_release_c85r7codec.py"),
        "public-worker", "--runtime", str(runtime), "--output", str(pdir)], label="C85R7 R + S_R7",
        env=codecs.child_env(env, codec, branch="PR7", telemetry_dir=tele)))
    receipt, _ = collect(adir, pdir, output)
    summary = receipt.get("codecs") or {}
    print("C85R7CODEC_CODECS", json.dumps({k: summary.get(k) for k in ("status", "s_r7_gate", "totals", "telemetry_files")}
                                          | {"branches": {b: {k: v for k, v in x.items() if k != "by_ts"}
                                                          for b, x in (summary.get("branches") or {}).items()}}),
          flush=True)
    return receipt


# ---------------------------------------------------------------------------------------- packaging
def source_payloads():
    files = base.source_payloads()
    p.check_pin(Path(base.__file__), C85R7_SHA)
    need(hashlib.sha256(files["exp169_release_c85r7.py"]).hexdigest() == C85R7_SHA, "C85-R7 v1.0.0 source drift")
    p.check_pin(Path(codecs.__file__), CODECS_SHA)
    files.update({"exp169_release_c85r7codec.py": Path(__file__).read_bytes(),
                  "hb126_codecs.py": Path(codecs.__file__).read_bytes()})
    return files


def package():
    files = source_payloads()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            archive.writestr(info, data)
    return payload.getvalue(), {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}


def notebook_payload(blob, pins):
    digest = hashlib.sha256(blob).hexdigest()
    bootstrap = "\n".join([
        "import base64, hashlib, io, sys, zipfile", "from pathlib import Path",
        f"print('C85R7+codecs v{VERSION}', flush=True)",
        f"BLOB = {base64.b64encode(blob).decode()!r}", f"PINS = {pins!r}",
        "blob = base64.b64decode(BLOB)", f"assert hashlib.sha256(blob).hexdigest() == {digest!r}",
        f"runtime = Path({RUNTIME_DIR!r})", "assert not runtime.exists(), 'stale runtime'",
        "with zipfile.ZipFile(io.BytesIO(blob)) as archive:",
        "    assert len(archive.namelist()) == len(PINS) and set(archive.namelist()) == set(PINS)",
        "    for name in archive.namelist():",
        "        assert not name.startswith(('/', '\\\\')) and '\\\\' not in name and ':' not in name and '..' not in Path(name).parts",
        "        assert hashlib.sha256(archive.read(name)).hexdigest() == PINS[name]",
        "    runtime.mkdir()", "    archive.extractall(runtime)", "sys.path.insert(0, str(runtime))",
        "import exp169_release_c85r7codec as release",
        "release.run_release(runtime, Path('/kaggle/working'))", ""])
    compile(bootstrap, "<c85r7codec-bootstrap>", "exec")
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
        display_name="Python 3", language="python", name="python3")), cells=[
        dict(cell_type="markdown", id="c85r7codec-policy", metadata={}, source=[
            f"# C85-R7+codecs v{VERSION}\n" + FORMULA + "\nC85-R7 v1.0.0 unchanged (C85 v1.0.4 own branch, single-pass "
            f"public R, S member from heliosli's R7 bundle {r7.BUNDLE_ID}) plus an offline python-gdcm 3.2.6 decoder in "
            "every child so JPEG Lossless / JPEG-LS series decode instead of being blanked or dropped; every plugin that "
            "decoded before stays first. Install failure keeps C85-R7 v1.0.0 behaviour. "
            "Credits: nartaa public R checkpoint (rsna-knee-publication-swa-weights-20261007, used under its model-use "
            "terms; OAI data via NDA, doi 10.15154/0hcg-f676); heliosli R7 bundle (rsna-knee-compact-r7cnxfsss, licence "
            "other) and S-branch serving sources vendored byte-for-byte from heliosli's public notebook "
            "0-957-latest-dataset-score-in-one-click (Apache 2.0); heliosli code dataset "
            "rsna-orthofoundation-fold0-epoch7-20261004; python-gdcm 3.2.6 (Apache-2.0; GDCM BSD-style licence) from "
            "PyPI via our private dataset. Per-study survivor substitution; child exits, anatomy, codec and replay "
            f"checks are telemetry. Writes {P_BRANCH} for visible parity only. No weight search, no timeout; hidden "
            "runtime unknown. CPU-prepared, not qualified.\n"]),
        dict(cell_type="code", id="c85r7codec-run", metadata={}, execution_count=None, outputs=[],
             source=bootstrap.splitlines(True))])


def metadata():
    value = base.metadata()
    value.update(id=KERNEL_ID, title=KERNEL_ID.split("/")[1], dataset_sources=DATASETS)
    return value


def stress_notebook(blob, pins):
    notebook = notebook_payload(blob, pins)
    notebook["cells"][0]["source"] = [f"# C85-R7+codecs v{VERSION} N32 diagnostic only; never submit\n"
        "Full release path (codec install, hook, all children) on 32 training studies selected by max/min series and "
        "slice counts. No labels. Phase exit files, prediction journals and the receipt's codecs field retain timings, "
        "per-process hook records and exceptions.\n"]
    code = "".join(notebook["cells"][1]["source"])
    anchor = "release.run_release(runtime, Path('/kaggle/working'))"
    need(code.count(anchor) == 1, "stress anchor")
    code = code.replace(anchor, "\n".join([
        "import os, time", "import exp169_c85_stress as stress",
        "competition = release.v5q._mount(release.p.COMPETITION, 'train_series.csv')",
        "data = Path('/kaggle/working/c85-stress-data')",
        "selected = stress.prepare(competition, data, n=32)",
        "os.environ['C85_COMPETITION'] = str(data)",
        "print('C85R7CODEC_STRESS_SELECTION', len(selected), flush=True)",
        "started = time.monotonic()",
        "release.run_release(runtime, Path('/kaggle/working'))",
        "print('C85R7CODEC_STRESS_TOTAL_SECONDS', time.monotonic()-started, flush=True)"]))
    compile(code, "<c85r7codec-stress>", "exec")
    notebook["cells"][1]["source"] = code.splitlines(True)
    return notebook


def stress_metadata():
    value = metadata()
    value.update(id=KERNEL_ID + "-stress-n32", title=KERNEL_ID.split("/")[1] + "-stress-n32")
    return value


def prepare(out=OUT):
    blob, pins = package()
    write_once(out / "kernel/release.ipynb", json_bytes(notebook_payload(blob, pins)))
    write_once(out / "kernel/kernel-metadata.json", json_bytes(metadata()))
    write_once(out / "stress-kernel/release.ipynb", json_bytes(stress_notebook(blob, pins)))
    write_once(out / "stress-kernel/kernel-metadata.json", json_bytes(stress_metadata()))
    record = dict(status="PREPARED_LOCAL_ONLY", version=VERSION, release=RELEASE, formula=FORMULA,
        kernel_id=KERNEL_ID, files=pins, archive_sha256=hashlib.sha256(blob).hexdigest(),
        notebook_sha256=sha(out / "kernel/release.ipynb"), metadata_sha256=sha(out / "kernel/kernel-metadata.json"),
        stress_notebook_sha256=sha(out / "stress-kernel/release.ipynb"),
        qualification="PENDING probe v1.1.0 GO_CODECS_RELEASE_READY, independent review, visible run, N32 timing")
    write_once(out / "build-receipt.json", json_bytes(record))
    identities = read_json(C85R7_STAGE / "input-identities.json")
    identities.update(c85r7_v100=dict(source_sha256=C85R7_SHA, stage=str(C85R7_STAGE.relative_to(ROOT)).replace("\\", "/"),
                                      notebook_sha256=read_json(C85R7_STAGE / "build-receipt.json")["notebook_sha256"]),
                      codecs=dict(module_sha256=CODECS_SHA, module_version=codecs.VERSION, dataset=codecs.DATASET_SLUG,
                                  wheel=codecs.WHEEL, hook_version=codecs.HOOK_VERSION,
                                  selftest_array_sha256=codecs.SELFTEST["array_sha256"]),
                      source_pins=pins, dataset_sources=DATASETS)
    write_once(out / "input-identities.json", json_bytes(identities))
    return record


def validate(out=OUT):
    b = read_json(out / "build-receipt.json")
    blob, pins = package()
    need(b["files"] == pins and b["archive_sha256"] == hashlib.sha256(blob).hexdigest(), "source archive drift")
    for name, data, key in (("kernel/release.ipynb", json_bytes(notebook_payload(blob, pins)), "notebook_sha256"),
                            ("kernel/kernel-metadata.json", json_bytes(metadata()), "metadata_sha256"),
                            ("stress-kernel/release.ipynb", json_bytes(stress_notebook(blob, pins)), "stress_notebook_sha256")):
        p.check_pin(out / name, b[key])
        need((out / name).read_bytes() == data, "stage differs: " + name)
    need((out / "stress-kernel/kernel-metadata.json").read_bytes() == json_bytes(stress_metadata()),
         "stress stage differs: kernel-metadata.json")
    meta = read_json(out / "kernel/kernel-metadata.json")
    old = read_json(C85R7_STAGE / "kernel/kernel-metadata.json")
    need(meta["dataset_sources"] == old["dataset_sources"] + [codecs.DATASET_SLUG] and meta["is_private"] is True
         and meta["enable_internet"] is False and meta["docker_image"] == p.IMAGE == old["docker_image"]
         and all(meta[k] == old[k] for k in ("machine_shape", "enable_gpu", "competition_sources")),
         "release metadata policy")
    return dict(status="PASS_CPU_VALIDATE", files=len(pins), notebook_sha256=b["notebook_sha256"],
                archive_sha256=b["archive_sha256"], qualified_for_submission=False)


def register(out=OUT):
    record = dict(**validate(out), version=VERSION, release=RELEASE, formula=FORMULA, kernel_id=KERNEL_ID,
                  public_veto="C85R7+codecs <= C85-R7 v1.0.0 public -0.002",
                  role="candidate (paired with C85-R7 v1.0.0); B gain/duplicate review")
    record["status"] = "REGISTERED_LOCAL_ONLY"
    write_once(out / "registration.json", json_bytes(record))
    return record


ALLOWED_RECEIPT_DIFF = {"version", "release", "codecs"}


def receipt_diff(ours: dict, theirs: dict) -> list[str]:
    """Top-level receipt keys that differ, with the PR7 source_version difference folded in."""
    keys = sorted(k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k))
    out = []
    for k in keys:
        if k == "pr7":
            a = {kk: vv for kk, vv in ours["pr7"].items() if kk != "source_version"}
            b = {kk: vv for kk, vv in theirs["pr7"].items() if kk != "source_version"}
            if a != b:
                out.append("pr7")
        else:
            out.append(k)
    return out


def cpu_replay(output):
    """C85-R7 v1.0.0's arithmetic replay through v1.1.0's collect, plus v1.0.0's collect side by side."""
    import numpy as np
    p.check_pin(c85.A5Q / "exp169-executed-candidate.json", c85.A5Q_RECEIPT_SHA)
    p.check_pin(p.REPLAY / "p956-receipt.json", c85.P956_REPLAY_SHA)
    p.check_pin(p.REPLAY / "submission.csv", read_json(p.REPLAY / "p956-receipt.json")["submission_sha256"])
    staged = stage_replay_input(output / "replay-input/pr7")
    receipt, payload = collect(c85.A5Q, staged, output, diagnostic=True)
    _, repeated = collect(c85.A5Q, staged, output, diagnostic=True)
    need(payload == repeated, "non-deterministic replay")
    need(receipt["telemetry"]["healthy_reference_exact"] is True, "independent reference formula differs")
    old_dir = output / "c85r7-v1.0.0-side-by-side"
    old_receipt, old_payload = base.collect(c85.A5Q, staged, old_dir, diagnostic=True)
    same_csv = payload == old_payload
    same_p = (output / P_BRANCH).read_bytes() == (old_dir / P_BRANCH).read_bytes()
    diff = receipt_diff(receipt, old_receipt)
    need(same_csv and same_p, "C85R7+codecs replay differs from C85-R7 v1.0.0")
    need(set(diff) <= ALLOWED_RECEIPT_DIFF, "unexpected receipt differences: " + ",".join(diff))
    c85_equal = hashlib.sha256(payload).hexdigest() == base.C85_REPLAY_CSV_SHA
    need(c85_equal, "C85R7+codecs with S6 stand-in differs from C85 v1.0.4 replay")
    pids, pvals = v5q._read_csv(output / P_BRANCH)
    rids, rvals = v5q._read_csv(p.REPLAY / "submission.csv")
    p_equal = set(pids) == set(rids) and bool(np.array_equal(
        pvals.astype(np.float32), rvals[[rids.index(u) for u in pids]].astype(np.float32)))
    need(p_equal, "P-branch differs from recorded P956 public-formula submission")
    result = dict(status="PASS_CPU_ARITHMETIC_REPLAY", candidate_status=receipt["status"], version=VERSION,
        studies=receipt["studies"], findings=12, independent_formula_exact=True, deterministic_csv=True,
        c85r7_v100_csv_identical=same_csv, c85r7_v100_p_branch_identical=same_p, receipt_diff_vs_v100=diff,
        c85_v104_csv_identical=c85_equal, p_branch_equals_p956_public_formula=p_equal,
        diagnostic_sha256=hashlib.sha256(payload).hexdigest(), p_branch_sha256=sha(output / P_BRANCH),
        codecs_status=receipt["codecs"]["status"], fallbacks=receipt["fallbacks"], qualified_for_submission=False,
        s_member=base.STAND_IN,
        scope="recorded real R/own/Q and S6 outputs; no inference, no codec install, no hidden-runtime qualification")
    write_once(output / "replay.json", json_bytes(result))
    return result


def admit(out=OUT, replay_dir=None):
    validation = validate(out)
    reg = read_json(out / "registration.json")
    need(reg["notebook_sha256"] == validation["notebook_sha256"], "registration drift")
    replay_dir = replay_dir or out / "cpu-replay"
    replay = read_json(replay_dir / "replay.json")
    need(replay["status"] == "PASS_CPU_ARITHMETIC_REPLAY" and replay["independent_formula_exact"]
         and replay["deterministic_csv"] and replay["c85_v104_csv_identical"] and replay["c85r7_v100_csv_identical"]
         and replay["c85r7_v100_p_branch_identical"] and replay["p_branch_equals_p956_public_formula"],
         "CPU arithmetic replay required")
    p.check_pin(replay_dir / DIAGNOSTIC, replay["diagnostic_sha256"])
    return dict(status="HOLD_NOT_QUALIFIED", cpu_preparation="PASS", qualified_for_submission=False,
        candidate_replay_status=replay["candidate_status"],
        remaining=["probe v1.1.0 on Kaggle: HB126_VERDICT GO_CODECS_RELEASE_READY",
                   "independent review (tools/hb126_codecs.py, tools/exp169_release_c85r7codec.py)",
                   "visible run: codecs ACTIVE, S_R7 gate ACTIVE, zero decode failures, submission.csv and "
                   "p-branch-r7.csv equal to the C85-R7 v1.0.0 visible run (compare-csv)",
                   "N32: codecs ACTIVE in every pydicom process, timing inside Kaggle's 9 h", "B gain/duplicate review"])


def compare_csv(ours, theirs):
    """Visible check: two submission-format CSVs, UID-aligned, exact float32 and max abs diff."""
    import numpy as np
    oids, ov = v5q._read_csv(Path(ours))
    tids, tv = v5q._read_csv(Path(theirs))
    need(set(oids) == set(tids) and len(oids) == len(tids), "CSV cohorts differ")
    aligned = tv[[tids.index(uid) for uid in oids]]
    exact = bool(np.array_equal(ov.astype(np.float32), aligned.astype(np.float32)))
    diff = np.abs(ov.astype(np.float64) - aligned.astype(np.float64))
    return dict(status="CSV_IDENTICAL" if exact else "CSV_DIFFERS", exact_float32=exact, studies=len(oids),
                max_abs_diff=float(diff.max()) if diff.size else 0.0, cells_differing=int((diff > 0).sum()),
                ours_sha256=sha(Path(ours)), theirs_sha256=sha(Path(theirs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "register", "admit"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--out", type=Path, default=OUT)
        if name == "admit":
            cmd.add_argument("--dry-run", required=True, action="store_true")
            cmd.add_argument("--replay-dir", type=Path)
    cmd = sub.add_parser("cpu-replay")
    cmd.add_argument("--output", type=Path, default=OUT / "cpu-replay")
    cmd = sub.add_parser("public-worker")
    cmd.add_argument("--runtime", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd = sub.add_parser("run")
    cmd.add_argument("--runtime", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    for name in ("compare-p", "compare-csv"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--ours", type=Path, required=True)
        cmd.add_argument("--public" if name == "compare-p" else "--theirs", type=Path, required=True)
    cmd = sub.add_parser("codec-telemetry")
    cmd.add_argument("--output", type=Path, required=True, help="a release output dir (contains c85r7-run/)")
    args = parser.parse_args()
    if args.command == "public-worker": result = run_public(args.runtime, args.output)
    elif args.command == "compare-p": result = compare_p(args.ours, args.public)
    elif args.command == "compare-csv": result = compare_csv(args.ours, args.theirs)
    elif args.command == "codec-telemetry": result = codec_summary(args.output)
    elif args.command == "admit": result = admit(args.out, args.replay_dir)
    elif args.command == "run": result = run_release(args.runtime, args.output)
    elif args.command == "cpu-replay": result = cpu_replay(args.output)
    else: result = globals()[args.command](args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
