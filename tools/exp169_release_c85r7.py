"""C85-R7 v1.0.0: C85 v1.0.4 with the S member served from heliosli's public R7 bundle.

2026-10-13 (Claude opus builder stand-in for B, H-B-125): new wrapper only. C85 v1.0.4,
P956, A172/v5q, EXP189 and every historical stage are imported unchanged and hash-pinned.
Formula unchanged: P=.5 r(R)+.5 r(S_R7); O=(17 r(EXP143v2)+17 r(EXP155)+12 r(Q_A5q))/46;
C85R7=.85 r(P)+.15 r(O). Only S changes: bundle 8c7b666e27dc0e8f (heliosli/
rsna-knee-compact-r7cnxfsss) served by exp169_r7_port with the R7 notebook's own sources
and manifest-driven worker arguments. R, Q and both own arms run exactly as in C85 v1.0.4
(R through C85's r-worker, one pass).
Kept from C85 v1.0.4: successful-row nested ranks with survivor substitution (c85.blend),
isolated child phases, recovered child journals, telemetry-only vetoes, verified atomic
publication and FAILED_EXECUTION receipts. New: p-branch-r7.csv, the raw P branch in the
public cell-7 formula (the heliosli notebook's own submission), written best-effort for
visible parity; it is never a submission and never a veto.
CPU commands never infer, install, submit, query scores, or operate on a Kaggle account.
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
    from tools import exp169_release_c85 as c85, exp169_r7_port as r7
except ImportError:
    import exp169_release_c85 as c85
    import exp169_r7_port as r7

p, v5q, s6, a172 = c85.p, c85.v5q, c85.s6, c85.a172
need, sha, read_json, write_once, json_bytes = p.need, p.sha, p.read_json, p.write_once, p.json_bytes
telemetry, optional_json, competition_root = c85.telemetry, c85.optional_json, c85.competition_root
ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.0.0"
KERNEL_ID = "taeyangg4/knee-exp169-c85r7-v1"
FORMULA = "P=.5*r(R)+.5*r(S_R7); O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46; C85R7=.85*r(P)+.15*r(O)"
C85_SHA = "291f63d4bc9b79ec7bd06957842d7c2360e352cfd5aad257c64daf003348e27c"
C85_STRESS_SHA = "4d10508371139e9a9125f96c183db08e079b6502c41ad4ebb4b568b5f5cb42d1"
C85_REPLAY_CSV_SHA = "a59be5c6066f87b5309e242ecced132fab1894b8dd4b69169e8450a171ddee8a"
R7_PORT_SHA = "dd6eae30d742a4f02eb6602ca882e4f9df8b54824e83154b0fac5de4e119f4da"
EXP155_ARCHIVE_SHA = "40b8d02847bf36938d82a79e88cb1dad4296db0cc95691b8cd0d2a2ce1a7e5fa"
DATASETS = [c85.DATASETS[0], c85.DATASETS[1], p.DATASETS[0], r7.DATASET_SLUG, v5q.CODE_SLUG]
MEMBERS = ["R", "S_R7", "EXP143v2", "EXP155", "Q_A5q"]
RUN, RECEIPT, STAGE, DIAGNOSTIC = "c85r7-run", "c85r7-receipt.json", ".c85r7-publication", "diagnostic-c85r7.csv"
P_BRANCH = "p-branch-r7.csv"
STAND_IN = "S6 v1.0.0 Kaggle raw outputs relabelled S_R7: arithmetic replay only, not R7 inference"


def blend(ids, r, ours, exp143, exp155, q, failures=None):
    """C85 v1.0.4's nested blend unchanged; only the S member's failure key is renamed."""
    failures = dict(failures or {})
    if "S_R7" in failures:
        failures["S6"] = failures.pop("S_R7")
    return c85.blend(ids, r, ours, exp143, exp155, q, failures)


reference = c85.reference


def p_branch(r, ours, s_failed):
    """Public cell 7 (P956's independent transcription): float32 P branch only."""
    return p.public_formula(r, ours, s_failed)


def authenticated_pr7(directory):
    """R journal (or recorded R raw) + R7 worker rows; identity is hard, the rest telemetry."""
    import csv
    import numpy as np
    directory = Path(directory)
    work = directory / "pr7-run"
    checks = {}
    rr = optional_json(work / "r/receipt.json")
    controller = optional_json(work / "s_r7/controller/submission_receipt.json")
    if (directory / "ids.json").exists(): ids = v5q.validate_ids(read_json(directory / "ids.json"))
    elif (work / "r/raw.npz").exists():
        with np.load(work / "r/raw.npz", allow_pickle=False) as raw: ids = v5q.validate_ids(raw["ids"].tolist())
    elif (work / "s_r7/predictions.csv").exists(): ids, _ = v5q._read_csv(work / "s_r7/predictions.csv")
    else: ids = []
    r, ours = np.full((len(ids), 12), .5), np.full((len(ids), 12), .5)
    rf = dict.fromkeys(ids, "MissingPrediction: R child aborted")
    sf = dict.fromkeys(ids, "MissingPrediction: S_R7 child aborted")
    if (work / "r/journal-identity.json").exists():
        vals, fails = c85.read_journal(work / "r", ids, ["R"], {"checkpoint_sha256": p.R_SHA, "source_pins": p.PINS})
        r, rf = vals["R"], fails["R"]
    elif (work / "r/raw.npz").exists():  # recorded P956 R output (CPU replay); bound by its own receipt
        need(rr.get("status") == "PASS_R", "R raw output lacks its receipt")
        p.check_pin(work / "r/raw.npz", rr["raw_sha256"])
        with np.load(work / "r/raw.npz", allow_pickle=False) as raw:
            need(set(raw.files) == {"ids", "targets", "public"} and raw["targets"].tolist() == list(p.LABELS), "R target/schema mismatch")
            rids = v5q.validate_ids(raw["ids"].tolist())
            need(set(ids) == set(rids), "R study mismatch")
            r, rf = c85.clean_member(ids, raw["public"][[rids.index(uid) for uid in ids]], rr.get("failures", {}))
    rf.update({uid: "FailedStudySentinel: public cell 7" for uid, failed in zip(ids, np.all(r == .5, axis=1)) if failed})
    prov_path = work / "s_r7/provenance.json"
    workers = sorted((work / "s_r7/controller").glob("worker*"))
    identity = {}
    if prov_path.exists() or workers:
        prov = read_json(prov_path)
        expected = read_json(work / "r7-identity.json")
        for key, value in [("dataset_slug", r7.DATASET_SLUG), ("bundle_id", r7.BUNDLE_ID), ("source_sha256", r7.SOURCE_HASHES),
                           ("anchor_weights_sha256", s6.ANATOMY_SHA256), ("manifest_sha256", expected["manifest_sha256"]),
                           ("file_sha256", expected["file_sha256"])]:
            need(prov.get(key) == value, "S_R7 identity: " + key)
        need(expected["bundle_id"] == r7.BUNDLE_ID and expected["dataset_slug"] == r7.DATASET_SLUG, "S_R7 preflight identity")
        need(r7.MANIFEST_SHA256 is None or prov["manifest_sha256"] == r7.MANIFEST_SHA256, "S_R7 identity: manifest pin")
        identity = dict(manifest_sha256=prov["manifest_sha256"], manifest_pinned=prov.get("manifest_pinned"),
                        features=prov.get("features"), worker_args=prov.get("worker_args"), stand_in=prov.get("stand_in"))
        assigned, scored = set(), set()
        for worker in workers:
            if not worker.is_dir(): continue
            if not (worker / "ids.json").exists():
                checks[worker.name] = "child stopped before assignment was written"
                continue
            part = read_json(worker / "ids.json")
            need(len(set(part)) == len(part) and not assigned.intersection(part) and set(part) <= set(ids), "S_R7 assignment cohort")
            assigned.update(part)
            wr = optional_json(worker / "receipt.json")
            sf.update({uid: why for uid, why in wr.get("fallback", {}).items() if uid in ids})
            prediction = worker / "predictions.csv"
            if not prediction.exists(): continue
            with prediction.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(line for line in stream if line.endswith("\n"))
                need(reader.fieldnames == [p.UID, *p.LABELS], "S_R7 prediction schema")
                for row in reader:
                    uid = row[p.UID]
                    need(uid in part and uid not in scored, "S_R7 prediction UID differs")
                    try: value = p.RBinding.prediction([float(row[t]) for t in p.LABELS])
                    except (ValueError, TypeError) as exc:
                        sf[uid] = str(exc)
                        continue
                    ours[ids.index(uid)] = value
                    if uid not in wr.get("fallback", {}): sf.pop(uid, None)
                    scored.add(uid)
        checks["assigned_cohort"] = assigned == set(ids)
    telemetry(checks, "controller_crosscheck", lambda: p.s6_failure_evidence(work / "s_r7/controller", ids, controller))
    anchors = dict(scored=len(ids) - len(sf), model_anchor=0, fallback=0, methods={})
    try: anchors = p.anatomy_anchors(controller)
    except (ValueError, KeyError, TypeError) as exc: checks["anatomy"] = str(exc)
    if (work / "s_r7/predictions.csv").exists():
        def controller_rows():
            sids, values = v5q._read_csv(work / "s_r7/predictions.csv")
            good = [uid for uid in ids if uid not in sf]
            return bool(set(sids) == set(ids) and all((values[sids.index(u)] == ours[ids.index(u)]).all() for u in good))
        telemetry(checks, "controller_rows_match_workers", controller_rows)
    checks.update(r_fallback_count=rr.get("failure_count", len(rr.get("failures", {}))) == len(rf))
    return ids, r, ours, dict(studies=len(ids), executed_arms=["R", "S_R7"], source_version="C85R7 " + VERSION,
        worker_exit_codes=controller.get("worker_exit_codes", {}), r_prediction_fallbacks=len(rf),
        s_r7_prediction_fallbacks=len(sf), s_r7_anatomy_anchors=anchors, r7_identity=identity,
        failures={"R": rf, "S_R7": sf}, telemetry=checks, child=optional_json(directory / "child-exit.json"),
        phases={name: optional_json(directory / (name + "-exit.json")) for name in ("R", "S_R7")})


def candidate_receipt(ids, a, b, payload):
    return dict(status="COMPLETE_C85R7", version=VERSION, formula=FORMULA,
        rank="average / successful rows, float64 nested ranks; final CSV float32",
        failure_policy="renormalise own members per study; surviving branch rank; both branches failed 0.5",
        intended_members=MEMBERS, executed_members=MEMBERS,
        execution_passes=[dict(serving="A172/v5q own-only (C85 v1.0.4)", arms=["EXP155", "EXP143v2", "Q_A5q"]),
                          dict(serving="C85 v1.0.4 R single pass + exp169_r7_port " + r7.VERSION, arms=["R", "S_R7"])],
        studies=len(ids), findings=12,
        fallbacks=dict(a172_prediction=a["prediction_fallbacks"], r_prediction=b["r_prediction_fallbacks"],
                       s_r7_prediction=b["s_r7_prediction_fallbacks"]),
        s_r7_anatomy_anchors=b["s_r7_anatomy_anchors"], r7_identity=b["r7_identity"], a172=a, pr7=b,
        submission_sha256=hashlib.sha256(payload).hexdigest(), qualified_for_submission=False)


def failed_execution(output, error, completed, a172_dir, pr7_dir):
    """Only after this invocation claimed a fresh output; keeps staged/child evidence."""
    for name in ("submission.csv", DIAGNOSTIC, RECEIPT):
        (output / name).unlink(missing_ok=True)
    children = {}
    for label, path in (("A172", a172_dir / "exp169-executed-candidate.json"),
                        ("R", pr7_dir / "pr7-run/r/receipt.json"),
                        ("S_R7", pr7_dir / "pr7-run/s_r7/controller/submission_receipt.json")):
        if path.is_file():
            children[label] = dict(path=str(path), sha256=sha(path))
    payload = json_bytes(dict(status="FAILED_EXECUTION", version=VERSION, intended_members=MEMBERS,
        authenticated_completed_components=completed, child_receipts=children,
        error=f"{type(error).__name__}: {error}", submission_sha256=None, qualified_for_submission=False))
    staged = output / STAGE / "failed-receipt.json"
    write_once(staged, payload)
    c85.publish(staged, output / RECEIPT, payload)


def collect(a172_dir, pr7_dir, output, *, diagnostic=False):
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
        receipt = candidate_receipt(ids, a, b, payload)
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


def preflight(runtime, output):
    """Identity checks precede every recoverable child error; R7 identity is recorded once."""
    import csv
    comp = competition_root()
    ids, _ = v5q._read_csv(comp / "sample_submission.csv")
    with (comp / "test.csv").open(newline="", encoding="utf-8") as stream:
        test_ids = v5q.validate_ids([row[p.UID] for row in csv.DictReader(stream)])
    need(set(ids) == set(test_ids), "test/sample cohort")
    p.check_pin(v5q._mount(p.DATASETS[0], p.R_NAME) / p.R_NAME, p.R_SHA)
    for slug, filename, digest in [(DATASETS[0], "exp169-assets.zip.bin", v5q.A172_ARCHIVE_SHA),
                                   (DATASETS[1], "exp155-assets.zip.bin", EXP155_ARCHIVE_SHA)]:
        p.check_pin(v5q._mount(slug, filename) / filename, digest)
    work = output / RUN / "pr7" / "pr7-run"
    source = runtime / "exp169_r7_src"
    r7.verify_source(source)
    _, identity = r7.verify_bundle(v5q._mount(r7.DATASET_SLUG, "manifest.json"), source_dir=source)
    write_once(work / "r7-identity.json", json_bytes(identity))
    print("C85R7_R7_IDENTITY", json.dumps(dict(manifest_sha256=identity["manifest_sha256"],
                                               features=identity["features"])), flush=True)
    code, anatomy = s6.prepare_code(v5q._mount(v5q.CODE_SLUG, "anatomy.pt"), work / "s-code", install_wheels=False)
    write_once(work / "code-paths.json", json_bytes(dict(code=str(code), anatomy=str(anatomy))))
    return ids


def run_public(runtime, output):
    """Child: C85's single R pass, then the R7 S stage; each phase isolated."""
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
    env = dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1")
    c85.phase(output, "R", lambda: v5q.run_process_group([sys.executable, "-u", str(runtime / "exp169_release_c85.py"),
        "r-worker", "--runtime", str(runtime), "--weights", str(weights), "--competition", str(comp),
        "--output", str(work / "r")], label="C85R7 R", env=env))
    def infer_s():
        bundle = v5q._mount(r7.DATASET_SLUG, "manifest.json")
        s6.install_wheels(sorted((work / "s-code/wheels").glob("*.whl")), code)
        r7.infer_studies([comp / "test_series" / uid for uid in ids], bundle_dir=bundle, code_dir=code,
            anatomy_weights=anatomy, output_dir=work / "s_r7", workers=2, amp="auto", device="cuda",
            max_fallback_fraction=1.0, source_dir=runtime / "exp169_r7_src",
            expected_manifest_sha256=identity["manifest_sha256"])
    c85.phase(output, "S_R7", infer_s)


def runtime_pins(runtime):
    c85.runtime_pins(runtime)
    for name, digest in (("exp169_release_c85.py", C85_SHA), ("exp169_c85_stress.py", C85_STRESS_SHA),
                         ("exp169_r7_port.py", R7_PORT_SHA)):
        p.check_pin(runtime / name, digest)
    r7.verify_source(runtime / "exp169_r7_src")


def run_release(runtime, output):
    """Target-only orchestration; CPU tests replace the process execution boundaries."""
    runtime_pins(runtime)
    need(not any((output / x).exists() for x in ("submission.csv", RUN, RECEIPT, STAGE, DIAGNOSTIC, P_BRANCH)),
         "stale C85R7 output")
    work = output / RUN
    work.mkdir(parents=True)
    adir, pdir = work / "a172", work / "pr7"
    adir.mkdir(); pdir.mkdir()
    env = dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    ids = preflight(runtime, output)
    for directory in (adir, pdir): write_once(directory / "ids.json", json_bytes(ids))
    write_once(adir / "q-a5q-manifest.json", (runtime / "q-a5q-manifest.json").read_bytes())
    script = work / "a172.py"
    write_once(script, c85.a172_source((runtime / "a172-cell.py.txt").read_text(encoding="utf-8"), adir).encode())
    c85.phase(adir, "child", lambda: v5q.run_process_group([sys.executable, "-u", str(script)],
        label="C85R7 frozen A172", env=env))
    c85.phase(pdir, "child", lambda: v5q.run_process_group([sys.executable, "-u", str(runtime / "exp169_release_c85r7.py"),
        "public-worker", "--runtime", str(runtime), "--output", str(pdir)], label="C85R7 R + S_R7", env=env))
    receipt, _ = collect(adir, pdir, output)
    return receipt


def source_payloads():
    files = c85.source_payloads()
    p.check_pin(Path(c85.__file__), C85_SHA)
    for name, digest in (("exp169_release_c85.py", C85_SHA), ("exp169_c85_stress.py", C85_STRESS_SHA)):
        need(hashlib.sha256(files[name]).hexdigest() == digest, "C85 v1.0.4 source drift: " + name)
    r7.verify_source()
    p.check_pin(Path(r7.__file__), R7_PORT_SHA)
    files.update({"exp169_release_c85r7.py": Path(__file__).read_bytes(),
                  "exp169_r7_port.py": Path(r7.__file__).read_bytes(),
                  **{"exp169_r7_src/" + name: (r7.SOURCE / name).read_bytes() for name in r7.SOURCE_HASHES}})
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
        f"print('C85R7 v{VERSION}', flush=True)",
        f"BLOB = {base64.b64encode(blob).decode()!r}", f"PINS = {pins!r}",
        "blob = base64.b64decode(BLOB)", f"assert hashlib.sha256(blob).hexdigest() == {digest!r}",
        "runtime = Path('/kaggle/working/c85r7-runtime')", "assert not runtime.exists(), 'stale runtime'",
        "with zipfile.ZipFile(io.BytesIO(blob)) as archive:",
        "    assert len(archive.namelist()) == len(PINS) and set(archive.namelist()) == set(PINS)",
        "    for name in archive.namelist():",
        "        assert not name.startswith(('/', '\\\\')) and '\\\\' not in name and ':' not in name and '..' not in Path(name).parts",
        "        assert hashlib.sha256(archive.read(name)).hexdigest() == PINS[name]",
        "    runtime.mkdir()", "    archive.extractall(runtime)", "sys.path.insert(0, str(runtime))",
        "import exp169_release_c85r7 as release",
        "release.run_release(runtime, Path('/kaggle/working'))", ""])
    compile(bootstrap, "<c85r7-bootstrap>", "exec")
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
        display_name="Python 3", language="python", name="python3")), cells=[
        dict(cell_type="markdown", id="c85r7-policy", metadata={}, source=[
            f"# C85-R7 v{VERSION}\n" + FORMULA + "\nC85 v1.0.4 (frozen A5q own branch, single-pass public R) with the S member "
            f"served from heliosli's R7 bundle {r7.BUNDLE_ID}. "
            "Credits: nartaa public R checkpoint (rsna-knee-publication-swa-weights-20261007, used under its model-use "
            "terms; OAI data via NDA, doi 10.15154/0hcg-f676); heliosli R7 bundle (rsna-knee-compact-r7cnxfsss, licence other) and "
            "S-branch serving sources vendored byte-for-byte from heliosli's public notebook "
            "0-957-latest-dataset-score-in-one-click (Apache 2.0); heliosli code dataset "
            "rsna-orthofoundation-fold0-epoch7-20261004. "
            "Per-study survivor substitution; child exits, anatomy and replay checks are telemetry. "
            f"Writes {P_BRANCH} (raw public-formula P branch) for visible parity only. "
            "No weight search, no timeout; hidden runtime unknown. CPU-prepared, not qualified.\n"]),
        dict(cell_type="code", id="c85r7-run", metadata={}, execution_count=None, outputs=[], source=bootstrap.splitlines(True))])


def metadata():
    return dict(id=KERNEL_ID, title=KERNEL_ID.split("/")[1], code_file="release.ipynb",
        language="python", kernel_type="notebook", is_private=True, enable_gpu=True, enable_tpu=False,
        enable_internet=False, machine_shape=p.MACHINE, docker_image=p.IMAGE,
        dataset_sources=DATASETS, competition_sources=[p.COMPETITION], kernel_sources=[], model_sources=[])


def stress_notebook(blob, pins):
    notebook = notebook_payload(blob, pins)
    notebook["cells"][0]["source"] = [f"# C85-R7 v{VERSION} N32 diagnostic only; never submit\n"
        "Full release path on 32 training studies selected by max/min series and slice counts. "
        "No labels. Phase exit files and prediction journals retain timings and exceptions.\n"]
    code = "".join(notebook["cells"][1]["source"])
    code = code.replace("release.run_release(runtime, Path('/kaggle/working'))", "\n".join([
        "import os, time", "import exp169_c85_stress as stress",
        "competition = release.v5q._mount(release.p.COMPETITION, 'train_series.csv')",
        "data = Path('/kaggle/working/c85-stress-data')",
        "selected = stress.prepare(competition, data, n=32)",
        "os.environ['C85_COMPETITION'] = str(data)",
        "print('C85R7_STRESS_SELECTION', len(selected), flush=True)",
        "started = time.monotonic()",
        "release.run_release(runtime, Path('/kaggle/working'))",
        "print('C85R7_STRESS_TOTAL_SECONDS', time.monotonic()-started, flush=True)"]))
    compile(code, "<c85r7-stress>", "exec")
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
    record = dict(status="PREPARED_LOCAL_ONLY", version=VERSION, formula=FORMULA,
        kernel_id=KERNEL_ID, files=pins, archive_sha256=hashlib.sha256(blob).hexdigest(),
        notebook_sha256=sha(out / "kernel/release.ipynb"), metadata_sha256=sha(out / "kernel/kernel-metadata.json"),
        stress_notebook_sha256=sha(out / "stress-kernel/release.ipynb"),
        qualification="PENDING independent review, fresh visible run with P-branch parity, N32 timing")
    write_once(out / "build-receipt.json", json_bytes(record))
    m = c85.manifest(c85.A5Q / "exp169-partner/release-manifest.json")
    write_once(out / "input-identities.json", json_bytes(dict(q_manifest_sha256=c85.MANIFEST_SHA,
        q_checkpoints=m["entries"], anchor_binding=m["anchor_binding"],
        r=dict(name=p.R_NAME, sha256=p.R_SHA, dataset=p.DATASETS[0], serving_pins=p.PINS),
        s_r7=dict(dataset=r7.DATASET_SLUG, bundle_id=r7.BUNDLE_ID, manifest_sha256=r7.MANIFEST_SHA256,
                  manifest_policy="unpinned before first visible run: bundle_id + manifest SHA256 map, observed digest recorded",
                  roster=list(r7.ROSTER), source_sha256=r7.SOURCE_HASHES, notebook=r7.NOTEBOOK,
                  code_dataset=v5q.CODE_SLUG, code_tree_sha256=s6.CODE_TREE_SHA256, anatomy_sha256=s6.ANATOMY_SHA256,
                  port_sha256=R7_PORT_SHA),
        c85_v104=dict(source_sha256=C85_SHA, replay_csv_sha256=C85_REPLAY_CSV_SHA),
        a172_archive_sha256=v5q.A172_ARCHIVE_SHA, a172_notebook_sha256=v5q.A172_NOTEBOOK_SHA,
        a172_cell_sha256=v5q.A172_CELL_SHA, p956_source_sha256=c85.P956_SHA,
        source_pins=pins, dataset_sources=DATASETS)))
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
    need(meta["dataset_sources"] == DATASETS and s6.DATASET_SLUG not in meta["dataset_sources"]
         and meta["is_private"] is True and meta["enable_internet"] is False and meta["docker_image"] == p.IMAGE,
         "release metadata policy")
    return dict(status="PASS_CPU_VALIDATE", files=len(pins), notebook_sha256=b["notebook_sha256"],
                archive_sha256=b["archive_sha256"], qualified_for_submission=False)


def register(out=OUT):
    record = dict(**validate(out), version=VERSION, formula=FORMULA, kernel_id=KERNEL_ID,
                  public_veto="C85R7 <= own C85 v1.0.4 public -0.002", role="candidate; B gain/duplicate review")
    record["status"] = "REGISTERED_LOCAL_ONLY"
    write_once(out / "registration.json", json_bytes(record))
    return record


REPLAY_FILES = {"p956-run/r/raw.npz": "pr7-run/r/raw.npz", "p956-run/r/receipt.json": "pr7-run/r/receipt.json",
                "p956-run/s6/predictions.csv": "pr7-run/s_r7/predictions.csv",
                "p956-run/s6/controller/submission_receipt.json": "pr7-run/s_r7/controller/submission_receipt.json",
                **{f"p956-run/s6/controller/worker{i}/{n}": f"pr7-run/s_r7/controller/worker{i}/{n}"
                   for i in range(2) for n in ("ids.json", "predictions.csv", "receipt.json")}}


def stage_replay_input(directory):
    """Recorded real R/S6 outputs in the R7 layout, S6 explicitly labelled as a stand-in."""
    for source, target in REPLAY_FILES.items():
        write_once(directory / target, (p.REPLAY / source).read_bytes())
    files = {name: "stand-in" for name in r7.ROSTER}
    identity = dict(dataset_slug=r7.DATASET_SLUG, bundle_id=r7.BUNDLE_ID, manifest_sha256="stand-in", manifest_pinned=False,
                    file_sha256=files, declared=list(r7.MODELS), features={}, stand_in=STAND_IN)
    write_once(directory / "pr7-run/r7-identity.json", json_bytes(identity))
    write_once(directory / "pr7-run/s_r7/provenance.json", json_bytes(dict(identity, source_sha256=r7.SOURCE_HASHES,
        anchor_weights_sha256=s6.ANATOMY_SHA256, worker_args=None)))
    return directory


def cpu_replay(output):
    import numpy as np
    p.check_pin(c85.A5Q / "exp169-executed-candidate.json", c85.A5Q_RECEIPT_SHA)
    p.check_pin(p.REPLAY / "p956-receipt.json", c85.P956_REPLAY_SHA)
    p.check_pin(p.REPLAY / "submission.csv", read_json(p.REPLAY / "p956-receipt.json")["submission_sha256"])
    staged = stage_replay_input(output / "replay-input/pr7")
    receipt, payload = collect(c85.A5Q, staged, output, diagnostic=True)
    _, repeated = collect(c85.A5Q, staged, output, diagnostic=True)
    need(payload == repeated, "non-deterministic replay")
    need(receipt["telemetry"]["healthy_reference_exact"] is True, "independent reference formula differs")
    c85_equal = hashlib.sha256(payload).hexdigest() == C85_REPLAY_CSV_SHA
    need(c85_equal, "C85R7 with S6 stand-in differs from C85 v1.0.4 replay")
    need(receipt["telemetry"].get("p_branch_written") is True, "P-branch diagnostic missing")
    pids, pvals = v5q._read_csv(output / P_BRANCH)
    rids, rvals = v5q._read_csv(p.REPLAY / "submission.csv")
    need(set(pids) == set(rids), "P-branch cohort")
    p_equal = bool(np.array_equal(pvals.astype(np.float32), rvals[[rids.index(u) for u in pids]].astype(np.float32)))
    need(p_equal, "P-branch differs from recorded P956 public-formula submission")
    result = dict(status="PASS_CPU_ARITHMETIC_REPLAY", candidate_status=receipt["status"],
        studies=receipt["studies"], findings=12, independent_formula_exact=True, deterministic_csv=True,
        c85_v104_csv_identical=c85_equal, p_branch_equals_p956_public_formula=p_equal,
        p_branch_csv_bytes_equal=(output / P_BRANCH).read_bytes() == (p.REPLAY / "submission.csv").read_bytes(),
        diagnostic_sha256=hashlib.sha256(payload).hexdigest(), p_branch_sha256=sha(output / P_BRANCH),
        fallbacks=receipt["fallbacks"], qualified_for_submission=False, s_member=STAND_IN,
        scope="recorded real R/own/Q and S6 outputs; no R7 inference, no hidden-runtime qualification")
    write_once(output / "replay.json", json_bytes(result))
    return result


def admit(out=OUT, replay_dir=None):
    validation = validate(out)
    reg = read_json(out / "registration.json")
    need(reg["notebook_sha256"] == validation["notebook_sha256"], "registration drift")
    replay_dir = replay_dir or out / "cpu-replay"
    replay = read_json(replay_dir / "replay.json")
    need(replay["status"] == "PASS_CPU_ARITHMETIC_REPLAY" and replay["independent_formula_exact"]
         and replay["deterministic_csv"] and replay["c85_v104_csv_identical"]
         and replay["p_branch_equals_p956_public_formula"], "CPU arithmetic replay required")
    p.check_pin(replay_dir / DIAGNOSTIC, replay["diagnostic_sha256"])
    return dict(status="HOLD_NOT_QUALIFIED", cpu_preparation="PASS", qualified_for_submission=False,
        candidate_replay_status=replay["candidate_status"],
        remaining=["independent review", "fresh visible run: R7 identity + p-branch-r7.csv == heliosli notebook submission",
                   "N32 timing demonstrating the 9 h Kaggle limit; hidden runtime unknown", "B gain/duplicate review"])


def compare_p(ours, public):
    """Visible parity: our p-branch-r7.csv vs the heliosli R7 notebook's submission.csv, UID-aligned."""
    import numpy as np
    oids, ov = v5q._read_csv(Path(ours))
    pids, pv = v5q._read_csv(Path(public))
    need(set(oids) == set(pids) and len(oids) == len(pids), "P-branch cohorts differ")
    aligned = pv[[pids.index(uid) for uid in oids]]
    exact = bool(np.array_equal(ov.astype(np.float32), aligned.astype(np.float32)))
    diff = np.abs(ov.astype(np.float64) - aligned.astype(np.float64))
    return dict(status="P_BRANCH_MATCH" if exact else "P_BRANCH_DIFFERS", exact_float32=exact, studies=len(oids),
                max_abs_diff=float(diff.max()) if diff.size else 0.0, cells_differing=int((diff > 0).sum()),
                ours_sha256=sha(Path(ours)), public_sha256=sha(Path(public)))


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
    cmd = sub.add_parser("compare-p")
    cmd.add_argument("--ours", type=Path, required=True)
    cmd.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "public-worker": result = run_public(args.runtime, args.output)
    elif args.command == "compare-p": result = compare_p(args.ours, args.public)
    elif args.command == "admit": result = admit(args.out, args.replay_dir)
    elif args.command == "run": result = run_release(args.runtime, args.output)
    elif args.command == "cpu-replay": result = cpu_replay(args.output)
    else: result = globals()[args.command](args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
