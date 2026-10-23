"""C85-R7+codecs v1.2.2: deadline/process insurance only; frozen v1.1.0 math.

Changelog 2026-10-13 v1.2.2: bound required R7 receipts and sticky journal failures;
atomic publication and original authenticated fallback arithmetic retained.
New wrapper, never changes the submitted stage. H-B-129 helper is imported intact.
The root seeds the cohort, supervises setup and each independent branch, then
uses the original authenticated collector outside the compute deadline.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

try:
    from tools import exp169_release_c85r7codec as old, release_deadline_v1 as insurance
except ImportError:
    import exp169_release_c85r7codec as old
    import release_deadline_v1 as insurance

try:
    from tools import release_metadata_v122 as metadata_io
except ImportError:
    import release_metadata_v122 as metadata_io

base, c85, p, r7, v5q, s6 = old.base, old.c85, old.p, old.r7, old.v5q, old.s6
codecs = old.codecs  # one hook implementation, imported unchanged
need, sha, read_json, write_once, json_bytes = old.need, old.sha, old.read_json, old.write_once, old.json_bytes
write_once = metadata_io.write_once
telemetry, optional_json, competition_root = old.telemetry, old.optional_json, old.competition_root
blend, reference, p_branch, authenticated_pr7 = old.blend, old.reference, old.p_branch, old.authenticated_pr7
codec_summary = old.codec_summary
ROOT, FORMULA, MEMBERS = old.ROOT, old.FORMULA, old.MEMBERS
RUN, RECEIPT, STAGE, DIAGNOSTIC, P_BRANCH = old.RUN, old.RECEIPT, old.STAGE, old.DIAGNOSTIC, old.P_BRANCH
VERSION, RELEASE = '1.2.2', 'C85R7+codecs+insurance'
MODULE = 'exp169_release_c85r7codec_deadline_v122'
OUT = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.2.2'
KERNEL_ID = 'taeyangg4/knee-exp169-c85r7ins-v122'
RUNTIME_DIR = '/kaggle/working/c85r7ins-runtime'
OLD_SHA = 'fb53f74b91b492979b73ff4d6d8756993a259ee9e84cebd702c6959a89952f42'
INSURANCE_SHA = '0f6ecc410680a136529739cf926d889ed4c9c454d9fc5a1a8b891d038c08e8d3'
SERVING_SHA = '25ef6ab9d2e942ac291433a498246d0507481c2aefa5630cdd601d0a36852cf2'
STRESS_ENV, FAULT_UID_ENV = 'C85R7_INSURANCE_STRESS', 'C85R7_INSURANCE_FAULT_UID'
MARKER_ENV, JOURNAL_ENV = 'C85R7_INSURANCE_MARKER', 'C85R7_INSURANCE_JOURNAL'
FIRST_ENV = 'C85R7_INSURANCE_FIRST_UID'
_ORIGINAL_SERVING = c85.serving_source


def replace_once(source, anchor, replacement):
    need(source.count(anchor) == 1, 'insurance source anchor: ' + anchor[:100])
    return source.replace(anchor, replacement)


def before_study(uid):
    """No-op in production. Diagnostic injection sits inside the original try.

    A target worker waits for an actual inference journal row, not merely a
    decoded image. Diagnostic ordering moves the target out of the first chunk.
    """
    mode = os.environ.get(insurance.FAULT_ENV, 'off')
    if mode == 'off':
        return
    need(os.environ.get(STRESS_ENV) == '1', 'production fault injection rejected')
    target = os.environ[FAULT_UID_ENV]
    if uid != target:
        return
    journal = Path(os.environ[JOURNAL_ENV])
    first = os.environ[FIRST_ENV]
    while True:
        if journal.exists():
            for line in metadata_io.journal_text(journal).splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get('uid') == first and len(row.get('values', [])) == 12:
                    insurance.FaultInjection(mode, target).before_study(uid, marker=os.environ[MARKER_ENV])
                    return
        time.sleep(.05)


def study_order(ids):
    if os.environ.get(insurance.FAULT_ENV, 'off') == 'off':
        return ids
    need(os.environ.get(STRESS_ENV) == '1' and len(ids) == len(set(ids)) == 32, 'diagnostic N32 only')
    target = os.environ[FAULT_UID_ENV]
    rest = [uid for uid in ids if uid != target]
    return [*rest[:2], target, *rest[2:]]


def serving_source(source):
    need(hashlib.sha256(source.encode()).hexdigest() == SERVING_SHA, 'public-serving source pin')
    source = _ORIGINAL_SERVING(source)
    source = replace_once(source, '    with mp.get_context("fork").Pool(nproc, initializer=_pre_init) as pool:',
        '    from release_deadline_v1 import Deadline, WatchdogPool\n'
        '    with WatchdogPool(nproc, initializer=_pre_init, deadline=Deadline.from_env(), context="fork") as pool:')
    source = replace_once(source, '        vol, mask = _READER(sid, _SER, _TSDIR)',
        f'        from {MODULE} import before_study\n'
        '        before_study(sid)\n        vol, mask = _READER(sid, _SER, _TSDIR)')
    source = replace_once(source, '_exp152_bounded_preprocess(pool, test_ids, _pre_one, QUEUE_MAX)',
        f'_exp152_bounded_preprocess(pool, __import__({MODULE!r}).study_order(test_ids), _pre_one, QUEUE_MAX)')
    compile(source, '<c85r7ins-serving>', 'exec')
    return metadata_io.atomic_source(source)


def install_adapters(log_dir):
    """Explicit per-child adapters for BOTH frozen detach boundaries; file capture."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    serial = 0

    def runner(args, env=None, *, label='S6 controller', capture=False):
        nonlocal serial
        serial += 1
        log = directory / f'nested-{os.getpid()}-{serial}.log'
        with log.open('wb') as stream, insurance.Supervisor(insurance.Deadline.from_env()) as supervisor:
            child = supervisor.run(args, label=label, env=env, stdout=stream, stderr=subprocess.STDOUT)
        result = dict(label=label, exit_code=child.returncode, reason=child.reason, cleanup_error=child.cleanup_error)
        telemetry({}, 'nested_process', lambda: write_once(log.with_suffix('.json'), json_bytes(result)))
        if child.returncode or child.cleanup_error:
            raise RuntimeError('supervised child: ' + json.dumps(result))
        return log.read_text(encoding='utf-8', errors='replace') if capture else None

    v5q.run_process_group = runner
    s6._run_group = runner
    c85.serving_source = serving_source
    return runner


def branch_phase(output, name, action):
    record = phase(output, name, action)
    if record['exit_code']:
        raise SystemExit(record['exit_code'])
    return record


def phase(output, name, action):
    """A failed optional phase receipt cannot prevent survivor publication."""
    start = time.monotonic()
    record = dict(phase=name, exit_code=0)
    try:
        action()
    except Exception as exc:
        record.update(exit_code=1, error=f'{type(exc).__name__}: {exc}')
    record['seconds'] = time.monotonic() - start
    telemetry(record, 'phase_written', lambda: write_once(Path(output) / (name + '-exit.json'), json_bytes(record)))
    print('C85_INSURANCE_PHASE', json.dumps(record), flush=True)
    return record


def a172_script(cell, output, branch):
    """Split the frozen combined script without altering either setup/model body."""
    source = c85.a172_source(cell, output)
    prefix, body = source.split('# No pip,', 1)
    own, partner = ('# No pip,' + body).split("_c85.phase(_out, 'own', main)\n", 1)
    prefix += f'import {MODULE} as _ins\n_ins.install_adapters(_out / "insurance-logs")\n'
    if branch == 'own':
        result = prefix + own + "_ins.branch_phase(_out, 'own', main)\n"
    else:
        need(branch == 'Q_A5q', 'unknown A172 branch')
        # Q setup previously inherited only these names from own's asset setup.
        result = prefix + 'import hashlib, json, zipfile\n'
        result += "def _hash(p):\n    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()\n"
        result += partner.replace('_c85.phase(', '_ins.branch_phase(')
    compile(result, '<c85r7ins-' + branch + '>', 'exec')
    return metadata_io.atomic_source(result)


def runtime_pins(runtime):
    old.runtime_pins(runtime)
    p.check_pin(runtime / 'exp169_release_c85r7codec.py', OLD_SHA)
    p.check_pin(runtime / 'release_deadline_v1.py', INSURANCE_SHA)


def seed(runtime, output):
    """Only cohort/manifest I/O before supervised setup; no native imports/install."""
    comp = competition_root()
    with (comp / 'sample_submission.csv').open(newline='', encoding='utf-8') as stream:
        ids = v5q.validate_ids([row[p.UID] for row in csv.DictReader(stream)])
    with (comp / 'test.csv').open(newline='', encoding='utf-8') as stream:
        test_ids = v5q.validate_ids([row[p.UID] for row in csv.DictReader(stream)])
    need(ids and set(ids) == set(test_ids), 'test/sample cohort')
    c85.manifest(runtime / 'q-a5q-manifest.json')
    for name in ('a172', 'pr7'):
        metadata_io.telemetry({}, 'seed_assignment', lambda: write_once(output / RUN / name / 'ids.json', json_bytes(ids)))
    metadata_io.telemetry({}, 'seed_manifest', lambda: write_once(output / RUN / 'a172/q-a5q-manifest.json', (runtime / 'q-a5q-manifest.json').read_bytes()))
    return ids


def run_s(runtime, output):
    work = output / RUN / 'pr7/pr7-run'
    ids = read_json(output / RUN / 'pr7/ids.json')
    identity, paths = read_json(work / 'r7-identity.json'), read_json(work / 'code-paths.json')
    code, anatomy = Path(paths['code']), Path(paths['anatomy'])
    codec = old.codec_receipt(output / RUN)
    s6.install_wheels(sorted((work / 's-code/wheels').glob('*.whl')), code)
    old.gate_s_r7(output / RUN, codec, code)
    r7.infer_studies([competition_root() / 'test_series' / uid for uid in ids],
        bundle_dir=v5q._mount(r7.DATASET_SLUG, 'manifest.json'), code_dir=code, anatomy_weights=anatomy,
        output_dir=work / 's_r7', workers=2, amp='auto', device='cuda', max_fallback_fraction=1.,
        source_dir=runtime / 'exp169_r7_src', expected_manifest_sha256=identity['manifest_sha256'])


def _child_entry(runtime, output, branch, input_root='/kaggle/input'):
    runtime_pins(runtime)
    ids = metadata_io.metadata(output / RUN / 'a172/ids.json')
    insurance.FaultInjection.cases(ids, stress=os.environ.get(STRESS_ENV) == '1')
    insurance.Deadline.from_env()
    install_adapters(output / RUN / 'insurance-logs')
    if branch == 'setup':
        try:
            need(base.preflight(runtime, output) == ids, 'setup cohort changed')
        except (OSError, metadata_io.MetadataUnavailable):
            raise  # operational metadata interruption; root retains the cohort
        except Exception as exc:
            # A returned identity/setup exception is fatal. A killed/stalled setup
            # has no success marker and no predictions, so seeded recovery is safe.
            write_once(output / RUN / 'setup-fatal.json', json_bytes(dict(error=f'{type(exc).__name__}: {exc}')))
            raise
        write_once(output / RUN / 'setup-ready.json', json_bytes(dict(status='READY')))
    elif branch == 'codecs':
        old.install_codecs(runtime, output, input_root=input_root)
    elif branch in ('own', 'Q_A5q'):
        script = a172_script((runtime / 'a172-cell.py.txt').read_text(encoding='utf-8'), output / RUN / 'a172', branch)
        exec(compile(script, '<isolated-' + branch + '>', 'exec'), dict(__name__='__main__'))
    elif branch == 'R':
        weights = v5q._mount(p.DATASETS[0], p.R_NAME) / p.R_NAME
        c85.run_r(runtime, weights, competition_root(), output / RUN / 'pr7/pr7-run/r')
    elif branch == 'S_R7':
        run_s(runtime, output)
    else:
        raise ValueError('unknown child branch')


def _run_release(runtime, output, *, input_root='/kaggle/input', stress=False, fault=None):
    clock = insurance.Deadline.from_env()
    if not stress:
        insurance.FaultInjection.cases([], stress=False)
    runtime_pins(runtime)
    need(not any((output / x).exists() for x in ('submission.csv', RUN, RECEIPT, STAGE, DIAGNOSTIC, P_BRANCH)),
         'stale C85R7 output')
    ids = seed(runtime, output)
    cases = insurance.FaultInjection.cases(ids, stress=stress)
    need(len(cases) == 1, 'run each diagnostic fault in its own output/process trees')
    fault = fault or cases[0]
    need(fault.mode == cases[0].mode and fault.uid == cases[0].uid, 'fault configuration/env mismatch')
    clock = fault.deadline(clock)
    work, adir, pdir = output / RUN, output / RUN / 'a172', output / RUN / 'pr7'
    events = []
    with insurance.Supervisor(clock) as supervisor:
        for branch in ('setup', 'codecs', 'own', 'Q_A5q', 'R', 'S_R7'):
            directory = adir if branch in ('own', 'Q_A5q') else pdir if branch in ('R', 'S_R7') else work
            env = old.child_base_env(runtime)
            env[STRESS_ENV] = '1' if stress else '0'
            codec = old.codec_receipt(work)
            env = codecs.child_env(env, codec, branch='A172:' + branch if branch in ('own', 'Q_A5q') else branch,
                                   telemetry_dir=old.codec_dirs(work)[1])
            if fault.mode != 'off':
                journal = adir / 'exp169-trio' if branch == 'own' else pdir / 'pr7-run/r'
                env.update({FAULT_UID_ENV: fault.uid, MARKER_ENV: str(work / f'{branch}-fault.marker'),
                            JOURNAL_ENV: str(journal), FIRST_ENV: ids[0]})
            def action():
                need(branch == 'setup' or metadata_io.optional_json(work / 'setup-ready.json').get('status') == 'READY',
                     'setup not completed; branch skipped')
                with (work / (branch + '.log')).open('wb') as stream:
                    child = supervisor.run([sys.executable, '-u', str(runtime / (MODULE + '.py')), 'child',
                        '--runtime', str(runtime), '--output', str(output), '--branch', branch,
                        '--input-root', str(input_root)], label=branch, env=env, stdout=stream, stderr=subprocess.STDOUT)
                event = dict(branch=branch, exit_code=child.returncode, reason=child.reason, cleanup_error=child.cleanup_error)
                events.append(event)
                if child.returncode or child.cleanup_error:
                    raise RuntimeError(json.dumps(event))
            phase(directory, branch + '-supervised', action)
            if metadata_io.optional_json(work / 'setup-fatal.json').get('error') or (branch == 'setup' and events and events[-1]['exit_code'] == 78):
                raise ValueError('fatal setup/identity rejection (setup exit 78 or fatal marker)')
    # No compute supervisor around recovery/publication; 20 min reserved headroom.
    need(not any(event['cleanup_error'] for event in events), 'process-tree cleanup failed; publication refused')
    telemetry({}, 'insurance_events', lambda: write_once(work / 'insurance-events.json', json_bytes(events)))
    receipt, _ = collect(adir, pdir, output, diagnostic=stress, cohort=ids)
    return receipt



def child_entry(*args, **kwargs):
    with metadata_io.writers(sys.modules[__name__]):
        return _child_entry(*args, **kwargs)


def run_release(*args, **kwargs):
    with metadata_io.writers(sys.modules[__name__]):
        return _run_release(*args, **kwargs)


def collect(a172_dir, pr7_dir, output, *, diagnostic=False, cohort=None):
    import numpy as np
    # Root keeps the original sample order in memory across killable child work.
    if cohort is None:
        for directory in (a172_dir, pr7_dir):
            try:
                cohort = v5q.validate_ids(metadata_io.metadata(directory / 'ids.json'))
                break
            except metadata_io.MetadataUnavailable:
                pass
        if cohort is None:  # historical replay has authenticated NPZ cohorts
            cohort = c85.authenticated_a172(a172_dir)[0]
    cohort = v5q.validate_ids(cohort)
    need(cohort, 'trusted publication cohort required')
    a, b = metadata_io.collectors(sys.modules[__name__], cohort)
    def ordered_a(directory):
        ids, x, y, q, receipt = a(directory)
        need(set(ids) == set(cohort), 'publication cohort mismatch')
        indices = [ids.index(uid) for uid in cohort]
        return cohort, x[indices], y[indices], q[indices], receipt
    # A valid but different top-level assignment is an authentication failure.
    for directory in (a172_dir, pr7_dir):
        try:
            assigned = metadata_io.metadata(directory / 'ids.json')
        except metadata_io.MetadataUnavailable:
            continue
        need(set(assigned) == set(cohort), 'root cohort mismatch')
    with metadata_io.writers(sys.modules[__name__]):
        return _collect(a172_dir, pr7_dir, output, diagnostic=diagnostic, _a172=ordered_a, _pr7=b)


def candidate_receipt(ids, a, b, payload, codec):
    receipt = old.candidate_receipt(ids, a, b, payload, codec)
    receipt.update(version=VERSION, release=RELEASE, insurance=dict(module_sha256=INSURANCE_SHA,
        default_deadline_seconds=31200, default_pool_stall_seconds=1200, publication_after_cleanup=True))
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

def _collect(a172_dir, pr7_dir, output, *, diagnostic=False, _a172=None, _pr7=None):
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
        ids, exp143, exp155, q, a = _a172(a172_dir)
        completed.append("A172")
        pids, r, ours, b = _pr7(pr7_dir)
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
        try:
            write_once(staged, receipt_payload)
            c85.publish(staged, record, receipt_payload)
        except OSError as exc:
            # A committed, verified CSV survives optional receipt I/O failure.
            receipt['telemetry']['receipt_write_error'] = type(exc).__name__
        return receipt, payload
    except BaseException as error:
        if not repeat:
            failed_execution(output, error, completed, a172_dir, pr7_dir)
        raise


def source_payloads():
    files = old.source_payloads()
    p.check_pin(Path(old.__file__), OLD_SHA)
    p.check_pin(Path(insurance.__file__), INSURANCE_SHA)
    serving_source(files['public-serving.py'].decode())
    for branch in ('own', 'Q_A5q'):
        a172_script(files['a172-cell.py.txt'].decode(), Path('/kaggle/working/check'), branch)
    files.update({MODULE + '.py': Path(__file__).read_bytes(),
        'release_deadline_v1.py': Path(insurance.__file__).read_bytes(),
        'release_metadata_v122.py': Path(metadata_io.__file__).read_bytes(),
        'release_cleanup_checks_v121.py': Path(__file__).with_name('release_cleanup_checks_v121.py').read_bytes(),
        'exp169_c85r7ins_checks_v122.py': Path(__file__).with_name('exp169_c85r7ins_checks_v122.py').read_bytes()})
    return files


def package():
    files, payload = source_payloads(), io.BytesIO()
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            archive.writestr(info, data)
    return payload.getvalue(), {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}


def notebook_payload(blob, pins, kind='kernel'):
    nb = old.notebook_payload(blob, pins)
    code = ''.join(nb['cells'][-1]['source'])
    code = ('import os, time\n'
            'os.environ.setdefault("KNEE_RELEASE_START_MONOTONIC_V1", repr(time.monotonic()))\n'
            'os.environ.setdefault("KNEE_RELEASE_DEADLINE_SECONDS_V1", "31200")\n' + code)
    code = code.replace(old.RUNTIME_DIR, RUNTIME_DIR).replace('import exp169_release_c85r7codec as release',
                                                            f'import {MODULE} as release')
    code = code.replace('C85R7+codecs v1.1.0', 'C85R7+codecs+insurance v1.2.2')
    if kind != 'kernel':
        need(kind in ('stress-kernel', 'stress-kernel-clean'), 'unknown kernel kind')
        code = replace_once(code, "release.run_release(runtime, Path('/kaggle/working'))",
            'import exp169_c85r7ins_checks_v122 as checks\n'
            f'os.environ[release.insurance.FAULT_ENV] = {"all" if kind == "stress-kernel" else "off"!r}\n'
            f'checks.target_stress(runtime, Path("/kaggle/working"), clean_only={kind == "stress-kernel-clean"!r})')
    compile(code, '<c85r7ins-notebook>', 'exec')
    nb['cells'][-1]['source'] = code.splitlines(True)
    credit = ''.join(nb['cells'][0]['source']).replace('v1.1.0', 'v1.2.2')
    credit = credit.replace('No weight search, no timeout; hidden runtime unknown.',
        'Cumulative 8 h 40 m compute deadline; original collector publishes after process-tree cleanup. '
        'Own/R fork-pool watchdog: 20 m without progress. Frozen v1.1.0 healthy prediction math.')
    if kind != 'kernel':
        credit += '\nDIAGNOSTIC N32 ONLY, NEVER SUBMIT. Writes diagnostic CSVs only. '
        credit += 'Fault injection ON: separate decode, kill, hang and deadline cases for own and R.\n' if kind == 'stress-kernel' else 'Fault injection OFF.\n'
    nb['cells'][0]['source'] = [credit]
    return nb


def metadata(kind='kernel'):
    value = old.metadata()
    suffix = {'kernel': '', 'stress-kernel': '-stress-n32', 'stress-kernel-clean': '-clean-n32'}[kind]
    value.update(id=KERNEL_ID + suffix, title=KERNEL_ID.split('/')[1] + suffix)
    return value


def stage_payloads():
    blob, pins = package()
    files = {}
    for kind in ('kernel', 'stress-kernel', 'stress-kernel-clean'):
        files[kind + '/release.ipynb'] = json_bytes(notebook_payload(blob, pins, kind))
        files[kind + '/kernel-metadata.json'] = json_bytes(metadata(kind))
    return blob, pins, files


def prepare(out=OUT):
    blob, pins, files = stage_payloads()
    for name, data in files.items():
        write_once(out / name, data)
    record = dict(status='PREPARED_LOCAL_ONLY', version=VERSION, release=RELEASE, kernel_id=KERNEL_ID,
        formula=FORMULA, files=pins, archive_sha256=hashlib.sha256(blob).hexdigest(),
        artifacts={k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
        submitted_base_ref=57133175, base_source_sha256=OLD_SHA, insurance_sha256=INSURANCE_SHA,
        qualification='HOLD: Windows CPU evidence only; Linux fork/GDCM/process groups and B decision pending')
    write_once(out / 'build-receipt.json', json_bytes(record))
    ident = read_json(old.OUT / 'input-identities.json')
    ident.update(source_pins=pins, submitted_base_ref=57133175,
        insurance=dict(module_sha256=INSURANCE_SHA, default_compute_seconds=31200,
                       publication_reserve_seconds=1200, pool_stall_seconds=1200))
    write_once(out / 'input-identities.json', json_bytes(ident))
    return record


def validate(out=OUT):
    b = read_json(out / 'build-receipt.json')
    blob, pins, files = stage_payloads()
    need(b['files'] == pins and b['archive_sha256'] == hashlib.sha256(blob).hexdigest(), 'source archive drift')
    for name, data in files.items():
        need((out / name).read_bytes() == data and sha(out / name) == b['artifacts'][name], 'stage drift: ' + name)
    for kind in ('kernel', 'stress-kernel', 'stress-kernel-clean'):
        meta = read_json(out / kind / 'kernel-metadata.json')
        need({k: v for k, v in meta.items() if k not in ('id', 'title')} ==
             {k: v for k, v in old.metadata().items() if k not in ('id', 'title')}, 'metadata policy drift')
    need(read_json(out / 'input-identities.json')['source_pins'] == pins, 'identity source drift')
    return dict(status='PASS_CPU_VALIDATE', version=VERSION, files=len(pins),
                notebook_sha256=b['artifacts']['kernel/release.ipynb'], qualified_for_submission=False)


def register(out=OUT):
    record = dict(validate(out), formula=FORMULA, kernel_id=KERNEL_ID, base_ref=57133175)
    record['status'] = 'REGISTERED_LOCAL_ONLY'
    write_once(out / 'registration.json', json_bytes(record))
    return record


def cpu_replay(output):
    staged = base.stage_replay_input(output / 'replay-input/pr7')
    receipt, payload = collect(c85.A5Q, staged, output, diagnostic=True)
    _, repeated = collect(c85.A5Q, staged, output, diagnostic=True)
    prior = output / 'v110'
    old_receipt, old_payload = old.collect(c85.A5Q, staged, prior, diagnostic=True)
    need(payload == repeated == old_payload, 'v1.1.0 CSV byte parity')
    need((output / P_BRANCH).read_bytes() == (prior / P_BRANCH).read_bytes(), 'v1.1.0 P byte parity')
    need(receipt['telemetry']['healthy_reference_exact'], 'independent formula differs')
    diff = old.receipt_diff(receipt, old_receipt)
    need(set(diff) <= {'version', 'release', 'insurance'}, 'non-metadata receipt drift')
    result = dict(status='PASS_CPU_ARITHMETIC_REPLAY', studies=receipt['studies'], version=VERSION,
        independent_formula_exact=True, deterministic_csv=True, v110_csv_identical=True,
        v110_p_branch_identical=True, receipt_diff=diff, diagnostic_sha256=hashlib.sha256(payload).hexdigest(),
        scope='recorded R/own/Q predictions with S6 stand-in; no R7 inference', qualified_for_submission=False)
    write_once(output / 'replay.json', json_bytes(result))
    return result


def admit(out=OUT, replay_dir=None):
    val, reg = validate(out), read_json(out / 'registration.json')
    need(val['notebook_sha256'] == reg['notebook_sha256'], 'registration drift')
    replay_dir = replay_dir or out / 'cpu-replay'
    replay = read_json(replay_dir / 'replay.json')
    need(replay['status'] == 'PASS_CPU_ARITHMETIC_REPLAY' and all(replay[k] for k in
        ('independent_formula_exact', 'deterministic_csv', 'v110_csv_identical', 'v110_p_branch_identical')), 'replay gate')
    p.check_pin(replay_dir / DIAGNOSTIC, replay['diagnostic_sha256'])
    return dict(status='HOLD_NOT_QUALIFIED', cpu_preparation='PASS', qualified_for_submission=False,
        remaining=['Pinned Linux/CPython fork+GDCM N32 fault and nested process-group qualification',
                   'Healthy target CSV byte identity to submitted v1.1.0; hidden runtime',
                   'Independent wrapper review and B submission/gain/duplicate decision'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'validate', 'register', 'admit'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--out', type=Path, default=OUT)
        if name == 'admit':
            cmd.add_argument('--dry-run', required=True, action='store_true')
            cmd.add_argument('--replay-dir', type=Path)
    cmd = commands.add_parser('cpu-replay')
    cmd.add_argument('--output', type=Path, required=True)
    for name in ('child', 'run'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--runtime', type=Path, required=True)
        cmd.add_argument('--output', type=Path, required=True)
        cmd.add_argument('--input-root', default='/kaggle/input')
        if name == 'child':
            cmd.add_argument('--branch', choices=['setup', 'codecs', 'own', 'Q_A5q', 'R', 'S_R7'], required=True)
    args = parser.parse_args()
    if args.command == 'child':
        # A failed fork pool can leave Python/GPU consumer threads alive. Never
        # enter interpreter thread joins; the outer Supervisor cleans descendants.
        import traceback
        code = 0
        try:
            child_entry(args.runtime, args.output, args.branch, args.input_root)
        except BaseException as exc:
            traceback.print_exc()
            # Preserve fatal setup/identity rejection even if its marker cannot
            # be written. Operational timeout/kill remains Supervisor-owned.
            code = 78 if args.branch == 'setup' and not isinstance(exc, (OSError, metadata_io.MetadataUnavailable)) else 1
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(code)
    if args.command == 'run':
        result = run_release(args.runtime, args.output, input_root=args.input_root)
    elif args.command == 'cpu-replay':
        result = cpu_replay(args.output)
    elif args.command == 'admit':
        result = admit(args.out, args.replay_dir)
    else:
        result = globals()[args.command](args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    # Generated own code and fork decoders import this canonical name. Reuse the
    # CLI module so an import after install_adapters cannot capture its already
    # patched serving_source as the supposedly original implementation.
    sys.modules[MODULE] = sys.modules[__name__]
    main()
