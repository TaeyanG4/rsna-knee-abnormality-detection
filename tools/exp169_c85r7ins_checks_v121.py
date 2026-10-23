"""C85R7 v1.2.1 CPU replay and diagnostic-only N32 insurance checks.

CPU pools execute the actual transformed decoder and batching AST with recorded
scores and optional real DICOM pixel decoding. Target stress uses actual own/R
inference and the original journals, never a replacement prediction policy.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
try:
    from tools import exp169_release_c85r7codec_deadline_v121 as m
except ImportError:
    import exp169_release_c85r7codec_deadline_v121 as m
d, c = m.insurance, m.c85
VISIBLE = m.ROOT / '.local/exp189-cpu-v120/input/visible_data'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    m.metadata_io.atomic_write(path, m.json_bytes(value))


def snapshot(adir, pdir):
    if (adir / 'ids.json').exists():
        cohort = m.read_json(adir / 'ids.json')
        a172, pr7 = m.metadata_io.collectors(m, cohort)
    else:
        a172, pr7 = c.authenticated_a172, m.authenticated_pr7
    ids, a, b, q, ar = a172(adir)
    pids, r, s, pr = pr7(pdir)
    order = [pids.index(u) for u in ids]
    return ids, dict(R=r[order], S_R7=s[order], EXP143v2=a, EXP155=b, Q_A5q=q), {**ar['failures'], **pr['failures']}


def journal(directory, ids, branch):
    identity = (dict(checkpoint_sha256=m.p.R_SHA, source_pins=m.p.PINS) if branch == 'R'
                else c.manifest(directory.parent / 'q-a5q-manifest.json')['anchor_binding'])
    with m.metadata_io.writers(m):
        return c.Journal(directory, ids, identity)


def journal_path(output, branch):
    return output / m.RUN / ('pr7/pr7-run/r' if branch == 'R' else 'a172/exp169-trio')


def members(branch):
    return ['R'] if branch == 'R' else ['EXP143v2', 'EXP155']


def copy_inputs(source, output, exclude=None):
    """Whitelist small authenticated predictions only, never models/caches."""
    adir, pdir = source / m.RUN / 'a172', source / m.RUN / 'pr7'
    fixed = ['a172/ids.json', 'a172/q-a5q-manifest.json', 'pr7/ids.json',
             'pr7/pr7-run/r7-identity.json', 'pr7/pr7-run/s_r7/provenance.json']
    for group, files in [('own', ['a172/exp169-trio/ids.json', 'a172/exp169-trio/journal-identity.json',
                                 'a172/exp169-trio/predictions.jsonl']),
                         ('R', ['pr7/pr7-run/r/ids.json', 'pr7/pr7-run/r/journal-identity.json',
                                'pr7/pr7-run/r/predictions.jsonl'])]:
        if group != exclude:
            fixed += files
    qjournal = adir / 'q-journal'
    if (qjournal / 'journal-identity.json').exists():
        fixed += ['a172/q-journal/' + x for x in ('ids.json', 'journal-identity.json', 'predictions.jsonl')]
        fixed += [row.relative_to(source / m.RUN).as_posix() for row in sorted((qjournal / 'journal-rows').glob('*.json'))]
    else:
        fixed += ['a172/partner-raw.npz']
    for worker in sorted((pdir / 'pr7-run/s_r7/controller').glob('worker*')):
        for name in ('ids.json', 'predictions.csv', 'receipt.json'):
            fixed.append('pr7/' + (worker / name).relative_to(pdir).as_posix())
    for branch in ('own', 'R'):
        if branch != exclude:
            j = journal_path(source, branch) / 'journal-rows'
            for row in sorted(j.glob('*.json')):
                fixed.append(row.relative_to(source / m.RUN).as_posix())
    for name in fixed:
        src = source / m.RUN / name
        if src.exists():
            m.write_once(output / m.RUN / name, src.read_bytes())


def fixture(output, ids):
    """Explicit N32 fixture repeats authenticated three-study vectors; S6 stand-in."""
    import numpy as np
    pdir = m.old.OUT / 'cpu-replay/replay-input/pr7'
    original, raw, failures = snapshot(c.A5Q, pdir)
    assert not any(failures.values())
    raw = {k: np.stack([v[i % len(original)] for i in range(len(ids))]) for k, v in raw.items()}
    adir, pdir2 = output / m.RUN / 'a172', output / m.RUN / 'pr7'
    for folder in (adir, pdir2):
        m.write_once(folder / 'ids.json', m.json_bytes(ids))
    manifest = c.A5Q / 'q-a5q-manifest.json'
    if not manifest.exists():
        manifest = c.A5Q / 'exp169-partner/release-manifest.json'
    m.write_once(adir / 'q-a5q-manifest.json', manifest.read_bytes())
    # Q fixture retains per-model probabilities, not repeated aggregate ranks.
    with np.load(c.A5Q / 'partner-raw.npz', allow_pickle=False) as z:
        qi = z['ids'].tolist()
        indices = [qi.index(original[i % len(original)]) for i in range(len(ids))]
        np.savez(adir / 'partner-raw.npz', ids=np.asarray(ids),
                 **{k: z[k][:, indices, :] for k in ('l2', 'dino', 'convnext')})
    for branch in ('own', 'R'):
        j = journal(journal_path(output, branch), ids, branch)
        for i, uid in enumerate(ids):
            for member in members(branch):
                j.record(uid, member, raw[member][i])
    for name in ('r7-identity.json', 's_r7/provenance.json'):
        m.write_once(pdir2 / 'pr7-run' / name, (pdir / 'pr7-run' / name).read_bytes())
    w = pdir2 / 'pr7-run/s_r7/controller/worker0'
    m.write_once(w / 'ids.json', m.json_bytes(ids))
    m.write_once(w / 'predictions.csv', m.p.csv_bytes(ids, raw['S_R7']))
    return snapshot(adir, pdir2)


def _reader(uid, series, directory):
    import numpy as np
    if directory:
        import pydicom
        assert pydicom.dcmread(directory).pixel_array.size > 0
    return np.zeros((1, 2, 2), np.uint8), np.ones(1, np.uint8)


def _initialize(source, dicom):
    import numpy as np
    global _decoder
    # Spawn imports __main__ by a different name; the dynamic function remains
    # worker-local and is called by the stable top-level _decode task.
    sys.path.insert(0, str(Path(m.__file__).parent))
    ns = dict(np=np, _READER=_reader, _SER=None, _TSDIR=dicom, MAXS=1, IMG=2)
    tree = ast.parse(source)
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_pre_one')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-patched-decoder>', 'exec'), ns)
    _decoder = ns['_pre_one']


def _decode(uid):
    return _decoder(uid)


def pool_child(config):
    import numpy as np
    spec = m.read_json(Path(config))
    ids, branch = spec['ids'], spec['branch']
    source = m.serving_source(Path(spec['source']).read_text(encoding='utf-8'))
    tree = ast.parse(source)
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_exp152_bounded_preprocess')
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-batching>', 'exec'), ns)
    j = journal(Path(spec['journal']), ids, branch)
    with np.load(spec['raw'], allow_pickle=False) as z:
        raw = {k: z[k].copy() for k in members(branch)}
    reason = 'complete'
    with d.WatchdogPool(1, initializer=_initialize, initargs=(source, spec.get('dicom')),
                       deadline=d.Deadline.from_env(), stall_seconds=spec['stall'], poll_seconds=.01,
                       context='spawn' if os.name == 'nt' else 'fork') as pool:
        try:
            for uid, vol, mask, error in ns['_exp152_bounded_preprocess'](pool, m.study_order(ids), _decode, 4):
                for member in members(branch):
                    j.record(uid, member, raw[member][ids.index(uid)] if error is None else None, error=error)
        except d.PoolAborted as exc:
            reason = str(exc)
    dump(Path(spec['result']), dict(reason=reason, workers_stopped=all(not p.is_alive() for p in pool._workers)))


def recovered_check(baseline, output, branch, fault, marker):
    import numpy as np
    ids, expected, _ = baseline
    got_ids, raw, fails = snapshot(output / m.RUN / 'a172', output / m.RUN / 'pr7')
    assert ids == got_ids
    expected = {k: a.copy() for k, a in expected.items()}
    for member in m.MEMBERS:
        for i, uid in enumerate(ids):
            if uid in fails[member]:
                assert member in members(branch), (member, uid)
                expected[member][i] = .5
            else:
                np.testing.assert_array_equal(raw[member][i], expected[member][i])
    oracle = m.p.csv_bytes(ids, m.old.blend(ids, *(expected[k] for k in m.MEMBERS), fails))
    receipt, payload = m.collect(output / m.RUN / 'a172', output / m.RUN / 'pr7', output, diagnostic=True)
    prior_payload = legacy_collect(output, output / 'v110')[1]
    assert payload == prior_payload == oracle
    assert not (output / 'submission.csv').exists() and not receipt['qualified_for_submission']
    for member in members(branch):
        fault.assert_csv(ids, payload, oracle, marker=marker, failed_ids=fails[member])
        assert ids[0] not in fails[member], 'no complete inference survivor'
        if fault.mode == 'off':
            assert not fails[member]
        elif fault.mode == 'decode':
            assert set(fails[member]) == {fault.uid}
    return dict(branch=branch, fault=fault.mode, studies=len(ids),
                failed={k: len(v) for k, v in fails.items()}, csv_sha256=hashlib.sha256(payload).hexdigest(),
                original_collector_identical=True, original_fallback_identical=True)


def legacy_collect(source, output):
    """Export atomic rows to a separate legacy fixture; never edit runtime evidence."""
    target = output / 'input'
    copy_inputs(source, target)
    for directory in (target / m.RUN).rglob('journal-rows'):
        m.metadata_io.atomic_text(directory.parent / 'predictions.jsonl', m.metadata_io.journal_text(directory.parent))
    return m.old.collect(target / m.RUN / 'a172', target / m.RUN / 'pr7', output, diagnostic=True)


def cpu_smoke(output, real_decode=True):
    import numpy as np
    ids = [f'study{i:02d}' for i in range(32)]
    baseline = fixture(output / 'baseline', ids)
    source = m.old.source_payloads()['public-serving.py']
    m.write_once(output / 'public-serving.py', source)
    np.savez(output / 'raw.npz', **baseline[1])
    dicom = str(next((VISIBLE / 'test_series').rglob('*.dcm'))) if real_decode else None
    results = []
    for branch in ('own', 'R'):
        for fault in [d.FaultInjection(), *d.FaultInjection.cases(ids, stress=True, env={d.FAULT_ENV:'all'}, early_seconds=5.)]:
            work = output / branch / fault.mode
            copy_inputs(output / 'baseline', work, branch)
            path = journal_path(work, branch)
            spec = dict(ids=ids, branch=branch, source=str(output / 'public-serving.py'), raw=str(output / 'raw.npz'),
                        journal=str(path), result=str(work / 'result.json'), dicom=dicom,
                        stall=30 if fault.mode == 'deadline' else 2.)
            dump(work / 'config.json', spec)
            env = dict(os.environ, **{d.FAULT_ENV:fault.mode, m.STRESS_ENV:'1', m.FAULT_UID_ENV:ids[1],
                       m.JOURNAL_ENV:str(path), m.FIRST_ENV:ids[0], m.MARKER_ENV:str(work / 'fault.marker')})
            clock = fault.deadline(d.Deadline(time.monotonic(), 30))
            with (work / 'child.log').open('wb') as log, d.Supervisor(clock, poll_seconds=.01) as supervisor:
                child = supervisor.run([sys.executable, '-B', str(Path(__file__).resolve()), 'pool-child',
                                        '--output', str(work / 'config.json')], label=branch, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            assert child.cleanup_error is None
            result = recovered_check(baseline, work, branch, fault, work / 'fault.marker')
            rec = m.optional_json(work / 'result.json')
            if fault.mode in ('kill', 'hang'):
                assert ('pool_worker_exit' if fault.mode == 'kill' else 'pool_no_result') in rec['reason'], rec
            elif fault.mode == 'deadline':
                assert clock.expired and child.reason.startswith('deadline'), child
            else:
                assert child.returncode == 0 and rec['reason'] == 'complete', rec
            if rec:
                assert rec['workers_stopped']
            results.append(dict(result, process_reason=child.reason, pool_reason=rec.get('reason'), cleanup_error=child.cleanup_error))
    result = dict(status='PASS', context='spawn' if os.name == 'nt' else 'fork', real_dicom_decode=real_decode,
                  scope='actual patched decoder/batching AST; original journals/collector; recorded repeated vectors; S6 stand-in', cases=results)
    dump(output / 'summary.json', result)
    return result


def target_stress(runtime, output, clean_only=False):
    """Never submit. One clean N32 baseline, then isolated real own/R fault cases."""
    try:
        from tools import exp169_c85_stress as selector
        from tools.release_cleanup_checks_v121 import CleanupAudit, publication_anchor
    except ImportError:
        import exp169_c85_stress as selector
        from release_cleanup_checks_v121 import CleanupAudit, publication_anchor
    clock = d.Deadline.from_env()
    root = output / 'insurance-stress'
    ids = selector.prepare(m.competition_root(), root / 'competition')
    cases = d.FaultInjection.cases(ids, stress=True, early_seconds=900.)
    if not clean_only:
        assert [f.mode for f in cases] == ['decode','kill','hang','deadline'], 'fault kernel must enable all cases'
    os.environ.update(C85_COMPETITION=str(root / 'competition'), **{m.STRESS_ENV:'1', d.FAULT_ENV:'off'})
    clean = root / 'clean'
    publication = {}
    original_collect = m.collect
    def timed_collect(*args, **kwargs):
        publication['started'] = time.monotonic()
        return original_collect(*args, **kwargs)
    m.collect = timed_collect
    try:
        with CleanupAudit() as audit:
            m.run_release(runtime, clean, stress=True)
            clean_cleanup = audit.verify(min(clock.started + clock.seconds, publication['started']))
    finally:
        m.collect = original_collect
    baseline = snapshot(clean / m.RUN / 'a172', clean / m.RUN / 'pr7')
    assert not any(baseline[2].values()), 'clean N32 must score every member'
    healthy = (clean / m.DIAGNOSTIC).read_bytes()
    _, prior = legacy_collect(clean, root / 'v110')
    assert healthy == prior
    results = []
    if not clean_only:
        for branch in ('own', 'R'):
            for fault in cases:
                work = root / branch / fault.mode
                copy_inputs(clean, work, branch)
                # Match the baseline codec hook; never reinstall into a new image.
                codec = m.old.codec_receipt(clean / m.RUN)
                env = m.codecs.child_env(m.old.child_base_env(runtime), codec, branch='stress:' + branch,
                                        telemetry_dir=work / 'codec-telemetry')
                env.update({d.FAULT_ENV:fault.mode, m.STRESS_ENV:'1', m.FAULT_UID_ENV:fault.uid,
                            m.JOURNAL_ENV:str(journal_path(work, branch)), m.FIRST_ENV:ids[0],
                            m.MARKER_ENV:str(work / 'fault.marker'),
                            d.STALL_ENV:'1200' if fault.mode == 'deadline' else '120'})
                case_clock = fault.deadline(clock)
                with CleanupAudit() as audit:
                    with (work / 'child.log').open('wb') as log, d.Supervisor(case_clock) as supervisor:
                        child = supervisor.run([sys.executable, '-u', str(runtime / (m.MODULE + '.py')), 'child',
                            '--runtime', str(runtime), '--output', str(work), '--branch', branch], label=branch,
                            env=env, stdout=log, stderr=subprocess.STDOUT)
                    assert child.cleanup_error is None
                    anchor = publication_anchor(case_clock, work / 'fault.marker')
                    result = recovered_check(baseline, work, branch, fault, work / 'fault.marker')
                    cleanup = audit.verify(anchor)
                logtext = (work / 'child.log').read_text(errors='replace')
                if fault.mode in ('kill', 'hang'):
                    assert ('pool_worker_exit' if fault.mode == 'kill' else 'pool_no_result') in logtext
                elif fault.mode == 'deadline':
                    assert child.reason.startswith('deadline')
                else:
                    assert child.returncode == 0
                results.append(dict(result, process_reason=child.reason, cleanup_error=child.cleanup_error,
                                    cleanup_verification=cleanup))
    assert not (output / 'submission.csv').exists()
    result = dict(status='PASS_DIAGNOSTIC_ONLY', qualified_for_submission=False, cases=results,
                  clean_csv_sha256=hashlib.sha256(healthy).hexdigest(), same_input_v110_collector_identical=True,
                  clean_cleanup_verification=clean_cleanup,
                  remaining_target_gates=['Nested Q/R7 aborts including native-worker/queue failure',
                    'Live interruption at metadata commit boundaries',
                    'Full-cohort cleanup and publication <=1200 seconds',
                    'Healthy actual-inference final/P byte equality with v1.1.0'],
                  note='Full healthy inference rerun vs submitted v1.1.0 remains a separate gate.')
    dump(root / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['pool-child', 'cpu-smoke'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'pool-child':
        pool_child(args.output)
    else:
        print(json.dumps(cpu_smoke(args.output), indent=2))


if __name__ == '__main__':
    main()
