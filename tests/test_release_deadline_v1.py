"""H-B-129 CPU scheduling tests and recorded-output N32 stress harness.

Direct smoke (no stages): python -B tests/test_release_deadline_v1.py smoke --out
.local/scratch/sol-4/hb129/smoke. Fault env defaults off; set
KNEE_RELEASE_FAULT_V1=all for four isolated stress cases on BOTH release lines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import release_deadline_v1 as d


def echo(value):
    return value


def hang(value):
    time.sleep(60)


def crash(value=None):
    os._exit(87)


def task_error(value):
    raise RuntimeError("decoder escaped its study handler")


def deadline(seconds=20):
    return d.Deadline(time.monotonic(), seconds)


def command(*args):
    return [sys.executable, "-B", str(Path(__file__).resolve()), *map(str, args)]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_cumulative_start_inheritance(monkeypatch):
    import pytest
    env = {d.BUDGET_ENV: "12"}
    a = d.Deadline.start(env=env)
    time.sleep(.02)
    b = d.Deadline.start(env=env)
    assert a == b and b.remaining < 12
    assert d.Deadline.from_env(a.environment()) == a
    assert d.DEFAULT_DEADLINE_SECONDS == 31200 and d.DEFAULT_STALL_SECONDS == 1200
    with pytest.raises(ValueError, match="reset"):
        d.Deadline.start(13, env=env)
    with pytest.raises(KeyError):
        d.Deadline.from_env({})
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            deadline(invalid)
    old = d.Deadline(time.monotonic() - 10, 30)
    short = d.FaultInjection("deadline", "u", .5).deadline(old)
    assert short.started == old.started and 0 < short.remaining <= .5
    assert d.FaultInjection("deadline", "u", 100).deadline(old) == old


def test_supervisor_inherits_budget_and_skips_expired(tmp_path):
    output = tmp_path / "child.json"
    clock = deadline(5)
    with d.Supervisor(clock, poll_seconds=.01) as supervisor:
        child = supervisor.run(command("env", output), label="environment")
        assert child.returncode == 0 and child.reason == "completed"
    assert d.Deadline.from_env(json.loads(output.read_text())) == clock
    with d.Supervisor(d.Deadline(time.monotonic()-2, 1)) as supervisor:
        child = supervisor.start(["never-a-real-executable"], label="skipped")
        assert child.process is None and child.returncode == 124


def test_supervisor_concurrent_and_sequential_use_same_deadline(tmp_path):
    clock = deadline(.7)
    started = time.monotonic()
    with d.Supervisor(clock, poll_seconds=.01) as supervisor:
        one = supervisor.start([sys.executable, "-c", "import time;time.sleep(60)"], label="one")
        two = supervisor.start([sys.executable, "-c", "import time;time.sleep(60)"], label="two")
        assert supervisor.wait(one).returncode == 124
        assert supervisor.wait(two).returncode == 124
        three = supervisor.run(["not-executed"], label="three")
        assert three.process is None
    assert time.monotonic() - started < 5
    assert all(c.process.poll() is not None and c.cleanup_error is None for c in (one, two))


def test_supervisor_tree_cleanup_including_orphan(tmp_path):
    # Heartbeats prove grandchildren stop even after their parent exits normally.
    for mode in ("running", "orphan"):
        work = tmp_path / mode
        work.mkdir()
        with d.Supervisor(deadline(1.7), poll_seconds=.01) as supervisor:
            child = supervisor.run(command("tree", work, mode), label=mode)
        assert child.returncode == (0 if mode == "orphan" else 124)
        assert child.cleanup_error is None
        pulse = work / "pulse"
        assert pulse.exists(), "grandchild never started"
        first = pulse.read_bytes()
        time.sleep(.15)
        assert pulse.read_bytes() == first


def test_supervisor_errors_and_cleanup_on_exception(tmp_path):
    import pytest
    with d.Supervisor(deadline()) as supervisor:
        with pytest.raises(ValueError, match="PIPE"):
            supervisor.start([sys.executable], label="pipe", stdout=subprocess.PIPE)
        for option in ({"start_new_session": True}, {"process_group": 0}):
            with pytest.raises(ValueError, match="owns"):
                supervisor.start([sys.executable], label="session", **option)
        with pytest.raises(OSError):
            supervisor.start([str(tmp_path / "absent")], label="launch-failed")
        bad = supervisor.run([sys.executable, "-c", "raise SystemExit(7)"], label="exit")
        assert bad.returncode == 7 and bad.reason == "child_exit"
        running = supervisor.start([sys.executable, "-c", "import time;time.sleep(60)"], label="cleanup")
    assert running.process.poll() is not None
    with pytest.raises(RuntimeError, match="closed"):
        supervisor.start([], label="late")


def test_pool_chunksize_two_reuse_and_slow_consumer():
    with d.WatchdogPool(1, deadline=deadline(), stall_seconds=2, poll_seconds=.01) as pool:
        assert sorted(pool.imap_unordered(echo, range(7), chunksize=2)) == list(range(7))
        pool.stall = .08
        for result in pool.imap_unordered(echo, range(4), chunksize=2):
            time.sleep(.12)  # queued results must not count as a stalled decoder
        assert result == 3
    assert all(not w.is_alive() for w in pool._workers)


def test_pool_kill_hang_deadline_and_task_exception():
    import pytest
    for fn, seconds, stall, reason in (
        (crash, 10, 5, "pool_worker_exit"),
        (hang, 10, .5, "pool_no_result"),
        (hang, .8, 5, "deadline"),
        (task_error, 10, 5, "pool_task_error"),
    ):
        started = time.monotonic()
        with pytest.raises(d.PoolAborted, match=reason):
            with d.WatchdogPool(1, deadline=deadline(seconds), stall_seconds=stall, poll_seconds=.01) as pool:
                list(pool.imap_unordered(fn, range(4), chunksize=2))
        assert time.monotonic() - started < 6
        assert all(not w.is_alive() for w in pool._workers)


def test_pool_fast_initializer_death_retained():
    import pytest
    with pytest.raises(d.PoolAborted, match="pool_worker_exit"):
        with d.WatchdogPool(1, initializer=crash, deadline=deadline(), stall_seconds=5, poll_seconds=.01) as pool:
            list(pool.imap_unordered(echo, [1, 2], chunksize=2))


def test_pool_abandoned_iteration_and_expired_start():
    import pytest
    with d.WatchdogPool(1, deadline=deadline(), stall_seconds=5) as pool:
        it = pool.imap_unordered(hang, [1, 2], chunksize=2)
        with pytest.raises(ValueError, match="one active"):
            pool.imap_unordered(echo, [3])
    with pytest.raises(d.PoolAborted):
        next(it)
    with pytest.raises(d.PoolAborted, match="before_pool"):
        d.WatchdogPool(1, deadline=d.Deadline(time.monotonic()-2, 1))


def test_atomic_complete_rows_identity_and_corruption(tmp_path):
    import numpy as np
    import pytest
    ids, members = ["a", "b", "c"], ["R", "S_R7"]
    j = d.CompleteRows(tmp_path / "rows", ids, members, {"pin": "abc"}, create=True)
    row = np.linspace(.1, .9, 12, dtype=np.float32)
    j.commit("a", "R", row)
    (j.directory / "m000-s000001.part").write_bytes(b"interrupted")
    raw, failures = j.recovered()
    assert raw['R'].dtype == row.dtype
    np.testing.assert_array_equal(raw['R'][0], row)
    assert np.isnan(raw['R'][1:]).all() and set(failures['R']) == {'b', 'c'}
    c85, masks = j.c85_inputs()
    assert (c85['R'][1:] == .5).all() and masks == failures
    for bad in (row[:11], np.full(12, np.nan), np.full(12, 2.), np.ones(12, np.int64)):
        with pytest.raises(ValueError, match="complete"):
            j.commit("b", "R", bad)
    with pytest.raises(ValueError, match="duplicate"):
        j.commit("a", "R", row)
    with pytest.raises(ValueError, match="identity"):
        d.CompleteRows(j.directory, ids, members, {"pin": "bad"})
    with pytest.raises(ValueError, match="identity"):
        d.CompleteRows(j.directory, ids[::-1], members, {"pin": "abc"})
    (j.directory / "m000-s000001.npy").write_bytes(b"bad committed file")
    with pytest.raises(ValueError):
        j.recovered()


def test_fault_gate():
    import pytest
    ids = [f"s{i}" for i in range(32)]
    env = {d.FAULT_ENV: "all"}
    assert [x.mode for x in d.FaultInjection.cases(ids, stress=True, env=env)] == ['decode', 'kill', 'hang', 'deadline']
    for cohort, stress in ((ids, False), (ids[:31], True), (["x"]*32, True)):
        with pytest.raises(ValueError, match="N32"):
            d.FaultInjection.cases(cohort, stress=stress, env=env)
    with pytest.raises(ValueError, match="unknown"):
        d.FaultInjection.cases(ids, stress=True, env={d.FAULT_ENV: "oops"})
    assert d.FaultInjection.cases([], stress=False, env={}) == [d.FaultInjection()]


def _decode(task):
    uid, values, fault, marker, dicom = task
    try:
        fault.before_study(uid, marker=marker)
        if dicom:
            import pydicom
            # Pixel data only. Never read/print reports, metadata or label rows.
            assert pydicom.dcmread(dicom).pixel_array.size > 0
        return uid, values
    except d.InjectedDecodeFailure:
        return uid, None


def pool_child(config_path):
    import numpy as np
    spec = json.loads(Path(config_path).read_text())
    ids, member = spec['ids'], spec['member']
    fault = d.FaultInjection(**spec['fault'])
    journal = d.CompleteRows(spec['journal'], ids, spec['members'], spec['identity'])
    with np.load(spec['input'], allow_pickle=False) as z:
        rows = z[member].copy()
    tasks = [(uid, rows[i], fault, spec['marker'], spec.get('dicom')) for i, uid in enumerate(ids)]
    reason = 'complete'
    try:
        with d.WatchdogPool(1, deadline=d.Deadline.from_env(), stall_seconds=spec['stall'], poll_seconds=.01) as pool:
            # A completed chunk before injection makes partial branch survival observable.
            batches = ([tasks[0], tasks[2]], [tasks[1], *tasks[3:]]) if len(ids) == 32 else (tasks,)
            for batch in batches:
                for uid, row in pool.imap_unordered(_decode, batch, chunksize=2):
                    if row is not None:
                        journal.commit(uid, member, row)
    except d.PoolAborted as exc:
        reason = str(exc)
    dump(Path(spec['result']), {'reason': reason})


def recorded_inputs():
    from tools import exp169_release_c85r7codec as c, hb127_vnextr7_verify as v
    reg, vids, vraw = v.recorded()  # authenticated recorded inference, no label files
    ids, e143, e155, q, a = c.c85.authenticated_a172(c.c85.A5Q)
    stage = c.OUT / 'cpu-replay/replay-input/pr7'
    pids, r, s, b = c.authenticated_pr7(stage)
    order = [pids.index(uid) for uid in ids]
    assert not any(a['failures'].values()) and not any(b['failures'].values())
    craw = dict(R=r[order], S_R7=s[order], EXP143v2=e143, EXP155=e155, Q_A5q=q)
    return c, v, reg, (ids, craw), (vids, vraw)


def blend_payload(line, ids, raw, failures, reg):
    from tools import exp169_release_c85r7codec as c, hb127_vnextr7_release as v
    import numpy as np
    if line == 'c85':
        values = {k: np.nan_to_num(a, nan=.5) for k, a in raw.items()}
        pred = c.blend(ids, *(values[k] for k in c.MEMBERS), failures)
    else:
        pred, _ = v.blend(reg, raw, line[-1])
    return c.p.csv_bytes(ids, pred)


def exercise(work, line, ids, raw, reg, fault, *, dicom=None):
    import numpy as np
    from tools import hb127_vnextr7_release as v
    member = 'R' if line == 'c85' else 'S_R7'
    journal = d.CompleteRows(work / 'rows', ids, list(raw), {'test': 'recorded-cpu-v1'}, create=True)
    for k, a in raw.items():
        if k != member:
            for uid, row in zip(ids, a):
                journal.commit(uid, k, row)
    np.savez(work / 'input.npz', **raw)
    spec = dict(ids=ids, member=member, members=list(raw), identity=journal.header['identity'],
        journal=str(journal.directory), input=str(work / 'input.npz'), fault=vars(fault),
        marker=str(work / 'fault.marker'), result=str(work / 'child-result.json'),
        stall=30 if fault.mode == 'deadline' else 1.5,
        dicom=str(dicom) if dicom else None)
    dump(work / 'job.json', spec)
    clock = fault.deadline(deadline(25))
    with (work / 'child.log').open('w') as log, d.Supervisor(clock, poll_seconds=.01) as supervisor:
        child = supervisor.run(command('pool-child', work / 'job.json'), label=line,
                               stdout=log, stderr=subprocess.STDOUT)
    assert child.cleanup_error is None
    assert child.returncode in (0, 124), (work / 'child.log').read_text()
    recovered, failures = journal.recovered()
    for k in raw:
        for i, uid in enumerate(ids):
            if uid not in failures[k]:
                np.testing.assert_array_equal(recovered[k][i], raw[k][i])
            else:
                assert np.isnan(recovered[k][i]).all()
    expected_raw = {k: a.copy() for k, a in raw.items()}
    for k in raw:
        for i, uid in enumerate(ids):
            if uid in failures[k]:
                expected_raw[k][i] = np.nan
    expected = blend_payload(line, ids, expected_raw, failures, reg)
    actual = blend_payload(line, ids, recovered, failures, reg)
    # Exercise the unchanged vNext atomic publisher, not just its blend.
    if line != 'c85':
        candidate = line[-1]
        # Frozen publisher only permits diagnostic mode for A. B is a scratch-only
        # CPU replay of its normal publisher; this does not build a release stage.
        diagnostic = candidate == 'A'
        v.h.dump(work / 'runtime/build-config.json', v.config(candidate, diagnostic))
        v.finish(work / 'runtime', work / 'export', reg, ids, recovered, {}, {}, 0., clock.started,
                 candidate=candidate, stress=diagnostic, device='cpu')
        output_name = 'diagnostic-hb127.csv' if diagnostic else 'submission.csv'
        assert (work / 'export' / output_name).read_bytes() == actual
    (work / 'survivor.csv').write_bytes(actual)
    if len(ids) == 32:
        fault.assert_csv(ids, actual, expected, marker=spec['marker'], failed_ids=failures[member])
    else:
        assert actual == expected
    if fault.mode == 'off':
        assert not any(failures.values())
        assert actual == blend_payload(line, ids, raw, {}, reg)
    else:
        assert not failures[member].get(ids[0]) and not failures[member].get(ids[2])
        if fault.mode == 'decode':
            assert set(failures[member]) == {fault.uid}
    if fault.mode in ('kill', 'hang'):
        reason = json.loads((work / 'child-result.json').read_text())['reason']
        assert ('pool_worker_exit' if fault.mode == 'kill' else 'pool_no_result') in reason
    if fault.mode == 'deadline':
        assert clock.expired
    return dict(line=line, fault=fault.mode, studies=len(ids), failed=len(failures[member]),
                csv_sha256=hashlib.sha256(actual).hexdigest(), child_reason=child.reason)


def test_recorded_replay_and_all_faults(tmp_path):
    summary = smoke(tmp_path, real_decode=False, env={d.FAULT_ENV: 'all'})
    assert len(summary['stress']) == 12 and len(summary['replay']) == 3


def test_all_failed_existing_fallback_and_assertion_rejects_partial():
    import numpy as np
    import pytest
    c, v, reg, _, (_, source) = recorded_inputs()
    ids = [f"study{i:02d}" for i in range(32)]
    for line, members in [('c85', c.MEMBERS), ('vnext-A', list(source)), ('vnext-B', list(source))]:
        raw = {k: np.full((32, 12), np.nan) for k in members}
        failures = {k: dict.fromkeys(ids, 'failed') for k in members}
        payload = blend_payload(line, ids, raw, failures, reg)
        assert payload == c.p.csv_bytes(ids, np.full((32, 12), .5))
        d.FaultInjection().assert_csv(ids, payload, payload)
        with pytest.raises(AssertionError, match='incomplete'):
            d.FaultInjection().assert_csv(ids, payload[:-20], payload)
        with pytest.raises(AssertionError, match='differs'):
            d.FaultInjection().assert_csv(ids, payload, b'other')


def smoke(out, *, real_decode=True, env=None):
    import numpy as np
    c, v, reg, cr, vr = recorded_inputs()
    dicom = next(v.INPUT.joinpath('visible_data').rglob('*.dcm')) if real_decode else None
    summary = dict(status='PASS', scope='authenticated recorded predictions; S6 stand-in for R7',
                   real_dicom_decode=bool(dicom), pool_context='spawn' if os.name == 'nt' else 'fork',
                   replay=[], stress=[])
    for line, (ids, raw) in [('c85', cr), ('vnext-A', vr), ('vnext-B', vr)]:
        summary['replay'].append(exercise(out / line / 'replay', line, ids, raw, reg,
                                          d.FaultInjection(), dicom=dicom))
        cohort = [f"study{i:02d}" for i in range(32)]
        repeated = {k: np.stack([a[i % len(ids)] for i in range(32)]) for k, a in raw.items()}
        for fault in d.FaultInjection.cases(cohort, stress=True, env=env, early_seconds=3.5):
            summary['stress'].append(exercise(out / line / fault.mode, line, cohort, repeated, reg, fault, dicom=dicom))
    # Exact existing-stage baselines; no stages are prepared or changed.
    reference = c.OUT / 'cpu-replay/diagnostic-c85r7.csv'
    assert (out / 'c85/replay/survivor.csv').read_bytes() == reference.read_bytes()
    saved = v.h.read(v.m.OUT / 'validation/replay.json')
    for record in summary['replay'][1:]:
        assert record['csv_sha256'] == saved['results'][record['line'][-1]]['csv_sha256']
    dump(out / 'summary.json', summary)
    return summary


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('path', nargs='?', type=Path)
    parser.add_argument('detail', nargs='?')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.mode == 'pool-child':
        pool_child(args.path)
    elif args.mode == 'env':
        dump(args.path, {k: os.environ[k] for k in (d.START_ENV, d.BUDGET_ENV)})
    elif args.mode == 'pulse':
        while True:
            args.path.write_bytes(str(time.monotonic()).encode())
            time.sleep(.02)
    elif args.mode == 'tree':
        subprocess.Popen(command('pulse', args.path / 'pulse'))
        end = time.monotonic() + 5
        while not (args.path / 'pulse').exists() and time.monotonic() < end:
            time.sleep(.01)
        if args.detail == 'running':
            time.sleep(60)
    elif args.mode == 'smoke':
        # Reset only this task's fixed scratch subtree, never stages or raw data.
        import shutil
        allowed = (ROOT / '.local/scratch/sol-4/hb129/smoke').resolve()
        if args.out.resolve() != allowed:
            raise ValueError('smoke output must be fixed task scratch/smoke')
        if allowed.exists():
            shutil.rmtree(allowed)
        result = smoke(allowed)
        print(json.dumps(result, indent=2))
    else:
        raise ValueError(args.mode)


if __name__ == '__main__':
    main()
