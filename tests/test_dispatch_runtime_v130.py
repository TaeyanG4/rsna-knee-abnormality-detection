"""v1.3.1 migration and overflow tests: isolated roots, fake Codex only."""
import concurrent.futures
import os
from pathlib import Path
import time

import pytest

from tools import dispatch_runtime as dr, job_runtime as rt
from test_dispatch_runtime import sandbox, cli, terminal


@pytest.mark.parametrize('i', range(1, 6))
def test_alias_resolution_and_roles(tmp_path, i):
    runtime = dr.Lanes(tmp_path)
    assert dr.canonical_lane(f'sol-{i}') == f'astra-b{i}'
    assert runtime.file(f'sol-{i}', 'id') == runtime.file(f'astra-b{i}', 'id')
    assert runtime.role(f'sol-{i}') == runtime.role(f'astra-b{i}')
    assert 'sol-' not in ' '.join(dr.LANES)
    assert runtime.role('astra-1').startswith('VERIFIER for astra-b1')
    assert runtime.role('astra-b4').startswith('POOL implementer')
    assert runtime.role('astra-b5').startswith('POOL implementer')


def legacy_files(runtime, lane='sol-2', phase='terminal'):
    records = {'id': '11111111-2222-3333-4444-555555555555\n', 'last': 'last job\n',
               'model': 'gpt-6-astra\n', 'n': '2\n', 'opened': '10-13 01:00\n', 'lock': ''}
    for suffix, text in records.items():
        runtime.raw_file(lane, suffix).write_text(text, encoding='utf-8')
    rt.atomic_json(runtime.raw_file(lane, 'state.json'), dict(phase=phase))
    return {s: runtime.raw_file(lane, s).read_bytes() for s in dr.LANE_SUFFIXES}


def test_idle_migration_preserves_all_bytes_busy_deferred_and_idempotent(sandbox):
    root, _ = sandbox; runtime = dr.Lanes(root)
    idle = legacy_files(runtime)
    busy = legacy_files(runtime, 'sol-4', 'running')
    assert runtime.file('astra-b2', 'id').name == 'sol-2.id'
    result = cli(sandbox, 'migrate')
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'sol-4 -> astra-b4: BUSY; deferred' in result.stdout
    for suffix, data in idle.items():
        assert runtime.raw_file('astra-b2', suffix).read_bytes() == data
        assert not runtime.raw_file('sol-2', suffix).exists()
    for suffix, data in busy.items():
        assert runtime.raw_file('sol-4', suffix).read_bytes() == data
        assert not runtime.raw_file('astra-b4', suffix).exists()
    assert runtime.file('astra-b4', 'id').name == 'sol-4.id'
    assert cli(sandbox, 'migrate').returncode == 0
    rt.atomic_json(runtime.raw_file('sol-4', 'state.json'), dict(phase='terminal'))
    assert cli(sandbox, 'migrate').returncode == 0
    assert runtime.file('sol-4', 'id').name == 'astra-b4.id'
    assert cli(sandbox, 'migrate').returncode == 0


def test_live_legacy_lock_blocks_migration_without_state(tmp_path, monkeypatch):
    runtime = dr.Lanes(tmp_path)
    lock = runtime.raw_file('sol-4', 'lock'); lock.write_text('123\n')
    class Alive:
        returncode = 0
    monkeypatch.setattr(dr.subprocess, 'run', lambda *a, **k: Alive())
    assert runtime.migrate() == 0
    assert lock.read_text() == '123\n'
    assert not runtime.raw_file('astra-b4', 'lock').exists()


def test_migration_conflicts_never_overwrite(tmp_path):
    runtime = dr.Lanes(tmp_path); before = legacy_files(runtime)
    target = runtime.raw_file('astra-b2', 'id'); target.write_text('different')
    with pytest.raises(ValueError, match='already has records'):
        runtime.migrate()
    assert target.read_text() == 'different'
    assert {s: runtime.raw_file('sol-2', s).read_bytes() for s in before} == before


def test_interrupted_migration_is_unavailable_until_atomic_commit(tmp_path, monkeypatch):
    runtime = dr.Lanes(tmp_path); before = legacy_files(runtime)
    original = rt.replace_file
    def interrupt(source, target):
        if Path(source).name == 'sol-2.model':
            raise OSError('simulated interrupted move')
        original(source, target)
    monkeypatch.setattr(rt, 'replace_file', interrupt)
    with pytest.raises(OSError, match='interrupted'):
        runtime.migrate()
    for lane in ('sol-2', 'astra-b2'):
        with pytest.raises(ValueError, match='migration interrupted'):
            runtime.file(lane, 'id')
    monkeypatch.setattr(rt, 'replace_file', original)
    assert runtime.migrate() == 0
    assert not runtime.migration_file('astra-b2').exists()
    assert {s: runtime.file('sol-2', s).read_bytes() for s in before} == before


def test_alias_resume_migrate_wait_and_close_existing_session(sandbox):
    root, _ = sandbox; runtime = dr.Lanes(root)
    legacy_files(runtime)
    result = cli(sandbox, 'astra-b2', 'prompt.txt', 'report.md')
    assert result.returncode == 0, result.stdout+result.stderr
    job, _ = terminal(sandbox)
    assert 'resume' in rt.read_json(Path(job['run'])/'fake-args.json')
    assert runtime.file('astra-b2', 'id').name == 'sol-2.id'
    assert cli(sandbox, 'migrate').returncode == 0
    for lane in ('sol-2', 'astra-b2'):
        result = cli(sandbox, 'report.md', lane, wait=True)
        assert result.returncode == 0, result.stdout+result.stderr
    assert cli(sandbox, 'close', 'sol-2').returncode == 0
    assert not runtime.file('astra-b2', 'id').exists()


def mark_busy(runtime, lane='sol-2'):
    rt.atomic_json(runtime.file(lane, 'state.json'), dict(phase='reserved'))
    return runtime.file(lane, 'state.json').read_bytes()


@pytest.mark.parametrize('lane', ['astra-x1', 'astra-x7'])
@pytest.mark.parametrize('busy', [False, True])
@pytest.mark.parametrize('no_overflow', [False, True])
def test_direct_overflow_dispatch_rejected_without_job_or_lane_changes(sandbox, lane, busy, no_overflow):
    root, env = sandbox; runtime = dr.Lanes(root)
    env['DISPATCH_MAX_OVERFLOW'] = '7'
    if no_overflow:
        env['DISPATCH_NO_OVERFLOW'] = '1'
    if busy:
        mark_busy(runtime, lane)
        mark_busy(runtime)
    before = {p.relative_to(runtime.directory): p.read_bytes()
              for p in runtime.directory.rglob('*') if p.is_file()}
    result = cli(sandbox, lane, 'prompt.txt', 'rejected.md')
    assert result.returncode == 2, result.stdout + result.stderr
    assert f'cannot dispatch directly to {lane}' in result.stdout
    assert 'allocated internally only when a requested lane is BUSY' in result.stdout
    assert {p.relative_to(runtime.directory): p.read_bytes()
            for p in runtime.directory.rglob('*') if p.is_file()} == before
    assert not any(p.exists() for p in dr.publication_paths(root/'rejected.md'))


@pytest.mark.parametrize('configured_cap', [None, '7'])
def test_six_is_default_overflow_cap_not_hard_ceiling(sandbox, configured_cap):
    root, env = sandbox; runtime = dr.Lanes(root)
    before = mark_busy(runtime)
    occupied = {}
    for i in range(1, 7):
        lane = f'astra-x{i}'
        run = runtime.directory/'runs'/lane/'fixture'
        metadata = run/'overflow.json'
        rt.atomic_json(metadata, dict(schema='dispatch-overflow/v1', requested_lane='astra-b2', keep_session=False))
        state = runtime.file(lane, 'state.json')
        rt.atomic_json(state, dict(lane=lane, phase='reserved', run=str(run),
                                   sources={str(metadata): dr.sha(metadata)}))
        occupied[lane] = state.read_bytes()
    if configured_cap is not None:
        env['DISPATCH_MAX_OVERFLOW'] = configured_cap
    result = cli(sandbox, 'sol-2', 'prompt.txt', 'seventh.md')
    if configured_cap is None:
        assert result.returncode == 3 and 'overflow cap reached' in result.stdout
        assert not runtime.file('astra-x7', 'state.json').exists()
        assert not (root/'seventh.md.dispatch.json').exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'actual lane astra-x7' in result.stdout
        job, receipt = terminal(sandbox, 'astra-x7')
        assert receipt['state'] == 'SUCCESS'
        assert dr.overflow_metadata(job)['requested_lane'] == 'astra-b2'
        assert not runtime.file('astra-x7', 'id').exists()
    assert runtime.file('sol-2', 'state.json').read_bytes() == before
    for lane, data in occupied.items():
        assert runtime.file(lane, 'state.json').read_bytes() == data


@pytest.mark.parametrize('mode', ['normal', 'fail', 'no_report'])
def test_overflow_same_model_access_fresh_autoclose_and_waiters(sandbox, mode):
    root, env = sandbox; runtime = dr.Lanes(root)
    original = mark_busy(runtime)
    env.update(FORCE_MODEL='gpt-6-sol', FAKE_MODE=mode)
    result = cli(sandbox, 'sol-2', 'prompt.txt', 'overflow.md', 'nosubagents')
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'requested astra-b2; actual lane astra-x1' in result.stdout
    job, receipt = terminal(sandbox, 'astra-x1'); run = Path(job['run'])
    args = rt.read_json(run/'fake-args.json')
    assert 'resume' not in args and 'gpt-6-sol' in args
    assert 'danger-full-access' in args and 'multi_agent' in args
    assert (run/'prompt.txt').read_text(encoding='utf-8') == (root/'prompt.txt').read_text()+dr.ACCESS
    assert job['count'] == 0 and job['session'] == ''
    assert dr.overflow_metadata(job)['requested_lane'] == 'astra-b2'
    assert runtime.file('sol-2', 'state.json').read_bytes() == original
    assert all(not runtime.file('astra-x1', s).exists() for s in ('id', 'n', 'model', 'opened', 'last'))
    assert receipt['state'] == ('SUCCESS' if mode == 'normal' else 'FAILED')
    assert '| astra-x1 |' not in (runtime.directory/'REGISTRY.md').read_text()
    for lane in ('sol-2', 'astra-b2', 'astra-x1'):
        assert cli(sandbox, 'overflow.md', lane, wait=True).returncode == (0 if mode == 'normal' else 1)
    if mode == 'normal':
        assert cli(sandbox, 'overflow.md', 'astra-b3', wait=True).returncode == 1
        rt.atomic_json(run/'overflow.json', dict(schema='dispatch-overflow/v1',
            requested_lane='astra-b3', keep_session=False))
        assert cli(sandbox, 'overflow.md', 'astra-b3', wait=True).returncode == 1


def test_overflow_parallel_first_free_cap_registry_busy_close_and_reuse(sandbox):
    root, env = sandbox; runtime = dr.Lanes(root); mark_busy(runtime)
    env.update(FAKE_GATE=str(root/'release'), DISPATCH_MAX_OVERFLOW='2')
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda i: cli(sandbox, 'sol-2', 'prompt.txt', f'out-{i}.md'), range(2)))
    assert all(r.returncode == 0 for r in results), [(r.stdout,r.stderr) for r in results]
    assert {rt.read_json(root/f'out-{i}.md.dispatch.json')['lane'] for i in range(2)} == {'astra-x1','astra-x2'}
    registry = cli(sandbox, 'lanes')
    assert registry.returncode == 0
    assert '| astra-x1 | OVERFLOW for astra-b2 |' in registry.stdout
    assert '| astra-x2 | OVERFLOW for astra-b2 |' in registry.stdout
    assert cli(sandbox, 'close', 'astra-x1').returncode == 3
    capped = cli(sandbox, 'astra-b2', 'prompt.txt', 'capped.md')
    assert capped.returncode == 3 and 'overflow cap reached' in capped.stdout
    assert not (root/'capped.md.dispatch.json').exists()
    (root/'release').write_text('go')
    terminal(sandbox, 'astra-x1'); terminal(sandbox, 'astra-x2')
    assert cli(sandbox, 'astra-b2', 'prompt.txt', 'reused.md').returncode == 0
    job, _ = terminal(sandbox, 'astra-x1')
    assert job['count'] == 0 and job['session'] == ''


def test_overflow_retention_then_fresh_reuse_and_manual_close(sandbox):
    root, env = sandbox; runtime = dr.Lanes(root); mark_busy(runtime)
    env.update(DISPATCH_KEEP_OVERFLOW='1', SIMPLE='1')
    for i in range(2):
        assert cli(sandbox, 'sol-2', 'prompt.txt', f'keep-{i}.md').returncode == 0
        job, _ = terminal(sandbox, 'astra-x1')
        assert job['session'] == '' and job['count'] == 0 and job['model'] == 'gpt-6-sol'
        assert 'resume' not in rt.read_json(Path(job['run'])/'fake-args.json')
        assert runtime.read('astra-x1', 'id') and runtime.read('astra-x1', 'n') == '1'
    env['DISPATCH_MAX_OVERFLOW']='0'
    assert '| astra-x1 | OVERFLOW for astra-b2 |' in cli(sandbox, 'lanes').stdout
    assert cli(sandbox, 'close', 'astra-x1').returncode == 0
    assert not runtime.file('astra-x1', 'id').exists()
    assert cli(sandbox, 'close', 'astra-x1').returncode == 0


@pytest.mark.parametrize('settings', [dict(DISPATCH_NO_OVERFLOW='1'), dict(DISPATCH_MAX_OVERFLOW='0')])
def test_no_overflow_or_zero_cap_preserves_busy_lane(sandbox, settings):
    root, env = sandbox; runtime = dr.Lanes(root); before = mark_busy(runtime)
    env.update(settings)
    result = cli(sandbox, 'sol-2', 'prompt.txt', 'blocked.md')
    assert result.returncode == 3 and 'BUSY lane astra-b2' in result.stdout
    assert runtime.file('sol-2', 'state.json').read_bytes() == before
    assert not list(runtime.directory.glob('astra-x*'))
    assert not (root/'blocked.md.dispatch.json').exists()


def test_overflow_cap_invalid_and_extra_busy_behavior(sandbox):
    root, env = sandbox; runtime = dr.Lanes(root); mark_busy(runtime)
    env['DISPATCH_MAX_OVERFLOW']='bad'
    assert cli(sandbox, 'sol-2', 'prompt.txt', 'invalid.md').returncode == 2
    mark_busy(runtime, 'extra')
    assert cli(sandbox, 'extra', 'prompt.txt', 'extra.md', 'astra').returncode == 3


def test_lowered_cap_counts_busy_overflow_above_new_limit(sandbox):
    root, env = sandbox; runtime = dr.Lanes(root); mark_busy(runtime)
    mark_busy(runtime, 'astra-x2')
    env['DISPATCH_MAX_OVERFLOW']='1'
    result = cli(sandbox, 'sol-2', 'prompt.txt', 'capped.md')
    assert result.returncode == 3 and 'overflow cap reached' in result.stdout
    assert not runtime.file('astra-x1', 'state.json').exists()


def test_legacy_terminal_waiters_survive_pending_lane_migration(sandbox):
    root, _ = sandbox; runtime = dr.Lanes(root)
    assert cli(sandbox, 'sol-2', 'prompt.txt', 'old.md').returncode == 0
    job, receipt = terminal(sandbox); run = Path(job['run'])
    # Synthesize v1.2 terminal provenance in the isolated fixture, keeping all
    # content hashes and identities intact; legacy jobs stored the alias in JSON.
    pointer = rt.read_json(root/'old.md.dispatch.json')
    for value in (job, receipt, pointer):
        value['lane']='sol-2'
    rt.atomic_json(run/'job.json', job)
    rt.atomic_json(run/'receipt.json', receipt)
    rt.atomic_json(root/'old.md.receipt.json', receipt)
    rt.atomic_json(root/'old.md.dispatch.json', pointer)
    rt.atomic_json(runtime.migration_file('astra-b2'), dict(in_progress=True))
    for lane in ('sol-2', 'astra-b2'):
        result = cli(sandbox, 'old.md', lane, wait=True)
        assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.parametrize('keep', [False, True])
def test_audited_release_closes_overflow_unless_kept(tmp_path, monkeypatch, keep):
    runtime = dr.Lanes(tmp_path)
    lane='astra-x1'; run_id='e'*32; run=runtime.directory/'runs'/lane/run_id
    metadata=run/'overflow.json'
    rt.atomic_json(metadata, dict(schema='dispatch-overflow/v1', requested_lane='astra-b2', keep_session=keep))
    job=dict(lane=lane, run_id=run_id, run=str(run), phase='running', owner={}, worker={},
             sources={str(metadata):dr.sha(metadata)})
    rt.atomic_json(run/'job.json',job); rt.atomic_json(runtime.file(lane,'state.json'),job)
    runtime.file(lane,'id').write_text('retained-session')
    runtime.file(lane,'n').write_text('0')
    monkeypatch.setattr(dr, 'contained_tree_state', lambda *args:'dead')
    monkeypatch.setattr(rt, 'liveness', lambda identity:'dead')
    assert runtime.release(lane) == 0
    assert runtime.file(lane,'id').exists() == keep
    assert rt.read_json(run/'release.json')['success'] is False


def test_output_mutex_also_covers_overflow(sandbox):
    root, env = sandbox; runtime = dr.Lanes(root); mark_busy(runtime)
    env['FAKE_GATE']=str(root/'release')
    assert cli(sandbox, 'astra-b2', 'prompt.txt', 'same.md').returncode == 0
    for lane in ('sol-2','astra-b3'):
        assert cli(sandbox, lane, 'prompt.txt', 'same.md').returncode == 3
        assert cli(sandbox, lane, 'prompt.txt', 'same').returncode == 3
    assert not runtime.file('astra-x2', 'state.json').exists()
    (root/'release').write_text('go'); terminal(sandbox, 'astra-x1')


def test_real_numeric_cpu_smoke_migration_overflow_and_alias_waiters(sandbox):
    root, env = sandbox; runtime = dr.Lanes(root)
    source = Path(__file__).resolve().parents[1]/'artifacts/experiments/EXP185_FINDING_MIL/outputs/S42/results/EXP185_FINDING_MIL_F0_S42/oof_k36_flip_tta.csv'
    digest = dr.sha(source); env['FAKE_DATA']=str(source)
    legacy_files(runtime)
    busy = legacy_files(runtime, 'sol-4', 'reserved')
    assert cli(sandbox, 'astra-b2', 'prompt.txt', 'numeric.md').returncode == 0
    terminal(sandbox, 'sol-2')
    assert cli(sandbox, 'migrate').returncode == 0
    assert runtime.file('sol-2', 'id').name == 'astra-b2.id'
    assert cli(sandbox, 'numeric.md', 'sol-2', wait=True).returncode == 0
    assert cli(sandbox, 'astra-b4', 'prompt.txt', 'numeric-overflow.md').returncode == 0
    terminal(sandbox, 'astra-x1')
    for lane in ('sol-4', 'astra-b4', 'astra-x1'):
        assert cli(sandbox, 'numeric-overflow.md', lane, wait=True).returncode == 0
    for name in ('numeric.md', 'numeric-overflow.md'):
        assert rt.read_json(root/name) == dict(rows=888, sha256=digest)
    assert {s: runtime.raw_file('sol-4', s).read_bytes() for s in busy} == busy
    assert dr.sha(source) == digest
    assert not runtime.file('astra-x1', 'id').exists()
    assert cli(sandbox, 'lanes').returncode == 0


def test_access_and_wrapper_body_byte_compatible():
    import ast
    root = Path(__file__).resolve().parents[1]
    backup = root/'.local/backups/dispatch-v1.2.0'
    tree = ast.parse((backup/'dispatch_runtime.py').read_text(encoding='utf-8'))
    old_access = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ACCESS' for t in node.targets))
    assert dr.ACCESS == old_access
    current = (root/'.local/astra-improve/dispatch.sh').read_bytes()
    previous = (backup/'dispatch.sh').read_bytes()
    assert current[current.index(b'set -u'):].replace(b'\r\n',b'\n') == previous[previous.index(b'set -u'):].replace(b'\r\n',b'\n')
