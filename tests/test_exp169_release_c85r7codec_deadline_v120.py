"""C85 v1.2.0 orchestration gates: frozen math, original journals, CPU only."""
import ast
import builtins
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from tools import exp169_release_c85r7codec_deadline_v120 as m
from tools import exp169_c85r7ins_checks as check


def runtime(tmp):
    blob, pins = m.package()
    rt = tmp / 'runtime'
    m.p.unpack(blob, hashlib.sha256(blob).hexdigest(), rt, pins)
    return rt


def env_clock(monkeypatch, seconds=60, expired=False):
    clock = m.insurance.Deadline(time.monotonic() - (100 if expired else 0), seconds)
    for k, v in clock.environment({}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(m.insurance.FAULT_ENV, 'off')
    monkeypatch.setenv('C85_COMPETITION', str(check.VISIBLE))
    return clock


def test_frozen_functions_and_sources():
    assert m.blend is m.old.blend and m.p_branch is m.old.p_branch
    assert m.authenticated_pr7 is m.old.authenticated_pr7
    for name in ('collect', 'failed_execution'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(m, name)))) == ast.dump(ast.parse(inspect.getsource(getattr(m.old, name))))
    assert m.sha(Path(m.old.__file__)) == m.OLD_SHA
    assert m.sha(Path(m.insurance.__file__)) == m.INSURANCE_SHA
    original = m.old.source_payloads()['public-serving.py'].decode()
    patched = m.serving_source(original)
    assert 'with mp.get_context("fork").Pool(' not in patched
    assert patched.count('with WatchdogPool(') == 1
    plain = m._ORIGINAL_SERVING(original)
    patched = patched.replace('    from release_deadline_v1 import Deadline, WatchdogPool\n', '')
    patched = patched.replace('WatchdogPool(nproc, initializer=_pre_init, deadline=Deadline.from_env(), context="fork")', 'mp.get_context("fork").Pool(nproc, initializer=_pre_init)')
    patched = patched.replace(f'        from {m.MODULE} import before_study\n        before_study(sid)\n', '')
    patched = patched.replace(f'__import__({m.MODULE!r}).study_order(test_ids)', 'test_ids')
    assert patched == plain  # batching, dtype, preprocessing and model math unchanged
    with pytest.raises(ValueError, match='source pin'):
        m.serving_source(original + '\n')


def test_own_q_script_isolation():
    cell = m.old.source_payloads()['a172-cell.py.txt'].decode()
    own = m.a172_script(cell, Path('/kaggle/working/unit'), 'own')
    q = m.a172_script(cell, Path('/kaggle/working/unit'), 'Q_A5q')
    assert '_c85.run_q(' not in own and "'own', main)" in own
    assert '_c85.bind_own' not in q and '_c85.serving_source' not in q
    assert 'import hashlib, json, zipfile' in q and 'def _hash(p):' in q
    frozen = m.c85.a172_source(cell, Path('/kaggle/working/unit'))
    partner = frozen.split("_c85.phase(_out, 'own', main)\n", 1)[1]
    assert q.endswith(partner.replace('_c85.phase(', '_ins.branch_phase('))
    assert '_ins.install_adapters' in own and '_ins.install_adapters' in q
    # Compile each script in isolation; executing native/model bodies is target-only.
    compile(own, '<own>', 'exec'); compile(q, '<q>', 'exec')


def test_cli_own_import_reuses_module_after_adapter_install(tmp_path):
    rt = runtime(tmp_path)
    # Execute the real __main__ bootstrap, replacing only argument dispatch with
    # an audit: this reproduces the child -> generated own -> canonical import.
    code = '''
import ast, importlib, pathlib, sys, types
name = 'exp169_release_c85r7codec_deadline_v120'
path = pathlib.Path(name + '.py')
tree = ast.parse(path.read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
main.body = ast.parse("""
install_adapters(Path('insurance-logs'))
canonical = importlib.import_module(MODULE)
assert canonical is sys.modules['__main__']
canonical.install_adapters(Path('insurance-logs'))
source = canonical.c85.serving_source(Path('public-serving.py').read_text())
assert source.count('with WatchdogPool(') == 1
print('CLI_IMPORT_IDENTITY_PASS')
""").body
ast.fix_missing_locations(tree)
module = types.ModuleType('__main__')
module.__file__ = str(path.resolve())
module.importlib = importlib
sys.modules['__main__'] = module
exec(compile(tree, str(path), 'exec'), module.__dict__)
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=rt,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(rt)),
        capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'CLI_IMPORT_IDENTITY_PASS'


def test_production_fault_rejected_before_any_write(tmp_path, monkeypatch):
    env_clock(monkeypatch)
    monkeypatch.setenv(m.insurance.FAULT_ENV, 'kill')
    with pytest.raises(ValueError, match='diagnostic N32'):
        m.run_release(tmp_path / 'absent-runtime', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
    ids = [str(i) for i in range(32)]
    monkeypatch.setenv(m.insurance.FAULT_ENV, 'off')
    assert m.study_order(ids) is ids
    m.before_study('unused')


def test_expired_setup_publishes_seeded_all_failed_cohort(tmp_path, monkeypatch):
    env_clock(monkeypatch, expired=True)
    rt, out = runtime(tmp_path), tmp_path / 'output'
    result = m.run_release(rt, out)
    assert result['studies'] == 3 and not result['qualified_for_submission']
    ids, values = m.v5q._read_csv(out / 'submission.csv')
    assert len(ids) == 3 and np.all(values == .5)
    events = m.read_json(out / m.RUN / 'insurance-events.json')
    assert len(events) == 1 and events[0]['reason'] == 'deadline_before_start'
    assert not (out / m.RUN / 'setup-ready.json').exists()


def test_setup_identity_error_is_fatal(tmp_path, monkeypatch):
    env_clock(monkeypatch)
    rt, out = runtime(tmp_path), tmp_path / 'output'
    m.seed(rt, out)
    monkeypatch.setattr(m, 'install_adapters', lambda *a: None)
    monkeypatch.setattr(m.base, 'preflight', lambda *a: (_ for _ in ()).throw(ValueError('identity mismatch')))
    with pytest.raises(ValueError, match='identity mismatch'):
        m.child_entry(rt, out, 'setup')
    assert (out / m.RUN / 'setup-fatal.json').exists()
    assert not (out / 'submission.csv').exists()


@pytest.mark.parametrize('failure', ['setup78', 'cleanup', 'own', 'codecs'])
def test_root_branch_isolation_and_fatal_status(tmp_path, monkeypatch, failure):
    env_clock(monkeypatch)
    rt, out = runtime(tmp_path), tmp_path / 'output'
    events = []
    class Supervisor:
        def __init__(self, clock): self.clock = clock
        def __enter__(self): return self
        def __exit__(self, *args): events.append('cleaned')
        def run(self, args, *, label, env, **kw):
            events.append(label)
            assert env['PYTHONPATH'].split(os.pathsep)[0] == str(rt)
            assert env[m.STRESS_ENV] == '0'
            if label == 'setup' and failure != 'setup78':
                m.write_once(out / m.RUN / 'setup-ready.json', m.json_bytes({'status':'READY'}))
            return types.SimpleNamespace(returncode=78 if failure == 'setup78' else int(label == failure),
                reason='child_exit' if label == failure else 'completed', cleanup_error='injected' if failure == 'cleanup' else None)
    monkeypatch.setattr(m.insurance, 'Supervisor', Supervisor)
    original = m.collect
    def collect(*a, **kw):
        assert events[-1] == 'cleaned'
        return original(*a, **kw)
    monkeypatch.setattr(m, 'collect', collect)
    if failure in ('setup78', 'cleanup'):
        with pytest.raises(ValueError, match='fatal setup|cleanup failed'):
            m.run_release(rt, out)
        assert not (out / 'submission.csv').exists()
    else:
        result = m.run_release(rt, out)
        assert result['studies'] == 3
        assert events == ['setup','codecs','own','Q_A5q','R','S_R7','cleaned']
        assert np.all(m.v5q._read_csv(out / 'submission.csv')[1] == .5)


def test_phase_receipt_failure_is_telemetry_only(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'write_once', lambda *a: (_ for _ in ()).throw(OSError('disk')))
    calls = []
    assert m.phase(tmp_path, 'ok', lambda: calls.append(1))['exit_code'] == 0
    assert calls == [1]
    with pytest.raises(SystemExit) as info:
        m.branch_phase(tmp_path, 'bad', lambda: (_ for _ in ()).throw(m.insurance.PoolAborted('pool_no_result')))
    assert info.value.code == 1


def test_both_nested_adapters_preserve_capture_and_deadline(tmp_path, monkeypatch):
    clock = env_clock(monkeypatch)
    monkeypatch.setattr(m.v5q, 'run_process_group', m.v5q.run_process_group)
    monkeypatch.setattr(m.s6, '_run_group', m.s6._run_group)
    monkeypatch.setattr(m.c85, 'serving_source', m.c85.serving_source)
    runner = m.install_adapters(tmp_path)
    assert runner is m.v5q.run_process_group is m.s6._run_group
    text = runner([sys.executable, '-c', 'import os,json; print(json.dumps(dict(start=os.environ["KNEE_RELEASE_START_MONOTONIC_V1"], budget=os.environ["KNEE_RELEASE_DEADLINE_SECONDS_V1"], tree=os.environ["KNEE_RELEASE_TREE_V1"])))'], capture=True)
    result = json.loads(text)
    assert float(result['start']) == clock.started and float(result['budget']) == clock.seconds and result['tree'] == '1'
    with pytest.raises(RuntimeError, match='supervised child'):
        runner([sys.executable, '-c', 'raise SystemExit(9)'])


def test_cpu_faults_original_journals_and_fallbacks(tmp_path):
    result = check.cpu_smoke(tmp_path, real_decode=False)
    assert result['status'] == 'PASS' and len(result['cases']) == 10
    for case in result['cases']:
        assert case['original_collector_identical'] and case['original_fallback_identical']
        assert not case['cleanup_error']


def test_replay_and_local_cards(tmp_path):
    assert m.prepare(tmp_path) == m.prepare(tmp_path)
    assert m.validate(tmp_path)['status'] == 'PASS_CPU_VALIDATE'
    assert m.register(tmp_path)['status'] == 'REGISTERED_LOCAL_ONLY'
    result = m.cpu_replay(tmp_path / 'cpu-replay')
    assert result == m.cpu_replay(tmp_path / 'cpu-replay')
    assert result['diagnostic_sha256'] == m.base.C85_REPLAY_CSV_SHA
    assert result['v110_csv_identical'] and result['v110_p_branch_identical']
    assert set(result['receipt_diff']) == {'version','release','insurance'}
    assert m.admit(tmp_path)['status'] == 'HOLD_NOT_QUALIFIED'


def test_deterministic_superset_flat_runtime(tmp_path):
    blob, pins = m.package()
    assert (blob, pins) == m.package() and len(pins) == 32
    prior = m.old.package()[1]
    assert {k:pins[k] for k in prior} == prior
    rt = runtime(tmp_path)
    result = subprocess.run([sys.executable, '-B', '-c', f'import {m.MODULE} as m; import exp169_c85r7ins_checks as c; m.runtime_pins(m.Path(".")); print(m.VERSION)'], cwd=rt,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(rt)), capture_output=True, text=True, check=True)
    assert result.stdout.strip() == '1.2.0'


@pytest.mark.parametrize('kind', ['kernel','stress-kernel','stress-kernel-clean'])
def test_notebook_entrypoints(tmp_path, monkeypatch, kind):
    blob, pins = m.package()
    code = ''.join(m.notebook_payload(blob, pins, kind)['cells'][-1]['source'])
    assert code.index('KNEE_RELEASE_START_MONOTONIC_V1') < code.index('import base64')
    code = code.replace('/kaggle/working', tmp_path.as_posix())
    seen = []
    original = builtins.__import__
    def importer(name, *a, **kw):
        if name == m.MODULE:
            return types.SimpleNamespace(insurance=m.insurance, run_release=lambda rt,out: seen.append(('production',rt,out)))
        if name == 'exp169_c85r7ins_checks':
            return types.SimpleNamespace(target_stress=lambda rt,out,clean_only: seen.append(('stress',clean_only,os.environ[m.insurance.FAULT_ENV])))
        return original(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', importer)
    monkeypatch.setenv(m.insurance.FAULT_ENV, 'off')
    monkeypatch.setenv(m.insurance.START_ENV, '1')  # bootstrap must retain earlier anchor
    monkeypatch.setenv(m.insurance.BUDGET_ENV, '31200')
    original_path = sys.path[:]
    try:
        exec(compile(code, '<bootstrap>', 'exec'), {})
    finally:
        sys.path[:] = original_path
    assert os.environ[m.insurance.START_ENV] == '1'
    if kind == 'kernel':
        assert seen[0][0] == 'production' and 'target_stress(' not in code
    else:
        assert seen == [('stress',kind == 'stress-kernel-clean','off' if kind.endswith('clean') else 'all')]
        assert 'NEVER SUBMIT' in ''.join(m.notebook_payload(blob,pins,kind)['cells'][0]['source'])
    meta = m.metadata(kind)
    assert meta['id'].startswith('taeyangg4/knee-exp169-c85r7ins-v1')
    assert {k:v for k,v in meta.items() if k not in ('id','title')} == {k:v for k,v in m.old.metadata().items() if k not in ('id','title')}
