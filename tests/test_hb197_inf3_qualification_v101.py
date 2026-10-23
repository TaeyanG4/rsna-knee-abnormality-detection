"""TEST ONLY: synthetic cohorts/native archives; no forwards, GPU or transport."""
import ast
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import types
import numpy as np
import pytest
from tools import hb197_inf3_qualification_v101 as b
from tools import hb197_inf3_observer_v101 as observer
from tools import hb197_inf3_collect_v101 as c
from tools import strict_variant_gate_inf3_qualification_v101 as g
from tools import strict_variant_gate_inf3_v103 as old


def put(root, name, value):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else b.encoded(value))
    return old.ref(path, root)


def definitions(data, names, scope):
    tree = ast.parse(data)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-pure-definitions>', 'exec'), scope)
    return scope


def pure_functions(files):
    from tools import exp169_release_c85 as base
    q = definitions(files['exp169_partner_bundle_release.py'], {'rank', 'aggregate'}, {'need': b.need})
    ns = b.science.runtime(files)
    rank = definitions(files['exp169_release_c85.py'], {'rank', 'scored_rank'}, dict(vars(base)))
    c85 = types.SimpleNamespace(**{k: rank[k] for k in ('rank', 'scored_rank')})
    final = definitions(files['exp169_r224.py'], {'failure_gate', 'r_half', 'p_branch', 'blend', 'reference'},
                        dict(np=np, c85=c85, p=base.p, v5q=base.v5q))
    export = definitions(files['exp169_release_p956.py'], {'csv_bytes'}, dict(vars(base.p)))
    return dict(q0=q['aggregate'], q3=ns['inf3_aggregate'], rank=rank['rank'],
        blend=final['blend'], reference=final['reference'], p_branch=final['p_branch'], csv=export['csv_bytes'])


@pytest.fixture(scope='module')
def payload():
    return b.payload()


def test_exact31_two_changes_and_verbatim_function(payload):
    nb, metadata, files, proof, diff, ndiff = payload
    original = b.science.unpack(b.science.read(b.science.OUT / 'kernel/release.ipynb'))[0]
    assert len(files) == 31 and sum(files[n] == original[n] for n in files) == 29
    assert files['raptor224_run.py'].decode().replace(b.ROOT_PREFIX, '', 1).encode() == original['raptor224_run.py']
    assert metadata['id'] == b.SLUG and '/' not in metadata['title']
    assert b.ENV in ''.join(nb['cells'][2]['source']) and 'stress.prepare' in ''.join(nb['cells'][2]['source'])
    assert diff and ndiff


@pytest.mark.parametrize('case', ['direct0', 'direct1', 'walkseries', 'walktest', 'absent'])
def test_absent_override_original_behavior(payload, case):
    files = payload[2]; calls = []
    fake = types.SimpleNamespace(environ={}, path=types.SimpleNamespace(
        exists=lambda x: x.startswith('/kaggle/input/' + ('competitions/' if case == 'direct0' else '') + 'rsna-') and case.startswith('direct'),
        isdir=lambda x: case == 'walkseries'), walk=lambda x: [('synthetic-root', [], ['test.csv'])] if case.startswith('walk') else [])
    original = b.science.unpack(b.science.read(b.science.OUT / 'kernel/release.ipynb'))[0]
    results = []
    for data in (original['raptor224_run.py'], files['raptor224_run.py']):
        f = definitions(data, {'find_test_root'}, dict(os=fake, json=json))['find_test_root']
        try: results.append(f())
        except RuntimeError as error: results.append(str(error))
    assert results[0] == results[1]


def synthetic_cohort(tmp_path, files):
    source = tmp_path / 'source'; source.mkdir()
    from tools import exp169_release_p956 as p
    targets = list(p.LABELS)
    rows = []
    for i in range(40):
        uid = f'SYNTHETIC-{i:03}'; series = f'SERIES-{i:03}'
        folder = source / 'train_series' / uid / series; folder.mkdir(parents=True)
        for j in range(i % 5 + 1): (folder / f'{j}.dcm').write_bytes(b'SYNTHETIC DICOM NAME ONLY')
        rows.append([uid, series, 'Sagittal', 'True'])
    with (source / 'train_series.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive']); writer.writerows(rows)
    scope = dict(collections=__import__('collections'), csv=csv, json=json, os=os, Path=Path,
        time=__import__('time'), UID='StudyInstanceUID', LABELS=targets,
        COLS=['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive'])
    selector = definitions(files['exp169_c85_stress.py'], {'dump', 'table', 'prepare'}, scope)
    root = tmp_path / 'subset'; ids = selector['prepare'](source, root, 32)
    another = selector['prepare'](source, tmp_path / 'subset2', 32)
    assert ids == another and len(ids) == 32
    observer.seal_cohort(root, ids)
    return root


def test_valid_exact32_and_no_visible_fallback(tmp_path, payload, monkeypatch):
    root = synthetic_cohort(tmp_path, payload[2]); monkeypatch.setenv(b.ENV, str(root)); monkeypatch.setenv('C85_COMPETITION', str(root))
    f = definitions(payload[2]['raptor224_run.py'], {'find_test_root'}, dict(os=os, json=json))['find_test_root']
    assert f() == str(root)
    for value in ('', 'relative', str(tmp_path / 'missing')):
        monkeypatch.setenv(b.ENV, value)
        with pytest.raises((RuntimeError, OSError)): f()
    monkeypatch.setenv(b.ENV, str(root)); (root / 'test.csv').write_text('StudyInstanceUID\nSTALE-VISIBLE3\n')
    with pytest.raises(RuntimeError, match='stale'): f()


def test_actual_full1300_global_benchmark(payload, tmp_path):
    result = observer.benchmark_functions(pure_functions(payload[2]), payload[3]['diagnostic'], hidden=1300)
    assert result['hidden_studies'] == 1300 and result['seconds'] > 0 and result['model_forward'] is False
    assert result['leaf_shapes']['l2'] == [10, 1300, 12]
    put(tmp_path, 'actual-synthetic-cost.json', result)


class Chain:
    def __init__(self, root):
        self.root = root; self.count = 0
        spec = importlib.util.spec_from_file_location('prior_fixture_only', old.ROOT / 'tests/test_hb197_inf3_fix1.py')
        prior_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior_module)
        self.prior = prior_module.Chain(root)
        self.build = b.validate(); self.sources = b.sources()
        for p in self.sources: put(root, p, (b.ROOT / p).read_bytes())
        prefix = b.WORK.relative_to(b.ROOT)
        names = ['diagnostic-kernel/release.ipynb', 'diagnostic-kernel/kernel-metadata.json',
                 'FULL-EMBEDDED.diff', 'FULL-NOTEBOOK.diff'] + ['diagnostic-runtime/' + n for n in b.payload()[2]]
        for name in names:
            put(root, (prefix / name).as_posix(), (b.WORK / name).read_bytes())
        self.artifacts = {}
        self.build_ref = put(root, 'fixture-q/build.json', self.build)
        self.diff = put(root, 'fixture-q/diff.json', b.payload()[3])
        previous = self.read(self.prior.receipts['BUILD_PASS'])
        self.preservation = put(root, 'fixture-q/preservation.json', dict(schema='h197-qualification-preservation/v1',
            prior_build_sha256=self.prior.receipts['BUILD_PASS']['sha256'], bindings=previous['consumed_sha256']))
        log = put(root, 'fixture-q/tests.log', b'101 passed -- TEST ONLY parser fixture\n')
        self.tests = put(root, 'fixture-q/tests.json', dict(schema='h197-qualification-tests/v1', fixture_only=True,
            actual_run=True, exit_code=0, failed=0, passed=101, source_sha256=self.sources, log=log))
        self.artifacts = {'full_visible': self.run('FULL_VISIBLE'), 'n32': self.run('N32')}
        self.artifacts['population'] = put(root, 'fixture-q/population.json', dict(schema='h197-hidden-population/v1',
            recorded_by='B', applicable=True, studies=1300, source=put(root, 'fixture-q/population-source.txt', b'TEST ONLY hidden population\n')))
        self.subjects = {}; self.envelopes = {}
        for phase in g.PHASES:
            e, v = self.phase(phase); self.subjects[phase] = e; self.envelopes[phase] = v

    def read(self, item): return json.loads((self.root / item['path']).read_text())

    def review(self, subject_ref, subject, **changes):
        self.count += 1; prefix = f'fixture-q/review-{self.count}'
        claim = dict(schema='h197-qualification-review/v1', candidate=b.science.KEY, phase=subject['phase'],
            builder=b.BUILDER, reviewer='/root/fixture_qualification_reviewer', verdict='PASS', subject_sha256=subject_ref['sha256'],
            source_sha256=subject['sources'], release=subject['release'], diagnostic=subject['diagnostic'],
            checks={k: True for k in g.CHECKS[subject['phase']]})
        claim.update(changes)
        report = put(self.root, prefix + '.md', ('TEST ONLY, no actual completion\nH197-QUALIFICATION-ATTESTATION: ' +
            json.dumps(claim, sort_keys=True) + '\nVERDICT: PASS\n').encode())
        prompt = put(self.root, prefix + '.prompt.txt', b'TEST ONLY native parser fixture\n')
        completed = put(self.root, prefix + '.completion.json', dict(schema=old.native.SCHEMA, host=old.native.HOST, model=old.native.MODEL,
            agent_id=claim['reviewer'], lane='native-astra-r8', prompt=str(self.root / prompt['path']), report=str(self.root / report['path']),
            terminal_status='completed', observed_at='2026-10-18T00:00:00+09:00', prompt_sha256=prompt['sha256'], report_sha256=report['sha256'],
            completion_message='TEST ONLY synthetic archive, no real review'))
        receipt = old.native.record(self.root / completed['path'], 'TEST ONLY fixture', self.root)
        return put(self.root, prefix + '.wrapper.json', dict(claim, report=report, native_receipt=old.ref(receipt, self.root)))

    def phase(self, phase, edits=None, review_edits=None):
        e = dict(schema='h197-qualification-subject/v1', candidate=b.science.KEY, phase=phase, fixture_only=True, sources=self.sources,
            prior_build=self.prior.receipts['BUILD_PASS'], build=self.build_ref, diff=self.diff, tests=self.tests, preservation=self.preservation,
            release=dict(slug=b.science.SLUG, version=None if phase == 'LAUNCH_READY' else 9, kernel_sha256=b.KERNEL, metadata_sha256=b.METADATA),
            diagnostic=dict(slug=b.SLUG, version=None if phase == 'LAUNCH_READY' else 2,
                kernel_sha256=self.build['diagnostic_kernel_sha256'], metadata_sha256=self.build['diagnostic_metadata_sha256']),
            artifacts={} if phase == 'LAUNCH_READY' else copy.deepcopy(self.artifacts))
        if edits: edits(e)
        self.count += 1; subject = put(self.root, f'fixture-q/subject-{self.count}.json', e)
        rv = self.review(subject, e, **(review_edits or {}))
        return subject, put(self.root, f'fixture-q/envelope-{self.count}.json', dict(schema='h197-qualification-envelope/v1', subject=subject, review=rv))

    def run(self, kind):
        n = 3 if kind == 'FULL_VISIBLE' else 32; prefix = 'fixture-q/' + kind + '/'
        ids = [f'TEST-ONLY-{i:03}' for i in range(n)]; targets = self.prior.targets
        def local(name, value): return put(self.root, prefix + name, value)
        def output(name, uids=ids):
            return local(name, ('StudyInstanceUID,' + ','.join(targets) + '\n' + ''.join(u + ',' + ','.join(['0.4'] * 12) + '\n' for u in uids)).encode())
        notebook = b.science.OUT / 'kernel/release.ipynb' if kind == 'FULL_VISIBLE' else b.OUT / 'release.ipynb'
        metadata = notebook.parent / 'kernel-metadata.json'; slug = b.science.SLUG if n == 3 else b.SLUG; version = 9 if n == 3 else 2
        phases = []; branches = {}
        for role, name in c.ROLES.items():
            phase = dict(phase=name, exit_code=0, seconds=1.0); phases.append(phase)
            part = dict(exit=local(role + '-exit.json', phase), logs=[local(role + '.log', b'TEST ONLY actual-log-shaped COMPLETE\n')])
            if role in ('own', 'q', 'raptor'):
                members = ['EXP143v2', 'EXP155'] if role == 'own' else [str(i) for i in range(20)] if role == 'q' else ['R']
                journal = ''.join(json.dumps(dict(uid=u, member=m, values=[.4] * 12)) + '\n' for u in ids for m in members)
                part['journal'] = local(role + '.jsonl', journal.encode())
            if role == 'q':
                stream = io.BytesIO(); np.savez(stream, ids=np.asarray(ids), **{f: np.full((10 if f == 'l2' else 5, n, 12), .4, np.float32) for f in ('l2', 'dino', 'convnext')})
                part['leaves'] = local('leaves.npz', stream.getvalue())
                part['timings'] = local('timings.json', dict(completed=[n // 2, n - n // 2], study_seconds=[1.] * n, gpu_peak_bytes=[1000, 1000]))
                part['manifest'] = local('q-manifest.json', b.payload()[2]['q-a5q-manifest.json'])
            elif role == 'r224':
                part.update(raw=output('r224raw.csv'), ranked=output('r224rank.csv'))
                part['receipt'] = local('r224receipt.json', dict(status='COMPLETE_R224', checkpoint_sha256=c.literal('exp169_r224.py', 'CHECKPOINT_SHA'),
                    files={'raptor224_raw.csv': part['raw'], 'raptor224_sub.csv': part['ranked']}))
            elif role == 'r6':
                part['workers'] = {}
                for i in range(2):
                    assigned = ids[i::2]; worker = dict(ids=local(f'w{i}-ids.json', assigned), output=output(f'w{i}.csv', assigned),
                        receipt=local(f'w{i}-receipt.json', dict(status='PASS', studies=len(assigned), device='cuda', cuda_initialized=True, fallback={})),
                        log=local(f'w{i}.log', b'TEST ONLY WORKER_DONE\n'))
                    part['workers'][str(i)] = worker
                part['controller'] = local('controller.json', dict(status='PASS', rows=n, worker_exit_codes={'0': 0, '1': 0}))
                part['output'] = output('r6.csv')
                part['invocation'] = local('invocation.json', dict(device='cuda', args=['--device', 'cuda']))
                ident = dict(manifest_sha256='a' * 64, file_sha256={'test': 'b' * 64}, dataset_slug='fixture/bundle', bundle_id='fixture')
                part['provenance'] = local('provenance.json', ident); part['identity'] = local('identity.json', ident)
            branches[role] = part
        outer = {}
        for i, name in enumerate(('a172', 'public')):
            p = dict(phase='child', exit_code=0, seconds=i + 2.); phases.append(p); outer[name] = local(name + '-child.json', p)
        out = output('submission.csv')
        status = local('status.txt', b'TEST ONLY EXECUTING then COMPLETE; owned children gone\n')
        spec = dict(schema='h197-completed-run/v1', fixture_only=True, candidate=b.science.KEY, kind=kind,
            remote=local('remote.json', dict(schema='h197-remote-completion/v1', fixture_only=True, recorded_by='B', status='COMPLETE',
                slug=slug, version=version, owned_children_dead=True, fast=False, executing_started_at='2026-10-18T01:00:00+00:00',
                completed_at='2026-10-18T01:10:00+00:00', raw_status_evidence=[status])),
            notebook=local('release.ipynb', notebook.read_bytes()), metadata=local('kernel-metadata.json', metadata.read_bytes()),
            remote_metadata=local('remote-metadata.json', dict(id=slug, versionNumber=version)),
            console=local('console.log', ('TEST ONLY\n' + ''.join('C85_PHASE ' + json.dumps(p) + '\n' for p in phases)).encode()),
            ids=local('ids.json', ids), targets=targets, fixture_frozen=self.prior.frozen, output=out,
            release_receipt=local('receipt.json', dict(status='COMPLETE_C85R7', studies=n, submission_sha256=out['sha256'],
                exposure='EXPOSED', clean_oof=False, codecs=dict(status='ACTIVE', s_r7_gate='ACTIVE'), version=b.science.VERSION,
                findings=12, formula=c.literal('exp169_r224.py', 'FORMULA'), fallbacks={k: 0 for k in ('a172_prediction', 'r_prediction', 's_r7_prediction', 'r224_prediction')})),
            branches=branches, outer_exits=outer, device_evidence=[status])
        if n == 32:
            cohort_files = {'selection.json': local('selection.json', dict(n=32, method='round-robin max/min series/max/min slices', selection=[dict(uid=u) for u in ids]))}
            for name in ('test.csv', 'test_series.csv'):
                cohort_files[name] = local(name, ('StudyInstanceUID\n' + '\n'.join(ids) + '\n').encode())
            cohort_files['sample_submission.csv'] = output('sample_submission.csv')
            spec['cohort_files'] = cohort_files
            spec['cohort'] = local('cohort.json', dict(schema='h197-n32-cohort/v1', studies=32, ids=ids, sha256={k: v['sha256'] for k, v in cohort_files.items()}))
            gpu = dict(uuid='TEST-GPU', name='TEST ONLY T4', total_mib=16000, used_mib=1000)
            spec['observer'] = local('observer.json', dict(schema='h197-observer/v1.0.1', version='1.0.1', status='STOPPED', platform='Linux TEST ONLY',
                cleanup_complete=True, cleanup_errors=[], observer_started=True, observer_thread_stopped=True, original_failure=None,
                source_sha256=b.sha(b.ROOT / 'tools/hb197_inf3_observer_v101.py'), owned_children_alive=[], watchdog_abort=None,
                telemetry_errors=[], samples=[dict(gpu=[gpu]), dict(gpu=[gpu])], process_inventory=[dict(pid=123)], monotonic_wall_seconds=600.))
            spec['global_cost'] = local('cost.json', dict(schema='h197-global-cost/v1', status='COMPLETE_SYNTHETIC_COST_ONLY',
                model_forward=False, real_predictions=False, repeats=3, measured_seconds=[1., 1., 1.], seconds=1., hidden_studies=1300,
                source_sha256=b.payload()[3]['diagnostic'], observer_source_sha256=b.sha(b.ROOT / 'tools/hb197_inf3_observer_v101.py'),
                leaf_shapes={f: [10 if f == 'l2' else 5, 1300, 12] for f in ('l2', 'dino', 'convnext')}, leaf_dtype='float32', pool_dtype='float64'))
            spec['diagnostic_diff'] = self.diff
        return local('manifest.json', spec)

    def evaluate(self, phase='QUALIFICATION_READY', pair=None):
        e, v = pair or (self.subjects[phase], self.envelopes[phase])
        return g.evaluate_fixture(phase, e, v, root=self.root, frozen=self.prior.frozen)


@pytest.fixture(scope='module')
def chain(tmp_path_factory): return Chain(tmp_path_factory.mktemp('TEST-ONLY-qualification'))


@pytest.mark.parametrize('phase', g.PHASES)
def test_real_schema_full_archive_positive_not_admission(chain, phase):
    result = chain.evaluate(phase)
    assert result['fixture_only'] and not result['qualified_for_submission'] and not result['launch_authorized']
    assert len(result['consumed_sha256']) > 450
    if phase == 'QUALIFICATION_READY': assert result['release']['version'] == 9 and result['diagnostic']['version'] == 2


@pytest.mark.parametrize('change', [dict(reviewer=b.BUILDER), dict(builder='/root/other'), dict(subject_sha256='0' * 64),
    dict(source_sha256={}), dict(phase='WRONG'), dict(checks={}), dict(verdict='FAIL')])
def test_review_binding_failures(chain, change):
    with pytest.raises(ValueError): chain.evaluate('LAUNCH_READY', chain.phase('LAUNCH_READY', review_edits=change))


@pytest.mark.parametrize('edge', ['report.md', 'receipt.json', 'prompt.txt', 'completion.json', 'run.log'])
def test_native_archive_mutation(chain, edge):
    env = chain.read(chain.envelopes['LAUNCH_READY']); review = chain.read(env['review']); receipt = chain.read(review['native_receipt'])
    path = chain.root / '.local/astra-improve/lanes/runs' / receipt['lane'] / receipt['run_id'] / edge
    original = path.read_bytes()
    try:
        path.write_bytes(original + b' ')
        with pytest.raises(ValueError): chain.evaluate('LAUNCH_READY')
    finally: path.write_bytes(original)


@pytest.mark.parametrize('mutation', ['childlog', 'qcost', 'qnan', 'r224three', 'r6exit', 'missingbranch', 'observer',
    'wall', 'globalnan', 'shape', 'diff', 'version', 'source', 'cohort'])
def test_complete_run_negative_fields(chain, mutation):
    spec = chain.read(chain.artifacts['n32']); root = chain.root
    def changed(item, fn):
        val = chain.read(item); fn(val); return put(root, 'fixture-q/negative-' + mutation + '.json', val)
    if mutation == 'childlog': spec['branches']['own']['logs'] = []
    elif mutation == 'qcost': spec['branches']['q']['timings'] = changed(spec['branches']['q']['timings'], lambda x: x.update(study_seconds=[]))
    elif mutation == 'qnan':
        stream = io.BytesIO(); np.savez(stream, ids=np.asarray(chain.read(spec['ids'])), **{f: np.full((10 if f == 'l2' else 5, 32, 12), np.nan, np.float32) for f in ('l2', 'dino', 'convnext')})
        spec['branches']['q']['leaves'] = put(root, 'fixture-q/nan.npz', stream.getvalue())
    elif mutation == 'r224three': spec['branches']['r224']['raw'] = chain.read(chain.artifacts['full_visible'])['branches']['r224']['raw']
    elif mutation == 'r6exit': spec['branches']['r6']['controller'] = changed(spec['branches']['r6']['controller'], lambda x: x.update(worker_exit_codes={'0': 1, '1': 0}))
    elif mutation == 'missingbranch': del spec['branches']['q']
    elif mutation == 'observer': spec['observer'] = changed(spec['observer'], lambda x: x.update(owned_children_alive=[1]))
    elif mutation == 'wall': spec['observer'] = changed(spec['observer'], lambda x: x.update(monotonic_wall_seconds=-1))
    elif mutation == 'globalnan':
        val = chain.read(spec['global_cost']); val['measured_seconds'] = [float('inf')] * 3
        spec['global_cost'] = put(root, 'fixture-q/nonfinite-cost.json', json.dumps(val).encode())
    elif mutation == 'shape': spec['global_cost'] = changed(spec['global_cost'], lambda x: x.update(leaf_shapes={}))
    elif mutation == 'diff': spec['diagnostic_diff'] = changed(spec['diagnostic_diff'], lambda x: x.update(unchanged=30))
    elif mutation == 'version': spec['remote_metadata'] = changed(spec['remote_metadata'], lambda x: x.update(versionNumber=999))
    elif mutation == 'source': spec['notebook'] = chain.read(chain.artifacts['full_visible'])['notebook']
    elif mutation == 'cohort': spec['cohort'] = changed(spec['cohort'], lambda x: x.update(studies=3))
    item = put(root, 'fixture-q/negative-run-' + mutation + '.json', spec)
    with pytest.raises((ValueError, KeyError)):
        c.execution(old.Inputs(root), item, chain.build, fixture=True)


def test_late_recheck_source_and_fixture_boundary(chain):
    result = chain.evaluate('LAUNCH_READY'); inputs = old.Inputs(chain.root); inputs.pins(result['consumed_sha256'])
    path = chain.root / 'tools/hb197_inf3_collect_v101.py'; original = path.read_bytes()
    try:
        path.write_bytes(original + b'\n')
        with pytest.raises(ValueError, match='stale'): inputs.recheck()
    finally: path.write_bytes(original)
    with pytest.raises(ValueError): g.late_recheck(result)
    with pytest.raises(ValueError): g.evaluate_fixture('LAUNCH_READY', {}, {}, root=b.ROOT, frozen={})


def test_only9h_projection_gate(chain):
    run = c.execution(old.Inputs(chain.root), chain.artifacts['n32'], chain.build, fixture=True)
    population = chain.read(chain.artifacts['population'])
    assert c.projection(run, population)['status'] == 'PASS'
    run['complete_wall_seconds'] = 1000
    assert c.projection(run, population)['status'] == 'HOLD'


class FakeThread:
    def __init__(self, **kwargs):
        self.alive = False; self.hung = False; self.join_error = False; self.state_error = False
    def start(self): self.alive = True
    def is_alive(self):
        if self.state_error: raise OSError('TEST ONLY state query failure')
        return self.alive
    def join(self, timeout):
        assert timeout == 20
        if self.join_error: raise OSError('TEST ONLY join failure')
        if not self.hung: self.alive = False


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    """Actual observer methods; only OS/thread/time/telemetry external boundaries fake."""
    clock = [100.]
    monkeypatch.setattr(observer.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(observer.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(observer.threading, 'Thread', FakeThread)
    monkeypatch.setattr(observer.signal, 'SIGKILL', 9, raising=False)  # Linux boundary on Windows test host
    watch = observer.Observer(tmp_path, clock[0])
    table = {watch.owner: dict(pid=watch.owner, ppid=1, start_ticks=100, state='S')}
    signals = []; records = []; stubborn = set(); forbidden = set(); hooks = {}
    monkeypatch.setattr(observer, 'process_table', lambda: {p: dict(v) for p, v in table.items()})
    monkeypatch.setattr(observer, 'gpu_sample', lambda: [dict(uuid='TEST-ONLY', name='TEST-ONLY', total_mib=16, used_mib=1,
        utilization_percent=0, power_watts=0)])
    monkeypatch.setattr(observer, 'progress_bytes', lambda _: 0)
    monkeypatch.setattr(observer, 'dump', lambda p, value: records.append(value))
    def kill(pid, sig):
        assert pid in table and pid != watch.owner
        signals.append((pid, table[pid]['start_ticks'], sig))
        if pid in forbidden: raise PermissionError('TEST ONLY signal denied')
        if pid in hooks: hooks[pid](sig)
        if sig == observer.signal.SIGKILL or pid not in stubborn: table.pop(pid, None)
    monkeypatch.setattr(observer.os, 'kill', kill)
    def child(pid=424242, parent=None, start=200):
        table[pid] = dict(pid=pid, ppid=watch.owner if parent is None else parent, start_ticks=start, state='S')
        return table[pid]
    return types.SimpleNamespace(w=watch, table=table, signals=signals, records=records, child=child,
        clock=clock, stubborn=stubborn, forbidden=forbidden, hooks=hooks)


def test_finish_terminates_unseen_live_owned_child(lifecycle):
    f=lifecycle; f.w.start(); f.child()
    result=f.w.finish()
    assert result['cleanup_complete'] and result['owned_children_alive']==[] and result['observer_thread_stopped']
    assert f.signals==[(424242,200,observer.signal.SIGTERM)]
    assert result['process_inventory'][0]['pid']==424242


def test_finish_escalates_term_kill_and_verifies_death(lifecycle):
    f=lifecycle; f.w.start(); f.child(); f.stubborn.add(424242)
    result=f.w.finish()
    assert result['cleanup_complete']
    assert [x[2] for x in f.signals]==[observer.signal.SIGTERM,observer.signal.SIGKILL]
    assert result['cleanup_signals'][1]['elapsed_seconds']>=observer.TERM_GRACE_SECONDS


def test_finish_discovers_child_spawned_during_term_cleanup(lifecycle):
    f=lifecycle; f.w.start(); f.child(); f.stubborn.add(424242)
    def spawn(sig):
        if sig==observer.signal.SIGTERM: f.child(424243,424242,201)
    f.hooks[424242]=spawn
    result=f.w.finish()
    assert result['cleanup_complete'] and {r['pid'] for r in result['process_inventory']}=={424242,424243}
    assert (424243,201,observer.signal.SIGTERM) in f.signals


def test_finish_discovers_descendant_created_while_joining(lifecycle):
    f=lifecycle; f.w.start()
    def join(timeout): f.child(); f.w.thread.alive=False
    f.w.thread.join=join
    assert f.w.finish()['cleanup_complete'] and f.signals==[(424242,200,observer.signal.SIGTERM)]


def test_finish_handles_known_orphan_and_its_new_descendant(lifecycle):
    f=lifecycle; f.w.start(); old=f.child(); f.w.seen[(424242,200)]=dict(old)
    old['ppid']=1; f.child(424243,424242,201)
    assert f.w.finish()['cleanup_complete']
    assert {x[0] for x in f.signals}=={424242,424243}


def test_finish_never_signals_reused_unrelated_or_baseline_process(lifecycle):
    f=lifecycle; f.child(424244); f.w.start()  # baseline child and its subtree are excluded
    f.child(424245,424244,203); f.child(424246,1,204)
    f.w.seen[(424242,200)]=dict(pid=424242,ppid=f.w.owner,start_ticks=200,state='S')
    f.child(424242,1,999)
    assert f.w.finish()['cleanup_complete'] and not f.signals
    assert {424242,424244,424245,424246}<=set(f.table)


def test_signal_rechecks_identity_immediately_before_each_attempt(lifecycle,monkeypatch):
    f=lifecycle; f.w.start(); old=f.child(); f.w.seen[(424242,200)]=dict(old)
    f.table[424242]['start_ticks']=999
    assert f.w._signal_owned((424242,200)) is False
    assert f.w._signal_owned((424242,200),force=True) is False
    assert not f.signals


@pytest.mark.parametrize('failure',['bootstrap','inference','keyboard'])
def test_actual_lifecycle_preserves_original_failure_and_cleans(lifecycle,failure):
    f=lifecycle; original=KeyboardInterrupt('TEST ONLY') if failure=='keyboard' else ValueError('TEST ONLY '+failure)
    with pytest.raises(type(original)) as raised:
        with f.w.lifecycle():
            f.child()
            raise original
    assert raised.value is original and f.records[-1]['original_failure']==type(original).__name__
    assert f.records[-1]['cleanup_complete'] and f.signals


def test_activation_exception_also_enters_cleanup(lifecycle,monkeypatch):
    f=lifecycle
    def start(thread): f.child(); raise RuntimeError('TEST ONLY thread activation')
    monkeypatch.setattr(FakeThread,'start',start)
    with pytest.raises(RuntimeError,match='activation') as raised:
        with f.w.lifecycle(): pytest.fail('activation must fail before body')
    assert f.signals and f.records[-1]['status']=='CLEANUP_FAILED'
    assert any('cleanup failed' in note for note in raised.value.__notes__)


@pytest.mark.parametrize('mode',['hung','stopped','join-error','state-error','stop-event'])
def test_bad_observer_cannot_bypass_cleanup_or_qualify(lifecycle,mode):
    f=lifecycle; f.w.start(); f.child()
    if mode=='hung':f.w.thread.hung=True
    if mode=='stopped':f.w.thread.alive=False
    if mode=='join-error':f.w.thread.join_error=True
    if mode=='state-error':f.w.thread.state_error=True
    if mode=='stop-event':f.w.stop.set()
    with pytest.raises(RuntimeError,match='cleanup'):f.w.finish()
    assert f.signals and f.records[-1]['status']=='CLEANUP_FAILED'
    assert f.records[-1]['cleanup_complete'] is False


def test_incomplete_cleanup_visible_and_original_failure_preserved(lifecycle):
    f=lifecycle; f.forbidden.add(424242); original=ValueError('TEST ONLY inference failure')
    with pytest.raises(ValueError) as raised:
        with f.w.lifecycle():f.child();raise original
    assert raised.value is original and raised.value.__notes__
    assert f.records[-1]['status']=='CLEANUP_FAILED' and f.records[-1]['owned_children_alive']
    assert {x[2] for x in f.signals}=={observer.signal.SIGTERM,observer.signal.SIGKILL}


@pytest.mark.parametrize('boundary',['gpu','progress','write'])
def test_optional_telemetry_or_write_failure_happens_after_owned_cleanup(lifecycle,monkeypatch,boundary):
    f=lifecycle; f.w.start(); f.child()
    def fail(*args,**kwargs):raise OSError('TEST ONLY diagnostic failure')
    monkeypatch.setattr(observer,{'gpu':'gpu_sample','progress':'progress_bytes','write':'dump'}[boundary],fail)
    if boundary=='write':
        with pytest.raises(OSError):f.w.finish()
    else:
        result=f.w.finish();assert result['telemetry_errors'] and result['cleanup_complete']
    assert f.signals and 424242 not in f.table


def test_write_failure_does_not_replace_primary_exception(lifecycle,monkeypatch):
    f=lifecycle; original=ValueError('TEST ONLY bootstrap')
    def fail(*args):raise OSError('TEST ONLY write failure')
    monkeypatch.setattr(observer,'dump',fail)
    with pytest.raises(ValueError) as raised:
        with f.w.lifecycle():f.child();raise original
    assert raised.value is original and f.signals and raised.value.__notes__


def test_actual_watchdog_loop_discovers_and_signals_owned_child(lifecycle,monkeypatch):
    f=lifecycle; f.w.start(); f.child(); f.clock[0]+=901
    monkeypatch.setattr(f.w.stop,'wait',lambda _:f.w.stop.set())
    f.w._loop()
    assert f.w.abort=='15min without output progress' and f.signals


def test_generated_notebook_covers_all_bootstrap_and_inference(payload):
    nb=payload[0]; setup=ast.parse(''.join(nb['cells'][0]['source']));body=ast.parse(''.join(nb['cells'][2]['source']))
    assert not any(isinstance(n,ast.Call) and ast.unparse(n.func)=='_h197_watch.start' for n in ast.walk(setup))
    assert len(body.body)==2 and isinstance(body.body[0],ast.With)
    scope=body.body[0];assert ast.unparse(scope.items[0].context_expr)=='_h197_watch.lifecycle()'
    assert len(scope.body)==1 and isinstance(scope.body[0],ast.If)
    calls=[ast.unparse(n.func) for n in ast.walk(scope) if isinstance(n,ast.Call)]
    assert 'zipfile.ZipFile' in calls and '_h197_stress.prepare' in calls and 'release.run_release' in calls
    assert ast.unparse(body.body[1].value.func)=='_h197.benchmark'


@pytest.mark.parametrize('field,value',[('schema','h197-observer/v1'),('cleanup_complete',False),('cleanup_errors',[{}]),
    ('observer_started',False),('observer_thread_stopped',False),('original_failure','ValueError')])
def test_result_collector_requires_successful_new_lifecycle(chain,field,value):
    spec=chain.read(chain.artifacts['n32']);record=chain.read(spec['observer']);record[field]=value
    spec['observer']=put(chain.root,'fixture-q/lifecycle-'+field+'.json',record)
    ref=put(chain.root,'fixture-q/run-lifecycle-'+field+'.json',spec)
    with pytest.raises(ValueError):c.execution(old.Inputs(chain.root),ref,chain.build,fixture=True)
