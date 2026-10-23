"""CPU-only crash-boundary, authentication and frozen-replay checks."""
import ast
import inspect
import json
import os
from pathlib import Path
import shutil
import types
import numpy as np
import pytest
from tools import exp169_release_c85r7codec_deadline_v122 as m
from tools import exp169_c85r7ins_checks_v122 as check
from tools import release_metadata_v122 as io

IDS = ['study02', 'study00', 'study01']
SITES = [
    'a172/ids.json', 'pr7/ids.json', 'a172/q-a5q-manifest.json',
    'a172/exp169-trio/ids.json', 'a172/exp169-trio/journal-identity.json',
    'a172/exp169-trio/receipt.json', 'a172/exp169-trio/binding.json',
    'a172/exp169-trio/journal-rows/000000001.json',
    'a172/q-journal/ids.json', 'a172/q-journal/journal-identity.json',
    'a172/q-journal/journal-rows/000000001.json', 'a172/exp169-executed-candidate.json',
    'pr7/pr7-run/r/ids.json', 'pr7/pr7-run/r/journal-identity.json',
    'pr7/pr7-run/r/journal-rows/000000001.json', 'pr7/pr7-run/r/receipt.json',
    'pr7/pr7-run/r7-identity.json', 'pr7/pr7-run/s_r7/provenance.json',
    'pr7/pr7-run/s_r7/identity.json', 'pr7/pr7-run/s_r7/invocation.json',
    'pr7/pr7-run/s_r7/r7_runtime/runtime_sha256.json',
    'pr7/pr7-run/s_r7/controller/worker0/ids.json',
    'pr7/pr7-run/s_r7/controller/worker0/receipt.json',
    'pr7/pr7-run/s_r7/controller/submission_receipt.json',
]

@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp('metadata-fixture')
    check.fixture(root, IDS)
    # Exercise the actual Q journal path with the recorded per-model values.
    adir = root / m.RUN / 'a172'
    with m.metadata_io.writers(m):
        j = m.c85.Journal(adir / 'q-journal', IDS, {'manifest_sha256':m.c85.MANIFEST_SHA})
    with np.load(adir / 'partner-raw.npz', allow_pickle=False) as raw:
        for i, uid in enumerate(IDS):
            for k, row in enumerate(np.concatenate([raw[n][:, i, :] for n in ('l2','dino','convnext')])):
                j.record(uid, str(k), row)
    _, _, failures = check.snapshot(adir, root / m.RUN / 'pr7')
    assert not any(failures.values()), 'healthy fixture silently fell back'
    return root

def copy(baseline, tmp_path):
    shutil.copytree(baseline / m.RUN, tmp_path / m.RUN)
    return tmp_path / m.RUN / 'a172', tmp_path / m.RUN / 'pr7'

@pytest.mark.parametrize('site', SITES)
@pytest.mark.parametrize('bad', [None, b'', b'{"interrupted":', b'null', b'{}', b'42'])
def test_missing_corrupt_each_boundary_publishes(baseline, tmp_path, site, bad):
    a,b = copy(baseline, tmp_path)
    target = tmp_path / m.RUN / site
    if bad is None:
        target.unlink(missing_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bad)
    receipt, payload = m.collect(a,b,tmp_path,cohort=IDS)
    ids, values = m.v5q._read_csv(tmp_path / 'submission.csv')
    assert ids == IDS and values.shape == (3,12) and np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    assert payload == (tmp_path / 'submission.csv').read_bytes()
    assert receipt['status'] == 'COMPLETE_C85R7'
    _, healthy, _ = check.snapshot(baseline / m.RUN / 'a172', baseline / m.RUN / 'pr7')
    _, recovered, failures = check.snapshot(a, b) if site not in ('a172/ids.json', 'pr7/ids.json') else (None, {}, {})
    for member in recovered:
        # Q's frozen internal ranks are recomputed over its surviving cohort.
        # Losing one Q row legitimately changes the other Q rows' ranks.
        if member == 'Q_A5q' and failures[member]:
            continue
        for i, uid in enumerate(IDS):
            if uid not in failures[member]:
                np.testing.assert_array_equal(recovered[member][i], healthy[member][i])
    # Empty R identity exactly reproduces the independent review's blocker.
    if '/r/' in site and bad is not None:
        assert receipt['pr7']['r_prediction_fallbacks'] == len(IDS)

@pytest.mark.parametrize('site', SITES + ['setup-ready.json','setup-fatal.json',
    'pr7/pr7-run/code-paths.json','insurance-events.json','own-supervised-exit.json',
    'codecs/install.json','codecs/gate-s_r7.json', 'nested-1.json', m.RECEIPT])
@pytest.mark.parametrize('boundary', ['fsync','replace'])
def test_atomic_interrupt_each_write_site(tmp_path, monkeypatch, site, boundary):
    path = tmp_path / site
    old = b'{"old":true}'
    io.atomic_write(path, old)
    def abort(*args):
        raise OSError('simulated interrupt')
    monkeypatch.setattr(io.os, boundary, abort)
    with pytest.raises(OSError):
        io.atomic_write(path, b'{"new":true}')
    assert path.read_bytes() == old
    path.unlink()
    with pytest.raises(OSError):
        io.atomic_write(path, b'{"new":true}')
    assert not path.exists()
    assert not list(path.parent.glob('.*.tmp'))

@pytest.mark.parametrize('site,key', [
    ('pr7/pr7-run/r/journal-identity.json','checkpoint_sha256'),
    ('a172/q-journal/journal-identity.json','manifest_sha256'),
    ('pr7/pr7-run/s_r7/provenance.json','dataset_slug'),
    ('a172/q-a5q-manifest.json','candidate')])
def test_committed_identity_mismatch_stays_fatal(baseline,tmp_path,site,key):
    a,b=copy(baseline,tmp_path)
    path=tmp_path/m.RUN/site
    value=json.loads(path.read_text()); value[key]='wrong-committed-identity'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): m.collect(a,b,tmp_path,cohort=IDS)
    assert not (tmp_path/'submission.csv').exists()

@pytest.mark.parametrize('site', ['a172/ids.json','pr7/pr7-run/r/ids.json',
                                 'pr7/pr7-run/s_r7/controller/worker0/ids.json'])
def test_committed_assignment_mismatch_stays_fatal(baseline,tmp_path,site):
    a,b=copy(baseline,tmp_path)
    (tmp_path/m.RUN/site).write_text(json.dumps(['wrong-study']))
    with pytest.raises(ValueError): m.collect(a,b,tmp_path,cohort=IDS)
    assert not (tmp_path/'submission.csv').exists()

@pytest.mark.parametrize('boundary', ['staged','published'])
def test_receipt_interrupt_keeps_verified_csv(baseline,tmp_path,monkeypatch,boundary):
    a,b=copy(baseline,tmp_path)
    original = io.os.replace
    def fail(src,dst):
        if Path(dst).name == m.RECEIPT and ((m.STAGE in Path(dst).parts) == (boundary == 'staged')):
            raise OSError('receipt interrupt')
        return original(src,dst)
    monkeypatch.setattr(io.os,'replace',fail)
    receipt,payload=m.collect(a,b,tmp_path,cohort=IDS)
    assert (tmp_path/'submission.csv').read_bytes()==payload
    assert receipt['telemetry']['receipt_write_error']=='OSError'

def test_healthy_replay_and_frozen_payloads(tmp_path):
    result=m.cpu_replay(tmp_path/'replay')
    assert result['v110_csv_identical'] and result['v110_p_branch_identical']
    assert result['diagnostic_sha256']=='a59be5c6066f87b5309e242ecced132fab1894b8dd4b69169e8450a171ddee8a'
    old=m.old.source_payloads(); new=m.source_payloads()
    assert all(new[k]==v for k,v in old.items())
    assert m.blend is m.old.blend and m.p_branch is m.old.p_branch

def test_atomic_source_all_syntaxes(tmp_path,monkeypatch):
    source='''from pathlib import Path
import json
p = Path(folder)
(p/'a.json').write_text(json.dumps({'x':1}),encoding='utf-8')
(p/'b.json').write_bytes(b'{}')
json.dump({'x':2},open(p/'c.json','w'))
with open(p/'d.json','w',encoding='utf-8') as f:
    json.dump({'x':3},f)
'''
    transformed=io.atomic_source(source)
    import release_metadata_v122 as runtime_io
    called=[]; original=runtime_io.os.fsync
    monkeypatch.setattr(runtime_io.os,'fsync',lambda fd:(called.append(fd),original(fd))[1])
    exec(transformed,{'folder':str(tmp_path)})
    assert len(called)==4
    assert all(json.loads(p.read_text()) is not None for p in tmp_path.glob('*.json'))

def test_runtime_generated_metadata_is_atomic(tmp_path):
    with io.writers(m):
        runtime=m.r7.diagnostic_source(m.r7.SOURCE,tmp_path)
        for name in ('probe_dual.py','compact_infer.py','px_cache.py','knee_anchor_bank_v2.py'):
            source=(runtime/name).read_text()
            tree=ast.parse(source)
            assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                           and n.func.attr in ('write_text','write_bytes') for n in ast.walk(tree))
            assert '_metadata_' in source
        hook=m.codecs.hook_source(tmp_path)
        assert 'handle.flush()' in hook and '_hb126_os.fsync(handle.fileno())' in hook
        compile(hook,'<hook>','exec')


def test_own_q_generated_metadata_is_atomic_and_adapters_restore():
    before = dict(m.r7.__dict__)
    files = m.old.source_payloads()
    for source in [m.serving_source(files['public-serving.py'].decode()), *[
            m.a172_script(files['a172-cell.py.txt'].decode(), Path('/kaggle/working/check'), arm)
            for arm in ('own', 'Q_A5q')]]:
        tree = ast.parse(source)
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr in ('write_text', 'write_bytes', 'dump') for n in ast.walk(tree))
        compile(source, '<atomic-own-q>', 'exec')
    with io.writers(m):
        assert m.r7._metadata_text is io.atomic_text
    assert m.r7.__dict__ == before

def test_expired_deadline_and_seed_metadata_failure(tmp_path,monkeypatch):
    monkeypatch.setenv('C85_COMPETITION',str(check.VISIBLE))
    monkeypatch.setenv(m.insurance.FAULT_ENV,'off')
    monkeypatch.setenv(m.insurance.START_ENV, '0')
    monkeypatch.setenv(m.insurance.BUDGET_ENV, '1')
    rt=tmp_path/'runtime'; rt.mkdir()
    (rt/'q-a5q-manifest.json').write_bytes(m.old.source_payloads()['q-a5q-manifest.json'])
    monkeypatch.setattr(m,'runtime_pins',lambda path:None)
    original=io.os.replace
    def fail(src,dst):
        if Path(dst).name in ('ids.json','q-a5q-manifest.json'):
            raise OSError('interrupted seed')
        return original(src,dst)
    monkeypatch.setattr(io.os,'replace',fail)
    out=tmp_path/'run'
    receipt=m.run_release(rt,out)
    ids,values=m.v5q._read_csv(out/'submission.csv')
    assert ids and values.shape==(len(ids),12) and np.all(values==.5)
    assert receipt['status']=='COMPLETE_C85R7'


@pytest.mark.parametrize('name', ['ids.json', 'journal-identity.json', '000000001.json'])
@pytest.mark.parametrize('branch', ['own', 'R', 'Q_A5q'])
@pytest.mark.parametrize('boundary', ['fsync', 'replace'])
def test_actual_journal_commit_interruption(baseline, tmp_path, monkeypatch, name, branch, boundary):
    a,b=copy(baseline,tmp_path)
    directory = (a/'q-journal' if branch=='Q_A5q' else check.journal_path(tmp_path,branch))
    shutil.rmtree(directory)
    identity = ({'manifest_sha256':m.c85.MANIFEST_SHA} if branch=='Q_A5q' else
                {'checkpoint_sha256':m.p.R_SHA,'source_pins':m.p.PINS} if branch=='R' else
                m.c85.manifest(a/'q-a5q-manifest.json')['anchor_binding'])
    original=io.os.replace if boundary=='replace' else io.os.fsync
    calls=[]
    # fsync has no path argument; stop the corresponding initialization/row commit.
    index=['ids.json','journal-identity.json','000000001.json'].index(name)+1
    def stop(*args):
        calls.append(args)
        if (boundary=='replace' and Path(args[1]).name==name) or (boundary=='fsync' and len(calls)==index):
            raise OSError('interrupted actual journal commit')
        return original(*args)
    with monkeypatch.context() as patch:
        patch.setattr(io.os,boundary,stop)
        with pytest.raises(OSError), io.writers(m):
            journal=m.c85.Journal(directory,IDS,identity)
            journal.record(IDS[0], '0' if branch=='Q_A5q' else check.members(branch)[0], [.25]*12)
    receipt,payload=m.collect(a,b,tmp_path,cohort=IDS)
    assert (tmp_path/'submission.csv').read_bytes()==payload
    assert receipt['status']=='COMPLETE_C85R7'


def test_cleanup_audit_mocked_only(monkeypatch):
    from tools import release_cleanup_checks_v121 as cleanup
    root=(os.getpid(),1.)
    child=(999999,2.)
    state={root:dict(pid=root[0],username='test')}
    memory={'GPU-test':0}; allocations={}
    audit=cleanup.CleanupAudit(process_snapshot=lambda:dict(state),
                               gpu_snapshot=lambda:(dict(memory),dict(allocations)))
    state[child]=dict(pid=child[0],username='test')
    audit.sample()
    state.pop(child)
    now=cleanup.time.monotonic()
    assert audit.verify(now)['descendants_observed']==1
    state[child]=dict(pid=child[0],username='test')
    with pytest.raises(AssertionError,match='remain'):
        audit.verify(now-1201)
    state.pop(child); memory['GPU-test']=1
    with pytest.raises(AssertionError,match='remain'):
        audit.verify(now-1201)
    memory['GPU-test']=0
    with pytest.raises(AssertionError,match='exceeded'):
        audit.verify(now-1201)


def test_actual_provenance_commit_is_atomic(tmp_path,monkeypatch):
    ident={k:{} for k in ('file_sha256','declared','features')}
    ident.update(manifest_sha256='test',manifest_pinned=True)
    code={k:{} for k in ('files','closure_sha256','r7_closure_sha256')}
    code['tree_sha256']='test'
    path=tmp_path/'provenance.json'
    with io.writers(m):
        m.r7.write_provenance(path,identity=ident,code=code,args=[])
        before=path.read_bytes()
        monkeypatch.setattr(io.os,'replace',lambda *a: (_ for _ in ()).throw(OSError('interrupted')))
        with pytest.raises(OSError): m.r7.write_provenance(path,identity=ident,code=code,args=[])
        assert path.read_bytes()==before


RECEIPT_STATES = ['missing', 'truncated', 'corrupt', 'incomplete', 'stale_identity']


def worker_receipt(worker, failures):
    ids = m.read_json(worker / 'ids.json')
    return dict(status='PASS', studies=len(ids), scored=len(ids)-len(failures),
                fallback=failures, publication_binding=io.worker_binding(worker))


def damage_receipt(path, state, value):
    if state == 'missing':
        path.unlink(missing_ok=True)
        return
    if state == 'truncated': data = b'{"fallback":'
    elif state == 'corrupt': data = b'\xff\x00'
    elif state == 'incomplete': data = b'{"interrupted":true}'
    else:
        if 'publication_binding' in value:
            value['publication_binding']['provenance'] = 'stale'
        else:
            value['journal_identity'] = dict(value['journal_identity'])
            key = next(iter(value['journal_identity']))
            old = value['journal_identity'][key]
            value['journal_identity'][key] = ({**old, 'stale': 'identity'} if isinstance(old, dict) else 'stale')
        data = m.json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.mark.parametrize('branch', ['R', 'R7', 'own'])
@pytest.mark.parametrize('state', RECEIPT_STATES)
def test_receipt_state_never_upgrades_failed_branch(baseline, tmp_path, branch, state):
    a, b = copy(baseline, tmp_path)
    if branch == 'R7':
        directory = b / 'pr7-run/s_r7/controller/worker0'
        record = worker_receipt(directory, dict.fromkeys(IDS, 'failed inference'))
        expected_members = ['S_R7']
    else:
        directory = check.journal_path(tmp_path, branch)
        rows = directory / 'journal-rows'
        # Retain the original healthy rows, append failure followed by stale success.
        # Failure must be sticky regardless of order or optional aggregate receipt.
        index = len(list(rows.glob('*.json')))
        for uid in IDS:
            for member in check.members(branch):
                for row in (dict(uid=uid, member=member, error='failed inference'),
                            dict(uid=uid, member=member, values=[.87]*12)):
                    index += 1
                    (rows/f'{index:09d}.json').write_text(json.dumps(row)+'\n')
        record = dict(status='RECORDED', failures=dict.fromkeys(IDS, 'failed inference'),
                      journal_identity=m.read_json(directory/'journal-identity.json'), assignment=IDS)
        expected_members = check.members(branch)
    path = directory / 'receipt.json'
    path.write_bytes(m.json_bytes(record))
    before, expected = m.collect(a, b, tmp_path/'known-failed', cohort=IDS)
    _, _, failures = check.snapshot(a,b)
    assert all(set(failures[member]) == set(IDS) for member in expected_members)
    damage_receipt(path, state, record)
    if state == 'stale_identity':
        with pytest.raises(ValueError, match='identity'):
            m.collect(a,b,tmp_path/'recovered',cohort=IDS)
        assert not (tmp_path/'recovered/submission.csv').exists()
        return
    after, actual = m.collect(a,b,tmp_path/'recovered',cohort=IDS)
    assert actual == expected
    assert after['fallbacks'] == before['fallbacks']
    _, _, failures = check.snapshot(a,b)
    assert all(set(failures[member]) == set(IDS) for member in expected_members)


@pytest.mark.parametrize('state', RECEIPT_STATES[:-1])
def test_r7_partial_failure_with_lost_receipt_fails_assigned_worker(baseline,tmp_path,state):
    a,b=copy(baseline,tmp_path)
    worker=b/'pr7-run/s_r7/controller/worker0'
    # Establish the full-worker-failed oracle with the frozen fallback arithmetic.
    record=worker_receipt(worker,dict.fromkeys(IDS,'failed'))
    (worker/'receipt.json').write_bytes(m.json_bytes(record))
    failed, expected=m.collect(a,b,tmp_path/'failed',cohort=IDS)
    record=worker_receipt(worker,{IDS[0]:'failed'})
    (worker/'receipt.json').write_bytes(m.json_bytes(record))
    partial,_=m.collect(a,b,tmp_path/'partial',cohort=IDS)
    assert partial['pr7']['s_r7_prediction_fallbacks']==1
    damage_receipt(worker/'receipt.json',state,record)
    recovered,actual=m.collect(a,b,tmp_path/'recovered',cohort=IDS)
    assert recovered['fallbacks']==failed['fallbacks'] and actual==expected
    assert recovered['pr7']['s_r7_prediction_fallbacks']==len(IDS)


@pytest.mark.parametrize('member', ['R','EXP143v2','EXP155'])
@pytest.mark.parametrize('bad_row', ['error_then_success','invalid_then_success','values_and_error'])
def test_journal_failure_is_sticky(baseline,tmp_path,member,bad_row):
    a,b=copy(baseline,tmp_path)
    branch='R' if member=='R' else 'own'
    directory=check.journal_path(tmp_path,branch)
    rows=directory/'journal-rows'
    index=len(list(rows.glob('*.json')))
    failure=dict(uid=IDS[0],member=member,error='failed')
    if bad_row=='invalid_then_success': failure=dict(uid=IDS[0],member=member,values=[-1]*12)
    if bad_row=='values_and_error': failure['values']=[.99]*12
    for i,row in enumerate([failure,dict(uid=IDS[0],member=member,values=[.99]*12)],index+1):
        (rows/f'{i:09d}.json').write_text(json.dumps(row)+'\n')
    _,values,failures=check.snapshot(a,b)
    assert set(failures[member])=={IDS[0]}
    assert np.all(values[member][0]==.5)


@pytest.mark.parametrize('state', ['missing','truncated','corrupt','incomplete','stale_identity'])
@pytest.mark.parametrize('branch', ['R','own'])
def test_legacy_raw_receipt_required(tmp_path,branch,state):
    a=tmp_path/'a172'; b=tmp_path/'pr7'
    # Whitelist only recorded prediction and metadata inputs.
    for name in ('q-a5q-manifest.json','partner-raw.npz','exp169-executed-candidate.json',
                 'exp169-trio/raw.npz','exp169-trio/binding.json','exp169-trio/receipt.json'):
        src=m.c85.A5Q/name
        if name=='q-a5q-manifest.json' and not src.exists(): src=m.c85.A5Q/'exp169-partner/release-manifest.json'
        (a/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,a/name)
    m.base.stage_replay_input(b)
    cohort=m.c85.authenticated_a172(a)[0]
    path=(b/'pr7-run/r/receipt.json' if branch=='R' else a/'exp169-trio/receipt.json')
    record=m.read_json(path)
    if state=='stale_identity':
        if branch=='R': record['raw_sha256']='0'*64
        else: record['files']['raw.npz']='0'*64
        path.write_bytes(m.json_bytes(record))
        with pytest.raises(ValueError): m.collect(a,b,tmp_path/'recovered',cohort=cohort)
        assert not (tmp_path/'recovered/submission.csv').exists()
        return
    # No prediction artifact is the reference failed branch, using identical survivors.
    raw=path.with_name('raw.npz')
    raw.rename(raw.with_suffix('.saved'))
    reference,expected=m.collect(a,b,tmp_path/'failed',cohort=cohort)
    raw.with_suffix('.saved').rename(raw)
    damage_receipt(path,state,record)
    result,actual=m.collect(a,b,tmp_path/'recovered',cohort=cohort)
    assert result['fallbacks']==reference['fallbacks'] and actual==expected


def test_worker_binding_in_actual_generated_code(tmp_path):
    with io.writers(m):
        runtime=m.r7.diagnostic_source(m.r7.SOURCE,tmp_path)
    source=(runtime/'compact_infer.py').read_text()
    assert "receipt['publication_binding'] = worker_binding(out)" in source
    assert source.index("receipt['publication_binding']") < source.index("_metadata_text(out / 'receipt.json'")


def test_missing_receipt_is_local_to_worker(baseline,tmp_path):
    a,b=copy(baseline,tmp_path)
    parent=b/'pr7-run/s_r7/controller'
    first=parent/'worker0'; second=parent/'worker1'; second.mkdir()
    _,values=m.v5q._read_csv(first/'predictions.csv')
    for worker,part,pred in [(first,IDS[:1],values[:1]),(second,IDS[1:],values[1:])]:
        (worker/'ids.json').write_bytes(m.json_bytes(part))
        (worker/'predictions.csv').write_bytes(m.p.csv_bytes(part,pred))
        (worker/'receipt.json').write_bytes(m.json_bytes(worker_receipt(worker,{})))
    (first/'receipt.json').unlink()
    _,_,failures=check.snapshot(a,b)
    assert set(failures['S_R7'])==set(IDS[:1])


@pytest.mark.parametrize('field',['status','studies','scored','fallback','publication_binding'])
def test_each_worker_receipt_field_is_required(baseline,tmp_path,field):
    a,b=copy(baseline,tmp_path)
    worker=b/'pr7-run/s_r7/controller/worker0'
    record=worker_receipt(worker,{})
    record.pop(field)
    (worker/'receipt.json').write_bytes(m.json_bytes(record))
    _,_,failures=check.snapshot(a,b)
    assert set(failures['S_R7'])==set(IDS)


@pytest.mark.parametrize('branch',['R','own'])
def test_actual_serving_receipt_is_bound(baseline,tmp_path,branch):
    a,b=copy(baseline,tmp_path)
    directory=check.journal_path(tmp_path,branch)
    binding=types.SimpleNamespace(work=directory,members=check.members(branch),failures={})
    with io.writers(m):
        m.c85.ServingBinding.finish_receipt(binding,directory,0.)
    record=m.read_json(directory/'receipt.json')
    assert record['journal_identity']==m.read_json(directory/'journal-identity.json')
    assert record['assignment']==IDS
    assert not any(check.snapshot(a,b)[2].values())


@pytest.mark.parametrize('artifact',['ids.json','predictions.csv','../../provenance.json'])
def test_bound_worker_artifact_mutation_is_fatal(baseline,tmp_path,artifact):
    a,b=copy(baseline,tmp_path)
    worker=b/'pr7-run/s_r7/controller/worker0'
    path=worker/artifact
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='identity/output'):
        m.collect(a,b,tmp_path/'recovered',cohort=IDS)
    assert not (tmp_path/'recovered/submission.csv').exists()
