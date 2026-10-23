"""Offline admission tests. No Kaggle, SSH, CUDA, or real process termination."""
import base64
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / '.local/run-scripts/b-preflight-v1.py'
spec = importlib.util.spec_from_file_location('b_preflight_v1', SCRIPT)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
PY = sys.executable.replace('/', '\\')
NOW = 1_800_000_000


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding='utf-8')
    return path


def proc(pid=101, parent=1, name='bash.exe', command='bash resource-sentinel-v1.sh', created='stamp'):
    return dict(pid=pid, parent=parent, name=name, command=command, created=created)


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    def forbidden(*a, **kw):
        pytest.fail('unmocked subprocess or termination')
    monkeypatch.setattr(b, 'run', forbidden)
    monkeypatch.setattr(b, 'kill_verified', forbidden)
    monkeypatch.setattr(b, 'processes', lambda: [proc()])
    monkeypatch.setattr(b.time, 'time', lambda: NOW)


def gpu(monkeypatch, apps=(), free=13000, cards=None):
    monkeypatch.setattr(b, 'smi', lambda q: list(apps) if 'compute-apps' in q else
                        (cards if cards is not None else [['GPU-1', '4070 desktop', str(free)]]))


def smoke(tmp_path, **changes):
    return put(tmp_path/'smoke.json', dict(status='PASS', full_shape=True, includes_tta_flip=True,
        gpu_name='4070 desktop', tag='test', command=[PY, 'fit.py'], peak_reserved_bytes=8*b.GIB, **changes))


def test_gpu_pass_desktop_apps_and_exact_reserve(tmp_path, monkeypatch, capsys):
    gpu(monkeypatch, apps=[['GPU-1', '900', 'explorer.exe'], ['GPU-2', '901', 'python.exe']], free=10240)
    p = smoke(tmp_path)
    assert b.main(['gpu-local', '--smoke', str(p), 'test', PY, 'fit.py']) == 0
    assert 'measured VRAM plus 2 GiB' in capsys.readouterr().out


def test_gpu_no_smoke_labels_limited_scope(monkeypatch, capsys):
    gpu(monkeypatch)
    assert b.main(['gpu-local', 'test', '--', PY, 'fit.py']) == 0
    assert 'NOT fit qualification' in capsys.readouterr().out


@pytest.mark.parametrize('name', ['python.exe', 'pythonw.exe', 'python3.12', 'Python.EXE'])
def test_gpu_reject_python(monkeypatch, name):
    gpu(monkeypatch, apps=[['GPU-1', '900', name]])
    assert b.main(['gpu-local', 'test', PY]) == 2


def test_gpu_resolve_wddm_permissions_through_win32(monkeypatch):
    gpu(monkeypatch, apps=[['GPU-1', '900', '[Insufficient Permissions]']])
    monkeypatch.setattr(b, 'processes', lambda: [proc(), proc(900, name='python.exe')])
    assert b.main(['gpu-local', 'test', PY]) == 2
    monkeypatch.setattr(b, 'processes', lambda: [proc(), proc(900, name='explorer.exe')])
    assert b.main(['gpu-local', 'test', PY]) == 0
    monkeypatch.setattr(b, 'processes', lambda: [proc()])
    assert b.main(['gpu-local', 'test', PY]) == 2


@pytest.mark.parametrize('case', ['interpreter', 'sentinel', 'card', 'free', 'tag'])
def test_gpu_admission_failures(monkeypatch, case):
    gpu(monkeypatch, cards=[] if case == 'card' else None, free='NaN' if case == 'free' else 13000)
    if case == 'sentinel':
        monkeypatch.setattr(b, 'processes', lambda: [])
    assert b.main(['gpu-local', 'bad/tag' if case == 'tag' else 'test',
                   'python' if case == 'interpreter' else PY]) == 2


@pytest.mark.parametrize('field,value', [('status','FAIL'), ('full_shape',False), ('includes_tta_flip',False),
    ('gpu_name','other'), ('tag','other'), ('command',[PY]), ('peak_reserved_bytes',0),
    ('peak_reserved_bytes',float('nan')), ('peak_reserved_bytes',11*b.GIB)])
def test_gpu_smoke_binding(tmp_path, monkeypatch, field, value):
    gpu(monkeypatch)
    p = smoke(tmp_path)
    data = json.loads(p.read_text()); data[field] = value; put(p, data)
    assert b.main(['gpu-local', '--smoke', str(p), 'test', PY, 'fit.py']) == 2


def test_gpu_insufficient_free_memory(tmp_path, monkeypatch):
    gpu(monkeypatch, free=10239)
    assert b.main(['gpu-local', '--smoke', str(smoke(tmp_path)), 'test', PY, 'fit.py']) == 2


def vast_ready(monkeypatch, **changes):
    result = dict(locks_busy=[], gpu_pids=[], pins_lf=True, pins_count=2, free_bytes=51*b.GIB)
    result.update(changes)
    monkeypatch.setattr(b, 'processes', lambda: [proc(command='bash vast-idle-watch-v1.sh 19960')])
    monkeypatch.setattr(b, 'remote_probe', lambda a: result)


def test_vast_pass(monkeypatch):
    vast_ready(monkeypatch)
    assert b.main(['vast', '19960', '/workspace/package']) == 0


@pytest.mark.parametrize('change', [dict(locks_busy=['/fit.lock']), dict(gpu_pids=[100]), dict(pins_lf=False),
    dict(pins_count=0), dict(free_bytes=49*b.GIB)])
def test_vast_blockers(monkeypatch, change):
    vast_ready(monkeypatch, **change)
    assert b.main(['vast', '19960', '/workspace/package']) == 2


def test_vast_wrong_watch_port(monkeypatch):
    vast_ready(monkeypatch)
    assert b.main(['vast', '19961', '/workspace/package']) == 2


def test_remote_transport_quotes_paths_and_pins_trust(monkeypatch):
    calls=[]
    monkeypatch.setattr(b, 'run', lambda argv, **kw: calls.append((argv, kw)) or '{}')
    args=b.parser().parse_args(['vast','19960',"/workspace/a'; echo bad", '--fit-lock','/workspace/a lock'])
    assert b.remote_probe(args) == {}
    argv, kw=calls[0]
    assert argv[:3] == ['ssh','-o','BatchMode=yes']
    assert 'StrictHostKeyChecking=yes' in argv
    import shlex
    assert shlex.split(argv[-1]) == ['python3','-',args.remote_dir,json.dumps(args.fit_lock)]
    assert kw['input'] == b.REMOTE_PROBE


@pytest.mark.parametrize('argv', [['vast','0','/x'], ['vast','22','relative'], ['vast','22','/x','--host=-bad'],
                                 ['vast','22','/x','--fit-lock','relative']])
def test_remote_bad_arguments(argv):
    with pytest.raises(b.CheckFailed):
        b.remote_probe(b.parser().parse_args(argv))


@pytest.mark.parametrize('crlf,busy,gpu_pid', [(False,False,''), (True,False,''), (False,True,'123')])
def test_actual_remote_program_cpu_fixture(tmp_path, monkeypatch, capsys, crlf, busy, gpu_pid):
    # Execute the Linux probe body with only fcntl/nvidia telemetry replaced.
    (tmp_path/'pin.sha256').write_bytes(('0'*64+'  weights.pt\n').encode())
    (tmp_path/'SHA256SUMS').write_bytes(('0'*64+'  pin.sha256'+ ('\r\n' if crlf else '\n')).encode())
    put(tmp_path/'fit.lock', '')
    def flock(fd, mode):
        if busy and mode == 3:
            raise BlockingIOError()
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,LOCK_UN=4,flock=flock))
    monkeypatch.setattr(sys,'argv',['probe',str(tmp_path),json.dumps([str(tmp_path/'fit.lock')])])
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw: SimpleNamespace(stdout=gpu_pid))
    exec(compile(b.REMOTE_PROBE,'remote_probe','exec'),{})
    out=json.loads(capsys.readouterr().out)
    assert out['pins_lf'] is not crlf
    assert bool(out['locks_busy']) is busy
    assert out['gpu_pids'] == ([123] if gpu_pid else [])


def stage_evidence(tmp_path):
    put(tmp_path/'tools/kaggle_submit_once.py',
        (SCRIPT.parents[2]/'tools/kaggle_submit_once.py').read_text(encoding='utf-8'))
    stage=tmp_path/'stage'
    meta=put(stage/'kernel/kernel-metadata.json',dict(id='owner/test-slug',title='test-slug'))
    put(stage/'kernel/main.py','print(1)\n')
    put(stage/'registration.json',dict(version='1.0.0'))
    files=list(stage.rglob('*'))
    hashes={p.relative_to(tmp_path).as_posix():b.sha(p) for p in files if p.is_file()}
    for p in files:
        if p.is_file(): os.utime(p,(NOW-30,NOW-30))
    gate=put(tmp_path/'.local/state/gates/test-slug.json',dict(gate_version='1.1.0',status='PASS',errors=[],
        kernel_id='owner/test-slug',stage=str(stage),input_sha256=hashes))
    os.utime(gate,(NOW-20,NOW-20))
    quota=put(tmp_path/'.local/state/kaggle-quota.json',dict(read_at=datetime.fromtimestamp(NOW-10,timezone.utc).isoformat(),
        remaining_hours=2,source='kaggle.com/settings'))
    os.utime(quota,(NOW-10,NOW-10))
    return stage,meta,gate,quota


@pytest.mark.parametrize('version', ['1.0.0', '1.1.0', '1.2.0'])
def test_push_pass(tmp_path, version):
    stage,meta,gate,quota=stage_evidence(tmp_path)
    receipt=b.read(gate)
    receipt['gate_version']=version
    put(gate,receipt); os.utime(gate,(NOW-20,NOW-20))
    assert b.main(['kaggle-push',str(stage),'--expected-hours','1']) == 0


@pytest.mark.parametrize('case', ['gate_fail', 'wrong_stage', 'unsupported_version'])
def test_gate_v120_rejections(tmp_path, case):
    stage,meta,gate,quota=stage_evidence(tmp_path)
    receipt=b.read(gate)
    receipt['gate_version']='1.2.0'
    if case=='gate_fail': receipt['status']='FAIL'
    if case=='wrong_stage': receipt['stage']=str(tmp_path/'source-stage')
    if case=='unsupported_version': receipt['gate_version']='1.3.0'
    put(gate,receipt); os.utime(gate,(NOW-20,NOW-20))
    with pytest.raises(b.CheckFailed, match='strict_variant_gate receipt mismatch/FAIL'):
        b.gate_check(stage,b.read(meta))


@pytest.mark.parametrize('case', ['title','gate_missing','stale','hash','unbound','gate_fail','wrong_stage',
    'wrong_kernel','fixture','quota_missing','quota_stale','quota_future','quota_mtime','quota_small','quota_source',
    'quota_timezone','expected_zero','waiter','push'])
def test_push_failures(tmp_path, monkeypatch, case):
    stage,meta,gate,quota=stage_evidence(tmp_path)
    g=b.read(gate); q=b.read(quota)
    if case=='title': put(meta,dict(id='owner/test-slug',title='owner/test-slug'))
    if case=='gate_missing': gate.unlink()
    if case=='stale': os.utime(stage/'registration.json',(NOW-20,NOW-20))
    if case=='hash': put(stage/'kernel/main.py','drift'); os.utime(stage/'kernel/main.py',(NOW-30,NOW-30))
    if case=='unbound':
        p=put(stage/'kernel/extra.py','extra'); os.utime(p,(NOW-30,NOW-30))
    if case in ('gate_fail','wrong_stage','wrong_kernel','fixture'):
        key,value={'gate_fail':('status','FAIL'),'wrong_stage':('stage',str(tmp_path)),
                   'wrong_kernel':('kernel_id','owner/wrong'),'fixture':('fixture_only',True)}[case]
        g[key]=value; put(gate,g); os.utime(gate,(NOW-20,NOW-20))
    if case=='quota_missing': quota.unlink()
    if case in ('quota_stale','quota_future','quota_timezone'):
        q['read_at'] = (datetime.fromtimestamp(NOW-900 if case=='quota_stale' else NOW+10,timezone.utc).isoformat()
                        if case!='quota_timezone' else '2027-01-15T08:00:00')
    if case=='quota_small': q['remaining_hours']=1.99
    if case=='quota_source': q['source']='estimate'
    if case.startswith('quota_') and case!='quota_missing':
        put(quota,q); os.utime(quota,(NOW-900 if case=='quota_mtime' else NOW-10,)*2)
    if case in ('waiter','push'):
        command='bash kaggle-visible-wait-v2.sh' if case=='waiter' else '"C:\\bin\\kaggle.exe" kernels push -p stage'
        monkeypatch.setattr(b,'processes',lambda:[proc(command=command)])
    assert b.main(['kaggle-push',str(stage),'--expected-hours','0' if case=='expected_zero' else '1']) == 2


def submission(tmp_path):
    receipt=put(tmp_path/'visible.json',dict(status='COMPLETE_C85R7',kernel='owner/slug',kernel_version=2))
    review=put(tmp_path/'review.md','owner/slug\nVERDICT: PASS\n')
    ledger=put(tmp_path/'ledger.jsonl','')
    return ['submit','owner/slug','2','--receipt',str(receipt),'--review',str(review),'--ledger',str(ledger)]


def test_submit_pass(tmp_path):
    assert b.main(submission(tmp_path)) == 0


@pytest.mark.parametrize('case', ['attempt','resolved','unconfirmed','malformed','review_fail','wrong_slug',
    'review_suffix','two_verdicts','incomplete','wrong_version','wrong_receipt','nonobject'])
def test_submit_rejections(tmp_path, case):
    argv=submission(tmp_path)
    if case in ('attempt','resolved','unconfirmed'):
        put(tmp_path/'ledger.jsonl',dict(kernel='owner/slug',version='2',attempt='a' if case=='attempt' else None,
            outcome='SUBMITTED' if case=='attempt' else 'RESOLVED_ABSENT' if case=='resolved' else 'UNCONFIRMED'))
    if case=='malformed': put(tmp_path/'ledger.jsonl','{')
    reviews={'review_fail':'owner/slug\nVERDICT: FAIL','wrong_slug':'owner/slug-other\nVERDICT: PASS',
             'review_suffix':'owner/slug\nVERDICT: PASS\nmore','two_verdicts':'owner/slug\nVERDICT: FAIL\nVERDICT: PASS'}
    if case in reviews: put(tmp_path/'review.md',reviews[case])
    if case in ('incomplete','wrong_version','wrong_receipt','nonobject'):
        data=b.read(tmp_path/'visible.json')
        if case=='incomplete': data['status']='INCOMPLETE'
        if case=='wrong_version': data['kernel_version']=3
        if case=='wrong_receipt': data['kernel']='owner/other'
        put(tmp_path/'visible.json',[] if case=='nonobject' else data)
    assert b.main(argv) == 2


def test_submit_sidecar_and_tamper(tmp_path):
    argv=submission(tmp_path)
    put(tmp_path/'visible.json',dict(status='COMPLETE_C85R7'))
    put(tmp_path/'submission-binding.json',dict(schema_version='1.0.0',kernel='owner/slug',kernel_version=2,
        receipt_sha256=b.sha(tmp_path/'visible.json')))
    assert b.main(argv) == 0
    put(tmp_path/'visible.json',dict(status='COMPLETE_C85R7',kernel='owner/wrong'))
    assert b.main(argv) == 2


def stop_rows():
    return [proc(os.getpid(),500,'python.exe','checker'), proc(500,1,'powershell.exe','parent'),
            proc(101,1,command='bash unique-test-loop.sh'), proc(102,101,'python.exe','python child.py'),
            proc(103,102,'python.exe','grandchild'), proc(104,1,'python.exe','unrelated')]


def test_stop_only_tree_and_roots_first(monkeypatch):
    rows=stop_rows(); killed=[]
    monkeypatch.setattr(b,'processes',lambda: [p for p in rows if p['pid'] not in killed])
    monkeypatch.setattr(b,'kill_verified',lambda p:killed.append(p['pid']))
    assert b.main(['stop-loop','unique-test-loop.sh','--root-pid','101']) == 0
    assert killed==[101,102,103]


def test_stop_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(b,'processes',stop_rows)
    assert b.main(['stop-loop','unique-test-loop.sh','--dry-run']) == 0
    assert '3 processes would be stopped' in capsys.readouterr().out


@pytest.mark.parametrize('case',['broad','wrong_pid','agent','unverifiable','survivor','new_child','identity_race'])
def test_stop_fail_closed(monkeypatch, case):
    rows=stop_rows(); calls=[]
    if case=='agent': rows[3]['name']='codex.exe'
    if case=='unverifiable': rows[3]['created']=None
    def snapshots():
        if calls and case=='new_child': return [proc(888,101,command='child')]
        return rows
    monkeypatch.setattr(b,'processes',snapshots)
    def kill(p):
        if case=='identity_race': raise b.CheckFailed('process identity changed')
        calls.append(p['pid'])
    monkeypatch.setattr(b,'kill_verified',kill)
    argv=['stop-loop','bash' if case=='broad' else 'unique-test-loop.sh']
    if case=='wrong_pid': argv+=['--root-pid','104']
    assert b.main(argv) == 2
    if case in ('broad','wrong_pid','agent','unverifiable'): assert calls==[]


def test_stop_protects_ancestors(monkeypatch):
    rows=stop_rows()[:2]; rows[1]['command']='bash unique-test-loop.sh'
    monkeypatch.setattr(b,'processes',lambda:rows)
    assert b.main(['stop-loop','unique-test-loop.sh','--root-pid','500']) == 2


def test_kill_script_rechecks_creation_and_command(monkeypatch):
    # Recover the implementation replaced by the autouse no-kill guard.
    original=importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    calls=[]; monkeypatch.setattr(original,'powershell',calls.append)
    original.kill_verified(proc())
    assert '.CreationDate' in calls[0] and '$p.CommandLine -cne $r.command' in calls[0]
    assert calls[0].index('exit 3') < calls[0].index('Stop-Process')


def test_powershell_encoding_and_process_parse(monkeypatch):
    calls=[]
    def output(argv,**kw):
        calls.append(argv)
        return json.dumps(proc())
    monkeypatch.setattr(b,'run',output)
    original=importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    monkeypatch.setattr(original,'run',output)
    assert original.processes()==[proc()]
    text=base64.b64decode(calls[0][-1]).decode('utf-16le')
    assert 'Get-CimInstance Win32_Process' in text and 'ConvertTo-Json' in text


@pytest.mark.parametrize('error',[subprocess.TimeoutExpired('secret',30),OSError('secret'),ValueError('secret')])
def test_probe_failure_no_sensitive_error_echo(monkeypatch,capsys,error):
    def fail(): raise error
    monkeypatch.setattr(b,'processes',fail)
    assert b.main(['gpu-local','test',PY])==2
    out=capsys.readouterr().out
    assert 'FAIL' in out and 'secret' not in out


@pytest.mark.parametrize('path',['../outside','.local/relabel/secret.json','.ssh/key','.kaggle/key','api-keys/key'])
def test_input_path_safety(path):
    with pytest.raises(b.CheckFailed): b.local(path)


@pytest.mark.parametrize('slug', ['knee-hb157-rkd-standalone-v1', 'knee-hb156-cnxs-l2-v1'])
@pytest.mark.parametrize('mutation', [None, 'status', 'errors', 'fixture_only',
    'nested_fixture', 'kernel_id', 'stage', 'parent', 'other_child'])
def test_gate_v130_real_receipts_read_only(monkeypatch, slug, mutation):
    # Real gate-copy inputs and immutable parent pins; mutate receipt bytes only in memory.
    root = SCRIPT.parents[2]
    monkeypatch.setattr(b, 'ROOT', root)
    path = root/'.local/state/gates'/f'{slug}.json'
    receipt = json.loads(path.read_text(encoding='utf-8-sig'))
    stage = b.local(receipt['stage'])
    metadata = b.read(stage/'kernel/kernel-metadata.json')
    if mutation == 'status': receipt['status'] = 'FAIL'
    if mutation == 'errors': receipt['errors'] = ['invalid']
    if mutation == 'fixture_only': receipt['fixture_only'] = True
    if mutation == 'nested_fixture':
        receipt['stage_inventory'] = dict(receipt.get('stage_inventory') or {}, fixture_only=True)
    if mutation == 'kernel_id': receipt['kernel_id'] = 'other/wrong'
    if mutation == 'stage': receipt['stage'] = str(root/'.local/scratch/wrong-stage')
    if mutation == 'parent': receipt['parent'] = str(stage)
    if mutation == 'other_child':
        receipt['kernel_id'] = metadata['id'] = 'other/unlisted'
        metadata['title'] = 'unlisted'
    requested = root/'.local/state/gates'/(metadata['id'].split('/')[1]+'.json')
    raw = json.dumps(receipt).encode('utf-8')
    original_read = b.read
    original_bytes = Path.read_bytes
    original_sha = b.sha
    monkeypatch.setattr(b, 'read', lambda p: receipt if b.local(p) == requested else original_read(p))
    monkeypatch.setattr(Path, 'read_bytes', lambda p: raw if p.resolve() == requested else original_bytes(p))
    monkeypatch.setattr(b, 'sha', lambda p: b.hashlib.sha256(raw).hexdigest()
                        if b.local(p) == requested else original_sha(p))
    if mutation is None:
        b.gate_check(stage, metadata)
    else:
        with pytest.raises(b.CheckFailed, match='strict_variant_gate receipt mismatch/FAIL'):
            b.gate_check(stage, metadata)
