"""Regression tests v1.0.3: WDDM memory/name gate; scientific cases unchanged."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from tools import k1_syn_head_v1 as h
from tools import k1_syn_runtime_v1 as rt


def metrics(deltas=(.008,.008,.008,0.)):
    from tools.l1_rco_k1_eval_v1 import TARGETS
    return dict(per_target=[dict(ruler=r,target=t,rco_auc=.7+(d if t=='Synovitis' else 0.),
        r7_auc=.7,delta=d if t=='Synovitis' else 0.)
        for r,d in zip(h.RULE['go']['rulers'],deltas) for t in TARGETS])


def test_go_boundary_and_macro():
    m=metrics(); assert h.go_rule(m,True)
    assert sum(r['delta'] for r in m['per_target'])/48==pytest.approx(.0005)


@pytest.mark.parametrize('deltas',[(.01,.01,-.001,-.001),(.005,.005,.005,.005),(.006,.006,.006,.00599)])
def test_go_rejects_gate_violation(deltas): assert not h.go_rule(metrics(deltas),True)


@pytest.mark.parametrize('mutation',['missing','duplicate','nan','effusion','other','unchanged'])
def test_go_fails_closed(mutation):
    m=metrics()
    if mutation=='missing': m['per_target'].pop()
    if mutation=='duplicate': m['per_target'][0]=m['per_target'][1].copy()
    if mutation=='nan': m['per_target'][0]['rco_auc']=float('nan')
    if mutation in ('effusion','other'):
        next(r for r in m['per_target'] if r['target']==('Effusion' if mutation=='effusion' else 'ACL'))['delta']=.001
    assert not h.go_rule(m,mutation!='unchanged')


def test_soft_mass_equalization_and_boundaries():
    y=np.array([.1,.7,.3,.9,.2,.8]); e=np.array([0.,.2,1/3,.5,2/3,1.])
    p,n,bins=h.soft_weights(y,e)
    for i in range(3):
        assert p[2*i:2*i+2].sum()==pytest.approx(1/6)
        assert n[2*i:2*i+2].sum()==pytest.approx(1/6)
        assert bins[i]['n']==2
    assert (p+n).sum()==pytest.approx(1.)


@pytest.mark.parametrize('y,e', [([0.,0.,0.],[.1,.5,.9]),([.5,.5],[.1,.5]),([np.nan,.5,.5],[.1,.5,.9]),([.5,.5,.5],[-.1,.5,.9])])
def test_bad_strata_rejected(y,e):
    with pytest.raises(ValueError): h.soft_weights(y,e)


def test_fitter_uses_features_and_regularization():
    x=np.zeros((12,384)); x[:,0]=np.tile([-1.,1.],6)
    y=np.tile([.15,.85],6); eff=np.repeat([.1,.5,.9],4)
    initial=np.zeros(385); result=h.fit_head(x,y,eff,initial)
    theta=np.array(result['theta'])
    assert result['converged'] and 0<theta[0]<3
    assert np.max(abs(theta[1:]))<1e-6
    p=np.ones((12,12),np.float32)*.5; changed=h.replace_syn(p,x,theta)
    assert np.array_equal(p[:,np.arange(12)!=8],changed[:,np.arange(12)!=8])
    assert changed[1,8]>.5>changed[0,8]


def test_trio_changes_syn_only_with_ties():
    rng=np.random.default_rng(7); a=np.round(rng.random((21,12)),1); b=a.copy(); b[:,8]=np.arange(21)/21
    old=h.trio(a,a,a); new=h.trio(a,b,a)
    assert np.array_equal(old[:,:8],new[:,:8]) and np.array_equal(old[:,9:],new[:,9:])
    assert np.array_equal(h.ranks(np.ones((5,12))),np.full((5,12),.5))


def test_training_labels_reject_holdout(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'ROOT',tmp_path); monkeypatch.setattr(h,'PARENT','parent'); (tmp_path/'parent').mkdir()
    (tmp_path/'parent/targets.csv').write_text('StudyInstanceUID,Synovitis|prior,Effusion|prior\nhold,.5,.5\n')
    with pytest.raises(ValueError,match='train only'): h.training_targets(dict(train=['train'],holdout=['hold']))


def test_resume_hash_and_signature(tmp_path):
    path=tmp_path/'0.npz'; row=dict(features=np.ones(384),predictions=np.ones(12)*.5,initial=np.zeros(385))
    h.npz(path,**row); h.write(path.with_suffix('.json'),dict(signature='good',uid_digest=h.digest('u'),sha256=h.sha(path)))
    assert rt.row_read(path,'good','u')['features'].shape==(384,)
    with pytest.raises(ValueError): rt.row_read(path,'other','u')
    with pytest.raises(ValueError): rt.row_read(path,'good','other')
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): rt.row_read(path,'good','u')


def test_feature_full_roster_required(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'OUT',tmp_path); (tmp_path/'features').mkdir()
    h.write(tmp_path/'PREREGISTRATION.json',{})
    h.write(tmp_path/'REVIEW.json',{})
    path=tmp_path/'features/train.npz'
    h.npz(path,ids=np.array(['other']),features=np.zeros((1,384)),predictions=np.zeros((1,12)),initial=np.zeros(385))
    h.write(tmp_path/'features/receipt.json',dict(status='COMPLETE',registration_sha256=h.sha(tmp_path/'PREREGISTRATION.json'),outputs=dict(train=h.sha(path)),checkpoint_sha256=h.CHECKPOINT_SHA,review_sha256=h.sha(tmp_path/'REVIEW.json')))
    with pytest.raises(ValueError,match='roster'): h.load_features('train',dict(train=['wanted']))


def test_missing_review_stops_before_gpu(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'OUT',tmp_path); h.write(tmp_path/'PREREGISTRATION.json',{})
    h.write(tmp_path/'CPU-SMOKE.json',dict(status='PASS',cuda_initialized=False,registration_sha256=h.sha(tmp_path/'PREREGISTRATION.json')))
    h.write(tmp_path/'OPERATIONAL-PINS-v102.json',{})
    h.write(tmp_path/'TEST-RESULTS.json',dict(status='PASS'))
    monkeypatch.setattr(h,'validate',lambda *a: {})
    monkeypatch.setattr(rt,'owner_check',lambda *a: pytest.fail('GPU ownership path reached without review'))
    assert rt.admit_dry_run('local')['status']=='BLOCKED'
    with pytest.raises(ValueError,match='review'): rt.run('local')


def test_self_review_or_stale_binding_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'OUT',tmp_path); monkeypatch.setattr(h,'ROOT',tmp_path)
    for p in ('PREREGISTRATION.json','OPERATIONAL-PINS-v102.json','CPU-SMOKE.json','TEST-RESULTS.json'): h.write(tmp_path/p,{})
    (tmp_path/'review.md').write_text('independent test report')
    review=dict(verdict='PASS',reviewer='astra-b4',bindings=rt.review_bindings(),approved_hosts=['local'],report='review.md',report_sha256=h.sha(tmp_path/'review.md'))
    h.write(tmp_path/'REVIEW.json',review)
    with pytest.raises(ValueError,match='independent'): rt.require_review()
    review['reviewer']='independent'; h.write(tmp_path/'REVIEW.json',review); rt.require_review()
    h.write(tmp_path/'CPU-SMOKE.json',dict(changed=True))
    with pytest.raises(ValueError,match='binding'): rt.require_review()


def test_direct_worker_fails_before_torch(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'OUT',tmp_path)
    h.write(tmp_path/'ADMISSION.json',dict(supervisor=dict(pid=-10,create_time=0)))
    with pytest.raises(ValueError,match='owned worker'): rt.worker('local')


def test_watchdog_no_progress_and_collapse():
    status=dict(time=0,phase='train',done=100,healthy_rate=1.)
    assert '15-minute' in rt.stall(status,901,dict(util=0,power=10),[])
    assert 'WDDM' in rt.stall(status,601,dict(util=100,power=70),[])
    status['time']=1800
    assert 'throughput' in rt.stall(status,1800,dict(util=30,power=150),[dict(time=0,phase='train',done=0)])
    status['done']=1000
    assert rt.stall(status,1800,dict(util=30,power=150),[dict(time=0,phase='train',done=0)]) is None


def test_windows_replace_retry_bounded(monkeypatch,tmp_path):
    calls=[]
    def blocked(*a): calls.append(1); raise PermissionError('sharing lock')
    monkeypatch.setattr(h.os,'replace',blocked); monkeypatch.setattr(h.time,'sleep',lambda *a:None)
    with pytest.raises(PermissionError): h.replace(tmp_path/'a',tmp_path/'b')
    assert len(calls)==40


@pytest.mark.parametrize('host,util,accepted',[('local',59,True),('local',60,False),('vast',19,True),('vast',20,False)])
def test_utilization_boundary(monkeypatch,host,util,accepted):
    monkeypatch.setattr(rt,'snapshot',lambda host:dict(util=util))
    monkeypatch.setattr(rt.subprocess,'check_output',lambda *a,**k:'')
    if accepted: assert rt.idle(host)['util']==util
    else:
        with pytest.raises(ValueError,match='busy'): rt.idle(host)


@pytest.mark.parametrize('row',[
    '123, tritonserver, 2048 MiB','123, python.exe, [N/A]',
    '123, PYTHONW.EXE, 0','123, train.exe, [N/A]',
    '123, torchrun.exe, [N/A]','123, unknown.exe, 1',
    '123, explorer.exe, 2048 MiB','123, desktop.exe, [Not Supported]',
    '123, desktop.exe, -1','123, desktop.exe, nan',
    '123, desktop.exe, inf','123, desktop.exe,','0, dwm.exe, [N/A]',
    'N/A, N/A','malformed'])
def test_unknown_compute_process_refused(monkeypatch,row):
    import psutil
    def denied(pid): raise psutil.AccessDenied(pid)
    monkeypatch.setattr(psutil,'Process',denied)
    monkeypatch.setattr(rt,'snapshot',lambda host:dict(util=0))
    monkeypatch.setattr(rt.subprocess,'check_output',lambda *a,**k:row)
    with pytest.raises(ValueError,match='busy'): rt.idle('local')


@pytest.mark.parametrize('name',['[Insufficient Permissions]','dwm.exe',
    'explorer.exe','SearchApp.exe','StartMenuExperienceHost.exe',
    'msedgewebview2.exe','Docker Desktop.exe','claude.exe',
    r'C:\Windows\System32\dwm.exe'])
def test_wddm_desktop_allowed_only_local(monkeypatch,name):
    import psutil
    monkeypatch.setattr(rt,'snapshot',lambda host:dict(util=0))
    monkeypatch.setattr(rt.subprocess,'check_output',lambda *a,**k:'4688, '+name+', [N/A]')
    def denied(pid): raise psutil.AccessDenied(pid)
    monkeypatch.setattr(psutil,'Process',denied)
    assert rt.idle('local')['util']==0
    with pytest.raises(ValueError,match='busy'): rt.idle('vast')


def test_observed_desktop_rows_and_hidden_python(monkeypatch):
    import psutil
    from types import SimpleNamespace
    rows='\n'.join(f'{4688+i}, {name}, [N/A]' for i,name in enumerate([
        '[Insufficient Permissions]','explorer.exe','SearchApp.exe',
        'StartMenuExperienceHost.exe','msedgewebview2.exe','Docker Desktop.exe','claude.exe']))
    monkeypatch.setattr(rt,'snapshot',lambda host:dict(util=59))
    def query(args,**kwargs):
        assert '--query-compute-apps=pid,process_name,used_memory' in args
        return rows
    monkeypatch.setattr(rt.subprocess,'check_output',query)
    def denied(pid): raise psutil.AccessDenied(pid)
    monkeypatch.setattr(psutil,'Process',denied)
    assert rt.idle('local')['util']==59
    monkeypatch.setattr(rt,'snapshot',lambda host:dict(util=60))
    with pytest.raises(ValueError,match='busy'): rt.idle('local')
    monkeypatch.setattr(rt,'snapshot',lambda host:dict(util=0))
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(name=lambda:'python.exe'))
    with pytest.raises(ValueError,match='busy'): rt.idle('local')


def test_memory_zero_and_own_pid_do_not_exempt_ml_names(monkeypatch):
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(name=lambda:'desktop.exe'))
    monkeypatch.setattr(rt.os,'getpid',lambda:123)
    assert rt.desktop_process('local','124, desktop.exe, 0 MiB')
    assert rt.desktop_process('local','123, owned.exe, 2048 MiB')
    assert not rt.desktop_process('local','124, owned.exe, 2048 MiB')
    assert not rt.desktop_process('local','123, python.exe, [N/A]')
    assert not rt.desktop_process('local','123, python.exe, 2048 MiB')


def test_vast_does_not_write_parent():
    c=rt.config('vast')
    assert c['cache'].startswith('/root/work/h224-folds-v2/')
    assert c['owner']=='/root/work/k1-syn-head-v102'
    assert not str(rt.VAST_ROOT).startswith('/root/work/h224-folds-v2')
    assert rt.VAST_PREVIOUS_OWNER.as_posix()=='/root/work/h224-failure-v1/.local/scratch/astra-b2/h224-folds-v1/gpu0-owner'
    assert rt.VAST_PREVIOUS_TAG=='K1_FAILURE_WEIGHT_F0_V1B'


@pytest.mark.parametrize('invalid',['failed','old','partial','command'])
def test_carry_over_rejects_failed_old_partial_or_wrong_command(tmp_path,monkeypatch,invalid):
    from tools import job_runtime as jobs
    from tools import l1_rco_k1_eval_v1 as ev
    monkeypatch.setattr(h,'ROOT',tmp_path)
    monkeypatch.setattr(h,'BASE_CHECKPOINTS',{})
    cmd=['python','-B',str(tmp_path/h.BASE/'stride_screen_v1.py'),'run']
    job=dict(command=cmd,run_id='8c52a77fa1c540df8e3bdca6edbc27f9',command_sha256=jobs.command_digest(cmd,tmp_path))
    receipt=dict(state='SUCCESS',exit_code=0,run_id=job['run_id'],command_sha256=job['command_sha256'])
    for m in ('v2','s0','s1'):
        path=tmp_path/h.BASE/f'preds/{m}_s2.csv'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('bound predictions')
    monkeypatch.setattr(ev,'predictions',lambda *a:(['u'],np.zeros((1,12))))
    h.write(tmp_path/'artifacts/logs/K1_STRIDE_SCREEN_V1/job.json',job)
    h.write(tmp_path/'artifacts/logs/K1_STRIDE_SCREEN_V1/receipt.json',receipt)
    assert h.baseline_provenance(['u'])['run_id']==job['run_id']
    if invalid=='failed': receipt['state']='FAILED'; receipt['exit_code']=1
    if invalid=='old': receipt['run_id']='old'
    if invalid=='partial': monkeypatch.setattr(ev,'predictions',lambda *a:([],np.zeros((0,12))))
    if invalid=='command': job['command'][-1]='evaluate'
    h.write(tmp_path/'artifacts/logs/K1_STRIDE_SCREEN_V1/job.json',job)
    h.write(tmp_path/'artifacts/logs/K1_STRIDE_SCREEN_V1/receipt.json',receipt)
    with pytest.raises(ValueError): h.baseline_provenance(['u'])


def test_vast_requires_explicit_b_transfer_and_success(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'OUT',tmp_path)
    h.write(tmp_path/'PREREGISTRATION.json',{})
    with pytest.raises(FileNotFoundError): rt.vast_transfer()
    value=dict(owner='B',exclusive_until_terminal=True,registration_sha256=h.sha(tmp_path/'PREREGISTRATION.json'),
               destination=rt.VAST_ROOT.as_posix(),previous_owner=rt.VAST_PREVIOUS_OWNER.as_posix(),admit_before_epoch=rt.time.time()+3600)
    h.write(tmp_path/'OWNER-TRANSFER.json',value)
    def blocked(**kwargs): raise ValueError('SUCCESS required')
    monkeypatch.setattr(rt,'predecessor_success',blocked)
    with pytest.raises(ValueError,match='SUCCESS'): rt.vast_transfer()
    monkeypatch.setattr(rt,'predecessor_success',lambda **k:{})
    assert rt.vast_transfer()==value
    value['admit_before_epoch']=0; h.write(tmp_path/'OWNER-TRANSFER.json',value)
    with pytest.raises(ValueError,match='transfer'): rt.vast_transfer()


@pytest.mark.parametrize('mutation',['none','failed','old_run','command','wrapper','child','marker','missing','live'])
def test_exact_predecessor_success_authentication(tmp_path,monkeypatch,mutation):
    from tools import job_runtime as jobs
    monkeypatch.setattr(rt,'VAST_PREVIOUS_OWNER',tmp_path)
    cmd=['python','fixed-predecessor']
    pin=jobs.command_digest(cmd,tmp_path)
    monkeypatch.setattr(rt,'VAST_PREVIOUS_COMMAND_SHA',pin)
    d=tmp_path/'artifacts/logs'/rt.VAST_PREVIOUS_TAG
    job=dict(schema='gpu-job/v2',tag=rt.VAST_PREVIOUS_TAG,run_id=rt.VAST_PREVIOUS_RUN,
        command=cmd,cwd=tmp_path.as_posix(),command_sha256=pin,pid=10,create_time=1.,child_identity=dict(pid=11,create_time=2.))
    receipt=dict(schema='gpu-job-receipt/v1',state='SUCCESS',exit_code=0,tag=rt.VAST_PREVIOUS_TAG,
        run_id=rt.VAST_PREVIOUS_RUN,command_sha256=pin,wrapper_identity=dict(pid=10,create_time=1.),child_identity=job['child_identity'].copy())
    if mutation=='failed': receipt['state']='FAILED'; receipt['exit_code']=1
    if mutation=='old_run': receipt['run_id']='old'
    if mutation=='command': job['command']=['other']
    if mutation=='wrapper': receipt['wrapper_identity']['pid']=100
    if mutation=='child': receipt['child_identity']['pid']=100
    h.write(d/'job.json',job)
    if mutation!='missing': h.write(d/'receipt.json',receipt)
    (d/'SUCCESS').write_text('exit 0 at 2026-10-22T01:00:00' if mutation!='marker' else 'SUCCESS')
    monkeypatch.setattr(jobs,'liveness',lambda *a:pytest.fail('CPU dry-run must not query processes'))
    if mutation in ('none','live'):
        assert rt.predecessor_success()['run_id']==rt.VAST_PREVIOUS_RUN
        monkeypatch.setattr(jobs,'job_status',lambda *a,**k:dict(state='unknown' if mutation=='live' else 'SUCCESS',receipt_kind='bound'))
        if mutation=='live':
            with pytest.raises(ValueError,match='live'): rt.predecessor_success(launch=True)
        else: assert rt.predecessor_success(launch=True)['run_id']==rt.VAST_PREVIOUS_RUN
    else:
        with pytest.raises((ValueError,FileNotFoundError)): rt.predecessor_success()



def test_unreviewed_environment_blocked(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'OUT',tmp_path)
    h.write(tmp_path/'PREREGISTRATION.json',{})
    h.write(tmp_path/'CPU-SMOKE.json',dict(status='PASS',cuda_initialized=False,registration_sha256=h.sha(tmp_path/'PREREGISTRATION.json')))
    h.write(tmp_path/'OPERATIONAL-PINS-v102.json',{})
    h.write(tmp_path/'TEST-RESULTS.json',{})
    monkeypatch.setattr(h,'validate',lambda *a:{})
    monkeypatch.setattr(rt,'require_review',lambda:dict(approved_hosts=['local'],approved_environments=dict(local={'python':'wrong'})))
    monkeypatch.setattr(rt,'cpu_environment',lambda:dict(python='actual'))
    assert rt.admit_dry_run('local')['status']=='BLOCKED'


def test_operational_overlay_preserves_science_and_inputs():
    reg=h.read(h.OUT/'PREREGISTRATION.json')
    pins=h.operational_pins(reg)
    allowed={'tools/k1_syn_head_v1.py','tools/k1_syn_runtime_v1.py','tests/test_k1_syn_head_v1.py'}
    assert {n for n in pins if pins[n]!=reg['pins'][n]}==allowed
    assert reg['rule']==h.RULE
    for n in allowed: assert pins[n]==h.sha(h.ROOT/n)


@pytest.mark.parametrize('mutation',['extra_input','old_pin','registration'])
def test_operational_overlay_rejects_scope_drift(monkeypatch,mutation):
    reg=h.read(h.OUT/'PREREGISTRATION.json')
    amendment=h.read(h.OUT/'OPERATIONAL-PINS-v102.json')
    if mutation=='extra_input': amendment['replacements'][h.CHECKPOINT]=dict(before='a',after='b'*64)
    if mutation=='old_pin': amendment['replacements']['tools/k1_syn_head_v1.py']['before']='wrong'
    if mutation=='registration': amendment['registration_sha256']='wrong'
    monkeypatch.setattr(h,'read',lambda p:amendment)
    with pytest.raises(ValueError): h.operational_pins(reg)
