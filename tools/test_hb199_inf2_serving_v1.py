"""INF2 cheap CPU parser/IO tests; all completion and result fixtures TEST ONLY."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import pytest
from tools import hb199_inf2_serving_v1 as m

def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(v if isinstance(v,bytes) else (json.dumps(v,sort_keys=True)+'\n').encode())
    return p

def native_report(root,report,text,agent='/root/test_inf2_reviewer'):
    put(report,text.encode());prompt=put(report.with_suffix('.prompt.txt'),b'TEST ONLY, NOT AN ACTUAL TASK')
    value=dict(schema=m.native.SCHEMA,host=m.native.HOST,model=m.native.MODEL,agent_id=agent,
        lane='native-astra-r2',prompt=str(prompt),report=str(report),terminal_status='completed',
        observed_at='2026-10-18T01:00:00+00:00',prompt_sha256=m.sha(prompt),report_sha256=m.sha(report),
        completion_message='TEST-ONLY synthetic parser fixture; no production approval')
    p=put(report.with_suffix('.completion.json'),value)
    m.native.record(p,'TEST ONLY fixture, never real completion',root)

def approve(f,phase,subject,claim=None,changes=None,verdict='VERDICT: PASS',agent='/root/test_inf2_reviewer'):
    body=m.attestation(phase,m.sha(f.w/'registration.json'),subject,claim,agent);body.update(changes or {})
    native_report(f.root,f.w/(phase+'-REVIEW.md'),
        '# TEST-ONLY complete native fixture\nReviewer-Lane: native-astra-r2\n'+m.PREFIX+json.dumps(body)+'\n'+verdict+'\n',agent)

class Fixture:
    def __init__(self,root):
        self.root=root;self.w=root/m.REL;self.w.mkdir(parents=True)
        self.source={p:self.file(p,(m.ROOT/p).read_bytes()) for p in m.SOURCES}
        self.source['tools/hb105_serving_v101.py']=self.file('tools/hb105_serving_v101.py',b'TEST ONLY original helper source')
        ids=[f'test-study-{i:03}' for i in range(58)]
        self.metadata={m.CORPUS+'/'+n:self.file(m.CORPUS+'/'+n,b'TEST ONLY '+n.encode()) for n in ('ids.npy','masks.npy','manifest.json')}
        roster='.local/scratch/fixture-same58.npz';roster_sha=self.file(roster,b'TEST ONLY roster source')
        bridge=put(root/m.BRIDGE/'registration.json',dict(bindings={roster:roster_sha}))
        entries=[]
        for j,name in enumerate(m.MEMBERS):
            p='.local/scratch/models/'+name+'.pt';h=self.file(p,b'TEST ONLY NO MODEL '+name.encode())
            e=dict(id=name,family='S',seed=42 if j<5 else 43,fold=j%5,arch='convnext_small.fb_in22k_ft_in1k',
                path=p,sha256=h,bytes=(root/p).stat().st_size,spec_sha256='a'*64)
            entries.append(e)
        manifest=dict(candidate=m.KEY,registration_sha256=m.sha(bridge),
            small_members=[dict(identity=e,source_pin={k:e[k] for k in ('bytes','sha256')}) for e in entries],
            sources={'tools/hb105_serving_v101.py':dict(sha256=self.source['tools/hb105_serving_v101.py'])},
            corpus=dict(hb105_original=dict(metadata={Path(p).name:dict(sha256=h) for p,h in self.metadata.items()})),
            gold_roster=dict(source=roster,ordered_uid_sha256=m.uid_hash(ids),targets=m.TARGETS))
        mp=put(root/m.BRIDGE/'missing-serving-inputs.json',manifest)
        rp=root/m.BRIDGE/'RESULT-REVIEW.md'
        native_report(root,rp,'# TEST ONLY prior bridge\nVERDICT: PASS-WITH-NOTES\n','/root/k1_result_review')
        self.anchor={p.relative_to(root).as_posix():m.sha(p) for p in (bridge,mp,rp)}
        c=self.ctx();m.prerequisites(c)
        rows=[dict(uid=u,shape=[64,336,336],dtype='uint8',mask_dtype='uint8',centers=[1]*36,
            volume_sha256='b'*64,mask_sha256='c'*64,shard=m.CORPUS+'/test.npy',shard_row=i,global_row=i) for i,u in enumerate(ids)]
        data=dict(schema='h199-serving-data/v1',corpus=m.CORPUS,ids=ids,targets=m.TARGETS,rows=rows,
            ordered_uid_sha256=m.uid_hash(ids),metadata=self.metadata,source=dict(path=roster,sha256=roster_sha),
            contract='raptor64_336_c140_s02',spec={},renderer_executed=False,model_forwards=0,clean_oof=False)
        put(self.w/'data.json',data)
        for mode in ('cpu','memory','cache'):
            put(self.w/(mode+'-plan.json'),m.plan(mode))
            if mode!='cpu':put(self.w/(mode+'-queue.json'),m.queue(mode,root))
        self.reg=dict(schema='h199-serving-registration/v1',candidate=m.KEY,version=m.VERSION,builder=m.BUILDER,
            fixture_only=True,workspace=str(root),sources=self.source,inherited=c.pins.copy(),data=c.ref(self.w/'data.json'),
            members=entries,plans={k:c.ref(self.w/(k+'-plan.json')) for k in ('cpu','memory','cache')},
            queues={k:c.ref(self.w/(k+'-queue.json')) for k in m.TAGS},execution_authorized=False,qualified_for_submission=False)
        put(self.w/'registration.json',self.reg);m.prepare(self.ctx())
    def file(self,path,body):return m.sha(put(self.root/path,body))
    def ctx(self):return m.Context(self.root,fixture_anchor=self.anchor)
    def claim(self,mode):
        p=put(self.w/(mode+'-claim.json'),dict(schema='h199-serving-resource/v1',candidate=m.KEY,owner='B',state='CLAIMED',
            tag=m.TAGS[mode],exclusive=True,physical_gpu=0,logical_device='cuda:0',device_name='NVIDIA GeForce RTX 4070 Ti SUPER',
            reservation='TEST-ONLY-RESOURCE',shared_evidence='TEST ONLY, not an actual claim'))
        return m.sha(p)
    def launch(self,mode):
        if not (self.w/'BUILD-REVIEW.md').exists():approve(self,'BUILD',m.sha(self.w/'registration.json'))
        phase={'cpu':'CPU_SMOKE_LAUNCH','memory':'MEMORY_LAUNCH','cache':'CACHE_LAUNCH'}[mode]
        approve(self,phase,self.reg['plans'][mode]['sha256'],None if mode=='cpu' else self.claim(mode))
        v=m.admission(self.ctx(),mode);put(self.w/(mode+'-admission.json'),v);return v
    def complete(self,mode):
        self.launch(mode);c=self.ctx();reg,data=m.validate(c);run=self.w/(mode+'-run');run.mkdir()
        env=dict(os='win32',python='TEST',torch='TEST',numpy='TEST',timm='TEST',device='cpu' if mode=='cpu' else 'cuda:0')
        journals=[]
        phases=(['render']*58+['model_load','prediction']) if mode=='cpu' else (['model_load']*10+['render','prediction']*(10 if mode=='memory' else 580))
        for i,phase in enumerate(phases):
            p=put(run/'journal'/f'{i}.json',dict(identity=m.KEY,phase=phase,index=i,start=1.,end=2.,seconds=1.))
            journals.append(c.ref(p))
        worker=dict(schema='h199-serving-worker/v1',state='COMPLETE',mode=mode,source_sha256=reg['sources'][m.SELF],
            registration=c.ref(self.w/'registration.json'),data=reg['data'],plan=reg['plans'][mode],
            admission=c.ref(self.w/(mode+'-admission.json')),member_ids=m.MEMBERS,targets=m.TARGETS,
            ordered_uid_sha256=data['ordered_uid_sha256'],clean_oof=False,exposure='EXPOSED',failures=0,fallback_count=0,
            journals=journals,environment=env,device=env['device'])
        if mode=='cpu':worker.update(renderer_rows=58,render_checks=[dict(index=i,max_abs=0.) for i in range(58)],
            forward_member=m.MEMBERS[0],forward_index=0,forward_max_abs=0.,shape=[36,3,224,224],cuda_initialized=False)
        else:
            desktop=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',desktop_bytes=m.GIB,total_bytes=16376*2**20)
            put(run/'desktop.json',desktop);parent=dict(pid=12345,create_time=123.)
            put(run/'ticket.json',dict(mode=mode,parent=parent,admission=worker['admission']))
            put(run/'bound-job.json',dict(schema='gpu-job/v2',phase='running',tag=m.TAGS[mode],command=m.command(mode,self.root),
                command_sha256=m.jr.command_digest(m.command(mode,self.root),self.root),cwd=str(self.root),child_identity=parent))
            refs=[]
            for i,e in enumerate(reg['members']):
                p=put(run/f'model-{i}.json',dict(member=m.MEMBERS[i],model_sha256=e['sha256'],peak_reserved_bytes=8*m.GIB,
                    peak_allocated_bytes=7*m.GIB,synchronized=True,deleted=True,views=2,shape=[36,3,224,224],
                    studies=1 if mode=='memory' else 58,registration=worker['registration'],environment=env))
                refs.append(c.ref(p))
            worker.update(physical_gpu=0,precision=m.PRECISION,model_receipts=refs,peaks=[8*m.GIB]*10,npz=None,reference_max_abs=0. if mode=='memory' else None)
            if mode=='cache':
                m.atomic_npz(run/'probabilities.npz',ids=np.asarray(data['ids']),targets=np.asarray(m.TARGETS),
                    member_ids=np.asarray(m.MEMBERS),probabilities=np.full((10,58,12),.5,np.float32))
                worker['npz']=c.ref(run/'probabilities.npz')
        put(run/'worker.json',worker);put(run/'worker.log',b'TEST ONLY completed synthetic fixture')
        process=dict(child_exit_code=0,owned_cleanup_complete=True,survivors=[],source_sha256=reg['sources'][m.SELF])
        put(run/'process.json',process);m.finish(self.ctx(),mode,process)
        phase={'cpu':'CPU_SMOKE_RESULT','memory':'MEMORY_RESULT','cache':'CACHE_RESULT'}[mode]
        approve(self,phase,m.sha(run/'result.json'));return worker

@pytest.fixture
def f(tmp_path):return Fixture(tmp_path)

def test_complete_same_parser_cpu_memory_cache_positive(f):
    f.complete('cpu');f.complete('memory');f.complete('cache')
    for mode in ('cpu','memory','cache'):
        v=m.result(f.ctx(),mode);assert v['state']=='COMPLETE_PENDING_REVIEW'
        assert m.verify_admission(f.ctx(),mode)['fixture_only'] is True
    assert not m.read(f.w/'memory-run/worker.json')['npz']

def test_first_memory_is_acyclic_after_reviewed_cpu(f):
    f.complete('cpu');v=f.launch('memory')
    assert v['state']=='ADMITTED_MEMORY_ONLY'
    assert not (f.w/'memory-run').exists() and not (f.w/'cache-admission.json').exists()
    assert not any('MEMORY_RESULT' in p or 'memory-run' in p for p in v['consumed_sha256'])

@pytest.mark.parametrize('name',['registration.json','data.json','prepare.json','cpu-plan.json','memory-plan.json','cache-plan.json','memory-queue.json','cache-queue.json'])
def test_missing_phase_input_blocks(f,name):
    (f.w/name).rename(f.w/(name+'.TEST-ONLY-MISSING'))
    with pytest.raises((ValueError,OSError,KeyError)):m.admission(f.ctx(),'cpu')

@pytest.mark.parametrize('kind',['report','sidecar','completion','prompt','archive_report','archive_receipt','archive_completion','archive_prompt','archive_log'])
def test_missing_native_chain_blocks(f,kind):
    approve(f,'BUILD',m.sha(f.w/'registration.json'));report=f.w/'BUILD-REVIEW.md'
    side=Path(str(report)+'.receipt.json');v=m.read(side);arc=f.root/'.local/astra-improve/lanes/runs'/v['lane']/v['run_id']
    complete=Path(v['completion_source']);cv=m.read(complete)
    paths=dict(report=report,sidecar=side,completion=complete,prompt=Path(cv['prompt']),archive_report=arc/'report.md',
        archive_receipt=arc/'receipt.json',archive_completion=arc/'completion.json',archive_prompt=arc/'prompt.txt',archive_log=arc/'run.log')
    p=paths[kind];p.rename(p.with_name(p.name+'.TEST-ONLY-MISSING'))
    with pytest.raises((ValueError,OSError,KeyError)):m.admission(f.ctx(),'cpu')

@pytest.mark.parametrize('change,verdict,agent',[
    ({},'VERDICT: FAIL','/root/test_inf2_reviewer'),({},'VERDICT: PASS\nVERDICT: PASS','/root/test_inf2_reviewer'),
    ({},' VERDICT: FAIL\nVERDICT: PASS','/root/test_inf2_reviewer'),({},'VERDICT: PASS',m.BUILDER),
    ({'phase':'CACHE_LAUNCH'},'VERDICT: PASS','/root/test_inf2_reviewer'),
    ({'candidate':'C4'},'VERDICT: PASS','/root/test_inf2_reviewer'),
    ({'subject_sha256':'0'*64},'VERDICT: PASS','/root/test_inf2_reviewer'),
    ({'registration_sha256':'0'*64},'VERDICT: PASS','/root/test_inf2_reviewer'),
    ({'resource_claim_sha256':'0'*64},'VERDICT: PASS','/root/test_inf2_reviewer'),
    ({'reviewer':'/root/another_agent'},'VERDICT: PASS','/root/test_inf2_reviewer')])
def test_successful_native_completion_is_not_approval(f,change,verdict,agent):
    approve(f,'BUILD',m.sha(f.w/'registration.json'),changes=change,verdict=verdict,agent=agent)
    with pytest.raises(ValueError):m.admission(f.ctx(),'cpu')

@pytest.mark.parametrize('path',[m.SELF,m.TEST,'tools/gpu_job.py','tools/job_runtime.py','tools/hb105_serving_v101.py'])
def test_current_source_drift_blocks(f,path):
    put(f.root/path,(f.root/path).read_bytes()+b'\nTEST drift')
    with pytest.raises(ValueError):m.validate(f.ctx())

def test_late_native_mutation_blocks(f,monkeypatch):
    approve(f,'BUILD',m.sha(f.w/'registration.json'));original=m.Context.recheck;changed=[]
    def mutate(c):
        if any('BUILD-REVIEW' in p for p in c.pins) and not changed:
            p=f.w/'BUILD-REVIEW.md';put(p,p.read_bytes()+b'\nTEST drift');changed.append(True)
        original(c)
    monkeypatch.setattr(m.Context,'recheck',mutate)
    with pytest.raises(ValueError):m.admission(f.ctx(),'cpu')
    assert changed

@pytest.mark.parametrize('kind',['source','seed','fold','count','arch'])
def test_member_identity_rejects(kind,f):
    manifest=m.read(f.root/m.BRIDGE/'missing-serving-inputs.json')
    if kind=='count':manifest['small_members'].pop()
    else:
        e=manifest['small_members'][0]['identity'];e[{'source':'sha256'}.get(kind,kind)]={'source':'0'*64,'seed':43,'fold':1,'arch':'wrong'}[kind]
    with pytest.raises((ValueError,KeyError)):m.identities(manifest)

@pytest.mark.parametrize('kind',['id_duplicate','target','dtype','centers','hash'])
def test_data_shape_rejects(f,kind):
    data=m.read(f.w/'data.json')
    if kind=='id_duplicate':data['ids'][0]=data['ids'][1]
    if kind=='target':data['targets']=list(reversed(data['targets']))
    if kind=='dtype':data['rows'][0]['dtype']='float32'
    if kind=='centers':data['rows'][0]['centers']=[0]*35
    if kind=='hash':data['rows'][0]['volume_sha256']='invalid'
    with pytest.raises(ValueError):m.data_shape(data)

@pytest.mark.parametrize('peak,used,total,passes',[(12,1,16,True),(12.01,1,16,False),(12,3,16,False),(8,1,12,False),(float('nan'),1,16,False)])
def test_measured_memory(peak,used,total,passes):
    v=dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER',desktop_bytes=used*m.GIB,total_bytes=total*m.GIB)
    if passes:m.memory_ok(v,[peak*m.GIB])
    else:
        with pytest.raises(ValueError):m.memory_ok(v,[peak*m.GIB])

@pytest.mark.parametrize('kind',['review','claim','measurement','worker','model','queue'])
def test_cache_needs_real_reviewed_memory(f,kind):
    f.complete('cpu');f.complete('memory')
    phase='CACHE_LAUNCH';claim=f.claim('cache');approve(f,phase,f.reg['plans']['cache']['sha256'],claim)
    choices=dict(review=f.w/'MEMORY_RESULT-REVIEW.md',claim=f.w/'cache-claim.json',measurement=f.w/'memory-run/result.json',
        worker=f.w/'memory-run/worker.json',model=f.w/'memory-run/model-0.json',queue=f.w/'cache-queue.json')
    p=choices[kind];p.rename(p.with_name(p.name+'.TEST-ONLY-MISSING'))
    with pytest.raises((ValueError,OSError,KeyError)):m.admission(f.ctx(),'cache')

@pytest.mark.parametrize('bad',['nan','dtype','shape','ids','targets','members'])
def test_complete_npz_contract(f,bad):
    data=m.read(f.w/'data.json');v=dict(ids=np.asarray(data['ids']),targets=np.asarray(m.TARGETS),
        member_ids=np.asarray(m.MEMBERS),probabilities=np.full((10,58,12),.5,np.float32))
    if bad=='nan':v['probabilities'][0,0,0]=np.nan
    if bad=='dtype':v['probabilities']=v['probabilities'].astype(np.float64)
    if bad=='shape':v['probabilities']=v['probabilities'][:9]
    if bad=='ids':v['ids'][0]=v['ids'][1]
    if bad=='targets':v['targets']=v['targets'][::-1]
    if bad=='members':v['member_ids']=v['member_ids'][::-1]
    p=f.w/'bad.npz';m.atomic_npz(p,**v)
    with pytest.raises(ValueError):m.check_npz(p,data)

def test_nonfatal_diagnostic_and_required_publication(tmp_path,monkeypatch):
    def fail(*a,**k):raise PermissionError('TEST ONLY persistent write failure')
    monkeypatch.setattr(m.io,'atomic_json',fail)
    assert m.diagnostic(tmp_path/'phase.json',{}) is False
    with pytest.raises(PermissionError):m.fresh(tmp_path/'worker.json',{})

def test_create_only_and_npz_publish_failure(tmp_path,monkeypatch):
    p=tmp_path/'immutable.json';m.fresh(p,{'old':True})
    with pytest.raises(ValueError):m.fresh(p,{'new':True})
    assert m.read(p)=={'old':True}
    def fail(*a,**k):raise PermissionError('TEST ONLY NPZ publication error')
    monkeypatch.setattr(m.io,'rename',fail)
    with pytest.raises(PermissionError):m.atomic_npz(tmp_path/'values.npz',x=np.zeros(1))
    assert not (tmp_path/'values.npz').exists()

def test_process_cleanup_precedes_publication_error(tmp_path,monkeypatch):
    run=tmp_path/'supervised';run.mkdir();seen=[];original=m.terminate_owned
    def cleaned(root,owned):
        value=original(root,owned);seen.append(value);return value
    monkeypatch.setattr(m,'terminate_owned',cleaned)
    def fail(*a,**k):raise PermissionError('TEST ONLY final process publication')
    monkeypatch.setattr(m,'fresh',fail)
    with pytest.raises(PermissionError):m.supervise([sys.executable,'-B','-c','print("TEST ONLY NON-ML")'],run,dict(os.environ),'TEST',True,poll=.01)
    assert seen==[[]]

def test_failed_child_cannot_succeed(tmp_path):
    run=tmp_path/'supervised';run.mkdir()
    value=m.supervise([sys.executable,'-B','-c','raise SystemExit(9)'],run,dict(os.environ),'TEST',True,poll=.01)
    assert value['child_exit_code']==9 and value['owned_cleanup_complete']

@pytest.mark.parametrize('change',['pid','start','command','tag','parent','digest'])
def test_exact_job_identity_rejects(change,monkeypatch):
    own=dict(pid=100,create_time=1.);parent=dict(pid=99,create_time=.5);cmd=m.command('memory')
    v=dict(schema='gpu-job/v2',phase='running',tag=m.TAGS['memory'],cwd=str(m.ROOT),command=cmd,
        command_sha256=m.jr.command_digest(cmd,m.ROOT),child_identity=own.copy(),pid=99,create_time=.5,run_id='test-run')
    args=[sys.executable,str(m.ROOT/'tools/gpu_job.py'),'--_run-job',m.TAGS['memory'],'test-run']
    monkeypatch.setattr(m.jr,'liveness',lambda value:'alive')
    assert m.check_job(v,'memory',own,parent,cmd,args)==v
    if change=='pid':v['child_identity']['pid']=101
    if change=='start':v['child_identity']['create_time']=2.
    if change=='command':v['command']=[*cmd[:-1],'cache']
    if change=='tag':v['tag']='K2-UNRELATED'
    if change=='parent':args[-1]='different-run'
    if change=='digest':v['command_sha256']='0'*64
    with pytest.raises(ValueError):m.check_job(v,'memory',own,parent,cmd,args)

def test_missing_worker_never_publishes_success(f):
    f.launch('cpu');run=f.w/'cpu-run';run.mkdir()
    process=dict(child_exit_code=0,owned_cleanup_complete=True,survivors=[])
    with pytest.raises(OSError):m.finish(f.ctx(),'cpu',process)
    assert not (run/'result.json').exists()

def test_result_publication_validates_before_success(f):
    f.complete('cpu');run=f.w/'cpu-run'
    (run/'result.json').rename(run/'result.TEST-ONLY-OLD.json')
    value=m.read(run/'worker.json');value['forward_max_abs']=.1;put(run/'worker.json',value)
    process=m.read(run/'process.json')
    with pytest.raises(ValueError,match='reference parity'):m.finish(f.ctx(),'cpu',process)
    assert not (run/'result.json').exists()

def test_no_root_env_or_fixture_cli_bypass(tmp_path):
    with pytest.raises(ValueError):m.Context(tmp_path)
    cp=subprocess.run([sys.executable,'-B',str(m.ROOT/m.SELF),'validate','--root',str(tmp_path)],capture_output=True,text=True)
    assert cp.returncode!=0
    source=(m.ROOT/m.SELF).read_text()
    assert 'hb199_inf2_bridge_v1' not in source and 'hb198_c4_gold58_v102' not in source and 'active_jobs(' not in source
    cp=subprocess.run([sys.executable,'-B','-c','import tools.hb199_inf2_serving_v1; import sys; assert "torch" not in sys.modules'],cwd=m.ROOT,capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr
