"""INF2-only same58 Small adapter v1.0.0; PRE-SMOKE until independent reviews.

Static commands never import torch. All production paths are fixed. No labels,
metrics, training, download, release, canonical transport or submission.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import native_codex_dispatch_record_v1 as native
from tools import hb145_output_io as io
from tools import job_runtime as jr
from tools.hb155_watchdog_v1 import collect_owned, terminate_owned, Policy

VERSION = '1.0.0'
KEY = 'R6W958-INF2-V1'
BUILDER = '/root/candidate_scout'
SELF = 'tools/hb199_inf2_serving_v1.py'
TEST = 'tools/test_hb199_inf2_serving_v1.py'
REL = Path('.local/scratch/slot-b/batch5-1018/inf2-serving')
BRIDGE = Path('.local/scratch/slot-b/batch5-1018/inf2-bridge')
CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
TAGS = {'memory':'INF2_SMALL58_V1_MEMORY_1','cache':'INF2_SMALL58_V1_CACHE_1'}
TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
           'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
MEMBERS = [f'S-s{s}-f{f}' for s in (42,43) for f in range(5)]
PRECISION = 'FP16 CUDA forward; FP32 sigmoid and normal/flip probability mean'
PREFIX = 'H199-SERVING-ATTESTATION: '
GIB = 2**30
SOURCES = (SELF,TEST,'tools/native_codex_dispatch_record_v1.py','tools/hb145_output_io.py',
    'tools/gpu_queue.py','tools/gpu_job.py','tools/job_runtime.py','tools/hb155_watchdog_v1.py','tools/hb150_io_v1.py')
ANCHOR = {
    (BRIDGE/'RESULT-REVIEW.md').as_posix():'1355c0a1b5b9f39cca5f13117be6fda094c4388a02da6ec8f90fd8cd5553e8ef',
    (BRIDGE/'missing-serving-inputs.json').as_posix():'624bab62c7ae6b8298adfcb0587dd95b2070b9f489a67c6296ec4f4962b26f79',
    (BRIDGE/'registration.json').as_posix():'e22cb92a96e3b3eba661f4a29d57b27fb5466fc94eba08e5015c1bb67ff33342'}
need = native.need

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return native.parse(Path(path).read_bytes())
def uid_hash(ids):return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def array_hash(x):return hashlib.sha256(x.tobytes()).hexdigest()

def fresh(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive writer lock plus same-directory Windows retry publication.
    lock=path.with_name(path.name+'.publish-lock')
    io.retry(lock.mkdir)
    try:
        need(not path.exists(),'immutable output exists: '+path.name)
        io.atomic_json(path,value)
    finally:io.retry(lock.rmdir)

def diagnostic(path,value):
    try:io.atomic_json(path,value);return True
    except OSError:return False

class Context:
    def __init__(self,root=ROOT,*,fixture_anchor=None):
        self.root=Path(root).resolve();self.fixture=self.root!=ROOT
        need((not self.fixture and fixture_anchor is None) or
             (self.fixture and fixture_anchor and self.root.is_relative_to(ROOT/'.local/scratch')),
             'fixed production root or explicit isolated TEST-ONLY parser fixture')
        if not self.fixture:
            need(os.environ.get('KNEE_WORKSPACE_ROOT') in (None,str(ROOT)),'alternate runtime root forbidden')
        self.anchor=fixture_anchor or ANCHOR;self.work=self.root/REL;self.pins={}
    def ref(self,path):
        p=native.local(path,self.root);return dict(path=p.relative_to(self.root).as_posix(),sha256=sha(p))
    def file(self,item):
        need(type(item) is dict and set(item)=={'path','sha256'},'exact file reference')
        p=native.local(item['path'],self.root);key=p.relative_to(self.root).as_posix()
        need(key==item['path'] and re.fullmatch('[0-9a-f]{64}',item['sha256']) and
             self.pins.get(key,item['sha256'])==item['sha256'] and sha(p)==item['sha256'],'stale evidence: '+key)
        self.pins[key]=item['sha256'];return p
    def json(self,item):return read(self.file(item))
    def add(self,mapping):
        need(type(mapping) is dict and bool(mapping),'nonempty pin map')
        for p,h in mapping.items():self.file(dict(path=p,sha256=h))
    def recheck(self):self.add(self.pins.copy())

def archive(c,report):
    report=c.ref(report);p=c.file(report);side=c.ref(str(p)+'.receipt.json');v=c.json(side)
    need(native.local(v['out'],c.root)==p and v['report_sha256']==report['sha256'],'native report identity')
    completion=c.json(dict(path=native.local(v['completion_source'],c.root).relative_to(c.root).as_posix(),
        sha256=v['completion_evidence_sha256']))
    prompt=native.local(completion['prompt'],c.root)
    c.file(dict(path=prompt.relative_to(c.root).as_posix(),sha256=v['prompt_sha256']))
    need(re.fullmatch('native-astra-[br][1-9][0-9]*',v['lane']) and re.fullmatch('[0-9a-f]{32}',v['run_id']), 'native lane/run')
    base='.local/astra-improve/lanes/runs/'+v['lane']+'/'+v['run_id']+'/'
    for name,h in [('receipt.json',side['sha256']),('report.md',report['sha256']),
        ('prompt.txt',v['prompt_sha256']),('completion.json',v['completion_evidence_sha256']),('run.log',v['log_sha256'])]:
        c.file(dict(path=base+name,sha256=h))
    actual=native.verify(c.file(side),c.root)
    completed=dt.datetime.fromisoformat(actual['completed']);recorded=dt.datetime.fromisoformat(actual['recorded_at'])
    need(completed.utcoffset() is not None and recorded.utcoffset() is not None and
         completed<=recorded<=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=5),'native chronology')
    return actual,p.read_text(encoding='utf-8-sig')

def prerequisites(c):
    c.add(c.anchor)
    actual,text=archive(c,c.root/BRIDGE/'RESULT-REVIEW.md')
    need(actual['agent_id']=='/root/k1_result_review' and
         re.findall(r'^[ \t]*VERDICT[ \t]*:.*$',text,re.M)==['VERDICT: PASS-WITH-NOTES'], 'fixed independent bridge authority')
    manifest=read(c.root/BRIDGE/'missing-serving-inputs.json')
    need(manifest['candidate']==KEY and manifest['registration_sha256']==c.anchor[(BRIDGE/'registration.json').as_posix()], 'bridge manifest identity')
    return manifest

def identities(manifest):
    entries=[x['identity'] for x in manifest['small_members']]
    need(len(entries)==10 and [e['id'] for e in entries]==MEMBERS,'ten ordered Small members')
    for e,s,f in zip(entries,[42]*5+[43]*5,list(range(5))*2):
        need(e['family']=='S' and e['seed']==s and e['fold']==f and
             e['arch']=='convnext_small.fb_in22k_ft_in1k' and e['spec_sha256'] and
             all(e[k]==manifest['small_members'][entries.index(e)]['source_pin'][k] for k in ('bytes','sha256')),'Small seed/fold/source identity')
    return entries

def data_snapshot(c,manifest):
    """Read/hash only named same58 rows; no torch/render/model/label access."""
    import numpy as np
    from training.src import raptor_common as common
    need(not c.fixture and common.ROOT==ROOT,'real data binding only in fixed workspace')
    metadata=manifest['corpus']['hb105_original']['metadata']
    for name,pin in metadata.items():c.file(dict(path=CORPUS+'/'+name,sha256=pin['sha256']))
    source=manifest['gold_roster']['source'];bindings=read(c.root/BRIDGE/'registration.json')['bindings']
    pin=bindings[source];pin=pin['sha256'] if isinstance(pin,dict) else pin
    with np.load(c.file(dict(path=source,sha256=pin)),allow_pickle=False) as z:ids=z['ids'].astype(str).tolist()
    need(len(ids)==len(set(ids))==58 and uid_hash(ids)==manifest['gold_roster']['ordered_uid_sha256'] and
         manifest['gold_roster']['targets']==TARGETS,'exact same58 roster and targets')
    corpus=common.Corpus(c.root/CORPUS)
    need(corpus.contract=='raptor64_336_c140_s02' and corpus.depth==64 and len(set(corpus.ids))==len(corpus.ids), 'corpus contract')
    rows=[]
    for u in ids:
        need(u in corpus,'missing corpus ID');volume=corpus.volume(u);mask=corpus.mask(u)
        need(volume.shape==(64,336,336) and volume.dtype==np.uint8 and mask.shape==(64,) and
             np.isin(mask,[0,1]).all(),'real volume/mask shape/type')
        centers=[int(x) for x in common.select_centers(mask,64,36,corpus.spec,'within-slot-valid')]
        row=corpus.row[u];bounds=[b for b in corpus._bounds if b[0]<=row<b[1]]
        need(len(bounds)==1,'unambiguous shard row');a,b,file=bounds[0]
        rows.append(dict(uid=u,volume_sha256=array_hash(volume),mask_sha256=array_hash(mask),
            centers=centers,shape=list(volume.shape),dtype=str(volume.dtype),mask_dtype=str(mask.dtype),
            shard=CORPUS+'/'+file,shard_row=row-a,global_row=row))
    return dict(schema='h199-serving-data/v1',corpus=CORPUS,ids=ids,targets=TARGETS,
        ordered_uid_sha256=uid_hash(ids),rows=rows,metadata={CORPUS+'/'+n:v['sha256'] for n,v in metadata.items()},
        contract=corpus.contract,spec=corpus.spec,source=dict(path=source,sha256=pin),
        renderer_executed=False,model_forwards=0,clean_oof=False)

def data_shape(data):
    need(data['schema']=='h199-serving-data/v1' and data['corpus']==CORPUS and data['targets']==TARGETS and
         len(data['ids'])==len(set(data['ids']))==len(data['rows'])==58 and uid_hash(data['ids'])==data['ordered_uid_sha256'], 'same58 data contract')
    for u,row in zip(data['ids'],data['rows']):
        need(row['uid']==u and row['shape']==[64,336,336] and row['dtype']=='uint8' and
             len(row['centers'])==36 and all(type(x) is int and -1<=x<=62 for x in row['centers']) and
             re.fullmatch('[0-9a-f]{64}',row['volume_sha256']) and re.fullmatch('[0-9a-f]{64}',row['mask_sha256']), 'exact row-byte/center binding')

def command(mode,root=ROOT):return [PYTHON,'-B',str(root/SELF),mode]
def queue(mode,root=ROOT):return [dict(tag=TAGS[mode],command=command(mode,root),continue_on_failure=False)]

def plan(mode):
    return dict(schema='h199-serving-plan/v1',mode=mode,candidate=KEY,version=VERSION,
        representative_index=0,member_ids=MEMBERS,input_shape=[36,3,224,224],targets=TARGETS,
        precision='FP32 CPU reference only' if mode=='cpu' else PRECISION,
        device='cpu' if mode=='cpu' else 'cuda:0',scientific_precision=PRECISION,serial_models=True,views_per_member=2,
        full_cache_permitted=mode=='cache',renderer_rows=58 if mode=='cpu' else 1 if mode=='memory' else 58,
        cpu_forward_member='S-s42-f0',parity_tolerance=1e-6,
        cuda_reference_first_member=mode=='memory',
        peak_limit_bytes=12*GIB,headroom_bytes=2*GIB,executor='B')

def register(c):
    manifest=prerequisites(c);entries=identities(manifest)
    c.add({p:v['sha256'] for p,v in manifest['sources'].items()})
    for e in entries:c.file(dict(path=e['path'],sha256=e['sha256']))
    sources={p:sha(c.root/p) for p in SOURCES};sources.update({p:v['sha256'] for p,v in manifest['sources'].items()})
    c.add(sources);c.file(c.ref(c.work/'SPEC.md'))
    need(not c.fixture,'production register needs actual row bytes')
    need(not any((c.root/'artifacts/logs'/tag).exists() for tag in TAGS.values()),'prospective tag already consumed')
    data=data_snapshot(c,manifest);fresh(c.work/'data.json',data)
    for mode in ('cpu','memory','cache'):
        fresh(c.work/(mode+'-plan.json'),plan(mode))
        if mode!='cpu':fresh(c.work/(mode+'-queue.json'),queue(mode,c.root))
    c.recheck()
    value=dict(schema='h199-serving-registration/v1',candidate=KEY,version=VERSION,builder=BUILDER,
        fixture_only=False,workspace=str(c.root),sources=sources,inherited=c.pins.copy(),
        data=c.ref(c.work/'data.json'),members=entries,plans={m:c.ref(c.work/(m+'-plan.json')) for m in ('cpu','memory','cache')},
        queues={m:c.ref(c.work/(m+'-queue.json')) for m in TAGS},execution_authorized=False,qualified_for_submission=False)
    fresh(c.work/'registration.json',value);return value

def validate(c,raw=False):
    reg=c.json(c.ref(c.work/'registration.json'));manifest=prerequisites(c)
    need(reg['schema']=='h199-serving-registration/v1' and reg['candidate']==KEY and reg['version']==VERSION and
         reg['builder']==BUILDER and reg['fixture_only'] is c.fixture and reg['workspace']==str(c.root),'registration identity')
    need(set(SOURCES)<=set(reg['sources']) and all(reg['sources'].get(p)==v['sha256'] for p,v in manifest['sources'].items()),
         'required original/current source closure');c.add(reg['inherited']);c.add(reg['sources'])
    need(reg['members']==identities(manifest),'exact original member identities')
    for e in reg['members']:c.file(dict(path=e['path'],sha256=e['sha256']))
    data=c.json(reg['data']);data_shape(data)
    need(data['ordered_uid_sha256']==manifest['gold_roster']['ordered_uid_sha256'] and
         data['source']['path']==manifest['gold_roster']['source'] and data['metadata']=={
            CORPUS+'/'+n:v['sha256'] for n,v in manifest['corpus']['hb105_original']['metadata'].items()},'exact original corpus/roster pins')
    inherited=read(c.root/BRIDGE/'registration.json')['bindings'][data['source']['path']]
    need(data['source']['sha256']==(inherited['sha256'] if isinstance(inherited,dict) else inherited),'original same58 source identity')
    c.add(data['metadata']);c.file(data['source'])
    for mode in ('cpu','memory','cache'):need(c.json(reg['plans'][mode])==plan(mode),'fixed phase plan')
    need(set(reg['plans'])=={'cpu','memory','cache'} and set(reg['queues'])==set(TAGS),'complete phase interfaces')
    for mode in TAGS:need(c.json(reg['queues'][mode])==queue(mode,c.root),'exact B-assigned queue')
    if raw:need(data_snapshot(c,manifest)==data,'actual same58 volume/mask drift')
    c.recheck();return reg,data

def prepare(c):
    validate(c)
    value=dict(schema='h199-serving-prepare/v1',registration=c.ref(c.work/'registration.json'),
        executor='B',phases=['BUILD','CPU_SMOKE_LAUNCH','CPU_SMOKE_RESULT','MEMORY_LAUNCH','MEMORY_RESULT','CACHE_LAUNCH','CACHE_RESULT'],
        fixture_only=c.fixture,execution_authorized=False)
    fresh(c.work/'prepare.json',value);return value

def attestation(phase,registration,subject,claim=None,reviewer='ACTUAL_DISTINCT_NATIVE_AGENT'):
    return dict(schema='h199-serving-review/v1',candidate=KEY,version=VERSION,builder=BUILDER,
        reviewer=reviewer,phase=phase,verdict='PASS',registration_sha256=registration,
        subject_sha256=subject,resource_claim_sha256=claim)

def approval(c,phase,subject,claim_sha=None):
    actual,text=archive(c,c.work/(phase+'-REVIEW.md'))
    need(actual['agent_id']!=BUILDER,'builder cannot review own construction')
    need(re.findall(r'^[ \t]*VERDICT[ \t]*:.*$',text,re.M)==['VERDICT: PASS'],'sole independent PASS required')
    lines=[x for x in text.splitlines() if x.lstrip().startswith('H199-SERVING-ATTESTATION')]
    need(len(lines)==1 and lines[0].startswith(PREFIX),'one exact attestation')
    need(native.parse(lines[0][len(PREFIX):].encode())==attestation(phase,sha(c.work/'registration.json'),subject,claim_sha,actual['agent_id']), 'exact independent phase/hash/identity approval')
    need(re.findall(r'^Reviewer-Lane: (\S+)$',text,re.M)==[actual['lane']],'actual native lane')
    c.recheck();return actual

def resource(c,mode):
    item=c.ref(c.work/(mode+'-claim.json'));v=c.json(item)
    need(v['schema']=='h199-serving-resource/v1' and v['candidate']==KEY and v['owner']=='B' and
         v['state']=='CLAIMED' and v['tag']==TAGS[mode] and v['exclusive'] is True and
         v['physical_gpu']==0 and v['logical_device']=='cuda:0' and
         v['device_name']=='NVIDIA GeForce RTX 4070 Ti SUPER' and
         isinstance(v['reservation'],str) and len(v['reservation'])>=8 and
         isinstance(v['shared_evidence'],str) and bool(v['shared_evidence']),'actual fresh exclusive B claim')
    return item

def memory_ok(desktop,peaks):
    values=[desktop['desktop_bytes'],desktop['total_bytes'],*peaks]
    need(len(peaks)>0 and all(type(x) in (int,float) and math.isfinite(x) for x in values) and
         '4070 Ti SUPER' in desktop['device_name'] and 16000*2**20<=desktop['total_bytes']<=16400*2**20 and
         0<=desktop['desktop_bytes']<desktop['total_bytes'] and all(0<=x<=12*GIB for x in peaks) and
         desktop['desktop_bytes']+max(peaks)+2*GIB<=desktop['total_bytes'],'measured 12GiB/2GiB memory gate')

def result(c,mode,reviewed=True,*,draft=None):
    reg,data=validate(c);run=c.work/(mode+'-run')
    need(draft is None or not reviewed,'unpublished result cannot have review')
    item=None if draft is not None else c.ref(run/'result.json')
    v=draft if draft is not None else c.json(item)
    need(v['schema']=='h199-serving-result/v1' and v['mode']==mode and v['candidate']==KEY and
         v['state']=='COMPLETE_PENDING_REVIEW' and v['registration']==c.ref(c.work/'registration.json') and
         v['source_sha256']==reg['sources'][SELF] and v['data']==reg['data'] and
         v['plan']==reg['plans'][mode] and v['member_ids']==MEMBERS and v['failures']==v['fallback_count']==0,
         'complete result identity')
    worker=c.json(v['worker']);process=c.json(v['process']);c.file(v['log'])
    need(process['child_exit_code']==0 and process['owned_cleanup_complete'] is True and not process['survivors'] and
         worker['state']=='COMPLETE' and worker['mode']==mode and worker['source_sha256']==v['source_sha256'] and
         worker['registration']==v['registration'] and worker['data']==reg['data'] and worker['plan']==reg['plans'][mode], 'actual worker/process completion')
    old=c.json(v['admission'])
    replay=Context(c.root,fixture_anchor=c.anchor if c.fixture else None)
    need(old==admission(replay,mode),'actual admission replay');c.add(replay.pins)
    need(worker['admission']==v['admission'],'worker admission binding')
    need(worker['member_ids']==MEMBERS and worker['targets']==TARGETS and worker['ordered_uid_sha256']==data['ordered_uid_sha256'] and
         worker['clean_oof'] is False and worker['exposure']=='EXPOSED','output identity/order/exposure')
    need(worker['failures']==worker['fallback_count']==0 and
         process['source_sha256']==v['source_sha256'] and
         all(worker['environment'].get(k) for k in ('os','python','torch','numpy','timm')) and
         worker['environment']['device']==worker['device'],'runtime/source/failure provenance')
    expected={'cpu':60,'memory':30,'cache':1170}[mode]
    need(len(worker['journals'])==len({x['path'] for x in worker['journals']})==expected,'complete unique phase journals')
    phases={}
    for p in worker['journals']:
        row=c.json(p);need(row['identity']==KEY and row['phase'] in ('model_load','render','prediction') and
            all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('start','end','seconds')) and
            row['seconds']>0 and row['end']>=row['start'] and abs(row['seconds']-(row['end']-row['start']))<.01,
            'complete actual phase timing')
        phases[row['phase']]=phases.get(row['phase'],0)+1
    need(phases==({'render':58,'model_load':1,'prediction':1} if mode=='cpu' else
        {'render':10 if mode=='memory' else 580,'model_load':10,'prediction':10 if mode=='memory' else 580}), 'exact phase workload')
    if mode=='cpu':
        need(worker['device']=='cpu' and worker['renderer_rows']==58 and len(worker['render_checks'])==58 and
             all(type(x['max_abs']) in (int,float) and math.isfinite(x['max_abs']) and 0<=x['max_abs']<=1e-6 for x in worker['render_checks']) and
             worker['forward_member']=='S-s42-f0' and worker['forward_index']==0 and
             math.isfinite(worker['forward_max_abs']) and 0<=worker['forward_max_abs']<=1e-6 and
             worker['shape']==[36,3,224,224] and worker['cuda_initialized'] is False,'actual CPU smoke reference parity')
    else:
        desktop=c.json(v['desktop']);ticket=c.json(v['ticket']);job=c.json(v['bound_job'])
        need(job['schema']=='gpu-job/v2' and job['tag']==TAGS[mode] and job['phase']=='running' and
             job['command']==command(mode,c.root) and job['command_sha256']==jr.command_digest(command(mode,c.root),c.root) and
             job['cwd']==str(c.root) and job['child_identity']==ticket['parent'],'exact bound GPU job')
        need(ticket['mode']==mode and ticket['admission']==v['admission'] and worker['device']=='cuda:0' and
             worker['precision']==PRECISION and worker['physical_gpu']==0 and
             len(worker['model_receipts'])==len(worker['peaks'])==10,'full ten-model CUDA result')
        memory_ok(desktop,worker['peaks'])
        if mode=='memory':need(type(worker['reference_max_abs']) in (int,float) and
            math.isfinite(worker['reference_max_abs']) and 0<=worker['reference_max_abs']<=1e-6,'actual FP16 CUDA reference parity')
        for idx,ref in enumerate(worker['model_receipts']):
            row=c.json(ref)
            need(row['member']==MEMBERS[idx] and row['model_sha256']==reg['members'][idx]['sha256'] and
                 row['peak_reserved_bytes']==worker['peaks'][idx] and 0<row['peak_allocated_bytes']<=row['peak_reserved_bytes'] and
                 row['synchronized'] is True and row['deleted'] is True and row['views']==2 and
                 row['shape']==[36,3,224,224] and row['studies']==(1 if mode=='memory' else 58) and
                 row['registration']==v['registration'] and row['environment']==worker['environment'], 'raw measured model provenance')
        need(worker['npz'] is None if mode=='memory' else worker['npz'] is not None,'measurement cannot populate cache')
        if mode=='cache':check_npz(c.file(worker['npz']),data)
    if reviewed:approval(c,{'cpu':'CPU_SMOKE_RESULT','memory':'MEMORY_RESULT','cache':'CACHE_RESULT'}[mode],item['sha256'])
    c.recheck();return v

def admission(c,mode):
    # Production approvals rehash the actual corresponding volume/mask bytes.
    # TEST-ONLY fixtures exercise the same approval parser without real images.
    reg,_=validate(c,raw=not c.fixture);prep=c.json(c.ref(c.work/'prepare.json'))
    need(prep['schema']=='h199-serving-prepare/v1' and prep['registration']==c.ref(c.work/'registration.json') and
         prep['fixture_only'] is c.fixture and prep['executor']=='B','exact prepared boundary')
    build=approval(c,'BUILD',sha(c.work/'registration.json'))
    if mode!='cpu':result(c,'cpu')
    if mode=='cache':result(c,'memory')
    claim=None if mode=='cpu' else resource(c,mode)['sha256']
    phase={'cpu':'CPU_SMOKE_LAUNCH','memory':'MEMORY_LAUNCH','cache':'CACHE_LAUNCH'}[mode]
    launch=approval(c,phase,reg['plans'][mode]['sha256'],claim)
    need(dt.datetime.fromisoformat(launch['completed'])>=dt.datetime.fromisoformat(build['completed']),'launch precedes construction review')
    c.recheck()
    return dict(schema='h199-serving-admission/v1',mode=mode,state='ADMITTED_'+mode.upper()+'_ONLY',
        candidate=KEY,registration=c.ref(c.work/'registration.json'),fixture_only=c.fixture,
        consumed_sha256=c.pins.copy(),qualified_for_submission=False)

def verify_admission(c,mode):
    saved=read(c.work/(mode+'-admission.json'));fresh_value=admission(c,mode)
    need(saved==fresh_value,'admission/current evidence mismatch');c.recheck();return saved

def check_npz(path,data):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)=={'ids','targets','member_ids','probabilities'} and z['ids'].tolist()==data['ids'] and
             z['targets'].tolist()==TARGETS and z['member_ids'].tolist()==MEMBERS,'exact NPZ roster/target/member order')
        p=z['probabilities'];need(p.shape==(10,58,12) and p.dtype==np.float32 and
             np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'ten x58 x12 finite FP32')

def snapshot():
    cp=subprocess.run(['nvidia-smi','-i','0','--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    name,total,used,util,power=[x.strip() for x in cp.stdout.strip().split(',')]
    v=dict(device_name=name,total_bytes=float(total)*2**20,desktop_bytes=float(used)*2**20,util=float(util),power=float(power))
    need(all(math.isfinite(v[k]) for k in ('total_bytes','desktop_bytes','util','power')),'invalid live GPU snapshot')
    return v

def optional_snapshot():
    try:return snapshot()
    except (OSError,ValueError,subprocess.SubprocessError):return None

def check_job(item,mode,own,parent,actual_command,parent_args):
    need(item['schema']=='gpu-job/v2' and item['phase']=='running' and item['tag']==TAGS[mode] and
         item['cwd']==str(ROOT) and item['command']==command(mode) and
         item['command_sha256']==jr.command_digest(command(mode),ROOT) and item['child_identity']==own and
         dict(pid=item['pid'],create_time=item['create_time'])==parent and actual_command==command(mode), 'exact gpu_job PID/start/command')
    need(str(ROOT/'tools/gpu_job.py') in parent_args and
         parent_args[-3:]==['--_run-job',TAGS[mode],item['run_id']] and
         jr.liveness(own)==jr.liveness(parent)=='alive','actual owned gpu_job parent')
    return item

def job_identity(mode,own=None):
    import psutil
    own=own or jr.identity();proc=psutil.Process(own['pid']);parent=jr.identity(proc.ppid());last=None
    for _ in range(40):
        try:return check_job(read(ROOT/'artifacts/logs'/TAGS[mode]/'job.json'),mode,own,parent,
            proc.cmdline(),psutil.Process(parent['pid']).cmdline())
        except (ValueError,KeyError,OSError) as exc:last=exc;time.sleep(.05)
    raise ValueError('gpu_job identity handoff failed') from last

def supervise(cmd,run,env,token,cpu,*,poll=1.,sample=optional_snapshot):
    """INF2 controller, owned cleanup even if authoritative publication fails."""
    import psutil
    child=root=None;owned={};survivors=[];code=125;failure=None;started=time.monotonic()
    policy=Policy(started,total=None);identity=None;last_units=0;last_sequence=-1
    try:
        with (run/'worker.log').open('xb',buffering=0) as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            root=psutil.Process(child.pid);identity=jr.identity(child.pid)
            while True:
                collect_owned(root,owned)
                if child.poll() is not None:code=child.returncode;break
                beat=None
                try:beat=read(run/'phase.json')
                except (OSError,ValueError):pass
                now=time.monotonic()
                if beat:
                    need(beat['pid']==identity['pid'] and beat['token']==token and
                         identity['create_time']-1<=beat['wall_time']<=time.time()+2 and
                         type(beat['sequence']) is int and beat['sequence']>=last_sequence and
                         type(beat['detail']['cumulative_units']) is int and
                         last_units<=beat['detail']['cumulative_units']<=580 and
                         beat['phase'] in ('startup','model_load','render','prediction','study_complete','finished'), 'forged/regressing owned heartbeat')
                    last_sequence=beat['sequence'];last_units=beat['detail']['cumulative_units']
                    rate=beat['detail'].get('measured_rate')
                    if rate is not None:
                        need(type(rate) in (int,float) and math.isfinite(rate) and rate>0,'actual finite throughput rate')
                        policy.rates['prediction']=max(policy.rates.get('prediction',0),rate)
                why=policy.check(now,beat,None)
                telemetry=None if cpu else sample()
                if telemetry and now-policy.last>=600 and telemetry['util']>=95 and telemetry['power']<90 and telemetry['desktop_bytes']/telemetry['total_bytes']>=.90:
                    why=why or 'WDDM stall 600s'
                need(not why,why or '')
                diagnostic(run/'status.json',dict(phase=policy.phase,units=last_units,seconds=now-started,pid=os.getpid()))
                time.sleep(poll)
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc);code=125
    finally:
        try:
            if root is not None:survivors=terminate_owned(root,owned)
        except BaseException as exc:survivors=[dict(error=type(exc).__name__)]
        if child is not None:
            try:
                if child.poll() is None:child.kill()
                child.wait(timeout=5)
            except BaseException as exc:survivors.append(dict(error=type(exc).__name__))
        if survivors:code=125
        value=dict(schema='h199-serving-process/v1',child_exit_code=code,child=identity,
            owned_cleanup_complete=not survivors,survivors=survivors,failure=failure,
            start_monotonic=started,end_monotonic=time.monotonic(),seconds=time.monotonic()-started,
            source_sha256=sha(ROOT/SELF))
        fresh(run/'process.json',value)
    return value

def atomic_npz(path,**arrays):
    import numpy as np
    path=Path(path);need(not path.exists(),'immutable NPZ exists');tmp=path.with_suffix('.partial.npz')
    def write():
        with tmp.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    io.retry(write);io.rename(tmp,path)

def worker(mode):
    import psutil
    c=Context();run=c.work/(mode+'-run');ticket=read(run/'ticket.json')
    need(ticket['mode']==mode and ticket['parent']==jr.identity(os.getppid()) and
         ticket['token_sha256']==jr.digest(os.environ.get('INF2_SERVING_TOKEN','')) and
         psutil.Process(os.getppid()).cmdline()==command(mode),'owned worker token/PID/start/command')
    if mode!='cpu':job_identity(mode,ticket['parent'])
    expected=c.ref(c.work/(mode+'-admission.json'));need(ticket['admission']==expected,'actual admission bytes')
    verify_admission(c,mode);reg,data=validate(c,raw=True)
    need(not c.fixture and ticket['registration']==c.ref(c.work/'registration.json'),'production registration boundary')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==('-1' if mode=='cpu' else '0'),'exact CUDA visibility')
    from tools import hb105_serving_v101 as helper
    modules=helper.imports();np,torch,mil,serving=modules
    serving.backend(strict=True)
    device='cpu' if mode=='cpu' else 'cuda:0'
    if mode!='cpu':
        torch.cuda.set_device(0);need(torch.cuda.device_count()==1,'one visible GPU')
    else:need(not torch.cuda.is_initialized(),'CPU smoke cannot initialize CUDA')
    import timm
    env=dict(os=sys.platform,python=sys.version,torch=torch.__version__,numpy=np.__version__,timm=timm.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=device,
        gpu='cpu' if mode=='cpu' else torch.cuda.get_device_name(0),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),deterministic=torch.are_deterministic_algorithms_enabled())
    corpus=mil.common.Corpus(ROOT/CORPUS);journals=[];sequence=0;units=0;rate=None;started=time.monotonic()
    def progress(phase,committed=False):
        nonlocal sequence
        sequence+=1
        diagnostic(run/'phase.json',dict(pid=os.getpid(),token=os.environ['INF2_SERVING_TOKEN'],wall_time=time.time(),
            completed_at=time.monotonic(),sequence=sequence,event='commit' if committed else 'enter',phase=phase,
            detail=dict(progress_kind='prediction' if phase in ('prediction','study_complete') else phase,
                cumulative_units=units,measured_rate=rate)))
    def event(label,index,begin,**extra):
        path=run/'journal'/f'{len(journals):04}.json'
        value=dict(identity=KEY,phase=label,index=index,start=begin,end=time.monotonic(),seconds=time.monotonic()-begin,**extra)
        io.journal(path,value,run);journals.append(c.ref(path))
    def windows(i):
        row=data['rows'][i];u=data['ids'][i];vol=corpus.volume(u);mask=corpus.mask(u)
        need(array_hash(vol)==row['volume_sha256'] and array_hash(mask)==row['mask_sha256'],'current volume/mask bytes')
        centers=[int(x) for x in mil.common.select_centers(mask,64,36,corpus.spec,'within-slot-valid')]
        need(centers==row['centers'],'current centres');x=serving.mil_windows(vol,centers)
        need(x.shape==(36,3,224,224) and x.dtype==np.float32 and np.isfinite(x).all(),'original full renderer shape')
        return x,vol,centers
    payload={}
    if mode=='cpu':
        checks=[]
        for i in range(58):
            progress('render');tick=time.monotonic();x,vol,centers=windows(i)
            reference=(mil.scale_windows(vol,centers)-np.asarray([.485,.456,.406],np.float32).reshape(3,1,1))/np.asarray([.229,.224,.225],np.float32).reshape(3,1,1)
            error=float(np.max(np.abs(x-reference)));need(error<=1e-6,'original renderer reference parity')
            checks.append(dict(index=i,max_abs=error,windows_sha256=array_hash(x)))
            event('render',i,tick);progress('study_complete',True)
        progress('model_load');tick=time.monotonic();entry=reg['members'][0]
        model=helper.load_model(ROOT/entry['path'],entry,'cpu',modules)
        try:
            event('model_load',0,tick);x,_,_=windows(0);progress('prediction');tick=time.monotonic()
            value,measures=helper.prediction(model,x,data['ids'][0],corpus,'cpu',modules)
            reference=serving.mil_prediction(model,x,data['ids'][0],corpus,'cpu')
            error=float(np.max(np.abs(value-reference)));need(error<=1e-6,'CPU full-shape forward reference parity')
            units=1;event('prediction',0,tick,helper_timings=measures);progress('study_complete',True)
        finally:del model
        need(not torch.cuda.is_initialized(),'CPU route initialized CUDA')
        payload=dict(renderer_rows=58,render_checks=checks,forward_member=MEMBERS[0],forward_index=0,
            forward_max_abs=error,shape=[36,3,224,224],cuda_initialized=False,precision='FP32 CPU reference only')
    else:
        desktop=read(run/'desktop.json');selected=[0] if mode=='memory' else list(range(58))
        values=None if mode=='memory' else np.empty((10,58,12),np.float32);receipts=[];peaks=[];reference_error=None
        for j,entry in enumerate(reg['members']):
            progress('model_load');tick=time.monotonic();model=None
            torch.cuda.synchronize(0);torch.cuda.reset_peak_memory_stats(0)
            try:
                model=helper.load_model(ROOT/entry['path'],entry,'cuda:0',modules)
                torch.cuda.synchronize(0);event('model_load',j,tick)
                for i in selected:
                    progress('render');tick=time.monotonic();x,_,_=windows(i);event('render',i,tick,member=j)
                    progress('prediction');tick=time.monotonic()
                    value,measures=helper.prediction(model,x,data['ids'][i],corpus,'cuda:0',modules)
                    duration=time.monotonic()-tick
                    need(value.shape==(12,) and value.dtype==np.float32 and np.isfinite(value).all() and
                         ((value>=0)&(value<=1)).all(),'complete original probability output')
                    if values is not None:values[j,i]=value
                    if mode=='memory' and j==0:
                        reference_start=time.monotonic()
                        reference=serving.mil_prediction(model,x,data['ids'][i],corpus,'cuda:0')
                        reference_error=float(np.max(np.abs(value-reference)))
                        need(reference_error<=1e-6,'original FP16 CUDA full-shape reference parity')
                        measures=dict(measures,reference_seconds=time.monotonic()-reference_start,reference_max_abs=reference_error)
                    torch.cuda.synchronize(0);memory_ok(desktop,[torch.cuda.max_memory_reserved(0)])
                    units+=1;rate=max(rate or 0,1/max(duration,1e-9))
                    event('prediction',i,tick,member=j,helper_timings=measures);progress('study_complete',True)
                torch.cuda.synchronize(0);allocated=torch.cuda.max_memory_allocated(0);reserved=torch.cuda.max_memory_reserved(0)
                need(0<allocated<=reserved,'positive synchronized allocated/reserved peaks');memory_ok(desktop,[reserved]);peaks.append(reserved)
            finally:
                if model is not None:del model
                torch.cuda.empty_cache();torch.cuda.synchronize(0)
            row=dict(member=MEMBERS[j],model_sha256=entry['sha256'],studies=len(selected),views=2,shape=[36,3,224,224],
                peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,synchronized=True,deleted=True,
                registration=c.ref(c.work/'registration.json'),environment=env)
            p=run/f'model-{j}.json';fresh(p,row);receipts.append(c.ref(p))
        payload=dict(physical_gpu=0,precision=PRECISION,model_receipts=receipts,peaks=peaks,npz=None,reference_max_abs=reference_error)
        if mode=='cache':
            need(units==580,'all ten x58 study commits')
            atomic_npz(run/'probabilities.npz',ids=np.asarray(data['ids']),targets=np.asarray(TARGETS),
                member_ids=np.asarray(MEMBERS),probabilities=values)
            check_npz(run/'probabilities.npz',data);payload['npz']=c.ref(run/'probabilities.npz')
        else:need(units==10,'all ten predetermined full-shape measurements')
    c.recheck();progress('finished',True)
    out=dict(schema='h199-serving-worker/v1',state='COMPLETE',mode=mode,source_sha256=reg['sources'][SELF],
        registration=c.ref(c.work/'registration.json'),data=reg['data'],plan=reg['plans'][mode],admission=ticket['admission'],
        member_ids=MEMBERS,targets=TARGETS,ordered_uid_sha256=data['ordered_uid_sha256'],
        device=device,environment=env,journals=journals,failures=0,fallback_count=0,
        start=started,end=time.monotonic(),seconds=time.monotonic()-started,
        exposure='EXPOSED',clean_oof=False,**payload)
    fresh(run/'worker.json',out);return out

def finish(c,mode,process):
    run=c.work/(mode+'-run');reg,data=validate(c,raw=not c.fixture)
    need(process['child_exit_code']==0 and process['owned_cleanup_complete'] is True and not process['survivors'], 'failed worker/unconfirmed cleanup')
    worker=c.json(c.ref(run/'worker.json'))
    need(worker['state']=='COMPLETE' and worker['mode']==mode and worker['failures']==worker['fallback_count']==0 and
         worker['registration']==c.ref(c.work/'registration.json') and worker['source_sha256']==reg['sources'][SELF] and
         worker['data']==reg['data'] and worker['plan']==reg['plans'][mode] and
         worker['admission']==c.ref(c.work/(mode+'-admission.json')), 'exact completed worker')
    c.recheck()
    v=dict(schema='h199-serving-result/v1',state='COMPLETE_PENDING_REVIEW',mode=mode,candidate=KEY,
        registration=c.ref(c.work/'registration.json'),data=reg['data'],plan=reg['plans'][mode],
        source_sha256=reg['sources'][SELF],member_ids=MEMBERS,failures=0,fallback_count=0,
        worker=c.ref(run/'worker.json'),process=c.ref(run/'process.json'),log=c.ref(run/'worker.log'),
        admission=c.ref(c.work/(mode+'-admission.json')),qualified_for_submission=False)
    if mode!='cpu':v.update(desktop=c.ref(run/'desktop.json'),ticket=c.ref(run/'ticket.json'),bound_job=c.ref(run/'bound-job.json'))
    result(Context(c.root,fixture_anchor=c.anchor if c.fixture else None),mode,reviewed=False,draft=v)
    fresh(run/'result.json',v)
    return v

def run(mode):
    c=Context();need(os.name=='nt','Windows execution contract')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==('-1' if mode=='cpu' else '0'),'exact device visibility')
    verify_admission(c,mode);validate(c,raw=True)
    job=None;desktop=None
    if mode!='cpu':
        job=job_identity(mode);desktop=snapshot();memory_ok(desktop,[0])
        if mode=='cache':
            measured=read(c.work/'memory-run/worker.json');memory_ok(desktop,measured['peaks'])
        cp=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
            capture_output=True,text=True,check=True,timeout=3,creationflags=subprocess.CREATE_NO_WINDOW)
        for line in cp.stdout.splitlines():
            parts=line.split(',',1)
            if len(parts)==2 and 'python' in parts[1].lower():
                need(parts[0].strip() in (str(os.getpid()),str(job['pid'])),'other Python GPU owner')
    out=c.work/(mode+'-run');out.mkdir(exist_ok=False);(out/'journal').mkdir();token=uuid.uuid4().hex
    fresh(out/'ticket.json',dict(mode=mode,parent=jr.identity(),token_sha256=jr.digest(token),
        admission=c.ref(c.work/(mode+'-admission.json')),registration=c.ref(c.work/'registration.json')))
    if mode!='cpu':fresh(out/'desktop.json',desktop);fresh(out/'bound-job.json',job)
    env=dict(os.environ,INF2_SERVING_TOKEN=token,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        process=supervise([PYTHON,'-B',str(ROOT/SELF),'_'+mode],out,env,token,mode=='cpu')
        return finish(c,mode,process)
    except BaseException as exc:
        diagnostic(out/'failure.json',dict(state='FAILED',error=type(exc).__name__,automatic_retry=False));raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register','validate','prepare']+
        [p+m for m in ('cpu','memory','cache') for p in ('admit-','dry-run-','result-')]+
        ['cpu','memory','cache','_cpu','_memory','_cache'])
    args=parser.parse_args(argv);c=Context();cmd=args.command
    try:
        if cmd=='register':v=register(c)
        elif cmd=='validate':validate(c,raw=True);v=dict(status='STATIC_VALID_PRE_SMOKE')
        elif cmd=='prepare':v=prepare(c)
        elif cmd.startswith(('admit-','dry-run-')):
            mode=cmd.split('-')[-1];v=admission(c,mode)
            if cmd.startswith('admit-'):fresh(c.work/(mode+'-admission.json'),v)
        elif cmd.startswith('result-'):result(c,cmd[7:],reviewed=False);v=dict(status='RAW_RESULT_VALID_REVIEW_REQUIRED')
        elif cmd.startswith('_'):v=worker(cmd[1:])
        else:v=run(cmd)
        print(json.dumps({k:v[k] for k in ('schema','status','state','fixture_only','qualified_for_submission') if k in v}));return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='HOLD',error=str(exc),execution_authorized=False)));return 2

if __name__=='__main__':raise SystemExit(main())
