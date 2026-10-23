"""RN3-only cold T4 cost experiment; passive import, metadata-only prepare.

Original all-six producer/runtime files are immutable. prepare transports the
saved engineering NPZ as opaque bytes; array interpretation occurs only in run.
No training, labels, anchor, B model inference or competition submission.
"""
from __future__ import annotations
import argparse
import base64
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE='docs/handoffs/evidence/2026-10/2026-10-04-rn3-cost-v1'
STAGE='.local/rn3-cost-v1/20261004-v1'
SCHEMA='RN3_COLD_T4_COST_V1'
NAMES=['RN_F0','RN_F1','RN_F2']
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
INDICES=[1,3,5]
ASSET_SHA='40f5a04dd0f770f05e9af5cf9a3827ccf5748c7989847c0778ba189fde40f80e'
PRODUCER_SHA='bd0fcaf4a2ddb9c8eb975a80f9ef1ae5250bd36d7b5ea6a4eefbf371014c7b5b'
RESOURCE_SHA='15c0fe61d5d0b350557093ec38a1dc902b73e12c76328c4951934cc88f1da954'
PARITY_SHA='298f11855c7e8e86007be612bf7e0831469da54d601aeedabab9f85a6259754b'
OPENCV_SHA='236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'
BUDGET={'internal_seconds':900,'platform_seconds':1200,'rss_bytes':8*1024**3,'output_bytes':64*1024**2,'retries':0,'cpu_threads':4,'visible_T4s':1}


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def load(path,name,digest):
    need(sha(path)==digest,'source identity changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def size_tree(root):
    total=0
    for p in Path(root).rglob('*'):
        try:s=p.stat()
        except FileNotFoundError:continue
        if stat.S_ISREG(s.st_mode):total+=s.st_size
    return total


def check_budget(elapsed,rss,output):
    need(math.isfinite(elapsed) and 0<=elapsed<900,'whole cold deadline exceeded')
    need(0<=rss<=BUDGET['rss_bytes'],'sampled tree RSS exceeds8GiB')
    need(0<=output<=BUDGET['output_bytes'],'output exceeds64MiB')


def validate_subset(members,all_members):
    expected=[all_members[i] for i in INDICES]
    need([m['name'] for m in members]==NAMES and members==expected,'exact RN three-member subset required')
    need(all(m['family']=='RN' and m['arch']=='resnet34.a1_in1k' and m['epoch_1based']==20 for m in members),'RN architecture/epoch differs')


def numerical_gate(raw,reference,ids,expected,names,targets,parity):
    import numpy as np
    need(ids==expected and len(ids)==100 and len(set(ids))==100,'exact complete ordered100 required')
    need(names==NAMES and targets==TARGETS,'RN member/target order differs')
    for value in (raw,reference):
        need(value.dtype==np.float32 and value.shape==(3,100,12),'float32 RN3x100x12 required')
        need(bool(np.isfinite(value).all() and (value>=0).all() and (value<=1).all()),'invalid raw probabilities')
    ranks=np.stack([parity.rank_pct(x) for x in raw]);ref_ranks=np.stack([parity.rank_pct(x) for x in reference])
    bound=np.stack([parity.rank_bound(x,.002) for x in reference]);rows=[]
    limits={'max_abs':.002,'p99_abs':.002,'mean_abs':.0005}
    for i,name in enumerate(NAMES):
        for j,target in enumerate(targets):
            delta=np.abs(raw[i,:,j].astype(np.float64)-reference[i,:,j].astype(np.float64))
            stats=parity.metric_delta(delta);rankdelta=np.abs(ranks[i,:,j]-ref_ranks[i,:,j])
            rows.append(dict(member=name,target=target,errors=stats,probability_pass=all(stats[k]<=v for k,v in limits.items()),rank_pass=bool((rankdelta<=bound[i,:,j]+1e-15).all())))
    mean_rank=ranks.mean(0,dtype=np.float64);mean_reference=ref_ranks.mean(0,dtype=np.float64)
    mean_pass=bool((np.abs(mean_rank-mean_reference)<=bound.mean(0)+1e-15).all())
    return dict(status='PASS' if all(r['probability_pass'] and r['rank_pass'] for r in rows) and mean_pass else 'FAIL',limits=limits,rtol=0.,comparisons=rows,
        mean_per_fold_rank_pass=mean_pass,rank_scope='whole100 eachfold/target average ties/N, then equal mean; no final rerank'),mean_rank


class Timers:
    """Additive observation; synchronized boundaries do not mutate tensors."""
    def __init__(self,synchronize):self.sync=synchronize;self.seconds={};self.counts={};self.sync_seconds=0.;self.hook_starts={}
    def boundary(self):
        t=time.perf_counter();self.sync();self.sync_seconds+=time.perf_counter()-t
    def call(self,name,fn,*a,**k):
        self.boundary();t=time.perf_counter()
        try:return fn(*a,**k)
        finally:
            self.boundary();self.seconds[name]=self.seconds.get(name,0.)+time.perf_counter()-t;self.counts[name]=self.counts.get(name,0)+1
    def pre(self,module,args):self.boundary();self.hook_starts[id(module)]=time.perf_counter()
    def post(self,module,args,result):
        self.boundary();start=self.hook_starts.pop(id(module));self.seconds['forward']=self.seconds.get('forward',0.)+time.perf_counter()-start;self.counts['forward']=self.counts.get('forward',0)+1


class TimedCorpus:
    def __init__(self,base,timer):
        self.base=base;self.timer=timer
        self.ids,self.contract,self.depth,self.spec=base.ids,base.contract,base.depth,base.spec
    def __contains__(self,u):return u in self.base
    def mask(self,u):return self.timer.call('dicom_access',self.base.mask,u)
    def volume(self,u):return self.timer.call('dicom_access',self.base.volume,u)


def check_protocol(bundle,digest):
    bundle=Path(bundle);need(sha(bundle/'protocol.json')==digest,'protocol identity differs');p=read(bundle/'protocol.json')
    need(p['schema']==SCHEMA and p['budget']==BUDGET,'fixed cost experiment changed')
    for name,value in p['files'].items():need(sha(bundle/name)==value,'bundle changed: '+name)
    validate_subset(p['members'],p['all_members']);need(len(p['ids'])==100 and len(set(p['ids']))==100,'protocol100')
    need(hashlib.sha256('\n'.join(p['ids']).encode()).hexdigest()==p['ordered_uids_sha256'],'protocol UID order differs')
    return p


def worker(args):
    started=time.monotonic();bundle,assets,work=map(Path,(args.bundle,args.assets,args.work));timing={};members=[];hooks=[];source=None;timer=None;primary=None
    p=check_protocol(bundle,args.protocol_sha256)
    need(sys.platform.startswith('linux'),'actual Linux T4 worker required')
    need(sha(assets/'b-rn3f-assets.json')==ASSET_SHA,'original assets manifest differs')
    original=read(assets/'b-rn3f-assets.json');validate_subset(p['members'],original['members'])
    producer=load(assets/'b_rn3f_serving.py','rn3_original_producer',PRODUCER_SHA)
    need(not any(k in sys.modules for k in ['torch','timm','pydicom','raptor_common']),'fresh cold worker required')
    # Reuse exactly the original environment path. It verifies original asset
    # hashes but does not load any B checkpoint as a model.
    environment_root=Path(args.environment_root)
    need(environment_root==Path('/kaggle/temp/rn3-cost-v1-environment') and environment_root.is_dir(),'owned scratch environment required')
    t=time.perf_counter();producer.activate(assets,environment_root,'t4-engineering',args.opencv_wheel);timing['setup_validate_offline_import']=time.perf_counter()-t
    import numpy as np
    import torch
    import cv2
    from raptor_common import environment
    from raptor_dicom import DicomCorpus,DirectoryDicomSource
    from raptor_ensemble import load_members,predict_ensemble,validate_members
    import raptor_runtime
    allocation=producer.cpu_allocation(4);need(allocation['selected_threads']==4,'requires actual CPU allocation at least4 and internal4')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(7,5) and 'T4' in torch.cuda.get_device_name(0),'exact one visible TeslaT4 required')
    runtime=raptor_runtime.configure_runtime('fp16','cuda',policy='strict');timer=Timers(torch.cuda.synchronize)
    subset=[{**m,'path':str(assets/m['path'])} for m in p['members']]
    write(work/'rn3-members.json',{'members':subset})
    original_prepare=raptor_runtime.prepare_input
    receipt={'schema':SCHEMA,'status':'RUNNING','protocol_sha256':args.protocol_sha256,'pid':os.getpid(),'timing':timing,'allocation':allocation,'runtime':runtime,'no_B_or_anchor_models':True}
    try:
        members,_=timer.call('model_load_cpu_hash_and_deserialize',load_members,work/'rn3-members.json')
        need(validate_members(members)==p['input'],'loaded input contract differs')
        for member,expected in zip(members,p['members']):
            need(member.name==expected['name'] and member.checkpoint_sha256==expected['sha256'] and member.metadata['arch']==expected['arch'] and member.metadata['tag']==expected['training_run'] and member.metadata['epoch']+1==20,'loaded RN provenance differs')
            hooks.extend([member.model.register_forward_pre_hook(timer.pre),member.model.register_forward_hook(timer.post)])
        ids=p['ids'];uid_csv=work/'cohort.csv'
        with uid_csv.open('x',newline='',encoding='utf-8') as f:writer=csv.writer(f);writer.writerow(['StudyInstanceUID']);writer.writerows([[u] for u in ids])
        t=time.perf_counter();source=DirectoryDicomSource(Path(args.competition)/'train_series')
        corpus=DicomCorpus(uid_csv,Path(args.competition)/'train_series.csv',source,p['input']['contract'],'train')
        timing['corpus_metadata_setup']=time.perf_counter()-t
        observed=producer.ObservedCorpus(TimedCorpus(corpus,timer))
        def timed_prepare(*a,**k):return timer.call('prepare_input_including_dicom',original_prepare,*a,**k)
        raptor_runtime.prepare_input=timed_prepare
        mean,raw,inference=timer.call('inference_including_residency_and_runtime_cleanup',predict_ensemble,members,observed,ids,amp='fp16',device='cuda',batch=1,residency='resident',runtime_policy='strict')
        need(inference['status']=='COMPLETE' and inference['studies']==100 and inference['member_count']==3 and inference['fp32_fallbacks']==inference['neutral_fills']==0 and not inference['failures'] and not inference.get('cleanup_failures'),'inference incomplete/fallback/cleanupfailure')
        need(inference['input_preparations']==inference['completed_batches']==100 and timer.counts.get('forward')==300,'exact onecold300forward/100input contract')
        need([r['name'] for r in inference['member_results']]==NAMES and all(r['completed_studies']==100 for r in inference['member_results']),'RN completion order/count')
        # Reference is saved original T4 engineering output, not labels or OOF.
        t=time.perf_counter();refreceipt=read(bundle/'reference-receipt.json')
        need(refreceipt['status']=='COMPLETE' and refreceipt['studies']==100 and refreceipt['output_sha256']['members.npz']==p['files']['reference-members.npz'],'original reference receipt differs')
        with np.load(bundle/'reference-members.npz',allow_pickle=False) as ref:
            need(ref['study_uids'].tolist()==ids and ref['member_names'].tolist()==[m['name'] for m in p['all_members']] and ref['targets'].tolist()==p['targets'],'reference exact100/members/targets')
            reference=ref['member_probabilities'][INDICES]
        parity=load(bundle/'b_rn3f_parity.py','rn3_fixed_parity',PARITY_SHA)
        numeric,ranks=numerical_gate(raw,reference,ids,ids,NAMES,p['targets'],parity)
        input_exact=all(observed.records[u]==refreceipt['runtime']['input_evidence'][i] for i,u in enumerate(ids))
        normalized_exact=inference['input_tensor_hashes']==refreceipt['inference']['input_tensor_hashes']
        need(input_exact and normalized_exact,'sameT4/thread4 original volume/mask/centers/normalizedinput linkage differs')
        timing['reference_and_numerical_validation']=time.perf_counter()-t
        t=time.perf_counter()
        with (work/'raw.npz').open('xb') as f:np.savez_compressed(f,study_uids=np.asarray(ids),member_names=np.asarray(NAMES),targets=np.asarray(p['targets']),member_probabilities=raw,mean_per_fold_rank=ranks)
        write(work/'numeric.json',numeric);write(work/'inference.json',inference)
        timing['serialization']=time.perf_counter()-t
        receipt.update(status='COMPLETE' if numeric['status']=='PASS' else 'NUMERICAL_FAIL',numerical_status=numeric['status'],studies=100,members=NAMES,targets=p['targets'],forward_calls=300,
            source_input_exact=input_exact,normalized_input_exact=normalized_exact,environment=environment(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        need(numeric['status']=='PASS','original numerical tolerance failed')
    except BaseException as exc:
        primary=exc;receipt.update(status='FAILED',error={'type':type(exc).__name__,'message':str(exc)})
        if hasattr(exc,'receipt'):receipt['inference_failure_receipt']=exc.receipt
        raise
    finally:
        cleanup=[];t=time.perf_counter();raptor_runtime.prepare_input=original_prepare
        for hook in hooks:
            try:hook.remove()
            except BaseException as exc:cleanup.append(repr(exc))
        for member in members:
            try:
                need(all(x.device.type=='cpu' for x in member.model.parameters()),'runtime did not release RN from GPU')
            except BaseException as exc:cleanup.append(repr(exc))
        try:
            if source is not None:source.close()
            members.clear();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        except BaseException as exc:cleanup.append(repr(exc))
        timing['post_inference_final_cleanup']=time.perf_counter()-t
        if timer is not None:
            timing.update(timer.seconds);receipt['timer_counts']=timer.counts;receipt['cuda_sync_wall_seconds']=timer.sync_seconds
        receipt['timing_scope']='nested observational timers; dicom_access includes cache/decode calls, not pure decoder; inference includes placement+original modelCPUcleanup; sync overhead included, do not sum overlapping fields'
        receipt['cleanup']={'clean':not cleanup,'errors':cleanup};receipt['whole_worker_seconds']=time.monotonic()-started
        receipt['output_bytes_before_receipt']=size_tree(work)
        if cleanup:receipt['status']='FAILED'
        write(work/'worker-receipt.json',receipt)
        if cleanup and primary is None:raise RuntimeError('cleanup failure')


def validate_resource(r):
    need(r['schema']=='process-resource-receipt-v1' and r['supervisor_sha256']==RESOURCE_SHA and r['retries']==0,'resource identity/retry differs')
    need(r['status']=='COMPLETE' and r['wrapper_exit_code']==r['child_exit_code']==r['child_exit_code_before_cleanup']==0 and r['failure'] is None,'resource/child did not naturally complete')
    need(r['sample_count']>0 and r['failed_sample_attempts']==r['access_failure_count']==0 and r['cgroup_telemetry_complete'] is True,'resource observation incomplete')
    need(not any(r['cleanup'][k] for k in ['actions','errors','survivors']),'owned descendants not naturally reaped')
    before=r['cgroup_start']['values'].get('memory.events',{});after=r['cgroup_end']['values'].get('memory.events',{})
    need(all(after.get(k,0)==before.get(k,0) for k in ['oom','oom_kill','oom_group_kill']),'new OOM event')


def publish_result(work,result,start,rss):
    """Late serialization/deadline failure cannot leave a stale PASS receipt."""
    work=Path(work);temporary=work/'result.pending.json';final=work/'result.json'
    try:
        write(temporary,result);check_budget(time.monotonic()-start,rss,size_tree(work))
        os.replace(temporary,final);check_budget(time.monotonic()-start,rss,size_tree(work))
    except BaseException:
        for path in (temporary,final):
            if path.exists():path.unlink()
        raise


def run(args):
    start=float(args.started_monotonic);bundle=Path(args.bundle);p=check_protocol(bundle,args.protocol_sha256);work=Path(args.work)
    need(sys.platform.startswith('linux'),'benchmark run is explicit remoteLinux only');work.mkdir(parents=True,exist_ok=False)
    resource=load(bundle/'process_resource_receipt.py','rn3_resource',RESOURCE_SHA)
    sampler=resource.OwnedTree.sample
    def capped_sample(tree):
        sample=sampler(tree)
        check_budget(time.monotonic()-start,sample['rss_bytes'],size_tree(work))
        return sample
    resource.OwnedTree.sample=capped_sample
    remaining=900-(time.monotonic()-start)-15;need(remaining>0,'no remaining cold budget')
    environment_root=Path('/kaggle/temp/rn3-cost-v1-environment');environment_root.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(bundle/'rn3_cost_v1.py'),'worker','--bundle',str(bundle),'--protocol-sha256',args.protocol_sha256,'--assets',args.assets,'--competition',args.competition,'--opencv-wheel',args.opencv_wheel,'--work',str(work),'--environment-root',str(environment_root)]
    result={'schema':SCHEMA,'status':'RUNNING','protocol_sha256':args.protocol_sha256,'submission_ready':False}
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        r=resource.supervise(command,work/'resource',remaining,str(bundle));validate_resource(r)
        need(r['argv']==command and r['timeout_seconds']==remaining,'resource command/deadline differs')
        wr=read(work/'worker-receipt.json');need(wr['status']=='COMPLETE' and wr['numerical_status']=='PASS' and wr['cleanup']['clean'] and wr['protocol_sha256']==args.protocol_sha256 and wr['members']==NAMES and wr['studies']==100 and wr['forward_calls']==300,'worker completion contract failed')
        check_budget(time.monotonic()-start,r['sampled_tree_rss_max_bytes'],size_tree(work))
        result.update(status='PASS_RN3_ENGINEERING_COST_ONLY',whole_cold_seconds=time.monotonic()-start,
            peak_tree_RSS_bytes=r['sampled_tree_rss_max_bytes'],files={n:sha(work/n) for n in ['worker-receipt.json','numeric.json','inference.json','raw.npz','resource/receipt.json']},
            accuracy_gain=False,hidden_runtime_PASS=False)
    except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
    finally:
        try:
            t=time.monotonic();result['offline_environment_bytes']=size_tree(environment_root)
            shutil.rmtree(environment_root)
            result['offline_environment_cleanup_seconds']=time.monotonic()-t
            result['offline_environment_cleanup_complete']=not environment_root.exists()
        except BaseException as exc:result.update(status='FAIL',environment_cleanup_error=repr(exc))
        result['finished_cold_seconds']=time.monotonic()-start;result['whole_cold_seconds']=time.monotonic()-start;result['output_bytes']=size_tree(work)
        if result['status']!='PASS_RN3_ENGINEERING_COST_ONLY':write(work/'result.json',result)
    need(result['status']=='PASS_RN3_ENGINEERING_COST_ONLY','owned scratch cleanup failed')
    try:publish_result(work,result,start,result['peak_tree_RSS_bytes'])
    except BaseException as exc:
        write(work/'result.json',{**result,'status':'FAIL','error':'finalization: '+repr(exc)});raise
    return result


def prepare():
    stage=ROOT/STAGE;stage.mkdir(parents=True,exist_ok=False);bundle=stage/'bundle';bundle.mkdir()
    assets=ROOT/'.local/b-rn3f-staging/20261004-v1/assets';a=read(assets/'b-rn3f-assets.json')
    ids=read(assets/'engineering100.json')['ids'];rbase=ROOT/'.local/b-rn3f-runs/20261004-v1-t4-evidence/work/producer/result'
    ref=read(rbase/'receipt.json');need(ref['status']=='COMPLETE' and ref['ordered_uids_sha256']=='824251f64754854339b3972f5fe3c30fce8b347e2b4712f34a3fdd3eaae46981','completed originalreference required')
    files={}
    for dest,source in {'rn3_cost_v1.py':Path(__file__),'process_resource_receipt.py':ROOT/'tools/process_resource_receipt.py','b_rn3f_parity.py':ROOT/'tools/b_rn3f_parity.py','reference-receipt.json':rbase/'receipt.json','reference-members.npz':rbase/'members.npz'}.items():
        data=source.read_bytes();files[dest]=hashlib.sha256(data).hexdigest();(bundle/dest).write_bytes(data)
    need(files['reference-members.npz']==ref['output_sha256']['members.npz'],'opaque savedreference bytes changed')
    need(files['process_resource_receipt.py']==RESOURCE_SHA and files['b_rn3f_parity.py']==PARITY_SHA,'reused passive helper changed')
    p=dict(schema=SCHEMA,status='PREPARED_NO_EXECUTION',budget=BUDGET,ids=ids,ordered_uids_sha256=ref['ordered_uids_sha256'],members=[a['members'][i] for i in INDICES],all_members=a['members'],targets=a['policy']['input']['targets'],input=a['policy']['input'],files=files,
        reference_scope='original100 T4 sixmember raw engineering outputs; RN indices1,3,5 only; opaque-copy during prepare, no numpy load',source_files=a['files'],
        root_cpu_decision_sha256=sha(ROOT/EVIDENCE/'root-cpu-decision.json'),test_sha256=sha(ROOT/'tests/test_rn3_cost_v1.py'),probability_limits={'max_abs':.002,'p99_abs':.002,'mean_abs':.0005},single_cold_attempt=True)
    validate_subset(p['members'],a['members']);write(bundle/'protocol.json',p);digest=sha(bundle/'protocol.json')
    payload={q.name:base64.b64encode(q.read_bytes()).decode() for q in bundle.iterdir() if q.is_file()}
    code='''from pathlib import Path
import base64,hashlib,importlib.util,json,os,sys,time
started=time.monotonic()
base=Path('/kaggle/working/rn3-cost-v1');bundle=base/'bundle';bundle.mkdir(parents=True,exist_ok=False)
'''+f'payload={payload!r}\n'+'''for name,value in payload.items():
    assert Path(name).name==name
    (bundle/name).write_bytes(base64.b64decode(value))
helper=bundle/'rn3_cost_v1.py'
'''+f'assert hashlib.sha256(helper.read_bytes()).hexdigest()=={files["rn3_cost_v1.py"]!r}\n'+'''spec=importlib.util.spec_from_file_location('rn3_cost',helper);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def find_file(roots,pattern,digest):
    found={p.resolve() for root in roots if root.is_dir() for p in root.glob(pattern) if p.is_file() and m.sha(p)==digest}
    assert len(found)==1
    return next(iter(found))
assets=find_file([Path('/kaggle/input/knee-b-rn3f-v1-assets'),Path('/kaggle/input/datasets/taeyangg4/knee-b-rn3f-v1-assets')],'b-rn3f-assets.json',m.ASSET_SHA).parent
wheel=find_file([Path('/kaggle/input/opencv-python-headless-4120088-x86'),Path('/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86')],'opencv_python_headless-4.12.0.88-*.whl',m.OPENCV_SHA)
roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
competition=[p for p in roots if (p/'train_series.csv').is_file() and (p/'train_series').is_dir()]
assert len(competition)==1
from types import SimpleNamespace
'''+f"args=SimpleNamespace(bundle=str(bundle),protocol_sha256={digest!r},assets=str(assets),competition=str(competition[0]),opencv_wheel=str(wheel),work=str(base/'work'),started_monotonic=started)\n"+'''result=m.run(args)
print(json.dumps(result,indent=2))
assert result['status']=='PASS_RN3_ENGINEERING_COST_ONLY'
'''
    need(len(code.encode())<1_000_000,'notebook payload exceeds1MB')
    nb={'nbformat':4,'nbformat_minor':4,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    directory=stage/'notebook';directory.mkdir();write(directory/'rn3-cost.ipynb',nb)
    meta=read(ROOT/'.local/b-rn3f-t4-staging/20261004-v1/notebook/kernel-metadata.json');meta.update(id='taeyangg4/knee-rn3-cold-cost-v1',title='knee-rn3-cold-cost-v1',code_file='rn3-cost.ipynb');write(directory/'kernel-metadata.json',meta)
    plan=dict(status='ROOT_REVIEW_AND_CURRENT_QUOTA_REQUIRED_NOT_EXECUTED',stage=str(stage),protocol_sha256=digest,notebook_sha256=sha(directory/'rn3-cost.ipynb'),metadata_sha256=sha(directory/'kernel-metadata.json'),source_sha256=files['rn3_cost_v1.py'],budget=BUDGET,
        push_argv=[str(ROOT/'.venv/Scripts/kaggle.exe'),'kernels','push','-p',str(directory),'--timeout','1200'],
        no_dataset_upload=True,existing_private_dataset='taeyangg4/knee-b-rn3f-v1-assets',expected_GPU_hours_max_planning=2*1200/3600,
        arrays_interpreted_during_prepare=0,opaque_reference_bytes=(bundle/'reference-members.npz').stat().st_size,
        result_path='rn3-cost-v1/work/result.json',first_result_scope='cost/numerical only; notsubmission/accuracy/hiddenfullPASS')
    write(stage/'execution-plan.json',plan);print(json.dumps(plan,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True);sub.add_parser('prepare')
    for name in ['run','worker']:
        a=sub.add_parser(name)
        for k in ['bundle','protocol-sha256','assets','competition','opencv-wheel','work']:a.add_argument('--'+k,required=True)
        if name=='run':a.add_argument('--started-monotonic',type=float,required=True)
        else:a.add_argument('--environment-root',required=True)
    a=ap.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='worker':worker(a)
    else:run(a)

if __name__=='__main__':main()
