"""One bounded engineering trial: original anchor133, exit, v3 R2, public3.

Stage is CPU-only. No remote upload or competition-candidate builder exists.
The immutable helper's train100 gate and historical failures are not rewritten.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[1]
V3_SHA = "3800aec2662bf8f0d8a0219cd0adb8d50ad97f1943c63c45cadfcdf55c434ba9"
ANCHOR_SHA = "cd5b8b69f89bcfaa6b6ee065f7b18f7134dc71968be3827c87ca3e5672007cac"
UID = "StudyInstanceUID"
GIB = 1 << 30


def resolve_competition(expected_sha, candidates=None):
    """Resolve one complete flat/nested mount, then bind exact train metadata."""
    candidates = candidates or [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                                Path('/kaggle/input/rsna-knee-abnormality-detection')]
    matches = {}
    for candidate in map(Path, candidates):
        if (all((candidate/name).is_file() for name in ('train_series.csv','test.csv','test_series.csv','sample_submission.csv'))
                and all((candidate/name).is_dir() for name in ('train_series','test_series'))):
            matches[str(candidate.resolve())] = candidate
    need(len(matches)==1, f'expected one complete flat/nested competition mount; found {len(matches)}')
    found=next(iter(matches.values()))
    need(sha(found/'train_series.csv')==expected_sha,'competition metadata changed')
    return found


def install_proxy_probe_mask(proxy, competition, flat_root=None):
    """Select explicit engineering proxy without mutating immutable anchor cells.

    Original find_root probes env proxy plus flat/test.csv. Only that unused
    flat existence probe is masked in the anchor child; its actual data stays
    readable, and R2/public3 are separate unmodified processes.
    """
    proxy,competition=Path(proxy),Path(competition)
    flat=Path('/kaggle/input/rsna-knee-abnormality-detection') if flat_root is None else Path(flat_root)
    original=Path.exists
    if competition.resolve()!=flat.resolve():return lambda:None
    need(proxy.resolve()!=flat.resolve() and (proxy/'test.csv').is_file(), 'explicit distinct complete proxy required')
    hidden=flat/'test.csv'
    def exists(path,*args,**kwargs):
        return False if path==hidden else original(path,*args,**kwargs)
    Path.exists=exists
    def restore():
        if Path.exists is exists:Path.exists=original
    return restore


def need(ok, why):
    if not ok: raise ValueError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value; spec.loader.exec_module(value); return value


def extension_ids(prior, eligible, excluded):
    need(len(prior) == 100 and len(set(prior)) == 100, "frozen100 prefix required")
    allowed = set(eligible) - set(excluded)
    need(set(prior) <= allowed, "prior cohort eligibility changed")
    extra = sorted(allowed - set(prior), key=lambda u: hashlib.sha256(("d01-runtime-20261003:" + u).encode()).hexdigest())[:33]
    need(len(extra) == 33, "exactly33 extension studies required")
    return list(prior) + extra


def projection(core_seconds):
    need(isinstance(core_seconds, (int, float)) and math.isfinite(core_seconds) and core_seconds > 0, "finite measured core time")
    sample = core_seconds * 1300 / 133
    estimate = max(sample, 30938.468180919); remaining = 32400 - estimate
    return {"measured133_seconds": core_seconds, "sample1300_seconds": sample,
            "prior_separate100_floor_seconds": 30938.468180919, "projected1300_seconds": estimate,
            "minimum_reserve_seconds": 300, "remaining_seconds": remaining,
            "minimum_reserve_pass": remaining >= 300, "thirty_minute_target_pass": remaining >= 1800,
            "thirty_percent_runtime_overhead_pass": estimate * 1.3 <= 32400,
            "scope": "whole133 production core scaled1300/133, includes startup/cleanup/blend; public3 postcheck excluded; engineering estimate, not hidden-runtime guarantee"}


def verify_r2(result, raw_ids, raw, prior, prior_ids, prior_raw, protocol, probability):
    import numpy as np
    ids = protocol["ids"]
    need(result.get("status") == "COMPLETE" and result.get("ids") == ids and result.get("studies") == 133, "R2133 completion/identity")
    need(result.get("release_helper_sha256") == V3_SHA and result.get("policy_sha256") == protocol["helper_policy_sha256"], "frozen v3 helper/policy")
    need(result.get("member_count") == 5 and result.get("all_finite") is True and result.get("fp32_fallbacks") == 0
         and result.get("neutral_fills") == 0 and result.get("mean_before_rank") is True, "R2 inference contract")
    need(raw_ids == ids and prior_ids == ids[:100] and raw.shape == (5,133,12) and raw.dtype == np.float32
         and np.isfinite(raw).all() and np.all((raw >= 0) & (raw <= 1)), "R2 exact output coverage/shape")
    need(result["member_identities"] == [{"name":m["name"],"sha256":m["checkpoint_sha256"]} for m in prior["inference"]["member_results"]], "five reference model identities")
    current = {r["uid"]:r for r in result["study_receipts"]}; old = {r["uid"]:r for r in prior["study_receipts"]}
    tensors = {r["uid"]:r for r in result["input_tensor_hashes"]}; before = {r["uid"]:r for r in prior["inference"]["input_tensor_hashes"]}
    need(len(current) == len(result["study_receipts"]) == len(tensors) == 133 and set(current) == set(tensors) == set(ids), "receipt coverage")
    checks = [{"uid":u,"full_receipt_exact":current[u] == old[u],"normalized_tensor_exact":tensors[u] == before[u]} for u in ids[:100]]
    extra = [{"uid":u,"volume_exact":current[u]["volume_sha256"] == r["volume_sha256"],
              "mask_exact":current[u]["mask_sha256"] == r["mask_sha256"]} for u,r in zip(ids[100:],protocol["extension_input_reference"],strict=True)]
    need([r["uid"] for r in protocol["extension_input_reference"]] == ids[100:], "extension reference order")
    probability_result = probability(raw[:,:100], prior_raw)
    need(all(all(v for k,v in r.items() if k != "uid") for r in checks + extra) and probability_result["pass"], "input/probability regression gate")
    return {"prefix100":checks,"extension33":extra,"probability100":probability_result,
            "extension_probability_scope":"no separate GPU reference; finite complete five-model production path only"}


def anchor_notebook(original, protocol):
    """Add a label-free proxy prefix and audit suffix; original30 cells untouched."""
    ids = protocol["ids"]
    prefix = f'''from pathlib import Path as _TrialPath
import os as _trial_os,json as _trial_json,hashlib as _trial_hash,time as _trial_time
import pandas as _trial_pd
_TRIAL_IDS = {ids!r}
_TRIAL_OUT = _TrialPath(_trial_os.environ['D01_TRIAL_PHASE'])
_TRIAL_START = _trial_time.monotonic()
_trial_comp = _TrialPath(_trial_os.environ['D01_TRIAL_COMP'])
assert _trial_hash.sha256((_trial_comp/'train_series.csv').read_bytes()).hexdigest()=={protocol['train_series_sha256']!r}
_trial_meta = _trial_pd.read_csv(_trial_comp/'train_series.csv',dtype={{'StudyInstanceUID':str,'SeriesInstanceUID':str}})
assert {{'StudyInstanceUID','SeriesInstanceUID','Anatomical_Plane','Fluid_Sensitive','Fat_Suppression'}} <= set(_trial_meta.columns)
assert 'Report' not in _trial_meta.columns
_trial_rows = _trial_meta[_trial_meta.StudyInstanceUID.isin(_TRIAL_IDS)].copy()
assert set(_trial_rows.StudyInstanceUID)==set(_TRIAL_IDS)
_trial_proxy = _TRIAL_OUT/'proxy'; _trial_proxy.mkdir(exist_ok=False); (_trial_proxy/'test_series').mkdir()
_trial_pd.DataFrame({{'StudyInstanceUID':_trial_meta.StudyInstanceUID.drop_duplicates().tolist()}}).to_csv(_trial_proxy/'train.csv',index=False)
_trial_pd.DataFrame({{'StudyInstanceUID':_TRIAL_IDS}}).to_csv(_trial_proxy/'test.csv',index=False)
_trial_rows.to_csv(_trial_proxy/'test_series.csv',index=False)
_trial_cols=list(_trial_pd.read_csv(_trial_comp/'sample_submission.csv',nrows=0).columns)
_trial_sample=_trial_pd.DataFrame(.5,index=range(133),columns=_trial_cols);_trial_sample['StudyInstanceUID']=_TRIAL_IDS
_trial_sample.to_csv(_trial_proxy/'sample_submission.csv',index=False)
for _u,_s in _trial_rows[['StudyInstanceUID','SeriesInstanceUID']].drop_duplicates().itertuples(index=False,name=None):
    assert all('/' not in v and '..' not in v for v in (_u,_s))
    _src=_trial_comp/'train_series'/_u/_s; assert _src.is_dir() and _src.resolve().is_relative_to((_trial_comp/'train_series').resolve())
    _dst=_trial_proxy/'test_series'/_u/_s;_dst.mkdir(parents=True,exist_ok=False)
    for _f in sorted(_src.glob('*.dcm')):
        assert _f.resolve().is_relative_to((_trial_comp/'train_series').resolve())
        (_dst/_f.name).symlink_to(_f)
_trial_os.environ['RSNA_COMP_ROOT']=str(_trial_proxy)
import importlib.util as _trial_import
_trial_spec=_trial_import.spec_from_file_location('d01_trial_path_adapter',_trial_os.environ['D01_TRIAL_WRAPPER'])
_trial_adapter=_trial_import.module_from_spec(_trial_spec);_trial_spec.loader.exec_module(_trial_adapter)
_trial_restore_probe=_trial_adapter.install_proxy_probe_mask(_trial_proxy,_trial_comp)
_trial_os.environ.pop('RSNA_PARALLEL_COAT_READERS',None)
_trial_os.environ.pop('RSNA_CACHE_RAM_GIB',None)
_trial_os.environ.pop('RSNA_PIXEL_SCRATCH',None)
'''
    suffix = f'''_trial_audit=dict(_RSNA_AUDIT)
assert _trial_audit.get('status')=='COMPLETE' and _trial_audit.get('cohort')==133
assert _trial_audit.get('coat_readers_parallel_pairs') is False and _trial_audit.get('coat_family_members')==4
assert (_trial_audit.get('dino_members'),_trial_audit.get('a5_folds'),_trial_audit.get('raptor_views'))==(20,5,4)
_trial_memmap=[r for r in _trial_audit['events'] if r.get('kind')=='scratch_allocation' and r.get('tag')=='test g1' and r.get('bytes',0)>{GIB}]
assert len(_trial_memmap)==1 and _trial_memmap[0]['bytes']=={protocol['expected133_dino_bytes']},'DINO133 must use unchanged native memmap branch'
assert not any(_TrialPath(p).exists() for p in _RSNA_CACHE_FILES),'anchor scratch files not released'
_trial_target=_TRIAL_OUT/'engineering-anchor.csv';assert not _trial_target.exists()
_trial_os.replace('/kaggle/working/submission.csv',_trial_target)
_trial_audit.update(schema='d01-integrated-anchor133',original_notebook_sha256={ANCHOR_SHA!r},
    original_cells=30,memmap_events=_trial_memmap,anchor_process_seconds=_trial_time.monotonic()-_TRIAL_START,
    cache_paths=list(_RSNA_CACHE_FILES),scope='engineering train133 proxy; no clinical metric or competition candidate',competition_root=str(_trial_comp),flat_probe_mask_scope='only anchor process Path.exists(flat/test.csv), physical file unchanged')
(_TRIAL_OUT/'anchor-audit.json').write_text(_trial_json.dumps(_trial_audit,indent=2,default=str))
_trial_restore_probe()
'''
    cell = lambda source: {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":source.splitlines(keepends=True)}
    ast.parse(prefix); ast.parse(suffix)
    nb = copy.deepcopy(original); nb["cells"] = [cell(prefix),*original["cells"],cell(suffix)]
    need(len(original["cells"]) == 30 and nb["cells"][1:-1] == original["cells"], "original30 cells modified")
    return nb


def cgroup():
    root = Path('/sys/fs/cgroup'); answer = {}
    # Kaggle cgroup v2 mount normally exposes this container at the mount root.
    for name in ('memory.current','memory.max','memory.peak','memory.events'):
        path = root/name
        if path.is_file():
            raw=path.read_text().strip();answer[name] = dict((k,int(v)) for k,v in (l.split() for l in raw.splitlines())) if name=='memory.events' else int(raw) if raw.isdigit() else raw
    if not answer:
        root=Path('/sys/fs/cgroup/memory')
        for old,new in [('memory.usage_in_bytes','memory.current'),('memory.limit_in_bytes','memory.max'),('memory.max_usage_in_bytes','memory.peak'),('memory.failcnt','memory.failcnt')]:
            p=root/old
            if p.is_file():answer[new]=int(p.read_text().strip())
    return answer


def snapshot():
    import psutil
    proc=psutil.Process();children=[];rss=proc.memory_info().rss
    for p in proc.children(recursive=True):
        try:rss+=p.memory_info().rss;children.append({'pid':p.pid,'created':p.create_time()})
        except psutil.Error:pass
    query=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
    gpu=[]
    for row in csv.reader(query.stdout.splitlines()):gpu.append({'index':int(row[0]),'name':row[1].strip(),'used_mib':float(row[2]),'total_mib':float(row[3])})
    disks={}
    for name in ('/kaggle/working','/kaggle/temp','/tmp'):
        path=Path(name)
        if path.exists():d=shutil.disk_usage(path);disks[name]={'free':d.free,'total':d.total}
    return {'observed_at':time.time(),'cgroup':cgroup(),'rss_tree':rss,'children':children,'gpu':gpu,'disks':disks}


def resource_gate(before, after):
    a,b=before['cgroup'],after['cgroup']
    need('memory.current' in b and 'memory.max' in b, 'cgroup memory telemetry unavailable')
    need(isinstance(b['memory.max'],int) and b['memory.max'] < 1<<60, 'finite container memory limit required')
    need(b['memory.current'] < b['memory.max'], 'container memory exhausted')
    for k in ('oom','oom_kill'):
        need(b.get('memory.events',{}).get(k,0)==a.get('memory.events',{}).get(k,0), 'new cgroup '+k)
    if 'memory.failcnt' in b:need(b['memory.failcnt']==a.get('memory.failcnt',0),'new cgroup allocation failure')
    need(len(after['gpu'])==2 and all('T4' in g['name'] and g['used_mib']<g['total_mib'] for g in after['gpu']), 'two T4 memory/device gate')
    need(after['disks']['/kaggle/working']['free'] > GIB, 'working disk headroom exhausted')


def reap_owned(proc, known, supervisor):
    import psutil
    supervisor.stop_tree(proc)
    # nbconvert/kernel and R2 grandchildren can own separate sessions. Only
    # terminate PID/create-time pairs observed under our launched child.
    for pid,created in known.items():
        try:
            p=psutil.Process(pid)
            if p.create_time()==created:p.kill()
        except psutil.Error:pass


def run_phase(name, command, env, work, cap, deadline, supervisor):
    import psutil
    work=Path(work);work.mkdir(parents=True,exist_ok=False)
    before=snapshot();resource_gate(before,before);known={};samples=[];start=time.monotonic()
    report={'phase':name,'status':'RUNNING','before':before,'timeout_seconds':min(cap,deadline-start)}
    need(report['timeout_seconds']>0,'shared trial deadline expired')
    out=(work/'stdout.log').open('wb');err=(work/'stderr.log').open('wb');proc=None
    try:
        proc=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=os.name=='posix')
        report['pid']=proc.pid
        with (work/'telemetry.jsonl').open('w',encoding='utf-8') as stream:
            while True:
                try:
                    children=psutil.Process(proc.pid).children(recursive=True)
                    for p in children:
                        try:known[p.pid]=p.create_time()
                        except psutil.Error:pass
                except psutil.Error:pass
                sample=snapshot();resource_gate(before,sample)
                samples.append(sample);stream.write(json.dumps(sample)+'\n');stream.flush()
                code=proc.poll()
                if code is not None:
                    need(code==0,f'{name} child exit {code}');break
                need(time.monotonic()-start < report['timeout_seconds'],f'{name} deadline exceeded')
                time.sleep(2)
        # Successful children must release their kernels and inference workers.
        alive=[]
        for pid,created in known.items():
            try:
                p=psutil.Process(pid)
                if p.create_time()==created and p.status()!=psutil.STATUS_ZOMBIE:alive.append(pid)
            except psutil.Error:pass
        need(not alive,f'{name} descendants survived normal exit: {alive}')
        after=snapshot();resource_gate(before,after)
        base={g['index']:g['used_mib'] for g in before['gpu']}
        report.update(status='COMPLETE',after=after,exit_code=code,all_owned_processes_reaped=True,
            gpu_used_after_minus_before_mib={str(g['index']):g['used_mib']-base[g['index']] for g in after['gpu']},
            memory_cleanup_scope='all observed owned processes exited; remaining OS cache/driver allocations are measured, not classified as a leak',
            rss_peak=max(s['rss_tree'] for s in samples),cgroup_peak=max(s['cgroup']['memory.current'] for s in samples),
            gpu_peak_mib={str(i):max(next(g['used_mib'] for g in s['gpu'] if g['index']==i) for s in samples) for i in (0,1)})
        return report
    except BaseException as exc:
        report.update(status='FAILED',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        if proc is not None:reap_owned(proc,known,supervisor)
        out.close();err.close();report['whole_phase_seconds']=time.monotonic()-start
        dump(work/'phase-receipt.json',report)


def execute(base, started):
    base=Path(base);protocol=read(base/'trial-protocol.json');work=base/'results';work.mkdir(exist_ok=False)
    receipt={'schema':'d01-integrated133-trial-v2-path','status':'RUNNING','protocol_sha256':sha(base/'trial-protocol.json'),
        'wrapper_sha256':sha(__file__),'helper_sha256':V3_SHA,'phases':{},'failures':[]}
    deadline=float(started)+protocol['internal_deadline_seconds']
    try:
        need(os.name=='posix' and base.resolve().is_relative_to(Path('/kaggle/working')), 'explicit Kaggle trial path required')
        manifest=read(base/'trial-manifest.json')
        for name,digest in manifest['files'].items():need(sha(base/name)==digest,'changed trial file: '+name)
        need(sha(base/'helper/d01_release_v3.py')==V3_SHA and sha(base/'anchor-original.ipynb')==ANCHOR_SHA,'frozen source identities')
        helper=module(base/'helper/d01_release_v3.py','d01_trial_frozen_helper');helper.validate_bundle(base/'helper')
        supervisor=module(base/'helper/d01_r2_shard_benchmark.py','d01_trial_supervisor')
        need(supervisor.effective_cpu()['effective_cpus']>=4,'four effective CPUs required')
        assets_roots=[Path('/kaggle/input/knee-d01-r2-assets'),Path('/kaggle/input/datasets/taeyangg4/knee-d01-r2-assets')]
        assets=[p.parent for root in assets_roots if root.is_dir() for p in root.rglob('d01-assets-manifest.json') if sha(p)==helper.ASSETS_SHA]
        need(len(assets)==1,'one exact R2 assets mount');assets=assets[0]
        codec=next(p for p in [Path('/kaggle/input/knee-d01-codec-assets'),Path('/kaggle/input/datasets/taeyangg4/knee-d01-codec-assets')] if p.is_dir())
        competition=resolve_competition(protocol['train_series_sha256'])
        receipt['competition_root']=str(competition)
        need(sha(competition/'train_series.csv')==protocol['train_series_sha256'],'competition metadata changed')
        initial=snapshot();resource_gate(initial,initial);receipt['initial_resources']=initial
        scratch_roots=[p for p in (Path('/kaggle/temp'),Path('/tmp')) if p.is_dir()]
        need(any(shutil.disk_usage(p).free>=protocol['full1300_dino_bytes']+GIB for p in scratch_roots),'full1300 largest-cache disk capacity absent')
        scratch_before={str(p) for root in scratch_roots for p in (root/'rsna_pixels_v559').glob('pixels_*.npy')}
        anchor_dir=work/'anchor';anchor_dir.mkdir();nb=anchor_notebook(read(base/'anchor-original.ipynb'),protocol)
        dump(anchor_dir/'worker.ipynb',nb)
        env=dict(os.environ,D01_TRIAL_PHASE=str(anchor_dir),D01_TRIAL_COMP=str(competition),D01_TRIAL_WRAPPER=str(base/'d01_integrated_trial_v2.py'))
        for key in ('RSNA_PARALLEL_COAT_READERS','RSNA_CACHE_RAM_GIB','RSNA_PIXEL_SCRATCH','CUDA_VISIBLE_DEVICES'):env.pop(key,None)
        command=[sys.executable,'-m','jupyter','nbconvert','--to','notebook','--execute',str(anchor_dir/'worker.ipynb'),
            '--output','executed.ipynb','--output-dir',str(anchor_dir),'--ExecutePreprocessor.timeout=-1']
        receipt['phases']['anchor']=run_phase('anchor',command,env,work/'anchor-process',protocol['anchor_cap_seconds'],deadline,supervisor)
        anchor=read(anchor_dir/'anchor-audit.json');need(anchor['original_notebook_sha256']==ANCHOR_SHA,'anchor receipt identity')
        actual_scratch=Path(anchor['memmap_events'][0]['directory'])
        need(shutil.disk_usage(actual_scratch).free>=protocol['full1300_dino_bytes']+GIB,'actual memmap filesystem lacks full1300 cache capacity')
        scratch_after={str(p) for root in scratch_roots for p in (root/'rsna_pixels_v559').glob('pixels_*.npy')}
        need(not scratch_after-scratch_before,'new anchor scratch files remain after process exit')
        receipt['scratch_cleanup']={'no_new_pixel_files':True,'actual_directory':str(actual_scratch),'free_after':shutil.disk_usage(actual_scratch).free,'full1300_capacity_only':True}
        need(not Path('/kaggle/working/submission.csv').exists(),'engineering anchor output was not moved')
        receipt['anchor_audit_sha256']=sha(anchor_dir/'anchor-audit.json')
        dump(work/'trial-progress.json',{'stage':'anchor complete; starting R2133','elapsed_seconds':time.monotonic()-float(started)})
        common=[sys.executable,str(base/'d01_integrated_trial_v2.py')]
        args=['--base',str(base),'--assets',str(assets),'--codec',str(codec),'--competition',str(competition)]
        command=[*common,'r2',*args,'--work',str(work/'r2'),'--deadline',str(deadline)]
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        env.pop('CUDA_VISIBLE_DEVICES',None)
        receipt['phases']['r2']=run_phase('r2',command,env,work/'r2-process',protocol['r2_cap_seconds'],deadline,supervisor)
        import numpy as np
        import pandas as pd
        current=read(work/'r2/receipt.json');raw=np.load(work/'r2/members.npz',allow_pickle=False)
        old=np.load(base/'helper/reference-members.npz',allow_pickle=False);prior=read(base/'helper/reference-receipt.json')
        receipt['r2_checks']=verify_r2(current,raw['ids'].tolist(),raw['member_probabilities'],prior,old['study_uids'].tolist(),old['member_probabilities'],protocol,supervisor.probability_gate)
        ids=protocol['ids'];anchor_frame=pd.read_csv(anchor_dir/'engineering-anchor.csv',dtype={UID:str});r2_frame=pd.read_csv(work/'r2/mean.csv',dtype={UID:str})
        frame=helper.blend(assets,anchor_frame,r2_frame,ids)
        need(frame[UID].tolist()==ids and frame.shape==(133,13) and np.isfinite(frame.iloc[:,1:].to_numpy()).all(),'whole133 blend/schema')
        labels=list(frame.columns[1:]);expected=.9*anchor_frame.set_index(UID).loc[ids,labels].rank(method='average',pct=True)+.1*r2_frame.set_index(UID).loc[ids,labels].rank(method='average',pct=True)
        need(np.array_equal(frame[labels].to_numpy(),expected.to_numpy()),'original90/10 whole133 ranking changed')
        frame.to_csv(work/'engineering-blend133.csv',index=False)
        receipt['engineering_blend_sha256']=sha(work/'engineering-blend133.csv')
        receipt['core_receipts']={name:sha(work/path) for name,path in [('anchor_phase','anchor-process/phase-receipt.json'),('r2_phase','r2-process/phase-receipt.json'),('r2','r2/receipt.json')]}
        dump(work/'production-core-receipt.json',receipt)
        receipt['production_core_seconds']=time.monotonic()-float(started)
        receipt['production_core_receipt_sha256']=sha(work/'production-core-receipt.json')
        receipt['projection']=projection(receipt['production_core_seconds'])
        dump(work/'trial-progress.json',{'stage':'133 production path complete; runtime gate then public3','projection':receipt['projection']})
        need(receipt['projection']['minimum_reserve_pass'],'133 measured runtime projection lacks unchanged5min reserve; stop without retry')
        command=[*common,'public',*args,'--work',str(work/'public3')]
        receipt['phases']['public3']=run_phase('public3',command,env,work/'public3-process',protocol['public3_cap_seconds'],deadline,supervisor)
        public=read(work/'public3/t4-gate.json');need(public['status']=='PASS' and public['mode']=='public3' and public['release_helper_sha256']==V3_SHA,'exact v3 public3 gate failed')
        receipt['public3_gate_sha256']=sha(work/'public3/t4-gate.json')
        need(time.monotonic()<deadline,'shared deadline exhausted before verdict')
        receipt.update(status='PASS',readiness='READY_FOR_EXPLORATORY_SUBMISSION',scope='source/inputs/numerical/production order/sample resources and projected runtime only; no accuracy gain or hidden-full PASS; separate user approval required for any submission',
            limits=protocol['limits'],submission_or_candidate_created=False)
    except BaseException as exc:
        receipt['status']='NO_GO';receipt['failures'].append({'error':str(exc),'traceback':traceback.format_exc()});raise
    finally:
        receipt['total_notebook_seconds']=time.monotonic()-float(started);dump(work/'integrated-trial-result.json',receipt)
    return receipt


def worker(action, base, assets, codec, competition, work, deadline=None):
    base=Path(base);helper=module(base/'helper/d01_release_v3.py','d01_integrated_frozen_v3')
    need(sha(base/'helper/d01_release_v3.py')==V3_SHA,'worker helper identity')
    if action=='r2':
        return helper.run_r2(base/'helper',assets,codec,competition,work,read(base/'trial-protocol.json')['ids'],'train',float(deadline),False)
    return helper.preflight(base/'helper',assets,codec,competition,'public3',work,time.monotonic())


def stage(out):
    """Reuse all frozen evidence bytes; replace only path wrapper and manifest."""
    import io
    import lzma
    import tarfile
    out=Path(out);need(not out.exists(),'new immutable stage path required')
    previous=ROOT/'.local/d01-integrated-trial/20261003-v2'
    original=ROOT/'tools/d01_integrated_trial.py'
    need(sha(original)=='599e7d08d208586ad2b306b25a9a4bcb6c946592377ec94ca619ac7915ba4135','original wrapper changed')
    old_manifest=read(previous/'payload/trial-manifest.json')
    raw={}
    for name,digest in old_manifest['files'].items():
        value=(previous/'payload'/name).read_bytes();need(hashlib.sha256(value).hexdigest()==digest,'old payload changed: '+name)
        if name!='d01_integrated_trial.py':raw[name]=value
    raw['d01_integrated_trial_v2.py']=Path(__file__).read_bytes()
    protocol=json.loads(raw['trial-protocol.json'])
    need(hashlib.sha256(raw['helper/d01_release_v3.py']).hexdigest()==V3_SHA,'v3 helper changed')
    need(hashlib.sha256(raw['anchor-original.ipynb']).hexdigest()==ANCHOR_SHA,'anchor changed')
    anchor_notebook(json.loads(raw['anchor-original.ipynb']),protocol)
    manifest={'schema':'d01-integrated133-manifest-v2-path','files':{k:hashlib.sha256(v).hexdigest() for k,v in raw.items()}}
    raw['trial-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    need(len(raw)==17,'exact17 payload contract')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w',format=tarfile.USTAR_FORMAT) as archive:
        for name,value in sorted(raw.items()):
            entry=tarfile.TarInfo(name);entry.size=len(value);entry.mtime=0;entry.mode=0o644
            archive.addfile(entry,io.BytesIO(value))
    compressed=lzma.compress(buffer.getvalue(),preset=9)
    packed=base64.b85encode(compressed).decode('ascii')
    expected={k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in raw.items()}
    # Independent roundtrip before exposing the stage.
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(base64.b85decode(packed))),mode='r:') as archive:
        need(set(archive.getnames())==set(raw),'archive file set differs')
        for entry in archive.getmembers():need(entry.isfile() and archive.extractfile(entry).read()==raw[entry.name],'archive byte roundtrip failed')
    code=f'''from pathlib import Path
import ast,base64,hashlib,importlib.util,io,json,lzma,sys,tarfile,time
started=time.monotonic()
base=Path('/kaggle/working/d01-integrated133-v2');base.mkdir(exist_ok=False)
expected={expected!r}
packed={packed!r}
compressed=base64.b85decode(packed)
assert hashlib.sha256(compressed).hexdigest()=={hashlib.sha256(compressed).hexdigest()!r}
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(compressed)),mode='r:') as archive:
    entries=archive.getmembers()
    assert len(entries)==len(expected)==17 and {{e.name for e in entries}}==set(expected)
    for entry in entries:
        assert entry.isfile() and not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts
        value=archive.extractfile(entry).read();item=expected[entry.name]
        assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
        if entry.name.endswith('.py'):ast.parse(value.decode('utf-8'))
        target=base/entry.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
spec=importlib.util.spec_from_file_location('d01_integrated_trial_v2',base/'d01_integrated_trial_v2.py')
trial=importlib.util.module_from_spec(spec);sys.modules['d01_integrated_trial_v2']=trial;spec.loader.exec_module(trial)
result=trial.execute(base,started)
print(json.dumps({{k:result[k] for k in ('status','production_core_seconds','total_notebook_seconds','projection','scope')}},indent=2))
'''
    ast.parse(code)
    for name,value in raw.items():
        if name.endswith('.py'):ast.parse(value.decode('utf-8'))
    nb={'nbformat':4,'nbformat_minor':4,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['# D01 integrated133 path version2\nPrior0.4546s mount failure preserved. CPU-staged path adapter; no automatic remote retry.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    notebook_bytes=(json.dumps(nb,ensure_ascii=False,indent=2)+'\n').encode()
    need(len(notebook_bytes)<=1_000_000,'notebook exceeds1MB transport limit')
    meta=read(previous/'kernel-metadata.json')
    out.mkdir(parents=True)
    for name,value in raw.items():p=out/'payload'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
    (out/meta['code_file']).write_bytes(notebook_bytes);dump(out/'kernel-metadata.json',meta)
    review={'status':'CPU_STAGED_PATH_FIX_ONLY_NO_REMOTE','wrapper_sha256':sha(__file__),'prior_wrapper_sha256':sha(original),
        'helper_sha256':V3_SHA,'anchor_sha256':ANCHOR_SHA,'protocol_sha256':sha(out/'payload/trial-protocol.json'),
        'notebook_sha256':sha(out/meta['code_file']),'notebook_bytes':len(notebook_bytes),'transport':'deterministic17-file tar+LZMA9+base85, each uncompressed byte/size/SHA verified',
        'compressed_bytes':len(compressed),'unchanged_payload_files':sorted(set(raw)-{'d01_integrated_trial_v2.py','trial-manifest.json'}),
        'unchanged':'helper/anchor/weights/gates/133IDs/protocol/timeformula/5400cap all unchanged',
        'path_change':'one complete flat/nested mount+exacttrainSHA; anchor-child-only unusedflat/test.csv Path.exists mask avoids original _one_direct ambiguity without input-file changes; preserve all official series metadata including required Fat_Suppression, matching successful anchor benchmark proxy',
        'automatic_retry':False,'session_cap_seconds':5400,'new_kernel_version_expected':'root must read actual push result; no remote push performed',
        'output_root':'/kaggle/working/d01-integrated133-v2'}
    dump(out/'stage-review.json',review);return review


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('stage');p.add_argument('--out',required=True)
    for action in ('r2','public'):
        p=sub.add_parser(action)
        for name in ('base','assets','codec','competition','work'):p.add_argument('--'+name,required=True)
        if action=='r2':p.add_argument('--deadline',required=True,type=float)
    args=vars(parser.parse_args());action=args.pop('action')
    if action=='stage':print(json.dumps(stage(**args),indent=2))
    else:worker(action,**args)

