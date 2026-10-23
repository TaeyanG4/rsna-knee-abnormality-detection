"""C85-R7 v1.5.0 shared visible/N32 diagnostic entry point. B launches on 2xT4.

Both variants consume one inference run. Qualification evidence remains HOLD
until independently audited; this program never authorizes a submission.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp169_release_c85r7_pl15_v150 as m
from tools import c85r7_v150_safety as safety


def asset(path):
    return dict(path=str(Path(path).resolve()), sha256=m.pl.sha(path))


def load_runtime(stage, output):
    runtime = Path(output)/'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    pins = m.read(Path(stage)/'flip-v130-pins.json')
    with zipfile.ZipFile(Path(stage)/'flip-v130-runtime.zip') as archive:
        for name, expected in pins.items():
            data = archive.read(name)
            m.need(m.hashlib.sha256(data).hexdigest()==expected, 'frozen flip archive pin')
            m.need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'archive path')
            safety.atomic_write(runtime/name, data, immutable=True)
    name = m.flip_build.MODULE
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location(name, runtime/(name+'.py'))
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    # Only invocation infrastructure changes; exact v1.3.0 generated source remains intact.
    module.MODULE = 'c85r7_v150_base_child'
    shim = ('from pathlib import Path\nimport sys\n'
            f'sys.path.insert(0,{str(ROOT)!r})\n'
            'from tools.c85r7_v150_runner import base_child\nbase_child()\n')
    safety.atomic_write(runtime/(module.MODULE+'.py'), shim.encode(), immutable=True)
    bootstrap = Path(output)/'bootstrap'; bootstrap.mkdir(exist_ok=True)
    startup = ('import sys\n'+f'sys.path.insert(0,{str(ROOT)!r})\n'+
        'from tools.c85r7_v150_safety import install_io, metadata_io\ninstall_io()\n'
        'from tools import release_metadata_v122 as _meta\n'
        '_io=metadata_io(_meta)\n_io.__enter__()\n'
        'sys.modules["release_metadata_v122"]=_meta\n')
    safety.atomic_write(bootstrap/'sitecustomize.py', startup.encode(), immutable=True)
    return module, runtime, bootstrap


def base_child():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['child'])
    p.add_argument('--runtime', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--branch'); p.add_argument('--input-root',default='/kaggle/input')
    a=p.parse_args()
    sys.path.insert(0,str(a.runtime))
    spec=importlib.util.spec_from_file_location(m.flip_build.MODULE,a.runtime/(m.flip_build.MODULE+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    with safety.metadata_io(module.metadata_io):
        safety.progress(a.branch+':start')
        module.child_entry(a.runtime,a.output,a.branch,input_root=a.input_root)


def native_child(job_path):
    from tools import vnext_r7_pl_worker_v1 as worker
    job=m.read(job_path)
    with safety.metadata_io(m.pl.metadata):
        safety.progress('PL:model-load')
        worker.native(job)


def plan(stage, mode, binding=None, review=None, health=None, image_sha256=None):
    m.validate(stage)
    m.need(mode in ('visible','n32'), 'diagnostic mode')
    reasons=[]
    if binding is None: reasons.append('genuine PL bindings absent')
    else:
        b=m.read(binding); m.pl.validate_bindings(m.pl.registration(Path(stage)/'student'/b['arm']),b)
    if review is None: reasons.append('independent review absent')
    else: m.review_gate(stage,review)
    if health is None:reasons.append('measured full-shape memory and healthy journal rates absent')
    else:m.health_gate(stage,binding,image_sha256,health)
    return dict(status='HOLD' if reasons else 'READY_FOR_DIAGNOSTIC', reasons=reasons,
        mode=mode,variants=['a','b'],shared_inference=True,n32_runs=1 if mode=='n32' else 0,
        compute_cap_seconds=27000,publication_reserve_seconds=1200,
        remaining='target evidence is audited after execution; no automatic PASS qualification',
        qualified_for_submission=False)


def run(stage, mode, binding, review, output, image_sha256, health=None):
    m.need(plan(stage,mode,binding,review,health,image_sha256)['status']=='READY_FOR_DIAGNOSTIC', 'diagnostic admission HOLD')
    measured=m.health_gate(stage,binding,image_sha256,health)
    m.need(len(image_sha256)==64 and all(c in '0123456789abcdef' for c in image_sha256), 'image SHA256 required')
    # This runner deliberately rejects local consumer-GPU execution.
    import subprocess
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True,timeout=10).splitlines()
    m.need(len(names)==2 and all('T4' in n for n in names), 'qualification requires exactly 2xT4')
    output=Path(output).resolve();m.need(not output.exists(), 'fresh diagnostic output required')
    output.mkdir(parents=True)
    os.environ.setdefault(m.base.insurance.START_ENV, repr(time.monotonic()))
    os.environ.setdefault(m.base.insurance.BUDGET_ENV,'27000')
    clock=m.base.insurance.Deadline.from_env();m.need(clock.seconds<=27000,'shared compute cap')
    b=m.read(binding); runtime=dict(b['runtime'])
    reg=m.pl.registration(Path(stage)/'student'/b['arm']);m.pl.runtime_preflight(reg,runtime)
    module, rt, bootstrap=load_runtime(stage,output)
    os.environ['C85_COMPETITION']=runtime['competition']
    os.environ[module.STRESS_ENV]='1';os.environ[module.insurance.FAULT_ENV]='off'
    from tools import exp169_c85r7ins_checks_v122 as checks
    from tools.release_cleanup_checks_v121 import CleanupAudit
    original_module=checks.m
    checks.m=module
    clean=output/'base'
    fault_summary=None
    started=time.monotonic()
    try:
        with safety.metadata_io(module.metadata_io), safety.watchdog(module.insurance,output/'watchdog',bootstrap,measured['healthy_commits_per_second']):
            if mode=='n32':
                os.environ[module.insurance.FAULT_ENV]='all'
                # ONE shared N32 base clean + real own/R decode/kill/hang/deadline suite.
                fault_summary=checks.target_stress(rt,output)
                clean=output/'insurance-stress/clean'
                runtime['competition']=str(output/'insurance-stress/competition')
                os.environ[module.insurance.FAULT_ENV]='off'
            else:
                with CleanupAudit() as audit:
                    module.run_release(rt,clean,stress=True)
                    audit.verify(time.monotonic())
            ids,raw,failures=checks.snapshot(clean/module.RUN/'a172',clean/module.RUN/'pr7')
            m.need(len(ids)==32 if mode=='n32' else len(ids)>0, 'diagnostic cohort size')
            m.need(not any(failures.values()), 'healthy base fallback')
            receipt=m.read(clean/module.RECEIPT)
            m.need(receipt.get('flip_policy')==m.flip.policy('FLIP_F0F2'), 'base flip receipt')
            receipt['snapshot_sha256']=m.snapshot_digest(ids,raw,failures)
            baseline=output/'authenticated-baseline.json';safety.json_write(baseline,receipt)
            with (Path(runtime['competition'])/'test_series.csv').open(newline='',encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            run_id=m.digest([m.digest(m.registration(stage)),m.pl.sha(binding),str(output)])
            shards=[];exits=[];pl_start=time.monotonic()
            codec=module.old.codec_receipt(clean/module.RUN)
            with CleanupAudit() as audit:
                with module.insurance.Supervisor(clock) as supervisor, m.pl.context_logs() as logs:
                    children=[]
                    for i in range(2):
                        subset=ids[i::2]
                        if not subset:continue
                        work=output/'PL'/str(i)
                        job=m.native_job(stage,binding,subset,rows,runtime,work,run_id)
                        job['codec']=codec
                        path=work/'job.json';safety.json_write(path,job)
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                        env=module.codecs.child_env(env,codec,branch='PL',telemetry_dir=output/'codec-telemetry')
                        log=logs.open(work/'child.log')
                        child=supervisor.start([sys.executable,'-B',str(Path(__file__)), 'native-child','--job',str(path)],
                            label='PL-'+str(i),env=env,stdout=log,stderr=log)
                        children.append(child);shards.extend(job['branches'])
                    for child in children:
                        supervisor.wait(child)
                        exits.append(dict(code=child.returncode,reason=child.reason,cleanup_error=child.cleanup_error))
                cleanup=audit.verify(time.monotonic())
            pl_seconds=time.monotonic()-pl_start
            m.need(all(e['code']==0 and e['cleanup_error'] is None for e in exits),'PL child failure; retained partial attempts')
            # Both weights consume precisely the same snapshots, jobs and journals.
            for shard in shards:
                _,bad,count,reason=m.student_predictions(shard['path'],shard['ids'],shard['identity'])
                m.need(not bad and count==0 and reason is None,'healthy PL fallback')
            outputs={v:m.compose(stage,binding,v,clean/module.RUN/'a172',clean/module.RUN/'pr7',shards,run_id,
                output/('variant-'+v),base_receipt=asset(baseline)) for v in m.VARIANTS}
            result=dict(status='AUDIT_REQUIRED',mode=mode,variants=['a','b'],studies=len(ids),
                registration_sha256=m.digest(m.registration(stage)),binding_sha256=m.pl.sha(binding),
                image_sha256=image_sha256,health=health,flip_policy=m.flip.policy('FLIP_F0F2'),outputs=outputs,
                cleanup=cleanup,exits=exits,elapsed_seconds=time.monotonic()-started,pl_seconds=pl_seconds,
                observed_pl_seconds_per_study=pl_seconds/len(ids),base_fault_suite=fault_summary,
                remaining=['independent actual-inference native parity audit',
                    'PL native-worker/queue fault N32 and metadata-interruption audit',
                    'conservative full-cohort runtime upper bound including setup/cleanup'],
                qualified_for_submission=False,exposure='EXPOSED_EXPLORATORY',clean_oof=False)
            safety.json_write(output/'execution.json',result)
            return result
    except BaseException as exc:
        try:safety.json_write(output/'failure.json',dict(status='HOLD',error=type(exc).__name__,partial_attempts=str(output)))
        except OSError:pass
        raise
    finally:
        checks.m=original_module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['visible','n32','native-child'])
    p.add_argument('--stage',type=Path,default=m.OUT);p.add_argument('--bindings',type=Path)
    p.add_argument('--review',type=Path);p.add_argument('--output',type=Path);p.add_argument('--job',type=Path)
    p.add_argument('--health',type=Path)
    p.add_argument('--image-sha256');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.mode=='native-child':native_child(a.job);return
    review=None if a.review is None else asset(a.review)
    health=None if a.health is None else asset(a.health)
    if a.dry_run:result=plan(a.stage,a.mode,a.bindings,review,health,a.image_sha256)
    else:
        m.need(a.output is not None and a.image_sha256 is not None,'output and image SHA256 required')
        result=run(a.stage,a.mode,a.bindings,review,a.output,a.image_sha256,health)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
