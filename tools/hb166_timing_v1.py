"""HB166 N32 timing harness 1.0.0; CPU preparation, B-only future GPU use.

Local sequential old-Q/HB166 ABBA diagnostic. Never a two-T4 schedule claim.
Reuses the HB164 cohort, telemetry, memory/idle/WDDM/rate guards and owned cleanup.
No gate binding, upload, wall deadline or uncalibrated T4 projection.
Draft correction: resolve project series metadata before the frozen code namespace.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import hb166_release_v1 as b, hb166_serving_v1 as s
from tools import c85r7_r224_timing as timing
from tools import hb145_output_io as io

VERSION='1.0.0'
TAG='HB166_N32'
WORK=b.WORK/'timing'
ORDER=['oldQ','HB166','HB166','oldQ']
PYTHON=timing.GPU_PYTHON


def write(name,value):
    s.need(value.get('identity',TAG)==TAG,'journal identity drift')
    io.journal(WORK/name,dict(value,identity=TAG),WORK)


def binding():
    validated=b.validate()
    return dict(version=VERSION,tag=TAG,kernel_sha256=validated['kernel_sha256'],
        manifest_sha256=s.sha(b.OUT/'assets/manifest.json'),
        tool_sha256=s.sha(__file__),registration_sha256=s.sha(b.WORK/'registration.json'),
        dependencies={p:s.sha(ROOT/p) for p in ['tools/c85r7_r224_timing.py',
            'tools/c85r7_flipall_timing.py','tools/hb149_monitor_v125.py','tools/hb145_output_io.py',
            'tools/gpu_job.py','tools/job_runtime.py']},order=ORDER,
        exposure='EXPOSED',clean_oof=False,cohort=timing.shared.cohort()[0],
        differences=['Both arms sequential on one 4070, not the two-T4 parent Q schedule',
            'Same parent Q preprocessing/inference byte spans and all 20 models',
            'No notebook cells; model loading, preprocessing, TTA, transfer and journals included; module imports excluded',
            'Full release overlap/startup and T4 performance remain unmeasured'],
        targets=s.TARGETS,arm_identities=arm_identities(),
        oldq_blocks_sha256=[hashlib.sha256(x.encode()).hexdigest() for x in oldq_blocks()],
        deterministic_algorithms=True,torch_threads=1,
        runtime_limit_seconds=32400,projection='HOLD: end-to-end T4 evidence required')


def arm_identities():
    parent=b.previous.snapshots()['parent']
    return dict(oldQ=dict(receipt_sha256=parent['receipt_sha256'],kernel_files=parent['kernel_files']),
                HB166=dict(manifest_sha256=s.sha(b.OUT/'assets/manifest.json')))


def register():
    value=binding(); old=WORK/'registration.json'
    if old.exists(): s.need(io.read_json(old)==dict(identity=TAG,**value),'timing registration drift; reseal review required')
    write('registration.json',value)
    return dict(status='REGISTERED_CPU',tag=TAG,cohort=32,order=ORDER)


def registered():
    value=io.read_json(WORK/'registration.json')
    s.need(value==dict(identity=TAG,**binding()),'timing binding drift')
    return value


def prepare(dry_run):
    reg=registered(); inv=timing.inventory()
    s.need(inv['studies']==reg['cohort'],'N32 cohort differs')
    probe=timing.cpu_decode_probe(inv)
    result=dict(binding_sha256=timing.digest(reg),inventory_sha256=timing.digest(inv),
        studies=inv['studies'],series=inv['series'],probe=probe,files={},dry_run=dry_run)
    if not dry_run:
        data=WORK/'data'
        with zipfile.ZipFile(timing.shared.ZIP) as archive:
            for entry in inv['members']:
                rel=Path(*Path(entry['target']).parts[1:])
                path=data/'test_series'/rel
                s.need(path.resolve().is_relative_to(data.resolve()),'unsafe data path')
                path.parent.mkdir(parents=True,exist_ok=True)
                raw=archive.read(entry['member'])
                if not path.exists() or path.read_bytes()!=raw: io.retry(path.write_bytes,raw)
                result['files'][str(path.relative_to(data))]=s.sha(path)
        for name,rows in [('test.csv',[{'StudyInstanceUID':u} for u in inv['studies']]),('test_series.csv',inv['series'])]:
            with (data/name).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            result['files'][name]=s.sha(data/name)
    write('prepare-dry-run.json' if dry_run else 'prepare.json',result)
    return dict(status='PASS_CPU_PREPARE',decoded_studies=probe['decoded_studies'],dry_run=dry_run,
                gpu_executed=False,full_roster_ready=False)


def admit(dry_run=True):
    registered()
    holds=list(b.registered()['holds'])
    holds+=['Authenticated old Q asset directory and timing launch review',
            'Local sequential method is not T4 process-schedule parity']
    return dict(admission='HOLD',holds=holds,gpu_executed=False)


def gpu_admission():
    # No GPU call occurs before this complete-roster and independent-review check.
    reg=registered()
    s.roster(b.OUT/'assets',reg['manifest_sha256'])
    review=io.read_json(WORK/'launch-review.json')
    s.need(review.get('verdict')=='PASS' and review.get('binding_sha256')==timing.digest(reg)
           and review.get('builder')!=review.get('reviewer') and review.get('reviewer'),
           'independent exact-binding launch review required')
    prep=io.read_json(WORK/'prepare.json')
    s.need(prep['binding_sha256']==timing.digest(reg) and not prep['dry_run'],'CPU prepare required')
    for name,digest in prep['files'].items(): s.need(s.sha(WORK/'data'/name)==digest,'DICOM drift')
    return reg,prep,review


def validate_values(values,ids,expected):
    import numpy as np
    a=np.asarray(values)
    s.need(list(ids)==list(expected) and len(ids)==len(set(ids)),'cohort order drift')
    s.need(a.shape==(len(ids),12) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),
           'finite Nx12 probabilities required')


def validate_measurement(value,ids,identities=None):
    s.need(len(ids)==32 and value['tag']==TAG and value['arm'] in ORDER and value['done']==32
           and value['state']=='COMPLETE' and value['clean_oof'] is False,'incomplete N32')
    validate_values(value['values'],value['ids'],ids)
    s.need(value.get('targets')==s.TARGETS,'target order drift')
    expected=identities if identities is not None else arm_identities()
    s.need(value.get('arm_identity')==expected[value['arm']],'arm identity drift')
    s.need(value['fallback']==0 and math.isfinite(value['seconds']) and value['seconds']>0,'fallback or time invalid')
    s.need(0<value['peak_reserved_bytes']<=12*2**30,'peak memory invalid')
    timing.shared.memory_admission(value['desktop'],[value['peak_reserved_bytes']])
    s.need(value['telemetry']['complete'] is True,'telemetry incomplete')


def summarize(records,ids,identities=None):
    import numpy as np
    for value in records: validate_measurement(value,ids,identities)
    s.need([v['arm'] for v in records]==ORDER,'ABBA order differs')
    s.need(len({v['runtime_sha256'] for v in records})==1,'GPU/software stack drift')
    old=np.mean([records[i]['seconds'] for i in (0,3)])
    new=np.mean([records[i]['seconds'] for i in (1,2)])
    return dict(tag=TAG,old_seconds=float(old),new_seconds=float(new),
        local_inclusive_delta_seconds=float(new-old),fallback=0,
        runtime='HOLD',qualified_for_submission=False,exposure='EXPOSED',clean_oof=False)


def projection(delta):
    s.need(math.isfinite(delta),'nonfinite delta')
    return dict(runtime='HOLD',qualified_for_submission=False,
        reason='Inclusive local N32 totals cannot be scaled to hidden. Require independently reviewed, authenticated end-to-end T4 measurements at relevant scale and the real HB167 hidden receipt.')


def replay_real_output():
    """Exercise common validator on preserved REAL CPU output, no N32 claim."""
    import numpy as np
    receipt=s.read(b.WORK/'cpu-replay.json')
    path=b.WORK/'cpu-predictions.npz'
    s.need(receipt['status']=='PASS' and s.sha(path)==receipt['predictions_sha256'],'real CPU replay receipt drift')
    with np.load(path,allow_pickle=False) as value:
        validate_values(value['served'],value['ids'].tolist(),b.registered()['cohort'])
        s.need(value['targets'].tolist()==s.TARGETS,'target order drift')
    result=dict(status='PASS_REAL_CPU_OUTPUT_VALIDATOR',studies=2,sha256=s.sha(path),
                n32_postrun_replay='PENDING_REAL_N32_OUTPUT',gpu_executed=False)
    write('real-output-replay.json',result); return result


def oldq_blocks():
    """Lift math from the admitted parent verbatim; exclude its two-T4 scheduling."""
    raw=b.previous.snapshots()['parent']['files']['exp169_partner_bundle_release.py'].decode()
    a=raw.index('                volume, mask = corpus.volume(uid)')
    z=raw.index('                prepared = time.monotonic()',a)
    prep=textwrap.dedent(raw[a:z])
    a=raw.index('                indices = np.asarray(flags, bool)')
    z=raw.index('                ended = time.monotonic()',a)
    infer=textwrap.dedent(raw[a:z])
    return prep,infer


def authenticate_runtime(directory,expected):
    for name,raw in expected.items():
        s.need((directory/name).read_bytes()==raw,'extracted runtime drift: '+name)


def oldq_preprocess(corpus,uid,namespace):
    """CPU-smokable exact parent preprocessing block; no model load."""
    scope=dict(namespace,corpus=corpus,uid=uid,i=0)
    exec(compile(oldq_blocks()[0],'<parent-Q-preprocessing>','exec'),scope)
    return scope['payload']


def cpu_oldq_replay():
    """Real pixels through the exact lifted old-Q preprocessing, no weights/GPU."""
    import importlib.util
    import numpy as np
    from training.src.raptor_corpus import series_records
    rows=series_records(); ids=b.registered()['cohort']
    partner=ROOT/'artifacts/releases/EXP169_A172_v2_0_2_stage/assets'
    s.need(s.sha(partner/'release-manifest.json')=='39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc','oldQ manifest drift')
    manifest=s.read(partner/'release-manifest.json')
    for name,entry in manifest['files'].items():
        if name.startswith('code/'):
            digest=entry['sha256'] if isinstance(entry,dict) else entry
            s.need(s.sha(partner/name)==digest,'oldQ code drift')
    sys.path.insert(0,str(partner/'code'))
    setup=oldq_setup()
    raw=b.previous.snapshots()['parent']['files']['exp169_partner_bundle_release.py']
    namespace={'__name__':'hb166_oldq_cpu','__file__':str(b.WORK/'parent-runtime/exp169_partner_bundle_release.py')}
    exec(compile(raw,'<authenticated-parent-Q>','exec'),namespace)
    namespace.update(setup)
    corpus=setup['DicomCorpus'].from_records(ids,[r for uid in ids for r in rows[('train',uid)]],
        setup['DirectoryDicomSource'](b.WORK/'real-data/test_series'))
    result=[]
    for uid in ids:
        payload=oldq_preprocess(corpus,uid,namespace)
        arrays=payload[1:4]
        s.need(all(x.shape==(36,3,224,224) and np.isfinite(x).all() for x in arrays),'oldQ preprocess shape/nonfinite')
        result.append(dict(shapes=[list(x.shape) for x in arrays],sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in arrays]))
    receipt=dict(status='PASS_REAL_OLDQ_PREPROCESS',studies=len(result),arrays=result,gpu_executed=False)
    write('cpu-oldq-replay.json',receipt)
    return dict(status=receipt['status'],studies=len(result),gpu_executed=False)


def oldq_setup():
    import numpy as np
    from raptor_dicom import DicomCorpus,DirectoryDicomSource
    from raptor_common import select_centers,windows_from
    from raptor_radimagenet import center3_windows_from,normalise
    from raptor_runtime import normalize_input
    return dict(np=np,DicomCorpus=DicomCorpus,DirectoryDicomSource=DirectoryDicomSource,
        select_centers=select_centers,windows_from=windows_from,center3_windows_from=center3_windows_from,
        normalise=normalise,normalize_input=normalize_input,
        blocks=tuple(compile(x,'<parent-Q-block>','exec') for x in oldq_blocks()))


def oldq(c85,partner,data,ids,output,commit,setup):
    np=setup['np']
    module=c85.a172; module.reset_backend()
    manifest=c85.manifest(partner/'release-manifest.json'); module.verify_assets(partner,manifest)
    entries=manifest['entries']; own=module._load_models(partner,entries,'cuda:0')
    commit(0,'MODELS_LOADED')
    corpus=setup['DicomCorpus'](data/'test.csv',data/'test_series.csv',setup['DirectoryDicomSource'](data/'test_series'))
    scores={name:np.full((10 if name=='l2' else 5,len(ids),12),np.nan,np.float32) for name in ('l2','dino','convnext')}
    journal=c85.Journal(output/'q-journal',ids,{'manifest_sha256':c85.MANIFEST_SHA})
    prepare,infer=setup['blocks']
    scope=dict(module.base.__dict__); scope.update(setup)
    scope.update(corpus=corpus,entries=entries,own=own,device='cuda:0',scores=scores)
    for i,uid in enumerate(ids):
        scope.update(i=i,uid=uid)
        exec(prepare,scope)
        commit(i,'PREPROCESSED')
        _,scope['rad'],scope['img'],scope['triplet'],scope['flags']=scope['payload']
        exec(infer,scope)
        for m,e in enumerate(entries):
            row=e['fold']+(5 if e['family']=='l2' and e['seed']==43 else 0)
            journal.record(uid,str(m),scores[e['family']][row,i])
        commit(i+1)
    return module.aggregate(scores)[0]


def launch_command():
    return [str(PYTHON),'-B',str(Path(__file__).resolve()),'run']


def lane():
    import psutil
    from tools.job_runtime import LOGS,job_status
    parents=psutil.Process().parents()
    job=job_status(LOGS,TAG,expected_command=launch_command(),tail=False)
    s.need(job['state']=='running' and job['pid'] in {p.pid for p in parents},'exclusive gpu_job HB166_N32 required')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=5)
    s.need(not timing.shared.gpu_python_pids(apps.stdout,{os.getpid(),*[p.pid for p in parents]}),'another Python ML job')
    desktop=timing.shared.desktop_snapshot(); timing.shared.live_memory_guard(desktop['used_mib'],desktop)
    return desktop


def worker(index):
    s.need(os.environ.get('HB166_CONTROLLER_PID')==str(os.getppid()),'supervised worker only')
    reg,prep,review=gpu_admission(); lane()
    # Each arm gets a fresh process; runtime bytes checked by binding before import.
    runtime=b.WORK/('parent-runtime' if ORDER[index]=='oldQ' else 'candidate-runtime')
    expected=b.previous.snapshots()['parent']['files'] if ORDER[index]=='oldQ' else b.payload()[1]
    authenticate_runtime(runtime,expected)
    partner=Path(review['partner_dir'])
    # Worker-only package namespace, so project tools cannot shadow sealed code.
    import types
    package=types.ModuleType('tools'); package.__path__=[str(runtime)]
    for name in list(sys.modules):
        if name=='tools' or name.startswith('tools.'): del sys.modules[name]
    sys.modules['tools']=package
    sys.path[:0]=[str(runtime),str(partner/'code'),str(partner/'code/training/src')]
    import exp169_release_c85 as c85
    manifest=c85.manifest(partner/'release-manifest.json')
    c85.a172.verify_assets(partner,manifest)
    import torch
    s.need(torch.cuda.device_count()==1 and '4070 Ti SUPER' in torch.cuda.get_device_name(0),'registered 4070 only')
    c85.a172.reset_backend()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if ORDER[index]=='oldQ': setup=oldq_setup()
    else:
        import hb166_serving_v1 as adapter
        prepared_recipe=adapter.recipe(runtime/'hb166-sources.json')
        adapter.recipe=lambda _:prepared_recipe
    runtime_info=timing.runtime_snapshot(dict(torch=torch,np=__import__('numpy'),timm=__import__('timm')))
    runtime_info.update(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),torch_threads=torch.get_num_threads())
    output=WORK/f'arm-{index}'
    s.need(not output.exists(),'preserve prior real arm output; review/reset before rerun')
    output.mkdir(); c85.write_once(output/'ids.json',c85.json_bytes(prep['studies']))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    sequence=0
    def commit(done,phase='STUDY_COMPLETE'):
        nonlocal sequence
        sequence+=1
        io.receipt(WORK/f'worker-{index}.json',dict(state=phase,done=done,sequence=sequence,seconds=time.perf_counter()-start))
    commit(0,'LOAD')
    if ORDER[index]=='oldQ': values=oldq(c85,partner,WORK/'data',prep['studies'],output,commit,setup)
    else:
        # Instrument completed journals only; all math and original journal writes unchanged.
        original=c85.Journal
        completed=[0]
        class Journal(original):
            def record(self,*args,**kwargs):
                value=super().record(*args,**kwargs)
                completed[0]+=1; commit(completed[0])
                return value
        c85.Journal=Journal
        class Progress:
            def phase_start(self,phase): commit(completed[0],phase)
            def completed(self,phase,counts): commit(completed[0],phase)
        adapter.run(c85,WORK/'data',output,asset_dir=b.OUT/'assets',device='cuda',progress=Progress())
        values,failures=adapter.authenticated(c85,output,prep['studies'])
        s.need(not failures,'HB166 fallback detected')
    torch.cuda.synchronize()
    result=dict(tag=TAG,arm=ORDER[index],done=32,state='COMPLETE',ids=prep['studies'],values=values.tolist(),
        targets=list(c85.p.LABELS),arm_identity=reg['arm_identities'][ORDER[index]],binding_sha256=timing.digest(reg),
        seconds=time.perf_counter()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        fallback=0,clean_oof=False,exposure='EXPOSED',runtime_sha256=timing.digest(runtime_info))
    write(f'worker-{index}.json',result)


def supervise(index,desktop):
    import psutil
    monitor=timing.monitor
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             HB166_CONTROLLER_PID=str(os.getpid()))
    child=job=None; tracked={}; telemetry=timing._TimingTelemetry(env,.1,desktop)
    start=last=time.perf_counter(); completed=0; sequence=-1; rate=None; error=None; cleanup=[]
    try:
        child=subprocess.Popen([str(PYTHON),'-B',str(Path(__file__).resolve()),'_worker','--index',str(index)],
            cwd=ROOT,env=env,start_new_session=os.name!='nt')
        process=psutil.Process(child.pid); tracked[process.pid]=process
        job=monitor._WindowsJob(child); telemetry.start()
        while child.poll() is None:
            now=time.perf_counter(); metrics=telemetry.snapshot()
            s.need(metrics['memory_failure'] is None,metrics['memory_failure'])
            try: state=io.read_json(WORK/f'worker-{index}.json')
            except (OSError,ValueError): state={}
            done=state.get('done',0)
            if state.get('sequence',-1)>sequence:
                sequence=state['sequence']; last=now
            if done>completed:
                completed=done; last=now
                if rate is None and completed>=2:
                    rate=monitor.RateGuard(completed*60/(now-start),now)
                    rate.history.clear(); rate.history.append((now,completed))
            timing.watchdog(now,last,metrics['latest_gpu'],desktop)
            if rate is not None:
                reason=rate.update(now,completed); s.need(reason is None,reason)
            for p in [process,*process.children(recursive=True)]: tracked.setdefault(p.pid,p)
            time.sleep(.1)
        s.need(child.returncode==0,'timing worker failed')
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc); raise
    finally:
        cleanup=monitor._cleanup(child,tracked,job,psutil); telemetry.close()
        metrics=telemetry.snapshot(final=True)
        io.receipt(WORK/f'supervision-{index}.json',dict(error=error,cleanup=cleanup,telemetry=metrics))
    s.need(not cleanup and metrics['memory_failure'] is None,'cleanup or memory failure')
    value=io.read_json(WORK/f'worker-{index}.json'); value.update(desktop=desktop,telemetry=metrics)
    s.need(value['binding_sha256']==timing.digest(registered()),'result binding drift')
    validate_measurement(value,registered()['cohort']); write(f'measurement-{index}.json',value)
    return value


def run():
    reg,_,_=gpu_admission(); desktop=lane()
    records=[supervise(i,desktop) for i in range(4)]
    result=summarize(records,reg['cohort']); write('summary.json',result)
    return result


def postcheck():
    reg=registered(); records=[io.read_json(WORK/f'measurement-{i}.json') for i in range(4)]
    for record in records: s.need(record['binding_sha256']==timing.digest(reg),'result binding drift')
    result=summarize(records,reg['cohort']); write('postcheck.json',result); return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','prepare','validate','admit','replay-real-output','cpu-oldq-replay','run','_worker','postcheck'])
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--index',type=int,choices=range(4)); a=p.parse_args()
    if a.command not in ('run','_worker'): b.cpu()
    if a.command=='prepare': result=prepare(a.dry_run)
    elif a.command=='admit': s.need(a.dry_run,'only admission dry-run supported'); result=admit()
    elif a.command=='validate': registered(); result=dict(status='PASS_CPU_BINDING',admission='HOLD')
    elif a.command=='_worker': worker(a.index); return
    elif a.command=='replay-real-output': result=replay_real_output()
    elif a.command=='cpu-oldq-replay': result=cpu_oldq_replay()
    else: result=globals()[a.command]()
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
