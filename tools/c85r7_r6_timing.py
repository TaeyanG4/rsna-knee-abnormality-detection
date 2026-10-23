"""H-B-174 N32 ABBA v1.0.1, local diagnostics only; B owns GPU execution.

Changelog v1.0.1: rehash actual N32 data/path contracts at execution and replay;
authenticate prepared instrumentation against independent review evidence.
Backups: .local/backups/hb174/*-v1.0.0.py. No serving kernel changes.
v1.0.0: R7-full/R6-full paired cost; reviewed existing supervisor, no serving
changes. Historical real-output identity control and mandatory post-run replay.
No automatic gate binding, submission, admission or GPU launch from CPU actions.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import c85r7_r224r6fq_v1130 as b
from tools import c85r7_r224flip_diagnostics as d
from tools import hb145_output_io as io
need,sha,digest=d.need,d.sha,d.digest
WORK=b.WORK/'timing'
SCRIPT=Path(__file__).resolve()
TAG='HB174_N32'
VERSION='1.0.1'
HELPER_PATHS=[SCRIPT,Path(b.__file__),Path(d.__file__),Path(d.r224.__file__),Path(d.r7.__file__),
              Path(d.measure.__file__),Path(io.__file__),ROOT/'tools/c85r7_r6_cpu_parity.py',
              ROOT/'tests/test_c85r7_r224r6fq_v1130.py']
HELPER_PATHS += [ROOT/'tools'/name for name in (
    'hb149_monitor_v125.py','gpu_job.py','gpu_queue.py','job_runtime.py',
    'exp189_s6_port.py','exp169_r7_port.py','hb128_gold58_pbranch_v1.py',
    'c85r7_stackq_v1125.py','c85r7_flipall_v1115.py')]
ORDER=('r7-1','r6-1','r6-2','r7-2')
HISTORY=ROOT/'.local/scratch/hb167-harness/hb167/timing/v1.0.3'
HISTORY_REG='cae1ca31e27a9f3d96392059fc63cbc9efa00ab516565949487d2b6cccdf26d6'
HISTORY_RESULT='8e78678d7f5bcdd3c5d63ea80d5a37eae4566abf024216003ab8483791c084ab'
RULES=dict(order=list(ORDER),keep_score_min=.956,parent_hours=6.7,parent_upper_hours=6.73,
           limit_hours=9,exposure='EXPOSED',clean_oof=False,
           projection='signed mean paired supervisor-wall delta * authenticated R7 hidden/local-plain scale; '
           'gate: 6.73h + max(0, both paired wall and GPU-second deltas) * same scale; no speedup credit',
           limitation='4070-to-Kaggle incremental cost transfer assumed; calibration review PENDING; '
           'parent runtime is user evidence, not a rigorous bound. Independent result review required.')


def save(name,**value):
    result=dict(identity=TAG,version=VERSION,exposure='EXPOSED',clean_oof=False,**value)
    io.journal(WORK/(name+'.json'),result,WORK)
    return result


def data_config():
    # Process-local path configuration only; frozen shared modules are not edited.
    d.WORK=b.WORK
    d.mode_work=lambda mode: WORK
    d.prepare_data('timing')
    return d.data_source('timing')


def data_identity_check():
    """Read-only check of the exact files the worker will open; never repairs drift."""
    data=WORK/'data'
    ids,rows=d.r7.cohort()
    need((data/'train_series.csv').read_bytes()==d.series_bytes(rows),'owned N32 metadata drift')
    from tools import exp189_s6_port as s6
    resolved_ids,paths=s6._study_ids([data/'train_series'/u for u in ids])
    need(resolved_ids==ids and all(p.parent.parent==data.resolve() for p in paths),
         'owned N32 study root drift')
    for row in rows:
        rel=Path('train_series')/row['StudyInstanceUID']/row['SeriesInstanceUID']
        need((data/rel).resolve(strict=True)==(d.R224_DATA/'data'/rel).resolve(strict=True),
             'owned N32 series junction drift')
    receipt=io.read_json(d.R224_DATA/'prepare.json')
    need(receipt.get('state')=='CPU_PREPARED' and receipt['studies']==ids and
         receipt.get('real_study_cpu_replay',{}).get('exact_window_parity') is True,
         'complete N32 preparation required')
    expected={Path(name).as_posix():value for name,value in receipt['files'].items()}
    actual={p.relative_to(data).as_posix() for row in rows
            for p in (data/'train_series'/row['StudyInstanceUID']/row['SeriesInstanceUID']).glob('*.dcm')}
    need(actual==set(expected),'owned N32 DICOM inventory drift')
    for name,expected_sha in expected.items():
        need(sha(data/name)==expected_sha,'owned N32 DICOM hash drift')
    return dict(files=len(actual),inventory_sha256=digest(expected))


def files(arm):
    return b.payload()[2] if arm.startswith('r6') else b.parent_builder.parent()[1]


def bundle(arm):
    return b.BUNDLE if arm.startswith('r6') else d.r7.MANIFEST.parent


def binding():
    validated=b.validate()
    ids=d.r7.cohort()[0]
    return dict(kernel_sha256=validated['kernel_sha256'],ids=ids,rules=RULES,
        stage_registration_sha256=sha(b.OUT/'registration.json'),
        helpers={str(p.relative_to(ROOT)):sha(p) for p in HELPER_PATHS},
        manifests=dict(r6=sha(b.BUNDLE/'manifest.json'),r7=sha(d.r7.MANIFEST)),
        series_sha256=sha(d.r7.SERIES),data_receipt_sha256=sha(d.R224_DATA/'prepare.json'),
        anatomy_sha256=sha(d.ANATOMY),code=code_binding(),
        calibration=d.r224.transfer_calibration(),history_registration=HISTORY_REG,history_result=HISTORY_RESULT)


def code_binding():
    from tools import exp189_s6_port as s6
    return s6.verify_code(d.CODE,d.ANATOMY)


def register():
    value=dict(identity=TAG,version=VERSION,state='REGISTERED',binding=binding())
    value['binding_sha256']=digest(value['binding'])
    b.old.write(WORK/'registration.json',b.old.encoded(value))
    from tools.gpu_queue import validate_plan
    io.atomic_json(WORK/'gpu-plan.json',validate_plan([dict(tag=TAG,command=command())]))
    return dict(state='REGISTERED',binding_sha256=value['binding_sha256'])


def current():
    value=io.read_json(WORK/'registration.json')
    need(value['binding']==binding() and value['binding_sha256']==digest(value['binding']),'registration drift')
    return value


def prepare():
    reg=current()
    data=data_config()
    inventory=d.verify_data('timing')
    data_identity_check()
    from tools import exp189_s6_port as s6
    ids,paths=s6._study_ids([data/'train_series'/u for u in reg['binding']['ids']])
    need(ids==reg['binding']['ids'] and all(p.parent.parent==data.resolve() for p in paths),'resolved data path drift')
    need(sha(data/'train_series.csv')==b.sha(d.series_bytes(d.series_rows('timing'))),'metadata drift')
    identities={}
    for kind in ('r7','r6'):
        runtime=WORK/'cpu-runtime'/kind
        for name,raw in files(kind).items(): b.old.write(runtime/name,raw)
        sys.path[:0]=[str(runtime),str(d.gold.RUNTIME/'deps')]
        port=d.import_path('hb174_preflight_'+kind,runtime/b.PORT)
        m,identity=port.verify_bundle(bundle(kind),source_dir=runtime/'exp169_r7_src',
            expected_manifest_sha256=reg['binding']['manifests'][kind])
        import shutil
        def scratch_copy(src,dst,*args,**kwargs):
            need(Path(dst).resolve().is_relative_to((WORK/'cpu-runtime').resolve()),'scratch boundary')
            return shutil.copytree(src,dst,*args,**dict(kwargs,dirs_exist_ok=True))
        port.shutil=types.SimpleNamespace(copytree=scratch_copy)
        source=port.diagnostic_source(runtime/'exp169_r7_src',runtime)
        instrumented=d.instrument((source/'compact_infer.py').read_text(encoding='utf-8'))
        compile(instrumented,'<hb174-'+kind+'>','exec')
        b.old.write(runtime/'instrumented.py',instrumented.encode())
        identities[kind]=dict(manifest=identity['manifest_sha256'],source_sha256=b.sha(instrumented.encode()),
                              worker_args=port.worker_args(m,bundle(kind),d.ANATOMY))
    return save('prepare',state='CPU_PREPARED',binding_sha256=reg['binding_sha256'],
                data=inventory,identities=identities,studies=len(ids),cuda_executed=False)


def validate():
    reg=current()
    prepared=prepare()
    parity=io.read_json(b.WORK/'cpu-parity/result.json')
    from tools import c85r7_r6_cpu_parity as p
    need(parity['state']=='PASS_REAL_FULL_STUDY_CPU_PARITY' and parity['binding']==p.inputs()[1],
         'fresh real CPU parity required')
    # Recompute from the actual two output CSVs, never trust a PASS string alone.
    p.evaluate()
    parity_receipts()
    evidence={str((b.WORK/'cpu-parity'/arm/name).relative_to(ROOT)):sha(b.WORK/'cpu-parity'/arm/name)
              for arm in ('serving','public')
              for name in ('predictions.csv','controller/submission.csv','controller/submission_receipt.json',
                           'parity-receipt.json','provenance.json','invocation.json')}
    evidence[str((WORK/'cpu-parity-receipts.json').relative_to(ROOT))]=sha(WORK/'cpu-parity-receipts.json')
    return save('validate',state='CPU_VALIDATED',binding_sha256=reg['binding_sha256'],
                prepare_sha256=sha(WORK/'prepare.json'),parity_sha256=sha(b.WORK/'cpu-parity/result.json'),
                evidence=evidence,studies=prepared['studies'],cuda_executed=False)


def cpu_evidence_check():
    validation=io.read_json(WORK/'validate.json')
    need(sha(WORK/'prepare.json')==validation['prepare_sha256'],'prepared evidence drift')
    prepared=io.read_json(WORK/'prepare.json')
    need(prepared['binding_sha256']==validation['binding_sha256'],'prepared binding drift')
    for kind in ('r7','r6'):
        raw=(WORK/'cpu-runtime'/kind/'instrumented.py').read_text(encoding='utf-8')
        need(b.sha(raw.encode())==prepared['identities'][kind]['source_sha256'],
             'prepared instrumentation drift')
    need(validation.get('evidence'),'real CPU evidence pins missing')
    for name,expected in validation['evidence'].items():
        path=(ROOT/name).resolve()
        need(path.is_relative_to(b.WORK.resolve()),'CPU evidence outside owned scratch')
        need(sha(path)==expected,'real CPU evidence drift')
    need(sha(b.WORK/'cpu-parity/result.json')==validation['parity_sha256'],'parity result drift')


def parity_receipts():
    """Check completed REAL outputs, including silent auxiliary fallbacks."""
    expected=io.read_json(b.BUNDLE/'manifest.json')
    evidence={}
    for arm in ('serving','public'):
        path=b.WORK/'cpu-parity'/arm
        saved=io.read_json(path/'parity-receipt.json')
        receipt=io.read_json(path/'controller/submission_receipt.json')
        need(saved['controller_receipt']==receipt and saved['rows']==2,'CPU controller receipt drift')
        need(receipt['status']=='PASS' and receipt['rows']==receipt['scored']==2 and
             receipt['fallback_studies']==0 and receipt['fallback_reasons']=={} and
             receipt['worker_exit_codes'] and all(v==0 for v in receipt['worker_exit_codes'].values()),
             'CPU fallback or failed worker')
        workers=receipt.get('workers')
        need(workers,'CPU worker receipts missing')
        scored=0
        for rec in workers:
            need(rec['status']=='PASS' and rec['models']==expected['models'] and rec['group2'] is None,
                 'CPU model recipe drift')
            need(rec.get('flip_tta',{}).get('mcl')=='plain pass only','CPU flip missing')
            need(rec['studies']==rec['scored'] and rec['fallback']=={} and rec['device']=='cpu' and
                 rec['auxiliary_device']=='cpu' and rec['cuda_initialized'] is False and rec['amp']=='fp32',
                 'CPU partial output or device drift')
            need(rec['fluid_stacker']['stack_fallback']=={} and rec['state_stacker']['stack_fallback']=={},
                 'CPU stacker fallback')
            scored+=rec['scored']
        need(scored==2,'CPU worker roster incomplete')
        evidence[arm]=dict(workers=len(workers),scored=scored,controller_sha256=sha(path/'controller/submission_receipt.json'))
    return save('cpu-parity-receipts',state='PASS_REAL_CPU_RECEIPTS',arms=evidence)


def keep_decision(score, runtime, independent_result_review=False):
    """User-supplied score only; this never admits a stage or launches work."""
    need(isinstance(score,(int,float)) and math.isfinite(score) and 0<=score<=1,'finite supplied score required')
    hours=runtime.get('gate_hours')
    need(isinstance(hours,(int,float)) and math.isfinite(hours) and hours>0,'valid replay runtime required')
    keep=score>=RULES['keep_score_min'] and hours<=RULES['limit_hours'] and independent_result_review is True
    return dict(decision='KEEP' if keep else 'HOLD',keep_score_min=RULES['keep_score_min'],
                exposure='EXPOSED',clean_oof=False,gate_admitted=False)


def check_arm(value,supervision,kind,ids):
    import numpy as np
    need(value.get('state')=='COMPLETE' and value.get('done')==32 and len(set(ids))==32,'incomplete N32')
    value=d.canonicalize_rows(value,ids)
    need(value.get('exposure')=='EXPOSED' and value.get('clean_oof') is False,'exposure missing')
    m=io.read_json(bundle(kind)/'manifest.json')
    expected=dict(models=m['models'],models2=[],flip_tta=True,workers=2,
                  **{k:m[k] for k in ('size','crop','views','slices')})
    need(value['runtime']==expected and len(m['models'])==4,'wrong model or recipe')
    need('4070 Ti SUPER' in value['device'],'wrong GPU')
    x=np.asarray(value['predictions'],float)
    need(x.shape==(32,12) and np.isfinite(x).all() and ((x>=0)&(x<=1)).all(),'invalid predictions')
    need(supervision['exit_code']==0 and supervision['error'] is None and not supervision['cleanup'],
         'failed supervision or unresolved cleanup')
    telemetry=supervision['telemetry']
    need(telemetry['complete'] and telemetry['memory_failure'] is None,'incomplete telemetry')
    d.r7.memory_admission(supervision['desktop_before_cuda'],[r['peak_reserved_bytes'] for r in value['rows']],
                          telemetry['peak_gpu_mib']*2**20)
    for row in value['rows']:
        d.measure.window_gate(row['windows'],row['batch_windows'])
        need(0<row['peak_allocated_bytes']<=row['peak_reserved_bytes'],'invalid allocator peaks')
        need(all(math.isfinite(row[k]) and row[k]>0 for k in ('gpu_seconds','study_seconds')),'invalid timing')
    need(math.isfinite(supervision['seconds']) and supervision['seconds']>0,'invalid wall time')
    return dict(value,seconds=supervision['seconds'])


def projection(arms,calibration):
    need(set(arms)==set(ORDER),'ABBA four arms required')
    reference=[(r['study'],r['windows'],r['batch_windows']) for r in arms['r7-1']['rows']]
    need(len(reference)==32 and len({r[0] for r in reference})==32,'unique N32 required')
    need(all([(r['study'],r['windows'],r['batch_windows']) for r in a['rows']]==reference for a in arms.values()),
         'paired window identity mismatch')
    hidden,local=calibration['r7_hidden_seconds'],calibration['conservative_plain_seconds']
    need(all(math.isfinite(v) and v>0 for v in (hidden,local)),'invalid transfer calibration')
    delta=[];wall=[]
    for i in (1,2):
        a,c=arms['r7-'+str(i)],arms['r6-'+str(i)]
        need(all(math.isfinite(x['seconds']) and x['seconds']>0 for x in (a,c)),'invalid elapsed time')
        wall.append(c['seconds']-a['seconds'])
        delta.extend([wall[-1],sum(r['gpu_seconds'] for r in c['rows'])-sum(r['gpu_seconds'] for r in a['rows'])])
    need(all(math.isfinite(v) for v in delta),'nonfinite delta')
    scale=hidden/local
    central=6.7+sum(wall)/2*scale/3600
    upper=6.73+max(0,*delta)*scale/3600
    return dict(central_hours=central,gate_hours=upper,limit_hours=9,runtime_pass=upper<=9,
                signed_delta_hours=sum(wall)/2*scale/3600,conservative_delta_hours=max(0,*delta)*scale/3600,
                method=RULES['projection'],limitation=RULES['limitation'],qualified_for_submission=False)


def history_replay():
    reg=current()
    need(sha(HISTORY/'registration.json')==HISTORY_REG and sha(HISTORY/'result.json')==HISTORY_RESULT,'history pin drift')
    hreg=io.read_json(HISTORY/'registration.json');hadmit=io.read_json(HISTORY/'admission.json')
    need(hreg['binding_sha256']==digest(hreg['binding'])==hadmit['binding_sha256'],'history binding drift')
    review=io.read_json(hadmit['review'])
    need(sha(hadmit['review'])==hadmit['review_sha256'] and review['verdict']=='PASS' and
         review['builder']!=review['reviewer'] and sha(HISTORY/'validate.json')==hadmit['validation_sha256'],
         'historical independent review missing')
    result=io.read_json(HISTORY/'result.json')
    need(sha(HISTORY/'attempt.json')==result['attempt_sha256'],'historical result/attempt drift')
    attempt=io.read_json(HISTORY/'attempt.json')
    need(attempt['state']=='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW','historical run incomplete')
    controls={};pins={}
    for i in (1,2):
        name='flip-'+str(i);path=HISTORY/name
        value=io.read_json(path/'worker-status.json');sup=io.read_json(path/'supervision.json')
        need(value['binding_sha256']==hreg['binding_sha256'] and value['predictions']==attempt['arms'][name]['predictions'],
             'historical predictions drift')
        checked=check_arm(value,sup,'r7',reg['binding']['ids'])
        prediction_check(path,checked,reg['binding']['ids'])
        for file in ('worker-status.json','supervision.json','output/predictions.csv','output/controller/submission.csv',
                     'output/controller/submission_receipt.json'):
            pins[str((path/file).relative_to(ROOT))]=sha(path/file)
        controls['r7-'+str(i)]=controls['r6-'+str(i)]=checked
    projected=projection(controls,reg['binding']['calibration'])
    need(projected['signed_delta_hours']==0 and projected['gate_hours']==6.73,'identity control failed')
    return save('cpu-replay',state='PASS_REAL_OUTPUT_IDENTITY_CONTROL',binding_sha256=reg['binding_sha256'],
                files=pins,projection=projected,measured_r6=False,limitation='Historical full-flip R7 vs itself; no R6 timing claim.')


def prediction_check(path,value,ids):
    import numpy as np
    from tools import exp189_s6_port as s6
    receipt=io.read_json(path/'output/controller/submission_receipt.json')
    need(receipt['status']=='PASS' and receipt['rows']==receipt['scored']==32 and receipt['fallback_studies']==0 and
         receipt['fallback_reasons']=={} and receipt['worker_exit_codes'] and
         all(v==0 for v in receipt['worker_exit_codes'].values()),'failed/fallback output')
    workers=receipt.get('workers')
    need(workers and sum(w['scored'] for w in workers)==32,'worker receipts missing')
    for w in workers:
        need(w['models']==value['runtime']['models'] and w['group2'] is None and
             w.get('flip_tta',{}).get('mcl')=='plain pass only','worker recipe differs')
        need(w['fluid_stacker']['stack_fallback']=={} and w['state_stacker']['stack_fallback']=={},
             'stacker fallback invalidates parity/timing evidence')
    for file in ('output/predictions.csv','output/controller/submission.csv'):
        rows=s6.read_predictions(path/file,ids)
        need(np.array_equal(np.array([[r[t] for t in d.gold.LABELS] for r in rows]),np.array(value['predictions'])),
             'real output/measurement mismatch')


def review_check(review,reg):
    need(review.get('verdict')=='PASS' and review.get('builder') and review.get('reviewer') and
         review['builder']!=review['reviewer'] and review.get('execution_authority')=='B','independent PASS required')
    need(review.get('binding_sha256')==reg['binding_sha256'] and
         review.get('validation_sha256')==sha(WORK/'validate.json') and
         review.get('replay_sha256')==sha(WORK/'cpu-replay.json') and
         review.get('kernel_sha256')==reg['binding']['kernel_sha256'] and
         review.get('scope')=='H-B-174 build and N32 execution','review binding/scope drift')


def admit(review=None,dry_run=False):
    reg=current()
    need(io.read_json(WORK/'validate.json')['binding_sha256']==reg['binding_sha256'],'validation drift')
    need(io.read_json(WORK/'cpu-replay.json')['binding_sha256']==reg['binding_sha256'],'replay drift')
    need(review is not None,'independent review missing; dry-run cannot authorize GPU')
    review=Path(review).resolve();review_check(io.read_json(review),reg)
    cpu_evidence_check()
    data_identity_check()
    return save('admit-dry-run' if dry_run else 'admission',state='DRY_RUN_PASS' if dry_run else 'ADMITTED',
        binding_sha256=reg['binding_sha256'],review=str(review),review_sha256=sha(review),
        validation_sha256=sha(WORK/'validate.json'),replay_sha256=sha(WORK/'cpu-replay.json'))


def admitted():
    reg=current();value=io.read_json(WORK/'admission.json')
    need(value['state']=='ADMITTED' and value['binding_sha256']==reg['binding_sha256'] and
         sha(value['review'])==value['review_sha256'] and sha(WORK/'validate.json')==value['validation_sha256'] and
         sha(WORK/'cpu-replay.json')==value['replay_sha256'],'admission drift')
    review_check(io.read_json(value['review']),reg)
    cpu_evidence_check()
    data_identity_check()
    return value


def command():
    return [str(d.r224.GPU_PYTHON),'-B',str(SCRIPT),'run']


def lane():
    import psutil
    from tools.job_runtime import LOGS,job_status
    parents=psutil.Process().parents();own={os.getpid(),*[p.pid for p in parents]}
    need(any('gpu_job.py' in ' '.join(p.cmdline()) for p in parents),'gpu_job lock required')
    job=job_status(LOGS,TAG,expected_command=command(),tail=False)
    need(job['state']=='running' and job['pid'] in own,'exclusive HB174 command-bound lane required')
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
                          capture_output=True,text=True,check=True,timeout=5)
    need(not d.r7.gpu_python_pids(result.stdout,own),'another Python ML process owns GPU')
    desktop=d.r7.desktop_snapshot();d.r7.live_memory_guard(desktop['used_mib'],desktop)
    need(desktop['uuid']==current()['binding']['calibration']['device_uuid'],'calibration 4070 UUID required')
    return desktop


def configure(arm):
    need(arm in ORDER,'unknown arm')
    work=WORK/arm
    d.r224.WORK,d.r224.TAG,d.r224.__file__=work,TAG,str(SCRIPT)
    d.r224.monitor=types.SimpleNamespace(**dict(vars(d._MONITOR),_cleanup=d.cleanup_owned))
    return work


def worker():
    need(os.environ.get('HB164_CONTROLLER_PID')==str(os.getppid()),'supervised worker only')
    admission=admitted();lane()
    arm=os.environ['HB174_ARM'];work=configure(arm)
    runtime=work/'runtime'
    for name,raw in files(arm).items(): b.old.write(runtime/name,raw)
    deps=d.gold.RUNTIME/'deps'
    sys.path[:0]=[str(runtime),str(deps)]
    os.environ['PYTHONPATH']=os.pathsep.join([str(runtime),str(deps),str(ROOT)])
    os.environ['HB153_MEASURE_STATUS']=str(work/'worker-status.json')
    port=d.import_path('exp169_r7_port',runtime/b.PORT)
    frozen=port.diagnostic_source
    def diagnostic(*args,**kwargs):
        path=frozen(*args,**kwargs);target=path/'compact_infer.py'
        io.retry(target.write_text,d.instrument(target.read_text(encoding='utf-8')),encoding='utf-8')
        return path
    port.diagnostic_source=diagnostic
    ids=current()['binding']['ids'];started=time.perf_counter()
    predictions=port.infer_studies([WORK/'data/train_series'/u for u in ids],bundle_dir=bundle(arm),
        code_dir=d.CODE,anatomy_weights=d.ANATOMY,output_dir=work/'output',source_dir=runtime/'exp169_r7_src',
        workers=2,amp='auto',device='cuda',max_fallback_fraction=0.)
    need([p[d.gold.UID] for p in predictions]==ids,'prediction order drift')
    value=io.read_json(work/'worker-status.json')
    value.update(identity=TAG,version=VERSION,state='COMPLETE',done=len(ids),
        predictions=[[p[t] for t in d.gold.LABELS] for p in predictions],seconds=time.perf_counter()-started,
        binding_sha256=admission['binding_sha256'])
    io.journal(work/'worker-status.json',value,work)
    return dict(state='WORKER_COMPLETE')


def evaluate():
    reg=current();attempt=io.read_json(WORK/'attempt.json')
    cpu_evidence_check()
    data_identity_check()
    need(attempt['state']=='COMPLETE_REQUIRES_REPLAY' and attempt['binding_sha256']==reg['binding_sha256'] and
         attempt['order']==list(ORDER),'complete registered ABBA required')
    arms={};pins={}
    for arm in ORDER:
        path=WORK/arm;raw=io.read_json(path/'worker-status.json');sup=io.read_json(path/'supervision.json')
        need(raw['identity']==TAG and raw['binding_sha256']==reg['binding_sha256'],'arm binding drift')
        need(sup['desktop_before_cuda']['uuid']==reg['binding']['calibration']['device_uuid'],'GPU identity drift')
        value=check_arm(raw,sup,arm,reg['binding']['ids']);prediction_check(path,value,reg['binding']['ids'])
        provenance=io.read_json(path/'output/provenance.json')
        need(provenance['manifest_sha256']==reg['binding']['manifests'][arm[:2]],'bundle provenance drift')
        for name,expected in files(arm).items(): need((path/'runtime'/name).read_bytes()==expected,'runtime source drift')
        expected=(WORK/'cpu-runtime'/arm[:2]/'instrumented.py').read_text(encoding='utf-8')
        need((path/'output/r7_runtime/compact_infer.py').read_text(encoding='utf-8')==expected,'instrumentation drift')
        for file in ('worker-status.json','supervision.json','output/predictions.csv','output/controller/submission.csv',
                     'output/controller/submission_receipt.json','output/provenance.json','output/invocation.json'):
            pins[str((path/file).relative_to(ROOT))]=sha(path/file)
        arms[arm]=value
    result=projection(arms,reg['binding']['calibration'])
    return save('post-run-replay',state='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW',
                binding_sha256=reg['binding_sha256'],files=pins,result=result,gate_admitted=False)


def run():
    admission=admitted();lane()
    need(not (WORK/'attempt.json').exists(),'preserve existing attempt; B must archive deliberately')
    attempt=dict(state='RUNNING',binding_sha256=admission['binding_sha256'],order=list(ORDER),completed=[])
    save('attempt',**attempt)
    try:
        for arm in ORDER:
            desktop=lane();work=configure(arm)
            need(not (work/'worker-status.json').exists(),'existing arm output')
            os.environ['HB174_ARM']=arm
            value,telemetry=d.r224.supervise(desktop)
            check_arm(value,io.read_json(work/'supervision.json'),arm,current()['binding']['ids'])
            attempt['completed'].append(arm);save('attempt',**attempt)
        attempt['state']='COMPLETE_REQUIRES_REPLAY';save('attempt',**attempt)
        return evaluate()
    except BaseException as exc:
        attempt.update(state='HOLD',reason=f'{type(exc).__name__}: {exc}');save('attempt',**attempt)
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','prepare','validate','replay','admit','run','_worker','evaluate'])
    p.add_argument('--review',type=Path);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    try:
        if a.action not in ('run','_worker'):
            need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU commands require CUDA_VISIBLE_DEVICES present empty')
        need(not a.dry_run or a.action=='admit','only admit supports dry-run')
        if a.action=='admit': result=admit(a.review,a.dry_run)
        else: result={'replay':history_replay,'_worker':worker}.get(a.action,globals().get(a.action))()
        print(json.dumps({k:result[k] for k in ('state','binding_sha256') if k in result}))
        return 0
    except (ValueError,OSError,ImportError) as exc:
        print(json.dumps(dict(state='HOLD',action=a.action,reason=str(exc))))
        return 2


if __name__=='__main__': raise SystemExit(main())
