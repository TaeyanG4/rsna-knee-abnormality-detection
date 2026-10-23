"""Q23 EXP138 S42 native four-head Gold58 readout v1.0.1.

Changelog 1.0.1: supervisor-inclusive time, scoped standalone readout and tested
fail-closed admission. Unexecuted 1.0.0 registration/source preserved in backups.
Changelog 1.0.0: new diagnostic, authenticated five-fold plain K36 contract,
preregistered final-column routing, CPU parity, B-only supervised GPU admission.
Frozen predecessors unchanged. No public notebook execution or downloads.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import r6_gold58_precheck_v1 as r
from tools import s6all_gold58_precheck_v1 as s
from tools import exp138_bundle_aug1_mh as e
from tools import exp151_bundle_release as release
from tools import exp117_multihead as mh
from tools import exp124_flip_tta as flip
from tools import gpu_queue

c,d,t=r.c,r.d,r.t
need,sha,digest,io=r.need,r.sha,r.digest,r.io
VERSION='1.0.1'
TAG='Q23_EXP138_GOLD58_V101'
BUILDER='Codex/q23-build'
SCRIPT=Path(__file__).resolve()
WORK=ROOT/'.local/scratch/q23-build/q23'
RULES=dict(r.RULES,seed=230016,macro_floor=0.,contusion_floor=0.)
HOLD=r.HOLD
PAIRS=[('EXP138',x) for x in ('R7flip','R6flip','O')]+[
    (b+'_'+route,b) for b in ('HB167','HB174') for route in ('EXP138','halfO')]
CONTRACT=dict(seed=42,folds=list(range(5)),heads=list(mh.HEAD_NAMES),arch='resnet50',
    corpus='raptor64_336_c140_s02',k=36,res=224,input_policy='center3',
    window_policy='within-slot-valid',norm='radimagenet',tta=False,batch_studies=1,
    workers=0,cpu_threads=4,cuda_amp='fp16',cpu_amp='none',runtime_policy='strict',
    aggregation='mean_folds(mean_heads(rank_full_cohort(sigmoid_logits_float32)))')


def stamp(**kw):
    return dict(identity=TAG,version=VERSION,exposure='EXPOSED',clean_oof=False,**kw)


def save(name,value):
    io.journal(WORK/(name+'.json'),value,WORK)
    return value


def launch_command(): return [str(t.GPU_PYTHON),'-B',str(SCRIPT),'run']


def modules():
    sys.path.insert(0,str(ROOT/'training/src'))
    from training.src import raptor_common as common
    from training.src import raptor_radimagenet as rad
    from training.src.raptor_runtime import configure_runtime
    return common,rad,configure_runtime


def corpus_for(p):
    common,_,_=modules()
    corpus=common.Corpus(ROOT/p['paths']['corpus'])
    need(corpus.contract==CONTRACT['corpus'],'corpus contract mismatch')
    return corpus


def data_identity(corpus,ids):
    import numpy as np
    need(len(ids)==58 and set(ids)<=set(corpus.ids),'Gold58 corpus coverage mismatch')
    # Hash only the selected real volumes/masks, no report text or labels.
    records={}
    for uid in ids:
        vol=np.asarray(corpus.volume(uid)); mask=np.asarray(corpus.mask(uid))
        records[uid]=dict(volume=hashlib.sha256(vol.tobytes()).hexdigest(),
            mask=hashlib.sha256(mask.tobytes()).hexdigest(),shape=list(vol.shape),dtype=str(vol.dtype))
    return dict(manifest_sha256=sha(corpus.dir/'manifest.json'),
        ids_sha256=sha(corpus.dir/'ids.npy'),masks_sha256=sha(corpus.dir/'masks.npy'),records=records)


def ancestry(ids):
    p=e.load_protocol(42); files={}; checkpoints=[]; counts=[]
    for fold in range(5):
        tag=e.tag_for(42,fold); run=ROOT/'artifacts/experiments'/tag
        files.update(e.verify_run(run,p,fold))
        entry=release.checkpoint_entry(ROOT,42,fold); entry['path']=str((run/'model_best.pt').relative_to(ROOT))
        checkpoints.append(entry)
        ordered=io.read_json(ROOT/p['folds'][str(fold)]['ordered_ids']['path'])
        overlaps={part:len(set(ids)&set(ordered[part])) for part in ('train','val')}
        need(not any(overlaps.values()),'Gold58 training/validation study overlap')
        counts.append(dict(fold=fold,overlaps=overlaps,train_count=len(ordered['train']),val_count=len(ordered['val'])))
    return p,dict(checkpoints=checkpoints,folds=counts,authenticated_outputs=files,
        study_exclusion='PASS: authenticated train/val UID lists, fixed-final provenance gold_access disabled',
        patient_independence='HOLD: Gold58 absent from training component map; unknown repeats/bilateral patients',
        q22='Own-fold OOF only; F1/F2 validation UIDs occur in the other fold training; not external independent replication')


def register():
    c.cpu_only()
    need(sha(c.PINS)==c.PINS_SHA,'historical cache seal drift')
    old=io.read_json(c.SOURCE/'registration.json'); ids=old['binding']['ids']
    need(len(ids)==len(set(ids))==58 and old['binding_sha256']==digest(old['binding']),'historical cohort drift')
    p,audit=ancestry(ids); corpus=corpus_for(p)
    inputs=[SCRIPT,ROOT/'tests/test_q23_exp138_gold58_v1.py',WORK/'SPEC.md',WORK/'PRE-REVIEW.md',
        c.PINS,c.SOURCE/'registration.json',ROOT/'storage/competition/train.csv']
    inputs += [ROOT/'tools'/x for x in ('exp144_gold58_radr50_variants.py','gpu_job.py','job_runtime.py')]
    inputs += [ROOT/'training/src'/x for x in ('raptor_common.py','raptor_radimagenet.py','raptor_runtime.py')]
    inputs += [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and any(Path(m.__file__).resolve().is_relative_to(ROOT/x) for x in ('tools','training'))]
    inputs += [r.WORK/n for n in ('registration.json','measurement.json','attempt.json','gpu-review.json')]+[s.R6_REVIEW]
    inputs += [e.manifest_path(42),ROOT/e.PROTOCOL/'manifest.json']
    need(all(x.is_file() for x in inputs),'registration input missing (R6 is required, never silently omitted)')
    calibration=t.transfer_calibration()
    uuid=io.read_json(c.SOURCE/'plain-1/supervision.json')['desktop_before_cuda']['uuid']
    need(uuid==calibration['device_uuid'],'calibration device mismatch')
    binding=dict(builder=BUILDER,ids=ids,targets=list(d.gold.LABELS),rules=RULES,
        comparisons=[list(x) for x in PAIRS],contract=CONTRACT,ancestry=audit,
        data=data_identity(corpus,ids),calibration=calibration,historical_device_uuid=uuid,
        r6_available=True,command=launch_command(),headroom_hours=2.27,decision=HOLD,
        formula='FINAL Contusion only: r(EXP138) or r(.5*O+.5*EXP138); O=r((17r(E143)+17r(E155)+12r(Q))/46)',
        inputs={str(x.relative_to(ROOT)):sha(x) for x in sorted(set(inputs))})
    value=stamp(state='REGISTERED',binding=binding,binding_sha256=digest(binding))
    if (WORK/'registration.json').exists(): need(io.read_json(WORK/'registration.json')==value,'immutable registration drift')
    save('registration',value)
    save('gpu-plan',gpu_queue.validate_plan([dict(tag=TAG,command=launch_command())]))
    return value


def current():
    reg=io.read_json(WORK/'registration.json'); b=reg['binding']
    need(reg['state']=='REGISTERED' and reg['binding_sha256']==digest(b) and b['builder']==BUILDER
         and b['contract']==CONTRACT and b['rules']==RULES and b['command']==launch_command()
         and b['comparisons']==[list(x) for x in PAIRS] and b['decision']==HOLD,'registration policy drift')
    for name,h in b['inputs'].items():
        path=(ROOT/name).resolve(); need(path.is_relative_to(ROOT) and sha(path)==h,'source drift: '+name)
    return reg


def prereview(reg):
    rev=io.read_json(WORK/'pre-review.json')
    need(rev.get('verdict')=='PASS' and rev.get('builder')==BUILDER and rev.get('reviewer')
         and rev['reviewer']!=BUILDER and rev.get('binding_sha256')==reg['binding_sha256']
         and rev.get('scope')=='CPU_IMPLEMENTATION_TESTS_PARITY','independent CPU prereview required')


def prepare():
    c.cpu_only(); reg=current(); prereview(reg)
    p,audit=ancestry(reg['binding']['ids'])
    need(audit==reg['binding']['ancestry'],'ancestry drift')
    need(data_identity(corpus_for(p),reg['binding']['ids'])==reg['binding']['data'],'Gold corpus drift')
    return save('prepare',stamp(state='CPU_PREPARED',binding_sha256=reg['binding_sha256']))


def dataset(p,ids):
    import numpy as np
    _,rad,_=modules()
    return rad.CenterSliceWindows(ROOT/p['paths']['corpus'],ids,
        {uid:np.zeros(12,np.float32) for uid in ids},224,36,False,'radimagenet',42,
        window_policy='within-slot-valid',input_policy='center3')


def load_model(entry,device):
    import torch
    common,_,_=modules(); path=ROOT/entry['path']
    need(sha(path)==entry['sha256'],'checkpoint drift')
    # Authenticated local checkpoint, safe tensor loader, no unsafe fallback.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ck=torch.load(path,map_location='cpu',weights_only=True)
    need(ck.get('tag')==entry['tag'] and ck.get('fold')==entry['fold'] and ck.get('arch')=='resnet50','checkpoint identity drift')
    need(ck.get('heads')==list(mh.HEAD_NAMES) and ck.get('lab')==list(d.gold.LABELS),'checkpoint head/target order drift')
    model=mh.build_multihead(common.build_model('resnet50',pretrained=False),4)
    model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval()


def probabilities(model,x,device):
    import torch
    with torch.inference_mode(), (torch.autocast('cuda',dtype=torch.float16) if device=='cuda' else contextlib.nullcontext()):
        return flip.served_probabilities('exp117',model(x.to(device)))


def aggregate(raw):
    import numpy as np
    a=np.asarray(raw)
    need(a.ndim==4 and a.shape[0]==5 and a.shape[2:]==(4,12) and a.shape[1]>0
         and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'invalid five-fold head probabilities')
    return np.mean([mh.rank_mean(x) for x in a],axis=0)


def infer(ids,device,reg,parity=False):
    start=time.perf_counter()
    import numpy as np
    import torch
    common,rad,runtime=modules(); torch.set_num_threads(4)
    runtime_receipt=runtime('none' if device=='cpu' else 'fp16',device,policy='strict')
    p=e.load_protocol(42); ds=dataset(p,ids); raw=[]; rows=[]; errors=[]
    corpus=corpus_for(p)
    for entry in reg['binding']['ancestry']['checkpoints']:
        model=load_model(entry,device); fold=[]
        if device=='cuda': save('gpu/worker-status',stamp(state='MODEL_LOADED',done=len(rows),rows=rows))
        for i,uid in enumerate(ids):
            if device=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            mark=time.perf_counter(); x,_,got=ds[i]; need(got==uid and tuple(x.shape)==(36,3,224,224),'input shape/order drift')
            pred=probabilities(model,x[None],device); fold.append(pred[0])
            if device=='cuda': torch.cuda.synchronize()
            row=dict(study=uid,fold=entry['fold'],seconds=time.perf_counter()-mark)
            if device=='cuda':
                row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            rows.append(row)
            if parity:
                centers=common.select_centers(corpus.mask(uid),corpus.depth,36,corpus.spec,'within-slot-valid',None)
                oracle_x=rad.normalise(rad.center3_windows_from(corpus.volume(uid),centers,224),'radimagenet')
                need(np.array_equal(x.numpy(),oracle_x),'direct windows oracle mismatch')
                with torch.inference_mode(): oracle=torch.sigmoid(model(torch.from_numpy(oracle_x)[None]).float()).numpy()
                error=float(np.max(np.abs(pred-oracle))); need(error<=1e-7,'raw four-head CPU parity mismatch')
                errors.append(error)
            if device=='cuda': save('gpu/worker-status',stamp(state='RUNNING',done=len(rows),rows=rows))
        raw.append(fold); del model
        if device=='cuda': torch.cuda.empty_cache()
    a=np.asarray(raw); result=dict(predictions=aggregate(a).tolist(),raw=a.tolist(),rows=rows,
        seconds=time.perf_counter()-start,parity_max_abs=errors,torch=str(torch.__version__),runtime=runtime_receipt)
    if device=='cuda': result['device']=torch.cuda.get_device_name(0)
    return result


def smoke():
    c.cpu_only(); reg=current(); prereview(reg)
    result=infer(reg['binding']['ids'][:1],'cpu',reg,parity=True); c.cpu_only()
    need(len(result['parity_max_abs'])==5,'all checkpoints must pass CPU parity')
    return save('cpu-smoke',stamp(state='PASS_REAL_ONE_STUDY_CPU_PARITY',binding_sha256=reg['binding_sha256'],
        folds=5,raw_shape=[5,1,4,12],max_abs=max(result['parity_max_abs']),seconds=result['seconds'],
        torch=result['torch'],note='raw probabilities and direct windows checked; single-study ranks intentionally not the oracle'))


def outputs(ids,common,failed,r7,r6,candidate):
    import numpy as np
    base=r.outputs(ids,common,failed,r7,r6); candidate=c.check_values(candidate)
    own=r.rank((17*r.rank(common['EXP143v2'])+17*r.rank(common['EXP155'])+12*r.rank(common['Q_A5q']))/46)
    values=dict(EXP138=candidate,R7flip=base['R7flip'],R6flip=base['R6flip'],O=own,
        HB167=base['HB167'],HB174=base['replacement'])
    col=list(d.gold.LABELS).index('Contusion')
    for b in ('HB167','HB174'):
        for route,pred in [('EXP138',r.rank(candidate)),('halfO',r.rank(.5*own+.5*candidate))]:
            value=values[b].copy(); value[:,col]=pred[:,col]
            need(np.array_equal(np.delete(value,col,axis=1),np.delete(values[b],col,axis=1)),'non-Contusion mutation')
            values[b+'_'+route]=value
    return values


def paired_metrics(labels,values,pairs=PAIRS):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    names=list(values); arrays=[c.check_values(values[n]) for n in names]
    boot,weights,attempts=r.bootstrap_auc(labels,arrays,RULES)
    auc=np.array([[roc_auc_score(labels[:,j],v[:,j]) for j in range(12)] for v in arrays])
    col=list(d.gold.LABELS).index('Contusion'); out={}
    for a,b in pairs:
        delta=auc[names.index(a)]-auc[names.index(b)]; draws=boot[names.index(a)]-boot[names.index(b)]
        item=dict(contusion=r.metric(delta[col],draws[:,col]))
        if a.startswith('HB'):
            item['macro12']=r.metric(delta.mean(),draws.mean(1))
        if a.startswith('HB'):
            unchanged=np.array_equal(np.delete(values[a],col,axis=1),np.delete(values[b],col,axis=1))
            item.update(other_targets_identical=unchanged,operational='PASS' if unchanged and delta[col]>0 and delta.mean()>=0 else 'FAIL')
        out[a+'-minus-'+b]=item
    return dict(comparisons=out,contusion_auc={n:float(auc[i,col]) for i,n in enumerate(names)},
        bootstrap=dict(valid_draws=len(weights),attempts=attempts,rules=RULES),decision=HOLD,qualified_for_submission=False)


def replay():
    import numpy as np
    c.cpu_only(); reg=current(); prereview(reg)
    old,common,failed,arms=c.context(); ids,labels=d.gold.gold(); need(ids==reg['binding']['ids'],'Gold label roster drift')
    r6=s.cached_r6(reg); need(r6 is not None,'required R6 cache unavailable')
    ref=arms['flip-1']['predictions']; values=outputs(ids,common,failed,ref,r6,ref)
    stats=paired_metrics(labels,{'HB167_copy':values['HB167'].copy(),'HB167':values['HB167']},[('HB167_copy','HB167')])
    need(stats['comparisons']['HB167_copy-minus-HB167']['macro12']['ci95']==[0.,0.],'identity bootstrap mismatch')
    for b in ('HB167','HB174'):
        for route in ('EXP138','halfO'):
            col=list(d.gold.LABELS).index('Contusion')
            need(np.array_equal(np.delete(values[b],col,1),np.delete(values[b+'_'+route],col,1)),'real routing replay changed other targets')
    return save('replay',stamp(state='PASS_REAL_IDENTITY_CONTROL',binding_sha256=reg['binding_sha256'],
        valid_draws=2000,r6_authenticated=True,candidate_metrics_read=False))


def validate():
    c.cpu_only(); reg=current(); prereview(reg); evidence={}
    for name,state in [('prepare','CPU_PREPARED'),('cpu-smoke','PASS_REAL_ONE_STUDY_CPU_PARITY'),('replay','PASS_REAL_IDENTITY_CONTROL')]:
        v=io.read_json(WORK/(name+'.json')); need(v['state']==state and v['binding_sha256']==reg['binding_sha256'],'invalid '+name)
        evidence[name]=sha(WORK/(name+'.json'))
    return save('validate',stamp(state='CPU_VALIDATED',binding_sha256=reg['binding_sha256'],evidence=evidence))


def review_ready():
    reg=current(); prereview(reg); valid=io.read_json(WORK/'validate.json'); rev=io.read_json(WORK/'gpu-review.json')
    need(valid['state']=='CPU_VALIDATED' and valid['binding_sha256']==reg['binding_sha256'],'invalid validation')
    for name,h in valid['evidence'].items(): need(sha(WORK/(name+'.json'))==h,'CPU receipt drift')
    need(rev.get('verdict')=='PASS' and rev.get('builder')==BUILDER and rev.get('reviewer') and rev['reviewer']!=BUILDER
         and rev.get('execution_authority')=='B' and rev.get('tag')==TAG and rev.get('binding_sha256')==reg['binding_sha256']
         and rev.get('validation_sha256')==sha(WORK/'validate.json'),'independent bound GPU review required')
    return reg


def launch_check():
    c.cpu_only(); reg=review_ready()
    need(not (WORK/'attempt.json').exists(),'previous attempt exists; no retry')
    return stamp(state='REVIEW_BOUND_LAUNCH_CHECK_PASS_NO_ADMISSION',binding_sha256=reg['binding_sha256'])


def admit_dry_run(): return launch_check()


def configure():
    t.WORK=WORK/'gpu'; t.TAG=TAG; t.__file__=str(SCRIPT)
    t.monitor=types.SimpleNamespace(**dict(vars(d._MONITOR),_cleanup=d.cleanup_owned))


def lane(reg):
    desktop=t.lane_admission()
    need(desktop['uuid']==reg['binding']['historical_device_uuid'],'same physical 4070 required')
    return desktop


def worker():
    need(os.environ.get('HB164_CONTROLLER_PID')==str(os.getppid()),'supervised worker only')
    reg=review_ready(); configure(); lane(reg)
    result=infer(reg['binding']['ids'],'cuda',reg)
    save('gpu/worker-status',stamp(state='COMPLETE',binding_sha256=reg['binding_sha256'],
        ids=reg['binding']['ids'],done=290,contract=CONTRACT,**result))
    return 0


def check_measurement(value,reg,desktop,telemetry):
    import numpy as np
    need(value['state']=='COMPLETE' and value['identity']==TAG and value['version']==VERSION
         and value['binding_sha256']==reg['binding_sha256'] and value['ids']==reg['binding']['ids']
         and value['contract']==CONTRACT and value['done']==290 and '4070 Ti SUPER' in value['device']
         and telemetry['complete'] and telemetry['memory_failure'] is None,'measurement incomplete')
    need(value['runtime']['effective_amp']=='fp16' and value['runtime']['policy']=='strict'
         and value['runtime']['device']=='cuda' and not value['runtime']['precision_fallback_allowed'],'GPU precision drift')
    expected=[(fold,uid) for fold in range(5) for uid in reg['binding']['ids']]
    need([(x['fold'],x['study']) for x in value['rows']]==expected,'measurement row identity/order mismatch')
    for row in value['rows']:
        need(math.isfinite(row['seconds']) and row['seconds']>0 and 0<row['peak_allocated_bytes']<=row['peak_reserved_bytes'],'invalid GPU row')
    need(math.isfinite(value['seconds']) and value['seconds']>=sum(x['seconds'] for x in value['rows']),'invalid full worker wall')
    need(math.isfinite(value['supervised_seconds']) and value['supervised_seconds']>=value['seconds'],'invalid supervisor wall')
    actual=c.check_values(value['predictions'])
    need(np.asarray(value['raw']).shape==(5,58,4,12) and np.array_equal(aggregate(value['raw']),actual),'native aggregation mismatch')
    d.r7.memory_admission(desktop,[x['peak_reserved_bytes'] for x in value['rows']],telemetry['peak_gpu_mib']*2**20)


def run():
    reg=review_ready(); p,audit=ancestry(reg['binding']['ids'])
    need(audit==reg['binding']['ancestry'] and data_identity(corpus_for(p),reg['binding']['ids'])==reg['binding']['data'],'runtime inputs drift')
    need(not (WORK/'attempt.json').exists() and not (WORK/'gpu').exists(),'previous attempt preserved; no automatic retry')
    configure(); desktop=lane(reg); (WORK/'gpu').mkdir()
    attempt=stamp(state='RUNNING',binding_sha256=reg['binding_sha256']); save('attempt',attempt)
    try:
        value,telemetry=t.supervise(desktop)
        value['supervised_seconds']=io.read_json(WORK/'gpu/supervision.json')['seconds']
        check_measurement(value,reg,desktop,telemetry)
        value.update(desktop=desktop,telemetry=telemetry)
        attempt.update(state='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW',value=value); save('attempt',attempt)
        save('measurement',stamp(state=attempt['state'],binding_sha256=reg['binding_sha256'],
            attempt_sha256=sha(WORK/'attempt.json'),review_sha256=sha(WORK/'gpu-review.json'),
            files={name:sha(WORK/name) for name in ('gpu/worker-status.json','gpu/supervision.json')}))
        return stamp(state=attempt['state'])
    except BaseException as exc:
        attempt.update(state='HOLD',reason=f'{type(exc).__name__}: {exc}'); save('attempt',attempt); raise


def projection(seconds,calibration):
    need(math.isfinite(seconds) and seconds>0,'invalid measured seconds')
    factor=calibration['scale']; need(math.isfinite(factor) and factor>0,'invalid calibration scale')
    extra=seconds*32/58*factor/3600
    return dict(incremental_hours=extra,historical_headroom_hours=2.27,projected_total_hours=6.73+extra,
        fits_historical_headroom=extra<=2.27,local58_seconds=seconds,local32_to_hidden_scale=factor,
        implied_same_count_t4_factor=factor*32/1300,local58_limit_seconds=2.27*3600*58/32/factor,
        method='supervisor-inclusive local58 wall *32/58 * authenticated R7 localN32-to-hidden scale /3600',
        qualification='HOLD: cross-architecture transfer; Gold cached-corpus excludes corpus creation/decode; Linux/T4 not measured',
        assumption=calibration['assumption'])


def evaluate():
    from tools.job_runtime import job_status,LOGS
    c.cpu_only(); reg=review_ready()
    need(job_status(LOGS,TAG,expected_command=launch_command(),tail=False)['state']=='SUCCESS','successful exclusive job required')
    measured=io.read_json(WORK/'measurement.json'); attempt=io.read_json(WORK/'attempt.json')
    need(measured['state']==attempt['state']=='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW'
         and measured['binding_sha256']==attempt['binding_sha256']==reg['binding_sha256']
         and measured['attempt_sha256']==sha(WORK/'attempt.json') and measured['review_sha256']==sha(WORK/'gpu-review.json'),'measurement seal drift')
    for name,h in measured['files'].items():
        path=(WORK/name).resolve(); need(path.is_relative_to(WORK) and sha(path)==h,'GPU artifact drift')
    value=attempt['value']; check_measurement(value,reg,value['desktop'],value['telemetry'])
    sup=io.read_json(WORK/'gpu/supervision.json')
    need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[],'supervision failure')
    need(value['supervised_seconds']==sup['seconds'],'supervisor time mismatch')
    old,common,failed,arms=c.context(); ids,labels=d.gold.gold(); need(ids==reg['binding']['ids'],'label roster drift')
    stats=paired_metrics(labels,outputs(ids,common,failed,arms['flip-1']['predictions'],s.cached_r6(reg),value['predictions']))
    current(); c.cpu_only()
    return save('evaluation',stamp(state='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW',binding_sha256=reg['binding_sha256'],
        result=stats,timing=projection(value['supervised_seconds'],reg['binding']['calibration']),decision=HOLD,qualified_for_submission=False))


def main(argv=None):
    import json
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['register','prepare','smoke','replay','validate','launch-check','admit-dry-run','run','_worker','evaluate'])
    args=ap.parse_args(argv)
    if args.action=='_worker': return worker()
    value=globals()[args.action.replace('-','_')]()
    print(json.dumps({k:value[k] for k in ('identity','version','state')})); return 0


if __name__=='__main__': raise SystemExit(main())
