"""Gold58 candidate extension v1.0.7 of c85r7_r224flip_diagnostics.

Changelog 1.0.7: authenticated stack/mirror formulas, registered paired CIs,
one independently admitted R224 pair job. No serving-stage edits. EXPOSED.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r224flip_diagnostics as d
from tools import c85r7_stackmirq_v1127 as composition
from tools import c85r7_r224mir_cpu as mirror_cpu
from tools import gpu_queue

VERSION = '1.0.7'
TAG = 'HB170_GOLD58_V107'
BUILDER = 'astra-b1'
WORK = d.WORK/'gold/v1.0.7'
SOURCE = d.WORK/'gold/v1.0.5'
PINS = d.WORK/'gold/v1.0.6/preserved-pins.json'
PINS_SHA = 'a3debbd5a122a0763756fcbacf0c6e85e0f60d11ffefab776d10805c4309dcd6'
SCRIPT = Path(__file__).resolve()
t, io, need, sha, digest = d.r224, d.io, d.need, d.sha, d.digest
RULES = dict(macro_floor=-.001, weak_four_floor=-.002, draws=2000,
             seed=167170172, max_attempts=200000, quantiles=[.025,.975],
             quantile_method='linear', reject='any single-class target',
             ranks='full58 once before paired study bootstrap',
             decision='point estimate floors only; independent result review required',
             exposure='EXPOSED', clean_oof=False)
PAIRS = [('HB168','plain'), ('HB168','HB167'), ('HB170','HB167'),
         ('HB172','HB168'), ('HB172','HB167')]


def stamp(**kw):
    return dict(identity=TAG, version=VERSION, exposure='EXPOSED', clean_oof=False, **kw)


def save(name, value):
    io.journal(WORK/(name+'.json'), value, WORK)
    return value


def cpu_only():
    import torch
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
         'present-empty CUDA_VISIBLE_DEVICES and uninitialized CUDA required')


def context():
    reg, arrays, failed, read = d.preserved_context(SOURCE, PINS, PINS_SHA)
    arms = {a:d.preserved_arm(SOURCE, a, reg, read)[0] for a in ('plain-1','flip-1')}
    need([(r['study'],r['windows'],r['batch_windows']) for r in arms['plain-1']['rows']] ==
         [(r['study'],r['windows'],r['batch_windows']) for r in arms['flip-1']['rows']], 'paired R7 window drift')
    return reg, arrays, failed, arms


def authenticated_sources():
    snapshots = composition.snapshots()
    combined, proof = composition.merge(snapshots)
    # All embedded members are pinned; no notebook cell is executed.
    files = {name:s['files'] for name,s in snapshots.items()}
    need(files['parent'][composition.HELPER] == d.stage_files(d.PARENT)[composition.HELPER],
         'plain/HB167 blend identity drift')
    return files, dict(gates={k:s['receipt_sha256'] for k,s in snapshots.items()},
        embedded={k:{n:d.pack.digest(b) for n,b in f.items()} for k,f in files.items()},
        combined={n:d.pack.digest(b) for n,b in combined.items()}, composition_proof=proof)


def launch_command():
    return [str(t.GPU_PYTHON), '-B', str(SCRIPT), 'run']


def common_runtime(reg):
    common=reg['binding']['common']['receipt']
    proof=io.read_json(ROOT/common['members']['R224']['receipt_path'])
    folder=(ROOT/common['predictions_path']).parent
    path=folder/'worker.json'; worker=io.read_json(path)
    need(sha(path)==proof['worker_sha256']==common['worker_sha256'] and
         worker['state']=='INFERENCE_COMPLETE' and worker['studies']==58,
         'authenticated common worker required')
    return folder,worker


def register():
    cpu_only()
    reg, _, _, _ = context()
    _, sources = authenticated_sources()
    need(sha(t.CHECKPOINT) == t.CHECKPOINT_SHA, 'R224 checkpoint drift')
    # Gold DICOM provenance is authenticated by the already sealed preparation.
    manifest = d.gold.GOLD_DICOM/'manifest.json'
    expected = reg['binding']['data']['manifest_sha256']
    need(sha(manifest) == expected, 'historical Gold DICOM manifest drift')
    common_folder,common_worker=common_runtime(reg)
    paths = [SCRIPT, d.SCRIPT, WORK/'SPEC.md', ROOT/'tests/test_c85r7_gold58_candidates.py',
             ROOT/'tests/test_c85r7_r224flip_diagnostics.py', t.SOURCE,
             t.CHECKPOINT, d.r7.SERIES, manifest, d.gold.GOLD_DICOM/'train_series.csv',
             common_folder/'worker.json', common_folder/'desktop.json']
    paths += [Path(m.__file__).resolve() for m in list(sys.modules.values())
              if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(ROOT/'tools')]
    binding = dict(builder=BUILDER, rules=RULES, comparisons=[list(x) for x in PAIRS],
        ids=reg['binding']['ids'], targets=list(d.gold.LABELS), weak_four=list(d.gold.WEAK),
        historical_binding_sha256=reg['binding_sha256'], preserved_evidence_sha256=PINS_SHA,
        common=reg['binding']['common'], sources=sources, command=launch_command(),
        common_runtime=dict(torch=common_worker['torch'],device=common_worker['device'],
            r224_seconds=common_worker['member_seconds']['R224'],
            desktop=io.read_json(common_folder/'desktop.json'),
            limitation='Historical common worker records torch/device, not full dependency versions'),
        inputs={str(p.resolve().relative_to(ROOT)):sha(p) for p in sorted(set(paths))},
        differences=['Exact admitted AST functions; canonical sequential Windows reader, not Linux fork/fastread',
                     'Single local 4070 R224 plain/mirror pair; 94 windows; first two are steady-state memory smoke',
                     'Full-flip R7/common outputs reused; EXPOSED; no clean OOF or Linux parity claim'])
    value = stamp(state='REGISTERED', binding=binding, binding_sha256=digest(binding))
    path = WORK/'registration.json'
    if path.exists(): need(io.read_json(path) == value, 'registration immutable; changed inputs require new version')
    save('registration', value)
    save('gpu-plan', gpu_queue.validate_plan([dict(tag=TAG, command=launch_command())]))
    return value


def current():
    reg = io.read_json(WORK/'registration.json')
    b = reg['binding']
    need(reg['state'] == 'REGISTERED' and reg['binding_sha256'] == digest(b) and
         b['rules'] == RULES and b['comparisons'] == [list(x) for x in PAIRS] and
         b['command'] == launch_command(), 'registration policy drift')
    for name,h in b['inputs'].items(): need(sha(ROOT/name) == h, 'registered input drift: '+name)
    need(sha(PINS) == b['preserved_evidence_sha256'] == PINS_SHA, 'preserved seal drift')
    _, sources = authenticated_sources()
    need(sources == b['sources'], 'admitted gate/source drift')
    return reg


def namespace():
    files,_ = authenticated_sources()
    raw = files['mirror'][composition.SOURCE]
    ns = t.public_namespace()
    allowed = set(t.FUNCTIONS) | set(mirror_cpu.EXTRA)
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in allowed
             or isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in
                 (*t.CONSTANTS,'SLOT_SIZES','SAGITTAL_SLOTS') for x in n.targets)]
    need({n.name for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef))} == allowed,
         'admitted function allowlist incomplete')
    exec(compile(ast.Module(body=nodes,type_ignores=[]), '<admitted-r224mirq-functions>', 'exec'), ns)
    need(ns['IMG']==384 and ns['CROP_FRAC']==.8 and ns['K_EVAL']==94 and ns['NORM']=='imagenet'
         and ns['ARMS']==t.source_parts()[0]['ARMS'] and ns['LAB']==list(d.gold.LABELS), 'serving recipe drift')
    return ns


def prepare():
    cpu_only(); reg=current()
    verified=d.verify_data('gold')
    ns=namespace()
    need(ns['MAXS']==96, 'volume depth drift')
    value=stamp(state='CPU_PREPARED', binding_sha256=reg['binding_sha256'], data=verified,
                series_sha256=digest(d.series_rows('gold')), cuda_initialized=False)
    return save('prepare',value)


def observed_pair(ns, model, plain, mirrored, device):
    """Transparent observation of the admitted function; no inference reimplementation."""
    original=ns['infer_probs']; outputs=[]; precision=[]
    def observed(model_arg, windows, dev):
        seen=[]
        def forward(x):
            seen.append('float16' if ns['torch'].is_autocast_enabled('cuda') else 'float32')
            return model_arg(x)
        p=original(forward,windows,dev)
        outputs.append(p); precision.append(seen)
        return p
    ns['infer_probs']=observed
    try: result=ns['infer_pair_probs'](model,plain,mirrored,device)
    finally: ns['infer_probs']=original
    need(len(outputs)==2 and ns['np'].array_equal(result,.5*(outputs[0]+outputs[1])), 'pair arithmetic drift')
    need(all(x in (['float16'],['float16','float32'],['float32']) for x in precision), 'pair precision drift')
    return result,precision


def check_values(values, n=58):
    import numpy as np
    a=np.asarray(values)
    need(a.shape==(n,12) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(), 'finite bounded Nx12 required')
    return a


def smoke():
    cpu_only(); reg=current(); pre=io.read_json(WORK/'prepare.json')
    need(pre['binding_sha256']==reg['binding_sha256'], 'fresh prepare required')
    ns=namespace(); torch,np=ns['torch'],ns['np']; torch.set_num_threads(4)
    arm=ns['ARMS'][0]; model,res=ns['load_model'](str(t.CHECKPOINT),arm['arch'],arm['res'],'cpu')
    grouped,reader=t.group_series(d.series_rows('gold')),ns['_make_reader']()
    results=[]; coverage=set(); start=time.perf_counter()
    for uid in reg['binding']['ids'][:2]:
        vol,mask=ns['build_study'](uid,grouped,str(d.gold.GOLD_DICOM/'train_series'),reader)
        t.validate_volume(vol,mask)
        cs=np.array([max(1,min(c,94)) for c in ns['_eval_centers'](mask,96,94)])
        tri=torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs-1,cs,cs+1],1)]))
        slots=np.searchsorted(np.cumsum([0,26,22,18,12,18]),cs,side='right')-1
        coverage.update(map(int,slots))
        oracle=torch.stack([x.flip(0) if slot in (0,1) else x.flip(-1) for x,slot in zip(tri,slots)])
        need(torch.equal(ns['_mirror_tri'](tri,cs,96),oracle), 'anatomical uint8 mirror axis mismatch')
        xw=ns['gpu_windows'](vol,mask,94,res,'cpu')
        xm=ns['gpu_windows'](vol,mask,94,res,'cpu',mirror=True)
        expected=torch.nn.functional.interpolate(oracle.float()[:,:,38:345,38:345]/255.,
                 size=(224,224),mode='bilinear',align_corners=False)
        need(torch.equal(xm,(expected-ns['_MEAN'])/ns['_STD']), 'mirror crop/normalization mismatch')
        p,precision=observed_pair(ns,model,xw,xm,'cpu'); check_values(p[None],1)
        results.append(dict(study=uid, shape=list(xw.shape), mirror_shape=list(xm.shape),
            volume_sha256=hashlib.sha256(vol.tobytes()).hexdigest(),
            pair_sha256=hashlib.sha256(p.tobytes()).hexdigest(), precision=precision,
            anatomical_oracle=True,pair_arithmetic_exact=True))
    need(coverage=={0,1,2,3,4}, 'all anatomical slots required')
    cpu_only()
    return save('cpu-smoke',stamp(state='PASS_CPU_REAL_GOLD_PAIR',binding_sha256=reg['binding_sha256'],
        studies=results, seconds=time.perf_counter()-start, cuda_initialized=False,
        limitation='Two real Gold58 studies; CPU engineering only; no CUDA/performance evidence'))


def blends(ids,common,failed,arms,files,mirror=None):
    import numpy as np
    need(len(ids)==len(set(ids))==58 and set(common)==set(failed)==set(d.COMMON), 'cohort/member drift')
    for m in d.COMMON:
        check_values(common[m]); need(np.asarray(failed[m]).shape==(58,) and
            np.asarray(failed[m]).dtype==np.bool_, 'explicit boolean masks required')
    masks={m:{u:'authenticated common failure' for u,f in zip(ids,failed[m]) if f} for m in d.COMMON}
    plain=check_values(arms['plain-1']['predictions']); flip=check_values(arms['flip-1']['predictions'])
    parent=composition.fluid.math_functions(files['parent'])['blend']
    stack=composition.fluid.math_functions(files['fluid'])['blend']
    def blend(fn,r224,r7,m):
        return check_values(fn(ids,common['R'],r224,r7,common['EXP143v2'],common['EXP155'],common['Q_A5q'],m))
    outputs=dict(plain=blend(parent,common['R224'],plain,masks),
                 HB167=blend(parent,common['R224'],flip,masks),
                 HB168=blend(stack,common['R224'],flip,masks))
    if mirror is not None:
        # Measured mirror requires zero fallback; never carry old plain-R224 failures.
        mirror=check_values(mirror); d.formula.failure_gate(np.all(mirror==.5,axis=1))
        mir_masks=dict(masks,R224={})
        outputs.update(HB170=blend(parent,mirror,flip,mir_masks),HB172=blend(stack,mirror,flip,mir_masks))
    return outputs


def paired_metrics(labels,outputs):
    """Exact empirical AUC via tie-aware pair counts, validated against sklearn in tests."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=np.asarray(labels)
    need(y.shape==(58,12) and np.isin(y,[0,1]).all() and
         all(len(np.unique(y[:,j]))==2 for j in range(12)), 'binary nondegenerate Gold58 required')
    names=list(outputs); values=np.stack([check_values(outputs[n]) for n in names])
    auc=np.array([[roc_auc_score(y[:,j],v[:,j]) for j in range(12)] for v in values])
    rng=np.random.default_rng(RULES['seed']); weights=[]; attempts=0
    while len(weights)<RULES['draws']:
        need(attempts<RULES['max_attempts'], 'bootstrap degeneracy limit')
        w=np.bincount(rng.integers(0,58,58),minlength=58); attempts+=1
        pos=w@y
        if np.any((pos==0)|(pos==58)): continue
        weights.append(w)
    weights=np.asarray(weights,dtype=float)
    boot=np.empty((len(names),RULES['draws'],12))
    for j in range(12):
        pos=y[:,j]==1; neg=~pos; wp=weights[:,pos]; wn=weights[:,neg]
        denominator=wp.sum(1)*wn.sum(1)
        for k,v in enumerate(values):
            a=v[pos,j,None]; b=v[neg,j][None,:]
            pair=(a>b).astype(float)+.5*(a==b)
            boot[k,:,j]=np.sum((wp@pair)*wn,axis=1)/denominator
    result={}
    for a,b in PAIRS:
        if a not in outputs: continue
        delta=auc[names.index(a)]-auc[names.index(b)]
        draws=boot[names.index(a)]-boot[names.index(b)]
        macro,weak=float(delta.mean()),float(delta[list(d.gold.WEAK)].mean())
        result[a+'-minus-'+b]=dict(macro_delta=macro,weak_four_delta=weak,
            macro_ci95=np.quantile(draws.mean(1),RULES['quantiles'],method='linear').tolist(),
            weak_four_ci95=np.quantile(draws[:,list(d.gold.WEAK)].mean(1),RULES['quantiles'],method='linear').tolist(),
            per_target_delta=delta.tolist(), no_harm_pass=macro>=RULES['macro_floor'] and weak>=RULES['weak_four_floor'])
    return dict(comparisons=result, per_target_auc={n:auc[i].tolist() for i,n in enumerate(names)},
                bootstrap=dict(rules=RULES,valid_draws=len(weights),attempts=attempts,rejected=attempts-len(weights)),
                qualified_for_submission=False)


def validate_measurement(value,ids,binding_sha,desktop,telemetry):
    need(value.get('identity')==TAG and value.get('version')==VERSION and value.get('state')=='COMPLETE'
         and value.get('done')==58 and value.get('binding_sha256')==binding_sha
         and value.get('clean_oof') is False and value.get('exposure')=='EXPOSED', 'complete bound mirror measurement required')
    rows=value['rows']
    need(len(ids)==len(set(ids))==58 and [r['study'] for r in rows]==ids, 'mirror study order/coverage drift')
    check_values([r['probabilities'] for r in rows])
    for r in rows:
        need(r['windows']==94 and r['resolution']==224 and r['failed_study_fallback'] is False
             and r['actual_pair'] is True and r['retain_until_replacement'] is True
             and len(r['forward_precision'])==2 and all(x in (['float16'],['float16','float32'],['float32'])
                 for x in r['forward_precision']) and math.isfinite(r['seconds']) and r['seconds']>0
             and 0<r['peak_allocated_bytes']<=r['peak_reserved_bytes'], 'invalid pair row/recipe/memory')
    need(math.isfinite(value['seconds']) and value['seconds']>=sum(r['seconds'] for r in rows)-1e-6,
         'invalid measured time')
    need(telemetry['memory_failure'] is None, 'telemetry memory failure')
    need(desktop['uuid']==value['device_uuid'], 'GPU provenance mismatch')
    t.shared.memory_admission(desktop,[r['peak_reserved_bytes'] for r in rows],
        telemetry['peak_gpu_mib']*2**20 if telemetry['peak_gpu_mib']>0 else None)


def replay_measurement(reg,arrays):
    """Real common memory/probability schema control; NOT measured mirror evidence."""
    folder,worker=common_runtime(reg)
    desktop=io.read_json(folder/'desktop.json')
    memories=[r for r in worker['memory_measurements'] if r['member']=='R224']
    need([r['study_index'] for r in memories]==list(range(58)), 'common memory order drift')
    seconds=worker['member_seconds']['R224']
    rows=[dict(study=uid,probabilities=p.tolist(),windows=94,resolution=224,
        failed_study_fallback=False,actual_pair=True,retain_until_replacement=True,
        forward_precision=[['float16'],['float16']],seconds=seconds/58,
        peak_allocated_bytes=m['peak_allocated_bytes'],peak_reserved_bytes=m['peak_reserved_bytes'])
        for uid,p,m in zip(reg['binding']['ids'],arrays['R224'],memories)]
    adapted=stamp(state='COMPLETE',done=58,binding_sha256=reg['binding_sha256'],
        rows=rows,seconds=seconds,device_uuid=desktop['uuid'])
    telemetry=reg['binding']['common']['receipt']['telemetry']
    validate_measurement(adapted,reg['binding']['ids'],reg['binding_sha256'],desktop,telemetry)
    return dict(state='PASS_SCHEMA_CONTROL',actual_mirror=False,
        source_worker_sha256=sha(folder/'worker.json'),
        limitation='Real plain outputs/memory; pair flags and equal per-study timing are schema adapters only')


def replay():
    """Real preserved outputs exercise all post-evaluation paths, never mirror evidence."""
    cpu_only(); reg=current(); old,arrays,failed,arms=context(); files,_=authenticated_sources()
    ids,labels=d.gold.gold()
    normal=blends(ids,arrays,failed,arms,files)
    adapted=blends(ids,arrays,failed,arms,files,arrays['R224'])
    import numpy as np
    need(not failed['R224'].any(), 'identity replay requires zero failed plain R224')
    need(np.array_equal(normal['HB167'],adapted['HB170']) and
         np.array_equal(normal['HB168'],adapted['HB172']), 'real identity control differs')
    # Exercise paired bootstrap for all five comparison paths, without publishing surrogate metrics.
    metrics=paired_metrics(labels,adapted)
    need(metrics['comparisons']['HB170-minus-HB167']['macro_ci95']==[0.,0.], 'identity CI drift')
    measurement_control=replay_measurement(reg,arrays)
    return save('postchecks-replay',stamp(state='PASS_REAL_IDENTITY_CONTROL',binding_sha256=reg['binding_sha256'],
        comparisons_exercised=list(metrics['comparisons']),mirrored_predictions_measured=False,
        measurement_schema_control=measurement_control,
        limitation='Actual common plain R224 substitutes for mirror ONLY in this CPU control; no candidate-b result'))


def validate():
    cpu_only(); reg=current()
    evidence={}
    for name,state in [('prepare','CPU_PREPARED'),('cpu-smoke','PASS_CPU_REAL_GOLD_PAIR'),
                       ('postchecks-replay','PASS_REAL_IDENTITY_CONTROL')]:
        p=WORK/(name+'.json'); value=io.read_json(p)
        need(value['state']==state and value['binding_sha256']==reg['binding_sha256'], 'fresh '+name+' required')
        evidence[name]=sha(p)
    return save('validate',stamp(state='CPU_VALIDATED',binding_sha256=reg['binding_sha256'],evidence=evidence))


def check_review(review,reg,validation):
    need(review.get('verdict')=='PASS' and review.get('builder')==BUILDER and
         isinstance(review.get('reviewer'),str) and review['reviewer'].strip() and review['reviewer']!=BUILDER and
         review.get('execution_authority')=='B' and review.get('binding_sha256')==reg['binding_sha256'] and
         review.get('validation_sha256')==validation and review.get('tag')==TAG and
         review.get('approved_differences')==reg['binding']['differences'], 'independent bound PASS required')


def validation_current(reg):
    value=io.read_json(WORK/'validate.json')
    need(value['state']=='CPU_VALIDATED' and value['binding_sha256']==reg['binding_sha256'], 'fresh validation required')
    for name,h in value['evidence'].items(): need(sha(WORK/(name+'.json'))==h, 'CPU evidence drift')
    return sha(WORK/'validate.json')


def admit(review,dry_run=False):
    cpu_only(); reg=current(); v=validation_current(reg)
    need(review is not None, 'independent review required even for dry-run')
    check_review(io.read_json(review),reg,v)
    return save('admit-dry-run' if dry_run else 'admission',stamp(state='DRY_RUN_PASS' if dry_run else 'ADMITTED',
        binding_sha256=reg['binding_sha256'],validation_sha256=v,review=str(Path(review).resolve()),review_sha256=sha(review)))


def admitted():
    reg=current(); a=io.read_json(WORK/'admission.json')
    need(a['state']=='ADMITTED' and a['binding_sha256']==reg['binding_sha256'] and
         a['validation_sha256']==validation_current(reg) and sha(a['review'])==a['review_sha256'], 'admission drift/missing')
    check_review(io.read_json(a['review']),reg,a['validation_sha256'])
    return reg,a


def install_supervisor():
    t.WORK=WORK; t.TAG=TAG; t.VERSION=VERSION; t.__file__=str(SCRIPT)
    t.monitor._cleanup=d.cleanup_owned  # v1.0.6 bounded identity-safe exit observation.


def worker():
    need(os.environ.get('HB164_CONTROLLER_PID')==str(os.getppid()), 'supervised worker only')
    reg,_=admitted(); install_supervisor(); desktop=t.lane_admission()
    ns=namespace(); torch=ns['torch']
    need(str(torch.__version__)==reg['binding']['common_runtime']['torch'], 'common-cache torch version drift')
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and
         '4070 Ti SUPER' in torch.cuda.get_device_name(0), 'exclusive registered 4070 required')
    torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True
    runtime=t.runtime_snapshot(ns); arm=ns['ARMS'][0]
    rows=[]; start=time.perf_counter()
    def publish(state):
        save('worker-status',stamp(state=state,done=len(rows),rows=rows,runtime=runtime,
            device_uuid=desktop['uuid'],binding_sha256=reg['binding_sha256'],seconds=time.perf_counter()-start))
    model,res=ns['load_model'](str(t.CHECKPOINT),arm['arch'],arm['res'],'cuda:0')
    need(res==224,'resolution drift'); publish('MODEL_LOADED')
    grouped,reader=t.group_series(d.series_rows('gold')),ns['_make_reader']()
    for uid in reg['binding']['ids']:
        tick=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        vol,mask=ns['build_study'](uid,grouped,str(d.gold.GOLD_DICOM/'train_series'),reader)
        t.validate_volume(vol,mask)
        # Deliberately retain preceding xw/xm until assignment completes, as serving does.
        xw=ns['gpu_windows'](vol,mask,94,res,'cuda:0')
        xm=ns['gpu_windows'](vol,mask,94,res,'cuda:0',mirror=True)
        need(list(xw.shape)==list(xm.shape)==[94,3,224,224], 'full pair shape drift')
        p,precision=observed_pair(ns,model,xw,xm,'cuda:0'); torch.cuda.synchronize()
        check_values(p[None],1)
        allocated,reserved=torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
        t.shared.memory_admission(desktop,[reserved])
        rows.append(dict(study=uid,probabilities=p.tolist(),windows=94,resolution=224,
            failed_study_fallback=False,actual_pair=True,retain_until_replacement=True,
            forward_precision=precision,seconds=time.perf_counter()-tick,
            peak_allocated_bytes=allocated,peak_reserved_bytes=reserved))
        publish('INFERENCE')
    publish('COMPLETE')
    return 0


def run():
    reg,_=admitted()
    need(not (WORK/'worker-status.json').exists() and not (WORK/'supervision.json').exists(),
         'existing attempt preserved; no automatic retry')
    need(d.verify_data('gold')==io.read_json(WORK/'prepare.json')['data'], 'prepared DICOM drift')
    install_supervisor(); desktop=t.lane_admission()
    need(desktop['uuid']==io.read_json(SOURCE/'plain-1/supervision.json')['desktop_before_cuda']['uuid'],
         'pair/cache must use same physical GPU')
    need(desktop['uuid']==reg['binding']['common_runtime']['desktop']['uuid'], 'common-cache physical GPU drift')
    value,telemetry=t.supervise(desktop)
    validate_measurement(value,reg['binding']['ids'],reg['binding_sha256'],desktop,telemetry)
    return save('measurement',stamp(state='MEASURED_REQUIRES_INDEPENDENT_RESULT_REVIEW' if telemetry['complete']
        else 'HOLD_TELEMETRY',binding_sha256=reg['binding_sha256'],raw_sha256=sha(WORK/'worker-status.json'),
        supervision_sha256=sha(WORK/'supervision.json'),seconds=value['seconds'],telemetry=telemetry,
        desktop=desktop,runtime=value['runtime'],mirrored_predictions_measured=True))


def measured_mirror(reg):
    from tools.job_runtime import job_status, LOGS
    need(job_status(LOGS,TAG,expected_command=launch_command(),tail=False)['state']=='SUCCESS',
         'completed successful exclusive job required before output read')
    _,a=admitted(); measured=io.read_json(WORK/'measurement.json')
    need(measured['state']=='MEASURED_REQUIRES_INDEPENDENT_RESULT_REVIEW' and
         measured['binding_sha256']==reg['binding_sha256'] and measured['mirrored_predictions_measured'] is True and
         measured['raw_sha256']==sha(WORK/'worker-status.json') and
         measured['supervision_sha256']==sha(WORK/'supervision.json'), 'measured mirror receipt drift')
    sup=io.read_json(WORK/'supervision.json')
    need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[] and
         sup['telemetry']==measured['telemetry'] and sup['telemetry']['complete'], 'mirror supervision failure')
    value=io.read_json(WORK/'worker-status.json')
    validate_measurement(value,reg['binding']['ids'],a['binding_sha256'],measured['desktop'],measured['telemetry'])
    return check_values([r['probabilities'] for r in value['rows']])


def evaluate(part='a'):
    cpu_only(); reg=current(); _,arrays,failed,arms=context(); files,_=authenticated_sources()
    ids,labels=d.gold.gold(); need(ids==reg['binding']['ids'], 'label roster drift')
    outputs=blends(ids,arrays,failed,arms,files,measured_mirror(reg) if part=='b' else None)
    result=paired_metrics(labels,outputs)
    # Reauthenticate preserved inputs before commit; original receipts never rewritten.
    seal=io.read_json(PINS)
    need(sha(PINS)==PINS_SHA and all(sha(ROOT/p)==h for p,h in seal['files'].items()), 'evidence changed during evaluation')
    current(); cpu_only()
    return save('evaluation-'+part,stamp(state='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW',
        binding_sha256=reg['binding_sha256'],part=part,result=result,mirrored_predictions_measured=part=='b',
        cuda_initialized=False,qualified_for_submission=False))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['register','prepare','smoke','replay','validate','admit','evaluate','run','_worker'])
    ap.add_argument('--part',choices=['a','b'],default='a'); ap.add_argument('--review',type=Path)
    ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args(argv)
    try:
        need(not a.dry_run or a.action=='admit', 'dry-run only supported for admission')
        if a.action=='_worker': return worker()
        if a.action=='admit': value=admit(a.review,a.dry_run)
        elif a.action=='evaluate': value=evaluate(a.part)
        else: value=globals()[a.action]()
        print(json.dumps({k:value[k] for k in ('identity','version','state')})); return 0
    except (OSError,ValueError,ImportError,KeyError) as exc:
        print(json.dumps(stamp(state='HOLD',action=a.action,reason=str(exc)))); return 2


if __name__=='__main__': raise SystemExit(main())
