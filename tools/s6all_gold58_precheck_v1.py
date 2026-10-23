"""H-B-136 S6-all Gold58 precheck v1.0.1.

Changelog 1.0.1: bind copied adapter code verifier to trusted original source
closure (Python default SOURCE was captured at copied-module import time).
1.0.0 preparation receipts preserved under fixed scratch/v1.0.0; no inference ran.

Changelog 1.0.0: separate diagnostic extension of pinned S6 adapter; safe
weights-only loading, preregistered three contrasts, authenticated cache reuse,
CPU controller/direct parity and independent B-only GPU admission. Frozen
predecessors are unchanged. Downloaded code and s6_extras.bin never execute.
"""
from __future__ import annotations
import argparse
import contextlib
import difflib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import r6_gold58_precheck_v1 as r
from tools import exp189_s6_port as s6
from tools import gpu_queue

c,d,t = r.c,r.d,r.t
need,sha,digest,io = r.need,r.sha,r.digest,r.io
VERSION='1.0.1'
TAG='S6ALL_GOLD58_V101'
BUILDER='Codex/s6all-build'
SCRIPT=Path(__file__).resolve()
WORK=ROOT/'.local/scratch/s6all-build/hb136'
BUNDLE=ROOT/'storage/external/nartaa-s6fall-v1'
MODEL='E_prior_s42_fall.pt'
MODEL_SHA='0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'
NOTEBOOK=ROOT/'.local/idea-scout/20261016-0519/notebooks/nartaa__rsna-knee-s6fall-solo/rsna-knee-s6fall-solo.py'
R6_REVIEW=r.WORK/'INDEPENDENT-RESULT-REVIEW.md'
RULES=dict(r.RULES,per_target_floor=-.01)
PAIRS=[('S6all','R7flip'),('S6all_HB167','HB167'),('S6all_HB174','HB174')]
HOLD=r.HOLD
DIFFERENCES=[
    'Pinned S6 adapter; new safe checkpoint, E_prior arm, size320 crop288, no state/side/stacker; chunk192 explicit',
    'Local CPU present-empty envelope and receipt fields; GPU observation only; downloaded code never executed',
    '4070 amp auto/bf16 matches preserved cache; original public T4 fp16/Linux parity NOT CHECKED',
    'EXPOSED selection panel; no clean OOF or submission qualification claim']


def stamp(**kw):
    return dict(identity=TAG,version=VERSION,exposure='EXPOSED',clean_oof=False,selection_exposed=True,**kw)


def save(name,value):
    io.journal(WORK/(name+'.json'),value,WORK)
    return value


def write(path,raw):
    path=Path(path)
    need(path.resolve().is_relative_to(WORK.resolve()),'write outside fixed scratch')
    path.parent.mkdir(parents=True,exist_ok=True)
    io.retry(path.write_bytes,raw)


def bundle_identity(load=False):
    need(sha(BUNDLE/MODEL)==MODEL_SHA,'S6-all checkpoint hash mismatch')
    need({p.name for p in BUNDLE.glob('E_prior_s*_f*.pt')}=={MODEL},'unexpected checkpoint roster')
    config=io.read_json(BUNDLE/'E_prior_s42_fall.run_config.json')
    expected=dict(backbone='convnext_small',size=320,crop=288,views='v4',slices=3,
        phys_mm=3.5,arms='E_prior',spatial_attn=True,state_spec=None,seed_base=42,all_data=True)
    need(all(config.get(k)==v for k,v in expected.items()),'S6-all registered recipe mismatch')
    # Recipe is fixed by the public notebook and registered separately from training settings.
    value=dict(checkpoint=MODEL,sha256=MODEL_SHA,bytes=(BUNDLE/MODEL).stat().st_size,
               config_sha256=sha(BUNDLE/'E_prior_s42_fall.run_config.json'))
    if load:
        import torch
        try: state=torch.load(BUNDLE/MODEL,map_location='cpu',weights_only=True)
        except Exception as exc: raise ValueError('weights_only=True failed closed; no unsafe retry: '+type(exc).__name__) from exc
        need(type(state) is dict and state and all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()),
             'checkpoint must be plain tensor state dictionary')
        need(len(state)==362 and {'sp_q','sp_key.weight','sp_pos.weight','sp_head_norm.weight',
            'sp_head_norm.bias'}<=set(state),'S6-all state architecture mismatch')
        value['state_keys']=len(state)
        del state
    return value


def adapter_bytes():
    """Exactly scoped changes to trusted local adapter; no downloaded source import."""
    raw=Path(s6.__file__).read_bytes()
    for old,new in [
        (b'"--bundle", str(bundle_dir), "--arm", "E_lrprior"',b'"--bundle", str(bundle_dir), "--arm", "E_prior"'),
        (b'"--backbone", "convnext_small", "--size", "384", "--crop", "352",',b'"--backbone", "convnext_small", "--size", "320", "--crop", "288", "--chunk", "192",'),
        (b'                    "--state-spec", str(Path(bundle_dir) / "state_spec.json"),\n'
         b'                    "--side-model", str(Path(bundle_dir) / "side_cls.pt"),\n'
         b'                    "--side-threshold", "2.0", "--side-position-mm", "20.0", "--side-max-x", "0.0",\n',b''),
        (b'env["CUDA_VISIBLE_DEVICES"] = "-1"',b'env["CUDA_VISIBLE_DEVICES"] = ""')]:
        raw=r.once(raw,old,new)
    return raw


def export_adapter():
    raw=adapter_bytes()
    write(WORK/'contract/exp189_s6_port.py',raw)
    diff=''.join(difflib.unified_diff(Path(s6.__file__).read_text().splitlines(True),raw.decode().splitlines(True),
        fromfile='trusted/exp189_s6_port.py',tofile='s6all/exp189_s6_port.py'))
    write(WORK/'adapter.diff',diff.encode())
    return d.pack.digest(raw)


def launch_command(): return [str(t.GPU_PYTHON),'-B',str(SCRIPT),'run']


def register():
    c.cpu_only()
    # Metadata and file hashes only, before ANY labels/prediction deserialization.
    need(sha(c.PINS)==c.PINS_SHA,'historical seal drift')
    old=io.read_json(c.SOURCE/'registration.json')
    ids=old['binding']['ids']
    need(len(ids)==len(set(ids))==58 and old['binding_sha256']==digest(old['binding']),'historical cohort invalid')
    uuid=io.read_json(c.SOURCE/'plain-1/supervision.json')['desktop_before_cuda']['uuid']
    contract=export_adapter(); identity=bundle_identity(load=True)
    inputs=[SCRIPT,ROOT/'tests/test_s6all_gold58_precheck_v1.py',WORK/'SPEC.md',WORK/'PRE-REVIEW.md',NOTEBOOK,
        WORK/'v1.0.0/registration.json',WORK/'v1.0.0/replay.json',
        c.SOURCE/'registration.json',c.PINS,d.gold.GOLD_DICOM/'manifest.json',d.gold.GOLD_DICOM/'train_series.csv',
        d.gold.RUNTIME/'prepare.json',ROOT/'storage/competition/train.csv',WORK/'contract/exp189_s6_port.py']
    inputs += [ROOT/'tools'/n for n in ('gpu_job.py','job_runtime.py','hb167_common_gold58_v1.py')]
    inputs += [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
               and Path(m.__file__).resolve().is_relative_to(ROOT/'tools')]
    inputs += [p for p in BUNDLE.iterdir() if p.is_file()]
    inputs += list(s6.SOURCE.glob('*.py'))
    r6_available=all(p.is_file() for p in [r.WORK/'registration.json',r.WORK/'measurement.json',r.WORK/'attempt.json',R6_REVIEW])
    if r6_available:
        inputs += [r.WORK/n for n in ('registration.json','measurement.json','attempt.json','gpu-review.json')]+[R6_REVIEW]
    need(all(p.is_file() for p in inputs),'registration input missing')
    binding=dict(builder=BUILDER,ids=ids,targets=list(d.gold.LABELS),rules=RULES,comparisons=[list(p) for p in PAIRS],
        formula='A=.65r(R)+.35r(R224); O=r((17r(E143)+17r(E155)+12r(Q))/46); F(S)=.85r(.5A+.5S)+.15O; mixed slot=.60r(base)+.40r(S6all)',
        decision=HOLD,identity=identity,adapter_sha256=contract,preserved_evidence_sha256=c.PINS_SHA,
        historical_device_uuid=uuid,r6_available=r6_available,command=launch_command(),differences=DIFFERENCES,
        prior_exposure='Original v1.0.0 registration fixed decisions before Gold labels; zero-delta R7 replay then read labels. v1.0.1 fixes source closure only; no S6 predictions read.',
        inherited_preregistration_sha256=sha(WORK/'v1.0.0/registration.json'),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(inputs))})
    value=stamp(state='REGISTERED',binding=binding,binding_sha256=digest(binding))
    if (WORK/'registration.json').exists(): need(io.read_json(WORK/'registration.json')==value,'immutable registration drift')
    save('registration',value)
    save('gpu-plan',gpu_queue.validate_plan([dict(tag=TAG,command=launch_command())]))
    return value


def current():
    reg=io.read_json(WORK/'registration.json'); b=reg['binding']
    need(reg['state']=='REGISTERED' and b['builder']==BUILDER and reg['binding_sha256']==digest(b)
         and b['rules']==RULES and b['comparisons']==[list(p) for p in PAIRS] and b['decision']==HOLD
         and b['command']==launch_command() and b['differences']==DIFFERENCES,'registration policy drift')
    for name,h in b['inputs'].items():
        path=(ROOT/name).resolve(); need(path.is_relative_to(ROOT) and sha(path)==h,'registered input drift: '+name)
    need(d.pack.digest(adapter_bytes())==b['adapter_sha256'],'adapter drift')
    return reg


def prereview(reg):
    review=io.read_json(WORK/'pre-review.json')
    need(review.get('verdict')=='PASS' and review.get('builder')==BUILDER and review.get('reviewer')
         and review['reviewer']!=BUILDER and review.get('binding_sha256')==reg['binding_sha256']
         and review.get('scope')=='CPU_IMPLEMENTATION_TESTS_PARITY','independent preregistration review required')


def prepare():
    c.cpu_only(); reg=current(); prereview(reg)
    record=io.read_json(d.gold.RUNTIME/'prepare.json'); deps=Path(record['deps'])
    need(d.gold.dir_tree(deps)['tree_sha256']==record['deps_tree_sha256'],'dependency tree drift')
    need(d.gold.probe_deps(deps,d.CODE)==record['versions'],'dependency versions drift')
    s6.verify_source(); code=s6.verify_code(d.CODE,d.ANATOMY)
    data=d.gold.verify_dicom(d.gold.GOLD_DICOM,reg['binding']['ids'])
    return save('prepare',stamp(state='CPU_PREPARED',binding_sha256=reg['binding_sha256'],data=data,
        code=code,deps_tree_sha256=record['deps_tree_sha256'],versions=record['versions']))


def instrument(raw):
    raw=r.once(raw,'    out = Path(a.output)',
        '    from tools import c85r7_r224flip_diagnostics as hb153\n'
        '    hb153_rows = []\n    torch.cuda.reset_peak_memory_stats()\n'
        '    hb153.window_gate(1, a.chunk)\n    out = Path(a.output)')
    raw=r.once(raw,'        tg = time.perf_counter()',
        '        hb153.window_gate(len(item["images"]), a.chunk)\n'
        '        torch.cuda.synchronize()\n        hb153_started = time.perf_counter()\n        tg = time.perf_counter()')
    raw=r.once(raw,'        if a.dump_logits:\n',
        '        hb153.commit(torch, Path(os.environ["HB153_MEASURE_STATUS"]), hb153_rows,\n'
        '                     uid, len(item["images"]), a.chunk, hb153_started,\n'
        '                     item["seconds"][0], item["seconds"][1],\n'
        '                     item["seconds"][2] if len(item["seconds"]) > 2 else 0., wall_started=mark,\n'
        '                     runtime=dict(models=[p.name for p in files], models2=[],\n'
        '                         flip_tta=False, workers=a.workers, size=a.size, crop=a.crop, views=a.views, slices=a.slices))\n'
        '        if a.dump_logits:\n')
    compile(raw,'<S6-all-measurement>','exec')
    return raw


def diagnostic(source,out,device):
    runtime=s6._diagnostic_source(source,out)
    controller=runtime/'probe_dual.py'
    raw=r.once(controller.read_bytes(),b'names, visible = ["CPU"], ["-1"]',b'names, visible = ["CPU"], [""]')
    write(controller,raw)
    worker=runtime/'compact_infer.py'; raw=worker.read_text(encoding='utf-8')
    raw=r.once(raw,'    dev = torch.device(a.device)',
        '    if a.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":\n'
        '        raise RuntimeError("CPU visibility must be present-empty")\n    dev = torch.device(a.device)')
    raw=r.once(raw,'"cuda_initialized": torch.cuda.is_initialized(),',
        '"cuda_initialized": torch.cuda.is_initialized(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), '
        '"size": a.size, "crop": a.crop, "slices": a.slices, "chunk": a.chunk, "state_spec": a.state_spec,')
    if device=='cuda': raw=instrument(raw)
    write(worker,raw.encode())
    io.journal(runtime/'runtime_sha256.json',{p.name:sha(p) for p in sorted(runtime.glob('*.py'))},WORK)
    return runtime


def port_module(device):
    port=d.import_path('hb136_s6all_port',WORK/'contract/exp189_s6_port.py')
    port.SOURCE=s6.SOURCE
    port.verify_code=s6.verify_code
    port.verify_bundle=lambda bundle: bundle_identity(load=True)
    def provenance(path,**kwargs):
        io.journal(path,stamp(checkpoint_sha256=MODEL_SHA,source_sha256=s6.SOURCE_HASHES,
                             code=s6.verify_code(d.CODE,d.ANATOMY)),WORK)
    port.write_provenance=provenance
    port._diagnostic_source=lambda source,out: diagnostic(source,out,device)
    return port


def worker_check(worker,ids,device):
    need(worker['status']=='PASS' and worker['arm']=='E_prior' and worker['models']==[MODEL]
        and worker['group2'] is None and worker['backbone']=='convnext_small'
        and worker['size']==320 and worker['crop']==288 and worker['slices']==3 and worker['chunk']==192
        and worker['views']=='v4' and worker['phys_mm']==3.5 and worker['state_spec'] is None
        and worker['side_check'] is None and worker['studies']==worker['scored']==len(ids)
        and worker['fallback']=={} and worker['device']==worker['auxiliary_device']==device
        and worker['timm']=='1.0.25' and worker['anchor_weights_sha256']==s6.ANATOMY_SHA256,'S6-all worker recipe/fallback drift')
    if device=='cpu':
        need(worker['amp']=='fp32' and worker['cuda_initialized'] is False and worker['cuda_visible_devices']=='','CPU envelope drift')
    else: need(worker['amp']=='bf16' and worker['torch']=='2.5.1+cu121','cached 4070 precision/software drift')


def check_controller(out,ids,device):
    import numpy as np
    proof=io.read_json(out/'controller/submission_receipt.json')
    need(proof['status']=='PASS' and proof['rows']==proof['scored']==len(ids) and proof['fallback_studies']==0
        and proof['fallback_reasons']=={} and proof['worker_exit_codes'] and all(v==0 for v in proof['worker_exit_codes'].values())
        and len(proof['workers'])==1,'controller fallback/incomplete')
    worker=proof['workers'][0]; worker_check(worker,ids,device)
    args=io.read_json(out/'invocation.json')['args']; wa=json.loads(args[args.index('--worker-args')+1])
    need(wa==worker_args() and args[args.index('--arm')+1]=='E_prior' and args[args.index('--loader-workers')+1]=='2'
        and args[args.index('--max-fallback-fraction')+1]=='0.0','S6-all invocation drift')
    need(io.read_json(out/'provenance.json')['checkpoint_sha256']==MODEL_SHA,'checkpoint provenance drift')
    arrays=[]
    for name in ('predictions.csv','controller/submission.csv'):
        rows=s6.read_predictions(out/name,ids)
        arrays.append(c.check_values([[row[k] for k in d.gold.LABELS] for row in rows],len(ids)))
    need(np.array_equal(*arrays),'prediction copies differ')
    return arrays[0],worker


def worker_args():
    return ['--backbone','convnext_small','--size','320','--crop','288','--chunk','192',
        '--eval-stride','1','--views','v4','--slices','3','--phys-mm','3.5','--anchor-weights',str(d.ANATOMY.resolve())]


def infer(ids,device,out):
    port=port_module(device)
    with d.gold.worker_pythonpath(d.gold.RUNTIME/'deps'):
        port.infer_studies([d.gold.GOLD_DICOM/'train_series'/uid for uid in ids],bundle_dir=BUNDLE,
            code_dir=d.CODE,anatomy_weights=d.ANATOMY,output_dir=out,source_dir=s6.SOURCE,
            workers=2,amp='auto',device=device,max_fallback_fraction=0.)
    return check_controller(out,ids,device)


def smoke():
    import numpy as np
    c.cpu_only(); reg=current(); prereview(reg)
    need(io.read_json(WORK/'prepare.json')['binding_sha256']==reg['binding_sha256'],'fresh prepare required')
    ids=reg['binding']['ids'][:1]; started=time.perf_counter(); out=WORK/'cpu-smoke/controller'
    values,worker=infer(ids,'cpu',out)
    # Independent direct CLI worker route checks adapter/controller argument forwarding.
    direct=WORK/'cpu-smoke/direct'; need(not direct.exists(),'CPU parity output already exists')
    direct.mkdir(parents=True); stage=direct/'data'; stage.mkdir()
    s6._stage_studies([d.gold.GOLD_DICOM/'train_series'/uid for uid in ids],ids,stage)
    runtime=diagnostic(s6.SOURCE,direct,'cpu'); assignment=direct/'assignment.json'
    io.journal(assignment,ids,WORK)
    args=[str(t.GPU_PYTHON),'-u','-B',str(runtime/'compact_infer.py'),'--data',str(stage),'--split','test',
        '--code',str(d.CODE),'--models',str(BUNDLE),'--arm','E_prior','--assignment',str(assignment),
        '--output',str(direct/'worker'),'--amp','auto','--workers','2','--device','cpu',*worker_args()]
    with d.gold.worker_pythonpath(d.gold.RUNTIME/'deps'):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
            PYTHONPATH=os.pathsep.join([str(runtime),str(d.CODE),os.environ.get('PYTHONPATH','')]))
        s6._run_group(args,env,label='S6-all direct CPU parity')
    direct_receipt=io.read_json(direct/'worker/receipt.json'); worker_check(direct_receipt,ids,'cpu')
    rows=s6.read_predictions(direct/'worker/predictions.csv',ids)
    reference=c.check_values([[row[k] for k in d.gold.LABELS] for row in rows],1)
    need(np.array_equal(values,reference),'controller/direct six-decimal prediction parity failed')
    c.cpu_only(); current()
    files={str(p.relative_to(WORK)):sha(p) for p in (WORK/'cpu-smoke').rglob('*') if p.is_file() and p.name in
        ('submission_receipt.json','receipt.json','predictions.csv','submission.csv','invocation.json','runtime_sha256.json')}
    return save('cpu-smoke',stamp(state='PASS_REAL_ONE_STUDY_CPU_PARITY',binding_sha256=reg['binding_sha256'],
        exact_controller_direct_parity=True,study_count=1,seconds=time.perf_counter()-started,files=files,
        cuda_initialized=False,cuda_visible_devices='',limitation='CPU engineering only; no CUDA/Linux/performance qualification'))


def outputs(ids,common,failed,r7,s6all,r6=None):
    import numpy as np
    base=r.outputs(ids,common,failed,r7,r7 if r6 is None else r6)
    s6all=c.check_values(s6all)
    rr=.65*r.rank(common['R'])+.35*r.rank(common['R224'])
    own=r.rank((17*r.rank(common['EXP143v2'])+17*r.rank(common['EXP155'])+12*r.rank(common['Q_A5q']))/46)
    def final(slot): return .85*r.rank(.5*rr+.5*slot)+.15*own
    values=dict(S6all=s6all,R7flip=c.check_values(r7),HB167=base['HB167'],
        S6all_HB167=final(.60*r.rank(r7)+.40*r.rank(s6all)))
    if r6 is not None:
        values.update(HB174=base['replacement'],S6all_HB174=final(.60*r.rank(r6)+.40*r.rank(s6all)))
    need(np.array_equal(values['HB167'],base['HB167']),'HB167 identity drift')
    return values


def floors(delta):
    import numpy as np
    a=np.asarray(delta,float)
    need(a.shape==(12,) and np.isfinite(a).all(),'invalid target deltas')
    return bool(a.mean()>=RULES['macro_floor'] and a[RULES['weak_four']].mean()>=RULES['weak_four_floor']
                and np.all(a>=RULES['per_target_floor']))


def paired_metrics(labels,values):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    names=list(values); arrays=[c.check_values(values[n]) for n in names]
    boot,weights,attempts=r.bootstrap_auc(labels,arrays,RULES)
    auc=np.array([[roc_auc_score(labels[:,j],v[:,j]) for j in range(12)] for v in arrays]); result={}
    for a,b in PAIRS:
        if a not in values or b not in values:
            result[a+'-minus-'+b]=dict(operational='HOLD',reason='authenticated R6 baseline unavailable'); continue
        delta=auc[names.index(a)]-auc[names.index(b)]; draws=boot[names.index(a)]-boot[names.index(b)]
        result[a+'-minus-'+b]=dict(macro12=r.metric(delta.mean(),draws.mean(1)),
            weak_four=r.metric(delta[RULES['weak_four']].mean(),draws[:,RULES['weak_four']].mean(1)),
            per_target={target:r.metric(delta[j],draws[:,j]) for j,target in enumerate(d.gold.LABELS)},
            operational='PASS' if floors(delta) else 'FAIL',exposure='EXPOSED',clean_oof=False)
    return dict(comparisons=result,per_target_auc={n:auc[i].tolist() for i,n in enumerate(names)},
        bootstrap=dict(rules=RULES,valid_draws=len(weights),attempts=attempts,rejected=attempts-len(weights)),
        decision=HOLD,qualified_for_submission=False,training_exclusion='Author-attributed; unaudited. Gold selection-exposed.')


def cached_r6(reg):
    """Authenticate completed R6 without reading/computing its evaluation output."""
    import numpy as np
    from tools.job_runtime import job_status,LOGS
    if not reg['binding']['r6_available']: return None
    old=r.review_ready()
    need(job_status(LOGS,r.TAG,expected_command=r.launch_command(),tail=False)['state']=='SUCCESS','R6 exclusive job not successful')
    need(old['binding']['ids']==reg['binding']['ids'] and old['binding']['historical_device_uuid']==reg['binding']['historical_device_uuid'],
        'R6 cohort/device mismatch')
    need(R6_REVIEW.read_text(encoding='utf-8').startswith('# Independent RESULT review: R6/R7 Gold58 v1.0.1\n')
        and '1. PASS - Authenticated SUCCESS' in R6_REVIEW.read_text(encoding='utf-8'),'R6 independent result review absent')
    measured=io.read_json(r.WORK/'measurement.json'); attempt=io.read_json(r.WORK/'attempt.json')
    need(measured['state']==attempt['state']=='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW' and
        measured['binding_sha256']==attempt['binding_sha256']==old['binding_sha256'] and
        measured['attempt_sha256']==sha(r.WORK/'attempt.json') and measured['review_sha256']==sha(r.WORK/'gpu-review.json'),
        'R6 completion seal drift')
    for name,h in measured['files'].items():
        path=(r.WORK/name).resolve(); need(path.is_relative_to(r.WORK) and sha(path)==h,'R6 artifact drift')
    value=attempt['arms']['flip']; r.check_measurement(value,old,'flip',value['desktop'],value['telemetry'])
    sup=io.read_json(r.WORK/'gpu/flip/supervision.json')
    need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[],'R6 supervision failure')
    actual,_=r.check_controller(r.WORK/'gpu/flip/output',reg['binding']['ids'],'flip','cuda')
    need(np.array_equal(actual,value['predictions']),'R6 cached predictions drift')
    return actual


def replay():
    c.cpu_only(); reg=current(); prereview(reg)
    old,common,failed,arms=c.context(); ids,labels=d.gold.gold()
    need(ids==reg['binding']['ids'],'label roster drift')
    cached=cached_r6(reg)  # authentication only; no scientific metrics on this arm
    ref=arms['flip-1']['predictions']; stats=paired_metrics(labels,outputs(ids,common,failed,ref,ref,ref))
    for value in stats['comparisons'].values():
        need(value['macro12']['ci95']==value['weak_four']['ci95']==[0.,0.] and
             all(x['ci95']==[0.,0.] for x in value['per_target'].values()),'identity contrast nonzero')
    return save('replay',stamp(state='PASS_REAL_IDENTITY_CONTROL',binding_sha256=reg['binding_sha256'],
        valid_draws=stats['bootstrap']['valid_draws'],r6_authenticated=cached is not None,s6all_measured=False,
        limitation='R7 substituted for S6/R6 solely as identity control; no candidate metrics'))


def validate():
    c.cpu_only(); reg=current(); prereview(reg); evidence={}
    for name,state in [('prepare','CPU_PREPARED'),('cpu-smoke','PASS_REAL_ONE_STUDY_CPU_PARITY'),('replay','PASS_REAL_IDENTITY_CONTROL')]:
        value=io.read_json(WORK/(name+'.json'))
        need(value['state']==state and value['binding_sha256']==reg['binding_sha256'],'fresh '+name+' required')
        evidence[name]=sha(WORK/(name+'.json'))
    return save('validate',stamp(state='CPU_VALIDATED',binding_sha256=reg['binding_sha256'],evidence=evidence))


def check_review(review,reg,validation_sha):
    need(review.get('verdict')=='PASS' and review.get('builder')==BUILDER and review.get('reviewer')
        and review['reviewer']!=BUILDER and review.get('execution_authority')=='B' and review.get('tag')==TAG
        and review.get('binding_sha256')==reg['binding_sha256'] and review.get('validation_sha256')==validation_sha
        and review.get('approved_differences')==DIFFERENCES,'independent bound PASS required')


def review_ready():
    reg=current(); prereview(reg); valid=io.read_json(WORK/'validate.json')
    need(valid['state']=='CPU_VALIDATED' and valid['binding_sha256']==reg['binding_sha256'],'validation drift')
    for name,h in valid['evidence'].items(): need(sha(WORK/(name+'.json'))==h,'CPU evidence drift')
    for name,h in io.read_json(WORK/'cpu-smoke.json')['files'].items():
        path=(WORK/name).resolve(); need(path.is_relative_to(WORK) and sha(path)==h,'CPU parity artifact drift')
    check_review(io.read_json(WORK/'gpu-review.json'),reg,sha(WORK/'validate.json'))
    return reg


def launch_check():
    c.cpu_only(); reg=review_ready()
    return stamp(state='REVIEW_BOUND_LAUNCH_CHECK_PASS_NO_ADMISSION',binding_sha256=reg['binding_sha256'])


def configure():
    t.WORK=WORK/'gpu'; t.TAG=TAG; t.__file__=str(SCRIPT)
    t.monitor=types.SimpleNamespace(**dict(vars(d._MONITOR),_cleanup=d.cleanup_owned))
    return t.WORK


def lane(reg):
    desktop=t.lane_admission()
    need(desktop['uuid']==reg['binding']['historical_device_uuid'],'same physical 4070 required')
    return desktop


def runtime_check(value):
    need(value['runtime']==dict(models=[MODEL],models2=[],flip_tta=False,workers=2,size=320,crop=288,views='v4',slices=3),
        'S6-all measured runtime drift')


def worker():
    need(os.environ.get('HB164_CONTROLLER_PID')==str(os.getppid()),'supervised worker only')
    reg=review_ready(); work=configure(); lane(reg)
    os.environ['HB153_MEASURE_STATUS']=str(work/'worker-status.json')
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT),os.environ.get('PYTHONPATH','')])
    start=time.perf_counter(); ids=reg['binding']['ids']; values,receipt=infer(ids,'cuda',work/'output')
    measured=io.read_json(work/'worker-status.json'); runtime_check(measured)
    value=stamp(state='COMPLETE',binding_sha256=reg['binding_sha256'],done=len(ids),ids=ids,
        predictions=values.tolist(),rows=measured['rows'],runtime=measured['runtime'],device=measured['device'],
        seconds=time.perf_counter()-start,torch=receipt['torch'])
    io.journal(work/'worker-status.json',value,WORK); return 0


def check_measurement(value,reg,desktop,telemetry):
    runtime_check(value); ids=reg['binding']['ids']
    need(value['state']=='COMPLETE' and value['identity']==TAG and value['version']==VERSION
        and value['binding_sha256']==reg['binding_sha256'] and value['done']==58 and value['ids']==ids
        and [row['study'] for row in value['rows']]==ids and telemetry['complete'] and telemetry['memory_failure'] is None,
        'complete S6-all measurement required')
    c.check_values(value['predictions'])
    for row in value['rows']:
        d.measure.window_gate(row['windows'],row['batch_windows'])
        need(math.isfinite(row['gpu_seconds']) and row['gpu_seconds']>0 and
             0<row['peak_allocated_bytes']<=row['peak_reserved_bytes'],'invalid timing/memory')
    d.r7.memory_admission(desktop,[row['peak_reserved_bytes'] for row in value['rows']],telemetry['peak_gpu_mib']*2**20)


def run():
    reg=review_ready(); prepared=io.read_json(WORK/'prepare.json')
    record=io.read_json(d.gold.RUNTIME/'prepare.json')
    need(d.gold.dir_tree(Path(record['deps']))['tree_sha256']==prepared['deps_tree_sha256'],'runtime dependency drift')
    need(s6.verify_code(d.CODE,d.ANATOMY)==prepared['code'],'prepared code drift')
    need(d.gold.verify_dicom(d.gold.GOLD_DICOM,reg['binding']['ids'])==prepared['data'],'DICOM drift')
    need(not (WORK/'attempt.json').exists(),'previous attempt preserved; no automatic retry')
    work=configure(); desktop=lane(reg); need(not work.exists(),'GPU output already exists')
    work.mkdir(parents=True); attempt=stamp(state='RUNNING',binding_sha256=reg['binding_sha256']); save('attempt',attempt)
    try:
        value,telemetry=t.supervise(desktop); value=d.canonicalize_rows(value,reg['binding']['ids'])
        check_measurement(value,reg,desktop,telemetry); value.update(desktop=desktop,telemetry=telemetry)
        attempt.update(state='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW',value=value); save('attempt',attempt)
        files={str(p.relative_to(WORK)):sha(p) for p in work.rglob('*') if p.is_file() and p.name in
            ('supervision.json','worker-status.json','submission_receipt.json','submission.csv','predictions.csv','provenance.json','invocation.json','runtime_sha256.json')}
        save('measurement',stamp(state=attempt['state'],binding_sha256=reg['binding_sha256'],files=files,
            attempt_sha256=sha(WORK/'attempt.json'),review_sha256=sha(WORK/'gpu-review.json')))
        return stamp(state=attempt['state'])
    except BaseException as exc:
        attempt.update(state='HOLD',reason=f'{type(exc).__name__}: {exc}'); save('attempt',attempt); raise


def evaluate():
    import numpy as np
    from tools.job_runtime import job_status,LOGS
    c.cpu_only(); reg=review_ready()
    need(job_status(LOGS,TAG,expected_command=launch_command(),tail=False)['state']=='SUCCESS','successful exclusive job required')
    measured=io.read_json(WORK/'measurement.json'); attempt=io.read_json(WORK/'attempt.json')
    need(measured['state']==attempt['state']=='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW'
        and measured['binding_sha256']==attempt['binding_sha256']==reg['binding_sha256']
        and measured['attempt_sha256']==sha(WORK/'attempt.json') and measured['review_sha256']==sha(WORK/'gpu-review.json'),'measurement seal drift')
    for name,h in measured['files'].items():
        path=(WORK/name).resolve(); need(path.is_relative_to(WORK) and sha(path)==h,'measured artifact drift')
    value=attempt['value']; check_measurement(value,reg,value['desktop'],value['telemetry'])
    sup=io.read_json(WORK/'gpu/supervision.json')
    need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[],'supervision cleanup failure')
    actual,_=check_controller(WORK/'gpu/output',reg['binding']['ids'],'cuda')
    need(np.array_equal(actual,value['predictions']),'measurement predictions drift')
    old,common,failed,arms=c.context(); ids,labels=d.gold.gold(); need(ids==reg['binding']['ids'],'label roster drift')
    baseline=cached_r6(reg)
    stats=paired_metrics(labels,outputs(ids,common,failed,arms['flip-1']['predictions'],actual,baseline))
    current(); c.cpu_only()
    return save('evaluation',stamp(state='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW',binding_sha256=reg['binding_sha256'],
        measurement_sha256=sha(WORK/'measurement.json'),result=stats,decision=HOLD,qualified_for_submission=False))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['register','prepare','smoke','replay','validate','launch-check','run','_worker','evaluate'])
    args=ap.parse_args(argv)
    try:
        if args.action=='_worker': return worker()
        value=globals()[args.action.replace('-','_')]()
        print(json.dumps({k:value[k] for k in ('identity','version','state')})); return 0
    except (OSError,ValueError,ImportError,KeyError) as exc:
        print(json.dumps(stamp(state='HOLD',action=args.action,reason=str(exc)))); return 2


if __name__=='__main__': raise SystemExit(main())
