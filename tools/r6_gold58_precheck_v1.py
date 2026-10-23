"""R6/R7 Gold58 pre-check v1.0.1. New extension; frozen predecessors unchanged.

Changelog 1.0.1: CPU diagnostic explicitly eager; torch.compile initialized
CUDA even with empty visibility. Failed 1.0.0 evidence and source preserved.
Changelog 1.0.0: pinned R6 plain/flip contracts, exposed three-comparison
registration, real CPU smoke and one B-only supervised serial GPU pair.
No admission command; a separate bound independent review is required by run.
Public notebook is parsed as literal data, never executed.
"""
from __future__ import annotations
import argparse
import ast
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
from tools import c85r7_gold58_candidates as c
from tools import gpu_queue

d, t = c.d, c.t
need, sha, digest, io = c.need, c.sha, c.digest, c.io
VERSION = '1.0.1'
BUILDER = 'Codex/r6-gold58-build'
TAG = 'R6_R7_GOLD58_V101'
SCRIPT = Path(__file__).resolve()
WORK = ROOT/'.local/scratch/r6-gold58-build/r6-gold58'
BUNDLE = ROOT/'storage/external/heliosli-r6-fe055406-v1'
MANIFEST_SHA = '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648'
NOTEBOOK = ROOT/'.local/scratch/slot-b/ext-1016/picond9576/rsna-knee-9576-r6-v1-fixed-readable.ipynb'
STAGES = [d.PARENT.parent / x for x in ('v1.1.30-r224r6fq','v1.1.31-r224r76fq')]
SPEC_SHA = '8c0b60971d45e5a1a6d483a9985ad7202772c20a9ba0a7d9d9082eb744c8950e'
ARMS = ('plain', 'flip')
PAIRS = [('R6flip', 'R7flip'), ('replacement', 'HB167'), ('stack', 'HB167')]
RULES = dict(seed=174175, draws=2000, max_attempts=200000, quantiles=[.025,.975],
             quantile_method='linear', macro_floor=-.001, weak_four_floor=-.002,
             weak_four=[3,5,6,8], ranks='full58 once before paired study bootstrap',
             reject='any single-class target', se_ddof=1, mde_multiplier=2.8)
HOLD = 'HOLD_PENDING_INDEPENDENT_RESULT_REVIEW_AND_NON_GOLD_EVIDENCE'


def stamp(**kw):
    return dict(identity=TAG, version=VERSION, exposure='EXPOSED', clean_oof=False,
                selection_exposed=True, **kw)


def save(name, value):
    io.journal(WORK/(name+'.json'), value, WORK)
    return value


def once(raw, old, new):
    need(raw.count(old)==1, 'contract patch anchor mismatch')
    return raw.replace(old,new)


def contracts():
    files=d.stage_files(d.PARENT)
    need((ROOT/'tools/exp189_s6_port.py').read_bytes()==files['exp189_s6_port.py'],'local shared serving adapter drift')
    original=files['exp169_r7_port.py']
    release = [d.stage_files(stage) for stage in STAGES]
    flip=release[0]['exp169_r7_port.py']
    plain=once(flip,b'    args += ["--flip-tta"]\n',b'    if manifest.get("flip_tta"):\n        args += ["--flip-tta"]\n')
    # H175 uses the same math with an explicit bundle selector; prove the
    # actual worker/source and diagnostic adapters match, ignoring lookup only.
    def function(raw,name):
        text=raw.decode(); node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.dump(node,include_attributes=False)
    for name in ('worker_args','diagnostic_source','prepare_bundle'):
        need(function(flip,name)==function(release[1]['exp169_r7_port.py'],name),'H174/H175 serving function drift: '+name)
    for name in files:
        if name.startswith('exp169_r7_src/') or name=='exp189_s6_port.py':
            need(release[0][name]==release[1][name]==files[name],'release serving dependency drift: '+name)
    # Only notebook literals are evaluated. No notebook Python is compiled/run.
    nb=io.read_json(NOTEBOOK); source=nb['cells'][9]['source']
    source=''.join(source) if isinstance(source,list) else source
    literal={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('SOURCES','SOURCE_SHA'):
                    literal[target.id]=ast.literal_eval(node.value)
    need(len(literal['SOURCES'])==7 and set(literal['SOURCES'])==set(literal['SOURCE_SHA']), '9576 source roster drift')
    for name,code in literal['SOURCES'].items():
        raw=code.encode()
        need(files['exp169_r7_src/'+name]==raw and d.pack.digest(raw)==literal['SOURCE_SHA'][name],
             '9576 source differs from embedded parent: '+name)
    need("manifest.get('flip_tta') or FLIP_TTA" in source, '9576 forced flip branch missing')
    return files, {'plain':plain,'flip':flip}


def write_owned(path, raw):
    path=Path(path)
    need(path.resolve().is_relative_to(WORK.resolve()), 'write outside fixed scratch')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): need(path.read_bytes()==raw, 'prepared bytes drift: '+path.name)
    else: path.write_bytes(raw)


def export_contracts():
    files,ports=contracts()
    for arm,raw in ports.items():
        for name,content in files.items():
            write_owned(WORK/'contracts'/arm/name, raw if name=='exp169_r7_port.py' else content)
        diff=''.join(difflib.unified_diff(files['exp169_r7_port.py'].decode().splitlines(True),
            raw.decode().splitlines(True),fromfile='r224q/exp169_r7_port.py',tofile=arm+'/exp169_r7_port.py'))
        write_owned(WORK/('contract-'+arm+'.diff'),diff.encode())
    return files,ports


def port_module(arm):
    need(arm in ARMS, 'unknown arm')
    runtime=WORK/'contracts'/arm
    return d.import_path('r6_gold58_port_'+arm,runtime/'exp169_r7_port.py')


def launch_command():
    return [str(t.GPU_PYTHON),'-B',str(SCRIPT),'run']


def register():
    c.cpu_only()
    need(sha(WORK/'SPEC.md')==SPEC_SHA, 'pre-reviewed specification drift')
    old,common,failed,arms=c.context()
    need(all(not mask.any() for mask in failed.values()), 'common failures prohibited')
    files,ports=export_contracts()
    port=port_module('plain')
    manifest,identity=port.verify_bundle(BUNDLE,MANIFEST_SHA,WORK/'contracts/plain/exp169_r7_src')
    need(manifest['tag']=='cnx_r6fm_c352' and manifest['arm']=='E_lrprior', 'R6 recipe identity drift')
    inputs=[SCRIPT,ROOT/'tests/test_r6_gold58_precheck_v1.py',WORK/'SPEC.md',WORK/'PRE-REVIEW.md',
            NOTEBOOK,d.PARENT/'kernel/release.ipynb',d.gold.GOLD_DICOM/'manifest.json',
            d.gold.GOLD_DICOM/'train_series.csv',d.gold.RUNTIME/'prepare.json',
            ROOT/'storage/competition/train.csv',WORK/'SERVING-ADDENDUM.md',WORK/'ADDENDUM-REVIEW.md',
            WORK/'FIX-SPEC.md',WORK/'FIX-REVIEW.md']
    inputs += [s/'kernel/release.ipynb' for s in STAGES]
    inputs += [ROOT/'tools'/n for n in ('gpu_job.py','job_runtime.py')]
    need(all(p.is_file() for p in inputs),'required registration input missing')
    inputs += [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
               and Path(m.__file__).resolve().is_relative_to(ROOT/'tools')]
    inputs += list(BUNDLE.glob('*'))
    inputs += [p for p in (WORK/'contracts').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    binding=dict(builder=BUILDER,rules=RULES,comparisons=[list(x) for x in PAIRS],
        decision=HOLD,ids=old['binding']['ids'],targets=list(d.gold.LABELS),
        historical_binding_sha256=old['binding_sha256'],preserved_evidence_sha256=c.PINS_SHA,
        historical_device_uuid=io.read_json(c.SOURCE/'plain-1/supervision.json')['desktop_before_cuda']['uuid'],
        common=old['binding']['common'],r6_identity=identity,
        embedded={n:d.pack.digest(b) for n,b in files.items()},
        contracts={a:d.pack.digest(b) for a,b in ports.items()},
        command=launch_command(),inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(inputs)) if p.is_file()},
        historical_pair_seconds={a:arms[a+'-1']['seconds'] for a in ARMS},
        differences=['Exact H174 embedded flip port; plain differs only by flip condition; H175 worker_args/diagnostic_source/prepare_bundle AST-identical; all seven workers byte-identical to 9576',
          'Local diagnostic CPU empty CUDA envelope and explicit eager-only override; GPU measurement observers only; exported contracts unchanged',
          'GPU amp=auto matches preserved HB167, original HB128 explicit fp16 differs',
          'EXPOSED selection panel; no release-stage or Linux parity claim'])
    value=stamp(state='REGISTERED',binding=binding,binding_sha256=digest(binding))
    if (WORK/'registration.json').exists():
        need(io.read_json(WORK/'registration.json')==value,'immutable registration drift; new version required')
    save('registration',value)
    save('gpu-plan',gpu_queue.validate_plan([dict(tag=TAG,command=launch_command())]))
    return value


def current():
    reg=io.read_json(WORK/'registration.json'); b=reg['binding']
    need(reg['state']=='REGISTERED' and reg['binding_sha256']==digest(b) and b['builder']==BUILDER
         and b['rules']==RULES and b['comparisons']==[list(x) for x in PAIRS] and b['decision']==HOLD
         and b['command']==launch_command(), 'registration policy drift')
    for path,h in b['inputs'].items():
        p=(ROOT/path).resolve()
        need(p.is_relative_to(ROOT) and sha(p)==h,'registered input drift: '+path)
    files,ports=contracts()
    need({n:d.pack.digest(v) for n,v in files.items()}==b['embedded'] and
         {n:d.pack.digest(v) for n,v in ports.items()}==b['contracts'], 'embedded contract drift')
    need(sha(c.PINS)==c.PINS_SHA==b['preserved_evidence_sha256'],'preserved seal drift')
    return reg


def prepare():
    c.cpu_only(); reg=current()
    # Existing HB128 runtime is strictly read-only and must already be complete.
    record=io.read_json(d.gold.RUNTIME/'prepare.json')
    deps=Path(record['deps'])
    need(d.gold.dir_tree(deps)['tree_sha256']==record['deps_tree_sha256'],'dependency tree drift')
    need(d.gold.probe_deps(deps,d.CODE)==record['versions'],'pinned dependency probe drift')
    from tools import exp189_s6_port as s6
    code=s6.verify_code(d.CODE,d.ANATOMY)
    data=d.gold.verify_dicom(d.gold.GOLD_DICOM,reg['binding']['ids'])
    need(data['manifest_sha256']==reg['binding']['inputs'][str((d.gold.GOLD_DICOM/'manifest.json').relative_to(ROOT))],
         'historical data manifest drift')
    return save('prepare',stamp(state='CPU_PREPARED',binding_sha256=reg['binding_sha256'],
        data=data,deps_tree_sha256=record['deps_tree_sha256'],versions=record['versions'],code=code))


def prepare_runtime_check():
    record=io.read_json(d.gold.RUNTIME/'prepare.json')
    need(d.gold.dir_tree(Path(record['deps']))['tree_sha256']==record['deps_tree_sha256'],'dependency tree drift')
    from tools import exp189_s6_port as s6
    need(s6.verify_code(d.CODE,d.ANATOMY)==io.read_json(WORK/'prepare.json')['code'],'prepared code drift')


def cpu_diagnostic_source(port, source, out):
    """Local envelope only: propagate present-empty to controller, worker, loaders."""
    runtime=port.diagnostic_source(source,out)
    path=runtime/'probe_dual.py'
    raw=once(path.read_bytes(),b'names, visible = ["CPU"], ["-1"]',b'names, visible = ["CPU"], [""]')
    path.write_bytes(raw)
    worker=runtime/'compact_infer.py'
    raw=once(worker.read_bytes(),b'    dev = torch.device(a.device)',
        b'    if a.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":\n'
        b'        raise RuntimeError("CPU CUDA visibility must be present-empty")\n'
        b'    cpu_compile_requested = bool(a.compile) if a.device == "cpu" else None\n'
        b'    if a.device == "cpu":\n'
        b'        a.compile = False  # diagnostic only: compiler can initialize CUDA on CPU\n'
        b'    dev = torch.device(a.device)')
    raw=once(raw,b'"cuda_initialized": torch.cuda.is_initialized(),',
             b'"cuda_initialized": torch.cuda.is_initialized(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu_compile_requested": cpu_compile_requested,')
    worker.write_bytes(raw)
    (runtime/'runtime_sha256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(runtime.glob('*.py'))}),encoding='utf-8')
    return runtime


@contextlib.contextmanager
def execution_port(arm, device):
    port=port_module(arm); original=port.diagnostic_source; runner=port.s6._run_controller
    if device=='cpu':
        proxy=types.SimpleNamespace(diagnostic_source=original)
        port.diagnostic_source=lambda source,out:cpu_diagnostic_source(proxy,source,out)
        def empty_runner(args,env):
            c.cpu_only()
            return runner(args,dict(env,CUDA_VISIBLE_DEVICES=''))
        port.s6._run_controller=empty_runner
    else:
        def observed(source,out):
            runtime=original(source,out); target=runtime/'compact_infer.py'
            target.write_text(d.instrument(target.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
            return runtime
        port.diagnostic_source=observed
    try: yield port
    finally: port.s6._run_controller=runner


def check_controller(out, ids, arm, device):
    import numpy as np
    from tools import exp189_s6_port as s6
    proof=io.read_json(out/'controller/submission_receipt.json')
    need(proof['status']=='PASS' and proof['rows']==proof['scored']==len(ids) and
         proof['fallback_studies']==0 and proof['fallback_reasons']=={} and
         proof['worker_exit_codes'] and all(v==0 for v in proof['worker_exit_codes'].values()),'controller incomplete/fallback')
    need(len(proof['workers'])==1, 'single local device worker required')
    worker=proof['workers'][0]; manifest=io.read_json(BUNDLE/'manifest.json')
    need(worker['models']==manifest['models'] and worker['group2'] is None and worker['backbone']=='convnext_small'
         and worker['crop']==352 and worker['views']=='v4' and worker['phys_mm']==3.5
         and bool(worker.get('flip_tta'))==(arm=='flip') and worker['device']==device
         and worker['auxiliary_device']==device and worker['studies']==worker['scored']==len(ids)
         and worker['fallback']=={} and worker['timm']=='1.0.25','worker serving recipe drift')
    for key in ('fluid_stacker','state_stacker'):
        need(worker[key]['sha256']==manifest['sha256'][manifest[key]],'stacker identity drift')
    need(worker['side_check']['model_sha256']==manifest['sha256']['side_cls.pt'] and
         worker['anchor_weights_sha256']==manifest['anchor_weights_sha256'],'auxiliary model drift')
    if device=='cpu':
        need(worker['amp']=='fp32' and worker['cuda_initialized'] is False and
             worker['cuda_visible_devices']=='','CPU device/env drift')
        need(worker['cpu_compile_requested'] is True and worker['compile']['requested'] is False
             and worker['compile']['active'] is False,'CPU eager diagnostic drift')
    else: need(worker['amp']=='bf16' and worker['torch']=='2.5.1+cu121','preserved 4070 precision/software drift')
    invocation=io.read_json(out/'invocation.json'); args=invocation['args']
    wa=json.loads(args[args.index('--worker-args')+1])
    need(('--flip-tta' in wa)==(arm=='flip') and args[args.index('--loader-workers')+1]=='2'
         and args[args.index('--max-fallback-fraction')+1]=='0.0','invocation drift')
    provenance=io.read_json(out/'provenance.json')
    need(provenance['manifest_sha256']==MANIFEST_SHA,'output manifest drift')
    values=[]
    for name in ('predictions.csv','controller/submission.csv'):
        rows=s6.read_predictions(out/name,ids)
        need([r[d.gold.UID] for r in rows]==ids,'prediction ID order drift')
        values.append(c.check_values([[r[k] for k in d.gold.LABELS] for r in rows],len(ids)))
    need(np.array_equal(*values),'prediction copies disagree')
    return values[0],worker


def infer(arm,ids,device,out):
    with execution_port(arm,device) as port, d.gold.worker_pythonpath(d.gold.RUNTIME/'deps'):
        port.infer_studies([d.gold.GOLD_DICOM/'train_series'/uid for uid in ids],bundle_dir=BUNDLE,
            code_dir=d.CODE,anatomy_weights=d.ANATOMY,output_dir=out,
            source_dir=WORK/'contracts'/arm/'exp169_r7_src',workers=2,amp='auto',device=device,
            max_fallback_fraction=0.,expected_manifest_sha256=MANIFEST_SHA)
    return check_controller(out,ids,arm,device)


def smoke():
    c.cpu_only(); reg=current(); pre=io.read_json(WORK/'prepare.json')
    need(pre['binding_sha256']==reg['binding_sha256'],'fresh prepare required')
    ids=reg['binding']['ids'][:1]; results={}; start=time.perf_counter()
    for arm in ARMS:
        out=WORK/'cpu-smoke'/arm
        tick=time.perf_counter()
        values,worker=infer(arm,ids,'cpu',out)
        results[arm]=dict(rows=len(values),models=worker['models'],amp=worker['amp'],
            cuda_initialized=worker['cuda_initialized'],cuda_visible_devices=worker['cuda_visible_devices'],
            compile_requested_by_serving=worker['cpu_compile_requested'],compile_active=worker['compile']['active'],
            flip=bool(worker.get('flip_tta')),seconds=time.perf_counter()-tick,
            receipt_sha256=sha(out/'controller/submission_receipt.json'),predictions_sha256=sha(out/'predictions.csv'))
    c.cpu_only(); current()
    return save('cpu-smoke',stamp(state='PASS_REAL_ONE_STUDY_BOTH_ARMS',binding_sha256=reg['binding_sha256'],
        study_count=1,results=results,seconds=time.perf_counter()-start,cuda_initialized=False,
        limitation='CPU engineering only; no CUDA, Linux or R6 Gold58 performance result'))


def rank(values):
    import pandas as pd
    return pd.DataFrame(values).rank(method='average',pct=True).to_numpy()


def outputs(ids,common,failed,r7,r6):
    import numpy as np
    need(len(ids)==len(set(ids))==58 and set(common)==set(failed)==set(d.COMMON),'cohort/member drift')
    for name in d.COMMON:
        c.check_values(common[name])
        mask=np.asarray(failed[name])
        need(mask.shape==(58,) and mask.dtype==np.bool_ and not mask.any(),'common failures prohibited')
    need(not any(np.all(common[n]==.5,axis=1).any() for n in ('R','R224')),'R failure sentinel')
    r7,r6=c.check_values(r7),c.check_values(r6)
    rr=.65*rank(common['R'])+.35*rank(common['R224'])
    own=rank((17*rank(common['EXP143v2'])+17*rank(common['EXP155'])+12*rank(common['Q_A5q']))/46)
    def final(slot): return .85*rank(.5*rr+.5*slot)+.15*own
    values=dict(R7flip=r7,R6flip=r6,HB167=final(rank(r7)),replacement=final(rank(r6)),
                stack=final(.5*rank(r7)+.5*rank(r6)))
    oracle=d.formula.blend(ids,common['R'],common['R224'],r7,common['EXP143v2'],common['EXP155'],common['Q_A5q'])
    need(np.array_equal(values['HB167'],oracle),'embedded parent formula disagreement')
    return values


def bootstrap_auc(labels,values,rules=RULES):
    import numpy as np
    y=np.asarray(labels)
    need(y.shape==(58,12) and np.isin(y,[0,1]).all() and
         all(len(np.unique(y[:,j]))==2 for j in range(12)),'binary nondegenerate Gold58 required')
    values=np.stack([c.check_values(v) for v in values])
    rng=np.random.default_rng(rules['seed']); weights=[]; attempts=0
    while len(weights)<rules['draws']:
        need(attempts<rules['max_attempts'],'bootstrap degeneracy limit')
        w=np.bincount(rng.integers(0,58,58),minlength=58); attempts+=1
        pos=w@y
        if np.any((pos==0)|(pos==58)): continue
        weights.append(w)
    weights=np.asarray(weights,float); boot=np.empty((len(values),len(weights),12))
    for j in range(12):
        pos=y[:,j]==1; wp=weights[:,pos]; wn=weights[:,~pos]
        for k,v in enumerate(values):
            a=v[pos,j,None]; b=v[~pos,j][None,:]
            pairs=(a>b).astype(float)+.5*(a==b)
            boot[k,:,j]=np.sum((wp@pairs)*wn,axis=1)/(wp.sum(1)*wn.sum(1))
    return boot,weights,attempts


def metric(point,draws):
    import numpy as np
    lo,hi=np.quantile(draws,RULES['quantiles'],method='linear')
    se=float(np.std(draws,ddof=1))
    return dict(delta=float(point),ci95=[float(lo),float(hi)],se=se,mde=2.8*se,
        bootstrap_positive_fraction=float(np.mean(draws>0)),
        statistical='GAIN' if lo>0 else 'HARM' if hi<0 else 'UNRESOLVED')


def paired_metrics(labels,values):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    names=list(values); arrays=[c.check_values(values[n]) for n in names]
    boot,weights,attempts=bootstrap_auc(labels,arrays)
    auc=np.array([[roc_auc_score(labels[:,j],v[:,j]) for j in range(12)] for v in arrays])
    results={}
    for a,b in PAIRS:
        delta=auc[names.index(a)]-auc[names.index(b)]
        draws=boot[names.index(a)]-boot[names.index(b)]
        macro=metric(delta.mean(),draws.mean(1)); weak=metric(delta[RULES['weak_four']].mean(),draws[:,RULES['weak_four']].mean(1))
        results[a+'-minus-'+b]=dict(scope='standalone proxy' if a=='R6flip' else 'exact final blend',
            macro12=macro,weak_four=weak,per_target_delta=delta.tolist(),
            operational='PASS' if macro['delta']>=RULES['macro_floor'] and weak['delta']>=RULES['weak_four_floor'] else 'FAIL',
            exposure='EXPOSED',clean_oof=False,selection_exposed=True)
    return dict(comparisons=results,per_target_auc={n:auc[i].tolist() for i,n in enumerate(names)},
        bootstrap=dict(rules=RULES,valid_draws=len(weights),attempts=attempts,rejected=attempts-len(weights)),
        decision=HOLD,qualified_for_submission=False,
        training_exclusion='R6/R7 author-attributed gold_in_training=0; unaudited; R7 selected using Gold58 D-B-163')


def replay():
    c.cpu_only(); reg=current(); old,common,failed,arms=c.context()
    ids,labels=d.gold.gold(); need(ids==reg['binding']['ids'],'label roster drift')
    # Identity control exercises real full58 postprocessing, never reports R6 scores.
    r7=arms['flip-1']['predictions']; values=outputs(ids,common,failed,r7,r7)
    stats=paired_metrics(labels,values)
    need(all(v['macro12']['ci95']==v['weak_four']['ci95']==[0.,0.] for v in stats['comparisons'].values()),'identity control nonzero')
    return save('replay',stamp(state='PASS_REAL_IDENTITY_CONTROL',binding_sha256=reg['binding_sha256'],
        comparisons_exercised=[list(p) for p in PAIRS],valid_draws=stats['bootstrap']['valid_draws'],
        r6_measured=False,limitation='Preserved R7 substituted as R6 only for equality control; no candidate numbers'))


def validate():
    c.cpu_only(); reg=current(); evidence={}
    for name,state in [('prepare','CPU_PREPARED'),('cpu-smoke','PASS_REAL_ONE_STUDY_BOTH_ARMS'),('replay','PASS_REAL_IDENTITY_CONTROL')]:
        value=io.read_json(WORK/(name+'.json'))
        need(value['state']==state and value['binding_sha256']==reg['binding_sha256'],'fresh '+name+' required')
        evidence[name]=sha(WORK/(name+'.json'))
    return save('validate',stamp(state='CPU_VALIDATED',binding_sha256=reg['binding_sha256'],evidence=evidence))


def check_review(review,reg,validation_sha):
    need(review.get('verdict')=='PASS' and review.get('builder')==BUILDER and
         isinstance(review.get('reviewer'),str) and review['reviewer'].strip() and review['reviewer']!=BUILDER
         and review.get('execution_authority')=='B' and review.get('tag')==TAG
         and review.get('binding_sha256')==reg['binding_sha256'] and review.get('validation_sha256')==validation_sha
         and review.get('approved_differences')==reg['binding']['differences'],'independent bound PASS required')


def review_ready():
    reg=current(); valid=io.read_json(WORK/'validate.json')
    need(valid['state']=='CPU_VALIDATED' and valid['binding_sha256']==reg['binding_sha256'],'validation drift')
    for name,h in valid['evidence'].items(): need(sha(WORK/(name+'.json'))==h,'CPU evidence drift')
    smoke=io.read_json(WORK/'cpu-smoke.json')
    for arm in ARMS:
        for key,path in [('receipt_sha256','controller/submission_receipt.json'),('predictions_sha256','predictions.csv')]:
            need(sha(WORK/'cpu-smoke'/arm/path)==smoke['results'][arm][key],'CPU artifact drift')
    check_review(io.read_json(WORK/'gpu-review.json'),reg,sha(WORK/'validate.json'))
    return reg


def launch_check():
    c.cpu_only(); reg=review_ready()
    return stamp(state='REVIEW_BOUND_LAUNCH_CHECK_PASS_NO_ADMISSION',binding_sha256=reg['binding_sha256'])


def configure(arm):
    need(arm in ARMS,'unknown GPU arm')
    t.WORK=WORK/'gpu'/arm; t.TAG=TAG; t.__file__=str(SCRIPT)
    t.monitor=types.SimpleNamespace(**dict(vars(d._MONITOR),_cleanup=d.cleanup_owned))
    return t.WORK


def lane():
    desktop=t.lane_admission(); reg=current()
    need(desktop['uuid']==reg['binding']['historical_device_uuid'],'same physical 4070 as R7 required')
    return desktop


def runtime_check(value,arm):
    r=value['runtime']
    need(r['models']==[f'E_lrprior_s0_f{i}.pt' for i in range(4)] and r['models2']==[] and
         r['flip_tta']==(arm=='flip') and r['workers']==2 and r['size']==384 and r['crop']==352
         and r['views']=='v4' and r['slices']==3,'measured R6 runtime drift')


def worker():
    need(os.environ.get('HB164_CONTROLLER_PID')==str(os.getppid()),'supervised worker only')
    reg=review_ready(); arm=os.environ['R6_GOLD58_ARM']; work=configure(arm); lane()
    os.environ['HB153_MEASURE_STATUS']=str(work/'worker-status.json')
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT),os.environ.get('PYTHONPATH','')])
    start=time.perf_counter(); ids=reg['binding']['ids']
    values,receipt=infer(arm,ids,'cuda',work/'output')
    measured=io.read_json(work/'worker-status.json'); runtime_check(measured,arm)
    value=stamp(state='COMPLETE',binding_sha256=reg['binding_sha256'],done=len(ids),
        ids=ids,predictions=values.tolist(),rows=measured['rows'],runtime=measured['runtime'],
        device=measured['device'],seconds=time.perf_counter()-start,torch=receipt['torch'])
    io.journal(work/'worker-status.json',value,work)
    return 0


def check_measurement(value,reg,arm,desktop,telemetry):
    ids=reg['binding']['ids']; runtime_check(value,arm)
    need(value['state']=='COMPLETE' and value['identity']==TAG and value['version']==VERSION
         and value['binding_sha256']==reg['binding_sha256'] and value['done']==58 and value['ids']==ids
         and [r['study'] for r in value['rows']]==ids and telemetry['complete']
         and telemetry['memory_failure'] is None,'complete ordered measurement required')
    c.check_values(value['predictions'])
    for row in value['rows']:
        d.measure.window_gate(row['windows'],row['batch_windows'])
        need(math.isfinite(row['gpu_seconds']) and row['gpu_seconds']>0 and
             0<row['peak_allocated_bytes']<=row['peak_reserved_bytes'],'invalid time/memory')
    d.r7.memory_admission(desktop,[r['peak_reserved_bytes'] for r in value['rows']],telemetry['peak_gpu_mib']*2**20)


def run():
    reg=review_ready()
    prepare_runtime_check()
    need(not (WORK/'attempt.json').exists(),'previous attempt preserved; no automatic retry')
    need(d.gold.verify_dicom(d.gold.GOLD_DICOM,reg['binding']['ids'])==io.read_json(WORK/'prepare.json')['data'],'DICOM drift')
    attempt=stamp(state='RUNNING',binding_sha256=reg['binding_sha256'],arms={})
    save('attempt',attempt)
    try:
        for arm in ARMS:
            work=configure(arm); desktop=lane()
            need(not work.exists(),'previous GPU output preserved')
            work.mkdir(parents=True)
            os.environ['R6_GOLD58_ARM']=arm
            value,telemetry=t.supervise(desktop)
            value=d.canonicalize_rows(value,reg['binding']['ids'])
            check_measurement(value,reg,arm,desktop,telemetry)
            value.update(desktop=desktop,telemetry=telemetry)
            attempt['arms'][arm]=value
            save('attempt',attempt)
        need([(r['study'],r['windows'],r['batch_windows']) for r in attempt['arms']['plain']['rows']]==
             [(r['study'],r['windows'],r['batch_windows']) for r in attempt['arms']['flip']['rows']], 'R6 paired windows drift')
        attempt['state']='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW'
        save('attempt',attempt)
        artifacts={str(p.relative_to(WORK)):sha(p) for p in (WORK/'gpu').rglob('*')
            if p.is_file() and p.name in ('supervision.json','worker-status.json','submission_receipt.json','submission.csv','predictions.csv','provenance.json','invocation.json')}
        save('measurement',stamp(state=attempt['state'],binding_sha256=reg['binding_sha256'],
            attempt_sha256=sha(WORK/'attempt.json'),files=artifacts,review_sha256=sha(WORK/'gpu-review.json')))
        return attempt
    except BaseException as exc:
        attempt.update(state='HOLD',reason=f'{type(exc).__name__}: {exc}')
        save('attempt',attempt); raise


def evaluate():
    import numpy as np
    from tools.job_runtime import job_status,LOGS
    c.cpu_only(); reg=review_ready()
    need(job_status(LOGS,TAG,expected_command=launch_command(),tail=False)['state']=='SUCCESS','successful exclusive job required')
    measured=io.read_json(WORK/'measurement.json'); attempt=io.read_json(WORK/'attempt.json')
    need(set(attempt['arms'])==set(ARMS),'complete plain/flip pair required')
    need(measured['state']==attempt['state']=='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW' and
         measured['binding_sha256']==attempt['binding_sha256']==reg['binding_sha256'] and
         measured['attempt_sha256']==sha(WORK/'attempt.json') and measured['review_sha256']==sha(WORK/'gpu-review.json'),'measurement seal drift')
    for name,h in measured['files'].items():
        path=(WORK/name).resolve(); need(path.is_relative_to(WORK) and sha(path)==h,'measured artifact drift')
    for arm,value in attempt['arms'].items():
        check_measurement(value,reg,arm,value['desktop'],value['telemetry'])
        sup=io.read_json(WORK/'gpu'/arm/'supervision.json')
        need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[],'supervision/cleanup failure')
        actual,_=check_controller(WORK/'gpu'/arm/'output',reg['binding']['ids'],arm,'cuda')
        need(np.array_equal(actual,value['predictions']),'measured predictions drift')
    old,common,failed,arms=c.context(); ids,labels=d.gold.gold()
    need(ids==reg['binding']['ids'],'label cohort drift')
    values=outputs(ids,common,failed,arms['flip-1']['predictions'],attempt['arms']['flip']['predictions'])
    stats=paired_metrics(labels,values); current(); c.cpu_only()
    return save('evaluation',stamp(state='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW',
        binding_sha256=reg['binding_sha256'],measurement_sha256=sha(WORK/'measurement.json'),
        result=stats,decision=HOLD,qualified_for_submission=False))


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
