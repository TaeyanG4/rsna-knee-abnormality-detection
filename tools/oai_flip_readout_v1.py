"""Q28 OAI flip ablation v1.0.1; 2026-10-16: include CPU oracle metadata alias.

v1.0.0: authenticated flip reuse + plain557. Backup: .local/backups/q28-v100.

An isolated predecessor module reuses supervision without changing its files or
the imported public predecessor. All parameter substitutions are local to it.
"""
from __future__ import annotations
import argparse
import copy
import csv
import io as stringio
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import r6_oai_readout_v110 as predecessor
g=predecessor.g
d,t=g.d,g.t
need,sha,digest,io=g.need,g.sha,g.digest,g.io
e=d.import_path('q28_isolated_oai',Path(predecessor.__file__))
VERSION,TAG,BUILDER='1.0.1','Q28_OAI_FLIP_V101','q28-build'
SCRIPT=Path(__file__).resolve()
WORK=ROOT/'.local/scratch/q28-build/q28'
ARMS=('R7plain','R6plain')
ALL_ARMS=('R7plain','R7flip','R6plain','R6flip')
RULES=dict(predecessor.RULES,n=557,primary='within-model flip-minus-plain',
           simultaneous='Bonferroni97.5% marginal intervals for two flip effects per endpoint',
           interaction='R6 flip effect minus R7 flip effect')
MACRO7=predecessor.MACRO7
save,read,stamp=e.save,e.read,e.stamp
label_contract,bootstrap,values=e.label_contract,e.bootstrap,e.values
launch_command=e.launch_command


def extension_registration():
    r=io.read_json(predecessor.WORK/'registration.json')
    need(r['binding_sha256']==digest(r['binding']),'extension binding drift')
    need(r['binding']['ids']==predecessor.select_ids(e.raw.ids_only()),'extension roster drift')
    for p,h in r['binding']['sources'].items(): need(sha(ROOT/p)==h,'extension source drift: '+p)
    return r


def queue_plan():
    return [dict(tag=TAG,command=launch_command(),after=predecessor.TAG)]


def export_contracts():
    files=predecessor.contract_files(); proof={}
    for arm in ARMS:
        before=files[arm.replace('plain','flip')]
        after=dict(before)
        after['exp169_r7_port.py']=g.once(before['exp169_r7_port.py'],
            b'    args += ["--flip-tta"]\n',
            b'    if manifest.get("flip_tta"):\n        args += ["--flip-tta"]\n')
        proof[arm]={n:dict(before=d.pack.digest(v),after=d.pack.digest(after[n]),
                          changed=v!=after[n]) for n,v in before.items()}
        need([n for n in before if before[n]!=after[n]]==['exp169_r7_port.py'],'embedded diff exceeded flip flag')
        for n,v in after.items():
            if n in ('exp169_r7_port.py','exp189_s6_port.py') or n.startswith('exp169_r7_src/'):
                p=WORK/'contracts'/arm/n; p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists(): need(p.read_bytes()==v,'contract drift')
                else: p.write_bytes(v)
    return proof


def register():
    g.c.cpu_only(); need(not (WORK/'attempt.json').exists(),'attempt exists')
    embedded=export_contracts(); old=extension_registration(); b0=old['binding']
    identities={a:e.port_module(a).verify_bundle(e.BUNDLES[a],e.MANIFESTS[a],WORK/'contracts'/a/'exp169_r7_src')[1] for a in ARMS}
    need({a.replace('plain','flip'):v for a,v in identities.items()}==b0['identities'],'bundle identities differ')
    for a in ARMS: need(not io.read_json(e.BUNDLES[a]/'manifest.json').get('flip_tta'),'plain manifest forces flip')
    paths=[SCRIPT,ROOT/'tests/test_oai_flip_readout_v1.py',WORK/'SPEC.md',WORK/'PRE-REVIEW.md',
           Path(predecessor.__file__),predecessor.WORK/'registration.json',predecessor.WORK/'prepare.json',
           predecessor.PILOT/'prepare.json']
    sources=dict(b0['sources']); sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    for p in (WORK/'contracts').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts: sources[p.relative_to(ROOT).as_posix()]=sha(p)
    b=dict(builder=BUILDER,ids=b0['pooled_ids'],pilot_ids=b0['pilot_ids'],confirmation_ids=b0['ids'],
           pooled_ids=b0['pooled_ids'],smoke_id=b0['smoke_id'],targets=b0['targets'],rules=RULES,
           identities=identities,routes=b0['routes'],audit=b0['audit'],sources=sources,embedded=embedded,
           extension_binding_sha256=old['binding_sha256'],command=launch_command(),queue=queue_plan(),
           memory={a:b0['memory'][a.replace('plain','flip')] for a in ARMS},ancestry=b0['ancestry'])
    r=stamp(state='REGISTERED_BEFORE_LABELS_AND_PREDICTIONS',binding=b,binding_sha256=digest(b))
    if (WORK/'registration.json').exists(): need(read('registration')==r,'immutable registration differs')
    else: save('registration',r)
    save('queue',queue_plan()); save('embedded-diff',embedded)
    return stamp(state=r['state'],binding_sha256=r['binding_sha256'],studies=557)


def current():
    r=read('registration'); b=r['binding']
    need(digest(b)==r['binding_sha256'] and b['rules']==RULES,'registration drift')
    for p,h in b['sources'].items(): need(sha(ROOT/p)==h,'source drift: '+p)
    ext=extension_registration()
    need(ext['binding_sha256']==b['extension_binding_sha256'] and b['ids']==ext['binding']['pooled_ids'],'predecessor/roster drift')
    for a in ARMS:
        need(e.port_module(a).verify_bundle(e.BUNDLES[a],e.MANIFESTS[a],WORK/'contracts'/a/'exp169_r7_src')[1]==b['identities'][a],'bundle drift')
    return r


def prepare(dry_run=False):
    g.c.cpu_only(); r=current(); b=r['binding']; rows=[]; pins={}
    for module,ids in [(predecessor.pilot,b['pilot_ids']),(predecessor,b['confirmation_ids'])]:
        pre=io.read_json(module.WORK/'prepare.json'); old=io.read_json(module.WORK/'registration.json')
        need(pre['state']=='PASS' and pre['binding_sha256']==old['binding_sha256'],'predecessor prepare mismatch')
        for uid in ids:
            prefix='test_series/'+uid+'/'
            selected={p:h for p,h in pre['data'].items() if p.startswith(prefix)}
            need(selected and b['routes'][uid]==old['binding']['routes'][uid],'source route/input absent')
            for p,h in selected.items():
                source=module.WORK/'data'/p; need(sha(source)==h,'cached input byte drift')
                pins[p]=h
                if not dry_run:
                    target=WORK/'data'/p; target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists(): need(sha(target)==h,'staged bytes differ')
                    else: os.link(source,target)
            rows.extend({k:v for k,v in row.items() if k!='group_key'} for row in b['routes'][uid])
    if dry_run: return save('prepare-dry-run',stamp(state='PASS',binding_sha256=r['binding_sha256'],studies=557))
    s=stringio.StringIO(newline=''); w=csv.DictWriter(s,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (WORK/'data/test_series.csv').write_text(s.getvalue(),encoding='utf-8',newline='')
    need(e.panel.validate()==e.raw.ids_only(),'panel drift')
    labels={name:label_contract(e.panel.outcomes(b[key])) for name,key in cohorts()}
    if (WORK/'labels.json').exists(): need(read('labels')==labels,'labels drift')
    else: save('labels',labels)
    from tools import exp189_s6_port as s6
    dep=io.read_json(d.gold.RUNTIME/'prepare.json')
    need(d.gold.dir_tree(Path(dep['deps']))['tree_sha256']==dep['deps_tree_sha256'],'dependency drift')
    pins['test_series.csv']=sha(WORK/'data/test_series.csv')
    return save('prepare',stamp(state='PASS',binding_sha256=r['binding_sha256'],data=pins,
        labels_sha256=sha(WORK/'labels.json'),code=s6.verify_code(d.CODE,d.ANATOMY),deps_tree_sha256=dep['deps_tree_sha256']))


def infer(arm,ids,device,out):
    with e.serving(arm),g.execution_port('plain',device) as port,d.gold.worker_pythonpath(d.gold.RUNTIME/'deps'):
        original=port.diagnostic_source
        expected={u:sorted(r['SeriesInstanceUID'] for r in read('registration')['binding']['routes'][u]) for u in ids}
        def observed(source,destination):
            runtime=original(source,destination); target=runtime/'compact_infer.py'
            target.write_text(e.series_probe(target.read_text(encoding='utf-8'),expected,out/'series-proof'),encoding='utf-8',newline='\n')
            (runtime/'runtime_sha256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(runtime.glob('*.py'))}),encoding='utf-8')
            return runtime
        port.diagnostic_source=observed
        port.infer_studies([WORK/'data/test_series'/u for u in ids],bundle_dir=e.BUNDLES[arm],code_dir=d.CODE,
            anatomy_weights=d.ANATOMY,output_dir=out,source_dir=WORK/'contracts'/arm/'exp169_r7_src',
            workers=2,amp='auto',device=device,max_fallback_fraction=0.,expected_manifest_sha256=e.MANIFESTS[arm])
        result=g.check_controller(out,ids,'plain',device); e.verify_series(out,expected)
        return result


def smoke():
    import numpy as np
    g.c.cpu_only(); r=e.verify_prepared(); ids=[r['binding']['smoke_id']]; evidence={}
    # Direct Gold helper expects these two aliases, both local hardlinks only.
    source_csv=WORK/'data/test_series.csv'; oracle_csv=WORK/'data/train_series.csv'
    if oracle_csv.exists(): need(oracle_csv.read_bytes()==source_csv.read_bytes(),'oracle metadata differs')
    else: os.link(source_csv,oracle_csv)
    for p in (WORK/'data/test_series'/ids[0]).rglob('*.dcm'):
        target=WORK/'data/train_series'/p.relative_to(WORK/'data/test_series')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): os.link(p,target)
    for arm in ARMS:
        for p in (WORK/'contracts'/arm/'exp169_r7_src').glob('*.py'):
            target=WORK/'contracts/plain/exp169_r7_src'/p.name; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): need(target.read_bytes()==p.read_bytes(),'oracle source differs')
            else: os.link(p,target)
        start=time.perf_counter(); actual,worker=infer(arm,ids,'cpu',WORK/'cpu-smoke'/arm)
        with e.serving(arm): oracle,ow=g.infer('plain',ids,'cpu',WORK/'cpu-reference'/arm)
        need(np.array_equal(actual,oracle) and worker['torch']==ow['torch'],'CPU raw probability parity differs')
        evidence[arm]=dict(exact=True,max_abs_delta=float(np.max(abs(actual-oracle))),seconds=time.perf_counter()-start,
                           cuda_initialized=worker['cuda_initialized'],amp=worker['amp'])
    files={p.relative_to(WORK).as_posix():sha(p) for folder in ('cpu-smoke','cpu-reference') for p in (WORK/folder).rglob('*')
           if p.is_file() and (p.name in ('predictions.csv','submission_receipt.json','invocation.json','provenance.json') or p.parent.name=='series-proof')}
    return save('cpu-smoke',stamp(state='PASS_REAL_ONE_STUDY_PARITY',binding_sha256=r['binding_sha256'],studies=1,arms=evidence,files=files))


def cohorts(): return [('pilot','pilot_ids'),('confirmation','confirmation_ids'),('pooled','pooled_ids')]


def metrics(y,pred,contract,draws=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    need(set(pred)==set(ALL_ARMS),'four arms required'); need(label_contract(y)==contract,'label contract drift')
    arrays=[values(pred[k],len(y)) for k in ALL_ARMS]; eligible=contract['eligible']
    auc=np.array([[roc_auc_score(y[np.isfinite(y[:,j]),j],v[np.isfinite(y[:,j]),j]) for j in eligible] for v in arrays])
    primary=[j for j in MACRO7 if j in eligible]; pb,weights,attempts=bootstrap(y,arrays,primary,draws)
    boot=np.empty((4,len(weights),len(eligible)))
    for q,j in enumerate(eligible):
        boot[:,:,q]=pb[:,:,primary.index(j)] if j in primary else bootstrap(y,arrays,[j],draws)[0][:,:,0]
    def summary(point,rep,simultaneous):
        result=g.metric(point,rep)
        if simultaneous: result['ci97_5_family_two']=np.percentile(rep,[1.25,98.75],method='linear').tolist()
        return result
    result={}
    for name,coef,sim in [('R6flip-minus-R6plain',[0,0,-1,1],True),('R7flip-minus-R7plain',[-1,1,0,0],True),
                           ('R6-effect-minus-R7-effect',[1,-1,-1,1],False)]:
        delta=np.einsum('a,aj->j',coef,auc); bd=np.einsum('a,abj->bj',coef,boot)
        def group(indices):
            if not set(indices)<=set(eligible): return None
            ix=[eligible.index(j) for j in indices]; return summary(float(delta[ix].mean()),bd[:,ix].mean(1),sim)
        macro,weak=group(MACRO7),group(RULES['weak_four'])
        status='INCOMPLETE' if macro is None or weak is None else 'FAIL' if macro['delta']<RULES['macro_floor'] or weak['delta']<RULES['weak_four_floor'] else 'PASS'
        result[name]=dict(macro7=macro,weak_four=weak,point_floors=status if sim else 'NOT_APPLICABLE_INTERACTION',
            per_target={str(j):summary(float(delta[q]),bd[:,q],sim) for q,j in enumerate(eligible)})
    return dict(comparisons=result,per_target_auc={a:dict(zip(map(str,eligible),auc[k].tolist())) for k,a in enumerate(ALL_ARMS)},
                valid_draws=len(weights),attempts=attempts,eligible=eligible,macro_targets=MACRO7,
                decision='HOLD_PENDING_INDEPENDENT_RESULT_REVIEW',qualified_for_submission=False)


def replay():
    import numpy as np
    g.c.cpu_only(); r=current(); evidence={}
    for name,key in cohorts():
        y=e.panel.outcomes(r['binding'][key]); v=np.tile(np.linspace(.1,.9,len(y))[:,None],(1,12))
        result=metrics(y,{a:v for a in ALL_ARMS},read('labels')[name])
        for c in result['comparisons'].values(): need(c['macro7']['ci95']==[0.,0.],'identity control differs')
        evidence[name]=dict(draws=result['valid_draws'],studies=len(y))
    return save('replay',stamp(state='PASS_REAL_LABEL_IDENTITY_CONTROL',binding_sha256=r['binding_sha256'],cohorts=evidence,model_efficacy_measured=False))


def validate():
    g.c.cpu_only(); r=e.verify_prepared(); evidence={}
    for name,state in [('cpu-smoke','PASS_REAL_ONE_STUDY_PARITY'),('replay','PASS_REAL_LABEL_IDENTITY_CONTROL'),('prepare','PASS')]:
        value=read(name); need(value['state']==state and value['binding_sha256']==r['binding_sha256'],'stale '+name)
        evidence[name]=sha(WORK/(name+'.json'))
    for p,h in read('cpu-smoke')['files'].items(): need(sha(WORK/p)==h,'smoke drift')
    g.gpu_queue.validate_plan(read('queue')); need(read('queue')==queue_plan(),'queue drift')
    return save('validate',stamp(state='CPU_VALIDATED',binding_sha256=r['binding_sha256'],evidence=evidence))


def review_ready():
    r=current(); v=read('validate'); need(v['state']=='CPU_VALIDATED' and v['binding_sha256']==r['binding_sha256'],'validation drift')
    for p,h in v['evidence'].items(): need(sha(WORK/(p+'.json'))==h,'validation evidence drift')
    for p,h in read('cpu-smoke')['files'].items(): need(sha(WORK/p)==h,'CPU output drift')
    need(read('queue')==queue_plan(),'queue argv drift'); e.check_review(read('gpu-review'),r,sha(WORK/'validate.json'))
    return r


def check_measurement(v,reg,arm,desktop,telemetry):
    # The predecessor gate is identical apart from the registered TTA boolean.
    need(v['runtime']['flip_tta'] is False,'plain runtime unexpectedly flipped')
    adapted=copy.deepcopy(v); adapted['runtime']['flip_tta']=True
    predecessor.check_measurement(adapted,reg,arm,desktop,telemetry)


def authenticate_cache(module):
    """Never inspect an incomplete job's output; hash seals before prediction reads."""
    import numpy as np
    from tools.job_runtime import job_status,LOGS
    need(job_status(LOGS,module.TAG,expected_command=module.launch_command(),tail=False)['state']=='SUCCESS','cache predecessor incomplete')
    reg=io.read_json(module.WORK/'registration.json'); m=io.read_json(module.WORK/'measurement.json')
    need(m['state']=='COMPLETE_PENDING_RESULT_REVIEW' and m['binding_sha256']==reg['binding_sha256'],'cache measurement incomplete')
    need(m['attempt_sha256']==sha(module.WORK/'attempt.json') and m['review_sha256']==sha(module.WORK/'gpu-review.json'),'cache seal drift')
    module.check_review(io.read_json(module.WORK/'gpu-review.json'),reg,sha(module.WORK/'validate.json'))
    valid=io.read_json(module.WORK/'validate.json')
    need(valid['state']=='CPU_VALIDATED' and valid['binding_sha256']==reg['binding_sha256'],'cache validation binding')
    for p,h in valid['evidence'].items(): need(sha(module.WORK/(p+'.json'))==h,'cache validation evidence drift')
    need(m['files'],'empty cache manifest')
    for p,h in m['files'].items(): need(sha(module.WORK/p)==h,'cache output drift')
    aa=io.read_json(module.WORK/'attempt.json')
    need(aa['state']==m['state'] and aa['binding_sha256']==reg['binding_sha256'] and set(aa['arms'])==set(module.ARMS),'cache attempt incomplete')
    for arm,v in aa['arms'].items():
        module.check_measurement(v,reg,arm,v['desktop'],v['telemetry'])
        sup=io.read_json(module.WORK/'gpu'/arm/'supervision.json')
        need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[],'cache supervision failed')
        with module.serving(arm): actual,_=g.check_controller(module.WORK/'gpu'/arm/'output',reg['binding']['ids'],'flip','cuda')
        module.verify_series(module.WORK/'gpu'/arm/'output',{u:sorted(r['SeriesInstanceUID'] for r in reg['binding']['routes'][u]) for u in reg['binding']['ids']})
        need(np.array_equal(actual,v['predictions']),'cache predictions differ')
    return aa


def caches():
    r=current(); ext=extension_registration(); old=predecessor.pilot_registration()
    need(old['binding']['identities']==ext['binding']['identities']=={a.replace('plain','flip'):v for a,v in r['binding']['identities'].items()},'cache model drift')
    return authenticate_cache(predecessor.pilot),authenticate_cache(predecessor)


def check_paired_cache(plain,pilot,confirmation,reg):
    import numpy as np
    b=reg['binding']; pooled={}
    need(not set(b['pilot_ids'])&set(b['confirmation_ids']) and sorted(b['pilot_ids']+b['confirmation_ids'])==b['ids'],'cohort overlap')
    for arm in ARMS:
        flip=arm.replace('plain','flip'); p=plain['arms'][arm]; ordered={}; rows={}
        need(p['ids']==b['ids'],'plain order drift')
        for cache,key in [(pilot,'pilot_ids'),(confirmation,'confirmation_ids')]:
            c=cache['arms'][flip]; need(c['ids']==b[key],'cached cohort order drift')
            rt=dict(c['runtime']); need(rt.pop('flip_tta') is True,'cache not flip'); pr=dict(p['runtime']); need(pr.pop('flip_tta') is False,'new arm not plain')
            need(rt==pr and c['torch']==p['torch'] and c['desktop']['uuid']==p['desktop']['uuid'],'paired stack/device drift')
            ordered.update(zip(c['ids'],values(c['predictions'],len(c['ids']))))
            rows.update({x['study']:(x['windows'],x['batch_windows']) for x in c['rows']})
        need(all(rows[x['study']]==(x['windows'],x['batch_windows']) for x in p['rows']),'paired windows drift')
        pooled[flip]=np.asarray([ordered[u] for u in b['ids']]); pooled[arm]=values(p['predictions'],len(b['ids']))
    return pooled


def run():
    review_ready(); e.verify_prepared(); need(not (WORK/'attempt.json').exists(),'previous attempt preserved')
    pilot,confirmation=caches()
    cache_files={str(m.WORK.relative_to(ROOT)):sha(m.WORK/'measurement.json') for m in (predecessor.pilot,predecessor)}
    e.run()
    try:
        check_paired_cache(read('attempt'),pilot,confirmation,current())
        m=read('measurement'); m['cache_measurements']=cache_files; return save('measurement',m)
    except BaseException as exc:
        a=read('attempt'); a.update(state='HOLD',error=str(exc)); save('attempt',a); raise


def evaluate():
    import numpy as np
    from tools.job_runtime import job_status,LOGS
    g.c.cpu_only(); r=review_ready(); e.verify_prepared()
    need(job_status(LOGS,TAG,expected_command=launch_command(),tail=False)['state']=='SUCCESS','Q28 job incomplete')
    m=read('measurement'); need(m['binding_sha256']==r['binding_sha256'] and m['attempt_sha256']==sha(WORK/'attempt.json') and m['review_sha256']==sha(WORK/'gpu-review.json'),'Q28 measurement seal drift')
    for p,h in m['files'].items(): need(sha(WORK/p)==h,'Q28 output drift')
    for p,h in m['cache_measurements'].items(): need(sha(ROOT/p/'measurement.json')==h,'cache measurement changed')
    aa=read('attempt'); need(aa['state']==m['state']=='COMPLETE_PENDING_RESULT_REVIEW' and set(aa['arms'])==set(ARMS),'Q28 incomplete')
    for arm,v in aa['arms'].items():
        check_measurement(v,r,arm,v['desktop'],v['telemetry'])
        sup=io.read_json(WORK/'gpu'/arm/'supervision.json'); need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[],'supervision failed')
        with e.serving(arm): actual,_=g.check_controller(WORK/'gpu'/arm/'output',r['binding']['ids'],'plain','cuda')
        need(np.array_equal(actual,v['predictions']),'Q28 prediction copies differ')
    pilot,confirmation=caches(); pred=check_paired_cache(aa,pilot,confirmation,r); results={}
    for name,key in cohorts():
        ids=r['binding'][key]; ix=[r['binding']['ids'].index(u) for u in ids]
        results[name]=metrics(e.panel.outcomes(ids),{a:v[ix] for a,v in pred.items()},read('labels')[name])
    return save('evaluation',stamp(state='COMPLETE_PENDING_RESULT_REVIEW',binding_sha256=r['binding_sha256'],measurement_sha256=sha(WORK/'measurement.json'),
        result=results,primary_cohort='confirmation',ancestry=r['binding']['ancestry'],decision='HOLD_PENDING_INDEPENDENT_RESULT_REVIEW',qualified_for_submission=False))


def admit_dry_run():
    g.c.cpu_only(); r=current(); evidence={}
    # No GPU query/admission: exercise the measured memory formula using the sealed historical desktop.
    aa=io.read_json(predecessor.MEMORY_SOURCE/'attempt.json'); desktop=aa['arms']['flip']['desktop']
    for arm in ARMS: evidence[arm]=e.admit_memory(r,arm,desktop)
    return save('admit-dry-run',stamp(state='PASS_NO_GPU_ADMISSION',binding_sha256=r['binding_sha256'],measured_peaks=evidence))


# Configure only the private module; public predecessor stays read-only.
for name in ('VERSION','TAG','BUILDER','SCRIPT','WORK','ARMS','RULES'): setattr(e,name,globals()[name])
e.BUNDLES={a:predecessor.BUNDLES[a.replace('plain','flip')] for a in ARMS}
e.MANIFESTS={a:predecessor.MANIFESTS[a.replace('plain','flip')] for a in ARMS}
for name in ('current','review_ready','infer','check_measurement'): setattr(e,name,globals()[name])
_data_pins=e.data_pins
def data_pins():
    # The CPU oracle aliases are not serving inputs.
    return {p:h for p,h in _data_pins().items() if p!='train_series.csv' and not p.startswith('train_series/')}
e.data_pins=data_pins


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['register','prepare','smoke','replay','validate','admit-dry-run','launch-check','run','_worker','evaluate'])
    ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args()
    need(not args.dry_run or args.action=='prepare','dry-run only for prepare')
    if args.action not in ('run','_worker'): os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.action=='prepare': result=prepare(args.dry_run)
    elif args.action=='_worker': result=e.worker()
    elif args.action=='launch-check': result=save('launch-check',e.launch_check())
    else: result=globals()[args.action.replace('-','_')]()
    if isinstance(result,dict): print(json.dumps({k:v for k,v in result.items() if k in ('state','studies','binding_sha256','command')},indent=2))


if __name__=='__main__': main()
