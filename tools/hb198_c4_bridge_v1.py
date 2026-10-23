"""H198 C4 fixed bridge v1.0.1; CPU readers and a separately admitted B cache.

No automatic inference, network or submission. OOF commands install the audited
CPU tripwire; checkpoint/renderer commands are separate processes without it.
Changelog v1.0.1: exact parent (average rank-1)/(N-1) for the OOF proxy.
The unrevised rank/N draft was archived as invalid before independent review.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import c85r7_r6w958c4_v1147 as build
need,sha,read,rel,write=build.need,build.sha,build.read,build.rel,build.write
OUT=build.SCRATCH
VERSION='1.0.1'
CORPUS=ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02'
COMMON=ROOT/'.local/scratch/hb167-common/hb167/gpu'
COMMON_REG=ROOT/'artifacts/experiments/HB167_PARENT_COMMON/v1.0.3/registration.json'
ROUTING=ROOT/'.local/scratch/slot-b/routing-1016/astra/input_hashes.json'
R6=ROOT/'.local/scratch/r6-gold58-build/r6-gold58/gpu/flip/output'
QPOOL=ROOT/'.local/scratch/slot-b/cpu-small/QPOOL'
SCREEN=ROOT/'artifacts/experiments/EXP183_MULTISCALE_5FOLD/screen/result.json'
AUTH=ROOT/'artifacts/experiments/HB105_SERVING_V1/input-manifest.json'
TAG='H-B-198-C4-GOLD58'
PYTHON='C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'


def save(name,value):
    from tools.multi_lens_readout import clean
    write(OUT/name,build.encoded(clean(value)))


def probabilities(a,n=None):
    import numpy as np
    a=np.asarray(a)
    need(a.ndim==2 and a.shape[1]==12 and len(a)>0 and (n is None or len(a)==n) and
         np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'probability shape/range/finite')
    return a


def same_entries(a,b):
    keys=('family','seed','fold','sha256','source_sha256')
    return [tuple(e[k] for k in keys) for e in a]==[tuple(e[k] for k in keys) for e in b]


def authenticate_oof():
    from tools import hb105_aligned_oof_v1 as aligned
    from tools import ensemble_ablation_v1 as ea
    import numpy as np
    bindings={}
    def bound(p,expected=None):
        p=Path(p); p=p if p.is_absolute() else ROOT/p
        h=sha(p); need(expected is None or h==expected,'input changed: '+rel(p));bindings[rel(p)]=h;return p
    # Reuse the audited ancestry/flip loaders, not their unrelated Q7 formulas.
    allraw,rosters,frozen,pin=aligned.load_frozen()
    bound(ROOT/aligned.EXP/'input-source-manifest.json',pin['sha256'])
    for p,info in frozen['bindings'].items(): bound(p,info['sha256'])
    hist=read(bound(QPOOL/'results.json'))
    bound(QPOOL/'REGISTRATION.md',hist['registration_sha256']);bound(QPOOL/'analyze.py',hist['script_sha256'])
    original=build.parent()[1]; entries=json.loads(original['q-a5q-manifest.json'])['entries']
    records=hist['parent_members']
    need(same_entries(entries,[r['parent'] for r in records]),'parent source/deploy tuple order')
    raw={f:{'R42':allraw[f]['R42'],'R43':allraw[f]['R43'],'C4':allraw[f]['C4']} for f in range(5)}
    for record,e in zip(records,entries):
        directory=(ROOT/e['source'].replace('\\','/')).parent
        inv=read(bound(directory/('receipt.json' if e['family']=='dino' else 'output-inventory.json')))
        if e['family']=='dino':
            need(inv['status']=='PASS' and inv['phase']=='full','DINO receipt');inv=inv['outputs']
        need(inv['model_best.pt']['sha256']==e['source_sha256'],'model ancestry')
        prov=read(bound(directory/'provenance.json',inv['provenance.json']['sha256']))
        summary=read(bound(directory/'summary.json',inv['summary.json']['sha256']))
        f=e['fold'];r=rosters[f]
        roster=dict(train_sha256=aligned.serving.uid_hash(r['train']),sorted_train_sha256=aligned.serving.uid_hash(sorted(r['train'])),val_sha256=aligned.serving.uid_hash(r['val']))
        ea.validate_summary({**summary,'provenance':prov},roster,len(r['val']),f)
        key='R'+str(e['seed']) if e['family']=='l2' else 'DINO' if e['family']=='dino' else 'CNX'
        need(record['key']==key and record['fold']==f,'OOF fold/key')
        path=bound(record['prediction'],hist['bindings'][record['prediction']])
        values=aligned.load_csv(path,r['val']);probabilities(values,len(r['val']))
        if key.startswith('R'): need(np.array_equal(raw[f][key],values),'R50 saved flip mismatch')
        raw[f][key]=values
    screen=read(bound(SCREEN))
    for v in screen['input_files'].values(): bound(v['path'],v['sha256'])
    bound(ROOT/'.local/exp183-overlay/manifest.json',screen['manifest_sha256'])
    bound(ROOT/'.local/exp183-overlay/verification.md',screen['verification_sha256'])
    for p,h in frozen['ruler_bindings'].items(): bound(p,h)
    bound(AUTH,aligned.AUTH_SHA)
    for p,info in aligned.source_pins().items(): bound(p,info['sha256'])
    for p in (Path(__file__),OUT/'SPEC.md',build.REVIEW,ROOT/'tools/c85r7_r6w958c4_v1147.py',ROOT/'tools/exp169_release_v2.py',ROOT/'tools/exp169_release_v4.py',ROOT/'training/src/raptor_exp179.py',ROOT/'artifacts/experiments/EXP183_MULTISCALE_5FOLD/PROTOCOL.md'): bound(p)
    return raw,rosters,dict(bindings=bindings,ruler_bindings=frozen['ruler_bindings'],parent_entries=entries,historical_screen=screen)


def oof_values(raw):
    from tools.exp169_release_v2 import rank
    import numpy as np
    need(set(raw)==set(range(5)),'five folds required')
    fam={k:[] for k in ('R','D','C','C4')}
    for f in range(5):
        row=raw[f];need(set(row)=={'R42','R43','DINO','CNX','C4'},'fixed OOF member set')
        need(len({a.shape for a in row.values()})==1,'fold shapes')
        for a in row.values():
            probabilities(a);need(a.dtype==np.float32,'CSV FP32 required')
        rr=(row['R42']+row['R43'])/2
        for k,a in zip(fam,(rr,row['DINO'],row['CNX'],row['C4'])): fam[k].append(a)
    ranks={k:rank(np.concatenate(v)) for k,v in fam.items()}
    return rank((ranks['R']+ranks['D']+ranks['C'])/3),rank((ranks['R']+ranks['D']+ranks['C4'])/3)


def q32(delta):
    import numpy as np
    d=np.asarray(delta,float);need(d.shape==(4,12),'4x12 deltas')
    return bool(np.isfinite(d).all() and d[:2].mean()>0 and d[:2][:,[3,5,6,8]].mean()>=0 and (d>=-.01).all())


def report(ctx,a,b):
    from tools import multi_lens_readout as ml
    import numpy as np
    d=ctx.delta(a,b);aa,bb=ctx.score(a),ctx.score(b)
    return ml.clean(dict(gate_pass=q32(d[0]),all48_finite=bool(np.isfinite(d[0]).all()),
        primary_macro=ml.stat(d[:,:2].mean((1,2))),primary_weak4=ml.stat(d[:,:2][:,:,ml.WEAK].mean((1,2))),
        min_target_delta=float(d[0].min()),rulers={r:dict(macro=ml.stat(d[:,i].mean(1)),weak4=ml.stat(d[:,i,ml.WEAK].mean(1)),
        targets={t:dict(delta=ml.stat(d[:,i,j]),control_auc=bb[0,i,j],candidate_auc=aa[0,i,j],minimum_class=ctx.refs[r][2],
                       positives=int(ctx.refs[r][0][ctx.refs[r][1][:,j],j].sum()),negatives=int((1-ctx.refs[r][0][ctx.refs[r][1][:,j],j]).sum())) for j,t in enumerate(ml.TARGETS)}) for i,r in enumerate(ml.RULERS)}))


def freeze():
    raw,rosters,proof=authenticate_oof()
    screen=proof.pop('historical_screen')
    pins=dict(version=VERSION,candidate=build.KEY,builder=build.BUILDER,design_review_sha256=build.REVIEW_SHA,
        exposure='EXPOSED',clean_oof=False,rows=sum(len(r['val']) for r in rosters),fold_sizes=[len(r['val']) for r in rosters],
        bootstrap=dict(draws=300,seed=20261003,unit='connected component',conditional=True),
        arithmetic='FP32 paired R50 seed probabilities; concat heldout folds; exact parent (average rank-1)/(N-1); FP64 equal-third Q mean/rerank',
        support={'jev':1,'consensus':2,'dread':1,'aquila':1},historical_status=screen['status'],**proof)
    save('input-pins.json',pins)
    return dict(status='FROZEN_BEFORE_METRICS',rows=pins['rows'],inputs=len(pins['bindings']))


def validate_pins():
    pins=read(OUT/'input-pins.json')
    need(pins['candidate']==build.KEY and pins['design_review_sha256']==build.REVIEW_SHA,'candidate design')
    for p,h in pins['bindings'].items():need(sha(ROOT/p)==h,'frozen input drift: '+p)
    return pins


def oof():
    pins=validate_pins();raw,rosters,proof=authenticate_oof()
    need(proof['bindings']==pins['bindings'],'input set changed')
    from tools import multi_lens_readout as ml
    import numpy as np
    ids=sum((r['val'] for r in rosters),[]);need(len(ids)==len(set(ids))==4343,'unique heldout4343')
    q0,q4=oof_values(raw);ctx=ml.make_context(ids,ROOT);need(ctx.bindings==pins['ruler_bindings'],'ruler mask binding')
    result=report(ctx,q4,q0);folds={};offset=0
    for f,r in enumerate(rosters):
        n=len(r['val']);folds[str(f)]=report(ml.make_context(r['val'],ROOT),q4[offset:offset+n],q0[offset:offset+n]);offset+=n
    from sklearn.metrics import roc_auc_score
    errors=[]
    for a in (q0,q4):
        scores=ctx.score(a)[0]
        for i,r in enumerate(ml.RULERS):
            y,mask,_=ctx.refs[r]
            for j in range(12):errors.append(abs(scores[i,j]-roc_auc_score(y[mask[:,j],j],a[mask[:,j],j])))
    need(max(errors)<1e-12,'independent sklearn AUC')
    hist=proof['historical_screen']
    output=dict(version=VERSION,candidate=build.KEY,status='COMPLETE_Q_ONLY_PROXY',input_pins_sha256=sha(OUT/'input-pins.json'),
        primary=result,fold_diagnostics=folds,historical=dict(status=hist['status'],registered_gate=hist['registered_gate'],ruler_advisory=hist['ruler_advisory'],direct=hist['direct'],bundle_replacement=hist['bundle_replacement']),
        independent_auc_max_error=max(errors),independent_auc_checks=len(errors),qualified_for_submission=False,
        exposure='EXPOSED',clean_oof=False,release='HOLD_MISSING_GOLD58_AND_INDEPENDENT_REVIEW_NEW_PATH_QUALIFICATION')
    validate_pins();save('OOF-PROXY.json',output)
    lines=['# H198 fixed C4 Q-only proxy','',f"Q32: {'PASS' if result['gate_pass'] else 'HOLD'}; historical EXP183 status: {hist['status']}.",
           '4343 generator-heldout rows; 300 paired component draws seed20261003. Reused development rulers; not exact H189 F or independent clean efficacy.','',
           f"Primary macro: {result['primary_macro']}; weak4: {result['primary_weak4']}; min of48: {result['min_target_delta']}.",
           '', '| Ruler | Macro delta | CI95 | Weak4 |','|---|---:|---|---:|']
    for r in ml.RULERS:
        row=result['rulers'][r];lines.append(f"| {r} | {row['macro']['value']:.9f} | {row['macro']['ci95']} | {row['weak4']['value']:.9f} |")
    lines+=['','All12 targets, original direct PF OA/Contusion effects and all fold slices are in OOF-PROXY.json. Historical REVIEW caution remains. Gold58 missing; no release decision.','SUB-AGENTS: none']
    write(OUT/'OOF-REPORT.md','\n'.join(lines)+'\n')
    return dict(status=output['status'],q32='PASS' if result['gate_pass'] else 'HOLD',macro=result['primary_macro']['value'],weak4=result['primary_weak4']['value'],min48=result['min_target_delta'])


def functions(data,names,scope=None):
    text=data.decode() if isinstance(data,bytes) else data
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes}==set(names),'actual extracted function coverage')
    ns={} if scope is None else dict(scope)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<C4-extracted-functions>','exec'),ns)
    return ns


def modules():
    sys.path.insert(0,str(ROOT/'training/src'))
    import raptor_common,raptor_radimagenet,raptor_runtime,raptor_exp179
    return raptor_common,raptor_radimagenet,raptor_runtime,raptor_exp179


def candidate_windows(volume,centers,scope):
    source=(OUT/'candidate-runtime/exp169_partner_bundle_release.py').read_text(encoding='utf-8')
    start=source.index('                center3 = center3_windows_from(volume, centers, 224)')
    end=source.index('\n                prepared = time.monotonic()',start)
    import textwrap
    snippet=textwrap.dedent(source[start:end]);ns=dict(scope,volume=volume,centers=centers,i=0,flags=[False]*36)
    exec(compile(snippet,'<actual-C4-producer-payload>','exec'),ns)
    return ns


def corpus_registration():
    import numpy as np
    common,_,_,_=modules();corpus=common.Corpus(CORPUS)
    auth=read(AUTH)
    for name,pin in auth['corpus']['metadata'].items():need(sha(CORPUS/name)==pin['sha256'],'corpus authority metadata: '+name)
    reg=read(COMMON_REG);ids=reg['ids'];need(len(ids)==len(set(ids))==58,'same58 corpus IDs')
    need(corpus.contract=='raptor64_336_c140_s02' and corpus.depth==64 and set(ids)<=set(corpus.ids),'corpus UID/contract')
    volumes=[]
    for uid in ids:
        v=corpus.volume(uid);need(v.dtype==np.uint8 and v.shape==(64,336,336),'cached volume contract')
        volumes.append(dict(uid=uid,sha256=hashlib.sha256(v.tobytes()).hexdigest(),centers=list(map(int,common.select_centers(corpus.mask(uid),64,36,corpus.spec,'within-slot-valid')))))
    return corpus,dict(path=rel(CORPUS),contract=corpus.contract,metadata={n:sha(CORPUS/n) for n in auth['corpus']['metadata']},volumes=volumes,ids=ids,targets=reg['targets'])


def parity():
    import numpy as np
    import torch
    torch.set_num_threads(1);need(not torch.cuda._initialized,'CPU only')
    common,rad,runtime,exp179=modules();corpus,cp=corpus_registration()
    parent=build.parent()[1];candidate=build.unpack(read(build.OUT/'kernel/release.ipynb'))[0]
    ns=functions(candidate['exp169_partner_bundle_release.py'],['plane_flags'],dict(need=need))
    flip=functions((ROOT/'tools/exp124_flip_tta.py').read_bytes(),['plane_ranges','flip_mask','mirror_windows'])
    scope=dict(center3_windows_from=rad.center3_windows_from,windows_from=common.windows_from,normalise=rad.normalise,normalize_input=runtime.normalize_input)
    rng=np.random.default_rng(20261018);synthetic=rng.integers(0,256,(64,336,336),dtype=np.uint8)
    cases=[(synthetic,np.ones(64,np.uint8),'synthetic'),(synthetic,np.zeros(64,np.uint8),'empty')]+[(corpus.volume(u),corpus.mask(u),'real'+str(i)) for i,u in enumerate(cp['ids'][:3])]
    receipts=[]
    for v,mask,name in cases:
        centers=common.select_centers(mask,64,36,corpus.spec,'within-slot-valid',None)
        if name=='synthetic':centers[3]=centers[8]=-1
        actual=candidate_windows(v,centers,scope);payload=actual['payload']
        reference=rad.normalise(exp179.scale_windows(v,centers,224,'mixed'),'imagenet')
        need(len(payload)==5 and np.array_equal(payload[2],reference),'candidate normalized mixed parity')
        need(np.array_equal(payload[1],rad.normalise(rad.center3_windows_from(v,centers,224),'radimagenet')),'R50 unchanged')
        need(np.array_equal(payload[3],runtime.normalize_input(common.windows_from(v,centers,224),'imagenet')),'DINO unchanged')
        flags=ns['plane_flags'](centers,corpus.spec);mirror=payload[2].copy();mirror[np.asarray(flags,bool)]=mirror[np.asarray(flags,bool),:,:,::-1]
        trained=flip['mirror_windows'](torch.from_numpy(reference),flip['flip_mask'](centers,flip['plane_ranges'](corpus.spec))).numpy()
        need(np.array_equal(mirror,trained) and reference.dtype==np.float32,'mirror FP32 parity')
        receipts.append(dict(case=name,windows=36,empty=centers.count(-1),exact=True))
    for f in ('_infer_one','aggregate','plane_flags'):
        a=next(n for n in ast.parse(parent['exp169_partner_bundle_release.py']).body if isinstance(n,ast.FunctionDef) and n.name==f)
        b=next(n for n in ast.parse(candidate['exp169_partner_bundle_release.py']).body if isinstance(n,ast.FunctionDef) and n.name==f)
        need(ast.dump(a)==ast.dump(b),'frozen function drift '+f)
    save('RENDERER-PARITY.json',dict(status='PASS_CPU_REAL_RENDERER',kernel_sha256=sha(build.OUT/'kernel/release.ipynb'),cases=receipts,corpus=cp,cuda_initialized=False))
    return dict(status='PASS_CPU_REAL_RENDERER',cases=len(cases),exact=True)


def strict_models():
    import torch
    torch.set_num_threads(1);need(not torch.cuda._initialized,'CPU only')
    common,_,_,_=modules();files=build.unpack(read(build.OUT/'kernel/release.ipynb'))[0]
    from tools import exp169_partner_bundle_release as original
    ns=functions(files['exp169_partner_bundle_release.py'],['model_meta'],dict(need=need,TARGETS=original.TARGETS))
    cm=read(build.C4_DIR/'release-manifest.json');records=[]
    for e in cm['entries']:
        p=ROOT/e['source'].replace('\\','/');need(sha(p)==e['sha256']==e['source_sha256'],'original/deploy pin')
        ck=torch.load(p,map_location='cpu',weights_only=False);ns['model_meta'](ck,e)
        model=common.build_model(ck['arch'],pretrained=False);out=model.load_state_dict(ck['model'],strict=True)
        need(not out.missing_keys and not out.unexpected_keys,'strict state')
        records.append(dict(fold=e['fold'],sha256=e['sha256'],strict=True,parameters=sum(p.numel() for p in model.parameters())))
        del model,ck
    save('CHECKPOINT-STRICT.json',dict(status='PASS_CPU_STRICT_NO_FORWARD',kernel_sha256=sha(build.OUT/'kernel/release.ipynb'),models=records,cuda_initialized=False))
    return dict(status='PASS_CPU_STRICT_NO_FORWARD',models=len(records))


def prepare_gold():
    build.validate();reg=read(COMMON_REG);routing=read(ROUTING)
    files={}
    for p in (COMMON/'worker.json',COMMON/'predictions.npz',COMMON/'q-leaves.npz',R6/'predictions.csv',R6/'controller/submission_receipt.json'):
        need(sha(p)==routing[rel(p)],'parent cache hash');files[rel(p)]=sha(p)
    worker=read(COMMON/'worker.json');need(worker['state']=='INFERENCE_COMPLETE' and worker['studies']==58 and worker['registration_sha256']==sha(COMMON_REG),'parent cache receipt')
    _,cp=corpus_registration();cm=read(build.C4_DIR/'release-manifest.json')
    need(same_entries(reg['q_entries'],json.loads(build.parent()[1]['q-a5q-manifest.json'])['entries']),'retained source/deploy cache tuples')
    card=dict(schema='h198-c4-cache/v1',candidate=build.KEY,tag=TAG,executor='B_ONLY',workload=290,models=cm['entries'],corpus=cp,
        parent_files=files,kernel_sha256=sha(build.OUT/'kernel/release.ipynb'),input_pins_sha256=sha(OUT/'input-pins.json'),
        source_sha256=sha(__file__),renderer_sha256=sha(OUT/'candidate-runtime/exp169_partner_bundle_release.py'),
        sources={rel(ROOT/p):sha(ROOT/p) for p in ('tools/hb149_monitor_v125.py','tools/hb145_output_io.py','tools/gpu_queue.py','tools/gpu_job.py','tools/job_runtime.py','training/src/raptor_common.py','training/src/raptor_radimagenet.py','training/src/raptor_runtime.py','training/src/raptor_exp179.py')},
        outputs=['gold58/c4-members.npz','gold58/worker.json'],expected_shape=[5,58,12],dtype='float32',
        execution='serial5 models x same58, CUDA_VISIBLE_DEVICES admitted physical index; logical cuda:0; FP16 autocast, FP32 sigmoid, original plane mirror probability mean',
        guards=dict(progress_seconds=900,wddm_seconds=600,throughput_fraction=.25,throughput_seconds=1800,max_peak_reserved_gib=12,desktop_plus_peak_headroom_gib=2),
        cache_worker_sha256=sha(ROOT/'tools/hb198_c4_gold58_v1.py'),gate_sha256=sha(ROOT/'tools/strict_variant_gate_c4_v1.py'),
        independent_review=None,resource_claim=None,memory_measurement=None,status='PREPARED_NOT_ADMITTED',exposure='EXPOSED',clean_oof=False)
    save('gold58-launch.json',card)
    queue=[dict(tag=TAG,command=[PYTHON,'-B',str(ROOT/'tools/hb198_c4_gold58_v1.py'),'run'],continue_on_failure=False)]
    from tools.gpu_queue import validate_plan
    validate_plan(queue);save('gpu-queue.json',queue)
    return dict(status='GOLD_CARD_PREPARED_NOT_ADMITTED',workload=290,models=5,studies=58)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','validate','oof','parity','strict-models','prepare-gold'));p.add_argument('--out');p.add_argument('--manifest');a=p.parse_args()
    if a.out:need(Path(a.out).resolve()==OUT.resolve(),'fixed output')
    if a.manifest:need(Path(a.manifest).resolve()==(OUT/'input-pins.json').resolve(),'fixed input pins')
    try:
        result={'freeze':freeze,'validate':lambda:dict(status='PIN_VALID',inputs=len(validate_pins()['bindings'])),'oof':oof,'parity':parity,'strict-models':strict_models,'prepare-gold':prepare_gold}[a.command]()
        print(json.dumps(result,sort_keys=True));return 0
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps(dict(status='HOLD',reason=str(exc))));return 2


if __name__=='__main__':raise SystemExit(main())
