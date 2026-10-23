"""R6/R7 OAI readout v1.1.0. Confirmation extension; v1.0.0 unchanged.

Changelog 2026-10-16: register pilot96/complement461/pooled557; authenticated
pilot reuse; fixed macro7 and original floors; inherited serving and watchdog.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import io as stringio
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import r6_gold58_precheck_v1 as g
from tools import hb144_oai_public_v1 as raw
from tools import oai_moaks_panel as panel
from tools import r6_oai_readout_v1 as pilot

VERSION, TAG, BUILDER = '1.1.0', 'R6_R7_OAI_V110', 'astra-oai-extension'
WORK = ROOT/'.local/scratch/astra-oai-extension/r6-oai'
PILOT = pilot.WORK
PILOT_TOOL_SHA = 'c2e43f49523ae996fbb9966cd81078744d1ee221f86fe9847c65b4f93d051329'
PILOT_REG_SHA = '7af60db77a6605f157b1647a7c0495e0bcad67aaf92c3a928484a1a91bcf80f6'
PILOT_MEASUREMENT_SHA = '74a236e355c415b28c8ab2d360f4d4bf1ed3a6e327341d6dccc4f24aa086e14f'
MACRO7 = [2,3,5,6,7,8,9]
SCRIPT = Path(__file__).resolve()
need, sha, digest, io = g.need, g.sha, g.digest, g.io
d, t = g.d, g.t
ARMS = ('R7flip', 'R6flip')
BUNDLES = dict(R7flip=ROOT/'storage/external/heliosli-r7cnxfsss-v1', R6flip=g.BUNDLE)
MANIFESTS = dict(R7flip=d.gold.R7_MANIFEST_SHA, R6flip=g.MANIFEST_SHA)
STAGES = dict(R7flip=d.STAGE, R6flip=g.STAGES[0])
MEMORY_SOURCE = g.WORK
MAPPED = [0,2,3,4,5,6,7,8,9]
COMMON = ['R','R224','EXP143v2','EXP155','Q_A5q']
RULES = dict(g.RULES, ranks='each fixed cohort once before bootstrap', min_class=5, n=461,
             macro7=MACRO7, pilot_n=96, pooled_n=557,
             mapped=MAPPED, primary='R6flip-minus-R7flip', exposure='EXPOSED', clean_oof=False)


def save(name, obj):
    io.journal(WORK/(name+'.json'), obj, WORK)
    return obj


def read(name):
    return io.read_json(WORK/(name+'.json'))


def stamp(**kw):
    return dict(identity=TAG, version=VERSION, exposure='EXPOSED', clean_oof=False, **kw)


def select_ids(ids):
    need(len(ids)==len(set(ids))==RULES['pooled_n'], 'duplicate/short roster')
    need(len({u.split('_')[1] for u in ids})==len(ids),'duplicate participant')
    chosen=sorted(set(ids)-set(pilot.select_ids(ids)))
    need(len(chosen)==RULES['n'],'confirmation size differs')
    return chosen


def pilot_registration():
    # Hash bytes first; no prediction is deserialized during registration.
    need(sha(Path(pilot.__file__))==PILOT_TOOL_SHA,'frozen v1 tool drift')
    need(sha(PILOT/'registration.json')==PILOT_REG_SHA,'pilot registration drift')
    need(sha(PILOT/'measurement.json')==PILOT_MEASUREMENT_SHA,'pilot measurement drift')
    reg=io.read_json(PILOT/'registration.json')
    need(digest(reg['binding'])==reg['binding_sha256'],'pilot binding drift')
    need(reg['binding']['ids']==pilot.select_ids(raw.ids_only()),'pilot roster drift')
    measurement=io.read_json(PILOT/'measurement.json')
    need(measurement['review_sha256']==sha(PILOT/'gpu-review.json'),'pilot review chain drift')
    review=io.read_json(PILOT/'gpu-review.json')
    pilot.check_review(review,reg,sha(PILOT/'validate.json'))
    validation=io.read_json(PILOT/'validate.json')
    need(validation['state']=='CPU_VALIDATED' and validation['binding_sha256']==reg['binding_sha256'],
         'pilot validation binding drift')
    for name in ('cpu-smoke','prepare','replay'):
        need(validation['evidence'][name]==sha(PILOT/(name+'.json')),'pilot CPU evidence chain drift: '+name)
    return reg


def contract_files():
    g.contracts()  # Includes the notebook literal and source identity proof; no execution.
    result={a:d.stage_files(s) for a,s in STAGES.items()}
    for n in result['R7flip']:
        if n.startswith('exp169_r7_src/') or n=='exp189_s6_port.py':
            need(result['R7flip'][n]==result['R6flip'][n],'worker source drift')
    return result


def export_contracts():
    for arm, files in contract_files().items():
        for name, content in files.items():
            if name=='exp169_r7_port.py' or name=='exp189_s6_port.py' or name.startswith('exp169_r7_src/'):
                p=WORK/'contracts'/arm/name
                p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists(): need(p.read_bytes()==content,'existing contract differs')
                else: p.write_bytes(content)


def port_module(arm):
    need(arm in ARMS,'unknown arm')
    return d.import_path('oai_port_'+arm,WORK/'contracts'/arm/'exp169_r7_port.py')


def source_pins():
    paths=[SCRIPT,ROOT/'tests/test_r6_oai_readout_v110.py',WORK/'SPEC.md',WORK/'PRE-REVIEW.md',
           raw.PANEL/'registration.json',raw.PANEL/'PROTOCOL.md',raw.PANEL/'runtime-v1.0.3.json',
           raw.PANEL/'vendor/manifest.json',g.NOTEBOOK,d.gold.RUNTIME/'prepare.json']
    paths += [s/'kernel/release.ipynb' for s in STAGES.values()]
    paths += [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
              and Path(m.__file__).resolve().is_relative_to(ROOT/'tools')]
    paths += [ROOT/'tools'/n for n in ('gpu_job.py','job_runtime.py','gpu_queue.py')]
    paths += [MEMORY_SOURCE/(n+'.json') for n in ('registration','measurement','attempt')]
    paths += [g.c.PINS,g.c.SOURCE/'registration.json',g.c.SOURCE/'flip-1/worker-status.json',g.c.SOURCE/'flip-1/supervision.json']
    paths += [PILOT/(n+'.json') for n in ('registration','measurement','attempt','evaluation','gpu-review','validate','replay','cpu-smoke','prepare','labels')]
    paths += [ROOT/'.local/scratch/r6-oai-result-review/r6-oai/RESULT-REVIEW.md']
    reg=io.read_json(raw.PANEL/'registration.json')
    paths += [panel.registered_path(p,raw.PANEL) for p in reg['pins']]
    paths += [p for p in (raw.PANEL/'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in (WORK/'contracts').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}


def launch_command():
    return [str(t.GPU_PYTHON),'-B',str(SCRIPT),'run']


def memory_evidence():
    """Reuse completed real4070 full192-window, four-fold+flip measurements."""
    from tools.job_runtime import job_status,LOGS
    need(job_status(LOGS,g.TAG,expected_command=g.launch_command(),tail=False)['state']=='SUCCESS','historical R6 job incomplete')
    rr=io.read_json(MEMORY_SOURCE/'registration.json'); mm=io.read_json(MEMORY_SOURCE/'measurement.json')
    aa=io.read_json(MEMORY_SOURCE/'attempt.json')
    need(rr['binding_sha256']==digest(rr['binding'])==mm['binding_sha256']==aa['binding_sha256'] and
         mm['attempt_sha256']==sha(MEMORY_SOURCE/'attempt.json') and
         aa['state']==mm['state']=='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW','historical R6 seal drift')
    for name,h in mm['files'].items(): need(sha(MEMORY_SOURCE/name)==h,'historical R6 artifact drift')
    r6=aa['arms']['flip']; g.check_measurement(r6,rr,'flip',r6['desktop'],r6['telemetry'])
    need(sha(g.c.PINS)==g.c.PINS_SHA,'historical R7 seal drift')
    seal=io.read_json(g.c.PINS)['files']
    def sealed(name):
        p=g.c.SOURCE/name
        need(sha(p)==seal[str(p.relative_to(ROOT))],'historical R7 artifact drift')
        return io.read_json(p)
    old=sealed('registration.json'); r7=sealed('flip-1/worker-status.json'); sup=sealed('flip-1/supervision.json')
    need(rr['binding']['r6_identity']['manifest_sha256']==MANIFESTS['R6flip'] and
         old['binding']['r7_manifest_sha256']==MANIFESTS['R7flip'],'historical memory bundle mismatch')
    need(sup['exit_code']==0 and sup['error'] is None and r7['binding_sha256']==old['binding_sha256'],'historical R7 incomplete')
    d.check_arm('gold','flip-1',d.canonicalize_rows(r7,old['binding']['ids']),sup['desktop_before_cuda'],sup['telemetry'])
    result={}
    for arm,v,desktop,telemetry in [('R6flip',r6,r6['desktop'],r6['telemetry']),('R7flip',r7,sup['desktop_before_cuda'],sup['telemetry'])]:
        need(v['runtime']==r6['runtime'] and max(x['batch_windows'] for x in v['rows'])==192,'historical full-shape recipe mismatch')
        peak=d.r7.memory_admission(desktop,[x['peak_reserved_bytes'] for x in v['rows']],telemetry['peak_gpu_mib']*2**20)
        result[arm]=dict(uuid=desktop['uuid'],peak_bytes=peak,batch_windows=192,runtime=v['runtime'])
    need(result['R6flip']['uuid']==result['R7flip']['uuid'],'historical paired device mismatch')
    return result


def admit_memory(reg,arm,desktop):
    evidence=reg['binding']['memory'][arm]
    need(desktop['uuid']==evidence['uuid'],'measured physical4070 required')
    return d.r7.memory_admission(desktop,[evidence['peak_bytes']])


def register():
    g.c.cpu_only()
    need(not (WORK/'attempt.json').exists(),'attempt already exists')
    export_contracts()
    identities={a:port_module(a).verify_bundle(BUNDLES[a],MANIFESTS[a],WORK/'contracts'/a/'exp169_r7_src')[1] for a in ARMS}
    old=pilot_registration()
    ids=select_ids(raw.ids_only())  # ID-only; no labels or predictions opened.
    pilot_ids=old['binding']['ids']; all_ids=sorted(ids+pilot_ids)
    a=raw.audit()
    rows={u:raw.records(a,u) for u in all_ids}
    need(all(len(r)>=4 and len({x['SeriesInstanceUID'] for x in r})==len(r) for r in rows.values()),'missing or duplicate native series')
    need(identities==old['binding']['identities'],'pilot model identity differs')
    need(all(rows[u]==old['binding']['routes'][u] for u in pilot_ids),'pilot route differs')
    b=dict(builder=BUILDER,ids=ids,pilot_ids=pilot_ids,pooled_ids=all_ids,smoke_id=pilot_ids[0],
           pilot_binding_sha256=old['binding_sha256'],
           targets=list(d.gold.LABELS),rules=RULES,identities=identities,
           sources=source_pins(),audit=a.binding(),routes=rows,command=launch_command(),memory=memory_evidence(),
           ancestry='R7 fm_aux present; R6 complete roster absent; RSNA-only/OAI-free unproved',
           descriptive='HOLD_MISSING_AUTHENTICATED_COMMON_OAI_PREDICTIONS',
           estimated_minutes=[132,180])
    value=stamp(state='REGISTERED_BEFORE_LABELS_AND_PREDICTIONS',binding=b,binding_sha256=digest(b))
    if (WORK/'registration.json').exists(): need(read('registration')==value,'registration immutable; new version required')
    else: save('registration',value)
    save('queue',[dict(tag=TAG,command=launch_command())])
    return stamp(state=value['state'],studies=len(ids),binding_sha256=value['binding_sha256'])


def current():
    r=read('registration'); b=r['binding']
    need(digest(b)==r['binding_sha256'] and b['rules']==RULES and b['ids']==select_ids(raw.ids_only()),'registration drift')
    for name,h in b['sources'].items(): need(sha(ROOT/name)==h,'source drift: '+name)
    for arm in ARMS:
        need(port_module(arm).verify_bundle(BUNDLES[arm],MANIFESTS[arm],WORK/'contracts'/arm/'exp169_r7_src')[1]==b['identities'][arm],'bundle drift')
    return r


def label_contract(y):
    import numpy as np
    y=np.asarray(y,float)
    need(y.ndim==2 and y.shape[1]==12 and len(y)>0 and not np.isinf(y).any() and np.isin(y[np.isfinite(y)],[0,1]).all(),'invalid masked labels')
    counts={str(j):dict(positive=int(np.sum(y[:,j]==1)),negative=int(np.sum(y[:,j]==0)),unknown=int(np.isnan(y[:,j]).sum())) for j in MAPPED}
    eligible=[j for j in MAPPED if min(counts[str(j)]['positive'],counts[str(j)]['negative'])>=RULES['min_class']]
    need(eligible,'no eligible targets')
    need(np.isnan(y[:,[1,10,11]]).all(),'unmapped targets must be unknown')
    return dict(eligible=eligible,macro7=MACRO7,counts=counts,labels_sha256=hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest())


def original_group(a, group, destination, old_receipt):
    """Reuse byte-authenticated HB144 extraction, otherwise extract audited tar bytes."""
    src=raw.SCRATCH/'dicom/test_images'/group['study_key']/group['series_key']
    old=old_receipt.get('sources',{}).get(group['group_key'])
    if old is None or not src.is_dir():
        return raw.extract_group(a,group,destination)
    ar=a.archives[group['series_key']]; st=Path(ar['path']).stat()
    need(st.st_size==ar['size']==old['size'] and st.st_mtime_ns==ar['mtime_ns']==old['mtime_ns'],'archive drift')
    import pydicom
    hashes={}; destination.mkdir(parents=True,exist_ok=True)
    for name,sop in zip(group['members'],group['sops']):
        source=src/(hashlib.sha256(name.encode()).hexdigest()+'.dcm')
        hdr=pydicom.dcmread(source,stop_before_pixels=True)
        need(str(hdr.SOPInstanceUID)==sop,'original SOP drift')
        hashes[name]=sha(source)
        target=destination/source.name
        if target.exists(): need(sha(target)==hashes[name],'staged DICOM drift')
        else: os.link(source,target)
    need(hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()==old['frame_bytes_sha256'],'original byte proof differs')
    need(len(list(destination.glob('*.dcm')))==len(hashes),'extra DICOM')
    return old


def data_pins():
    base=WORK/'data'
    return {p.relative_to(base).as_posix():sha(p) for p in sorted(base.rglob('*')) if p.is_file()}


def prepare(dry_run=False):
    g.c.cpu_only(); reg=current(); a=raw.audit(); ids=reg['binding']['ids']
    need(a.binding()==reg['binding']['audit'],'audit drift')
    if dry_run:
        for u in ids:
            need(raw.records(a,u)==reg['binding']['routes'][u],'route drift')
            for row in reg['binding']['routes'][u]:
                ar=a.archives[row['SeriesInstanceUID']]; st=Path(ar['path']).stat()
                need(st.st_size==ar['size'] and st.st_mtime_ns==ar['mtime_ns'],'archive drift')
        return save('prepare-dry-run',stamp(state='PASS',studies=len(ids),binding_sha256=reg['binding_sha256']))
    # Registration exists and is authenticated before first in-memory label access.
    need(panel.validate()==raw.ids_only(),'frozen panel differs')
    labels={name:label_contract(panel.outcomes(reg['binding'][key])) for name,key in
            [('pilot','pilot_ids'),('confirmation','ids'),('pooled','pooled_ids')]}
    if (WORK/'labels.json').exists(): need(read('labels')==labels,'label contract drift')
    else: save('labels',labels)
    rows=[]
    staged_ids=sorted(ids+[reg['binding']['smoke_id']])
    for i,u in enumerate(staged_ids):
        previous=raw.SCRATCH/'prepared'/(u+'.npz.json')
        old=io.read_json(previous) if previous.exists() else {}
        for row in reg['binding']['routes'][u]:
            group=a.groups[row['group_key']]
            original_group(a,group,WORK/'data/test_series'/u/row['SeriesInstanceUID'],old)
            rows.append({k:v for k,v in row.items() if k!='group_key'})
        if (i+1)%16==0: print(json.dumps(dict(prepared=i+1,total=len(ids))),flush=True)
    s=stringio.StringIO(newline=''); w=csv.DictWriter(s,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (WORK/'data/test_series.csv').write_text(s.getvalue(),encoding='utf-8',newline='')
    from tools import exp189_s6_port as s6
    dep=io.read_json(d.gold.RUNTIME/'prepare.json')
    need(d.gold.dir_tree(Path(dep['deps']))['tree_sha256']==dep['deps_tree_sha256'],'dependency drift')
    need(d.gold.probe_deps(Path(dep['deps']),d.CODE)==dep['versions'],'dependency version drift')
    return save('prepare',stamp(state='PASS',binding_sha256=reg['binding_sha256'],data=data_pins(),
        labels_sha256=sha(WORK/'labels.json'),code=s6.verify_code(d.CODE,d.ANATOMY),deps_tree_sha256=dep['deps_tree_sha256']))


def verify_prepared():
    reg=current(); pre=read('prepare')
    need(pre['binding_sha256']==reg['binding_sha256'] and pre['data']==data_pins(),'prepared data drift')
    need(pre['labels_sha256']==sha(WORK/'labels.json'),'labels seal drift')
    from tools import exp189_s6_port as s6
    need(pre['code']==s6.verify_code(d.CODE,d.ANATOMY),'code tree drift')
    dep=io.read_json(d.gold.RUNTIME/'prepare.json')
    need(d.gold.dir_tree(Path(dep['deps']))['tree_sha256']==pre['deps_tree_sha256'],'dependency tree drift')
    return reg


@contextlib.contextmanager
def serving(arm):
    # Scope all predecessor parameter substitutions; exact math/worker source unchanged.
    old=(g.WORK,g.BUNDLE,g.MANIFEST_SHA,g.port_module,d.gold.GOLD_DICOM)
    g.WORK=WORK; g.BUNDLE=BUNDLES[arm]; g.MANIFEST_SHA=MANIFESTS[arm]
    g.port_module=lambda _:port_module(arm)
    d.gold.GOLD_DICOM=WORK/'data'
    # g.infer uses train_series; our wrapper calls infer_studies on test_series explicitly.
    try: yield
    finally: g.WORK,g.BUNDLE,g.MANIFEST_SHA,g.port_module,d.gold.GOLD_DICOM=old


def infer(arm,ids,device,out):
    with serving(arm),g.execution_port('flip',device) as port,d.gold.worker_pythonpath(d.gold.RUNTIME/'deps'):
        original=port.diagnostic_source
        expected={u:sorted(r['SeriesInstanceUID'] for r in read('registration')['binding']['routes'][u]) for u in ids}
        def observed(source,destination):
            runtime=original(source,destination)
            target=runtime/'compact_infer.py'
            target.write_text(series_probe(target.read_text(encoding='utf-8'),expected,out/'series-proof'),encoding='utf-8',newline='\n')
            (runtime/'runtime_sha256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(runtime.glob('*.py'))}),encoding='utf-8')
            return runtime
        port.diagnostic_source=observed
        port.infer_studies([WORK/'data/test_series'/u for u in ids],bundle_dir=BUNDLES[arm],code_dir=d.CODE,
            anatomy_weights=d.ANATOMY,output_dir=out,source_dir=WORK/'contracts'/arm/'exp169_r7_src',
            workers=2,amp='auto',device=device,max_fallback_fraction=0.,expected_manifest_sha256=MANIFESTS[arm])
        result=g.check_controller(out,ids,'flip',device)
        verify_series(out,expected)
        return result


def series_probe(source,expected,destination):
    marker='            item = self.builder.build(uid)\n'
    need(source.count(marker)==1,'series probe source drift')
    addition=(f'            expected_oai = {expected!r}[uid]\n'
              '            retained_oai = sorted(b["series"] for b in item["blocks"])\n'
              '            if retained_oai != expected_oai:\n'
              '                raise ValueError("audited OAI native series missing or duplicate")\n'
              f'            proof_oai = Path({str(destination.resolve())!r})\n'
              '            proof_oai.mkdir(parents=True, exist_ok=True)\n'
              '            (proof_oai / (hashlib.sha256(uid.encode()).hexdigest()+".json")).write_text(json.dumps(retained_oai), encoding="utf-8")\n')
    return source.replace(marker,marker+addition)


def verify_series(out,expected):
    paths=list((out/'series-proof').glob('*.json'))
    need(len(paths)==len(expected),'series proof count differs')
    for u,series in expected.items():
        need(io.read_json(out/'series-proof'/(hashlib.sha256(u.encode()).hexdigest()+'.json'))==series,'retained series differ')


def verify_pilot_smoke(reg):
    old=pilot_registration(); smoke=io.read_json(PILOT/'cpu-smoke.json')
    need(smoke['state']=='PASS_REAL_ONE_STUDY_PARITY' and
         smoke['binding_sha256']==old['binding_sha256']==reg['binding']['pilot_binding_sha256'],
         'pilot CPU oracle binding drift')
    for name,h in smoke['files'].items(): need(sha(PILOT/name)==h,'pilot CPU oracle artifact drift')
    pre=io.read_json(PILOT/'prepare.json'); uid=reg['binding']['smoke_id']
    prefix='test_series/'+uid+'/'
    selected={n:h for n,h in pre['data'].items() if n.startswith(prefix)}
    need(selected,'pilot CPU data absent')
    for name,h in selected.items(): need(sha(WORK/'data'/name)==h,'pilot CPU input bytes differ')
    return smoke


def smoke():
    import numpy as np
    g.c.cpu_only(); reg=verify_prepared(); ids=[reg['binding']['smoke_id']]; evidence={}
    verify_pilot_smoke(reg)
    for arm in ARMS:
        # Fresh full-recipe CPU inference, compared with the sealed independent
        # predecessor CPU execution from v1. No old receipt is rewritten.
        start=time.perf_counter(); out=WORK/'cpu-smoke'/arm
        v,worker=infer(arm,ids,'cpu',out)
        with pilot.serving(arm):
            ref,ref_worker=g.check_controller(PILOT/'cpu-reference'/arm,ids,'flip','cpu')
        need(worker['torch']==ref_worker['torch'],'CPU reference software differs')
        need(np.array_equal(v,ref),'real CPU parity differs')
        evidence[arm]=dict(exact=True,max_abs_delta=float(np.max(np.abs(v-ref))),seconds=time.perf_counter()-start,
                           cuda_initialized=worker['cuda_initialized'],amp=worker['amp'])
    files={p.relative_to(WORK).as_posix():sha(p) for p in (WORK/'cpu-smoke').rglob('*')
           if p.is_file() and (p.name in ('predictions.csv','submission_receipt.json','invocation.json','provenance.json') or p.parent.name=='series-proof')}
    g.c.cpu_only()
    return save('cpu-smoke',stamp(state='PASS_REAL_ONE_STUDY_PARITY',binding_sha256=reg['binding_sha256'],
                                 studies=1,arms=evidence,files=files,oracle_sha256=sha(PILOT/'cpu-smoke.json'),
                                 limitation='Fresh CPU eager FP32 vs authenticated frozen oracle; CUDA unproved'))


def values(v,n):
    return g.c.check_values(v,n)


def outputs(common,r7,r6):
    import numpy as np
    n=len(r7); r7=values(r7,n); r6=values(r6,n)
    result=dict(R7flip=r7,R6flip=r6)
    if common is None: return result
    need(set(common)==set(COMMON),'incomplete common members')
    common={k:values(v,n) for k,v in common.items()}
    rank=g.rank
    rr=.65*rank(common['R'])+.35*rank(common['R224'])
    own=rank((17*rank(common['EXP143v2'])+17*rank(common['EXP155'])+12*rank(common['Q_A5q']))/46)
    result.update(HB167=.85*rank(.5*rr+.5*rank(r7))+.15*own,
                  replacement=.85*rank(.5*rr+.5*rank(r6))+.15*own)
    oracle=d.formula.blend([str(i) for i in range(n)],*[common[k] for k in ('R','R224')],r7,*[common[k] for k in ('EXP143v2','EXP155','Q_A5q')])
    need(np.array_equal(result['HB167'],oracle),'embedded blend mismatch')
    return result


def bootstrap(y,arrays,eligible,draws=None):
    import numpy as np
    n=len(y); count=RULES['draws'] if draws is None else draws
    rng=np.random.default_rng(RULES['seed']); weights=[]; attempts=0
    while len(weights)<count:
        need(attempts<RULES['max_attempts'],'bootstrap degeneracy cap')
        w=np.bincount(rng.integers(0,n,n),minlength=n); attempts+=1
        if any(w[y[:,j]==1].sum()==0 or w[y[:,j]==0].sum()==0 for j in eligible): continue
        weights.append(w)
    weights=np.asarray(weights,float); boot=np.empty((len(arrays),count,len(eligible)))
    for q,j in enumerate(eligible):
        pos=y[:,j]==1; neg=y[:,j]==0; wp=weights[:,pos]; wn=weights[:,neg]
        for k,v in enumerate(arrays):
            a=v[pos,j,None]; b=v[neg,j][None,:]
            pairs=(a>b).astype(float)+.5*(a==b)
            boot[k,:,q]=np.sum((wp@pairs)*wn,axis=1)/(wp.sum(1)*wn.sum(1))
    return boot,weights,attempts


def metrics(y,pred,contract,draws=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    need(label_contract(y)==contract,'frozen target eligibility/label drift')
    names=list(pred); arrays=[values(pred[k],len(y)) for k in names]; eligible=contract['eligible']
    auc=np.array([[roc_auc_score(y[np.isfinite(y[:,j]),j],v[np.isfinite(y[:,j]),j]) for j in eligible] for v in arrays])
    primary=[j for j in MACRO7 if j in eligible]
    primary_boot,weights,attempts=bootstrap(y,arrays,primary,draws)
    boot=np.empty((len(arrays),len(weights),len(eligible)))
    secondary_attempts={}
    for q,j in enumerate(eligible):
        if j in primary: boot[:,:,q]=primary_boot[:,:,primary.index(j)]
        else:
            secondary,_,tries=bootstrap(y,arrays,[j],draws)
            boot[:,:,q]=secondary[:,:,0]; secondary_attempts[str(j)]=tries
    pairs=[('R6flip','R7flip')]+([('replacement','HB167')] if 'replacement' in pred else [])
    result={}
    for a,b in pairs:
        delta=auc[names.index(a)]-auc[names.index(b)]; bd=boot[names.index(a)]-boot[names.index(b)]
        macro=None; weak=None
        if set(MACRO7)<=set(eligible):
            ix=[eligible.index(j) for j in MACRO7]
            macro=g.metric(delta[ix].mean(),bd[:,ix].mean(1))
        if set(RULES['weak_four'])<=set(eligible):
            ix=[eligible.index(j) for j in RULES['weak_four']]; weak=g.metric(delta[ix].mean(),bd[:,ix].mean(1))
        status='FAIL' if (macro and macro['delta']<RULES['macro_floor']) or (weak and weak['delta']<RULES['weak_four_floor']) else 'INCOMPLETE' if weak is None or macro is None else 'PASS'
        result[a+'-minus-'+b]=dict(macro_available=macro,weak_four=weak,point_floors=status,
                                   per_target_delta=dict(zip(map(str,eligible),delta.tolist())),
                                   per_target={str(j):g.metric(delta[q],bd[:,q]) for q,j in enumerate(eligible)})
    return dict(comparisons=result,eligible=eligible,macro_targets=MACRO7,valid_draws=len(weights),attempts=attempts,
                secondary_attempts=secondary_attempts,bootstrap_scope='joint fixed-primary; secondary targets individually',
                per_target_auc={name:dict(zip(map(str,eligible),auc[k].tolist())) for k,name in enumerate(names)},
                descriptive_status='AVAILABLE_DESCRIPTIVE_ONLY' if 'replacement' in pred else 'HOLD_MISSING_AUTHENTICATED_COMMON_OAI_PREDICTIONS',
                decision='HOLD_PENDING_INDEPENDENT_RESULT_REVIEW',qualified_for_submission=False)


def replay():
    import numpy as np
    g.c.cpu_only(); reg=current(); contracts=read('labels'); results={}
    # Exercise all cohort readout paths on real labels with synthetic identical
    # arms. No new model predictions or efficacy results are read here.
    for name,key in [('pilot','pilot_ids'),('confirmation','ids'),('pooled','pooled_ids')]:
        y=panel.outcomes(reg['binding'][key]); v=np.tile(np.linspace(.1,.9,len(y))[:,None],(1,12))
        result=metrics(y,outputs(None,v,v),contracts[name])
        for x in result['comparisons'].values():
            need(x['macro_available'] is not None and x['macro_available']['ci95']==[0.,0.],
                 'identity control differs or fixed macro7 unavailable')
        results[name]=dict(draws=result['valid_draws'],studies=len(y))
    old=authenticated_pilot(reg)
    pilot_result=metrics(panel.outcomes(reg['binding']['pilot_ids']),
                         {a:old['arms'][a]['predictions'] for a in ARMS},contracts['pilot'])
    verify_frozen_pilot(pilot_result)
    return save('replay',stamp(state='PASS_REAL_LABEL_IDENTITY_CONTROL',binding_sha256=reg['binding_sha256'],
                              cohorts=results,model_efficacy_measured=False,
                              authenticated_frozen_pilot_reproduced=True,pilot_evaluation_sha256=sha(PILOT/'evaluation.json')))


def validate():
    g.c.cpu_only(); reg=verify_prepared(); evidence={}
    for name,state in [('cpu-smoke','PASS_REAL_ONE_STUDY_PARITY'),('replay','PASS_REAL_LABEL_IDENTITY_CONTROL')]:
        value=read(name); need(value['state']==state and value['binding_sha256']==reg['binding_sha256'],'stale '+name)
        evidence[name]=sha(WORK/(name+'.json'))
    for p,h in read('cpu-smoke')['files'].items(): need(sha(WORK/p)==h,'CPU artifact drift')
    evidence['prepare']=sha(WORK/'prepare.json')
    g.gpu_queue.validate_plan(read('queue'))
    need(read('queue')==[dict(tag=TAG,command=launch_command())],'queue argv drift')
    return save('validate',stamp(state='CPU_VALIDATED',binding_sha256=reg['binding_sha256'],evidence=evidence))


def check_review(review,reg,validation):
    need(review.get('verdict')=='PASS' and review.get('reviewer') not in (None,'',BUILDER) and
         review.get('builder')==BUILDER and review.get('tag')==TAG and review.get('execution_authority')=='B' and
         review.get('binding_sha256')==reg['binding_sha256'] and review.get('validation_sha256')==validation,
         'independent bound PASS required')


def review_ready():
    reg=current(); valid=read('validate')
    need(valid['state']=='CPU_VALIDATED' and valid['binding_sha256']==reg['binding_sha256'],'validation drift')
    for name,h in valid['evidence'].items(): need(sha(WORK/(name+'.json'))==h,'validation artifact drift')
    for p,h in read('cpu-smoke')['files'].items(): need(sha(WORK/p)==h,'smoke evidence drift')
    verify_pilot_smoke(reg)
    need(read('queue')==[dict(tag=TAG,command=launch_command())],'queue argv drift')
    check_review(read('gpu-review'),reg,sha(WORK/'validate.json'))
    return reg


def launch_check():
    g.c.cpu_only(); reg=review_ready(); verify_prepared()
    return stamp(state='PASS_NO_GPU_ADMISSION',binding_sha256=reg['binding_sha256'],command=launch_command())


def configure(arm):
    need(arm in ARMS,'unknown GPU arm')
    t.WORK=WORK/'gpu'/arm; t.TAG=TAG; t.__file__=str(SCRIPT)
    t.launch_command=launch_command
    t.monitor=types.SimpleNamespace(**dict(vars(d._MONITOR),_cleanup=d.cleanup_owned))
    return t.WORK


def worker():
    need(os.environ.get('HB164_CONTROLLER_PID')==str(os.getppid()),'supervised worker only')
    reg=review_ready(); arm=os.environ['R6_OAI_ARM']; work=configure(arm); desktop=t.lane_admission()
    need('4070 Ti SUPER' in desktop['device_name'],'4070 required')
    os.environ['HB153_MEASURE_STATUS']=str(work/'worker-status.json')
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT),os.environ.get('PYTHONPATH','')])
    start=time.perf_counter(); ids=reg['binding']['ids']; v,rc=infer(arm,ids,'cuda',work/'output')
    measured=io.read_json(work/'worker-status.json')
    value=stamp(state='COMPLETE',binding_sha256=reg['binding_sha256'],done=len(ids),ids=ids,
                predictions=v.tolist(),rows=measured['rows'],runtime=measured['runtime'],device=measured['device'],
                seconds=time.perf_counter()-start,torch=rc['torch'])
    io.journal(work/'worker-status.json',value,work)
    return 0


def check_measurement(v,reg,arm,desktop,telemetry):
    import math
    ids=reg['binding']['ids']; rt=v['runtime']
    need(v['state']=='COMPLETE' and v['binding_sha256']==reg['binding_sha256'] and v['ids']==ids and v['done']==len(ids)
         and [r['study'] for r in v['rows']]==ids and telemetry['complete'] and telemetry['memory_failure'] is None,'incomplete measurement')
    values(v['predictions'],len(ids))
    need(rt['models']==[f'E_lrprior_s0_f{i}.pt' for i in range(4)] and rt['models2']==[] and rt['flip_tta'] is True
         and rt['workers']==2 and rt['size']==384 and rt['crop']==352 and rt['views']=='v4' and rt['slices']==3,'runtime recipe drift')
    for row in v['rows']:
        d.measure.window_gate(row['windows'],row['batch_windows'])
        need(math.isfinite(row['gpu_seconds']) and row['gpu_seconds']>0 and 0<row['peak_allocated_bytes']<=row['peak_reserved_bytes'],'invalid timing/memory')
    d.r7.memory_admission(desktop,[r['peak_reserved_bytes'] for r in v['rows']],telemetry['peak_gpu_mib']*2**20)


def run():
    reg=review_ready(); verify_prepared()
    need(not (WORK/'attempt.json').exists(),'previous attempt preserved; no automatic retry')
    attempt=stamp(state='RUNNING',binding_sha256=reg['binding_sha256'],arms={}); save('attempt',attempt)
    try:
        uuid=None
        for arm in ARMS:
            verify_prepared()
            work=configure(arm); desktop=t.lane_admission()
            need('4070 Ti SUPER' in desktop['device_name'],'4070 required')
            admit_memory(reg,arm,desktop)
            if uuid is None: uuid=desktop['uuid']
            need(uuid==desktop['uuid'],'paired device mismatch')
            need(not work.exists(),'previous output preserved'); work.mkdir(parents=True)
            os.environ['R6_OAI_ARM']=arm
            v,telemetry=t.supervise(desktop); v=d.canonicalize_rows(v,reg['binding']['ids'])
            check_measurement(v,reg,arm,desktop,telemetry)
            v.update(desktop=desktop,telemetry=telemetry); attempt['arms'][arm]=v; save('attempt',attempt)
        need([(x['study'],x['windows'],x['batch_windows']) for x in attempt['arms'][ARMS[0]]['rows']]==
             [(x['study'],x['windows'],x['batch_windows']) for x in attempt['arms'][ARMS[1]]['rows']],'paired windows differ')
        verify_prepared()
        attempt['state']='COMPLETE_PENDING_RESULT_REVIEW'; save('attempt',attempt)
        files={p.relative_to(WORK).as_posix():sha(p) for p in (WORK/'gpu').rglob('*') if p.is_file() and
               (p.name in ('supervision.json','worker-status.json','submission_receipt.json','submission.csv','predictions.csv','provenance.json','invocation.json') or p.parent.name=='series-proof')}
        return save('measurement',stamp(state=attempt['state'],binding_sha256=reg['binding_sha256'],attempt_sha256=sha(WORK/'attempt.json'),
                                         review_sha256=sha(WORK/'gpu-review.json'),files=files))
    except BaseException as exc:
        attempt.update(state='HOLD',error=f'{type(exc).__name__}: {exc}'); save('attempt',attempt); raise


def authenticated_pilot(reg):
    """Read cached predictions only AFTER all registered hashes pass."""
    import numpy as np
    from tools.job_runtime import job_status,LOGS
    old=pilot_registration()
    need(job_status(LOGS,pilot.TAG,expected_command=pilot.launch_command(),tail=False)['state']=='SUCCESS',
         'pilot job not successfully complete')
    m=io.read_json(PILOT/'measurement.json')
    need(m['attempt_sha256']==sha(PILOT/'attempt.json') and m['review_sha256']==sha(PILOT/'gpu-review.json'),
         'pilot attempt/review seal drift')
    for name,h in m['files'].items(): need(sha(PILOT/name)==h,'pilot prediction artifact drift')
    aa=io.read_json(PILOT/'attempt.json')
    need(aa['state']==m['state']=='COMPLETE_PENDING_RESULT_REVIEW' and
         aa['binding_sha256']==m['binding_sha256']==old['binding_sha256']==reg['binding']['pilot_binding_sha256'],
         'pilot completion/binding drift')
    need(old['binding']['identities']==reg['binding']['identities'],'pilot model identity mismatch')
    need(set(aa['arms'])==set(ARMS),'pilot pair incomplete')
    for arm,v in aa['arms'].items():
        pilot.check_measurement(v,old,arm,v['desktop'],v['telemetry'])
        sup=io.read_json(PILOT/'gpu'/arm/'supervision.json')
        need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[],'pilot supervision failed')
        with pilot.serving(arm): actual,_=g.check_controller(PILOT/'gpu'/arm/'output',old['binding']['ids'],'flip','cuda')
        pilot.verify_series(PILOT/'gpu'/arm/'output',{u:sorted(r['SeriesInstanceUID'] for r in old['binding']['routes'][u]) for u in old['binding']['ids']})
        need(np.array_equal(actual,v['predictions']),'pilot cached predictions differ')
    return aa


def cohort_results(reg,old,attempt,contracts,draws=None):
    import numpy as np
    b=reg['binding']; pids=b['pilot_ids']; cids=b['ids']; allids=b['pooled_ids']
    need(not set(pids)&set(cids) and sorted(pids+cids)==allids and len(set(allids))==len(allids),'cohort overlap/union drift')
    for arm in ARMS:
        a,z=old['arms'][arm],attempt['arms'][arm]
        need(a['ids']==pids and z['ids']==cids,'cohort prediction order drift')
        need(a['desktop']['uuid']==z['desktop']['uuid'] and a['runtime']==z['runtime'] and a['torch']==z['torch'],
             'pilot/confirmation device or software recipe mismatch')
    index={u:i for i,u in enumerate(pids+cids)}
    pooled={arm:np.concatenate([values(old['arms'][arm]['predictions'],len(pids)),
                               values(attempt['arms'][arm]['predictions'],len(cids))])[[index[u] for u in allids]] for arm in ARMS}
    results={}
    for name,ids,pred in [('pilot',pids,{a:old['arms'][a]['predictions'] for a in ARMS}),
                          ('confirmation',cids,{a:attempt['arms'][a]['predictions'] for a in ARMS}),('pooled',allids,pooled)]:
        results[name]=metrics(panel.outcomes(ids),pred,contracts[name],draws=draws)
    return results


def verify_frozen_pilot(result):
    frozen=io.read_json(PILOT/'evaluation.json')
    need(frozen['measurement_sha256']==sha(PILOT/'measurement.json'),'pilot evaluation binding drift')
    for key in ('macro_available','weak_four','point_floors','per_target_delta'):
        need(result['comparisons']['R6flip-minus-R7flip'][key]==
             frozen['result']['comparisons']['R6flip-minus-R7flip'][key],'frozen pilot readout differs')
    return frozen


def evaluate():
    import numpy as np
    from tools.job_runtime import job_status,LOGS
    g.c.cpu_only(); reg=review_ready(); verify_prepared()
    need(job_status(LOGS,TAG,expected_command=launch_command(),tail=False)['state']=='SUCCESS','successful completed exclusive job required')
    m=read('measurement'); attempt=read('attempt')
    need(m['state']==attempt['state']=='COMPLETE_PENDING_RESULT_REVIEW' and m['binding_sha256']==reg['binding_sha256'] and
         m['attempt_sha256']==sha(WORK/'attempt.json') and m['review_sha256']==sha(WORK/'gpu-review.json'),'measurement seal drift')
    for p,h in m['files'].items(): need(sha(WORK/p)==h,'GPU output drift')
    need(set(attempt['arms'])==set(ARMS),'incomplete pair')
    for arm,v in attempt['arms'].items():
        check_measurement(v,reg,arm,v['desktop'],v['telemetry'])
        sup=io.read_json(WORK/'gpu'/arm/'supervision.json')
        need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[],'supervision failed')
        with serving(arm): actual,_=g.check_controller(WORK/'gpu'/arm/'output',reg['binding']['ids'],'flip','cuda')
        verify_series(WORK/'gpu'/arm/'output',{u:sorted(r['SeriesInstanceUID'] for r in reg['binding']['routes'][u]) for u in reg['binding']['ids']})
        need(np.array_equal(actual,v['predictions']),'prediction seal differs')
    old=authenticated_pilot(reg)
    result=cohort_results(reg,old,attempt,read('labels'))
    frozen=verify_frozen_pilot(result['pilot'])
    return save('evaluation',stamp(state='COMPLETE_PENDING_RESULT_REVIEW',binding_sha256=reg['binding_sha256'],
                                   measurement_sha256=sha(WORK/'measurement.json'),result=result,
                                   frozen_pilot_evaluation=frozen,primary_cohort='confirmation',
                                   pooled_status='DESCRIPTIVE_PRECISION_UPDATE_NOT_FRESH_VALIDATION',
                                   ancestry=reg['binding']['ancestry'],decision='HOLD_PENDING_INDEPENDENT_RESULT_REVIEW'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['register','prepare','smoke','replay','validate','launch-check','run','_worker','evaluate'])
    ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args()
    need(not args.dry_run or args.action=='prepare','dry-run only valid for prepare')
    if args.action not in ('run','_worker'): os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.action=='prepare': result=prepare(args.dry_run)
    else: result=globals()[{'launch-check':'launch_check','_worker':'worker'}.get(args.action,args.action)]()
    # Never dump binding, patient rows, predictions or labels to the terminal.
    if isinstance(result,dict): print(json.dumps({k:v for k,v in result.items() if k in ('state','studies','draws','binding_sha256','command')},indent=2))


if __name__=='__main__': main()
