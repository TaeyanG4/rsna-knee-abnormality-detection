"""H197 INF3 release builder v1.0.3, one fixed H189 child; CPU cached replay only.

Changelog: add inverse-cost Q pooling at the final authenticated reader; preserve
the producer and legacy receipt authentication. No forward, launch or sender.
v1.0.1: direct frozen-cache authentication avoids widening the completed bridge's
historical dynamic sys.modules closure. Original draft retained in draft-v100.
v1.0.2: avoid shadowing the parent loader with a local replay array; draft-v101
and its failed replay remain preserved. Kernel scientific payload is unchanged.
v1.0.3: pin the existing native completion archive verifier used by the gate.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import hb145_output_io as output_io
VERSION='1.1.46-r6w958inf3'
KEY='R6W958-INF3-V1'
SLUG='taeyangg4/knee-lb-r6w958inf3-v1'
BUILDER='/root/candidate_scout'
BASE=ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q'
OUT=BASE.parent/('v'+VERSION)
SCRATCH=ROOT/'.local/scratch/slot-b/batch5-1018/inf3-release'
BRIDGE=SCRATCH.parent/'inf3-bridge'
GATE=ROOT/'.local/state/gates/knee-lb-r6w958q-v1.json'
PARENT_SHA='fcb2df01b47d5cc9b83c8a11e35ceca10d19ba5884e2b8fd40aee13867f3e8c9'
META_SHA='f794876c887318a85887458dee54c0dffa1ca89edf5823db7c1242f17c15dbf9'
Q_SHA='39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc'
REG_SHA='0f6a2e52906b549ffa7b71ea5bbcbe8e9b98e4e3d19137c4feec1f2458df9aa2'
RESULT_SHA='8f682925cc0bf360126f9c0777807a5d2a7bb59c89aa7217542fa62bc083fe56'
BRIDGE_SOURCE_SHA='198a4a7482814804e182115bd4196c7d978b0a77163bcf3d6a87a3984fc87f9b'
REVIEW_SHA='d50c5c26214754e92aa824594804037343e43d8d50a93de957ff1c25f8d27022'
SLATE_SHA='9ee3af029b9c4a2a5663f325eb71b4c2c487909e7b933c0939cfbadba651b7dc'
FAMILIES=('l2','dino','convnext')
SOURCE_FILES=('tools/c85r7_r6w958inf3_v1146.py','tools/strict_variant_gate_inf3_v1.py',
              'tests/test_hb197_inf3_release_v1.py','tools/hb145_output_io.py','tools/native_codex_dispatch_record_v1.py')


def need(ok,message):
    if not ok: raise ValueError(message)


def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def encoded(obj):return (json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def write(path,data):
    path=Path(path);data=data.encode() if isinstance(data,str) else data
    need(path.resolve().is_relative_to(SCRATCH.resolve()) or path.resolve().is_relative_to(OUT.resolve()),'output scope')
    if path.exists():
        need(path.read_bytes()==data,'immutable output drift: '+rel(path));return
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    output_io.retry(lambda:tmp.write_bytes(data));output_io.replace(tmp,path)


def replace(data,before,after):
    if isinstance(data,bytes):return replace(data.decode(),before,after).encode()
    need(data.count(before)==1,'exact patch anchor: '+before[:80])
    return data.replace(before,after)


def unpack(nb):
    tree=ast.parse(''.join(nb['cells'][2]['source']))
    a={n.targets[0].id:ast.literal_eval(n.value) for n in ast.walk(tree)
       if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
       and n.targets[0].id in ('BLOB','PINS')}
    blob=base64.b64decode(a['BLOB'])
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=z.namelist();need(len(names)==len(set(names))==31,'exact31 members')
        need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names),'unsafe member')
        files={n:z.read(n) for n in names}
    need(a['PINS']=={n:digest(v) for n,v in files.items()},'embedded source pins')
    return files,blob,a


def authority():
    for path,expected in [(BRIDGE/'registration.json',REG_SHA),(BRIDGE/'results.json',RESULT_SHA),
                          (ROOT/'tools/hb197_inf3_bridge_v1.py',BRIDGE_SOURCE_SHA),
                          (BRIDGE/'RESULT-REVIEW.md',REVIEW_SHA),(SCRATCH.parent/'SLATE-REVIEW.md',SLATE_SHA)]:
        need(sha(path)==expected,'frozen authority drift: '+rel(path))
    for path in (BRIDGE/'RESULT-REVIEW.md',SCRATCH.parent/'SLATE-REVIEW.md'):
        r=read(str(path)+'.receipt.json')
        need(r.get('schema')=='dispatch-terminal/v1' and r.get('state')=='SUCCESS' and r.get('exit_code')==0 and
             r.get('agent_id')=='/root/k1_result_review' and r.get('agent_id')!=BUILDER and
             r.get('host')=='Codex app native subagent' and r.get('report_sha256')==sha(path), 'genuine separate prereview')
        need(sha(r['completion_source'])==r['completion_evidence_sha256'],'completion receipt evidence drift')
    return read(BRIDGE/'registration.json')


def parent(replay=False):
    need(sha(BASE/'kernel/release.ipynb')==PARENT_SHA and sha(BASE/'kernel/kernel-metadata.json')==META_SHA,'H189 pin')
    if replay:
        from tools import strict_variant_gate_r224_v145 as gate
        p=gate.verify_receipt(GATE,gate.g.Inputs(ROOT))
        need(p['kernel_id']=='taeyangg4/knee-lb-r6w958q-v1','parent admission slug')
    nb=read(BASE/'kernel/release.ipynb');files,blob,a=unpack(nb)
    proof=authority();need({n:digest(v) for n,v in files.items()}==proof['parent_embedded'],'authenticated full parent inventory')
    need(digest(files['q-a5q-manifest.json'])==Q_SHA,'parent Q manifest')
    return nb,files,blob,a


def costs():
    proof=authority();entries=json.loads(parent()[1]['q-a5q-manifest.json'])['entries']
    members=proof['members'];need([m['identity'] for m in members]==entries,'cost/member identity order')
    result={f:[] for f in FAMILIES}
    for m in members:
        w=m['actual_pos_weight'];need(len(w)==12 and all(type(x) in (int,float) and x>=1 and x<float('inf') for x in w),'cost values')
        result[m['identity']['family']].append(w)
    need([len(result[f]) for f in FAMILIES]==[10,5,5],'cost family coverage')
    return result


POOL_SOURCE='''
def inf3_aggregate(scores, np):
    """Checkpoint costs after FP32 TTA; one rank/N over full successful cohort."""
    need(set(scores) == set(INF3_COSTS) == {'l2', 'dino', 'convnext'}, 'INF3 families')
    means, sizes = [], set()
    for family in ('l2', 'dino', 'convnext'):
        a = np.asarray(scores[family])
        w = np.asarray(INF3_COSTS[family], dtype=np.float64)
        count = 10 if family == 'l2' else 5
        need(a.ndim == 3 and a.shape[0] == count and a.shape[2] == 12 and a.shape[1] > 0 and
             a.dtype == np.float32 and w.shape == (count, 12), 'INF3 shape/dtype/order')
        need(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() and
             np.isfinite(w).all() and (w >= 1).all(), 'INF3 finite probabilities/costs')
        sizes.add(a.shape[1])
        corrected = np.stack([x.astype(np.float64) / (y - (y - 1) * x.astype(np.float64))
                              for x, y in zip(a, w)])
        if family == 'l2': corrected = (corrected[:5] + corrected[5:]) / 2
        means.append(corrected.mean(0))
    need(len(sizes) == 1, 'INF3 family cohort')
    return rank(sum(means) / 3)


'''


def patched(original):
    files=dict(original);s=files['exp169_release_c85.py']
    s=replace(s,'def authenticated_a172(directory):',
        'INF3_COSTS = '+repr(costs())+'\n'+POOL_SOURCE+'def authenticated_a172(directory):')
    s=replace(s,'    q = np.full((len(ids), 12), .5)\n','    q = np.full((len(ids), 12), .5)\n    legacy_q = q.copy()  # original child receipt authentication only\n')
    s=replace(s,'            scored, families = a172.aggregate({name: value[:, good] for name, value in arrays.items()})\n',
        '            legacy_scored, families = a172.aggregate({name: value[:, good] for name, value in arrays.items()})\n'
        '            scored = inf3_aggregate({name: value[:, good] for name, value in arrays.items()}, np)\n')
    s=replace(s,'                qfail.pop(uid, None)\n','                qfail.pop(uid, None)\n'
        '            for uid, row in zip(np.asarray(qids)[good], legacy_scored):\n                legacy_q[ids.index(uid)] = row\n')
    s=replace(s,"checks['bundle_rank_digest'] = array_sha(scored)","checks['bundle_rank_digest'] = array_sha(legacy_scored)")
    s=replace(s,"expected = a172.final_values(tv[[tids.index(uid) for uid in ids]], q, 'A172')",
        "expected = a172.final_values(tv[[tids.index(uid) for uid in ids]], legacy_q, 'A172')")
    files['exp169_release_c85.py']=s
    wrapper='exp169_release_c85r7codec.py'
    for before,after in [('VERSION = "1.1.39-r6w958q"','VERSION = "'+VERSION+'"'),
                         ('taeyangg4/knee-lb-r6w958q-v1',SLUG),('v1.1.39-r6w958q','v'+VERSION)]:
        files[wrapper]=replace(files[wrapper],before,after)
    # Rewrite original digest literals from the initial draft until dependency
    # closure stabilizes. No runtime function changes accompany these pin edits.
    initial=dict(files)
    for _ in range(10):
        changes={digest(v):digest(files[n]) for n,v in original.items() if files[n]!=v}
        updated={}
        for n,v in initial.items():
            for old,new in changes.items():v=v.replace(old.encode(),new.encode())
            updated[n]=v
        if files==updated:break
        files=updated
    else:raise ValueError('source pin cycle')
    return files


def payload():
    nb,original,old_blob,a=parent();files=patched(original);buffer=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as src,zipfile.ZipFile(buffer,'w') as dst:
        for info in src.infolist():dst.writestr(info,files[info.filename])
    blob=buffer.getvalue();s=''.join(nb['cells'][2]['source'])
    s=replace(s,repr(a['BLOB']),repr(base64.b64encode(blob).decode()))
    s=replace(s,'PINS = '+repr(a['PINS']),'PINS = '+repr({n:digest(v) for n,v in files.items()}))
    s=replace(s,repr(digest(old_blob)),repr(digest(blob)))
    nb['cells'][2]['source']=s.splitlines(True)
    need('FAST = True' in ''.join(nb['cells'][0]['source']),'parent FAST policy drift')
    nb['cells'][0]['source']=['FAST = False\n']
    meta=read(BASE/'kernel/kernel-metadata.json');meta.update(id=SLUG,title=SLUG.split('/')[1])
    return nb,meta,files,blob


def scope(nb,meta):
    expected,em,files,blob=payload();need(nb==expected and meta==em,'forbidden notebook/metadata diff')
    af,ab,_=unpack(nb);need(af==files and ab==blob,'forbidden embedded diff')
    original=parent()[1]
    changed={n for n,v in files.items() if original[n]!=v}
    need(changed=={'exp169_release_c85.py','exp169_release_c85r7.py','exp169_release_c85r7codec.py'},'unexpected embedded closure')
    replacements={digest(files[n]):digest(original[n]) for n in changed}
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
        if name not in changed:need(data==original[name],'untouched member drift')
        if name not in changed:continue
        before=ast.parse(original[name]);normalized=data
        for new,old in replacements.items():normalized=normalized.replace(new.encode(),old.encode())
        after=ast.parse(normalized)
        imports=lambda t:[ast.dump(n) for n in ast.walk(t) if isinstance(n,(ast.Import,ast.ImportFrom))]
        need(imports(before)==imports(after),'runtime import changed')
        funcs=lambda t:{n.name:ast.dump(n) for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        a,z=funcs(before),funcs(after)
        expected_changes={'authenticated_a172'} if name=='exp169_release_c85.py' else set()
        need({n for n in a if a[n]!=z.get(n)}==expected_changes,'forbidden runtime function diff: '+name)
        need(set(z)-set(a)==({'inf3_aggregate'} if expected_changes else set()),'extra runtime function')
    return files,blob


def sources():return {p:sha(ROOT/p) for p in SOURCE_FILES}


def registration():
    proof=authority();binding={p:h for p,h in proof['bindings'].items()}
    for p,h in binding.items():need(sha(ROOT/p)==h,'frozen bridge dependency drift: '+p)
    paths=[SCRATCH/'SPEC.md',BRIDGE/'SPEC.md',BRIDGE/'registration.json',BRIDGE/'results.json',BRIDGE/'BUILD-REPORT.md',
           BRIDGE/'RESULT-REVIEW.md',Path(str(BRIDGE/'RESULT-REVIEW.md')+'.receipt.json'),
           SCRATCH.parent/'SLATE-REVIEW.md',Path(str(SCRATCH.parent/'SLATE-REVIEW.md')+'.receipt.json'),GATE]
    binding.update({rel(p):sha(p) for p in paths});binding.update(sources())
    return dict(schema='h197-release-registration/v1',candidate=KEY,version=VERSION,slug=SLUG,builder=BUILDER,
        parent_kernel_sha256=PARENT_SHA,parent_metadata_sha256=META_SHA,bindings=binding,source_sha256=sources(),
        costs=costs(),member_identities=[m['identity'] for m in proof['members']],targets=proof['targets'],
        formula='H189_F_Q3',full_visible_required=True,n32_required=True,fast=False,exposure='EXPOSED',clean_oof=False,
        scientific_result=rel(BRIDGE/'results.json'),scientific_release_gain_established=False,
        qualified_for_submission=False,transport_binding='deferred until actual finalized canonical integration')


def register():
    parent(replay=True);r=registration();write(SCRATCH/'registration.json',encoded(r))
    return dict(status='REGISTERED_BEFORE_BUILD',sha256=sha(SCRATCH/'registration.json'),bindings=len(r['bindings']))


def artifacts():
    nb,meta,files,blob=payload();scope(nb,meta);pnb,original,_,_=parent()
    inventory={n:dict(parent=digest(original[n]),candidate=digest(v),changed=original[n]!=v) for n,v in files.items()}
    def diff(a,b,name):return ''.join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='H189/'+name,tofile='H197/'+name))
    full='ALL31 EMBEDDED FILES; NO ADDITION OR REMOVAL\n'
    for n in files:
        full+=n+': '+inventory[n]['parent']+' -> '+inventory[n]['candidate']+'\n'+diff(original[n],files[n],n)
    notebook_diff=diff(encoded(pnb),encoded(nb),'release.ipynb')
    metadata_diff=diff(encoded(read(BASE/'kernel/kernel-metadata.json')),encoded(meta),'kernel-metadata.json')
    return nb,meta,files,blob,inventory,full,notebook_diff+metadata_diff


def prepare():
    need(read(SCRATCH/'registration.json')==registration(),'register exact source closure first')
    nb,meta,files,blob,inventory,full,nbdiff=artifacts()
    for name in ('kernel/release.ipynb','kernel/kernel-metadata.json'):
        write(SCRATCH/'parent-backup'/name,(BASE/name).read_bytes())
    write(SCRATCH/'parent-backup/admission.json',GATE.read_bytes())
    for name,data in [('kernel/release.ipynb',encoded(nb)),('kernel/kernel-metadata.json',encoded(meta)),
        ('embedded-diff.json',encoded(inventory)),('full-embedded-diff.txt',full),('notebook-metadata-diff.txt',nbdiff),
        ('SPEC.md',(SCRATCH/'SPEC.md').read_bytes()),('CHANGELOG.md','# '+VERSION+'\nOne H189 INF3 child. Post-TTA inverse-cost Q pooling only; all20 checkpoints/producer/manifest and downstream F unchanged. Original Q0 retained for original child telemetry. FAST=False full-visible/new-path policy. Three source/pin members changed;28 identical. All admissions HOLD.\n')]:write(OUT/name,data)
    for n,v in files.items():write(SCRATCH/'candidate-runtime'/n,v)
    rec=dict(schema='h197-release-build/v1',status='PREPARED_CPU_ONLY',candidate=KEY,version=VERSION,slug=SLUG,builder=BUILDER,
        registration_sha256=sha(SCRATCH/'registration.json'),kernel_sha256=sha(OUT/'kernel/release.ipynb'),metadata_sha256=sha(OUT/'kernel/kernel-metadata.json'),
        archive_sha256=digest(blob),source_sha256=sources(),all_embedded=inventory,unchanged=28,changed=[n for n,v in inventory.items() if v['changed']],
        qualified_for_submission=False)
    write(OUT/'build-receipt.json',encoded(rec));return validate()


def validate():
    need(read(SCRATCH/'registration.json')==registration(),'registered inputs/sources drift')
    nb,meta,files,blob,inventory,full,nbdiff=artifacts()
    for path,data in [('kernel/release.ipynb',encoded(nb)),('kernel/kernel-metadata.json',encoded(meta)),
        ('embedded-diff.json',encoded(inventory)),('full-embedded-diff.txt',full.encode()),('notebook-metadata-diff.txt',nbdiff.encode()),('SPEC.md',(SCRATCH/'SPEC.md').read_bytes())]:
        need((OUT/path).read_bytes()==data,'stage artifact drift: '+path)
    for n,v in files.items():need((SCRATCH/'candidate-runtime'/n).read_bytes()==v,'extracted runtime drift: '+n)
    r=read(OUT/'build-receipt.json')
    need(r['registration_sha256']==sha(SCRATCH/'registration.json') and r['source_sha256']==sources() and
         r['kernel_sha256']==sha(OUT/'kernel/release.ipynb') and r['metadata_sha256']==sha(OUT/'kernel/kernel-metadata.json') and
         r['all_embedded']==inventory and r['archive_sha256']==digest(blob) and r['qualified_for_submission'] is False,'build receipt drift')
    return dict(status='PASS_CPU_VALIDATE',candidate=KEY,kernel_sha256=r['kernel_sha256'],unchanged=28,changed=3,fast=False)


def runtime(files):
    """Execute only actual extracted pure reader/math definitions, no notebook."""
    from tools import exp169_release_c85 as base
    namespace=dict(vars(base));tree=ast.parse(files['exp169_release_c85.py'])
    names={'inf3_aggregate','authenticated_a172'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or
           isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INF3_COSTS' for t in n.targets)]
    need(len(nodes)==3,'exact extracted Q helper definitions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'extracted-inf3-c85','exec'),namespace)
    return namespace


def replay():
    from tools import hb197_inf3_bridge_v1 as bridge
    import numpy as np
    import pandas as pd
    validate();proof=read(BRIDGE/'registration.json')
    # Registration already rehashes every original bridge input. Loading only
    # this serving closure avoids a new OOF/label computation and does not alter
    # the frozen bridge's deliberately exact imported-module inventory.
    with np.load(bridge.COMMON/'predictions.npz',allow_pickle=False) as z:common={k:z[k].copy() for k in z.files}
    with np.load(bridge.COMMON/'q-leaves.npz',allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
    gids=common['ids'].tolist();creg=read(bridge.COMMON_REG)
    need(gids==raw['ids'].tolist()==creg['ids'] and len(gids)==len(set(gids))==58 and
         common['targets'].tolist()==raw['targets'].tolist()==proof['targets'],'frozen common roster/targets')
    need(digest(bridge.encoded(gids))==proof['gold_roster_sha256'],'frozen roster hash')
    leaves={f:raw[f] for f in FAMILIES};cost={f:np.asarray(v) for f,v in costs().items()};original=parent()[1]
    for f,a in leaves.items():need(a.dtype==np.float32 and a.shape==(10 if f=='l2' else 5,58,12),'real leaf layout')
    for name in ('R','R224','EXP143v2','EXP155','Q_A5q'):
        bridge.probabilities(common[name],58);mask=common['failed_'+name]
        need(mask.shape==(58,) and mask.dtype==bool and not mask.any(),'observed failure differs from fixed bridge')
    frame=pd.read_csv(bridge.R6/'predictions.csv',dtype={bridge.ml.UID:str},float_precision='round_trip')
    need(frame.columns.tolist()==[bridge.ml.UID,*proof['targets']],'frozen R6 target order')
    common['S_R7']=bridge.align(frame[bridge.ml.UID].tolist(),frame[proof['targets']].to_numpy(float),gids)
    files=unpack(read(OUT/'kernel/release.ipynb'))[0];ns=runtime(files)
    actual=ns['inf3_aggregate'](leaves,np);expected=bridge.serving_q3(leaves,cost)
    need(actual.tobytes()==expected.tobytes(),'actual extracted Q3 differs from reviewed bridge')
    q0,_=bridge.qparent.aggregate(leaves);need(np.array_equal(q0,common['Q_A5q']),'authentic Q0')
    pm=bridge.parent.math_functions(original);cm=bridge.parent.math_functions(files)
    args=[common[n] for n in bridge.NAMES]
    baseline=pm['blend'](gids,*args);noop=cm['blend'](gids,*args[:-1],q0)
    need(baseline.tobytes()==noop.tobytes() and pm['p'].csv_bytes(gids,baseline)==cm['p'].csv_bytes(gids,noop),'parent CSV/no-op')
    candidate=cm['blend'](gids,*args[:-1],actual)
    reference=pm['blend'](gids,*args[:-1],expected)
    need(candidate.tobytes()==reference.tobytes(),'unchanged downstream F')
    csvs={k:digest(cm['p'].csv_bytes(gids,v)) for k,v in [('F0',baseline),('F3',candidate)]}
    saved=read(BRIDGE/'results.json')
    need(csvs==saved['gold58']['csv_sha256'],'saved independently reviewed final CSV hash')
    for p,h in proof['bindings'].items():need(sha(ROOT/p)==h,'input changed during replay')
    result=dict(status='PASS_EXTRACTED_REAL58_REPLAY',candidate=KEY,kernel_sha256=sha(OUT/'kernel/release.ipynb'),
        rows=58,targets=12,models=20,leaf_dtype='float32',pool_dtype='float64',rank='average rank/N on full successful Q cohort',
        bridge_result_sha256=RESULT_SHA,registration_sha256=sha(SCRATCH/'registration.json'),source_sha256=sources(),
        exact_q3_bytes=True,exact_final_float64=True,parent_noop_float64_and_csv_exact=True,csv_sha256=csvs,
        q3_sha256=digest(actual.tobytes()),roster_sha256=proof['gold_roster_sha256'],observed_fallbacks=0,
        exposure='EXPOSED',new_metrics_computed=False,model_forward=False,cuda=False,qualified_for_submission=False)
    write(SCRATCH/'CPU-REPLAY.json',encoded(result));return result


def admit():
    r=validate();cpu=read(SCRATCH/'CPU-REPLAY.json')
    need(cpu['kernel_sha256']==r['kernel_sha256'] and cpu['exact_q3_bytes'] and cpu['exact_final_float64'],'CPU replay required')
    return dict(status='BUILD_READY_FOR_INDEPENDENT_REVIEW',kernel_sha256=r['kernel_sha256'],qualified_for_submission=False,
        hold=['different actual build/result reviewer','full visible and N32','CUDA/Linux/memory/codecs/fallback/runtime<=9h',
              'canonical sender integration/current pins/late recheck','duplicate and independent minimum review'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('register','prepare','validate','cpu-replay','admit'))
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.command in ('prepare','admit'):need(a.dry_run,'CPU dry-run required')
    result={'register':register,'prepare':prepare,'validate':validate,'cpu-replay':replay,'admit':admit}[a.command]()
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
