"""H-B-188 builder 1.0.0: Fracture joins admitted H-B-184 OA plus fluid R route.

Changelog 1.0.0: selectors [7,9] -> [7,9,11]; unchanged parent survivor policy.
Reuses strict archive/replay infrastructure; no frozen/launch/timing edits.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

# A present-empty value is required; PowerShell assignment of '' deletes it.
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_stackq_v1125 as infrastructure

old, fast, rbuild, gate = (infrastructure.old, infrastructure.fast,
                          infrastructure.rbuild, infrastructure.gate)
VERSION = '1.1.38-maxstackq'
SLUG = 'taeyangg4/knee-lb-maxstackq-v1'
BASE = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.35-stackoaq'
PARENT_GATE = ROOT / '.local/scratch/astra-b5/hb184-gate/gate-copy-v145'
RECEIPT = ROOT / '.local/state/gates/knee-lb-stackoaq-v1.json'
PARENT_KERNEL = '8200085ae28dcfcf5d2bc5c8cc95a6941f6d5a1e25d688e739ddae249a5eb506'
PARENT_RECEIPT = 'a50e77e66a0b096601029de2439ca2acc962f6892730321bdd4a465051d32d41'
OUT = old.BASE.parent / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/hb188-build/hb188'
BACKUP = ROOT / '.local/backups/hb188-build/hb188-parent'
SPEC = SCRATCH / 'SPEC.md'
PRE_REVIEW = SCRATCH / 'PRE-REVIEW.md'
HELPER, WRAPPER = infrastructure.HELPER, infrastructure.WRAPPER
DIFF = 'full-diff-vs-v1.1.35-stackoaq.txt'
FORMULA = ('0.85*r(P)+0.15*r(O); r_R=0.65*r(R384)+0.35*r(R224); '
           'P_t=rank(R7flip)_t for Lateral OA/PF OA when compact succeeds, else r_R_t; '
           'P_t=r_R_t for Effusion, Bakers Cyst and Fracture with R-only failure eligibility; other P=0.5*r_R+0.5*r(R7flip); '
           'O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46')
PROTECTED = tuple(range(11))
MEMBERS = ('R','R224','S_R7','EXP143v2','EXP155','Q_A5q')


def authenticated(stage, source, receipt, receipt_sha, kernel_sha):
    old.pin(receipt, receipt_sha)
    saved=old.read(receipt)
    old.need(saved['status']=='PASS' and saved['errors']==[] and saved['fixture_only'] is False,
             'not an admitted parent/reference')
    inv=gate.kernel(stage,gate.Inputs(ROOT))[2]
    old.need(inv==saved['stage_inventory'], 'admitted inventory mismatch')
    old.need(gate.sha(gate.encoded(inv['kernel_files']))==kernel_sha, 'kernel identity')
    old.need(gate.kernel(source,gate.Inputs(ROOT))[3]==gate.kernel(stage,gate.Inputs(ROOT))[3],
             'fixed source differs from admitted gate kernel')
    nb=old.read(source/'kernel/release.ipynb')
    files,blob,assignments=old.unpack(fast.restored(nb))
    old.need(len(files)==31,'embedded roster')
    return nb,files,blob,assignments


def parent():
    return authenticated(PARENT_GATE,BASE,RECEIPT,PARENT_RECEIPT,PARENT_KERNEL)


def pre_review():
    review=PRE_REVIEW.read_text(encoding='utf-8')
    old.need('Verdict: PASS for CPU-only build/tests/replay of SPEC.md v1.0.0' in review
             and old.digest(SPEC.read_bytes()) in review, 'independent pre-execution PASS required')
    return dict(spec_sha256=old.digest(SPEC.read_bytes()),
                review_path=PRE_REVIEW.relative_to(ROOT).as_posix(),
                review_sha256=old.digest(PRE_REVIEW.read_bytes()))


def registration():
    parent()
    return dict(item='H-B-188', version=VERSION, kernel_id=SLUG,
        purpose='Taeyang-requested max-stack bundle: admitted H-B-184 OA plus fluid route, add Fracture to R; ONE candidate',
        parent=BASE.relative_to(ROOT).as_posix(), parent_kernel_sha256=PARENT_KERNEL,
        parent_receipt_sha256=PARENT_RECEIPT,
        expected_public=.957, upside_public=.958, formula=FORMULA, routed_columns=[7,9,11],
        added_route=[11], unchanged_oa_columns=[5,6], protected_columns=list(PROTECTED), pre_execution=pre_review(),
        preregistration='register precedes prepare/tests/replay; no label-based evaluation or selection; roster authentication only, no printed labels',
        failure_rule='Exact fluid rule: independent success-only R384/R224 ranks, .65/.35 mix; single R survivor alone; both R failed => O only even if R7 succeeds; O failed => R only; both failed => .5. OA compact success uses compact ranks, compact failure uses combined R fallback. Own survivor weights and R224 >3% abort unchanged.',
        parent_clarification='H-B-184 on hidden-COMPLETE H-B-168: R7 compact, FULL flip, FAST, hard OA and fluid R, no compo15; unchanged own 17/17/12 EXP143v2/EXP155/Q_A5q and dataset sources.',
        public_reading=dict(reference_value=.957, expected=.957, upside=.958,
            rule='retain iff displayed >=0.957 AND >= H-B-184 displayed score when both read; missing comparison HOLD; either failed condition DROP',
            source='user-supplied readings only; no score polling'),
        timing=dict(source='user supplied H-B-168 hidden COMPLETE submission 57251559',
            inference_delta='none', child_measured_hours=None,
            limitation='H-B-168 completion is runtime basis; no new inference, unmeasured child duration.'),
        sources=['Taeyang 2026-10-16 max-stack request; Gold58 OA +.0047, Fracture +.0013 P approximately .83; motivation only, not additive gain evidence'],
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        runtime_qualification='CPU arithmetic replay only; no fresh model forward/CUDA/Linux qualification',
        gate_binding='HOLD: B slug-generic v1.4.5 gate with authentic dispatcher builder/reviewer receipts')


def decision(displayed_score, parent_displayed_score=None):
    for value in (displayed_score, parent_displayed_score):
        if value is not None:
            old.need(type(value) in (int,float) and math.isfinite(value)
                     and 0 <= value <= 1, 'finite displayed public score required')
    if displayed_score is not None and displayed_score < .957: return 'DROP'
    if displayed_score is None or parent_displayed_score is None: return 'HOLD'
    return 'RETAIN' if displayed_score >= parent_displayed_score else 'DROP'


def patched(original):
    files = dict(original)
    s = files[HELPER]
    line = next(x for x in s.decode().splitlines() if x.startswith('FORMULA = '))
    s = old.replace(s, line, 'FORMULA = ' + repr(FORMULA))
    # Exactly three assignment lines, each with LHS and RHS selector.
    old.need(s.count(b'[7,9]') == 6, 'parent fluid selector count drift')
    s = s.replace(b'[7,9]', b'[7,9,11]')
    fx = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.37-stackfxq'
    inv=gate.kernel(fx,gate.Inputs(ROOT))[2]
    old.need(gate.sha(gate.encoded(inv['kernel_files']))=='c61907d2e98ec00038ff5fba4bfe1eb21e41d292ee5ebfa6884e7b9b5349d4d6','H-B-187 reference drift')
    fxfiles=old.unpack(fast.restored(old.read(fx/'kernel/release.ipynb')))[0]
    lines=lambda raw:[line for line in raw.splitlines(keepends=True) if b'[7,9,11]' in line]
    old.need(len(lines(s))==3 and lines(s)==lines(fxfiles[HELPER]),'H-B-187 fracture hunk mismatch')
    files[HELPER] = s
    s = old.replace(files[WRAPPER], old.digest(original[HELPER]), old.digest(s))
    for before, after in [
        ('VERSION = "1.1.35-stackoaq"', 'VERSION = "'+VERSION+'"'),
        ('RELEASE = "C85R7+codecs R224 FLIP_ALL fluidR hard OA"', 'RELEASE = "C85R7+codecs R224 FLIP_ALL fluidR hard OA FractureR"'),
        ('KERNEL_ID = "taeyangg4/knee-lb-stackoaq-v1"', 'KERNEL_ID = "'+SLUG+'"'),
        ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.35-stackoaq"',
         'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v'+VERSION+'"'),
        ('formula="public OA=rank(S_R7flip) with combined-R fallback; fluid=combined-R with R-only eligibility; other public=.5*(.65*r(R384)+.35*r(R224))+.5*r(S_R7); own-success ranks; float32",',
         'formula="public OA=rank(S_R7flip) with combined-R fallback; fluid/Fracture=combined-R with R-only eligibility; other public=.5*(.65*r(R384)+.35*r(R224))+.5*r(S_R7); own-success ranks; float32",')]:
        s = old.replace(s, before, after)
    files[WRAPPER] = s
    return files


def payload():
    nb, original, old_blob, assignments = parent()
    files = patched(original)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source, zipfile.ZipFile(buf,'w') as target:
        for info in source.infolist(): target.writestr(info, files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before, after in [
        ('BLOB = '+repr(assignments['BLOB']), 'BLOB = '+repr(base64.b64encode(blob).decode())),
        ('PINS = '+repr(assignments['PINS']), 'PINS = '+repr({k:old.digest(v) for k,v in files.items()})),
        (repr(old.digest(old_blob)), repr(old.digest(blob)))]:
        code = old.replace(code,before,after)
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    meta = old.read(BASE/'kernel/kernel-metadata.json')
    meta.update(id=SLUG,title=SLUG.split('/')[1])
    return nb,meta,files,blob


def scope(nb,meta):
    expected,em,files,blob = payload()
    old.need(nb==expected and meta==em, 'forbidden notebook/metadata diff')
    actual,ab,_ = old.unpack(fast.restored(nb))
    old.need(actual==files and ab==blob, 'forbidden embedded diff')
    original=parent()[1]
    old.need(set(files)==set(original) and {k for k in files if files[k]!=original[k]}=={HELPER,WRAPPER}, 'member scope')
    for name,data in files.items():
        if name.endswith('.py'): compile(data,name,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    return files,blob


def artifacts():
    nb,meta,files,blob=payload()
    pnb,original,_,_=parent()
    manifest=dict(version=VERSION,added=[],removed=[],changed=[],unchanged=[])
    full='ALL 31 EMBEDDED FILES: 29 IDENTICAL, 2 REGISTERED CHANGES; NO ADDED/REMOVED MEMBERS\n'
    for k in sorted(files):
        a,b=old.digest(original[k]),old.digest(files[k])
        manifest['changed' if a!=b else 'unchanged'].append(dict(name=k,before_sha256=a,after_sha256=b))
        full+=f'{k}: {a} -> {b} '+('REGISTERED_CHANGE' if a!=b else 'IDENTICAL')+'\n'
        full+=rbuild.diff(original[k],files[k],'v1.1.35-stackoaq/'+k,VERSION+'/'+k)
    for name,before,after in [('notebook',pnb,nb),('kernel-metadata.json',old.read(BASE/'kernel/kernel-metadata.json'),meta)]:
        full+=rbuild.diff(old.encoded(before),old.encoded(after),'parent/'+name,'candidate/'+name)
    return nb,meta,files,blob,manifest,full.encode()


def register():
    value=registration()
    old.write(OUT/'registration.json',old.encoded(value))
    return dict(status='REGISTERED_BEFORE_EVALUATION',version=VERSION,exposure='EXPOSED',rule=value['public_reading']['rule'])


def prepare():
    old.need((OUT/'registration.json').read_bytes()==old.encoded(registration()),'register first')
    nb,meta,files,blob,manifest,full=artifacts()
    scope(nb,meta)
    for name in ('kernel/release.ipynb','kernel/kernel-metadata.json'):
        old.write(BACKUP/name,(BASE/name).read_bytes())
    old.write(BACKUP/'parent-admission-receipt.json',RECEIPT.read_bytes())
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
        ('new-members.json',old.encoded(manifest)),(DIFF,full),('SPEC.md',SPEC.read_bytes())]:
        old.write(OUT/name,data)
    for label,contents in [('candidate',files),('parent',parent()[1])]:
        for name,data in contents.items(): old.write(SCRATCH/(label+'-runtime')/name,data)
    rec=dict(status='PREPARED_CPU_ONLY',version=VERSION,builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],archive_sha256=old.digest(blob),
        embedded={k:old.digest(v) for k,v in files.items()},unchanged=29,changed=[HELPER,WRAPPER],added=[],removed=[],
        parent_kernel_sha256=PARENT_KERNEL,parent_receipt_sha256=PARENT_RECEIPT,
        qualified_for_submission=False,exposure='EXPOSED',clean_oof=False)
    old.write(OUT/'build-receipt.json',old.encoded(rec))
    return validate()


def validate():
    nb,meta,files,blob,manifest,full=artifacts()
    for name,data in [('registration.json',old.encoded(registration())),('kernel/release.ipynb',old.encoded(nb)),
        ('kernel/kernel-metadata.json',old.encoded(meta)),('new-members.json',old.encoded(manifest)),
        (DIFF,full),('SPEC.md',SPEC.read_bytes())]:
        old.need((OUT/name).read_bytes()==data,'stage drift: '+name)
    scope(old.read(OUT/'kernel/release.ipynb'),old.read(OUT/'kernel/kernel-metadata.json'))
    rec=old.read(OUT/'build-receipt.json')
    old.need(rec['builder_sha256']==old.digest(Path(__file__).read_bytes()),'builder drift')
    old.need(rec['embedded']=={k:old.digest(v) for k,v in files.items()} and rec['archive_sha256']==old.digest(blob),'archive drift')
    old.need(rec['kernel_files']==gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],'kernel drift')
    old.need(rec['qualified_for_submission'] is False and rec['unchanged']==29,'receipt scope')
    from tools import strict_variant_gate_r224_v145 as generic
    pins=gate.Inputs(ROOT)
    generic.metadata_delta(gate.kernel(BASE,pins),gate.kernel(OUT,pins),SLUG,registration())
    return dict(status='PASS_CPU_VALIDATE',kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),
                identical=29,changed=2,metadata_raw_values_identical_except_id_title=True)


def math_functions(files):
    """Test-only extraction; the real replay imports both actual embedded archives."""
    return infrastructure.math_functions(files)


def oracle(arrays,masks):
    """Independent column-wise survivor oracle, using NaN rather than zero ranks."""
    import numpy as np
    import pandas as pd
    def rank(v,failed):
        out=np.full(len(v),np.nan)
        out[~failed]=pd.Series(v[~failed]).rank(method='average',pct=True).to_numpy()
        return out
    result=np.empty_like(arrays[0])
    rf,f224,sf,*ofs=masks
    for col in range(12):
        a,b,s,*owns=[rank(v[:,col],f) for v,f in zip(arrays,masks)]
        rr=np.where(rf,b,np.where(f224,a,.65*a+.35*b))
        rr[rf & f224]=.5
        if col in (7,9,11):
            public=rr.copy()
        elif col in (5,6):
            public=s.copy()
            public[sf]=rr[sf]
        else:
            public=.5*rr+.5*s
            public[rf & f224]=s[rf & f224]
            public[sf]=rr[sf]
        pf=(rf & f224) if col in (7,9,11) else (rf & f224 & sf)
        public[pf]=.5
        den=sum(w*(~f).astype(int) for w,f in zip((17,17,12),ofs))
        own=sum(w*np.nan_to_num(v) for w,v in zip((17,17,12),owns))/np.maximum(den,1)
        pr,oo=rank(public,pf),rank(own,den==0)
        out=.85*pr+.15*oo
        out[den==0]=pr[den==0]
        out[pf]=oo[pf]
        out[pf & (den==0)]=.5
        result[:,col]=out
    return result


REPLAY_SCRIPT = r'''
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
runtime,source,dest=map(Path,sys.argv[1:4])
sys.path.insert(0,str(runtime))
import exp169_release_c85r7codec as c
import exp169_r224 as h
assert c.r224 is h and Path(h.__file__).resolve().parent==runtime.resolve()
assert Path(c.__file__).resolve().parent==runtime.resolve()
assert all(not name.startswith('tools.') for name in sys.modules)
data=np.load(source,allow_pickle=False)
ids=list(data['ids']); arrays=[data['a'+str(i)] for i in range(6)]
masks=[data['m'+str(i)] for i in range(6)]
names=('R','R224','S_R7','EXP143v2','EXP155','Q_A5q')
failed={name:dict.fromkeys([u for u,f in zip(ids,m) if f],'authenticated cached failure') for name,m in zip(names,masks)}
result=h.blend(ids,*arrays,failed)
assert np.isfinite(result).all() and np.all((result>=0)&(result<=1))
if not any(m.any() for m in masks):
    assert np.array_equal(result,h.reference(*arrays))
np.save(dest.with_suffix('.npy'),result,allow_pickle=False)
dest.with_suffix('.csv').write_bytes(c.p.csv_bytes(ids,result))
import torch
assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
print(json.dumps({'status':'PASS','version':c.VERSION,'studies':len(ids),'targets':12,'embedded_imports':True,'cuda_initialized':False}))
'''


def replay():
    import numpy as np
    import torch
    from tools import c85r7_gold58_candidates as cache
    old.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only replay required')
    valid=validate()
    # Authentication reads the historical label table for roster IDs only;
    # label values never enter the replay or output.
    reg,common,failed,arms=cache.context()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    def protected_csv(path):
        rows=path.read_bytes().splitlines()
        return b'\n'.join(b','.join(row.split(b',')[j] for j in (0,*[i+1 for i in PROTECTED])) for row in rows)
    cases=[]
    for n in (2,58):
        ids=reg['binding']['ids'][:n]
        old.need(len(ids)==n,'real cohort size')
        arrays=[np.asarray(common[k],float)[:n] for k in ('R','R224')]
        arrays += [np.asarray(arms['flip-1']['predictions'],float)[:n]]
        arrays += [np.asarray(common[k],float)[:n] for k in MEMBERS[3:]]
        masks=[np.asarray(failed[k],bool)[:n].copy() if k in failed else np.zeros(n,bool) for k in MEMBERS]
        old.need(not any(m.any() for m in masks),'real smoke requires healthy recorded rows')
        old.need(not any(np.all(a==.5,axis=1).any() for a in arrays[:2]),'unexpected real R sentinel')
        scenarios={'observed_healthy':(), 'injected_compact_failure':(2,),
            'injected_own_failure':(3,4,5), 'injected_partial_own_failure':(3,),
            'injected_r384_failure':(0,), 'injected_compact_and_own_failure':(2,3,4,5)}
        if n==58:
            scenarios.update(injected_r224_failure=(1,),injected_both_r_failure=(0,1),
                injected_both_r_compact_failure=(0,1,2),injected_both_r_own_failure=(0,1,3,4,5),
                injected_all_failure=(0,1,2,3,4,5))
        for case,indices in scenarios.items():
            current=[m.copy() for m in masks]
            for i in indices: current[i][0]=True
            stem=f'real{n}-'+case
            source=SCRATCH/(stem+'-input.npz')
            np.savez(source,ids=np.asarray(ids),**{'a'+str(i):a for i,a in enumerate(arrays)},
                     **{'m'+str(i):m for i,m in enumerate(current)})
            imports=[]
            for label in ('parent','candidate'):
                output=SCRATCH/(stem+'-'+label)
                run=subprocess.run([sys.executable,'-B','-I','-c',REPLAY_SCRIPT,
                    str(SCRATCH/(label+'-runtime')),str(source),str(output)],cwd=SCRATCH,env=env,capture_output=True,text=True)
                old.need(run.returncode==0,'isolated real replay failed: '+run.stderr[-6000:])
                imports.append(json.loads(run.stdout.strip().splitlines()[-1]))
            prior,result=[np.load(SCRATCH/(stem+'-'+label+'.npy'),allow_pickle=False) for label in ('parent','candidate')]
            expected=oracle(arrays,current)
            old.need(result.tobytes()==expected.tobytes(),'independent pandas byte parity')
            a,b=[np.ascontiguousarray(v[:,PROTECTED]).tobytes() for v in (prior,result)]
            old.need(a==b,'protected-eleven float64 bytes changed')
            aa,bb=[protected_csv(SCRATCH/(stem+'-'+label+'.csv')) for label in ('parent','candidate')]
            old.need(aa==bb,'protected-eleven serialized CSV fields changed')
            cases.append(dict(case=case,studies=n,first_study_injected_failed_members=[MEMBERS[i] for i in indices],
                observed_failures=False,embedded_runs=imports,independent_pandas_bytes_exact=True,
                protected_eleven_float64_bytes_exact=True,protected_eleven_csv_fields_exact=True,
                protected_float64_sha256=old.digest(a),protected_csv_sha256=old.digest(aa),
                candidate_output_sha256=old.digest(result.tobytes()),
                candidate_csv_sha256=old.digest((SCRATCH/(stem+'-candidate.csv')).read_bytes()),
                cohort_sha256=old.digest(old.encoded(ids)),inputs_sha256=[old.digest(a.tobytes()) for a in arrays]))
            print(json.dumps(dict(progress=stem,status='PASS')),flush=True)
    report=dict(status='PASS',kernel_sha256=valid['kernel_sha256'],studies=[2,58],targets=12,
        source='authenticated preserved Gold58 raw R224/FLIP_ALL and unchanged own member predictions',
        historical_binding_sha256=reg['binding_sha256'],preserved_pins_sha256=cache.PINS_SHA,
        independent_pandas_bytes_exact=True,protected_eleven_float64_bytes_exact=True,
        protected_eleven_csv_fields_exact=True,cases=cases,exposure='EXPOSED',clean_oof=False,
        limitation='Real cached-serving arithmetic replay of actual embedded imports; healthy and explicitly injected failures. Pinned roster authentication reads label table internally for IDs only; no label evaluation/selection/printing. No AUC, model forward, CUDA/Linux qualification or new runtime measurement.')
    old.write(SCRATCH/'replay.json',old.encoded(report))
    return dict(status=report['status'],studies=report['studies'],cases=len(cases),kernel_sha256=valid['kernel_sha256'])


def admit():
    valid=validate(); smoke=old.read(SCRATCH/'replay.json')
    old.need(smoke['status']=='PASS' and smoke['kernel_sha256']==valid['kernel_sha256'],'CPU replay missing/stale')
    old.need(smoke['protected_eleven_float64_bytes_exact'] and smoke['protected_eleven_csv_fields_exact'],'protected-eleven parity missing')
    return dict(status='HOLD_B_SLUG_GENERIC_GATE_AND_INDEPENDENT_REVIEW',qualified_for_submission=False,
                kernel_sha256=valid['kernel_sha256'],gate='tools/strict_variant_gate_r224_v145.py')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','prepare','validate','cpu-replay','admit'])
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.command in ('prepare','admit'): old.need(args.dry_run,'CPU dry-run required')
    result={'register':register,'prepare':prepare,'validate':validate,'cpu-replay':replay,'admit':admit}[args.command]()
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
