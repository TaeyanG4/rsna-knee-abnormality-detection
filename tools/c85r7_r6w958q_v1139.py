"""H-B-189 builder 1.0.0: source-fixed per-target P weights on admitted H174.

Changelog 1.0.0: one weights-only R6 candidate; authentic parent and cached
serving replay reuse H181 infrastructure. No frozen/scored project edits.
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

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import c85r7_r224r6oaq_v1134 as reuse
old,fast,rbuild,gate=reuse.old,reuse.fast,reuse.rbuild,reuse.gate
BASE,RECEIPT=reuse.BASE,reuse.RECEIPT
PARENT_KERNEL,PARENT_RECEIPT=reuse.PARENT_KERNEL,reuse.PARENT_RECEIPT
HELPER,WRAPPER=reuse.HELPER,reuse.WRAPPER
MEMBERS,PRIVATE_DATASETS=reuse.MEMBERS,reuse.PRIVATE_DATASETS
parent,math_functions,authenticated_r6=reuse.parent,reuse.math_functions,reuse.authenticated_r6
VERSION='1.1.39-r6w958q'
SLUG='taeyangg4/knee-lb-r6w958q-v1'
OUT=old.BASE.parent/('v'+VERSION)
SCRATCH=ROOT/'.local/scratch/hb189-build/hb189'
BACKUP=ROOT/'.local/backups/hb189-build/hb189-parent'
SPEC=SCRATCH/'SPEC.md'
PRE_REVIEW=SCRATCH/'PRE-REVIEW.md'
SOURCE=ROOT/'.local/idea-scout/20261016-2319/notebooks/goodpjw2008__rsna-knee-coatnet-r6-2-5d-mil-lb-0-958/SOURCE.txt'
FABLE=ROOT/'.local/scratch/slot-b/stack-audit-1016/fable/stack_gold58.py'
DIFF='full-diff-vs-v1.1.30-r224r6fq.txt'
WEIGHTS=(.5,.5,.5,.5,.5,.2,.6,.5,.8,.35,.5,.5)
CHANGED=(5,6,8,9)
PROTECTED=tuple(i for i in range(12) if i not in CHANGED)
FORMULA=('0.85*r(P)+0.15*r(O); rr=0.65*r(R384)+0.35*r(R224); '
    'P_t=w_t*rr_t+(1-w_t)*r(R6flip)_t; w=.5 except Lateral OA=.2, PF OA=.6, Synovitis=.8, Baker\'s=.35; '
    'O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46')


def pre_review():
    import re
    raw=PRE_REVIEW.read_text(encoding='utf-8')
    sha=old.digest(SPEC.read_bytes())
    old.need('Verdict: PASS for CPU-only build/tests/replay of SPEC.md v1.0.0' in raw
             and re.search(r'\b'+sha+r'\b',raw),'independent pre-execution PASS bound to spec required')
    return dict(spec_sha256=sha,review_sha256=old.digest(PRE_REVIEW.read_bytes()),
                review_path=PRE_REVIEW.relative_to(ROOT).as_posix())


def registration():
    parent()
    return dict(item='H-B-189',version=VERSION,kernel_id=SLUG,
        purpose='Source-fixed per-target Raptor/compact weights transfer into C85 on admitted H174 R6',
        parent=BASE.relative_to(ROOT).as_posix(),parent_kernel_sha256=PARENT_KERNEL,
        parent_receipt_sha256=PARENT_RECEIPT,parent_submission=57251520,parent_displayed_public=.957,
        formula=FORMULA,weights=list(WEIGHTS),changed_columns=list(CHANGED),protected_columns=list(PROTECTED),
        pre_execution=pre_review(),source_sha256=old.digest(SOURCE.read_bytes()),
        source_path=SOURCE.relative_to(ROOT).as_posix(),source_lines=['1772-1852','2307-2415'],
        source_differences='Same rr rank mixture without reranking rr. Retain parent independent R224 failures and unmirrored R224; public uses R384 failure mask and R224 mirror. Retain C85 O; do not copy reader overlay, float32 base reload or final rerank.',
        failure_rule='Unchanged parent survivor overrides and per-column ranks; no new missing-member semantics; R224 >3% abort retained.',
        public_reading=dict(rule='RETAIN iff displayed >=0.957',expected_public=[.957,.958],
            direction='.958 would reproduce advertised gain direction, not isolate its cause',source='user supplied; no polling'),
        timing=dict(source='User confirms H174 hidden completion, submitted 57251520; weights-only inheritance',
            inference_delta='none',child_measured_hours=None,hidden_hours=None),
        gold58=dict(exposure='EXPOSED',use='descriptive only, not a gate',weak_four=[3,5,6,8],
            replay=FABLE.relative_to(ROOT).as_posix(),replay_sha256=old.digest(FABLE.read_bytes()),
            ranks='full 58 studies, average ties, success-only ranks'),
        exposure='EXPOSED',clean_oof=False,qualified_for_submission=False,
        runtime_qualification='CPU cached-serving arithmetic only, no CUDA/Linux/model-forward qualification',
        gate_binding='B must pin slug-generic strict_variant_gate_r224_v145 with authentic dispatcher builder/reviewer receipts')


def decision(displayed):
    old.need(type(displayed) in (int,float) and math.isfinite(displayed) and 0<=displayed<=1,'finite displayed score required')
    return 'RETAIN' if displayed>=.957 else 'DROP'


def patched(original):
    files=dict(original)
    s=files[HELPER]
    line=next(x for x in s.decode().splitlines() if x.startswith('FORMULA = '))
    s=old.replace(s,line,'FORMULA = '+repr(FORMULA))
    # Broadcast vector contains exact .5 defaults, keeping eight columns byte-identical.
    s=old.replace(s,'    public = .5 * rr + .5 * ss\n',
        '    w = np.asarray('+repr(WEIGHTS)+')\n    public = w * rr + (1.0 - w) * ss\n')
    s=old.replace(s,
        '    return .85 * rank(.5 * (.65 * rank(r) + .35 * rank(r224)) + .5 * rank(ours)) + .15 * rank(',
        '    w = np.asarray('+repr(WEIGHTS)+')\n'
        '    return .85 * rank(w * (.65 * rank(r) + .35 * rank(r224)) + (1.0 - w) * rank(ours)) + .15 * rank(')
    files[HELPER]=s
    s=old.replace(files[WRAPPER],old.digest(original[HELPER]),old.digest(s))
    for before,after in [
        ('VERSION = "1.1.30-r224r6fq"','VERSION = "'+VERSION+'"'),
        ('RELEASE = "C85R7+codecs R224 R6 FLIP_ALL"','RELEASE = "C85R7+codecs R224 R6 FLIP_ALL per-target W958"'),
        ('KERNEL_ID = "taeyangg4/knee-lb-r224r6fq-v1"','KERNEL_ID = "'+SLUG+'"'),
        ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.30-r224r6fq"',
         'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v'+VERSION+'"'),
        ('formula="public cell 8: .5*(.65*r(R384)+.35*r(R224))+.5*r(S_R7); own-success ranks; float32",',
         'formula="public w*(.65*r(R384)+.35*r(R224))+(1-w)*r(S_R6flip); w=.5 except LatOA=.2/PFOA=.6/Syn=.8/Baker=.35; own-success ranks; float32",')]:
        s=old.replace(s,before,after)
    files[WRAPPER]=s
    return files


def payload():
    nb,original,old_blob,assignments=parent()
    files=patched(original)
    buf=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source,zipfile.ZipFile(buf,'w') as target:
        for info in source.infolist(): target.writestr(info,files[info.filename])
    blob=buf.getvalue()
    code=''.join(nb['cells'][-1]['source']).encode()
    for before,after in [
        ('BLOB = '+repr(assignments['BLOB']),'BLOB = '+repr(base64.b64encode(blob).decode())),
        ('PINS = '+repr(assignments['PINS']),'PINS = '+repr({k:old.digest(v) for k,v in files.items()})),
        (repr(old.digest(old_blob)),repr(old.digest(blob)))]: code=old.replace(code,before,after)
    nb['cells'][-1]['source']=code.decode().splitlines(True)
    meta=old.read(BASE/'kernel/kernel-metadata.json')
    meta.update(id=SLUG,title=SLUG.split('/')[1])
    return nb,meta,files,blob


def scope(nb,meta):
    expected,em,files,blob=payload()
    old.need(nb==expected and meta==em,'forbidden notebook/metadata diff')
    actual,ab,_=old.unpack(fast.restored(nb))
    old.need(actual==files and ab==blob,'forbidden embedded diff')
    original=parent()[1]
    old.need(set(files)==set(original) and {k for k in files if files[k]!=original[k]}=={HELPER,WRAPPER},'member scope')
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
        full+=rbuild.diff(original[k],files[k],'H174/'+k,'H189/'+k)
    for name,before,after in [('notebook',pnb,nb),('kernel-metadata.json',old.read(BASE/'kernel/kernel-metadata.json'),meta)]:
        full+=rbuild.diff(old.encoded(before),old.encoded(after),'parent/'+name,'candidate/'+name)
    return nb,meta,files,blob,manifest,full.encode()


CHANGELOG=b'''# v1.1.39-r6w958q
H-B-189: four source-fixed P weights on admitted H174; unchanged eight targets,
outer own .15, R224, R6 FULL flip, FAST, serving, all failure overrides/datasets.
Only helper arithmetic/reference/formula and wrapper version/id/formula/pin.
29/31 embedded files identical. No original/frozen/scored files edited.
Parent backup: .local/backups/hb189-build/hb189-parent. EXPOSED; no new inference.
Keep displayed >=.957, expected .957-.958; Gold58 descriptive, not a gate.
'''


def register():
    data=old.encoded(registration())
    path=OUT/'registration.json'
    if path.exists(): old.need(path.read_bytes()==data,'immutable registration drift')
    else:
        old.need(not (OUT/'kernel').exists(),'registration must precede kernel')
        old.write(path,data)
    old.write(OUT/'registration.sha256',(old.digest(data)+'\n').encode())
    return dict(status='REGISTERED_BEFORE_EVALUATION',sha256=old.digest(data),exposure='EXPOSED')


def prepare():
    old.need((OUT/'registration.json').read_bytes()==old.encoded(registration()),'register first')
    nb,meta,files,blob,manifest,full=artifacts()
    scope(nb,meta)
    for name in ('kernel/release.ipynb','kernel/kernel-metadata.json'):
        old.write(BACKUP/name,(BASE/name).read_bytes())
    old.write(BACKUP/'parent-admission-receipt.json',RECEIPT.read_bytes())
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
        ('CHANGELOG.md',CHANGELOG),('new-members.json',old.encoded(manifest)),(DIFF,full),('SPEC.md',SPEC.read_bytes())]:
        old.write(OUT/name,data)
    for label,contents in [('candidate',files),('parent',parent()[1])]:
        for name,data in contents.items(): old.write(SCRATCH/(label+'-runtime')/name,data)
    rec=dict(status='PREPARED_CPU_ONLY',version=VERSION,builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],archive_sha256=old.digest(blob),
        embedded={k:old.digest(v) for k,v in files.items()},unchanged=29,changed=[HELPER,WRAPPER],added=[],removed=[],
        parent_kernel_sha256=PARENT_KERNEL,parent_receipt_sha256=PARENT_RECEIPT,qualified_for_submission=False)
    old.write(OUT/'build-receipt.json',old.encoded(rec))
    return validate()


def validate():
    nb,meta,files,blob,manifest,full=artifacts()
    for name,data in [('registration.json',old.encoded(registration())),('kernel/release.ipynb',old.encoded(nb)),
        ('kernel/kernel-metadata.json',old.encoded(meta)),('CHANGELOG.md',CHANGELOG),
        ('new-members.json',old.encoded(manifest)),(DIFF,full),('SPEC.md',SPEC.read_bytes())]:
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
    return dict(status='PASS_CPU_VALIDATE',kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),identical=29,changed=2,
                metadata_raw_values_identical_except_id_title=True)


def oracle(arrays,masks):
    """Independent column-wise pandas oracle, with NaN failure representation."""
    import numpy as np
    import pandas as pd
    def rank(v,failed):
        out=np.full(len(v),np.nan)
        out[~failed]=pd.Series(v[~failed]).rank(method='average',pct=True).to_numpy()
        return out
    result=np.empty_like(arrays[0])
    rf,f224,sf,*ofs=masks
    for col,w in enumerate(WEIGHTS):
        a,b,s,*owns=[rank(v[:,col],f) for v,f in zip(arrays,masks)]
        rr=np.where(rf,b,np.where(f224,a,.65*a+.35*b)); rr[rf & f224]=.5
        public=w*rr+(1-w)*s
        public[rf & f224]=s[rf & f224]; public[sf]=rr[sf]
        pf=rf & f224 & sf; public[pf]=.5
        den=sum(weight*(~f).astype(int) for weight,f in zip((17,17,12),ofs))
        own=sum(weight*np.nan_to_num(v) for weight,v in zip((17,17,12),owns))/np.maximum(den,1)
        pr,oo=rank(public,pf),rank(own,den==0)
        out=.85*pr+.15*oo
        out[den==0]=pr[den==0]; out[pf]=oo[pf]; out[pf & (den==0)]=.5
        result[:,col]=out
    return result


def replay():
    import importlib.util
    import numpy as np
    import torch
    from tools import c85r7_gold58_candidates as cache
    old.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only required')
    valid=validate()
    reg,common,failed,arms=cache.context()
    r6_reg,compact,r6_auth=authenticated_r6()
    ids=reg['binding']['ids']
    old.need(r6_reg['binding']['ids']==ids and len(ids)==58,'R6/H174 Gold58 ordered cohort drift')
    arrays=[np.asarray(common[k],float) for k in ('R','R224')]+[compact]+[np.asarray(common[k],float) for k in MEMBERS[3:]]
    masks=[np.asarray(failed[k],bool).copy() if k in failed else np.zeros(58,bool) for k in MEMBERS]
    old.need(not any(m.any() for m in masks),'Gold58 observed failures require review')
    old.need(not any(np.all(a==.5,axis=1).any() for a in arrays[:2]),'unexpected real R sentinel')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONPATH',None)
    cases=[]; results={}
    def protected_csv(path):
        return b'\n'.join(b','.join(row.split(b',')[j] for j in (0,*[i+1 for i in PROTECTED])) for row in path.read_bytes().splitlines())
    scenarios={'healthy':(),'compact_failed':(2,),'own_failed':(3,4,5),'partial_own_failed':(3,),
        'r384_failed':(0,),'compact_own_failed':(2,3,4,5)}
    for n in (2,58):
        scenarios_n=dict(scenarios)
        if n==58: scenarios_n.update(r224_failed=(1,),both_raptor_failed=(0,1),all_failed=(0,1,2,3,4,5))
        for case,indices in scenarios_n.items():
            current=[m[:n].copy() for m in masks]
            for i in indices: current[i][0]=True
            aa=[a[:n] for a in arrays]
            stem=f'real{n}-{case}'
            source=SCRATCH/(stem+'-input.npz')
            np.savez(source,ids=np.asarray(ids[:n]),**{'a'+str(i):a for i,a in enumerate(aa)},**{'m'+str(i):m for i,m in enumerate(current)})
            imports=[]
            for label in ('parent','candidate'):
                output=SCRATCH/(stem+'-'+label)
                run=subprocess.run([sys.executable,'-B','-I','-c',reuse.REPLAY_SCRIPT,
                    str(SCRATCH/(label+'-runtime')),str(source),str(output)],cwd=SCRATCH,env=env,capture_output=True,text=True)
                old.need(run.returncode==0,'isolated replay failed: '+run.stderr[-6000:])
                imports.append(json.loads(run.stdout.strip().splitlines()[-1]))
            prior,result=[np.load(SCRATCH/(stem+'-'+label+'.npy'),allow_pickle=False) for label in ('parent','candidate')]
            old.need(result.tobytes()==oracle(aa,current).tobytes(),'independent pandas byte parity')
            old.need(np.ascontiguousarray(prior[:,PROTECTED]).tobytes()==np.ascontiguousarray(result[:,PROTECTED]).tobytes(),'protected-eight float64 bytes')
            old.need(protected_csv(SCRATCH/(stem+'-parent.csv'))==protected_csv(SCRATCH/(stem+'-candidate.csv')),'protected-eight CSV fields')
            # Direct P-branch truth table on real arrays, including pre-outer-rank weights.
            public=math_functions(payload()[2])['p_branch'](*aa[:3],*current[:3])
            import pandas as pd
            def rank(v,m):
                z=np.zeros_like(v); z[~m]=pd.DataFrame(v[~m]).rank(method='average',pct=True).to_numpy(); return z
            ar,br,sr=[rank(a,m) for a,m in zip(aa,current)] [0:3]
            rf,f224,sf=current[:3]; rr=.65*ar+.35*br
            rr[rf]=br[rf]; rr[f224]=ar[f224]; rr[rf & f224]=.5
            for j,w in enumerate(WEIGHTS):
                expected=w*rr[:,j]+(1-w)*sr[:,j]
                expected[rf & f224]=sr[rf & f224,j]; expected[sf]=rr[sf,j]; expected[rf & f224 & sf]=.5
                old.need(public[:,j].tobytes()==expected.tobytes(),'real column weight truth table')
            cases.append(dict(studies=n,case=case,embedded_runs=imports,independent_pandas_bytes_exact=True,
                protected_eight_float64_and_csv_bytes_exact=True,all12_p_branch_weight_truth_table=True,
                candidate_output_sha256=old.digest(result.tobytes())))
            if n==58 and case=='healthy': results=dict(parent=prior,child=result)
    # Existing exact-blend analysis helper is trusted project code, not public SOURCE.
    spec=importlib.util.spec_from_file_location('hb189_fable',FABLE)
    fable=importlib.util.module_from_spec(spec); spec.loader.exec_module(fable)
    label_ids,labels=fable.d.gold.gold(); labels=np.asarray(labels,float)
    old.need(list(label_ids)==ids and labels.shape==(58,12),'Gold58 labels/roster alignment')
    M={k:a for k,a in zip(MEMBERS,arrays)}
    old.need(np.array_equal(results['parent'],fable.final(M,compact)),'existing exact-blend parent mismatch')
    metrics={}
    for key,value in results.items():
        aucs=np.asarray([fable.auc(labels[:,j],value[:,j]) for j in range(12)])
        metrics[key]=dict(macro=float(aucs.mean()),weak_four=float(aucs[fable.WEAK].mean()))
    metrics['delta']={key:metrics['child'][key]-metrics['parent'][key] for key in ('macro','weak_four')}
    report=dict(status='PASS',kernel_sha256=valid['kernel_sha256'],studies=[2,58],targets=12,
        cases=cases,protected_eight_float64_and_csv_bytes_exact=True,all12_p_branch_weight_truth_table=True,
        source='Authenticated R6 FULL-flip compact and five pinned common member cached predictions',
        cohort_sha256=old.digest(old.encoded(ids)),historical_binding_sha256=reg['binding_sha256'],
        r6_authentication=r6_auth,preserved_pins_sha256=cache.PINS_SHA,inputs_sha256=[old.digest(a.tobytes()) for a in arrays],
        gold58=metrics,exposure='EXPOSED',clean_oof=False,gold58_gate=False,
        limitation='Real cached-serving arithmetic and actual embedded imports; no model-forward/CUDA/Linux qualification. Labels read internally for descriptive AUC, never printed.')
    old.need(not torch.cuda.is_initialized(),'CUDA unexpectedly initialized')
    old.write(SCRATCH/'replay.json',old.encoded(report))
    return report


def admit():
    valid=validate(); smoke=old.read(SCRATCH/'replay.json')
    old.need(smoke['status']=='PASS' and smoke['kernel_sha256']==valid['kernel_sha256'],'CPU replay missing/stale')
    old.need(smoke['protected_eight_float64_and_csv_bytes_exact'] and smoke['all12_p_branch_weight_truth_table'] and smoke['studies']==[2,58],'CPU parity missing')
    return dict(status='HOLD_B_SLUG_GENERIC_GATE_AND_INDEPENDENT_REVIEW',qualified_for_submission=False,
        kernel_sha256=valid['kernel_sha256'],gate='tools/strict_variant_gate_r224_v145.py')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','prepare','validate','cpu-replay','admit'])
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.command in ('prepare','admit'): old.need(args.dry_run,'CPU dry-run required')
    result={'register':register,'prepare':prepare,'validate':validate,'cpu-replay':replay,'admit':admit}[args.command]()
    old.write(SCRATCH/(args.command+'-result.json'),old.encoded(result))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
