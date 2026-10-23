"""H-B-185 builder 1.0.0: admitted fluid route on fixed H181 R6 hard-OA parent.

Changelog 1.0.0: byte-identical H168 fluid hunks, existing H181 R6/OA retained;
dependent pins/version/id only. No frozen edits or added model inference.
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
VERSION = '1.1.36-r6stackoaq'
SLUG = 'taeyangg4/knee-lb-r6stackoaq-v1'
BASE = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.34-r224r6oaq'
PARENT_GATE = ROOT / '.local/scratch/astra-b5/hb181-gate/gate-copy-v145'
RECEIPT = ROOT / '.local/state/gates/knee-lb-r224r6oaq-v1.json'
PARENT_KERNEL = 'b24cfd0205bcadd365fdab92cb775676bc2763969ca0cd4eee610e0f932d1517'
PARENT_RECEIPT = 'a732818c76b9ee2bec998de2d088cae7e38f5fe676f508a036e6926329cdb77c'
REFERENCE = ROOT / '.local/scratch/astra-b5/hb168-gate/gate-copy-v144'
REFERENCE_SOURCE = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.25-stackq'
REFERENCE_RECEIPT = ROOT / '.local/state/gates/knee-lb-stackq-v1.json'
REFERENCE_RECEIPT_SHA = 'b3a9b5579f9a476e6b9112830850f81d15badea41cda59029e2011b4789ae640'
REFERENCE_KERNEL = '9e184865452a3a4a7a5b637c0be9701fac965f7546b6a0dccb6956bb1dbce89b' 
OUT = old.BASE.parent / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/hb185-build/hb185'
BACKUP = ROOT / '.local/backups/hb185-build/hb185-parent'
SPEC = SCRATCH / 'SPEC.md'
PRE_REVIEW = SCRATCH / 'PRE-REVIEW.md'
HELPER, WRAPPER = infrastructure.HELPER, infrastructure.WRAPPER
DIFF = 'full-diff-vs-v1.1.34-r224r6oaq.txt'
ROUTE = '    public[:, [5,6]] = np.where(sf[:, None], rr[:, [5,6]], ss[:, [5,6]])\n'
FORMULA = ('0.85*r(P)+0.15*r(O); r_R=0.65*r(R384)+0.35*r(R224); '
           'P_t=r(R6flip)_t for Lateral OA and PF OA when compact succeeds, else r_R_t; '
           'P_t=r_R_t for Effusion and Bakers Cyst, with R-only failure eligibility; other P=0.5*r_R+0.5*r(R6flip); '
           'O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46')
PROTECTED = (0,1,2,3,4,5,6,8,10,11)
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


def reference_blocks():
    _,files,_,_=authenticated(REFERENCE,REFERENCE_SOURCE,REFERENCE_RECEIPT,
                              REFERENCE_RECEIPT_SHA,REFERENCE_KERNEL)
    blocks=['    public[failed], public[sf], public[failed & sf] = ss[failed], rr[sf], .5\n'
            '    public[:, [7,9]] = rr[:, [7,9]]\n',
            '    rr, rf = r_half(r, r224, rf, f224)\n'
            '    rr = c85.scored_rank(rr, rf)\n' + old.ROUTE,
            '    rr = .65 * rank(r) + .35 * rank(r224)\n'
            '    public = .5 * rr + .5 * rank(ours)\n'
            '    public[:, [7,9]] = rr[:, [7,9]]\n']
    for block in blocks:
        old.need(block.encode() in files[HELPER], 'fluid reference bytes missing')
    return blocks


def pre_review():
    old.pin(SPEC,'bf41956afb42ba3425b24530a36b247f4a6b7e5d46e49b2bd86b7caf5cbbd858')
    old.need('Verdict: PASS for CPU-only build/tests/replay of SPEC.md v1.0.0' in
             PRE_REVIEW.read_text(encoding='utf-8'), 'independent pre-execution PASS required')
    return dict(spec_sha256=old.digest(SPEC.read_bytes()),
                review_path=PRE_REVIEW.relative_to(ROOT).as_posix(),
                review_sha256=old.digest(PRE_REVIEW.read_bytes()))


def registration():
    parent(); reference_blocks()
    return dict(item='H-B-185',version=VERSION,kernel_id=SLUG,
        purpose='ONE final-pair candidate: admitted H168 fluid route on fixed admitted H181 R6 hard-OA parent',
        parent=BASE.relative_to(ROOT).as_posix(),parent_kernel_sha256=PARENT_KERNEL,
        parent_receipt_sha256=PARENT_RECEIPT,parent_submission=57263371,
        reference_kernel_sha256=REFERENCE_KERNEL,reference_receipt_sha256=REFERENCE_RECEIPT_SHA,
        formula=FORMULA,routed_columns=[7,9],unchanged_oa_columns=[5,6],protected_columns=list(PROTECTED),
        pre_execution=pre_review(),preregistration='register precedes prepare/tests/replay; no label evaluation or printing',
        failure_rule='Fluid uses combined R .65/.35 or sole survivor, R-only failure eligibility; both R failed uses own even if compact succeeds. OA public uses success-only R6 compact ranks, failed compact uses combined R. Other public targets unchanged. Public failed: own only; own failed: public only; both failed: .5. Own survivor 17/17/12 and R224 >3% abort unchanged.',
        parent_clarification='No compo15. Fixed H181 own 17/17/12 and private R6 dataset list preserved. Compact runtime slot retains legacy S_R7 name but serves R6.',
        public_reading=dict(reference_value=.957,
            submit_rule='submit only after user-supplied H181 status COMPLETE and displayed >=0.957; pending HOLD; below threshold DROP',
            rule='retain only if displayed >=0.957 AND auto-selection keeps candidate >=H-B-181; missing evidence HOLD; either failed condition DROP',
            parent_reading=None,parent_status=None,source='user-supplied scores and auto-selection only; no polling'),
        timing=dict(source='user requested H181/H168 hidden completion basis; H181 reading/completion pending at dispatch',
            h168_historical_hidden_hours_upper=7.7,includes_queue=True,h181_measured_hours=None,
            inference_delta='none',child_measured_hours=None,limitation='No measured child runtime bound; arithmetic port only. B verifies H181 completion before submission.'),
        sources=['D-B-207 user-supplied auto-selection H168/H174 >=H167','H184 mirror port; no additive gain claim'],
        exposure='EXPOSED',clean_oof=False,qualified_for_submission=False,
        runtime_qualification='CPU cached arithmetic only; no fresh model forward/CUDA/Linux qualification',
        gate_binding='HOLD: B slug-generic v145 gate with authentic dispatcher builder/reviewer receipts')


def eligibility(parent_status=None,parent_displayed=None):
    if parent_displayed is not None:
        old.need(type(parent_displayed) in (int,float) and math.isfinite(parent_displayed)
                 and 0<=parent_displayed<=1, 'finite parent score required')
        if parent_displayed<.957: return 'DROP_PARENT_BELOW_0957'
    if parent_status!='COMPLETE' or parent_displayed is None: return 'HOLD_PARENT_READING_PENDING'
    return 'ELIGIBLE_FOR_B_REVIEW'


def decision(displayed_score, auto_selection_ge_parent=None):
    if displayed_score is not None:
        old.need(type(displayed_score) in (int,float) and math.isfinite(displayed_score)
                 and 0 <= displayed_score <= 1, 'finite displayed public score required')
    old.need(auto_selection_ge_parent is None or type(auto_selection_ge_parent) is bool,
             'explicit auto-selection comparison required')
    if (displayed_score is not None and displayed_score < .957) or auto_selection_ge_parent is False:
        return 'DROP'
    if displayed_score is None or auto_selection_ge_parent is None:
        return 'HOLD'
    return 'RETAIN'


def patched(original):
    files=dict(original)
    blocks=reference_blocks()
    raw=files[HELPER]
    line=next(x for x in raw.decode().splitlines() if x.startswith('FORMULA = '))
    raw=old.replace(raw,line,'FORMULA = '+repr(FORMULA))
    raw=old.replace(raw,'    public[failed], public[sf], public[failed & sf] = ss[failed], rr[sf], .5\n',blocks[0])
    raw=old.replace(raw,'    return result\n\n\ndef reference(',blocks[1]+'    return result\n\n\ndef reference(')
    raw=old.replace(raw,'    public = .5 * (.65 * rank(r) + .35 * rank(r224)) + .5 * rank(ours)\n',blocks[2])
    files[HELPER]=raw
    raw=old.replace(files[WRAPPER],old.digest(original[HELPER]),old.digest(raw))
    for before,after in [
        ('VERSION = "1.1.34-r224r6oaq"','VERSION = "'+VERSION+'"'),
        ('RELEASE = "C85R7+codecs R224 R6 FLIP_ALL hard OA"','RELEASE = "C85R7+codecs R224 R6 FLIP_ALL hard OA fluidR"'),
        ('KERNEL_ID = "taeyangg4/knee-lb-r224r6oaq-v1"','KERNEL_ID = "'+SLUG+'"'),
        ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.34-r224r6oaq"','OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v'+VERSION+'"'),
        ('formula="public OA=rank(S_R6flip) with combined-R fallback; other public=.5*(.65*r(R384)+.35*r(R224))+.5*r(S_R7); own-success ranks; float32",',
         'formula="public OA=rank(S_R6flip) with combined-R fallback; fluid=combined-R with R-only eligibility; other public=.5*(.65*r(R384)+.35*r(R224))+.5*r(S_R6flip); own-success ranks; float32",')]:
        raw=old.replace(raw,before,after)
    files[WRAPPER]=raw
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
        full+=rbuild.diff(original[k],files[k],'v1.1.34-r224r6oaq/'+k,VERSION+'/'+k)
    for name,before,after in [('notebook',pnb,nb),('kernel-metadata.json',old.read(BASE/'kernel/kernel-metadata.json'),meta)]:
        full+=rbuild.diff(old.encoded(before),old.encoded(after),'parent/'+name,'candidate/'+name)
    return nb,meta,files,blob,manifest,full.encode()


CHANGELOG = b"# v1.1.36-r6stackoaq\nH185: admitted H168 fluid hunks on fixed H181 R6/OA parent; dependent pins/version/id only. All 29 other embedded members and R6 datasets unchanged; no compo15 or new inference. Parent backup .local/backups/hb185-build/hb185-parent; no frozen edits.\n"

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
        ('new-members.json',old.encoded(manifest)),(DIFF,full),('SPEC.md',SPEC.read_bytes()),('CHANGELOG.md',CHANGELOG)]:
        old.write(OUT/name,data)
    for label,contents in [('candidate',files),('parent',parent()[1])]:
        for name,data in contents.items(): old.write(SCRATCH/(label+'-runtime')/name,data)
    rec=dict(status='PREPARED_CPU_ONLY',version=VERSION,builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],archive_sha256=old.digest(blob),
        embedded={k:old.digest(v) for k,v in files.items()},unchanged=29,changed=[HELPER,WRAPPER],added=[],removed=[],
        parent_kernel_sha256=PARENT_KERNEL,parent_receipt_sha256=PARENT_RECEIPT,
        qualified_for_submission=False,exposure='EXPOSED',clean_oof=False)
    old.write(SCRATCH/'fluid-reference-proof.json',old.encoded(dict(blocks=[dict(reference_stage=REFERENCE.relative_to(ROOT).as_posix(),reference_member=HELPER,child_member=HELPER,text=block) for block in reference_blocks()])))
    old.write(OUT/'build-receipt.json',old.encoded(rec))
    return validate()


def validate():
    nb,meta,files,blob,manifest,full=artifacts()
    for name,data in [('registration.json',old.encoded(registration())),('kernel/release.ipynb',old.encoded(nb)),
        ('kernel/kernel-metadata.json',old.encoded(meta)),('new-members.json',old.encoded(manifest)),
        (DIFF,full),('SPEC.md',SPEC.read_bytes()),('CHANGELOG.md',CHANGELOG)]:
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
        if col in (7,9):
            public=rr.copy()
        elif col in (5,6):
            public=s.copy()
            public[sf]=rr[sf]
        else:
            public=.5*rr+.5*s
            public[rf & f224]=s[rf & f224]
            public[sf]=rr[sf]
        pf=(rf & f224) if col in (7,9) else (rf & f224 & sf)
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
import os,sys,json,itertools,hashlib
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
names=('R','R224','S_R7','EXP143v2','EXP155','Q_A5q')
outputs={}; checks=[]
def csv_hash(raw,cols):
    v=b'\n'.join(b','.join(row.split(b',')[j] for j in (0,*[i+1 for i in cols])) for row in raw.splitlines())
    return hashlib.sha256(v).hexdigest()
for n in (2,len(ids)):
    aa=[a[:n] for a in arrays]
    for case,bits in enumerate(itertools.product((False,True),repeat=6)):
        key=f'n{n}-case{case}'
        failed={name:({ids[0]:'injected cached member failure'} if bit else {}) for name,bit in zip(names,bits)}
        try:
            result=h.blend(ids[:n],*aa,failed)
        except ValueError as exc:
            assert n==2 and bits[1] and 'more than 3%' in str(exc),str(exc)
            checks.append(dict(key=key,status='EXPECTED_R224_ABORT'))
            continue
        assert not(n==2 and bits[1]),'R224 threshold bypass'
        assert np.isfinite(result).all() and np.all((result>=0)&(result<=1))
        if not any(bits): assert np.array_equal(result,h.reference(*aa))
        outputs[key]=result
        raw=c.p.csv_bytes(ids[:n],result)
        checks.append(dict(key=key,status='PASS',all_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
            protected_csv_sha256=csv_hash(raw,(0,1,2,3,4,5,6,8,10,11)),
            fluid_csv_sha256=csv_hash(raw,(7,9))))
np.savez(dest.with_suffix('.npz'),**outputs)
import torch
assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
dest.with_suffix('.json').write_text(json.dumps(dict(version=c.VERSION,checks=checks,cuda_initialized=False,embedded_imports=True),indent=2))
print(json.dumps(dict(status='PASS',version=c.VERSION,cases=len(checks),successes=len(outputs),expected_aborts=len(checks)-len(outputs),cuda_initialized=False)))
'''


def cross_reference():
    from tools import c85r7_stackoaq_v1135 as h184
    inv=gate.kernel(h184.OUT,gate.Inputs(ROOT))[2]
    old.need(gate.sha(gate.encoded(inv['kernel_files']))=='8200085ae28dcfcf5d2bc5c8cc95a6941f6d5a1e25d688e739ddae249a5eb506','H184 cross-reference kernel drift')
    old.pin(h184.SCRATCH/'REVIEW.md','8551d35b5508c5337867fd29058de3745f01157d8525686727a7daa6c76c8dfd')
    return old.unpack(fast.restored(old.read(h184.OUT/'kernel/release.ipynb')))[0]


def replay():
    import itertools
    import numpy as np
    import torch
    from tools import c85r7_gold58_candidates as cache
    from tools import c85r7_r224r6oaq_v1134 as h181
    old.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only replay required')
    valid=validate()
    reg,common,failed,arms=cache.context()
    r6_reg,compact,r6_auth=h181.authenticated_r6()
    old.need(r6_reg['binding']['ids']==reg['binding']['ids'] and len(reg['binding']['ids'])==58,'ordered real cohort drift')
    ids=reg['binding']['ids']
    arrays=[np.asarray(common[k],float) for k in ('R','R224')]+[compact]+[np.asarray(common[k],float) for k in MEMBERS[3:]]
    r7_arrays=[a.copy() for a in arrays]; r7_arrays[2]=np.asarray(arms['flip-1']['predictions'],float)
    masks=[np.asarray(failed[k],bool) if k in failed else np.zeros(len(ids),bool) for k in MEMBERS]
    old.need(not any(m.any() for m in masks),'real replay requires healthy recorded rows')
    old.need(not any(np.all(a==.5,axis=1).any() for a in arrays[:2]),'unexpected real R sentinel')
    # Explicitly prove the measured compact contract equals this fixed parent/child.
    for name,digest in r6_auth['admitted_H174_exact_contracts'].items():
        old.need(old.digest(parent()[1][name])==digest and old.digest(payload()[2][name])==digest,'R6 measured contract differs')
    reference=authenticated(REFERENCE,REFERENCE_SOURCE,REFERENCE_RECEIPT,REFERENCE_RECEIPT_SHA,REFERENCE_KERNEL)[1]
    cross=cross_reference()
    for label,files in [('fluid',reference),('cross',cross)]:
        for name,data in files.items(): old.write(SCRATCH/(label+'-runtime')/name,data)
    for label,values in [('r6',arrays),('r7',r7_arrays)]:
        np.savez(SCRATCH/(label+'-real-input.npz'),ids=np.asarray(ids),**{'a'+str(i):a for i,a in enumerate(values)})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONPATH',None)
    runs={}; outputs={}
    for label in ('parent','candidate','fluid','cross'):
        source=SCRATCH/(('r6' if label in ('parent','candidate') else 'r7')+'-real-input.npz')
        output=SCRATCH/('real-'+label)
        proc=subprocess.run([sys.executable,'-B','-I','-c',REPLAY_SCRIPT,str(SCRATCH/(label+'-runtime')),str(source),str(output)],cwd=SCRATCH,env=env,capture_output=True,text=True)
        old.need(proc.returncode==0,'isolated replay failed: '+proc.stderr[-6000:])
        runs[label]=old.read(output.with_suffix('.json'))
        with np.load(output.with_suffix('.npz'),allow_pickle=False) as data: outputs[label]={k:data[k] for k in data.files}
    checks={label:{row['key']:row for row in run['checks']} for label,run in runs.items()}
    cases=[]
    for n in (2,len(ids)):
        for case,bits in enumerate(itertools.product((False,True),repeat=6)):
            key=f'n{n}-case{case}'
            row=checks['candidate'][key]
            old.need(all(checks[label][key]['status']==row['status'] for label in checks),'failure gate differs')
            if row['status']=='EXPECTED_R224_ABORT':
                cases.append(dict(key=key,status=row['status'])); continue
            current=[np.zeros(n,bool) for _ in arrays]
            for mask,bit in zip(current,bits): mask[0]=bit
            actual=outputs['candidate'][key]
            expected=oracle([a[:n] for a in arrays],current)
            old.need(actual.tobytes()==expected.tobytes(),'independent pandas byte parity: '+key)
            old.need(np.ascontiguousarray(actual[:,PROTECTED]).tobytes()==np.ascontiguousarray(outputs['parent'][key][:,PROTECTED]).tobytes(),'protected-ten bytes: '+key)
            old.need(row['protected_csv_sha256']==checks['parent'][key]['protected_csv_sha256'],'protected-ten CSV: '+key)
            for label in ('fluid','cross'):
                old.need(np.ascontiguousarray(actual[:,[7,9]]).tobytes()==np.ascontiguousarray(outputs[label][key][:,[7,9]]).tobytes(),'fluid-two bytes: '+key)
                old.need(row['fluid_csv_sha256']==checks[label][key]['fluid_csv_sha256'],'fluid-two CSV: '+key)
            cases.append(dict(key=key,status='PASS',independent_pandas_bytes_exact=True,protected_ten_bytes_exact=True,fluid_two_bytes_exact=True,csv_fields_exact=True))
    report=dict(status='PASS',kernel_sha256=valid['kernel_sha256'],real_two_studies=True,full_real_cohort=58,
        scenarios=64,two_study_successes=32,two_study_expected_r224_aborts=32,full_cohort_successes=64,
        protected_ten_float64_bytes_exact=True,protected_ten_csv_fields_exact=True,fluid_two_float64_bytes_exact=True,fluid_two_csv_fields_exact=True,
        independent_pandas_bytes_exact=True,cases=cases,embedded_runs=runs,
        cohort_sha256=old.digest(old.encoded(ids)),historical_binding_sha256=reg['binding_sha256'],r6_authentication=r6_auth,
        inputs_sha256={'R6':[old.digest(a.tobytes()) for a in arrays],'R7':[old.digest(a.tobytes()) for a in r7_arrays]},
        exposure='EXPOSED',clean_oof=False,limitation='Authenticated real cached serving arithmetic; no labels used for evaluation/selection or printed, no AUC/model forward/CUDA/Linux qualification. Explicit injected missing-member masks; R224 >3% preserved, not bypassed.')
    old.write(SCRATCH/'replay.json',old.encoded(report))
    return {k:v for k,v in report.items() if k not in ('cases','embedded_runs','r6_authentication')}



def admit():
    valid=validate(); smoke=old.read(SCRATCH/'replay.json')
    old.need(smoke['status']=='PASS' and smoke['kernel_sha256']==valid['kernel_sha256'],'CPU replay missing/stale')
    old.need(smoke['protected_ten_float64_bytes_exact'] and smoke['protected_ten_csv_fields_exact'],'protected-ten parity missing')
    old.need(smoke['fluid_two_float64_bytes_exact'] and smoke['fluid_two_csv_fields_exact'],'fluid-two parity missing')
    return dict(status='HOLD_PARENT_READING_AND_B_SLUG_GENERIC_GATE_AND_INDEPENDENT_REVIEW',parent_reading_status=eligibility(),qualified_for_submission=False,
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
