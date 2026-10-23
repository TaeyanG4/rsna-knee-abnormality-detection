"""H-B-181 builder 1.0.0: one EXPOSED hard-OA route on admitted H-B-174 R6.

Changelog 1.0.0: route public columns 5/6 to the compact success rank, with
the H180 byte-identical route with existing combined-R fallback. H174 private
R6 source/order retained. No serving/launch/timing or frozen edits.
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
VERSION = '1.1.34-r224r6oaq'
SLUG = 'taeyangg4/knee-lb-r224r6oaq-v1'
BASE = ROOT / '.local/scratch/astra-b5/hb174-gate/gate-copy-v145'
RECEIPT = ROOT / '.local/state/gates/knee-lb-r224r6fq-v1.json'
PARENT_KERNEL = 'c073de7e769251bc042b4754d9a0feae88ddf8bdd2802db610c699a2cd845716'
PARENT_RECEIPT = 'b548ca990a20ba6b0aad71f627ba3116fd6b707f30c510da00549b936074d762'
ROUTE_BASE = ROOT / '.local/scratch/astra-b5/hb180-gate/gate-copy-v145'
ROUTE_RECEIPT = ROOT / '.local/state/gates/knee-lb-r224flipoaq-v1.json'
ROUTE_KERNEL = '549db918091eab03328abb5bb5873ef3a80ae8ee6d8a5f0301865d1b3d1e7bcc'
CUMULATIVE_DIFF = 'full-cumulative-diff-vs-v1.1.24-r224flipq.txt'
PRIVATE_DATASETS = ['taeyangg4/knee-exp169-a172-v2-assets',
    'taeyangg4/knee-exp155-own0949-oai-v1-2-assets',
    'nartaa/rsna-knee-publication-swa-weights-20261007',
    'taeyangg4/knee-r6-fe055406-v1',
    'heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
    'taeyangg4/knee-hb126-codecs-cp313-v1']
OUT = old.BASE.parent / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/hb181-build/hb181'
BACKUP = ROOT / '.local/backups/hb181-build/hb181-parent'
SPEC = SCRATCH / 'SPEC.md'
PRE_REVIEW = ROOT / '.local/scratch/hb181-lead/hb181/PRE-REVIEW.md'
HELPER, WRAPPER = infrastructure.HELPER, infrastructure.WRAPPER
DIFF = 'full-diff-vs-v1.1.30-r224r6fq.txt'
ROUTE = '    public[:, [5,6]] = np.where(sf[:, None], rr[:, [5,6]], ss[:, [5,6]])\n'
FORMULA = ('0.85*r(P)+0.15*r(O); r_R=0.65*r(R384)+0.35*r(R224); '
           'P_t=r(R6flip)_t for Lateral OA and PF OA when compact succeeds, else r_R_t; '
           'other P=0.5*r_R+0.5*r(R6flip); '
           'O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46')
PROTECTED = (0,1,2,3,4,7,8,9,10,11)
MEMBERS = ('R','R224','S_R7','EXP143v2','EXP155','Q_A5q')


def admitted(base,receipt,receipt_hash,kernel_hash,notebook_hash,metadata_hash,archive_hash):
    old.pin(receipt,receipt_hash)
    old.need(old.read(receipt)['status']=='PASS','admitted reference required')
    old.need(gate.sha(gate.encoded(gate.kernel(base,gate.Inputs(ROOT))[2]['kernel_files']))==kernel_hash,
             'admitted kernel identity')
    old.pin(base/'kernel/release.ipynb',notebook_hash)
    old.pin(base/'kernel/kernel-metadata.json',metadata_hash)
    nb=old.read(base/'kernel/release.ipynb')
    files,blob,assignments=old.unpack(fast.restored(nb))
    old.need(len(files)==31 and old.digest(blob)==archive_hash,'admitted archive identity')
    return nb,files,blob,assignments


def parent():
    from tools import c85r7_r224r6fq_v1130 as h174
    value=admitted(BASE,RECEIPT,PARENT_RECEIPT,PARENT_KERNEL,
        '0a6de175679923172e6a3c4094a31828a45be6cb7192f41a2048a2d7aedd97ae',
        'abb9df52e51f3116de8c15dd8f846463489390c2aeb889c4ff3ca1d5c5bf8f27',
        '0e7fdf9bd87023f4e50214fc143f194460bccd69760977cf1bfef445f328c2f1')
    old.need(value[1]==h174.patched(infrastructure.parent()[1]),'H174 cumulative R6 patch drift')
    old.need(old.read(BASE/'kernel/kernel-metadata.json')['dataset_sources']==PRIVATE_DATASETS,
             'fixed private R6 dataset list drift')
    return value


def route_reference():
    value=admitted(ROUTE_BASE,ROUTE_RECEIPT,
        'bfafc01e97b3f08c2b37a645bf206156f4a124903ac031040bb24baebe7da177',ROUTE_KERNEL,
        'eba3a39ffdd8c1a685ea642887dae060482a97325a73a829f72a67c7dd341b37',
        '17671605c7480cf579eff7cb20312a9373135ef948368a28a2fdbe926cce97b9',
        '1a60d44820197e4ae1b826d10e0f2aedc3f3d3e093963015822723395c3f427e')
    old.need(old.digest(value[1][HELPER])=='bc644ab9eadf3c2120779ee07259d7859dbfbe6c4845cd6cc82016d6817bbdd8',
             'admitted OA helper drift')
    review=ROOT/'.local/scratch/slot-b/hb180/gate-review-out.md'
    old.pin(review,'82744d9fc49eb0ad4302b0b04c831ebe76a19ab29aee87664e8f73087dc81f44')
    old.need('VERDICT: PASS' in review.read_text(encoding='utf-8'),'H180 combined review missing')
    return value


def function_bytes(raw,name):
    text=raw.decode(); n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ''.join(text.splitlines(True)[n.lineno-1:n.end_lineno]).encode()


def pre_review():
    old.pin(SPEC,'86ef899dfd4620d8a55a1ec14357bd6c309cbb3a25f1f1ac970733a79e86e090')
    old.need('Verdict: PASS for CPU-only build/tests/replay of SPEC.md v1.0.0' in
             PRE_REVIEW.read_text(encoding='utf-8'), 'independent pre-execution PASS required')
    return dict(spec_sha256=old.digest(SPEC.read_bytes()),
                review_path=PRE_REVIEW.relative_to(ROOT).as_posix(),
                review_sha256=old.digest(PRE_REVIEW.read_bytes()))


def registration():
    parent(); route_reference()
    return dict(item='H-B-181', version=VERSION, kernel_id=SLUG,
        purpose='ONE admitted H180 hard-OA route ported byte-identically onto fixed admitted H174 R6 parent',
        parent=BASE.relative_to(ROOT).as_posix(), parent_kernel_sha256=PARENT_KERNEL,
        parent_receipt_sha256=PARENT_RECEIPT, parent_submission=57251520,
        parent_displayed_public=None, hidden_proven_base_submission=57224580,
        cumulative_base_kernel_sha256=infrastructure.PARENT_KERNEL, route_reference_kernel_sha256=ROUTE_KERNEL, formula=FORMULA, routed_columns=[5,6],
        protected_columns=list(PROTECTED), pre_execution=pre_review(),
        preregistration='register must precede prepare, tests and CPU replay; no labels used for evaluation/selection and none printed; pinned roster authentication may read labels internally',
        failure_rule='Compact success: success-only rank(compact). Compact failed: combined R .65/.35, or sole surviving R. All public failed: own only; own failed: public only; both failed: .5. Existing own 17/17/12 survivor renormalization and R224 >3% abort unchanged.',
        public_reading=dict(reference_value=.957,
            rule='submit only after user-supplied H174 displayed >=0.957; retain child only displayed >=max(0.957,H174 displayed); pending parent fails closed',
            parent_reading_required=True, parent_reading=None,
            tie_action='meets registered keep threshold only; no claim about hidden exact order',
            source='user-supplied result only; no score polling'),
        timing=dict(parent_conservative_hours_upper=6.83, source='user supplied H174 N32 conservative basis',
            inference_delta='none', child_measured_hours=None,
            limitation='Parent runtime is the basis, not a measured child bound. No new inference; only blend arithmetic.'),
        sources=['.local/scratch/slot-b/ideas-1016/debate-astra-REPORT.md sections 1, 5, 7',
                 '.local/scratch/slot-b/ideas-1016/debate-fable-REPORT.md', 'user fixed-parent instruction'],
        evidence='R6 Gold58 naive +0.0051, selection-adjusted approximately +0.0030 to +0.0035; D-B-206 same-sign OAI R6>R7; supplied evidence only; EXPOSED, not recomputed here',
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        runtime_qualification='No CUDA/Linux/model-forward qualification by this CPU-only builder',
        gate_binding='HOLD: B must use slug-generic gate v1.4.5 with authentic dispatcher builder/reviewer receipts')


def score(value):
    old.need(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,
             'finite user-supplied displayed public score required')
    return value


def eligibility(parent_displayed):
    if parent_displayed is None: return 'HOLD_PARENT_READING_PENDING'
    return 'ELIGIBLE_FOR_B_REVIEW' if score(parent_displayed)>=.957 else 'DROP_PARENT_BELOW_0957'


def decision(displayed_score,parent_displayed):
    state=eligibility(parent_displayed)
    if state!='ELIGIBLE_FOR_B_REVIEW': return state
    return 'RETAIN' if score(displayed_score)>=max(.957,parent_displayed) else 'DROP'


def patched(original):
    files = dict(original)
    s = files[HELPER]
    line = next(x for x in s.decode().splitlines() if x.startswith('FORMULA = '))
    s = old.replace(s, line, 'FORMULA = ' + repr(FORMULA))
    s = old.replace(s, '    return public\n', ROUTE + '    return public\n')
    s = old.replace(s,
        '    return .85 * rank(.5 * (.65 * rank(r) + .35 * rank(r224)) + .5 * rank(ours)) + .15 * rank(',
        '    public = .5 * (.65 * rank(r) + .35 * rank(r224)) + .5 * rank(ours)\n'
        '    public[:, [5,6]] = rank(ours)[:, [5,6]]\n'
        '    return .85 * rank(public) + .15 * rank(')
    reference=route_reference()[1][HELPER]
    for name in ('p_branch','reference'):
        old.need(function_bytes(s,name)==function_bytes(reference,name),'H180 route function drift: '+name)
    files[HELPER] = s
    s = old.replace(files[WRAPPER], old.digest(original[HELPER]), old.digest(s))
    for before, after in [
        ('VERSION = "1.1.30-r224r6fq"', 'VERSION = "'+VERSION+'"'),
        ('RELEASE = "C85R7+codecs R224 R6 FLIP_ALL"', 'RELEASE = "C85R7+codecs R224 R6 FLIP_ALL hard OA"'),
        ('KERNEL_ID = "taeyangg4/knee-lb-r224r6fq-v1"', 'KERNEL_ID = "'+SLUG+'"'),
        ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.30-r224r6fq"',
         'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v'+VERSION+'"'),
        ('formula="public cell 8: .5*(.65*r(R384)+.35*r(R224))+.5*r(S_R7); own-success ranks; float32",',
         'formula="public OA=rank(S_R6flip) with combined-R fallback; other public=.5*(.65*r(R384)+.35*r(R224))+.5*r(S_R7); own-success ranks; float32",')]:
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
        full+=rbuild.diff(original[k],files[k],'v1.1.30-r224r6fq/'+k,VERSION+'/'+k)
    for name,before,after in [('notebook',pnb,nb),('kernel-metadata.json',old.read(BASE/'kernel/kernel-metadata.json'),meta)]:
        full+=rbuild.diff(old.encoded(before),old.encoded(after),'parent/'+name,'candidate/'+name)
    return nb,meta,files,blob,manifest,full.encode()


def cumulative():
    """Prove H167 -> admitted H174 -> H181 without regenerating either parent."""
    from tools import c85r7_r224r6fq_v1130 as h174
    nb,meta,files,_=payload()
    pnb,original,_,_=infrastructure.parent()
    admitted_parent=parent()[1]
    old.need(admitted_parent==h174.patched(original),'cumulative H174 change drift')
    old.need(files==patched(h174.patched(original)),'cumulative change outside H174 plus OA')
    old.need(files[h174.PORT]==admitted_parent[h174.PORT] and
             files[h174.CHAIN]==admitted_parent[h174.CHAIN],'R6 port/base bytes changed')
    changed={k for k in files if files[k]!=original[k]}
    old.need(changed=={HELPER,WRAPPER,h174.PORT,h174.CHAIN},'cumulative member scope')
    proof=dict(status='PASS',base_kernel_sha256=infrastructure.PARENT_KERNEL,
        parent_kernel_sha256=PARENT_KERNEL,route_reference_kernel_sha256=ROUTE_KERNEL,
        construction='candidate == OA_patch(H174_R6_patch(H167)); admitted H174 == H174_R6_patch(H167)',
        changed=sorted(changed),unchanged=sorted(set(files)-changed),
        route_functions_byte_identical_to_H180=['p_branch','reference'],
        private_R6_dataset_list_exact=PRIVATE_DATASETS,
        members={k:dict(h167=old.digest(original[k]),h174=old.digest(admitted_parent[k]),
                        h181=old.digest(files[k])) for k in sorted(files)})
    full='H167 -> admitted H174 R6 changes -> H181 hard OA; 27 IDENTICAL, 4 CHANGED\n'
    for k in sorted(files):
        full+=f'{k}: {old.digest(original[k])} -> {old.digest(files[k])}\n'
        full+=rbuild.diff(original[k],files[k],'H167/'+k,'H181/'+k)
    for name,a,b in [('notebook',pnb,nb),('kernel-metadata.json',old.read(infrastructure.BASE/'kernel/kernel-metadata.json'),meta)]:
        full+=rbuild.diff(old.encoded(a),old.encoded(b),'H167/'+name,'H181/'+name)
    return proof,full.encode()


CHANGELOG = b'''# v1.1.34-r224r6oaq
H-B-181: byte-identical admitted H180 hard-OA p_branch/reference functions
on fixed admitted H174 R6 parent. Only columns 5/6 route to compact success
rank, with original combined-R fallback and outer own .15 unchanged.
H174 private R6 attachments, serving, workers, ranks, and all ten other
targets unchanged. Dependent helper pin, formula metadata, version/id only.
No original/frozen code or scored stage edits. Parent backup:
.local/backups/hb181-build/hb181-parent. EXPOSED, no new inference.
Submission requires user-supplied H174 displayed >=.957; pending closes it.
Retain child iff displayed >=max(.957,H174 displayed). B owns admission.
'''


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
    cp,cd=cumulative()
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
        ('cumulative-proof.json',old.encoded(cp)),(CUMULATIVE_DIFF,cd),('CHANGELOG.md',CHANGELOG),
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
    cp,cd=cumulative()
    for name,data in [('registration.json',old.encoded(registration())),('kernel/release.ipynb',old.encoded(nb)),
        ('cumulative-proof.json',old.encoded(cp)),(CUMULATIVE_DIFF,cd),('CHANGELOG.md',CHANGELOG),
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
                identical=29,changed=2,cumulative_identical=27,cumulative_changed=4,
                H180_route_functions_byte_identical=True,
                metadata_raw_values_identical_except_id_title=True)


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
        if col in (5,6):
            public=s.copy()
            public[sf]=rr[sf]
        else:
            public=.5*rr+.5*s
            public[rf & f224]=s[rf & f224]
            public[sf]=rr[sf]
        pf=rf & f224 & sf
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


def authenticated_r6():
    """Read-only validation of closed serving artifacts; never invokes evaluate/run."""
    import numpy as np
    from tools import r6_gold58_precheck_v1 as r6
    sealed={
        'registration.json':'3efe812179c2218117a3d5ed4d3a8e679e0157ab434d42e62c36fba48929e293',
        'measurement.json':'d37804bfaf899ca1e817df51370c549770360a5f063fd04a43f3cd02535a4ac8',
        'attempt.json':'a89f3ec58d956b5337c5890bac442440eb50c17c2925cef2d15eac75bd3a4347',
        'validate.json':'3800a8f16c97c762eb70ff31629dd19cf20f8708b60e8801f7f8b030a5bfd2e9',
        'gpu-review.json':'2c4a294a96fae52a7cac2c26b761e7f662233bf2fd0916f61bbe9e9762d1deeb'}
    for name,h in sealed.items(): old.pin(r6.WORK/name,h)
    reg=r6.review_ready()  # Includes every historical input pin. No stale-input bypass.
    old.need(reg['binding_sha256']=='b2ddd615f9321d0acbcd08a51d2788d2c03a660693935a0837d08da6002bf0da',
             'R6 historical binding drift')
    measured=old.read(r6.WORK/'measurement.json'); attempt=old.read(r6.WORK/'attempt.json')
    old.need(set(attempt['arms'])==set(r6.ARMS),'complete R6 pair required')
    old.need(measured['state']==attempt['state']=='COMPLETE_REQUIRES_INDEPENDENT_RESULT_REVIEW' and
        measured['binding_sha256']==attempt['binding_sha256']==reg['binding_sha256'] and
        measured['attempt_sha256']==r6.sha(r6.WORK/'attempt.json') and
        measured['review_sha256']==r6.sha(r6.WORK/'gpu-review.json'),'R6 measurement seal drift')
    for name,h in measured['files'].items():
        path=(r6.WORK/name).resolve()
        old.need(path.is_relative_to(r6.WORK.resolve()) and r6.sha(path)==h,'measured R6 artifact drift: '+name)
    pf=parent()[1]
    contract=r6.WORK/'contracts/flip'
    exact=['exp169_r7_port.py','exp189_s6_port.py']+[k for k in pf if k.startswith('exp169_r7_src/')]
    for name in exact:
        old.need((contract/name).read_bytes()==pf[name],'admitted H174 vs measured R6 contract drift: '+name)
    for arm,value in attempt['arms'].items():
        r6.check_measurement(value,reg,arm,value['desktop'],value['telemetry'])
        sup=old.read(r6.WORK/'gpu'/arm/'supervision.json')
        old.need(sup['exit_code']==0 and sup['error'] is None and sup['cleanup']==[],'R6 supervision failure')
        actual,_=r6.check_controller(r6.WORK/'gpu'/arm/'output',reg['binding']['ids'],arm,'cuda')
        old.need(np.array_equal(actual,value['predictions']),'R6 measured predictions differ from controller CSV')
    return reg,np.asarray(attempt['arms']['flip']['predictions'],float),dict(
        binding_sha256=reg['binding_sha256'],sealed_files=sealed,
        admitted_H174_exact_contracts={k:old.digest(pf[k]) for k in exact},
        historical_inputs_valid=True,closed_complete_pair=True,controller_csv_predictions_exact=True,
        R6_identity=reg['binding']['r6_identity'])


def replay():
    import numpy as np
    import torch
    from tools import c85r7_gold58_candidates as cache
    old.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only replay required')
    valid=validate()
    # Proven authentication retained: roster('gold') internally reads the label
    # table for IDs. Labels never enter this arithmetic/evaluation or its output.
    reg,common,failed,arms=cache.context()
    r6_reg,compact,r6_auth=authenticated_r6()
    old.need(r6_reg['binding']['ids']==reg['binding']['ids'] and len(reg['binding']['ids'])==58,
             'R6/H167 full Gold58 ordered cohort drift')
    ids=reg['binding']['ids'][:2]
    arrays=[np.asarray(common[k],float)[:2] for k in ('R','R224')]
    arrays += [compact[:2]]
    arrays += [np.asarray(common[k],float)[:2] for k in MEMBERS[3:]]
    masks=[np.asarray(failed[k],bool)[:2].copy() if k in failed else np.zeros(2,bool) for k in MEMBERS]
    old.need(not any(m.any() for m in masks),'real two-study smoke requires healthy recorded rows')
    old.need(not any(np.all(a==.5,axis=1).any() for a in arrays[:2]),'unexpected real R sentinel')
    cases=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    def protected_csv(path):
        rows=path.read_bytes().splitlines()
        return b'\n'.join(b','.join(row.split(b',')[j] for j in (0,*[i+1 for i in PROTECTED])) for row in rows)
    scenarios={'observed_healthy':(), 'injected_compact_failure':(2,),
        'injected_own_failure':(3,4,5), 'injected_partial_own_failure':(3,),
        'injected_r384_failure':(0,), 'injected_compact_and_own_failure':(2,3,4,5)}
    for case,indices in scenarios.items():
        current=[m.copy() for m in masks]
        for i in indices: current[i][0]=True
        source=SCRATCH/('real2-'+case+'-input.npz')
        np.savez(source,ids=np.asarray(ids),**{'a'+str(i):a for i,a in enumerate(arrays)},
                 **{'m'+str(i):m for i,m in enumerate(current)})
        imports=[]
        for label in ('parent','candidate'):
            output=SCRATCH/('real2-'+case+'-'+label)
            run=subprocess.run([sys.executable,'-B','-I','-c',REPLAY_SCRIPT,
                str(SCRATCH/(label+'-runtime')),str(source),str(output)],cwd=SCRATCH,env=env,capture_output=True,text=True)
            old.need(run.returncode==0,'isolated real2 replay failed: '+run.stderr[-6000:])
            imports.append(json.loads(run.stdout.strip().splitlines()[-1]))
        prior,result=[np.load(SCRATCH/('real2-'+case+'-'+label+'.npy'),allow_pickle=False) for label in ('parent','candidate')]
        expected=oracle(arrays,current)
        old.need(result.tobytes()==expected.tobytes(),'independent pandas byte parity')
        a,b=[np.ascontiguousarray(v[:,PROTECTED]).tobytes() for v in (prior,result)]
        old.need(a==b,'protected-ten float64 bytes changed')
        aa,bb=[protected_csv(SCRATCH/('real2-'+case+'-'+label+'.csv')) for label in ('parent','candidate')]
        old.need(aa==bb,'protected-ten serialized CSV fields changed')
        cases.append(dict(case=case,first_study_injected_failed_members=[MEMBERS[i] for i in indices],
            observed_failures=False,embedded_runs=imports,independent_pandas_bytes_exact=True,
            protected_ten_float64_bytes_exact=True,protected_ten_csv_fields_exact=True,
            protected_float64_sha256=old.digest(a),protected_csv_sha256=old.digest(aa),
            candidate_output_sha256=old.digest(result.tobytes()),
            candidate_csv_sha256=old.digest((SCRATCH/('real2-'+case+'-candidate.csv')).read_bytes())))
    report=dict(status='PASS',kernel_sha256=valid['kernel_sha256'],studies=2,targets=12,
        source='authenticated R6 Gold58 FULL-flip compact plus preserved H167 five common members, first two aligned registered IDs',
        cohort_sha256=old.digest(old.encoded(ids)),historical_binding_sha256=reg['binding_sha256'], r6_authentication=r6_auth,
        preserved_pins_sha256=cache.PINS_SHA,inputs_sha256=[old.digest(a.tobytes()) for a in arrays],
        independent_pandas_bytes_exact=True,protected_ten_float64_bytes_exact=True,
        protected_ten_csv_fields_exact=True,cases=cases,
        exposure='EXPOSED',clean_oof=False,
        limitation='Real cached-serving arithmetic replay of actual embedded imports, healthy and explicitly injected failures. Pinned roster authentication reads the label table internally for IDs; no labels are used for evaluation/selection or printed. No AUC, model forward, CUDA/Linux qualification or runtime estimate.')
    old.write(SCRATCH/'replay.json',old.encoded(report))
    return report


def admit():
    valid=validate(); smoke=old.read(SCRATCH/'replay.json')
    old.need(smoke['status']=='PASS' and smoke['kernel_sha256']==valid['kernel_sha256'],'CPU replay missing/stale')
    old.need(smoke['protected_ten_float64_bytes_exact'] and smoke['protected_ten_csv_fields_exact'],'protected-ten parity missing')
    return dict(status='HOLD_PARENT_READING_AND_B_SLUG_GENERIC_GATE_AND_INDEPENDENT_REVIEW',
                parent_reading_status=eligibility(None),qualified_for_submission=False,
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
