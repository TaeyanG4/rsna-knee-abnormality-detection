"""H-B-168 builder 1.0.0: R224 + FULL flip parent, fluid route only.

Changelog: strict single-candidate arithmetic patch; compo15 rejected before
evaluation because its native S43/Q7S execution cannot map onto the C85 O slot.
No frozen files, launch mechanics, GPU execution or gate receipts are changed.
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
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_fluidr_v119 as old
from tools import c85r7_flipallq_v1122 as fast
from tools import c85r7_r224q_v1123 as rbuild
from tools import strict_variant_gate as gate

VERSION = '1.1.25-stackq'
SLUG = 'taeyangg4/knee-lb-stackq-v1'
BASE = ROOT / '.local/scratch/astra-b5/hb167-gate/gate-copy-v143'
RECEIPT = ROOT / '.local/state/gates/knee-lb-r224flipq-v1.json'
OUT = old.BASE.parent / ('v' + VERSION)
COMPO = ROOT / 'artifacts/experiments/HB116_VNEXT/v1.3.0-compo15'
SCRATCH = ROOT / '.local/scratch/astra-b1/hb168'
HELPER, WRAPPER = 'exp169_r224.py', 'exp169_release_c85r7codec.py'
PARENT_KERNEL = '04a3344e1d3d44ed5abc42e34aa1f6fae9f37e1f77470ef47eb2540046517ae5'
PARENT_RECEIPT = 'd76d96d085d0b7c8f467d292f94f2c5e15f649176e6f29c63591b2e8df571378'
FORMULA = '0.85*r(P)+0.15*r(O); r_R=0.65*r(R384)+0.35*r(R224); P_t=r_R_t for Effusion and Bakers Cyst, otherwise P=0.5*r_R+0.5*r(R7); O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46'


def parent():
    old.pin(RECEIPT, PARENT_RECEIPT)
    old.need(old.read(RECEIPT)['status'] == 'PASS', 'parent not admitted')
    kfiles = gate.kernel(BASE, gate.Inputs(ROOT))[2]['kernel_files']
    old.need(gate.sha(gate.encoded(kfiles)) == PARENT_KERNEL, 'wrong parent kernel')
    nb = old.read(BASE / 'kernel/release.ipynb')
    files, blob, assignments = old.unpack(fast.restored(nb))
    old.need(len(files) == 31, 'parent roster')
    old.pin(BASE / 'kernel/release.ipynb', 'c96f5d0d9ed53c5a57870b97eca4e03fd4226c131fa79489d6bea1d02956c95a')
    return nb, files, blob, assignments


def references():
    old.pin(old.OUT / 'build-receipt.json', 'a764eae1d30627ff833e4dd7ec3839d1d4e1e6c74163ded919845519cc32c23a')
    old.pin(old.OUT / 'kernel/release.ipynb', 'f0911784fb46986151e2afcb9446448badf333f66edbf5ebc4051b553cf62dfa')
    fluid = old.unpack(old.read(old.OUT / 'kernel/release.ipynb'))[0]
    original = old.parent()[1]
    old.need(fluid == old.allowed_files(original), 'fluid reference patch drift')
    old.need(old.ROUTE.encode() in fluid['exp169_release_c85.py'], 'fluid route provenance')
    old.pin(COMPO / 'source-pins.json', '7393815bd57689a10d0d3d8b2def94c6dbfcf59d9ad92c1f2326113020ba6c0c')
    old.pin(COMPO / 'kernel/release.ipynb', '0d0c1b93d801d19646a1d14c4eba78a1bd9f564d1c196e23749b6f97d603eb11')
    tree = ast.parse(''.join(old.read(COMPO / 'kernel/release.ipynb')['cells'][1]['source']))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'blob' for t in n.targets))
    blob = base64.b64decode(ast.literal_eval(node.value.args[0]))
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        native = {name: z.read(name) for name in z.namelist()}
        old.need(len(native) == len(z.namelist()), 'duplicate native member')
    old.need({k:old.digest(v) for k,v in native.items()} == old.read(COMPO / 'source-pins.json'), 'compo archive pins')
    return fluid, native


def mapping():
    _, native = references()
    _, c85, _, _ = parent()
    n = native['tools/hb116_vnext_release.py'].decode()
    old.need("(['EXP143v2','S43'] if s43 else ['EXP143v2'])" in n, 'OWN2 source')
    old.need("for family in ('R50','C4','S')" in n, 'Q7S source')
    old.need('tools/hb116_native_runtime.py' in native and 'hb116_native_runtime.py' not in c85, 'native execution identity')
    return dict(status='NOT_STRICTLY_MAPPABLE_COMPO15_OMITTED', native_files=len(native), c85_files=len(c85),
        c85_own='(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46',
        compo_own='OWN2=mean_prob(EXP143v2,S43); O_A=(17*r(OWN2)+17*r(EXP155)+12*r(Q7S))/46',
        c85_q='20 checkpoints: l2 seeds42/43, dino42, convnext42; five folds each',
        compo_q='25 checkpoints: R50 seeds42/43, C4 seed42, S seeds42/43; five folds each. R50 seeds mean within fold then folds mean; C4/S member means; family means float32; Q7S mean of family ranks.',
        failure_difference='Native mean_survivors/ranks use per-study finite masks and NaN until final fill; C85 uses explicit failures and zero-filled scored ranks.',
        reason='S43 inference and native Q7S roster/aggregation are not served by the C85 O slot. Importing native serving, adding worker/model execution or replacing aggregation is a new code path, not the permitted strict byte patch.',
        action='Build only fluid addition; B must re-register the candidate as R224 + FULL flip + fluid. No compo weights/code copied.',
        authenticated_native_pins={k:old.digest(native[k]) for k in ['tools/hb116_vnext_release.py','tools/hb116_native_runtime.py','tools/c85r7_lb_compo15_runtime.py']})


def registration():
    return dict(item='H-B-168', version=VERSION, kernel_id=SLUG, purpose='final-pair candidate',
        parent=str(BASE.relative_to(ROOT)).replace('\\','/'), parent_kernel_sha256=PARENT_KERNEL,
        parent_receipt_sha256=PARENT_RECEIPT, parent_submission=57224580,
        parent_hidden_status='Still scoring at dispatch; admitted bytes are not yet a hidden-completed combined parent.',
        formula=FORMULA, changes=['Fluid route only; compo15 deliberately omitted under user fallback'],
        preregistration='Written by register before prepare, tests or CPU replay.',
        interaction='r_R uses independent success-only ranks of R384 and R224, weighted .65/.35, with existing single-survivor fallback. Fluid P=r_R, then outer success-only rank(r_R); never raw R384 only, never unranked r_R in the final mixture. Other ten targets unchanged.',
        failure_rule='For fluid targets failed_R=failed_R384 AND failed_R224. A surviving R arm is used alone. Both R arms failed => O only even if R7 succeeds; O failed => ranked combined R only; both failed => 0.5. Existing R224 >3% abort remains. R7 never influences fluid ranks or failure eligibility.',
        public_reading=dict(reference='H-B-167 user-supplied reading', reference_value=None, min_delta=-.001,
                            rule='keep iff candidate reading >= H-B-167 reading - 0.001; no score polling'),
        timing=dict(limit_hours=9, measured_hidden_hours=None, delta_seconds=None, assumption='No added model inference; extra rank arithmetic only; wall-time delta unmeasured. Parent hidden still pending; do not add unpaired historical runtimes.'),
        sources=['D-B-163','D-B-171','D-B-193','D-B-194','D-B-195'],
        scientific_limit='EXPOSED evidence only; D-B-193/194/195 CHALLENGED. No additive gain, clean OOF, runtime bound or final-pair qualification inferred.',
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        gate_binding='DEFERRED separate exact-diff pin and independent review; no gate copy', compo_mapping=mapping())


def decision(parent_score, candidate_score, measured_hours):
    numbers = [parent_score,candidate_score,measured_hours]
    old.need(all(type(v) in (int,float) and math.isfinite(v) for v in numbers), 'finite measured decisions required')
    old.need(0 <= parent_score <= 1 and 0 <= candidate_score <= 1 and measured_hours > 0, 'decision domain')
    return candidate_score >= parent_score-.001 and measured_hours <= 9


def patched(original):
    files = dict(original)
    s = files[HELPER]
    line = next(x for x in s.decode().splitlines() if x.startswith('FORMULA = '))
    s = old.replace(s, line, 'FORMULA = ' + repr(FORMULA))
    s = old.replace(s, '    return public\n', '    public[:, [7,9]] = rr[:, [7,9]]\n    return public\n')
    # Keep the proven fluid override byte-for-byte. The adapter supplies its
    # R rank/mask from the already registered combined R half, not raw R384.
    route = ('    rr, rf = r_half(r, r224, rf, f224)\n'
             '    rr = c85.scored_rank(rr, rf)\n' + old.ROUTE)
    s = old.replace(s, '    return result\n\n\ndef reference(', route + '    return result\n\n\ndef reference(')
    s = old.replace(s,
        '    return .85 * rank(.5 * (.65 * rank(r) + .35 * rank(r224)) + .5 * rank(ours)) + .15 * rank(',
        '    rr = .65 * rank(r) + .35 * rank(r224)\n    public = .5 * rr + .5 * rank(ours)\n    public[:, [7,9]] = rr[:, [7,9]]\n    return .85 * rank(public) + .15 * rank(')
    files[HELPER] = s
    s = old.replace(files[WRAPPER], old.digest(original[HELPER]), old.digest(files[HELPER]))
    for before,after in [
        ('VERSION = "1.1.24-r224flipq"', 'VERSION = "'+VERSION+'"'),
        ('RELEASE = "C85R7+codecs R224 FLIP_ALL"', 'RELEASE = "C85R7+codecs R224 FLIP_ALL fluidR"'),
        ('KERNEL_ID = "taeyangg4/knee-lb-r224flipq-v1"', 'KERNEL_ID = "'+SLUG+'"'),
        ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq"', 'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v'+VERSION+'"'),
        ('formula="public cell 8: .5*(.65*r(R384)+.35*r(R224))+.5*r(S_R7); own-success ranks; float32",', 'formula="r_R=.65*r(R384)+.35*r(R224); P=r_R for Effusion/Bakers, else .5*r_R+.5*r(S_R7); own-success ranks; float32",')]:
        s = old.replace(s,before,after)
    files[WRAPPER] = s
    return files


def payload():
    nb, original, old_blob, assignments = parent()
    references()
    files = patched(original)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source, zipfile.ZipFile(buf,'w') as target:
        for info in source.infolist(): target.writestr(info, files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before,after in [
        ('BLOB = '+repr(assignments['BLOB']), 'BLOB = '+repr(base64.b64encode(blob).decode())),
        ('PINS = '+repr(assignments['PINS']), 'PINS = '+repr({k:old.digest(v) for k,v in files.items()})),
        (repr(old.digest(old_blob)),repr(old.digest(blob)))]:
        code = old.replace(code,before,after)
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    meta = old.read(BASE/'kernel/kernel-metadata.json')
    meta.update(id=SLUG,title=SLUG.split('/')[1])
    return nb,meta,files,blob


def scope(nb,meta):
    expected,em,files,blob = payload()
    old.need(nb==expected and meta==em, 'forbidden notebook/metadata diff')
    actual,ab,_ = old.unpack(fast.restored(nb))
    old.need(actual==files and ab==blob,'forbidden embedded diff')
    original = parent()[1]
    old.need(set(files)==set(original) and {k for k in files if files[k]!=original[k]}=={HELPER,WRAPPER},'member scope')
    for name,data in files.items():
        if name.endswith('.py'): compile(data,name,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    return files,blob


def artifacts():
    nb,meta,files,blob = payload()
    original_nb, original, _, _ = parent()
    manifest = dict(version=VERSION,added=[],removed=[],changed=[],unchanged=[])
    full = 'ALL 31 EMBEDDED FILES: 29 IDENTICAL, 2 REGISTERED CHANGES; NO ADDED/REMOVED MEMBERS\n'
    for k in sorted(files):
        a,b = old.digest(original[k]),old.digest(files[k])
        changed = a!=b
        manifest['changed' if changed else 'unchanged'].append(dict(name=k,before_sha256=a,after_sha256=b))
        full += f'{k}: {a} -> {b} '+('REGISTERED_CHANGE' if changed else 'IDENTICAL')+'\n'
        full += rbuild.diff(original[k],files[k],'v1.1.24-r224flipq/'+k,VERSION+'/'+k)
    for name,before,after in [('notebook',original_nb,nb),('kernel-metadata.json',old.read(BASE/'kernel/kernel-metadata.json'),meta)]:
        full += rbuild.diff(old.encoded(before),old.encoded(after),'parent/'+name,'candidate/'+name)
    return nb,meta,files,blob,manifest,full.encode()


def register():
    reg=registration()
    old.write(OUT/'registration.json',old.encoded(reg))
    old.write(OUT/'compo-mapping.json',old.encoded(reg['compo_mapping']))
    return dict(status='REGISTERED_BEFORE_EVALUATION',compo15='OMITTED',formula=FORMULA)


def prepare():
    old.need((OUT/'registration.json').read_bytes()==old.encoded(registration()),'register first')
    nb,meta,files,blob,manifest,full=artifacts()
    scope(nb,meta)
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
                      ('new-members.json',old.encoded(manifest)),('full-diff-vs-v1.1.24-r224flipq.txt',full)]:
        old.write(OUT/name,data)
    for name,data in files.items(): old.write(SCRATCH/'candidate-runtime'/name,data)
    rec=dict(status='PREPARED_CPU_ONLY',version=VERSION,builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],archive_sha256=old.digest(blob),
        embedded={k:old.digest(v) for k,v in files.items()},unchanged=29,changed=[HELPER,WRAPPER],added=[],removed=[],
        qualified_for_submission=False,exposure='EXPOSED',clean_oof=False)
    old.write(OUT/'build-receipt.json',old.encoded(rec))
    return validate()


def validate():
    nb,meta,files,blob,manifest,full=artifacts()
    for name,data in [('registration.json',old.encoded(registration())),('compo-mapping.json',old.encoded(mapping())),
        ('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
        ('new-members.json',old.encoded(manifest)),('full-diff-vs-v1.1.24-r224flipq.txt',full)]:
        old.need((OUT/name).read_bytes()==data,'stage drift: '+name)
    scope(old.read(OUT/'kernel/release.ipynb'),old.read(OUT/'kernel/kernel-metadata.json'))
    rec=old.read(OUT/'build-receipt.json')
    old.need(rec['builder_sha256']==old.digest(Path(__file__).read_bytes()),'builder drift')
    old.need(rec['embedded']=={k:old.digest(v) for k,v in files.items()} and rec['archive_sha256']==old.digest(blob),'archive drift')
    old.need(rec['kernel_files']==gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],'kernel drift')
    old.need(rec['qualified_for_submission'] is False and rec['unchanged']==29,'receipt scope')
    return dict(status='PASS_CPU_VALIDATE',kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),identical=29,changed=2)


def math_functions(files):
    import numpy as np
    from tools import exp169_r224 as h
    ns=dict(np=np,c85=h.c85,p=h.p,v5q=h.v5q)
    tree=ast.parse(files[HELPER])
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['failure_gate','r_half','p_branch','blend','reference']]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<candidate-embedded-math>','exec'),ns)
    return ns


def replay():
    import numpy as np
    import pandas as pd
    from tools import exp169_release_c85r7codec as runtime
    valid=validate()
    old.pin(old.VISIBLE/'c85r7-receipt.json',old.VISIBLE_SHA)
    rec=old.read(old.VISIBLE/'c85r7-receipt.json')
    old.need(rec['status']=='COMPLETE_C85R7' and rec['studies']==3 and all(v==0 for v in rec['fallbacks'].values()),'completed real3 required')
    old.pin(old.VISIBLE/'submission.csv',rec['submission_sha256'])
    ids,final=runtime.v5q._read_csv(old.VISIBLE/'submission.csv')
    inputs=[]
    for name,pin in old.VISIBLE_INPUTS.items():
        old.pin(old.VISIBLE/name,pin)
        u,a=runtime.v5q._read_csv(old.VISIBLE/name)
        old.need(len(u)==len(set(u))==3 and set(u)==set(ids),'real3 cohort')
        inputs.append(a[[u.index(uid) for uid in ids]])
    r,s=inputs
    rank=lambda a:pd.DataFrame(a).rank(method='average',pct=True).to_numpy()
    o=old.recover_own_rank(final,rank(.5*rank(r)+.5*rank(s)))
    legacy=old.math_functions(old.parent()[1])['blend'](ids,r,s,o,o,o)
    old.need(runtime.p.csv_bytes(ids,legacy)==(old.VISIBLE/'submission.csv').read_bytes(),'real3 baseline byte parity')
    new=math_functions(payload()[2]); prior=math_functions(parent()[1])
    summaries=[]
    # R224 predictions were not present in the completed real3 run. Two explicit
    # rank-surrogates exercise interaction without inventing measured R224 data.
    for label,r224 in [('identity_surrogate',r.copy()),('discordant_surrogate',r[::-1].copy())]:
        result=new['blend'](ids,r,r224,s,o,o,o)
        rr=.65*rank(r)+.35*rank(r224)
        pp=.5*rr+.5*rank(s); pp[:,[7,9]]=rr[:,[7,9]]
        expected=.85*rank(pp)+.15*o
        np.testing.assert_array_equal(result,expected)
        np.testing.assert_array_equal(result,new['reference'](r,r224,s,o,o,o))
        keep=[j for j in range(12) if j not in (7,9)]
        np.testing.assert_array_equal(result[:,keep],prior['blend'](ids,r,r224,s,o,o,o)[:,keep])
        output=runtime.p.csv_bytes(ids,result)
        old.write(SCRATCH/(label+'.csv'),output)
        summaries.append(dict(case=label,output_sha256=old.digest(output),exact_pandas=True,unchanged_ten=True))
    report=dict(status='PASS',kernel_sha256=valid['kernel_sha256'],studies=3,targets=12,
        baseline_byte_parity=True,source_receipt_sha256=old.VISIBLE_SHA,input_pins=old.VISIBLE_INPUTS,cases=summaries,
        exposure='EXPOSED',clean_oof=False,
        limitation='Real completed R384/R7 predictions and uniquely recovered own-rank sufficient statistics; R224 is explicitly synthetic, not measured. No claim of full R224+flip neural replay, CUDA qualification or runtime measurement.')
    old.write(SCRATCH/'replay.json',old.encoded(report))
    return report


def admit():
    valid=validate(); smoke=old.read(SCRATCH/'replay.json')
    old.need(smoke['status']=='PASS' and smoke['kernel_sha256']==valid['kernel_sha256'],'CPU replay missing/stale')
    return dict(status='HOLD_INDEPENDENT_REVIEW_EXACT_DIFF_PIN_AND_RUNTIME',qualified_for_submission=False,kernel_sha256=valid['kernel_sha256'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','prepare','validate','cpu-replay','admit'])
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    if args.command in ('prepare','admit'): old.need(args.dry_run,'CPU dry-run required')
    print(json.dumps({'register':register,'prepare':prepare,'validate':validate,'cpu-replay':replay,'admit':admit}[args.command](),indent=2))


if __name__=='__main__': main()
