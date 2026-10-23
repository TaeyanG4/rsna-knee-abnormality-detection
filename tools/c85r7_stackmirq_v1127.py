"""H-B-172 strict composition builder 1.0.0; CPU only.

Changelog: byte-identical admitted fluid and mirror composition, explicit pin
merge, three-way diffs and existing real-data replay adapters. Originals remain
untouched; backup .local/backups/hb172-build/hb172.
"""
from __future__ import annotations
import argparse
import ast
import base64
from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import c85r7_r224mirq_v1126 as mirror
from tools import c85r7_stackq_v1125 as fluid
from tools import strict_variant_gate_r224_v144 as gate144
old,gate,flipq,r224=mirror.old,mirror.gate,mirror.flipq,mirror.r224
VERSION='1.1.27-stackmirq'
SLUG='taeyangg4/knee-lb-stackmirq-v1'
OUT=ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7'/('v'+VERSION)
SCRATCH=ROOT/'.local/scratch/hb172-build/hb172'
BACKUP=ROOT/'.local/backups/hb172-build/hb172'
BASE=mirror.BASE
STAGES={'parent':BASE,'fluid':ROOT/'.local/scratch/astra-b5/hb168-gate/gate-copy-v144',
        'mirror':ROOT/'.local/scratch/astra-b5/hb170-gate/gate-copy-v144'}
SLUGS={'parent':'knee-lb-r224flipq-v1','fluid':'knee-lb-stackq-v1','mirror':'knee-lb-r224mirq-v1'}
SOURCE=mirror.SOURCE
WRAPPER=mirror.WRAPPER
HELPER=fluid.HELPER


def cpu_only():
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')


def snapshots():
    result={}
    for name,path in STAGES.items():
        inputs=gate.Inputs(ROOT)
        receipt=ROOT/'.local/state/gates'/(SLUGS[name]+'.json')
        rec=gate144.verify_receipt(receipt,inputs)
        old.need(gate.local_path(rec['stage'],ROOT)==path,'admitted stage mismatch')
        nb=old.read(path/'kernel/release.ipynb')
        files,blob,assignments=old.unpack(flipq.restored(nb))
        old.need(len(files)==31,'member count')
        result[name]=dict(nb=nb,files=files,blob=blob,assignments=assignments,
            meta=old.read(path/'kernel/kernel-metadata.json'),receipt_sha256=gate.sha(receipt.read_bytes()),
            kernel_files=rec['stage_inventory']['kernel_files'])
        inputs.recheck()
    return result


def merge(s):
    p,f,m=(s[k]['files'] for k in ('parent','fluid','mirror'))
    old.need(set(p)==set(f)==set(m),'member set drift')
    old.need({k for k in p if p[k]!=f[k]}=={HELPER,WRAPPER},'fluid scope drift')
    old.need({k for k in p if p[k]!=m[k]}=={SOURCE,WRAPPER},'mirror scope drift')
    # Authenticate each historical transformation too; inputs come from gates.
    old.need(fluid.patched(p)==f and mirror.patch_files(p)==m,'admitted hunk identity drift')
    combined=dict(p,**{HELPER:f[HELPER],SOURCE:m[SOURCE]})
    w=old.replace(f[WRAPPER],old.digest(p[SOURCE]),old.digest(m[SOURCE]))
    w=fluid_version(w)
    w=old.replace(w,fluid.SLUG,SLUG)
    combined[WRAPPER]=w
    # Independent merge direction: fluid wrapper hunks applied to mirror wrapper.
    import difflib
    a=p[WRAPPER].splitlines(keepends=True); z=f[WRAPPER].splitlines(keepends=True)
    branch=m[WRAPPER]
    branch=old.replace(branch,mirror.VERSION,'1.1.24-r224flipq')
    branch=old.replace(branch,mirror.SLUG,'taeyangg4/knee-lb-r224flipq-v1')
    hunks=[]
    for tag,i,j,x,y in difflib.SequenceMatcher(a=a,b=z,autojunk=False).get_opcodes():
        if tag=='equal': continue
        before,after=b''.join(a[i:j]),b''.join(z[x:y])
        old.need(before and branch.count(before)==1,'overlapping fluid hunk')
        branch=branch.replace(before,after,1)
        hunks.append(dict(before=before.decode(),after=after.decode(),after_sha256=old.digest(after)))
    branch=fluid_version(branch)
    branch=old.replace(branch,fluid.SLUG,SLUG)
    old.need(branch==w,'two merge orders disagree')
    proof=dict(source_members={HELPER:dict(source='fluid',sha256=old.digest(f[HELPER]),byte_identical=True),
        SOURCE:dict(source='mirror',sha256=old.digest(m[SOURCE]),byte_identical=True)},
        wrapper_fluid_hunks=hunks,mirror_pin=dict(before=old.digest(p[SOURCE]),after=old.digest(m[SOURCE])),
        overlapping_version_resolution=VERSION,overlapping_kernel_id_resolution=SLUG,
        fluid_out_metadata_updated=True,both_merge_orders_equal=True)
    return combined,proof


def fluid_version(raw):
    raw=old.replace(raw,'VERSION = "'+fluid.VERSION+'"','VERSION = "'+VERSION+'"')
    return old.replace(raw,'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v'+fluid.VERSION+'"',
                       'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v'+VERSION+'"')


def payload(s=None):
    s=s or snapshots(); p=s['parent']; files,proof=merge(s)
    nb=json.loads(json.dumps(p['nb'])); buf=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(p['blob'])) as src,zipfile.ZipFile(buf,'w') as dst:
        for info in src.infolist(): dst.writestr(info,files[info.filename])
    blob=buf.getvalue(); code=''.join(nb['cells'][-1]['source']).encode()
    for a,b in [('BLOB = '+repr(p['assignments']['BLOB']),'BLOB = '+repr(base64.b64encode(blob).decode())),
        ('PINS = '+repr(p['assignments']['PINS']),'PINS = '+repr({k:old.digest(v) for k,v in files.items()})),
        (repr(old.digest(p['blob'])),repr(old.digest(blob)))]: code=old.replace(code,a,b)
    nb['cells'][-1]['source']=code.decode().splitlines(True)
    meta=dict(p['meta'],id=SLUG,title=SLUG.split('/')[1])
    for name,data in files.items():
        if name.endswith('.py'): compile(data,name,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    return nb,meta,files,blob,proof


def registration(s=None):
    s=s or snapshots()
    review=SCRATCH/'PRE-REVIEW.md'
    review_text=review.read_text(encoding='utf-8')
    old.need(review_text.rstrip().endswith(('PASS','PASS-WITH-NOTES'))
        and gate.sha((SCRATCH/'SPEC.md').read_bytes()) in review_text
        and 'distinct from builder' in review_text,'independent prereview required')
    return dict(item='H-B-172',version=VERSION,kernel_id=SLUG,purpose='2026-10-17 candidate',
        sources={k:dict(stage=str(STAGES[k].relative_to(ROOT)),receipt_sha256=v['receipt_sha256'],kernel_files=v['kernel_files']) for k,v in s.items()},
        spec_sha256=gate.sha((SCRATCH/'SPEC.md').read_bytes()),formula=fluid.FORMULA,
        prereview_sha256=gate.sha(review.read_bytes()),
        mirror='admitted R224 equal plain/anatomical-mirror probabilities; uint8 mirror before crop',
        submit_requires='Both single readings supplied; no score polling',keep_rule='combined >= max(fluid, mirror) - 0.001',
        runtime='H-B-167 actual hidden seconds + reviewed HB170_N32 projected hidden increment + 0.05 seconds',
        runtime_limit_seconds=32400,fluid_seconds_allowance=.05,qualified_for_submission=False,
        exposure='EXPOSED',clean_oof=False,gate='v1.4.4 pin follows completed dispatcher receipt; no gate copy')


def artifacts():
    s=snapshots(); nb,meta,files,blob,proof=payload(s)
    result={'kernel/release.ipynb':old.encoded(nb),'kernel/kernel-metadata.json':old.encoded(meta),
            'registration.json':gate.encoded(registration(s)),'merge-proof.json':gate.encoded(proof)}
    manifest=dict(version=VERSION,added=[],removed=[],changed=[],unchanged=[])
    for name,data in sorted(files.items()):
        before=s['parent']['files'][name]; changed=before!=data
        manifest['changed' if changed else 'unchanged'].append(dict(name=name,before_sha256=old.digest(before),after_sha256=old.digest(data)))
    old.need(len(manifest['changed'])==3 and len(manifest['unchanged'])==28,'combined scope')
    result['new-members.json']=gate.encoded(manifest)
    for key,version in [('parent','1.1.24-r224flipq'),('fluid','1.1.25-stackq'),('mirror','1.1.26-r224mirq')]:
        ref=s[key]; full='ALL 31 EMBEDDED MEMBERS; 0 ADDED/REMOVED\n'
        for name,data in sorted(files.items()):
            before=ref['files'][name]
            full+=f'{name}: {old.digest(before)} -> {old.digest(data)} '+('IDENTICAL' if before==data else 'REGISTERED_CHANGE')+'\n'
            full+=r224.diff(before,data,version+'/'+name,VERSION+'/'+name)
        for name,before,after in [('notebook',ref['nb'],nb),('kernel-metadata.json',ref['meta'],meta)]:
            full+=r224.diff(old.encoded(before),old.encoded(after),version+'/'+name,VERSION+'/'+name)
        result['full-diff-vs-v'+version+'.txt']=full.encode()
    return result,files,blob


def register():
    value=registration(); old.write(SCRATCH/'registration.json',gate.encoded(value))
    return dict(status='REGISTERED_BEFORE_EXECUTION')


def prepare():
    result,files,blob=artifacts()
    old.need((SCRATCH/'registration.json').read_bytes()==result['registration.json'],'register first')
    old.need(not (OUT/'kernel').exists(),'existing canonical stage: refuse overwrite')
    for key,path in STAGES.items():
        for name in ('kernel/release.ipynb','kernel/kernel-metadata.json'):
            old.write(BACKUP/key/name,(path/name).read_bytes())
    for name,data in result.items(): old.write(OUT/name,data)
    for name,data in files.items(): old.write(SCRATCH/'candidate-runtime'/name,data)
    rec=dict(status='PREPARED_CPU_ONLY',builder_sha256=gate.sha(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],archive_sha256=old.digest(blob),
        embedded={k:old.digest(v) for k,v in files.items()},qualified_for_submission=False)
    old.write(OUT/'build-receipt.json',gate.encoded(rec))
    old.write(OUT/'CHANGELOG.md',b'# 1.1.27-stackmirq / builder 1.0.0\nCompose admitted fluid and anatomical mirror bytes. Three changed embedded files; 28 identical. Explicit wrapper pin/version merge; no new serving logic. Sources untouched; backup .local/backups/hb172-build/hb172. CPU only; no gate copy.\n')
    return validate()


def validate():
    result,files,blob=artifacts()
    for name,data in result.items(): old.need((OUT/name).read_bytes()==data,'stage drift: '+name)
    rec=old.read(OUT/'build-receipt.json')
    old.need(rec['builder_sha256']==gate.sha(Path(__file__).read_bytes()),'builder drift')
    old.need(rec['archive_sha256']==old.digest(blob) and rec['embedded']=={k:old.digest(v) for k,v in files.items()},'embedded drift')
    old.need(rec['kernel_files']==gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],'kernel drift')
    return dict(status='PASS_CPU_VALIDATE',kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),identical=28,changed=3)


@contextmanager
def temporary(module,**changes):
    saved={k:getattr(module,k) for k in changes}
    try:
        for k,v in changes.items(): setattr(module,k,v)
        yield
    finally:
        for k,v in saved.items(): setattr(module,k,v)


def replay():
    cpu_only()
    from tools import c85r7_r224mir_cpu as mc
    # Reuse the exact real-data CPU procedure; one diagnostic environment value
    # changes from '-1' to EMPTY as requested. No serving source is transformed.
    tree=ast.parse(Path(mc.__file__).read_bytes())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assign=node.body[0]
    old.need(isinstance(assign,ast.Assign) and ast.unparse(assign.targets[0])=="os.environ['CUDA_VISIBLE_DEVICES']" and assign.value.value=='-1','CPU adapter anchor drift')
    assign.value=ast.Constant(value=''); ast.fix_missing_locations(node)
    namespace=dict(mc.__dict__)
    with temporary(mc,b=sys.modules[__name__]):
        namespace['b']=sys.modules[__name__]
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<hb172-cpu-replay-env-only>','exec'),namespace)
        mirror_result=namespace['run']()
    old.need(os.environ['CUDA_VISIBLE_DEVICES']=='','CUDA environment drift')
    with temporary(fluid,OUT=OUT,SCRATCH=SCRATCH,validate=validate,payload=lambda:payload()[:4]):
        fluid_result=fluid.replay()
    report=dict(status='PASS_CPU_REPLAY',kernel_sha256=validate()['kernel_sha256'],
        mirror=mirror_result,fluid=fluid_result,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        helper_sha256={str(Path(x).relative_to(ROOT)):gate.sha(Path(x).read_bytes()) for x in (__file__,mc.__file__,fluid.__file__)},
        limitations='Real mirror CPU forwards; fluid replay uses explicit R224 surrogates. No full joint neural run, CUDA/runtime or gain evidence.',exposure='EXPOSED',clean_oof=False)
    old.write(SCRATCH/'cpu-replay.json',gate.encoded(report))
    return dict(status=report['status'],kernel_sha256=report['kernel_sha256'],cuda_visible_devices='')


def decision(fluid_score,mirror_score,combined_score,hidden_seconds,mirror_increment):
    vals=(fluid_score,mirror_score,combined_score,hidden_seconds,mirror_increment)
    old.need(all(type(v) in (int,float) and math.isfinite(v) for v in vals),'both readings and measured runtime required')
    old.need(all(0<=v<=1 for v in vals[:3]) and hidden_seconds>0 and mirror_increment>0,'decision domain')
    return combined_score>=max(fluid_score,mirror_score)-.001 and hidden_seconds+mirror_increment+.05<=32400


def admit():
    valid=validate(); replayed=old.read(SCRATCH/'cpu-replay.json')
    old.need(replayed['status']=='PASS_CPU_REPLAY' and replayed['kernel_sha256']==valid['kernel_sha256'],'fresh replay required')
    for name,digest in replayed['helper_sha256'].items(): old.need(gate.sha((ROOT/name).read_bytes())==digest,'replay helper drift')
    return dict(status='HOLD_EXACT_PIN_SINGLE_READINGS_AND_RUNTIME',kernel_sha256=valid['kernel_sha256'],qualified_for_submission=False)


def main():
    cpu_only(); ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','prepare','validate','cpu-replay','admit']); ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    if a.command in ('prepare','admit'): old.need(a.dry_run,'CPU dry-run required')
    print(json.dumps({'register':register,'prepare':prepare,'validate':validate,'cpu-replay':replay,'admit':admit}[a.command](),indent=2))


if __name__=='__main__': main()
