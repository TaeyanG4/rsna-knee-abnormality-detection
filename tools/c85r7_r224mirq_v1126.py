"""H-B-170 strict one-candidate builder 1.0.0, CPU only.

Changelog: anatomical mirror before crop; two changed embedded members only.
Original files untouched; parent backup under .local/backups/hb170-build/hb170.
"""
from __future__ import annotations
import argparse
import ast
import base64
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_flipall_v1115 as old
from tools import c85r7_flipallq_v1122 as flipq
from tools import c85r7_r224q_v1123 as r224
from tools import strict_variant_gate as gate
from tools import strict_variant_gate_r224_v143 as previous

VERSION = '1.1.26-r224mirq'
SLUG = 'taeyangg4/knee-lb-r224mirq-v1'
BASE = ROOT / '.local/scratch/astra-b5/hb167-gate/gate-copy-v143'
OUT = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7' / ('v'+VERSION)
SCRATCH = ROOT / '.local/scratch/hb170-build/hb170'
BACKUP = ROOT / '.local/backups/hb170-build/hb170'
SOURCE = 'raptor224_run.py'
WRAPPER = 'exp169_release_c85r7codec.py'
PARENT_RECEIPT = ROOT / '.local/state/gates/knee-lb-r224flipq-v1.json'


def parent():
    inputs = gate.Inputs(ROOT)
    k = gate.kernel(BASE, inputs)
    old.need(k[2]['kernel_files'] == previous.CHILD_PINS, 'parent kernel drift')
    rec = inputs.json(PARENT_RECEIPT)
    old.need(rec['status']=='PASS' and rec['errors']==[] and rec['fixture_only'] is False
        and rec['gate_version']=='1.4.3' and rec['kernel_id']==previous.SLUG
        and Path(rec['stage'])==BASE, 'admitted parent identity mismatch')
    for name,h in rec['input_sha256'].items():
        old.need(gate.sha(inputs.read(gate.local_path(name,ROOT)))==h, 'parent admission input drift: '+name)
    nb=old.read(BASE/'kernel/release.ipynb')
    files,blob,assignments=old.unpack(flipq.restored(nb))
    return nb,files,blob,assignments


def patch_files(original):
    files=dict(original)
    public=original['public-serving.py']
    start=public.index(b'SLOT_SIZES = ')
    end=public.index(b'def eval_windows(',start)
    port=public[start:end]
    old.need(port.count(b'(94,3,320,320)')==1, 'pair shape anchor changed')
    port=port.replace(b'(94,3,320,320)',b'(94,3,224,224)')
    code=old.replace(original[SOURCE], 'def eval_windows(',port.decode()+'def eval_windows(')
    code=old.replace(code,'def gpu_windows(vol, mask, k, res, dev, norm=NORM):',
        'def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):')
    anchor='    key = (str(dev), norm)'
    code=old.replace(code,anchor,'    if mirror:\n        tri = _mirror_tri(tri, cs, D)\n'+anchor)
    code=old.replace(code,'                        arm_probs[a][i] = infer_probs(model, xw, d)',
        '                        xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)\n'
        '                        arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)')
    files[SOURCE]=code
    wrapper=old.replace(original[WRAPPER], old.digest(original[SOURCE]),old.digest(code))
    wrapper=old.replace(wrapper,'VERSION = "1.1.24-r224flipq"','VERSION = "'+VERSION+'"')
    wrapper=old.replace(wrapper,'KERNEL_ID = "'+previous.SLUG+'"','KERNEL_ID = "'+SLUG+'"')
    files[WRAPPER]=wrapper
    return files


def payload():
    nb,original,original_blob,assignments=parent()
    files=patch_files(original)
    buf=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original_blob)) as src, zipfile.ZipFile(buf,'w') as dst:
        for info in src.infolist(): dst.writestr(info, files[info.filename])
    blob=buf.getvalue()
    code=''.join(nb['cells'][-1]['source']).encode()
    for before,after in [('BLOB = '+repr(assignments['BLOB']), 'BLOB = '+repr(base64.b64encode(blob).decode())),
        ('PINS = '+repr(assignments['PINS']), 'PINS = '+repr({k:old.digest(v) for k,v in files.items()})),
        (repr(old.digest(original_blob)),repr(old.digest(blob)))]:
        code=old.replace(code,before,after)
    nb['cells'][-1]['source']=code.decode().splitlines(True)
    meta=old.read(BASE/'kernel/kernel-metadata.json')
    meta.update(id=SLUG,title=SLUG.split('/')[1])
    return nb,meta,files,blob


def scope(nb,meta):
    expected,emeta,files,blob=payload()
    old.need(nb==expected and meta==emeta,'forbidden notebook/metadata difference')
    actual,ab,_=old.unpack(flipq.restored(nb))
    old.need(actual==files and ab==blob,'forbidden embedded difference')
    for name,data in files.items():
        if name.endswith('.py'): compile(data,name,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    return files,blob


def registration():
    return dict(version=VERSION,item='H-B-170',kernel_id=SLUG,parent=str(BASE.relative_to(ROOT)),
        parent_kernel_files=previous.CHILD_PINS,parent_submission=57224580,
        parent_receipt_sha256=old.digest(PARENT_RECEIPT.read_bytes()),
        spec_sha256=old.digest((SCRATCH/'SPEC.md').read_bytes()),formula=r224.FORMULA,
        mirror_order='uint8 mirror -> 307 crop -> 224 bilinear resize -> ImageNet normalization',
        purpose='2026-10-16/17 candidate; hypothesis, not established gain',
        public_keep_delta=-.001, runtime_limit_hours=9,
        hidden_total='HOLD until H-B-167 real hidden receipt plus measured mirror increment',
        gold58='optional EXPOSED no-harm lens deferred',exposure='EXPOSED',clean_oof=False,
        qualified_for_submission=False, gate_binding='v1.4.4 exact pin proposal; no gate copy')


def artifacts():
    nb,meta,files,blob=payload()
    original_nb,original,_,_=parent()
    changed=sorted(k for k in files if original[k]!=files[k])
    old.need(changed==sorted([SOURCE,WRAPPER]) and len(files)==31,'member scope drift')
    manifest=dict(version=VERSION,added=[],removed=[],
        changed=[dict(name=k,before_sha256=old.digest(original[k]),after_sha256=old.digest(files[k])) for k in changed],
        unchanged=[dict(name=k,sha256=old.digest(v)) for k,v in sorted(files.items()) if k not in changed])
    full='ALL 31 EMBEDDED FILES: 29 IDENTICAL, 2 REGISTERED CHANGES, 0 ADDED/REMOVED\n'
    for k in sorted(files):
        full+=f'{k}: {old.digest(original[k])} -> {old.digest(files[k])} '+('IDENTICAL' if k not in changed else 'REGISTERED_CHANGE')+'\n'
        full+=r224.diff(original[k],files[k],'parent/'+k,'stage/'+k)
    for name,before,after in [('notebook',original_nb,nb),('kernel-metadata.json',old.read(BASE/'kernel/kernel-metadata.json'),meta)]:
        full+=r224.diff(old.encoded(before),old.encoded(after),'parent/'+name,'stage/'+name)
    return nb,meta,files,blob,manifest,full.encode()


def prepare():
    nb,meta,files,blob,manifest,full=artifacts()
    for name in ('kernel/release.ipynb','kernel/kernel-metadata.json','registration.json'):
        old.write(BACKUP/name,(BASE/name).read_bytes())
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
        ('registration.json',gate.encoded(registration())),('new-members.json',gate.encoded(manifest)),
        ('full-diff-vs-v1.1.24-r224flipq.txt',full)]: old.write(OUT/name,data)
    rec=dict(status='PREPARED_CPU_ONLY',version=VERSION,builder_sha256=old.digest(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],archive_sha256=old.digest(blob),
        embedded={k:old.digest(v) for k,v in files.items()},unchanged=29,changed=[SOURCE,WRAPPER],
        qualified_for_submission=False,exposure='EXPOSED',clean_oof=False)
    old.write(OUT/'build-receipt.json',gate.encoded(rec))
    old.write(OUT/'CHANGELOG.md',b'# 1.1.26-r224mirq / builder 1.0.0\nAnatomical uint8 mirror before crop, equal probability pair. Two changed embedded files; 29 byte-identical. Only dependent pins and VERSION/KERNEL_ID metadata. Original files untouched; parent backup .local/backups/hb170-build/hb170. No Kaggle/GPU launch; no gate copy. EXPOSED hypothesis.\n')
    for n,v in files.items(): old.write(SCRATCH/'candidate-runtime'/n,v)
    return validate()


def validate():
    nb,meta,files,blob,manifest,full=artifacts()
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
        ('registration.json',gate.encoded(registration())),('new-members.json',gate.encoded(manifest)),
        ('full-diff-vs-v1.1.24-r224flipq.txt',full)]:
        old.need((OUT/name).read_bytes()==data,'stage drift: '+name)
    scope(nb,meta)
    rec=old.read(OUT/'build-receipt.json')
    old.need(rec['builder_sha256']==old.digest(Path(__file__).read_bytes()),'builder drift')
    old.need(rec['embedded']=={k:old.digest(v) for k,v in files.items()} and rec['archive_sha256']==old.digest(blob),'receipt drift')
    old.need(rec['kernel_files']==gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],'kernel drift')
    return dict(status='PASS_CPU_VALIDATE',unchanged=29,changed=2,
        kernel_sha256=gate.sha(gate.encoded(rec['kernel_files'])),qualified_for_submission=False)


def decision(parent_score,candidate_score,hidden_seconds,increment):
    import math
    values=(parent_score,candidate_score,hidden_seconds,increment)
    old.need(all(type(x) in (int,float) and math.isfinite(x) for x in values),'finite decision inputs required')
    old.need(0<=parent_score<=1 and 0<=candidate_score<=1 and hidden_seconds>0 and increment>0,'decision domain')
    return candidate_score>=parent_score-.001 and hidden_seconds+increment<=9*3600


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','prepare','validate','cpu-smoke','admit'])
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1')
    if a.command in ('prepare','admit'): old.need(a.dry_run,'CPU dry-run required')
    if a.command=='register':
        old.write(SCRATCH/'registration.json',gate.encoded(registration())); result={'status':'REGISTERED'}
    elif a.command=='prepare': result=prepare()
    elif a.command=='validate': result=validate()
    elif a.command=='cpu-smoke':
        from tools import c85r7_r224mir_cpu as smoke
        result=smoke.run()
    else:
        result=validate(); result['status']='HOLD_INDEPENDENT_GATE_REVIEW_AND_4070_ABBA'
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
