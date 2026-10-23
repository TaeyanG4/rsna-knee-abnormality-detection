"""H-B-166 engineering release builder 1.0.0. CPU only; no publication.

Changelog: one Q-slot replacement, sealed original definitions, missing-fold
fail-closed package. No frozen source files modified.
Draft metadata amendment: independently reviewed DESIGN v1.0.3 registers the
replacement member scheduling; unchanged kernel and numerical recipe.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import c85r7_stackmirq_v1127 as previous
from tools import hb166_serving_v1 as serving
old,gate=previous.old,previous.gate
WORK=ROOT/'.local/scratch/hb166-serving/hb166'
OUT=ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.28-multiviewq'
VERSION='1.1.28-multiviewq'
SLUG='taeyangg4/knee-lb-multiviewq-v1'
CONFIG=ROOT/'.local/scratch/slot-b/hb166/pilot-v100/configs/multiview-s42.json'
CONFIG_SHA='9cf2808a13a00d72eb11220a5e26efbe8fdaaf9451e30c859050b1ed64eb2681'
FIT=ROOT/'.local/scratch/slot-b/hb166/pilot-v100/fits/multiview-s42'
WEIGHT_SHA='b2c4ac840b8684a1efa67205b217f6e4dc4cc1c600bf1fa78a0169d3f1747f5e'
DESIGN_SHA='873f1f85514994063a8d66fc86ee8356a7e0d6dd14cf5abf650faa528ffd5276'
PRE_REVIEW='PRE-REVIEW-v103.md'
PRE_REVIEW_SHA='1ef2eea25b250be001fae9b244f34a27b1ae2243c48adc248c0c9fe38301839d'
SELECTION={
 'common':('training/src/raptor_common.py','LAB UID SLOTS64 SLOTS44 CONTRACTS eval_centers train_centers study_rng select_centers build_backbone'),
 'tokens':('training/src/raptor_exp185.py','valid_windows spatial_tokens finding_attention'),
 'mil':('training/arms/mil.py','FindingMIL strict_load MILWindows flip_inputs autocast'),
 'native':('training/arms/native_resolution_v2.py','CONTRACT scale_windows NativeWindows'),
 'geometry':('tools/hb155_crop_v1.py','CONTRACT compartment_name project outside_slab slab_prior series_center stack_points crop read_series build_study'),
 'crops':('training/arms/hb155_crops_v1.py','CONTROL_CONTRACT SIDE_CONTRACT CONTRACTS side_flag Windows'),
 'mv':('training/arms/hb166_multiview_v1.py','FULL VIEW_NAMES CROP_STREAMS FULL_CONTRACT CROP_CONTRACT CropWindows MultiViewWindows make_dataset collate MultiViewFindingMIL attach_views MultiViewArm ARM flip_views evaluate'),
 'raw':('training/src/raptor_corpus.py','_WORKER pick_series _order_series _read_pixels _crop_resize build_study'),
}

def cpu():
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')

def encode(value):
    # Preserve dependency module insertion order, unlike canonical gate JSON.
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()

def config():
    old.need(serving.sha(CONFIG)==CONFIG_SHA,'fit config drift')
    return serving.read(CONFIG)

def source_payload():
    cfg=config(); payload=dict(version='1.0.0',config_sha256=CONFIG_SHA,modules={})
    for key,(path,names) in SELECTION.items():
        raw=(ROOT/path).read_bytes(); digest=old.digest(raw)
        old.need(cfg['inputs'][path]==digest,'fit-bound source changed: '+path)
        lines=raw.splitlines(keepends=True); selected=[]; remaining=set(names.split())
        for n in ast.parse(raw).body:
            name=getattr(n,'name',None)
            if isinstance(n,ast.Assign):
                target=n.targets[0]
                name=target.id if isinstance(target,ast.Name) else target.elts[0].id if isinstance(target,ast.Tuple) else None
            if name not in remaining: continue
            start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])
            span=b''.join(lines[start-1:n.end_lineno])
            selected.append(dict(name=name,start=start,end=n.end_lineno,sha256=old.digest(span),bytes=base64.b64encode(span).decode()))
            remaining.remove(name)
        old.need(not remaining,'missing definition: '+str(remaining))
        payload['modules'][key]=dict(path=path,sha256=digest,spans=selected)
    return encode(payload)

def manifest():
    import torch
    old.need(serving.read(FIT/'heartbeat.json')['status']=='complete','running fit forbidden')
    old.need(serving.sha(FIT/'final.pt')==WEIGHT_SHA,'checkpoint drift')
    state=torch.load(FIT/'final.pt',weights_only=True,map_location='cpu')
    cfg=config()
    # Read identity only; no labels or saved probability rows are printed.
    return dict(version='1.0.0',slot='Q_A5q',model='HB166_MULTIVIEW',exposure='EXPOSED',clean_oof=False,
        targets=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture'],
        aggregation='equal probability mean before slot rank',roster=[
            dict(fold=2,seed=42,status='PRESENT',file='fold2-s42.pt',sha256=WEIGHT_SHA,bytes=(FIT/'final.pt').stat().st_size,identity=state['identity']),
            dict(fold=3,seed=44,status='MISSING',file='fold3-s44.pt',sha256=None,bytes=None,identity=None)],
        config_sha256=CONFIG_SHA,source_sha256=old.digest(source_payload()),qualified_for_submission=False)

def registered():
    old.need(serving.sha(WORK/'DESIGN.md')==DESIGN_SHA,'design drift')
    old.need(serving.sha(WORK/PRE_REVIEW)==PRE_REVIEW_SHA,'independent review drift')
    review=(WORK/PRE_REVIEW).read_text(encoding='utf-8')
    old.need(DESIGN_SHA in review and 'Verdict: PASS for CPU-only harness preparation' in review,'independent review missing')
    p=previous.snapshots()['parent']
    return dict(item='H-B-166',version=VERSION,kernel_id=SLUG,design_sha256=DESIGN_SHA,
        prereview_sha256=serving.sha(WORK/PRE_REVIEW),parent_receipt_sha256=p['receipt_sha256'],
        source_sha256=old.digest(source_payload()),config_sha256=CONFIG_SHA,checkpoint_sha256=WEIGHT_SHA,
        fit_receipt_sha256='30577165945f70df5e0b70b953ae8a48e66fd7b85ecf1c81f4ed84be354383f7',
        fit_csv_sha256='bed548653c598817112cea9cfb797c54e81cd7bf6de908b92ab601b5b26d09ca',
        cache_manifest_sha256=['23c02b668fe9cb90c7e3c07fff4422c922066dfbc09bc1cc67690dd846a7a1dd',
                              '75fcf0d2e8e7af85b6203642a2841ad3421e2171ea7a8593743777b913f258ec'],
        cohort=config()['val'][:2],cpu_atol=1e-6,saved_fp16_atol=.005,rtol=0,raw_uint8='exact',
        slot='Q_A5q',weights=[17,17,12],denominator=46,final=[.85,.15],r224=[.65,.35],
        qualified_for_submission=False,exposure='EXPOSED',clean_oof=False,
        runtime='HOLD: inclusive 4070 ABBA local delta only; require reviewed end-to-end T4 evidence and actual HB167 hidden receipt',
        holds=['missing fold3','fold3/five-fold scientific decision','independent result review','N32 real post-check replay','CUDA/Linux/T4 runtime','new exact gate pin'])

def register():
    r=registered(); old.write(WORK/'registration.json',encode(r))
    return dict(status='REGISTERED_BEFORE_EXECUTION',design_sha256=DESIGN_SHA,studies=2)

def patch_c85(raw,adapter_sha):
    start=raw.index(b'    q = np.full((len(ids), 12), .5)\n',raw.index(b'def authenticated_a172'))
    end=raw.index(b"    failures['Q_A5q'] = qfail",start)
    raw=raw[:start]+("    p.check_pin(Path(__file__).with_name('hb166_serving_v1.py'), '"+adapter_sha+"')\n"
        "    import hb166_serving_v1 as hb166\n    q, qfail = hb166.authenticated(__import__(__name__), directory, ids)\n").encode()+raw[end:]
    raw=old.replace(raw,"executed_arms=['EXP155','EXP143v2','Q_A5q']","executed_arms=['EXP155','EXP143v2','HB166_MULTIVIEW'], slot_alias={'Q_A5q':'HB166_MULTIVIEW'}")
    start=raw.index(b'    import inspect\n',raw.index(b'def run_q(')); end=raw.index(b'\n\ndef a172_source',start)
    raw=raw[:start]+("    module.reset_backend()\n"
        "    p.check_pin(Path(__file__).with_name('hb166_serving_v1.py'), '"+adapter_sha+"')\n"
        "    import hb166_serving_v1 as hb166\n    return hb166.run(__import__(__name__), competition, output)\n").encode()+raw[end:]
    return raw

def payload():
    p=previous.snapshots()['parent']; files=dict(p['files']); src=source_payload(); m=encode(manifest())
    adapter=(ROOT/'tools/hb166_serving_v1.py').read_bytes().replace(b'BUILD_SOURCE_SHA',old.digest(src).encode()).replace(b'BUILD_MANIFEST_SHA',old.digest(m).encode())
    files['hb166_serving_v1.py']=adapter; files['hb166-sources.json']=src
    files['exp169_release_c85.py']=patch_c85(files['exp169_release_c85.py'],old.digest(adapter))
    files['exp169_release_c85r7.py']=old.replace(files['exp169_release_c85r7.py'],old.digest(p['files']['exp169_release_c85.py']),old.digest(files['exp169_release_c85.py']))
    key='exp169_release_c85r7codec.py'; wrapper=files[key]
    for a,b in [(old.digest(p['files']['exp169_release_c85r7.py']),old.digest(files['exp169_release_c85r7.py'])),('1.1.24-r224flipq',VERSION),('taeyangg4/knee-lb-r224flipq-v1',SLUG),('DATASETS = [*base.DATASETS, codecs.DATASET_SLUG]',f'DATASETS = [*base.DATASETS, codecs.DATASET_SLUG, {serving.DATASET!r}]')]:
        # Version occurs in VERSION and OUT only; both dependent metadata.
        old.need(a.encode() in wrapper,'wrapper anchor missing'); wrapper=wrapper.replace(a.encode(),b.encode())
    files[key]=wrapper
    changed={k for k in p['files'] if files[k]!=p['files'][k]}
    old.need(changed=={'exp169_release_c85.py','exp169_release_c85r7.py',key},'scope drift')
    nb=json.loads(json.dumps(p['nb'])); buf=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(p['blob'])) as zin,zipfile.ZipFile(buf,'w') as zout:
        for info in zin.infolist(): zout.writestr(info,files[info.filename])
        for name in ('hb166_serving_v1.py','hb166-sources.json'):
            info=zipfile.ZipInfo(name,(2026,10,16,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            zout.writestr(info,files[name])
    blob=buf.getvalue(); code=''.join(nb['cells'][-1]['source']).encode()
    for a,b in [('BLOB = '+repr(p['assignments']['BLOB']),'BLOB = '+repr(base64.b64encode(blob).decode())),('PINS = '+repr(p['assignments']['PINS']),'PINS = '+repr({k:old.digest(v) for k,v in files.items()})),(repr(old.digest(p['blob'])),repr(old.digest(blob)))]: code=old.replace(code,a,b)
    nb['cells'][-1]['source']=code.decode().splitlines(True)
    meta=dict(p['meta'],id=SLUG,title=SLUG.split('/')[1],dataset_sources=[*p['meta']['dataset_sources'],serving.DATASET])
    for name,data in files.items():
        if name.endswith('.py'): compile(data,name,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    return p,files,nb,meta,blob,m

def artifacts():
    p,files,nb,meta,blob,m=payload(); r=registered()
    result={'kernel/release.ipynb':old.encoded(nb),'kernel/kernel-metadata.json':old.encoded(meta),
        'registration.json':encode(r),'assets/manifest.json':m,'DESIGN.md':(WORK/'DESIGN.md').read_bytes()}
    inventory=[]; diff='Complete member, notebook and metadata diff. No notebook execution.\n'
    for name,data in sorted(files.items()):
        before=p['files'].get(name,b''); status='added' if name not in p['files'] else 'identical' if before==data else 'changed'
        inventory.append(dict(name=name,status=status,before_sha256=old.digest(before) if before else None,after_sha256=old.digest(data)))
        diff+=f'{name}: {status}; {old.digest(before)} -> {old.digest(data)}\n'
        diff+=previous.r224.diff(before,data,'parent/'+name,VERSION+'/'+name)
    for name,before,after in [('notebook',p['nb'],nb),('metadata',p['meta'],meta)]: diff+=previous.r224.diff(old.encoded(before),old.encoded(after),'parent/'+name,VERSION+'/'+name)
    result['embedded-inventory.json']=encode(inventory); result['full-diff.txt']=diff.encode()
    # Direct existing no-flip parent module compatibility, no second stage.
    noflip=ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.23-r224q/kernel/release.ipynb'
    nfiles,_,_=old.unpack(previous.flipq.restored(old.read(noflip)))
    old.need(nfiles['exp169_release_c85.py']==p['files']['exp169_release_c85.py'],'no-flip O module differs')
    result['no-flip-compatibility.json']=encode(dict(parent=str(noflip.relative_to(ROOT)),notebook_sha256=serving.sha(noflip),
        o_module_byte_identical=True,patched_o_module_sha256=old.digest(patch_c85(nfiles['exp169_release_c85.py'],old.digest(files['hb166_serving_v1.py']))),
        second_stage_built=False,flip_all_unchanged=True))
    return result,files,blob

def prepare():
    result,files,blob=artifacts()
    old.need((WORK/'registration.json').read_bytes()==result['registration.json'],'register first')
    if (OUT/'kernel').exists():
        return validate()  # Existing draft is verification-only; never overwrite drift.
    for name,data in result.items(): old.write(OUT/name,data)
    # Explicit actual fold2 placeholder; absent fold3 is manifest-only, not a fake file.
    shutil.copyfile(FIT/'final.pt',OUT/'assets/fold2-s42.pt')
    for name,data in files.items(): old.write(WORK/'candidate-runtime'/name,data)
    old.write(OUT/'CHANGELOG.md',b'# 1.1.28-multiviewq / builder 1.0.0\nReplace Q slot by HB166 with sealed fitted definitions. Engineering only: fold3 MISSING; no admission, gate binding or upload. Originals untouched; no backup required.\n')
    return validate()

def validate():
    result,files,blob=artifacts()
    for name,data in result.items(): old.need((OUT/name).read_bytes()==data,'stage drift: '+name)
    old.need(serving.sha(OUT/'assets/fold2-s42.pt')==WEIGHT_SHA,'packaged weight drift')
    return dict(status='PASS_CPU_VALIDATE',kernel_sha256=old.digest(result['kernel/release.ipynb']),changed=3,identical=28,added=2,qualified_for_submission=False)

def admit():
    valid=validate()
    return dict(**valid,admission='HOLD',holds=registered()['holds'],gate_bound=False)

def main():
    cpu(); p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['register','prepare','validate','admit']); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
    if a.command in ('prepare','admit'): old.need(a.dry_run,'dry-run required')
    print(json.dumps(globals()[a.command](),indent=2))

if __name__=='__main__': main()
