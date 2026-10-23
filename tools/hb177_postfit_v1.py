"""H-B-177 post-fit v1.0.0: offline, checkpoint-only HB156 standalone derivative.

Changelog: final seed42 EMA in the sealed HB156 ABI; exhaustive embedded identity.
No frozen/running stage is modified. CUDA is hidden before any model import.
"""
from __future__ import annotations
import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hb151_killtest_v102 import need, sha, digest, read, encoded, write, retry, archive_files

VERSION = '1.0.0'
OUT = ROOT/'artifacts/experiments/HB177_KILLTEST/v1.0.0'
SCRATCH = ROOT/'.local/scratch/hb177-postfit/H-B-177'
PARENT = ROOT/'artifacts/experiments/HB156_KILLTEST/v1.0.0'
EXPORT = ROOT/'.local/scratch/hb177/H-B-177/v1.0.3'
CHECKPOINT = EXPORT/'local/fit/hb177-final-ema.pt'
CHECKPOINT_SHA = '5f4fda6c29cebf37d6b5e1cc270bee50e0904edd23752fc21a5547f03c500a7c'
PROTOCOL_SHA = '13238133a4fa10b0f91275fb99573bce3a59711747758b9a9e468e94d08239cb'
NAME = 'exp143-v2-e14-e11-e12-bn.pt'
ARCHIVE = 'exp152-assets.zip.bin'
RUNTIME = 'exp152_own0949_release.py'
ARCH = 'convnext_small.fb_in22k_ft_in1k_384'
SLUG = 'taeyangg4/knee-hb177-cnxs-recipe-v1'
DATASET = 'taeyangg4/knee-hb177-cnxs-s42-v1'
PINS = {
    'kernel/kernel-metadata.json': 'f73ddbbe333bdd823ca1884bf025585fa227be0fcb297adb82473ef5d5709238',
    'kernel/release.ipynb': 'e795ec936bcc847b99ae06c37a22645069dcc9be49f9d2b60e0179833eaa4081',
    'build-receipt.json': 'faca8f213fc9bf22aea25ef3a64387550e9c7efea56c6e3b8c82aeb3760a12a9',
    'dataset/dataset-metadata.json': '864db4d0066555650e346f1f4cd4c2c6dd8572ea382524c12594222ea5c460db',
    'dataset/'+ARCHIVE: '20a1d26b5c8e1916a94022cd1b61b0aaa052168c72f41d5ee44e0fc9b578f5b4',
}
DECISION = dict(alive_at_least=.950, dead_at_most=.946, otherwise='INCONCLUSIVE',
                standalone=True, public_is_exposed=True, no_ratio_tuning=True)


def cpu_only():
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    import torch
    need(not torch.cuda.is_initialized(), 'CPU-only process required')
    torch.set_num_threads(2)


def canonical(value):
    from tools.strict_variant_gate import encoded as wire
    return wire(value)


def authenticate():
    for rel, pin in PINS.items():
        need(sha(PARENT/rel) == pin, 'HB156 parent pin: '+rel)
    hashes, small = archive_files(PARENT/'dataset'/ARCHIVE)
    need(hashes == read(PARENT/'build-receipt.json')['files'], 'parent archive inventory')
    gate = read(ROOT/'.local/state/gates/knee-hb156-cnxs-l2-v1.json')
    need(gate['status']=='PASS' and gate['kernel_id']=='taeyangg4/knee-hb156-cnxs-l2-v1'
         and gate['stage_inventory']['kernel_files']=={Path(n).name:v for n,v in PINS.items() if n.startswith('kernel/')},
         'admitted HB156 parent identity')
    contract = read(EXPORT/'kernel-contract.json')
    need(contract['archive_sha256']==PINS['dataset/'+ARCHIVE], 'sealed contract parent drift')
    for name, record in contract['source'].items():
        need(sha(EXPORT/name)==record['sha256'] and hashes[record['member']]==record['member_sha256'], 'sealed source drift')
        source=small[record['member']].decode()
        snippets=['\n'.join(source.splitlines()[a-1:z]) for a,z in record['segments']]
        need((EXPORT/name).read_text(encoding='utf-8')=='\n\n'.join(snippets)+'\n', 'sealed source segment mismatch')
    return hashes, small


def identity():
    need(sha(CHECKPOINT)==CHECKPOINT_SHA, 'final EMA pin')
    complete=read(CHECKPOINT.parent/'complete.json')
    # A completion is provenance, never an independent result approval.
    from tools.hb177_export_v103 import load_with_kernel
    import torch
    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    p=ck['training_provenance']
    need(ck['fixture'] is False and ck['artifact_role']=='trained-final' and
         p['seed']==42 and p['host']=='local' and p['protocol_sha256']==PROTOCOL_SHA and
         p['checkpoint_kind']=='final-epoch-EMA' and p['swa_epochs_1based']==[] and
         p['ema_updates']==5440 and ck['clean_oof'] is False and ck['exposure']=='EXPOSED', 'final training provenance')
    model=load_with_kernel(CHECKPOINT)
    del model, ck
    return dict(sha256=CHECKPOINT_SHA, training_provenance=p, exposure='EXPOSED',clean_oof=False,
        placeholder=False,serving_name=NAME,complete_sha256=sha(CHECKPOINT.parent/'complete.json'),
        completion_status=complete.get('status',complete.get('state')),
        legacy_envelope='HB156 loader ABI only; not training history; no SWA or BN replay')


def manifest(original,binding):
    m=json.loads(original['release-manifest.json'])
    m['checkpoints'][NAME]['sha256']=CHECKPOINT_SHA
    m['asset_sha256'][NAME]=CHECKPOINT_SHA
    m['hb151_export']['sha256']=CHECKPOINT_SHA
    m['dataset']=DATASET
    m['hb177_provenance']=binding
    return m


def payload(original,binding,archive_sha):
    nb=read(PARENT/'kernel/release.ipynb')
    meta=read(PARENT/'kernel/kernel-metadata.json')
    src=''.join(nb['cells'][5]['source'])
    changes={
        'knee-hb156-cnxs-l2-weights-v1':(DATASET.split('/')[1],2),
        PINS['dataset/'+ARCHIVE]:(archive_sha,1),
        digest(original['release-manifest.json']):(digest(encoded(manifest(original,binding))),1),
    }
    for a,(b,n) in changes.items():
        need(src.count(a)==n, 'cell5 replacement count: '+a)
        src=src.replace(a,b)
    nb['cells'][5]['source']=src
    meta.update(id=SLUG,title=SLUG.split('/')[1],dataset_sources=[DATASET])
    return nb,meta


def dataset_metadata():
    value=read(PARENT/'dataset/dataset-metadata.json')
    value.update(id=DATASET,title=DATASET.split('/')[1],description=
        'Private H-B-177 ConvNeXt-S seed42 final epoch EMA. EXPOSED / clean_oof=false. Legacy SWA metadata is loader ABI only. No training data.')
    return value


def diff(nb,meta,small,hashes):
    from tools.hb156_killtest_v1 import embedded
    ph,ps=authenticate()
    pn=read(PARENT/'kernel/release.ipynb')
    pairs=[('notebook-top',encoded({k:v for k,v in pn.items() if k!='cells'}),encoded({k:v for k,v in nb.items() if k!='cells'}))]
    need(len(pn['cells'])==len(nb['cells']), 'cell inventory drift')
    pairs += [(f'cell-{i}',encoded(a),encoded(b)) for i,(a,b) in enumerate(zip(pn['cells'],nb['cells']))]
    pairs += [('kernel-metadata',encoded(read(PARENT/'kernel/kernel-metadata.json')),encoded(meta))]
    for n in ps:
        pairs.append(('archive/'+n,ps[n],small[n]))
        if n.endswith('.py'):
            need(ps[n]==small[n], 'serving code must be byte-identical')
            a,b=embedded(ps[n]),embedded(small[n]); need(a==b,'nested source drift')
            pairs += [('decoded/'+n+'/'+k,v,b[k]) for k,v in a.items()]
    pairs += [('checkpoint-sha256',ph[NAME].encode(),hashes[NAME].encode())]
    text=['Only checkpoint/dataset identity, truthful provenance and chained hashes may differ.\n']
    for n,a,b in pairs:
        text.append(f'{n}: {"IDENTICAL" if a==b else "REGISTERED_CHANGE"} {digest(a)} -> {digest(b)}\n')
        if a!=b: text.extend(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='HB156/'+n,tofile='HB177/'+n))
    return ''.join(text).encode()


def build():
    cpu_only()
    _,original=authenticate()
    binding=identity()
    if (OUT/'build-receipt.json').exists():
        return validate()
    OUT.mkdir(parents=True,exist_ok=True)
    small={**original,'release-manifest.json':encoded(manifest(original,binding))}
    dest=OUT/'dataset'/ARCHIVE; dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as z:
        for name in sorted([NAME,*small]):
            with z.open(zipfile.ZipInfo(name,date_time=(2026,10,16,0,0,0)),'w',force_zip64=True) as target:
                if name==NAME:
                    with CHECKPOINT.open('rb') as stream: shutil.copyfileobj(stream,target,4<<20)
                else: target.write(small[name])
    retry(lambda:os.replace(temp,dest))
    nb,meta=payload(original,binding,sha(dest)); hashes,_=archive_files(dest)
    outputs={'kernel/release.ipynb':encoded(nb),'kernel/kernel-metadata.json':encoded(meta),
        'dataset/dataset-metadata.json':encoded(dataset_metadata()),'export-binding.json':encoded(binding),
        'full-diff-vs-parent.txt':diff(nb,meta,small,hashes)}
    for name,data in outputs.items(): write(OUT/name,data)
    files={n:dict(sha256=sha(OUT/'dataset'/n),bytes=(OUT/'dataset'/n).stat().st_size) for n in (ARCHIVE,'dataset-metadata.json')}
    write(OUT/'dataset/files.json',encoded(dict(version=VERSION,id=DATASET,private=True,files=files,
        archive_members=hashes,checkpoint_sha256=CHECKPOINT_SHA)))
    artifacts={n:sha(OUT/n) for n in [*outputs,'dataset/'+ARCHIVE,'dataset/files.json']}
    rec=dict(version=VERSION,builder_sha256=sha(__file__),parent_pins=PINS,artifacts=artifacts,
        files=hashes,placeholder=False,exposure='EXPOSED',clean_oof=False,decision=DECISION,
        qualified_for_submission=False,status='BOUND_NOT_REVIEWED')
    write(OUT/'build-receipt.json',encoded(rec))
    return validate()


def validate():
    _,original=authenticate(); rec=read(OUT/'build-receipt.json'); binding=read(OUT/'export-binding.json')
    need(rec['builder_sha256']==sha(__file__) and rec['parent_pins']==PINS and rec['decision']==DECISION,'build registration drift')
    need(binding['sha256']==CHECKPOINT_SHA and binding['placeholder'] is False and
         binding['training_provenance']['protocol_sha256']==PROTOCOL_SHA,'binding drift')
    need(rec['placeholder'] is False and rec['clean_oof'] is False and rec['exposure']=='EXPOSED' and
         rec['qualified_for_submission'] is False,'receipt status drift')
    for name,pin in rec['artifacts'].items(): need(sha(OUT/name)==pin,'artifact pin: '+name)
    hashes,small=archive_files(OUT/'dataset'/ARCHIVE)
    need(hashes==rec['files'] and hashes[NAME]==CHECKPOINT_SHA,'archive identity drift')
    need(small=={**original,'release-manifest.json':encoded(manifest(original,binding))},'embedded scope drift')
    nb,meta=payload(original,binding,sha(OUT/'dataset'/ARCHIVE))
    need(read(OUT/'kernel/release.ipynb')==nb and read(OUT/'kernel/kernel-metadata.json')==meta,'kernel scope drift')
    need(read(OUT/'dataset/dataset-metadata.json')==dataset_metadata(),'dataset metadata drift')
    need((OUT/'full-diff-vs-parent.txt').read_bytes()==diff(nb,meta,small,hashes),'diff drift')
    need(set(p.name for p in (OUT/'dataset').iterdir())=={ARCHIVE,'files.json','dataset-metadata.json'},'dataset inventory drift')
    kfiles={n:sha(OUT/'kernel'/n) for n in ('kernel-metadata.json','release.ipynb')}
    return dict(status='PASS_CPU_SCOPE',kernel_files=kfiles,kernel_sha256=digest(canonical(kfiles)),qualified_for_submission=False)


def register():
    result=validate()
    value=dict(version=VERSION,purpose='Standalone public kill test of the registered HB177 recipe; no blend tuning',
        decision=DECISION,exposure='EXPOSED',clean_oof=False,kernel_id=SLUG,kernel_files=result['kernel_files'],
        checkpoint_sha256=CHECKPOINT_SHA,parent=PARENT.relative_to(ROOT).as_posix(),
        fit_result_review='REQUIRED; completion is not approval',
        runtime=dict(status='HOLD',basis='HB156 measured hidden seconds multiplied by measured paired serving cost ratio; <=32400 seconds',
                     hidden_seconds=None,prohibited_substitutes=['training duration','FAST3 duration']),
        source_sha256=sha(__file__),build_receipt_sha256=sha(OUT/'build-receipt.json'))
    path=OUT/'kill-registration.json'
    if path.exists(): need(read(path)==value,'immutable kill registration drift')
    else: write(path,encoded(value))
    return dict(status='REGISTERED',runtime='HOLD_MISSING_HB156_HIDDEN_SECONDS',kernel_sha256=result['kernel_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['build','validate','register','cpu-smoke'])
    a=p.parse_args()
    cpu_only()
    if a.command=='cpu-smoke':
        from tools.hb177_postfit_smoke_v1 import run
        result=run()
    else: result={'build':build,'validate':validate,'register':register}[a.command]()
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
