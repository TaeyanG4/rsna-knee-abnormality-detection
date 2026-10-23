"""H-B-157 standalone public-R kill test v1.0.0; local preparation only.

Changelog: exact public-R notebook + reviewed cell zero; checkpoint bindings
only. No runtime launch/import/process changes. No account operations.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import gc
import json
import math
import os
from pathlib import Path
import shutil
from tools import hb157_rkd_v1 as b
from tools.hb150_io_v1 import atomic_json,safe_replace,_retry

ROOT=b.ROOT
SCRATCH=b.WORK
OUT=SCRATCH/'killtest-v1'
PARENT=ROOT/'storage/external/nartaa__rsna-knee-0949-anatomical-mirror/rsna-knee-0949-anatomical-mirror.ipynb'
PARENT_SHA='8f1f96530b08543bd6caeed48489eea7c01657d6994f22e21a8400047fd4e100'
DONOR=ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.18-own25q3/kernel/release.ipynb'
DONOR_SHA='98668e17d4419249ccce63641b17b90e5ec9bb3612f72ffb875ef3b33d3f5855'
NAME='raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
DATASET='taeyangg4/knee-hb157-rkd-weights-v1'
SLUG='taeyangg4/knee-hb157-rkd-standalone-v1'
need,read,sha=b.need,b.read,b.sha
source=lambda cell: ''.join(cell['source'])

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    _retry(lambda:temp.write_bytes(data));safe_replace(temp,path)

def parents():
    need(sha(PARENT)==PARENT_SHA and sha(DONOR)==DONOR_SHA,'serving parent/donor changed')
    return read(PARENT),read(DONOR)

def payload(checkpoint_sha):
    nb,donor=parents();nb=copy.deepcopy(nb)
    replacements={b.PARENT_SHA:checkpoint_sha,
        '/kaggle/input/rsna-knee-publication-swa-weights-20261007':'/kaggle/input/'+DATASET.split('/')[1],
        '/kaggle/input/datasets/nartaa/rsna-knee-publication-swa-weights-20261007':'/kaggle/input/datasets/'+DATASET,
        '/kaggle/input/rsna-knee-oai-original384-s42-pair':'/kaggle/input/'+DATASET.split('/')[1],
        '/kaggle/input/datasets/nartaa/rsna-knee-oai-original384-s42-pair':'/kaggle/input/datasets/'+DATASET}
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        s=source(cell)
        for before,after in replacements.items():s=s.replace(before,after)
        cell['source']='if not FAST:\n'+''.join('    '+line for line in s.splitlines(True))
        compile(cell['source'],'<guarded-public-R>','exec')
    nb['cells'].insert(0,copy.deepcopy(donor['cells'][0]))
    return nb

def embedded(nb,guarded=False):
    found={}
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']!='code':continue
        for n in ast.walk(ast.parse(source(cell))):
            if isinstance(n,ast.Assign):
                for target in n.targets:
                    if isinstance(target,ast.Name) and target.id.endswith('_B64'):
                        need(isinstance(n.value,ast.Constant),'nonliteral embedded payload')
                        found[target.id]=base64.b64decode(n.value.value,validate=True)
    need(set(found)=={'_FASTREAD_B64'},'unexpected embedded-file inventory')
    return found

def metadata():
    return dict(id=SLUG,title=SLUG.split('/')[1],code_file='release.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[DATASET],competition_sources=['rsna-knee-abnormality-detection'],kernel_sources=[])

def dataset_metadata():
    return dict(id=DATASET,title=DATASET.split('/')[1],licenses=[dict(name='other')],
        description='Private H-B-157 R-KD EXPOSED clean_oof=false checkpoint. Prepared locally; not uploaded.')

def scope():
    receipt=read(OUT/'build-receipt.json');nb=read(OUT/'kernel/release.ipynb');parent,_=parents()
    need(nb==payload(receipt['checkpoint_sha256']),'out-of-scope notebook change')
    need(read(OUT/'kernel/kernel-metadata.json')==metadata(),'out-of-scope kernel metadata')
    need(read(OUT/'dataset/dataset-metadata.json')==dataset_metadata(),'out-of-scope dataset metadata')
    need(sha(OUT/'dataset'/NAME)==receipt['checkpoint_sha256'],'staged checkpoint bytes changed')
    before,after=embedded(parent),embedded(nb,True)
    need(before==after,'embedded file changed')
    need((OUT/'unpacked/_fastread.py').read_bytes()==after['_FASTREAD_B64'],'unpacked bytes changed')
    # Full JSON diff includes every cell and all notebook metadata; embedded diff
    # is separately emitted even when identical. Exact payload equality is gate.
    diff=''.join(difflib.unified_diff(json.dumps(parent,indent=2,ensure_ascii=False).splitlines(True),
        json.dumps(nb,indent=2,ensure_ascii=False).splitlines(True),fromfile='public-R',tofile='HB157'))
    write(OUT/'full-notebook.diff',diff.encode('utf8'))
    write(OUT/'full-metadata.diff',('NEW LOCAL kernel-metadata.json\n'+json.dumps(metadata(),indent=2)+
        '\nNEW LOCAL dataset-metadata.json\n'+json.dumps(dataset_metadata(),indent=2)).encode())
    write(OUT/'embedded.diff',b'Identical complete inventory: _FASTREAD_B64 -> _fastread.py\n')
    return dict(status='PASS_CPU_SCOPE',placeholder=receipt['placeholder'],embedded_files=1,
        runtime_changes='checkpoint SHA/mount bindings and exact reviewed FAST guard only',
        parent_sha256=PARENT_SHA,donor_sha256=DONOR_SHA)

def stage(checkpoint,placeholder):
    b.validate();parents();pin=sha(checkpoint)
    if (OUT/'build-receipt.json').exists():
        old=read(OUT/'build-receipt.json')
        need(old['placeholder'] or (not placeholder and old['checkpoint_sha256']==pin),
             'final stage immutable; new version required')
    dest=OUT/'dataset'/NAME;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists() or sha(dest)!=pin:
        temp=dest.with_suffix('.partial');_retry(lambda:shutil.copyfile(checkpoint,temp));safe_replace(temp,dest)
    atomic_json(OUT/'kernel/release.ipynb',payload(pin))
    atomic_json(OUT/'kernel/kernel-metadata.json',metadata());atomic_json(OUT/'dataset/dataset-metadata.json',dataset_metadata())
    write(OUT/'unpacked/_fastread.py',embedded(payload(pin))['_FASTREAD_B64'])
    atomic_json(OUT/'build-receipt.json',dict(version='1.0.0',checkpoint_sha256=pin,
        source=str(checkpoint.relative_to(ROOT)),placeholder=placeholder,exposure='EXPOSED',clean_oof=False,
        protocol_sha256=sha(b.PROTOCOL),parent_sha256=PARENT_SHA,donor_sha256=DONOR_SHA,
        standalone_alive=0.952,standalone_dead=0.949,otherwise='INCONCLUSIVE',swap_keep=0.957))
    return scope()

def prepare():
    need(sha(ROOT/b.PARENT)==b.PARENT_SHA,'placeholder public-R identity')
    return stage(ROOT/b.PARENT,True)

def final_checkpoint():
    budget=read(b.WORK/'budget.json')
    receipt=read(b.WORK/('fit-attempt-'+str(budget['attempt']))/'receipt.json')
    need(not budget['running'] and receipt['exit_code']==0 and receipt['reason']=='completed' and
         not receipt['survivors'],'fit supervisor not successfully complete')
    summary=read(b.RUN/'summary.json');checkpoint=b.RUN/'model_best.pt'
    need(summary['status']=='COMPLETE_TRAINING_AND_READOUT' and summary['updates']==1632 and
         summary['protocol_sha256']==sha(b.PROTOCOL) and summary['model_sha256']==sha(checkpoint),
         'incomplete/stale fit')
    import torch
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    p=ck['provenance'];need(p['mode']=='fit' and p['device']=='cuda' and p['epochs']==6 and
        p['run_dir']==b.RUN.relative_to(ROOT).as_posix() and p['protocol_sha256']==sha(b.PROTOCOL) and
        p['soft_label_sha256']==b.LABEL_SHA and p['public_parent_sha256']==b.PARENT_SHA and
        p['pos_weight']==b.scientific()['pos_weight'] and p['bn']['unique_studies']==4347 and
        ck['seed']==44 and ck['swa_epochs_1based']==[4,5,6] and ck['clean_oof'] is False and
        ck['exposure']=='EXPOSED','candidate provenance differs')
    return checkpoint

def bind():
    b.require_fit();return stage(final_checkpoint(),False)

def outcome(score):
    need(math.isfinite(score) and 0<=score<=1,'invalid public result')
    return 'ALIVE' if score>=.952 else 'DEAD' if score<=.949 else 'INCONCLUSIVE'

def admit():
    scope();r=read(OUT/'build-receipt.json')
    need(not r['placeholder'],'placeholder cannot be released')
    need(sha(final_checkpoint())==r['checkpoint_sha256'],'stale candidate')
    smoke=read(OUT/'cpu-smoke.json')
    need(smoke['status']=='PASS' and smoke['notebook_sha256']==sha(OUT/'kernel/release.ipynb') and
         smoke['build_receipt_sha256']==sha(OUT/'build-receipt.json') and
         smoke['protocol_sha256']==sha(b.PROTOCOL),'release smoke missing/stale')
    b.check_review(read(OUT/'release-review.json'),dict(protocol_sha256=b.PROTOCOL,
        build_receipt_sha256=OUT/'build-receipt.json',cpu_smoke_sha256=OUT/'cpu-smoke.json',
        notebook_sha256=OUT/'kernel/release.ipynb'))
    return dict(status='READY_LOCAL_ONLY',network_operations=False)

def loader_smoke():
    import torch,timm
    torch.set_num_threads(4)
    nb=read(OUT/'kernel/release.ipynb');nodes=[]
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            nodes.extend(n for n in ast.walk(ast.parse(source(cell))) if isinstance(n,(ast.FunctionDef,ast.ClassDef))
                         and n.name in {'build_backbone','RaptorClassifier','load_model'})
    need(len(nodes)==3,'exact serving loader definitions')
    ns=dict(torch=torch,nn=torch.nn,timm=timm,gc=gc,INPUT_RES=320)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-R-loader-only>','exec'),ns)
    from unittest.mock import patch
    with patch('hashlib.file_digest') as wrong:
        wrong.return_value.hexdigest.return_value='0'*64
        try:ns['load_model'](OUT/'dataset'/NAME,'invalid',384,'cpu')
        except AssertionError:pass
        else:raise AssertionError('loader accepted wrong SHA')
    model,res=ns['load_model'](OUT/'dataset'/NAME,'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',384,'cpu')
    need(res==320 and not model.training and next(model.parameters()).device.type=='cpu','loader shape/device')
    del model;gc.collect();need(not torch.cuda.is_initialized(),'CPU loader initialized CUDA')
    return dict(status='PASS',exact_R_loader=True,strict_load=True,res=320,wrong_sha='REJECT',cuda_initialized=False)

def cpu_smoke():
    scope()
    from tools.hb157_rkd_fixtures_v1 import run_smoke
    results=run_smoke(SCRATCH/'fast-cases')
    result=dict(status='PASS',fast_path_cases=len(results['cases']),loader=loader_smoke(),
        limitation=results['limitation'],protocol_sha256=sha(b.PROTOCOL),
        notebook_sha256=sha(OUT/'kernel/release.ipynb'),build_receipt_sha256=sha(OUT/'build-receipt.json'))
    atomic_json(OUT/'cpu-smoke.json',result);return result

def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1')
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','validate','cpu-smoke','bind','admit-dry-run']);args=ap.parse_args()
    if args.action=='admit-dry-run':
        try:result=admit()
        except (OSError,ValueError,KeyError) as exc:result=dict(status='HOLD',reason=str(exc),network_operations=False)
        atomic_json(OUT/'admit-dry-run.json',result)
    else:result={'prepare':prepare,'validate':scope,'cpu-smoke':cpu_smoke,'bind':bind}[args.action]()
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
