"""Exact K2 phase gate k2-1.0.0. No historical policy/global mutation.

pin -> independent dispatched review -> bind/check-build. Final admission adds
independently reviewed real qualification evidence; BUILD_READY cannot submit.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import re
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import strict_variant_gate as base
from tools import l1_rco_k2_v101 as b
from tools import l1_rco_k2_worker_v101 as w

VERSION='k2-1.0.1'
PHASES={'BUILD_READY','FINAL_READY'}


def context():
    """Private cloned namespace, following HB177; old gates remain untouched."""
    g=types.ModuleType('_k2_private_gate')
    g.__dict__.update(vars(base))
    def kernel(stage,inputs):
        material=snapshot(stage,inputs)
        return (material['metadata'],{},dict(kernel_files=material['kernel_files']),{},[])
    g.kernel=kernel
    f=base.source_binding
    g.source_binding=types.FunctionType(f.__code__,g.__dict__,f.__name__,f.__defaults__,f.__closure__)
    return g


def snapshot(stage,inputs):
    stage=base.local_path(stage,inputs.root)
    base.need(stage.relative_to(inputs.root).as_posix()==b.STAGE.relative_to(b.ROOT).as_posix(), 'exact K2 stage')
    reg=inputs.json(stage/'registration.json')
    base.need(reg['schema']=='k2-registration/v1' and reg['candidate_id']==b.CANDIDATE and
        reg['kernel_id']==w.SLUG and reg['version']==w.VERSION and reg['contract']==w.CONTRACT and
        reg['qualified_for_submission'] is False, 'K2 registration contract')
    base.need(all(reg['runtime'][k]==v for k,v in b.TIMING_POLICY.items()),'registered complete runtime policy')
    source=inputs.json(stage/'source-manifest.json')
    base.need(base.sha(inputs.read(stage/'source-manifest.json'))==reg['source_manifest_sha256'], 'source manifest binding')
    for name,pin in source.items():
        base.need(inputs.digest(name)==pin, 'frozen/new source pin: '+name)
    base.need(set(b.NEW+b.FROZEN)<=set(source), 'required policy source inventory')
    for name,pin in b.PINS.items():
        base.need(source[Path(name).as_posix()]==pin, 'authenticated K1 lineage')
    assets=base.local_path(reg['assets'],inputs.root)
    am=inputs.json(assets/'asset-manifest.json')
    base.need(base.sha(inputs.read(assets/'asset-manifest.json'))==reg['asset_manifest_sha256'], 'asset manifest pin')
    base.need(am['contract']==w.CONTRACT and am['version']==w.VERSION and am['dataset']==b.DATASET and am['private'] is True and
        am['wheels']==b.s6.WHEEL_HASHES and am['deployment_versions']==b.s6.WHEEL_VERSIONS, 'asset settings/wheels')
    paths=sorted(p for p in assets.rglob('*') if p.is_file());inputs.folders[assets]=paths
    base.need({p.relative_to(assets).as_posix() for p in paths}==set(am['files'])|{'asset-manifest.json','dataset-metadata.json'}, 'added/removed asset')
    for name,pin in am['files'].items():
        base.safe_name(name)
        base.need(inputs.digest(assets/name)==pin,'asset tamper: '+name)
    base.need({n for n in am['files'] if n.endswith(('.pt','.safetensors'))}=={'weights/E_prior_s0_f0.pt','weights/anatomy.pt'} and
        am['files']['weights/E_prior_s0_f0.pt']==w.MODEL_SHA and am['files']['weights/anatomy.pt']==w.ANATOMY_SHA, 'one model/anatomy')
    dmeta=inputs.json(assets/'dataset-metadata.json')
    base.need(dmeta['id']==b.DATASET and dmeta['title']==b.DATASET.split('/')[1], 'asset slug')
    origins=inputs.json(b.WORK.relative_to(b.ROOT)/'asset-origins.json')
    for name,v in origins.items():
        base.need(am['files'][name]==v['sha256']==inputs.digest(v['source']) and v['relation']=='byte-identical','origin diff')
    raw={n:inputs.read(stage/'kernel'/n) for n in ('release.ipynb','kernel-metadata.json')}
    paths=sorted(p for p in (stage/'kernel').rglob('*') if p.is_file()); inputs.folders[stage/'kernel']=paths
    base.need({p.name for p in paths}==set(raw),'kernel inventory')
    metadata=base.parse(raw['kernel-metadata.json'])
    expected=dict(id=w.SLUG,title=w.SLUG.split('/')[1],code_file='release.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[b.DATASET,b.CODEC_DATASET],competition_sources=['rsna-knee-abnormality-detection'],kernel_sources=[])
    base.need(metadata==expected,'exact K2 metadata')
    base.need(base.parse(raw['release.ipynb'])==b.notebook(reg['asset_manifest_sha256']), 'notebook differs from registered new path')
    inv=inputs.json(stage/'embedded-file-manifest.json')
    base.need(inv['new_sources']=={n:source[n] for n in b.NEW} and inv['inherited_assets']==origins and
        inv['removed']==[] and inv['changed_inherited']==[] and
        inv['notebook_sha256']==base.sha(raw['release.ipynb']) and
        inv['metadata_sha256']==base.sha(raw['kernel-metadata.json']) and
        inv['complete_diff_sha256']==base.sha(inputs.read(stage/'full-inventory-diff.txt')), 'full inventory proof')
    required=[stage/n for n in ('registration.json','source-manifest.json','embedded-file-manifest.json',
        'full-inventory-diff.txt','kernel/release.ipynb','kernel/kernel-metadata.json')]
    _,manifest_sha=base.manifest(stage,inputs,required)
    kernel_files={n:base.sha(v) for n,v in raw.items()}
    return dict(registration=reg,metadata=metadata,kernel_files=kernel_files,
        kernel_sha256=base.sha(base.encoded(kernel_files)),manifest_sha256=manifest_sha)


def reused_cpu_check(rec,inputs,stage):
    base.need(rec['status']=='REUSE_REQUIRES_INDEPENDENT_BUILD_REVIEW' and rec['old_measurements_relabelled'] is False and
        rec['qualified_for_submission'] is False and rec['registration_sha256']==inputs.digest(stage/'registration.json'),
        'narrow derivative CPU reuse identity')
    old_stage=base.local_path(rec['old_stage'],inputs.root)
    base.need(old_stage.relative_to(inputs.root)==b.OLD_STAGE.relative_to(b.ROOT) and
        inputs.digest(old_stage/'registration.json')==b.OLD_REG_SHA,'reuse exact old stage')
    # Replays the historical authentic source/assets snapshot; no old timing gate.
    from tools import strict_variant_gate_k2_v1 as old_gate
    old_gate.snapshot(old_stage,inputs)
    p=rec['proof'];base.need(inputs.digest(p['path'])==p['sha256'],'derivative proof hash')
    proof=inputs.json(p['path'])
    base.need(proof==b.derivative_proof(old_stage,stage),'derivative proof replay')
    c=rec['old_collection'];base.need(c['sha256']==b.OLD_COLLECTION_SHA==inputs.digest(c['path']),'old completed CPU collection')
    collection=inputs.json(c['path'])
    # Authenticate every old receipt/output without decoding any prediction rows.
    base.need(collection['status']=='PASS' and collection['registration_sha256']==b.OLD_REG_SHA and
        collection['raw_cached_images_exact'] is True and collection['qualified_for_submission'] is False and
        set(collection['parts'])=={'model','spawn','raw-visible','raw-holdout','cached-reference'},'old CPU collection scope')
    parts={}
    for name,ref in collection['parts'].items():
        base.need(inputs.digest(ref['path'])==ref['sha256'],'old CPU part hash')
        part=inputs.json(ref['path']);parts[name]=part
        base.need(part['part']==name and part['status']=='PASS' and part['registration_sha256']==b.OLD_REG_SHA and
            part['manifest_sha256']==inputs.json(old_stage/'registration.json')['asset_manifest_sha256'] and
            part['environment']['device']=='cpu' and part['environment']['cuda_initialized'] is False and
            part['cuda_visible_devices']=='-1','old CPU isolation/source')
        parent=base.local_path(ref['path'],inputs.root).parent
        if name.startswith('raw-'):
            wr=inputs.json(parent/'worker-receipt.json')
            base.need(inputs.digest(parent/'worker-receipt.json')==part['worker_receipt_sha256'] and
                inputs.digest(parent/'submission.csv')==part['prediction_sha256']==wr['output_sha256'] and
                wr['status']=='COMPLETE' and wr['version']=='1.0.0-rcof0' and wr['contract']==w.CONTRACT and
                wr['fallbacks']['prediction']==0 and wr['study_count']==(3 if name=='raw-visible' else 2),'old raw CPU binding')
        if name=='cached-reference':base.need(inputs.digest(parent/'reference.csv')==part['reference_sha256'] and part['k1_bf16']['status']=='PASS','old cached output')
    base.need(parts['model']['initialization_plus_deploy_exact'] is True and parts['spawn']['exact'] is True and
        parts['spawn']['workers']==[0,3] and parts['spawn']['studies']==2,'old init/spawn scope')
    rawdir=base.local_path(collection['parts']['raw-holdout']['path'],inputs.root).parent
    raw=inputs.json(rawdir/'worker-receipt.json');cached=parts['cached-reference']
    base.need([{k:s[k] for k in ('images_sha256','metadata_sha256','windows')} for s in raw['studies']]==cached['tensors'] and
        parts['raw-holdout']['prediction_sha256']==cached['reference_sha256'],'old raw/cache exact byte parity')
    base.need(collection['same_runtime']['status']=='PASS' and collection['same_runtime']['max_abs']==0 and
        collection['same_runtime']['mean_abs']==0,'old exact FP32 comparison')
    return rec


def cpu_check(path,inputs,stage):
    rec=inputs.json(path)
    if rec.get('schema')=='k2-cpu-reuse/v1.0.1':return reused_cpu_check(rec,inputs,stage)
    base.need(rec['schema']=='k2-cpu-collection/v1' and rec['status']=='PASS' and
        rec['qualified_for_submission'] is False and rec['raw_cached_images_exact'] is True and
        rec['same_runtime']['status']=='PASS' and
        rec['registration_sha256']==base.sha(inputs.read(stage/'registration.json')), 'complete real CPU qualification')
    base.need(set(rec['parts'])=={'model','spawn','raw-visible','raw-holdout','cached-reference'}, 'CPU parts missing')
    for name,p in rec['parts'].items():
        part=inputs.json(p['path'])
        base.need(base.sha(inputs.read(p['path']))==p['sha256'] and part['part']==name and part['status']=='PASS', 'CPU part changed')
        base.need(part['environment']['device']=='cpu' and part['environment']['cuda_initialized'] is False and
            part['cuda_visible_devices']=='-1' and part['registration_sha256']==rec['registration_sha256'], 'CPU isolation/identity')
        parent=base.local_path(p['path'],inputs.root).parent
        if name.startswith('raw-'):
            worker=inputs.json(parent/'worker-receipt.json')
            base.need(base.sha(inputs.read(parent/'worker-receipt.json'))==part['worker_receipt_sha256'] and
                inputs.digest(parent/'submission.csv')==part['prediction_sha256']==worker['output_sha256'] and
                worker['status']=='COMPLETE' and worker['fallbacks']['prediction']==0 and
                worker['study_count']==(3 if name=='raw-visible' else 2), 'raw CPU evidence')
        if name=='spawn':
            base.need(part['workers']==[0,3] and part['studies']==2 and part['exact'] is True, 'actual spawn workers3')
        if name=='model':base.need(part['initialization_plus_deploy_exact'] is True,'strict initialization equivalence')
        if name=='cached-reference':
            base.need(inputs.digest(parent/'reference.csv')==part['reference_sha256'],'CPU reference changed')
            base.need(part['k1_bf16']['status']=='PASS' and part['studies']==2,'CPU saved-bf16 parity')
    rawpath=base.local_path(rec['parts']['raw-holdout']['path'],inputs.root).parent
    refpath=base.local_path(rec['parts']['cached-reference']['path'],inputs.root).parent
    raw=inputs.json(rawpath/'worker-receipt.json');cached=inputs.json(refpath/'result.json')
    base.need([{k:s[k] for k in ('images_sha256','metadata_sha256','windows')} for s in raw['studies']]==cached['tensors'],
              'CPU raw/cache tensor parity replay')
    ids=inputs.json(stage/'registration.json')['smoke_holdout_ids']
    comparison=b.parity(b.prediction_table(rawpath/'submission.csv',ids),b.prediction_table(refpath/'reference.csv',ids),'same_runtime')
    base.need(comparison==rec['same_runtime'] and comparison['status']=='PASS','CPU probability parity replay')
    return rec


def pin(stage,cpu,builder_receipt,output,root=ROOT):
    inputs=base.Inputs(Path(root));stage=base.local_path(stage,inputs.root)
    m=snapshot(stage,inputs);cpu_check(cpu,inputs,stage)
    g=context()
    binding=dict(source_stage=stage.relative_to(inputs.root).as_posix(),review='',
        builder_receipt=base.local_path(builder_receipt,inputs.root).relative_to(inputs.root).as_posix(),
        builder_receipt_sha256=base.sha(inputs.read(builder_receipt)))
    lane,_=g.source_binding(dict(review_binding=binding),g.kernel(stage,inputs),inputs)
    rec=dict(schema='k2-build-pin/v1',status='PINNED_NOT_ADMITTED',gate_version=VERSION,
        kernel_id=w.SLUG,stage=stage.relative_to(inputs.root).as_posix(),kernel_sha256=m['kernel_sha256'],
        manifest_sha256=m['manifest_sha256'],builder_lane=lane,builder_receipt=binding['builder_receipt'],
        cpu=base.local_path(cpu,inputs.root).relative_to(inputs.root).as_posix(),input_sha256=inputs.hashes(),
        qualified_for_submission=False)
    inputs.recheck();w.need(not Path(output).exists(),'pin output exists');w.write(output,rec)
    return rec


def checked_pin(path,inputs):
    pin=inputs.json(path)
    base.need(pin['schema']=='k2-build-pin/v1' and pin['status']=='PINNED_NOT_ADMITTED' and
        pin['kernel_id']==w.SLUG and pin['gate_version']==VERSION,'K2 pin identity')
    for name,digest in pin['input_sha256'].items():base.need(inputs.digest(name)==digest,'pin dependency changed')
    stage=base.local_path(pin['stage'],inputs.root);m=snapshot(stage,inputs);cpu_check(pin['cpu'],inputs,stage)
    lane,report,_=base.dispatch(pin['builder_receipt'],inputs)
    base.need(lane==pin['builder_lane'] and
        re.findall(r'^Kernel-SHA256: ([0-9a-f]{64})\s*$',report.decode('utf-8-sig'),re.M)==[m['kernel_sha256']], 'builder identity/hash')
    base.need(pin['kernel_sha256']==m['kernel_sha256'] and pin['manifest_sha256']==m['manifest_sha256'],'pin material changed')
    return pin


def independent_review(path,pin_path,builder_lane,inputs,label='K2-Pin-SHA256'):
    path=base.local_path(path,inputs.root)
    text=base.verdict(inputs.read(path))
    lane,report,actual=base.dispatch(str(path)+'.receipt.json',inputs)
    base.need(lane!=builder_lane and actual==path and report==inputs.read(path),'independent dispatched review required')
    base.need(re.findall(r'^Reviewer-Lane: ([A-Za-z0-9_-]+)\s*$',text,re.M)==[lane],'reviewer identity')
    base.need(re.findall('^'+label+r': ([0-9a-f]{64})\s*$',text,re.M)==[base.sha(inputs.read(pin_path))], 'review binding pin')
    base.need(w.SLUG in text,'review exact slug')
    return lane


def bind(pin_path,review,output,root=ROOT):
    inputs=base.Inputs(Path(root));pin=checked_pin(pin_path,inputs)
    lane=independent_review(review,pin_path,pin['builder_lane'],inputs)
    rec=dict(schema='k2-gate/v1',status='PASS',errors=[],gate_version=VERSION,phase='BUILD_READY',
        qualified_for_submission=False,fixture_only=False,kernel_id=w.SLUG,stage=pin['stage'],
        kernel_sha256=pin['kernel_sha256'],pin=base.local_path(pin_path,inputs.root).relative_to(inputs.root).as_posix(),
        review=base.local_path(review,inputs.root).relative_to(inputs.root).as_posix(),reviewer_lane=lane,
        input_sha256=inputs.hashes())
    inputs.recheck();w.need(not Path(output).exists(),'gate output exists');w.write(output,rec)
    return rec


def verify_receipt(path,inputs,phase='FINAL_READY'):
    rec=inputs.json(path)
    base.need(phase in PHASES and rec['schema']=='k2-gate/v1' and rec['status']=='PASS' and rec['errors']==[] and
        rec['gate_version']==VERSION and rec['phase']==phase and rec['kernel_id']==w.SLUG and
        rec['qualified_for_submission'] is (phase=='FINAL_READY') and rec['fixture_only'] is False,'K2 phase not admitted')
    for name,pin in rec['input_sha256'].items():base.need(inputs.digest(name)==pin,'gate dependency changed')
    pin=checked_pin(rec['pin'],inputs)
    reviewer=independent_review(rec['review'],rec['pin'],pin['builder_lane'],inputs)
    base.need(reviewer==rec['reviewer_lane'] and rec['stage']==pin['stage'] and
        rec['kernel_sha256']==pin['kernel_sha256'],'build gate replay')
    if phase=='FINAL_READY':
        from tools import l1_rco_k2_admission_v101 as admission
        admission.replay_final(rec,inputs,pin)
    inputs.recheck()
    return rec


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('pin')
    for n in ('stage','cpu','builder-receipt','output'):q.add_argument('--'+n,type=Path,required=True)
    q=s.add_parser('bind')
    for n in ('pin','review','output'):q.add_argument('--'+n,type=Path,required=True)
    for mode in ('check-build','check-final'):
        q=s.add_parser(mode);q.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='pin':r=pin(a.stage,a.cpu,a.builder_receipt,a.output)
        elif a.command=='bind':r=bind(a.pin,a.review,a.output)
        else:r=verify_receipt(a.receipt,base.Inputs(ROOT),'BUILD_READY' if a.command=='check-build' else 'FINAL_READY')
        print(w.encoded({k:r[k] for k in ('status','kernel_id','qualified_for_submission')} ).decode());return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(w.encoded(dict(status='FAIL',error=str(exc),qualified_for_submission=False)).decode());return 1


if __name__=='__main__':sys.exit(main())
