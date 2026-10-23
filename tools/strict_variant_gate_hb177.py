"""Strict HB177 standalone gate v1.3.1. Additive HB156 checkpoint-only policy.

Reuses v1.3.0 bind/review/dispatcher rules through private function namespaces;
never mutates the admitted HB156 gate or its global policy.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import types
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import strict_variant_gate as base
from tools import strict_variant_gate_standalone as s
from tools import hb177_postfit_v1 as b
VERSION='1.3.1'


def kernel(stage,inputs,g):
    raw,paths=s.raw_kernel(stage,inputs,g)
    meta=g.parse(raw['kernel-metadata.json'])
    g.need(meta['id']==b.SLUG,'HB177 exact slug required')
    reg=inputs.json(stage/'registration.json') if (stage/'registration.json').exists() else {}
    source=g.local_path(reg.get('external_source',str(stage)),inputs.root)
    original,_=s.raw_kernel(source,inputs,g)
    g.need(raw==original,'bound source kernel differs')
    rec=inputs.json(source/'build-receipt.json'); binding=inputs.json(source/'export-binding.json')
    g.need(rec['version']=='1.0.0' and rec['parent_pins']==b.PINS and rec['decision']==b.DECISION and
           rec['placeholder'] is False and rec['clean_oof'] is False and rec['exposure']=='EXPOSED' and
           rec['qualified_for_submission'] is False and rec['status']=='BOUND_NOT_REVIEWED','HB177 build identity')
    g.need(binding['sha256']==b.CHECKPOINT_SHA and binding['placeholder'] is False and binding['clean_oof'] is False
           and binding['exposure']=='EXPOSED','HB177 final checkpoint identity')
    provenance=binding['training_provenance']
    g.need(provenance['protocol_sha256']==b.PROTOCOL_SHA and provenance['seed']==42 and provenance['host']=='local'
           and provenance['ema_updates']==5440 and provenance['checkpoint_kind']=='final-epoch-EMA'
           and provenance['swa_epochs_1based']==[],'HB177 authoritative EMA provenance')
    dataset=source/'dataset'; dataset_paths=sorted(p for p in dataset.rglob('*') if p.is_file())
    inputs.folders[dataset]=dataset_paths
    g.need({p.relative_to(dataset).as_posix() for p in dataset_paths}=={b.ARCHIVE,'files.json','dataset-metadata.json'}
           and all(not p.is_symlink() and p.resolve().is_relative_to(dataset.resolve()) for p in dataset_paths),'HB177 dataset file inventory')
    for name,pin in b.PINS.items():
        path=inputs.root/b.PARENT.relative_to(b.ROOT)/name
        actual=inputs.digest(path) if name.endswith('.bin') else g.sha(inputs.read(path))
        g.need(actual==pin,'HB156 immutable parent pin: '+name)
    parent=inputs.root/b.PARENT.relative_to(b.ROOT)
    praw={n:inputs.read(parent/'kernel'/n) for n in ('release.ipynb','kernel-metadata.json')}
    ph,ps=s.h.archive(parent/'dataset'/b.ARCHIVE,b.PINS['dataset/'+b.ARCHIVE],inputs,g)
    ash=rec['artifacts']['dataset/'+b.ARCHIVE]
    hashes,small=s.h.archive(dataset/b.ARCHIVE,ash,inputs,g)
    g.need(hashes==rec['files'] and hashes[b.NAME]==b.CHECKPOINT_SHA,'HB177 pinned final checkpoint bytes')
    # Exact checkpoint hash is authoritative, not an unsafe string search for the
    # legitimate fixture=False key in its torch pickle. No unpickling in gate.
    m=g.parse(ps['release-manifest.json'])
    m['checkpoints'][b.NAME]['sha256']=b.CHECKPOINT_SHA
    m['asset_sha256'][b.NAME]=b.CHECKPOINT_SHA
    m['hb151_export']['sha256']=b.CHECKPOINT_SHA
    m['dataset']=b.DATASET; m['hb177_provenance']=binding
    expected={**ps,'release-manifest.json':s.h.wire(m)}
    g.need(small==expected,'HB177 embedded code/manifest exceeds checkpoint-only policy')
    nb=g.parse(praw['release.ipynb']); text=''.join(nb['cells'][5]['source']).encode()
    for before,after,count in [
        (b'knee-hb156-cnxs-l2-weights-v1',b.DATASET.split('/')[1].encode(),2),
        (b.PINS['dataset/'+b.ARCHIVE].encode(),ash.encode(),1),
        (g.sha(ps['release-manifest.json']).encode(),g.sha(expected['release-manifest.json']).encode(),1)]:
        text=s.replace(text,before,after,count,g)
    nb['cells'][5]['source']=text.decode()
    expected_meta=g.parse(praw['kernel-metadata.json'])
    expected_meta.update(id=b.SLUG,title=b.SLUG.split('/')[1],dataset_sources=[b.DATASET])
    g.need(g.parse(raw['release.ipynb'])==nb and meta==expected_meta,'HB177 notebook/metadata exceeds registered delta')
    dmeta=inputs.json(parent/'dataset/dataset-metadata.json')
    dmeta.update(id=b.DATASET,title=b.DATASET.split('/')[1],description=
        'Private H-B-177 ConvNeXt-S seed42 final epoch EMA. EXPOSED / clean_oof=false. Legacy SWA metadata is loader ABI only. No training data.')
    g.need(inputs.json(dataset/'dataset-metadata.json')==dmeta,'HB177 dataset metadata drift')
    spec=inputs.json(dataset/'files.json')
    g.need(spec['id']==b.DATASET and spec['private'] is True and spec['archive_members']==hashes
           and spec['checkpoint_sha256']==b.CHECKPOINT_SHA,'private upload spec drift')
    for name in (b.ARCHIVE,'dataset-metadata.json'):
        g.need(spec['files'][name]==dict(sha256=ash if name==b.ARCHIVE else g.sha(inputs.read(dataset/name)),
                                      bytes=(dataset/name).stat().st_size),'upload file pin drift')
    for name,pin in rec['artifacts'].items():
        actual=inputs.digest(source/name) if name.endswith('.bin') else g.sha(inputs.read(source/name))
        g.need(actual==pin,'HB177 artifact pin: '+name)
    smoke=inputs.json(source/'cpu-smoke.json')
    g.need(smoke['status']=='PASS' and smoke['studies']==2 and smoke['placeholder'] is False and
           smoke['cuda_visible_devices']=='' and smoke['cuda_initialized'] is False and
           smoke['build_receipt_sha256']==g.sha(inputs.read(source/'build-receipt.json')),'HB177 CPU2 smoke stale')
    for tool in ('hb177_postfit_v1.py','hb177_postfit_smoke_v1.py','strict_variant_gate_hb177.py',
                 'strict_variant_gate_standalone.py','strict_variant_gate_hb151.py'):
        inputs.read(inputs.root/'tools'/tool)
    g.need(rec['builder_sha256']==g.sha(inputs.read(inputs.root/'tools/hb177_postfit_v1.py')) and
           smoke['smoke_source_sha256']==g.sha(inputs.read(inputs.root/'tools/hb177_postfit_smoke_v1.py')),'HB177 tool drift')
    inv=s.inventory(raw,small,hashes,g); inv['dataset/dataset-metadata.json']=inputs.read(dataset/'dataset-metadata.json')
    details=dict(kernel_files={n:g.sha(v) for n,v in raw.items()},embedded={n:g.sha(v) for n,v in small.items()},
        archive_sha256=ash,fixture_only=False,standalone_policy=dict(version=VERSION,family=b.SLUG,
        parent_kernel_sha256=g.sha(praw['release.ipynb']),checkpoint_sha256=b.CHECKPOINT_SHA,
        source_stage=source.relative_to(inputs.root).as_posix(),dataset=b.DATASET,
        build_receipt_sha256=g.sha(inputs.read(source/'build-receipt.json'))))
    return meta,inv,details,raw,paths


def context(admission=False):
    """Private clones keep every historical gate's original globals untouched."""
    g=types.ModuleType('_hb177_private_gate'); g.__dict__.update(vars(base))
    old_kernel=base.kernel
    def route(stage,inputs):
        path=stage/'kernel/kernel-metadata.json'
        if path.is_file() and base.parse(path.read_bytes()).get('id')==b.SLUG: return kernel(stage,inputs,g)
        return old_kernel(stage,inputs)
    g.kernel=route
    for name in ('source_binding','review_check'):
        f=getattr(base,name); setattr(g,name,types.FunctionType(f.__code__,g.__dict__,name,f.__defaults__,f.__closure__))
    def checked_kernel(stage,inputs,gate):
        result=kernel(stage,inputs,gate)
        if admission:
            from tools import hb177_release_admission_v1 as approval
            approval.check(inputs.root/result[2]['standalone_policy']['source_stage'],inputs,gate)
        return result
    ns=dict(vars(s)); ns.update(VERSION=VERSION,BP=b.PARENT.relative_to(b.ROOT).as_posix(),BPINS=b.PINS,
                                B=b.SLUG,IDS={b.SLUG},kernel=checked_kernel,__file__=__file__)
    for name in ('parent_material','parent_files','bind','evaluate'):
        f=getattr(s,name); ns[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
    return g,ns


def main():
    argv=sys.argv[1:]; mode=argv.pop(0) if argv and argv[0] in ('pin','bind') else 'check'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    if mode!='pin':
        for name in (('parent','output','builder-receipt','review') if mode=='bind' else ('parent','allow','review')):
            p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fast',action='store_true'); p.add_argument('--check-only',action='store_true')
    a=p.parse_args(argv); g,ns=context(admission=mode=='check')
    try:
        if mode=='pin':
            inputs=g.Inputs(ROOT); new=kernel(g.local_path(a.stage,ROOT),inputs,g); inputs.recheck()
            result=dict(status='PINNED_NOT_ADMITTED',kernel_sha256=g.sha(g.encoded(new[2]['kernel_files'])))
        elif mode=='bind': result=ns['bind'](a.stage,a.parent,a.output,a.builder_receipt,a.review,a.fast,ROOT,g)
        else:
            result=ns['evaluate'](a.stage,a.parent,a.allow,a.review,ROOT,g)
            if result['status']=='PASS' and not a.check_only:
                g.write_receipt(ROOT/'.local/state/gates'/ (b.SLUG.split('/')[1]+'.json'),result)
            result.pop('full_diff',None)
        print(json.dumps(result,indent=2)); return 1 if result['status']=='FAIL' else 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='FAIL',error=str(exc)))); return 1


if __name__=='__main__': sys.exit(main())
