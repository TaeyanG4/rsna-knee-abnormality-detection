"""H-B-156 standalone kill-test 1.0.0. Offline CPU builder; no release actions.

Changelog 1.0.0: strict HB151 v1.0.2 constructor/checkpoint/identity patch.
The inherited hb151_export manifest key is retained to keep the loader unchanged.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import difflib
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hb151_killtest_v102 import need, sha, digest, read, encoded, write, retry, archive_files

VERSION = '1.0.0'
PARENT = ROOT/'artifacts/experiments/HB151_KILLTEST/v1.0.2'
GATE = ROOT/'artifacts/experiments/HB151_KILLTEST/v1.0.2-gate-v120f'
OUT = ROOT/'artifacts/experiments/HB156_KILLTEST/v1.0.0'
SCRATCH = ROOT/'.local/scratch/hb156-killtest/H-B-156'
PACKAGE = ROOT/'.local/scratch/hb156-l2/H-B-156-L2/package'
PROTOCOL = PACKAGE/'stage-convnext-l2-s42-v1/manifest.json'
PROTOCOL_SHA = '788d98ed4b22a970bc4e09e9fcd17a864918c62e4c635fe88380e6d556a2f39c'
RUN = 'stage-convnext-l2-s42-v1/S42'
NAME = 'exp143-v2-e14-e11-e12-bn.pt'
ARCHIVE = 'exp152-assets.zip.bin'
RUNTIME = 'exp152_own0949_release.py'
ARCH = 'convnext_small.fb_in22k_ft_in1k_384'
# Exact recipe metadata in the SHA-authenticated L2 registration (not a display label).
LABEL_SET = 'registered-replacement-soft'
OLD_ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
SLUG = 'taeyangg4/knee-hb156-cnxs-l2-v1'
DATASET = 'taeyangg4/knee-hb156-cnxs-l2-weights-v1'
DECISION = dict(alive_at_least=.950, dead_at_most=.946, otherwise_seed=43,
                baseline_public=.946, baseline_ref='57005140', standalone=True)
PINS = {
    'build-receipt.json': '289aeee90284e82827c7eb8e3009eae25dcb09d4d7b6b7564b4a70ddb64eb436',
    'kernel/release.ipynb': 'd1a557f282fd429b61d0f1b70998d5604741410511773da5cf52f3cd51e3d8ed',
    'kernel/kernel-metadata.json': 'a9f0296f00e9acdd35bb3b7016e5c9e8cc4eeb1dc732fa5c95cc5b9e012f1680',
    'dataset/dataset-metadata.json': '1315737d9366e4c08d696e2c0d4d942a9505af78108cdb5c6a8b439e01bd9ae7',
    'dataset/'+ARCHIVE: 'dd4198a770c65ad1eccb8e0b90f278e31e6c699d2e4432aa30d9a4c48783e312',
}
SOURCE_PINS = {
    'training/src/hb156_model_v1.py': '9f93a04336038da17bf3627ad8e002d70c016224600ae5db845485983538ce88',
    'training/src/exp143_own0949.py': '77aab67cd9abb8a2d90cbe74334f66b23137806c64db40ab548b523bbeecca58',
    'training/src/raptor_common.py': 'bc6db5b3e8cd13b52fbff8c039d3674cdf573f86c1959e22fe9b2fd94255d7dc',
}


def cpu_only():
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    import torch
    need(not torch.cuda.is_initialized(), 'CPU-only process required')
    torch.set_num_threads(2)


def authenticate():
    for rel, pin in PINS.items():
        need(sha(PARENT/rel) == pin, 'parent pin: '+rel)
    for rel in ('kernel/release.ipynb', 'kernel/kernel-metadata.json'):
        need(sha(GATE/rel) == PINS[rel], 'gate copy pin: '+rel)
    need(sha(PROTOCOL) == PROTOCOL_SHA, 'protocol pin')
    need(read(PROTOCOL)['recipe']['labels'] == LABEL_SET, 'registered label-set identity')
    for rel, pin in SOURCE_PINS.items():
        need(sha(PACKAGE/rel) == pin and sha(ROOT/rel) == pin, 'training source pin: '+rel)
    need(sha(PACKAGE/'stage-convnext-l2-s42-v1/decision.json') ==
         'acef812133b5f5d18b5fdbf2de3bbc468c18ee0e1c94c91ac9f88f0fce301136', 'decision pin')
    hashes, small = archive_files(PARENT/'dataset'/ARCHIVE)
    need(hashes == read(PARENT/'build-receipt.json')['files'], 'parent archive members')
    return hashes, small


def replace_once(data, old, new):
    need(data.count(old) == 1, 'patch anchor must occur exactly once: '+repr(old))
    return data.replace(old, new)


def code_bytes(small):
    return {
        RUNTIME: replace_once(small[RUNTIME], ('ARCH = "'+OLD_ARCH+'"').encode(),
                              ('ARCH = "'+ARCH+'"').encode()),
        'public-serving.py': replace_once(small['public-serving.py'],
            b'kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=INPUT_RES)',
            b'kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)'),
    }


def runtime(data):
    module = types.ModuleType('_hb156_cpu_runtime')
    module.__file__ = str(SCRATCH/RUNTIME)
    exec(compile(data, module.__file__, 'exec'), module.__dict__)
    return module


def training_model():
    cpu_only()
    sys.path.insert(0, str(ROOT/'training/src'))
    from hb156_model_v1 import build_model
    return build_model(backbone=ARCH).eval()


@lru_cache(maxsize=1)
def expected_schema():
    model = training_model()
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in sorted(model.state_dict().items())}


def identity(checkpoint, placeholder):
    return dict(sha256=sha(checkpoint), protocol_sha256=PROTOCOL_SHA, run_dir=RUN,
                source='H-B-156-L2-seed42', seed=42, exposure='EXPOSED', clean_oof=False,
                serving_name=NAME, placeholder=placeholder)


def compatibility(checkpoint, binding, rt):
    need(binding['sha256'] == sha(checkpoint), 'checkpoint SHA mismatch')
    expected = identity(checkpoint, binding['placeholder'])
    need(all(binding.get(k) == v for k,v in expected.items()), 'binding identity')
    ck = rt.safe_load(checkpoint)
    rt.own_meta(ck, PROTOCOL_SHA, run_dir=RUN, seed=42)
    need(ck['provenance'].get('seed') == 42 and ck.get('exposure') == 'EXPOSED'
         and ck.get('clean_oof') is False, 'checkpoint exposure/seed')
    need(ck.get('label_set') == LABEL_SET, 'checkpoint registered L2 label-set mismatch')
    marked = any(bool(ck.get(k)) for k in ('hb156_killtest_placeholder', 'hb151_killtest_placeholder', 'placeholder', 'fixture'))
    need(marked == binding['placeholder'], 'placeholder cannot be bound as final export')
    need(rt.state_schema(ck['model']) == expected_schema(), 'checkpoint state schema')
    # Strict-load before stage mutation, including the actual export at final bind.
    model = training_model()
    model.load_state_dict(ck['model'], strict=True)


def placeholder(rt):
    import torch
    model = training_model()
    ck = dict(model=model.state_dict(), arch=ARCH, res=384, inference_res=320,
              lab=rt.TARGETS, contract=rt.CONTRACT, k_eval=94, norm='imagenet',
              window_policy='legacy', oai=False, checkpoint_kind='fixed-swa-bn',
              swa_epochs_1based=[14,11,12], seed=42, label_set=LABEL_SET, exposure='EXPOSED', clean_oof=False,
              provenance=dict(protocol_sha256=PROTOCOL_SHA, run_dir=RUN, seed=42),
              hb156_killtest_placeholder=True)
    path = SCRATCH/'placeholder'/NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    torch.save(ck, temp)
    retry(lambda: os.replace(temp, path))
    bind = identity(path, True)
    write(path.with_suffix('.sha256'), (bind['sha256']+'  '+NAME+'\n').encode())
    write(path.with_suffix('.json'), encoded(bind))
    return path, bind


def manifest(small, binding):
    m = json.loads(small['release-manifest.json'])
    code = code_bytes(small)
    m['protocol_sha256'] = PROTOCOL_SHA
    m['own_state_schema'] = expected_schema()
    m['hb151_export'] = binding
    m['checkpoints'][NAME]['sha256'] = binding['sha256']
    m['asset_sha256'] = {NAME: binding['sha256'], **{n:digest(v) for n,v in code.items()}}
    m['serving_sha256'] = digest(code['public-serving.py'])
    m['dataset'] = DATASET
    # These inherited EXP143 fields are ancestry, not HB156 training evidence.
    m['hb156_provenance'] = dict(version=VERSION, training_sources=SOURCE_PINS,
        inherited_fields_are_parent_ancestry=['average_receipt_sha256','source_final_sha256','source_bindings'])
    return m


def payload(small, binding, archive_sha):
    nb = read(PARENT/'kernel/release.ipynb')
    meta = read(PARENT/'kernel/kernel-metadata.json')
    old = ''.join(nb['cells'][5]['source']).encode()
    changes = {
        'knee-hb151-pl-coat-ckpt-v1': DATASET.split('/')[1],
        PINS['dataset/'+ARCHIVE]: archive_sha,
        digest(small['release-manifest.json']): digest(encoded(manifest(small,binding))),
        digest(small[RUNTIME]): digest(code_bytes(small)[RUNTIME]),
    }
    # The dataset mount appears twice (flat and nested); all other patches once.
    for before, after in changes.items():
        need(old.count(before.encode()) == (2 if before.startswith('knee-') else 1), 'cell5 patch count: '+before)
        old = old.replace(before.encode(), after.encode())
    nb['cells'][5]['source'] = old.decode()
    meta.update(id=SLUG, title=SLUG.split('/')[1], dataset_sources=[DATASET])
    return nb, meta


def dataset_metadata():
    m = read(PARENT/'dataset/dataset-metadata.json')
    m.update(id=DATASET, title=DATASET.split('/')[1],
        description='Private H-B-156 L2 ConvNeXt-S seed42 SWA [14,11,12]; EXPOSED / clean_oof=false. No training data.')
    return m


def scope(nb, meta, hashes, small, original, binding, archive_sha):
    expected_nb, expected_meta = payload(original,binding,archive_sha)
    need(nb == expected_nb and meta == expected_meta, 'forbidden notebook/metadata diff')
    need(small == {**code_bytes(original), 'release-manifest.json':encoded(manifest(original,binding))},
         'forbidden embedded file diff')
    need(hashes[NAME] == binding['sha256'], 'checkpoint archive pin')


def embedded(data):
    result = {}
    for node in ast.walk(ast.parse(data)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith('_B64'):
                    result[target.id] = base64.b64decode(node.value.value, validate=True)
    return result


def full_diff(nb, meta, hashes, small, original, binding):
    parent_nb = read(PARENT/'kernel/release.ipynb')
    pairs = [('notebook-top-level', encoded({k:v for k,v in parent_nb.items() if k!='cells'}),
              encoded({k:v for k,v in nb.items() if k!='cells'}))]
    pairs += [(f'notebook-cell-{i}', encoded(a), encoded(b)) for i,(a,b) in enumerate(zip(parent_nb['cells'],nb['cells']))]
    pairs += [('kernel-metadata.json', (PARENT/'kernel/kernel-metadata.json').read_bytes(), encoded(meta)),
              ('dataset-metadata.json',(PARENT/'dataset/dataset-metadata.json').read_bytes(),encoded(dataset_metadata()))]
    for name in sorted(original):
        pairs.append(('archive/'+name,original[name],small[name]))
        if name.endswith('.py'):
            a,b = embedded(original[name]),embedded(small[name])
            need(a == b, 'embedded base64 drift: '+name)
            pairs += [(f'decoded/{name}/{key}',data,b[key]) for key,data in a.items()]
    pairs.append(('archive/'+NAME+' (binary SHA256)',
        (read(PARENT/'build-receipt.json')['files'][NAME]+'\n').encode(),(hashes[NAME]+'\n').encode()))
    for name,data in [(NAME+'.json',encoded(binding)),(NAME+'.sha256',(hashes[NAME]+'  '+NAME+'\n').encode())]:
        pairs.append(('dataset/'+name,(PARENT/'dataset'/name).read_bytes(),data))
    result = ['ALLOWLIST: constructor ARCH and img_size only; checkpoint/schema/provenance and chained pins; requested dataset/kernel identity.\n',
              'Cell0, cells1-4, execution/import/pool/thread/timing, preprocessing/TTA/K94/head/ranks/CSV: unchanged.\n']
    for name,a,b in pairs:
        result.append(f'\n{name}: {"IDENTICAL" if a==b else "ALLOWED_CHANGE"} {digest(a)} -> {digest(b)}\n')
        if a!=b:
            result.extend(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),
                          fromfile='HB151/'+name,tofile='HB156/'+name))
    return ''.join(result).encode()


def build(checkpoint, binding, original):
    need(not (OUT/'RELEASED.json').exists(), 'frozen stage; create new version')
    compatibility(checkpoint,binding,runtime(code_bytes(original)[RUNTIME]))
    if (OUT/'build-receipt.json').exists():
        prior = read(OUT/'build-receipt.json')
        need(prior['placeholder'], 'bound stage is immutable; create new version')
    outzip = OUT/'dataset'/ARCHIVE
    outzip.parent.mkdir(parents=True,exist_ok=True)
    temp = outzip.with_suffix('.partial')
    small = {**code_bytes(original),'release-manifest.json':encoded(manifest(original,binding))}
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as z:
        for name in sorted([NAME,*small]):
            with z.open(zipfile.ZipInfo(name,date_time=(2026,10,15,0,0,0)),'w',force_zip64=True) as dest:
                if name == NAME:
                    with Path(checkpoint).open('rb') as src: shutil.copyfileobj(src,dest,4<<20)
                else: dest.write(small[name])
    retry(lambda:os.replace(temp,outzip))
    archive_sha = sha(outzip)
    nb,meta = payload(original,binding,archive_sha)
    hashes,actual = archive_files(outzip)
    scope(nb,meta,hashes,actual,original,binding,archive_sha)
    for rel,data in {'kernel/release.ipynb':encoded(nb),'kernel/kernel-metadata.json':encoded(meta),
        'dataset/dataset-metadata.json':encoded(dataset_metadata()),'export-binding.json':encoded(binding),
        'dataset/'+NAME+'.json':encoded(binding),'dataset/'+NAME+'.sha256':(binding['sha256']+'  '+NAME+'\n').encode(),
        'full-diff-vs-parent.txt':full_diff(nb,meta,hashes,small,original,binding)}.items(): write(OUT/rel,data)
    artifacts = ['kernel/release.ipynb','kernel/kernel-metadata.json','dataset/dataset-metadata.json',
                 'dataset/'+ARCHIVE,'export-binding.json','full-diff-vs-parent.txt','dataset/'+NAME+'.json','dataset/'+NAME+'.sha256']
    rec = dict(version=VERSION, builder_sha256=sha(__file__), parent_pins=PINS, source_pins=SOURCE_PINS,
        artifacts={n:sha(OUT/n) for n in artifacts}, files=hashes, decision=DECISION,
        placeholder=binding['placeholder'], exposure='EXPOSED',clean_oof=False,qualified_for_submission=False,
        status='PLACEHOLDER_CPU_ONLY' if binding['placeholder'] else 'BOUND_NOT_REVIEWED')
    write(OUT/'build-receipt.json',encoded(rec))
    return validate()


def prepare():
    cpu_only()
    _,original = authenticate()
    path,binding = placeholder(runtime(code_bytes(original)[RUNTIME]))
    return build(path,binding,original)


def bind(export):
    cpu_only()
    _,original = authenticate()
    export = Path(export)
    need(export.is_file() and export.name == NAME, 'missing/wrong-name export')
    pin = export.with_suffix('.sha256')
    need(pin.is_file(), 'missing export pin')
    tokens = pin.read_text(encoding='utf-8').split()
    need(len(tokens)==2 and re.fullmatch('[0-9a-f]{64}',tokens[0]) and tokens[1]==NAME, 'invalid export pin')
    need(sha(export)==tokens[0], 'export SHA mismatch')
    sidecar = export.with_suffix('.json')
    need(sidecar.is_file(), 'missing export provenance')
    m = read(sidecar)
    binding = identity(export,False)
    for key in ('sha256','protocol_sha256','exposure','clean_oof','serving_name'):
        need(key in m and m[key] == binding[key], 'export provenance mismatch: '+key)
    need(not any(m.get(k) for k in ('placeholder','fixture')), 'placeholder export sidecar')
    binding.update(native_sidecars={pin.name:sha(pin)},native_json_sha256=sha(sidecar))
    return build(export,binding,original)


def validate():
    cpu_only()
    _,original = authenticate()
    rec = read(OUT/'build-receipt.json')
    need(rec['builder_sha256']==sha(__file__) and rec['parent_pins']==PINS and rec['source_pins']==SOURCE_PINS,
         'builder/source registration drift')
    need(rec['decision']==DECISION and rec['exposure']=='EXPOSED' and rec['clean_oof'] is False
         and rec['qualified_for_submission'] is False, 'decision/exposure/admission drift')
    for rel,pin in rec['artifacts'].items(): need(sha(OUT/rel)==pin, 'artifact pin: '+rel)
    hashes,small = archive_files(OUT/'dataset'/ARCHIVE)
    binding = read(OUT/'export-binding.json')
    fixed = dict(protocol_sha256=PROTOCOL_SHA,run_dir=RUN,source='H-B-156-L2-seed42',seed=42,
                 exposure='EXPOSED',clean_oof=False,serving_name=NAME)
    need(all(binding.get(k)==v for k,v in fixed.items()) and type(binding.get('placeholder')) is bool,
         'binding identity drift')
    need(rec['placeholder'] == binding['placeholder'] and hashes == rec['files'], 'receipt binding drift')
    nb,meta = read(OUT/'kernel/release.ipynb'),read(OUT/'kernel/kernel-metadata.json')
    scope(nb,meta,hashes,small,original,binding,sha(OUT/'dataset'/ARCHIVE))
    need(read(OUT/'dataset/dataset-metadata.json')==dataset_metadata(), 'dataset metadata diff')
    need(read(OUT/'dataset'/(NAME+'.json'))==binding, 'binding sidecar diff')
    need((OUT/'dataset'/(NAME+'.sha256')).read_text().split()==[hashes[NAME],NAME], 'SHA sidecar diff')
    need((OUT/'full-diff-vs-parent.txt').read_bytes()==full_diff(nb,meta,hashes,small,original,binding), 'diff drift')
    return dict(status='PASS_CPU_SCOPE',placeholder=binding['placeholder'],archive_members_checked=4,
                qualified_for_submission=False)


def smoke():
    validate()
    from tools.hb156_killtest_v1_smoke import run
    report = run()
    report.update(build_receipt_sha256=sha(OUT/'build-receipt.json'),smoke_source_sha256=sha(ROOT/'tools/hb156_killtest_v1_smoke.py'))
    write(OUT/'cpu-smoke.json',encoded(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','bind','validate','cpu-smoke'])
    parser.add_argument('--export',type=Path)
    args = parser.parse_args()
    if args.command=='bind': need(args.export is not None, 'bind requires --export')
    result = {'prepare':prepare,'bind':lambda:bind(args.export),'validate':validate,'cpu-smoke':smoke}[args.command]()
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
