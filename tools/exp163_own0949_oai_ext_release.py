"""EXP163 create-only release: fixed SWA/BN, standalone and three equal rank arms.

CPU average/stage/push-plan only; no credentials, uploads or submission execution.
Serving notebooks reuse authenticated EXP155 v1.2 / EXP152 implementations.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import re
from pathlib import Path
import shutil
import sys
import zipfile

TOOL = Path(__file__).resolve()
ROOT = TOOL.parents[1]
VERSION = '1.1.0'
PROTOCOL = ROOT / 'artifacts/experiments/EXP163_OWN0949_OAI_EXT/protocol-v1/manifest.json'
RUN = 'artifacts/experiments/EXP163_OWN0949_OAI_EXT/S42'
OWN_NAME = 'exp163-e14-e11-e12-oai-ext-bn.pt'
MEMBER_NAME = 'exp143-v2-e14-e11-e12-bn.pt'
ASSET_DATASET = 'taeyangg4/knee-exp163-own0949-oai-ext-assets'
KERNELS = {'own': 'knee-exp163-own0949-oai-ext', 'blend': 'knee-exp163-own0949-oai-ext-blend3'}
ADAPTER_NAME = 'exp155_own0949_oai_release_v12.py'
ADAPTER_SHA = 'f6fc591e29fd3090e796a48ac65d09565eb5de9c029a00e3969b1ced5a17ca52'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

adapter_path = TOOL.with_name(ADAPTER_NAME)
if sha(adapter_path) != ADAPTER_SHA:
    raise ValueError('Pinned EXP155 release implementation changed')
_spec = importlib.util.spec_from_file_location(__name__ + '_exp155', adapter_path)
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
_shared = _base._shared
for _module in (_base, _shared):
    for _name in ('VERSION', 'TOOL', 'ROOT', 'PROTOCOL', 'RUN', 'OWN_NAME', 'ASSET_DATASET', 'KERNELS'):
        setattr(_module, _name, globals()[_name])
for _name in ('need', 'read_json', 'write_json', 'write_text', 'fresh_dir', 'safe_load',
              'state_schema', 'average_states', 'merge_replayed_bn', 'public_source', 'serving_source',
              'ARCH', 'CONTRACT', 'TARGETS', 'EPOCHS', 'PUBLIC_NAME', 'PUBLIC_SHA', 'PUBLIC_DATASET',
              'SOURCE_SHA', 'TOL', 'POLICY', 'BASE_NAME', 'BASE_SHA256', 'BASE_PATH', 'percentile_average'):
    globals()[_name] = getattr(_base, _name)
_original_load = _base._original_load_module

def load_module(name, path):
    path = Path(path)
    if path.name in ('exp143_v2.py', 'exp155_own0949_oai_v11.py'):
        path = ROOT / 'tools/exp163_own0949_oai_ext.py'
    return _original_load(name, path)

def own_meta(ck, protocol_sha, run_dir=RUN):
    return _base.own_meta(ck, protocol_sha, run_dir)

_shared.own_meta = own_meta
# Adapt only experiment identity literals in the pinned CPU SWA/BN reconstruction.
_average_source = inspect.getsource(_base.average).replace('exp155_own0949_oai_v11.py', 'exp163_own0949_oai_ext.py')
_average_source = _average_source.replace('exp155-average-v1.1', 'exp163-average-v1').replace('EXP155 S42v1_1', 'EXP163 S42')
exec(compile(_average_source, str(TOOL) + ':average', 'exec'), globals())
_private_source = inspect.getsource(_base.assert_private_checkpoint).replace('S42v1_1', 'S42')
_private_source = _private_source.replace('l2-dread75-aquila25-plus-oai-three-target-b1', 'l2-dread75-aquila25-plus-oai-eight-target-ext-v1.1')
exec(compile(_private_source, str(TOOL) + ':privacy', 'exec'), globals())

def output_values(arms, variant, public_rank):
    import numpy as np
    need(variant in KERNELS and len(arms) == (1 if variant == 'own' else 3), 'Wrong arm set')
    shape = np.asarray(arms[0]).shape
    need(len(shape) == 2 and shape[0] > 0 and shape[1] == 12, 'Invalid prediction shape')
    need(all(np.asarray(a).shape == shape and np.isfinite(a).all() and
             (np.asarray(a) >= 0).all() and (np.asarray(a) <= 1).all() for a in arms), 'Invalid probabilities')
    if variant == 'own':
        return public_rank(np.clip(arms[0], 0, 1))
    ranks = [np.asarray([percentile_average(np.asarray(a)[:, j].tolist()) for j in range(12)]).T for a in arms]
    return sum(ranks) / 3.0

def bind(ns, manifest, files, variant, work):
    _shared.bind(ns, manifest, files, variant, work)
    names = [OWN_NAME] + ([PUBLIC_NAME, MEMBER_NAME] if variant == 'blend' else [])
    need(set(files) == set(names), 'Incomplete registered arm paths')
    ns['_E152'] = sys.modules[__name__]
    ns['ARMS'] = [dict(file=n, arch=ARCH, res=384, w=1.0 / len(names)) for n in names]

def checked_model(ns, path, arch, res, device):
    if Path(path).name == MEMBER_NAME:
        ck = safe_load(path)
        need(ck.get('provenance', {}).get('protocol_sha256') == ns['_E152_MANIFEST']['member_protocol_sha256'],
             'EXP143 protocol differs')
        need(ck.get('oai') is False and ck.get('checkpoint_kind') == 'fixed-swa-bn' and
             ck.get('swa_epochs_1based') == EPOCHS, 'EXP143 fixed SWA/BN metadata differs')
    return _shared.checked_model(ns, path, arch, res, device)

input_record, finish_receipt = _shared.input_record, _shared.finish_receipt

def save_raw(work, ids, arms, inputs, manifest, variant):
    import numpy as np
    need(len(ids) == len(set(ids)) and set(ids) == set(inputs), 'Incomplete cohort/input diagnostics')
    output_values(arms, variant, lambda a: a)
    names = ['own'] + (['public', 'exp143'] if variant == 'blend' else [])
    arrays = dict(ids=np.asarray(ids), targets=np.asarray(TARGETS))
    arrays.update(zip(names, arms))
    with (Path(work) / 'raw.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    write_json(Path(work) / 'inputs.json', inputs)
    write_json(Path(work) / 'runtime.json', _shared.runtime_info())
    write_json(Path(work) / 'binding.json', dict(manifest_sha256=_shared.jhash(manifest), variant=variant,
               checkpoint_sha256={n: manifest['checkpoints'][n]['sha256'] for n in
                   ([OWN_NAME] + ([PUBLIC_NAME, MEMBER_NAME] if variant == 'blend' else []))},
               source_sha256=SOURCE_SHA, serving_sha256=manifest['serving_sha256'], tolerance=TOL))

def verify_assets(path, manifest):
    _base._original_verify_assets(path, manifest)
    need(manifest.get('schema') == 'exp163-release-v1', 'EXP163 release schema mismatch')
    expected = {OWN_NAME, MEMBER_NAME, TOOL.name, ADAPTER_NAME, BASE_NAME, 'public-serving.py'}
    need(set(manifest['asset_sha256']) == expected, 'Unexpected release payload')
    need(manifest['asset_sha256'][ADAPTER_NAME] == ADAPTER_SHA and
         manifest['asset_sha256'][BASE_NAME] == BASE_SHA256, 'Shared release source mismatch')

def notebook(variant, dataset, archive_sha, manifest_sha, module_sha, credits):
    nb, _, _ = _base._original_notebook(variant, dataset, archive_sha, manifest_sha, module_sha, credits)
    code = nb['cells'][-1]['source'].replace('exp152-assets', 'exp163-assets')
    code = _shared.replace_once(code, "'exp152_own0949_release.py', 'public-serving.py'",
        repr(MEMBER_NAME) + ", 'exp163_own0949_oai_ext_release.py', 'exp155_own0949_oai_release_v12.py', 'exp152_own0949_release.py', 'public-serving.py'")
    code = code.replace("_hash(_assets/'exp152_own0949_release.py')", "_hash(_assets/'exp163_own0949_oai_ext_release.py')")
    code = code.replace('import exp152_own0949_release as _e152', 'import exp163_own0949_oai_ext_release as _e152')
    if variant == 'blend':
        code = code.replace('_work = Path(', "_files[" + repr(MEMBER_NAME) + "] = _assets/" + repr(MEMBER_NAME) + "\n_work = Path(")
    compile(code, 'exp163-notebook', 'exec')
    title = 'EXP163 extended OAI ' + ('equal three-member rank blend' if variant == 'blend' else 'standalone')
    description = ('Fixed epochs [14,11,12] with authenticated mixed train-only BN. Gold58 excluded. '
        'K94, native320, anatomical mirror, fp16 chunk16. Private; no raw OAI data or identifiers. '
        'Every submission requires Taeyang instruction. ')
    if variant == 'blend':
        description += 'Public 0.949, EXP143 v2 and EXP163 each have weight 1/3 in average-tie percentile ranks; no final re-ranking. '
    nb['cells'][0]['source'] = '# ' + title + '\n\n' + description
    nb['cells'][-1]['source'] = code
    return nb, title, description

def stage(args):
    receipt = read_json(Path(args.average) / 'average-receipt.json')
    need(receipt.get('schema') == 'exp163-average-v1' and receipt.get('oai') is True and
         receipt.get('status') == 'VERIFIED_FIXED_SWA_BN' and receipt.get('epochs_1based') == EPOCHS,
         'EXP163 average receipt invalid')
    need(receipt.get('shared_serving_sha256') == BASE_SHA256, 'Average serving binding differs')
    checkpoint = Path(args.average) / OWN_NAME
    need(sha(checkpoint) == receipt['checkpoint_sha256'], 'Average checkpoint SHA mismatch')
    ck = safe_load(checkpoint)
    own_meta(ck, receipt['protocol_sha256'])
    assert_private_checkpoint(ck)
    need(state_schema(ck['model']) == receipt['state_schema'], 'Own state schema differs')
    member = Path(args.exp143_checkpoint)
    member_receipt = read_json(member.parent / 'average-receipt.json')
    need(member.name == MEMBER_NAME and member_receipt.get('status') == 'VERIFIED_FIXED_SWA_BN' and
         member_receipt.get('epochs_1based') == EPOCHS and member_receipt.get('checkpoint_sha256') == sha(member),
         'EXP143 requires its authenticated average receipt')
    mck = safe_load(member)
    expected_protocol = ROOT / 'artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json'
    member_protocol_sha = sha(expected_protocol)
    need(member_receipt.get('protocol_sha256') == member_protocol_sha, 'EXP143 receipt protocol differs')
    _shared.need(mck.get('oai') is False and mck.get('provenance', {}).get('protocol_sha256') == member_protocol_sha and
                 mck.get('checkpoint_kind') == 'fixed-swa-bn' and mck.get('swa_epochs_1based') == EPOCHS,
                 'EXP143 member metadata differs')
    need(state_schema(mck['model']) == receipt['state_schema'], 'EXP143 architecture schema differs')
    import re
    need(re.fullmatch(r'[a-z0-9_-]+/[a-z0-9_-]+', args.dataset), 'Invalid dataset ID')
    source, credits = public_source()
    out = fresh_dir(args.out)
    assets = fresh_dir(out / 'assets')
    # EXP143 payload is deliberately reduced to inference fields so training IDs cannot be exported.
    permitted = {'model', 'arch', 'res', 'inference_res', 'lab', 'contract', 'k_eval', 'norm', 'window_policy',
                 'oai', 'checkpoint_kind', 'swa_epochs_1based', 'seed'}
    import torch
    with (assets / MEMBER_NAME).open('xb') as stream:
        torch.save({**{k: v for k, v in mck.items() if k in permitted},
                    'provenance': {'protocol_sha256': member_protocol_sha}}, stream)
    for src, name in ((checkpoint, OWN_NAME), (TOOL, TOOL.name), (adapter_path, ADAPTER_NAME), (BASE_PATH, BASE_NAME)):
        shutil.copyfile(src, assets / name)
    write_text(assets / 'public-serving.py', serving_source(source))
    asset_names = (OWN_NAME, MEMBER_NAME, TOOL.name, ADAPTER_NAME, BASE_NAME, 'public-serving.py')
    manifest = dict(schema='exp163-release-v1', version=VERSION, protocol_sha256=receipt['protocol_sha256'],
        member_protocol_sha256=member_protocol_sha, epochs_1based=EPOCHS, public_source_sha256=SOURCE_SHA,
        serving_sha256=sha(assets / 'public-serving.py'), checkpoints={
            OWN_NAME: dict(sha256=sha(assets / OWN_NAME), oai=True),
            MEMBER_NAME: dict(sha256=sha(assets / MEMBER_NAME), oai=False),
            PUBLIC_NAME: dict(sha256=PUBLIC_SHA, oai=True)}, own_state_schema=receipt['state_schema'],
        asset_sha256={n: sha(assets / n) for n in asset_names}, tolerance=TOL, policy=POLICY, dataset=args.dataset,
        ranks=dict(own='public ordinal rank', blend='1/3 each average-tie percentile rank; no final rerank'),
        source_member_sha256=sha(member), source_member_receipt_sha256=sha(member.parent / 'average-receipt.json'),
        average_receipt_sha256=sha(Path(args.average) / 'average-receipt.json'),
        approval_required={v: 'Taeyang instruction for each submission' for v in KERNELS})
    write_json(assets / 'release-manifest.json', manifest)
    verify_assets(assets, manifest)
    dataset = fresh_dir(out / 'dataset')
    archive = dataset / 'exp163-assets.zip.bin'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(assets.iterdir()):
            bundle.write(path, path.name)
    write_json(dataset / 'dataset-metadata.json', dict(title=args.dataset.split('/')[1], id=args.dataset,
        licenses=[{'name': 'other'}], description='Private EXP163 and EXP143 inference weights, authenticated serving. No training IDs or raw external data.'))
    kernels = {}
    for variant in KERNELS:
        dest = fresh_dir(out / variant)
        nb, title, description = notebook(variant, args.dataset, sha(archive), sha(assets / 'release-manifest.json'), sha(TOOL), credits)
        write_json(dest / 'release.ipynb', nb)
        write_json(dest / 'kernel-metadata.json', _shared.kernel_metadata(variant, args.dataset))
        write_text(dest / 'DESCRIPTION.md', '# ' + title + '\n\n' + description + '\n')
        kernels[variant] = {n: sha(dest / n) for n in ('release.ipynb', 'kernel-metadata.json', 'DESCRIPTION.md')}
    report = dict(status='STAGED_NOT_UPLOADED', version=VERSION, archive_sha256=sha(archive),
        manifest_sha256=sha(assets / 'release-manifest.json'), dataset_metadata_sha256=sha(dataset / 'dataset-metadata.json'),
        kernels=kernels, dataset=args.dataset, parity='PENDING: three-member local/T4 parity required before submission')
    write_json(out / 'build-receipt.json', report)
    return dict(status=report['status'], out=str(out), archive_sha256=report['archive_sha256'])

def checked_stage(stage_dir):
    path = Path(stage_dir).resolve()
    report = read_json(path / 'build-receipt.json')
    need(report.get('version') == VERSION, 'Stage version differs')
    for name, key in (('assets/release-manifest.json', 'manifest_sha256'), ('dataset/exp163-assets.zip.bin', 'archive_sha256'),
                      ('dataset/dataset-metadata.json', 'dataset_metadata_sha256')):
        need(sha(path / name) == report[key], 'Stage hash mismatch')
    manifest = read_json(path / 'assets/release-manifest.json')
    verify_assets(path / 'assets', manifest)
    need(sha(TOOL) == manifest['asset_sha256'][TOOL.name], 'Executing release differs')
    for variant, files in report['kernels'].items():
        for name, digest in files.items():
            need(sha(path / variant / name) == digest, 'Kernel hash mismatch')
    return path, report, manifest

def push_plan(args):
    result = _shared.push_plan(args)
    path = Path(args.out)
    text = path.read_text(encoding='utf-8').replace('EXP152', 'EXP163').replace('H-B-63-exp152-', 'EXP163-')
    text = text.replace('Run compare-parity against both downloaded output folders.',
        'Run a separately reviewed THREE-ARM local/T4 parity check on raw.npz (own/public/exp143); EXP152 two-arm compare-parity is unsupported.')
    path.write_text(text, encoding='utf-8', newline='\n')
    result['sha256'] = sha(path)
    return result

_shared.checked_stage = checked_stage

def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version', action='version', version=VERSION)
    sub = ap.add_subparsers(dest='command', required=True)
    a = sub.add_parser('average'); a.add_argument('--protocol', default=str(PROTOCOL)); a.add_argument('--checkpoint'); a.add_argument('--out', required=True)
    a = sub.add_parser('stage'); a.add_argument('--average', required=True); a.add_argument('--exp143-checkpoint', required=True); a.add_argument('--out', required=True); a.add_argument('--dataset', default=ASSET_DATASET)
    a = sub.add_parser('push-plan'); a.add_argument('--stage', required=True); a.add_argument('--out', required=True)
    return ap

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = globals()[args.command.replace('-', '_')](args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED', error=str(error))))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0

if __name__ == '__main__':
    sys.exit(main())
