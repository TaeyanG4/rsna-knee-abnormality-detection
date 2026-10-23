"""EXP120: RadImageNet DenseNet121 on the frozen EXP113 five-fold recipe.

CPU registration:  python tools/exp120_densenet_local.py prepare
GPU queue JSON:   python tools/exp120_densenet_local.py queue-json --after TAG
After a recorded F0 pass only: queue-json --folds 1,2,3,4 --screen-pass PATH --after TAG
CPU validation:   python tools/exp120_densenet_local.py validate --fold 0
CPU smoke:        python tools/exp120_densenet_local.py smoke --out NEW_DIR
Pooled OOF:       python tools/exp120_densenet_local.py assemble

No command in this tool starts a queued GPU job. prepare and validate explicitly refuse
a CUDA context. Source/weight hashes and the seed-42 initial state are bound at prepare.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp113_radr50_local as base
from tools import exp099_kfold as k99
from tools.exp113_radr50_local import (  # runner guard's shared, unchanged contract helpers
    worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,
    CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership,
    pos_weight, contract, UID, T, DREAD, KEYS_FILE, FOLDS, EXTRA_EVAL, RUN_FILES,
    fold_contract, state_hashes, mismatched_keys, dump, train_python, within,
    display_path, _set_flag,
)

SCHEMA = 'exp120-densenet-local-v1'
PROTOCOL = 'artifacts/experiments/EXP120_DENSENET_LOCAL_PROTOCOL_V1'
BASE_PROTOCOL = base.PROTOCOL
TOOL = 'tools/exp120_densenet_local.py'
RUNNER = 'training/runners/exp120_densenet_local.py'
WEIGHTS_LOCAL = 'storage/weights/lab-rasool__radimagenet/DenseNet121.pt'
WEIGHTS_PROVENANCE = 'storage/weights/lab-rasool__radimagenet/provenance.json'
WEIGHTS_README = 'storage/weights/lab-rasool__radimagenet/README.md'
WEIGHTS_SHA256 = '5011b427acf483b3c05d9d4ac6ea4ff0baf5714c23937c0fcdc7c446226d0a14'
ARCH = 'densenet121'
SEED = 42
TAG_PREFIX = 'EXP120_DENSENET_LOCAL'
OOF_DIR = 'artifacts/experiments/EXP120_DENSENET_LOCAL_OOF'
QUEUE_FILE = PROTOCOL + '/queue-entries.json'
SCREEN_FILE = PROTOCOL + '/f0_screen_pass.json'
LOCAL_WORKERS = 12
LABEL_SET = 'dread'


def tag_for(fold, mode='full'):
    if int(fold) not in FOLDS or mode not in ('full', 'smoke'):
        raise ValueError('invalid fold or mode')
    return f'{TAG_PREFIX}_F{int(fold)}' + ('_SMOKE' if mode == 'smoke' else '')


def manifest_path(root=ROOT):
    return Path(root) / PROTOCOL / 'manifest.json'


def map_densenet_state(state, target_keys):
    """Map checkpoint torchvision features to timm; report every unmapped key.

    The checkpoint pickle has 725 tensor keys under backbone.0.* (the features
    Sequential). timm densenet121 may name the same trunk either features.* or
    directly conv0/denseblock*. We select only an exact target-key match and
    reject missing, unexpected or duplicate keys. No classifier weights exist.
    """
    target = set(target_keys)
    mapped, unmapped, duplicate = {}, [], []
    for source, value in state.items():
        m = re.fullmatch(r'backbone\.0\.(.+)', source)
        if not m:
            unmapped.append(source)
            continue
        rest = m.group(1)
        candidates = ('features.' + rest, rest)
        matches = [k for k in candidates if k in target]
        if len(matches) != 1:
            unmapped.append(source)
            continue
        if matches[0] in mapped:
            duplicate.append(source)
        mapped[matches[0]] = value
    report = dict(source_keys=len(state), target_keys=len(target), mapped_keys=len(mapped),
                  unmapped_keys=sorted(unmapped), duplicate_keys=sorted(duplicate),
                  missing_keys=sorted(target - set(mapped)), unexpected_keys=sorted(set(mapped) - target),
                  key_rule='backbone.0.<rest> -> exact timm <rest> or features.<rest>')
    return mapped, report


def build_densenet_model(arch, weights_path, expected_sha256=WEIGHTS_SHA256):
    """CPU build + strict checkpoint load; the same RaptorClassifier head as EXP113."""
    if arch != ARCH:
        raise ValueError(f'expected {ARCH}, got {arch}')
    import torch
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_common import build_model
    from raptor_radimagenet import strict_load_backbone
    weights_path = Path(weights_path)
    digest = sha256(weights_path)
    if digest != expected_sha256:
        raise ValueError(f'DenseNet121 weight hash mismatch: {digest}')
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise TypeError('DenseNet121 checkpoint must be a state dict')
    model = build_model(arch, pretrained=False)
    mapped, report = map_densenet_state(state, model.backbone.state_dict())
    if report['unmapped_keys'] or report['duplicate_keys'] or report['missing_keys'] or report['unexpected_keys']:
        raise RuntimeError('DenseNet121 key map refused: ' + json.dumps(report))
    strict = strict_load_backbone(model.backbone, mapped)
    report.update(strict)
    report.update(weights_sha256=digest, weights_file=weights_path.name)
    return model, report


def _preflight(root):
    root = Path(root)
    bp_path = root / BASE_PROTOCOL / 'manifest.json'
    bp = read(bp_path)
    if bp.get('schema') != base.SCHEMA or bp['seed'] != SEED or bp['label_set']['name'] != LABEL_SET:
        raise ValueError('unexpected EXP113 base protocol')
    for name, digest in bp['source_bindings'].items():
        if sha256(relpath(root, name)) != digest:
            raise ValueError('EXP113 binding changed: ' + name)
    provenance = read(root / WEIGHTS_PROVENANCE)
    if provenance['files']['DenseNet121.pt']['sha256'] != WEIGHTS_SHA256 or sha256(root / WEIGHTS_LOCAL) != WEIGHTS_SHA256:
        raise ValueError('DenseNet121 checkpoint differs from provenance')
    if sha256(root / WEIGHTS_README) != provenance['files']['README.md']['sha256']:
        raise ValueError('RadImageNet README differs from provenance')
    if 'Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])' not in (root / WEIGHTS_README).read_text(encoding='utf-8'):
        raise ValueError('README no longer specifies ImageNet normalization')
    return bp, sha256(bp_path)


def prepare(root=ROOT):
    import numpy as np
    import torch
    import timm
    sys.path.insert(0, str(root / 'training' / 'src'))
    from raptor_common import seed_all
    root = Path(root)
    dest = root / PROTOCOL
    if manifest_path(root).exists() or (dest.exists() and any(p.name not in ('README.md', 'checkpoint-key-report.json') for p in dest.iterdir())):
        raise SystemExit('Refusing to overwrite EXP120 registration')
    bp, base_sha = _preflight(root)
    common = {**bp['common_args'], 'arch': ARCH, 'norm': 'imagenet'}
    if common['k'] != 12 or common['k_eval'] != 24 or common['epochs'] != common['fixed_epoch'] or common['epochs'] != 20:
        raise ValueError('EXP113 recipe drifted')
    worker_init(0)
    seed_all(SEED)
    model, report = build_densenet_model(ARCH, root / WEIGHTS_LOCAL)
    if torch.cuda.is_initialized():
        raise RuntimeError('prepare initialized CUDA')
    initial, keys = state_hashes(model)
    backbone_keys = sum(k.startswith('backbone.') for k in keys)
    if backbone_keys != report['mapped_keys']:
        raise ValueError('backbone initial state key count differs from mapped checkpoint')
    folds = {}
    labels, sem = k99._load_labels(root / DREAD)
    if sem != bp['label_set']['semantic_sha256']:
        raise ValueError('Dread semantic hash differs from EXP113')
    for fold in FOLDS:
        old = bp['folds'][str(fold)]
        ordered = read(root / old['ordered_ids']['path'])
        if not np.array_equal(pos_weight(labels, ordered['train']), np.asarray(old['pos_weight'], np.float32)):
            raise ValueError(f'fold {fold} pos_weight differs from Dread')
        if any(old[k] != v for k, v in fold_contract(len(ordered['train']), common).items()):
            raise ValueError(f'fold {fold} batch contract differs from EXP113')
        folds[str(fold)] = {**copy.deepcopy(old), 'tag': tag_for(fold)}
    dest.mkdir(parents=True, exist_ok=True)
    keys_file = PROTOCOL + '/' + KEYS_FILE
    dump(root / keys_file, dict(schema='exp120-initial-state-keys-v1', arch=ARCH, seed=SEED,
                               combined_sha256=initial, keys=keys, init_weights_sha256=WEIGHTS_SHA256))
    bindings = {**bp['source_bindings'], BASE_PROTOCOL + '/manifest.json': base_sha,
                TOOL: sha256(root / TOOL), RUNNER: sha256(root / RUNNER), keys_file: sha256(root / keys_file),
                WEIGHTS_PROVENANCE: sha256(root / WEIGHTS_PROVENANCE), WEIGHTS_README: sha256(root / WEIGHTS_README)}
    manifest = copy.deepcopy(bp)
    manifest.update(schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), plan_item='H-B-49',
                    status='REGISTERED_NO_EXECUTION', source_bindings=bindings, common_args=common,
                    initial_state_sha256=initial,
                    initial_state=dict(keys_file=keys_file, keys=len(keys), backbone_keys=backbone_keys, computed_on='cpu',
                                       seed=SEED, arch=ARCH, torch=torch.__version__, timm=timm.__version__,
                                       rule='worker_init(0), seed_all(42), build_model(densenet121, pretrained=False), strict DenseNet121 load; sorted state_dict sha256',
                                       enforcement='exact seed-42 state and local runtime; no remote relax'),
                    folds=folds,
                    tags=dict(prefix=TAG_PREFIX, pattern=TAG_PREFIX + '_F<K>[_SMOKE]', oof_dir=OOF_DIR),
                    init_weights=dict(name='RadImageNet DenseNet121', file='DenseNet121.pt', sha256=WEIGHTS_SHA256,
                                      bytes=(root / WEIGHTS_LOCAL).stat().st_size, local=WEIGHTS_LOCAL, licence='MIT (model card)',
                                      provenance=WEIGHTS_PROVENANCE, readme=WEIGHTS_README, load=report),
                    input={**bp.get('input', {}), 'norm':'imagenet',
                           'rule':'within-slot-valid centres; centre cache plane repeated x3; ImageNet mean/std per RadImageNet README'},
                    recipe={**bp['recipe'], 'exp120_change':'DenseNet121 trunk and README ImageNet mean/std; all EXP113 folds, labels, windows, schedule, K, seed and head fixed'},
                    evaluation={**bp['evaluation'], 'oof_dir': OOF_DIR, 'screen':'F0 first; F1-F4 require recorded screen PASS'},
                    local={**bp['local'], 'runner':RUNNER, 'guard':TOOL, 'weights':WEIGHTS_LOCAL,
                           'queue':'F0 default; F1-F4 only with --screen-pass recorded JSON',
                           'estimate':'F0 20-35 min; five folds 1.7-3 h on 4070 Ti SUPER, unbenchmarked'},
                    derived_from=dict(path=BASE_PROTOCOL + '/manifest.json', sha256=base_sha,
                                      changed_args={'arch':['resnet50',ARCH], 'norm':['radimagenet','imagenet']}))
    dump(manifest_path(root), manifest)
    print(json.dumps(dict(status='REGISTERED_NO_EXECUTION', manifest_sha256=sha256(manifest_path(root)),
                          initial_state_sha256=initial, key_map=report, cuda_initialized=False), indent=2))
    return manifest


def validate_arguments(args, root):
    root = Path(root)
    p_path = Path(args.protocol)
    if p_path.resolve() != manifest_path(root).resolve():
        raise ValueError('wrong protocol')
    p = read(p_path)
    if p.get('schema') != SCHEMA or p.get('derived_from', {}).get('sha256') != sha256(root / BASE_PROTOCOL / 'manifest.json'):
        raise ValueError('not the EXP120 registration')
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('local protocol refuses KNEE_REMOTE_RELAX_ENV')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('training source changed: ' + name)
    smoke = args.mode == 'smoke'
    for key, expected in p['common_args'].items():
        actual = getattr(args, key)
        if actual != expected and not (smoke and key in p['smoke_allowed'] and actual in p['smoke_allowed'][key]):
            raise ValueError('frozen argument changed: ' + key)
    if args.seed != SEED or str(args.fold) not in p['folds']:
        raise ValueError('seed or fold changed')
    if not p['resource_args']['workers']['min'] <= args.workers <= p['resource_args']['workers']['max']:
        raise ValueError('workers outside range')
    fold = p['folds'][str(args.fold)]
    if args.tag != tag_for(args.fold, args.mode):
        raise ValueError('tag/fold/mode mismatch')
    for key in ('corpus', 'labels', 'eval_labels'):
        if Path(getattr(args,key)).resolve() != relpath(root,p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    if Path(args.init_weights).resolve() != (root / WEIGHTS_LOCAL).resolve() or sha256(args.init_weights) != WEIGHTS_SHA256:
        raise ValueError('init weights differ from registered DenseNet121')
    for key, val in {**THREAD_ENV, 'HF_HUB_OFFLINE':'1'}.items():
        if os.environ.get(key) != val:
            raise ValueError('registered environment required: ' + key)
    for key in ('ordered_ids','groups'):
        if sha256(root / fold[key]['path']) != fold[key]['sha256']:
            raise ValueError('fold input changed: ' + key)
    ordered = read(root / fold['ordered_ids']['path'])
    for key in ('train','val'):
        if uid_hash(ordered[key]) != fold['ordered_ids'][key+'_sha256']:
            raise ValueError('ordered fold ids changed')
    out_root = Path(args.output_root).resolve() if args.output_root else (root / 'artifacts' / 'experiments').resolve()
    if smoke:
        if not args.output_root or within(out_root, root / 'artifacts') or args.smoke_train < args.bs or args.smoke_val < 1 or args.smoke_epochs < 1:
            raise ValueError('smoke requires a non-artifact output root and a nonempty tiny sample')
    elif out_root != (root / 'artifacts' / 'experiments').resolve():
        raise ValueError('full run output root must be artifacts/experiments')
    return p, fold, ordered, out_root


def runner_argv(p, fold, mode='full', workers=LOCAL_WORKERS, output_root=None, smoke=None, root=ROOT):
    root = Path(root)
    args = [train_python(), str(root / RUNNER), '--protocol', str(root / PROTOCOL / 'manifest.json'),
            '--fold',str(fold), '--mode',mode, '--tag',tag_for(fold, mode), '--seed',str(SEED), '--workers',str(workers)]
    for key, val in p['common_args'].items():
        flag = '--' + key.replace('_','-')
        if isinstance(val, bool):
            if val: args.append(flag)
        else:
            args.extend([flag,str(val)])
    for key in ('corpus','labels','eval_labels'):
        args.extend(['--'+key.replace('_','-'),str(root / p['paths'][key])])
    args.extend(['--init-weights',str(root / WEIGHTS_LOCAL)])
    if output_root:
        args.extend(['--output-root',str(output_root)])
    if smoke:
        for key in ('device','amp','eval_amp'):
            args = _set_flag(args,'--'+key.replace('_','-'),'cpu' if key=='device' else 'none')
        args.extend(['--smoke-train',str(smoke.get('train',16)), '--smoke-val',str(smoke.get('val',2)),
                     '--smoke-epochs',str(smoke.get('epochs',1))])
    return args


def recorded_screen(path, protocol_sha):
    rec = read(path)
    if rec.get('decision') != 'PASS' or rec.get('fold') != 0 or rec.get('protocol_sha256') != protocol_sha:
        raise ValueError('F1-F4 need a recorded F0 PASS for this exact EXP120 protocol')
    if not rec.get('evidence') or not rec.get('recorded_at'):
        raise ValueError('screen PASS needs evidence and recorded_at')
    return rec


def queue_entries(folds=(0,), after=None, screen_pass=None, workers=LOCAL_WORKERS, root=ROOT):
    folds = [int(k) for k in folds]
    if not folds or len(set(folds)) != len(folds) or any(k not in FOLDS for k in folds):
        raise ValueError('invalid folds')
    p = read(manifest_path(root))
    if any(k > 0 for k in folds):
        if not screen_pass:
            raise ValueError('F1-F4 require --screen-pass PATH')
        recorded_screen(screen_pass, sha256(manifest_path(root)))
    entries, previous = [], after
    for fold in folds:
        entry = dict(tag=tag_for(fold), command=runner_argv(p,fold,workers=workers,root=root), continue_on_failure=True)
        if previous: entry['after'] = previous
        entries.append(entry)
        previous = entry['tag']
    return entries


def assemble(root=ROOT):
    """Validate all five final runs, then make Dread-scored K24/K36 pooled OOF."""
    import pandas as pd
    root = Path(root)
    p = read(manifest_path(root))
    out = root / OOF_DIR
    if out.exists():
        raise FileExistsError(out)
    frames24, frames36, ids, sources = {}, {}, {}, {}
    for fold in FOLDS:
        f = p['folds'][str(fold)]
        ids[fold] = read(root / f['ordered_ids']['path'])['val']
        run = root / 'artifacts' / 'experiments' / tag_for(fold)
        for filename in RUN_FILES:
            if not (run / filename).is_file(): raise FileNotFoundError(run / filename)
        inventory = read(run / 'output-inventory.json')
        for name, meta in inventory.items():
            if sha256(run / name) != meta['sha256']: raise ValueError(f'fold {fold}: inventory {name} differs')
        prov = read(run / 'provenance.json')
        if any((prov.get(k) != v for k,v in dict(fold=fold, seed=SEED, arch=ARCH, norm='imagenet',
                     protocol_sha256=sha256(manifest_path(root)), initial_state_matches_registered=True,
                     environment_matches_baseline=True, remote_relaxed=False).items())):
            raise ValueError(f'fold {fold}: provenance mismatch')
        if prov.get('eval_ids_ordered_sha256') != f['ordered_ids']['val_sha256']:
            raise ValueError(f'fold {fold}: wrong validation ids')
        frames24[fold] = pd.read_csv(run / 'oof_best.csv',dtype={UID:str})
        frames36[fold] = pd.read_csv(run / 'oof_k36.csv',dtype={UID:str})
        sources[str(fold)] = dict(run=str(run), checkpoint_sha256=inventory['model_best.pt']['sha256'])
    result = k99.assemble_and_score(frames24,frames36,ids,root / DREAD)
    out.mkdir(parents=True)
    result['oof_k24'].to_csv(out / 'oof_k24.csv',index=False)
    result['oof_k36'].to_csv(out / 'oof_k36.csv',index=False)
    body = dict(schema='exp120-oof-v1', protocol_sha256=sha256(manifest_path(root)), rows=len(result['oof_k36']),
                metrics=result['metrics'], sources=sources,
                outputs={n:sha256(out/n) for n in ('oof_k24.csv','oof_k36.csv')})
    dump(out / 'result.json',body)
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('argv'); s.add_argument('--fold',type=int,choices=FOLDS,required=True)
    s = sub.add_parser('validate'); s.add_argument('--fold',type=int,choices=FOLDS,required=True)
    s = sub.add_parser('smoke'); s.add_argument('--out',required=True); s.add_argument('--fold',type=int,choices=FOLDS,default=0)
    s.add_argument('--train',type=int,default=16); s.add_argument('--val',type=int,default=2); s.add_argument('--epochs',type=int,default=1)
    s = sub.add_parser('queue-json'); s.add_argument('--folds',default='0'); s.add_argument('--after',required=True)
    s.add_argument('--screen-pass'); s.add_argument('--out',default='-')
    sub.add_parser('assemble')
    args = ap.parse_args(argv)
    if args.command == 'prepare':
        for key,val in {**THREAD_ENV,'HF_HUB_OFFLINE':'1'}.items(): os.environ[key]=val
        prepare()
    elif args.command in ('argv','validate','smoke'):
        p=read(manifest_path())
        if args.command == 'smoke':
            cmd=runner_argv(p,args.fold,'smoke',workers=2,output_root=args.out,
                            smoke=dict(train=args.train,val=args.val,epochs=args.epochs))
        else:
            cmd=runner_argv(p,args.fold)
        if args.command=='argv': print(json.dumps(cmd,indent=2)); return
        env={**os.environ,**THREAD_ENV,'HF_HUB_OFFLINE':'1'}
        if args.command=='validate': cmd.append('--validate-only')
        raise SystemExit(subprocess.call(cmd,env=env,cwd=ROOT))
    elif args.command=='queue-json':
        folds=[int(x) for x in args.folds.split(',')]
        entries=queue_entries(folds,None if args.after.lower()=='none' else args.after,args.screen_pass)
        if args.out!='-':
            path=Path(args.out)
            if path.exists(): raise FileExistsError(path)
            dump(path,entries)
        print(json.dumps(entries,indent=2))
    else:
        print(json.dumps(assemble(),indent=2))


if __name__=='__main__':
    main()
