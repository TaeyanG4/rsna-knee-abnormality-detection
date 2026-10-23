"""EXP101: 5-fold arch-generic MIL fits on the EXP099 folds; V1 = ResNet34 with the validated EXP084 RN_clip recipe (H-B-21, Kaggle T4).

  python tools/exp101_kfold_arch.py prepare                       # register EXP101_KFOLD_RN34_PROTOCOL_V1 (CPU; computes the RN34 seed-42 init hash)
  python tools/exp101_kfold_arch.py argv --fold K [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp101_kfold_arch.py validate --fold K             # runner --validate-only on CPU, no output
  python tools/exp101_kfold_arch.py smoke --fold K --out DIR [--train 24 --val 4 --epochs 1 --workers 2]   # tiny CPU fit, DIR outside artifacts/
  python tools/exp101_kfold_arch.py fetch [--kernel OWNER/SLUG] [--dest DIR]     # download Kaggle output, verify every fold by hash
  python tools/exp101_kfold_arch.py assemble [--runs-dir DIR] [--k36 K=PATH]     # pooled 5-fold OOF -> artifacts/experiments/EXP101_RN34_DREAD_OOF

Folds are not redrawn: every fold file of EXP099_KFOLD_PROTOCOL_V1 (ordered ids, group metadata, folds.csv, the Dread registration) is
referenced by path and sha256 and re-verified here and in the runner. The recipe (arch, epochs, learning rates, gradient checkpointing, ...)
comes from a registered arm of an earlier protocol (V1: EXP084 arm RN_clip) and is frozen in common_args; only the epoch selection differs
(fixed-final instead of OOF selection) and workers is a recorded resource argument (sampling is uid/epoch-deterministic). Per-fold contract:
len_train_loader = n_train // bs, steps = len_train_loader * epochs, FP16 skip gate steps // 100. pos_weight = the EXP099 rule on the fold's
Dread training labels (must equal the EXP099 fold vector). The seed-42 initial state hash (sorted state_dict) is computed on CPU at prepare
and enforced by the runner unless KNEE_REMOTE_RELAX_ENV is set (then recorded with the mismatching state_dict keys).
Nothing here trains on or touches the GPU.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.exp084_group_loss import (worker_init, read_rows, install_gold_blocker,  # noqa: E402,F401
                                     uid_hash, relpath, sha256, read, CODE_NAMES, ENV_KEYS, THREAD_ENV)
from tools import exp099_kfold as k99  # noqa: E402
from tools.exp099_kfold import canonical_order, validate_fold_membership, pos_weight, contract, UID, T  # noqa: E402,F401

SCHEMA = 'exp101-kfold-arch-v1'
PROTOCOL = 'artifacts/experiments/EXP101_KFOLD_RN34_PROTOCOL_V1'
FOLD_PROTOCOL = 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1'
FOLD_SCHEMA = 'exp099-kfold-b8-v1'
RECIPE_SOURCE = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
RECIPE_ARM = 'RN_clip'
RUNNER = 'training/runners/exp101_kfold_arch.py'
TOOL = 'tools/exp101_kfold_arch.py'
DREAD = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
LABEL_SET = 'dread'
TAG_PREFIX = 'EXP101_RN34_DREAD'
OOF_DIR = 'artifacts/experiments/EXP101_RN34_DREAD_OOF'
KEYS_FILE = 'initial-state-keys.json'
GPU_PYTHON = Path(r'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe')
SEED = 42
FOLDS = [0, 1, 2, 3, 4]
FROZEN_KEYS = ('arch', 'res', 'k', 'k_eval', 'epochs', 'fixed_epoch', 'bs', 'accum', 'grad_ckpt', 'amp', 'amp_init_scale', 'eval_amp', 'runtime_policy',
               'norm', 'bb_lr', 'head_lr', 'wd', 'pretrained', 'window_policy', 'device')
# the H-B-21 request, written out so that a drifted source arm cannot silently change the recipe
EXPECTED_RECIPE = dict(arch='resnet34.a1_in1k', res=224, k=12, k_eval=24, epochs=20, fixed_epoch=20, bs=8, accum=1, grad_ckpt=False, amp='fp16',
                       amp_init_scale=1024.0, eval_amp='fp16', runtime_policy='strict', norm='imagenet', bb_lr=5e-4, head_lr=1e-3, wd=0.02, pretrained='timm',
                       window_policy='within-slot-valid', device='cuda')
RESOURCE_ARGS = dict(workers=dict(default=4, min=0, max=16, note='DataLoader workers; Kaggle T4 hosts have 4 vCPUs. Window sampling is '
                                                                 'sha256(seed, epoch, uid)-deterministic and batch order comes from a seeded generator, '
                                                                 'so the count changes speed only'))
SMOKE_ALLOWED = {'device': ('cuda', 'cpu'), 'amp': ('fp16', 'none'), 'eval_amp': ('fp16', 'none')}
EXTRA_EVAL = dict(k_eval=36, file='oof_k36.csv', source='D-B-012: same final model re-evaluated with 36 evenly spaced windows')
RUN_FILES = ('summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv', 'output-inventory.json')
DOCKER_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
KERNEL = 'taeyangg4/knee-run-v3-exp101-rn34-5fold'


# ------------------------------------------------------------------ small helpers
def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def tag_for(fold, mode='full'):
    return f'{TAG_PREFIX}_F{int(fold)}' + ('_SMOKE' if mode == 'smoke' else '')


def train_python():
    return str(GPU_PYTHON) if GPU_PYTHON.exists() else sys.executable


def within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def display_path(path, root=ROOT):
    p = Path(path).resolve()
    return p.relative_to(Path(root).resolve()).as_posix() if within(p, root) else str(p)


# ------------------------------------------------------------------ pure recipe / contract logic (unit-tested)
def recipe_args(expected):
    """Frozen common_args from a registered arm's expected_args. Fixed-final only; FP16 only (T4 lane, D-B-015); no accumulation loop."""
    missing = [k for k in FROZEN_KEYS if k not in expected]
    if missing:
        raise ValueError('recipe arm lacks: ' + ', '.join(missing))
    args = {k: expected[k] for k in FROZEN_KEYS}
    args['amp_init_scale'] = float(args['amp_init_scale'])
    for k in ('bb_lr', 'head_lr', 'wd'):
        args[k] = float(args[k])
    if args['fixed_epoch'] != args['epochs']:
        raise ValueError('fixed-final requires fixed_epoch == epochs')
    if args['amp'] != 'fp16' or args['eval_amp'] != 'fp16':
        raise ValueError('the T4 lane trains and evaluates in FP16 (no native bf16, D-B-015)')
    if args['accum'] != 1:
        raise ValueError('the runner has no gradient-accumulation loop; accum must be 1')
    if args['device'] != 'cuda' or args['pretrained'] != 'timm':
        raise ValueError('full runs are CUDA with timm-pretrained weights')
    return args


def fold_contract(n_train, args):
    """EXP099 contract rule with this recipe's bs/accum/epochs (steps = floor(n_train / bs) * epochs for accum 1)."""
    c = contract(n_train, args)
    if c['steps'] != (n_train // args['bs']) * args['epochs']:
        raise AssertionError('contract rule drifted')
    return c


def state_hashes(model):
    """(combined sha256 over sorted keys exactly as the EXP099 runner, {key: [sha256, dtype, shape]})."""
    combined, keys = hashlib.sha256(), {}
    for key, value in sorted(model.state_dict().items()):
        t = value.detach().cpu().contiguous()
        raw = t.numpy().tobytes()
        combined.update(key.encode()); combined.update(raw)
        keys[key] = [hashlib.sha256(raw).hexdigest(), str(t.dtype).replace('torch.', ''), list(t.shape)]
    return combined.hexdigest(), keys


def mismatched_keys(keys, registered):
    return sorted(k for k in set(keys) | set(registered) if list(keys.get(k) or []) != list(registered.get(k) or []))


def should_start_fold(elapsed, durations, budget_seconds, prior_seconds=3600.0, margin=1.3):
    """Start another fold only if the slowest fold so far (or the prior) times the margin still fits the session budget."""
    need = margin * max([prior_seconds, *durations])
    return elapsed + need <= budget_seconds, need


# ------------------------------------------------------------------ prepare
def _load_labels(path):
    return k99._load_labels(path)


def build_initial_model(arch, seed):
    """Same order as the runner: bounded threads -> seed_all -> build_model (pretrained timm weights from the local HF cache)."""
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_common import build_model, seed_all
    worker_init(0)
    seed_all(seed)
    return build_model(arch, pretrained=True)


def prepare(root=ROOT, model_factory=None):
    import numpy as np
    root = Path(root)
    destination = root / PROTOCOL
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + PROTOCOL)
    fold_manifest_path = root / FOLD_PROTOCOL / 'manifest.json'
    fp = read(fold_manifest_path)
    if fp['schema'] != FOLD_SCHEMA:
        raise ValueError('fold protocol schema changed')
    if sha256(root / fp['paths']['folds']) != fp['folds_csv_sha256']:
        raise ValueError('EXP099 folds.csv changed')
    reg, reg_file = k99.load_registration(root, fp, LABEL_SET, fold_manifest_path)
    if reg['path'] != DREAD or fp['label_sets'][LABEL_SET]['path'] != DREAD:
        raise ValueError('EXP099 dread registration points elsewhere')
    src = read(root / RECIPE_SOURCE)
    arm = src['arms'][RECIPE_ARM]
    if arm['loss_policy'] != 'clipped_positive_term' or arm['expected_args']['seed'] != SEED:
        raise ValueError('recipe arm changed (loss policy / seed)')
    common = recipe_args(arm['expected_args'])
    if common != EXPECTED_RECIPE:
        raise ValueError('RN_clip expected_args differ from the H-B-21 recipe: ' + str({k: (common[k], EXPECTED_RECIPE.get(k)) for k in common if common[k] != EXPECTED_RECIPE.get(k)}))
    if fp['training_code_sha256'] != src['training_code_sha256']:
        raise ValueError('EXP099 and EXP084 bind different training sources')
    for name, expected in fp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    labels, semantic = _load_labels(root / DREAD)
    if semantic != reg['semantic_sha256'] or sha256(root / DREAD) != reg['sha256']:
        raise ValueError('Dread file differs from the EXP099 registration')
    canonical = canonical_order(read(root / fp['paths']['base_ordered_ids']))
    corpus_dir = root / fp['paths']['corpus']
    if sha256(corpus_dir / 'manifest.json') != fp['corpus_manifest_sha256']:
        raise ValueError('corpus manifest differs from EXP099')
    corpus_ids = np.load(corpus_dir / 'ids.npy', allow_pickle=True).astype(str).tolist()
    folds, cover = {}, []
    for k in FOLDS:
        f99 = fp['folds'][str(k)]
        for key in ('ordered_ids', 'groups'):
            if sha256(root / f99[key]['path']) != f99[key]['sha256']:
                raise ValueError(f'EXP099 fold {k} {key} file changed')
        ordered = read(root / f99['ordered_ids']['path'])
        if uid_hash(ordered['train']) != f99['ordered_ids']['train_sha256'] or uid_hash(ordered['val']) != f99['ordered_ids']['val_sha256']:
            raise ValueError(f'EXP099 fold {k} ordered ids changed')
        rows = read_rows(root / f99['groups']['path'])
        train, val = validate_fold_membership(corpus_ids, ordered, rows, canonical)
        pw = pos_weight(labels, train)
        if not np.array_equal(pw, np.asarray(f99['pos_weight'], np.float32)) or not np.array_equal(pw, np.asarray(reg['pos_weight'][str(k)], np.float32)):
            raise ValueError(f'fold {k} pos_weight differs from EXP099')
        c = fold_contract(len(train), common)
        folds[str(k)] = dict(fold=k, tag=tag_for(k), ordered_ids=dict(f99['ordered_ids']), groups=dict(f99['groups']), n_train=len(train), n_val=len(val),
                             train_components=f99['train_components'], val_components=f99['val_components'], **c, pos_weight=pw.tolist(),
                             pos_weight_equals_exp099=True, val_positive_counts=f99['val_positive_counts'], macro12_valid=f99['macro12_valid'])
        cover += val
    if sorted(cover) != sorted(canonical) or len(set(cover)) != len(canonical):
        raise ValueError('fold validation sets do not partition the 4,343 studies')
    model = (model_factory or build_initial_model)(common['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init, keys = state_hashes(model)
    import timm
    destination.mkdir(parents=True)
    dump(destination / KEYS_FILE, dict(schema='exp101-initial-state-keys-v1', arch=common['arch'], seed=SEED, combined_sha256=init, keys=keys,
                                       rule='per state_dict key: sha256 of the contiguous CPU tensor bytes; combined = sha256 over sorted (key bytes + tensor bytes)'))
    bindings = [RUNNER, TOOL, 'tools/exp099_kfold.py', 'tools/exp087_training.py', 'tools/exp084_group_loss.py', RECIPE_SOURCE,
                fp['paths']['base_ordered_ids'], fp['paths']['base_groups'], FOLD_PROTOCOL + '/manifest.json', fp['paths']['folds'],
                display_path(reg_file, root), DREAD, PROTOCOL + '/' + KEYS_FILE]
    manifest = dict(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item='H-B-21',
        source_bindings={s: sha256(root / s) for s in bindings},
        training_code_sha256=fp['training_code_sha256'], environment=fp['environment'], corpus_manifest_sha256=fp['corpus_manifest_sha256'],
        excluded_uids=fp['excluded_uids'],
        recipe=dict(source=RECIPE_SOURCE, arm=RECIPE_ARM, tag=arm['tag'], source_expected_args=arm['expected_args'], loss_policy=arm['loss_policy'],
                    changes=['checkpoint selection: fixed-final epoch 20 (RN_clip expected_args say select=oof; its run used the fixed final epoch, '
                             'checkpoint fixed-final in its train.log); no OOF metric is read during training',
                             'cohort: EXP099 fold K instead of the single G1 split; pos_weight per fold (EXP099 rule)',
                             'workers: resource argument (default 4 for Kaggle T4), not frozen',
                             'K36 extra evaluation of the same final model (oof_k36.csv)']),
        common_args=common, seed=SEED, resource_args=RESOURCE_ARGS, smoke_allowed=SMOKE_ALLOWED,
        initial_state_sha256=init,
        initial_state=dict(keys_file=PROTOCOL + '/' + KEYS_FILE, keys=len(keys), computed_on='cpu', seed=SEED, arch=common['arch'],
                           torch=torch.__version__, timm=timm.__version__, rule='seed_all(42) after worker_init(0), build_model(arch, pretrained timm); sorted state_dict sha256',
                           equals_recipe_arm_initial_state=init == arm['initial_state_sha256'], recipe_arm_initial_state_sha256=arm['initial_state_sha256'],
                           enforcement='runner: seed 42 must match, unless KNEE_REMOTE_RELAX_ENV is set (then recorded with mismatching keys; D-B-008)'),
        fold_protocol=dict(path=FOLD_PROTOCOL + '/manifest.json', sha256=sha256(fold_manifest_path), folds_csv_sha256=fp['folds_csv_sha256'],
                           label_registration=dict(path=display_path(reg_file, root), sha256=sha256(reg_file)), reuse='fold files referenced by path + sha256, not copied'),
        paths=dict(corpus=fp['paths']['corpus'], labels=DREAD, eval_labels=DREAD, base_ordered_ids=fp['paths']['base_ordered_ids'],
                   base_groups=fp['paths']['base_groups'], folds=fp['paths']['folds']),
        label_set=dict(name=LABEL_SET, path=DREAD, sha256=reg['sha256'], semantic_sha256=reg['semantic_sha256']),
        folds=folds, total_steps=sum(v['steps'] for v in folds.values()),
        pos_weight_rule=fp['pos_weight_rule'],
        contract_rule='len_train_loader = n_train // bs (drop_last); steps = len_train_loader * epochs (accum 1) = scheduler total_steps; '
                      'FP16 skip gate: skipped > steps // 100 fails; eval batch bs // 2',
        extra_eval=EXTRA_EVAL, tags=dict(prefix=TAG_PREFIX, pattern=TAG_PREFIX + '_F<K>[_SMOKE]', oof_dir=OOF_DIR),
        remote=dict(platform='Kaggle T4 (NvidiaTeslaT4), FP16 + GradScaler 1024', docker_image=DOCKER_IMAGE, kernel=KERNEL,
                    relax_env='KNEE_REMOTE_RELAX_ENV=1: environment and seed-42 initial state recorded, not enforced; independent trajectories, '
                              'never paired with a local arm (D-B-008)', output_root='/kaggle/working/<tag>/ written fold by fold'),
        evaluation=dict(labels=DREAD, rule='float32 then numpy.rint (tools/g1_compare.py)', metric='macro12 per fold and pooled over 4343 OOF (K24 and K36)',
                        oof_dir=OOF_DIR),
        caveats=['weak Dread labels; folds 1-4 include G1 val studies in training, so fold OOF is not comparable to single G1 V888 numbers',
                 'remote T4 trajectories differ from local ones (Windows vs Linux window pipeline, D-B-008); compare only within the T4 lane',
                 'component grouping does not prove patient independence', 'OOF is for ensembling / teacher pseudo-labels; not private evidence'])
    dump(destination / 'manifest.json', manifest)
    report = dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, manifest_sha256=sha256(destination / 'manifest.json'), initial_state_sha256=init,
                  equals_recipe_arm_initial_state=manifest['initial_state']['equals_recipe_arm_initial_state'], total_steps=manifest['total_steps'],
                  folds={k: dict(n_train=v['n_train'], n_val=v['n_val'], len_train_loader=v['len_train_loader'], steps=v['steps'], max_fp16_skips=v['max_fp16_skips'])
                         for k, v in folds.items()})
    print(json.dumps(report, indent=1))
    return manifest


# ------------------------------------------------------------------ runner guard
def validate_arguments(args, root):
    """Called by the runner before any training import. Returns (protocol, fold spec, ordered ids, output root)."""
    root = Path(root)
    manifest_path = Path(args.protocol)
    p = read(manifest_path)
    if p.get('schema') != SCHEMA or manifest_path.resolve() != (root / PROTOCOL / 'manifest.json').resolve():
        raise ValueError('wrong protocol')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed')
    smoke = args.mode == 'smoke'
    for k, v in p['common_args'].items():
        got = getattr(args, k, None)
        if got != v and not (smoke and k in p['smoke_allowed'] and got in p['smoke_allowed'][k]):
            raise ValueError('frozen argument changed: ' + k)
    if args.seed != p['seed']:
        raise ValueError('seed changed')
    w = p['resource_args']['workers']
    if not w['min'] <= args.workers <= w['max']:
        raise ValueError('workers outside the registered range')
    if str(args.fold) not in p['folds']:
        raise ValueError('unregistered fold')
    fold = p['folds'][str(args.fold)]
    if tag_for(args.fold, args.mode) != args.tag:
        raise ValueError('tag/fold/mode mismatch')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    for key in ('ordered_ids', 'groups'):
        if sha256(relpath(root, fold[key]['path'])) != fold[key]['sha256']:
            raise ValueError(f'fold {args.fold} {key} file changed')
    ordered = read(relpath(root, fold['ordered_ids']['path']))
    for k in ('train', 'val'):
        if uid_hash(ordered[k]) != fold['ordered_ids'][k + '_sha256']:
            raise ValueError('ordered fold ids changed')
    default_root = (root / 'artifacts' / 'experiments').resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else default_root
    if smoke:
        if not args.output_root or within(out_root, root / 'artifacts'):
            raise ValueError('smoke runs need --output-root outside artifacts/')
        if args.smoke_train < args.bs or args.smoke_val < 1 or args.smoke_epochs < 1:
            raise ValueError('smoke needs >= one batch of training studies, >= 1 validation study and >= 1 epoch')
    elif out_root != default_root and not os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('a non-default --output-root is for remote runs (KNEE_REMOTE_RELAX_ENV=1)')
    return p, fold, ordered, out_root


def _set_flag(argv, flag, value):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag, str(value)]
    return argv


def runner_argv(p, fold, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None):
    root = Path(root)
    argv = [python or train_python(), str(root / RUNNER), '--protocol', str(root / PROTOCOL / 'manifest.json'), '--fold', str(int(fold)),
            '--mode', mode, '--tag', tag_for(fold, mode), '--seed', str(p['seed']),
            '--workers', str(p['resource_args']['workers']['default'] if workers is None else int(workers))]
    for k, v in p['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--corpus', str(relpath(root, p['paths']['corpus'])), '--labels', str(relpath(root, p['paths']['labels'])),
             '--eval-labels', str(relpath(root, p['paths']['eval_labels']))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if mode == 'smoke':
        smoke = {**dict(device='cpu', amp='none', eval_amp='none', train=24, val=4, epochs=1), **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), smoke[key])
        argv += ['--smoke-train', str(smoke['train']), '--smoke-val', str(smoke['val']), '--smoke-epochs', str(smoke['epochs'])]
    return argv


def _protocol(root=ROOT):
    return read(Path(root) / PROTOCOL / 'manifest.json')


def local_env():
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(ROOT)}
    env.pop('KNEE_REMOTE_RELAX_ENV', None)
    return env


def validate_cli(fold):
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path)."""
    argv = runner_argv(_protocol(), fold, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(fold, out, train, val, epochs, workers):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    argv = runner_argv(_protocol(), fold, 'smoke', workers=workers, output_root=out, smoke=dict(train=train, val=val, epochs=epochs))
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for(fold, 'smoke')
    report = dict(returncode=r.returncode, seconds=round(time.time() - t0), run=str(run), files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json')
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], val=s['val'],
                      initial_state_matches_registered=s['provenance']['initial_state_matches_registered'], history=s['history'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ verification of finished fold runs (local or downloaded from Kaggle)
def verify_run(run, p, k, manifest_sha):
    """Problems (empty list = complete and consistent) for one fold directory."""
    run = Path(run); fold = p['folds'][str(k)]
    missing = [n for n in RUN_FILES if not (run / n).is_file()]
    if missing:
        return [f'fold {k}: missing ' + ', '.join(missing)]
    problems = []
    inv = read(run / 'output-inventory.json')
    for name, rec in inv.items():
        if not (run / name).is_file() or sha256(run / name) != rec['sha256']:
            problems.append(f'fold {k}: {name} differs from output-inventory.json (incomplete download?)')
    s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expect = dict(fold=k, label_set=LABEL_SET, label_semantic_sha256=p['label_set']['semantic_sha256'], protocol_sha256=manifest_sha,
                  eval_ids_ordered_sha256=fold['ordered_ids']['val_sha256'], train_ids_ordered_sha256=fold['ordered_ids']['train_sha256'],
                  training_mode='full', checkpoint_selection='fixed-final')
    for key, value in expect.items():
        if prov.get(key) != value:
            problems.append(f'fold {k}: provenance {key} = {prov.get(key)!r}, expected {value!r}')
    if s.get('optimizer_updates') != fold['steps'] or s.get('skipped_updates', 10 ** 9) > fold['max_fp16_skips']:
        problems.append(f"fold {k}: updates {s.get('optimizer_updates')} (expected {fold['steps']}), skipped {s.get('skipped_updates')}")
    if s.get('selected_epoch_1based') != p['common_args']['epochs'] or s.get('train') != fold['n_train'] or s.get('val') != fold['n_val']:
        problems.append(f'fold {k}: epoch/count mismatch')
    return problems


def kaggle_cli():
    exe = ROOT / '.venv' / 'Scripts' / 'kaggle.exe'
    return [str(exe)] if exe.exists() else ['kaggle']


def fetch(kernel=KERNEL, dest=None, retries=3, root=ROOT):
    """Download the kernel output fold by fold (small pages, --force) until every fold verifies by hash or retries run out."""
    root = Path(root); p = _protocol(root); manifest_sha = sha256(root / PROTOCOL / 'manifest.json')
    dest = Path(dest or root / '.local' / 'kaggle-out' / kernel.split('/')[-1]).resolve(); dest.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    st = subprocess.run([*kaggle_cli(), 'kernels', 'status', kernel], capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    print((st.stdout or st.stderr).strip()[-400:])
    patterns = ['exp101-status', 'remote_run_v3'] + [tag_for(k) for k in FOLDS]
    state = {}
    for pattern in patterns:
        for attempt in range(retries):
            r = subprocess.run([*kaggle_cli(), 'kernels', 'output', kernel, '-p', str(dest), '-o', '--file-pattern', pattern, '--page-size', '200'],
                               capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
            if pattern.startswith(TAG_PREFIX):
                k = int(pattern.rsplit('_F', 1)[1])
                problems = verify_run(dest / pattern, p, k, manifest_sha)
                state[pattern] = problems or 'OK'
                if not problems:
                    break
            else:
                state[pattern] = 'downloaded' if r.returncode == 0 else (r.stderr or r.stdout)[-300:]
                break
    status_file = dest / 'exp101-status.json'
    print(json.dumps(dict(dest=str(dest), folds=state, kernel_status=read(status_file).get('state') if status_file.exists() else None), indent=1))
    return state


# ------------------------------------------------------------------ assemble
def assemble(runs_dir=None, k36_override=None, root=ROOT):
    import pandas as pd
    root = Path(root); k36_override = k36_override or {}
    manifest_path = root / PROTOCOL / 'manifest.json'
    p = read(manifest_path); manifest_sha = sha256(manifest_path)
    runs = Path(runs_dir).resolve() if runs_dir else root / 'artifacts' / 'experiments'
    out = root / OOF_DIR
    if out.exists():
        raise SystemExit('output exists: ' + str(out))
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in p['folds'].items()}
    frames24, frames36, sources, problems = {}, {}, {}, []
    for k in sorted(val_ids):
        run = runs / tag_for(k)
        bad = verify_run(run, p, k, manifest_sha)
        if bad:
            problems += bad; continue
        prov = read(run / 'provenance.json')
        frames24[k] = pd.read_csv(run / 'oof_best.csv', dtype={UID: str})
        k36 = Path(k36_override[k]) if k in k36_override else run / EXTRA_EVAL['file']
        if k36.is_file():
            frames36[k] = pd.read_csv(k36, dtype={UID: str})
        sources[k] = dict(run=display_path(run, root), oof_k24_sha256=sha256(run / 'oof_best.csv'), oof_k36=display_path(k36, root) if k36.is_file() else None,
                          oof_k36_sha256=sha256(k36) if k36.is_file() else None, checkpoint_sha256=read(run / 'output-inventory.json')['model_best.pt']['sha256'],
                          initial_state_matches_registered=prov.get('initial_state_matches_registered'), environment=prov.get('environment'),
                          remote_relaxed=prov.get('remote_relaxed'))
    if problems:
        print(json.dumps(dict(status='INCOMPLETE', problems=problems), indent=1))
        raise SystemExit(2)
    result = k99.assemble_and_score(frames24, frames36, val_ids, root / DREAD)
    out.mkdir(parents=True)
    result['oof_k24'].to_csv(out / 'oof_k24.csv', index=False)
    if result['oof_k36'] is not None:
        result['oof_k36'].to_csv(out / 'oof_k36.csv', index=False)
    body = dict(schema='exp101-oof-v1', label_set=LABEL_SET, label_set_sha256=p['label_set']['sha256'], observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                protocol_sha256=manifest_sha, runs_dir=display_path(runs, root), rows=int(len(result['oof_k24'])), sources={str(k): v for k, v in sources.items()},
                outputs={f: sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv') if (out / f).exists()},
                scoring=dict(labels=DREAD, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py'),
                metrics=result['metrics'], k36_missing_folds=result['k36_missing'],
                caveats=['weak Dread labels', 'fold 0 val = exposed G1 V888; folds 1-4 models saw G1 val in training', 'pooled AUC mixes 5 models',
                         'Kaggle T4 trajectories (KNEE_REMOTE_RELAX_ENV); compare with local arms only as separate families'])
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    md = k99.result_markdown(LABEL_SET, body).replace(f'# EXP099 B8 5-fold OOF ({LABEL_SET})', f'# EXP101 ResNet34 5-fold OOF ({LABEL_SET}, Kaggle T4)', 1)
    (out / 'result.md').write_text(md, encoding='utf-8')
    print(json.dumps(dict(rows=body['rows'], metrics={k: {kk: (round(vv['macro12'], 6) if isinstance(vv, dict) else vv) for kk, vv in v.items()}
                                                      for k, v in body['metrics'].items()}, k36_missing_folds=body['k36_missing_folds']), indent=1))
    return body


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('argv'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--mode', choices=['full', 'smoke'], default='full')
    s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=None)
    s = sub.add_parser('validate'); s.add_argument('--fold', type=int, choices=FOLDS, required=True)
    s = sub.add_parser('smoke'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--out', required=True)
    s.add_argument('--train', type=int, default=24); s.add_argument('--val', type=int, default=4); s.add_argument('--epochs', type=int, default=1)
    s.add_argument('--workers', type=int, default=2)
    s = sub.add_parser('fetch'); s.add_argument('--kernel', default=KERNEL); s.add_argument('--dest', default=None); s.add_argument('--retries', type=int, default=3)
    s = sub.add_parser('assemble'); s.add_argument('--runs-dir', default=None, help='directory holding EXP101_RN34_DREAD_F<K>/ (default artifacts/experiments)')
    s.add_argument('--k36', action='append', default=[], help='FOLD=PATH oof_k36.csv override')
    a = ap.parse_args()
    if a.command == 'prepare':
        for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():  # before torch/timm/huggingface_hub are imported
            os.environ[k] = v
        prepare()
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), a.fold, a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli(a.fold))
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.fold, a.out, a.train, a.val, a.epochs, a.workers))
    elif a.command == 'fetch':
        fetch(a.kernel, a.dest, a.retries)
    else:
        assemble(a.runs_dir, {int(x.split('=', 1)[0]): x.split('=', 1)[1] for x in a.k36})


if __name__ == '__main__':
    main()
