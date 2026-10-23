"""EXP107: RadImageNet-initialised ResNet50 MIL family, centre slice x3, 5-fold on the EXP099 folds with Dread labels (H-B-24, Kaggle T4).

  python tools/exp107_kfold_radr50.py prepare                   # register EXP107_KFOLD_RADR50_PROTOCOL_V1 (CPU; computes the seed-42 init hash)
  python tools/exp107_kfold_radr50.py argv --fold K [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp107_kfold_radr50.py validate --fold K         # runner --validate-only on CPU, no output
  python tools/exp107_kfold_radr50.py smoke --fold K --out DIR [--train 24 --val 4 --epochs 1 --workers 2]   # tiny CPU fit, DIR outside artifacts/
  python tools/exp107_kfold_radr50.py fetch [--kernel OWNER/SLUG] [--dest DIR]   # download the Kaggle output, verify every fold by hash
  python tools/exp107_kfold_radr50.py assemble [--runs-dir DIR]                  # pooled 5-fold OOF -> artifacts/experiments/EXP107_RADR50_DREAD_OOF

Derived from the EXP101 ResNet34 RN_clip registration (EXP101_KFOLD_RN34_PROTOCOL_V1, manifest sha256 BASE_MANIFEST_SHA256, not changed):
prepare copies it and re-verifies every fold file, the per-fold pos_weight (recomputed from the Dread training labels), the loader contract
and every source binding. Changed variables (RECIPE_CHANGES): the backbone (timm resnet50 with the RadImageNet checkpoint, strict load,
file sha256 RADIMAGENET_SHA256), the input (centre cache plane x3, same within-slot-valid centres) and the normalisation (2x - 1). Learning
rates, weight decay, epochs, batch, FP16 + GradScaler 1024, K12 / K24 / K36 and the clipped pos_weight stay RN_clip's. The weight file is
not shipped in the bundle: locally it is WEIGHTS_LOCAL, remotely the public dataset WEIGHTS_DATASET attached to the kernel (located by
sha256; env WEIGHTS_ENV). Runner: training/runners/exp107_kfold_radr50.py. Nothing here trains on or touches the GPU.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp101_kfold_arch as base  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
# names the runner reads from its guard module (`guard.<name>`)
from tools.exp101_kfold_arch import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                     CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
                                     FOLD_PROTOCOL, RECIPE_SOURCE, RECIPE_ARM, DREAD, LABEL_SET, KEYS_FILE, FOLDS, EXTRA_EVAL, RUN_FILES,
                                     DOCKER_IMAGE, recipe_args, fold_contract, state_hashes, mismatched_keys, should_start_fold, dump,
                                     train_python, within, display_path, kaggle_cli, _set_flag)

SCHEMA = 'exp107-kfold-radr50-v1'
BASE_PROTOCOL = base.PROTOCOL
BASE_MANIFEST_SHA256 = 'edb5b124dfbc033c2b7ad5b4c546f6f0377430454b05d00f9799f2778bdff0d1'
BASE_SEED = 42
PROTOCOL = 'artifacts/experiments/EXP107_KFOLD_RADR50_PROTOCOL_V1'
RUNNER = 'training/runners/exp107_kfold_radr50.py'
TOOL = 'tools/exp107_kfold_radr50.py'
RAD_MODULE = 'training/src/raptor_radimagenet.py'
SEED = 42
TAG_PREFIX = 'EXP107_RADR50_DREAD'
OOF_DIR = 'artifacts/experiments/EXP107_RADR50_DREAD_OOF'
KERNEL = 'taeyangg4/knee-run-v3-exp107-radr50-5fold'
BUNDLE_DATASET = 'knee-runners-v3-exp107'
REMOTE_ENTRY = 'tools/remote_run_v3_exp107.py'
STATUS_FILE = 'exp101-status.json'  # written by tools/remote_run_v3.py for every job
WEIGHTS_LOCAL = 'storage/weights/marwanmath__resnet-50-radimagenet-marwan/ResNet50.pt'
WEIGHTS_ENV = 'KNEE_RADIMAGENET_R50'
WEIGHTS_DATASET = 'marwanmath/resnet-50-radimagenet-marwan'
WEIGHTS_NAME = 'ResNet50.pt'
RADIMAGENET_SHA256 = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
WEIGHTS_LICENCE = 'CC-BY-NC-SA-4.0 (Kaggle dataset licence field, observed 2026-10-07; empty description)'
RECIPE_CHANGES = dict(arch='resnet50', norm='radimagenet', pretrained='radimagenet', input_policy='center3')
EXPECTED_RECIPE = {**base.EXPECTED_RECIPE, **RECIPE_CHANGES}
CHANGE_NOTES = [
    'backbone: resnet34.a1_in1k (timm ImageNet) -> timm resnet50 initialised from the RadImageNet ResNet50 checkpoint (public Kaggle dataset '
    'marwanmath/resnet-50-radimagenet-marwan, sha256 08629f7e..., the encoder the 0.943 anchor pins); keys backbone.{0,1,4,5,6,7} -> '
    'conv1/bn1/layer1-4, strict load (D-B-013: RadImageNet beat ImageNet R50 by +0.063 on the forum)',
    'input: centre cache plane repeated as the three channels (input_policy center3) instead of the (c-1, c, c+1) cache-neighbour triplet; '
    'window centres, within-slot-valid policy, sampling streams and intensity jitter unchanged (D-B-013: +0.020 OOF for RadImageNet)',
    'normalisation: 2x - 1 = the anchor uint8 / 127.5 - 1, instead of ImageNet mean/std (conv1 outputs on cache slices match the '
    "checkpoint's bn1 running statistics best at [-1, 1])",
    'kept from RN_clip: 224 px, bs 8, K12 train / K24 eval (+K36), 20 epochs fixed-final, AdamW bb_lr 5e-4 / head_lr 1e-3 / wd 0.02, '
    'OneCycle pct 0.15, FP16 + GradScaler 1024, clipped pos_weight per fold, seed 42; no GPU was available for the lr check on fold 0 '
    '(H-B-24 Next test), so the validated recipe is kept and the family is judged as a whole',
]
# manifest fields that may differ from the EXP101 seed-42 registration
CHANGED_FIELDS = ('schema', 'created_at', 'plan_item', 'source_bindings', 'recipe', 'common_args', 'initial_state_sha256', 'initial_state', 'folds',
                  'tags', 'remote', 'evaluation', 'caveats', 'init_weights', 'input', 'derived_from')
FOLD_CHANGED_FIELDS = ('tag',)


def tag_for(fold, mode='full'):
    return f'{TAG_PREFIX}_F{int(fold)}' + ('_SMOKE' if mode == 'smoke' else '')


def _protocol(root=ROOT):
    return read(Path(root) / PROTOCOL / 'manifest.json')


def weights_path(root=ROOT):
    """WEIGHTS_ENV if set (remote: the attached public dataset), else the local copy."""
    env = os.environ.get(WEIGHTS_ENV)
    return Path(env) if env else Path(root) / WEIGHTS_LOCAL


def build_initial_model(arch, seed, weights=None):
    """Same order as the runner: bounded threads -> seed_all -> build_radimagenet_model (CPU)."""
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_common import seed_all
    from raptor_radimagenet import build_radimagenet_model
    worker_init(0)
    seed_all(seed)
    return build_radimagenet_model(arch, weights or weights_path(), RADIMAGENET_SHA256)


# ------------------------------------------------------------------ prepare
def prepare(root=ROOT, model_factory=None):
    import numpy as np
    root = Path(root)
    destination = root / PROTOCOL
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + PROTOCOL)
    base_path = root / BASE_PROTOCOL / 'manifest.json'
    if sha256(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError('EXP101 seed-42 manifest changed')
    bp = read(base_path)
    if bp['schema'] != base.SCHEMA or bp['seed'] != BASE_SEED or bp['common_args'] != base.EXPECTED_RECIPE:
        raise ValueError('unexpected EXP101 registration')
    for path, expected in bp['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('EXP101 binding changed: ' + path)
    for name, expected in bp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    arm = read(root / RECIPE_SOURCE)['arms'][RECIPE_ARM]
    if recipe_args(arm['expected_args']) != base.EXPECTED_RECIPE:
        raise ValueError('RN_clip expected_args differ from the EXP101 recipe')
    common = {**bp['common_args'], **RECIPE_CHANGES}
    if common != EXPECTED_RECIPE:
        raise AssertionError('recipe construction drifted')
    labels, semantic = k99._load_labels(root / DREAD)
    if semantic != bp['label_set']['semantic_sha256'] or sha256(root / DREAD) != bp['label_set']['sha256']:
        raise ValueError('Dread file differs from the EXP101 registration')
    if sha256(root / bp['fold_protocol']['path']) != bp['fold_protocol']['sha256'] or sha256(root / bp['paths']['folds']) != bp['fold_protocol']['folds_csv_sha256']:
        raise ValueError('EXP099 fold protocol changed')
    canonical = canonical_order(read(root / bp['paths']['base_ordered_ids']))
    corpus_dir = root / bp['paths']['corpus']
    if sha256(corpus_dir / 'manifest.json') != bp['corpus_manifest_sha256']:
        raise ValueError('corpus manifest differs from the EXP101 registration')
    corpus_ids = np.load(corpus_dir / 'ids.npy', allow_pickle=True).astype(str).tolist()
    folds, cover = {}, []
    for k in FOLDS:
        f42 = bp['folds'][str(k)]
        for key in ('ordered_ids', 'groups'):
            if sha256(root / f42[key]['path']) != f42[key]['sha256']:
                raise ValueError(f'fold {k} {key} file changed')
        ordered = read(root / f42['ordered_ids']['path'])
        if uid_hash(ordered['train']) != f42['ordered_ids']['train_sha256'] or uid_hash(ordered['val']) != f42['ordered_ids']['val_sha256']:
            raise ValueError(f'fold {k} ordered ids changed')
        train, val = validate_fold_membership(corpus_ids, ordered, read_rows(root / f42['groups']['path']), canonical)
        pw = pos_weight(labels, train)
        if not np.array_equal(pw, np.asarray(f42['pos_weight'], np.float32)):
            raise ValueError(f'fold {k} pos_weight differs from the EXP101 registration')
        c = fold_contract(len(train), common)
        if any(c[x] != f42[x] for x in c) or (len(train), len(val)) != (f42['n_train'], f42['n_val']):
            raise ValueError(f'fold {k} contract differs from the EXP101 registration')
        folds[str(k)] = {**copy.deepcopy(f42), 'tag': tag_for(k)}
        cover += val
    if sorted(cover) != sorted(canonical) or len(set(cover)) != len(canonical):
        raise ValueError('fold validation sets do not partition the 4,343 studies')
    weights = weights_path(root)
    if sha256(weights) != RADIMAGENET_SHA256:
        raise ValueError(f'RadImageNet weight file differs from {RADIMAGENET_SHA256}: {weights}')
    model, load_report = (model_factory or build_initial_model)(common['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init, keys = state_hashes(model)
    backbone_keys = sorted(k for k in keys if k.startswith('backbone.'))
    if len(backbone_keys) != load_report['source_keys'] or load_report['missing'] or load_report['unexpected']:
        raise ValueError('RadImageNet load report inconsistent with the model state')
    import timm
    destination.mkdir(parents=True)
    dump(destination / KEYS_FILE, dict(schema='exp101-initial-state-keys-v1', arch=common['arch'], seed=SEED, combined_sha256=init, keys=keys,
                                       init='radimagenet', init_weights_sha256=RADIMAGENET_SHA256,
                                       rule='per state_dict key: sha256 of the contiguous CPU tensor bytes; combined = sha256 over sorted (key bytes + tensor bytes)'))
    extra = [BASE_PROTOCOL + '/manifest.json', RUNNER, TOOL, RAD_MODULE, PROTOCOL + '/' + KEYS_FILE]
    bindings = {**bp['source_bindings'], **{s: sha256(root / s) for s in extra}}
    manifest = copy.deepcopy(bp)
    manifest.update(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item='H-B-24', source_bindings=bindings,
        recipe={**bp['recipe'], 'changes': [*bp['recipe']['changes'], *CHANGE_NOTES]}, common_args=common, seed=SEED,
        initial_state_sha256=init,
        initial_state=dict(keys_file=PROTOCOL + '/' + KEYS_FILE, keys=len(keys), backbone_keys=len(backbone_keys), computed_on='cpu', seed=SEED,
                           arch=common['arch'], torch=torch.__version__, timm=timm.__version__,
                           rule='worker_init(0), seed_all(42), build_model(resnet50, pretrained=False) [random init drawn as for any timm backbone], '
                                'then the RadImageNet backbone loaded strictly; sorted state_dict sha256',
                           enforcement='runner: seed 42 must match, unless KNEE_REMOTE_RELAX_ENV is set (then recorded with mismatching keys; D-B-008); '
                                       'a mismatching backbone.* key always stops the run (the checkpoint fixes every backbone byte)'),
        init_weights=dict(name='RadImageNet ResNet50', file=WEIGHTS_NAME, sha256=RADIMAGENET_SHA256, bytes=weights.stat().st_size,
                          local=WEIGHTS_LOCAL, kaggle_dataset=WEIGHTS_DATASET, licence=WEIGHTS_LICENCE, env=WEIGHTS_ENV,
                          upstream='github.com/BMEII-AI/RadImageNet (MIT repository licence; weights via Google Drive; equality to the Kaggle copy not verified)',
                          anchor='docs/handoffs/evidence/2026-10/2026-10-03-anchor-study/anchor_source.py _RAD_ENCODER_SHA256 (same file)',
                          load=load_report, module=RAD_MODULE, provenance='storage/weights/marwanmath__resnet-50-radimagenet-marwan/provenance.json'),
        input=dict(input_policy='center3', norm='radimagenet', rule='window centres from select_centers(within-slot-valid) as EXP101; each window = the '
                   'centre cache plane (uint8/255, bilinear 336->224) repeated x3; train intensity jitter as StudyWindows; then 2x - 1',
                   module=RAD_MODULE),
        folds=folds,
        tags=dict(prefix=TAG_PREFIX, pattern=TAG_PREFIX + '_F<K>[_SMOKE]', oof_dir=OOF_DIR, exp101_prefix=bp['tags']['prefix']),
        remote={**bp['remote'], 'kernel': KERNEL, 'bundle_dataset': 'taeyangg4/' + BUNDLE_DATASET, 'remote_entry': REMOTE_ENTRY, 'runner': RUNNER,
                'weights_dataset': WEIGHTS_DATASET + ' (public, attached to the kernel; ResNet50.pt located by sha256)',
                'relax_env': 'KNEE_REMOTE_RELAX_ENV=1: environment and seed-42 initial state recorded, not enforced, except that backbone keys must '
                             'equal the RadImageNet bytes; independent trajectories, never paired with a local arm (D-B-008)'},
        evaluation={**bp['evaluation'], 'oof_dir': OOF_DIR},
        caveats=[*bp['caveats'], 'RadImageNet weights: Kaggle copy licensed CC-BY-NC-SA-4.0 with no provenance text; same sha256 as the 0.943 anchor encoder',
                 'single seed, single run per fold; recipe not re-tuned for ResNet50 / RadImageNet'],
        derived_from=dict(path=BASE_PROTOCOL + '/manifest.json', sha256=BASE_MANIFEST_SHA256, seed=BASE_SEED, arch=bp['common_args']['arch'],
                          tag_prefix=bp['tags']['prefix'], oof_dir=bp['tags']['oof_dir'], kernel=bp['remote']['kernel'],
                          changed_fields=list(CHANGED_FIELDS), fold_changed_fields=list(FOLD_CHANGED_FIELDS), changed_args=RECIPE_CHANGES))
    dump(destination / 'manifest.json', manifest)
    report = dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, manifest_sha256=sha256(destination / 'manifest.json'), seed=SEED,
                  initial_state_sha256=init, keys=len(keys), backbone_keys=len(backbone_keys), load=load_report,
                  total_steps=manifest['total_steps'], tags=[v['tag'] for v in folds.values()])
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
    if p.get('derived_from', {}).get('sha256') != BASE_MANIFEST_SHA256 or p['tags']['prefix'] != TAG_PREFIX:
        raise ValueError('not the EXP107 registration')
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
    if tag_for(args.fold, args.mode) != args.tag or (not smoke and fold['tag'] != args.tag):
        raise ValueError('tag/fold/mode mismatch')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    weights = Path(args.init_weights)
    if not weights.is_file() or sha256(weights) != p['init_weights']['sha256'] or p['init_weights']['sha256'] != RADIMAGENET_SHA256:
        raise ValueError('init weights differ from the registered RadImageNet file')
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


def runner_argv(p, fold, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None, weights=None):
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
             '--eval-labels', str(relpath(root, p['paths']['eval_labels'])), '--init-weights', str(weights or weights_path(root))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if mode == 'smoke':
        smoke = {**dict(device='cpu', amp='none', eval_amp='none', train=24, val=4, epochs=1), **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), smoke[key])
        argv += ['--smoke-train', str(smoke['train']), '--smoke-val', str(smoke['val']), '--smoke-epochs', str(smoke['epochs'])]
    return argv


def local_env():
    return base.local_env()


def validate_cli(fold):
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path)."""
    argv = runner_argv(_protocol(), fold, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-6000:]); print(r.stderr[-3000:], file=sys.stderr)
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
        prov = s['provenance']
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], val=s['val'], seed=prov['seed'],
                      initial_state_sha256=prov['initial_state_sha256'], initial_state_matches_registered=prov['initial_state_matches_registered'],
                      init_weights_sha256=prov['init_weights']['weights_sha256'], input_policy=prov['input_policy'], norm=prov['norm'], history=s['history'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ verification, download and assembly
def verify_run(run, p, k, manifest_sha):
    """Problems (empty list = complete and consistent) for one EXP107 fold directory: the EXP101 checks plus tag, weights and input."""
    problems = base.verify_run(run, p, k, manifest_sha)
    if problems:
        return problems
    run = Path(run)
    prov = read(run / 'provenance.json')
    if prov.get('seed') != SEED or prov.get('arch') != EXPECTED_RECIPE['arch']:
        problems.append(f"fold {k}: provenance seed/arch = {prov.get('seed')!r}/{prov.get('arch')!r}")
    if (prov.get('init_weights') or {}).get('weights_sha256') != RADIMAGENET_SHA256:
        problems.append(f'fold {k}: init weights are not the registered RadImageNet file')
    if prov.get('input_policy') != 'center3' or prov.get('norm') != 'radimagenet':
        problems.append(f'fold {k}: input policy / norm differ')
    if any(x.startswith('backbone.') for x in prov.get('initial_state_mismatched_keys') or []):
        problems.append(f'fold {k}: backbone initial state differs from the RadImageNet bytes')
    s = read(run / 'summary.json')
    if (s.get('args') or {}).get('tag', run.name) != tag_for(k) or run.name != tag_for(k):
        problems.append(f'fold {k}: run directory / tag is not {tag_for(k)}')
    return problems


def fetch(kernel=KERNEL, dest=None, retries=3, root=ROOT):
    """Download the kernel output fold by fold (small pages, --force) until every fold verifies by hash or retries run out."""
    root = Path(root); p = _protocol(root); manifest_sha = sha256(root / PROTOCOL / 'manifest.json')
    dest = Path(dest or root / '.local' / 'kaggle-out' / kernel.split('/')[-1]).resolve(); dest.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    st = subprocess.run([*kaggle_cli(), 'kernels', 'status', kernel], capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    print((st.stdout or st.stderr).strip()[-400:])
    state = {}
    for pattern in ['exp101-status'] + [tag_for(k) for k in FOLDS]:  # a tag pattern also matches logs/<tag>.console.log
        for _ in range(retries):
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
    status_file = dest / STATUS_FILE
    status = read(status_file) if status_file.exists() else {}
    if status and status.get('protocol') not in (None, PROTOCOL):
        state['exp101-status'] = f"status file belongs to {status.get('protocol')}, not {PROTOCOL}"
    print(json.dumps(dict(dest=str(dest), folds=state, kernel_status=status.get('state'), job=status.get('job')), indent=1))
    return state


def assemble(runs_dir=None, k36_override=None, root=ROOT):
    import pandas as pd
    root = Path(root); k36_override = k36_override or {}
    manifest_path = root / PROTOCOL / 'manifest.json'
    p = read(manifest_path); manifest_sha = sha256(manifest_path)
    runs = Path(runs_dir).resolve() if runs_dir else root / '.local' / 'kaggle-out' / KERNEL.split('/')[-1]
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
                          seed=prov.get('seed'), initial_state_sha256=prov.get('initial_state_sha256'),
                          initial_state_matches_registered=prov.get('initial_state_matches_registered'), environment=prov.get('environment'),
                          remote_relaxed=prov.get('remote_relaxed'), init_weights_sha256=(prov.get('init_weights') or {}).get('weights_sha256'))
    if problems:
        print(json.dumps(dict(status='INCOMPLETE', problems=problems), indent=1))
        raise SystemExit(2)
    result = k99.assemble_and_score(frames24, frames36, val_ids, root / DREAD)
    out.mkdir(parents=True)
    result['oof_k24'].to_csv(out / 'oof_k24.csv', index=False)
    if result['oof_k36'] is not None:
        result['oof_k36'].to_csv(out / 'oof_k36.csv', index=False)
    body = dict(schema='exp107-oof-v1', label_set=LABEL_SET, label_set_sha256=p['label_set']['sha256'], observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                protocol=PROTOCOL, protocol_sha256=manifest_sha, seed=SEED, exp101_protocol_sha256=BASE_MANIFEST_SHA256, runs_dir=display_path(runs, root),
                rows=int(len(result['oof_k24'])), sources={str(k): v for k, v in sources.items()},
                outputs={f: sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv') if (out / f).exists()},
                scoring=dict(labels=DREAD, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py'),
                metrics=result['metrics'], k36_missing_folds=result['k36_missing'],
                caveats=['weak Dread labels', 'fold 0 val = exposed G1 V888; folds 1-4 models saw G1 val in training', 'pooled AUC mixes 5 models',
                         'Kaggle T4 trajectories (KNEE_REMOTE_RELAX_ENV); compare with local arms only as separate families',
                         'RadImageNet ResNet50, centre slice x3, 2x-1 normalisation; RN_clip recipe otherwise'])
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    md = k99.result_markdown(LABEL_SET, body).replace(f'# EXP099 B8 5-fold OOF ({LABEL_SET})', f'# EXP107 RadImageNet ResNet50 5-fold OOF ({LABEL_SET}, Kaggle T4)', 1)
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
    s = sub.add_parser('assemble')
    s.add_argument('--runs-dir', default=None, help=f'directory holding {TAG_PREFIX}_F<K>/ (default .local/kaggle-out/{KERNEL.split("/")[-1]})')
    s.add_argument('--k36', action='append', default=[], help='FOLD=PATH oof_k36.csv override')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Kaggle CLI output can contain characters the cp949 console cannot encode (D-B-015)
        stream.reconfigure(encoding='utf-8', errors='replace')
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
