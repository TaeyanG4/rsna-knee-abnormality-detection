"""EXP106: the CoAtNet B8 recipe on EXP099 fold 0 (= G1 T3455/V888, = EXP094L_B8_SEED42) trained for 16 epochs instead of 8 (H-B-23, local GPU).

  python tools/exp106_b8_e16.py prepare                 # register EXP106_B8_E16_PROTOCOL_V1 (CPU; recomputes the CoAtNet seed-42 init hash, must be B)
  python tools/exp106_b8_e16.py argv [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp106_b8_e16.py validate                # runner --validate-only on CPU, no output
  python tools/exp106_b8_e16.py smoke --out DIR [--train 24 --val 4 --epochs 1 --workers 2]   # 3-update CPU fit, DIR outside artifacts/
  python tools/exp106_b8_e16.py queue                   # write the gpu_queue entry to <protocol>/queue.json (nothing is started)
  python tools/exp106_b8_e16.py verify [--run DIR]      # check a finished EXP106_B8_E16_F0 run against the protocol

Single changed variable: epochs 8 -> 16 (fixed_epoch 8 -> 16). Consequences of that change only: OneCycleLR(pct_start 0.15) spans
431 * 16 = 6896 updates instead of 3448 and the 1% FP16 skip gate is 6896 // 100 = 68. Everything else is the EXP099 fold-0 / EXP094 V4 arm
seed42 recipe, re-verified at prepare against EXP099_KFOLD_PROTOCOL_V1 (manifest pinned by sha256), EXP094_B8_VARIANT_PROTOCOL_V4 (pinned)
and the finished reference run EXP094L_B8_SEED42 (provenance and summary args): coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k, 224 px, K12 train /
K24 eval windows (within-slot-valid), bs 8, FP16 + GradScaler init 1024, FP16 eval, strict runtime, gradient checkpointing, workers 12,
bb_lr 3e-5, head_lr 1e-3, wd 0.02, fixed-final checkpoint, Dread labels, pos_weight = the B8 fold-0 vector, seed 42, initial state = B
(e4ea3ae6..., recomputed on CPU at prepare and enforced by the runner). After training the same final model is also evaluated with 36 windows
(oof_k36.csv; D-B-012).

Training runs the shared EXP101 runner file unchanged (training/runners/exp101_kfold_arch.py, whose loop is the EXP099 runner's, itself the
EXP094 V4 loop) through training/runners/exp106_b8_e16.py, which only points the runner's guard import at this module. This module therefore
re-exports every helper the shared runner reads from its guard. The fold files of EXP099 are referenced by path and sha256, not copied.
Nothing here trains on or touches the GPU.
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
# names the shared runner reads from its guard module (training/runners/exp101_kfold_arch.py uses `guard.<name>`)
from tools.exp101_kfold_arch import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                     CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
                                     FOLD_PROTOCOL, DREAD, LABEL_SET, KEYS_FILE, FROZEN_KEYS, SMOKE_ALLOWED, EXTRA_EVAL, RUN_FILES, GPU_PYTHON,
                                     recipe_args, fold_contract, state_hashes, mismatched_keys, dump, train_python, within, display_path,
                                     build_initial_model, _set_flag)

SCHEMA = 'exp106-b8-e16-v1'
PROTOCOL = 'artifacts/experiments/EXP106_B8_E16_PROTOCOL_V1'
RUNNER = 'training/runners/exp106_b8_e16.py'
SHARED_RUNNER = base.RUNNER
BASE_TOOL = base.TOOL
TOOL = 'tools/exp106_b8_e16.py'
PLAN_ITEM = 'H-B-23'
FOLD_MANIFEST_SHA256 = '707205dbf69ffcaf1c8fdd3df84aa90a2cd63464408d4cfa3a1bdce6e8aee53d'  # EXP099_KFOLD_PROTOCOL_V1/manifest.json
V4_PROTOCOL = 'artifacts/experiments/EXP094_B8_VARIANT_PROTOCOL_V4/manifest.json'
V4_MANIFEST_SHA256 = 'ccac9b24d5cac9c9c87c9eee6cc47b6e3af13dcb50a0176eee5420f0f715205a'
V4_ARM = 'seed42'
BASE_RUN = 'artifacts/experiments/EXP094L_B8_SEED42'
BASE_TAG = 'EXP094L_B8_SEED42'
BASE_RUN_FILES = ('provenance.json', 'summary.json', 'output-inventory.json')
FOLD0_REUSE = FOLD_PROTOCOL + '/fold0-reuse.json'
B_INITIAL_STATE = 'e4ea3ae625eacde8e3ab563b6c663d715600a0cea039e1c9ee9d201367d5b38e'
B_K36 = 'artifacts/experiments/EXP096_K_SWEEP/B_K36.csv'
SEED = 42
FOLDS = [0]
BASE_EPOCHS = 8
EPOCHS = 16
TAG_PREFIX = 'EXP106_B8_E16'
TAG = TAG_PREFIX + '_F0'
FULL_WORKERS = 12
# the B8 recipe (EXP099 common_args without the workers resource argument), written out so a drifted source cannot change it silently
BASE_RECIPE = dict(arch='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', res=224, k=12, k_eval=24, epochs=BASE_EPOCHS, fixed_epoch=BASE_EPOCHS, bs=8, accum=1,
                   grad_ckpt=True, amp='fp16', amp_init_scale=1024.0, eval_amp='fp16', runtime_policy='strict', norm='imagenet', bb_lr=3e-5,
                   head_lr=1e-3, wd=0.02, pretrained='timm', window_policy='within-slot-valid', device='cuda')
EXPECTED_RECIPE = {**BASE_RECIPE, 'epochs': EPOCHS, 'fixed_epoch': EPOCHS}
RESOURCE_ARGS = dict(workers=dict(default=FULL_WORKERS, min=0, max=16, full=FULL_WORKERS,
                                  note='DataLoader workers; full runs use the B8 value 12 (EXP094L_B8_SEED42 requested_workers); smoke may use fewer. '
                                       'Window sampling is sha256(seed, epoch, uid)-deterministic and batch order comes from a seeded generator'))
QUEUE_FILE = PROTOCOL + '/queue.json'
QUEUE_AFTER = 'EXP094L_B8_SEED43'
REFERENCE_SECONDS = 2677  # EXP094L_B8_SEED42: 8 epochs, K24 evaluation only, same machine
SINGLE_CHANGE = ('epochs 8 -> 16 (fixed_epoch 8 -> 16): OneCycleLR(pct_start 0.15) total_steps = len_train_loader * 16 = 6896 instead of 3448 '
                 '(warm-up 15% of the longer run), FP16 skip gate = 6896 // 100 = 68 instead of 34; nothing else changes')


def tag_for(fold, mode='full'):
    if int(fold) not in FOLDS:
        raise ValueError('EXP106 registers fold 0 only')
    return TAG + ('_SMOKE' if mode == 'smoke' else '')


def _protocol(root=ROOT):
    return read(Path(root) / PROTOCOL / 'manifest.json')


def e16_recipe(common8):
    """The EXP106 common_args from the registered B8 common_args: epochs and fixed_epoch 8 -> 16, nothing else."""
    args = recipe_args(common8)
    if args != BASE_RECIPE:
        raise ValueError('B8 common_args differ from the registered recipe: ' + str({k: (args[k], BASE_RECIPE.get(k)) for k in args if args[k] != BASE_RECIPE.get(k)}))
    out = recipe_args({**args, 'epochs': EPOCHS, 'fixed_epoch': EPOCHS})
    if {k for k in out if out[k] != args[k]} != {'epochs', 'fixed_epoch'}:
        raise AssertionError('more than the epoch count changed')
    return out


def estimate_seconds(history, epochs=EPOCHS, k24_seconds=47, k36_seconds=64, setup_seconds=15):
    """Training time of the reference epochs scaled to `epochs` plus K24 + K36 evaluation (EXP099_B8_DISTILL_V1_F0: 47 s / 64 s on fold 0)."""
    per_epoch = sum(h['epoch_seconds'] for h in history) / len(history)
    return round(per_epoch * epochs + k24_seconds + k36_seconds + setup_seconds)


# ------------------------------------------------------------------ prepare
def prepare(root=ROOT, model_factory=None):
    import numpy as np
    root = Path(root)
    destination = root / PROTOCOL
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + PROTOCOL)
    fold_manifest_path = root / FOLD_PROTOCOL / 'manifest.json'
    if sha256(fold_manifest_path) != FOLD_MANIFEST_SHA256:
        raise ValueError('EXP099 manifest changed')
    fp = read(fold_manifest_path)
    if fp['schema'] != base.FOLD_SCHEMA or fp['seed'] != SEED or fp['variant'] != 'seed' or fp['initial_state_sha256'] != B_INITIAL_STATE:
        raise ValueError('unexpected EXP099 registration')
    if sha256(root / fp['paths']['folds']) != fp['folds_csv_sha256']:
        raise ValueError('EXP099 folds.csv changed')
    for path, expected in fp['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('EXP099 binding changed: ' + path)
    for name, expected in fp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    reg, reg_file = k99.load_registration(root, fp, LABEL_SET, fold_manifest_path)
    if reg['path'] != DREAD or fp['label_sets'][LABEL_SET]['path'] != DREAD:
        raise ValueError('EXP099 dread registration points elsewhere')
    # the B8 recipe: EXP099 common_args == EXP094 V4 common_args (arm seed42) == the args of the finished EXP094L_B8_SEED42 run
    if sha256(root / V4_PROTOCOL) != V4_MANIFEST_SHA256:
        raise ValueError('EXP094 V4 manifest changed')
    v4 = read(root / V4_PROTOCOL)
    arm = v4['arms'][V4_ARM]
    if v4['common_args'] != fp['common_args'] or v4['initial_state_sha256'] != B_INITIAL_STATE or arm['seed'] != SEED or arm['variant'] != 'seed' \
            or arm['slot_dropout'] != 0.0 or arm['tags']['full'] != BASE_TAG:
        raise ValueError('EXP094 V4 arm seed42 differs from EXP099')
    if fp['common_args']['workers'] != FULL_WORKERS:
        raise ValueError('B8 workers changed')
    common8 = recipe_args(fp['common_args'])
    common = e16_recipe(fp['common_args'])
    if common != EXPECTED_RECIPE:
        raise ValueError('EXP106 recipe differs from the registered 16-epoch recipe')
    labels, semantic = k99._load_labels(root / DREAD)
    if semantic != reg['semantic_sha256'] or sha256(root / DREAD) != reg['sha256']:
        raise ValueError('Dread file differs from the EXP099 registration')
    canonical = canonical_order(read(root / fp['paths']['base_ordered_ids']))
    corpus_dir = root / fp['paths']['corpus']
    if sha256(corpus_dir / 'manifest.json') != fp['corpus_manifest_sha256']:
        raise ValueError('corpus manifest differs from EXP099')
    corpus_ids = np.load(corpus_dir / 'ids.npy', allow_pickle=True).astype(str).tolist()
    f99 = fp['folds']['0']
    for key in ('ordered_ids', 'groups'):
        if sha256(root / f99[key]['path']) != f99[key]['sha256']:
            raise ValueError(f'EXP099 fold 0 {key} file changed')
    ordered = read(root / f99['ordered_ids']['path'])
    if uid_hash(ordered['train']) != f99['ordered_ids']['train_sha256'] or uid_hash(ordered['val']) != f99['ordered_ids']['val_sha256']:
        raise ValueError('EXP099 fold 0 ordered ids changed')
    train, val = validate_fold_membership(corpus_ids, ordered, read_rows(root / f99['groups']['path']), canonical)
    if (len(train), len(val)) != (f99['n_train'], f99['n_val']) or (len(train), len(val)) != (3455, 888):
        raise ValueError('fold 0 is not the G1 T3455/V888 split')
    pw = pos_weight(labels, train)
    for name, vec in (('EXP099 fold 0', f99['pos_weight']), ('dread registration', reg['pos_weight']['0']), ('EXP094 V4', v4['loss']['baseline_pos_weight'])):
        if not np.array_equal(pw, np.asarray(vec, np.float32)):
            raise ValueError('fold 0 pos_weight differs from ' + name)
    c8 = fold_contract(len(train), common8)
    if any(c8[x] != f99[x] for x in c8):
        raise ValueError('8-epoch fold-0 contract differs from EXP099')
    c16 = fold_contract(len(train), common)
    if c16['steps'] != c8['steps'] * EPOCHS // BASE_EPOCHS or c16['max_fp16_skips'] != c16['steps'] // 100:
        raise AssertionError('16-epoch contract rule drifted')
    # the finished 8-epoch reference run B (EXP094L_B8_SEED42 = the EXP099 fold-0 member)
    reuse = read(root / FOLD0_REUSE)
    if reuse['status'] != 'REUSED_AS_FOLD0' or reuse['run'] != BASE_RUN or reuse['protocol_sha256'] != FOLD_MANIFEST_SHA256 or not all(reuse['checks'].values()):
        raise ValueError('EXP099 fold-0 reuse record does not name ' + BASE_RUN)
    prov = read(root / BASE_RUN / 'provenance.json')
    summ = read(root / BASE_RUN / 'summary.json')
    inv = read(root / BASE_RUN / 'output-inventory.json')
    if reuse['outputs']['provenance.json']['sha256'] != sha256(root / BASE_RUN / 'provenance.json') or \
            reuse['outputs']['summary.json']['sha256'] != sha256(root / BASE_RUN / 'summary.json'):
        raise ValueError('reference run files differ from the EXP099 reuse record')
    expect = dict(initial_state_sha256=B_INITIAL_STATE, initial_state_matches_B=True, seed=SEED, variant='seed', slot_dropout=0.0, training_mode='full',
                  checkpoint_selection='fixed-final', loss_policy='clipped_positive_term', train_ids_ordered_sha256=f99['ordered_ids']['train_sha256'],
                  eval_ids_ordered_sha256=f99['ordered_ids']['val_sha256'], label_hash=reg['semantic_sha256'], eval_label_hash=reg['semantic_sha256'],
                  corpus_manifest_sha256=fp['corpus_manifest_sha256'], scheduler_total_steps=c8['steps'], requested_workers=FULL_WORKERS,
                  environment_matches_baseline=True, environment=fp['environment'], code_sha256=fp['training_code_sha256'])
    bad = [k for k, v in expect.items() if prov.get(k) != v]
    if bad or not np.array_equal(np.asarray(prov['actual_pos_weight'], np.float32), pw):
        raise ValueError('reference run provenance differs: ' + ', '.join(bad or ['actual_pos_weight']))
    run_args = summ['args']
    bad = [k for k in FROZEN_KEYS if run_args.get(k) != common8[k]] + [k for k in ('workers', 'seed') if run_args.get(k) != dict(workers=FULL_WORKERS, seed=SEED)[k]]
    if bad or summ['optimizer_updates'] != c8['steps'] or summ['selected_epoch_1based'] != BASE_EPOCHS or (summ['train'], summ['val']) != (3455, 888):
        raise ValueError('reference run args differ: ' + ', '.join(bad or ['updates/epoch/counts']))
    model = (model_factory or build_initial_model)(common['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init, keys = state_hashes(model)
    if init != B_INITIAL_STATE:
        raise ValueError(f'CPU seed-42 initial state {init} is not B ({B_INITIAL_STATE})')
    import timm
    destination.mkdir(parents=True)
    dump(destination / KEYS_FILE, dict(schema='exp101-initial-state-keys-v1', arch=common['arch'], seed=SEED, combined_sha256=init, keys=keys,
                                       rule='per state_dict key: sha256 of the contiguous CPU tensor bytes; combined = sha256 over sorted (key bytes + tensor bytes)'))
    bindings = [SHARED_RUNNER, BASE_TOOL, RUNNER, TOOL, 'tools/exp099_kfold.py', 'tools/exp087_training.py', 'tools/exp084_group_loss.py',
                FOLD_PROTOCOL + '/manifest.json', fp['paths']['folds'], fp['paths']['base_ordered_ids'], fp['paths']['base_groups'],
                display_path(reg_file, root), DREAD, V4_PROTOCOL, FOLD0_REUSE, *(BASE_RUN + '/' + n for n in BASE_RUN_FILES), PROTOCOL + '/' + KEYS_FILE]
    fold = dict(fold=0, tag=TAG, ordered_ids=dict(f99['ordered_ids']), groups=dict(f99['groups']), n_train=len(train), n_val=len(val),
                train_components=f99['train_components'], val_components=f99['val_components'], **c16, pos_weight=pw.tolist(),
                pos_weight_equals_exp099=True, pos_weight_equals_b8=True, val_positive_counts=f99['val_positive_counts'], macro12_valid=f99['macro12_valid'],
                contract_8_epochs=c8)
    history = read(root / BASE_RUN / 'history.json') if (root / BASE_RUN / 'history.json').exists() else summ['history']
    manifest = dict(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item=PLAN_ITEM,
        hypothesis='B8 trained for 16 instead of 8 epochs (OneCycle over the longer run, everything else fixed) raises fold-0 G1 V888 macro12 by >= 0.006',
        source_bindings={s: sha256(root / s) for s in bindings},
        training_code_sha256=fp['training_code_sha256'], environment=fp['environment'], corpus_manifest_sha256=fp['corpus_manifest_sha256'],
        excluded_uids=fp['excluded_uids'],
        recipe=dict(source=FOLD_PROTOCOL + '/manifest.json', source_sha256=FOLD_MANIFEST_SHA256, source_common_args=fp['common_args'],
                    v4_protocol=V4_PROTOCOL, v4_sha256=V4_MANIFEST_SHA256, v4_arm=V4_ARM, base_run=BASE_RUN, base_tag=BASE_TAG, base_recipe=common8,
                    base_output_inventory=inv, single_change=SINGLE_CHANGE,
                    changes=[SINGLE_CHANGE,
                             'workers: resource argument (full runs must use 12 = B8; only CPU smoke may use fewer), frozen in B8 common_args',
                             'K36 extra evaluation of the same final model (oof_k36.csv), as the EXP099 runner',
                             f'runner: {SHARED_RUNNER} unchanged (loop = EXP099 runner = EXP094 V4 seed arm) through {RUNNER} with this guard']),
        common_args=common, seed=SEED, resource_args=RESOURCE_ARGS, smoke_allowed=SMOKE_ALLOWED,
        initial_state_sha256=init,
        initial_state=dict(keys_file=PROTOCOL + '/' + KEYS_FILE, keys=len(keys), computed_on='cpu', seed=SEED, arch=common['arch'], torch=torch.__version__,
                           timm=timm.__version__, rule='seed_all(42) after worker_init(0), build_model(arch, pretrained timm); sorted state_dict sha256',
                           equals_B=True, B=B_INITIAL_STATE, base_run_initial_state_sha256=prov['initial_state_sha256'],
                           enforcement='runner: seed 42 must match (no remote relaxation for this local protocol)'),
        fold_protocol=dict(path=FOLD_PROTOCOL + '/manifest.json', sha256=FOLD_MANIFEST_SHA256, folds_csv_sha256=fp['folds_csv_sha256'],
                           label_registration=dict(path=display_path(reg_file, root), sha256=sha256(reg_file)), reuse='fold-0 files referenced by path + sha256, not copied'),
        paths=dict(corpus=fp['paths']['corpus'], labels=DREAD, eval_labels=DREAD, base_ordered_ids=fp['paths']['base_ordered_ids'],
                   base_groups=fp['paths']['base_groups'], folds=fp['paths']['folds']),
        label_set=dict(name=LABEL_SET, path=DREAD, sha256=reg['sha256'], semantic_sha256=reg['semantic_sha256']),
        folds={'0': fold}, total_steps=c16['steps'],
        pos_weight_rule=fp['pos_weight_rule'],
        contract_rule='len_train_loader = n_train // bs (drop_last); steps = len_train_loader * epochs (accum 1) = scheduler total_steps; '
                      'FP16 skip gate: skipped > steps // 100 fails; eval batch bs // 2',
        extra_eval=EXTRA_EVAL, tags=dict(prefix=TAG_PREFIX, full=TAG, pattern=TAG_PREFIX + '_F0[_SMOKE]'),
        queue=dict(file=QUEUE_FILE, after=QUEUE_AFTER, continue_on_failure=True, python=str(GPU_PYTHON),
                   environment='queue runner started with OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 (as LOCAL_GPU_CHAIN_V8)'),
        budget=dict(reference_run=BASE_RUN, reference_seconds=REFERENCE_SECONDS, reference_epoch_seconds=[h['epoch_seconds'] for h in history],
                    estimate_seconds=estimate_seconds(history), rule='mean reference epoch seconds * 16 + K24 47 s + K36 64 s + 15 s setup'),
        evaluation=dict(compare_with=dict(k24=BASE_RUN + '/oof_best.csv', k36=B_K36), labels=DREAD, tool='tools/g1_compare.py',
                        rule='float32 then numpy.rint (tools/g1_compare.py); same machine, seed, data order and windows as B (D-B-017)',
                        metric='macro12 on G1 V888 (fold 0), K24 and K36',
                        decision='adopt 16 epochs for the 5-fold models only if macro12 >= B + 0.006 (above the H-B-17 seed spread)'),
        caveats=['weak Dread labels; G1 V888 has been inspected many times (exposed validation)', 'one seed; a single paired contrast against B',
                 'the first 8 epochs see the same windows and batch order as B but a different learning-rate schedule'])
    dump(destination / 'manifest.json', manifest)
    report = dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, manifest_sha256=sha256(destination / 'manifest.json'), initial_state_sha256=init,
                  equals_B=init == B_INITIAL_STATE, tag=TAG, steps=c16['steps'], max_fp16_skips=c16['max_fp16_skips'], len_train_loader=c16['len_train_loader'],
                  estimate_seconds=manifest['budget']['estimate_seconds'])
    print(json.dumps(report, indent=1))
    return manifest


# ------------------------------------------------------------------ runner guard (the EXP101 checks with EXP106 constants)
def validate_arguments(args, root):
    """Called by the shared runner before any training import. Returns (protocol, fold spec, ordered ids, output root)."""
    root = Path(root)
    manifest_path = Path(args.protocol)
    p = read(manifest_path)
    if p.get('schema') != SCHEMA or manifest_path.resolve() != (root / PROTOCOL / 'manifest.json').resolve():
        raise ValueError('wrong protocol')
    if p.get('seed') != SEED or p['tags']['full'] != TAG or p['common_args'] != EXPECTED_RECIPE or p['initial_state_sha256'] != B_INITIAL_STATE:
        raise ValueError('not the EXP106 registration')
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
    if not smoke and args.workers != w['full']:
        raise ValueError(f"full runs use the B8 workers {w['full']}")
    if str(args.fold) not in p['folds']:
        raise ValueError('unregistered fold')
    fold = p['folds'][str(args.fold)]
    if tag_for(args.fold, args.mode) != args.tag or (not smoke and fold['tag'] != args.tag):
        raise ValueError('tag/fold/mode mismatch')
    for key in ('labels', 'eval_labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('EXP106 is a local same-machine contrast; KNEE_REMOTE_RELAX_ENV is not allowed')
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
    elif out_root != default_root:
        raise ValueError('full runs write to artifacts/experiments (no --output-root)')
    return p, fold, ordered, out_root


def runner_argv(p, fold=0, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None):
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


def queue_entry(p, root=ROOT):
    """The gpu_queue job (format of LOCAL_GPU_CHAIN_V8.json entries): the registered full fold-0 run on the GPU Python, after EXP094L_B8_SEED43."""
    q = p['queue']
    return dict(tag=TAG, command=runner_argv(p, 0, 'full', root=root, python=q['python']), after=q['after'], continue_on_failure=q['continue_on_failure'])


def write_queue(root=ROOT):
    root = Path(root)
    entry = queue_entry(_protocol(root), root)
    dump(root / QUEUE_FILE, [entry])
    print(json.dumps(dict(queue=QUEUE_FILE, sha256=sha256(root / QUEUE_FILE), tag=entry['tag'], after=entry['after']), indent=1))
    return entry


def local_env():
    return base.local_env()


def validate_cli():
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path)."""
    argv = runner_argv(_protocol(), 0, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(out, train, val, epochs, workers):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    argv = runner_argv(_protocol(), 0, 'smoke', workers=workers, output_root=out, smoke=dict(train=train, val=val, epochs=epochs))
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for(0, 'smoke')
    report = dict(returncode=r.returncode, seconds=round(time.time() - t0), run=str(run), files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json')
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], val=s['val'], seed=s['provenance']['seed'],
                      arch=s['provenance']['arch'], initial_state_sha256=s['provenance']['initial_state_sha256'],
                      initial_state_matches_registered=s['provenance']['initial_state_matches_registered'], history=s['history'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ verification of the finished run
def verify_run(run, p, k=0, manifest_sha=None, root=ROOT):
    """Problems (empty list = complete and consistent) for the EXP106 run directory: the EXP101 checks plus local-contrast checks."""
    root = Path(root)
    manifest_sha = manifest_sha or sha256(root / PROTOCOL / 'manifest.json')
    problems = base.verify_run(run, p, k, manifest_sha)
    if problems:
        return problems
    run = Path(run)
    prov, s = read(run / 'provenance.json'), read(run / 'summary.json')
    expect = dict(seed=SEED, arch=p['common_args']['arch'], initial_state_sha256=B_INITIAL_STATE, initial_state_matches_registered=True,
                  environment_matches_baseline=True, remote_relaxed=False, requested_workers=FULL_WORKERS, scheduler_total_steps=p['folds'][str(k)]['steps'],
                  max_fp16_skips=p['folds'][str(k)]['max_fp16_skips'], label_set=LABEL_SET)
    for key, value in expect.items():
        if prov.get(key) != value:
            problems.append(f'provenance {key} = {prov.get(key)!r}, expected {value!r}')
    if run.name != tag_for(k) or (s.get('args') or {}).get('tag') != tag_for(k):
        problems.append(f'run directory / tag is not {tag_for(k)}')
    if len(s.get('history') or []) != p['common_args']['epochs']:
        problems.append(f"history has {len(s.get('history') or [])} epochs, expected {p['common_args']['epochs']}")
    return problems


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('argv'); s.add_argument('--mode', choices=['full', 'smoke'], default='full')
    s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=None)
    sub.add_parser('validate')
    s = sub.add_parser('smoke'); s.add_argument('--out', required=True)
    s.add_argument('--train', type=int, default=24); s.add_argument('--val', type=int, default=4); s.add_argument('--epochs', type=int, default=1)
    s.add_argument('--workers', type=int, default=2)
    sub.add_parser('queue')
    s = sub.add_parser('verify'); s.add_argument('--run', default=None, help=f'run directory (default artifacts/experiments/{TAG})')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')
    if a.command == 'prepare':
        for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():  # before torch/timm/huggingface_hub are imported
            os.environ[k] = v
        prepare()
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), 0, a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli())
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.out, a.train, a.val, a.epochs, a.workers))
    elif a.command == 'queue':
        write_queue()
    else:
        problems = verify_run(Path(a.run) if a.run else ROOT / 'artifacts' / 'experiments' / TAG, _protocol())
        print(json.dumps(dict(run=a.run or TAG, status='OK' if not problems else 'PROBLEMS', problems=problems), indent=1))
        sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
