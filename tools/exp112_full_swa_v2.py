"""EXP112 V2 (H-B-33): EXP112 with the public v5 SWA epochs: 16 epochs, model_best.pt = fixed SWA of 1-based epochs 13, 15, 16.

  python tools/exp112_full_swa_v2.py prepare                     # register EXP112_FULL_SWA_PROTOCOL_V2 + queue-entry.json (CPU; init must be B)
  python tools/exp112_full_swa_v2.py prepare-scratch --dest DIR [--tiny]   # engineering registration outside artifacts/ (smoke runs only)
  python tools/exp112_full_swa_v2.py argv [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp112_full_swa_v2.py validate                    # runner --validate-only on CPU (no output, no CUDA)
  python tools/exp112_full_swa_v2.py smoke --out DIR [--protocol MANIFEST] [--train 8 --epochs 3 --swa 1,2,3 --workers 0]
  python tools/exp112_full_swa_v2.py verify [--run DIR] [--quick]
  python tools/exp112_full_swa_v2.py estimate
  python tools/exp112_full_swa_v2.py versions                    # the V1 / V2 epoch, SWA and protocol constants

Why V2 (decision 2026-10-07 17:55 KST, source D-A-050 + independent review M1, scratchpad review_exp112_113.md): the public v5 checkpoint's
swa_over [14, 12, 15] is ZERO-based. The public trainer records "epoch": ep for ep in range(a.epochs) and swa_over = [int(t["epoch"]) ...]
(docs/handoffs/evidence/2026-10/2026-10-05-preprocessing-survey-v1/notebooks/dreaddevelopment__knee-mri-training-the-twelve-finding-model/
code.py.txt:509,544,622; training/src/raptor_train.py:241,260; HA14 audit.md:66), i.e. 1-based epochs 15, 13, 16 of a run of >= 16 epochs.
V1 (EXP112_FULL_SWA_PROTOCOL_V1: 15 epochs, SWA of 1-based 12/14/15) averaged one pass earlier and trained one pass shorter.
V2 changes exactly that: epochs 15 -> 16 (OneCycle over all 16 epochs: 542 * 16 = 8672 updates, FP16 skip gate 86) and SWA members 1-based
13, 15, 16 (0-based loop indices 12, 14, 15 = the public swa_over). Everything else is V1 = the EXP099 B8 recipe on all 4,343 G1 studies.

Selectable versions: VERSIONS below holds both protocols' constants. V1 stays reproducible with its own, unchanged files
(tools/exp112_full_swa.py + training/runners/exp112_full_swa.py, sha256-bound by the V1 manifest and queued in LOCAL_GPU_CHAIN_V14); this
module re-uses every version-independent V1 helper by import (training set, SWA mean, BatchNorm pass, release format) and redefines only
the version-dependent parts. The V2 runner (training/runners/exp112_full_swa_v2.py) is the V1 loop with this guard and one fix (review m1):
model_best.pt records swa_over in the public 0-based convention next to swa_epochs_1based / epoch_1based and an explicit numbering note.
Nothing here trains on or touches the GPU.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp112_full_swa as v1  # noqa: E402
from tools import exp101_kfold_arch as base  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
# version-independent helpers and constants, re-used from V1 (the runner reads them as guard.<name>)
from tools.exp112_full_swa import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                   CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
                                   FOLD_PROTOCOL, DREAD, LABEL_SET, KEYS_FILE, FROZEN_KEYS, SMOKE_ALLOWED, GPU_PYTHON, recipe_args,
                                   fold_contract, state_hashes, mismatched_keys, dump, train_python, within, display_path, build_initial_model,
                                   _set_flag, FOLD_MANIFEST_SHA256, V4_PROTOCOL, V4_MANIFEST_SHA256, V4_ARM, B_INITIAL_STATE, SEED, N_STUDIES,
                                   N_GOLD, BASE_EPOCHS, FULL_WORKERS, TRAIN_IDS_FILE, EPOCH_DIR, INVENTORY, BASE_RECIPE, RESOURCE_ARGS, LOOP,
                                   RELEASE_KEYS, RELEASE_REQUIRED, RELEASE_META, RELEASE_CODE, RUN_FILES, REFERENCE_RUNS, OVERHEAD_SECONDS,
                                   SMOKE_DEFAULTS, epoch_file, parse_epochs, check_swa_epochs, full_contract, canonical_json_sha256,
                                   full_training_set, check_no_gold, validate_training_set, state_dict_sha256, swa_average, batchnorm_modules,
                                   recompute_bn, release_payload, release_check, tiny_smoke_model, build_tiny_initial, build_run_model,
                                   reference_runs, estimate_from)

VERSION = 'v2'
SCHEMA = 'exp112-full-swa-v2'
PROTOCOL = 'artifacts/experiments/EXP112_FULL_SWA_PROTOCOL_V2'
RUNNER = 'training/runners/exp112_full_swa_v2.py'
TOOL = 'tools/exp112_full_swa_v2.py'
PLAN_ITEM = 'H-B-33'
EPOCHS = 16
SWA_EPOCHS = (13, 15, 16)  # 1-based: epoch e = state after e full passes over the training loader (loop index e - 1)
PUBLIC_SWA_OVER = (14, 12, 15)  # public v5 checkpoint field, 0-based loop indices (D-A-050)
BN_DRAW_EPOCH = EPOCHS  # 0-based window-draw index for the BatchNorm pass: 16 = the draw after the last training epoch (15)
TAG = 'EXP112_B8_FULL_SWA_V2'
QUEUE_FILE = PROTOCOL + '/queue-entry.json'
QUEUE_AFTER = 'EXP099_B8_FLIGHT_V4_F4'
EXPECTED_RECIPE = {**BASE_RECIPE, 'epochs': EPOCHS, 'fixed_epoch': EPOCHS}
V1_MANIFEST_SHA256 = '6ac27ba64afc8ca553b759def5ebc48c28d5767293a61476b74692e08a5b0df4'
VERSIONS = {
    'v1': dict(protocol=v1.PROTOCOL, schema=v1.SCHEMA, tool=v1.TOOL, runner=v1.RUNNER, tag=v1.TAG, epochs=v1.EPOCHS, swa_epochs_1based=v1.SWA_EPOCHS,
               steps=542 * v1.EPOCHS, manifest_sha256=V1_MANIFEST_SHA256,
               status='frozen; superseded by v2 (its SWA members are one pass earlier than the public 0-based [14, 12, 15])'),
    'v2': dict(protocol=PROTOCOL, schema=SCHEMA, tool=TOOL, runner=RUNNER, tag=TAG, epochs=EPOCHS, swa_epochs_1based=SWA_EPOCHS, steps=542 * EPOCHS,
               manifest_sha256=None, status='current'),
}
EPOCH_MAPPING = dict(
    public_swa_over=list(PUBLIC_SWA_OVER), public_numbering='0-based loop index',
    public_evidence=['public trainer: for ep in range(a.epochs) ... "epoch": ep ... "swa_over": [int(t["epoch"]) for t in topk] '
                     '(docs/handoffs/evidence/2026-10/2026-10-05-preprocessing-survey-v1/notebooks/'
                     'dreaddevelopment__knee-mri-training-the-twelve-finding-model/code.py.txt:509,544,622)',
                     'training/src/raptor_train.py:241,260 (same convention)',
                     'artifacts/experiments/HA14_RAPTOR_RECIPE_AUDIT/audit.md:66: consistent with zero-based epochs, at least 16 epochs'],
    mapping='0-based loop index i -> 1-based epoch i + 1: public {14, 12, 15} -> 1-based {15, 13, 16} = sorted 13, 15, 16; needs >= 16 epochs',
    epochs_1based=list(SWA_EPOCHS), loop_indices_0based=[e - 1 for e in SWA_EPOCHS], epochs=EPOCHS,
    source='D-A-050 (public v5 swa_over [14, 12, 15]) + independent review MAJOR M1 (scratchpad review_exp112_113.md, 2026-10-07 ~17:50 KST)',
    decision='coordinator 2026-10-07 17:55 KST: match the public convention (V2 = 16 epochs, SWA 1-based 13/15/16); V1 kept unchanged',
    remaining_differences=['public members were the Gold58 top-3 checkpoints; ours are fixed before training (no Gold, no metric)',
                           'public averaged every float tensor including BatchNorm running statistics (float64 mean, code.py.txt:610-618) with no '
                           'BatchNorm recompute; we use the same float64 mean and then recompute BatchNorm statistics on the training data',
                           'public total epochs, labels, uid list, precision and normalization of v5 remain unknown (D-A-050)'])
SWA = dict(v1.SWA, epochs_1based=list(SWA_EPOCHS), loop_indices_0based=[e - 1 for e in SWA_EPOCHS],
           selection='fixed before training (H-B-33): the public v5 members swa_over [14, 12, 15] read as 0-based loop indices (D-A-050, review M1) '
                     '= 1-based 13, 15, 16; never chosen on Gold, OOF or any metric',
           checkpoint_fields='model_best.pt: epoch and swa_over are 0-based loop indices (public / raptor_train / EXP099 convention); '
                             'epoch_1based and swa_epochs_1based count completed passes')
BN_RECOMPUTE = dict(v1.BN_RECOMPUTE, window_draw_epoch_0based=BN_DRAW_EPOCH,
                    loader='the training DataLoader itself (all 4,343 studies, bs 8, shuffle with the seeded generator, drop_last): '
                           'one more pass after the 16 training epochs = 542 batches / 4,336 studies')
EPOCH_NUMBERING = ('epoch and swa_over: 0-based loop indices (public raptor_train / EXP099 checkpoint convention); epoch_1based and '
                   'swa_epochs_1based: completed passes over the training loader (1-based)')


# ------------------------------------------------------------------ names / recipe
def tag_for(mode='full'):
    if mode not in ('full', 'smoke'):
        raise ValueError('mode is full or smoke')
    return TAG + ('_SMOKE' if mode == 'smoke' else '')


def version(name):
    if name not in VERSIONS:
        raise ValueError('version is v1 or v2')
    return VERSIONS[name]


def public_to_1based(swa_over):
    """Public 0-based loop indices -> sorted 1-based epochs (public v5 [14, 12, 15] -> (13, 15, 16))."""
    return tuple(sorted(int(i) + 1 for i in swa_over))


def full_recipe(common8):
    """V2 common_args from the registered B8 common_args: epochs and fixed_epoch 8 -> 16, nothing else."""
    args = recipe_args(common8)
    if args != BASE_RECIPE:
        raise ValueError('B8 common_args differ from the registered recipe: ' + str({k: (args[k], BASE_RECIPE.get(k)) for k in args if args[k] != BASE_RECIPE.get(k)}))
    out = recipe_args({**args, 'epochs': EPOCHS, 'fixed_epoch': EPOCHS})
    if {k for k in out if out[k] != args[k]} != {'epochs', 'fixed_epoch'}:
        raise AssertionError('more than the epoch count changed')
    return out


def recipe_record(common, pos_weight_vector, n_train):
    record = dict(common_args=common, seed=SEED, loop=LOOP, swa=SWA, bn_recompute=BN_RECOMPUTE, pos_weight=[float(x) for x in pos_weight_vector],
                  n_train=n_train, validation='none', epoch_mapping=EPOCH_MAPPING,
                  checkpoint_selection='none (fixed SWA of 1-based epochs 13, 15, 16 = public 0-based swa_over [14, 12, 15])')
    return record, canonical_json_sha256(record)


# ------------------------------------------------------------------ prepare
def prepare(root=ROOT, destination=None, model_factory=None, tiny=False):
    """Register the V2 protocol (default) or, with destination outside artifacts/, a scratch registration for smoke runs only."""
    import numpy as np
    root = Path(root)
    registered = (root / PROTOCOL).resolve()
    dest = Path(destination).resolve() if destination else registered
    scratch = dest != registered
    if scratch and within(dest, root / 'artifacts'):
        raise SystemExit('scratch registrations must live outside artifacts/')
    if not scratch and (tiny or model_factory is not None):
        raise ValueError('the registered protocol is built with the real CoAtNet initial model')
    if dest.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + str(dest))
    if public_to_1based(PUBLIC_SWA_OVER) != SWA_EPOCHS or check_swa_epochs(SWA_EPOCHS, EPOCHS) != SWA_EPOCHS:
        raise AssertionError('V2 SWA epochs are not the public 0-based swa_over mapped to 1-based')
    v1_manifest = root / v1.PROTOCOL / 'manifest.json'
    if sha256(v1_manifest) != V1_MANIFEST_SHA256:
        raise ValueError('EXP112 V1 manifest changed or missing')
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
    if reg['path'] != DREAD or fp['label_sets'][LABEL_SET]['path'] != DREAD or reg['pos_weight_source'] != 'labels':
        raise ValueError('EXP099 dread registration points elsewhere')
    if sha256(root / V4_PROTOCOL) != V4_MANIFEST_SHA256:
        raise ValueError('EXP094 V4 manifest changed')
    v4 = read(root / V4_PROTOCOL)
    arm = v4['arms'][V4_ARM]
    if v4['common_args'] != fp['common_args'] or v4['initial_state_sha256'] != B_INITIAL_STATE or arm['seed'] != SEED or arm['variant'] != 'seed' \
            or arm['slot_dropout'] != 0.0:
        raise ValueError('EXP094 V4 arm seed42 differs from EXP099')
    if fp['common_args']['workers'] != FULL_WORKERS:
        raise ValueError('B8 workers changed')
    common8 = recipe_args(fp['common_args'])
    common = full_recipe(fp['common_args'])
    if common != EXPECTED_RECIPE:
        raise ValueError('EXP112 V2 recipe differs from the registered 16-epoch recipe')
    labels, semantic = k99._load_labels(root / DREAD)
    if semantic != reg['semantic_sha256'] or sha256(root / DREAD) != reg['sha256']:
        raise ValueError('Dread file differs from the EXP099 registration')
    canonical = canonical_order(read(root / fp['paths']['base_ordered_ids']))
    corpus_dir = root / fp['paths']['corpus']
    if sha256(corpus_dir / 'manifest.json') != fp['corpus_manifest_sha256']:
        raise ValueError('corpus manifest differs from EXP099')
    corpus_ids = np.load(corpus_dir / 'ids.npy', allow_pickle=True).astype(str).tolist()
    trains, vals, fold_files = {}, {}, {}
    for k in sorted(fp['folds'], key=int):
        f = fp['folds'][k]
        for key in ('ordered_ids', 'groups'):
            if sha256(root / f[key]['path']) != f[key]['sha256']:
                raise ValueError(f'EXP099 fold {k} {key} file changed')
        ordered = read(root / f['ordered_ids']['path'])
        if uid_hash(ordered['train']) != f['ordered_ids']['train_sha256'] or uid_hash(ordered['val']) != f['ordered_ids']['val_sha256']:
            raise ValueError(f'EXP099 fold {k} ordered ids changed')
        trains[k], vals[k] = validate_fold_membership(corpus_ids, ordered, read_rows(root / f['groups']['path']), canonical)
        if len(trains[k]) != f['n_train'] or len(vals[k]) != f['n_val']:
            raise ValueError(f'EXP099 fold {k} size changed')
        fold_files[k] = dict(path=f['ordered_ids']['path'], sha256=f['ordered_ids']['sha256'], train_sha256=f['ordered_ids']['train_sha256'],
                             n_train=f['n_train'])
    train = full_training_set(canonical, trains, vals)
    if len(train) != N_STUDIES:
        raise ValueError('G1 population drift')
    gold = k99.gold_uids(root)
    if len(gold) != N_GOLD:
        raise ValueError(f'expected {N_GOLD} Gold uids, found {len(gold)}')
    check_no_gold(train, gold)
    if set(train) & set(fp['excluded_uids']):
        raise ValueError('an EXP099 excluded uid is in the training set')
    missing = [u for u in train if u not in labels]
    if missing:
        raise ValueError(f'Dread misses {len(missing)} training uid(s)')
    arr = np.stack([labels[u] for u in train])
    if arr.shape != (N_STUDIES, 12) or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
        raise ValueError('invalid Dread values')
    pw = pos_weight(labels, train)
    pw_f0 = np.asarray(fp['folds']['0']['pos_weight'], np.float32)
    c = full_contract(len(train), common)
    c8 = full_contract(fp['folds']['0']['n_train'], common8)
    if c['len_train_loader'] != N_STUDIES // common['bs'] or c['steps'] != c['len_train_loader'] * EPOCHS or c['max_fp16_skips'] != c['steps'] // 100:
        raise AssertionError('contract rule drifted')
    factory = model_factory or (build_tiny_initial if tiny else build_initial_model)
    model = factory(common['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init, keys = state_hashes(model)
    bn_modules = len(batchnorm_modules(model))
    model_kind = 'tiny-smoke' if (tiny or model_factory is not None) else 'arch'
    if not scratch and init != B_INITIAL_STATE:
        raise ValueError(f'CPU seed-42 initial state {init} is not B ({B_INITIAL_STATE})')
    if model_kind == 'arch' and bn_modules == 0:
        raise ValueError('the CoAtNet has no BatchNorm modules; the BatchNorm pass would be empty')
    import timm
    dest.mkdir(parents=True)
    dump(dest / TRAIN_IDS_FILE, dict(schema='exp112-train-ids-v1', population='all 4,343 G1 studies = union of the EXP099 fold training lists',
                                     order='canonical EXP099 order: EXP084 ordered-ids train list then val list', n=len(train), train=train))
    dump(dest / KEYS_FILE, dict(schema='exp101-initial-state-keys-v1', arch=common['arch'], model=model_kind, seed=SEED, combined_sha256=init, keys=keys,
                                rule='per state_dict key: sha256 of the contiguous CPU tensor bytes; combined = sha256 over sorted (key bytes + tensor bytes)'))
    recipe, recipe_sha = recipe_record(common, pw, len(train))
    bindings = [RUNNER, TOOL, v1.TOOL, base.TOOL, 'tools/exp099_kfold.py', 'tools/exp087_training.py', 'tools/exp084_group_loss.py',
                FOLD_PROTOCOL + '/manifest.json', fp['paths']['folds'], fp['paths']['base_ordered_ids'], fp['paths']['base_groups'],
                *(v['path'] for v in fold_files.values()), display_path(reg_file, root), DREAD, V4_PROTOCOL, fp['paths']['corpus'] + '/manifest.json',
                v1.PROTOCOL + '/manifest.json']
    manifest = dict(
        schema=SCHEMA, version=VERSION, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item=PLAN_ITEM,
        registration='scratch' if scratch else 'registered',
        supersedes=dict(protocol=v1.PROTOCOL, manifest_sha256=V1_MANIFEST_SHA256, tag=v1.TAG, epochs=v1.EPOCHS, swa_epochs_1based=list(v1.SWA_EPOCHS),
                        files_kept_unchanged=[v1.TOOL, v1.RUNNER, v1.PROTOCOL + '/*'],
                        reason='V1 SWA members (1-based 12/14/15 of 15) are one pass earlier than the public 0-based swa_over [14, 12, 15]'),
        hypothesis='The public-v5 recipe bundle (all labelled G1 studies, >= 16 epochs, fixed SWA of 0-based epochs 12/14/15 = 1-based 13/15/16) '
                   'applied to our CoAtNet-2 224 B8 recipe raises one model to >= 0.923 public (decision rule in plan.md H-B-33; release v5 arm, '
                   'K62 if H-B-30 passes else K36)',
        source_bindings={s: sha256(root / s) for s in bindings},
        protocol_files={TRAIN_IDS_FILE: sha256(dest / TRAIN_IDS_FILE), KEYS_FILE: sha256(dest / KEYS_FILE)},
        training_code_sha256=fp['training_code_sha256'], environment=fp['environment'], corpus_manifest_sha256=fp['corpus_manifest_sha256'],
        excluded_uids=fp['excluded_uids'],
        recipe_source=dict(fold_protocol=FOLD_PROTOCOL + '/manifest.json', fold_protocol_sha256=FOLD_MANIFEST_SHA256, source_common_args=fp['common_args'],
                           v4_protocol=V4_PROTOCOL, v4_sha256=V4_MANIFEST_SHA256, v4_arm=V4_ARM, base_recipe=common8,
                           runner_loop='training/runners/exp112_full_swa.py loop (= exp099_kfold.py / exp101_kfold_arch.py = EXP094 V4), '
                                       'without the evaluation'),
        changes=['training set: all 4,343 G1 studies (union of the five EXP099 fold training lists; Gold58 absent)',
                 'pos_weight: EXP099 Dread rule recomputed on the 4,343 studies',
                 'epochs 8 -> 16 (fixed_epoch 16): OneCycle total_steps 8672, FP16 skip gate 86',
                 'no validation set, no evaluation, no checkpoint selection (no OOF, no Gold, no metric)',
                 'checkpoint after every epoch (epochs/epoch_NN.pt, 1-based NN)',
                 'model_best.pt = SWA mean of 1-based epochs 13, 15, 16 (= public 0-based swa_over [14, 12, 15]) + BatchNorm recompute over the '
                 'training loader; model_final.pt = epoch 16',
                 'vs V1: epochs 15 -> 16 and SWA 12/14/15 -> 13/15/16 (1-based); model_best.pt swa_over now 0-based with explicit 1-based fields'],
        common_args=common, seed=SEED, epochs=EPOCHS, resource_args=RESOURCE_ARGS, smoke_allowed=SMOKE_ALLOWED,
        recipe=recipe, recipe_sha256=recipe_sha, swa=SWA, epoch_mapping=EPOCH_MAPPING, epoch_numbering=EPOCH_NUMBERING,
        bn_recompute=dict(BN_RECOMPUTE, expected_batches=c['len_train_loader'], expected_studies=c['len_train_loader'] * common['bs'],
                          bn_modules=bn_modules),
        initial_state_sha256=init,
        initial_state=dict(keys_file=KEYS_FILE, keys=len(keys), model=model_kind, computed_on='cpu', seed=SEED, arch=common['arch'],
                           torch=torch.__version__, timm=timm.__version__, equals_B=init == B_INITIAL_STATE, B=B_INITIAL_STATE, bn_modules=bn_modules,
                           rule='seed_all(42) after worker_init(0), build_model(arch, pretrained timm); sorted state_dict sha256',
                           enforcement='runner: the seed-42 initial state must equal this hash (no remote relaxation)'),
        training_set=dict(file=TRAIN_IDS_FILE, n=len(train), uid_sha256=uid_hash(train), sorted_uid_sha256=uid_hash(sorted(train)),
                          rule='canonical EXP099 order (EXP084 ordered-ids train then val) = union of the fold 0-4 training lists',
                          folds=fold_files, each_study_in_fold_train_lists=len(trains) - 1,
                          gold=dict(source=k99.GOLD_SOURCE + ' (rows with non-null ACL; uids only)', n=len(gold), overlap=0,
                                    sorted_uid_sha256=uid_hash(sorted(gold))),
                          excluded_uids_overlap=0, validation='none'),
        paths=dict(corpus=fp['paths']['corpus'], labels=DREAD, base_ordered_ids=fp['paths']['base_ordered_ids'], base_groups=fp['paths']['base_groups'],
                   folds=fp['paths']['folds']),
        label_set=dict(name=LABEL_SET, path=DREAD, sha256=reg['sha256'], semantic_sha256=reg['semantic_sha256'],
                       registration=display_path(reg_file, root), registration_sha256=sha256(reg_file)),
        pos_weight=pw.tolist(), pos_weight_rule=fp['pos_weight_rule'].replace('in fold canonical order', 'of all 4,343 studies in canonical order'),
        pos_weight_fold0=pw_f0.tolist(), pos_weight_minus_fold0=[round(float(a - b), 6) for a, b in zip(pw, pw_f0)],
        contract=c, contract_8_epochs_fold0=c8, contract_rule='len_train_loader = n_train // bs (drop_last); steps = len_train_loader * epochs '
                                                              '(accum 1) = OneCycle total_steps; FP16 skip gate: skipped > steps // 100 fails',
        tags=dict(full=TAG, smoke=tag_for('smoke')),
        outputs=dict(run_dir='artifacts/experiments/' + TAG, files=list(RUN_FILES), epoch_checkpoints=EPOCH_DIR + '/epoch_NN.pt (NN = 01..16)',
                     model_best='SWA(1-based 13,15,16) + BatchNorm recompute, release format (keys of EXP099_B8_DREAD_F1/model_best.pt); '
                                'swa_over [12, 14, 15] 0-based, swa_epochs_1based [13, 15, 16], epoch 15 / epoch_1based 16',
                     model_final='plain epoch-16 weights, release format', predictions='none'),
        release=dict(keys=list(RELEASE_KEYS), required=list(RELEASE_REQUIRED), code_bound=list(RELEASE_CODE),
                     note='tools/exp104_release_v5.py / exp111_release_v6.py stage-assets read model_best.pt; arm K chosen at release (K62 if H-B-30 '
                          'passes, else K36); checkpoint k_eval stays the trained K24 like every B8 checkpoint'),
        budget=estimate(root),
        caveats=['weak Dread labels; no validation set, so the run has no local score by design (judged on public only, plan H-B-33)',
                 'one seed; D-B-017: one-seed differences of ~0.006 occur on fold 0',
                 'per-epoch checkpoints hold weights only (no optimizer / scaler / scheduler / sampler state): later SWA variants yes, exact resume no',
                 'the public v5 recipe is not reproduced exactly (labels, uid list, precision, normalization, total epochs unknown: D-A-050)'])
    if not scratch:
        entry = queue_entry(manifest, root)
        manifest['queue'] = dict(file=QUEUE_FILE, after=QUEUE_AFTER, continue_on_failure=True, python=str(GPU_PYTHON),
                                 environment='queue runner started with OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 (as LOCAL_GPU_CHAIN_V14)',
                                 started=False, entry=entry,
                                 note='gpu_queue.py reads its plan once at start: using V2 needs a new chain file and a queue-runner restart; '
                                      'the V1 entry EXP112_B8_FULL_SWA in LOCAL_GPU_CHAIN_V14 is not changed here')
    dump(dest / 'manifest.json', manifest)
    if not scratch:
        dump(root / QUEUE_FILE, [manifest['queue']['entry']])
    report = dict(status='REGISTERED_NO_EXECUTION', version=VERSION, registration=manifest['registration'], protocol=display_path(dest, root),
                  manifest_sha256=sha256(dest / 'manifest.json'), initial_state_sha256=init, equals_B=init == B_INITIAL_STATE, bn_modules=bn_modules,
                  n_train=len(train), uid_sha256=manifest['training_set']['uid_sha256'], recipe_sha256=recipe_sha, contract=c,
                  swa_epochs_1based=list(SWA_EPOCHS), public_swa_over=list(PUBLIC_SWA_OVER), queue=(QUEUE_FILE if not scratch else None),
                  estimate_hours=manifest['budget']['hours'], estimate_range_hours=[manifest['budget']['low_hours'], manifest['budget']['high_hours']])
    print(json.dumps(report, indent=1))
    return manifest


# ------------------------------------------------------------------ GPU time estimate
def estimate(root=ROOT):
    refs = reference_runs(root)
    c = full_contract(N_STUDIES, EXPECTED_RECIPE)
    est = estimate_from([r['seconds_per_update'] for r in refs.values()], c['steps'], c['len_train_loader'])
    return dict(reference_runs=refs, steps=c['steps'], len_train_loader=c['len_train_loader'], overhead_seconds=OVERHEAD_SECONDS, **est,
                rule='median/min/max seconds per update of the reference runs (training epochs only) * 8672 updates + BatchNorm pass '
                     '(542 batches at 0.25-1.0 x the update time) + overhead; the 2026-10-07 flight-V4 runs ran at 1.00-1.06 s/update under '
                     'CPU contention, i.e. near the high end')


# ------------------------------------------------------------------ runner guard
def validate_arguments(args, root):
    """Called by the V2 runner before any training import. Returns (protocol, registered training ids, output root, plan)."""
    root = Path(root)
    manifest_path = Path(args.protocol).resolve()
    p = read(manifest_path)
    smoke = args.mode == 'smoke'
    if p.get('schema') != SCHEMA or p.get('version') != VERSION:
        raise ValueError('wrong protocol')
    if manifest_path == (root / PROTOCOL / 'manifest.json').resolve():
        if p.get('registration') != 'registered':
            raise ValueError('wrong protocol registration')
    elif not smoke or p.get('registration') != 'scratch' or within(manifest_path, root / 'artifacts'):
        raise ValueError('wrong protocol: full runs use ' + PROTOCOL + '; scratch registrations (outside artifacts/) are for smoke only')
    if p.get('seed') != SEED or p['tags']['full'] != TAG or p['common_args'] != EXPECTED_RECIPE or p.get('epochs') != EPOCHS \
            or tuple(p['swa']['epochs_1based']) != SWA_EPOCHS or tuple(p['epoch_mapping']['public_swa_over']) != PUBLIC_SWA_OVER:
        raise ValueError('not the EXP112 V2 registration')
    if p['registration'] == 'registered' and (p['initial_state_sha256'] != B_INITIAL_STATE or p['initial_state']['model'] != 'arch'):
        raise ValueError('registered protocol must start from B')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['protocol_files'].items():
        if sha256(manifest_path.parent / name) != expected:
            raise ValueError('protocol file changed: ' + name)
    for name, expected in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed')
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
    if parse_epochs(args.swa_epochs) != tuple(p['swa']['epochs_1based']):
        raise ValueError('SWA epochs differ from the registration')
    if tag_for(args.mode) != args.tag:
        raise ValueError('tag/mode mismatch')
    for key in ('labels', 'corpus'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('EXP112 is a local run; KNEE_REMOTE_RELAX_ENV is not allowed')
    ids = read(manifest_path.parent / p['training_set']['file'])['train']
    if uid_hash(ids) != p['training_set']['uid_sha256'] or len(ids) != p['training_set']['n']:
        raise ValueError('registered training list changed')
    default_root = (root / 'artifacts' / 'experiments').resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else default_root
    if smoke:
        if not args.output_root or within(out_root, root / 'artifacts'):
            raise ValueError('smoke runs need --output-root outside artifacts/')
        if args.smoke_train < args.bs or args.smoke_epochs < 1:
            raise ValueError('smoke needs >= one batch of training studies and >= 1 epoch')
        plan = dict(epochs=args.smoke_epochs, swa_epochs=check_swa_epochs(parse_epochs(args.smoke_swa_epochs), args.smoke_epochs), smoke=True)
    else:
        if out_root != default_root:
            raise ValueError('full runs write to artifacts/experiments (no --output-root)')
        plan = dict(epochs=p['epochs'], swa_epochs=check_swa_epochs(p['swa']['epochs_1based'], p['epochs']), smoke=False)
    return p, ids, out_root, plan


def runner_argv(p, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None, protocol_path=None):
    root = Path(root)
    manifest_path = Path(protocol_path) if protocol_path else root / PROTOCOL / 'manifest.json'
    argv = [python or train_python(), str(root / RUNNER), '--protocol', str(manifest_path), '--mode', mode, '--tag', tag_for(mode),
            '--seed', str(p['seed']), '--workers', str(p['resource_args']['workers']['default'] if workers is None else int(workers)),
            '--swa-epochs', ','.join(str(e) for e in p['swa']['epochs_1based'])]
    for k, v in p['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--corpus', str(relpath(root, p['paths']['corpus'])), '--labels', str(relpath(root, p['paths']['labels']))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if mode == 'smoke':
        s = {**SMOKE_DEFAULTS, **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), s[key])
        argv += ['--smoke-train', str(s['train']), '--smoke-epochs', str(s['epochs']), '--smoke-swa-epochs', ','.join(str(e) for e in s['swa'])]
    return argv


def queue_entry(p, root=ROOT):
    """The tools/gpu_queue.py job (LOCAL_GPU_CHAIN_V14 entry format) for the registered V2 full run, after the flight-V4 fold 4."""
    return dict(tag=TAG, command=runner_argv(p, 'full', root=root, python=str(GPU_PYTHON)), after=QUEUE_AFTER, continue_on_failure=True)


def _protocol(root=ROOT):
    return read(Path(root) / PROTOCOL / 'manifest.json')


def local_env():
    return base.local_env()


def validate_cli():
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path)."""
    argv = runner_argv(_protocol(), 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(out, protocol, train, epochs, swa, workers):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    manifest_path = Path(protocol).resolve() if protocol else ROOT / PROTOCOL / 'manifest.json'
    argv = runner_argv(read(manifest_path), 'smoke', workers=workers, output_root=out, protocol_path=manifest_path,
                       smoke=dict(train=train, epochs=epochs, swa=parse_epochs(swa)))
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for('smoke')
    report = dict(returncode=r.returncode, seconds=round(time.time() - t0), run=str(run), files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json')
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], swa=s['swa'], history=s['history'],
                      initial_state_matches_registered=s['provenance']['initial_state_matches_registered'], cuda_initialized=s['provenance']['cuda_initialized'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ verification of the finished run
def verify_run(run, p, root=ROOT, hash_epochs=True):
    """Problems (empty list = complete and consistent) for a finished full EXP112 V2 run directory."""
    root, run = Path(root), Path(run)
    missing = [n for n in RUN_FILES if not (run / n).is_file()]
    if missing:
        return ['missing ' + ', '.join(missing)]
    problems = []
    for name, rec in read(run / INVENTORY).items():
        if not (run / name).is_file() or sha256(run / name) != rec['sha256'] or (run / name).stat().st_size != rec['bytes']:
            problems.append(f'{name} differs from {INVENTORY}')
    epochs = read(run / EPOCH_DIR / INVENTORY)
    if list(epochs) != [epoch_file(e) for e in range(1, p['epochs'] + 1)]:
        problems.append('epoch checkpoints are not epoch_01..epoch_%02d' % p['epochs'])
    elif hash_epochs:
        for name, rec in epochs.items():
            if sha256(run / EPOCH_DIR / name) != rec['sha256']:
                problems.append(f'{EPOCH_DIR}/{name} differs from its inventory')
    if sorted(x.name for x in run.glob('*.csv')):
        problems.append('prediction files present: the full run must not evaluate')
    s, prov, swa = read(run / 'summary.json'), read(run / 'provenance.json'), read(run / 'swa.json')
    expect = dict(seed=SEED, arch=p['common_args']['arch'], initial_state_sha256=B_INITIAL_STATE, initial_state_matches_registered=True,
                  environment_matches_baseline=True, requested_workers=FULL_WORKERS, scheduler_total_steps=p['contract']['steps'],
                  max_fp16_skips=p['contract']['max_fp16_skips'], label_set=LABEL_SET, training_mode='full', validation='none',
                  checkpoint_selection=p['recipe']['checkpoint_selection'], gold_access='disabled', metric_access='none',
                  train_ids_ordered_sha256=p['training_set']['uid_sha256'], label_semantic_sha256=p['label_set']['semantic_sha256'],
                  protocol_sha256=sha256(root / PROTOCOL / 'manifest.json'), recipe_sha256=p['recipe_sha256'], cuda_initialized=True)
    for key, value in expect.items():
        if prov.get(key) != value:
            problems.append(f'provenance {key} = {prov.get(key)!r}, expected {value!r}')
    if s.get('optimizer_updates') != p['contract']['steps'] or s.get('skipped_updates', 10 ** 9) > p['contract']['max_fp16_skips']:
        problems.append(f"updates {s.get('optimizer_updates')} (expected {p['contract']['steps']}), skipped {s.get('skipped_updates')}")
    if len(s.get('history') or []) != p['epochs'] or s.get('train') != p['training_set']['n'] or s.get('val') != 0:
        problems.append('history/epoch/count mismatch')
    if swa.get('epochs_1based') != list(SWA_EPOCHS) or [m['epoch_1based'] for m in swa.get('members', [])] != list(SWA_EPOCHS) \
            or swa.get('swa_over_0based') != [e - 1 for e in SWA_EPOCHS]:
        problems.append('SWA members are not 1-based epochs 13, 15, 16 (0-based 12, 14, 15)')
    bn = swa.get('bn_recompute') or {}
    if bn.get('batches') != p['contract']['len_train_loader'] or bn.get('studies') != p['contract']['len_train_loader'] * p['common_args']['bs'] \
            or bn.get('num_batches_tracked') != [p['contract']['len_train_loader']] or bn.get('bn_modules') != p['initial_state']['bn_modules'] \
            or bn.get('window_draw_epoch_0based') != BN_DRAW_EPOCH:
        problems.append(f'BatchNorm pass record differs: {bn}')
    return problems


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('prepare-scratch'); s.add_argument('--dest', required=True); s.add_argument('--tiny', action='store_true')
    s = sub.add_parser('argv'); s.add_argument('--mode', choices=['full', 'smoke'], default='full')
    s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=None)
    sub.add_parser('validate')
    s = sub.add_parser('smoke'); s.add_argument('--out', required=True); s.add_argument('--protocol', default=None)
    s.add_argument('--train', type=int, default=SMOKE_DEFAULTS['train']); s.add_argument('--epochs', type=int, default=SMOKE_DEFAULTS['epochs'])
    s.add_argument('--swa', default=','.join(map(str, SMOKE_DEFAULTS['swa']))); s.add_argument('--workers', type=int, default=0)
    s = sub.add_parser('verify'); s.add_argument('--run', default=None, help=f'run directory (default artifacts/experiments/{TAG})')
    s.add_argument('--quick', action='store_true', help='skip re-hashing the 16 epoch checkpoints')
    sub.add_parser('estimate')
    s = sub.add_parser('versions'); s.add_argument('--version', choices=sorted(VERSIONS), default=None)
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')
    if a.command in ('prepare', 'prepare-scratch'):
        for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():  # before torch/timm/huggingface_hub are imported
            os.environ[k] = v
        prepare() if a.command == 'prepare' else prepare(destination=a.dest, tiny=a.tiny)
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli())
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.out, a.protocol, a.train, a.epochs, a.swa, a.workers))
    elif a.command == 'estimate':
        print(json.dumps(estimate(), indent=1))
    elif a.command == 'versions':
        print(json.dumps({k: dict(v, swa_epochs_1based=list(v['swa_epochs_1based'])) for k, v in VERSIONS.items() if a.version in (None, k)}, indent=1))
    else:
        problems = verify_run(Path(a.run) if a.run else ROOT / 'artifacts' / 'experiments' / TAG, _protocol(), hash_epochs=not a.quick)
        print(json.dumps(dict(run=a.run or TAG, status='OK' if not problems else 'PROBLEMS', problems=problems), indent=1))
        sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
