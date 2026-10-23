"""EXP114 (H-B-39): the EXP107/EXP113 RadImageNet ResNet50 centre-slice x3 5-fold on the relabel_v1 labels, on a free Colab T4, resumable.

  python tools/exp114_radr50_colab.py prepare [--dry-run DIR]   # register EXP114_RADR50_COLAB_PROTOCOL_V1 (CPU, no model build, no GPU)
  python tools/exp114_radr50_colab.py argv --fold K [--mode full|smoke] [--workers N] [--output-root DIR] [--ckpt-dir DIR] [--weights FILE]
  python tools/exp114_radr50_colab.py validate --fold K          # runner --validate-only on CPU (no output)
  python tools/exp114_radr50_colab.py smoke --fold K --out DIR [--train 8 --val 1 --epochs 2 --stop-after-epoch N --workers 2]
  python tools/exp114_radr50_colab.py bundle                     # code bundle -> kaggle/packages/KNEE_RUNNERS_V3_EXP114/exp114_bundle.tar.gz (+ .sha256)
  python tools/exp114_radr50_colab.py publish-command            # print (never run) the command for the optional private bundle dataset
  python tools/exp114_radr50_colab.py colab-job [...]            # ON THE COLAB VM (started by tools/colab/bootstrap_v3_radr50.py)
  python tools/exp114_radr50_colab.py run-folds [...]            # ON THE COLAB VM: folds one after another, resumable, skip finished ones
  python tools/exp114_radr50_colab.py verify [--runs-dir DIR]    # check downloaded fold directories (default .local/colab-exp114/runs)
  python tools/exp114_radr50_colab.py assemble [--runs-dir DIR]  # pooled OOF -> artifacts/experiments/EXP114_RADR50_COLAB_RELABEL_V1_OOF

Protocol: a copy of EXP113_RADR50_LOCAL_PROTOCOL_V1 (manifest sha256 BASE_MANIFEST_SHA256, unchanged) with its registered relabel_v1 label set
bound as THE label set (EXP113 label-sets/relabel_v1.json, sha256 LABEL_REGISTRATION_SHA256: label file and semantic hashes, per-fold
pos_weight recorded at registration and recomputed here from the labels), the EXP099 folds, seed 42, RadImageNet weights, RN_clip recipe,
K12 / K24 / K36, windows, input policy and normalisation unchanged. What changes is where and how it runs: a Colab T4 (FP16 + GradScaler,
strict runtime), environment and seed-42 initial state recorded but not enforced (KNEE_REMOTE_RELAX_ENV=1, as EXP107; a backbone key that
differs from the RadImageNet bytes still stops the run), outputs under an explicit --output-root, and a full checkpoint after every epoch
(training/runners/exp114_radr50_colab.py = the EXP107 runner + EXP114-marked checkpoint/resume lines; tools/exp114_ckpt.py). The Colab
trajectory is an independent ensemble member, never paired with the Kaggle T4 EXP107 or local EXP113 runs (D-B-008). OOF is scored
against Dread (primary, as EXP099/EXP107/EXP113) and also against relabel_v1 and flight_v4 (multi-reference rule, H-B-34), all np.rint.
Nothing here trains on or touches the local GPU.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp107_kfold_radr50 as e107  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
from tools import exp114_ckpt as ckpt  # noqa: E402
# names the runner reads from its guard module (`guard.<name>`), re-exported unchanged from the EXP107 guard
from tools.exp107_kfold_radr50 import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                       CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
                                       DREAD, KEYS_FILE, FOLDS, EXTRA_EVAL, RUN_FILES, fold_contract, state_hashes, mismatched_keys, dump,
                                       train_python, within, display_path, _set_flag)

SCHEMA = 'exp114-radr50-colab-v1'
BASE_PROTOCOL = 'artifacts/experiments/EXP113_RADR50_LOCAL_PROTOCOL_V1'
BASE_MANIFEST_SHA256 = 'a134327a10bfe051a0d58f3bb83969b2bdb8b9fcc4f54a309893504c960bd31d'
BASE_KEYS_SHA256 = '359f3710c7a12d054379bdc2849add4d25bf9732703a9c292086138709b7cc33'
BASE_LABEL_SCHEMA = 'exp113-label-set-v1'
LABEL_SET = 'relabel_v1'
LABEL_REGISTRATION = BASE_PROTOCOL + '/label-sets/relabel_v1.json'
LABEL_REGISTRATION_SHA256 = '489fdc98a843f12d7cc8b6b5365e9d3faba85fa4d5dcd063a82531e2eaa424fb'
FLIGHT_REGISTRATION = BASE_PROTOCOL + '/label-sets/flight_v4.json'
FLIGHT_REGISTRATION_SHA256 = 'f40c9d89a6a87245b89a9369fe6ac2fc333c7b0100e2f9f4b97708de11e56324'
PROTOCOL = 'artifacts/experiments/EXP114_RADR50_COLAB_PROTOCOL_V1'
RUNNER = 'training/runners/exp114_radr50_colab.py'
TOOL = 'tools/exp114_radr50_colab.py'
CKPT_MODULE = 'tools/exp114_ckpt.py'
SHARED_RUNNER = e107.RUNNER
EXP113_CODE = ('training/runners/exp113_radr50_local.py', 'tools/exp113_radr50_local.py')  # bound by EXP113, not executed here
SEED = 42
TAG_PREFIX = 'EXP114_RADR50_COLAB_RELABEL_V1'
OOF_DIR = 'artifacts/experiments/EXP114_RADR50_COLAB_RELABEL_V1_OOF'
LOCAL_DIR = '.local/colab-exp114'
LOCAL_RUNS = LOCAL_DIR + '/runs'
LOCAL_CKPT = LOCAL_DIR + '/ckpt'
WEIGHTS_LOCAL = e107.WEIGHTS_LOCAL
WEIGHTS_ENV = e107.WEIGHTS_ENV
WEIGHTS_DATASET = e107.WEIGHTS_DATASET
WEIGHTS_NAME = e107.WEIGHTS_NAME
RADIMAGENET_SHA256 = e107.RADIMAGENET_SHA256
KAGGLE_USER = 'taeyangg4'
CORPUS_DATASET = 'knee-raptor-corpus-336'
CORPUS_DIR_NAME = 'RAPTOR64_336_C140_S02'
# the private corpus dataset, file by file (the same list as tools/colab/prefetch_corpus_v3.py)
CORPUS_FILES = [('manifest.json', 4923), ('ids.npy', 1129088), ('masks.npy', 282368), ('picks.jsonl', 5225387), ('uids.csv', 312024)] + \
               [(f'vols_shard{k:02d}.npy', 2890137728) for k in range(11)] + [('vols_shard11.npy', 72253568)]
BUNDLE_DATASET = 'knee-runners-v3-exp114'
PACKAGE = ROOT / 'kaggle' / 'packages' / 'KNEE_RUNNERS_V3_EXP114'  # the tar.gz + .sha256 (+ dataset-metadata.json for the optional dataset)
BUNDLE_NAME = 'exp114_bundle.tar.gz'
BUNDLE_TREE = ROOT / LOCAL_DIR / 'bundle' / 'tree'
CODE_FILES = [
    'training/src/raptor_common.py', 'training/src/raptor_data.py', 'training/src/raptor_runtime.py', 'training/src/raptor_train.py',
    'training/src/raptor_radimagenet.py', RUNNER, SHARED_RUNNER, 'training/runners/exp101_kfold_arch.py',
    'tools/__init__.py', 'tools/exp084_group_loss.py', 'tools/exp087_training.py', 'tools/exp099_kfold.py', 'tools/exp101_kfold_arch.py',
    'tools/exp107_kfold_radr50.py', TOOL, CKPT_MODULE, 'tools/colab/run_remote.py',
]
COLAB = dict(base='/content', results='/content/results', ckpt='/content/ckpt', corpus='/content/corpus', xfer='/content/xfer',
             weights='/content/exp114/weights', config='/content/exp114/config.json', session='knee-radr50-rl')
T4_ESTIMATE = dict(seconds_per_epoch_kaggle_t4=168, seconds_per_fold_kaggle_t4=[3452, 3478], colab_minutes_per_fold=75,
                   basis='EXP107 Kaggle T4 history.json (168 s/epoch x 20 + K24/K36 evaluation, 4 vCPU, 4 workers); a free Colab T4 VM has 2 vCPU '
                         '(2 workers), so ~75 min per fold is assumed; plus corpus staging per new VM')
TRAJECTORY_NOTE = ('Colab T4 seed-42 trajectory on relabel_v1: an independent ensemble member, never paired with EXP107 (Kaggle T4, Dread) or '
                   'EXP113 (local); resumed epochs reproduce an uninterrupted run only on the same software stack (sessions are recorded)')
CHANGED_FIELDS = ('schema', 'created_at', 'plan_item', 'source_bindings', 'paths', 'label_set', 'initial_state', 'folds', 'tags', 'remote',
                  'evaluation', 'caveats', 'derived_from', 'checkpoint')
REMOVED_FIELDS = ('local', 'label_sets')
FOLD_CHANGED_FIELDS = ('tag', 'pos_weight', 'pos_weight_source', 'pos_weight_equals_exp099', 'pos_weight_equals_registration', 'val_positive_counts',
                       'dread_val_positive_counts', 'macro12_valid')
ENFORCEMENT = ('Colab runner: KNEE_REMOTE_RELAX_ENV=1 is required for full runs; environment and seed-42 initial state are recorded, not '
               'enforced (D-B-008), except that a backbone key differing from the RadImageNet bytes always stops the run')
CAVEATS = [
    'Colab T4 trajectory (Linux, the Colab image of each session, relaxed initial state, FP16 + GradScaler 1024): an independent member, '
    'never a paired arm of EXP107 (Kaggle T4, Dread) or EXP113 (local); a resume after a VM reclaim is exact only when the new session has '
    'the same software stack (every session is recorded in provenance.sessions)',
    'trained on relabel_v1 (report labels re-extracted with an LLM; pos_weight from relabel_v1 itself); OOF scored against Dread (primary) and '
    'also relabel_v1 and flight_v4, which share the report-text ceiling (D-B-014)',
]


def tag_for(fold, mode='full'):
    return f'{TAG_PREFIX}_F{int(fold)}' + ('_SMOKE' if mode == 'smoke' else '')


def _manifest_path(root=ROOT):
    return Path(root) / PROTOCOL / 'manifest.json'


def _protocol(root=ROOT):
    return read(_manifest_path(root))


def weights_path(root=ROOT):
    env = os.environ.get(WEIGHTS_ENV)
    return Path(env) if env else Path(root) / WEIGHTS_LOCAL


def _json_line(value):
    print(json.dumps(value, indent=1, default=str), flush=True)


# ------------------------------------------------------------------ prepare
def check_label_registration(root, base_manifest_sha=BASE_MANIFEST_SHA256):
    """The EXP113 relabel_v1 registration, every hash in its chain re-checked. Returns (registration, labels)."""
    root = Path(root)
    f = root / LABEL_REGISTRATION
    if sha256(f) != LABEL_REGISTRATION_SHA256:
        raise ValueError('EXP113 relabel_v1 registration changed: ' + LABEL_REGISTRATION)
    r = read(f)
    if r.get('schema') != BASE_LABEL_SCHEMA or r.get('name') != LABEL_SET or r.get('protocol_sha256') != base_manifest_sha:
        raise ValueError('relabel_v1 registration does not belong to the EXP113 manifest')
    if r.get('pos_weight_source') != 'labels':
        raise ValueError('relabel_v1 pos_weight is not computed from its own labels')
    if sha256(relpath(root, r['path'])) != r['sha256']:
        raise ValueError('relabel_v1 label file changed: ' + r['path'])
    src = r['source_registration']
    if sha256(relpath(root, src['path'])) != src['sha256']:
        raise ValueError('EXP099 relabel_v1 registration changed: ' + src['path'])
    labels, semantic = k99._load_labels(relpath(root, r['path']))
    if semantic != r['semantic_sha256']:
        raise ValueError('relabel_v1 semantic hash changed')
    return r, labels


def prepare(root=ROOT, dest=None):
    import numpy as np
    root = Path(root)
    destination = Path(dest) if dest else root / PROTOCOL
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + str(destination))
    base_path = root / BASE_PROTOCOL / 'manifest.json'
    if sha256(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError('EXP113 manifest changed')
    bp = read(base_path)
    if bp['schema'] != 'exp113-radr50-local-v1' or bp['seed'] != SEED or bp['label_set']['name'] != 'dread' or bp['common_args'] != e107.EXPECTED_RECIPE:
        raise ValueError('unexpected EXP113 registration')
    for path, expected in bp['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('EXP113 binding changed: ' + path)
    for name, expected in bp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    if sha256(root / bp['initial_state']['keys_file']) != BASE_KEYS_SHA256:
        raise ValueError('EXP113 initial-state keys file changed')
    if sha256(root / FLIGHT_REGISTRATION) != FLIGHT_REGISTRATION_SHA256:
        raise ValueError('EXP113 flight_v4 registration changed')
    reg, labels = check_label_registration(root)
    dread, _ = k99._load_labels(root / DREAD)
    canonical = canonical_order(read(root / bp['paths']['base_ordered_ids']))
    corpus_ids = np.load(root / bp['paths']['corpus'] / 'ids.npy', allow_pickle=True).astype(str).tolist()
    folds, cover = {}, []
    for k in FOLDS:
        f113 = bp['folds'][str(k)]
        for key in ('ordered_ids', 'groups'):
            if sha256(root / f113[key]['path']) != f113[key]['sha256']:
                raise ValueError(f'fold {k} {key} file changed')
        ordered = read(root / f113['ordered_ids']['path'])
        train, val = validate_fold_membership(corpus_ids, ordered, read_rows(root / f113['groups']['path']), canonical)
        if (len(train), len(val)) != (f113['n_train'], f113['n_val']):
            raise ValueError(f'fold {k} sizes changed')
        if any(u not in labels for u in train) or any(u not in labels for u in val):
            raise ValueError(f'fold {k}: relabel_v1 does not cover the fold')
        pw = pos_weight(labels, train)
        if not np.array_equal(pw, np.asarray(reg['pos_weight'][str(k)], np.float32)):
            raise ValueError(f'fold {k}: relabel_v1 pos_weight recomputed from the labels differs from the registration')
        y = np.rint(np.stack([labels[u] for u in val]).astype(np.float32)).astype(int)
        yd = np.rint(np.stack([dread[u] for u in val]).astype(np.float32)).astype(int)
        counts = {t: int(y[:, j].sum()) for j, t in enumerate(T)}
        fold = copy.deepcopy(f113)
        fold.pop('pos_weight_equals_exp099', None)
        fold.update(tag=tag_for(k), pos_weight=pw.tolist(), pos_weight_source='labels', pos_weight_equals_registration=True,
                    val_positive_counts=counts, dread_val_positive_counts={t: int(yd[:, j].sum()) for j, t in enumerate(T)},
                    macro12_valid=all(0 < c < len(val) for c in counts.values()))
        folds[str(k)] = fold
        cover += val
    if sorted(cover) != sorted(canonical):
        raise ValueError('fold validation sets do not partition the 4,343 studies')
    for rel in (RUNNER, TOOL, CKPT_MODULE):
        if not (root / rel).is_file():
            raise ValueError('missing ' + rel)
    destination.mkdir(parents=True)
    (destination / KEYS_FILE).write_bytes((root / bp['initial_state']['keys_file']).read_bytes())
    if sha256(destination / KEYS_FILE) != BASE_KEYS_SHA256:
        raise AssertionError('keys file copy differs')
    bindings = {p: h for p, h in bp['source_bindings'].items() if p not in EXP113_CODE}
    bindings.update({BASE_PROTOCOL + '/manifest.json': BASE_MANIFEST_SHA256, LABEL_REGISTRATION: LABEL_REGISTRATION_SHA256,
                     reg['source_registration']['path']: reg['source_registration']['sha256'], reg['path']: reg['sha256'],
                     RUNNER: sha256(root / RUNNER), TOOL: sha256(root / TOOL), CKPT_MODULE: sha256(root / CKPT_MODULE),
                     PROTOCOL + '/' + KEYS_FILE: sha256(destination / KEYS_FILE)})
    flight = read(root / FLIGHT_REGISTRATION)
    manifest = copy.deepcopy(bp)
    for field in REMOVED_FIELDS:
        manifest.pop(field, None)
    manifest.update(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item='H-B-39', source_bindings=bindings,
        paths={**bp['paths'], 'labels': reg['path'], 'eval_labels': reg['path']},
        label_set=dict(name=LABEL_SET, path=reg['path'], sha256=reg['sha256'], semantic_sha256=reg['semantic_sha256'], registration=LABEL_REGISTRATION,
                       registration_sha256=LABEL_REGISTRATION_SHA256, source_registration=reg['source_registration']['path'],
                       pos_weight_source='labels', note='the EXP107 runner recomputes pos_weight from --labels and requires equality with folds[*].pos_weight'),
        initial_state={**bp['initial_state'], 'keys_file': PROTOCOL + '/' + KEYS_FILE, 'enforcement': ENFORCEMENT},
        folds=folds,
        tags=dict(prefix=TAG_PREFIX, pattern=TAG_PREFIX + '_F<K>[_SMOKE]', oof_dir=OOF_DIR, exp113_prefix=bp['tags']['prefix'],
                  exp107_prefix=bp['tags']['exp107_prefix']),
        remote=dict(platform='Google Colab T4 (free tier; sessions can be reclaimed at any time)', runner=RUNNER, shared_runner=SHARED_RUNNER,
                    guard=TOOL, checkpoint_module=CKPT_MODULE, bootstrap='tools/colab/bootstrap_v3_radr50.py', poller='tools/colab/sync_exp114.py',
                    runbook='docs/runbooks/colab-exp114.md', relax_env=ENFORCEMENT, weights=f'{WEIGHTS_DATASET} ({WEIGHTS_NAME}, sha256 checked)',
                    corpus=f'{KAGGLE_USER}/{CORPUS_DATASET} (per-file, resumable; manifest sha256 checked by the runner)', paths=COLAB,
                    workers='min(resource_args.workers.default, os.cpu_count()) = 2 on a 2-vCPU Colab VM (speed only)', estimate=T4_ESTIMATE,
                    independence='independent trajectory, never paired with a local or Kaggle arm (D-B-008)'),
        checkpoint=dict(module=CKPT_MODULE, schema=ckpt.SCHEMA, layout='<ckpt-dir>/<tag>/epoch_NNN/{part_000..., manifest.json} + latest.json',
                        part_mib=32, keep_remote=3, contents=list(ckpt.PAYLOAD_KEYS), rng=list(ckpt.RNG_KEYS),
                        identity='tag, fold, mode, seed, epochs, steps, n_train, n_val, protocol / runner / guard / checkpoint-module sha256, label '
                                 'semantic hash, ordered train/eval id hashes, device, amp',
                        rule='written after every epoch; --resume auto continues from the newest intact checkpoint of the same identity; a '
                             'different identity stops the run; resumed epochs draw the uninterrupted run\'s permutations and RNG streams'),
        evaluation={**bp['evaluation'], 'oof_dir': OOF_DIR, 'label_sets': None,
                    'references': dict(dread=DREAD, relabel_v1=reg['path'], flight_v4=flight['path']),
                    'primary': 'dread', 'rule': 'float32 then numpy.rint; average-rank AUC as tools/g1_compare.py; every reference reported'},
        caveats=[c for c in bp['caveats'] if not c.startswith('local RTX')] + CAVEATS,
        derived_from=dict(path=BASE_PROTOCOL + '/manifest.json', sha256=BASE_MANIFEST_SHA256, label_registration=LABEL_REGISTRATION,
                          label_registration_sha256=LABEL_REGISTRATION_SHA256, keys_sha256=BASE_KEYS_SHA256, seed=bp['seed'],
                          arch=bp['common_args']['arch'], tag_prefix=bp['tags']['prefix'], changed_fields=list(CHANGED_FIELDS),
                          removed_fields=list(REMOVED_FIELDS), fold_changed_fields=list(FOLD_CHANGED_FIELDS), changed_args={}))
    dump(destination / 'manifest.json', manifest)
    report = dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, written_to=str(destination), manifest_sha256=sha256(destination / 'manifest.json'),
                  label_set=LABEL_SET, total_steps=manifest['total_steps'], tags=[v['tag'] for v in folds.values()],
                  macro12_valid={k: v['macro12_valid'] for k, v in folds.items()})
    _json_line(report)
    return manifest


# ------------------------------------------------------------------ runner guard (called by training/runners/exp114_radr50_colab.py)
def validate_arguments(args, root):
    """Before any training import. Returns (protocol, fold spec, ordered ids, output root)."""
    root = Path(root)
    manifest_path = Path(args.protocol)
    p = read(manifest_path)
    if p.get('schema') != SCHEMA or manifest_path.resolve() != _manifest_path(root).resolve():
        raise ValueError('wrong protocol')
    if p.get('derived_from', {}).get('sha256') != BASE_MANIFEST_SHA256 or p['tags']['prefix'] != TAG_PREFIX or p['label_set']['name'] != LABEL_SET:
        raise ValueError('not the EXP114 registration')
    smoke = args.mode == 'smoke'
    relaxed = bool(os.environ.get('KNEE_REMOTE_RELAX_ENV'))
    if not smoke and not relaxed:
        raise ValueError('full EXP114 runs are Colab runs: KNEE_REMOTE_RELAX_ENV=1 is required (a local full run belongs to EXP113)')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
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
    if not args.output_root:
        raise ValueError('EXP114 runs need an explicit --output-root (Colab: /content/results; smoke: a directory outside artifacts/)')
    out_root = Path(args.output_root).resolve()
    if within(out_root, root / 'artifacts'):
        raise ValueError('--output-root must be outside artifacts/ (results are verified and assembled after download)')
    if smoke and (args.smoke_train < args.bs or args.smoke_val < 1 or args.smoke_epochs < 1):
        raise ValueError('smoke needs >= one batch of training studies, >= 1 validation study and >= 1 epoch')
    if not smoke and not args.ckpt_dir:
        raise ValueError('full EXP114 runs need --ckpt-dir (the point of this protocol is resuming after a reclaim)')
    if args.ckpt_dir:
        if within(args.ckpt_dir, root / 'artifacts'):
            raise ValueError('--ckpt-dir must be outside artifacts/')
        if args.resume == 'never' and ckpt.complete_epochs(Path(args.ckpt_dir) / args.tag):
            raise ValueError('--resume never, but checkpoints of this tag exist; move them aside first')
        if args.keep_ckpt < 1 or args.part_mib < 1:
            raise ValueError('--keep-ckpt and --part-mib must be >= 1')
    return p, fold, ordered, out_root


def run_identity(args, protocol, train_ids, val_ids, epochs, steps, runner_file, ckpt_module=ckpt):
    """What a checkpoint must match to be resumed by this process (tools/exp114_ckpt.find_resume)."""
    return dict(schema=ckpt_module.SCHEMA, tag=args.tag, fold=int(args.fold), mode=args.mode, seed=int(args.seed), epochs=int(epochs), steps=int(steps),
                n_train=len(train_ids), n_val=len(val_ids), protocol_sha256=sha256(args.protocol), runner_sha256=sha256(runner_file),
                guard_sha256=sha256(__file__), ckpt_module_sha256=sha256(ckpt_module.__file__),
                label_semantic_sha256=protocol['label_set']['semantic_sha256'], train_ids_ordered_sha256=uid_hash(train_ids),
                eval_ids_ordered_sha256=uid_hash(val_ids), device=args.device, amp=args.amp)


def runner_argv(p, fold, mode='full', root=ROOT, python=None, workers=None, output_root=None, ckpt_dir=None, smoke=None, weights=None,
                keep_ckpt=3, part_mib=32, resume='auto', stop_after_epoch=None):
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
    if ckpt_dir:
        argv += ['--ckpt-dir', str(ckpt_dir), '--keep-ckpt', str(int(keep_ckpt)), '--part-mib', str(int(part_mib)), '--resume', resume]
    if stop_after_epoch is not None:
        argv += ['--stop-after-epoch', str(int(stop_after_epoch))]
    if mode == 'smoke':
        smoke = {**dict(device='cpu', amp='none', eval_amp='none', train=8, val=1, epochs=2), **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), smoke[key])
        argv += ['--smoke-train', str(smoke['train']), '--smoke-val', str(smoke['val']), '--smoke-epochs', str(smoke['epochs'])]
    return argv


def local_env():
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(ROOT)}
    for k in ('KNEE_REMOTE_RELAX_ENV', WEIGHTS_ENV):
        env.pop(k, None)
    return env


def remote_env(root, weights=None):
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(root),
           'KNEE_REMOTE_RELAX_ENV': '1'}
    if weights:
        env[WEIGHTS_ENV] = str(weights)
    return env


def validate_cli(fold):
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path; smoke-free, no output)."""
    argv = runner_argv(_protocol(), fold, 'full', output_root=ROOT / LOCAL_DIR / 'validate', ckpt_dir=ROOT / LOCAL_DIR / 'validate-ckpt') + ['--validate-only']
    env = {**local_env(), 'KNEE_REMOTE_RELAX_ENV': '1'}
    r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-6000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(fold, out, train=8, val=1, epochs=2, workers=2, stop_after_epoch=None, ckpt_dir=None):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    argv = runner_argv(_protocol(), fold, 'smoke', workers=workers, output_root=out, ckpt_dir=ckpt_dir or out / 'ckpt',
                       smoke=dict(train=train, val=val, epochs=epochs), stop_after_epoch=stop_after_epoch)
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    _json_line(dict(returncode=r.returncode, seconds=round(time.time() - t0), run=str(out / tag_for(fold, 'smoke'))))
    return r.returncode


# ------------------------------------------------------------------ verification of finished fold directories
def verify_run(run, p, k, manifest_sha):
    """Problems (empty list = complete and consistent) for one EXP114 fold directory (Colab full run)."""
    import numpy as np
    run = Path(run); fold = p['folds'][str(k)]; tag = tag_for(k)
    missing = [n for n in RUN_FILES if not (run / n).is_file()]
    if missing:
        return [f'fold {k}: missing ' + ', '.join(missing)]
    problems = []
    try:
        inv = read(run / 'output-inventory.json')
        s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    except ValueError as e:
        return [f'fold {k}: unreadable json ({e})']
    for name in RUN_FILES:
        if name != 'output-inventory.json' and name not in inv:
            problems.append(f'fold {k}: {name} not in output-inventory.json')
    for name, rec in inv.items():
        if not (run / name).is_file() or sha256(run / name) != rec['sha256']:
            problems.append(f'fold {k}: {name} differs from output-inventory.json')
    expect = dict(fold=k, label_set=LABEL_SET, label_semantic_sha256=p['label_set']['semantic_sha256'], label_file_sha256=p['label_set']['sha256'],
                  protocol_sha256=manifest_sha, eval_ids_ordered_sha256=fold['ordered_ids']['val_sha256'],
                  train_ids_ordered_sha256=fold['ordered_ids']['train_sha256'], training_mode='full', checkpoint_selection='fixed-final', seed=SEED,
                  arch=p['common_args']['arch'], input_policy='center3', norm='radimagenet', remote_relaxed=True,
                  runner_sha256=p['source_bindings'][RUNNER], guard_sha256=p['source_bindings'][TOOL])
    for key, value in expect.items():
        if prov.get(key) != value:
            problems.append(f'fold {k}: provenance {key} = {prov.get(key)!r}, expected {value!r}')
    if (prov.get('init_weights') or {}).get('weights_sha256') != RADIMAGENET_SHA256:
        problems.append(f'fold {k}: init weights are not the registered RadImageNet file')
    if any(x.startswith('backbone.') for x in prov.get('initial_state_mismatched_keys') or []):
        problems.append(f'fold {k}: backbone initial state differs from the RadImageNet bytes')
    if not np.array_equal(np.asarray(prov.get('actual_pos_weight') or [], np.float32), np.asarray(fold['pos_weight'], np.float32)):
        problems.append(f'fold {k}: pos_weight differs from the relabel_v1 registration')
    if s.get('optimizer_updates') != fold['steps'] or s.get('skipped_updates', 10 ** 9) > fold['max_fp16_skips']:
        problems.append(f"fold {k}: updates {s.get('optimizer_updates')} (expected {fold['steps']}), skipped {s.get('skipped_updates')}")
    if s.get('selected_epoch_1based') != p['common_args']['epochs'] or s.get('train') != fold['n_train'] or s.get('val') != fold['n_val']:
        problems.append(f'fold {k}: epoch/count mismatch')
    if run.name != tag or (s.get('args') or {}).get('tag') != tag:
        problems.append(f'fold {k}: run directory / tag is not {tag}')
    return problems


# ------------------------------------------------------------------ the fold loop on the Colab VM
def plan_fold(run, ckpt_tag_dir, p, k, manifest_sha, done_elsewhere=()):
    """What run-folds does with fold k: done | done-elsewhere | resume | fresh-moved-partial | fresh."""
    run = Path(run)
    if run.is_dir() and not verify_run(run, p, k, manifest_sha):
        return 'done'
    if tag_for(k) in set(done_elsewhere or ()):
        return 'done-elsewhere'
    if ckpt.complete_epochs(ckpt_tag_dir):
        return 'resume'
    if run.is_dir() and any(run.iterdir()):
        return 'fresh-moved-partial'
    return 'fresh'


def ensure_xfer(run, xfer_tag_dir, part_bytes=ckpt.PART_BYTES):
    """model_best.pt as parts under <xfer>/<tag>/model_best.pt/ for the poller (small run files are downloaded directly)."""
    run, d = Path(run), Path(xfer_tag_dir) / 'model_best.pt'
    expected = read(run / 'output-inventory.json')['model_best.pt']['sha256']
    if d.is_dir():
        if not ckpt.verify_parts(d) and ckpt.read_manifest(d).get('sha256') == expected:
            return d
        shutil.rmtree(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    m = ckpt.split_file(run / 'model_best.pt', d, dict(tag=run.name), part_bytes)
    if m['sha256'] != expected:
        raise RuntimeError('model_best.pt parts differ from output-inventory.json')
    return d


def _subprocess_launch(argv, env, cwd):
    sys.stdout.flush(); sys.stderr.flush()
    return subprocess.run(argv, cwd=cwd, env=env).returncode  # inherits stdout: the runner log lands in runner.log


def run_folds(root, folds, results, ckpt_dir, xfer_dir, weights, workers=None, done_elsewhere=(), attempts=2, keep_ckpt=3, part_mib=32,
              launch=None, python=None):
    """Train the given folds one after another; skip finished ones; every fold resumes from its newest checkpoint. Returns the status dict."""
    root, results, ckpt_dir, xfer_dir = Path(root), Path(results), Path(ckpt_dir), Path(xfer_dir)
    launch = launch or _subprocess_launch
    results.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = _protocol(root); manifest_sha = sha256(_manifest_path(root))
    workers = max(0, min(p['resource_args']['workers']['default'], os.cpu_count() or 1)) if workers is None else int(workers)
    env = remote_env(root, weights)
    status_path = results / 'exp114-status.json'
    status = dict(protocol=PROTOCOL, protocol_sha256=manifest_sha, state='TRAINING', folds_requested=list(folds), workers=workers,
                  started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), folds={})

    def write_status():
        status['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        ckpt.write_json_atomic(status_path, status)

    for k in folds:
        tag = tag_for(k); run = results / tag
        action = plan_fold(run, ckpt_dir / tag, p, k, manifest_sha, done_elsewhere)
        rec = status['folds'][str(k)] = dict(tag=tag, action=action, started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        print(f'=== fold {k} {tag}: {action}', flush=True)
        if action == 'done':
            ensure_xfer(run, xfer_dir / tag)
            rec['state'] = 'OK'; write_status(); continue
        if action == 'done-elsewhere':
            rec['state'] = 'DONE_ELSEWHERE'; write_status(); continue
        if action == 'fresh-moved-partial':
            moved = run.with_name(f'{tag}.partial-{time.strftime("%Y%m%d-%H%M%S")}')
            os.rename(run, moved); rec['moved_partial'] = str(moved)
        argv = runner_argv(p, k, 'full', root=root, python=python or sys.executable, workers=workers, output_root=results, ckpt_dir=ckpt_dir,
                           weights=weights, keep_ckpt=keep_ckpt, part_mib=part_mib)
        rec.update(state='RUNNING', argv=argv); write_status()
        t0, codes, problems = time.time(), [], ['not run']
        for attempt in range(1, attempts + 1):
            code = launch(argv, env, root)
            codes.append(code)
            problems = verify_run(run, p, k, manifest_sha) if run.exists() else ['run directory missing']
            print(f'=== fold {k} attempt {attempt}: exit {code}, problems {problems[:3]}', flush=True)
            if code == 0 and not problems:
                break
        rec.update(state='OK' if codes[-1] == 0 and not problems else 'FAILED', exit_codes=codes, problems=problems, seconds=round(time.time() - t0),
                   finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        if rec['state'] == 'OK':
            ensure_xfer(run, xfer_dir / tag)
            s = read(run / 'summary.json')
            (run / 'remote_run.json').write_text(json.dumps(dict(tag=tag, fold=k, exit_codes=codes, seconds=rec['seconds'], sessions=s.get('sessions'),
                                                                 protocol_sha256=manifest_sha, finished_at=rec['finished_at']), indent=1, default=str),
                                                 encoding='utf-8')
        write_status()
    states = [v['state'] for v in status['folds'].values()]
    status['state'] = 'COMPLETE' if states and all(x in ('OK', 'DONE_ELSEWHERE') for x in states) else ('PARTIAL' if 'OK' in states else 'FAILED')
    write_status()
    print('=== run-folds ' + json.dumps({k: v['state'] for k, v in status['folds'].items()}) + ' -> ' + status['state'], flush=True)
    return status


# ------------------------------------------------------------------ Colab VM job (dependencies, corpus, weights, staging, folds)
def _pip(*specs):
    return subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *specs], capture_output=True, text=True)


def ensure_software():
    """Install only what is missing (never re-install torch / numpy / opencv over the Colab image)."""
    probe = ('import json, importlib\nd = {}\nfor m in ("torch", "timm", "numpy", "pandas", "cv2", "pydicom"):\n'
             '    try:\n        d[m] = importlib.import_module(m).__version__\n    except Exception as e:\n        d[m] = None\n'
             'import torch\nd["cuda"] = torch.cuda.is_available()\nd["gpu"] = torch.cuda.get_device_name(0) if d["cuda"] else None\nprint(json.dumps(d))')

    def versions():
        r = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else dict(error=r.stderr[-800:])
    v = versions()
    need = []
    timm_v = v.get('timm')
    if not timm_v or tuple(int(x) for x in timm_v.split('.')[:3] if x.isdigit()) < (1, 0, 19):
        need.append('timm>=1.0.19')
    if not v.get('pydicom'):
        need.append('pydicom')
    if not v.get('pandas'):
        need.append('pandas')
    if need:
        r = _pip(*need)
        print(f'pip install {need}: exit {r.returncode} {r.stderr[-300:]}', flush=True)
        v = versions()
    missing = [m for m in ('torch', 'timm', 'numpy', 'pandas', 'cv2') if not v.get(m)]
    if missing or not v.get('cuda'):
        raise RuntimeError(f'software not ready: missing {missing}, cuda {v.get("cuda")}: {v}')
    return v


def corpus_state(corpus_dir):
    corpus_dir = Path(corpus_dir)
    have = {name: (corpus_dir / name).stat().st_size if (corpus_dir / name).is_file() else None for name, _ in CORPUS_FILES}
    missing = [name for name, size in CORPUS_FILES if have[name] != size]
    return dict(complete=not missing, missing=missing, bytes=sum(v or 0 for v in have.values()), total=sum(s for _, s in CORPUS_FILES))


def _prefetch_alive():
    for pid in os.listdir('/proc') if os.path.isdir('/proc') else []:
        if pid.isdigit():
            try:
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes().decode('utf-8', 'replace')
            except OSError:
                continue
            if 'prefetch_corpus_v3' in cmd and int(pid) != os.getpid():
                return int(pid)
    return None


def ensure_corpus(corpus_dir, root, user=KAGGLE_USER, wait_hours=3.0, poll_seconds=60):
    """Use /content/corpus as is when complete (sizes as the prefetch); wait for a running prefetch; else download missing files only."""
    corpus_dir = Path(corpus_dir)
    t0 = time.time()
    while not corpus_state(corpus_dir)['complete'] and _prefetch_alive() and time.time() - t0 < wait_hours * 3600:
        s = corpus_state(corpus_dir)
        print(f'waiting for prefetch_corpus_v3 (pid {_prefetch_alive()}): {s["bytes"] / 1e9:.1f}/{s["total"] / 1e9:.1f} GB, missing {len(s["missing"])}',
              flush=True)
        time.sleep(poll_seconds)
    s = corpus_state(corpus_dir)
    if not s['complete']:
        print(f'corpus incomplete ({len(s["missing"])} files); downloading the missing files one by one', flush=True)
        _pip('-U', 'kaggle')
        sys.path.insert(0, str(Path(root) / 'tools' / 'colab'))
        import run_remote  # the resumable per-file download of bootstrap_v2 / run_remote.stage
        run_remote.ensure_dataset(f'{user}/{CORPUS_DATASET}', corpus_dir)
        s = corpus_state(corpus_dir)
        if not s['complete']:
            raise RuntimeError(f'corpus still incomplete: {s["missing"]}')
    p = _protocol(root)
    if sha256(corpus_dir / 'manifest.json') != p['corpus_manifest_sha256']:
        raise RuntimeError('corpus manifest differs from the registered corpus')
    return dict(s, waited_seconds=round(time.time() - t0))


def ensure_weights(weights_dir, user_hint=None):
    d = Path(weights_dir); d.mkdir(parents=True, exist_ok=True)
    target = d / WEIGHTS_NAME
    if target.is_file() and sha256(target) == RADIMAGENET_SHA256:
        return target
    _pip('-U', 'kaggle')
    for _ in range(3):
        subprocess.run(['kaggle', 'datasets', 'download', '-d', WEIGHTS_DATASET, '-f', WEIGHTS_NAME, '-p', str(d), '--unzip', '-q', '-o'],
                       capture_output=True, text=True)
        if target.is_file() and sha256(target) == RADIMAGENET_SHA256:
            return target
        time.sleep(20)
    raise RuntimeError(f'could not fetch {WEIGHTS_DATASET}/{WEIGHTS_NAME} with sha256 {RADIMAGENET_SHA256}')


def stage_corpus_link(root, corpus_dir):
    p = _protocol(root)
    link = Path(root) / p['paths']['corpus']
    if link.exists() or link.is_symlink():
        if link.resolve() != Path(corpus_dir).resolve():
            raise RuntimeError(f'{link} points elsewhere ({link.resolve()})')
        return link
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(Path(corpus_dir).resolve(), link, target_is_directory=True)
    return link


def colab_job(a):
    root = ROOT
    print(f'=== colab-job start {time.strftime("%Y-%m-%d %H:%M:%S")} | workspace {root} | folds {a.folds} | done elsewhere {a.done_elsewhere}', flush=True)
    software = ensure_software()
    print('software ' + json.dumps(software), flush=True)
    corpus = ensure_corpus(a.corpus, root, a.kaggle_user, a.wait_prefetch_hours)
    print('corpus ' + json.dumps(corpus), flush=True)
    weights = ensure_weights(a.weights_dir)
    print(f'weights {weights} sha256 ok', flush=True)
    stage_corpus_link(root, a.corpus)
    status = run_folds(root, [int(x) for x in a.folds.split(',') if x != ''], a.results, a.ckpt_dir, a.xfer, weights, a.workers,
                       [x for x in a.done_elsewhere.split(',') if x], a.attempts, a.keep_ckpt, a.part_mib)
    status.update(software=software, corpus=corpus)
    ckpt.write_json_atomic(Path(a.results) / 'exp114-status.json', status)
    print('=== ALL DONE ' + status['state'], flush=True)
    return 0 if status['state'] == 'COMPLETE' else 1


# ------------------------------------------------------------------ bundle (code + protocol inputs; colab upload or optional private dataset)
def bundle_files(root=ROOT):
    from tools.remote_bundle_v3 import protocol_files, FORBIDDEN_PREFIXES
    expected = {f: None for f in CODE_FILES}
    for f, h in protocol_files(root, PROTOCOL).items():
        if expected.get(f) not in (None, h):
            raise ValueError('conflicting hashes for ' + f)
        expected[f] = h if h is not None else expected.get(f)
    bad = [f for f in expected if f.startswith(FORBIDDEN_PREFIXES) or Path(f).name == 'train.csv' or Path(f).is_absolute() or '..' in Path(f).parts
           or f.startswith('storage/weights') or f.endswith('.pt') or f.startswith('.local')]
    if bad:
        raise ValueError('refusing to ship: ' + ', '.join(bad))
    return dict(sorted(expected.items()))


def build_bundle(root=ROOT, tree=BUNDLE_TREE, package=PACKAGE):
    """Workspace skeleton (code, protocol, bound inputs) -> one tar.gz + .sha256 in PACKAGE (uploaded by sync_exp114.py deploy, or the only
    file of the optional private dataset). The tar holds bundle_manifest.json with every file's sha256; the VM checks both."""
    from tools.remote_bundle_v3 import WORKSPACE_TOML
    root, tree, package = Path(root), Path(tree), Path(package)
    files = bundle_files(root)
    if tree.exists():
        shutil.rmtree(tree)
    for rel, expected in files.items():
        src, dst = root / rel, tree / rel
        if expected is not None and sha256(src) != expected:
            raise ValueError(f'{rel} differs from the protocol binding')
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    (tree / 'config').mkdir(exist_ok=True)
    (tree / 'config' / 'workspace.toml').write_bytes(WORKSPACE_TOML.encode('utf-8'))
    shipped = sorted(p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file())
    manifest = {'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': shipped, 'sha256': {f: sha256(tree / f) for f in shipped},
                'protocol': dict(path=PROTOCOL, sha256=sha256(tree / PROTOCOL / 'manifest.json')), 'tool': TOOL, 'entry': 'colab-job'}
    (tree / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    package.mkdir(parents=True, exist_ok=True)
    tar_path = package / BUNDLE_NAME
    tmp = tar_path.with_name(tar_path.name + '.tmp')
    with tarfile.open(tmp, 'w:gz', format=tarfile.PAX_FORMAT) as tar:
        for rel in [*shipped, 'bundle_manifest.json']:
            info = tar.gettarinfo(str(tree / rel), arcname=rel)
            info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, '', '', 0o644
            with (tree / rel).open('rb') as f:
                tar.addfile(info, f)
    os.replace(tmp, tar_path)
    digest = sha256(tar_path)
    (package / (BUNDLE_NAME + '.sha256')).write_text(f'{digest}  {BUNDLE_NAME}
', encoding='utf-8')
    meta = {'title': BUNDLE_DATASET, 'id': f'{KAGGLE_USER}/{BUNDLE_DATASET}', 'licenses': [{'name': 'other'}], 'isPrivate': True,
            'description': 'Private: EXP114 RadImageNet ResNet50 relabel_v1 5-fold Colab runner (resumable) as one tar.gz: code, frozen protocol '
                           'inputs (EXP084 split, EXP099 folds), relabel_v1 and Dread labels (no report text, no Gold, no weights).'}
    (package / 'dataset-metadata.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    report = dict(tree=str(tree), files=len(shipped) + 1, bytes=sum((tree / f).stat().st_size for f in shipped), tar=str(tar_path),
                  tar_bytes=tar_path.stat().st_size, tar_sha256=digest, protocol=manifest['protocol'], dataset_dir=str(package))
    _json_line(report)
    return report


def publish_command(package=PACKAGE):
    """The optional private-dataset route (NOT needed when sync_exp114.py deploy uploads the tar with `colab upload`). Never run here;
    absolute backslash path per D-B-015 items 9/16; the user/agent with the Kaggle CLI token runs it only if the dataset route is chosen."""
    return f'kaggle datasets create -p "{Path(package).resolve()}"'


# ------------------------------------------------------------------ assembly of downloaded folds (local)
def assemble(runs_dir=None, root=ROOT, out_dir=None):
    import pandas as pd
    root = Path(root)
    manifest_path = _manifest_path(root)
    p = read(manifest_path); manifest_sha = sha256(manifest_path)
    runs = Path(runs_dir).resolve() if runs_dir else root / LOCAL_RUNS
    out = root / (out_dir or OOF_DIR)
    if out.exists():
        raise SystemExit('output exists: ' + str(out))
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in p['folds'].items()}
    frames24, frames36, sources, problems = {}, {}, {}, []
    for k in sorted(val_ids):
        run = runs / tag_for(k)
        bad = verify_run(run, p, k, manifest_sha)
        if bad:
            problems += bad; continue
        prov, s = read(run / 'provenance.json'), read(run / 'summary.json')
        frames24[k] = pd.read_csv(run / 'oof_best.csv', dtype={UID: str})
        frames36[k] = pd.read_csv(run / EXTRA_EVAL['file'], dtype={UID: str})
        sources[k] = dict(run=display_path(run, root), oof_k24_sha256=sha256(run / 'oof_best.csv'), oof_k36_sha256=sha256(run / EXTRA_EVAL['file']),
                          checkpoint_sha256=read(run / 'output-inventory.json')['model_best.pt']['sha256'], seed=prov.get('seed'),
                          initial_state_sha256=prov.get('initial_state_sha256'), initial_state_matches_registered=prov.get('initial_state_matches_registered'),
                          environment=prov.get('environment'), sessions=s.get('sessions'), fp16_skipped_updates=s.get('skipped_updates'),
                          training_seconds_all_sessions=s.get('training_seconds_all_sessions'))
    if problems:
        _json_line(dict(status='INCOMPLETE', problems=problems))
        raise SystemExit(2)
    references = p['evaluation']['references']
    results = {}
    for name, path in references.items():
        try:
            results[name] = k99.assemble_and_score(frames24, frames36, val_ids, root / path)
        except KeyError as e:  # a reference that does not label every OOF study is reported, not silently dropped
            results[name] = dict(error=f'reference does not cover study {e}')
    primary = results['dread']
    out.mkdir(parents=True)
    primary['oof_k24'].to_csv(out / 'oof_k24.csv', index=False)
    primary['oof_k36'].to_csv(out / 'oof_k36.csv', index=False)
    body = dict(schema='exp114-oof-v1', label_set=LABEL_SET, label_set_sha256=p['label_set']['sha256'], observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                protocol=PROTOCOL, protocol_sha256=manifest_sha, seed=SEED, exp113_protocol_sha256=BASE_MANIFEST_SHA256, runs_dir=display_path(runs, root),
                rows=int(len(primary['oof_k24'])), sources={str(k): v for k, v in sources.items()},
                outputs={f: sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv')},
                scoring=dict(primary='dread', references=references, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py'),
                metrics=primary['metrics'], k36_missing_folds=primary['k36_missing'],
                metrics_by_reference={n: (r['metrics'] if 'metrics' in r else r) for n, r in results.items()},
                caveats=['trained on relabel_v1; Dread is the primary scoring reference (as EXP099/EXP107/EXP113); relabel_v1 and flight_v4 share '
                         'the report-text ceiling (D-B-014)', 'fold 0 val = exposed G1 V888; folds 1-4 models saw G1 val in training',
                         'pooled AUC mixes 5 models', 'Colab T4 trajectories (relaxed environment; resumed sessions recorded per fold): an '
                         'independent member, never paired with EXP107 or EXP113 (D-B-008)'])
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    md = k99.result_markdown('dread', body).replace('# EXP099 B8 5-fold OOF (dread)', '# EXP114 RadImageNet ResNet50 relabel_v1 5-fold OOF (Colab T4), scored vs Dread', 1)
    lines = ['', '## Other references (pooled macro12)', '', '| reference | K24 | K36 |', '|---|---:|---:|']
    for n, m in body['metrics_by_reference'].items():
        if 'k24' in m:
            lines.append(f"| {n} | {m['k24']['pooled']['macro12']:.6f} | {m['k36']['pooled']['macro12']:.6f} |")
        else:
            lines.append(f"| {n} | {m.get('error')} | |")
    (out / 'result.md').write_text(md + '\n'.join(lines) + '\n', encoding='utf-8')
    _json_line(dict(rows=body['rows'], pooled={n: ({kk: round(vv['pooled']['macro12'], 6) for kk, vv in m.items()} if 'k24' in m else m)
                                              for n, m in body['metrics_by_reference'].items()}))
    return body


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--dry-run', default=None, help='write the registration to this new directory instead (outside artifacts/)')
    s = sub.add_parser('argv'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--mode', choices=['full', 'smoke'], default='full')
    s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=COLAB['results']); s.add_argument('--ckpt-dir', default=COLAB['ckpt'])
    s.add_argument('--weights', default=None)
    s = sub.add_parser('validate'); s.add_argument('--fold', type=int, choices=FOLDS, required=True)
    s = sub.add_parser('smoke'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--out', required=True)
    s.add_argument('--train', type=int, default=8); s.add_argument('--val', type=int, default=1); s.add_argument('--epochs', type=int, default=2)
    s.add_argument('--workers', type=int, default=2); s.add_argument('--stop-after-epoch', type=int, default=None); s.add_argument('--ckpt-dir', default=None)
    sub.add_parser('bundle')
    sub.add_parser('publish-command')
    for name in ('colab-job', 'run-folds'):
        s = sub.add_parser(name)
        s.add_argument('--folds', default='0,1,2,3,4'); s.add_argument('--done-elsewhere', default='', help='comma list of tags already verified locally')
        s.add_argument('--results', default=COLAB['results']); s.add_argument('--ckpt-dir', default=COLAB['ckpt']); s.add_argument('--xfer', default=COLAB['xfer'])
        s.add_argument('--workers', type=int, default=None); s.add_argument('--attempts', type=int, default=2)
        s.add_argument('--keep-ckpt', type=int, default=3); s.add_argument('--part-mib', type=int, default=32)
        if name == 'colab-job':
            s.add_argument('--corpus', default=COLAB['corpus']); s.add_argument('--weights-dir', default=COLAB['weights'])
            s.add_argument('--kaggle-user', default=KAGGLE_USER); s.add_argument('--wait-prefetch-hours', type=float, default=3.0)
        else:
            s.add_argument('--weights', required=True)
    s = sub.add_parser('verify'); s.add_argument('--runs-dir', default=None)
    s = sub.add_parser('assemble'); s.add_argument('--runs-dir', default=None)
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')
    if a.command == 'prepare':
        if a.dry_run and within(a.dry_run, ROOT / 'artifacts'):
            raise SystemExit('--dry-run must be outside artifacts/')
        prepare(dest=a.dry_run)
    elif a.command == 'argv':
        _json_line(runner_argv(_protocol(), a.fold, a.mode, workers=a.workers, output_root=a.output_root, ckpt_dir=a.ckpt_dir, weights=a.weights))
    elif a.command == 'validate':
        sys.exit(validate_cli(a.fold))
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.fold, a.out, a.train, a.val, a.epochs, a.workers, a.stop_after_epoch, a.ckpt_dir))
    elif a.command == 'bundle':
        build_bundle()
    elif a.command == 'publish-command':
        print(publish_command())
    elif a.command == 'colab-job':
        sys.exit(colab_job(a))
    elif a.command == 'run-folds':
        status = run_folds(ROOT, [int(x) for x in a.folds.split(',') if x != ''], a.results, a.ckpt_dir, a.xfer, a.weights, a.workers,
                           [x for x in a.done_elsewhere.split(',') if x], a.attempts, a.keep_ckpt, a.part_mib)
        sys.exit(0 if status['state'] == 'COMPLETE' else 1)
    elif a.command == 'verify':
        p = _protocol(); manifest_sha = sha256(_manifest_path())
        runs = Path(a.runs_dir) if a.runs_dir else ROOT / LOCAL_RUNS
        _json_line({tag_for(k): (verify_run(runs / tag_for(k), p, k, manifest_sha) or 'OK') for k in FOLDS})
    else:
        assemble(a.runs_dir)


if __name__ == '__main__':
    main()
