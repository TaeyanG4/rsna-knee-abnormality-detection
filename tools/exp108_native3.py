"""EXP108 (H-B-25): the B8 seed-42 fold-0 recipe with native-adjacent DICOM triplets as the three window channels.

  python tools/exp108_native3.py prepare                 # register EXP108_NATIVE3_PROTOCOL_V1 (CPU; needs the finished native3 cache + checks)
  python tools/exp108_native3.py argv [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp108_native3.py validate                # runner --validate-only on CPU, no output
  python tools/exp108_native3.py smoke --out DIR [--train 24 --val 3 --epochs 1 --workers 2]   # tiny CPU fit, DIR outside artifacts/
  python tools/exp108_native3.py queue-json [--after EXP106_B8_E16_F0] [--out FILE]
  python tools/exp108_native3.py compare-argv            # tools/g1_compare.py command for the finished run

Arm EXP108_B8_NATIVE3_F0 = EXP094 V4 arm seed42 = EXP094L_B8_SEED42 (B8: CoAtNet-2 rmlp 224, FP16 + GradScaler init 1024, 8 epochs
fixed-final, bs 8, lr 3e-5 / 1e-3, wd 0.02, grad checkpointing, strict runtime, within-slot-valid windows, K12 train / K24 eval,
clipped pos_weight, seed 42, init e4ea3ae6..., Gold blocker, no metric access) on the G1 split (train 3455 / val 888), Dread labels.
The ONLY changed variable is the window channel source:
  [cache c-1, c, cache c+1] of RAPTOR64_336_C140_S02  ->  [native c-1, c, native c+1] of RAPTOR64N3_336_C140_S02
(training/src/raptor_native3.py, raptor_native3_data.py, built by raptor_native3_corpus.py). The centre plane, the 64-entry mask,
the slot layout, the centres drawn per epoch (same sha256(seed, epoch, uid) streams) and the intensity jitter are byte-identical to
the baseline; the outer channels are the slices directly before/after the picked slice in the series' IOP/IPP order, mapped with the
slot's own p2/p98 window (computed over the 64 picks, as before), with the centre replicated at series edges.
Extra evaluation of the same final model: K36 (oof_k36.csv; D-B-012/D-B-022 serve K36; baseline K36 = EXP096_K_SWEEP/B_K36.csv).
Per-run contract: len_train_loader = 3455 // 8 = 431, steps = 3448, FP16 skip gate 34, eval batch 4. Nothing here trains or touches the GPU.
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
from tools.exp084_group_loss import (worker_init, read_rows, install_gold_blocker,  # noqa: E402,F401
                                     uid_hash, relpath, sha256, read, CODE_NAMES, ENV_KEYS, THREAD_ENV)
from tools import exp099_kfold as k99  # noqa: E402
from tools.exp099_kfold import canonical_order, validate_fold_membership, pos_weight, contract, UID, T  # noqa: E402,F401

SCHEMA = 'exp108-native3-b8-v1'
PLAN_ITEM = 'H-B-25'
PROTOCOL = 'artifacts/experiments/EXP108_NATIVE3_PROTOCOL_V1'
BASE = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
V4 = 'artifacts/experiments/EXP094_B8_VARIANT_PROTOCOL_V4/manifest.json'
RUNNER = 'training/runners/exp108_native3.py'
TOOL = 'tools/exp108_native3.py'
VERIFY_TOOL = 'tools/exp108_native3_verify.py'
EXT_FILES = ['training/src/raptor_native3.py', 'training/src/raptor_native3_data.py', 'training/src/raptor_native3_corpus.py',
             'training/src/raptor_corpus.py']
CONTRACT = 'raptor64n3_336_c140_s02'
CORPUS = 'storage/cache/derived/RAPTOR64N3_336_C140_S02'
BASE_CONTRACT = 'raptor64_336_c140_s02'
BASE_CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
CHECKS = {'dicom': CORPUS + '/checks/dicom_verify.json', 'consistency': CORPUS + '/checks/consistency.json'}
DREAD = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
BASELINE_RUN = 'artifacts/experiments/EXP094L_B8_SEED42'
BASELINE_K36 = 'artifacts/experiments/EXP096_K_SWEEP/B_K36.csv'
GPU_PYTHON = Path(r'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe')
SEED = 42
FOLD = 0
AFTER = 'EXP106_B8_E16_F0'
TAGS = {'full': 'EXP108_B8_NATIVE3_F0', 'smoke': 'EXP108_B8_NATIVE3_F0_SMOKE'}
MODES = tuple(TAGS)
NATIVE3_SPEC_KEYS = ('planes', 'plane_order', 'neighbours', 'neighbour_normalisation', 'base_contract')
EXTRA_EVALS = [dict(name='k36', k_eval=36, file='oof_k36.csv', batch='eval_batch',
                    source='D-B-012 / D-B-022: same final model with 36 evenly spaced within-slot-valid windows; baseline ' + BASELINE_K36)]
SMOKE_ALLOWED = {'device': ('cuda', 'cpu'), 'amp': ('fp16', 'none'), 'eval_amp': ('fp16', 'none')}
RESOURCE_ARGS = dict(workers=dict(default=12, min=0, max=16, note='DataLoader workers (EXP094 V4 used 12). Window sampling is sha256(seed, epoch, uid)-'
                                                                  'deterministic and batch order comes from a seeded generator, so the count changes speed only'))


# ------------------------------------------------------------------ small helpers
def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def tag_for(mode):
    if mode not in TAGS:
        raise ValueError('unknown mode: ' + str(mode))
    return TAGS[mode]


def is_smoke(mode):
    return mode.endswith('smoke')


def train_python():
    return str(GPU_PYTHON) if GPU_PYTHON.exists() else sys.executable


def within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ pure recipe / contract logic (unit-tested)
def recipe_args(v4_common):
    """EXP094 V4 common_args unchanged (K12 / K24 included); only the resource argument `workers` is moved out."""
    if v4_common != k99.COMMON:
        raise ValueError('EXP094 V4 common_args differ from the B8 recipe frozen in EXP099')
    if (v4_common['k'], v4_common['k_eval'], v4_common['window_policy']) != (12, 24, 'within-slot-valid'):
        raise ValueError('base windows are not K12 / K24 within-slot-valid')
    if v4_common['workers'] != RESOURCE_ARGS['workers']['default']:
        raise ValueError('V4 worker count differs from the registered default')
    return {k: v for k, v in v4_common.items() if k != 'workers'}


def run_contract(n_train, args):
    """EXP099 rule from the train size: len_train_loader = n_train // bs, steps = len * epochs (accum 1), skip gate steps // 100."""
    c = contract(n_train, args)
    if args['accum'] != 1 or c['steps'] != (n_train // args['bs']) * args['epochs']:
        raise AssertionError('contract rule drifted')
    return c


def corpus_checks(n3_manifest, base_manifest, n3_ids, base_ids, n3_masks, base_masks):
    """The native3 cache differs from the base cache only by the two outer planes per slot position; same studies, order and masks."""
    import numpy as np
    d, b = n3_manifest, base_manifest
    problems = []
    if d.get('contract') != CONTRACT or b.get('contract') != BASE_CONTRACT:
        problems.append('contract names')
    if not d.get('finished') or not all(s.get('done') for s in d.get('shards', [])):
        problems.append('native3 cache not finished')
    spec_core = {k: v for k, v in d.get('spec', {}).items() if k not in NATIVE3_SPEC_KEYS}
    if spec_core != b.get('spec'):
        problems.append('spec differs from the base spec beyond the native3 keys')
    if d.get('spec', {}).get('planes') != 3 or d.get('spec', {}).get('plane_order') != ['native_prev', 'centre', 'native_next']:
        problems.append('native3 plane description')
    if d.get('shape_total') != [b['studies'], 64, 3, b['spec']['img'], b['spec']['img']]:
        problems.append('native3 shape')
    for key in ('studies', 'shard_size', 'order'):
        if d.get(key) != b.get(key):
            problems.append('manifest differs: ' + key)
    if d.get('base_builder') != b.get('builder'):
        problems.append('base builder version differs')
    if d.get('archive', {}).get('directory_sha256') != b.get('archive', {}).get('directory_sha256'):
        problems.append('source archive differs')
    if d.get('centre_mask_mismatches') != 0 or d.get('masks_equal_base') is not True:
        problems.append('build reported centre/mask differences to the base cache')
    if not np.array_equal(np.asarray(n3_ids).astype(str), np.asarray(base_ids).astype(str)):
        problems.append('study ids differ from the base cache')
    if not np.array_equal(np.asarray(n3_masks), np.asarray(base_masks)):
        problems.append('masks differ from the base cache')
    return problems


def check_reports(dicom, consistency):
    """Pass conditions for the two verification reports written by tools/exp108_native3_verify.py."""
    problems = []
    if dicom['rendered'] < 10 or dicom['cache_equal'] != dicom['rendered'] or dicom['centre_equal_dicom64'] != dicom['rendered'] \
            or dicom['centre_equal_base_cache'] != dicom['rendered'] or dicom['mask_equal'] != dicom['rendered']:
        problems.append('independent DICOM render does not match the cache on every rendered study')
    if any(not (f.get('cache_all_zero') or f.get('base64_fails_too')) for f in dicom.get('render_failed', [])):
        problems.append('a study failed only in the native3 render (the 64-slot serving path renders it)')
    if consistency['failures'] != 0 or consistency['checks'] < 50000 or not consistency['ids_equal_base'] or not consistency['masks_equal_base']:
        problems.append('whole-cache consistency check failed')
    return problems


def state_hash(model):
    """Combined sha256 over sorted state_dict keys and tensor bytes (the EXP094 / EXP099 rule)."""
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_initial_model(arch, seed):
    """Same order as the runner: bounded threads -> seed_all -> build_model (pretrained timm weights from the local HF cache)."""
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_common import build_model, seed_all
    worker_init(0)
    seed_all(seed)
    return build_model(arch, pretrained=True)


# ------------------------------------------------------------------ prepare
def prepare(root=ROOT, model_factory=None, write=True):
    import numpy as np
    root = Path(root)
    destination = root / PROTOCOL
    if write and destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + PROTOCOL)
    base, v4 = read(root / BASE), read(root / V4)
    arm = v4['arms']['seed42']
    if arm['seed'] != SEED or arm['variant'] != 'seed' or arm.get('slot_dropout', 0.0) != 0.0:
        raise ValueError('EXP094 V4 seed42 arm changed')
    common = recipe_args(v4['common_args'])
    for name, expected in v4['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    if base['ordered_ids'] != v4['ordered_ids'] or base['groups']['sha256'] != v4['groups']['sha256']:
        raise ValueError('EXP084 and EXP094 V4 bind different G1 split files')
    ordered_path, groups_path = base['ordered_ids']['path'], base['groups']['path']
    if sha256(root / ordered_path) != base['ordered_ids']['sha256'] or sha256(root / groups_path) != base['groups']['sha256']:
        raise ValueError('G1 split files changed')
    if v4['paths']['labels'] != DREAD or v4['paths']['eval_labels'] != DREAD or v4['paths']['corpus'] != BASE_CORPUS:
        raise ValueError('EXP094 V4 paths changed')
    labels, label_hash = k99._load_labels(root / DREAD)
    if label_hash != v4['label_hash'] or sha256(root / DREAD) != v4['labels']['sha256']:
        raise ValueError('Dread labels differ from EXP094 V4')
    # caches
    n3_dir, base_dir = root / CORPUS, root / BASE_CORPUS
    dm, bm = read(n3_dir / 'manifest.json'), read(base_dir / 'manifest.json')
    if sha256(base_dir / 'manifest.json') != v4['corpus_manifest_sha256']:
        raise ValueError('base cache manifest differs from EXP094 V4')
    n3_ids = np.load(n3_dir / 'ids.npy', allow_pickle=True).astype(str)
    base_ids = np.load(base_dir / 'ids.npy', allow_pickle=True).astype(str)
    n3_masks, base_masks = np.load(n3_dir / 'masks.npy'), np.load(base_dir / 'masks.npy')
    problems = corpus_checks(dm, bm, n3_ids, base_ids, n3_masks, base_masks)
    reg = dm.get('code_registry') or {}
    if (reg.get('module_sha256') != sha256(root / EXT_FILES[0]) or reg.get('builder_sha256') != sha256(root / EXT_FILES[2])
            or reg.get('base_builder_sha256') != sha256(root / EXT_FILES[3])
            or reg.get('raptor_common_sha256') != v4['training_code_sha256']['raptor_common.py']):
        problems.append('cache was not built by the bound native3 module / builder')
    if dm.get('base_corpus', {}).get('manifest_sha256') != v4['corpus_manifest_sha256']:
        problems.append('cache was built against another base cache manifest')
    if dm.get('mask_sha256') != sha256(n3_dir / 'masks.npy') or dm.get('ids_sha256') != sha256(n3_dir / 'ids.npy'):
        problems.append('native3 masks/ids differ from their manifest hashes')
    dicom, consistency = read(root / CHECKS['dicom']), read(root / CHECKS['consistency'])
    problems += check_reports(dicom, consistency)
    if problems:
        raise ValueError('native3 cache checks failed: ' + '; '.join(problems))
    # split, labels, loss, contract
    ordered = read(root / ordered_path)
    rows = read_rows(root / groups_path)
    canonical = canonical_order(ordered)
    train, val = validate_fold_membership(n3_ids.tolist(), ordered, rows, canonical)
    if (len(train), len(val)) != (v4['expected_ids']['train'], v4['expected_ids']['val']) or (len(train), len(val)) != (3455, 888):
        raise ValueError('G1 sizes changed')
    if uid_hash(train) != base['ordered_ids']['train_sha256'] or uid_hash(val) != base['ordered_ids']['val_sha256']:
        raise ValueError('G1 ordered ids changed')
    pw = pos_weight(labels, train)
    if not np.array_equal(pw, np.asarray(v4['loss']['baseline_pos_weight'], np.float32)):
        raise ValueError('pos_weight differs from the B8 baseline vector')
    c = run_contract(len(train), common)
    if (c['len_train_loader'], c['steps'], c['max_fp16_skips'], c['eval_batch']) != (431, 3448, 34, 4):
        raise ValueError('run contract differs from the B8 fold-0 contract')
    # baseline predictions this arm is compared with
    binv = read(root / BASELINE_RUN / 'output-inventory.json')
    for name in ('model_best.pt', 'oof_best.csv'):
        if sha256(root / BASELINE_RUN / name) != binv[name]['sha256']:
            raise ValueError('baseline run file differs from its inventory: ' + name)
    bprov = read(root / BASELINE_RUN / 'provenance.json')
    if bprov['initial_state_sha256'] != v4['initial_state_sha256'] or bprov['eval_ids_ordered_sha256'] != base['ordered_ids']['val_sha256']:
        raise ValueError('baseline run is not the B8 seed-42 G1 run')
    import pandas as pd
    k36 = pd.read_csv(root / BASELINE_K36, dtype={UID: str})
    if k36[UID].tolist() != val or list(k36.columns[1:]) != T:
        raise ValueError('baseline K36 predictions do not cover the G1 val ids in order')
    # initial state (CPU)
    model = (model_factory or build_initial_model)(common['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init = state_hash(model)
    if init != v4['initial_state_sha256']:
        raise ValueError(f'seed-42 initial state {init} differs from the registered B state {v4["initial_state_sha256"]}')
    import timm
    bindings = [RUNNER, TOOL, VERIFY_TOOL, 'tools/exp099_kfold.py', 'tools/exp087_training.py', 'tools/exp084_group_loss.py', *EXT_FILES, BASE, V4, DREAD,
                ordered_path, groups_path, BASELINE_RUN + '/output-inventory.json', BASELINE_RUN + '/oof_best.csv', BASELINE_K36,
                CHECKS['dicom'], CHECKS['consistency']]
    shard_stats = {}
    for s in dm['shards']:
        for key, value in (s.get('neighbour_stats') or {}).items():
            shard_stats[key] = shard_stats.get(key, 0) + value
    manifest = dict(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item=PLAN_ITEM,
        sources=['D-A-011', 'D-A-016', 'D-B-013', 'D-B-017', 'D-B-019', 'D-B-023'],
        source_bindings={s: sha256(root / s) for s in bindings},
        training_code_sha256=v4['training_code_sha256'], environment=v4['environment'],
        corpus=dict(path=CORPUS, contract=CONTRACT, manifest_sha256=sha256(n3_dir / 'manifest.json'), depth=64, planes=3,
                    slots=dm['spec']['slots'], studies=int(dm['studies']), ids_sha256=dm['ids_sha256'], masks_sha256=dm['mask_sha256'],
                    ids_equal_base=True, masks_equal_base=True, centre_planes_equal_base=True,
                    shards={s['file']: s['sha256'] for s in dm['shards']}, code_registry=reg, neighbour_stats=shard_stats,
                    build_seconds=round(sum(s.get('seconds', 0) for s in dm['shards']), 1), bytes=int(sum(s.get('bytes', 0) for s in dm['shards']))),
        base_corpus=dict(path=BASE_CORPUS, contract=BASE_CONTRACT, manifest_sha256=v4['corpus_manifest_sha256']),
        corpus_manifest_sha256=sha256(n3_dir / 'manifest.json'),
        cache_checks=dict(dicom={k: v for k, v in dicom.items() if k != 'results'}, consistency=consistency,
                          files={k: dict(path=v, sha256=sha256(root / v)) for k, v in CHECKS.items()}),
        initial_state_sha256=v4['initial_state_sha256'],
        initial_state=dict(computed_on='cpu', seed=SEED, arch=common['arch'], torch=torch.__version__, timm=timm.__version__, equals_b=True,
                           rule='worker_init(0), seed_all(42), build_model(arch, pretrained timm); sha256 over sorted state_dict (key + bytes)'),
        recipe=dict(source=V4, arm='seed42', base_common_args=v4['common_args'], baseline_run=BASELINE_RUN,
                    changes=['window channels: [cache c-1, c, cache c+1] of RAPTOR64_336_C140_S02 -> [native c-1, c, native c+1] of '
                             'RAPTOR64N3_336_C140_S02 (same centre plane, mask, slot layout, centres and jitter; outer channels = the slices '
                             'directly before/after the picked slice in IOP/IPP order, slot p2/p98 window of the 64 picks, centre replicated at '
                             'series edges)',
                             'extra evaluation of the same final model: K36 (oof_k36.csv)']),
        common_args=common, seed=SEED, resource_args=RESOURCE_ARGS, smoke_allowed=SMOKE_ALLOWED,
        paths=dict(corpus=CORPUS, base_corpus=BASE_CORPUS, labels=DREAD, eval_labels=DREAD, ordered_ids=ordered_path, groups=groups_path),
        label_set=dict(name='dread', path=DREAD, sha256=v4['labels']['sha256'], semantic_sha256=label_hash),
        excluded_uids=v4['excluded_uids'],
        split=dict(fold=FOLD, ordered_ids=dict(base['ordered_ids']), groups=dict(base['groups']), n_train=len(train), n_val=len(val),
                   train_components=v4['expected_ids']['train_components'], val_components=v4['expected_ids']['val_components'],
                   **c, pos_weight=pw.tolist(), note='G1 = EXP099 fold 0 = EXP094L_B8_SEED42 cohort'),
        contract_rule='len_train_loader = n_train // bs (drop_last); steps = ceil(len_train_loader/accum) * epochs = scheduler total_steps; '
                      'FP16 skip gate: skipped > steps // 100 fails; eval batch bs // 2',
        extra_evals=EXTRA_EVALS,
        baseline=dict(k24=dict(path=BASELINE_RUN + '/oof_best.csv', sha256=binv['oof_best.csv']['sha256'], note='g1_compare reference B (EXP084_B_G1 = EXP094L_B8_SEED42 bit-exact, D-B-017)'),
                      k36=dict(path=BASELINE_K36, sha256=sha256(root / BASELINE_K36), note='B checkpoint re-evaluated at K36 (EXP096, D-B-012)')),
        tags=dict(TAGS), training_budgets=dict(full_seconds=21600, retry=0), queue=dict(after=AFTER, continue_on_failure=True),
        evaluation=dict(labels=DREAD, rule='float32 then numpy.rint; tools/g1_compare.py (873 components, 500 draws, seed 20261003)',
                        primary='macro12(EXP108 K24) - macro12(B K24)', secondary='macro12(EXP108 K36) - macro12(B K36)',
                        hypothesis='>= +0.006 (H-B-25); adopt only if the gain exceeds the same-machine seed spread (H-B-17 part 2) or replicates on another fold/seed'),
        caveats=['single seed, single split (fold 0 is the easiest fold, D-B-022); repeat noise of separately trained models is about +-0.006 (D-B-010)',
                 'the window-validity rule still looks at the cache neighbours c-1, c+1 (mask); a native neighbour can be valid when the cache neighbour is not, '
                 'and vice versa - the same centres are used on purpose so only the channel source changes',
                 'for dense picks (step 1, short series) the native neighbour equals the cache neighbour, so part of the windows is unchanged '
                 '(fraction in cache_checks.consistency); slot-edge picks never are window centres',
                 'neighbours inherit the slot window (computed over the 64 picks only): a neighbour outside that population can clip more than a pick',
                 'serving needs training/src/raptor_native3.py (DicomCorpusNative3 / prepare_input_native3) and ~3x the decode of picked slices; '
                 'the shared-decode adapter of D-B-018 covers only the picks'])
    if write:
        destination.mkdir(parents=True)
        dump(destination / 'manifest.json', manifest)
        dump(destination / 'queue.json', queue_entries(manifest, after=AFTER, root=root))
        print(json.dumps(dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, manifest_sha256=sha256(destination / 'manifest.json'),
                              initial_state_sha256=init, corpus_manifest_sha256=manifest['corpus_manifest_sha256'], split=dict(
                                  n_train=len(train), n_val=len(val), len_train_loader=c['len_train_loader'], steps=c['steps'],
                                  max_fp16_skips=c['max_fp16_skips'])), indent=1))
    return manifest


# ------------------------------------------------------------------ runner guard
def validate_arguments(args, root):
    """Called by the runner before any training import. Returns (protocol, ordered ids, output root)."""
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
    if args.mode not in MODES:
        raise ValueError('unknown mode')
    smoke = is_smoke(args.mode)
    for k, v in p['common_args'].items():
        got = getattr(args, k, None)
        if got != v and not (smoke and k in p['smoke_allowed'] and got in p['smoke_allowed'][k]):
            raise ValueError('frozen argument changed: ' + k)
    if args.seed != p['seed'] or args.fold != FOLD:
        raise ValueError('seed/fold changed')
    w = p['resource_args']['workers']
    if not w['min'] <= args.workers <= w['max']:
        raise ValueError('workers outside the registered range')
    if args.mode == 'full' and args.workers != w['default']:
        raise ValueError('the full run uses the registered worker count (EXP094 V4 conditions)')
    if tag_for(args.mode) != args.tag:
        raise ValueError('tag/mode mismatch')
    if Path(args.corpus).resolve() != relpath(root, p['paths']['corpus']).resolve():
        raise ValueError('input path differs: corpus')
    for key in ('labels', 'eval_labels'):
        if Path(getattr(args, key)).resolve() != relpath(root, p['paths'][key]).resolve():
            raise ValueError('input path differs: ' + key)
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    split = p['split']
    for key in ('ordered_ids', 'groups'):
        if sha256(relpath(root, split[key]['path'])) != split[key]['sha256']:
            raise ValueError(key + ' file changed')
    ordered = read(relpath(root, split['ordered_ids']['path']))
    for k in ('train', 'val'):
        if uid_hash(ordered[k]) != split['ordered_ids'][k + '_sha256']:
            raise ValueError('ordered ids changed')
    default_root = (root / 'artifacts' / 'experiments').resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else default_root
    if smoke:
        if not args.output_root or within(out_root, root / 'artifacts'):
            raise ValueError('smoke runs need --output-root outside artifacts/')
        if args.smoke_train < args.bs or args.smoke_val < 1 or args.smoke_epochs < 1:
            raise ValueError('smoke needs >= one batch of training studies, >= 1 validation study and >= 1 epoch')
    elif out_root != default_root:
        raise ValueError('full runs write to artifacts/experiments')
    return p, ordered, out_root


def _set_flag(argv, flag, value):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag, str(value)]
    return argv


def runner_argv(p, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None):
    root = Path(root)
    argv = [python or train_python(), str(root / RUNNER), '--protocol', str(root / PROTOCOL / 'manifest.json'), '--fold', str(FOLD),
            '--mode', mode, '--tag', tag_for(mode), '--seed', str(p['seed'])]
    for k, v in p['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--workers', str(p['resource_args']['workers']['default'] if workers is None else int(workers))]
    argv += ['--corpus', str(relpath(root, p['paths']['corpus'])), '--labels', str(relpath(root, p['paths']['labels'])),
             '--eval-labels', str(relpath(root, p['paths']['eval_labels']))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if is_smoke(mode):
        smoke = {**dict(device='cpu', amp='none', eval_amp='none', train=24, val=3, epochs=1), **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), smoke[key])
        argv += ['--smoke-train', str(smoke['train']), '--smoke-val', str(smoke['val']), '--smoke-epochs', str(smoke['epochs'])]
    return argv


def queue_entries(p, after=AFTER, root=ROOT):
    """gpu_queue job list (LOCAL_GPU_CHAIN_V8 format): one training job after the given tag, continue_on_failure."""
    entry = dict(tag=tag_for('full'), command=runner_argv(p, 'full', root=root), after=after, continue_on_failure=True)
    if after is None:
        entry.pop('after')
    return [entry]


def _protocol(root=ROOT):
    return read(Path(root) / PROTOCOL / 'manifest.json')


def local_env():
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(ROOT)}
    env.pop('KNEE_REMOTE_RELAX_ENV', None)
    return env


def validate_cli(mode='full'):
    """runner --validate-only on CPU (frozen device stays 'cuda'; the runner never touches CUDA in this path)."""
    argv = runner_argv(_protocol(), mode) + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-6000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(out, train, val, epochs, workers):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    argv = runner_argv(_protocol(), 'smoke', workers=workers, output_root=out, smoke=dict(train=train, val=val, epochs=epochs))
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-6000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for('smoke')
    report = dict(returncode=r.returncode, seconds=round(time.time() - t0), run=str(run), files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json')
        report.update({k: s.get(k) for k in ('optimizer_updates', 'skipped_updates', 'train', 'val', 'history', 'window_counts', 'input_check')})
        report['initial_state_matches_registered'] = s.get('provenance', {}).get('initial_state_matches_registered')
    print(json.dumps(report, indent=1))
    return r.returncode


def compare_argv(root=ROOT):
    py = train_python(); tool = str(Path(root) / 'tools/g1_compare.py')
    run = 'artifacts/experiments/' + TAGS['full']
    return [[py, tool, '--out', 'artifacts/experiments/EXP108_NATIVE3_REPORT',
             '--candidate', f'n3k24={run}/oof_best.csv', '--candidate', f'n3k36={run}/oof_k36.csv', '--candidate', f'bk36={BASELINE_K36}',
             '--contrast', 'n3k24-B', '--contrast', 'n3k36-bk36', '--contrast', 'n3k36-B', '--contrast', 'bk36-B']]


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('argv'); s.add_argument('--mode', choices=MODES, default='full'); s.add_argument('--workers', type=int, default=None)
    s.add_argument('--output-root', default=None)
    sub.add_parser('validate')
    s = sub.add_parser('smoke'); s.add_argument('--out', required=True); s.add_argument('--train', type=int, default=24); s.add_argument('--val', type=int, default=3)
    s.add_argument('--epochs', type=int, default=1); s.add_argument('--workers', type=int, default=2)
    s = sub.add_parser('queue-json'); s.add_argument('--after', default=AFTER); s.add_argument('--out', default=None)
    sub.add_parser('compare-argv')
    a = ap.parse_args()
    if a.command == 'prepare':
        for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():  # before torch/timm/huggingface_hub are imported
            os.environ[k] = v
        prepare()
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli())
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.out, a.train, a.val, a.epochs, a.workers))
    elif a.command == 'queue-json':
        text = json.dumps(queue_entries(_protocol(), a.after if a.after != 'none' else None), indent=1)
        if a.out:
            with Path(a.out).open('x', encoding='utf-8') as f:
                f.write(text + '\n')
        print(text)
    else:
        print(json.dumps(compare_argv(), indent=1))


if __name__ == '__main__':
    main()
