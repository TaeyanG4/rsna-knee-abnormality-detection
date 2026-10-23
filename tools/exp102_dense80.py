"""EXP102 (H-B-22): the B8 seed-42 fold-0 recipe on the 80-slice public-layout cache, windows scaled to depth.

  python tools/exp102_dense80.py prepare                 # register EXP102_DENSE80_PROTOCOL_V1 (CPU; needs the finished RAPTOR80 cache)
  python tools/exp102_dense80.py argv [--mode full|smoke|baseline-eval|baseline-eval-smoke] [--workers N] [--output-root DIR]
  python tools/exp102_dense80.py validate [--mode full|baseline-eval]      # runner --validate-only on CPU, no output
  python tools/exp102_dense80.py smoke --out DIR [--train 24 --val 3 --epochs 1 --workers 2]   # tiny CPU fit, DIR outside artifacts/
  python tools/exp102_dense80.py queue-json --after EXP099_B8_DREAD_F4 [--with-baseline-eval] [--out FILE]
  python tools/exp102_dense80.py compare-argv            # tools/g1_compare.py commands for the finished runs

Arm EXP102_B8_DENSE80_F0 = EXP094 V4 arm seed42 (B8: CoAtNet-2 rmlp 224, FP16 + GradScaler init 1024, 8 epochs fixed-final, bs 8,
lr 3e-5 / 1e-3, wd 0.02, grad checkpointing, strict runtime, within-slot-valid windows, clipped pos_weight, seed 42, Gold blocker,
no metric access) on the G1 split (EXP084 ordered-ids: train 3455 / val 888) with Dread labels. Changed variables, nothing else:
  corpus RAPTOR64_336_C140_S02 -> RAPTOR80_336_C140_S02 (contract raptor80_336_c140_s02, 22/18/15/10/15, training/src/raptor_contracts_ext.py)
  K train 12 -> 16, K eval 24 -> 31 (usable centres x1.30, D-B-019)
Extra evaluations of the same final model: K24 (oof_k24.csv, the baseline's window count) and every within-slot-valid centre once
(oof_allpos.csv, batch 1; training/src/raptor_windows_ext.py). Arm EXP102_B8_SEED42_EVAL64 (optional, inference only) re-evaluates the
baseline checkpoint EXP094L_B8_SEED42 on the 64-slice cache at K24 (must reproduce its oof_best.csv) and at all positions, so that
both models can be compared at saturation. The seed-42 initial state is the B state (e4ea3ae6..., model and seed unchanged) and is
recomputed on CPU at prepare and enforced by the runner. Per-run contract from the train size: len_train_loader = 3455 // 8 = 431,
steps = 431 * 8 = 3448, FP16 skip gate 34, eval batch 4. Nothing here trains or touches the GPU.
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

SCHEMA = 'exp102-dense80-b8-v1'
PROTOCOL = 'artifacts/experiments/EXP102_DENSE80_PROTOCOL_V1'
BASE = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
V4 = 'artifacts/experiments/EXP094_B8_VARIANT_PROTOCOL_V4/manifest.json'
RUNNER = 'training/runners/exp102_dense80.py'
TOOL = 'tools/exp102_dense80.py'
EXT_FILES = ['training/src/raptor_contracts_ext.py', 'training/src/raptor_windows_ext.py', 'training/src/raptor_corpus_ext.py', 'training/src/raptor_corpus.py']
CONTRACT = 'raptor80_336_c140_s02'
CORPUS = 'storage/cache/derived/RAPTOR80_336_C140_S02'
BASE_CONTRACT = 'raptor64_336_c140_s02'
BASE_CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
DREAD = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
BASELINE_RUN = 'artifacts/experiments/EXP094L_B8_SEED42'
GPU_PYTHON = Path(r'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe')
SEED = 42
FOLD = 0
K_TRAIN, K_EVAL = 16, 31
TAGS = {'full': 'EXP102_B8_DENSE80_F0', 'smoke': 'EXP102_B8_DENSE80_F0_SMOKE',
        'baseline-eval': 'EXP102_B8_SEED42_EVAL64', 'baseline-eval-smoke': 'EXP102_B8_SEED42_EVAL64_SMOKE'}
MODES = tuple(TAGS)
EXTRA_EVALS = [dict(name='k24', k_eval=24, file='oof_k24.csv', batch='eval_batch', source='window count of the 64-slice baseline (EXP094L_B8_SEED42 oof_best.csv)'),
               dict(name='allpos', k_eval='all', file='oof_allpos.csv', batch=1,
                    source='saturation: every within-slot-valid centre once (raptor_windows_ext.AllValidWindows), batch 1; D-B-012, D-B-019')]
BASELINE_EVALS = [dict(name='k24', k_eval=24, file='oof_k24.csv', batch='eval_batch', source='reproduction check against EXP094L_B8_SEED42/oof_best.csv'),
                  dict(name='allpos', k_eval='all', file='oof_allpos.csv', batch=1, source='saturation on the 64-slice cache')]
SMOKE_ALLOWED = {'device': ('cuda', 'cpu'), 'amp': ('fp16', 'none'), 'eval_amp': ('fp16', 'none')}
RESOURCE_ARGS = dict(workers=dict(default=12, min=0, max=16, note='DataLoader workers (EXP094 V4 used 12). Window sampling is sha256(seed, epoch, uid)-'
                                                                  'deterministic and batch order comes from a seeded generator, so the count changes speed only'))
REPRO_TOLERANCE = 1e-5


# ------------------------------------------------------------------ small helpers
def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def tag_for(mode):
    if mode not in TAGS:
        raise ValueError('unknown mode: ' + str(mode))
    return TAGS[mode]


def is_baseline(mode):
    return mode.startswith('baseline-eval')


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
def dense_args(v4_common):
    """EXP094 V4 common_args with only the window counts scaled to depth."""
    if v4_common != k99.COMMON:
        raise ValueError('EXP094 V4 common_args differ from the B8 recipe frozen in EXP099')
    if (v4_common['k'], v4_common['k_eval']) != (12, 24):
        raise ValueError('base window counts are not K12 / K24')
    args = {k: v for k, v in v4_common.items() if k != 'workers'}  # workers: resource argument, default 12 = V4 (enforced for the full run)
    if v4_common['workers'] != RESOURCE_ARGS['workers']['default']:
        raise ValueError('V4 worker count differs from the registered default')
    args.update(k=K_TRAIN, k_eval=K_EVAL)
    return args


def run_contract(n_train, args):
    """EXP099 rule from the train size: len_train_loader = n_train // bs, steps = len * epochs (accum 1), skip gate steps // 100."""
    c = contract(n_train, args)
    if args['accum'] != 1 or c['steps'] != (n_train // args['bs']) * args['epochs']:
        raise AssertionError('contract rule drifted')
    return c


def corpus_checks(dense_manifest, base_manifest, slots80, dense_ids, base_ids):
    """The dense cache differs from the base cache only in the slot table (and depth); same studies in the same order."""
    import numpy as np
    d, b = dense_manifest, base_manifest
    problems = []
    if d.get('contract') != CONTRACT or b.get('contract') != BASE_CONTRACT:
        problems.append('contract names')
    if not d.get('finished') or not all(s.get('done') for s in d.get('shards', [])):
        problems.append('dense cache not finished')
    if [list(s) for s in d['spec']['slots']] != [list(s) for s in slots80] or int(d['spec']['depth']) != 80:
        problems.append('dense slot table')
    for key in ('img', 'crop_mm', 'span', 'normalization', 'order', 'resize', 'series_selection'):
        if d['spec'].get(key) != b['spec'].get(key):
            problems.append('spec differs: ' + key)
    for key in ('studies', 'shard_size', 'order', 'builder'):
        if d.get(key) != b.get(key):
            problems.append('manifest differs: ' + key)
    if d.get('archive', {}).get('directory_sha256') != b.get('archive', {}).get('directory_sha256'):
        problems.append('source archive differs')
    if not np.array_equal(np.asarray(dense_ids).astype(str), np.asarray(base_ids).astype(str)):
        problems.append('study ids differ from the base cache')
    return problems


def window_coverage(masks, spec, rows, k_train, k_eval):
    """Per-study within-slot-valid centre counts for the given cache rows; how often K exceeds them (repeats)."""
    import numpy as np
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_windows_ext import valid_centers
    depth = int(spec['depth'])
    n = np.array([len(valid_centers(masks[r], depth, spec)) for r in rows])
    return dict(studies=int(len(n)), mean=round(float(n.mean()), 3), min=int(n.min()), p05=float(np.percentile(n, 5)), max=int(n.max()),
                zero=int((n == 0).sum()), below_k_train=int((n < k_train).sum()), below_k_eval=int((n < k_eval).sum()))


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
    common = dense_args(v4['common_args'])
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
    sys.path.insert(0, str(root / 'training' / 'src'))
    import raptor_contracts_ext
    dense_dir, base_dir = root / CORPUS, root / BASE_CORPUS
    dm, bm = read(dense_dir / 'manifest.json'), read(base_dir / 'manifest.json')
    if sha256(base_dir / 'manifest.json') != v4['corpus_manifest_sha256']:
        raise ValueError('base cache manifest differs from EXP094 V4')
    dense_ids = np.load(dense_dir / 'ids.npy', allow_pickle=True).astype(str)
    base_ids = np.load(base_dir / 'ids.npy', allow_pickle=True).astype(str)
    problems = corpus_checks(dm, bm, raptor_contracts_ext.SLOTS80, dense_ids, base_ids)
    reg = dm.get('contract_registry') or {}
    if reg.get('module_sha256') != sha256(root / EXT_FILES[0]) or reg.get('builder_sha256') != sha256(root / EXT_FILES[3]):
        problems.append('cache was not built by the bound contract module / builder')
    if dm.get('mask_sha256') != sha256(dense_dir / 'masks.npy') or dm.get('ids_sha256') != sha256(dense_dir / 'ids.npy'):
        problems.append('dense masks/ids differ from their manifest hashes')
    if problems:
        raise ValueError('dense cache checks failed: ' + '; '.join(problems))
    # split, labels, loss, contract
    ordered = read(root / ordered_path)
    rows = read_rows(root / groups_path)
    canonical = canonical_order(ordered)
    train, val = validate_fold_membership(dense_ids.tolist(), ordered, rows, canonical)
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
    dmask, bmask = np.load(dense_dir / 'masks.npy'), np.load(base_dir / 'masks.npy')
    drow = {u: i for i, u in enumerate(dense_ids)}
    coverage = {}
    for split, ids in (('train', train), ('val', val)):
        rr = [drow[u] for u in ids]
        coverage[split] = dict(dense80=window_coverage(dmask, dm['spec'], rr, K_TRAIN, K_EVAL), base64=window_coverage(bmask, bm['spec'], rr, 12, 24))
    # baseline checkpoint (inference-only arm)
    binv = read(root / BASELINE_RUN / 'output-inventory.json')
    for name in ('model_best.pt', 'oof_best.csv'):
        if sha256(root / BASELINE_RUN / name) != binv[name]['sha256']:
            raise ValueError('baseline run file differs from its inventory: ' + name)
    bprov = read(root / BASELINE_RUN / 'provenance.json')
    if bprov['initial_state_sha256'] != v4['initial_state_sha256'] or bprov['eval_ids_ordered_sha256'] != base['ordered_ids']['val_sha256']:
        raise ValueError('baseline run is not the B8 seed-42 G1 run')
    # initial state (CPU)
    model = (model_factory or build_initial_model)(common['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init = state_hash(model)
    if init != v4['initial_state_sha256']:
        raise ValueError(f'seed-42 initial state {init} differs from the registered B state {v4["initial_state_sha256"]}')
    import timm
    bindings = [RUNNER, TOOL, 'tools/exp099_kfold.py', 'tools/exp087_training.py', 'tools/exp084_group_loss.py', *EXT_FILES, BASE, V4, DREAD,
                ordered_path, groups_path, BASELINE_RUN + '/output-inventory.json', BASELINE_RUN + '/oof_best.csv']
    manifest = dict(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item='H-B-22',
        sources=['D-B-019', 'D-B-017', 'D-B-012'],
        source_bindings={s: sha256(root / s) for s in bindings},
        training_code_sha256=v4['training_code_sha256'], environment=v4['environment'],
        corpus=dict(path=CORPUS, contract=CONTRACT, manifest_sha256=sha256(dense_dir / 'manifest.json'), depth=80, slots=dm['spec']['slots'],
                    studies=int(dm['studies']), ids_sha256=dm['ids_sha256'], masks_sha256=dm['mask_sha256'], ids_equal_base=True,
                    shards={s['file']: s['sha256'] for s in dm['shards']}, contract_registry=reg),
        base_corpus=dict(path=BASE_CORPUS, contract=BASE_CONTRACT, manifest_sha256=v4['corpus_manifest_sha256']),
        corpus_manifest_sha256=sha256(dense_dir / 'manifest.json'),
        initial_state_sha256=v4['initial_state_sha256'],
        initial_state=dict(computed_on='cpu', seed=SEED, arch=common['arch'], torch=torch.__version__, timm=timm.__version__, equals_b=True,
                           rule='worker_init(0), seed_all(42), build_model(arch, pretrained timm); sha256 over sorted state_dict (key + bytes)'),
        recipe=dict(source=V4, arm='seed42', base_common_args=v4['common_args'], baseline_run=BASELINE_RUN,
                    changes=['corpus: RAPTOR64_336_C140_S02 -> RAPTOR80_336_C140_S02 (public 80-slice layout 22/18/15/10/15, same span/crop/size/normalisation)',
                             'windows: K train 12 -> 16, K eval 24 -> 31 (usable within-slot centres x1.30, D-B-019)',
                             'extra evaluations of the same final model: K24 and all within-slot-valid positions (batch 1)']),
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
        baseline_eval=dict(tag=TAGS['baseline-eval'], checkpoint=BASELINE_RUN + '/model_best.pt', checkpoint_sha256=binv['model_best.pt']['sha256'],
                           reference_oof=BASELINE_RUN + '/oof_best.csv', reference_oof_sha256=binv['oof_best.csv']['sha256'], corpus=BASE_CORPUS,
                           evals=BASELINE_EVALS, reproduction_tolerance=REPRO_TOLERANCE,
                           note='inference only; K24 must reproduce the stored baseline predictions (max |diff| recorded)'),
        window_coverage=coverage,
        tags=dict(TAGS), training_budgets=dict(full_seconds=21600, baseline_eval_seconds=1800, retry=0),
        evaluation=dict(labels=DREAD, rule='float32 then numpy.rint; tools/g1_compare.py (873 components, 500 draws, seed 20261003)',
                        primary='macro12(EXP102 K31) - macro12(B K24); saturation: macro12(EXP102 allpos) - macro12(B allpos on 64)',
                        secondary='fixed K24 contrast; adopt only if the gain is >= the same-machine seed spread (H-B-17) or replicates with seed 43'),
        caveats=['single seed, single split: a +0.004 expectation is inside seed noise (D-B-010, D-B-016)',
                 'repeated slice picks in 16.6% of studies at 80 slots vs 1.9% at 64 (short series): duplicate planes inside a slot',
                 'the 2-98 percentile window is computed over the 80 picked slices, a different population than the 64 picks: intensities of shared slices can differ by a few grey levels',
                 'group-metadata cached_mask subgroups come from the 64-slot cache masks (tools/g1_compare.py diagnostics only)',
                 'K16 training raises GPU memory and time by about a third (B8 peak 7.8 GiB at K12)'])
    if write:
        destination.mkdir(parents=True)
        dump(destination / 'manifest.json', manifest)
        dump(destination / 'queue.json', queue_entries(manifest, after='EXP099_B8_DREAD_F4', with_baseline=True, root=root))
        print(json.dumps(dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, manifest_sha256=sha256(destination / 'manifest.json'),
                              initial_state_sha256=init, corpus_manifest_sha256=manifest['corpus_manifest_sha256'], split=dict(
                                  n_train=len(train), n_val=len(val), len_train_loader=c['len_train_loader'], steps=c['steps'],
                                  max_fp16_skips=c['max_fp16_skips']), window_coverage=coverage), indent=1))
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
    corpus_key = 'base_corpus' if is_baseline(args.mode) else 'corpus'
    if Path(args.corpus).resolve() != relpath(root, p['paths'][corpus_key]).resolve():
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
        if (not is_baseline(args.mode) and args.smoke_train < args.bs) or args.smoke_val < 1 or args.smoke_epochs < 1:
            raise ValueError('smoke needs >= one batch of training studies, >= 1 validation study and >= 1 epoch')
    elif out_root != default_root:
        raise ValueError('full and baseline-eval runs write to artifacts/experiments')
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
    corpus_key = 'base_corpus' if is_baseline(mode) else 'corpus'
    argv += ['--corpus', str(relpath(root, p['paths'][corpus_key])), '--labels', str(relpath(root, p['paths']['labels'])),
             '--eval-labels', str(relpath(root, p['paths']['eval_labels']))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if is_smoke(mode):
        smoke = {**dict(device='cpu', amp='none', eval_amp='none', train=24, val=3, epochs=1), **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), smoke[key])
        argv += ['--smoke-train', str(smoke['train']), '--smoke-val', str(smoke['val']), '--smoke-epochs', str(smoke['epochs'])]
    return argv


def queue_entries(p, after='EXP099_B8_DREAD_F4', with_baseline=False, root=ROOT):
    entries = [dict(tag=tag_for('full'), command=runner_argv(p, 'full', root=root), after=after, continue_on_failure=True)]
    if with_baseline:  # inference only (~5 min); independent of the training result
        entries.append(dict(tag=tag_for('baseline-eval'), command=runner_argv(p, 'baseline-eval', root=root), after=tag_for('full'), continue_on_failure=True))
    return entries


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


def smoke_cli(out, train, val, epochs, workers, mode='smoke'):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    argv = runner_argv(_protocol(), mode, workers=workers, output_root=out, smoke=dict(train=train, val=val, epochs=epochs))
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-6000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for(mode)
    report = dict(returncode=r.returncode, seconds=round(time.time() - t0), run=str(run), files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json')
        report.update({k: s.get(k) for k in ('optimizer_updates', 'skipped_updates', 'train', 'val', 'history', 'window_counts', 'reproduction')})
        report['initial_state_matches_registered'] = s.get('provenance', {}).get('initial_state_matches_registered')
    print(json.dumps(report, indent=1))
    return r.returncode


def compare_argv(root=ROOT):
    py = train_python(); tool = str(Path(root) / 'tools/g1_compare.py')
    run, base = 'artifacts/experiments/' + TAGS['full'], 'artifacts/experiments/' + TAGS['baseline-eval']
    return [[py, tool, '--out', 'artifacts/experiments/EXP102_DENSE80_REPORT',
             '--candidate', f'd80k31={run}/oof_best.csv', '--candidate', f'd80k24={run}/oof_k24.csv', '--candidate', f'd80all={run}/oof_allpos.csv',
             '--candidate', f'b64all={base}/oof_allpos.csv',
             '--contrast', 'd80k31-B', '--contrast', 'd80k24-B', '--contrast', 'd80all-b64all', '--contrast', 'd80all-d80k31', '--contrast', 'b64all-B']]


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('argv'); s.add_argument('--mode', choices=MODES, default='full'); s.add_argument('--workers', type=int, default=None)
    s.add_argument('--output-root', default=None)
    s = sub.add_parser('validate'); s.add_argument('--mode', choices=['full', 'baseline-eval'], default='full')
    s = sub.add_parser('smoke'); s.add_argument('--out', required=True); s.add_argument('--train', type=int, default=24); s.add_argument('--val', type=int, default=3)
    s.add_argument('--epochs', type=int, default=1); s.add_argument('--workers', type=int, default=2)
    s.add_argument('--mode', choices=['smoke', 'baseline-eval-smoke'], default='smoke')
    s = sub.add_parser('queue-json'); s.add_argument('--after', default='EXP099_B8_DREAD_F4'); s.add_argument('--with-baseline-eval', action='store_true')
    s.add_argument('--out', default=None)
    sub.add_parser('compare-argv')
    a = ap.parse_args()
    if a.command == 'prepare':
        for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():  # before torch/timm/huggingface_hub are imported
            os.environ[k] = v
        prepare()
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli(a.mode))
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.out, a.train, a.val, a.epochs, a.workers, a.mode))
    elif a.command == 'queue-json':
        entries = queue_entries(_protocol(), a.after if a.after != 'none' else None, a.with_baseline_eval)
        if entries[0]['after'] is None:
            entries[0].pop('after')
        text = json.dumps(entries, indent=1)
        if a.out:
            with Path(a.out).open('x', encoding='utf-8') as f:
                f.write(text + '\n')
        print(text)
    else:
        print(json.dumps(compare_argv(), indent=1))


if __name__ == '__main__':
    main()
