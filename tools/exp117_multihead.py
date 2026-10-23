"""EXP117 (H-B-47) v1.0.0: label-source heads. One RadR50 trunk exactly as EXP113 (local RTX 4070 Ti SUPER, seed-42 RadImageNet ResNet50,
EXP099 folds, RN_clip recipe, K12 / K24 / K36, same loader contract) with four 12-output heads, one per label source:
dread, relabel_v1, steven_v4 (Steven llm_labels_v4_blend) and pilkwang_v2 (Pilkwang report_labels_v2).

  python tools/exp117_multihead.py prepare [--dry-run DIR]       # register EXP117_MULTIHEAD_PROTOCOL_V1 (CPU; builds the seed-42 model once)
  python tools/exp117_multihead.py argv --fold K [--mode full|smoke] [--workers N] [--output-root DIR]
  python tools/exp117_multihead.py validate --fold K              # runner --validate-only on CPU, no output
  python tools/exp117_multihead.py smoke --fold K --out DIR [--train 16 --val 2 --epochs 1 --workers 2]   # CPU fit, DIR outside artifacts/
  python tools/exp117_multihead.py queue-json --after TAG [--out FILE|-]   # F0 ONLY (the matched screen), continue_on_failure false (not started)
  python tools/exp117_multihead.py queue-json --folds 1-4 --screen-pass F0_SCREEN.json --after EXP117_MULTIHEAD_F0 [--out FILE|-]
                                                                  # F1-F4 only after a recorded F0 screen PASS (astra review 2026-10-08)
  python tools/exp117_multihead.py assemble [--runs-dir DIR]      # pooled OOF (head rank mean + every head) -> EXP117_MULTIHEAD_OOF

Model: the EXP113 seed-42 RaptorClassifier is built exactly as EXP113 (its state must equal EXP113's registered initial_state_sha256), then
its head (LayerNorm + attention pooling + per-finding classifier, every non-backbone key) is copied four times; head 0 (dread) is EXP113's
head. Loss: the sum over heads of the weighted BCE of that head's source, with that source's own EXP099-rule pos_weight per fold
(recomputed from its training labels); a row whose source has no label is masked for that head (each head term is the mean over its
present cells, so with every source present the dread term equals EXP113's loss). Validation and serving output: the equal per-target mean of
the heads' percentile ranks (average rank / n within the evaluated study set) -> oof_best.csv (K24) / oof_k36.csv (K36); every head's own
probabilities are saved as oof_head_<name>.csv / oof_head_<name>_k36.csv. OOF is scored with Dread (np.rint) as EXP099; the per-head tables
let H-B-42's rulers score each head on other references (cross-set columns).
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
from tools import exp113_radr50_local as e113  # noqa: E402
from tools import exp107_kfold_radr50 as e107  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
from tools.exp113_radr50_local import (worker_init, read_rows, install_gold_blocker, uid_hash, relpath, sha256, read,  # noqa: E402,F401
                                       CODE_NAMES, ENV_KEYS, THREAD_ENV, canonical_order, validate_fold_membership, pos_weight, contract, UID, T,
                                       DREAD, KEYS_FILE, FOLDS, EXTRA_EVAL, fold_contract, state_hashes, mismatched_keys, dump,
                                       train_python, within, display_path, _set_flag)

VERSION = '1.0.0'
SCHEMA = 'exp117-multihead-v1'
BASE_PROTOCOL = e113.PROTOCOL
BASE_MANIFEST_SHA256 = 'a134327a10bfe051a0d58f3bb83969b2bdb8b9fcc4f54a309893504c960bd31d'
LOOP_SOURCE = e113.SHARED_RUNNER  # the EXP107 runner: this runner is a copy of its loop with the EXP117 changes marked
PROTOCOL = 'artifacts/experiments/EXP117_MULTIHEAD_PROTOCOL_V1'
QUEUE_FILE = PROTOCOL + '/queue-entries.json'
RUNNER = 'training/runners/exp117_multihead.py'
TOOL = 'tools/exp117_multihead.py'
SEED = e113.SEED
TAG_PREFIX = 'EXP117_MULTIHEAD'
OOF_DIR = 'artifacts/experiments/EXP117_MULTIHEAD_OOF'
LOCAL_WORKERS = e113.LOCAL_WORKERS
DEFAULT_AFTER = 'EXP113_RADR50_LOCAL_F4'
QUEUE_FILE_REST = PROTOCOL + '/queue-entries-f1-f4.json'
SCREEN_FOLD = 0
# F1-F4 are queued only with a recorded F0 screen pass (H-B-47: the head mean is not below EXP113 F0 by more than 0.002 on any reference).
SCREEN_PASS_RULE = ('a JSON object with "verdict": "PASS" (exact), "tag": "EXP117_MULTIHEAD_F0" and "protocol_sha256" equal to the sha256 of '
                    'the current EXP117 manifest, recorded after the H-B-47 fold-0 matched screen against EXP113 F0')
LABEL_SET = 'multihead4'
HEADS = (('dread', DREAD),
         ('relabel_v1', 'storage/labels/derived_relabel/relabel_v1.csv'),
         ('steven_v4', 'storage/labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'),
         ('pilkwang_v2', 'storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv'))
HEAD_NAMES = tuple(n for n, _ in HEADS)
RUN_FILES = e113.RUN_FILES + tuple(f for n in HEAD_NAMES for f in (f'oof_head_{n}.csv', f'oof_head_{n}_k36.csv'))
SERVING_RULE = ('per target, the equal mean over the four heads of avg_rank(head probability) / n within the evaluated study set (a fold\'s '
                'validation set for OOF; the test set at serving); every head\'s probabilities are kept')
LOSS_RULE = ('sum over heads h of mean over present cells of BCEWithLogits(logit_h, y_h, pos_weight_h) (pos_weight_h = EXP099 rule on head h\'s '
             'own fold training labels; absent source rows masked for that head); same AdamW / OneCycle / clip 3.0 / FP16 gate as EXP113')
CHANGED_FIELDS = ('schema', 'created_at', 'plan_item', 'source_bindings', 'folds', 'tags', 'local', 'evaluation', 'caveats', 'derived_from',
                  'label_sets', 'version', 'heads', 'multihead', 'prepare_preflight')
FOLD_CHANGED_FIELDS = ('tag', 'head_pos_weight')
CAVEATS = [
    'label intervention (label sources as heads): Dread OOF of the head mean cannot decide it alone; judged by H-B-42\'s validated ruler or ONE '
    'public submission, as pre-registered in H-B-47',
    'the loss is a sum of four BCE terms: the trunk gradient is about four times a single-source gradient; AdamW is largely invariant to a '
    'uniform scale, but clip_grad_norm 3.0 binds more often than in EXP113 (pre-clip norms are recorded in history.json)',
    'relabel_v1, Steven v4 and Pilkwang v2 share report-text errors (D-B-040); the heads add definitions, not independent image information',
    'Steven v4 and Pilkwang v2 files contain Gold58 rows; only G1 fold ids are ever read into training (registration checks no fold id is Gold)',
    'OOF of the head mean is a per-fold rank mean: pooled AUC mixes five per-fold rank scales (D-A-049 found within-fold vs pooled ranks similar)',
]


def tag_for(fold, mode='full'):
    return f'{TAG_PREFIX}_F{int(fold)}' + ('_SMOKE' if mode == 'smoke' else '')


def _manifest_path(root=ROOT):
    return Path(root) / PROTOCOL / 'manifest.json'


def _protocol(root=ROOT):
    return read(_manifest_path(root))


# ------------------------------------------------------------------ pure helpers (unit-tested; torch imported lazily)
def head_pos_weight(labels, train_ids):
    """EXP099 rule on the training ids this source labels (all of them -> exactly e107.pos_weight(labels, train_ids))."""
    present = [u for u in train_ids if u in labels]
    if not present:
        raise ValueError('source labels none of the training ids')
    return pos_weight(labels, present)


def stack_sources(head_labels, names, ids):
    """{uid: float32[12 * len(names)]}, head-major; NaN marks a source that has no row for the uid."""
    import numpy as np
    out = {}
    for u in ids:
        parts = [np.asarray(head_labels[n][u], np.float32) if u in head_labels[n] else np.full(len(T), np.nan, np.float32) for n in names]
        out[u] = np.concatenate(parts).astype(np.float32)
    return out


def rank_mean(P):
    """P (n, H, 12) head probabilities -> (n, 12) equal mean of per-head average-rank / n, per target."""
    import numpy as np
    P = np.asarray(P, np.float64)
    n, H, C = P.shape
    return np.stack([np.mean([k99.avg_rank(P[:, h, c]) / n for h in range(H)], axis=0) for c in range(C)], 1)


def masked_multihead_bce(logits, y, pos_weight):
    """logits (B, H, C); y (B, H * C) head-major with NaN = absent; pos_weight (H, C). Returns (sum of head losses, per-head losses);
    each head loss is the mean over its present cells (= BCEWithLogitsLoss(pos_weight) mean when nothing is absent)."""
    import torch
    import torch.nn.functional as F
    B, H, C = logits.shape
    y = y.reshape(B, H, C).to(logits.dtype)
    present = torch.isfinite(y)
    target = torch.where(present, y, torch.zeros_like(y))
    per = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight.to(logits.dtype), reduction='none')
    m = present.to(per.dtype)
    head_losses = (per * m).sum((0, 2)) / m.sum((0, 2)).clamp_min(1.0)
    return head_losses.sum(), head_losses


def build_multihead(base, n_heads):
    """MultiHeadRaptor around a built RaptorClassifier: base.backbone shared, n_heads copies of its head (norm, att, clsW, clsb), each
    computing exactly RaptorClassifier.head. forward(x) -> logits (B, n_heads, 12)."""
    import torch
    import torch.nn as nn

    class SourceHead(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.norm = copy.deepcopy(src.norm)
            self.att = copy.deepcopy(src.att)
            self.clsW = nn.Parameter(src.clsW.detach().clone())
            self.clsb = nn.Parameter(src.clsb.detach().clone())

        def forward(self, feats):  # = raptor_common.RaptorClassifier.head
            h = self.norm(feats)
            a = torch.softmax(self.att(h), dim=1)
            pooled = torch.einsum('bkn,bkf->bnf', a, h)
            return (pooled * self.clsW).sum(-1) + self.clsb

    class MultiHeadRaptor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base.backbone
            self.heads = nn.ModuleList([SourceHead(base) for _ in range(n_heads)])
            self.n = base.n

        def encode(self, x):  # = RaptorClassifier.encode
            B, K = x.shape[:2]
            return self.backbone(x.flatten(0, 1)).view(B, K, -1)

        def forward(self, x):
            f = self.encode(x)
            return torch.stack([h(f) for h in self.heads], 1)

    return MultiHeadRaptor()


def build_initial_models(arch, seed, weights=None):
    """(base RaptorClassifier as EXP113 builds it, its load report, the multi-head model). CPU."""
    model, report = e107.build_initial_model(arch, seed, weights or e113.weights_path())
    return model, report, build_multihead(model, len(HEAD_NAMES))


# ------------------------------------------------------------------ prepare
def prepare(root=ROOT, dest=None, model_factory=None):
    import numpy as np
    import pandas as pd
    root = Path(root)
    destination = Path(dest) if dest else root / PROTOCOL
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + str(destination))
    base = root / BASE_PROTOCOL / 'manifest.json'
    if sha256(base) != BASE_MANIFEST_SHA256:
        raise ValueError('EXP113 manifest changed')
    bp = read(base)
    if bp['schema'] != e113.SCHEMA or bp['seed'] != SEED or bp['label_set']['name'] != 'dread' or bp['tags']['prefix'] != e113.TAG_PREFIX:
        raise ValueError('unexpected EXP113 registration')
    for path, expected in bp['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('EXP113 binding changed: ' + path)
    for name, expected in bp['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    gold = k99.gold_uids(root)
    trains = {k: read(root / bp['folds'][str(k)]['ordered_ids']['path'])['train'] for k in FOLDS}
    vals = {k: read(root / bp['folds'][str(k)]['ordered_ids']['path'])['val'] for k in FOLDS}
    g1 = set().union(*map(set, vals.values()))
    if g1 & gold:
        raise ValueError('a fold study is Gold')
    heads, head_labels = [], {}
    for name, path in HEADS:
        labels, semantic = k99._load_labels(root / path)
        raw = pd.read_csv(root / path, dtype={UID: str}, usecols=[UID])[UID].astype(str).tolist()
        head_labels[name] = labels
        heads.append(dict(name=name, path=path, sha256=sha256(root / path), semantic_sha256=semantic, rows=len(raw), loaded_rows=len(labels),
                          gold_rows_in_file=len(set(raw) & gold), g1_covered=len(g1 & set(labels)), g1_missing=len(g1 - set(labels))))
    if heads[0]['sha256'] != bp['label_set']['sha256'] or heads[0]['semantic_sha256'] != bp['label_set']['semantic_sha256']:
        raise ValueError('the dread head is not the EXP113 label file')
    relabel_reg = root / e113.LABEL_SETS_DIR / 'relabel_v1.json'
    folds = {}
    for k in FOLDS:
        f = copy.deepcopy(bp['folds'][str(k)])
        hp = {name: head_pos_weight(head_labels[name], trains[k]) for name in HEAD_NAMES}
        if not np.array_equal(hp['dread'], np.asarray(f['pos_weight'], np.float32)):
            raise ValueError(f'fold {k}: dread head pos_weight differs from EXP113')
        if relabel_reg.is_file() and not np.array_equal(hp['relabel_v1'], np.asarray(read(relabel_reg)['pos_weight'][str(k)], np.float32)):
            raise ValueError(f'fold {k}: relabel_v1 head pos_weight differs from the EXP113 relabel_v1 registration')
        f.update(tag=tag_for(k), head_pos_weight={n: v.tolist() for n, v in hp.items()})
        folds[str(k)] = f
    for k, v in {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        os.environ.setdefault(k, v)
    base_model, load_report, multi = (model_factory or (lambda arch, seed: build_initial_models(arch, seed)))(bp['common_args']['arch'], SEED)
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit('prepare touched CUDA')
    init, _ = state_hashes(base_model)
    if init != bp['initial_state_sha256']:
        raise ValueError('seed-42 trunk + head state re-computed here differs from EXP113')
    multi_init, multi_keys = state_hashes(multi)
    extra = [BASE_PROTOCOL + '/manifest.json', RUNNER, TOOL, *[p for _, p in HEADS]]
    if relabel_reg.is_file():
        extra.append(display_path(relabel_reg, root))
    bindings = {**bp['source_bindings'], **{s: sha256(root / s) for s in extra}}
    manifest = copy.deepcopy(bp)
    manifest.update(
        schema=SCHEMA, version=VERSION, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION', plan_item='H-B-47',
        source_bindings=bindings, folds=folds, label_sets=None,
        heads=heads,
        multihead=dict(names=list(HEAD_NAMES), outputs=12, loss=LOSS_RULE, serving=SERVING_RULE,
                       init='EXP113 seed-42 state (initial_state_sha256, enforced) then the head copied per source; head 0 = dread = EXP113\'s head',
                       multihead_initial_state_sha256=multi_init, multihead_keys=len(multi_keys),
                       head_keys=sorted(k for k in multi_keys if not k.startswith('backbone.')),
                       outputs_files=dict(served_k24='oof_best.csv', served_k36='oof_k36.csv',
                                          heads={n: [f'oof_head_{n}.csv', f'oof_head_{n}_k36.csv'] for n in HEAD_NAMES}),
                       missing_rows='rows absent from a source are masked for that head (G1: ' +
                                    ', '.join(f"{h['name']} {h['g1_missing']}" for h in heads) + ' missing)'),
        tags=dict(prefix=TAG_PREFIX, pattern=TAG_PREFIX + '_F<K>[_SMOKE]', oof_dir=OOF_DIR, exp113_prefix=bp['tags']['prefix']),
        local={**bp['local'], 'runner': RUNNER, 'guard': TOOL, 'shared_runner': None, 'loop_source': LOOP_SOURCE,
               'runner_rule': 'training/runners/exp117_multihead.py is a copy of the EXP107 runner with every change marked "EXP117:" '
                              '(model wrapper, four label sources, per-head pos_weight, masked summed loss, head outputs, rank-mean OOF); it '
                              'supplies OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 when absent before torch is imported',
               'queue': 'tools/exp117_multihead.py queue-json -> entries for tools/gpu_queue.py, continue_on_failure false: by default F0 '
                        'only (H-B-47 matched screen against EXP113 F0, after = the current chain tail); F1-F4 only with --folds 1-4 '
                        '--screen-pass <recorded F0 screen result: ' + SCREEN_PASS_RULE + '> (astra review 2026-10-08)'},
        evaluation={**bp['evaluation'], 'oof_dir': OOF_DIR,
                    'label_sets': 'head rank mean and every head scored with Dread (np.rint) as EXP099; per-head OOF for cross-set rulers'},
        caveats=list(bp['caveats']) + CAVEATS,
        derived_from=dict(path=BASE_PROTOCOL + '/manifest.json', sha256=BASE_MANIFEST_SHA256, tag_prefix=bp['tags']['prefix'],
                          changed_fields=list(CHANGED_FIELDS), fold_changed_fields=list(FOLD_CHANGED_FIELDS), changed_args={},
                          unchanged='seed, folds, recipe (common_args), trunk + dread head initial state, environment, weights, K, windows, '
                                    'contract, the dread pos_weight vectors and --labels / --eval-labels (Dread)'),
        prepare_preflight=dict(initial_state_recomputed_on='cpu', initial_state_equal_to_exp113=True, load_report=load_report))
    destination.mkdir(parents=True)
    dump_lf(destination / 'manifest.json', manifest)
    print(json.dumps(dict(status='REGISTERED_NO_EXECUTION', version=VERSION, protocol=display_path(destination, root),
                          manifest_sha256=sha256(destination / 'manifest.json'), tags=[f['tag'] for f in folds.values()],
                          multihead_initial_state_sha256=multi_init, heads=[{k: h[k] for k in ('name', 'g1_covered', 'g1_missing', 'gold_rows_in_file')}
                                                                            for h in heads]), indent=1))
    return manifest


# ------------------------------------------------------------------ runner guard (imported by training/runners/exp117_multihead.py)
def validate_arguments(args, root):
    """Called by the runner before any training import. Returns (protocol, fold spec, ordered ids, output root)."""
    root = Path(root)
    manifest_path = Path(args.protocol)
    p = read(manifest_path)
    if p.get('schema') != SCHEMA or manifest_path.resolve() != _manifest_path(root).resolve():
        raise ValueError('wrong protocol')
    if p.get('derived_from', {}).get('sha256') != BASE_MANIFEST_SHA256 or p['tags']['prefix'] != TAG_PREFIX:
        raise ValueError('not the EXP117 registration')
    if [h['name'] for h in p['heads']] != list(HEAD_NAMES):
        raise ValueError('head list differs from the registration')
    if os.environ.get('KNEE_REMOTE_RELAX_ENV'):
        raise ValueError('local protocol: KNEE_REMOTE_RELAX_ENV must not be set (environment and initial state are enforced)')
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
    for h in p['heads']:
        if sha256(relpath(root, h['path'])) != h['sha256']:
            raise ValueError('head label file changed: ' + h['name'])
    weights = Path(args.init_weights)
    if weights.resolve() != e113.weights_path(root).resolve():
        raise ValueError('init weights must be the local RadImageNet file ' + e113.WEIGHTS_LOCAL)
    if not weights.is_file() or sha256(weights) != p['init_weights']['sha256'] or p['init_weights']['sha256'] != e113.RADIMAGENET_SHA256:
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
    elif out_root != default_root:
        raise ValueError('local full runs write to artifacts/experiments only (no --output-root)')
    return p, fold, ordered, out_root


def runner_argv(p, fold, mode='full', root=ROOT, python=None, workers=None, output_root=None, smoke=None):
    root = Path(root)
    argv = [python or train_python(), str(root / RUNNER), '--protocol', str(root / PROTOCOL / 'manifest.json'),
            '--fold', str(int(fold)), '--mode', mode, '--tag', tag_for(fold, mode), '--seed', str(p['seed']),
            '--workers', str(LOCAL_WORKERS if workers is None else int(workers))]
    for k, v in p['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--corpus', str(relpath(root, p['paths']['corpus'])), '--labels', str(relpath(root, p['paths']['labels'])),
             '--eval-labels', str(relpath(root, p['paths']['eval_labels'])), '--init-weights', str(e113.weights_path(root))]
    if output_root:
        argv += ['--output-root', str(output_root)]
    if mode == 'smoke':
        smoke = {**dict(device='cpu', amp='none', eval_amp='none', train=16, val=2, epochs=1), **(smoke or {})}
        for key in ('device', 'amp', 'eval_amp'):
            argv = _set_flag(argv, '--' + key.replace('_', '-'), smoke[key])
        argv += ['--smoke-train', str(smoke['train']), '--smoke-val', str(smoke['val']), '--smoke-epochs', str(smoke['epochs'])]
    return argv


def parse_folds(text):
    """'0' / '1-4' / '1,2,3,4' -> sorted registered folds, each once."""
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition('-')
        try:
            out += list(range(int(lo), int(hi) + 1)) if sep else [int(lo)]
        except ValueError:
            raise SystemExit(f'bad --folds value {text!r} (use 0, 1-4 or 1,2,3,4)')
    if not out or any(k not in FOLDS for k in out) or len(set(out)) != len(out):
        raise SystemExit(f'bad --folds value {text!r}: registered folds are {list(FOLDS)}, each once')
    return sorted(out)


def check_screen_pass(path, root=ROOT):
    """The recorded F0 screen result that admits F1-F4 (fail closed): it must exist and say PASS for this protocol's F0."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        raise SystemExit(f'--screen-pass {path!r} not found: F1-F4 are queued only after a recorded F0 screen PASS ({SCREEN_PASS_RULE})')
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise SystemExit(f'--screen-pass {p} is not readable JSON ({exc})')
    if not isinstance(d, dict) or d.get('verdict') != 'PASS':
        raise SystemExit(f'--screen-pass {p} does not say "verdict": "PASS" (got {d.get("verdict") if isinstance(d, dict) else d!r})')
    if d.get('tag') != tag_for(SCREEN_FOLD):
        raise SystemExit(f'--screen-pass {p} is not the {tag_for(SCREEN_FOLD)} screen (tag {d.get("tag")!r})')
    manifest_sha = sha256(_manifest_path(root))
    if d.get('protocol_sha256') != manifest_sha:
        raise SystemExit(f'--screen-pass {p} protocol_sha256 {str(d.get("protocol_sha256"))[:12]} is not the current manifest {manifest_sha[:12]}')
    return dict(path=str(p), sha256=sha256(p), protocol_sha256=manifest_sha)


def queue_entries(folds=(SCREEN_FOLD,), after=None, workers=LOCAL_WORKERS, root=ROOT, screen_pass=None):
    """tools/gpu_queue.py entries with continue_on_failure false (a failed fold stops the chain). Default: F0 only (the H-B-47 matched
    screen against EXP113 F0). F1-F4 only with screen_pass = a recorded F0 screen PASS (check_screen_pass); F0 is never mixed with them."""
    folds = [int(k) for k in folds]
    if not folds or len(set(folds)) != len(folds):
        raise SystemExit('give each fold once')
    if folds == [SCREEN_FOLD]:
        if screen_pass is not None:
            raise SystemExit('--screen-pass admits F1-F4; F0 (the screen itself) is queued without it')
    elif SCREEN_FOLD in folds:
        raise SystemExit(f'F{SCREEN_FOLD} is queued alone: it is the matched screen, and F1-F4 follow only after its recorded PASS')
    else:
        check_screen_pass(screen_pass, root)
    p = _protocol(root)
    entries, previous = [], after
    for k in folds:
        if str(k) not in p['folds']:
            raise SystemExit(f'unregistered fold {k}')
        e = dict(tag=tag_for(k), command=runner_argv(p, k, 'full', root=root, workers=workers))
        if previous:
            e['after'] = previous
        e['continue_on_failure'] = False
        entries.append(e); previous = e['tag']
    return entries


def dump_lf(path, value):
    """k99.dump (exclusive create, indent 2, trailing newline) with LF newlines."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def validate_cli(fold):
    argv = runner_argv(_protocol(), fold, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=e113.local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-6000:]); print(r.stderr[-3000:], file=sys.stderr)
    return r.returncode


def smoke_cli(fold, out, train=16, val=2, epochs=1, workers=2):
    out = Path(out).resolve()
    if within(out, ROOT / 'artifacts'):
        raise SystemExit('smoke output must be outside artifacts/')
    argv = runner_argv(_protocol(), fold, 'smoke', workers=workers, output_root=out, smoke=dict(train=train, val=val, epochs=epochs))
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, env=e113.local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-4000:]); print(r.stderr[-3000:], file=sys.stderr)
    run = out / tag_for(fold, 'smoke')
    report = dict(version=VERSION, returncode=r.returncode, seconds=round(time.time() - t0), run=str(run),
                  files=sorted(x.name for x in run.iterdir()) if run.exists() else [])
    if (run / 'summary.json').exists():
        s = read(run / 'summary.json'); prov = s['provenance']
        report.update(optimizer_updates=s['optimizer_updates'], skipped=s['skipped_updates'], train=s['train'], val=s['val'], seed=prov['seed'],
                      initial_state_matches_registered=prov['initial_state_matches_registered'],
                      multihead_initial_state_matches_registered=prov['multihead_initial_state_matches_registered'],
                      cuda_initialized=prov.get('cuda_initialized'), history=s['history'])
    print(json.dumps(report, indent=1))
    return r.returncode


# ------------------------------------------------------------------ verification and assembly
def verify_run(run, p, k, manifest_sha):
    import numpy as np
    run = Path(run); fold = p['folds'][str(k)]; tag = tag_for(k)
    missing = [n for n in RUN_FILES if not (run / n).is_file()]
    if missing:
        return [f'fold {k}: missing ' + ', '.join(missing)]
    problems = []
    inv = read(run / 'output-inventory.json')
    if set(inv) != set(RUN_FILES) - {'output-inventory.json'}:
        problems.append(f'fold {k}: output-inventory.json does not list every output')
    for name, rec in inv.items():
        if not (run / name).is_file() or sha256(run / name) != rec['sha256']:
            problems.append(f'fold {k}: {name} differs from output-inventory.json')
    s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    expect = dict(fold=k, label_set=LABEL_SET, label_semantic_sha256=p['label_set']['semantic_sha256'], protocol_sha256=manifest_sha,
                  eval_ids_ordered_sha256=fold['ordered_ids']['val_sha256'], train_ids_ordered_sha256=fold['ordered_ids']['train_sha256'],
                  training_mode='full', checkpoint_selection='fixed-final', seed=SEED, arch=p['common_args']['arch'], input_policy='center3',
                  norm='radimagenet', remote_relaxed=False, initial_state_sha256=p['initial_state_sha256'], initial_state_matches_registered=True,
                  multihead_initial_state_sha256=p['multihead']['multihead_initial_state_sha256'], multihead_initial_state_matches_registered=True,
                  environment_matches_baseline=True, runner_sha256=p['source_bindings'][RUNNER], guard_sha256=p['source_bindings'][TOOL],
                  heads=list(HEAD_NAMES))
    for key, value in expect.items():
        if prov.get(key) != value:
            problems.append(f'fold {k}: provenance {key} = {prov.get(key)!r}, expected {value!r}')
    for name in HEAD_NAMES:
        if not np.array_equal(np.asarray((prov.get('head_pos_weight') or {}).get(name) or [], np.float32),
                              np.asarray(fold['head_pos_weight'][name], np.float32)):
            problems.append(f'fold {k}: {name} head pos_weight differs from the registration')
    if s.get('optimizer_updates') != fold['steps'] or s.get('skipped_updates', 10 ** 9) > fold['max_fp16_skips']:
        problems.append(f"fold {k}: updates {s.get('optimizer_updates')} (expected {fold['steps']}), skipped {s.get('skipped_updates')}")
    if s.get('selected_epoch_1based') != p['common_args']['epochs'] or s.get('train') != fold['n_train'] or s.get('val') != fold['n_val']:
        problems.append(f'fold {k}: epoch/count mismatch')
    if run.name != tag or (s.get('args') or {}).get('tag') != tag:
        problems.append(f'fold {k}: run directory / tag is not {tag}')
    return problems


def assemble(runs_dir=None, root=ROOT):
    import pandas as pd
    root = Path(root)
    manifest_path = _manifest_path(root)
    p = read(manifest_path); manifest_sha = sha256(manifest_path)
    runs = Path(runs_dir).resolve() if runs_dir else root / 'artifacts' / 'experiments'
    out = root / OOF_DIR
    if out.exists():
        raise SystemExit('output exists: ' + str(out))
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in p['folds'].items()}
    tables = {('served', 'k24'): {}, ('served', 'k36'): {}, **{(n, kk): {} for n in HEAD_NAMES for kk in ('k24', 'k36')}}
    files = {('served', 'k24'): 'oof_best.csv', ('served', 'k36'): 'oof_k36.csv',
             **{(n, 'k24'): f'oof_head_{n}.csv' for n in HEAD_NAMES}, **{(n, 'k36'): f'oof_head_{n}_k36.csv' for n in HEAD_NAMES}}
    sources, problems = {}, []
    for k in sorted(val_ids):
        run = runs / tag_for(k)
        bad = verify_run(run, p, k, manifest_sha)
        if bad:
            problems += bad; continue
        for key, name in files.items():
            tables[key][k] = pd.read_csv(run / name, dtype={UID: str})
        s = read(run / 'summary.json')
        sources[k] = dict(run=display_path(run, root), outputs={n: sha256(run / n) for n in files.values()},
                          checkpoint_sha256=read(run / 'output-inventory.json')['model_best.pt']['sha256'], total_seconds=s.get('total_seconds'),
                          fp16_skipped_updates=s.get('skipped_updates'))
    if problems:
        print(json.dumps(dict(status='INCOMPLETE', problems=problems), indent=1))
        raise SystemExit(2)
    metrics, outputs = {}, {}
    out.mkdir(parents=True)
    for who in ('served',) + HEAD_NAMES:
        result = k99.assemble_and_score(tables[(who, 'k24')], tables[(who, 'k36')], val_ids, root / DREAD)
        stem = 'oof' if who == 'served' else f'oof_head_{who}'
        for kk in ('k24', 'k36'):
            name = f'{stem}_{kk}.csv'
            result[f'oof_{kk}'].to_csv(out / name, index=False)
            outputs[name] = sha256(out / name)
        metrics[who] = result['metrics']
    body = dict(schema='exp117-oof-v1', version=VERSION, label_set=LABEL_SET, heads=list(HEAD_NAMES), observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                protocol=PROTOCOL, protocol_sha256=manifest_sha, seed=SEED, exp113_protocol_sha256=BASE_MANIFEST_SHA256, runs_dir=display_path(runs, root),
                rows=sum(len(v) for v in val_ids.values()), sources={str(k): v for k, v in sources.items()}, outputs=outputs,
                served=SERVING_RULE, scoring=dict(labels=DREAD, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py'),
                metrics=metrics,
                caveats=['weak Dread labels; a label-source intervention is not decided by Dread OOF alone (H-B-47 judging rule)',
                         'served OOF = per-fold rank mean; pooled AUC mixes five per-fold rank scales', 'fold 0 val = exposed G1 V888'])
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = ['# EXP117 multi-head RadR50 5-fold OOF (scored with Dread rint)', '', '| output | K24 pooled | K36 pooled |', '|---|---:|---:|']
    for who, m in metrics.items():
        lines.append(f"| {who} | {m['k24']['pooled']['macro12']:.6f} | {m['k36']['pooled']['macro12']:.6f} |")
    (out / 'result.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({who: {kk: round(v[kk]['pooled']['macro12'], 6) for kk in v} for who, v in metrics.items()}, indent=1))
    return body


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--dry-run', default=None, help='write the registration to this new directory instead (outside artifacts/)')
    s = sub.add_parser('argv'); s.add_argument('--fold', type=int, choices=FOLDS, required=True)
    s.add_argument('--mode', choices=['full', 'smoke'], default='full'); s.add_argument('--workers', type=int, default=None); s.add_argument('--output-root', default=None)
    s = sub.add_parser('validate'); s.add_argument('--fold', type=int, choices=FOLDS, required=True)
    s = sub.add_parser('smoke'); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--out', required=True)
    s.add_argument('--train', type=int, default=16); s.add_argument('--val', type=int, default=2)
    s.add_argument('--epochs', type=int, default=1); s.add_argument('--workers', type=int, default=2)
    s = sub.add_parser('queue-json')
    s.add_argument('--folds', default=str(SCREEN_FOLD), help='default 0 (the F0 screen only); "1-4" (or 1,2,3,4) needs --screen-pass')
    s.add_argument('--screen-pass', default=None, help='recorded F0 screen result that admits F1-F4: ' + SCREEN_PASS_RULE)
    s.add_argument('--after', required=True, help=f'tag the first entry waits for (the current GPU chain tail, e.g. {DEFAULT_AFTER}; '
                                                  f'{tag_for(SCREEN_FOLD)} for F1-F4); "none" only for an idle GPU')
    s.add_argument('--workers', type=int, default=LOCAL_WORKERS)
    s.add_argument('--out', default=None, help=f'new JSON file for the entries (default {QUEUE_FILE} for F0, {QUEUE_FILE_REST} for F1-F4; '
                                               '"-" = print only)')
    s = sub.add_parser('assemble'); s.add_argument('--runs-dir', default=None)
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')
    print(f'exp117_multihead v{VERSION}', file=sys.stderr)
    if a.command == 'prepare':
        if a.dry_run and within(a.dry_run, ROOT / 'artifacts'):
            raise SystemExit('--dry-run must be outside artifacts/')
        prepare(dest=a.dry_run)
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), a.fold, a.mode, workers=a.workers, output_root=a.output_root), indent=1))
    elif a.command == 'validate':
        sys.exit(validate_cli(a.fold))
    elif a.command == 'smoke':
        sys.exit(smoke_cli(a.fold, a.out, a.train, a.val, a.epochs, a.workers))
    elif a.command == 'queue-json':
        folds = parse_folds(a.folds)
        entries = queue_entries(folds, None if a.after.lower() == 'none' else a.after, a.workers, screen_pass=a.screen_pass)
        out = a.out or str(ROOT / (QUEUE_FILE if folds == [SCREEN_FOLD] else QUEUE_FILE_REST))
        if out != '-':
            dump_lf(out, entries)
            print('written ' + out, file=sys.stderr)
        print(json.dumps(entries, indent=1))
    else:
        assemble(a.runs_dir)


if __name__ == '__main__':
    main()
