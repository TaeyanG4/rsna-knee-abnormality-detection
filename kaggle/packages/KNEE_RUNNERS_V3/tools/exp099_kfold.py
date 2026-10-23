"""EXP099: 5-fold version of the B8 seed42 recipe (EXP094 V4 COMMON) for OOF teacher labels and a fold ensemble.

  python tools/exp099_kfold.py prepare
  python tools/exp099_kfold.py register-labels --name dread --path storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv
  python tools/exp099_kfold.py argv --fold 1 --labels dread --mode full
  python tools/exp099_kfold.py queue-json --labels dread --folds 1,2,3,4 --after TAG|none [--out FILE]
  python tools/exp099_kfold.py launch --fold 1 --labels dread --mode pilot|full
  python tools/exp099_kfold.py status --fold 1 --labels dread --mode full
  python tools/exp099_kfold.py reuse-fold0
  python tools/exp099_kfold.py assemble --labels dread [--k36 0=path/to/oof_k36.csv]

Folds over the 4,343 G1 studies (EXP084 ordered-ids train+val). Fold 0 is exactly the G1 val set (888). Folds 1-4 partition the
3,455 G1 train studies by whole component_id, greedy-balanced on study count and per-target positive counts (Dread rounded with
np.rint), deterministic (seed 20261007). Every per-fold contract (train-loader length, optimizer updates, scheduler steps, FP16 skip
gate, pos_weight) is computed here and frozen in the manifest; the runner recomputes and must match.
Label sets are registered separately (label-sets/NAME.json, bound to the manifest sha256) so the same folds can be rerun with other
label files without code changes. Tags: EXP099_B8_<NAME>_F<K>[_PILOT]. Nothing here trains or touches the GPU.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.exp084_group_loss import (worker_init, read_rows, install_gold_blocker,  # noqa: E402,F401
                                     uid_hash, relpath, sha256, read, CODE_NAMES, ENV_KEYS, THREAD_ENV)

SCHEMA = 'exp099-kfold-b8-v1'
LABEL_SCHEMA = 'exp099-label-set-v1'
PROTOCOL = 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1'
BASE = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
V4 = 'artifacts/experiments/EXP094_B8_VARIANT_PROTOCOL_V4/manifest.json'
RUNNER = 'training/runners/exp099_kfold.py'
DREAD = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
GOLD_SOURCE = 'storage/competition/train.csv'
REUSE_RUN = 'artifacts/experiments/EXP094L_B8_SEED42'
GPU_PYTHON = Path(r'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe')
UID = 'StudyInstanceUID'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
FOLD_SEED = 20261007
N_SPLIT_FOLDS = 4  # folds 1..4 partition G1 train; fold 0 = G1 val
FOLDS = list(range(N_SPLIT_FOLDS + 1))
SEED = 42
COMMON = dict(arch='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', res=224, k=12, k_eval=24, epochs=8, fixed_epoch=8, bs=8, accum=1, grad_ckpt=True, workers=12,
              amp='fp16', amp_init_scale=1024.0, eval_amp='fp16', runtime_policy='strict', norm='imagenet', bb_lr=3e-05, head_lr=0.001, wd=0.02, pretrained='timm',
              window_policy='within-slot-valid', device='cuda')
EXTRA_EVAL = dict(k_eval=36, file='oof_k36.csv', source='D-B-012: same final model re-evaluated with 36 evenly spaced windows')
CODE = [RUNNER, 'tools/exp099_kfold.py', 'tools/exp087_training.py', 'tools/exp084_group_loss.py']
GROUP_COLS = [UID, 'component_id', 'report_group_sha256', 'split', 'scanner_group', 'field', 'manufacturer', 'cached_mask']
NAME_RE = re.compile(r'^[a-z][a-z0-9_]{0,31}$')


# ------------------------------------------------------------------ small helpers
def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def tag_for(name, fold, mode):
    return f'EXP099_B8_{name.upper()}_F{int(fold)}' + ('_PILOT' if mode == 'pilot' else '')


def oof_dir_for(name):
    return f'artifacts/experiments/EXP099_B8_{name.upper()}_OOF'


def train_python():
    return str(GPU_PYTHON) if GPU_PYTHON.exists() else sys.executable


def workspace_rel(root, path):
    """Workspace-relative POSIX path for a file that must live inside the workspace."""
    p = Path(path)
    p = (p if p.is_absolute() else Path(root) / p).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        raise ValueError('label file must be inside the workspace: ' + str(path)) from None


def avg_rank(x):
    import numpy as np
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); sx = x[o]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def auc(y, s):
    """Identical to tools/g1_compare.py (average-rank Mann-Whitney AUC)."""
    import numpy as np
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return np.nan
    r = avg_rank(s); return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def macro12(Y, S):
    import numpy as np
    per = [auc(Y[:, t], S[:, t]) for t in range(12)]
    return float(np.mean(per)), {T[t]: float(per[t]) for t in range(12)}


# ------------------------------------------------------------------ pure fold logic (unit-tested)
def canonical_order(base_ordered):
    """Canonical study order: the order of the original ordered-ids train list followed by its val list."""
    return list(base_ordered['train']) + list(base_ordered['val'])


def build_folds(base_ordered, rows, labels, seed=FOLD_SEED, n_folds=N_SPLIT_FOLDS):
    """Return {uid: fold}. Fold 0 = base val exactly; folds 1..n_folds = greedy whole-component partition of base train.

    Components are processed largest first, then by rare-positive weight, then by a seeded random tiebreak over sorted component
    ids (so the result does not depend on row order). Each goes to the fold whose squared relative deviation from the equal share
    (study count + mean over 12 targets of positive counts, labels rounded with np.rint) grows least; ties -> fewer studies -> lower fold.
    """
    import numpy as np
    comp = {r[UID]: r['component_id'] for r in rows}
    train, val = list(base_ordered['train']), list(base_ordered['val'])
    if set(train) & set(val) or len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError('base split not disjoint/unique')
    if set(comp) != set(train) | set(val):
        raise ValueError('group rows differ from base train+val')
    if {comp[u] for u in train} & {comp[u] for u in val}:
        raise ValueError('base split crosses a component')
    groups = {}
    for u in train:
        groups.setdefault(comp[u], []).append(u)
    comps = sorted(groups)
    y = {u: np.rint(np.asarray(labels[u], dtype=np.float32)).astype(np.int64) for u in train}
    size = {c: len(groups[c]) for c in comps}
    pos = {c: np.sum([y[u] for u in groups[c]], axis=0) for c in comps}
    total_pos = np.sum([pos[c] for c in comps], axis=0).astype(np.float64)
    rarity = {c: float((pos[c] / np.maximum(total_pos, 1.0)).sum()) for c in comps}
    tiebreak = dict(zip(comps, np.random.default_rng(seed).permutation(len(comps)).tolist()))
    order = sorted(comps, key=lambda c: (-size[c], -rarity[c], tiebreak[c]))
    exp_n = len(train) / n_folds
    exp_p = np.maximum(total_pos / n_folds, 1e-9)
    N = np.zeros(n_folds); P = np.zeros((n_folds, 12))
    assignment = {}
    for c in order:
        best = None
        for f in range(n_folds):
            dn = ((N[f] + size[c] - exp_n) ** 2 - (N[f] - exp_n) ** 2) / exp_n ** 2
            dp = float((((P[f] + pos[c] - exp_p) ** 2 - (P[f] - exp_p) ** 2) / exp_p ** 2).mean())
            key = (dn + dp, N[f], f)
            if best is None or key < best:
                best = key
        f = best[2]
        assignment[c] = f + 1; N[f] += size[c]; P[f] += pos[c]
    fold = {u: 0 for u in val}
    fold.update({u: assignment[comp[u]] for u in train})
    return fold


def fold_split(fold_of, canonical, k):
    """Ordered train/val ids of fold k, both in canonical order."""
    return dict(train=[u for u in canonical if fold_of[u] != k], val=[u for u in canonical if fold_of[u] == k])


def pos_weight(labels, train_ids):
    """Exactly the runner's rule: prev = clip(mean(float32 train labels), .03, .7); clip((1-prev)/prev, 1, 10) as float32."""
    import numpy as np
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    return np.clip((1 - prev) / prev, 1, 10).astype(np.float32)


def contract(n_train, args=COMMON):
    len_tl = n_train // args['bs']  # DataLoader(drop_last=True)
    steps = max(math.ceil(len_tl / args['accum']) * args['epochs'], 1)
    return dict(len_train_loader=len_tl, steps=steps, scheduler_total_steps=steps, max_fp16_skips=steps // 100, eval_batch=max(1, args['bs'] // 2))


def validate_fold_membership(corpus_ids, ordered, rows, canonical):
    """Fold-file membership: exact cover/disjointness against the fold group file, canonical order, no component crossing."""
    train, val = list(ordered['train']), list(ordered['val'])
    split = {r[UID]: r['split'] for r in rows}
    if len(split) != len(rows) or set(split.values()) - {'train', 'val'}:
        raise ValueError('fold group file malformed')
    position = {u: i for i, u in enumerate(canonical)}
    if (len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val)
            or set(train) != {u for u, s in split.items() if s == 'train'} or set(val) != {u for u, s in split.items() if s == 'val'}
            or not set(split) <= set(corpus_ids) or len(corpus_ids) != len(set(corpus_ids)) or not set(split) <= set(position)):
        raise ValueError('Fold cohort coverage/disjointness mismatch')
    for ids in (train, val):
        idx = [position[u] for u in ids]
        if any(b <= a for a, b in zip(idx, idx[1:])):
            raise ValueError('Fold ids are not in canonical order')
    tcomp = {r['component_id'] for r in rows if r['split'] == 'train'}
    vcomp = {r['component_id'] for r in rows if r['split'] == 'val'}
    if tcomp & vcomp:
        raise ValueError('Component crosses train/validation')
    return train, val


def check_label_set(labels, raw_uids, gold_uids, fold_train_lists):
    """Registration gate: no Gold rows at all, every train uid of every fold labelled with 12 finite values in [0, 1]."""
    import numpy as np
    overlap = sorted(set(raw_uids) & set(gold_uids))
    if overlap:
        raise ValueError(f'label file contains {len(overlap)} Gold uid(s), e.g. {overlap[0]}')
    for k, ids in fold_train_lists.items():
        missing = [u for u in ids if u not in labels]
        if missing:
            raise ValueError(f'label file misses {len(missing)} fold-{k} train uid(s), e.g. {missing[0]}')
        arr = np.stack([labels[u] for u in ids])
        if arr.shape != (len(ids), 12) or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ValueError(f'invalid label values for fold {k}')
    return True


def assemble_frames(fold_frames, fold_val_ids):
    """Concatenate per-fold prediction frames (UID + 12 targets) after exact per-fold order/coverage checks; return sorted frame."""
    import numpy as np
    import pandas as pd
    parts = []
    for k in sorted(fold_val_ids):
        df = fold_frames[k]
        if list(df.columns) != [UID] + T:
            raise ValueError(f'fold {k}: unexpected columns')
        if df[UID].astype(str).tolist() != list(fold_val_ids[k]):
            raise ValueError(f'fold {k}: predicted uids differ from the fold validation ids')
        if not np.isfinite(df[T].to_numpy(dtype=np.float64)).all():
            raise ValueError(f'fold {k}: non-finite predictions')
        parts.append(df.assign(**{UID: df[UID].astype(str)}))
    out = pd.concat(parts, ignore_index=True)
    if out[UID].duplicated().any():
        raise ValueError('a uid appears in more than one fold')
    return out.sort_values(UID, kind='mergesort').reset_index(drop=True)


# ------------------------------------------------------------------ prepare
def _load_labels(path):
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_common import load_label_csv
    return load_label_csv(Path(path))


def prepare(root=ROOT):
    import numpy as np
    root = Path(root)
    destination = root / PROTOCOL
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + PROTOCOL)
    baseline, v4 = read(root / BASE), read(root / V4)
    if v4['common_args'] != COMMON:
        raise ValueError('COMMON differs from EXP094 V4 common_args')
    if v4['arms']['seed42']['seed'] != SEED or v4['arms']['seed42']['variant'] != 'seed':
        raise ValueError('EXP094 V4 seed42 arm changed')
    for name, expected in v4['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed: ' + name)
    base_ordered_path, base_groups_path = baseline['ordered_ids']['path'], baseline['groups']['path']
    if sha256(root / base_ordered_path) != baseline['ordered_ids']['sha256'] or sha256(root / base_groups_path) != baseline['groups']['sha256']:
        raise ValueError('base split files changed')
    if baseline['paths']['labels'] != DREAD or v4['paths']['labels'] != DREAD:
        raise ValueError('Dread path differs from the base protocols')
    base_ordered = read(root / base_ordered_path)
    rows = read_rows(root / base_groups_path)
    labels, label_hash = _load_labels(root / DREAD)
    if label_hash != baseline['label_hash'] or label_hash != v4['label_hash']:
        raise ValueError('Dread semantic hash changed')
    canonical = canonical_order(base_ordered)
    if len(canonical) != 4343 or len(set(canonical)) != 4343:
        raise ValueError('G1 population drift')
    fold_of = build_folds(base_ordered, rows, labels)
    comp = {r[UID]: r['component_id'] for r in rows}
    destination.mkdir(parents=True)
    with (destination / 'folds.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow([UID, 'component_id', 'fold'])
        for u in sorted(canonical):
            w.writerow([u, comp[u], fold_of[u]])
    folds, table = {}, []
    for k in FOLDS:
        split = fold_split(fold_of, canonical, k)
        sub = destination / f'fold{k}'; sub.mkdir()
        dump(sub / 'ordered-ids.json', split)
        val_set = set(split['val'])
        with (sub / 'group-metadata.csv').open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=GROUP_COLS, lineterminator='\n'); w.writeheader()
            for r in rows:
                w.writerow({**{c: r[c] for c in GROUP_COLS}, 'split': 'val' if r[UID] in val_set else 'train'})
        group_rows = read_rows(sub / 'group-metadata.csv')
        validate_fold_membership(canonical, split, group_rows, canonical)
        pw = pos_weight(labels, split['train'])
        yv = np.rint(np.stack([labels[u] for u in split['val']]).astype(np.float32)).astype(int)
        yt = np.rint(np.stack([labels[u] for u in split['train']]).astype(np.float32)).astype(int)
        rel = f'{PROTOCOL}/fold{k}'
        folds[str(k)] = dict(
            fold=k, ordered_ids=dict(path=rel + '/ordered-ids.json', sha256=sha256(sub / 'ordered-ids.json'),
                                     train_sha256=uid_hash(split['train']), val_sha256=uid_hash(split['val'])),
            groups=dict(path=rel + '/group-metadata.csv', sha256=sha256(sub / 'group-metadata.csv')),
            n_train=len(split['train']), n_val=len(split['val']),
            train_components=len({comp[u] for u in split['train']}), val_components=len({comp[u] for u in split['val']}),
            **contract(len(split['train'])), pos_weight=pw.tolist(),
            val_positive_counts={t: int(yv[:, j].sum()) for j, t in enumerate(T)},
            val_prevalence={t: round(float(yv[:, j].mean()), 6) for j, t in enumerate(T)},
            train_prevalence={t: round(float(yt[:, j].mean()), 6) for j, t in enumerate(T)},
            macro12_valid=bool(all(0 < yv[:, j].sum() < len(yv) for j in range(12))))
        table.append(folds[str(k)])
    f0 = folds['0']
    if (f0['ordered_ids']['train_sha256'] != baseline['ordered_ids']['train_sha256'] or f0['ordered_ids']['val_sha256'] != baseline['ordered_ids']['val_sha256']
            or f0['groups']['sha256'] != baseline['groups']['sha256'] or f0['len_train_loader'] != 431 or f0['steps'] != 3448
            or not np.array_equal(np.asarray(f0['pos_weight'], np.float32), np.asarray(baseline['loss']['baseline_pos_weight'], np.float32))):
        raise ValueError('fold 0 does not reproduce the G1 B8 contract')
    bindings = CODE + [BASE, V4, DREAD, base_ordered_path, base_groups_path]
    manifest = dict(
        schema=SCHEMA, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NO_EXECUTION',
        source_bindings={s: sha256(root / s) for s in bindings},
        training_code_sha256=v4['training_code_sha256'], environment=v4['environment'], corpus_manifest_sha256=v4['corpus_manifest_sha256'],
        excluded_uids=v4['excluded_uids'], initial_state_sha256=v4['initial_state_sha256'],
        paths=dict(corpus=v4['paths']['corpus'], eval_labels=DREAD, base_ordered_ids=base_ordered_path, base_groups=base_groups_path,
                   folds=PROTOCOL + '/folds.csv', label_sets_dir=PROTOCOL + '/label-sets'),
        folds_csv_sha256=sha256(destination / 'folds.csv'),
        common_args=COMMON, seed=SEED, variant='seed', base_recipe='EXP094 V4 arm seed42 (B8 CoAtNet 224, FP16 GradScaler 1024, fixed-final epoch 8)',
        label_sets={'dread': dict(path=DREAD, sha256=sha256(root / DREAD), semantic_sha256=label_hash)},
        fold_construction=dict(seed=FOLD_SEED, population='4343 = EXP084 ordered-ids train (3455) + val (888)',
                               fold0='exactly EXP084 G1 val (888); fold-0 train = G1 train, identical to EXP094L_B8_SEED42',
                               folds1_4='whole component_id partition of G1 train; greedy: components by size desc, rare-positive weight desc, seeded tiebreak; '
                                        'fold with least increase of squared relative deviation (count + mean over 12 targets of np.rint Dread positives)',
                               canonical_order='original ordered-ids train list then val list; fold train/val lists keep that order'),
        folds=folds,
        pos_weight_rule='prev = clip(mean(float32 train labels in fold canonical order), .03, .7); clip((1-prev)/prev, 1, 10) float32; per label set in label-sets/NAME.json',
        contract_rule='len_train_loader = n_train // bs (drop_last); steps = ceil(len_train_loader/accum) * epochs = scheduler total_steps; '
                      'FP16 skip gate: skipped > steps // 100 fails; eval batch bs // 2',
        extra_eval=EXTRA_EVAL, tags=dict(pattern='EXP099_B8_<LABELSET>_F<K>[_PILOT]', reuse_fold0_dread=REUSE_RUN),
        training_budgets=dict(pilot_seconds=600, full_seconds=21600, retry=0),
        evaluation=dict(labels=DREAD, rule='float32 then numpy.rint (tools/g1_compare.py)', metric='macro12 per fold and pooled over 4343 OOF',
                        oof_dir_pattern='artifacts/experiments/EXP099_B8_<LABELSET>_OOF'),
        caveats=['weak Dread labels; folds 1-4 include G1 val studies in training, so fold OOF is not comparable to the single G1 V888 numbers',
                 'component grouping does not prove patient independence', 'OOF is for teacher pseudo-labels; not D01/private evidence'])
    dump(destination / 'manifest.json', manifest)
    report = dict(status='REGISTERED_NO_EXECUTION', protocol=PROTOCOL, manifest_sha256=sha256(destination / 'manifest.json'),
                  folds={k: dict(n_train=v['n_train'], n_val=v['n_val'], train_components=v['train_components'], val_components=v['val_components'],
                                 len_train_loader=v['len_train_loader'], steps=v['steps'], max_fp16_skips=v['max_fp16_skips'],
                                 pos_weight=[round(x, 4) for x in v['pos_weight']]) for k, v in folds.items()})
    print(json.dumps(report, indent=1))
    print(prevalence_table(folds))
    return manifest


def prevalence_table(folds):
    head = '| target | ' + ' | '.join(f'F{k} pos (rate)' for k in folds) + ' |'
    lines = [head, '|---|' + '---:|' * len(folds)]
    for t in T:
        lines.append(f'| {t} | ' + ' | '.join(f"{v['val_positive_counts'][t]} ({v['val_prevalence'][t]:.3f})" for v in folds.values()) + ' |')
    lines.append('| n_val / components | ' + ' | '.join(f"{v['n_val']} / {v['val_components']}" for v in folds.values()) + ' |')
    return '\n'.join(lines)


# ------------------------------------------------------------------ label sets
def gold_uids(root=ROOT):
    """Gold58 = rows of storage/competition/train.csv with a non-null ACL (uids only; label values are not used)."""
    import pandas as pd
    df = pd.read_csv(Path(root) / GOLD_SOURCE, dtype={UID: str}, usecols=[UID, 'ACL'])
    return set(df.loc[df['ACL'].notna(), UID].astype(str))


def fold_train_lists(root, manifest):
    return {int(k): read(Path(root) / v['ordered_ids']['path'])['train'] for k, v in manifest['folds'].items()}


def register_labels(name, path, pos_weight_source='labels', root=ROOT):
    import numpy as np
    import pandas as pd
    root = Path(root)
    if not NAME_RE.match(name):
        raise SystemExit('label-set name must match ' + NAME_RE.pattern)
    manifest_path = root / PROTOCOL / 'manifest.json'
    m = read(manifest_path)
    if m['schema'] != SCHEMA:
        raise SystemExit('wrong protocol')
    out = root / m['paths']['label_sets_dir'] / f'{name}.json'
    if out.exists():
        raise SystemExit('label set already registered: ' + str(out))
    rel = workspace_rel(root, path)
    if name in m['label_sets'] and m['label_sets'][name]['path'] != rel:
        raise SystemExit(f'{name} is bound to {m["label_sets"][name]["path"]} in the manifest')
    if name == 'dread' and pos_weight_source != 'labels':
        raise SystemExit('dread uses its own pos_weight')
    file = root / rel
    labels, semantic = _load_labels(file)
    raw = pd.read_csv(file, dtype={UID: str}, usecols=[UID])[UID].astype(str).tolist()
    trains = fold_train_lists(root, m)
    check_label_set(labels, raw, gold_uids(root), trains)
    if pos_weight_source == 'labels':
        weights = {str(k): pos_weight(labels, ids).tolist() for k, ids in trains.items()}
    else:  # fixed Dread loss weighting with a different target file
        weights = {k: v['pos_weight'] for k, v in m['folds'].items()}
    if name in m['label_sets']:
        if sha256(file) != m['label_sets'][name]['sha256'] or semantic != m['label_sets'][name]['semantic_sha256']:
            raise SystemExit(f'{name} file hash differs from the manifest')
        for k, v in m['folds'].items():
            if not np.array_equal(np.asarray(weights[k], np.float32), np.asarray(v['pos_weight'], np.float32)):
                raise SystemExit(f'{name} fold {k} pos_weight differs from the manifest')
    eligible = set().union(*map(set, trains.values()))
    record = dict(schema=LABEL_SCHEMA, name=name, registered_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), path=rel, sha256=sha256(file),
                  semantic_sha256=semantic, rows=len(raw), loaded_rows=len(labels), eligible_covered=len(eligible), gold_overlap=0,
                  extra_rows_outside_g1=len(set(labels) - eligible), protocol_sha256=sha256(manifest_path),
                  pos_weight_source=pos_weight_source, pos_weight=weights, tags={k: tag_for(name, k, 'full') for k in FOLDS})
    out.parent.mkdir(exist_ok=True)
    dump(out, record)
    print(json.dumps({k: record[k] for k in ('name', 'path', 'sha256', 'semantic_sha256', 'rows', 'loaded_rows', 'eligible_covered', 'gold_overlap',
                                             'pos_weight_source', 'protocol_sha256')}, indent=1))
    return record


def load_registration(root, manifest, name, manifest_path):
    f = Path(root) / manifest['paths']['label_sets_dir'] / f'{name}.json'
    if not f.is_file():
        raise ValueError(f'label set {name!r} is not registered ({f})')
    r = read(f)
    if r['schema'] != LABEL_SCHEMA or r['name'] != name or r['protocol_sha256'] != sha256(manifest_path):
        raise ValueError('label-set registration does not belong to this protocol')
    if sha256(relpath(root, r['path'])) != r['sha256']:
        raise ValueError('label file changed since registration: ' + r['path'])
    if name in manifest['label_sets'] and (manifest['label_sets'][name]['sha256'] != r['sha256'] or manifest['label_sets'][name]['path'] != r['path']):
        raise ValueError('label-set registration differs from the manifest registry')
    return r, f


# ------------------------------------------------------------------ runner guard
def validate_arguments(args, root):
    """Called by the runner before any training import. Returns (protocol, fold spec, ordered ids, registration, registration file)."""
    root = Path(root)
    manifest_path = Path(args.protocol)
    p = read(manifest_path)
    if p['schema'] != SCHEMA or manifest_path.resolve() != (root / PROTOCOL / 'manifest.json').resolve():
        raise ValueError('wrong protocol')
    for path, expected in p['source_bindings'].items():
        if sha256(relpath(root, path)) != expected:
            raise ValueError('frozen binding changed: ' + path)
    for name, expected in p['training_code_sha256'].items():
        if sha256(root / 'training/src' / name) != expected:
            raise ValueError('original training source changed')
    for k, v in p['common_args'].items():
        if getattr(args, k, None) != v:
            raise ValueError('frozen argument changed: ' + k)
    if args.seed != p['seed']:
        raise ValueError('seed changed')
    if str(args.fold) not in p['folds']:
        raise ValueError('unregistered fold')
    fold = p['folds'][str(args.fold)]
    if tag_for(args.label_set, args.fold, args.mode) != args.tag:
        raise ValueError('tag/fold/label-set/mode mismatch')
    reg, reg_file = load_registration(root, p, args.label_set, manifest_path)
    if Path(args.labels).resolve() != relpath(root, reg['path']).resolve():
        raise ValueError('--labels differs from the registered label file')
    if Path(args.eval_labels).resolve() != relpath(root, p['paths']['eval_labels']).resolve():
        raise ValueError('--eval-labels differs from the registered Dread file')
    if Path(args.corpus).resolve() != relpath(root, p['paths']['corpus']).resolve():
        raise ValueError('input path differs: corpus')
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
    return p, fold, ordered, reg, reg_file


def runner_argv(p, fold, name, mode, root=ROOT, reg=None):
    reg = reg or read(Path(root) / p['paths']['label_sets_dir'] / f'{name}.json')
    argv = [train_python(), str(Path(root) / RUNNER), '--protocol', str(Path(root) / PROTOCOL / 'manifest.json'), '--fold', str(int(fold)),
            '--label-set', name, '--mode', mode, '--tag', tag_for(name, fold, mode), '--seed', str(p['seed'])]
    for k, v in p['common_args'].items():
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        else:
            argv += ['--' + k.replace('_', '-'), str(v)]
    argv += ['--corpus', str(relpath(root, p['paths']['corpus'])), '--labels', str(relpath(root, reg['path'])),
             '--eval-labels', str(relpath(root, p['paths']['eval_labels']))]
    return argv


def _protocol():
    return read(ROOT / PROTOCOL / 'manifest.json')


def queue_entries(name, folds, after=None, mode='full'):
    p = _protocol(); reg, _ = load_registration(ROOT, p, name, ROOT / PROTOCOL / 'manifest.json')
    entries, previous = [], after
    for k in folds:
        if str(k) not in p['folds']:
            raise SystemExit(f'unregistered fold {k}')
        e = dict(tag=tag_for(name, k, mode), command=runner_argv(p, k, name, mode, reg=reg))
        if previous:
            e['after'] = previous
        e['continue_on_failure'] = True  # folds are independent fits; a failed fold does not invalidate the next
        entries.append(e); previous = e['tag']
    return entries


def launch(fold, name, mode):
    p = _protocol(); tag = tag_for(name, fold, mode)
    reg, _ = load_registration(ROOT, p, name, ROOT / PROTOCOL / 'manifest.json')
    if str(fold) not in p['folds']:
        raise SystemExit('unregistered fold')
    if int(fold) == 0 and name == 'dread' and mode == 'full' and (ROOT / PROTOCOL / 'fold0-reuse.json').exists():
        raise SystemExit('fold 0 / dread is already covered by fold0-reuse.json (EXP094L_B8_SEED42)')
    out = ROOT / 'artifacts/experiments' / tag
    receipt_path = ROOT / PROTOCOL / f'launch-{tag}.json'
    if (out.exists() and any(out.iterdir())) or receipt_path.exists():
        raise SystemExit('output or launch receipt exists: ' + tag)
    env = {**os.environ, **THREAD_ENV, 'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(ROOT)}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    argv = [train_python(), str(ROOT / 'tools/gpu_job.py'), '--tag', tag, '--', *runner_argv(p, fold, name, mode, reg=reg)]
    receipt = dict(status='LAUNCHED', fold=int(fold), label_set=name, mode=mode, tag=tag, argv=argv, protocol_sha256=sha256(ROOT / PROTOCOL / 'manifest.json'),
                   label_set_sha256=reg['sha256'], started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), budget_seconds=p['training_budgets'][mode + '_seconds'], retry=0)
    r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    receipt.update(launcher_returncode=r.returncode, launcher_stdout=r.stdout[-2000:], launcher_stderr=r.stderr[-2000:])
    dump(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'tag', 'started_at', 'launcher_returncode', 'launcher_stdout', 'launcher_stderr')}, ensure_ascii=False, indent=1))


def status(fold, name, mode):
    tag = tag_for(name, fold, mode)
    r = subprocess.run([sys.executable, str(ROOT / 'tools/gpu_job.py'), '--status', tag], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-1500:])
    for f in ('pilot-receipt.json', 'summary.json'):
        path = ROOT / 'artifacts/experiments' / tag / f
        if path.exists():
            print(f, path.read_text(encoding='utf-8')[:800])


# ------------------------------------------------------------------ fold-0 reuse
def reuse_fold0(root=ROOT):
    import numpy as np
    import pandas as pd
    root = Path(root)
    manifest_path = root / PROTOCOL / 'manifest.json'
    p = read(manifest_path); f0 = p['folds']['0']
    out = root / PROTOCOL / 'fold0-reuse.json'
    if out.exists():
        raise SystemExit('fold0-reuse.json already exists')
    reg, _ = load_registration(root, p, 'dread', manifest_path)
    run = root / REUSE_RUN; tag = Path(REUSE_RUN).name
    needed = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'output-inventory.json']
    missing = [n for n in needed if not (run / n).is_file()]
    log_dir = root / 'artifacts/logs' / tag
    if missing or not (log_dir / 'SUCCESS').is_file():
        state = 'FAILED' if (log_dir / 'FAILED').exists() else 'running/absent'
        print(json.dumps(dict(status='NOT_FINISHED', run=REUSE_RUN, job_state=state, missing=missing), indent=1))
        raise SystemExit(2)
    s, prov, inv = read(run / 'summary.json'), read(run / 'provenance.json'), read(run / 'output-inventory.json')
    a = s['args']
    checks = dict(
        train_ids=prov['train_ids_ordered_sha256'] == f0['ordered_ids']['train_sha256'],
        eval_ids=prov['eval_ids_ordered_sha256'] == f0['ordered_ids']['val_sha256'],
        common_args=all(a.get(k) == v for k, v in p['common_args'].items()),
        seed=a.get('seed') == SEED and prov.get('seed') == SEED, variant=a.get('variant') == 'seed' and a.get('slot_dropout') == 0.0,
        labels=Path(a['labels']).resolve() == (root / DREAD).resolve() and Path(a['eval_labels']).resolve() == (root / DREAD).resolve(),
        corpus=Path(a['corpus']).resolve() == relpath(root, p['paths']['corpus']).resolve(),
        label_hash=prov['label_hash'] == reg['semantic_sha256'] == p['label_sets']['dread']['semantic_sha256'],
        pos_weight=np.array_equal(np.asarray(prov['actual_pos_weight'], np.float32), np.asarray(f0['pos_weight'], np.float32)),
        initial_state=prov['initial_state_sha256'] == p['initial_state_sha256'],
        updates=s['optimizer_updates'] == f0['steps'] and s['skipped_updates'] <= f0['max_fp16_skips'],
        fixed_final=s['selected_epoch_1based'] == p['common_args']['epochs'] and prov['checkpoint_selection'] == 'fixed-final',
        counts=s['train'] == f0['n_train'] and s['val'] == f0['n_val'],
        inventory=all(sha256(run / n) == inv[n]['sha256'] for n in inv),
    )
    oof = pd.read_csv(run / 'oof_best.csv', dtype={UID: str})
    checks['oof_order'] = oof[UID].tolist() == read(root / f0['ordered_ids']['path'])['val'] and list(oof.columns) == [UID] + T
    if not all(checks.values()):
        print(json.dumps(dict(status='MISMATCH', checks=checks), indent=1))
        raise SystemExit(3)
    record = dict(status='REUSED_AS_FOLD0', fold=0, label_set='dread', run=REUSE_RUN, tag=tag, checks=checks, protocol_sha256=sha256(manifest_path),
                  outputs={n: inv[n] for n in inv}, k36_available=(run / EXTRA_EVAL['file']).is_file(),
                  note='identical fold-0 contract (G1 T3455/V888, same args, seed 42, Dread); trained by training/runners/exp094_b8_variant.py, '
                       'so no oof_k36.csv unless produced separately (assemble --k36 0=PATH)',
                  registered_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    dump(out, record)
    print(json.dumps(dict(status=record['status'], checks=checks), indent=1))
    return record


# ------------------------------------------------------------------ assemble
def fold_run_dir(root, p, name, k):
    reuse = Path(root) / PROTOCOL / 'fold0-reuse.json'
    if k == 0 and name == 'dread' and reuse.exists():
        r = read(reuse)
        if r['label_set'] == 'dread' and r['protocol_sha256'] == sha256(Path(root) / PROTOCOL / 'manifest.json'):
            return Path(root) / r['run'], True
    return Path(root) / 'artifacts/experiments' / tag_for(name, k, 'full'), False


def assemble(name, k36_override=None, root=ROOT):
    import numpy as np
    import pandas as pd
    root = Path(root); k36_override = k36_override or {}
    manifest_path = root / PROTOCOL / 'manifest.json'
    p = read(manifest_path)
    reg, _ = load_registration(root, p, name, manifest_path)
    out = root / oof_dir_for(name)
    if out.exists():
        raise SystemExit('output exists: ' + str(out))
    val_ids = {int(k): read(root / v['ordered_ids']['path'])['val'] for k, v in p['folds'].items()}
    frames24, frames36, sources, problems = {}, {}, {}, []
    for k in sorted(val_ids):
        run, reused = fold_run_dir(root, p, name, k)
        if not (run / 'summary.json').is_file() or not (run / 'oof_best.csv').is_file():
            problems.append(f'fold {k}: {run.relative_to(root).as_posix()} not finished'); continue
        s = read(run / 'summary.json'); prov = s['provenance']
        if not reused and (prov.get('fold') != k or prov.get('label_set') != name or prov.get('label_semantic_sha256') != reg['semantic_sha256']):
            problems.append(f'fold {k}: provenance does not match fold/label set'); continue
        if prov['eval_ids_ordered_sha256'] != p['folds'][str(k)]['ordered_ids']['val_sha256']:
            problems.append(f'fold {k}: evaluation ids differ'); continue
        frames24[k] = pd.read_csv(run / 'oof_best.csv', dtype={UID: str})
        k36 = Path(k36_override[k]) if k in k36_override else run / EXTRA_EVAL['file']
        if k36.is_file():
            frames36[k] = pd.read_csv(k36, dtype={UID: str})
        sources[k] = dict(run=run.relative_to(root).as_posix(), reused=reused, oof_k24_sha256=sha256(run / 'oof_best.csv'),
                          oof_k36=(str(k36) if k36.is_file() else None),
                          oof_k36_sha256=sha256(k36) if k36.is_file() else None)
    if problems:
        print(json.dumps(dict(status='INCOMPLETE', problems=problems), indent=1))
        raise SystemExit(2)
    result = assemble_and_score(frames24, frames36, val_ids, root / DREAD)
    out.mkdir(parents=True)
    result['oof_k24'].to_csv(out / 'oof_k24.csv', index=False)
    if result['oof_k36'] is not None:
        result['oof_k36'].to_csv(out / 'oof_k36.csv', index=False)
    body = dict(schema='exp099-oof-v1', label_set=name, label_set_sha256=reg['sha256'], observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                protocol_sha256=sha256(manifest_path), rows=int(len(result['oof_k24'])), sources={str(k): v for k, v in sources.items()},
                outputs={f: sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv') if (out / f).exists()},
                scoring=dict(labels=DREAD, rule='float32 then numpy.rint; average-rank AUC as tools/g1_compare.py'),
                metrics=result['metrics'], k36_missing_folds=result['k36_missing'],
                caveats=['weak Dread labels', 'fold 0 val = exposed G1 V888; folds 1-4 models saw G1 val in training', 'pooled AUC mixes 5 models'])
    (out / 'result.json').write_text(json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (out / 'result.md').write_text(result_markdown(name, body), encoding='utf-8')
    print(json.dumps(dict(rows=body['rows'], metrics={k: {kk: (round(vv['macro12'], 6) if isinstance(vv, dict) else vv) for kk, vv in v.items()}
                                                      for k, v in body['metrics'].items()}, k36_missing_folds=body['k36_missing_folds']), indent=1))
    return body


def assemble_and_score(frames24, frames36, val_ids, label_path):
    import numpy as np
    labels, _ = _load_labels(label_path)
    expected = sorted(u for ids in val_ids.values() for u in ids)
    if len(expected) != len(set(expected)):
        raise ValueError('fold validation sets overlap')
    metrics, k36_missing, tables = {}, sorted(set(val_ids) - set(frames36)), {}
    for key, frames in (('k24', frames24), ('k36', frames36)):
        if key == 'k36' and k36_missing:
            tables[key] = None; continue
        table = assemble_frames(frames, val_ids)
        if table['StudyInstanceUID'].tolist() != expected:
            raise ValueError(f'{key}: OOF does not cover every study exactly once')
        m = {}
        for k in sorted(val_ids):
            ids = val_ids[k]
            Y = np.rint(np.stack([labels[u] for u in ids]).astype(np.float32)).astype(int)
            S = frames[k][T].to_numpy(dtype=np.float64)
            macro, per = macro12(Y, S); m[f'fold{k}'] = dict(n=len(ids), macro12=macro, targets=per)
        Y = np.rint(np.stack([labels[u] for u in table['StudyInstanceUID']]).astype(np.float32)).astype(int)
        macro, per = macro12(Y, table[T].to_numpy(dtype=np.float64))
        m['pooled'] = dict(n=len(table), macro12=macro, targets=per)
        m['mean_of_folds'] = float(np.mean([m[f'fold{k}']['macro12'] for k in sorted(val_ids)]))
        metrics[key] = m; tables[key] = table
    return dict(oof_k24=tables['k24'], oof_k36=tables.get('k36'), metrics=metrics, k36_missing=k36_missing)


def result_markdown(name, body):
    lines = [f'# EXP099 B8 5-fold OOF ({name})', '', f"rows {body['rows']}; labels Dread rint; K36 missing folds: {body['k36_missing_folds'] or 'none'}", '',
             '| scope | ' + ' | '.join(body['metrics']) + ' |', '|---|' + '---:|' * len(body['metrics'])]
    scopes = list(next(iter(body['metrics'].values())))
    for s in scopes:
        cells = []
        for m in body['metrics'].values():
            v = m.get(s)
            cells.append(f"{v['macro12']:.6f}" if isinstance(v, dict) else (f'{v:.6f}' if isinstance(v, float) else '-'))
        lines.append(f'| {s} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| target | ' + ' | '.join(f'pooled {k}' for k in body['metrics']) + ' |', '|---|' + '---:|' * len(body['metrics'])]
    for t in T:
        lines.append(f'| {t} | ' + ' | '.join(f"{m['pooled']['targets'][t]:.6f}" for m in body['metrics'].values()) + ' |')
    return '\n'.join(lines + [''])


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('register-labels'); s.add_argument('--name', required=True); s.add_argument('--path', required=True)
    s.add_argument('--pos-weight', choices=['labels', 'dread'], default='labels', help='loss pos_weight from this file (default) or fixed Dread vectors')
    for cmd in ('launch', 'status', 'argv'):
        s = sub.add_parser(cmd); s.add_argument('--fold', type=int, choices=FOLDS, required=True); s.add_argument('--labels', required=True)
        s.add_argument('--mode', choices=['pilot', 'full'], default='full')
    s = sub.add_parser('queue-json'); s.add_argument('--labels', required=True); s.add_argument('--folds', default='1,2,3,4')
    s.add_argument('--after', required=True, help='tag the first entry waits for (the current GPU chain tail); "none" only for an otherwise idle GPU'); s.add_argument('--mode', choices=['pilot', 'full'], default='full')
    s.add_argument('--out', default=None, help='also write the entries to this new JSON file')
    sub.add_parser('reuse-fold0')
    s = sub.add_parser('assemble'); s.add_argument('--labels', required=True)
    s.add_argument('--k36', action='append', default=[], help='FOLD=PATH oof_k36.csv override (e.g. a separate K36 re-evaluation of the reused fold 0)')
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare()
    elif a.command == 'register-labels':
        register_labels(a.name, a.path, a.pos_weight)
    elif a.command == 'argv':
        print(json.dumps(runner_argv(_protocol(), a.fold, a.labels, a.mode), indent=1))
    elif a.command == 'queue-json':
        entries = queue_entries(a.labels, [int(x) for x in a.folds.split(',')], None if a.after.lower() == 'none' else a.after, a.mode)
        if a.out:
            dump(a.out, entries)
        print(json.dumps(entries, indent=1))
    elif a.command == 'launch':
        launch(a.fold, a.labels, a.mode)
    elif a.command == 'status':
        status(a.fold, a.labels, a.mode)
    elif a.command == 'reuse-fold0':
        reuse_fold0()
    else:
        assemble(a.labels, {int(x.split('=', 1)[0]): x.split('=', 1)[1] for x in a.k36})


if __name__ == '__main__':
    main()
