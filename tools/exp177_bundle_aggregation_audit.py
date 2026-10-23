"""EXP177 v1.1.0: CPU-only H-B-78 aggregation diagnostic; B runs the audit.

Reads only registered OOF predictions/metadata and structured reference labels.
Writes aggregate results to a NEW directory. No inference or submission.

Changelog v1.1.0: timing is measurement only; legacy seconds arguments and
protocol max_seconds do not stop work. Scientific/source/identity gates remain.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VERSION = '1.1.0'
EXP = Path('artifacts/experiments/EXP177_BUNDLE_AGG_AUDIT')
PROTOCOL = EXP / 'PROTOCOL.md'
TOOL = 'tools/exp177_bundle_aggregation_audit.py'
EXP170 = Path('artifacts/experiments/EXP170_CONVNEXT_T_L2/outputs')
FAMILIES = ('l2', 'dino', 'convnext')
TTA_INPUTS = ('l2_s42', 'l2_s43', 'dino', 'convnext')
INPUTS = (*TTA_INPUTS, 'h1')
REFS = ('jev', 'aquila', 'dread', 'consensus')
DRAWS, SEED, MAX_SECONDS = 300, 20261003, 7200
TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
BIND_FILES = (TOOL, 'tools/exp169_bundle_screen.py', 'tools/exp149_label_sources.py',
              'tools/exp115_vlm_eval.py', 'tools/exp156_k_sweep.py',
              'tests/test_exp177_bundle_aggregation_audit.py')
RELEASE = 'tools/exp169_partner_bundle_release.py'
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'BLIS_NUM_THREADS')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_bindings(root):
    root = Path(root)
    src = (root / RELEASE).read_text(encoding='utf-8')
    funcs = {n.name: hashlib.sha256(ast.get_source_segment(src, n).encode()).hexdigest()
             for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
             and n.name in ('rank', 'aggregate', 'final_values')}
    need(len(funcs) == 3, 'Serving aggregation functions missing')
    return {'files': {p: sha(root / p) for p in BIND_FILES}, 'serving_functions': funcs}


def verify_protocol(root):
    text = (Path(root) / PROTOCOL).read_text(encoding='utf-8')
    m = re.search(r'<!-- EXP177_CONTRACT\s+(.*?)\s+-->', text, re.S)
    need(m is not None, 'Register EXP177 protocol first')
    contract = json.loads(m[1])
    need(contract.get('version') == VERSION and contract.get('draws') == DRAWS and
         contract.get('seed') == SEED,
         'EXP177 protocol parameters unregistered or changed')
    need(contract.get('source_bindings') == source_bindings(root),
         'Registered source changed; review and re-register before running')
    return contract


class Budget:
    def __init__(self, seconds=MAX_SECONDS):
        # Keep the legacy argument accepted but never use it to limit work.
        self.wall, self.cpu = time.monotonic(), time.process_time()

    def usage(self):
        return {'wall_seconds': time.monotonic() - self.wall,
                'cpu_seconds': time.process_time() - self.cpu}

    def check(self):
        """Compatibility measurement checkpoint; never a time-based admission gate."""
        return self.usage()


@contextlib.contextmanager
def cpu_limits(threads):
    need(threads in (1, 2, 3, 4), 'Threads must be 1..4')
    previous = {k: os.environ.get(k) for k in (*THREAD_ENV, 'CUDA_VISIBLE_DEVICES')}
    try:
        for k in THREAD_ENV:
            os.environ[k] = str(threads)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # Also applies when numpy was imported by a test or embedding process.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=threads):
            yield
    finally:
        for k, value in previous.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value


def paths(root, fold):
    from tools import exp169_bundle_screen as helper
    need(fold in range(5), 'Invalid held-out fold')
    old = helper.paths(Path(root), fold)
    # Never read old['convnext']: it is the wrong Dread EXP166/167 family.
    plain = {n: old[n] for n in INPUTS if n != 'convnext'}
    plain['convnext'] = (Path(root) / EXP170 / 'results' /
                         f'EXP170_CONVNEXT_T_L2_S42_F{fold}' / 'oof_k36.csv')
    flip = {n: plain[n].with_name('oof_k36_flip_tta.csv') for n in TTA_INPUTS}
    return plain, flip


def prediction(path, ids, records, root):
    """Reuse EXP169 strict CSV/receipt helpers; reject its summary-only fallback."""
    from tools import exp169_bundle_screen as helper
    need(path.is_file(), 'Missing registered prediction: ' + str(path))
    receipts = [path.parent / n for n in
                ('output-inventory.json', 'receipt.json', 'collect-receipt.json')]
    need(any(p.is_file() for p in receipts), 'Prediction has no hash-bound receipt: ' + str(path))
    helper._verify_inventory(path)
    before = sha(path)
    values = helper.load_oof(path, ids)
    need(sha(path) == before, 'Prediction changed while reading')
    records[str(path.relative_to(root))] = before
    for p in receipts:
        if p.is_file():
            records[str(p.relative_to(root))] = sha(p)
    return values


def load_inputs(root, budget):
    from tools import exp149_label_sources as e149
    root = Path(root)
    records = {}
    # fold_rosters needs metadata only. Do not call _load_manifest, which validates
    # unrelated training sources and is unnecessary for this held-out audit.
    p = read(root / e149.STUDENT_MANIFEST)
    canonical, fold_of = e149.fold_rosters(p, root)
    need(len(canonical) == len(set(canonical)) == 4343, 'Expected EXP099 4343 studies')
    roster_files = [e149.STUDENT_MANIFEST, p['paths']['base_ordered_ids']]
    for f in range(5):
        entry = p['folds'][str(f)]['ordered_ids']
        need(sha(root / entry['path']) == entry['sha256'], 'Registered fold roster changed')
        roster_files.append(entry['path'])
    for path in roster_files:
        records[path] = sha(root / path)
    collection = read(root / EXP170 / 'collection.json')
    need(collection['schema'] == 'exp170-convnext-t-l2-v1' and
         collection['status'] == 'VERIFIED' and collection['completed_folds'] == list(range(5)) and
         collection['manifest_sha256'] == sha(root / EXP170 / 'manifest.json'),
         'EXP170 collection/manifest incomplete or changed')
    for name in ('collection.json', 'manifest.json'):
        records[str(EXP170 / name)] = sha(root / EXP170 / name)
    by_fold = {}
    for f in range(5):
        budget.check()
        ids = [u for u in canonical if fold_of[u] == f]
        plain, flip = paths(root, f)
        run = plain['convnext'].parent
        summary = read(run / 'summary.json')
        need(summary['fold'] == f and summary['rows'] == len(ids) and
             summary['backbone'] == 'convnext_tiny.fb_in22k_ft_in1k' and
             summary['label_set'] == 'l2_dread75_aquila25_v1' and
             summary['gold_access'] == 'disabled', 'Wrong EXP170 L2 fold identity')
        inventory = read(run / 'output-inventory.json')
        need(inventory['summary.json']['sha256'] == sha(run / 'summary.json'),
             'EXP170 summary inventory differs')
        records[str((run / 'summary.json').relative_to(root))] = sha(run / 'summary.json')
        values = {n: prediction(path, ids, records, root) for n, path in plain.items()}
        flips = {n: prediction(path, ids, records, root) for n, path in flip.items() if path.is_file()}
        need('convnext' in flips, 'EXP170 flip prediction is required')
        by_fold[f] = {'ids': ids, 'plain': values, 'flip': flips}
    return by_fold, records


def rank(a):
    from tools.exp169_bundle_screen import global_rank
    return global_rank(a)


def pct(a):
    from tools.exp115_vlm_eval import pct_rank
    return pct_rank(a)


def rank_bundle(family):
    """Exact rank/rerank formula in serving; also supports fixed LOFO halves."""
    need(1 <= len(family) <= 3 and set(family) <= set(FAMILIES), 'Invalid families')
    # Canonical summation order avoids arbitrary dict order changing float ties.
    return rank(sum(rank(family[n]) for n in FAMILIES if n in family) / len(family))


def serving_aggregate(scores):
    """Full ten/five/five reducer, parity-tested against EXP169 on CPU only."""
    import numpy as np
    need(set(scores) == set(FAMILIES), 'Incomplete serving families')
    for n, a in scores.items():
        a = np.asarray(a)
        need(a.ndim == 3 and a.shape[0] == (10 if n == 'l2' else 5) and
             a.shape[2] == 12 and a.shape[1] > 0 and np.isfinite(a).all() and
             ((a >= 0) & (a <= 1)).all(), 'Invalid serving probabilities')
    need(len({np.asarray(a).shape[1] for a in scores.values()}) == 1, 'Serving cohorts differ')
    l2 = np.asarray(scores['l2'])
    family = {'l2': rank(((l2[:5] + l2[5:]) / 2).mean(0)),
              'dino': rank(np.asarray(scores['dino']).mean(0)),
              'convnext': rank(np.asarray(scores['convnext']).mean(0))}
    return rank_bundle(family), family


def build_variants(by_fold):
    """No labels read; return frozen variants/contrasts and their common cohort."""
    import numpy as np
    from tools.exp169_bundle_screen import bundle_of
    folds = sorted(by_fold)
    ids = [u for f in folds for u in by_fold[f]['ids']]
    need(len(ids) == len(set(ids)) and len(ids) > 0, 'Cohort duplicate or empty')
    cat = lambda parts: np.concatenate(parts)
    raw = {n: cat([by_fold[f]['plain'][n] for f in folds]) for n in INPUTS}
    raw['convnext'] = cat([by_fold[f]['flip']['convnext'] for f in folds])
    for f in folds:
        for a in (*by_fold[f]['plain'].values(), *by_fold[f]['flip'].values()):
            need(a.shape == (len(by_fold[f]['ids']), 12) and np.isfinite(a).all() and
                 ((a >= 0) & (a <= 1)).all(), 'Invalid fold predictions')
    adm = {'l2': cat([(pct(by_fold[f]['plain']['l2_s42']) +
                       pct(by_fold[f]['plain']['l2_s43'])) / 2 for f in folds]),
           'dino': cat([pct(by_fold[f]['plain']['dino']) for f in folds]),
           'convnext': cat([pct(by_fold[f]['flip']['convnext']) for f in folds])}
    ab, ar = bundle_of(adm)
    variants = {'admission.bundle': ab, **{'admission.' + n: v for n, v in ar.items()},
                'baseline.h1': rank(cat([pct(by_fold[f]['plain']['h1']) for f in folds]))}
    pairs = {}
    def add(key, a, b):
        pairs[key] = {'candidate': a, 'control': b}
    fp = dict(adm, l2=cat([pct((by_fold[f]['plain']['l2_s42'] +
                                by_fold[f]['plain']['l2_s43']) / 2) for f in folds]))
    variants['fold_probability.l2'] = rank(fp['l2'])
    variants['fold_probability.bundle'] = rank_bundle(fp)
    for n in ('l2', 'bundle'):
        add('seed_probability_minus_rank.' + n, 'fold_probability.' + n, 'admission.' + n)
    means64 = {'l2': (raw['l2_s42'] + raw['l2_s43']) / 2,
               'dino': raw['dino'], 'convnext': raw['convnext']}
    raw32 = {n: a.astype(np.float32) for n, a in raw.items()}
    means32 = dict(dino=raw32['dino'], convnext=raw32['convnext'],
                   l2=(raw32['l2_s42'] + raw32['l2_s43']) / 2)
    variants['pooled64.bundle'] = rank_bundle(means64)
    variants['serving_oof.bundle'] = rank_bundle(means32)
    variants['family_probability.bundle'] = rank(sum(means64.values()) / 3)
    add('raw_pooling_minus_fold_normalisation', 'pooled64.bundle', 'fold_probability.bundle')
    add('float32_minus_float64', 'serving_oof.bundle', 'pooled64.bundle')
    add('family_probability_minus_rank', 'family_probability.bundle', 'pooled64.bundle')
    add('serving_oof_minus_admission', 'serving_oof.bundle', 'admission.bundle')
    add('bundle_minus_h1', 'admission.bundle', 'baseline.h1')
    add('serving_oof_minus_h1', 'serving_oof.bundle', 'baseline.h1')
    for family in FAMILIES:
        add('bundle_minus_single.' + family, 'admission.bundle', 'admission.' + family)
        for prefix, source in (('admission', adm), ('serving_oof', means32)):
            key = prefix + '.without_' + family
            variants[key] = rank_bundle({n: a for n, a in source.items() if n != family})
            add(key, key, prefix + '.bundle')
    for seed in ('42', '43'):
        name = 'l2_s' + seed
        seedrank = cat([pct(by_fold[f]['plain'][name]) for f in folds])
        key = 'admission.seed' + seed
        variants[key] = rank(seedrank)
        variants[key + '.bundle'] = rank_bundle(dict(adm, l2=seedrank))
        add(key, key, 'admission.l2')
        add(key + '.bundle', key + '.bundle', 'admission.bundle')
        key = 'serving_oof.seed' + seed + '.bundle'
        variants[key] = rank_bundle(dict(means32, l2=raw32[name]))
        add(key, key, 'serving_oof.bundle')
    convplain = cat([pct(by_fold[f]['plain']['convnext']) for f in folds])
    variants['admission.convnext_plain'] = rank(convplain)
    variants['admission.convnext_plain.bundle'] = rank_bundle(dict(adm, convnext=convplain))
    add('convnext_flip_minus_plain', 'admission.convnext', 'admission.convnext_plain')
    add('convnext_flip_minus_plain.bundle', 'admission.bundle', 'admission.convnext_plain.bundle')
    complete = {n: all(n in by_fold[f]['flip'] for f in folds) for n in TTA_INPUTS}
    for n in ('l2_s42', 'l2_s43', 'dino'):
        if complete[n]:
            flipped = cat([by_fold[f]['flip'][n] for f in folds]).astype(np.float32)
            family = n if n == 'dino' else 'l2'
            value = flipped if n == 'dino' else (flipped + raw32['l2_s43' if n == 'l2_s42' else 'l2_s42']) / 2
            key = 'serving_oof.flip_' + n
            variants[key] = rank_bundle(dict(means32, **{family: value}))
            add(key, key, 'serving_oof.bundle')
    if complete['l2_s42'] and complete['l2_s43']:
        value = sum(cat([by_fold[f]['flip'][n] for f in folds]).astype(np.float32)
                    for n in ('l2_s42', 'l2_s43')) / 2
        variants['serving_oof.flip_l2_both'] = rank_bundle(dict(means32, l2=value))
        add('serving_oof.flip_l2_both', 'serving_oof.flip_l2_both', 'serving_oof.bundle')
    return ids, variants, pairs, means32


def component_draws(ids, fold_of, root):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    membership = {}
    for item in read(Path(root) / ev.COMPONENTS):
        for uid in item['members']:
            need(uid not in membership, 'Duplicate component membership')
            membership[uid] = item['component_id']
    need(all(u in membership for u in ids), 'Missing validation component; no singleton fallback')
    grouped_folds = {}
    for uid in ids:
        grouped_folds.setdefault(membership[uid], set()).add(fold_of[uid])
    need(all(len(v) == 1 for v in grouped_folds.values()), 'Component crosses held-out folds')
    groups = ev.component_groups(ids, root)
    draws = ev.bootstrap_index(groups, DRAWS, SEED)
    weights = np.stack([np.bincount(ii, minlength=len(ids)) for ii in draws]).astype(np.float64)
    draw_hash = hashlib.sha256(b''.join(np.asarray(ii, '<i8').tobytes() for ii in draws)).hexdigest()
    return weights, {'unit': 'validation component', 'draws': DRAWS, 'seed': SEED,
                     'groups': len(groups), 'draws_sha256': draw_hash,
                     'components_sha256': sha(Path(root) / ev.COMPONENTS)}


def weighted_auc(values, ref, weights, budget=None):
    """Exact repeated-row AUC via integer multiplicities, with half credit for ties."""
    import numpy as np
    y, mask, minimum = ref
    need(values.shape == y.shape == mask.shape and values.shape[1] == 12 and
         weights.ndim == 2 and weights.shape[1] == len(values) and
         np.isfinite(values).all() and np.isfinite(weights).all() and
         (weights >= 0).all() and np.isin(y[mask], (0, 1)).all(), 'Invalid AUC inputs')
    out = np.full((len(weights), 12), np.nan)
    for t in range(12):
        if budget:
            budget.check()
        order = np.flatnonzero(mask[:, t])
        order = order[np.argsort(values[order, t], kind='stable')]
        if not len(order):
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(values[order, t])) + 1]
        yy = y[order, t]
        for begin in range(0, len(weights), 32):
            w = weights[begin:begin + 32, order]
            pos = np.add.reduceat(w * yy, starts, axis=1)
            neg = np.add.reduceat(w * (1 - yy), starts, axis=1)
            total_p, total_n = pos.sum(1), neg.sum(1)
            good = (total_p >= minimum) & (total_n >= minimum)
            numerator = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(1)
            np.divide(numerator, total_p * total_n, out=out[begin:begin + len(w), t], where=good)
    return out


def finite_or_none(a):
    import numpy as np
    return [float(x) if np.isfinite(x) else None for x in a]


def interval(values):
    import numpy as np
    good = np.asarray(values)[np.isfinite(values)]
    return {'ci95': np.percentile(good, [2.5, 97.5]).tolist() if len(good) else None,
            'valid_draws': len(good), 'invalid_draws': len(values) - len(good)}


def fixed_macro(targets):
    # np.mean deliberately propagates NaN: never change the twelve-target set.
    return targets.mean(axis=-1)


def evaluate(variants, pairs, refs, weights, fold_labels, budget):
    import numpy as np
    samples = np.vstack([np.ones(weights.shape[1]), weights])
    cache, scores = {}, {}
    for name, values in variants.items():
        cache[name], scores[name] = {}, {}
        for refname in REFS:
            all_auc = weighted_auc(values, refs[refname], samples, budget)
            macro = fixed_macro(all_auc)
            cache[name][refname] = all_auc
            scores[name][refname] = {'macro': finite_or_none([macro[0]])[0],
                                    'scorable_targets': int(np.isfinite(all_auc[0]).sum()),
                                    'targets': dict(zip(TARGETS, finite_or_none(all_auc[0])))}
    comparisons = {}
    for name, pair in pairs.items():
        comparisons[name] = {**pair, 'references': {}}
        for refname in REFS:
            a, b = cache[pair['candidate']][refname], cache[pair['control']][refname]
            delta = a - b
            comparisons[name]['references'][refname] = {
                'candidate': scores[pair['candidate']][refname]['macro'],
                'control': scores[pair['control']][refname]['macro'],
                'delta': finite_or_none([fixed_macro(delta[0])])[0],
                **interval(fixed_macro(delta[1:])),
                'targets': {t: {'delta': finite_or_none([delta[0, j]])[0], **interval(delta[1:, j])}
                            for j, t in enumerate(TARGETS)}}
    per_fold = {}
    for f in sorted(set(fold_labels)):
        ix = np.flatnonzero(fold_labels == f)
        per_fold[str(int(f))] = {}
        for name, values in variants.items():
            per_fold[str(int(f))][name] = {}
            for refname, (y, mask, minimum) in refs.items():
                auc = weighted_auc(values[ix], (y[ix], mask[ix], minimum), np.ones((1, len(ix))), budget)[0]
                per_fold[str(int(f))][name][refname] = {
                    'macro': finite_or_none([fixed_macro(auc)])[0],
                    'targets': dict(zip(TARGETS, finite_or_none(auc)))}
    # Exact per-fold contrast deltas; CIs are pooled component intervals only.
    fold_deltas = {f: {name: {r: None if table[p['candidate']][r]['macro'] is None or
                                    table[p['control']][r]['macro'] is None else
                                    table[p['candidate']][r]['macro'] - table[p['control']][r]['macro']
                             for r in REFS} for name, p in pairs.items()}
                   for f, table in per_fold.items()}
    ties = {name: {t: int(len(a) - len(np.unique(a[:, j]))) for j, t in enumerate(TARGETS)}
            for name, a in variants.items()}
    coverage = {n: {t: {'scored': int(m[:, j].sum()), 'positive': int(y[m[:, j], j].sum()),
                        'negative': int(m[:, j].sum() - y[m[:, j], j].sum())}
                    for j, t in enumerate(TARGETS)} for n, (y, m, _) in refs.items()}
    return {'scores': scores, 'comparisons': comparisons, 'per_fold': per_fold,
            'per_fold_deltas': fold_deltas, 'tie_excess_rows': ties, 'reference_coverage': coverage}


def reconstruct_shards(ids, shards):
    """Rejoin raw rows by ID before ranking; duplicated/missing IDs fail closed."""
    import numpy as np
    need(len(ids) == len(set(ids)), 'Duplicate canonical IDs')
    by_id = {}
    for shard_ids, values in shards:
        need(len(shard_ids) == len(values), 'Shard shape differs')
        for uid, row in zip(shard_ids, values):
            need(uid not in by_id, 'Duplicate shard ID')
            by_id[uid] = row
    need(set(by_id) == set(ids), 'Missing/extra shard IDs')
    return np.stack([by_id[u] for u in ids])


def invariance(by_fold, means):
    import numpy as np
    from tools import exp169_partner_bundle_release as release
    ids, original, _, _ = build_variants(by_fold)
    rng = np.random.default_rng(SEED)
    p = rng.permutation(len(ids))
    base = rank_bundle(means)
    perm = rank_bundle({n: a[p] for n, a in means.items()})[np.argsort(p)]
    need(np.array_equal(base, perm), 'Study permutation changed ranks')
    parts = np.array_split(p, 7)
    joined = {n: reconstruct_shards(ids, [([ids[i] for i in ix], a[ix]) for ix in parts])
              for n, a in means.items()}
    need(np.array_equal(base, rank_bundle(joined)), 'Shard reconstruction changed ranks')
    # Reorder rows within every fold and reverse fold input insertion order.
    shuffled = {}
    for f in reversed(sorted(by_fold)):
        block = by_fold[f]
        q = rng.permutation(len(block['ids']))
        shuffled[f] = {'ids': [block['ids'][i] for i in q],
                       **{kind: {n: a[q] for n, a in block[kind].items()} for kind in ('plain', 'flip')}}
    shuffled_ids, other, _, _ = build_variants(shuffled)
    for n, a in original.items():
        aligned = reconstruct_shards(ids, [(shuffled_ids, other[n])])
        need(np.array_equal(a, aligned), 'Fold/row order changed variant: ' + n)
    synthetic = {n: rng.random((10 if n == 'l2' else 5, 37, 12), dtype=np.float32) for n in FAMILIES}
    synthetic['l2'][:, :3, :] = .5  # ties
    expected, _ = release.aggregate(synthetic)
    actual, _ = serving_aggregate(synthetic)
    need(np.array_equal(expected, actual), 'EXP169 aggregate parity failed')
    trio = rng.random(actual.shape)
    need(np.array_equal(release.final_values(trio, actual), .85 * rank(trio) + .15 * rank(actual)),
         'EXP169 final_values parity failed')
    q = np.array([4, 2, 0, 3, 1])
    reordered = {n: a[np.r_[q, q + 5] if n == 'l2' else q] for n, a in synthetic.items()}
    ordered, _ = serving_aggregate(reordered)
    seq_delta = float(np.max(np.abs(ordered - actual)))
    tied = rank(np.full((9, 12), .5))
    need(np.all(tied == .5) and np.all(rank(np.full((1, 12), .5)) == 0), 'Tie convention differs')
    probe = np.tile(np.arange(9)[:, None], (1, 12))
    local = np.concatenate([rank(probe[:3]), rank(probe[3:])])
    need(not np.array_equal(rank(probe), local), 'Shard-local negative control failed')
    return {'study_order': 'PASS', 'raw_shard_rejoin': 'PASS', 'fold_row_order': 'PASS',
            'serving_aggregate_float32': 'PASS', 'serving_final_values': 'PASS', 'average_ties': 'PASS',
            'synthetic_model_order_max_rank_delta': seq_delta,
            'shard_local_rank_negative_control': 'DETECTED',
            'model_order_note': 'Float32 reduction order is not guaranteed bitwise invariant; canonical order retained.',
            'shard_local_rank_warning': 'Only raw shard rejoin before global ranking is valid.'}


def cohort_report(ids, variants, pairs, fold_of, root, budget):
    import numpy as np
    from tools import exp156_k_sweep as e156
    raw_refs = e156.references(ids, root)
    refs = {n: raw_refs[k] for n, k in zip(REFS, ('jev_hybrid', 'aquila_v1', 'dread', 'consensus'))}
    weights, boot = component_draws(ids, fold_of, root)
    return {'studies': len(ids), 'ordered_ids_sha256': digest(ids), 'bootstrap': boot,
            **evaluate(variants, pairs, refs, weights, np.asarray([fold_of[u] for u in ids]), budget)}


def markdown(result):
    note = result['verification_note']
    lines = ['# EXP177 bundle aggregation diagnostic v' + result['version'], '',
             'Verification note: `' + note['path'] + '`',
             'Verification note SHA-256: `' + note['sha256'] + '`', '',
             'DIAGNOSTIC_COMPLETE. No weight search or candidate admission.', '',
             'All contrasts are candidate minus control; paired component CI95. '
             'The serving OOF surrogate has only held-out members; five-fold serving gains are unmeasured.', '']
    for cohort, report in result['cohorts'].items():
        lines += ['## ' + cohort, '', f"Studies: {report['studies']}", '',
                  '| Contrast | JEV delta [CI95] | Aquila | Dread | Consensus |',
                  '|---|---|---|---|---|']
        for name, pair in report['comparisons'].items():
            cells = []
            for ref in REFS:
                v = pair['references'][ref]
                ci = v['ci95']
                point = f"{v['delta']:+.6f}" if v['delta'] is not None else 'unscorable'
                cells.append(point + ' ' + (f"[{ci[0]:+.6f}, {ci[1]:+.6f}]" if ci else '[unscorable]') +
                             f" (n={v['valid_draws']})")
            lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Unmeasured and interpretation', '']
    lines += ['- ' + value for value in result['limitations']]
    lines += ['', 'Input hashes, per-fold/target details, coverage, ties and invariance checks are in result.json.', '']
    return '\n'.join(lines)


def audit(root=ROOT, output=None, threads=1, verification_note=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / EXP / 'run-v1'
    budget = Budget()
    need(verification_note is not None, 'A verification-note path is required')
    note = (root / verification_note).resolve()
    need(note.is_file(), 'Verification note is missing: ' + str(note))
    provenance = {'version': VERSION,
                  'verification_note': {'path': str(note), 'sha256': sha(note)}}
    contract = verify_protocol(root)
    # A caller cannot use this writer outside the authorized workspace.
    need(output.is_relative_to(root / EXP), 'Output must be inside EXP177_BUNDLE_AGG_AUDIT')
    output.mkdir(parents=True, exist_ok=False)
    try:
        with cpu_limits(threads):
            import numpy as np
            import scipy
            from tools import exp156_k_sweep as e156
            from tools import exp115_vlm_eval as ev
            ref_files = (e156.JEV, e156.AQUILA, e156.DREAD, *e156.CONSENSUS)
            reference_hashes = {p: sha(root / p) for p in dict.fromkeys(ref_files)}
            component_hash = sha(root / ev.COMPONENTS)
            by_fold, inputs = load_inputs(root, budget)
            fold_of = {u: f for f, block in by_fold.items() for u in block['ids']}
            ids, variants, pairs, means = build_variants(by_fold)
            engineering = invariance(by_fold, means)
            cohorts = {'full_oof': cohort_report(ids, variants, pairs, fold_of, root, budget)}
            availability = {}
            for n in TTA_INPUTS:
                available = [f for f in sorted(by_fold) if n in by_fold[f]['flip']]
                availability[n] = {'available_folds': available,
                                   'missing_folds': [f for f in range(5) if f not in available],
                                   'status': 'FULL' if len(available) == 5 else 'PARTIAL' if available else 'UNMEASURED'}
                if available:
                    subset_ids = [u for f in available for u in by_fold[f]['ids']]
                    vv = {policy: rank(np.concatenate([pct(by_fold[f][policy][n]) for f in available]))
                          for policy in ('plain', 'flip')}
                    cohorts['tta_' + n] = cohort_report(subset_ids, vv,
                        {'flip_minus_plain': {'candidate': 'flip', 'control': 'plain'}}, fold_of, root, budget)
            for path, expected in {**inputs, **reference_hashes, ev.COMPONENTS: component_hash}.items():
                need(sha(root / path) == expected, 'Input changed during diagnostic: ' + path)
            need(verify_protocol(root) == contract, 'Protocol changed during diagnostic')
            need(sha(note) == provenance['verification_note']['sha256'],
                 'Verification note changed during diagnostic')
            result = {'schema': 'exp177-bundle-aggregation-audit-v1', **provenance,
                      'status': 'DIAGNOSTIC_COMPLETE', 'plan_item': 'H-B-78',
                      'protocol_sha256': sha(root / PROTOCOL), 'contract': contract,
                      'input_sha256': inputs, 'reference_sha256': reference_hashes,
                      'cohorts': cohorts, 'invariance': engineering, 'tta_availability': availability,
                      'resource': {'threads': threads, 'processes': 1,
                                   'timing_policy': 'measurement_only', **budget.usage()},
                      'software': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__},
                      'limitations': [
                          'Weak-label diagnostics; no private-score or CoAtNet-trio gain established.',
                          'Aquila is L2 training-source referenced and an H1 head source; Dread and consensus have shared ancestry.',
                          'H1 is the fixed plain K36 S42 baseline, within-fold percentile ranked then globally ranked; it is not a bundle member.',
                          'OOF uses one held-out fold per study: five-fold serving ensemble effects are unmeasured.',
                          'Serving OOF baseline uses plain RadR50; matched flip effects exist only in explicitly named TTA contrasts.',
                          'DINO remains Dread-trained; EXP172 and direct L2 K62 are outside this audit.',
                          'Float32 pooled arithmetic is a serving surrogate; original CSV admission uses float64 and within-fold ranks.',
                          'Bootstrap intervals condition on fitted models and fixed ranks; they omit training/selection uncertainty.',
                          'TTA missing files are UNMEASURED; partial TTA cohorts do not estimate the full-roster effect.',
                          'No weight search, admission, submission, calibration or multiple-comparison claim.']}
            budget.check()
            write(output / 'result.json', result)
            (output / 'result.md').write_text(markdown(result), encoding='utf-8')
            write(output / 'receipt.json', {'status': result['status'], **provenance,
                  'protocol_sha256': result['protocol_sha256'],
                  'outputs_sha256': {n: sha(output / n) for n in ('result.json', 'result.md')},
                  **budget.usage()})
            return {'status': result['status'], **provenance,
                    'result': str(output / 'result.json'), **budget.usage()}
    except Exception as exc:
        write(output / 'failure.json', {'status': 'FAILED_INCOMPLETE', **provenance,
              'error': type(exc).__name__,
              'reason': str(exc), **budget.usage()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--verification-note', type=Path, required=True,
                        help='Independent verification note for this version (relative to --root or absolute)')
    parser.add_argument('--version', action='version', version=VERSION)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.root, args.output, args.threads, args.verification_note), indent=2))


if __name__ == '__main__':
    main()
