"""H-C-12: CPU-only mask/UID audit; prepare freezes inputs, run counts windows.

Usage: system Python tools/hc12_window_reuse.py prepare|run
No prediction values, pixels, checkpoints, labels or torch are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/experiments/HC12_WINDOW_REUSE_AUDIT_V1'
CORPUS = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
MEMBERSHIP = ROOT / 'artifacts/experiments/EXP084_B_G1/oof_best.csv'
SOURCE = ROOT / 'training/src/raptor_common.py'
INPUTS = [MEMBERSHIP, CORPUS / 'manifest.json', CORPUS / 'ids.npy', CORPUS / 'masks.npy',
          SOURCE, ROOT / 'tools/exp096_k_sweep.py', Path(__file__).resolve()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def read_uids():
    # Only the first field is interpreted; hashes bind the entire source file.
    with MEMBERSHIP.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        assert next(reader)[0] == 'StudyInstanceUID'
        uids = [row[0] for row in reader]
    assert len(uids) == len(set(uids)) == 888
    return uids


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in INPUTS}
    uids = read_uids()
    save(OUT / 'manifest.json', dict(
        task='H-C-12', frozen_at=datetime.now().astimezone().isoformat(), inputs=inputs,
        ordered_uids=uids, n=888, escalation_count=222, k_initial=24, k_refine=36,
        policy='within-slot-valid', budget_seconds=600, expected_savings_gate=0.20,
        gate_comparator='deduplicated always K36, sentinel counts as a feature',
        protocol='plan.md H-C-12 registered 2026-10-07 15:37 KST',
        exclusions=['labels', 'prediction values', 'pixels', 'models', 'GPU', 'routing fitting'],
        duplicate_semantics='encode each unique center once per study/checkpoint; gather original ordered bag including multiplicities',
    ))
    print(json.dumps({'state': 'FROZEN', 'manifest_sha256': digest(OUT / 'manifest.json')}))


def independent_centers(mask, sizes, k):
    valid = {i for i, v in enumerate(mask) if v > 0}
    pool = []
    start = 0
    for size in sizes:
        for center in range(start + 1, start + size - 1):
            if {center - 1, center, center + 1} <= valid:
                pool.append(center)
        start += size
    if not pool:
        return [-1] * k
    # Exact rational nearest index with round-to-even. Independent of linspace.
    selected = []
    for j in range(k):
        q, r = divmod(j * (len(pool) - 1), k - 1)
        if 2 * r > k - 1 or (2 * r == k - 1 and q % 2):
            q += 1
        selected.append(pool[q])
    return selected


def run():
    started = time.monotonic()
    manifest_path = OUT / 'manifest.json'
    manifest_hash = digest(manifest_path)
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert not (OUT / 'result.json').exists(), 'Frozen results must not be overwritten'
    for path, expected in m['inputs'].items():
        assert digest(ROOT / path) == expected, f'input changed: {path}'
    assert read_uids() == m['ordered_uids']
    spec = json.loads((CORPUS / 'manifest.json').read_text(encoding='utf-8'))['spec']
    sizes = [int(s[2]) for s in spec['slots']]
    assert sizes == [18, 14, 12, 8, 12] and sum(sizes) == 64
    ids = np.load(CORPUS / 'ids.npy', allow_pickle=False).astype(str).tolist()
    masks = np.load(CORPUS / 'masks.npy', allow_pickle=False)
    assert masks.shape == (len(ids), 64) and len(set(ids)) == len(ids)
    assert set(np.unique(masks)).issubset({0, 1})
    row_by_uid = {u: i for i, u in enumerate(ids)}
    assert set(m['ordered_uids']) <= row_by_uid.keys()
    sys.path.insert(0, str(ROOT / 'training/src'))
    from raptor_common import select_centers

    rows = []
    for uid in m['ordered_uids']:
        mask = masks[row_by_uid[uid]]
        centers = {}
        for k in (24, 36):
            got = independent_centers(mask, sizes, k)
            reference = select_centers(mask, 64, k, spec, 'within-slot-valid')
            assert got == reference, (uid, k, 'ordered centers mismatch')
            assert all(c == -1 or 1 <= c <= 62 for c in got)
            unique = list(dict.fromkeys(got))
            gather = [unique.index(c) for c in got]
            assert [unique[i] for i in gather] == got
            centers[k] = got
        a, b = set(centers[24]), set(centers[36])
        rows.append(dict(uid=uid, centers24=centers[24], centers36=centers[36],
                         unique24=len(a), unique36=len(b), intersection=len(a & b),
                         novel36=len(b - a), union=len(a | b),
                         duplicates24=24-len(a), duplicates36=36-len(b),
                         invalid_sentinel24=centers[24].count(-1),
                         invalid_sentinel36=centers[36].count(-1),
                         nested=a <= b))
        assert time.monotonic() - started <= m['budget_seconds']

    keys = ['unique24', 'unique36', 'intersection', 'novel36', 'union',
            'duplicates24', 'duplicates36', 'invalid_sentinel24', 'invalid_sentinel36']
    totals = {key: sum(r[key] for r in rows) for key in keys}
    n, e = m['n'], m['escalation_count']
    novel = sorted(r['novel36'] for r in rows)
    costs = {'uniform_random_expected': totals['unique24'] + e/n * totals['novel36'],
             'minimum_exactly_222': totals['unique24'] + sum(novel[:e]),
             'maximum_exactly_222': totals['unique24'] + sum(novel[-e:])}
    comparators = {'current_always_K36': n*36,
                   'deduplicated_always_K36': totals['unique36']}
    savings = {base: {mode: 1 - cost/denominator for mode, cost in costs.items()}
               for base, denominator in comparators.items()}
    count_gate = savings['deduplicated_always_K36']['uniform_random_expected'] >= m['expected_savings_gate']
    coverage_blockers = []
    if totals['invalid_sentinel24'] or totals['invalid_sentinel36']:
        coverage_blockers.append('empty-input sentinel present: routing efficacy requires explicit handling')
    result = dict(task='H-C-12', manifest_sha256=manifest_hash,
                  completed_at=datetime.now().astimezone().isoformat(),
                  n=n, escalated=e, numpy=np.__version__, totals=totals,
                  nested_studies=sum(r['nested'] for r in rows),
                  nonnested_studies=sum(not r['nested'] for r in rows),
                  per_study={key: {'min': min(r[key] for r in rows),
                                   'max': max(r[key] for r in rows),
                                   'mean': totals[key]/n} for key in keys},
                  comparators=comparators, adaptive_costs=costs, savings=savings,
                  gate=dict(threshold=m['expected_savings_gate'], count_pass=count_gate,
                            coverage_blockers=coverage_blockers,
                            propose_routing_replay=count_gate and not coverage_blockers),
                  validation=dict(ordered_reference_matches=2*n,
                                  duplicate_bag_reconstructions=2*n,
                                  uid_count=n, mask_depth=64, no_model_import='torch' not in sys.modules),
                  limitations=['encoder-window counts, not measured time or 9h feasibility',
                               'random escalation opportunity, not learned routing efficacy',
                               'features reusable only inside one fixed checkpoint/resolution/eval preprocessing',
                               'MIL gather must retain order and multiplicity; sentinel is encoded black input',
                               'GPU batching numerical parity remains untested'],
                  elapsed_seconds=time.monotonic()-started)
    assert result['validation']['no_model_import']
    for path, expected in m['inputs'].items():
        assert digest(ROOT / path) == expected, f'input changed during audit: {path}'
    assert digest(manifest_path) == manifest_hash
    assert time.monotonic()-started <= m['budget_seconds']
    save(OUT / 'rows.json', rows)
    save(OUT / 'result.json', result)
    save(OUT / 'VALIDATED_SUCCESS.json', dict(state='VALIDATED_SUCCESS',
        scope='mask-count audit only; no efficacy/runtime validation',
        manifest_sha256=manifest_hash,
        artifacts={p.name: digest(p) for p in [OUT/'rows.json', OUT/'result.json']},
        validation=result['validation']))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else run()
