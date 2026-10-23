"""H-B-95 v1.3.0: CPU two-seed anchor gate with release Q7 composition.

Changelog v1.3.0 2026-10-12: honest fold0 CONTROL/CONTROL43 probability mean,
release-selected MIL43, five fixed weights, paired multi-lens comparisons;
fail closed on deployment readout mismatch. Legacy analyses retained as APIs.

Changelog v1.2.0 2026-10-11: post-hoc fine global grid, split-averaged
curves, fixed component halves and discrete 0.0003 flat sets.
Changelog v1.1.0 2026-10-11: post-hoc extended grid, pooled-inner guard repair,
full curves and Q-alone contrasts. v1.0.0 outputs preserved; source backed up.
Changelog v1.0.0 2026-10-11: preregistered two-family search and paired bootstrap
with inner reselection. No GPU/model loading, release or launch operations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import rankdata
from tools.exp162_dino_folds_kaggle import TARGETS
from tools.exp169_bundle_screen import load_oof, sha

VERSION = '1.3.0'
SEED = 20261003
FINE_WEIGHTS = (.35, .40, .45, .50, .55, .60, .65, .70)
WEIGHTS = (.15, *FINE_WEIGHTS)
FLAT_TOLERANCE = .0003
WEAK = np.array([TARGETS.index(t) for t in
                 ('Lateral Meniscus', 'PF OA', 'Lateral OA', 'Synovitis')])
OTHER = np.array([i for i in range(12) if i not in WEAK])
REFS = ('jev', 'consensus', 'dread', 'aquila')
OUT = Path('artifacts/experiments/HB95_PARTNER_WEIGHT/v1.3.0')
FILES = {
    'anchor': 'artifacts/experiments/EXP181_COATNET_F0_BRIDGE/CONTROL/oof_k94_mirror_tta.csv',
    'C4': 'artifacts/experiments/EXP179_MULTISCALE/outputs/M42/results/EXP179_M42_F0/oof_k36_flip_tta.csv',
    'MIL': 'artifacts/experiments/EXP185_FINDING_MIL/outputs/S42/results/EXP185_FINDING_MIL_F0_S42/oof_k36_flip_tta.csv',
    'R42': 'artifacts/experiments/EXP149_L2_S42_F0/oof_k36.csv',
    'R43': 'artifacts/experiments/EXP157_L2_S43_F0/oof_k36.csv',
    'D172': 'artifacts/experiments/EXP172_DINO_BASE_L2/outputs/artifacts/experiments/EXP172_DINO_BASE_L2_S42_F0/oof_best.csv',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rank(a):
    return rankdata(a, axis=0, method='average') / len(a)


def predictions(raw, indices):
    """Only subset predictions enter ranking; labels never enter this function."""
    p = {k: v[indices] for k, v in raw.items()}
    a = rank(p['anchor'])
    r = rank((rank(p['R42']) + rank(p['R43'])) / 2)
    d = rank(p['D172'])
    out = {}
    for q in ('C4', 'MIL'):
        qr = rank((r + rank(p[q]) + d) / 3)
        out[q] = [(1-w)*a + w*qr for w in WEIGHTS]
    return out


def candidates():
    result = []
    for q in ('C4', 'MIL'):
        pairs = {(w, w) for w in WEIGHTS}
        for w, o in sorted(pairs):
            result.append(dict(q=q, weak=w, other=o, global_weight=w == o,
                               name=f'{q}_w{w:.2f}_o{o:.2f}'))
    return result


CANDIDATES = candidates()


def baseline(q):
    return next(i for i, c in enumerate(CANDIDATES)
                if c['q'] == q and c['weak'] == c['other'] == .15)


def tie_key(c):
    return (abs(c['weak']-.15) + abs(c['other']-.15),
            not c['global_weight'], c['q'] != 'C4', c['weak'], c['other'])


def policies():
    return {f'{q}_global': (
        [i for i, c in enumerate(CANDIDATES) if q == 'joint' or c['q'] == q],
        baseline('C4' if q == 'joint' else q)) for q in ('C4', 'MIL', 'joint')}


def component_weights(groups, n, draws=300, seed=SEED):
    """One shared draw roster; no independent resampling of repeated test rows."""
    joined = np.concatenate(groups)
    need(np.array_equal(np.sort(joined), np.arange(n)), 'Components must partition rows')
    rng = np.random.default_rng(seed)
    counts = np.zeros((draws+1, len(groups)), dtype=np.int64)
    counts[0] = 1
    for b in range(1, draws+1):
        counts[b] = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
    weights = np.zeros((draws+1, n), dtype=np.float64)
    for j, group in enumerate(groups):
        weights[:, group] = counts[:, j, None]
    return weights, digest_array(counts.astype('<i8'))


def nested_splits(groups, seed=SEED):
    rng = np.random.default_rng(seed)
    result = []
    for rep in range(5):
        halves = np.array_split(rng.permutation(len(groups)), 2)
        for side in range(2):
            train, test = halves[1-side], halves[side]
            inner = np.array_split(rng.permutation(train), 2)
            result.append((rep, side, [np.concatenate([groups[g] for g in half])
                                      for half in inner],
                           np.concatenate([groups[g] for g in test])))
    return result


def weighted_auc(y, score, mask, minimum, weights):
    """Vectorized exact tied-score AUC for integer bootstrap multiplicities."""
    use = np.flatnonzero(mask)
    order = use[np.argsort(score[use], kind='stable')]
    if not len(order):
        return np.full(len(weights), np.nan)
    start = np.r_[0, np.flatnonzero(np.diff(score[order]) != 0)+1]
    pos = weights[:, order] * y[order]
    neg = weights[:, order] * (1-y[order])
    p, n = pos.sum(1), neg.sum(1)
    pos = np.add.reduceat(pos, start, axis=1)
    neg = np.add.reduceat(neg, start, axis=1)
    numerator = (pos * (np.cumsum(neg, axis=1) - .5*neg)).sum(1)
    out = np.full(len(weights), np.nan)
    ok = (p >= minimum) & (n >= minimum)
    out[ok] = numerator[ok] / (p[ok]*n[ok])
    return out


def score_partition(raw, refs, indices, weights):
    """Scores [draw including point, candidate, reference, target]."""
    pp = predictions(raw, indices)
    ww = weights[:, indices]
    scores = {}
    for q, arrays in pp.items():
        for wi, a in enumerate(arrays):
            values = np.empty((len(weights), len(REFS), 12))
            for ri, ref in enumerate(REFS):
                y, mask, minimum = refs[ref]
                for t in range(12):
                    values[:, ri, t] = weighted_auc(y[indices, t], a[:, t],
                                                   mask[indices, t], minimum, ww)
            scores[q, wi] = values
    result = np.empty((len(weights), len(CANDIDATES), len(REFS), 12))
    for ci, c in enumerate(CANDIDATES):
        result[:, ci] = scores[c['q'], WEIGHTS.index(c['other'])]
        result[:, ci, :, WEAK] = scores[c['q'], WEIGHTS.index(c['weak'])][:, :, WEAK].transpose(2, 0, 1)
    return result


def measures(delta):
    """Leading axes arbitrary; final axes references and targets."""
    primary_targets = delta[..., :2, :].mean(axis=-2)
    primary = primary_targets[..., WEAK].mean(axis=-1)
    macro = primary_targets.mean(axis=-1)
    dread = delta[..., 2, :].mean(axis=-1)
    valid = np.isfinite(delta[..., :3, :]).all(axis=(-2, -1))
    guard = valid & (macro >= -.001-1e-12) & (primary_targets[..., OTHER].min(axis=-1) >= -.010-1e-12) & (dread >= -.003-1e-12)
    return primary, macro, dread, guard


def repair_inner_guards(parts, pooled):
    """Repair unsupported guard cells using outer-training-only pooled scores.

    Arrays are [draw, candidate, reference, target]. AUC support is independent
    of candidate predictions; use a common repair mask for every candidate.
    Objective scores remain the original two-partition mean.
    """
    inner = (parts[0] + parts[1]) / 2
    missing = (~np.isfinite(parts[0]) | ~np.isfinite(parts[1])).any(axis=1)
    missing[:, 3:] = False  # aquila is diagnostic, not a guard
    guard = np.where(missing[:, None], pooled, inner)
    unresolved = (~np.isfinite(guard[:, :, :3])).any(axis=1)
    diagnostic = dict(
        point_repaired_cells=[f'{REFS[r]}:{TARGETS[t]}' for r, t in zip(*np.where(missing[0]))],
        bootstrap_repaired_cells=int(missing[1:].sum()),
        point_unresolved_cells=int(unresolved[0].sum()),
        bootstrap_unresolved_cells=int(unresolved[1:].sum()))
    return inner, guard, diagnostic


def selection_utility(inner, allowed, base, guard_inner=None):
    order = sorted(allowed, key=lambda i: tie_key(CANDIDATES[i]))
    delta = inner[:, order] - inner[:, base, None]
    primary = measures(delta)[0]
    gg = inner if guard_inner is None else guard_inner
    guard = measures(gg[:, order] - gg[:, base, None])[3]
    utility = np.where(guard & np.isfinite(primary), primary, -np.inf)
    return order, utility


def select(inner, allowed, base, guard_inner=None):
    order, utility = selection_utility(inner, allowed, base, guard_inner)
    selected = np.array(order)[utility.argmax(axis=1)]
    selected[~np.isfinite(utility).any(axis=1)] = base
    return selected


def interval(values):
    x = np.asarray(values)
    valid = x[np.isfinite(x)]
    return dict(ci95=np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else [None, None],
                valid_draws=len(valid))


def summary(delta):
    p, m, d, g = measures(delta)
    out = {name: dict(delta=float(arr[0]), **interval(arr[1:]))
           for name, arr in [('primary_weak_four', p), ('macro_mean_jev_consensus', m), ('dread_macro', d)]}
    out['harm_guard'] = dict(passed=bool(g[0]), all_point_cells_defined=bool(np.isfinite(delta[0, :3]).all()),
                            worst_other_target_delta=float(delta[0, :2].mean(0)[OTHER].min()))
    out['mean_jev_consensus_targets'] = {
        t: dict(delta=float(delta[0, :2, j].mean()), **interval(delta[1:, :2, j].mean(1)))
        for j, t in enumerate(TARGETS)}
    out['references'] = {}
    for ri, ref in enumerate(REFS):
        out['references'][ref] = dict(macro=dict(delta=float(delta[0, ri].mean()),
            **interval(delta[1:, ri].mean(1))), targets={t: dict(delta=float(delta[0, ri, j]),
            **interval(delta[1:, ri, j])) for j, t in enumerate(TARGETS)})
    return out


def grid_summary(delta):
    return [dict(**c, reference_only=c['weak'] == .15,
                 versus_C4_015=summary(delta[:, i]),
                 versus_own_015=summary(delta[:, i] - delta[:, baseline(c['q'])]))
            for i, c in enumerate(CANDIDATES)]


def flat_regions(grid):
    result = {}
    for q in ('C4', 'MIL'):
        result[q] = {}
        for guarded in (False, True):
            pool = [c for c in grid if c['q'] == q and c['weak'] in FINE_WEIGHTS and
                    (not guarded or c['versus_C4_015']['harm_guard']['passed'])]
            metrics = {}
            for metric in ('primary_weak_four', 'macro_mean_jev_consensus'):
                eligible = [c for c in pool if np.isfinite(c['versus_C4_015'][metric]['delta'])]
                if not eligible:
                    metrics[metric] = dict(peak_weight=None, peak_delta=None, flat_weights=[])
                    continue
                best = max(eligible, key=lambda c: (c['versus_C4_015'][metric]['delta'], -c['weak']))
                value = best['versus_C4_015'][metric]['delta']
                metrics[metric] = dict(peak_weight=best['weak'], peak_delta=value,
                    flat_weights=[c['weak'] for c in eligible
                                  if c['versus_C4_015'][metric]['delta'] >= value-FLAT_TOLERANCE-1e-12])
            result[q]['guarded' if guarded else 'unconstrained'] = metrics
    return result


def analyze(raw, refs, groups, draws=300, progress=False):
    n = len(raw['anchor'])
    weights, draw_hash = component_weights(groups, n, draws)
    splits = nested_splits(groups)
    pol = policies()
    totals = {name: np.zeros((draws+1, 4, 12)) for name in pol}
    absolute = {name: np.zeros((4, 12)) for name in pol}
    fixed = np.zeros((draws+1, len(CANDIDATES), 4, 12))
    base_scores = np.zeros((2, 4, 12))
    records = []
    halves = []
    freq = {name: Counter() for name in pol}
    bootfreq = {name: Counter() for name in pol}
    for rep, side, inner_indices, outer_indices in splits:
        parts = [score_partition(raw, refs, ix, weights) for ix in inner_indices]
        pooled = score_partition(raw, refs, np.concatenate(inner_indices), weights)
        inner, guard_inner, repair = repair_inner_guards(parts, pooled)
        outer = score_partition(raw, refs, outer_indices, weights)
        delta_fixed = outer - outer[:, baseline('C4'), None]
        fixed += delta_fixed
        if rep == 0:
            grid = grid_summary(delta_fixed)
            halves.append(dict(name=f'half_{side}', studies=len(outer_indices),
                components=sum(bool(np.isin(g, outer_indices).all()) for g in groups),
                indices_sha256=digest_array(outer_indices.astype('<i8')),
                fixed_grid=grid, flat_regions=flat_regions(grid)))
        for qi, q in enumerate(('C4', 'MIL')):
            base_scores[qi] += outer[0, baseline(q)]
        rec = dict(repetition=rep, side=side, outer_studies=len(outer_indices),
                   inner_studies=[len(ix) for ix in inner_indices],
                   outer_indices_sha256=digest_array(outer_indices.astype('<i8')),
                   inner_indices_sha256=[digest_array(ix.astype('<i8')) for ix in inner_indices],
                   guard_repair=repair, selections={})
        for name, (allowed, base) in pol.items():
            chosen = select(inner, allowed, base, guard_inner)
            utility = selection_utility(inner, allowed, base, guard_inner)[1]
            no_eligible = ~np.isfinite(utility).any(axis=1)
            values = outer[np.arange(draws+1), chosen]
            delta = values - outer[:, base]
            totals[name] += delta
            absolute[name] += values[0]
            freq[name].update([CANDIDATES[chosen[0]]['name']])
            bootfreq[name].update(CANDIDATES[i]['name'] for i in chosen[1:])
            pp, mm, dd, gg = measures(delta[0])
            rec['selections'][name] = dict(candidate=CANDIDATES[chosen[0]]['name'],
                primary_weak_four_delta=float(pp), macro_delta=float(mm), dread_delta=float(dd),
                harm_guard=bool(gg), references=delta[0].tolist(),
                no_eligible_point=bool(no_eligible[0]),
                no_eligible_bootstrap_draws=int(no_eligible[1:].sum()))
        records.append(rec)
        if progress:
            print(f'CPU nested outer {len(records)}/10 complete', flush=True)
    results = {}
    for name in pol:
        totals[name] /= len(splits)
        results[name] = dict(**summary(totals[name]), selections=dict(freq[name]),
            bootstrap_selection_counts=dict(bootfreq[name]),
            candidate_outer_auc={ref: {t: float(absolute[name][ri, j]/10)
                for j, t in enumerate(TARGETS)} for ri, ref in enumerate(REFS)})
    fixed /= len(splits)
    fixed_results = grid_summary(fixed)
    primary = results['joint_global']
    advance = ((primary['primary_weak_four']['delta'] >= .001 or
                primary['macro_mean_jev_consensus']['delta'] >= .001) and primary['harm_guard']['passed'])
    numerical = (primary['primary_weak_four']['delta'] > 0 and
                 primary['primary_weak_four']['ci95'][0] is not None and
                 primary['primary_weak_four']['ci95'][0] > 0 and
                 primary['harm_guard']['passed'] and
                 all(v['selections']['joint_global']['harm_guard'] for v in records) and
                 primary['primary_weak_four']['valid_draws'] == draws)
    return dict(studies=n, components=len(groups), targets=list(TARGETS),
        bootstrap=dict(draws=draws, seed=SEED, component_counts_sha256=draw_hash,
            method='paired global component multiplicities; nested reselection; fixed partition ranks'),
        policies=results, fixed_grid=fixed_results, outer_splits=records,
        curve_method='mean of ten outer fixed-candidate deltas; no fitted optimum',
        flat_tolerance=FLAT_TOLERANCE, flat_regions=flat_regions(fixed_results),
        robustness=dict(seed=SEED, source='repeat zero outer component halves', halves=halves),
        baseline_outer_auc={q: {ref: {t: float(base_scores[qi, ri, j]/10)
            for j, t in enumerate(TARGETS)} for ri, ref in enumerate(REFS)} for qi, q in enumerate(('C4', 'MIL'))},
        recommendation=dict(release_v6='keep 0.15 pending trio validation',
            single_anchor_advance=bool(advance), nested_numeric_support=bool(numerical),
            outer_guard_breaches=sum(not v['selections']['joint_global']['harm_guard'] for v in records),
            next_weight_policy='global',
            reason='Post-hoc single-CoAtNet screen cannot establish the optimal weight beside the stronger release trio; Synovitis image-read evidence absent.'))


def load_inputs(root):
    from training.arms import mil_screen_replay as replay
    from tools import exp115_vlm_eval as ev
    _, ids, refs, _, metadata, bindings, _ = replay.context(root, 'mean-seed-f0')
    historical_path = root / replay.HISTORIES['mean-seed-f0']
    history = read(historical_path)
    p183 = root / 'artifacts/experiments/EXP183_MULTISCALE_5FOLD/screen/result.json'
    history183 = read(p183)
    receipt_path = (root / FILES['anchor']).with_name('oof-receipt.json')
    receipt = read(receipt_path)
    need(receipt['arm'] == 'CONTROL' and receipt['fold'] == 0 and receipt['rows'] == 888,
         'Anchor receipt differs')
    expected = dict(anchor=receipt['outputs']['oof_k94_mirror_tta.csv'],
        C4=history['input_sha256']['control_S42'], MIL=history['input_sha256']['candidate_S42'],
        **{k: history['input_sha256'][k] for k in ('R42', 'R43', 'D172')})
    need(history183['input_files']['mixed_f0']['sha256'] == expected['C4'], 'EXP183 fold-0 binding differs')
    raw, inputs = {}, {}
    for name, rel in FILES.items():
        path = root / rel
        need(sha(path) == expected[name], 'Prediction receipt hash mismatch: '+name)
        raw[name] = load_oof(path, ids)
        inputs[name] = dict(path=rel, sha256=expected[name])
    for path in (historical_path, p183, receipt_path):
        bindings[path.relative_to(root).as_posix()] = sha(path)
    return raw, refs, ev.component_groups(ids, root), dict(inputs=inputs, context=metadata,
                                                         context_input_sha256=bindings)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def render(result, result_sha):
    lines = ['# H-B-95 v1.2.0 ? POST-HOC fine global grid', '',
        '**Release v6: keep 0.15 pending the same comparison beside the stronger anchor trio.**',
        'The peak here belongs to a single-CoAtNet screen, not the final trio.', '',
        f"{result['studies']} studies / {result['components']} components; nested 5x2; "
        f"{result['bootstrap']['draws']} paired component draws, seed {SEED}.",
        'All deltas below are against C4/0.15. Main curve averages ten outer evaluations.',
        'No fitted optimum; flat sets contain tested weights within 0.0003 of each peak.', '',
        '| Q | Weight | Weak four | Macro 12 | Dread | Guard |',
        '|---|---:|---:|---:|---:|---|']
    for c in result['fixed_grid']:
        if c['reference_only']:
            continue
        v = c['versus_C4_015']
        lines.append(f"| {c['q']} | {c['weak']:.2f} | "
            f"{v['primary_weak_four']['delta']:+.6f} | {v['macro_mean_jev_consensus']['delta']:+.6f} | "
            f"{v['dread_macro']['delta']:+.6f} | {v['harm_guard']['passed']} |")
    lines += ['', '| View | Q | Macro peak / flat set | Weak-four peak / flat set |',
              '|---|---|---|---|']
    views = [('split average', result['flat_regions'])]
    views += [(f"{h['name']} ({h['studies']} studies, {h['components']} components)", h['flat_regions'])
              for h in result['robustness']['halves']]
    for name, region in views:
        for q in ('C4', 'MIL'):
            values = region[q]['unconstrained']
            cells = []
            for metric in ('macro_mean_jev_consensus', 'primary_weak_four'):
                v = values[metric]
                cells.append(f"{v['peak_weight']} / {v['flat_weights']}")
            lines.append(f"| {name} | {q} | {' | '.join(cells)} |")
    lines += ['', 'Unconstrained sets shown above; guarded sets are in result.json.',
        'Half curves are component-disjoint but contribute to the main average; they are robustness diagnostics, not new confirmation.',
        '', '| Nested selector | Weak four [95% CI] | Macro [95% CI] | Dread [95% CI] | Guard |',
        '|---|---|---|---|---|']
    for name, row in result['policies'].items():
        cells = []
        for metric in ('primary_weak_four', 'macro_mean_jev_consensus', 'dread_macro'):
            x = row[metric]
            cells.append(f"{x['delta']:+.6f} [{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}] ({x['valid_draws']} draws)")
        lines.append(f"| {name} | {' | '.join(cells)} | {row['harm_guard']['passed']} |")
    joint = result['policies']['joint_global']
    lines += ['', f"Joint selections: {joint['selections']}.",
        f"Outer guard breaches: {result['recommendation']['outer_guard_breaches']}/10.",
        f"Point no-eligible fallback: {sum(s['selections']['joint_global']['no_eligible_point'] for s in result['outer_splits'])}/10; "
        f"bootstrap no-eligible: {sum(s['selections']['joint_global']['no_eligible_bootstrap_draws'] for s in result['outer_splits'])}/{10*result['bootstrap']['draws']}.",
        'Pooled-inner repair uses outer training only; unsupported outer scores remain undefined.',
        'Every point/half, target, ruler, guard and CI is in result.json; grid.csv provides all point deltas.',
        '', 'Limits: exposed post-hoc cohort; shared ruler ancestry; aquila is diagnostic; intervals condition on fixed ranks/splits.',
        'Historical R50 seed-rank OOF reduction differs from deployed multi-fold probability-mean/flip serving.',
        'Synovitis still needs image-read evidence. No target-specific release decision follows from report rulers alone.',
        '', 'Source bumped with changelog; prior source/tests backed up under .local/backups/HB95_PARTNER_WEIGHT-v1.1.0/.',
        'v1.0.0 and v1.1.0 output files are preserved. No launch card applies: standalone CPU analysis.',
        '', f"result.json SHA256: {result_sha}",
        f"PROTOCOL.md SHA256: {result['protocol_sha256']}",
        'All deliverable/source hashes: SHA256SUMS.txt.', '', 'SUB-AGENTS: none', '']
    return '\n'.join(lines)


def write_curve_artifacts(result, out):
    """All point/target curves in CSV; split-average and fixed halves in PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    views = [('split_average', result['fixed_grid'])]
    views += [(h['name'], h['fixed_grid']) for h in result['robustness']['halves']]
    rows = []
    for view, grid in views:
        for c in grid:
            row = dict(view=view, **{k: c[k] for k in
                ('name', 'q', 'weak', 'other', 'global_weight', 'reference_only')})
            for contrast in ('versus_C4_015', 'versus_own_015'):
                ss = c[contrast]
                for metric in ('primary_weak_four', 'macro_mean_jev_consensus', 'dread_macro'):
                    row[f'{contrast}_{metric}'] = ss[metric]['delta']
                for key, value in ss['harm_guard'].items():
                    row[f'{contrast}_harm_guard_{key}'] = value
                for t in TARGETS:
                    row[f'{contrast}_mean_jev_consensus_{t}'] = ss['mean_jev_consensus_targets'][t]['delta']
                    for ref in REFS:
                        row[f'{contrast}_{ref}_{t}'] = ss['references'][ref]['targets'][t]['delta']
            rows.append(row)
    with (out/'grid.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for qi, q in enumerate(('C4', 'MIL')):
        for ax, metric, title in zip(axes[qi],
                ('primary_weak_four', 'macro_mean_jev_consensus', 'dread_macro'),
                ('Weak four', 'Macro 12', 'Dread macro')):
            for vi, (view, grid) in enumerate(views):
                rr = [c for c in grid if c['q'] == q and not c['reference_only']]
                x = [c['weak'] for c in rr]
                y = [c['versus_C4_015'][metric]['delta'] for c in rr]
                ax.plot(x, y, marker='o', linewidth=2 if vi == 0 else 1,
                        linestyle='-' if vi == 0 else '--', label=view)
                if vi == 0:
                    best = max(y)
                    ax.axhspan(best-FLAT_TOLERANCE, best, color='gray', alpha=.15)
            ax.axhline(0, color='gray', linewidth=.7)
            ax.set(title=f'{q}: {title}', xlabel='Global partner weight',
                   ylabel='AUC delta vs C4 / 0.15')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle('H-B-95 v1.2.0 POST-HOC: single CoAtNet; gray band = 0.0003 below main peak')
    fig.savefig(out/'curve.png', dpi=160)
    plt.close(fig)


def legacy_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--smoke', action='store_true', help='All paths with three bootstrap draws; no artifacts written')
    args = p.parse_args(argv)
    root = args.root.resolve()
    protocol = root / OUT / 'PROTOCOL.md'
    need(protocol.is_file(), 'Register PROTOCOL.md first')
    protocol_sha = sha(protocol)
    out = root / OUT
    if not args.smoke:
        need(not any((out/f).exists() for f in ('result.json', 'REPORT.md', 'grid.csv', 'curve.png')),
             'Create-only outputs already exist')
    started = time.time()
    raw, refs, groups, meta = load_inputs(root)
    result = analyze(raw, refs, groups, draws=3 if args.smoke else 300, progress=True)
    sources = {str(Path(__file__).relative_to(root)).replace('\\', '/'),
               'tests/test_hb95_partner_weight.py'}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(root):
            sources.add(Path(path).resolve().relative_to(root).as_posix())
    source_hashes = {path: sha(root/path) for path in sorted(sources)}
    need(sha(protocol) == protocol_sha, 'Protocol changed during run')
    for rel, digest in meta['context_input_sha256'].items():
        need(sha(root/rel) == digest, 'Evaluation input changed during run')
    for row in meta['inputs'].values():
        need(sha(root/row['path']) == row['sha256'], 'Prediction changed during run')
    result.update(schema='hb95-partner-weight-v1.2', tool_version=VERSION, post_hoc=True,
        protocol_sha256=protocol_sha, source_sha256=source_hashes, elapsed_seconds=time.time()-started,
        **meta)
    result = clean(result)
    if args.smoke:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='hb95-smoke-') as tmp:
            write_curve_artifacts(result, Path(tmp))
            render(result, 'smoke')
    else:
        write_curve_artifacts(result, out)
        with (out/'result.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        with (out/'REPORT.md').open('x', encoding='utf-8') as stream:
            stream.write(render(result, sha(out/'result.json')))
    print(json.dumps(dict(mode='SMOKE' if args.smoke else 'FULL',
        studies=result['studies'], components=result['components'], draws=result['bootstrap']['draws'],
        primary=result['policies']['joint_global']['primary_weak_four'],
        recommendation=result['recommendation'], elapsed_seconds=result['elapsed_seconds']), indent=2))
    return 0


GATE_WEIGHTS = (.15, .30, .40, .50, .60)
CONTROL43 = Path('.local/refactor/qual-4070-control-fold0-s43-8594f460d5d24d09a9f0f3f6796ab082')
MIL43 = Path('.local/refactor/mil-s43-f0-collected-v3')


def load_study_id_oof(path, ids, scratch):
    """Adapt only the registered ID header; retain strict HB95 probability loader."""
    import tempfile
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    need(rows and rows[0] == ['study_id', *TARGETS], 'CONTROL43 columns differ')
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix='adapter-') as tmp:
        dest = Path(tmp)/'predictions.csv'
        with dest.open('x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['StudyInstanceUID', *TARGETS])
            writer.writerows(rows[1:])
        return load_oof(dest, ids)


def verify_tta_mean(plain, mirror, mean):
    expected = (plain.astype(np.float32)+mirror.astype(np.float32))*np.float32(.5)
    need(np.array_equal(expected, mean.astype(np.float32)), 'CONTROL43 TTA mean differs')
    return dict(float32_mean_exact=True, max_abs_error=float(np.max(np.abs(expected-mean))))


def load_gate_inputs(root):
    from training.arms import mil_screen_replay as replay
    from tools import exp115_vlm_eval as ev
    _, ids, refs, _, metadata, bindings, _ = replay.context(root, 'mean-seed-f0')
    def bound(rel):
        rel = Path(rel).as_posix()
        bindings[rel] = sha(root/rel)
        return root/rel
    history = read(bound(replay.HISTORIES['mean-seed-f0']))
    h183 = read(bound('artifacts/experiments/EXP183_MULTISCALE_5FOLD/screen/result.json'))
    receipt = read(bound(Path(FILES['anchor']).with_name('oof-receipt.json')))
    need(receipt['arm'] == 'CONTROL' and receipt['fold'] == 0 and receipt['rows'] == 888,
         'CONTROL scope differs')
    expected = dict(anchor=receipt['outputs']['oof_k94_mirror_tta.csv'],
        C4=history['input_sha256']['control_S42'], MIL=history['input_sha256']['candidate_S42'],
        R42=history['input_sha256']['R42'], R43=history['input_sha256']['R43'])
    need(expected['C4'] == h183['input_files']['mixed_f0']['sha256'], 'C4 binding differs')
    raw, inputs = {}, {}
    def prediction(name, rel, digest=None, adapted=False):
        path = bound(rel)
        need(digest is None or sha(path) == digest, 'Prediction receipt differs: '+name)
        raw[name] = (load_study_id_oof(path, ids, root/OUT/'scratch') if adapted else load_oof(path, ids))
        inputs[name] = dict(path=Path(rel).as_posix(), sha256=sha(path), rows=len(ids))
    for name, digest in expected.items():
        prediction(name, FILES[name], digest)
    manifest = read(bound('artifacts/experiments/EXP181_COATNET_F0_BRIDGE/protocol-v1/manifest-CONTROL.json'))
    need(manifest['val_ids'] == ids and len(manifest['train_ids']) == 3455 and
         not set(ids).intersection(manifest['train_ids']), 'CONTROL held-out roster differs')
    def uid_digest(values, newline=False):
        return hashlib.sha256(('\n'.join(values)+('\n' if newline else '')).encode()).hexdigest()
    train_hash, val_hash = uid_digest(manifest['train_ids']), uid_digest(ids)
    spec = read(bound(CONTROL43/'spec.json'))
    prov = read(bound(CONTROL43/'train/outputs/provenance.json'))
    summ = read(bound(CONTROL43/'train/outputs/summary.json'))
    cfg = spec['config']
    need(spec['seed'] == 43 and spec['fold'] == 0 and cfg['targets'] == list(TARGETS) and
         cfg['readout'] == 'held-out-fold0' and cfg['factory']['crop'] == 'context', 'CONTROL43 scope differs')
    need(cfg['train_sha256'] == uid_digest(manifest['train_ids'], True) and
         cfg['val_sha256'] == uid_digest(ids, True), 'CONTROL43 train/val differs')
    need(prov['spec_sha256'] == sha(root/CONTROL43/'spec.json') and prov['scientific'] == cfg and
         prov['identity']['mode'] == 'full' and summ['status'] == 'COMPLETE_TRAINING_AND_READOUT' and
         summ['results']['checkpoint_reloaded'] and summ['results']['validation_batches'] == 888 and
         summ['bn']['studies_sha256'] == train_hash, 'CONTROL43 completion/provenance differs')
    for view in ('plain', 'mirror', 'mean'):
        prediction('A43_'+view, CONTROL43/f'train/outputs/predictions-{view}.csv', adapted=True)
    tta = verify_tta_mean(raw['A43_plain'], raw['A43_mirror'], raw['A43_mean'])
    mspec = read(bound(MIL43/'spec.json'))
    mr = read(bound(MIL43/'fold0/receipt.json'))
    need(sha(root/MIL43/'spec.json') == mr['spec_sha256'] ==
         '43fd7a50bd45b6b4c2047975b313c7bf3557424abad30a072c14ec433906f435', 'Release MIL43 spec differs')
    need(mr['status'] == 'ARTIFACTS_VERIFIED' and mr['exit_code'] == 0 and
         mspec['fold'] == 0 and mspec['seed'] == 43 and
         mspec['config']['roots']['train_sha256'] == train_hash and
         mspec['config']['roots']['val_sha256'] == val_hash, 'MIL43 held-out provenance differs')
    prediction('MIL43', MIL43/'fold0/outputs/oof_k36_flip_tta.csv', mr['outputs']['oof_k36_flip_tta.csv'])
    for name in ('R42', 'R43', 'C4', 'MIL'):
        p = read(bound(Path(FILES[name]).with_name('provenance.json')))
        need(p['train_ids_ordered_sha256'] == train_hash and p['eval_ids_ordered_sha256'] == val_hash,
             'Partner held-out provenance differs: '+name)
    for rel in ('artifacts/selection/v6-partner-weight-menu.md', 'tools/exp169_release_v6.py',
                'training/runners/exp149_label_sources.py', 'training/arms/coatnet_fit.py',
                'training/arms/coatnet.py', 'training/src/exp143_own0949.py'):
        bound(rel)
    audit = dict(anchor_tta=tta, anchor_pairs={
        'TTA': [FILES['anchor'], inputs['A43_mean']['path']],
        'plain': [str(Path(FILES['anchor']).with_name('oof_k94.csv')), inputs['A43_plain']['path']]},
        mirror_csv_is='flipped view only, not a TTA mean',
        all_members_held_out_fold=0, all_data_models_used=False,
        training_studies=3455, validation_studies=888,
        deployment_readout_exact=False,
        deployment_gap='R50 S42/S43 oof_k36.csv are single-view (EXP149 runner evaluate); release v6 uses K36 flip TTA. Other-fold models excluded to preserve honest OOF.',
        q7='equal rank families: R50 probability mean S42/S43, C4 S42, MIL probability mean S42/release-collected S43; no DINO; Q7 is release recipe name, five honest partner predictions here',
        anchor_combination='restore float32 CSV predictions, cast each seed to float64, probability mean then rank, as B6 blend')
    return ids, raw, refs, ev.component_groups(ids, root), dict(inputs=inputs, context=metadata,
        context_input_sha256=bindings, provenance_audit=audit)


def gate_predictions(raw):
    from tools import exp169_release_v6 as release
    v2, _ = release.maths()
    need(release.VERSION == '1.2.1', 'Release version differs')
    p = {k: v.astype(np.float32) for k, v in raw.items()}
    family = dict(l2=v2.rank((p['R42']+p['R43'])/2), convnext=v2.rank(p['C4']),
                  mil=v2.rank(np.stack([p['MIL'], p['MIL43']]).mean(0)))
    q = v2.rank(sum(v2.rank(x) for x in family.values())/3)
    anchors = dict(S42=p['anchor'].astype(np.float64),
                   A2=(p['anchor'].astype(np.float64)+p['A43_mean'].astype(np.float64))/2)
    pred = {a: {w: (1-w)*v2.rank(v)+w*v2.rank(q) for w in GATE_WEIGHTS} for a, v in anchors.items()}
    return pred, q


def gate_verdict(comparison, exact_readout):
    weak = comparison['primary_weak_four']['delta']
    macro = comparison['macro_mean_jev_consensus']['delta']
    numeric = bool(np.isfinite([weak, macro]).all() and weak >= 0 and macro >= 0 and
                   comparison['multi_lens']['harm_guard']['PASS'])
    return dict(numeric_pass=numeric, deployment_readout_exact=exact_readout,
        verdict='PASS' if numeric and exact_readout else 'HOLD / FALL BACK',
        reason=('Nonnegative weak/macro with guards on available honest OOF.' if numeric else
                'B0.50 fails nonnegative weak/macro or guard on two-seed anchor.') +
               ('' if exact_readout else ' Exact release Q7 readout is not established: R50 flip-TTA OOF is missing.'),
        rule='A2 B0.50 minus B0.30: weak >= 0 AND macro >= 0 AND multi-lens harm guard; no CI significance gate')


def analyze_gate(ids, raw, refs, groups, draws=300):
    from tools import multi_lens_readout as ml
    w, draw_hash = component_weights(groups, len(ids), draws)
    components = [dict(component_id=f'{j:08d}', members=[ids[i] for i in g]) for j, g in enumerate(groups)]
    ctx = ml.Context(ids, refs, components)
    need(np.array_equal(ctx.w[:draws+1], w), 'HB95/multi-lens component draws differ')
    ctx.w = w
    preds, q = gate_predictions(raw)
    rows, curves = [], {}
    for anchor, variants in preds.items():
        scores = {weight: ctx.score(p) for weight, p in variants.items()}
        # Independent implementation parity for every target/ruler, including bootstrap ties.
        for weight, p in variants.items():
            for ri, ref in enumerate(REFS):
                y, mask, minimum = refs[ref]
                for t in range(12):
                    np.testing.assert_allclose(scores[weight][:, ri, t],
                        weighted_auc(y[:, t], p[:, t], mask[:, t], minimum, w), rtol=0, atol=1e-14, equal_nan=True)
            contrasts = {}
            for base in (.30, .15):
                delta = scores[weight]-scores[base]
                contrasts[f'{base:.2f}'] = summary(delta)
                contrasts[f'{base:.2f}']['multi_lens'] = ml.evaluate(ctx,
                    {'fixed': p}, {'fixed': variants[base]})
            weak, macro, dread, _ = measures(scores[weight])
            rows.append(dict(anchor=anchor, weight=weight, absolute={
                'weak_four': float(weak[0]), 'macro12': float(macro[0]), 'dread': float(dread[0])},
                comparisons=contrasts))
        pool = [r for r in rows if r['anchor'] == anchor]
        curves[anchor] = {}
        for guarded in (False, True):
            eligible = [r for r in pool if not guarded or all(
                r['comparisons'][base]['multi_lens']['harm_guard']['PASS'] for base in ('0.30', '0.15'))]
            curves[anchor]['guarded' if guarded else 'unconstrained'] = {
                metric: max(eligible, key=lambda r: (r['absolute'][metric], -r['weight']))['weight']
                if eligible else None for metric in ('weak_four', 'macro12')}
        print(f'CPU gate {anchor}: all five weights and both baselines complete', flush=True)
    # Same-anchor weight contrasts are independent of anchor seed uplift.
    uplift = {f'{weight:.2f}': ml.evaluate(ctx, {'fixed': preds['A2'][weight]},
              {'fixed': preds['S42'][weight]}) for weight in GATE_WEIGHTS}
    return dict(studies=len(ids), components=len(groups), targets=list(TARGETS), rows=rows,
        seed_anchor_uplift=uplift, descriptive_grid_peaks=curves,
        bootstrap=dict(draws=draws, seed=SEED, component_counts_sha256=draw_hash,
                       weights_sha256=digest_array(w), method='paired component multiplicities; fixed full-fold ranks'),
        metric_parity='HB95 weighted_auc == multi_lens Context.score for every prediction/ruler/target/draw',
        q_sha256=digest_array(q), limits='Exposed fold0 only; grid peaks descriptive, no independent confirmation or public score used.')


def render_gate(result):
    lines = ['# HB95 v1.3.0 two-seed anchor gate', '', '**'+result['gate']['verdict']+'**: '+result['gate']['reason'], '',
        '888 honest fold0 studies; '+str(result['components'])+' components; '+str(result['bootstrap']['draws'])+
        ' paired draws, seed 20261003. Weak four/macro = mean JEV + consensus; Dread = its macro12.',
        'Table deltas versus the SAME anchor at weight 0.30. Both baselines, all targets, rulers and CIs: result.json / per_target.csv.', '',
        '| Anchor | w | Weak four | Macro12 | Dread | Guard |', '|---|---:|---:|---:|---:|---|']
    for row in result['rows']:
        c = row['comparisons']['0.30']
        lines.append(f"| {row['anchor']} | {row['weight']:.2f} | {c['primary_weak_four']['delta']:+.6f} | "
                     f"{c['macro_mean_jev_consensus']['delta']:+.6f} | {c['dread_macro']['delta']:+.6f} | "
                     f"{c['multi_lens']['harm_guard']['PASS']} |")
    lines += ['', 'Descriptive tested-grid peaks (not a fitted optimum): '+json.dumps(result['descriptive_grid_peaks']), '',
        'Anchor TTA pairs CONTROL/oof_k94_mirror_tta.csv with CONTROL43/train/outputs/predictions-mean.csv.',
        'CONTROL43 predictions-mirror.csv is the flipped view alone. Float32 (plain+mirror)/2 matches mean exactly.',
        'Anchor seed probability mean precedes ranking. Q7 uses release-collected MIL43, not historical EXP185 S43.',
        result['provenance_audit']['deployment_gap'],
        'Deployment percentile ranks use (average_rank-1)/(N-1), affine to HB95 rank/N; release helper used to preserve its floating-point tie behavior.',
        'Guards: macro >= -0.001, each other-eight target >= -0.010, Dread >= -0.003, all point cells finite.',
        'No CI significance threshold; multi-lens ADVANCE thresholds do not replace the requested nonnegative gate.',
        'A numeric pass alone cannot clear the exact-serving requirement. No public/model/GPU operations performed.',
        'Source backup: .local/backups/HB95_PARTNER_WEIGHT-v1.3.0/hb95_partner_weight.py. v1.0-v1.2 outputs preserved.',
        'Tests/commands: VALIDATION.md and LAUNCH.md. All hashes: SHA256SUMS.txt.', '', 'SUB-AGENTS: none', '']
    return '\n'.join(lines)


def write_gate_artifacts(result, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, text in [('result.json', json.dumps(clean(result), indent=2, allow_nan=False)+'\n'),
                       ('REPORT.md', render_gate(result))]:
        with (out/name).open('x', encoding='utf-8') as f:
            f.write(text)
    grid, targets = [], []
    for row in result['rows']:
        for base, c in row['comparisons'].items():
            shared = dict(anchor=row['anchor'], weight=row['weight'], baseline=base)
            grid.append(dict(**shared, **row['absolute'],
                weak_delta=c['primary_weak_four']['delta'], macro_delta=c['macro_mean_jev_consensus']['delta'],
                dread_delta=c['dread_macro']['delta'], guard=c['multi_lens']['harm_guard']['PASS']))
            for target in TARGETS:
                for ref, values in [('mean_jev_consensus', c['mean_jev_consensus_targets']),
                                    *[(r, c['references'][r]['targets']) for r in REFS]]:
                    v = values[target]
                    targets.append(dict(**shared, ruler=ref, target=target, delta=v['delta'],
                        ci_low=v['ci95'][0], ci_high=v['ci95'][1], valid_draws=v['valid_draws']))
    for name, rows in [('grid.csv', grid), ('per_target.csv', targets)]:
        with (out/name).open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--smoke', action='store_true', help='Real-data three-draw run, including artifact writer under v1.3.0/smoke')
    modes.add_argument('--validate', action='store_true', help='Validate all inputs and TTA/cohort contracts without scoring')
    args = p.parse_args(argv)
    root, started = args.root.resolve(), time.time()
    out = root/OUT
    protocol = out/'PROTOCOL.md'
    need(protocol.is_file(), 'Register PROTOCOL.md first')
    destination = out/'smoke' if args.smoke else out
    if not args.validate:
        need(not any((destination/f).exists() for f in ('result.json', 'REPORT.md', 'grid.csv', 'per_target.csv')),
             'Create-only outputs already exist')
    protocol_sha = sha(protocol)
    ids, raw, refs, groups, metadata = load_gate_inputs(root)
    if args.validate:
        print(json.dumps(dict(mode='VALIDATE', studies=len(ids), components=len(groups),
            provenance=metadata['provenance_audit']), indent=2))
        return 0
    result = analyze_gate(ids, raw, refs, groups, draws=3 if args.smoke else 300)
    c = next(r for r in result['rows'] if r['anchor'] == 'A2' and r['weight'] == .50)['comparisons']['0.30']
    result['gate'] = gate_verdict(c, metadata['provenance_audit']['deployment_readout_exact'])
    source_hashes = {}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(root):
            source_hashes[Path(path).resolve().relative_to(root).as_posix()] = sha(path)
    need(sha(protocol) == protocol_sha, 'Protocol changed during run')
    for rel, digest in metadata['context_input_sha256'].items():
        need(sha(root/rel) == digest, 'Input changed during run: '+rel)
    torch = sys.modules.get('torch')
    need(torch is None or not torch.cuda.is_initialized(), 'CPU analysis initialized CUDA')
    result.update(schema='hb95-partner-weight-v1.3', tool_version=VERSION, **metadata,
        source_sha256=source_hashes, protocol_sha256=protocol_sha, elapsed_seconds=time.time()-started,
        cpu_cuda_initialized=False)
    write_gate_artifacts(result, destination)
    print(json.dumps(dict(mode='SMOKE' if args.smoke else 'FULL', studies=len(ids),
        draws=result['bootstrap']['draws'], gate=result['gate'], peaks=result['descriptive_grid_peaks'],
        elapsed_seconds=result['elapsed_seconds']), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
