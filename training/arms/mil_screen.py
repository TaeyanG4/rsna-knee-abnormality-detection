"""MIL CPU screen equations v1.0.0 (2026-10-11).

Changelog: explicit numerical functions preserve EXP185 mean-seed and EXP186B
pooled-fold estimands, weak-target guards, rank boundaries and shared draws.
No generated runner, module delegation, model launch or timing gate.
"""
from __future__ import annotations

import numpy as np
from training.arms.lifecycle import ContractError
from tools import exp169_bundle_screen as bundle
from tools import exp179_screen as validation
from tools.exp170_convnext_t_l2 import paired_jev_consensus

VERSION = "1.0.0"


def need(condition, message):
    if not condition:
        raise ContractError(message)


def weak_target_readout(candidate, control, refs, draws, evaluator):
    """Pure paired primary and harm guard; masks/draws already frozen by caller."""
    import numpy as np
    from tools.exp162_dino_folds_kaggle import TARGETS

    weak = ('Lateral Meniscus', 'PF OA', 'Lateral OA', 'Synovitis')
    other = [t for t in TARGETS if t not in weak]
    names = list(TARGETS)

    def values(ix):
        per = {}
        macros = {}
        for name in ('jev', 'consensus', 'dread', 'aquila'):
            y, mask, minimum = refs[name]
            a = evaluator.auc_targets(y[ix], candidate[ix], mask[ix], minimum)
            b = evaluator.auc_targets(y[ix], control[ix], mask[ix], minimum)
            per[name] = a - b
            macros[name] = evaluator.macro(a) - evaluator.macro(b)
        avg = (per['jev'] + per['consensus']) / 2
        primary = float(np.mean([avg[names.index(t)] for t in weak]))
        macro = float((macros['jev'] + macros['consensus']) / 2)
        return per, macros, avg, primary, macro

    per, macros, avg, primary, macro = values(np.arange(len(candidate)))
    boot = np.asarray([values(ix)[3] for ix in draws], dtype=np.float64)
    need(np.isfinite(primary) and np.isfinite(macro) and np.isfinite(boot).all(),
         'undefined weak-target primary/bootstrap')
    guard = {'macro12_mean_jev_consensus_ge_minus_0_001': macro >= -.001,
             'other_targets_ge_minus_0_010': all(np.isfinite(avg[names.index(t)]) and
                                                    avg[names.index(t)] >= -.010 for t in other),
             'dread_macro12_ge_minus_0_003': np.isfinite(macros['dread']) and
                                                 macros['dread'] >= -.003}
    guard = {name: bool(value) for name, value in guard.items()}
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    numeric = ('HOLD_REVIEW' if not all(guard.values()) else
               'ADVANCE' if primary > 0 and ci[0] > 0 else 'HOLD')
    # Report-ruler Synovitis cannot by itself qualify the candidate. Gold58
    # is exposed and report-based; an independent image-read check is needed.
    status = 'REVIEW_IMAGE_READ' if numeric == 'ADVANCE' else numeric
    return {'primary_weak4_mean_delta': primary, 'primary_ci95': ci,
            'macro12_mean_jev_consensus_delta': macro,
            'per_target_mean_jev_consensus_delta': {t: float(avg[i]) if np.isfinite(avg[i]) else None
                                                    for i, t in enumerate(TARGETS)},
            'per_reference_macro_delta': {k: float(v) if np.isfinite(v) else None
                                          for k, v in macros.items()},
            'harm_guard': guard, 'numeric_gate': numeric, 'status': status,
            'synovitis_policy': 'report ruler deltas; image-read evidence needed for decision',
            'gold58': 'report-only; no new Gold58 inference in fold-0 screen',
            'aquila': 'report-only; L2 label self-reference'}


def _score_folds(fold_values, refs, draws, evaluator):
    import numpy as np
    from tools import exp169_bundle_screen as bundle
    from tools.exp162_dino_folds_kaggle import TARGETS
    need(set(fold_values) == set(range(5)), 'five folds required')
    direct_new, direct_old, new_bundles, old_bundles = [], [], [], []
    per_fold = {}
    offset = 0
    for fold in range(5):
        row = fold_values[fold]
        need(set(row) == {'candidate', 'control', 'rad42', 'rad43', 'dino'},
             'wrong fixed family inputs')
        shapes = {np.asarray(v).shape for v in row.values()}
        need(len(shapes) == 1 and next(iter(shapes))[1] == 12,
             'fold shape differs')
        length = len(row['candidate'])
        radial = (evaluator.pct_rank(row['rad42']) +
                  evaluator.pct_rank(row['rad43'])) / 2
        fixed = {'l2': radial, 'dino': evaluator.pct_rank(row['dino'])}
        new_bundles.append({**fixed, 'convnext': evaluator.pct_rank(row['candidate'])})
        old_bundles.append({**fixed, 'convnext': evaluator.pct_rank(row['control'])})
        direct_new.append(row['candidate'])
        direct_old.append(row['control'])
        segment = slice(offset, offset + length)
        per_fold[str(fold)] = {}
        for name, (y, mask, minimum) in refs.items():
            a = evaluator.auc_targets(y[segment], row['candidate'], mask[segment], minimum)
            b = evaluator.auc_targets(y[segment], row['control'], mask[segment], minimum)
            macro_delta = evaluator.macro(a)-evaluator.macro(b)
            per_fold[str(fold)][name] = {
                'macro_delta': float(macro_delta) if np.isfinite(macro_delta) else None,
                'per_target_delta': {t: float(v) if np.isfinite(v) else None
                                     for t, v in zip(TARGETS, a-b)}}
        offset += length
    need(offset == next(iter(refs.values()))[0].shape[0], 'pooled reference length differs')
    candidate, control = np.concatenate(direct_new), np.concatenate(direct_old)
    fixed_new = {name: np.concatenate([row[name] for row in new_bundles])
                 for name in ('l2', 'dino', 'convnext')}
    fixed_old = {name: np.concatenate([row[name] for row in old_bundles])
                 for name in ('l2', 'dino', 'convnext')}
    candidate_bundle, _ = bundle.bundle_of(fixed_new)
    control_bundle, _ = bundle.bundle_of(fixed_old)
    direct = paired_jev_consensus(candidate, control, refs, draws, evaluator)
    try:
        weak = weak_target_readout(candidate, control, refs, draws, evaluator)
    except ValueError as error:
        if str(error) != 'undefined weak-target primary/bootstrap':
            raise
        weak = {'primary_weak4_mean_delta': None, 'primary_ci95': None,
                'numeric_gate': 'HOLD_REVIEW', 'status': 'HOLD_REVIEW',
                'reason': str(error)}
    replacement = paired_jev_consensus(candidate_bundle, control_bundle,
                                             refs, draws, evaluator)
    return {'direct': direct, 'weak_target_readout': weak,
            'fixed_bundle_replacement_diagnostic': replacement,
            'per_fold_direct': per_fold, 'status': weak['status']}


def score_folds(values, refs, draws, evaluator):
    """Five disjoint folds; raw direct AUC, within-fold ranks then pooled bundle."""
    need(type(values) is dict and set(values) == set(range(5)), 'five folds required')
    normalized = {}
    for fold, row in values.items():
        need(type(row) is dict and set(row) == {'candidate', 'control', 'rad42', 'rad43', 'dino'},
             'wrong fixed family inputs')
        candidate = validation._predictions(row['candidate'])
        normalized[fold] = {name: validation._predictions(value, candidate.shape)
                            for name, value in row.items()}
    length = sum(len(row['candidate']) for row in normalized.values())
    refs = validation._references(refs, (length, 12))
    draws = validation._draws(draws, length)
    return _score_folds(normalized, refs, draws, evaluator)


def score_seeds(candidates, controls, rad, dino, refs, draws, evaluator):
    """Fold zero: mean of matched seed AUCs, never AUC of averaged predictions."""
    candidates = validation._seed_predictions(candidates)
    shape = candidates[42].shape
    controls = validation._seed_predictions(controls, shape)
    rad = validation._seed_predictions(rad, shape)
    dino = validation._predictions(dino, shape)
    refs = validation._references(refs, shape)
    draws = validation._draws(draws, shape[0])
    candidate = np.stack([candidates[s] for s in (42, 43)], axis=2)
    control = np.stack([controls[s] for s in (42, 43)], axis=2)
    mean = validation._MeanSeedEvaluator(evaluator)
    details = paired_jev_consensus(candidate, control, refs, draws, mean)
    weak = weak_target_readout(candidate, control, refs, draws, mean)
    radial = (evaluator.pct_rank(rad[42]) + evaluator.pct_rank(rad[43])) / 2
    fixed = {'l2': radial, 'dino': evaluator.pct_rank(dino)}
    new, old = {}, {}
    for seed in (42, 43):
        new[seed], _ = bundle.bundle_of({**fixed, 'convnext': evaluator.pct_rank(candidates[seed])})
        old[seed], _ = bundle.bundle_of({**fixed, 'convnext': evaluator.pct_rank(controls[seed])})
    replacement = validation._paired_seed_mean(new, old, refs, draws, evaluator)
    return {'status': weak['status'], 'weak_target_readout': weak,
            'all_targets_and_references': details,
            'per_seed': {str(seed): paired_jev_consensus(candidates[seed], controls[seed], refs,
                                                       draws, evaluator) for seed in (42, 43)},
            'fixed_bundle_replacement_diagnostic': replacement}
