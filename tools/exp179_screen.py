"""EXP179 CPU screen: an array-level extension of frozen EXP170/172 helpers.

The caller validates collected-run identity, same-3090 provenance, verification
PASS, and input hashes before calling this module. No model or data job starts
at import. ``score_pairs`` accepts aligned, raw fold-0 probabilities; it never
averages ConvNeXt seed predictions before AUC. ``load_context`` is intended for
the verified EXP179 wrapper, not a standalone unverified analysis entry point.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tools.exp162_dino_folds_kaggle import TARGETS
from tools.exp170_convnext_t_l2 import need, paired_jev_consensus

VERSION = '1.0.0'
SCHEMA = 'exp179-multiscale-screen-v1'
SEEDS = (42, 43)
REFERENCES = ('jev', 'aquila', 'dread', 'consensus')
BOOTSTRAP_DRAWS = 300
BOOTSTRAP_SEED = 20261003
CAUTION_TARGETS = ('Synovitis', 'Effusion', 'Contusion', 'Fracture', 'Lateral OA')


def _predictions(values, shape=None, name='predictions'):
    a = np.asarray(values, dtype=np.float64)
    need(a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] == len(TARGETS),
         name + ': expected N x 12 probabilities')
    need(shape is None or a.shape == shape, name + ': shape differs')
    need(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(),
         name + ': nonfinite/out-of-range probabilities')
    return a


def _seed_predictions(values, shape=None, name='predictions'):
    need(isinstance(values, dict) and set(values) == set(SEEDS),
         name + ': exactly integer seeds 42 and 43 required')
    result = {}
    for seed in SEEDS:
        result[seed] = _predictions(values[seed], shape, f'{name} S{seed}')
        shape = result[seed].shape
    return result


def _references(refs, shape):
    need(set(refs) == set(REFERENCES), 'exact registered four references required')
    result = {}
    for name in REFERENCES:
        value = refs[name]
        need(isinstance(value, (tuple, list)) and len(value) == 3,
             name + ': reference requires labels, mask, minimum class count')
        y, mask, minimum = value
        y, mask = np.asarray(y, dtype=np.float64), np.asarray(mask)
        need(y.shape == mask.shape == shape and mask.dtype == np.dtype(bool),
             name + ': reference shape/boolean mask differs')
        need(isinstance(minimum, (int, np.integer)) and not isinstance(minimum, bool)
             and minimum >= 1, name + ': invalid minimum class count')
        need(np.isfinite(y[mask]).all() and np.isin(y[mask], [0., 1.]).all(),
             name + ': observed reference labels must be binary')
        result[name] = (y, mask, minimum)
    return result


def _draws(values, n):
    result = []
    for value in values:
        ix = np.asarray(value)
        need(ix.ndim == 1 and len(ix) > 0 and ix.dtype.kind in 'iu'
             and (ix >= 0).all() and (ix < n).all(),
             'bootstrap draw must contain in-range integer row indices')
        result.append(ix)
    need(len(result) > 0, 'at least one shared bootstrap draw required')
    return result


class _MeanSeedEvaluator:
    """Adapt EXP170 without editing it or reconstructing CIs from endpoints.

    Predictions have a final seed axis. The reference and mask are identical
    across seeds, so defined target AUCs have identical support. Taking their
    arithmetic mean before the unchanged target macro equals the mean of seed
    macros. EXP170 indexes each shared component draw *before* invoking this
    adapter, retaining covariance across seeds and across JEV/consensus.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def auc_targets(self, y, predictions, mask, minimum):
        need(predictions.ndim == 3 and predictions.shape[2] == len(SEEDS),
             'seed-stacked predictions required')
        values = [self.evaluator.auc_targets(y, predictions[:, :, i], mask, minimum)
                  for i in range(len(SEEDS))]
        return np.mean(values, axis=0)

    def macro(self, values):
        return self.evaluator.macro(values)


def _paired_seed_mean(candidates, controls, refs, draws, evaluator):
    return paired_jev_consensus(
        np.stack([candidates[s] for s in SEEDS], axis=2),
        np.stack([controls[s] for s in SEEDS], axis=2),
        refs, draws, _MeanSeedEvaluator(evaluator))


def _ruler_flags(pair, ruler_map):
    """Report LOW JEV cells without changing the preregistered numeric gate."""
    if ruler_map is None:
        low = set(CAUTION_TARGETS)
        source = 'agents.md per-target caution list; no ruler map supplied'
    else:
        need(ruler_map.get('schema') == 'evidence-rulers-v1', 'wrong ruler map schema')
        rows = [row for row in ruler_map.get('rows', [])
                if row.get('reference') == 'jev_hybrid']
        need(len(rows) == len(TARGETS) and
             {row['target'] for row in rows} == set(TARGETS),
             'ruler map must contain exactly twelve JEV target cells')
        need(all(row.get('validity') in ('LOW', 'MEDIUM', 'HIGH') for row in rows),
             'invalid JEV validity tier')
        low = {row['target'] for row in rows if row['validity'] == 'LOW'}
        source = 'frozen H-B-42 JEV target cells; Gold58 diagnostic, not private validity'
    losses = []
    for target in TARGETS:
        delta = pair['references']['jev']['per_target_delta'][target]
        if target in low and delta is not None and delta < 0:
            losses.append({'target': target, 'jev_delta': delta,
                           'consensus_delta': pair['references']['consensus'][
                               'per_target_delta'][target]})
    return {'status': 'REVIEW' if losses else 'NO_LOW_VALIDITY_LOSS_FLAG',
            'source': source, 'low_validity_jev_losses': losses,
            'policy': 'LOW JEV loss is not evidence of harm; inspect consensus-agreement '
                      'and eligible image-read evidence; never automatically promote a failed gate'}


def admission(direct, bundle):
    """Frozen H-B-96 numerical gate, separate from the ruler advisory."""
    dread = direct['references']['dread']['delta']
    complete = all(direct['references'][name]['per_target_delta'][target] is not None
                   and bundle['references'][name]['per_target_delta'][target] is not None
                   for name in ('jev', 'consensus') for target in TARGETS)
    checks = {
        'primary_mean_at_least_0_002': direct['primary_mean_delta'] >= .002,
        'primary_ci_lower_positive': direct['primary_mean_ci95'][0] > 0,
        'dread_at_least_minus_0_003': dread is not None and dread >= -.003,
        'fixed_bundle_primary_positive': bundle['primary_mean_delta'] > 0,
        'all_primary_point_targets_defined': complete,
    }
    return {'status': 'ADVANCE' if all(checks.values()) else 'HOLD',
            **checks,
            'rule': 'direct mean(JEV,consensus)>=0.002; paired CI95 lower>0; '
                    'direct Dread>=-0.003; fixed-bundle primary delta>0; '
                    'report bundle CI, no unregistered bundle-CI threshold'}


def score_pairs(candidates, controls, refs, draws, evaluator, rad, dino, *,
                ruler_map=None, verification_note=None):
    """Score two matched fold-0 seeds with one common component-draw roster.

    All prediction inputs are aligned raw N x 12 probabilities. ``rad`` is the
    fixed L2 RadR50 {42,43} PLAIN K36 family; ``dino`` is EXP172 L2 DINO-base S42 fold 0.
    Candidate/control ConvNeXt use K36 flip TTA. The fixed family composition
    follows the historical EXP172 admission protocol: mean Rad seed ranks,
    global family ranks, equal thirds and final bundle rerank. This is an
    OOF admission proxy, not exact deployed flip/probability-mean serving.
    This same fixed Rad family is shared by both arms. Each ConvNeXt seed
    separately replaces its matched control. The final statistic averages
    seed AUC deltas; it is not the AUC of averaged ConvNeXt predictions.

    This pure numerical API cannot certify file identity or component groups;
    the wrapper must validate them before evaluation and bind all inputs.
    """
    from tools.exp169_bundle_screen import bundle_of

    candidates = _seed_predictions(candidates, name='candidates')
    shape = candidates[42].shape
    controls = _seed_predictions(controls, shape, 'controls')
    rad = _seed_predictions(rad, shape, 'L2 RadR50')
    dino = _predictions(dino, shape, 'EXP172 L2 DINO-base')
    refs = _references(refs, shape)
    draws = _draws(draws, shape[0])
    radial = (evaluator.pct_rank(rad[42]) + evaluator.pct_rank(rad[43])) / 2.0
    dino_rank = evaluator.pct_rank(dino)
    new_bundles, old_bundles, per_seed = {}, {}, {}
    for seed in SEEDS:
        fixed = {'l2': radial, 'dino': dino_rank}
        new_bundles[seed], _ = bundle_of({**fixed,
            'convnext': evaluator.pct_rank(candidates[seed])})
        old_bundles[seed], _ = bundle_of({**fixed,
            'convnext': evaluator.pct_rank(controls[seed])})
        per_seed[str(seed)] = {
            'direct': paired_jev_consensus(candidates[seed], controls[seed], refs,
                                           draws, evaluator),
            'bundle_replacement': paired_jev_consensus(new_bundles[seed], old_bundles[seed],
                                                      refs, draws, evaluator)}
    direct = _paired_seed_mean(candidates, controls, refs, draws, evaluator)
    bundle = _paired_seed_mean(new_bundles, old_bundles, refs, draws, evaluator)
    registered = admission(direct, bundle)
    advisory = {'direct': _ruler_flags(direct, ruler_map),
                'bundle_replacement': _ruler_flags(bundle, ruler_map)}
    review = any(item['status'] == 'REVIEW' for item in advisory.values())
    return {'schema': SCHEMA, 'tool_version': VERSION,
            'verification_note': str(verification_note) if verification_note is not None else None,
            'fold': 0, 'seeds': list(SEEDS), 'studies': shape[0],
            'status': 'REVIEW' if review else registered['status'],
            'registered_gate': registered, 'ruler_advisory': advisory,
            'direct': direct, 'bundle_replacement': bundle, 'per_seed': per_seed,
            'aquila_policy': 'L2 training self-reference; report only, never a gate',
            'bootstrap': {'unit': 'validation component', 'draws': len(draws),
                          'shared_across': 'seeds, arms, references, direct and bundle',
                          'estimator': 'mean of two matched seed AUC deltas; not AUC of seed-mean predictions',
                          'seed_variability': 'two fixed seeds; CI resamples components, not seeds'},
            'aggregation': 'EXP172 admission-bundle rule: L2 RadR50 S42/S43 plain K36 seed-rank mean; '
                           'within-fold target ranks; '
                           'global family ranks; equal thirds; global bundle rerank; '
                           'separate ConvNeXt seed replacements then mean AUC delta',
            'limitations': ['exposed fold-0 V888 screen, not pooled confirmation or private-LB proof',
                            'report-derived rulers share ancestry and label errors',
                            'historical EXP172 admission differs from deployed probability-mean '
                            'flip-TTA serving; candidate and control use the same fixed family',
                            'null per-target AUC follows registered mask/minimum-class rules; '
                            'missing primary point targets prevent advancement']}


def load_context(root):
    """Load only registered fold-0 rosters/references/component groups, silently.

    No raw report or credential files are opened. Derived label arrays stay in
    memory; only aggregate hashes are returned as serializable metadata.
    The outer verified wrapper controls whether this data-access step can run.
    """
    from tools import exp115_vlm_eval as ev
    from tools import exp149_label_sources as e149
    from tools import exp156_k_sweep as e156
    from tools.exp170_convnext_t_l2 import sha, uid_hash

    root = Path(root)
    canonical, fold_of = e149.fold_rosters(e149._load_manifest('student', root), root)
    ids = [uid for uid in canonical if fold_of[uid] == 0]
    need(len(ids) == len(set(ids)) == 888, 'registered fold-0 V888 roster differs')
    raw = e156.references(ids, root)
    refs = {name: raw[key] for name, key in
            (('jev', 'jev_hybrid'), ('aquila', 'aquila_v1'),
             ('dread', 'dread'), ('consensus', 'consensus'))}
    refs = _references(refs, (len(ids), len(TARGETS)))
    groups = ev.component_groups(ids, root)
    draws = ev.bootstrap_index(groups, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    metadata = {
        'ordered_ids_sha256': uid_hash(ids),
        'components_sha256': sha(root / ev.COMPONENTS),
        'bootstrap_seed': BOOTSTRAP_SEED,
        'reference_sha256': {name: sha(root / path) for name, path in
            (('jev', e156.JEV), ('aquila', e156.AQUILA), ('dread', e156.DREAD),
             *[(f'consensus_{i}', path) for i, path in enumerate(e156.CONSENSUS)])},
        'mask_sha256': {name: hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8)
                       .tobytes()).hexdigest() for name, (_, mask, _) in refs.items()},
        'coverage': {name: {target: int(mask[:, i].sum())
                           for i, target in enumerate(TARGETS)}
                     for name, (_, mask, _) in refs.items()},
    }
    return ids, refs, draws, metadata
