"""H-B-121 v1.0.0: CPU nested stacker builder, never release admission.

Changelog 1.0.0: 5x2 component CV, same-source-fold ranks, small state
logistic maps and coarse relative family weights; reject ancestral leakage.

The in-memory API requires externally authenticated, COMPLETE transitive
exposure sets (training, teachers, selection and initialization). A hash-shaped
string or an OOF filename is not authentication. Ordinary stitched fivefold
OOF does not supply the nested FeatureBlocks this API requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from tools import multi_lens_readout as base
from tools import multi_lens_readout_l4_v2 as readout

VERSION = '1.0.0'
FAMILIES = ('R50', 'C4', 'S')
STATES = {'Synovitis': (8, 7), 'Lateral OA': (5, 4, 6)}
LAMBDAS = (.1, 1., 10.)
RELATIVE_WEIGHT_GRID = (.2, .33, .5)
TARGETS = readout.TARGETS


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


@dataclass(frozen=True)
class Ancestry:
    """Authenticated union over ALL feature-producing ancestors, not just leaf fit."""
    exposed_components: frozenset[str]
    complete: bool
    evidence_sha256: str
    oai_free: bool | None = None


@dataclass(frozen=True)
class FeatureBlock:
    rows: tuple[int, ...]
    # FP32 seed probability means, family order R50/C4/S; [rows, 3, 12].
    probabilities: np.ndarray
    source_groups: tuple[str, ...]
    ancestry: dict[str, Ancestry]


@dataclass(frozen=True)
class Split:
    train: FeatureBlock
    test: FeatureBlock


@dataclass(frozen=True)
class OuterFold:
    train: FeatureBlock
    test: FeatureBlock
    inner: tuple[Split, Split]


def probability_means(members):
    """Only arithmetic seed means; no ranking individual seeds first."""
    need(set(members) == set(FAMILIES), 'family coverage')
    result = []
    for family in FAMILIES:
        a = np.asarray(members[family])
        need(a.dtype == np.float32 and a.ndim == 3 and a.shape[0] > 0
             and a.shape[1] > 0 and a.shape[2] == 12
             and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(),
             'invalid FP32 member probabilities')
        result.append(a.mean(axis=0, dtype=np.float32))
    need(len({a.shape for a in result}) == 1, 'family row shapes differ')
    return np.stack(result, axis=1)


def block_check(block, components, forbidden=()):
    rows = block.rows
    need(bool(rows) and len(set(rows)) == len(rows)
         and all(type(i) is int and 0 <= i < len(components) for i in rows), 'block rows')
    p = np.asarray(block.probabilities)
    need(p.shape == (len(rows), 3, 12) and p.dtype == np.float32
         and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'block probabilities')
    need(len(block.source_groups) == len(rows)
         and set(block.source_groups) == set(block.ancestry), 'source group coverage')
    for group, source in block.ancestry.items():
        own = {components[i] for i, g in zip(rows, block.source_groups) if g == group}
        overlap = source.exposed_components & (set(forbidden) | own)
        need(not overlap, 'ancestral component exposure: '+str(len(overlap))+' components')
        need(source.complete is True and bool(re.fullmatch('[0-9a-f]{64}', source.evidence_sha256)),
             'incomplete transitive ancestry')


def _partition(train, test, universe, components):
    a, b = set(train.rows), set(test.rows)
    need(not a & b and a | b == universe, 'split must partition its parent rows')
    need(not {components[i] for i in a} & {components[i] for i in b}, 'component crosses split')


def validate_nested(folds, components):
    """Validate ALL ten inner and five outer splits before touching labels/fitting."""
    need(len(folds) == 5 and bool(components), 'exactly five outer folds required')
    need(all(isinstance(c, str) and c for c in components), 'component identity')
    universe = set(range(len(components)))
    seen = []
    for outer in folds:
        _partition(outer.train, outer.test, universe, components)
        heldout = {components[i] for i in outer.test.rows}
        block_check(outer.train, components, heldout)
        block_check(outer.test, components, heldout)
        need(len(outer.inner) == 2, 'exactly two inner folds required')
        inner_seen = []
        for inner in outer.inner:
            _partition(inner.train, inner.test, set(outer.train.rows), components)
            excluded = heldout | {components[i] for i in inner.test.rows}
            block_check(inner.train, components, excluded)
            block_check(inner.test, components, excluded)
            inner_seen.extend(inner.test.rows)
        need(len(inner_seen) == len(set(inner_seen))
             and set(inner_seen) == set(outer.train.rows), 'inner test coverage')
        seen.extend(outer.test.rows)
    need(len(seen) == len(set(seen)) and set(seen) == universe, 'outer test coverage')


def same_fold_rank(values, groups):
    """Average ties, (rank-1)/(n-1); singleton=.5. No pooled-fold ranks."""
    a = np.asarray(values, dtype=np.float64)
    need(a.ndim >= 2 and len(a) == len(groups) and len(a) > 0
         and np.isfinite(a).all(), 'rank shape/nonfinite')
    result = np.empty_like(a)
    gg = np.asarray(groups)
    for group in dict.fromkeys(groups):
        ix = np.flatnonzero(gg == group)
        result[ix] = ((rankdata(a[ix], axis=0, method='average') - 1) / (len(ix)-1)
                      if len(ix) > 1 else .5)
    return result


def features(block):
    family = same_fold_rank(block.probabilities, block.source_groups)
    # Cross-target features and control use the identical Q7S rank-mean scale.
    control = same_fold_rank(family.mean(axis=1), block.source_groups)
    return family, control


def weight_grid():
    # P8's .2/.33/.5 are relative weights: no triple of them sums to 1.
    values = [np.full(3, 1/3)]
    for triple in product(RELATIVE_WEIGHT_GRID, repeat=3):
        if len(set(triple)) == 1:
            continue
        w = np.asarray(triple) / sum(triple)
        if not any(np.allclose(w, prior, rtol=0, atol=1e-15) for prior in values):
            values.append(w)
    return np.asarray(values)


def _refs_check(refs, n):
    need(set(refs) == set(readout.RULERS), 'four named rulers required')
    for y, mask, minimum in refs.values():
        need(y.shape == mask.shape == (n, 12) and mask.dtype == bool
             and np.isfinite(y[mask]).all() and np.isin(y[mask], [0, 1]).all()
             and isinstance(minimum, int) and minimum >= 1, 'ruler values/masks')


def fit_state(x, y, mask, penalty):
    """2/3 slopes plus intercept = 3/4 fitted parameters; Dread labels only."""
    need(x.ndim == 2 and x.shape[1] in (2, 3) and len(x) == len(y) == len(mask)
         and np.isfinite(x).all() and mask.dtype == bool
         and penalty in LAMBDAS, 'state inputs')
    x, y = x[mask], y[mask]
    need(np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2, 'state fit needs both classes')
    xx = np.c_[np.ones(len(x)), x]
    def objective(beta):
        z = xx @ beta
        loss = (np.logaddexp(0, z) - y*z).sum() + .5*penalty*np.dot(beta[1:], beta[1:])
        grad = xx.T @ (expit(z)-y) + penalty*np.r_[0., beta[1:]]
        return loss / len(x), grad / len(x)
    fit = minimize(objective, np.zeros(xx.shape[1]), method='L-BFGS-B', jac=True,
                   options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
    need(fit.success and np.isfinite(fit.x).all(), 'state optimizer failed')
    return fit.x


def _predict_state(control, indices, beta):
    return expit(beta[0] + control[:, indices] @ beta[1:])


def _target_score(pred, rows, refs, target):
    scores = []
    for ruler in ('jev', 'consensus'):
        y, mask, minimum = refs[ruler]
        ix = np.asarray(rows)
        score = base.auc_draws(pred[:, None], y[ix, target, None],
                              mask[ix, target, None], np.ones((1, len(rows))), minimum)[0, 0]
        need(np.isfinite(score), 'inner target lacks fixed class support')
        scores.append(score)
    return float(np.mean(scores))


def nested_crossfit(folds, components, refs):
    """Return honest nested predictions only if every genealogy check passes.

    refs follows multi_lens_readout.Context (ruler -> y, mask, min_class).
    Tune logistic L2 on two equally weighted inner validation AUCs, refit on
    outer train. Choose relative family weights by the same inner objective;
    weights are constants, requiring no additional coefficient fit. Deterministic
    ties use grid order, equal thirds first. Outer labels never select anything.
    """
    validate_nested(folds, components)
    _refs_check(refs, len(components))
    y, mask, _ = refs['dread']
    names = ('Synovitis', 'Lateral OA', 'state_both', 'family_weights')
    arms = {name: np.full((len(components), 12), np.nan) for name in names}
    control = np.full((len(components), 12), np.nan)
    fitted = []
    grid = weight_grid()
    for outer in folds:
        inner = [(s, *features(s.train), *features(s.test)) for s in outer.inner]
        _, train_control = features(outer.train)
        test_family, test_control = features(outer.test)
        ti, oi = np.asarray(outer.train.rows), np.asarray(outer.test.rows)
        control[oi] = test_control
        for name in names:
            arms[name][oi] = test_control
        constants = {'state': {}, 'family_weights': {}}
        for name, indices in STATES.items():
            target = indices[0]
            cv = []
            for penalty in LAMBDAS:
                scores = []
                for split, _, tr, _, va in inner:
                    ix = np.asarray(split.train.rows)
                    beta = fit_state(tr[:, indices], y[ix, target], mask[ix, target], penalty)
                    scores.append(_target_score(_predict_state(va, indices, beta),
                                                split.test.rows, refs, target))
                cv.append(float(np.mean(scores)))
            chosen = int(np.argmax(cv))
            beta = fit_state(train_control[:, indices], y[ti, target], mask[ti, target], LAMBDAS[chosen])
            prediction = same_fold_rank(_predict_state(test_control, indices, beta)[:, None],
                                        outer.test.source_groups)[:, 0]
            arms[name][oi, target] = arms['state_both'][oi, target] = prediction
            constants['state'][name] = {'features': [TARGETS[j] for j in indices],
                'coefficients_intercept_first': beta.tolist(), 'parameters': len(beta),
                'penalty': LAMBDAS[chosen], 'inner_scores': cv}
        for target, name in enumerate(TARGETS):
            cv = [np.mean([_target_score(va[:, :, target] @ w, split.test.rows, refs, target)
                           for split, _, _, va, _ in inner]) for w in grid]
            chosen = int(np.argmax(cv))
            prediction = same_fold_rank((test_family[:, :, target] @ grid[chosen])[:, None],
                                        outer.test.source_groups)[:, 0]
            arms['family_weights'][oi, target] = prediction
            constants['family_weights'][name] = {'weights': grid[chosen].tolist(),
                'parameters': 2, 'inner_score': float(cv[chosen])}
        fitted.append(constants)
    need(np.isfinite(control).all() and all(np.isfinite(a).all() for a in arms.values()), 'incomplete predictions')
    return {'version': VERSION, 'arms': arms, 'control': control, 'outer_constants': fitted,
            'ship_constants': None, 'scope': 'NESTED_OOF_ONLY; outer fits are not deployment constants'}


def image_read_entry(ids, arm, control, labels, mask, components, ancestry, *, gold58=False):
    """An OAI entry is admitted only with complete, explicitly OAI-free ancestry.

    Include ancestry of both feature generators AND the fitted second stage.
    Gold58 is always diagnostic. Input authentication is the caller's obligation.
    """
    eligible = bool(ancestry) and all(s.complete is True and s.oai_free is True
                                   and re.fullmatch('[0-9a-f]{64}', s.evidence_sha256)
                                   for s in ancestry)
    shape = (len(ids), 12)
    need(len(set(ids)) == len(ids) > 0 and all(a.shape == shape for a in (arm, control, labels, mask))
         and np.isfinite(arm).all() and np.isfinite(control).all() and mask.dtype == bool
         and np.isin(labels[mask], [0, 1]).all(), 'image-read shape/values')
    covered = [u for c in components for u in c['members'] if u in set(ids)]
    need(len(covered) == len(set(covered)) == len(ids) and set(covered) == set(ids),
         'image-read component coverage')
    panel_components = {c['component_id'] for c in components if set(c['members']) & set(ids)}
    eligible = eligible and all(not s.exposed_components & panel_components for s in ancestry)
    need(gold58 or eligible, 'OAI-free transitive ancestry unproven')
    return dict(ids=ids, arm=arm, control=control, labels=labels, mask=mask,
                components=components, eligible=eligible and not gold58,
                component_policy='supplied patient components; Gold58 diagnostic only' if gold58
                                 else 'supplied OAI patient components')


def seven_lenses(result, ids, refs, components, image_reads=None):
    """Corrected H119 API, preserving masks, AUC and conditional component CIs.

    image_reads: optional per-arm mapping from image_read_entry(). No synthetic
    seeds: outer folds are not seeds. No L6 claim without the partner bundle.
    """
    readout.authenticate_base()
    need(result.get('scope', '').startswith('NESTED_OOF_ONLY'), 'nested results required')
    ctx = base.Context(ids, refs, components)
    return {name: readout.evaluate(ctx, {'nested_5x2': arm}, {'nested_5x2': result['control']},
                                  image_reads=(image_reads or {}).get(name))
            for name, arm in result['arms'].items()}


def audit_aligned():
    """Authenticate the named real inputs, exercise preprocessing, reject leakage.

    No labels parsed, model/checkpoint loaded, fit, metric or bootstrap performed.
    Only the fixed H121 result.json is written; inherited run/freeze not invoked.
    """
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    import time

    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.local/scratch/sol-4/hb121'
    source = 'tools/hb105_aligned_oof_v1.py'
    manifest = 'artifacts/experiments/HB105_ALIGNED_OOF_V1/input-source-manifest.json'
    saved = '.local/scratch/sol-2/hb105-aligned-readout/readout.json'
    panel_path = '.local/scratch/sol-2/q7s-panel/paired-l7-v1.1.json'
    recipes_path = '.local/scratch/sol-2/q7s-panel/recipes.json'
    expected = {
        source: 'a06425ee1f2b1c592e506a2760f5da84d38e007a2e88e0a25d3106dbf6163fff',
        manifest: 'f439d836e46eeac920146cfdf3d1b180a10075a61b3c7fb2b1d3e5de5eb35806',
        saved: '401573a9aa76783ba351ff9f244506adc6ff4ab76cfc1efad3bf4961c27e0825',
        panel_path: 'a12b2d5d6c50495a3ea3684e90d77b8bbe3ed78048cc378f49a33df1d94e088a',
        recipes_path: '40b5e261a875405b5598482706ecfb1fb0e97d0793e51a9fa2359482ed8f4d3f',
        'tools/multi_lens_readout_l4_v2.py': '6fd4d8946906bf4571e2a01871d4fee5de91ec754d8f02318de45a0669efaeb7',
        'tools/multi_lens_readout.py': readout.BASE_SHA256,
    }
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def check_pins():
        for rel, sha in expected.items():
            need(digest(root/rel) == sha, 'H121 input/source pin changed: '+rel)
    check_pins()
    from tools import hb105_aligned_oof_v1 as aligned
    raw, rosters, frozen, manifest_pin = aligned.load_frozen()
    prior = json.loads((root/saved).read_text())
    need(prior['status'] == 'COMPLETE_OOF_SURROGATE_ONLY'
         and prior['input_manifest'] == manifest_pin, 'prior aligned readout identity')
    panel = json.loads((root/panel_path).read_text())
    recipes = json.loads((root/recipes_path).read_text())
    need(panel['input_manifest_sha256'] == expected[manifest]
         and panel['recipes_sha256'] == expected[recipes_path]
         and recipes['manifest_sha256'] == expected[manifest], 'panel recipe identity')
    ids = sum((r['val'] for r in rosters), [])
    idset = set(ids)
    component_rows = json.loads((root/aligned.ml.COMPONENTS).read_text())
    membership = {}
    for group in component_rows:
        for uid in group['members']:
            if uid in idset:
                need(uid not in membership, 'duplicate component membership')
                membership[uid] = group['component_id']
    need(set(membership) == idset, 'component coverage differs')
    components = tuple(membership[u] for u in ids)
    q7s = [r['member'] for r in frozen['members'] if r['member']['family'] in FAMILIES]
    need(len(q7s) == 25, 'Q7S member coverage')
    blocks, offset = [], 0
    for f, roster in enumerate(rosters):
        family_probs = probability_means({k: np.stack([raw[f][s] for s in aligned.FAMILIES[k]])
                                          for k in FAMILIES})
        rows = tuple(range(offset, offset+len(roster['val'])))
        group = 'base_fold_'+str(f)
        # The direct exposure is proven. Do NOT falsely certify unknown ancestors.
        ancestry = Ancestry(frozenset(membership[u] for u in roster['train']), False, expected[manifest])
        blocks.append(FeatureBlock(rows, family_probs, (group,)*len(rows), {group: ancestry}))
        offset += len(rows)
    # Preprocess the actual saved predictions. No fitted arm or invented labels.
    smoke = [features(b) for b in blocks]
    control = np.concatenate([c for _, c in smoke])
    need(control.shape == (4343, 12) and np.isfinite(control).all(), 'real preprocessing smoke failed')
    audits = []
    for f, test in enumerate(blocks):
        train_blocks = [b for k,b in enumerate(blocks) if k != f]
        train = FeatureBlock(tuple(i for b in train_blocks for i in b.rows),
                             np.concatenate([b.probabilities for b in train_blocks]),
                             tuple(g for b in train_blocks for g in b.source_groups),
                             {g:s for b in train_blocks for g,s in b.ancestry.items()})
        heldout = {components[i] for i in test.rows}
        _partition(train, test, set(range(len(ids))), components)
        try:
            block_check(train, components, heldout)
        except ValueError as exc:
            need(str(exc).startswith('ancestral component exposure:'), 'unexpected preflight rejection')
            rejection = str(exc)
        else:
            raise AssertionError('ordinary stitched OOF was admitted as nested')
        counts = [len(set(rosters[e['fold']]['train']) & set(rosters[f]['val']))
                  for e in q7s if e['fold'] != f]
        need(len(counts) == 20 and counts == [len(test.rows)]*20, 'direct ancestry audit differs')
        audits.append(dict(outer=f, test_rows=len(test.rows), test_components=len(heldout),
                           fit_rows=len(train.rows), leaking_feature_generators=20,
                           test_rows_seen_by_every_fit_generator=len(test.rows),
                           fit_rows_with_ancestral_exposure=len(train.rows),
                           component_crossings=0, production_gate_rejection=rejection))
    eligible = []
    unproven = []
    for e in q7s:
        recipe = recipes['recipes'][e['id']]
        need(recipe['sha256'] == e['sha256'] and recipe['bundle_family'] == e['family'],
             'panel member identity differs')
        # Frozen adapter provides a leaf-roster/init-allowlist claim, not the
        # complete transitive exclusion demanded by D-B-157's correction.
        need(set(recipe['ancestry']) == {'oai_trained', 'basis'}, 'new ancestry evidence requires review')
        unproven.append(e['id'])
    check_pins()
    import torch
    need(not torch.cuda._initialized and torch.get_num_threads() <= 4, 'CPU contract')
    reasons = {
        'L1': 'No admissible nested arm: outer-test exposure in all meta-fit generators',
        'L2': 'No admissible nested arm: outer-test exposure in all meta-fit generators',
        'L3': 'No admissible nested arm: outer-test exposure in all meta-fit generators',
        'L4': 'No admissible real delta tensor; corrected H119 rule was not invoked on fabricated deltas',
        'L5': 'No admissible paired stacker seed fits; folds are not seeds',
        'L6': 'No admissible arm or authenticated heldout own-anchor comparison',
        'L7': 'No admissible fitted arm; no proven complete OAI-free Q7S ancestry in supplied evidence',
    }
    result = dict(task='H-B-121', version=VERSION, status='BLOCKED_NESTED_ANCESTRY',
        suggestion='HOLD', ship_constants=None, launch_authorized=False,
        rows=len(ids), components=len(set(components)), q7s_members=25,
        input_manifest=manifest_pin, pins=expected,
        code_pins={p:digest(root/p) for p in ('tools/hb121_state_stackers_v1.py',
                                            'tests/test_hb121_state_stackers_v1.py')},
        authenticated_h105_bindings=len(frozen['bindings']),
        authenticated_h105_sources=len(frozen['sources']),
        outer_ancestry_audit=audits,
        cpu=dict(cuda_initialized=False, torch_threads=torch.get_num_threads(),
                 numeric_thread_env={k:os.environ.get(k) for k in
                                    ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
                 python=sys.version.split()[0], seconds=round(time.perf_counter()-start,3)),
        real_smoke=dict(status='PASS_PREPROCESSING_AND_REJECTION',
                        probability_mean_then_same_fold_rank=True, finite_control=True,
                        control_sha256=hashlib.sha256(control.astype('<f8').tobytes()).hexdigest(),
                        state_fit='NOT_RUN: ancestry gate rejects before fitting or label parsing',
                        family_weight_selection='NOT_RUN: same ancestral exposure'),
        lenses={k:dict(status='NOT_EVALUABLE',value=None,ci95=None,PASS=False,eligible=False,reason=v)
                for k,v in reasons.items()},
        target_deltas={name:{t:dict(value=None,ci95=None,reason='Honest nested arm unavailable') for t in TARGETS}
                       for name in ('Synovitis','Lateral OA','state_both','family_weights')},
        oai=dict(panel_rows=panel['rows'], eligible_members=eligible, unproven_members=sorted(unproven),
                 reason='Competition-only leaf rosters and initializer allowlists do not prove full ancestor exclusion; D-B-157 correction',
                 rerun=False),
        gold58_synovitis=dict(status='NOT_EVALUABLE',value=None,ci95=None,
                             reason='No admissible fitted arm; H105 manifest supplies no Gold58 paired predictions',
                             diagnostic_only=True),
        policy=dict(families=list(FAMILIES),control='equal family rank mean, same source-fold ranks',
                    inner_objective='mean(JEV,consensus) target AUC, equal mean over two inner validations',
                    state_fit='Dread binary/masked, 3 parameters Synovitis / 4 Lateral OA, L2 grid [.1,1,10]',
                    family_grid='normalize relative weights {.2,.33,.5}; 25 unique triples, equal-thirds first',
                    ci='Inherited paired 300-draw component percentile CI; conditional on fitted predictions, not full refit uncertainty',
                    l4='Exact >=.003 on >=3/4 named ruler point deltas; inherited mean CI is not a CI of this rule'),
        next_dependency='Authenticated outer-excluded fit features and both inner split feature blocks, including every ancestor; complete OAI-free evidence and paired panel/Gold58 predictions for an admissible fit',
        limitations=['No nested real screen, fit, AUC, bootstrap, model load or GPU operation performed',
                     'Synthetic tests exercise the full engine; preprocessing/rejection only were exercised on real inputs',
                     'No automatic production adoption or deployment constants',
                     'D-B-003/037 require pipeline-level exclusion; ordinary OOF plus meta-CV is insufficient'])
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return dict(status=result['status'], suggestion='HOLD', rows=len(ids), members=25,
                leaking_generators_per_outer=[a['leaking_feature_generators'] for a in audits],
                real_smoke=result['real_smoke']['status'], oai_eligible_members=eligible,
                result='.local/scratch/sol-4/hb121/result.json', ship_constants=None)


def main():
    import argparse
    import json
    import os
    # Run with numerical thread env set before Python imports this module.
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        need(os.environ.get(key) in ('1','2','3','4'), 'set CPU thread environment before launch: '+key)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit',))
    parser.parse_args()
    print(json.dumps(audit_aligned(), indent=2))


if __name__ == '__main__':
    main()
