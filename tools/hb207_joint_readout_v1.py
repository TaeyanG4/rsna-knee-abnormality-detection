"""H207 fixed four-arm exposed OOF readout v1.0.0.

Changelog 1.0.0: pinned existing loaders, exact nested Q rank tail, three
registered contrasts, create-only outputs and an independent source-review gate.
No model execution, checkpoint reads, release qualification or exact-F fallback.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb199_inf2_bridge_v1 as inf2  # audited CPU/checkpoint tripwires
from tools import hb198_c4_bridge_v1 as c4
import numpy as np

ml, aligned = inf2.ml, inf2.aligned
VERSION = '1.0.0'
BUILDER = 'native-astra-r1'
CANDIDATE = 'R6W958-C4INF2-V1'
WORK = ROOT / '.local/scratch/slot-b/research-resume-1018/h207-build'
OUT = WORK / 'analysis-v1'
REG = WORK / 'JOINT-REGISTRATION.json'
REG_SHA = 'f216f73cebe6a211ba24cd73071e5650c3c500b4c83c66b42647229c38704516'
ARMS = ('Q0', 'Q4', 'Q2', 'Q24')
CONTRASTS = ('Q24-Q0', 'Q24-Q4', 'Q24-Q2')
KEYS = {'R42', 'R43', 'DINO', 'CNX', 'C4', 'S42', 'S43'}
RULERS = ('jev', 'consensus', 'dread', 'aquila')
MIN_CLASS = {'jev': 1, 'consensus': 2, 'dread': 1, 'aquila': 1}
HISTORIES = {
    'c4': '.local/scratch/slot-b/batch5-1018/c4-bridge/OOF-PROXY.json',
    'inf2': '.local/scratch/slot-b/batch5-1018/inf2-bridge/results.json',
}
need, read, pin, rel = inf2.need, inf2.read, inf2.pin, inf2.rel


def authenticated_registration():
    need(pin(REG)['sha256'] == REG_SHA, 'joint registration identity')
    reg = read(REG)
    need(reg['candidate'] == CANDIDATE and reg['arms'] == list(ARMS)
         and reg['contrasts'] == list(CONTRASTS), 'registered design')
    need(reg['targets'] == list(ml.TARGETS) and tuple(ml.RULERS) == RULERS
         and ml.WEAK == (3, 5, 6, 8) and (ml.DRAWS, ml.SEED) == (300, 20261003),
         'target/ruler/bootstrap policy')
    for p, expected in {**reg['sources'], **reg['input_registration_refs']}.items():
        need(pin(ROOT / p) == expected, 'registered source or metadata drift: ' + p)
    return reg


def operators(reg):
    """Only pure rank, aggregate's final expression, and registered point rule.

    OOF seed pairs are formed row-heldout below. Never emulate deployment by
    passing repeated heldout arrays through the all-five-fold serving mean.
    """
    path = WORK / 'candidate-runtime/exp169_partner_bundle_release.py'
    need(pin(path)['sha256'] == reg['operators'][path.name], 'Q operator identity')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fun = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    aggregate = fun['aggregate']
    tail = [n for n in aggregate.body if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == 'bundle']
    need(len(tail) == 1, 'aggregate bundle expression')
    expected = ast.parse('bundle = rank(sum(rank(x) for x in family.values()) / 3)').body[0]
    need(ast.dump(tail[0]) == ast.dump(expected), 'nested rank expression changed')
    wrapper = ast.parse('def bundle(family):\n    return None').body[0]
    wrapper.body = [ast.Return(value=tail[0].value)]
    env = dict(need=need)
    module = ast.fix_missing_locations(ast.Module(body=[fun['rank'], wrapper], type_ignores=[]))
    exec(compile(module, '<pinned-Q-rank-and-bundle-tail>', 'exec'), env)
    source = (ROOT / 'tools/c85r7_r6w958c4inf2_v1149.py').read_text(encoding='utf-8')
    inf2.extracted_functions(source, ('decision_readout',), env)
    return env


def prepare():
    """Source/registration inspection and hash-only historical pins; no arrays."""
    reg = authenticated_registration()
    operators(reg)
    sources = dict(aligned.source_pins())
    # Historical closures contain script pins outside the normal imported tools.
    for registration in reg['input_registration_refs']:
        for p, value in read(ROOT / registration)['bindings'].items():
            if p.endswith('.py'):
                actual = pin(ROOT / p)
                need(actual['sha256'] == (value['sha256'] if isinstance(value, dict) else value),
                     'inherited source drift: ' + p)
                sources[p] = actual
    for p in (Path(__file__), ROOT / 'tools/test_hb207_joint_readout_v1.py',
              OUT / 'REVIEW-SPEC.md', OUT / 'TEST-RECEIPT.json',
              WORK / 'JOINT-READOUT-SPEC.md',
              WORK / 'candidate-runtime/exp169_partner_bundle_release.py'):
        sources[rel(p)] = pin(p)
    document = dict(schema='h207-fixed4-contract/v1', version=VERSION, builder=BUILDER,
        candidate=CANDIDATE, joint_registration_sha256=REG_SHA, source_pins=sources,
        historical_pins={name: dict(path=p, **pin(ROOT / p)) for name, p in HISTORIES.items()},
        execution_scope='fixed4_exposed_heldout_oof_only', arms=list(ARMS), contrasts=list(CONTRASTS),
        checkpoint_policy='No content reads; inherited inventory/provenance/size checks only',
        exact_F='NOT_IMPLEMENTED_PENDING_REVIEWED_H202_H204_RECEIPTS',
        qualified_for_submission=False, independent_result_review='REQUIRED',
        output=rel(OUT / 'RESULT.json'))
    inf2.write(OUT / 'CONTRACT.json', document)
    return dict(status='SOURCE_READY_REVIEW_REQUIRED', contract_sha256=pin(OUT / 'CONTRACT.json')['sha256'])


def validate_contract():
    reg = authenticated_registration()
    contract = read(OUT / 'CONTRACT.json')
    need(contract['schema'] == 'h207-fixed4-contract/v1' and contract['version'] == VERSION
         and contract['builder'] == BUILDER and contract['candidate'] == CANDIDATE
         and contract['joint_registration_sha256'] == REG_SHA
         and contract['arms'] == list(ARMS) and contract['contrasts'] == list(CONTRASTS)
         and contract['execution_scope'] == 'fixed4_exposed_heldout_oof_only'
         and contract['output'] == rel(OUT / 'RESULT.json'), 'contract scope')
    for p, expected in contract['source_pins'].items():
        need(pin(ROOT / p) == expected, 'reviewed source drift: ' + p)
    for name, expected in contract['historical_pins'].items():
        need(name in HISTORIES and expected['path'] == HISTORIES[name], 'historical identity')
        need(pin(ROOT / expected['path']) == {k: expected[k] for k in ('bytes', 'sha256')},
             'historical output drift')
    need(set(contract['historical_pins']) == set(HISTORIES), 'missing historical check')
    return reg, contract


def review_gate(review_path, expected_sha):
    """A separate reviewer supplies a file and B pins that exact review digest."""
    need(review_path is not None and expected_sha is not None, 'independent source review required')
    path = Path(review_path).resolve()
    need(path.is_relative_to(ROOT), 'review outside project')
    need(pin(path)['sha256'] == expected_sha, 'review artifact hash')
    review = read(path)
    need(review.get('schema') == 'h207-source-review/v1' and review.get('verdict') == 'PASS'
         and isinstance(review.get('reviewer'), str) and bool(review['reviewer'].strip())
         and review['reviewer'] != BUILDER and review.get('builder') == BUILDER
         and review.get('scope') == 'fixed4_exposed_heldout_oof_only'
         and review.get('contract_sha256') == pin(OUT / 'CONTRACT.json')['sha256']
         and review.get('joint_registration_sha256') == REG_SHA
         and review.get('test_receipt_sha256') == pin(OUT / 'TEST-RECEIPT.json')['sha256'],
         'independent source review identity/scope')
    evidence = ROOT / review.get('evidence_path', '')
    need(evidence.resolve().is_relative_to(ROOT) and evidence.is_file()
         and pin(evidence)['sha256'] == review.get('evidence_sha256'), 'review checklist evidence')
    return dict(path=rel(path), sha256=expected_sha, reviewer=review['reviewer'],
                evidence_path=rel(evidence), evidence_sha256=pin(evidence)['sha256'])


def verify_closure(bindings):
    """Frozen OOF/metadata/rulers hash in full; checkpoints are never opened."""
    for p, expected in bindings.items():
        path = (ROOT / p).resolve()
        need(path.is_relative_to(ROOT), 'binding outside project')
        if path.suffix.lower() in ('.pt', '.pth', '.safetensors', '.ckpt'):
            need(isinstance(expected, dict) and path.stat().st_size == expected['bytes'],
                 'checkpoint metadata/size changed')
        else:
            actual = pin(path)
            need(actual == expected if isinstance(expected, dict) else actual['sha256'] == expected,
                 'inherited input drift: ' + p)


def check_rosters(rosters, expected):
    need(len(rosters) == 5, 'five rosters')
    aligned.serving.validate_rosters(rosters)
    ids = sum((r['val'] for r in rosters), [])
    need(len(ids) == len(set(ids)) == expected['rows']
         and aligned.serving.uid_hash(ids) == expected['ordered_uid_sha256'], 'joint UID roster')
    for f, (r, e) in enumerate(zip(rosters, expected['folds'])):
        need(e['fold'] == f and len(r['val']) == e['rows']
             and aligned.serving.uid_hash(r['val']) == e['ordered_uid_sha256'], 'joint fold order')
    return ids


def merge_inputs(raw_c4, roster_c4, raw_small, roster_small):
    need(roster_c4 == roster_small, 'component train/heldout roster disagreement')
    need(set(raw_c4) == set(raw_small) == set(range(5)), 'component fold set')
    merged = {}
    for f in range(5):
        need(set(raw_c4[f]) == {'R42', 'R43', 'DINO', 'CNX', 'C4'}
             and set(raw_small[f]) == inf2.RAW_KEYS, 'component members')
        for k in ('R42', 'R43', 'DINO', 'CNX'):
            need(np.array_equal(raw_c4[f][k], raw_small[f][k]), 'overlap predictions disagree')
        merged[f] = {**raw_c4[f], 'S42': raw_small[f]['S42'], 'S43': raw_small[f]['S43']}
        need(all(a.shape == (len(roster_c4[f]['val']), 12) for a in merged[f].values()),
             'fold probability row count')
    return merged


def four_arms(raw, op):
    need(set(raw) == set(range(5)), 'five heldout folds')
    families = {k: [] for k in ('R', 'D', 'C', 'C4', 'S')}
    for f in range(5):
        row = raw[f]
        need(set(row) == KEYS, 'fixed family/member set')
        shape = row['DINO'].shape
        need(len(shape) == 2 and shape[0] > 0 and shape[1] == 12, 'fold shape')
        need(all(a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all()
                 and ((a >= 0) & (a <= 1)).all() for a in row.values()), 'FP32 finite probabilities')
        # Pair each row's two heldout seeds before any rank. No training/allfold ensemble.
        for k, a in zip(families, ((row['R42'] + row['R43']) / 2, row['DINO'],
                                  row['CNX'], row['C4'], (row['S42'] + row['S43']) / 2)):
            families[k].append(a)
    ranked = {k: op['rank'](np.concatenate(a)) for k, a in families.items()}
    hybrid = ranked['D'].copy()
    hybrid[:, :8] = ranked['S'][:, :8]
    result = {name: op['bundle'](dict(l2=ranked['R'], dino=d, convnext=c))
              for name, d, c in (('Q0', ranked['D'], ranked['C']),
                                 ('Q4', ranked['D'], ranked['C4']),
                                 ('Q2', hybrid, ranked['C']), ('Q24', hybrid, ranked['C4']))}
    need(np.array_equal(result['Q0'][:, 8:], result['Q2'][:, 8:])
         and np.array_equal(result['Q4'][:, 8:], result['Q24'][:, 8:]), 'retained DINO columns')
    return result


def interval(values):
    # Reuse conditional finite-draw percentiles, without inherited threshold/PASS fields.
    old = ml.stat(values)
    return {k: old[k] for k in ('value', 'ci95', 'valid_draws')}


def check_context(ctx):
    n = len(ctx.ids)
    need(n > 0 and len(set(ctx.ids)) == n and set(ctx.refs) == set(RULERS), 'context roster/rulers')
    need(ctx.w.shape == (301, n) and np.isfinite(ctx.w).all() and (ctx.w >= 0).all()
         and np.equal(ctx.w, np.floor(ctx.w)).all() and np.all(ctx.w[0] == 1), 'paired component weights')
    for ruler, (labels, mask, minimum) in ctx.refs.items():
        need(labels.shape == mask.shape == (n, 12) and mask.dtype == np.bool_
             and minimum == MIN_CLASS[ruler], 'ruler mask/minimum class')
        need(np.isfinite(labels[mask]).all() and np.isin(labels[mask], [0, 1]).all(), 'masked labels')


def describe_scores(scores):
    return {r: dict(macro=interval(scores[:, i].mean(1)),
                    weak4=interval(scores[:, i][:, ml.WEAK].mean(1)),
                    targets={t: interval(scores[:, i, j]) for j, t in enumerate(ml.TARGETS)})
            for i, r in enumerate(RULERS)}


def readout(ctx, arms, op):
    check_context(ctx)
    need(tuple(arms) == ARMS, 'exact four arms and order')
    scores = {k: ctx.score(a) for k, a in arms.items()}
    need(all(a.shape == (301, 4, 12) for a in scores.values()), 'score shape')
    contrasts = {key: scores['Q24'] - scores[key.split('-')[1]] for key in CONTRASTS}
    points = {k: np.stack((a[0].mean(1), a[0][:, ml.WEAK].mean(1)), axis=1)
              for k, a in scores.items()}
    decision = op['decision_readout'](points, {k: a[0] for k, a in contrasts.items()})
    support = inf2.support(ctx)
    for ruler, (y, mask, _) in ctx.refs.items():
        support[ruler]['positive_counts'] = [int(y[mask[:, j], j].sum()) for j in range(12)]
        support[ruler]['negative_counts'] = [int((1-y[mask[:, j], j]).sum()) for j in range(12)]
    return dict(rows=len(ctx.ids), ordered_uid_sha256=aligned.serving.uid_hash(ctx.ids),
        support=support, bootstrap_weights_sha256=hashlib.sha256(ctx.w.tobytes()).hexdigest(),
        arms={k: describe_scores(v) for k, v in scores.items()},
        contrasts={k: describe_scores(v) for k, v in contrasts.items()},
        registered_point_rule=decision, independent_result_review='REQUIRED')


def historical_checks(ctx, raw_c4, raw_small, arms, histories, op):
    q0c, q4 = c4.oof_values(raw_c4)
    q0h, q2h = inf2.q_pair(raw_small, ml.rank)
    q0e, q2e = inf2.q_pair(raw_small, op['rank'])
    # Old gates are replayed solely to authenticate old result scope, never applied to joint.
    need(ml.clean(c4.report(ctx, q4, q0c)) == histories['c4']['primary'], 'old C4 replay differs')
    eh = inf2.assert_historical(inf2.describe(ctx, q2h, q0h), histories['inf2']['primary'])
    ee = inf2.assert_historical(inf2.describe(ctx, q2e, q0e),
                                histories['inf2']['exact_parent_q_arithmetic'])
    comparisons = {'Q0_old_c4': ('Q0', q0c), 'Q4_old_c4': ('Q4', q4),
                   'Q0_old_inf2_exact': ('Q0', q0e), 'Q2_old_inf2_exact': ('Q2', q2e),
                   'Q0_old_inf2_rankN': ('Q0', q0h), 'Q2_old_inf2_rankN': ('Q2', q2h)}
    return dict(status='AUTHENTICATED_HISTORICAL_SINGLE_SCOPES_ONLY',
        historical_inf2_max_point_error=eh, exact_inf2_max_point_error=ee,
        old_gates_used_for_joint=False,
        nested_rank_vs_single={name: dict(max_prediction_abs=float(np.max(np.abs(arms[key] - old))),
            unequal_values=int(np.count_nonzero(arms[key] != old)),
            max_point_auc_abs=float(np.nanmax(np.abs(ctx.score(arms[key])[0] - ctx.score(old)[0]))))
            for name, (key, old) in comparisons.items()},
        scope='Historical rank/N and earlier single percentile formulas are separate; no FP64 tie equivalence assumed.')


def run(review_path, review_sha):
    # Gate before any input loader or context; existence check avoids a repeated result attempt.
    review = review_gate(review_path, review_sha)
    need(not (OUT / 'RESULT.json').exists(), 'immutable result already exists')
    reg, contract = validate_contract()
    op = operators(reg)
    regs = {p: read(ROOT / p) for p in reg['input_registration_refs']}
    for old in regs.values():
        verify_closure(old['bindings'])
    raw_c4, roster_c4, proof = c4.authenticate_oof()
    raw_small, roster_small, frozen, bindings, members, _ = inf2.load_inputs(False)
    cr, sr = list(regs.values())  # registration preserves sorted C4, INF2 authority order
    need(cr['candidate'] == 'R6W958-C4-V1' and sr['candidate'] == 'R6W958-INF2-V1', 'component authorities')
    need(members == sr['members'], 'INF2 source/deployment members')
    for p, actual in bindings.items():
        need(sr['bindings'].get(p) == actual, 'INF2 unexpected input: ' + p)
    for p, actual in proof['bindings'].items():
        expected = cr['bindings'].get(p)
        if expected is None and p in contract['source_pins']:
            expected = contract['source_pins'][p]['sha256']
        need(actual == expected, 'C4 unexpected input/source: ' + p)
    need(proof['ruler_bindings'] == frozen['ruler_bindings'] == reg['ruler_bindings'], 'shared ruler closure')
    ids = check_rosters(roster_c4, reg['roster'])
    raw = merge_inputs(raw_c4, roster_c4, raw_small, roster_small)
    arms = four_arms(raw, op)
    ctx = ml.make_context(ids, ROOT)
    need(ctx.bindings == reg['ruler_bindings'], 'pooled ruler mask bindings')
    pooled = readout(ctx, arms, op)
    histories = {k: read(ROOT / p) for k, p in HISTORIES.items()}
    historical = historical_checks(ctx, raw_c4, raw_small, arms, histories, op)
    folds, offset = {}, 0
    for f, roster in enumerate(roster_c4):
        end = offset + len(roster['val'])
        fc = ml.make_context(roster['val'], ROOT)
        need(fc.bindings == reg['ruler_bindings'], 'fold ruler mask bindings')
        folds[str(f)] = readout(fc, {k: a[offset:end] for k, a in arms.items()}, op)
        folds[str(f)]['scope'] = 'Fixed full-roster ranks sliced; correlated diagnostic, not a new gate'
        offset = end
    for old in regs.values():
        verify_closure(old['bindings'])
    validate_contract()
    need(review_gate(review_path, review_sha) == review, 'review changed during run')
    import torch
    need(not torch.cuda._initialized, 'CUDA initialized')
    result = dict(schema='h207-fixed4-result/v1', version=VERSION, candidate=CANDIDATE,
        status='COMPLETE_EXPOSED_Q_PROXY_REQUIRES_INDEPENDENT_RESULT_REVIEW',
        contract_sha256=pin(OUT / 'CONTRACT.json')['sha256'], source_review=review,
        joint_registration_sha256=REG_SHA, targets=list(ml.TARGETS),
        pooled=pooled, folds=folds, historical=historical, bootstrap=reg['bootstrap'],
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        release_verdict='HOLD_EXACT_F_AND_OTHER_QUALIFICATION_MISSING',
        lenses=dict(L1='Exposed Q weak4 only', L2='Exposed Q macro only',
            L3='Correlated rulers/folds', L4='All targets/folds; no selection',
            L5='Unavailable independent paired-seed replication',
            L6='Exact-F pending reviewed H202/H204 terminal receipts; no approximation',
            L7='No independently eligible image-labelled joint evidence'),
        limitations=['Conditional CIs exclude training/selection uncertainty.',
            'Four rulers and five folds are not independent cohorts.',
            'Related D-B-131/140/157 Q7 constructions already tried; no novelty/additive gain claim.',
            'Checkpoint contents not rehashed; authenticated ancestry and size only.'],
        checkpoint_loads=0, model_forwards=0, cuda_initialized=False)
    inf2.write(OUT / 'RESULT.json', result)
    return dict(status=result['status'], output=rel(OUT / 'RESULT.json'),
                sha256=pin(OUT / 'RESULT.json')['sha256'],
                point_rule=pooled['registered_point_rule'], release_verdict=result['release_verdict'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'run'))
    parser.add_argument('--review')
    parser.add_argument('--review-sha256')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        answer = prepare()
    elif args.command == 'validate':
        validate_contract()
        answer = dict(status='STATIC_CONTRACT_VALID_NO_NUMERICAL_EXECUTION')
    else:
        answer = run(args.review, args.review_sha256)
    print(json.dumps(answer, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # Third-party loader errors may contain row identifiers; never echo them.
        print(json.dumps(dict(status='HOLD_INVALID_OR_MISSING_DEPENDENCY',
                              error_type=type(exc).__name__, usable_result=False)))
        raise SystemExit(2)
