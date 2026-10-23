"""H-B-165 Lateral-only follow-up v1.0.0 (CPU, existing predictions only).

Changelog 1.0.0: fixed Lateral route, authenticated fold2/fold3 pairs, original
fold-specific L6 recipes, structural-zero CIs, throughput and provenance output.
Historical readouts/registrations are never changed. EXPOSED, not release approval.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from tools import hb165_readout_v1 as old
from tools import hb155_readout_v1 as r155
from tools import lens_completion_1015_v1 as completion

VERSION = '1.0.0'
ROOT = old.b.ROOT
OUT = ROOT / '.local/scratch/astra-b5/hb165-lateral'
SPEC = OUT / 'SPEC-v100.md'
TARGETS, RULERS = old.TARGETS, old.RULERS
LAT = TARGETS.index('Lateral OA')
OTHER = tuple(i for i in range(12) if i != LAT)
SOURCES = ['tools/hb165_lateral_readout_v1.py', 'tests/test_hb165_lateral_readout_v1.py',
           'tools/hb165_readout_v1.py', 'tools/hb155_readout_v1.py',
           'tools/lens_completion_1015_v1.py', 'tools/hb105_aligned_oof_v1.py',
           'tools/multi_lens_readout.py', 'tools/multi_lens_readout_l4_v2.py',
           'tools/hb155_build_v1.py', 'tools/hb165_build_v1.py']


def bindings():
    return {p: old.b.sha(ROOT/p) for p in [*SOURCES, old.b.rel(SPEC)]}


def require_review(path):
    review = old.b.read(path)
    old.need(review.get('verdict') == 'PASS' and review.get('independent') is True
             and review.get('reviewer') not in (None, '', 'astra-b5', 'builder', '/root')
             and review.get('bindings') == bindings(), 'independent pre-review missing/stale')


def probability(p, n):
    p = np.asarray(p, float)
    old.need(p.shape == (n, 12) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
             'finite aligned probabilities required')
    return p


def route(control, crop):
    control = probability(control, len(control)); crop = probability(crop, len(control))
    routed = control.copy(); routed[:, LAT] = crop[:, LAT]
    old.need(np.array_equal(routed[:, OTHER], control[:, OTHER]), 'eleven columns differ')
    return routed


def structural(a, c):
    old.need(a.shape == c.shape and np.array_equal(a[..., OTHER], c[..., OTHER], equal_nan=True),
             'unchanged AUC cells differ')
    delta = a - c; delta[..., OTHER] = 0
    return delta


class RoutedContext:
    """Preserve all lens arithmetic; define zero only for identical untouched columns."""
    def __init__(self, ctx):
        self.ctx = ctx

    def __getattr__(self, name):
        return getattr(self.ctx, name)

    def delta(self, a, c):
        old.need(np.array_equal(a[:, OTHER], c[:, OTHER]), 'non-Lateral predictions differ')
        return structural(self.ctx.score(a), self.ctx.score(c))


def stats(values):
    v = np.asarray(values)
    count = int(np.isfinite(v[1:]).sum())
    return dict(delta=float(v[0]), ci95=np.percentile(v[1:], [2.5, 97.5]).tolist()
                if count == len(v)-1 and count else None,
                valid_draws=count, invalid_draws=len(v)-1-count,
                ci_policy='all fixed draws required; no redraw or deletion')


def summarize_delta(d):
    macro = d.mean(axis=2)
    return dict(lateral_by_ruler={r: stats(d[:, i, LAT]) for i, r in enumerate(RULERS)},
                lateral_mean=stats(d[:, :, LAT].mean(axis=1)),
                macro_by_ruler={r: stats(macro[:, i]) for i, r in enumerate(RULERS)},
                macro_mean=stats(macro.mean(axis=1)))


def check_config(c, fold):
    old.need(c['fold'] == fold and c['recipe']['targets'] == TARGETS, 'fold/target order differs')
    old.need(len(c['val']) == len(set(c['val'])) and len(c['train']) == len(set(c['train']))
             and not set(c['train']) & set(c['val']), 'duplicate or overlapping train/validation IDs')


def load_fold(fold):
    b = r155.b if fold == 2 else old.b
    seeds = ('42', '43') if fold == 2 else ('44', '45')
    controls, crops, provenance, timings = {}, {}, {}, []
    ids, runtime = None, None
    if fold == 3:
        old.registration_check()
    for arm, table in (('control', controls), ('crop', crops)):
        for seed in seeds:
            cp = b.config_path(arm, int(seed)); directory = b.WORK/'fits'/b.member(arm, int(seed))
            old.need(b.read(directory/'heartbeat.json').get('status') == 'complete', 'fit incomplete')
            if fold == 2:
                member = completion.mil_member(arm, int(seed))
                c = member['config']; p = completion.mil_predictions(member)
            else:
                c = b.require_fit(cp, b.WORK/'review.json')
                b.require_runtime(cp, b.read(directory/'runtime.json'))
                old.validate_records(c, b.read(directory/'summary.json'), b.read(directory/'val.receipt.json'), c['val'],
                    config_hash=b.sha(cp), csv_hash=b.sha(directory/'val.csv'), checkpoint_hash=b.sha(directory/'final.pt'),
                    prepare_hash=b.sha(b.WORK/'prepared.json'), runtime_hash=b.sha(directory/'runtime.json'))
                p = old.load_prediction(directory/'val.csv', c['val'])
            check_config(c, fold)
            ids = ids or c['val']; old.need(ids == c['val'], 'paired rosters differ')
            summary = b.read(directory/'summary.json'); this_runtime = summary['identity']['runtime']
            runtime = runtime or this_runtime; old.need(runtime == this_runtime, 'paired runtimes differ')
            table[seed] = probability(p, len(ids))
            for path in [cp, directory/'heartbeat.json', directory/'summary.json', directory/'val.csv',
                         directory/'val.receipt.json', directory/'runtime.json', b.WORK/'review.json',
                         b.WORK/'prepared.json', b.WORK/'readout.json']:
                provenance[b.rel(path)] = b.sha(path)
            timings.append(dict(arm=arm, seed=seed, fold=fold, rows=len(ids),
                train_rows=len(c['train']), train_val_disjoint=True, gpu=summary['gpu_name'],
                checkpoint=b.rel(directory/'final.pt'), checkpoint_sha256=summary['checkpoint_sha256'],
                k36flip_seconds=summary['validation_seconds']['val'],
                seconds_per_study=summary['validation_seconds']['val']/len(ids)))
    return ids, controls, crops, provenance, timings


def make_bundle(fold, ctx, ids, routed, controls):
    if fold == 2:
        (qa, qc), auth = completion.aligned_bundle(2, ids, routed, controls, 'replace')
        return (dict(context=ctx, arm=qa, control=qc, recipe=dict(
            scope='ALIGNED_OOF_SURROGATE_ONLY', fold=2, family_replaced='S42/S43',
            fixed_partner='Q7S (R50, C4, S); all other folds unchanged',
            arithmetic='FP32 seed mean; pooled family and bundle ranks on all 4343 rows; fold-2 slice',
            source='lens_completion_1015_v1.readout_hb155 / aligned_bundle replace')),
            auth['bindings'])
    partners, provenance = old.load_partners(ids, ROOT)
    bundle = old.bundle(ctx, partners, np.mean(list(routed.values()), axis=0),
                        np.mean(list(controls.values()), axis=0))
    return bundle, provenance


def analyze(fold):
    ids, controls, crops, provenance, timings = load_fold(fold)
    ctx = old.lens._base.make_context(ids, ROOT); routed_ctx = RoutedContext(ctx)
    scorer = old.Scorer(ids, ctx.refs, old.b.read(ROOT/old.lens._base.COMPONENTS),
                        old.b.GATE['draws'], old.b.GATE['bootstrap_seed'])
    routed = {s: route(controls[s], crops[s]) for s in controls}
    cm, xm = np.mean(list(controls.values()), axis=0), np.mean(list(crops.values()), axis=0)
    rm = route(cm, xm)
    old.need(np.array_equal(rm, np.mean(list(routed.values()), axis=0)), 'seed-mean route differs')
    main = summarize_delta(structural(scorer.score(rm), scorer.score(cm)))
    per_seed = {s: summarize_delta(structural(scorer.score(routed[s], True), scorer.score(controls[s], True)))
                for s in controls}
    plain = scorer.score(xm, True)[0] - scorer.score(cm, True)[0]
    transparency = {t: dict(mean=float(plain[:, j].mean()), rulers=dict(zip(RULERS, plain[:, j])))
                    for j, t in enumerate(TARGETS)}
    bundle, partner_provenance = make_bundle(fold, routed_ctx, ids, routed, controls)
    lenses = old.lens.evaluate(routed_ctx, routed, controls, bundle=bundle)
    lenses['lenses']['L7'] = old.lens._base.unavailable(
        'No paired OAI MOAKS predictions/cache for either input contract. Gold58 is descriptive only '
        'and cannot satisfy L7; no new panel inference in this existing-predictions analysis.')
    lenses['lenses']['L6']['final_proxy'] = old.lens._base.unavailable(
        'No heldout CoAtNet anchor on this fold; EXP181 CONTROL anchor only has fold 0.')
    if fold == 2:
        (sa, sc), _ = completion.aligned_bundle(2, ids, routed, controls, 'add')
        d = routed_ctx.delta(sa, sc)
        secondary = old.lens._base.stat(d[:, :2].mean((1, 2)), .0005, eligible=False,
                                       reason='Historical add bundle, descriptive only; not selection')
        secondary['PASS'] = False
        lenses['lenses']['L6']['secondary_add_bundle'] = secondary
    return dict(fold=fold, rows=len(ids), seeds=list(controls), eleven_columns_identical=True,
        ordered_uid_sha256=old.b.uid_hash(ids), main=main, per_seed=per_seed,
        transparency_crop_minus_control_seed_mean=transparency,
        lens_summary=old.lens_table(lenses, 'Lateral-only route minus control'), lenses=lenses,
        prediction_bindings=provenance, ruler_bindings=ctx.bindings,
        partner_provenance=partner_provenance, timings=timings)


def run(review):
    require_review(review)
    started = time.monotonic()
    result = dict(version=VERSION, exposure='EXPOSED', clean_oof=False,
        route='Lateral OA crop; all other 11 targets control', release_authorized=False,
        notes=['Route fixed by owner before this follow-up, after prior target results were known.',
               'Main: AUC of seed-mean probabilities, mean over four rulers, 2000 component draws.',
               'Lenses: original 300 draws; L1-L4 average seed AUC deltas, JEV/consensus primary.',
               'Conditional CIs exclude fit, seed and selection uncertainty; rulers are correlated.',
               'Fold-specific L6 recipes differ and are not directly comparable; historical gates unchanged.'],
        source_bindings=bindings(), pre_review_sha256=old.b.sha(review),
        folds={str(f): analyze(f) for f in (2, 3)})
    crop = [t for f in result['folds'].values() for t in f['timings'] if t['arm'] == 'crop']
    result['runtime'] = dict(measured_crop_4090_seconds_per_study=[t['seconds_per_study'] for t in crop],
        paired_models_4090_seconds_per_study={f: sum(t['seconds_per_study'] for t in v['timings'] if t['arm'] == 'crop')
                                             for f, v in result['folds'].items()},
        four_models_4090_seconds_per_study=sum(t['seconds_per_study'] for t in crop),
        t4_estimate='Multiply measured model inference cost by an explicitly assumed T4/4090 slowdown. '
                    'No crop T4 measurement exists in these receipts; preprocessing and integration unmeasured.')
    result['seconds'] = time.monotonic()-started
    old.b.atomic_json(OUT/'readout-v100.json', old.lens._base.clean(result))
    print(json.dumps(dict(status='CPU_COMPLETE', seconds=round(result['seconds'], 1),
                          out=old.b.rel(OUT/'readout-v100.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    run(parser.parse_args().review)
