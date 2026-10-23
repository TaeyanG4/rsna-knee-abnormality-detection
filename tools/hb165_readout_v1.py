"""H-B-165 v1.0.0: registered OA-only routing readout (fold 3, seeds 44/45) plus seven lenses.

Registered before any fit: registration.json binds this file's SHA256 and the four scientific
configs bind registration.json. The readout authenticates the four fits against the admission
review, routes Medial/Lateral OA to the crop model, verifies the other 10 targets are
byte-identical to control and applies the registered gate (hb165_build_v1.GATE) to AUC of
seed-mean K36flip probabilities with a 2,000-draw paired connected-component bootstrap.
Seven lenses for the routed model AND for plain crop-vs-control are descriptive evidence only
(L6 = fixed HB105 fold-3 heldout partner bundle; L7 = Gold58 descriptive, OAI MOAKS unavailable).
Never prints label rows; only aggregate metrics are written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from tools import multi_lens_readout_l4_v2 as lens
from tools import hb165_build_v1 as b

VERSION = '1.0.0'
SEEDS = ('44', '45')
UID = 'StudyInstanceUID'
TARGETS = list(lens.TARGETS)
RULERS = list(lens.RULERS)
OA = tuple(TARGETS.index(t) for t in b.GATE['routed_targets'])
NON_OA = tuple(i for i in range(12) if i not in OA)
HB105 = 'artifacts/experiments/HB105_ALIGNED_OOF_V1/input-source-manifest.json'
HB105_SHA256 = 'f439d836e46eeac920146cfdf3d1b180a10075a61b3c7fb2b1d3e5de5eb35806'
PARTNER_SEEDS = {'R50': [42, 43], 'C4': [42], 'T': [42, 43]}
REPLACED_FAMILY = 'S'
FIT_EXPERIMENT = 'H-B-165 OA crop expert fold3 confirmation'
L6_RECIPE = dict(
    family_replaced='S (ConvNeXt-S MIL family; the paired H-B-165 model takes its slot)',
    fixed_families=list(PARTNER_SEEDS), Q='equal family percentile ranks of seed-mean probabilities (multi_lens run_spec arithmetic)',
    partner_source=HB105+' fold-3 heldout members (trained without fold 3; D-B-157 evaluation set, EXPOSED)',
    anchor_weight=None, anchor_reason='EXP181 CONTROL final-proxy anchor exists only on fold-0 rows; no fold-3 anchor')
L7_REASON = ('Ineligible by rule: OAI MOAKS predictions do not exist for either registered input contract '
    '(native hb147_raptor64_448_c140_s02_v1 or crop hb155_compartment64_448_side_v2 have no OAI cache; building them '
    'needs an OAI native-448 cache plus the R7 slab-prior crop geometry and laterality audit on OAI, a separate data '
    'build outside this one-change confirmation). Gold58 predictions ARE reported (58 study singletons, no component '
    'contract) and are descriptive only by the lens rule.')
STAT_NOTE = 'Bootstrap is conditional on the saved fits; seed/fit uncertainty enters only through the per-seed condition.'


def need(ok, why):
    b.need(ok, why)


def checked(table, rows, name):
    need(set(table) == set(SEEDS), name+': paired seeds 44/45 required')
    out = {}
    for s in SEEDS:
        p = np.asarray(table[s], dtype=float)
        need(p.shape == (rows, 12) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
             name+': aligned finite probabilities required')
        out[s] = p
    return out


def route(control, crop):
    """Crop predictions for the two OA targets, control for the other 10 (verified identical)."""
    control, crop = np.asarray(control, dtype=float), np.asarray(crop, dtype=float)
    need(control.shape == crop.shape and control.ndim == 2 and control.shape[1] == 12, 'aligned predictions required')
    routed = control.copy()
    routed[:, OA] = crop[:, OA]
    need(np.array_equal(routed[:, NON_OA], control[:, NON_OA]), 'non-OA columns differ')
    return routed


class Scorer:
    """AUC per ruler/target on the point row plus `draws` paired connected-component resamples."""
    def __init__(self, ids, refs, components, draws, seed):
        d = lens._base.component_draws(list(ids), components, draws=draws, seed=seed)
        n = len(ids)
        self.ids, self.refs, self.draws = list(ids), refs, draws
        self.w = np.vstack([np.ones(n), *[np.bincount(x, minlength=n) for x in d]])
        self.cache = {}

    def score(self, p, point_only=False):
        p = np.ascontiguousarray(p, dtype=float); w = self.w[:1] if point_only else self.w
        need(p.shape == (len(self.ids), 12) and np.isfinite(p).all(), 'prediction shape/nonfinite')
        key = (hashlib.sha256(p.tobytes()).hexdigest(), point_only)
        if key not in self.cache:
            self.cache[key] = np.stack([lens._base.auc_draws(p, *self.refs[r][:2], w, self.refs[r][2]) for r in RULERS], axis=1)
        return self.cache[key]


def gate(delta, seed_points, non_oa_identical, engineering=False):
    """Registered decision. delta: [1+draws, 4 rulers, 12 targets] routed seed-mean minus control
    seed-mean with the non-OA cells set to their structural 0 by `structural_delta` (identity
    verified there); seed_points: {seed: [4, 12]} routed_s minus control_s point deltas, same rule."""
    g = b.GATE; tol = g['tolerance']
    d = np.asarray(delta, dtype=float)
    base = dict(rule=g, not_release_authorization=True, engineering_only=bool(engineering))
    if d.ndim != 3 or d.shape[1:] != (4, 12) or d.shape[0] != g['draws']+1 or set(seed_points) != set(SEEDS):
        return lens._base.clean(dict(base, status='HOLD', PASS=False, reason='malformed evidence tensor'))
    sp = {s: np.asarray(seed_points[s], dtype=float) for s in SEEDS}
    if not non_oa_identical:
        return lens._base.clean(dict(base, status='HOLD', PASS=False, reason='non-OA predictions differ from control'))
    if (not np.isfinite(d[0, :, list(OA)]).all() or
            not all(v.shape == (4, 12) and np.isfinite(v[:, list(OA)]).all() for v in sp.values())):
        return lens._base.clean(dict(base, status='HOLD', PASS=False, reason='nonfinite point evidence'))
    if not np.all(d[:, :, list(NON_OA)] == 0) or not all(np.all(v[:, list(NON_OA)] == 0) for v in sp.values()):
        return lens._base.clean(dict(base, status='HOLD', PASS=False, reason='non-OA AUC deltas are not exactly zero'))
    macro = d.mean(axis=2)                       # [1+draws, 4]; non-OA cells contribute exactly 0
    mean = macro.mean(axis=1)                    # [1+draws]
    if not np.isfinite(mean[1:]).all():
        return lens._base.clean(dict(base, status='HOLD', PASS=False, reason='nonfinite bootstrap draw (no redraw/deletion)',
                                     invalid_draws=int((~np.isfinite(mean[1:])).sum())))
    lo, hi = np.percentile(mean[1:], [2.5, 97.5])
    c_macro = bool(mean[0] >= g['routed_macro_mean_min'] - tol)
    c_ci = bool(lo > g['ci_lower_strictly_above'])
    targets = {}
    for t in OA:
        points = d[0, :, t]
        count = int(np.count_nonzero(points >= g['oa_target_min'] - tol))
        targets[TARGETS[t]] = dict(ruler_points=dict(zip(RULERS, points)), mean_over_rulers=float(points.mean()),
                                   rulers_ge_threshold=count, PASS=count >= g['oa_target_min_rulers'])
    c_targets = all(v['PASS'] for v in targets.values())
    seed_means = {s: float(sp[s][:, list(OA)].mean()) for s in SEEDS}
    c_seeds = all(v > g['each_seed_mean_oa_strictly_above'] for v in seed_means.values())
    passed = c_macro and c_ci and c_targets and c_seeds
    return lens._base.clean(dict(base, status='ENGINEERING_ONLY' if engineering else ('PASS' if passed else 'FAIL'),
        PASS=bool(passed and not engineering),
        routed_macro_by_ruler=dict(zip(RULERS, macro[0])), routed_macro_mean=float(mean[0]),
        routed_macro_mean_ci95=[float(lo), float(hi)], valid_draws=int(len(mean)-1),
        oa_targets=targets, seed_mean_oa_delta=seed_means,
        conditions=dict(routed_macro_mean_ge_min=c_macro, ci_lower_gt_zero=c_ci,
                        each_oa_target_ge_min_on_3_of_4=c_targets, each_seed_mean_oa_gt_zero=c_seeds,
                        non_oa_identical=True)))


def structural_delta(routed_scores, control_scores):
    """Routed minus control AUC tensors with the 10 non-OA cells fixed at their structural 0.
    Returns (delta, identical, undefined_non_oa_cells): `identical` requires the non-OA per-draw
    AUCs, NaN pattern included, to be exactly equal (they come from byte-identical predictions)."""
    a, c = np.asarray(routed_scores, dtype=float), np.asarray(control_scores, dtype=float)
    same = bool(a.shape == c.shape and np.array_equal(a[..., list(NON_OA)], c[..., list(NON_OA)], equal_nan=True))
    d = a - c
    undefined = int(np.isnan(c[..., list(NON_OA)]).sum())
    d[..., list(NON_OA)] = 0.0
    return d, same, undefined


def descriptive(scorer, arm, control):
    d = scorer.score(arm) - scorer.score(control)
    a = scorer.score(arm, point_only=True)[0]; c = scorer.score(control, point_only=True)[0]
    def ci(v):
        f = v[1:][np.isfinite(v[1:])]
        return [float(x) for x in np.percentile(f, [2.5, 97.5])] if len(f) else None
    out = {}
    for j, t in enumerate(TARGETS):
        out[t] = dict(mean_delta_over_rulers=float(d[0, :, j].mean()), mean_delta_ci95=ci(d[:, :, j].mean(axis=1)),
                      rulers={r: dict(delta=float(d[0, i, j]), ci95=ci(d[:, i, j]), arm_auc=float(a[i, j]), control_auc=float(c[i, j]))
                              for i, r in enumerate(RULERS)})
    macro = d.mean(axis=2)
    return dict(targets=out, macro_by_ruler={r: dict(delta=float(macro[0, i]), ci95=ci(macro[:, i])) for i, r in enumerate(RULERS)},
                macro_mean=dict(delta=float(macro[0].mean()), ci95=ci(macro.mean(axis=1))), draws=scorer.draws)


def load_partners(ids, root):
    """Fixed fold-3 heldout partner probabilities per family, aligned to `ids` (pinned bytes)."""
    import pandas as pd
    root = Path(root); mp = root/HB105
    need(b.sha(mp) == HB105_SHA256, 'HB105 partner manifest changed')
    found, provenance = {}, []
    for x in b.read(mp)['members']:
        m = x['member']
        if m.get('fold') != 3 or m.get('family') not in PARTNER_SEEDS:
            continue
        need(m['heldout_count'] == len(ids) and m['heldout_uid_sha256'] == b.uid_hash(list(ids)),
             'partner heldout roster differs: '+m['id'])
        p = root/x['prediction'].replace('\\', '/')
        need(p.is_file() and b.sha(p) == x['prediction_pin']['sha256'], 'partner prediction pin mismatch: '+m['id'])
        f = pd.read_csv(p, usecols=[UID, *TARGETS], dtype={UID: str}, float_precision='round_trip')
        need(not f[UID].duplicated().any() and set(f[UID]) == set(ids), 'partner rows differ: '+m['id'])
        v = f.set_index(UID).loc[list(ids), TARGETS].to_numpy(float)
        need(np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), 'partner probabilities invalid: '+m['id'])
        found.setdefault(m['family'], {})[m['seed']] = v
        provenance.append(dict(id=m['id'], path=x['prediction'].replace('\\', '/'), sha256=x['prediction_pin']['sha256']))
    need({k: sorted(v) for k, v in found.items()} == PARTNER_SEEDS, 'partner family/seed set differs')
    probs = {fam: np.mean([found[fam][s] for s in PARTNER_SEEDS[fam]], axis=0) for fam in PARTNER_SEEDS}
    return probs, sorted(provenance, key=lambda e: e['id'])


def bundle(ctx, partners, arm, control):
    fixed = [lens._base.rank(p) for p in partners.values()]
    qa = np.mean([lens._base.rank(arm), *fixed], axis=0); qc = np.mean([lens._base.rank(control), *fixed], axis=0)
    return dict(context=ctx, arm=qa, control=qc, recipe=L6_RECIPE)


def gold_reads(ids, arm, control, root):
    import pandas as pd
    f = pd.read_csv(Path(root)/b.GOLD_SOURCE, usecols=[UID, *TARGETS], dtype={UID: str}).set_index(UID)
    need(not f.index.has_duplicates and set(ids) <= set(f.index), 'Gold58 labels missing')
    y = f.loc[list(ids), TARGETS].to_numpy(float); mask = np.isfinite(y)
    need(np.isin(y[mask], [0, 1]).all(), 'Gold58 labels must be binary or missing')
    return {'gold58': dict(ids=list(ids), arm=np.asarray(arm, float), control=np.asarray(control, float),
                           labels=np.nan_to_num(y), mask=mask, components=[{'component_id': u, 'members': [u]} for u in ids],
                           eligible=False, component_policy='Gold58 study singletons: no validation component contract; descriptive only')}


def lens_table(result, routed_or_plain):
    ls = result['lenses']; rows = {}
    for k in ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'):
        v = ls[k]
        rows[k] = dict(value=v.get('value'), ci95=v.get('ci95'), PASS=v.get('PASS'), eligible=v.get('eligible', True),
                       reason=v.get('reason'))
    return dict(comparison=routed_or_plain, lenses=rows, passing=result.get('passing_lenses'), decision_label=result.get('decision_label'))


def annotate(result, partners_reason, gold_present):
    ls = result['lenses']
    if partners_reason:
        ls['L6'] = lens._base.unavailable(partners_reason)
    reason = L7_REASON if gold_present else 'Gold58 predictions missing (HOLD for L7 evidence). ' + L7_REASON
    ls['L7'] = {**ls['L7'], 'eligible': False, 'PASS': False, 'reason': reason}
    return result


def evaluate_all(ctx, scorer, controls, crops, *, partners=None, partners_reason=None,
                 gold=None, engineering=False):
    n = len(ctx.ids)
    controls, crops = checked(controls, n, 'controls'), checked(crops, n, 'crops')
    routed = {s: route(controls[s], crops[s]) for s in SEEDS}
    cm = np.mean([controls[s] for s in SEEDS], axis=0); xm = np.mean([crops[s] for s in SEEDS], axis=0)
    rm = route(cm, xm)
    identical = (np.array_equal(rm[:, list(NON_OA)], cm[:, list(NON_OA)]) and
                 np.array_equal(rm, np.mean([routed[s] for s in SEEDS], axis=0)) and
                 all(np.array_equal(routed[s][:, list(NON_OA)], controls[s][:, list(NON_OA)]) for s in SEEDS))
    delta, same, undefined = structural_delta(scorer.score(rm), scorer.score(cm))
    seed_points, seed_undefined = {}, {}
    for s in SEEDS:
        sd, ss, su = structural_delta(scorer.score(routed[s], point_only=True), scorer.score(controls[s], point_only=True))
        seed_points[s], seed_undefined[s] = sd[0], su; same = same and ss
    identical = bool(identical and same)
    registered = gate(delta, seed_points, identical, engineering)
    registered['undefined_non_oa_cells'] = dict(seed_mean_draws=undefined, seed_points=seed_undefined,
        note='rare non-OA target AUC undefined in BOTH identical models on these resamples; structural delta 0')
    images = {}
    if gold is not None:
        gc = checked(gold['controls'], len(gold['ids']), 'gold controls'); gx = checked(gold['crops'], len(gold['ids']), 'gold crops')
        gcm = np.mean([gc[s] for s in SEEDS], axis=0); gxm = np.mean([gx[s] for s in SEEDS], axis=0)
        images['routed'] = gold_reads(gold['ids'], route(gcm, gxm), gcm, gold['root'])
        images['plain'] = gold_reads(gold['ids'], gxm, gcm, gold['root'])
    out = {}
    for name, arms, arm_mean in (('routed', routed, rm), ('plain', crops, xm)):
        bun = bundle(ctx, partners, arm_mean, cm) if partners else None
        r = lens.evaluate(ctx, arms, controls, label_change=False, bundle=bun, image_reads=images.get(name))
        out[name] = annotate(r, None if partners else (partners_reason or 'partner bundle not supplied'), bool(images))
    return lens._base.clean(dict(
        schema='hb165-readout/v1', version=VERSION, experiment=b.EXPERIMENT, rows=n, seeds=list(SEEDS),
        exposure='EXPOSED', clean_oof=False, exposure_statement=b.EXPOSURE_STATEMENT, estimand=b.COORDINATOR_DECISION['estimand'],
        registered_gate=registered, non_oa_identical=bool(identical),
        routed_descriptive=descriptive(scorer, rm, cm), plain_descriptive=descriptive(scorer, xm, cm),
        lens_summary={k: lens_table(v, k) for k, v in out.items()}, lenses_routed=out['routed'], lenses_plain=out['plain'],
        notes=[STAT_NOTE, 'Lenses are evidence, not gates; only registered_gate is the H-B-165 decision.',
               'Four correlated report-derived rulers; EXPOSED, not clean OOF; not release authorization.',
               'H-B-155 thresholds unchanged and not applied retroactively.']))


def validate_records(c, summary, receipt, ids, *, config_hash, csv_hash, checkpoint_hash, prepare_hash, runtime_hash):
    i = receipt.get('identity', {}); need(i == summary.get('identity'), 'summary/receipt identity differs')
    expected = dict(experiment=FIT_EXPERIMENT, version='1.0.0', status='EXPOSED_PAIRED_FOLD3', fold=3, arm=c['arm'], seed=c['seed'],
        resolution=224, config_sha256=config_hash, registration_sha256=config_hash,
        train_sha256=b.uid_hash(c['train']), val_sha256=b.uid_hash(c['val']), workers=2, epochs=20, smoke=False,
        engineering_only=False, gpu_name=b.GPU, weights=c['weights'], model=c['recipe']['model'], batch=8, microbatch=2,
        accumulation=4, corpus_contract=c['corpus_contract'], exclusions_sha256=c['exclusions_sha256'],
        side_unknown_policy=c['side_unknown_policy'], prepare_sha256=prepare_hash, runtime_snapshot_sha256=runtime_hash,
        labels_sha256=c['inputs'][c['label_path']], exposure='EXPOSED', clean_oof=False, estimand=c['estimand'])
    for key, value in expected.items():
        need(i.get(key) == value, 'old/foreign fit identity: '+key)
    need(summary.get('passed') is True and summary.get('smoke') is False and summary.get('device') == 'cuda'
         and summary.get('steps') == c['optimizer_steps'] and summary.get('counters', {}).get('updates') == c['optimizer_steps']
         and 0 <= summary['counters']['skips'] <= c['optimizer_steps']//100, 'partial fit forbidden')
    need(receipt.get('k') == 36 and receipt.get('tta') is True and receipt.get('engineering_only') is False, 'scientific K36 flip required')
    need(receipt.get('rows') == len(ids) and receipt.get('uid_sha256') == b.uid_hash(ids), 'prediction roster differs')
    need(receipt.get('csv_sha256') == csv_hash and receipt.get('checkpoint_sha256') == summary.get('checkpoint_sha256') == checkpoint_hash,
         'prediction/checkpoint binding differs')
    r = i.get('runtime', {})
    need(r.get('policy') == 'strict' and r.get('precision_fallback_allowed') is False and str(r.get('device', '')).startswith('cuda'),
         'runtime differs')
    return i


def registration_check():
    b.registration_binding()
    reg = b.read(b.WORK/'registration.json')
    need(reg['gate'] == b.GATE and reg['readout']['source_sha256'] == b.sha(Path(__file__)),
         'readout source/rule differs from the registration; a versioned, reviewed amendment is required')
    return reg


def load_prediction(path, ids):
    import pandas as pd
    f = pd.read_csv(path, dtype={UID: str}, float_precision='round_trip')
    need(f[UID].astype(str).tolist() == list(ids), 'CSV order differs: '+str(path))
    return f[TARGETS].to_numpy(dtype=float)


def run(review, root=None):
    root = Path(root or b.ROOT); reg = registration_check()
    controls, crops, gold_c, gold_x, bindings, runtime = {}, {}, {}, {}, {}, None
    val = gold = None
    for arm, table, gtable in (('control', controls, gold_c), ('crop', crops, gold_x)):
        for seed in (44, 45):
            cp = b.config_path(arm, seed); c = b.require_fit(cp, review)
            d = b.WORK/'fits'/b.member(arm, seed)
            need(b.read(d/'heartbeat.json').get('status') == 'complete', 'fit incomplete; do not open outputs')
            summary = b.read(d/'summary.json'); b.require_runtime(cp, b.read(d/'runtime.json'))
            kw = dict(config_hash=b.sha(cp), checkpoint_hash=b.sha(d/'final.pt'), prepare_hash=b.sha(b.WORK/'prepared.json'),
                      runtime_hash=b.sha(d/'runtime.json'))
            for name, ids, tab in (('val', c['val'], table), ('gold', c['gold'], gtable)):
                receipt = b.read(d/(name+'.receipt.json'))
                i = validate_records(c, summary, receipt, ids, csv_hash=b.sha(d/(name+'.csv')), **kw)
                runtime = runtime or i['runtime']; need(runtime == i['runtime'], 'paired runtime differs')
                tab[str(seed)] = load_prediction(d/(name+'.csv'), ids)
                bindings[b.member(arm, seed)+'/'+name] = receipt
            val, gold = val or c['val'], gold or c['gold']
            need(val == c['val'] and gold == c['gold'], 'member rosters differ')
    started = time.monotonic()
    ctx = lens._base.make_context(val, root)
    comps = json.loads((root/lens._base.COMPONENTS).read_text())
    scorer = Scorer(val, ctx.refs, comps, b.GATE['draws'], b.GATE['bootstrap_seed'])
    partners, provenance = load_partners(val, root)
    result = evaluate_all(ctx, scorer, controls, crops, partners=partners,
                          gold=dict(ids=gold, controls=gold_c, crops=gold_x, root=root))
    result.update(prediction_bindings=bindings, ruler_input_hashes=ctx.bindings, partner_provenance=provenance,
                  registration={b.rel(b.WORK/'registration.json'): b.sha(b.WORK/'registration.json')},
                  readout_source_sha256=reg['readout']['source_sha256'], seconds=time.monotonic()-started)
    b.immutable(b.WORK/'readout.json', lens._base.clean(result))
    return dict(status=result['registered_gate']['status'], readout=b.rel(b.WORK/'readout.json'))


def engineering(root=None, out=None):
    """Synthetic predictions on the REAL fold-3 roster, rulers, partners and Gold58 labels:
    exercises the full path (2,000 draws) without any model-derived fold-3 analysis."""
    root = Path(root or b.ROOT); val = b.roster()[1]; gold = b.gold_ids()
    rng = np.random.default_rng(165)
    def pair(n):
        c = {s: rng.random((n, 12)) for s in SEEDS}
        return c, {s: np.clip(c[s]+rng.normal(0, .05, (n, 12)), 0, 1) for s in SEEDS}
    controls, crops = pair(len(val)); gc, gx = pair(len(gold))
    started = time.monotonic()
    ctx = lens._base.make_context(val, root)
    comps = json.loads((root/lens._base.COMPONENTS).read_text())
    scorer = Scorer(val, ctx.refs, comps, b.GATE['draws'], b.GATE['bootstrap_seed'])
    partners, provenance = load_partners(val, root)
    result = evaluate_all(ctx, scorer, controls, crops, partners=partners,
                          gold=dict(ids=gold, controls=gc, crops=gx, root=root), engineering=True)
    result.update(status='ENGINEERING_ONLY_SYNTHETIC', partner_provenance=provenance, seconds=time.monotonic()-started)
    out = Path(out or b.WORK/'engineering-readout.json')
    b.atomic_json(out, lens._base.clean(result))
    eligibility = {k: {L: v['lenses'][L]['eligible'] for L in v['lenses']} for k, v in result['lens_summary'].items()}
    return dict(status='ENGINEERING_ONLY_SYNTHETIC', gate_status=result['registered_gate']['status'],
                non_oa_identical=result['non_oa_identical'], valid_draws=result['registered_gate'].get('valid_draws'),
                lens_eligibility=eligibility, seconds=round(result['seconds'], 1), out=str(out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path); p.add_argument('--engineering', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    if a.engineering:
        print(json.dumps(engineering(out=a.out))); return 0
    if a.review is None:
        p.error('--review is required for the scientific readout')
    result = run(a.review)
    print(json.dumps(result)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
