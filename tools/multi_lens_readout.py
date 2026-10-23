"""Multi-lens paired CPU readout v1.0.0.

Changelog 1.0.0: seven lenses, paired component CIs, fixed support, retro adapters.
Mechanical gate implements the dispatch request; it is not B's holistic release decision.
No CI significance gate. L5 passes when seed-mean predictions meet L1 or L2 and
all seed deltas of the SAME statistic are positive. L6 uses equal-family rank Q;
its optional final proxy is 50:50 rank(Q)/rank(EXP181 CONTROL), never LB tuned.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.exp156_k_sweep import TARGETS, UID, COMPONENTS, references, JEV, AQUILA, DREAD, CONSENSUS
from tools.exp125_radr50_k16_local import component_draws
VERSION = '1.0.0'
ROOT = Path(__file__).resolve().parents[1]
RULERS = ('jev', 'consensus', 'dread', 'aquila')
WEAK = (3, 5, 6, 8)
OTHER = tuple(i for i in range(12) if i not in WEAK)
DRAWS, SEED = 300, 20261003


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rank(p): return rankdata(p, axis=0, method='average') / len(p)


def load_oof(paths, root=ROOT, fold_rank=False):
    """Multiple paths are disjoint folds, not seeds; rank within fold before pooling."""
    import pandas as pd
    if isinstance(paths, dict) and 'rank_folds_like' in paths:
        ids, p, provenance = load_oof(paths['files'], root)
        index = {u: i for i, u in enumerate(ids)}; covered = []
        for fold in paths['rank_folds_like']:
            ri, _, pr = load_oof(fold, root)
            if not set(ri) <= set(ids): raise ValueError('Fold roster outside pooled OOF')
            ix = [index[u] for u in ri]; p[ix] = rank(p[ix]); covered.extend(ri)
            provenance.extend(pr)
        if sorted(covered) != ids: raise ValueError('Fold rosters must partition pooled OOF')
        return ids, p, provenance
    if isinstance(paths, dict) and 'blend' in paths:
        roster, total, provenance = None, None, []
        for part in paths['blend']:
            ids, p, prov = load_oof(part['files'], root, fold_rank)
            if roster is not None and ids != roster: raise ValueError('Blend study rosters differ')
            roster = ids
            mode = paths.get('rank', True)
            p = rank(p) if mode == 'pooled' or (mode and not fold_rank) else p
            total = part['weight'] * p if total is None else total + part['weight'] * p
            provenance.extend(prov)
        if abs(sum(x['weight'] for x in paths['blend']) - 1) > 1e-9: raise ValueError('Blend weights must sum to one')
        return roster, total, provenance
    if isinstance(paths, (str, Path, dict)): paths = [paths]
    ids, arrays, provenance = [], [], []
    for entry in paths:
        entry = {'path': str(entry)} if not isinstance(entry, dict) else entry
        path = Path(str(entry['path']).replace('\\', '/'))
        if not path.is_absolute(): path = Path(root) / path
        if not path.is_file(): raise FileNotFoundError(str(path))
        digest = sha(path)
        if entry.get('sha256') and digest != entry['sha256']: raise ValueError('OOF hash mismatch: ' + str(path))
        df = pd.read_csv(path, usecols=[UID, *TARGETS], dtype={UID: str}, float_precision='round_trip')
        p = df[list(TARGETS)].to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any(): raise ValueError('Invalid probabilities')
        ids.extend(df[UID].tolist()); arrays.append(rank(p) if fold_rank else p)
        provenance.append({'path': str(path), 'sha256': digest, 'rows': len(df)})
    if len(set(ids)) != len(ids) or not ids: raise ValueError('Empty or duplicate study roster')
    order = np.argsort(ids)
    return np.asarray(ids)[order].tolist(), np.concatenate(arrays)[order], provenance


def weights(ids, components):
    draws = component_draws(ids, components, draws=DRAWS, seed=SEED)
    return np.vstack([np.ones(len(ids)), *[np.bincount(d, minlength=len(ids)) for d in draws]])


def auc_draws(p, y, mask, w, min_class=1):
    """Exactly equivalent to expanding component draws; tied scores get half credit."""
    out = np.full((len(w), p.shape[1]), np.nan)
    for t in range(p.shape[1]):
        keep = np.flatnonzero(mask[:, t]); order = keep[np.argsort(p[keep, t], kind='stable')]
        if not len(order): continue
        starts = np.r_[0, np.flatnonzero(np.diff(p[order, t])) + 1]
        yy = y[order, t]; ww = w[:, order]
        pos = np.add.reduceat(ww * yy, starts, axis=1)
        neg = np.add.reduceat(ww * (1 - yy), starts, axis=1)
        npos, nneg = pos.sum(1), neg.sum(1)
        valid = (npos >= min_class) & (nneg >= min_class)
        num = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(1)
        out[valid, t] = num[valid] / (npos[valid] * nneg[valid])
    return out


class Context:
    def __init__(self, ids, refs, components):
        self.ids, self.refs = ids, refs
        self.w = weights(ids, components)
        self.cache = {}
    def score(self, p):
        p = np.asarray(p, float)
        if p.shape != (len(self.ids), 12) or not np.isfinite(p).all(): raise ValueError('Prediction shape/nonfinite')
        key = hashlib.sha256(p.tobytes()).hexdigest()
        if key not in self.cache:
            self.cache[key] = np.stack([auc_draws(p, *self.refs[r][:2], self.w, self.refs[r][2]) for r in RULERS], axis=1)
        return self.cache[key]
    def delta(self, a, c): return self.score(a) - self.score(c)


def stat(v, threshold=0, eligible=True, reason=None):
    v = np.asarray(v); finite = v[1:][np.isfinite(v[1:])]
    return {'value': clean(v[0]), 'ci95': clean(np.percentile(finite, [2.5, 97.5])) if len(finite) else None,
            'valid_draws': len(finite), 'threshold': threshold, 'PASS': bool(np.isfinite(v[0]) and v[0] >= threshold - 1e-12),
            'eligible': eligible, **({'reason': reason} if reason else {})}


def unavailable(reason):
    return {'value': None, 'ci95': None, 'PASS': False, 'eligible': False, 'reason': reason}


def lenses(delta, label_change=False):
    primary = delta[:, :2].mean(1)
    macro = delta.mean(2)
    weak = primary[:, WEAK].mean(1); overall = primary.mean(1)
    guard = bool(np.isfinite(delta[0]).all() and overall[0] >= -.001 - 1e-12 and
                 np.all(primary[0, OTHER] >= -.010 - 1e-12) and macro[0, 2] >= -.003 - 1e-12)
    l3 = {r: stat(macro[:, i], .002) for i, r in enumerate(RULERS)}
    for i, r in enumerate(RULERS):
        l3[r]['PASS'] &= bool(np.all(np.delete(macro[0], i) >= -.001 - 1e-12))
        l3[r]['self_referential'] = bool(label_change and r == 'aquila')
    target = {}
    for t, name in enumerate(TARGETS):
        target[name] = stat(primary[:, t], .003)
        target[name]['positive_rulers'] = int((delta[0, :, t] > 0).sum())
        target[name]['PASS'] &= target[name]['positive_rulers'] >= 3
    selected = [t for t, s in target.items() if s['PASS']]
    return {'L1': stat(weak, .001), 'L2': stat(overall, .001),
            'L3': {**stat(macro.max(1), .002), 'PASS': any(s['PASS'] for s in l3.values()), 'rulers': l3},
            'L4': {**stat(primary.max(1), .003), 'PASS': bool(selected), 'targets': target, 'selected_targets': selected},
            'L5': unavailable('Fewer than two paired seeds'),
            'L6': unavailable('No compatible partner bundle supplied'),
            'L7': unavailable('No paired image-read predictions supplied')}, {
                'PASS': guard, 'macro12': stat(overall, -.001), 'dread_macro': stat(macro[:, 2], -.003),
                'other_targets': {TARGETS[t]: stat(primary[:, t], -.010) for t in OTHER}}, {
                    r: {'macro': stat(macro[:, i]), 'targets': {t: stat(delta[:, i, j]) for j, t in enumerate(TARGETS)}}
                    for i, r in enumerate(RULERS)}


def verdict(ls, guard):
    passing = [k for k, v in ls.items() if v['PASS'] and v.get('eligible', True)]
    advance = guard['PASS'] and bool(passing)
    targets = ls['L4']['selected_targets']
    only_target = advance and passing == ['L4']
    return {'verdict': 'ADVANCE' if advance else 'HOLD', 'passing_lenses': passing,
            'decision_label': 'PER-TARGET ADVANCE (' + ', '.join(targets) + ')' if only_target else ('ADVANCE' if advance else 'HOLD'),
            'release_scope': targets if only_target else ('whole arm' if advance else 'none')}


def evaluate(ctx, arms, controls, label_change=False, bundle=None, image_reads=None):
    if set(arms) != set(controls) or not arms: raise ValueError('Paired seed names differ')
    seed_d = {s: ctx.delta(arms[s], controls[s]) for s in arms}
    delta = np.mean(list(seed_d.values()), axis=0)
    ls, guard, rulers = lenses(delta, label_change)
    aa = np.mean([ctx.score(p)[0] for p in arms.values()], axis=0)
    cc = np.mean([ctx.score(p)[0] for p in controls.values()], axis=0)
    for i, r in enumerate(RULERS):
        for j, target in enumerate(TARGETS):
            y, mask, _ = ctx.refs[r]
            rulers[r]['targets'][target].update(arm_auc=aa[i,j], control_auc=cc[i,j],
                positives=int(y[mask[:,j],j].sum()), negatives=int((1-y[mask[:,j],j]).sum()))
    if len(arms) >= 2:
        sd = ctx.delta(np.mean(list(arms.values()), axis=0), np.mean(list(controls.values()), axis=0))
        sl, sg, sr = lenses(sd, label_change)
        agrees = {k: all(lenses(d)[0][k]['value'] is not None and lenses(d)[0][k]['value'] > 0 for d in seed_d.values()) for k in ('L1', 'L2')}
        ls['L5'] = {**sl['L2'], 'PASS': sg['PASS'] and any(sl[k]['PASS'] and agrees[k] for k in agrees),
                    'weak_four': sl['L1'], 'macro12': sl['L2'], 'sign_agreement': agrees, 'rulers': sr,
                    'rule': 'AUC of arithmetic seed-mean predictions; L1 or L2 threshold + all seeds positive for same statistic + averaged guard'}
    if bundle is not None:
        bd = bundle['context'].delta(bundle['arm'], bundle['control'])
        ls['L6'] = {**stat(bd[:, :2].mean((1, 2)), .0005), 'recipe': bundle['recipe'], 'rows': len(bundle['context'].ids)}
        if 'anchor_arm' in bundle:
            ad = bundle['context'].delta(bundle['anchor_arm'], bundle['anchor_control'])
            ls['L6']['final_proxy'] = stat(ad[:, :2].mean((1, 2)), .0005)
            ls['L6']['PASS'] |= ls['L6']['final_proxy']['PASS']
    if image_reads:
        entries = {}
        for name, item in image_reads.items():
            d = auc_draws(item['arm'], item['labels'], item['mask'], weights(item['ids'], item['components'])) - auc_draws(item['control'], item['labels'], item['mask'], weights(item['ids'], item['components']))
            ix = WEAK if name == 'oai_moaks' else tuple(np.flatnonzero(np.isfinite(d[0])))
            values = d[:, ix].mean(1) if ix else np.full(DRAWS+1, np.nan)
            entries[name] = stat(values, .002 if name == 'oai_moaks' else 0,
                                 eligible=name == 'oai_moaks' and bool(item.get('eligible', False)))
            entries[name]['fixed_targets'] = [TARGETS[t] for t in ix]
            entries[name]['component_policy'] = item.get('component_policy', 'supplied connected components')
            entries[name]['targets'] = {t: stat(d[:, i]) for i, t in enumerate(TARGETS)}
        ls['L7'] = {'value': {k: v['value'] for k, v in entries.items()}, 'ci95': {k: v['ci95'] for k, v in entries.items()},
                    'PASS': any(v['PASS'] and v['eligible'] for v in entries.values()), 'entries': entries,
                    'eligible': any(v['eligible'] for v in entries.values())}
    return clean({'version': VERSION, 'rows': len(ctx.ids), 'seeds': list(arms), 'lenses': ls, 'harm_guard': guard,
                  'rulers': rulers, 'per_seed': {s: lenses(d, label_change)[0] for s, d in seed_d.items()},
                  **verdict(ls, guard), 'warnings': ['Exploratory multiple lenses: no multiplicity adjustment; public reading arbitrates.',
                  'Synovitis/PF OA report-ruler gains need image-read review; Gold58 cannot select alone.',
                  'Mechanical dispatch gate; B retains holistic judgment.']})


def make_context(ids, root=ROOT):
    raw = references(ids, root)
    refs = dict(zip(RULERS, [raw[k] for k in ('jev_hybrid', 'consensus', 'dread', 'aquila_v1')]))
    ctx = Context(ids, refs, json.loads((Path(root) / COMPONENTS).read_text()))
    ctx.bindings = {p: sha(Path(root)/p) for p in (COMPONENTS, JEV, AQUILA, DREAD, *CONSENSUS)}
    return ctx


def run_spec(spec, root=ROOT, context_cache=None):
    cache = {} if context_cache is None else context_cache
    arms, controls, provenance = {}, {}, {}
    ids = None
    for seed, pair in spec['seeds'].items():
        for side, dest in [('arm', arms), ('control', controls)]:
            roster, p, prov = load_oof(pair[side], root, spec.get('fold_rank', False))
            if ids is not None and ids != roster: raise ValueError('Paired study rosters differ (no silent intersection)')
            ids = roster; dest[seed] = p; provenance[f'{seed}/{side}'] = prov
    def context(roster):
        key = tuple(roster)
        if key not in cache: cache[key] = make_context(roster, root)
        return cache[key]
    bundle = None
    bundle_reason = None
    if spec.get('partners'):
        family = spec.get('family')
        partners = {}; partner_ids = {}
        for fam, files in spec['partners'].items():
            if fam == family: continue
            try: ri, p, pr = load_oof(files, root, spec.get('fold_rank', False))
            except FileNotFoundError as e:
                bundle_reason = 'Missing optional partner: ' + str(e); break
            partners[fam] = p; partner_ids[fam] = ri; provenance['partner/' + fam] = pr
        common = set(ids)
        for ri in partner_ids.values(): common &= set(ri)
        anchor = None
        if spec.get('anchor') and family != 'coatnet' and bundle_reason is None:
            try:
                ri, p, pr = load_oof(spec['anchor'], root)
                common &= set(ri); anchor = (ri, p); provenance['anchor'] = pr
            except FileNotFoundError as e: bundle_reason = 'Missing optional anchor: ' + str(e)
        shared = sorted(common)
        if shared and partners and bundle_reason is None:
            def align(ri, p):
                ix = {u: i for i, u in enumerate(ri)}
                return p[[ix[u] for u in shared]]
            fixed = [rank(align(partner_ids[f], p)) for f, p in partners.items()]
            a = align(ids, np.mean(list(arms.values()), axis=0)); c = align(ids, np.mean(list(controls.values()), axis=0))
            qa = np.mean([rank(a), *fixed], axis=0); qc = np.mean([rank(c), *fixed], axis=0)
            if family == 'coatnet':
                q = rank(np.mean(fixed, axis=0))
                qa, qc = (q + rank(a))/2, (q + rank(c))/2
            bundle = {'context': context(shared), 'arm': qa, 'control': qc,
                      'recipe': {'family_replaced': family, 'fixed_families': list(partners), 'Q': 'equal family percentile ranks', 'anchor_weight': .5 if anchor else None}}
            if anchor:
                ap = rank(align(*anchor)); bundle.update(anchor_arm=(rank(qa)+ap)/2, anchor_control=(rank(qc)+ap)/2)
    image_reads = {}
    for name, item in spec.get('image_reads', {}).items():
        ri, a, ap = load_oof(item['arm'], root); rc, c, cp = load_oof(item['control'], root)
        if ri != rc: raise ValueError('Image-read paired rosters differ')
        import pandas as pd
        lp = Path(root) / item['labels']; df = pd.read_csv(lp, usecols=[UID, *TARGETS], dtype={UID: str}).set_index(UID)
        if df.index.has_duplicates: raise ValueError('Duplicate image-read labels')
        y = df.loc[ri, list(TARGETS)].to_numpy(float); mask = np.isfinite(y)
        if not np.isin(y[mask], [0, 1]).all(): raise ValueError('Image-read labels must be binary or missing')
        if item.get('components') == 'gold58_singletons' and name == 'gold58':
            components = [{'component_id': u, 'members': [u]} for u in ri]
            policy = 'Gold58 study singletons: no validation component contract available; diagnostic only'
            component_hash = None
        else:
            components = json.loads((Path(root) / item['components']).read_text())
            policy = 'supplied connected components'; component_hash = sha(Path(root)/item['components'])
        image_reads[name] = dict(ids=ri, arm=a, control=c, labels=np.nan_to_num(y), mask=mask, components=components, eligible=item.get('eligible', False), component_policy=policy)
        provenance['image/' + name] = {'arm': ap, 'control': cp, 'labels_sha256': sha(lp), 'components_sha256': component_hash}
    out = evaluate(context(ids), arms, controls, spec.get('label_change', False), bundle, image_reads)
    if bundle_reason: out['lenses']['L6'] = unavailable(bundle_reason)
    out.update(name=spec.get('name', 'pair'), old_verdict=spec.get('old_verdict', 'not supplied'), provenance=provenance,
               evaluation_bindings=getattr(context(ids), 'bindings', {}),
               ordered_roster_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=VERSION)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--spec', type=Path, help='JSON: seeds:{seed:{arm:[OOFs],control:[OOFs]}}, optional partners/family/anchor/image_reads')
    parser.add_argument('--arm', type=Path); parser.add_argument('--control', type=Path)
    parser.add_argument('--out', type=Path, help='Result JSON file for a pair, output directory for --retro')
    parser.add_argument('--retro', action='store_true')
    args = parser.parse_args()
    if args.retro:
        from tools.multi_lens_retro import run
        run(args.root, args.out)
    else:
        if args.spec: spec = json.loads(args.spec.read_text(encoding='utf-8'))
        elif args.arm and args.control: spec = {'seeds': {'single': {'arm': str(args.arm), 'control': str(args.control)}}}
        else: parser.error('Use --retro, --spec, or --arm and --control')
        result = run_spec(spec, args.root); write(args.out or args.root/'artifacts/experiments/MULTI_LENS_RETRO/pair-result.json', result)
        print(json.dumps({k: result[k] for k in ('name', 'verdict', 'decision_label', 'passing_lenses')}))

if __name__ == '__main__': main()
