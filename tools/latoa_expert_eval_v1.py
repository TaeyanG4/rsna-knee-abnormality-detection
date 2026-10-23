"""v1.0.3: v2 accepts exactly the preregistered seed-46 expert.

Changelog v1.0.3: rank the single expert probability; reject pair/wrong-seed
registrations and preserve the GO gate and other eleven baseline columns.

Changelog: real optimizer updates must equal the unchanged registered 7960;
require contained supervisor success. No route, roster or GO change.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from tools import latoa_expert_v1 as b
from tools import exp156_k_sweep as rulers
from tools.c85r7_rco_overlay_v1 import ranks

LAT = rulers.TARGETS.index('Lateral OA')
OTHER = [i for i in range(12) if i != LAT]
CHECKPOINT_PINS = {
    'v2': '2c3c033e25605b5fb8bcc08e0030bed081c726e10fc56d8da62524ac08d9638c',
    's0': '6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5',
    's1': '29fc2b2bbaa4e4e8ad341bba17e47c5110388db697458a8cbb9ff8d83fa9399b'}


def route(baseline, crop):
    baseline, crop = np.asarray(baseline), np.asarray(crop)
    b.need(baseline.shape == crop.shape and baseline.ndim == 2 and baseline.shape[1] == 12
           and baseline.dtype.kind == 'f' and np.isfinite(baseline).all()
           and np.isfinite(crop).all(), 'route shape/finite/dtype')
    out = baseline.copy()
    out[:, LAT] = .5*ranks(baseline)[:, LAT] + .5*ranks(crop)[:, LAT]
    b.need(out[:, OTHER].tobytes() == baseline[:, OTHER].tobytes(), 'other columns changed')
    return out


def metrics(baseline, candidate, refs):
    b.need(baseline.shape == candidate.shape and baseline.ndim == 2 and baseline.shape[1] == 12
           and np.isfinite(baseline).all() and np.isfinite(candidate).all(), 'metric inputs')
    b.need(set(refs) == set(b.RULERS), 'all four registered rulers required')
    result = {}
    for name in b.RULERS:
        y, mask, _ = refs[name]
        b.need(y.shape == mask.shape == baseline.shape and mask.dtype == bool, 'ruler shape')
        valid = mask[:, LAT]; labels = y[valid, LAT]
        b.need(np.isfinite(labels).all() and set(np.unique(labels)) == {0, 1}, 'nondegenerate binary ruler')
        a = float(roc_auc_score(labels, baseline[valid, LAT]))
        c = float(roc_auc_score(labels, candidate[valid, LAT]))
        result[name] = dict(n=int(valid.sum()), baseline_auc=a, routed_auc=c, delta=c-a)
    return dict(rulers=result, mean_delta=float(np.mean([v['delta'] for v in result.values()])),
        nonnegative_rulers=sum(v['delta'] >= 0 for v in result.values()),
        other_11_identical=baseline.dtype == candidate.dtype and
            baseline[:, OTHER].tobytes() == candidate[:, OTHER].tobytes())


def go_rule(m):
    if set(m.get('rulers', {})) != set(b.RULERS) or m.get('other_11_identical') is not True:
        return False
    ds = np.asarray([m['rulers'][n]['delta'] for n in b.RULERS], dtype=float)
    return bool(np.isfinite(ds).all() and (ds >= 0).sum() >= 3 and ds.mean() >= .006)


def load_baseline(ids):
    from tools.h216_pair_eval_v1 import load
    p = b.BASELINE; r = b.read(p/'pilot-receipt.json')
    b.need(r['status'] == 'COMPLETE' and
        r['preregistration_sha256'] == b.sha(p/'PREREGISTRATION.json') and
        r['review_sha256'] == b.sha(p/'REVIEW.json'), 'baseline incomplete or identity drift')
    review = b.read(p/'REVIEW.json')
    b.need(review.get('verdict') == 'PASS', 'baseline independent review missing')
    values = []
    for name, pin in CHECKPOINT_PINS.items():
        f = p/'preds'/f'{name}_s2.csv'; arm = r['outputs'][f'{name}_s2']
        b.need(arm['csv_sha256'] == b.sha(f) and arm['checkpoint_sha256'] == pin
               and arm['eval_stride'] == 2, 'baseline exact checkpoint/stride/CSV required')
        a = load(f, ids)
        b.need(a.shape == (977,12) and np.isfinite(a).all(), 'baseline exact977 finite')
        values.append(ranks(a))
    return ranks(np.mean(values, axis=0)), b.sha(p/'pilot-receipt.json')


def check_crop_identity(identity, seed, config_hash, prepared_hash, ids_hash):
    b.need(type(seed) is int and seed in b.SEEDS, 'unregistered expert seed')
    b.need(identity['experiment'] == 'LATOA_EXPERT_V1 OA crop expert fold0 confirmation' and
        identity['seed'] == seed and identity['fold'] == 0 and identity['epochs'] == 20 and
        identity['smoke'] is False and identity['engineering_only'] is False and
        identity['config_sha256'] == config_hash and identity['prepare_sha256'] == prepared_hash and
        identity['val_sha256'] == ids_hash and identity['clean_oof'] is False,
        'crop final identity differs')


def load_crop(seed, ids):
    b.need(type(seed) is int and seed in b.SEEDS, 'unregistered expert seed')
    from training.arms.mil_screen_hooks_v3 import torch
    p = b.OUT/'fits'/f'crop-s{seed}'; s = b.read(p/'summary.json'); r = b.read(p/'val.receipt.json')
    b.need(s['passed'] and s['full_shape'] and not s['tiny_model'] and not s['engineering_only']
           and s['steps'] == 7960 and s['counters']['updates'] == 7960, 'complete fixed-final scientific fit required')
    actual = b.exact_optimizer_steps(s['counters'], 7960)
    b.need(s.get('optimizer_steps') == actual and s.get('attempted_steps') == 7960,
           'actual optimizer steps differ')
    watch = b.read(b.OUT/f'watch-fit-s{seed}'/'receipt.json')
    b.need(watch.get('status') == 'PASS' and watch.get('exit_code') == 0 and
           watch.get('cleanup_confirmed') is True, 'watchdog failure cannot qualify')
    check_crop_identity(s['identity'], seed, b.sha(b.config_path(seed)), b.sha(b.OUT/'prepared.json'), b.uid_hash(ids))
    pin = b.sha(p/'final.pt')
    b.need(s['checkpoint_sha256'] == r['checkpoint_sha256'] == pin and
        r['csv_sha256'] == b.sha(p/'val.csv') and r['identity'] == s['identity'] and
        r['k'] == 36 and r['tta'] is True and r['engineering_only'] is False and
        r['rows'] == 977 and r['uid_sha256'] == b.uid_hash(ids), 'crop output receipt identity')
    ck = torch.load(p/'final.pt', map_location='cpu', weights_only=True)
    b.need(ck['epoch'] == 19 and ck['identity'] == s['identity'] and bool(ck['model']), 'fixed epoch20 checkpoint required')
    values = rulers.csv_values(p/'val.csv', ids)
    return values, dict(checkpoint_sha256=pin, receipt_sha256=b.sha(p/'val.receipt.json'),
                        summary_sha256=b.sha(p/'summary.json'),
                        watchdog_sha256=b.sha(b.OUT/f'watch-fit-s{seed}'/'receipt.json'))


def load_experts(ids, registration_path=None):
    reg = b.read(registration_path or b.OUT/b.PREREG)
    b.need(reg.get('schema') == 'latoa-expert/v2' and reg.get('version') == '1.0.3'
           and reg.get('seeds') == [46] and type(reg['seeds'][0]) is int
           and reg.get('rule') == b.RULE, 'v2 requires exactly one seed-46 expert')
    values, receipt = load_crop(46, ids)
    return values, {'46': receipt}


def evaluate():
    b.prepared(); b.require_fit(b.config_path(46))
    _, ids, _ = b.population()
    baseline, baseline_receipt = load_baseline(ids)
    values, receipts = load_experts(ids)
    candidate = route(baseline, values)
    m = metrics(baseline, candidate, rulers.references(ids, root=b.ROOT))
    result = dict(status='GO' if go_rule(m) else 'NO_GO', rule=b.RULE, metrics=m,
        n=977, fold=0, clean_oof=False, exposure='EXPOSED_EXPLORATORY',
        preregistration_sha256=b.sha(b.OUT/b.PREREG),
        prepared_sha256=b.sha(b.OUT/'prepared.json'), baseline_receipt_sha256=baseline_receipt,
        crops=receipts, qualified_for_submission=False, independent_result_review='REQUIRED')
    b.immutable(b.OUT/'RESULT.json', result)
    return result
