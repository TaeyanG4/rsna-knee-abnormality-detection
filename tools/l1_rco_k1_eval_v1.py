"""H-B-182 K1 v1.1.0: registered CPU-only cohort evaluator; never print rows.

Reuses fable's exact exp156 references(), including consensus masks and rounding.
Changelog v1.1.0: approved exact976 primary / all977 output; paired study
jackknife normal-approximation probability and signed direction per ruler.
v1.0.0: exact977 comparison (superseded before admission or candidate output).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp156_k_sweep as rulers
from tools.l1_rco_stage0_v1 import write, replace

VERSION = '1.1.0'
WORK = ROOT / '.local/scratch/l1-rco/k1'
TEACHER = ROOT / '.local/scratch/astra-b4/H-B-146/launch-v115/teacher-f0.csv'
TEACHER_RECEIPT = TEACHER.with_suffix('.json')
FABLE = ROOT / '.local/scratch/slot-b/routing-1016/fable/cohort977_compact_vs_own.py'
Q12B_REG = ROOT / '.local/scratch/slot-b/q12b/registration.json'
TARGETS = tuple(rulers.TARGETS)
UNCERTAINTY = dict(method='paired delete-one-study jackknife on each ruler macro AUC delta',
    deletion='same study deleted from both models and every target/ruler; original ruler masks retained',
    se='sqrt((n-1)/n * sum((delete_one_macro_delta - mean(delete_one_macro_delta))**2))',
    ci95='point macro delta +/- 1.959963984540054 * jackknife SE',
    p_delta_positive='Phi(point macro delta / SE); zero SE uses 1, 0.5, 0 for positive, zero, negative delta',
    direction='sign of point delta, regardless of CI overlap with zero',
    assumptions='studies are independent sampling units; shared paired deletion retains target/ruler/model dependence',
    interpretation='descriptive normal approximation, not a Bayesian posterior or a change to the KEEP rule',
    undefined='report probability/CI unavailable when any leave-one-study macro is undefined; never omit targets',
    seed=None, resampling='deterministic exact deletion; no bootstrap')
RULE = dict(cohort_n=976, inference_n=977, auc='unweighted mean of all 12 per-target AUCs per ruler',
            margin=0.005, rulers_required=3, ruler_count=4,
            spearman='unweighted mean of all 12 per-target Spearman correlations; average ties',
            spearman_max=0.93, missing='exact registered976 primary and all977 RCO outputs required; no further shrink or target omission')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ids_digest(ids):
    # Identical canonical JSON serialization to hb167_common_gold58_v1.digest.
    return hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def excluded_metadata(path, missing):
    """Sanitized operational metadata only; never expose UID, predictions or free text."""
    fields = ('input_series', 'rendered_series', 'used_series', 'dropped_series',
              'skipped_series', 'dicom_files', 'decoded_files', 'decode_failures')
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = {row['StudyInstanceUID']: row for row in csv.DictReader(stream) if row['StudyInstanceUID'] in missing}
    results = []
    for uid in missing:
        row = rows.get(uid)
        result = dict(id_sha256=hashlib.sha256(uid.encode()).hexdigest(),
                      reason='R7 row fails registered status=ok eligibility' if row else 'R7 row absent',
                      root_cause='unknown; operational status does not establish archive corruption')
        if row:
            result['status'] = row['status'] if row['status'] in ('ok', 'dropped series') else 'other non-ok status'
            result['metadata'] = {name: int(row[name]) for name in fields if row.get(name, '').isdigit()}
            result['error_present'] = bool(row.get('error', '').strip())
        results.append(result)
    return results


def cohort_registration(holdout):
    ids = list(holdout)
    if len(ids) != 977 or len(set(ids)) != 977 or ids != sorted(ids):
        raise ValueError('Exact sorted unique 977 roster required')
    if any(int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 != 0 for u in ids):
        raise ValueError('Non-fold-0 holdout')
    available, _ = predictions(TEACHER, ids, teacher=True)
    q = json.loads(Q12B_REG.read_text(encoding='utf-8'))
    if len(available) != 976 or available != q['ids']:
        raise ValueError('Primary cohort must equal exact Q12b976 and valid R7 intersection')
    missing = sorted(set(ids) - set(available))
    return dict(ids=q['ids'], id_list_sha256=ids_digest(q['ids']), n=976,
                serialization='UTF-8 json.dumps(ids, sort_keys=True, separators=(comma,colon), allow_nan=False); no newline',
                q12b_registration=str(Q12B_REG), q12b_registration_sha256=sha(Q12B_REG), q12b_equal=True,
                missing=excluded_metadata(TEACHER, missing))


def evaluator_registration(holdout):
    receipt = json.loads(TEACHER_RECEIPT.read_text(encoding='utf-8'))
    if not (receipt['csv_sha256'] == sha(TEACHER) and receipt['complete'] is True
            and receipt['mode'] == 'screen_f0' and receipt['model_selection'] == 'f0_only'
            and receipt['output_kind'] == 'pre_stacker' and receipt['clean_oof'] is False):
        raise ValueError('R7-f0 authenticated receipt/identity mismatch')
    paths = [Path(__file__), Path(rulers.__file__), ROOT / 'tools/exp115_vlm_eval.py', FABLE, Q12B_REG,
             TEACHER, TEACHER_RECEIPT, *[ROOT / p for p in sorted(set((rulers.JEV, rulers.AQUILA, rulers.DREAD, *rulers.CONSENSUS)))]]
    return dict(version=VERSION, rule=RULE, uncertainty=UNCERTAINTY, primary=cohort_registration(holdout), teacher=str(TEACHER),
                pins={str(p): sha(p) for p in paths},
                exposure='EXPOSED_EXPLORATORY; report rulers are not independent clean labels; R7 pooled-prior exposure')


def auc(y, scores):
    # Byte-for-byte arithmetic and tie convention from fable's auc().
    import numpy as np
    from scipy.stats import rankdata
    y = np.asarray(y, bool); r = rankdata(scores); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_delete_one(y, scores):
    """Exact tie-aware AUC after each observation deletion, without repeated sorting."""
    import numpy as np
    y, scores = np.asarray(y, bool), np.asarray(scores, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    point = float(auc(y, scores)); result = np.full(len(y), np.nan)
    if not np.isfinite(point): return result
    positive, negative = np.sort(scores[y]), np.sort(scores[~y])
    if n1 > 1:
        left = np.searchsorted(negative, scores[y], side='left')
        right = np.searchsorted(negative, scores[y], side='right')
        result[y] = (point*n1*n0 - (left+right)/2) / ((n1-1)*n0)
    if n0 > 1:
        left = np.searchsorted(positive, scores[~y], side='left')
        right = np.searchsorted(positive, scores[~y], side='right')
        result[~y] = (point*n1*n0 - (n1-(left+right)/2)) / (n1*(n0-1))
    return result


def normal_summary(delta, leave_one):
    import numpy as np
    from statistics import NormalDist
    values = np.asarray(leave_one, float)
    direction = 'positive' if delta > 0 else 'negative' if delta < 0 else 'zero' if delta == 0 else 'undefined'
    result = dict(direction=direction, uncertainty=UNCERTAINTY, jackknife_studies=len(values))
    if len(values) < 2 or not np.isfinite(values).all() or not np.isfinite(delta):
        return dict(result, uncertainty_status='UNDEFINED', jackknife_se=None, ci95=None, p_delta_positive_normal=None)
    # Center near zero first: identical decimal leaves must have exactly zero SE,
    # not artificial variance from rounding their nonzero arithmetic mean.
    centered = values-values[0]
    se = float(np.sqrt((len(values)-1)/len(values) * np.sum((centered-centered.mean())**2)))
    p = float(NormalDist().cdf(delta/se)) if se else (1.0 if delta > 0 else 0.0 if delta < 0 else .5)
    return dict(result, uncertainty_status='DEFINED', jackknife_se=se,
                ci95=[float(delta-1.959963984540054*se), float(delta+1.959963984540054*se)],
                p_delta_positive_normal=p)


def metrics(a, b, refs):
    import numpy as np
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 12 or not np.isfinite([a, b]).all():
        raise ValueError('Prediction shape/nonfinite mismatch')
    if set(refs) != {'jev_hybrid', 'aquila_v1', 'dread', 'consensus'}:
        raise ValueError('Exact four rulers required')
    rho = [float(spearmanr(a[:, j], b[:, j]).statistic) for j in range(12)]
    table, means = [], {}
    valid = bool(np.isfinite(rho).all())
    for name, (y, mask, _weight) in refs.items():
        y, mask = np.asarray(y), np.asarray(mask)
        if y.shape != a.shape or mask.shape != a.shape or mask.dtype != bool or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('Invalid ruler values/shape/mask')
        aa, bb, leave_one = [], [], []
        for j, target in enumerate(TARGETS):
            m = mask[:, j]
            x, z = float(auc(y[m, j], a[m, j])), float(auc(y[m, j], b[m, j]))
            aa.append(x); bb.append(z)
            deleted = np.full(len(a), x-z)
            deleted[m] = auc_delete_one(y[m, j], a[m, j]) - auc_delete_one(y[m, j], b[m, j])
            leave_one.append(deleted)
            table.append(dict(ruler=name, target=target, n=int(m.sum()), rco_auc=x, r7_auc=z, delta=x-z, spearman=rho[j]))
        finite = bool(np.isfinite([aa, bb]).all()); valid &= finite
        ma, mb = float(np.mean(aa)), float(np.mean(bb))
        means[name] = dict(rco=ma, r7=mb, delta=ma-mb, pass_=finite and ma >= mb-RULE['margin'],
                           **normal_summary(ma-mb, np.mean(leave_one, axis=0)))
    mean_rho = float(np.mean(rho))
    votes = sum(x['pass_'] for x in means.values())
    decision = ('KEEP' if votes >= RULE['rulers_required'] and mean_rho <= RULE['spearman_max'] else 'DROP') if valid else 'HOLD'
    return dict(decision=decision, n=len(a), rulers=means, rulers_passed=votes,
                spearman_mean=mean_rho, per_target=table, rule=RULE)


def predictions(path, ids, teacher=False):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    required = {'StudyInstanceUID', 'status', 'model_fold', *TARGETS}
    if teacher: required.add('fold_k')
    if not required.issubset(df.columns): raise ValueError('Prediction columns missing')
    if not teacher and (len(df) != len(ids) or set(df['StudyInstanceUID']) != set(ids)):
        raise ValueError('Candidate output must contain only the exact holdout roster')
    df = df[df['StudyInstanceUID'].isin(ids)].copy()
    if df['StudyInstanceUID'].duplicated().any(): raise ValueError('Duplicate prediction IDs')
    good = df['status'].eq('ok') & df['model_fold'].astype(str).eq('0')
    if teacher: good &= df['fold_k'].astype(str).eq('0')
    numeric = df[list(TARGETS)].to_numpy(dtype=float)
    good &= np.isfinite(numeric).all(axis=1) & ((numeric >= 0) & (numeric <= 1)).all(axis=1)
    df = df[good].set_index('StudyInstanceUID')
    present = [u for u in ids if u in df.index]
    return present, df.loc[present, list(TARGETS)].to_numpy(float)


def clean(obj):
    import math
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj): return None
    return obj


def evaluate(reg, dry_run=False):
    import numpy as np
    spec = reg['evaluator']
    if spec['rule'] != RULE or spec['version'] != VERSION: raise ValueError('Evaluator registration differs')
    for p, digest in spec['pins'].items():
        if sha(p) != digest: raise ValueError('Registered evaluator input changed: ' + str(p))
    # Never accept a registration with omitted pins or a substituted comparator.
    if spec != evaluator_registration(reg['holdout']): raise ValueError('Evaluator identity/pins differ')
    ids = list(reg['holdout'])
    if len(ids) != 977 or len(set(ids)) != 977: raise ValueError('Exact unique 977 roster required')
    if any(int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 != 0 for u in ids):
        raise ValueError('Non-fold-0 holdout')
    primary = spec['primary']; primary_ids = primary['ids']
    if (len(primary_ids) != 976 or len(set(primary_ids)) != 976 or primary_ids != sorted(primary_ids)
            or not set(primary_ids) < set(ids) or primary['id_list_sha256'] != ids_digest(primary_ids)):
        raise ValueError('Registered primary cohort roster/hash differs')
    refs = rulers.references(primary_ids)  # exact original fable source and exact Q12b cohort
    available, b = predictions(spec['teacher'], ids, teacher=True)
    if available != primary_ids: raise ValueError('Registered R7 primary intersection changed')
    base = dict(version=VERSION, contract_id=reg['contract_id'], cohort_n=976, inference_n=977,
                r7_valid=len(available), rule=RULE, exposure=spec['exposure'], dry_run=dry_run,
                primary_cohort_id_list_sha256=primary['id_list_sha256'], q12b_equal=primary['q12b_equal'],
                excluded_from_primary=primary['missing'], evaluator_pins=spec['pins'])
    if dry_run:
        # Exercise real masked labels and real comparator scores, without reading candidate outputs.
        diagnostic = metrics(b, b, refs)
        result = dict(base, decision='DRY_RUN', final_decision_eligible=False,
                      diagnostic_self_comparison=diagnostic,
                      blocker=None)
    else:
        fit = WORK / 'fit'
        receipt = json.loads((fit / 'result.json').read_text(encoding='utf-8'))
        pred = fit / 'holdout.csv'
        if receipt['contract_id'] != reg['contract_id'] or receipt['holdout_sha256'] != sha(pred):
            raise ValueError('Candidate receipt/hash differs')
        present, a = predictions(pred, ids)
        if present != ids: raise ValueError('Candidate requires 977 valid predictions')
        take = [ids.index(u) for u in primary_ids]
        result = dict(base, **metrics(a[take], b, refs), candidate_sha256=sha(pred), candidate_valid=977)
    out = WORK / ('evaluate-dry-run' if dry_run else 'evaluate')
    out.mkdir(parents=True, exist_ok=True)
    result = clean(result)
    write(out / 'result.json', result)
    rows = result.get('per_target', result.get('diagnostic_self_comparison', {}).get('per_target', []))
    fields = ['ruler', 'target', 'n', 'rco_auc', 'r7_auc', 'delta', 'spearman']
    with (out / 'per-target.tmp').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    replace(out / 'per-target.tmp', out / 'per-target.csv')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, default=WORK/'registration.json')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    result = evaluate(json.loads(args.registration.read_text(encoding='utf-8')), args.dry_run)
    print(json.dumps({k: result[k] for k in ('decision', 'cohort_n', 'r7_valid', 'dry_run')}))


if __name__ == '__main__': main()
