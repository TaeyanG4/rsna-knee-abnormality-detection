"""Check and compare only the preregistered input A/B runs. No training/expansion."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/src'))
from raptor_common import UID, LAB, ROOT, load_label_csv, load_folds, macro_auc, sha256_file


def validate_sources(manifest, root=ROOT):
    for relative, expected in manifest['source_file_sha256'].items():
        if sha256_file(root / relative) != expected:
            raise ValueError('Preregistered source changed: '+relative)
    if sha256_file(root / manifest['corpus'] / 'manifest.json') != manifest['corpus_manifest_sha256']:
        raise ValueError('Preregistered corpus manifest changed')
    labels, label_hash = load_label_csv(root / manifest['labels'])
    _, split_hash = load_folds(root / manifest['fold_file'])
    if label_hash != manifest['label_hash'] or split_hash != manifest['split_hash']:
        raise ValueError('Canonical label/split hash differs from preregistration')
    return labels


def validate_summary(summary, manifest, run):
    """Compare every run to the frozen protocol, not merely to the other arm."""
    p, args = summary['provenance'], summary['args']
    fixed = (summary['no_oof_epoch_selection'] and summary['no_gold_epoch_selection']
             and summary['selected_epoch_1based'] == manifest['epochs']
             and summary['selected_checkpoint'] == 'best'
             and p['checkpoint_selection'] == 'fixed-final'
             and set(summary['results']) == {'best'})
    if not fixed:
        raise ValueError('Checkpoint not selected by preregistered fixed epoch')
    expected = {**manifest['expected_args'], 'tag':run['tag'], 'fold':run['fold'],
                'window_policy':run['policy']}
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError('Argument differs from preregistration: '+key)
    for key, value in {'label_hash':manifest['label_hash'], 'eval_label_hash':manifest['label_hash'],
                       'split_hash':manifest['split_hash'], 'max_epochs':manifest['epochs'],
                       'window_policy':run['policy'], 'fold':run['fold'],
                       'backbone':expected['arch'], 'img':expected['res']}.items():
        if summary.get(key) != value:
            raise ValueError('Summary contract differs from preregistration: '+key)
    for key, value in {'code_sha256':manifest['training_code_sha256'],
                       'initial_state_sha256':manifest['initial_state_sha256'],
                       'corpus_manifest_sha256':manifest['corpus_manifest_sha256'],
                       'eval_label_hash':manifest['label_hash'],
                       'excluded_ids':sorted(manifest['excluded_uids'])}.items():
        if p.get(key) != value:
            raise ValueError('Provenance differs from preregistration: '+key)
    for key, value in manifest['expected_ids'][str(run['fold'])].items():
        if (p if key.endswith('_sha256') else summary).get(key) != value:
            raise ValueError('Study population differs from preregistration: '+key)
    if summary['rows'] != manifest['expected_ids'][str(run['fold'])]['val']:
        raise ValueError('Evaluation row count mismatch')
    if len(summary['history']) != manifest['epochs'] or any(
        h['epoch'] != i or h['oof_auc'] is not None or h['gold_auc'] is not None
        for i,h in enumerate(summary['history'])
    ):
        raise ValueError('Fixed-epoch run inspected labels during training')
    for field, amp in [('runtime', expected['amp']), ('eval_runtime', expected['eval_amp'])]:
        receipt = p[field]
        if (receipt['requested_amp'] != amp or receipt['effective_amp'] != amp
                or receipt['policy'] != expected['runtime_policy']
                or receipt['precision_fallback_allowed']):
            raise ValueError('Runtime differs from preregistration: '+field)
    if summary['optimizer_updates'] <= 0 or summary['skipped_updates'] < 0:
        raise ValueError('Invalid optimizer update counts')


def apply_gate(comparisons, gate):
    mean = float(np.mean([c['delta'] for c in comparisons]))
    numeric = all(max(c['skipped_update_fraction'].values()) <= gate['max_skipped_update_fraction'] for c in comparisons)
    target_review = any(min(c['per_target_delta'].values()) < gate['target_drop_review_below'] for c in comparisons)
    positive = not gate['both_fold_delta_positive'] or all(c['delta'] > 0 for c in comparisons)
    return {'mean_fold_delta':mean, 'numeric_gate':numeric,
            'development_screen':numeric and mean >= gate['mean_fold_delta_at_least'] and positive,
            'target_review_required':target_review}


def grouped_bootstrap(y, pa, pb, groups, reps=500, seed=20261003):
    """Resample entire report groups; invalid 12-target draws are discarded."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    delta = []
    for _ in range(reps):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        a, at = macro_auc(y[idx], pa[idx]); b, bt = macro_auc(y[idx], pb[idx])
        if len(at) == len(bt) == len(LAB):
            delta.append(b-a)
    if not delta:
        raise ValueError('No computable 12-target bootstrap draws')
    return {'unit':'report_group','groups':len(unique),'requested_reps':reps,'valid_reps':len(delta),
            'low':float(np.quantile(delta,.025)),'high':float(np.quantile(delta,.975)),
            'caveat':'Conditional on these fixed predictions/weak labels; not seed, label, selection or private-distribution uncertainty.'}


def mean_fold_bootstrap(whole, reps=500, seed=20261003):
    """Match the primary mean-of-fold-AUC estimand; sample groups within each fold."""
    rng = np.random.default_rng(seed)
    members = [[np.flatnonzero(g == x) for x in np.unique(g)] for _,_,_,g in whole]
    values = []
    for _ in range(reps):
        deltas = []
        for (y,pa,pb,_), group in zip(whole, members):
            idx = np.concatenate([group[i] for i in rng.integers(0,len(group),len(group))])
            ma,ta=macro_auc(y[idx],pa[idx]); mb,tb=macro_auc(y[idx],pb[idx])
            if len(ta) != len(LAB) or len(tb) != len(LAB):
                break
            deltas.append(mb-ma)
        if len(deltas) == len(whole):
            values.append(float(np.mean(deltas)))
    if not values:
        raise ValueError('No computable mean-fold 12-target bootstrap draws')
    return {'unit':'report_group_within_fold', 'estimand':'mean_fold_auc_delta',
            'requested_reps':reps,'valid_reps':len(values),
            'low':float(np.quantile(values,.025)), 'high':float(np.quantile(values,.975)),
            'caveat':'Conditional development interval; ignores seed, label, prior selection and private-distribution uncertainty.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    labels = validate_sources(manifest)
    folds = pd.read_csv(ROOT / manifest['fold_file'],dtype={UID:str}).set_index(UID)
    if not folds.index.is_unique or folds['report_group_sha256'].isna().any():
        raise ValueError('Duplicate study or missing report group in split')
    if folds.groupby('report_group_sha256')['fold'].nunique().max() > 1:
        raise ValueError('Report group crosses folds')
    if len(manifest['runs']) != 2*len(manifest['folds']) or len({r['tag'] for r in manifest['runs']}) != len(manifest['runs']):
        raise ValueError('Expected exactly two unique runs per fold')
    comparisons, whole = [], []
    for fold in manifest['folds']:
        runs = {r['arm']:r for r in manifest['runs'] if r['fold']==fold}
        if set(runs) != {'A','B'}:
            raise ValueError('Missing or duplicate A/B arm')
        s, frame = {}, {}
        for arm in ['A','B']:
            root = ROOT/'artifacts/experiments'/runs[arm]['tag']
            s[arm] = json.loads((root/'summary.json').read_text(encoding='utf-8'))
            validate_summary(s[arm], manifest, runs[arm])
            frame[arm] = pd.read_csv(root/'oof_best.csv',dtype={UID:str}).set_index(UID)
            expected = set(folds.index[(folds.fold==fold)]) - set(manifest['excluded_uids'])
            if not frame[arm].index.is_unique or set(frame[arm].index)!=expected or list(frame[arm].columns)!=LAB:
                raise ValueError('OOF study/target contract mismatch')
        for field in ['runtime','eval_runtime']:
            if s['A']['provenance'][field]!=s['B']['provenance'][field]:
                raise ValueError('Uncontrolled A/B difference: '+field)
        for field in ['label_hash','split_hash','max_epochs']:
            if s['A'][field]!=s['B'][field]:
                raise ValueError('Uncontrolled A/B difference: '+field)
        for field in ['amp','eval_amp','runtime_policy','seed','bs','accum','k','k_eval','bb_lr','head_lr','norm']:
            if s['A']['args'][field]!=s['B']['args'][field]:
                raise ValueError('Uncontrolled A/B argument: '+field)
        ids = sorted(frame['A'].index)
        y = np.stack([labels[u] for u in ids])
        pa,pb = [frame[arm].loc[ids,LAB].to_numpy() for arm in ['A','B']]
        if not np.isfinite(pa).all() or not np.isfinite(pb).all() or min(pa.min(),pb.min())<0 or max(pa.max(),pb.max())>1:
            raise ValueError('Nonfinite or out-of-range probabilities')
        ma,ta=macro_auc(y,pa); mb,tb=macro_auc(y,pb)
        if len(ta)!=12 or len(tb)!=12:
            raise ValueError('Not a 12-target comparison')
        per={target:tb[target]-ta[target] for target in LAB}
        groups=folds.loc[ids,'report_group_sha256'].to_numpy()
        skipped = {arm:s[arm]['skipped_updates']/max(1,s[arm]['optimizer_updates']+s[arm]['skipped_updates']) for arm in ['A','B']}
        comparisons.append({'fold':fold,'studies':len(ids),'macro_a':ma,'macro_b':mb,'delta':mb-ma,
                            'per_target_delta':per,'bootstrap':grouped_bootstrap(y,pa,pb,groups),
                            'skipped_update_fraction':skipped,
                            'runtime_seconds':{arm:s[arm]['total_seconds'] for arm in ['A','B']}})
        whole.append((y,pa,pb,groups))
    y,pa,pb,groups=[np.concatenate([w[i] for w in whole]) for i in range(4)]
    result={'manifest_sha256':sha256_file(Path(a.manifest)),'folds':comparisons,
            **apply_gate(comparisons,manifest['gate']), 'applied_gate':manifest['gate'],
            'mean_fold_bootstrap':mean_fold_bootstrap(whole),
            'pooled_delta':macro_auc(y,pb)[0]-macro_auc(y,pa)[0],
            'group_bootstrap_pooled':grouped_bootstrap(y,pa,pb,groups),
            'pooled_caveat':'Secondary pooled-AUC estimand differs from the primary mean-fold delta and can reflect between-fold score calibration.',
            'decision':'Review target/scanner effects and runtime before proposing any expansion. No automatic training or submission.',
            'caveat':'Exposed development folds and report-derived weak labels; no clinical/anchor/private improvement claim.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['mean_fold_delta','numeric_gate','development_screen','target_review_required']},ensure_ascii=False))


if __name__ == '__main__':
    main()
