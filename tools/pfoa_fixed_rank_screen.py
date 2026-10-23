"""Prepare then score ONE frozen90/10 rank blend on EXP078 F0/F1. CPU only.

Passive report/metric adapters only: no training/acquisition imports, model loads,
Gold/LB scoring, weight parameter, target routing, or automatic expansion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_ab_report as primary
from input_ab_diagnostics import target_pair

ROOT, UID, LAB = primary.ROOT, primary.UID, primary.LAB
SCHEMA = 'pfoa-fixed-rank-screen-v1'
RANK_RULE = 'within each validation fold and target, average ties, rank/n'
FIXED = {'baseline': .9, 'resnet': .1, 'folds': [0, 1], 'studies_per_fold': 869,
         'targets': LAB, 'rank_rule': RANK_RULE, 'checkpoint': 'best-only'}
BACKENDS = {'cudnn_benchmark': False, 'cudnn_deterministic': True,
            'cudnn_allow_tf32': False, 'matmul_allow_tf32': False,
            'float32_matmul_precision': 'highest', 'deterministic_algorithms': True,
            'cublas_workspace_config': ':4096:8'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def uri(path):
    return 'workspace://' + Path(path).absolute().relative_to(ROOT.absolute()).as_posix()


def unuri(value):
    if not value.startswith('workspace://'):
        raise ValueError('Expected workspace URI')
    path = ROOT / value.removeprefix('workspace://')
    if '..' in path.relative_to(ROOT).parts:
        raise ValueError('Path escapes workspace')
    return path


def validate_predictions(frame, ids):
    if (list(frame.columns) != [UID, *LAB] or frame[UID].isna().any()
            or not frame[UID].is_unique or set(frame[UID]) != set(ids)):
        raise ValueError('Exact study/12-target contract mismatch')
    values = frame.set_index(UID).loc[ids, LAB].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid probabilities')
    return values


def fixed_rank_blend(baseline, resnet):
    if baseline.shape != resnet.shape or baseline.ndim != 2 or baseline.shape[1] != 12:
        raise ValueError('Expected matching n by12 predictions')
    for values in (baseline, resnet):
        if not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid probabilities')
    a, b = [pd.DataFrame(v).rank(axis=0, method='average', pct=True).to_numpy()
            for v in (baseline, resnet)]
    return a, b, .9*a + .1*b


def validate_runtime_receipts(receipts):
    if len(receipts) != 8:
        raise ValueError('Require train/eval receipts for all four fits')
    for receipt in receipts:
        if any(receipt['backends'].get(k) != v for k, v in BACKENDS.items()):
            raise ValueError('Strict runtime backend mismatch')
    if any(r != receipts[0] for r in receipts[1:]):
        raise ValueError('Runtime receipts differ across four fits')


def validate_primary(manifest, report, diagnostics, manifest_sha, primary_sha,
                     selected_arm, target_review_complete):
    if selected_arm not in ('A', 'B') or not target_review_complete:
        raise ValueError('Explicit arm and completed target review are required')
    keys = [(r['fold'], r['arm']) for r in manifest['runs']]
    if (manifest['folds'] != [0, 1] or sorted(keys) != [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B')]
            or len({r['tag'] for r in manifest['runs']}) != 4):
        raise ValueError('Require exactly four unique EXP078 runs')
    if (report['manifest_sha256'] != manifest_sha or diagnostics['manifest_sha256'] != manifest_sha
            or diagnostics['primary_report_sha256'] != primary_sha):
        raise ValueError('Stale primary/diagnostics protocol binding')
    if (sorted(r['fold'] for r in report['folds']) != [0, 1]
            or sorted(r['fold'] for r in diagnostics['folds']) != [0, 1]
            or report['applied_gate'] != manifest['gate']
            or diagnostics['primary_gate_unchanged'] != manifest['gate']):
        raise ValueError('Incomplete or changed primary gate')
    expected = primary.apply_gate(report['folds'], manifest['gate'])
    if any(report.get(k) != v for k, v in expected.items()) or not expected['numeric_gate']:
        raise ValueError('Primary gate result inconsistent or numeric gate failed')
    if selected_arm == 'B' and not expected['development_screen']:
        raise ValueError('B did not meet the original primary screen')


def collect(inputs, selected_arm, target_review_complete):
    """Validate artifacts/receipts without computing any AUC or blend."""
    paths = {key: unuri(value) for key, value in inputs.items()}
    hashes = {}

    def bind(path, expected=None):
        digest = primary.sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError('Changed artifact: ' + uri(path))
        hashes[uri(path)] = digest

    for path in paths.values():
        bind(path)
    manifest, report, diagnostics, prior = [read(paths[k]) for k in ('manifest', 'primary', 'diagnostics', 'prior')]
    validate_primary(manifest, report, diagnostics, hashes[inputs['manifest']], hashes[inputs['primary']],
                     selected_arm, target_review_complete)
    labels = primary.validate_sources(manifest)
    for relative, digest in manifest['source_file_sha256'].items():
        bind(ROOT / relative, digest)
    bind(ROOT / manifest['corpus'] / 'manifest.json', manifest['corpus_manifest_sha256'])
    for name, digest in manifest['training_code_sha256'].items():
        bind(ROOT / 'training/src' / name, digest)
    for name in ('pfoa_fixed_rank_screen.py', 'input_ab_report.py', 'input_ab_diagnostics.py', 'oof_error_audit.py'):
        bind(ROOT / 'tools' / name)
    split = pd.read_csv(ROOT / manifest['fold_file'], dtype={UID: str}).set_index(UID)
    if (not split.index.is_unique or split.index.isna().any()
            or split.report_group_sha256.isna().any()
            or split.groupby('report_group_sha256').fold.nunique().max() > 1):
        raise ValueError('Invalid split/groups')
    audited = {x['path']: x['sha256'] for x in prior['sources']}
    if prior['reference']['canonical_label_hash'] != manifest['label_hash'] or prior['reference']['split_hash'] != manifest['split_hash']:
        raise ValueError('Historical ResNet reference differs')
    diagnosed = {x['path']: x['sha256'] for x in diagnostics['input_hashes']}
    data, receipts = [], []
    for fold in (0, 1):
        ids = sorted(set(split.index[split.fold == fold]) - set(manifest['excluded_uids']))
        if len(ids) != 869 or manifest['expected_ids'][str(fold)]['val'] != 869:
            raise ValueError('Expected exactly869 studies per fold')
        from raptor_common import sha256_text
        id_hash = sha256_text('\n'.join(ids))
        if id_hash != manifest['expected_ids'][str(fold)]['eval_ids_sha256']:
            raise ValueError('Actual evaluation UID hash differs')
        if split.loc[ids, ['is_gold', 'quarantined_gold_report_duplicate']].to_numpy().any():
            raise ValueError('Gold/quarantine overlap')
        y = np.stack([labels[u] for u in ids])
        if not np.isfinite(y).all() or any(len(np.unique(np.rint(y[:, j]))) != 2 for j in range(12)):
            raise ValueError('Invalid or one-class12-target reference')
        arms = {}
        for arm in ('A', 'B'):
            run = next(r for r in manifest['runs'] if r['fold'] == fold and r['arm'] == arm)
            folder = ROOT / 'artifacts/experiments' / run['tag']
            summary_path, prediction_path = folder/'summary.json', folder/'oof_best.csv'
            for path in (summary_path, prediction_path):
                if uri(path) not in diagnosed:
                    raise ValueError('Missing diagnostic source hash')
                bind(path, diagnosed[uri(path)])
            summary = read(summary_path)
            primary.validate_summary(summary, manifest, run)
            observed = next(r for r in report['folds'] if r['fold'] == fold)
            skip_fraction = summary['skipped_updates'] / (summary['optimizer_updates'] + summary['skipped_updates'])
            if skip_fraction != observed['skipped_update_fraction'][arm]:
                raise ValueError('Optimizer receipt differs from primary gate')
            for field in ('runtime', 'eval_runtime'):
                receipt = summary['provenance'][field]
                receipts.append(receipt)
            arms[arm] = validate_predictions(pd.read_csv(prediction_path, dtype={UID: str}), ids)
        rn_folder = ROOT / 'artifacts/experiments' / f'EXP071_R34_LR_F{fold}'
        for name in ('summary.json', 'oof_best.csv'):
            path = rn_folder/name
            key = path.relative_to(ROOT).as_posix()
            if key not in audited:
                raise ValueError('ResNet file absent from verified prior')
            bind(path, audited[key])
        summary = read(rn_folder/'summary.json')
        if (summary['fold'] != fold or summary['label_hash'] != manifest['label_hash']
                or summary['split_hash'] != manifest['split_hash'] or summary['args']['select'] != 'oof'
                or summary['args']['arch'] != 'resnet34.a1_in1k' or summary['rows'] != 869):
            raise ValueError('Historical ResNet contract mismatch')
        rn = validate_predictions(pd.read_csv(rn_folder/'oof_best.csv', dtype={UID: str}), ids)
        data.append({'fold': fold, 'ids': ids, 'eval_ids_sha256': id_hash, 'labels': y,
                     'arms': arms, 'resnet': rn})
    validate_runtime_receipts(receipts)
    for key, digest in hashes.items():
        if primary.sha256_file(unuri(key)) != digest:
            raise ValueError('Input changed during validation')
    return data, hashes, report


def prepare(manifest, primary_path, diagnostics, prior, arm, reason, reviewed, out):
    if Path(out).exists():
        raise FileExistsError(out)
    if not reason.strip():
        raise ValueError('Record the original primary-gate arm decision')
    inputs = {k: uri(v) for k, v in dict(manifest=manifest, primary=primary_path, diagnostics=diagnostics, prior=prior).items()}
    data, hashes, _ = collect(inputs, arm, reviewed)
    plan = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'FROZEN_BEFORE_BLEND_SCORING', 'fixed': FIXED, 'inputs': inputs,
            'selected_arm': arm, 'selection_reason': reason, 'target_review_complete': reviewed,
            'source_hashes': hashes, 'eval_ids_sha256': {str(d['fold']): d['eval_ids_sha256'] for d in data},
            'scope': 'Exposed weak-label development screen; historical best-only ResNet with different recipe/device/AMP. Not clinical, private or cross-GPU parity validation.'}
    write_new(out, plan)
    return {'plan': str(out), 'plan_sha256': primary.sha256_file(Path(out)), 'metrics_computed': False}


def score_fold(y, baseline, resnet):
    rank_a, rank_b, blend = fixed_rank_blend(baseline, resnet)
    rows = target_pair(y, rank_a, blend)
    candidate = target_pair(y, rank_a, rank_b)
    if any(row['delta'] is None for row in rows):
        raise ValueError('One-class target cannot produce macro12')
    for row, rn in zip(rows, candidate):
        row['resnet_auc'] = rn['auc_b']
    return {'macro_baseline': float(np.mean([r['auc_a'] for r in rows])),
            'macro_blend': float(np.mean([r['auc_b'] for r in rows])),
            'macro_delta': float(np.mean([r['delta'] for r in rows])),
            'pfoa_delta': next(r['delta'] for r in rows if r['target'] == 'PF OA'),
            'other11_delta': float(np.mean([r['delta'] for r in rows if r['target'] != 'PF OA'])),
            'targets': rows}


def run(plan_path, plan_sha256, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if primary.sha256_file(Path(plan_path)) != plan_sha256:
        raise ValueError('Frozen plan hash mismatch')
    plan = read(plan_path)
    if plan['schema'] != SCHEMA or plan['status'] != 'FROZEN_BEFORE_BLEND_SCORING' or plan['fixed'] != FIXED:
        raise ValueError('Fixed screen contract changed')
    created_at = datetime.fromisoformat(plan['created_at'])
    if created_at.tzinfo is None or created_at >= datetime.now(timezone.utc):
        raise ValueError('Plan timestamp must precede scoring')
    for key, digest in plan['source_hashes'].items():
        if primary.sha256_file(unuri(key)) != digest:
            raise ValueError('Frozen source changed: ' + key)
    data, hashes, report = collect(plan['inputs'], plan['selected_arm'], plan['target_review_complete'])
    if hashes != plan['source_hashes'] or {str(d['fold']): d['eval_ids_sha256'] for d in data} != plan['eval_ids_sha256']:
        raise ValueError('Frozen inputs differ')
    # Reserve a new result directory only after validation; the plan is retained on failure.
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'analysis_plan.json', plan)
    for data_fold in data:
        fold, y, arms = data_fold['fold'], data_fold['labels'], data_fold['arms']
        observed = next(r for r in report['folds'] if r['fold'] == fold)
        for arm, key in (('A', 'macro_a'), ('B', 'macro_b')):
            macro, targets = primary.macro_auc(y, arms[arm])
            if len(targets) != 12 or abs(macro-observed[key]) > 1e-10:
                raise ValueError('Saved predictions differ from primary scores')
    results = []
    for data_fold in data:
        fold, y, arms = data_fold['fold'], data_fold['labels'], data_fold['arms']
        results.append({'fold': fold, 'studies': len(y), 'eval_ids_sha256': data_fold['eval_ids_sha256'],
                        **score_fold(y, arms[plan['selected_arm']], data_fold['resnet'])})
    result = {'schema': SCHEMA, 'observed_at': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': plan_sha256, 'plan_created_at': plan['created_at'], 'selected_arm': plan['selected_arm'],
              'fixed': FIXED, 'source_hashes': hashes, 'folds': results,
              'mean_fold_delta': float(np.mean([r['macro_delta'] for r in results])),
              'mean_fold_pfoa_delta': float(np.mean([r['pfoa_delta'] for r in results])),
              'mean_fold_other11_delta': float(np.mean([r['other11_delta'] for r in results])),
              'decision': 'Review only; no automatic fit, target routing, ratio search or submission.',
              'caveats': [plan['scope'], 'Within-fold ranks depend on this evaluation cohort. No pooled ranking or hidden-test transfer claim.',
                          'EXP078 arm selection and prior ResNet best-epoch selection reused exposed folds; no independent effect estimate or externally attested preregistration.',
                          'Historical ResNet used200-row OOF epoch subsampling and included the two subsequently quarantined training studies.',
                          'Historical ResNet uses20epochs/LR5e-4/T4FP16; old R1 used8epochs/LR3e-5/4070BF16. This screen is not an architecture-only test.']}
    write_new(out/'report.json', result)
    pd.DataFrame([{'fold': r['fold'], **t} for r in results for t in r['targets']]).to_csv(out/'targets.csv', index=False)
    return {k: result[k] for k in ('selected_arm', 'mean_fold_delta', 'mean_fold_pfoa_delta', 'mean_fold_other11_delta')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    for option in ('manifest', 'primary', 'diagnostics', 'prior', 'selection-reason', 'out'):
        prep.add_argument('--'+option, required=True)
    prep.add_argument('--selected-arm', choices=['A', 'B'], required=True)
    prep.add_argument('--target-review-complete', action='store_true', required=True)
    execute = commands.add_parser('run')
    for option in ('plan', 'plan-sha256', 'out'):
        execute.add_argument('--'+option, required=True)
    a = p.parse_args()
    result = prepare(a.manifest, a.primary, a.diagnostics, a.prior, a.selected_arm,
                     a.selection_reason, a.target_review_complete, a.out) if a.command == 'prepare' else run(a.plan, a.plan_sha256, a.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
