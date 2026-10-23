"""Evaluate completed EXP079 strict20epoch ResNet F0/F1 against frozen EXP078 B.

One fixed90/10 within-fold rank blend; original A is descriptive only. No model,
training/acquisition import, Gold scoring, ratio search, selection or submission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_ab_report as primary
from input_ab_diagnostics import target_pair, target_intervals
from raptor_common import sha256_text

ROOT, UID, LAB = primary.ROOT, primary.UID, primary.LAB
SCHEMA = 'exp079-complement-report-v1'
BACKENDS = {'cudnn_benchmark': False, 'cudnn_deterministic': True, 'cudnn_allow_tf32': False,
            'matmul_allow_tf32': False, 'float32_matmul_precision': 'highest',
            'deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8'}
SHARED = ['training_code_sha256', 'labels', 'label_hash', 'fold_file', 'split_hash',
          'source_file_sha256', 'corpus', 'corpus_manifest_sha256', 'excluded_uids', 'expected_ids']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def uri(path):
    return 'workspace://' + Path(path).absolute().relative_to(ROOT.absolute()).as_posix()


def unuri(value):
    if not value.startswith('workspace://') or '..' in Path(value.removeprefix('workspace://')).parts:
        raise ValueError('Expected contained workspace URI')
    return ROOT / value.removeprefix('workspace://')


def validate_protocol(manifest, parent, parent_hash):
    if manifest['schema'] != 'exp079-r34-complement-v1' or manifest['parent_manifest_sha256'] != parent_hash:
        raise ValueError('Wrong EXP079 or parent manifest')
    if any(manifest[k] != parent[k] for k in SHARED) or manifest['eval_labels'] != parent['labels']:
        raise ValueError('Candidate changed shared data/code/UID contract')
    if parent['folds'] != [0, 1] or sorted((r['fold'], r['arm']) for r in parent['runs']) != [(0,'A'),(0,'B'),(1,'A'),(1,'B')]:
        raise ValueError('Require original four-fit parent')
    expected_runs = [(f, f'EXP079_R34_B_F{f}') for f in (0, 1)]
    expected_b = [(f, f'EXP078_INPUT_B_F{f}') for f in (0, 1)]
    if sorted((r['fold'], r['tag']) for r in manifest['runs']) != expected_runs or sorted((r['fold'], r['tag']) for r in manifest['baselines']) != expected_b:
        raise ValueError('Require exactly two new fits and two B baselines')
    if any(r['initial_state_sha256'] != manifest['pretrained']['initial_full_model_sha256'] for r in manifest['runs']):
        raise ValueError('Candidate initial states differ')
    if any(r['initial_state_sha256'] != parent['initial_state_sha256'] for r in manifest['baselines']):
        raise ValueError('B initial state changed')
    required = {'arch':'resnet34.a1_in1k', 'epochs':20, 'fixed_epoch':20,
                'window_policy':'within-slot-valid', 'checkpoint':'fixed-final',
                'amp':'fp16', 'eval_amp':'fp16', 'runtime_policy':'strict', 'device':'cuda'}
    if any(manifest['recipe'].get(k) != v for k,v in required.items()):
        raise ValueError('Not the fixed strict20epoch ResNet recipe')
    if manifest['evaluation']['fixed_global_rank_weights'] != {'baseline':.9, 'candidate':.1}:
        raise ValueError('Fixed90/10 weights changed')
    if manifest['evaluation']['ties'] != 'average rank/n within fold and target, no final re-rank':
        raise ValueError('Fixed rank policy changed')
    if (manifest['stop_review']['optimizer_skip_fraction'] != .01
            or manifest['stop_review']['consecutive_skips_review'] != 10
            or manifest['stop_review']['epoch_metric_early_stopping']):
        raise ValueError('Frozen numeric/early-stopping guard changed')
    for k in ('res','k','k_eval','seed','bs','accum','amp','eval_amp','runtime_policy','head_lr','wd'):
        if manifest['recipe'][k] != parent['expected_args'][k]:
            raise ValueError('Unregistered shared recipe difference: '+k)


def candidate_protocol(manifest, parent, run):
    """Adapter to the passive fixed-epoch summary validator; no train import."""
    protocol = deepcopy(parent)
    protocol.update({k: deepcopy(manifest[k]) for k in SHARED})
    recipe = {k:v for k,v in manifest['recipe'].items() if k not in ('checkpoint','window_policy')}
    protocol['expected_args'].update(recipe)
    protocol.update(epochs=20, initial_state_sha256=run['initial_state_sha256'])
    return protocol, {**run, 'arm':'candidate', 'policy':'within-slot-valid'}


def validate_run(summary, protocol, run, skip_limit):
    primary.validate_summary(summary, protocol, run)
    p = summary['provenance']
    if p['sampling'] != 'sha256(seed,epoch,uid,stream)-v1':
        raise ValueError('Stable sampling contract differs')
    for field in ('runtime','eval_runtime'):
        receipt = p[field]
        if receipt['device'] != 'cuda' or any(receipt['backends'].get(k) != v for k,v in BACKENDS.items()):
            raise ValueError('Strict device/backend mismatch')
    updated, skipped = summary['optimizer_updates'], summary['skipped_updates']
    def valid_count(value):
        return (isinstance(value,(int,float)) and not isinstance(value,bool)
                and np.isfinite(value) and value >= 0 and float(value).is_integer())
    if (not valid_count(updated) or not valid_count(skipped) or updated <= 0
            or not np.isfinite(skip_limit) or not 0 <= skip_limit <= 1):
        raise ValueError('Invalid optimizer counter/skip limit')
    if summary['best_epoch'] != protocol['epochs']-1:
        raise ValueError('Saved best epoch is not the fixed final epoch')
    fraction = skipped / (updated + skipped)
    if fraction > skip_limit:
        raise ValueError('Optimizer skip fraction exceeds review limit')
    old_u = old_s = 0
    for h in summary['history']:
        if (not np.isfinite(h['loss']) or not valid_count(h['optimizer_updates']) or not valid_count(h['skipped_updates'])
                or h['optimizer_updates'] < old_u or h['skipped_updates'] < old_s):
            raise ValueError('Nonfinite loss or nonmonotonic update history')
        old_u, old_s = h['optimizer_updates'], h['skipped_updates']
    if (old_u,old_s) != (updated,skipped):
        raise ValueError('Final update receipt differs from history')
    return fraction


def predictions(path, ids):
    frame = pd.read_csv(path, dtype={UID:str})
    if list(frame.columns) != [UID,*LAB] or frame[UID].isna().any() or not frame[UID].is_unique or set(frame[UID]) != set(ids):
        raise ValueError('Exact869 UID/12-target contract mismatch')
    values = frame.set_index(UID).loc[ids,LAB].to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid predictions')
    return values


def fixed_blend(baseline, candidate):
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[1] != 12 or not len(baseline):
        raise ValueError('Expected matching n by12 predictions')
    if any(not np.isfinite(v).all() or np.any((v < 0) | (v > 1)) for v in (baseline,candidate)):
        raise ValueError('Invalid probabilities')
    b,r = [pd.DataFrame(v).rank(axis=0, method='average', pct=True).to_numpy() for v in (baseline,candidate)]
    return b, .9*b + .1*r


def paired_rows(y, baseline, candidate, original_a):
    """Compare the SAME B+candidate blend to B and original A, never A+candidate."""
    b,blend = fixed_blend(baseline,candidate)
    vs_b, vs_a, rn = target_pair(y,b,blend), target_pair(y,original_a,blend), target_pair(y,b,candidate)
    rows = [{'target':x['target'], 'P':x['P'], 'N':x['N'], 'baseline_B_auc':x['auc_a'],
             'candidate_auc':z['auc_b'], 'original_A_auc':a['auc_a'], 'blend_auc':x['auc_b'],
             'delta_vs_B':x['delta'], 'delta_vs_A':a['delta'],
             'macro_contribution_vs_B':x['macro_contribution']} for x,a,z in zip(vs_b,vs_a,rn)]
    return rows, b, blend


def aggregate(rows):
    if any(r['delta_vs_B'] is None for r in rows):
        return {'macro12_delta_vs_B':None, 'macro12_delta_vs_A':None}
    return {'macro12_delta_vs_B':float(np.mean([r['delta_vs_B'] for r in rows])),
            'macro12_delta_vs_A':float(np.mean([r['delta_vs_A'] for r in rows])),
            'macro_baseline_B':float(np.mean([r['baseline_B_auc'] for r in rows])),
            'macro_candidate':float(np.mean([r['candidate_auc'] for r in rows])),
            'macro_blend':float(np.mean([r['blend_auc'] for r in rows])),
            'macro_original_A':float(np.mean([r['original_A_auc'] for r in rows])),
            'pfoa_delta_vs_B':next(r['delta_vs_B'] for r in rows if r['target']=='PF OA'),
            'other11_delta_vs_B':float(np.mean([r['delta_vs_B'] for r in rows if r['target']!='PF OA']))}


def load_inputs(manifest_path, manifest_sha256, parent_path, primary_path, diagnostics_path):
    hashes = {}
    def bind(path, expected=None):
        digest = primary.sha256_file(Path(path))
        if expected is not None and digest != expected:
            raise ValueError('Frozen source changed: '+uri(path))
        hashes[uri(path)] = digest
        return digest
    bind(manifest_path, manifest_sha256)
    manifest = read(manifest_path)
    parent_hash = bind(parent_path, manifest['parent_manifest_sha256']); parent = read(parent_path)
    validate_protocol(manifest,parent,parent_hash)
    bind(unuri(manifest['source_decision']['path']),manifest['source_decision']['sha256'])
    decision = read(unuri(manifest['source_decision']['path']))
    if decision['selected_arm'] != 'B' or decision['manifest_sha256'] != parent_hash:
        raise ValueError('B parent decision binding failed')
    bind(primary_path,decision['primary_report_sha256']); bind(diagnostics_path,decision['diagnostics_sha256'])
    report,diag = read(primary_path),read(diagnostics_path)
    if (report['manifest_sha256'] != parent_hash or diag['manifest_sha256'] != parent_hash
            or diag['primary_report_sha256'] != hashes[uri(primary_path)]
            or report['applied_gate'] != parent['gate'] or diag['primary_gate_unchanged'] != parent['gate']
            or sorted(x['fold'] for x in report['folds']) != [0,1] or sorted(x['fold'] for x in diag['folds']) != [0,1]):
        raise ValueError('Incomplete/stale parent report binding')
    gate = primary.apply_gate(report['folds'],parent['gate'])
    if any(report[k] != v for k,v in gate.items()) or not gate['development_screen']:
        raise ValueError('Original B primary gate failed')
    for field in ('cpu_screen','queue'):
        bind(unuri(manifest[field]['path']),manifest[field]['sha256'])
    labels = primary.validate_sources(manifest)
    for path,digest in manifest['source_file_sha256'].items(): bind(ROOT/path,digest)
    bind(ROOT/manifest['corpus']/'manifest.json',manifest['corpus_manifest_sha256'])
    for name,digest in manifest['training_code_sha256'].items(): bind(ROOT/'training/src'/name,digest)
    for name in ('exp079_complement_report.py','input_ab_report.py','input_ab_diagnostics.py','oof_error_audit.py'):
        bind(ROOT/'tools'/name)
    # Both new completed summaries/OOFs must exist before any data scoring can start.
    for run in manifest['runs']:
        for name in ('summary.json','oof_best.csv'): bind(ROOT/'artifacts/experiments'/run['tag']/name)
    split = pd.read_csv(ROOT/manifest['fold_file'],dtype={UID:str}).set_index(UID)
    if (not split.index.is_unique or split.index.isna().any() or split.report_group_sha256.isna().any()
            or split.groupby('report_group_sha256').fold.nunique().max()>1):
        raise ValueError('Invalid split/report groups')
    diagnosed = {v['path']:v['sha256'] for v in diag['input_hashes']}
    meta_path = ROOT/'storage/labels/derived_meta/study_meta.csv'; bind(meta_path,diagnosed[uri(meta_path)])
    meta = pd.read_csv(meta_path,dtype={UID:str}).set_index(UID)
    corpus = ROOT/manifest['corpus']
    for name in ('ids.npy','masks.npy'): bind(corpus/name,diagnosed[uri(corpus/name)])
    corpus_ids = np.load(corpus/'ids.npy').astype(str); masks = np.load(corpus/'masks.npy',mmap_mode='r')
    if not meta.index.is_unique or len(set(corpus_ids)) != len(corpus_ids) or len(corpus_ids) != len(masks):
        raise ValueError('Duplicate metadata/corpus IDs or mask count mismatch')
    missing = pd.Series(np.where((masks == 0).any(axis=1),'some_zero_mask','all_nonzero'),index=corpus_ids)
    data, receipts, environments, execution = [],[],[],[]
    for fold in (0,1):
        ids = sorted(set(split.index[split.fold==fold])-set(manifest['excluded_uids']))
        train_ids = sorted(set(split.index[(split.fold>=0)&(split.fold!=fold)])-set(manifest['excluded_uids']))
        expected = manifest['expected_ids'][str(fold)]
        if len(ids)!=869 or len(train_ids)!=expected['train'] or sha256_text('\n'.join(ids))!=expected['eval_ids_sha256'] or sha256_text('\n'.join(train_ids))!=expected['train_ids_sha256']:
            raise ValueError('Actual study sets differ from manifest')
        if split.loc[ids,['is_gold','quarantined_gold_report_duplicate']].to_numpy().any():
            raise ValueError('Gold/quarantine evaluation overlap')
        y = np.stack([labels[u] for u in ids])
        if not np.isfinite(y).all() or any(len(np.unique(y[:,j].round()))!=2 for j in range(12)):
            raise ValueError('Invalid or one-class12-target labels')
        values = {}
        runs = [(key,next(r for r in parent['runs'] if r['fold']==fold and r['arm']==key)) for key in ('A','B')]
        runs.append(('candidate',next(r for r in manifest['runs'] if r['fold']==fold)))
        for key,run in runs:
            folder=ROOT/'artifacts/experiments'/run['tag']; summary_path=folder/'summary.json'; oof_path=folder/'oof_best.csv'
            if key in ('A','B'):
                for path in (summary_path,oof_path): bind(path,diagnosed[uri(path)])
                if key=='B':
                    baseline=next(r for r in manifest['baselines'] if r['fold']==fold)
                    bind(summary_path,baseline['summary_sha256']); bind(oof_path,baseline['oof_sha256'])
                protocol,run_contract=parent,run
            else:
                protocol,run_contract=candidate_protocol(manifest,parent,run)
            summary=read(summary_path)
            fraction=validate_run(summary,protocol,run_contract,manifest['stop_review']['optimizer_skip_fraction'])
            if key in ('A','B') and fraction!=next(r for r in report['folds'] if r['fold']==fold)['skipped_update_fraction'][key]:
                raise ValueError('Parent optimizer receipt changed')
            receipts.extend([summary['provenance'][f] for f in ('runtime','eval_runtime')])
            environments.append({k:summary['environment'][k] for k in ('torch','cuda','cudnn','gpu','timm','numpy','pydicom','cv2')})
            execution.append({'tag':run['tag'],'total_seconds':summary['total_seconds'],'optimizer_updates':summary['optimizer_updates'],
                              'skipped_updates':summary['skipped_updates'],'skip_fraction':fraction,'fixed_epoch':protocol['epochs'],
                              'consecutive_skip_evidence':'Completed frozen training code raises at10; maximum consecutive count not separately recorded.'})
            values[key]=predictions(oof_path,ids)
        groups=meta.reindex(ids)[['scanner_group','field','manufacturer']].fillna('unknown')
        groups['cached_mask']=missing.reindex(ids)
        if groups.cached_mask.isna().any(): raise ValueError('Missing corpus mask group')
        data.append({'fold':fold,'ids':ids,'y':y,'values':values,'report_groups':split.loc[ids,'report_group_sha256'].to_numpy(),'groups':groups})
    if any(r!=receipts[0] for r in receipts[1:]) or any(e!=environments[0] for e in environments[1:]):
        raise ValueError('Uncontrolled baseline/candidate runtime/environment difference')
    for path,digest in hashes.items():
        if primary.sha256_file(unuri(path))!=digest: raise ValueError('Input changed during validation')
    return manifest,report,data,hashes,execution


def build(manifest_path, manifest_sha256, parent_path, primary_path, diagnostics_path, out):
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    manifest,parent_report,data,hashes,execution=load_inputs(manifest_path,manifest_sha256,parent_path,primary_path,diagnostics_path)
    if [d['fold'] for d in data]!=[0,1]: raise ValueError('Both completed folds required before scoring')
    out.mkdir(parents=True,exist_ok=False)
    plan={'schema':SCHEMA,'created_at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':manifest_sha256,
          'source_hashes':hashes,'weights':{'baseline':.9,'candidate':.1},'bootstrap_reps':500,'bootstrap_seed':20261003,
          'scope':'Mean-fold12-target fixed-rank comparison; original A descriptive; exposed weak labels; no selection.'}
    write_new(out/'analysis_plan.json',plan)
    # Recheck every reused primary A/B score before constructing any blend.
    for d in data:
        observed=next(x for x in parent_report['folds'] if x['fold']==d['fold'])
        for arm,key in (('A','macro_a'),('B','macro_b')):
            macro,targets=primary.macro_auc(d['y'],d['values'][arm])
            if len(targets)!=12 or abs(macro-observed[key])>1e-10: raise ValueError('Parent primary score mismatch')
    folds,group_rows,whole,whole_a=[],[],[],[]
    for d in data:
        v,y=d['values'],d['y']
        rows,b,blend=paired_rows(y,v['B'],v['candidate'],v['A'])
        folds.append({'fold':d['fold'],'studies':len(y),'targets':rows,**aggregate(rows)})
        whole.append((y,b,blend,d['report_groups'])); whole_a.append((y,v['A'],blend,d['report_groups']))
        for field in d['groups'].columns:
            vals=d['groups'][field].astype(str).to_numpy()
            for group in sorted(set(vals)):
                take=vals==group
                # Keep the full-fold frozen ranks when inspecting a subgroup.
                bb,aa,rn=target_pair(y[take],b[take],blend[take]),target_pair(y[take],v['A'][take],blend[take]),target_pair(y[take],b[take],v['candidate'][take])
                targets=[{'target':x['target'],'P':x['P'],'N':x['N'],'baseline_B_auc':x['auc_a'],'candidate_auc':z['auc_b'],
                          'original_A_auc':a['auc_a'],'blend_auc':x['auc_b'],'delta_vs_B':x['delta'],'delta_vs_A':a['delta'],
                          'macro_contribution_vs_B':x['macro_contribution']} for x,a,z in zip(bb,aa,rn)]
                group_rows.append({'fold':d['fold'],'field':field,'group':group,'studies':int(take.sum()),'small_group':int(take.sum())<40,
                                   'targets':targets,**aggregate(targets)})
    target_ci=target_intervals(whole,reps=500,seed=20261003); original_ci=target_intervals(whole_a,reps=500,seed=20261003)
    mean=float(np.mean([f['macro12_delta_vs_B'] for f in folds])); pfoa=float(np.mean([f['pfoa_delta_vs_B'] for f in folds]))
    result={'schema':SCHEMA,'observed_at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':manifest_sha256,
            'analysis_plan_sha256':primary.sha256_file(out/'analysis_plan.json'),'source_hashes':hashes,'folds':folds,'groups':group_rows,
            'mean_fold_delta':mean,'mean_fold_pfoa_delta':pfoa,'mean_fold_other11_delta':float(np.mean([f['other11_delta_vs_B'] for f in folds])),
            'mean_fold_target_delta':{t:float(np.mean([next(r['delta_vs_B'] for r in f['targets'] if r['target']==t) for f in folds])) for t in LAB},
            'mean_fold_macro_ci':primary.mean_fold_bootstrap(whole,reps=500,seed=20261003),'mean_fold_target_ci':target_ci,
            'original_A_descriptive':{'mean_fold_delta':float(np.mean([f['macro12_delta_vs_A'] for f in folds])),
                                      'mean_fold_macro_ci':primary.mean_fold_bootstrap(whole_a,reps=500,seed=20261003),'mean_fold_target_ci':original_ci},
            'fracture':{'folds':[{'fold':f['fold'],**next(t for t in f['targets'] if t['target']=='Fracture')} for f in folds],
                        'ci_vs_B':target_ci['Fracture'],'ci_vs_A':original_ci['Fracture']},
            'screen':{'both_fold_macro_positive':all(f['macro12_delta_vs_B']>0 for f in folds),'mean_pfoa_positive':pfoa>0,
                      'major_drop_review':[{'fold':f['fold'],'target':t['target'],'delta':t['delta_vs_B']} for f in folds for t in f['targets'] if t['delta_vs_B']<manifest['evaluation']['major_drop_review']],
                      'small_effect_rule':'No .002 cutoff; signs/conditional CI/cost and all-target/group losses require review; no automatic expansion.'},
            'execution':execution,'pretrained_provenance':manifest['pretrained'],
            'pretrained_validation':'Both saved full initial-state receipts must match the preregistered candidate digest; the upstream pretrained file SHA is retained provenance, not rehashed from a weight file by this CPU evaluator.',
            'decision':'Review only; no training, routing, ratio/epoch tuning or submission follows automatically.',
            'caveats':['Exposed development folds and report-derived labels, not independent clinical/private validation.',
                       'Paired report-group CI conditions on fixed predictions AND full-fold ranks; bootstrap does not rerank or include seed/label/selection/distribution uncertainty.',
                       'Target/group inspection has no multiplicity correction; intervals are descriptive.',
                       'Report groups are not verified patients; small or one-class groups remain visible; no partial-target macro12.',
                       'Baseline8epochs vs candidate20epochs compares practical recipes, not architecture-only effects.',
                       'Local strict receipt agreement does not establish Kaggle T4 parity or full hidden-serving behavior.',
                       'Gold final diagnostics may exist in training summaries; no Gold predictions/metrics are evaluated or selected by this report.']}
    write_new(out/'report.json',result)
    pd.DataFrame([{'fold':f['fold'],**t} for f in folds for t in f['targets']]).to_csv(out/'targets.csv',index=False)
    return {k:result[k] for k in ('mean_fold_delta','mean_fold_pfoa_delta','mean_fold_other11_delta','screen')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','manifest-sha256','parent-manifest','parent-primary','parent-diagnostics','out'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.manifest,args.manifest_sha256,args.parent_manifest,args.parent_primary,args.parent_diagnostics,args.out),ensure_ascii=False))
