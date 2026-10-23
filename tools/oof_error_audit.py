"""CPU-only error decomposition of saved OOF; never train, infer, tune, or overwrite.

Uses the passive raptor_common label/metric adapter. All models use their existing
best checkpoint predictions, even when an old summary selected SWA. R2 remains
teacher-leaky, checkpoint-selected development evidence after reference rescoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/src'))
from raptor_common import ROOT, UID, LAB, load_label_csv, load_folds, macro_auc, sha256_file

PREFIXES = {'R1':'EXP061_R1', 'ResNet34':'EXP071_R34_LR', 'R2':'EXP072_R2MIX_R34'}
CAVEATS = {
    'R1':'OOF checkpoint selected on exposed report-label development folds; not untouched clinical validation.',
    'ResNet34':'OOF checkpoint selected on exposed report-label development folds; best-only here, not summary-selected SWA.',
    'R2':'Indirect teacher-to-student fold leakage; epochs selected against mixed labels. Original-label rescoring does not remove either bias.',
}


def weighted_auc(y, p, weights):
    """Tie-aware AUC for nonnegative row weights; equivalent to repeated rows."""
    order = np.argsort(p, kind='stable')
    y,p,w = y[order],p[order],weights[order]
    starts = np.r_[0,np.flatnonzero(np.diff(p))+1]
    pos = np.add.reduceat(w*y,starts)
    neg = np.add.reduceat(w*(1-y),starts)
    total = pos.sum()*neg.sum()
    if total == 0:
        return np.nan
    lower_neg = np.cumsum(neg)-neg
    return float(np.sum(pos*(lower_neg+.5*neg))/total)


def conditional_intervals(y, predictions, groups, reps, seed):
    """Pooled target AUC intervals, paired cluster bootstrap on report groups."""
    _,codes = np.unique(groups,return_inverse=True)
    count = codes.max()+1
    rng = np.random.default_rng(seed)
    draws = {name:np.empty((reps,len(LAB))) for name in predictions}
    for b in range(reps):
        cluster_count = np.bincount(rng.integers(0,count,count),minlength=count)
        weights = cluster_count[codes].astype(float)
        for name,p in predictions.items():
            draws[name][b] = [weighted_auc(y[:,j],p[:,j],weights) for j in range(len(LAB))]
    out = {}
    for name,v in draws.items():
        out[name] = {'target_auc':{}, 'paired_delta_vs_R1':{}}
        for j,target in enumerate(LAB):
            a=v[:,j];ok=np.isfinite(a)
            out[name]['target_auc'][target]={'low':float(np.quantile(a[ok],.025)),
                'high':float(np.quantile(a[ok],.975)), 'valid_reps':int(ok.sum())}
            delta=a-draws['R1'][:,j];ok=np.isfinite(delta)
            out[name]['paired_delta_vs_R1'][target]={'low':float(np.quantile(delta[ok],.025)),
                'high':float(np.quantile(delta[ok],.975)), 'valid_reps':int(ok.sum())}
    return {'unit':'report_group','groups':int(count),'reps':reps,'seed':seed,
            'estimand':'pooled-target AUC, fixed saved predictions; not mean fold AUC',
            'caveat':'Conditional descriptive intervals only. Does not remove checkpoint selection, R2 leakage, label error, seed/environment or private-distribution uncertainty; no multiplicity correction.',
            'models':out}


def target_metrics(y,p):
    macro,per=macro_auc(y,p)
    rows=[]
    for j,target in enumerate(LAB):
        pos=int(y[:,j].sum());neg=len(y)-pos;auc=per.get(target)
        rows.append({'target':target,'P':pos,'N':neg,'positive_negative_pairs':pos*neg,
            'auc':auc,'macro_deficit':None if auc is None else (1-auc)/12,
            'discordant_pair_equivalents':None if auc is None else (1-auc)*pos*neg,
            'one_pair_macro_value':None if not pos*neg else 1/(12*pos*neg)})
    return {'studies':len(y),'computable_targets':len(per),
            'macro_12':macro if len(per)==12 else None,
            'partial_target_mean':macro if len(per)!=12 and np.isfinite(macro) else None,
            'targets':rows}


def row_error_contributions(y,p):
    """Split each discordant +/- pair equally between its two study endpoints."""
    pos=p[y==1];neg=p[y==0]
    if not len(pos) or not len(neg):
        return np.zeros(len(y))
    ps,ns=np.sort(pos),np.sort(neg)
    errors=np.empty(len(y),float)
    positive=y==1
    pp=p[positive];nn=p[~positive]
    errors[positive]=len(ns)-np.searchsorted(ns,pp,'right')+.5*(np.searchsorted(ns,pp,'right')-np.searchsorted(ns,pp,'left'))
    errors[~positive]=np.searchsorted(ps,nn,'left')+.5*(np.searchsorted(ps,nn,'right')-np.searchsorted(ps,nn,'left'))
    return errors/(2*len(pos)*len(neg)*12)


def read_frame(path, expected):
    df=pd.read_csv(path,dtype={UID:str})
    if list(df.columns)!=[UID,*LAB] or df[UID].isna().any() or df[UID].duplicated().any():
        raise ValueError(f'Invalid UID/column contract: {path}')
    df=df.set_index(UID)
    if set(df.index)!=set(expected):
        raise ValueError(f'UID mismatch: {path}')
    arr=df.loc[expected,LAB].to_numpy()
    if not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1:
        raise ValueError(f'Invalid predictions: {path}')
    return arr


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--bootstrap',type=int,default=500)
    ap.add_argument('--seed',type=int,default=20261003)
    a=ap.parse_args()
    out=Path(a.out_dir)
    if out.exists(): raise FileExistsError(out)
    if a.bootstrap<10: raise ValueError('At least10 bootstrap draws required')
    paths=[]
    label_path=ROOT/'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
    fold_path=ROOT/'artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv'
    meta_path=ROOT/'storage/labels/derived_meta/study_meta.csv'
    corpus=ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02'
    series_path=ROOT/'storage/competition/train_series.csv'
    labels,label_hash=load_label_csv(label_path);fold_map,split_hash=load_folds(fold_path)
    folds=pd.read_csv(fold_path,dtype={UID:str}).set_index(UID)
    ids=sorted(u for u in labels if fold_map.get(u,-1)>=0)
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate labels')
    ysoft=np.stack([labels[u] for u in ids]);y=np.rint(ysoft).astype(int)
    groups=folds.loc[ids,'report_group_sha256'].to_numpy()
    if pd.isna(groups).any() or folds.groupby('report_group_sha256')['fold'].nunique().max()>1:
        raise ValueError('Invalid report groups')
    predictions,models,fold_results,gold_rows={},{},[],[]
    paths.extend([label_path,fold_path,meta_path,corpus/'manifest.json',corpus/'ids.npy',corpus/'masks.npy',series_path])
    gold=pd.read_csv(ROOT/'storage/competition/train.csv',usecols=[UID,*LAB],dtype={UID:str}).dropna(subset=LAB).set_index(UID)
    gids=sorted(gold.index);gy=gold.loc[gids,LAB].to_numpy()
    if set(ids)&set(gids): raise ValueError('Gold overlap')
    for name,prefix in PREFIXES.items():
        accumulated={};run_meta=[];gold_pred=[]
        for fold in range(5):
            folder=ROOT/'artifacts/experiments'/f'{prefix}_F{fold}'
            sp=folder/'summary.json';fp=folder/'oof_best.csv';gp=folder/'gold_best.csv'
            s=json.loads(sp.read_text(encoding='utf-8'))
            if s['fold']!=fold or s['split_hash']!=split_hash or s['args']['select']!='oof':
                raise ValueError(f'Unexpected saved-run contract: {folder}')
            fid=[u for u in ids if fold_map[u]==fold];p=read_frame(fp,fid)
            accumulated.update(zip(fid,p));pg=read_frame(gp,gids);gold_pred.append(pg)
            run_meta.append({'tag':folder.name,'fold':fold,'rows':len(fid),'historical_selected_checkpoint':s['selected_checkpoint'],
                'analysis_checkpoint':'best','label_hash':s['label_hash'],'best_epoch':s['best_epoch'],
                'args':{k:s['args'].get(k) for k in ['arch','res','epochs','k','k_eval','norm','seed','amp','bb_lr','head_lr','select']},
                'environment':s['environment'],'split_hash':s['split_hash'],'study_set_hash':s['study_set_hash'],
                'corpus_ids_hash':s.get('corpus_manifest')})
            fr=target_metrics(np.stack([np.rint(labels[u]).astype(int) for u in fid]),p)
            fold_results.append({'model':name,'fold':fold,**fr})
            gold_rows.append({'model':name,'fold':fold,'status':'Gold58 exposed diagnostic; never used for this audit selection',**target_metrics(gy,pg)})
            paths.extend([sp,fp,gp])
        predictions[name]=np.stack([accumulated[u] for u in ids])
        models[name]={'status':CAVEATS[name],'runs':run_meta,**target_metrics(y,predictions[name])}
        gold_rows.append({'model':name,'fold':'mean5_probabilities','status':'Gold58 exposed diagnostic, ensemble across folds; not OOF or selection criterion',**target_metrics(gy,np.mean(gold_pred,axis=0))})
    intervals=conditional_intervals(y,predictions,groups,a.bootstrap,a.seed)
    target_rows=[];case_rows=[]
    for name,model in models.items():
        for j,row in enumerate(model['targets']):
            row.update({'auc_ci':intervals['models'][name]['target_auc'][row['target']],
                        'delta_vs_R1':row['auc']-models['R1']['targets'][j]['auc'],
                        'delta_macro_contribution':(row['auc']-models['R1']['targets'][j]['auc'])/12,
                        'delta_ci_vs_R1':intervals['models'][name]['paired_delta_vs_R1'][row['target']]})
            target_rows.append({'model':name,'status':CAVEATS[name],**{k:v for k,v in row.items() if not isinstance(v,dict)},
                'auc_ci_low':row['auc_ci']['low'],'auc_ci_high':row['auc_ci']['high'],
                'delta_ci_low':row['delta_ci_vs_R1']['low'],'delta_ci_high':row['delta_ci_vs_R1']['high']})
            contrib=row_error_contributions(y[:,j],predictions[name][:,j])
            if not np.isclose(contrib.sum(),row['macro_deficit']): raise ValueError('Error contribution check failed')
            for k in np.argsort(-contrib,kind='stable')[:10]:
                case_rows.append({'model':name,'target':row['target'],UID:ids[k], 'fold':fold_map[ids[k]],
                    'reference_soft_label':float(ysoft[k,j]),'binary_reference':int(y[k,j]),
                    'prediction':float(predictions[name][k,j]),'allocated_macro_deficit':float(contrib[k]),
                    'meaning':'Half-allocation of weak-label discordant pairs; not clinical error or achievable gain.'})
    meta=pd.read_csv(meta_path,dtype={UID:str}).set_index(UID).reindex(ids)
    if meta.index.has_duplicates: raise ValueError('Duplicate study metadata')
    corpus_ids=np.load(corpus/'ids.npy').astype(str);masks=np.load(corpus/'masks.npy')
    rowmap={u:i for i,u in enumerate(corpus_ids)};mask=masks[[rowmap[u] for u in ids]]
    cm=json.loads((corpus/'manifest.json').read_text(encoding='utf-8'))
    starts=np.cumsum([0]+[s[2] for s in cm['spec']['slots']]);present=[]
    for j,slot in enumerate(cm['spec']['slots']):
        sm=mask[:,starts[j]:starts[j+1]];present.append((sm>0).any(axis=1))
    present=np.stack(present,axis=1)
    meta['cached_slot_presence']=pd.Series(present.sum(1),index=ids).map(lambda x:'5_present' if x==5 else '4_present' if x==4 else '0_to_3_present')
    meta['cached_any_zero_mask']=np.where((mask==0).any(axis=1),'some_zero_mask','all_mask_nonzero')
    meta['cached_valid_fraction']=np.count_nonzero(mask,axis=1)/mask.shape[1]
    meta['laterality_available']=np.where(meta['laterality'].fillna('unknown').isin(['L','R']),'known','unknown')
    series=pd.read_csv(series_path,dtype={UID:str})
    planes=series.groupby(UID)['Anatomical_Plane'].agg(lambda x:set(x))
    plane_count=np.array([sum(p in planes.get(u,set()) for p in ['Sagittal','Coronal','Axial']) for u in ids])
    meta['metadata_plane_coverage']=np.where(plane_count==3,'3_planes','fewer_than3')
    # Plane+FS candidate multiplicity is a selection ambiguity indicator, not demonstrated wrong selection.
    mult=series.groupby([UID,'Anatomical_Plane','Fat_Suppression']).size().groupby(level=0).max()
    meta['multiple_same_plane_fs_candidates']=np.where(mult.reindex(ids).fillna(0).to_numpy()>1,'multiple','single')
    group_results=[]
    group_fields=['scanner_group','field','manufacturer','laterality_available','cached_slot_presence','cached_any_zero_mask','metadata_plane_coverage','multiple_same_plane_fs_candidates']
    for field in group_fields:
        vals=meta[field].fillna('unknown').astype(str).to_numpy()
        for group in sorted(set(vals)):
            selected=vals==group;n=int(selected.sum())
            row={'field':field,'group':group,'studies':n,'models':{},'low_count':n<40}
            if n>=40:
                for name,p in predictions.items():row['models'][name]=target_metrics(y[selected],p[selected])
                for name in ['ResNet34','R2']:
                    ar=row['models']['R1'];br=row['models'][name]
                    br['delta_macro12_vs_R1']=None if ar['macro_12'] is None or br['macro_12'] is None else br['macro_12']-ar['macro_12']
            group_results.append(row)
    fold_pairs=[]
    for fold in range(5):
        rs={r['model']:r for r in fold_results if r['fold']==fold}
        for name in ['ResNet34','R2']:
            fold_pairs.append({'fold':fold,'candidate':name,'baseline':'R1','delta':rs[name]['macro_12']-rs['R1']['macro_12'],
                'target_delta':{t:rs[name]['targets'][j]['auc']-rs['R1']['targets'][j]['auc'] for j,t in enumerate(LAB)}})
    fingerprints=meta['fingerprint'].fillna('unknown')
    artifacts=[]
    paths.extend([ROOT/'storage/competition/train.csv',Path(__file__)])
    for p in paths:
        artifacts.append({'uri':'workspace://'+p.resolve().relative_to(ROOT).as_posix(),'sha256':sha256_file(p),'bytes':p.stat().st_size})
    report={'observed_at':datetime.now(timezone.utc).isoformat(),'scope':'Saved 15 best OOF + Gold58 CSVs; no new inference/training. Metadata/masks only; no volume/weight loading.',
        'reference':{'uri':'workspace://'+label_path.relative_to(ROOT).as_posix(),'canonical_label_hash':label_hash,'split_hash':split_hash,
            'binary_rule':'numpy.rint after float32 load, matching historical macro_auc; exactly0.5 ->0 (ties-to-even)',
            'exact_half_cells':int((ysoft==.5).sum()),'studies':len(ids),'targets':LAB,'Gold_overlap':0,
            'report_groups':len(np.unique(groups)),'patient_independence':'Unverified: report groups are a proxy, no patient-ID claim.'},
        'interpretation':['12 targets equally weighted; target Delta contributes Delta/12 to macro.',
            '(1-AUC)/12 is descriptive weak-label deficit, not recoverable score or private improvement forecast.',
            'AUC counts correctly ranked positive/negative pairs, ties half; no threshold-specific FP/FN claim.',
            'Pooled OOF can reflect between-fold score calibration; per-fold results retained.',
            'R1/ResNet selected development OOF and R2 indirect leakage/changed-label selection are not independent evidence.',
            'Scanner slices are observational, not scanner-held-out testing; label/scanner mix can confound gaps.',
            'Cached mask denotes nonzero pixels, not clinical/acquisition availability; plane metadata is separate.',
            'Gold58 separate exposed diagnostic; no label/arm selection based on these values.'],
        'models':models,'fold_metrics':fold_results,'fold_paired_differences':fold_pairs,
        'conditional_intervals':intervals,'group_diagnostics':group_results,'gold58_diagnostic':gold_rows,
        'metadata_summary':{'studies_with_missing_metadata':int(meta['scanner_group'].isna().sum()),'scanner_fingerprints':int(fingerprints.nunique()),
            'scanner_fingerprint_not_patient_id':True,'empty_cached_studies':[ids[i] for i in np.flatnonzero(mask.sum(1)==0)],
            'valid_mask_fraction_quantiles':dict(zip(['min','p25','median','p75','max'],map(float,np.quantile(meta['cached_valid_fraction'],[0,.25,.5,.75,1]))))},
        'source_manifest':artifacts}
    out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(target_rows).to_csv(out/'targets.csv',index=False)
    pd.DataFrame(case_rows).to_csv(out/'weak_rank_error_cases.csv',index=False)
    meta.reset_index(names=UID).to_csv(out/'study_diagnostics.csv',index=False)
    print(json.dumps({'out':str(out),'macro':{n:m['macro_12'] for n,m in models.items()},'rows':len(ids),'groups':len(group_results),'source_files':len(artifacts)},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
