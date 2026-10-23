"""One CPU comparison of completed EXP082 against frozen EXP080 B/RN predictions.

No models or Gold are loaded. All ranks are frozen on the full F2 before group
diagnostics or bootstrap. The two contrasts are the registered primary blend
change and secondary RN-only change; this module does not select a ratio.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp082_r34_loss as guard
import exp080_f2_report as f2
import input_ab_report as primary
from exp079_complement_report import fixed_blend, validate_run
from oof_error_audit import weighted_auc

ROOT, UID, LAB = primary.ROOT, primary.UID, primary.LAB
PAIRS = {'primary_blend_ones_minus_clip': ('blend_clip', 'blend_ones'),
         'secondary_rn_ones_minus_clip': ('RN_clip', 'RN_ones')}
SCHEMA = 'exp082-loss-report-v1'


def scores_for(b, old, new):
    rb, old_blend = fixed_blend(b, old)
    rb2, new_blend = fixed_blend(b, new)
    if not np.array_equal(rb, rb2):
        raise ValueError('Baseline ranks changed between arms')
    return {'B': rb, 'RN_clip': old, 'RN_ones': new,
            'blend_clip': old_blend, 'blend_ones': new_blend}


def target_rows(y, scores):
    y = np.rint(np.asarray(y, dtype=np.float32)).astype(int)
    if y.ndim != 2 or y.shape[1] != 12 or any(v.shape != y.shape for v in scores.values()):
        raise ValueError('Matching n by12 labels/scores required')
    rows = []
    for j, target in enumerate(LAB):
        pos = int(y[:, j].sum()); neg = len(y)-pos
        auc = {k: float(weighted_auc(y[:, j], v[:, j], np.ones(len(y))))
               if pos and neg else None for k, v in scores.items()}
        delta = {k: auc[b]-auc[a] if pos and neg else None for k, (a,b) in PAIRS.items()}
        rows.append(dict(target=target, P=pos, N=neg, auc=auc, delta=delta,
                         macro_contribution={k: None if v is None else v/12 for k,v in delta.items()}))
    return rows


def aggregate(rows):
    def mean(values):
        return None if any(v is None for v in values) else float(np.mean(values))
    return {k: {'macro12': mean([r['delta'][k] for r in rows]),
                'PFOA': next(r['delta'][k] for r in rows if r['target']=='PF OA'),
                'other11': mean([r['delta'][k] for r in rows if r['target']!='PF OA'])}
            for k in PAIRS}


def intervals(y, scores, groups, reps=500, seed=20261003):
    y = np.rint(np.asarray(y, dtype=np.float32)).astype(int)
    unique, codes = np.unique(groups, return_inverse=True)
    if len(groups)!=len(y) or not len(unique) or reps<1:
        raise ValueError('Invalid grouped bootstrap inputs')
    rng = np.random.default_rng(seed)
    draws = {k: np.empty((reps,12)) for k in PAIRS}
    names = sorted({n for pair in PAIRS.values() for n in pair})
    for i in range(reps):
        weights = np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[codes]
        auc = {k: np.array([weighted_auc(y[:,j],scores[k][:,j],weights) for j in range(12)]) for k in names}
        for key,(a,b) in PAIRS.items():
            draws[key][i] = auc[b]-auc[a]
    def ci(v):
        good = v[np.isfinite(v)]
        return dict(requested_reps=reps, valid_reps=int(len(good)),
                    low=float(np.quantile(good,.025)) if len(good) else None,
                    high=float(np.quantile(good,.975)) if len(good) else None)
    pf = LAB.index('PF OA')
    return {'unit':'report_group', 'groups':len(unique), 'seed':seed,
            'scope':'Fixed full-fold predictions/ranks only; no rerank, label/seed/selection/private or multiplicity uncertainty.',
            'contrasts':{k:dict(macro12=ci(v.mean(1)), PFOA=ci(v[:,pf]),
                other11=ci(np.delete(v,pf,axis=1).mean(1)), targets={t:ci(v[:,j]) for j,t in enumerate(LAB)})
                for k,v in draws.items()}}


def require_complete(root, queue):
    if len(queue)!=1 or queue[0]['tag']!=guard.TAG:
        raise ValueError('Expected exactly the single EXP082 fit')
    log=root/'artifacts/logs'/guard.TAG; run=root/'artifacts/experiments'/guard.TAG
    required=[log/'SUCCESS',log/'job.json',run/'summary.json',run/'oof_best.csv',run/'provenance.json']
    if (log/'FAILED').exists() or any(not p.is_file() for p in required):
        raise ValueError('Completed SUCCESS, summary, provenance and predictions required before any scoring')
    if re.fullmatch(r'exit 0 at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s*',required[0].read_text()) is None:
        raise ValueError('Invalid successful exit marker')
    job=guard.read(required[1])
    if job['tag']!=guard.TAG or job['command']!=queue[0]['command']:
        raise ValueError('Actual training command differs from frozen queue')
    if list(run.glob('gold_*.csv')) or (run/'model_swa.pt').exists():
        raise ValueError('Unexpected Gold or SWA output')
    return run, required


def load(manifest_path, expected_sha, root=ROOT):
    root=Path(root); hashes={}
    def bind(path, expected=None):
        path=Path(path); digest=guard.sha256(path)
        if expected is not None and digest!=expected:
            raise ValueError('Changed registered artifact: '+str(path))
        hashes[f2.uri(path,root)]=digest
        return digest
    bind(manifest_path,expected_sha);m=guard.read(manifest_path)
    if (m['schema']!=guard.SCHEMA or m['tag']!=guard.TAG or m['targets']!=LAB
            or m['loss']['actual_pos_weight']!=[1.]*12 or m['evaluation']['fixed_weights']!=[.9,.1]
            or m['evaluation']['conditional_ci']!={'reps':500,'seed':20261003,'unit':'report_group'}):
        raise ValueError('Wrong frozen EXP082 experiment/evaluation contract')
    queue_path=Path(manifest_path).parent/'queue.json'; bind(queue_path)
    queue=guard.read(queue_path);run,required=require_complete(root,queue)
    # No prediction values or label values are read until completion is confirmed.
    for path,digest in {**m['source_bindings'],**m['baselines']}.items():bind(root/path,digest)
    for name,digest in m['training_code_sha256'].items():bind(root/'training/src'/name,digest)
    for path in required:bind(path)
    ev=root/'docs/handoffs/evidence/2026-10/2026-10-03-continuous-research'
    launch=guard.read(ev/'exp082-launch-receipt.json');decision=guard.read(ev/'exp082-launch-decision.json')
    bind(ev/'exp082-launch-receipt.json');bind(ev/'exp082-launch-decision.json')
    if (launch['exit_code']!=0 or launch['protocol_sha256']!=expected_sha
            or decision['manifest_sha256']!=expected_sha or decision['queue_sha256']!=guard.sha256(queue_path)
            or launch['job']['command']!=queue[0]['command']):
        raise ValueError('Launch binding mismatch')
    parent=guard.read(root/guard.PARENT)
    summary=guard.read(run/'summary.json');prov=guard.read(run/'provenance.json')
    if summary['provenance']!=prov:raise ValueError('Summary/provenance mismatch')
    guard.validate_pretraining(prov,summary['environment'],m)
    expected_prov=dict(adapter_sha256=m['source_bindings'][guard.ENTRYPOINT],
        contract_helper_sha256=m['source_bindings']['tools/exp082_r34_loss.py'],protocol_sha256=expected_sha,
        actual_pos_weight=m['loss']['actual_pos_weight'],baseline_pos_weight=m['loss']['baseline_pos_weight'])
    if any(prov.get(k)!=v for k,v in expected_prov.items()):raise ValueError('Adapter/ones/initial contract mismatch')
    if summary.get('gold')!=0 or summary.get('gold_access')!='disabled' or summary.get('loss_policy')!='ones_positive_term':
        raise ValueError('Gold or loss policy mismatch')
    cp=deepcopy(parent);cp.update(epochs=20,expected_args=m['expected_args'],initial_state_sha256=m['initial_state_sha256'])
    skip=validate_run(summary,cp,dict(tag=guard.TAG,fold=2,policy='within-slot-valid'),.01)
    if summary['optimizer_updates']+summary['skipped_updates']!=8680:raise ValueError('Optimizer attempt count differs')
    old_runs={x['arm']:x for x in parent['runs']}
    for arm in ('B','RN_B'):
        old=guard.read(root/'artifacts/experiments'/old_runs[arm]['tag']/'summary.json')
        validate_run(old,f2.protocol_for_run(parent,old_runs[arm]),old_runs[arm],.01)
        if any(old['provenance'][k]!=prov[k] for k in ('runtime','eval_runtime')):
            raise ValueError('Uncontrolled runtime difference')
        if any(old['environment'][k]!=summary['environment'][k] for k in guard.ENV_KEYS):
            raise ValueError('Uncontrolled environment difference')
    order=guard.read(root/m['ordered_ids']['path']);ids=order['eval']
    split=pd.read_csv(root/m['paths']['fold_file'],dtype={UID:str}).set_index(UID)
    corpus=root/m['paths']['corpus'];corpus_ids=np.load(corpus/'ids.npy',allow_pickle=False).astype(str).tolist()
    train,recreated=guard.reconstruct_ids(corpus_ids,split.fold.to_dict(),set(m['excluded_uids']),m['expected_ids'])
    guard.validate_ordered_ids(train,recreated,order)
    if set(split.loc[train,'report_group_sha256']) & set(split.loc[ids,'report_group_sha256']):
        raise ValueError('Report group crosses F2')
    labels=primary.validate_sources(parent,root)
    y=np.stack([labels[u] for u in ids]).astype(np.float32)
    if not np.isfinite(y).all() or np.any((y<0)|(y>1)):raise ValueError('Invalid fixed evaluation labels')
    meta_path=root/'storage/labels/derived_meta/study_meta.csv'
    meta_ref=parent['metadata_source_hashes']['workspace://storage/labels/derived_meta/study_meta.csv']
    bind(meta_path,meta_ref);meta=pd.read_csv(meta_path,dtype={UID:str}).set_index(UID)
    if not meta.index.is_unique or not set(ids)<=set(meta.index):raise ValueError('Invalid metadata coverage')
    groups=meta.loc[ids,['scanner_group','field','manufacturer']].fillna('unknown')
    masks=np.load(corpus/'masks.npy',allow_pickle=False,mmap_mode='r')
    groups['cached_mask']=pd.Series(np.where((masks==0).any(1),'some_zero_mask','all_nonzero'),index=corpus_ids).loc[ids]
    b=f2.predictions(root/'artifacts/experiments/EXP080_INPUT_B_F2/oof_best.csv',ids)
    old=f2.predictions(root/'artifacts/experiments/EXP080_R34_B_F2/oof_best.csv',ids)
    new=f2.predictions(run/'oof_best.csv',ids)
    return dict(manifest=m,y=y,scores=scores_for(b,old,new),groups=groups,
                report_groups=split.loc[ids,'report_group_sha256'].to_numpy(),hashes=hashes,
                execution=dict(optimizer_updates=summary['optimizer_updates'],skipped_updates=summary['skipped_updates'],
                    skip_fraction=skip,total_seconds=summary['total_seconds'],gold_access='disabled'))


def build(manifest, sha, out, root=ROOT):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    data=load(Path(manifest),sha,root)
    tools={n:guard.sha256(Path(__file__).parent/n) for n in
           ('exp082_loss_report.py','exp082_r34_loss.py','exp080_f2_report.py','exp079_complement_report.py','input_ab_report.py','oof_error_audit.py')}
    out.mkdir(parents=True,exist_ok=False)
    f2.write_new(out/'analysis_plan.json',dict(schema=SCHEMA,observed_at=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha,contrasts=PAIRS,evaluator_sha256=tools,source_hashes=data['hashes'],evaluation=data['manifest']['evaluation']))
    y,scores=data['y'],data['scores'];rows=target_rows(y,scores);sums=aggregate(rows)
    if any(v['macro12'] is None for v in sums.values()):raise ValueError('Missing full12 target AUC')
    ci=intervals(y,scores,data['report_groups']);group_rows=[]
    for field in data['groups'].columns:
        vals=data['groups'][field].astype(str).to_numpy()
        for value in sorted(set(vals)):
            take=vals==value; rr=target_rows(y[take],{k:v[take] for k,v in scores.items()})
            group_rows.append(dict(field=field,group=value,studies=int(take.sum()),targets=rr,contrasts=aggregate(rr)))
    for uri,digest in data['hashes'].items():
        if guard.sha256(f2.unuri(uri,root))!=digest:raise ValueError('Source changed during scoring')
    report=dict(schema=SCHEMA,observed_at=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha,
        evaluator_sha256=tools,source_hashes=data['hashes'],studies=len(y),targets=rows,contrasts=sums,
        conditional_intervals=ci,groups=group_rows,execution=data['execution'],
        interpretation='Development evidence only; no automatic adoption, training, ratio search or submission.',
        caveats=['Exposed weak F2/one seed; known duplicate pair is both train, other duplicates unresolved.',
                 'Fixed-score bootstrap excludes label, seed, prior selection and private uncertainty; no multiplicity correction.',
                 'This is B+RN recipe evidence, not anchor0.943 or D01 accuracy evidence.',
                 'Single-class subgroup targets and macro12 stay null; masks do not establish clinical normality.'])
    f2.write_new(out/'report.json',report);pd.json_normalize(rows).to_csv(out/'targets.csv',index=False)
    return dict(studies=len(y),contrasts=sums,conditional_macro_ci={k:v['macro12'] for k,v in ci['contrasts'].items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(build(a.manifest,a.manifest_sha256,a.out),ensure_ascii=False,indent=2))
