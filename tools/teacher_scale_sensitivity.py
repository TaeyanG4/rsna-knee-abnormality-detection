"""CPU-only target-order and loss-gradient sensitivity; no label files are produced."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
UID = 'StudyInstanceUID'

def pair_order(a, b):
    """Exact strict reversals and ties, counted once per unordered study pair."""
    counts = dict(pairs=0, strict_both=0, reversed=0, tied_a_only=0, tied_b_only=0, tied_both=0)
    n = len(a)
    for start in range(0,n,128):
        end = min(start+128,n)
        ia = np.arange(start,end)[:,None]
        mask = ia < np.arange(n)[None,:]
        da = np.sign(a[start:end,None]-a[None,:])[mask]
        db = np.sign(b[start:end,None]-b[None,:])[mask]
        counts['pairs'] += len(da)
        for name, select in [('strict_both',(da!=0)&(db!=0)),('reversed',da*db<0),('tied_a_only',(da==0)&(db!=0)),('tied_b_only',(da!=0)&(db==0)),('tied_both',(da==0)&(db==0))]:
            counts[name] += int(np.count_nonzero(select))
    assert counts['pairs'] == n*(n-1)//2
    counts['reversal_fraction_all_pairs'] = counts['reversed']/counts['pairs'] if counts['pairs'] else None
    counts['spearman'] = float(spearmanr(a,b).statistic)
    return counts

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    protocol={'schema':'teacher-scale-sensitivity-v1','scope':'descriptive, not clinical efficacy or corrected calibration',
              'cohort':'fixed 4343 EXP099 folds.csv studies; exclude Gold58',
              'targets':TARGETS,'teacher_windows':24,'dread_alpha':.5,
              'transform':'per teacher, per producing fold: q/(w-(w-1)*q), then equal mean, then mix with Dread',
              'order':'exact all unordered pairs, pooled and within each fold; strict reversals separated from ties',
              'gradient':'unreduced weighted-BCE dL/dz=(1+(w-1)*y)*sigmoid(z)-w*y; student fold weight on its own training rows',
              'fixed_logits':[-2.,0.,2.], 'decision':'material order/gradient change supports distinct intervention, never proves better supervision; no training authorized by this diagnostic'}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    hashes={}
    def read(path):
        path=ROOT/path;data=path.read_bytes();hashes[str(path.relative_to(ROOT))]=hashlib.sha256(data).hexdigest();return data.decode('utf-8-sig')
    def js(path):return json.loads(read(path))
    def rows(path):return list(csv.DictReader(io.StringIO(read(path))))
    folds=rows('artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/folds.csv')
    ids=[r[UID] for r in folds];fold=np.array([int(r['fold']) for r in folds]);assert len(ids)==len(set(ids))==4343
    def table(path):
        rr=rows(path);assert len(rr)==len({r[UID] for r in rr});rr={r[UID]:r for r in rr}
        values=np.array([[float(rr[u][t]) for t in TARGETS] for u in ids]);assert np.isfinite(values).all() and ((0<=values)&(values<=1)).all();return values,set(rr)
    train=rows('storage/competition/train.csv');gold={r[UID] for r in train if r['ACL'].strip()};assert not (set(ids)&gold) and len(gold)==58
    dread,_=table('storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv')
    receipt=js('storage/labels/derived_distill/v1/receipt.json')
    registration=js('artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/label-sets/distill_v1.json')
    actual,_=table(registration['path']);weight=np.array([registration['pos_weight'][str(f)] for f in range(5)])
    raw=[];inverse=[];bindings={}
    for teacher,folder in [('coatnet','EXP099_B8_DREAD_OOF'),('rn34','EXP101_RN34_DREAD_OOF')]:
        q,covered=table(receipt['teachers'][teacher]['path']);assert covered==set(ids)
        assert hashes[str(Path(receipt['teachers'][teacher]['path']))]==receipt['teachers'][teacher]['sha256']
        result=js('artifacts/experiments/'+folder+'/result.json');inv=np.empty_like(q);bindings[teacher]={}
        for fs,src in result['sources'].items():
            f=int(fs);pv=js(src['run']+'/provenance.json');w=np.array(pv['actual_pos_weight']);assert np.array_equal(w,weight[f])
            rr=rows(src['run']+'/oof_best.csv');assert {r[UID] for r in rr}=={ids[i] for i in np.flatnonzero(fold==f)}
            rr={r[UID]:r for r in rr};ref=np.array([[float(rr[ids[i]][t]) for t in TARGETS] for i in np.flatnonzero(fold==f)])
            assert np.max(np.abs(ref-q[fold==f]))<1e-7
            inv[fold==f]=q[fold==f]/(w-(w-1)*q[fold==f]);bindings[teacher][fs]={'actual_pos_weight':w.tolist(),'rows':len(rr),'source_run':src['run']}
        raw.append(q);inverse.append(inv)
    raw_mix=.5*dread+.5*np.mean(raw,axis=0);inv_mix=.5*dread+.5*np.mean(inverse,axis=0)
    error=float(np.max(np.abs(actual-raw_mix)));assert error<=5.01e-7
    result={'schema':protocol['schema'],'rows':len(ids),'mix_reconstruction_max_abs':error,'bindings':bindings,'targets':{}}
    for j,t in enumerate(TARGETS):
        target={'pooled':pair_order(raw_mix[:,j],inv_mix[:,j]),'by_producing_fold':{str(f):pair_order(raw_mix[fold==f,j],inv_mix[fold==f,j]) for f in range(5)},'mean_raw':float(raw_mix[:,j].mean()),'mean_inverse':float(inv_mix[:,j].mean()),'gradient':{}}
        for z in protocol['fixed_logits']:
            s=1/(1+np.exp(-z));gr=[];gi=[]
            for f in range(5):
                w=weight[f,j];yr=raw_mix[fold!=f,j];yi=inv_mix[fold!=f,j]
                gr.append((1+(w-1)*yr)*s-w*yr);gi.append((1+(w-1)*yi)*s-w*yi)
            gr=np.concatenate(gr);gi=np.concatenate(gi);delta=gi-gr
            target['gradient'][str(z)]={'student_training_row_appearances':len(gr),'mean_raw':float(gr.mean()),'mean_inverse':float(gi.mean()),'mean_abs_delta':float(np.abs(delta).mean()),'max_abs_delta':float(np.abs(delta).max()),'opposite_sign_fraction':float(np.mean(gr*gi<0))}
        result['targets'][t]=target
    # Independent finite-difference check of the analytic weighted-BCE derivative.
    maxerr=0.
    for w in [1.,2.5,8.]:
        for y in [0.,.1,.5,.9,1.]:
            for z in protocol['fixed_logits']:
                loss=lambda v:(1-y)*np.logaddexp(0,v)+w*y*np.logaddexp(0,-v)
                numerical=(loss(z+1e-5)-loss(z-1e-5))/2e-5
                exact=(1+(w-1)*y)/(1+np.exp(-z))-w*y;maxerr=max(maxerr,abs(numerical-exact))
    assert maxerr<1e-8
    # Exact inversion implementation sanity check includes one new tie and one reversal.
    check=pair_order(np.array([0.,1.,2.,3.]),np.array([0.,2.,1.,2.]));assert check['reversed']==1 and check['tied_b_only']==1
    result.update(gradient_finite_difference_max_abs=maxerr,input_sha256=hashes,scope=protocol['scope'])
    (a.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(ids),'mix_error':error,'finite_difference_error':maxerr,'targets':{t:{'rho':v['pooled']['spearman'],'flip_fraction':v['pooled']['reversal_fraction_all_pairs'],'gradient_at_zero':v['gradient']['0.0']} for t,v in result['targets'].items()}},indent=2))

if __name__=='__main__':main()
