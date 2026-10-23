"""CPU-only, aggregate-only fallback comparison; no label rows are emitted."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import exp169_release_v6 as v
from tools.exp157_l2_advance import ensemble_tag

def main():
    pins = []
    def table(p):
        pins.append(v.file_pin(p))
        return pd.read_csv(p).set_index(v.UID).loc[:, list(v.LABELS)]
    labels = table('storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv')
    folds = []
    for f in range(5):
        r = [table(f'artifacts/experiments/{ensemble_tag(s,f)}/oof_k36.csv') for s in (42,43)]
        c = table('artifacts/experiments/EXP179_MULTISCALE/outputs/M42/results/EXP179_M42_F0/oof_k36_flip_tta.csv' if f == 0 else f'artifacts/experiments/EXP183_MULTISCALE_5FOLD/outputs/fold{f}/results/EXP183_M42_F{f}/oof_k36_flip_tta.csv')
        m = [table('artifacts/experiments/EXP185_FINDING_MIL/outputs/S42/results/EXP185_FINDING_MIL_F0_S42/oof_k36_flip_tta.csv' if f == 0 else f'artifacts/experiments/EXP186_FINDING_MIL_5FOLD/outputs-b/fold{f}/results/EXP186B_FINDING_MIL_S42_F{f}/oof_k36_flip_tta.csv'),
             table('.local/refactor/mil-s43-f0-collected-v3/fold0/outputs/oof_k36_flip_tta.csv' if f == 0 else f'.local/refactor/mil-s43-members-collected-v1/fold{f}/outputs/oof_k36_flip_tta.csv')]
        ids = c.index
        assert all(set(t.index) == set(ids) for t in [*r,*m])
        folds.append((ids, np.stack([(r[0].loc[ids].values+r[1].loc[ids].values)/2,c.values,(m[0].loc[ids].values+m[1].loc[ids].values)/2])))
    results = {}
    for name, fs in [('fold0',folds[:1]),('fivefold',folds)]:
        ids = [i for ix,_ in fs for i in ix]
        assert len(ids) == len(set(ids))
        a = np.concatenate([a for _,a in fs],axis=1)
        y = np.rint(labels.loc[ids].values.astype(np.float32))
        assert np.isfinite(y).all() and set(np.unique(y)) <= {0,1}
        ranks = (rankdata(a,axis=1,method='average')-1)/(len(ids)-1)
        baseline = roc_auc_score(y,ranks.mean(0),average='macro')
        losses = {k:[] for k in ['remaining_ensemble','single_donor_0','single_donor_1']}
        for seed in range(100):
            rng = np.random.default_rng(130000+seed)
            ix = rng.choice(len(ids),max(1,round(.01*len(ids))),replace=False)
            for missing in range(3):
                others = [k for k in range(3) if k != missing]
                # Re-rank only the observed cohort, as the release does.
                valid = np.ones(len(ids),bool); valid[ix] = False
                damaged = ranks.copy()
                damaged[missing,valid] = (rankdata(a[missing,valid],axis=0)-1)/(valid.sum()-1)
                for policy, replacement in [('remaining_ensemble',ranks[others][:,ix].mean(0)),('single_donor_0',ranks[others[0],ix]),('single_donor_1',ranks[others[1],ix])]:
                    damaged[missing,ix] = replacement
                    # Q7 re-ranks each repaired family before its equal blend.
                    repaired = (rankdata(damaged,axis=1,method='average')-1)/(len(ids)-1)
                    losses[policy].append(baseline-roc_auc_score(y,repaired.mean(0),average='macro'))
        results[name] = dict(studies=len(ids),baseline_auc=baseline,replicates=300,loss={k:dict(mean=float(np.mean(x)),std=float(np.std(x))) for k,x in losses.items()})
    result = dict(choice='other model rank: MIL then C4; weighted remaining members if neither survives',metric='12-target macro AUC; Dread teacher labels float32 then np.rint; all targets/studies fixed',drop='random 1% studies, each member, 100 seeds 130000..130099',single_donor_order='remaining R50/C4/MIL order; donor_1 means MIL except C4 for missing MIL',limitations='Teacher/weak OOF, not gold. R50 K36, C4/MIL K36 flip TTA. CoAtNet release trained on all studies (validation none), so no honest CoAtNet OOF available. Multiple-member fallback is the requested weighted policy, not OOF-estimated.',results=results,inputs=pins)
    out=Path(sys.argv[1]); v.write_new(out,result)
    print(json.dumps({k:val for k,val in result.items() if k != 'inputs'},indent=2))
if __name__ == '__main__': main()
