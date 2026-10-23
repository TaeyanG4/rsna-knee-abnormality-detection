"""Re-score public numeric predictions, not a claim of re-running inference.

Uses only competition training studies. No test reports, test labels, or test
predictions enter this audit. Published gold-selected models remain explicitly
selection-exposed; bootstrapping cannot undo their selection bias.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from label_audit import ROOT, TARGETS, UID, auc_vector, bootstrap_macro, validate_ids

def paired_delta(y, baseline, candidate, repeats=5000, seed=20260912):
    rng=np.random.default_rng(seed)
    delta=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(y),len(y));yy=y[idx]
        if np.any(np.ptp(yy,axis=0)==0):continue
        delta.append(float((auc_vector(yy,candidate[idx])-auc_vector(yy,baseline[idx])).mean()))
    a=np.asarray(delta)
    return {"observed_delta":float((auc_vector(y,candidate)-auc_vector(y,baseline)).mean()),
            "valid_repeats":len(a),"repeats":repeats,"mean":float(a.mean()),"std":float(a.std(ddof=1)),
            "ci95_low":float(np.quantile(a,.025)),"ci95_high":float(np.quantile(a,.975)),
            "bootstrap_fraction_delta_positive":float((a>0).mean()),"seed":seed,
            "warning":"Exploratory conditional uncertainty; does not correct prior gold selection or multiple comparisons."}

def main():
    gold=pd.read_csv(ROOT/"gold"/"gold_labels.csv",dtype={UID:str})
    validate_ids(gold,"official_gold")
    gold=gold.set_index(UID);ids=gold.index.tolist();y=gold[TARGETS].to_numpy(float)
    entries={};per=[];receipt={"utc":datetime.now(timezone.utc).isoformat(),"execution":"recalculated_from_public_predictions_not_inference","models":[]}
    path=ROOT/"external"/"pilkwang__rsna-knee-weights"/"oof.npz"
    with np.load(path,allow_pickle=False) as z:
        order=z["targets"].astype(str).tolist()
        if order!=TARGETS:raise ValueError("Public OOF target order differs from official schema")
        uid=z["ids"].astype(str)
        pred=pd.DataFrame(z["pred"],columns=TARGETS,index=uid)
        if pred.index.duplicated().any() or len(pred)!=4407:raise ValueError("Invalid OOF IDs")
        mask=z["gold_mask"].astype(bool)
        if set(uid[mask])!=set(ids):raise ValueError("OOF gold mask is not official gold")
        yy=z["y_derived"][~mask].astype(float);pp=z["pred"][~mask].astype(float)
        receipt["public_weak_oof"]={"n":len(yy),"macro_auc":float(auc_vector(yy,pp).mean()),
                                    "target":"public artifact y_derived; NOT image gold",
                                    "fold_provenance_independently_verified":False}
        entries["Pilkwang_public_OOF"]=pred.loc[ids].to_numpy(float)
    path=ROOT/"external"/"mattiaangeli__rsna-knee-coat-resgated-ep10-top3"/"gold58_top3_predictions.npz"
    with np.load(path,allow_pickle=False) as z:
        uid=z["study_uids"].astype(str).tolist()
        if len(uid)!=58 or len(set(uid))!=58 or set(uid)!=set(ids):raise ValueError("CoAt gold UID mismatch")
        np.testing.assert_array_equal(gold.loc[uid,TARGETS].to_numpy(),z["truth"])
        pos=[uid.index(x) for x in ids]
        probs=z["raw_probabilities"][:,pos,:].astype(float)
        epochs=z["epochs"].tolist()
        for epoch,p in zip(epochs,probs): entries[f"CoAt_gold_selected_e{epoch:02d}"]=p
        calculated_ranks=rankdata(probs,method="average",axis=1)/58
        np.testing.assert_allclose(calculated_ranks,z["checkpoint_percentile_ranks"][:,pos,:],atol=1e-12)
        entries["CoAt_top3_rank"]=calculated_ranks.mean(0)
        entries["CoAt_top3_probability"]=probs.mean(0)
        np.testing.assert_allclose(entries["CoAt_top3_rank"],z["rank_ensemble"][pos],atol=1e-12)
        np.testing.assert_allclose(entries["CoAt_top3_probability"],z["probability_mean"][pos],atol=1e-7)
    for name,p in entries.items():
        if p.shape!=(58,12) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError(name)
        auc=auc_vector(y,p)
        summary={"model":name,"n_gold":58,"gold_macro_auc":float(auc.mean()),
                 "selection_exposure":"checkpoint selection on Gold58" if name.startswith("CoAt") else "public OOF artifact; fold lineage unverified",
                 "bootstrap":bootstrap_macro(y,p,3000)}
        receipt["models"].append(summary)
        per.extend({"model":name,"target":t,"n":58,"positive_count":int(y[:,j].sum()),"auc":float(auc[j])} for j,t in enumerate(TARGETS))
        pd.DataFrame(p,columns=TARGETS,index=ids).rename_axis(UID).to_csv(ROOT/"predictions"/(name+"_gold58_public.csv"))
    correlations=[]
    a,b=entries["Pilkwang_public_OOF"],entries["CoAt_top3_probability"]
    for j,t in enumerate(TARGETS):
        correlations.append({"target":t,"n":58,"pearson":float(np.corrcoef(a[:,j],b[:,j])[0,1]),
                             "spearman":float(spearmanr(a[:,j],b[:,j]).statistic),
                             "threshold_disagreement":float(((a[:,j]>=.5)!=(b[:,j]>=.5)).mean())})
    # These are diagnostics, not selected ensemble weights.
    receipt["rank_vs_probability_delta"]=paired_delta(y,entries["CoAt_top3_rank"],entries["CoAt_top3_probability"])
    pd.DataFrame(per).to_csv(ROOT/"experiments"/"public_prediction_per_target.csv",index=False)
    pd.DataFrame(correlations).to_csv(ROOT/"experiments"/"public_prediction_correlation.csv",index=False)
    (ROOT/"reports"/"public_prediction_audit.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print("PUBLIC_WEAK_OOF",json.dumps(receipt["public_weak_oof"]))
    for r in receipt["models"]:print(r["model"],f"gold={r['gold_macro_auc']:.8f}",r["bootstrap"]["ci95_low"],r["bootstrap"]["ci95_high"])
    print("PROB_MINUS_RANK",json.dumps(receipt["rank_vs_probability_delta"]))

if __name__=="__main__":main()
