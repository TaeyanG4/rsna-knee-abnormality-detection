"""Predeclared, family-balanced teacher diagnostics and training-only targets.

No fitted weights, no optimizer on Gold58, no image-model claim. Explicit unknown
status is preserved separately from probability, annotation confidence, and
between-teacher disagreement. All 58 gold studies are absent from training exports.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from label_audit import ROOT,TARGETS,UID,load_teacher,auc_vector,bootstrap_macro
from public_prediction_audit import paired_delta

def main():
    cfgpath=ROOT/"experiments"/"EXP003_label_consensus_config.json"
    cfg=json.loads(cfgpath.read_text(encoding="utf-8"))
    train=pd.read_csv(ROOT/"data"/"train.csv",dtype={UID:str}).set_index(UID)
    gold=pd.read_csv(ROOT/"gold"/"gold_labels.csv",dtype={UID:str}).set_index(UID)
    ids=train.index.tolist();gid=gold.index.tolist();gpos=train.index.get_indexer(gid)
    y=gold[TARGETS].to_numpy(float)
    blocks=[];unknown=[];annotations=[];origins=[];source_tables={}
    for name,rel in cfg["sources"].items():
        path=ROOT/"labels"/rel
        frame,info=load_teacher(path);frame=frame.set_index(UID).reindex(ids)
        if info["licenses"]!=["CC0-1.0"]:raise ValueError(f"Declared license gate: {name}")
        p=frame[TARGETS].to_numpy(float);present=np.isfinite(p)
        unk=np.zeros_like(p,dtype=bool);known_annotation=np.zeros_like(p,dtype=bool)
        if name=="steven_raw":
            unk=p==.5;known_annotation=present.copy()
        if name=="pilkwang_llm":
            for j,t in enumerate(TARGETS):
                verdict=frame[t+"__verdict"].astype(str).str.upper()
                unk[:,j]=verdict.isin(["UNK","UNKNOWN","UNCERTAIN"])
                known_annotation[:,j]=verdict.isin(["YES","NO","UNK","UNKNOWN","UNCERTAIN"]) & present[:,j]
        if name=="qwen_raw":
            for j,t in enumerate(TARGETS):
                unk[:,j]=(frame[t+"__mentioned"]==0).to_numpy()
                known_annotation[:,j]=frame[t+"__mentioned"].notna().to_numpy() & present[:,j]
        # Lixin confidence is not assumed to mean "not addressed".
        blocks.append(p);unknown.append(unk);annotations.append(known_annotation)
        origins.append({"family":name,"source":rel,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        source_tables[name]=p
    p=np.stack(blocks);u=np.stack(unknown);ann=np.stack(annotations)
    if np.any(np.isfinite(p).sum(0)==0):raise ValueError("No available teacher for a study/target")
    mean=np.nanmean(p,axis=0);std=np.nanstd(p,axis=0)
    annotation_count=ann.sum(0)
    uf=np.divide((u&ann).sum(0),annotation_count,out=np.full_like(mean,np.nan),where=annotation_count>0)
    adjusted=np.where(u&ann,.5,p)
    variants={"family4_probability_mean":mean,"family4_explicit_unknown_to_0_5":np.nanmean(adjusted,axis=0)}
    reference,_=load_teacher(ROOT/"labels"/"flight0234__rsna-knee-hybrid-report-labels"/"report_labels_v4hybrid.csv")
    reference=reference.set_index(UID).loc[gid,TARGETS].to_numpy(float)
    reports=[];per=[]
    for name,pred in variants.items():
        a=auc_vector(y,pred[gpos]);r={"variant":name,"n_gold":58,"macro_auc":float(a.mean()),
            "bootstrap":bootstrap_macro(y,pred[gpos],3000),
            "delta_vs_public_flight":paired_delta(y,reference,pred[gpos],cfg["bootstrap_repeats"]),
            "status":"teacher_diagnostic_only_no_student_trained"}
        reports.append(r)
        per.extend({"variant":name,"target":t,"auc":float(a[j])} for j,t in enumerate(TARGETS))
        print(name,f"AUC={a.mean():.8f}","DELTA",json.dumps(r["delta_vs_public_flight"]))
    feature_rows=[]
    for i,uid in enumerate(ids):
        for j,t in enumerate(TARGETS):
            available=np.isfinite(p[:,i,j]);agree=max(int((p[:,i,j]>=.5).sum()),int((p[:,i,j]<.5).sum()))/max(1,int(available.sum()))
            feature_rows.append({UID:uid,"target":t,"label_mean":float(mean[i,j]),"label_std":float(std[i,j]),
               "teacher_count":int(available.sum()),"teacher_agreement":agree,
               "num_positive_teachers":int((p[:,i,j]>=.8).sum()),"num_negative_teachers":int((p[:,i,j]<=.2).sum()),
               "explicit_unknown_fraction":float(uf[i,j]),"unknown_annotation_coverage":float(annotation_count[i,j]/len(blocks)),
               "missing_teacher_fraction":float(1-available.mean()),"agreement_loss_weight_candidate":float(np.clip(1-4*std[i,j]**2,0,1)),
               "is_gold":uid in gold.index})
    allfeatures=pd.DataFrame(feature_rows)
    out=ROOT/"labels"/"derived_EXP003";out.mkdir(exist_ok=True)
    allfeatures.to_csv(out/"teacher_uncertainty_all_train.csv",index=False)
    notgold=~train.index.isin(gid)
    for name,pred in variants.items():
        frame=pd.DataFrame(pred,columns=TARGETS,index=ids).rename_axis(UID)
        frame.loc[notgold].to_csv(out/(name+"_weak4349.csv"))
        assert not (set(frame.loc[notgold].index)&set(gid))
    pairs=[];names=list(source_tables)
    for ia,name in enumerate(names):
        for other in names[ia+1:]:
            for j,t in enumerate(TARGETS):
                a,b=source_tables[name][:,j],source_tables[other][:,j]
                ok=np.isfinite(a)&np.isfinite(b)
                pairs.append({"a":name,"b":other,"target":t,"n":int(ok.sum()),"pearson":float(np.corrcoef(a[ok],b[ok])[0,1]),"spearman":float(spearmanr(a[ok],b[ok]).statistic)})
    # Isolate the user's requested unknown=0 versus unknown=0.5 knob without fitting.
    idx=names.index("pilkwang_llm");valid=np.isfinite(p[idx,gpos]).all(1)
    probe=[]
    for value in [0.0,0.5]:
        pp=np.where(u[idx]&ann[idx],value,p[idx])[gpos][valid]
        aa=auc_vector(y[valid],pp)
        probe.append({"unknown_replacement":value,"n_gold":int(valid.sum()),"macro_auc":float(aa.mean()),**dict(zip(TARGETS,aa))})
    pd.DataFrame(per).to_csv(ROOT/"experiments"/"EXP003_per_target.csv",index=False)
    pd.DataFrame(pairs).to_csv(ROOT/"experiments"/"teacher_family_correlation.csv",index=False)
    pd.DataFrame(probe).to_csv(ROOT/"experiments"/"unknown_policy_ablation.csv",index=False)
    receipt={"utc":datetime.now(timezone.utc).isoformat(),"config_sha256":hashlib.sha256(cfgpath.read_bytes()).hexdigest(),
             "sources":origins,"variants":reports,"unknown_ablation":probe,"training_export_rows":int(notgold.sum()),
             "limitations":["Scores are teacher extraction quality, not MRI inference","Source families not guaranteed independent LLM architectures","Gold has prior community selection exposure","Disagreement weight has not been tested in student training"]}
    (ROOT/"reports"/"EXP003_teacher_consensus.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")

if __name__=="__main__":main()
