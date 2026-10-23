"""Evaluate local report-teacher tables against official image-derived gold.

No network or external-code execution. Unknown values remain explicit. A score
from this script is a teacher audit, never an image-model validation score.
"""
from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from scipy.stats import rankdata

ROOT = workspace_root()
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = "StudyInstanceUID"

def validate_ids(frame: pd.DataFrame, name: str) -> None:
    if UID not in frame:
        raise ValueError(f"{name}: UID column absent")
    if frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError(f"{name}: null or duplicated UIDs")
    if (frame[UID].astype(str).str.strip() != frame[UID]).any():
        raise ValueError(f"{name}: UID whitespace; fix explicitly, not silently")

def bootstrap_macro(y: np.ndarray, p: np.ndarray, repeats: int, seed: int = 20260912) -> dict:
    """Study-level bootstrap, retaining all twelve targets in every replicate.

    Reject class-degenerate replicates instead of silently changing the macro's
    target set. Same random seed across teachers permits a paired comparison
    only when the matched study set is exactly identical.
    """
    rng = np.random.default_rng(seed)
    values = []
    invalid = 0
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], p[idx]
        if not np.isfinite(pp).all() or np.any(yy.min(0) == yy.max(0)):
            invalid += 1
            continue
        values.append(float(auc_vector(yy, pp).mean()))
    a = np.asarray(values)
    return {"repeats_requested": repeats, "valid_repeats": len(a), "invalid_repeats": invalid,
            "mean": float(a.mean()) if len(a) else None,
            "std": float(a.std(ddof=1)) if len(a)>1 else None,
            "ci95_low": float(np.quantile(a, .025)) if len(a) else None,
            "ci95_high": float(np.quantile(a, .975)) if len(a) else None,
            "seed": seed, "unit": "study", "targets_required_per_replicate": 12}

def auc_vector(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Tie-aware Mann-Whitney AUC, vectorized over the twelve findings."""
    positive=y.sum(axis=0)
    negative=len(y)-positive
    if np.any(positive==0) or np.any(negative==0):
        raise ValueError("AUC requires both gold classes for each finding")
    ranks=rankdata(p,method="average",axis=0)
    return ((ranks*y).sum(axis=0)-positive*(positive+1)/2)/(positive*negative)

def load_teacher(path: Path) -> tuple[pd.DataFrame, dict]:
    table=pd.read_csv(path,dtype={UID:str})
    validate_ids(table,str(path))
    mapping={t:t for t in TARGETS}
    adapter="direct_probability_columns"
    if path.parent.name.startswith("barun2104__"):
        mapping={t:"pseudo_"+t for t in TARGETS}
        adapter="documented_pseudo_columns_not_original_gold_columns"
    elif path.parent.name.startswith("laymond__") and path.name.startswith("weak_labels_v2_"):
        mapping={t:t+"__confidence" for t in TARGETS}
        adapter="documented_v2_P_present_not_predicted_class_confidence"
    absent=set(mapping.values())-set(table.columns)
    if absent: raise ValueError(f"{path.name}: explicit column adapter needed: {sorted(absent)}")
    pred=table[[UID]+list(mapping.values())].rename(columns={v:k for k,v in mapping.items()}).copy()
    pred[TARGETS]=pred[TARGETS].apply(pd.to_numeric,errors="raise")
    values=pred[TARGETS].to_numpy(float)
    finite=values[np.isfinite(values)]
    if np.isinf(values).any() or np.any(finite<0) or np.any(finite>1):
        raise ValueError(f"{path.name}: invalid target values")
    for t in TARGETS:
        for suffix in ["__verdict","__mentioned","__conf"]:
            if t+suffix in table:pred[t+suffix]=table[t+suffix]
    meta_path=path.parent/"dataset-metadata.json"
    info=json.loads(meta_path.read_text(encoding="utf-8-sig")).get("info",{}) if meta_path.exists() else {}
    licenses=[x.get("name","") for x in info.get("licenses",[])]
    return pred,{"adapter":adapter,"licenses":licenses,"declared_cc0":licenses==["CC0-1.0"],"upstream_rights_verified":False}

def evaluate(path: Path, gold: pd.DataFrame, source: str, repeats: int = 0) -> tuple[list[dict], dict]:
    table,adapter = load_teacher(path)
    pred = table[[UID]+TARGETS].copy()
    pred[TARGETS] = pred[TARGETS].apply(pd.to_numeric, errors="raise")
    values = pred[TARGETS].to_numpy(float)
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0) or np.any(values[np.isfinite(values)] > 1):
        raise ValueError(f"{source}: invalid nonprobability target values; needs documented adapter")
    merged = gold.merge(pred, on=UID, how="left", suffixes=("__gold", "__teacher"), validate="one_to_one", indicator=True)
    y = merged[[t+"__gold" for t in TARGETS]].to_numpy(float)
    p = merged[[t+"__teacher" for t in TARGETS]].to_numpy(float)
    complete = np.isfinite(p).all(1)
    exact_rows = complete & np.all(y == p, axis=1)
    overlap = int((merged["_merge"] == "both").sum())
    rows=[]
    for j,t in enumerate(TARGETS):
        valid=np.isfinite(p[:,j])
        yy,pp=y[valid,j],p[valid,j]
        auc=float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None
        unknown_col=t+"__verdict"
        unknown=float(table[unknown_col].astype(str).str.upper().isin(["UNK","UNKNOWN","UNCERTAIN"]).mean()) if unknown_col in table else None
        if t+"__mentioned" in table: unknown=float((table[t+"__mentioned"]==0).mean())
        if path.name=="llm_labels_full.csv": unknown=float((table[t]==.5).mean())
        rows.append({"source":source,"target":t,"n_gold":len(gold),"n_matched":int(valid.sum()),
                     "n_positive":int(yy.sum()),"auc":auc,
                     "accuracy_at_0_5":float(accuracy_score(yy,pp>=.5)) if len(yy) else None,
                     "brier":float(brier_score_loss(yy,pp)) if len(yy) else None,
                     "logloss":float(log_loss(yy,np.clip(pp,1e-7,1-1e-7),labels=[0,1])) if len(yy) else None,
                     "mean_probability_gold":float(pp.mean()) if len(pp) else None,
                     "positive_prevalence_gold":float((pp>=.5).mean()) if len(pp) else None,
                     "mean_probability_all":float(np.nanmean(values[:,j])) if np.isfinite(values[:,j]).any() else None,
                     "positive_prevalence_all":float((values[np.isfinite(values[:,j]),j]>=.5).mean()) if np.isfinite(values[:,j]).any() else None,
                     "unknown_fraction_all":unknown,
                     "missing_fraction_all":float(np.isnan(values[:,j]).mean())})
    comparable=overlap==len(gold) and complete.all() and all(r["auc"] is not None for r in rows)
    suspicious=bool(complete.all() and np.array_equal(y,p))
    matched_macro=float(auc_vector(y[complete],p[complete]).mean()) if complete.sum()>1 and np.all(np.ptp(y[complete],axis=0)>0) else None
    canonical=pred.sort_values(UID)[[UID]+TARGETS].to_csv(index=False,float_format="%.10g",na_rep="NA").encode()
    summary={"source":source,"file":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
             "prediction_sha256":hashlib.sha256(canonical).hexdigest(),
             "rows":len(table),"gold_overlap":overlap,"gold_complete_rows":int(complete.sum()),
             "exact_binary_gold_rows":int(exact_rows.sum()),"all_gold_exact_copy_flag":suspicious,
             "macro_auc":float(np.mean([r["auc"] for r in rows])) if comparable else None,
             "matched_only_macro_auc":matched_macro,
             "full58_comparable":bool(comparable),"independent_teacher_verified":False,
             "evaluation_type":"report_teacher_not_image_model",
             "notes":"Gold overwrite and prompt-tuning provenance require source audit; exact-copy flag is screening, not proof."}
    summary.update(adapter)
    summary["eligible_for_teacher_comparison"]=bool(not suspicious and complete.sum()>=50)
    if repeats and matched_macro is not None:
        summary["bootstrap"]=bootstrap_macro(y[complete],p[complete],repeats)
    return rows,summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bootstrap",type=int,default=0)
    args=ap.parse_args()
    gold=pd.read_csv(ROOT/"gold"/"gold_labels.csv",dtype={UID:str})
    validate_ids(gold,"official_gold")
    per,board,skipped=[],[],[]
    for path in sorted((ROOT/"labels").glob("*/*.csv")):
        if path.parent.name.startswith("derived_"):
            continue
        source=str(path.relative_to(ROOT/"labels")).replace("\\","/")
        try:
            rows,summary=evaluate(path,gold,source,args.bootstrap)
            per.extend(rows);board.append(summary)
        except (ValueError,KeyError,pd.errors.ParserError) as exc:
            skipped.append({"file":source,"reason":str(exc)})
    (ROOT/"reports"/"label_audit_receipt.json").write_text(json.dumps({"utc":datetime.now(timezone.utc).isoformat(),"teachers":board,"requires_adapter":skipped},indent=2,allow_nan=False),encoding="utf-8")
    if per: pd.DataFrame(per).to_csv(ROOT/"experiments"/"label_per_target_metrics.csv",index=False)
    if board:
        flat=[{k:v for k,v in r.items() if k!="bootstrap"} for r in board]
        frame=pd.DataFrame(flat).sort_values(["eligible_for_teacher_comparison","macro_auc"],ascending=[False,False],na_position="last")
        frame.to_csv(ROOT/"experiments"/"label_leaderboard.csv",index=False)
        print(frame[["source","rows","gold_overlap","macro_auc","matched_only_macro_auc","all_gold_exact_copy_flag"]].to_string(index=False))
        eligible=[r for r in board if r["eligible_for_teacher_comparison"]]
        common=set(gold[UID])
        loaded={}
        for r in eligible:
            df,_=load_teacher(ROOT/r["file"])
            df=df.set_index(UID)[TARGETS]
            loaded[r["source"]]=df
            common &= set(df.dropna().index)
        ids=sorted(common)
        common_rows=[]
        if ids:
            y=gold.set_index(UID).loc[ids,TARGETS].to_numpy(float)
            if np.all(np.ptp(y,axis=0)>0):
                for name,df in loaded.items():
                    p=df.loc[ids].to_numpy(float)
                    a=auc_vector(y,p)
                    common_rows.append({"source":name,"n_common_gold":len(ids),"macro_auc":float(a.mean()),**dict(zip(TARGETS,a))})
                pd.DataFrame(common_rows).sort_values("macro_auc",ascending=False).to_csv(ROOT/"experiments"/"label_common_gold_metrics.csv",index=False)
                (ROOT/"gold"/"teacher_common_uids.json").write_text(json.dumps(ids,indent=2),encoding="utf-8")
                print("COMMON_GOLD",len(ids),"TEACHER_TABLES",len(common_rows))
    print("ADAPTERS_REQUIRED",json.dumps(skipped))

if __name__=="__main__": main()
