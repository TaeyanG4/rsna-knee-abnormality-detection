"""Build a train-only report review queue and contamination-aware source inventory."""
from __future__ import annotations
import hashlib
import json
import pandas as pd
import numpy as np
from label_audit import ROOT,TARGETS,UID,load_teacher,auc_vector

def main():
    train=pd.read_csv(ROOT/'data'/'train.csv',dtype={UID:str}).set_index(UID)
    gold=pd.read_csv(ROOT/'gold'/'gold_labels.csv',dtype={UID:str}).set_index(UID)
    rel='flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv'
    teacher,_=load_teacher(ROOT/'labels'/rel);teacher=teacher.set_index(UID)
    joined=teacher.loc[gold.index,TARGETS];rows=[]
    for j,t in enumerate(TARGETS):
        y=gold[t].to_numpy();p=joined[t].to_numpy()
        for i in np.flatnonzero((p>=.5)!=y):
            uid=gold.index[i]
            rows.append({UID:uid,'target':t,'gold':int(y[i]),'teacher_probability':float(p[i]),
                'error_at_0_5':'false_positive' if p[i]>=.5 else 'false_negative',
                'absolute_error':float(abs(p[i]-y[i])),'review_status':'UNREVIEWED',
                'error_taxonomy':'UNASSIGNED','Report':train.loc[uid,'Report']})
    pd.DataFrame(rows).sort_values(['target','absolute_error'],ascending=[True,False]).to_csv(ROOT/'reports'/'label_error_review_queue_train_only.csv',index=False)
    out=ROOT/'labels'/'derived_EXP002';out.mkdir(exist_ok=True)
    weak=train.index[~train.index.isin(gold.index)]
    assert len(weak)==4349 and not(set(weak)&set(gold.index))
    teacher.loc[weak,TARGETS].to_csv(out/'flight_public_control_weak4349.csv')
    (out/'provenance.json').write_text(json.dumps({'source':rel,'source_sha256':hashlib.sha256((ROOT/'labels'/rel).read_bytes()).hexdigest(),'rows':len(weak),'gold_excluded':True,'derived_public_blend_not_independent_teacher':True,'trained_student':False},indent=2),encoding='utf-8')
    special=[]
    parquet=ROOT/'labels'/'tsuyu122__rsna-knee-mri-balanced-labels-public'/'labels.parquet'
    if parquet.exists():
        df=pd.read_parquet(parquet).set_index(UID)
        matched=gold.index.intersection(df.index)
        exact=len(matched)==58 and np.array_equal(gold.loc[matched,TARGETS].to_numpy(),df.loc[matched,TARGETS].to_numpy())
        special.append({'source':str(parquet.relative_to(ROOT)),'rows':len(df),'gold_overlap':len(matched),'all_gold_exact_copy':bool(exact),'status':'QUARANTINE','reason':'External-cohort access/licensing and label provenance unverified; README full-RSNA-gold claim conflicts with official 58 gold'})
    html=ROOT/'labels'/'juanicarlos__rsna-knee-llm-labels'/'rsna-knee-llm-labels'
    if html.exists():
        text=html.read_text(encoding='utf-8',errors='replace').lstrip().lower()
        special.append({'source':str(html.relative_to(ROOT)),'html_detected':text.startswith('<!doctype html') or '<html' in text[:500],
            'status':'INVALID_LABEL_ARTIFACT','reason':'Downloaded payload is HTML/CAPTCHA, not probabilities; never executed or bypassed'})
    (ROOT/'reports'/'special_label_sources.json').write_text(json.dumps(special,indent=2),encoding='utf-8')
    print('TRAIN_ONLY_REVIEW_ROWS',len(rows),'CONTROL_WEAK_EXPORT',len(weak))
    print(json.dumps(special,indent=2))

if __name__=='__main__':main()
