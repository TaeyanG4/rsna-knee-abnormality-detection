"""Read-only CoAt-family receipt classification for future release diagnostics.

No existing release imports this module. Missing audit metadata is not a failed
prediction. A usable current-run fallback never claims to restore a scored run.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MEMBERS = ('resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3')
TARGETS = ('ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture')
CONTRACT = 'public_raptor_private_residual_coat_v32_no_inner_rerank_v1'

def classify(receipt, audit=None, *, csv_valid, study_count):
    """Classify evidence separately from the preserve-valid-output policy.

    csv_valid must come from the caller's actual cohort/schema/value validation.
    full_reported concerns four CoAt family members, not the entire anchor graph.
    """
    notes=[];conflicts=[]
    audit=audit if isinstance(audit,dict) else {}
    events=audit.get('events') or []
    events=events if isinstance(events,list) else []
    failures=[e for e in events if isinstance(e,dict) and str(e.get('kind','')).startswith('coat_') and any(w in str(e.get('kind','')) for w in ('child_failed','output_rejected','probability_unavailable','family_partial','family_unavailable'))]
    count=audit.get('coat_family_members')
    count_valid=type(count) is int and 0<=count<=4
    if not audit:notes.append('live audit namespace unavailable; do not interpret as member failure')
    if count is not None and not count_valid:conflicts.append('invalid audit member count')
    members=None;state='unknown';recipe='unknown'
    if isinstance(receipt,dict):
        members=receipt.get('members')
        if isinstance(members,list) and all(isinstance(m,str) for m in members) and len(members)==len(set(members)) and set(members)<=set(MEMBERS):
            state='full_reported' if len(members)==4 else 'partial_reported' if members else 'none_reported'
            if count_valid and count!=len(members):conflicts.append('receipt and audit member counts disagree')
            if type(receipt.get('study_count')) is not int or receipt.get('study_count')!=study_count:conflicts.append('receipt study count missing or mismatched')
            expected={'contract':CONTRACT,'inner_rerank':False,'finding_specific_weights':False,'private_alpha':0.4,'public_raptor_alpha':0.6}
            missing=[k for k in expected if k not in receipt]
            changed=[k for k,v in expected.items() if k in receipt and (receipt[k]!=v or (isinstance(v,bool) and type(receipt[k]) is not bool))]
            reduction=receipt.get('family_reduction')
            weights=receipt.get('within_coat')
            weight_ok=isinstance(weights,dict) and set(weights)==set(members) and bool(members) and all(type(v) in (float,int) and math.isfinite(v) and abs(v-1/len(members))<1e-12 for v in weights.values())
            if missing or reduction is None or weights is None:
                notes.append('recipe fields incomplete: '+','.join(missing));recipe='unknown'
            elif changed or reduction!='rank_of_member_probability_mean' or not weight_ok:
                recipe='changed_reported'
            else:recipe='expected_reported'
        else:conflicts.append('receipt member identities are missing, duplicate or unknown')
    elif receipt is not None:
        conflicts.append('receipt is not an object')
    else:
        notes.append('producer receipt unavailable; valid CSV remains usable')
        # Producer returns before receipt write if no CoAt arm is usable.
        if count_valid and count==0 and any(e.get('kind')=='coat_family_unavailable' for e in failures):
            state='none_reported';recipe='public_raptor_only_reported'
    if failures:
        notes.append('CoAt degradation events are reported; inspect their exact kind')
        if state=='full_reported' and recipe=='expected_reported':conflicts.append('full expected recipe conflicts with degradation events')
    if conflicts:state='conflicting_evidence'
    full=state=='full_reported' and recipe=='expected_reported'
    return {
        'schema':'anchor-receipt-policy-v1',
        'policy':'preserve_valid_current_run_output',
        'csv_valid':bool(csv_valid), 'coat_member_state':state,
        'coat_reduction_state':recipe, 'members':members,
        'full_coat_contract_reported':full,
        'current_anchor_use_permitted':bool(csv_valid),
        'output_action':'preserve_usable_current_run_csv' if csv_valid else 'reject_for_blending_do_not_overwrite',
        'execution_label':'current_run_'+state+'_coat_anchor' if csv_valid else 'unusable_current_run_anchor',
        'audit_status':audit.get('status'), 'failure_events':failures,
        'evidence_conflicts':conflicts,'notes':notes,
        'scored_anchor_prediction_or_score_recovered':False,
        'scope':'CoAt evidence only; neither other anchor families, run binding, runtime, score equivalence nor own-arm gates are established. No production integration.',
    }

def validate_csv(submission, sample):
    expected=['StudyInstanceUID',*TARGETS]
    try:
        with Path(sample).open(encoding='utf-8',newline='') as f:
            rd=csv.DictReader(f)
            if rd.fieldnames!=expected:return False,0,'sample column contract'
            ids=[r['StudyInstanceUID'] for r in rd]
    except (OSError,UnicodeError,csv.Error):return False,0,'unreadable/invalid sample'
    if len(ids)!=len(set(ids)) or not ids or any(not x for x in ids):return False,len(ids),'sample UID contract'
    try:
        with Path(submission).open(encoding='utf-8',newline='') as f:
            rd=csv.DictReader(f)
            if rd.fieldnames!=expected:return False,len(ids),'submission columns'
            rows=list(rd)
        if [r['StudyInstanceUID'] for r in rows]!=ids:return False,len(ids),'submission UID/order contract'
        for r in rows:
            if None in r:return False,len(ids),'extra submission row fields'
            for t in TARGETS:
                v=float(r[t])
                if not math.isfinite(v) or not 0<=v<=1:return False,len(ids),'nonfinite/out-of-range probability'
    except (OSError,UnicodeError,csv.Error,ValueError,KeyError,TypeError):return False,len(ids),'unreadable/invalid submission'
    return True,len(ids),None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipt',type=Path)
    ap.add_argument('--audit',type=Path)
    ap.add_argument('--submission',type=Path,required=True)
    ap.add_argument('--sample',type=Path,required=True)
    ap.add_argument('--current-anchor-backup',type=Path)
    a=ap.parse_args()
    evidence={};errors=[]
    for name,path in [('receipt',a.receipt),('audit',a.audit)]:
        try:evidence[name]=json.loads(path.read_text(encoding='utf-8')) if path else None
        except (OSError,ValueError) as exc:evidence[name]=None;errors.append(name+': '+type(exc).__name__)
    ok,n,error=validate_csv(a.submission,a.sample)
    result=classify(evidence['receipt'],evidence['audit'],csv_valid=ok,study_count=n)
    result.update(csv_error=error,metadata_read_errors=errors)
    result['current_run_backup_bytes_match']=None
    if a.current_anchor_backup and a.current_anchor_backup.is_file() and a.submission.is_file():
        result['current_run_backup_bytes_match']=hashlib.sha256(a.current_anchor_backup.read_bytes()).digest()==hashlib.sha256(a.submission.read_bytes()).digest()
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
