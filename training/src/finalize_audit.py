"""Local-only registries, result tables, frozen control, and immutable snapshots."""
from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

ROOT=workspace_root()
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def load(path):
    raw=Path(path).read_bytes()
    return json.loads(raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def upsert(path,rows,key):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=list(reader.fieldnames or []);old=list(reader)
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    values={r[key]:r for r in old}
    for row in rows:values[row[key]]={**values.get(row[key],{}),**row}
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values.values())

def table(df):
    def cell(x):
        if pd.isna(x):return '--'
        return (f'{x:.5f}' if isinstance(x,float) else str(x)).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join('---' for _ in df.columns)+' |\n'+'\n'.join('| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot',action='store_true');args=ap.parse_args()
    now=datetime.now(timezone.utc).isoformat()
    labels=load(ROOT/'reports/label_audit_receipt.json')
    consensus=load(ROOT/'reports/EXP003_teacher_consensus.json')
    public=load(ROOT/'reports/public_prediction_audit.json')
    geometry=load(ROOT/'reports/kaggle_train_geometry_v1/geometry_audit_summary.json')
    binary=load(ROOT/'metadata/component_static_audit.json')
    cp=ROOT/'reports/kaggle_coat_gold_v1/coat_gold58_reproduction_receipt.json'
    component=load(cp) if cp.exists() else None
    component_status=component['status'] if component else ('FAILED' if (cp.parent/'component_failure.json').exists() else 'RESULT_NOT_RETRIEVED')
    best=max((r for r in labels['teachers'] if r['eligible_for_teacher_comparison'] and r['full58_comparable']),key=lambda r:r['macro_auc'])
    per=pd.read_csv(ROOT/'experiments/label_per_target_metrics.csv')
    image_per=pd.read_csv(ROOT/'experiments/public_prediction_per_target.csv')
    alltargets=pd.read_csv(ROOT/'gold/gold_target_counts.csv').set_index('target')[['positive_count']]
    alltargets['Flight_teacher_AUC']=per.loc[per.source==best['source']].set_index('target').auc
    for name,title in [('Pilkwang_public_OOF','Public_DINO_gold_AUC'),('CoAt_top3_probability','Public_CoAt_probability_AUC'),('CoAt_top3_rank','Public_CoAt_rank_AUC')]:
        alltargets[title]=image_per.loc[image_per.model==name].set_index('target').auc
    if component:alltargets['Reexecuted_CoAt_rank_AUC']=pd.Series(component['per_target_rank_auc'])
    alltargets=alltargets.reindex(TARGETS).reset_index()
    alltargets.to_csv(ROOT/'experiments/per_target_summary.csv',index=False)
    rows=[
      {'experiment_id':'EXP001_PUBLIC_SOTA_REPRO','base_model':'mattiaangeli/bend-the-knee-to-the-dinosaurs','metric_kind':'FULL_PUBLIC_PIPELINE','status':'BLOCKED_LICENSE_VERSION_RUNTIME','keep':'control_candidate','notes':'Source frozen; NC-SA RadImageNet compatibility unresolved; no complete inference or LB reproduction.'},
      {'experiment_id':'EXP001A_COAT_COMPONENT_REPRO','base_model':'CoAt residual-gated top3 e04/e06/e08','resolution':384,'crop_mm':130,'window_count':'max76','metric_kind':'TRAIN_IMAGE_COMPONENT_INFERENCE','status':component_status,'keep':'diagnostic','gold_macro_auc':component['rank_macro_auc'] if component else '', 'runtime_sec':component['runtime_sec'] if component else '', 'notes':'Not full SOTA. Public model/pixels with our logged scheduler. Gold-selected checkpoints.'},
      {'experiment_id':'EXP002_PUBLIC_LABEL_AUDIT','label_source':best['source'],'label_version':best['sha256'],'gold_macro_auc':best['macro_auc'],'gold_bootstrap_mean':best['bootstrap']['mean'],'gold_bootstrap_std':best['bootstrap']['std'],'metric_kind':'REPORT_TEACHER_ONLY','status':'COMPLETED','keep':'teacher_control','notes':'19 datasets,21CSV tables,8gold-overwritten excluded; not MRI AUC.'},
      {'experiment_id':'EXP003_FAMILY4_CONSENSUS','label_source':'Steven raw/Pilkwang LLM/Lixin raw/Qwen raw','gold_macro_auc':consensus['variants'][0]['macro_auc'],'metric_kind':'REPORT_TEACHER_ONLY','status':'COMPLETED_NO_GAIN','keep':'no','notes':'Predeclared equal weights. Paired delta CI below zero. No student training.'},
      {'experiment_id':'EXP003B_UNKNOWN_HALF','label_source':'same4; explicit unknown to0.5','gold_macro_auc':consensus['variants'][1]['macro_auc'],'metric_kind':'REPORT_TEACHER_ONLY','status':'COMPLETED_NO_GAIN','keep':'no'},
      {'experiment_id':'EXP004_UNCERTAINTY_EXPORT','label_source':'four families','loss':'candidate clip(1-4variance,0,1)','metric_kind':'TARGET_PREPARATION_ONLY','status':'EXPORTED_NOT_TRAINED','keep':'features','notes':'4349 non-gold training rows; weighting effectiveness not tested.'},
      {'experiment_id':'EXP005_TRAIN_GEOMETRY_AUDIT','crop_mm':140,'metric_kind':'DICOM_AUDIT_ONLY','status':'KAGGLE_COMPLETED','runtime_sec':geometry['elapsed_seconds'],'keep':'audit','notes':'74 train studies,428series,102decoded renders; crop AUC not measured.'},
      {'experiment_id':'EXP010_PUBLIC_RANK_VS_PROBABILITY','base_model':'CoAt public Gold58 output','gold_macro_auc':next(r['gold_macro_auc'] for r in public['models'] if r['model']=='CoAt_top3_probability'),'metric_kind':'PUBLIC_PREDICTION_RESCORE','status':'COMPLETED_CI_INCLUDES_ZERO','keep':'diagnostic','notes':'Probability.911559/rank.909470. Selection bias not corrected by bootstrap.'},
    ]
    for row in rows:row['datetime']=now
    upsert(ROOT/'experiments/experiments.csv',rows,'experiment_id')
    names=['ACL_auc','MCL_auc','MM_auc','LM_auc','MedOA_auc','LatOA_auc','PFOA_auc','Effusion_auc','Synovitis_auc','Baker_auc','Contusion_auc','Fracture_auc']
    lr=[]
    for i,r in enumerate(labels['teachers'],1):
        a=per.loc[per.source==r['source']].set_index('target').auc
        row={'label_exp':f'LBL{i:03d}','teacher':r['source'],'prompt_version':'public','language_method':'source-specific','negation':'source-specific','severity_logic':'source-specific','unknown_handling':r['adapter'],'macro_auc':r['macro_auc'] if r['macro_auc'] is not None else '', 'notes':f"matched={r['gold_overlap']};eligible={r['eligible_for_teacher_comparison']};gold_exact={r['all_gold_exact_copy_flag']};teacher_not_image"}
        row.update({n:float(a[t]) if pd.notna(a[t]) else '' for n,t in zip(names,TARGETS)});lr.append(row)
    upsert(ROOT/'experiments/label_experiments.csv',lr,'label_exp')
    mr=[]
    for i,r in enumerate(binary['weights'],1):
        path=workspace_path(r['path']);coat='gold_e' in path.name;v8='v8_' in path.name
        geom='384px/130mm/within-series/max76' if coat else ('384px/140mm/44slices/42windows/span.06-.94' if v8 else '140mm/64slices/62windows/span.02-.98')
        mr.append({'model_id':f'W{i:03d}','source':path.parent.name.replace('__','/'),'version':path.name,'path':r['path'],'sha256':r['sha256'],'bytes':r['bytes'],'architecture':'CoAt spatial residual finding MIL' if coat else 'CoAt Raptor finding MIL','training_labels':'SHA in training history; source mapping unverified' if coat else 'public weak labels; exact label hash unverified','fold':'gold-selected epoch' if coat else 'not independently verified','preprocessing':geom,'inference_geometry':geom,'license':'CC0 dataset declaration; upstream rights not independently audited','status':'acquired_sha_verified'})
    upsert(ROOT/'experiments/model_manifest.csv',mr,'model_id')
    catalog=[]
    for folder in sorted((ROOT/'kaggle_notebooks').iterdir()):
        mp=folder/'kernel-metadata.json'
        if folder.name.startswith('own_') or not mp.exists():continue
        m=load(mp);p=folder/m['code_file']
        if not p.exists():continue
        text=(folder/'source_review.py').read_text(encoding='utf-8') if (folder/'source_review.py').exists() else ''
        families=[label for word,label in [('dinov2','DINOv2'),('radimage','RadImageNet'),('coatnet','CoAtNet'),('RaptorClassifier','Raptor MIL'),('DiagnosisSlotHead','pathology query fusion')] if word.lower() in text.lower()]
        score='not verified';evidence='source SHA pinned; numeric current score/version absent'
        if m['id']=='hdhsjdjd/rsna-knee-raptor-v5-w035':score='0.933';evidence='indexed V1; not reproduced'
        if m['id']=='nishantkharga/rsna-knee-full-4-arm-ensemble-v55':score='0.936';evidence='indexed olderURL V56; downloaded-version identity unresolved'
        if '0941' in m['id']:score='0.941 title only'
        if '0-937' in m['id'] or '0937' in m['id']:score='0.937 title only'
        catalog.append({'ID':f'NB{len(catalog)+1:03d}','Notebook':m['id'],'Public_LB':score,'Score_evidence':evidence,'Architecture_source_references':';'.join(families),'Input_geometry':'See config_evidence/source; not inferred from title','Labels':'Exact training source partly unresolved','Ensemble':'Deduplicate forks/checkpoints','Runtime':'Full1300 not measured','License':'Notebook/input licenses are different','Author':m['id'].split('/')[0],'Version':'numeric unresolved; content SHA pinned','Copied_from':'unresolved unless explicitly attributed','Dependent_datasets':';'.join(m.get('dataset_sources',[])),'Dependent_models':';'.join(m.get('model_sources',[])),'Source_SHA256':sha(p)})
    pd.DataFrame(catalog).to_csv(ROOT/'reports/public_solutions.csv',index=False)
    external=[]
    for area in ['labels','external','weights','processed']:
        for mp in (ROOT/area).glob('*/dataset-metadata.json'):
            info=load(mp).get('info',{});name=mp.parent.name.replace('__','/')
            license_names=';'.join(x.get('name','') for x in info.get('licenses',[]));status='DECLARED_PUBLIC_LICENSE_UPSTREAM_UNVERIFIED'
            if 'radimagenet-marwan' in name or 'barun2104' in name:status='HOLD_RESTRICTIVE_OR_OTHER_LICENSE'
            if any(s in name for s in ['tsuyu122','jezrealmomoh','zaidaliiq1000']):status='QUARANTINE_PROVENANCE_OR_LICENSE'
            if 'juanicarlos' in name:status='INVALID_HTML_ARTIFACT'
            external.append({'name':name,'dataset_id':info.get('datasetId'),'source':'https://www.kaggle.com/datasets/'+name,'declared_license':license_names,'access':'public Kaggle download','commercial_rights':'not independently established','competition_compatibility':status,'metadata_path':str(mp.relative_to(ROOT))})
    pd.DataFrame(external).to_csv(ROOT/'reports/external_data_audit.csv',index=False)
    gate={'ready_to_execute':False,'ready_to_submit':False,'blocking':['NC-SA RadImageNet input/winner-license compatibility unresolved','Current numeric source version and LB identity unresolved','Complete input hashes not acquired','1300study runtime not measured'],'not_blocked':['Kaggle authentication','official data access','local disk'],'do_not':['silently drop an arm','call a title score observed LB','treat gold-selected scores as independent validation']}
    save(ROOT/'reports/EXP001_release_gate.json',gate)
    folder=ROOT/'kaggle_submission/EXP001_PUBLIC_SOTA_REPRO';folder.mkdir(exist_ok=True)
    original=ROOT/'kaggle_notebooks/mattiaangeli__bend-the-knee-to-the-dinosaurs';m=load(original/'kernel-metadata.json')
    dest=folder/'public_sota_original.ipynb'
    if not dest.exists():shutil.copy2(original/m['code_file'],dest)
    if sha(dest)!=sha(original/m['code_file']):raise ValueError('Frozen source changed')
    meta={k:v for k,v in m.items() if k not in ['id_no','id','title','code_file']}
    meta.update({'id':'taeyangg4/rsna-knee-public-sota-exp001','title':'RSNA Knee Public SOTA EXP001','code_file':dest.name,'is_private':True,'enable_internet':False})
    save(folder/'kernel-metadata.json',meta);save(folder/'release_gate.json',gate)
    candidates={'FINAL_A':{'status':'DESIGN_ONLY_NOT_ELIGIBLE','proposal':'licensed exact strongest ensemble plus validated specialists/coverage'},'FINAL_B':{'status':'DESIGN_ONLY_NOT_ELIGIBLE','proposal':'permissive-input simpler CoAt/DINO blend with group-aware validation'}}
    save(ROOT/'experiments/final_candidates.json',candidates)
    submission={'competition_submissions_this_session':0,'submission_id':None,'public_score':None,'rank':None,'submission_csv':None,'daily_remaining_observed':5,'final_selection_limit_verified':False}
    save(ROOT/'reports/submission_status.json',submission)
    execution=[{'kernel':'taeyangg4/rsna-knee-training-geometry-audit','version':1,'status':'COMPLETE','type':'CPU_TRAIN_ONLY','internal_runtime_sec':geometry['elapsed_seconds'],'competition_submission':False},{'kernel':'taeyangg4/rsna-knee-coat-gold-component-reproduction','version':1,'status':component_status,'type':'GPU_TRAIN_ONLY','internal_runtime_sec':component['runtime_sec'] if component else None,'competition_submission':False}]
    pd.DataFrame(execution).to_csv(ROOT/'experiments/kaggle_execution_log.csv',index=False)
    summary={'utc':now,'public_notebooks':len(catalog),'label_datasets':19,'label_csv_tables':len(labels['teachers']),'gold_overwrite_excluded':sum(r['all_gold_exact_copy_flag'] for r in labels['teachers']),'actual_weight_files':len(mr),'actual_weight_bytes':sum(r['bytes'] for r in binary['weights']),'best_teacher':best,'component_status':component_status,'component':component,'geometry':geometry,'consensus':consensus,'submission':submission,'release_gate':gate,'git_exists':(ROOT/'.git').exists(),'disk_free_bytes':shutil.disk_usage(ROOT).free}
    save(ROOT/'reports/session_summary.json',summary)
    rendered='# Recomputed evidence tables\n\nTeacher AUC, public prediction rescore, and actual MRI inference are not interchangeable.\n\n## Per target\n\n'+table(alltargets)+'\n## Experiments\n\n'+table(pd.DataFrame(rows)[['experiment_id','metric_kind','status','gold_macro_auc','runtime_sec','keep']].fillna(''))+'\n## Verified checkpoints\n\n'+table(pd.DataFrame(mr)[['model_id','version','sha256','inference_geometry']])
    (ROOT/'reports/execution_tables.md').write_text(rendered,encoding='utf-8')
    records=[]
    for area in ['src','metadata','data','labels','weights','external','processed','kaggle_notebooks','gold','predictions','reports','experiments','kaggle_submission']:
        for p in sorted((ROOT/area).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.name=='artifact_manifest.csv':continue
            records.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p),'source':'see source manifests','acquired_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),'license':'see external_data_audit.csv','status':'present_hashed'})
    pd.DataFrame(records).to_csv(ROOT/'experiments/artifact_manifest.csv',index=False)
    if args.snapshot:
        captures={'EXP001_PUBLIC_SOTA_REPRO':['reports/EXP001_release_gate.json','reports/solution_lineage.md'],'EXP002_PUBLIC_LABEL_AUDIT':['reports/label_audit_receipt.json','experiments/label_per_target_metrics.csv'],'EXP003_LABEL_CONSENSUS':['experiments/EXP003_label_consensus_config.json','reports/EXP003_teacher_consensus.json'],'EXP005_TRAIN_GEOMETRY':['reports/kaggle_train_geometry_v1/geometry_audit_summary.json','reports/kaggle_train_geometry_v1/train_series_geometry_audit.csv'],'EXP001A_COAT_COMPONENT':['reports/kaggle_coat_gold_v1/coat_gold58_reproduction_receipt.json']}
        for exp,files in captures.items():
            snapshot=ROOT/'snapshots'/(exp+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
            if snapshot.exists():continue
            for name in ['code','config','logs','metrics']:(snapshot/name).mkdir(parents=True,exist_ok=True)
            for p in (ROOT/'src').glob('*.py'):shutil.copy2(p,snapshot/'code'/p.name)
            save(snapshot/'config/capture.json',{'captured_utc':now,'experiment':exp,'capture_stage':'post_execution_audit','git_created':False,'weights_referenced_by_sha':True})
            for rel in files:
                p=ROOT/rel
                if p.exists():shutil.copy2(p,snapshot/'metrics'/p.name)
            shutil.copy2(ROOT/'experiments/experiments.csv',snapshot/'logs/experiments.csv')
            shutil.copy2(ROOT/'experiments/model_manifest.csv',snapshot/'config/model_manifest.csv')
            save(snapshot/'SHA256.json',[{'path':str(p.relative_to(snapshot)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(snapshot.rglob('*')) if p.is_file()])
    print(json.dumps({'public_notebooks':len(catalog),'label_csv_tables':len(labels['teachers']),'checkpoints':len(mr),'artifact_files':len(records),'component_status':component_status,'snapshot':args.snapshot},indent=2))

if __name__=='__main__':main()
