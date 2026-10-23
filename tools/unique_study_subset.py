"""Freeze the preregistered HA18 component-hash prefix; CPU only, no training."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UID='StudyInstanceUID'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def uid_hash(ids):return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    manifest_path=ROOT/'artifacts/experiments/EXP094_B8_VARIANT_PROTOCOL_V4/manifest.json'
    m=json.loads(manifest_path.read_text());op=ROOT/m['ordered_ids']['path'];gp=ROOT/m['groups']['path']
    assert sha(op)==m['ordered_ids']['sha256'] and sha(gp)==m['groups']['sha256']
    ordered=json.loads(op.read_text());train=ordered['train'];val=ordered['val'];assert len(train)==3455 and len(val)==888
    with gp.open(newline='',encoding='utf-8') as f:groups={r[UID]:r for r in csv.DictReader(f)}
    bycomp=defaultdict(list)
    for u in train:bycomp[groups[u]['component_id']].append(u)
    assert not(set(bycomp)&{groups[u]['component_id'] for u in val})
    prefix='HA18-v1|20261007|'
    comps=sorted(bycomp,key=lambda c:(hashlib.sha256((prefix+c).encode()).hexdigest(),c))
    cumulative=[0]
    for c in comps:cumulative.append(cumulative[-1]+len(bycomp[c]))
    n=min(range(len(cumulative)),key=lambda i:(abs(cumulative[i]-2764),cumulative[i]))
    selected=set(comps[:n]);subset=[u for u in train if groups[u]['component_id'] in selected]
    a.out.mkdir(parents=True,exist_ok=True);dest=a.out/'subset_manifest.json'
    if dest.exists():raise SystemExit('Refuse to overwrite frozen subset')
    rec={'schema':'hc05-unique-study-subset-v1','source_plan':'H-C-05 from H-A-18',
         'rule':'component SHA256 prefix closest to2764, ties fewer studies; preserve full canonical UID order',
         'prefix':prefix,'target_studies':2764,'actual_studies':len(subset),'components':len(selected),
         'full_train':train,'subset_train':subset,'validation':val,'selected_components':comps[:n],
         'ordered_uid_sha256':{k:uid_hash(v) for k,v in [('full_train',train),('subset_train',subset),('validation',val)]},
         'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [manifest_path,op,gp]},
         'selection_uses_labels_or_scores':False,'training_launched':False,'control_reuse_gate':'PENDING',
         'control_seeds':{'42':{'attempts':3448,'successful':3448},'43':{'attempts':3448,'successful':3447}},
         'gpu_budget_seconds':{'estimate':5400,'reserve':6480,'hard_cap':7200}}
    with dest.open('x',encoding='utf-8') as f:json.dump(rec,f,indent=2)
    # Descriptive distribution checks happen only after selection is frozen.
    lp=ROOT/m['paths']['labels']
    with lp.open(newline='',encoding='utf-8') as f:lr=csv.DictReader(f);targets=lr.fieldnames[1:];labels={r[UID]:r for r in lr}
    desc={}
    for name,ids in [('full',train),('subset',subset),('validation',val)]:
        y=np.rint(np.asarray([[float(labels[u][t]) for t in targets] for u in ids],dtype=np.float32))
        desc[name]={'studies':len(ids),'positive_counts':dict(zip(targets,y.sum(0).astype(int).tolist())),
                    'negative_counts':dict(zip(targets,(len(ids)-y.sum(0)).astype(int).tolist())),
                    'group_distributions':{k:dict(Counter(groups[u][k] for u in ids)) for k in m['groups']['subgroup_columns']}}
    desc.update(subset_manifest_sha256=sha(dest),label_file_sha256=sha(lp),selection_repeated=False)
    with (a.out/'descriptive_counts.json').open('x',encoding='utf-8') as f:json.dump(desc,f,indent=2)
    print(json.dumps({k:rec[k] for k in ['actual_studies','components','ordered_uid_sha256','control_reuse_gate']},indent=2))
if __name__=='__main__':main()
