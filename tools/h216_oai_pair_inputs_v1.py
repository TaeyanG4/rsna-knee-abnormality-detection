"""H216 fixed production-input binding v1.0.0, metadata only.

Reuses completed native-task data; freezes the exact800 OAI draws for400updates.
No raw census, pixels, checkpoint, GPU, allocation or label output to stdout.
"""
from __future__ import annotations
import csv
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from tools.l1_rco_stage0_v1 import read,write,sha,digest,need,LABELS
from tools import h216_oai_pair_core_v1 as core
from training.arms.oai_exclusions import participant

OUT=ROOT/'.local/scratch/slot-b/h216-oai-pair-inputs-v1'
INPUT=ROOT/'.local/scratch/slot-b/h216-oai-inputs-v1'
K1=ROOT/'.local/scratch/l1-rco/k1v3-grouping'


def schedule_binding(oai_ids,rsna_ids):
    need(len(set(oai_ids))==len(oai_ids) and len(set(rsna_ids))==len(rsna_ids),'DUPLICATE_ROSTER')
    need(oai_ids==sorted(oai_ids) and rsna_ids==sorted(rsna_ids),'UNSORTED_ROSTER')
    oa=core.batch_schedule(len(oai_ids),400,21601)
    ra=core.batch_schedule(len(rsna_ids),400,21601)
    recovery=core.batch_schedule(len(rsna_ids),800,21602)
    selected=[oai_ids[i] for _,batch in oa for i in batch]
    need(len(selected)==len(set(selected))==800,'EXACT800_DISTINCT_DRAWS')
    return dict(oai_adapt=oa,rsna_adapt=ra,rsna_recovery=recovery,
                oai_required_ids=selected,oai_logical_roster=oai_ids,rsna_logical_roster=rsna_ids)


def native_targets(rows,tasks,excluded):
    need(len(tasks)==len(set(tasks))==30,'TASK_ORDER_CARDINALITY')
    ids=[r['study_key'] for r in rows]
    need(ids==sorted(ids) and len(ids)==len(set(ids)),'NATIVE_ROSTER')
    target=np.full((len(rows),30),-1,np.int64)
    for i,r in enumerate(rows):
        need(participant(r['study_key']) not in excluded,'EXCLUDED_PARTICIPANT')
        need(len(r['values'])==len(r['mask'])==30,'NATIVE_WIDTH')
        need(r['mask']==[int(v is not None) for v in r['values']],'MASK_DRIFT')
        need(all(v is None or type(v) is int and 0<=v<=3 for v in r['values']),'NATIVE_GRADE')
        target[i]=[-1 if v is None else v for v in r['values']]
        need(np.any(target[i]>=0),'EMPTY_NATIVE_STUDY')
    return target


def rsna_targets(path,ids,holdout):
    with Path(path).open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    by={r['StudyInstanceUID']:r for r in rows}
    need(len(by)==len(rows) and set(by)==set(ids) and not set(ids)&set(holdout),'RSNA_TARGET_ROSTER')
    target=np.empty((len(ids),12),np.float32)
    for i,uid in enumerate(ids):
        r=by[uid]
        need(r['is_gold']=='0' and r['review_holdout']=='0' and int(r['fold']) not in (-3,0),'RSNA_EXCLUSION')
        target[i]=[float(r[label+'|prior']) for label in LABELS]
    need(np.isfinite(target).all() and np.all((target>=0)&(target<=1)),'RSNA_TARGET_RANGE')
    return target


class ScheduledStudies:
    """Keep the original1703 index space; cache only the predetermined800 draws.

    The core's permutation/target indexing remains unchanged. Unscheduled reads
    fail explicitly instead of substituting another study or rerolling the seed.
    """
    def __init__(self,ct,cache,logical_ids,required_ids):
        self.uids=list(logical_ids); self.epoch=0; self.seed=22601
        need(len(set(required_ids))==len(required_ids) and set(required_ids)<=set(logical_ids),'REQUIRED_ROSTER')
        self.lookup={self.uids.index(uid):j for j,uid in enumerate(required_ids)}
        self.delegate=ct.Studies(str(cache),required_ids,64,'train',self.seed,train_sampling='cover',
                                 cover_stride=3,slices=3,slice_jitter=.25,phys_mm=3.5)
        for k in ('mode','n_win','train_sampling','cover_stride','slices','slice_jitter','phys_mm'):
            setattr(self,k,getattr(self.delegate,k))
    def __len__(self):return len(self.uids)
    def __getitem__(self,i):
        need(i in self.lookup,'UNSCHEDULED_STUDY_READ')
        # Frozen Studies seeds windows using seed+epoch+index. Offset its seed so
        # physical800 index produces exactly the original logical1703 RNG stream.
        j=self.lookup[i]; self.delegate.epoch=self.epoch; self.delegate.seed=self.seed+i-j
        x,m,_=self.delegate[j]
        return x,m,i


def prepare():
    need(not OUT.exists(),'OUTPUT_EXISTS')
    result=read(INPUT/'RESULT.json'); reg=read(INPUT/'REGISTRATION.json')
    need(result['status']=='CPU_INPUT_TASK_PREPARED','NATIVE_INPUT_INCOMPLETE')
    for n,h in result['outputs'].items(): need(sha(INPUT/n)==h,'INPUT_OUTPUT_DRIFT')
    cohort=read(INPUT/'cohort.private.json'); rows=cohort['studies']
    native=native_targets(rows,cohort['tasks'],set(cohort['excluded_participants']))
    k1=read(K1/'registration.json'); prep=read(K1/'prepare.json')
    rsna_ids=k1['train']; holdout=k1['holdout']; oai_ids=[r['study_key'] for r in rows]
    need((len(oai_ids),len(rsna_ids),len(holdout))==(1703,3190,977),'EXACT_ROSTER_COUNTS')
    need(sha(K1/'targets.csv')==prep['targets_sha256'],'ORIGINAL_L2_DRIFT')
    rsna=rsna_targets(K1/'targets.csv',rsna_ids,holdout)
    schedule=schedule_binding(oai_ids,rsna_ids); required=set(schedule['oai_required_ids'])
    selected=[r for r in rows if r['study_key'] in required]
    pilot=read(INPUT/'pilot8.private.json')['study_ids']
    paths=[Path(__file__),ROOT/'tools/h216_oai_pair_core_v1.py',ROOT/'training/arms/oai_exclusions.py',
           ROOT/'tests/test_h216_oai_pair_inputs_v1.py',ROOT/'docs/research/h216-oai-pair-inputs-v1.md',
           INPUT/'REGISTRATION.json',INPUT/'RESULT.json',INPUT/'cohort.private.json',INPUT/'pilot8.private.json',
           K1/'registration.json',K1/'prepare.json',K1/'targets.csv',
           ROOT/'.local/scratch/slot-b/h216-oai-pair-core-v1/CPU-MECHANICS.json',
           ROOT/'.local/scratch/slot-b/h216-oai-pair-core-v1/PARAMETER-MANIFESTS.json']
    binding=dict(version='1.0.0',sources={str(p):sha(p) for p in paths},recipe=core.RECIPE,
                 task_order=cohort['tasks'],rsna_target_order=LABELS,base_sha256=core.BASE_SHA,
                 oai_logical_roster=oai_ids,rsna_train=rsna_ids,holdout=holdout,
                 schedules=schedule,selected_oai=selected,pilot_reusable_ids=sorted(required&set(pilot)),
                 excluded_participants=cohort['excluded_participants'],qualified_input_cache=False,
                 proposed_cache='G:/temp_data/rsna-h216-oai-v4-v1',cuda=False,checkpoint_reads=0)
    binding['contract_id']=digest(binding); OUT.mkdir()
    np.save(OUT/'native-targets.private.npy',native); np.save(OUT/'rsna-targets.private.npy',rsna)
    write(OUT/'REGISTRATION.private.json',binding)
    selected_indices=[oai_ids.index(u) for u in schedule['oai_required_ids']]
    result=dict(status='PRODUCTION_INPUTS_BOUND_NOT_MATERIALIZED',version='1.0.0',contract_id=binding['contract_id'],
                oai_eligible=1703,oai_required_studies=len(required),oai_draws=800,oai_adaptation_steps=400,
                rsna_train=3190,rsna_holdout=977,rsna_control_adaptation_draws=800,rsna_recovery_draws=1600,
                selected_series=sum(len(r['series']) for r in selected),
                selected_frames=sum(s['frame_count'] for r in selected for s in r['series']),
                selected_participants=len({participant(u) for u in required}),
                selected_observed_cells=int((native[selected_indices]>=0).sum()),
                reusable_pilot_studies=len(binding['pilot_reusable_ids']),new_cohort_census=False,
                models_loaded=0,cuda=False,pixels_read=0,training_ready=False,
                output_hashes={n:sha(OUT/n) for n in ('REGISTRATION.private.json','native-targets.private.npy','rsna-targets.private.npy')})
    write(OUT/'RESULT.json',result)
    print(__import__('json').dumps({k:v for k,v in result.items() if k!='output_hashes'}))


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:prepare()
    except Exception as exc:
        code=str(exc) if type(exc) is ValueError and str(exc).replace('_','').isalnum() else 'PRIVATE_DETAILS_SUPPRESSED'
        print(__import__('json').dumps(dict(status='FAILED',error_type=type(exc).__name__,code=code)))
        raise SystemExit(1)
