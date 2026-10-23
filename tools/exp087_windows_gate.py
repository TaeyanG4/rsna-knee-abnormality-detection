"""Explicit180s production-Windows spawn input gate; no models or real labels.

Only run after root source review. Full membership guards stay active. Eight
chosen T indices exercise enabled/disabled/forced-original datasets over epoch0
then1 with12 persistent workers. Parent comparison uses independent objects.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import exp087_training as guard


def execute(protocol_path,output):
    if os.name!='nt':raise RuntimeError('actual Windows spawn is required')
    p=guard.read(protocol_path)
    if os.environ.get('EXP087_AUTHORIZED_PROTOCOL_SHA256')!=guard.sha256(protocol_path):raise ValueError('authorized exact supervisor required')
    if output.resolve()!=(ROOT/p['windows_gate']['output']).resolve() or os.environ.get('EXP087_PHASE')!='windows' or os.environ.get('EXP087_PHASE_AUTHORIZATION_SHA256')!=guard.sha256(output/'phase-authorization.json'):
        raise ValueError('Windows phase/output authorization mismatch')
    guard.validate_bindings(p,cpu=True);guard.install_gold_blocker(ROOT)
    blocked=(ROOT/'storage/labels').resolve()
    def no_labels(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            try:path=Path(os.fsdecode(args[0])).resolve()
            except (OSError,ValueError):return
            if path.is_relative_to(blocked):raise PermissionError('CPU input gate may not open label files')
    sys.addaudithook(no_labels)
    sys.path.insert(0,str(ROOT/'training/src'))
    import torch
    from torch.utils.data import DataLoader
    from raptor_data import StudyWindows,collate
    from exp087_coverage_data import CoverageStore,CoverageStudyWindows
    from tools.exp087_series_coverage import windows,array_hash
    guard.worker_init(0)
    ordered=guard.read(ROOT/p['ordered_ids']['path']);ids=ordered['train']
    labels={uid:np.zeros(12,np.float32) for uid in ids}
    def make(mode):
        base=StudyWindows(ROOT/p['paths']['corpus'],ids,labels,224,12,True,'imagenet',42,window_policy='within-slot-valid')
        store=CoverageStore(ROOT/p['coverage_manifest'],p['coverage_manifest_sha256'])
        return CoverageStudyWindows(base,store,enabled=mode!='disabled',force_original=mode=='forced')
    parent=make('enabled')
    outcomes=[];changes=[];cleanups=[];started=time.monotonic()
    for mode in ('disabled','forced','enabled'):
        dataset=make(mode)
        if dataset.baseline.corpus is not None:raise ValueError('production dataset touched before spawn')
        production=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=12,drop_last=True,collate_fn=collate,
            generator=torch.Generator().manual_seed(42),worker_init_fn=guard.worker_init,persistent_workers=True,pin_memory=False,multiprocessing_context='spawn')
        if len(production)!=431:raise ValueError('production431steps mismatch')
        del production  # Never iterated; actual bounded sampler below retains full3455 dataset.
        loader=DataLoader(dataset,batch_size=8,sampler=p['windows_gate']['indices'],num_workers=12,drop_last=True,
            collate_fn=collate,generator=torch.Generator().manual_seed(42),worker_init_fn=guard.worker_init,
            persistent_workers=True,pin_memory=False,multiprocessing_context='spawn')
        try:
            epoch_tensor_hash=[]
            for epoch in (0,1):
                dataset.set_epoch(epoch);parent.set_epoch(epoch)
                seen=[];batches=0
                # Exhaust the bounded iterator before changing the shared epoch.
                for x,y,uids in loader:
                    batches+=1;seen+=uids
                    if x.shape!=(8,12,3,224,224) or y.shape!=(8,12) or not torch.isfinite(x).all():raise ValueError('production tensor contract differs')
                    per=[]
                    for j,uid in enumerate(uids):
                        idx=ids.index(uid);parent.baseline[idx]  # Separate parent-only corpus.
                        corpus=parent.baseline.corpus;row=parent.store.entry(uid,'train')
                        raw,br=windows(corpus.volume(uid),corpus.mask(uid),corpus.spec,uid,train=True,epoch=epoch,enabled=False)
                        kwargs={}
                        if row is not None and row['reason'] is None:kwargs=dict(alternate=parent.store.volume(row),alt_mask=row['mask'],native_indices=row['native_indices'])
                        before=(random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone())
                        expected,trace=windows(corpus.volume(uid),corpus.mask(uid),corpus.spec,uid,train=True,epoch=epoch,
                            enabled=mode!='disabled',force_original=mode=='forced',**kwargs)
                        after=np.random.get_state()
                        if random.getstate()!=before[0] or not np.array_equal(after[1],before[1][1]) or not torch.equal(torch.random.get_rng_state(),before[2]):raise ValueError('parent pure sample global RNG changed')
                        if not np.array_equal(x[j].numpy(),expected) or not np.array_equal(y[j].numpy(),labels[uid]):raise ValueError('spawn output/y differs from parent contract')
                        replaced={int(k) for k in trace['replacements']};changed=[i for i in range(12) if not np.array_equal(raw[i],expected[i])]
                        if set(changed)-replaced:raise ValueError('unreplaced tensor changed')
                        if mode!='enabled' and changed:raise ValueError('no-op adapter differs from baseline')
                        changes+= [dict(mode=mode,epoch=epoch,uid=uid,changed=changed,replacement_map=trace['replacements'])]
                        per.append(dict(uid=uid,input_sha256=array_hash(expected),baseline_sha256=array_hash(raw),
                            window_rng_sha256=hashlib.sha256(str(trace['window_rng_after']).encode()).hexdigest(),intensity=trace['intensity'],centers=trace['centers']))
                    outcomes.append(dict(mode=mode,epoch=epoch,rows=per));epoch_tensor_hash.append(array_hash(x.numpy()))
                if batches!=1 or seen!=p['windows_gate']['uids']:raise ValueError('bounded delivered order differs')
            if epoch_tensor_hash[0]==epoch_tensor_hash[1]:raise ValueError('shared epoch propagation not observed')
        finally:cleanups.append(guard.stop_loader(loader))
    active=p['windows_gate']['cases']['eligible_active'];zero=p['windows_gate']['cases']['eligible_zero_epoch0'];missing=p['windows_gate']['cases']['missing']
    if not any(r['mode']=='enabled' and r['epoch']==0 and r['uid']==active and r['changed'] for r in changes):raise ValueError('eligible treatment not exercised')
    if any(r['changed'] for r in changes if r['uid']==missing or (r['uid']==zero and r['epoch']==0)):raise ValueError('missing/zero replacement no-op failed')
    guard.dump(output/'receipt.json',dict(status='PASS_WINDOWS_INPUT_INTEGRATION',schema='exp087-windows-input-gate-v1',
        protocol_sha256=guard.sha256(protocol_path),coverage_manifest_sha256=p['coverage_manifest_sha256'],
        population=3455,production_batches=431,workers=12,internal=1,batch=8,K=12,epochs=[0,1],modes=['disabled','forced','enabled'],
        production_baseline_corpusNone_at_spawn=True,multiprocessing_context='spawn',parent_comparator_is_separate=True,
        labels='in-memory synthetic zero12 only; no real labels opened',model_loads=0,GPU=0,results=outcomes,replacement_checks=changes,
        worker_cleanup=cleanups,seconds=time.monotonic()-started,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    try:execute(a.protocol,a.output)
    except BaseException as exc:
        guard.dump(a.output/'FAIL.json',dict(status='FAIL',error=f'{type(exc).__name__}: {exc}'));raise

if __name__=='__main__':main()
