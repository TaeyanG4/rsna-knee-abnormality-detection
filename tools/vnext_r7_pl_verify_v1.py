"""Offline evidence for vNext-R7-PL; predictions/images only, never labels."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools import vnext_r7_pl_v1 as p
from tools import vnext_r7_pl_worker_v1 as w


def build():
    results={}
    for arm in ('global','fill'):
        for mode in ('fivefold','all-data'):
            for seeds in ([42],[42,43]):
                name=f'{arm}-{mode}-s'+ '-'.join(map(str,seeds))
                stage=p.STAGE/name
                results[name]=dict(register=p.register(stage,arm,mode,seeds),
                    prepare=p.prepare(stage),validate=p.validate(stage),admit=p.admit(stage))
    p.dump(p.SCRATCH/'build-results.json',results)
    return dict(status='PASS',variants=len(results),all_admissions='HOLD')


def replay():
    import numpy as np
    from tools import hb127_vnextr7_verify as old
    _,ids,recorded=old.recorded()
    job=p.read(old.NATIVE/'NATIVE/worker0/job.json')
    members=[e['id'] for e in job['entries'] if e.get('family')=='S' and e['id'] in recorded]
    p.need(members,'recorded MIL members')
    spec=importlib.util.spec_from_file_location('pl_reference',ROOT/'tests/test_vnext_r7_pl_v1.py')
    checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
    results={}
    for arm in ('global','fill'):
        reg=p.registration(p.STAGE/f'{arm}-all-data-s42')
        raw={e['id']: (np.mean(np.stack([recorded[k] for k in members]),axis=0,dtype=np.float32)
             if e['kind']=='MIL' else recorded['EXP143v2' if e['kind']=='CoAt' else e['id']].copy()) for e in reg['members']}
        for failed in (None,'own','R','R7','all'):
            label=failed or 'healthy';folder=p.SCRATCH/'v1.1.1/replay'/arm/label
            shards=[];values=copy.deepcopy(raw)
            for branch in ('own','R','R7'):
                entries=[e for e in reg['members'] if e['branch']==branch]
                ident=p.identity(reg,branch,entries,'recorded-replay')
                shard=dict(path=folder/branch,ids=ids,identity=ident);shards.append(shard)
                journal=p.Journal(shard['path'],ids,ident)
                for e in entries:
                    for i,uid in enumerate(ids):
                        bad=failed in (branch,'all')
                        journal.record(uid,e['id'],values[e['id']][i],error='forced failed' if bad else None)
                        if bad: values[e['id']][i]=np.nan
                journal.finish()
            for weight in ('primary','sensitivity'):
                np.testing.assert_array_equal(p.blend(reg,values,weight),checks.reference(reg,values,weight))
                diagnostic = weight == 'sensitivity'
                receipt=p.publish(reg,ids,shards,folder/weight,weight,
                    filename='diagnostic.csv' if diagnostic else 'submission.csv', diagnostic=diagnostic)
                p.need(receipt['fallbacks']==sum(np.isnan(v).all(1).sum() for v in values.values()),'fallback accounting')
                results[f'{arm}/{label}/{weight}']=receipt
    result=dict(status='PASS',studies=len(ids),cases=len(results),independent_exact=True,
        source_pins={str(old.NATIVE/'raw.npz'):p.sha(old.NATIVE/'raw.npz'),
                     str(old.OLD/'cpu-member-raw.npz'):p.sha(old.OLD/'cpu-member-raw.npz')},
        scope='Authenticated recorded predictions; old MIL/CoAt and S6 stand in for untrained PL/R7. Arithmetic only.',
        results=results)
    p.dump(p.SCRATCH/'replay-results.json',result)
    return {k:result[k] for k in ('status','studies','cases','independent_exact','scope')}


def native_smoke():
    import numpy as np
    import torch
    from unittest.mock import patch
    from tools import hb127_vnextr7_verify as old
    from tools import hb116_vnext_release as h, hb116_native_runtime as native, exp169_release_v6 as v6
    source=p.read(old.NATIVE/'NATIVE/worker0/job.json')
    reg=p.registration(p.STAGE/'global-all-data-s42')
    entries=[]
    for template in reg['members']:
        if template['kind']=='R7': continue
        prior=next(e for e in source['entries'] if (e.get('family')=='S' if template['kind']=='MIL'
            else e['id']==('EXP143v2' if template['kind']=='CoAt' else template['id'])))
        entry=dict(prior,**template,path=prior['resolved'])
        if template['kind']=='MIL':
            receipt=p.SCRATCH/'v1.1.1/native-smoke/mil-receipt.json'
            p.dump(receipt,dict(spec_sha256=prior['spec_sha256']))
            entry['training_receipt']=dict(path=str(receipt),sha256=p.sha(receipt))
        entries.append(entry)
    ids=source['ids'];directory=p.SCRATCH/'v1.1.1/native-smoke/run'
    shards=[dict(path=str(directory/branch),ids=ids,
        identity=p.identity(reg,branch,[e for e in entries if e['branch']==branch],'real-dicom-probes')) for branch in ('own','R')]
    task=dict(source,entries=entries,branches=shards,output=str(directory),device='cpu')
    def worker(path,device):
        ns,_,_=h.anchor_setup(task,[],device,{})
        def model(*a): return torch.nn.Linear(1,1)
        def infer(model,tensor,device,**kwargs):
            return np.full(12,torch.sigmoid(tensor.double().mean()).item(),np.float32)
        def forbidden(*a,**k): raise AssertionError('CUDA forbidden')
        with patch.object(h,'q_model',model),patch.object(h,'anchor_setup',lambda *a:(ns,{e['id']:model() for e in entries if e['kind']!='MIL'},None)), \
             patch.object(v6,'infer',infer),patch.object(torch,'set_num_threads',lambda n:None),patch.object(torch.cuda,'_lazy_init',forbidden):
            native.worker(path,device)
    w.native(task,worker)
    for shard in shards:
        p.need(p.collect(shard['path'],ids,shard['identity'])[1]==0,'real DICOM native smoke fallbacks')
    p.need(not torch.cuda.is_initialized(),'CUDA initialized')
    result=dict(status='PASS',studies=len(ids),members=len(entries),fallbacks=0,cuda_initialized=False,
        scope='Real frozen DICOM/native preprocessing through new worker/task/journal adapters; deterministic tensor probes, no trained PL inference.')
    p.dump(p.SCRATCH/'native-smoke-results.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','replay','native-smoke'])
    parser.add_argument('--stage',type=Path,default=p.STAGE)
    args=parser.parse_args()
    p.STAGE=args.stage
    print(json.dumps(globals()[args.command.replace('-','_')]()))


if __name__=='__main__': main()
