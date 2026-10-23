"""CPU proof of explicit full-length RandomSampler equivalence to the pinned control.

This compares algorithms, not an unavailable historical recorded execution trace.
Only one lightweight loader worker per loader is used; model/CUDA work is absent.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training/src'))
from raptor_data import StudyWindows
from raptor_common import load_label_csv

def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1)
    subset=json.loads((a.out/'subset_manifest.json').read_text())
    ids=subset['full_train'];subids=subset['subset_train'];result={'schema':'hc05-source-backed-parity-v1','seeds':{},'status':'PENDING',
        'scope':'Prospective pinned-code full-branch parity, not replay against historical stored batch/tensor traces (none retained).',
        'loader_workers_tested':1,'historical_loader_workers':12,
        'worker_count_reason':'Main-process sampler/base-seed generation is worker-count independent for positive workers with persistent=True; per-UID input RNG is seed/epoch/UID hashed, independent of worker RNG.'}
    labels,_=load_label_csv(ROOT/'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv')
    for seed in [42,43]:
        ga=torch.Generator().manual_seed(seed);gb=torch.Generator().manual_seed(seed)
        base=torch.utils.data.DataLoader(list(range(len(ids))),batch_size=8,shuffle=True,drop_last=True,
            num_workers=1,persistent_workers=True,generator=ga)
        explicit=torch.utils.data.RandomSampler(list(range(len(ids))),replacement=False,num_samples=len(ids),generator=gb)
        derived=torch.utils.data.DataLoader(list(range(len(ids))),batch_size=8,sampler=explicit,shuffle=False,drop_last=True,
            num_workers=1,persistent_workers=True,generator=gb)
        epochs=[];inputs=[]
        da=StudyWindows(ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02',ids,labels,224,12,True,'imagenet',seed,window_policy='within-slot-valid')
        db=StudyWindows(ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02',ids,labels,224,12,True,'imagenet',seed,window_policy='within-slot-valid')
        for ep in range(8):
            ia=torch.cat(list(base)).tolist();ib=torch.cat(list(derived)).tolist()
            assert ia==ib and len(ia)==3448 and len(set(ia))==3448
            assert torch.equal(ga.get_state(),gb.get_state())
            epochs.append({'epoch':ep,'batch_count':len(ia)//8,'dropped':len(ids)-len(ia),
                'uid_sequence_sha256':digest('\n'.join(ids[i] for i in ia).encode()),
                'generator_state_sha256':digest(ga.get_state().numpy().tobytes())})
            # Bounded input check, one UID per virtual epoch; entire input function is source-hash pinned.
            da.set_epoch(ep);db.set_epoch(ep);xa,ya,ua=da[ia[0]];xb,yb,ub=db[ib[0]]
            assert ua==ub and torch.equal(xa,xb)
            inputs.append({'epoch':ep,'uid':ua,'tensor_sha256':digest(xa.numpy().tobytes())})
        base._iterator._shutdown_workers();derived._iterator._shutdown_workers()
        # The actual subset stream uses the same RandomSampler, repeated permutations to 3455 positions.
        g=torch.Generator().manual_seed(seed)
        sampler=torch.utils.data.RandomSampler(list(range(len(subids))),replacement=False,num_samples=3455,generator=g)
        loader=torch.utils.data.DataLoader(list(range(len(subids))),batch_size=8,sampler=sampler,shuffle=False,drop_last=True,
            num_workers=1,persistent_workers=True,generator=g)
        subset_epochs=[]
        for ep in range(8):
            idx=torch.cat(list(loader)).tolist();assert len(idx)==3448 and len(set(idx))==2764
            counts=torch.bincount(torch.tensor(idx),minlength=2764);assert counts.min()==1 and counts.max()==2
            subset_epochs.append({'epoch':ep,'positions':len(idx),'unique':len(set(idx)),
                'uid_sequence_sha256':digest('\n'.join(subids[i] for i in idx).encode()),'repeated_studies':int((counts==2).sum())})
        loader._iterator._shutdown_workers()
        result['seeds'][str(seed)]={'full_epochs':epochs,'bounded_inputs':inputs,'subset_epochs':subset_epochs}
    assert not torch.cuda.is_initialized()
    result['status']='PASS';result['cuda_initialized']=False
    result['source_sha256']={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in [ROOT/'training/runners/exp094_b8_variant.py',ROOT/'training/src/raptor_data.py',ROOT/'training/src/raptor_common.py',Path(__file__)]}
    result['torch_version']=torch.__version__
    with (a.out/'parity.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'seeds':list(result['seeds']),'epochs_each':8,'input_tensors':16,'cuda_initialized':False}))
if __name__=='__main__':main()
