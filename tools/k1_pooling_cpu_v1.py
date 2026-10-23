"""v1.0.0 CPU preparation; full real DICOM-derived cached bags, no GPU use.

Changelog: strict-load all three heads, paired trained-output reconstruction,
singleton proof and aggregate-only real-ruler evaluator self-control smoke.
"""
import os
import time
from tools import k1_pooling_v1 as p


def prepare():
    p.validate()
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    ct=p.base.import_trainer()
    import torch
    import numpy as np
    from tools import k1sr_runtime_v11 as runtime
    from tools import l1_rco_k2_worker_v102 as offline
    from tools import exp156_k_sweep as rulers
    torch.set_num_threads(4)
    ids=p.cohort(); ds=p.base.dataset(ct,ids,2); counts=p.base.window_counts(ct,ds)
    p.need(len(counts)==977 and min(counts)>0, 'nonempty exact977 bags')
    selected=sorted(range(977),key=lambda i:(counts[i],i))[:3]
    observations=[]; started=time.monotonic()
    with offline.offline(),runtime.precision():
        for name in p.ORDER:
            net=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False)
            net.load_state_dict(torch.load(p.ROOT/p.CHECKPOINTS[name][0],map_location='cpu',weights_only=True),strict=True)
            net.eval().requires_grad_(False)
            for i in selected:
                img,meta,index=ds[i]
                p.need(index==i and len(img)==counts[i] and img.dtype==torch.uint8, 'real complete cache bag identity')
                t=time.monotonic()
                pooled,win,proof=p.extract(net,runtime.transfer_inputs(img,meta,'cpu'),prove_singletons=True)
                changed=p.candidate(pooled,win)
                p.need(np.isfinite(changed).all(), 'finite candidate')
                observations.append(dict(member=p.MEMBERS[name],study_index=i,full_bag=True,windows=len(win),
                    seconds=time.monotonic()-t,strict_load=True,**proof))
                # Progress contains no IDs, predictions, or labels.
                print('CPU complete:',p.MEMBERS[name], 'study index',i,'windows',len(win),flush=True)
            del net
    # Exercise complete dump validation, trio rank/sigmoid, real four-ruler
    # availability and evaluator. Deterministic synthetic logits are NOT results.
    from tools.k1_pooling_v1 import load_dump,sha,trio,metrics,go_rule
    fixture=p.OUT/'cpu-fixture'; fixture.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261021)
    pooled=rng.normal(size=(977,12)).astype(np.float32)
    window=pooled.copy()
    path=fixture/'synthetic.npz'
    np.savez(path,ids=np.asarray(ids),offsets=np.arange(978,dtype=np.int64),pooled=pooled,windows=window)
    baseline,changed=load_dump(path,sha(path),ids,[1]*977)
    p.need(np.array_equal(baseline,changed), 'constant singleton self-control')
    refs=rulers.references(ids,root=p.ROOT)
    m=metrics(trio([changed]*3),trio([baseline]*3),refs)
    p.need(not go_rule(m) and all(v['delta']==0 for v in m['rulers'].values()), 'self-control must not GO')
    p.need(not torch.cuda.is_initialized(), 'CPU only')
    result=dict(status='PASS',version=p.VERSION,preregistration_sha256=sha(p.OUT/'PREREGISTRATION.json'),
        cuda_initialized=False,device='cpu',data='frozen real DICOM-derived cache, original complete stride2 bags',
        original_dicom_decode='not repeated; unchanged existing cache path',
        studies=3,members=3,full_bags=True,counts=counts,total_windows=sum(counts),max_windows=max(counts),
        observations=observations,seconds=time.monotonic()-started,
        max_pooled_abs=max(x['pooled_max_abs'] for x in observations),
        max_singleton_abs=max(x['singleton_max_abs'] for x in observations),
        evaluator_smoke=dict(kind='synthetic identity predictions + real structured rulers; not a candidate result',
            n=977,rulers=4,targets=48,all_macro_deltas_zero=True,go=False),
        source_pins=p.sources())
    p.write(p.OUT/'cpu-smoke.json',result)
    return {k:v for k,v in result.items() if k not in ('counts','source_pins','observations')}
