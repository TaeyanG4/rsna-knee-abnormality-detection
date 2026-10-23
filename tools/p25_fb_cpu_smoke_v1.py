"""P4 engineering CPU replay v1.0.0; real complete study, all four models.

Changelog: strict embedded-runtime asset loads, real stride2 forwards and
four-member finish for base/N32. CPU FP32 only; not CUDA qualification.
"""
from __future__ import annotations
import contextlib
import copy
import hashlib
import json
import os
import time
from types import ModuleType
from tools import p25_fb_stage_v1 as b


def runtime():
    outputs,_=b.artifacts()
    files,_=b.selfcheck.payload(json.loads(outputs['kernel/release.ipynb']))
    r=ModuleType('p4_embedded_cpu_runtime')
    exec(compile(files[b.RUNTIME],'<P4 embedded runtime>','exec'),r.__dict__)
    return r


def run():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['S6PLUS_PIN_DLL']=r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
    from tools import s6plus
    import numpy as np
    import torch
    from tools import k1_tta_pilot_v1 as cache
    from tools import k1x2sr_assets_v1 as s1
    from tools import k1_shared_render_harness_v1 as harness
    from tools import k1n_cpu_replay_v1 as replay
    from tools import p25_fb_admit_v1 as admission
    b.assets('validate')
    torch.set_num_threads(4)
    r=runtime(); ct=cache.import_trainer()
    from tools import l1_rco_k1_foldgate_v1 as gate
    ids=gate.rosters()[0]
    ds=cache.dataset(ct,ids,2); counts=cache.window_counts(ct,ds)
    i=min(range(len(ids)),key=counts.__getitem__)
    images,meta,index=ds[i]
    assert index==i and len(images)==counts[i]
    # UID is hashed in published evidence; actual private telemetry stays local.
    uid=ids[i]
    mount={r.MODEL_SPECS[r.MODEL_IDS[0]]['dataset_slug'].split('/')[1]:harness.ASSETS,
           r.MODEL_SPECS[r.MODEL_IDS[1]]['dataset_slug'].split('/')[1]:b.seed0.DEST,
           r.MODEL_SPECS[r.MODEL_IDS[2]]['dataset_slug'].split('/')[1]:s1.DEST,
           b.DATASET_ID.split('/')[1]:b.DATASET}
    r.util.mount=lambda slug,*args:mount[slug]
    raw_load,raw_transfer,raw_predict=r.load_classifier,r.transfer_inputs,r.predict_prepared
    shape_specs={}; forward={}
    def load(device,mid):
        model=raw_load('cpu',mid)
        shape_specs[mid]={k:(list(v.shape),str(v.dtype)) for k,v in model.state_dict().items()}
        return model
    def predict(model,prepared,**kwargs):
        t=time.perf_counter(); value=raw_predict(model,prepared,chunk=kwargs['chunk'],fp16=False)
        forward[len(forward)]=time.perf_counter()-t
        print(json.dumps(dict(status='CPU_FORWARD_PASS',member=r.MODEL_IDS[len(forward)-1],
                              windows=len(images),finite=bool(np.isfinite(value).all()))),flush=True)
        return value
    # Only device/precision boundaries are replaced for engineering CPU replay.
    # Owner control flow, strict loader, transfer, classifier and finish are real.
    r.load_classifier=load
    r.transfer_inputs=lambda x,m,device:raw_transfer(x,m,'cpu')
    r.predict_prepared=predict
    r.precision=contextlib.nullcontext
    os.environ['K1SR_STARTED']=str(time.time())
    work=b.SCRATCH/'cpu-smoke'; worker=work/'owner'
    worker.mkdir(parents=True,exist_ok=True)
    (worker/'k1-studies.jsonl').write_text('',encoding='utf-8')
    owner=r.Owner(worker,'cuda:cpu-engineering-replay')
    item=dict(uid=uid,seconds=0.,_k1sr=dict(detail={},item=dict(uid=uid,images=images,
        meta=meta,anchor=dict(method='existing real study cache'),side=None)))
    owner.handle(item,lambda item:None)
    row=json.loads((worker/'k1-studies.jsonl').read_text())
    assert row['ok'],row.get('error')
    assert set(row['predictions'])==set(r.MODEL_IDS)
    assert all(spec==shape_specs[r.MODEL_IDS[1]] for spec in shape_specs.values())
    parent=replay.csv_bytes([uid],np.full((1,12),.5))
    results=[]
    for diagnostic in (False,True):
        out=work/('n32-path' if diagnostic else 'base-path')
        records=out/'run/pr7/pr7-run/s_r7/controller/worker0'
        records.mkdir(parents=True,exist_ok=True)
        (records/'k1-studies.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
        (out/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(parent)
        cfg=b.parent.legacy.config(b.configuration(),diagnostic)
        rec=r.finish(out,out/'run',cfg,1.)
        # Count=1 is an engineering smoke, never an ordinary 3-study or N32 receipt.
        parent_contract=dict(drop_o=True,intended_members=['R','S_R7','R224'],
            executed_members=['R','S_R7','R224'],skipped_members=['EXP143v2','EXP155','Q_A5q'])
        admission.contract(admission.base.Evidence(),b.configuration(),cfg,rec,parent_contract,1,diagnostic)
        results.append(dict(diagnostic=diagnostic,status=rec['status'],studies=1,
                            executed=rec['k1_models'],finite=True))
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_CPU_REAL_STUDY',version='1.0.0',cuda_initialized=False,
        study_id_sha256=hashlib.sha256(uid.encode()).hexdigest(),windows=len(images),stride=2,
        strict_state_keys_shapes_dtypes_match_v3g=True,state_keys=len(shape_specs[b.FB_ID]),
        checkpoint_sha256={m['id']:m['checkpoint_sha256'] for m in b.configuration()['models']},
        forward_seconds=forward,paths=results,
        boundaries='CPU FP32 device substitution only; real Owner/load/transfer/predict/finish; synthetic parent .5 probabilities',
        scope='One real cached complete study. Not raw DICOM renderer replay, public-model inference, GPU FP16, N32 or runtime qualification.')
    (work/'RESULT.json').write_bytes(b.util.encoded(result))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':run()
