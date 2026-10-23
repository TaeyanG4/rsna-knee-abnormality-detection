"""H-B-147 fold1 confirmation readout v1.0.0, 2026-10-14.

Changelog: fresh matched fold1 controls; L4-v2 telemetry plus preregistered
seed-mean confirmation gate. Gold58/OAI remain descriptive, never rescue GO.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from tools import multi_lens_readout_l4_v2 as lens
from tools import hb147_confirm_build_v1 as b

SEEDS=('42','43')

def gate(seed_points,mean_points,engineering=False):
    seeds,mean=np.asarray(seed_points,dtype=float),np.asarray(mean_points,dtype=float)
    valid=seeds.shape==(2,4,12) and mean.shape==(4,12) and np.isfinite(seeds).all() and np.isfinite(mean).all()
    if not valid: return dict(GO=False,status='INVALID_EVIDENCE',rule=b.GATE)
    macro=mean.mean(axis=1); weak=mean[:,lens._base.WEAK].mean(axis=1)
    qualifies=(macro>=.001)&(weak>=.001)
    syn=bool((seeds[:,:,8]>=-.005).all() and (mean[:,8]>=-.005).all())
    floor=bool((seeds>=-.005).all() and (mean>=-.005).all())
    passed=bool(qualifies.sum()>=3 and syn and floor and not engineering)
    return lens._base.clean(dict(GO=passed,status='ENGINEERING_ONLY' if engineering else 'GO' if passed else 'NO_GO',
        macro=macro,weak4=weak,qualifying_rulers=[r for r,q in zip(lens.RULERS,qualifies) if q],
        synovitis_floor_pass=syn,all_cell_floor_pass=floor,rule=b.GATE,
        seed_points=seeds,seed_mean_points=mean,not_release_authorization=True))

def evaluate(ctx,arms,controls,image_reads=None,engineering=False):
    for table in (arms,controls):
        b.need(set(table)==set(SEEDS),'paired seeds 42/43 required')
        for p in table.values():
            p=np.asarray(p); b.need(p.shape==(len(ctx.ids),12) and np.isfinite(p).all() and
                                  ((p>=0)&(p<=1)).all(),'aligned finite probabilities required')
    images={k:{**v,'eligible':False} for k,v in (image_reads or {}).items()} or None
    result=lens.evaluate(ctx,arms,controls,label_change=False,image_reads=images)
    seeds=np.stack([lens._tensor(ctx.delta(arms[s],controls[s]))[0] for s in SEEDS])
    mean=lens._tensor(ctx.delta(np.mean([arms[s] for s in SEEDS],axis=0),
                               np.mean([controls[s] for s in SEEDS],axis=0)))[0]
    result['registered_go']=gate(seeds,mean,engineering)
    result['notes']=['EXP099 fold1, non-PL L2; fresh native448 224 vs 320, seeds42/43.',
        'AUC of arithmetic seed-mean predictions; correlated rulers are not independent replications.',
        'Gold58/OAI descriptive only. L1-L7 diagnostic; registered_go is the confirmation decision.',
        'Fold0 pilot remains HOLD; confirmation is not release authorization.']
    return lens._base.clean(result)

def validate_records(c,summary,receipt,*,config_hash,csv_hash,checkpoint_hash):
    i=receipt.get('identity',{}); b.need(i==summary.get('identity'),'summary/receipt identity differs')
    expected=dict(experiment='H-B-147 fold1 confirmation',fold=1,arm=c['arm'],seed=c['seed'],
       resolution=c['resolution'],config_sha256=config_hash,registration_sha256=config_hash,
       train_sha256=b.uid_hash(c['train']),val_sha256=b.uid_hash(c['val']),workers=2,
       epochs=20,smoke=False,engineering_only=False,gpu_name=b.GPU,weights=c['weights'],model=c['recipe']['model'])
    for key,value in expected.items(): b.need(i.get(key)==value,'old/foreign fit identity: '+key)
    b.need(i.get('labels_sha256')==c['inputs'][c['label_path']] and
           i.get('prepare_sha256')==b.sha(b.WORK/'prepared.json'),'label/preparation identity differs')
    b.need(summary.get('passed') is True and summary.get('smoke') is False and summary.get('device')=='cuda'
           and summary.get('steps')==c['optimizer_steps'] and
           summary.get('counters',{}).get('updates')==c['optimizer_steps'] and
           0<=summary['counters']['skips']<=c['optimizer_steps']//100,'partial fit forbidden')
    b.need(receipt.get('k')==36 and receipt.get('tta') is True and receipt.get('engineering_only') is False,
           'scientific K36 flip required')
    b.need(receipt.get('rows')==len(c['val']) and receipt.get('uid_sha256')==b.uid_hash(c['val']),
           'prediction roster differs')
    b.need(receipt.get('csv_sha256')==csv_hash and receipt.get('checkpoint_sha256')==
           summary.get('checkpoint_sha256')==checkpoint_hash,'prediction/checkpoint binding differs')
    runtime=i.get('runtime',{})
    b.need(runtime.get('policy')=='strict' and runtime.get('precision_fallback_allowed') is False and
           str(runtime.get('device','')).startswith('cuda'),'runtime differs')
    return i

def run(review):
    # Caller runs only after the entire detached fit job has completed.
    arms,controls,bindings={},{},{}; runtime=None
    for res,table in ((224,controls),(320,arms)):
        for seed in (42,43):
            cp=b.config_path(res,seed); c=b.require_fit(cp,review)
            directory=b.WORK/'fits'/b.member(res,seed)
            b.need(b.read(directory/'heartbeat.json').get('status')=='complete','fit incomplete; do not open outputs')
            summary=b.read(directory/'summary.json'); receipt=b.read(directory/'val.receipt.json')
            i=validate_records(c,summary,receipt,config_hash=b.sha(cp),csv_hash=b.sha(directory/'val.csv'),
                               checkpoint_hash=b.sha(directory/'final.pt'))
            if runtime is None: runtime=i['runtime']
            b.need(runtime==i['runtime'],'paired runtime differs')
            # Prediction CSV only; never report text. Reader validates exact UID order.
            import pandas as pd
            f=pd.read_csv(directory/'val.csv'); targets=c['recipe']['targets']
            b.need(f['StudyInstanceUID'].astype(str).tolist()==c['val'],'CSV ordering differs')
            table[str(seed)]=f[targets].to_numpy(dtype=float); bindings[b.member(res,seed)]=receipt
    ctx=lens._base.make_context(c['val'],b.ROOT)
    result=evaluate(ctx,arms,controls)
    result.update(prediction_bindings=bindings,ruler_input_hashes=ctx.bindings,
                  gold58='descriptive only; optional image_reads API, no Gold labels accessed by this build')
    b.immutable(b.WORK/'readout.json',result)
    return dict(status=result['registered_go']['status'],readout=str(b.WORK/'readout.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--review',type=Path,required=True)
    print(json.dumps(run(p.parse_args().review)))
