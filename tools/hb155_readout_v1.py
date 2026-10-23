"""H-B-155 v1.2.0: authenticated paired predictions and preregistered GO.

Changelog v1.2.0: require exclusion and side-policy identity in fit receipts.
Changelog: paired fold2 EXPOSED estimand and explicit exposure in readout.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from tools import multi_lens_readout_l4_v2 as lens
from tools import hb155_build_v1 as b

SEEDS=('42','43')

def gate(seed_points,mean_points,engineering=False):
    seeds,mean=np.asarray(seed_points,dtype=float),np.asarray(mean_points,dtype=float)
    if seeds.shape!=(2,4,12) or mean.shape!=(4,12) or not np.isfinite(seeds).all() or not np.isfinite(mean).all():
        return dict(GO=False,status='INVALID_EVIDENCE',rule=b.GATE)
    macro=mean.mean(axis=1); weak=mean[:,lens._base.WEAK].mean(axis=1)
    positive=bool((seeds.mean(axis=2)>0).all())
    floor=bool((seeds>=-.003).all() and (mean>=-.003).all())
    weak_pass=bool((weak>=0).all() and (seeds[:,:,lens._base.WEAK].mean(axis=2)>=0).all())
    passed=bool((macro>=.002).all() and positive and floor and weak_pass and not engineering)
    return lens._base.clean(dict(GO=passed,status='ENGINEERING_ONLY' if engineering else 'GO' if passed else 'NO_GO',
        macro=macro,weak4=weak,both_seeds_positive=positive,all_cell_floor_pass=floor,
        weak4_pass=weak_pass,rule=b.GATE,seed_points=seeds,seed_mean_points=mean,not_release_authorization=True))

def evaluate(ctx,arms,controls,engineering=False):
    for table in (arms,controls):
        b.need(set(table)==set(SEEDS),'paired seeds 42/43 required')
        for p in table.values():
            p=np.asarray(p)
            b.need(p.shape==(len(ctx.ids),12) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),
                   'aligned finite probabilities required')
    result=lens.evaluate(ctx,arms,controls,label_change=False)
    seeds=np.stack([lens._tensor(ctx.delta(arms[s],controls[s]))[0] for s in SEEDS])
    mean=lens._tensor(ctx.delta(np.mean([arms[s] for s in SEEDS],axis=0),
                               np.mean([controls[s] for s in SEEDS],axis=0)))[0]
    result.update(exposure='EXPOSED',clean_oof=False,estimand=b.COORDINATOR_DECISION['estimand'])
    result['registered_go']=gate(seeds,mean,engineering)
    result['notes']=['AUC of seed-mean probabilities; four correlated report-derived rulers.',
        'Fresh paired 4090 controls only; L2 and R7-prior exposure, clean_oof=false.',
        'Fold2 selection exposure shared by both arms; registered quantity is paired delta.',
        'Fixed final epoch20, K36 flip; Gold58 descriptive only; not release authorization.']
    return lens._base.clean(result)

def validate_records(c,summary,receipt,*,config_hash,csv_hash,checkpoint_hash,prepare_hash,runtime_hash):
    i=receipt.get('identity',{}); b.need(i==summary.get('identity'),'summary/receipt identity differs')
    expected=dict(experiment='H-B-155 compartment crop paired pilot',fold=c['fold'],arm=c['arm'],seed=c['seed'],
        resolution=224,config_sha256=config_hash,registration_sha256=config_hash,
        train_sha256=b.uid_hash(c['train']),val_sha256=b.uid_hash(c['val']),workers=2,
        epochs=20,smoke=False,engineering_only=False,gpu_name=b.GPU,weights=c['weights'],model=c['recipe']['model'],
        batch=8,microbatch=2,accumulation=4,corpus_contract=c['corpus_contract'],
        exclusions_sha256=c['exclusions_sha256'],side_unknown_policy=c['side_unknown_policy'],
        prepare_sha256=prepare_hash,runtime_snapshot_sha256=runtime_hash,labels_sha256=c['inputs'][c['label_path']])
    for key,value in expected.items(): b.need(i.get(key)==value,'old/foreign fit identity: '+key)
    b.need(summary.get('passed') is True and summary.get('smoke') is False and summary.get('device')=='cuda'
        and summary.get('steps')==c['optimizer_steps'] and summary.get('counters',{}).get('updates')==c['optimizer_steps']
        and 0<=summary['counters']['skips']<=c['optimizer_steps']//100,'partial fit forbidden')
    b.need(receipt.get('k')==36 and receipt.get('tta') is True and receipt.get('engineering_only') is False,'scientific K36 flip required')
    b.need(receipt.get('rows')==len(c['val']) and receipt.get('uid_sha256')==b.uid_hash(c['val']),'prediction roster differs')
    b.need(receipt.get('csv_sha256')==csv_hash and receipt.get('checkpoint_sha256')==summary.get('checkpoint_sha256')==checkpoint_hash,
        'prediction/checkpoint binding differs')
    r=i.get('runtime',{})
    b.need(r.get('policy')=='strict' and r.get('precision_fallback_allowed') is False and
        str(r.get('device','')).startswith('cuda'),'runtime differs')
    return i

def run(review):
    arms,controls,bindings={},{},{}; runtime=None
    for arm,table in (('control',controls),('crop',arms)):
        for seed in (42,43):
            cp=b.config_path(arm,seed); c=b.require_fit(cp,review)
            directory=b.WORK/'fits'/b.member(arm,seed)
            b.need(b.read(directory/'heartbeat.json').get('status')=='complete','fit incomplete; do not open outputs')
            summary=b.read(directory/'summary.json'); receipt=b.read(directory/'val.receipt.json')
            b.require_runtime(cp,b.read(directory/'runtime.json'))
            i=validate_records(c,summary,receipt,config_hash=b.sha(cp),csv_hash=b.sha(directory/'val.csv'),
                checkpoint_hash=b.sha(directory/'final.pt'),prepare_hash=b.sha(b.WORK/'prepared.json'),runtime_hash=b.sha(directory/'runtime.json'))
            if runtime is None: runtime=i['runtime']
            b.need(runtime==i['runtime'],'paired runtime differs')
            import pandas as pd
            f=pd.read_csv(directory/'val.csv')
            b.need(f['StudyInstanceUID'].astype(str).tolist()==c['val'],'CSV order differs')
            table[str(seed)]=f[c['recipe']['targets']].to_numpy(dtype=float); bindings[b.member(arm,seed)]=receipt
    ctx=lens._base.make_context(c['val'],b.ROOT)
    result=evaluate(ctx,arms,controls); result.update(prediction_bindings=bindings,ruler_input_hashes=ctx.bindings)
    b.immutable(b.WORK/'readout.json',result)
    return dict(status=result['registered_go']['status'],readout=str(b.WORK/'readout.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--review',type=Path,required=True)
    print(json.dumps(run(p.parse_args().review)))
