"""1.1.0 target-source adapter for the unchanged EXP143v2 execution function.

Scoped substitutions replace only registration, admission and destination. The
frozen function still owns optimizer, augmentation, seeds, accumulation, fixed
SWA [14,11,12] and its no-augmentation training-only BN replay.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


def protocol(spec,target,labels,output):
    from tools import vnext_r7_pl_train_v110 as t
    from tools.exp143_v2 import RECIPE
    t.p.need(spec['recipe']==RECIPE,'CoAt frozen recipe differs')
    t.p.need(target['fold']=='all' and not target['val_ids'] and set(labels)==set(target['train_ids']),
             'CoAt training-only target roster')
    out=str(Path(output).resolve().relative_to(t.ROOT.resolve()))
    return dict(recipe=spec['recipe'], train_ids=target['train_ids'],
        train_labels=labels, pos_weight=target['weights'], run_dirs=dict(local=out,kaggle=out),
        budgets=dict(local=12,kaggle=12,reserve_seconds=1200))


@contextmanager
def target_hook(spec,target,labels,output,*,validate):
    """Only in the dedicated trainer process; restore every substitution."""
    from tools import exp143_v2 as guard
    from tools import vnext_r7_pl_train_v110 as t
    document=protocol(spec,target,labels,output)
    old=(guard.load_registration,guard.require_validation,guard.registered_run)
    def load(path,corpus=None,init_weights=None):
        validate()
        t.p.need(Path(corpus).resolve()==Path(spec['corpus']).resolve(),'CoAt corpus differs')
        t.p.need(t.p.sha(init_weights)==spec['inputs'][1]['sha256'],'CoAt initialization differs')
        return document
    def require(path):
        validate()
    def destination(p,lane='local'):
        t.p.need(p is document and lane=='local','CoAt PL is single-GPU s42')
        return Path(output)
    guard.load_registration,guard.require_validation,guard.registered_run=load,require,destination
    try:
        yield document
    finally:
        guard.load_registration,guard.require_validation,guard.registered_run=old


def fit(spec,target,labels,output,*,device,smoke,go):
    import torch
    from training.src import exp143_v2 as frozen
    from tools import vnext_r7_pl_train_v110 as t
    t.p.need(spec['member']['seed']==42,'CoAt seed42 only')
    if smoke:
        return smoke_cpu(spec,target,labels,output)
    t.p.need(device=='cuda' and go and go['status']=='ADMITTED','CoAt scientific admission')
    # train() imports its original guard; the scoped hook supplies authenticated
    # targets without opening either original labels or report-derived sources.
    path=Path(output)/'fit'
    args=SimpleNamespace(protocol=Path(spec['target_bundle']['path']),corpus=Path(spec['corpus']),
        init_weights=Path(spec['inputs'][1]['path']),device='cuda',precision='fp16',seed=42,
        grad_checkpointing=True,workers=0,resume=False)
    def validate():
        for entry in [*spec['source_pins'],*spec['inputs'],spec['target_bundle']]:
            t.p.pin(entry)
        t.p.pin(go['go'])
    with target_hook(spec,target,labels,path,validate=validate):
        frozen.train(args)
    ck=torch.load(path/'model_best.pt',map_location='cpu',weights_only=False)
    summary=t.p.read(path/'summary.json')
    t.p.need(summary['status']=='COMPLETE_TRAINING_NO_EVALUATION' and summary['epochs']==16,
             'CoAt fixed-SWA-BN incomplete')
    t.p.need(summary['model_sha256']==t.p.sha(path/'model_best.pt'),'CoAt summary/checkpoint mismatch')
    model=frozen.build_model(384)
    model.load_state_dict(ck['model'],strict=True)
    return t.export(spec,target,model,output,smoke=False,
        counters=dict(updates=summary['updates'],skips=summary['skips']),
        runtime=summary['identity'],go=go,coat_checkpoint=ck)


def smoke_cpu(spec,target,labels,output):
    """Real cache augmentation + BN inputs and optimizer/SWA/BN on a tiny model.

    This is adapter engineering evidence, not a full CoAt parity or timing test.
    It exercises the real target-source and every frozen CPU-capable operation.
    """
    import torch
    from training.src import exp143_v2 as f
    from tools import vnext_r7_pl_train_v110 as t
    from tools.exp112_full_swa import swa_average,recompute_bn
    ids=target['train_ids'][:2]
    t.p.need(len(ids)==2,'CoAt smoke needs two training studies')
    from tools import exp143_v2 as guard
    def validate():
        for entry in spec.get('inputs',[]):t.p.pin(entry)
    with target_hook(spec,target,labels,output,validate=validate) as document:
        guard.load_registration('engineering',spec['corpus'],spec['inputs'][1]['path'])
        guard.require_validation('engineering')
        t.p.need(document['train_labels'] is labels,'target hook identity')
        guard.execution_settings(document,1,'fp16',42,True)
    class TinyBackbone(torch.nn.Module):
        num_features=4
        def __init__(self):
            super().__init__()
            self.conv=torch.nn.Conv2d(3,4,1)
            self.bn=torch.nn.BatchNorm2d(4)
        def forward(self,x):
            return self.bn(self.conv(x)).mean((-1,-2))
    f._cpu_seed(42)
    model=f.build_model(384,backbone_factory=TinyBackbone)
    ds=f.StudyDataset(spec['corpus'],ids,labels,42)
    opt,sched,scaler=f.make_optimization(model,spec['recipe'],2,'none')
    lossf=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target['weights']))
    for index in range(2):
        x,y,uid=ds[index]
        # Real windows/augmentation at native384 are constructed before tiny
        # pooling reduces engineering smoke memory; never used in full fit.
        x=torch.nn.functional.adaptive_avg_pool2d(x,(16,16))[None]
        loss=lossf(model(x),y[None]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
        opt.step();sched.step();opt.zero_grad(set_to_none=True)
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(swa_average([state,state,state]),strict=True)
    bn_ds=f.StudyDataset(spec['corpus'],ids,labels,42,augment=False)
    bn_ds.set_epoch(16)
    batches=[]
    for index in range(2):
        x,y,uid=bn_ds[index]
        batches.append((torch.nn.functional.adaptive_avg_pool2d(x,(16,16))[None],y[None],[uid]))
    bn=recompute_bn(model,batches,'cpu',f.amp_context('cpu','none'))
    t.p.need(not torch.cuda.is_initialized(),'CPU initialized CUDA')
    result=dict(status='ENGINEERING_SMOKE',real_cache=True,tiny_backbone=True,updates=2,
        swa_epochs_1based=[14,11,12],bn=bn,training_only_bn=True,cuda_initialized=False)
    ck=dict(arch=t.p.COAT_ARCH,lab=t.p.TARGETS,seed=42,res=384,inference_res=320,
        contract='exp143_raptor96_384_c140_s02',k_eval=94,norm='imagenet',window_policy='legacy',
        oai=False,checkpoint_kind='fixed-swa-bn',swa_epochs_1based=[14,11,12],epoch=0,
        tiny_engineering_backbone=True)
    result['export']=t.export(spec,target,model,output,smoke=True,counters=dict(updates=2,skips=0),
        runtime=dict(device='cpu',tiny_backbone=True),coat_checkpoint=ck)
    t.p.dump(Path(output)/'coat-smoke.json',result)
    return result
