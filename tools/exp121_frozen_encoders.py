"""EXP121 frozen medical visual encoders and fixed-final attention MIL probes.

Offline only. `queue-json --after TAG` prints jobs; it never starts jobs.
`extract --encoder NAME --device cuda` requires an allocated GPU lane.
`head --encoder NAME --device cpu` trains five EXP099 heads (20 fixed epochs).
`compare --encoder NAME --coat CSV --rad CSV --consensus CSV --out NEW_DIR`
reports diagnostic pooled OOF and conditional component bootstrap, not private evidence.
Outputs are immutable: an existing destination is refused.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/src'))
OUT = ROOT / 'artifacts/experiments/EXP121_FROZEN_ENCODERS'
PROTOCOL = ROOT / 'artifacts/experiments/EXP113_RADR50_LOCAL_PROTOCOL_V1/manifest.json'
ENCODERS = {
    'biomedclip': dict(folder='microsoft__biomedclip-pubmedbert_256-vit_base_patch16_224', size=224,
                      mean=[.48145466,.4578275,.40821073], std=[.26862954,.26130258,.27577711],
                      dim=768, minutes=[12,30]),
    'medsiglip': dict(folder='google__medsiglip-448', size=448, mean=[.5]*3, std=[.5]*3,
                     dim=1152, minutes=[75,180]),
    'mri-core': dict(folder='mazurowski-lab__mri-core', size=512,
                     mean=[0.]*3, std=[1.]*3, dim=768, minutes=[45,120]),
}

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def device_context(device):
    # No availability query, seed_all(), environment(), or CUDA API on the CPU path.
    import torch
    if device not in ('cpu', 'cuda'): raise ValueError('device must be cpu or cuda')
    return torch.device(device)

def normalize(x, name):
    import torch
    cfg = ENCODERS[name]
    mean = torch.tensor(cfg['mean'], device=x.device).view(1,3,1,1)
    std = torch.tensor(cfg['std'], device=x.device).view(1,3,1,1)
    return (x-mean)/std

def unwrap_checkpoint(path):
    import torch
    s = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('state_dict','model','teacher'):
        if key in s and isinstance(s[key],dict): s=s[key]
    return {k.removeprefix('module.'):v for k,v in s.items()}

def map_mri_teacher(state,target):
    """Explicit MRI-CORE DINO teacher -> SAM trunk mapping, with full exclusions.

    SAM neck is absent from teacher weights and deliberately unused. DINO's class
    token, heads and layer-scale parameters are not silently treated as SAM weights.
    """
    import math
    import re
    import torch.nn.functional as F
    mapped={}; source_map={}; excluded={}
    for original,value in state.items():
        key=original.removeprefix('backbone.').removeprefix('image_encoder.')
        key=re.sub(r'^blocks\.\d+\.(\d+)\.',r'blocks.\1.',key)
        key=key.replace('.fc1.','.lin1.').replace('.fc2.','.lin2.')
        if key=='pos_embed' and value.ndim==3:
            value=value[:,1:,:]; side=math.isqrt(value.shape[1])
            if side*side!=value.shape[1]: raise ValueError('MRI position grid is not square')
            value=F.interpolate(value.reshape(1,side,side,-1).permute(0,3,1,2),
                                size=target[key].shape[1:3],mode='bilinear',align_corners=False).permute(0,2,3,1)
        if key not in target:
            excluded[original]='not present in SAM trunk (DINO class token/final norm/layer scale/head)'
            continue
        if value.shape!=target[key].shape:
            raise ValueError(f'MRI shape mismatch {original} -> {key}: {tuple(value.shape)} != {tuple(target[key].shape)}')
        if key in mapped: raise ValueError('duplicate mapped MRI key: '+key)
        mapped[key]=value; source_map[original]=key
    missing=[k for k in target if k not in mapped]
    unsupported=[k for k in missing if not k.startswith('neck.')]
    if unsupported: raise ValueError('unmapped MRI trunk keys: '+', '.join(unsupported))
    return mapped,dict(mapped=source_map,unmapped=excluded,initialized_unused=missing)

def build_encoder(name, root=ROOT):
    """Construct on CPU and strictly load only the visual trunk, without a text model."""
    import torch
    import torch.nn.functional as F
    folder = Path(root)/'storage/weights'/ENCODERS[name]['folder']
    provenance=load(folder/'provenance.json')
    # Fail before loading if locally staged weights no longer match their receipt.
    for filename,receipt in provenance.get('files',{}).items():
        expected=receipt.get('sha256') if isinstance(receipt,dict) else receipt
        if expected and digest(folder/filename)!=expected:
            raise ValueError('weight provenance mismatch: '+filename)
    mapping = {}
    if name == 'biomedclip':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        state = unwrap_checkpoint(folder/'open_clip_pytorch_model.bin')
        prefix='visual.trunk.'
        visual={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        model.load_state_dict(visual, strict=True)
        mapping=dict(loaded=len(visual), excluded_nontrunk=[k for k in state if not k.startswith(prefix)],
                     representation='768-dimensional visual trunk pooled token; CLIP projection excluded')
    elif name == 'medsiglip':
        from transformers import SiglipVisionModel
        model,info=SiglipVisionModel.from_pretrained(str(folder),local_files_only=True,output_loading_info=True)
        missing=info.get('missing_keys',[])
        unexpected=[k for k in info.get('unexpected_keys',[]) if k.startswith('vision_model.')]
        mismatched=info.get('mismatched_keys',[])
        if missing or unexpected or mismatched:
            raise ValueError(f'MedSigLIP vision load incomplete: missing={missing[:10]}, unexpected={unexpected[:10]}, mismatched={mismatched[:10]}')
        mapping=dict(loaded_vision_checkpoint_keys=448, missing_keys=missing,
                     unexpected_vision_keys=unexpected, mismatched_keys=mismatched,
                     representation='1152-dimensional vision pooler_output')
    else:
        from functools import partial
        from types import SimpleNamespace
        sys.path.insert(0,str(Path(root)/'storage/code/mazurowski-lab__mri_foundation'))
        from models.sam.modeling.image_encoder import ImageEncoderViT
        args=SimpleNamespace(if_encoder_adapter=False,encoder_adapter_depths=[])
        state=unwrap_checkpoint(folder/'MRI_CORE_vitb.pth')
        pref='image_encoder.'
        visual={k[len(pref):]:v for k,v in state.items() if k.startswith(pref)}
        if not visual: visual=state
        # MRI-CORE build_sam disables relative position. If a staged checkpoint does
        # contain relative position weights, preserve their original shapes and let
        # SAM get_rel_pos interpolate to the current token grid at forward time.
        relpos=any('rel_pos_' in k for k in visual)
        model=ImageEncoderViT(args,img_size=1024,patch_size=16,embed_dim=768,depth=12,num_heads=12,
                              norm_layer=partial(torch.nn.LayerNorm,eps=1e-6),out_chans=256,
                              use_rel_pos=relpos,window_size=14,global_attn_indexes=(2,5,8,11))
        # Absolute grid is configured at 512 before explicit teacher mapping.
        model.pos_embed=torch.nn.Parameter(torch.zeros(1,32,32,768),requires_grad=False)
        mapped,report=map_mri_teacher(visual,model.state_dict())
        full=model.state_dict(); full.update(mapped); model.load_state_dict(full,strict=True)
        model.img_size=512
        mapping=dict(loaded=len(mapped),key_report=report,
                     absolute_position='drop teacher CLS token, square grid -> [1,32,32,768] bilinear',
                     relative_position=('original tensors retained; SAM get_rel_pos interpolates at runtime' if relpos else 'disabled, matching MRI-CORE build_sam'),
                     representation='768-dimensional pre-neck token mean; randomly initialized SAM neck is never used',
                     caveat='Teacher DINO -> SAM-compatible trunk probe; CLS, layer scales and final norm excluded and reported. Architecture and reduced resolution unvalidated.')
    model.eval().requires_grad_(False)
    return model,mapping

def forward_encoder(model,x,name):
    if name=='medsiglip': return model(pixel_values=x).pooler_output
    if name=='mri-core':
        z=model.patch_embed(x)+model.pos_embed
        for block in model.blocks: z=block(z)
        return z.mean((1,2))
    z=model(x)
    return z

def extract(name,device='cuda',batch=None,limit=None,out=None):
    import numpy as np
    import torch
    from raptor_common import Corpus, select_centers
    from raptor_radimagenet import center3_windows_from
    from tools.exp099_kfold import canonical_order
    p=load(PROTOCOL)
    batch=batch or {'biomedclip':32,'medsiglip':4,'mri-core':4}[name]
    if batch<1 or (limit is not None and limit<1): raise ValueError('batch and limit must be positive')
    for key in ('base_ordered_ids','folds'):
        path=ROOT/p['paths'][key]
        expected=p['source_bindings'].get(p['paths'][key])
        if expected and digest(path)!=expected: raise ValueError('source binding changed: '+key)
    corpus=Corpus(ROOT/p['paths']['corpus'])
    if digest(corpus.dir/'manifest.json')!=p['corpus_manifest_sha256']: raise ValueError('corpus changed')
    ids=canonical_order(load(ROOT/p['paths']['base_ordered_ids']))
    if len(ids)!=4343 or len(set(ids))!=4343: raise ValueError('expected 4,343 canonical G1 studies')
    if limit is not None: ids=ids[:limit]
    dest=Path(out) if out else OUT/name
    if dest.exists(): raise FileExistsError(dest)
    dev=device_context(device)
    model,mapping=build_encoder(name)
    model=model.to(dev)
    dest.mkdir(parents=True)
    np.save(dest/'uids.npy',np.asarray(ids,dtype=str),allow_pickle=False)
    centers=np.asarray([select_centers(corpus.mask(u),corpus.depth,36,corpus.spec,'within-slot-valid') for u in ids],dtype=np.int16)
    np.save(dest/'centers.npy',centers,allow_pickle=False)
    emb=np.lib.format.open_memmap(dest/'embeddings.npy',mode='w+',dtype=np.float16,shape=(len(ids),36,ENCODERS[name]['dim']))
    start=time.perf_counter()
    with torch.inference_mode():
        for i,u in enumerate(ids):
            # Exact EXP113 deterministic centres and repeated centre slice. Bicubic image
            # resize follows FM processor interpolation, instead of RN bilinear resize.
            x=torch.from_numpy(center3_windows_from(corpus.volume(u),centers[i],336))
            x=torch.nn.functional.interpolate(x,size=(ENCODERS[name]['size'],)*2,mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            for j in range(0,36,batch):
                xb=normalize(x[j:j+batch].to(dev),name)
                ctx=torch.autocast('cuda',dtype=torch.float16) if device=='cuda' else contextlib.nullcontext()
                with ctx: z=forward_encoder(model,xb,name)
                emb[i,j:j+len(z)]=z.float().cpu().numpy().astype(np.float16)
    emb.flush()
    folder=ROOT/'storage/weights'/ENCODERS[name]['folder']
    weight_hashes={f.name:digest(f) for f in folder.iterdir() if f.is_file() and f.suffix in ('.bin','.pth','.safetensors','.json')}
    dump(dest/'manifest.json',dict(schema='exp121-embeddings-v1',encoder=name,studies=len(ids),complete_g1=limit is None,
        k=36,window_policy='within-slot-valid',input_policy='center3',config=ENCODERS[name],mapping=mapping,
        device=device,batch=batch,seconds=time.perf_counter()-start,weights=weight_hashes,
        protocol_sha256=digest(PROTOCOL),corpus_manifest_sha256=digest(corpus.dir/'manifest.json'),
        artifacts={f.name:digest(f) for f in dest.iterdir() if f.is_file()},
        caveat='Runtime estimates are unmeasured; MRI-CORE reduced resolution must pass a CPU shape test before queueing.'))

def make_head(dim):
    import torch
    from torch import nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__(); self.norm=nn.LayerNorm(dim)
            self.att=nn.Sequential(nn.Linear(dim,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,12))
            self.weight=nn.Parameter(torch.empty(12,dim)); self.bias=nn.Parameter(torch.zeros(12))
            nn.init.trunc_normal_(self.weight,std=.02)
        def forward(self,x):
            h=self.norm(x); a=self.att(h).softmax(1)
            return (torch.einsum('bkn,bkf->bnf',a,h)*self.weight).sum(-1)+self.bias
    return Head()

def train_heads(name,device='cpu',epochs=20,batch=32,out=None):
    import numpy as np
    import pandas as pd
    import torch
    from tools.exp099_kfold import T,UID
    source=OUT/name; m=load(source/'manifest.json')
    if not m['complete_g1'] or m['studies']!=4343: raise ValueError('head requires full G1 extraction')
    for f,h in m['artifacts'].items():
        if digest(source/f)!=h: raise ValueError('embedding binding changed: '+f)
    if digest(PROTOCOL)!=m['protocol_sha256']: raise ValueError('protocol binding changed')
    p=load(PROTOCOL); ids=np.load(source/'uids.npy',allow_pickle=False).astype(str).tolist()
    if digest(ROOT/p['paths']['labels'])!=p['label_set']['sha256']:
        raise ValueError('Dread label binding changed')
    if digest(ROOT/p['paths']['folds'])!=p['fold_protocol']['folds_csv_sha256']:
        raise ValueError('EXP099 folds binding changed')
    labels=pd.read_csv(ROOT/p['paths']['labels']).set_index(UID)[T].loc[ids].to_numpy(dtype=np.float32)
    folds=pd.read_csv(ROOT/p['paths']['folds']).set_index(UID).loc[ids,'fold'].to_numpy()
    x=np.load(source/'embeddings.npy',mmap_mode='r'); dev=device_context(device)
    dest=Path(out) if out else source/'heads'
    dest.mkdir(parents=True,exist_ok=False)
    pred=np.full((len(ids),12),np.nan,np.float32)
    for fold in range(5):
        # CPU generator seed avoids torch.manual_seed's CUDA lazy seed side effect.
        torch.random.default_generator.manual_seed(42); rng=np.random.default_rng(42)
        model=make_head(x.shape[-1]).to(dev)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        tr=np.flatnonzero(folds!=fold); va=np.flatnonzero(folds==fold)
        prevalence=np.clip(labels[tr].mean(0),.03,.7)
        pw=np.clip((1-prevalence)/prevalence,1,10).astype(np.float32)
        loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=dev))
        for ep in range(epochs):
            model.train()
            for group in np.array_split(rng.permutation(tr),max(1,int(np.ceil(len(tr)/batch)))):
                bx=torch.tensor(np.array(x[group]),dtype=torch.float32,device=dev)
                by=torch.tensor(labels[group],device=dev)
                opt.zero_grad(set_to_none=True); value=loss(model(bx),by); value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(va),batch):
                ii=va[start:start+batch]
                pred[ii]=model(torch.tensor(np.array(x[ii]),dtype=torch.float32,device=dev)).sigmoid().cpu().numpy()
        torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},fold=fold,pos_weight=pw,
                        epochs=epochs,seed=42),dest/f'fold-{fold}.pt')
    if not np.isfinite(pred).all(): raise ValueError('OOF incomplete')
    table=pd.DataFrame(pred,columns=T); table.insert(0,UID,ids); table.to_csv(dest/'oof.csv',index=False)
    dump(dest/'manifest.json',dict(schema='exp121-head-v1',encoder=name,seed=42,epochs=epochs,
         optimizer='AdamW lr=0.0003 weight_decay=.01; fixed-final; clip_grad_norm=1; no scheduler',
         embedding_manifest_sha256=digest(source/'manifest.json'),protocol_sha256=digest(PROTOCOL),
         oof_sha256=digest(dest/'oof.csv'),labels_sha256=digest(ROOT/p['paths']['labels']),folds_sha256=digest(ROOT/p['paths']['folds'])))

def comparison(ids,head,coat,rad,dread,consensus,components,reps=300):
    import numpy as np
    from tools.exp099_kfold import avg_rank
    from tools.g1_compare import auc
    rank=lambda x: np.stack([avg_rank(x[:,j])/len(x) for j in range(12)],axis=1)
    base=(rank(coat)+rank(rad))/2; scores={'head':head,'base':base,'plus_head':(rank(coat)+rank(rad)+rank(head))/3}
    rng=np.random.default_rng(20261003); groups=np.unique(components)
    lookup={g:np.flatnonzero(components==g) for g in groups}
    draws=[np.concatenate([lookup[g] for g in rng.choice(groups,len(groups),replace=True)]) for _ in range(reps)]
    results={}
    for label,y in [('dread',dread),('consensus',consensus)]:
        mask=np.isfinite(y); macro=lambda s,ii: np.nanmean([auc(y[ii,j][mask[ii,j]],s[ii,j][mask[ii,j]]) for j in range(12)])
        allrows=np.arange(len(ids)); obs={k:float(macro(s,allrows)) for k,s in scores.items()}
        deltas=[macro(scores['plus_head'],ii)-macro(base,ii) for ii in draws]
        finite=np.asarray(deltas)[np.isfinite(deltas)]
        results[label]=dict(auc=obs,delta=obs['plus_head']-obs['base'],ci95=np.quantile(finite,[.025,.975]).tolist(),valid_draws=len(finite))
    return dict(results=results,reps=reps,seed=20261003,unit='component_id',
                caveat='Conditional bootstrap of fixed OOF predictions; equal family ranks predeclared; no public-LB tuning.')

def compare(name,coat,rad,consensus,out,reps=300):
    import numpy as np
    import pandas as pd
    from tools.exp099_kfold import UID,T
    from tools.diagnostic_quarantine import require_decision_eligible
    paths=[OUT/name/'heads/oof.csv',Path(coat),Path(rad)]
    for path in paths: require_decision_eligible(path,'EXP121',ROOT)
    frames=[pd.read_csv(f).set_index(UID) for f in paths]; ids=frames[0].index.tolist()
    if len(ids)!=4343 or len(set(ids))!=4343: raise ValueError('full G1 OOF required')
    scores=[f.loc[ids,T].to_numpy(dtype=np.float64) for f in frames]
    if not all(np.isfinite(s).all() for s in scores): raise ValueError('invalid predictions')
    p=load(PROTOCOL)
    y=pd.read_csv(ROOT/p['paths']['labels']).set_index(UID).loc[ids,T].to_numpy(dtype=np.float32)
    c=pd.read_csv(consensus).set_index(UID).loc[ids,T].to_numpy(dtype=np.float32)
    # Consensus must explicitly encode uncertain labels as NaN; retain that pair mask.
    if not np.isin(c[np.isfinite(c)],[0,1]).all(): raise ValueError('consensus must contain only 0/1/NaN')
    meta=pd.read_csv(ROOT/'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/group-metadata.csv').set_index(UID)
    result=comparison(ids,*scores,np.rint(y),c,meta.loc[ids,'component_id'].to_numpy(),reps)
    result['inputs']={str(f):digest(f) for f in [*paths,Path(consensus),ROOT/p['paths']['labels']]}
    dest=Path(out); dest.mkdir(parents=True,exist_ok=False); dump(dest/'comparison.json',result)

def queue_entries(after=None):
    entries=[]
    for name in ENCODERS:
        tag='EXP121_EXTRACT_'+name.upper().replace('-','_')
        e=dict(tag=tag,command=[str(ROOT/'.venv-fm/Scripts/python.exe'),str(ROOT/'tools/exp121_frozen_encoders.py'),
                               'extract','--encoder',name,'--device','cuda'],continue_on_failure=True)
        if after: e['after']=after
        entries.append(e); after=tag
    return entries

def check_encoder(name,forward=False):
    """CPU-only offline strict checkpoint/shape check; never queries CUDA."""
    import torch
    start=time.perf_counter(); model,mapping=build_encoder(name)
    result=dict(encoder=name,mapping=mapping,parameters=sum(p.numel() for p in model.parameters()),
                load_seconds=time.perf_counter()-start,device='cpu')
    if forward:
        x=normalize(torch.zeros(1,3,ENCODERS[name]['size'],ENCODERS[name]['size'],device='cpu'),name)
        start=time.perf_counter()
        with torch.inference_mode(): y=forward_encoder(model,x,name)
        if tuple(y.shape)!=(1,ENCODERS[name]['dim']) or not torch.isfinite(y).all():
            raise ValueError('encoder shape or finiteness failure')
        result.update(shape=list(y.shape),cpu_forward_seconds=time.perf_counter()-start)
    return result

def main():
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    ap=argparse.ArgumentParser(__doc__); sub=ap.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('queue-json'); q.add_argument('--after'); q.add_argument('--out',default='-')
    e=sub.add_parser('extract'); e.add_argument('--encoder',choices=ENCODERS,required=True)
    e.add_argument('--device',choices=['cpu','cuda'],default='cuda'); e.add_argument('--batch',type=int)
    e.add_argument('--limit',type=int); e.add_argument('--out')
    h=sub.add_parser('head'); h.add_argument('--encoder',choices=ENCODERS,required=True)
    h.add_argument('--device',choices=['cpu','cuda'],default='cpu'); h.add_argument('--epochs',type=int,default=20)
    h.add_argument('--batch',type=int,default=32); h.add_argument('--out')
    c=sub.add_parser('compare'); c.add_argument('--encoder',choices=ENCODERS,required=True)
    for key in ('coat','rad','consensus','out'): c.add_argument('--'+key,required=True)
    c.add_argument('--reps',type=int,default=300)
    v=sub.add_parser('check-encoder'); v.add_argument('--encoder',choices=ENCODERS,required=True)
    v.add_argument('--forward',action='store_true')
    a=ap.parse_args(); kw=vars(a); cmd=kw.pop('cmd')
    if cmd=='queue-json':
        data=queue_entries(a.after)
        if a.out=='-': print(json.dumps(data,indent=2))
        else:
            if Path(a.out).exists(): raise FileExistsError(a.out)
            dump(a.out,data)
    elif cmd=='extract': kw['name']=kw.pop('encoder'); extract(**kw)
    elif cmd=='head': kw['name']=kw.pop('encoder'); train_heads(**kw)
    elif cmd=='check-encoder': print(json.dumps(check_encoder(kw['encoder'],kw['forward']),indent=2))
    else: kw['name']=kw.pop('encoder'); compare(**kw)

if __name__=='__main__': main()
