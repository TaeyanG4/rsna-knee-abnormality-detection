try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import timm

ROOT=workspace_root()
CACHE=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'
OUT=ROOT/'experiments'/'exp024_dinov2_features'
OUT.mkdir(parents=True,exist_ok=True)
CENTERS=[2,4,6]

class ShardCache:
    def __init__(self, root):
        mani=json.loads((root/'train_cache_shards_manifest.json').read_text())
        self.shards=[]
        for s in mani['shards']:
            self.shards.append((s['start'],s['stop'],np.load(root/s['file'],mmap_mode='r')))
    def get(self,idx):
        for a,b,x in self.shards:
            if a<=idx<b: return x[idx-a]
        raise IndexError(idx)

def main():
    dev='cuda'
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    model=timm.create_model('vit_small_patch14_dinov2',pretrained=True,num_classes=0,img_size=224).to(dev).eval()
    for p in model.parameters(): p.requires_grad=False
    cache=ShardCache(CACHE)
    masks=np.load(CACHE/'train_cache_masks.npy')
    n=masks.shape[0]
    feats=np.lib.format.open_memmap(OUT/'features.npy',mode='w+',dtype=np.float16,shape=(n,6,len(CENTERS),384))
    valid=np.zeros((n,6,len(CENTERS)),dtype=np.uint8)
    mean=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
    t0=time.time(); batch_imgs=[]; batch_keys=[]
    def flush():
        nonlocal batch_imgs,batch_keys
        if not batch_imgs: return
        x=torch.from_numpy(np.stack(batch_imgs)).to(dev,non_blocking=True).float().div_(255.0)
        x=(x-mean)/std
        with torch.no_grad(), torch.amp.autocast('cuda',dtype=torch.float16):
            for a in range(0,len(x),64):
                z=model(x[a:a+64]).float().cpu().numpy().astype(np.float16)
                for r,v in enumerate(z):
                    i,s,c=batch_keys[a+r]; feats[i,s,c]=v; valid[i,s,c]=1
        batch_imgs=[]; batch_keys=[]
    for i in range(n):
        arr=cache.get(i)  # 6,9,H,W uint8
        for s in range(6):
            for ci,c in enumerate(CENTERS):
                if masks[i,s,c-1:c+2].min()==0: continue
                rgb=np.stack([arr[s,c-1],arr[s,c],arr[s,c+1]],axis=0)
                batch_imgs.append(rgb); batch_keys.append((i,s,ci))
                if len(batch_imgs)>=256: flush()
        if (i+1)%100==0:
            flush(); dt=time.time()-t0; print(f'{i+1}/{n} {dt/60:.1f}m eta {dt/(i+1)*(n-i-1)/60:.1f}m valid={int(valid[:i+1].sum())}',flush=True)
    flush(); feats.flush(); np.save(OUT/'valid.npy',valid)
    mani={'contract':'exp024_dinov2s_triplet_features_v1','shape':[n,6,len(CENTERS),384],'centers':CENTERS,'backbone':'vit_small_patch14_dinov2 pretrained lvd142m','img':224,'dtype':'float16','valid_triplets':int(valid.sum()),'elapsed_seconds':time.time()-t0}
    (OUT/'manifest.json').write_text(json.dumps(mani,indent=2))
    print(json.dumps(mani,indent=2),flush=True)
if __name__=='__main__': main()
