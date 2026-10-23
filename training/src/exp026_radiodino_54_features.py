try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import json,time
from pathlib import Path
import numpy as np
import torch
import timm
from safetensors.torch import load_file

ROOT=workspace_root()
CACHE=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'
WROOT=ROOT/'external'/'research_20260913'/'Snarcy__RadioDino-s16'
OUT=ROOT/'experiments'/'EXP026_RADIODINO_54_FEATURES'; OUT.mkdir(parents=True,exist_ok=True)

class Shards:
    def __init__(self,root):
        m=json.loads((root/'train_cache_shards_manifest.json').read_text())
        self.s=[(r['start'],r['stop'],np.load(root/r['file'],mmap_mode='r')) for r in m['shards']]
    def get(self,i):
        for a,b,x in self.s:
            if a<=i<b:return x[i-a]
        raise IndexError(i)

def main():
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    dev='cuda'
    cfg=json.loads((WROOT/'config.json').read_text())
    model=timm.create_model(cfg['architecture'],pretrained=False,num_classes=0,global_pool=cfg.get('global_pool','token'))
    sd=load_file(str(WROOT/'model.safetensors'))
    miss,unexp=model.load_state_dict(sd,strict=False)
    if miss or unexp:
        raise RuntimeError(f'weight mismatch missing={miss[:10]} unexpected={unexp[:10]}')
    model=model.to(dev).eval()
    for p in model.parameters():p.requires_grad=False
    shards=Shards(CACHE); masks=np.load(CACHE/'train_cache_masks.npy')
    n=masks.shape[0]; dim=int(cfg['num_features'])
    fp=OUT/'features_f16.npy'; vp=OUT/'valid.npy'
    feat=np.lib.format.open_memmap(fp,mode='w+',dtype=np.float16,shape=(n,6,9,dim))
    valid=np.zeros((n,6,9),np.uint8)
    mean=torch.tensor(cfg['pretrained_cfg']['mean'],device=dev).view(1,3,1,1)
    std=torch.tensor(cfg['pretrained_cfg']['std'],device=dev).view(1,3,1,1)
    imgs=[]; keys=[]; t0=time.time(); batch=512
    def flush():
        nonlocal imgs,keys
        if not imgs:return
        x=torch.from_numpy(np.stack(imgs)).to(dev,non_blocking=True).float().div_(255.)
        x=x[:,None,:,:].repeat(1,3,1,1)
        x=(x-mean)/std
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
            for a in range(0,len(x),batch):
                z=model(x[a:a+batch]).float().cpu().numpy().astype(np.float16)
                for j,v in enumerate(z):
                    i,s,p=keys[a+j]; feat[i,s,p]=v; valid[i,s,p]=1
        imgs=[];keys=[]
    for i in range(n):
        arr=shards.get(i)
        for s in range(6):
            for p in range(9):
                if masks[i,s,p]:
                    imgs.append(arr[s,p]);keys.append((i,s,p))
        if len(imgs)>=batch:flush()
        if (i+1)%100==0:
            flush();dt=time.time()-t0
            print(f'{i+1}/{n} elapsed={dt/60:.2f}m eta={dt/(i+1)*(n-i-1)/60:.2f}m valid={int(valid[:i+1].sum())}',flush=True)
    flush();feat.flush();np.save(vp,valid)
    mani={'contract':'radiodino_s16_54_independent_slices_v1','backbone':'Snarcy/RadioDino-s16','shape':[n,6,9,dim],'dtype':'float16','normalization':'model ImageNet mean/std','input':'independent sampled grayscale slice repeated RGB; no fake adjacent triplets','valid_slices':int(valid.sum()),'elapsed_seconds':time.time()-t0}
    (OUT/'manifest.json').write_text(json.dumps(mani,indent=2),encoding='utf-8')
    print(json.dumps(mani,indent=2),flush=True)
if __name__=='__main__':main()
