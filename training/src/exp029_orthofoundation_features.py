from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=workspace_root()
CACHE=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'
ORI=ROOT/'kaggle_notebooks'/'own_exp027_orientation_metadata'/'orientation_metadata.csv'
OUT=ROOT/'experiments'/'EXP029_ORTHOFOUNDATION_CANONICAL_54_FEATURES'; OUT.mkdir(parents=True,exist_ok=True)
DINO=next((ROOT/'external'/'research_20260913'/'dinov3_6876159a').glob('dinov3-*')); sys.path.insert(0,str(DINO))
WEIGHT=ROOT/'external'/'research_20260913'/'OrthoFoundation'/'OrthoFoudation-L.pth'
from dinov3.hub.backbones import dinov3_vitl16
SLOTS=['SAG_FLUID','SAG_STRUCT','COR_FLUID','COR_STRUCT','AX_FLUID','AX_STRUCT']
class Shards:
    def __init__(self):
        m=json.loads((CACHE/'train_cache_shards_manifest.json').read_text()); self.s=[(r['start'],r['stop'],np.load(CACHE/r['file'],mmap_mode='r')) for r in m['shards']]
    def get(self,i):
        for a,b,x in self.s:
            if a<=i<b:return x[i-a]
        raise IndexError(i)
def main():
    u=pd.read_csv(CACHE/'train_cache_uids.csv'); ids=u.StudyInstanceUID.astype(str).tolist(); masks=np.load(CACHE/'train_cache_masks.npy').astype(np.uint8); sh=Shards()
    o=pd.read_csv(ORI,dtype={'StudyInstanceUID':str}); od={(r.StudyInstanceUID,r.slot):r for r in o.itertuples(index=False)}
    model=dinov3_vitl16(pretrained=False); raw=torch.load(WEIGHT,map_location='cpu',weights_only=True); sd={k.removeprefix('backbone.'):v for k,v in raw.items()}; model.load_state_dict(sd,strict=True); del raw,sd; model.eval().cuda(); torch.backends.cuda.matmul.allow_tf32=True
    feat=np.lib.format.open_memmap(OUT/'features_f16.npy',mode='w+',dtype=np.float16,shape=(len(ids),6,9,1024)); valid=np.zeros((len(ids),6,9),np.uint8)
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1); std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    imgs=[]; keys=[]; t0=time.time(); transforms={'reverse':0,'flip_h':0,'flip_v':0}
    def flush():
        nonlocal imgs,keys
        if not imgs:return
        x=torch.from_numpy(np.stack(imgs)).cuda(non_blocking=True).float().div_(255.); x=(x-mean)/std
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
            for a in range(0,len(x),16):
                z=model(x[a:a+16]).float().cpu().numpy().astype(np.float16)
                for rr,v in enumerate(z):
                    i,s,j=keys[a+rr]; feat[i,s,j]=v; valid[i,s,j]=1
        imgs=[];keys=[]
    for i,uid in enumerate(ids):
        arr=np.array(sh.get(i),copy=False)
        for s,slot in enumerate(SLOTS):
            if masks[i,s].sum()==0:continue
            im=arr[s]; mm=masks[i,s].copy(); r=od.get((uid,slot))
            if r is not None and getattr(r,'header_ok',0)==1:
                if int(getattr(r,'reverse_slice_to_canonical',0) or 0): im=im[::-1]; mm=mm[::-1]; transforms['reverse']+=1
                if float(getattr(r,'row_dot_desired_h',1))<0: im=im[:,:,::-1]; transforms['flip_h']+=1
                if float(getattr(r,'col_dot_desired_v',1))<0: im=im[:,::-1,:]; transforms['flip_v']+=1
            for j in range(9):
                if not mm[j]:continue
                g=np.asarray(im[j]); rgb=np.stack([g,g,g],axis=0)
                imgs.append(rgb);keys.append((i,s,j))
                if len(imgs)>=256:flush()
        if (i+1)%100==0:
            flush(); dt=time.time()-t0; print(f'{i+1}/{len(ids)} elapsed={dt/60:.2f}m eta={dt/(i+1)*(len(ids)-i-1)/60:.2f}m valid={int(valid[:i+1].sum())}',flush=True)
    flush(); feat.flush(); np.save(OUT/'valid.npy',valid)
    mani={'contract':'orthofoundation_l_canonical_54_independent_slices_v1','backbone':'ytrsk/OrthoFoundation OrthoFoudation-L.pth','dinov3_commit':'6876159a11b4df116f30f667f8c9888617df0751','shape':[len(ids),6,9,1024],'dtype':'float16','normalization':'ImageNet mean/std per official OrthoFoundation augmentation.py','canonicalization':'EXP027 geometry: slice reverse + horizontal/vertical axis sign','valid_slices':int(valid.sum()),'transforms':transforms,'elapsed_seconds':time.time()-t0}
    (OUT/'manifest.json').write_text(json.dumps(mani,indent=2)); print(json.dumps(mani,indent=2),flush=True)
if __name__=='__main__':main()