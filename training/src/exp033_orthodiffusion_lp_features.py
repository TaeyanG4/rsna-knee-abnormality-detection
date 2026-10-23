from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=workspace_root()
CODE=ROOT/'external'/'research_20260913'/'OrthoDiffusion'/'code_99d15186'/'OrthoDiffusion-99d15186bb8face1728263023d8414d9405c8f7e'
WEIGHTS=ROOT/'external'/'research_20260913'/'OrthoDiffusion'/'orthodiffusion'
sys.path.insert(0,str(CODE))
from diffusion_model.unet import create_model
from diffusion_model.trainer import GaussianDiffusion

PLANES=[
 ('Sagittal','sagittal_model.pt',50,'mid_0'),
 ('Coronal','coronal_model.pt',100,'mid_2'),
 ('Axial','axial_model.pt',50,'mid_0'),
]

class Shards:
 def __init__(self,cache:Path):
  mp=next((cache/n for n in ('orthodiffusion_cache_manifest.json','orthodiff_manifest.json') if (cache/n).is_file()),None)
  if mp is None: raise FileNotFoundError('OrthoDiffusion cache manifest not found')
  m=json.loads(mp.read_text())
  self.manifest=m; self.shards=[]
  for r in m['shards']:
   self.shards.append((int(r['start']),int(r['stop']),np.load(cache/r['file'],mmap_mode='r')))
  self.n=int(m['studies'])
 def batch(self,idx,plane):
  out=np.empty((len(idx),16,256,256),np.uint8)
  for q,i in enumerate(idx):
   for a,b,x in self.shards:
    if a<=i<b:
     out[q]=x[i-a,plane]; break
   else: raise IndexError(i)
  return out

def load_model(weight:Path):
 ck=torch.load(weight,map_location='cpu',weights_only=True)
 model=create_model(256,64,1,in_channels=1,out_channels=1).cuda()
 diff=GaussianDiffusion(model,image_size=256,depth_size=16,timesteps=1000,loss_type='l2').cuda()
 sd={}
 for k,v in ck['ema'].items():
  kk=k.replace('denoise_fn.module.','denoise_fn.') if k.startswith('denoise_fn.module.') else k
  sd[kk]=v
 ret=diff.load_state_dict(sd,strict=False)
 if ret.missing_keys or ret.unexpected_keys:
  raise RuntimeError({'missing':ret.missing_keys[:10],'unexpected':ret.unexpected_keys[:10]})
 return diff.eval()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out',default=str(ROOT/'experiments'/'EXP033_ORTHODIFF_LP_FEATURES')); ap.add_argument('--batch',type=int,default=8); args=ap.parse_args()
 cache=workspace_path(args.cache_dir); out=workspace_path(args.out); out.mkdir(parents=True,exist_ok=True)
 sh=Shards(cache); mask_path=next((cache/n for n in ('orthodiffusion_cache_masks.npy','orthodiff_masks.npy') if (cache/n).is_file()),None); uid_path=next((cache/n for n in ('orthodiffusion_cache_uids.csv','orthodiff_uids.csv') if (cache/n).is_file()),None)
 if mask_path is None or uid_path is None: raise FileNotFoundError('OrthoDiffusion mask/UID files not found')
 masks=np.load(mask_path).astype(np.uint8); uids=pd.read_csv(uid_path).StudyInstanceUID.astype(str).tolist()
 assert len(uids)==sh.n==len(masks)
 receipts=[]
 for pi,(plane,wname,tval,block) in enumerate(PLANES):
  fp=out/f'{plane.lower()}_features_f16.npy'; valid=out/f'{plane.lower()}_valid.npy'
  mm=np.lib.format.open_memmap(fp,mode='w+',dtype=np.float16,shape=(sh.n,256,1,16,16)); vm=np.lib.format.open_memmap(valid,mode='w+',dtype=np.uint8,shape=(sh.n,)); vm[:]=masks[:,pi]
  model=load_model(WEIGHTS/wname); torch.manual_seed(20260913+pi); torch.cuda.manual_seed_all(20260913+pi); t0=time.time(); done=0
  with torch.no_grad():
   for a in range(0,sh.n,args.batch):
    ii=np.arange(a,min(a+args.batch,sh.n)); x=sh.batch(ii,pi).astype(np.float32)/127.5-1.0; xt=torch.from_numpy(x[:,None]).cuda(non_blocking=True); tt=torch.full((len(ii),),tval,dtype=torch.long,device='cuda')
    with torch.amp.autocast('cuda',dtype=torch.float16): feat=model.get_feature(xt,tt,name=block,use_amp=True)
    z=feat.detach().cpu().numpy().astype(np.float16); z[masks[ii,pi]==0]=0; mm[ii]=z; done+=len(ii)
    if done%400 < args.batch: print(plane,done,'/',sh.n,'elapsed',round(time.time()-t0,1),flush=True)
  mm.flush();vm.flush(); receipts.append({'plane':plane,'weight':wname,'timestep':tval,'block':block,'shape':[sh.n,256,1,16,16],'valid':int(masks[:,pi].sum()),'seconds':time.time()-t0})
  del model,mm,vm; torch.cuda.empty_cache()
 (out/'manifest.json').write_text(json.dumps({'contract':'orthodiffusion_knee_LP-Opt_official_v1','repo_commit':'99d15186bb8face1728263023d8414d9405c8f7e','hf_revision':'d6920d49e7d1b3dd0774eed6b8e47e386a54dd81','noise_seed_by_plane':{'Sagittal':20260913,'Coronal':20260914,'Axial':20260915},'planes':receipts},indent=2),encoding='utf-8')
 print(json.dumps(receipts,indent=2),flush=True)
if __name__=='__main__': main()
