import modal
app=modal.App('rsna-knee-orthodiffusion-audit')
vol=modal.Volume.from_name('rsna-public-models',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12')
       .apt_install('git')
       .pip_install('torch==2.5.1','torchvision==0.20.1','numpy==1.26.4','einops','tqdm','nibabel','pillow','tensorboard','requests>=2.32'))
REV='d6920d49e7d1b3dd0774eed6b8e47e386a54dd81'
COMMIT='99d15186bb8face1728263023d8414d9405c8f7e'
FILES={
 'axial_model.pt':('f7daa0406c6312dacebf1e56ecfdf14b1f3c98c59b24f2fa75d1899490d31a3e',552919566),
 'coronal_model.pt':('dcce97a061af1a6df1f902cc0d5a7f604dc9bd0f52f3b3714be39b288dc541e6',552919566),
 'sagittal_model.pt':('3afc323c58b6446c1ea6dc4f09133d744e5a1609dba7662ac9a15ca688e34951',552919566),
}

def _hash(path):
 import hashlib
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

@app.function(image=image,volumes={'/models':vol},gpu='L4',timeout=2400)
def audit():
 import os,time,requests,torch,subprocess,sys,json
 os.makedirs('/models/orthodiffusion',exist_ok=True)
 downloads={}
 for name,(sha,size) in FILES.items():
  dst='/models/orthodiffusion/'+name
  ok=os.path.exists(dst) and os.path.getsize(dst)==size and _hash(dst)==sha
  if not ok:
   url=f'https://huggingface.co/lanstat0123/orthodiffusion/resolve/{REV}/{name}'
   tmp=dst+'.part';h=__import__('hashlib').sha256();n=0;t0=time.time()
   with requests.get(url,stream=True,timeout=(20,180),headers={'User-Agent':'rsna-public-model-audit/1.0'}) as r:
    r.raise_for_status()
    with open(tmp,'wb') as f:
     for b in r.iter_content(8<<20):
      if b:f.write(b);h.update(b);n+=len(b)
   if n!=size or h.hexdigest()!=sha: raise RuntimeError((name,n,h.hexdigest()))
   os.replace(tmp,dst); downloads[name]={'status':'downloaded','seconds':round(time.time()-t0,2),'bytes':n,'sha256':sha}
  else: downloads[name]={'status':'cached_verified','bytes':size,'sha256':sha}
 vol.commit()
 repo='/tmp/OrthoDiffusion'; subprocess.run(['git','clone','-q','https://github.com/lt-0123/OrthoDiffusion.git',repo],check=True); subprocess.run(['git','-C',repo,'checkout','-q',COMMIT],check=True); sys.path.insert(0,repo)
 from diffusion_model.unet import create_model
 from diffusion_model.trainer import GaussianDiffusion
 results={}
 for name,(sha,size) in FILES.items():
  ck=torch.load('/models/orthodiffusion/'+name,map_location='cpu',weights_only=True)
  top={k:(type(v).__name__,len(v) if isinstance(v,dict) else None) for k,v in ck.items()} if isinstance(ck,dict) else {'type':type(ck).__name__}
  model=create_model(256,64,1,in_channels=1,out_channels=1).cuda()
  diffusion=GaussianDiffusion(model,image_size=256,depth_size=16,timesteps=1000,loss_type='l2').cuda()
  sd=ck['ema'] if isinstance(ck,dict) and 'ema' in ck else ck
  ns={}
  for k,v in sd.items():
   kk=k.replace('denoise_fn.module.','denoise_fn.') if k.startswith('denoise_fn.module.') else k
   ns[kk]=v
  ret=diffusion.load_state_dict(ns,strict=False)
  miss=list(ret.missing_keys); unexp=list(ret.unexpected_keys)
  # deterministic synthetic feature probe. reset seed before each extraction.
  x=torch.linspace(-1,1,16*256*256,device='cuda',dtype=torch.float32).reshape(1,1,16,256,256)
  probes=[]
  for tval,block in [(100,'mid_2'),(200,'mid_0'),(150,'mid_0'),(50,'mid_2')]:
   torch.manual_seed(12345);torch.cuda.manual_seed_all(12345);torch.cuda.reset_peak_memory_stats();t0=time.time()
   with torch.no_grad():
    f=diffusion.get_feature(x,torch.full((1,),tval,dtype=torch.long,device='cuda'),name=block,use_amp=True)
   torch.cuda.synchronize();probes.append({'t':tval,'block':block,'shape':list(f.shape),'dtype':str(f.dtype),'seconds':round(time.time()-t0,3),'peak_gib':round(torch.cuda.max_memory_allocated()/2**30,2)})
  results[name]={'top':top,'missing_n':len(miss),'unexpected_n':len(unexp),'missing_head':miss[:10],'unexpected_head':unexp[:10],'params_m':round(sum(p.numel() for p in diffusion.parameters())/1e6,2),'probes':probes}
  del diffusion,model,ck,sd,ns,x,f;torch.cuda.empty_cache()
 receipt={'repo_commit':COMMIT,'hf_revision':REV,'device':torch.cuda.get_device_name(0),'downloads':downloads,'results':results}
 with open('/models/orthodiffusion/audit_receipt.json','w') as f:json.dump(receipt,f,indent=2)
 vol.commit();return receipt

@app.local_entrypoint()
def main():
 import pprint;pprint.pp(audit.remote())