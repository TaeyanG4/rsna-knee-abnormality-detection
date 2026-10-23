import modal
app=modal.App('rsna-knee-orthodiffusion-batch-bench')
vol=modal.Volume.from_name('rsna-public-models',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12')
       .apt_install('git')
       .pip_install('torch==2.5.1','torchvision==0.20.1','numpy==1.26.4','einops','tqdm','nibabel','pillow','tensorboard'))
COMMIT='99d15186bb8face1728263023d8414d9405c8f7e'
@app.function(image=image,volumes={'/models':vol},gpu='L4',timeout=1200)
def bench():
 import os,sys,subprocess,time,torch
 repo='/tmp/OrthoDiffusion'
 subprocess.run(['git','clone','-q','https://github.com/lt-0123/OrthoDiffusion.git',repo],check=True)
 subprocess.run(['git','-C',repo,'checkout','-q',COMMIT],check=True)
 sys.path.insert(0,repo)
 from diffusion_model.unet import create_model
 from diffusion_model.trainer import GaussianDiffusion
 ck=torch.load('/models/orthodiffusion/axial_model.pt',map_location='cpu',weights_only=True)
 model=create_model(256,64,1,in_channels=1,out_channels=1).cuda()
 diff=GaussianDiffusion(model,image_size=256,depth_size=16,timesteps=1000,loss_type='l2').cuda()
 sd={}
 for k,v in ck['ema'].items():
  kk=k.replace('denoise_fn.module.','denoise_fn.') if k.startswith('denoise_fn.module.') else k
  sd[kk]=v
 ret=diff.load_state_dict(sd,strict=False)
 assert not ret.missing_keys and not ret.unexpected_keys
 diff.eval(); out=[]
 for b in [1,2,4,6,8,10,12,16]:
  try:
   x=torch.linspace(-1,1,b*16*256*256,device='cuda',dtype=torch.float32).reshape(b,1,16,256,256)
   t=torch.full((b,),50,dtype=torch.long,device='cuda')
   for _ in range(2):
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    with torch.no_grad(): y=diff.get_feature(x,t,name='mid_0',use_amp=True)
   torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); st=time.time(); steps=8
   for _ in range(steps):
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    with torch.no_grad(): y=diff.get_feature(x,t,name='mid_0',use_amp=True)
   torch.cuda.synchronize(); dt=time.time()-st
   out.append({'batch':b,'volumes_per_s':round(b*steps/dt,2),'step_ms':round(1000*dt/steps,2),'peak_gib':round(torch.cuda.max_memory_allocated()/2**30,2),'feature_shape':list(y.shape)})
   del x,t,y; torch.cuda.empty_cache()
  except Exception as e:
   out.append({'batch':b,'error':repr(e)}); torch.cuda.empty_cache()
 return {'device':torch.cuda.get_device_name(0),'params_m':round(sum(p.numel() for p in diff.parameters())/1e6,2),'lp_opt_axial':'t50_mid0','results':out}
@app.local_entrypoint()
def main(): print(bench.remote())
