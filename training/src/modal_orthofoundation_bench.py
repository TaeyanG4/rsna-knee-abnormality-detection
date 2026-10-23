import modal
app=modal.App('rsna-knee-orthofoundation-bench')
image=(modal.Image.debian_slim(python_version='3.12')
       .apt_install('git')
       .pip_install('torch==2.5.1','torchvision==0.20.1','numpy>=2.0','requests>=2.32','einops','omegaconf','ftfy')
       .pip_install('torchmetrics','regex','scikit-learn','submitit','termcolor'))

@app.function(image=image,gpu='L4',timeout=1200)
def bench():
    import os,subprocess,requests,hashlib,time,torch
    repo='/tmp/dinov3'; commit='6876159a11b4df116f30f667f8c9888617df0751'
    subprocess.run(['git','clone','-q','https://github.com/facebookresearch/dinov3.git',repo],check=True)
    subprocess.run(['git','-C',repo,'checkout','-q',commit],check=True)
    url='https://media.githubusercontent.com/media/ytrsk/OrthoFoundation/4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470/OrthoFoudation-L.pth'
    p='/tmp/ortho.pth'; h=hashlib.sha256(); n=0
    with requests.get(url,stream=True,timeout=(20,180)) as r:
        r.raise_for_status()
        with open(p,'wb') as f:
            for b in r.iter_content(8<<20):
                if b: f.write(b); h.update(b); n+=len(b)
    assert h.hexdigest()=='385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9'
    ck=torch.load(p,map_location='cpu',weights_only=True)
    sd={k[len('backbone.'):]:v for k,v in ck.items() if k.startswith('backbone.')}
    t0=time.time()
    model=torch.hub.load(repo,'dinov3_vitl16',source='local',pretrained=False).eval()
    miss,unexp=model.load_state_dict(sd,strict=False)
    if miss or unexp:
        return {'ok':False,'missing':miss[:20],'unexpected':unexp[:20],'n_missing':len(miss),'n_unexpected':len(unexp)}
    model=model.cuda(); load_s=time.time()-t0
    results=[]
    for batch in [4,8,16]:
        try:
            x=torch.randn(batch,3,224,224,device='cuda',dtype=torch.float16)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                for _ in range(3): y=model(x)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t=time.time(); steps=10
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                for _ in range(steps): y=model(x)
            torch.cuda.synchronize(); dt=time.time()-t
            results.append({'batch':batch,'images_per_s':round(batch*steps/dt,1),'step_ms':round(dt/steps*1000,1),'peak_vram_gib':round(torch.cuda.max_memory_allocated()/2**30,2),'shape':list(y.shape) if hasattr(y,'shape') else str(type(y))})
            del x,y; torch.cuda.empty_cache()
        except Exception as e:
            results.append({'batch':batch,'error':repr(e)})
            torch.cuda.empty_cache()
    return {'ok':True,'device':torch.cuda.get_device_name(0),'load_s':round(load_s,2),'params_m':round(sum(p.numel() for p in model.parameters())/1e6,1),'results':results}

@app.local_entrypoint()
def main(): print(bench.remote())

