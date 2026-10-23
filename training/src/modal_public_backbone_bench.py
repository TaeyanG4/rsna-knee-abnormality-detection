import modal

app = modal.App('rsna-knee-public-backbone-bench')
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('git')
         .pip_install('torch==2.5.1','timm==1.0.19','huggingface_hub>=0.34','safetensors>=0.5','numpy>=2.0','requests>=2.32'))

@app.function(image=image, gpu='L4', timeout=1200)
def bench_timm(name: str, batch: int = 32, size: int = 224):
    import time, torch, timm
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    t0=time.time()
    try:
        model=timm.create_model(name, pretrained=True, num_classes=0).eval().cuda()
        load_s=time.time()-t0
        x=torch.randn(batch,3,size,size,device='cuda',dtype=torch.float16)
        # warmup
        with torch.autocast('cuda',dtype=torch.float16):
            for _ in range(5): y=model(x)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t1=time.time()
        n=20
        with torch.autocast('cuda',dtype=torch.float16):
            for _ in range(n): y=model(x)
        torch.cuda.synchronize(); dt=time.time()-t1
        return {'model':name,'ok':True,'load_s':round(load_s,2),'batch':batch,'size':size,
                'images_per_s':round(batch*n/dt,1),'step_ms':round(dt/n*1000,1),
                'peak_vram_gib':round(torch.cuda.max_memory_allocated()/2**30,2),
                'out_shape':list(y.shape),'device':torch.cuda.get_device_name(0)}
    except Exception as e:
        return {'model':name,'ok':False,'error':repr(e)}

@app.function(image=image, gpu='L4', timeout=1200)
def inspect_orthofoundation():
    import requests, torch, os, hashlib, time
    url='https://media.githubusercontent.com/media/ytrsk/OrthoFoundation/4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470/OrthoFoudation-L.pth'
    path='/tmp/OrthoFoudation-L.pth'
    t0=time.time(); h=hashlib.sha256(); n=0
    with requests.get(url,stream=True,timeout=(20,180)) as r:
        r.raise_for_status()
        with open(path,'wb') as f:
            for b in r.iter_content(8<<20):
                if b:
                    f.write(b); h.update(b); n+=len(b)
    expected='385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9'
    if h.hexdigest()!=expected:
        return {'ok':False,'stage':'hash','bytes':n,'sha256':h.hexdigest()}
    obj=torch.load(path,map_location='cpu',weights_only=True)
    def tensor_summary(d):
        if not isinstance(d,dict): return {'type':type(d).__name__}
        keys=list(d.keys())
        tensors=[]
        for k,v in d.items():
            if torch.is_tensor(v): tensors.append((k,list(v.shape),str(v.dtype)))
        return {'key_count':len(keys),'first_keys':keys[:30],'tensor_count':len(tensors),'first_tensors':tensors[:30]}
    out={'ok':True,'bytes':n,'sha256':h.hexdigest(),'download_s':round(time.time()-t0,2),'top_type':type(obj).__name__}
    if isinstance(obj,dict):
        out['top_keys']=list(obj.keys())[:50]
        for candidate in ['model','student','teacher','state_dict']:
            if candidate in obj: out[candidate]=tensor_summary(obj[candidate])
        if not any(c in obj for c in ['model','student','teacher','state_dict']): out['root']=tensor_summary(obj)
    return out

@app.local_entrypoint()
def main():
    models=['hf_hub:Snarcy/RadioDino-s16','vit_small_patch14_dinov2.lvd142m']
    handles=[bench_timm.spawn(m,32,224) for m in models]
    ortho=inspect_orthofoundation.spawn()
    for h in handles: print(h.get())
    print(ortho.get())
