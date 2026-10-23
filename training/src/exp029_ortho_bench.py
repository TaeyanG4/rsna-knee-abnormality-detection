try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import sys,time,torch
from pathlib import Path
root=workspace_root(); d=next((root/'external'/'research_20260913'/'dinov3_6876159a').glob('dinov3-*')); sys.path.insert(0,str(d))
from dinov3.hub.backbones import dinov3_vitl16
m=dinov3_vitl16(pretrained=False); raw=torch.load(root/'external'/'research_20260913'/'OrthoFoundation'/'OrthoFoudation-L.pth',map_location='cpu',weights_only=True); sd={k.removeprefix('backbone.'):v for k,v in raw.items()}; m.load_state_dict(sd,strict=True); m.eval().cuda(); torch.backends.cuda.matmul.allow_tf32=True
for b in [8,16,24,32,40]:
    try:
        x=torch.randn(b,3,224,224,device='cuda'); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
            for _ in range(2): m(x)
            torch.cuda.synchronize(); t=time.time()
            for _ in range(10): y=m(x)
            torch.cuda.synchronize(); dt=time.time()-t
        print({'batch':b,'img_s':round(b*10/dt,1),'step_ms':round(dt/10*1000,1),'peak_gib':round(torch.cuda.max_memory_allocated()/2**30,2),'shape':list(y.shape)},flush=True)
        del x,y
    except RuntimeError as e:
        print({'batch':b,'error':str(e)[:200]},flush=True); torch.cuda.empty_cache()