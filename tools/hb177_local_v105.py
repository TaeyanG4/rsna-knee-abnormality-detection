"""H-B-177 v1.0.5 local-only environment checks; unchanged v1.0.4 functions.
Changelog: bind environment reference to the new seed44 stage. No remote actions.
"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'.local/scratch/hb177/H-B-177/v1.0.5'

PACKAGES=('torch','torchvision','timm','numpy','opencv-python-headless','psutil','pydicom','safetensors','pillow','huggingface-hub','pyyaml')

def need(value,message):
    if not value: raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def snapshot(cuda=False):
    """CPU snapshot never probes CUDA availability. CUDA details only inside admitted smoke."""
    import torch
    libs={}
    for name in PACKAGES:
        try: libs[name]=metadata.version(name)
        except metadata.PackageNotFoundError: libs[name]=None
    # Import actual runtime modules, not just distribution metadata.
    import cv2, timm, numpy, torchvision, pydicom
    value=dict(python=platform.python_version(),platform=platform.system(),packages=libs,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        imports=dict(cv2=cv2.__version__,timm=timm.__version__,numpy=numpy.__version__,
                     torchvision=torchvision.__version__,pydicom=pydicom.__version__),
        settings=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
            mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math_sdp=torch.backends.cuda.math_sdp_enabled()),cuda_initialized=torch.cuda.is_initialized())
    if cuda:
        p=torch.cuda.get_device_properties(0)
        value.update(gpu=p.name,capability=list(torch.cuda.get_device_capability(0)),gpu_bytes=p.total_memory)
    else:
        need(not torch.cuda.is_initialized(),'CPU environment check touched CUDA')
    return value

def require_local_environment():
    reference=read(STAGE/'local-environment.json')
    current={name:metadata.version(name) for name in PACKAGES}
    need(current==reference['packages'],'live local package drift')
    need(platform.python_version()==reference['python'] and platform.system()==reference['platform']=='Windows',
         'live local Python/platform drift')
