import modal

app = modal.App('rsna-knee-modal-smoke')
image = modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.5.1')

@app.function(image=image, gpu='L4', timeout=600)
def gpu_smoke():
    import subprocess, torch, time
    info = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'], text=True).strip()
    assert torch.cuda.is_available(), 'CUDA unavailable'
    x = torch.randn(4096,4096, device='cuda', dtype=torch.float16)
    y = torch.randn(4096,4096, device='cuda', dtype=torch.float16)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(20):
        z = x @ y
    torch.cuda.synchronize(); dt=time.time()-t0
    return {
        'nvidia_smi': info,
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
        'vram_gib': round(torch.cuda.get_device_properties(0).total_memory/2**30, 2),
        'matmul20_seconds': round(dt, 3),
        'checksum': float(z[0,0].float().cpu()),
    }

@app.local_entrypoint()
def main():
    print(gpu_smoke.remote())
