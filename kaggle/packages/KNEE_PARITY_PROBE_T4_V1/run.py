"""Private parity probe (H-B-16): the same B8 probe on Kaggle T4, default image then pinned to the local stack."""
import glob, json, os, subprocess, sys, time
t0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout, flush=True)
probe = glob.glob("/kaggle/input/**/tools/parity_b8_probe.py", recursive=True); assert probe, "knee-runners-v2 not attached"
bundle = os.path.dirname(os.path.dirname(probe[0]))
corpus = os.path.dirname(glob.glob("/kaggle/input/**/vols_shard00.npy", recursive=True)[0])
ckpt = glob.glob("/kaggle/input/**/weights/B224.pt", recursive=True); assert len(ckpt) == 1, ckpt
os.makedirs("/kaggle/working/parity", exist_ok=True)
base_env = {**os.environ, "KNEE_WORKSPACE_ROOT": bundle, "PYTHONUTF8": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": "0"}
def probe_run(name, env):
    cmd = [sys.executable, probe[0], "--out", f"/kaggle/working/parity/{name}", "--b-ckpt", ckpt[0], "--corpus", corpus]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    open(f"/kaggle/working/parity/{name}.log", "w").write(r.stdout[-20000:] + "\n--- stderr\n" + r.stderr[-20000:])
    print(name, "exit", r.returncode, r.stdout[-400:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode
status = {"default": probe_run("kaggle_default", base_env)}
pin = "/kaggle/temp/pin"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", pin, "torch==2.5.1", "torchvision==0.20.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], capture_output=True, text=True)
print("pin torch exit", r.returncode, r.stderr[-800:], flush=True)
r2 = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", pin, "--no-deps", "timm==1.0.19"], capture_output=True, text=True)
print("pin timm exit", r2.returncode, r2.stderr[-400:], flush=True)
pin_env = {**base_env, "PYTHONPATH": pin + os.pathsep + os.environ.get("PYTHONPATH", "")}
check = subprocess.run([sys.executable, "-c", "import torch,timm;print(torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),timm.__version__,torch.cuda.get_device_name(0))"], env=pin_env, capture_output=True, text=True)
print("pinned stack:", check.stdout.strip(), check.stderr[-600:], flush=True)
status["pinned"] = probe_run("kaggle_pinned", pin_env) if r.returncode == 0 and check.returncode == 0 else "pin_failed"
status["pinned_stack"] = check.stdout.strip(); status["seconds"] = round(time.time() - t0)
json.dump(status, open("/kaggle/working/parity/status.json", "w"), indent=1)
print(json.dumps(status), flush=True)
