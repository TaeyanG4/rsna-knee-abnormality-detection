"""Colab session bootstrap for bundle v2 (run with `colab exec -s <name> -f tools/colab/bootstrap_v2.py`).

Expects /root/.kaggle/access_token to exist on the VM (uploaded with `colab upload`). Installs deps, downloads the private
datasets knee-runners-v2 (small) and knee-raptor-corpus-336 (32 GB, per file, resumable) into /content, then starts the job
runner in the background: for each job in JOBS it runs tools/remote_run_v2.py sequentially, logging to /content/results/runner.log.
Edit JOBS below before exec. Poll with: `echo "print(open('/content/results/runner.log').read()[-3000:])" | colab exec -s <name>`.
"""
import os, subprocess, sys, time, glob, json
JOBS = os.environ.get('KNEE_JOBS', 'exp094:seed43,exp094:geoaug,exp094:strat').split(',')
USER = os.environ.get('KAGGLE_USERNAME_HINT', 'taeyangg4')
os.makedirs('/content/results', exist_ok=True)
tok = '/root/.kaggle/access_token'
assert os.path.exists(tok), 'upload the Kaggle access token first'
os.chmod(tok, 0o600); os.environ['KAGGLE_API_TOKEN'] = open(tok).read().strip()
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout, flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'kaggle', 'timm>=1.0.19', 'psutil', 'opencv-python-headless', 'pydicom'], check=True)
if not glob.glob('/content/tools/tools/remote_run_v2.py'):
    subprocess.run(['kaggle', 'datasets', 'download', '-d', f'{USER}/knee-runners-v2', '-p', '/content/tools', '--unzip', '-q', '-o'], check=True)
print('tools ok', flush=True)
script = f"""
import os, subprocess, sys, time, json
os.environ['KAGGLE_API_TOKEN'] = open('{tok}').read().strip()
log = open('/content/results/runner.log', 'a', buffering=1)
jobs = {JOBS!r}
summary = []
for job in jobs:
    t0 = time.time(); log.write(f'=== {{job}} start {{time.strftime("%H:%M:%S")}}\\n')
    cmd = [sys.executable, '/content/tools/tools/remote_run_v2.py', '--job', job, '--tools', '/content/tools', '--corpus', '/content/corpus', '--kaggle-user', '{USER}']
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    summary.append(dict(job=job, exit=p.returncode, seconds=round(time.time()-t0)))
    log.write(f'=== {{job}} exit {{p.returncode}} after {{round(time.time()-t0)}}s\\n')
    json.dump(summary, open('/content/results/runner-summary.json','w'), indent=1)
log.write('=== ALL DONE\\n')
"""
open('/content/results/runner.py', 'w').write(script)
p = subprocess.Popen(['nohup', sys.executable, '/content/results/runner.py'], stdout=open('/content/results/runner.out', 'a'), stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
print('runner started pid', p.pid, 'jobs', JOBS, flush=True)
