"""Colab VM: prefetch the private corpus dataset file by file (resumable) while the job tooling is being built.

Run on the VM:  nohup python /content/prefetch_corpus_v3.py > /content/results/prefetch.out 2>&1 &
Reads the Kaggle token from /root/.kaggle/access_token into the environment only; never prints it.
"""
import json, os, subprocess, sys, time
from pathlib import Path

REF = 'taeyangg4/knee-raptor-corpus-336'
DEST = Path('/content/corpus')
LOG = Path('/content/results/corpus.log')
FILES = [('manifest.json', 4923), ('ids.npy', 1129088), ('masks.npy', 282368), ('picks.jsonl', 5225387), ('uids.csv', 312024)] + \
        [(f'vols_shard{k:02d}.npy', 2890137728) for k in range(11)] + [('vols_shard11.npy', 72253568)]


def log(*a):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a) + '\n')


tok = Path('/root/.kaggle/access_token')
if not tok.exists():
    log('no token file; stop'); sys.exit(1)
os.environ['KAGGLE_API_TOKEN'] = tok.read_text().strip()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'kaggle'], check=False)
DEST.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for name, size in FILES:
    target = DEST / name
    if target.exists() and target.stat().st_size == size:
        log('have', name); continue
    for attempt in range(4):
        r = subprocess.run(['kaggle', 'datasets', 'download', '-d', REF, '-f', name, '-p', str(DEST), '--unzip', '-q', '-o'],
                           capture_output=True, text=True)
        if target.exists() and target.stat().st_size == size:
            break
        log('retry', name, attempt, (r.stdout + r.stderr)[-200:].replace('\n', ' '))
        time.sleep(30)
    ok = target.exists() and target.stat().st_size == size
    log('done' if ok else 'FAILED', name, f'{size / 1e9:.2f} GB', f'{time.time() - t0:.0f}s')
    if not ok:
        sys.exit(2)
log('CORPUS COMPLETE', f'{time.time() - t0:.0f}s')
json.dump(dict(complete=True, seconds=round(time.time() - t0)), open('/content/results/corpus_done.json', 'w'))
