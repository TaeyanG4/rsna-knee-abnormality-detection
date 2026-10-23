"""Push a Kaggle kernel directory safely.

Waits until every attached dataset reports status "ready" (exits 2 without pushing otherwise), retries client SSL
errors and never passes --timeout (D-B-015). It also removes `machine_shape` from kernel-metadata.json (original kept
next to the directory) — harmless but NOT a fix: with --accelerator the CLI ignores the metadata key (D-B-039, corrected
2026-10-08: the empty-log ERRORs of 2026-10-07 were intermittent and a plain retry succeeded).

  python tools/kaggle_push.py <kernel_dir> [--accelerator NvidiaTeslaT4]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = [str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-m', 'kaggle']


def strip_machine_shape(kernel_dir):
    """Remove `machine_shape` from kernel-metadata.json; save the original as <dir>.kernel-metadata.orig.json. Returns the removed value."""
    d = Path(kernel_dir)
    meta = json.loads((d / 'kernel-metadata.json').read_text(encoding='utf-8'))
    if 'machine_shape' not in meta:
        return None
    backup = d.parent / f'{d.name}.kernel-metadata.orig.json'
    if not backup.exists():
        backup.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    value = meta.pop('machine_shape')
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return value


def kag(*a, timeout=1800):
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    out = ''
    for _ in range(4):
        r = subprocess.run([*K, *a], capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=timeout)
        out = (r.stdout + r.stderr).strip()
        if 'SSLError' not in out:
            break
        time.sleep(20)
    return out


def wait_ready(datasets, status=None, tries=60, pause=30):
    """Return the datasets that never reported status "ready" (empty list = all ready).

    Every attached dataset is checked, not only the kernel owner's (astra review 2026-10-07: shared private
    datasets were skipped, and a timeout fell through to the push)."""
    status = status or (lambda ds: kag('datasets', 'status', ds))
    missing = []
    for ds in datasets:
        last = ''
        for i in range(tries):
            lines = (status(ds) or '').splitlines()
            last = lines[-1].strip() if lines else ''
            if last == 'ready':
                break
            if i + 1 < tries:
                time.sleep(pause)
        print(ds, 'status', last, flush=True)
        if last != 'ready':
            missing.append(ds)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kernel_dir')
    ap.add_argument('--accelerator', default='NvidiaTeslaT4')
    a = ap.parse_args()
    removed = strip_machine_shape(a.kernel_dir)
    print('machine_shape removed:', removed, flush=True)
    meta = json.loads((Path(a.kernel_dir) / 'kernel-metadata.json').read_text(encoding='utf-8'))
    missing = wait_ready(meta.get('dataset_sources', []))
    if missing:
        print('NOT PUSHED: datasets not ready:', missing, flush=True)
        sys.exit(2)
    out = kag('kernels', 'push', '-p', a.kernel_dir, '--accelerator', a.accelerator)
    print(out.splitlines()[-1] if out else '', flush=True)
    sys.exit(0 if 'successfully pushed' in out else 1)


if __name__ == '__main__':
    main()
