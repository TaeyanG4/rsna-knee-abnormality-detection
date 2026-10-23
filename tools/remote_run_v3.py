"""Remote entry for bundle v3 (Kaggle T4): stage the workspace, then train every fold of a registered EXP101 protocol in one session.

  python tools/remote_run_v3.py --job exp101:rn34-5fold --tools DIR --corpus DIR [--folds 0,1,2,3,4] [--results DIR] [--budget-hours 11]

Each fold runs training/runners/exp101_kfold_arch.py with --output-root <results>, so <results>/<tag>/ (train.log, history.json after every
epoch, model_best.pt, oof_best.csv, oof_k36.csv, summary/provenance/output-inventory) is written while the session runs and survives a later
timeout. <results>/logs/<tag>.console.log keeps the runner's stdout/stderr; <results>/exp101-status.json is rewritten after every step
(software, GPU, backbone weight file hash, per-fold exit code / seconds / updates / verification). A failed fold does not stop the next one
(independent fits). A fold starts only if 1.3 x the slowest finished fold (prior 1 h) still fits the budget. Environment for the runner:
thread variables = 1, HF_HUB_OFFLINE=1 after the backbone weights were downloaded once with internet, KNEE_REMOTE_RELAX_ENV=1 (environment and
initial state recorded, not enforced; D-B-008). Always exits 0 so Kaggle keeps the output: success is exp101-status.json state COMPLETE plus
tools/exp101_kfold_arch.py fetch verifying every fold by hash, never the kernel status alone (D-A-008).
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = {'exp101:rn34-5fold': dict(tool='exp101_kfold_arch', folds=[0, 1, 2, 3, 4])}
STATUS = 'exp101-status.json'
TOOLS_DATASET = 'knee-runners-v3'
SOFTWARE_PROBE = r'''
import json, platform, sys
d = {"python": sys.version.split()[0], "platform": platform.platform()}
for m in ("torch", "torchvision", "timm", "numpy", "cv2", "pandas", "huggingface_hub", "safetensors"):
    try:
        d[m] = __import__(m).__version__
    except Exception as e:
        d[m] = "missing: " + type(e).__name__
try:
    import torch
    d.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), cuda_available=torch.cuda.is_available(), gpus=torch.cuda.device_count(),
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception as e:
    d["torch_error"] = repr(e)
print(json.dumps(d))
'''


def load_run_remote(tools_dir):
    spec = importlib.util.spec_from_file_location('run_remote', Path(tools_dir) / 'tools' / 'colab' / 'run_remote.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def write_status(results, status):
    status['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    tmp = Path(results) / (STATUS + '.tmp')
    tmp.write_text(json.dumps(status, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, Path(results) / STATUS)


def run_json(argv, env, timeout):
    r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'{argv[:3]} exit {r.returncode}: {(r.stderr or r.stdout)[-1500:]}')
    return json.loads(r.stdout.strip().splitlines()[-1])


def software(python, env):
    info = run_json([python, '-c', SOFTWARE_PROBE], env, 300)
    if str(info.get('timm', 'missing')).startswith('missing'):
        r = subprocess.run([python, '-m', 'pip', 'install', '-q', 'timm>=1.0.19'], capture_output=True, text=True)
        info = {**run_json([python, '-c', SOFTWARE_PROBE], env, 300), 'timm_installed': r.returncode == 0}
    return info


def fetch_weights(python, arch, env):
    """Download the pretrained backbone once with internet, then prove it loads with HF_HUB_OFFLINE=1 (the runner's registered setting)."""
    probe = ('import json, timm; m = timm.create_model(%r, pretrained=True, num_classes=0); '
             'print(json.dumps(dict(params=sum(p.numel() for p in m.parameters()))))' % arch)
    t0 = time.time()
    online = run_json([python, '-c', probe], {**env, 'HF_HUB_OFFLINE': '0'}, 1800)
    offline = run_json([python, '-c', probe], {**env, 'HF_HUB_OFFLINE': '1'}, 600)
    roots = [os.environ.get('HF_HUB_CACHE'), os.path.join(os.environ['HF_HOME'], 'hub') if os.environ.get('HF_HOME') else None,
             str(Path.home() / '.cache' / 'huggingface' / 'hub')]
    files = []
    for r in [x for x in roots if x]:
        files += glob.glob(os.path.join(r, 'models--timm--' + arch, 'snapshots', '*', '*.safetensors'))
        files += glob.glob(os.path.join(r, 'models--timm--' + arch, 'snapshots', '*', '*.bin'))
    hashes = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sorted(set(files))}
    return dict(arch=arch, online=online, offline=offline, files=hashes, seconds=round(time.time() - t0))


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--job', choices=list(JOBS), required=True)
    ap.add_argument('--tools'); ap.add_argument('--corpus'); ap.add_argument('--workspace'); ap.add_argument('--results')
    ap.add_argument('--folds', default=None, help='comma list (default: all folds of the job)')
    ap.add_argument('--budget-hours', type=float, default=11.0, help='session budget; Kaggle stops a session at 12 h')
    ap.add_argument('--fold-timeout-hours', type=float, default=3.0)
    a = ap.parse_args()
    t0 = time.time()
    on_kaggle = Path('/kaggle/working').exists()
    results = Path(a.results or ('/kaggle/working' if on_kaggle else '/content/results')).resolve(); results.mkdir(parents=True, exist_ok=True)
    workspace = Path(a.workspace or '/tmp/knee').resolve()
    folds = [int(x) for x in a.folds.split(',')] if a.folds else JOBS[a.job]['folds']
    budget = a.budget_hours * 3600
    status = dict(job=a.job, state='STAGING', started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), budget_seconds=budget, folds_requested=folds, folds={},
                  platform='kaggle' if on_kaggle else 'other', cpus=os.cpu_count())
    write_status(results, status)
    try:
        tools_dir = Path(a.tools).resolve() if a.tools else HERE.parent
        rr = load_run_remote(tools_dir); rr.TOOLS_DS = TOOLS_DATASET
        corpus = rr.stage(workspace, tools_dir, Path(a.corpus).resolve() if a.corpus else None, '')
        sys.path.insert(0, str(workspace))
        tool = importlib.import_module('tools.' + JOBS[a.job]['tool'])
        if Path(tool.ROOT).resolve() != workspace:
            raise RuntimeError(f'tool imported from {tool.ROOT}, expected {workspace}')
        manifest_path = workspace / tool.PROTOCOL / 'manifest.json'
        p = tool.read(manifest_path); manifest_sha = tool.sha256(manifest_path)
        env = {**os.environ, **tool.THREAD_ENV, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(workspace),
               'KNEE_REMOTE_RELAX_ENV': '1', 'HF_HUB_OFFLINE': '1'}
        status.update(protocol=tool.PROTOCOL, protocol_sha256=manifest_sha, corpus=str(Path(corpus).resolve()), gpu=rr.gpu_name(), workspace=str(workspace))
        status['software'] = software(sys.executable, env)
        write_status(results, status)
        status['weights'] = fetch_weights(sys.executable, p['common_args']['arch'], env)
        workers = max(0, min(p['resource_args']['workers']['default'], os.cpu_count() or 1))
        status.update(workers=workers, staging_seconds=round(time.time() - t0))
    except (Exception, SystemExit):
        status.update(state='FAILED_STAGING', error=traceback.format_exc()[-3000:]); write_status(results, status)
        print(json.dumps(status, indent=1, default=str), flush=True)
        return 0
    status['state'] = 'TRAINING'; write_status(results, status)
    (results / 'logs').mkdir(exist_ok=True)
    durations = []
    for k in folds:
        tag = tool.tag_for(k)
        ok, need = tool.should_start_fold(time.time() - t0, durations, budget)
        if not ok:
            status['folds'][str(k)] = dict(tag=tag, state='SKIPPED_TIME', needed_seconds=round(need), elapsed_seconds=round(time.time() - t0))
            write_status(results, status); continue
        argv = tool.runner_argv(p, k, 'full', root=workspace, python=sys.executable, workers=workers, output_root=results)
        rec = status['folds'][str(k)] = dict(tag=tag, state='RUNNING', started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), argv=argv)
        write_status(results, status)
        timeout = max(600.0, min(a.fold_timeout_hours * 3600, budget - (time.time() - t0)))
        fold_t0 = time.time()
        console = results / 'logs' / f'{tag}.console.log'
        try:
            with console.open('w', encoding='utf-8') as f:
                code = subprocess.run(argv, cwd=workspace, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = 'TIMEOUT'
        seconds = round(time.time() - fold_t0)
        run = results / tag
        problems = tool.verify_run(run, p, k, manifest_sha) if run.exists() else ['run directory missing']
        s = read_json(run / 'summary.json') or {}
        prov = s.get('provenance') or read_json(run / 'provenance.json') or {}
        rec.update(state='OK' if code == 0 and not problems else 'FAILED', exit_code=code, seconds=seconds, problems=problems,
                   optimizer_updates=s.get('optimizer_updates'), skipped_updates=s.get('skipped_updates'), runner_seconds=s.get('total_seconds'),
                   initial_state_matches_registered=prov.get('initial_state_matches_registered'),
                   initial_state_mismatched_keys=(prov.get('initial_state_mismatched_keys') or [])[:10], environment=prov.get('environment'),
                   epoch_seconds=[h.get('epoch_seconds') for h in (read_json(run / 'history.json') or [])])
        if rec['state'] == 'OK':
            durations.append(seconds)
        if run.exists():
            (run / 'remote_run.json').write_text(json.dumps(dict(job=a.job, tag=tag, fold=k, exit_code=code, seconds=seconds, gpu=status.get('gpu'),
                                                                 software=status.get('software'), weights=status.get('weights'), protocol_sha256=manifest_sha,
                                                                 problems=problems, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')), indent=1, default=str), encoding='utf-8')
        write_status(results, status)
        tail = console.read_text(encoding='utf-8', errors='replace')[-1500:] if console.exists() else ''
        print(f'fold {k} {rec["state"]} exit {code} in {seconds}s\n{tail}', flush=True)
    states = [v['state'] for v in status['folds'].values()]
    status['state'] = 'COMPLETE' if states and all(x == 'OK' for x in states) and len(states) == len(folds) else ('PARTIAL' if 'OK' in states else 'FAILED')
    status['total_seconds'] = round(time.time() - t0)
    write_status(results, status)
    print(json.dumps({k: status[k] for k in ('job', 'state', 'total_seconds')} | {'folds': {k: (v['state'], v.get('seconds')) for k, v in status['folds'].items()}}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
