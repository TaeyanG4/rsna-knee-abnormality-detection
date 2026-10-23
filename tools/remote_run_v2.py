"""Remote entry for bundle v2: stage workspace, run one registered EXP094/EXP095 job, pack results.

  python tools/remote_run_v2.py --job exp094:seed43 [--tools DIR] [--corpus DIR] [--no-upload] [--kaggle-user U]

Runs on Colab (/content) or Kaggle (/kaggle/working). Requires a bf16-capable GPU (L4/A100/Ada) for the EXP094/EXP095
runners. Results: <results>/<tag>/ with summary.json, provenance.json, history.json, oof_best.csv|oof_fold.csv, model_best.pt,
train.log, remote_run.json; uploaded as private dataset knee-raptor-res-<tag> unless --no-upload.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = {'exp094:seed43': ('exp094_b8_variant', 'seed43'), 'exp094:geoaug': ('exp094_b8_variant', 'geoaug'), 'exp094:strat': ('exp094_b8_variant', 'strat'), 'exp094:slotdrop': ('exp094_b8_variant', 'slotdrop'),
        'exp095:teacher0': ('exp095_teacher_mix', 'teacher0'), 'exp095:teacher1': ('exp095_teacher_mix', 'teacher1'), 'exp095:teacher2': ('exp095_teacher_mix', 'teacher2'), 'exp095:student': ('exp095_teacher_mix', 'student')}
RESULT_FILES = ('summary.json', 'history.json', 'provenance.json', 'train.log', 'oof_best.csv', 'oof_fold.csv', 'model_best.pt', 'pilot-receipt.json', 'output-inventory.json')


def load_run_remote(tools_dir):
    spec = importlib.util.spec_from_file_location('run_remote', tools_dir / 'tools' / 'colab' / 'run_remote.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job', choices=list(JOBS), required=True); ap.add_argument('--tools'); ap.add_argument('--corpus'); ap.add_argument('--workspace'); ap.add_argument('--results')
    ap.add_argument('--mode', choices=['pilot', 'full'], default='full'); ap.add_argument('--no-upload', action='store_true'); ap.add_argument('--kaggle-user', default=''); ap.add_argument('--no-install', action='store_true')
    a = ap.parse_args()
    on_kaggle = Path('/kaggle/working').exists()
    workspace = Path(a.workspace or ('/tmp/knee' if on_kaggle else '/content/knee')).resolve()
    results_root = Path(a.results or ('/kaggle/working/results' if on_kaggle else '/content/results')).resolve()
    tools_dir = Path(a.tools).resolve() if a.tools else HERE.parent
    rr = load_run_remote(tools_dir)
    user = a.kaggle_user or (rr.kaggle_user() if rr.setup_token() else '')
    if not a.no_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm>=1.0.19', 'pandas', 'numpy', 'psutil', 'opencv-python-headless', 'pydicom'], check=False)
    rr.TOOLS_DS = 'knee-runners-v2'
    corpus = rr.stage(workspace, tools_dir, Path(a.corpus).resolve() if a.corpus else None, user)
    sys.path.insert(0, str(workspace))
    import importlib
    tool = importlib.import_module('tools.' + JOBS[a.job][0])
    assert Path(tool.ROOT).resolve() == workspace, (tool.ROOT, workspace)
    p = tool.read(workspace / tool.PROTOCOL / 'manifest.json')
    if JOBS[a.job][0] == 'exp094_b8_variant':
        argv = tool.runner_argv(p, JOBS[a.job][1], a.mode); tag = p['arms'][JOBS[a.job][1]]['tags'][a.mode]
    else:
        argv = tool.runner_argv(p, JOBS[a.job][1]); tag = tool.JOBS[JOBS[a.job][1]][2]
    argv = [str(workspace / 'storage/cache/derived/RAPTOR64_336_C140_S02') if x == str(workspace / 'storage/cache/derived/RAPTOR64_336_C140_S02') else x for x in argv]
    if Path(corpus).resolve() != (workspace / 'storage/cache/derived/RAPTOR64_336_C140_S02').resolve():
        argv = [str(Path(corpus).resolve()) if x.endswith('RAPTOR64_336_C140_S02') else x for x in argv]
    env = {**os.environ, **tool.THREAD_ENV, 'HF_HUB_OFFLINE': '0', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'KNEE_WORKSPACE_ROOT': str(workspace), 'KNEE_REMOTE_RELAX_ENV': '1'}
    env['HF_HUB_OFFLINE'] = '1' if os.environ.get('KNEE_HF_OFFLINE') else env['HF_HUB_OFFLINE']
    # the guards require HF_HUB_OFFLINE=1; pre-download the backbone weights first, then run offline
    arch = p['common_args']['arch'] if 'common_args' in p else (p['teacher_args']['arch'] if 'teacher' in a.job else p['student_args']['arch'])
    subprocess.run([sys.executable, '-c', f"import timm; timm.create_model({arch!r}, pretrained=True)"], check=True, env={**env, 'HF_HUB_OFFLINE': '0'})
    env['HF_HUB_OFFLINE'] = '1'
    print('$', ' '.join(argv), flush=True); t0 = time.time()
    proc = subprocess.run(argv, cwd=workspace, env=env)
    exp = workspace / 'artifacts' / 'experiments' / tag; results = results_root / tag; results.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES:
        if (exp / name).exists():
            shutil.copy2(exp / name, results / name)
    (results / 'remote_run.json').write_text(json.dumps({'job': a.job, 'tag': tag, 'argv': argv, 'seconds': round(time.time() - t0), 'exit_code': proc.returncode, 'gpu': rr.gpu_name(),
                                                         'platform': 'kaggle' if on_kaggle else 'colab', 'protocol_sha256': tool.sha256(workspace / tool.PROTOCOL / 'manifest.json'),
                                                         'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}, indent=1), encoding='utf-8')
    print('results in', results, sorted(x.name for x in results.iterdir()), flush=True)
    if proc.returncode:
        raise SystemExit(f'job failed with exit code {proc.returncode}')
    if not a.no_upload:
        (results / 'summary.json').exists() or (results / 'summary.json').write_text('{}')
        rr.upload(results, user, tag)


if __name__ == '__main__':
    main()
