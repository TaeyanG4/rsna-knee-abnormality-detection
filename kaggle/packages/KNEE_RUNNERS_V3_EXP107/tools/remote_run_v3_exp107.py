"""Remote entry for the EXP107 bundle (Kaggle T4): tools/remote_run_v3.py unchanged, plus the RadImageNet ResNet50 5-fold job.

  python tools/remote_run_v3_exp107.py --job exp107:radr50-5fold --tools DIR --corpus DIR [--folds 0,1,2,3,4] [--results DIR] [--budget-hours 11]

remote_run_v3.py stages the workspace, probes the software, trains the folds one after another under the 1.3 x slowest-fold time guard,
writes per-fold console logs and <results>/exp101-status.json, and always exits 0. This entry registers the EXP107 job (guard/tool
tools/exp107_kfold_radr50.py) and replaces only its backbone-weight step: instead of downloading timm weights it locates ResNet50.pt under
/kaggle/input (the attached public dataset marwanmath/resnet-50-radimagenet-marwan) by the protocol's sha256, proves in a subprocess that it
loads strictly into the model, and exports its path in KNEE_RADIMAGENET_R50 so that every fold's runner argv passes it as --init-weights
(the runner re-checks the sha256). A missing or different file ends the session as FAILED_STAGING before any training.
"""
from __future__ import annotations

import glob
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = 'exp107:radr50-5fold'
JOB_SPEC = dict(tool='exp107_kfold_radr50', folds=[0, 1, 2, 3, 4])
TOOLS_DATASET = 'knee-runners-v3-exp107'
WEIGHTS_NAME = 'ResNet50.pt'
SEARCH_ROOTS = ('/kaggle/input',)
LOAD_PROBE = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from raptor_radimagenet import build_radimagenet_model
model, report = build_radimagenet_model("resnet50", sys.argv[2], sys.argv[3])
print(json.dumps(report))
'''


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def locate_weights(expected_sha256, roots=SEARCH_ROOTS, name=WEIGHTS_NAME):
    """(path, {candidate: sha256}) of the first file called `name` under `roots` whose sha256 equals the registered one."""
    hits = sorted({p for r in roots for p in glob.glob(os.path.join(r, '**', name), recursive=True)})
    hashes = {p: file_sha256(p) for p in hits}
    good = [p for p, h in hashes.items() if h == expected_sha256]
    if not good:
        raise RuntimeError(f'no {name} with sha256 {expected_sha256} under {list(roots)}; candidates {hashes}')
    return good[0], hashes


def make_fetch_weights(tool_name=JOB_SPEC['tool'], roots=SEARCH_ROOTS):
    def fetch_weights(python, arch, env):
        tool = importlib.import_module('tools.' + tool_name)  # the staged workspace copy (already imported by remote_run_v3.main)
        p = tool._protocol()
        expected = p['init_weights']['sha256']
        if arch != p['common_args']['arch'] or expected != tool.RADIMAGENET_SHA256:
            raise RuntimeError('protocol arch / weights differ from the EXP107 guard')
        t0 = time.time()
        path, candidates = locate_weights(expected, roots)
        os.environ[tool.WEIGHTS_ENV] = path  # read by tool.runner_argv for every fold (--init-weights)
        r = subprocess.run([python, '-c', LOAD_PROBE, str(Path(tool.ROOT) / 'training' / 'src'), path, expected],
                           env={**env, tool.WEIGHTS_ENV: path}, capture_output=True, text=True, timeout=600, cwd=str(tool.ROOT))
        if r.returncode:
            raise RuntimeError(f'RadImageNet strict-load probe failed: {(r.stderr or r.stdout)[-1500:]}')
        report = json.loads(r.stdout.strip().splitlines()[-1])
        return dict(arch=arch, source='RadImageNet ResNet50 (' + p['init_weights']['kaggle_dataset'] + ')', path=path, sha256=expected,
                    candidates=candidates, load=report, seconds=round(time.time() - t0))
    return fetch_weights


def load_base():
    spec = importlib.util.spec_from_file_location('remote_run_v3', HERE / 'remote_run_v3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.JOBS[JOB] = dict(JOB_SPEC)
    module.TOOLS_DATASET = TOOLS_DATASET
    module.fetch_weights = make_fetch_weights()
    return module


if __name__ == '__main__':
    sys.exit(load_base().main())
