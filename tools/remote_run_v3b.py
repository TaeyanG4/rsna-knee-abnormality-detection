"""Remote entry for bundle v3b (Kaggle T4): tools/remote_run_v3.py unchanged, plus the EXP101 ResNet34 seed-43 job.

  python tools/remote_run_v3b.py --job exp101:rn34-s43-5fold --tools DIR --corpus DIR [--folds 0,1,2,3,4] [--results DIR] [--budget-hours 11]

remote_run_v3.py only knows the seed-42 job. This entry loads it from the same directory by path, registers the seed-43 job (guard/tool
tools/exp101_kfold_rn34_s43.py, folds 0-4) and calls its main(): staging, software probe, one online download of the backbone weights,
the 1.3 x slowest-fold time guard, per-fold console logs, <results>/exp101-status.json and the always-exit-0 rule are those of
remote_run_v3.py. Output tags are EXP101_RN34_S43_DREAD_F<K>, so they never collide with the seed-42 run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = 'exp101:rn34-s43-5fold'
JOB_SPEC = dict(tool='exp101_kfold_rn34_s43', folds=[0, 1, 2, 3, 4])
TOOLS_DATASET = 'knee-runners-v3b'


def load_base():
    spec = importlib.util.spec_from_file_location('remote_run_v3', HERE / 'remote_run_v3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.JOBS[JOB] = dict(JOB_SPEC)
    module.TOOLS_DATASET = TOOLS_DATASET
    return module


if __name__ == '__main__':
    sys.exit(load_base().main())
