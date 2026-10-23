"""EXP139 POS launch shim v1.0.0 (B, 2026-10-08 22:29 KST).

Why: on CUDA the EXP139 runner builds its training DataLoader with pin_memory=True, and
torch's pin_memory turns the POS batch input tuple (images, positions) into a LIST;
raptor_exp129.move_input only recurses into tuples, so POS s42 failed at its first step
(AttributeError: 'list' object has no attribute 'to'). CPU validation never pins memory,
so it could not see this. The frozen EXP139 sources stay byte-identical (their sha256 are
bound in the POS/LATE/CTX manifests); this shim only converts a list input back to the
tuple it was before pinning, then runs the unchanged runner as __main__. No numerical change.

Usage: python training/runners/exp139_pos_pinfix.py <the runner's own arguments>
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'  # same values the runner registers, before numpy/torch import
os.environ['HF_HUB_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'training' / 'runners' / 'exp139_aug1_variants.py'
sys.path.insert(0, str(ROOT / 'training' / 'src'))

import raptor_exp129  # noqa: E402
import raptor_exp139  # noqa: E402

_original = raptor_exp129.move_input


def move_input(x, device, non_blocking=False):
    if isinstance(x, list):
        x = tuple(x)
    return _original(x, device, non_blocking)


def install():
    raptor_exp129.move_input = move_input  # recursion inside the original resolves to this name
    raptor_exp139.move_input = move_input  # the runner imports move_input from raptor_exp139


if __name__ == '__main__':
    install()
    print('exp139_pos_pinfix v1.0.0: list->tuple input shim installed', flush=True)
    sys.argv[0] = str(RUNNER)
    runpy.run_path(str(RUNNER), run_name='__main__')
