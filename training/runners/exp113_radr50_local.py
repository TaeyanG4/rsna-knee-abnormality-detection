"""EXP113 (H-B-34): one fold of EXP113_RADR50_LOCAL_PROTOCOL_V1 on the local GPU with the EXP107 runner code unchanged.

training/runners/exp107_kfold_radr50.py (bound by sha256 in both protocols) imports its argument guard as `tools.exp107_kfold_radr50`, whose
validate_arguments accepts only the EXP107 protocol path and the EXP107_RADR50_DREAD_F<K> tags. This entry points that module name at
tools/exp113_radr50_local.py (same helper functions, re-exported; EXP113 protocol path, EXP113_RADR50_LOCAL[_<LABELSET>]_F<K> tags, local
weights, strict environment and initial state) for this process only, then runs the unchanged runner's main(). The training loop is therefore
byte-identical to the Kaggle T4 job; provenance records runner_sha256 = the shared runner and guard_sha256 = the EXP113 tool.

Two things happen here before the shared runner is loaded:
  * `--label-set NAME` (default dread) is removed from argv and bound in the guard (tools/exp113_radr50_local.bind_label_set); the shared
    runner's parser never sees it. A non-dread name must be registered with `tools/exp113_radr50_local.py register-labels --name NAME`.
  * OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 are supplied when absent (before torch is imported), so a gpu_queue runner
    started without them does not fail the job (D-B-017); a different value is left alone and refused by the guard.
Arguments, modes (full / smoke / --validate-only) and outputs are otherwise exactly those of the shared runner.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED_RUNNER = ROOT / 'training' / 'runners' / 'exp107_kfold_radr50.py'
GUARD = 'exp113_radr50_local'
SHARED_GUARD = 'exp107_kfold_radr50'
REGISTERED_ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1'}
DEFAULT_LABEL_SET = 'dread'


def split_label_set(argv):
    """(label set, argv without --label-set). At most one --label-set NAME / --label-set=NAME."""
    name, rest, seen, i = DEFAULT_LABEL_SET, [], False, 0
    argv = list(argv)
    while i < len(argv):
        item = argv[i]
        if item == '--label-set' or item.startswith('--label-set='):
            if seen:
                raise SystemExit('--label-set given more than once')
            if item == '--label-set':
                if i + 1 >= len(argv) or argv[i + 1].startswith('--'):
                    raise SystemExit('--label-set needs a value')
                name, i = argv[i + 1], i + 2
            else:
                name, i = item.split('=', 1)[1], i + 1
            seen = True
            continue
        rest.append(item); i += 1
    return name, rest


def supply_registered_env(environ=None):
    environ = os.environ if environ is None else environ
    supplied = [k for k in REGISTERED_ENV if k not in environ]
    for k in supplied:
        environ[k] = REGISTERED_ENV[k]
    return supplied


def load_runner(label_set=DEFAULT_LABEL_SET):
    """The shared EXP107 runner module, with `from tools import exp107_kfold_radr50` resolving to the EXP113 guard bound to label_set."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tools
    from tools import exp113_radr50_local as guard
    if Path(guard.__file__).resolve() != (ROOT / 'tools' / (GUARD + '.py')).resolve():
        raise SystemExit(f'EXP113 guard imported from {guard.__file__}, expected {ROOT / "tools"}')
    guard.bind_label_set(label_set)
    sys.modules['tools.' + SHARED_GUARD] = guard
    setattr(tools, SHARED_GUARD, guard)
    spec = importlib.util.spec_from_file_location('exp107_kfold_radr50_shared_runner', SHARED_RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if Path(runner.ROOT).resolve() != ROOT.resolve():
        raise SystemExit('shared runner resolved a different workspace root')
    return runner


def build_parser():
    return load_runner().build_parser()


def main(argv=None):
    label_set, rest = split_label_set(sys.argv[1:] if argv is None else argv)
    supply_registered_env()
    return load_runner(label_set).main(rest)


if __name__ == '__main__':
    main()
