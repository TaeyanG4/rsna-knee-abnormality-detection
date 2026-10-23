"""EXP118 (H-B-34): one fold of EXP118_RADR50_LOCAL_S43_PROTOCOL_V1 (EXP113 with seed 43) on the local GPU with the EXP107 runner code unchanged.

training/runners/exp107_kfold_radr50.py (bound by sha256 in EXP107, EXP113 and EXP118) imports its argument guard as
`tools.exp107_kfold_radr50`. This entry points that module name at tools/exp118_radr50_local_s43.py (the same helper functions, re-exported;
EXP118 protocol path, EXP118_RADR50_LOCAL_S43_F<K> tags, seed 43, local weights, strict environment and seed-43 initial state) for this
process only, then runs the unchanged runner's main(). The training loop is therefore byte-identical to EXP113's; provenance records
runner_sha256 = the shared runner and guard_sha256 = the EXP118 tool.

Reused from the EXP113 entry (training/runners/exp113_radr50_local.py, loaded by path, not modified):
  * supply_registered_env: OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 are supplied when absent (before torch is imported),
    so a gpu_queue runner started without them does not fail the job (D-B-017); a different value is left alone and refused by the guard;
  * split_label_set: an optional `--label-set` is removed from argv; EXP118 is the dread control, so any name other than dread is refused.
Arguments, modes (full / smoke / --validate-only) and outputs are otherwise exactly those of the shared runner.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED_RUNNER = ROOT / 'training' / 'runners' / 'exp107_kfold_radr50.py'
EXP113_ENTRY = ROOT / 'training' / 'runners' / 'exp113_radr50_local.py'
GUARD = 'exp118_radr50_local_s43'
SHARED_GUARD = 'exp107_kfold_radr50'
LABEL_SET = 'dread'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_E113 = _load('exp113_runner_entry_for_exp118', EXP113_ENTRY)
REGISTERED_ENV = _E113.REGISTERED_ENV
supply_registered_env = _E113.supply_registered_env


def strip_label_set(argv):
    """argv without an optional `--label-set dread`; any other label set is refused (EXP118 is the dread seed-43 control)."""
    name, rest = _E113.split_label_set(argv)
    if name != LABEL_SET:
        raise SystemExit(f'EXP118 runs the dread label set only (got --label-set {name})')
    return rest


def load_runner():
    """The shared EXP107 runner module, with `from tools import exp107_kfold_radr50` resolving to the EXP118 guard."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tools
    from tools import exp118_radr50_local_s43 as guard
    if Path(guard.__file__).resolve() != (ROOT / 'tools' / (GUARD + '.py')).resolve():
        raise SystemExit(f'EXP118 guard imported from {guard.__file__}, expected {ROOT / "tools"}')
    sys.modules['tools.' + SHARED_GUARD] = guard
    setattr(tools, SHARED_GUARD, guard)
    runner = _load('exp107_kfold_radr50_shared_runner', SHARED_RUNNER)
    if Path(runner.ROOT).resolve() != ROOT.resolve():
        raise SystemExit('shared runner resolved a different workspace root')
    return runner


def build_parser():
    return load_runner().build_parser()


def main(argv=None):
    rest = strip_label_set(sys.argv[1:] if argv is None else argv)
    supply_registered_env()
    return load_runner().main(rest)


if __name__ == '__main__':
    main()
