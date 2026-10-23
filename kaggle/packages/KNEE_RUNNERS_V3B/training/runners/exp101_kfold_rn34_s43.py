"""EXP101 seed 43: one fold of EXP101_KFOLD_RN34_S43_PROTOCOL_V1 with the seed-42 runner code unchanged.

training/runners/exp101_kfold_arch.py (bound by sha256 in both protocols) imports its argument guard as `tools.exp101_kfold_arch`, whose
validate_arguments accepts only the seed-42 protocol path and the EXP101_RN34_DREAD_F<K> tags. This entry points that module name at
tools/exp101_kfold_rn34_s43.py (same helper functions, re-exported; seed-43 protocol path, seed 43 and EXP101_RN34_S43_DREAD_F<K> tags) for this
process only, then runs the unchanged runner's main(). The training loop is therefore byte-identical to the seed-42 job; provenance records
runner_sha256 = the shared runner and guard_sha256 = the seed-43 guard, and the protocol binds this file, the shared runner and both tools.
Arguments, modes (full / smoke / --validate-only) and outputs are exactly those of the shared runner.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED_RUNNER = ROOT / 'training' / 'runners' / 'exp101_kfold_arch.py'
GUARD = 'exp101_kfold_rn34_s43'


def load_runner():
    """The shared EXP101 runner module, with `from tools import exp101_kfold_arch` resolving to the seed-43 guard."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tools
    from tools import exp101_kfold_rn34_s43 as guard
    if Path(guard.__file__).resolve() != (ROOT / 'tools' / (GUARD + '.py')).resolve():
        raise SystemExit(f'seed-43 guard imported from {guard.__file__}, expected {ROOT / "tools"}')
    sys.modules['tools.exp101_kfold_arch'] = guard
    tools.exp101_kfold_arch = guard
    spec = importlib.util.spec_from_file_location('exp101_kfold_arch_shared_runner', SHARED_RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if Path(runner.ROOT).resolve() != ROOT.resolve():
        raise SystemExit('shared runner resolved a different workspace root')
    return runner


def build_parser():
    return load_runner().build_parser()


def main(argv=None):
    return load_runner().main(argv)


if __name__ == '__main__':
    main()
