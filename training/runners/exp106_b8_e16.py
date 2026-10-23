"""EXP106: the B8 CoAtNet recipe on EXP099 fold 0 for 16 epochs (EXP106_B8_E16_PROTOCOL_V1) with the shared EXP101 runner code unchanged.

training/runners/exp101_kfold_arch.py (bound by sha256 in the EXP101 protocols and in EXP106) imports its argument guard as
`tools.exp101_kfold_arch`, whose validate_arguments accepts only the EXP101 ResNet34 protocol path and tags. This entry points that module name
at tools/exp106_b8_e16.py (same helper functions, re-exported; EXP106 protocol path, seed 42, the B8 recipe with epochs 16, tag
EXP106_B8_E16_F0) for this process only, then runs the unchanged runner's main(). The training loop is therefore the EXP099 / EXP094 V4 loop;
provenance records runner_sha256 = the shared runner and guard_sha256 = tools/exp106_b8_e16.py, and the protocol binds this file, the shared
runner and both tools. Arguments, modes (full / smoke / --validate-only) and outputs are exactly those of the shared runner.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED_RUNNER = ROOT / 'training' / 'runners' / 'exp101_kfold_arch.py'
GUARD = 'exp106_b8_e16'


def load_runner():
    """The shared EXP101 runner module, with `from tools import exp101_kfold_arch` resolving to the EXP106 guard."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tools
    from tools import exp106_b8_e16 as guard
    if Path(guard.__file__).resolve() != (ROOT / 'tools' / (GUARD + '.py')).resolve():
        raise SystemExit(f'EXP106 guard imported from {guard.__file__}, expected {ROOT / "tools"}')
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
