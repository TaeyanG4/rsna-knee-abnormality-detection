"""EXP125 entry: select its guard, execute the byte-identical EXP107 runner.

No training-loop copy. The existing EXP113/118 entries bind their own guards;
this new entry supplies the EXP125 guard and removes the production opt-in.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED_RUNNER = ROOT / 'training/runners/exp107_kfold_radr50.py'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_E113 = _load('exp113_entry_for_exp125', ROOT / 'training/runners/exp113_radr50_local.py')
supply_registered_env = _E113.supply_registered_env


def split_options(argv):
    name, rest = _E113.split_label_set(argv)
    if name != 'dread':
        raise SystemExit('EXP125 uses dread only')
    flag = '--allow-production-after-screen-pass'
    if rest.count(flag) > 1:
        raise SystemExit('duplicate production flag')
    return flag in rest, [a for a in rest if a != flag]


def load_runner(allow_production=False):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tools
    from tools import exp125_radr50_k16_local as guard
    guard.ALLOW_PRODUCTION = allow_production
    sys.modules['tools.exp107_kfold_radr50'] = guard
    tools.exp107_kfold_radr50 = guard
    runner = _load('exp107_shared_runner_for_exp125', SHARED_RUNNER)
    if Path(runner.ROOT).resolve() != ROOT.resolve():
        raise SystemExit('shared runner resolved a different workspace')
    return runner


def main(argv=None):
    allow, rest = split_options(sys.argv[1:] if argv is None else argv)
    supply_registered_env()
    return load_runner(allow).main(rest)


if __name__ == '__main__':
    main()
