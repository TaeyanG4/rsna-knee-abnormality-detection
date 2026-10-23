"""EXP116 (H-B-46) v1.1.1: one fold of EXP116_TEACHER_FILL_PROTOCOL_V1 on the local GPU with the EXP107 runner code unchanged (as EXP113).

training/runners/exp107_kfold_radr50.py (bound by sha256 in EXP107, EXP113 and EXP116) imports its argument guard as
`tools.exp107_kfold_radr50`. This entry points that module name at tools/exp116_teacher_fill.py for this process only, then runs the unchanged
runner's main(). The EXP116 guard re-exports the EXP107 helpers (sampling, hashing, contract, membership) unchanged and differs in three
places: validate_arguments accepts only the EXP116 protocol and EXP116_TEACHER_FILL_F<K>[_SMOKE] tags; the protocol view of fold K names
that fold's teacher-fill label file as --labels / --eval-labels; and pos_weight(labels, ids) returns the EXP099 rule applied to the ORIGINAL
Dread labels, so the runner's equality check against the manifest fold vector (= EXP113's Dread vector) binds the fixed Dread loss weighting.
The training loop, loss, optimizer, scheduler, FP16 gate, evaluation and outputs are therefore byte-identical to EXP113; provenance records
runner_sha256 = the shared runner and guard_sha256 = the EXP116 tool.

Quarantine (v1.1.1, D-B-024): the student OOF is LEAKY by teacher ancestry, so every fold run directory carries DIAGNOSTIC_ONLY.json
(tools/diagnostic_quarantine.py, decision_eligible false) from before training. The shared runner refuses a non-empty run directory, so the
marker cannot be placed in advance: this entry gives the loaded runner module its own `open` (a module global; the runner file is unchanged)
that writes the marker the moment the runner opens <run>/train.log, i.e. right after it created the empty run directory and before the model
is built or any OOF CSV is written. After a completed run the marker is annotated with the sha256 of oof_best.csv / oof_k36.csv. The
summary.json caveat the shared runner writes ("for ensembling / teacher pseudo-labels") does not apply to EXP116; the marker overrides it.
A run that reaches training without the marker is a failure (exit), never a silent success.

OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and HF_HUB_OFFLINE=1 are supplied when absent (before torch is imported), as in EXP113 (D-B-017).
Arguments, modes (full / smoke / --validate-only) and outputs are exactly those of the shared runner (plus the marker).
"""
from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED_RUNNER = ROOT / 'training' / 'runners' / 'exp107_kfold_radr50.py'
GUARD = 'exp116_teacher_fill'
SHARED_GUARD = 'exp107_kfold_radr50'
RUN_LOG = 'train.log'
REGISTERED_ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1'}


def supply_registered_env(environ=None):
    environ = os.environ if environ is None else environ
    supplied = [k for k in REGISTERED_ENV if k not in environ]
    for k in supplied:
        environ[k] = REGISTERED_ENV[k]
    return supplied


def load_runner():
    """The shared EXP107 runner module, with `from tools import exp107_kfold_radr50` resolving to the EXP116 guard."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tools
    from tools import exp116_teacher_fill as guard
    if Path(guard.__file__).resolve() != (ROOT / 'tools' / (GUARD + '.py')).resolve():
        raise SystemExit(f'EXP116 guard imported from {guard.__file__}, expected {ROOT / "tools"}')
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


def install_marker_hook(runner, args, guard):
    """Give `runner` (the loaded shared-runner module) a module-global `open` that writes DIAGNOSTIC_ONLY.json into <run> when the runner
    opens <run>/train.log for appending, <run> being the directory named after args.tag. Returns the state dict ('run', 'marker')."""
    state = {}
    real_open = builtins.open

    def marking_open(file, mode='r', *a, **kw):
        if 'marker' not in state and isinstance(file, (str, os.PathLike)) and 'a' in str(mode):
            p = Path(file)
            if p.name == RUN_LOG and p.parent.name == args.tag:
                state['run'] = p.parent
                state['marker'] = guard.dq.write_marker(p.parent, guard.run_marker(args.tag, args.fold, args.mode, args.protocol))
        return real_open(file, mode, *a, **kw)

    runner.open = marking_open
    return state


def main(argv=None):
    supply_registered_env()
    argv = sys.argv[1:] if argv is None else list(argv)
    runner = load_runner()
    args = runner.build_parser().parse_args(argv)
    from tools import exp116_teacher_fill as guard
    state = install_marker_hook(runner, args, guard)
    result = runner.main(argv)
    if not args.validate_only:
        if 'marker' not in state:
            raise SystemExit(f'EXP116: the run finished without {guard.dq.MARKER} in its directory; treat every output as unquarantined and discard it')
        guard.annotate_run_marker(state['run'])
    return result


if __name__ == '__main__':
    main()
