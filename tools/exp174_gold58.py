"""EXP174 Gold58 gross-failure guard input (final-b-rule-v1 item 2). Version 1.0.2.

Changelog v1.0.2: validate both AUCs before reference or guard comparisons.

Claude opus for B, 2026-10-10. Runs the frozen EXP143 v2 ``predict-gold58`` and
``report`` code paths against the EXP174 registration, so EXP174 is scored
exactly like EXP143 v2's registered 0.9350 (model_best.pt, same corpus, fp16).
tools/exp143_v2.py and tools/exp174_own0949_jev.py are not edited: the two
registration functions are swapped in private function globals only.

  check    CPU: registration, completed fit, checkpoint identity, Gold58 membership.
  predict  GPU (B-claimed lane, 58 studies): S42/gold58.csv + receipt.
  report   CPU: S42/gold58-report.json (key ``own143`` = EXP174 standalone).
  guard    CPU: EXP174 macro12 >= EXP143 v2 macro12 - 0.02 -> gold58-guard.json.
No labels are printed; outputs are aggregate AUCs only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / 'training/src'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from tools import exp174_own0949_jev as e174

VERSION = '1.0.2'
base = e174.base
EXP143_REPORT = ROOT / 'artifacts/experiments/EXP143_OWN0949/S42v2/gold58-report.json'
EXP143_REGISTERED = 0.9350  # D-B-087
MARGIN = 0.02
GUARD_OUT = e174.BASE / 'gold58-guard.json'


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _bound(fn):
    """Frozen EXP143 v2 function with the EXP174 registration in private globals."""
    scope = {**fn.__globals__, 'load_registration': e174.load_registration,
             'registered_run': e174.registered_run}
    return types.FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__)


def _args(command, extra):
    return base.parser().parse_args([command, '--protocol', str(e174.PROTOCOL), *extra])


def check(competition=ROOT / 'storage/competition', checkpoint=True):
    p = e174.require_completed(e174.PROTOCOL)
    run = e174.registered_run(p)
    summary = base.read_json(run / 'summary.json')
    _need(summary.get('model_sha256') == base.sha256(run / 'model_best.pt'), 'model_best.pt differs from summary')
    _, ids, _labels = base.official_population(competition)
    _need(len(ids) == 58 and set(ids) == set(p['excluded_gold_ids']) and not set(ids) & set(p['train_ids']),
          'Gold58 membership/leakage check failed')
    reference = base.read_json(EXP143_REPORT)
    exp143 = reference['results']['own143']['macro12']
    _need(reference.get('n') == 58 and abs(exp143 - EXP143_REGISTERED) < 5e-4, 'EXP143 v2 Gold58 reference differs')
    result = dict(status='CPU_CHECK_PASS', version=VERSION, gold_studies=len(ids), train_studies=len(p['train_ids']),
                  model_sha256=summary['model_sha256'], exp143_v2_macro12=exp143,
                  threshold=exp143 - MARGIN, outputs_exist=(run / 'gold58.csv').exists())
    if checkpoint:  # CPU deserialization of our own registered checkpoint, as predict does
        import torch
        ck = torch.load(run / 'model_best.pt', map_location='cpu', weights_only=False)
        _bound(base.check_checkpoint)(ck, p, str(e174.PROTOCOL), 'local')
        result['checkpoint_identity'] = 'PASS'
    return result


def guard(report=None, out=GUARD_OUT):
    p = e174.load_registration(e174.PROTOCOL)
    run = e174.registered_run(p)
    report = Path(report or run / 'gold58-report.json')
    own = base.read_json(report)
    _need(own.get('status') == 'REPORT_ONLY_NO_SELECTION' and own.get('n') == 58 and
          own.get('protocol_sha256') == base.sha256(e174.PROTOCOL) and
          own.get('run_dir') == e174.RUN_DIRS['local'], 'EXP174 Gold58 report identity differs')
    reference = base.read_json(EXP143_REPORT)
    exp174 = own['results']['own143']['macro12']  # EXP143 report key; holds the EXP174 standalone here
    exp143 = reference['results']['own143']['macro12']
    for name, value in (('EXP174', exp174), ('EXP143 v2', exp143)):
        _need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
              and 0.0 <= value <= 1.0, f'{name} Gold58 macro12 is not a valid AUC')
    _need(abs(exp143 - EXP143_REGISTERED) < 5e-4, 'EXP143 v2 Gold58 reference differs')
    result = dict(schema='exp174-gold58-guard-v1', version=VERSION, rule='artifacts/selection/final-b-rule-v1.md item 2',
                  status='PASS' if exp174 >= exp143 - MARGIN else 'FAIL', exp174_macro12=exp174,
                  exp143_v2_macro12=exp143, margin=MARGIN, threshold=exp143 - MARGIN,
                  exp174_report_sha256=base.sha256(report), exp143_report_sha256=base.sha256(EXP143_REPORT),
                  caveat='Gross-failure guard only; Gold58 is exposed and small (58 studies)')
    e174.write_new_json(out, result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('--competition', default=str(ROOT / 'storage/competition'))
    c.add_argument('--no-checkpoint', action='store_true')
    for name in ('predict', 'report'):
        sub.add_parser(name, help='other options go to the frozen EXP143 v2 parser')
    g = sub.add_parser('guard')
    g.add_argument('--report')
    g.add_argument('--out', default=str(GUARD_OUT))
    args, rest = ap.parse_known_args(argv)
    _need(not rest or args.command in ('predict', 'report'), 'Unknown arguments: ' + ' '.join(rest))
    if args.command == 'check':
        result = check(args.competition, not args.no_checkpoint)
    elif args.command == 'predict':
        return _bound(base.predict)(_args('predict-gold58', rest), gold=True)
    elif args.command == 'report':
        return _bound(base.report)(_args('report', rest))
    else:
        result = guard(args.report, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
