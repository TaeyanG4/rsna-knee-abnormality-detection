"""CoAtNet CPU runner v1.0.0. Changelog: explicit contract inspection and smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.arms import coatnet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    contract = sub.add_parser('inspect-control')
    contract.add_argument('--root', type=Path, required=True)
    smoke = sub.add_parser('cpu-smoke')
    smoke.add_argument('--root', type=Path, required=True)
    smoke.add_argument('--variant', choices=coatnet.PRESETS, required=True)
    smoke.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect-control':
            result = coatnet.load_control(args.root).report()
        else:
            result = coatnet.cpu_smoke(args.root, args.variant, args.out)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        print(json.dumps(dict(status='REJECTED', error=str(exc), launch_authorized=False)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
