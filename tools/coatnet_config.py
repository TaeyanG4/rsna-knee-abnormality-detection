"""CoAtNet preset frontend v1.0.0 (2026-10-12).

Changelog: explicit presets dispatch to the registered lifecycle without changing
its source or scientific recipe. Historical arm names cannot override presets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

VERSION = '1.0.0'
PRESETS = {**{f'{v}-f0-s42': (v, 'fold0', 42)
              for v in ('control', 'coverage', 'crop', 'aug1')},
           'control-full-s43': ('control', 'full', 43)}


def dispatch(preset, argv):
    from training.runners import coatnet_lifecycle as runner
    from training.arms import coatnet_lifecycle as contract
    if preset not in PRESETS:
        raise ValueError('unknown CoAtNet preset')
    args = list(argv)
    if not args:
        raise ValueError('lifecycle command required')
    variant, population, seed = PRESETS[preset]
    fixed = ('--variant', '--population', '--seed')
    if any(a.split('=', 1)[0] in fixed for a in args):
        raise ValueError('preset fields cannot be overridden')
    # Use the authoritative parser, including its defaults, to identify the spec.
    if args[0] == 'register':
        args += ['--variant', variant, '--population', population, '--seed', str(seed)]
    parsed = runner.parser().parse_args(args)
    if getattr(parsed, 'spec', None):
        spec = contract.Spec(Path(parsed.spec).read_bytes()).document
        actual = (spec['config']['variant'], spec['config']['population'], spec['seed'])
        if actual != PRESETS[preset]:
            raise ValueError('registered spec differs from selected preset')
    return runner.main(args)


def main(argv=None, *, preset=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if preset is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('preset', choices=PRESETS)
        parser.add_argument('arguments', nargs=argparse.REMAINDER)
        parsed = parser.parse_args(args)
        preset, args = parsed.preset, parsed.arguments
    try:
        return dispatch(preset, args)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps(dict(status='REJECTED', launch_authorized=False, reason=str(error))))
        return 2


if __name__ == '__main__':
    sys.exit(main())
