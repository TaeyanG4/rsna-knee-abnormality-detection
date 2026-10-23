"""CPU MIL screen CLI v1.1.0 (2026-10-11).

Changelog: add receipt-verified collected-run registration and holistic readout.
Historical replay and collected paired readout; no model admission or GPU operations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from training.arms import lifecycle as life
from training.arms import mil_screen_replay as replay
from training.arms import collected_screen as collected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('register-replay', 'register-collected', 'validate', 'screen'):
        p = commands.add_parser(name)
        p.add_argument('--root', required=True)
        if name == 'register-replay':
            p.add_argument('--kind', choices=replay.HISTORIES, required=True)
            p.add_argument('--historical-sha256', required=True)
        elif name == 'register-collected':
            p.add_argument('--request', required=True)
        else:
            p.add_argument('--spec', required=True)
        if name != 'validate':
            p.add_argument('--out', required=True)
    a = parser.parse_args(argv)
    try:
        if a.command == 'register-replay':
            spec = replay.register(a.root, a.kind, a.historical_sha256, a.out)
            status = 'REGISTERED_REPLAY'
        elif a.command == 'register-collected':
            spec = collected.register(a.root, collected.read(a.request), a.out)
            status = 'REGISTERED_COLLECTED_SCREEN'
        else:
            raw = Path(a.spec).read_bytes()
            spec = life.parse_json(raw)
            replay.need(raw == life.canonical_bytes(spec), 'canonical replay spec required')
            engine = collected if spec.get('schema') == collected.SCHEMA else replay
            if a.command == 'validate':
                engine.validate(a.root, spec)
                status = 'VALIDATED_COLLECTED_SCREEN' if engine is collected else 'VALIDATED_REPLAY'
            else:
                engine.screen(a.root, spec, a.out)
                status = 'SCREENED_B_DECISION_PENDING' if engine is collected else 'PARITY_VERIFIED'
        print(life.canonical_bytes(dict(status=status, spec_sha256=life.content_sha256(spec),
              scientific_status=('B_DECISION_PENDING' if spec.get('schema') == collected.SCHEMA
                                 else 'HISTORICAL_REPLAY_ONLY'), launch_authorized=False)).decode())
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(life.canonical_bytes(dict(status='REJECTED', error_type=type(error).__name__)).decode())
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
