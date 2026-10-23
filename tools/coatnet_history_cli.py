"""CoAtNet historical CLI v1.0.0.

Changelog: explicit preset dispatch and read-only authenticated historical
records; collection/screens use fresh outputs and confer no launch authority.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv, *, preset, protocol, candidate):
    from tools import coatnet_config
    from training.arms import coatnet_history as history, coatnet_screen as screen
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == 'lifecycle':
        return coatnet_config.main(args[1:], preset=preset)
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('historical-validate')
    for command in ('collect', 'screen'):
        sub = subs.add_parser(command)
        sub.add_argument('--out', required=True)
        sub.add_argument('--verification-note', required=True)
        if command == 'screen':
            sub.add_argument('--collection', required=True)
            sub.add_argument('--readout')
    parsed = parser.parse_args(args)
    try:
        cand = history.verified_arm(protocol)
        if parsed.command == 'historical-validate':
            result = dict(status='HISTORICAL_ARTIFACTS_VERIFIED', launch_authorized=False,
                          historical_only=True, protocol_sha256=cand['protocol_sha256'])
        else:
            ctrl = history.verified_arm(history.CONTROL)
            kw = dict(candidate=candidate, out=parsed.out, verification_note=parsed.verification_note)
            if parsed.command == 'collect':
                result = screen.collect_records(ctrl, cand, **kw)
            else:
                result = screen.screen_records(ctrl, cand, collection=parsed.collection,
                                               readout=parsed.readout, **kw)
        print(json.dumps(dict(status=result['status'], launch_authorized=False)))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps(dict(status='REJECTED', launch_authorized=False, reason=str(error))))
        return 2
