"""Compare EXP128 Public3 CSV to the public notebook's own output CSV, exactly."""
from __future__ import annotations

import argparse
import json
from exp128_oai_light import compare_csv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ours', required=True)
    p.add_argument('--public', required=True)
    a = p.parse_args(argv)
    result = compare_csv(a.ours, a.public)
    print(json.dumps(result, indent=2))
    return 0 if result['studies'] == 3 and result['exact'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
