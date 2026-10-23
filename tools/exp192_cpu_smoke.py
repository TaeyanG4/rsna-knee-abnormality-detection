"""EXP192 CPU smoke frontend v2.0.0.

Changelog: use registered real-data lifecycle smoke; remove global patches and
synthetic legacy training. Requires the same spec/stage as every other preset.
"""
from __future__ import annotations
import sys

def main(argv=None):
    from tools import coatnet_config
    return coatnet_config.main(['smoke', *(sys.argv[1:] if argv is None else argv)],
                               preset='crop-f0-s42')

if __name__ == '__main__':
    sys.exit(main())
