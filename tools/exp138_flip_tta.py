"""EXP124 plane-aware flip inference on EXP138 checkpoints (one seed's five folds)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import exp124_flip_tta as flip
from tools import exp138_bundle_aug1_mh as bundle


def tta_dir(seed, root=bundle.ROOT):
    return Path(root) / bundle.PROTOCOL / 'tta' / f's{seed}' / 'exp117'


def run(seed, device='cuda', workers=2):
    p = bundle.load_protocol(seed)
    original = flip.model_config
    def config(name):
        if name != 'exp117':
            raise ValueError('EXP138 TTA is multi-head only')
        return bundle, bundle.PROTOCOL + f'/s{seed}', f'EXP138_BUNDLE_AUG1_MH_S{seed}_F', bundle.PROTOCOL
    flip.model_config = config
    try:
        return flip.run('exp117', device=device, output_root=tta_dir(seed).parent, workers=workers)
    finally:
        flip.model_config = original


def queue_entries(after, seed, workers=2):
    bundle.load_protocol(seed)
    tag = f'EXP138_BUNDLE_AUG1_MH_S{seed}_TTA'
    cmd = [bundle.aug.train_python(), str(bundle.ROOT / 'tools/exp138_flip_tta.py'),
           'run', '--seed', str(seed), '--workers', str(workers)]
    item = {'tag': tag, 'command': cmd, 'continue_on_failure': False}
    if after.lower() != 'none':
        item['after'] = after
    return [item]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('run')
    p.add_argument('--seed', type=int, choices=bundle.SEEDS, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p = sub.add_parser('queue-json')
    p.add_argument('--seed', type=int, choices=bundle.SEEDS, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--after', required=True)
    p.add_argument('--out', required=True)
    a = parser.parse_args(argv)
    if a.action == 'run':
        run(a.seed, a.device, a.workers)
    else:
        entries = queue_entries(a.after, a.seed, a.workers)
        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(entries, stream, indent=2)
        print(json.dumps({'output': str(path), 'jobs': [v['tag'] for v in entries]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
