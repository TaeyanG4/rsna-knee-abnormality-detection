"""L1 RCO K0 v1.2.0: evaluation encoder chunk 32; unchanged training.

Changelog: isolated V2 attempt; retain v1 data/cache contract and all memory gates.
No GPU execution here. Existing v1 files and failed attempt are immutable.
"""
from __future__ import annotations
import argparse
import difflib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.l1_rco_stage0_v1 import *  # sealed data contract, unchanged
from tools import l1_rco_stage0_v1 as old

VERSION = '1.2.0'
WORK = ROOT / '.local/scratch/l1-rco/k0-v2'
EVAL_CHUNK = 32


def eval_pins():
    return {name: sha(ROOT / 'tools' / name) for name in
            ('l1_rco_stage0_v120.py', 'l1_rco_runtime_v120.py', 'l1_rco_evalproof_v120.py')}


def register():
    reg = old.validate(read(old.WORK / 'registration.json'))
    write(WORK / 'registration.json', reg)
    return reg


def derive_trainer(source):
    parent = old.derive_trainer(source)
    patches = [
        ('from tools.l1_rco_runtime_v1 import Monitor',
         'from tools.l1_rco_runtime_v120 import Monitor'),
        ('        l1.phase("eval")',
         '        if dev.type == "cuda":\n            torch.cuda.empty_cache()  # evaluation-only allocator cleanup\n        l1.phase("eval")'),
        ('mask.to(dev), chunk=PREDICT_CHUNK)', 'mask.to(dev), chunk=32)'),
    ]
    result = parent
    for before, after in patches:
        need(result.count(before) == 1, 'eval patch occurrence: ' + before)
        result = result.replace(before, after)
    return result


def trainer_argv(cache, check=False):
    cmd = old.trainer_argv(cache, check)
    return [x.replace(str(old.WORK), str(WORK)) for x in cmd]


def prepare(reg, cache):
    old.validate(reg)
    prior = read(old.WORK / 'prepare.json')
    need(prior['contract_id'] == reg['contract_id'], 'parent prepare contract')
    need(prior['targets_sha256'] == sha(old.WORK / 'targets.csv'), 'parent targets changed')
    need(prior['trainer_sha256'] == sha(old.WORK / 'compact_train_l1.py'), 'parent trainer changed')
    source = (ROOT / SOURCE / 'compact_train.py').read_text(encoding='utf-8')
    parent = old.derive_trainer(source)
    need(parent == (old.WORK / 'compact_train_l1.py').read_text(encoding='utf-8'), 'parent derivative mismatch')
    # Copy only the already-sealed training/neutral-holdout targets, never source labels.
    (WORK / 'targets.csv').write_bytes((old.WORK / 'targets.csv').read_bytes())
    derived = derive_trainer(source)
    (WORK / 'compact_train_l1.py').write_text(derived, encoding='utf-8')
    (WORK / 'trainer.diff').write_text(''.join(difflib.unified_diff(
        parent.splitlines(True), derived.splitlines(True), fromfile='K0-v1', tofile='K0-v1.2.0')), encoding='utf-8')
    receipt = dict(prior, version=VERSION, eval_chunk=EVAL_CHUNK, eval_pins=eval_pins(),
                   trainer_sha256=sha(WORK / 'compact_train_l1.py'), argv=trainer_argv(cache),
                   cache=str(Path(cache).resolve()))
    write(WORK / 'prepare.json', receipt)
    command = [PYTHON, str(ROOT / 'tools/l1_rco_runtime_v120.py'), 'run',
               '--cache', str(cache), '--review', str(WORK / 'K0-REVIEW.json')]
    plan = [{'tag': 'L1_RCO_K0_V2', 'command': command}]
    from tools.gpu_queue import validate_plan
    validate_plan(plan)
    write(WORK / 'gpu-queue.json', plan)
    write(WORK / 'projection.json', read(old.WORK / 'projection.json'))
    return receipt


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['register', 'validate', 'prepare'])
    ap.add_argument('--cache', type=Path, default=CACHE)
    a = ap.parse_args()
    reg = register() if a.action == 'register' else old.validate(read(WORK / 'registration.json'))
    if a.action == 'prepare':
        prepare(reg, a.cache)
    print(json.dumps({'status': 'PASS', 'action': a.action, 'version': VERSION, 'contract_id': reg['contract_id']}))


if __name__ == '__main__':
    main()
