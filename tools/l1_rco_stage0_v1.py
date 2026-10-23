"""L1 RCO Stage 0 v1.0.0. CPU preparation only; B owns K0 execution.

Changelog: initial sealed roster, prior-free trainer derivative, subset contract.
EXPOSED_EXPLORATORY; clean_oof=false. This is NOT a submission candidate.
"""
from __future__ import annotations
import argparse
import csv
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VERSION = "1.0.0"
WORK = ROOT / '.local/scratch/l1-rco/stage0'
CACHE = Path('G:/temp_data/rsna-knee/l1rco-cache')
PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
UID = 'StudyInstanceUID'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
REG = '.local/scratch/astra-b4/H-B-146/launch-v115/registration.json'
TARGETS = 'storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv'
SOURCE = 'tools/exp189_s6_src'
CODE = '.local/exp189-cpu-v120/input/cloud_code'
ANATOMY = 'storage/weights/external/heliosli_rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt'
INIT = 'storage/weights/exp171/convnext_small_fb_in22k_ft_in1k.safetensors'
RAW = 'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip'
PINS = {
    REG: '3e98b8d279e18e393f606d15b7e272ebef5a8e8a0620694afc23115e04e7c14d',
    TARGETS: '79ce307e246cc54a9a90f32fd1ab21d58ebd23b4b7e7b1a80207adcea1bcc2c9',
    'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/folds.csv': '21bec603e6130f9afcb6fd81dcdfe4bdd98128fc3d229ceceeb4356175f4ded0',
    ANATOMY: '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e',
    INIT: '2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9',
    SOURCE + '/compact_train.py': '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80',
    SOURCE + '/px_cache.py': 'b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632',
    SOURCE + '/knee_anchor_bank_v2.py': 'e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309',
    SOURCE + '/probe_infer.py': 'a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980',
    SOURCE + '/side_common.py': 'bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566',
    SOURCE + '/compact_infer.py': '34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a',
    SOURCE + '/probe_dual.py': '9171522ada7ad94145529bb0a47ec48b6f0d8d32d651a60f3f207558649e36b8',
}

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def replace(src, dst):
    for attempt in range(40):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(.05 * (attempt + 1), 1))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    replace(tmp, path)

def fold(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4

def verify_pins():
    for p, h in PINS.items():
        need(sha(ROOT / p) == h, 'pin mismatch: ' + p)
    from tools.exp189_s6_port import verify_code
    verify_code(ROOT / CODE, ROOT / ANATOMY)

def family_pins():
    paths = sorted((ROOT / 'tools').glob('l1_rco_*_v1.py'))
    paths += [ROOT / 'tools/exp189_s6_port.py', ROOT / 'tools/s6plus.py']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths}

def validate_roster(r):
    tr, va, gold = set(r['train']), set(r['val']), set(r['gold'])
    need(len(tr) == len(r['train']) == 3190 and len(va) == len(r['val']) == 977, 'full roster counts')
    need(not tr & va and not (tr | va) & gold, 'roster overlap')
    need(all(fold(u) != 0 for u in tr) and all(fold(u) == 0 for u in va), 'fold-0 mismatch')
    tc, vc = {r['components'][u] for u in tr}, {r['components'][u] for u in va}
    need(not tc & vc, 'connected component leakage')
    return {'train': digest(sorted(tr)), 'holdout': digest(sorted(va)),
            'components': digest(r['components']), 'gold': digest(sorted(gold))}

def register(train_n=200, holdout_n=32):
    need(2 <= train_n <= 200 and 1 <= holdout_n <= 32, 'Stage 0 subset cap: 200 train + 32 holdout')
    verify_pins()
    r = read(ROOT / REG)
    roster = validate_roster(r)
    order = lambda ids: sorted(ids, key=lambda u: hashlib.sha256(('L1-RCO-v1:' + u).encode()).hexdigest())
    tr, va = order(r['train'])[:train_n], order(r['val'])[:holdout_n]
    from tools.s6plus import raw_index
    reg = {'version': VERSION, 'pins': PINS, 'family_pins': family_pins(), 'roster': roster,
           'train': tr, 'holdout': va, 'raw': raw_index(ROOT / RAW, tr + va),
           'exposure': 'EXPOSED_EXPLORATORY', 'clean_oof': False, 'mirror': False,
           'limitations': ['Known-component isolation only; anchor ancestry unresolved.',
                          'SKM-TEA/Gold58 heuristic tuning; label-derived rulers not independent image truth.',
                          'Exact 14767 trainer, not the 04df7 v9.8 mirror trainer.'],
           'recipe': {'size': 384, 'views': 'v4', 'crop': 352, 'slices': 3, 'phys_mm': 3.5,
                      'batch': 2, 'windows': 64, 'steps': 200, 'epochs_schedule': 16,
                      'folds': 4, 'only_folds': '0', 'holdout_fold': None, 'bias': 'zero-prior-free',
                      'eval_stride': 3, 'predict_chunk': 512, 'side_model': None}}
    reg['contract_id'] = digest(reg)
    write(WORK / 'registration.json', reg)
    return reg

def validate(reg):
    need(reg['contract_id'] == digest({k: v for k, v in reg.items() if k != 'contract_id'}), 'registration digest')
    need(reg['version'] == VERSION and reg['pins'] == PINS and reg['family_pins'] == family_pins(), 'tool/input identity changed; register again')
    verify_pins()
    full = read(ROOT / REG)
    need(reg['roster'] == validate_roster(full), 'sealed roster changed')
    need(2 <= len(reg['train']) <= 200 and 1 <= len(reg['holdout']) <= 32, 'subset cap')
    need(set(reg['train']) <= set(full['train']) and set(reg['holdout']) <= set(full['val']), 'subset membership')
    need(len(set(reg['train'] + reg['holdout'])) == len(reg['train']) + len(reg['holdout']), 'duplicate UID')
    return reg

def target_rows(reg, source_rows):
    """Never even parse numeric targets of any UID outside the sealed train subset."""
    wanted = set(reg['train'])
    values = {}
    for row in source_rows:
        u = row[UID]
        if u not in wanted:
            continue
        need(u not in values, 'duplicate training target UID')
        a = [float(row[l]) for l in LABELS]
        need(all(math.isfinite(x) and 0 <= x <= 1 for x in a), 'invalid training targets')
        values[u] = a
    need(set(values) == wanted, 'missing training targets')
    rows = []
    for u in reg['train'] + reg['holdout']:
        row = {UID: u, 'is_gold': '0', 'review_holdout': '0', 'fold': str(fold(u))}
        row.update({l + '|prior': v for l, v in zip(LABELS, values[u] if u in wanted else [.5] * 12)})
        rows.append(row)
    return rows

def derive_trainer(source):
    """Exact-count patches only: preserve all unregistered training math."""
    patches = [
        ('    bias0 = torch.logit(col(pool, "prior", float("nan")).mean(0).clamp(1e-3, 1 - 1e-3))',
         '    bias0 = torch.zeros(C, device=dev)  # L1 K1: no labels enter bias initialization'),
        ('    a = ap.parse_args()', '    a = ap.parse_args()\n    assert a.folds == 4 and a.only_folds == "0" and a.holdout_fold is None\n    from tools.l1_rco_runtime_v1 import Monitor\n    l1 = Monitor(a.out, enabled=not a.check_targets)'),
        ('            for ep in range(start_ep, a.epochs):', '            l1.phase("train")\n            for ep in range(start_ep, a.epochs):'),
        ('                    n += len(idx)', '                    n += len(idx)\n                    l1.step(loss, mask)\n                    if l1.steps >= 200:\n                        break'),
        ('            steps_total += total', '                if l1.steps >= 200:\n                    break\n            l1.phase("checkpoint")\n            steps_total += l1.steps'),
        ('        model.eval()\n        ds, dl = make_loader(uids, "eval", seed, False)',
         '        model.eval()\n        l1.phase("eval")\n        ds, dl = make_loader(uids, "eval", seed, False)'),
        ('            out[idx.to(dev)] = z.float()', '            out[idx.to(dev)] = z.float()\n            l1.evaluation(z, mask)'),
        ('            logs.append([log[0], log[-1]])',
         '            l1.finish()\n            return  # Stage 0 never continues to a full fit or emits OOF\n            logs.append([log[0], log[-1]])'),
        ('                    os.replace(ckpt_path + ".tmp", ckpt_path)',
         '                    from tools.l1_rco_stage0_v1 import replace\n                    replace(ckpt_path + ".tmp", ckpt_path)'),
    ]
    for old, new in patches:
        need(source.count(old) == 1, 'trainer patch occurrence: ' + old[:70])
        source = source.replace(old, new)
    return source

def trainer_argv(cache, check=False):
    cmd = [PYTHON, str(WORK / 'compact_train_l1.py'), '--px', str(cache), '--targets', str(WORK / 'targets.csv'),
           '--out', str(WORK / 'k0'), '--arms', 'E_prior', '--backbone', 'convnext_small', '--init', str(ROOT / INIT),
           '--seeds', '1', '--folds', '4', '--only-folds', '0', '--epochs', '16', '--batch', '2',
           '--train-windows', '64', '--crop', '352', '--slices', '3', '--slice-jitter', '.25', '--ema', '.999',
           '--grad-ckpt', '--phys-mm', '3.5', '--lr', '3e-4', '--head-lr', '1e-3', '--wd', '.02',
           '--workers', '6', '--eval-workers', '3', '--eval-stride', '3', '--train-sampling', 'cover',
           '--cover-stride', '3', '--spatial-attn', '--ckpt-every', '0', '--i-have-user-authorisation-to-finetune']
    return cmd + (['--check-targets'] if check else [])

def prepare(reg, cache):
    validate(reg)
    with (ROOT / TARGETS).open(newline='', encoding='utf-8') as f:
        rows = target_rows(reg, csv.DictReader(f))
    p = WORK / 'targets.csv'
    with p.with_suffix('.tmp').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    replace(p.with_suffix('.tmp'), p)
    original = (ROOT / SOURCE / 'compact_train.py').read_text(encoding='utf-8')
    derived = derive_trainer(original)
    (WORK / 'compact_train_l1.py').write_text(derived, encoding='utf-8')
    (WORK / 'trainer.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), derived.splitlines(True), fromfile='14767-parent', tofile='L1-v1.0.0')), encoding='utf-8')
    receipt = {'contract_id': reg['contract_id'], 'targets_sha256': sha(p), 'trainer_sha256': sha(WORK / 'compact_train_l1.py'),
               'train_count': len(reg['train']), 'holdout_neutral_count': len(reg['holdout']), 'bias': 'zero',
               'argv': trainer_argv(cache), 'cache': str(Path(cache).resolve()), 'clean_oof': False}
    write(WORK / 'prepare.json', receipt)
    command = [PYTHON, str(ROOT / 'tools/l1_rco_runtime_v1.py'), 'run', '--cache', str(cache),
               '--review', str(WORK / 'K0-REVIEW.json')]
    plan = [{'tag': 'L1_RCO_K0_V1', 'command': command}]
    from tools.gpu_queue import validate_plan
    validate_plan(plan)
    write(WORK / 'gpu-queue.json', plan)
    projection(reg)
    return receipt

def projection(reg):
    p = {'version': VERSION, 'contract_id': reg['contract_id'], 'status': 'UNMEASURED',
         'full_cache_bytes': None, 'full_cache_hours': None, 'fold0_fit_hours': None,
         'kaggle_T4_serving_increment_hours': None, 'T4_headroom_hours_reference_only': 2.2,
         'formulas': {'cache': 'measured bytes/study * 4167; optional 4407-all-study scenario separately',
                      'cache_time': 'measured seconds/study * 4167 / 3600 (serial same CPU)',
                      'fit_lower_bound': '3190 * 64 * 16 / measured_loader_inclusive_windows_per_s / 3600; validation/checkpoints extra',
                      'T4': 'requires separately approved T4 execution and model/serving parity; 4070 K0 cannot establish it'},
         'claimed_390GB': '4407 * 606 * 384**2 = 393.802 GB decimal; row mean UNMEASURED, excludes metadata',
         'authorization': 'No full cache, full fit, GPU launch or Kaggle action authorized by this sheet.',
         'go': 'K0 exactly 200 steps; finite decreasing loss; >=80 real windows/s; peak <=15e9 bytes AND <=12 GiB AND desktop+peak+2GiB<=total. Independent result review; T4 remains blocked.'}
    write(WORK / 'projection.json', p)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['register', 'validate', 'prepare'])
    ap.add_argument('--train', type=int, default=200); ap.add_argument('--holdout', type=int, default=32)
    ap.add_argument('--cache', type=Path, default=CACHE)
    a = ap.parse_args()
    if a.action == 'register':
        r = register(a.train, a.holdout)
    else:
        r = validate(read(WORK / 'registration.json'))
        if a.action == 'prepare':
            prepare(r, a.cache)
    print(json.dumps({'action': a.action, 'status': 'PASS', 'contract_id': r['contract_id'], 'train': len(r['train']), 'holdout': len(r['holdout'])}))

if __name__ == '__main__':
    main()
