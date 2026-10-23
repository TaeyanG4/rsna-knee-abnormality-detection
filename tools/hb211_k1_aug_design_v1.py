"""H211 v1.0.0: source-bound, CPU-only design check for one K1 augmentation arm.

Does not register/admit/launch training, import a trainer, read labels/pixels/
checkpoints, or contact Kaggle. Reuses the unchanged parent augmentation
function through AST extraction. This is preparation, not efficacy evidence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / '.local/scratch/l1-rco/k1v3-grouping'
PARENT_CONTRACT = 'a58f1476ea4ca3c8d64fe06c80bc6e24d78fdf2f347c4de68c1eba5d9c2ea83d'
PARENT_TRAINER = '0d5fbae473eb1dc451871be0d610689d6baff8e6ac258acb091e66343ad0b791'


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_parent():
    reg = json.loads((PARENT / 'registration.json').read_text(encoding='utf-8'))
    prep = json.loads((PARENT / 'prepare.json').read_text(encoding='utf-8'))
    src_path = PARENT / 'compact_train_l1.py'
    require(reg['contract_id'] == prep['contract_id'] == PARENT_CONTRACT, 'parent contract drift')
    require(sha(src_path) == prep['trainer_sha256'] == PARENT_TRAINER, 'parent source drift')
    argv = prep['argv']
    require('--aug' not in argv and '--aug-res' not in argv, 'parent already overrides augmentation')
    tree = ast.parse(src_path.read_text(encoding='utf-8'))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'augment']
    require(len(functions) == 1, 'augmentation source identity')
    defaults = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument':
            if n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value in ('--aug', '--aug-res'):
                defaults[n.args[0].value] = next(ast.literal_eval(k.value) for k in n.keywords if k.arg == 'default')
    require(defaults == {'--aug': 'crop', '--aug-res': 0.0}, 'parent parser default drift')
    fn = functions[0]
    return fn, dict(parent_contract=PARENT_CONTRACT, parent_trainer_sha256=PARENT_TRAINER,
                    parent_prepare_sha256=sha(PARENT / 'prepare.json'), defaults=defaults,
                    augmentation_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                    planned_argument_delta=['--aug', 'v2', '--aug-res', '0.0'],
                    source_change='none in augmentation/math/serving; separate wrapper identity still required')


def cpu_checks(fn):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'unexpected preexisting CUDA context')
    options = SimpleNamespace(crop=32, aug='v2', aug_res=0.0)
    namespace = {'torch': torch, 'F': F, 'math': math, 'a': options}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<parent-augment>', 'exec'), namespace)
    augment = namespace['augment']
    # Identical windows and channels test one transform per study, not independent
    # slice/view distortion. Deliberately use an asymmetric, nonconstant image.
    base = ((torch.arange(40*40).reshape(40, 40)*7 + 23) % 251).to(torch.uint8)
    image = base[None, None, None].repeat(2, 3, 3, 1, 1)
    before = image.clone()
    gen = torch.Generator().manual_seed(110211)
    saved = gen.get_state()
    transformed = augment(image, gen)
    checks = {}
    checks['shape_dtype_range'] = (tuple(transformed.shape) == (2, 3, 3, 32, 32)
                                  and transformed.dtype == torch.uint8)
    checks['input_immutable'] = torch.equal(image, before)
    checks['same_transform_all_windows'] = all(torch.equal(transformed[:, 0], transformed[:, i]) for i in (1, 2))
    checks['same_transform_adjacent_channels'] = all(torch.equal(transformed[:, :, 0], transformed[:, :, i]) for i in (1, 2))
    gen.set_state(saved)
    checks['generator_restore_replay'] = torch.equal(transformed, augment(image, gen))
    options.aug = 'crop'
    gen.set_state(saved)
    cropped = augment(image, gen)
    reference = torch.Generator().manual_seed(110211)
    oy = int(torch.randint(0, 9, (1,), generator=reference))
    ox = int(torch.randint(0, 9, (1,), generator=reference))
    checks['control_crop_byte_parity'] = torch.equal(cropped, image[..., oy:oy+32, ox:ox+32])
    checks['augmentation_is_active'] = not torch.equal(transformed, cropped)
    # Full spatial size but two synthetic windows: exercise actual interpolation
    # dimensions without model/GPU/full-shape memory admission.
    options.crop, options.aug = 352, 'v2'
    actual_size = torch.arange(384*384).remainder(251).to(torch.uint8).reshape(1, 1, 1, 384, 384).repeat(1, 2, 3, 1, 1)
    actual = augment(actual_size, torch.Generator().manual_seed(110211))
    checks['real_spatial_dimensions'] = tuple(actual.shape) == (1, 2, 3, 352, 352)
    checks['cpu_only'] = actual.device.type == 'cpu' and not torch.cuda.is_initialized()
    require(all(checks.values()), 'failed CPU augmentation contract: ' + str([k for k, v in checks.items() if not v]))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    require(not args.out.exists(), 'preserve original evidence; output already exists')
    fn, identity = inspect_parent()
    result = dict(version='1.0.0', status='CPU_DESIGN_CHECKED_NOT_LAUNCH_READY', identity=identity,
                  checks=cpu_checks(fn), review='B_DIRECT_NOT_INDEPENDENT',
                  measured_efficacy=False, launch_allowed=False, model_loaded=False,
                  not_checked=['CUDA augmentation peak/throughput', 'full training wrapper binding and resume',
                               'real-cache CPU smoke', 'new checkpoint efficacy or exact final blend'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'checks_passed': len(result['checks']),
                      'launch_allowed': False, 'cuda_initialized': False}))


if __name__ == '__main__':
    main()
