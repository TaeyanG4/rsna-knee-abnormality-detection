"""EXP166 v1.0.0: offline ConvNeXt-T AUG1 partner, Colab/Vast staging and screen.

Local commands are CPU only. ``remote-run`` is the only GPU entry point and is
invoked by B in a reviewed remote workspace, never by prepare or stage.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp127_radr50_single as old
from tools import exp150_consensus_weight as e150

VERSION = '1.0.0'
ARCH = 'convnext_tiny.fb_in22k_ft_in1k'
WEIGHT_REPO = 'timm/convnext_tiny.fb_in22k_ft_in1k'
WEIGHT = 'storage/weights/exp166/convnext_tiny_fb_in22k_ft_in1k.safetensors'
INITIAL = 'artifacts/experiments/EXP166_CONVNEXT_T/initial-state.pt'
INITIAL_KEYS = 'artifacts/experiments/EXP166_CONVNEXT_T/initial-state-keys.json'
BASE = 'artifacts/experiments/EXP127_AUG1_PROTOCOL_V1/s42/manifest.json'
PROTOCOL = 'artifacts/experiments/EXP166_CONVNEXT_T/PROTOCOL.md'
MANIFEST = 'artifacts/experiments/EXP166_CONVNEXT_T/manifest.json'
BASE_RUNNER = 'training/runners/exp127_radr50_single.py'
RUNNER = 'training/runners/exp166_convnext_t.py'
TOOL = 'tools/exp166_convnext_t.py'
RECEIPT = 'exp166-bundle.json'
ARCHIVE = 'exp166-code.zip.bin'
CODE_DS = 'taeyangg4/knee-exp166-convnext-code'
RESULT_DS = 'taeyangg4/knee-exp166-convnext-f0'
CORPUS_DS = 'taeyangg4/knee-raptor-corpus-336'
TAG = 'EXP166_CONVNEXT_T_S42_F0'
CHECKPOINT = 'epoch-state.pt'
CP_META = 'epoch-state.json'
SCREEN_DIR = 'artifacts/experiments/EXP166_CONVNEXT_T/screen'
RESULT_DIR = 'artifacts/experiments/EXP166_CONVNEXT_T/outputs'
THREADS = e150.THREADS
TRANSITIVE_MODULES = (
    'tools/exp166_convnext_t.py', 'tools/exp127_radr50_single.py',
    'tools/exp150_consensus_weight.py', 'tools/exp087_training.py',
    'tools/exp124_flip_tta.py', 'tools/colab/exp159_remote.py',
    'training/src/raptor_common.py', 'training/src/raptor_data.py',
    'training/src/raptor_exp127.py', 'training/src/raptor_exp127_aug1.py',
    'training/src/raptor_runtime.py', 'training/src/raptor_radimagenet.py',
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def safe(name):
    p = PurePosixPath(name.replace(chr(92), '/'))  # B fix: on Windows Path('/root/x').is_absolute() is False
    need(name and not name.startswith('/') and not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\' not in name,
         'Unsafe bundle path: ' + name)


def fetch_weights(root=ROOT):
    """Explicit local preparation only; no network calls occur inside runs."""
    from huggingface_hub import hf_hub_download
    root = Path(root)
    target = root / WEIGHT
    need(not target.exists(), 'Weight destination already exists')
    source = Path(hf_hub_download(repo_id=WEIGHT_REPO, filename='model.safetensors', token=False))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {'path': WEIGHT, 'sha256': sha(target), 'bytes': target.stat().st_size,
            'repo': WEIGHT_REPO, 'file': 'model.safetensors'}


def build_offline_model(arch, weights_path, expected_sha256):
    """Construct the unchanged Raptor MIL head and strictly load the CNN trunk."""
    need(arch == ARCH and os.environ.get('HF_HUB_OFFLINE') == '1', 'Offline ConvNeXt-T required')
    need(sha(weights_path) == expected_sha256, 'ConvNeXt pretrained weight SHA256 differs')
    from safetensors.torch import load_file
    from training.src.raptor_common import build_model
    import timm
    import torch
    model = build_model(arch, pretrained=False)
    state = load_file(str(weights_path), device='cpu')
    # The repository checkpoint has a 1k classification head; our Raptor
    # backbone has num_classes=0 and uses its own MIL head.
    for name in ('head.fc.weight', 'head.fc.bias'):
        state.pop(name, None)
    expected = model.backbone.state_dict()
    need(set(state) == set(expected), 'ConvNeXt backbone key set differs')
    for key, tensor in state.items():
        need(tuple(tensor.shape) == tuple(expected[key].shape) and tensor.dtype == expected[key].dtype,
             'ConvNeXt tensor shape/dtype differs: ' + key)
    model.backbone.load_state_dict(state, strict=True)
    manifest = ROOT / MANIFEST
    if manifest.is_file():
        registered = read(manifest)
        initial = ROOT / INITIAL
        need(sha(initial) == registered['initial_state_file_sha256'], 'Initial state file differs')
        model.load_state_dict(torch.load(initial, map_location='cpu', weights_only=True), strict=True)
    cfg = timm.data.resolve_model_data_config(model.backbone)
    mean, std = tuple(cfg['mean']), tuple(cfg['std'])
    need(all(abs(a-b) < 1e-6 for a, b in zip(mean, (.485, .456, .406))) and
         all(abs(a-b) < 1e-6 for a, b in zip(std, (.229, .224, .225))),
         'Backbone normalization is not the registered ImageNet mean/std')
    return model, {'weights_sha256': expected_sha256, 'weights_file': Path(weights_path).name,
                   'strict_result': 'all backbone keys loaded', 'mean': mean, 'std': std,
                   'backbone_parameters': sum(p.numel() for p in model.backbone.parameters())}


def runner_source(source):
    """Narrowly derive the EXP127 runner; fail if any base anchor drifts."""
    changes = [
        ('from tools import exp127_radr50_single as guard', 'from tools import exp166_convnext_t as guard'),
        ('choices=["radimagenet"]', 'choices=["imagenet"]'),
        ('choices=["radimagenet"], required=True)  # EXP107: RadImageNet checkpoint',
         'choices=["timm-in22k"], required=True)  # EXP166: offline ConvNeXt checkpoint'),
        ('return build_radimagenet_model(a.arch, Path(a.init_weights), weights_sha)',
         'return guard.build_offline_model(a.arch, Path(a.init_weights), weights_sha)'),
        ('combined == protocol["initial_state_sha256"]', 'combined == protocol["initial_state_sha256"]'),
        ('best.update(input_policy=a.input_policy, init="radimagenet", init_weights_sha256=weights_sha)',
         'best.update(input_policy=a.input_policy, init="timm-in22k", init_weights_sha256=weights_sha, '
         'flip_tta="coronal_axial_horizontal_mean_probability")'),
        ('if out.exists() and any(out.iterdir()):',
         'if out.exists() and any(out.iterdir()) and not guard.resume_directory_allowed(out):'),
        ('history, best, optimizer_updates = [], None, 0',
         'history, best, optimizer_updates, skipped, start_epoch = guard.restore_epoch(\n'
         '        out, model, opt, sched, scaler, shuffle_generator, provenance, skipped)'),
        ('for ep in range(epochs):', 'for ep in range(start_epoch, epochs):'),
        ('(out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")',
         '(out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")\n'
         '        guard.save_epoch(out, ep, model, opt, sched, scaler, shuffle_generator,\n'
         '                         history, best, optimizer_updates, skipped, provenance)\n'
         '        guard.publish_epoch(out, ep)\n'
         '        if a.pilot_only:\n'
         '            if ep != 0 or rec["epoch_seconds"] > 300: raise RuntimeError("invalid <=5-minute pilot")\n'
         '            guard.put(out / "pilot.json", {"epoch_seconds": rec["epoch_seconds"],\n'
         '                "peak_gpu_gib": rec["peak_gpu_gib"], "projected_fit_seconds": rec["epoch_seconds"] * 20})\n'
         '            return'),
    ]
    # Use distinctive comments to avoid replacing normalization twice.
    changes[1] = ('ap.add_argument("--norm", choices=["radimagenet"], required=True)',
                  'ap.add_argument("--norm", choices=["imagenet"], required=True)')
    for old_text, new_text in changes:
        if old_text == new_text:
            continue
        need(source.count(old_text) == 1, 'EXP127 runner anchor changed: ' + old_text[:60])
        source = source.replace(old_text, new_text)
    source = source.replace(' | {a.arch} RadImageNet | ', ' | {a.arch} ImageNet-22k | ')
    # Add pilot flag, plane-aware flip evaluation, and K36-TTA readout.
    anchor = 'ap.add_argument("--validate-only", action="store_true")'
    need(source.count(anchor) == 1, 'pilot parser anchor changed')
    source = source.replace(anchor, anchor + '\n    ap.add_argument("--pilot-only", action="store_true")')
    anchor = 'optimizer_updates += 1; tot += loss.item()'
    need(source.count(anchor) == 1, 'pilot timing anchor changed')
    source = source.replace(anchor, anchor + '\n            if a.pilot_only and time.time() - t0 > 300:\n'
                            '                raise RuntimeError("Pilot epoch exceeded five minutes")')
    anchor = 'P36, U36 = evaluate(oof_ids, k_extra)'
    need(source.count(anchor) == 1, 'K36 evaluation anchor changed')
    source = source.replace(anchor, anchor + '\n    PTTA, UTTA = guard.evaluate_flip_tta(\n'
                            '        model, corpus, oof_ids, all_labels, a, dev, eval_dtype, autocast)')
    anchor = 'write_predictions(out / protocol["extra_eval"]["file"], U36, P36)'
    need(source.count(anchor) == 1, 'OOF write anchor changed')
    source = source.replace(anchor, anchor + '\n    if UTTA != U36: raise RuntimeError("TTA UID order changed")\n'
                            '    write_predictions(out / "oof_k36_flip_tta.csv", UTTA, PTTA)')
    anchor = 'names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv", protocol["extra_eval"]["file"]]'
    need(source.count(anchor) == 1, 'inventory anchor changed')
    source = source.replace(anchor, anchor[:-1] + ', "oof_k36_flip_tta.csv"]')
    anchor = 'provenance["cuda_initialized"] = bool(torch.cuda.is_initialized())'
    need(source.count(anchor) == 1, 'TTA provenance anchor changed')
    source = source.replace(anchor, anchor + '\n    provenance["flip_tta"] = "coronal_axial_horizontal_mean_probability"')
    # The original RadImageNet import remains packaged, but this derived
    # runner never uses the old builder or its weight file.
    compile(source, RUNNER, 'exec')
    return source


def prepare(root=ROOT):
    root = Path(root)
    dest = root / MANIFEST
    need(not dest.exists(), 'Registration already exists')
    weight = root / WEIGHT
    need(weight.is_file() and not weight.is_symlink(), 'Fetch the private ConvNeXt weight asset first')
    base = old.load_protocol('AUG1', 42, root)
    need(base['folds']['0']['n_train'] == 3455 and base['folds']['0']['n_val'] == 888,
         'Control roster changed')
    protocol_sha = sha(root / PROTOCOL)
    derived = runner_source((root / BASE_RUNNER).read_text(encoding='utf-8'))
    p = copy.deepcopy(base)
    p['schema'] = 'exp166-convnext-t-v1'
    p['status'] = 'REGISTERED_NO_EXECUTION'
    p['parent_manifest_sha256'] = sha(root / BASE)
    p['protocol_sha256'] = protocol_sha
    p['runner_sha256'] = hashlib.sha256(derived.encode()).hexdigest()
    p['tool_sha256'] = sha(root / TOOL)
    p['folds']['0']['tag'] = TAG
    p['common_args'].update(arch=ARCH, norm='imagenet', pretrained='timm-in22k')
    p['tags'] = {'prefix': TAG, 'screen_dir': SCREEN_DIR}
    p['recipe']['changes'] = ['EXP127 AUG1 seed-42 fold-0 recipe retained; only backbone and its required normalization changed',
                               'timm ConvNeXt-T ImageNet-22k -> ImageNet-1k fine-tuned weights; offline strict backbone load',
                               'ImageNet mean/std instead of RadImageNet 2x-1 normalization',
                               'AdamW backbone/head LR and weight decay unchanged',
                               'K36 plane-aware coronal/axial horizontal flip TTA reported beside plain K36']
    p['evaluation']['exp166_screen'] = {'draws': 300, 'bootstrap_seed': 20261003,
        'gate': 'H1 w0.1 mean(JEV,aquila) delta >0, CI lower >0, point > max(0.0013, DINO-base H1 w0.1)'}
    p['init_weights'] = {'local': WEIGHT, 'sha256': sha(weight), 'bytes': weight.stat().st_size,
                         'repo': WEIGHT_REPO, 'file': 'model.safetensors'}
    p['backbone_change'] = {'normalization': 'ImageNet mean/std from timm pretrained_cfg',
                            'lr_wd': 'unchanged 5e-4 / 1e-3 / 0.02', 'tta': 'plane-aware coronal/axial horizontal flip'}
    p['initial_state_sha256'] = None  # filled after deterministic CPU construction below
    os.environ['HF_HUB_OFFLINE'] = '1'
    from training.src.raptor_common import seed_all
    old.worker_init(0)
    seed_all(42)
    model, report = build_offline_model(ARCH, weight, p['init_weights']['sha256'])
    combined, keys = old.state_hashes(model)
    p['initial_state_sha256'] = combined
    p['initial_state']['keys_file'] = INITIAL_KEYS
    p['initial_state']['arch'] = ARCH
    p['initial_state']['keys'] = len(keys)
    p['initial_state']['backbone_keys'] = sum(name.startswith('backbone.') for name in keys)
    p['initial_state']['rule'] = 'CPU seed 42; strict offline ConvNeXt backbone load; save full initial state for cross-host parity'
    p['initial_state']['enforcement'] = 'strict full state file and sorted state_dict SHA256 before training'
    put(root / INITIAL_KEYS, {'keys': keys, 'combined_sha256': combined, 'seed': 42})
    import torch
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, root / INITIAL)
    p['initial_state_file_sha256'] = sha(root / INITIAL)
    p['pretrained_load'] = report
    put(dest, p)
    return {'status': p['status'], 'manifest': MANIFEST, 'weight_sha256': p['init_weights']['sha256'],
            'runner_sha256': p['runner_sha256'], 'initial_state_sha256': p['initial_state_sha256']}


def registration(root=ROOT):
    root = Path(root)
    p = read(root / MANIFEST)
    need(p['schema'] == 'exp166-convnext-t-v1' and p['status'] == 'REGISTERED_NO_EXECUTION',
         'Wrong EXP166 registration')
    need(sha(root / BASE) == p['parent_manifest_sha256'] and sha(root / PROTOCOL) == p['protocol_sha256']
         and sha(root / TOOL) == p['tool_sha256'], 'Registration source drift')
    need(sha(root / WEIGHT) == p['init_weights']['sha256'], 'Pretrained asset drift')
    need(sha(root / INITIAL) == p['initial_state_file_sha256'], 'Registered initial state drift')
    derived = runner_source((root / BASE_RUNNER).read_text(encoding='utf-8'))
    need(hashlib.sha256(derived.encode()).hexdigest() == p['runner_sha256'], 'Derived runner drift')
    return p, derived


# The derived runner reads these names. Explicit aliases keep its guard contract
# auditable, and package_files includes their transitive modules.
CODE_NAMES = old.CODE_NAMES
NEW_CODE = old.NEW_CODE
LABEL_SET = old.LABEL_SET
ENV_KEYS = old.ENV_KEYS
read_rows = old.read_rows
canonical_order = old.canonical_order
validate_fold_membership = old.validate_fold_membership
pos_weight = old.pos_weight
fold_contract = old.fold_contract
worker_init = old.worker_init
install_gold_blocker = old.install_gold_blocker
uid_hash = old.uid_hash
state_hashes = old.state_hashes
mismatched_keys = old.mismatched_keys
load_label_csv = old.load_label_csv
forbid_cuda_init = old.forbid_cuda_init


def validate_arguments(a, root):
    root = Path(root)
    p, _ = registration(root)
    need(Path(a.protocol).resolve() == (root / MANIFEST).resolve() and a.variant == 'AUG1' and
         a.fold == 0 and a.seed == 42 and a.tag == TAG and a.mode == 'full', 'Wrong fold/seed/tag')
    for key, value in p['common_args'].items():
        need(getattr(a, key) == value, 'Frozen argument changed: ' + key)
    need(0 <= a.workers <= 12 and a.device == 'cuda', 'Worker/device contract changed')
    for key in ('corpus', 'labels', 'eval_labels'):
        need(Path(getattr(a, key)).resolve() == (root / p['paths'][key]).resolve(),
             'Input path differs: ' + key)
    need(Path(a.init_weights).resolve() == (root / WEIGHT).resolve(), 'Weight path differs')
    need(os.environ.get('HF_HUB_OFFLINE') == '1' and all(os.environ.get(k) == '1' for k in THREADS),
         'Offline/thread environment required')
    out = Path(a.output_root).resolve()
    need(out != root / 'artifacts/experiments' and out.parent.is_dir(), 'Remote output root required')
    return p, p['folds']['0'], read(root / p['folds']['0']['ordered_ids']['path']), out


def resume_directory_allowed(out):
    out = Path(out)
    return (out / CHECKPOINT).is_file() and (out / CP_META).is_file() and not (out / 'summary.json').exists()


def environment_key(provenance):
    return {'environment': provenance['environment'], 'runner_sha256': provenance['runner_sha256'],
            'protocol_sha256': provenance['protocol_sha256'],
            'initial_state_sha256': provenance['initial_state_sha256']}


def save_epoch(out, ep, model, opt, sched, scaler, generator, history, best, updates, skipped, provenance):
    import numpy as np
    import torch
    out = Path(out)
    state = {'schema': 'exp166-epoch-v1', 'epoch': ep, 'model': model.state_dict(),
             'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'scaler': scaler.state_dict(),
             'generator': generator.get_state(), 'python_rng': random.getstate(),
             'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all(), 'history': history, 'best': best,
             'optimizer_updates': updates, 'skipped': skipped, 'environment': environment_key(provenance)}
    tmp = out / (CHECKPOINT + '.tmp')
    torch.save(state, tmp)
    tmp.replace(out / CHECKPOINT)
    put(out / CP_META, {'schema': state['schema'], 'epoch': ep, 'checkpoint_sha256': sha(out / CHECKPOINT),
                        'environment': state['environment']})


def restore_epoch(out, model, opt, sched, scaler, generator, provenance, skipped):
    import numpy as np
    import torch
    out = Path(out)
    if not (out / CHECKPOINT).exists():
        return [], None, 0, skipped, 0
    meta = read(out / CP_META)
    need(meta['schema'] == 'exp166-epoch-v1' and sha(out / CHECKPOINT) == meta['checkpoint_sha256']
         and meta['environment'] == environment_key(provenance), 'Resume metadata differs')
    state = torch.load(out / CHECKPOINT, map_location='cpu', weights_only=False)
    need(state['schema'] == meta['schema'] and state['epoch'] == meta['epoch'] and
         0 <= state['epoch'] < 20 and state['environment'] == meta['environment'] and
         len(state['history']) == state['epoch'] + 1 and
         state['optimizer_updates'] == state['history'][-1]['optimizer_updates'],
         'Invalid epoch checkpoint')
    model.load_state_dict(state['model'], strict=True)
    opt.load_state_dict(state['optimizer'])
    sched.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    generator.set_state(state['generator'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['history'], state['best'], state['optimizer_updates'], state['skipped'], state['epoch'] + 1


def evaluate_flip_tta(model, corpus, ids, labels, a, dev, eval_dtype, autocast):
    """Match EXP124: mirror coronal/axial windows only; mean post-sigmoid predictions."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from training.src.raptor_common import select_centers
    from training.src.raptor_data import collate
    from training.src.raptor_radimagenet import CenterSliceWindows
    from tools.exp124_flip_tta import plane_ranges, flip_mask, mirror_windows
    ds = CenterSliceWindows(corpus.dir, ids, labels, a.res, 36, False, a.norm, a.seed,
                            window_policy=a.window_policy, input_policy=a.input_policy)
    dl = DataLoader(ds, batch_size=max(1, a.bs // 2), shuffle=False, num_workers=min(a.workers, 2),
                    collate_fn=collate, worker_init_fn=worker_init)
    ranges = plane_ranges(corpus.spec)
    scores, uids = [], []
    model.eval()
    with torch.inference_mode():
        for x, _y, batch_uids in dl:
            mirrored = []
            for xi, uid in zip(x, batch_uids):
                centers = select_centers(corpus.mask(uid), corpus.depth, 36, corpus.spec,
                                         a.window_policy, None)
                mirrored.append(mirror_windows(xi, flip_mask(centers, ranges)))
            xm = torch.stack(mirrored)
            with autocast(eval_dtype):
                plain = model(x.to(dev))
                flip = model(xm.to(dev))
            need(bool(torch.isfinite(plain).all()) and bool(torch.isfinite(flip).all()),
                 'Nonfinite TTA logits')
            scores.append(((torch.sigmoid(plain.float()) + torch.sigmoid(flip.float())) / 2).cpu().numpy())
            uids += list(batch_uids)
    return np.concatenate(scores), uids


def publish_epoch(out, ep, surface=None):
    """Require verified remote persistence before training the following epoch."""
    out = Path(out)
    if (surface or os.environ.get('EXP166_REMOTE')) != 'colab':
        return True
    from tools.colab.exp159_remote import dataset_files
    put(out / 'dataset-metadata.json', {'id': RESULT_DS, 'title': 'EXP166 ConvNeXt fold 0 private',
        'licenses': [{'name': 'other'}], 'isPrivate': True,
        'description': 'Private resumable EXP166 results.'})
    status = subprocess.run(['kaggle', 'datasets', 'status', RESULT_DS], capture_output=True, text=True, timeout=30)
    cmd = (['kaggle', 'datasets', 'version', '-p', str(out), '-m', f'EXP166 epoch {ep+1}']
           if status.returncode == 0 else ['kaggle', 'datasets', 'create', '-p', str(out)])
    response = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    need(response.returncode == 0 and 'error' not in (response.stdout + response.stderr).lower(),
         'Private checkpoint publication failed')
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            files = dict(dataset_files(RESULT_DS))
            if (files.get(CHECKPOINT) == (out / CHECKPOINT).stat().st_size and
                    files.get(CP_META) == (out / CP_META).stat().st_size):
                return True
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            pass
        time.sleep(10)
    raise RuntimeError('Private epoch checkpoint was not verified in Kaggle dataset')


def package_files(root=ROOT):
    root = Path(root)
    p, derived = registration(root)
    files = e150.package_files(root)
    extra = [PROTOCOL, MANIFEST, WEIGHT, INITIAL, INITIAL_KEYS, TOOL, 'tools/colab/exp159_remote.py',
             'tools/exp124_flip_tta.py']
    for name in extra:
        safe(name)
        source = root / name
        need(source.is_file() and not source.is_symlink(), 'Missing package input: ' + name)
        files[name] = {'sha256': sha(source), 'bytes': source.stat().st_size}
    files[RUNNER] = {'sha256': hashlib.sha256(derived.encode()).hexdigest(),
                     'bytes': len(derived.encode())}
    need(files[RUNNER]['sha256'] == p['runner_sha256'], 'Runner registration differs')
    need(set(TRANSITIVE_MODULES).issubset(files), 'A transitive runner import is not packaged')
    return files, derived


def stage(surface, out, root=ROOT):
    root, out = Path(root), Path(out)
    need(surface in ('colab', 'vast') and not out.exists(), 'New Colab/Vast stage directory required')
    files, derived = package_files(root)
    receipt = {'schema': 'exp166-package-v1', 'version': VERSION, 'surface': surface,
               'files': files, 'manifest_sha256': sha(root / MANIFEST),
               'protocol_sha256': sha(root / PROTOCOL), 'no_execution': True}
    code = out / 'code-dataset' if surface == 'colab' else out / 'package'
    code.mkdir(parents=True)
    archive = code / ARCHIVE
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            if name == RUNNER:
                z.writestr(name, derived.encode())
            else:
                z.write(root / name, name)
        z.writestr(RECEIPT, json.dumps(receipt, indent=2).encode())
    if surface == 'colab':
        put(code / 'dataset-metadata.json', {'id': CODE_DS, 'title': 'EXP166 private ConvNeXt code',
            'isPrivate': True, 'licenses': [{'name': 'other'}],
            'description': 'EXP166 offline code and pretrained weights.'})
    else:
        bundle_sha = hashlib.sha256(json.dumps(receipt, indent=2).encode()).hexdigest()
        put(out / 'vast-queue.json', [{'tag': TAG, 'command': ['python', TOOL, 'remote-run',
            '--surface', 'vast', '--bundle-sha', bundle_sha], 'continue_on_failure': False}])
    put(out / 'stage-receipt.json', {'schema': 'exp166-stage-v1', 'surface': surface,
        'archive_sha256': sha(archive), 'manifest_sha256': receipt['manifest_sha256'],
        'runner_sha256': files[RUNNER]['sha256'], 'weight_sha256': files[WEIGHT]['sha256']})
    verify_stage(out, root)
    if surface == 'colab':
        (out / 'colab-bootstrap-cell.py').write_text(bootstrap_cell(out), encoding='utf-8')
    return read(out / 'stage-receipt.json')


def verify_stage(out, root=ROOT):
    out, root = Path(out), Path(root)
    stage_receipt = read(out / 'stage-receipt.json')
    surface = stage_receipt['surface']
    archive = out / ('code-dataset' if surface == 'colab' else 'package') / ARCHIVE
    need(sha(archive) == stage_receipt['archive_sha256'], 'Stage archive differs')
    files, derived = package_files(root)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and set(names) == {*files, RECEIPT}, 'Bundle entries differ')
        receipt = json.loads(z.read(RECEIPT))
        need(receipt['surface'] == surface and receipt['files'] == files, 'Bundle receipt differs')
        for name, rec in files.items():
            safe(name)
            raw = z.read(name)
            need(len(raw) == rec['bytes'] and hashlib.sha256(raw).hexdigest() == rec['sha256'],
                 'Bundle member differs: ' + name)
            if name == RUNNER:
                need(raw == derived.encode(), 'Derived runner differs')
    if surface == 'colab':
        need(read(out / 'code-dataset/dataset-metadata.json')['isPrivate'] is True,
             'Code dataset must be private')
    return stage_receipt


def bootstrap_cell(out):
    r = verify_stage(out)
    need(r['surface'] == 'colab', 'Colab stage required')
    cell = f'''import os, pathlib, subprocess, sys, zipfile, hashlib
from google.colab import userdata
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
token = userdata.get('KAGGLE_API_TOKEN')
assert token, 'Add KAGGLE_API_TOKEN to Colab secrets'
os.environ['KAGGLE_API_TOKEN'] = token
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', '--no-deps', 'kaggle'], check=True)
subprocess.run(['kaggle', '--version'], check=True)
base = pathlib.Path('/content/exp166-code'); base.mkdir(exist_ok=True)
subprocess.run(['kaggle', 'datasets', 'download', '-d', '{CODE_DS}', '-f', '{ARCHIVE}', '-p', str(base), '--unzip', '-q', '-o'], check=True)
archive = base / '{ARCHIVE}'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '{r['archive_sha256']}', 'code asset SHA differs'
work = pathlib.Path('/content/exp166'); work.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    names = z.namelist(); assert len(names) == len(set(names))
    assert all(not n.startswith('/') and '..' not in pathlib.PurePosixPath(n).parts and ':' not in n and chr(92) not in n for n in names)
    z.extractall(work)
subprocess.run([sys.executable, str(work / '{TOOL}'), 'remote-run', '--surface', 'colab', '--bundle-sha', hashlib.sha256((work / '{RECEIPT}').read_bytes()).hexdigest()], cwd=work, check=True)
'''
    need(len(cell.splitlines()) <= 25, 'Bootstrap exceeds 25 lines')
    return cell


def verify_bundle(root, bundle_sha, surface):
    root = Path(root)
    need(sha(root / RECEIPT) == bundle_sha, 'Remote bundle digest differs')
    receipt = read(root / RECEIPT)
    need(receipt['schema'] == 'exp166-package-v1' and receipt['surface'] == surface,
         'Wrong remote bundle')
    for name, rec in receipt['files'].items():
        safe(name)
        file = root / name
        need(file.is_file() and file.stat().st_size == rec['bytes'] and sha(file) == rec['sha256'],
             'Remote package input differs: ' + name)
    registration(root)
    return receipt


def runner_args(root, output, *, workers=2, pilot=False):
    root, output = Path(root), Path(output)
    p = read(root / MANIFEST)
    args = old.runner_argv(p, root=root, workers=workers, python=sys.executable)
    args[1] = str(root / RUNNER)
    def set_flag(flag, value):
        i = args.index(flag)
        args[i+1] = str(value)
    for flag, value in [('--protocol', root / MANIFEST), ('--tag', TAG),
                        ('--arch', ARCH), ('--norm', 'imagenet'),
                        ('--pretrained', 'timm-in22k'), ('--init-weights', root / WEIGHT)]:
        set_flag(flag, value)
    args += ['--output-root', str(output)]
    if pilot:
        args += ['--pilot-only']
    return args


def _remote_corpus(root, surface):
    from tools.colab.exp159_remote import ensure_dataset
    p = read(root / MANIFEST)
    if surface == 'colab':
        corpus = ensure_dataset(CORPUS_DS, Path('/content/exp166-corpus'))
    else:
        source = Path('/workspace/rsna-exp158-dino-strong-v1') / p['paths']['corpus']
        need(source.is_dir(), 'Expected reusable Vast corpus missing')
        corpus = root / 'corpus-hardlinks'
        need(not corpus.exists(), 'Vast corpus hardlink destination exists')
        corpus.mkdir()
        for item in source.iterdir():
            if item.is_file():
                os.link(item, corpus / item.name)
    e150.verify_corpus(corpus, root / e150.CORPUS_LOCAL,
                       read(root / e150.MANIFEST)['corpus_file_list'])
    target = root / p['paths']['corpus']
    target.parent.mkdir(parents=True, exist_ok=True)
    need(not target.exists() and not target.is_symlink(), 'Corpus target exists')
    target.symlink_to(corpus, target_is_directory=True)
    return corpus


def remote_run(surface, bundle_sha, root=ROOT):
    """B-only remote entry point. Pilot times one epoch before setting any wall cap."""
    root = Path(root).resolve()
    need(sys.platform.startswith('linux') and surface in ('colab', 'vast') and
         os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Wrong remote environment')
    if surface == 'colab':
        need(str(root) == '/content/exp166' and Path('/content').is_dir(), 'Wrong Colab workspace')
    else:
        need(str(root) == '/workspace/exp166', 'Wrong Vast workspace')
    verify_bundle(root, bundle_sha, surface)
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=30).stdout.strip().splitlines()
    need(len(smi) >= 1 and ('T4' if surface == 'colab' else '3090') in smi[0],
         'Target GPU absent')
    _remote_corpus(root, surface)
    import torch
    import timm
    import numpy as np
    need(torch.cuda.is_available(), 'CUDA unavailable')
    output = Path('/content/exp166-results' if surface == 'colab' else '/workspace/exp166-results')
    output.mkdir(parents=True, exist_ok=True)
    run = output / TAG
    if surface == 'colab' and not (run / CHECKPOINT).exists() and not (run / 'summary.json').exists():
        from tools.colab.exp159_remote import dataset_files, ensure_dataset
        try:
            files = dataset_files(RESULT_DS)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            files = []
        if files:
            ensure_dataset(RESULT_DS, run)
    if (run / 'summary.json').exists():
        verify_run(run, root)
        return {'state': 'ALREADY_COMPLETE', 'surface': surface}
    if (run / CHECKPOINT).is_file() and not (run / 'pilot.json').is_file():
        history = read(run / 'history.json')
        need(history and history[0]['epoch'] == 0, 'Published checkpoint lacks pilot epoch')
        first = history[0]
        put(run / 'pilot.json', {'epoch_seconds': first['epoch_seconds'],
            'peak_gpu_gib': first['peak_gpu_gib'],
            'projected_fit_seconds': first['epoch_seconds'] * 20})
    env = dict(os.environ, EXP166_REMOTE=surface, KNEE_WORKSPACE_ROOT=str(root),
               KNEE_REMOTE_RELAX_ENV='1', HF_HUB_OFFLINE='1', PYTHONUTF8='1',
               PYTHONDONTWRITEBYTECODE='1')
    for key in THREADS:
        env[key] = '1'
    start = time.monotonic()
    if not (run / 'pilot.json').is_file():
        cmd = runner_args(root, output, workers=2 if surface == 'colab' else 8, pilot=True)
        subprocess.run(cmd, cwd=root, env=env, check=True, timeout=1200)
    pilot = read(run / 'pilot.json')
    need(0 < pilot['epoch_seconds'] <= 300 and pilot['peak_gpu_gib'] < (14 if surface == 'colab' else 20),
         'Pilot time or memory gate failed')
    # Measured epoch 1, remaining 19 epochs, and a conservative evaluation
    # reserve. Reconnects may use the existing checkpoint and pilot receipt.
    wall_cap = int(max(900, 1.5 * 19 * pilot['epoch_seconds'] + 1200))
    put(run / 'execution-plan.json', {'surface': surface, 'pilot': pilot,
        'full_wall_cap_seconds': wall_cap, 'pilot_started_monotonic': start,
        'rule': '1.5 * 19 measured epoch seconds + 1200 evaluation reserve'})
    cmd = runner_args(root, output, workers=2 if surface == 'colab' else 8)
    subprocess.run(cmd, cwd=root, env=env, check=True, timeout=wall_cap)
    verify_run(run, root)
    publish_epoch(run, 20, surface)
    return {'state': 'COMPLETE', 'surface': surface, 'result_dataset': RESULT_DS if surface == 'colab' else None,
            'pilot_epoch_seconds': pilot['epoch_seconds'], 'wall_cap_seconds': wall_cap,
            'hardware': {'gpu': smi[0], 'python': sys.version.split()[0], 'torch': torch.__version__,
                         'timm': timm.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda}}


def verify_run(run, root=ROOT):
    root, run = Path(root), Path(run)
    p, _ = registration(root)
    need(run.name == TAG, 'Wrong run tag')
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                'oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'}
    inv = read(run / 'output-inventory.json')
    need(set(inv) == required, 'Incomplete output inventory')
    for name, record in inv.items():
        need(sha(run / name) == record['sha256'] and (run / name).stat().st_size == record['bytes'],
             'Output inventory differs: ' + name)
    summary, prov = read(run / 'summary.json'), read(run / 'provenance.json')
    args = summary['args']
    need(summary['selected_epoch_1based'] == 20 and summary['optimizer_updates'] == 8620 and
         summary['train'] == 3455 and summary['val'] == 888 and summary['rows'] == 888 and
         summary['skipped_updates'] <= 86 and len(summary['history']) == 20 and
         [h['epoch'] for h in summary['history']] == list(range(20)),
         'Fixed-final fit incomplete')
    need(all(args.get(k) == v for k, v in p['common_args'].items()) and args['seed'] == 42 and
         args['fold'] == 0 and args['tag'] == TAG and args['mode'] == 'full',
         'Recorded recipe differs')
    need(prov['initial_state_sha256'] == p['initial_state_sha256'] and
         prov['init_weights']['weights_sha256'] == p['init_weights']['sha256'] and
         prov.get('flip_tta') == 'coronal_axial_horizontal_mean_probability' and
         prov['train_ids_ordered_sha256'] == p['folds']['0']['ordered_ids']['train_sha256'] and
         prov['eval_ids_ordered_sha256'] == p['folds']['0']['ordered_ids']['val_sha256'],
         'Run provenance differs')
    from tools import exp115_vlm_eval as ev
    ids = read(root / p['folds']['0']['ordered_ids']['path'])['val']
    for name in ('oof_k36.csv', 'oof_k36_flip_tta.csv'):
        pred = ev.load_pred(run / name, ids)
        need(pred.shape == (888, 12), 'Wrong V888 prediction shape')
    return {name: inv[name]['sha256'] for name in required}


def collect(surface, source, root=ROOT):
    root, source = Path(root), Path(source)
    need(surface in ('colab', 'vast'), 'Unknown surface')
    run = source / TAG if (source / TAG).is_dir() else source
    hashes = verify_run(run, root)
    target = root / RESULT_DIR / surface / TAG
    need(not target.exists(), 'Collected output exists')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run, target)
    need(verify_run(target, root) == hashes, 'Collection changed')
    return {'state': 'COLLECTED_VALIDATED', 'surface': surface, 'path': str(target),
            'oof_flip_tta_sha256': hashes['oof_k36_flip_tta.csv']}


def download_results(out):
    """Local per-file Kaggle download; no whole-dataset 404 path or row printing."""
    from tools.colab.exp159_remote import parse_dataset_files
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run([sys.executable, '-m', 'kaggle', 'datasets', 'files', RESULT_DS,
                              '--csv', '--page-size', '200'], capture_output=True, text=True, check=True)
    files = parse_dataset_files(listing.stdout)
    need(files, 'Private result dataset listing empty')
    for name, size in files:
        safe(name)
        target = out / name
        if target.is_file() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', RESULT_DS,
                        '-f', name, '-p', str(target.parent), '--unzip', '-q', '-o'], check=True)
        need(target.is_file() and target.stat().st_size == size, 'Result download size differs: ' + name)
    return {'files': len(files), 'path': str(out)}


def screen_gate(primary, dino_h1):
    return (primary['mean_delta'] > 0 and primary['mean_ci95'][0] > 0 and
            primary['mean_delta'] > max(.0013, dino_h1['mean_delta']))


def screen(run, out, root=ROOT):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp160_dino_partner as e160
    root, run, out = Path(root), Path(run), Path(out)
    need(not out.exists(), 'Screen output already exists')
    verify_run(run, root)
    p = read(root / MANIFEST)
    ids = read(root / p['folds']['0']['ordered_ids']['path'])['val']
    need(len(ids) == 888, 'V888 roster changed')
    paths = {'ConvNeXt-T': run / 'oof_k36_flip_tta.csv',
             'RadR50': root / 'artifacts/experiments/EXP127_AUG1_S42_F0/oof_k36.csv',
             'DINO-base': root / '.local/exp160-collected-v1/BASE/oof_best.csv',
             'H1': root / 'artifacts/experiments/EXP149_H1_S42_F0/oof_k36.csv'}
    preds = {name: ev.load_pred(path, ids) for name, path in paths.items()}
    refs = {}
    for name, path in [('jev', e150.JEV), ('aquila', e150.AQUILA)]:
        soft = e150.table(root / path, ids)
        need(np.isfinite(soft).all(), 'Missing JEV/aquila label')
        y = (soft >= .5).astype(float)
        refs[name] = (y, np.ones(y.shape, bool))
    d = e150.table(root / e150.SETS['dread'], ids)
    refs['dread'] = (np.rint(d.astype(np.float32)).astype(float), np.isfinite(d))
    stack = np.stack([e150.table(root / path, ids) for path in e150.SETS.values()])
    valid = np.isfinite(stack).all(axis=0)
    binary = stack >= .5
    agree = valid & (binary.all(axis=0) | (~binary).all(axis=0))
    refs['consensus'] = (binary[0].astype(float), agree)
    standalone = {name: {ref: float(e160._metrics(y, pred, mask, ev).mean())
                         for ref, (y, mask) in refs.items()}
                  for name, pred in preds.items()}
    indices = ev.bootstrap_index(ev.component_groups(ids, root), 300, 20261003)
    pair_refs = {key: refs[key] for key in ('jev', 'aquila')}
    blends = {partner: e160._blend(preds['ConvNeXt-T'], preds[partner], pair_refs,
                                  indices, (.1, .2), ev)
              for partner in ('H1', 'RadR50')}
    dino_h1 = e160._blend(preds['DINO-base'], preds['H1'], pair_refs, indices, (.1,), ev)['0.1']
    primary = blends['H1']['0.1']
    advance = screen_gate(primary, dino_h1)
    ranks = {name: ev.pct_rank(preds[name]) for name in preds}
    correlation = {}
    for comparator in ('RadR50', 'DINO-base'):
        per_target = [float(np.corrcoef(ranks['ConvNeXt-T'][:, j], ranks[comparator][:, j])[0, 1])
                      for j in range(12)]
        need(np.isfinite(per_target).all(), 'Undefined rank correlation')
        correlation[comparator] = {'per_target': per_target, 'mean': float(np.mean(per_target))}
    result = {'schema': 'exp166-screen-v1', 'version': VERSION,
        'status': 'ADVANCE' if advance else 'STOP',
        'rule': 'H1 w0.1 mean delta >0, CI lower >0, point delta > max(0.0013, measured DINO-base H1 w0.1)',
        'standalone': standalone, 'rank_correlation': correlation,
        'blends': blends, 'dino_base_h1_w01': dino_h1,
        'primary': primary, 'historical_dino_base_h1_gain': .0013,
        'bootstrap': {'unit': 'component_id', 'draws': 300, 'seed': 20261003},
        'inputs_sha256': {name: sha(path) for name, path in paths.items()},
        'references_sha256': {name: sha(root / path) for name, path in
            [('jev', e150.JEV), ('aquila', e150.AQUILA),
             *[(key, value) for key, value in e150.SETS.items()]]}}
    out.mkdir(parents=True)
    put(out / 'result.json', result)
    return {'status': result['status'], 'primary_mean_delta': primary['mean_delta'],
            'primary_ci95': primary['mean_ci95'], 'dino_base_h1_mean_delta': dino_h1['mean_delta'],
            'result': str(out / 'result.json')}


def push_plan(out):
    r = verify_stage(out)
    need(r['surface'] == 'colab', 'Colab stage required')
    code = (Path(out) / 'code-dataset').as_posix()
    return '\n'.join([
        '# B confirms private dataset visibility, then runs one of:',
        f'.venv/Scripts/python.exe -X utf8 -m kaggle datasets create -p {code} --dir-mode zip',
        f'.venv/Scripts/python.exe -X utf8 -m kaggle datasets version -p {code} -m "EXP166 v1.0.0" --dir-mode zip',
        '# Paste the bootstrap cell from the command below into B\'s Colab T4 notebook:',
        f'py -3.12 {TOOL} bootstrap-cell --stage {Path(out).as_posix()}',
        '# After completion, download the private results dataset per file; whole-dataset CLI 2.2.4 may 404.',
        f'.venv/Scripts/python.exe {TOOL} download-results --out .local/exp166-download-v1',
        f'py -3.12 {TOOL} collect --surface colab --source .local/exp166-download-v1',
        f'py -3.12 {TOOL} screen --run {RESULT_DIR}/colab/{TAG} --out {SCREEN_DIR}',
    ])


def vast_plan(out):
    r = verify_stage(out)
    need(r['surface'] == 'vast', 'Vast stage required')
    return '\n'.join([
        '# B checks live 3090 identity, balance, corpus path and Active compute claim.',
        f'# Transfer {(Path(out) / "package" / ARCHIVE).as_posix()} to /workspace/{ARCHIVE}.',
        f'python -m zipfile -e /workspace/{ARCHIVE} /workspace/exp166',
        f'# Transfer {(Path(out) / "vast-queue.json").as_posix()} to /workspace/exp166/vast-queue.json.',
        '# Run one gpu_queue job on the 3090, after the actual GPU is idle:',
        'cd /workspace/exp166',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTHONUTF8=1 '
        'python tools/gpu_queue.py /workspace/exp166/vast-queue.json --foreground',
        '# Copy /workspace/exp166-results/EXP166_CONVNEXT_T_S42_F0 back, then:',
        f'py -3.12 {TOOL} collect --surface vast --source <download-directory>',
        f'py -3.12 {TOOL} screen --run {RESULT_DIR}/vast/{TAG} --out {SCREEN_DIR}',
    ])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('fetch-weights')
    sub.add_parser('prepare')
    for name in ('stage', 'bootstrap-cell', 'push-plan', 'vast-plan'):
        s = sub.add_parser(name)
        s.add_argument('--stage', required=True)
        if name == 'stage':
            s.add_argument('--surface', choices=('colab', 'vast'), required=True)
    r = sub.add_parser('remote-run')
    r.add_argument('--surface', choices=('colab', 'vast'), required=True)
    r.add_argument('--bundle-sha', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--surface', choices=('colab', 'vast'), required=True)
    c.add_argument('--source', required=True)
    d = sub.add_parser('download-results')
    d.add_argument('--out', required=True)
    s = sub.add_parser('screen')
    s.add_argument('--run', required=True)
    s.add_argument('--out', default=SCREEN_DIR)
    args = ap.parse_args(argv)
    if args.action == 'fetch-weights':
        value = fetch_weights()
    elif args.action == 'prepare':
        value = prepare()
    elif args.action == 'stage':
        value = stage(args.surface, args.stage)
    elif args.action == 'bootstrap-cell':
        value = bootstrap_cell(args.stage)
    elif args.action == 'push-plan':
        value = push_plan(args.stage)
    elif args.action == 'vast-plan':
        value = vast_plan(args.stage)
    elif args.action == 'remote-run':
        value = remote_run(args.surface, args.bundle_sha)
    elif args.action == 'collect':
        value = collect(args.surface, args.source)
    elif args.action == 'download-results':
        value = download_results(args.out)
    else:
        value = screen(args.run, args.out)
    print(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
