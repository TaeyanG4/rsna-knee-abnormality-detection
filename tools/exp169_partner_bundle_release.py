"""EXP169: stage a private trio + three-family partner, then serve on two T4s.

Only ``serve`` touches CUDA. ``stage`` and ``push-plan`` are offline and perform
no Kaggle operation. A submission still requires Taeyang's separate instruction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
EXP = ROOT / 'artifacts/experiments/EXP169_PARTNER_BUNDLE'
PROTOCOL = EXP / 'PROTOCOL.md'
DATASET = 'taeyangg4/knee-exp169-partner-bundle-assets'
KERNEL = 'taeyangg4/knee-exp169-partner-bundle'
SCHEMA = 'exp169-release-v1'
EXP170 = Path('artifacts/experiments/EXP170_CONVNEXT_T_L2')
K = 36
LIMIT_SECONDS = 7 * 3600
RESERVE_SECONDS = 900
QUEUE_MAX = 4
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
CODE = {
    'exp169_partner_bundle_release.py': 'tools/exp169_partner_bundle_release.py',
    'raptor_common.py': 'training/src/raptor_common.py',
    'raptor_dicom.py': 'training/src/raptor_dicom.py',
    'raptor_runtime.py': 'training/src/raptor_runtime.py',
    'raptor_radimagenet.py': 'training/src/raptor_radimagenet.py',
    'raptor_data.py': 'training/src/raptor_data.py',
    'exp160_base_cpu_pos.py': '.local/exp160-overlay-v1_1/BASE/training/adapters/exp160_base_cpu_pos.py',
    'exp160_base_patch_mil.py': '.local/exp160-overlay-v1_1/BASE/training/adapters/exp160_base_patch_mil.py',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_paths(root=ROOT):
    from tools.exp157_l2_advance import ensemble_tag
    root = Path(root)
    base = root / 'artifacts/experiments'
    runs = []
    for seed in (42, 43):
        for fold in range(5):
            runs.append(dict(family='l2', seed=seed, fold=fold,
                             path=base / ensemble_tag(seed, fold) / 'model_best.pt'))
    for fold in range(5):
        path = (root / '.local/exp160-collected-v1/BASE/model_best.pt' if fold == 0 else
                base / 'EXP164_DINO_BASE_FOLDS/outputs' /
                f'EXP164_DINO_BASE_S42_F{fold}' / 'model_best.pt')
        runs.append(dict(family='dino', seed=42, fold=fold, path=path))
    for fold in range(5):
        path = base / 'EXP170_CONVNEXT_T_L2/outputs/results' / f'EXP170_CONVNEXT_T_L2_S42_F{fold}' / 'model_best.pt'
        runs.append(dict(family='convnext', seed=42, fold=fold, path=path))
    return runs


def guard_attributes():
    """Local structural check for every callable used by the runtime."""
    import ast
    checks = {'raptor_common.py': {'build_model', 'select_centers', 'windows_from'},
              'raptor_dicom.py': {'DicomCorpus', 'DirectoryDicomSource'},
              'raptor_radimagenet.py': {'center3_windows_from', 'normalise'},
              'exp160_base_cpu_pos.py': {'install'},
              'exp160_base_patch_mil.py': {'install'}}
    for name, required in checks.items():
        source = (ROOT / CODE[name]).read_text(encoding='utf-8')
        parsed = ast.parse(source)
        actual = {n.name for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        need(required <= actual, f'Missing runtime guard attributes in {name}: {required - actual}')
    return {k: sorted(v) for k, v in checks.items()}


def guard_transitive_imports(root=ROOT):
    """Keep every project module imported by packaged serving code in CODE."""
    import ast
    root = Path(root)
    packaged = {Path(name).stem for name in CODE}
    dependencies = {}
    for name, rel in CODE.items():
        tree = ast.parse((root / rel).read_text(encoding='utf-8'))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        local = {module for module in imports if module.startswith(('raptor_', 'exp160_'))}
        need(local <= packaged, f'Unpackaged serving imports in {name}: {local - packaged}')
        dependencies[name] = sorted(local)
    return dependencies


def model_meta(ck, entry):
    family = entry['family']
    expected = {'l2': ('resnet50', 'radimagenet'),
                'dino': ('vit_base_patch14_dinov2.lvd142m', 'imagenet'),
                'convnext': ('convnext_tiny.fb_in22k_ft_in1k', 'imagenet')}
    arch, norm = expected[family]
    need(ck.get('arch') == arch and ck.get('res') == 224 and ck.get('lab') == TARGETS and
         ck.get('contract') == 'raptor64_336_c140_s02' and
         ck.get('window_policy') == 'within-slot-valid' and ck.get('norm') == norm,
         'Checkpoint serving metadata differs: ' + entry['tag'])
    need(ck.get('tag') == entry['tag'] and isinstance(ck.get('model'), dict) and ck['model'],
         'Checkpoint tag/model differs: ' + entry['tag'])
    if family == 'dino':
        need(ck.get('aggregation_policy') == 'dinov2-normalized-patch256-existing-mil-v1',
             'DINO patch-MIL aggregation differs')
    if family == 'l2':
        need(ck.get('input_policy') == 'center3', 'L2 center3 input differs')
    if family == 'convnext':
        need(ck.get('input_policy') == 'center3', 'ConvNeXt center3 input differs')


def checkpoint_entries(root=ROOT):
    """Verify training receipts and record original source hashes before export."""
    from tools import exp157_l2_release as l2
    root = Path(root)
    entries = []
    for item in source_paths(root):
        path = item['path']
        need(path.is_file(), 'Missing checkpoint: ' + str(path))
        f, seed, fold = item['family'], item['seed'], item['fold']
        if f == 'l2':
            verified = l2.checkpoint_entry(root, seed, fold)
            need(verified['sha256'] == sha(path), 'L2 inventory differs')
        elif f == 'convnext':
            receipt = read_json(root / EXP170 / 'outputs' / f'validated-fold{fold}.json')
            inventory = read_json(path.parent / 'output-inventory.json')
            need(receipt.get('status') == 'VALIDATED_SUCCESS' and receipt.get('fold') == fold and
                 receipt.get('tag') == path.parent.name and
                 receipt['output_inventory_sha256'] == sha(path.parent / 'output-inventory.json') and
                 receipt['output_sha256']['model_best.pt'] == sha(path) and
                 inventory['model_best.pt']['sha256'] == sha(path) and
                 inventory['model_best.pt']['bytes'] == path.stat().st_size,
                 'EXP170 validated checkpoint inventory differs')
        else:
            receipt = path.parent / ('collect-receipt.json' if fold == 0 else 'receipt.json')
            need(receipt.is_file(), 'DINO collect receipt missing')
            info = read_json(receipt)
            need(info.get('status') in ('COLLECTED_VALIDATED', 'PASS'), 'DINO fold not verified')
            if fold:
                need(info['outputs']['model_best.pt']['sha256'] == sha(path), 'DINO inventory differs')
            else:
                need(info.get('checkpoint_sha256') == sha(path), 'DINO fold-0 checkpoint differs')
        tag = 'EXP160_DINO_BASE' if f == 'dino' and fold == 0 else path.parent.name
        entries.append(dict(family=f, seed=seed, fold=fold, tag=tag,
                            path=f'models/{f}_s{seed}_f{fold}.pt',
                            source_sha256=sha(path), source=str(path.relative_to(root))))
    need(len(entries) == 20 and len({e['source_sha256'] for e in entries}) == 20,
         'Twenty distinct partner checkpoints required')
    return entries


def admitted_screen(root=ROOT):
    """Bind the release to EXP170's prospective label-parity admission gate."""
    root = Path(root)
    collection = read_json(root / EXP170 / 'outputs/collection.json')
    screen_path = root / EXP170 / 'screen/result.json'
    result = read_json(screen_path)
    pair = result['bundle_minus_l2']
    need(collection.get('status') == 'VERIFIED' and
         collection.get('completed_folds') == list(range(5)) and
         result.get('schema') == 'exp170-convnext-t-l2-v1' and
         result.get('protocol_sha256') == sha(root / EXP170 / 'PROTOCOL.md') and
         result.get('gate') == 'mean(JEV,consensus) delta >0, paired CI95 lower >0, Dread delta >= -0.003' and
         result.get('status', {}).get('status') == 'ADMIT' and
         pair['primary_mean_delta'] > 0 and pair['primary_mean_ci95'][0] > 0 and
         pair['references']['dread']['delta'] >= -.003 and
         result['oof_policy']['convnext'] == 'flip-TTA oof_k36_flip_tta.csv',
         'EXP170 label-parity bundle is not admitted')
    for fold in range(5):
        key = f'convnext_f{fold}'
        oof = root / EXP170 / 'outputs/results' / f'EXP170_CONVNEXT_T_L2_S42_F{fold}' / 'oof_k36_flip_tta.csv'
        need(result['input_sha256'][key] == sha(oof), 'EXP170 screened OOF hash differs')
    return screen_path, result


def verify_trio_averages(root, trio_manifest):
    """Reuse both authenticated averages; EXP143 is repackaged in the trio stage."""
    root = Path(root)
    for directory, filename, binding in (
        ('EXP155_S42v1_1_average_v12', 'exp155-v1_1-e14-e11-e12-oai-bn.pt',
         'average_receipt_sha256'),
        ('EXP143_S42v2_average_v101', 'exp143-v2-e14-e11-e12-bn.pt',
         'source_member_receipt_sha256'),
    ):
        folder = root / 'artifacts/releases' / directory
        receipt_path = folder / 'average-receipt.json'
        receipt = read_json(receipt_path)
        need(receipt.get('status') == 'VERIFIED_FIXED_SWA_BN' and
             receipt.get('checkpoint') == filename and
             sha(folder / filename) == receipt['checkpoint_sha256'] and
             sha(receipt_path) == trio_manifest[binding],
             'Authenticated trio average differs: ' + directory)
        if directory.startswith('EXP143'):
            need(receipt['checkpoint_sha256'] == trio_manifest['source_member_sha256'],
                 'EXP143 original average binding differs')
        else:
            need(receipt['checkpoint_sha256'] == trio_manifest['checkpoints'][filename]['sha256'],
                 'EXP155 staged average binding differs')


def _zip_unique(path, files):
    need(len(files) == len(set(files)), 'Duplicate archive entries')
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED) as archive:
        for name, source in sorted(files.items()):
            need(not name.startswith('/') and '..' not in Path(name).parts,
                 'Unsafe archive member')
            archive.write(source, name)
    with zipfile.ZipFile(path) as archive:
        need(archive.namelist() == sorted(files), 'Archive entries differ')
        need(len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate ZIP member')


def stage(trio_stage, out, root=ROOT, dataset=DATASET):
    """Build local private asset dataset and notebook. Does not upload or run."""
    from tools import exp155_own0949_oai_release_v12 as trio
    import torch
    root, out = Path(root), Path(out)
    need(not out.exists(), 'Choose a fresh EXP169 stage directory')
    screen_path, result = admitted_screen(root)
    trio_dir, trio_report, trio_manifest = trio.checked_stage(trio_stage)
    need('blend3' in trio_report['kernels'] and trio_manifest['dataset'] == trio_report['dataset'],
         'EXP155 blend3 stage is incomplete')
    verify_trio_averages(root, trio_manifest)
    need(dataset == DATASET, 'Frozen private dataset ID differs')
    need(projection()['projected_seconds'] <= LIMIT_SECONDS - RESERVE_SECONDS,
         'Planning projection exceeds seven-hour budget')
    guards = guard_attributes()
    imports = guard_transitive_imports(root)
    entries = checkpoint_entries(root)
    out.mkdir(parents=True)
    assets = out / 'assets'
    assets.mkdir()
    code_dir = assets / 'code'
    code_dir.mkdir()
    files = {}
    for name, rel in CODE.items():
        src = root / rel
        dst = code_dir / name
        shutil.copyfile(src, dst)
        files['code/' + name] = dst
    marker = assets / 'config/workspace.toml'
    marker.parent.mkdir()
    marker.write_text('# EXP169 isolated test-only serving root\n', encoding='utf-8')
    files['config/workspace.toml'] = marker
    for entry in entries:
        source = root / entry['source']
        ck = torch.load(source, map_location='cpu', weights_only=False)
        model_meta(ck, entry)
        permitted = ('arch', 'res', 'lab', 'contract', 'norm', 'window_policy',
                     'input_policy', 'aggregation_policy', 'tag')
        clean = {key: ck[key] for key in permitted if key in ck}
        clean['model'] = ck['model']
        clean['source_sha256'] = entry['source_sha256']
        dst = assets / entry['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clean, dst)
        entry['sha256'] = sha(dst)
        entry['bytes'] = dst.stat().st_size
        files[entry['path']] = dst
    manifest = dict(schema=SCHEMA, protocol_sha256=sha(root / 'artifacts/experiments/EXP169_PARTNER_BUNDLE/PROTOCOL.md'),
                    screen_sha256=sha(screen_path), screen_protocol_sha256=result['protocol_sha256'],
                    trio_manifest_sha256=trio_report['manifest_sha256'], trio_dataset=trio_report['dataset'],
                    dataset=dataset, kernel=KERNEL, k_eval=K, weights=dict(trio=.85, bundle=.15),
                    family_weights=dict(l2=1/3, dino=1/3, convnext=1/3),
                    source_code={name: sha(root / rel) for name, rel in CODE.items()},
                    guard_attributes=guards, transitive_imports=imports, entries=entries,
                    files={name: dict(sha256=sha(path), bytes=path.stat().st_size)
                           for name, path in files.items()},
                    serving_projection=projection())
    manifest_file = assets / 'release-manifest.json'
    write_json(manifest_file, manifest)
    files['release-manifest.json'] = manifest_file
    ds = out / 'dataset'
    ds.mkdir()
    archive = ds / 'exp169-assets.zip.bin'
    _zip_unique(archive, files)
    write_json(ds / 'dataset-metadata.json', dict(title=dataset.split('/')[1], id=dataset,
               isPrivate=True, licenses=[{'name': 'other'}],
               description='Private own L2 RadR50, DINO-base and L2 ConvNeXt-T inference weights and code; no raw OAI or training data.'))
    kernel = out / 'kernel'
    kernel.mkdir()
    nb = notebook(trio_dir, dataset, sha(archive), sha(manifest_file))
    write_json(kernel / 'release.ipynb', nb)
    metadata = read_json(trio_dir / 'blend3/kernel-metadata.json')
    metadata.update(id=KERNEL, title=KERNEL.split('/')[1],
                    dataset_sources=[dataset, trio_report['dataset'], trio.PUBLIC_DATASET])
    write_json(kernel / 'kernel-metadata.json', metadata)
    write_json(out / 'build-receipt.json', dict(status='STAGED_NOT_UPLOADED', dataset=dataset,
               kernel=KERNEL, archive_sha256=sha(archive), manifest_sha256=sha(manifest_file),
               trio_manifest_sha256=trio_report['manifest_sha256'],
               notebook_sha256=sha(kernel / 'release.ipynb'),
               checkpoint_count=len(entries), projection=projection(), gpu_runs=0,
               uploads=0, submissions=0))
    return dict(status='STAGED_NOT_UPLOADED', out=str(out), dataset=dataset, kernel=KERNEL,
                archive_sha256=sha(archive), projection=projection())


def notebook(trio_stage, dataset, archive_sha, manifest_sha):
    """Execute the already staged EXP155 blend3 cell, redirecting only its work path."""
    trio_stage = Path(trio_stage)
    source_nb = read_json(trio_stage / 'blend3/release.ipynb')
    source = source_nb['cells'][-1]['source']
    old = "_work = Path('/kaggle/working')"
    need(source.count(old) == 1 and source.count('main()') == 1,
         'Frozen EXP155 blend3 notebook entry changed')
    source = source.replace(old, "_work = Path('/kaggle/working/exp169-trio')\n_work.mkdir(exist_ok=False)")
    slug = dataset.split('/')[1]
    suffix = f'''\n# Partner only starts after a completed trio artifact. Root CSV is written by the wrapper.
import os as _os
_os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
_partner_mounts = [Path('/kaggle/input/datasets/{dataset}'), Path('/kaggle/input/{slug}')]
_partner_archive = next((p/'exp169-assets.zip.bin' for p in _partner_mounts if (p/'exp169-assets.zip.bin').is_file()), None)
if _partner_archive is None: raise FileNotFoundError('Attach the private EXP169 partner asset dataset')
if _hash(_partner_archive) != {archive_sha!r}: raise ValueError('EXP169 archive SHA mismatch')
_partner = Path('/kaggle/working/exp169-partner')
_partner.mkdir(exist_ok=False)
with zipfile.ZipFile(_partner_archive) as _z:
    _names = _z.namelist()
    if len(_names) != len(set(_names)) or any(n.startswith('/') or '..' in Path(n).parts for n in _names):
        raise ValueError('EXP169 archive entries invalid')
    _z.extractall(_partner)
if _hash(_partner/'release-manifest.json') != {manifest_sha!r}: raise ValueError('EXP169 manifest SHA mismatch')
sys.path.insert(0, str(_partner/'code'))
import exp169_partner_bundle_release as _e169
_os.environ['KNEE_WORKSPACE_ROOT'] = str(_partner)
_e169.verify_assets(_partner, _e169.read_json(_partner/'release-manifest.json'))
import torch as _torch
_e169.write_json(_work/'exp169-trio-resources.json', {{
    'seconds': _e169.read_json(_work/'receipt.json')['seconds'],
    'ram': _e169._memory(),
    'gpu_peak_bytes': [int(_torch.cuda.max_memory_allocated(i)) for i in range(2)]}})
_pilot_dir = Path('/kaggle/working/exp169-pilot')
_e169.pilot(_partner, Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), _pilot_dir,
            studies=min(40, len(_e169.validate_csv(_work/'submission.csv',
                Path('/kaggle/input/competitions/rsna-knee-abnormality-detection/sample_submission.csv'))[0])))
import shutil as _shutil
_shutil.copyfile(_pilot_dir/'partner-pilot.json', _partner/'partner-pilot.json')
_e169.serve(_partner, _work, Path('/kaggle/working'),
            Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'))
'''
    # If staging/pilot fails after the trio ran, publish only the trio and mark it.
    suffix = '\ntry:\n' + ''.join('    ' + line + '\n' for line in suffix.strip().splitlines()) + '''except BaseException as _bundle_error:
    import shutil as _shutil, json as _json, hashlib as _hashlib
    _fallback = Path('/kaggle/working/submission.csv')
    if _fallback.exists(): _fallback.unlink()
    _shutil.copyfile(_work/'submission.csv', _fallback)
    def _digest(p):
        with p.open('rb') as f: return _hashlib.file_digest(f, 'sha256').hexdigest()
    (_fallback.parent/'exp169-executed-candidate.json').write_text(_json.dumps({
        'schema': 'exp169-executed-candidate-v1', 'status': 'FAILED_BUNDLE',
        'candidate': 'TRIO_ONLY_FALLBACK', 'executed_arms': ['EXP155_BLEND3'],
        'reason': type(_bundle_error).__name__ + ': ' + str(_bundle_error),
        'trio_sha256': _digest(_work/'submission.csv'), 'submission_sha256': _digest(_fallback)
    }, indent=2) + '\\n')
    print('EXP169 FAILED_BUNDLE:', type(_bundle_error).__name__, str(_bundle_error), flush=True)
finally:
    _exp169_signal.alarm(0)
'''
    # The EXP155 cell imports torch only after CUBLAS is set here.
    prelude = ("import os, signal as _exp169_signal, time as _exp169_time\n"
               "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
               "_exp169_started = _exp169_time.monotonic()\n"
               "def _exp169_deadline(_signum, _frame): raise TimeoutError('EXP169 seven-hour deadline guard')\n"
               "_exp169_signal.signal(_exp169_signal.SIGALRM, _exp169_deadline)\n"
               f"_exp169_signal.alarm({LIMIT_SECONDS - RESERVE_SECONDS})\n")
    source = prelude + source + suffix
    compile(source, 'EXP169 release cell', 'exec')
    source_nb['cells'][-1]['source'] = source
    source_nb['cells'][0]['source'] = ('# EXP169 fixed CoAtNet trio plus equal-third partner bundle\n\n'
       'Fixed final 0.85 rank(EXP155 blend3) + 0.15 rank(L2 RadR50/DINO-base/EXP170 L2 ConvNeXt-T bundle). '
       'K36 all partner families; one private 2xT4 kernel. Any bundle failure is recorded as FAILED_BUNDLE. '
       'No submission without Taeyang chat instruction.')
    return source_nb


def projection(n=1300):
    need(isinstance(n, int) and n > 0, 'Positive cohort required')
    trio = 1.5 * 3 * 1.52 * n + 300  # EXP152 conservative blend3 proxy.
    partner = 2.0 * n + 900  # planning only; must be replaced by T4 pilot measurements.
    total = trio + partner
    return dict(studies=n, gpus='2xT4', trio_seconds=trio, partner_seconds=partner,
                projected_seconds=total, max_seconds=LIMIT_SECONDS,
                basis='EXP152 conservative 3-arm proxy plus 2.0 s/study partner planning allowance; NOT a full-cohort measurement',
                qualification='Requires actual T4 pilot and full-cohort timing/RAM/VRAM; never infer admission from this planning projection')


def rank(a):
    import numpy as np
    from scipy.stats import rankdata
    a = np.asarray(a, dtype=np.float64)
    need(a.ndim == 2 and a.shape[1] == 12 and len(a) > 0 and np.isfinite(a).all(),
         'Invalid family rank input')
    return (rankdata(a, axis=0, method='average') - 1) / max(1, len(a)-1)


def aggregate(scores):
    """Global family ranks and exact fixed final weight, with no rank re-tuning."""
    import numpy as np
    need(set(scores) == {'l2', 'dino', 'convnext'}, 'Incomplete partner families')
    for name, value in scores.items():
        a = np.asarray(value, np.float64)
        need(a.ndim == 3 and a.shape[2] == 12 and a.shape[0] == (10 if name == 'l2' else 5)
             and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(),
             'Invalid family probabilities: ' + name)
    shapes = {np.asarray(value).shape[1] for value in scores.values()}
    need(len(shapes) == 1, 'Partner cohort shapes differ')
    l2 = np.asarray(scores['l2'])
    l2_fold_seed_mean = (l2[:5] + l2[5:]) / 2
    family = {'l2': rank(l2_fold_seed_mean.mean(0)),
              'dino': rank(np.asarray(scores['dino']).mean(0)),
              'convnext': rank(np.asarray(scores['convnext']).mean(0))}
    bundle = rank(sum(rank(x) for x in family.values()) / 3)
    return bundle, family


def final_values(trio, bundle):
    import numpy as np
    trio, bundle = np.asarray(trio, np.float64), np.asarray(bundle, np.float64)
    need(trio.shape == bundle.shape and trio.ndim == 2 and trio.shape[1] == 12 and
         np.isfinite(trio).all() and np.isfinite(bundle).all() and
         ((trio >= 0) & (trio <= 1)).all() and ((bundle >= 0) & (bundle <= 1)).all(),
         'Invalid trio/bundle values')
    final = .85 * rank(trio) + .15 * rank(bundle)
    need(np.isfinite(final).all() and ((final >= 0) & (final <= 1)).all(), 'Final blend invalid')
    return final


def validate_csv(path, sample):
    import numpy as np
    with Path(sample).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [UID, *TARGETS], 'Sample submission columns differ')
        ids = [row[UID] for row in reader]
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [UID, *TARGETS], 'Submission columns differ')
        rows = list(reader)
    need(ids and len(set(ids)) == len(ids) and [r[UID] for r in rows] == ids,
         'Submission IDs/order differ')
    values = np.asarray([[float(row[t]) for t in TARGETS] for row in rows], np.float64)
    need(values.shape == (len(ids), 12) and np.isfinite(values).all() and
         ((values >= 0) & (values <= 1)).all(), 'Submission values invalid')
    return ids, values


def write_csv(path, ids, values, sample):
    import numpy as np
    path = Path(path)
    need(not path.exists(), 'Refuse overwrite: ' + str(path))
    values = np.asarray(values)
    need(values.shape == (len(ids), 12), 'Output matrix shape differs')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([UID, *TARGETS])
        writer.writerows([[uid, *[format(float(v), '.10g') for v in row]]
                          for uid, row in zip(ids, values)])
    validate_csv(path, sample)


def plane_flags(centers, spec):
    """EXP124 plane-aware coronal/axial mirror rule; no sagittal flips."""
    need(isinstance(spec.get('slots'), list), 'Missing RAPTOR64 slots')
    spans, start = [], 0
    for plane, _fluid, count in spec['slots']:
        need(plane in ('Sagittal', 'Coronal', 'Axial') and count >= 3, 'Invalid plane slot')
        spans.append((plane, start, start + count))
        start += count
    need(start == spec['depth'] and {x[0] for x in spans} == {'Sagittal', 'Coronal', 'Axial'},
         'RAPTOR64 plane coverage differs')
    flags = []
    for c in centers:
        if c == -1:
            flags.append(False)
            continue
        match = [p for p, lo, hi in spans if lo + 1 <= c < hi - 1]
        need(len(match) == 1, 'Window crosses a plane boundary')
        flags.append(match[0] in ('Coronal', 'Axial'))
    return flags


def _memory():
    try:
        import resource
        return dict(maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except ImportError:
        return dict(maxrss_kib=None)


def _load_models(work, entries, device):
    import torch
    from raptor_common import build_model
    from exp160_base_cpu_pos import install as install_pos
    from exp160_base_patch_mil import install as install_patch
    models = []
    for entry in entries:
        path = work / entry['path']
        need(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
             'Partner weight differs')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model_meta(ck, entry)
        need(ck.get('source_sha256') == entry['source_sha256'],
             'Partner source checkpoint binding differs')
        model = build_model(ck['arch'], pretrained=False)
        if entry['family'] == 'dino':
            install_pos(model.backbone)
            install_patch(model)
        model.load_state_dict(ck['model'], strict=True)
        models.append(model.eval().to(device))
        del ck
    return models


def _infer_one(model, x, device, dino=False):
    import torch
    tensor = torch.from_numpy(x).to(device, non_blocking=False)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        if dino:
            # The patch-MIL head attends jointly over all K36*256 patch tokens.
            # Encode windows in chunks, then apply the identical head once.
            chunks = [model.encode(tensor[i:i+4][None]) for i in range(0, len(tensor), 4)]
            logits = model.head(torch.cat(chunks, dim=1))
        else:
            logits = model(tensor[None])
    need(tuple(logits.shape) == (1, 12) and bool(torch.isfinite(logits).all()),
         'Invalid partner logits')
    result = torch.sigmoid(logits.float())[0].cpu().numpy()
    need(bool(torch.isfinite(torch.as_tensor(result)).all()), 'Invalid partner probabilities')
    return result


def partner_scores(work, competition, ids, manifest, deadline):
    """One bounded CPU producer and two GPU consumers; merge all shards before ranks."""
    import queue
    import numpy as np
    import torch
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    from raptor_common import select_centers, windows_from
    from raptor_radimagenet import center3_windows_from, normalise
    from raptor_runtime import normalize_input
    need(torch.cuda.device_count() == 2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)),
         'Exactly two visible T4 GPUs required')
    torch.set_num_threads(1)  # EXP160 CPU position interpolation contract.
    torch.use_deterministic_algorithms(True)
    corpus = DicomCorpus(competition / 'test.csv', competition / 'test_series.csv',
                         DirectoryDicomSource(competition / 'test_series'))
    need(set(ids) <= set(corpus.ids), 'Partner test cohort differs')
    n = len(ids)
    entries = manifest['entries']
    scores = {name: np.full((10 if name == 'l2' else 5, n, 12), np.nan, np.float32)
              for name in ('l2', 'dino', 'convnext')}
    work_queue = queue.Queue(maxsize=QUEUE_MAX)
    loaded_barrier = threading.Barrier(2)
    errors, lock = [], threading.Lock()
    timings = dict(decode_seconds=0., prepare_seconds=0., inference_seconds=0.,
                   study_seconds=[], decode_study_seconds=[], prepare_study_seconds=[],
                   gpu_peak_bytes=[0, 0], completed=[0, 0])
    completed = set()

    def producer():
        try:
            for i, uid in enumerate(ids):
                need(time.monotonic() < deadline, 'Partner producer deadline exceeded')
                started = time.monotonic()
                volume, mask = corpus.volume(uid), corpus.mask(uid)
                decoded = time.monotonic()
                centers = select_centers(mask, corpus.depth, K, corpus.spec, 'within-slot-valid')
                flags = plane_flags(centers, corpus.spec)
                center3 = center3_windows_from(volume, centers, 224)
                triplet = windows_from(volume, centers, 224)
                payload = (i, normalise(center3, 'radimagenet'),
                           normalize_input(center3, 'imagenet'),
                           normalize_input(triplet, 'imagenet'), flags)
                prepared = time.monotonic()
                with lock:
                    timings['decode_seconds'] += decoded - started
                    timings['prepare_seconds'] += prepared - decoded
                    timings['decode_study_seconds'].append(decoded - started)
                    timings['prepare_study_seconds'].append(prepared - decoded)
                work_queue.put(payload)
        except BaseException as exc:
            with lock:
                errors.append('producer: ' + type(exc).__name__ + ': ' + str(exc))
        finally:
            for _ in range(2):
                work_queue.put(None)

    def consumer(gpu):
        device = f'cuda:{gpu}'
        active = False
        try:
            torch.cuda.set_device(gpu)
            torch.cuda.reset_peak_memory_stats(gpu)
            own = _load_models(work, entries, device)
            loaded_barrier.wait(timeout=600)
            while True:
                job = work_queue.get()
                if job is None:
                    work_queue.task_done()
                    break
                active = True
                i, rad, img, triplet, flags = job
                started = time.monotonic()
                need(started < deadline, 'Partner GPU deadline exceeded')
                indices = np.asarray(flags, bool)
                for m, entry in enumerate(entries):
                    family = entry['family']
                    x = triplet if family == 'dino' else rad if family == 'l2' else img
                    plain = _infer_one(own[m], x, device, family == 'dino')
                    if family == 'dino':
                        value = plain
                    else:
                        mirror = x.copy()
                        mirror[indices] = mirror[indices, :, :, ::-1]
                        value = (plain + _infer_one(own[m], mirror, device)) / 2
                    slot = entry['seed'] == 43 and family == 'l2'
                    row = entry['fold'] + (5 if slot else 0)
                    scores[family][row, i] = value
                ended = time.monotonic()
                with lock:
                    need(i not in completed, 'Duplicate partner study')
                    completed.add(i)
                    timings['completed'][gpu] += 1
                    timings['inference_seconds'] += ended - started
                    timings['study_seconds'].append(ended - started)
                work_queue.task_done()
                active = False
            timings['gpu_peak_bytes'][gpu] = int(torch.cuda.max_memory_allocated(gpu))
        except BaseException as exc:
            loaded_barrier.abort()
            with lock:
                errors.append(f'gpu{gpu}: {type(exc).__name__}: {exc}')
            if active:
                work_queue.task_done()
            # Drain to allow the bounded producer to terminate after a worker error.
            while True:
                job = work_queue.get()
                work_queue.task_done()
                if job is None:
                    break

    workers = [threading.Thread(target=consumer, args=(i,), daemon=True) for i in range(2)]
    for worker in workers:
        worker.start()
    produce = threading.Thread(target=producer, daemon=True)
    produce.start()
    produce.join()
    work_queue.join()
    for worker in workers:
        worker.join()
    need(not errors, 'Partner failed: ' + repr(errors))
    need(completed == set(range(n)) and all(c > 0 for c in timings['completed']) and
         all(np.isfinite(a).all() for a in scores.values()),
         'Incomplete partner shards')
    timings['ram'] = _memory()
    for name in ('study_seconds', 'decode_study_seconds', 'prepare_study_seconds'):
        timings['p95_' + name] = float(np.percentile(timings.pop(name), 95))
    return scores, timings


def _extract(dataset, work, manifest):
    need(not work.exists(), 'Fresh extraction directory required')
    archive = Path(dataset) / 'exp169-assets.zip.bin'
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) == len(manifest['files']) + 1 and
             set(names) == set(manifest['files']) | {'release-manifest.json'},
             'EXP169 archive inventory differs')
        for name in names:
            need(not name.startswith('/') and '..' not in Path(name).parts,
                 'Unsafe EXP169 archive entry')
        z.extractall(work)
    for name, info in manifest['files'].items():
        path = work / name
        need(path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
             'EXP169 extracted asset differs: ' + name)
    need((work / 'config/workspace.toml').is_file(), 'Workspace config missing')
    return work


def verify_assets(work, manifest):
    work = Path(work)
    need(manifest.get('schema') == SCHEMA and len(manifest.get('entries', [])) == 20,
         'EXP169 release manifest differs')
    for name, info in manifest['files'].items():
        path = work / name
        need(path.is_file() and path.stat().st_size == info['bytes'] and
             sha(path) == info['sha256'], 'EXP169 asset differs: ' + name)
    need((work / 'config/workspace.toml').is_file(), 'Missing workspace config')
    return True


def serve(work, trio_work, output, competition, *, clock=time.monotonic,
          partner=partner_scores):
    """Publish the fixed candidate, or a visibly failed trio-only fallback."""
    import numpy as np
    work, trio_work, output, competition = map(Path, (work, trio_work, output, competition))
    manifest = read_json(work / 'release-manifest.json')
    need(manifest.get('schema') == SCHEMA and manifest.get('k_eval') == K and
         manifest.get('weights') == {'trio': .85, 'bundle': .15} and
         len(manifest.get('entries', [])) == 20, 'EXP169 manifest differs')
    need(sha(work / 'code/exp169_partner_bundle_release.py') ==
         manifest['files']['code/exp169_partner_bundle_release.py']['sha256'],
         'Executing EXP169 source differs')
    sample = competition / 'sample_submission.csv'
    ids, trio = validate_csv(trio_work / 'submission.csv', sample)
    need((trio_work / 'receipt.json').is_file() and
         read_json(trio_work / 'receipt.json')['status'] == 'COMPLETE_INFERENCE',
         'EXP155 trio did not finish')
    start = clock()
    estimate = projection(len(ids))
    trio_seconds = read_json(trio_work / 'receipt.json')['seconds']
    # Before the first T4 full run, a pilot rate must be supplied in a receipt.
    pilot = work / 'partner-pilot.json'
    need(pilot.is_file(), 'Measured two-T4 partner pilot receipt required before full cohort')
    p = read_json(pilot)
    remaining = LIMIT_SECONDS - trio_seconds - p.get('elapsed_seconds', math.inf) - RESERVE_SECONDS
    need(p.get('status') == 'PASS' and p.get('gpus') == 2 and
         p.get('studies', 0) >= min(30, len(ids)) and
         p.get('projected_full_seconds', math.inf) <= remaining and
         p.get('peak_vram_bytes', math.inf) < 15 * 2**30 and
         p.get('peak_ram_bytes', math.inf) < 27 * 2**30,
         'Measured partner pilot does not fit the seven-hour/memory gate')
    deadline = start + remaining
    receipt = dict(schema='exp169-executed-candidate-v1', status='RUNNING',
                   candidate='TRIO_PLUS_BUNDLE', executed_arms=['EXP155_BLEND3'],
                   studies=len(ids), qualification='VISIBLE_ONLY' if len(ids) < 30 else 'FULL_COHORT',
                   weights=manifest['weights'], projection=estimate,
                   trio_seconds=trio_seconds, pilot=p, stages={
                       'trio': dict(status='PASS', **read_json(trio_work / 'exp169-trio-resources.json'))
                       if (trio_work / 'exp169-trio-resources.json').is_file()
                       else dict(status='PASS', seconds=trio_seconds),
                       'pilot': dict(status='PASS', seconds=p['elapsed_seconds'],
                                     ram=p['peak_ram_bytes'], gpu_peak_bytes=p['peak_vram_bytes'])},
                   trio_sha256=sha(trio_work / 'submission.csv'))
    try:
        begun = clock()
        scores, metrics = partner(work, competition, ids, manifest, deadline)
        receipt['stages']['partner'] = dict(status='PASS', seconds=clock()-begun, **metrics)
        with (output / 'partner-raw.npz').open('xb') as stream:
            np.savez(stream, ids=np.asarray(ids), **scores)
        receipt['partner_raw_sha256'] = sha(output / 'partner-raw.npz')
        parity_started = clock()
        need(sha(p['scores_path']) == p['scores_sha256'], 'Pilot raw score hash differs')
        with np.load(p['scores_path'], allow_pickle=False) as reference:
            pilot_ids = reference['ids'].tolist()
            need(len(pilot_ids) == p['studies'] and len(set(pilot_ids)) == len(pilot_ids),
                 'Pilot parity IDs differ')
            index = {uid: i for i, uid in enumerate(ids)}
            need(set(pilot_ids) <= set(index), 'Pilot parity cohort differs')
            check_ix = [index[uid] for uid in pilot_ids]
            parity = {name: float(np.max(np.abs(scores[name][:, check_ix] - reference[name])))
                      for name in ('l2', 'dino', 'convnext')}
        need(all(delta <= 2e-5 for delta in parity.values()), 'Pilot/full partner parity failed')
        receipt['stages']['parity'] = dict(status='PASS', seconds=clock()-parity_started,
                                         ram=_memory(), gpu_peak_bytes=metrics['gpu_peak_bytes'],
                                         max_abs=parity, tolerance=2e-5)
        begun = clock()
        bundle, families = aggregate(scores)
        need(clock() < deadline, 'Partner/global rank exceeded deadline')
        final = final_values(trio, bundle)
        write_csv(output / 'submission.csv', ids, final, sample)
        receipt['stages']['aggregate_output'] = dict(status='PASS', seconds=clock()-begun,
            ram=_memory(), gpu_peak_bytes=metrics['gpu_peak_bytes'])
        need(trio_seconds + p['elapsed_seconds'] + clock() - start <= LIMIT_SECONDS - RESERVE_SECONDS,
             'Full-cohort seven-hour deadline exceeded')
        receipt['status'] = 'PASS_BUNDLE'
        receipt['executed_arms'] += ['L2_RADR50_10', 'DINO_BASE_5', 'EXP170_L2_CONVNEXT_T_5']
        receipt['family_rank_sha256'] = {name: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                                         for name, x in families.items()}
        receipt['bundle_rank_sha256'] = hashlib.sha256(np.ascontiguousarray(bundle).tobytes()).hexdigest()
    except BaseException as exc:
        # A root CSV may be used only when it is clearly marked as trio-only.
        receipt['status'] = 'FAILED_BUNDLE'
        receipt['candidate'] = 'TRIO_ONLY_FALLBACK'
        receipt['stages']['partner'] = dict(status='FAILED', error=type(exc).__name__ + ': ' + str(exc))
        if (output / 'submission.csv').exists():
            (output / 'submission.csv').unlink()
        shutil.copyfile(trio_work / 'submission.csv', output / 'submission.csv')
        validate_csv(output / 'submission.csv', sample)
    finally:
        receipt['elapsed_seconds'] = trio_seconds + p['elapsed_seconds'] + clock() - start
        receipt['ram'] = _memory()
        if (output / 'submission.csv').is_file():
            receipt['submission_sha256'] = sha(output / 'submission.csv')
        write_json(output / 'exp169-executed-candidate.json', receipt)
    return receipt


def pilot(work, competition, out, studies=40):
    """Explicit two-T4 pilot; never writes a submission."""
    import numpy as np
    work, competition, out = Path(work), Path(competition), Path(out)
    need(not out.exists(), 'Fresh pilot output required')
    manifest = read_json(work / 'release-manifest.json')
    sample = competition / 'sample_submission.csv'
    with sample.open(encoding='utf-8-sig', newline='') as stream:
        ids = [row[UID] for row in csv.DictReader(stream)]
    need(min(30, len(ids)) <= studies <= len(ids) and len(ids) >= 3,
         'Pilot needs 30+ studies for a full cohort or all three visible studies')
    # Cover both long-series tails and the whole file order without reading labels.
    from collections import Counter
    with (competition / 'test_series.csv').open(encoding='utf-8-sig', newline='') as stream:
        series_counts = Counter(row[UID] for row in csv.DictReader(stream))
    tail_count = min(20, studies // 2)
    tails = sorted(ids, key=lambda uid: (-series_counts[uid], ids.index(uid)))[:tail_count]
    spaced = [ids[min(len(ids)-1, int(i * len(ids) / max(1, studies-tail_count)))]
              for i in range(studies-tail_count)]
    selected = list(dict.fromkeys(tails + spaced))
    for uid in ids:
        if len(selected) >= studies:
            break
        if uid not in selected:
            selected.append(uid)
    started = time.monotonic()
    scores, metrics = partner_scores(work, competition, selected, manifest, started + 3600)
    need(all(np.isfinite(x).all() for x in scores.values()), 'Pilot arm values invalid')
    elapsed = time.monotonic()-started
    projected = 1.5 * elapsed * len(ids) / studies + 600
    receipt = dict(status='PASS', gpus=2, studies=studies, full_cohort=len(ids),
                   elapsed_seconds=elapsed, projected_full_seconds=projected,
                   peak_vram_bytes=max(metrics['gpu_peak_bytes']),
                   peak_ram_bytes=(metrics['ram']['maxrss_kib'] or 0) * 1024,
                   p95_study_seconds=metrics['p95_study_seconds'],
                   p95_decode_seconds=metrics['p95_decode_study_seconds'],
                   selection='highest series-count tails plus evenly spaced cohort IDs',
                   selected_ids_sha256=hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
                   metrics=metrics)
    out.mkdir(parents=True)
    with (out / 'pilot-scores.npz').open('xb') as stream:
        np.savez(stream, ids=np.asarray(selected), **scores)
    receipt['scores_path'] = str(out / 'pilot-scores.npz')
    receipt['scores_sha256'] = sha(out / 'pilot-scores.npz')
    write_json(out / 'partner-pilot.json', receipt)
    return receipt


def push_plan(stage_dir):
    """Return exact private dataset/kernel commands; never execute them."""
    stage_dir = Path(stage_dir)
    receipt = read_json(stage_dir / 'build-receipt.json')
    need(receipt['status'] == 'STAGED_NOT_UPLOADED' and
         sha(stage_dir / 'dataset/exp169-assets.zip.bin') == receipt['archive_sha256'],
         'EXP169 stage differs')
    venv = ROOT / '.venv/Scripts/python.exe'
    return dict(status='PLAN_ONLY', commands=[
        f'Push-Location -LiteralPath "{stage_dir / "dataset"}"',
        f'& "{venv}" -m kaggle datasets create -p .',
        'Pop-Location',
        f'& "{venv}" -m kaggle kernels push -p "{stage_dir / "kernel"}" --accelerator NvidiaTeslaT4',
        f'& "{venv}" -m kaggle kernels output {KERNEL} -p "{stage_dir / "kernel-output"}"'],
        note='Verify private dataset, quota, Public3 parity, full-cohort timing and PASS_BUNDLE receipt; submission only on Taeyang instruction.')


def parity_plan(stage_dir, out, root=ROOT):
    """Freeze the existing visible-test three-arm reference before kernel push."""
    stage_dir, out, root = Path(stage_dir), Path(out), Path(root)
    need(not out.exists(), 'Fresh parity plan directory required')
    stage_receipt = read_json(stage_dir / 'build-receipt.json')
    need(stage_receipt['status'] == 'STAGED_NOT_UPLOADED' and
         sha(stage_dir / 'dataset/exp169-assets.zip.bin') == stage_receipt['archive_sha256'],
         'EXP169 stage differs before parity plan')
    ref = root / 'artifacts/releases/EXP155_S42v1_1_stage_v12/blend3-t4-output'
    receipt = read_json(ref / 'receipt.json')
    need(receipt.get('status') == 'COMPLETE_INFERENCE', 'Trio reference is incomplete')
    for name in ('raw.npz', 'inputs.json', 'binding.json', 'submission.csv'):
        need(sha(ref / name) == receipt['files'][name], 'Trio reference hash differs: ' + name)
    out.mkdir(parents=True)
    plan = dict(schema='exp169-parity-plan-v1', status='FROZEN_NOT_RUN',
                stage_archive_sha256=stage_receipt['archive_sha256'],
                stage_notebook_sha256=stage_receipt['notebook_sha256'],
                reference=str(ref), reference_files={name: receipt['files'][name] for name in
                    ('raw.npz', 'inputs.json', 'binding.json', 'submission.csv')},
                arms=['own', 'public', 'exp143'],
                tolerance=dict(max_abs=2e-3, mean_abs=5e-4),
                partner='Compare per-fold raw probabilities against a locally computed visible-test reference when available; require identical IDs and all 20 arms.',
                limitation='Three visible studies establish trio serving parity only; full-cohort timing and partner parity require separate receipts.')
    write_json(out / 'parity-plan.json', plan)
    return dict(status=plan['status'], plan_sha256=sha(out / 'parity-plan.json'), out=str(out))


def compare_parity(plan_dir, kernel_output, out, partner_local=None):
    """Compare downloaded kernel arms with SHA-bound local visible-test arms."""
    import numpy as np
    plan_dir, kernel_output, out = Path(plan_dir), Path(kernel_output), Path(out)
    need(not out.exists(), 'Fresh parity result required')
    plan = read_json(plan_dir / 'parity-plan.json')
    need(plan.get('schema') == 'exp169-parity-plan-v1', 'EXP169 parity plan differs')
    ref = Path(plan['reference'])
    for name, digest in plan['reference_files'].items():
        need(sha(ref / name) == digest, 'Local parity reference changed: ' + name)
    trio = kernel_output / 'exp169-trio'
    receipt = read_json(trio / 'receipt.json')
    need(receipt.get('status') == 'COMPLETE_INFERENCE', 'Kernel trio did not complete')
    for name in ('raw.npz', 'inputs.json', 'binding.json', 'submission.csv'):
        need(sha(trio / name) == receipt['files'][name], 'Kernel trio output hash differs: ' + name)
    need(read_json(ref / 'binding.json') == read_json(trio / 'binding.json'),
         'Trio checkpoint/source binding differs')
    reference_inputs = read_json(ref / 'inputs.json')
    kernel_inputs = read_json(trio / 'inputs.json')
    input_equal = reference_inputs == kernel_inputs
    with np.load(ref / 'raw.npz', allow_pickle=False) as local, \
         np.load(trio / 'raw.npz', allow_pickle=False) as remote:
        need(set(local.files) == set(remote.files) == {'ids', 'targets', *plan['arms']} and
             np.array_equal(local['ids'], remote['ids']) and
             np.array_equal(local['targets'], remote['targets']) and
             local['targets'].tolist() == TARGETS and len(local['ids']) == 3,
             'Visible-test trio cohort/targets differ')
        diffs = {}
        for arm in plan['arms']:
            a, b = local[arm], remote[arm]
            need(a.shape == b.shape == (3, 12) and np.isfinite(a).all() and
                 np.isfinite(b).all(), 'Trio arm values invalid: ' + arm)
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            diffs[arm] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()))
    tol = plan['tolerance']
    arms_pass = all(d['max_abs'] <= tol['max_abs'] and d['mean_abs'] <= tol['mean_abs']
                    for d in diffs.values())
    partner = dict(status='NOT_AVAILABLE')
    if partner_local is not None:
        with np.load(partner_local, allow_pickle=False) as local, \
             np.load(kernel_output / 'partner-raw.npz', allow_pickle=False) as remote:
            need(set(local.files) == set(remote.files) == {'ids', 'l2', 'dino', 'convnext'} and
                 np.array_equal(local['ids'], remote['ids']),
                 'Partner local/kernel cohort differs')
            differences = {}
            for arm in ('l2', 'dino', 'convnext'):
                a, b = local[arm], remote[arm]
                need(a.shape == b.shape and a.shape[0] == (10 if arm == 'l2' else 5) and
                     a.shape[2] == 12 and np.isfinite(a).all() and np.isfinite(b).all(),
                     'Partner arm shape/value differs: ' + arm)
                differences[arm] = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
        partner = dict(status='PASS' if all(v <= 2e-5 for v in differences.values()) else 'HOLD',
                       max_abs=differences, tolerance=2e-5)
    result = dict(status='PASS_VISIBLE_TRIO' if input_equal and arms_pass and
                  sha(ref / 'submission.csv') == sha(trio / 'submission.csv') and
                  partner['status'] != 'HOLD' else 'HOLD_PARITY',
                  input_hashes_exact=input_equal, trio_raw=diffs,
                  trio_csv_exact=sha(ref / 'submission.csv') == sha(trio / 'submission.csv'),
                  partner=partner, plan_sha256=sha(plan_dir / 'parity-plan.json'))
    write_json(out, result)
    need(result['status'] == 'PASS_VISIBLE_TRIO', 'Visible-test parity failed; inspect result')
    return result


def timing_check(kernel_output, require_full=False):
    """Read downloaded receipts; assert full-cohort bounds only when requested."""
    kernel_output = Path(kernel_output)
    receipt = read_json(kernel_output / 'exp169-executed-candidate.json')
    need(receipt.get('status') == 'PASS_BUNDLE' and
         receipt.get('candidate') == 'TRIO_PLUS_BUNDLE' and
         receipt.get('executed_arms') == ['EXP155_BLEND3', 'L2_RADR50_10',
                                           'DINO_BASE_5', 'EXP170_L2_CONVNEXT_T_5'] and
         sha(kernel_output / 'submission.csv') == receipt['submission_sha256'] and
         sha(kernel_output / 'partner-raw.npz') == receipt['partner_raw_sha256'],
         'Downloaded output is not the admitted executed bundle')
    stages = receipt['stages']
    need(all(stages[key]['status'] == 'PASS' for key in
             ('trio', 'pilot', 'partner', 'parity', 'aggregate_output')) and
         receipt['elapsed_seconds'] <= LIMIT_SECONDS - RESERVE_SECONDS and
         receipt['pilot']['projected_full_seconds'] <=
             LIMIT_SECONDS - RESERVE_SECONDS - receipt['trio_seconds'] -
             receipt['pilot']['elapsed_seconds'] and
         max(stages['partner']['gpu_peak_bytes']) < 15 * 2**30 and
         (stages['partner']['ram']['maxrss_kib'] or math.inf) * 1024 < 27 * 2**30,
         'Timing or resource receipt exceeds release bounds')
    full = receipt['qualification'] == 'FULL_COHORT' and receipt['studies'] >= 30
    need(not require_full or full, 'Visible-test run cannot qualify full-cohort timing')
    return dict(status='PASS_FULL_COHORT_TIMING' if full else 'PASS_VISIBLE_ONLY',
                studies=receipt['studies'], elapsed_seconds=receipt['elapsed_seconds'],
                projected_partner_seconds=receipt['pilot']['projected_full_seconds'],
                partner_peak_vram_bytes=stages['partner']['gpu_peak_bytes'],
                partner_peak_ram_kib=stages['partner']['ram']['maxrss_kib'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    a = sub.add_parser('stage')
    a.add_argument('--trio-stage', required=True, type=Path)
    a.add_argument('--out', required=True, type=Path)
    a = sub.add_parser('push-plan')
    a.add_argument('--stage', required=True, type=Path)
    a = sub.add_parser('parity-plan')
    a.add_argument('--stage', required=True, type=Path)
    a.add_argument('--out', required=True, type=Path)
    a = sub.add_parser('compare-parity')
    a.add_argument('--plan', required=True, type=Path)
    a.add_argument('--kernel-output', required=True, type=Path)
    a.add_argument('--out', required=True, type=Path)
    a.add_argument('--partner-local', type=Path)
    a = sub.add_parser('timing-check')
    a.add_argument('--kernel-output', required=True, type=Path)
    a.add_argument('--require-full', action='store_true')
    a = sub.add_parser('pilot')
    a.add_argument('--work', required=True, type=Path)
    a.add_argument('--competition', required=True, type=Path)
    a.add_argument('--out', required=True, type=Path)
    a.add_argument('--studies', type=int, default=40)
    a = sub.add_parser('serve')
    for name in ('work', 'trio-work', 'output', 'competition'):
        a.add_argument('--' + name, required=True, type=Path)
    args = ap.parse_args(argv)
    if args.command == 'stage':
        result = stage(args.trio_stage, args.out)
    elif args.command == 'push-plan':
        result = push_plan(args.stage)
    elif args.command == 'parity-plan':
        result = parity_plan(args.stage, args.out)
    elif args.command == 'compare-parity':
        result = compare_parity(args.plan, args.kernel_output, args.out, args.partner_local)
    elif args.command == 'timing-check':
        result = timing_check(args.kernel_output, args.require_full)
    elif args.command == 'pilot':
        result = pilot(args.work, args.competition, args.out, args.studies)
    else:
        result = serve(args.work, args.trio_work, args.output, args.competition)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
