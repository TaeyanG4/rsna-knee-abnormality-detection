"""EXP179: frozen EXP170 extension; CPU preparation, serial Vast arms, collection.

No account API, secret access, download, submission or implicit GPU execution.
The EXP166/170 files remain unchanged; only the derived runner/loader differs.
Changelog 1.1.0 (2026-10-12): elapsed/projection values are telemetry; no self-imposed run deadline.
"""
from __future__ import annotations

import argparse
import copy
from functools import partial
import hashlib
import math
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = '1.1.0'
SCHEMA = 'exp179-multiscale-v1'
EXP = 'artifacts/experiments/EXP179_MULTISCALE'
TOOL = 'tools/exp179_multiscale.py'
LOADER = 'training/src/raptor_exp179.py'
SCREEN = 'tools/exp179_screen.py'
PROTOCOL = EXP + '/PROTOCOL.md'
RUNNER = 'training/runners/exp179_multiscale.py'
REMOTE_ROOT = '/workspace/exp179'
REMOTE_RESULTS = '/workspace/exp179-results'
SOURCE_ROOT = '/workspace/exp166'
ARMS = {'C43': (43, 'context'), 'M42': (42, 'mixed'), 'M43': (43, 'mixed')}
ARM_CAP, TOTAL_CAP, PILOT_CAP = 4200, 14400, 300
C42 = 'artifacts/experiments/EXP170_CONVNEXT_T_L2/outputs/results/EXP170_CONVNEXT_T_L2_S42_F0'
C42_PRED_SHA = '6eeb3432291701c4f5b7efd58f2ae1f47e64b32df166416964a55c68c479e4ae'
BASE_OVERLAY = '.local/exp170-overlay-v1'
VERIFICATION = 'verification.md'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def safe(name):
    p = PurePosixPath(name)
    need(name and p.as_posix() == name and not p.is_absolute() and
         '..' not in p.parts and ':' not in name and '\\' not in name, 'unsafe relative path')
    return name


def __getattr__(name):
    from tools import exp170_convnext_t_l2 as base
    return getattr(base, name)


def tag(arm):
    need(arm in ARMS, 'unknown EXP179 arm')
    return f'EXP179_{arm}_F0'


def arm_from_tag(value):
    matches = [a for a in ARMS if tag(a) == value]
    need(len(matches) == 1, 'unknown run tag')
    return matches[0]


def dataset_factory(value):
    # Runtime imports follow the frozen runner's sys.path convention.
    from raptor_exp179 import ScaleSliceWindows
    return partial(ScaleSliceWindows, crop_policy=ARMS[arm_from_tag(value)][1])


def evaluate_flip_tta(model, corpus, ids, labels, a, dev, eval_dtype, autocast):
    # Reuse the exact EXP166 inference loop, changing only its input dataset.
    from tools import exp166_convnext_t as old
    from training.src import raptor_radimagenet as module
    original = module.CenterSliceWindows
    module.CenterSliceWindows = dataset_factory(a.tag)
    try:
        return old.evaluate_flip_tta(model, corpus, ids, labels, a, dev, eval_dtype, autocast)
    finally:
        module.CenterSliceWindows = original


def runner_source(root=ROOT):
    from tools import exp170_convnext_t_l2 as base
    source = base.runner_source(root)
    changes = {
        'from tools import exp170_convnext_t_l2 as guard':
            'from tools import exp179_multiscale as guard',
        'CenterSliceWindows = loader_factory(a.variant)':
            'CenterSliceWindows = guard.dataset_factory(a.tag)',
        'if a.seed == protocol["seed"] and not init_matches and not relaxed:':
            'if not init_matches:',
        'provenance["flip_tta"] = "coronal_axial_horizontal_mean_probability"':
            'provenance["flip_tta"] = "coronal_axial_horizontal_mean_probability"\n'
            '    provenance.update(exp179=protocol["exp179"])',
        'names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv", protocol["extra_eval"]["file"], "oof_k36_flip_tta.csv"]':
            'guard.stamp_outputs(out, protocol)\n'
            '    names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv", protocol["extra_eval"]["file"], "oof_k36_flip_tta.csv"]',
    }
    for before, after in changes.items():
        need(source.count(before) == 1, 'frozen runner anchor drift: ' + before[:60])
        source = source.replace(before, after)
    compile(source, RUNNER, 'exec')
    return source


def stamp_outputs(out, protocol):
    for name in ('summary.json', 'provenance.json'):
        p = Path(out) / name
        value = read(p)
        value['exp179'] = protocol['exp179']
        write(p, value)


def queue():
    return [{'tag': tag(a), 'command': ['python', TOOL, 'run', '--arm', a],
             'continue_on_failure': False} for a in ARMS]


def check_verification(path, root=ROOT):
    """Human review remains B's responsibility; require exact reviewed source hashes."""
    text = Path(path).read_text(encoding='utf-8')
    need('Verdict: PASS' in text and 'Verdict: FAIL' not in text,
         'independent verification PASS note required')
    for name in (TOOL, LOADER, SCREEN, PROTOCOL):
        need(sha(Path(root) / name) in text, 'verification lacks current hash: ' + name)
    return sha(path)


def control_receipt(root=ROOT):
    from tools import exp170_convnext_t_l2 as base
    root = Path(root)
    collection = read(root / Path(C42).parents[1] / 'collection.json')
    need(collection['status'] == 'VERIFIED' and
         collection['manifest_sha256'] == sha(root / BASE_OVERLAY / 'manifest.json'),
         'C42 collection not verified/bound')
    hashes = base.verify_run(root / C42, 0, root / BASE_OVERLAY)
    need(hashes['oof_k36_flip_tta.csv'] == C42_PRED_SHA, 'wrong C42 control')
    return {'path': C42, 'hashes': hashes,
            'summary': read(root / C42 / 'summary.json'),
            'provenance': read(root / C42 / 'provenance.json'),
            'manifest_sha256': collection['manifest_sha256']}


def verify_derivation_sources(root, files):
    from tools import exp166_convnext_t as source
    root = Path(root)
    for name in (source.BASE_RUNNER, 'tools/exp166_convnext_t.py',
                 'training/src/raptor_common.py', 'training/src/raptor_radimagenet.py',
                 'training/src/raptor_exp127_aug1.py'):
        need(name in files and sha(root / name) == files[name]['sha256'],
             'local derivation differs from frozen source: ' + name)
    name = 'tools/exp170_convnext_t_l2.py'
    need(sha(root / name) == sha(root / BASE_OVERLAY / name), 'EXP170 wrapper source drift')


def prepare(out, verification, root=ROOT):
    from tools import exp170_convnext_t_l2 as base
    root, out = Path(root), Path(out)
    need(not out.exists(), 'fresh overlay required')
    note_sha = check_verification(verification, root)
    control = control_receipt(root)
    base.validate(root / BASE_OVERLAY)
    q0 = read(root / BASE_OVERLAY / 'protocols/fold0.json')
    files, receipt_sha = base._source_files(root)
    verify_derivation_sources(root, files)
    need(q0['label_set']['sha256'] == base.LABEL_SHA, 'wrong L2 protocol')
    derived = runner_source(root)
    out.mkdir(parents=True)
    manifest = {'schema': SCHEMA, 'version': VERSION, 'status': 'PREPARED_NOT_LAUNCHED',
        'source_root': SOURCE_ROOT, 'remote_root': REMOTE_ROOT,
        'source_bundle_sha256': receipt_sha, 'control': control,
        'verification_note': str(Path(verification).resolve()), 'verification_sha256': note_sha,
        'protocol_sha256': sha(root / PROTOCOL), 'arms': {},
        'caps_seconds': {'arm': ARM_CAP, 'total': TOTAL_CAP, 'pilot': PILOT_CAP},
        'files': {n: {**v, 'origin': 'exp166'} for n, v in files.items()}}
    payload = {n: root / n for n in (TOOL, LOADER, SCREEN, PROTOCOL,
               'tools/exp170_convnext_t_l2.py', 'config/workspace.toml',
               base.LABEL, base.LABEL_PROVENANCE, base.FOLD0_ROSTER)}
    payload[VERIFICATION] = Path(verification)
    for name, path in payload.items():
        safe(name)
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        manifest['files'][name] = base._entry(dest, 'overlay')
    need(sha(out / base.LABEL) == base.LABEL_SHA, 'L2 bytes changed')
    dest = out / RUNNER
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(derived, encoding='utf-8', newline='\n')
    manifest['files'][RUNNER] = base._entry(dest, 'overlay')
    for arm, (seed, policy) in ARMS.items():
        q = copy.deepcopy(q0)
        q.update(schema=SCHEMA, seed=seed, protocol_sha256=manifest['protocol_sha256'],
                 runner_sha256=sha(dest), tool_sha256=sha(out / TOOL))
        q['folds']['0']['tag'] = tag(arm)
        q['tags']['prefix'] = tag(arm)
        q['exp179'] = {'version': VERSION, 'arm': arm, 'crop_policy': policy,
            'verification_note': manifest['verification_note'], 'verification_sha256': note_sha,
            'loader_sha256': sha(out / LOADER), 'tool_sha256': sha(out / TOOL),
            'nominal_mm': [140, 100] if policy == 'mixed' else [140],
            'detail_slice': [48, 288], 'allocation': 'even=context,odd=detail' if policy == 'mixed' else 'context',
            'shared_frozen_initial_state': True}
        name = f'protocols/{arm}.json'
        write(out / name, q)
        manifest['files'][name] = base._entry(out / name, 'overlay')
        manifest['arms'][arm] = {'protocol': name, 'sha256': sha(out / name)}
    write(out / 'queue.json', queue())
    manifest['files']['queue.json'] = base._entry(out / 'queue.json', 'overlay')
    write(out / 'manifest.json', manifest)
    validate(out)
    return {'status': manifest['status'], 'manifest_sha256': sha(out / 'manifest.json'),
            'overlay': str(out), 'new_arms': list(ARMS), 'max_gpu_hours': TOTAL_CAP / 3600,
            'verification_note': manifest['verification_note'], 'version': VERSION}


def validate(package, full=False):
    from tools import exp170_convnext_t_l2 as base
    package = Path(package)
    m = read(package / 'manifest.json')
    need(m['schema'] == SCHEMA and m['version'] in ('1.0.0', '1.1.0', VERSION) and
         set(m['arms']) == set(ARMS) and m['remote_root'] == REMOTE_ROOT and
         m['source_root'] == SOURCE_ROOT and m['caps_seconds'] ==
         {'arm': ARM_CAP, 'total': TOTAL_CAP, 'pilot': PILOT_CAP}, 'manifest contract differs')
    need(check_verification(package / VERIFICATION, package) == m['verification_sha256'],
         'verification binding differs')
    need(read(package / 'queue.json') == queue(),
         'serial queue/budget differs')
    required = base.REQUIRED_MODULES | {TOOL, LOADER, SCREEN, RUNNER, PROTOCOL,
        'tools/exp170_convnext_t_l2.py', 'config/workspace.toml', VERIFICATION,
        base.LABEL, base.LABEL_PROVENANCE, base.SOURCE_WEIGHT, base.SOURCE_MANIFEST}
    need(required.issubset(m['files']), 'missing transitive dependency')
    for name, entry in m['files'].items():
        safe(name)
        need(entry['origin'] in ('exp166', 'overlay'), 'bad origin')
        if full or entry['origin'] == 'overlay':
            path = package / name
            need(path.is_file() and not path.is_symlink() and
                 path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                 'package drift: ' + name)
    need(m['protocol_sha256'] == sha(package / PROTOCOL) and
         m['files'][base.LABEL]['sha256'] == base.LABEL_SHA and
         m['files'][base.SOURCE_WEIGHT]['sha256'] == base.WEIGHT_SHA and
         m['control']['hashes']['oof_k36_flip_tta.csv'] == C42_PRED_SHA,
         'registered input/control differs')
    ids = read(package / base.FOLD0_ROSTER)
    need(len(ids['train']) == len(set(ids['train'])) == 3455 and
         len(ids['val']) == len(set(ids['val'])) == 888 and
         not set(ids['train']) & set(ids['val']), 'roster duplication/overlap')
    for arm, (seed, policy) in ARMS.items():
        p = m['arms'][arm]
        q = read(package / p['protocol'])
        f, args = q['folds']['0'], q['common_args']
        need(sha(package / p['protocol']) == p['sha256'] and q['schema'] == SCHEMA and
             q['seed'] == seed and f['tag'] == tag(arm) and q['exp179']['crop_policy'] == policy and
             q['exp179']['verification_sha256'] == m['verification_sha256'] and
             q['exp179']['loader_sha256'] == m['files'][LOADER]['sha256'] and
             q['initial_state_sha256'] == m['control']['provenance']['initial_state_sha256'] and
             q['tool_sha256'] == m['files'][TOOL]['sha256'] and
             q['runner_sha256'] == m['files'][RUNNER]['sha256'] and
             q['protocol_sha256'] == m['protocol_sha256'] and
             f['ordered_ids']['sha256'] == sha(package / base.FOLD0_ROSTER) and
             f['ordered_ids']['train_sha256'] == base.uid_hash(ids['train']) and
             f['ordered_ids']['val_sha256'] == base.uid_hash(ids['val']) and f['steps'] == 8620 and
             q['paths']['labels'] == q['paths']['eval_labels'] == base.LABEL and
             q['label_set']['sha256'] == base.LABEL_SHA,
             'arm binding differs')
        # Compare the ENTIRE training argument contract against the reused control.
        need(all(m['control']['summary']['args'].get(k) == v for k, v in args.items()) and
             args['arch'] == 'convnext_tiny.fb_in22k_ft_in1k' and args['res'] == 224 and
             args['k'] == 12 and args['epochs'] == args['fixed_epoch'] == 20 and
             args['input_policy'] == 'center3' and args['amp'] == args['eval_amp'] == 'fp16' and
             q['extra_eval']['k_eval'] == 36, 'recipe differs from C42')
    if full:
        need(str(package.resolve()) == REMOTE_ROOT and
             sha(package / base.SOURCE_RECEIPT) == m['source_bundle_sha256'], 'wrong remote/source')
    return m


def vast_stage(overlay, source=SOURCE_ROOT, destination=REMOTE_ROOT):
    from tools import exp170_convnext_t_l2 as base
    need(sys.platform.startswith('linux'), 'Vast staging is Linux-only')
    overlay, source, destination = map(Path, (overlay, source, destination))
    need(str(source.resolve()) == SOURCE_ROOT and str(destination.resolve()) == REMOTE_ROOT
         and not destination.exists(), 'fresh exact remote roots required')
    m = validate(overlay)
    need(sha(source / base.SOURCE_RECEIPT) == m['source_bundle_sha256'], 'wrong remote source')
    destination.mkdir(mode=0o700)
    for name, entry in m['files'].items():
        origin = (source if entry['origin'] == 'exp166' else overlay) / name
        need(origin.is_file() and not origin.is_symlink() and sha(origin) == entry['sha256'],
             'remote source drift: ' + name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry['origin'] == 'exp166':
            os.link(origin, target)
        else:
            shutil.copyfile(origin, target)
    shutil.copyfile(overlay / 'manifest.json', destination / 'manifest.json')
    os.link(source / base.SOURCE_RECEIPT, destination / base.SOURCE_RECEIPT)
    validate(destination, full=True)
    write(destination / 'stage-receipt.json', {'status': 'STAGED_VERIFIED_NO_GPU',
          'version': VERSION, 'verification_note': m['verification_note'],
          'manifest_sha256': sha(destination / 'manifest.json')})
    return {'status': 'STAGED_VERIFIED_NO_GPU', 'destination': REMOTE_ROOT}


def validate_arguments(a, root):
    from tools import exp170_convnext_t_l2 as base
    root = Path(root)
    m = validate(root)
    arm = arm_from_tag(a.tag)
    q = read(root / m['arms'][arm]['protocol'])
    need(a.seed == ARMS[arm][0] and a.fold == 0 and a.variant == 'AUG1' and
         a.mode == 'full' and not a.validate_only and a.device == 'cuda' and
         Path(a.protocol).resolve() == (root / m['arms'][arm]['protocol']).resolve() and
         all(getattr(a, k) == v for k, v in q['common_args'].items()), 'runtime arguments changed')
    for key in ('corpus', 'labels', 'eval_labels'):
        need(Path(getattr(a, key)).resolve() == (root / q['paths'][key]).resolve(), 'input path changed')
    need(0 <= a.workers <= 12 and os.environ.get('HF_HUB_OFFLINE') == '1' and
         all(os.environ.get(k) == '1' for k in base.THREADS) and
         Path(a.init_weights).resolve() == (root / base.SOURCE_WEIGHT).resolve() and
         Path(a.output_root).resolve() == Path(REMOTE_RESULTS), 'environment/output changed')
    f = q['folds']['0']
    return q, f, read(root / f['ordered_ids']['path']), Path(REMOTE_RESULTS)


def runner_args(arm, pilot=False, root=ROOT):
    from tools import exp166_convnext_t as source
    from tools import exp170_convnext_t_l2 as base
    root = Path(root)
    args = source.runner_args(root, REMOTE_RESULTS, workers=8, pilot=pilot)
    args[1] = str(root / RUNNER)
    for flag, value in {'--protocol': root / f'protocols/{arm}.json', '--tag': tag(arm),
                        '--fold': 0, '--seed': ARMS[arm][0], '--labels': root / base.LABEL,
                        '--eval-labels': root / base.LABEL}.items():
        args[args.index(flag) + 1] = str(value)
    return args


def verify_run(run_dir, arm, package=ROOT):
    from tools import exp170_convnext_t_l2 as base
    from tools.exp162_dino_folds_kaggle import _prediction_values
    package, out = Path(package), Path(run_dir)
    m = validate(package)
    q = read(package / m['arms'][arm]['protocol'])
    f = q['folds']['0']
    need(out.name == tag(arm), 'wrong output identity')
    inv = read(out / 'output-inventory.json')
    need(set(inv) == base.REQUIRED_OUTPUTS, 'incomplete output inventory')
    for name, e in inv.items():
        safe(name)
        need((out / name).is_file() and not (out / name).is_symlink() and
             sha(out / name) == e['sha256'] and (out / name).stat().st_size == e['bytes'],
             'output changed: ' + name)
    s, p, h = (read(out / n) for n in ('summary.json', 'provenance.json', 'history.json'))
    need(s['fold'] == 0 and s['train'] == 3455 and s['val'] == s['rows'] == 888 and
         s['selected_epoch_1based'] == 20 and s['optimizer_updates'] == 8620 and
         s['skipped_updates'] <= f['max_fp16_skips'] and
         [v['epoch'] for v in h] == list(range(20)) and s['gold_access'] == 'disabled' and
         s['weak_oof_macro_auc'] is None and s['label_set'] == base.LABEL_SET and
         s['exp179'] == p['exp179'] == q['exp179'], 'incomplete or wrong fixed-final arm')
    need(p['environment'] == m['control']['provenance']['environment'] and
         p['initial_state_sha256'] == q['initial_state_sha256'] and
         p['init_weights']['weights_sha256'] == base.WEIGHT_SHA and
         p['runner_sha256'] == m['files'][RUNNER]['sha256'] and
         p['guard_sha256'] == m['files'][TOOL]['sha256'] and
         p['protocol_sha256'] == m['arms'][arm]['sha256'] and
         p['train_ids_ordered_sha256'] == f['ordered_ids']['train_sha256'] and
         p['eval_ids_ordered_sha256'] == f['ordered_ids']['val_sha256'] and
         p['actual_pos_weight'] == f['pos_weight'] and
         p['label_file_sha256'] == base.LABEL_SHA and
         p['label_semantic_sha256'] == q['label_set']['semantic_sha256'] and
         p['flip_tta'] == 'coronal_axial_horizontal_mean_probability', 'run provenance mismatch')
    a = s['args']
    need(a['seed'] == ARMS[arm][0] and a['tag'] == tag(arm) and a['mode'] == 'full' and
         all(a.get(k) == v for k, v in q['common_args'].items()) and
         a['labels'] == a['eval_labels'] == f'{REMOTE_ROOT}/{base.LABEL}', 'run args mismatch')
    ids = read(package / base.FOLD0_ROSTER)['val']
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        _prediction_values(out / name, ids)
    return {name: inv[name]['sha256'] for name in inv}


def check_environment(control, hostname, gpu, versions):
    expected = control['provenance']['environment']
    need(hostname == control['summary']['environment']['hostname'] and gpu == expected['gpu'],
         'C42 control was not trained on this host/GPU')
    need(set(versions) == set(expected) - {'gpu'} and
         all(versions[k] == v for k, v in expected.items() if k != 'gpu'),
         'control software environment differs before training')


def launcher_pids(arm, proc_root=Path('/proc'), parent=None):
    """Allow only an ancestor gpu_job logger, which may own a CUDA context."""
    pid = os.getppid() if parent is None else parent
    allowed = set()
    for _ in range(4):
        node = Path(proc_root) / str(pid)
        try:
            args = (node / 'cmdline').read_bytes().decode().split('\0')
            cwd = (node / 'cwd').resolve()
            if (str(cwd) == REMOTE_ROOT and any(Path(a).as_posix() in
                    ('tools/gpu_job.py', REMOTE_ROOT + '/tools/gpu_job.py') for a in args)
                    and '--tag' in args and '--wait' in args and
                    args[args.index('--tag') + 1] == tag(arm)):
                allowed.add(pid)
            line = next(v for v in (node / 'status').read_text().splitlines() if v.startswith('PPid:'))
            pid = int(line.split()[1])
            if pid <= 1:
                break
        except (OSError, ValueError, StopIteration, IndexError):
            break
    return allowed


def check_idle_gpu(gpu_line, process_text, allowed):
    gpu = [v.strip() for v in gpu_line.strip().split(',')]
    need(len(gpu) == 3 and '3090' in gpu[0] and int(gpu[1]) >= 20000 and int(gpu[2]) < 5,
         'RTX3090 must have >=20000MiB free and be idle')
    pids = {int(row.strip()) for row in process_text.splitlines() if row.strip()}
    need(pids <= allowed, 'unrelated GPU process is active')
    return gpu[0]


def run(arm):
    need(sys.platform.startswith('linux') and str(ROOT.resolve()) == REMOTE_ROOT,
         'run requires the exact Vast root')
    import fcntl
    with (ROOT / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(arm)


def check_prior_arms(root, arm):
    root = Path(root)
    need(arm in ARMS, 'unknown arm')
    for prior in list(ARMS)[:list(ARMS).index(arm)]:
        receipt = root / f'validated-{prior}.json'
        need(receipt.is_file(), 'prior arm missing or started-but-unvalidated: ' + prior)
        record = read(receipt)
        need(record['status'] == 'VALIDATED_SUCCESS' and record['arm'] == prior and
             record['manifest_sha256'] == sha(root / 'manifest.json'), 'prior arm not validated')
        verify_run(Path(REMOTE_RESULTS) / tag(prior), prior, root)


def _run_locked(arm):
    from tools import exp170_convnext_t_l2 as base
    need(sys.platform.startswith('linux') and str(ROOT.resolve()) == REMOTE_ROOT,
         'run requires the exact Vast root')
    m = validate(ROOT, full=True)
    need(arm in ARMS, 'unknown arm')
    check_prior_arms(ROOT, arm)
    admission = read(ROOT / 'admission.json')
    need(admission == {'status': 'READY', 'manifest_sha256': sha(ROOT / 'manifest.json'),
        'verification_sha256': m['verification_sha256'], 'max_gpu_seconds': TOTAL_CAP,
        'arms': list(ARMS)}, 'exact reviewed admission missing')
    gpu_line = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
    process_text = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, check=True).stdout
    gpu = check_idle_gpu(gpu_line, process_text, launcher_pids(arm))
    # CPU-only package versions checked before any CUDA model construction.
    import torch
    import timm
    import numpy
    import cv2
    import pydicom
    actual = dict(torch=torch.__version__, timm=timm.__version__, numpy=numpy.__version__,
                  cv2=cv2.__version__, pydicom=pydicom.__version__, cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version())
    check_environment(m['control'], os.uname().nodename, gpu, actual)
    base._corpus(ROOT)
    output = Path(REMOTE_RESULTS)
    output.mkdir(exist_ok=True)
    out = output / tag(arm)
    need(not out.exists(), 'no automatic retry/resume of a started scientific arm')
    # Reserve a unique attempt; historical cap fields are telemetry only.
    reservation = ROOT / f'started-{arm}.json'
    with reservation.open('x', encoding='utf-8') as f:
        json.dump({'arm': arm, 'reserved_seconds': ARM_CAP, 'version': VERSION,
                   'manifest_sha256': sha(ROOT / 'manifest.json'),
                   'verification_note': m['verification_note']}, f)
    env = dict(os.environ, KNEE_WORKSPACE_ROOT=REMOTE_ROOT, KNEE_REMOTE_RELAX_ENV='1',
        HF_HUB_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUTF8='1',
        PYTHONDONTWRITEBYTECODE='1', EXP166_REMOTE='vast')
    env.update({k: '1' for k in base.THREADS})
    started = time.monotonic()
    subprocess.run(runner_args(arm, True), cwd=ROOT, env=env, check=True)
    pilot = read(out / 'pilot.json')
    need(math.isfinite(pilot['epoch_seconds']) and pilot['epoch_seconds'] > 0 and
         math.isfinite(pilot['peak_gpu_gib']) and 0 <= pilot['peak_gpu_gib'] < 20,
         'invalid pilot timing or memory')
    subprocess.run(runner_args(arm), cwd=ROOT, env=env, check=True)
    elapsed = time.monotonic() - started
    hashes = verify_run(out, arm)
    receipt = {'status': 'VALIDATED_SUCCESS', 'arm': arm, 'version': VERSION,
        'manifest_sha256': sha(ROOT / 'manifest.json'), 'output_sha256': hashes,
        'elapsed_seconds': elapsed, 'verification_note': m['verification_note'],
        'verification_sha256': m['verification_sha256']}
    write(ROOT / f'validated-{arm}.json', receipt)
    export(arm, f'/workspace/exp179-export-{arm}-fold0.tar')
    return receipt


def export(arm, out, root=ROOT):
    from tools import exp170_convnext_t_l2 as base
    root, out = Path(root), Path(out)
    need(str(root.resolve()) == REMOTE_ROOT and not out.exists(), 'fresh exact-root export required')
    validate(root, full=True)
    run_dir = Path(REMOTE_RESULTS) / tag(arm)
    hashes = verify_run(run_dir, arm, root)
    receipt = read(root / f'validated-{arm}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and receipt['output_sha256'] == hashes and
         receipt['manifest_sha256'] == sha(root / 'manifest.json'), 'receipt mismatch')
    with tarfile.open(out, 'x') as tar:
        tar.add(root / 'manifest.json', arcname='manifest.json')
        tar.add(root / f'validated-{arm}.json', arcname=f'validated-{arm}.json')
        for n in sorted(base.REQUIRED_OUTPUTS | {'output-inventory.json'}):
            tar.add(run_dir / n, arcname=f'results/{tag(arm)}/{n}')
    return {'status': 'EXPORTED_VALIDATED', 'sha256': sha(out), 'path': str(out),
            'version': VERSION, 'verification_note': receipt['verification_note']}


def collect(archive, overlay, out, arm):
    from tools import exp170_convnext_t_l2 as base
    archive, overlay, out = map(Path, (archive, overlay, out))
    m = validate(overlay)
    need(not out.exists(), 'fresh collection required')
    expected = {'manifest.json', f'validated-{arm}.json'} | {
        f'results/{tag(arm)}/{n}' for n in base.REQUIRED_OUTPUTS | {'output-inventory.json'}}
    with tarfile.open(archive, 'r') as tar:
        members = tar.getmembers()
        need(len(members) == len(expected) and {x.name for x in members} == expected and
             all(x.isfile() and safe(x.name) == x.name for x in members), 'unsafe/incomplete archive')
        out.mkdir(parents=True)
        for member in members:
            dest = out / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, dest.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    need(sha(out / 'manifest.json') == sha(overlay / 'manifest.json'), 'wrong exported manifest')
    hashes = verify_run(out / 'results' / tag(arm), arm, overlay)
    receipt = read(out / f'validated-{arm}.json')
    need(receipt['status'] == 'VALIDATED_SUCCESS' and receipt['arm'] == arm and
         receipt['manifest_sha256'] == sha(overlay / 'manifest.json') and
         receipt['output_sha256'] == hashes and math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] > 0,
         'invalid exported success')
    write(out / 'collection.json', {'status': 'VERIFIED', 'arm': arm, 'version': VERSION,
        'manifest_sha256': sha(overlay / 'manifest.json'), 'archive_sha256': sha(archive),
        'verification_note': m['verification_note'], 'overlay': str(overlay.resolve())})
    return {'status': 'VERIFIED', 'arm': arm, 'path': str(out), 'version': VERSION,
            'verification_note': m['verification_note']}


def vast_plan(overlay='/workspace/exp179-overlay'):
    import shlex
    need(overlay.startswith('/workspace/') and ':' not in overlay and '\\' not in overlay
         and '..' not in PurePosixPath(overlay).parts, 'absolute /workspace overlay required')
    q = shlex.quote(overlay)
    return '\n'.join([
        '# Windows Git Bash: export MSYS_NO_PATHCONV=1 before ssh/scp.',
        '# B: verify note, spot-check PASS, check live idle GPU and Active compute claim.',
        f'python {q}/{TOOL} vast-stage --overlay {q}', f'cd {REMOTE_ROOT}',
        f'export KNEE_WORKSPACE_ROOT={REMOTE_ROOT} HF_HUB_OFFLINE=1 PYTHONUTF8=1',
        f'python {TOOL} validate --package . --full',
        'python -c \'import json,hashlib; m=json.load(open("manifest.json")); h=hashlib.file_digest(open("manifest.json","rb"),"sha256").hexdigest(); json.dump(dict(status="READY",manifest_sha256=h,verification_sha256=m["verification_sha256"],max_gpu_seconds=14400,arms=["C43","M42","M43"]),open("admission.json","x"))\'',
        'setsid nohup python tools/gpu_queue.py queue.json --foreground > exp179-queue.log 2>&1 < /dev/null & echo $! > exp179-queue.pid',
        '# Each arm exports /workspace/exp179-export-{C43,M42,M43}-fold0.tar immediately.',
    ])


def screen(outputs, overlay, out, root=ROOT):
    """Verified CPU analysis of the preregistered existing EXP172 bundle."""
    from tools import exp179_screen as helper
    from tools import exp115_vlm_eval as ev
    from tools import exp169_bundle_screen as bundle
    from tools import exp172_dino_base_l2 as dino_tool
    root, outputs, overlay, out = map(Path, (root, outputs, overlay, out))
    started = time.monotonic()
    m = validate(overlay)
    check_verification(overlay / VERIFICATION, root)
    need(not out.exists(), 'fresh screen output required')
    fixed = bundle.paths(root, 0)
    rad = {s: fixed[f'l2_s{s}'] for s in (42, 43)}
    # Identity is fixed by canonical model directories, not caller-selected CSVs.
    for path in rad.values():
        need(path.is_file(), 'required L2 RadR50 plain K36 OOF missing: ' + str(path))
        bundle._verify_inventory(path)
    dino_outputs = root / dino_tool.EXP / 'outputs'
    dc = read(dino_outputs / 'collection.json')
    need(dc['status'] == 'VERIFIED' and dc['manifest_sha256'] == sha(dino_outputs / 'manifest.json'),
         'L2 DINO collection differs')
    dp = Path(dc['overlay'])
    dm = dino_tool.validate_manifest(dp / 'manifest.json', dp)
    dq = dm['fold_protocols']['0']
    dr = dino_outputs / 'artifacts/experiments' / dino_tool.tag(0)
    dino_tool._verify_fold(dr, 0, read(dp / dq['path']), dq['sha256'], dp)
    dino = dr / 'oof_best.csv'
    control = control_receipt(root)
    need(control['hashes'] == m['control']['hashes'], 'C42 changed since preparation')
    paths = {'C42': root / C42 / 'oof_k36_flip_tta.csv'}
    for arm in ARMS:
        directory = outputs / arm
        collection = read(directory / 'collection.json')
        need(collection['status'] == 'VERIFIED' and collection['arm'] == arm and
             collection['manifest_sha256'] == sha(overlay / 'manifest.json'), 'unverified collected arm')
        run_dir = directory / 'results' / tag(arm)
        hashes = verify_run(run_dir, arm, overlay)
        receipt = read(directory / f'validated-{arm}.json')
        need(receipt['output_sha256'] == hashes and receipt['status'] == 'VALIDATED_SUCCESS',
             'collected receipt differs')
        paths[arm] = run_dir / 'oof_k36_flip_tta.csv'
    ids, refs, draws, context = helper.load_context(root)
    values = {a: bundle.load_oof(p, ids) for a, p in paths.items()}
    result = helper.score_pairs({42: values['M42'], 43: values['M43']},
        {42: values['C42'], 43: values['C43']}, refs, draws, ev,
        {s: bundle.load_oof(p, ids) for s, p in rad.items()}, bundle.load_oof(dino, ids),
        verification_note=m['verification_note'])
    inputs = {**paths, **{f'R{s}': p for s, p in rad.items()}, 'D172': dino}
    result['diagnostics'] = {}
    for label, filename in (('plain_k36', 'oof_k36.csv'), ('plain_k24', 'oof_best.csv')):
        diagnostic_paths = {arm: path.with_name(filename) for arm, path in paths.items()}
        diagnostic_values = {arm: bundle.load_oof(p, ids) for arm, p in diagnostic_paths.items()}
        result['diagnostics'][label] = helper._paired_seed_mean(
            {42: diagnostic_values['M42'], 43: diagnostic_values['M43']},
            {42: diagnostic_values['C42'], 43: diagnostic_values['C43']}, refs, draws, ev)
        inputs.update({f'{label}_{a}': p for a, p in diagnostic_paths.items()})
    result.update(context=context, input_files={k: str(p) for k, p in inputs.items()},
        input_sha256={k: sha(p) for k, p in inputs.items()},
        manifest_sha256=sha(overlay / 'manifest.json'),
        verification_sha256=m['verification_sha256'], version=VERSION,
        elapsed_seconds=time.monotonic() - started)
    need(math.isfinite(result['elapsed_seconds']) and result['elapsed_seconds'] > 0, 'invalid screen timing')
    write(out / 'result.json', result)
    return {'status': result['status'], 'registered_gate': result['registered_gate'],
            'path': str(out / 'result.json'), 'version': VERSION,
            'verification_note': m['verification_note']}


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--out', required=True); a.add_argument('--verification', required=True)
    a = sub.add_parser('validate'); a.add_argument('--package', required=True); a.add_argument('--full', action='store_true')
    a = sub.add_parser('vast-stage'); a.add_argument('--overlay', required=True)
    a = sub.add_parser('vast-plan'); a.add_argument('--overlay', default='/workspace/exp179-overlay')
    a = sub.add_parser('run'); a.add_argument('--arm', choices=ARMS, required=True)
    a = sub.add_parser('export'); a.add_argument('--arm', choices=ARMS, required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('collect'); a.add_argument('--arm', choices=ARMS, required=True); a.add_argument('--archive', required=True); a.add_argument('--overlay', required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('screen'); a.add_argument('--outputs', default=EXP + '/outputs'); a.add_argument('--overlay', required=True); a.add_argument('--out', default=EXP + '/screen')
    args = p.parse_args(argv)
    if args.command == 'prepare': result = prepare(args.out, args.verification)
    elif args.command == 'validate':
        m = validate(args.package, args.full)
        result = {'status': 'VERIFIED_PACKAGE', 'version': VERSION, 'verification_note': m['verification_note']}
    elif args.command == 'vast-stage': result = vast_stage(args.overlay)
    elif args.command == 'vast-plan': result = vast_plan(args.overlay)
    elif args.command == 'run': result = run(args.arm)
    elif args.command == 'export': result = export(args.arm, args.out)
    elif args.command == 'screen': result = screen(args.outputs, args.overlay, args.out)
    else: result = collect(args.archive, args.overlay, args.out, args.arm)
    print(result if isinstance(result, str) else json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
