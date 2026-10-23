"""EXP159: Colab T4 admission of the hash-bound EXP150 C42 control.

Local stage/collect/compare are CPU-only. Only ``remote-run`` performs network
or GPU work, inside a Colab runtime after B pastes the printed bootstrap cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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
from tools import exp150_consensus_weight as e150

EXP = Path('artifacts/experiments/EXP159_COLAB_ADMISSION')
PROTOCOL = EXP / 'PROTOCOL.md'
TOOL = 'tools/exp159_colab_admission.py'
DERIVED = 'training/runners/exp159_colab_admission.py'
BASE_RUNNER = e150.RUNNER
ARCHIVE = 'exp159-code.zip.bin'
RECEIPT = e150.RECEIPT
CODE_DS = 'taeyangg4/knee-exp159-colab-code'
RESULT_DS = 'taeyangg4/knee-exp159-colab-c42'
TAG = 'EXP150_C42_F0'
CHECKPOINT = 'epoch-state.pt'
CP_META = 'epoch-state.json'
REFS = ('jev_hybrid', 'aquila_v1', 'dread', 'consensus')
LIMITS = dict(jev_hybrid=.0045, aquila_v1=.0031, dread=.0027, consensus=.0028)
REFERENCE_4070 = dict(jev_hybrid=.8811, aquila_v1=.9140, dread=.9013, consensus=.9612)
REFERENCE_T4 = dict(jev_hybrid=.8819, aquila_v1=.9140, dread=.9018, consensus=.9624)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def put_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def transform_runner(source):
    """Make a narrowly derived runner; fail closed if the bound EXP150 changes."""
    changes = [
        ('from tools import exp150_consensus_weight as e150',
         'from tools import exp150_consensus_weight as e150\n'
         '    from tools import exp159_colab_admission as e159\n'
         '    e150.validate_runner_args = e159.validate_runner_args_colab'),
        ('if out.exists() and any(out.iterdir()):',
         'if out.exists() and any(out.iterdir()) and not e159.resume_directory_allowed(out):'),
        ('history, best, optimizer_updates = [], None, 0',
         'history, best, optimizer_updates, skipped, start_epoch = e159.restore_epoch(\n'
         '        out, model, opt, sched, scaler, shuffle_generator, provenance, skipped)'),
        ('for ep in range(epochs):', 'for ep in range(start_epoch, epochs):'),
        ('(out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")',
         '(out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")\n'
         '        e159.save_epoch(out, ep, model, opt, sched, scaler, shuffle_generator,\n'
         '                        history, best, optimizer_updates, skipped, provenance)\n'
         '        e159.publish_epoch(out, ep)'),
    ]
    for old, new in changes:
        need(source.count(old) == 1, 'EXP150 runner anchor changed: ' + old[:55])
        source = source.replace(old, new)
    compile(source, DERIVED, 'exec')
    return source


_ORIGINAL_VALIDATE = e150.validate_runner_args


def validate_runner_args_colab(args, root):
    need(sys.platform.startswith('linux') and os.environ.get('EXP150_REMOTE') == 'colab',
         'Colab runner requires Linux and explicit Colab marker')
    # EXP150's exact recipe guard has a two-surface admission switch. Reuse
    # all of its checks, changing only that switch for this one Colab run.
    marker = os.environ['EXP150_REMOTE']
    try:
        os.environ['EXP150_REMOTE'] = 'kaggle'
        return _ORIGINAL_VALIDATE(args, root)
    finally:
        os.environ['EXP150_REMOTE'] = marker


def stage(out, root=ROOT):
    root, out = Path(root), Path(out)
    need(not out.exists(), 'Use a new stage directory')
    files = e150.package_files(root)
    need(e150.sha(root / BASE_RUNNER) == e150.read(root / e150.MANIFEST)['source_sha256'][BASE_RUNNER],
         'EXP150 runner binding changed')
    derived = transform_runner((root / BASE_RUNNER).read_text(encoding='utf-8'))
    # EXP150.package_files already includes every top-level tools/*.py file,
    # including this wrapper. The nested Colab helper and new protocol are extra.
    additions = [PROTOCOL.as_posix(), 'tools/colab/run_remote.py',
                 'tools/colab/exp159_remote.py']
    for name in additions:
        need(name not in files, 'Duplicate package entry: ' + name)
        e150.safe(name)
        files[name] = dict(sha256=e150.sha(root / name), bytes=(root / name).stat().st_size)
    need(DERIVED not in files, 'Duplicate derived runner')
    derived_bytes = derived.encode('utf-8')
    files[DERIVED] = dict(sha256=hashlib.sha256(derived_bytes).hexdigest(), bytes=len(derived_bytes))
    marker = b'# EXP159 isolated Colab workspace; no credentials.\n'
    files['config/workspace.toml'] = dict(sha256=hashlib.sha256(marker).hexdigest(), bytes=len(marker))
    bundle = dict(schema='exp150-package-v1', surface='colab', files=files,
                  manifest_sha256=e150.sha(root / e150.MANIFEST),
                  protocol_sha256=e150.sha(root / PROTOCOL),
                  base_runner_sha256=e150.sha(root / BASE_RUNNER),
                  derived_runner_sha256=files[DERIVED]['sha256'], no_execution=True)
    rb = (json.dumps(bundle, indent=2, ensure_ascii=False) + '\n').encode()
    code = out / 'code-dataset'
    code.mkdir(parents=True)
    with zipfile.ZipFile(code / ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            if name == DERIVED:
                z.writestr(name, derived_bytes)
            elif name == 'config/workspace.toml':
                z.writestr(name, marker)
            else:
                z.write(root / name, name)
        z.writestr(RECEIPT, rb)
    put_json(code / 'dataset-metadata.json', dict(id=CODE_DS, title='EXP159 Colab admission code',
             isPrivate=True, licenses=[{'name': 'other'}],
             description='Private EXP159 code and fixed EXP150 initial state.'))
    put_json(out / 'stage-receipt.json', dict(schema='exp159-stage-v1',
             archive_sha256=e150.sha(code / ARCHIVE), bundle_sha256=hashlib.sha256(rb).hexdigest(),
             protocol_sha256=bundle['protocol_sha256'], derived_runner_sha256=files[DERIVED]['sha256']))
    verify_stage(out, root)
    return e150.read(out / 'stage-receipt.json')


def verify_stage(out, root=ROOT):
    out, root = Path(out), Path(root)
    r = e150.read(out / 'stage-receipt.json')
    archive = out / 'code-dataset' / ARCHIVE
    need(r['schema'] == 'exp159-stage-v1' and e150.sha(archive) == r['archive_sha256'],
         'Stage receipt/archive mismatch')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)), 'Duplicate archive entry')
        b = json.loads(z.read(RECEIPT))
        need(set(names) == {*b['files'], RECEIPT} and b['surface'] == 'colab',
             'Bundle entry/surface mismatch')
        need(hashlib.sha256(z.read(RECEIPT)).hexdigest() == r['bundle_sha256'],
             'Bundle receipt mismatch')
        for name, record in b['files'].items():
            e150.safe(name)
            data = z.read(name)
            need(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'],
                 'Bundle file mismatch: ' + name)
            if name == DERIVED:
                need(data.decode() == transform_runner((root / BASE_RUNNER).read_text(encoding='utf-8')),
                     'Derived runner drift')
            elif name != 'config/workspace.toml':
                need(e150.sha(root / name) == record['sha256'], 'Local source drift: ' + name)
    need(e150.read(out / 'code-dataset/dataset-metadata.json')['isPrivate'] is True,
         'Code dataset must be private')
    return r


def bootstrap_cell(out):
    verify_stage(out)
    cell = f'''import os, pathlib, subprocess, sys, zipfile, hashlib
from google.colab import userdata
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
token = userdata.get('KAGGLE_API_TOKEN')
assert token, 'Add KAGGLE_API_TOKEN to Colab secrets'
os.environ['KAGGLE_API_TOKEN'] = token
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', '--no-deps', 'kaggle'], check=True)
subprocess.run(['kaggle', '--version'], check=True)
base = pathlib.Path('/content/exp159-code'); base.mkdir(exist_ok=True)
subprocess.run(['kaggle', 'datasets', 'download', '-d', '{CODE_DS}', '-p', str(base), '--unzip', '-q', '-o'], check=True)
archive = base / '{ARCHIVE}'
work = pathlib.Path('/content/exp159')
with zipfile.ZipFile(archive) as z: expected_sha = hashlib.sha256(z.read('{RECEIPT}')).hexdigest()
with zipfile.ZipFile(archive) as z:
    names = z.namelist(); assert len(names) == len(set(names))
    assert all(not n.startswith('/') and '..' not in pathlib.PurePosixPath(n).parts and ':' not in n and chr(92) not in n for n in names)
    z.extractall(work)
bundle_sha = hashlib.sha256((work / '{RECEIPT}').read_bytes()).hexdigest()
assert bundle_sha == expected_sha, 'Colab workspace is from a different code dataset version'
subprocess.run([sys.executable, str(work / '{TOOL}'), 'remote-run', '--bundle-sha', bundle_sha], cwd=work, check=True)
'''
    need(len(cell.splitlines()) <= 25, 'Bootstrap exceeds 25 lines')
    return cell


def resume_directory_allowed(out):
    out = Path(out)
    return (out / CHECKPOINT).is_file() and (out / CP_META).is_file() and not (out / 'summary.json').exists()


def environment_key(provenance):
    return dict(provenance['environment'], exp150_manifest=e150.sha(ROOT / e150.MANIFEST),
                derived_runner=e150.sha(ROOT / DERIVED), protocol=e150.sha(ROOT / PROTOCOL))


def save_epoch(out, ep, model, opt, sched, scaler, generator, history, best, updates, skipped, provenance):
    import numpy as np
    import torch
    out = Path(out)
    state = dict(schema='exp159-epoch-v1', epoch=ep, model=model.state_dict(),
                 optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                 generator=generator.get_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                 history=history, best=best, optimizer_updates=updates, skipped=skipped,
                 environment=environment_key(provenance))
    tmp = out / (CHECKPOINT + '.tmp')
    torch.save(state, tmp)
    tmp.replace(out / CHECKPOINT)
    put_json(out / CP_META, dict(schema='exp159-epoch-v1', epoch=ep,
             checkpoint_sha256=e150.sha(out / CHECKPOINT), environment=state['environment']))


def restore_epoch(out, model, opt, sched, scaler, generator, provenance, skipped):
    import numpy as np
    import torch
    out = Path(out)
    if not (out / CHECKPOINT).exists():
        return [], None, 0, skipped, 0
    meta = e150.read(out / CP_META)
    need(meta['schema'] == 'exp159-epoch-v1' and
         e150.sha(out / CHECKPOINT) == meta['checkpoint_sha256'] and
         meta['environment'] == environment_key(provenance), 'Resume checkpoint/environment mismatch')
    state = torch.load(out / CHECKPOINT, map_location='cpu', weights_only=False)
    need(state['schema'] == 'exp159-epoch-v1' and state['epoch'] == meta['epoch'] and
         0 <= state['epoch'] < 20 and
         state['environment'] == meta['environment'] and
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


def publish_epoch(out, ep, max_wait=600, poll_interval=10):
    """Best-effort private copy; the local epoch checkpoint is authoritative."""
    out = Path(out)
    put_json(out / 'dataset-metadata.json', dict(title='EXP159 Colab C42 results', id=RESULT_DS,
             licenses=[{'name': 'other'}], isPrivate=True,
             description='Private EXP159 control replicate and epoch checkpoint.'))
    expected_size = (out / CHECKPOINT).stat().st_size
    from tools.colab.exp159_remote import dataset_files

    def warning(reason):
        print(f'WARNING: EXP159 epoch {ep} Kaggle checkpoint verification: {reason}; '
              'continuing from local checkpoint', flush=True)
        return False

    def response_text(result):
        return (result.stdout + result.stderr).strip()

    def transient(message):
        return bool(re.search(r'(?<!\d)(?:403|5\d\d)(?!\d)', message))

    try:
        status = subprocess.run(['kaggle', 'datasets', 'status', RESULT_DS],
                                capture_output=True, text=True, timeout=30)
        verb = 'version' if status.returncode == 0 else 'create'
        cmd = (['kaggle', 'datasets', 'version', '-p', str(out), '-m', f'EXP159 epoch {ep + 1}']
               if verb == 'version' else ['kaggle', 'datasets', 'create', '-p', str(out)])
        result = subprocess.run(cmd, capture_output=True, text=True)
        message = response_text(result).lower()
        if verb == 'create' and result.returncode and 'already exists' in message:
            result = subprocess.run(['kaggle', 'datasets', 'version', '-p', str(out),
                                     '-m', f'EXP159 epoch {ep + 1}'],
                                    capture_output=True, text=True)
            message = response_text(result).lower()
        if result.returncode or 'dataset creation error' in message or 'dataset versioning error' in message:
            return warning(f'upload failed: {message[-300:]}')

        deadline = time.monotonic() + max_wait
        transient_failures = 0
        while time.monotonic() < deadline:
            status = subprocess.run(['kaggle', 'datasets', 'status', RESULT_DS],
                                    capture_output=True, text=True,
                                    timeout=max(1, min(30, deadline - time.monotonic())))
            message = response_text(status)
            if status.returncode == 0 and re.search(r'\bready\b', message, re.I):
                break
            if status.returncode:
                if not transient(message) or transient_failures >= 2:
                    return warning(f'dataset status failed: {message[-300:]}')
                transient_failures += 1
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
        else:
            return warning(f'dataset did not become ready within {max_wait}s')

        for attempt in range(3):
            try:
                uploaded = dict(dataset_files(RESULT_DS))
            except Exception as exc:
                if attempt < 2 and transient(str(exc)) and time.monotonic() < deadline:
                    time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
                    continue
                return warning(f'file listing failed: {exc}')
            if uploaded.get(CHECKPOINT) == expected_size:
                return True
            return warning(f'checkpoint missing or size differs (expected {expected_size}, '
                           f'listed {uploaded.get(CHECKPOINT)})')
    except Exception as exc:
        return warning(f'Kaggle upload/check failed: {exc}')


def remote_run(bundle_sha):
    need(sys.platform.startswith('linux') and Path('/content').is_dir(), 'Colab Linux only')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS setting missing')
    receipt = e150.verify_bundle(ROOT, bundle_sha)
    need(receipt['surface'] == 'colab' and receipt['protocol_sha256'] == e150.sha(ROOT / PROTOCOL),
         'Wrong EXP159 bundle')
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=30).stdout.strip().splitlines()
    need(len(smi) == 1 and 'T4' in smi[0], 'Colab CUDA T4 required')
    import torch
    import timm
    import numpy as np
    need(torch.cuda.is_available() and not torch.cuda.is_initialized(), 'CUDA T4 unavailable or already initialized')
    from tools.colab.exp159_remote import ensure_dataset
    corpus = ensure_dataset('taeyangg4/knee-raptor-corpus-336', Path('/content/exp159-corpus'))
    rad = ensure_dataset('marwanmath/resnet-50-radimagenet-marwan', Path('/content/exp159-rad'))
    need(any(e150.sha(p) == e150.RAD_HASH for p in rad.rglob('*') if p.is_file()),
         'RadImageNet dataset binding missing')
    e150.verify_corpus(corpus, ROOT / e150.CORPUS_LOCAL,
                       e150.read(ROOT / e150.MANIFEST)['corpus_file_list'])
    base = e150.read(ROOT / e150.BASE)
    link = ROOT / base['paths']['corpus']
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        need(link.is_symlink() and link.resolve() == corpus.resolve(), 'Unexpected corpus path')
    else:
        link.symlink_to(corpus, target_is_directory=True)
    output = Path('/content/exp159-results')
    run = output / TAG
    # If a prior epoch was published before a disconnect, bring it back first.
    from tools.colab.exp159_remote import dataset_files
    try:
        files = dataset_files(RESULT_DS)
    except (RuntimeError, ValueError, subprocess.TimeoutExpired):
        files = []
    if files and not (run / CHECKPOINT).is_file() and not (run / 'summary.json').is_file():
        ensure_dataset(RESULT_DS, run)
    if not (run / 'summary.json').exists():
        os.environ['EXP150_REMOTE'] = 'colab'
        os.environ['KNEE_WORKSPACE_ROOT'] = str(ROOT)
        os.environ['KNEE_REMOTE_RELAX_ENV'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        for key in e150.THREADS:
            os.environ[key] = '1'
        args = e150.runner_args(ROOT, 'C', 42, ROOT / e150.INITIAL, output)
        args[args.index('--workers') + 1] = '2'
        cmd = [sys.executable, str(ROOT / DERIVED), *args]
        subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), check=True)
    hardware = dict(physical_gpu=smi[0], driver=smi[0].split(',')[-1].strip(),
                    python=sys.version.split()[0], torch=torch.__version__, timm=timm.__version__,
                    numpy=np.__version__, cuda=torch.version.cuda)
    prov = e150.read(run / 'provenance.json')
    summary = e150.read(run / 'summary.json')
    prov['exp150'].update(surface='colab', hardware=hardware, bundle_sha256=bundle_sha,
                          manifest_sha256=e150.sha(ROOT / e150.MANIFEST),
                          protocol_sha256=e150.sha(ROOT / PROTOCOL))
    summary['provenance'] = prov
    put_json(run / 'provenance.json', prov)
    put_json(run / 'summary.json', summary)
    put_json(run / 'exp150-run.json', dict(schema='exp150-run-v1', state='COMPLETE', arm='C', seed=42,
             surface='colab', hardware=hardware, bundle_sha256=bundle_sha,
             initial_state_sha256=prov['initial_state_sha256'],
             initial_file_sha256=e150.sha(ROOT / e150.INITIAL),
             weight_csv_sha256=e150.sha(ROOT / e150.WEIGHTS),
             oof_k36_sha256=e150.sha(run / 'oof_k36.csv')))
    names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt',
             'oof_best.csv', 'oof_k36.csv', 'exp150-run.json']
    put_json(run / 'output-inventory.json', {n: dict(sha256=e150.sha(run / n), bytes=(run / n).stat().st_size)
                                               for n in names})
    verify_run(run, ROOT)
    publish_epoch(run, 20)
    return dict(state='COMPLETE', surface='colab', dataset=RESULT_DS, hardware=hardware)


def verify_run(run, root=ROOT):
    root, run = Path(root), Path(run)
    m, p = e150.registration(root)
    need(run.name == TAG, 'Wrong run tag')
    inv = e150.read(run / 'output-inventory.json')
    required = {'summary.json', 'provenance.json', 'history.json', 'model_best.pt',
                'oof_best.csv', 'oof_k36.csv', 'exp150-run.json'}
    need(set(inv) == required, 'Incomplete run inventory')
    for name, rec in inv.items():
        need(e150.sha(run / name) == rec['sha256'] and (run / name).stat().st_size == rec['bytes'],
             'Run output binding mismatch: ' + name)
    summary, prov, receipt = (e150.read(run / x) for x in ('summary.json', 'provenance.json', 'exp150-run.json'))
    need(receipt['surface'] == 'colab' and receipt['state'] == 'COMPLETE' and
         receipt['arm'] == 'C' and receipt['seed'] == 42 and
         receipt['oof_k36_sha256'] == e150.sha(run / 'oof_k36.csv') and
         receipt['initial_file_sha256'] == m['initial_state_file_sha256'] and
         receipt['weight_csv_sha256'] == m['weights_sha256'], 'Run receipt mismatch')
    need(prov['exp150']['surface'] == 'colab' and 'T4' in prov['exp150']['hardware']['physical_gpu'] and
         prov['exp150']['initial_state_file_sha256'] == m['initial_state_file_sha256'] and
         prov['exp150']['consensus_weights_sha256'] == m['weights_sha256'] and
         prov['initial_state_sha256'] == m['initial_state_sha256'] and
         prov['corpus_manifest_sha256'] == e150.CORPUS_HASH and
         prov['train_ids_ordered_sha256'] == m['roster']['train_sha256'] and
         prov['eval_ids_ordered_sha256'] == m['roster']['val_sha256'] and
         prov['actual_pos_weight'] == p['folds']['0']['pos_weight'], 'Colab provenance mismatch')
    need(summary['provenance'] == prov and summary['selected_epoch_1based'] == 20 and
         summary['optimizer_updates'] == p['folds']['0']['steps'] and summary['rows'] == 888 and
         summary['skipped_updates'] <= p['folds']['0']['max_fp16_skips'] and
         [h['epoch'] for h in summary['history']] == list(range(20)) and
         e150.read(run / 'history.json') == summary['history'], 'Incomplete fixed-final fit')
    args = summary['args']
    need(all(args.get(k) == v for k, v in p['common_args'].items()) and
         args['seed'] == 42 and args['arm'] == 'C' and args['variant'] == 'AUG1' and
         args['fold'] == 0 and args['mode'] == 'full' and args['tag'] == TAG and
         summary['train'] == p['folds']['0']['n_train'] and
         summary['val'] == p['folds']['0']['n_val'] and summary['gold'] == 0,
         'Recorded EXP150 control recipe mismatch')
    import csv
    ids = e150.read(root / p['folds']['0']['ordered_ids']['path'])['val']
    with (run / 'oof_k36.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames == [e150.UID, *e150.T], 'OOF columns changed')
        got = []
        for row in reader:
            got.append(row[e150.UID])
            need(all(0 <= float(row[t]) <= 1 for t in e150.T), 'Invalid OOF probability')
    need(got == ids, 'V888 UID order changed')
    return {n: rec['sha256'] for n, rec in inv.items()}


def collect(source, root=ROOT):
    source, root = Path(source), Path(root)
    run = source / TAG if (source / TAG).is_dir() else source
    verify_run(run, root)
    dest = root / EXP / 'outputs' / 'colab' / TAG
    need(not dest.exists(), 'Collected run exists')
    shutil.copytree(run, dest)
    verify_run(dest, root)
    return dict(state='VERIFIED', destination=str(dest), oof_sha256=e150.sha(dest / 'oof_k36.csv'))


def admission(scores):
    need(set(scores) == set(REFS), 'Admission requires all four references')
    return all(abs(scores[name]['delta']) <= LIMITS[name] for name in REFS)


def compare(root=ROOT):
    import numpy as np
    from tools import exp115_vlm_eval as ev
    from tools import exp125_radr50_k16_local as e125
    root = Path(root)
    m, p = e150.registration(root)
    run = root / EXP / 'outputs' / 'colab' / TAG
    verify_run(run, root)
    ids = e150.read(root / p['folds']['0']['ordered_ids']['path'])['val']
    for name, digest in m['readout_sha256'].items():
        need(e150.sha(root / name) == digest, 'Readout binding mismatch: ' + name)
    refs = {}
    jev = e150.table(root / e150.JEV, ids)
    refs['jev_hybrid'] = (jev, np.ones(jev.shape, bool), 1)
    aq = e150.table(root / e150.AQUILA, ids)
    refs['aquila_v1'] = ((aq >= .5).astype(float), np.ones(aq.shape, bool), 1)
    dread = e150.table(root / e150.SETS['dread'], ids)
    refs['dread'] = (np.rint(dread.astype(np.float32)).astype(float), np.ones(dread.shape, bool), 1)
    stack = np.stack([e150.table(root / path, ids) for path in e150.SETS.values()])
    valid = np.isfinite(stack).all(axis=0)
    binary = stack >= .5
    agree = valid & (binary.all(axis=0) | (~binary).all(axis=0))
    refs['consensus'] = (binary[0].astype(float), agree, 2)
    colab = e125.load_table(run / 'oof_k36.csv', ids, prediction=True)
    local_path = root / 'artifacts/experiments/EXP127_AUG1_S42_F0/oof_k36.csv'
    local = e125.load_table(local_path, ids, prediction=True)
    scores = {}
    for name in REFS:
        a = ev.score(colab, refs[name])['macro']
        b = ev.score(local, refs[name])['macro']
        need(np.isfinite(a) and np.isfinite(b) and
             abs(b - REFERENCE_4070[name]) <= .00006,
             'Reference scoring does not reproduce D-B-085: ' + name)
        delta = a - b
        scores[name] = dict(colab=a, local_4070_s42=b, delta=delta, abs_delta=abs(delta),
                            max_abs_delta=LIMITS[name], pass_gate=abs(delta) <= LIMITS[name],
                            d_b_085_rounded=dict(local_4070_s42=REFERENCE_4070[name], kaggle_t4_c42=REFERENCE_T4[name]))
    result = dict(schema='exp159-compare-v1', verdict='ADMIT' if admission(scores) else 'HOLD',
                  scores=scores, colab_oof_sha256=e150.sha(run / 'oof_k36.csv'),
                  local_oof_sha256=e150.sha(local_path), protocol_sha256=e150.sha(root / PROTOCOL),
                  manifest_sha256=e150.sha(root / e150.MANIFEST), rows=len(ids))
    out = root / EXP / 'comparison.json'
    need(not out.exists(), 'Comparison exists; refuse overwrite')
    put_json(out, result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('stage', 'verify-stage', 'bootstrap-cell'):
        sub.add_parser(name).add_argument('--out', required=True)
    sub.add_parser('remote-run').add_argument('--bundle-sha', required=True)
    sub.add_parser('collect').add_argument('--source', required=True)
    sub.add_parser('compare')
    a = ap.parse_args(argv)
    if a.command == 'stage': value = stage(a.out)
    elif a.command == 'verify-stage': value = verify_stage(a.out)
    elif a.command == 'bootstrap-cell': value = bootstrap_cell(a.out)
    elif a.command == 'remote-run': value = remote_run(a.bundle_sha)
    elif a.command == 'collect': value = collect(a.source)
    else: value = compare()
    print(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
