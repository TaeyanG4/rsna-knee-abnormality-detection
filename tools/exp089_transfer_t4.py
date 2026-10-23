"""CPU package preparation and explicit private T4 engineering, never submission.

Each recipe runs in a fresh child on the same stored engineering100 UIDs. V8
probabilities check cross-device numerics without labels or score computation.
Existing DICOM byte evidence is a reference, not a repeated standalone audit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / '.local/exp089-recipe-transfer/t4-v3'
DATASET = 'taeyangg4/knee-exp089-recipe-transfer-v1'
OPENCV_SHA = '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)


def find_competition(roots=None):
    roots = roots if roots is not None else [
        Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
        Path('/kaggle/input/rsna-knee-abnormality-detection')]
    found = [Path(p) for p in roots if (Path(p) / 'train_series.csv').is_file()
             and (Path(p) / 'train_series').is_dir()]
    need(len(found) == 1, 'Actual competition path missing or ambiguous')
    return found[0]


def prepare():
    local = read(ROOT / 'artifacts/experiments/EXP089_RECIPE_TRANSFER_G1_PROTOCOL_V1/manifest.json')
    acceptance = read(ROOT / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/local-root-acceptance.json')
    need(acceptance['status'] == 'PASS_LOCAL_ENGINEERING_ONLY', 'Completed local check required')
    assets = ROOT / '.local/b-rn3f-staging/20261004-v1/assets'
    previous = read(assets / 'b-rn3f-assets.json')
    old_reference = read(ROOT / '.local/b-rn3f-runs/20261004-v1-t4-evidence/work/producer/result/receipt.json')
    ids = read(assets / 'engineering100.json')['ids']
    need(len(ids) == len(set(ids)) == 100, 'Exact original engineering100 required')
    need([row['uid'] for row in old_reference['runtime']['input_evidence']] == ids,
         'Stored input proof order differs')
    STAGE.mkdir(parents=True, exist_ok=False)
    dataset = STAGE / 'dataset'
    dataset.mkdir()
    files = {}

    def copy(source, relative, expected):
        target = dataset / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = sha(target)
        need(digest == expected, 'New package copy differs: ' + relative)
        files[relative] = digest

    for relative in ['config/workspace.toml', 'b_rn3f_serving.py',
                     'training/src/raptor_common.py', 'training/src/raptor_runtime.py',
                     'training/src/raptor_dicom.py', 'training/src/raptor_ensemble.py',
                     *[row['path'] for row in previous['wheels']]]:
        copy(assets / relative, relative, previous['files'][relative])
    copy(Path(__file__), 'exp089_transfer_t4.py', sha(Path(__file__)))
    copy(ROOT / 'tools/process_resource_receipt.py', 'process_resource_receipt.py',
         local['bindings']['tools/process_resource_receipt.py'])
    arms = []
    for row in local['arms']:
        arm = dict(row)
        arm['checkpoint'] = 'weights/' + row['name'] + '.pt'
        copy(ROOT / row['checkpoint'], arm['checkpoint'], local['bindings'][row['checkpoint']])
        arm['reference'] = 'reference/' + row['name'] + '-engineering.csv'
        ref = ROOT / '.local/exp089-recipe-transfer/local-v1/worker' / (row['name'] + '-engineering.csv')
        copy(ref, arm['reference'], acceptance['binding'][ref.relative_to(ROOT).as_posix()])
        arms.append(arm)
    protocol = {'schema': 'exp089-transfer-t4-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'PREPARED_NOT_EXECUTED', 'arms': arms, 'ids': ids, 'reference_ids': local['uids'],
        'targets': local['targets'], 'input_reference': old_reference['runtime']['input_evidence'],
        'source_byte_proof': 'original completed engineering100 used as reference; no new standalone census/audit',
        'contract': local['corpus_contract'], 'norm': 'imagenet', 'window_policy': 'within-slot-valid',
        'k_eval': 24, 'amp': 'fp16', 'batch': 1, 'runtime_policy': 'strict',
        'tolerances': local['tolerances'], 'files': files, 'wheels': previous['wheels'],
        'opencv_sha256': OPENCV_SHA, 'child_timeout_seconds': 900, 'platform_seconds': 2100,
        'postrun_RSS_bytes': 8 * 2**30, 'postrun_output_bytes': 64 * 2**20,
        'resource_limits_note': 'Sampled RSS/output acceptance; timeout monitored, no RSS/output OS kill cap',
        'retries': 0, 'CPU_request': 12, 'CPU_actual_cap': 4, 'GPU_visible': 1,
        'allocation_exposure': 'Account for all platform T4s, even if only device0 is visible',
        'cold_scope': 'Fresh worker/model load, OS file cache not cleared; second arm may see warmed files',
        'accuracy_claim': False, 'labels_Gold_training': False,
        'local_acceptance_sha256': sha(ROOT / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/local-root-acceptance.json')}
    write(dataset / 'exp089-t4-protocol.json', protocol)
    digest = sha(dataset / 'exp089-t4-protocol.json')
    write(dataset / 'dataset-metadata.json', {'title': 'EXP089 fixed recipe transfer private engineering',
          'id': DATASET, 'licenses': [{'name': 'other'}], 'isPrivate': True,
          'description': 'Own fixed final G1 weights and immutable engineering sources; private research only.'})
    # Known mount alternatives only; never recurse through the entire DICOM tree.
    code = """import json, os, subprocess, sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
candidates=[Path('/kaggle/input/knee-exp089-recipe-transfer-v1'),Path('/kaggle/input/datasets/taeyangg4/knee-exp089-recipe-transfer-v1')]
assets=next(p for p in candidates if (p/'exp089-t4-protocol.json').is_file())
opencv_dirs=[Path('/kaggle/input/opencv-python-headless-4120088-x86'),Path('/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86')]
opencv=next(w for p in opencv_dirs if p.is_dir() for w in p.glob('*.whl'))
out=Path('/kaggle/working/exp089-t4-v1');out.mkdir(exist_ok=False)
protocol=assets/'exp089-t4-protocol.json'
for arm in ['B224','R384']:
    cmd=[sys.executable,str(assets/'process_resource_receipt.py'),'--out',str(out/arm/'supervisor'),'--timeout','900','--',sys.executable,str(assets/'exp089_transfer_t4.py'),'worker','--assets',str(assets),'--protocol',str(protocol),'--sha256',DIGEST,'--arm',arm,'--out',str(out/arm/'worker'),'--opencv',str(opencv)]
    subprocess.run(cmd,check=True)
    resource=json.loads((out/arm/'supervisor/receipt.json').read_text())
    receipt=json.loads((out/arm/'worker/receipt.json').read_text())
    assert resource['status']=='COMPLETE' and resource['child_exit_code']==0 and not resource['cleanup']['errors'] and not resource['cleanup']['survivors']
    assert resource['sampled_tree_rss_max_bytes']<=8*2**30 and receipt['status']=='PASS'
    assert sum(p.stat().st_size for p in (out/arm).rglob('*') if p.is_file())<=64*2**20
print(json.dumps({'status':'PASS_ENGINEERING_ONLY','models':2,'labels_training_scoring':0,'not_submission':True}))
""".replace('DIGEST', repr(digest))
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {},
                'cells': [{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [code]}]}
    nb = STAGE / 'notebook'
    nb.mkdir()
    write(nb / 'trial.ipynb', notebook)
    metadata = read(ROOT / '.local/b-rn3f-t4-staging/20261004-v1/notebook/kernel-metadata.json')
    metadata.update(id='taeyangg4/knee-exp089-fixed-recipe-t4-v3', title='knee-exp089-fixed-recipe-t4-v3',
                    code_file='trial.ipynb', dataset_sources=[DATASET, 'mattiaangeli/opencv-python-headless-4120088-x86'])
    write(nb / 'kernel-metadata.json', metadata)
    plan = {'status': 'SEPARATE_SOURCE_BUDGET_REVIEW_AND_LIVE_QUOTA_REQUIRED_NOT_EXECUTED',
        'protocol_sha256': digest, 'notebook_sha256': sha(nb / 'trial.ipynb'),
        'metadata_sha256': sha(nb / 'kernel-metadata.json'), 'source_sha256': files['exp089_transfer_t4.py'],
        'dataset_bytes': sum(p.stat().st_size for p in dataset.rglob('*') if p.is_file()),
        'private_dataset': DATASET, 'platform_seconds': 2100,
        'T4x2_accounted_max_GPU_hours': 2 * 2100 / 3600,
        'upload_argv': [str(ROOT / '.venv/Scripts/kaggle.exe'), 'datasets', 'create', '-p', str(dataset), '--dir-mode', 'zip'],
        'push_argv': [str(ROOT / '.venv/Scripts/kaggle.exe'), 'kernels', 'push', '-p', str(nb), '--timeout', '2100'],
        'submission': False, 'public': False, 'paid': False}
    write(STAGE / 'execution-plan.json', plan)
    print(json.dumps(plan))


def worker(a):
    assets, out = Path(a.assets), Path(a.out)
    need(sha(a.protocol) == a.sha256, 'Exact T4 protocol required')
    p = read(a.protocol)
    need(p['schema'] == 'exp089-transfer-t4-v1', 'Wrong schema')
    arm = next(row for row in p['arms'] if row['name'] == a.arm)
    # Each child reads only its model; immutable shared sources/reference are bound.
    for name, digest in p['files'].items():
        if name.startswith('weights/') and name != arm['checkpoint']:
            continue
        need(sha(assets / name) == digest, 'T4 package changed: ' + name)
    need(sha(a.opencv) == p['opencv_sha256'], 'Pinned OpenCV wheel required')
    need(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 12), 'Pinned Linux CPython3.12 required')
    competition = find_competition()
    out.mkdir(parents=True, exist_ok=False)
    envdir = Path('/kaggle/temp') / ('exp089-env-' + a.arm)
    need(not envdir.exists(), 'Fresh dependency directory required')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', KNEE_WORKSPACE_ROOT=str(assets))
    spec = importlib.util.spec_from_file_location('exp089_passive_observer', assets / 'b_rn3f_serving.py')
    passive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(passive)
    allocation = passive.cpu_allocation(12)
    threads = min(4, allocation['selected_threads'])
    os.environ.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    receipt = {'status': 'RUNNING', 'arm': a.arm, 'protocol_sha256': a.sha256, 'cleanup_errors': [],
               'model_runs': 1, 'training': 0, 'scoring': 0, 'label_Gold_reads': 0, 'retries': 0}
    source, model, started = None, None, time.perf_counter()
    try:
        wheels = [str(assets / row['path']) for row in p['wheels']] + [a.opencv]
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--target', str(envdir), *wheels], check=True)
        sys.path[:0] = [str(envdir), str(assets / 'training/src')]
        import numpy as np
        import pandas as pd
        import torch
        import timm, cv2, pydicom, gdcm
        need((timm.__version__, cv2.__version__, pydicom.__version__, gdcm.Version.GetVersion()) ==
             ('1.0.19', '4.12.0', '3.0.2', '3.2.6'), 'Pinned offline dependencies required')
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
             'T4' in torch.cuda.get_device_name(0), 'One visible T4 required')
        from raptor_common import LAB, load_checkpoint, write_predictions, environment
        from raptor_runtime import configure_runtime, predict
        from raptor_dicom import DirectoryDicomSource, DicomCorpus
        runtime = configure_runtime('fp16', 'cuda', 'strict')
        need(LAB == p['targets'], 'Exact12 target order required')
        t = time.perf_counter()
        model, meta = load_checkpoint(assets / arm['checkpoint'])
        expected = {'arch': arm['arch'], 'res': arm['res'], 'epoch': arm['epoch_0based'], 'lab': LAB,
                    'contract': p['contract'], 'k_eval': 24, 'norm': 'imagenet', 'window_policy': 'within-slot-valid'}
        need(all(meta.get(k) == v for k, v in expected.items()), 'Checkpoint metadata differs')
        receipt['model_load_seconds'] = time.perf_counter() - t
        rows = pd.read_csv(competition / 'train_series.csv',
                          dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
        source = DirectoryDicomSource(competition / 'train_series')
        corpus = DicomCorpus.from_records(p['ids'], rows.to_dict('records'), source, p['contract'], 'train')
        observed = passive.ObservedCorpus(corpus)
        probs, timing = predict(model, observed, p['ids'], 24, arm['res'], 'fp16', 'cuda',
                                norm='imagenet', batch=1, window_policy='within-slot-valid')
        need(probs.dtype == np.float32 and probs.shape == (100, 12) and
             bool(np.isfinite(probs).all()) and timing['fp32_fallbacks'] == 0, 'Incomplete/nonfinite inference')
        need([observed.records[u] for u in p['ids']] == p['input_reference'], 'Frozen336 reader/input byte proof differs')
        with (assets / arm['reference']).open(encoding='utf-8', newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        need([r['StudyInstanceUID'] for r in csv_rows] == p['reference_ids'], 'Reference8 order differs')
        need(list(csv_rows[0]) == ['StudyInstanceUID', *LAB], 'Reference12 order differs')
        reference = np.asarray([[float(r[t]) for t in LAB] for r in csv_rows], dtype=np.float64)
        local = probs[[p['ids'].index(u) for u in p['reference_ids']]].astype(np.float64)
        delta = np.abs(local - reference)
        errors = {'max_abs': float(delta.max()), 'p99_abs': float(np.quantile(delta, .99)), 'mean_abs': float(delta.mean())}
        need(all(errors[k] <= p['tolerances'][k] for k in errors), 'T4 versus local8 fixed numerical tolerance failed')
        write_predictions(out / 'engineering100.csv', p['ids'], probs)
        receipt.update(status='PASS', studies=100, reference_studies=8, probability_errors=errors,
                       tolerances=p['tolerances'], timing=timing, runtime=runtime, environment=environment(),
                       input_byte_reference_exact=True, allocation={**allocation, 'selected_internal_threads': threads},
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cold_scope=p['cold_scope'], DICOM_receipts=corpus.receipts)
    except BaseException as exc:
        receipt.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        for label, action in [('GPU', lambda: model.to('cpu') if model is not None else None),
                              ('source', lambda: source.close() if source is not None else None)]:
            try:
                action()
            except Exception as exc:
                receipt['cleanup_errors'].append(label + ': ' + str(exc))
        if 'torch' in sys.modules:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as exc:
                receipt['cleanup_errors'].append(str(exc))
        if receipt['cleanup_errors']:
            receipt['status'] = 'FAIL'
        receipt['worker_wall_seconds'] = time.perf_counter() - started
        write(out / 'receipt.json', receipt)
    need(receipt['status'] == 'PASS', 'Engineering/cleanup failed')
    print(json.dumps({'status': receipt['status'], 'arm': a.arm, 'seconds': receipt['worker_wall_seconds']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    w = sub.add_parser('worker')
    for key in ['assets', 'protocol', 'sha256', 'arm', 'out', 'opencv']:
        w.add_argument('--' + key, required=True)
    a = parser.parse_args()
    prepare() if a.command == 'prepare' else worker(a)


if __name__ == '__main__':
    main()
