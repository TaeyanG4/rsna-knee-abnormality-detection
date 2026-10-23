"""Fixed B224/R384 test-only release, reusing the completed EXP089 T4 assets.

v3 (2026-10-06, after an independent review of v2): the cgroup memory gate no longer fails on
memory.events max/high (page cache charged to the container cgroup grows ~75-110 MB per study and
reaches memory.max after ~300 studies without any OOM); oom/oom_kill and the sampled RSS <= 8 GiB
gate stay fatal; cgroup telemetry gaps are recorded, not fatal. Everything else equals v2.
v2 (2026-10-06, Claude Code review of v1): identical compute/publication path;
only the output budget (v1 64MiB failed hidden N >~2200 because per-study DICOM
receipts are ~30KB) and the push timeout (one 32400s version serves Public3 and
the competition rerun, so the scored version is the one observed) change.

prepare reads metadata/source bytes only; it never loads models or predictions.
run requires the exact packaged manifest and publishes only after natural child
exit, full ordered test output, source/resource and cleanup validation. No ranks,
training, labels, precision fallback, retry or candidate selection occurs here.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'taeyangg4/knee-exp089-recipe-transfer-v1'
ASSETS_PROTOCOL_SHA = '3e0e5529191570955b3229458adf511073401dc354de04b33164988f830687a8'
SCHEMA = 'exp089-test-release-v3'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
TIME = {'engineering100_supervisor_seconds': 652.503319258, 'inflation': 1.3,
        'fixed_seconds': 300, 'screen_seconds': 30600, 'full_seconds': 32100,
        'public3_seconds': 600, 'finalization_reserve_seconds': 30,
        'platform_public_seconds': 900, 'platform_full_seconds': 32400}
RSS = 8 * 2**30
OUTPUT = 1 * 2**30


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    need(len(rows) > 1 and len(rows[0]) == len(set(rows[0])), 'Empty or duplicate header')
    need(all(len(r) == len(rows[0]) for r in rows[1:]), 'CSV width differs')
    return rows[0], rows[1:]


def safe_ids(values):
    ids = list(values)
    need(bool(ids) and all(isinstance(u, str) and u and not any(c in u for c in
         ['/', '\\', ':', '\x00', '..']) for u in ids), 'Unsafe/empty UID')
    need(len(ids) == len(set(ids)), 'Duplicate UID')
    return ids


def test_contract(competition):
    competition = Path(competition)
    need((competition / 'test_series').is_dir(), 'Missing test_series directory')
    th, tr = table(competition / 'test.csv')
    need(UID in th, 'Test UID absent')
    test_ids = safe_ids(r[th.index(UID)] for r in tr)
    sh, sr = table(competition / 'sample_submission.csv')
    need(sh == [UID, *TARGETS], 'Official12 column order differs')
    ids = safe_ids(r[0] for r in sr)
    need(set(ids) == set(test_ids), 'Sample/test UID membership differs')
    rh, rr = table(competition / 'test_series.csv')
    need(UID in rh and 'SeriesInstanceUID' in rh, 'Test series schema differs')
    need({r[rh.index(UID)] for r in rr} <= set(ids), 'Unexpected series study')
    safe_ids(dict.fromkeys(r[rh.index('SeriesInstanceUID')] for r in rr))
    return {'ids': ids, 'studies': len(ids), 'columns': sh,
            'metadata_sha256': {n: sha(competition / n) for n in
                                ['test.csv', 'test_series.csv', 'sample_submission.csv']}}


def operational_plan(n):
    need(type(n) is int and n > 0, 'Positive count required')
    projected = TIME['inflation'] * TIME['engineering100_supervisor_seconds'] * n / 100 + TIME['fixed_seconds']
    need(projected <= TIME['screen_seconds'], 'N-dependent time screen failed')
    return {'studies': n, 'projected_seconds': projected,
            'internal_seconds': TIME['public3_seconds'] if n <= 3 else TIME['full_seconds'],
            'scope': 'Conservative engineering scenario, not hidden runtime proof; R384 may have warmed files'}


def csv_check(path, ids):
    header, rows = table(path)
    need(header == [UID, *TARGETS] and [r[0] for r in rows] == ids, 'Incomplete/reordered output')
    need(all(math.isfinite(float(x)) and 0 <= float(x) <= 1 for r in rows for x in r[1:]),
         'Nonfinite/out-of-range output')
    return sha(path)


def checked_manifest(path, digest):
    need(sha(path) == digest, 'Release manifest changed')
    p = read(path)
    need(p['schema'] == SCHEMA and p['time_policy'] == TIME and p['asset_protocol_sha256'] == ASSETS_PROTOCOL_SHA,
         'Release policy differs')
    need(sha(__file__) == p['release_source_sha256'], 'Executing release source changed')
    return p


def validate_resource(s, argv, worker_pid, timeout):
    need(s['schema'] == 'process-resource-receipt-v1' and s['status'] == 'COMPLETE'
         and s['failure'] is None and s['child_exit_code'] == s['child_exit_code_before_cleanup']
         == s['wrapper_exit_code'] == 0, 'Child/resource failed')
    need(s['argv'] == argv and s['child_identity']['pid'] == worker_pid, 'Child identity differs')
    need(all(not s['cleanup'][k] for k in ['actions', 'errors', 'survivors']), 'Non-natural cleanup')
    need(s['retries'] == 0 and s['timeout_seconds'] == timeout and 0 <= s['elapsed_seconds'] < timeout,
         'Deadline/retry differs')
    need(s['sample_count'] > 0 and 0 < s['sampled_tree_rss_max_bytes'] <= RSS
         and s['failed_sample_attempts'] == s['access_failure_count'] == 0
         and not s['access_failures'], 'Incomplete/excess resource observation')
    telemetry = {'cgroup_telemetry_complete': bool(s.get('cgroup_telemetry_complete')), 'snapshots': {}}
    for key in ['cgroup_start', 'cgroup_end']:
        snapshot = s.get(key) or {}
        v = snapshot.get('values') or {}
        events = v.get('memory.events') or {}
        telemetry['snapshots'][key] = {'errors': snapshot.get('errors'), 'memory.events': events, 'memory.peak': v.get('memory.peak'), 'memory.max': v.get('memory.max')}
        # Fatal only on a real OOM; max/high count reclaim pressure from page cache and are telemetry.
        need(all(events.get(k, 0) == 0 for k in ['oom', 'oom_kill']), 'Memory OOM event')
    return telemetry


def output_bytes(work):
    # A single stat avoids is_file/stat replacement races; other errors propagate.
    import stat
    total = 0
    for path in Path(work).rglob('*'):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def publish(raw, destination, expected_sha, deadline, final_check):
    """Own temp/final transaction; rollback also covers post-rename failures."""
    destination = Path(destination)
    need(not destination.exists(), 'Preserve existing final CSV')
    temporary = destination.with_name(destination.name + '.exp089-tmp')
    need(not temporary.exists(), 'Preserve existing temp CSV')
    published = False
    try:
        shutil.copyfile(raw, temporary)
        need(sha(temporary) == expected_sha and time.monotonic() < deadline, 'Copy/deadline failed')
        final_check()
        temporary.rename(destination)
        published = True
        need(sha(destination) == expected_sha and time.monotonic() < deadline, 'Final/deadline failed')
        final_check()
    except BaseException:
        if published and destination.is_file() and sha(destination) == expected_sha:
            destination.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def worker(a):
    p = checked_manifest(a.manifest, a.sha256)
    assets, work = Path(a.assets), Path(a.work)
    need(sha(assets / 'exp089-t4-protocol.json') == ASSETS_PROTOCOL_SHA, 'Frozen assets protocol changed')
    old = read(assets / 'exp089-t4-protocol.json')
    arm = next(x for x in old['arms'] if x['name'] == p['arm'])
    for name in ['b_rn3f_serving.py', 'process_resource_receipt.py', arm['checkpoint'],
                 *['training/src/' + n for n in ['raptor_common.py', 'raptor_runtime.py', 'raptor_dicom.py', 'raptor_ensemble.py']],
                 *[r['path'] for r in old['wheels']]]:
        need(sha(assets / name) == old['files'][name], 'Selected immutable asset changed: ' + name)
    need(sha(a.opencv) == old['opencv_sha256'], 'OpenCV wheel changed')
    need(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 12), 'Linux CPython3.12 required')
    context = read(work / 'launch.json')
    contract = test_contract(a.competition)
    need(contract == context['test_contract'], 'Test metadata changed')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', KNEE_WORKSPACE_ROOT=str(assets))
    spec = importlib.util.spec_from_file_location('exp089_passive_cpu', assets / 'b_rn3f_serving.py')
    passive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(passive)
    allocation = passive.cpu_allocation(12)
    threads = min(4, allocation['selected_threads'])
    os.environ.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    result = work / 'worker'
    result.mkdir()
    r = {'status': 'FAIL', 'pid': os.getpid(), 'arm': p['arm'], 'release_manifest_sha256': a.sha256,
         'cleanup_errors': [], 'training': 0, 'label_Gold_reads': 0, 'scoring': 0, 'retries': 0}
    source = model = None
    try:
        envdir = Path('/kaggle/temp') / ('exp089-release-env-' + p['arm'])
        need(not envdir.exists(), 'New dependency dir required')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target',
                        str(envdir), *[str(assets / x['path']) for x in old['wheels']], a.opencv], check=True)
        sys.path[:0] = [str(envdir), str(assets / 'training/src')]
        import numpy as np
        import pandas as pd
        import torch, timm, cv2, pydicom, gdcm
        need((torch.__version__, timm.__version__, cv2.__version__, pydicom.__version__, gdcm.Version.GetVersion())
             == ('2.10.0+cu128', '1.0.19', '4.12.0', '3.0.2', '3.2.6'), 'Completed T4 dependency versions required')
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
             'T4' in torch.cuda.get_device_name(0), 'One visible T4 required')
        from raptor_common import LAB, load_checkpoint, write_predictions, environment
        from raptor_runtime import configure_runtime, predict
        from raptor_dicom import DirectoryDicomSource, DicomCorpus
        runtime = configure_runtime('fp16', 'cuda', 'strict')
        need(LAB == TARGETS, 'Target order changed')
        model, meta = load_checkpoint(assets / arm['checkpoint'])
        expected = {'arch': arm['arch'], 'res': arm['res'], 'epoch': arm['epoch_0based'], 'lab': LAB,
                    'contract': old['contract'], 'k_eval': 24, 'norm': 'imagenet', 'window_policy': 'within-slot-valid'}
        need(all(meta.get(k) == v for k, v in expected.items()), 'Checkpoint metadata differs')
        rows = pd.read_csv(Path(a.competition) / 'test_series.csv', dtype={UID: str, 'SeriesInstanceUID': str})
        source = DirectoryDicomSource(Path(a.competition) / 'test_series')
        corpus = DicomCorpus.from_records(contract['ids'], rows.to_dict('records'), source, old['contract'], 'test')
        probs, timing = predict(model, corpus, contract['ids'], 24, arm['res'], 'fp16', 'cuda',
                                norm='imagenet', batch=1, window_policy='within-slot-valid')
        need(probs.dtype == np.float32 and probs.shape == (contract['studies'], 12)
             and bool(np.isfinite(probs).all()) and timing['fp32_fallbacks'] == 0, 'Incomplete/nonfinite inference')
        write_predictions(result / 'raw.csv', contract['ids'], probs)
        r.update(status='PASS', studies=contract['studies'], targets=LAB, runtime=runtime, timing=timing,
                 allocation={**allocation, 'selected_internal_threads': threads, 'data_loader_workers': 0},
                 checkpoint_sha256=old['files'][arm['checkpoint']], environment=environment(),
                 raw_sha256=sha(result / 'raw.csv'), source_bindings={n: old['files'][n] for n in old['files']
                     if n.startswith('training/src/')}, test_contract=contract,
                 DICOM_receipts=corpus.receipts, CUDA_peak_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        r['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        for label, action in [('model', lambda: model.to('cpu') if model is not None else None),
                              ('source', lambda: source.close() if source is not None else None)]:
            try:
                action()
            except Exception as exc:
                r['cleanup_errors'].append(label + ': ' + str(exc))
        if 'torch' in sys.modules:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as exc:
                r['cleanup_errors'].append(str(exc))
        if r['cleanup_errors']:
            r['status'] = 'FAIL'
        write(result / 'receipt.json', r)
    need(r['status'] == 'PASS', 'Worker cleanup failed')


def run(a):
    p = checked_manifest(a.manifest, a.sha256)
    work, assets, destination = Path(a.work), Path(a.assets), Path(a.out)
    need(not work.exists() and not destination.exists(), 'Preserve previous outputs')
    work.mkdir(parents=True)
    result = {'status': 'FAIL', 'schema': SCHEMA, 'arm': p['arm'], 'accuracy_claim': False}
    published_sha = None
    try:
        started = a.started
        need(math.isfinite(started) and 0 <= time.monotonic() - started < 600, 'Invalid bootstrap clock')
        contract = test_contract(a.competition)
        plan = operational_plan(contract['studies'])
        deadline = started + plan['internal_seconds']
        timeout = deadline - time.monotonic() - TIME['finalization_reserve_seconds']
        need(timeout > 0 and sha(assets / 'exp089-t4-protocol.json') == ASSETS_PROTOCOL_SHA, 'Budget/assets failed')
        old = read(assets / 'exp089-t4-protocol.json')
        need(sha(assets / 'process_resource_receipt.py') == old['files']['process_resource_receipt.py'], 'Supervisor changed')
        argv = [sys.executable, str(Path(__file__).resolve()), 'worker', '--manifest', a.manifest,
                '--sha256', a.sha256, '--assets', a.assets, '--work', a.work,
                '--competition', a.competition, '--opencv', a.opencv]
        context = {'test_contract': contract, 'operational_plan': plan, 'worker_argv': argv,
                   'deadline_monotonic': deadline, 'worker_timeout_seconds': timeout}
        write(work / 'launch.json', context)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        code = subprocess.run([sys.executable, str(assets / 'process_resource_receipt.py'), '--out',
              str(work / 'supervisor'), '--timeout', str(timeout), '--', *argv], env=env).returncode
        need(code == 0, 'Supervisor failed')
        w, s = read(work / 'worker/receipt.json'), read(work / 'supervisor/receipt.json')
        need(w['status'] == 'PASS' and w['release_manifest_sha256'] == a.sha256 and w['arm'] == p['arm']
             and w['test_contract'] == contract and w['studies'] == contract['studies'] and w['targets'] == TARGETS
             and not w['cleanup_errors'], 'Worker contract failed')
        need(w['timing']['all_finite'] and w['timing']['fp32_fallbacks'] == 0
             and [x['uid'] for x in w['timing']['input_tensor_hashes']] == contract['ids'], 'Input/finite trace failed')
        need(w['runtime']['effective_amp'] == 'fp16' and w['runtime']['policy'] == 'strict'
             and not w['runtime']['precision_fallback_allowed'] and not w['runtime']['backends']['cudnn_allow_tf32']
             and not w['runtime']['backends']['matmul_allow_tf32'], 'Runtime differs')
        cgroup_telemetry = validate_resource(s, argv, w['pid'], timeout)
        raw = work / 'worker/raw.csv'
        published_sha = csv_check(raw, contract['ids'])
        need(published_sha == w['raw_sha256'], 'Worker output changed')
        need(output_bytes(work) <= OUTPUT, 'Output budget exceeded')
        with zipfile.ZipFile(work / 'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for name in ['launch.json', 'worker/receipt.json', 'worker/raw.csv',
                         'supervisor/receipt.json', 'supervisor/stdout.log', 'supervisor/stderr.log']:
                z.write(work / name, name)
        need(output_bytes(work) <= OUTPUT, 'Archive output budget exceeded')
        result.update(status='PASS_TEST_COHORT_ONLY', studies=contract['studies'], columns=contract['columns'],
                      submission_sha256=published_sha, operational_plan=plan, worker_sha256=sha(work / 'worker/receipt.json'),
                      supervisor_sha256=sha(work / 'supervisor/receipt.json'), evidence_sha256=sha(work / 'evidence.zip'),
                      sampled_tree_rss_bytes=s['sampled_tree_rss_max_bytes'], cgroup_telemetry=cgroup_telemetry, elapsed_seconds=time.monotonic() - started,
                      scope='Actual supplied test cohort only; Public3 is not hidden runtime or accuracy proof')
        def final_check():
            need(test_contract(a.competition) == contract, 'Test metadata changed before finalization')
            need(output_bytes(work) <= OUTPUT, 'Final output budget exceeded')
        publish(raw, destination, published_sha, deadline, final_check)
        write(work / 'result.json', result)
        need(time.monotonic() < deadline and output_bytes(work) <= OUTPUT, 'Final acknowledgment budget failed')
        return result
    except BaseException as exc:
        if destination.is_file() and published_sha is not None and sha(destination) == published_sha:
            destination.unlink()
        result.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc)})
        write(work / 'failure.json', result)
        raise


def prepare(a):
    stage = Path(a.out)
    need(not stage.exists(), 'Choose a fresh release stage')
    assets = ROOT / '.local/exp089-recipe-transfer/t4-v3/dataset'
    need(sha(assets / 'exp089-t4-protocol.json') == ASSETS_PROTOCOL_SHA, 'Exact completed assets protocol required')
    acceptance = ROOT / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/t4-v2-root-acceptance.json'
    need(read(acceptance)['status'] == 'ROOT_T4_V2_ENGINEERING_PASS', 'Completed T4 acceptance required')
    stage.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    old = read(assets / 'exp089-t4-protocol.json')
    records = []
    for arm in ['B224', 'R384']:
        arm_dir = stage / arm
        arm_dir.mkdir()
        manifest = {'schema': SCHEMA, 'arm': arm, 'created_at': datetime.now(timezone.utc).isoformat(),
                    'asset_dataset': DATASET, 'asset_dataset_actual_version': 1,
                    'asset_protocol_sha256': ASSETS_PROTOCOL_SHA, 'release_source_sha256': sha(__file__),
                    'T4_acceptance_sha256': sha(acceptance), 'time_policy': TIME, 'RSS_bytes': RSS,
                    'output_bytes': OUTPUT, 'retries': 0, 'formula': 'single final weight raw sigmoid; no rank/blend',
                    'fixed_arms_before_Public': ['B224', 'R384'], 'study_count': 'dynamic complete official test cohort'}
        write(arm_dir / 'release-manifest.json', manifest)
        digest = sha(arm_dir / 'release-manifest.json')
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        code = """import time
started=time.monotonic()
import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
base=Path('/kaggle/working/exp089-release-ARM')
assert not base.exists() and not Path('/kaggle/working/submission.csv').exists()
base.mkdir()
source=base64.b85decode(SOURCE)
assert hashlib.sha256(source).hexdigest()==SOURCE_SHA
(base/'exp089_recipe_release.py').write_bytes(source)
(base/'release-manifest.json').write_text(MANIFEST,encoding='utf-8')
candidates=[Path('/kaggle/input/knee-exp089-recipe-transfer-v1'),Path('/kaggle/input/datasets/taeyangg4/knee-exp089-recipe-transfer-v1')]
assets=[p for p in candidates if (p/'exp089-t4-protocol.json').is_file()]
assert len(assets)==1
opencv_roots=[Path('/kaggle/input/opencv-python-headless-4120088-x86'),Path('/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86')]
opencv=[p for d in opencv_roots for p in d.glob('opencv_python_headless-4.12.0.88-*.whl')]
assert len(opencv)==1
comp_roots=[Path('/kaggle/input/rsna-knee-abnormality-detection'),Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
competition=[p for p in comp_roots if all((p/n).is_file() for n in ['test.csv','test_series.csv','sample_submission.csv']) and (p/'test_series').is_dir()]
assert len(competition)==1
argv=[sys.executable,str(base/'exp089_recipe_release.py'),'run','--manifest',str(base/'release-manifest.json'),'--sha256',DIGEST,'--assets',str(assets[0]),'--work',str(base/'work'),'--competition',str(competition[0]),'--opencv',str(opencv[0]),'--out','/kaggle/working/submission.csv','--started',str(started)]
subprocess.run(argv,check=True)
result=json.loads((base/'work/result.json').read_text())
assert result['status']=='PASS_TEST_COHORT_ONLY'
assert hashlib.sha256(Path('/kaggle/working/submission.csv').read_bytes()).hexdigest()==result['submission_sha256']
print(json.dumps(result))
""".replace('ARM', arm).replace('SOURCE_SHA', repr(sha(__file__))).replace('SOURCE', repr(base64.b85encode(source).decode()))
        code = code.replace('MANIFEST', repr(manifest_text)).replace('DIGEST', repr(digest))
        nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
              'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
              'language_info': {'name': 'python', 'version': '3.12'}}, 'cells': [
              {'cell_type': 'code', 'id': 'fixed-recipe-release', 'metadata': {},
               'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)}]}
        directory = arm_dir / 'notebook'
        directory.mkdir()
        write(directory / 'release.ipynb', nb)
        metadata = read(ROOT / '.local/exp089-recipe-transfer/t4-v4-notebook/notebook/kernel-metadata.json')
        metadata.update(id='taeyangg4/knee-exp089-' + arm.lower() + '-release-v3',
                        title='knee-exp089-' + arm.lower() + '-release-v3', code_file='release.ipynb',
                        is_private=True, enable_internet=False,
                        dataset_sources=[DATASET, 'mattiaangeli/opencv-python-headless-4120088-x86'])
        write(directory / 'kernel-metadata.json', metadata)
        record = {'arm': arm, 'manifest_sha256': digest, 'notebook_sha256': sha(directory / 'release.ipynb'),
                  'metadata_sha256': sha(directory / 'kernel-metadata.json'), 'kernel': metadata['id'],
                  'push_argv': [str(ROOT / '.venv/Scripts/kaggle.exe'), 'kernels', 'push', '-p',
                                str(directory.resolve()), '--timeout', str(TIME['platform_full_seconds'])],
                  'first_output': 'exp089-release-' + arm + '/work/result.json',
                  'evidence_output': 'exp089-release-' + arm + '/work/evidence.zip'}
        records.append(record)
    plan = {'status': 'CPU_PREPARED_SEPARATE_REVIEW_REQUIRED', 'arms': records,
            'source_sha256': sha(__file__), 'private': True, 'model_loads': 0, 'labels_Gold_reads': 0,
            'GPU_remote_network': 0, 'submission': False, 'dataset_upload': False,
            'platform_timeout_seconds': TIME['platform_full_seconds'], 'public3_internal_seconds': TIME['public3_seconds'],
            'full_platform_seconds': 32400, 'hidden_scoring_timeout': 'unknown; platform request is not a guarantee'}
    write(stage / 'execution-plan.json', plan)
    print(json.dumps({'status': plan['status'], 'stage': str(stage), 'arms': 2, 'dataset_upload': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('prepare')
    build.add_argument('--out', required=True)
    for mode in ['run', 'worker']:
        action = sub.add_parser(mode)
        for key in ['manifest', 'sha256', 'assets', 'work', 'competition', 'opencv']:
            action.add_argument('--' + key, required=True)
        if mode == 'run':
            action.add_argument('--out', required=True)
            action.add_argument('--started', type=float, required=True)
    a = parser.parse_args()
    {'prepare': prepare, 'run': run, 'worker': worker}[a.action](a)


if __name__ == '__main__':
    main()
