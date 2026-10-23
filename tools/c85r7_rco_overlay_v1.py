"""K3 fixed 90/10 whole-roster overlay; no model implementation or admission gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

VERSION = '1.1.45-r6w958rco'
KERNEL = 'taeyangg4/knee-lb-r6w958rco-v1'
CANDIDATE = 'H-B-182-K3-RCO10-V1'
FORMULA = 'rank(0.90*rank(F189_float32_csv)+0.10*rank(RCO_float32)); average ties/full roster; final float32'
UID = 'StudyInstanceUID'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
MANIFEST_SHA = 'a529559ad7efac3e88111b0007338ca931bde3e3a35b8e88b1d09ca3f44e68f3'
WORKER_SHA = 'cfcba1a26dddda54ebfeadd874d58538c1ac358a81e8b45bbabb2563c8b45e1f'
MODEL_SHA = '2c3c033e25605b5fb8bcc08e0030bed081c726e10fc56d8da62524ac08d9638c'
PARENT_VERSION = '1.1.39-r6w958q'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def replace(source, destination):
    for attempt in range(40):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32)) or attempt == 39:
                raise
            time.sleep(min(.05 * (attempt + 1), 1))


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def publish(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    write_once(tmp, payload)
    replace(tmp, path)


def table(payload, expected_ids=None):
    import numpy as np
    rows = list(csv.reader(io.StringIO(payload.decode('utf-8-sig')), strict=True))
    need(rows and rows[0] == [UID] + LABELS, 'target/header order')
    need(all(len(row) == 13 for row in rows[1:]), 'CSV row width')
    ids = [row[0] for row in rows[1:]]
    need(ids and all(x and x.strip() == x for x in ids) and len(set(ids)) == len(ids), 'blank/duplicate ID')
    values = np.asarray([row[1:] for row in rows[1:]], dtype=np.float64)
    need(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'invalid probabilities')
    values = values.astype(np.float32).astype(np.float64)
    if expected_ids is not None:
        need(len(ids) == len(expected_ids) and set(ids) == set(expected_ids), 'roster mismatch')
        lookup = {uid: i for i, uid in enumerate(ids)}
        values = values[[lookup[uid] for uid in expected_ids]]
        ids = list(expected_ids)
    return ids, values


def ranks(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    need(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] == 12 and np.isfinite(values).all(), 'rank input')
    result = np.empty_like(values)
    n = len(values)
    for column in range(12):
        order = np.argsort(values[:, column], kind='stable')
        ordered = values[order, column]
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        ends = np.r_[starts[1:], n]
        for begin, end in zip(starts, ends):
            result[order[begin:end], column] = (float(begin) + float(end) + 1.) / (2. * n)
    return result


def blend(parent_payload, rco_payload, expected_ids):
    import numpy as np
    ids, parent = table(parent_payload, expected_ids)
    _, rco = table(rco_payload, ids)
    values = ranks(.90 * ranks(parent) + .10 * ranks(rco)).astype(np.float32)
    buf = io.StringIO(newline='')
    writer = csv.writer(buf, lineterminator='\n')
    writer.writerow([UID] + LABELS)
    writer.writerows([uid, *[format(float(v), '.9g') for v in row]] for uid, row in zip(ids, values))
    payload = buf.getvalue().encode()
    saved_ids, saved = table(payload, ids)
    need(saved_ids == ids and np.array_equal(saved.astype(np.float32), values), 'float32 roundtrip')
    return payload


def owned_processes(token):
    import psutil
    owned = []
    for process in psutil.process_iter(['pid', 'status']):
        try:
            if process.info['status'] != psutil.STATUS_ZOMBIE and process.environ().get('K3_PROCESS_OWNER') == token:
                owned.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return owned


def run_child(command, *, env=None, cwd=None):
    """Only this invocation's token-bearing descendants may be stopped."""
    import psutil
    token = uuid.uuid4().hex
    child_env = dict(os.environ if env is None else env, K3_PROCESS_OWNER=token)
    process = subprocess.Popen(command, env=child_env, cwd=cwd, start_new_session=os.name != 'nt')
    try:
        code = process.wait()
    finally:
        owned = owned_processes(token)
        for child in owned:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
        need(not owned_processes(token), 'owned child survived; RCO must not start')
    need(code == 0, 'child failed: ' + str(code))


def mount(name, marker, input_root=Path('/kaggle/input')):
    candidates = [input_root / name, input_root / 'competitions' / name,
                  input_root / 'datasets' / name]
    for pattern in ('datasets/*/' + name, 'datasets/*/' + name + '/*', name + '/*'):
        candidates.extend(input_root.glob(pattern))
    candidates = list(dict.fromkeys(p.resolve() for p in candidates if (p / marker).is_file()))
    need(len(candidates) == 1, 'missing/ambiguous mount: ' + name)
    return candidates[0]


def prediction_counts_ok(value):
    """All explicit parent prediction counters must be zero, including R224."""
    need(isinstance(value, dict) and value, 'missing parent fallback counters')
    for key, count in value.items():
        if 'prediction' in key:
            need(type(count) is int and count == 0, 'parent prediction fallback: ' + key)
    need({'a172_prediction', 'r_prediction', 's_r7_prediction'} <= set(value), 'incomplete parent counters')


def check_rco(record, payload, ids, template_sha, manifest):
    need(record.get('status') == 'COMPLETE' and record.get('version') == '1.0.1-rcof0', 'RCO incomplete/version')
    need(record.get('kernel_id') == 'taeyangg4/knee-l1-rcof0-v1', 'RCO identity')
    need(record.get('contract') == manifest['contract'] and record['contract']['model_sha256'] == MODEL_SHA, 'RCO contract')
    need(record.get('manifest_sha256') == MANIFEST_SHA and record.get('template_sha256') == template_sha, 'RCO input identity')
    need(record.get('output_sha256') == hashlib.sha256(payload).hexdigest(), 'RCO CSV hash')
    need(record.get('study_count') == len(ids) == record.get('prediction_count'), 'RCO count')
    need(record.get('fast_path') is False and record.get('placeholder') is False, 'RCO placeholder')
    need((record.get('fallbacks') or {}).get('prediction') == 0, 'RCO prediction fallback')
    need((record.get('codecs') or {}).get('status') == 'ACTIVE', 'RCO codecs')
    studies = record.get('studies')
    need(isinstance(studies, list) and len(studies) == len(ids), 'RCO study evidence')
    for index, (uid, study) in enumerate(zip(ids, studies)):
        need(study.get('index') == index and study.get('uid_sha256') == hashlib.sha256(uid.encode()).hexdigest(), 'RCO study alignment')
        need(study.get('error') is None and study.get('prediction_fallback') == 0, 'RCO study failed')
    table(payload, ids)


def run(runtime, output=Path('/kaggle/working'), *, runner=run_child, assets=None, data=None):
    runtime, output = Path(runtime), Path(output)
    job = output / 'k3-rco-run'
    need(not job.exists() and not (output / 'k3-receipt.json').exists(), 'stale K3 output')
    need(not (output / 'submission.csv').exists(), 'stale parent submission')
    job.mkdir()
    started = time.monotonic()
    record = dict(schema='k3-runtime/v1', version=VERSION, kernel_id=KERNEL, candidate_id=CANDIDATE,
        formula=FORMULA, fast_path=False, placeholder=False, clean_oof=False, exposure='EXPOSED_EXPLORATORY',
        intended_members=['H189', 'RCO_K1_seed0_f0'], executed_members=[], qualified_for_submission=False,
        runtime_source_sha256=sha(__file__), parent_children_released=False)
    parent_payload = None
    try:
        parent_start = time.monotonic()
        runner([sys.executable, '-B', '-u', str(runtime / 'parent-run.py')], cwd=output)
        record['parent_seconds'] = time.monotonic() - parent_start
        record['parent_children_released'] = True
        parent = read(output / 'c85r7-receipt.json')
        payload = (output / 'submission.csv').read_bytes()
        need(parent.get('status') == 'COMPLETE_C85R7' and parent.get('version') == PARENT_VERSION, 'parent incomplete/identity')
        need(parent.get('submission_sha256') == hashlib.sha256(payload).hexdigest(), 'parent CSV digest')
        data = Path(data) if data is not None else mount('rsna-knee-abnormality-detection', 'sample_submission.csv')
        template = data / 'sample_submission.csv'
        ids, _ = table(template.read_bytes())
        table(payload, ids)
        need(parent.get('studies') == len(ids) and parent.get('findings') == 12, 'parent output dimensions')
        # The unchanged successful parent remains available even if RCO setup fails.
        write_once(job / 'parent-submission.csv', payload)
        write_once(job / 'parent-receipt.json', (output / 'c85r7-receipt.json').read_bytes())
        parent_payload = payload
        record.update(executed_members=['H189'], parent_submission_sha256=sha(job / 'parent-submission.csv'),
            parent_receipt_sha256=sha(job / 'parent-receipt.json'), parent_fallbacks=parent.get('fallbacks'),
            parent_codecs=parent.get('codecs'), studies=len(ids), findings=12)
        prediction_counts_ok(parent.get('fallbacks'))
        need((parent.get('codecs') or {}).get('status') == 'ACTIVE' and parent['codecs'].get('s_r7_gate') == 'ACTIVE', 'parent codecs')
        assets = Path(assets) if assets is not None else mount('knee-l1-rcof0-v1-assets', 'asset-manifest.json')
        need(sha(assets / 'asset-manifest.json') == MANIFEST_SHA, 'K2 asset manifest')
        manifest = read(assets / 'asset-manifest.json')
        worker = assets / 'source/l1_rco_k2_worker_v101.py'
        need(sha(worker) == WORKER_SHA, 'K2 worker identity')
        # Verify before installing; worker independently checks every remaining asset.
        wheels = []
        for name, digest in sorted(manifest['wheels'].items()):
            path = assets / 'wheels' / name
            need(path.resolve().is_relative_to(assets.resolve()) and sha(path) == digest, 'wheel identity')
            wheels.append(str(path))
        package_dir = job / 'packages'
        rco_start = time.monotonic()
        runner([sys.executable, '-B', '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', str(package_dir), *wheels], cwd=output)
        codec = read(output / 'c85r7-run/codecs/install.json')
        need(codec.get('status') == 'ACTIVE' and Path(codec['hook_dir']).is_dir(), 'parent codec hook unavailable')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=os.pathsep.join([str(package_dir), codec['hook_dir']]),
            KNEE_CODEC_BRANCH='K3:RCO', KNEE_CODEC_TELEMETRY=str(job / 'codec-telemetry'))
        runner([sys.executable, '-B', '-u', str(worker), '--assets', str(assets), '--manifest-sha', MANIFEST_SHA,
            '--data', str(data), '--template', str(template), '--output', str(job / 'rco.csv'),
            '--scratch', str(job / 'scratch'), '--device', 'cuda:0'], env=env, cwd=output)
        record['rco_seconds'] = time.monotonic() - rco_start
        rco_payload = (job / 'rco.csv').read_bytes()
        rco = read(job / 'worker-receipt.json')
        check_rco(rco, rco_payload, ids, sha(template), manifest)
        record.update(rco_receipt_sha256=sha(job / 'worker-receipt.json'), rco_output_sha256=sha(job / 'rco.csv'),
            asset_manifest_sha256=MANIFEST_SHA, worker_sha256=WORKER_SHA, model_sha256=MODEL_SHA,
            rco_fallbacks=rco['fallbacks'], rco_codecs=rco['codecs'], rco_environment=rco.get('environment'),
            rco_anatomy=rco.get('anatomy'))
        result = blend(parent_payload, rco_payload, ids)
        write_once(job / 'candidate.csv', result)
        record.update(status='COMPLETE_K3', executed_members=['H189', 'RCO_K1_seed0_f0'],
            fallbacks=dict(prediction=0, **parent['fallbacks']), codecs=dict(status='ACTIVE', s_r7_gate='ACTIVE'),
            submission_sha256=sha(job / 'candidate.csv'), total_seconds=time.monotonic() - started)
        # Publish COMPLETE only after the final CSV is durable; a later failure restores parent.
        publish(output / 'submission.csv', result)
        need(sha(output / 'submission.csv') == record['submission_sha256'], 'published CSV digest')
        publish(output / 'k3-receipt.json', encoded(record))
        return record
    except BaseException as exc:
        if parent_payload is not None:
            if not (output / 'submission.csv').is_file() or (output / 'submission.csv').read_bytes() != parent_payload:
                publish(output / 'submission.csv', parent_payload)
            record.update(status='DEGRADED_PARENT_ONLY', executed_members=['H189'],
                submission_sha256=hashlib.sha256(parent_payload).hexdigest())
        else:
            record.update(status='FAILED_PARENT', submission_sha256=None)
        record.update(error_type=type(exc).__name__, error=str(exc)[:500],
            total_seconds=time.monotonic() - started, qualified_for_submission=False)
        publish(output / 'k3-receipt.json', encoded(record))
        if parent_payload is None:
            raise
        print('K3 DEGRADED_PARENT_ONLY: ' + record['error'], flush=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.runtime)
    print(json.dumps({key: result.get(key) for key in ('status', 'studies', 'total_seconds', 'submission_sha256')}), flush=True)


if __name__ == '__main__':
    main()
