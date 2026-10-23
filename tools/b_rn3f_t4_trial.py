"""Prepare or execute one bounded private exact100 T4 engineering trial.

Build is CPU-only and refuses an unlinked local reference. Run is explicit remote
work, supervised externally; this module never publishes, pushes or submits.
"""
from __future__ import annotations
import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import io
import json
import lzma
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'B_RN3F_T4_TRIAL_V1'
DATASET = 'taeyangg4/knee-b-rn3f-v1-assets'
KERNEL = 'taeyangg4/knee-b-rn3f-v1-engineering100'
SERVING_SHA = '40f5a04dd0f770f05e9af5cf9a3827ccf5748c7989847c0778ba189fde40f80e'
PROTOCOL_SHA = '8e57613e51f082a2c932aa70c9c94590ac9ff1207bb82b213a7e73fb2f0ce1d7'
SOURCE_SHA = {
    'b_rn3f_parity.py': '298f11855c7e8e86007be612bf7e0831469da54d601aeedabab9f85a6259754b',
    'b_rn3f_acceptance.py': '3727ef865c820d478b96db776d45973cca7e755a2c728f4f0492840d3d210aee',
    'process_resource_receipt.py': '15c0fe61d5d0b350557093ec38a1dc902b73e12c76328c4951934cc88f1da954',
}
OPENCV_SHA = '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def load(path, name, expected=None):
    if expected: need(sha(path) == expected, 'helper identity changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def validate_resource(value, parent_pid=None, require_linux=False):
    need(value.get('schema') == 'process-resource-receipt-v1'
         and value.get('supervisor_sha256') == SOURCE_SHA['process_resource_receipt.py'], 'resource helper identity mismatch')
    need(value.get('status') == 'COMPLETE' and value.get('wrapper_exit_code') == value.get('child_exit_code') == 0
         and value.get('failure') is None and value.get('child_exit_code_before_cleanup') == 0,
         'resource execution not complete')
    need(value.get('sample_count', 0) > 0 and value.get('sampled_tree_rss_max_bytes', 0) > 0
         and value.get('failed_sample_attempts') == value.get('access_failure_count') == 0
         and not value.get('access_failures'), 'resource telemetry incomplete')
    need(not value.get('cleanup', {}).get('errors') and not value.get('cleanup', {}).get('survivors')
         and not value.get('cleanup', {}).get('actions'), 'process cleanup not naturally complete')
    need(value.get('retries') == 0 and 0 < value.get('timeout_seconds', 0) <= 1800
         and math.isfinite(value.get('elapsed_seconds', float('nan'))), 'resource budget mismatch')
    if parent_pid is not None:
        need(parent_pid in {r['pid'] for r in value.get('observed_identities', [])}, 'worker PID missing from owned tree')
    if require_linux:
        need(value.get('platform', '').startswith('linux') and value.get('cgroup_telemetry_complete') is True,
             'actual Linux cgroup telemetry incomplete')
    return value


def validate_reference(assets, bundle, tools_dir):
    """Never infer valid local linkage from producer exit/status alone."""
    assets, bundle, tools_dir = map(Path, (assets, bundle, tools_dir))
    need(sha(assets / 'b-rn3f-assets.json') == SERVING_SHA, 'serving assets changed')
    producer_sha = read(assets / 'b-rn3f-assets.json')['files']['b_rn3f_serving.py']
    producer = load(assets / 'b_rn3f_serving.py', 'b_trial_producer', producer_sha)
    acceptance = load(tools_dir / 'b_rn3f_acceptance.py', 'b_trial_acceptance', SOURCE_SHA['b_rn3f_acceptance.py'])
    parity = acceptance.load_helper(tools_dir / 'b_rn3f_parity.py')
    parity.validate_protocol(bundle / 'protocol.json', PROTOCOL_SHA)
    work = bundle / 'local-work'
    acceptance.parent_completion(work)
    ids = read(assets / 'engineering100.json')['ids']
    receipt = producer.validate_result(work, assets, ids, 'local-reference', 'engineering100')
    parity.validate_input_report(read(bundle / 'local-input.json'), receipt, ids, PROTOCOL_SHA, 'local-reference')
    validate_thread_binding(bundle, work, bundle / 'local-input.json', bundle / 'local-input-acceptance.json', 'local-reference')
    resource = validate_resource(read(bundle / 'local-resource.json'), read(work / 'run-receipt.json')['pid'])
    need('--target' in resource['argv'] and 'local-reference' in resource['argv'] and SERVING_SHA in resource['argv'],
         'resource argv does not bind fixed local producer')
    return {'status': 'LOCAL_COMPLETION_INPUT_LINKAGE_VALID', 'parent_sha256': sha(work / 'run-receipt.json'),
            'worker_sha256': sha(work / 'result/receipt.json'), 'local_input_sha256': sha(bundle / 'local-input.json'),
            'resource_sha256': sha(bundle / 'local-resource.json')}


def validate_thread_binding(bundle, work, report_path, acceptance_path, target):
    from types import SimpleNamespace
    bundle, work = Path(bundle), Path(work)
    thread_protocol = read(bundle / 'thread-protocol.json')
    bridge_sha = thread_protocol['adapter_sha256']
    bridge = load(bundle / 'b_rn3f_thread_input_v2.py', 'trial_thread_bridge', bridge_sha)
    helper = bridge.load_math(bundle / 'b_rn3f_parity.py')
    protocol_sha = sha(bundle / 'thread-protocol.json')
    args = SimpleNamespace(protocol=bundle / 'protocol.json', protocol_sha256=PROTOCOL_SHA,
                           thread_protocol=bundle / 'thread-protocol.json', thread_protocol_sha256=protocol_sha)
    bridge.validate_contract(args, helper)
    ids = [v['uid'] for v in read(report_path)['rows']]
    worker, binding = bridge.completed_input_receipt(work, helper, ids, target)
    receipt, report = read(acceptance_path), read(report_path)
    need(receipt.get('schema') == bridge.SCHEMA and receipt.get('status') == bridge.PASS
         and receipt.get('adapter_sha256') == bridge_sha and receipt.get('input_report_sha256') == sha(report_path)
         and receipt.get('thread_protocol_sha256') == protocol_sha and receipt.get('math_protocol_sha256') == PROTOCOL_SHA
         and receipt.get('math_helper_sha256') == SOURCE_SHA['b_rn3f_parity.py']
         and receipt.get('exact_bound_studies') == 100 and receipt.get('producer_binding') == binding,
         'thread bridge acceptance incomplete or unbound')
    adapter = report['thread_adapter']
    need(adapter.get('adapter_sha256') == bridge_sha and adapter.get('thread_protocol_sha256') == protocol_sha
         and adapter.get('producer_binding') == binding and adapter.get('actual_threads') == receipt['actual_threads'] == binding['selected_threads']
         and adapter.get('actual_interop_threads') == 1 and adapter.get('inference_mode') is True,
         'actual producer thread replay mismatch')
    helper.validate_input_report(report, worker, ids, PROTOCOL_SHA, target)
    return {'bridge_sha256': bridge_sha, 'thread_protocol_sha256': protocol_sha,
            'input_acceptance_sha256': sha(acceptance_path), 'actual_threads': binding['selected_threads']}


def pack_files(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, data in sorted(files.items()):
            path = PurePosixPath(name)
            need(not path.is_absolute() and '..' not in path.parts and ':' not in name and '\\' not in name, 'unsafe payload path')
            info = tarfile.TarInfo(name); info.size = len(data); info.mtime = 0; info.mode = 0o444
            archive.addfile(info, io.BytesIO(data))
    return lzma.compress(stream.getvalue(), preset=6)


def unpack_files(blob, destination):
    # This function is embedded verbatim in the notebook before helpers exist.
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    raw = lzma.decompress(blob)
    if len(raw) > 16 * 1024 * 1024: raise ValueError('payload exceeds16MiB cap')
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
        for item in archive:
            path = PurePosixPath(item.name)
            if not item.isfile() or path.is_absolute() or '..' in path.parts or ':' in item.name or '\\' in item.name or item.name in seen:
                raise ValueError('unsafe payload member')
            seen.add(item.name); target = destination / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write(archive.extractfile(item).read())


def sdk_text(notebook):
    value = copy.deepcopy(notebook)
    for cell in value.get('cells', []):
        if cell.get('cell_type') == 'code' and 'outputs' in cell: cell['outputs'] = []
        if isinstance(cell.get('source'), list): cell['source'] = ''.join(cell['source'])
    return json.dumps(value)


def sdk_check(stage):
    """Actual installed SDK request serialization; no Kaggle client/auth/network."""
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    stage = Path(stage); directory = stage / 'notebook'
    meta = read(directory / 'kernel-metadata.json'); text = sdk_text(read(directory / meta['code_file']))
    request = ApiSaveKernelRequest()
    request.slug = meta['id']; request.new_title = meta['title']; request.text = text
    request.language = meta['language']; request.kernel_type = meta['kernel_type']; request.is_private = meta['is_private']
    request.enable_gpu = meta['enable_gpu']; request.enable_tpu = meta.get('enable_tpu', False); request.enable_internet = meta['enable_internet']
    request.dataset_data_sources = meta['dataset_sources']; request.competition_data_sources = meta['competition_sources']
    request.kernel_data_sources = meta['kernel_sources']; request.model_data_sources = meta['model_sources']
    request.category_ids = meta.get('keywords', []); request.docker_image = meta.get('docker_image')
    request.machine_shape = meta.get('machine_shape'); request.session_timeout_seconds = 1800
    encoded = request.to_json().encode('utf-8')
    result = {'status': 'CPU_SDK_PAYLOAD_SIZE_PASS', 'actual_sdk_request_bytes': len(encoded),
              'script_body_bytes': len(text.encode('utf-8')), 'request_sha256': hashlib.sha256(encoded).hexdigest(),
              'notebook_sha256': sha(directory / meta['code_file']), 'metadata_sha256': sha(directory / 'kernel-metadata.json'),
              'credential_reads': 0, 'network_calls': 0, 'conservative_limit_bytes': 1_000_000}
    need(max(result['actual_sdk_request_bytes'], result['script_body_bytes']) < 1_000_000, 'actual SDK payload exceeds1MB')
    write_new(stage / 'sdk-payload-review.json', result); return result


def build(args):
    root, assets, out = Path(args.root), Path(args.assets), Path(args.out)
    need(not out.exists(), 'new immutable trial stage required')
    # Small reference copy, no model/package work until local input linkage passes.
    out.mkdir(parents=True)
    bundle = out / 'payload'; bundle.mkdir()
    reference_files = {
        'protocol.json': Path(args.protocol), 'local-input.json': Path(args.local_input),
        'thread-protocol.json': Path(args.thread_protocol), 'local-input-acceptance.json': Path(args.local_input_acceptance),
        'local-resource.json': Path(args.local_resource) / 'receipt.json',
        **{'local-work/' + n: Path(args.local_work) / n for n in
           ['run-receipt.json', 'result/receipt.json', 'result/members.npz', 'result/predictions.csv']}}
    for name, source in reference_files.items():
        target = bundle / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    for name, digest in SOURCE_SHA.items():
        need(sha(root / 'tools' / name) == digest, 'reviewed helper changed')
        shutil.copyfile(root / 'tools' / name, bundle / name)
    bridge_source = root / 'tools/b_rn3f_thread_input_v2.py'
    need(sha(bridge_source) == read(bundle / 'thread-protocol.json')['adapter_sha256'], 'bridge source differs from new frozen contract')
    shutil.copyfile(bridge_source, bundle / bridge_source.name)
    shutil.copyfile(__file__, bundle / 'b_rn3f_t4_trial.py')
    try:
        link = validate_reference(assets, bundle, bundle)
    except Exception as exc:
        write_new(out / 'build-failure.json', {'status': 'HOLD_LOCAL_REFERENCE_LINKAGE', 'error': str(exc),
                  'exception_type': type(exc).__name__, 'upload_ready': False, 'remote_operations': 0})
        raise
    files = {p.relative_to(bundle).as_posix(): sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    manifest = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(), 'files': files,
                'serving_sha256': SERVING_SHA, 'protocol_sha256': PROTOCOL_SHA, 'local_linkage': link,
                'thread_protocol_sha256': sha(bundle / 'thread-protocol.json'), 'thread_bridge_sha256': sha(bundle / bridge_source.name),
                'session_cap_seconds': 1800, 'owned_workflow_cap_seconds': 1700, 'retries': 0,
                'producer': 'Existing B8/RNclip20 exact6; no EXP084 replacement; full cohort rank90/10',
                'submission_ready': False, 'hidden_release': False}
    write_new(bundle / 'trial-manifest.json', manifest)
    upload = out / 'upload'; upload.mkdir()
    linked_bytes = 0
    inventory = [*read(assets / 'b-rn3f-assets.json')['files'], 'b-rn3f-assets.json']
    for relative in sorted(inventory):
        path = assets / relative
        target = upload / relative; target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target); linked_bytes += path.stat().st_size
    write_new(upload / 'dataset-metadata.json', {'title': DATASET.split('/')[1], 'id': DATASET,
              'licenses': [{'name': 'other'}], 'isPrivate': True,
              'description': 'Private frozen exact B8/RNclip20 serving assets; engineering only, no performance or submission claim.'})
    raw = {p.relative_to(bundle).as_posix(): p.read_bytes() for p in bundle.rglob('*') if p.is_file()}
    blob = pack_files(raw); blob_sha = hashlib.sha256(blob).hexdigest()
    manifest_sha = sha(bundle / 'trial-manifest.json')
    bootstrap = '''from pathlib import Path, PurePosixPath
import base64, hashlib, importlib.util, io, json, lzma, os, subprocess, sys, tarfile, time
'''+inspect.getsource(unpack_files)+f'''
started=time.monotonic()
blob=base64.b85decode({base64.b85encode(blob).decode()!r})
assert hashlib.sha256(blob).hexdigest()=={blob_sha!r}
base=Path('/kaggle/working/b-rn3f-v1')
unpack_files(blob,base/'bundle')
helper=base/'bundle/b_rn3f_t4_trial.py'
spec=importlib.util.spec_from_file_location('trial',helper)
trial=importlib.util.module_from_spec(spec);spec.loader.exec_module(trial)
assets=trial.find_mount('b-rn3f-assets.json',{SERVING_SHA!r})
opencv=trial.find_mount('opencv_python_headless-4.12.0.88-*.whl',{OPENCV_SHA!r},return_file=True,dataset='mattiaangeli/opencv-python-headless-4120088-x86')
competition=trial.find_competition()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1')
remaining=max(1,1700-int(time.monotonic()-started))
command=[sys.executable,str(base/'bundle/process_resource_receipt.py'),'--out',str(base/'resource'),'--timeout',str(remaining),'--',sys.executable,str(helper),'workflow','--bundle',str(base/'bundle'),'--manifest-sha256',{manifest_sha!r},'--assets',str(assets),'--competition',str(competition),'--opencv-wheel',str(opencv),'--work',str(base/'work')]
code=subprocess.run(command,env=env).returncode
result=trial.finalize(base,assets,{manifest_sha!r},code,time.monotonic()-started)
print(json.dumps(result,indent=2))
assert result['status']=='PASS_ENGINEERING_NUMERICAL_AND_RESOURCE_ONLY'
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 4, 'metadata': {'kernelspec': {'display_name': 'Python3', 'language': 'python', 'name': 'python3'}},
                'cells': [{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': bootstrap.splitlines(keepends=True)}]}
    size = len(sdk_text(notebook).encode('utf-8')); need(size < 1_000_000, 'SDK notebook payload exceeds conservative1MB')
    notebook_dir = out / 'notebook'; notebook_dir.mkdir()
    write_new(notebook_dir / 'trial.ipynb', notebook)
    metadata = read(root / '.local/d01-runtime-staging/20261003-v4/kernel-metadata.json')
    metadata.update(id=KERNEL, title=KERNEL.split('/')[1], code_file='trial.ipynb', is_private=True,
                    enable_gpu=True, enable_internet=False, dataset_sources=[DATASET, 'mattiaangeli/opencv-python-headless-4120088-x86'],
                    kernel_sources=[], model_sources=[], competition_sources=['rsna-knee-abnormality-detection'])
    write_new(notebook_dir / 'kernel-metadata.json', metadata)
    result = {'schema': SCHEMA, 'status': 'CPU_STAGE_READY_REQUIRES_ROOT_DISPATCH', 'trial_manifest_sha256': manifest_sha,
              'dataset': DATASET, 'kernel': KERNEL, 'visibility': 'private', 'local_linkage': link,
              'linked_asset_bytes': linked_bytes, 'physical_weight_copies': 0, 'payload_raw_bytes': sum(map(len, raw.values())),
              'payload_lzma_bytes': len(blob), 'payload_lzma_sha256': blob_sha, 'sdk_script_body_bytes': size,
              'notebook_sha256': sha(notebook_dir / 'trial.ipynb'), 'remote_operations': 0, 'gpu_execution': False,
              'upload_argv': [str(root / '.venv/Scripts/kaggle.exe'), 'datasets', 'create', '-p', str(upload), '--dir-mode', 'zip'],
              'push_argv': [str(root / '.venv/Scripts/kaggle.exe'), 'kernels', 'push', '-p', str(notebook_dir), '--timeout', '1800'],
              'retry_count': 0, 'submission_ready': False}
    write_new(out / 'stage-review.json', result); return result


def find_mount(filename, expected, return_file=False, dataset=DATASET):
    owner, slug = dataset.split('/')
    roots = [Path('/kaggle/input') / slug, Path('/kaggle/input/datasets') / owner / slug]
    candidates = sorted({p.resolve() for root in roots if root.is_dir() for p in root.glob(filename)
                         if p.is_file() and sha(p) == expected})
    need(len(candidates) == 1, 'exact mounted file missing/ambiguous: ' + filename)
    return candidates[0] if return_file else candidates[0].parent


def find_competition():
    paths = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')]
    found = [p for p in paths if (p / 'train_series.csv').is_file() and (p / 'train_series').is_dir()]
    need(len(found) == 1, 'actual competition path absent/ambiguous'); return found[0]


def check_bundle(bundle, digest):
    bundle = Path(bundle); need(sha(bundle / 'trial-manifest.json') == digest, 'trial manifest changed')
    manifest = read(bundle / 'trial-manifest.json'); need(manifest['schema'] == SCHEMA, 'wrong trial schema')
    for relative, expected in manifest['files'].items():
        path = PurePosixPath(relative)
        need(not path.is_absolute() and '..' not in path.parts and ':' not in relative and '\\' not in relative, 'unsafe manifest path')
        need(sha(bundle / relative) == expected, 'trial payload changed: ' + relative)
    return manifest


def workflow(args):
    bundle, assets, work = Path(args.bundle), Path(args.assets), Path(args.work)
    check_bundle(bundle, args.manifest_sha256)
    validate_reference(assets, bundle, bundle)
    need(not work.exists(), 'fresh workflow directory required'); work.mkdir()
    commands = []
    def call(command, env=None):
        commands.append(command); subprocess.run(command, check=True, env=env)
    fixed = ['--parity-helper', str(bundle / 'b_rn3f_parity.py'), '--protocol', str(bundle / 'protocol.json'),
             '--protocol-sha256', PROTOCOL_SHA, '--assets', str(assets)]
    call([sys.executable, str(assets / 'b_rn3f_serving.py'), 'run', '--assets', str(assets), '--manifest-sha256', SERVING_SHA,
          '--work', str(work / 'producer'), '--target', 't4-engineering', '--cohort', 'engineering100',
          '--competition', args.competition, '--opencv-wheel', args.opencv_wheel, '--cpu-threads', '12', '--max-seconds', '1600'])
    acceptance = load(bundle / 'b_rn3f_acceptance.py', 'workflow_acceptance', SOURCE_SHA['b_rn3f_acceptance.py'])
    acceptance.parent_completion(work / 'producer')
    environment = work / 'producer/environment'; need(environment.is_dir(), 'producer offline dependencies absent')
    env = dict(os.environ, PYTHONPATH=str(environment), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    call([sys.executable, str(bundle / 'b_rn3f_thread_input_v2.py'), 'input', *fixed,
          '--thread-protocol', str(bundle / 'thread-protocol.json'), '--thread-protocol-sha256', sha(bundle / 'thread-protocol.json'),
          '--work', str(work / 'producer'), '--target', 't4-engineering', '--competition', args.competition,
          '--out', str(work / 'input.json'), '--acceptance-receipt', str(work / 'input-acceptance.json')], env)
    validate_thread_binding(bundle, work / 'producer', work / 'input.json', work / 'input-acceptance.json', 't4-engineering')
    call([sys.executable, str(bundle / 'b_rn3f_acceptance.py'), 'evaluate', *fixed,
          '--local-work', str(bundle / 'local-work'), '--t4-work', str(work / 'producer'),
          '--local-input', str(bundle / 'local-input.json'), '--t4-input', str(work / 'input.json'),
          '--out', str(work / 'paired.json'), '--acceptance-receipt', str(work / 'paired-acceptance.json')], env)
    need(read(work / 'paired-acceptance.json')['status'] == 'PASS_COMPLETION_AND_NUMERICAL_ONLY'
         and read(work / 'paired.json')['status'] == 'PASS_NUMERICAL_ONLY', 'paired numerical comparison did not pass')
    write_new(work / 'workflow.json', {'schema': SCHEMA, 'status': 'WORKFLOW_NUMERICAL_COMPLETE_RESOURCE_PENDING',
              'commands': commands, 'trial_manifest_sha256': args.manifest_sha256,
              'paired_sha256': sha(work / 'paired.json'), 'paired_acceptance_sha256': sha(work / 'paired-acceptance.json'),
              'submission_ready': False})


def finalize(base, assets, manifest_sha, exit_code, elapsed):
    base, assets = Path(base), Path(assets)
    result = {'schema': SCHEMA, 'status': 'FAILED', 'trial_manifest_sha256': manifest_sha,
              'elapsed_seconds_notebook_code': elapsed, 'submission_ready': False, 'hidden_runtime_claim': False,
              'competitive_quality_claim': False, 'observed_at': datetime.now(timezone.utc).isoformat()}
    try:
        check_bundle(base / 'bundle', manifest_sha)
        need(exit_code == 0 and elapsed < 1800, 'supervisor exit/budget failed')
        parent = read(base / 'work/producer/run-receipt.json')
        thread = validate_thread_binding(base / 'bundle', base / 'work/producer', base / 'work/input.json',
                                         base / 'work/input-acceptance.json', 't4-engineering')
        resource = validate_resource(read(base / 'resource/receipt.json'), parent['pid'], require_linux=True)
        done = read(base / 'work/workflow.json')
        need(done['status'] == 'WORKFLOW_NUMERICAL_COMPLETE_RESOURCE_PENDING'
             and done['trial_manifest_sha256'] == manifest_sha, 'workflow completion mismatch')
        need(sha(base / 'work/paired.json') == done['paired_sha256'] and sha(base / 'work/paired-acceptance.json') == done['paired_acceptance_sha256'], 'paired artifacts changed')
        acceptance = read(base / 'work/paired-acceptance.json'); paired = read(base / 'work/paired.json')
        need(acceptance['status'] == 'PASS_COMPLETION_AND_NUMERICAL_ONLY' and paired['status'] == 'PASS_NUMERICAL_ONLY'
             and acceptance['numerical_output_sha256'] == sha(base / 'work/paired.json'), 'actual final gate failed')
        need(len(acceptance['parent_receipts']) == 2, 'missing acceptance parent evidence')
        for work, receipt in zip([base / 'bundle/local-work', base / 'work/producer'], acceptance['parent_receipts']):
            need(sha(work / 'run-receipt.json') == receipt['parent_sha256']
                 and sha(work / 'result/receipt.json') == receipt['worker_sha256'], 'actual parent/worker receipt changed')
        result.update(status='PASS_ENGINEERING_NUMERICAL_AND_RESOURCE_ONLY',
                      thread_bridge=thread,
                      resource_sha256=sha(base / 'resource/receipt.json'), paired_sha256=sha(base / 'work/paired.json'),
                      acceptance_sha256=sha(base / 'work/paired-acceptance.json'),
                      sampled_tree_rss_max_bytes=resource['sampled_tree_rss_max_bytes'])
    except Exception as exc:
        result.update(error=str(exc), exception_type=type(exc).__name__)
    archive = base / 'evidence.zip'
    names = ['resource/receipt.json', 'resource/stdout.log', 'resource/stderr.log', 'resource/samples.jsonl',
             'work/workflow.json', 'work/input.json', 'work/input-acceptance.json', 'work/paired.json', 'work/paired-acceptance.json',
             'work/producer/run-receipt.json', 'work/producer/worker-failure.json', 'work/producer/launch.json', 'work/producer/run.log',
             'work/producer/result/receipt.json', 'work/producer/result/members.npz', 'work/producer/result/predictions.csv']
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            if (base / name).is_file(): z.write(base / name, name)
    result.update(evidence_zip_sha256=sha(archive), evidence_zip_bytes=archive.stat().st_size,
                  artifact_sha256={name: sha(base / name) for name in names if (base / name).is_file()})
    write_new(base / 'result.json', result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build'); b.add_argument('--root', default=str(ROOT))
    for name in ['out', 'assets', 'local-work', 'local-input', 'local-resource', 'protocol', 'thread-protocol', 'local-input-acceptance']:
        b.add_argument('--' + name, required=True)
    w = sub.add_parser('workflow')
    for name in ['bundle', 'manifest-sha256', 'assets', 'competition', 'opencv-wheel', 'work']: w.add_argument('--' + name, required=True)
    s = sub.add_parser('sdk-check'); s.add_argument('--stage', required=True)
    args = parser.parse_args()
    result = build(args) if args.action == 'build' else sdk_check(args.stage) if args.action == 'sdk-check' else workflow(args)
    if result: print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
