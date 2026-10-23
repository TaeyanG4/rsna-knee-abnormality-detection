"""CPU-only packaging of the unchanged qualified B/RN runtime with a 32400s request.

This module never imports the producer, models, training, Kaggle client or credentials.
Build/verify read small frozen metadata/payloads; sdk-check serializes a request only.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STAGE = '.local/b-rn3f-release-staging/20261004-v3'
OLD_MANIFEST = 'a2fa8118649bb438bd810ed8b8ab156798f70f124d7ff1aa2be2692d2a6a3f5f'
RUNTIME_SHA = '4bca01aa59eac625fd8b85707b63e44f845873c0b61aa67425dad446e09356ae'
OLD_NOTEBOOK = '4e3bdefa3ccf39974db27ab1738f3e7c12cd814d3ee84eec6cafb5856c1822c4'
EVIDENCE = 'docs/handoffs/evidence/2026-10/2026-10-04-b-rn3f-release-v1'
SECONDS = 32400


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest()
def sha(path): return digest(Path(path).read_bytes())
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(json_bytes(value))


def safe_name(name):
    p = PurePosixPath(name)
    need(not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\' not in name, 'unsafe payload path')


def pack(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, data in sorted(files.items()):
            safe_name(name)
            info = tarfile.TarInfo(name); info.size = len(data); info.mtime = 0; info.mode = 0o444
            archive.addfile(info, io.BytesIO(data))
    return lzma.compress(stream.getvalue(), preset=6)


def unpack(blob):
    raw = lzma.decompress(blob)
    need(len(raw) <= 16 * 1024 * 1024, 'payload size exceeds inherited limit')
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
        for item in archive:
            safe_name(item.name)
            need(item.isfile() and item.name not in files, 'nonregular/duplicate payload member')
            files[item.name] = archive.extractfile(item).read()
    return files


def source(notebook):
    need(len(notebook['cells']) == 1, 'expected qualified single cell')
    c = notebook['cells'][0]
    need(c['cell_type'] == 'code' and c['execution_count'] is None and c['outputs'] == [], 'unexecuted code cell required')
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']


def embedded_blob(text):
    tree = ast.parse(text)
    need(ast.unparse(tree.body[0]) == 'import time' and ast.unparse(tree.body[1]) == 'started = time.monotonic()', 'first-cell clock changed')
    matches = [n for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'blob' for t in n.targets)]
    need(len(matches) == 1, 'one payload assignment required')
    expr = matches[0].value
    need(isinstance(expr, ast.Call) and ast.unparse(expr.func) == 'base64.b85decode' and len(expr.args) == 1, 'payload expression changed')
    encoded = ast.literal_eval(expr.args[0])
    need(isinstance(encoded, str), 'literal base85 required')
    return encoded, base64.b85decode(encoded)


def request_text(notebook):
    value = copy.deepcopy(notebook)
    for c in value['cells']:
        c['outputs'] = []
        if isinstance(c['source'], list): c['source'] = ''.join(c['source'])
    return json.dumps(value)


def verify(root, stage):
    root, stage = Path(root), Path(stage)
    old = root / SOURCE_STAGE; oldm = read(old / 'bundle/release-manifest.json')
    need(sha(old / 'bundle/release-manifest.json') == OLD_MANIFEST, 'qualified parent manifest changed')
    oldnb = read(old / 'notebook/release.ipynb')
    need(sha(old / 'notebook/release.ipynb') == OLD_NOTEBOOK, 'qualified parent notebook changed')
    b = stage / 'bundle'; m = read(b / 'release-manifest.json')
    need(m['time_policy'] == oldm['time_policy'] and m['policy'] == oldm['policy'], 'runtime/aggregation contract changed')
    allowed = {'created_at', 'files', 'timeout_contract', 'scope', 'configuration_origin'}
    need({k for k in set(m) | set(oldm) if m.get(k) != oldm.get(k)} <= allowed, 'unapproved manifest change')
    need(m['configuration_origin']['qualified_manifest_sha256'] == OLD_MANIFEST, 'qualification lineage differs')
    for name, expected in oldm['files'].items():
        need(m['files'][name] == expected and sha(b / name) == expected == sha(old / 'bundle' / name), 'inherited file changed: ' + name)
    for name, expected in m['files'].items():
        safe_name(name); need(sha(b / name) == expected, 'manifest binding differs: ' + name)
    need(sha(b / 'b_rn3f_release.py') == RUNTIME_SHA == sha(root / 'tools/b_rn3f_release.py'), 'runtime source changed')
    meta = read(stage / 'notebook/kernel-metadata.json')
    need(meta == read(old / 'notebook/kernel-metadata.json'), 'qualified runtime metadata changed')
    nb = read(stage / 'notebook/release.ipynb'); text = source(nb); oldtext = source(oldnb)
    encoded, blob = embedded_blob(text); oldencoded, oldblob = embedded_blob(oldtext)
    replacements = [(encoded, oldencoded), (digest(blob), digest(oldblob)), (sha(b / 'release-manifest.json'), OLD_MANIFEST)]
    restored = text
    for a, z in replacements:
        need(restored.count(repr(a)) == 1, 'binding literal not unique')
        restored = restored.replace(repr(a), repr(z), 1)
    need(restored == oldtext and ast.dump(ast.parse(restored)) == ast.dump(ast.parse(oldtext)), 'notebook changed beyond three binding constants')
    files = unpack(blob)
    need(set(files) == set(m['files']) | {'release-manifest.json'}, 'payload inventory differs')
    need(all(data == (b / name).read_bytes() for name, data in files.items()), 'embedded bytes differ')
    timeout = m['timeout_contract']
    need(timeout['platform_request_seconds_actual'] == SECONDS and timeout['public_internal_seconds'] == 600
         and timeout['full_internal_seconds'] == 32100 and timeout['hard_public_900_seconds'] is False
         and timeout['unexpected_hang_worst_two_T4_GPU_hours'] == 18, 'active timeout contract differs')
    return {'status': 'CPU_CONFIGURATION_BINDINGS_PASS_NOT_EXECUTED', 'runtime_source_sha256': RUNTIME_SHA,
            'manifest_sha256': sha(b / 'release-manifest.json'), 'notebook_sha256': sha(stage / 'notebook/release.ipynb'),
            'metadata_sha256': sha(stage / 'notebook/kernel-metadata.json'), 'payload_lzma_sha256': digest(blob),
            'payload_lzma_bytes': len(blob), 'payload_member_count': len(files), 'unchanged_parent_files': len(oldm['files']),
            'notebook_only_three_binding_constants_changed': True, 'original_runtime_time_policy_unchanged': True,
            'platform_request_seconds_actual': SECONDS, 'public_internal_seconds': 600, 'full_internal_seconds': 32100,
            'push_argv': [str(root / '.venv/Scripts/kaggle.exe'), 'kernels', 'push', '-p', str(stage / 'notebook'), '--timeout', str(SECONDS)],
            'expected_public_GPU_hours': [0.1, 0.2], 'worst_two_T4_GPU_hours': 18, 'automatic_retries': 0,
            'GPU_execution': 0, 'network_calls': 0, 'weight_upload_bytes': 0, 'submission_authorized': False}


def build(root, stage):
    root, stage = Path(root).resolve(), Path(stage).resolve()
    need(not stage.exists(), 'new immutable configuration stage required')
    old = root / SOURCE_STAGE; oldm = read(old / 'bundle/release-manifest.json')
    need(sha(old / 'bundle/release-manifest.json') == OLD_MANIFEST, 'qualified manifest changed')
    decision = root / EVIDENCE / 'root-final-config-decision.json'; d = read(decision)
    need(d['status'] == 'AUTHORIZE_CONFIG_ONLY_CPU_STAGE_AND_ONE_PRIVATE_RUN_AFTER_ROOT_REVIEW'
         and d['root_accepts_free_quota_exposure'] is True and d['unexpected_hang_worst_two_T4_GPU_hours'] == 18
         and d['automatic_retries'] == 0 and d['competition_submission'] is False, 'root configuration decision absent')
    files = {}
    for name, expected in oldm['files'].items():
        data = (old / 'bundle' / name).read_bytes(); need(digest(data) == expected, 'qualified payload changed'); files[name] = data
    files.update({'b_rn3f_full_config.py': Path(__file__).read_bytes(), 'root-final-config-decision.json': decision.read_bytes(),
                  'public3-independent-review.json': (root / EVIDENCE / 'independent-result-review.json').read_bytes()})
    m = copy.deepcopy(oldm); m['created_at'] = datetime.now(timezone.utc).isoformat()
    m['files'] = {name: digest(data) for name, data in sorted(files.items())}
    m['scope'] = 'Explicit32400 platform configuration; same qualified runtime/calculation. New-version public3 and root review pending; no submission authorization or hidden-runtime claim.'
    m['timeout_contract'] = {'platform_request_seconds_actual': SECONDS, 'public_internal_seconds': 600,
        'full_internal_seconds': 32100, 'hidden_scoring_timeout': 'UNKNOWN_SERVER_BEHAVIOR',
        'full_readiness': 'EXPLICIT_PLATFORM_CONFIG_PREPARED_NEW_VERSION_PUBLIC3_PENDING',
        'hard_public_900_seconds': False, 'expected_public_GPU_hours': [0.1, 0.2],
        'unexpected_hang_worst_two_T4_GPU_hours': 18, 'automatic_retries': 0,
        'scope': 'SDK/push request32400; observed100s is an estimate, internal600 is not whole-notebook hard kill'}
    m['configuration_origin'] = {'qualified_manifest_sha256': OLD_MANIFEST, 'qualified_kernel': 'taeyangg4/knee-b-rn3f-release-v1',
        'qualified_version': 1, 'runtime_source_sha256': RUNTIME_SHA,
        'legacy_fields': 'time_policy.public3_session_seconds=900 and root-timeout-decision.json are immutable inherited V3 qualification history. This field is not read by run(); current platform request is timeout_contract.platform_request_seconds_actual=32400. Original build/sdk-check are not used.',
        'decision_sha256': digest(files['root-final-config-decision.json'])}
    files['release-manifest.json'] = json_bytes(m); manifest_sha = digest(files['release-manifest.json'])
    blob = pack(files); nb = read(old / 'notebook/release.ipynb'); text = source(nb); oldencoded, oldblob = embedded_blob(text)
    for a, z in [(oldencoded, base64.b85encode(blob).decode()), (digest(oldblob), digest(blob)), (OLD_MANIFEST, manifest_sha)]:
        need(text.count(repr(a)) == 1, 'qualified notebook binding not unique'); text = text.replace(repr(a), repr(z), 1)
    nb['cells'][0]['source'] = text.splitlines(keepends=True)
    for name, data in files.items():
        target = stage / 'bundle' / name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(data)
    (stage / 'notebook').mkdir()
    write_new(stage / 'notebook/release.ipynb', nb)
    with (stage / 'notebook/kernel-metadata.json').open('xb') as stream: stream.write((old / 'notebook/kernel-metadata.json').read_bytes())
    result = verify(root, stage); write_new(stage / 'stage-review.json', result); return result


def sdk_check(root, stage):
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    stage = Path(stage); verified = verify(root, stage); meta = read(stage / 'notebook/kernel-metadata.json')
    text = request_text(read(stage / 'notebook' / meta['code_file'])); req = ApiSaveKernelRequest()
    for field, key in [('slug', 'id'), ('new_title', 'title'), ('language', 'language'), ('kernel_type', 'kernel_type'),
        ('is_private', 'is_private'), ('enable_gpu', 'enable_gpu'), ('enable_tpu', 'enable_tpu'), ('enable_internet', 'enable_internet'),
        ('dataset_data_sources', 'dataset_sources'), ('competition_data_sources', 'competition_sources'),
        ('kernel_data_sources', 'kernel_sources'), ('model_data_sources', 'model_sources'), ('docker_image', 'docker_image'), ('machine_shape', 'machine_shape')]:
        setattr(req, field, meta[key])
    req.text = text; req.category_ids = meta.get('keywords', []); req.session_timeout_seconds = SECONDS
    encoded = req.to_json().encode(); serialized = json.loads(encoded)
    need(serialized.get('sessionTimeoutSeconds') == SECONDS, 'serialized SDK timeout differs')
    need(max(len(encoded), len(text.encode())) < 1_000_000, 'SDK payload exceeds limit')
    result = {'status': 'CPU_ACTUAL_SDK_32400_SERIALIZATION_PASS', 'manifest_sha256': verified['manifest_sha256'],
              'session_timeout_seconds': SECONDS, 'request_sha256': digest(encoded), 'request_bytes': len(encoded),
              'script_body_bytes': len(text.encode()), 'credential_reads': 0, 'network_calls': 0}
    write_new(stage / 'sdk-review.json', result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['build', 'verify', 'sdk-check'])
    p.add_argument('--root', default=str(ROOT)); p.add_argument('--stage', required=True); args = p.parse_args()
    operation = {'build': build, 'verify': verify, 'sdk-check': sdk_check}[args.action]
    print(json.dumps(operation(args.root, args.stage), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
