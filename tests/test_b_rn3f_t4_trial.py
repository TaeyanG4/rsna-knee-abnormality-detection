import copy
import ast
import importlib.util
import io
import json
import lzma
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trial', ROOT / 'tools/b_rn3f_t4_trial.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)


def resource():
    return {'schema': 'process-resource-receipt-v1', 'supervisor_sha256': t.SOURCE_SHA['process_resource_receipt.py'],
            'status': 'COMPLETE', 'wrapper_exit_code': 0, 'child_exit_code': 0, 'child_exit_code_before_cleanup': 0,
            'failure': None, 'sample_count': 100, 'sampled_tree_rss_max_bytes': 2000,
            'failed_sample_attempts': 0, 'access_failure_count': 0, 'access_failures': [],
            'cleanup': {'errors': [], 'survivors': [], 'actions': []}, 'retries': 0,
            'timeout_seconds': 1700, 'elapsed_seconds': 30., 'observed_identities': [{'pid': 99}],
            'platform': 'linux', 'cgroup_telemetry_complete': True}


@pytest.mark.parametrize('bad', [None, 'status', 'child', 'worker', 'access', 'samples', 'cleanup', 'survivor', 'kill', 'cgroup', 'helper', 'retry'])
def test_resource_requires_actual_completion_and_telemetry(bad):
    value = resource()
    if bad == 'status': value['status'] = 'FAILED'
    if bad == 'child': value['child_exit_code'] = 1
    if bad == 'worker': value['observed_identities'] = []
    if bad == 'access': value['access_failure_count'] = 1
    if bad == 'samples': value['sample_count'] = 0
    if bad == 'cleanup': value['cleanup']['errors'] = ['AccessDenied']
    if bad == 'survivor': value['cleanup']['survivors'] = [99]
    if bad == 'kill': value['cleanup']['actions'] = [{'pid': 99, 'action': 'kill'}]
    if bad == 'cgroup': value['cgroup_telemetry_complete'] = False
    if bad == 'helper': value['supervisor_sha256'] = '0' * 64
    if bad == 'retry': value['retries'] = 1
    if bad:
        with pytest.raises(ValueError): t.validate_resource(value, 99, require_linux=True)
    else: t.validate_resource(value, 99, require_linux=True)


def test_deterministic_compressed_payload_exact_roundtrip(tmp_path):
    files = {'test/a.txt': b'abc', 'ref/array.bin': bytes(range(255))}
    blob = t.pack_files(files)
    assert blob == t.pack_files(dict(reversed(list(files.items()))))
    t.unpack_files(blob, tmp_path / 'decoded')
    assert {p.relative_to(tmp_path / 'decoded').as_posix(): p.read_bytes()
            for p in (tmp_path / 'decoded').rglob('*') if p.is_file()} == files
    with pytest.raises(FileExistsError): t.unpack_files(blob, tmp_path / 'decoded')


@pytest.mark.parametrize('name,kind', [('../escape', 'file'), ('/absolute', 'file'), ('C:/absolute', 'file'), ('..\\escape', 'file'), ('symlink', 'symlink')])
def test_payload_refuses_traversal_symlink(tmp_path, name, kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        item = tarfile.TarInfo(name)
        if kind == 'symlink': item.type = tarfile.SYMTYPE; item.linkname = '../escape'
        else: item.size = 1
        archive.addfile(item, None if kind == 'symlink' else io.BytesIO(b'x'))
    with pytest.raises(ValueError): t.unpack_files(lzma.compress(stream.getvalue()), tmp_path / 'decoded')


def test_sdk_transformation_removes_outputs_joins_source_without_mutation():
    nb = {'cells': [{'cell_type': 'code', 'source': ['x=1\n', 'print(x)'], 'outputs': [{'huge': 'x' * 1000}]}]}
    original = copy.deepcopy(nb)
    result = json.loads(t.sdk_text(nb))
    assert result['cells'][0]['source'] == 'x=1\nprint(x)' and result['cells'][0]['outputs'] == []
    assert nb == original


def test_bundle_source_tamper_rejected(tmp_path):
    p = tmp_path / 'helper.py'; p.write_text('safe', encoding='utf-8')
    t.write_new(tmp_path / 'trial-manifest.json', {'schema': t.SCHEMA, 'files': {'helper.py': t.sha(p)}})
    digest = t.sha(tmp_path / 'trial-manifest.json')
    t.check_bundle(tmp_path, digest)
    p.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError): t.check_bundle(tmp_path, digest)


def test_build_stops_before_assets_upload_if_local_link_fails(tmp_path, monkeypatch):
    root = tmp_path / 'root'; (root / 'tools').mkdir(parents=True)
    for name, digest in t.SOURCE_SHA.items():
        (root / 'tools' / name).write_text('tiny', encoding='utf-8')
    bridge = root / 'tools/b_rn3f_thread_input_v2.py'; bridge.write_text('bridge', encoding='utf-8')
    monkeypatch.setattr(t, 'SOURCE_SHA', {n: t.sha(root / 'tools' / n) for n in t.SOURCE_SHA})
    reference = tmp_path / 'reference'; (reference / 'result').mkdir(parents=True)
    for name in ['run-receipt.json', 'result/receipt.json', 'result/members.npz', 'result/predictions.csv', 'receipt.json', 'input.json', 'protocol.json']:
        (reference / name).write_text('{}', encoding='utf-8')
    t.write_new(reference / 'thread-protocol.json', {'adapter_sha256': t.sha(bridge)})
    t.write_new(reference / 'input-acceptance.json', {})
    def fail(*args): raise ValueError('actual normalized hash mismatch')
    monkeypatch.setattr(t, 'validate_reference', fail)
    out = tmp_path / 'stage'
    args = SimpleNamespace(root=root, assets=tmp_path / 'absent-assets', out=out, protocol=reference / 'protocol.json',
                           local_input=reference / 'input.json', local_resource=reference, local_work=reference,
                           thread_protocol=reference / 'thread-protocol.json', local_input_acceptance=reference / 'input-acceptance.json')
    with pytest.raises(ValueError, match='normalized'): t.build(args)
    assert t.read(out / 'build-failure.json')['status'] == 'HOLD_LOCAL_REFERENCE_LINKAGE'
    assert not (out / 'upload').exists() and not (out / 'notebook').exists()


def test_exit_zero_without_receipts_is_failure_not_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'check_bundle', lambda *args: {})
    result = t.finalize(tmp_path, tmp_path / 'assets', 'manifest', 0, 2.)
    assert result['status'] == 'FAILED' and result['submission_ready'] is False
    assert (tmp_path / 'evidence.zip').is_file()


@pytest.mark.parametrize('tamper', [False, True])
def test_finalizer_requires_parent_worker_hash_and_explicit_numerical_pass(tmp_path, monkeypatch, tamper):
    monkeypatch.setattr(t, 'check_bundle', lambda *args: {})
    monkeypatch.setattr(t, 'validate_thread_binding', lambda *args: {})
    base = tmp_path
    for directory in ['resource', 'work/producer/result', 'bundle/local-work/result']:
        (base / directory).mkdir(parents=True)
    t.write_new(base / 'resource/receipt.json', resource())
    parents = []
    for work in [base / 'bundle/local-work', base / 'work/producer']:
        t.write_new(work / 'run-receipt.json', {'pid': 99})
        t.write_new(work / 'result/receipt.json', {'status': 'COMPLETE'})
        parents.append({'parent_sha256': t.sha(work / 'run-receipt.json'), 'worker_sha256': t.sha(work / 'result/receipt.json')})
    t.write_new(base / 'work/paired.json', {'status': 'PASS_NUMERICAL_ONLY'})
    t.write_new(base / 'work/paired-acceptance.json', {'status': 'PASS_COMPLETION_AND_NUMERICAL_ONLY',
                'numerical_output_sha256': t.sha(base / 'work/paired.json'), 'parent_receipts': parents})
    t.write_new(base / 'work/workflow.json', {'status': 'WORKFLOW_NUMERICAL_COMPLETE_RESOURCE_PENDING',
                'trial_manifest_sha256': 'manifest', 'paired_sha256': t.sha(base / 'work/paired.json'),
                'paired_acceptance_sha256': t.sha(base / 'work/paired-acceptance.json')})
    if tamper: (base / 'work/producer/result/receipt.json').write_text('changed', encoding='utf-8')
    result = t.finalize(base, base / 'assets', 'manifest', 0, 3.)
    assert (result['status'] == 'PASS_ENGINEERING_NUMERICAL_AND_RESOURCE_ONLY') is not tamper
    assert result['submission_ready'] is False


def test_synthetic_build_generates_valid_small_notebook_and_hardlinks(tmp_path, monkeypatch):
    root = tmp_path / 'root'; (root / 'tools').mkdir(parents=True)
    for name in t.SOURCE_SHA: (root / 'tools' / name).write_text('fixture', encoding='utf-8')
    bridge = root / 'tools/b_rn3f_thread_input_v2.py'; bridge.write_text('bridge', encoding='utf-8')
    monkeypatch.setattr(t, 'SOURCE_SHA', {n: t.sha(root / 'tools' / n) for n in t.SOURCE_SHA})
    metadata = root / '.local/d01-runtime-staging/20261003-v4/kernel-metadata.json'
    metadata.parent.mkdir(parents=True); metadata.write_text('{}', encoding='utf-8')
    assets = tmp_path / 'assets'; assets.mkdir(); weight = assets / 'small.pt'; weight.write_bytes(b'tinyfixture')
    t.write_new(assets / 'b-rn3f-assets.json', {'files': {'small.pt': t.sha(weight)}})
    reference = tmp_path / 'reference'; (reference / 'result').mkdir(parents=True)
    for name in ['run-receipt.json', 'result/receipt.json', 'result/members.npz', 'result/predictions.csv', 'receipt.json', 'input.json', 'protocol.json']:
        (reference / name).write_text('{}', encoding='utf-8')
    t.write_new(reference / 'thread-protocol.json', {'adapter_sha256': t.sha(bridge)})
    t.write_new(reference / 'input-acceptance.json', {})
    monkeypatch.setattr(t, 'validate_reference', lambda *args: {'status': 'synthetic'})
    out = tmp_path / 'stage'
    args = SimpleNamespace(root=root, assets=assets, out=out, protocol=reference / 'protocol.json',
                           local_input=reference / 'input.json', local_resource=reference, local_work=reference,
                           thread_protocol=reference / 'thread-protocol.json', local_input_acceptance=reference / 'input-acceptance.json')
    result = t.build(args)
    assert result['physical_weight_copies'] == 0 and result['sdk_script_body_bytes'] < 1_000_000
    assert (out / 'upload/small.pt').stat().st_ino == weight.stat().st_ino
    nb = t.read(out / 'notebook/trial.ipynb'); ast.parse(''.join(nb['cells'][0]['source']))
    meta = t.read(out / 'notebook/kernel-metadata.json')
    assert meta['is_private'] and not meta['enable_internet'] and meta['enable_gpu']
    assert result['push_argv'][-2:] == ['--timeout', '1800']
    assert '--public' not in result['upload_argv']


@pytest.mark.parametrize('bad', [None, 'status', 'actual_threads', 'source', 'input_hash', 'binding'])
def test_thread_bridge_requires_actual_thread_and_producer_binding(tmp_path, monkeypatch, bad):
    binding = {'parent_sha256': 'p', 'worker_sha256': 'w', 'selected_threads': 12}
    helper = SimpleNamespace(validate_input_report=lambda *args: None)
    bridge = SimpleNamespace(SCHEMA='B_RN3F_THREAD_INPUT_V2', PASS='PASS_INPUT_AND_PRODUCER_LINK_ONLY',
                              load_math=lambda *args: helper, validate_contract=lambda *args: None,
                              completed_input_receipt=lambda *args: ({}, binding))
    monkeypatch.setattr(t, 'load', lambda *args: bridge)
    t.write_new(tmp_path / 'thread-protocol.json', {'adapter_sha256': 'bridge'})
    protocol_sha = t.sha(tmp_path / 'thread-protocol.json')
    report = {'rows': [{'uid': str(i)} for i in range(100)], 'thread_adapter': {
                'adapter_sha256': 'bridge', 'thread_protocol_sha256': protocol_sha, 'producer_binding': binding,
                'actual_threads': 12, 'actual_interop_threads': 1, 'inference_mode': True}}
    if bad == 'actual_threads': report['thread_adapter']['actual_threads'] = 1
    t.write_new(tmp_path / 'input.json', report)
    receipt = {'schema': bridge.SCHEMA, 'status': bridge.PASS, 'adapter_sha256': 'bridge',
                'input_report_sha256': t.sha(tmp_path / 'input.json'), 'thread_protocol_sha256': protocol_sha,
                'math_protocol_sha256': t.PROTOCOL_SHA, 'math_helper_sha256': t.SOURCE_SHA['b_rn3f_parity.py'],
                'exact_bound_studies': 100, 'producer_binding': binding, 'actual_threads': 12}
    if bad == 'status': receipt['status'] = 'FAILED'
    if bad == 'source': receipt['adapter_sha256'] = 'different'
    if bad == 'input_hash': receipt['input_report_sha256'] = 'different'
    if bad == 'binding': receipt['producer_binding'] = {'selected_threads': 12}
    t.write_new(tmp_path / 'acceptance.json', receipt)
    if bad:
        with pytest.raises(ValueError): t.validate_thread_binding(tmp_path, tmp_path, tmp_path / 'input.json', tmp_path / 'acceptance.json', 'local-reference')
    else: t.validate_thread_binding(tmp_path, tmp_path, tmp_path / 'input.json', tmp_path / 'acceptance.json', 'local-reference')
