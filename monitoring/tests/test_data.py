import hashlib
import json
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from monitoring.kneeops.api import create_app
from monitoring.kneeops.data_cache import cache_catalog, cache_descriptor, cache_origin, inspect_cache
from monitoring.kneeops.data_manager import DataManager, RESERVE, destination, settings, studies
from monitoring.kneeops.data_remote import safe_remote, tree
from monitoring.kneeops.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv('KNEE_WORKSPACE_ROOT', raising=False)
    monkeypatch.delenv('KNEEOPS_DOWNLOAD_DIR', raising=False)
    root = tmp_path / 'project'
    (root / 'config').mkdir(parents=True)
    (root / 'config/workspace.toml').write_text('schema_version=1\ncompetition="rsna-knee-abnormality-detection"\n[paths]\ncache="storage/cache"\ndata="storage/competition"\n', encoding='utf-8')
    return Store(root)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'x//a', 'x/../a', 'C:/x', 'x\\a', '.', 'x./a', 'x?token=secret'])
def test_remote_traversal_rejected(name):
    with pytest.raises(ValueError): safe_remote(name)


def test_destination_and_selected_metadata_only(store):
    assert destination(store, 'storage/downloads') == store.ws.root / 'storage/downloads'
    assert settings(store)['destination'].endswith('rsna-knee-abnormality-detection')
    for value in ['', '..', str(store.ws.root), str(store.state), '//other/pc']:
        with pytest.raises(ValueError): destination(store, value)
    folder = store.ws.resolve('data'); folder.mkdir(parents=True)
    (folder / 'train_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID,Report\n1.2,3,private report\n1.2,4,private report\n1.3,5,other report\n')
    result = studies(store, query='1.2')
    assert result['items'] == [{'uid': '1.2', 'series': 2}]
    assert 'report' not in json.dumps(result)


def fake_transfer(request, report):
    folder = Path(request['staging'])
    (folder / 'payload').write_bytes(b'abc')
    (folder / 'complete.json').write_text(json.dumps({'file': 'payload', 'bytes': 3}))
    report()


def wait_job(manager, id):
    for _ in range(100):
        record = manager.get(id)
        if record['status'] not in {'queued', 'running', 'stopping'}:
            return record
        time.sleep(.01)
    pytest.fail('Download worker did not finish')


def test_plan_snapshot_preserves_tree_and_existing_files(store):
    manager = DataManager(store)
    file = {'path': 'train_series/1.2/3.4/5.6.dcm', 'bytes': 3}
    try:
        with patch('monitoring.kneeops.data_manager.remote_catalog', return_value=[file]):
            plan = manager.preview('selected', ['1.2'])
        old_dest = Path(plan['destination'])
        store.set_meta('data_destination', str(store.ws.root / 'changed-destination'))
        with patch.object(manager, 'transfer', side_effect=fake_transfer):
            manager.start(plan['id'])
            assert wait_job(manager, plan['id'])['status'] == 'complete'
        assert (old_dest / file['path']).read_bytes() == b'abc'
        assert not (store.ws.root / 'changed-destination').exists()
        store.set_meta('data_destination', str(old_dest))
        with patch('monitoring.kneeops.data_manager.remote_catalog', return_value=[file]):
            conflict = manager.preview('selected', ['1.2'])
        assert conflict['conflict_count'] == 1
        with pytest.raises(ValueError, match='기존 파일'): manager.start(conflict['id'])
        assert (old_dest / file['path']).read_bytes() == b'abc'
    finally:
        manager.close()


def test_disk_rechecked_and_queue_excludes_duplicate_starts(store):
    manager = DataManager(store)
    try:
        with patch('monitoring.kneeops.data_manager.remote_catalog', return_value=[{'path': 'test.csv', 'bytes': 3}]):
            plan = manager.preview('metadata', [])
        with patch('monitoring.kneeops.data_manager.disk', return_value={'free_bytes': 10}):
            with pytest.raises(ValueError, match='부족'): manager.start(plan['id'])
        started = threading.Event()
        def stopped(request, report):
            started.set()
            manager.cancel_event.wait(3)
            raise InterruptedError()
        with patch.object(manager, 'transfer', side_effect=stopped):
            manager.start(plan['id']); assert started.wait(2)
            with pytest.raises(ValueError, match='이미'): manager.start(plan['id'])
            manager.cancel(plan['id'])
            assert wait_job(manager, plan['id'])['status'] == 'cancelled'
    finally:
        manager.close()


def test_cancel_retains_partial_then_retries_and_promotes(store):
    manager = DataManager(store)
    try:
        with patch('monitoring.kneeops.data_manager.remote_catalog', return_value=[{'path': 'test.csv', 'bytes': 3}]):
            plan = manager.preview('metadata', [])
        def interrupted(request, report):
            p = Path(request['staging']) / 'payload'
            p.write_bytes(b'a')
            p.with_name('payload.kaggle-partial').write_text('identity marker')
            raise InterruptedError()
        with patch.object(manager, 'transfer', side_effect=interrupted):
            manager.start(plan['id'])
            assert wait_job(manager, plan['id'])['status'] == 'cancelled'
        folder = Path(plan['destination']) / '.kneeops' / plan['id'] / '0'
        assert (folder / 'payload').read_bytes() == b'a'
        assert not (Path(plan['destination']) / 'test.csv').exists()
        def resumed(request, report):
            assert (folder / 'payload').read_bytes() == b'a'
            assert (folder / 'payload.kaggle-partial').exists()
            fake_transfer(request, report)
        with patch.object(manager, 'transfer', side_effect=resumed):
            manager.start(plan['id'])
            assert wait_job(manager, plan['id'])['status'] == 'complete'
        assert (Path(plan['destination']) / 'test.csv').read_bytes() == b'abc'
    finally:
        manager.close()


def test_restart_marks_interrupted_without_deleting_partial(store):
    manager = DataManager(store)
    with patch('monitoring.kneeops.data_manager.remote_catalog', return_value=[{'path': 'test.csv', 'bytes': 3}]):
        plan = manager.preview('metadata', [])
    manager.save(plan['id'], 'running', {'current_bytes': 2})
    manager.close()
    again = DataManager(store)
    try:
        r = again.get(plan['id'])
        assert r['status'] == 'interrupted' and r['progress']['current_bytes'] == 2
    finally:
        again.close()


def npy(path, shape, dtype='|u1'):
    import math
    header = repr({'descr': dtype, 'fortran_order': False, 'shape': tuple(shape)}).encode() + b'\n'
    path.write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + bytes(math.prod(shape)))


def make_cache(store):
    folder = store.ws.resolve('cache') / 'small'; folder.mkdir(parents=True)
    npy(folder / 'shard.npy', [2, 4, 9, 2, 2]); npy(folder / 'mask.npy', [2, 4, 9])
    (folder / 'uids.csv').write_text('index,StudyInstanceUID\n0,1.2\n1,1.3\n')
    m = {'contract': 'adjacent_center9_4slot_288_crop140_v1', 'studies': 2, 'shape_total': [2, 4, 9, 2, 2], 'dtype': 'uint8',
         'shards': [{'start': 0, 'stop': 2, 'file': 'shard.npy', 'bytes': (folder / 'shard.npy').stat().st_size, 'sha256': 'declared-not-verified'}],
         'mask_file': 'mask.npy', 'uid_file': 'uids.csv', 'mask_sha256': hashlib.sha256((folder / 'mask.npy').read_bytes()).hexdigest()}
    path = folder / 'cache_manifest.json'; path.write_text(json.dumps(m))
    return path, m


def test_cache_validation_descriptor_and_duplicate_alias(store):
    path, m = make_cache(store)
    (path.parent / 'other_manifest.json').write_text(json.dumps(m))
    item = inspect_cache(path)
    assert item['valid'] and not item['hash_verified']
    assert len(cache_catalog(store)) == 1
    d = cache_descriptor(store, str(path))
    assert d['cache_manifest'].startswith('workspace://') and 'NOT verified' in d['validation']
    assert d['provenance'] is None
    (path.parent / 'uids.csv').write_text('index,StudyInstanceUID\n0,1.2\n1,1.2\n')
    assert not inspect_cache(path)['valid']
    with pytest.raises(ValueError): cache_descriptor(store, str(path))


def test_cache_origin_distinguishes_generator_from_research_reference():
    adjacent = cache_origin('7b74403d1696eb3e1376ace935903b72f962128d07b987f86afadec95c952c71')
    ortho = cache_origin('6ec26d6b5295133ad9f77cf5a543641b901bccbd4f22427a9c014e00cf0f8020')
    for item in (adjacent, ortho):
        assert not item['official_cache'] and item['creator_account'] == 'taeyangg4'
        assert item['visibility'] == 'private_in_local_metadata'
        assert item['notebook_url'] == 'https://www.kaggle.com/code/' + item['notebook_id']
        assert item['notebook_version'] is None
        assert len(item['evidence']) == 3
        assert all(e['path'].startswith('workspace://') and len(e['sha256']) == 64 for e in item['evidence'])
    assert 'reference' not in adjacent
    assert ortho['reference']['owner'] == 'lt-0123'
    assert ortho['reference']['url'].endswith(ortho['reference']['revision'])
    assert cache_origin(None) is None
    assert cache_origin('unknown') is None


def test_cache_truncated_or_traversal_never_connects(store):
    path, m = make_cache(store)
    (path.parent / 'shard.npy').write_bytes(b'incomplete')
    assert not inspect_cache(path)['valid']
    m['shards'][0]['file'] = '../../escape.npy'; path.write_text(json.dumps(m))
    assert 'escapes' in inspect_cache(path)['errors'][0]


def test_data_api_preview_does_not_start_and_requires_token(store):
    app = create_app(store.ws.root, store.state)
    with TestClient(app) as client:
        headers = {'X-Kneeops-Token': client.get('/api/session').json()['token']}
        assert client.get('/api/data').status_code == 200
        assert client.post('/api/data/preview', json={'mode': 'all'}).status_code == 403
        plan = client.post('/api/data/preview', headers=headers, json={'mode': 'all'}).json()
        assert plan['file_count'] == 1 and plan['estimated_bytes'] > 500_000_000_000
        assert client.get('/api/data/downloads').json()[0]['status'] == 'planned'
        assert not Path(plan['destination']).exists()
        assert client.post('/api/data/preview', headers=headers, json={'mode': 'selected', 'studies': ['../bad']}).status_code == 400
        assert client.put('/api/data/settings', headers=headers, json={'destination': 'storage/new-downloads'}).status_code == 200
        manifest = client.get(f"/api/data/downloads/{plan['id']}/manifest")
        assert manifest.json()['plan']['destination'] == plan['destination']
        assert 'attachment' in manifest.headers['content-disposition']
        assert client.get('/api/data/cache/descriptor').status_code == 400


def test_tree_pages_preserve_paths_and_reject_scope_escape():
    responses = [
        SimpleNamespace(directories=[], files=[SimpleNamespace(relative_url='train_series/1.2/a.dcm', name='a.dcm', total_bytes=3)], next_page_token='next'),
        SimpleNamespace(directories=[], files=[SimpleNamespace(relative_url='train_series/1.2/b.dcm', name='b.dcm', total_bytes=4)], next_page_token=''),
    ]
    calls = []
    def listing(request):
        calls.append(request.page_token)
        return responses.pop(0)
    client = SimpleNamespace(competitions=SimpleNamespace(competition_api_client=SimpleNamespace(list_data_tree_files=listing)))
    dirs, files = tree(client, 'rsna-knee-abnormality-detection', 'train_series/1.2')
    assert not dirs and calls == ['', 'next'] and sum(f['bytes'] for f in files) == 7
    responses.append(SimpleNamespace(directories=[], files=[SimpleNamespace(relative_url='test_series/escape', name='escape', total_bytes=1)], next_page_token=''))
    with pytest.raises(ValueError, match='Unexpected directory'): tree(client, 'rsna-knee-abnormality-detection', 'train_series/1.2')


def test_download_process_is_terminated_on_cancel(store):
    from unittest.mock import MagicMock
    manager = DataManager(store)
    process = MagicMock()
    process.poll.return_value = None
    manager.cancel_event.set()
    try:
        with patch('monitoring.kneeops.data_manager.subprocess.Popen', return_value=process):
            with pytest.raises(InterruptedError): manager.transfer({'staging': 'unused'}, lambda: None)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
    finally:
        manager.close()


def test_wrong_size_never_promoted(store):
    manager = DataManager(store)
    try:
        with patch('monitoring.kneeops.data_manager.remote_catalog', return_value=[{'path': 'test.csv', 'bytes': 10}]):
            plan = manager.preview('metadata', [])
        with patch.object(manager, 'transfer', side_effect=fake_transfer):
            manager.start(plan['id'])
            assert wait_job(manager, plan['id'])['status'] == 'failed'
        assert not (Path(plan['destination']) / 'test.csv').exists()
    finally:
        manager.close()


def test_resuming_archive_accounts_for_existing_partial_space(store):
    manager = DataManager(store)
    try:
        with patch.dict('monitoring.kneeops.data_manager.SNAPSHOT', {'bytes': 1000}):
            plan = manager.preview('all', [])
        stage = Path(plan['destination']) / '.kneeops' / plan['id'] / '0'
        stage.mkdir(parents=True)
        (stage / 'payload').write_bytes(b'x' * 900)
        manager.save(plan['id'], 'cancelled', {})
        with patch('monitoring.kneeops.data_manager.disk', return_value={'free_bytes': RESERVE + 250}), patch.object(manager, 'transfer', side_effect=fake_transfer):
            manager.start(plan['id'])
            assert wait_job(manager, plan['id'])['status'] == 'complete'
    finally:
        manager.close()


def test_flat_raptor_corpus_manifest_is_valid(store):
    import numpy as np
    folder = store.ws.resolve('cache') / 'RAPTOR_TEST'; folder.mkdir(parents=True)
    np.save(folder / 'vols_shard00.npy', np.zeros((2, 4, 8, 8), np.uint8))
    np.save(folder / 'masks.npy', np.ones((2, 4), np.uint8))
    (folder / 'uids.csv').write_text('StudyInstanceUID,index\n1.2\n1.3\n'.replace('1.2\n', '1.2,0\n').replace('1.3\n', '1.3,1\n'))
    m = {'contract': 'raptor64_test', 'studies': 2, 'shape_total': [2, 4, 8, 8], 'dtype': 'uint8',
         'shards': [{'start': 0, 'stop': 2, 'file': 'vols_shard00.npy', 'bytes': (folder / 'vols_shard00.npy').stat().st_size, 'sha256': 'x'}],
         'mask_file': 'masks.npy', 'uid_file': 'uids.csv', 'mask_sha256': hashlib.sha256((folder / 'masks.npy').read_bytes()).hexdigest()}
    path = folder / 'manifest.json'; path.write_text(json.dumps(m))
    item = inspect_cache(path)
    assert item['valid'] and item['shape'] == [2, 4, 8, 8], item['errors']
    from monitoring.kneeops.viewer import Viewer
    assert all(len(c['shape']) == 5 for c in Viewer(store).catalog()['caches'])  # flat corpora stay out of the MRI viewer
    m['shape_total'] = [2, 1, 4, 8, 8]; path.write_text(json.dumps(m))
    assert not inspect_cache(path)['valid']
