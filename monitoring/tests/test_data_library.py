import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from monitoring.kneeops.api import create_app
from monitoring.kneeops.data_library import build_library, library, study_path
from monitoring.kneeops.data_manager import settings
from monitoring.tests.test_viewer import dicom_file, viewer


def save_plan(store, root, files):
    with store.connect() as con:
        con.execute('CREATE TABLE downloads(plan TEXT, created_at TEXT)')
        con.execute('INSERT INTO downloads VALUES (?, ?)',
                    (json.dumps({'destination': str(root), 'files': files}), '2026-10-01'))


def test_inventory_counts_missing_sizes_and_preserves_originals(viewer):
    store = viewer.store
    root = Path(settings(store)['destination'])
    relative = 'test_series/1.9/1.9.1/1.dcm'
    file = root / relative
    dicom_file(file, study='1.9', series='1.9.1')
    original = file.read_bytes()
    save_plan(store, root, [
        {'path': relative, 'bytes': len(original)},
        {'path': 'test_series/1.9/1.9.1/2.dcm', 'bytes': len(original)},
        {'path': 'test_series/1.8/1.8.1/1.dcm', 'bytes': len(original)},
    ])
    (store.ws.resolve('data') / 'test_series.csv').write_text(
        'StudyInstanceUID,SeriesInstanceUID\n1.9,1.9.1\n1.9,1.9.2\n1.8,1.8.1\n')
    result = build_library(store)
    items = {row['uid']: row for row in result['studies']}
    assert items['1.9']['status'] == 'partial'
    assert items['1.9']['missing_files'] == items['1.9']['missing_series'] == 1
    assert items['1.8']['status'] == 'missing'
    assert result['files'] == 1 and result['expected_files'] == 3
    assert not any(row['training_allowed'] for row in result['studies'])
    assert library(store) == result
    assert study_path(store, items['1.9']['id'])[0] == file.parent.parent
    for key in (items['1.8']['id'], '../escape', 'unknown'):
        with pytest.raises(ValueError):
            study_path(store, key)
    descriptor = json.loads(Path(result['descriptor']).read_text(encoding='utf-8'))
    assert descriptor['caches'][0]['studies'] == 2
    assert descriptor['raw'][1]['file_manifest'][0]['path'].startswith('workspace://')
    assert file.read_bytes() == original
    file.write_bytes(original[:-1])
    changed = next(row for row in build_library(store)['studies'] if row['uid'] == '1.9')
    assert changed['invalid_sizes'] == 1 and not changed['training_allowed']


def test_training_inputs_require_train_split_and_expected_file_sizes(viewer):
    store = viewer.store
    root = Path(settings(store)['destination'])
    files = []
    for split, uid in [('train', '1.2.3'), ('test', '1.9'), ('train', '1.2.4')]:
        relative = f'{split}_series/{uid}/{uid}.1/1.dcm'
        file = root / relative
        dicom_file(file, study=uid, series=uid + '.1')
        if uid != '1.2.4':
            files.append({'path': relative, 'bytes': file.stat().st_size})
    save_plan(store, root, files)
    items = {row['uid']: row for row in build_library(store)['studies']}
    assert items['1.2.3']['training_allowed']
    assert items['1.9']['status'] == 'complete' and not items['1.9']['training_allowed']
    assert items['1.2.4']['status'] == 'unverified' and not items['1.2.4']['training_allowed']
    assert viewer.studies('raw', mode='labeled')['items'][0]['uid'] == '1.2.3'
    assert viewer.studies('raw', target='ACL')['total'] == 1
    assert viewer.studies('raw', mode='missing')['total'] == 1
    assert viewer.studies('raw', offset=40)['items'] == []


def test_raw_inventory_api_opens_dicom_and_exports_descriptor(viewer):
    store = viewer.store
    root = Path(settings(store)['destination'])
    dicom_file(root / 'test_series/1.9/1.9.1/1.dcm', study='1.9', series='1.9.1')
    result = build_library(store)
    with TestClient(create_app(store.ws.root, store.state)) as client:
        assert client.get('/api/data/library').json()['files'] == 1
        assert client.get('/api/data/library/descriptor').json()['raw'][0]['split'] == 'test'
        headers = {'x-kneeops-token': client.get('/api/session').json()['token']}
        body = {'raw': result['studies'][0]['id']}
        assert client.post('/api/viewer/open', json=body).status_code == 403
        opened = client.post('/api/viewer/open', json=body, headers=headers).json()
        assert opened['local_inventory']['status'] == 'unverified'
        assert opened['metadata']['report'] == ''
        assert any('학습 입력' in warning for warning in opened['warnings'])
        frame = client.post('/api/viewer/frame', json={'session': opened['id']}, headers=headers).json()
        assert frame['shape'] == [12, 16]
        assert client.post('/api/viewer/open', json={'raw': 'bad'}, headers=headers).status_code == 400
        assert client.post('/api/viewer/open', json={**body, 'path': str(root)}, headers=headers).status_code == 400
