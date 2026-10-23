import json
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from monitoring.kneeops.api import create_app
from monitoring.kneeops.data_archive import index_archive, structure, materialize, exclusive
from monitoring.kneeops.data_library import build_library, library, study_path, input_descriptor
from monitoring.kneeops.data_manager import settings
from monitoring.kneeops.data_cleanup import cache_plan, apply_cache_plan, remove_verified_loose
from monitoring.kneeops.data_cache import cache_catalog
from monitoring.tests.test_viewer import viewer, dicom_file


def archive_fixture(viewer):
    store = viewer.store
    root = Path(settings(store)['destination']); root.mkdir(parents=True, exist_ok=True)
    path = root / (store.ws.config['competition'] + '.zip')
    original = root / 'train_series/1.2.3/1.2.3.4/1.dcm'
    dicom_file(original)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(original, original.relative_to(root).as_posix())
        test = root / 'test_series/1.9/1.9.1/1.dcm'
        dicom_file(test, study='1.9', series='1.9.1')
        z.write(test, test.relative_to(root).as_posix())
        for name in ('train.csv', 'test.csv'):
            z.writestr(name, (store.ws.resolve('data') / name).read_bytes())
    return root, original, path


def test_archive_lazy_open_structure_crc_reuse_and_test_separation(viewer):
    store = viewer.store
    root, original, path = archive_fixture(viewer)
    (store.ws.resolve('data') / 'train_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n1.2.3,1.2.3.4,1,1,Sagittal\n')
    data = build_library(store)
    assert len(data['studies']) == 2 and data['files'] == 2  # loose+ZIP copies collapse
    train = next(s for s in data['studies'] if s['split'] == 'train')
    test = next(s for s in data['studies'] if s['split'] == 'test')
    assert not store.ws.resolve(train['path']).exists()
    assert train['status'] == 'archived' and not train['training_allowed']
    tree = structure(store, train['id'], '1.2.3.4')
    assert tree['series'][0]['description'] == 'Sagittal · 수분 민감 · 지방억제'
    assert tree['total'] == 1 and tree['files'][0]['path'].endswith('1.dcm')
    assert not store.ws.resolve(train['path']).exists()  # browse does not extract
    folder, ready = study_path(store, train['id'])
    assert ready['training_allowed'] and ready['status'] == 'complete'
    file = folder / '1.2.3.4/1.dcm'
    assert file.read_bytes() == original.read_bytes()
    sig = file.stat().st_mtime_ns
    assert study_path(store, train['id'])[0] == folder and file.stat().st_mtime_ns == sig
    assert library(store)['prepared_studies'] == 1
    assert not study_path(store, test['id'])[1]['training_allowed']
    file.unlink()
    exported = input_descriptor(store)
    assert not next(r for r in exported['raw'] if r['id'] == train['id'])['training_allowed']
    study_path(store, train['id'])  # resume repairs a missing previously extracted member
    assert file.read_bytes() == original.read_bytes()
    with TestClient(create_app(store.ws.root, store.state)) as client:
        headers = {'x-kneeops-token': client.get('/api/session').json()['token']}
        assert client.get('/api/data/structure', params={'raw': train['id']}).status_code == 200
        assert client.post('/api/viewer/open', json={'raw': train['id']}).status_code == 403
        opened = client.post('/api/viewer/open', json={'raw': train['id']}, headers=headers).json()
        assert opened['metadata']['report'] == 'training report'
        frame = client.post('/api/viewer/frame', json={'session': opened['id']}, headers=headers).json()
        assert frame['shape'] == [12, 16]
    removed = remove_verified_loose(store, data['archives'][0], train, root)
    assert len(removed['files']) == 1 and not original.exists() and file.exists()


@pytest.mark.parametrize('name', ['../escape.dcm', '/absolute.dcm', 'train_series/1.2/1.3/../x.dcm', 'train_series/1.2/1.3/1.dcm:secret', 'train_series\\1.2\\1.3\\1.dcm'])
def test_archive_rejects_unsafe_names(viewer, name):
    path = viewer.store.ws.root / 'unsafe.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name, b'bad')
    if '\\' in name:
        path.write_bytes(path.read_bytes().replace(name.replace('\\', '/').encode(), name.encode()))
    with pytest.raises(ValueError):
        index_archive(viewer.store, path)


def test_archive_rejects_duplicate_members_symlinks_and_changed_archive(viewer):
    path = viewer.store.ws.root / 'unsafe.zip'
    with zipfile.ZipFile(path, 'w') as z:
        info = zipfile.ZipInfo('train.csv'); info.create_system = 3; info.external_attr = 0o120777 << 16
        z.writestr(info, b'target')
    with pytest.raises(ValueError):
        index_archive(viewer.store, path)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('train.csv', b'a'); z.writestr('train.csv', b'a')
    with pytest.raises(ValueError):
        index_archive(viewer.store, path)
    archive_fixture(viewer)
    data = build_library(viewer.store)
    original = viewer.store.ws.resolve(data['archives'][0]['path'])
    with original.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='변경'):
        study_path(viewer.store, data['studies'][0]['id'])


def test_extract_resume_conflict_space_crc_and_lock(viewer, monkeypatch):
    import monitoring.kneeops.data_archive as module
    store = viewer.store
    root, original, path = archive_fixture(viewer)
    data = build_library(store); row = next(r for r in data['studies'] if r['split'] == 'train')
    target = store.ws.resolve(row['path']) / '1.2.3.4/1.dcm'
    real_usage = module.shutil.disk_usage
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 0})())
    with pytest.raises(ValueError, match='공간'):
        study_path(store, row['id'])
    monkeypatch.setattr(module.shutil, 'disk_usage', real_usage)
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b'preserve me')
    with pytest.raises(ValueError, match='기존 파일'):
        study_path(store, row['id'])
    assert target.read_bytes() == b'preserve me'
    target.unlink()
    with exclusive(store.state / 'archive-index' / (row['id'] + '.extract.lock')):
        with pytest.raises(ValueError, match='준비 중'):
            study_path(store, row['id'])
    # Flip compressed payload after indexing, keeping size/mtime: CRC/decode must catch it.
    st = path.stat()
    with zipfile.ZipFile(path) as z:
        info = z.getinfo('train_series/1.2.3/1.2.3.4/1.dcm')
        position = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    with path.open('r+b') as f:
        f.seek(position); byte = f.read(1); f.seek(position); f.write(bytes([byte[0] ^ 255]))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ValueError, match='CRC'):
        study_path(store, row['id'])
    assert not target.exists() and library(store)['prepared_studies'] == 0


def test_cache_dedup_full_hash_preserves_paths_and_unique_files(viewer):
    import shutil
    store = viewer.store
    first = store.ws.resolve('cache') / 'fixture'
    second = first.with_name('copy'); shutil.copytree(first, second)
    (second / 'unique.log').write_text('keep this')
    plan_path, plan = cache_plan(store, first, second, lambda _: None)
    assert plan['reclaimable_bytes'] > 0
    apply_cache_plan(store, plan_path)
    assert (second / 'unique.log').read_text() == 'keep this'
    assert os.path.samefile(first / 'pixels.npy', second / 'pixels.npy')
    assert len(cache_catalog(store)) == 1 and cache_catalog(store)[0]['aliases']
    assert viewer.open(path=str(second / 'cache_manifest.json'))['kind'] == 'cache'
