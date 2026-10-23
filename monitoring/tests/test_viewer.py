import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, RLELossless, generate_uid

from monitoring.kneeops.api import create_app
from monitoring.kneeops.store import Store
from monitoring.kneeops.viewer import Viewer, region, stats


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.delenv('KNEE_WORKSPACE_ROOT', raising=False)
    monkeypatch.delenv('KNEEOPS_DOWNLOAD_DIR', raising=False)
    root = tmp_path / 'workspace'
    (root / 'config').mkdir(parents=True)
    (root / 'config/workspace.toml').write_text('schema_version=1\ncompetition="rsna-knee-abnormality-detection"\n[paths]\ncache="storage/cache"\ndata="storage/competition"\n')
    store = Store(root)
    folder = root / 'storage/cache/fixture'; folder.mkdir(parents=True)
    array = np.arange(2 * 2 * 3 * 12 * 16, dtype=np.uint8).reshape(2, 2, 3, 12, 16)
    np.save(folder / 'pixels.npy', array)
    mask = np.ones((2, 2, 3), dtype=np.uint8); mask[1, 0, 1] = 0
    np.save(folder / 'mask.npy', mask)
    (folder / 'uids.csv').write_text('index,StudyInstanceUID\n0,1.2.3\n1,1.2.4\n')
    m = {'contract': 'adjacent_fixture', 'studies': 2, 'shape_total': list(array.shape), 'dtype': 'uint8',
         'slots': [{'name': 'AX_FS'}, {'name': 'SAG_FS'}], 'mask_file': 'mask.npy', 'uid_file': 'uids.csv',
         'shards': [{'file': 'pixels.npy', 'start': 0, 'stop': 2, 'bytes': (folder / 'pixels.npy').stat().st_size}]}
    (folder / 'cache_manifest.json').write_text(json.dumps(m))
    data = store.ws.resolve('data'); data.mkdir(parents=True)
    (data / 'train.csv').write_text('StudyInstanceUID,Report,ACL,MCL\n1.2.3,training report,1,0\n1.2.4,unlabeled report,,\n')
    (data / 'test.csv').write_text('StudyInstanceUID,Report,ACL\n1.9,test report must be hidden,\n')
    return Viewer(store)


def open_cache(viewer, uid='1.2.3'):
    return viewer.open(source=viewer.catalog()['caches'][0]['id'], uid=uid)


def dicom_file(path, position=0, instance=1, study='1.2.3', series='1.2.3.4', geometry=True, mono='MONOCHROME2', compressed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = MRImageStorage; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study; ds.SeriesInstanceUID = series; ds.SeriesDescription = 'Synthetic MRI'
    ds.PatientName = 'PRIVATE_SHOULD_NOT_BE_RETURNED'
    ds.Rows = 12; ds.Columns = 16; ds.InstanceNumber = instance
    ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = mono
    ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
    ds.PixelSpacing = [.5, .75]; ds.RescaleSlope = 2; ds.RescaleIntercept = -10
    if geometry:
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, position]
    ds.PixelData = (np.arange(192, dtype=np.uint16).reshape(12, 16) + instance).tobytes()
    if compressed:
        ds.compress(RLELossless)
    ds.save_as(path, enforce_file_format=True)
    return ds


def test_cache_filters_labels_and_missing_masks(viewer):
    source = viewer.catalog()['caches'][0]['id']
    all_items = viewer.studies(source)
    assert all_items['total'] == 2
    assert all_items['summary']['targets']['ACL'] == {'positive': 1, 'negative': 0, 'unknown': 1}
    assert viewer.studies(source, mode='missing')['items'][0]['uid'] == '1.2.4'
    assert viewer.studies(source, mode='labeled')['total'] == 1
    assert viewer.studies(source, target='MCL')['total'] == 0
    assert viewer.studies(source, offset=40)['items'] == []
    opened = open_cache(viewer, '1.2.4')
    assert opened['metadata']['labels']['ACL'] is None
    f = viewer.frame(opened['id'], 0, 1, roi=[0, 0, .5, .5])
    assert not f['valid'] and f['roi'] is None
    assert viewer.labels()['1.9']['report'] == ''


def test_slice_mmap_render_roi_window_and_contact(viewer, monkeypatch):
    opened = open_cache(viewer)
    calls = []
    original = np.load
    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'load', record)
    frame = viewer.frame(opened['id'], 0, 0, roi=[0, 0, .5, .5])
    assert calls == [{'mmap_mode': 'r', 'allow_pickle': False}]
    assert frame['shape'] == [12, 16]
    assert frame['roi']['pixels'] == 48 and frame['roi']['area_mm2'] is None
    assert frame['stats']['mean'] == 95.5
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(frame['image'].split(',')[1]))))
    assert image[0, 0] == 0 and image[-1, -1] == 191
    inverse = viewer.frame(opened['id'], 0, 0, low=0, high=255, invert=True)
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(inverse['image'].split(',')[1]))))
    assert image[0, 0] == 255
    assert len(viewer.contact(opened['id'], 0)['items']) == 3
    for kwargs in [{'low': 2, 'high': 1}, {'slot': -1}, {'z': 30}, {'roi': [0, 0, 2, 2]}]:
        with pytest.raises(ValueError): viewer.frame(opened['id'], **kwargs)


def test_reviews_persist_version_and_never_change_source(viewer):
    opened = open_cache(viewer)
    source = viewer.session(opened['id'])['path']
    original = source.read_bytes()
    saved = viewer.review(opened['id'], 0, 1, 'crop inspection', ['crop'], roi=[0, 0, .5, .5], low=1, high=190)
    assert saved['frame']['roi']['pixels'] == 48
    assert saved['frame']['pixel_sha256'] and saved['manifest_sha256']
    assert 'image' not in saved['frame']
    reopened = open_cache(Viewer(viewer.store))
    assert reopened['reviews'][0]['id'] == saved['id']
    assert source.read_bytes() == original
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='변경'):
        viewer.frame(opened['id'])
    updated = open_cache(viewer)
    assert updated['revision'] != opened['revision']
    assert updated['reviews'][0]['revision'] == opened['revision']


def test_dicom_geometry_rescale_monochrome_spacing_and_compression(viewer, tmp_path):
    folder = tmp_path / 'dicom'
    dicom_file(folder / 'a.dcm', position=3, instance=1, mono='MONOCHROME1', compressed=True)
    dicom_file(folder / 'z.dcm', position=0, instance=9, mono='MONOCHROME1', compressed=True)
    opened = viewer.open(path=str(folder))
    assert opened['series'][0]['positions_mm'] == [0, 3]
    assert opened['series'][0]['ordering'] == 'IOP/IPP 물리 위치'
    assert 'PRIVATE_SHOULD_NOT_BE_RETURNED' not in json.dumps(opened)
    frame = viewer.frame(opened['id'], 0, 0, low=0, high=400, roi=[0, 0, .5, .5])
    assert frame['stats']['min'] == 8  # instance 9 sorted first: 9 * 2 - 10
    assert frame['roi']['area_mm2'] == 18
    assert frame['monochrome1']
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(frame['image'].split(',')[1]))))
    assert image[0, 0] > image[-1, -1]


def test_dicom_fallback_multiple_studies_and_multiframe_rejected(viewer, tmp_path):
    folder = tmp_path / 'fallback'
    dicom_file(folder / 'a.dcm', instance=8, geometry=False)
    dicom_file(folder / 'b.dcm', instance=2, geometry=False)
    opened = viewer.open(path=str(folder))
    assert 'InstanceNumber' in opened['series'][0]['ordering']
    assert opened['warnings']
    assert viewer.frame(opened['id'])['stats']['min'] == -6
    dicom_file(folder / 'c.dcm', study='1.4.5')
    with pytest.raises(ValueError, match='여러 study'):
        viewer.open(path=str(folder))
    file = tmp_path / 'multi.dcm'; ds = dicom_file(file); ds.NumberOfFrames = 2; ds.save_as(file, enforce_file_format=True)
    with pytest.raises(ValueError, match='단일 프레임'):
        viewer.open(path=str(file))


def test_dicom_duplicate_geometry_and_spacing_mismatch_warning(viewer, tmp_path):
    folder = tmp_path / 'spacing'
    dicom_file(folder / 'a.dcm')
    file = folder / 'b.dcm'; ds = dicom_file(file); ds.PixelSpacing = [2, 2]; ds.save_as(file, enforce_file_format=True)
    opened = viewer.open(path=str(folder))
    assert opened['series'][0]['spacing'] is None
    assert any('중복' in w for w in opened['warnings'])
    assert viewer.frame(opened['id'])['spacing'] is None


def test_npy_nonfinite_unsupported_shapes_and_pickle(viewer, tmp_path):
    path = tmp_path / 'array.npy'
    np.save(path, np.array([[1, np.nan], [np.inf, 3]], dtype=np.float32))
    opened = viewer.open(path=str(path))
    frame = viewer.frame(opened['id'])
    assert frame['stats']['nonfinite'] == 2 and frame['stats']['mean'] == 2
    assert frame['spacing'] is None and opened['metadata'] is None
    np.save(path, np.ones((2, 2, 2, 2)))
    with pytest.raises(ValueError, match='단독 NPY'):
        viewer.open(path=str(path))
    np.save(path, np.array([{'code': 'never execute'}], dtype=object))
    with pytest.raises(ValueError): viewer.open(path=str(path))
    np.save(path, np.ones((2, 2)))
    assert viewer.frame(viewer.open(path=str(path))['id'])['window'] == [1, 2]
    assert stats(np.array([[np.nan]], dtype=np.float32))['mean'] is None
    np.save(path, np.array([[-4, -2], [0, 0]], dtype=np.float32))
    assert viewer.frame(viewer.open(path=str(path))['id'])['window'][1] == 0


def test_metadata_browse_confined_and_mask_values_validated(viewer):
    root = next(r for r in viewer.catalog()['roots'] if r['name'] == '캐시 폴더')
    assert viewer.browse(root['id'])['items'][0]['name'] == 'fixture'
    for relative in ('../competition', '/etc', 'C:/Windows'):
        with pytest.raises(ValueError): viewer.browse(root['id'], relative)
    path = viewer.store.ws.resolve('cache') / 'fixture/mask.npy'
    mask = np.load(path, allow_pickle=False); mask[0, 0, 0] = 2; np.save(path, mask)
    with pytest.raises(ValueError, match='0 또는 1'): open_cache(viewer)


def test_viewer_api_origin_token_and_export(viewer):
    app = create_app(viewer.store.ws.root, viewer.store.state)
    with TestClient(app) as client:
        catalog = client.get('/api/viewer/catalog').json()
        body = {'source': catalog['caches'][0]['id'], 'uid': '1.2.3'}
        assert client.post('/api/viewer/open', json=body).status_code == 403
        headers = {'x-kneeops-token': client.get('/api/session').json()['token']}
        assert client.post('/api/viewer/open', json=body, headers={**headers, 'origin': 'https://outside.example'}).status_code == 403
        opened = client.post('/api/viewer/open', json=body, headers=headers).json()
        request = {'session': opened['id'], 'slot': 0, 'slice': 0, 'note': 'review', 'flags': ['checked']}
        assert client.post('/api/viewer/reviews', json=request, headers=headers).status_code == 200
        exported = client.get('/api/viewer/export', params={'session': opened['id']})
        assert exported.status_code == 200 and exported.json()['reviews'][0]['note'] == 'review'
        assert 'Report' not in exported.text
        assert 'no-store' in exported.headers['cache-control']
        assert client.post('/api/viewer/frame', json={'session': opened['id'], 'slice': -1}, headers=headers).status_code == 422
        assert client.get('/api/viewer/studies', params={'source': body['source'], 'target': 'fake'}).status_code == 400
