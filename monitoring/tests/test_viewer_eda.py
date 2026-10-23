import base64
import hashlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from monitoring.kneeops.api import create_app
from monitoring.kneeops.viewer_eda import dataset_summary, preprocessing_preview, series_audit
from monitoring.tests.test_viewer import dicom_file, open_cache, viewer


def test_dataset_denominators_provenance_and_no_pixels(viewer, monkeypatch):
    data = viewer.store.ws.resolve('data')
    (data / 'train_series.csv').write_text(
        'StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive,Fat_Suppression\n'
        '1.2.3,1.2.3.4,Axial,1,1\n1.2.3,1.2.3.5,Sagittal,1,0\n1.2.4,1.2.4.5,Coronal,0,0\n')
    original = np.load
    def metadata_only(path, *args, **kwargs):
        assert str(path).endswith('mask.npy'), 'Dataset EDA must not load image shards'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', metadata_only)
    summary = dataset_summary(viewer)
    assert summary['splits']['train']['studies'] == 2
    assert summary['splits']['test']['studies'] == 1
    assert summary['splits']['train']['series'] == 3
    assert summary['splits']['train']['missing_planes']['Axial']['count'] == 1
    assert summary['labels']['ACL'] == {'positive': 1, 'negative': 0, 'unknown': 1}
    assert summary['labels']['MCL'] == {'positive': 0, 'negative': 1, 'unknown': 1}
    ai, mi = summary['targets'].index('ACL'), summary['targets'].index('MCL')
    assert summary['copositive'][ai][mi] == {'known': 1, 'both_positive': 0}
    assert summary['labeled_studies'] == 1 and summary['report_present'] == 2
    assert summary['caches'][0]['missing_studies'] == 1
    assert summary['caches'][0]['slots'][0]['missing_slices'] == 1
    assert summary['caches'][0]['missing_train'] == 0
    assert 'training report' not in json.dumps(summary)
    assert 'test report' not in json.dumps(summary)
    sha = hashlib.sha256((data / 'train.csv').read_bytes()).hexdigest()
    assert any(s['sha256'] == sha for s in summary['sources'])
    json.dumps(summary, allow_nan=False)


def test_duplicate_uid_and_train_test_overlap_do_not_erase_train_labels(viewer):
    data = viewer.store.ws.resolve('data')
    with (data / 'train.csv').open('a') as f:
        f.write('1.2.3,training report,1,0\n')
    (data / 'test.csv').write_text('StudyInstanceUID,Report,ACL\n1.2.3,hidden,\n')
    result = dataset_summary(viewer)
    assert result['train_test_uid_overlap'] == 1
    assert result['splits']['train']['duplicate_study_rows'] == 1
    assert result['labels']['ACL']['positive'] == 1
    assert result['report_present'] == 2


def test_series_audit_masks_bounds_and_source_change(viewer):
    opened = open_cache(viewer, '1.2.4')
    result = series_audit(viewer, opened['id'], 0, 8)
    assert result['sampled_slices'] == 3 and result['all_slices']
    assert result['frames'][1] == {'slice': 1, 'valid': False, 'flags': ['결측 마스크']}
    assert result['frames'][2]['adjacent_change'] is None
    assert result['geometry']['fov_mm'] is None
    assert result['recommended_window']
    for slot, samples in ((0, 65), (-1, 8), (2, 8)):
        with pytest.raises(ValueError):
            series_audit(viewer, opened['id'], slot, samples)
    path = viewer.store.ws.resolve('cache') / 'fixture/cache_manifest.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='변경'):
        series_audit(viewer, opened['id'], 0)


def test_audit_geometry_physical_file_order_and_header_privacy(viewer, tmp_path):
    folder = tmp_path / 'dicom'
    for name, position, instance in [('z', 0, 3), ('a', 3, 2), ('b', 3, 1), ('c', 9, 4)]:
        path = folder / (name + '.dcm')
        ds = dicom_file(path, position=position, instance=instance)
        ds.EchoTime = 35; ds.RepetitionTime = 2500; ds.MagneticFieldStrength = 3
        ds.SliceThickness = 3; ds.save_as(path, enforce_file_format=True)
    opened = viewer.open(path=str(folder))
    assert opened['series'][0]['filenames'][0] == 'z.dcm'
    result = series_audit(viewer, opened['id'], 0)
    assert result['geometry']['duplicate_positions'] == 1
    assert result['geometry']['irregular_steps'] == 2
    assert result['geometry']['coverage_mm'] == 9
    assert result['geometry']['fov_mm'] == [6, 12]
    assert result['series']['header']['EchoTime'] == 35
    assert 'PRIVATE_SHOULD_NOT_BE_RETURNED' not in json.dumps(result)
    assert 'files' not in result['series']


def test_nonfinite_constant_duplicate_and_bounded_sampling(viewer, tmp_path):
    path = tmp_path / 'quality.npy'
    volume = np.stack([np.zeros((8, 8)), np.zeros((8, 8)), np.full((8, 8), np.nan),
                       np.arange(64).reshape(8, 8)]).astype(np.float32)
    volume[3, 0, 0] = np.inf
    np.save(path, volume)
    opened = viewer.open(path=str(path))
    result = series_audit(viewer, opened['id'], 0)
    assert '균일한 영상' in result['frames'][0]['flags']
    assert any('픽셀 동일' in s for s in result['frames'][1]['flags'])
    assert '유한 픽셀 없음' in result['frames'][2]['flags']
    assert result['frames'][3]['nonfinite'] == 1
    json.dumps(result, allow_nan=False)
    np.save(path, np.arange(100 * 16, dtype=np.float32).reshape(100, 4, 4))
    result = series_audit(viewer, viewer.open(path=str(path))['id'], 0, 8)
    assert result['sampled_slices'] == 8 and not result['all_slices']
    assert result['frames'][0]['slice'] == 0 and result['frames'][-1]['slice'] == 99
    assert all(f['adjacent_change'] is None for f in result['frames'])


def test_preview_crop_padding_provenance_and_unchanged_source(viewer, tmp_path):
    path = tmp_path / 'preview.npy'
    np.save(path, np.arange(8 * 16, dtype=np.float32).reshape(8, 16))
    original = path.read_bytes()
    opened = viewer.open(path=str(path))
    result = preprocessing_preview(viewer, opened['id'], 0, 0, 128, 50, 'minmax')
    assert result['crop_box'] == [4, 2, 12, 6]
    assert result['retained_area_percent'] == 25
    assert result['output_shape'] == [128, 128] and result['upsampled']
    assert result['source'] and result['pixel_sha256'] and result['observed_at']
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(result['image'].split(',')[1]))))
    assert decoded.shape == (128, 128)
    assert not decoded[:32].any() and not decoded[96:].any()
    assert decoded[32:96].max() == 255
    assert path.read_bytes() == original
    for resolution, crop, norm in [(17, 100, 'p1'), (128, 0, 'p1'), (128, 100, 'unknown')]:
        with pytest.raises(ValueError):
            preprocessing_preview(viewer, opened['id'], 0, 0, resolution, crop, norm)
    masked = open_cache(viewer, '1.2.4')
    with pytest.raises(ValueError, match='결측'):
        preprocessing_preview(viewer, masked['id'], 0, 1, 128, 100, 'p1')


def test_eda_api_tree_csrf_and_input_validation(viewer, monkeypatch):
    app = create_app(viewer.store.ws.root, viewer.store.state)
    with TestClient(app) as client:
        source = client.get('/api/viewer/catalog').json()['caches'][0]['id']
        tree = client.get('/api/viewer/tree/series', params={'source': source, 'uid': '1.2.4'})
        assert tree.status_code == 200
        assert tree.json()['series'][0]['valid'] == [True, False, True]
        assert client.get('/api/viewer/eda/dataset').status_code == 200
        headers = {'x-kneeops-token': client.get('/api/session').json()['token']}
        opened = client.post('/api/viewer/open', json={'source': source, 'uid': '1.2.3'}, headers=headers).json()
        body = {'session': opened['id']}
        for endpoint in ('series', 'preview'):
            url = '/api/viewer/eda/' + endpoint
            assert client.post(url, json=body).status_code == 403
            assert client.post(url, json=body, headers={**headers, 'origin': 'https://other.test'}).status_code == 403
            assert client.post(url, json=body, headers=headers).status_code == 200
        assert client.post('/api/viewer/eda/series', json={**body, 'samples': 65}, headers=headers).status_code == 422
        assert client.post('/api/viewer/eda/preview', json={**body, 'resolution': 1024}, headers=headers).status_code == 422
        assert client.get('/api/viewer/studies?source=raw&split=invalid').status_code == 422


def test_raw_tree_root_and_split_filters(viewer, monkeypatch):
    from monitoring.kneeops import data_library
    def inventory(_):
        return {'studies': [
            {'uid': '1.2.3', 'id': 'a', 'root': 'workspace://data', 'archive_id': 'zip1', 'split': 'train', 'files': 5, 'status': 'archived'},
            {'uid': '1.9', 'id': 'b', 'root': 'workspace://data', 'archive_id': 'zip1', 'split': 'test', 'files': 7, 'status': 'archived'},
            {'uid': '1.2.4', 'id': 'c', 'root': 'workspace://other', 'split': 'train', 'files': 4, 'status': 'complete'}]}
    monkeypatch.setattr(data_library, 'library', inventory)
    assert [r['id'] for r in viewer.studies('raw', split='train', root='zip1')['items']] == ['a']
    assert viewer.studies('raw', split='test', root='zip1')['total'] == 1
    assert viewer.studies('raw', root='workspace://other')['total'] == 1
    assert viewer.studies('raw', split='train', root='zip1', target='MCL')['total'] == 0
