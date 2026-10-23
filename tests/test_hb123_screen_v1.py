"""Synthetic adapter/authentication tests; no real outcomes or report text."""
import copy
import io
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from tools import hb123_screen_v1 as screen
from tools import hb123_oai_side_v1 as side
from training.arms import laterality_v1 as lat


def header(x=70, plane='Coronal', unsafe=False):
    iop = [1.,0.,0.,0.,0.,-1.] if plane == 'Coronal' else [1.,0.,0.,0.,1.,0.]
    if unsafe: iop[0] = -1.
    return dict(ImageOrientationPatient=iop, ImagePositionPatient=[x-4.5*iop[0],0.,0.],
                PixelSpacing=[1.,1.], Rows=10, Columns=10)


def evidence(x=70, desc='L', unsafe=False):
    return [dict(summary=lat.summarize_series([header(x, plane, unsafe)], plane),
                 description_sides=[desc]) for plane in ('Coronal','Axial')]


def test_oai_id_and_description_do_not_assign_side():
    row = side.resolve('OAI_fake_R_V00', evidence())
    assert row['side'] == 'L' and row['id_disagree'] and row['description_agree']
    row = side.resolve('OAI_fake_L_V00', evidence(desc='R'))
    assert row['side'] == 'L' and row['description_disagree']
    assert side.resolve('OAI_fake_L_V00', evidence(x=0))['side'] == 'U'


def test_unsafe_orientation_and_conflicts_unknown():
    assert side.resolve('OAI_fake_L_V00', evidence(unsafe=True))['reason'] == 'unsafe_orientation'
    e = evidence(); e[1] = evidence(-70, 'R')[1]
    assert side.resolve('OAI_fake_L_V00', e)['side'] == 'U'
    e = evidence(); e[0]['summary']['tags'] = ['R']
    assert side.resolve('OAI_fake_L_V00', e)['reason'] == 'tag_geometry_disagreement'
    assert side.resolve('OAI_fake_L_V00', evidence()[:1])['reason'] == 'insufficient_evidence'


def test_raw_header_sampling_identity_geometry_and_no_pixels(tmp_path, monkeypatch):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    path = tmp_path/'dicoms.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        for i in range(5):
            ds = Dataset(); ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'; ds.SOPInstanceUID = f'1.2.3.{i+1}'
            ds.SeriesInstanceUID = '1.2.4'; ds.SeriesDescription = 'COR_LEFT'
            for k, v in header().items(): setattr(ds, k, v)
            # Deliberately undecodable pixels: successful header reading must not decode.
            ds.BitsAllocated = 16
            ds.PixelData = b'bad pixels'
            stream = io.BytesIO(); pydicom.dcmwrite(stream, ds, enforce_file_format=True)
            payload = stream.getvalue(); item = tarfile.TarInfo(f'./{i:03d}')
            item.size = len(payload); tar.addfile(item, io.BytesIO(payload))
    stat = path.stat()
    group = dict(frame_refs=[[f'{i:03d}',1,f'1.2.3.{i+1}'] for i in range(5)],
        series_instance_uid='1.2.4', rows=10, columns=10,
        iop=header()['ImageOrientationPatient'], pixel_spacing=[1.,1.], plane='Coronal')
    record = dict(archive=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns, group=group)
    monkeypatch.setattr(side, 'ROOT', tmp_path)
    got = side.sampled_headers(record)
    assert [r[0] for r in got['selected_refs']] == ['000','002','004']
    assert got['summary']['headers'] == 3 and got['summary']['geometry_side'] == 'L'
    assert got['description_sides'] == ['L']
    broken = copy.deepcopy(record); broken['group']['frame_refs'][0][2] = '9.9'
    with pytest.raises(ValueError, match='identity'): side.sampled_headers(broken)
    broken = copy.deepcopy(record); broken['group']['iop'][0] = -1
    with pytest.raises(ValueError, match='geometry'): side.sampled_headers(broken)
    broken = copy.deepcopy(record); broken['mtime_ns'] += 1
    with pytest.raises(ValueError, match='archive changed'): side.sampled_headers(broken)


class FakeSides:
    def __init__(self, sides): self.sides = sides
    def require(self, ids):
        if set(ids)-self.sides.keys(): raise ValueError('missing ID')
    def side(self, uid): return self.sides[uid]


@pytest.mark.parametrize('value', ['L','R','U'])
def test_canonical_cache_exact_planes_mask_no_mutation(value):
    volume = np.arange(5*3*4, dtype=np.uint8).reshape(5,3,4)
    original = volume.copy(); mask = np.ones(5, np.uint8)
    base = SimpleNamespace(spec=dict(slots=[('Sagittal',1,2),('Coronal',1,2),('Axial',-1,1)]),
        get=lambda uid: (volume,mask), verified={'one'}, mask=lambda uid:mask)
    cache = screen.CanonicalCache(base, FakeSides({'k':value}), ['k'])
    actual, actual_mask = cache.get('k')
    expected = volume.copy()
    if value == 'L': expected[2:] = volume[2:,:,::-1]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(volume, original)
    assert actual_mask is mask and cache.mask('k') is mask and cache.verified is base.verified
    if value != 'L': assert actual is volume
    with pytest.raises(ValueError): screen.CanonicalCache(base, FakeSides({}), ['k'])


def test_control_is_identity_arm_only_registration(monkeypatch):
    base = object(); contract = screen.rendering('control')
    assert screen.serving_cache('control', base, contract, ['k']) is base
    with pytest.raises(ValueError): screen.serving_cache('control', base, {'adapter':'canonical'}, ['k'])
    with pytest.raises(ValueError): screen.key('rank',42)
    monkeypatch.setattr(side, 'validate', lambda:dict(sha256='tampered',path='none'))
    with pytest.raises(ValueError, match='digest'): screen.rendering('right')


def test_missing_control_fails_before_any_recipe_or_readout(monkeypatch):
    calls=[]
    def absent(kind, seed, smoke=False):
        calls.append((kind,seed,smoke)); raise FileNotFoundError('control missing')
    monkeypatch.setattr(screen.shared, 'snapshot', absent)
    monkeypatch.setattr(screen.shared, 'panel_context', lambda:pytest.fail('must gate fits first'))
    monkeypatch.setattr(screen.build, 'same_or_new', lambda *a:pytest.fail('must not write'))
    with pytest.raises(FileNotFoundError): screen.panel_prepare()
    with pytest.raises(FileNotFoundError): screen.readout()
    assert calls == [('control',42,False)]*2


def test_runtime_pairing_and_two_seed_inventory(monkeypatch):
    monkeypatch.setattr(screen, 'snapshot', lambda k,s,smoke=False:dict(runtime_sha256=str(s)))
    with pytest.raises(ValueError, match='different runtimes'): screen.all_snapshots()
    monkeypatch.setattr(screen, 'snapshot', lambda k,s,smoke=False:dict(runtime_sha256='same'))
    assert set(screen.all_snapshots()) == {'control-s42','control-s43','right-s42','right-s43'}


def test_imported_seven_lens_math_and_new_source_pins():
    assert screen.evaluate is screen.shared.evaluate
    assert screen.fixed_bundle is screen.shared.fixed_bundle
    assert screen.image_item is screen.shared.image_item
    pins = screen.source_pins()
    for p in ('tools/multi_lens_readout_l4_v2.py','tools/hb123_oai_side_v1.py','tools/hb123_screen_v1.py'):
        assert p in pins


def test_map_covers_exact_panel_and_qa_aggregates():
    result = side.validate()
    assert result['sha256'] == screen.OAI_MAP_SHA256
    q = result['counts']
    assert q['studies'] == 557 and sum(q['side_'+s] for s in ('L','R','U')) == 557
    assert q['headers'] == q['unique_series']*3
    assert q['id_disagree'] == q['description_disagree'] == 0
    assert q['id_comparable'] == q['id_agree'] == q['side_L']+q['side_R']


def test_launch_hides_cuda_cpu_and_gpu_is_explicit():
    script = (screen.ROOT/'.local/run-scripts/hb123-screen-v1.ps1').read_text()
    assert "if ($Mode -ne 'GPU') { $env:CUDA_VISIBLE_DEVICES = '' }" in script
    assert "tools/gpu_job.py" in script and "--engineering-smoke" in script
    assert "if ($Mode -eq 'PlanGPU')" in script
