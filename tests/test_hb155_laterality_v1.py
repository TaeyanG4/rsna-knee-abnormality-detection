import copy
from types import SimpleNamespace
import numpy as np
import pytest
from tools import hb155_laterality_v1 as s

@pytest.mark.parametrize('fields,expected',[
    ({'ImageLaterality':'','Laterality':'R'},{'R'}),
    ({'ImageLaterality':'L','Laterality':'R'},{'L','R'}),
    ({'BodyPartExamined':'KNEE LT'},{'L'}),
    ({'SeriesDescription':'RIGHT KNEE PD FS'},{'R'}),
    ({'SeriesDescription':'COR PD FS RIGHT'},{'R'}),
    ({'SeriesDescription':'L KNEE'},{'L'}),
    ({'SeriesDescription':'KNEE R'},{'R'}),
    ({'SeriesDescription':'FLAIR COR STIR' },set()),
    ({'SeriesDescription':'COR L SPINE'},set()),
])
def test_tag_parsing(fields,expected):
    assert s.tag_sides(SimpleNamespace(**fields))==expected

@pytest.mark.parametrize('tags,score,side,source',[
    (['L'],None,'L','dicom'),([],3.,'L','classifier'),([],-3.,'R','classifier'),
    (['L'],-3.,'R','classifier_conflict'),(['L','R'],3.,'L','classifier_conflict'),
    (['L','R'],1.,'side_unknown','unknown'),([],1.,'side_unknown','unknown'),
    (['R'],1.,'R','dicom'),([],2.,'L','classifier'),([],-2.,'R','classifier')])
def test_decision(tags,score,side,source):
    got=s.resolve(tags,score)
    assert (got['side'],got['source'])==(side,source)
    assert not got['conflict'] or got['conflict_resolved'] or side=='side_unknown'

def test_nonfinite_fail_closed():
    with pytest.raises(ValueError): s.resolve([],float('nan'))

SPEC={'slots':[['Sagittal',1,16],['Sagittal',0,16],['Coronal',1,16],['Axial',1,16]]}

def test_permutation_mask_and_channels():
    p=s.permutation(SPEC,'R')
    assert p[:16].tolist()==list(range(8,16))+list(range(8))
    assert p[48:].tolist()==list(range(48,64))
    assert np.array_equal(s.permutation(SPEC,'side_unknown'),np.arange(64))
    x=np.arange(64,dtype=np.uint8)[:,None,None]*np.ones((64,2,2),np.uint8)
    mask=np.arange(64)%3==0
    np.testing.assert_array_equal(s.channels(x[p],1)[:,0],x[p])
    assert np.all(s.channels(x,1)[:,1]==1)
    assert np.all(s.channels(x,0)[:,1]==0)
    np.testing.assert_array_equal(mask[p][:8],mask[8:16])

def test_provenance_conflict_recovery_missing_not_failure():
    old=dict(laterality='unknown',failures={'missing_series':2,'laterality_conflict':1},
        slots=[dict(crops=[dict(compartment='low_lps_x'),dict(compartment='high_lps_x')]) for _ in range(4)])
    got=s.updated_record(old,s.resolve(['L','R'],-3),SPEC)
    assert got['failures']=={} and got['missing_series']==2
    assert [r['compartment'] for r in got['slots'][0]['crops']]==['medial','lateral']
    assert old['failures']['laterality_conflict']==1

def test_exact_native_inputs_and_conflicting_tags(monkeypatch):
    from pydicom.dataset import Dataset
    from tools.exp169_r7_src import side_common
    geom=s.load_geometry(); monkeypatch.setattr(s,'_GEOM',geom); monkeypatch.setattr(s,'_SC',side_common)
    datasets={}
    for i in range(7):
        ds=Dataset(); ds.InstanceNumber=i; ds.SOPInstanceUID=f'1.2.3.{i}'
        ds.ImagePositionPatient=[0,float(i),0]; ds.ImageOrientationPatient=[-1,0,0,0,0,1]
        ds.PixelSpacing=[.5,.6]; ds.Rows=8; ds.Columns=10
        ds.Laterality='L'; ds.ImageLaterality='R'; ds.PhotometricInterpretation='MONOCHROME2'
        ds.RescaleSlope=1.; ds.RescaleIntercept=0
        ds.file_meta=Dataset(); ds.file_meta.TransferSyntaxUID='1.2.840.10008.1.2.1'
        ds.SamplesPerPixel=1; ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=0
        ds.PixelData=(np.arange(80,dtype=np.uint16).reshape(8,10)+i*80).tobytes()
        datasets[str(i)]=ds
    result=s.series_inputs(datasets,'Coronal')
    images=[geom.canonicalize_payload(geom.pixels_from_dataset(ds),'Coronal').image for ds in datasets.values()]
    vol=np.stack(images).astype(np.float32)
    native=np.rint((vol-vol.min())/(vol.max()-vol.min())*65535).astype(np.uint16)
    np.testing.assert_array_equal(result,side_common.side_inputs(native))
    assert all(ds.Laterality=='L' and ds.ImageLaterality=='R' for ds in datasets.values())

def test_roundoff_border_does_not_drop_geometry():
    from tools import hb155_crop_v1 as c
    a=np.eye(3); a[:,2]=0
    image=np.ones((20,20),np.float32)
    assert c.crop(image,a,[-1e-12,10,0])[2]==0
    with pytest.raises(ValueError): c.crop(image,a,[-1e-5,10,0])

def test_registered_threshold_not_relaxed():
    assert s.THRESHOLDS['max_unknown_laterality_fraction']==.02
    for k in ('empty_studies','decode_failures','laterality_conflicts','geometry_failures'):
        assert s.THRESHOLDS[k]==0


def test_threshold_boundary_and_fail_closed():
    assert not s.limits({'unknown_laterality':2,'missing_series':20},100,6400,0,0)[1]
    assert 'unknown_laterality_fraction' in s.limits({'unknown_laterality':3},100,6400,0,0)[1]
    assert 'laterality_conflicts' in s.limits({'laterality_conflict':1},100,6400,0,0)[1]
    assert 'decode_failures' in s.limits({'header_decode_error':1},100,6400,0,0)[1]
    assert 'empty_studies' in s.limits({},100,6400,0,1)[1]


def test_audit_resume_tamper_rejected(tmp_path):
    row=dict(uid='one',**s.resolve(['L'],3.))
    path=tmp_path/'audit.json'; item=dict(identity={},rows=[row],rows_sha256=s.rows_hash([row]))
    s.atomic_json(path,item)
    assert s.checked_audit(path,{},['one'])==item
    item['rows'][0]['side']='R'; s.atomic_json(path,item)
    with pytest.raises(ValueError,match='audit rows changed'): s.checked_audit(path,{},['one'])


def test_overlay_coverage_and_hash_gate():
    names={f'overlays/study-{i:04d}.png':str(i) for i in range(24)}
    parent=dict(shards=[dict(overlays=names)],identity=dict(overlay_positions=list(range(24))))
    assert s.overlays_valid(names,parent)
    assert not s.overlays_valid({},parent)
    changed=dict(names); changed['overlays/study-0000.png']='changed'
    assert not s.overlays_valid(changed,parent)


@pytest.mark.parametrize('target,message',[
    ('picks.jsonl','source picks drift'),('raptor_corpus.py','raw builder drift')])
def test_registration_rejects_parent_input_drift(monkeypatch,target,message):
    original=s.crop.sha
    monkeypatch.setattr(s.crop,'sha',lambda p:'changed' if str(p).endswith(target) else original(p))
    with pytest.raises(ValueError,match=message): s.registration()
