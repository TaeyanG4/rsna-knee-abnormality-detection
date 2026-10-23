import numpy as np
import pytest
from tools import hb155_crop_v1 as c

def affine(origin=(0,0,0),u=(1,0,0),v=(0,1,0)):
    return np.column_stack((u,v,origin)).astype(float)

def test_projection_matches_world():
    a=np.stack([affine((0,0,z)) for z in (0,2,4)])
    assert c.project(a,[7,9,2.1])[:3]==(1,7.,9.)
    assert c.project(a,[7,9,2.1])[3]==pytest.approx(.1)
    assert not c.outside_slab(a,[7,9,4.9])
    assert c.outside_slab(a,[7,9,5.1])

def test_singular_and_nonfinite_rejected():
    with pytest.raises(ValueError): c.project(np.zeros((1,3,3)),[0,0,0])
    with pytest.raises(ValueError): c.project(np.full((1,3,3),np.nan),[0,0,0])

def test_r7_prior_axis_selection():
    sag=dict(plane='Sagittal',shape=(11,11),aff=np.array([affine((30,40,50),(0,1,0),(0,0,1))]))
    cor=dict(plane='Coronal',shape=(11,11),aff=np.array([affine((10,20,60),(1,0,0),(0,0,1))]))
    np.testing.assert_array_equal(c.slab_prior([sag,cor]),[30,20,60])
    with pytest.raises(ValueError): c.slab_prior([sag])

def test_crop_anisotropic_world_and_padding():
    image=np.ones((100,100),np.float32)
    a=affine(u=(2,0,0),v=(0,1,0))
    x,box,pad=c.crop(image,a,[100,50,0],width_mm=40,size=32)
    assert box==[40,30,20,40] and pad==0 and np.all(x==1)
    _,_,pad=c.crop(image,a,[2,1,0],width_mm=40,size=32)
    assert pad==0  # translated in-field; no avoidable black borders
    with pytest.raises(ValueError): c.crop(image,a,[-1,50,0])


def test_crop_small_fov_caps_square_physical_width_without_padding():
    image=np.ones((20,30),np.float32)
    out,box,pad=c.crop(image,affine(),[15,10,0],width_mm=40,size=32)
    assert box[2:]==[20,20] and pad==0
    assert np.all(out==1)


def test_crop_single_pixel_dimension_never_synthesizes_border():
    out,box,pad=c.crop(np.ones((1,3),np.float32),affine(),[1,0,0])
    assert box[2:]==[1,1] and pad==0 and np.all(out==1)


def test_read_series_accepts_valid_oblique_but_rejects_wrong_plane_and_conflict():
    from types import SimpleNamespace
    angle=np.deg2rad(36)
    d=SimpleNamespace(ImageOrientationPatient=[1,0,0,0,np.sin(angle),np.cos(angle)],
        ImagePositionPatient=[0,0,0],PixelSpacing=[1,1],Rows=10,Columns=12,ImageLaterality='L')
    raw=SimpleNamespace(_WORKER={'pydicom':SimpleNamespace(dcmread=lambda *a,**kw:d)},
        _member_bytes=lambda *a:b'header')
    series=c.read_series(raw,[('one',None)],'Coronal','test')
    assert series['oblique'] and series['sides']=={'L'}
    with pytest.raises(ValueError,match='oblique_geometry'):
        c.read_series(raw,[('one',None)],'Axial','test')
    d.ImageOrientationPatient=[1,0,0,1,0,0]
    with pytest.raises(ValueError,match='invalid_geometry'):
        c.read_series(raw,[('one',None)],'Coronal','test')


def test_series_fov_fallback_and_valid_prior():
    s=dict(shape=(100,100),aff=np.array([affine((0,0,z)) for z in (0,2,4)]))
    for pt,reason in (([10,110,2],'prior_outside_fov'),([10,10,50],'prior_outside_slab'),(None,'missing_orthogonal_prior')):
        center,actual=c.series_center(s,pt)
        np.testing.assert_allclose(center,[49.5,49.5,2]); assert actual==reason
    center,reason=c.series_center(s,[10,10,2])
    np.testing.assert_array_equal(center,[10,10,2]); assert reason=='slab_prior'
    assert c.series_center(dict(s,oblique=True),[10,10,2])[1]=='oblique_series'


@pytest.mark.parametrize('z,span',[(0,2),(4,2),(50,2),(2,24)])
def test_stack_window_inside_volume_without_repeated_edge_clamping(z,span):
    a=np.array([affine((0,0,i)) for i in range(5)])
    points,record=c.stack_points(a,np.array([10.,10.,z]),span,5)
    assert all(not c.outside_slab(a,p) for p in points)
    assert len({c.project(a,p)[0] for p in points})==5
    assert record['span_mm']==4

def test_unknown_laterality_not_named_medial():
    assert c.compartment_name('Sagittal',0,None)=='low_lps_x'
    assert c.compartment_name('Coronal',1,'')=='high_lps_x'
    assert c.compartment_name('Coronal',0,'L')=='medial'
    assert c.compartment_name('Coronal',0,'R')=='lateral'
    assert c.compartment_name('Sagittal',1,'R')=='medial'
    assert c.compartment_name('Axial',0,None)=='axial_center'

def test_real_smoke_corpus_windows():
    from training.arms import hb155_crops_v1 as adapter
    from training.arms import native_resolution_v2 as native
    path=c.WORK/'crop-smoke'
    if not (path/'manifest.json').exists(): pytest.skip('real CPU smoke generated separately')
    corpus=native.h.common.Corpus(path)
    uid=str(corpus.ids[0]); labels={uid:np.full(12,.5,np.float32)}
    inner=native.h.MILWindows(path,[uid],labels,12,False,42)
    ds=adapter.Windows(inner,224)
    x,y,outuid=ds[0]
    assert x.shape==(12,3,224,224) and outuid==uid and np.isfinite(x.numpy()).all()
