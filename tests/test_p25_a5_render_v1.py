"""A5 series selection and successful-input pixel parity with the frozen anchor."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import p25_a5_render_v1 as r

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/handoffs/evidence/2026-10/2026-10-03-anchor-study/anchor_source.py'


def original_renderer(root):
    source=SOURCE.read_text(encoding='utf-8')
    names={'ordered_files','series_side','read_crop','window','render','build_study'}
    tree=ast.parse(source)
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(definitions)==len(names)
    namespace={k:getattr(r,k) for k in ('Path','cv2','np','pd','pydicom','CROP_MM','SIZE',
        'SLICE_BAND','N_SLICE','INTENSITY','SLOTS','N_SLOT')}
    namespace.update(SERIES_ROOT=root,rsna_event=lambda *x,**kw:None,rsna_finite=r.rsna_finite)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace


@pytest.mark.parametrize('count',[1,5,16,35,80])
@pytest.mark.parametrize('plane',['Sagittal','Coronal','Axial'])
def test_first_series_band_padding_flip_and_no_cap(tmp_path,monkeypatch,count,plane):
    folder=tmp_path/'study'/'first'; folder.mkdir(parents=True)
    for i in range(count): (folder/f'{count-i:03d}.dcm').touch()
    def read(path,stop_before_pixels=False):
        n=count-int(Path(path).stem)
        arr=(np.arange(28*30).reshape(28,30)+n*7).astype(np.uint16)
        return SimpleNamespace(InstanceNumber=n,PixelSpacing=[5.,5.],
                               ImagePositionPatient=[-1,0,0],pixel_array=arr)
    monkeypatch.setattr(r.pydicom,'dcmread',read)
    monkeypatch.setattr(r,'SERIES_ROOT',tmp_path)
    rows=[dict(Anatomical_Plane=plane,Fat_Suppression=1,SeriesInstanceUID='first'),
          dict(Anatomical_Plane=plane,Fat_Suppression=1,SeriesInstanceUID='second-not-present')]
    result=r.build_study((7,'study',rows))
    reference=original_renderer(tmp_path)['build_study']((7,'study',rows))
    np.testing.assert_array_equal(result[1],reference[1])
    np.testing.assert_array_equal(result[2],reference[2])
    assert len(r.ordered_files(folder))==count
    assert Path(r.ordered_files(folder)[0]).stem==f'{count:03d}'
    assert result[1].shape==(6,16,336,336) and result[1].dtype==np.uint8
    slot={'Sagittal':0,'Coronal':2,'Axial':4}[plane]
    selected=min(16,round(.88*(count-1))-round(.12*(count-1))+1)
    assert result[2][slot]==selected and (result[2]>0).sum()==1
    offset=(16-selected)//2
    assert not result[1][slot,:offset].any()


def test_missing_acquisition_vs_operational_failures(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'SERIES_ROOT',tmp_path)
    row=dict(Anatomical_Plane='Sagittal',Fat_Suppression=1,SeriesInstanceUID='series')
    with pytest.raises(RuntimeError,match='missing metadata'): r.build_study((0,'study',[]))
    with pytest.raises(RuntimeError,match='no readable files'): r.build_study((0,'study',[row]))
    with pytest.raises(RuntimeError,match='unsafe'): r.build_study((0,'../bad',[row]))
    absent=dict(row,Anatomical_Plane='Unmatched')
    _,img,mask=r.build_study((0,'study',[absent]))
    assert not img.any() and not mask.any()
    monkeypatch.setattr(r,'ordered_files',lambda p:['one','two'])
    monkeypatch.setattr(r,'render',lambda *args:None)
    with pytest.raises(RuntimeError,match='decode failure'): r.build_study((0,'study',[row]))


def test_unreadable_order_fatal(tmp_path,monkeypatch):
    (tmp_path/'x.dcm').touch()
    monkeypatch.setattr(r.pydicom,'dcmread',lambda *x,**kw:SimpleNamespace())
    with pytest.raises(RuntimeError,match='InstanceNumber'): r.ordered_files(tmp_path)


def test_model_definitions_exact_anchor():
    source=SOURCE.read_text(encoding='utf-8')
    new=(ROOT/'tools/p25_a5_model_v1.py').read_text(encoding='utf-8')
    body=source[source.index('N_SLOT_TYPES, MASK_IDX'):source.index('models=[]',source.index('N_SLOT_TYPES, MASK_IDX'))]
    assert ast.dump(ast.parse(new[new.index('N_SLOT_TYPES, MASK_IDX'):]))==ast.dump(ast.parse(body))
