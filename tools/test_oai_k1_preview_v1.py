import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools import oai_k1_preview_v1 as p

class PreviewTests(unittest.TestCase):
    def test_id_only_index_overlay_leaves_other_reads(self):
        class Parent:
            def read(self,item):return b'unchanged'
        obj=p.source_class(Parent,['fixture'])()
        self.assertEqual(obj.read('test.csv'),b'StudyInstanceUID\nfixture\n')
        self.assertEqual(obj.read('test_series.csv'),b'unchanged')
        
    def test_selection_is_order_independent(self):
        ids=[f'fixture_{i}' for i in range(20)]
        self.assertEqual(p.select(ids),p.select(ids[::-1]));self.assertEqual(len(p.select(ids)),8)
    def test_duplicate_selection_refused(self):
        with self.assertRaises(ValueError):p.select(['x']*8)
    def test_positions(self):
        self.assertEqual(p.positions(1),[0])
        self.assertEqual(len(p.positions(100)),5)
        with self.assertRaises(ValueError):p.positions(0)
    def test_display_is_bounded(self):
        x=p.display(np.arange(100).reshape(10,10))
        self.assertEqual(x.dtype,np.uint8);self.assertEqual(int(x.min()),0);self.assertEqual(int(x.max()),255)
    def test_fullfield_matches_original_v4_bytes(self):
        torch,_,px=p.setup();fn=px.series_views
        base=p.ROOT/'.local/scratch/slot-b/oai-k1-preview-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            d=Path(temp);(d/'series').mkdir();(d/'studies').mkdir()
            uid='fixture';vol=(np.arange(10*64*80).reshape(10,64,80)%65535).astype(np.uint16)
            path=d/'series/fixture.npy';np.save(path,vol)
            s=dict(SeriesInstanceUID='fixture',path=str(path),positions=list(range(0,30,3)),
                spacing_mm=[[.7,.8]]*10,native_shape=list(vol.shape),plane='Sagittal',slot=0,
                fat_suppression=1,fluid_sensitive=1,side='L',intensity={'display_window_uint16':[0,50000]})
            Path(px.study_json(d,uid)).write_text(json.dumps({'StudyInstanceUID':uid,'series':[s]}),encoding='utf-8')
            original=px.StudySlices(str(d),[uid],384,'v4',None).build(uid)
            try:
                px.series_views=p.field_only(fn)
                actual=px.StudySlices(str(d),[uid],384,'v4',None).build(uid)
            finally:px.series_views=fn
            off=next(v[1] for v in original['blocks'][0]['views'] if v[0]==0)
            np.testing.assert_array_equal(actual['px'].numpy(),original['px'][off:off+10].numpy())
            np.testing.assert_array_equal(actual['meta'].numpy(),original['meta'][off:off+10].numpy())
            self.assertFalse(torch.cuda.is_initialized())
    def test_model_load_refused(self):
        torch,_,_=p.setup()
        with self.assertRaisesRegex(RuntimeError,'CHECKPOINT_LOAD_FORBIDDEN'):torch.load('unused')

if __name__=='__main__':unittest.main()
