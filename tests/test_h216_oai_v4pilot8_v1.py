import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from tools import h216_oai_v4pilot8_v1 as m
import numpy as np
torch, pxmod, ct, anchors = m.modules()


class V4PilotContract(unittest.TestCase):
    def study(self, uid='synthetic'):
        return dict(StudyInstanceUID=uid, series=[dict(SeriesInstanceUID='one',plane='Sagittal',
                    slot=1,fluid_sensitive=1,fat_suppression=1)])

    def anchor(self):
        return dict(study='synthetic', method='skm_tea_joint', joint_lps=[1,2,3],slab_prior_lps=[1,2,3],
                    patella_lps=None,why=[],series={'one':dict(joint=[.4,.6],patella=None,in_field=True,dist_mm=0)})

    def item(self, uid='synthetic', value=1):
        meta = np.zeros((12,12),np.float32)
        for v, start in zip([0,1,4,5],[0,3,6,9]):
            meta[start:start+3] = np.array([[0,1,0,v,k-1,np.log1p(3.5),1,1,1,k,3,0] for k in range(3)])
        return dict(uid=uid,error=None,skipped_series=[],removed=0,
                    px=torch.full((12,384,384),value,dtype=torch.uint8),meta=torch.from_numpy(meta),
                    blocks=[dict(series='one',plane='Sagittal',n_slices=3,views=[[0,0],[1,3],[4,6],[5,9]])])

    def test_complete_anchor(self):
        r = m.anchor_check(self.anchor(),self.study())
        self.assertFalse(r['readiness_hold']); self.assertFalse(r['slab_prior'])

    def test_caught_model_error_is_not_fallback_pass(self):
        r=self.anchor(); r['why']=['error:RuntimeError(synthetic)']; r['method']='slab_prior'
        with self.assertRaisesRegex(ValueError,'ANCHOR_ERROR'): m.anchor_check(r,self.study())

    def test_missing_projection_is_rejected(self):
        r=self.anchor(); r['series']={}
        with self.assertRaisesRegex(ValueError,'ANCHOR_SERIES_ROSTER'): m.anchor_check(r,self.study())

    def test_nonfinite_anchor_and_projection_rejected(self):
        r=self.anchor(); r['joint_lps'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'ANCHOR_COORDINATES'): m.anchor_check(r,self.study())
        r=self.anchor(); r['series']['one']['joint'][0]=float('inf')
        with self.assertRaisesRegex(ValueError,'ANCHOR_PROJECTION'): m.anchor_check(r,self.study())

    def test_outside_field_is_explicit_hold(self):
        r=self.anchor(); r['series']['one'].update(joint=[1.4,.6],in_field=False)
        v=m.anchor_check(r,self.study()); self.assertTrue(v['readiness_hold'])
        self.assertEqual(v['field_centre_substitutions'],['one'])

    def test_no_sagittal_is_distinct_from_segmentation_success(self):
        r=self.anchor(); r['method']='slab_prior'; st=self.study(); st['series'][0]['plane']='Coronal'
        v=m.anchor_check(r,st); self.assertTrue(v['no_sagittal']); self.assertTrue(v['slab_prior'])

    def test_complete_render(self):
        px,meta=m.item_check(self.item(),self.study()); self.assertEqual(px.shape,(12,384,384))

    def test_series_loss_is_rejected(self):
        st=self.study(); st['series'].append(dict(st['series'][0],SeriesInstanceUID='two'))
        with self.assertRaisesRegex(ValueError,'SERIES_ACCOUNTING'): m.item_check(self.item(),st)

    def test_view_loss_and_row_overlap_rejected(self):
        item=self.item(); item['blocks'][0]['views'][2][0]=3
        with self.assertRaisesRegex(ValueError,'VIEW_ROSTER'): m.item_check(item,self.study())
        item=self.item(); item['blocks'][0]['views'][1][1]=2
        with self.assertRaisesRegex(ValueError,'ROW_PARTITION'): m.item_check(item,self.study())

    def test_slot_drift_and_missing_fatsat_rejected(self):
        for col, value in ((1,7),(8,-1)):
            item=self.item(); item['meta'][:,col]=value
            with self.assertRaisesRegex(ValueError,'ROW_SEMANTICS'): m.item_check(item,self.study())

    def test_nonfinite_meta_and_zero_pixels_rejected(self):
        item=self.item(); item['meta'][0,4]=float('nan')
        with self.assertRaisesRegex(ValueError,'METADATA_CONTRACT'): m.item_check(item,self.study())
        with self.assertRaisesRegex(ValueError,'PIXEL_CONTRACT'): m.item_check(self.item(value=0),self.study())

    def test_mmap_offsets_and_full_window_parity_and_corruption(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b') as tmp:
            out=Path(tmp); recs=[]
            for uid,val in [('syntheticA',3),('syntheticB',79)]:
                item=self.item(uid,val); px,meta=m.item_check(item,self.study(uid))
                # Nonconstant frames make wrong centre/neighbour ordering detectable.
                for i in range(len(px)): px[i,:,:]=val+i
                base=out/'shards'/m.key(uid); base.mkdir(parents=True)
                np.save(base/'px.npy',px); np.save(base/'meta.npy',meta)
                recs.append(dict(uid=uid,rows=len(px),blocks=item['blocks'],serving=m.served_hashes(px,meta,item['blocks'],ct),
                                 hashes={n:m.sha(base/n) for n in ('px.npy','meta.npy')}))
            m.assemble(out,recs,pxmod); m.loader_parity(out,recs,ct)
            p=out/'index.json'; index=m.read(p); index['blocks']['syntheticB'][0]['views'][0][1]=0; m.write(p,index)
            with self.assertRaisesRegex(ValueError,'SERVING_CACHE_PARITY'): m.loader_parity(out,recs,ct)

    def test_source_binding_detects_change(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b') as tmp:
            p=Path(tmp)/'source'; p.write_bytes(b'old'); bind={str(p):m.sha(p)}; p.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError,'BOUND_FILE_CHANGED'): m.check(bind)

    def test_no_cuda_and_no_checkpoint_loaded_for_contract_tests(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(m.command()[-1],'run')


if __name__ == '__main__': unittest.main()
