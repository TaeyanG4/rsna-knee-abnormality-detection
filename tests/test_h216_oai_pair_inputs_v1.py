import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest
import numpy as np
from tools import h216_oai_pair_inputs_v1 as m


class DummyStudies:
    def __init__(self,cache,ids,n,mode,seed,**kw):
        self.uids=ids;self.n_win=n;self.mode=mode;self.seed=seed;self.epoch=0
        for k,v in kw.items():setattr(self,k,v)
    def __getitem__(self,i):
        r=np.random.default_rng(self.seed+1000003*self.epoch+i).integers(0,99999,20)
        return r,r.copy(),i


class ProductionInputs(unittest.TestCase):
    def test_exact800_is_deterministic_and_distinct(self):
        oa=[f'oa{i:04}' for i in range(1703)]; rs=[f'rs{i:04}' for i in range(3190)]
        a=m.schedule_binding(oa,rs);b=m.schedule_binding(oa,rs)
        self.assertEqual(a,b);self.assertEqual(len(set(a['oai_required_ids'])),800)
        self.assertEqual(len(a['rsna_recovery']),800)

    def test_reordered_or_duplicate_roster_fails(self):
        oa=[f'oa{i:04}' for i in range(1703)];rs=[f'rs{i:04}' for i in range(3190)]
        with self.assertRaisesRegex(ValueError,'UNSORTED'):m.schedule_binding(oa[::-1],rs)
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):m.schedule_binding(oa+[oa[0]],rs)

    def test_sparse_physical_cache_preserves_logical_window_rng_and_index(self):
        ct=type('CT',(),{'Studies':DummyStudies}); logical=['a','b','c','d','e']; required=['e','b','d']
        ds=m.ScheduledStudies(ct,'unused',logical,required)
        full=DummyStudies('unused',logical,64,'train',22601)
        for ep in (0,1,4):
            ds.epoch=ep;full.epoch=ep
            for i in (4,1,3):
                x,meta,idx=ds[i];expected=full[i][0]
                self.assertTrue(np.array_equal(x,expected));self.assertEqual(idx,i)
        with self.assertRaisesRegex(ValueError,'UNSCHEDULED'):ds[0]

    def test_missing_native_mask_not_negative(self):
        rows=[dict(study_key='OAI_0000001_R_V00',values=[None]+[0]*29,mask=[0]+[1]*29)]
        a=m.native_targets(rows,list(range(30)),set())
        self.assertEqual(a[0,0],-1)
        rows[0]['mask'][0]=1
        with self.assertRaisesRegex(ValueError,'MASK_DRIFT'):m.native_targets(rows,list(range(30)),set())

    def test_excluded_participant_and_invalid_grade_fail(self):
        rows=[dict(study_key='OAI_0000001_R_V00',values=[0]*30,mask=[1]*30)]
        with self.assertRaisesRegex(ValueError,'EXCLUDED'):m.native_targets(rows,list(range(30)),{m.participant(rows[0]['study_key'])})
        rows[0]['values'][0]=4
        with self.assertRaisesRegex(ValueError,'NATIVE_GRADE'):m.native_targets(rows,list(range(30)),set())


if __name__=='__main__':unittest.main()
