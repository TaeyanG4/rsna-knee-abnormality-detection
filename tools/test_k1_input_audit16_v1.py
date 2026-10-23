"""Synthetic checks for audit interpretations; no diagnosis or data rows."""
import unittest
from collections import Counter
import numpy as np
from tools import k1_input_audit16_v1 as a


class AuditTests(unittest.TestCase):
    def test_slab_remainders_match_actual_source(self):
        for n in [1,3,7,10,11,19,31,61,62,63]:
            for spacing in [.4,.7,1.,1.5,1.75,3.5]:
                groups=a.slab_groups(n,spacing)
                count,starts=a.slab_source_oracle(n,spacing)
                self.assertEqual(count,len(groups))
                self.assertEqual(starts,[g[0] for g in groups])

    def test_thin_tail_is_not_claimed_included(self):
        groups=a.slab_groups(61,.7)
        self.assertEqual(len(groups),12)
        self.assertEqual(sum(map(len,groups)),60)
        self.assertNotIn(60,[i for g in groups for i in g])

    def test_short_series_not_falsely_slabbed(self):
        self.assertEqual(a.slab_groups(3,.4),[[0],[1],[2]])

    def test_thick_series_no_tail_loss(self):
        self.assertEqual(sum(map(len,a.slab_groups(31,3.5))),31)

    def test_bound_window_source_does_not_cross_views(self):
        w=a.window_functions()
        blocks=[dict(n_slices=2,views=[(0,0,10)]),dict(n_slices=7,views=[(0,1,20)])]
        meta=lambda f,r:np.array([0,0,0,0,0,np.log1p(3.5)])
        rows=w['window_rows'](blocks,meta,'eval',3,3,3,3.5)
        self.assertTrue(rows)
        for row in rows:
            lo,hi=(10,12) if row[1]<20 else (20,27)
            self.assertTrue(all(lo<=x<hi for x in row[2:]))

    def test_border_only_content_not_full_blank(self):
        x=np.zeros((384,384),np.uint8);x[:8,:]=255
        p=a.pixel_check(x)
        self.assertFalse(p['all_zero_full']);self.assertTrue(p['all_zero_crop'])
        self.assertEqual(p['nonzero_removed_fraction'],1.)

    def test_constant_nonzero_is_not_all_zero(self):
        p=a.pixel_check(np.full((384,384),11,np.uint8))
        self.assertTrue(p['constant_full']);self.assertFalse(p['all_zero_full'])

    def test_selected_scope_excludes_calibration_and_has_fixed_strata(self):
        reg=a.read(a.OUT/'REGISTRATION.json');rows=a.read(a.OUT/'selection.private.json')
        self.assertEqual(a.sha(a.OUT/'selection.private.json'),reg['selection_sha256'])
        self.assertEqual(len({r['component'] for r in rows}),16)
        self.assertEqual(set(Counter(r['audit_stratum'] for r in rows).values()),{4})
        self.assertTrue(all(r['role'] in ['training','audit_dev'] for r in rows))


if __name__=='__main__':unittest.main()
