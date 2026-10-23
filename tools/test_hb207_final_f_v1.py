"""Focused tests for new final-F adapter boundaries, not repeated inference."""
import copy
import unittest
import numpy as np
from scipy.stats import rankdata
from tools import hb207_final_f_v1 as a
from tools import c85r7_r6w958c4_v1147 as builder


class FinalFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = {}
        for arm,(version,pin) in a.STAGES.items():
            path = a.ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7'/version/'kernel/release.ipynb'
            assert a.old.sha(path)==pin
            cls.files[arm],_,_ = builder.unpack(a.old.read(path))

    def scores(self):
        rng = np.random.default_rng(8)
        return {k:rng.integers(0,8,size=(n,58,12)).astype(np.float32)/8
                for k,n in [('l2',10),('dino',5),('convnext',5),('small',10)]}

    def op(self,arm): return a.q_operator(self.files[arm]['exp169_partner_bundle_release.py'])

    def test_nested_rank_ties_fp32_reference(self):
        s = self.scores()
        rank = lambda x:(rankdata(np.asarray(x,np.float64),axis=0,method='average')-1)/57
        families = [rank(((s['l2'][:5]+s['l2'][5:])/2).mean(0)),rank(s['dino'].mean(0)),rank(s['convnext'].mean(0))]
        families[1][:,:8] = rank(((s['small'][:5]+s['small'][5:])/2).mean(0))[:,:8]
        expected = rank(sum(rank(x) for x in families)/3)
        np.testing.assert_array_equal(self.op('F24')(s)[0],expected)

    def test_last_four_keep_dino(self):
        s = self.scores(); original = {k:v for k,v in s.items() if k!='small'}
        np.testing.assert_array_equal(self.op('F2')(s)[0][:,8:],self.op('F0')(original)[0][:,8:])

    def test_fp64_small_rejected(self):
        s = self.scores(); s['small'] = s['small'].astype(np.float64)
        with self.assertRaises(ValueError): self.op('F24')(s)

    def test_missing_family_and_nonfinite_rejected(self):
        s = self.scores(); del s['small']
        with self.assertRaises(ValueError): self.op('F2')(s)
        s = self.scores(); s['small'][0,0,0] = np.nan
        with self.assertRaises(ValueError): self.op('F24')(s)

    def test_paired_actual_operators(self):
        s = self.scores()
        np.testing.assert_array_equal(self.op('F2')(s)[0],self.op('F24')(s)[0])
        del s['small']
        np.testing.assert_array_equal(self.op('F0')(s)[0],self.op('F4')(s)[0])

    def archive(self):
        ids = [str(i) for i in range(58)]; targets = [str(i) for i in range(12)]
        return dict(ids=np.array(ids),targets=np.array(targets),
            member_ids=np.array([f'S-s{s}-f{f}' for s in (42,43) for f in range(5)]),
            probabilities=self.scores()['small']),ids,targets

    def test_cache_order_rejects(self):
        z,ids,targets = self.archive()
        a.cache_values(z,ids,targets,'small')
        for key in ('ids','targets','member_ids'):
            bad=copy.deepcopy(z); bad[key]=bad[key][::-1]
            with self.assertRaises(ValueError): a.cache_values(bad,ids,targets,'small')

    def test_cache_dtype_range_shape_rejects(self):
        z,ids,targets = self.archive()
        for values in (z['probabilities'].astype(np.float64),z['probabilities'][:9],z['probabilities']+2):
            bad=dict(z,probabilities=values)
            with self.assertRaises(ValueError): a.cache_values(bad,ids,targets,'small')

    def test_c4_fold_order_rejects(self):
        _,ids,targets=self.archive()
        z=dict(ids=np.array(ids),targets=np.array(targets),folds=np.arange(5),values=self.scores()['convnext'])
        a.cache_values(z,ids,targets,'convnext'); z['folds']=z['folds'][::-1]
        with self.assertRaises(ValueError): a.cache_values(z,ids,targets,'convnext')

    def test_ast_extraction_does_not_execute_top_level(self):
        source="raise RuntimeError('not executed')\ndef rank(x): return x\ndef aggregate(x): return rank(x)\n"
        self.assertEqual(a.q_operator(source)(3),3)

    def test_outer_closure_and_changed_rank_rejection(self):
        a.verify_outer(self.files)
        bad=copy.deepcopy(self.files)
        bad['F2']['exp169_release_c85.py']=bad['F2']['exp169_release_c85.py'].replace(
            b'result[~failed] = rank(values[~failed])',b'result[~failed] = 0.5 * rank(values[~failed])')
        with self.assertRaises(ValueError): a.verify_outer(bad)


if __name__=='__main__': unittest.main()
