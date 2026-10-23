"""Synthetic unit tests; these results are not competition experiments."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from label_audit import ROOT,TARGETS,UID,auc_vector,validate_ids,evaluate,load_teacher

class LabelAuditTests(unittest.TestCase):
    def test_auc_matches_sklearn_with_ties(self):
        rng=np.random.default_rng(47)
        for _ in range(20):
            y=rng.integers(0,2,(58,12));p=np.round(rng.uniform(0,1,(58,12)),1)
            np.testing.assert_allclose(auc_vector(y,p),roc_auc_score(y,p,average=None),atol=1e-12)

    def test_constant_prediction_is_half(self):
        y=np.tile(np.arange(58)[:,None]%2,(1,12))
        np.testing.assert_allclose(auc_vector(y,np.full((58,12),.5)),.5)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): validate_ids(pd.DataFrame({UID:["a","a"]}),"synthetic")

    def test_partial_coverage_not_full58_score(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"logs") as td:
            y=np.tile(np.arange(58)[:,None]%2,(1,12))
            gold=pd.DataFrame(y,columns=TARGETS);gold.insert(0,UID,[f"synthetic_{i}" for i in range(58)])
            teacher=gold.iloc[:-1].copy();teacher[TARGETS]=.1+.8*teacher[TARGETS]
            path=Path(td)/"synthetic.csv";teacher.to_csv(path,index=False)
            _,r=evaluate(path,gold,"synthetic",0)
            self.assertIsNone(r["macro_auc"])
            self.assertEqual(r["gold_overlap"],57)
            self.assertAlmostEqual(r["matched_only_macro_auc"],1.)

    def test_exact_gold_copy_ineligible(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"logs") as td:
            y=np.tile(np.arange(58)[:,None]%2,(1,12))
            gold=pd.DataFrame(y,columns=TARGETS);gold.insert(0,UID,[f"synthetic_{i}" for i in range(58)])
            path=Path(td)/"synthetic.csv";gold.to_csv(path,index=False)
            _,r=evaluate(path,gold,"synthetic",0)
            self.assertTrue(r["all_gold_exact_copy_flag"])
            self.assertFalse(r["eligible_for_teacher_comparison"])

if __name__=="__main__":unittest.main()
