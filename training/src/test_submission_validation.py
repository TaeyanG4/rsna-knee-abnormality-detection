import unittest
import numpy as np
import pandas as pd
from submission_validation import UID,TARGETS,validate_submission

class SubmissionTests(unittest.TestCase):
    def frame(self):
        f=pd.DataFrame(np.full((3,12),.5),columns=TARGETS)
        f.insert(0,UID,['synthetic_a','synthetic_b','synthetic_c']);return f
    def test_valid(self):
        f=self.frame();self.assertTrue(validate_submission(f,f[UID].tolist())['valid'])
    def test_nan_rejected(self):
        f=self.frame();f.loc[0,'ACL']=np.nan
        with self.assertRaises(ValueError):validate_submission(f,f[UID].tolist())
    def test_probability_range(self):
        f=self.frame();f.loc[0,'MCL']=1.01
        with self.assertRaises(ValueError):validate_submission(f,f[UID].tolist())
    def test_order(self):
        f=self.frame()
        with self.assertRaises(ValueError):validate_submission(f,list(reversed(f[UID].tolist())))
    def test_columns(self):
        f=self.frame().rename(columns={'PF OA':'PF_OA'})
        with self.assertRaises(ValueError):validate_submission(f,f[UID].tolist())

if __name__=='__main__':unittest.main()
