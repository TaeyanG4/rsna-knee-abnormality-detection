"""Fail-closed competition schema validation. Never fills missing predictions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

UID='StudyInstanceUID'
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def validate_submission(frame: pd.DataFrame, expected_uids: list[str]) -> dict:
    if list(frame.columns)!=[UID]+TARGETS:
        raise ValueError('Column names/order must exactly match the official schema')
    if len(expected_uids)==0 or len(set(expected_uids))!=len(expected_uids):
        raise ValueError('Expected test UIDs must be nonempty and unique')
    if frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError('Null or duplicate submission UIDs')
    if frame[UID].tolist()!=expected_uids:
        raise ValueError('Submission UID order/set differs from official test.csv')
    values=frame[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
        raise ValueError('Predictions must be finite probabilities in [0,1]')
    return {'rows':len(frame),'targets':len(TARGETS),'uid_order_match':True,
            'min_probability':float(values.min()),'max_probability':float(values.max()),'valid':True}

def validate_file(path: str|Path, test_csv: str|Path) -> dict:
    test=pd.read_csv(test_csv,dtype={UID:str},usecols=[UID])
    frame=pd.read_csv(path,dtype={UID:str})
    return validate_submission(frame,test[UID].tolist())
