"""Failure-oriented guards for completed v3 cache use; no forward or metrics."""
import copy
import pytest
from tools import hb209_k3_v3_joint_v1 as joint


def fixture():
    _, overlay, _, _ = joint.old.dependencies()
    worker = joint.old.read(joint.V3/'run1/worker-receipt.json')
    cache = joint.old.read(joint.V3/'run1/cache-complete.json')
    payload = (joint.V3/'run1/predictions.csv').read_bytes()
    ids, _ = overlay.table(payload)
    contract = joint.old.read(joint.old.LIVE/'worker-receipt.json')['contract']
    return worker,cache,payload,ids,contract,overlay


def test_actual_complete_input():
    assert joint.validate_v3(*fixture()).shape == (58,12)


@pytest.mark.parametrize('defect',['wrong_model','precision','degraded','study_error','wrong_hash','wrong_roster','clean_claim'])
def test_reject_material_mismatch(defect):
    worker,cache,payload,ids,contract,overlay = fixture()
    worker,cache = copy.deepcopy(worker),copy.deepcopy(cache)
    if defect == 'wrong_model': worker['contract']['model_sha256'] = contract['model_sha256']
    if defect == 'precision': worker['contract']['precision'] = 'float16'
    if defect == 'degraded': worker['fallbacks']['prediction'] = 1
    if defect == 'study_error': worker['studies'][0]['error'] = 'synthetic-test'
    if defect == 'wrong_hash': cache['output_sha256'] = '0'*64
    if defect == 'wrong_roster': ids = ids[::-1]
    if defect == 'clean_claim': cache['clean_oof'] = True
    with pytest.raises(ValueError): joint.validate_v3(worker,cache,payload,ids,contract,overlay)
