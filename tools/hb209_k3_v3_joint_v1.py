"""H209 fixed v3-K3 joint using authenticated completed caches, CPU only."""
from pathlib import Path
import hashlib, json, sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import hb200_k3_gold58_v1 as old
HERE = old.HERE / 'h209-joint'
V3 = old.HERE / 'h209-v3-gold58-v3'
TAG = 'H209_K1V3_GOLD58_V3'
MODEL = '6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5'
OUTPUT = '74277df8607eb1da7b42e0924bcba76bd38cdd679450cf8f63ad129a9ad3668a'
VERSION = '1.0.0'


def validate_v3(worker, cache, payload, ids, v2_contract, overlay):
    need = old.need
    actual_ids, values = overlay.table(payload, ids)
    need(actual_ids == ids and len(ids) == len(set(ids)) == 58, 'exact ordered58 roster')
    expected_contract = dict(v2_contract, model_sha256=MODEL)
    need(worker['contract'] == expected_contract, 'only checkpoint may change')
    need(hashlib.sha256(payload).hexdigest() == OUTPUT == worker['output_sha256'] == cache['output_sha256'], 'v3 output pin')
    need(worker['status'] == 'COMPLETE' and worker['study_count'] == worker['prediction_count'] == 58, 'complete v3 worker')
    need(worker['fallbacks']['prediction'] == 0 and len(worker['studies']) == 58 and
         all(s['error'] is None and s['prediction_fallback'] == 0 for s in worker['studies']), 'no failed v3 studies')
    need(worker['codecs']['status'] == 'ACTIVE', 'active codecs')
    need(cache['tag'] == TAG and cache['clean_oof'] is False and cache['qualified_for_submission'] is False, 'cache scope')
    digest = hashlib.sha256((json.dumps(ids, sort_keys=True, indent=2, ensure_ascii=False)+'\n').encode()).hexdigest()
    need(worker['roster_sha256'] == digest, 'receipt roster')
    return values


def run():
    old.need(not (HERE/'RESULT.json').exists(), 'do not repeat completed comparison')
    bridge, overlay, io, jr = old.dependencies()
    state = jr.job_status(ROOT/'artifacts/logs', TAG, tail=False)
    old.need(state.get('state') == 'SUCCESS' and state.get('receipt_kind') == 'bound', 'v3 terminal bound SUCCESS required')
    frozen, inventory = old.frozen_inputs()
    old.k3_source(overlay, bridge.unpack_parent())
    ids, parent_payload, parent = old.parent_replay(bridge, overlay, inventory)
    v2_payload, v2_receipts = old.completed_rco(ids, inventory, overlay, jr)
    v2_joint_payload = overlay.blend(parent_payload, v2_payload, ids)
    expected = old.read(old.OUT/'RESULT.json')['csv_sha256']
    for name, data in [('H189',parent_payload),('RCO',v2_payload),('K3',v2_joint_payload)]:
        old.need(hashlib.sha256(data).hexdigest() == expected[name], 'original control CSV mismatch: '+name)
    cache = old.read(V3/'run1/cache-complete.json')
    worker_path = V3/'run1/worker-receipt.json'
    old.need(old.sha(worker_path) == cache['worker_receipt_sha256'], 'v3 worker/cache binding')
    worker = old.read(worker_path)
    v3_payload = (V3/'run1/predictions.csv').read_bytes()
    v2_contract = old.read(old.LIVE/'worker-receipt.json')['contract']
    v3 = validate_v3(worker, cache, v3_payload, ids, v2_contract, overlay)
    v3_joint_payload = overlay.blend(parent_payload, v3_payload, ids)
    _, c2 = overlay.table(v2_joint_payload, ids)
    _, c3 = overlay.table(v3_joint_payload, ids)
    import numpy as np
    oracle = overlay.ranks(.90*overlay.ranks(parent)+.10*overlay.ranks(v3)).astype(np.float32)
    old.need(np.array_equal(oracle, c3.astype(np.float32)), 'fixed deployed formula mismatch')
    label_ids, labels = bridge.gold.gold()
    labels = bridge.align(label_ids, labels, ids)
    weights = bridge.gold.bootstrap_weights(ids)
    old.need(weights.shape == (301,58), 'registered paired draws')
    auc = bridge.gold.auc_table(dict(H189=parent,K3v2=c2,K3v3=c3),labels,weights)
    metrics = {n:bridge.gold.block(auc['K3v3']-auc[ref]) for n,ref in
               [('v3_minus_v2','K3v2'),('v3_minus_H189','H189')]}
    inputs = [Path(__file__), HERE/'SPEC.md', V3/'B-RESULT-CHECK.json',
        worker_path, V3/'run1/cache-complete.json', V3/'run1/predictions.csv',
        old.OUT/'RESULT.json', old.OUT/'PREPARATION.json']
    out = dict(version=VERSION,status='COMPLETE_EXPOSED_B_DIRECT_CHECK_REQUIRED',
        formula=overlay.FORMULA,studies=58,targets=overlay.LABELS,
        source_inputs={old.rel(p):old.sha(p) for p in inputs},
        original_control_hashes_reproduced=expected,
        v3_joint_sha256=hashlib.sha256(v3_joint_payload).hexdigest(),
        v3_raw_sha256=OUTPUT,model_sha256=MODEL,
        bootstrap=dict(draws=300,seed=20261003,paired=True),metrics=metrics,
        clean_oof=False,exposure='EXPOSED_GOLD58',qualified_for_submission=False,
        independent_review=False, limitations=['Small repeatedly exposed panel',
        'Conditional intervals omit training/selection uncertainty','No hidden or LB forecast'])
    io.atomic_json(HERE/'RESULT.json',out)
    print(json.dumps(dict(status=out['status'],contrasts={k:v['macro12'] for k,v in metrics.items()})))

if __name__ == '__main__': run()
