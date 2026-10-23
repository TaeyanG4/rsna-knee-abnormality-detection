"""Synthetic CPU checks only; never reads H200 run1 or real label values."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import rankdata
from tools import hb200_k3_gold58_v1 as m
from tools import c85r7_rco_overlay_v1 as overlay
from tools import l1_rco_k2_worker_v101 as worker


def payload(ids, values):
    buf = io.StringIO(newline=''); writer = csv.writer(buf)
    writer.writerow([overlay.UID, *overlay.LABELS])
    writer.writerows([u, *[format(float(v), '.17g') for v in row]] for u, row in zip(ids, values))
    return buf.getvalue().encode()


@pytest.fixture(scope='module')
def helpers():
    return m.dependencies()[0]


@pytest.mark.parametrize('state,kind', [('running', None), ('FAILED', 'bound'), ('unknown', None), ('SUCCESS', None)])
def test_nonterminal_blocks_all_output_reads(monkeypatch, state, kind):
    jr = SimpleNamespace(job_status=lambda *a, **k: dict(state=state, receipt_kind=kind))
    monkeypatch.setattr(m, 'dependencies', lambda: (None, None, None, jr))
    monkeypatch.setattr(m, 'read', lambda p: pytest.fail('read reached before completion barrier'))
    with pytest.raises(ValueError, match='no run-output reads'):
        m.run()


def completed_fixture():
    ids = ['synthetic-%02d' % i for i in range(58)]
    p = payload(ids, np.full((58, 12), .5))
    inventory = m.read(m.HERE / 'input-inventory.json')  # static launch metadata only
    child = dict(pid=123456, create_time=1.); launch = dict(pid=123455, create_time=1.)
    rec = dict(status='COMPLETE', version='1.0.1-rcof0', kernel_id='taeyangg4/knee-l1-rcof0-v1',
        contract=worker.CONTRACT, manifest_sha256=overlay.MANIFEST_SHA,
        template_sha256=inventory['cohort']['template_sha256'], output_sha256=hashlib.sha256(p).hexdigest(),
        study_count=58, prediction_count=58, fast_path=False, placeholder=False, fallbacks=dict(prediction=0),
        codecs=dict(status='ACTIVE', files=58, syntaxes={'synthetic':dict(decoded=True)}),
        studies=[dict(index=i, uid_sha256=hashlib.sha256(u.encode()).hexdigest(), error=None, prediction_fallback=0) for i,u in enumerate(ids)],
        roster_sha256=hashlib.sha256((json.dumps(ids,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()).hexdigest(),
        environment=dict(pid=child['pid'], device='cuda:0', tf32=False, autocast=False), anatomy=dict(dtype='torch.float32'),
        peak_reserved_bytes=2**30, memory=dict(used_before_bytes=2**30,total_bytes=16*2**30))
    records = dict(worker=rec, job=dict(tag=m.TAG,child_identity=launch),
        prelaunch=dict(tag=m.TAG,job_identity=launch,inventory_sha256=m.INVENTORY_SHA,
            launcher_sha256=m.sha(m.HERE/'launch-k1-gold58.py'),
            supervisor_sha256=inventory['pins']['tools/l1_rco_k2_local_v103.py']['sha256'],
            worker_command=inventory['worker_command'],memory=dict(desktop_bytes=2**30,total_bytes=16*2**30)),
        process=dict(schema='k2-local-process/v1',version='1.0.3',exit_code=0,reason='completed',survivors=[],child=child,owned_descendants=[]),
        cache=dict(status='COMPLETE_PENDING_INDEPENDENT_RESULT_REVIEW',studies=58,tag=m.TAG,
            exposure='EXPOSED_GOLD58',clean_oof=False,qualified_for_submission=False,
            worker_receipt_sha256='synthetic-receipt',output_sha256=hashlib.sha256(p).hexdigest()))
    return records, dict(worker='synthetic-receipt'), p, ids, inventory, dict(contract=worker.CONTRACT)


def test_completed_receipts_accept_exact_synthetic(helpers):
    m.validate_completed(*completed_fixture(), overlay, SimpleNamespace(liveness=lambda x:'dead'))


@pytest.mark.parametrize('defect', ['survivor','incomplete','source','hash','order','fallback','precision','memory','nan'])
def test_completed_receipts_reject_material_defects(helpers, defect):
    args = list(completed_fixture()); r = args[0]
    if defect == 'survivor': r['process']['survivors'] = [dict(pid=1)]
    elif defect == 'incomplete': r['cache']['status'] = 'RUNNING'
    elif defect == 'source': r['prelaunch']['launcher_sha256'] = 'wrong'
    elif defect == 'hash': r['cache']['worker_receipt_sha256'] = 'wrong'
    elif defect == 'order': r['worker']['studies'][0]['uid_sha256'] = 'wrong'
    elif defect == 'fallback': r['worker']['fallbacks']['prediction'] = 1
    elif defect == 'precision': r['worker']['environment']['autocast'] = True
    elif defect == 'memory': r['worker']['peak_reserved_bytes'] = 13*2**30
    else:
        a = np.full((58,12),.5); a[0,0] = np.nan; args[2] = payload(args[3],a)
        h = hashlib.sha256(args[2]).hexdigest(); r['worker']['output_sha256'] = r['cache']['output_sha256'] = h
    with pytest.raises(ValueError):
        m.validate_completed(*args, overlay, SimpleNamespace(liveness=lambda x:'dead'))


def test_unknown_child_is_not_clean_completion(helpers):
    with pytest.raises(ValueError, match='remains alive or unknown'):
        m.validate_completed(*completed_fixture(), overlay, SimpleNamespace(liveness=lambda x:'unknown'))


def parent_fixture(bridge):
    ids = ['synthetic-%02d' % i for i in range(58)]
    common = dict(ids=np.array(ids),targets=np.array(overlay.LABELS))
    for name in ('R','R224','EXP143v2','EXP155','Q_A5q'):
        common[name] = np.full((58,12),.5); common['failed_'+name] = np.zeros(58,bool)
    leaves = dict(ids=np.array(ids),targets=np.array(overlay.LABELS))
    leaves.update({f:np.full((10 if f=='l2' else 5,58,12),.5,np.float32) for f in bridge.FAMILIES})
    entries = [dict(family=f,seed=s,fold=k,source_sha256=f'{i:064x}',sha256=f'{i+100:064x}') for i,(f,s,k) in enumerate(bridge.EXPECTED)]
    reg = dict(ids=ids,q_entries=entries)
    inv = dict(cohort=dict(ordered_uid_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest()))
    return common, leaves, reg, entries, inv


@pytest.mark.parametrize('defect', ['mask','nan','duplicate','target','leaf_precision','member_identity'])
def test_parent_cache_rejects_incompatible_inputs(helpers, defect):
    args = list(parent_fixture(helpers)); common, leaves, reg, entries, _ = args
    if defect == 'mask': common['failed_R'][0] = True
    elif defect == 'nan': common['R'][0,0] = np.nan
    elif defect == 'duplicate': common['ids'][0] = common['ids'][1]
    elif defect == 'target': common['targets'] = common['targets'][::-1]
    elif defect == 'leaf_precision': leaves['l2'] = leaves['l2'].astype(np.float64)
    else: reg['q_entries'] = copy.deepcopy(entries); reg['q_entries'][0]['sha256'] = 'f'*64
    with pytest.raises(ValueError):
        m.parent_arrays(*args, helpers, overlay)


def test_parent_exact_synthetic(helpers):
    ids, leaves = m.parent_arrays(*parent_fixture(helpers), helpers, overlay)
    assert len(ids) == 58 and set(leaves) == set(helpers.FAMILIES)


def test_fixed_overlay_float32_and_ties():
    ids = ['synthetic-%02d' % i for i in range(58)]
    a = np.tile(np.linspace(.1,.9,58)[:,None],(1,12)); a[1] = a[0]+1e-10
    b = a[::-1].copy(); b[1] = b[0]; b[4:8] = .4
    out = overlay.blend(payload(ids,a),payload(ids,b),ids)
    _, got = overlay.table(out)
    rank = lambda x:rankdata(x,axis=0,method='average')/58
    expected = rank(.9*rank(a.astype(np.float32))+.1*rank(b.astype(np.float32))).astype(np.float32)
    np.testing.assert_array_equal(got.astype(np.float32), expected)
    assert got[0,0] == got[1,0]


def test_true_spearman_and_constant_columns():
    a = np.tile(np.linspace(0,1,58)[:,None],(1,12)); b = a**10
    weights = np.ones((2,58),int); weights[1,::2] = 2; weights[1,1::2] = 0
    out = m.spearman_draws(a,b,weights)
    np.testing.assert_allclose(out,1.,atol=1e-14)
    assert np.corrcoef(a[:,0],b[:,0])[0,1] < .9
    b[:,0] = .5
    assert np.isnan(m.spearman_draws(a,b,weights)[:,0]).all()


def test_paired_auc_and_fixed_invalid_draws(helpers, monkeypatch):
    gold = helpers.gold; ids = ['synthetic-%02d' % i for i in range(58)]
    y = np.tile((np.arange(58)%2)[:,None],(1,12)).astype(float)
    weights = np.zeros((301,58),int); weights[0] = 1; weights[1:,0] = 58
    monkeypatch.setattr(gold,'bootstrap_weights',lambda _:weights)
    result = m.statistics(ids,y,y,1-y,y,gold)
    for block in ('macro12','weak_four'):
        assert result['delta_k3_minus_h189'][block]['observed'] == -1.
        assert result['delta_k3_minus_h189'][block]['valid_draws'] == 0
        assert result['delta_k3_minus_h189'][block]['ci95'] is None


def test_pin_change_and_live_pin_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path); monkeypatch.setattr(m,'LIVE',tmp_path/'run1')
    p = tmp_path/'static'; p.write_bytes(b'original')
    pins = {'static':dict(bytes=8,sha256=hashlib.sha256(b'original').hexdigest())}
    m.verify_pins(pins); p.write_bytes(b'changed!')
    with pytest.raises(ValueError,match='changed input/source'):m.verify_pins(pins)
    with pytest.raises(ValueError,match='inside live output'):m.verify_pins({'run1/file':pins['static']})
