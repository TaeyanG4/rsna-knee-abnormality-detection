"""CPU-only regression for versioned OAI lineage/readout/executable gates."""
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pytest

from tools import hb166_oai_readout_v2 as m


@pytest.fixture
def fixture():
    ids = [str(i) for i in range(40)]
    y = np.tile((np.arange(40) % 2)[:, None], (1, 12)).astype(float)
    y[:, [1, 10, 11]] = np.nan
    good = np.tile((np.arange(40) % 2)[:, None], (1, 12)).astype(np.float32)
    pp = {k: good.copy() if k.startswith('multiview') else np.full((40, 12), .5, np.float32)
          for k in m.KEYS}
    return ids, y, pp


def test_memory_gate_boundaries_and_invalid():
    assert m.memory_gate(2 << 30, 12 << 30, 16 << 30)['PASS']
    for values in [(2 << 30, (12 << 30)+1, 17 << 30), ((2 << 30)+1, 12 << 30, 16 << 30),
                   (-1, 0, 1), (0, 1, 0), (float('nan'), 1, 2), (0, float('inf'), 2)]:
        with pytest.raises(ValueError): m.memory_gate(*values)


def test_qualification_reserve_only_then_measured_peak():
    assert m.qualification_headroom(3 << 30, 16 << 30)['PASS']
    assert m.memory_gate(3 << 30, int(10.9 * (1 << 30)), 16 << 30)['PASS']
    for baseline, total in [(15 << 30, 16 << 30), (-1, 1), (0, 0), (float('nan'), 1)]:
        with pytest.raises(ValueError): m.qualification_headroom(baseline, total)


def test_qualification_worker_uses_measured_peak(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'WORK', tmp_path)
    monkeypatch.setattr(m, 'scientific_gate', lambda _: None)
    monkeypatch.setattr(m, 'check_review', lambda _: None)
    monkeypatch.setattr(m, 'require_queue_context', lambda: {'run_id': 'test'})
    monkeypatch.setattr(m, 'panel_ids', lambda: ['a', 'b', 'c', 'd'])
    monkeypatch.setenv('HB155_TOKEN', 'test'); monkeypatch.setenv('HB155_HEARTBEAT', 'test')
    key = 'multiview-s42'; peak = int(10.9 * (1 << 30))
    r = dict(members=[dict(key=key)])
    baseline_path = tmp_path/'baselines'/('qualify-'+key+'.json')
    m.io.atomic_json(baseline_path, dict(registration_sha256=m.digest(r), reservation={'run_id': 'test'},
        key=key, mode='qualify', baseline_bytes=3 << 30, total_bytes=16 << 30))
    calls = []
    def measured(entry, ids, *, device):
        calls.append((entry['key'], len(ids), device))
        return np.full((4, 12), .5, np.float32), dict(peak_reserved_bytes=peak)
    monkeypatch.setattr(m, 'inference', measured)
    assert m.worker(r, key, 'qualify', baseline_path)['status'] == 'WORKER_COMPLETE'
    assert calls == [(key, 4, 'cuda')]
    assert m.read(tmp_path/'qualify'/(key+'.json'))['memory_gate']['peak_reserved_bytes'] == peak


@pytest.mark.parametrize('kind', ['native_original', 'native_completion', 'crop_completion'])
def test_frozen_cache_completion_binding(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(m, 'BASE', tmp_path)
    native = tmp_path/'native'; native.mkdir()
    crop = tmp_path/'crop'; crop.mkdir()
    monkeypatch.setattr(m, 'NATIVE', native); monkeypatch.setattr(m, 'CROP', crop)
    (native/'manifest.json').write_text('native'); (crop/'manifest.json').write_text('crop')
    old = dict(native_manifest_sha256=m.sha(native/'manifest.json'))
    completion = dict(native_manifest_sha256=old['native_manifest_sha256'],
                      crop_manifest_sha256=m.sha(crop/'manifest.json'))
    path = tmp_path/'completion-evidence.json'; m.io.atomic_json(path, completion)
    assert m.frozen_cache_binding(old)['original_completion_evidence_sha256'] == m.sha(path)
    if kind == 'native_original': old['native_manifest_sha256'] = '0'*64
    elif kind == 'native_completion': completion['native_manifest_sha256'] = '0'*64
    else: completion['crop_manifest_sha256'] = '0'*64
    m.io.atomic_json(path, completion)
    with pytest.raises(ValueError, match='frozen .* manifest'): m.frozen_cache_binding(old)


def test_ineligible_stops_before_review_device_queue(monkeypatch):
    r = dict(lineage=dict(eligible=False, verdict='INELIGIBLE', members=[]))
    monkeypatch.setattr(m, 'device_baseline', lambda: pytest.fail('device touched'))
    monkeypatch.setattr(m, 'check_review', lambda _: pytest.fail('later gate reached'))
    with pytest.raises(ValueError, match='INELIGIBLE'): m.serial_run(r)
    with pytest.raises(ValueError, match='INELIGIBLE'): m.worker(r, 'multiview-s42', 'full', 'none')


def test_review_binding_and_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path); monkeypatch.setattr(m, 'WORK', tmp_path)
    r = {'version': 'test'}
    with pytest.raises(ValueError, match='missing'): m.check_review(r)
    p = tmp_path/'REVIEW.md'; p.write_text('independent PASS')
    smoke = tmp_path/'cpu-smoke.json'; smoke.write_text('{}')
    v = dict(verdict='PASS', reviewer='reviewer', scope='H-B-166 OAI4070 executable readout',
             registration_sha256=m.digest(r), report='REVIEW.md', report_sha256=m.sha(p),
             cpu_smoke_sha256=m.sha(smoke))
    m.io.atomic_json(tmp_path/'gpu-review.json', v); m.check_review(r)
    with pytest.raises(ValueError, match='binding'): m.check_review({'version': 'other'})
    smoke.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='smoke'): m.check_review(r)
    smoke.write_text('{}'); p.write_text('changed')
    with pytest.raises(ValueError, match='report'): m.check_review(r)


def test_fold_pooled_probability_mean_and_no_ineligible_pass(fixture):
    ids, y, pp = fixture
    pp['multiview-s44'] *= .2
    pp['multiview-s45'] = 1 - pp['multiview-s45']
    out = m.compute_readout(ids, pp, y)
    assert list(out['views']) == ['fold2', 'fold3', 'pooled']
    for name, seeds in [('fold2', (42, 43)), ('fold3', (44, 45)), ('pooled', (42, 43, 44, 45))]:
        means = np.stack([pp[f'multiview-s{s}'] for s in seeds]).mean(0, dtype=np.float32)
        row = out['views'][name]
        assert row['probability_mean_sha256']['multiview'] == hashlib.sha256(means.tobytes()).hexdigest()
        assert row['descriptive'] == (name == 'pooled')
        assert not row['weak4']['PASS'] and not row['weak4']['scientific_PASS']
        assert not row['mapped_nine']['PASS']
        assert not row['H2']['eligible'] and 'Contusion' in row['H2']['reason']
        assert row['blend_proxy']['status'] == 'UNAVAILABLE'
    assert out['views']['fold2']['weak4']['point_threshold_met']
    eligible = m.compute_readout(ids, pp, y, eligible=True)
    assert eligible['views']['fold2']['weak4']['PASS']
    assert not eligible['views']['pooled']['weak4']['PASS']


def test_weak_unknown_mask_and_threshold(fixture):
    ids, y, pp = fixture
    y[:, 3] = np.nan
    row = m.compute_readout(ids, pp, y, eligible=True)['views']['fold2']
    assert row['weak4']['value'] is None and not row['weak4']['PASS']
    assert row['weak4']['threshold'] == .002
    y[:, 10] = 0
    with pytest.raises(ValueError, match='Contusion'): m.compute_readout(ids, pp, y)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'nan', 'range', 'shape'])
def test_invalid_readout_inputs(fixture, kind):
    ids, y, pp = fixture
    if kind == 'missing': pp.pop(m.KEYS[-1])
    elif kind == 'duplicate': ids[-1] = ids[0]
    elif kind == 'nan': pp[m.KEYS[0]][0, 0] = np.nan
    elif kind == 'range': pp[m.KEYS[0]][0, 0] = 2
    else: pp[m.KEYS[0]] = pp[m.KEYS[0]][:-1]
    with pytest.raises(ValueError): m.compute_readout(ids, pp, y)


def test_h166_add_exact_family_formula_and_conditional_readout(fixture):
    from tools.multi_lens_readout import rank
    ids, y, pp = fixture
    rng = np.random.default_rng(123)
    p = {k: rng.random((40, 12), dtype=np.float32) for k in m.PARTNERS}
    partners = dict(members=p)
    a, c = pp[m.KEYS[0]], pp[m.KEYS[1]]
    fixed = [rank(np.stack([p['r50-s42'], p['r50-s43']]).mean(0, dtype=np.float32)),
             rank(p['c4-s42']), rank(np.stack([p['t-s42'], p['t-s43']]).mean(0, dtype=np.float32))]
    qa, qc = m.proxy_probabilities(a, c, partners)['H166_add']
    np.testing.assert_array_equal(qa, np.mean([rank(a), *fixed], axis=0))
    np.testing.assert_array_equal(qc, np.mean([rank(c), *fixed], axis=0))
    out = m.compute_readout(ids, pp, y, partners=partners)
    assert all(not v['blend_proxy']['H166_add']['weak4']['PASS'] for v in out['views'].values())


def test_exact_roster_npz_alignment(tmp_path):
    ids = ['a', 'b']; p = np.full((2, 12), .3, np.float32); p[1] = .8
    path = tmp_path/'p.npz'; m.atomic_npz(path, ids[::-1], p[::-1])
    np.testing.assert_array_equal(m.aligned_predictions(path, ids), p)
    for bad in [['a', 'a'], ['a', 'c']]:
        m.atomic_npz(path, bad, p)
        with pytest.raises(ValueError, match='roster'): m.aligned_predictions(path, ids)


def make_partners(tmp_path):
    from tools.hb166_build_v1 import L6_PARTNERS
    ids = ['a', 'b']; members = {}
    for key in m.PARTNERS:
        path = tmp_path/(key+'.npz'); rp = path.with_suffix('.json')
        m.atomic_npz(path, ids, np.full((2, 12), .5, np.float32))
        family, seed = key.rsplit('-s', 1); family = dict(r50='R50', c4='C4', t='T')[family]
        source, sourcehash = L6_PARTNERS[family][seed]
        rec = dict(complete=True, prediction_sha256=m.sha(path), ordered_ids_sha256=m.digest(ids),
                   oai_eligible=True, family=family, seed=seed, source_oof=source,
                   source_oof_sha256=sourcehash, checkpoint_sha256='a'*64)
        m.io.atomic_json(rp, rec)
        members[key] = dict(path=path.name, sha256=m.sha(path), receipt=rp.name, receipt_sha256=m.sha(rp),
            source_oof=source, source_oof_sha256=sourcehash, checkpoint_sha256='a'*64)
    return ids, dict(status='AVAILABLE', members=members)


@pytest.mark.parametrize('kind', ['partial', 'hash', 'receipt', 'family', 'checkpoint', 'source'])
def test_partner_authentication_fail_closed(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    ids, spec = make_partners(tmp_path)
    assert len(m.load_partners(spec, ids)['members']) == 5
    x = spec['members'][m.PARTNERS[0]]
    if kind == 'partial': spec['members'].pop(m.PARTNERS[-1])
    elif kind == 'hash': x['sha256'] = '0'*64
    elif kind == 'checkpoint': x['checkpoint_sha256'] = 'b'*64
    elif kind == 'source': x['source_oof_sha256'] = 'b'*64
    else:
        rp = tmp_path/x['receipt']; rec = m.read(rp)
        if kind == 'receipt': rec['oai_eligible'] = False
        else: rec['family'] = 'unrelated'
        m.io.atomic_json(rp, rec); x['receipt_sha256'] = m.sha(rp)
    with pytest.raises(ValueError): m.load_partners(spec, ids)


def test_queue_plan_serial_and_direct_call_rejected():
    from tools.gpu_queue import validate_plan
    assert validate_plan(m.queue_plan()) == m.queue_plan()
    assert len(m.queue_plan()) == 1 and not m.queue_plan()[0]['continue_on_failure']
    with pytest.raises(ValueError, match='ancestor'): m.require_queue_context()


def test_serial_failure_stops_before_next_member(tmp_path, monkeypatch):
    from tools import hb155_watchdog_v1 as watchdog
    monkeypatch.setattr(m, 'WORK', tmp_path)
    monkeypatch.setattr(m, 'scientific_gate', lambda _: None)
    monkeypatch.setattr(m, 'check_review', lambda _: None)
    monkeypatch.setattr(m, 'require_queue_context', lambda: {'run_id': 'test'})
    monkeypatch.setattr(m, 'device_baseline', lambda: dict(baseline_bytes=3 << 30, total_bytes=16 << 30))
    calls = []
    def fail(command, attempt, **kw): calls.append(command); assert kw['total'] is None; return 7
    monkeypatch.setattr(watchdog, 'supervise', fail)
    r = dict(members=[dict(seed=42, key='multiview-s42', arm='multiview'),
                      dict(seed=42, key='control-s42', arm='control')])
    with pytest.raises(ValueError, match='serial queue stopped'): m.serial_run(r)
    assert len(calls) == 1 and m.read(tmp_path/'execution.json')['status'] == 'IN_FLIGHT'
    with pytest.raises(ValueError, match='attempt exists'): m.serial_run(r)


def test_qualification_current_baseline_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'WORK', tmp_path)
    r = dict(members=[dict(key=f'{arm}-s42', checkpoint_sha256='ck', config_sha256='cfg')
                      for arm in ('multiview', 'control')])
    for e in r['members']:
        v = dict(registration_sha256=m.digest(r), recipe=m.RECIPE, studies=4, mode='qualify',
                 complete=True, device='cuda', checkpoint_sha256='ck', config_sha256='cfg',
                 memory_gate=m.memory_gate(1 << 30, 12 << 30, 16 << 30),
                 peak_reserved_bytes=12 << 30, seconds=10, batch_seconds=[8])
        m.io.atomic_json(tmp_path/'qualify'/(e['key']+'.json'), v)
    assert m.enforce_memory_receipts(r) == dict(multiview=.4, control=.4)
    with pytest.raises(ValueError, match='desktop'):
        m.enforce_memory_receipts(r, baseline=dict(baseline_bytes=3 << 30, total_bytes=16 << 30))
    v['recipe'] = dict(m.RECIPE, k_per_view=24)
    m.io.atomic_json(tmp_path/'qualify'/'control-s42.json', v)
    with pytest.raises(ValueError, match='qualification'): m.enforce_memory_receipts(r)


def test_no_admit_cli_and_cpu_isolation():
    r = subprocess.run([sys.executable, '-B', '-m', 'tools.hb166_oai_readout_v2', 'admit'],
                       capture_output=True, text=True)
    assert r.returncode == 2 and 'invalid choice' in r.stderr
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
