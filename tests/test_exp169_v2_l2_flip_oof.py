"""CPU-only synthetic tests for tools/exp169_v2_l2_flip_oof.py v1.0.1 (Claude opus for B, 2026-10-10)."""
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import exp169_v2_l2_flip_oof as ex

SPEC = {'depth': 64, 'slots': [['Sagittal', 1, 18], ['Sagittal', 0, 14],
                               ['Coronal', 1, 12], ['Coronal', 0, 8], ['Axial', -1, 12]]}
RANGES = ex.e124.plane_ranges(SPEC)


def test_ten_registered_sources_and_folds():
    assert [(s.tag, s.seed, s.fold) for s in ex.SOURCES] == (
        [('EXP149_L2_S42_F0', 42, 0), ('EXP149_L2_S42_F1', 42, 1)] +
        [(f'EXP157_L2_S42_F{f}', 42, f) for f in (2, 3, 4)] +
        [(f'EXP157_L2_S43_F{f}', 43, f) for f in range(5)])
    assert ex.PLAIN_TOL == 0.002 and ex.K == 36 and ex.PILOT_MAX == 40 and ex.CAP_MINUTES == 120


# ---------------------------------------------------------------- mirror: coronal/axial only

class _Base:
    k, window_policy = 6, 'within-slot-valid'

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.x, torch.zeros(12), 'uid0'


def test_mirror_touches_only_coronal_axial_windows(monkeypatch):
    before = torch.cuda.is_initialized()
    centers = [5, 20, 36, 47, 58, -1]       # sag, sag, cor, cor, axial, padding
    monkeypatch.setattr(ex.e124, 'select_centers', lambda *a, **k: list(centers))
    x = torch.randn(6, 3, 4, 5)
    x[5] = 0                                 # padding window
    corpus = SimpleNamespace(spec=SPEC, depth=64, mask=lambda uid: None)
    plain, mirrored, _y, uid = ex.CheckedPlaneWindows(_Base(x.clone()), corpus, RANGES)[0]
    assert uid == 'uid0' and torch.equal(plain, x)
    for i in (0, 1, 5):                      # sagittal and padding untouched
        assert torch.equal(mirrored[i], x[i])
    for i in (2, 3, 4):                      # coronal/axial horizontal mirror (last axis only)
        assert torch.equal(mirrored[i], torch.flip(x[i], dims=(-1,)))
        assert not torch.equal(mirrored[i], x[i])
    assert torch.cuda.is_initialized() == before


def test_plane_rules_agree_and_boundary_window_fails(monkeypatch):
    rng = np.random.default_rng(0)
    valid = [c for r in RANGES for c in range(r['start'] + 1, r['stop'] - 1)] + [-1]
    for _ in range(50):
        centers = [int(c) for c in rng.choice(valid, size=36)]
        flags = ex.window_flags(centers, RANGES, SPEC, 64)
        assert flags == ex.serving.plane_flags(centers, SPEC)
    with pytest.raises(ValueError):
        ex.window_flags([32], RANGES, SPEC, 64)  # crosses the sagittal/coronal slot boundary


# ---------------------------------------------------------------- TTA formula

def test_tta_formula_matches_sigmoid_average():
    torch.manual_seed(0)
    a = (torch.randn(5, 12) * 4).half()
    b = (torch.randn(5, 12) * 4).half()
    plain = ex.e124.served_probabilities('l2', a)
    mirrored = ex.e124.served_probabilities('l2', b)
    # EXP169 serving single-head conversion is torch.sigmoid(logits.float())
    assert np.array_equal(plain, torch.sigmoid(a.float()).numpy())
    expected = ((torch.sigmoid(a.float()) + torch.sigmoid(b.float())) / 2).double().numpy()
    got = ex.tta_mean(plain, mirrored)
    assert got.dtype == np.float64 and np.allclose(got, expected, rtol=0, atol=1e-7)


def test_tta_through_infer_pair_on_cpu():
    class Lin(torch.nn.Module):
        def forward(self, x):                    # (B,K,C,H,W) -> (B,12); asymmetric in W
            w = torch.linspace(-1, 1, x.shape[-1])
            return (x * w).mean(dim=(1, 2, 3, 4))[:, None].expand(-1, 12) * 3

    x = torch.rand(2, 4, 3, 2, 5)
    xm = torch.flip(x, dims=(-1,))
    loader = [(x, xm, torch.zeros(2, 12), ['a', 'b'])]
    plain, mirrored, ids = ex.e124.infer_pair(Lin(), loader, 'cpu', 'l2')
    m = Lin()
    expected = (torch.sigmoid(m(x).float()) + torch.sigmoid(m(xm).float())) / 2
    assert ids == ['a', 'b'] and np.allclose(ex.tta_mean(plain, mirrored), expected.numpy(), atol=1e-7)
    assert not np.allclose(plain, mirrored)


def test_tta_rejects_invalid_probabilities():
    ok = np.full((2, 12), .5)
    with pytest.raises(ValueError):
        ex.tta_mean(ok, np.full((2, 12), np.nan))
    with pytest.raises(ValueError):
        ex.tta_mean(ok, np.full((2, 12), 1.5))
    with pytest.raises(ValueError):
        ex.tta_mean(ok, np.full((3, 12), .5))


# ---------------------------------------------------------------- roster: own held-out fold only

def _synthetic_protocol(tmp_path):
    canonical = [f'u{i:02d}' for i in range(20)]
    fold_of = {u: i % 5 for i, u in enumerate(canonical)}
    (tmp_path / 'base.json').write_text(json.dumps({'train': canonical[:15], 'val': canonical[15:]}), encoding='utf-8')
    folds = {}
    for k in range(5):
        tr = [u for u in canonical if fold_of[u] != k]
        va = [u for u in canonical if fold_of[u] == k]
        (tmp_path / f'f{k}.json').write_text(json.dumps({'train': tr, 'val': va}), encoding='utf-8')
        folds[str(k)] = {'n_val': len(va), 'ordered_ids': {'path': f'f{k}.json', 'val_sha256': ex.uid_hash(va)}}
    p = {'paths': {'base_ordered_ids': 'base.json'}, 'folds': folds}
    return p, canonical, fold_of


def _spec(fold, p):
    return {'fold': fold, 'eval_folds': [fold], 'train_folds': [k for k in range(5) if k != fold],
            'eval_ids_sha256': p['folds'][str(fold)]['ordered_ids']['val_sha256'],
            'n_val': p['folds'][str(fold)]['n_val']}


def test_roster_selects_only_the_models_own_held_out_fold(tmp_path):
    p, canonical, fold_of = _synthetic_protocol(tmp_path)
    for fold in range(5):
        src = ex.Source(f'T{fold}', 42, fold, 'unused')
        ids, rec = ex.roster(src, p, _spec(fold, p), tmp_path)
        assert ids == [u for u in canonical if fold_of[u] == fold]
        assert all(fold_of[u] == fold for u in ids) and rec['held_out'] == 4 and rec['canonical_studies'] == 20
    assert ex.held_out_ids(canonical, fold_of, 3) == ['u03', 'u08', 'u13', 'u18']


def test_roster_rejects_foreign_fold_or_hash(tmp_path):
    p, _canonical, _fold_of = _synthetic_protocol(tmp_path)
    with pytest.raises(ValueError):          # checkpoint fold 1 asked to run on fold 2's spec
        ex.roster(ex.Source('T', 42, 1, 'unused'), p, _spec(2, p), tmp_path)
    bad = _spec(2, p)
    bad['eval_ids_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        ex.roster(ex.Source('T', 42, 2, 'unused'), p, bad, tmp_path)
    overlap = _spec(2, p)
    overlap['train_folds'] = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        ex.roster(ex.Source('T', 42, 2, 'unused'), p, overlap, tmp_path)


# ---------------------------------------------------------------- plain-vs-frozen gate

def test_plain_gate_fails_above_0_002():
    stored = np.full((4, 12), .4)
    rec = ex.plain_gate(stored + 0.0019, stored)
    assert rec['passed'] and rec['max_abs_diff'] == pytest.approx(0.0019) and rec['tolerance'] == 0.002
    assert ex.plain_gate(np.full((4, 12), .002), np.zeros((4, 12)))['passed']   # boundary is inclusive
    with pytest.raises(ex.GateError) as err:
        ex.plain_gate(stored + 0.0021, stored)
    assert err.value.record['max_abs_diff'] == pytest.approx(0.0021) and not err.value.record['passed']
    with pytest.raises(ex.GateError):
        ex.plain_gate(np.full((4, 12), np.nan), stored)
    with pytest.raises(ValueError):
        ex.plain_gate(stored[:3], stored)


# ---------------------------------------------------------------- strict state load

@pytest.fixture(scope='module')
def l2_state():
    model = ex.e124.build_model('resnet50', pretrained=False)
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_strict_load_accepts_exact_schema(l2_state):
    model, rec = ex.load_l2_model({'arch': 'resnet50', 'model': l2_state})
    assert not model.training and rec['tensors'] == len(l2_state) and 'All keys matched' in rec['strict_result']
    k = next(iter(l2_state))
    assert torch.equal(model.state_dict()[k], l2_state[k])
    assert not torch.cuda.is_initialized()


def test_strict_load_rejects_schema_drift(l2_state):
    key = next(iter(l2_state))
    missing = {k: v for k, v in l2_state.items() if k != key}
    extra = dict(l2_state, **{'extra.weight': torch.zeros(1)})
    reshaped = dict(l2_state, **{key: torch.zeros(1)})
    for state in (missing, extra, reshaped):
        with pytest.raises(ValueError):
            ex.load_l2_model({'arch': 'resnet50', 'model': state})
    with pytest.raises(ValueError):
        ex.load_l2_model({'arch': 'resnet50', 'model': l2_state, 'heads': ['a']})
    with pytest.raises(ValueError):
        ex.load_l2_model({'arch': 'resnet34', 'model': l2_state})


# ---------------------------------------------------------------- outputs / queue

def test_output_csv_check_exact_ids_and_columns(tmp_path):
    ids = ['a', 'b', 'c']
    ex.e124.write_predictions(tmp_path / 'x.csv', ids, np.full((3, 12), .25))
    assert ex.check_output_csv(tmp_path / 'x.csv', ids).shape == (3, 12)
    with pytest.raises(ValueError):
        ex.check_output_csv(tmp_path / 'x.csv', ['a', 'c', 'b'])
    ex.e124.write_predictions(tmp_path / 'y.csv', ids, np.full((3, 12), 1.25))
    with pytest.raises(ValueError):
        ex.check_output_csv(tmp_path / 'y.csv', ids)


def test_queue_pilot_gates_run_after_predecessor():
    pilot, run = ex.queue_entries('EXP174_OWN0949_JEV_S42')
    assert pilot['tag'] == ex.JOB_ID + '_PILOT' and pilot['after'] == 'EXP174_OWN0949_JEV_S42'
    assert pilot['command'][2:5] == ['pilot', '--tag', 'EXP149_L2_S42_F0'] and '40' in pilot['command']
    assert run['tag'] == ex.JOB_ID and run['after'] == pilot['tag'] and not run.get('continue_on_failure')
    assert run['command'][-3:] == ['run', '--device', 'cuda'] and run['command'][1].endswith('exp169_v2_l2_flip_oof.py')
    assert 'after' not in ex.queue_entries(None)[0]


# ---------------------------------------------------------------- v1.0.1: receipts, reader contract, resume

SRC_SHA = 'f' * 64


def _ids(n=6, prefix='1.2.840.'):
    return [f'{prefix}{i}' for i in range(n)]


def _write_tag(out_base, src, ids, seed=0):
    rng = np.random.default_rng(seed)
    plain = rng.uniform(.05, .95, (len(ids), 12)).astype(np.float32)
    mirrored = np.clip(plain + rng.normal(0, .01, plain.shape), 0, 1).astype(np.float32)
    gate = ex.plain_gate(plain, plain.astype(np.float64) + 1e-7)
    extra = dict(source={'files': {'model_best.pt': {'sha256': SRC_SHA}}, 'protocol': {'sha256': '0' * 64}},
                 durations=dict(authenticate_s=0., model_load_s=0., inference_s=0.))
    return ex.produce(out_base, src, ids, plain, ex.tta_mean(plain, mirrored), gate, extra, begun=0.)


def test_receipt_contract_accepted_by_marginal_reader(tmp_path):
    from tools import exp169_v2_dino_marginal as reader   # read-only import of the check-A consumer
    src, ids = ex.SOURCES[0], _ids()
    folder, receipt = _write_tag(tmp_path, src, ids)
    assert folder == tmp_path / src.tag and receipt['status'] == ex.RECEIPT_STATUS == 'ADMITTED'
    assert reader.FLIP_RECEIPT_CONTRACT == {'schema': ex.SCHEMA, 'job_id': ex.JOB_ID, 'status': ex.RECEIPT_STATUS}
    assert (reader.FLIP_FILE, reader.FLIP_PLAIN, reader.FLIP_INVENTORY, reader.FLIP_RECEIPT) == (
        ex.TTA, ex.PLAIN, ex.INVENTORY, ex.RECEIPT)
    assert reader.FLIP_ROOT == ex.OUT_REL and reader.PLAIN_CEILING == ex.PLAIN_TOL
    got = reader.check_flip_receipt(folder, src.tag, src.seed, src.fold)
    assert got == receipt and reader.e172.uid_hash(ids) == got['outputs']['ids_sha256']
    assert sorted(p.name for p in folder.iterdir()) == sorted([ex.PLAIN, ex.TTA, ex.INVENTORY, ex.RECEIPT])
    assert ex.verify_folder(folder, src, ids, SRC_SHA)['rows'] == len(ids)
    with pytest.raises(ValueError):                        # the reader checks identity
        reader.check_flip_receipt(folder, src.tag, src.seed, src.fold + 1)


def test_verify_folder_rejects_tampering(tmp_path):
    src, ids = ex.SOURCES[1], _ids()
    folder, _ = _write_tag(tmp_path, src, ids)
    with pytest.raises(ValueError):
        ex.verify_folder(folder, src, ids, 'e' * 64)       # other checkpoint
    with pytest.raises(ValueError):
        ex.verify_folder(folder, src, ids[:-1], SRC_SHA)   # other roster
    with (folder / ex.TTA).open('a', encoding='utf-8') as f:
        f.write('\n')
    with pytest.raises(ValueError):
        ex.verify_folder(folder, src, ids, SRC_SHA)        # bytes differ from inventory
    with pytest.raises(ValueError):
        _write_tag(tmp_path, src, ids)                     # never overwrites an existing tag


def test_resume_skips_verified_tag_and_stops_on_bad_folder(tmp_path):
    s0, s1, s2 = ex.SOURCES[:3]
    ids = {s.tag: _ids(prefix=f'{s.fold}.') for s in (s0, s1, s2)}
    _write_tag(tmp_path, s0, ids[s0.tag])
    (tmp_path / f'.{s1.tag}.partial-1-x').mkdir()           # stale staging dir is not an output
    produced = []

    def produce_one(src):
        produced.append(src.tag)
        return dict(tag=src.tag, status=ex.RECEIPT_STATUS)

    def verify(folder, src):
        return ex.verify_folder(folder, src, ids[src.tag], SRC_SHA)

    results = ex.resume_loop((s0, s1, s2), tmp_path, verify, produce_one, log=lambda line: None)
    assert produced == [s1.tag, s2.tag]
    assert results[0]['status'] == 'SKIPPED_VERIFIED' and results[0]['tag'] == s0.tag and 'ids' not in results[0]
    (tmp_path / s0.tag / ex.PLAIN).write_text('x', encoding='utf-8')
    produced.clear()
    with pytest.raises(ValueError):
        ex.resume_loop((s0, s1), tmp_path, verify, produce_one, log=lambda line: None)
    assert produced == []                                    # an unverifiable folder stops the job


# ---------------------------------------------------------------- v1.0.1: pilot STOP -> exit 3 -> queue stops

def test_pilot_status_and_exit_codes(monkeypatch):
    assert ex.pilot_status(120.0, True) == 'PASS'
    assert ex.pilot_status(120.1, True) == 'STOP_PROJECTION_EXCEEDS_CAP'
    assert ex.pilot_status(10.0, False) == 'STOP_PLAIN_SUBSET_GATE'
    for status, code in (('STOP_PROJECTION_EXCEEDS_CAP', 3), ('STOP_PLAIN_SUBSET_GATE', 3), ('PASS', None)):
        monkeypatch.setattr(ex, 'pilot', lambda *a, _s=status, **k: {'status': _s})
        if code is None:
            assert ex.main(['pilot']) is None
        else:
            with pytest.raises(SystemExit) as err:
                ex.main(['pilot'])
            assert err.value.code == 3

    def broken(*a, **k):
        raise ValueError('GPU free memory below the 4 GiB rule')
    monkeypatch.setattr(ex, 'pilot', broken)
    with pytest.raises(ValueError):                          # any error -> non-zero interpreter exit
        ex.main(['pilot'])


@pytest.mark.parametrize('pilot_code,expected_calls,plan_rc', [(3, 1, 1), (1, 1, 1), (0, 2, 0)])
def test_gpu_queue_starts_run_only_after_pilot_success(tmp_path, monkeypatch, pilot_code, expected_calls, plan_rc):
    from tools import gpu_queue as gq
    job_src = (ex.ROOT / 'tools/gpu_job.py').read_text(encoding='utf-8')
    assert "'SUCCESS' if code==0 else 'FAILED'" in job_src and 'sys.exit(code)' in job_src
    monkeypatch.setattr(gq, 'LOGS', tmp_path / 'logs')
    calls = []

    def fake_run(cmd, **kw):                                 # stands in for `gpu_job.py --tag T --wait -- ...`
        tag = cmd[cmd.index('--tag') + 1]
        calls.append(tag)
        code = pilot_code if tag.endswith('_PILOT') else 0
        d = tmp_path / 'logs' / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / 'job.json').write_text('{}', encoding='utf-8')
        (d / ('SUCCESS' if code == 0 else 'FAILED')).write_text(f'exit {code}', encoding='utf-8')
        return subprocess.CompletedProcess(cmd, code)

    monkeypatch.setattr(gq.subprocess, 'run', fake_run)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(ex.queue_entries(None)), encoding='utf-8')
    assert gq.run_plan(plan) == plan_rc
    assert calls == [ex.JOB_ID + '_PILOT', ex.JOB_ID][:expected_calls]


# ---------------------------------------------------------------- v1.0.1: lane registration and pilot-first

def test_lane_registration_required_and_bound(tmp_path):
    with pytest.raises(ValueError):
        ex.require_lane(tmp_path)
    path = tmp_path / ex.LANE_REL
    ex.write_json_x(path, ex.lane_body('Sat Oct 10 20:50:00 KST 2026'))
    lane, lane_sha = ex.require_lane(tmp_path, 'NVIDIA GeForce RTX 4070 Ti SUPER')
    assert lane['cap_gpu_minutes'] == 120 and lane['pilot_first'] is True and lane_sha == ex.sha(path)
    with pytest.raises(ValueError):
        ex.require_lane(tmp_path, 'NVIDIA GeForce RTX 3090')
    body = ex.lane_body('x')
    body['cap_gpu_minutes'] = 240
    bad = tmp_path / 'bad'
    ex.write_json_x(bad / ex.LANE_REL, body)
    with pytest.raises(ValueError):
        ex.require_lane(bad)


def test_run_requires_pass_pilot_bound_to_lane(tmp_path):
    lane_sha = 'a' * 64
    with pytest.raises(ValueError):
        ex.require_pilot_pass(tmp_path, lane_sha)

    def pilot_file(name, status, projected=20.0, sha=lane_sha, version=ex.VERSION):
        ex.write_json_x(tmp_path / '_pilot' / name, dict(
            schema=ex.PILOT_SCHEMA, job_id=ex.JOB_ID, status=status, tool={'version': version},
            lane_sha256=sha, projection={'projected_minutes': projected}))

    pilot_file('pilot-a-1.json', 'STOP_PROJECTION_EXCEEDS_CAP', 130.0)
    pilot_file('pilot-a-2.json', 'PASS', sha='b' * 64)
    pilot_file('pilot-a-3.json', 'PASS', version='1.0.0')
    pilot_file('pilot-a-4.json', 'PASS', projected=121.0)
    with pytest.raises(ValueError):
        ex.require_pilot_pass(tmp_path, lane_sha)
    pilot_file('pilot-a-5.json', 'PASS')
    assert ex.require_pilot_pass(tmp_path, lane_sha).name == 'pilot-a-5.json'
