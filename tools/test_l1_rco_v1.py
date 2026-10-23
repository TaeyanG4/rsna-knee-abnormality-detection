"""CPU safety and regression tests for L1 RCO Stage 0 v1.0.0."""
import ast
import copy
import os
from pathlib import Path
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pytest
from tools import l1_rco_stage0_v1 as s
from tools import l1_rco_runtime_v1 as r
from tools import l1_rco_cache_v1 as c

def test_pins_and_full_roster():
    s.verify_pins()
    reg = s.read(s.ROOT / s.REG)
    hashes = s.validate_roster(reg)
    assert len(hashes['train']) == 64
    bad = copy.deepcopy(reg); bad['train'][0] = bad['val'][0]
    with pytest.raises(ValueError): s.validate_roster(bad)
    bad = copy.deepcopy(reg); bad['components'][bad['train'][0]] = bad['components'][bad['val'][0]]
    with pytest.raises(ValueError): s.validate_roster(bad)

def test_holdout_targets_never_parsed_and_neutral():
    reg = {'train': ['train'], 'holdout': ['val']}
    rows = [{s.UID: 'train', **{l: '.25' for l in s.LABELS}},
            {s.UID: 'val', **{l: 'SECRET-NONNUMERIC' for l in s.LABELS}},
            {s.UID: 'outside', **{l: 'DO-NOT-PARSE' for l in s.LABELS}}]
    a = s.target_rows(reg, rows)
    rows[1].update({l: 'nan' for l in s.LABELS})
    assert s.target_rows(reg, rows) == a
    assert all(a[1][l + '|prior'] == .5 for l in s.LABELS)
    with pytest.raises(ValueError): s.target_rows(reg, rows + [rows[0]])
    rows[0]['ACL'] = 'nan'
    with pytest.raises(ValueError): s.target_rows(reg, rows)

def test_exact_trainer_patch_and_fold_arguments():
    source = (s.ROOT / s.SOURCE / 'compact_train.py').read_text(encoding='utf-8')
    derived = s.derive_trainer(source)
    ast.parse(derived)
    assert 'bias0 = torch.logit' not in derived
    assert 'bias0 = torch.zeros(C, device=dev)' in derived
    assert 'if l1.steps >= 200:' in derived and 'l1.finish()' in derived
    # Execute the exact derived bias expression, not a duplicate test implementation.
    import torch
    expr = next(n.value for n in ast.walk(ast.parse(derived)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'bias0' for t in n.targets))
    assert torch.equal(eval(compile(ast.Expression(expr), '<derived-bias>', 'eval'), {'torch': torch, 'C': 12, 'dev': 'cpu'}), torch.zeros(12))
    with pytest.raises(ValueError): s.derive_trainer(source.replace('bias0 = torch.logit', 'bias0 = changed'))
    cmd = s.trainer_argv(s.CACHE)
    assert cmd[cmd.index('--folds') + 1] == '4'
    assert cmd[cmd.index('--only-folds') + 1] == '0'
    assert '--holdout-fold' not in cmd and '--mirror' not in cmd and '--resume' not in cmd
    assert cmd[cmd.index('--batch') + 1] == '2'

def test_full_977_holdout_mutation_cannot_enter_training():
    full = s.read(s.ROOT / s.REG)
    rows = [{s.UID: u, **{l: '.3' for l in s.LABELS}} for u in full['train']]
    reg = {'train': full['train'], 'holdout': full['val']}
    baseline = s.target_rows(reg, rows)
    poisoned = rows + [{s.UID: u, **{l: object() for l in s.LABELS}} for u in full['val']]
    assert s.target_rows(reg, poisoned) == baseline
    adapted = {x[s.UID]: x for x in baseline}
    selected_train = [u for u in adapted if s.fold(u) != 0]
    assert set(selected_train) == set(full['train'])
    assert not set(selected_train) & set(full['val'])

def measurements():
    return {'steps': 200, 'eval_batches': 16, 'eval_finite': True, 'windows_per_s': 80.,
            'losses': [.7] * 180 + [.6] * 20, 'peak_reserved': 11 * r.GIB,
            'desktop_bytes': 2 * r.GIB, 'total_bytes': 16 * r.GIB}

def test_go_thresholds_and_measured_headroom():
    m = measurements(); assert r.go_rule(m)
    for key, val in [('steps', 199), ('eval_batches', 0), ('eval_finite', False), ('windows_per_s', 79.99),
                     ('peak_reserved', 12 * r.GIB + 1), ('desktop_bytes', 4 * r.GIB), ('losses', [.7] * 200)]:
        n = dict(m); n[key] = val; assert not r.go_rule(n)
    assert not r.memory_ok(14e9, 0, 24 * r.GIB)  # fable 15GB alone insufficient
    assert not r.memory_ok(float('nan'), 0, 16 * r.GIB)

def test_watchdog_and_paths():
    assert r.stale_reason({'time': 0}, 900)
    assert r.stale_reason({'time': 0}, 600, {'util': 100, 'power': 80, 'desktop_bytes': 15, 'total_bytes': 16})
    assert r.stale_reason({'time': 0}, 599) is None
    assert c.safe_output(s.CACHE) == s.CACHE.resolve()
    with pytest.raises(ValueError): c.safe_output(s.ROOT.parent)
    with pytest.raises(ValueError): c.clear_owned(s.ROOT)
    with pytest.raises(ValueError): s.register(3190, 977)

def test_bounded_windows_replace(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.write_text('safe')
    real = os.replace
    with patch('tools.l1_rco_stage0_v1.os.replace', side_effect=[PermissionError(), None]) as call, patch('time.sleep'):
        s.replace(a, b); assert call.call_count == 2
    with patch('tools.l1_rco_stage0_v1.os.replace', side_effect=PermissionError()), patch('time.sleep'):
        with pytest.raises(PermissionError): s.replace(a, b)
    real(a, b); assert b.read_text() == 'safe'

def test_fail_closed_review(tmp_path):
    reg = {'train': [str(i) for i in range(200)], 'holdout': [str(i) for i in range(32)]}
    with patch.object(r, 'validate', return_value=reg), patch.object(r, 'read', return_value=reg), \
         patch.object(c, 'check_cache', side_effect=ValueError('missing cache')), patch.object(r, 'write'):
        assert r.admit(tmp_path, tmp_path / 'missing', True)['status'] == 'BLOCKED'
        with pytest.raises(ValueError): r.admit(tmp_path, tmp_path / 'missing')

def test_no_gpu_initialized():
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()


def test_exact_review_gate(tmp_path):
    reg = {'contract_id': 'contract', 'train': list(range(200)), 'holdout': list(range(32))}
    binding = {'contract_id': 'contract'}
    smoke = {'contract_id': 'contract', 'status': 'PASS', 'cuda_initialized': False}
    review = {'verdict': 'PASS', 'builder': 'builder', 'reviewer': 'reviewer', 'bindings': binding, 'report': 'report', 'report_sha256': 'hash'}
    review_path = tmp_path / 'review.json'
    def fake_read(path):
        if Path(path) == review_path:
            if review is None: raise FileNotFoundError('missing review')
            return review
        return smoke if Path(path).name == 'cpu-smoke.json' else reg
    with patch.object(r, 'validate', return_value=reg), patch.object(r, 'read', side_effect=fake_read), patch.object(c, 'check_cache'), patch.object(r, 'bindings', return_value=binding), patch.object(r, 'sha', return_value='hash'), patch.object(r, 'write'):
        assert r.admit(tmp_path, review_path, True)['status'] == 'CPU_ADMISSIBLE'
        valid = copy.deepcopy(review)
        for name, val in [('reviewer', 'builder'), ('bindings', {}), ('report_sha256', 'changed'), ('verdict', 'FAIL')]:
            review = {**valid, name: val}
            assert r.admit(tmp_path, review_path, True)['status'] == 'BLOCKED'
        review = None
        assert r.admit(tmp_path, review_path, True)['status'] == 'BLOCKED'


def test_gpu_job_identity_and_direct_run_blocked(tmp_path):
    from tools.job_runtime import command_digest
    command = ['python', 'runtime', 'run']
    current, parent = {'pid': 123, 'create_time': 1.}, {'pid': 456, 'create_time': 2.}
    argv = ['python', 'tools/gpu_job.py', '--_run-job', 'L1_RCO_K0_V1', 'run1']
    job = {'schema': 'gpu-job/v2', 'tag': 'L1_RCO_K0_V1', 'phase': 'running', 'child_identity': current, **parent,
           'command': command, 'command_sha256': command_digest(command, s.ROOT), 'cwd': str(s.ROOT), 'run_id': 'run1'}
    r.verify_job_record(job, command, current, parent, argv)
    for key, value in [('child_identity', parent), ('create_time', 3.), ('command_sha256', 'forged'), ('phase', 'spawning'), ('command', [])]:
        with pytest.raises(ValueError): r.verify_job_record({**job, key: value}, command, current, parent, argv)
    with pytest.raises(ValueError): r.verify_job_record(job, command, current, parent, ['python', 'other.py'])
    with patch.object(r, 'require_gpu_job', side_effect=ValueError('no owner')), patch.object(r, 'admit') as admission, patch.object(r.subprocess, 'Popen') as launch:
        with pytest.raises(ValueError): r.run(tmp_path, tmp_path)
        admission.assert_not_called(); launch.assert_not_called()


def test_geometry_valid_area():
    from types import SimpleNamespace
    g = SimpleNamespace(crop_xyxy=(0, 0, 100, 80), valid_xyxy=(0, 32, 384, 352), resized_shape=(320, 384))
    records = []
    check = c.geometry_checked(lambda *a, **k: g, records)
    assert check((80, 100), (1, 1), 384) is g and records == [320 / 384]
    g.valid_xyxy = (0, 32, 384, 385)
    with pytest.raises(ValueError): check((80, 100), (1, 1), 384)
