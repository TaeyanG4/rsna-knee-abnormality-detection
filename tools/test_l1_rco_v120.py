"""Regression tests for evaluation-only K0 V2; CUDA disabled."""
import ast
import copy
import inspect
import os
from pathlib import Path
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pytest
from tools import l1_rco_stage0_v1 as old
from tools import l1_rco_runtime_v1 as prior
from tools import l1_rco_stage0_v120 as s
from tools import l1_rco_runtime_v120 as r
from tools import l1_rco_fullcache_v110 as full


def test_entire_trainer_diff_is_eval_only():
    source = (s.ROOT / s.SOURCE / 'compact_train.py').read_text(encoding='utf-8')
    parent, new = old.derive_trainer(source), s.derive_trainer(source)
    ast.parse(new)
    restored = new.replace('from tools.l1_rco_runtime_v120 import Monitor', 'from tools.l1_rco_runtime_v1 import Monitor')
    restored = restored.replace('        if dev.type == "cuda":\n            torch.cuda.empty_cache()  # evaluation-only allocator cleanup\n', '')
    restored = restored.replace('mask.to(dev), chunk=32)', 'mask.to(dev), chunk=PREDICT_CHUNK)')
    assert restored == parent  # entire source, not just selected training lines
    before_tree, after_tree = ast.parse(parent), ast.parse(new)
    for name in ('Net', 'Studies', 'collate', 'bce_soft'):
        before = next(n for n in ast.walk(before_tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
        after = next(n for n in ast.walk(after_tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
        assert ast.dump(before) == ast.dump(after)
    predict = next(n for n in ast.walk(after_tree) if isinstance(n, ast.FunctionDef) and n.name == 'predict')
    assert ast.unparse(predict.decorator_list[0]) == 'torch.no_grad()'
    with pytest.raises(ValueError): s.derive_trainer(source.replace('chunk=PREDICT_CHUNK)', 'chunk=999)'))


def test_training_argv_and_monitor_gates_identical():
    expected = [x.replace(str(old.WORK), str(s.WORK)) for x in old.trainer_argv(s.CACHE)]
    assert s.trainer_argv(s.CACHE) == expected
    for name in ('memory_ok', 'go_rule', 'stale_reason'):
        assert inspect.getsource(getattr(r, name)) == inspect.getsource(getattr(prior, name))
    for name in ('step', 'phase', 'emit', 'evaluation'):
        assert inspect.getsource(getattr(r.Monitor, name)) == inspect.getsource(getattr(prior.Monitor, name))
    assert not r.memory_ok(12 * r.GIB + 1, 0, 20 * r.GIB)
    assert not r.memory_ok(11 * r.GIB, 4 * r.GIB, 16 * r.GIB)


def test_new_attempt_and_exact_gpu_job_identity():
    from tools.job_runtime import command_digest
    command = [s.PYTHON, str(s.ROOT / 'tools/l1_rco_runtime_v120.py'), 'run', '--cache', str(s.CACHE), '--review', str(s.WORK / 'K0-REVIEW.json')]
    current, parent = {'pid': 2, 'create_time': 2.}, {'pid': 1, 'create_time': 1.}
    job = dict(schema='gpu-job/v2', tag='L1_RCO_K0_V2', phase='running', child_identity=current,
               command=command, command_sha256=command_digest(command, s.ROOT), cwd=str(s.ROOT), run_id='unit', **parent)
    argv = ['python', 'tools/gpu_job.py', '--_run-job', 'L1_RCO_K0_V2', 'unit']
    r.verify_job_record(job, command, current, parent, argv)
    job['tag'] = 'L1_RCO_K0_V1'
    with pytest.raises(ValueError): r.verify_job_record(job, command, current, parent, argv)
    assert s.WORK != old.WORK and not s.WORK.is_relative_to(old.WORK)


def test_eval_binding_rejects_stale_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'WORK', tmp_path)
    s.write(tmp_path / 'prepare.json', {'eval_pins': {}, 'version': s.VERSION, 'eval_chunk': 32})
    with pytest.raises(ValueError, match='evaluation source'): r.bindings({}, tmp_path)


def test_fullcache_v110_accepts_new_result_path_with_unchanged_data_contract(tmp_path, monkeypatch):
    # Uses actual numeric gate, no patched go_rule. Never writes a genuine admission.
    monkeypatch.setattr(full, 'ROOT', tmp_path)
    monkeypatch.setattr(full, 'WORK', tmp_path / 'fullcache')
    monkeypatch.setattr(full.s, 'WORK', tmp_path / 'stage0')
    s.write(full.s.WORK / 'registration.json', {'contract_id': 'sealed-data'})
    s.write(full.WORK / 'proof.json', {'status': 'PASS', 'contract_id': 'full'})
    result_path = tmp_path / '.local/scratch/l1-rco/k0-v2/k0/result.json'
    result = dict(version='1.2.0', contract_id='sealed-data', K0_numeric_gate='PASS', steps=200,
                  eval_batches=32, eval_finite=True, windows_per_s=113.6, losses=[.58] * 180 + [.56] * 20,
                  peak_reserved=11 * r.GIB, desktop_bytes=1.84e9, total_bytes=17.17e9)
    s.write(result_path, result)
    report = tmp_path / 'review.md'; report.write_text('TEST FIXTURE ONLY')
    review = tmp_path / 'review.json'
    verdict = dict(verdict='PASS', builder='l1-rco', reviewer='other', contract_id='full',
                   K0_result_sha256=s.sha(result_path), proof_sha256=s.sha(full.WORK / 'proof.json'),
                   report='review.md', report_sha256=s.sha(report))
    s.write(review, verdict)
    full.authorize_full({'contract_id': 'full'}, result_path, review)
    result['peak_reserved'] = 12 * r.GIB + 1
    s.write(result_path, result); verdict['K0_result_sha256'] = s.sha(result_path); s.write(review, verdict)
    with pytest.raises(ValueError, match='K0 did not pass'): full.authorize_full({'contract_id': 'full'}, result_path, review)
