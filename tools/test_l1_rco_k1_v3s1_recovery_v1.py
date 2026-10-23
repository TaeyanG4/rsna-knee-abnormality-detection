"""H210 recovery safety and routing; CPU only, no production writes."""
import copy
import sys
from pathlib import Path

import pytest

from tools import l1_rco_k1_v3s1_recovery_v1 as r


def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'WORK', tmp_path)
    monkeypatch.setattr(r, 'OUT', tmp_path / 'attempt/fit')
    monkeypatch.setattr(r, 'MANIFEST', tmp_path / 'RECOVERY.json')
    monkeypatch.setattr(r, 'REVIEW', tmp_path / 'REVIEW.json')


def test_argv_changes_only_output_and_keeps_full_recipe():
    reg = r.read(r.k.WORK / 'registration.json')
    old = r.k.trainer_argv(r.k.CACHE, reg)
    new = r.trainer_argv(reg)
    assert len(old) == len(new)
    assert [i for i, pair in enumerate(zip(old, new)) if pair[0] != pair[1]] == [old.index('--out') + 1]
    assert new[new.index('--out') + 1] == str(r.OUT)
    assert new[new.index('--seed-base') + 1] == '1'
    assert new[new.index('--epochs') + 1] == '10'
    assert r.STEP == 10400 and r.TOTAL - r.STEP == 5550
    assert r.EPOCH * (3190 // 2) + r.BATCH == r.STEP


def test_launch_uses_module_command_and_separate_namespace():
    assert r.TAG != r.k.TAG and r.OUT != r.k.OUT
    assert r.launch_command() == [r.k.PYTHON, str(r.ROOT / 'tools/gpu_job.py'), '--tag', r.TAG, '--', *r.command()]
    assert r.command('train')[-1] == 'train'


def original_fixture(monkeypatch):
    from tools import job_runtime as jr
    reg = {'contract_id': r.CONTRACT, 'recipe': {'seed_base': 1}, 'train': list(range(3190)), 'holdout': list(range(977))}
    ck = dict(step_total=r.STEP, epoch=r.EPOCH, batch=r.BATCH, identity=r.STATE_ID,
              sha256=r.PAUSE_SHA, path=str(r.k.OUT / r.CHECKPOINT), backup=str(r.BACKUP))
    job = dict(run_id=r.RUN_ID, tag=r.k.TAG, command_sha256=jr.command_digest(r.k.run_command(), r.ROOT),
               child_identity={'pid': -1})
    records = {
        r.k.WORK / 'registration.json': reg,
        r.PAUSE / 'PAUSE-VERIFIED.json': {'checkpoint': ck},
        r.PAUSE / 'STOPPED.json': dict(state='USER_PAUSED_FOR_ROI', trainer_stopped=True, checkpoint=copy.deepcopy(ck)),
        jr.LOGS / r.k.TAG / 'job.json': job,
        jr.LOGS / r.k.TAG / 'receipt.json': dict(job, state='FAILED'),
        r.k.OUT / 'admission.json': dict(contract_id=r.CONTRACT, review_binding={'sealed': True}),
        r.k.OUT / 'ckpt/best.json': dict(sha256=r.PAUSE_SHA, step_total=9570),
    }
    monkeypatch.setattr(r, 'read', lambda p: records[Path(p)])
    monkeypatch.setattr(r, 'sha', lambda p: r.PAUSE_SHA)
    monkeypatch.setattr(r.k, 'validate', lambda x: x)
    monkeypatch.setattr(r.k, 'review_binding', lambda x: {'sealed': True})
    monkeypatch.setattr(jr, 'liveness', lambda x: 'dead')
    return records, reg


@pytest.mark.parametrize('fault', ['contract', 'seed', 'step', 'receipt', 'pause', 'hash', 'live', 'collapse', 'best', 'admission'])
def test_original_rejects_changed_contract_state_or_owner(monkeypatch, fault):
    from tools import job_runtime as jr
    records, reg = original_fixture(monkeypatch)
    if fault == 'contract': reg['contract_id'] = 'wrong'
    if fault == 'seed': reg['recipe']['seed_base'] = 0
    if fault == 'step':
        for name in ('PAUSE-VERIFIED.json', 'STOPPED.json'):
            records[r.PAUSE / name]['checkpoint']['step_total'] = 10433
    if fault == 'receipt': records[jr.LOGS / r.k.TAG / 'receipt.json']['run_id'] = 'wrong'
    if fault == 'pause': records[r.PAUSE / 'STOPPED.json']['trainer_stopped'] = False
    if fault == 'hash': monkeypatch.setattr(r, 'sha', lambda p: 'wrong')
    if fault == 'live': monkeypatch.setattr(jr, 'liveness', lambda x: 'alive')
    if fault == 'collapse':
        real = Path.exists
        monkeypatch.setattr(Path, 'exists', lambda p: True if p == r.k.OUT / r.k.COLLAPSE else real(p))
    if fault == 'best': records[r.k.OUT / 'ckpt/best.json']['step_total'] = 10433
    if fault == 'admission': records[r.k.OUT / 'admission.json']['contract_id'] = 'wrong'
    with pytest.raises(ValueError): r.original()


def test_original_accepts_bound_intentional_stop(monkeypatch):
    _, reg = original_fixture(monkeypatch)
    assert r.original() is reg


def test_registration_refuses_changed_manifest_and_unregistered_output(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(r, 'original', lambda: {})
    monkeypatch.setattr(r, 'expected_manifest', lambda reg: {'fixed': 1})
    assert r.register()['status'] == 'PASS'
    assert r.register()['status'] == 'PASS'
    r.write(r.MANIFEST, {'fixed': 2})
    with pytest.raises(ValueError, match='immutable'): r.register()
    monkeypatch.setattr(r, 'MANIFEST', tmp_path / 'absent.json')
    r.OUT.mkdir(parents=True)
    with pytest.raises(ValueError, match='unregistered'): r.register()


def test_prepare_copies_and_never_overwrites_progress(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path / 'new')
    old = tmp_path / 'old'
    monkeypatch.setattr(r.k, 'OUT', old)
    monkeypatch.setattr(r, 'BACKUP', tmp_path / 'backup.pt')
    monkeypatch.setattr(r, 'validate', lambda: {})
    r.write(r.MANIFEST, {'fixed': 1})
    r.BACKUP.write_bytes(b'paused-checkpoint')
    for rel in (r.k.BEST, 'ckpt/best.json', 'parameter-groups.json'):
        path = old / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'original')
    assert r.prepare()['status'] == 'PASS'
    assert r.prepare()['status'] == 'PASS'
    (r.OUT / r.CHECKPOINT).write_bytes(b'progressed')
    with pytest.raises(ValueError, match='do not overwrite'): r.prepare()
    assert (r.OUT / r.CHECKPOINT).read_bytes() == b'progressed'
    r.write(r.OUT / 'admission.json', {})
    with pytest.raises(ValueError, match='already admitted'): r.prepare()


def test_review_gate_and_report_binding(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'review_binding', lambda: {'fixed': 1})
    with pytest.raises(OSError): r.require_review()
    report = tmp_path / 'report.md'; report.write_text('Independent fixture only')
    good = dict(verdict='PASS', builder='astra-b2', reviewer='independent-fixture', bindings={'fixed': 1},
                report='report.md', report_sha256=r.sha(report))
    r.write(r.REVIEW, good); r.require_review()
    for key, val in [('reviewer', 'astra-b2'), ('bindings', {'fixed': 2}), ('verdict', 'FAIL'), ('report_sha256', 'wrong')]:
        r.write(r.REVIEW, dict(good, **{key: val}))
        with pytest.raises(ValueError): r.require_review()


def test_dry_admission_redirects_original_write_and_keeps_review_pending(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(r, 'validate', lambda: {})
    r.write(r.MANIFEST, {'fixed': 1})
    r.write(r.WORK / 'prepare.json', {'manifest_sha256': r.sha(r.MANIFEST)})
    r.write(r.WORK / 'cpu-check.json', dict(status='PASS', manifest_sha256=r.sha(r.MANIFEST), cuda_initialized=False))
    def admission(*args):
        value = dict(status='CPU_ADMISSIBLE', blockers=[])
        r.k.write(r.k.WORK / 'admit-dry-run.json', value)
        return value
    monkeypatch.setattr(r.k, 'admit', admission)
    original_write = r.k.write
    result = r.admit()
    assert result['recovery_review'] == 'PENDING' and not result['launch_ready']
    assert (r.WORK / 'original-admit-dry-run.json').exists()
    assert r.k.write is original_write


def test_replay_patch_is_scoped_and_restores_on_error():
    from tools.l1_rco_k1_state_v2 import TrainerState
    prior = TrainerState.batches
    with pytest.raises(RuntimeError):
        with r.replay_adapter():
            assert TrainerState.batches is r.replay.replay_batches
            raise RuntimeError('fixture')
    assert TrainerState.batches is prior


def test_bare_run_requires_owner_before_any_mutation(monkeypatch):
    def rejected(): raise ValueError('no owner')
    monkeypatch.setattr(r, 'owner', rejected)
    with pytest.raises(ValueError, match='no owner'): r.run()


def test_run_routes_both_children_and_export_checks_to_new_output(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(r, 'owner', lambda: 123)
    monkeypatch.setattr(r, 'sha', lambda p: r.PAUSE_SHA)
    monkeypatch.setattr(r, 'admit', lambda *a: {'contract_id': r.CONTRACT})
    reg = {'collapse_guard': {'constant_loss': .5}}
    monkeypatch.setattr(r, 'validate', lambda: reg)
    calls = []
    monkeypatch.setattr(r.k, 'supervise', lambda action, cmd, env, out, constant: calls.append((action, cmd, out)))
    monkeypatch.setattr(r.k, 'final_training_log', lambda reg, out: calls.append(('final', out)))
    monkeypatch.setattr(r.k, 'completed_fit', lambda reg, out: {'optimizer_steps': r.TOTAL} if out == r.OUT else {})
    r.write(r.OUT / 'result.json', {'status': 'COMPLETE'})
    assert r.run()['status'] == 'COMPLETE'
    assert calls == [('train', r.command('train'), r.OUT), ('final', r.OUT), ('infer', r.command('infer'), r.OUT)]
    with pytest.raises(ValueError, match='one recovery attempt'): r.run()


def test_train_child_runs_original_trainer_with_scoped_replay(monkeypatch):
    from tools.l1_rco_k1_state_v2 import TrainerState
    reg = r.read(r.k.WORK / 'registration.json')
    monkeypatch.setattr(r, 'owner', lambda child=False: 123 if child else None)
    monkeypatch.setattr(r, 'validate', lambda: reg)
    monkeypatch.setattr(r, 'require_review', lambda: None)
    monkeypatch.setattr(r, 'sha', lambda p: r.PAUSE_SHA)
    monkeypatch.setattr(sys, 'argv', [r.__file__, 'train'])
    called = []
    def runpath(path, run_name):
        assert path == str(r.k.WORK / 'compact_train_l1.py') and run_name == '__main__'
        assert sys.argv == r.trainer_argv(reg)[1:]
        assert TrainerState.batches is r.replay.replay_batches
        called.append(True)
    monkeypatch.setattr(r.runpy, 'run_path', runpath)
    r.main()
    assert called == [True]


def test_infer_child_keeps_original_977_roster(monkeypatch):
    reg = {'holdout': list(range(977))}
    monkeypatch.setattr(r, 'owner', lambda child=False: 123)
    monkeypatch.setattr(r, 'validate', lambda: reg)
    monkeypatch.setattr(r, 'require_review', lambda: None)
    monkeypatch.setattr(sys, 'argv', [r.__file__, 'infer'])
    called = []
    def inference(value, cache, out):
        assert value is reg and cache == r.k.CACHE and out == r.OUT
        called.append(True); return {'status': 'COMPLETE'}
    monkeypatch.setattr(r.k, 'inference', inference)
    r.main()
    assert called == [True]
