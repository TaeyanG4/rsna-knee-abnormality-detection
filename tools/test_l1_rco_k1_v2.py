"""K1 v2.0.0 CPU tests: registered changes (LR / warm-up / epochs), collapse guard,
best-epoch checkpoint copy, launcher COLLAPSE marking and backstop, plus the ported v1
fail-closed admission, data-isolation and durable-state regressions."""
import ast
import copy
import csv
import difflib
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pytest
from tools import l1_rco_k1_v1 as v1
from tools import l1_rco_k1_v2 as k
from tools import l1_rco_k1_state_v2 as st
from tools.l1_rco_k1_state_v2 import Monitor, TrainerState, ThroughputGuard

# K1 v1 committed epoch losses, fit/train.log epochs 0-7 (D-B-214), and the train3190 constant.
V1_LOG = [0.5255, 0.5110, 0.5050, 0.5303, 0.5295, 0.5311, 0.5308, 0.5306]
CONST = 0.5302471482607706


def registration():
    return k.read(k.WORK / 'registration.json')


# ---------------------------------------------------------------- collapse guard

def test_guard_v1_collapse_sequence_triggers_at_epoch_3():
    hit = k.collapse_reason(V1_LOG, CONST)
    assert hit['reason'] == 'COLLAPSE' and hit['epoch'] == 3 and hit['loss'] == 0.5303 and hit['best_previous'] == 0.5050
    assert len(hit['criteria']) == 2  # both the constant-loss and the +0.015 rise criteria fire
    for n in range(4): assert k.collapse_reason(V1_LOG[:n], CONST) is None
    assert k.collapse_reason(V1_LOG[:4], CONST)['epoch'] == 3


def test_v1_log_constant_matches_v1_train_log_and_registration():
    log = v1.OUT / 'train.log'  # stopped v1 run, evidence (read only)
    if log.exists():
        import re
        seen = [float(x) for x in re.findall(r'epoch \d+ loss ([\d.]+)', log.read_text(encoding='utf-8'))]
        assert seen[:8] == V1_LOG
    reg = registration(); guard = reg['collapse_guard']
    with (k.WORK / 'targets.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3190 and k.constant_loss(rows) == guard['constant_loss'] == CONST
    assert guard['threshold'] == CONST - .002 and {x: guard[x] for x in k.GUARD} == k.GUARD
    assert k.collapse_reason(V1_LOG, guard['constant_loss'])['epoch'] == 3


@pytest.mark.parametrize('log,epoch,n', [
    ([0.52, 0.50, 0.49, 0.485, 0.48, 0.47, 0.46, 0.455, 0.45, 0.449], None, 0),  # healthy 10 epochs
    ([0.5300, 0.5299], None, 0),                         # epochs 0-1 may sit at the constant (gate e >= 2)
    ([0.5300, 0.5299, 0.52825], 2, 1),                   # loss_e >= const - .002 (rounded boundary hit)
    ([0.5300, 0.5299, 0.52824], None, 0),
    ([0.50, 0.48, 0.49501], 2, 1),                       # rise 0.01501 > 0.015 only
    ([0.50, 0.48, 0.495], None, 0),                      # rise exactly 0.015 is not > 0.015
    ([0.40, 0.4151], 1, 1),                              # rise criterion is not epoch-gated
    ([0.45, 0.40, 0.41, 0.416], 3, 1),                   # rise measured against the best, not the last epoch
])
def test_guard_synthetic_sequences(log, epoch, n):
    const = 0.53025  # threshold 0.52825
    hit = k.collapse_reason(log, const)
    assert (hit is None) if epoch is None else (hit['epoch'] == epoch and len(hit['criteria']) == n)


def test_guard_rejects_invalid_inputs():
    with pytest.raises(ValueError): k.collapse_reason([0.5, float('nan')], CONST)
    for bad in (0.0, -1.0, float('inf'), 1):
        with pytest.raises(ValueError): k.collapse_reason([0.5], bad)


# ---------------------------------------------------------------- warm-up option

def _scheduler(source, epochs, warm_frac, steps=k.STEPS_PER_EPOCH, cpu_test=False):
    import torch
    start = source.index('            total = len(dl) * a.epochs')
    end = source.index('            tr_t = torch.from_numpy(tr)', start)
    printed = []
    ns = dict(len=len, max=max, int=int, math=math, torch=torch, dl=range(steps), print=lambda *x, **y: printed.append(x),
              a=SimpleNamespace(epochs=epochs, warm_frac=warm_frac, l1_cpu_test=cpu_test),
              opt=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.))
    exec(textwrap.dedent(source[start:end]), ns)
    return ns['warm'], ns['total'], ns['sched'].lr_lambdas[0], printed


def test_warm_frac_default_is_parent_formula_and_registered_value_is_one_epoch():
    parent = (k.ROOT / k.s.SOURCE / 'compact_train.py').read_text(encoding='utf-8')
    derived = k.derive_trainer(parent)
    assert 'ap.add_argument("--warm-frac", type=float, default=.05' in derived
    for epochs in (10, 16):  # default .05: identical warm length and every LR multiplier
        pw, pt, pl, _ = _scheduler(parent, epochs, None)
        dw, dt, dl, _ = _scheduler(derived, epochs, .05, cpu_test=True)  # production assert tested below
        assert (pw, pt) == (dw, dt) and all(pl(i) == dl(i) for i in range(pt + 2))
    assert _scheduler(parent, 16, None)[0] == 1276 < k.STEPS_PER_EPOCH  # v1: 0.8 epoch
    warm, total, lam, printed = _scheduler(derived, 10, .1)
    assert (warm, total) == (1595, 15950) and warm == k.STEPS_PER_EPOCH  # v2: exactly 1 epoch
    assert lam(0) == 1 / 1595 and lam(1594) == 1. and lam(1595) == 1. and lam(15949) < 1e-7
    assert printed and '1595 of 15950 (1.00 epochs' in printed[0][0]
    with pytest.raises(AssertionError): _scheduler(derived, 10, .05)  # production refuses < 1 epoch warm-up
    assert _scheduler(derived, 10, .05, cpu_test=True)[0] == 797  # CPU fixtures exempt


def test_registered_recipe_and_argv_change_only_registered_flags():
    reg = registration(); r = reg['recipe']
    assert (r['lr'], r['head_lr'], r['epochs'], r['warm_frac']) == (1e-4, 3e-4, 10, .1)
    assert (r['warm_steps'], r['steps_per_epoch'], r['total_steps'], r['batch'], r['windows'], r['crop']) == (1595, 1595, 15950, 2, 64, 352)
    assert reg['changes_vs_v1'] == {'lr': {'v1': 3e-4, 'v2': 1e-4}, 'head_lr': {'v1': 1e-3, 'v2': 3e-4},
                                    'epochs': {'v1': 16, 'v2': 10}, 'warm_frac': {'v1': .05, 'v2': .1}}
    cache = k.CACHE; old, new = v1.trainer_argv(cache), k.trainer_argv(cache, reg)
    assert new[len(old):] == ['--warm-frac', '0.1', '--l1-guard-constant', repr(CONST)]
    changed = {old[i - 1] if i > 1 else i for i in range(len(old)) if old[i] != new[i]}
    assert changed == {1, '--targets', '--out', '--epochs', '--lr', '--head-lr'}
    value = lambda flag: new[new.index(flag) + 1]
    assert (value('--lr'), value('--head-lr'), value('--epochs'), value('--batch'), value('--ema')) == ('0.0001', '0.0003', '10', '2', '.999')
    assert float(value('--lr')) == r['lr'] and float(value('--head-lr')) == r['head_lr']
    assert new[1] == str(k.WORK / 'compact_train_l1.py') and value('--out') == str(k.OUT) and value('--targets') == str(k.WORK / 'targets.csv')


def test_trainer_args_check_rejects_unregistered_values():
    reg = registration(); r = reg['recipe']
    good = dict(lr=r['lr'], head_lr=r['head_lr'], epochs=r['epochs'], warm_frac=r['warm_frac'], batch=2, train_windows=64,
                crop=352, l1_guard_constant=reg['collapse_guard']['constant_loss'])
    k.check_trainer_args(SimpleNamespace(**good))
    for key, bad in dict(lr=3e-4, head_lr=1e-3, epochs=16, warm_frac=.05, batch=1, train_windows=32, crop=384, l1_guard_constant=.5).items():
        with pytest.raises(ValueError, match='registered K1 v2 recipe'): k.check_trainer_args(SimpleNamespace(**dict(good, **{key: bad})))


# ---------------------------------------------------------------- derivative

def test_derivative_is_v1_derivative_plus_registered_patches_only():
    source = (k.ROOT / k.s.SOURCE / 'compact_train.py').read_text(encoding='utf-8')
    base, new = v1.derive_trainer(source), k.derive_trainer(source)
    before, after = ast.parse(source), ast.parse(new)
    for name in ('Net', 'Studies', 'collate', 'bce_soft', 'lr_loss', 'augment'):
        nodes = lambda tree: [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        assert ast.dump(nodes(before)[0]) == ast.dump(nodes(after)[0])
    lines = list(difflib.unified_diff(base.splitlines(), new.splitlines(), lineterm='', n=0))
    removed = [x[1:] for x in lines if x.startswith('-') and not x.startswith('---')]
    added = [x[1:] for x in lines if x.startswith('+') and not x.startswith('+++')]
    assert removed == ['    from tools.l1_rco_k1_state_v1 import Monitor, TrainerState',
                       '        assert not a.l1_stop_steps',
                       '        from tools.l1_rco_k1_v1 import check_training_ids',
                       '            warm = max(1, int(.05 * total))',
                       '                                    resume=a.resume, every=100, stop_steps=a.l1_stop_steps)']
    expected = [
        '# K1 v2.0.0 derivative generated by tools/l1_rco',
        '# (default .05 = parent), registered LR/epoch/gu',
        'ap.add_argument("--l1-guard-constant", type=flo',
        'ap.add_argument("--warm-frac", type=float, defa',
        'from tools.l1_rco_k1_state_v2 import Monitor, T',
        'assert not a.l1_stop_steps and a.l1_guard_const',
        'from tools.l1_rco_k1_v2 import check_trainer_ar',
        'check_trainer_args(a)',
        'from tools.l1_rco_k1_v2 import check_training_i',
        'warm = max(1, int(a.warm_frac * total))',
        'print("warm-up steps %d of %d (%.2f epochs, warm',
        'assert a.l1_cpu_test or warm >= len(dl)  # K1 v',
        'resume=a.resume, every=100, stop_steps=a.l1_sto',
        'guard_constant=a.l1_guard_constant)']
    assert len(added) == len(expected) and all(x.strip().startswith(e) for x, e in zip(added, expected)), added
    assert (k.WORK / 'compact_train_l1.py').read_text(encoding='utf-8') == new  # prepared artifact regenerates exactly
    with pytest.raises(ValueError): k.derive_trainer(source.replace('            warm = max(1, int(.05 * total))', '            warm = 1'))


# ---------------------------------------------------------------- best checkpoint + trainer guard

def _state(out, guard=None, epochs=10):
    import torch
    model = torch.nn.Linear(2, 1); opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    ds = torch.utils.data.TensorDataset(torch.zeros(2, 2)); ds.epoch = 0
    dl = torch.utils.data.DataLoader(ds, batch_size=1, generator=torch.Generator().manual_seed(3))
    state = TrainerState(out, model, opt, sched, None, torch.Generator(), dl, ['a', 'b'], epochs, Monitor(out, cpu=True), guard_constant=guard)
    return state, model, ds


def _epoch(state, model, ds, epoch, log):
    import torch
    state.begin_epoch(epoch, ds)
    with torch.no_grad(): model.weight.add_(1.)  # distinct checkpoint content per epoch
    state.epoch_end(epoch, log)


def test_best_epoch_copy_keeps_first_minimum(tmp_path):
    state, model, ds = _state(tmp_path)
    logs = [[.5], [.5, .4], [.5, .4, .45], [.5, .4, .45, .4], [.5, .4, .45, .4, .41]]
    for epoch, log in enumerate(logs):
        _epoch(state, model, ds, epoch, log)
        if epoch == 1: expected = k.sha(tmp_path / 'ckpt/E_prior_s0_f0.pt')
    meta = k.read(tmp_path / st.BEST_META)
    assert meta['epoch'] == 1 and meta['loss'] == .4 and meta['sha256'] == expected == k.sha(tmp_path / st.BEST)
    assert k.sha(tmp_path / 'ckpt/E_prior_s0_f0.pt') != expected  # rolling file moved on; best copy survives
    import torch
    ck = torch.load(tmp_path / st.BEST, map_location='cpu', weights_only=False)
    assert ck['epoch'] == 2 and ck['batch'] == 0 and ck['log'] == [.5, .4]
    sidecar = k.read(tmp_path / st.EPOCHS)
    assert sidecar['log'] == logs[-1] and sidecar['best_epoch'] == 1 and sidecar['committed_epochs'] == 5


def test_missed_best_copy_is_repaired_on_resume(tmp_path):
    state, model, ds = _state(tmp_path)
    _epoch(state, model, ds, 0, [.5])
    state.begin_epoch(1, ds)
    st.TrainerStateV1.epoch_end(state, 1, [.5, .4])  # crash after the rolling save, before the copy
    assert k.read(tmp_path / st.BEST_META)['epoch'] == 0
    _state(tmp_path)  # resume
    meta = k.read(tmp_path / st.BEST_META)
    assert meta['epoch'] == 1 and meta['sha256'] == k.sha(tmp_path / 'ckpt/E_prior_s0_f0.pt') == k.sha(tmp_path / st.BEST)


def test_trainer_guard_stops_after_committed_checkpoint_and_restops_on_resume(tmp_path):
    import torch
    state, model, ds = _state(tmp_path, guard=CONST)
    for epoch in range(3): _epoch(state, model, ds, epoch, V1_LOG[:epoch + 1])
    with pytest.raises(SystemExit) as stop: _epoch(state, model, ds, 3, V1_LOG[:4])
    assert stop.value.code == st.COLLAPSE_EXIT == k.GUARD['exit_code']
    receipt = k.read(tmp_path / st.COLLAPSE)
    assert receipt['epoch'] == 3 and receipt['checkpoint_kept'] and receipt['best']['epoch'] == 2 and receipt['status'] == 'FAILED'
    ck = torch.load(tmp_path / 'ckpt/E_prior_s0_f0.pt', map_location='cpu', weights_only=False)
    assert ck['epoch'] == 4 and ck['batch'] == 0 and ck['log'] == V1_LOG[:4]
    best = torch.load(tmp_path / st.BEST, map_location='cpu', weights_only=False)
    assert best['epoch'] == 3 and best['log'] == V1_LOG[:3]  # pre-collapse epoch-2 state survives
    rolling = k.sha(tmp_path / 'ckpt/E_prior_s0_f0.pt')
    with pytest.raises(SystemExit) as again: _state(tmp_path, guard=CONST)
    assert again.value.code == 4 and k.sha(tmp_path / 'ckpt/E_prior_s0_f0.pt') == rolling
    other = tmp_path / 'unguarded'; state, model, ds = _state(other)  # CPU fixture: guard disabled
    for epoch in range(4): _epoch(state, model, ds, epoch, V1_LOG[:epoch + 1])
    assert not (other / st.COLLAPSE).exists()


# ---------------------------------------------------------------- launcher marking / backstop

FAKE = '''import json, os, sys, time
from pathlib import Path
out = Path(sys.argv[1]); mode = sys.argv[2]; log = json.loads(sys.argv[3])
(out / 'ckpt').mkdir(parents=True, exist_ok=True)
(out / 'ckpt/E_prior_s0_f0.pt').write_bytes(b'rolling checkpoint fixture')
(out / 'child.pid').write_text(str(os.getpid()))
(out / 'ckpt/epochs.json').write_text(json.dumps(dict(log=log, committed_epochs=len(log))))
if mode == 'exit4':
    (out / 'collapse.json').write_text(json.dumps(dict(reason='COLLAPSE', epoch=3, loss=log[3], criteria=['fixture'], status='FAILED')))
    sys.exit(4)
if mode == 'hang': time.sleep(120)
sys.exit(1 if mode == 'error' else 0)
'''


def _supervise(tmp_path, mode, log, grace=60.):
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / 'fake_trainer.py'; script.write_text(FAKE, encoding='utf-8')
    out = tmp_path / 'fit'; out.mkdir(exist_ok=True)
    import json
    return out, lambda: k.supervise('train', [sys.executable, str(script), str(out), mode, json.dumps(log)],
                                    os.environ.copy(), out, CONST, poll=.2, grace=grace, gpu=False)


def test_trainer_exit4_marks_failed_collapse_and_keeps_checkpoint(tmp_path):
    out, call = _supervise(tmp_path, 'exit4', V1_LOG[:4])
    with pytest.raises(k.CollapseStop): call()
    failure = k.read(out / 'failure.json')
    assert failure['status'] == 'FAILED' and failure['reason'] == 'COLLAPSE' and failure['epoch'] == 3 and failure['checkpoint_kept']
    assert (out / 'ckpt/E_prior_s0_f0.pt').read_bytes() == b'rolling checkpoint fixture'
    assert k.read(out / 'status.json')['phase'] == 'failed-collapse'


def test_launcher_backstop_kills_trainer_that_ignores_guard(tmp_path):
    import psutil
    out, call = _supervise(tmp_path, 'hang', V1_LOG[:4], grace=1.)
    with pytest.raises(k.CollapseStop, match='epoch 3'): call()
    receipt, failure = k.read(out / 'collapse.json'), k.read(out / 'failure.json')
    assert receipt['epoch'] == 3 and 'backstop' in receipt['source'] and failure['reason'] == 'COLLAPSE' and failure['checkpoint_kept']
    pid = int((out / 'child.pid').read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_healthy_child_passes_and_error_child_is_generic_failure(tmp_path):
    out, call = _supervise(tmp_path, 'ok', [0.52, 0.50, 0.49])
    assert call() is None and not (out / 'failure.json').exists()
    out, call = _supervise(tmp_path / 'e', 'error', [0.52])
    with pytest.raises(ValueError, match='child failed'): call()
    assert k.read(out / 'failure.json')['reason'] == 'ERROR'


def test_post_fit_log_check(tmp_path):
    reg = dict(collapse_guard=dict(constant_loss=CONST), recipe=dict(epochs=10))
    healthy = [0.52, 0.50, 0.49, 0.485, 0.48, 0.47, 0.46, 0.455, 0.45, 0.449]
    k.write(tmp_path / k.EPOCHS, dict(log=healthy, committed_epochs=10))
    assert k.final_training_log(reg, tmp_path)['log'] == healthy
    k.write(tmp_path / k.EPOCHS, dict(log=healthy[:9], committed_epochs=9))
    with pytest.raises(ValueError, match='completed epoch log'): k.final_training_log(reg, tmp_path)
    k.write(tmp_path / k.EPOCHS, dict(log=V1_LOG + [0.53, 0.53], committed_epochs=10))
    with pytest.raises(k.CollapseStop): k.final_training_log(reg, tmp_path)
    assert k.read(tmp_path / 'failure.json')['reason'] == 'COLLAPSE'


def test_collapsed_run_is_never_relaunched(tmp_path, monkeypatch):
    monkeypatch.setattr(k, 'WORK', tmp_path); monkeypatch.setattr(k, 'OUT', tmp_path / 'fit')
    monkeypatch.setattr(k, 'require_gpu_job', lambda c, r: 1)
    monkeypatch.setattr(k, 'admit', lambda *x, **y: pytest.fail('admission after COLLAPSE'))
    k.write(tmp_path / 'registration.json', {}); k.write(tmp_path / 'fit' / k.COLLAPSE, dict(epoch=3, criteria=['x']))
    with pytest.raises(k.CollapseStop): k.run(tmp_path, tmp_path / 'review.json')


def test_evaluate_uses_unchanged_evaluator_on_v2_dir(monkeypatch):
    from tools import l1_rco_k1_eval_v1 as ev
    original = ev.WORK
    monkeypatch.setattr(ev, 'evaluate', lambda reg, dry_run=False: (ev.WORK, dry_run))
    assert k.evaluate({}, dry_run=True) == (k.WORK, True) and ev.WORK == original == v1.WORK
    assert registration()['evaluator']['version'] == ev.VERSION == '1.1.0'


# ---------------------------------------------------------------- ported v1 regressions

def _assert_k0_evidence_mutation_rejected(tmp_path, monkeypatch, path, key, message):
    reg = registration()
    k.validate(reg)  # Control: current authenticated inputs remain valid.
    actual_sha = k.sha
    assert actual_sha(path) == reg[key]
    monkeypatch.setattr(k, 'sha', lambda p: '0' * 64 if Path(p) == path else actual_sha(p))
    with pytest.raises(ValueError, match=message):
        k.validate(reg)
    monkeypatch.setattr(k, 'WORK', tmp_path)
    monkeypatch.setattr(k, 'complete_cache', lambda r, c: dict(studies=4167))
    result = k.admit(reg, tmp_path, tmp_path / 'review.json')
    assert result['status'] == 'BLOCKED' and result['blockers'] == [message]
    assert actual_sha(path) == reg[key]


def test_registered_k0_result_mutation_rejected_before_admission(tmp_path, monkeypatch):
    _assert_k0_evidence_mutation_rejected(tmp_path, monkeypatch, k.K0_RESULT, 'K0_result_sha256', 'registered K0 result changed')


def test_registered_k0_review_mutation_rejected_before_admission(tmp_path, monkeypatch):
    _assert_k0_evidence_mutation_rejected(tmp_path, monkeypatch, k.K0_REVIEW, 'K0_review_sha256', 'registered K0 review changed')


@pytest.mark.parametrize('evidence', ['admission.json', 'holdout.csv', 'ckpt/E_prior_s0_f0.pt', k.BEST, k.COLLAPSE])
@pytest.mark.parametrize('registration_present', [True, False])
def test_post_candidate_admission_or_output_registration_change_refused(tmp_path, monkeypatch, evidence, registration_present):
    old = registration(); old['version'] = 'old-version'
    monkeypatch.setattr(k, 'WORK', tmp_path); monkeypatch.setattr(k, 'OUT', tmp_path / 'fit')
    (k.OUT / evidence).parent.mkdir(parents=True, exist_ok=True)
    (k.OUT / evidence).write_text('existing production evidence')
    if registration_present: k.write(tmp_path / 'registration.json', old)
    with pytest.raises(ValueError, match='immutable|cannot change'): k.register()


def test_recipe_tamper_rejected_by_validate():
    reg = registration(); k.validate(reg)
    for key, value in dict(lr=3e-4, head_lr=1e-3, epochs=16, warm_frac=.05, warm_steps=797).items():
        bad = copy.deepcopy(reg); bad['recipe'][key] = value; bad['contract_id'] = k.digest({x: y for x, y in bad.items() if x != 'contract_id'})
        with pytest.raises(ValueError, match='registered recipe'): k.validate(bad)
    bad = copy.deepcopy(reg); bad['collapse_guard']['rise'] = .05; bad['contract_id'] = k.digest({x: y for x, y in bad.items() if x != 'contract_id'})
    with pytest.raises(ValueError, match='collapse guard'): k.validate(bad)


def test_training_exact_roster_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(k, 'WORK', tmp_path)
    k.write(tmp_path / 'registration.json', dict(train=['a', 'b'], holdout=['c']))
    k.check_training_ids(['b', 'a'])
    for ids in (['a'], ['a', 'b', 'c'], ['a', 'b', 'b']):
        with pytest.raises(ValueError): k.check_training_ids(ids)


def test_production_targets_contain_only_train(monkeypatch):
    reg = registration()
    with (k.WORK / 'targets.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    ids = {r[k.s.UID] for r in rows}
    assert ids == set(reg['train']) and not ids & set(reg['holdout']) and k.read(k.WORK / 'prepare.json')['targets_sha256'] == k.sha(k.WORK / 'targets.csv')
    assert k.sha(k.WORK / 'targets.csv') == k.sha(v1.WORK / 'targets.csv')  # same train-only targets as v1


def test_incomplete_cache_refuses_before_manifest_read(tmp_path):
    k.write(tmp_path / 'status.json', dict(phase='assemble', done=4167, total=4167))
    with pytest.raises(ValueError, match='incomplete'): k.complete_cache({}, tmp_path)


def test_review_admission_cannot_self_approve(tmp_path, monkeypatch):
    reg = registration()
    monkeypatch.setattr(k, 'WORK', tmp_path)
    monkeypatch.setattr(k, 'validate', lambda r: r)
    monkeypatch.setattr(k, 'review_binding', lambda r: dict(contract_id='c'))
    monkeypatch.setattr(k, 'complete_cache', lambda r, c: dict(studies=4167))
    k.write(tmp_path / 'prepare.json', dict(argv=k.trainer_argv(tmp_path, reg), cache=str(tmp_path.resolve())))
    review = tmp_path / 'review.json'
    k.write(review, dict(verdict='PASS', builder='same', reviewer='same', bindings=dict(contract_id='c')))
    result = k.admit(reg, tmp_path, review)
    assert result['status'] == 'BLOCKED' and 'independent' in result['blockers'][0]
    with pytest.raises(ValueError, match='independent'): k.admit(reg, tmp_path, review, dry_run=False)


def test_checkpoint_contract_identity_rejects_different_recipe(tmp_path):
    import torch
    model = torch.nn.Linear(2, 1); opt = torch.optim.AdamW(model.parameters()); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    ds = torch.utils.data.TensorDataset(torch.zeros(2, 2)); dl = torch.utils.data.DataLoader(ds, batch_size=1, generator=torch.Generator().manual_seed(3))
    monitor = Monitor(tmp_path, cpu=True); monitor.admission = dict(contract_id='recipe-a')
    state = TrainerState(tmp_path, model, opt, sched, None, torch.Generator(), dl, ['a', 'b'], 2, monitor)
    state.epoch_loader = dl.generator.get_state(); state.save(1, 0, [])
    monitor.admission = dict(contract_id='recipe-b')
    with pytest.raises(ValueError, match='training identity'):
        TrainerState(tmp_path, model, opt, sched, None, torch.Generator(), dl, ['a', 'b'], 2, monitor)


def test_epoch_boundary_restores_loader_generator(tmp_path):
    import torch
    model = torch.nn.Linear(2, 1); opt = torch.optim.AdamW(model.parameters()); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    ds = torch.utils.data.TensorDataset(torch.zeros(2, 2)); ds.epoch = 0
    dl = torch.utils.data.DataLoader(ds, batch_size=1, generator=torch.Generator().manual_seed(3))
    monitor = Monitor(tmp_path, cpu=True)
    state = TrainerState(tmp_path, model, opt, sched, None, torch.Generator(), dl, ['a', 'b'], 2, monitor)
    list(dl); expected = dl.generator.get_state().clone(); state.save(1, 0, [])
    dl.generator.manual_seed(3)
    loaded = TrainerState(tmp_path, model, opt, sched, None, torch.Generator(), dl, ['a', 'b'], 2, monitor)
    loaded.begin_epoch(1, ds)
    assert torch.equal(loaded.epoch_loader, expected)


def test_checkpoint_heartbeats_do_not_hide_throughput_collapse():
    guard = ThroughputGuard(0)
    for t in range(0, 241, 15): guard.observe(dict(phase='train', windows=t * 100), t)
    with pytest.raises(ValueError, match='throughput collapse'):
        for t in range(255, 2400, 15):
            guard.observe(dict(phase='checkpoint', windows=24000), t - 1)
            guard.observe(dict(phase='train', windows=24000 + (t - 240) * 5), t)


def test_completed_model_must_match_contract_epochs_and_hash(tmp_path):
    (tmp_path / 'models').mkdir(); (tmp_path / 'models/E_prior_s0_f0.pt').write_bytes(b'model fixture')
    reg = dict(contract_id='sealed', recipe=dict(epochs=10))
    rec = dict(status='COMPLETE', epochs=10, model_fold=0, train_studies=3190, contract_id='sealed', model_sha256=k.sha(tmp_path / 'models/E_prior_s0_f0.pt'))
    k.write(tmp_path / 'fit-complete.json', rec)
    assert k.completed_fit(reg, tmp_path) == rec
    with pytest.raises(ValueError): k.completed_fit(dict(reg, contract_id='other'), tmp_path)
    k.write(tmp_path / 'fit-complete.json', dict(rec, epochs=16))
    with pytest.raises(ValueError): k.completed_fit(reg, tmp_path)
    k.write(tmp_path / 'fit-complete.json', rec); (tmp_path / 'models/E_prior_s0_f0.pt').write_bytes(b'changed')
    with pytest.raises(ValueError): k.completed_fit(reg, tmp_path)


def test_new_paths_queue_and_v1_evidence_separation():
    assert k.TAG == 'L1_RCO_K1_F0_V2' != v1.TAG and k.QUEUE != v1.QUEUE and k.WORK != v1.WORK
    assert v1.WORK not in k.OUT.parents and k.OUT == k.ROOT / '.local/scratch/l1-rco/k1v2/fit'
    reg = registration(); argv = k.trainer_argv(k.CACHE, reg)
    for path in (argv[0], argv[1], *[argv[argv.index(flag) + 1] for flag in ('--px', '--out', '--targets', '--init')]): assert Path(path).is_absolute()
    assert argv[argv.index('--only-folds') + 1] == '0' and '--resume' in argv
    plan = k.read(k.QUEUE)
    assert plan == [dict(tag=k.TAG, command=k.run_command(k.CACHE))]
    command = plan[0]['command']
    for path in (command[0], command[1], command[command.index('--cache') + 1], command[command.index('--review') + 1]): assert Path(path).is_absolute()
    assert command[1].endswith('l1_rco_k1_v2.py') and command[-1].endswith('K1V2-REVIEW.json')
    from tools.gpu_queue import validate_plan
    validate_plan(plan)
    assert reg['parent']['tag'] == v1.TAG and reg['parent']['registration_sha256'] == k.sha(v1.WORK / 'registration.json')
