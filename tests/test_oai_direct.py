"""Direct OAI API v1.2.0: real private register/validate card; canonical bytes preserved."""
import ast
import copy
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DistributedSampler
from tools import exp190_coatnet_oai_f0 as arm
import exp155_own0949_oai_v11 as fit

ROOT = arm.ROOT
BACKUP = ROOT/'.local/backups/refactor-step7-invocation9/tools/exp190_coatnet_oai_f0.py'


class Fake:
    def __init__(self, augment=True):
        self.ids = [f'R{i}' for i in range(3455)] + [f'O{i}' for i in range(2175)]
        self.rsna_ids, self.oai_ids = set(self.ids[:3455]), set(self.ids[3455:])
        self.rsna = SimpleNamespace(augment=augment, epoch=torch.zeros(1, dtype=torch.int64))
        self.oai = SimpleNamespace(augment=augment, epoch=torch.zeros(1, dtype=torch.int64))
    def __len__(self): return len(self.ids)
    def __getitem__(self, index): return self.ids[index]
    def set_epoch(self, epoch):
        self.rsna.epoch.fill_(epoch)
        self.oai.epoch.fill_(epoch)


def legacy_sampler():
    tree = ast.parse(BACKUP.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in ('FixedStepMixedDataset', 'fixed_draw_counts')]
    ns = dict(bridge=arm.bridge, lru_cache=lru_cache)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BACKUP), 'exec'), ns)
    cls = ns['FixedStepMixedDataset']
    cls.original_class = Fake
    return cls


@pytest.mark.parametrize('epoch', range(16))
def test_all_epochs_match_legacy_draw_and_loader_order(epoch):
    old, new = legacy_sampler()(), arm.FixedStepMixedDataset(Fake())
    old.set_epoch(epoch); new.set_epoch(epoch)
    assert old.selection(epoch) == new.selection(epoch)
    a = DistributedSampler(old, 1, 0, shuffle=True, seed=42, drop_last=True)
    b = DistributedSampler(new, 1, 0, shuffle=True, seed=42, drop_last=True)
    a.set_epoch(epoch); b.set_epoch(epoch)
    assert [old[i] for i in a] == [new[i] for i in b]
    assert new.draws == dict(per_epoch=3455, rsna=2120, oai=1335,
                             updates_per_epoch=216, total_updates=3456)
    old_bn, new_bn = legacy_sampler()(False), arm.FixedStepMixedDataset(Fake(False))
    assert len(new_bn) == 5630
    assert [old_bn[i] for i in range(5630)] == [new_bn[i] for i in range(5630)]


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.2))
        self.head = nn.Linear(4, 12)
    def forward(self, x): return self.head(self.backbone(x.reshape(x.shape[0], 3)))


def stack():
    fit._cpu_seed(42)
    model = Tiny()
    recipe = dict(bb_lr=.003, head_lr=.006, wd=.02, pct_start=.3, clip=3., epochs=16)
    return model, *fit.make_optimization(model, recipe, 3456, 'none'), recipe


def batches(count=31):
    g = torch.Generator().manual_seed(58)
    data = []
    for i in range(count):
        mask = torch.ones(1, 12) if i%2 else torch.zeros(1, 12)
        mask[:, [5, 6, 8]] = 1
        data.append((torch.randn(1,1,3,1,1,generator=g),
                     torch.rand(1,12,generator=g), mask, ['fixture']))
    return data


def assert_nested(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_nested(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested(x, y)
    else: assert a == b


@pytest.mark.parametrize('count', [31, 3455])
def test_masked_adamw_onecycle_and_tail_match_reference(count):
    model, opt, sched, scaler, recipe = stack()
    ref, ro, rs, _, _ = stack()
    data, weights = batches(count), torch.linspace(.5, 2, 12)
    fit._cpu_seed(42000)
    actual = fit.train_epoch(model, data, opt, sched, scaler, weights, recipe, 'cpu', 'none', 16)
    fit._cpu_seed(42000)
    for start in range(0, count, 16):
        group = data[start:start+16]
        ro.zero_grad(set_to_none=True)
        for x, y, mask, _ in group:
            loss = nn.functional.binary_cross_entropy_with_logits(ref(x), y, pos_weight=weights, reduction='none')
            ((loss*mask).sum()/12/len(group)).backward()
        nn.utils.clip_grad_norm_(ref.parameters(), recipe['clip'], error_if_nonfinite=True)
        ro.step(); rs.step()
    assert actual['updates'] == (count+15)//16 and actual['batches'] == count and actual['skips'] == 0
    assert_nested(model.state_dict(), ref.state_dict())
    assert_nested(opt.state_dict(), ro.state_dict())
    assert_nested(sched.state_dict(), rs.state_dict())
    assert scaler.state_dict() == {}


def test_epoch_boundary_resume_preserves_optimizer_and_predictions(tmp_path):
    model, opt, sched, scaler, recipe = stack()
    data, weights = batches(), torch.ones(12)
    fit._cpu_seed(42000)
    first = fit.train_epoch(model, data, opt, sched, scaler, weights, recipe, 'cpu', 'none', 16)
    path = tmp_path/'epoch.pt'
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                    scaler=scaler.state_dict(), epoch=1, updates=first['updates'], skips=0), path)
    fit._cpu_seed(42001)
    expected = fit.train_epoch(model, data, opt, sched, scaler, weights, recipe, 'cpu', 'none', 16,
                              updates=first['updates'])
    resumed, oo, ss, sc, _ = stack()
    saved = torch.load(path, weights_only=True)
    resumed.load_state_dict(saved['model'], strict=True)
    oo.load_state_dict(saved['optimizer']); ss.load_state_dict(saved['scheduler']); sc.load_state_dict(saved['scaler'])
    fit._cpu_seed(42001)
    actual = fit.train_epoch(resumed, data, oo, ss, sc, weights, recipe, 'cpu', 'none', 16,
                            updates=saved['updates'], skips=saved['skips'])
    assert actual == expected
    assert_nested(model.state_dict(), resumed.state_dict())
    assert_nested(opt.state_dict(), oo.state_dict())
    assert_nested(sched.state_dict(), ss.state_dict())
    assert_nested(scaler.state_dict(), sc.state_dict())
    assert torch.equal(model.eval()(data[0][0]), resumed.eval()(data[0][0]))


def test_nonfinite_loss_fails_before_update():
    model, opt, sched, scaler, recipe = stack()
    data = batches(1); data[0][0].fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='nonfinite loss'):
        fit.train_epoch(model, data, opt, sched, scaler, torch.ones(12), recipe, 'cpu', 'none', 16)
    assert sched.last_epoch == 0 and not opt.state


def test_skipped_update_does_not_advance_scheduler():
    model, opt, sched, _, recipe = stack()
    class Skip:
        value = 128.
        def scale(self, value): return value
        def unscale_(self, _): pass
        def get_scale(self): return self.value
        def step(self, _): pass
        def update(self): self.value /= 2
    result = fit.train_epoch(model, batches(17), opt, sched, Skip(), torch.ones(12), recipe,
                             'cpu', 'none', 16, total_steps=3456)
    assert result['skips'] == 2 and result['updates'] == 0 and sched.last_epoch == 0
    with pytest.raises(FloatingPointError, match='skipped-update'):
        fit.train_epoch(model, batches(1), opt, sched, Skip(), torch.ones(12), recipe,
                         'cpu', 'none', 16, total_steps=99)


def test_train_admission_fails_before_gpu_or_attempt(monkeypatch):
    monkeypatch.setattr(arm, 'admit', lambda args: dict(status='HOLD'))
    def forbidden(*args): raise AssertionError('GPU gate must not be reached')
    monkeypatch.setattr(arm.bridge, 'require_gpu_job', forbidden)
    with pytest.raises(ValueError, match='not launch-qualified'):
        arm.train(SimpleNamespace())


def test_readout_float32_order_and_mean_match_legacy(monkeypatch, tmp_path):
    values = [(np.linspace(.01,.91,12,dtype=np.float32),np.linspace(.07,.98,12,dtype=np.float32)),
              (np.full(12,.3,dtype=np.float32),np.full(12,.7,dtype=np.float32))]
    monkeypatch.setattr(arm.bridge, 'infer_pair', lambda model,i,**kwargs: values[i])
    corpus = SimpleNamespace(get=lambda uid: (int(uid),))
    a,b = arm.infer_readout(None, corpus, ['1','0'], 'cpu', 'none', 2)
    assert a.dtype == b.dtype == np.float32
    old = np.stack([values[i][0] for i in (1,0)])*.5 + np.stack([values[i][1] for i in (1,0)])*.5
    assert np.array_equal((a+b)*.5, old)
    path = tmp_path/'predictions.csv'
    arm.bridge.base.write_predictions(path, ['1','0'], (a+b)*.5)
    assert np.array_equal(arm.bridge.load_oof(path, ['1','0']).astype(np.float32), (a+b)*.5)


def test_removed_time_policy_is_not_executed():
    assert fit.scientific_execution(dict(batch=1,max_hours=0,reserve_seconds=10**9)) == dict(batch=1)
    for name in (fit.__file__, arm.__file__, ROOT/'training/arms/oai_smoke.py'):
        tree = ast.parse(Path(name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not any(k.arg == 'timeout' for k in node.keywords)
                assert not (isinstance(node.func, ast.Attribute) and node.func.attr in ('Timer','alarm'))
        assert not any(isinstance(n, ast.Import) and any(x.name=='dis' for x in n.names) for n in ast.walk(tree))


def test_real_direct_cpu_cli_card():
    from oai_registration_fixture import route, parity_and_canonical_rejection
    """Source-bound registration + actual CoAtNet train/BN/K94 reload, CPU only."""
    root = Path(os.environ.get('OAI_SMOKE_OUT', ROOT/f'.local/refactor/oai-card-{uuid.uuid4().hex}'))
    root.mkdir(parents=True, exist_ok=False)
    commands = []
    def run(argv, code, status=None):
        command = [sys.executable, '-B', str(ROOT/'tests/oai_registration_fixture.py'),
                   str(root/'registration-v1'), *map(str, argv)]
        log = root/f'command-{len(commands):02d}.log'
        with log.open('x', encoding='utf-8') as stream:
            done = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
        commands.append(dict(argv=command, exit_code=done.returncode, expected_exit=code,
                             log=str(log.relative_to(ROOT)).replace('\\','/'), log_sha256=arm.HASH(log)))
        (root/'command-results.json').write_text(json.dumps(commands,indent=2), encoding='utf-8')
        assert done.returncode == code, f'Inspect {log}'
        if status: assert status in log.read_text(encoding='utf-8')
    run(['register'], 0, 'REGISTERED_NO_TRAINING')
    run(['validate'], 0, 'VALIDATED_NO_TRAINING')
    p = parity_and_canonical_rejection(root/'registration-v1')
    with route(root/'registration-v1'):
        arm.require_validation()
    run(['admit'], 0, 'HOLD')
    run(['smoke','--device','cpu','--workers','0','--out',root/'smoke'], 0, 'ENGINEERING_SMOKE_COMPLETE')
    receipt = json.loads((root/'smoke/receipt.json').read_text(encoding='utf-8'))
    assert receipt['full_bn_count'] == len(p['train_ids']) and receipt['optimizer']['updates'] == 1
    assert receipt['oai_exclusion'] == {k:v for k,v in p['oai_exclusion'].items() if k != 'participants'}
    assert receipt['results']['real_data'] and not receipt['cuda_initialized']
    assert receipt['verdict'] == 'HOLD' and not receipt['launch_authorized']
    before = arm.HASH(root/'smoke/receipt.json')
    run(['smoke','--device','cpu','--workers','0','--out',root/'smoke'], 1, 'smoke output exists')
    assert arm.HASH(root/'smoke/receipt.json') == before
