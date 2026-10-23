"""H-B-105 v1.0.0: real-data CPU cards, frozen parity and failure contracts.

Never fabricate target PASS; actual smoke evidence stays HOLD.
"""
import ast
import copy
import gc
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from training.arms import lifecycle as life, mil, mil_small as small
from tools import hb105_mil_small as ops

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'


def cli(module, *args):
    command = [sys.executable, '-B', '-m', module, *map(str, args)]
    env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1')
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return life.parse_json(result.stdout)


@pytest.mark.parametrize('seed', [42, 43])
def test_only_backbone_changed(seed):
    old = life.ArmSpec((ROOT / f'training/configs/mil-s{seed}-f0-v3.json').read_bytes())
    spec = ops.spec_for(ROOT, seed)
    a, b = old.document['config'], spec.document['config']
    a.pop('model'); b.pop('model')
    assert a == b
    assert spec.document['config']['model']['arch'] == small.ARCH
    assert (spec.document['seed'], spec.document['fold']) == (seed, 0)
    _, train, val, _, _ = small.validate_data(spec, ROOT, CORPUS)
    assert (len(train), len(val)) == (3455, 888)
    assert spec.document['target_environment']['hostname'] == '44a123d2ba2a'


def test_frozen_fit_function_only_retention_and_metadata_differ():
    old = inspect.getsource(mil.run).replace('save_epoch(work /', 'save_latest_epoch(work /')
    old = old.replace('"arch": ARCH', '"arch": config["model"]["arch"]')
    assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(inspect.getsource(small.run)))
    for name in ('build_model', 'FindingMIL', 'MILWindows', 'optimizer_stack', 'train_epoch',
                 'evaluate', 'record_smoke', 'restore_epoch', 'positive_weights'):
        assert getattr(small, name) is getattr(mil, name)


@pytest.mark.parametrize('change', ['head', 'arch', 'init', 'seed', 'labels', 'lr', 'crop', 'steps'])
def test_control_mismatch_rejected(change):
    spec = ops.spec_for(ROOT, 42)
    summary, provenance = small.control_records(spec, ROOT)
    if change == 'head': summary['exp185']['model_kind'] = 'global'
    if change == 'arch': summary['args']['arch'] = small.ARCH
    if change == 'init': provenance['initial_state_sha256'] = '0' * 64
    if change == 'seed': summary['args']['seed'] = 43
    if change == 'labels': provenance['label_semantic_sha256'] = '0' * 64
    if change == 'lr': summary['args']['bb_lr'] *= 2
    if change == 'crop': provenance['exp179']['nominal_mm'] = [140]
    if change == 'steps': summary['optimizer_updates'] -= 1
    with pytest.raises(life.ContractError):
        small.validate_control(spec, summary, provenance)


@pytest.mark.parametrize('section,key,value', [('windows', 'train_k', 24), ('updates', 'epochs', 21),
    ('sampler', 'batch', 4), ('model', 'arch', mil.ARCH), ('readout', 'selection', 'best')])
def test_recipe_mismatch_rejected(section, key, value):
    doc = ops.spec_for(ROOT, 42).document
    doc['config'][section][key] = value
    with pytest.raises(life.ContractError): small.validate_recipe(life.ArmSpec.from_document(doc))


def test_small_real_initial_rng_and_logits_match_legacy():
    from training.src.raptor_exp185 import install_spatial_pooling
    torch.set_num_threads(2)
    spec = ops.spec_for(ROOT, 42)
    config = spec.document['config']['model']
    mil.seed_all(42)
    current = small.build_model(config, ROOT).eval()
    rng = torch.get_rng_state().clone()
    mil.seed_all(42)
    legacy = install_spatial_pooling(mil.common.build_model(small.ARCH, pretrained=False)).eval()
    assert torch.equal(rng, torch.get_rng_state())
    initial = torch.load(ROOT / config['initial_state'], map_location='cpu', weights_only=True)
    small.strict_load(legacy, initial)
    assert current.backbone.num_features == 768 and current.clsW.shape == (12, 768)
    _, train, _, labels, _ = small.validate_data(spec, ROOT, CORPUS)
    dataset = small.MILWindows(CORPUS, train[:1], labels, 12, True, 42)
    x = dataset[0][0][None]
    with torch.no_grad():
        torch.testing.assert_close(current(x), legacy(x), rtol=0, atol=0)
    del current, legacy, initial, x
    gc.collect()
    assert not torch.cuda.is_initialized()


def test_checkpoint_retention_resume_and_failed_publish(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    generator = torch.Generator().manual_seed(42)
    identity = {'test': 'owned'}
    kw = dict(model=model, opt=opt, sched=sched, scaler=scaler, generator=generator,
        history=[], counters={'updates': 0, 'skips': 0}, identity=identity, device='cpu')
    path0, path1 = tmp_path/'epoch-000.pt', tmp_path/'epoch-001.pt'
    small.save_latest_epoch(path0, epoch=0, **kw)
    real_save = mil.save_epoch
    def fail(*a, **k): raise OSError('simulated failed publish')
    monkeypatch.setattr(mil, 'save_epoch', fail)
    with pytest.raises(OSError): small.save_latest_epoch(path1, epoch=1, **kw)
    assert path0.exists() and path0.with_suffix('.receipt.json').exists()
    monkeypatch.setattr(mil, 'save_epoch', real_save)
    small.save_latest_epoch(path1, epoch=1, **kw)
    assert list(tmp_path.glob('*.pt')) == [path1]
    assert not path0.with_suffix('.receipt.json').exists()
    resume = {k: kw[k] for k in ('model','opt','sched','scaler','generator','identity','device')}
    assert small.restore_epoch(path1, **resume)[0] == 2
    path1.write_bytes(path1.read_bytes()+b'tampered')
    with pytest.raises(life.ContractError): small.restore_epoch(path1, **resume)


@pytest.mark.parametrize('seed', [42, 43])
def test_real_cpu_launch_card(seed, tmp_path):
    spec = ops.spec_for(ROOT, seed)
    spec_path = ROOT / ops.config_path(seed)
    registered, stage = tmp_path/'registered.json', tmp_path/'stage'
    assert cli('tools.arm_lifecycle','register','--draft',spec_path,'--root',ROOT,'--out',registered)['status'] == 'REGISTERED_NO_EXECUTION'
    assert registered.read_bytes() == spec_path.read_bytes()
    cli('tools.arm_lifecycle','validate','--spec',registered,'--root',ROOT)
    cli('tools.arm_lifecycle','prepare','--spec',registered,'--root',ROOT,'--out',stage)
    cli('tools.arm_lifecycle','validate','--spec',registered,'--root',stage,'--stage')
    cli('training.runners.mil_small_arms','validate','--spec',registered,'--stage',stage,'--corpus',CORPUS)
    out = tmp_path/f'cpu-s{seed}'
    args = ('--spec',registered,'--stage',stage,'--corpus',CORPUS,'--device','cpu','--workers','0')
    result = cli('training.runners.mil_small_arms','smoke',*args,'--out',out)
    assert result['results']['optimizer_steps'] == 1
    assert result['results']['validation_batches'] == 3
    assert ops.read(out/'outputs/summary.json')['cpu_cuda_initialized'] is False
    ck = torch.load(out/'outputs/model_best.pt', map_location='cpu', weights_only=False)
    assert ck['arch'] == small.ARCH and ck['epoch'] == 0 and ck['model_kind'] == 'finding_spatial_tokens_v1'
    del ck; gc.collect()
    note = out/'verification.json'
    assert cli('tools.arm_lifecycle','admit','--dry-run','--spec',registered,'--stage',stage,
        '--note',note,'--evidence-root',out)['status'] == 'HOLD'
    resumed = tmp_path/'resume'
    result = cli('training.runners.mil_small_arms','smoke',*args,'--resume',out/'epochs/epoch-000.pt','--out',resumed)
    assert result['results']['optimizer_steps'] == 0
    for p in ops.READOUTS: assert (resumed/'outputs'/p).read_bytes() == (out/'outputs'/p).read_bytes()
    receipt, archive = tmp_path/'receipt.json', tmp_path/'export.tar'
    with pytest.raises(life.ContractError):
        ops.finish(spec,out/'outputs',note,archive,receipt)
    result = cli('tools.hb105_mil_small','--root',ROOT,'finish','--seed',seed,'--outputs',out/'outputs',
        '--note',note,'--archive',archive,'--receipt',receipt,'--engineering-smoke')
    collected = tmp_path/'collected'
    cli('tools.arm_lifecycle','collect','--spec',registered,'--archive',archive,
        '--receipt-sha256',result['receipt_sha256'],'--out',collected)
    assert (collected/'outputs/model_best.pt').read_bytes() == (out/'outputs/model_best.pt').read_bytes()
    # Persistent builder evidence copied only after the complete card passed.
    destination = ROOT / '.local/hb105-validation-v1' / f'cpu-s{seed}'
    import shutil
    if destination.exists():
        life.verify_evidence(spec, ops.read(destination/'verification.json'), destination)
        assert ops.read(destination/'card-result.json')['status'] == 'CPU_CARD_PASSED'
        return  # Preserve prior executed evidence; this invocation also passed.
    shutil.copytree(out, destination)
    ops.save(destination/'card-result.json', {'status':'CPU_CARD_PASSED','seed':seed,
        'register':True,'validate':True,'prepare':True,'family_validate':True,'admit':'HOLD',
        'resume_zero_updates_identical_predictions':True,'export_collect':True,
        'smoke':ops.read(out/'outputs/summary.json')})


def test_control_replay_specs_are_pinned(tmp_path):
    result = ops.screen_specs(ROOT,None,tmp_path/'screen',replay=True)
    assert len(result['specs']) == 3
    for p in result['specs']:
        doc = ops.read(p)
        for pair in doc['seeds'].values(): assert pair['arm'][0]['sha256'] == pair['control'][0]['sha256']


def test_old_configs_still_validate():
    for seed in (42,43):
        spec = life.ArmSpec((ROOT/f'training/configs/mil-s{seed}-f0-v3.json').read_bytes())
        mil.validate_data(spec,ROOT,CORPUS)


def test_combine_rejects_cpu_as_target(tmp_path):
    spec = ops.spec_for(ROOT,42)
    cpu = ROOT/'.local/hb105-validation-v1/cpu-s42'
    with pytest.raises(life.ContractError, match='two original smokes required'):
        ops.combine(spec,cpu,cpu,tmp_path/'pass','B','Claude Code')
    assert not (tmp_path/'pass').exists()
