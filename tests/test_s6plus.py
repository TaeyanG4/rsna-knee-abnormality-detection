"""CPU contracts for H-B-108; no label/report rows or network/GPU access."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from tools import s6plus as s
import torch

torch.set_num_threads(2)


def schema():
    return {'labels': s.TARGETS, 'categories': {t: ['a', 'b'] if j else ['a', 'b', 'c']
            for j, t in enumerate(s.TARGETS)}, 'binaries': {'b': s.TARGETS[:2]},
            'prior': {t: [.1, .9] if j else [.1, .5, .9] for j, t in enumerate(s.TARGETS)},
            'base': dict.fromkeys(s.TARGETS, .5)}


def groups():
    return [{s.UID: u, 'component_id': u, 'report_group_sha256': u, 'split': split}
            for split, u in [('train', 'a'), ('train', 'b'), ('val', 'c')]]


@pytest.mark.parametrize('kind', ['study', 'component_id', 'report_group_sha256', 'split', 'corrupt'])
def test_split_rejects_leakage(kind):
    split = {'train': ['a', 'b'], 'val': ['c']}
    g = groups()
    if kind == 'study': split['val'] = ['a']
    elif kind == 'split': g[2]['split'] = 'train'
    elif kind == 'corrupt': split['train'][0] = next(iter(s.CORRUPT))
    else: g[2][kind] = 'a'
    with pytest.raises(ValueError): s.check_split(split, g, expected=False)


def test_valid_split():
    s.check_split({'train': ['a', 'b'], 'val': ['c']}, groups(), expected=False)


def test_prior_replaces_public_pool_with_training_only():
    sc = schema()
    x = np.linspace(.02, .98, 36).reshape(3, 12)
    prior, clean = s.training_prior(x, sc)
    assert np.allclose(prior, x.mean(0))
    for j, t in enumerate(s.TARGETS):
        assert clean['prior'][t] == [prior[j]] * len(sc['categories'][t])
        assert clean['base'][t] == prior[j]
    assert sc == schema()  # original teacher schema remains immutable


def shard(tmp_path, uid='a', size=12):
    rng = np.random.default_rng(3)
    px = rng.integers(0, 255, (11, size, size), dtype=np.uint8)
    meta = np.zeros((11, 11), dtype=np.float32)
    meta[:, 1] = 1; meta[:, 3] = 0; meta[:, 4] = np.arange(11)/11
    meta[:, 5] = np.log1p(2.)
    blocks = [{'series': 'x', 'plane': 'Sagittal', 'n_slices': 11, 'views': [[0, 0]]}]
    p, j = s.cache_paths(tmp_path, uid)
    np.savez_compressed(p, px=px, meta=meta)
    rec = {'uid': uid, 'blocks': blocks, 'bytes': p.stat().st_size, 'sha256': s.sha(p),
           'contract_id': 'test', 'raw_identity': {'files': 11}}
    s.write(j, rec)
    return px, meta, blocks


@pytest.mark.parametrize('mode,epoch', [('train', 0), ('train', 3), ('eval', 0)])
def test_shards_equal_public_studies_windows(tmp_path, mode, epoch):
    px, meta, blocks = shard(tmp_path)
    np.save(tmp_path/'px.u8.npy', px); np.save(tmp_path/'meta.f32.npy', meta)
    s.write(tmp_path/'index.json', {'blocks': {'a': blocks}})
    recipe = {**s.DEFAULT['recipe'], 'train_windows': 4}
    old = s.compact().Studies(str(tmp_path), ['a'], 4, mode, 9000,
                             eval_stride=1, slices=3, phys_mm=3.5, train_sampling='uniform', extent_meta=True)
    old.epoch = epoch
    new = s.Shards(tmp_path, ['a'], mode, recipe, epoch)
    for a, b in zip(old[0][:2], new[0][:2]): assert torch.equal(a, b)


def test_cache_tamper_rejected(tmp_path):
    shard(tmp_path)
    r = {'contract_id': 'test', 'raw': {'a': {'files': 11}}}
    s.check_cache(r, tmp_path, ['a'])
    p, _ = s.cache_paths(tmp_path, 'a')
    data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
    with pytest.raises(ValueError, match='hash'): s.check_cache(r, tmp_path, ['a'])


def test_categorical_kl_binary_bce_and_mask():
    sc = schema()
    valid = torch.tensor([[True]*3] + [[True, True, False]]*11)
    sl = torch.randn(2, 12, 3, requires_grad=True)
    bl = torch.randn(2, 2, requires_grad=True)
    q = torch.softmax(torch.randn(2, 12, 3).masked_fill(~valid, -torch.inf), -1)
    b = torch.full((2, 2), .3)
    got = s.distill_loss(sl, bl, q, b, valid, 2.)
    manual = sum(torch.nn.functional.kl_div(torch.log_softmax(sl[:, j, valid[j]]/2, -1),
                 q[:, j, valid[j]], reduction='batchmean') for j in range(12))/12
    manual = 2*(manual + torch.nn.functional.binary_cross_entropy_with_logits(bl/2, b))
    torch.testing.assert_close(got, manual)
    got.backward()
    assert torch.isfinite(sl.grad).all() and (sl.grad[:, ~valid] == 0).all()
    s.check_teacher_arrays(q[0].numpy(), b[0].numpy(), sc)
    bad = q[0].numpy().copy(); bad[1, 2] = .01
    with pytest.raises(ValueError, match='mass/mask'): s.check_teacher_arrays(bad, b[0].numpy(), sc)


def test_teacher_heldout_forbidden(tmp_path):
    r = {'split': {'train': ['a'], 'val': ['v']}}
    with pytest.raises(ValueError, match='holdout'): s.load_teacher(r, tmp_path, tmp_path, ['v'], False)


def test_teacher_full_cannot_use_smoke(tmp_path):
    shard(tmp_path)
    sc = schema()
    r = {'contract_id': 'test', 'split': {'train': ['a'], 'val': ['v']}, 'state_spec': sc,
         'config': s.DEFAULT, 'bundle': {f'E_lrprior_s0_f{f}.pt': 'weight' for f in range(4)}}
    for fold in range(4):
        p = s.teacher_path(tmp_path/'teacher', 'a', fold); p.parent.mkdir(parents=True)
        q = np.zeros((12, 3), dtype='float32'); q[0] = 1/3; q[1:, :2] = .5
        np.savez_compressed(p, categories=q, binaries=np.full(2, .5, dtype='float32'))
        s.write(p.with_suffix('.json'), {**s.teacher_metadata(r, 'a', tmp_path, fold, True), 'sha256': s.sha(p)})
    q, b = s.load_teacher(r, tmp_path/'teacher', tmp_path, ['a'], True)
    assert q.shape == (1, 12, 3) and b.shape == (1, 2)
    with pytest.raises(ValueError, match='scope'): s.load_teacher(r, tmp_path/'teacher', tmp_path, ['a'], False)


@pytest.mark.parametrize('spatial', [False, True])
def test_chunked_forward_matches_upstream_and_backprop(spatial):
    # Actual S6 ConvNeXt and MIL head, reduced spatial size for a unit check.
    m = s.compact().Net('convnext_small', '', state_spec=schema(), spatial=spatial, extent=spatial).eval()
    x = torch.randint(255, (1, 2, 3, 32, 32), dtype=torch.uint8)
    meta = torch.zeros(1, 2, 9 if spatial else 6); mask = torch.ones(1, 2, dtype=torch.bool)
    with torch.no_grad():
        old = m(x, meta, mask, chunk=2, return_aux=True)
        new = s.forward(m, x, meta, mask, chunk=1)
    for a, b in zip(old, new): torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    z, _, _ = s.forward(m, x, meta, mask, chunk=1, train=True)
    z.sum().backward()
    assert m.enc.stem[0].weight.grad.abs().sum() > 0
    assert not torch.cuda.is_initialized()


def test_public_weight_architecture_and_pretrained_norm():
    # Actual published checkpoint contract: spatial/extent flags are not in manifest.
    from safetensors.torch import load as load_bytes
    spec = s.read(s.ROOT/s.BUNDLE/'state_spec.json')
    prior, clean = s.training_prior(np.full((2, 12), .3), spec)
    student = s.make_model(s.DEFAULT, clean, prior)
    assert student.spatial and student.extent and not student.state_gate.requires_grad
    sd = load_bytes(s.input_path(s.DEFAULT, 'init').read_bytes())
    assert torch.equal(student.sp_head_norm.weight, sd['head.norm.weight'])
    assert torch.equal(student.sp_head_norm.bias, sd['head.norm.bias'])
    assert torch.all(student.state_prior[student.state_valid] == .3)
    del student, sd
    teacher = s.make_model(s.DEFAULT, spec, teacher=True)
    for fold in range(4):
        sd = torch.load(s.ROOT/s.BUNDLE/f'E_lrprior_s0_f{fold}.pt', map_location='cpu', weights_only=True)
        teacher.load_deploy_state_dict(sd)  # strict published key/shape coverage
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('kind', ['roster', 'shape', 'nan', 'range'])
def test_oof_rejects_invalid(tmp_path, kind):
    ids = ['a', 'b']; p = np.full((2, 12), .5)
    if kind == 'roster': ids.reverse()
    if kind == 'shape': p = p[:, :11]
    if kind == 'nan': p[0, 0] = np.nan
    if kind == 'range': p[0, 0] = 1.1
    with pytest.raises(ValueError): s.export_oof(tmp_path/'x.csv', ids, p, ['a', 'b'])


def test_oof_readable_by_multi_lens(tmp_path):
    from tools.multi_lens_readout import load_oof
    p = np.full((2, 12), .5)
    s.export_oof(tmp_path/'x.csv', ['b', 'a'], p, ['b', 'a'])
    ids, got, _ = load_oof(tmp_path/'x.csv')
    assert ids == ['a', 'b']; np.testing.assert_array_equal(got, p)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1000)
        self.enc = torch.nn.Linear(1, 4)
        self.head = torch.nn.Linear(4, 12)
        self.drop = torch.nn.Dropout(.2)


def test_real_training_loop_resume_is_bit_exact(tmp_path, monkeypatch):
    # Actual optimizer/scheduler/checkpoint/sampler/augmentation loop, tiny encoder.
    c = copy.deepcopy(s.DEFAULT)
    c['recipe'].update(size=12, crop=10, epochs=2, train_windows=3, checkpoint_every_updates=1)
    ids = ['a', 'b', 'c', 'd']
    for uid in ids: shard(tmp_path, uid)
    r = {'contract_id': 'test', 'config': c, 'split': {'train': ids, 'val': ['v']},
         'state_spec': schema(), 'bias_prior': [.5]*12, 'raw': {u: {'files': 11} for u in ids}}
    monkeypatch.setattr(s, 'contract', lambda *a, **k: r)
    monkeypatch.setattr(s, 'make_model', lambda *a, **k: Tiny())
    monkeypatch.setattr(s, 'labels', lambda *a: np.full((4, 12), .3, dtype='float32'))
    def forward(m, img, meta, mask, chunk, train=False):
        feat = img.float().mean((1, 2, 3, 4))[:, None]/255
        return m.head(m.drop(m.enc(feat))), None, None
    monkeypatch.setattr(s, 'forward', forward)
    s.train('reg', tmp_path, None, tmp_path/'all', 'S-B')
    part = s.train('reg', tmp_path, None, tmp_path/'resumed', 'S-B', stop_after_updates=1)
    assert part['status'] == 'CHECKPOINTED' and part['updates'] == 1
    s.train('reg', tmp_path, None, tmp_path/'resumed', 'S-B', resume=True)
    a = torch.load(tmp_path/'all/latest.pt', weights_only=False)
    b = torch.load(tmp_path/'resumed/latest.pt', weights_only=False)
    assert a['updates'] == b['updates'] == 4
    for name in a['model']: assert torch.equal(a['model'][name], b['model'][name])
    assert a['scheduler'] == b['scheduler']
    assert torch.equal(a['rng']['cpu'], b['rng']['cpu'])
    assert (tmp_path/'resumed/previous.pt').is_file()


def test_admission_rejects_partial_schedule(tmp_path, monkeypatch):
    r = {'contract_id': 'test', 'config': s.DEFAULT, 'split': {'train': ['a']*3455, 'val': ['v']}}
    monkeypatch.setattr(s, 'contract', lambda *a, **k: r)
    for arm in s.DEFAULT['arms']:
        p = tmp_path/arm/'oof.csv'; s.export_oof(p, ['v'], np.full((1, 12), .5), ['v'])
        s.write(str(p)+'.json', {'contract_id': 'test', 'arm': arm, 'scope': 'full', 'sha256': s.sha(p)})
        s.write(p.parent/'train-receipt.json', {'contract_id': 'test', 'arm': arm, 'status': 'CHECKPOINTED'})
    got = s.admit('reg', None, tmp_path, dry_run=True)
    assert not got['admitted'] and len(got['missing_or_invalid']) == 3


def test_contract_tamper_rejected(tmp_path):
    r = {'version': s.VERSION, 'config': s.DEFAULT}
    r['contract_id'] = s.digest(r); r['unregistered'] = True
    s.write(tmp_path/'reg.json', r)
    with pytest.raises(ValueError, match='tamper'): s.contract(tmp_path/'reg.json', verify=False)
