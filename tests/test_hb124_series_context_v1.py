"""Focused H-B-124 tests; synthetic labels only, no network/CUDA or report I/O."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.src.hb124_series_context_v1 import SeriesProvenance, WithinSeriesContext, sha

torch.set_num_threads(2)


def sample():
    torch.manual_seed(7)
    x = torch.randn(2, 6, 3, 8)
    mask = torch.ones(2, 6, 3, dtype=torch.bool)
    meta = dict(series=torch.tensor([[0, 1, 0, 1, 0, -1], [0, 0, 1, 1, 2, -1]]),
                position=torch.tensor([[0., 0., .4, 1., 1., 0.], [0., 1., .2, .8, 0., 0.]]),
                valid=torch.tensor([[True]*5+[False]]*2))
    return x, mask, meta


def active():
    m = WithinSeriesContext(8, 4)
    with torch.no_grad():
        m.out.weight.fill_(.1)
    return m


def test_identity_shape_rng_and_checkpoint():
    x, mask, meta = sample()
    before = torch.random.get_rng_state().clone()
    m = WithinSeriesContext(8, 4)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(m(x, mask, **meta), x)
    clone = WithinSeriesContext(8, 4)
    clone.load_state_dict(m.state_dict(), strict=True)
    encoded = (x.flatten(1, 2), mask.flatten(1, 2))
    out, outmask = clone.encoded(encoded, meta)
    assert torch.equal(out, encoded[0]) and torch.equal(outmask, encoded[1])


def test_construction_does_not_seed_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("context must never change the control CUDA RNG")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    before = torch.random.get_rng_state().clone()
    WithinSeriesContext(8, 4)
    assert torch.equal(before, torch.random.get_rng_state())


def test_cpu_autocast_finite_backward():
    x, mask, meta = sample()
    x.requires_grad_()
    m = active()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m(x, mask, **meta)
        loss = y.square().mean()
    loss.backward()
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_existing_optimizer_preserves_groups_and_schedule():
    from training.arms import mil
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(8, 8)
    model.head = torch.nn.Linear(8, 12)
    control, schedule, _ = mil.optimizer_stack(model, 100, "cpu", "none")
    model.series_context = WithinSeriesContext(8, 4)
    arm, arm_schedule, _ = mil.optimizer_stack(model, 100, "cpu", "none")
    assert schedule.state_dict() == arm_schedule.state_dict()
    assert [id(p) for p in control.param_groups[0]["params"]] == [id(p) for p in arm.param_groups[0]["params"]]
    assert {id(p) for p in arm.param_groups[1]["params"]} == {
        id(p) for p in list(model.head.parameters()) + list(model.series_context.parameters())}
    for left, right in zip(control.param_groups, arm.param_groups):
        assert {k: v for k, v in left.items() if k != "params"} == {
            k: v for k, v in right.items() if k != "params"}


def test_no_cross_series_or_cross_study_influence():
    x, mask, meta = sample()
    m = active()
    y = m(x, mask, **meta)
    changed = x.clone()
    changed[0, [1, 3, 5]] = torch.randn_like(changed[0, [1, 3, 5]]) * 100
    changed[1] = torch.randn_like(changed[1]) * 100
    z = m(changed, mask, **meta)
    assert torch.equal(y[0, [0, 2, 4]], z[0, [0, 2, 4]])


def test_neighbors_and_normalized_gaps():
    _, _, meta = sample()
    lw, rw, gaps, has = WithinSeriesContext.neighbor_weights(**meta)
    assert lw[0, 2, 0] == rw[0, 2, 4] == 1
    assert lw[0, 0].sum() == rw[0, 4].sum() == 0
    torch.testing.assert_close(gaps[0, 2], torch.tensor([-.4, 0., .6]))
    assert not has[1, 4] and not has[:, 5].any()


def test_active_gradients_and_masking():
    x, mask, meta = sample()
    x.requires_grad_()
    m = active()
    m(x, mask, **meta)[0, 2].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert x.grad[0, 0].abs().sum() > 0 and x.grad[0, 4].abs().sum() > 0
    assert x.grad[0, [1, 3, 5]].abs().sum() == 0 and x.grad[1].abs().sum() == 0


def test_zero_init_projection_learns_first_then_conv():
    x, mask, meta = sample()
    m = WithinSeriesContext(8, 4)
    optimizer = torch.optim.SGD(m.parameters(), lr=.1)
    m(x, mask, **meta).square().mean().backward()
    assert m.out.weight.grad.abs().sum() > 0
    assert torch.equal(m.conv.weight.grad, torch.zeros_like(m.conv.weight.grad))
    optimizer.step()
    optimizer.zero_grad()
    m(x, mask, **meta).square().mean().backward()
    assert m.conv.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_empty_singleton_invalid_and_partial_token_masks():
    x, mask, meta = sample()
    m = active()
    mask[0, 0, 1:] = False
    output = m(x, mask, **meta)
    assert torch.equal(output[0, 0, 1:], x[0, 0, 1:])
    assert torch.equal(output[:, 5], x[:, 5]) and torch.equal(output[1, 4], x[1, 4])
    meta["valid"][:] = False
    meta["series"][:] = -1
    assert torch.equal(m(x, mask, **meta), x)


def test_permutation_and_duplicate_source_positions():
    x, mask, meta = sample()
    meta["position"][0, 2] = 0
    lw, rw, gaps, _ = WithinSeriesContext.neighbor_weights(**meta)
    assert lw[0, 4, 0] == lw[0, 4, 2] == .5
    assert lw[0, 2].sum() == 0 and rw[0, 2, 4] == 1
    m = active()
    perm = torch.tensor([4, 2, 5, 0, 3, 1])
    y = m(x, mask, **meta)
    z = m(x[:, perm], mask[:, perm], **{k: v[:, perm] for k, v in meta.items()})
    torch.testing.assert_close(y[:, perm], z)


@pytest.mark.parametrize("broken", ["shape", "dtype", "position", "series"])
def test_reject_bad_metadata(broken):
    x, mask, meta = sample()
    if broken == "shape":
        meta["valid"] = meta["valid"][:, :-1]
    if broken == "dtype":
        meta["series"] = meta["series"].float()
    if broken == "position":
        meta["position"][0, 0] = float("nan")
    if broken == "series":
        meta["series"][0, 0] = -1
    with pytest.raises(ValueError):
        WithinSeriesContext(8)(x, mask, **meta)


def fake_corpus(tmp_path, *, missing=False, duplicate=False):
    np.save(tmp_path / "ids.npy", np.array(["synthetic"]))
    mask = np.ones((1, 6), np.uint8)
    np.save(tmp_path / "masks.npy", mask)
    row = dict(uid="synthetic", shard=0, row=0, slots=[
        {"series": "real-a", "files": 5, "slices": [0, 2, 4]},
        None if missing else {"series": "real-b", "files": 3, "slices": [0, 1, 2]}])
    (tmp_path / "picks.jsonl").write_text((json.dumps(row) + "\n") * (2 if duplicate else 1))
    return SimpleNamespace(dir=tmp_path, row={"synthetic": 0}, depth=6,
        spec={"slots": [["a", 1, 3], ["b", 0, 3]]}, mask=lambda _: mask[0],
        manifest={"ids_sha256": sha(tmp_path / "ids.npy"), "masks_sha256": sha(tmp_path / "masks.npy"),
                  "shards": [{"start": 0, "stop": 1}]})


def test_metadata_mapping_preserves_center_order(tmp_path):
    p = SeriesProvenance(fake_corpus(tmp_path))
    m = p.windows("synthetic", [4, 1, -1, 2, 1])
    assert m["series"].tolist() == [1, 0, -1, 0, 0]
    assert m["position"].tolist() == [.5, .5, 0., 1., .5]
    assert m["valid"].tolist() == [True, True, False, True, True]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "mask_hash", "row"])
def test_metadata_fails_closed(tmp_path, kind):
    corpus = fake_corpus(tmp_path, missing=kind == "missing", duplicate=kind == "duplicate")
    if kind == "mask_hash":
        corpus.manifest["masks_sha256"] = "bad"
    if kind == "row":
        corpus.row["synthetic"] = 1
    with pytest.raises(ValueError):
        SeriesProvenance(corpus)
