"""v1.0.0: full-bag BCE with bounded-image-memory encoder gradient replay.

Changelog: split after spatial_pool, retain full head autograd, replay encoder
RNG, and preserve the post-head RNG for the following step. No optimizer steps.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def rng_state(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device)
            if device.type == "cuda" else None)


def set_rng(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def augmentation_plan(img, gen, a):
    """Draw exactly the frozen augment() CPU-generator sequence, once."""
    B, _, _, S, _ = img.shape
    plan = dict(oy=int(torch.randint(0, S-a.crop+1, (1,), generator=gen)),
                ox=int(torch.randint(0, S-a.crop+1, (1,), generator=gen)))
    if a.aug != "v2":
        return plan
    ang = (torch.rand(B, generator=gen)*20-10)*math.pi/180
    sc = torch.rand(B, generator=gen)*.2+.9
    cos, sin = torch.cos(ang)/sc, torch.sin(ang)/sc
    plan["theta"] = torch.stack((torch.stack((cos, -sin, torch.zeros(B)), 1),
                                torch.stack((sin, cos, torch.zeros(B)), 1)), 1)
    if a.aug_res > 0:
        plan["hit"] = torch.rand(B, generator=gen) < a.aug_res
        plan["fac"] = torch.rand(B, generator=gen)*.75+1.25
    plan["gamma"] = torch.rand(B, generator=gen)*.45+.8
    plan["contrast"] = torch.rand(B, generator=gen)*.2+.9
    return plan


def augment_chunk(raw, studies, plan, a, device):
    """Only this image microbatch is transferred/augmented on the device."""
    y, x = plan["oy"], plan["ox"]
    raw = raw[..., y:y+a.crop, x:x+a.crop].to(device, non_blocking=True)
    if a.aug != "v2":
        return raw
    value = raw.float()
    theta = plan["theta"][studies].to(device)
    grid = F.affine_grid(theta, value.shape, align_corners=False)
    value = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if a.aug_res > 0:
        for b in studies.unique().tolist():
            if bool(plan["hit"][b]):
                selected = (studies == b).to(device)
                size = max(8, int(round(a.crop/float(plan["fac"][b]))))
                lo = F.interpolate(value[selected], size=(size, size), mode="area")
                value[selected] = F.interpolate(lo, size=(a.crop, a.crop), mode="bilinear", align_corners=False)
    gamma = plan["gamma"][studies].to(device).view(-1, 1, 1, 1)
    contrast = plan["contrast"][studies].to(device).view(-1, 1, 1, 1)
    value = (value/255.).clamp_(0, 1).pow(gamma)
    return (((value-.5)*contrast+.5).clamp_(0, 1)*255.).round_().to(torch.uint8)


def encode(model, images):
    x = (images.float()/255.-model.mean)/model.std
    x = x.to(memory_format=torch.channels_last)
    out = model.enc(x)
    return model.spatial_pool(out) if model.spatial else out


def backward(model, chunks, meta, mask, loss_fn, *, replay=True, amp=False):
    """chunks(): repeatable (images, real_count) iterator, padded like Net.

    Reference mode keeps every microbatch graph for ordinary autograd.
    Replay mode retains embeddings and RNG only. Caller zeros/clips/steps once.
    The iterator must use saved augmentation parameters and no global RNG.
    """
    device = meta.device
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError("fullbag replay rejects BatchNorm/stateful batch statistics")
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    features, states = [], []
    with torch.set_grad_enabled(not replay), torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=amp):
        for images, count in chunks():
            states.append(rng_state(device))
            features.append(encode(model, images)[:count])
    for n, b in model.named_buffers():
        if not torch.equal(b, buffers[n]):
            raise ValueError("encoder mutated buffer: "+n)
    embedding = torch.cat(features)
    del features
    if replay:
        embedding = embedding.detach().requires_grad_()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        output = model.head(embedding, meta, mask)
        logits = output[0] if isinstance(output, tuple) else output
        # Preserve the frozen trainer's autocast loss scope: BCE is promoted to
        # fp32 even when the head produces bf16 logits.
        loss = loss_fn(logits)
    loss.backward()
    if replay:
        gradient = embedding.grad.detach()
        post_head = rng_state(device)
        offset = 0
        try:
            for index, (images, count) in enumerate(chunks()):
                set_rng(states[index], device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    value = encode(model, images)[:count]
                value.backward(gradient[offset:offset+count])
                offset += count
            if offset != len(gradient):
                raise ValueError("replay chunk count changed")
        finally:
            set_rng(post_head, device)
    return loss.detach()


def training_backward(model, img, meta, mask, gen, a, loss_fn, *, replay=True):
    if img.device.type != "cpu" or mask.device.type != "cpu":
        raise ValueError("fullbag images/mask must remain on CPU")
    if not bool(mask.any(1).all()):
        raise ValueError("empty fullbag study")
    device = next(model.parameters()).device
    plan = augmentation_plan(img, gen, a)
    locations = mask.nonzero()
    chunk = 32

    def chunks():
        for start in range(0, len(locations), chunk):
            loc = locations[start:start+chunk]
            value = augment_chunk(img[loc[:, 0], loc[:, 1]], loc[:, 0], plan, a, device)
            count = len(value)
            if count % 32:
                value = torch.cat((value, value.new_zeros((32-count, *value.shape[1:]))))
            yield value, count

    return backward(model, chunks, meta.to(device), mask.to(device), loss_fn,
                    replay=replay, amp=device.type == "cuda")
