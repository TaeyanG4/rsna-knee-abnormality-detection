"""H-B-156 v1.0.1: constructor-only ConvNeXt option for the EXP143 MIL model.

Changelog: v1.0.1 preserves full replay for the BatchNorm-free ConvNeXt model.
v1.0.0 adds the registered timm ConvNeXt-S backbone without changing
the EXP143/CoAtNet implementation or any preprocessing, pooling, or training code.
No downloads occur: both constructors use pretrained=False; ImageNet state is
loaded explicitly by the caller with exp143_own0949.load_imagenet.

Standalone serving requires only selecting this constructor with
build_model(res, backbone=checkpoint['arch']); load checkpoint['model'] strictly.
The state keys and exact Raptor head are inherited from EXP143 unchanged.
"""
from __future__ import annotations

VERSION = "1.0.1"
ARCH = "convnext_small.fb_in22k_ft_in1k_384"
COATNET_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"


def build_model(res=384, backbone_factory=None, *, backbone=COATNET_ARCH):
    """Build the unchanged EXP143 head on the selected offline backbone.

    CoAtNet calls the original constructor directly, including its img_size
    argument and optional test factory. ConvNeXt is resolution-independent and
    must not receive CoAtNet's img_size keyword. Feature width is read by the
    original head from timm's num_features (768 for the registered ConvNeXt-S).
    """
    from exp143_own0949 import build_model as original_model

    if backbone == COATNET_ARCH:
        return original_model(res, backbone_factory=backbone_factory)
    if backbone != ARCH:
        raise ValueError(f"unregistered H-B-156 backbone: {backbone}")
    if backbone_factory is None:
        import timm

        def backbone_factory():
            return timm.create_model(ARCH, pretrained=False, num_classes=0,
                                     in_chans=3, global_pool="avg")
    return original_model(res, backbone_factory=backbone_factory)


def recompute_bn(model, loader, device, autocast_ctx=None, update_bn=None):
    """Delegate BN models unchanged; replay every batch for BN-free ConvNeXt.

    The registered final full-loader replay remains in train/no-grad mode with
    the caller's autocast context, even though ConvNeXt has no running statistics
    to recalibrate. Restore all original module modes and buffer values. No
    optimizer is involved, so model parameters are untouched. The receipt uses
    the original fields, records zero BN modules, and includes no raw UIDs.
    """
    import contextlib
    import time
    import torch
    from tools.exp112_full_swa import batchnorm_modules, uid_hash
    from tools.exp112_full_swa import recompute_bn as original_recompute_bn

    if batchnorm_modules(model):
        return original_recompute_bn(model, loader, device, autocast_ctx, update_bn)
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    batches = []
    t0 = time.time()
    try:
        model.train()
        with torch.no_grad(), (autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()):
            for x, _y, uids in loader:
                model(x.to(device, non_blocking=True))
                batches.append(list(uids))
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for module, mode in modes:
            module.training = mode
    if not batches:
        raise RuntimeError("BatchNorm-free replay incomplete: zero batches")
    uids = [uid for batch in batches for uid in batch]
    return dict(function=f"{recompute_bn.__module__}.{recompute_bn.__qualname__}",
                batches=len(batches), studies=len(uids), unique_studies=len(set(uids)),
                studies_sha256=uid_hash(sorted(uids)),
                batch_sizes=sorted({len(batch) for batch in batches}), bn_modules=0,
                num_batches_tracked=[], mode="train", gradients="none",
                seconds=round(time.time() - t0, 1),
                note="Full registered replay completed; no BatchNorm running statistics exist.")
