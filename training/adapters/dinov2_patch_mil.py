"""Instance-local DINO spatial-token adapter; no loading, RNG, or device side effects."""
from types import MethodType

POLICY = 'dinov2-normalized-patch256-existing-mil-v1'


def encode_patches(self, x):
    if x.ndim != 5 or tuple(x.shape[2:]) != (3, 224, 224):
        raise ValueError('Expected [B,K,3,224,224]')
    batch, windows = x.shape[:2]
    tokens = self.backbone.forward_features(x.flatten(0, 1))
    if tuple(tokens.shape) != (batch * windows, 257, 384):
        raise ValueError('Expected final normalized CLS+256 patch tokens')
    return tokens[:, 1:, :].reshape(batch, windows * 256, 384)


def install(model):
    import torch.nn as nn
    bb = model.backbone
    if getattr(model, '_hc06_aggregation_policy', None) is not None:
        raise ValueError('Patch adapter already installed')
    if (bb.num_prefix_tokens != 1 or bb.num_features != 384
            or bb.global_pool != 'token' or not bb.dynamic_img_size
            or tuple(bb.patch_embed.patch_size) != (14, 14)
            or not isinstance(bb.fc_norm, nn.Identity)
            or not isinstance(bb.head, nn.Identity) or bb.head_drop.p != 0):
        raise ValueError('Frozen DINO-S token configuration required')
    model.encode = MethodType(encode_patches, model)
    model._hc06_aggregation_policy = POLICY
    return model
