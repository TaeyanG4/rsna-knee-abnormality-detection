"""INF2 Small10 scientific helper v1.0.0, appended to the existing Q module.

Exact HB105 model/math; no launcher, threads, GPU selection or model forward
on import. TARGETS, need and sha are supplied by the existing Q module.
"""

INF2_HELPER_VERSION = '1.0.0'


def inf2_small_meta(ck, entry):
    expected = dict(arch="convnext_small.fb_in22k_ft_in1k", res=224, lab=list(TARGETS),
                    contract="raptor64_336_c140_s02", norm="imagenet", window_policy="within-slot-valid",
                    fold=entry["fold"], epoch=19)
    need(all(ck.get(k) == v for k, v in expected.items()), "MIL metadata mismatch")
    need(entry['arch'] == expected['arch'], 'Small entry architecture differs')
    need(ck.get("model_kind", ck.get("provenance", {}).get("model_kind")) == "finding_spatial_tokens_v1",
         "not finding-specific spatial MIL")
    need(ck.get("input_policy") == "center3" and ck.get("k_eval") == 24 and
         ck.get("flip_tta") == "coronal_axial_horizontal_mean_probability", "MIL inference policy differs")
    if entry.get("spec_sha256"):
        need(ck.get("provenance", {}).get("spec_sha256") == entry["spec_sha256"], "MIL spec binding differs")
    else:
        need(ck.get("seed", ck.get("provenance", {}).get("seed")) == entry["seed"], "MIL seed differs")


def inf2_load_small(path, entry, device):
    import torch
    from torch import nn
    import timm

    # Verbatim original scientific functions, nested only to avoid importing
    # training modules or initializing torch before the existing Q loader.
    def valid_windows(x):
        """Detect the loader's zero-image padding after ImageNet normalization."""
        import torch

        if x.ndim != 5 or x.shape[2] != 3:
            raise ValueError('expected B,K,3,H,W windows')
        mean = x.new_tensor((.485, .456, .406)).view(1, 1, 3, 1, 1)
        std = x.new_tensor((.229, .224, .225)).view(1, 1, 3, 1, 1)
        background = -mean / std
        present = ~torch.isclose(x, background, rtol=0, atol=1e-6).all(dim=(2, 3, 4))
        # Empty studies remain finite; the first padding window is the fixed fallback.
        present[:, 0] |= ~present.any(dim=1)
        return present

    def spatial_tokens(model, x):
        """Return B,K*H*W,F tokens and a matching validity mask."""
        if x.ndim != 5:
            raise ValueError('expected B,K,C,H,W')
        b, k = x.shape[:2]
        fmap = model.backbone.forward_features(x.flatten(0, 1))
        features = model.backbone.num_features
        if fmap.ndim != 4 or fmap.shape[1] != features:
            raise ValueError('ConvNeXt NCHW spatial features required')
        # ConvNeXt's global head applies this norm after spatial averaging. For
        # retained tokens apply the *same existing* norm per spatial position.
        fmap = model.backbone.head.norm(fmap)
        _, _, h, w = fmap.shape
        tokens = fmap.permute(0, 2, 3, 1).reshape(b, k * h * w, features)
        mask = valid_windows(x).repeat_interleave(h * w, dim=1)
        return tokens, mask

    def finding_attention(model, tokens, mask):
        """Twelve independently normalized token distributions and logits."""
        import torch

        if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('token/mask shape differs')
        if not mask.any(dim=1).all():
            raise ValueError('at least one token per study required')
        h = model.norm(tokens)
        scores = model.att(h).float().masked_fill(~mask[..., None], torch.finfo(torch.float32).min)
        weights = torch.softmax(scores, dim=1).to(h.dtype)
        pooled = torch.einsum('btn,btf->bnf', weights, h)
        logits = (pooled * model.clsW).sum(-1) + model.clsb
        return logits, weights

    class FindingMIL(nn.Module):
        """Same construction order, RNG draws and state keys as the legacy model."""

        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            features = backbone.num_features
            self.norm = nn.LayerNorm(features)
            self.att = nn.Sequential(nn.Linear(features, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
            self.clsW = nn.Parameter(torch.zeros(12, features))
            self.clsb = nn.Parameter(torch.zeros(12))
            nn.init.trunc_normal_(self.clsW, std=.02)
            self.n = 12
            self.pool_policy = "finding_spatial_tokens_v1"

        def encode(self, x):
            return spatial_tokens(self, x)

        def head(self, encoded):
            return finding_attention(self, *encoded)[0]

        def forward(self, x):
            return self.head(self.encode(x))

    def strict_load(model, state):
        expected = model.state_dict()
        need(set(state) == set(expected), "model state keys differ")
        for key, tensor in state.items():
            need(tensor.shape == expected[key].shape and tensor.dtype == expected[key].dtype
                 and bool(torch.isfinite(tensor).all()), "model tensor shape, dtype or finiteness differs")
        model.load_state_dict(state, strict=True)

    need(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Small checkpoint differs')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    inf2_small_meta(ck, entry)
    model = FindingMIL(timm.create_model(entry['arch'], pretrained=False, num_classes=0,
                                        in_chans=3, global_pool='avg'))
    strict_load(model, ck['model'])
    need(model.pool_policy == 'finding_spatial_tokens_v1', 'Small pooling differs')
    return model.eval().to(device)


def inf2_small_windows(vol, centers):
    import numpy as np
    import torch

    def scale_windows(vol, centers):
        need(vol.shape == (64, 336, 336) and vol.dtype == np.uint8 and len(centers) in (12, 24, 36),
             "registered mixed windows required")
        out = np.empty((len(centers), 3, 224, 224), np.float32)
        for j, center in enumerate(centers):
            if center == -1:
                out[j] = 0
                continue
            center = max(1, min(int(center), 62))
            image = vol[center] if j % 2 == 0 else vol[center, 48:288, 48:288]
            plane = torch.from_numpy(image[None].astype(np.float32) / 255.)
            out[j] = torch.nn.functional.interpolate(plane[None], size=(224, 224), mode="bilinear",
                                                    align_corners=False)[0].numpy()
        return out

    need(len(centers) == 36, 'Small serving requires exactly 36 centers')
    x = scale_windows(vol, centers)
    return (x - np.array([.485, .456, .406], np.float32).reshape(3, 1, 1)) / np.array(
        [.229, .224, .225], np.float32).reshape(3, 1, 1)
