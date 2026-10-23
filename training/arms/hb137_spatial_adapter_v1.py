"""H-B-137 v1.0.0: frozen honest MIL-S parent, weak-four spatial residual.

New module; no deployed source changes. Head-only and adapter share every
training/data hook. The residual is per window, after the backbone's token
norm, before finding-MIL LayerNorm. Full-width head arithmetic preserves
initial bit parity; only four query/classifier rows are trainable.
"""
from copy import deepcopy
from math import isqrt
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from training.arms import mil_screen_hooks_v1 as hooks
from training.src.raptor_exp185 import finding_attention

VERSION = "1.0.0"
WORK = ".local/scratch/astra-b2/H-B-137"
MANIFEST = WORK + "/parents-v1.json"
PROTOCOL = WORK + "/PROTOCOL-v1.md"
WEAK = (3, 5, 6, 8)
PROTECTED = (0, 1, 2, 4, 7, 9, 10, 11)


class WeakLinear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.register_buffer("base_weight", original.weight.detach().clone())
        self.register_buffer("base_bias", original.bias.detach().clone())
        self.register_buffer("indices", torch.tensor(WEAK))
        self.weight = nn.Parameter(original.weight.detach()[list(WEAK)].clone())
        self.bias = nn.Parameter(original.bias.detach()[list(WEAK)].clone())

    def forward(self, x):
        return F.linear(x, self.base_weight.index_copy(0, self.indices, self.weight),
                        self.base_bias.index_copy(0, self.indices, self.bias))


class WeakHead(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.norm = deepcopy(parent.norm).requires_grad_(False)
        self.att = nn.Sequential(deepcopy(parent.att[0]).requires_grad_(False),
                                 nn.Tanh(), nn.Identity(), WeakLinear(parent.att[3]))
        self.register_buffer("base_clsW", parent.clsW.detach().clone())
        self.register_buffer("base_clsb", parent.clsb.detach().clone())
        self.register_buffer("indices", torch.tensor(WEAK))
        self.weak_clsW = nn.Parameter(parent.clsW.detach()[list(WEAK)].clone())
        self.weak_clsb = nn.Parameter(parent.clsb.detach()[list(WEAK)].clone())

    @property
    def clsW(self):
        return self.base_clsW.index_copy(0, self.indices, self.weak_clsW)

    @property
    def clsb(self):
        return self.base_clsb.index_copy(0, self.indices, self.weak_clsb)

    def forward(self, tokens, mask):
        return finding_attention(self, tokens, mask)[0]


class SpatialAdapter(nn.Module):
    def __init__(self, features):
        super().__init__()
        # The optional branch must not advance sampler/augmentation RNG state.
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Conv2d(features, 16, 1, bias=False)
            self.spatial = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
            self.out = nn.Conv2d(16, features, 1, bias=False)
            nn.init.zeros_(self.out.weight)

    def forward(self, tokens, windows):
        b, count, features = tokens.shape
        hooks.need(type(windows) is int and windows > 0, "invalid window count")
        side = isqrt(count // windows)
        hooks.need(side > 0 and count == windows * side * side, "non-square window token grid")
        maps = tokens.reshape(b * windows, side, side, features).permute(0, 3, 1, 2)
        residual = self.out(F.gelu(self.spatial(F.gelu(self.down(maps)))))
        return tokens + residual.permute(0, 2, 3, 1).reshape(b, count, features)


class FrozenWeakMIL(nn.Module):
    def __init__(self, parent, adapter):
        super().__init__()
        self.parent = parent.requires_grad_(False).eval()
        self.weak_head = WeakHead(parent)
        self.adapter = SpatialAdapter(parent.backbone.num_features) if adapter else None
        self.register_buffer("indices", torch.tensor(WEAK))

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()  # also freezes dropout/stochastic depth on every epoch
        return self

    def forward(self, x):
        with torch.no_grad():
            tokens, mask = self.parent.encode(x)
            original = self.parent.head((tokens, mask))
        adapted = self.adapter(tokens, x.shape[1]) if self.adapter is not None else tokens
        weak = self.weak_head(adapted, mask).index_select(1, self.indices)
        return original.index_copy(1, self.indices, weak)


def parents():
    doc = hooks.read(hooks.ROOT / MANIFEST)
    hooks.need(doc["version"] == VERSION and set(doc["parents"]) == {"42", "43"}, "parent manifest differs")
    return doc["parents"]


def check_parent_checkpoint(checkpoint, spec, snap):
    c, p = spec.document["config"], checkpoint["provenance"]
    hooks.need(checkpoint["epoch"] == 19 and checkpoint["fold"] == 0
               and checkpoint["lab"] == hooks.TARGETS and checkpoint["arch"] == c["model"]["arch"]
               and checkpoint["model_kind"] == "finding_spatial_tokens_v1"
               and p["spec_sha256"] == snap["spec_sha256"] and p["checkpoint_selection"] == "fixed-final"
               and p["identity"]["mode"] == "full" and p["identity"]["device"] == "cuda"
               and p["train_ids_ordered_sha256"] == c["roots"]["train_sha256"]
               and p["eval_ids_ordered_sha256"] == c["roots"]["val_sha256"], "parent ancestry differs")


class SpatialArm(hooks.Arm):
    section = "model"
    key = "frozen_weak_spatial_v1"
    screen = "HB137_FROZEN_WEAK_V1"

    def __init__(self, adapter):
        self.adapter = adapter
        self.name = "HB137_ADAPTER" if adapter else "HB137_HEAD"

    @property
    def sources(self):
        return tuple(sorted({"training/arms/hb137_spatial_adapter_v1.py", MANIFEST, PROTOCOL,
                             WORK + "/panel-v1.json", WORK + "/launch-b-v1.ps1",
                             "tools/hb122_screen_v1.py", "tools/hb137_screen_v1.py",
                             *(p for snap in parents().values() for p in snap["bindings"])}))

    def contract(self, seed):
        snap = parents()[str(seed)]
        return dict(version=VERSION, variant="adapter" if self.adapter else "head",
                    parent=snap, weak=list(WEAK), protected=list(PROTECTED), parent_mode="frozen_eval",
                    adapter_rank=16 if self.adapter else 0, kernel=3 if self.adapter else 0,
                    output_init="zero", location="after_backbone_token_norm_before_finding_norm",
                    refit="weak_query_and_classifier_rows_only", dropout="disabled_both_branches",
                    objective="unchanged_BCE_all_mean12", draw_policy="unchanged_shared_hooks",
                    manifest_sha256=hooks.sha(hooks.ROOT / MANIFEST))

    def attach(self, model, spec, stage):
        self.validate_recipe(spec)
        from tools.hb122_screen_v1 import snapshot
        seed = spec.document["seed"]
        snap = self.contract(seed)["parent"]
        hooks.need(snapshot("control", seed, False) == snap, "parent lifecycle drift")
        path = Path(stage) / snap["checkpoint"]
        hooks.need(hooks.sha(path) == snap["bindings"][snap["checkpoint"]], "parent checkpoint digest differs")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        check_parent_checkpoint(checkpoint, spec, snap)
        hooks.strict_load(model, checkpoint["model"])
        return FrozenWeakMIL(model, self.adapter)

    def provenance(self, spec):
        return dict(hb137=self.contract(spec.document["seed"]),
                    replication="two matched head seeds; not independent backbone replication")


HEAD = SpatialArm(False)
ADAPTER = SpatialArm(True)
