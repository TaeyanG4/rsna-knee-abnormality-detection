"""Engineering-only demo arm v1.0.0 (2026-10-13) for mil_screen_hooks_v1.

Carries the exact sampled window centers through dataset -> collate -> move ->
forward (training, all readouts, flip TTA, reload) but IGNORES them, so its
outputs must be byte-identical to the shared control. It exists to prove the
H-B-124 metadata path end to end on CPU. Never a scientific arm; never fitted
on GPU. Its specs live under .local/scratch/claude-hb122/demo/configs.
"""
from training.arms import mil_screen_hooks_v1 as hooks


class CentersNoOp(hooks.Arm):
    name = "HBDEMO_CENTERS"
    section = "windows"
    key = "hooks_demo_v1"
    sources = ("training/arms/mil_screen_demo_arm_v1.py",)
    screen = "HBDEMO_CENTERS_V1"
    collate_fn = staticmethod(hooks.collate_centers)

    def contract(self, seed):
        return dict(version="1.0.0", metadata="exact sampled centers int64[K]",
                    forward="ignores metadata", engineering_only=True)

    def dataset(self, dataset, spec, stage, *, train):
        return hooks.CenterWindows(dataset)

    def probe(self, device):
        return hooks.torch.zeros(1, 1, dtype=hooks.torch.int64, device=device)

    def move(self, extra, device):
        hooks.need(len(extra) == 1, "centers metadata required")
        return extra[0].to(device, non_blocking=True)

    def forward(self, model, x, extra):
        hooks.need(extra is not None and extra.dtype == hooks.torch.int64
                   and tuple(extra.shape) == tuple(x.shape[:2]), "centers metadata shape differs")
        return model(x)

    def provenance(self, spec):
        return dict(demo_arm="metadata carried and ignored")


ARM = CentersNoOp()
