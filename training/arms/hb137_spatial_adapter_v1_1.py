"""H137 v1.1.0: unchanged science, completed-forward watchdog heartbeats.

Changelog: new stage/source identity; old hash-bound v1 sources stay intact.
"""
from training.arms.hb137_spatial_adapter_v1 import *
from training.arms import hb137_spatial_adapter_v1 as previous

VERSION = "1.1.0"
WORK = previous.WORK + "/attempt-v1.1"
MANIFEST = WORK + "/parents-v1.json"
PROTOCOL = WORK + "/PROTOCOL-v1.md"


class WatchedArm(previous.SpatialArm):
    def __init__(self, adapter):
        super().__init__(adapter)
        self.name += "_V11"

    @property
    def sources(self):
        return tuple(sorted(set(super().sources) | {
            "training/arms/hb137_spatial_adapter_v1_1.py",
            "tools/hb137_screen_v1_1.py", "tools/hb137_safety_v1_1.py",
            MANIFEST, PROTOCOL, WORK + "/panel-v1.json", WORK + "/launch-b-v1.ps1"}))

    def contract(self, seed):
        return {**super().contract(seed), "execution_safety_version": VERSION}

    def forward(self, model, x, extra):
        from tools.hb137_safety_v1_1 import heartbeat
        result = super().forward(model, x, extra)
        # item synchronizes CUDA: queued work alone is not completed progress.
        hooks.need(bool(torch.isfinite(result).all()), "nonfinite watched forward")
        heartbeat()
        return result


HEAD = WatchedArm(False)
ADAPTER = WatchedArm(True)
