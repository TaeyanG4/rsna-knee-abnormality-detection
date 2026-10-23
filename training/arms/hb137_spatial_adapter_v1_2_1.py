"""H137 v1.2.1: unchanged science, phase-aware hooks and chunked parent reload.

Changelog: new stage/source identity; old hash-bound v1 sources stay intact.
"""
from training.arms.hb137_spatial_adapter_v1 import *
from training.arms import hb137_spatial_adapter_v1 as previous
from tools.hb137_safety_v1_2_1 import load_checkpoint, checkpoint_sha

VERSION = "1.2.1"
WORK = previous.WORK + "/attempt-v1.2.1"
MANIFEST = WORK + "/parents-v1.json"
PROTOCOL = WORK + "/PROTOCOL-v1.md"


class WatchedArm(previous.SpatialArm):
    def __init__(self, adapter):
        super().__init__(adapter)
        self.name += "_V121"

    @property
    def sources(self):
        return tuple(sorted(set(super().sources) | {
            "training/arms/hb137_spatial_adapter_v1_2_1.py",
            "tools/hb137_screen_v1_2_1.py", "tools/hb137_safety_v1_2_1.py",
            "tools/mil_screen_arm_build_v1_2_1.py", "training/arms/mil_screen_hooks_v1_2_1.py",
            "training/runners/mil_screen_hooks_v1_2_1.py",
            MANIFEST, PROTOCOL, WORK + "/panel-v1.json", WORK + "/launch-b-v1.ps1"}))

    def contract(self, seed):
        return {**super().contract(seed), "execution_safety_version": VERSION}


    def attach(self, model, spec, stage):
        self.validate_recipe(spec)
        from tools.hb122_screen_v1 import snapshot
        seed = spec.document["seed"]
        snap = self.contract(seed)["parent"]
        hooks.need(snapshot("control", seed, False) == snap, "parent lifecycle drift")
        path = Path(stage) / snap["checkpoint"]
        hooks.need(checkpoint_sha(path) == snap["bindings"][snap["checkpoint"]], "parent checkpoint digest differs")
        checkpoint = load_checkpoint(path, map_location="cpu", weights_only=False)
        check_parent_checkpoint(checkpoint, spec, snap)
        hooks.strict_load(model, checkpoint["model"])
        return FrozenWeakMIL(model, self.adapter)

HEAD = WatchedArm(False)
ADAPTER = WatchedArm(True)
