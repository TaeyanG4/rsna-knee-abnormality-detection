"""H-B-124 label-free draft registration v1.0.0 (2026-10-13).

Creates the same specs/record as shared configure, without opening labels.
Does NOT issue stage, smoke, admission or scientific validation evidence.
B must run shared configure/Verify when label access is authorized.
"""
import argparse
from pathlib import Path

from training.arms.hb124_series_arm_v1 import ARM
from training.arms import mil_screen_hooks_v1 as hooks
from tools import mil_screen_arm_build_v1 as build

WORK = hooks.ROOT / ".local/scratch/sol-3/hb124"
REFERENCE = "training.arms.hb124_series_arm_v1:ARM"


def documents(seed, config_dir="training/configs"):
    control = hooks.control_spec(seed)
    doc = control.document
    declaration = doc["target_environment"]["evidence"]["declaration.json"]
    target = next(p for p, h in doc["inputs"]["evidence"].items() if h == declaration)
    recipe_control = next(p for p in doc["inputs"]["evidence"] if p != target)
    recipe = hooks.life.ArmSpec((hooks.ROOT / recipe_control).read_bytes()).document
    recipe["config"][ARM.section][ARM.key] = ARM.contract(seed)
    recipe["arm"] = f"{ARM.name}_RECIPE_S{seed}_F0"
    recipe = hooks.life.ArmSpec.from_document(recipe)
    recipe_path = f"{config_dir}/hb124-series-recipe-s{seed}-f0-v1.json"
    doc["config"][ARM.section][ARM.key] = ARM.contract(seed)
    doc["arm"] = f"{ARM.name}_S{seed}_F0"
    doc["inputs"]["sources"] = {p: hooks.sha(hooks.ROOT / p) for p in sorted(hooks.expected_sources(ARM, control))}
    doc["inputs"]["evidence"] = {recipe_path: recipe.sha256, target: declaration}
    spec = hooks.life.ArmSpec.from_document(doc)
    ARM.validate_recipe(spec)
    return recipe_path, recipe, f"{config_dir}/hb124-series-s{seed}-f0-v1.json", spec


def configure():
    specs = {}
    for seed in (42, 43):
        recipe_path, recipe, path, spec = documents(seed)
        build.same_or_new(hooks.ROOT / recipe_path, recipe.document)
        build.same_or_new(hooks.ROOT / path, spec.document)
        specs[str(seed)] = path
    build.same_or_new(WORK / "arm-build.json", dict(version=build.VERSION, arm=REFERENCE, arm_name=ARM.name,
        work=build.rel(WORK), specs=specs, workers=2,
        control_specs={str(s): hooks.control_path(s) for s in (42, 43)}))
    return dict(status="ARM_SPECS_DRAFTED_LABEL_FREE", specs=specs, full_validation=False,
                next="shared configure then Verify; labels were not opened")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("configure",))
    ap.parse_args()
    print(hooks.life.canonical_bytes(configure()).decode())
