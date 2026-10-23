"""EXP168 v1.2.1: frozen-source release shim with complete Public3 guard/arms.

Use this module in place of tools.exp168_own0949_oai_jev_release for new
average/stage/parity-plan/public3/compare-parity/push-plan artifacts. Public3
also requires --exp143-checkpoint (the original authenticated EXP143 average);
compare-parity requires --blend3. Existing artifacts and training bindings stay
unchanged. Import, packaging and CPU tests never initialize CUDA or use network.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

VERSION = "1.2.1"
TOOL = Path(__file__).resolve()
ROOT = TOOL.parents[1]
ADAPTER_NAME = "exp168_own0949_oai_jev_release.py"
ADAPTER_SHA256 = "3a485935339378fa8e2f62cfc55068ecdc3f3777b976bf972a4493a14be8123c"
ADAPTER_PATH = TOOL.with_name(ADAPTER_NAME)


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if _file_sha(ADAPTER_PATH) != ADAPTER_SHA256:
    raise ValueError("Pinned EXP168 release implementation has changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_frozen_exp168", ADAPTER_PATH)
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
_shared = _base._shared
ASSET_DATASET = _base.ASSET_DATASET.replace("v1-2-assets", "v1-2-1-assets")
KERNELS = {name: slug + "-1" for name, slug in _base.KERNELS.items()}
for _module in (_base, _shared):
    for _name in ("VERSION", "TOOL", "ROOT", "ASSET_DATASET", "KERNELS"):
        setattr(_module, _name, globals()[_name])

need, sha, replace_once = _shared.need, _shared.sha, _shared.replace_once
_original_load = _base._original_load_module
_original_notebook = _base.notebook
_original_parser = _base.parser
_original_push_plan = _base.push_plan
_original_kernel_metadata = _base.kernel_metadata


def load_module(name, path):
    """Supply only the missing reader; retain EXP168's registration/run guards."""
    path = Path(path)
    if path.resolve() == (ROOT / "tools/exp143_v2.py").resolve():
        guard = _original_load(name, ROOT / "tools/exp168_own0949_oai_jev.py")
        # A facade avoids adding attributes to either frozen guard module. Its
        # EXP143 dependency is already authenticated by load_registration.
        return SimpleNamespace(load_registration=guard.load_registration,
                               registered_run=guard.registered_run,
                               series_records=guard.base.series_records)
    return _original_load(name, path)


def notebook(variant, dataset, archive_sha, manifest_sha, module_sha, credits):
    nb, title, description = _original_notebook(
        variant, dataset, archive_sha, manifest_sha, module_sha, credits)
    code = nb["cells"][-1]["source"]
    code = replace_once(code, repr(ADAPTER_NAME) + ", 'exp152_own0949_release.py'",
                        repr(TOOL.name) + ", " + repr(ADAPTER_NAME) + ", 'exp152_own0949_release.py'")
    code = replace_once(code, "_hash(_assets/" + repr(ADAPTER_NAME) + ")",
                        "_hash(_assets/" + repr(TOOL.name) + ")")
    code = replace_once(code, "import exp168_own0949_oai_jev_release as _e152",
                        "import exp168_own0949_oai_jev_release_v121 as _e152")
    compile(code, "exp168-v121-notebook", "exec")
    title = title.replace("v1.2", "v1.2.1")
    nb["cells"][0]["source"] = "# " + title + "\n\n" + description
    nb["cells"][-1]["source"] = code
    return nb, title, description


def verify_assets(path, manifest):
    _base._original_verify_assets(path, manifest)
    need(manifest.get("schema") == "exp168-release-v1", "EXP168 release schema mismatch")
    need(manifest["checkpoints"][_base.OWN_NAME].get("oai") is True,
         "EXP168 OAI provenance missing")
    need(manifest["checkpoints"][_base.MEMBER_NAME].get("oai") is False,
         "EXP143 member must be OAI-free")
    expected = {_base.OWN_NAME, _base.MEMBER_NAME, TOOL.name, ADAPTER_NAME,
                _base.BASE_NAME, "public-serving.py"}
    need(set(manifest["asset_sha256"]) == expected, "Unexpected release payload")
    need(manifest["asset_sha256"][ADAPTER_NAME] == ADAPTER_SHA256 and
         manifest["asset_sha256"][_base.BASE_NAME] == _base.BASE_SHA256,
         "Frozen release source mismatch")


def parser():
    result = _original_parser()
    result.description = __doc__
    for action in result._subparsers._group_actions:
        action.choices["public3"].add_argument("--exp143-checkpoint", required=True)
        action.choices["compare-parity"].add_argument("--blend3", required=True)
    return result


def kernel_metadata(variant, dataset=ASSET_DATASET):
    return _original_kernel_metadata(variant, dataset)


def push_plan(args):
    result = _original_push_plan(args)
    path = Path(args.out)
    text = path.read_text(encoding="utf-8")
    text = text.replace("EXP168-v1-2-", "EXP168-v1-2-1-").replace("OAI-v1.2;", "OAI-v1.2.1;")
    text = text.replace(
        "For blend3, review three-arm raw.npz parity (own/public/exp143) separately; two-arm compare-parity is unsupported.",
        "Run the v1.2.1 compare-parity with --own, --blend and --blend3; all three variants must pass.")
    path.write_text(text, encoding="utf-8", newline="\n")
    result["sha256"] = sha(path)
    return result


# Marker-counted changes to SHA-pinned implementations: reuse all existing
# validation, data protection, parity thresholds, decoding and prediction math.
_stage_source = inspect.getsource(_base.stage)
_stage_source = replace_once(_stage_source,
    "    shutil.copyfile(TOOL, assets / TOOL.name)\n",
    "    shutil.copyfile(ADAPTER_PATH, assets / ADAPTER_NAME)\n"
    "    shutil.copyfile(TOOL, assets / TOOL.name)\n")
_stage_source = replace_once(_stage_source,
    "(OWN_NAME, MEMBER_NAME, TOOL.name, BASE_NAME, \"public-serving.py\")",
    "(OWN_NAME, MEMBER_NAME, TOOL.name, ADAPTER_NAME, BASE_NAME, \"public-serving.py\")")
_base.ADAPTER_PATH, _base.ADAPTER_NAME = ADAPTER_PATH, ADAPTER_NAME
exec(compile(_stage_source, str(TOOL) + ":stage", "exec"), _base.__dict__)

_public3_source = inspect.getsource(_shared.public3)
_public3_source = replace_once(_public3_source,
    '    comp = Path(args.competition)\n',
    '    member_path = Path(args.exp143_checkpoint)\n'
    '    need(sha(member_path) == manifest["source_member_sha256"], "EXP143 reference checkpoint mismatch")\n'
    '    comp = Path(args.competition)\n')
_public3_source = replace_once(_public3_source,
    '    start = time.time()\n',
    '    ns["_E152_FILES"][MEMBER_NAME] = stage_dir / "assets" / MEMBER_NAME\n'
    '    start = time.time()\n')
_public3_source = replace_once(_public3_source,
    '    train = load_module("exp152_local_training", ROOT / "training/src/exp143_own0949.py")\n',
    '    member = checked_model(ns, stage_dir / "assets" / MEMBER_NAME, ARCH, 384, args.device)[0]\n'
    '    train = load_module("exp152_local_training", ROOT / "training/src/exp143_own0949.py")\n')
_public3_source = replace_once(_public3_source,
    '    reference.eval().to(args.device)\n',
    '    reference.eval().to(args.device)\n'
    '    member_reference = train.build_model(320)\n'
    '    member_reference.load_state_dict(safe_load(member_path)["model"], strict=True)\n'
    '    member_reference.eval().to(args.device)\n'
    '    c, ref_c = [], []\n')
_public3_source = replace_once(_public3_source,
    '            ref_a.append(train.infer_volume(reference, local_vol, local_mask, args.device, "fp16"))\n',
    '            ref_a.append(train.infer_volume(reference, local_vol, local_mask, args.device, "fp16"))\n'
    '            c.append(ns["infer_pair_probs"](member, plain, mirror, args.device))\n'
    '            ref_c.append(train.infer_volume(member_reference, local_vol, local_mask, args.device, "fp16"))\n')
_public3_source = replace_once(_public3_source,
    '    comparisons = {}\n',
    '    c, ref_c = map(np.asarray, (c, ref_c))\n'
    '    raw_exact = raw_exact and array_sha(c) == array_sha(ref_c)\n'
    '    comparisons = {}\n')
_public3_source = replace_once(_public3_source,
    '        arms = [a] if variant == "own" else [a, b]\n'
    '        refs = [ref_a] if variant == "own" else [ref_a, ref_b]\n',
    '        arms = {"own": [a], "blend": [a, b], "blend3": [a, b, c]}[variant]\n'
    '        refs = {"own": [ref_a], "blend": [ref_a, ref_b], "blend3": [ref_a, ref_b, ref_c]}[variant]\n')
_public3_source = replace_once(_public3_source,
    '                  comparisons=comparisons, max_abs_own=float(np.max(np.abs(a-ref_a))),\n',
    '                  comparisons=comparisons, max_abs_exp143=float(np.max(np.abs(c-ref_c))),\n'
    '                  max_abs_own=float(np.max(np.abs(a-ref_a))),\n')
exec(compile(_public3_source, str(TOOL) + ":public3", "exec"), _shared.__dict__)

_compare_source = inspect.getsource(_shared.compare_parity)
_compare_source = replace_once(_compare_source,
    '        need(set(a) == set(b) == ({"ids", "targets", "own"} | ({"public"} if variant == "blend" else set())), "Raw array key mismatch")',
    '        names = {"own": ("own",), "blend": ("own", "public"), "blend3": ("own", "public", "exp143")}[variant]\n'
    '        need(set(a) == set(b) == {"ids", "targets", *names}, "Raw array key mismatch")')
_compare_source = replace_once(_compare_source,
    '        for name in ("own", "public") if variant == "blend" else ("own",):',
    '        for name in names:')
exec(compile(_compare_source, str(TOOL) + ":compare-parity", "exec"), _shared.__dict__)

_plan_source = inspect.getsource(_shared.parity_plan)
_plan_source = replace_once(_plan_source, "both final CSVs", "all three final CSVs")
_plan_source = replace_once(_plan_source, "Push the two private kernels", "Push the three private kernels")
exec(compile(_plan_source, str(TOOL) + ":parity-plan", "exec"), _shared.__dict__)

_shared.MEMBER_NAME = _base.MEMBER_NAME
for _name in ("load_module", "notebook", "verify_assets"):
    for _module in (_base, _shared):
        setattr(_module, _name, globals()[_name])
for _name in ("public3", "compare_parity", "parity_plan"):
    setattr(_base, _name, getattr(_shared, _name))
_base.parser, _base.push_plan = parser, push_plan
_base.kernel_metadata = kernel_metadata


def __getattr__(name):
    return getattr(_base, name)


if __name__ == "__main__":
    sys.exit(_base.main())
