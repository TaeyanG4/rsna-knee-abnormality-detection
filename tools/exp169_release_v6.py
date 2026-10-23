"""EXP169 release v6, 1.3.0 (2026-10-12).

Changelog 1.3.0: replace failed study members in rank space (OOF-selected
MIL/C4 donor, otherwise weighted remaining members), with receipts, no retries.

Changelog 1.2.1: declare the Python 3 kernelspec in every release notebook.
Changelog 1.2.0: Q7 = R50 + frozen A4 mixed-scale C4 + unchanged MIL;
import the v4 serving policy, remove DINO inference, and use slug titles.
Changelog 1.1.1: Amendment 2 menu: six candidates; B6-w50 replaces B6-w45;
remove A6/A6q w45; MIL dataset bytes unchanged.
1.1.0: bind collected member receipts and B's F0 selection; parameterise
receipt-bound partner weights; eight candidates share inference; loose MIL upload.
1.0.0: shared two-seed anchor pass; Q6 spatial finding-MIL; disjoint study
shards on two T4s; immutable local staging and fail-closed hash slots. No account
operations or application wall limits. Frozen scored stages are never modified.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import csv
import hashlib
import importlib.util
import io
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import textwrap
import traceback
import zipfile

VERSION = "1.3.0"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP = Path("artifacts/experiments/EXP169_RELEASE_V6")
B172 = Path("artifacts/releases/EXP169_B172_v2_0_2_stage")
V4 = Path("artifacts/releases/EXP169_V4_A4_B4_s43_stage_v413")
E155 = Path("artifacts/releases/EXP155_S42v1_1_stage_v12")
F0 = Path(".local/refactor/mil-s43-f0-collected-v3/fold0")
COLLECTED = Path(".local/refactor/mil-s43-members-f1-f4-v1")
COLLECTED_OUTPUTS = Path(".local/refactor/mil-s43-members-collected-v1")
UID = "StudyInstanceUID"
LABELS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
DEFAULT_WEIGHTS = {"A6": .15, "A6q": .15, "B6": .30, "A6-w30": .30,
                   "A6q-w30": .30, "B6-w50": .50}
CANDIDATES = tuple(DEFAULT_WEIGHTS)
PUBLIC_NAME = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
PUBLIC_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
S6_SHA = "6d08301e691e79c0076636ce109739247ec784b300ef7d3081c6e6033da0e6dd"
SPEC_PINS = ["2f91e82f604b80cdb61bc2ad3fc8aa268d68de5d7494c35136037a955f3d58c4",
             "356ef9b193d5472e0bff9abd228519c9760c4207fcc73e38b23cc384b1e9470c",
             "8b980c76e88fa1a0607626bc406325486a81b75a119fbe53ef8728b2c7136874",
             "bf1c4b8035f3c89e916a80e51effe2f898b341a4775fe94d31629e336785e665"]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def maths():
    # Local qualification uses the same frozen modules packaged for Kaggle.
    import tools
    frozen = ROOT / B172 / "assets/code"
    if frozen.exists():
        for name, p in (("exp169_partner_bundle_release", frozen / "exp169_partner_bundle_release.py"),
                        ("exp169_release_v2", frozen / "exp169_release_v2.py"),
                        ("exp152_own0949_release", ROOT / B172 / "own-anchor/assets/exp152_own0949_release.py")):
            old = getattr(tools, name, None)
            if old is None or Path(old.__file__).resolve() != p.resolve():
                setattr(tools, name, load_module("tools." + name, p))
    from tools import exp169_release_v2 as v2, exp169_release_v5q as v5
    return v2, v5


def probability(a, shape, name):
    import numpy as np
    a = np.asarray(a)
    need(a.shape == shape and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(),
         "invalid probabilities: " + name)
    return a


def q_values(scores):
    """Preserve frozen float32 member arithmetic; rank families, never folds."""
    import numpy as np
    v2, _ = maths()
    need(set(scores) == {"l2", "convnext", "mil"}, "Q7 family set differs")
    n = np.asarray(scores["l2"]).shape[1]
    r = probability(scores["l2"], (10, n, 12), "R50")
    c = probability(scores["convnext"], (5, n, 12), "C4")
    m = probability(scores["mil"], (10, n, 12), "MIL")
    family = {"l2": v2.rank(((r[:5] + r[5:]) / 2).mean(0)),
              "convnext": v2.rank(c.mean(0)), "mil": v2.rank(m.mean(0))}
    return v2.rank(sum(v2.rank(x) for x in family.values()) / 3), family


def a4_module():
    """Import the immutable v4 release module, also bundled verbatim at runtime."""
    maths()
    path = ROOT / "tools/exp169_v4_frozen.py"
    if not path.exists():
        path = ROOT / V4 / "code/assets/exp169_release_v4.py"
    return load_module("tools.exp169_v4_frozen", path)


def a4_policy():
    """Reuse v4's exact loader and crop overlay without its two-GPU scheduler.

    The frozen API couples these to a full Q4 scorer (including DINO). Extract
    its loader construction and crop insertion by AST, just as v4's own CPU
    window-parity check does. No policy is copied or reimplemented here.
    Structural changes to that imported source fail closed.
    """
    v4 = a4_module()
    tree = ast.parse(textwrap.dedent(inspect.getsource(v4.partner_scores_c4)))
    body = tree.body[0].body
    stops = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "source" for t in node.targets)]
    need(len(stops) == 5, "frozen A4 loader construction changed")
    scope = dict(v4.__dict__)
    exec(compile(ast.Module(body=body[:stops[0]], type_ignores=[]), "<imported-a4-loader>", "exec"), scope)
    overlays = [ast.literal_eval(node.args[2]) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace_once" and len(node.args) == 3
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "center3 = center3_windows_from(volume, centers, 224)"]
    need(len(overlays) == 1, "frozen A4 C4 overlay changed")
    overlay = textwrap.dedent("\n".join(line.strip() for line in overlays[0].splitlines()))
    code = compile(overlay, "<imported-a4-windows>", "exec")

    def windows(volume, centers):
        from raptor_radimagenet import center3_windows_from, normalise
        from raptor_runtime import normalize_input
        ns = dict(volume=volume, centers=centers, center3_windows_from=center3_windows_from)
        exec(code, ns)
        return {"l2": normalise(ns["center3"], "radimagenet"),
                "convnext": normalize_input(ns["c4"], "imagenet")}

    return scope["scope"]["_load_models"], windows


def q7_entries(manifest):
    entries = [e for e in manifest["entries"] if e["family"] in ("l2", "convnext")]
    need([(e["family"], e["seed"], e["fold"]) for e in entries] ==
         [("l2", s, f) for s in (42, 43) for f in range(5)] +
         [("convnext", 42, f) for f in range(5)], "Q7 partner roster changed")
    need(all(e.get("crop_policy") == "mixed" for e in entries if e["family"] == "convnext"),
         "Q7 requires A4 mixed-scale C4")
    return entries


def candidate_specs(overrides=None):
    weights = {**DEFAULT_WEIGHTS, **(overrides or {})}
    need(set(weights) == set(CANDIDATES), "unknown candidate weight override")
    for weight in weights.values():
        need(type(weight) in (int, float) and 0 <= weight <= 1, "invalid partner weight")
    return {name: dict(family=name.split('-')[0], partner_weight=float(weight))
            for name, weight in weights.items()}


def blend(candidate, ids, s42, s43, q, *, public=None, exp155=None, s6=None, partner_weight=None):
    import numpy as np
    v2, v5 = maths()
    ids = v5.validate_ids(ids)
    n = len(ids)
    own = (probability(s42, (n, 12), "S42").astype(np.float64) +
           probability(s43, (n, 12), "S43").astype(np.float64)) / 2
    q = probability(q, (n, 12), "Q")
    need(candidate in CANDIDATES, "unknown candidate")
    spec = candidate_specs({candidate: partner_weight} if partner_weight is not None else None)[candidate]
    weight, family = spec["partner_weight"], spec["family"]
    if family == "B6":
        need(public is None and exp155 is None and s6 is None, "B6 forbids public/OAI/S6")
        anchor = own
    else:
        anchor = v5.trio_values("A5-172" if family == "A6" else "A5q-172", ids,
                                public, s6, own, exp155)
    return (1 - weight) * v2.rank(anchor) + weight * v2.rank(q)


def candidate_values(specs, selected, ids, s42, s43, q, **extra):
    """All eligible final blends from already computed members; no inference."""
    need(selected in specs and specs == candidate_specs({k: s["partner_weight"] for k, s in specs.items()}),
         "candidate registration differs")
    names = [name for name in specs if specs[selected]["family"] != "B6" or specs[name]["family"] == "B6"]
    return {name: blend(name, ids, s42, s43, q, partner_weight=specs[name]["partner_weight"],
                        **(extra if specs[name]["family"] != "B6" else {})) for name in names}


def study_failure(job, uid, member, exc):
    job.setdefault("failures", []).append(dict(UID=uid, member=member,
        exception_type=type(exc).__name__, message=str(exc),
        traceback=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))))


def study_prediction(job, uid, member, predict):
    """One attempt; a missing whole-study vector travels to the rank blend."""
    import numpy as np
    try:
        value = np.asarray(predict())
        if not np.isfinite(value).all():
            raise FloatingPointError("nonfinite study output")
        return probability(value, (12,), member)
    except Exception as exc:
        study_failure(job, uid, member, exc)
        return np.full(12, np.nan, np.float32)


def observed_ranks(values):
    """Rank only observed studies; never let NaNs poison other study ranks."""
    import numpy as np
    v2, _ = maths()
    valid = np.isfinite(values).all(axis=1)
    result = np.full(values.shape, np.nan, np.float64)
    if valid.any():
        result[valid] = v2.rank(values[valid])
    return result


def release_values(specs, selected, ids, anchors, scores, *, s6=None, failures=()):
    """Healthy arithmetic stays verbatim. Failed probability families are absent.

    A blend member is a probability-averaged family (R50, C4, MIL, CoAtNet),
    or Public R / EXP155 / S6. One failed constituent invalidates that family's
    study vector: no damaged slices or incomplete seed/fold averages are used.
    Donors are original observed ranks only: MIL, then C4 (OOF-selected).
    If neither is observed, use remaining original weights renormalised.
    Replacements then pass through the original nested rank blend.
    """
    import numpy as np
    v2, v5 = maths()
    extra = dict(public=anchors['public'], exp155=anchors['exp155'], s6=s6) if s6 is not None else {}
    if all(np.isfinite(a).all() for a in [*anchors.values(), *scores.values(), *([s6] if s6 is not None else [])]):
        q, _ = q_values(scores)
        return candidate_values(specs, selected, ids, anchors['s42'], anchors['s43'], q, **extra), []
    raw = dict(own=(anchors['s42'].astype(np.float64)+anchors['s43'].astype(np.float64))/2,
               l2=((scores['l2'][:5]+scores['l2'][5:])/2).mean(0),
               convnext=scores['convnext'].mean(0), mil=scores['mil'].mean(0), **extra)
    ranked = {k: observed_ranks(a) for k,a in raw.items()}
    results, replacements = {}, []
    for name, spec in specs.items():
        family, w = spec['family'], spec['partner_weight']
        if specs[selected]['family'] == 'B6' and family != 'B6':
            continue
        weights = dict(l2=w/3, convnext=w/3, mil=w/3)
        weights.update(dict(own=1-w) if family == 'B6' else
                       dict(own=(1-w)/4, exp155=(1-w)/4, public=(1-w)/4, s6=(1-w)/4) if family == 'A6q' else
                       dict(own=(1-w)/3, exp155=(1-w)/3, public=(1-w)*.7/3, s6=(1-w)*.3/3))
        ranks = {k:ranked[k].copy() for k in weights}
        for i, uid in enumerate(ids):
            missing = [k for k in weights if not np.isfinite(ranks[k][i]).all()]
            if not missing:
                continue
            donors = {k:wt for k,wt in weights.items() if k not in missing and wt > 0}
            need(bool(donors), 'no remaining member for study ' + uid + ' / ' + name)
            preferred = next((k for k in ('mil','convnext') if k in donors), None)
            if preferred is not None:
                donors = {preferred: donors[preferred]}
            total = sum(donors.values())
            donor_weights = {k:wt/total for k,wt in donors.items()}
            replacement = sum(ranked[k][i]*wt for k,wt in donor_weights.items())
            for key in missing:
                ranks[key][i] = replacement
                component = ('s42', 's43') if key == 'own' else (key,)
                causes = [f for f in failures if f['UID'] == uid and any(f['member'] == c or f['member'].startswith(c+'/') for c in component)]
                if not causes:
                    tmp = {}
                    try:
                        raise FloatingPointError('nonfinite saved member study vector')
                    except FloatingPointError as exc:
                        study_failure(tmp, uid, key, exc)
                    causes = tmp['failures']
                replacements.append(dict(UID=uid, member=key, candidate=name,
                    exception_type=causes[0]['exception_type'], traceback=causes[0]['traceback'],
                    causes=causes, donor_weights=donor_weights, replacement_rank=replacement.tolist()))
        q = v2.rank(sum(v2.rank(ranks[k]) for k in ('l2','convnext','mil'))/3)
        if family == 'B6':
            anchor = ranks['own']
        elif family == 'A6q':
            anchor = v5.trio_round_trip(ids, sum(ranks[k] for k in ('exp155','public','own','s6'))/4)
        else:
            # P954's original percentile-rank convention and float32 boundary.
            n = len(ids)
            p954 = ((.7*ranks['public']+.3*ranks['s6'])*max(1,n-1)/n+1/n).astype(np.float32).astype(np.float64)
            anchor = v5.trio_round_trip(ids, (ranks['exp155']+v2.rank(p954)+ranks['own'])/3)
        results[name] = (1-w)*v2.rank(anchor)+w*v2.rank(q)
    return results, replacements


@contextlib.contextmanager
def s6_one_attempt(s6):
    """Adapt S6's executable diagnostic copy; pinned reader/source stay intact."""
    original = s6._diagnostic_source
    def source(source_dir, out):
        folder = original(source_dir, out)
        path = folder / 'compact_infer.py'
        code = path.read_text(encoding='utf-8')
        for old, new, count in [
            ('zs = run_shared(img, meta, mask, "fp32")', 'raise FloatingPointError("nonfinite S6 shared logits")', 1),
            ('z = run_model(m, img, meta, mask, "fp32")', 'raise FloatingPointError("nonfinite S6 logits")', 2)]:
            need(code.count(old) == count, 'S6 retry source changed')
            code = code.replace(old, new)
        old = 'fallback[uid] = "non-finite score"'
        new = ('try:\n                raise FloatingPointError("nonfinite S6 mean logits")\n'
               '            except FloatingPointError as exc:\n'
               '                fallback[uid] = f"{type(exc).__name__}: {exc}\\n" + traceback.format_exc()')
        code = s6._replace_once(code, old, new, 'nonfinite traceback')
        path.write_text(code, encoding='utf-8')
        return folder
    s6._diagnostic_source = source
    try:
        yield
    finally:
        s6._diagnostic_source = original


def s6_failed_rows(ids, values, receipt, failures):
    """Discard S6 controller placeholder means; only other members may donate."""
    import numpy as np
    import re
    values = values.copy()
    reasons = receipt['fallback_reasons']
    need(receipt['fallback_studies'] == len(reasons), 'S6 missing failure diagnostics')
    for uid, reason in reasons.items():
        need(uid in ids, 'S6 failure UID outside cohort')
        values[ids.index(uid)] = np.nan
        types = re.findall(r'(\w+(?:Error|Exception)):', reason)
        if 'Traceback (most recent call last)' in reason:
            failures.append(dict(UID=uid, member='s6', exception_type=types[-1] if types else 'RuntimeError', message=reason.splitlines()[0], traceback=reason))
        else:
            job = dict(failures=failures)
            try:
                raise RuntimeError('S6 worker failure: ' + reason)
            except RuntimeError as exc:
                study_failure(job, uid, 's6', exc)
    return values


def mil_meta(ck, entry):
    expected = dict(arch="convnext_tiny.fb_in22k_ft_in1k", res=224, lab=list(LABELS),
                    contract="raptor64_336_c140_s02", norm="imagenet", window_policy="within-slot-valid",
                    fold=entry["fold"], epoch=19)
    need(all(ck.get(k) == v for k, v in expected.items()), "MIL metadata mismatch")
    need(ck.get("model_kind", ck.get("provenance", {}).get("model_kind")) == "finding_spatial_tokens_v1",
         "not finding-specific spatial MIL")
    need(ck.get("input_policy") == "center3" and ck.get("k_eval") == 24 and
         ck.get("flip_tta") == "coronal_axial_horizontal_mean_probability", "MIL inference policy differs")
    if entry.get("spec_sha256"):
        need(ck.get("provenance", {}).get("spec_sha256") == entry["spec_sha256"], "MIL spec binding differs")
    else:
        need(ck.get("seed", ck.get("provenance", {}).get("seed")) == entry["seed"], "MIL seed differs")


def load_mil(path, entry, device):
    import torch
    from training.arms import mil
    need(entry.get("sha256") and sha(path) == entry["sha256"] and Path(path).stat().st_size == entry["bytes"],
         "missing or changed MIL hash slot")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mil_meta(ck, entry)
    model = mil.FindingMIL(mil.common.build_backbone(mil.ARCH, pretrained=False))
    mil.strict_load(model, ck["model"])
    return model.eval().to(device)


def mil_windows(vol, centers):
    import numpy as np
    from training.arms import mil
    x = mil.scale_windows(vol, centers)
    return (x - np.array([.485, .456, .406], np.float32).reshape(3, 1, 1)) / np.array(
        [.229, .224, .225], np.float32).reshape(3, 1, 1)


def infer(model, x, device, *, anchor=False):
    import numpy as np
    import torch
    tensor = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))
    tensor = tensor.to(device)
    amp = torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else contextlib.nullcontext()
    with torch.inference_mode(), amp:
        logits = model(tensor[None])
        if anchor:
            logits = logits.float()
            need(tuple(logits.shape) == (1, 12) and bool(torch.isfinite(logits).all()), "invalid logits")
            out = torch.sigmoid(logits)[0].cpu().numpy()
    if not anchor:
        need(tuple(logits.shape) == (1, 12) and bool(torch.isfinite(logits).all()), "invalid logits")
        out = torch.sigmoid(logits.float())[0].cpu().numpy()
    return probability(out, (12,), "model")


def mil_prediction(model, x, uid, corpus, device):
    import torch
    from training.arms import mil
    flipped = mil.flip_inputs(torch.from_numpy(x)[None], [uid], corpus, 36)[0]
    return (infer(model, x, device) + infer(model, flipped, device)) * .5


def source_files(root):
    """Only named code and its serving import closure, no training data."""
    root = Path(root)
    files = {"tools/exp169_release_v6.py": root / "tools/exp169_release_v6.py",
             "tools/exp169_v4_frozen.py": root / V4 / "code/assets/exp169_release_v4.py",
             "tools/exp169_release_v5q.py": root / "tools/exp169_release_v5q.py",
             "tools/exp189_s6_port.py": root / "tools/exp189_s6_port.py",
             "tools/exp152_own0949_release.py": root / B172 / "own-anchor/assets/exp152_own0949_release.py",
             "config/workspace.toml": root / "config/workspace.toml"}
    for p in (root / B172 / "assets/code").glob("*.py"):
        files["tools/" + p.name if p.name.startswith("exp169") else "legacy/" + p.name] = p
    for name in ("arms/mil.py", "arms/lifecycle.py", "arms/mil_registration.py", "runners/mil_arms.py",
                 "src/raptor_common.py", "src/raptor_exp185.py", "src/raptor_runtime.py"):
        files["training/" + name] = root / "training" / name
    for p in (root / "tools/exp189_s6_src").glob("*.py"):
        files["tools/exp189_s6_src/" + p.name] = p
    return files


def file_pin(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def register(out, *, root=ROOT, collected=None, f0_binding=None, candidate_weights=None):
    root = Path(root)
    collected = Path(collected) if collected else root / COLLECTED_OUTPUTS
    specs = candidate_specs(read(candidate_weights) if candidate_weights else None)
    entries = []
    for seed in (42, 43):
        for fold in range(5):
            if seed == 42:
                p = (root / "artifacts/experiments/EXP185_FINDING_MIL/outputs/S42/results/EXP185_FINDING_MIL_F0_S42/model_best.pt"
                     if fold == 0 else root / f"artifacts/experiments/EXP186_FINDING_MIL_5FOLD/outputs-b/fold{fold}/results/EXP186B_FINDING_MIL_S42_F{fold}/model_best.pt")
                spec = None
            else:
                p = root / F0 / "outputs/model_best.pt" if fold == 0 else collected / f"fold{fold}/outputs/model_best.pt"
                spec = "43fd7a50bd45b6b4c2047975b313c7bf3557424abad30a072c14ec433906f435" if fold == 0 else SPEC_PINS[fold-1]
            entry = {"seed": seed, "fold": fold, "name": f"mil-s{seed}-f{fold}.pt", "spec_sha256": spec,
                     "path": str(p.resolve()), "sha256": None, "bytes": None}
            if seed == 43:
                sp = root / F0.parent / "spec.json" if fold == 0 else root / COLLECTED / f"training/configs/mil-member-s43-f{fold}-v1.json"
                doc = read(sp)
                need(sha(sp) == spec and doc["seed"] == 43 and doc["fold"] == fold and
                     doc["family"] == "convnext-mil", "MIL spec changed")
                if fold:
                    need(doc["run_kind"] == "member" and doc["schema"] == "arm-member-spec/v1", "not registered member spec")
                entry["spec_pin"] = file_pin(sp)
                entry["scope"] = ("member/" if fold else "") + f"convnext-mil/{doc['arm']}/seed43/fold{fold}/{doc['attempt']}"
            if p.exists():
                entry.update(file_pin(p))
                import torch
                mil_meta(torch.load(p, map_location="cpu", weights_only=False), entry)
                if seed == 43:
                    receipt_path = p.parent.parent / "receipt.json"
                    rc = read(receipt_path)
                    need(rc["status"] == "ARTIFACTS_VERIFIED" and rc["exit_code"] == 0 and
                         rc["spec_sha256"] == spec and rc["scope"] == entry["scope"] and
                         rc["outputs"]["model_best.pt"] == entry["sha256"],
                         "MIL collection receipt differs")
                    entry["collection_receipt"] = file_pin(receipt_path)
                    collection_path = p.parent.parent / "collection.json"
                    collection = read(collection_path)
                    need(collection.get("schema") == "arm-collection/v1" and
                         collection.get("status") == "COLLECTED_ARTIFACTS_VERIFIED" and
                         collection.get("spec_sha256") == spec and
                         collection.get("receipt_sha256") == entry["collection_receipt"]["sha256"] and
                         len(collection.get("archive_sha256", "")) == 64,
                         "MIL collection binding differs")
                    entry["collection_record"] = file_pin(collection_path)
                    entry["collected_archive_sha256"] = collection["archive_sha256"]
            entries.append(entry)
    screen_pin = None
    if all(e["sha256"] for e in entries if e["seed"] == 43 and e["fold"]):
        screen = read(collected / "screen.json")
        need(screen.get("schema") == "mil-s43-members-screen/v1" and
             screen.get("status") == "SCREENED_FOLDS_1_4" and
             [f["fold"] for f in screen["folds"]] == [1, 2, 3, 4], "MIL package not screened")
        for item in screen["folds"]:
            need(item["status"] == "SCREENED" and
                 item["receipt_sha256"] == entries[5 + item["fold"]]["collection_receipt"]["sha256"],
                 "MIL screen receipt differs")
        screen_pin = file_pin(collected / "screen.json")
    sources = {name: file_pin(p) for name, p in source_files(root).items()}
    for fold in range(1, 5):
        spec = read(root / COLLECTED / f"training/configs/mil-member-s43-f{fold}-v1.json")
        for name, expected in spec["inputs"]["sources"].items():
            if name in sources:
                need(sources[name]["sha256"] == expected, "MIL member serving source changed: " + name)
    need(sources["tools/exp189_s6_port.py"]["sha256"] == S6_SHA, "S6 port changed")
    assets = {}
    for key, stage, filename, slug in (
        ("partner", B172, "exp169-assets.zip.bin", "taeyangg4/knee-exp169-b172-v2-assets"),
        ("s42", B172 / "own-anchor", "exp152-assets.zip.bin", "taeyangg4/knee-exp169-b172-own-v2-assets"),
        ("s43", V4 / "s43", "exp169-s43-v4.zip.bin", "taeyangg4/knee-exp169-s43-v4-assets"),
        ("exp155", E155, "exp155-assets.zip.bin", "taeyangg4/knee-exp155-own0949-oai-v1-2-assets")):
        assets[key] = {**file_pin(root / stage / "dataset" / filename), "dataset": slug, "filename": filename,
                       "manifest_sha256": sha(root / stage / "assets/release-manifest.json")}
    v4 = a4_module()
    assets["c4"] = {**file_pin(root / V4 / "c4/dataset" / v4.C4_ARCHIVE),
                    "dataset": v4.C4_DATASET, "filename": v4.C4_ARCHIVE,
                    "manifest_sha256": sha(root / V4 / "c4/release-manifest.json")}
    need(all(assets["c4"][k] == v4.UPLOADED[v4.C4_DATASET][k] for k in ("bytes", "sha256")),
         "frozen A4 C4 archive changed")
    binding = file_pin(f0_binding) if f0_binding else None
    if binding:
        b = read(f0_binding)
        need(b.get("schema") == "exp169-v6-f0-adoption/v1" and b.get("seed") == 43 and b.get("fold") == 0
             and b.get("checkpoint_sha256") == entries[5]["sha256"]
             and b.get("spec_sha256") == entries[5]["spec_sha256"]
             and b.get("collection_receipt_sha256") == entries[5]["collection_receipt"]["sha256"]
             and b.get("selected_by") == "B" and bool(b.get("registration_evidence")),
             "F0 member selection evidence missing; B must bind the actual registered choice")
    record = dict(schema="exp169-v6-registration/v1", version=VERSION, sources=sources, assets=assets,
                  mil=entries, f0_member_binding=binding, member_screen=screen_pin,
                  candidates=specs,
                  candidate_weight_input=file_pin(candidate_weights) if candidate_weights else None,
                  f0_selection="B selected explicit-v3 collected F0, 2026-10-11; same refactored MIL runner/recipe as F1-F4" if binding else "explicit-v3 F0 selection unresolved",
                  new_dataset="taeyangg4/knee-exp169-v6-mil-v1",
                  timing_policy="measurement only; Kaggle platform limit only")
    write_new(out, record)
    return validate(out)


def validate(registration, *, local=True):
    m = read(registration) if not isinstance(registration, dict) else registration
    need(m["version"] == VERSION and m["schema"] == "exp169-v6-registration/v1", "registration version differs")
    need(m["candidates"] == candidate_specs({k: s["partner_weight"] for k, s in m["candidates"].items()}),
         "candidate registration differs")
    need([(e["seed"], e["fold"]) for e in m["mil"]] == [(s, f) for s in (42, 43) for f in range(5)], "MIL order differs")
    pending = [e["name"] for e in m["mil"] if not e["sha256"]]
    if local:
        pins = [*m["sources"].values(), *m["assets"].values(), *[e for e in m["mil"] if e["sha256"]]]
        pins += [e["collection_receipt"] for e in m["mil"] if "collection_receipt" in e]
        pins += [e["collection_record"] for e in m["mil"] if "collection_record" in e]
        pins += [e["spec_pin"] for e in m["mil"] if "spec_pin" in e]
        if m["f0_member_binding"]:
            pins.append(m["f0_member_binding"])
        pins += [m[k] for k in ("member_screen", "candidate_weight_input") if m.get(k)]
        for p in pins:
            need(Path(p["path"]).stat().st_size == p["bytes"] and sha(p["path"]) == p["sha256"], "registered bytes changed: " + p["path"])
    blockers = ["missing " + p for p in pending]
    if not m["f0_member_binding"]:
        blockers.append("F0 explicit-v3 member adoption binding unresolved")
    if not m.get("member_screen"):
        blockers.append("S43 F1-F4 package screen missing")
    return dict(status="BLOCKED" if blockers else "READY_FOR_GPU_QUALIFICATION", blockers=blockers,
                pinned_members=10-len(pending), version=VERSION)


def deterministic_zip(path, files):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, data if isinstance(data, bytes) else Path(data).read_bytes())


def safe_extract(data, destination):
    destination = Path(destination).resolve()
    # File-backed assets avoid holding multi-GB archives twice in RAM.
    with zipfile.ZipFile(io.BytesIO(data) if isinstance(data, bytes) else data) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), "duplicate ZIP members")
        for entry in archive.infolist():
            name = entry.orig_filename  # ZipInfo normalises backslashes on Windows.
            target = (destination / name).resolve()
            need(target.is_relative_to(destination) and "\\" not in name, "unsafe ZIP member")
        archive.extractall(destination)


def mil_manifest(registration):
    return dict(schema="exp169-v6-mil-dataset/v1", dataset=registration["new_dataset"],
                weights=[{k: e[k] for k in ("name", "seed", "fold", "spec_sha256", "sha256", "bytes")}
                         for e in registration["mil"]],
                f0_binding_sha256=registration["f0_member_binding"]["sha256"])


def check_mil_upload(registration, folder):
    folder = Path(folder)
    need(read(folder / "mil-manifest.json") == mil_manifest(registration), "MIL upload manifest differs")
    for entry in registration["mil"]:
        p = folder / entry["name"]
        need(p.stat().st_size == entry["bytes"] and sha(p) == entry["sha256"], "MIL upload weight differs")
    return folder


def prepare(registration, out):
    import copy
    out = Path(out)
    need(not out.exists(), "stage must be new")
    status = validate(registration)
    m = read(registration)
    out.mkdir(parents=True)
    runtime = copy.deepcopy(m)
    for p in runtime["sources"].values():
        p.pop("path")
    for e in runtime["mil"]:
        e.pop("path")
        e.pop("collection_receipt", None)
        e.pop("collection_record", None)
        e.pop("spec_pin", None)
    if runtime["f0_member_binding"]:
        runtime["f0_member_binding"].pop("path")
    for name in ("member_screen", "candidate_weight_input"):
        if runtime.get(name):
            runtime[name].pop("path")
    for p in runtime["assets"].values():
        p.pop("path")
    files = {k: Path(p["path"]) for k, p in m["sources"].items()}
    for pkg in ("tools", "training", "training/arms", "training/src", "training/runners"):
        files[pkg + "/__init__.py"] = b""
    files["registration.json"] = (json.dumps(runtime, sort_keys=True) + "\n").encode()
    deterministic_zip(out / "runtime.zip", files)
    code_sha = sha(out / "runtime.zip")
    upload = out / "mil-dataset"
    upload.mkdir()
    # A complete loose-file dataset avoids uploading duplicate loose/ZIP copies.
    if not status["blockers"]:
        for entry in m["mil"]:
            shutil.copyfile(entry["path"], upload / entry["name"])
        write_new(upload / "mil-manifest.json", mil_manifest(m))
        check_mil_upload(m, upload)
    write_new(upload / "dataset-metadata.json", dict(id=m["new_dataset"], title="EXP169 v6 finding MIL weights v1",
              licenses=[{"name": "CC0-1.0"}]))
    pinned_bytes = sum(e["bytes"] or 0 for e in m["mil"])
    write_new(out / "upload-list.json", dict(new_datasets=[dict(id=m["new_dataset"], files=m["mil"],
              pinned_payload_bytes=pinned_bytes, missing_slots=10-status["pinned_members"],
              estimated_full_payload_bytes=pinned_bytes + (10-status["pinned_members"])*112189754,
              pinned_upload_seconds_at_800000_Bps=pinned_bytes/800000)],
              reuse_byte_identically=m["assets"], public_weights="nartaa/rsna-knee-publication-swa-weights-20261007",
              s6=["heliosli/rsna-knee-compact-v1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]))
    base_meta = read(ROOT / V4 / "A4/kernel/kernel-metadata.json")
    payload = base64.b64encode((out / "runtime.zip").read_bytes()).decode()
    for candidate, spec in m["candidates"].items():
        cell = f'''import base64, hashlib, io, zipfile, pathlib, subprocess, sys, os
p = base64.b64decode({payload!r})
assert hashlib.sha256(p).hexdigest() == {code_sha!r}
root = pathlib.Path('/kaggle/working/v6-code')
assert not root.exists(), 'fresh runtime required'
with zipfile.ZipFile(io.BytesIO(p)) as z:
    assert all(not n.startswith('/') and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
    z.extractall(root)
env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
subprocess.run([sys.executable, str(root/'tools/exp169_release_v6.py'), 'serve', '--registration', str(root/'registration.json'), '--candidate', {candidate!r}, '--out', '/kaggle/working/v6-run'], env=env, check=True)
'''
        folder = out / candidate
        folder.mkdir()
        write_new(folder / "release.ipynb", dict(nbformat=4, nbformat_minor=5,
            metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, cells=[
            dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=cell.splitlines(True))]))
        meta = copy.deepcopy(base_meta)
        slug = "knee-exp169-" + candidate.lower() + "-v1"
        meta.update(id="taeyangg4/" + slug, title=slug,
                    code_file="release.ipynb", dataset_sources=[m["assets"][k]["dataset"] for k in ("partner", "s42", "s43", "c4")] + [m["new_dataset"]])
        if spec["family"] != "B6":
            meta["dataset_sources"] += [m["assets"]["exp155"]["dataset"], "nartaa/rsna-knee-publication-swa-weights-20261007",
                                       "heliosli/rsna-knee-compact-v1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
        write_new(folder / "kernel-metadata.json", meta)
        write_new(folder / "candidate.json", dict(candidate=candidate, **spec, bundle="Q7", version=VERSION,
                  registration_sha256=sha(registration), runtime_sha256=code_sha))
    receipt = dict(status="STAGED_UNQUALIFIED", admission=status, registration_sha256=sha(registration),
                   candidates=m["candidates"],
                   files={str(p.relative_to(out)).replace("\\", "/"): sha(p) for p in out.rglob("*") if p.is_file()})
    write_new(out / "stage-receipt.json", receipt)
    return dict(status=receipt["status"], admission=status, stage=str(out), files=len(receipt["files"]))


def admit(registration, stage):
    status = validate(registration)
    rc = read(Path(stage) / "stage-receipt.json")
    need(rc["registration_sha256"] == sha(registration), "stage registration changed")
    for name, expected in rc["files"].items():
        need(sha(Path(stage) / name) == expected, "stage file changed: " + name)
    return {**status, "stage_integrity": "PASS", "qualification": "2xT4 visible and hidden untested"}


def dataset_root(slug):
    name = slug.split("/")[-1]
    roots = [Path("/kaggle/input") / name, Path("/kaggle/input/datasets") / slug,
             Path("/kaggle/input/competitions") / name]
    found = [p for p in roots if p.is_dir()]
    need(len(found) == 1, "missing/ambiguous dataset mount: " + slug)
    return found[0]


def mount(slug, filename):
    roots = [dataset_root(slug)]
    matches = [p for root in roots if root.exists() for p in root.rglob(filename)]
    need(len(matches) == 1, "missing/ambiguous dataset file: " + slug + "/" + filename)
    return matches[0]


_LEGACY_TF32 = None


def backend(strict=False):
    import torch
    global _LEGACY_TF32
    if _LEGACY_TF32 is None:
        _LEGACY_TF32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
                        torch.get_float32_matmul_precision())
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if strict:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        torch.set_float32_matmul_precision(_LEGACY_TF32[2])
        torch.backends.cudnn.allow_tf32 = _LEGACY_TF32[0]
        torch.backends.cuda.matmul.allow_tf32 = _LEGACY_TF32[1]


def anchor_models(paths, candidate, device, *, selected=None):
    ns = load_module("v6_anchor_serving", Path(paths["s42"]) / "public-serving.py").__dict__
    result = {}
    family = candidate_specs()[candidate]["family"]
    keys = ("s42", "s43", *(["exp155"] if family != "B6" else []))
    need(selected is None or set(selected) <= set(keys), "invalid anchor smoke selection")
    for key in keys if selected is None else selected:
        asset = Path(paths[key])
        manifest = read(asset / "release-manifest.json")
        need(sha(asset / "public-serving.py") == manifest["serving_sha256"], "serving source changed")
        adapter_path = (asset / "exp155_own0949_oai_release_v12.py" if key == "exp155" else
                        Path(paths["s42"]) / "exp152_own0949_release.py")
        adapter = load_module("v6_adapter_" + key, adapter_path)
        if key == "s43":
            adapter.OWN_NAME = manifest["checkpoint"]
            def s43_meta(ck, protocol_sha, adapter=adapter, run_dir=manifest["run_dir"]):
                expected = dict(arch=adapter.ARCH, res=384, inference_res=320, lab=list(LABELS),
                    contract=adapter.CONTRACT, k_eval=94, norm="imagenet", window_policy="legacy", oai=False,
                    checkpoint_kind="fixed-swa-bn", swa_epochs_1based=[14, 11, 12], seed=43)
                need(all(ck.get(k) == v for k, v in expected.items()) and
                     ck.get("provenance", {}).get("protocol_sha256") == protocol_sha and
                     ck["provenance"].get("run_dir") == run_dir, "S43 metadata mismatch")
            adapter.own_meta = s43_meta
            manifest = dict(protocol_sha256=manifest["protocol_sha256"], own_state_schema=manifest["own_state_schema"],
                            checkpoints={adapter.OWN_NAME: {"sha256": manifest["checkpoint_sha256"]}})
        name = adapter.OWN_NAME
        ns.update(_E152_MANIFEST=manifest, _E152_FILES={name: asset / name})
        result[key] = adapter.checked_model(ns, asset / name, adapter.ARCH, 384, device)[0]
    if family != "B6" and selected is None:
        p = Path(paths["public"])
        need(sha(p) == PUBLIC_SHA, "public R checkpoint changed")
        adapter = load_module("v6_public_adapter", Path(paths["s42"]) / "exp152_own0949_release.py")
        ns.update(_E152_MANIFEST={"checkpoints": {PUBLIC_NAME: {"sha256": PUBLIC_SHA}}}, _E152_FILES={PUBLIC_NAME: p})
        result["public"] = adapter.checked_model(ns, p, adapter.ARCH, 384, device)[0]
    backend()
    return ns, result


def anchor_reader(ns):
    import types
    # Decode the pinned embedded reader in memory: never writes into scored assets.
    fast = types.ModuleType("v6_fastread")
    exec(compile(base64.b64decode(ns["_FASTREAD_B64"]), "<pinned-fastread>", "exec"), fast.__dict__)
    original_records, original_result, original_glob = fast._records, fast._result, fast.glob.glob
    def records(out):
        need(all(row[3] for row in out), "anchor DICOM header failure")
        return original_records(out)
    def result(sel, got, med_ps):
        need(all(got[i][0] is not None for i, _ in sel), "anchor DICOM pixel failure")
        return original_result(sel, got, med_ps)
    def glob(pattern):
        found = original_glob(pattern)
        need(bool(found), "selected anchor series has no DICOM files")
        return found
    fast._records, fast._result, fast.glob = records, result, types.SimpleNamespace(glob=glob)
    # Windows lacks pread; whole-file I/O feeds the identical decoder/LUT path.
    variant = ns["FASTREAD_VARIANT"]
    if not hasattr(os, "pread"):
        fast._read_whole = lambda path: Path(path).read_bytes()
        variant = "studyone_" + "_".join(variant.split("_")[1:])
    return fast.make_builder(variant, 384, ns["SLOTS"], hdr_threads=ns["FR_HDR"], px_threads=ns["FR_PX"])


def anchor_worker(job, device):
    import numpy as np
    ns, models = anchor_models(job["paths"], job["candidate"], device)
    reader = anchor_reader(ns)
    rows = {}
    for row in job["rows"]:
        rows.setdefault(row[UID], []).append(row)
    outputs = {k: [] for k in models}
    for uid in job["ids"]:
        try:
            vol, mask = reader(uid, rows, job["images"])
            need(np.count_nonzero(mask) >= 3 and np.any(vol), "empty anchor study")
            plain = ns["gpu_windows"](vol, mask, 94, 320, device)
            mirror = ns["gpu_windows"](vol, mask, 94, 320, device, mirror=True)
        except Exception as exc:
            for key in models:
                study_failure(job, uid, key, exc)
                outputs[key].append(np.full(12, np.nan, np.float32))
            continue
        for key, model in models.items():
            outputs[key].append(study_prediction(job, uid, key, lambda:
                (infer(model, plain, device, anchor=True) + infer(model, mirror, device, anchor=True)) * .5))
    return {key: np.asarray(value, np.float32) for key, value in outputs.items()}


def partner_worker(job, device):
    import numpy as np
    import torch
    partner = Path(job["paths"]["partner"])
    sys.path.insert(0, str(partner / "code"))
    v2, _ = maths()
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    from raptor_common import select_centers
    manifest = read(partner / "release-manifest.json")
    entries = q7_entries(manifest)
    loader, windows = a4_policy()
    models = loader(partner, entries, device)
    mil_models = [load_mil(Path(job["paths"]["mil"]) / e["name"], e, device) for e in job["mil"]]
    backend()
    corpus = DicomCorpus.from_records(job["ids"], job["rows"], DirectoryDicomSource(job["images"]))
    scores = {k: np.full((count, len(job["ids"]), 12), np.nan, np.float32) for k, count in (("l2", 10), ("convnext", 5), ("mil", 10))}
    for i, uid in enumerate(job["ids"]):
        try:
            vol, mask = corpus.volume(uid), corpus.mask(uid)
            need(np.count_nonzero(mask) >= 3 and np.any(vol), "empty partner study")
            centers = select_centers(mask, 64, 36, corpus.spec, "within-slot-valid")
        except Exception as exc:
            for key in scores:
                study_failure(job, uid, key, exc)
            continue
        backend()
        try:
            flags = np.asarray(v2.base.plane_flags(centers, corpus.spec), bool)
            inputs = windows(vol, centers)
        except Exception as exc:
            for key in ('l2','convnext'):
                study_failure(job, uid, key, exc)
        else:
            for model, e in zip(models, entries):
                def predict():
                    x = inputs[e["family"]]
                    mirror = x.copy()
                    mirror[flags] = mirror[flags, :, :, ::-1]
                    return (infer(model, x, device) + infer(model, mirror, device)) / 2
                slot = e["fold"] + (5 if e["family"] == "l2" and e["seed"] == 43 else 0)
                scores[e["family"]][slot, i] = study_prediction(job, uid,
                    f"{e['family']}/s{e['seed']}/f{e['fold']}", predict)
        try:
            x = mil_windows(vol, centers)
        except Exception as exc:
            study_failure(job, uid, 'mil', exc)
            continue
        backend(strict=True)
        for j, model in enumerate(mil_models):
            scores["mil"][j, i] = study_prediction(job, uid, f'mil/slot{j}',
                lambda: mil_prediction(model, x, uid, corpus, device))
    return scores


def worker(job, out, device):
    import numpy as np
    job = read(job)
    started = time.monotonic()
    backend()
    values = anchor_worker(job, device) if job["phase"] == "anchor" else partner_worker(job, device)
    with Path(out).open("xb") as stream:
        np.savez(stream, ids=job["ids"], **values)
    write_new(str(out) + ".json", dict(status="PASS", seconds=time.monotonic()-started, sha256=sha(out),
                                     studies=len(job["ids"]), phase=job["phase"], device=device,
                                     failures=job.get('failures', [])))


def run_shards(job, out, phase):
    """Study errors are data; worker startup/process errors still propagate."""
    import numpy as np
    processes, outputs = [], []
    try:
        for gpu in range(min(2, len(job["ids"]))):
            spec = {**job, "phase": phase, "ids": job["ids"][gpu::2]}
            jp, op = Path(out) / f"{phase}-{gpu}.json", Path(out) / f"{phase}-{gpu}.npz"
            write_new(jp, spec)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8")
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__)), "worker", "--job", str(jp), "--out", str(op), "--device", "cuda:0"], env=env))
            outputs.append(op)
        # Polling detects either worker's failure without waiting on a sibling.
        while any(p.poll() is None for p in processes):
            need(all(p.poll() in (None, 0) for p in processes), "member worker failed")
            time.sleep(.25)
        need(all(p.returncode == 0 for p in processes), "member worker failed")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait()
    shards = []
    for p in outputs:
        receipt = read(str(p) + ".json")
        need(receipt["sha256"] == sha(p), "worker output changed")
        job.setdefault('failures', []).extend(receipt.get('failures', []))
        shards.append(np.load(p, allow_pickle=False))
    values = {}
    for key in shards[0].files:
        if key == "ids":
            continue
        axis = 0 if phase == "anchor" else 1
        merged = np.concatenate([s[key] for s in shards], axis=axis)
        ids = [str(uid) for s in shards for uid in s["ids"]]
        need(len(ids) == len(set(ids)) and set(ids) == set(job["ids"]), "worker cohort differs")
        values[key] = np.take(merged, [ids.index(uid) for uid in job["ids"]], axis=axis)
    return values


def serve(registration, candidate, out):
    import numpy as np
    import torch
    m = read(registration)
    need(not validate(m, local=False)["blockers"], "unfilled weight pins or unresolved F0: cannot serve")
    need(candidate in m["candidates"], "candidate not registered")
    family = m["candidates"][candidate]["family"]
    for name, pin in m["sources"].items():
        need(sha(ROOT / name) == pin["sha256"], "runtime source changed: " + name)
    need(torch.cuda.device_count() == 2 and all("T4" in torch.cuda.get_device_name(i) for i in range(2)), "two T4s required")
    out = Path(out)
    need(not out.exists() and not Path("/kaggle/working/submission.csv").exists(), "fresh output required")
    out.mkdir(parents=True)
    started = time.monotonic()
    paths = {}
    for key in ("partner", "s42", "s43", "c4", *(["exp155"] if family != "B6" else [])):
        pin = m["assets"][key]
        archive = mount(pin["dataset"], pin["filename"])
        need(sha(archive) == pin["sha256"], "asset archive changed")
        target = out / key
        safe_extract(archive, target)
        matches = list(target.rglob("release-manifest.json"))
        need(len(matches) == 1 and sha(matches[0]) == pin["manifest_sha256"], "asset manifest changed")
        paths[key] = str(matches[0].parent)
    a4_module().overlay_c4(paths["partner"], paths["c4"], m["assets"]["c4"]["manifest_sha256"])
    paths["mil"] = str(check_mil_upload(m, mount(m["new_dataset"], "mil-manifest.json").parent))
    if family != "B6":
        paths["public"] = str(mount("nartaa/rsna-knee-publication-swa-weights-20261007", PUBLIC_NAME))
    sample = mount("competitions/rsna-knee-abnormality-detection", "sample_submission.csv")
    competition = sample.parent
    _, v5 = maths()
    with sample.open(newline="") as stream:
        reader = csv.DictReader(stream)
        need(tuple(reader.fieldnames) == (UID, *LABELS), "sample schema differs")
        ids = v5.validate_ids([row[UID] for row in reader])
    with (competition / "test_series.csv").open(newline="") as stream:
        rows = [{k: row[k] for k in (UID, "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive")} for row in csv.DictReader(stream)]
    job = dict(candidate=candidate, ids=ids, rows=rows, images=str(competition / "test_series"), paths=paths, mil=m["mil"])
    anchors = run_shards(job, out, "anchor")
    scores = run_shards(job, out, "partner")
    extra = {}
    if family != "B6":
        from tools import exp189_s6_port as s6
        bundle = mount(s6.DATASET_SLUG, "state_spec.json").parent
        # Resolve the dataset root; prepare_code searches its archive and wheels.
        code_root = dataset_root("heliosli/rsna-orthofoundation-fold0-epoch7-20261004")
        code, anatomy = s6.prepare_code(code_root, out / "s6-code", install_wheels=True)
        with s6_one_attempt(s6):
            srows = s6.infer_studies([competition / "test_series" / uid for uid in ids], bundle_dir=bundle,
                                   code_dir=code, anatomy_weights=anatomy, output_dir=out / "s6", workers=2,
                                   amp="auto", device="cuda", max_fallback_fraction=1)
        v5.validate_ids(ids, [row[UID] for row in srows])
        extra = dict(public=anchors["public"], exp155=anchors["exp155"],
                     s6=s6_failed_rows(ids, np.asarray([[row[label] for label in LABELS] for row in srows]),
                         read(out / 's6/controller/submission_receipt.json'), job.setdefault('failures', [])))
    results, replacements = release_values(m["candidates"], candidate, ids, anchors, scores,
                                           s6=extra.get('s6'), failures=job.get('failures', []))
    with (out / "members.npz").open("xb") as stream:
        np.savez(stream, ids=ids, **anchors, **scores, **({"s6": extra["s6"]} if extra else {}))
    candidate_receipts = {}
    for name, result in results.items():
        path = out / (name + ".csv")
        v5._write_csv(path, ids, result)
        v5._read_csv(path)
        candidate_receipts[name] = dict(**m["candidates"][name], sha256=sha(path))
    temporary = out / "submission.pending.csv"
    shutil.copyfile(out / (candidate + ".csv"), temporary)
    v5._read_csv(temporary)
    write_new(out / "receipt.json", dict(status="PASS_COMPLETE_INFERENCE", candidate=candidate, version=VERSION,
              partner_weight=m["candidates"][candidate]["partner_weight"], candidates=candidate_receipts,
              registration_sha256=sha(registration), studies=len(ids), elapsed_seconds=time.monotonic()-started,
              members_sha256=sha(out / "members.npz"), submission_sha256=sha(temporary),
              no_fallback=not replacements, replacements=replacements, failures=job.get('failures', [])))
    os.replace(temporary, "/kaggle/working/submission.csv")


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("register")
    q.add_argument("--out", required=True)
    q.add_argument("--collected")
    q.add_argument("--f0-binding")
    q.add_argument("--candidate-weights", help="JSON map of candidate name to partner weight; registered before inference")
    for name in ("validate", "prepare", "admit"):
        q = sub.add_parser(name)
        q.add_argument("--registration", required=True)
        if name != "validate":
            q.add_argument("--out" if name == "prepare" else "--stage", required=True)
    q = sub.add_parser("worker")
    q.add_argument("--job", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--device", required=True)
    q = sub.add_parser("serve")
    q.add_argument("--registration", required=True)
    q.add_argument("--candidate", choices=CANDIDATES, required=True)
    q.add_argument("--out", required=True)
    args = vars(p.parse_args(argv))
    command = args.pop("command")
    result = globals()[command](**args)
    print(json.dumps(result, indent=2))
    if command == "admit" and result["status"] == "BLOCKED":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
