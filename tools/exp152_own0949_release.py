"""EXP152 v1.0.1: create-only EXP143-v2 serving, no account operations.

Only public3 initializes CUDA (explicitly run by B on an exclusively claimed GPU).
average/stage are CPU operations; parity-plan/push-plan only write local plans.
The generated notebooks reuse the SHA-pinned public serving implementation.
Changelog: v1.0.1 bounds decoded preprocessing results by submitting one
QUEUE_MAX batch at a time and materializes _fastread.py in the parent.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

VERSION = "1.0.1"
ROOT = Path(__file__).resolve().parents[1]
TOOL = Path(__file__).resolve()
PROTOCOL = ROOT / "artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json"
RUN = "artifacts/experiments/EXP143_OWN0949/S42v2"
SOURCE = ROOT / "storage/external/nartaa__rsna-knee-0949-anatomical-mirror/rsna-knee-0949-anatomical-mirror.ipynb"
SOURCE_SHA = "8f1f96530b08543bd6caeed48489eea7c01657d6994f22e21a8400047fd4e100"
PUBLIC_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
PUBLIC_NAME = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
PUBLIC_DATASET = "nartaa/rsna-knee-publication-swa-weights-20261007"
OWN_NAME = "exp143-v2-e14-e11-e12-bn.pt"
ASSET_DATASET = "taeyangg4/knee-exp152-own0949-v2-assets"
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
CONTRACT = "exp143_raptor96_384_c140_s02"
UID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
EPOCHS = [14, 11, 12]
KERNELS = {"own": "knee-exp152-own0949-v2", "blend": "knee-exp152-own0949-blend"}
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
TOL = {"precision": "fp16", "max_abs": 2e-3, "mean_abs": 5e-4,
       "integer_input_hashes": "exact", "normalized_window_hashes": "exact",
       "final_csv": "byte-identical", "on_failure": "HOLD; do not relax post hoc"}
POLICY = {"autocast": "fp16", "batch_studies": 1, "chunk_windows": 16,
          "eval": True, "compile": False, "cudnn_benchmark": True,
          "matmul_allow_tf32": True, "sdpa": "public notebook / timm defaults",
          "note": "Preserve qualified 0.949 runtime; no numerical-policy ablation in a checkpoint-only release"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_text(path, text):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def fresh_dir(path):
    path = Path(path).resolve()
    need(not path.exists(), "Output already exists; use a fresh versioned path: " + str(path))
    path.mkdir(parents=True)
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def state_schema(state):
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sorted(state.items())}


def safe_load(path):
    import torch
    # EXP143 run_identity stores torch.__version__, a TorchVersion str subclass.
    # Permit that one inert type; never fall back to unrestricted unpickling.
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def own_meta(ck, protocol_sha, run_dir=RUN):
    expected = dict(arch=ARCH, res=384, inference_res=320, lab=TARGETS, contract=CONTRACT,
                    k_eval=94, norm="imagenet", window_policy="legacy", oai=False,
                    checkpoint_kind="fixed-swa-bn", swa_epochs_1based=EPOCHS, seed=42)
    for k, v in expected.items():
        need(k in ck and ck[k] == v, "Own checkpoint metadata mismatch: " + k)
    need(ck.get("provenance", {}).get("protocol_sha256") == protocol_sha,
         "Own checkpoint protocol SHA mismatch")
    need(ck["provenance"].get("run_dir") == run_dir, "Own checkpoint run mismatch")
    need(isinstance(ck.get("model"), dict) and ck["model"], "Missing model state")


def average_states(states):
    """Exact copy of frozen exp112 swa_average arithmetic, without its broad imports.

    Preserve supplied [14,11,12] order and last-member integer-buffer semantics.
    CPU tests cross-check this against the frozen function extracted by AST.
    """
    import torch
    need(bool(states), "No SWA members")
    keys = list(states[-1])
    need(all(set(s) == set(keys) for s in states), "SWA keys mismatch")
    result = OrderedDict()
    for key in keys:
        vals = [s[key].detach().cpu() for s in states]
        ref = vals[-1]
        need(all(v.shape == ref.shape and v.dtype == ref.dtype for v in vals),
             "SWA shape/dtype mismatch: " + key)
        if ref.is_floating_point():
            need(all(bool(torch.isfinite(v).all()) for v in vals), "Nonfinite SWA member")
            acc = torch.zeros(ref.shape, dtype=torch.float64)
            for v in vals:
                acc += v.to(torch.float64)
            result[key] = (acc / len(vals)).to(ref.dtype)
        else:
            need(key.endswith("num_batches_tracked") or all(torch.equal(v, ref) for v in vals),
                 "Integer SWA member mismatch: " + key)
            result[key] = ref.clone()
    return result


def merge_replayed_bn(averaged, final, bn_keys, batches):
    """Use only the trainer's proven final BN buffers, never rerun BN on a new GPU."""
    import torch
    need(state_schema(averaged) == state_schema(final), "Final/SWA schema mismatch")
    need(bn_keys and set(bn_keys) <= set(final), "BN buffer set invalid")
    result = OrderedDict()
    for k, v in averaged.items():
        w = final[k].detach().cpu()
        need(not w.is_floating_point() or bool(torch.isfinite(w).all()), "Nonfinite final state")
        if k in bn_keys:
            if k.endswith("num_batches_tracked"):
                need(w.numel() == 1 and int(w.item()) == batches, "Incomplete BN replay")
            if k.endswith("running_var"):
                need(bool((w >= 0).all()), "Negative BN variance")
            result[k] = w.clone()
        else:
            need(torch.equal(v, w), "Final non-BN state is not registered average: " + k)
            result[k] = v
    need(all(torch.equal(result[k], final[k]) for k in result), "Reconstructed final state differs")
    return result


def average(args):
    import torch
    # load_registration verifies every frozen source hash and registration content.
    guard = load_module("exp152_v2_guard", ROOT / "tools/exp143_v2.py")
    p = guard.load_registration(args.protocol)
    need(p["recipe"]["swa_1based"] == EPOCHS and p["run_dirs"]["local"] == RUN,
         "Only Amendment A2 S42v2 [14,11,12] is supported")
    run = guard.registered_run(p, "local")
    finished = run / "model_best.pt"
    if args.checkpoint:
        need(Path(args.checkpoint).resolve() == finished.resolve(), "Checkpoint must be S42v2/model_best.pt")
    summary = read_json(run / "summary.json")
    need(summary.get("status") == "COMPLETE_TRAINING_NO_EVALUATION" and summary.get("epochs") == 16,
         "Training / BN replay not complete")
    need(summary.get("model_sha256") == sha(finished), "Final checkpoint SHA mismatch")
    ck = safe_load(finished)
    own_meta(ck, sha(args.protocol))
    need(summary.get("identity") == ck["provenance"], "Summary identity differs")
    n = len(p["train_ids"])
    bn, batches = summary["bn"], math.ceil(n / p["recipe"]["bn_batch"])
    need(bn.get("studies") == bn.get("unique_studies") == n and bn.get("batches") == batches
         and bn.get("num_batches_tracked") == [batches] and bn.get("mode") == "train"
         and bn.get("gradients") == "none", "Incomplete train-only BN receipt")
    expected_ids = hashlib.sha256("\n".join(sorted(p["train_ids"])).encode()).hexdigest()
    need(bn.get("studies_sha256") == expected_ids, "BN replay cohort differs")
    states, members = [], {}
    for e in EPOCHS:
        path = run / f"epoch_{e:02d}.pt"
        members[str(e)] = sha(path)
        need(summary.get("swa_member_sha256", {}).get(str(e)) == members[str(e)], "Epoch SHA mismatch")
        item = safe_load(path)
        need(item.get("epoch") == e and set(item) == {"model", "epoch"}, "Epoch payload mismatch")
        states.append(item["model"])
    avg = average_states(states)
    del states
    train = load_module("exp152_training_base", ROOT / "training/src/exp143_own0949.py")
    model = train.build_model(320)  # CPU construction, no forward / CUDA call.
    model.load_state_dict(ck["model"], strict=True)
    from torch.nn.modules.batchnorm import _BatchNorm
    modules = [(name, m) for name, m in model.named_modules() if isinstance(m, _BatchNorm)]
    bn_keys = {name + "." + k for name, m in modules for k in
               ("running_mean", "running_var", "num_batches_tracked") if getattr(m, k) is not None}
    need(len(modules) == bn.get("bn_modules"), "BN module count mismatch")
    reconstructed = merge_replayed_bn(avg, ck["model"], bn_keys, batches)
    model.load_state_dict(reconstructed, strict=True)
    del model
    # Metadata remains identical to final predictor checkpoint; torch file bytes may differ.
    payload = dict(ck, model=reconstructed)
    out = fresh_dir(args.out)
    with (out / OWN_NAME).open("xb") as f:
        torch.save(payload, f)
    receipt = dict(schema="exp152-average-v1", version=VERSION, status="VERIFIED_FIXED_SWA_BN",
                   checkpoint=OWN_NAME, checkpoint_sha256=sha(out / OWN_NAME),
                   source_final_sha256=sha(finished), summary_sha256=sha(run / "summary.json"),
                   protocol_sha256=sha(args.protocol), run_dir=RUN, epochs_1based=EPOCHS,
                   epoch_sha256=members, state_schema=state_schema(reconstructed),
                   bn_buffer_keys=sorted(bn_keys), bn=bn,
                   bn_handling="reconstructed float64 SWA + authenticated trainer replay buffers; exact final-state equality",
                   source_bindings=p["source_bindings"], oai=False)
    write_json(out / "average-receipt.json", receipt)
    return {"status": receipt["status"], "out": str(out), "checkpoint_sha256": receipt["checkpoint_sha256"]}


def replace_once(source, old, new):
    need(source.count(old) == 1, "Pinned source marker changed: " + old[:90])
    return source.replace(old, new, 1)


def replace_function(source, name, replacement):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    need(len(nodes) == 1, "Public function missing/duplicated: " + name)
    n = nodes[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[:n.lineno-1]) + replacement.rstrip() + "\n" + "".join(lines[n.end_lineno:])


def public_source():
    need(sha(SOURCE) == SOURCE_SHA, "Public notebook SHA mismatch")
    nb = read_json(SOURCE)
    need([c["cell_type"] for c in nb["cells"]] == ["markdown"] * 3 + ["code"] * 2,
         "Public notebook structure changed")
    code = nb["cells"][4]["source"]
    return "".join(code), nb["cells"][:3]


def serving_source(source):
    """Preserve all model/window/decode functions; patch bindings and scheduling.

    Shared-window movement is a scheduling optimization in blend, not new math.
    Checkpoint SHA/metadata/schema validation runs before any GPU thread starts.
    """
    s = replace_function(source, "load_model", '''def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    return _E152.checked_model(globals(), pt_path, arch_default, res_default, device)
''')
    s = replace_function(s, "find_weight_file", '''def find_weight_file(fname):
    return str(_E152_FILES[fname])
''')
    s = replace_once(s, 'def main():\n', '''def _exp152_bounded_preprocess(pool, study_ids, decoder, batch_size):
    """Limit the pool result handler to one batch of decoded studies at a time."""
    assert batch_size > 0
    for begin in range(0, len(study_ids), batch_size):
        batch = study_ids[begin:begin + batch_size]
        yield from pool.imap_unordered(decoder, batch, chunksize=2)


def main():
''')
    s = replace_once(s, 'if __name__ == "__main__":\n    main()', "# EXP152 entrypoint is invoked explicitly after binding verification.")
    s = replace_once(s, 'q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()',
                     'q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()\n    _exp152_errors = []\n    _exp152_inputs = {}')
    s = replace_once(s, '    threads = [threading.Thread(target=consumer, args=(d,), daemon=True) for d in devs]',
                     '    _load_fastread()  # Materialize/import once in the parent before pool workers.\n'
                     '    threads = [threading.Thread(target=consumer, args=(d,), daemon=True) for d in devs]')
    s = replace_once(s, 'pool.imap_unordered(_pre_one, test_ids, chunksize=2)',
                     '_exp152_bounded_preprocess(pool, test_ids, _pre_one, QUEUE_MAX)')
    old = '''                for a, (model, res) in enumerate(models[d]):
                    xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM) if xla is not None else gpu_windows(vol, mask, K_EVAL, res, d)
                    assert xla is None, "Anatomical L/R TTA requires CUDA/fp16"
                    xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)
                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)
                    del xm'''
    new = '''                assert xla is None, "Anatomical L/R TTA requires CUDA/fp16"
                assert mask.any(), "Empty decoded study"
                xw = gpu_windows(vol, mask, K_EVAL, INPUT_RES, d)
                xm = gpu_windows(vol, mask, K_EVAL, INPUT_RES, d, norm=NORM, mirror=True)
                _exp152_inputs[sid] = _E152.input_record(globals(), vol, mask, xw, xm, N == 3)
                for a, (model, res) in enumerate(models[d]):
                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)
                del xw, xm'''
    s = replace_once(s, old, new)
    s = replace_once(s, 'print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)',
                     '_exp152_errors.append((i, type(e).__name__, str(e)))')
    s = replace_once(s, '    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).',
                     '    _E152.need(not _exp152_errors, "Inference failed; no submission: " + repr(_exp152_errors))\n'
                     '    _E152.save_raw(_E152_WORK, test_ids, arm_probs, _exp152_inputs, _E152_MANIFEST, _E152_VARIANT)\n'
                     '    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).')
    old_rank = '''    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]),
                         axes=(0, 0))                                          # (N,12) in [0,1]'''
    s = replace_once(s, old_rank, '    ranks = _E152.output_values(arm_probs, _E152_VARIANT, rankpct)')
    s = replace_once(s, 'out = "/kaggle/working/submission.csv"', 'out = str(_E152_WORK / "submission.csv")')
    s = replace_once(s, '    print(sub.head().to_string(index=False), flush=True)',
                     '    _E152.finish_receipt(_E152_WORK, time.time()-t0)')
    compile(s, "exp152-public-serving.py", "exec")
    return s


def percentile_average(values):
    """(average 1-based rank - 1)/(N-1), the registered EXP143 fixed_blends rule."""
    need(values and all(math.isfinite(float(x)) for x in values), "Invalid rank values")
    order = sorted(range(len(values)), key=values.__getitem__)
    out, at = [0.] * len(values), 0
    while at < len(order):
        end = at + 1
        while end < len(order) and values[order[end]] == values[order[at]]:
            end += 1
        for i in order[at:end]:
            out[i] = ((at + end - 1) / 2) / max(1, len(order)-1)
        at = end
    return out


def output_values(arms, variant, public_rank):
    import numpy as np
    need(variant in KERNELS and len(arms) == (1 if variant == "own" else 2), "Wrong arm set")
    shape = np.asarray(arms[0]).shape
    need(len(shape) == 2 and shape[0] > 0 and shape[1] == 12, "Invalid prediction shape")
    need(all(np.asarray(a).shape == shape and np.isfinite(a).all() and
             (np.asarray(a) >= 0).all() and (np.asarray(a) <= 1).all() for a in arms), "Invalid probabilities")
    if variant == "own":
        # Exactly the public single-arm ordinal rank implementation, including ties.
        return public_rank(np.clip(arms[0], 0, 1))
    ranks = [np.asarray([percentile_average(np.asarray(a)[:, j].tolist()) for j in range(12)]).T for a in arms]
    return .5 * ranks[0] + .5 * ranks[1]  # no second rank / no tuning knob


def runtime_info():
    import importlib.metadata as md
    import torch
    versions = {}
    for package in ("torch", "timm", "numpy", "pandas", "pydicom", "opencv-python", "opencv-python-headless", "pylibjpeg"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = None
    return dict(python=sys.version, packages=versions, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                devices=[dict(name=torch.cuda.get_device_name(i), capability=list(torch.cuda.get_device_capability(i)))
                         for i in range(torch.cuda.device_count())],
                policy=POLICY, actual=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                    mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                    math_sdp=torch.backends.cuda.math_sdp_enabled(), threads=torch.get_num_threads()))


def array_sha(a):
    import numpy as np
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.ascontiguousarray(a)
    return jhash({"shape": list(a.shape), "dtype": str(a.dtype),
                  "bytes": hashlib.sha256(a.tobytes()).hexdigest()})


def input_record(ns, vol, mask, plain, mirror, detailed=True):
    record = {"volume": array_sha(vol), "mask": array_sha(mask),
              "centers": jhash([int(v) for v in ns["_eval_centers"](mask, 96, 94)])}
    if detailed:
        record.update(plain=array_sha(plain), mirror=array_sha(mirror))
    return record


def checked_model(ns, path, arch, res, device):
    import torch
    manifest = ns["_E152_MANIFEST"]
    matches = [name for name, p in ns.get("_E152_FILES", {}).items()
               if Path(p).resolve() == Path(path).resolve()]
    name = matches[0] if len(matches) == 1 else Path(path).name
    need(name in manifest["checkpoints"], "Unregistered checkpoint path")
    expected = manifest["checkpoints"][name]
    need(sha(path) == expected["sha256"], "Bound checkpoint SHA mismatch")
    need(arch == ARCH and res == 384, "Arm architecture/resolution mismatch")
    ck = safe_load(path)
    need(ck.get("arch") == ARCH and ck.get("res") == 384, "Checkpoint architecture/resolution mismatch")
    if name == OWN_NAME:
        own_meta(ck, manifest["protocol_sha256"])
        need(state_schema(ck["model"]) == manifest["own_state_schema"], "Own checkpoint keys/shapes/dtypes mismatch")
    bb = ns["build_backbone"](ARCH, pretrained=False)
    model = ns["RaptorClassifier"](bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    return model, 320


def save_raw(work, ids, arms, inputs, manifest, variant):
    import numpy as np
    need(len(ids) == len(set(ids)) and set(ids) == set(inputs), "Incomplete cohort/input diagnostics")
    output_values(arms, variant, lambda a: a)  # shape/finite/range checks
    arrays = {"ids": np.asarray(ids), "targets": np.asarray(TARGETS), "own": arms[0]}
    if variant == "blend":
        arrays["public"] = arms[1]
    with (Path(work) / "raw.npz").open("xb") as f:
        np.savez(f, **arrays)
    write_json(Path(work) / "inputs.json", inputs)
    write_json(Path(work) / "runtime.json", runtime_info())
    write_json(Path(work) / "binding.json", dict(manifest_sha256=jhash(manifest), variant=variant,
               own_sha256=manifest["checkpoints"][OWN_NAME]["sha256"],
               public_sha256=PUBLIC_SHA if variant == "blend" else None, source_sha256=SOURCE_SHA,
               serving_sha256=manifest["serving_sha256"], tolerance=TOL))


def finish_receipt(work, seconds):
    work = Path(work)
    write_json(work / "receipt.json", dict(status="COMPLETE_INFERENCE", seconds=seconds,
               files={n: sha(work / n) for n in ("raw.npz", "inputs.json", "runtime.json", "binding.json", "submission.csv")}))


def bind(ns, manifest, files, variant, work):
    need(variant in KERNELS, "Unknown variant")
    import torch
    need(torch.cuda.is_available(), "CUDA/fp16 is required")
    ns.update(_E152=sys.modules[__name__], _E152_MANIFEST=manifest, _E152_FILES=files,
              _E152_VARIANT=variant, _E152_WORK=Path(work))
    names = [OWN_NAME] + ([PUBLIC_NAME] if variant == "blend" else [])
    ns["ARMS"] = [dict(file=n, arch=ARCH, res=384, w=1. if variant == "own" else .5) for n in names]


def kernel_metadata(variant, dataset=ASSET_DATASET):
    return dict(id="taeyangg4/" + KERNELS[variant], title=KERNELS[variant], code_file="release.ipynb",
                language="python", kernel_type="notebook", is_private=True, enable_gpu=True,
                enable_tpu=False, enable_internet=False, machine_shape="NvidiaTeslaT4", docker_image=DOCKER,
                dataset_sources=[dataset] + ([PUBLIC_DATASET] if variant == "blend" else []),
                competition_sources=["rsna-knee-abnormality-detection"], kernel_sources=[], model_sources=[])


def notebook(variant, dataset, archive_sha, manifest_sha, module_sha, original_markdown):
    slug = dataset.split("/")[1]
    title = "EXP152 own0949 v2 — OAI-free own checkpoint" if variant == "own" else (
        "EXP152 own0949 blend — OAI-DERIVED; TAEYANG APPROVAL REQUIRED BEFORE SUBMISSION")
    description = ("Reproduced nartaa/dreaddevelopment 0.949 serving pipeline; K94, native320 crop, "
                   "anatomical mirror, fp16 chunk16. Fixed Amendment A2 epochs14/11/12 with train-only BN. "
                   "No training labels, report text or raw external data are packaged. Private, internet off. ")
    if variant == "blend":
        description += ("H-B-63 registered per-target average-tie percentile rank blend, own/public=0.5/0.5, "
                        "no final re-ranking. Public checkpoint includes OAI lineage. Dataset licence is 'other'; "
                        "author notices retained below. Every submission requires Taeyang's explicit approval. ")
    description += ("Original introductory markdown below describes an older recipe; the SHA-pinned executable "
                    "source is authoritative. Kernel metadata TITLE stays the exact requested slug for initial creation.")
    setup = f'''# No pip, network, training data or credentials.
from pathlib import Path
import hashlib, json, zipfile, sys
_mounts = [Path('/kaggle/input/{slug}'), Path('/kaggle/input/datasets/{dataset}')]
_archive = next((p/'exp152-assets.zip.bin' for p in _mounts if (p/'exp152-assets.zip.bin').is_file()), None)
if _archive is None: raise FileNotFoundError('Attach the private EXP152 .zip.bin asset dataset')
with _archive.open('rb') as _f:
    if hashlib.file_digest(_f, 'sha256').hexdigest() != {archive_sha!r}: raise ValueError('Archive SHA mismatch')
_assets = Path('/kaggle/working/exp152-assets')
_assets.mkdir(exist_ok=False)
with zipfile.ZipFile(_archive) as _z:
    _allowed = {{{OWN_NAME!r}, 'exp152_own0949_release.py', 'public-serving.py', 'release-manifest.json'}}
    if len(_z.namelist()) != len(_allowed) or set(_z.namelist()) != _allowed: raise ValueError('Archive member mismatch')
    _z.extractall(_assets)
def _hash(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
if _hash(_assets/'release-manifest.json') != {manifest_sha!r}: raise ValueError('Manifest SHA mismatch')
if _hash(_assets/'exp152_own0949_release.py') != {module_sha!r}: raise ValueError('Runtime source SHA mismatch')
sys.path.insert(0, str(_assets))
import exp152_own0949_release as _e152
_manifest = json.loads((_assets/'release-manifest.json').read_text())
_e152.verify_assets(_assets, _manifest)
_files = {{{OWN_NAME!r}: _assets/{OWN_NAME!r}}}
'''
    if variant == "blend":
        setup += f'''_public_mounts = [Path('/kaggle/input/{PUBLIC_DATASET.split('/')[1]}'), Path('/kaggle/input/datasets/{PUBLIC_DATASET}')]
_files[{PUBLIC_NAME!r}] = next((p/{PUBLIC_NAME!r} for p in _public_mounts if (p/{PUBLIC_NAME!r}).is_file()), None)
if _files[{PUBLIC_NAME!r}] is None: raise FileNotFoundError('Attach exact public 0.949 weights')
'''
    setup += f'''_work = Path('/kaggle/working')
if (_work/'submission.csv').exists(): raise FileExistsError('Stale submission.csv')
exec(compile((_assets/'public-serving.py').read_text(), 'public-serving.py', 'exec'), globals())
_e152.bind(globals(), _manifest, _files, {variant!r}, _work)
main()
'''
    compile(setup, "exp152-notebook", "exec")
    cells = [dict(cell_type="markdown", metadata={}, source="# " + title + "\n\n" + description)]
    cells += original_markdown
    cells += [dict(cell_type="code", metadata={}, source=setup, execution_count=None, outputs=[])]
    return dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                nbformat=4, nbformat_minor=4), title, description


def verify_assets(path, manifest):
    path = Path(path)
    need(manifest.get("version") == VERSION and manifest.get("public_source_sha256") == SOURCE_SHA,
         "Release version/source mismatch")
    for name, expected in manifest["asset_sha256"].items():
        need(Path(name).name == name and sha(path / name) == expected, "Asset hash mismatch: " + name)
    need(manifest["tolerance"] == TOL and manifest["epochs_1based"] == EPOCHS,
         "Frozen release tolerance/epoch list differs")


def stage(args):
    receipt = read_json(Path(args.average) / "average-receipt.json")
    need(receipt.get("status") == "VERIFIED_FIXED_SWA_BN" and receipt.get("epochs_1based") == EPOCHS,
         "Average receipt invalid")
    checkpoint = Path(args.average) / OWN_NAME
    need(sha(checkpoint) == receipt["checkpoint_sha256"], "Average checkpoint SHA mismatch")
    ck = safe_load(checkpoint)
    own_meta(ck, receipt["protocol_sha256"])
    need(state_schema(ck["model"]) == receipt["state_schema"], "Average schema mismatch")
    need(re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+", args.dataset), "Invalid dataset ID")
    source, credits = public_source()
    patched = serving_source(source)
    out = fresh_dir(args.out)
    assets = fresh_dir(out / "assets")
    shutil.copyfile(checkpoint, assets / OWN_NAME)
    shutil.copyfile(TOOL, assets / TOOL.name)
    write_text(assets / "public-serving.py", patched)
    manifest = dict(schema="exp152-release-v1", version=VERSION,
                    protocol_sha256=receipt["protocol_sha256"], epochs_1based=EPOCHS,
                    average_receipt_sha256=sha(Path(args.average) / "average-receipt.json"),
                    source_final_sha256=receipt["source_final_sha256"],
                    public_source_sha256=SOURCE_SHA, serving_sha256=sha(assets / "public-serving.py"),
                    checkpoints={OWN_NAME: dict(sha256=receipt["checkpoint_sha256"], oai=False),
                                 PUBLIC_NAME: dict(sha256=PUBLIC_SHA, oai=True)},
                    own_state_schema=receipt["state_schema"],
                    asset_sha256={n: sha(assets / n) for n in (OWN_NAME, TOOL.name, "public-serving.py")},
                    ranks=dict(own="public numpy double argsort / (N-1)",
                               blend="average ties (rank-1)/(N-1), .5 own + .5 public; no final re-rank"),
                    policy=POLICY, tolerance=TOL, dataset=args.dataset,
                    source_bindings=receipt["source_bindings"], approval_required={"own": "each submission", "blend": "OAI-derived; Taeyang explicit approval each submission"})
    write_json(assets / "release-manifest.json", manifest)
    dataset = fresh_dir(out / "dataset")
    archive = dataset / "exp152-assets.zip.bin"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
        for p in sorted(assets.iterdir()):
            z.write(p, p.name)
    write_json(dataset / "dataset-metadata.json", dict(title=args.dataset.split("/")[1], id=args.dataset,
               licenses=[{"name": "other"}], description="Private EXP143 v2 OAI-free model, fixed epochs14/11/12 and trainer BN. EXP152 serving runtime. No source training data."))
    kernels = {}
    for variant in KERNELS:
        dest = fresh_dir(out / variant)
        nb, title, desc = notebook(variant, args.dataset, sha(archive), sha(assets / "release-manifest.json"),
                                   sha(assets / TOOL.name), credits)
        write_json(dest / "release.ipynb", nb)
        write_json(dest / "kernel-metadata.json", kernel_metadata(variant, args.dataset))
        write_text(dest / "DESCRIPTION.md", "# " + title + "\n\n" + desc + "\n")
        kernels[variant] = {n: sha(dest / n) for n in ("release.ipynb", "kernel-metadata.json", "DESCRIPTION.md")}
    report = dict(status="STAGED_NOT_UPLOADED", version=VERSION, private_required=True,
                  archive_sha256=sha(archive), manifest_sha256=sha(assets / "release-manifest.json"),
                  dataset_metadata_sha256=sha(dataset / "dataset-metadata.json"),
                  kernels=kernels, projection=projection(), dataset=args.dataset)
    write_json(out / "build-receipt.json", report)
    return {"status": report["status"], "out": str(out), "archive_sha256": report["archive_sha256"]}


def checked_stage(stage_dir):
    path = Path(stage_dir).resolve()
    report = read_json(path / "build-receipt.json")
    need(report.get("version") == VERSION, "Stage version mismatch")
    need(sha(path / "assets/release-manifest.json") == report["manifest_sha256"], "Stage manifest mismatch")
    need(sha(path / "dataset/exp152-assets.zip.bin") == report["archive_sha256"], "Stage archive mismatch")
    need(sha(path / "dataset/dataset-metadata.json") == report["dataset_metadata_sha256"], "Dataset metadata mismatch")
    manifest = read_json(path / "assets/release-manifest.json")
    verify_assets(path / "assets", manifest)
    need(sha(TOOL) == manifest["asset_sha256"][TOOL.name], "Executing tool differs from staged tool")
    for variant, files in report["kernels"].items():
        for name, digest in files.items():
            need(sha(path / variant / name) == digest, "Kernel build hash mismatch")
    return path, report, manifest


def projection():
    return dict(studies=1300, gpus="2xT4", own_seconds=1.52*1300+180,
                blend_seconds=2*1.52*1300+240,
                conservative_own_seconds=1.5*1.52*1300+300,
                conservative_blend_seconds=1.5*2*1.52*1300+300,
                basis="D-B-065 corrected timing60: 88s/58 ~=1.52s/study, K94+mirror 2xT4. Blend doubles forward-work proxy despite shared decode. Setup180/240s and 1.5x+300s are allowances, not measurements.",
                caveat="EXP152 unmeasured; hidden tails, decode I/O, timm stack, dual-model VRAM and diagnostic overhead may differ. No hidden bound or score forecast.")


def parity_plan(args):
    stage_dir, report, manifest = checked_stage(args.stage)
    with Path(args.test_csv).open(encoding="utf-8-sig", newline="") as f:
        ids = [r[UID] for r in csv.DictReader(f)]
    need(len(ids) == len(set(ids)) == 3 and all(ids), "Plan requires exactly3 visible test UIDs")
    out = fresh_dir(args.out)
    write_json(out / "tol.json", TOL)
    plan = dict(schema="exp152-parity-plan-v1", frozen_before_run=True, version=VERSION,
                stage_manifest_sha256=report["manifest_sha256"], tolerance_sha256=sha(out / "tol.json"),
                ids=ids, test_csv_sha256=sha(args.test_csv), series_csv_sha256=sha(args.series_csv),
                cohort="all three visible test.csv UIDs in their original order; same12 targets; no labels",
                reference="Kaggle 2xT4 pinned Docker", development="B local CUDA / fp16",
                policy=POLICY, tolerance=TOL,
                sequence=["Local public3 compares decoded uint8/mask/centers and normalized windows with frozen local predictor",
                          "Local raw probabilities and both final CSVs must be byte-identical to predictor-derived references",
                          "Push the two private kernels only after local PASS; B downloads full outputs",
                          "compare-parity gates input hashes, finite raw max2e-3/mean5e-4, final CSV bytes; retain runtime differences"],
                restrictions=["No T4 tolerance relaxation after run", "No label/rank/weight/recipe tuning",
                              "Public3 cannot establish hidden runtime or full distribution parity",
                              "Public notebook benchmark/TF32/SDPA policy retained to isolate checkpoint swap; strict-parity reconfiguration would require another release"],
                notebooks=report["kernels"], checkpoint_sha256=manifest["checkpoints"])
    write_json(out / "parity-plan.json", plan)
    return {"status": "FROZEN_NOT_RUN", "out": str(out), "plan_sha256": sha(out / "parity-plan.json"), "tolerance_sha256": sha(out / "tol.json")}


def verify_plan(plan_dir, report):
    plan_dir = Path(plan_dir)
    p = read_json(plan_dir / "parity-plan.json")
    need(p.get("frozen_before_run") is True and p["stage_manifest_sha256"] == report["manifest_sha256"], "Parity plan binding mismatch")
    need(p["tolerance_sha256"] == sha(plan_dir / "tol.json") and read_json(plan_dir / "tol.json") == TOL,
         "Frozen tolerance mismatch")
    return p


def csv_write(path, ids, values):
    import numpy as np
    import pandas as pd
    need(not Path(path).exists(), "CSV already exists")
    frame = pd.DataFrame(np.asarray(values).astype(np.float32), columns=TARGETS)
    frame.insert(0, UID, ids)
    frame.to_csv(path, index=False)


def public3(args):
    """Explicit B-only GPU check. Uses real local predictor helpers, no labels/reports."""
    import numpy as np
    import torch
    stage_dir, report, manifest = checked_stage(args.stage)
    plan = verify_plan(args.plan, report)
    need(sha(args.protocol) == manifest["protocol_sha256"], "Public3 protocol binding mismatch")
    need(sha(args.public_checkpoint) == PUBLIC_SHA, "Public checkpoint SHA mismatch")
    # Revalidate protocol/source lineage before importing the predictor.
    guard = load_module("exp152_local_guard", ROOT / "tools/exp143_v2.py")
    p = guard.load_registration(args.protocol)
    run = guard.registered_run(p)
    need(sha(run / "model_best.pt") == manifest["source_final_sha256"], "Predictor final checkpoint mismatch")
    comp = Path(args.competition)
    with (comp / "test.csv").open(encoding="utf-8-sig", newline="") as f:
        ids = [r[UID] for r in csv.DictReader(f)]
    need(len(ids) == len(set(ids)) == 3, "public3 requires exactly3 unique visible test UIDs")
    need(ids == plan["ids"] and sha(comp / "test.csv") == plan["test_csv_sha256"]
         and sha(comp / "test_series.csv") == plan["series_csv_sha256"], "Public3 inputs differ from frozen plan")
    records = guard.series_records(comp / "test_series.csv")
    image_root = Path(args.dicom_root) if args.dicom_root else next(
        (comp / n for n in ("test_series", "test_images") if (comp / n).is_dir()), None)
    need(image_root is not None and image_root.is_dir(), "Test DICOM root missing")
    out = fresh_dir(args.out)
    ns = {"__name__": "exp152_local_public", "__file__": str(out / "public-serving.py")}
    exec(compile((stage_dir / "assets/public-serving.py").read_text(encoding="utf-8"), "public-serving.py", "exec"), ns)
    bind(ns, manifest, {OWN_NAME: stage_dir / "assets" / OWN_NAME, PUBLIC_NAME: Path(args.public_checkpoint)}, "blend", out)
    start = time.time()
    own = checked_model(ns, stage_dir / "assets" / OWN_NAME, ARCH, 384, args.device)[0]
    pub = checked_model(ns, args.public_checkpoint, ARCH, 384, args.device)[0]
    train = load_module("exp152_local_training", ROOT / "training/src/exp143_own0949.py")
    reference = train.build_model(320)
    reference.load_state_dict(safe_load(run / "model_best.pt")["model"], strict=True)
    reference.eval().to(args.device)
    from raptor_dicom import DirectoryDicomSource, prepare_study
    source = DirectoryDicomSource(image_root, header_only=True)
    reader = ns["_load_fastread"]().make_builder(ns["FASTREAD_VARIANT"], 384, ns["SLOTS"],
                                               hdr_threads=ns["FR_HDR"], px_threads=ns["FR_PX"])
    a, b, ref_a, ref_b, inputs, reference_inputs = [], [], [], [], {}, {}
    try:
        for uid in ids:
            vol, mask = reader(uid, records, str(image_root))
            local_vol, local_mask, _ = prepare_study(uid, records.get(uid, []), source, train.SPEC, "test")
            need(array_sha(vol) == array_sha(local_vol) and array_sha(mask) == array_sha(local_mask),
                 "Local/public decode mismatch; no parity claim")
            plain = ns["gpu_windows"](vol, mask, 94, 320, args.device)
            mirror = ns["gpu_windows"](vol, mask, 94, 320, args.device, mirror=True)
            inputs[uid] = input_record(ns, vol, mask, plain, mirror)
            local_plain, local_mirror = train.windows(local_vol, local_mask), train.windows(local_vol, local_mask, mirror=True)
            need(array_sha(plain) == array_sha(local_plain) and array_sha(mirror) == array_sha(local_mirror),
                 "Local/public normalized windows mismatch")
            reference_inputs[uid] = input_record(ns, local_vol, local_mask, local_plain, local_mirror)
            a.append(ns["infer_pair_probs"](own, plain, mirror, args.device))
            b.append(ns["infer_pair_probs"](pub, plain, mirror, args.device))
            ref_a.append(train.infer_volume(reference, local_vol, local_mask, args.device, "fp16"))
            # Independent CPU-built windows exercise public eval_windows vs GPU LUT.
            ref_b.append(ns["infer_pair_probs"](pub, ns["eval_windows"](vol, mask, 94, 320),
                          ns["eval_windows"](vol, mask, 94, 320, mirror=True), args.device))
    finally:
        source.close()
    a, b, ref_a, ref_b = map(np.asarray, (a, b, ref_a, ref_b))
    raw_exact = array_sha(a) == array_sha(ref_a) and array_sha(b) == array_sha(ref_b)
    comparisons = {}
    for variant in KERNELS:
        dest, ref = fresh_dir(out / variant), fresh_dir(out / (variant + "-predictor"))
        arms = [a] if variant == "own" else [a, b]
        refs = [ref_a] if variant == "own" else [ref_a, ref_b]
        csv_write(dest / "submission.csv", ids, output_values(arms, variant, ns["rankpct"]))
        csv_write(ref / "submission.csv", ids, output_values(refs, variant, ns["rankpct"]))
        save_raw(dest, ids, arms, inputs, manifest, variant)
        save_raw(ref, ids, refs, reference_inputs, manifest, variant)
        finish_receipt(dest, time.time()-start)
        finish_receipt(ref, time.time()-start)
        comparisons[variant] = dict(csv_exact=sha(dest / "submission.csv") == sha(ref / "submission.csv"),
                                    raw_exact=raw_exact, csv_sha256=sha(dest / "submission.csv"))
    passed = raw_exact and all(x["csv_exact"] for x in comparisons.values())
    result = dict(status="PASS_LOCAL_PUBLIC3" if passed else "FAIL_LOCAL_PUBLIC3", studies=3,
                  plan_sha256=sha(Path(args.plan) / "parity-plan.json"), manifest_sha256=report["manifest_sha256"],
                  comparisons=comparisons, max_abs_own=float(np.max(np.abs(a-ref_a))),
                  max_abs_public=float(np.max(np.abs(b-ref_b))))
    write_json(out / "public3-check.json", result)
    need(passed, "Local byte parity failed; inspect public3-check.json; do not push")
    return result


def load_run(path):
    import numpy as np
    path = Path(path)
    receipt = read_json(path / "receipt.json")
    need(receipt.get("status") == "COMPLETE_INFERENCE", "Incomplete inference receipt")
    for name in ("raw.npz", "inputs.json", "runtime.json", "binding.json", "submission.csv"):
        need(receipt.get("files", {}).get(name) == sha(path / name), "Output hash mismatch: " + name)
    with np.load(path / "raw.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    return arrays, read_json(path / "inputs.json"), read_json(path / "binding.json"), read_json(path / "runtime.json")


def compare_parity(args):
    import numpy as np
    _, report, manifest = checked_stage(args.stage)
    plan = verify_plan(args.plan, report)
    local_root = Path(args.local)
    local_check = read_json(local_root / "public3-check.json")
    need(local_check.get("status") == "PASS_LOCAL_PUBLIC3" and local_check["manifest_sha256"] == report["manifest_sha256"]
         and local_check["plan_sha256"] == sha(Path(args.plan) / "parity-plan.json"), "Local Public3/plan not PASS")
    results = {}
    for variant in KERNELS:
        lp, rp = local_root / variant, Path(getattr(args, variant))
        a, ai, ab, ar = load_run(lp)
        b, bi, bb, br = load_run(rp)
        need(ab == bb and ab["manifest_sha256"] == jhash(manifest) and ab["variant"] == variant,
             "Parity checkpoint/source/variant bindings mismatch")
        need(set(a) == set(b) == ({"ids", "targets", "own"} | ({"public"} if variant == "blend" else set())), "Raw array key mismatch")
        need(np.array_equal(a["ids"], b["ids"]) and a["ids"].tolist() == plan["ids"] and len(a["ids"]) == 3 and len(set(a["ids"].tolist())) == 3
             and np.array_equal(a["targets"], b["targets"]) and a["targets"].tolist() == TARGETS, "Raw cohort/target mismatch")
        input_pass = ai == bi and all(set(v) == {"volume", "mask", "centers", "plain", "mirror"} for v in ai.values())
        diffs = {}
        for name in ("own", "public") if variant == "blend" else ("own",):
            need(a[name].shape == b[name].shape == (3, 12) and np.isfinite(a[name]).all() and np.isfinite(b[name]).all(), "Raw output invalid")
            delta = np.abs(a[name].astype(float)-b[name].astype(float))
            diffs[name] = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()), p99_abs=float(np.quantile(delta, .99)),
                               passed=bool(delta.max() <= TOL["max_abs"] and delta.mean() <= TOL["mean_abs"]))
        final_pass = sha(lp / "submission.csv") == sha(rp / "submission.csv")
        reference_gpu = len(br["devices"]) == 2 and all("T4" in d["name"] for d in br["devices"])
        # Library differences are retained prominently; GPU/device differences are expected.
        runtime_diff = {k: {"local": ar[k], "reference": br[k]} for k in ("python", "packages", "cuda", "cudnn", "actual") if ar[k] != br[k]}
        results[variant] = dict(passed=input_pass and final_pass and reference_gpu and all(v["passed"] for v in diffs.values()),
                                input_hashes_exact=input_pass, raw=diffs, final_csv_exact=final_pass,
                                reference_2xT4=reference_gpu, runtime_differences=runtime_diff,
                                reference_receipt_sha256=sha(rp / "receipt.json"))
    passed = all(r["passed"] for r in results.values())
    result = dict(status="PASS_FINAL_PUBLIC3" if passed else "HOLD_PARITY", variants=results,
                  plan_sha256=sha(Path(args.plan) / "parity-plan.json"), tolerance=TOL,
                  limitation="Three visible studies only; differences in software are recorded, not attributed solely to GPUs; no hidden-data guarantee")
    write_json(args.out, result)
    need(passed, "Parity gate failed; no submission. Inspect report without relaxing thresholds")
    return result


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def push_plan(args):
    path, report, _ = checked_stage(args.stage)
    out = Path(args.out)
    venv = ROOT / ".venv/Scripts/python.exe"
    # Dataset create is private by default. Explicit cwd avoids Kaggle's upload-cache bug.
    commands = ["# PLAN ONLY. B checks local PASS, live quota, reserved lane and private settings first.",
                "Push-Location -LiteralPath " + ps_quote(path / "dataset"), "try {",
                "  & " + ps_quote(venv) + " -m kaggle datasets create -p .",
                "  if ($LASTEXITCODE -ne 0) { throw 'Dataset create failed; inspect CLI output' }",
                "} finally { Pop-Location }",
                "# Verify private dataset creation succeeded; CLI exit zero alone is insufficient."]
    for variant in KERNELS:
        commands += ["# Create TITLE equals slug. Notebook title + DESCRIPTION.md carry the OAI warning for blend.",
                     "# Use direct CLI: tools/kaggle_push.py rewrites frozen metadata. Check each attached dataset is ready first.",
                     "& " + ps_quote(venv) + " -m kaggle kernels push -p " + ps_quote(path / variant) + " --accelerator NvidiaTeslaT4",
                     "# Wait for COMPLETE and download outputs before any submission; preserve logs / receipt / raw.npz.",
                     "& " + ps_quote(venv) + " -m kaggle kernels output taeyangg4/" + KERNELS[variant] + " -p " + ps_quote(path / (variant + "-t4-output"))]
    commands += ["# Run compare-parity against both downloaded output folders. Do not submit on this plan's authority.",
                 "# Every submission needs Taeyang's chat instruction; blend additionally has OAI-derived lineage.",
                 "# After approval, replace <VERSION> with the verified COMPLETE version (never infer it):"]
    for variant in KERNELS:
        commands += ["# & " + ps_quote(venv) + " tools/kaggle_submit_once.py --id H-B-63-exp152-" + variant +
                     " --kernel taeyangg4/" + KERNELS[variant] + " --version <VERSION> --message " + ps_quote("EXP152 " + variant + " fixed A2; user approved")]
    write_text(out, "\n".join(commands) + "\n")
    return {"status": "PLAN_ONLY_NOT_EXECUTED", "out": str(out), "sha256": sha(out), "dataset": report["dataset"]}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", action="version", version=VERSION)
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("average")
    a.add_argument("--protocol", default=str(PROTOCOL)); a.add_argument("--checkpoint")
    a.add_argument("--out", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--average", required=True); s.add_argument("--out", required=True)
    s.add_argument("--dataset", default=ASSET_DATASET)
    p = sub.add_parser("parity-plan")
    p.add_argument("--stage", required=True); p.add_argument("--out", required=True)
    p.add_argument("--test-csv", default=str(ROOT / "storage/competition/test.csv"))
    p.add_argument("--series-csv", default=str(ROOT / "storage/competition/test_series.csv"))
    p = sub.add_parser("public3")
    p.add_argument("--stage", required=True); p.add_argument("--plan", required=True)
    p.add_argument("--protocol", default=str(PROTOCOL)); p.add_argument("--public-checkpoint", required=True)
    p.add_argument("--competition", default=str(ROOT / "storage/competition")); p.add_argument("--dicom-root")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--out", required=True)
    p = sub.add_parser("compare-parity")
    for name in ("stage", "plan", "local", "own", "blend", "out"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("push-plan")
    p.add_argument("--stage", required=True); p.add_argument("--out", required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = globals()[args.command.replace("-", "_")](args)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(e)}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
