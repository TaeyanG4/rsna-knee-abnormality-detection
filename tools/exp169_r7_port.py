"""EXP169 R7 port v1.0.0: pinned serving adapter for heliosli's public R7 S branch.

Source (data only, never executed): heliosli/0-957-latest-dataset-score-in-one-click
(Apache 2.0; notebook SHA256 cb2a0f57..., cell 4 SHA256 cb168cfe...). Cell 4's
SOURCES are vendored byte-for-byte in exp169_r7_src/ and pinned by the notebook's own
SOURCE_SHA. This adapter transcribes cell 4's manifest checks, bundle copy and
manifest-driven worker arguments, and reuses exp189_s6_port v1.2.0 for everything
the two S stages share: the code dataset (identical CODE_SHA, anatomy and wheel pins,
whole-tree pin), study staging, the process-group runner and the prediction reader.

Differences from the notebook's S stage (scheduling/diagnostics only; model, window,
TTA and stacker math untouched):
- no --deadline / --item-timeout (project rule: no self-imposed time limits);
- explicit --device argument, full fallback tracebacks, uncapped per-study timing and
  fallback-reason lists (the exp189 v1.2.0 diagnostic patch set, re-anchored to
  compact_infer v4c: three anchors moved, every patch must match exactly once);
- the bundle copy lives in a per-run temporary directory instead of /tmp/probe/bundle.

Changelog
- v1.0.0 (Claude opus builder for B, H-B-125, 2026-10-13): initial port.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

try:
    from tools import exp189_s6_port as s6
except ImportError:
    import exp189_s6_port as s6

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "exp169_r7_src"
UID, LABELS = s6.UID, s6.LABELS
BUNDLE_ID = "8c7b666e27dc0e8f"
DATASET_SLUG = "heliosli/rsna-knee-compact-r7cnxfsss"
CODE_SLUG = "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"
ARM = "E_lrprior"
MODELS = [f"{ARM}_s0_f{i}.pt" for i in range(4)]
# Files the R7 dataset lists (B's listing 2026-10-13). Every one must be mounted.
ROSTER = ("manifest.json", *MODELS, "state_spec.json", "side_cls.pt", "fluid_stacker.json", "state_stacker.json")
# Manifest digest is unknown before the first visible run (weights are not local).
# None = trust the bundle_id + the manifest's own SHA256 map, as the notebook does,
# and record the observed digest; a set value is enforced fail-closed.
MANIFEST_SHA256: str | None = None
SOURCE_HASHES = {
    "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
    "compact_train.py": "04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba",
    "compact_infer.py": "8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d",
    "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
    "probe_dual.py": "a6155eb08701f9bb746f838bc8ef8e2129c85f76a5c58e379692ffe2e76bbef0",
    "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309",
    "side_common.py": "bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566",
}
NOTEBOOK = {
    "slug": "heliosli/0-957-latest-dataset-score-in-one-click", "id_no": 138222340, "licence": "Apache 2.0",
    "sha256": "cb2a0f57962ca9b6668661e34461aad772ed9ed10d576d32772e7c2898f2e315",
    "metadata_sha256": "960a19cc13d04830f1ca9409f38f13cf1bcf830e658e8a36af370dc38e5a19b6",
    "s_cell_index": 4, "s_cell_sha256": "cb168cfe82f5a04114894f6c07bbd6daf0373ad2eb0dfcee34df3ecf625986af",
    "s_run_cell_sha256": "68f70dd31cc3544cb5734313d06538355a34b122ff430a32820756175c9c9dd9",
    "r_cell_sha256": "3858126b82681ba469934c84a79f360eb199df59c16159b7a168df94e415cb37",
    "r_stage_cell_sha256": "d08db7bd194a78dac8c923a50048d098dd297223cc971e0fec40ba996d840e78",
    "blend_cell_sha256": "dee2b7a4c8baa21549c2509af29fa7226a6c9d0cbadced62e1a66ebdf92e13bd",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def verify_source(source_dir: str | Path = SOURCE) -> None:
    for name, digest in SOURCE_HASHES.items():
        s6._check_hash(Path(source_dir) / name, digest)


def declared_files(manifest: dict) -> list[str]:
    """Cell 4's copy list, in its order: models (+models2), state spec, side model, stackers."""
    names = list(manifest["models"]) + list(manifest.get("models2") or [])
    names += [manifest[key] for key in ("state_spec", "side_model", "fluid_stacker", "state_stacker") if manifest.get(key)]
    return names


def features(manifest: dict) -> dict:
    keys = ("state_spec", "side_model", "fluid_stacker", "state_stacker", "flip_tta", "compile", "trainer_md5",
            "models2", "views", "eval_stride", "slices", "phys_mm", "backbone", "size", "crop")
    return {key: manifest.get(key) for key in keys}


def verify_bundle(bundle_dir: str | Path, expected_manifest_sha256: str | None = None,
                  source_dir: str | Path = SOURCE) -> tuple[dict, dict]:
    """Fail closed before any copy, install or GPU use. Returns (manifest, identity)."""
    bundle = Path(bundle_dir)
    missing = [name for name in ROSTER if not (bundle / name).is_file()]
    require(not missing, "R7 bundle files missing: " + ", ".join(missing))
    observed = {name: s6.sha256(bundle / name) for name in ROSTER}
    pin = MANIFEST_SHA256 if expected_manifest_sha256 is None else expected_manifest_sha256
    require(pin is None or observed["manifest.json"] == pin, "R7 manifest SHA256 differs from pin")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "R7 manifest is not an object")
    require(manifest.get("bundle_id") == BUNDLE_ID, f"R7 bundle_id differs: {manifest.get('bundle_id')!r}")
    require(manifest.get("models") == MODELS, f"R7 model roster differs: {manifest.get('models')!r}")
    require(not manifest.get("models2"), "unexpected second R7 model group")
    for key in ("backbone", "size", "crop"):
        require(manifest.get(key) not in (None, ""), "R7 manifest lacks " + key)
    require(manifest["backbone"] != "ofl_l", "OFL-L bundles need an unattached weights dataset")
    digests = manifest.get("sha256")
    require(isinstance(digests, dict), "R7 manifest SHA256 map missing")
    for name in declared_files(manifest):
        require(isinstance(name, str) and Path(name).name == name and name not in (".", ".."), "unsafe R7 file name")
        require(name in digests, "R7 manifest has no SHA256 for " + name)
        digest = observed.get(name) or s6.sha256(bundle / name)
        require(digest == digests[name], "SHA256 mismatch: R7 " + name)
    if manifest.get("views") in ("v3k", "v4"):
        require(manifest.get("anchor_weights_sha256") == s6.ANATOMY_SHA256, "R7 anchor checkpoint differs from anatomy.pt pin")
    if manifest.get("trainer_md5"):
        trainer = (Path(source_dir) / "compact_train.py").read_bytes()
        require(hashlib.md5(trainer).hexdigest() == manifest["trainer_md5"], "pinned trainer differs from the R7 manifest (md5)")
        require(SOURCE_HASHES["compact_train.py"] == (manifest.get("src_sha256") or {}).get("compact_train.py"),
                "pinned trainer differs from the R7 manifest (sha256)")
    identity = {"dataset_slug": DATASET_SLUG, "bundle_id": BUNDLE_ID, "manifest_sha256": observed["manifest.json"],
                "manifest_pinned": pin is not None, "file_sha256": observed, "declared": declared_files(manifest),
                "features": features(manifest)}
    return manifest, identity


def prepare_bundle(bundle_dir: str | Path, destination: str | Path, manifest: dict) -> Path:
    """Cell 4: copy exactly the declared files, so the worker's glob sees only them."""
    source, target = Path(bundle_dir), Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    for name in declared_files(manifest):
        shutil.copyfile(source / name, target / name)
        require(s6.sha256(target / name) == manifest["sha256"][name], "SHA256 mismatch: copied R7 " + name)
    return target


def worker_args(manifest: dict, bundle: str | Path, anatomy_weights: str | Path) -> list[str]:
    """Transcription of cell 4's WORKER_ARGS (same keys, defaults, str() and order)."""
    bundle = Path(bundle)
    require(not manifest.get("models2") and manifest["backbone"] != "ofl_l", "unsupported R7 manifest branch")
    args = ["--backbone", manifest["backbone"], "--size", str(manifest["size"]), "--crop", str(manifest["crop"]),
            "--eval-stride", str(manifest.get("eval_stride", 3)), "--views", manifest.get("views", "v1")]
    if manifest.get("state_spec"):
        args += ["--state-spec", str(bundle / manifest["state_spec"])]
    if manifest.get("side_model"):
        args += ["--side-model", str(bundle / manifest["side_model"]),
                 "--side-threshold", str(manifest.get("side_threshold", 2.0)),
                 "--side-position-mm", str(manifest.get("side_position_mm", 20.0)),
                 "--side-max-x", str(manifest.get("side_max_x", 0.0))]
    args += ["--slices", str(manifest.get("slices", 3)), "--phys-mm", str(manifest.get("phys_mm", 0.0))]
    if manifest.get("trainer_md5"):
        args += ["--trainer-md5", manifest["trainer_md5"]]
    if manifest.get("compile"):
        args += ["--compile"]
    if manifest.get("fluid_stacker"):
        args += ["--fluid-stacker", str(bundle / manifest["fluid_stacker"])]
    if manifest.get("flip_tta"):
        args += ["--flip-tta"]
    if manifest.get("state_stacker"):
        args += ["--state-stacker", str(bundle / manifest["state_stacker"])]
    if manifest.get("views") in ("v3k", "v4"):
        args += ["--anchor-weights", str(anatomy_weights)]
    return args


def diagnostic_source(source_dir: str | Path, stage: Path) -> Path:
    """exp189 v1.2.0's diagnostic/no-deadline patch set on a runtime copy of the R7 sources."""
    runtime = Path(stage) / "r7_runtime"
    shutil.copytree(source_dir, runtime)
    worker = runtime / "compact_infer.py"
    code = worker.read_text(encoding="utf-8")
    patches = (
        ("import argparse, contextlib, csv, hashlib, inspect, io, json, math, os, shutil, sys, time",
         "import argparse, contextlib, csv, hashlib, inspect, io, json, math, os, shutil, sys, time, traceback",
         "traceback import"),
        ('"error": repr(exc)[:300], "images":',
         '"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "images":', "decode exception"),
        ('fallback[uid] = item["error"] or "no usable window"',
         'fallback[uid] = (item["error"] or "no usable window") + ("\\n" + item.get("traceback", "") if item.get("traceback") else "")',
         "decode fallback"),
        ('fallback[uid] = "gpu: " + repr(exc)[:200]',
         'fallback[uid] = "gpu: " + f"{type(exc).__name__}: {exc}\\n" + traceback.format_exc()', "GPU fallback"),
        ('fallback[uids[pos]] = "loader: " + repr(exc)[:200]',
         'fallback[uids[pos]] = "loader: " + f"{type(exc).__name__}: {exc}\\n" + traceback.format_exc()',
         "loader fallback"),
        ('item["seconds"][1], round(dt, 3)])',
         'item["seconds"][1], round(dt, 3), item["seconds"][2] if len(item["seconds"]) > 2 else 0.])',
         "per-study anchor timing"),
        ('"per_study": per_study[:50]', '"per_study": per_study', "uncapped study timing"),
        ('    ap.add_argument("--item-timeout", type=float, default=900.)\n', '', 'no loader time limit'),
        ('    ap.add_argument("--deadline", type=float, default=0.)',
         '    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")', 'explicit worker device'),
        ('    dev = torch.device("cuda")',
         '    dev = torch.device(a.device)\n    if dev.type == "cpu":\n        torch.set_num_threads(2)', 'worker device'),
        ('side_position_mm=20.0, extent_meta=False, fluid=False):',
         'side_position_mm=20.0, extent_meta=False, fluid=False, device="cuda"):', 'dataset device argument'),
        ('        self.native = None', '        self.device = device\n        self.native = None', 'dataset device'),
        ('            dev = "cuda" if torch.cuda.is_available() else "cpu"',
         '            dev = self.device', 'anatomy device'),
        ('            sdev = "cuda" if torch.cuda.is_available() else "cpu"',
         '            sdev = self.device', 'side device'),
        ('fluid=STK is not None or (SST is not None and SST["needs_volume"]))',
         'fluid=STK is not None or (SST is not None and SST["needs_volume"]), device=a.device)', 'forward dataset device'),
        ('    cap = torch.cuda.get_device_capability()[0]\n    amp = a.amp if a.amp != "auto" else ("bf16" if cap >= 8 else "fp16")',
         '    cap = torch.cuda.get_device_capability()[0] if dev.type == "cuda" else 0\n'
         '    amp = "fp32" if dev.type == "cpu" else (a.amp if a.amp != "auto" else ("bf16" if cap >= 8 else "fp16"))', 'CPU fp32'),
        ('    timed_out = False\n    while pos < len(uids) and not timed_out:',
         '    while pos < len(uids):', 'no worker deadline'),
        ('multiprocessing_context="spawn" if a.workers else None,\n                                             timeout=a.item_timeout if a.workers else 0)',
         'multiprocessing_context="spawn" if a.workers else None)', 'unbounded loader'),
        ('                if a.deadline and time.time() > a.deadline:\n                    timed_out = True\n                    break\n', '', 'no early termination'),
        ('    for u in uids[pos:]:\n        fallback[u] = "not reached before the deadline"\n', '', 'no deadline fallback'),
        ('"fallback": fallback, "timed_out": timed_out, "amp": amp,',
         '"fallback": fallback, "device": dev.type, "auxiliary_device": a.device, "cuda_initialized": torch.cuda.is_initialized(), "amp": amp,', 'device receipt'),
        ('    receipt = {"status": "PASS",',
         '    if dev.type == "cpu" and torch.cuda.is_initialized():\n        raise RuntimeError("CPU worker initialized CUDA")\n    receipt = {"status": "PASS",', 'CPU device postcondition'),
        ('("studies", "scored", "timed_out", "seconds", "amp")',
         '("studies", "scored", "device", "seconds", "amp")', 'device completion log'),
    )
    for old, new, name in patches:
        code = s6._replace_once(code, old, new, name)
    require(code.count("torch.cuda.get_device_name()") == 2, "R7 expected two device-name diagnostics")
    code = code.replace("torch.cuda.get_device_name()",
                        '(torch.cuda.get_device_name() if dev.type == "cuda" else "CPU")')
    for token in ("a.deadline", "a.item_timeout", "timed_out"):
        require(token not in code, "R7 worker still references " + token)
    compile(code, str(worker), "exec")
    worker.write_text(code, encoding="utf-8", newline="\n")
    controller = runtime / "probe_dual.py"
    code = controller.read_text(encoding="utf-8")
    code = s6._replace_once(code, "for u in missing[:40]", "for u in missing", "uncapped fallback reasons")
    code = s6._replace_once(code, '    ap.add_argument("--deadline", type=float, default=0.)',
                            '    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")', "controller device")
    begin = code.index("    try:\n        names = subprocess.check_output")
    end = code.index("\n\n    def cost", begin)
    code = code[:begin] + '''    if a.device == "cpu":
        names, visible = ["CPU"], ["-1"]
    else:
        try:
            names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                            text=True).strip().splitlines()
        except Exception:
            names = ["unknown"]
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visibility is not None:
            visible = [v.strip() for v in visibility.split(",") if v.strip() and v.strip() != "-1"]
            if not visible:
                raise SystemExit("CUDA is disabled; select --device cpu explicitly")
            names = [names[int(v)] if v.isdigit() and int(v) < len(names) else v for v in visible]
        else:
            visible = [str(i) for i in range(max(1, len(names)))]
    n_gpu = len(visible)''' + code[end:]
    code = s6._replace_once(code, "CUDA_VISIBLE_DEVICES=str(gi)", "CUDA_VISIBLE_DEVICES=visible[gi]", "respect CUDA visibility")
    code = s6._replace_once(code, '"--deadline", str(a.deadline)', '"--device", a.device', "forward device")
    code = s6._replace_once(code, 'receipts.append({"status": "NO_RECEIPT", "worker": gi})',
        'receipts.append({"status": "NO_RECEIPT", "worker": gi, "fallback": {u: '
        '(wdir / "log.txt").read_text(encoding="utf-8", errors="replace") for u in part}})', "startup fallback traceback")
    require("deadline" not in code, "R7 controller still references a deadline")
    compile(code, str(controller), "exec")
    controller.write_text(code, encoding="utf-8", newline="\n")
    (runtime / "runtime_sha256.json").write_text(json.dumps(
        {p.name: s6.sha256(p) for p in sorted(runtime.glob("*.py"))}, indent=2) + "\n", encoding="utf-8")
    return runtime


def code_record(code_dir: str | Path, anatomy_weights: str | Path, source_dir: str | Path = SOURCE,
                code_tree_sha256: str | None = None) -> dict:
    """exp189's whole-tree/module/anatomy pins plus the R7 sources' own static closure."""
    record = s6.verify_code(code_dir, anatomy_weights, tree_sha256=code_tree_sha256)
    closure = s6.executed_closure(code_dir, source_dir)
    missing = [rel for rel in s6.DIRECT_CODE_MODULES if rel not in closure]
    require(not missing, "R7 executed code closure is incomplete: " + ", ".join(missing))
    tree = s6.code_tree_manifest(code_dir)["sha256"]
    require(all(tree.get(rel) == digest for rel, digest in closure.items()), "R7 executed closure differs from the pinned tree")
    return dict(record, r7_closure_sha256=closure)


def write_provenance(path: str | Path, *, identity: dict, code: dict, args: list[str]) -> dict:
    record = {
        "dataset_slug": DATASET_SLUG, "bundle_id": BUNDLE_ID, "licence": "other",
        "manifest_sha256": identity["manifest_sha256"], "manifest_pinned": identity["manifest_pinned"],
        "file_sha256": identity["file_sha256"], "declared": identity["declared"], "features": identity["features"],
        "source_sha256": SOURCE_HASHES, "notebook": NOTEBOOK, "code_sha256": s6.CODE_HASHES,
        "anchor_weights_sha256": s6.ANATOMY_SHA256, "port_version": VERSION, "exp189_port_version": s6.VERSION,
        "code_tree_sha256": code["tree_sha256"], "code_tree_files": code["files"],
        "code_closure_sha256": code["closure_sha256"], "r7_code_closure_sha256": code["r7_closure_sha256"],
        "wheel_sha256": s6.WHEEL_HASHES, "wheel_versions": s6.WHEEL_VERSIONS, "worker_args": args,
    }
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def infer_studies(study_dirs: Sequence[str | Path], *, bundle_dir: str | Path, code_dir: str | Path,
                  anatomy_weights: str | Path, output_dir: str | Path, workers: int = 2, amp: str = "auto",
                  device: str = "cuda", max_fallback_fraction: float = 1.0, source_dir: str | Path = SOURCE,
                  code_tree_sha256: str | None = None,
                  expected_manifest_sha256: str | None = None) -> list[dict[str, str | float]]:
    """Notebook cells 4-5 on study directories; returns rows in request order.

    No GPU is touched until bundle, source, code-tree, module and anatomy pins pass.
    """
    require(device in {"cpu", "cuda"}, "device must be cpu or cuda")
    ids, paths = s6._study_ids(study_dirs)
    bundle_dir, code_dir, anatomy_weights = (Path(p).resolve() for p in (bundle_dir, code_dir, anatomy_weights))
    manifest, identity = verify_bundle(bundle_dir, expected_manifest_sha256, source_dir)
    verify_source(source_dir)
    code = code_record(code_dir, anatomy_weights, source_dir, code_tree_sha256)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"R7 output directory must be empty: {out}")
    (out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="exp169_r7_stage_", dir=out) as stage_name:
        stage = Path(stage_name)
        bundle = prepare_bundle(bundle_dir, stage / "bundle", manifest)
        args_w = worker_args(manifest, bundle, anatomy_weights)
        write_provenance(out / "provenance.json", identity=identity, code=code, args=args_w)
        data = stage / "data"
        data.mkdir()
        s6._stage_studies(paths, ids, data)
        runtime_source = diagnostic_source(source_dir, out)
        args = [sys.executable, "-u", str(runtime_source / "probe_dual.py"),
                "--data", str(data), "--split", "test", "--code", str(code_dir),
                "--bundle", str(bundle), "--arm", ARM, "--output", str(out / "controller"),
                "--amp", amp, "--loader-workers", str(workers), "--device", device,
                "--worker", "compact_infer.py", "--worker-args", json.dumps(args_w),
                "--max-fallback-fraction", str(max_fallback_fraction)]
        env = dict(os.environ, PYTHONPATH=str(runtime_source) + os.pathsep + str(code_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        (out / "invocation.json").write_text(json.dumps({"args": args, "device": device,
            "data_source": str(paths[0].parent.parent), "series_sha256": s6.sha256(data / "test_series.csv")},
            indent=2) + "\n", encoding="utf-8")
        print("R7_WORKER_ARGS", json.dumps(args_w), flush=True)
        s6._run_controller(args, env)
    receipt = json.loads((out / "controller" / "submission_receipt.json").read_text(encoding="utf-8"))
    if receipt["rows"] != len(ids) or receipt["fallback_studies"] > max_fallback_fraction * len(ids):
        raise RuntimeError("R7 controller output incomplete")
    rows = s6.read_predictions(out / "controller" / "submission.csv", ids)
    shutil.copyfile(out / "controller" / "submission.csv", out / "predictions.csv")
    try:
        print("R7_ANATOMY_ANCHORS", json.dumps(s6.anatomy_fallbacks(receipt)), flush=True)
    except Exception as exc:  # logging only; the public notebook never rejects on anchors
        try:
            print("R7_ANATOMY_ANCHORS unavailable:", repr(exc)[:200], flush=True)
        except Exception:
            pass
    return rows
