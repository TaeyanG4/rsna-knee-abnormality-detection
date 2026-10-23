"""Reuse EXP155 v1.1 training bytecode with EXP168 guard and scratch paths.

No source file or global module is patched. Import and cloning are CPU passive;
only the explicit train/probe entry points can initialize CUDA.
"""
from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "training/src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tools import exp168_own0949_oai_jev as guard
import exp155_own0949_oai_v11 as prior

masked_study_bce = prior.masked_study_bce
MixedStudyDataset = prior.MixedStudyDataset
collate, unpack_batch = prior.collate, prior.unpack_batch
mixed_dataset, tiny_validation = prior.mixed_dataset, prior.tiny_validation


def scratch_directory(path=None):
    root = (ROOT / ".local/exp168-probes").resolve()
    out = Path(path or root / "probe-s42").resolve()
    if out == root or root not in out.parents:
        raise ValueError("probe output must be a new child of .local/exp168-probes")
    return out


def _import_guard(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tools" and "exp155_own0949_oai_v11" in fromlist:
        proxy = types.ModuleType("tools")
        proxy.exp155_own0949_oai_v11 = guard
        return proxy
    return builtins.__import__(name, globals, locals, fromlist, level)


def _reuse(function):
    env = dict(function.__globals__, scratch_directory=scratch_directory)
    env["__builtins__"] = dict(vars(builtins), __import__=_import_guard)
    return types.FunctionType(function.__code__, env, function.__name__, function.__defaults__, function.__closure__)


train = _reuse(prior.train)
probe_mode = _reuse(prior.probe_mode)


def probe(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("EXP168 probe measures one local GPU")
    p = guard.load_registration(args.protocol, args.corpus, args.init_weights, args.oai_corpus)
    guard.require_validation(args.protocol)
    settings = guard.execution_settings(p, 1, args.precision, 42, False)
    if not 1 <= args.steps <= settings["updates_per_epoch"] - 3 or not 0 <= args.workers <= 12:
        raise ValueError("probe steps/workers invalid")
    out = scratch_directory(args.out)
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "tools.exp168_own0949_oai_jev", "_probe-mode",
        "--protocol", str(Path(args.protocol).resolve()), "--corpus", str(Path(args.corpus).resolve()),
        "--oai-corpus", str(Path(args.oai_corpus).resolve()),
        "--init-weights", str(Path(args.init_weights).resolve()),
        "--steps", str(args.steps), "--workers", str(args.workers),
        "--precision", args.precision, "--out", str(out), "--mode", "off"]
    with (out / "off.log").open("x", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0 or not (out / "off.json").is_file():
        raise RuntimeError(f"probe failed; inspect {out / 'off.log'}")
    result = guard.read_json(out / "off.json")
    if result.get("protocol_sha256") != guard.sha256(args.protocol):
        raise ValueError("probe child registration changed")
    report = dict(status="SCRATCH_PROBE_ONLY_NO_FIT", protocol_sha256=guard.sha256(args.protocol),
        results=[result], steps_per_mode=args.steps, warmup_updates_per_mode=2,
        grad_checkpointing=False, caution="Short sample; full fit retains 100-update budget gate")
    guard.write_new_json(out / "report.json", report)
    print(json.dumps(dict(report, scratch_directory=str(out)), indent=2), flush=True)
    if result.get("admissible") is not True or result.get("mode") != "off":
        raise RuntimeError("EXP168 probe inadmissible; fit must not start")
