"""H-B-149 v1.2.5: hash-bound scheduling amendment, not a new prediction recipe.

Changelog 2026-10-14: immutable 551-journal carry-over; singleton GPU residency,
workers=1; real full-shape parity/memory admission and measured-rate watchdog.
The v1.2.0 scientific identity is retained. Original full-v122 is read-only.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hb149_r7_flip_v120 as h
from tools import hb149_r7_pipeline_v120 as pipe
from tools import hb149_monitor_v125 as monitor
from tools import hb145_output_io as io

VERSION = "1.2.5"
BASE = h.SCRATCH
SOURCE = BASE / "full-v122"
OUT = BASE / "full-v125"
SMOKE = BASE / "cuda-smoke-v125"
CPU = BASE / "cpu-smoke-v125"
REG = BASE / "resume-registration-v125.json"
REVIEW = BASE / "review-v125.json"
PIN = "512c7d5d696a27f137ee29af9c5aa83b761dde0bd8141d23476add60c102ab0d"
CODE_FILES = ["tools/hb149_resume_v125.py", "tools/hb149_monitor_v125.py",
              "tools/hb149_r7_flip_v120.py", "tools/hb149_r7_pipeline_v120.py",
              "tools/hb145_output_io.py", "tools/exp169_r7_port.py",
              ".local/scratch/astra-b5/H-B-149/LAUNCH-v125.ps1"]
sha = h.r7.s6.sha256
read = io.read_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def code_pins():
    return {n: sha(ROOT / n) for n in CODE_FILES}


def original():
    ident = read(SOURCE / "identity.json")
    require(h.fingerprint(ident) == PIN, "original scientific identity drift")
    records = h.load_done(SOURCE, ident["uids"], PIN)
    require(len(records) == 551 and len(list((SOURCE / "journal").glob("*.json"))) == 551,
            "expected exactly 551 committed journals")
    require(not list(SOURCE.rglob("*.tmp")), "uncommitted source temporary files")
    require(all(r["status"] != "failure" and r.get("decode_failures") == 0 for r in records.values()),
            "invalid original terminal records")
    pins = {h.record_path(SOURCE, u).name: sha(h.record_path(SOURCE, u)) for u in records}
    return ident, records, pins


def patch_runtime(code):
    def replace(old, new):
        nonlocal code
        require(code.count(old) == 1, "runtime patch anchor changed")
        code = code.replace(old, new)
    replace('ms.append(m.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last))',
            'ms.append(m.eval().requires_grad_(False).to("cpu").to(memory_format=torch.channels_last))')
    replace('    hb145_checked = set()', '    hb145_checked = set()\n    hb149_active = None')
    replace('        nonlocal models', '        nonlocal models, hb149_active')
    replace('        uid = item["uid"]\n        for k in PIPE.required_folds(uid):',
            '        uid = item["uid"]\n        HB.output_io.receipt(Path(HB.hook_config()["out"]) / "phase-main.json", {"phase": "inference", "monotonic": time.perf_counter()})\n        for k in PIPE.required_folds(uid):')
    replace('            models = [hb145_models[k]]', '''            if hb149_active != k:
                if hb149_active is not None:
                    hb145_models[hb149_active].to("cpu")
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                hb145_models[k].to(dev)
                hb149_active = k
            models = [hb145_models[k]]''')
    replace('    fh = open(out / "predictions.csv", "w", encoding="utf-8", newline="")',
            '    fh = HB.output_io.DerivedStream(out / "predictions.csv", encoding="utf-8", newline="")')
    for var, expr in [('ffh', 'a.fluid_dump or (out / "fluid_stacker.csv")'),
                      ('sfh', 'a.state_dump or (out / "state_stacker.csv")')]:
        replace(f'        {var} = open({expr}, "w", encoding="utf-8", newline="")',
                f'        {var} = HB.output_io.DerivedStream({expr}, encoding="utf-8", newline="")')
    replace('    (out / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")',
            '    HB.output_io.receipt(out / "receipt.json", receipt)')
    replace('        PIPE.commit(item, fallback.get(uid))',
            '        HB.output_io.receipt(Path(HB.hook_config()["out"]) / "phase-main.json", {"phase": "checkpoint", "monotonic": time.perf_counter()})\n        PIPE.commit(item, fallback.get(uid))')
    replace('PIPE.install_dataset(CStudies)', 'PIPE.install_dataset(CStudies)\nfrom tools.hb149_resume_v125 import install_phase_dataset\ninstall_phase_dataset(CStudies)')
    compile(code, "compact_infer-v125", "exec")
    return code


def install_phase_dataset(cls):
    original_get = cls.__getitem__
    def get(self, index):
        import time
        path = Path(h.hook_config()["out"]) / "phase-loader.json"
        io.receipt(path, {"phase": "decode_render_anatomy", "monotonic": time.perf_counter()})
        item = original_get(self, index)
        io.receipt(path, {"phase": "render_ready", "monotonic": time.perf_counter()})
        return item
    cls.__getitem__ = get


def registration():
    ident, records, journals = original()
    hook = read(SOURCE / "hook.json")
    require(hook.get("identity") == PIN and all(hook.get(k) == ident[k]
            for k in ("uids", "source", "flags", "models", "device")),
            "source hook/scientific identity drift")
    invocation = read(SOURCE / "invocation_dual.json")
    def val(flag):
        return Path(invocation[invocation.index(flag) + 1])
    runtime = SOURCE / "stage/r7_runtime"
    require(sha(runtime / "compact_infer.py") == ident["runtime_sha256"], "frozen runtime drift")
    require(h.r7.verify_bundle(val("--models"), h.MANIFEST_SHA)[1] == ident["bundle"], "bundle drift")
    require(h.r7.code_record(val("--code"), val("--anchor-weights")) == ident["code"], "code/anatomy drift")
    require(h.verify_deps(Path(ident["deps"]["path"]), val("--code")) == ident["deps"], "dependency drift")
    archive, series, _, _ = h.default_paths()
    require(str(archive.resolve()) == ident["source"], "archive path drift")
    st = archive.stat()
    require([st.st_size, st.st_mtime_ns] == ident["source_stat"], "archive stat drift")
    require(sha(series) == ident["series_sha256"], "series metadata drift")
    require(sha(h.pairing.SELECTION) == ident["selection_sha256"], "selection drift")
    # Full shapes, highest DICOM counts in each fold, plus next two largest.
    ranked = sorted(records, key=lambda u: (-records[u]["dicom_files"], u))
    sample = [next(u for u in ranked if h.fold_k(u) == k) for k in (1, 2, 3)]
    sample += [u for u in ranked if u not in sample][:2]
    tree = {p.relative_to(runtime).as_posix(): sha(p) for p in runtime.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    return {"version": VERSION, **h.EXPOSURE, "scientific_identity": PIN,
            "journals": journals, "source_runtime": tree, "code": code_pins(),
            "metadata": {n: sha(SOURCE / "stage/data" / n) for n in ("test.csv", "test_series.csv")},
            "invocation_sha256": sha(SOURCE / "invocation_dual.json"),
            "executed_runtime_sha256": hashlib.sha256(patch_runtime(
                (runtime / "compact_infer.py").read_text(encoding="utf-8")).encode()).hexdigest(),
            "hook_sha256": sha(SOURCE / "hook.json"), "smoke_uids": sample,
            "smoke_selection": "largest DICOM count per fold plus two largest remaining; full shape",
            "execution": {"workers": 1, "chunk": 32, "amp": "fp16", "auxiliary_device": "cuda",
                          "residency": "inactive folds CPU, active singleton CUDA"},
            "main_reserved_cap_bytes": 10 * 10**9, "total_gpu_cap_bytes": 11 * 10**9,
            "parity": "exact raw and post float equality on all five smoke studies",
            "healthy_rate": "four completion intervals after first smoke commit",
            "watchdog": {"fraction": .25, "seconds": 1800}}


def validate():
    reg = read(REG)
    require(reg == registration(), "registration/input/code drift; re-register and repeat smoke/review")
    return reg


def carry(out, reg, ident):
    out.mkdir(parents=True, exist_ok=True)
    amendment = out / "execution-amendment-v125.json"
    extras = {p.name for p in (out / "journal").glob("*.json")} - reg["journals"].keys()
    if extras:
        require(amendment.exists() and read(amendment).get("registration_sha256") == sha(REG),
                "additional journals require their original execution registration")
    identity = out / "identity.json"
    if identity.exists():
        require(read(identity) == ident, "destination identity mismatch")
    else:
        shutil.copy2(SOURCE / "identity.json", identity)
    (out / "journal").mkdir(exist_ok=True)
    for name, digest in reg["journals"].items():
        target = out / "journal" / name
        if not target.exists():
            shutil.copy2(SOURCE / "journal" / name, target)
        require(sha(target) == digest, "carried journal changed")
    h.load_done(out, ident["uids"], PIN)
    io.atomic_json(amendment, {
        "version": VERSION, "registration_sha256": sha(REG), "scientific_identity": PIN,
        "executed_runtime_sha256": reg["executed_runtime_sha256"],
        "carried": len(reg["journals"]), "journal_hashes": reg["journals"],
        "reason": "Prediction definition unchanged; scheduling/residency only. CUDA parity required.",
        "execution": reg["execution"], **h.EXPOSURE})


def setup(out, reg, mode):
    ident = read(SOURCE / "identity.json")
    if mode == "run":
        carry(out, reg, ident)
        uids = ident["uids"]
    else:
        uids = reg["smoke_uids"][:2] if mode == "preprocess" else reg["smoke_uids"]
        # Engineering artifacts cannot pass registered scientific readout.
        io.atomic_json(out / "identity.json", {**ident, "cpu_smoke": True, "engineering_only": True, "uids": uids})
        require(not list((out / "journal").glob("*.json")), "smoke has old journals; use reset-smoke first")
    stage = out / "stage"
    h.owned_remove(stage, out)
    stage.mkdir(parents=True)
    runtime = stage / "r7_runtime"
    shutil.copytree(SOURCE / "stage/r7_runtime", runtime, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    p = runtime / "compact_infer.py"
    p.write_text(patch_runtime(p.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    require(sha(p) == reg["executed_runtime_sha256"], "executed runtime hash mismatch")
    pins_path = runtime / "runtime_sha256.json"
    pins = read(pins_path)
    pins["compact_infer.py"] = sha(p)
    io.atomic_json(pins_path, pins)
    io.atomic_json(out / "execution-runtime-v125.json", {
        "version": VERSION, "registration_sha256": sha(REG), "runtime_sha256": sha(p),
        "runtime_manifest_sha256": sha(pins_path), "engineering_only": mode != "run"})
    data = stage / "data"
    data.mkdir()
    for name in ("test.csv", "test_series.csv"):
        shutil.copy2(SOURCE / "stage/data" / name, data / name)
    cfg = read(SOURCE / "hook.json")
    cfg.update(out=str(out.resolve()), uids=uids)
    io.atomic_json(out / "hook.json", cfg)
    inv = read(SOURCE / "invocation_dual.json")
    def val(flag):
        return Path(inv[inv.index(flag) + 1])
    a = SimpleNamespace(command=mode, out=out, workers=1, chunk=32,
        device="cpu" if mode == "preprocess" else "cuda", models="heldout_flip",
        code=val("--code"), bundle=val("--models"), anatomy=val("--anchor-weights"))
    env = h.worker_env(a.code, Path(ident["deps"]["path"]), runtime)
    env["HB145_CONFIG"] = str((out / "hook.json").resolve())
    if mode == "preprocess":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    manifest = dict(read(a.bundle / "manifest.json"), flip_tta=True)
    return a, manifest, runtime, data, env, uids


def smoke_result(reg):
    execution = read(SMOKE / "execution-runtime-v125.json")
    runtime = SMOKE / "stage/r7_runtime"
    manifest = runtime / "runtime_sha256.json"
    require(execution == {
        "version": VERSION, "registration_sha256": sha(REG),
        "runtime_sha256": reg["executed_runtime_sha256"],
        "runtime_manifest_sha256": sha(manifest), "engineering_only": True},
        "smoke execution registration/runtime binding drift")
    require(read(manifest).get("compact_infer.py") == reg["executed_runtime_sha256"],
            "smoke runtime manifest drift")
    receipt = io.read_receipt(SMOKE / "run_receipt.json")
    resource = io.read_receipt(SMOKE / "run_resources.json")
    records = h.load_done(SMOKE, reg["smoke_uids"], PIN)
    originals = h.load_done(SOURCE, reg["smoke_uids"], PIN)
    exact = len(records) == 5 and all(records[u]["predictions"] == originals[u]["predictions"] for u in records)
    seconds = receipt.get("steady_seconds_per_study")
    main, total = receipt.get("peak_main_reserved_bytes"), resource.get("peak_total_gpu_mib")
    passed = bool(exact and receipt.get("exit_code") == 0 and receipt.get("failures") == 0
        and receipt.get("codec_coverage_complete") and seconds and seconds > 0
        and main and 0 < main <= reg["main_reserved_cap_bytes"]
        and total and 0 < total * 2**20 <= reg["total_gpu_cap_bytes"]
        and resource.get("baseline_gpu_mib") and not resource.get("telemetry_errors")
        and resource.get("cleanup_complete") and resource.get("state") == "COMPLETED"
        and sha(SMOKE / "stage/r7_runtime/compact_infer.py") == reg["executed_runtime_sha256"])
    return {"version": VERSION, **h.EXPOSURE, "engineering_only": True,
        "registration_sha256": sha(REG), "passed": passed, "studies": len(records),
        "exact_raw_post_parity": exact, "main_peak_reserved_bytes": main,
        "executed_runtime_sha256": reg["executed_runtime_sha256"],
        "device_peak_including_desktop_mib": total,
        "healthy_studies_per_minute": 60 / seconds if seconds and seconds > 0 else None,
        "evidence": {n: sha(SMOKE / n) for n in ("run_receipt.json", "run_resources.json", "execution-runtime-v125.json")}}


def admit(reg, review_path):
    proof = read(SMOKE / "admission-v125.json")
    require(proof == smoke_result(reg) and proof["passed"], "CUDA smoke/parity/memory admission failed")
    review = read(review_path)
    require(review.get("verdict") == "CLOSED" and review.get("independent") is True
            and bool(review.get("reviewer")) and review.get("builder") == "Codex/astra-b5"
            and review["reviewer"] != review["builder"]
            and review.get("registration_sha256") == sha(REG)
            and review.get("smoke_sha256") == sha(SMOKE / "admission-v125.json"),
            "independent review must bind current registration and CUDA smoke")
    return proof


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["register", "validate", "prepare", "cpu-smoke", "smoke", "reset-smoke", "admit", "run"])
    ap.add_argument("--review", type=Path, default=REVIEW)
    a = ap.parse_args(argv)
    if a.command == "register":
        reg = registration()
        io.atomic_json(REG, reg)
        print(json.dumps({"registered": True, "carried": 551, "smoke_studies": 5}))
        return 0
    reg = validate()
    if a.command == "validate":
        print('{"validated":true,"carried":551,"pending":449}')
        return 0
    if a.command == "reset-smoke":
        for dest in (SMOKE, CPU):
            h.owned_remove(dest, BASE)
        return 0
    if a.command == "admit":
        proof = admit(reg, a.review)
        print(json.dumps({"admitted": True, "healthy_studies_per_minute": proof["healthy_studies_per_minute"]}))
        return 0
    if a.command == "prepare":
        OUT.mkdir(parents=True, exist_ok=True)
        with h.exclusive_run(OUT):
            carry(OUT, reg, read(SOURCE / "identity.json"))
        print('{"prepared":true,"carried_byte_identical":551,"pending":449}')
        return 0
    mode = "preprocess" if a.command == "cpu-smoke" else "run"
    dest = CPU if a.command == "cpu-smoke" else SMOKE if a.command == "smoke" else OUT
    proof = admit(reg, a.review) if a.command == "run" else None
    dest.mkdir(parents=True, exist_ok=True)
    with h.exclusive_run(dest):
        args = setup(dest, reg, mode if proof or mode == "preprocess" else "smoke")
        args[0].command = mode
        pipe.monitored_run = functools.partial(monitor.monitored_run,
            poll_seconds=.2 if a.command == "smoke" else 1,
            healthy_rate=proof["healthy_studies_per_minute"] if proof else None)
        done = h.load_done(dest, args[-1], PIN)
        try:
            code = pipe.execute_pipeline(*args, done, PIN)
        finally:
            # Original source and carried prefix remain byte-identical even after abort.
            require(original()[2] == reg["journals"], "original journal drift")
            if proof:
                require(all(sha(dest / "journal" / n) == digest for n, digest in reg["journals"].items()),
                        "carried journal drift")
        if a.command == "smoke":
            result = smoke_result(reg)
            io.atomic_json(SMOKE / "admission-v125.json", result)
            print(json.dumps(result))
            return int(code != 0 or not result["passed"])
        return code


if __name__ == "__main__":
    raise SystemExit(main())
