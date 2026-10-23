"""Stage/run a fixed two-T4 R2 study-sharding engineering benchmark.

Build is CPU-only. Run/child are explicit Kaggle-only actions. The successful
historical runtime100 worker and immutable legacy serving assets are reused
verbatim; no training, submission, header optimization or label metric exists.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[1]
ASSET_SHA = "e62893ebb1e22dbb1288ea6c805097bbc96ef5e43759f42a22d7cfb3bbef036b"
LEGACY_SHA = "e478626baaf757260de3f8b03f143e727ebb130f575404788e2d9dad1264867e"
REFERENCE_SHA = "7187999225a2e98ab4e61ba2222d6f4f1a2a4a672e8bb7e381b8bc61e4a7f1b6"
GDCM_SHA = "512d0d903e12c5ad389939a257817bf182ce288431b3d474d62cb0c0833b8b56"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for raw in iter(lambda: f.read(8 << 20), b""):
            h.update(raw)
    return h.hexdigest()


def dump(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def split_ids(ids):
    if len(ids) != 100 or len(set(ids)) != 100 or any(not isinstance(v, str) or not v for v in ids):
        raise ValueError("expected the fixed100 unique string IDs")
    return [ids[0::2], ids[1::2]]


def effective_cpu():
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    quota = None
    for path, period_path in [(Path("/sys/fs/cgroup/cpu.max"), None),
                               (Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))]:
        if path.is_file():
            parts = path.read_text().strip().split()
            if period_path and period_path.is_file():
                parts = [parts[0], period_path.read_text().strip()]
            if len(parts) == 2 and parts[0] != "max" and float(parts[0]) > 0:
                quota = float(parts[0]) / float(parts[1])
                break
    return {"affinity": affinity, "affinity_count": len(affinity), "cgroup_cpu_quota": quota,
            "effective_cpus": min(len(affinity), quota) if quota is not None else len(affinity)}


def stop_tree(process):
    """Terminate only this explicitly launched process group and its descendants."""
    if os.name == "posix":
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            if sig == signal.SIGTERM:
                time.sleep(.3)
    else:
        import psutil
        try:
            parent = psutil.Process(process.pid)
            family = parent.children(recursive=True) + [parent]
            for proc in family:
                try: proc.terminate()
                except psutil.Error: pass
            _, alive = psutil.wait_procs(family, timeout=.3)
            for proc in alive:
                try: proc.kill()
                except psutil.Error: pass
        except psutil.Error:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def supervise(commands, environments, work, timeout):
    import psutil
    started = time.perf_counter()
    processes, handles, launch = [], [], []
    peak, samples, state = 0, 0, "RUNNING"
    try:
        for index, (command, env) in enumerate(zip(commands, environments, strict=True)):
            out = (Path(work) / f"child-{index}.stdout.log").open("wb")
            err = (Path(work) / f"child-{index}.stderr.log").open("wb")
            handles.extend([out, err])
            proc = subprocess.Popen(command, env=env, stdout=out, stderr=err, start_new_session=os.name == "posix")
            processes.append(proc)
            launch.append({"index": index, "pid": proc.pid, "launched_seconds": time.perf_counter() - started,
                           "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES")})
        dump(Path(work) / "supervisor-progress.json", {"status": state, "launches": launch, "deadline_seconds": timeout})
        while True:
            elapsed = time.perf_counter() - started
            if elapsed >= timeout:
                raise TimeoutError(f"study-shard process-tree deadline exceeded: {timeout:.1f}s")
            codes = [p.poll() for p in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"study shard failed; child exit codes {codes}")
            parent = psutil.Process()
            total = parent.memory_info().rss
            for child in parent.children(recursive=True):
                try: total += child.memory_info().rss
                except psutil.Error: pass
            peak = max(peak, total); samples += 1
            if all(code == 0 for code in codes):
                state = "COMPLETE"
                return {"wall_seconds": elapsed, "launches": launch, "exit_codes": codes,
                        "rss_peak_parent_plus_children_bytes": peak, "rss_samples": samples,
                        "rss_interval_seconds": .2}
            time.sleep(.2)
    finally:
        # Also catch orphaned pip grandchildren when a child has already exited.
        for proc in processes:
            stop_tree(proc)
        for handle in handles:
            handle.close()
        dump(Path(work) / "supervisor-progress.json", {"status": state if state == "COMPLETE" else "FAILED_OR_INTERRUPTED",
            "launches": launch, "deadline_seconds": timeout, "wall_seconds": time.perf_counter() - started,
            "exit_codes": [p.poll() for p in processes], "rss_peak_parent_plus_children_bytes": peak,
            "rss_samples": samples, "all_children_reaped": all(p.poll() is not None for p in processes)})


def merge_arrays(expected_ids, parts):
    import numpy as np
    split_ids(expected_ids)
    locations = {}
    for part_index, (ids, values) in enumerate(parts):
        ids = list(map(str, ids))
        values = np.asarray(values)
        if values.shape != (5, len(ids), 12) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("each shard must have finite float32 [5,N,12] probabilities")
        if (values < 0).any() or (values > 1).any():
            raise ValueError("shard probabilities outside [0,1]")
        for index, uid in enumerate(ids):
            if uid in locations:
                raise ValueError("duplicate study across/within shards")
            locations[uid] = (part_index, index)
    if set(locations) != set(expected_ids):
        raise ValueError("missing or unexpected study in shard outputs")
    return np.stack([parts[locations[uid][0]][1][:, locations[uid][1], :] for uid in expected_ids], axis=1)


def probability_gate(actual, reference, maximum=.002, mean=.0005):
    import numpy as np
    if actual.shape != reference.shape or actual.shape != (5, 100, 12):
        raise ValueError("fixed five-member100 probability shape required")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite probability comparison")
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    members = [{"member": i, "max_abs": float(v.max()), "mean_abs": float(v.mean()),
                "pass": bool(v.max() <= maximum and v.mean() <= mean)} for i, v in enumerate(delta)]
    return {"maximum_atol": maximum, "mean_atol": mean, "rtol": 0., "members": members,
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "pass": all(v["pass"] for v in members),
            "scope": "new preregistered all5 x100 engineering comparison; same numerical limits as prior F0 test3, broader scope"}


def child(base, assets, competition, selection, work, codec):
    base, work = Path(base), Path(work)
    protocol = read(base / "protocol.json")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}:
        raise RuntimeError("explicit one-device child binding required")
    if any(os.environ.get(k) != "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise RuntimeError("two CPU threads per child required")
    if sha(base / "runtime_worker.py") != protocol["files_sha256"]["runtime_worker.py"]:
        raise ValueError("historical worker identity changed")
    sys.path.insert(0, str(Path(codec).resolve()))
    worker = load_module(base / "runtime_worker.py", "original_runtime_worker")
    result = worker.run(assets, competition, selection, work)
    import torch
    if torch.cuda.device_count() != 1 or torch.get_num_threads() != 2:
        raise RuntimeError("child GPU/CPU runtime isolation violated")
    isolation = {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "logical_gpu_count": torch.cuda.device_count(),
                 "torch_cpu_threads": torch.get_num_threads(), "gpu_name": torch.cuda.get_device_name(0),
                 "codec_gdcm_version": __import__("gdcm").Version.GetVersion(), "pid": os.getpid()}
    dump(work / "shard-isolation.json", isolation)
    if result["status"] != "COMPLETE" or not result["all_official_syntaxes_available"]:
        raise RuntimeError("child runtime or frozen codec capability gate failed")


def run(base, assets, competition, codec_wheel, work):
    started = time.perf_counter()
    base, assets, competition, work = map(Path, (base, assets, competition, work))
    if os.name != "posix" or not str(base.resolve()).startswith("/kaggle/working/"):
        raise RuntimeError("explicit Kaggle Linux run only; local build/tests must not start GPU jobs")
    work.mkdir(parents=True, exist_ok=False)
    protocol, selection = read(base / "protocol.json"), read(base / "selection.json")
    result = {"status": "RUNNING", "version": protocol["version"], "scope": protocol["scope"],
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "protocol_sha256": sha(base / "protocol.json"),
              "stages": {}, "failures": []}
    try:
        for name, expected in protocol["files_sha256"].items():
            if sha(base / name) != expected: raise ValueError(f"benchmark payload changed: {name}")
        if sha(assets / "d01-assets-manifest.json") != ASSET_SHA or sha(assets / "training/src/raptor_dicom.py") != LEGACY_SHA:
            raise ValueError("immutable model assets/legacy input reader changed")
        if sha(codec_wheel) != GDCM_SHA:
            raise ValueError("codec wheel identity changed")
        result["cpu"] = effective_cpu()
        if result["cpu"]["effective_cpus"] < 4:
            raise RuntimeError("two 2-thread children require at least four effective CPUs")
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=20).stdout.strip().splitlines()
        result["gpu_inventory"] = gpu
        if len(gpu) != 2 or any("T4" not in line for line in gpu):
            raise RuntimeError("exactly two actual T4 devices required")
        install = time.perf_counter()
        codec = work / "codec-environment"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(codec), str(codec_wheel)],
                       check=True, timeout=120)
        result["stages"]["codec_install_seconds"] = time.perf_counter() - install
        commands, environments = [], []
        shards = split_ids(selection["ids"])
        for index, ids in enumerate(shards):
            shard_selection = dict(selection, ids=ids)
            path = work / f"selection-{index}.json"
            dump(path, shard_selection)
            commands.append([sys.executable, str(base / "shard_benchmark.py"), "child", "--base", str(base),
                "--assets", str(assets), "--competition", str(competition), "--selection", str(path),
                "--work", str(work / f"child-{index}"), "--codec", str(codec)])
            environments.append(dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
                CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        dump(work / "shard-benchmark-result.json", result)
        result["parallel"] = supervise(commands, environments, work, protocol["supervisor_deadline_seconds"] - (time.perf_counter() - started))
        merge_start = time.perf_counter()
        import numpy as np
        prior = read(base / "reference-receipt.json")
        reference = np.load(base / "reference-members.npz", allow_pickle=False)
        ids = selection["ids"]
        if reference["study_uids"].tolist() != ids:
            raise ValueError("historical probability UID/order differs from fixed cohort")
        parts, children, current_receipts, tensors = [], [], {}, {}
        reference_receipts = {r["uid"]: r for r in prior["study_receipts"]}
        reference_tensors = {r["uid"]: r["sha256"] for r in prior["inference"]["input_tensor_hashes"]}
        input_checks = []
        for index, wanted in enumerate(shards):
            directory = work / f"child-{index}"
            data = np.load(directory / "engineering-members.npz", allow_pickle=False)
            child_report, isolation = read(directory / "benchmark-receipt.json"), read(directory / "shard-isolation.json")
            actual_ids = data["study_uids"].tolist()
            if actual_ids != wanted or child_report["completed_studies"] != 50 or child_report["failures"]:
                raise ValueError("shard completion/order contract failed")
            inf = child_report["inference"]
            if (inf["status"] != "COMPLETE" or inf["input_contract"] != prior["inference"]["input_contract"]
                or inf["runtime"] != prior["inference"]["runtime"] or inf["fp32_fallbacks"] or inf["neutral_fills"]
                or inf["member_count"] != 5 or inf["studies"] != 50 or not inf["all_finite"]):
                raise ValueError("shard input/runtime/member contract failed")
            member_ids = [(m["name"], m["checkpoint_sha256"]) for m in inf["member_results"]]
            if member_ids != [(m["name"], m["checkpoint_sha256"]) for m in prior["inference"]["member_results"]]:
                raise ValueError("member identity/order changed")
            if any(m["status"] != "COMPLETE" or m["completed_studies"] != 50 for m in inf["member_results"]):
                raise ValueError("one or more model members are incomplete")
            current_receipts.update({r["uid"]: r for r in child_report["study_receipts"]})
            tensors.update({r["uid"]: r["sha256"] for r in inf["input_tensor_hashes"]})
            parts.append((actual_ids, data["member_probabilities"]))
            children.append({"index": index, "isolation": isolation, "timings": child_report["timings"],
                "cuda_memory": child_report["cuda_memory"], "rss": child_report["rss"],
                "decode_time_distribution": child_report["decode_time_distribution"],
                "study_time_distribution": child_report["study_time_distribution"],
                "dicom_io": child_report["dicom_io"], "receipt_sha256": sha(directory / "benchmark-receipt.json")})
        raw = merge_arrays(ids, parts)
        if set(current_receipts) != set(ids) or set(tensors) != set(ids):
            raise ValueError("study/input receipt coverage differs from merged probabilities")
        for uid in ids:
            before, after = reference_receipts[uid], current_receipts[uid]
            input_checks.append({"uid": uid, "volume_exact": before["volume_sha256"] == after["volume_sha256"],
                "mask_exact": before["mask_sha256"] == after["mask_sha256"], "full_receipt_exact": before == after,
                "normalized_tensor_exact": reference_tensors[uid] == tensors[uid]})
        result["children"], result["input_checks"] = children, input_checks
        result["probability_gate"] = probability_gate(raw, reference["member_probabilities"],
            protocol["probability_max_atol"], protocol["probability_mean_atol"])
        result["merge"] = {"uids": ids, "shape": list(raw.shape), "dtype": str(raw.dtype), "finite": bool(np.isfinite(raw).all()),
                           "uid_order_exact": True, "disjoint_coverage_exact": True}
        np.savez_compressed(work / "engineering-sharded-members.npz", study_uids=np.asarray(ids), member_probabilities=raw)
        np.savez_compressed(work / "engineering-sharded-mean.npz", study_uids=np.asarray(ids), probabilities=raw.mean(axis=0, dtype=np.float32))
        result["stages"]["merge_validation_seconds"] = time.perf_counter() - merge_start
        if not all(all(v for k, v in row.items() if k != "uid") for row in input_checks) or not result["probability_gate"]["pass"]:
            raise RuntimeError("frozen input or probability gate failed; no retroactive tolerance change")
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAILED"
        result["failures"].append({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        raise
    finally:
        result["total_wall_seconds_including_child_model_load"] = time.perf_counter() - started
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        dump(work / "shard-benchmark-result.json", result)
    return result


def build(out, reference_members):
    import numpy as np
    out = Path(out)
    if out.exists(): raise FileExistsError("choose a new immutable staging directory")
    historic = ROOT / ".local/d01-runtime-staging/20261003-v4"
    nb = read(historic / "d01-runtime100.ipynb")
    tree = ast.parse("".join(nb["cells"][1]["source"]))
    worker = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "source" for t in n.targets))
    ast.parse(worker)
    selection = read(historic / "selection-manifest.json")
    ids = selection["ids"]; split_ids(ids)
    compact = {k: selection[k] for k in ("ids", "train_series_sha256", "assets_manifest_sha256", "excluded_corrupt_uids")}
    if sha(reference_members) != REFERENCE_SHA:
        raise ValueError("exact old T4 reference NPZ required")
    ref = np.load(reference_members, allow_pickle=False)
    if ref["study_uids"].tolist() != ids or ref["member_probabilities"].shape != (5, 100, 12) or not np.isfinite(ref["member_probabilities"]).all():
        raise ValueError("historical reference UID/shape/finite contract failed")
    old_receipt = historic / "output/d01-runtime-benchmark/results/benchmark-receipt.json"
    raw = {"shard_benchmark.py": Path(__file__).read_bytes(), "runtime_worker.py": worker.encode(),
           "selection.json": json.dumps(compact, separators=(",", ":")).encode(),
           "reference-members.npz": Path(reference_members).read_bytes(), "reference-receipt.json": old_receipt.read_bytes()}
    protocol = {"version": "d01-r2-two-t4-study-shard-v1", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "new unmeasured two-device scheduling policy; fixed100 engineering only, no accuracy/hidden budget claim",
        "shards": "fixed UID positions 0::2 and1::2, exactly50each", "cpu_threads_per_child": 2, "minimum_effective_cpus": 4,
        "gpu": "two actual T4, child CUDA_VISIBLE_DEVICES 0/1; each sees logical cuda:0",
        "input": "immutable legacy directory reader; five original models; resident batch1 FP16 strict; codec gdcm3.2.6",
        "input_gate": "prior volume/mask/full receipt/normalized tensor hashes exact; input107 v2 and codec gates are not replaced",
        "probability_max_atol": .002, "probability_mean_atol": .0005, "probability_rtol": 0,
        "probability_scope": "each of five members over100x12; new all5 scope uses unchanged prior F0 numeric tolerances",
        "supervisor_deadline_seconds": 1560, "kaggle_session_seconds": 1800,
        "files_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
        "model_assets_manifest_sha256": ASSET_SHA, "legacy_adapter_sha256": LEGACY_SHA, "codec_wheel_sha256": GDCM_SHA,
        "reference_kernel": "taeyangg4/knee-d01-r2-runtime100", "reference_kernel_version": 2,
        "historical_worker_preserved_verbatim": True,
        "failures": "child nonzero/deadline kills both process groups; mismatched inputs or predictions remain FAILED"}
    raw["protocol.json"] = json.dumps(protocol, ensure_ascii=False, indent=2).encode()
    payload = {k: {"sha256": hashlib.sha256(v).hexdigest(), "bytes": len(v), "data": base64.b64encode(zlib.compress(v, 9)).decode()} for k, v in raw.items()}
    code = f'''from pathlib import Path
import ast, base64, hashlib, importlib.util, json, os, sys, time, zlib
boot_started = time.perf_counter()
base = Path('/kaggle/working/d01-r2-shard100')
base.mkdir(exist_ok=False)
payload = {payload!r}
for name, entry in payload.items():
    raw = zlib.decompress(base64.b64decode(entry['data']))
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
    if name.endswith('.py'): ast.parse(raw.decode('utf-8'))
    (base/name).write_bytes(raw)
asset_roots = [Path('/kaggle/input/knee-d01-r2-assets'), Path('/kaggle/input/datasets/taeyangg4/knee-d01-r2-assets')]
assets = [p.parent for r in asset_roots if r.is_dir() for p in r.rglob('d01-assets-manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest() == {ASSET_SHA!r}]
assert len(assets) == 1
competition = next((p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p/'train_series.csv').is_file()),None)
assert competition is not None
codec_roots = [Path('/kaggle/input/knee-d01-codec-assets'),Path('/kaggle/input/datasets/taeyangg4/knee-d01-codec-assets')]
wheels = [p for r in codec_roots if r.is_dir() for p in r.rglob('python_gdcm-3.2.6-*.whl') if hashlib.sha256(p.read_bytes()).hexdigest() == {GDCM_SHA!r}]
assert len(wheels) == 1
spec=importlib.util.spec_from_file_location('shard_benchmark',base/'shard_benchmark.py')
module=importlib.util.module_from_spec(spec);sys.modules['shard_benchmark']=module;spec.loader.exec_module(module)
try:
    result=module.run(base,assets[0],competition,wheels[0],base/'results')
    print(json.dumps({{'status':result['status'],'wall_seconds':result['total_wall_seconds_including_child_model_load']}},indent=2))
finally:
    (base/'launcher-timing.json').write_text(json.dumps({{'notebook_cell_seconds':time.perf_counter()-boot_started,'scope':'includes staging; excludes Kaggle provisioning'}},indent=2))
'''
    ast.parse(code)
    notebook = {"nbformat": 4, "nbformat_minor": 4, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# D01 two-T4 study sharding\nFixed100 engineering benchmark. New scheduling policy; no submission or accuracy metric.\n"]},
                  {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)}]}
    metadata = read(historic / "kernel-metadata.json")
    metadata.update(id="taeyangg4/knee-d01-r2-shard100", title="knee-d01-r2-shard100", code_file="d01-r2-shard100.ipynb")
    metadata["dataset_sources"].append("taeyangg4/knee-d01-codec-assets")
    out.mkdir(parents=True)
    for name, data in [(metadata["code_file"], notebook), ("kernel-metadata.json", metadata)]:
        dump(out / name, data)
    (out / "protocol.json").write_bytes(raw["protocol.json"])
    review = {"status": "STAGED_ONLY_UNMEASURED", "notebook_bytes": (out / metadata["code_file"]).stat().st_size,
        "notebook_sha256": sha(out / metadata["code_file"]), "protocol_sha256": sha(out / "protocol.json"),
        "payloads": {k: {x: v[x] for x in ("sha256", "bytes")} for k, v in payload.items()},
        "budget": "max30minute2T4 session=at most1GPUhour; supervisor26minute total deadline; expected20-30min bounded preparation estimate",
        "command_not_executed": f'.venv/Scripts/kaggle.exe kernels push -p "{out.resolve()}" --accelerator NvidiaTeslaT4 --timeout 1800',
        "scope": "standalone100 engineering benchmark only; no serving-default change, full9h claim or competition submission"}
    dump(out / "package-review.json", review)
    return review


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    q = sub.add_parser("build"); q.add_argument("--out", required=True); q.add_argument("--reference-members", required=True)
    q = sub.add_parser("run")
    for name in ("base", "assets", "competition", "codec-wheel", "work"): q.add_argument("--" + name, required=True)
    q = sub.add_parser("child")
    for name in ("base", "assets", "competition", "selection", "work", "codec"): q.add_argument("--" + name, required=True)
    a = p.parse_args()
    if a.action == "build": print(json.dumps(build(a.out, a.reference_members), indent=2))
    elif a.action == "run": run(a.base, a.assets, a.competition, a.codec_wheel, a.work)
    else: child(a.base, a.assets, a.competition, a.selection, a.work, a.codec)
