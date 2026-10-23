"""V3 bounded integration of the reviewed Windows launcher repair.

This copy preserves V1 and binds the observed Windows venv redirector ancestry.
New bounded phase registered in H-C-15; exact allocation review still required before run.
Allocate and supervise ONE reviewed HC15 CPU sample; never a cache build.

Run allocate with the isolated CPU Python, then obtain an independent read-only
review. run accepts a review JSON with schema hc15-cache-sample-review-v3,
verdict READY, read_only true, reviewer, allocation_sha256 and source_sha256
(the exact allocation's source mapping). Neither allocation nor exit zero is
training/GPU admission. Failures retain the attempt marker; there is no retry.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import sysconfig
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.adapters.hc15_cache_store import (CacheStore, canonical, check_hash,
                                               digest_file, frozen_contract, require)

SOURCE_PATHS = {
    "adapter": "training/adapters/goodpjw_reader.py",
    "decoder": "training/adapters/hc15_archive_decoder.py",
    "store": "training/adapters/hc15_cache_store.py",
    "supervisor": "tools/process_resource_receipt.py",
    "cli": "tools/hc15_cache_sample_v3.py",
}
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
SAMPLE_EXECUTION_ENABLED = True  # One registered V3 integration; review/attempt gates remain.
MAX_BYTES = 256 * 1024 * 1024
SUPERVISION_ALLOWANCE = 4 * 1024 * 1024


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_new(path, value):
    """Exclusive durable write; a partial write also blocks automatic replay."""
    with Path(path).open("xb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())


def stat_identity(path):
    value = Path(path).stat()
    return {"path": str(Path(path).resolve()), "bytes": value.st_size,
            "mtime_ns": value.st_mtime_ns, "device": value.st_dev, "inode": value.st_ino}


def environment_identity():
    """Metadata only: no torch, numpy, OpenCV, pydicom or psutil import."""
    prefix = Path(sys.prefix).resolve()
    cfg = prefix / "pyvenv.cfg"
    require(sys.prefix != sys.base_prefix and cfg.is_file(), "Isolated CPU venv required")
    require("include-system-site-packages = false" in cfg.read_text().lower(),
            "System site packages must be disabled")
    # ROOT is on sys.path for the reviewed project modules, but its local
    # egg-info is not an installed backend dependency. Enumerate only this
    # venv's actual library directories rather than all sys.path metadata.
    package_directories = []
    for key in ("purelib", "platlib"):
        directory = Path(sysconfig.get_path(key, vars={"base": str(prefix), "platbase": str(prefix)})).resolve()
        require(directory.is_relative_to(prefix) and directory.is_dir(), "Package directory outside isolated CPU venv")
        if str(directory) not in package_directories:
            package_directories.append(str(directory))
    packages = {}
    for dist in importlib.metadata.distributions(path=package_directories):
        name = dist.metadata["Name"].lower().replace("_", "-")
        require(name not in packages, "Duplicate installed distribution")
        # A distribution (notably setuptools) may vendor other packages whose
        # nested *.dist-info/RECORD files also occur in its installed-file list.
        # Only the distribution's own top-level metadata directory binds it.
        records = [p for p in (dist.files or [])
                   if len(p.parts) == 2 and p.parts[0].endswith(".dist-info") and p.parts[1] == "RECORD"]
        require(len(records) == 1, "Environment package lacks exact RECORD binding")
        record = Path(dist.locate_file(records[0])).resolve()
        require(record.is_relative_to(prefix), "Package outside isolated CPU venv")
        packages[name] = {"version": dist.version, "record": str(record),
                          "record_sha256": digest_file(record)}
    require("+cpu" in packages.get("torch", {}).get("version", ""), "CPU-only torch distribution required")
    require(all(name in packages for name in ("numpy", "pydicom", "psutil"))
            and any(name in packages for name in ("opencv-python", "opencv-python-headless")),
            "Missing CPU backend/supervision package")
    executables = {str(Path(p).resolve()): digest_file(p)
                   for p in (sys.executable, getattr(sys, "_base_executable", sys.executable))}
    return {"python": str(Path(sys.executable).resolve()), "prefix": str(prefix),
            "version": sys.version, "executable_sha256": executables,
            "pyvenv_cfg_sha256": digest_file(cfg), "packages": packages,
            "package_directories": package_directories,
            "worker_environment": CPU_ENV}


def sample_members(index, rows, cap):
    result = []
    with closing(sqlite3.connect(Path(index).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        for row in rows:
            values = db.execute("SELECT path,info FROM files WHERE split='train' AND uid=? AND series=? ORDER BY path",
                                (row["StudyInstanceUID"], row["SeriesInstanceUID"])).fetchall()
            members = [{"path": path, "zip_info": json.loads(info)} for path, info in values if path.endswith(".dcm")]
            prefix = f"train_series/{row['StudyInstanceUID']}/{row['SeriesInstanceUID']}/"
            require(members and len({m['path'] for m in members}) == len(members)
                    and all(m["path"].startswith(prefix) for m in members), "Invalid sample source members")
            result.append({"series": row["SeriesInstanceUID"], "raw_frames": len(members),
                           "source_members_sha256": sha(members)})
            require(sum(s["raw_frames"] for s in result) <= cap, "Raw-frame cap exceeded; no adaptive sample selection")
    return result


def build_allocation(root, manifest, expected_sha, out, *, max_series=3, max_raw_frames=192,
                     max_bytes=MAX_BYTES, seconds=120, reserve_bytes=16*1024**3):
    root, manifest, out = Path(root).resolve(), Path(manifest).resolve(), Path(out).resolve()
    require(out.is_relative_to(root) and out != root, "Output must be a new child of project root")
    require(manifest.is_relative_to(root), "Manifest outside project root")
    require(type(max_series) is int and 1 <= max_series <= 3, "Sample series cap must be 1..3")
    require(type(max_raw_frames) is int and 1 <= max_raw_frames <= 192, "Raw-frame cap must be 1..192")
    require(type(max_bytes) is int and 0 < max_bytes <= MAX_BYTES, "Invalid sample byte budget")
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 120,
            "Invalid aggregate sample time budget")
    require(type(reserve_bytes) is int and reserve_bytes >= 0, "Invalid disk reserve")
    check_hash(expected_sha)
    require(digest_file(manifest) == expected_sha, "Preparation manifest drift")
    prep = json.loads(manifest.read_text(encoding="utf-8"))
    require(prep.get("gpu_dispatch_allowed") is False, "Preparation unexpectedly permits GPU dispatch")
    archive, index = Path(prep["archive"]["path"]), Path(prep["archive"]["index"])
    stats_before = {"archive": stat_identity(archive), "index": stat_identity(index)}
    require(stats_before["archive"]["bytes"] == prep["archive"]["bytes"], "Archive size drift")
    index_hash = digest_file(index)
    sources = {name: digest_file(root / path) for name, path in SOURCE_PATHS.items()}
    sources["manifest"] = expected_sha
    contract = frozen_contract(root, manifest, expected_sha, root / SOURCE_PATHS["adapter"],
                               root / SOURCE_PATHS["decoder"], index_sha256=index_hash)
    require(len(contract["series"]) > max_series, "Sample must leave frozen series unbuilt")
    selected = sample_members(index, contract["series"][:max_series], max_raw_frames)
    require(stats_before == {"archive": stat_identity(archive), "index": stat_identity(index)},
            "Archive/index changed while allocating")
    result = {"schema": "hc15-cache-sample-allocation-v3", "scope": "SAMPLE_ONLY",
              "root": str(root), "output": str(out), "manifest": str(manifest),
              "source_sha256": sources, "source_stats": stats_before,
              "environment": environment_identity(), "launcher_policy": launcher_policy(), "contract": contract,
              "sample": selected, "gpu_dispatch_allowed": False, "training_ready": False,
              "budget": {"max_series": max_series, "max_raw_frames": max_raw_frames,
                         "max_bytes": max_bytes, "seconds": seconds, "reserve_bytes": reserve_bytes},
              "limitations": ["Archive identity is stat-bound, not a full 265GB content digest; ZIP reads check CRC",
                              "Environment RECORD files are bound; installed native library bytes are not rehashed",
                              "Byte accounting gates cache writes and checks aggregate output; no OS filesystem quota",
                              "Deterministic first-series sample is not representative full-cache throughput evidence",
                              "Supervisor samples process trees; see its receipt for sampling/cleanup limitations"]}
    # The full cohort contract is intentionally retained in both allocation and
    # CacheStore identity. Allow its duplicate metadata plus bounded pixels.
    required = 2 * len(canonical(result)) + SUPERVISION_ALLOWANCE + 1024**2
    required += sum(min(s["raw_frames"], 64) * 384 * 384 + 65536 for s in selected)
    require(required <= max_bytes, "Byte budget cannot hold sample and full-contract metadata")
    return result


def allocate(root, manifest, expected_sha, out, **budget):
    require(SAMPLE_EXECUTION_ENABLED, "V3 allocation disabled pending new phase and exact review")
    require(not Path(out).exists(), "Allocation output already exists; no overwrite")
    allocation = build_allocation(root, manifest, expected_sha, out, **budget)
    Path(out).mkdir(parents=True, exist_ok=False)
    write_new(Path(out) / "allocation.json", allocation)
    return Path(out) / "allocation.json"


def validate(allocation_path, review_path):
    path = Path(allocation_path).resolve()
    allocation = json.loads(path.read_text(encoding="utf-8"))
    require(allocation.get("schema") == "hc15-cache-sample-allocation-v3", "Invalid allocation schema")
    require(Path(allocation["root"]).resolve() == ROOT, "Allocation belongs to a different project")
    require(path == Path(allocation["output"]) / "allocation.json", "Allocation moved from bound output")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require(review.get("schema") == "hc15-cache-sample-review-v3" and review.get("verdict") == "READY"
            and review.get("read_only") is True and bool(review.get("reviewer")), "Independent read-only READY review required")
    require(review.get("allocation_sha256") == digest_file(path)
            and review.get("source_sha256") == allocation["source_sha256"], "Review not bound to exact allocation/sources")
    budget = allocation["budget"]
    expected = build_allocation(allocation["root"], allocation["manifest"],
                                allocation["source_sha256"]["manifest"], allocation["output"], **budget)
    require(allocation == expected, "Allocation/source/environment drift")
    return allocation


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _supervise(argv, out, timeout, cwd):
    from tools.process_resource_receipt import supervise
    return supervise(argv, out, timeout, cwd)


def run(allocation_path, review_path):
    require(SAMPLE_EXECUTION_ENABLED, "V3 sample execution disabled pending new phase/allocation and exact review")
    started = time.monotonic()
    # Fail quickly on an earlier/crashed attempt, before expensive revalidation.
    out = Path(allocation_path).resolve().parent
    require(not (out / "attempt.json").exists(), "Earlier attempt exists; no automatic retry")
    allocation = validate(allocation_path, review_path)
    budget = allocation["budget"]
    require(not any((out / name).exists() for name in ("cache", "supervision", "worker.claim", "worker.json", "result.json")),
            "Existing sample output; no overwrite or retry")
    # Existing supervisor has bounded terminate/kill cleanup. Leave room inside
    # the aggregate wall budget instead of granting cleanup a new time budget.
    remaining = budget["seconds"] - (time.monotonic() - started) - 4
    require(remaining > 0, "Aggregate time budget exhausted before dispatch")
    token = uuid.uuid4().hex
    argv = worker_argv(allocation, allocation_path, review_path, token)
    attempt = {"schema": "hc15-cache-sample-attempt-v3", "allocation_sha256": digest_file(allocation_path),
               "review_sha256": digest_file(review_path), "parent_identity": process_identity(os.getpid()),
               "launcher_policy": launcher_policy(), "worker_argv": argv, "token": token,
               "started_unix": time.time(), "deadline_unix": time.time() + remaining,
               "budget": budget, "status": "ATTEMPTED_NO_RETRY"}
    write_new(out / "attempt.json", attempt)
    previous = {key: os.environ.get(key) for key in (*CPU_ENV, "PYTHONPATH", "PYTHONHOME")}
    result = {"schema": "hc15-cache-sample-result-v3", "status": "FAILED",
              "allocation_sha256": attempt["allocation_sha256"], "training_ready": False,
              "gpu_dispatch_allowed": False, "retries": 0, "budget": budget}
    try:
        os.environ.update(CPU_ENV)
        os.environ.pop("PYTHONPATH", None); os.environ.pop("PYTHONHOME", None)
        receipt = _supervise(argv, out / "supervision", remaining, allocation["root"])
        require(receipt["status"] == "COMPLETE" and receipt["child_exit_code"] == 0, "Supervised sample failed")
        worker_result = json.loads((out / "worker.json").read_text())
        require(worker_result.get("status") == "SAMPLE_VALIDATED"
                and worker_result.get("allocation_sha256") == attempt["allocation_sha256"]
                and worker_result.get("series") == allocation["sample"]
                and worker_result.get("complete") is False, "Sample receipt invalid")
        require(not (out / "cache" / "cache.json").exists(), "Sample unexpectedly finalized full cache")
        result["status"] = "SAMPLE_VALIDATED"
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        result["elapsed_seconds"] = time.monotonic() - started
        result["bytes_before_result"] = directory_bytes(out)
        if result["elapsed_seconds"] > budget["seconds"] or result["bytes_before_result"] + 65536 > budget["max_bytes"]:
            result["status"] = "FAILED"
            result["failure"] = {"type": "AggregateBudgetExceeded", "message": "Sample exceeded aggregate time/byte budget"}
        write_new(out / "result.json", result)
    return result



def process_identity(pid):
    """Read one live PID identity twice; creation time prevents PID reuse."""
    import psutil
    proc = psutil.Process(pid)
    created = proc.create_time()
    value = {"pid": proc.pid, "create_time": created, "ppid": proc.ppid(),
             "exe": str(Path(proc.exe()).resolve()), "argv": proc.cmdline()}
    require(psutil.Process(pid).create_time() == created, "PID reused during ancestry read")
    return value


def launcher_policy():
    """Static binary/path binding; one Windows redirector is the only exception."""
    launcher = Path(sys.executable).resolve()
    worker = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    return {"schema": "hc15-launcher-policy-v1", "platform": sys.platform,
            "max_intermediates": 1 if sys.platform == "win32" else 0,
            "launcher": {"path": str(launcher), "sha256": digest_file(launcher)},
            "worker": {"path": str(worker), "sha256": digest_file(worker)}}


def worker_argv(allocation, allocation_path, review_path, token):
    return [allocation["environment"]["python"], "-I", str(Path(allocation["root"]) / SOURCE_PATHS["cli"]),
            "_worker", "--allocation", str(Path(allocation_path).resolve()),
            "--review", str(Path(review_path).resolve()), "--token", token]


def verify_chain(attempt, chain):
    """Pure checks after capture: unrelated ancestors or reused PIDs fail."""
    policy = attempt["launcher_policy"]
    require(policy["platform"] == sys.platform and policy["max_intermediates"] == (1 if sys.platform == "win32" else 0),
            "Unreviewed launcher platform/depth")
    require(2 <= len(chain) <= 2 + policy["max_intermediates"], "Unexpected launcher ancestry depth")
    require(len({node["pid"] for node in chain}) == len(chain), "Cyclic launcher ancestry")
    parent = attempt["parent_identity"]
    require(chain[-1] == parent, "Parent PID identity changed or unrelated parent")
    for child, ancestor in zip(chain, chain[1:]):
        require(child["ppid"] == ancestor["pid"] and child["create_time"] >= ancestor["create_time"],
                "Broken launcher parent identity/order")
    argv = attempt["worker_argv"]
    worker = chain[0]
    require(Path(worker["exe"]) == Path(policy["worker"]["path"])
            and worker["argv"][1:] == argv[1:]
            and Path(worker["argv"][0]) in (Path(policy["worker"]["path"]), Path(policy["launcher"]["path"])),
            "Worker executable/argv differs")
    require(worker["create_time"] >= attempt["started_unix"], "Worker predates attempt")
    for intermediary in chain[1:-1]:
        require(Path(intermediary["exe"]) == Path(policy["launcher"]["path"])
                and intermediary["argv"] == argv
                and intermediary["create_time"] >= attempt["started_unix"],
                "Unapproved redirector executable/argv/creation time")
    for role in ("worker", "launcher"):
        require(digest_file(policy[role]["path"]) == policy[role]["sha256"], "Launcher binary changed")
    return {"schema": "hc15-launcher-proof-v1", "chain": chain, "policy": policy}


def validate_parent(attempt, *, reader=None, worker_pid=None):
    reader = reader or process_identity
    worker_pid = os.getpid() if worker_pid is None else worker_pid
    chain, pid = [], worker_pid
    # Worker, optional single Windows redirector, exact supervising parent.
    for _ in range(3):
        node = reader(pid)
        chain.append(node)
        if node["pid"] == attempt["parent_identity"]["pid"]:
            break
        pid = node["ppid"]
    proof = verify_chain(attempt, chain)
    # A process must remain the same live identity across the whole traversal.
    require(all(reader(node["pid"]) == node for node in chain), "Ancestry changed during verification")
    return proof


def claim_worker(allocation_path, review_path, token):
    """Hash/token/parent proof first, then durable single-use worker claim."""
    out = Path(allocation_path).resolve().parent
    attempt = json.loads((out / "attempt.json").read_text())
    require(attempt.get("schema") == "hc15-cache-sample-attempt-v3"
            and attempt["token"] == token
            and attempt["allocation_sha256"] == digest_file(allocation_path)
            and attempt["review_sha256"] == digest_file(review_path), "Worker lacks exact attempt binding")
    require(time.time() < attempt["deadline_unix"], "Attempt deadline exhausted")
    proof = validate_parent(attempt)
    write_new(out / "worker.claim", {"pid": os.getpid(), "token": token,
                                    "create_time": proof["chain"][0]["create_time"], "ancestry": proof})
    return proof



def worker(allocation_path, review_path, token):
    require(SAMPLE_EXECUTION_ENABLED, "V3 sample worker disabled pending new phase/allocation and exact review")
    out = Path(allocation_path).resolve().parent
    attempt = json.loads((out / "attempt.json").read_text())
    proof = claim_worker(allocation_path, review_path, token)
    require(all(os.environ.get(k) == v for k, v in CPU_ENV.items()), "CPU worker environment differs")
    allocation = validate(allocation_path, review_path)
    require(attempt["launcher_policy"] == allocation["launcher_policy"], "Launcher policy differs from reviewed allocation")
    require(attempt["worker_argv"] == worker_argv(allocation, allocation_path, review_path, token), "Worker argv differs from allocation")
    seconds = min(allocation["budget"]["seconds"], attempt["deadline_unix"] - time.time())
    require(seconds > 0, "Aggregate time budget exhausted")
    deadline = time.monotonic() + seconds
    from training.adapters.hc15_archive_decoder import ArchiveDecoder, cpu_backend, produce_chunk
    decode, preprocess = cpu_backend()
    decoder = ArchiveDecoder(allocation["contract"]["source_archive"],
                             allocation["contract"]["archive_index_sha256"], decode, preprocess)
    selected = {s["series"]: s for s in allocation["sample"]}

    def sample_decoder(row, deadline):
        require(row["SeriesInstanceUID"] in selected, "Decode escaped allocated sample")
        payload, count, members_hash = decoder(row, deadline)
        bound = selected[row["SeriesInstanceUID"]]
        require(count == bound["raw_frames"] and members_hash == bound["source_members_sha256"], "Decoded source/count drift")
        return payload, count, members_hash

    sample_decoder.archive_contract = decoder.archive_contract
    sample_decoder.index_sha256 = decoder.index_sha256
    cache_bytes = allocation["budget"]["max_bytes"] - directory_bytes(out) - SUPERVISION_ALLOWANCE
    with CacheStore(out / "cache", allocation["contract"], max_bytes=cache_bytes,
                    max_total_seconds=max(.001, deadline - time.monotonic()),
                    reserve_bytes=allocation["budget"]["reserve_bytes"]) as store:
        chunk = produce_chunk(store, sample_decoder, decoder_sha256=allocation["source_sha256"]["decoder"],
                              max_series=len(selected), deadline=deadline)
        require(chunk["completed_this_chunk"] == len(selected) and chunk["complete"] is False,
                "Incomplete sample or unexpected full-cache completion")
        for sid, bound in selected.items():
            record = store.records[sid]
            require(record["decoded_count"] == bound["raw_frames"]
                    and record["source_members_sha256"] == bound["source_members_sha256"], "Stored sample receipt drift")
    require(time.monotonic() < deadline, "Aggregate time budget exhausted")
    write_new(out / "worker.json", {"status": "SAMPLE_VALIDATED", "allocation_sha256": digest_file(allocation_path),
                                    "series": allocation["sample"], "complete": False,
                                    "training_ready": False, "gpu_dispatch_allowed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    allocate_parser = commands.add_parser("allocate")
    allocate_parser.add_argument("--manifest", required=True)
    allocate_parser.add_argument("--manifest-sha256", required=True)
    allocate_parser.add_argument("--out", required=True)
    allocate_parser.add_argument("--max-series", type=int, default=3)
    allocate_parser.add_argument("--max-raw-frames", type=int, default=192)
    allocate_parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    allocate_parser.add_argument("--seconds", type=float, default=120)
    allocate_parser.add_argument("--reserve-bytes", type=int, default=16*1024**3)
    for name in ("run", "_worker"):
        command = commands.add_parser(name)
        command.add_argument("--allocation", required=True)
        command.add_argument("--review", required=True)
        if name == "_worker":
            command.add_argument("--token", required=True)
    args = parser.parse_args()
    if args.command == "allocate":
        path = allocate(ROOT, args.manifest, args.manifest_sha256, args.out,
                        max_series=args.max_series, max_raw_frames=args.max_raw_frames,
                        max_bytes=args.max_bytes, seconds=args.seconds, reserve_bytes=args.reserve_bytes)
        print(json.dumps({"allocation": str(path), "sha256": digest_file(path), "status": "REVIEW_REQUIRED"}))
    elif args.command == "run":
        result = run(args.allocation, args.review)
        print(json.dumps({"status": result["status"], "training_ready": False, "gpu_dispatch_allowed": False}))
        return 0 if result["status"] == "SAMPLE_VALIDATED" else 1
    else:
        worker(args.allocation, args.review, args.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


