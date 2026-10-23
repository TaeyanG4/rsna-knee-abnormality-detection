"""Versioned D01 study-sharded R2 helper and private engineering checks only.

No candidate/submission builder exists. Original v2, original anchor cells,
immutable model assets, input107/codec gates and historical FAILs are preserved.
"""
from __future__ import annotations
import argparse
import ast
import base64
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[1]
ASSETS_SHA = "e62893ebb1e22dbb1288ea6c805097bbc96ef5e43759f42a22d7cfb3bbef036b"
GDCM_SHA = "512d0d903e12c5ad389939a257817bf182ce288431b3d474d62cb0c0833b8b56"
LEGACY_SHA = "e478626baaf757260de3f8b03f143e727ebb130f575404788e2d9dad1264867e"
SUPERVISOR_SHA = "fbf5790f79dffe9fdf01610fa328f48c2a65054ca80c6ecc8327d84ed40d82d8"
V2_SHA = "2be3441f7c574b46205c7df3c085f9ecf5262403ba436a243b897058842a9f36"
UID = "StudyInstanceUID"
POLICY = {"version": "d01-study-sharding-v3", "shards": 2, "partition": "input positions 0::2 and1::2",
          "cuda_visible_devices": ["0", "1"], "cpu_threads_per_child": 2, "minimum_effective_cpus": 4,
          "header_policy": "legacy-full-read", "member_count": 5, "amp": "fp16", "runtime_policy": "strict",
          "residency": "resident", "batch_size": 1, "mean": "float32 raw5 arithmetic mean before rank",
          "blend": {"anchor_rank": .9, "r2_rank_probability_mean": .1, "ties": "average", "final_rerank": False}}
POLICY_SHA = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def need(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value; spec.loader.exec_module(value); return value


def split_ids(ids):
    ids = list(ids)
    need(bool(ids) and len(ids) == len(set(ids)) and all(isinstance(u, str) and u for u in ids), "nonempty unique study IDs required")
    return [ids[::2], ids[1::2]]


def merge_members(ids, parts):
    import numpy as np
    split_ids(ids); locations = {}
    for part, (actual_ids, raw) in enumerate(parts):
        raw = np.asarray(raw)
        need(raw.dtype == np.float32 and raw.shape == (5, len(actual_ids), 12), "five-member shard shape/dtype")
        need(np.isfinite(raw).all() and (raw >= 0).all() and (raw <= 1).all(), "nonfinite/out-of-range shard probability")
        for index, uid in enumerate(actual_ids):
            need(uid not in locations, "duplicate shard study")
            locations[uid] = (part, index)
    need(set(locations) == set(ids), "missing/unexpected shard study")
    raw = np.stack([parts[locations[u][0]][1][:, locations[u][1], :] for u in ids], axis=1)
    return raw, raw.mean(axis=0, dtype=np.float32)


def validate_bundle(base):
    base = Path(base); manifest = read(base / "helper-manifest.json")
    need(manifest["schema"] == "d01-helper-v3" and manifest["policy"] == POLICY and manifest["policy_sha256"] == POLICY_SHA,
         "new scheduling identity differs")
    for name, digest in manifest["files"].items(): need(sha(base / name) == digest, f"changed helper/evidence: {name}")
    need(manifest["files"]["d01_release_v3.py"] == sha(__file__), "executing helper differs from frozen bundle")
    need(manifest["files"]["d01_r2_shard_benchmark.py"] == SUPERVISOR_SHA and manifest["files"]["d01_release_v2.py"] == V2_SHA,
         "pinned supervisor/static validator changed")
    v2 = module(base / "d01_release_v2.py", "d01_unchanged_v2_validator")
    p, expected = read(base / "protocol.json"), read(base / "expected_inputs.json")
    psha = sha(base / "protocol.json")
    need(p["schema"] == "d01-numerical-input-gate-v2" and p["v1_result"] == "FAILED_UNCHANGED", "prior failure/input schema")
    need((p["window_atol"], p["normalized_atol"], p["rtol"]) == (5e-7, 3e-6, 0.), "frozen numerical tolerance")
    need(p["assets_manifest_sha256"] == ASSETS_SHA and p["gdcm_wheel_sha256"] == GDCM_SHA, "frozen input dependencies")
    need(p["expected_sha256"] == sha(base / "expected_inputs.json") and p["helper_sha256"] == sha(base / "d01_input_gate.py"), "input reference bindings")
    v2.validate_numeric(read(base / "numeric_local.json"), p, psha, expected, "local_cache")
    v2.validate_numeric(read(base / "numeric_remote.json"), p, psha, expected, "remote_dicom")
    codec = read(base / "codec.json")
    need(codec["status"] == "PASS" and codec["gdcm_version"] == "3.2.6" and codec["pydicom_version"] == "3.0.2", "codec prior gate")
    syntaxes = {r["uid"] for r in codec["decoders"] if r.get("available") is True}
    need(syntaxes == {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1", *[f"1.2.840.10008.1.2.4.{n}" for n in (57, 70, 90, 91)]}, "six codec capabilities")
    need(len(codec["fixtures"]) >= 3 and all(r.get("decode") == "SUCCESS" for r in codec["fixtures"])
         and any(r.get("matches_uncompressed_fixture") is True for r in codec["fixtures"]), "codec3 fixture evidence")
    return manifest, p


def ids_from_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f: ids = [r[UID] for r in csv.DictReader(f)]
    split_ids(ids); return ids


def child(base, assets, competition, split, ids_path, codec_env, work, public_reference=False):
    base, assets, competition, work = map(Path, (base, assets, competition, work))
    manifest, protocol = validate_bundle(base)
    ids = read(ids_path)["ids"]; need(bool(ids), "empty shard must be skipped by supervisor")
    need(split in {"train", "test"} and os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "1"}, "explicit split and single GPU binding")
    work.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    receipt = {"status": "RUNNING", "schema": "d01-shard-child-v3", "ids": ids, "split": split,
               "release_helper_sha256": sha(__file__), "policy_sha256": POLICY_SHA, "failures": []}
    try:
        need(sha(assets / "d01-assets-manifest.json") == ASSETS_SHA and sha(assets / "training/src/raptor_dicom.py") == LEGACY_SHA, "old asset/legacy reader identity")
        package = module(assets / "d01_package.py", "d01_original_package")
        package._validate_assets(assets)
        sys.path.insert(0, str(codec_env)); package._activate_offline_environment(assets, work)
        import numpy as np
        import torch
        from raptor_common import environment, write_predictions, select_centers
        from raptor_dicom import DicomCorpus, DirectoryDicomSource
        from raptor_ensemble import load_members, predict_ensemble
        from pydicom.pixels import get_decoder
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and "T4" in torch.cuda.get_device_name(0), "one actual visible T4 per child")
        torch.set_num_threads(2)
        need(all(get_decoder(u).is_available for u in ["1.2.840.10008.1.2", "1.2.840.10008.1.2.1", *[f"1.2.840.10008.1.2.4.{n}" for n in (57,70,90,91)]]), "six decoder capabilities")
        with (competition / f"{split}_series.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = [{k: r[k] for k in (UID, "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive")}
                    for r in csv.DictReader(f) if r[UID] in set(ids)]
        load_start = time.perf_counter(); members, _ = load_members(assets / "d01-assets-manifest.json")
        receipt["model_load_seconds"] = time.perf_counter() - load_start
        need(len(members) == 5, "five R2 members required")
        torch.cuda.reset_peak_memory_stats(); infer_start = time.perf_counter()
        with DirectoryDicomSource(competition / f"{split}_series") as source:
            corpus = DicomCorpus.from_records(ids, rows, source, "raptor64_336_c140_s02", split=split)
            checks = []
            if public_reference:
                need(split == "test", "public numerical checks require test split")
                expected = read(assets / "validation/expected_test3.json")
                by_uid = {r["uid"]: r for r in expected["study_hashes"]}
                need(set(ids).issubset(by_uid), "public shard contains unregistered UID")
                input_helper = module(base / "d01_input_gate.py", "d01_frozen_numeric")
                for uid in ids:
                    volume, mask = corpus.volume(uid), corpus.mask(uid)
                    need(input_helper.array_sha(volume) == by_uid[uid]["volume_sha256"] and input_helper.array_sha(mask) == by_uid[uid]["mask_sha256"], "public integer input changed")
                    centers = select_centers(mask, corpus.depth, 24, corpus.spec, policy="legacy")
                    checks.append(input_helper.check_case(volume, mask, centers, input_helper.expected_row(uid, volume, mask, centers), protocol))
                need(all(r["pass"] for r in checks), "public3 numerical reference failure")
            mean, raw, inference = predict_ensemble(members, corpus, amp="fp16", device="cuda", batch=1, residency="resident", runtime_policy="strict")
            torch.cuda.synchronize()
            need(raw.shape == (5, len(ids), 12) and np.isfinite(raw).all() and torch.get_num_threads() == 2, "child output/thread contract")
            write_predictions(work / "mean.csv", ids, mean)
            np.savez_compressed(work / "members.npz", ids=np.asarray(ids), member_probabilities=raw)
            receipt.update(status="COMPLETE", inference=inference, study_receipts=[corpus.receipts[u] for u in ids],
                input_reference_checks=checks, inference_seconds=time.perf_counter()-infer_start,
                environment=environment(), cpu_threads=2, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                logical_gpu_count=1, codec_version=__import__("gdcm").Version.GetVersion(),
                cuda_memory={"peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()})
    except Exception as exc:
        receipt["status"] = "FAILED"; receipt["failures"].append({"error": str(exc), "traceback": traceback.format_exc()})
        if hasattr(exc, "receipt"): receipt["inference_failure"] = exc.receipt
        raise
    finally:
        receipt["total_seconds"] = time.perf_counter()-started; dump(work / "receipt.json", receipt)


def run_r2(base, assets, codec, competition, work, ids, split="test", deadline=None, public_reference=False):
    """Arbitrary N inference; empty total rejected, empty second shard skipped."""
    started = time.perf_counter(); shards = split_ids(ids)
    base, assets, codec, competition, work = map(Path, (base, assets, codec, competition, work))
    manifest, _ = validate_bundle(base)
    need(os.name == "posix" and str(work.resolve()).startswith("/kaggle/working/"), "Kaggle Linux explicit execution only")
    need(deadline is not None and deadline > time.monotonic(), "shared wall deadline required")
    work.mkdir(parents=True, exist_ok=False)
    result = {"status": "RUNNING", "schema": "d01-r2-serving-v3", "ids": ids, "split": split, "policy": POLICY,
              "policy_sha256": POLICY_SHA, "release_helper_sha256": sha(__file__), "bundle_sha256": sha(base / "helper-manifest.json"),
              "assets_manifest_sha256": ASSETS_SHA, "header_policy": "legacy-full-read", "failures": []}
    try:
        supervisor = module(base / "d01_r2_shard_benchmark.py", "d01_proven_supervisor")
        result["cpu"] = supervisor.effective_cpu(); need(result["cpu"]["effective_cpus"] >= 4, "at least four effective CPUs required")
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=20).stdout.strip().splitlines()
        need(len(gpu) == 2 and all("T4" in line for line in gpu), "exactly two actual T4 devices required"); result["gpu_inventory"] = gpu
        wheels = [p for p in codec.rglob("*.whl") if sha(p) == GDCM_SHA]; need(len(wheels) == 1, "frozen gdcm wheel")
        codec_env = work / "codec-environment"
        subprocess.run([sys.executable,"-m","pip","install","--no-index","--no-deps","--target",str(codec_env),str(wheels[0])], check=True, timeout=min(120, max(1, deadline-time.monotonic())))
        commands, envs, active = [], [], []
        result["skipped_shards"] = []
        for index, subset in enumerate(shards):
            if not subset:
                result["skipped_shards"].append({"index": index, "reason": "empty deterministic shard", "model_load": False}); continue
            active.append(index); path = work / f"ids-{index}.json"; dump(path, {"ids": subset})
            command = [sys.executable,str(base/"d01_release_v3.py"),"child","--base",str(base),"--assets",str(assets),
                "--competition",str(competition),"--split",split,"--ids-path",str(path),"--codec-env",str(codec_env),"--work",str(work/f"child-{index}")]
            if public_reference: command.append("--public-reference")
            commands.append(command); envs.append(dict(os.environ,CUDA_VISIBLE_DEVICES=str(index),CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        result["parallel"] = supervisor.supervise(commands, envs, work, deadline-time.monotonic())
        import numpy as np
        parts, receipts = [], []
        for index in active:
            directory = work / f"child-{index}"; r = read(directory / "receipt.json")
            arr = np.load(directory / "members.npz", allow_pickle=False); actual_ids = arr["ids"].tolist(); inf = r.get("inference", {})
            need(r["status"] == "COMPLETE" and not r["failures"] and actual_ids == shards[index] == r["ids"], "child completion or UID coverage")
            need(r["release_helper_sha256"] == sha(__file__) and r["policy_sha256"] == POLICY_SHA and r["cuda_visible_devices"] == str(index)
                 and r["logical_gpu_count"] == 1 and r["cpu_threads"] == 2, "child helper or device/thread isolation")
            need(inf.get("status") == "COMPLETE" and inf.get("all_finite") is True and inf.get("member_count") == 5
                 and inf.get("studies") == len(actual_ids) and inf.get("fp32_fallbacks") == 0 and inf.get("neutral_fills") == 0
                 and inf.get("effective_amp") == "fp16" and inf.get("runtime", {}).get("policy") == "strict", "child inference policy")
            need(all(m["status"] == "COMPLETE" and m["completed_studies"] == len(actual_ids) for m in inf["member_results"]), "child member incomplete")
            receipts.append(r); parts.append((actual_ids, arr["member_probabilities"]))
        if len(receipts) == 2:
            for field in ("input_contract", "runtime"):
                need(receipts[0]["inference"][field] == receipts[1]["inference"][field], "shards disagree on input/runtime contract")
            need([(m["name"],m["checkpoint_sha256"]) for m in receipts[0]["inference"]["member_results"]] ==
                 [(m["name"],m["checkpoint_sha256"]) for m in receipts[1]["inference"]["member_results"]], "shard member identities differ")
        raw, mean = merge_members(ids, parts)
        from_mapping = lambda key: {r["uid"]: r for child in receipts for r in child[key]}
        studies = from_mapping("study_receipts")
        tensors = {r["uid"]: r for c in receipts for r in c["inference"]["input_tensor_hashes"]}
        need(set(studies) == set(tensors) == set(ids), "merged input receipt coverage")
        numerical = from_mapping("input_reference_checks")
        np.savez_compressed(work / "members.npz", ids=np.asarray(ids), member_probabilities=raw)
        labels = receipts[0]["inference"]["input_contract"]["targets"]
        with (work / "mean.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow([UID,*labels]); writer.writerows([[uid,*row] for uid,row in zip(ids,mean,strict=True)])
        need(time.monotonic() < deadline, "shared wall deadline exhausted before R2 completion")
        result.update(status="COMPLETE",studies=len(ids),member_count=5,shape=list(raw.shape),all_finite=True,
            fp32_fallbacks=0,neutral_fills=0,mean_before_rank=True,children=receipts,
            study_receipts=[studies[u] for u in ids],input_tensor_hashes=[tensors[u] for u in ids],
            input_reference_checks=[numerical[u] for u in ids] if public_reference else [],
            member_identities=[{"name":m["name"],"sha256":m["checkpoint_sha256"]} for m in receipts[0]["inference"]["member_results"]])
    except BaseException as exc:
        result["status"] = "FAILED"; result["failures"].append({"error":str(exc),"traceback":traceback.format_exc()}); raise
    finally:
        result["total_wall_seconds"] = time.perf_counter()-started; dump(work / "receipt.json", result)
    return result


def blend(assets, anchor_frame, r2_frame, ids):
    """Delegate unchanged rank90/10 with mean5 already written by run_r2."""
    assets = Path(assets); need(sha(assets / "d01-assets-manifest.json") == ASSETS_SHA, "blend asset identity")
    return module(assets / "d01_package.py", "d01_original_blend").blend_frames(anchor_frame,r2_frame,ids)


def runtime_budget(r2_gate, r2_launcher, anchor, helper_sha, anchor_sha, safety_seconds=300):
    """Evaluate only supplied actual v3 receipts; never manufacture readiness."""
    need(r2_gate.get("schema") == "d01-t4-preflight-v3" and r2_gate.get("mode") == "train100" and r2_gate.get("status") == "PASS", "fresh v3 train100 PASS required")
    need(r2_gate["release_helper_sha256"] == helper_sha and r2_gate["policy_sha256"] == POLICY_SHA, "unmeasured helper/scheduling policy")
    need(r2_gate.get("studies") == 100 and r2_gate.get("member_count") == 5 and r2_gate.get("all_finite") is True
         and r2_gate.get("fp32_fallbacks") == 0 and r2_gate.get("neutral_fills") == 0
         and set(r2_gate.get("checks", {})) == {"fixed100","input_exact","normalized_tensor_exact","all5_probability_gate"}
         and all(r2_gate["checks"].values()), "incomplete new helper runtime gate")
    need(r2_launcher.get("status") == "COMPLETE" and r2_launcher.get("preflight_gate_sha256") == r2_gate["_source_sha256"], "launcher must bind exact train100 gate bytes")
    need(r2_launcher.get("mode") == "train100" and r2_launcher.get("release_helper_sha256") == helper_sha
         and r2_launcher.get("policy_sha256") == POLICY_SHA, "launcher helper/schedule binding")
    need(anchor.get("status") == "COMPLETE" and anchor.get("benchmark_studies") == 100 and anchor.get("coat_readers_parallel_pairs") is False, "complete anchor100 sequential receipt required")
    need(anchor.get("anchor_notebook_sha256") == anchor_sha and anchor.get("coat_family_members") == 4, "original anchor source/path binding")
    rw, aw = r2_launcher.get("notebook_cell_seconds"), anchor.get("total_cell_seconds")
    need(all(isinstance(v,(int,float)) and math.isfinite(v) and v > 0 for v in (rw,aw)), "finite full measured launcher times")
    need(safety_seconds >= 300, "minimum five-minute reserve is unchanged")
    r2, anc = rw*13, aw*13; remaining = 32400-r2-anc
    return {"schema":"d01-runtime-budget-v3","status":"PASS" if remaining>=safety_seconds else "FAILED",
        "release_helper_sha256":helper_sha,"policy_sha256":POLICY_SHA,"r2_gate_sha256":r2_gate["_source_sha256"],
        "r2_receipt_sha256":r2_gate["r2_receipt_sha256"],
        "r2_launcher_sha256":r2_launcher["_source_sha256"],"anchor_receipt_sha256":anchor["_source_sha256"],
        "max_studies":1300,"whole_seconds":32400,"r2_seconds":r2,"anchor_seconds":anc,"safety_seconds":safety_seconds,
        "remaining_seconds_before_reserve":remaining,"minimum_reserve_pass":remaining>=300,"thirty_minute_target_pass":remaining>=1800,
        "projection":"full cold launcher100 times13 plus full anchor100 times13; no startup deduction; engineering estimate, not hidden guarantee"}


def preflight(base, assets, codec, competition, mode, work, started):
    base, assets, competition, work = map(Path,(base,assets,competition,work)); validate_bundle(base)
    need(mode in {"public3","train100"}, "explicit preflight mode")
    need(not work.exists(), "new immutable preflight output required"); work.mkdir(parents=True)
    gate = {"schema":"d01-t4-preflight-v3","status":"RUNNING","mode":mode,"release_helper_sha256":sha(__file__),
        "policy_sha256":POLICY_SHA,"input_protocol_sha256":sha(base/"protocol.json"),"codec_wheel_sha256":GDCM_SHA,
        "assets_manifest_sha256":ASSETS_SHA,"bundle_sha256":sha(base/"helper-manifest.json"),"header_policy":"legacy-full-read"}
    try:
        import numpy as np
        if mode == "public3":
            ids = ids_from_csv(competition/"test.csv"); need(ids == read(assets/"validation/expected_test3.json")["ids"], "public3 fixed identity")
        else:
            selected = read(base/"selection.json"); ids = selected["ids"]
            need(len(ids)==100 and sha(competition/"train_series.csv")==selected["train_series_sha256"], "train100 fixed identity/series")
        result = run_r2(base,assets,codec,competition,work/"r2",ids,"test" if mode=="public3" else "train",started+1560,mode=="public3")
        raw = np.load(work/"r2/members.npz",allow_pickle=False)["member_probabilities"]
        if mode == "public3":
            with (assets/"validation/local_f0_test3.csv").open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
            labels=result["children"][0]["inference"]["input_contract"]["targets"]
            lookup={r[UID]:r for r in rows}; local=np.array([[float(lookup[u][label]) for label in labels] for u in ids])
            delta=np.abs(raw[0].astype(np.float64)-local)
            checks={"public_ids_exact":True,"three_studies":raw.shape==(5,3,12),"numeric_reference":all(r["pass"] for r in result["input_reference_checks"]),
                    "f0_max":delta.max()<=.002,"f0_mean":delta.mean()<=.0005}
            gate.update(f0_max_abs=float(delta.max()),f0_mean_abs=float(delta.mean()),tolerances={"f0_max_abs":.002,"f0_mean_abs":.0005},
                        input_reference_checks=result["input_reference_checks"])
        else:
            prior=read(base/"reference-receipt.json"); refs=np.load(base/"reference-members.npz",allow_pickle=False)
            need(refs["study_uids"].tolist()==ids,"old100 probability identity")
            need(result["member_identities"]==[{"name":m["name"],"sha256":m["checkpoint_sha256"]} for m in prior["inference"]["member_results"]],"old100 member identity/order")
            old={r["uid"]:r for r in prior["study_receipts"]}; old_tensors={r["uid"]:r for r in prior["inference"]["input_tensor_hashes"]}
            input_checks=[{"uid":r["uid"],"volume_exact":r["volume_sha256"]==old[r["uid"]]["volume_sha256"],
                "mask_exact":r["mask_sha256"]==old[r["uid"]]["mask_sha256"],"full_receipt_exact":r==old[r["uid"]]} for r in result["study_receipts"]]
            probability=module(base/"d01_r2_shard_benchmark.py","d01_probability_contract").probability_gate(raw,refs["member_probabilities"])
            checks={"fixed100":len(ids)==100,"input_exact":all(all(v for k,v in r.items() if k!="uid") for r in input_checks),
                "normalized_tensor_exact":all(r==old_tensors[r["uid"]] for r in result["input_tensor_hashes"]),"all5_probability_gate":probability["pass"]}
            gate.update(input_checks=input_checks,probability_gate=probability,reference_probability_sha256=sha(base/"reference-members.npz"),
                        reference_receipt_sha256=sha(base/"reference-receipt.json"))
        gate.update(checks={k:bool(v) for k,v in checks.items()},studies=len(ids),member_count=5,all_finite=True,
            fp32_fallbacks=0,neutral_fills=0,r2_receipt_sha256=sha(work/"r2/receipt.json"),r2_wall_seconds=result["total_wall_seconds"],
            status="PASS" if all(checks.values()) else "FAILED")
        dump(work/"t4-gate.json",gate); need(gate["status"]=="PASS","new v3 preflight failed; preserve evidence")
    except BaseException as exc:
        gate.update(status="FAILED",error=str(exc),traceback=traceback.format_exc()); raise
    finally: dump(work/"t4-gate.json",gate)
    return gate


def stage(out):
    """Only public3/train100 engineering notebooks; never a submission candidate."""
    out=Path(out); need(not out.exists(),"new immutable staging root")
    bindings=read(ROOT/".local/d01-release-v2-prep/20261003-v1/binding-spec.pending.json")
    raw={"d01_release_v3.py":Path(__file__).read_bytes(),"d01_r2_shard_benchmark.py":(ROOT/"tools/d01_r2_shard_benchmark.py").read_bytes(),
         "d01_release_v2.py":(ROOT/"tools/d01_release_v2.py").read_bytes(),"d01_input_gate.py":Path(bindings["input_helper"]).read_bytes()}
    need(hashlib.sha256(raw["d01_r2_shard_benchmark.py"]).hexdigest()==SUPERVISOR_SHA and hashlib.sha256(raw["d01_release_v2.py"]).hexdigest()==V2_SHA,"reused helper identity")
    for name in ("protocol","expected_inputs","numeric_local","numeric_remote","codec"):
        item=bindings["gates"][name]; need(sha(item["path"])==item["sha256"],"existing gate binding changed")
        raw[name+".json"]=Path(item["path"]).read_bytes()
    historic=ROOT/".local/d01-runtime-staging/20261003-v4"
    selected=read(historic/"selection-manifest.json"); raw["selection.json"]=json.dumps({k:selected[k] for k in ("ids","train_series_sha256")},separators=(",",":")).encode()
    raw["reference-receipt.json"]=(historic/"remote-exact/benchmark-receipt-v2-bound.json").read_bytes()
    raw["reference-members.npz"]=(historic/"remote-exact/engineering-members.npz").read_bytes()
    need(hashlib.sha256(raw["reference-members.npz"]).hexdigest()=="7187999225a2e98ab4e61ba2222d6f4f1a2a4a672e8bb7e381b8bc61e4a7f1b6","exact version2 all5 reference")
    manifest={"schema":"d01-helper-v3","policy":POLICY,"policy_sha256":POLICY_SHA,"v1_result":"FAILED_UNCHANGED",
              "v2":"preserved","files":{name:hashlib.sha256(value).hexdigest() for name,value in raw.items()},
              "scope":"engineering preflight only; runtime gate/candidate remain unbuilt","frozen_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    raw["helper-manifest.json"]=json.dumps(manifest,indent=2).encode()
    payload={name:{"bytes":len(value),"sha256":hashlib.sha256(value).hexdigest(),"data":base64.b64encode(zlib.compress(value,9)).decode()} for name,value in raw.items()}
    out.mkdir(parents=True); bundle=out/"helper";bundle.mkdir()
    for name,value in raw.items():(bundle/name).write_bytes(value)
    validate_bundle(bundle)
    for mode in ("public3","train100"):
        directory=out/mode;directory.mkdir()
        code=f'''from pathlib import Path
import ast,base64,hashlib,importlib.util,json,os,sys,time,zlib
started=time.monotonic()
base=Path('/kaggle/working/d01-v3-{mode}');base.mkdir(exist_ok=False)
payload={payload!r}
for name,item in payload.items():
    value=zlib.decompress(base64.b64decode(item['data']))
    assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
    if name.endswith('.py'):ast.parse(value.decode('utf-8'))
    (base/name).write_bytes(value)
assets_roots=[Path('/kaggle/input/knee-d01-r2-assets'),Path('/kaggle/input/datasets/taeyangg4/knee-d01-r2-assets')]
assets=[p.parent for root in assets_roots if root.is_dir() for p in root.rglob('d01-assets-manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest()=={ASSETS_SHA!r}]
assert len(assets)==1
codec=next(p for p in [Path('/kaggle/input/knee-d01-codec-assets'),Path('/kaggle/input/datasets/taeyangg4/knee-d01-codec-assets')] if p.is_dir())
competition=next(p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p/'test.csv').is_file())
spec=importlib.util.spec_from_file_location('d01_release_v3',base/'d01_release_v3.py');module=importlib.util.module_from_spec(spec);sys.modules['d01_release_v3']=module;spec.loader.exec_module(module)
state='FAILED'
try:
    gate=module.preflight(base,assets[0],codec,competition,{mode!r},base/'results',started)
    state='COMPLETE'
finally:
    path=base/'results/t4-gate.json'
    (base/'launcher-timing.json').write_text(json.dumps({{'schema':'d01-v3-preflight-launcher','status':state,'mode':{mode!r},'notebook_cell_seconds':time.monotonic()-started,'preflight_gate_sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,'release_helper_sha256':{manifest['files']['d01_release_v3.py']!r},'policy_sha256':{POLICY_SHA!r},'scope':'whole cold notebook cell including decode/staging/validation/model loads; excludes Kaggle provisioning'}},indent=2))
'''
        ast.parse(code)
        notebook={"nbformat":4,"nbformat_minor":4,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[
            {"cell_type":"markdown","metadata":{},"source":[f"# D01 v3 {mode} engineering preflight\nNo competition candidate or accuracy metric.\n"]},
            {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(keepends=True)}]}
        metadata=read(historic/"kernel-metadata.json"); metadata.update(id=f"taeyangg4/knee-d01-v3-{mode}",title=f"knee-d01-v3-{mode}",code_file=f"d01-v3-{mode}.ipynb")
        metadata["dataset_sources"]=list(dict.fromkeys(metadata["dataset_sources"]+["taeyangg4/knee-d01-codec-assets"]))
        dump(directory/metadata["code_file"],notebook); dump(directory/"kernel-metadata.json",metadata)
    review={"status":"STAGED_ONLY_NO_RUNTIME_GATE_OR_CANDIDATE","helper_sha256":sha(__file__),"policy_sha256":POLICY_SHA,
            "manifest_sha256":sha(bundle/"helper-manifest.json"),"reused_supervisor_sha256":SUPERVISOR_SHA,
            "stages":[{"mode":mode,"path":str((out/mode).resolve()),"max_session_seconds":1800,"max_two_t4_gpu_hours":1.0,
                "command_not_executed":f'.venv/Scripts/kaggle.exe kernels push -p "{(out/mode).resolve()}" --accelerator NvidiaTeslaT4 --timeout 1800'} for mode in ("public3","train100")],
            "runtime_plan":"only new train100 launcher full*13 + anchor100 full*13, minimum300s reserve;1800s reserve separate target",
            "original_anchor":"not copied or edited; eventual blend delegates unchanged original90/10 function"}
    dump(out/"stage-review.json",review);return review


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    q=sub.add_parser("stage");q.add_argument("--out",required=True)
    q=sub.add_parser("child")
    for name in ("base","assets","competition","split","ids-path","codec-env","work"):q.add_argument("--"+name,required=True)
    q.add_argument("--public-reference",action="store_true")
    args=vars(p.parse_args());action=args.pop("action")
    if action=="stage":print(json.dumps(stage(**args),indent=2))
    else:child(**args)
