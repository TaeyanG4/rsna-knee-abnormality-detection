"""H137 safety v1.1.0: measured 10 GB gate and progress/WDDM watchdog.

Changelog: cumulative cap retained; completed forward heartbeats, sustained
WDDM telemetry, isolated attempt logs and fail-closed measured admission.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import math

VERSION = "1.1.0"
LIMIT_BYTES = 10_000_000_000  # decimal GB: stricter than 10 GiB
WORK = Path(__file__).resolve().parents[1] / ".local/scratch/astra-b2/H-B-137/attempt-v1.1"


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def heartbeat():
    path = os.environ.get("HB137_PROGRESS")
    if path:
        atomic(path, {"completed_forward_ns": time.time_ns(), "pid": os.getpid()})


def telemetry():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    rows = [list(map(float, row.split(","))) for row in proc.stdout.strip().splitlines()]
    if len(rows) != 1 or not all(math.isfinite(x) for x in rows[0]):
        raise ValueError("invalid single-GPU telemetry")
    return dict(zip(("util", "watts", "used_mib", "total_mib"), rows[0]))


class Watchdog:
    def __init__(self, now, idle=900, stall=120):
        if not all(math.isfinite(x) and x > 0 for x in (idle, stall)):
            raise ValueError("invalid watchdog duration")
        self.last = now
        self.token = None
        self.since = None
        self.idle, self.stall = idle, stall

    def check(self, now, token, gpu=None):
        if token is not None and token != self.token:
            self.token, self.last = token, now
        fallback = gpu is not None and gpu["util"] >= 98 and 0 <= gpu["watts"] < 90 and (
            gpu["total_mib"] > 0 and gpu["used_mib"] / gpu["total_mib"] >= .94)
        self.since = (now if self.since is None else self.since) if fallback else None
        if self.since is not None and now - self.since >= self.stall:
            return "WDDM_FALLBACK_STALL"
        if now - self.last >= self.idle:
            return "NO_STEP_PROGRESS"
        return None


def run(command, ledger=None, limit=10500, *, idle=900, stall=120, poll=5, sample=telemetry):
    from tools.hb137_screen_v1 import ROOT, life, need, hooks
    ledger = Path(ledger or WORK / "gpu-budget-v1.1.json")
    doc = hooks.read(ledger) if ledger.exists() else dict(version=VERSION, seconds=0., running=False, attempts=[])
    need(doc["version"] == VERSION and not doc["running"], "budget interrupted; B review required")
    need(math.isfinite(doc["seconds"]) and doc["seconds"] >= 0 and math.isfinite(limit) and limit > 0,
         "invalid budget accounting")
    remaining = limit - doc["seconds"]
    need(remaining > 0 and command, "GPU screen budget exhausted")
    attempt = ledger.parent / "attempts" / f"attempt-{len(doc['attempts']) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    progress = attempt / "progress.json"
    doc.update(running=True)
    atomic(ledger, doc)
    start, child, code, reason = time.monotonic(), None, -1, None
    watchdog = Watchdog(start, idle, stall)
    try:
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with (attempt / "child.log").open("wb") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     env={**os.environ, "HB137_PROGRESS": str(progress)}, **kwargs)
            while True:
                now = time.monotonic()
                timeout = min(poll, max(.001, remaining - (now - start)))
                try:
                    code = child.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    token = json.loads(progress.read_text())["completed_forward_ns"] if progress.exists() else None
                    try:
                        gpu = sample()
                        atomic(attempt / "telemetry.json", gpu)
                    except (OSError, ValueError, subprocess.SubprocessError):
                        gpu = None  # idle + total cap remain active if nvidia-smi fails
                    reason = "TOTAL_RUNTIME_CAP" if now - start >= remaining else watchdog.check(now, token, gpu)
                    if reason:
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=True, capture_output=True)
                        else:
                            import signal
                            os.killpg(child.pid, signal.SIGKILL)
                        code = child.wait()
                        break
    finally:
        elapsed = time.monotonic() - start
        doc["seconds"] += elapsed
        doc["running"] = child is not None and child.poll() is None
        record = dict(command=command, seconds=elapsed, code=code, reason=reason,
                      timed_out=reason == "TOTAL_RUNTIME_CAP", partial_not_scoreable=code != 0,
                      attempt_dir=str(attempt), running=doc["running"])
        # Preserve references to runner-owned partial outputs; never move/delete them.
        if "--out" in command:
            record["partial_output"] = command[command.index("--out") + 1]
        atomic(attempt / "attempt-receipt.json", record)
        doc["attempts"].append(record)
        atomic(ledger, doc)
    need(code == 0 and reason is None, "child failed or watchdog expired; partial run is not scoreable")
    return dict(status="BUDGETED_COMMAND_PASS", seconds=elapsed, total_seconds=doc["seconds"])


def bindings():
    from tools import mil_screen_arm_build_v1 as b
    from training.arms import hb137_spatial_adapter_v1_1 as arm
    return {p: b.sha(b.ROOT / p) for p in arm.ADAPTER.sources}


def admission(receipt=None):
    from tools import mil_screen_arm_build_v1 as b
    r = b.read(receipt or WORK / "memory-receipt-v1.1.json")
    b.need(r["status"] == "PASS" and r["device"] == "cuda" and r["limit_bytes"] == LIMIT_BYTES
           and r["sources"] == bindings(), "missing/failed/stale CUDA memory gate")
    expected = {(k, p) for k in ("head", "adapter") for p in ("train", "validation", "panel")} | {("parent", "panel")}
    b.need({(v["kind"], v["phase"]) for v in r["measurements"]} == expected, "incomplete memory phases")
    for v in r["measurements"]:
        b.need(v["steps"] >= 2 and 0 < v["allocated_bytes"] <= v["reserved_bytes"] <= LIMIT_BYTES,
               "CUDA peak exceeds memory gate")
        b.need(v["shape"] == ([8, 12, 3, 224, 224] if v["phase"] == "train" else [4, 36, 3, 224, 224]),
               "memory probe shape mismatch")
    b.life.check_target_observation(b.hooks.control_spec(42), r["observation"])
    return r


def memory_smoke(device="cuda"):
    """Real cached images, same hooked model/BCE/optimizer/evaluation path; no scored output."""
    import torch
    import cv2
    import numpy as np
    from tools import mil_screen_arm_build_v1 as b
    from tools import hb122_screen_v1 as screen
    from training.arms import hb137_spatial_adapter_v1_1 as arm
    h = b.hooks
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    amp = "fp16" if device == "cuda" else "none"
    h.configure_runtime(amp, device, policy="strict")
    r = dict(version=VERSION, status="RUNNING", device=device, limit_bytes=LIMIT_BYTES,
             sources=bindings(), measurements=[], effective_batch=8, micro_batch=8, accumulation=1)
    path = WORK / ("memory-receipt-v1.1.json" if device == "cuda" else "cpu-memory-path-v1.1.json")
    if path.exists():
        raise ValueError("memory receipt exists; preserve it and explicitly version the next probe")
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        r.update(device_name=torch.cuda.get_device_name(), total_bytes=total, free_bytes=free,
                 observation=h.observation(device),
                 desktop_note="B correction: 16376 MiB total, about 5.1 GB desktop use; about 11 GB usable")
        b.life.check_target_observation(h.control_spec(42), r["observation"])
    atomic(path, r)
    try:
        spec = h.control_spec(42)
        corpus, train, val, labels, weights = h.validate_data_with(h.IDENTITY, spec, b.ROOT, b.CORPUS)
        _, panel_ids, _, _ = screen.panel_context()
        panel = h.common.Corpus(b.ROOT / b.PANEL_CACHE)
        for kind in ("head", "adapter", "parent"):
            h.seed_all(42)
            use_arm = getattr(arm, kind.upper()) if kind != "parent" else h.IDENTITY
            if kind == "parent":
                model = h.build_model(spec.document["config"]["model"], b.ROOT)
                snap = screen.snapshot("control", 42)
                checkpoint = torch.load(b.ROOT / snap["checkpoint"], map_location="cpu", weights_only=False)
                arm.check_parent_checkpoint(checkpoint, spec, snap)
                h.strict_load(model, checkpoint["model"])
                del checkpoint
            else:
                _, record, _ = b.state(WORK / kind)
                member = b.spec_for(record, 42)
                model = use_arm.attach(h.build_model(member.document["config"]["model"], b.ROOT), member, b.ROOT)
            model.to(device)
            opt, sched, scaler = h.optimizer_stack(model, 3, device, amp)
            for phase in (("panel",) if kind == "parent" else ("train", "validation", "panel")):
                training = phase == "train"
                count = (8 if training else 4) if device == "cuda" else (2 if training else 1)
                k = 12 if training else 36
                ids = (train if training else panel_ids if phase == "panel" else val)[:count]
                cache, corp = (b.ROOT / b.PANEL_CACHE, panel) if phase == "panel" else (b.CORPUS, corpus)
                h.verify_shards(corp, ids)
                yy = {u: labels[u] if training else np.zeros(12, np.float32) for u in ids}
                ds = h.MILWindows(cache, ids, yy, k, training, 42)
                loader = h.DataLoader(ds, batch_size=count, num_workers=0, collate_fn=use_arm.collate_fn)
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                if training:
                    counters = dict(updates=0, skips=0)
                    for step in range(3):
                        ds.set_epoch(step)
                        h.train_epoch_with(use_arm, model, loader, opt, sched, scaler, torch.from_numpy(weights),
                                           device, amp, 0, counters)
                    steps = 3
                else:
                    for _ in range(2):
                        h.evaluate_with(use_arm, model, loader, device, amp, corp, tta=True)
                    steps = 2
                if device == "cuda":
                    torch.cuda.synchronize()
                m = dict(kind=kind, phase=phase, shape=[count, k, 3, 224, 224], steps=steps,
                         allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                         reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None)
                r["measurements"].append(m)
                atomic(path, r)
                print(json.dumps(m), flush=True)
                if device == "cuda" and m["reserved_bytes"] > LIMIT_BYTES:
                    raise ValueError("memory limit exceeded; matched microbatch/accumulation revision required before admission")
            del model, opt, sched, scaler
            if device == "cuda":
                torch.cuda.empty_cache()
        r["status"] = "PASS" if device == "cuda" else "CPU_ENGINEERING_PASS"
    except BaseException as exc:
        r.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic(path, r)
    return r
