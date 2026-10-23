"""H-B-150 native320 PL fits v1.0.0 (2026-10-14).

Changelog: reuse native v2 pixels/forward and v3 accumulation/readouts; EXPOSED
f0 targets, resume journals, bounded Windows publication, full-shape CUDA smoke.
CPU engineering uses real native pixels and a declared tiny model, no label rows.
"""
import argparse
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from training.arms import native_resolution_v2 as native
from training.arms import mil_screen_hooks_v3 as h
from tools.hb147_progress_v1 import ProgressWriter
from tools.hb150_io_v1 import atomic_json, safe_replace, safe_unlink, JournalWriteError

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"


class Progress:
    """Count real completed commits; phase entry never advances progress."""
    def __init__(self, path=None, token=None):
        self.path = Path(path) if path else None
        self.token, self.sequence, self.completed_at = token, 0, None
        self.totals, self.phase, self.last_write = {}, "startup", 0.

    def phase_start(self, phase):
        self.phase = phase
        self._write("enter", {})

    def completed(self, phase, detail=None, *, throttle=False):
        self.phase = phase
        self.sequence += 1
        self.completed_at = time.monotonic()
        kind = {"training_step": "train_step", "validation": "validation",
                "panel_inference": "panel", "checkpoint_write": "checkpoint"}.get(phase, phase)
        self.totals[kind] = self.totals.get(kind, 0) + 1
        if not throttle or self.completed_at-self.last_write >= .5:
            self._write("completed", detail if isinstance(detail, dict) else {})

    def _write(self, event, detail):
        kind = {"training_step": "train_step", "validation": "validation",
                "panel_inference": "panel", "checkpoint_write": "checkpoint"}.get(self.phase, self.phase)
        detail = {**detail, "progress_kind": kind, "cumulative_units": self.totals.get(kind, 0)}
        if self.path is not None:
            atomic_json(self.path, dict(schema="hb147-progress/v1", token=self.token, pid=os.getpid(),
                sequence=self.sequence, completed_at=self.completed_at, phase=self.phase,
                event=event, detail=detail), critical=False)
        self.last_write = time.monotonic()


class TinyModel(h.torch.nn.Module):
    """Engineering only: exercise 320px real IO, accumulation, reload and TTA."""
    def __init__(self):
        super().__init__()
        self.head = h.torch.nn.Linear(3, 12)

    def forward(self, x):
        return self.head(x.mean(dim=(1, 3, 4)))


def read_config(path):
    path = Path(path)
    c = json.loads(path.read_text(encoding="utf-8"))
    h.need(c.get("schema") == "hb150-config/v1" and c["arm"] in ("global", "fill")
           and c["seed"] in (42, 43), "unregistered fit member")
    h.need((c["resolution"], c["batch"], c["microbatch"], c["epochs"], c["train_k"])
           == (320, 8, 2, 20, 12), "scientific recipe differs")
    h.need(c["exposure"] == "EXPOSED" and c["clean_oof"] is False, "EXPOSED contract required")
    h.need(c["corpus"] == native.CORPUS, "native448 cache required")
    for name, digest in c.get("pins", c.get("inputs", {})).items():
        h.need(h.sha(ROOT/name) == digest, "config input hash mismatch: " + name)
    return c


def save_epoch(path, *, epoch, model, opt, sched, scaler, generator, history, counters, identity, device, progress):
    payload = dict(schema="mil-epoch/v1", identity=identity, epoch=epoch,
        model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
        scaler=scaler.state_dict(), generator=generator.get_state(), history=history,
        counters=counters, python_rng=h.random.getstate(), numpy_rng=h.np.random.get_state(),
        torch_rng=h.torch.get_rng_state(), cuda_rng=h.torch.cuda.get_rng_state_all() if device == "cuda" else None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    progress.phase_start("checkpoint_write")
    try:
        with temp.open("wb") as stream:
            h.torch.save(payload, ProgressWriter(stream, progress=progress))
        safe_replace(temp, path)
        atomic_json(path.with_suffix(".receipt.json"), dict(schema="mil-epoch-receipt/v1",
            sha256=h.sha(path), identity=identity, epoch=epoch))
    except OSError as exc:
        atomic_json(path.with_suffix(".failure.json"), dict(status="FAILED", reason="persistent_journal_failure",
                                                         error=type(exc).__name__), critical=False)
        raise
    progress.completed("checkpoint_write", {"epoch": epoch, "receipt_verified": True})


def export_prediction(path, ids, values, checkpoint, identity, *, k, tta, engineering):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    try:
        temp.write_bytes(h.life.prediction_csv_bytes(ids, values, h.TARGETS, id_column=h.UID))
        if not safe_replace(temp, path, critical=False):
            return False
        return atomic_json(path.with_suffix(".receipt.json"), dict(csv_sha256=h.sha(path),
            checkpoint_sha256=h.sha(checkpoint), identity=identity, k=k, tta=tta,
            engineering_only=engineering, uid_sha256=h.uid_hash(ids), rows=len(ids),
            exposure="EXPOSED", clean_oof=False), critical=False)
    except OSError:
        return False
    finally:
        safe_unlink(temp)


def scientific_targets(c):
    """Admission already pins PL bytes; consume that table without re-opening its sources."""
    from tools import hb146_pl_labels_v111 as pl
    labels = pl.table(ROOT/c["label_path"])
    pl.aligned(labels, c["train"])
    return labels


def fit(config_path, out, *, device="cpu", smoke=True, review=None, resume=None):
    c = read_config(config_path)
    h.need(smoke or device == "cuda", "scientific fits require CUDA")
    if not smoke:
        from tools.hb150_build_v1 import require_fit
        require_fit(config_path, review)
    h.need(device == "cpu" or (os.environ.get("HB147_TOKEN") and os.environ.get("HB147_HEARTBEAT")),
           "CUDA execution requires watchdog supervision")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not smoke and (out/"final.pt").exists():
        status = json.loads((out/"heartbeat.json").read_text()) if (out/"heartbeat.json").exists() else {}
        h.need(resume is not None and status.get("status") != "complete",
               "scientific final checkpoint already exists; only explicit partial resume allowed")
    progress = Progress(os.environ.get("HB147_HEARTBEAT"), os.environ.get("HB147_TOKEN"))
    h.PROGRESS = progress
    progress.phase_start("startup")
    atomic_json(out/"heartbeat.json", dict(status="running", exposure="EXPOSED", clean_oof=False), critical=False)
    h.torch.set_num_threads(2)
    amp = "none" if device == "cpu" else "fp16"
    runtime = h.configure_runtime(amp, device, policy="strict")
    h.seed_all(c["seed"])
    if device == "cuda":
        gpu_name = h.torch.cuda.get_device_name(0)
        h.need(gpu_name == c["gpu_name"], "matched H146 comparison requires exact registered GPU")
        capacity = h.torch.cuda.get_device_properties(0).total_memory
        h.torch.cuda.set_per_process_memory_fraction(min(1., 10*(1<<30)/capacity))
        h.torch.cuda.reset_peak_memory_stats()
    else:
        gpu_name = "cpu"
    corpus = h.common.Corpus(ROOT/c["corpus"])
    h.need(corpus.contract == native.CONTRACT, "native448 corpus contract differs")
    train = c["train"][:2 if device == "cpu" else 16] if smoke else c["train"]
    val = c["val"][:1 if device == "cpu" else 16] if smoke else c["val"]
    gold = c["gold"][:1 if device == "cpu" else 16] if smoke else c["gold"]
    h.verify_shards(corpus, list(dict.fromkeys(train+val+gold)))
    if smoke:
        labels = {uid:h.np.full(12, .5, h.np.float32) for uid in train+val+gold}
    else:
        labels = scientific_targets(c)
    labels.update({uid:h.np.zeros(12, h.np.float32) for uid in val+gold})
    arm = native.NativeArm()
    ds = native.NativeWindows(h.MILWindows(ROOT/c["corpus"], train, labels, 12, True, c["seed"]), 320)
    batch = 2 if device == "cpu" else 8
    generator = h.torch.Generator().manual_seed(c["seed"])
    loader = h.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=0,
                          generator=generator, collate_fn=arm.collate_fn, pin_memory=device == "cuda")
    epochs = 1 if smoke else 20
    steps = len(loader)*epochs
    h.need(steps > 0 and (smoke or steps == c["optimizer_steps"]), "registered update count differs")
    model = TinyModel() if device == "cpu" else h.build_model(c["recipe"]["model"], ROOT)
    model = model.to(memory_format=h.torch.channels_last).to(device)
    opt, sched, scaler = h.optimizer_stack(model, steps, device, amp)
    identity = dict(version=VERSION, experiment="H-B-150", arm=c["arm"], seed=c["seed"],
        config_sha256=h.sha(config_path), registration_sha256=h.sha(ROOT/c["registration"]),
        prepare_sha256=None if smoke else h.sha(Path(config_path).parent.parent/f'prepared-{c["arm"]}.json'),
        labels_sha256=None if smoke else h.sha(ROOT/c["label_path"]), model=c["recipe"]["model"],
        weights=c["weights"], epochs=epochs, resolution=320, image_size=320,
        smoke=smoke, engineering_only=smoke, exposed_exploratory=True,
        exposure="EXPOSED", clean_oof=False, status="EXPOSED_EXPLORATORY",
        runtime=runtime, gpu_name=gpu_name, batch=batch, microbatch=2, fold=0,
        train_sha256=h.uid_hash(c["train"]), val_sha256=h.uid_hash(c["val"]))
    start_epoch, history, counters = 0, [], {"updates":0, "skips":0}
    if resume is not None:
        start_epoch, history, counters = h.restore_epoch(resume, model=model, opt=opt, sched=sched,
            scaler=scaler, generator=generator, identity=identity, device=device, loader=loader)
        h.need(0 <= start_epoch <= epochs and counters["updates"] == start_epoch*len(loader), "resume updates differ")
    start = time.monotonic()
    try:
        for epoch in range(start_epoch, epochs):
            ds.set_epoch(epoch)
            train_started = time.monotonic()
            record = h.train_epoch_accum(arm, model, loader, opt, sched, scaler,
                h.torch.tensor(c["weights"]), device, amp, 0 if smoke else max(1,int(steps*.01)), counters)
            record["wall_seconds"] = time.monotonic() - train_started
            history.append({**record, "epoch":epoch})
            save_epoch(out/"epochs"/f"epoch-{epoch:03d}.pt", epoch=epoch, model=model,opt=opt,
                sched=sched,scaler=scaler,generator=generator,history=history,counters=counters,
                identity=identity,device=device,progress=progress)
        h.need(counters["updates"] == steps and (not smoke or counters["skips"] == 0), "incomplete fixed-final fit")
        checkpoint = out/"final.pt"
        temp = out/"final.tmp"
        progress.phase_start("checkpoint_write")
        with temp.open("wb") as stream:
            h.torch.save(dict(model=model.state_dict(), identity=identity, epoch=epochs-1),
                         ProgressWriter(stream, progress=progress))
        safe_replace(temp, checkpoint)
        state = h.torch.load(checkpoint, map_location="cpu", weights_only=True)
        h.strict_load(model, state["model"])
        del state, opt, sched, scaler
        progress.completed("checkpoint_reload")
        durations, batches_by_name, export_failures = {}, {}, []
        for name, ids, k, tta in (("val_k24",val,24,False),("val_k36",val,36,False),
                                  ("val",val,36,True),("gold",gold,36,True)):
            eval_ds = native.NativeWindows(h.MILWindows(ROOT/c["corpus"],ids,labels,k,False,c["seed"]),320)
            eval_loader = h.DataLoader(eval_ds,batch_size=1 if device == "cpu" else 4,
                shuffle=False,num_workers=0,collate_fn=arm.collate_fn)
            then = time.monotonic()
            values, ordered, batches = h.evaluate_with(arm,model,eval_loader,device,amp,corpus,tta=tta)
            durations[name] = time.monotonic()-then
            batches_by_name[name] = batches
            h.need(ordered == ids, "prediction UID order differs")
            if not export_prediction(out/(name+".csv"),ids,values,checkpoint,identity,k=k,tta=tta,engineering=smoke):
                export_failures.append(name)
        healthy_rates = dict(train_step=sum(record["batches"] for record in history)
                            /sum(record["wall_seconds"] for record in history),
            validation=min(batches_by_name[name]/durations[name] for name in durations))
        healthy_rates["panel"] = healthy_rates["validation"]
        peak = h.torch.cuda.max_memory_reserved() if device == "cuda" else 0
        h.need(device == "cpu" or peak <= 10*(1<<30), "10 GiB full-shape gate failed")
        result = dict(version=VERSION, passed=not export_failures, status="PASS" if not export_failures else "EXPORT_FAILED",
            arm=c["arm"],seed=c["seed"], gpu_name=gpu_name, batch=batch, val_batch=1 if device == "cpu" else 4,
            optimizer_steps=counters["updates"]-counters["skips"], validation_shapes=["k24","k36","k36flip"],
            config_sha256=h.sha(config_path), engineering_only=smoke,smoke=smoke,device=device,
            full_shape=device == "cuda", tiny_model=device == "cpu",peak_reserved_bytes=peak,
            healthy_rates=healthy_rates, cuda_initialized=h.torch.cuda.is_initialized() if device == "cpu" else True,
            cpu_cuda_initialized=h.torch.cuda.is_initialized() if device == "cpu" else None,
            counters=counters,steps=steps,seconds=time.monotonic()-start,runtime=runtime,
            checkpoint_sha256=h.sha(checkpoint),identity=identity,history=history,
            validation_seconds=durations,export_failures=export_failures,exposure="EXPOSED",clean_oof=False)
        h.need(device != "cpu" or result["cuda_initialized"] is False, "CPU smoke initialized CUDA")
        if not atomic_json(out/"summary.json", result, critical=False):
            export_failures.append("summary")
            result.update(passed=False, status="EXPORT_FAILED")
        atomic_json(out/"heartbeat.json",dict(status="complete" if not export_failures else "export_failed",
            exposure="EXPOSED",clean_oof=False),critical=False)
        return result
    except BaseException as exc:
        atomic_json(out/"failure.json",dict(status="FAILED",error=type(exc).__name__,
            reason="persistent_journal_failure" if isinstance(exc, JournalWriteError) else "fit_failure"),critical=False)
        atomic_json(out/"heartbeat.json",dict(status="failed",error=type(exc).__name__),critical=False)
        raise
    finally:
        iterator = getattr(loader,"_iterator",None)
        if iterator is not None:
            iterator._shutdown_workers()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("smoke","fit"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu","cuda"),default="cpu")
    ap.add_argument("--review",type=Path)
    ap.add_argument("--resume",type=Path)
    args=ap.parse_args(argv)
    value=fit(args.config,args.out,device=args.device,smoke=args.command == "smoke",review=args.review,resume=args.resume)
    print(json.dumps({key:value[key] for key in ("passed","arm","seed","device","engineering_only","steps","seconds")}))
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
