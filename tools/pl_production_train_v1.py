"""PL production operational wrapper v1.1.0 (2026-10-14).

Changelog 1.1.0: configurable bounded, seeded persistent production loaders.
Changelog 1.0.0: existing A4/A5 recipe/export with completed-step progress and pinned
epoch-boundary restart journals. The frozen scientific hooks are reused intact.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import vnext_r7_pl_train_v110 as base
from tools import pl_production_watchdog_v1 as watchdog
from tools.hb150_io_v1 import safe_replace, atomic_json, _retry

VERSION="1.1.0"
EXTRA_SOURCES=[ROOT/"tools/pl_production_train_v1.py", ROOT/"tools/pl_production_targets_v1.py",
               ROOT/"tools/pl_production_loader_v1.py", ROOT/"tools/pl_production_watchdog_v1.py", ROOT/"tools/hb150_watchdog_v1.py",
               ROOT/"tools/hb150_io_v1.py", ROOT/"tools/hb147_progress_v1.py"]


@contextmanager
def adapters():
    from tools.pl_production_targets_v1 import load_materialized
    old_loader,old_sources=base.target_loader,base.SOURCES
    base.target_loader=load_materialized
    base.SOURCES=[*old_sources,*EXTRA_SOURCES]
    try:
        yield
    finally:
        base.target_loader,base.SOURCES=old_loader,old_sources


class Counters(dict):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._initial_updates=self.get("updates",0)
        self._initial_skips=self.get("skips",0)

    def __setitem__(self,key,value):
        super().__setitem__(key,value)
        if key=="updates":
            watchdog.PROGRESS.completed("training_step",dict(update=value-self._initial_updates,
                                        skips=self.get("skips",0)-self._initial_skips))


def identity(spec,go):
    return dict(schema="pl-production-resume/v1",spec_sha256=spec["spec_sha256"],
                go_sha256=None if go is None else go["go"]["sha256"], member=spec["member"])


def validate_resume(record, expected, steps_per_epoch, epochs):
    base.p.need(record.get("identity")==expected,"resume spec/GO/member identity changed")
    epoch=record.get("next_epoch")
    base.p.need(isinstance(epoch,int) and 0<=epoch<=epochs,"resume epoch range")
    counters=record["counters"]
    base.p.need(counters["updates"]==epoch*steps_per_epoch and 0<=counters["skips"]<=counters["updates"],
                "resume epoch/counters mismatch")
    return epoch


def save_resume(path,record):
    import torch
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    watchdog.PROGRESS.phase_start("checkpoint_write")
    try:
        with _retry(lambda:temporary.open("wb")) as stream:
            writer=watchdog.ProgressWriter(stream,watchdog.PROGRESS)
            torch.save(record,writer)
            writer.flush()
        safe_replace(temporary,path,critical=True)
    except Exception as exc:
        failure=dict(schema="pl-production-journal-failure/v1",status="FAILED",
                     reason="persistent_journal_failure",error=type(exc).__name__,path=str(path))
        if not atomic_json(Path(str(path)+".io-failure.json"),failure,critical=False):
            print(json.dumps(failure),flush=True)
        raise
    watchdog.PROGRESS.completed("checkpoint_write",dict(epoch=record["next_epoch"],journal_committed=True))


@contextmanager
def export_progress(torch):
    """Instrument A5's existing serialization without changing its payload."""
    original=torch.save
    def save(obj,stream,*args,**kwargs):
        if hasattr(stream,"write"):
            writer=watchdog.ProgressWriter(stream,watchdog.PROGRESS)
            result=original(obj,writer,*args,**kwargs)
            writer.flush()
            return result
        return original(obj,stream,*args,**kwargs)
    torch.save=save
    try:
        yield
    finally:
        torch.save=original


def mil_fit(spec,target,labels,output,*,device,smoke,go,resume=None,workers=None,loader_profile="local",prefetch_factor=2,ram_budget_gb=40):
    from tools import pl_production_loader_v1 as loading
    from training.arms.hb146_pl_v1 import Arm
    from training.arms import mil_screen_hooks_v1 as h
    output=Path(output)
    # Never overwrite an already exported or partially exported scientific fold.
    base.p.need(not (output/"training-receipt.json").exists()
                and not (output/("smoke.pt" if smoke else "model_best.pt")).exists(),
                "existing fold export preserved; validate/skip it or use a named recovery output")
    watchdog.PROGRESS.phase_start("startup_validation")
    amp="none" if device=="cpu" else "fp16"
    runtime=h.configure_runtime(amp,device,policy="strict")
    seed=spec["member"]["seed"]
    h.seed_all(seed)
    ids=target["train_ids"][:2] if smoke else target["train_ids"]
    corpus=h.common.Corpus(spec["corpus"])
    h.verify_shards(corpus,ids)
    watchdog.PROGRESS.completed("startup_io","shards verified")
    arm=Arm(labels)
    zeros={u:h.np.zeros(12,h.np.float32) for u in ids}
    ds=arm.dataset(h.MILWindows(spec["corpus"],ids,zeros,12,True,seed),train=True)
    loader,gen,loader_config=loading.make_loader(ds,arm.collate_fn,seed,
        workers=workers,profile=loader_profile,batch_size=1 if smoke else 8,
        prefetch_factor=prefetch_factor,ram_budget_gb=ram_budget_gb,device=device)
    runtime["production_loader"]=loader_config
    epochs=1 if smoke else 20
    steps=len(loader)*epochs
    base.p.need(steps==(2 if smoke else spec["recipe"]["updates"]["steps"]),"registered update count")
    model=h.build_model(spec["recipe"]["model"],ROOT).to(device)
    opt,sched,scaler=h.optimizer_stack(model,steps,device,amp)
    watchdog.PROGRESS.completed("startup_model","model and optimizer loaded")
    counters=Counters(updates=0,skips=0)
    start_epoch=0
    bound=dict(identity(spec,go),loader_version=loading.VERSION,shuffle="seed-plus-epoch")
    journal=output/"resume.pt"
    if resume is not None:
        record=h.torch.load(resume,map_location=device,weights_only=False)
        start_epoch=validate_resume(record,bound,len(loader),epochs)
        model.load_state_dict(record["model"],strict=True)
        opt.load_state_dict(record["optimizer"])
        sched.load_state_dict(record["scheduler"])
        scaler.load_state_dict(record["scaler"])
        counters=Counters(record["counters"])
        random.setstate(record["python_rng"])
        h.np.random.set_state(record["numpy_rng"])
        h.torch.set_rng_state(record["torch_rng"].cpu())
        gen.set_state(record["loader_rng"].cpu())
        if device=="cuda":
            h.torch.cuda.set_rng_state_all([v.cpu() for v in record["cuda_rng"]])
        watchdog.PROGRESS.completed("startup_resume",dict(epoch=start_epoch))
    elif journal.exists():
        raise ValueError("existing resume journal preserved; pass --resume explicitly")
    with loading.managed_loader(loader):
        for epoch in range(start_epoch,epochs):
            ds.set_epoch(epoch)
            gen.manual_seed(seed+epoch)
            watchdog.PROGRESS.phase_start("training_step")
            h.train_epoch_with(arm,model,loader,opt,sched,scaler,h.torch.tensor(target["weights"]),
                              device,amp,0 if smoke else max(1,int(steps*.01)),counters)
            record=dict(identity=bound,next_epoch=epoch+1,model=model.state_dict(),
                        optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),
                        counters=dict(counters),python_rng=random.getstate(),numpy_rng=h.np.random.get_state(),
                        torch_rng=h.torch.get_rng_state(),loader_rng=gen.get_state(),
                        cuda_rng=h.torch.cuda.get_rng_state_all() if device=="cuda" else [])
            save_resume(journal,record)
    base.p.need(counters["updates"]==steps,"incomplete production fit")
    watchdog.PROGRESS.phase_start("checkpoint_write")
    with export_progress(h.torch):
        result=base.export(spec,target,model,output,smoke=smoke,counters=dict(counters),runtime=runtime,go=go)
    watchdog.PROGRESS.completed("checkpoint_write","fixed-final export verified")
    base.p.need(device!="cpu" or not h.torch.cuda.is_initialized(),"CPU initialized CUDA")
    return result


def fit(path,output,*,device="cpu",smoke=True,go=None,resume=None,workers=None,prefetch_factor=2,ram_budget_gb=40):
    base.p.need(device in ("cpu","cuda") and (smoke or device=="cuda"),"full fitting requires CUDA")
    with adapters():
        spec,target,labels=base.validate(path)
        base.p.need(spec["member"]["kind"]=="MIL","production launch is MIL only")
        admission=base.admit(path,go) if not smoke else None
        base.p.need(smoke or admission["status"]=="ADMITTED","fit not admitted")
        return mil_fit(spec,target,labels,output,device=device,smoke=smoke,go=admission,resume=resume,workers=workers,prefetch_factor=prefetch_factor,ram_budget_gb=ram_budget_gb)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("prepare","validate","admit","fit"))
    for name in ("stage","targets","output","spec","go","resume"):
        ap.add_argument("--"+name,type=Path)
    ap.add_argument("--member")
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    ap.add_argument("--full",action="store_true")
    ap.add_argument("--workers",type=int)
    ap.add_argument("--prefetch-factor",type=int,default=2)
    ap.add_argument("--ram-budget-gb",type=float,default=40)
    args=ap.parse_args(argv)
    with adapters():
        if args.command=="prepare":
            result=base.prepare(args.stage,args.targets,args.output,args.member)
        elif args.command=="validate":
            spec,target,_=base.validate(args.spec)
            result=dict(status="PASS",spec_sha256=spec["spec_sha256"],train=len(target["train_ids"]),val=len(target["val_ids"]))
        elif args.command=="admit":
            result=base.admit(args.spec,args.go)
            base.p.need(result["status"]=="ADMITTED","H-B-146 GO absent")
        else:
            result=fit(args.spec,args.output,device=args.device,smoke=not args.full,go=args.go,resume=args.resume,workers=args.workers,prefetch_factor=args.prefetch_factor,ram_budget_gb=args.ram_budget_gb)
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
