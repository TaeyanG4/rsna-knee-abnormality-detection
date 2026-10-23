"""v1.0.0 fullbag numerical admission and real-cache CPU smoke.

Predeclared tolerance: fp32 atol=2e-5/rtol=2e-4; bf16 atol=2e-3/rtol=.03.
Reference uses ordinary autograd over identical microbatches (no detach).
Tiny GPU check is explicit, never invoked by the CPU CLI.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools import l1_rco_k1_fullbag_replay_v1 as replay

TOLERANCES = {False: dict(atol=2e-5, rtol=2e-4), True: dict(atol=2e-3, rtol=.03)}


class DropPath(torch.nn.Module):
    def forward(self, x):
        if self.training:
            return x * (torch.rand((len(x),1), device=x.device) >= .2) / .8
        return x


class TinyEncoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                         torch.nn.Linear(3,8), torch.nn.Dropout(.2), DropPath())


class TinyNet(torch.nn.Module):
    """Stochastic smoke encoder with the exact replay/head API."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.enc = TinyEncoder()
        self.proj = torch.nn.Linear(8,12)
        self.drop = torch.nn.Dropout(.2)
        self.bias = torch.nn.Parameter(torch.zeros(12))
        self.register_buffer("mean", torch.zeros(1,3,1,1))
        self.register_buffer("std", torch.ones(1,3,1,1))
        self.spatial = self.has_bn = False
    def head(self, features, meta, mask):
        rows = features.new_zeros((*mask.shape, features.shape[-1]))
        rows[mask] = features
        pooled = (rows*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        return self.proj(self.drop(pooled))+self.bias
    def forward(self, img, meta, mask, **kwargs):
        return self.head(replay.encode(self, img[mask]), meta, mask)
    def deploy_state_dict(self):
        return {n:p.detach().cpu() for n,p in self.state_dict().items()}


def weighted_bce(z, target, weight):
    value = torch.nn.functional.binary_cross_entropy_with_logits(z, target, reduction="none")*weight
    den = weight.sum(0)
    return (value.sum(0)[den>0]/den[den>0]).mean()


def compare_model(model, chunks, meta, mask, *, amp=False):
    """All named gradients, one AdamW/scheduler/EMA update, and next RNG."""
    device = meta.device
    initial = copy.deepcopy(model.state_dict())
    initial_rng = replay.rng_state(device)
    target = torch.full((len(mask),12), .37, device=device)
    weight = torch.ones_like(target)
    weight[:,0] = 0  # zero-denominator target is excluded by original BCE.
    outputs = []
    for use_replay in (False, True):
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        replay.set_rng(initial_rng, device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.02)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1/(step+1))
        loss = replay.backward(model, chunks, meta, mask,
                               lambda z: weighted_bce(z, target, weight),
                               replay=use_replay, amp=amp)
        gradients = {n: None if p.grad is None else p.grad.detach().cpu().clone()
                     for n,p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        sched.step()
        ema = {n: initial[n].to(p.device)*.2+p.detach()*.8 for n,p in model.named_parameters()}
        outputs.append(dict(loss=loss.cpu(), gradients=gradients, state=copy.deepcopy(model.state_dict()),
                            ema=ema, optimizer=copy.deepcopy(opt.state_dict()),
                            scheduler=sched.state_dict(), rng=replay.rng_state(device)))
        del opt, sched
    a, b = outputs
    tolerance = TOLERANCES[amp]
    maximum = 0.
    for name, grad in a["gradients"].items():
        other = b["gradients"][name]
        if grad is None:
            assert other is None, name
        else:
            assert other is not None, name
            torch.testing.assert_close(grad, other, **tolerance, msg=name)
            maximum = max(maximum, float((grad-other).abs().max()))
            if not torch.count_nonzero(grad):
                assert not torch.count_nonzero(other), name
    for field in ("loss", "state", "ema", "optimizer"):
        torch.testing.assert_close(a[field], b[field], **tolerance)
    assert a["scheduler"] == b["scheduler"]
    assert torch.equal(a["rng"][0], b["rng"][0])
    if device.type == "cuda":
        assert torch.equal(a["rng"][1], b["rng"][1])
    return dict(status="PASS", gradient_tensors=len(a["gradients"]),
                max_gradient_abs_error=maximum, tolerance=tolerance,
                optimizer_scheduler_ema_rng=True)


def numerical(device="cpu"):
    from tools import l1_rco_k1_fullbag_v1 as f
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    dev = torch.device(device)
    torch.manual_seed(1052)
    if device == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.reset_peak_memory_stats()
    cases = []
    for amp in (False, True):
        model = TinyNet().to(dev).train()
        for lengths in ((1,3), (4,2)):
            mask = torch.arange(max(lengths))[None,:] < torch.tensor(lengths)[:,None]
            images = torch.randint(0,256,(int(mask.sum()),3,16,16),dtype=torch.uint8).to(dev)
            def chunks():
                for part in images.split(2):
                    yield part, len(part)
            cases.append(compare_model(model, chunks, torch.zeros(2,max(lengths),6,device=dev),
                                       mask.to(dev), amp=amp))
    ct = f.k.import_trainer()
    # Same ConvNeXt+spatial head+checkpointing; small spatial shape qualifies
    # arithmetic only. Full production crop is covered by CPU real smoke.
    model = ct.Net("convnext_small", str(ROOT/f.k.s.INIT), crop=64,
                   in_chans=3, spatial=True, extent=False).to(dev).train()
    model.enc.set_grad_checkpointing(True)
    # Nonzero query exercises sp_key/sp_pos gradients beyond initial zero query.
    with torch.no_grad():
        model.sp_q.normal_(0,.01)
    images = torch.randint(0,256,(3,3,64,64),dtype=torch.uint8).to(dev)
    mask = torch.tensor([[True,False],[True,True]],device=dev)
    def chunks():
        for part in images.split(2):
            yield part, len(part)
    for amp in (False, True):
        cases.append(compare_model(model, chunks, torch.zeros(2,2,6,device=dev), mask, amp=amp))
    result = dict(status="PASS", version="1.0.0", device=device, actual_model=True,
                  amp=True, cases=cases, code_pins=f.code_pins(),
                  peak_reserved=torch.cuda.max_memory_reserved() if device=="cuda" else 0,
                  cuda_initialized=torch.cuda.is_initialized(),
                  scope="small-bag numerical equivalence; not full640 memory/throughput admission")
    f.k.write(f.WORK/f"numerical-{device}.json", result)
    return result


def cpu_smoke(reg, cache):
    from tools import l1_rco_k1_fullbag_v1 as f
    from tools.l1_rco_k1_foldstate_v1 import atomic_torch_save
    from tools.l1_rco_k1_foldsmoke_v1 import equal_state
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    k = f.k
    k.complete_cache(reg, cache)
    ct = k.import_trainer()
    work = f.WORK/"cpu"
    work.mkdir(parents=True, exist_ok=True)
    index = k.read(Path(cache)/"index.json")
    train = sorted(sorted(reg["train"], key=lambda u:index["done"][u][1])[:2])
    holdout = sorted(reg["holdout"], key=lambda u:index["done"][u][1])[:1]
    with (f.WORK/"targets.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r[k.s.UID] in train]
    target = work/"targets.csv"
    with target.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    real_net = ct.Net
    ct.Net = TinyNet

    def execute(out, stop=0, guard=None):
        argv = k.trainer_argv(cache, reg)[1:]
        for flag, value in {"--out":out, "--targets":target, "--workers":0,
                            "--epochs":3, "--batch":1}.items():
            argv[argv.index(flag)+1] = str(value)
        argv.remove("--grad-ckpt")
        i = argv.index("--l1-guard-constant"); del argv[i:i+2]
        if guard is not None:
            argv += ["--l1-guard-constant", str(guard)]
        argv += ["--l1-cpu-test","--l1-stop-steps",str(stop)]
        previous = sys.argv
        sys.argv = argv
        output = io.StringIO()
        code = 0
        try:
            with contextlib.redirect_stdout(output):
                ct.main()
        except SystemExit as exc:
            code = exc.code
        finally:
            sys.argv = previous
        out.mkdir(parents=True,exist_ok=True)
        (out/"smoke.log").write_text(output.getvalue(),encoding="utf-8")
        return code

    try:
        # Reset only known generated smoke artifacts, never production receipts.
        for name in ("base", "resume", "guard"):
            out = work/name
            for rel in ("ckpt/E_prior_s0_f0.pt","ckpt/best_E_prior_s0_f0.pt","ckpt/best.json",
                        "ckpt/epochs.json","collapse.json","fit-complete.json","models/E_prior_s0_f0.pt"):
                path = out/rel
                if path.exists():
                    for attempt in range(40):
                        try:
                            path.unlink()
                            break
                        except PermissionError:
                            if attempt == 39:
                                raise
                            time.sleep(.05)
        assert execute(work/"base") == 0
        assert execute(work/"resume", stop=1) == 3
        assert execute(work/"resume") == 0
        load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
        assert equal_state(load(work/"base/ckpt/E_prior_s0_f0.pt"),
                           load(work/"resume/ckpt/E_prior_s0_f0.pt"))
        assert execute(work/"guard", guard=1e-6) == 4
        pin = k.sha(work/"guard/ckpt/E_prior_s0_f0.pt")
        assert execute(work/"guard", guard=1e-6) == 4
        assert pin == k.sha(work/"guard/ckpt/E_prior_s0_f0.pt")
    finally:
        ct.Net = real_net
    # Real full-resolution cached images; retain exact complete candidates in
    # the loader, use only two windows for this explicitly CPU-test computation.
    ds = ct.Studies(str(cache),train,640,"train",9000,train_sampling="cover",
                    slices=3,phys_mm=3.5,slice_jitter=.25)
    loader = torch.utils.data.DataLoader(ds,batch_size=1,num_workers=1,collate_fn=ct.collate)
    batches = list(loader)
    img, meta, mask, _ = batches[0]
    images = img[mask][:2,...,16:368,16:368]
    small_meta = meta[:,:len(images)]
    small_mask = mask[:,:len(images)]
    model = real_net("convnext_small",str(ROOT/k.s.INIT),crop=352,
                     in_chans=3,spatial=True,extent=False).train()
    model.enc.set_grad_checkpointing(True)
    def chunks():
        for part in images.split(1):
            yield part, len(part)
    actual = compare_model(model, chunks, small_meta, small_mask)
    fullout = work/"real-net"
    atomic_torch_save(model.deploy_state_dict(),fullout/"models/E_prior_s0_f0.pt")
    del model
    served = k.inference(reg,cache,out=fullout,cpu=True,ids=holdout,limit_windows=2)
    assert served["cpu_test"] and served["holdout_rows"] == 1
    result = dict(status="PASS",contract_id=reg["contract_id"],cuda_initialized=False,
                  train=3190,holdout=977,worker_batches=len(batches),
                  exact_resume=True,collapse_no_retry=True,real_crop352=actual,
                  inference=dict(cpu_test=True,holdout_rows=1),
                  scope="real cache fullbag loader/augmentation/tiny trainer resume; real Net crop352 replay and serving; no production predictions")
    k.write(f.WORK/"cpu-smoke.json", result)
    return result


def qualify_binding():
    from tools import l1_rco_k1_fullbag_v1 as f
    reg = f.k.validate(f.k.read(f.WORK/"registration.json"))
    return dict(bindings=f.review_binding(reg), code_pins=f.code_pins(),
                protocol="4090 GPU0; after bound f3 SUCCESS; batch2 x640 crop352; two replay AdamW/EMA steps")


def fullshape(run_id):
    """B-only diagnostic, independently reviewed and owned; never auto-fit."""
    from tools import l1_rco_k1_fullbag_v1 as f
    from tools import job_runtime as runtime
    import psutil
    import signal
    k = f.k
    k.need(os.name != "nt", "fullshape qualification is Vast Linux only")
    scope = f.CANONICAL_OWNER
    k.need(Path(os.environ.get("KNEE_WORKSPACE_ROOT","")).resolve() == scope.resolve(),
           "canonical GPU0 owner required")
    spec = f.plan(run_id)
    predecessor = f.predecessor_success(spec)
    review = k.read(f.WORK/"REVIEW-QUALIFY.json")
    k.need(review["verdict"] == "PASS" and review["builder"] != review["reviewer"]
           and review["qualification"] == qualify_binding()
           and k.sha(ROOT/review["report"]) == review["report_sha256"],
           "independent fullshape qualification review required")
    tag = "DA052_K1_FULLBAG_QUALIFY"
    command = f.fullshape_command(run_id)
    with runtime.file_lock(runtime.LOGS/".gpu-claim.lock"):
        job = runtime.read_json(runtime.LOGS/tag/"job.json")
        parent = psutil.Process().parent()
        k.need(job["schema"] == "gpu-job/v2" and job["phase"] == "running"
               and job["tag"] == tag and job["command"] == command
               and job["child_identity"] == runtime.identity()
               and {n:job[n] for n in ("pid","create_time")} == runtime.identity(parent.pid)
               and Path(job["cwd"]).resolve() == scope.resolve()
               and job["command_sha256"] == runtime.command_digest(command, scope)
               and parent.cmdline()[-3:] == ["--_run-job",tag,job["run_id"]],
               "owned exact qualification command required")
        k.need(not k.active_ml_processes(parent.pid), "another Python ML GPU process exists")
    once = f.WORK/"fullshape-once.json"
    with runtime.file_lock(once.with_suffix(".lock"), blocking=False):
        k.need(not once.exists(), "qualification already attempted; no retry")
        runtime.atomic_json(once, dict(status="STARTED", run_id=job["run_id"], predecessor=predecessor))
    snapshot = k.gpu_snapshot()
    k.need("4090" in snapshot["name"], "4090 qualification required")
    reg = k.validate(k.read(f.WORK/"registration.json"))
    torch.set_num_threads(2)
    torch.manual_seed(1052)
    ct = k.import_trainer()
    def progress(phase):
        k.write(f.WORK/"qualification-status.json",
                dict(phase=phase, time=time.time(), run_id=job["run_id"]))
        signal.alarm(900)
    progress("initialize")
    model = ct.Net("convnext_small",str(ROOT/k.s.INIT),crop=352,
                   in_chans=3,spatial=True,extent=False).cuda().train()
    model.enc.set_grad_checkpointing(True)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.02)
    ema = {n:p.detach().clone() for n,p in model.named_parameters()}
    raw = torch.randint(0,256,(2,640,3,384,384),dtype=torch.uint8)
    mask = torch.ones(2,640,dtype=torch.bool)
    meta = torch.zeros(2,640,6)
    target = torch.full((2,12),.37,device="cuda")
    gen = torch.Generator().manual_seed(1052)
    augmentation = SimpleNamespace(crop=352,aug="crop",aug_res=0.)
    seconds = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        progress("step-"+str(_))
        torch.cuda.synchronize()
        started = time.monotonic()
        opt.zero_grad(set_to_none=True)
        replay.training_backward(model,raw,meta,mask,gen,augmentation,
                                 lambda z:weighted_bce(z,target,torch.ones_like(target)))
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        with torch.no_grad():
            for n,p in model.named_parameters():
                ema[n].mul_(.999).add_(p,alpha=.001)
        torch.cuda.synchronize()
        seconds.append(time.monotonic()-started)
        progress("committed-"+str(_+1))
    peak = torch.cuda.max_memory_reserved()
    k.need(peak <= 12*2**30 and snapshot["desktop_bytes"]+peak+2*2**30 <= snapshot["total_bytes"],
           "fullshape measured memory gate")
    result = dict(status="PASS",contract_id=reg["contract_id"],code_pins=f.code_pins(),
                  gpu="4090",batch=2,windows_per_study=640,crop=352,amp=True,
                  baseline_used=snapshot["desktop_bytes"],total_bytes=snapshot["total_bytes"],
                  peak_reserved=peak,step_seconds=seconds,run_id=job["run_id"],f3_run_id=run_id,
                  qualification_review_sha256=k.sha(f.WORK/"REVIEW-QUALIFY.json"),
                  scope="full encoder/head/AdamW/EMA shape; synthetic images; not real I/O throughput")
    k.write(f.WORK/"fullshape-cuda.json",result)
    return result


@contextlib.contextmanager
def qualification_watchdog():
    """Linux-only two-step diagnostic: fail after 15 minutes without a commit."""
    import signal
    def stalled(signum, frame):
        raise TimeoutError("qualification made no progress for 15 minutes; no retry")
    previous = signal.signal(signal.SIGALRM, stalled)
    signal.alarm(900)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    ap.add_argument("--full-shape",action="store_true")
    ap.add_argument("--f3-run-id")
    ap.add_argument("--qualification-binding",action="store_true")
    a = ap.parse_args()
    if a.qualification_binding:
        try:
            result = dict(status="PASS", qualification=qualify_binding())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result = dict(status="BLOCKED", blockers=[str(exc)])
    elif a.full_shape:
        if a.device != "cuda":
            raise ValueError("full-shape requires explicit --device cuda")
        if os.name == "nt":
            raise ValueError("fullshape qualification is Vast Linux only")
        with qualification_watchdog():
            result = fullshape(a.f3_run_id)
    else:
        if a.f3_run_id:
            raise ValueError("--f3-run-id requires --full-shape")
        result = numerical(a.device)
    print(json.dumps(result,allow_nan=False))


if __name__ == "__main__":
    main()
