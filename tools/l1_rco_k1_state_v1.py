"""L1 RCO K1 v1.0.0: resumable training state and shared measured monitor.

Changelog: initial exact mid-epoch restoration, durable epoch checkpoints.
No GPU launch entrypoint. Used by the derivative of the pinned compact trainer.
"""
from __future__ import annotations
import os
from pathlib import Path
import time

from tools.l1_rco_stage0_v1 import need, read, write, replace, digest, sha

VERSION = '1.0.0'


class ThroughputGuard:
    """Checkpoint heartbeats must not reset a sustained low-throughput clock."""
    def __init__(self, now):
        self.history = []; self.best = 0.; self.last_good = now

    def observe(self, status, now):
        if status['phase'] != 'train': return
        self.history.append((now,status['windows']))
        if len(self.history)>1 and now-self.history[0][0]>=120:
            rate=(status['windows']-self.history[0][1])/(now-self.history[0][0])
            self.best=max(self.best,rate)
            if rate>=.25*self.best: self.last_good=now
            need(now-self.last_good<1800,'30-minute throughput collapse watchdog')
            self.history=self.history[-9:]


def atomic_torch_save(value, path):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + '.tmp').open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    replace(path.with_name(path.name + '.tmp'), path)


class Monitor:
    def __init__(self, out, cpu=False):
        import torch
        self.torch = torch; self.out = Path(out); self.cpu = cpu
        self.steps = 0; self.windows = 0; self.train_seconds = 0.
        self.peak = 0; self.eval_batches = 0; self.started = None
        self.phase_name = 'init'; self.epoch = -1
        self.admission = {} if cpu else read(self.out / 'admission.json')
        self.emit()

    def emit(self):
        if not self.cpu:
            from tools.l1_rco_runtime_v120 import memory_ok
            self.peak = max(self.peak, self.torch.cuda.max_memory_reserved())
            need(memory_ok(self.peak, self.admission['desktop_bytes'], self.admission['total_bytes']),
                 'measured VRAM gate failed; recipe must not auto-shrink')
        status = dict(version=VERSION, phase=self.phase_name, epoch=self.epoch, time=time.time(),
                      steps=self.steps, windows=self.windows, train_seconds=self.train_seconds,
                      windows_per_s=self.windows / self.train_seconds if self.train_seconds else None,
                      peak_reserved=self.peak, eval_batches=self.eval_batches)
        try:
            write(self.out / 'status.json', status)
        except OSError:
            pass  # persistent missing heartbeat is handled by the owning outer watchdog

    def phase(self, value):
        self.phase_name = value
        if value == 'train': self.started = time.perf_counter()
        self.emit()

    def step(self, loss, mask):
        import math
        if not self.cpu: self.torch.cuda.synchronize()
        need(math.isfinite(float(loss.detach())), 'nonfinite training loss')
        self.steps += 1; self.windows += int(mask.sum())
        now = time.perf_counter()
        self.train_seconds += now - self.started; self.started = now
        self.emit()

    def evaluation(self, logits):
        need(bool(self.torch.isfinite(logits).all()), 'nonfinite holdout logits')
        self.eval_batches += 1; self.emit()


class TrainerState:
    """Replay the loader permutation, never replay committed optimizer steps.

    Studies uses seed+epoch+index, so skipped prefetched examples have no mutable
    randomness. The loader's pre-iterator generator restores its shuffle. Model,
    optimizer, scheduler, EMA and augmentation/CPU/CUDA RNG resume at the commit.
    """
    def __init__(self, out, model, opt, sched, ema, gen, dl, train_ids, epochs,
                 monitor, resume=True, every=100, stop_steps=0):
        import torch
        self.torch = torch; self.out = Path(out); self.monitor = monitor
        self.model, self.opt, self.sched, self.ema, self.gen = model, opt, sched, ema, gen
        self.dl = dl; self.epochs = epochs; self.every = every; self.stop_steps = stop_steps
        self.identity = digest(dict(train_ids=sorted(train_ids), epochs=epochs,
                                    admission={k: monitor.admission.get(k) for k in ('contract_id','review_binding','cache')},
                                    parameters=[(n, list(p.shape)) for n, p in model.named_parameters()]))
        self.path = self.out / 'ckpt/E_prior_s0_f0.pt'
        self.start_ep = 0; self.batch = 0; self.tot = 0.; self.n = 0; self.log = []
        self.epoch_loader = None; self.saved_rng = None; self.epoch = 0
        self.step_total = 0
        if resume and self.path.exists():
            ck = torch.load(self.path, map_location='cpu', weights_only=False)
            need(ck['version'] == VERSION and ck['identity'] == self.identity, 'checkpoint training identity')
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
            if ema is not None:
                need(set(ema) == set(ck['ema']), 'checkpoint EMA identity')
                for name in ema: ema[name].copy_(ck['ema'][name].to(ema[name].device))
            self.start_ep = ck['epoch']; self.batch = ck['batch']; self.epoch_loader = ck['epoch_loader']
            self.dl.generator.set_state(self.epoch_loader)
            self.tot = ck['tot']; self.n = ck['n']; self.log = ck['log']; self.step_total = ck['step_total']
            self.saved_rng = ck['rng']; self.restore_rng()
            print(f'resume epoch={self.start_ep} committed_batches={self.batch} optimizer_steps={self.step_total}', flush=True)

    def rng(self):
        torch = self.torch
        return dict(gen=self.gen.get_state(), cpu=torch.get_rng_state(),
                    cuda=[] if self.monitor.cpu else torch.cuda.get_rng_state_all())

    def restore_rng(self):
        if self.saved_rng is None: return
        self.gen.set_state(self.saved_rng['gen']); self.torch.set_rng_state(self.saved_rng['cpu'])
        if not self.monitor.cpu: self.torch.cuda.set_rng_state_all(self.saved_rng['cuda'])

    def begin_epoch(self, epoch, ds):
        self.epoch = epoch; ds.epoch = epoch; self.monitor.epoch = epoch
        if epoch != self.start_ep or not self.batch:
            self.batch = 0; self.tot = 0.; self.n = 0
            self.epoch_loader = self.dl.generator.get_state()
        self.monitor.phase('train')
        return self.tot, self.n

    def batches(self):
        self.dl.generator.set_state(self.epoch_loader)
        iterator = iter(self.dl)
        for _ in range(self.batch): next(iterator)
        self.restore_rng(); self.saved_rng = None
        yield from iterator

    def save(self, epoch, batch, log):
        self.monitor.phase('checkpoint')
        torch = self.torch
        atomic_torch_save(dict(version=VERSION, identity=self.identity, epoch=epoch, batch=batch,
                              epochs=self.epochs, step_total=self.step_total, train_studies=len(self.dl.dataset),
                              tot=self.tot if batch else 0., n=self.n if batch else 0, log=list(log),
                              epoch_loader=self.epoch_loader if batch else self.dl.generator.get_state(),
                              model=self.model.state_dict(), opt=self.opt.state_dict(), sched=self.sched.state_dict(),
                              ema=None if self.ema is None else {n: p.detach().cpu() for n, p in self.ema.items()},
                              rng=self.rng()), self.path)
        self.monitor.phase('train')

    def step(self, loss, mask, tot, n):
        self.batch += 1; self.step_total += 1; self.tot = tot; self.n = n
        self.monitor.step(loss, mask)
        if self.step_total % self.every == 0 or self.stop_steps == self.step_total:
            self.save(self.epoch, self.batch, self.log)
        if self.stop_steps and self.step_total == self.stop_steps:
            raise SystemExit(3)  # CPU test only, after a real atomic mid-epoch checkpoint

    def epoch_end(self, epoch, log):
        self.log = list(log); self.save(epoch + 1, 0, log)
        self.batch = 0; self.tot = 0.; self.n = 0

    def finish(self, model):
        self.monitor.phase('checkpoint')
        atomic_torch_save(model.deploy_state_dict(), self.out / 'models/E_prior_s0_f0.pt')
        write(self.out / 'fit-complete.json', dict(version=VERSION, status='COMPLETE',
              train_identity=self.identity, epochs=self.epochs, optimizer_steps=self.step_total,
              contract_id=self.monitor.admission.get('contract_id'),
              model_sha256=sha(self.out / 'models/E_prior_s0_f0.pt'),
              train_windows=self.monitor.windows, train_seconds=self.monitor.train_seconds,
              windows_per_s=self.monitor.windows / self.monitor.train_seconds if self.monitor.train_seconds else None,
              peak_reserved=self.monitor.peak, timing_scope='this invocation; checkpoints excluded',
              train_studies=len(self.dl.dataset), model_fold=0, clean_oof=False,
              exposure='EXPOSED_EXPLORATORY'))
        self.monitor.phase('fit-complete')
