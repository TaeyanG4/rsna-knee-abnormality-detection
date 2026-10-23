"""v6 1.3.0 bounded study/member recovery; healthy math stays in frozen callers."""
from __future__ import annotations
import contextlib
from contextvars import ContextVar
import gc
import json
from pathlib import Path
import traceback

MODE = ContextVar('v6_retry_mode', default='normal')


class Nonfinite(ValueError):
    pass


def event(job, uid, member, status, trace):
    record = dict(uid=str(uid), member=member, status=status, traceback=trace)
    job.setdefault('_events', []).append(record)
    if job.get('failure_receipt'):
        with Path(job['failure_receipt']).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')


def oom(exc):
    return isinstance(exc, MemoryError) or 'out of memory' in str(exc).lower()


def attempt(fn, job, uid, member, lower_memory=lambda: None):
    """At most one lower-memory retry and one FP32 retry, in either order."""
    import numpy as np
    memory, fp32 = False, False
    while True:
        token = MODE.set('fp32' if fp32 else 'normal')
        try:
            result = fn()
            if isinstance(result, np.ndarray) and not np.isfinite(result).all():
                raise Nonfinite('nonfinite member output')
            return result
        except Exception as exc:
            trace = traceback.format_exc()
            action = ('retry_lower_memory' if oom(exc) and not memory else
                      'retry_fp32' if isinstance(exc, Nonfinite) and not fp32 else 'failed')
            event(job, uid, member, action, trace)
            if action == 'failed':
                return None
        finally:
            MODE.reset(token)
        if action == 'retry_lower_memory':
            memory = True
            try:
                lower_memory()
            except Exception:
                event(job, uid, member, 'failed', traceback.format_exc())
                return None
        else:
            fp32 = True


class Residency:
    """After OOM, park idle weights on CPU and retain only the active member."""
    def __init__(self, models, device):
        self.models, self.device, self.serial = list(models), device, False

    def reduce(self):
        import torch
        self.serial = True
        for model in self.models:
            model.to('cpu')
        gc.collect()
        if str(self.device).startswith('cuda'):
            torch.cuda.empty_cache()

    def activate(self, model):
        if self.serial:
            for other in self.models:
                if other is not model:
                    other.to('cpu')
            model.to(self.device)


def check_budget(values, ids):
    """Global cohort denominator across both shards; exactly 5% is admissible."""
    import numpy as np
    counts = {}
    for name, array in values.items():
        array = np.asarray(array)
        if array.ndim == 2:
            array = array[None]
        if array.shape[1:] != (len(ids), 12):
            raise ValueError('member shape differs: ' + name)
        for i, a in enumerate(array):
            valid = np.isfinite(a).all(1)
            missing = np.isnan(a).all(1)
            if not (valid | missing).all() or not ((a[valid] >= 0) & (a[valid] <= 1)).all():
                raise ValueError('unmarked/invalid member output: ' + name)
            counts[name if len(array) == 1 else f'{name}/{i}'] = int(missing.sum())
    bad = {k: n for k, n in counts.items() if n * 20 > len(ids)}
    if bad:
        raise ValueError('member failures exceed 5%: ' + repr(bad))
    return counts


def verify_events(values, ids, events):
    import numpy as np
    expected = set()
    for name, value in values.items():
        a = np.asarray(value)
        for index, rows in enumerate(a[None] if a.ndim == 2 else a):
            member = name if a.ndim == 2 else f'{name}/{index}'
            expected.update((str(ids[i]), member) for i in np.flatnonzero(np.isnan(rows).all(1)))
    failed = {(e['uid'], e['member']) for e in events if e['status'] == 'failed'}
    if expected != failed or any(not e.get('traceback') for e in events):
        raise ValueError('failure masks/traceback receipts differ')


def available_mean(arrays, weights=None):
    """Missing slot takes the weighted blend of the remaining slots."""
    import numpy as np
    a = np.asarray(arrays)
    w = np.ones(len(a)) if weights is None else np.asarray(weights)
    good = np.isfinite(a).all(-1)
    den = (good * w[:, None]).sum(0)
    num = (np.where(good[..., None], a, 0) * w[:, None, None]).sum(0)
    return np.divide(num, den[:, None], out=np.full_like(num, np.nan), where=den[:, None] > 0)


def rank(a, ranker):
    """Rank successful studies only; failed slots receive the other slots' blend."""
    import numpy as np
    good = np.isfinite(a).all(1)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if good.any():
        out[good] = ranker(a[good])
    return out


def q_values(scores, v):
    import numpy as np
    if all(np.isfinite(a).all() for a in scores.values()):
        return v.q_values(scores)
    v2, _ = v.maths()
    repaired = {}
    for key, a in scores.items():
        a = np.array(a, copy=True)
        mean = available_mean(a)
        for slot in a:
            bad = ~np.isfinite(slot).all(1)
            slot[bad] = mean[bad]
        repaired[key] = a
    r, c, m = (repaired[k] for k in ('l2', 'convnext', 'mil'))
    family = {k: rank(a, v2.rank) for k, a in dict(
        l2=((r[:5] + r[5:]) / 2).mean(0), convnext=c.mean(0), mil=m.mean(0)).items()}
    return rank(available_mean([rank(x, v2.rank) for x in family.values()]), v2.rank), family


def candidates(specs, selected, ids, anchors, scores, s6, v):
    import numpy as np
    all_values = {**anchors, **scores, **({'s6': s6} if s6 is not None else {})}
    check_budget(all_values, ids)
    q, _ = q_values(scores, v)
    if all(np.isfinite(a).all() for a in all_values.values()):
        extra = {k: anchors[k] for k in ('public', 'exp155')} if s6 is not None else {}
        if s6 is not None:
            extra['s6'] = s6
        return v.candidate_values(specs, selected, ids, anchors['s42'], anchors['s43'], q, **extra)
    v2, v5 = v.maths()
    own = available_mean([anchors[k].astype(np.float64) for k in ('s42', 's43')])
    results = {}
    for name, spec in specs.items():
        if specs[selected]['family'] == 'B6' and spec['family'] != 'B6':
            continue
        anchor = own
        if spec['family'] != 'B6':
            r, e = anchors['public'], anchors['exp155']
            if spec['family'] == 'A6':
                from tools import exp189_s6_port as port
                pct = lambda a: np.asarray([port._pct(col.tolist()) for col in a.T]).T
                p954 = available_mean([rank(r, pct), rank(s6, pct)], [.7, .3]).astype(np.float32)
                arms = [e, p954, own]
            else:
                arms = [e, r, own, s6]
            trio = available_mean([rank(a, v2.rank) for a in arms])
            good = np.isfinite(trio).all(1)
            anchor = v5.trio_round_trip(ids, np.where(good[:, None], trio, .5))
            anchor[~good] = np.nan
        weight = spec['partner_weight']
        result = available_mean([rank(anchor, v2.rank), rank(q, v2.rank)], [1-weight, weight])
        v.probability(result, (len(ids), 12), 'final remaining-member blend')
        results[name] = result
    return results


class WindowViews:
    def __init__(self, pixels, rows):
        self.pixels, self.rows = pixels, rows

    def __len__(self):
        return len(self.rows)

    def chunk(self, start, stop):
        import numpy as np
        import torch
        return torch.from_numpy(np.stack([self.pixels[self.rows[start:stop, d]]
                                         for d in range(self.rows.shape[1])], 1))


def streamed_logits(model, images, meta, device, crop, chunk, mode):
    """Public Net.forward operations, padding and batches with bounded transfer."""
    import torch
    n = len(images)
    padded = n + (-n % 32)
    step = chunk or padded
    feats = []
    ctx = (torch.autocast('cuda', dtype=torch.float16 if mode == 'fp16' else torch.bfloat16)
           if mode != 'fp32' and str(device).startswith('cuda') else contextlib.nullcontext())
    with torch.inference_mode(), ctx:
        for start in range(0, padded, step):
            stop = min(start + step, padded)
            x = (images.chunk(start, min(stop, n)) if isinstance(images, WindowViews)
                 else images[start:min(stop, n)])
            off = (x.shape[-1] - crop) // 2
            x = x[:, :, off:off+crop, off:off+crop].to(device)
            if stop > n:
                x = torch.cat([x, x.new_zeros((stop-max(start, n),) + tuple(x.shape[1:]))])
            xs = x.float().div_(255.)
            xs = ((xs - model.mean) / model.std).contiguous(memory_format=torch.channels_last)
            e = model.enc(xs)
            feats.append(model.spatial_pool(e) if model.spatial else e)
        mask = torch.ones((1, n), dtype=torch.bool, device=device)
        z = model.head(torch.cat(feats)[:n], meta.to(device)[None], mask).float()
        if not bool(torch.isfinite(z).all()):
            raise Nonfinite('nonfinite S6 logits')
        return z


def s6_score(images, meta, uid, models, names, device, crop, chunk, amp, job):
    import torch
    pool = job.setdefault('_pool', Residency(models, device))
    zs = []
    for model, name in zip(models, names):
        reduced = [False]
        def lower():
            reduced[0] = True
            pool.reduce()
        def forward():
            pool.activate(model)
            return streamed_logits(model, images, meta, device, crop,
                                   min(chunk or 192, 32) if reduced[0] else chunk,
                                   'fp32' if MODE.get() == 'fp32' else amp)[0]
        z = attempt(forward, job, uid, 's6/' + str(name), lower)
        if z is not None:
            zs.append(z)
    if not zs:
        raise ValueError('all S6 fold members failed for study ' + uid)
    zs = torch.stack(zs)
    return zs.mean(0), zs


def s6_items(ds, models, device, job):
    """Synchronous decode; OOM retries with primary/auxiliary weights off GPU."""
    pool = job.setdefault('_pool', Residency(models, device))
    for i, uid in enumerate(ds.uids):
        item = ds[i]
        if item.get('error') and ('out of memory' in item['error'].lower() or 'MemoryError' in item['error']):
            event(job, uid, 's6/decode', 'retry_lower_memory', item.get('traceback', item['error']))
            try:
                pool.reduce()
                if ds.bank is not None:
                    ds.bank.model.to('cpu')
                    ds.bank.dev = 'cpu'
                if ds.side_net is not None:
                    ds.side_net.to('cpu')
                    ds.side_dev = 'cpu'
                ds.device = 'cpu'
                item = ds[i]
            except Exception:
                item = dict(uid=uid, error='decode recovery failed', traceback=traceback.format_exc(), images=[])
        yield item
