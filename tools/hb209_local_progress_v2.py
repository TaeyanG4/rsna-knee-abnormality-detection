"""H209 progress adapter v2: actual setup work and per-study raw reads.

Preserves frozen v103 supervision/cleanup and counter/throughput guards. A raw
source read is render work, including between studies. No windows are invented.
"""
import types
from tools import l1_rco_k2_local_v103 as parent

SETUP = frozenset(('configured', 'asset-verified', 'renderer-source-read',
                  'codec-header', 'codec-decoded', 'imports-complete',
                  'model-loaded', 'renderer-ready'))


class Progress(parent.Progress):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_count = 0
        self.inference_seen = False

    def update(self, now, heartbeat, sample, wall):
        if heartbeat is None:
            return super().update(now, None, sample, wall)
        h = dict(heartbeat)
        phase = h['phase']
        if phase not in SETUP:
            self.inference_seen = True
            return super().update(now, h, sample, wall)
        count = h.get('setup_completed')
        parent.need(type(count) is int and count >= self.setup_count, 'setup counter regression/type')
        if phase != 'renderer-source-read':
            parent.need(not self.inference_seen and h['completed'] == h['windows'] == 0,
                        'startup-only event after inference')
        else:
            self.inference_seen = True
        # Real completed reads count even if the sampler misses classifier or
        # study-boundary events. The parent validates their original counters,
        # PID/time and monotonic study/phase transitions as ordinary render work.
        h['phase'] = 'render'
        if count > self.setup_count:
            self.last = now
            self.events += 1
        reason = super().update(now, h, sample, wall)
        self.setup_count = count
        self.phase = phase
        return reason


def supervise(*args, **kwargs):
    f = parent._supervise
    engine = types.FunctionType(f.__code__, dict(f.__globals__, Progress=Progress),
                                f.__name__, f.__defaults__)
    engine.__kwdefaults__ = f.__kwdefaults__
    return engine(*args, **kwargs)
