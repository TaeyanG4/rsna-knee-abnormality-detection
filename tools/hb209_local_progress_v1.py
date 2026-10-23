"""H209 local supervisor v1: consume completed setup work from v103g telemetry.
Original local v103 engine and worker math remain immutable. No timers masquerade
as work; unchanged inference Progress handles classifier/study/throughput guards.
"""
import math
import types
from tools import l1_rco_k2_local_v103 as parent
SETUP = frozenset(('configured','asset-verified','renderer-source-read','codec-header',
                  'codec-decoded','imports-complete','model-loaded','renderer-ready'))
class Progress(parent.Progress):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.setup_count=0
        self.inference_seen=False
    def update(self,now,heartbeat,sample,wall):
        if heartbeat is None or heartbeat.get('phase') not in SETUP:
            if heartbeat is not None:self.inference_seen=True
            return super().update(now,heartbeat,sample,wall)
        h=heartbeat
        parent.need(not self.inference_seen,'setup after inference')
        parent.need(h['pid']==self.child['pid'] and type(h['pid']) is int,'setup PID')
        parent.need(type(h['time']) in (float,int) and math.isfinite(h['time']) and
                    self.child['create_time']-1<=h['time']<=wall+2 and h['time']>=self.stamp,'setup timestamp')
        count=h.get('setup_completed')
        parent.need(type(count) is int and count>=self.setup_count and
                    h['completed']==h['windows']==0,'setup counters')
        if count>self.setup_count:
            self.last=now;self.events+=1;self.setup_count=count
        self.stamp=h['time'];self.phase=h['phase']
        return super().update(now,None,sample,wall)
def supervise(*args,**kwargs):
    f=parent._supervise
    engine=types.FunctionType(f.__code__,dict(f.__globals__,Progress=Progress),f.__name__,f.__defaults__)
    engine.__kwdefaults__=f.__kwdefaults__
    return engine(*args,**kwargs)

