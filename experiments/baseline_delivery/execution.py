"""Bounded execution helpers. Framework objects are injected, never imported here."""
import copy
from concurrent.futures import ThreadPoolExecutor
import time


class BatchPrefetch:
    """One CPU batch ahead; immutable IDs, no framework work on the thread."""
    def __init__(self,load):
        self.load=load;self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='qkan-batch')
        self.pending=None;self.key=None

    def submit(self,ids):
        key=tuple(map(int,ids))
        if self.pending is not None: raise RuntimeError('Prefetch already has an outstanding batch')
        self.key=key;self.pending=self.pool.submit(self.load,key)

    def get(self,ids):
        key=tuple(map(int,ids))
        if self.pending is None: return self.load(key)
        if self.key!=key: raise ValueError('Prefetch sampler identity changed')
        future=self.pending;self.pending=None;self.key=None
        return future.result()

    def __enter__(self): return self
    def __exit__(self,*_):
        if self.pending is not None: self.pending.cancel()
        self.pool.shutdown(wait=True,cancel_futures=True)


class StatusWriter:
    """Throttle advisory I/O; full scientific checkpoints have a separate cadence."""
    def __init__(self,write,interval=5.,clock=time.monotonic):
        if interval<=0: raise ValueError('Invalid status interval')
        self.write=write;self.interval=interval;self.clock=clock;self.last=None;self.token=None

    def emit(self,callback,force=False,token=None):
        now=self.clock()
        if force or self.last is None or token!=self.token or now-self.last>=self.interval:
            callback(self.write);self.last=now;self.token=token;return True
        return False


class RollbackBuffer:
    """Reuse same-device tensor storage instead of offloading every update.

    Only tensor values are copied; tensor factories and device errors come from
    the injected runtime. CPU fallback is explicit and remains sticky for a job.
    """
    def __init__(self,torch,cpu=False): self.torch=torch;self.cpu=cpu;self.buffer=None

    def _copy(self,value,previous=None):
        torch=self.torch
        if isinstance(value,torch.Tensor):
            source=value.detach();device='cpu' if self.cpu else source.device
            if (isinstance(previous,torch.Tensor) and previous.shape==source.shape
                    and previous.dtype==source.dtype and str(previous.device)==str(device)):
                previous.copy_(source);return previous
            return source.to(device=device,copy=True)
        if isinstance(value,dict):
            old=previous if isinstance(previous,dict) else {}
            return {key:self._copy(item,old.get(key)) for key,item in value.items()}
        if isinstance(value,(list,tuple)):
            old=previous if isinstance(previous,(list,tuple)) else ()
            values=[self._copy(item,old[i] if i<len(old) else None) for i,item in enumerate(value)]
            return tuple(values) if isinstance(value,tuple) else values
        return copy.deepcopy(value)

    def capture(self,value):
        self.buffer=self._copy(value,self.buffer);return self.buffer

    def release(self): self.buffer=None
