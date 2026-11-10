"""Bounded CPU lookahead, pinned staging and numeric timing; no framework import."""
from contextlib import contextmanager, nullcontext
import time
import subprocess
import sys
from .execution import BatchPrefetch


def policy():
    return dict(schema='qkan-performance.v1',prefetch='bounded_max_logical_prefix',
        pinned_staging='torch_only_reuse',profile_updates=16,precision='fp32')


class Lookahead(BatchPrefetch):
    def get(self,ids):
        key=tuple(map(int,ids))
        if self.pending is None: return self.load(key)
        if not key or self.key[:len(key)]!=key:
            raise ValueError('Lookahead sampler prefix differs')
        future=self.pending;self.pending=None;self.key=None
        values=future.result()
        return {name:value[:len(key)] for name,value in values.items()}


class PinnedStaging:
    """One reusable host buffer per input, overwritten after prior update sync."""
    def __init__(self,torch,capacity): self.torch=torch;self.capacity=capacity;self.buffers={};self.enabled=True
    def batch(self,arrays):
        result={}
        for key,value in arrays.items():
            source=self.torch.as_tensor(value)
            if not self.enabled: result[key]=source;continue
            previous=self.buffers.get(key)
            shape=(self.capacity,*source.shape[1:])
            try:
                if previous is None or previous.shape!=shape or previous.dtype!=source.dtype:
                    previous=self.torch.empty(shape,dtype=source.dtype,device='cpu',pin_memory=True)
                    self.buffers[key]=previous
                view=previous[:len(source)];view.copy_(source);result[key]=view
            except RuntimeError:
                self.enabled=False;self.buffers.clear()
                return {name:self.torch.as_tensor(item) for name,item in arrays.items()}
        return result


class Profile:
    """First N committed updates only. CUDA events use the existing final sync."""
    def __init__(self,limit=16,event=None,clock=time.perf_counter,scope=None):
        if not isinstance(limit,int) or not 0<=limit<=64: raise ValueError('Invalid bounded profile limit')
        self.limit=limit;self.event=event;self.clock=clock;self.count=0;self.records={};self.active=False;self.scope=scope
    def begin(self):
        self.active=self.count<self.limit;self.pending=[];self.failed=False;self.started=self.clock()
    @contextmanager
    def phase(self,name,device=True):
        if not self.active: yield;return
        started=self.clock();a=b=None
        if device and self.event:
            a=self.event();b=self.event();a.record()
        try:
            with self.scope(name) if self.scope else nullcontext(): yield
        finally:
            if b: b.record()
            self.pending.append((name,self.clock()-started,a,b))
    def finish(self,retries=0):
        if not self.active: return False
        self.count+=1
        self.records.setdefault('update_wall_ms',[]).append((self.clock()-self.started)*1000)
        self.records.setdefault('oom_retries',[]).append(retries)
        # Failed-attempt events can be incomplete; no misleading phase breakdown.
        if not retries:
            for name,seconds,a,b in self.pending:
                self.records.setdefault(name+'_host_ms',[]).append(seconds*1000)
                if a: self.records.setdefault(name+'_stream_ms',[]).append(a.elapsed_time(b))
        self.pending=[];self.active=False
        return self.count==self.limit
    def report(self):
        return dict(schema='qkan-performance-profile.v1',sampled_updates=self.count,
            limit=self.limit,complete=self.count==self.limit,
            metrics={k:dict(count=len(v),mean=sum(v)/len(v),maximum=max(v),total=sum(v)) for k,v in self.records.items()},
            interpretation='Host wall and CUDA stream elapsed milliseconds overlap; do not sum them. First updates include warmup. OOM phase breakdown omitted. No speedup claim.')


class ProfileReturn:
    """One bounded uploader process; no network wait on the training scheduler."""
    def __init__(self,root): self.root=root;self.child=None;self.attempted=set()
    def tick(self,suite,jobs):
        if self.child is not None:
            if self.child.poll() is None: return
            self.child=None
        for job in jobs:
            folder=suite/'jobs'/job['id']
            if job['id'] in self.attempted or not (folder/'performance-profile.json').is_file(): continue
            self.attempted.add(job['id'])
            try:
                self.child=subprocess.Popen([sys.executable,'-B','-m','experiments.baseline_delivery.return_artifacts',
                    '--directory',str(folder),'--experiment','performance-profile',
                    '--run-id',suite.name+'--'+job['id'],'--file','performance-profile.json'],
                    cwd=self.root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            except OSError:
                self.child=None  # Keep numeric report locally; never fail a healthy fit for upload startup.
            return
    def close(self):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try: self.child.wait(timeout=10)
            except subprocess.TimeoutExpired: self.child.kill();self.child.wait()
