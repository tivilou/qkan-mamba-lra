"""One rollback-only operator observation; framework injected by the worker."""
from contextlib import nullcontext
import math
import re
import sys

PHASES=('forward_loss','backward','clip_optimizer_schedule','unattributed')
PREFIX='qkan_operator/'
TOP=12
MAX_EVENTS=100000
SAFE_NAME=re.compile(r'(?:aten::[A-Za-z0-9_.:]+|autograd::engine::evaluate_function: [A-Za-z0-9_:]+|[A-Za-z0-9_]+Backward[0-9]*)')


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0


def summarize(events):
    """CPU self-device times own kernels; exclude device rows to avoid duplicates.

    PyTorch v2.7.1 subtracts child device time in self_device_time_total.
    These are attributed kernel sums, not stream elapsed or critical-path time.
    """
    groups={phase:{} for phase in PHASES};count=invalid=device_rows=0
    for event in events:
        count+=1
        if count>MAX_EVENTS:
            return dict(status='event_limit_exceeded',event_limit=MAX_EVENTS,partial_ranking_omitted=True)
        if str(getattr(event,'device_type','')).rsplit('.',1)[-1]!='CPU':
            device_rows+=1;continue
        if getattr(event,'is_async',False): continue
        device=getattr(event,'self_device_time_total',None)
        host=getattr(event,'self_cpu_time_total',None)
        if not _number(device) or not _number(host): invalid+=1;continue
        if not device: continue
        phase='unattributed';parent=event
        for _ in range(64):
            if parent is None: break
            name=getattr(parent,'name','')
            if name in tuple(PREFIX+p for p in PHASES[:-1]):
                phase=name[len(PREFIX):];break
            parent=getattr(parent,'cpu_parent',None)
        name=getattr(event,'name','')
        name=name if isinstance(name,str) and len(name)<=160 and SAFE_NAME.fullmatch(name) else '<other_cpu_event>'
        record=groups[phase].setdefault(name,dict(operator=name,calls=0,self_cuda_ms=0.,self_cpu_ms=0.))
        record['calls']+=1;record['self_cuda_ms']+=device/1000;record['self_cpu_ms']+=host/1000
    phases={}
    for phase,group in groups.items():
        ranked=sorted(group.values(),key=lambda item:(-item['self_cuda_ms'],item['operator']))
        total=sum(item['self_cuda_ms'] for item in ranked)
        phases[phase]=dict(self_cuda_ms=total,calls_with_cuda=sum(item['calls'] for item in ranked),
            top_operators=[dict(item,phase_cuda_percent=100*item['self_cuda_ms']/total) for item in ranked[:TOP]],
            omitted_self_cuda_ms=sum(item['self_cuda_ms'] for item in ranked[TOP:]))
    total=sum(item['self_cuda_ms'] for item in phases.values())
    return dict(status='collected' if total else 'cuda_attribution_unavailable',events_seen=count,
        device_rows_excluded=device_rows,invalid_cpu_events=invalid,attributed_self_cuda_ms=total,phases=phases)


class OperatorObservation:
    """No native import, trace export, shapes, stacks, tensor values or files.

    Setup/collection failures are compact observations; update exceptions escape
    unchanged. The caller owns full-state rollback, even after profiler failure.
    """
    def __init__(self,torch,fatal=lambda _:False): self.torch=torch;self.fatal=fatal;self.active=False

    def phase(self,name):
        if not self.active or name not in PHASES[:-1]: return nullcontext()
        return self.torch.profiler.record_function(PREFIX+name)

    def _unavailable(self,error,stage):
        if self.fatal(error): raise error
        return dict(status='unavailable',stage=stage,error_type=type(error).__name__)

    def collect(self,callback,synchronize):
        result=dict(schema='qkan-operator-profile.v1',mode='eager',maximum_updates=1,updates_observed=0,
            top_per_phase=TOP,event_limit=MAX_EVENTS,training_updates_committed=0,
            record_shapes=False,profile_memory=False,with_stack=False,
            timing_scope='One additional instrumented eager update; excluded from admission benchmark and phase timings',
            interpretation='CPU operators own self CUDA kernel sums; device rows excluded to avoid double counting. '
                'Sums may overlap across streams and include profiler overhead; not wall time or a speedup measurement. '
                'First update may include lazy CUDA initialization. No Graph replay operator attribution or causal kernel claim. '
                'Exact rollback is verified separately.')
        profiler=None
        try:
            api=self.torch.profiler
            profiler=api.profile(activities=[api.ProfilerActivity.CPU,api.ProfilerActivity.CUDA],
                record_shapes=False,profile_memory=False,with_stack=False,with_flops=False,
                with_modules=False)
            profiler.__enter__()
        except Exception as error:
            if profiler is not None and not self.fatal(error):
                try: profiler.__exit__(type(error),error,error.__traceback__)
                except Exception: pass
            return dict(result,**self._unavailable(error,'start'))
        self.active=True
        try:
            callback();result['updates_observed']=1;synchronize()
        except BaseException:
            # Never disguise a scientific/CUDA/update failure as a profiling issue.
            details=sys.exc_info()
            self.active=False
            try: profiler.__exit__(*details)
            except BaseException: pass
            raise
        finally: self.active=False
        try:
            profiler.__exit__(None,None,None)
            result.update(summarize(profiler.events()))
        except Exception as error: result.update(self._unavailable(error,'collect'))
        return result
