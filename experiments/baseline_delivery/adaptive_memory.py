"""Opt-in microbatch adaptation. No framework imports or allocation by this policy.

Changing physical batches defines an adaptive experiment, not bitwise equivalence
to a fixed-batch run. This base policy keeps the logical batch fixed;
logical_batch.py adds the explicitly separate sample-scheduled joint policy.
"""
import copy
import json
import math
import os
import subprocess
import time
from pathlib import Path


def default_policy(batch_size):
    tiers=[n for n in range(1,batch_size+1) if batch_size%n==0]
    return dict(schema='qkan-adaptive-memory.v1',tiers=tiers,
        initial=max(n for n in tiers if n<=4),window=16,warmup=2,cooldown=256,gain=0.,
        confirm_windows=3,max_trial_windows=8,status_interval=5.,
        target_fraction=.85,reserve_bytes=512*1024**2,max_oom=8,
        execution=dict(logical_batch_cache=True,cpu_prefetch=1,rollback='gpu_preferred_cpu_fallback',
                       statistics='logical_batch',precision='fp32'))


def jax_oom(error):
    # Only the backend's allocation-exhaustion error, never a generic RuntimeError.
    return (type(error).__name__=='XlaRuntimeError' and type(error).__module__.startswith('jaxlib')
        and str(error).startswith('RESOURCE_EXHAUSTED:')
        and any(word in str(error).lower() for word in ('out of memory','failed to allocate')))


def board_memory():
    """JAX has no allocator-independent free-memory API; query the mapped board."""
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
    if not visible: return {}
    try:
        query=subprocess.run(['nvidia-smi','--id='+visible,
            '--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=2,check=True)
        free,total=map(float,query.stdout.strip().split(','))
        if not all(math.isfinite(n) and n>=0 for n in (free,total)): return {}
        return dict(free_bytes=int(free*1024**2),total_bytes=int(total*1024**2))
    except (OSError,ValueError,subprocess.SubprocessError): return {}


class Exhausted(RuntimeError):
    pass


class Controller:
    def __init__(self,policy,batch_size,saved=None):
        self.policy=copy.deepcopy(policy)
        tiers=policy.get('tiers',[])
        if (policy.get('schema')!='qkan-adaptive-memory.v1' or not tiers
                or any(type(n) is not int or n<1 or batch_size%n for n in tiers)
                or tiers!=sorted(set(tiers))):
            raise ValueError('Adaptive tiers must be increasing divisors of the logical batch')
        for name in ('window','warmup','cooldown','max_oom','reserve_bytes'):
            if type(policy[name]) is not int or policy[name]<1: raise ValueError('Invalid adaptive '+name)
        if not 0<policy['target_fraction']<1 or not 0<=policy['gain']<1:
            raise ValueError('Invalid adaptive memory target/gain')
        if (type(policy.get('confirm_windows',3)) is not int or type(policy.get('max_trial_windows',8)) is not int
                or not 1<=policy.get('confirm_windows',3)<=policy.get('max_trial_windows',8)):
            raise ValueError('Invalid throughput confirmation windows')
        if policy.get('initial') not in tiers: raise ValueError('Invalid initial adaptive tier')
        self.state=copy.deepcopy(saved) if saved else dict(index=tiers.index(policy['initial']),oom_count=0,
            ceiling=len(tiers)-1,samples=[],warmup=policy['warmup'],wait=0,baseline=None,
            previous=None,last_event='initial_known_tier',transitions=0)
        s=self.state
        s.setdefault('trial_windows',[])
        self.on_event=None
        self._event_index=s['index'];self.last_memory={}
        if not 0<=s['index']<=s['ceiling']<len(tiers) or s['oom_count']<0:
            raise ValueError('Invalid saved adaptive controller')

    @property
    def micro_batch(self): return self.policy['tiers'][self.state['index']]

    def snapshot(self): return copy.deepcopy(self.state)

    def event(self,name):
        self.state['last_event']=name;self.state['transitions']+=1
        self.state['samples']=[];self.state['warmup']=self.policy['warmup']
        event=dict(timestamp=time.time(),from_micro_batch=self.policy['tiers'][self._event_index],
            to_micro_batch=self.micro_batch,**self.telemetry(),memory=self.last_memory)
        self._event_index=self.state['index']
        if self.on_event: self.on_event(event)

    def oom(self):
        s=self.state;s['oom_count']+=1
        if s['index']==0:
            self.event('oom_exhausted')
            raise Exhausted('Adaptive microbatch tiers/retry budget exhausted')
        s['ceiling']=min(s['ceiling'],s['index']-1);s['index']-=1
        s['baseline']=s['previous']=None;s['trial_windows']=[];s['wait']=self.policy['cooldown']
        self.event('oom_downgrade')

    def observe(self,seconds,memory):
        """Full logical-update timing includes snapshot/restore and retries."""
        if not math.isfinite(seconds) or seconds<=0: raise ValueError('Invalid update duration')
        s=self.state;p=self.policy
        self.last_memory=memory.copy()
        if s['warmup']:
            s['warmup']-=1;return
        s['samples'].append(seconds)
        if len(s['samples'])<p['window']: return
        mean=sum(s['samples'])/len(s['samples']);s['samples']=[]
        if s['previous'] is not None:
            s['trial_windows'].append(mean)
            windows=s['trial_windows'];required=p.get('confirm_windows',3)
            if len(windows)<required: return
            average=sum(windows)/len(windows);threshold=s['baseline']/(1+p['gain'])
            faster=sum(value<threshold for value in windows)
            slower=sum(value>s['baseline'] for value in windows)
            if slower>=required and slower/len(windows)>=2/3 and average>s['baseline']:
                s['index']=s['previous'];s['wait']=p['cooldown']
                s['previous']=s['baseline']=None;s['trial_windows']=[]
                self.event('throughput_revert');return
            if faster>=required and faster/len(windows)>=2/3 and average<threshold:
                s['previous']=s['baseline']=None;s['trial_windows']=[]
                self.event('throughput_upgrade_accepted');return
            if len(windows)<p.get('max_trial_windows',8): return
            # An inconclusive trial is retained, without claiming a speed gain.
            s['previous']=s['baseline']=None;s['trial_windows']=[];s['wait']=p['cooldown']
            self.event('throughput_trial_inconclusive');return
        if s['wait']:
            s['wait']=max(0,s['wait']-p['window'])
            if not s['wait']: s['ceiling']=len(p['tiers'])-1
            return
        if s['index']>=s['ceiling']: return
        # No headroom telemetry means no speculative promotion. Free memory is
        # global; the peak is process-local. Neither is a GPU compute metric.
        total=memory.get('total_bytes');free=memory.get('free_bytes');peak=memory.get('peak_bytes')
        if not total or free is None or peak is None: return
        ratio=p['tiers'][s['index']+1]/self.micro_batch
        if peak*ratio>total*p['target_fraction'] or free<peak*(ratio-1)+p['reserve_bytes']:
            return
        s['previous']=s['index'];s['baseline']=mean;s['trial_windows']=[];s['index']+=1
        self.event('headroom_upgrade_trial')

    def telemetry(self):
        return dict(micro_batch=self.micro_batch,oom_retries=self.state['oom_count'],
            event=self.state['last_event'],transitions=self.state['transitions'])


def attach_journal(controller,output,step):
    """Compact local transition evidence, with no tensors or raw exceptions."""
    from .common import atomic_json
    output=Path(output)
    def emit(event):
        event.update(step=step(),logical_batch_unchanged=not bool(controller.policy.get('logical_tiers')))
        with (output/'adaptive-events.jsonl').open('a',encoding='utf8') as handle:
            handle.write(json.dumps(event,allow_nan=False)+'\n');handle.flush()
        atomic_json(output/'adaptive-memory.json',dict(policy=controller.policy,
            state=controller.snapshot(),event=event,step=step()))
    controller.on_event=emit
    emit(dict(timestamp=time.time(),**controller.telemetry()))


def transaction(controller,update,capture,restore,cleanup,is_oom,measure,synchronize,started_at=None,samples=None):
    """Rollback *all* mutable training state before rerunning the same update.

    Callbacks must include model buffers, optimizer/scheduler and RNG; caller
    commits the sampler cursor only after this returns. Cleanup runs after the
    failed traceback is released. Non-OOM errors are never retried.
    """
    started=time.perf_counter() if started_at is None else started_at
    snapshot=capture();retries=0
    while True:
        failed=False
        try:
            result=update(controller.micro_batch)
            synchronize()  # Include asynchronous kernel failures and time.
        except Exception as error:
            if not is_oom(error): raise
            failed=True
        if not failed:
            used=controller.micro_batch;memory=measure()
            used_logical=getattr(controller,'logical_batch',None)
            transitions=controller.state['transitions']
            if used_logical is not None: controller.used_samples=samples
            if used_logical is not None and retries:
                controller.committed_oom()
            else: controller.observe(time.perf_counter()-started,memory)
            if used_logical is not None and transitions!=controller.state['transitions']: controller.publish()
            return result,dict(used_micro_batch=used,**controller.telemetry(),**memory,
                **(dict(used_logical_batch=used_logical) if used_logical is not None else {}))
        # Outside except: no retained exception frame owns failed tensors.
        try: controller.last_memory=measure()
        except (RuntimeError,OSError): controller.last_memory={}
        cleanup();retries+=1
        if retries>controller.policy['max_oom']: raise Exhausted('Update OOM retry budget exhausted')
        controller.oom();restore(snapshot)


def evaluation(controller,run,cleanup,is_oom):
    """An inference pass has no mutable optimizer state; discard partial totals."""
    retries=0
    while True:
        failed=False
        try: return run(controller.micro_batch)
        except Exception as error:
            if not is_oom(error): raise
            failed=True
        if failed:
            cleanup();retries+=1
            if retries>controller.policy['max_oom']: raise Exhausted('Evaluation OOM retry budget exhausted')
            controller.oom()
