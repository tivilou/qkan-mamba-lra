"""Bounded joint batch policy and sample-based learning-rate clock (stdlib only)."""
import copy
import math
import time

from .adaptive_memory import Controller,default_policy


def policy(batch_size):
    value=default_policy(batch_size)
    value['logical_tiers']=[batch_size,batch_size*2,batch_size*4]
    value['tiers']=[n for n in range(1,batch_size*4+1) if (batch_size*4)%n==0]
    value['schedule']='sample-clock.v1'
    return value


class JointController(Controller):
    def __init__(self,config,batch_size,saved=None):
        logical=config.get('logical_tiers')
        if logical!=[batch_size,batch_size*2,batch_size*4] or config.get('schedule')!='sample-clock.v1':
            raise ValueError('Invalid logical batch tier contract')
        super().__init__(config,logical[-1],saved)
        self.state.setdefault('logical_index',0)
        self.state.setdefault('previous_logical',None)
        if not 0<=self.state['logical_index']<len(logical) or self.micro_batch>self.logical_batch:
            raise ValueError('Invalid saved joint batch state')
        self.used_samples=None
        self._published=(self.micro_batch,self.logical_batch)

    @property
    def logical_batch(self): return self.policy['logical_tiers'][self.state['logical_index']]

    def observe(self,seconds,memory):
        count=self.used_samples
        if not isinstance(count,int) or count<1: raise ValueError('Missing committed batch sample count')
        # Epoch tails and OOM retries are not representative throughput trials.
        if count!=self.logical_batch: return
        before=self.state['transitions'];s=self.state
        ready=(not s['warmup'] and not s['wait'] and s['previous'] is None
            and len(s['samples'])==self.policy['window']-1
            and s['logical_index']<len(self.policy['logical_tiers'])-1)
        baseline=(sum(s['samples'])+seconds/count)/self.policy['window'] if ready else None
        super().observe(seconds/count,memory)
        if self.micro_batch>self.logical_batch:
            self.state['previous_logical']=self.state['logical_index']
            self.state['logical_index']+=1
        elif ready and self.state['transitions']==before:
            # A larger logical batch may still help at the same physical slice:
            # fewer optimizer/snapshot operations, without doubling activations.
            total=memory.get('total_bytes');free=memory.get('free_bytes')
            if total and free is not None and free>=self.policy['reserve_bytes'] and free/total>=1-self.policy['target_fraction']:
                s['previous']=s['index'];s['baseline']=baseline
                s['previous_logical']=s['logical_index'];s['logical_index']+=1
                self.event('logical_upgrade_accumulation_trial')
        if self.state['transitions']!=before:
            if self.state['last_event']=='throughput_revert' and self.state['previous_logical'] is not None:
                self.state['logical_index']=self.state['previous_logical']
            if self.state['last_event'] in ('throughput_revert','throughput_upgrade_accepted','throughput_trial_inconclusive'):
                self.state['previous_logical']=None
        # Event publication is deferred until both indices are coherent.

    def event(self,name):
        # Base observe changes the physical index before logical metadata.
        callback=self.on_event;self.on_event=None
        super().event(name)
        self.on_event=callback

    def oom(self):
        # Retry the SAME logical batch with a smaller physical slice.
        try: super().oom()
        finally:
            self.state['previous_logical']=None
            self.publish()

    def committed_oom(self):
        """The failed batch committed intact; reduce only the NEXT batch."""
        s=self.state
        if s['logical_index']:
            s['logical_index']-=1
            s['index']=min(s['index'],self.policy['tiers'].index(self.logical_batch))
            s['previous']=s['baseline']=s['previous_logical']=None
            s['trial_windows']=[];s['wait']=self.policy['cooldown']
            self.event('logical_oom_cooldown')

    def publish(self):
        previous=self._published
        self._published=(self.micro_batch,self.logical_batch)
        if self.on_event:
            self.on_event(dict(timestamp=time.time(),from_micro_batch=previous[0],
                to_micro_batch=self.micro_batch,from_logical_batch=previous[1],
                to_logical_batch=self.logical_batch,**self.telemetry(),memory=self.last_memory))

    def telemetry(self):
        result=super().telemetry()
        if 'logical_index' in self.state:
            result.update(logical_batch=self.logical_batch,throughput_unit='seconds_per_sample')
        return result


def make_controller(config,batch_size,saved=None):
    return (JointController if config.get('logical_tiers') else Controller)(config,batch_size,saved)


def logical_size(controller,base): return getattr(controller,'logical_batch',base)


def canary_rows(config,synthetic):
    """Force bounded native recovery through every logical tier, never formal tuning."""
    tiers=config.get('canary_logical_tiers')
    if tiers is None: return 3*config['batch_size']
    if (not synthetic or config['epochs']!=2 or
            tiers!=(config.get('adaptive_memory') or {}).get('logical_tiers')):
        raise ValueError('Forced logical tiers are restricted to bounded recovery')
    return sum(tiers)


def canary_tier(controller,config,step):
    tiers=config.get('canary_logical_tiers')
    if tiers is not None:
        # The six-update recovery check uses a repeatable intervention rather
        # than wall-clock/headroom choices that differ between processes.
        controller.state['logical_index']=step%len(tiers)
        controller.state['index']=min(controller.state['index'],
            controller.policy['tiers'].index(controller.logical_batch))


class SampleClock:
    """Actual Adam counters stay unchanged; LR progress follows committed rows."""
    def __init__(self,total,warmup,base_rates,progress=0,anchor_rates=None,saved=None,reference_batch=32):
        if saved is not None:
            self.state=copy.deepcopy(saved)
            if (self.state.get('schema')!='qkan-sample-clock.v1' or self.state['total']!=total
                    or self.state['warmup']!=warmup or self.state['base_rates']!=list(base_rates)
                    or self.state['reference_batch']!=reference_batch):
                raise ValueError('Sample schedule identity changed')
        else:
            self.state=dict(schema='qkan-sample-clock.v1',total=total,warmup=warmup,
                base_rates=list(base_rates),progress=progress,anchor=progress,
                anchor_rates=list(anchor_rates) if anchor_rates is not None else None,reference_batch=reference_batch)
        s=self.state
        if (not 0<=s['anchor']<=s['progress']<=total or total<1 or warmup<0 or reference_batch<1
                or not base_rates or any(not math.isfinite(r) or r<=0 for r in base_rates)
                or (s['anchor_rates'] is not None and (len(s['anchor_rates'])!=len(base_rates)
                    or any(not math.isfinite(r) or r<0 for r in s['anchor_rates'])))):
            raise ValueError('Invalid sample progress')

    def rates(self):
        s=self.state;p=s['progress'];w=min(s['warmup'],s['total']);a=s['anchor']
        if s['anchor_rates'] is not None:
            if a<w and p<w:
                fraction=(p-a)/max(1,w-a)
                return [r+(b-r)*fraction for r,b in zip(s['anchor_rates'],s['base_rates'])]
            if a>=w:
                factor=.5*(1+math.cos(math.pi*min(1,(p-a)/max(1,s['total']-a))))
                return [r*factor for r in s['anchor_rates']]
        factor=min(1,(p+s['reference_batch'])/max(1,w)) if p<w else .5*(1+math.cos(math.pi*(p-w)/max(1,s['total']-w)))
        return [b*factor for b in s['base_rates']]

    def advance(self,count):
        if type(count) is not int or count<1 or self.state['progress']+count>self.state['total']:
            raise ValueError('Sample budget exceeded')
        self.state['progress']+=count

    def snapshot(self): return copy.deepcopy(self.state)


class TorchSampleSchedule:
    def __init__(self,optimizer,clock): self.optimizer=optimizer;self.clock=clock;self.apply()
    def apply(self):
        for group,rate in zip(self.optimizer.param_groups,self.clock.rates()): group['lr']=rate
    def step(self,count): self.clock.advance(count);self.apply()
    def state_dict(self): return self.clock.snapshot()
    def load_state_dict(self,value):
        if value.get('schema')!='qkan-sample-clock.v1': raise ValueError('Invalid sample scheduler')
        self.clock.state=copy.deepcopy(value);self.apply()
