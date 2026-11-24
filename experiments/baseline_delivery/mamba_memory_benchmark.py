"""Finite memory-for-speed adapters. Native objects are injected by mamba-speed."""
import math
import statistics
import time

from .common import atomic_json,digest,fingerprint,read
from .execution import BatchPrefetch
from .performance import PinnedStaging

METHODS=('memory_retain','memory_prefetch','memory_graph')
COMBINED='memory_combined'
FLAGS={METHODS[0]:('retain',),METHODS[1]:('prefetch',),METHODS[2]:('graph',)}
ORDER=(False,True,True,False)*2


def options(route,request):
    if route in FLAGS:return FLAGS[route]
    value=request.get('memory_combination')
    if route!=COMBINED or not isinstance(value,list) or len(value)<2 or len(set(value))!=len(value) or any(v not in ('retain','prefetch','graph') for v in value):
        raise ValueError('Explicit memory execution method required')
    return tuple(value)


def combination(results):
    """Combine only independently faster methods with exact RNG/sample clocks."""
    return [FLAGS[route][0] for route in METHODS if results.get(route,{}).get('measurement_completed') is True
        and results[route].get('status')!='failed' and results[route].get('timing',{}).get('timing_promising')
        and results[route].get('numerics',{}).get('rng_and_scheduler_exact') is True]


class InputPipeline:
    """Same CPU lookahead on both paths; candidate adds two GPU slots/copy stream.

    Queue the next H2D after the current first forward has launched. Wait on the
    copy event before reading a slot, on the compute event before overwriting it,
    and on the old copy before reusing its pinned host memory.
    """
    def __init__(self,torch,train,batches,overlap):
        self.t=torch;self.train=train;self.batches=batches;self.overlap=overlap
        count=2 if overlap else 1
        self.host=[PinnedStaging(torch,32) for _ in range(count)]
        self.buffers=[None]*count;self.ready=[None]*count;self.done=[None]*count
        self.stream=torch.cuda.Stream() if overlap else None
        self.prefetch=None;self.staged=-1;self.copies=0;self.overlapped_submissions=0
    def begin(self):
        self.prefetch=BatchPrefetch(lambda ids:self.train.numpy_batch(list(ids),validate=True))
        self.staged=-1;self._stage(0)
    def _stage(self,index):
        slot=index%len(self.buffers);t=self.t
        arrays=self.prefetch.get(self.batches[index])
        if index+1<len(self.batches):self.prefetch.submit(self.batches[index+1])
        if self.ready[slot] is not None:self.ready[slot].synchronize()
        pinned=self.host[slot].batch(arrays)
        if self.overlap and any(not value.is_pinned() for value in pinned.values()):
            raise ValueError('GPU overlap requires pinned host staging')
        stream=self.stream if self.overlap else t.cuda.current_stream()
        with t.cuda.stream(stream):
            if self.done[slot] is not None:stream.wait_event(self.done[slot])
            if self.buffers[slot] is None:
                self.buffers[slot]={k:t.empty_like(v,device='cuda:0') for k,v in pinned.items()}
            for key,value in pinned.items():self.buffers[slot][key].copy_(value,non_blocking=True)
            ready=t.cuda.Event();ready.record(stream);self.ready[slot]=ready
        self.copies+=1;self.staged=index
    def get(self,index):
        if self.staged!=index:self._stage(index)
        slot=index%len(self.buffers);current=self.t.cuda.current_stream()
        current.wait_event(self.ready[slot])
        for value in self.buffers[slot].values():value.record_stream(current)
        return self.buffers[slot]
    def after_forward(self,index):
        if self.overlap and index+1<len(self.batches):
            self._stage(index+1);self.overlapped_submissions+=1
    def consumed(self,index):
        event=self.t.cuda.Event();event.record(self.t.cuda.current_stream());self.done[index%len(self.buffers)]=event
    def end(self):
        if self.prefetch is not None:
            self.prefetch.__exit__(None,None,None);self.prefetch=None
    def close(self):
        try:self.t.cuda.synchronize()
        finally:self.end()


def controls_equal(left,right):
    from .runner import compare_states
    from .torch_acceleration import RecoveryMismatch
    try:
        for key in ('rng','scheduler'):compare_states(left[key],right[key])
    except RecoveryMismatch:return False
    return True


def numeric_summary(torch,initial,left,right):
    from .mamba_validation import tensor_errors
    delta=lambda result:{key:value-initial[key] for key,value in result['model'].items() if value.is_floating_point()}
    return dict(parameters=tensor_errors(torch,left['model'],right['model']),
        update_delta=tensor_errors(torch,delta(left),delta(right)),
        clipped_gradients=tensor_errors(torch,left['gradients'],right['gradients']),
        optimizer=tensor_errors(torch,left['optimizer'],right['optimizer']),
        rng_and_scheduler_exact=controls_equal(left,right))


def graph_adapter(c):
    # make_graphed_callables replaces the passed module's forward. Wrap the
    # ordinary callable separately so the paired eager reference stays eager.
    class Adapter(c['torch'].nn.Module):
        def __init__(self):super().__init__();self.base=c['forward']
        def forward(self,*values):return self.base(*values)
    return Adapter()


def operation(c,request_path,route):
    from .diagnostic_watch import mark
    from .mamba_diagnostic import summarize
    request=read(request_path);flags=options(route,request);t=c['torch'];policy=c['projection']
    if c['logical']!=32:raise ValueError('Keep logical32 for memory comparisons')
    cursor=c['state']['cursor'];count=min(4,(len(c['train'])-cursor)//32)
    if count<2:raise ValueError('At least two complete pending batches required for overlap')
    batches=[c['state']['permutation'][cursor+32*i:cursor+32*(i+1)].numpy().copy() for i in range(count)]
    samples=32*count;attempts=0;completed=0;graph_calls=0;graph_forward=None;setup=0.;times={False:[],True:[]}
    peaks={False:dict(allocated=0,reserved=0),True:dict(allocated=0,reserved=0)}
    reference=InputPipeline(t,c['train'],batches,False)
    candidate=InputPipeline(t,c['train'],batches,'prefetch' in flags)
    def select(enabled):policy.set_checkpoint_level(0 if enabled and 'retain' in flags else 1)
    def output():
        result=c['output']();result['gradients']=c['host']({name:p.grad for name,p in c['model'].named_parameters() if p.grad is not None})
        return result
    def block(enabled):
        nonlocal attempts,completed
        c['restore']();c['exact']();select(enabled);t.cuda.synchronize();t.cuda.reset_peak_memory_stats()
        pipeline=candidate if enabled else reference;losses=[]
        started=time.perf_counter()
        try:
            pipeline.begin()
            for i,ids in enumerate(batches):
                staged=pipeline.get(i);attempts+=1
                value=c['update'](True,micro_batch=8,batch_ids=ids,staged_batch=staged,
                    forward_override=graph_forward if enabled and 'graph' in flags else None,
                    graph_gradients=enabled and 'graph' in flags,
                    after_forward=lambda i=i:pipeline.after_forward(i))
                completed+=1
                pipeline.consumed(i);losses.append(value)
            t.cuda.synchronize();losses=[float(value) for value in losses]
            if not all(math.isfinite(value) for value in losses):raise ValueError('Nonfinite memory comparison')
            elapsed=time.perf_counter()-started
            peaks[enabled]['allocated']=max(peaks[enabled]['allocated'],t.cuda.max_memory_allocated())
            peaks[enabled]['reserved']=max(peaks[enabled]['reserved'],t.cuda.max_memory_reserved())
            return elapsed,losses
        finally:pipeline.end()
    try:
        if 'graph' in flags:
            mark('mamba_memory_graph_build',flush=True);select(True);started=time.perf_counter()
            sample=c['train'].batch(batches[0][:8],'cuda:0')
            captured=t.cuda.make_graphed_callables(graph_adapter(c),tuple(sample[k] for k in c['inputs']),num_warmup_iters=3)
            def graph_forward(*values):
                nonlocal graph_calls
                graph_calls+=1;return captured(*values)
            t.cuda.synchronize();setup=time.perf_counter()-started;c['restore']();c['exact']()
        # Two ordinary repeats calibrate FP32 repeat noise. These three blocks
        # warm both paths; only the following ABBAx2 blocks enter timing.
        mark('mamba_memory_reference_a',flush=True);block(False);left=output()
        mark('mamba_memory_reference_b',flush=True);block(False);repeat=output()
        mark('mamba_memory_candidate',flush=True);block(True);right=output()
        numerics=numeric_summary(t,c['state']['model'],left,right)
        repeat_noise=numeric_summary(t,c['state']['model'],left,repeat)
        del left,repeat,right
        for i,enabled in enumerate(ORDER):
            mark('mamba_memory_timing_'+str(i),flush=True);elapsed,_=block(enabled);times[enabled].append(elapsed)
        c['restore']();c['exact']()
        timing=summarize(times[False],times[True])
        for enabled in (False,True):
            peaks[enabled]['samples_per_second']=samples/statistics.median(times[enabled])
        result=dict(schema='qkan-mamba-memory-child.v1',measurement_completed=True,method=route,
            requested=list(flags),resolved=list(flags),request_sha256=digest(request_path),
            checkpoint_sha256=request['checkpoint']['sha256'],micro_batch=8,logical_batch=32,
            batch_count=count,samples_per_block=samples,row_ids_sha256=fingerprint([ids.tolist() for ids in batches]),
            reference_seconds=times[False],candidate_seconds=times[True],timing=timing,
            reference_memory=peaks[False],candidate_memory=peaks[True],numerics=numerics,reference_repeat_noise=repeat_noise,
            graph_setup_seconds=setup,graph_forward_replays=graph_calls,
            actual_fused_calls=policy.calls,checkpoint_level_calls={str(k):v for k,v in policy.level_calls.items()},
            gpu_prefetch_copies=candidate.copies,copy_stream_submissions=candidate.overlapped_submissions,
            mechanism={k:v for k,v in policy.mechanism.items() if k!='checkpoint_lvl'},
            reference_execution=dict(checkpoint_lvl=1,gpu_prefetch=False,graph=False),
            candidate_execution=dict(checkpoint_lvl=0 if 'retain' in flags else 1,gpu_prefetch='prefetch' in flags,graph='graph' in flags),
            source_restored_exact=True,disposable_updates=completed,disposable_update_attempts=attempts,
            formal_updates_committed=0,formal_release_allowed=False,
            timing_scope='complete consecutive updates: CPU lookahead/pinned staging/H2D/forward/backward/clip/Adam/schedule/sync; excludes state restore, graph build, numeric copies and thread shutdown',
            memory_scope='process peak; resident reference/candidate buffers and graph pool are shared across measurements',
            claim='Finite timing and numeric measurements only; no whole-dataset speed or formal recovery admission')
        if policy.calls<=0 or ('graph' in flags and graph_calls<=0) or ('prefetch' in flags and candidate.overlapped_submissions<=0):
            raise ValueError('Requested mechanism did not execute')
        atomic_json(request_path.parent/(route+'.json'),result);return result
    except Exception as error:
        from .torch_acceleration import backend_diagnostic
        atomic_json(request_path.parent/(route+'.json'),dict(schema='qkan-mamba-memory-child.v1',
            measurement_completed=False,method=route,requested=list(flags),request_sha256=digest(request_path),
            error=backend_diagnostic(error,'memory_comparison',route),disposable_updates=completed,disposable_update_attempts=attempts,
            formal_updates_committed=0,formal_release_allowed=False))
        raise
    finally:
        try:
            try:reference.close()
            finally:candidate.close()
        finally:
            policy.set_checkpoint_level(1);c['restore']()
