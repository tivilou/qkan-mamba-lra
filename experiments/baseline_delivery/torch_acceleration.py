"""Optional Torch compiler/graph execution. No framework imports at module scope."""
import statistics
import time
import gc


def policy():
    return dict(schema='qkan-torch-acceleration.v1',mode='compile_graph',backend='inductor',
        dynamic=False,fullgraph=True,fallback_random=True,graph_warmup=3,max_graphs=1,
        benchmark_pairs=2,precision='fp32',max_builds=8)


class BackendFailure(RuntimeError): pass


def fatal_cuda(error):
    while error is not None:
        text=str(error).lower()
        if any(term in text for term in ('illegal memory access','device-side assert','misaligned address','unspecified launch failure')):
            return True
        error=error.__cause__
    return False


class Executor:
    """Framework-free control plane; all trial optimizer updates are rolled back."""
    def __init__(self,runtime,config,saved=None,synthetic=False,publish=lambda _:None,clock=time.perf_counter,trial=lambda _:None,read_state=None,compare=None):
        if config!=policy(): raise ValueError('Unsupported Torch acceleration policy')
        self.runtime=runtime;self.config=config;self.synthetic=synthetic;self.publish=publish;self.clock=clock;self.trial=trial
        self.state=dict(saved) if saved else dict(mode='compile_graph',benchmarked=False,reason='requested',benchmark=None)
        if self.state['mode'] not in ('compile_graph','compile','eager'): raise ValueError('Invalid saved acceleration mode')
        self.reference=False;self.used=False;self.runtime.set_mode(self.state['mode'])
        self.read_state=read_state;self.compare=compare;self.validated=False

    def snapshot(self): return dict(self.state)

    def forward(self,batch):
        if not self.used and not self.reference:
            self.publish(dict(self.snapshot(),phase='compile_capture_warmup'))
        self.used=True
        return self.runtime.forward(batch,'eager' if self.reference else self.state['mode'])

    def downgrade(self,reason):
        old=self.state['mode']
        if old=='eager': return False
        self.state['mode']='compile' if old=='compile_graph' else 'eager'
        self.state.update(benchmarked=False,benchmark=None)
        self.validated=False
        self.state['reason']=reason;self.runtime.clear();self.runtime.set_mode(self.state['mode'])
        self.state['eager_agreement']='eager_only' if self.state['mode']=='eager' else 'pending'
        self.publish(self.snapshot());return True

    def run(self,callback,snapshot,restore,clear_grad,synchronize,is_oom):
        def restored():
            clear_grad();restore(snapshot)
        while True:
            try:
                if self.read_state and not self.validated and self.state['mode']!='eager':
                    self.publish(dict(self.snapshot(),phase='eager_numerical_check'))
                    self.trial(True)
                    self.reference=True;restored();callback();synchronize();baseline=self.read_state()
                    self.reference=False;restored();callback();synchronize();candidate=self.read_state()
                    try: self.compare(baseline,candidate)
                    except (AssertionError,ValueError) as error: raise BackendFailure('NumericalMismatch') from error
                    restored();self.trial(False);self.validated=True
                    self.state['eager_agreement']='passed_rtol1e-5_atol1e-6_rng_exact'
                if not self.synthetic and not self.state['benchmarked'] and self.state['mode']!='eager':
                    self.publish(dict(self.snapshot(),phase='bounded_benchmark'))
                    self.trial(True)
                    samples={True:[],False:[]}
                    # One warmup and two measured pairs. Same RNG/optimizer/samples.
                    for pair in range(self.config['benchmark_pairs']+1):
                        for reference in (True,False):
                            self.reference=reference;restored();start=self.clock()
                            callback();synchronize()
                            if pair: samples[reference].append(self.clock()-start)
                    restored();self.reference=False
                    self.trial(False)
                    baseline=statistics.median(samples[True]);candidate=statistics.median(samples[False])
                    self.state.update(benchmarked=True,benchmark=dict(eager_seconds=baseline,accelerated_seconds=candidate,
                        pairs=self.config['benchmark_pairs'],includes_optimizer=True,training_updates_committed=0))
                    if candidate>=baseline:
                        self.state.update(mode='eager',reason='benchmark_no_gain');self.runtime.clear();self.runtime.set_mode('eager')
                    else: self.state['reason']='benchmark_gain'
                    self.publish(self.snapshot())
                self.reference=False
                return callback()
            except Exception as error:
                self.trial(False)
                self.reference=False
                if fatal_cuda(error): raise
                # A graph pool can cause OOM. Release it before shrinking batches.
                recover=(isinstance(error,BackendFailure) or self.runtime.backend_error(error)
                    or (is_oom(error) and self.state['mode']=='compile_graph'))
                if not recover: raise
                reason=str(error)[:100] if isinstance(error,BackendFailure) else type(error).__name__
                error.__traceback__=None;error.__cause__=None
                if not self.downgrade(reason): raise
                restored()


def torch_runtime(torch,model,get_rng,set_rng,config):
    """Bind only the forward callable, preserving canonical model state_dict keys."""
    class Runtime:
        def __init__(self):
            self.compiled=None;self.graph=None;self.key=None;self.builds=0;self.mode=None
        def set_mode(self,mode): self.mode=mode
        def status(self):
            return dict(graph_builds=self.builds,graph_resident=self.graph is not None,
                compiled_callable_created=self.compiled is not None)
        def clear(self):
            self.graph=None;self.key=None
            if self.mode=='compile': self.compiled=None
            gc.collect();torch.cuda.empty_cache()
        def isolate_gradients(self):
            # make_graphed_callables returns reusable static parameter gradients.
            # Do not let AccumulateGrad steal those buffers across microbatches.
            for parameter in model.parameters():
                if parameter.grad is not None: parameter.grad=parameter.grad.clone()
        def backend_error(self,error):
            return type(error).__module__.startswith(('torch._dynamo','torch._inductor'))
        def eager(self,batch): return model(batch,validated_inputs=True)
        def forward(self,batch,mode):
            if mode=='eager': return self.eager(batch)
            if self.compiled is None:
                if not hasattr(torch,'compile'): raise BackendFailure('Compiler unavailable')
                try:
                    self.compiled=torch.compile(model.forward,backend='inductor',fullgraph=True,dynamic=False,
                        options={'fallback_random':True,'triton.cudagraphs':False})
                except Exception as error: raise BackendFailure(type(error).__name__) from error
            if mode=='compile': return self.compiled(batch,validated_inputs=True)
            keys=tuple(sorted(batch));values=tuple(batch[k] for k in keys)
            signature=tuple((k,tuple(v.shape),str(v.dtype),str(v.device)) for k,v in zip(keys,values))
            if signature!=self.key:
                self.clear()
                if self.builds>=config['max_builds']: raise BackendFailure('Graph build budget exhausted')
                self.builds+=1
                runtime=self
                class Adapter(torch.nn.Module):
                    def __init__(self): super().__init__();self.base=model
                    def forward(self,*args): return runtime.compiled(dict(zip(keys,args)),validated_inputs=True)
                rng=get_rng()
                try:
                    static=tuple(value.detach().clone() for value in values)
                    # Private graph pool; at most one live shape. Optimizer excluded.
                    self.graph=torch.cuda.make_graphed_callables(Adapter(),static,
                        num_warmup_iters=config['graph_warmup'],allow_unused_input=True)
                    self.key=signature
                except Exception as error:
                    raise BackendFailure(type(error).__name__) from error
                finally:
                    torch.cuda.synchronize();set_rng(rng)
            return self.graph(*values)
    return Runtime()
