"""Optional Torch compiler/graph execution. No framework imports at module scope."""
import statistics
import time
import gc
import copy

MODES=('compile_graph','compile','eager')


class RecoveryMismatch(ValueError):
    """An observed recovery disagreement, not permission to relax equality."""
    pass


BACKENDS=('eager','inductor','aot_eager_default_partitioner')


def policy(mode='compile_graph',backend='inductor',required=None):
    if mode not in MODES: raise ValueError('Unsupported acceleration mode')
    if backend not in BACKENDS or required not in (None,'compile','graph'):
        raise ValueError('Unsupported compiler requirement')
    return dict(schema='qkan-torch-acceleration.v3',mode=mode,backend=backend,required=required,
        dynamic=False,fullgraph=True,fallback_random=True,graph_warmup=3,max_graphs=1,
        benchmark_pairs=2,precision='fp32',max_builds=8,
        update_warmup=1,validation='each_execution_shape')


def legacy_policy(mode,version):
    value=policy(mode);value.pop('required');value['schema']='qkan-torch-acceleration.v'+str(version)
    if version==1: value.pop('update_warmup');value.pop('validation')
    if version not in (1,2): raise ValueError('Unsupported historical compiler policy')
    return value


class BackendFailure(RuntimeError):
    def __init__(self,message,diagnostic=None):
        super().__init__(message)
        if diagnostic is not None: self.diagnostic=diagnostic
class AccelerationRequired(RuntimeError): pass
class NumericalMismatch(BackendFailure):
    def __init__(self,report):
        super().__init__('NumericalMismatch');self.report=report


def backend_diagnostic(error,stage,backend):
    """Bounded error structure, without exception text, inputs or frame locals."""
    chain=[];seen=set();current=error
    categories=(('out_of_memory',('out of memory','cuda_error_out_of_memory')),
        ('capture_unsupported',('not permitted when stream is capturing','operation not permitted when stream is capturing',
                                'not supported during capture','cudaerrorstreamcaptureunsupported')),
        ('capture_invalidated',('capture was invalidated','cudaerrorstreamcaptureinvalidated')),
        ('rng_capture',('rng state','generator','philox')),
        ('data_dependent',('data dependent','data-dependent','tensor.item')),
        ('autograd_version',('modified by an inplace operation','version counter')),
        ('recompile_budget',('recompile_limit','cache_size_limit')))
    while current is not None and id(current) not in seen and len(chain)<4:
        seen.add(id(current));message=str(current).lower();frames=[];trace=current.__traceback__
        while trace:
            name=trace.tb_frame.f_code.co_filename.replace('\\','/')
            for prefix in ('experiments/baseline_delivery/','torch/','functorch/'):
                marker='/'+prefix
                if marker in name or name.startswith(prefix):
                    frames.append(dict(file=prefix+name.rsplit(marker,1)[-1]
                        if marker in name else name,line=trace.tb_lineno))
                    break
            trace=trace.tb_next
        category=next((name for name,terms in categories if any(term in message for term in terms)),'unclassified')
        chain.append(dict(type=type(current).__name__,category=category,frames=frames[-8:]))
        current=current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return dict(schema='qkan-backend-error.v1',stage=stage,backend=backend,exceptions=chain)


def require_actual_backend(receipt,config):
    """A same-eager canary pass cannot authorize real Graph calibration."""
    requested=config.get('torch_acceleration')
    if not requested: return
    state=(receipt.get('torch_acceleration') or {}).get('state') or {}
    if state.get('mode')!=requested['mode']:
        error=RecoveryMismatch('Synthetic worker did not retain requested accelerated mode')
        error.diagnostic=dict(requested_mode=requested['mode'],backend=requested['backend'],
            actual_mode=state.get('mode'),reason=state.get('reason'))
        raise error


def compare_update(left,right,tensor_compare,array_compare):
    """Framework adapters report numeric summaries, never array values."""
    failures=[];leaves=0;fields={};differing=0
    def record(path,result):
        nonlocal differing
        name=path.split('/')[1] if '/' in path else path
        entry=fields.setdefault(name,dict(exact=True,differing_leaves=0,max_abs=None))
        if result:
            differing+=1;entry['exact']=False;entry['differing_leaves']+=1
            if result.get('max_abs') is not None:
                entry['max_abs']=max(entry['max_abs'] or 0,result['max_abs'])
            if len(failures)<32: failures.append(dict(path=path,**result))
    def visit(a,b,path):
        nonlocal leaves
        for adapter in (tensor_compare,array_compare):
            result=adapter(a,b,path)
            if result is not None:
                leaves+=1
                record(path,result)
                return
        if type(a)!=type(b):
            record(path,dict(kind='type'))
            return
        if isinstance(a,dict):
            if a.keys()!=b.keys():
                record(path,dict(kind='keys'))
                return
            for key in a: visit(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):
                record(path,dict(kind='length'))
                return
            for i,(x,y) in enumerate(zip(a,b)): visit(x,y,path+'/'+str(i))
        else:
            leaves+=1
            record(path,dict(kind='control') if a!=b else {})
    visit(left,right,'state')
    if failures: raise NumericalMismatch(dict(schema='qkan-update-agreement.v1',passed=False,
        threshold_relaxed=False,rtol=1e-5,atol=1e-6,control_rng='exact',
        checked_leaves=leaves,differing_leaves=differing,fields=fields,
        first_differences=failures,details_truncated=differing>len(failures)))


def allowed(mode,required):
    return required is None or mode=='compile_graph' or (required=='compile' and mode=='compile')


def admit_recovery(config,probe):
    """All devices must pass fresh-process recovery at one shared mode ceiling."""
    candidate=copy.deepcopy(config);attempts=[]
    requested=config['torch_acceleration'];start=MODES.index(requested['mode'])
    # Native autograd FX + explicit Graph first: no AOT decomposition or fusion.
    # The pass-through FX backend has no standalone compiler speed claim.
    choices=[(mode,backend) for mode in MODES[start:] for backend in BACKENDS
             if (mode=='compile_graph' or backend!='eager')
             if (mode!='eager' or backend=='inductor') and allowed(mode,requested.get('required'))]
    for mode,backend in choices:
        candidate['torch_acceleration']=policy(mode,backend,requested.get('required'))
        try:
            receipts=probe(candidate)
        except RecoveryMismatch as error:
            attempts.append(dict(mode=mode,backend=backend,outcome='recovery_mismatch',detail=str(error)[:160]))
            if hasattr(error,'diagnostic'): attempts[-1]['diagnostic']=error.diagnostic
            if mode=='eager': raise
            continue
        actual=[r['torch_acceleration']['state']['mode'] for r in receipts]
        if not actual or any(value not in MODES for value in actual):
            raise ValueError('Missing or invalid acceleration admission receipt')
        ceiling=max([MODES.index(mode)]+[MODES.index(value) for value in actual])
        if ceiling>MODES.index(mode):
            attempts.append(dict(mode=mode,backend=backend,outcome='worker_fallback',selected=MODES[ceiling]))
            continue
        attempts.append(dict(mode=mode,backend=backend,outcome='passed'))
        return candidate,receipts,attempts
    error=AccelerationRequired('No requested accelerated backend passed native admission')
    error.diagnostic=dict(attempts=attempts);raise error


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
        if config!=policy(config.get('mode'),config.get('backend'),config.get('required')): raise ValueError('Unsupported Torch acceleration policy')
        self.runtime=runtime;self.config=config;self.synthetic=synthetic;self.publish=publish;self.clock=clock;self.trial=trial
        self.state=dict(saved) if saved else dict(mode=config['mode'],benchmarked=False,reason='requested',benchmark=None)
        if self.state['mode'] not in ('compile_graph','compile','eager'): raise ValueError('Invalid saved acceleration mode')
        if MODES.index(self.state['mode'])<MODES.index(config['mode']):
            raise ValueError('Saved mode exceeds recovery-admitted ceiling')
        self.reference=False;self.used=False;self.runtime.set_mode(self.state['mode'])
        self.read_state=read_state;self.compare=compare;self.validated=set();self.checks=0
        self.enforce=not synthetic;self.diagnostics=[];self.validating=False

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
        self.validated.clear()
        self.checks=0
        self.state['reason']=reason;self.runtime.clear();self.runtime.set_mode(self.state['mode'])
        self.state['eager_agreement']='eager_only' if self.state['mode']=='eager' else 'pending'
        self.publish(self.snapshot());return True

    def run(self,callback,snapshot,restore,clear_grad,synchronize,is_oom,execution_key=None):
        def restored():
            clear_grad();restore(snapshot)
        while True:
            try:
                if self.enforce and not allowed(self.state['mode'],self.config.get('required')):
                    restored();raise AccelerationRequired('Required acceleration is unavailable; no training update committed')
                def check_key():
                    generation=getattr(self.runtime,'generation',lambda:0)() if self.state['mode']=='compile_graph' else 0
                    return (self.state['mode'],execution_key,generation)
                key=check_key();needs_check=key not in self.validated
                if self.read_state and needs_check and self.state['mode']!='eager':
                    # Cold compilation includes lazy backward compilation/capture.
                    # Warm the complete update before comparing two steady calls.
                    # Cache readiness is process-local: checkpoint restore repeats
                    # this transaction, without changing optimizer/RNG/sample state.
                    if self.checks>=self.config['max_builds']:
                        raise BackendFailure('Execution shape validation budget exhausted')
                    self.checks+=1
                    self.publish(dict(self.snapshot(),phase='full_update_warmup'))
                    self.trial(True);self.reference=False
                    for _ in range(self.config['update_warmup']):
                        restored();callback();synchronize()
                    restored()
                    self.publish(dict(self.snapshot(),phase='eager_numerical_check'))
                    self.validating=True
                    self.reference=True;restored();callback();synchronize();baseline=self.read_state()
                    self.reference=False;restored();callback();synchronize();candidate=self.read_state()
                    try: self.compare(baseline,candidate)
                    except NumericalMismatch: raise
                    except (AssertionError,ValueError) as error: raise BackendFailure('NumericalMismatch') from error
                    self.validating=False;restored();self.trial(False);self.validated.add(check_key())
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
                if self.enforce and not allowed(self.state['mode'],self.config.get('required')):
                    restored();raise AccelerationRequired('Required acceleration is unavailable; no training update committed')
                return callback()
            except Exception as error:
                self.validating=False
                self.trial(False)
                self.reference=False
                if fatal_cuda(error): raise
                if isinstance(error,AccelerationRequired): raise
                if isinstance(error,NumericalMismatch):
                    diagnostic=dict(mode=self.state['mode'],backend=self.config['backend'],report=error.report)
                else:
                    diagnostic=dict(mode=self.state['mode'],backend=self.config['backend'],
                        error=getattr(error,'diagnostic',None) or backend_diagnostic(error,'update',self.config['backend']))
                self.diagnostics.append(diagnostic);self.diagnostics=self.diagnostics[-8:]
                self.publish(dict(self.snapshot(),diagnostics=self.diagnostics))
                if is_oom(error) and self.state['mode']=='compile_graph' and self.config.get('required')=='graph':
                    # Let the existing finite OOM transaction shrink the physical
                    # batch, keeping the requested backend and original samples.
                    clear_grad();self.runtime.clear();self.validated.clear()
                    self.state.update(benchmarked=False,benchmark=None,reason='graph_oom_retry_smaller_microbatch')
                    self.publish(self.snapshot());raise
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
            self.expected_rows=None;self.replays=0;self.tail_calls=0
            self.stage='idle';self.compiler_warmups=0
        def set_mode(self,mode): self.mode=mode
        def begin_update(self,micro_batch): self.expected_rows=micro_batch
        def generation(self): return self.builds
        def status(self):
            return dict(graph_builds=self.builds,graph_resident=self.graph is not None,
                compiled_callable_created=self.compiled is not None,graph_replays=self.replays,
                eager_tail_calls=self.tail_calls,stage=self.stage,compiler_warmups=self.compiler_warmups)
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
        def attempt(self,stage,callback,wrap=True):
            self.stage=stage
            try: return callback()
            except Exception as error:
                if fatal_cuda(error): raise
                diagnostic=backend_diagnostic(error,stage,config['backend'])
                diagnostic['graph_builds']=self.builds
                diagnostic['expected_rows']=self.expected_rows
                error.diagnostic=diagnostic
                # Compiler wrappers can retain a native allocation error as cause.
                current=error;seen=set()
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    if isinstance(current,getattr(torch.cuda,'OutOfMemoryError',())):
                        current.diagnostic=diagnostic
                        if current is error: raise
                        raise current from None
                    current=current.__cause__
                if not wrap and not self.backend_error(error): raise
                raise BackendFailure(type(error).__name__,diagnostic) from error
        def eager(self,batch): return model(batch,validated_inputs=True)
        def forward(self,batch,mode):
            if mode=='eager': return self.eager(batch)
            if self.compiled is None:
                if not hasattr(torch,'compile'): raise BackendFailure('Compiler unavailable')
                def compile_forward():
                    kwargs=dict(backend=config['backend'],fullgraph=True,dynamic=False)
                    if config['backend']=='inductor':
                        kwargs['options']={'fallback_random':True,'triton.cudagraphs':False,
                            'epilogue_fusion':False,'shape_padding':False,'layout_optimization':False}
                    return torch.compile(model.forward,**kwargs)
                self.compiled=self.attempt('compile_creation',compile_forward)
            if mode=='compile':
                return self.attempt('compiled_forward',lambda:self.compiled(batch,validated_inputs=True),wrap=False)
            keys=tuple(sorted(batch));values=tuple(batch[k] for k in keys)
            signature=tuple((k,tuple(v.shape),str(v.dtype),str(v.device)) for k,v in zip(keys,values))
            if self.expected_rows is not None and values[0].shape[0]<self.expected_rows:
                # Preserve the full-microbatch graph across rare epoch tails.
                # The complete mixed update still passes numerical validation.
                self.tail_calls+=1;return self.eager(batch)
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
                    adapter=Adapter()
                    def warm_compiler():
                        # Resolve Dynamo tracing and lazy AOT backward outside
                        # make_graphed_callables' side-stream warmup/capture.
                        output=runtime.compiled(dict(zip(keys,static)),validated_inputs=True)
                        parameters=tuple(p for p in model.parameters() if p.requires_grad)
                        if parameters:
                            gradients=torch.autograd.grad(output,parameters,torch.ones_like(output),allow_unused=True)
                            del gradients
                        del output
                        torch.cuda.synchronize()
                        # Capture uses its own pool; return only unused cold
                        # warmup cache before allocating that pool.
                        torch.cuda.empty_cache()
                        self.compiler_warmups+=1
                    self.attempt('compile_forward_backward_warmup',warm_compiler)
                    # Private graph pool; at most one live shape. Optimizer excluded.
                    self.graph=self.attempt('graph_capture',lambda:torch.cuda.make_graphed_callables(adapter,static,
                        num_warmup_iters=config['graph_warmup'],allow_unused_input=True))
                    self.key=signature
                finally:
                    torch.cuda.synchronize();set_rng(rng)
            value=self.attempt('graph_replay',lambda:self.graph(*values));self.replays+=1
            self.stage='ready';return value
    return Runtime()
