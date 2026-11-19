"""Optional Torch compiler/graph execution. No framework imports at module scope."""
import statistics
import time
import gc
import copy

MODES=('compile_graph','compile','eager')


class RecoveryMismatch(ValueError):
    """An observed recovery disagreement, not permission to relax equality."""
    pass


BACKENDS=('eager','inductor','aot_eager_default_partitioner','eager_parallel','eager_partitioned')


def policy(mode='compile_graph',backend='inductor',required=None):
    if mode not in MODES: raise ValueError('Unsupported acceleration mode')
    if backend not in BACKENDS or required not in (None,'compile','graph'):
        raise ValueError('Unsupported compiler requirement')
    value=dict(schema='qkan-torch-acceleration.v5',mode=mode,backend=backend,required=required,
        dynamic=False,fullgraph=True,fallback_random=True,graph_warmup=3,max_graphs=2,
        benchmark_pairs=2,precision='fp32',max_builds=8,
        update_warmup=1,validation='each_execution_shape',
        graph_partition='parallel_retrieval_documents' if backend=='eager_parallel' else 'native_retrieval_documents',
        measurement='balanced_synchronized.v1',no_gain_profile=backend not in ('eager_parallel','eager_partitioned'))
    if backend=='eager_partitioned':
        if mode!='compile_graph': raise ValueError('Partitioned backend requires CUDA Graphs')
        value.update(graph_partition='parallel_retrieval_layers_2',segment_layers=2,
                     replica_sync='per_update',max_graphs=4)
    return value


def legacy_policy(mode,version):
    value=policy(mode);value.pop('required');value['schema']='qkan-torch-acceleration.v'+str(version)
    value.pop('graph_partition');value['max_graphs']=1
    value.pop('measurement');value.pop('no_gain_profile')
    if version==1: value.pop('update_warmup');value.pop('validation')
    if version not in (1,2): raise ValueError('Unsupported historical compiler policy')
    return value


class BackendFailure(RuntimeError):
    def __init__(self,message,diagnostic=None):
        super().__init__(message)
        if diagnostic is not None: self.diagnostic=diagnostic
class AccelerationRequired(RuntimeError): pass
class CaptureCapacity(RuntimeError):
    """A declared graph RNG budget exceeded before any capture, not an OOM."""
    def __init__(self,report):
        super().__init__('Graph RNG capture capacity exceeded');self.diagnostic=report
class NumericalMismatch(BackendFailure):
    def __init__(self,report):
        super().__init__('NumericalMismatch');self.report=report


def backend_diagnostic(error,stage,backend):
    """Bounded error structure, without exception text, inputs or frame locals."""
    chain=[];seen=set();current=error
    categories=(('rng_offset_overflow',('increment causes overflow in the offset value',)),
        ('rng_offset_alignment',('rng offset must be a multiple of 4',)),
        ('out_of_memory',('out of memory','cuda_error_out_of_memory')),
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


def graph_rng_budget(model_config,rows,length,documents=1):
    """Conservative v2.7.1 SDPA/native-dropout Philox bound per forward graph.

    Efficient/cuDNN SDPA reserve B*H*Lq*Lk per layer. Native FF/residual
    dropout reserve at most their element counts (H + 4H + H per token).
    Backward reuses forward dropout state; each independent capture resets
    the uint32 intra-graph offset. This is not GPU memory or sample capacity.
    """
    if (not isinstance(model_config,dict) or model_config.get('model')!='transformer'
            or model_config.get('dropout',0)<=0): return None
    increment=documents*model_config['layers']*rows*(model_config['heads']*length*length+6*model_config['width']*length)
    return dict(schema='qkan-graph-rng-budget.v1',stage='capture_budget',
        category='rng_offset_capacity',estimated_increment=increment,limit=2**32-1,
        rows=rows,length=length,documents_per_graph=documents,passed=increment<=2**32-1)


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


def calibration_summary(report):
    """Explain a measured rejection using bounded numbers, never raw errors."""
    actual=report.get('actual') or {};benchmark=actual.get('benchmark') or {}
    if actual.get('reason')!='benchmark_no_gain': return None
    eager=benchmark.get('eager_seconds');candidate=benchmark.get('accelerated_seconds')
    if (not isinstance(eager,(int,float)) or not isinstance(candidate,(int,float))
            or not 0<eager<=candidate<float('inf')): return None
    agreement='数值/RNG已通过' if actual.get('eager_agreement')=='passed_rtol1e-5_atol1e-6_rng_exact' else '见数值诊断'
    return (f'真实断点校准：{agreement}；普通执行 {eager:.6f}s，候选 {candidate:.6f}s，'
        f'慢 {(candidate/eager-1)*100:.3f}%；未达到正收益要求，未启动正式训练。')


def admit_recovery(config,probe):
    """All devices must pass fresh-process recovery at one shared mode ceiling."""
    candidate=copy.deepcopy(config);attempts=[]
    requested=config['torch_acceleration'];start=MODES.index(requested['mode'])
    # Native autograd FX + explicit Graph first: no AOT decomposition or fusion.
    # The pass-through FX backend has no standalone compiler speed claim.
    choices=[(mode,backend) for mode in MODES[start:] for backend in BACKENDS if backend not in ('eager_parallel','eager_partitioned')
             if (mode=='compile_graph' or backend!='eager')
             if (mode!='eager' or backend=='inductor') and allowed(mode,requested.get('required'))]
    if requested['backend'] in ('eager_parallel','eager_partitioned'): choices=[(requested['mode'],requested['backend'])]
    for mode,backend in choices:
        candidate['torch_acceleration']=policy(mode,backend,requested.get('required'))
        try:
            receipts=probe(candidate)
        except RecoveryMismatch as error:
            attempts.append(dict(mode=mode,backend=backend,outcome='recovery_mismatch',detail=str(error)[:160]))
            if hasattr(error,'diagnostic'): attempts[-1]['diagnostic']=error.diagnostic
            message=calibration_summary(getattr(error,'diagnostic',{}))
            if message: print(message,flush=True)
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
    def __init__(self,runtime,config,saved=None,synthetic=False,publish=lambda _:None,clock=time.perf_counter,trial=lambda _:None,read_state=None,compare=None,diagnose=None,operators=None):
        if config!=policy(config.get('mode'),config.get('backend'),config.get('required')): raise ValueError('Unsupported Torch acceleration policy')
        self.runtime=runtime;self.config=config;self.synthetic=synthetic;self.publish=publish;self.clock=clock;self.trial=trial
        self.state=dict(saved) if saved else dict(mode=config['mode'],benchmarked=False,reason='requested',benchmark=None)
        if self.state['mode'] not in ('compile_graph','compile','eager'): raise ValueError('Invalid saved acceleration mode')
        if MODES.index(self.state['mode'])<MODES.index(config['mode']):
            raise ValueError('Saved mode exceeds recovery-admitted ceiling')
        self.reference=False;self.used=False;self.runtime.set_mode(self.state['mode'])
        self.read_state=read_state;self.compare=compare;self.validated=set();self.checks=0
        self.enforce=not synthetic;self.diagnostics=[];self.validating=False
        self.diagnose=diagnose if config['no_gain_profile'] else None
        self.operators=operators if config['no_gain_profile'] else None

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
                    samples={True:[],False:[]};orders=[]
                    # One warmup and two measured pairs. Same RNG/optimizer/samples.
                    for pair in range(self.config['benchmark_pairs']+1):
                        order=(False,True) if pair and pair%2==0 else (True,False)
                        if pair: orders.append(['eager' if r else self.state['mode'] for r in order])
                        for reference in order:
                            self.reference=reference;restored();synchronize();start=self.clock()
                            callback();synchronize()
                            if pair: samples[reference].append(self.clock()-start)
                    restored();self.reference=False
                    baseline=statistics.median(samples[True]);candidate=statistics.median(samples[False])
                    self.state.update(benchmarked=True,benchmark=dict(eager_seconds=baseline,accelerated_seconds=candidate,
                        pairs=self.config['benchmark_pairs'],includes_optimizer=True,training_updates_committed=0,
                        eager_samples_seconds=samples[True],accelerated_samples_seconds=samples[False],
                        pair_orders=orders,measurement=self.config['measurement'],
                        gain_fraction=1-candidate/baseline))
                    if candidate>=baseline:
                        if self.diagnose:
                            profiles={}
                            for reference in (True,False):
                                self.reference=reference;restored();synchronize()
                                profiles['eager' if reference else self.state['mode']]=self.diagnose(reference,callback,synchronize)
                            restored();self.reference=False
                            self.state['benchmark']['no_gain_phase_profiles']=profiles
                        if self.operators:
                            # Exactly one native eager observation, outside timing.
                            # Graph replay hides CPU operator ancestry; never rank
                            # its CUDA launch as if it identified backward kernels.
                            self.reference=True;restored();synchronize()
                            try:
                                observation=self.operators(callback,synchronize)
                            except BaseException as error:
                                self.reference=False
                                if not fatal_cuda(error): restored()
                                raise
                            restored();synchronize();self.reference=False
                            self.state['benchmark']['no_gain_operator_profile']=observation
                        self.state.update(mode='eager',reason='benchmark_no_gain');self.runtime.clear();self.runtime.set_mode('eager')
                    else: self.state['reason']='benchmark_gain'
                    self.trial(False)
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
                if isinstance(error,CaptureCapacity):
                    clear_grad();self.runtime.clear();self.validated.clear()
                    self.state.update(benchmarked=False,benchmark=None,reason='graph_rng_capacity_retry_smaller_microbatch')
                    self.publish(self.snapshot());raise
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
    if config['backend'] in ('eager_parallel','eager_partitioned'):
        from .parallel_attention import parallel_runtime
        return parallel_runtime(torch,model,get_rng,set_rng,config)
    class Runtime:
        def __init__(self):
            self.compiled=None;self.graph=None;self.key=None;self.builds=0;self.mode=None
            self.expected_rows=None;self.replays=0;self.tail_calls=0
            self.stage='idle';self.compiler_warmups=0
            self.encoder_compiled=None;self.capture_budget=None
            model_config=getattr(model,'config',None)
            self.model_config=model_config if isinstance(model_config,dict) else {}
            self.partitioned=(config['backend']=='eager' and self.model_config.get('model')=='transformer'
                and self.model_config.get('task')=='retrieval')
        def set_mode(self,mode): self.mode=mode
        def begin_update(self,micro_batch): self.expected_rows=micro_batch
        def generation(self): return self.builds
        def status(self):
            return dict(graph_builds=self.builds,graph_resident=self.graph is not None,
                compiled_callable_created=self.compiled is not None or self.encoder_compiled is not None,
                graph_replays=self.replays,eager_tail_calls=self.tail_calls,stage=self.stage,
                compiler_warmups=self.compiler_warmups,graph_callables=(2 if self.partitioned else 1) if self.graph else 0,
                capture_partition='retrieval_documents' if self.partitioned else 'whole_forward',capture_budget=self.capture_budget)
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
        def check_budget(self,batch):
            inputs=batch.get('inputs1') if 'inputs1' in batch else batch.get('inputs')
            if inputs is None or len(inputs.shape)<2: return
            if self.partitioned:
                budgets=[graph_rng_budget(self.model_config,value.shape[0],value.shape[1])
                    for value in (batch['inputs1'],batch['inputs2'])]
                self.capture_budget=max((v for v in budgets if v is not None),
                    key=lambda v:v['estimated_increment'],default=None)
            else:
                self.capture_budget=graph_rng_budget(self.model_config,inputs.shape[0],inputs.shape[1],
                    2 if 'inputs2' in batch else 1)
            if self.capture_budget is not None and not self.capture_budget['passed']:
                self.stage='capture_budget';raise CaptureCapacity(self.capture_budget)
        def capture(self,adapter,static,forward):
            def warm_compiler():
                output=forward(*static)
                parameters=tuple(p for p in model.parameters() if p.requires_grad)
                if parameters:
                    gradients=torch.autograd.grad(output,parameters,torch.ones_like(output),allow_unused=True)
                    del gradients
                del output
                torch.cuda.synchronize();torch.cuda.empty_cache();self.compiler_warmups+=1
            self.attempt('compile_forward_backward_warmup',warm_compiler)
            return self.attempt('graph_capture',lambda:torch.cuda.make_graphed_callables(adapter,static,
                num_warmup_iters=config['graph_warmup'],allow_unused_input=True))
        def documents(self,batch):
            self.check_budget(batch)
            if self.encoder_compiled is None:
                self.encoder_compiled=self.attempt('compile_creation',lambda:torch.compile(model.encode,
                    backend='eager',fullgraph=True,dynamic=False))
            inputs=(batch['inputs1'],batch['inputs2'])
            signature=tuple((tuple(v.shape),str(v.dtype),str(v.device)) for v in inputs)
            if self.expected_rows is not None and inputs[0].shape[0]<self.expected_rows:
                self.tail_calls+=1;return self.eager(batch)
            if signature!=self.key:
                self.clear()
                if self.builds>=config['max_builds']: raise BackendFailure('Graph build budget exhausted')
                self.builds+=1;runtime=self
                class Encoder(torch.nn.Module):
                    def __init__(self): super().__init__();self.base=model
                    def forward(self,x): return runtime.encoder_compiled(x,validated_inputs=True)
                rng=get_rng()
                try:
                    graphs=[]
                    for value in inputs:
                        adapter=Encoder();static=(value.detach().clone(),)
                        # Independent pools: document2 replay must not overwrite
                        # document1 activations before the shared loss backward.
                        graphs.append(self.capture(adapter,static,
                            lambda x:self.encoder_compiled(x,validated_inputs=True)))
                    self.graph=tuple(graphs);self.key=signature
                finally:
                    torch.cuda.synchronize();set_rng(rng)
            a=self.attempt('graph_replay',lambda:self.graph[0](inputs[0]));self.replays+=1
            b=self.attempt('graph_replay',lambda:self.graph[1](inputs[1]));self.replays+=1
            self.stage='ready'
            return model.classify_pair(a,b)
        def forward(self,batch,mode):
            if mode=='eager': return self.eager(batch)
            if mode=='compile_graph' and self.partitioned: return self.documents(batch)
            if mode=='compile_graph': self.check_budget(batch)
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
                    self.graph=self.capture(adapter,static,
                        lambda *args:runtime.compiled(dict(zip(keys,args)),validated_inputs=True))
                    self.key=signature
                finally:
                    torch.cuda.synchronize();set_rng(rng)
            value=self.attempt('graph_replay',lambda:self.graph(*values));self.replays+=1
            self.stage='ready';return value
    return Runtime()
