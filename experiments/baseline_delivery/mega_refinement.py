"""Disposable instance-only precision islands and graphable padding boundary."""
from contextlib import contextmanager
import inspect
from pathlib import Path
import textwrap
from .common import digest

ENCODER_SHA='001e2c9ee20154c0b97be09630714093991638c9ab5c5d08a5e9f96883dfb26f'
BRANCH='if not self.traceable and not self.tpu and not padding_mask.any():'


def linear_names(model,tier):
    names=[n for n,m in model.named_modules() if type(m).__name__=='Linear' and '.nffn.' in n]
    if not names:raise ValueError('Official MEGA FFN linear boundary missing')
    if tier=='tail':
        last=max(int(n.split('.layers.')[1].split('.')[0]) for n in names)
        names=[n for n in names if '.layers.'+str(last)+'.' in n]
    elif tier=='projections':
        names += [n for n,m in model.named_modules() if type(m).__name__=='Linear' and n.endswith('.mega_layer.h_proj')]
    elif tier!='ffn':raise ValueError('Unknown precision island tier')
    return names


class PrecisionIslands:
    def __init__(self,model,tier,torch):
        self.active=False;self.installed=[];self.calls=0;self.names=linear_names(model,tier)
        for name,module in model.named_modules():
            if name not in self.names:continue
            original=module.forward
            def forward(*args,_original=original,**kwargs):
                if not self.active:return _original(*args,**kwargs)
                self.calls+=1
                with torch.autocast('cuda',enabled=True,dtype=torch.bfloat16,cache_enabled=False):
                    return _original(*args,**kwargs).float()
            self.installed.append((module,'forward' in module.__dict__,original));module.forward=forward

    @contextmanager
    def activate(self,enabled=True):
        before=self.active;self.active=enabled
        try:yield self
        finally:self.active=before

    def close(self):
        self.active=False
        for module,owned,original in self.installed:
            if owned:module.forward=original
            else:del module.forward
        self.installed=[]


class GraphPadding:
    def __init__(self,encoder):
        path=Path(inspect.getfile(type(encoder)))
        if digest(path)!=ENCODER_SHA:raise ValueError('Pinned MEGA encoder differs')
        original=encoder.forward;source=textwrap.dedent(inspect.getsource(type(encoder).forward))
        if source.count(BRANCH)!=1:raise ValueError('Encoder padding branch differs')
        namespace=dict(original.__func__.__globals__)
        # Always retain the boolean mask; no CPU/GPU truth-value branch, no traceable output change.
        exec(compile(source.replace(BRANCH,'if False:'),str(path),'exec'),namespace)
        candidate=namespace['forward'];self.active=False;self.encoder=encoder
        self.owned='forward' in encoder.__dict__;self.original=original;self.calls=0
        def forward(*args,**kwargs):
            if not self.active:return original(*args,**kwargs)
            self.calls+=1;return candidate(encoder,*args,**kwargs)
        encoder.forward=forward

    @contextmanager
    def activate(self,enabled=True):
        before=self.active;self.active=enabled
        try:yield self
        finally:self.active=before

    def close(self):
        self.active=False
        if self.owned:self.encoder.forward=self.original
        else:del self.encoder.forward


def error_category(error):
    """Allowlisted native error classification; no raw messages or local values."""
    text=str(error).lower()
    if 'legacy stream' in text and 'captur' in text:return 'legacy_capture_dependency'
    for needles,code in ((('out of memory',),'cuda_oom'),(('not have been used','not used in the graph'),'unused_autograd_input'),
            (('operation not permitted when stream','operation not permitted when capturing','cuda stream capture'),'capture_forbidden_operation'),
            (('autocast','cache'),'autocast_capture'),(('cufft',),'fft_capture'),(('generator','philox'),'cuda_rng_capture'),(('cublas',),'blas_capture'),(('does not require grad','does not have a grad_fn'),'missing_autograd_output'),(('synchronize','synchronization'),'capture_synchronization')):
        if any(n in text for n in needles):return code
    return 'unclassified_native_error'


class KernelGrid:
    """Cache only parameter-independent arange conversion; never cache EMA kernels."""
    def __init__(self,model,torch):
        from .mega_fft import SOURCE_SHA
        self.active=False;self.installed=[];self.cache={};self.builds=0;self.calls=0
        expression='torch.arange(length).to(p)'
        for module in model.modules():
            if type(module).__name__!='MultiHeadEMA':continue
            path=Path(inspect.getfile(type(module)))
            if digest(path)!=SOURCE_SHA:raise ValueError('Pinned EMA source differs')
            original=module._compute_kernel;source=textwrap.dedent(inspect.getsource(type(module)._compute_kernel))
            if source.count(expression)!=1:raise ValueError('EMA kernel grid boundary differs')
            namespace=dict(original.__func__.__globals__,_qkan_grid=self.grid)
            exec(compile(source.replace(expression,'_qkan_grid(length,p)'),str(path),'exec'),namespace)
            candidate=namespace['_compute_kernel']
            def compute(*args,_module=module,_original=original,_candidate=candidate,**kwargs):
                return _candidate(_module,*args,**kwargs) if self.active else _original(*args,**kwargs)
            self.installed.append((module,'_compute_kernel' in module.__dict__,original));module._compute_kernel=compute
        if not self.installed:raise ValueError('EMA kernel modules missing')
        self.torch=torch

    def grid(self,length,p):
        key=(length,p.device,p.dtype)
        if key not in self.cache:
            if self.torch.cuda.is_current_stream_capturing():raise ValueError('Grid must be prepared before capture')
            self.cache[key]=self.torch.arange(length).to(p);self.builds+=1
        self.calls+=1;return self.cache[key]

    activate=PrecisionIslands.activate

    def close(self):
        self.active=False
        for module,owned,original in self.installed:
            if owned:module._compute_kernel=original
            else:del module._compute_kernel
        self.installed=[];self.cache.clear()


class FFNGraph:
    """Capture the last FFN only; no shared-module mutation, EMA remains eager."""
    def __init__(self,model,torch,phase):
        self.active=False;self.installed=[];self.calls=0;self.captures=0;self.torch=torch
        modules=[(n,m) for n,m in model.named_modules() if n.endswith('.nffn')]
        if not modules:raise ValueError('Official FFN modules missing')
        name,module=modules[-1];original=module.forward;self.name=name;graphed=None
        def forward(x):
            nonlocal graphed
            if not self.active:return original(x)
            if graphed is None:
                phase('ffn_graph_capture');before=torch.cuda.get_rng_state().clone();torch.cuda.synchronize()
                class Adapter(torch.nn.Module):
                    def __init__(self):
                        super().__init__();self.params=torch.nn.ParameterList(list(module.parameters()))
                    def forward(self,value):return original(value)
                graphed=make_graph(torch,Adapter(),(x,))
                torch.cuda.synchronize();torch.cuda.set_rng_state(before);self.captures+=1
            self.calls+=1;return graphed(x)
        self.installed.append((module,'forward' in module.__dict__,original));module.forward=forward

    activate=PrecisionIslands.activate
    close=PrecisionIslands.close


def native_trace(error):
    """Return framework code locations only, no exception text/values or full paths."""
    rows=[];tb=error.__traceback__
    while tb is not None:
        filename=tb.tb_frame.f_code.co_filename.replace('\\','/')
        if '/torch/' in filename or '/fairseq/' in filename:
            prefix='torch/' if '/torch/' in filename else 'fairseq/'
            rows.append(dict(file=prefix+filename.split('/'+prefix,1)[1],line=tb.tb_lineno,
                function=tb.tb_frame.f_code.co_name))
        tb=tb.tb_next
    return rows[-16:]


# Fixed vocabulary prevents private paths, tensor values and arbitrary messages
# from entering compact returns. Unknown words are replaced, never echoed.
ERROR_WORDS=set("would make depend blocking thrust cuda error operation failed due to a previous during capture stream is capturing not permitted when the default current legacy active invalidated status invalid argument unsupported dependency graph backward forward autograd engine called holding gil python lock generator philox seed current_seed cudageneratorimpl cannot call get_rng_state get_state rng state offset reset set_rng_state tensor tensors differentiated appears have been used in requires grad gradient output input function does require has no grad_fn memory allocation allocate cublas cufft initialization allocator event recording synchronize synchronization device host cpu asynchronous an assertion assert internal expected true false check please report bug at of on for with only all must once pending work global thread local end begin mode relaxed aborted capture_end cudaerrorstreamcaptureunsupported cudaerrorstreamcaptureinvalidated this was because prevent dependent outside captured null use attempting access are enabled disabled one computation gradients running unsafe empty caching cache detach already execution tensorcore fp32 bf16 int float double half replay exit notimplementederror runtimeerror valueerror".split())


def error_template(error):
    import re
    text=str(error)[:4096]
    tokens=re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[.,:;()]',text)
    out=[]
    for token in tokens[:160]:
        value=token.lower()
        safe=token if value in ERROR_WORDS or token in '.,:;()' else '<redacted>'
        if safe!='<redacted>' or not out or out[-1]!=safe:out.append(safe)
    return ' '.join(out)[:1024]


def error_chain(error):
    rows=[];seen=set()
    while error is not None and id(error) not in seen and len(rows)<8:
        seen.add(id(error));rows.append(dict(error_type=type(error).__name__,category=error_category(error),
            template=error_template(error),framework_frames=native_trace(error)))
        error=error.__cause__ if error.__cause__ is not None else error.__context__
    return rows


def make_graph(torch,adapter,values):
    # Samples must own independent storage/autograd edges. Passing an eager
    # intermediate retains its upstream grad_fn (and forward stream) in the
    # captured autograd input surface. Runtime calls still receive the ORIGINAL
    # values so the graphed custom backward propagates gradients upstream.
    samples=tuple(v.detach().clone().requires_grad_(v.requires_grad) for v in values)
    torch.cuda.synchronize()
    return torch.cuda.make_graphed_callables(adapter,samples)


def graph_probe(torch):
    """Nonleaf-input Linear Graph; check upstream and parameter gradients."""
    before=torch.cuda.get_rng_state().clone()
    try:
        model=torch.nn.Linear(8,8).cuda().train()
        source=torch.ones((2,8),device='cuda',requires_grad=True)
        x=source*2.0
        parameters=tuple(model.parameters())
        reference=torch.autograd.grad(model(x).sum(),(*parameters,source),retain_graph=True)
        saved=torch.cuda.get_rng_state().clone()
        graphed=make_graph(torch,model,(x,));torch.cuda.set_rng_state(saved)
        out=graphed(x);out.sum().backward();torch.cuda.synchronize()
        actual=tuple(p.grad for p in parameters)+(source.grad,)
        if (not bool(torch.isfinite(out).all()) or any(g is None or not bool(torch.isfinite(g).all()) for g in actual)
                or not all(bool(torch.allclose(a,b,rtol=1e-5,atol=1e-6)) for a,b in zip(reference,actual))):
            raise ValueError('Nonleaf Graph gradient probe differs')
        return dict(probe_passed=True,graph_replays=1,nonleaf_input=True,upstream_and_parameter_gradients_match=True,
            scope='Nonleaf Linear forward/backward only; no MEGA speed or numerical admission')
    finally:torch.cuda.set_rng_state(before)
