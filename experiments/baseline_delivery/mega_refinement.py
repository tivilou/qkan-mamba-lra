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
    for needles,code in ((('out of memory',),'cuda_oom'),(('not have been used','not used in the graph'),'unused_autograd_input'),
            (('operation not permitted when stream','operation not permitted when capturing','cuda stream capture'),'capture_forbidden_operation'),
            (('autocast','cache'),'autocast_capture'),(('cufft',),'fft_capture'),(('synchronize','synchronization'),'capture_synchronization')):
        if any(n in text for n in needles):return code
    return 'unclassified_native_error'
