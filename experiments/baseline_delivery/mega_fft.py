"""Instance-local FP32 EMA FFT padding experiment; no shared source mutation."""
from contextlib import contextmanager
import inspect
from pathlib import Path
import textwrap
from .common import digest

SOURCE_SHA='223ee827cffa474c4b2f4759af1999c69b4c2de868174260c3f789a86df8c225'


def fft_length(length):
    if type(length) is not int or length<1:raise ValueError('Positive FFT length required')
    return 1 << (length-1).bit_length()


class PowerTwoEMA:
    def __init__(self,model):
        self.active=False;self.calls=0;self.lengths={};self.installed=[]
        modules=[m for m in model.modules() if type(m).__name__=='MultiHeadEMA']
        if not modules:raise ValueError('Official MEGA EMA instances required')
        prepared=[]
        for module in modules:
            original=module.forward;path=Path(inspect.getfile(type(module)))
            if digest(path)!=SOURCE_SHA:raise ValueError('Pinned EMA source differs')
            source=textwrap.dedent(inspect.getsource(type(module).forward))
            if source.count('n=2 * fft_len')!=3:raise ValueError('EMA FFT expression boundary changed')
            namespace=dict(original.__func__.__globals__,_qkan_fft_length=self.observe)
            exec(compile(source.replace('n=2 * fft_len','n=_qkan_fft_length(2 * fft_len)'),str(path),'exec'),namespace)
            prepared.append((module,original,namespace['forward']))
        for module,original,candidate in prepared:
            def forward(*args,_module=module,_original=original,_candidate=candidate,**kwargs):
                return _candidate(_module,*args,**kwargs) if self.active else _original(*args,**kwargs)
            self.installed.append((module,'forward' in module.__dict__,original))
            module.forward=forward

    def observe(self,length):
        padded=fft_length(length);self.calls+=1
        key=str(length)+'->'+str(padded);self.lengths[key]=self.lengths.get(key,0)+1
        return padded

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
