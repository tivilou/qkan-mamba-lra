"""Offline, process-local optional convolution for a finite FP32 Mamba test."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

from .common import ROOT, atomic_json, digest, read

MANIFEST=ROOT/'experiments/baseline_delivery/mamba-conv-wheel.json'


def stage():
    """Extract a pinned wheel in this diagnostic checkout; no pip or imports."""
    from .install_mirror import fetch
    manifest=read(MANIFEST);entry=manifest['objects'][0]
    folder=ROOT/'.bootstrap/mamba-convolution'/entry['sha256']
    wheel=fetch(folder,manifest)/entry['filename']
    target=folder/'package';receipt=folder/'extracted.json'
    if folder.is_symlink() or target.is_symlink() or receipt.is_symlink():
        raise ValueError('Optional package root symlink rejected')
    with zipfile.ZipFile(wheel) as archive:
        selected=[n for n in archive.namelist() if n.startswith('causal_conv1d/') and n.endswith('.py')
            or n=='causal_conv1d_cuda.cpython-311-x86_64-linux-gnu.so']
        if not selected or any('..' in Path(n).parts or Path(n).is_absolute() for n in selected):
            raise ValueError('Invalid optional convolution wheel members')
        files={}
        for name in selected:
            path=target/name
            if path.is_symlink() or not path.resolve().is_relative_to(target.resolve()):
                raise ValueError('Optional package symlink rejected')
            path.parent.mkdir(parents=True,exist_ok=True)
            import hashlib
            raw=archive.read(name);sha=hashlib.sha256(raw).hexdigest()
            if path.exists():
                if digest(path)!=sha:raise ValueError('Existing optional package changed')
            else:path.write_bytes(raw)
            files[name]=sha
    atomic_json(receipt,dict(wheel_sha256=entry['sha256'],files=files))
    return dict(root=str(target.resolve()),receipt_sha256=digest(receipt),wheel_sha256=entry['sha256'])


def prepare_native(torch,request):
    """Run only in the collaborator's supervised candidate child."""
    if (torch.__version__!='2.7.1+cu126' or sys.version_info[:2]!=(3,11)
            or torch._C._GLIBCXX_USE_CXX11_ABI is not True):
        raise ValueError('Optional wheel requires original Torch2.7.1+cu126/CPython3.11/CXX11 ABI true')
    binding=request['convolution_package'];entry=read(MANIFEST)['objects'][0]
    target=Path(binding['root']).resolve()
    expected=(ROOT/'.bootstrap/mamba-convolution'/entry['sha256']/'package').resolve()
    if target!=expected or binding['wheel_sha256']!=entry['sha256']:
        raise ValueError('Optional convolution package root/hash differs')
    receipt=target.parent/'extracted.json'
    if digest(receipt)!=binding['receipt_sha256']:raise ValueError('Optional package receipt changed')
    contents=read(receipt)
    if contents['wheel_sha256']!=entry['sha256']:raise ValueError('Optional wheel identity changed')
    for name,sha in contents['files'].items():
        path=target/name
        if not path.resolve().is_relative_to(target) or path.is_symlink() or digest(path)!=sha:
            raise ValueError('Optional package member changed')
    sys.path.insert(0,str(target))
    for name in ('causal_conv1d','causal_conv1d_cuda'):
        module=importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(target):
            raise ValueError('Unexpected preloaded optional convolution package')


def legacy_cuda_api(functions):
    # Pinned Mamba2.2.4 expects the older return-value CUDA interface. The
    # official 1.5.2 Python wrappers expose exactly those 7/10 argument calls
    # over the newer output-buffer CUDA ABI. Do not alter native binaries.
    return SimpleNamespace(causal_conv1d_fwd=functions.causal_conv1d_fwd_function,
        causal_conv1d_bwd=functions.causal_conv1d_bwd_function)


class FusedFP32:
    def __init__(self,torch,model):
        from .mamba_diagnostic import InputProjectionBF16
        module=importlib.import_module('mamba_ssm.modules.mamba_simple')
        scan=importlib.import_module('mamba_ssm.ops.selective_scan_interface')
        functions=importlib.import_module('causal_conv1d.cpp_functions')
        if (digest(Path(module.__file__))!=InputProjectionBF16.source_sha256
                or digest(Path(scan.__file__))!='c79f4432536402fe4d6c1026ccf8d5fc0aaa62b790e9a622295aee14ac45dbcc'
                or module.causal_conv1d_fn is None or not model.blocks
                or any(type(b.mixer) is not module.Mamba or not b.mixer.use_fast_path
                    or any(p.dtype!=torch.float32 for p in b.mixer.parameters()) for b in model.blocks)):
            raise ValueError('Pinned FP32 Mamba and loaded optional convolution required')
        self.module=module;self.scan=scan;self.active=False;self.calls=0
        self.function=module.causal_conv1d_fn;self.inner=module.mamba_inner_fn
        self.cuda=scan.causal_conv1d_cuda;scan.causal_conv1d_cuda=legacy_cuda_api(functions)
        def counted(*args,**kwargs):
            if not self.active:raise ValueError('Unexpected fused execution in reference')
            self.calls+=1
            return self.inner(*args,**kwargs)
        module.mamba_inner_fn=counted
        module.causal_conv1d_fn=None  # exact original ordinary-convolution baseline
        self.mechanism=dict(policy='mamba-causal-conv-fused-fp32-v1',blocks=len(model.blocks),
            parameter_dtype='float32',compute_dtype='float32',graph=False,
            baseline_path='original_unfused',candidate_path='original_fused',
            convolution_version='1.5.2',cuda_api_adapter='official_cpp_functions_return_values',
            installed_into_original_environment=False)

    @contextmanager
    def activate(self):
        if self.active:raise ValueError('Nested convolution policy rejected')
        self.active=True;self.module.causal_conv1d_fn=self.function
        try:yield
        finally:self.module.causal_conv1d_fn=None;self.active=False

    def close(self):
        self.module.causal_conv1d_fn=self.function;self.module.mamba_inner_fn=self.inner
        self.scan.causal_conv1d_cuda=self.cuda;self.active=False
