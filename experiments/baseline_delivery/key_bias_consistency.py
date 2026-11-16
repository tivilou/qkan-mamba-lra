"""Explicit diagnostic-only analytic K-bias derivative; no native imports.

This changes the gradient policy. It does not claim to reproduce the floating
point BF16 program's derivative or to be merely an arithmetic dtype change.
"""
from contextlib import contextmanager

ID='gradient.key_bias_analytic_zero'
VERSION='1'
CANDIDATE='bf16-key-bias-analytic-zero-v1'
GRADIENT_POLICY='analytic_zero_k_bias_keep_adam_history_v1'
SCHEMA='qkan-key-bias-consistency.v1'
NAMES=tuple('encoder.layers.'+str(i)+'.self_attn.in_proj_bias' for i in range(4))
METADATA=dict(id=ID,version=VERSION,type='transformer',
    insertion_point='Instance-local leaf packed-bias backward hook before accumulation and clipping',
    hypothesis='Suppress spurious gradient in the exact row-shift-invariant K-bias direction before Adam.',
    input_schema='Four FP32 packed QKV bias gradients [384],Q[0:128],K[128:256],V[256:384]',
    output_schema='FP32 [384]; Q/V exact, K zero; parameter and Adam slots retained',
    requires=['precision.bf16_amp'],conflicts=['Graph','formal_training','other_gradient_hooks'],
    deterministic=True,resource_estimate='One supervised three-route single-update screen; no timing or formal training',
    failure_signatures=['gradient_scope_mismatch','nonfinite_raw_gradient','qv_changed','hook_count_mismatch','global_gate_failed'])


def resolve(config=None):
    if config is None: return False
    if (not isinstance(config,dict) or set(config)!={'schema','enabled','gradient_policy'}
            or config.get('schema')!=SCHEMA or type(config.get('enabled')) is not bool
            or config.get('gradient_policy')!=GRADIENT_POLICY):
        raise ValueError('Explicit key-bias consistency policy required')
    return config['enabled']


def manifest():
    return dict(id=ID,version=VERSION,candidate_id=CANDIDATE,gradient_policy=GRADIENT_POLICY,
        forward='existing BF16 attention route unchanged',
        parameter_membership='unchanged',adam_history='retain exp_avg/exp_avg_sq/step; normal decay and weight decay',
        metric_population='all53 tensors/807170 elements; original FP32 reference',
        formal_admission=False)


@contextmanager
def analytic_key_bias(torch,model,enabled=False,*,observe=None):
    if not enabled:
        yield dict(enabled=False,handles_removed=True)
        return
    params=dict(model.named_parameters());layers=list(model.encoder.layers)
    if len(layers)!=4: raise ValueError('Exactly four original Retrieval layers required')
    for name,layer in zip(NAMES,layers):
        attn=layer.self_attn;p=params.get(name)
        if (p is None or p is not attn.in_proj_bias or tuple(p.shape)!=(384,)
                or p.dtype!=torch.float32 or not p.requires_grad
                or attn.embed_dim!=128 or attn.num_heads!=4 or attn.bias_k is not None
                or attn.bias_v is not None or attn.add_zero_attn):
            raise ValueError('Frozen packed-bias scope differs')
        if getattr(p,'_backward_hooks',None):
            raise ValueError('Existing parameter gradient hooks conflict with isolated policy')
    evidence=dict(enabled=True,gradient_policy=GRADIENT_POLICY,handles_removed=False,
        hook_calls={n:0 for n in NAMES},qv_exact=True,key_zero=True)
    handles=[]
    def hook(name):
        def apply(gradient):
            if (tuple(gradient.shape)!=(384,) or gradient.dtype!=torch.float32
                    or not bool(torch.isfinite(gradient).all())):
                raise ValueError('Nonfinite or unexpected raw packed-bias gradient')
            corrected=gradient.clone();corrected[128:256]=0
            if not (torch.equal(gradient[:128],corrected[:128]) and torch.equal(gradient[256:],corrected[256:])):
                raise ValueError('Q/V gradient changed')
            if bool((corrected[128:256]!=0).any()): raise ValueError('K gradient correction incomplete')
            evidence['hook_calls'][name]+=1
            if evidence['hook_calls'][name]>8: raise ValueError('Unexpected extra backward callback')
            if observe is not None: observe(name,gradient,corrected)
            return corrected
        return apply
    try:
        for name in NAMES: handles.append(params[name].register_hook(hook(name)))
        yield evidence
    finally:
        for handle in reversed(handles): handle.remove()
        evidence['handles_removed']=True


def validate_observation(evidence):
    if (evidence.get('enabled') is not True or evidence.get('gradient_policy')!=GRADIENT_POLICY
            or evidence.get('hook_calls')!={n:8 for n in NAMES}
            or evidence.get('qv_exact') is not True or evidence.get('key_zero') is not True
            or evidence.get('handles_removed') is not True):
        raise ValueError('Analytic key-bias callback coverage/cleanup differs')
    return True
