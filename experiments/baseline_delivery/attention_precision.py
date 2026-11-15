"""Scoped BF16 attention mechanism; framework supplied only by native child."""
from contextlib import nullcontext
import math
import statistics
import time


def precision_context(torch,selection):
    if not selection.bf16_amp: return nullcontext()
    if selection.scope!='attention_probe': raise ValueError('BF16 is admitted only for the attention mechanism probe')
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16)


def forward(torch,selection,qkv,mask,dropout):
    """Force the existing efficient SDPA backend; never silently use math/Flash."""
    with precision_context(torch,selection),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
        # Explicit casts make the requested QKV precision observable even when
        # an operator's autocast registration changes. Cast backward returns FP32.
        values=tuple(x.to(torch.bfloat16) for x in qkv) if selection.bf16_amp else qkv
        return torch.nn.functional.scaled_dot_product_attention(*values,attn_mask=mask,dropout_p=dropout,is_causal=False)


def numerical_metrics(torch,left,right):
    reference=left.detach().float();candidate=right.detach().float()
    if reference.shape!=candidate.shape: raise ValueError('Attention result shape changed')
    finite=bool(torch.isfinite(reference).all()) and bool(torch.isfinite(candidate).all())
    if not finite: return dict(finite=False,relative_l2=None,max_absolute=None)
    difference=candidate-reference
    return dict(finite=True,relative_l2=float(difference.norm()/reference.norm().clamp_min(1e-12)),
        max_absolute=float(difference.abs().max()))


def screen(metrics):
    """A mechanism triage threshold, never training/convergence certification."""
    return (len(metrics)==4 and all(row['finite'] for row in metrics)
        and metrics[0]['relative_l2']<=.02 and all(row['relative_l2']<=.05 for row in metrics[1:]))


def balanced_times(call,restore,synchronize,clock=time.perf_counter):
    """Two AB/BA pairs; casts + forward + backward included, RNG setup outside."""
    values={'fp32':[],'bf16':[]};order=[]
    for pair in range(2):
        for route in ('fp32','bf16','bf16','fp32'):
            restore();synchronize();stamp=clock();call(route);synchronize()
            elapsed=clock()-stamp
            if not math.isfinite(elapsed) or elapsed<=0: raise ValueError('Invalid synchronized timing')
            values[route].append(elapsed);order.append(route)
    baseline=statistics.median(values['fp32']);candidate=statistics.median(values['bf16'])
    return dict(samples_seconds=values,order=order,pairs=2,fp32_median_seconds=baseline,
        bf16_median_seconds=candidate,throughput_gain_fraction=baseline/candidate-1,
        interpretation='One layer/document attention forward+backward only; not a complete training update.')


def native_probe(torch,qkv,mask,upstream,selection,dropout,mark=lambda _:None):
    from .optimization_plugins import resolve
    reference=resolve(scope='attention_probe');selections=dict(fp32=reference,bf16=selection)
    if not selection.bf16_amp: raise ValueError('BF16 probe requires explicit enabled selection')
    if not torch.cuda.is_bf16_supported(): raise ValueError('Native BF16 CUDA support missing')
    if any(x.dtype!=torch.float32 for x in qkv): raise ValueError('FP32 QKV leaves required')
    if len(qkv)!=3 or len({tuple(x.shape) for x in qkv})!=1: raise ValueError('Matched QKV shapes required')
    original=[x.detach().clone() for x in qkv]
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    def restore(): torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    def call(route,probability=dropout):
        output=forward(torch,selections[route],qkv,mask,probability)
        gradients=torch.autograd.grad(output,qkv,grad_outputs=upstream.to(output.dtype))
        expected=torch.bfloat16 if route=='bf16' else torch.float32
        if output.dtype!=expected or any(g.dtype!=torch.float32 for g in gradients):
            raise ValueError('Requested attention precision was not observed')
        if not all(bool(torch.isfinite(x).all()) for x in (output,*gradients)):
            raise ValueError('Nonfinite attention result')
        return output,gradients
    def timed(route):
        # Finite checks are outside timing; autograd.grad never accumulates leaf .grad.
        output=forward(torch,selections[route],qkv,mask,dropout)
        torch.autograd.grad(output,qkv,grad_outputs=upstream.to(output.dtype))
    result={}
    try:
        mark('attention_numeric_screen');restore();base=call('fp32',0.);restore();candidate=call('bf16',0.)
        metrics=[numerical_metrics(torch,a,b) for a,b in zip((base[0],*base[1]),(candidate[0],*candidate[1]))]
        result['dropout_off_numerics']=dict(fields=['output','dQ','dK','dV'],metrics=metrics,
            passed=screen(metrics),output_relative_l2_limit=.02,gradient_relative_l2_limit=.05,
            interpretation='Changed precision tolerance screen; cross-precision equality and training convergence are not claimed.')
        del base,candidate
        mark('attention_determinism');repeats={}
        for route in selections:
            restore();one=call(route);end=torch.cuda.get_rng_state().clone();cpu_end=torch.get_rng_state().clone()
            restore();two=call(route)
            exact=all(torch.equal(a,b) for a,b in zip((one[0],*one[1]),(two[0],*two[1])))
            exact=exact and torch.equal(end,torch.cuda.get_rng_state()) and torch.equal(cpu_end,torch.get_rng_state())
            if not exact: raise ValueError('Same-route deterministic replay differs: '+route)
            repeats[route]=dict(output_and_gradients_exact=True,end_rng_exact=True,
                output_dtype=str(one[0].dtype),gradient_dtypes=[str(g.dtype) for g in one[1]])
            del one,two
        result['same_route_replay']=repeats
        if not result['dropout_off_numerics']['passed']:
            result.update(status='numerical_screen_failed',mechanism_passed=False,timing=None)
            return result
        mark('attention_warmup')
        for route in selections:
            for _ in range(2): restore();timed(route)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        mark('attention_balanced_timing')
        result['timing']=balanced_times(timed,restore,torch.cuda.synchronize)
        mark('attention_kernel_proof');proof={}
        for route in selections:
            restore();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            allocated_before=torch.cuda.memory_allocated()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False,profile_memory=False,with_stack=False) as prof:
                timed(route);torch.cuda.synchronize()
            keys=sorted({event.key for event in prof.key_averages()})
            forward_seen=any('_scaled_dot_product_efficient_attention' in key and 'backward' not in key for key in keys)
            backward_seen=any('_scaled_dot_product_efficient_attention_backward' in key for key in keys)
            forbidden=any('_scaled_dot_product_flash_attention' in key or '_scaled_dot_product_attention_math' in key for key in keys)
            if not forward_seen or not backward_seen or forbidden: raise ValueError('Forced efficient attention kernel proof missing')
            proof[route]=dict(efficient_forward=True,efficient_backward=True,other_backend_seen=False,
                operators=[key for key in keys if 'scaled_dot_product' in key],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocated_before_bytes=allocated_before,
                incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-allocated_before,
                memory_scope='Profiler observation after both routes warmed; shared allocator reserve, not isolated-route footprint.')
        result.update(kernel_proof=proof,status='mechanism_measured',mechanism_passed=True)
        return result
    finally:
        restore()
        if (not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state())
                or not all(torch.equal(a,b) for a,b in zip(original,qkv)) or any(x.grad is not None for x in qkv)):
            raise ValueError('Attention input or RNG rollback differs')
        result['rollback']=dict(inputs_exact=True,rng_exact=True,leaf_gradients_unmodified=True)
