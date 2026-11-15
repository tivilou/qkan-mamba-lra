"""Injected-runtime, instance-local attention strategy and fixed diagnostic policy.

No framework import, registration by scanning, global monkeypatch, or training entry.
The all-off context does not even inspect the runtime/model.
"""
from contextlib import contextmanager
import math
import statistics
from types import MethodType


POLICY = dict(schema='qkan-attention-update-policy.v1', micro_batch=16,
    logical_batch=128, trajectory_updates=4, recovery_updates=2, warmup_updates=2,
    timing_pairs=2, dropout_off_limits=dict(logits_relative_l2=.02,
        loss_absolute=.02, gradient_relative_l2=.05, adam_delta_relative_l2=.20),
    trajectory_loss_absolute=.10, minimum_gain=0.0, trace_rows=2,
    precision='BF16 SDPA QKV only; FP32 projections/residual/FFN/head/loss/master/Adam',
    graph='disabled', comparison='matched eager full update; not incumbent Graph')


@contextmanager
def attention_route(torch, model, enabled=False, *, fp32_adapter=False, observe=None):
    """Patch only this model's layer methods; always restore original attributes.

    fp32_adapter exists solely to check this split against original FP32 MHA.
    Inputs/masks are the already canonicalized inputs of TransformerEncoderLayer.
    """
    if not enabled and not fp32_adapter:
        yield None
        return
    layers=list(model.encoder.layers)
    for layer in layers:
        attn=layer.self_attn
        if (not layer.norm_first or not attn.batch_first or not attn._qkv_same_embed_dim
                or attn.bias_k is not None or attn.bias_v is not None or attn.add_zero_attn
                or attn.embed_dim!=128 or attn.num_heads!=4):
            raise ValueError('Attention update supports only the frozen Retrieval Transformer')
    saved=[];stats=dict(calls=0,layer_calls=[0]*len(layers),qkv_dtypes=[],output_dtypes=[])

    def make_block(index):
        def block(layer, x, attn_mask, key_padding_mask, is_causal=False):
            if attn_mask is not None or is_causal or key_padding_mask is None:
                raise ValueError('Original noncausal key-padding-only attention required')
            if x.dtype!=torch.float32 or x.ndim!=3:
                raise ValueError('FP32 hidden state required outside attention')
            attn=layer.self_attn;batch,length,width=x.shape;heads=attn.num_heads
            # Match multi_head_attention_forward's sequence-first packed projection
            # and out-projection layout; only the SDPA arithmetic dtype changes.
            seq=x.transpose(0,1)
            projected=torch.nn.functional._in_projection_packed(seq,seq,seq,
                attn.in_proj_weight,attn.in_proj_bias)
            qkv=tuple(value.reshape(length,batch*heads,attn.head_dim).transpose(0,1)
                .reshape(batch,heads,length,attn.head_dim) for value in projected)
            mask=key_padding_mask
            if mask.dtype==torch.bool:
                mask=torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float('-inf'))
            if tuple(mask.shape)!=(batch,length) or mask.dtype!=torch.float32:
                raise ValueError('Original FP32 additive key PAD mask required')
            mask=mask.reshape(batch,1,1,length)
            actual=tuple(value.to(torch.bfloat16) for value in qkv) if enabled else qkv
            # CUDA efficient SDPA requires bias.dtype == query.dtype. PAD bias
            # contains only 0/-inf, both represented exactly in BF16; retain the
            # original coordinates and exclusions without enabling global AMP.
            if enabled: mask=mask.to(actual[0].dtype)
            backend=torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
            with torch.nn.attention.sdpa_kernel(backend):
                attention=torch.nn.functional.scaled_dot_product_attention(*actual,
                    attn_mask=mask,dropout_p=attn.dropout if layer.training else 0.0,is_causal=False)
            stats['calls']+=1;call=stats['layer_calls'][index];stats['layer_calls'][index]+=1
            stats['qkv_dtypes'].append(str(actual[0].dtype));stats['output_dtypes'].append(str(attention.dtype))
            if observe is not None: observe(index,call,x,qkv,actual,attention,mask)
            merged=attention.to(torch.float32).permute(2,0,1,3).contiguous().view(batch*length,width)
            output=torch.nn.functional.linear(merged,attn.out_proj.weight,attn.out_proj.bias)
            return layer.dropout1(output.view(length,batch,width).transpose(0,1))
        return block

    try:
        for index,layer in enumerate(layers):
            saved.append((layer,'_sa_block' in layer.__dict__,layer.__dict__.get('_sa_block')))
            layer._sa_block=MethodType(make_block(index),layer)
        yield stats
    finally:
        for layer,owned,value in reversed(saved):
            if owned: layer._sa_block=value
            else: delattr(layer,'_sa_block')


def numerical_gate(metrics):
    limits=POLICY['dropout_off_limits']
    if set(metrics)!=set(limits): raise ValueError('Numerical measurement fields differ')
    return all(type(value) in (int,float) and math.isfinite(value) and 0<=value<=limits[key]
        for key,value in metrics.items())


def timing_summary(samples):
    if set(samples)!={'fp32','bf16'} or any(len(row)!=4 or any(
            type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in row)
            for row in samples.values()):
        raise ValueError('Four positive synchronized measurements per route required')
    medians={key:statistics.median(row) for key,row in samples.items()}
    gain=medians['fp32']/medians['bf16']-1
    return dict(samples_seconds=samples,median_seconds=medians,throughput_gain=gain,
        update_time_reduction=1-medians['bf16']/medians['fp32'],gain_passed=gain>POLICY['minimum_gain'])


def validate_request(request, revision):
    from .common import fingerprint
    from .optimization_plugins import resolve
    if (request.get('schema')!='qkan-attention-update-input.v1'
            or request.get('source_revision')!=revision or request.get('policy')!=POLICY
            or fingerprint(request['source_config'])!=request.get('source_config_sha256')):
        raise ValueError('Attention update request/policy identity differs')
    selected=resolve(request['optimization_config'],scope='attention_update')
    if not selected.bf16_amp or selected.padding_trim:
        raise ValueError('Only the explicit isolated BF16 attention update is supported')
    return selected
