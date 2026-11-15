"""Explicit optional batch strategies; no framework imports or core mutation."""
from dataclasses import dataclass
from typing import Mapping, Protocol


SCHEMA = 'qkan-optional-optimizations.v1'
REGISTRY = {
    'padding_trim': dict(id='padding.trim_trailing', version='1', available=True,
        type='transformer', insertion_point='CPU microbatch view before transfer',
        hypothesis='Removing common trailing PAD reduces dense attention work.',
        input_schema='integer token matrices [B,L]; PAD=1',
        output_schema='same rows and token positions; [B,aligned_max_last_valid]',
        requires=[], conflicts=[], deterministic=True,
        resource_estimate='One bounded CPU microbatch; slicing views; no graph cache.',
        failure_signatures=['empty_input', 'invalid_token_matrix', 'valid_token_removed'],
        observability_contract=dict(trace_schema='sample-trace.v2',
            stages=['data','preprocess','scoring'],
            input_fields=['original_token_ids','valid_positions','document_identity'],
            output_fields=['trimmed_token_ids','original_position_map','logits'],
            input_refs=['pending_native_fixture/original_tokens'],
            output_refs=['pending_native_fixture/trimmed_tokens','pending_native_fixture/logits'],
            emitted_fields=['token_values','position_map','input_hash','execution_shape','logits'],
            shape_dtype_normalization=['integer token IDs; PAD=1; original absolute positions; FP32 scoring'],
            checks=['complete values and coverage','token/position parity','producer lineage','same-sample replay'],
            deterministic_replay=True)),
    'bf16_amp': dict(id='precision.bf16_amp', version='1', available=False,
        type='transformer', insertion_point='future compute context',
        hypothesis='Lower precision may accelerate eligible compute.',
        input_schema='FP32 master parameters', output_schema='declared mixed precision context',
        requires=[], conflicts=['fp32_attention_tuned'], deterministic=True,
        resource_estimate='Native numerical/recovery evaluation pending.',
        failure_signatures=['plugin_not_implemented']),
    'flash_attention': dict(id='attention.flash_varlen', version='1', available=False,
        type='transformer', insertion_point='future attention adapter',
        hypothesis='Packed Flash attention may accelerate padded sequence attention.',
        input_schema='BF16 QKV with explicit sequence offsets', output_schema='attention outputs in original coordinates',
        requires=['bf16_amp'], conflicts=['fp32_attention_tuned'], deterministic=True,
        resource_estimate='Offline pinned implementation and bounded native validation pending.',
        failure_signatures=['plugin_not_implemented', 'requested_kernel_not_used']),
    'fp32_attention_tuned': dict(id='attention.fp32_tuned', version='1', available=False,
        type='transformer', insertion_point='future attention adapter',
        hypothesis='A shape-specific deterministic FP32 kernel may reduce backward time.',
        input_schema='FP32 QKV', output_schema='FP32 attention outputs',
        requires=[], conflicts=['bf16_amp', 'flash_attention'], deterministic=True,
        resource_estimate='Kernel research pending.', failure_signatures=['plugin_not_implemented']),
}


class TokenOps(Protocol):
    def inspect(self, value): ...  # -> rows, width, per-row last valid position + 1
    def prefix(self, value, width): ...
    def verify(self, original, trimmed, width): ...


@dataclass(frozen=True)
class Selection:
    padding_trim: bool = False
    length_alignment: int = 128

    def manifest(self):
        return dict(schema=SCHEMA, selected=[dict(id=REGISTRY['padding_trim']['id'],
            version='1', length_alignment=self.length_alignment)] if self.padding_trim else [],
            baseline_adapter=not self.padding_trim, scope='diagnostic_only',
            graph_compatibility='not_admitted', precision='fp32', sample_order='preserved')


def resolve(config=None):
    """Reject misspellings, incompatible requests, and unimplemented enabled flags."""
    if config is None: return Selection()
    if not isinstance(config, dict) or set(config)-{'schema', 'optimizations'}:
        raise ValueError('Invalid optional optimization configuration')
    if config.get('schema') != SCHEMA: raise ValueError('Unknown optimization schema')
    options=config.get('optimizations')
    if not isinstance(options, dict) or set(options)-set(REGISTRY):
        raise ValueError('Unknown optimization option')
    enabled=set();alignment=128
    for name, value in options.items():
        allowed={'enabled', 'length_alignment'} if name=='padding_trim' else {'enabled'}
        if not isinstance(value, dict) or set(value)-allowed or type(value.get('enabled')) is not bool:
            raise ValueError('Invalid optimization fields: '+name)
        if value['enabled']: enabled.add(name)
        if name=='padding_trim':
            alignment=value.get('length_alignment',128)
            if type(alignment) is not int or not 1<=alignment<=4000:
                raise ValueError('Invalid length alignment')
    for name in enabled:
        metadata=REGISTRY[name]
        if not set(metadata['requires'])<=enabled: raise ValueError('Missing optimization dependency: '+name)
        if set(metadata['conflicts']) & enabled: raise ValueError('Optimization conflict: '+name)
    unavailable=sorted(name for name in enabled if not REGISTRY[name]['available'])
    if unavailable: raise ValueError('Optimization implementation pending: '+','.join(unavailable))
    return Selection('padding_trim' in enabled,alignment)


def transform_batch(tokens: Mapping, selection: Selection, ops: TokenOps):
    """Labels are not accepted at this boundary. Disabled path preserves identity."""
    if not selection.padding_trim: return tokens, {}
    if set(tokens) not in ({'inputs'}, {'inputs1','inputs2'}):
        raise ValueError('Batch plugin accepts token inputs only')
    result={};observations={};expected_rows=None
    for key, value in tokens.items():
        rows,width,lengths=ops.inspect(value)
        if type(rows) is not int or type(width) is not int or rows<1 or width<1:
            raise ValueError('Invalid token matrix dimensions')
        if expected_rows is not None and rows!=expected_rows: raise ValueError('Document row counts differ')
        expected_rows=rows
        if len(lengths)!=rows or any(type(n) is not int or not 1<=n<=width for n in lengths):
            raise ValueError('Empty or invalid token span')
        cut=min(width,((max(lengths)+selection.length_alignment-1)//selection.length_alignment)*selection.length_alignment)
        result[key]=ops.prefix(value,cut)
        # This correctness audit belongs to the diagnostic, never GPU timing.
        ops.verify(value,result[key],cut)
        observations[key]=dict(rows=rows,original_width=width,execution_width=cut,
            last_valid_positions=lengths,attention_cells_before=rows*width*width,
            attention_cells_after=rows*cut*cut)
    return result,observations
