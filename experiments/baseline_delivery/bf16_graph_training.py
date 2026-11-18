"""Explicit execution adapter for the unchanged BF16 worker update body.

The adapter lives only in one owned child. It replaces strategy.update for the
duration of worker.run, restores it on exit, and restores model.forward before
evaluation, checkpointing and trace collection. No optimizer or numeric body is
replaced. All-off calls the original context unchanged.
"""
from contextlib import contextmanager
import math
import os
import time
from pathlib import Path

from . import key_bias_training as strategy
from .common import atomic_json
from .key_bias_consistency import analytic_key_bias, NAMES

POLICY = dict(schema='qkan-bf16-graph-execution.v1', mode='bf16_graph16',
    capture='per_document_forward_backward', micro_batch=16, logical_batch=128,
    tail='bf16_eager_only_for_micro_rows_below16', evaluation='original_bf16_eager',
    compiler_backend='none', numerical_strategy='unchanged_bf16_analytic_zero_k_bias')


class Execution:
    def __init__(self, output, implementation, enabled=True, factory=None):
        self.output = Path(output); self.implementation = implementation; self.enabled = enabled
        self.factory = factory; self.graph = None; self.model = None; self.torch = None
        self.full_micros = 0; self.tail_micros = 0; self.updates = 0; self.last = None

    def prepare(self, torch, model):
        if self.graph is not None:
            if model is not self.model: raise ValueError('One canonical worker model per execution adapter')
            return
        from .worker import rng_state, restore_rng
        from .bf16_graph_runtime import create
        self.model = model; self.torch = torch
        self.graph = (self.factory or create)(torch, model, rng_state, restore_rng)
        # Shape-only preparation occurs before installing canonical gradient hooks.
        # Token0 is a valid non-PAD frozen vocabulary input. Capture values are
        # overwritten on every replay; preparation must restore original RNG.
        inputs = tuple(torch.zeros((16, 4000), dtype=torch.int64, device='cuda:0') for _ in range(2))
        self.graph.prepare(inputs)

    @contextmanager
    def update(self, torch, model, micro, count):
        if not self.enabled:
            with self.original(torch, model, micro, count): yield
            return
        if micro != 16 or not 1 <= count <= 128: raise ValueError('Explicit Graph strategy requires micro16/logical<=128')
        self.prepare(torch, model); self.graph.begin_update(16)
        owned = 'forward' in model.__dict__; prior = model.__dict__.get('forward'); original = model.forward
        before = self.graph.status(); full = 0; tails = 0
        def forward(batch, representations=False, validated_inputs=False):
            nonlocal full, tails
            if representations or not model.training or not validated_inputs:
                raise ValueError('Training-only Graph forward requires validated ordinary batch')
            rows = batch['inputs1'].shape[0]
            if rows == 16:
                full += 1
                return self.graph.forward(batch)
            if not 1 <= rows < 16: raise ValueError('Unexpected Graph microbatch shape')
            tails += 1
            return original(batch, validated_inputs=True)
        try:
            model.forward = forward
            # The canonical BF16 adapter is used by explicit short tails only;
            # full Graph micros execute the captured replica BF16 blocks.
            with strategy.attention_route(torch, model, True) as attention, analytic_key_bias(torch, model, True) as gradient:
                yield
            calls = math.ceil(count/16); after = self.graph.status()
            if (full != count//16 or tails != bool(count % 16) or full+tails != calls
                    or gradient['hook_calls'] != {name:calls for name in NAMES}
                    or not gradient['handles_removed'] or not gradient['qv_exact'] or not gradient['key_zero']
                    or attention['layer_calls'] != [2*tails]*4
                    or (tails and (set(attention['qkv_dtypes']) != {'torch.bfloat16'}
                                   or set(attention['output_dtypes']) != {'torch.bfloat16'}))
                    or not before['graph_resident'] or not after['graph_resident']
                    or before['graph_builds'] != 1 or after['graph_builds'] != 1
                    or after['forward_replays']-before['forward_replays'] != 2*full
                    or after['backward_replays']-before['backward_replays'] != 2*full
                    or after['replica_sync_rounds']-before['replica_sync_rounds'] != full):
                raise ValueError('Actual worker Graph/hook/tail coverage differs')
            self.full_micros += full; self.tail_micros += tails; self.updates += 1
            self.last = dict(full_graph_micros=full, eager_tail_micros=tails, rows=count,
                             hook_observation=gradient, graph=after)
        finally:
            if owned: model.forward = prior
            else: delattr(model, 'forward')

    def status(self):
        return dict(schema=POLICY['schema'], policy=POLICY, enabled=self.enabled,
            mode='bf16_graph16' if self.enabled else 'bf16_eager16_reference',
            worker_updates=self.updates, full_graph_micros=self.full_micros,
            eager_tail_micros=self.tail_micros, last_update=self.last,
            runtime=self.graph.status() if self.graph is not None else dict(graph_resident=False, graph_builds=0))

    def publish(self, state):
        # Persist execution lineage separately from the immutable arithmetic
        # strategy. Runtime counters remain process telemetry, not scientific state.
        state['bf16_graph_execution'] = dict(policy=POLICY, implementation=self.implementation,
            enabled=self.enabled, checkpoint_rule='rebuild derived Graphs from complete scientific checkpoint')
        atomic_json(self.output/'bf16-graph-execution.json', dict(self.status(), step=state['step'],
            implementation=self.implementation, pid=os.getpid(), updated_at=time.time()))

    @contextmanager
    def install(self):
        self.original = strategy.update
        if isinstance(getattr(self.original, '__self__', None), Execution): raise ValueError('Nested execution adapter rejected')
        strategy.update = self.update
        try: yield self
        finally:
            strategy.update = self.original
            if self.graph is not None: self.graph.clear()
