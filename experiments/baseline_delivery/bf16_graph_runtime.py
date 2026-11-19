"""Isolated BF16 encoder Graphs; canonical FP32 Adam and analytic hooks stay eager.

Not registered as a training strategy. Two document-private parameter/gradient
surfaces avoid static-gradient aliasing; forward replays serialize shared RNG.
Framework is supplied by the collaborator-only diagnostic.
"""
import copy
import gc

from .attention_update import attention_route
from .parallel_attention import encoder_parameters, fixed_merge, stream_graphed_encoder
from .torch_acceleration import graph_rng_budget, BackendFailure, CaptureCapacity
from .diagnostic_watch import mark


def budgets(config, inputs):
    if len(inputs) != 2 or any(tuple(v.shape) != (16, 4000) for v in inputs):
        raise BackendFailure('Frozen two-document micro16/length4000 required')
    result = [graph_rng_budget(config, 16, 4000, documents=1) for _ in inputs]
    if any(v is None or not v['passed'] for v in result):
        raise CaptureCapacity(next((v for v in result if v and not v['passed']), {}))
    return result


def create(torch, model, get_rng, set_rng):
    """Capture BF16 attention, not the CPU-observing canonical gradient hooks."""
    class Bridge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, owner, x1, x2, *parameters):
            ctx.owner = owner
            ctx.outputs = owner.encode((x1, x2))
            return tuple(v.detach() for v in ctx.outputs)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, g1, g2):
            owner = ctx.owner; main = torch.cuda.current_stream(); values = [None, None]
            for index, gradient in ((1, g2), (0, g1)):
                stream = owner.streams[index]; stream.wait_stream(main)
                with torch.cuda.stream(stream):
                    if gradient is None: gradient = torch.zeros_like(ctx.outputs[index])
                    gradient.record_stream(stream)
                    values[index] = torch.autograd.grad(ctx.outputs[index], owner.parameters[index],
                        gradient, allow_unused=True)
            for stream in owner.streams: main.wait_stream(stream)
            for row in values:
                for gradient in row:
                    if gradient is not None: gradient.record_stream(main)
            owner.backward_replays += 2
            # Fresh merged storage is essential: static Graph gradients are overwritten
            # on the next microbatch, whereas canonical .grad accumulates eight micros.
            return (None, None, None, *fixed_merge(torch, *values))

    class Graph:
        reference = False

        def __init__(self):
            self.runtime = self
            self.original = encoder_parameters(model)
            self.graphs = None; self.replicas = None; self.streams = ()
            self.parameters = (); self.key = None; self.builds = 0
            self.forward_replays = 0; self.backward_replays = 0; self.sync_rounds = 0
            self.capture_proof = []
            if not self.original or any(p.dtype != torch.float32 or p._backward_hooks for _, p in self.original):
                raise BackendFailure('Capture must precede canonical analytic hook installation')

        def begin_update(self, micro):
            if micro != 16 or self.graphs is None: raise BackendFailure('Prepared Graph16 required')

        def isolate_gradients(self): pass  # Bridge always returns fresh merged storage.

        def sync(self):
            with torch.no_grad():
                for replica in self.replicas:
                    replica.train(model.training)
                    for method in ('named_parameters', 'named_buffers'):
                        left, right = list(getattr(replica, method)()), list(getattr(model, method)())
                        if [n for n, _ in left] != [n for n, _ in right]:
                            raise BackendFailure('Replica parameter/buffer identity differs')
                        for (_, target), (_, source) in zip(left, right): target.copy_(source)
            self.sync_rounds += 1

        def prepare(self, inputs):
            self.capture_budget = budgets(model.config, inputs)
            if self.builds or any(p._backward_hooks for _, p in self.original):
                raise BackendFailure('One capture generation, outside canonical hooks required')
            self.builds = 1; rng = get_rng()
            try:
                mark('bf16_graph_replica_create', flush=True)
                self.replicas = tuple(copy.deepcopy(model) for _ in range(2))
                self.streams = tuple(torch.cuda.Stream() for _ in range(2))
                self.parameters = tuple(tuple(p for _, p in encoder_parameters(r)) for r in self.replicas)
                if any(tuple(n for n, _ in encoder_parameters(r)) != tuple(n for n, _ in self.original)
                       for r in self.replicas): raise BackendFailure('Encoder replica identity differs')
                self.sync(); graphs = []
                for index, (replica, value) in enumerate(zip(self.replicas, inputs)):
                    mark('bf16_graph_capture_' + str(index), flush=True)
                    with attention_route(torch, replica, True) as observed:
                        graphs.append(stream_graphed_encoder(torch,
                            lambda x, replica=replica: replica.encode(x, validated_inputs=True),
                            (value.detach().clone(),), self.parameters[index], self.streams[index], 2))
                    if (set(observed['qkv_dtypes']) != {'torch.bfloat16'} or
                            set(observed['output_dtypes']) != {'torch.bfloat16'} or
                            observed['layer_calls'] != [3]*4):
                        raise BackendFailure('BF16 capture coverage differs')
                    self.capture_proof.append(observed)
                self.graphs = tuple(graphs)
                self.key = tuple((tuple(v.shape), str(v.dtype), str(v.device)) for v in inputs)
            finally:
                torch.cuda.synchronize(); set_rng(rng)

        def encode(self, inputs):
            self.sync(); main = torch.cuda.current_stream(); outputs = []
            for index, value in enumerate(inputs):
                stream = self.streams[index]; stream.wait_stream(main)
                if index: stream.wait_stream(self.streams[index-1])
                with torch.cuda.stream(stream), torch.enable_grad():
                    value.record_stream(stream); outputs.append(self.graphs[index](value))
                self.forward_replays += 1
            for stream in self.streams: main.wait_stream(stream)
            for output in outputs: output.record_stream(main)
            return tuple(outputs)

        def forward(self, batch):
            inputs = (batch['inputs1'], batch['inputs2'])
            signature = tuple((tuple(v.shape), str(v.dtype), str(v.device)) for v in inputs)
            if self.graphs is None or signature != self.key:
                raise BackendFailure('No recapture, eager fallback or dynamic shape in this diagnostic')
            a, b = Bridge.apply(self, *inputs, *(p for _, p in self.original))
            return model.classify_pair(a, b)

        def status(self):
            return dict(graph_resident=self.graphs is not None, graph_builds=self.builds,
                forward_replays=self.forward_replays, backward_replays=self.backward_replays,
                replica_sync_rounds=self.sync_rounds, compiler_backend='none', inductor_fusion=False,
                capture_partition='per_document', canonical_gradient_hook='outside_capture',
                capture_proof=self.capture_proof, rng_budget=getattr(self, 'capture_budget', None))

        def clear(self):
            torch.cuda.synchronize(); self.graphs = None; self.replicas = None
            self.parameters = (); self.streams = (); gc.collect(); torch.cuda.empty_cache()

    return Graph()
