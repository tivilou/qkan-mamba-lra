"""Opt-in FP32 Retrieval document concurrency; framework supplied by native worker.

Encoder replicas have disjoint gradient storage. A single autograd bridge merges
their gradients on the caller stream in a fixed order into the original model.
No optimizer, canonical parameter, checkpoint key or attention kernel is replaced.
"""
import copy
import gc

from .torch_acceleration import BackendFailure,CaptureCapacity,backend_diagnostic,fatal_cuda,graph_rng_budget
from .diagnostic_watch import mark,cuda_fence


def encoder_parameters(model):
    return tuple((name,p) for name,p in model.named_parameters()
        if not name.startswith('head.') and p.requires_grad)


def fixed_merge(torch,left,right):
    """Document2 + document1, with fresh storage even when one side is unused."""
    if len(left)!=len(right): raise BackendFailure('Replica gradient structure differs')
    return tuple(b+a if a is not None and b is not None else
        (a.clone() if a is not None else b.clone() if b is not None else None)
        for a,b in zip(left,right))


def stream_graphed_encoder(torch,call,static,parameters,stream,warmup):
    """One FP32 encoder, explicit capture/replay stream and private graph pool.

    Unlike make_graphed_callables' shared default capture stream, each document
    captures its library workspace on its own persistent stream. Integer token
    inputs have no gradient; only the supplied replica parameters are trainable.
    Forward/backward share this document's pool and execute in that order.
    """
    if len(static)!=1 or static[0].requires_grad:
        raise BackendFailure('Document Graph requires one non-differentiable token input')
    main=torch.cuda.current_stream();stream.wait_stream(main)
    with torch.cuda.stream(stream),torch.enable_grad():
        for _ in range(warmup):
            output=call(*static)
            torch.autograd.grad(output,parameters,torch.ones_like(output),allow_unused=True)
    torch.cuda.synchronize()
    pool=torch.cuda.graph_pool_handle()
    forward=torch.cuda.CUDAGraph();backward=torch.cuda.CUDAGraph()
    with torch.cuda.graph(forward,pool=pool,stream=stream):
        output=call(*static)
    gradient=torch.empty_like(output)
    with torch.cuda.graph(backward,pool=pool,stream=stream):
        gradients=torch.autograd.grad(output,parameters,gradient,allow_unused=True)

    class GraphedEncoder(torch.autograd.Function):
        @staticmethod
        def forward(ctx,value,*replica_parameters):
            if value.data_ptr()!=static[0].data_ptr(): static[0].copy_(value)
            forward.replay()
            return output.detach()
        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx,incoming):
            if gradient.data_ptr()!=incoming.data_ptr(): gradient.copy_(incoming)
            backward.replay()
            return (None,*(g.detach() if g is not None else None for g in gradients))
    return lambda value:GraphedEncoder.apply(value,*parameters)


def parallel_runtime(torch,model,get_rng,set_rng,config):
    shape=getattr(model,'config',{})
    if (shape.get('model')!='transformer' or shape.get('task')!='retrieval'
            or config['mode']!='compile_graph' or config['precision']!='fp32'):
        raise BackendFailure('Parallel document candidate supports FP32 Transformer Retrieval Graph only')

    class Bridge(torch.autograd.Function):
        @staticmethod
        def forward(ctx,owner,x1,x2,*parameters):
            ctx.owner=owner
            ctx.outputs=owner.encode_pair((x1,x2))
            # Keep the inner graphs separate from the bridge's outer edges.
            return tuple(value.detach() for value in ctx.outputs)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx,g1,g2):
            owner=ctx.owner;main=torch.cuda.current_stream();values=[None,None]
            # Independent static gradients/pools: no concurrent AccumulateGrad
            # into shared model parameters, no allocator-reused gradient alias.
            for index,gradient in ((1,g2),(0,g1)):
                mark('document_backward_stream_wait',document=index,flush=owner.backward_replays==0)
                stream=owner.streams[index];stream.wait_stream(main)
                with torch.cuda.stream(stream):
                    output=ctx.outputs[index]
                    if gradient is None: gradient=torch.zeros_like(output)
                    # The incoming head gradient was allocated on the caller
                    # stream. Keep its storage alive through the side-stream copy.
                    gradient.record_stream(stream)
                    mark('document_backward_replay',document=index,flush=owner.backward_replays==0)
                    values[index]=torch.autograd.grad(output,owner.parameters[index],gradient,
                        allow_unused=True)
                    cuda_fence(torch,'document_backward_complete_wait',document=index)
            mark('backward_join')
            for stream in owner.streams: main.wait_stream(stream)
            for gradients in values:
                for gradient in gradients:
                    if gradient is not None: gradient.record_stream(main)
            mark('gradient_merge')
            owner.backward_replays+=2
            merged=fixed_merge(torch,*values)
            cuda_fence(torch,'gradient_merge_complete_wait')
            return (None,None,None,*merged)

    class Runtime:
        def __init__(self):
            self.mode=None;self.expected_rows=None;self.graph=None;self.key=None
            self.replicas=None;self.parameters=None;self.streams=None;self.compiled=None
            self.builds=0;self.replays=0;self.backward_replays=0;self.warmups=0;self.tails=0
            self.stage='idle';self.capture_budget=None
            self.original=encoder_parameters(model)
            if not self.original or any(p.dtype!=torch.float32 for _,p in self.original):
                raise BackendFailure('Parallel candidate requires FP32 encoder parameters')
        def set_mode(self,mode): self.mode=mode
        def begin_update(self,micro_batch): self.expected_rows=micro_batch
        def generation(self): return self.builds
        def status(self):
            return dict(graph_builds=self.builds,graph_resident=self.graph is not None,
                compiled_callable_created=self.compiled is not None,graph_replays=self.replays,
                backward_graph_replays=self.backward_replays,eager_tail_calls=self.tails,
                stage=self.stage,compiler_warmups=self.warmups,graph_callables=2 if self.graph else 0,
                capture_partition='parallel_retrieval_documents',capture_budget=self.capture_budget,
                execution_streams=2,encoder_replicas=2,gradient_merge='document2_plus_document1_fixed_order',
                forward_schedule='document1_then_document2_shared_rng',backward_schedule='independent_streams',
                capture_streams='per_document_explicit',library_workspace_scope='capture_stream_per_document',
                compiler_backend='eager',inductor_fusion=False)
        def clear(self):
            mark('graph_cleanup',flush=True)
            # Complete side-stream accesses before releasing any static buffers.
            if self.streams:
                main=torch.cuda.current_stream()
                for stream in self.streams: main.wait_stream(stream)
                torch.cuda.synchronize()
            self.graph=None;self.key=None;self.compiled=None
            self.replicas=None;self.parameters=None;self.streams=None
            gc.collect();torch.cuda.empty_cache()
        def isolate_gradients(self): pass  # Bridge returns fresh merged gradients.
        def backend_error(self,error):
            return type(error).__module__.startswith(('torch._dynamo','torch._inductor'))
        def attempt(self,stage,callback):
            self.stage=stage
            try: return callback()
            except Exception as error:
                if fatal_cuda(error): raise
                current=error;seen=set()
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    if isinstance(current,getattr(torch.cuda,'OutOfMemoryError',())):
                        if current is error: raise
                        raise current from None
                    current=current.__cause__
                diagnostic=backend_diagnostic(error,stage,'eager_parallel')
                raise BackendFailure(type(error).__name__,diagnostic) from error
        def sync_replicas(self):
            with torch.no_grad():
                for replica in self.replicas:
                    replica.train(model.training)
                    for (name,target),(original,source) in zip(replica.named_parameters(),model.named_parameters()):
                        if name!=original: raise BackendFailure('Replica parameter identity differs')
                        target.copy_(source)
                    for (name,target),(original,source) in zip(replica.named_buffers(),model.named_buffers()):
                        if name!=original: raise BackendFailure('Replica buffer identity differs')
                        target.copy_(source)
        def capture(self,inputs):
            self.clear()
            if self.builds>=config['max_builds']: raise BackendFailure('Parallel graph build budget exhausted')
            self.builds+=1
            mark('replica_create',flush=True)
            self.replicas=tuple(copy.deepcopy(model) for _ in range(2))
            self.parameters=tuple(tuple(p for _,p in encoder_parameters(replica)) for replica in self.replicas)
            if any(tuple(n for n,_ in encoder_parameters(replica))!=tuple(n for n,_ in self.original)
                    for replica in self.replicas):
                raise BackendFailure('Replica encoder parameter identity differs')
            mark('replica_initial_sync',flush=True)
            self.streams=tuple(torch.cuda.Stream() for _ in range(2));self.sync_replicas()
            mark('compile_callables',flush=True)
            self.compiled=tuple(torch.compile(replica.encode,backend='eager',fullgraph=True,dynamic=False)
                for replica in self.replicas)
            graphs=[];rng=get_rng()
            try:
                for index,value in enumerate(inputs):
                    compiled=self.compiled[index];replica=self.replicas[index]
                    class Encoder(torch.nn.Module):
                        def __init__(self,base,call): super().__init__();self.base=base;self.call=call
                        def forward(self,x): return self.call(x,validated_inputs=True)
                    adapter=Encoder(replica,compiled);static=(value.detach().clone(),)
                    # Compile and warm native backward before graph capture. The
                    # warmup is excluded from admission timing and RNG progression.
                    mark('compile_forward',document=index,flush=True)
                    output=compiled(*static,validated_inputs=True)
                    mark('warmup_backward',document=index,flush=True)
                    gradients=torch.autograd.grad(output,self.parameters[index],torch.ones_like(output),allow_unused=True)
                    del gradients,output
                    mark('warmup_synchronize',document=index,flush=True)
                    torch.cuda.synchronize();torch.cuda.empty_cache();self.warmups+=1
                    # Separate calls allocate independent pools: both documents'
                    # activations must coexist through their backwards.
                    mark('graph_capture',document=index,flush=True)
                    graphs.append(stream_graphed_encoder(torch,adapter,static,self.parameters[index],
                        self.streams[index],config['graph_warmup']))
                self.graph=tuple(graphs)
            finally:
                mark('capture_finalize_synchronize',flush=True)
                torch.cuda.synchronize();set_rng(rng)
            mark('capture_ready',flush=True)
        def encode_pair(self,inputs):
            mark('replica_refresh')
            self.sync_replicas();main=torch.cuda.current_stream();outputs=[]
            # Both graphs use the default generator's shared device seed/offset
            # tensors (PyTorch v2.7.1 replay_prologue). Host submission order alone
            # cannot protect them across streams: finish document1 before the
            # second graph overwrites that state. Backward graphs remain parallel.
            for index,value in enumerate(inputs):
                mark('document_forward_stream_wait',document=index)
                stream=self.streams[index];stream.wait_stream(main)
                if index: stream.wait_stream(self.streams[index-1])
                with torch.cuda.stream(stream),torch.enable_grad():
                    value.record_stream(stream)
                    mark('document_forward_replay',document=index)
                    outputs.append(self.graph[index](value))
                    cuda_fence(torch,'document_forward_complete_wait',document=index)
                self.replays+=1
            mark('forward_join')
            for stream in self.streams: main.wait_stream(stream)
            for output in outputs: output.record_stream(main)
            return tuple(outputs)
        def documents(self,batch):
            inputs=(batch['inputs1'],batch['inputs2'])
            if self.expected_rows is not None and inputs[0].shape[0]<self.expected_rows:
                self.tails+=1;return model(batch,validated_inputs=True)
            budgets=[graph_rng_budget(shape,value.shape[0],value.shape[1]) for value in inputs]
            self.capture_budget=max((v for v in budgets if v is not None),key=lambda v:v['estimated_increment'],default=None)
            if self.capture_budget is not None and not self.capture_budget['passed']:
                self.stage='capture_budget';raise CaptureCapacity(self.capture_budget)
            signature=tuple((tuple(v.shape),str(v.dtype),str(v.device)) for v in inputs)
            if signature!=self.key:
                self.attempt('parallel_graph_capture',lambda:self.capture(inputs));self.key=signature
            a,b=self.attempt('parallel_graph_forward',lambda:Bridge.apply(self,*inputs,*(p for _,p in self.original)))
            self.stage='ready';return model.classify_pair(a,b)
        def forward(self,batch,mode):
            if mode=='eager': return model(batch,validated_inputs=True)
            if mode!='compile_graph': raise BackendFailure('Parallel candidate requires document CUDA Graphs')
            return self.documents(batch)
    return Runtime()
