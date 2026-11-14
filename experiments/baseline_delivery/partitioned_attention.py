"""Two-layer FP32 Graph chains. Torch is supplied only by the native worker."""
from .torch_acceleration import BackendFailure
from .diagnostic_watch import mark


def stream_graphed_chain(torch,calls,statics,parameters,stream,warmup):
    """Capture F0,F1,B1,B0 in one document pool; preserve input VJPs."""
    if len(calls)!=2 or len(statics)!=2 or len(parameters)!=2:
        raise BackendFailure('Partitioned encoder requires two segments')
    surfaces=[];differentiable=[]
    for inputs,weights in zip(statics,parameters):
        indices=tuple(i for i,value in enumerate(inputs) if value.requires_grad)
        differentiable.append(indices)
        surfaces.append(tuple(inputs[i] for i in indices)+tuple(weights))
    if differentiable[0] or differentiable[1]!=(0,):
        raise BackendFailure('Expected token segment then differentiable hidden segment')
    main=torch.cuda.current_stream();stream.wait_stream(main)
    with torch.cuda.stream(stream),torch.enable_grad():
        for call,inputs,surface in zip(calls,statics,surfaces):
            for _ in range(warmup):
                output=call(*inputs)
                torch.autograd.grad(output,surface,torch.ones_like(output),allow_unused=True)
    torch.cuda.synchronize()
    pool=torch.cuda.graph_pool_handle()
    forwards=[torch.cuda.CUDAGraph() for _ in calls]
    backwards=[torch.cuda.CUDAGraph() for _ in calls]
    outputs=[];incoming=[];gradients=[None]*len(calls)
    for i,(call,inputs) in enumerate(zip(calls,statics)):
        mark('partition_forward_capture_'+str(i),flush=True)
        with torch.cuda.graph(forwards[i],pool=pool,stream=stream): outputs.append(call(*inputs))
        incoming.append(torch.empty_like(outputs[i]))
    for i in reversed(range(len(calls))):
        mark('partition_backward_capture_'+str(i),flush=True)
        with torch.cuda.graph(backwards[i],pool=pool,stream=stream):
            gradients[i]=torch.autograd.grad(outputs[i],surfaces[i],incoming[i],allow_unused=True)

    def wrap(index):
        inputs=statics[index];weights=parameters[index];indices=differentiable[index]
        forward=forwards[index];backward=backwards[index]
        output=outputs[index];gradient=incoming[index];result=gradients[index]
        class Segment(torch.autograd.Function):
            @staticmethod
            def forward(ctx,*args):
                for destination,value in zip(inputs,args[:len(inputs)]):
                    if destination.data_ptr()!=value.data_ptr(): destination.copy_(value)
                forward.replay()
                return output.detach()
            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx,value):
                if gradient.data_ptr()!=value.data_ptr(): gradient.copy_(value)
                backward.replay()
                input_gradients=[None]*len(inputs)
                for i,position in enumerate(indices):
                    input_gradients[position]=result[i].detach() if result[i] is not None else None
                return (*input_gradients,*(v.detach() if v is not None else None for v in result[len(indices):]))
        return lambda *values:Segment.apply(*values,*weights)
    first,second=(wrap(i) for i in range(len(calls)))
    return lambda tokens:second(first(tokens),tokens)


def partitioned_encoder(torch,replica,tokens,stream,warmup):
    """Keep the original named layers, FP32 masks, positions and pooling."""
    if len(replica.encoder.layers)!=4 or replica.encoder.norm is not None:
        raise BackendFailure('Partitioned candidate requires the pinned four-layer encoder')
    class Half(torch.nn.Module):
        def __init__(self,last): super().__init__();self.base=replica;self.last=last
        def forward(self,value,tokens=None):
            x=tokens if self.last else value
            valid=x.ne(1)
            if self.last: hidden=value
            else:
                hidden=self.base.embedding(x).masked_fill(~valid.unsqueeze(-1),0)
                hidden=hidden+self.base.position[:hidden.shape[1]]
            # Matches TransformerEncoder's bool-to-additive canonicalization.
            padding=torch.zeros_like(valid,dtype=hidden.dtype).masked_fill(~valid,float('-inf'))
            for index in ((2,3) if self.last else (0,1)):
                hidden=self.base.encoder.layers[index](hidden,src_key_padding_mask=padding,is_causal=False)
            if self.last:
                hidden=self.base.norm(hidden).masked_fill(~valid.unsqueeze(-1),0)
                return hidden.sum(1)/valid.sum(1).unsqueeze(-1)
            return hidden
    calls=tuple(torch.compile(Half(last),backend='eager',fullgraph=True,dynamic=False) for last in (False,True))
    first=[];second=[]
    for name,value in replica.named_parameters():
        if name.startswith('head.') or not value.requires_grad: continue
        target=first if name.startswith(('embedding.','encoder.layers.0.','encoder.layers.1.')) else second
        target.append(value)
    if not first or not second: raise BackendFailure('Incomplete partition parameter surface')
    static_tokens=tokens.detach().clone()
    mark('partition_compile_warmup',flush=True)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream),torch.enable_grad():
        hidden=calls[0](static_tokens).detach().requires_grad_(True)
    torch.cuda.synchronize()
    second_tokens=static_tokens.detach().clone()
    return stream_graphed_chain(torch,calls,((static_tokens,),(hidden,second_tokens)),
        (tuple(first),tuple(second)),stream,warmup),calls
