"""Optional MEGA training policy; the stable worker owns updates and persistence."""
from contextlib import ExitStack


class Policy:
    def __init__(self,torch,model,graph=False):
        from .mega_fft import PowerTwoEMA
        from .mega_refinement import PrecisionIslands,FFNGraph
        self.model=model;self.graph=graph;self.calls=0;self.original=model.forward
        self.owned='forward' in model.__dict__
        self.fft=PowerTwoEMA(model);self.islands=None;self.ffn=None
        try:
            self.islands=PrecisionIslands(model,'tail',torch)
            self.ffn=FFNGraph(model,torch,lambda name:None) if graph else None
        except BaseException:
            if self.islands:self.islands.close()
            self.fft.close();raise
        def forward(*args,**kwargs):
            training=bool(model.training)
            if training:self.calls+=1
            with ExitStack() as stack:
                stack.enter_context(self.fft.activate(training))
                stack.enter_context(self.islands.activate(training))
                if self.ffn:stack.enter_context(self.ffn.activate(training))
                return self.original(*args,**kwargs)
        model.forward=forward

    def status(self):
        return dict(plugin='mega.fft_tail_bf16',version=1,graph=self.graph,
            training_calls=self.calls,fft_calls=self.fft.calls,island_calls=self.islands.calls,
            graph_captures=self.ffn.captures if self.ffn else 0,graph_replays=self.ffn.calls if self.ffn else 0,
            evaluation='original_fp32',precision='tail FFN Linear BF16; FP32 master/Adam and all other operations')

    def __enter__(self):return self

    def close(self):
        if self.owned:self.model.forward=self.original
        else:del self.model.forward
        if self.ffn:self.ffn.close()
        self.islands.close();self.fft.close()

    def __exit__(self,*args):self.close()
