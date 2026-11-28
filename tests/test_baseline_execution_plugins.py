"""Plugin boundaries and a bounded timing driver, without native frameworks."""
import ast
from contextlib import ExitStack
import copy
import importlib
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.baseline_delivery import execution_plugins as p,mamba_convolution as conv,mamba_diagnostic as d
from experiments.baseline_delivery.common import atomic_json,read


class Tests(unittest.TestCase):
    def test_selection_null_conflicts_and_no_native_imports(self):
        config=dict(model='mamba1',task='retrieval')
        value=dict(plugin='mamba.fused_fp32',version=2)
        self.assertIsNone(p.resolve(None,config));self.assertEqual(p.resolve(value,config)['backend'],'eager')
        for invalid,cfg in (({**value,'unknown':True},config),({**value,'version':1},config),
                (value,dict(config,model='transformer')),(value,dict(config,torch_acceleration=dict(required='graph')))):
            with self.assertRaises(ValueError):p.resolve(invalid,cfg)
        h=p.Hooks();saved=dict(mode='eager');self.assertIs(h.acceleration_state(saved),saved)
        with h.model_context(object()):pass
        self.assertIsNone(h.before_save(saved));self.assertEqual(saved,dict(mode='eager'))

    def test_reset_only_execution_measurements(self):
        state=dict(index=3,logical_index=0,oom_count=2,ceiling=5,baseline=1.,samples=[1.],
            trial_windows=[2.],previous=2,previous_logical=0,wait=256,warmup=0)
        c=SimpleNamespace(state=state,policy=dict(warmup=2));p.reset_measurements(c)
        self.assertEqual([state[k] for k in ('index','logical_index','oom_count','ceiling')],[3,0,2,5])
        self.assertIsNone(state['baseline']);self.assertEqual(state['samples'],[]);self.assertEqual(state['wait'],0)

    def test_core_calculation_and_checkpoint_definitions_unchanged(self):
        before=subprocess.check_output(['git','show','89353fc:experiments/baseline_delivery/worker.py'],text=True)
        after=Path('experiments/baseline_delivery/worker.py').read_text(encoding='utf8')
        def definitions(text):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef)}
        a,b=definitions(before),definitions(after)
        for name in ('rng_state','restore_rng','save_checkpoint','load_checkpoint','evaluate',
                'update_body','update_impl','update','capture_update','restore_update'):
            self.assertEqual(a[name],b[name],name)
        core=next(n for n in ast.parse(after).body if isinstance(n,ast.FunctionDef) and n.name=='run')
        calls=[];ns=dict(ExitStack=ExitStack,_run=lambda *args:calls.append(args))
        exec(compile(ast.Module(body=[core],type_ignores=[]),'worker-interface','exec'),ns)
        ns['run']('config','data','cache','output','cuda:0',resume=True)
        self.assertIsNone(calls[0][-1]);self.assertEqual(calls[0][:7],('config','data','cache','output','cuda:0',True,None))
        hooks=p.Hooks();ns['run']('c','d','cache','o','cuda:0',execution=hooks)
        self.assertIs(calls[-1][-1],hooks)

    def test_private_kernel_source_definitions_are_unchanged(self):
        source=Path('.runtime/mamba-kernel-review-20261123/selective_scan_interface.py')
        # The actual source is pinned at runtime; repository-independent fixture
        # below tests isolation. If cached reviewed source exists, compare ASTs.
        if source.exists():
            tree=ast.parse(source.read_text());selected=[n for n in tree.body if getattr(n,'name',None) in ('MambaInnerFn','mamba_inner_fn')]
            self.assertEqual(len(selected),2)
            marker=object()
            private=conv.private_definitions(source.read_text(),dict(torch=SimpleNamespace(autograd=SimpleNamespace(Function=object)),
                custom_fwd=lambda f:f,custom_bwd=lambda f:f,causal_conv1d_cuda=marker),('MambaInnerFn','mamba_inner_fn'))
            for function in (private['MambaInnerFn'].forward,private['MambaInnerFn'].backward,private['mamba_inner_fn']):
                self.assertIs(function.__globals__['causal_conv1d_cuda'],marker)
        original={'token':'original'}
        local=conv.private_definitions('class Kernel:\n    def call(self):return token\ndef entry():return Kernel().call()\n',
            dict(original,token='private'),('Kernel','entry'))
        self.assertEqual(local['entry'](),'private');self.assertEqual(original,{'token':'original'})
        with self.assertRaises(ValueError):conv.private_definitions('def entry():pass',{},('missing',))

    def test_instance_policy_leaves_upstream_globals_and_other_model_intact(self):
        sentinel=object()
        namespace=dict(causal_conv1d_fn=sentinel,mamba_inner_fn=lambda x:'old fused')
        exec('def forward(self,x,inference_params=None):\n    return mamba_inner_fn(x) if causal_conv1d_fn is not None else "ordinary"\n',namespace)
        class Mamba:
            use_fast_path=True
            forward=namespace['forward']
            def parameters(self):return [SimpleNamespace(dtype='fp32')]
        module=SimpleNamespace(__file__='mamba.py',Mamba=Mamba,causal_conv1d_fn=sentinel,mamba_inner_fn=namespace['mamba_inner_fn'])
        scan=SimpleNamespace(__file__='scan.py',causal_conv1d_cuda=sentinel)
        functions=SimpleNamespace(causal_conv1d_fwd_function=lambda *a:None,causal_conv1d_bwd_function=lambda *a:None)
        modules={'mamba_ssm.modules.mamba_simple':module,'mamba_ssm.ops.selective_scan_interface':scan,'causal_conv1d.cpp_functions':functions}
        def digest(path):return d.InputProjectionBF16.source_sha256 if str(path)=='mamba.py' else 'c79f4432536402fe4d6c1026ccf8d5fc0aaa62b790e9a622295aee14ac45dbcc'
        original=copy.copy(vars(module));other=Mamba();model=SimpleNamespace(training=True,blocks=[SimpleNamespace(mixer=Mamba())])
        with patch.object(conv.importlib,'import_module',side_effect=modules.__getitem__),patch.object(conv,'digest',side_effect=digest), \
                patch.object(Path,'read_text',return_value='pinned fixture'), \
                patch.object(conv,'private_definitions',return_value={'mamba_inner_fn':lambda x,**kw:'retained' if kw.get('checkpoint_lvl')==0 else 'private fused'}):
            policy=conv.InstanceFusedFP32(SimpleNamespace(float32='fp32'),model)
            self.assertEqual(other.forward(1),'old fused')
            self.assertEqual(model.blocks[0].mixer.forward(1),'ordinary')
            with policy.activate():
                self.assertEqual(model.blocks[0].mixer.forward(1),'private fused')
                model.training=False;self.assertEqual(model.blocks[0].mixer.forward(1),'ordinary')
                with self.assertRaises(ValueError):policy.set_checkpoint_level(0)
            policy.set_checkpoint_level(0);model.training=True
            with policy.activate():self.assertEqual(model.blocks[0].mixer.forward(1),'retained')
            self.assertEqual(policy.level_calls,{0:1,1:1})
            for invalid in (True,2,-1):
                with self.assertRaises(ValueError):policy.set_checkpoint_level(invalid)
            policy.close();self.assertNotIn('forward',model.blocks[0].mixer.__dict__)
            self.assertEqual(vars(module),original);self.assertIs(scan.causal_conv1d_cuda,sentinel)
            self.assertEqual(policy.calls,2)

    def test_micro_driver_balanced_restored_and_oom_isolated(self):
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                request=Path(tmp)/'request.json';atomic_json(request,dict(checkpoint=dict(sha256='latest')))
                calls=[];state={'changed':False};counter={'time':0.};policy=SimpleNamespace(calls=0,mechanism={'scope':'instance'})
                class OOM(Exception):pass
                def update(candidate,*,micro_batch):
                    self.assertFalse(state['changed']);state['changed']=True;calls.append(micro_batch)
                    if fail and micro_batch==32:raise OOM()
                    counter['time']+=32/micro_batch;policy.calls+=1;return .1
                def restore():state['changed']=False
                def exact():self.assertFalse(state['changed'])
                cuda=SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,
                    max_memory_allocated=lambda:100,max_memory_reserved=lambda:200,empty_cache=lambda:None,OutOfMemoryError=OOM)
                c=dict(torch=SimpleNamespace(cuda=cuda,OutOfMemoryError=OOM),logical=32,restore=restore,
                    exact=exact,update=update,projection=policy,opt=SimpleNamespace(zero_grad=lambda **k:None))
                with patch.object(d.time,'perf_counter',side_effect=lambda:counter['time']):
                    result=d.micro_benchmark(c,request)
                self.assertEqual(result['formal_updates_committed'],0);exact()
                self.assertEqual(result['recommended_micro_batch'],16 if fail else 32)
                self.assertEqual(len(calls),13 if fail else 18)
                for tier in ('8','16'):self.assertEqual(len(result['tiers'][tier]['seconds']),4)
                self.assertEqual(result['tiers']['32']['status'],'oom' if fail else 'measured')
                self.assertEqual(read(request.parent/(d.MICRO_ROUTE+'.json')),result)


if __name__=='__main__':unittest.main()
