"""Stdlib-only tests for MEGA refinement boundaries; no native imports."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.baseline_delivery import mega_refinement as mod,mega_diagnostic as diag
from experiments.baseline_delivery.common import digest,atomic_json


class Tests(unittest.TestCase):
    def test_auto_job_skips_finished_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);states={job:dict(state='pending') for job in diag.JOBS}
            states['mega-listops-s42']['state']='computed'
            states['mega-retrieval-s42']['state']='paused'
            for task in ('listops','image','retrieval'):
                atomic_json(root/'jobs'/('mega-'+task+'-s42')/'checkpoints/latest.json',{})
            atomic_json(root/'status.json',states)
            self.assertEqual(diag.select_job(root),'mega-image-s42')
            atomic_json(root/'jobs/mega-image-s42/summary.json',{})
            self.assertEqual(diag.select_job(root),'mega-retrieval-s42')
            self.assertEqual(diag.select_job(root,'mega-retrieval-s42'),'mega-retrieval-s42')
            with self.assertRaisesRegex(ValueError,'auto'):diag.select_job(root,'mega-listops-s42')
            states['mega-retrieval-s42']['state']='computed';atomic_json(root/'status.json',states)
            self.assertIsNone(diag.select_job(root))
            states['mega-text-s42']['state']='running';atomic_json(root/'status.json',states)
            with self.assertRaisesRegex(ValueError,'Pause'):diag.select_job(root)

    def test_no_eligible_job_skips_native_and_upload_and_releases_locks(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/'status.json',{j:dict(state='computed') for j in diag.JOBS})
            args=SimpleNamespace(source_suite=root,job='auto',cache=root,gpu='0')
            with ExitStack() as stack:
                stack.enter_context(patch.object(diag,'ROOT',root));stack.enter_context(patch.object(diag,'revision',return_value='pinned'))
                stack.enter_context(patch('experiments.baseline_delivery.bootstrap.check_exchange'))
                stack.enter_context(patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})))
                stack.enter_context(patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}))
                run=stack.enter_context(patch('experiments.baseline_delivery.diagnostic_watch.Supervisor.run'))
                upload=stack.enter_context(patch('experiments.baseline_delivery.return_artifacts.upload_declared'))
                self.assertEqual(diag.campaign(args),0);run.assert_not_called();upload.assert_not_called()
            from experiments.baseline_delivery.common import lock
            with lock(root/'suite.lock'):pass

    def test_all_routes_use_real_supervisor_phase_contract(self):
        from experiments.baseline_delivery.diagnostic_watch import Recorder
        with tempfile.TemporaryDirectory() as tmp:
            recorder=Recorder(tmp,'fixture')
            for route in diag.METHODS+diag.STACKS+diag.REFINEMENTS+diag.GRAPH_REPAIR:
                for stage in ('reference_a','reference_b','candidate','graph_capture','graph_replay','timing_0'):
                    name=diag.phase_name(route,stage)
                    recorder.mark(name,flush=True)
                    self.assertEqual(recorder.phase,name)
            with self.assertRaises(ValueError):diag.phase_name('fft','invalid-stage')

    def test_precision_targets_and_restore(self):
        class Value:
            def float(self):return 'fp32-output'
        class Linear:
            def forward(self,*a,**k):return Value()
        names=['encoder.layers.0.nffn.fc1','encoder.layers.1.nffn.fc2','encoder.layers.0.mega_layer.h_proj','head']
        modules={n:Linear() for n in names}
        model=SimpleNamespace(named_modules=lambda:list(modules.items()))
        self.assertEqual(mod.linear_names(model,'tail'),[names[1]])
        self.assertEqual(mod.linear_names(model,'ffn'),names[:2])
        self.assertEqual(mod.linear_names(model,'projections'),names[:3])
        calls=[]
        torch=SimpleNamespace(bfloat16='bf16',autocast=lambda *a,**k:(calls.append(k) or nullcontext()))
        plugin=mod.PrecisionIslands(model,'ffn',torch)
        self.assertIsInstance(modules[names[0]].forward(),Value)
        with plugin.activate():
            self.assertEqual(modules[names[0]].forward(),'fp32-output')
            self.assertIsInstance(modules['head'].forward(),Value)
        self.assertEqual(plugin.calls,1);self.assertFalse(calls[0]['cache_enabled'])
        plugin.close();self.assertTrue(all('forward' not in m.__dict__ for m in modules.values()))

    def test_padding_fix_has_no_scalar_read_and_preserves_return_shape(self):
        source='''class Encoder:
    traceable=False
    tpu=False
    def forward(self,padding_mask):
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask=None
        return [padding_mask], 'pooled'
'''
        class Mask:
            def any(self):raise RuntimeError('GPU scalar branch')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'encoder.py';path.write_text(source,encoding='utf8')
            spec=importlib.util.spec_from_file_location('fake_refine_encoder',path);m=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=m;self.addCleanup(sys.modules.pop,spec.name,None);spec.loader.exec_module(m)
            encoder=m.Encoder();other=m.Encoder();original=m.Encoder.forward;mask=Mask()
            with patch.object(mod,'ENCODER_SHA',digest(path)):plugin=mod.GraphPadding(encoder)
            with self.assertRaises(RuntimeError):encoder.forward(mask)
            with plugin.activate():
                out=encoder.forward(mask);self.assertIs(out[0][0],mask);self.assertEqual(out[1],'pooled')
                with self.assertRaises(RuntimeError):other.forward(mask)
            self.assertIs(m.Encoder.forward,original);self.assertFalse(encoder.traceable)
            plugin.close();self.assertNotIn('forward',encoder.__dict__)

    def test_kernel_grid_caches_only_constant_and_refuses_build_in_capture(self):
        calls=[];capture=[False]
        class Grid:
            def to(self,p):calls.append(('to',p.device,p.dtype));return self
        plugin=object.__new__(mod.KernelGrid);plugin.cache={};plugin.calls=0;plugin.builds=0
        plugin.torch=SimpleNamespace(arange=lambda n:(calls.append(('arange',n)) or Grid()),
            cuda=SimpleNamespace(is_current_stream_capturing=lambda:capture[0]))
        p=SimpleNamespace(device='gpu',dtype='fp32')
        a=plugin.grid(2048,p);capture[0]=True
        self.assertIs(plugin.grid(2048,p),a);self.assertEqual(plugin.builds,1)
        with self.assertRaises(ValueError):plugin.grid(4096,p)
        self.assertEqual(calls,[('arange',2048),('to','gpu','fp32')])

    def test_kernel_grid_instance_override_keeps_live_parameter_calculation(self):
        from experiments.baseline_delivery import mega_fft
        source='''class MultiHeadEMA:
    def _compute_kernel(self,length):
        p=self.p
        grid=torch.arange(length).to(p)
        return grid,self.weight
'''
        class Grid:
            def to(self,p):return self
        calls=[]
        torch=SimpleNamespace(arange=lambda n:(calls.append(n) or Grid()),
            cuda=SimpleNamespace(is_current_stream_capturing=lambda:False))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ema_grid.py';path.write_text(source,encoding='utf8')
            spec=importlib.util.spec_from_file_location('fake_grid',path);m=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=m;self.addCleanup(sys.modules.pop,spec.name,None);spec.loader.exec_module(m);m.torch=torch
            layer=m.MultiHeadEMA();layer.p=SimpleNamespace(device='gpu',dtype='fp32');layer.weight=1
            model=SimpleNamespace(modules=lambda:[layer])
            with patch.object(mega_fft,'SOURCE_SHA',digest(path)):plugin=mod.KernelGrid(model,torch)
            layer._compute_kernel(8)
            with plugin.activate():
                a,w=layer._compute_kernel(8);self.assertEqual(w,1);layer.weight=2
                b,w=layer._compute_kernel(8);self.assertIs(a,b);self.assertEqual(w,2)
            layer._compute_kernel(8);self.assertEqual(calls,[8,8,8])
            plugin.close();self.assertNotIn('_compute_kernel',layer.__dict__)

    def test_partial_graph_is_instance_local_and_reference_stays_eager(self):
        calls=[];phases=[]
        class Module:
            def __init__(self):pass
        class FFN:
            def forward(self,x):calls.append(('eager',x));return x+1
            def parameters(self):return ['weight']
        layer=FFN();other=FFN();captures=[]
        def make(adapter,args):
            captures.append((adapter,args));adapter.forward(*args)
            return lambda x:(calls.append(('graph',x)) or x+1)
        cuda=SimpleNamespace(get_rng_state=lambda:SimpleNamespace(clone=lambda:'saved'),synchronize=lambda:None,
            stream=lambda s:nullcontext(),Stream=lambda:None,set_rng_state=lambda s:calls.append(('rng',s)),make_graphed_callables=make)
        torch=SimpleNamespace(cuda=cuda,nn=SimpleNamespace(Module=Module,ParameterList=list))
        model=SimpleNamespace(named_modules=lambda:[('encoder.layers.0.nffn',layer)])
        plugin=mod.FFNGraph(model,torch,phases.append)
        self.assertEqual(layer.forward(1),2)
        with plugin.activate():
            self.assertEqual(layer.forward(2),3);self.assertEqual(layer.forward(3),4)
            self.assertEqual(other.forward(4),5)
        self.assertEqual(layer.forward(5),6);self.assertEqual(len(captures),1)
        self.assertEqual(captures[0][0].params,['weight']);self.assertEqual(plugin.calls,2)
        self.assertEqual(phases,['ffn_graph_capture']);plugin.close();self.assertNotIn('forward',layer.__dict__)

    def test_native_trace_excludes_messages_and_local_paths(self):
        try:exec(compile("raise RuntimeError('private content')",'/hidden/site-packages/torch/cuda/graphs.py','exec'))
        except RuntimeError as error:frames=mod.native_trace(error)
        self.assertEqual(frames,[dict(file='torch/cuda/graphs.py',line=1,function='<module>')])

    def test_refinement_routes_and_allowlisted_failure(self):
        for route in diag.REFINEMENTS+diag.GRAPH_REPAIR:
            diag.settings(route);spec=diag.child_spec(Path('/fixture'),route)
            self.assertEqual(spec['argv'][-1],route)
        for route in ('bf16+island-ffn','island-tail+island-ffn'):
            with self.assertRaises(ValueError):diag.settings(route)
        self.assertEqual(mod.error_category(RuntimeError('operation not permitted when stream is capturing')),'capture_forbidden_operation')
        self.assertEqual(mod.error_category(RuntimeError('secret arbitrary content')),'unclassified_native_error')
        self.assertEqual(mod.error_category(RuntimeError('tensor appears to not have been used in graph')),'unused_autograd_input')

if __name__=='__main__':unittest.main()
