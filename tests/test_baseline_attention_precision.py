"""Bounded control and injected-runtime tests; no native framework execution."""
from contextlib import contextmanager
import copy
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery import attention_precision as probe,attention_precision_diagnostic as entry
from experiments.baseline_delivery import attention_precision_native as native,optimization_plugins as plugins
from experiments.baseline_delivery.common import ROOT,atomic_json,digest,fingerprint,read

PRESET=ROOT/'experiments/baseline_delivery/optimization-presets/bf16-attention-probe.json'
def selected(): return plugins.resolve(read(PRESET),scope='attention_probe')


class Tensor:
    shape=(2,4,16,32)
    grad=None
    def __init__(self,value=1,dtype='fp32'): self.value,self.dtype=value,dtype
    def detach(self): return self
    def clone(self): return Tensor(self.value,self.dtype)
    def to(self,dtype): return Tensor(self.value,dtype)


class Runtime:
    float32='fp32';bfloat16='bf16'
    def __init__(self):
        self.cpu=0;self.gpu=0;self.dropouts=[];self.events=[]
        self.cuda=NS(is_bf16_supported=lambda:True,get_rng_state=lambda:Tensor(self.gpu),
            set_rng_state=lambda x:setattr(self,'gpu',x.value),synchronize=lambda:None,
            reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:64,
            max_memory_reserved=lambda:128,memory_allocated=lambda:32)
        self.autograd=NS(grad=lambda output,inputs,grad_outputs:tuple(Tensor(x.value) for x in inputs))
        self.profiler=NS(ProfilerActivity=NS(CPU='cpu',CUDA='cuda'),profile=self.profile)
        self.keys=['aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward']
    def get_rng_state(self): return Tensor(self.cpu)
    def set_rng_state(self,x): self.cpu=x.value
    def isfinite(self,x): return NS(all=lambda:True)
    def equal(self,a,b): return a.value==b.value and a.dtype==b.dtype
    @contextmanager
    def profile(self,**kwargs):
        yield NS(key_averages=lambda:[NS(key=key) for key in self.keys])
    def forward(self,torch,selection,qkv,mask,dropout):
        self.dropouts.append(dropout)
        if dropout: self.gpu+=1
        return Tensor(qkv[0].value+self.gpu,'bf16' if selection.bf16_amp else 'fp32')


class PrecisionTests(unittest.TestCase):
    def test_scope_default_off_and_explicit_probe_only(self):
        self.assertFalse(plugins.resolve().bf16_amp)
        with self.assertRaisesRegex(ValueError,'pending'): plugins.resolve(read(PRESET))
        self.assertTrue(selected().bf16_amp)
        self.assertEqual(selected().manifest()['precision'],'bf16_autocast_fp32_inputs')
        self.assertFalse(selected().manifest()['baseline_adapter'])
        value=read(PRESET);value['optimizations']['padding_trim']['enabled']=True
        with self.assertRaisesRegex(ValueError,'full token'): plugins.resolve(value,scope='attention_probe')
        with self.assertRaises(ValueError): plugins.resolve(scope='typo')

    def test_null_precision_context_does_not_touch_runtime(self):
        runtime=Mock()
        with probe.precision_context(runtime,plugins.resolve()): pass
        self.assertEqual(runtime.mock_calls,[])
        with self.assertRaises(ValueError):
            probe.precision_context(runtime,plugins.Selection(bf16_amp=True))

    def test_forward_forces_kernel_casts_and_exits_contexts(self):
        events=[];inputs=[Tensor() for _ in range(3)];seen=[]
        @contextmanager
        def context(name):
            events.append('enter '+name)
            try: yield
            finally: events.append('exit '+name)
        def sdpa(*args,**kwargs):
            seen.append((args,kwargs));return Tensor(dtype=args[0].dtype)
        runtime=NS(bfloat16='bf16',autocast=Mock(return_value=context('autocast')),
            nn=NS(attention=NS(sdpa_kernel=Mock(return_value=context('efficient')),
                              SDPBackend=NS(EFFICIENT_ATTENTION='efficient')),
                  functional=NS(scaled_dot_product_attention=sdpa)))
        result=probe.forward(runtime,selected(),inputs,'mask',.1)
        self.assertEqual(result.dtype,'bf16');self.assertTrue(all(x.dtype=='fp32' for x in inputs))
        self.assertEqual(seen[0][1],dict(attn_mask='mask',dropout_p=.1,is_causal=False))
        self.assertEqual(events,['enter autocast','enter efficient','exit efficient','exit autocast'])
        self.assertEqual(runtime.autocast.call_args.kwargs,dict(device_type='cuda',dtype='bf16'))
        self.assertEqual(runtime.nn.attention.sdpa_kernel.call_args.args,('efficient',))

    def test_balanced_times_order_sync_boundaries_and_medians(self):
        events=[];ticks=iter([0,2,2,3,3,4,4,6,6,8,8,9,9,10,10,12])
        value=probe.balanced_times(lambda r:events.append(r),lambda:events.append('restore'),
            lambda:events.append('sync'),clock=lambda:next(ticks))
        self.assertEqual(value['order'],['fp32','bf16','bf16','fp32']*2)
        self.assertEqual(value['fp32_median_seconds'],2)
        self.assertEqual(value['throughput_gain_fraction'],1)
        self.assertEqual(events[:4],['restore','sync','fp32','sync'])
        with self.assertRaises(ValueError): probe.balanced_times(lambda _:None,lambda:None,lambda:None,clock=lambda:0)

    def run_probe(self,runtime=None,metrics=None):
        runtime=runtime or Runtime();qkv=[Tensor() for _ in range(3)]
        metrics=metrics or dict(finite=True,relative_l2=.001,max_absolute=.003)
        with patch.object(probe,'forward',side_effect=runtime.forward),patch.object(probe,'numerical_metrics',return_value=metrics):
            result=probe.native_probe(runtime,qkv,'mask',Tensor(),selected(),.1)
        self.assertEqual(runtime.gpu,0);self.assertEqual(runtime.cpu,0)
        return result,runtime,qkv

    def test_probe_success_proves_kernel_dtypes_rng_and_rollback(self):
        result,runtime,qkv=self.run_probe()
        self.assertTrue(result['mechanism_passed'])
        self.assertEqual(result['rollback'],dict(inputs_exact=True,rng_exact=True,leaf_gradients_unmodified=True))
        self.assertEqual(result['kernel_proof']['bf16']['incremental_peak_allocated_bytes'],32)
        self.assertEqual(result['same_route_replay']['bf16']['output_dtype'],'bf16')
        self.assertEqual(runtime.dropouts[:2],[0.,0.]);self.assertTrue(all(p==.1 for p in runtime.dropouts[2:]))
        self.assertTrue(all(x.grad is None for x in qkv))

    def test_numeric_rejection_has_no_benchmark_but_has_rollback(self):
        result,runtime,_=self.run_probe(metrics=dict(finite=True,relative_l2=.1,max_absolute=1))
        self.assertFalse(result['mechanism_passed']);self.assertIsNone(result['timing'])
        self.assertTrue(result['rollback']['rng_exact']);self.assertEqual(len(runtime.dropouts),6)

    def test_wrong_backend_or_missing_backward_fails_and_restores_rng(self):
        for keys in ([],['aten::_scaled_dot_product_efficient_attention'],
                     ['aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward',
                      'aten::_scaled_dot_product_flash_attention']):
            runtime=Runtime();runtime.keys=keys
            with self.subTest(keys=keys),self.assertRaisesRegex(ValueError,'kernel proof'): self.run_probe(runtime)
            self.assertEqual(runtime.gpu,0)

    def test_same_route_rng_mismatch_is_rejected(self):
        runtime=Runtime();original=runtime.forward;calls=[]
        def wrong(*args):
            value=original(*args);calls.append(1)
            if len(calls)==4: runtime.gpu+=1
            return value
        runtime.forward=wrong
        with self.assertRaisesRegex(ValueError,'deterministic replay'): self.run_probe(runtime)
        self.assertEqual(runtime.gpu,0)

    def test_no_bf16_or_mutated_leaf_is_rejected(self):
        runtime=Runtime();runtime.cuda.is_bf16_supported=lambda:False
        with self.assertRaisesRegex(ValueError,'support'): self.run_probe(runtime)
        runtime=Runtime();original=runtime.forward
        def wrong(*args):
            result=original(*args);args[2][0].value+=1;return result
        runtime.forward=wrong
        with self.assertRaisesRegex(ValueError,'rollback'): self.run_probe(runtime)

    def test_screen_thresholds_are_finite_and_distinguish_output_gradient(self):
        metrics=[dict(finite=True,relative_l2=value) for value in (.02,.05,.05,.05)]
        self.assertTrue(probe.screen(metrics));metrics[0]['relative_l2']=.0201
        self.assertFalse(probe.screen(metrics));metrics[0]['finite']=False
        self.assertFalse(probe.screen(metrics))


class EntryTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'runs'/'original';cache=root/'cache';data=root/'data';data.mkdir()
        job=source/'jobs'/entry.JOB
        config=dict(source_revision=entry.SOURCE_REVISION,model='transformer',task='retrieval',
            width=128,heads=4,layers=4,data_hashes={'train':'train-sha'})
        atomic_json(job/'config.json',config)
        atomic_json(source/'suite.json',dict(source_revision=entry.SOURCE_REVISION,data=str(data),
            jobs=[dict(id=entry.JOB,config_sha256=fingerprint(config))]))
        atomic_json(source/'context.json',dict(source_revision=entry.SOURCE_REVISION,cache=str(cache.resolve())))
        atomic_json(source/'status.json',{entry.JOB:dict(state='paused')})
        generation=job/'checkpoints'/'step-1.pt';generation.parent.mkdir();generation.write_bytes(b'fixture checkpoint, never deserialized')
        atomic_json(job/'checkpoints/latest.json',dict(schema='qkan-training-checkpoint.v1',file=generation.name,
            sha256=digest(generation),contract='fixture',step=1))
        atomic_json(cache/'arrays/train-sha/complete.json',{})
        return NS(source_suite=source,cache=cache,optimizations=PRESET,gpu='0',micro_batch=16,child_timeout=900,stall_timeout=300)

    def binding(self,args):
        with patch.object(entry.subprocess,'check_output',side_effect=[entry.SOURCE_REVISION,'']),\
             patch('experiments.baseline_delivery.migration.validate_source_admission'):
            return entry.bind_source(args.source_suite,args.cache)

    def test_source_binding_is_read_only_no_checkpoint_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));before=sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob('*'))
            binding=self.binding(args);entry.unchanged(binding)
            self.assertEqual(binding['checkpoint']['step'],1)
            self.assertEqual(sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob('*')),before)
            Path(binding['checkpoint']['path']).write_bytes(b'mutated')
            with self.assertRaisesRegex(ValueError,'changed'): entry.unchanged(binding)

    def test_completed_or_invalid_pointer_is_rejected(self):
        for invalid in ('completed','bad-hash','bad-config','missing-cache'):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp));job=args.source_suite/'jobs'/entry.JOB
                if invalid=='completed': atomic_json(job/'summary.json',{})
                elif invalid=='bad-hash': (job/'checkpoints/step-1.pt').write_bytes(b'changed')
                elif invalid=='missing-cache': args.cache=Path(tmp)/'no-cache'
                else:
                    config=read(job/'config.json');config['heads']=8;atomic_json(job/'config.json',config)
                with self.assertRaises(ValueError): self.binding(args)

    def test_busy_gpu_blocks_before_source_locks_or_native_launch(self):
        args=NS(optimizations=PRESET,gpu='0')
        with patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'0, GPU-fixture'})),\
             patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=False,compute_pids=[123])),\
             patch.object(entry,'lock') as lock:
            with self.assertRaisesRegex(ValueError,'compute processes'): entry.execute(args,Path('unused'),{})
            lock.assert_not_called()

    def test_actual_launch_argv_report_binding_and_source_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));binding=self.binding(args);folder=Path(tmp)/'probe-run';folder.mkdir()
            supervisor=NS(records=[])
            def run(spec,log,env):
                self.assertEqual(spec['argv'][3],entry.TARGET)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
                request_path=Path(spec['argv'][5]);request=read(request_path)
                self.assertEqual(request['micro_batch'],16);self.assertEqual(request['checkpoint']['sha256'],binding['checkpoint']['sha256'])
                output=Path(spec['argv'][7])
                atomic_json(output/'result.json',dict(request_sha256=digest(request_path),
                    plugin_manifest=selected().manifest(),checkpoint_sha256=binding['checkpoint']['sha256'],
                    rollback=dict(inputs_exact=True,rng_exact=True,leaf_gradients_unmodified=True),mechanism_passed=True))
                return 0
            supervisor.run=run;report=dict(source_revision='candidate')
            with patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'0, GPU-fixture'})),\
                 patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True,compute_pids=[])),\
                 patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',return_value=supervisor),\
                 patch.object(entry,'bind_source',return_value=binding):
                entry.execute(args,folder,report)
            self.assertTrue(report['passed']);self.assertTrue(report['original_checkpoint_and_metadata_unchanged'])

    def test_native_request_invalid_before_framework_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.json';atomic_json(path,dict(schema='wrong'))
            with self.assertRaisesRegex(ValueError,'request/source'): native.execute(path,Path(tmp))

    def test_main_returns_compact_report_on_success_failure_and_interrupt(self):
        for error in (None,ValueError('private exception text'),KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__),tempfile.TemporaryDirectory() as tmp:
                args=['precision','--source-suite','fixture','--cache','cache','--optimizations',str(PRESET)]
                def execute(args,folder,report):
                    if error is not None: raise error
                    report.update(passed=True,status='mechanism_measured')
                upload=Mock()
                with patch.object(sys,'argv',args),patch.object(entry,'ROOT',Path(tmp)),\
                     patch.object(entry,'revision',return_value='candidate'),patch.object(entry,'execute',side_effect=execute),\
                     patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                     patch('experiments.baseline_delivery.return_artifacts.upload_declared',upload):
                    if error is not None:
                        with self.assertRaises(type(error)): entry.main()
                    else: self.assertEqual(entry.main(),0)
                folder,kind,files=upload.call_args.args
                self.assertEqual(kind,'attention-precision');self.assertEqual(files,['attention-precision.json'])
                report=read(folder/files[0]);self.assertFalse(report['formal_training_started'])
                self.assertEqual(report['optimizer_updates'],0);self.assertEqual(report['passed'],error is None)
                self.assertNotIn('private exception text',json.dumps(report))

    def test_worker_projection_uses_pre_norm_and_preserves_mask(self):
        # Inject only the primitive tensor operations; test actual adapter order.
        from contextlib import nullcontext
        events=[]
        class View:
            shape=(16,4000);device='cuda:0'
            def __init__(self,name): self.name=name
            def ne(self,value): events.append(('ne',value));return View('valid')
            def any(self,axis): return NS(all=lambda:True)
            def __invert__(self): return View('not '+self.name)
            def unsqueeze(self,dim): return self
            def __getitem__(self,key): return self
            def __add__(self,other): events.append(('position_add',other.name));return self
            def masked_fill(self,mask,value): events.append(('pad_zero',value));return self
            def masked_fill_(self,mask,value): events.append(('key_mask',mask.name,value));return self
            def chunk(self,count,dim): return [View('q'),View('k'),View('v')]
            def reshape(self,*shape): events.append(('reshape',shape));return self
            def transpose(self,*dims): return self
            def contiguous(self): return self
            def detach(self): return self
            def requires_grad_(self,value): events.append(('leaf_grad',value));return self
        attn=NS(batch_first=True,bias_k=None,bias_v=None,add_zero_attn=False,num_heads=4,head_dim=32,
            in_proj_weight='weights',in_proj_bias='bias')
        layer=NS(norm_first=True,self_attn=attn,norm1=lambda x:(events.append(('pre_norm',x.name)) or x))
        model=NS(encoder=NS(layers=[layer]),embedding=lambda x:View('embedding'),position=View('position'))
        runtime=NS(no_grad=nullcontext,float32='fp32',zeros=lambda shape,**kwargs:View('mask'),
            nn=NS(functional=NS(linear=lambda h,w,b:(events.append(('projection',w,b)) or View('packed')))))
        qkv,mask=native.prepare_qkv(runtime,model,View('tokens'))
        self.assertEqual([x.name for x in qkv],['q','k','v'])
        self.assertLess(events.index(('position_add','position')),events.index(('pre_norm','embedding')))
        self.assertLess(events.index(('pre_norm','embedding')),events.index(('projection','weights','bias')))
        self.assertIn(('reshape',(16,4000,4,32)),events)
        self.assertIn(('key_mask','not valid',float('-inf')),events)
        layer.norm_first=False
        with self.assertRaisesRegex(ValueError,'adapter'): native.prepare_qkv(runtime,model,View('tokens'))

    def test_cli_help_and_invalid_flags_never_import_native(self):
        script="""import importlib.abc,runpy,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','jax','tensorflow'): raise RuntimeError('Native import forbidden')
sys.meta_path.insert(0,Guard())
sys.argv=[sys.argv[1],*sys.argv[2:]]
runpy.run_module(sys.argv[0],run_name='__main__')
"""
        for module in (entry.__name__,native.__name__):
            for flag,expected in (('--help',0),('--invalid',2)):
                result=subprocess.run([sys.executable,'-B','-X','utf8','-c',script,module,flag],capture_output=True,text=True)
                self.assertEqual(result.returncode,expected,result.stderr)


if __name__=='__main__': unittest.main()
