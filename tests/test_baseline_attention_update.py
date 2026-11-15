"""Stdlib/injected controls only. Native model/device evidence belongs to collaborator."""
from contextlib import contextmanager,nullcontext
import copy
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import attention_update as strategy
from experiments.baseline_delivery import attention_update_native as native
from experiments.baseline_delivery import attention_update_diagnostic as entry
from experiments.baseline_delivery import optimization_plugins as plugins
from experiments.baseline_delivery.common import ROOT,atomic_json,digest,fingerprint,read

PRESET=ROOT/'experiments/baseline_delivery/optimization-presets/bf16-attention-update.json'


class View:
    def __init__(self,shape=(2,4000,128),dtype='fp32',events=None): self.shape,self.dtype,self.events=shape,dtype,events if events is not None else []
    @property
    def ndim(self): return len(self.shape)
    def transpose(self,a,b):
        shape=list(self.shape);shape[a],shape[b]=shape[b],shape[a];return View(tuple(shape),self.dtype,self.events)
    def reshape(self,*shape): return View(shape,self.dtype,self.events)
    def permute(self,*order): return View(tuple(self.shape[i] for i in order),self.dtype,self.events)
    view=reshape
    def contiguous(self): return self
    def to(self,dtype): self.events.append(('cast',self.dtype,dtype));return View(self.shape,dtype,self.events)


def adapter_fixture():
    events=[]
    def projection(q,k,v,w,b):
        events.append(('in_projection',q.shape,q is k is v,w,b))
        return tuple(View(q.shape,q.dtype,events) for _ in range(3))
    def sdpa(q,k,v,**kwargs):
        events.append(('sdpa',q.shape,q.dtype,kwargs['attn_mask'].shape,kwargs['dropout_p'],kwargs['is_causal']))
        return View(q.shape,q.dtype,events)
    def linear(x,w,b): events.append(('out_projection',x.dtype,x.shape));return View(x.shape,x.dtype,events)
    f=NS(_in_projection_packed=projection,scaled_dot_product_attention=sdpa,linear=linear)
    runtime=NS(float32='fp32',bfloat16='bf16',bool='bool',nn=NS(functional=f,
        attention=NS(SDPBackend=NS(EFFICIENT_ATTENTION='efficient'),sdpa_kernel=lambda b:nullcontext())))
    layers=[]
    for _ in range(4):
        attn=NS(batch_first=True,_qkv_same_embed_dim=True,bias_k=None,bias_v=None,add_zero_attn=False,
            embed_dim=128,num_heads=4,head_dim=32,in_proj_weight='weight',in_proj_bias='bias',dropout=.1,
            out_proj=NS(weight='out_weight',bias='out_bias'))
        layers.append(NS(norm_first=True,training=True,self_attn=attn,dropout1=lambda x:x))
    return runtime,NS(encoder=NS(layers=layers)),events


class StrategyTests(unittest.TestCase):
    def test_null_option_does_not_inspect_runtime_or_model(self):
        with strategy.attention_route(object(),object()) as stats: self.assertIsNone(stats)

    def test_full_update_scope_is_explicit_and_combinations_rejected(self):
        config=read(PRESET)
        self.assertTrue(plugins.resolve(config,scope='attention_update').bf16_amp)
        self.assertEqual(plugins.resolve(config,scope='attention_update').manifest()['precision'],'bf16_sdpa_fp32_master')
        with self.assertRaises(ValueError): plugins.resolve(config)
        config['optimizations']['padding_trim']={'enabled':True}
        with self.assertRaisesRegex(ValueError,'full token'): plugins.resolve(config,scope='attention_update')
        config['optimizations'].pop('padding_trim');config['optimizations']['flash_attention']={'enabled':True}
        with self.assertRaisesRegex(ValueError,'pending'): plugins.resolve(config,scope='attention_update')

    def test_per_instance_bf16_sdpa_and_fp32_projection_layout_restored(self):
        runtime,model,events=adapter_fixture();layer=model.encoder.layers[0]
        with strategy.attention_route(runtime,model,True) as stats:
            result=layer._sa_block(View(events=events),None,View((2,4000)))
            self.assertEqual(result.shape,(2,4000,128));self.assertEqual(result.dtype,'fp32')
            self.assertEqual(stats['layer_calls'],[1,0,0,0])
        self.assertFalse(hasattr(layer,'_sa_block'))
        self.assertIn(('in_projection',(4000,2,128),True,'weight','bias'),events)
        self.assertIn(('sdpa',(2,4,4000,32),'bf16',(2,1,1,4000),.1,False),events)
        self.assertIn(('out_projection','fp32',(8000,128)),events)

    def test_fp32_adapter_has_no_bf16_cast_and_eval_dropout_zero(self):
        runtime,model,events=adapter_fixture();layer=model.encoder.layers[0];layer.training=False
        with strategy.attention_route(runtime,model,fp32_adapter=True): layer._sa_block(View(events=events),None,View((2,4000)))
        self.assertIn(('sdpa',(2,4,4000,32),'fp32',(2,1,1,4000),0.,False),events)
        self.assertFalse(any(event[-1]=='bf16' for event in events))

    def test_exception_restores_owned_attribute_and_other_model(self):
        runtime,model,_=adapter_fixture();other=adapter_fixture()[1];old=object();model.encoder.layers[0]._sa_block=old
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with strategy.attention_route(runtime,model,True):
                self.assertFalse(hasattr(other.encoder.layers[0],'_sa_block'));raise RuntimeError('injected')
        self.assertIs(model.encoder.layers[0]._sa_block,old)
        self.assertFalse(hasattr(model.encoder.layers[1],'_sa_block'))

    def test_reject_unknown_attention_or_masks_before_or_during_patch(self):
        runtime,model,_=adapter_fixture();model.encoder.layers[3].norm_first=False
        with self.assertRaises(ValueError):
            with strategy.attention_route(runtime,model,True): pass
        self.assertFalse(hasattr(model.encoder.layers[0],'_sa_block'))
        model.encoder.layers[3].norm_first=True
        for mask,causal in ((object(),False),(None,True)):
            with self.assertRaises(ValueError):
                with strategy.attention_route(runtime,model,True): model.encoder.layers[0]._sa_block(View(),mask,View((2,4000)),causal)
        self.assertFalse(hasattr(model.encoder.layers[0],'_sa_block'))

    def test_fixed_gate_rejects_nonfinite_missing_or_retuned_metrics(self):
        good={key:0. for key in strategy.POLICY['dropout_off_limits']}
        self.assertTrue(strategy.numerical_gate(good))
        for value in (float('nan'),float('inf'),-.01,.051):
            bad=dict(good,gradient_relative_l2=value);self.assertFalse(strategy.numerical_gate(bad))
        good.pop('loss_absolute')
        with self.assertRaises(ValueError): strategy.numerical_gate(good)

    def test_balanced_timing_reports_gain_and_rejects_insufficient_samples(self):
        summary=strategy.timing_summary(dict(fp32=[8,10,9,9],bf16=[4,6,5,5]))
        self.assertEqual(summary['median_seconds'],dict(fp32=9,bf16=5));self.assertTrue(summary['gain_passed'])
        self.assertAlmostEqual(summary['throughput_gain'],.8)
        self.assertFalse(strategy.timing_summary(dict(fp32=[5]*4,bf16=[6]*4))['gain_passed'])
        with self.assertRaises(ValueError): strategy.timing_summary(dict(fp32=[1],bf16=[.5]))

    def test_request_binds_revision_config_policy_and_option(self):
        request=dict(schema='qkan-attention-update-input.v1',source_revision='rev',policy=copy.deepcopy(strategy.POLICY),
            source_config={'seed':42},source_config_sha256=fingerprint({'seed':42}),optimization_config=read(PRESET))
        self.assertTrue(strategy.validate_request(request,'rev').bf16_amp)
        for key,value in (('source_revision','wrong'),('source_config',{'seed':43}),('policy',{})):
            bad=copy.deepcopy(request);bad[key]=value
            with self.assertRaises(ValueError): strategy.validate_request(bad,'rev')


def child(request,mode,passed=True):
    return dict(schema='qkan-attention-update-native.v1',mode=mode,request_sha256='request',
        checkpoint_sha256='original',plugin_manifest=plugins.resolve(read(PRESET),scope='attention_update').manifest(),
        policy_sha256=fingerprint(strategy.POLICY),formal_training_started=False,training_samples_committed=0,
        passed=passed,rollback=dict(full_state_exact=True,committed_samples=0,committed_updates=0))


class ControllerTests(unittest.TestCase):
    def test_child_identity_rollbacks_and_exit_status_are_required(self):
        request=dict(checkpoint={'sha256':'original'},optimization_config=read(PRESET));result=child(request,'benchmark')
        self.assertIs(entry.accept_child(result,request,'request','benchmark',0),result)
        for key,value in (('checkpoint_sha256','wrong'),('policy_sha256','wrong'),('rollback',{}),('training_samples_committed',128),('formal_training_started',True)):
            bad=copy.deepcopy(result);bad[key]=value
            with self.assertRaises(ValueError): entry.accept_child(bad,request,'request','benchmark',0)
        with self.assertRaises(ValueError): entry.accept_child(result,request,'request','benchmark',76)
        result['passed']=False;entry.accept_child(result,request,'request','benchmark',76)

    def recovery(self):
        a=dict(passed=True,full_state_sha256='end',step=12,cursor=256,schedule_progress=1256,updates=[{'loss':.5},{'loss':.4}])
        b=dict(passed=True,full_state_sha256='mid',step=11,cursor=128,schedule_progress=1128,updates=[{'loss':.5}])
        c=dict(a,updates=[{'loss':.4}]);return dict(uninterrupted=a,interrupt=b,resume=c)

    def test_recovery_requires_exact_full_state_rows_schedule_and_losses(self):
        rows=self.recovery();self.assertTrue(entry.recovery_agreement(rows)['passed'])
        for key,value in (('full_state_sha256','wrong'),('step',13),('cursor',255),('schedule_progress',1255),('updates',[{'loss':.41}])):
            bad=copy.deepcopy(rows);bad['resume'][key]=value
            with self.assertRaises(ValueError): entry.recovery_agreement(bad)

    def test_cleanup_only_removes_unchanged_own_disposable_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);p=folder/'diagnostic-state.pt';p.write_bytes(b'diagnostic fixture')
            receipt=dict(schema='qkan-bf16-diagnostic-checkpoint.v1',sha256=digest(p),bytes=p.stat().st_size,request_sha256='request',state_sha256='state')
            atomic_json(p.with_suffix('.json'),receipt);result=dict(diagnostic_checkpoint=dict(file=p.name,**receipt))
            sentinel=folder/'original.pt';sentinel.write_bytes(b'preserve')
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError): entry.remove_disposable_checkpoint(folder,result)
            self.assertTrue(p.exists());p.write_bytes(b'diagnostic fixture')
            self.assertTrue(entry.remove_disposable_checkpoint(folder,result)['removed'])
            self.assertEqual(sentinel.read_bytes(),b'preserve');self.assertFalse(p.exists())

    def test_controller_runs_fresh_children_recovery_then_benchmark_and_preserves_source(self):
        from tests.test_baseline_attention_precision import EntryTests
        source_fixture=EntryTests()
        with tempfile.TemporaryDirectory() as tmp:
            args=source_fixture.fixture(Path(tmp));args.optimizations=PRESET;binding=source_fixture.binding(args)
            output=Path(tmp)/'diagnostic';output.mkdir();request_holder={};executions=[]
            class Supervisor:
                def __init__(self,*a): self.records=[]
                def run(self,spec,log,env):
                    argv=spec['argv'];mode=argv[argv.index('--mode')+1];out=Path(argv[argv.index('--output')+1]);out.mkdir()
                    request=read(Path(argv[argv.index('--request')+1]));request_holder.update(request)
                    result=child(request,mode);result['request_sha256']=digest(output/'update-input.json')
                    result['checkpoint_sha256']=request['checkpoint']['sha256']
                    if mode!='benchmark': result.update(ControllerTests().recovery()[mode])
                    else: result.update(trace={'passed':True})
                    atomic_json(out/'result.json',result);executions.append((mode,argv,env))
                    self.records.append(dict(exit_code=0));return 0
            report={'source_revision':'new'}
            with patch.object(entry,'bind_source',return_value=binding),patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{'uuid':'gpu'}})),\
                 patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}),\
                 patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',Supervisor),patch.object(entry,'remove_disposable_checkpoint',return_value={'removed':True}):
                entry.execute(args,output,report)
            self.assertTrue(report['passed']);self.assertTrue(report['original_checkpoint_and_metadata_unchanged'])
            self.assertEqual([v[0] for v in executions],['uninterrupted','interrupt','resume','benchmark'])
            self.assertIn('--recovery-state',executions[2][1]);self.assertEqual(executions[3][2]['QKAN_DIAGNOSTIC_STAGE_FENCES'],'0')
            self.assertEqual(request_holder['policy'],strategy.POLICY)

    def test_busy_gpu_starts_no_child(self):
        args=NS(optimizations=PRESET,gpu='0')
        with patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})),\
             patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':False}),\
             patch('experiments.baseline_delivery.diagnostic_watch.Supervisor') as supervisor:
            with self.assertRaisesRegex(ValueError,'busy'): entry.execute(args,Path('.'),{})
            supervisor.assert_not_called()


class BenchmarkTests(unittest.TestCase):
    def fixture(self,metrics=None,keys=None):
        elapsed=[0.];runtime=NS(base={'position':0},position=0,updates=[])
        cuda=NS(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,memory_allocated=lambda:32,
            max_memory_allocated=lambda:64,max_memory_reserved=lambda:128)
        @contextmanager
        def profile(**kwargs):
            names=keys if keys is not None else ['aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward']
            yield NS(key_averages=lambda:[NS(key=name) for name in names])
        runtime.t=NS(cuda=cuda,profiler=NS(profile=profile,ProfilerActivity=NS(CPU='cpu',CUDA='cuda')))
        runtime.state_hash=lambda state:str(state['position'])
        runtime.snapshot=lambda:{'position':runtime.position}
        runtime.restore=lambda state:setattr(runtime,'position',state['position'])
        def update(enabled,**kwargs):
            runtime.position+=1;elapsed[0]+=1 if enabled else 2;runtime.updates.append((enabled,kwargs))
            return dict(loss=.5,row_ids_sha256=str(runtime.position),attention={'calls':64})
        runtime.update=update;runtime.metrics=lambda *a:dict(metrics or {key:0. for key in strategy.POLICY['dropout_off_limits']})
        return runtime,elapsed

    def test_full_benchmark_includes_balanced_updates_and_exact_rollback(self):
        runtime,elapsed=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,patch.object(native.time,'perf_counter',side_effect=lambda:elapsed[0]),\
             patch('experiments.baseline_delivery.attention_update_trace.collect_and_validate',return_value={'passed':True}):
            result=native.benchmark(runtime,Path(tmp))
            self.assertTrue(result['passed']);self.assertEqual(runtime.position,0)
            self.assertEqual(result['timing']['samples_seconds'],dict(fp32=[2]*4,bf16=[1]*4))
            self.assertEqual(result['timing_order'],['fp32','bf16','bf16','fp32']*2)
            self.assertFalse(result['trajectory']['convergence_certified'])
            self.assertTrue(read(Path(tmp)/'benchmark.json')['rollback']['full_state_exact'])
        timed=runtime.updates[-8:];self.assertTrue(all(not options for _,options in timed))

    def test_numerical_rejection_skips_trace_and_timing_but_rolls_back(self):
        runtime,_=self.fixture();calls=[0]
        def metrics(*args):
            calls[0]+=1;values={key:0. for key in strategy.POLICY['dropout_off_limits']}
            if calls[0]==2: values['gradient_relative_l2']=.2
            return values
        runtime.metrics=metrics
        with tempfile.TemporaryDirectory() as tmp,patch('experiments.baseline_delivery.attention_update_trace.collect_and_validate') as trace:
            result=native.benchmark(runtime,Path(tmp));self.assertFalse(result['passed'])
            self.assertEqual(runtime.position,0);trace.assert_not_called();self.assertNotIn('timing',result)
            self.assertTrue(read(Path(tmp)/'benchmark.json')['rollback']['full_state_exact'])

    def test_wrong_or_forward_only_kernel_cannot_admit_candidate(self):
        for keys in ([],['aten::_scaled_dot_product_efficient_attention'],['aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward','aten::_scaled_dot_product_flash_attention']):
            runtime,_=self.fixture(keys=keys)
            with tempfile.TemporaryDirectory() as tmp,patch('experiments.baseline_delivery.attention_update_trace.collect_and_validate',return_value={'passed':True}):
                with self.assertRaisesRegex(ValueError,'kernel proof'): native.benchmark(runtime,Path(tmp))
                self.assertEqual(runtime.position,0)


class UpdateFlowTests(unittest.TestCase):
    def test_real_update_control_accumulates_micros_then_steps_optimizer_and_sample_clock_once(self):
        events=[];parameter=NS(grad=None,value=1.)
        class Number:
            def __init__(self,value): self.value=value
            def detach(self): return self
            def __truediv__(self,n): return Number(self.value/n)
            def __mul__(self,n): return Number(self.value*n)
            def __add__(self,other): return Number(self.value+(other.value if isinstance(other,Number) else other))
            __radd__=__add__
            def __float__(self): return float(self.value)
            def __int__(self): return int(self.value)
            def backward(self): parameter.grad=(parameter.grad or 0)+self.value;events.append('backward')
            def sum(self): return self
        class Array:
            def __getitem__(self,key): return self
        class Logits:
            def detach(self): return self
            def argmax(self,*args): return self
            def __eq__(self,other): return Number(16)
        class Model:
            def train(self): events.append('train')
            def __call__(self,batch,**kwargs): events.append('forward');return Logits()
            def parameters(self): return [parameter]
        model=Model()
        def step(): events.append('optimizer');parameter.value-=parameter.grad
        torch=NS(as_tensor=lambda x,**kwargs:x,isfinite=lambda x:True if isinstance(x,Number) else NS(all=lambda:True),
            stack=lambda x:NS(all=lambda:True),cuda=NS(synchronize=lambda:None),nn=NS(
                functional=NS(cross_entropy=lambda *a,**k:Number(16)),
                utils=NS(clip_grad_norm_=lambda *a,**k:events.append('clip')),
                attention=NS(SDPBackend=NS(EFFICIENT_ATTENTION='efficient'),sdpa_kernel=lambda b:nullcontext())))
        runtime=native.NativeUpdate.__new__(native.NativeUpdate);runtime.t=torch;runtime.model=model
        runtime.opt=NS(zero_grad=lambda **kwargs:setattr(parameter,'grad',None),step=step)
        runtime.schedule=NS(step=lambda count:events.append(('schedule',count)))
        runtime.progress=dict(cursor=0,step=7,epoch_loss=10.,epoch_correct=2,epoch_samples=4)
        class Ids:
            def __getitem__(self,key): return self
            def numpy(self): return self
            def tobytes(self): return b'fixed128rows'
        runtime.permutation=Ids();runtime.config={'clip_norm':1.}
        class Split:
            def __len__(self): return 1000
            def numpy_batch(self,ids,validate): return dict(inputs1=Array(),inputs2=Array(),labels=Array())
        runtime.split=Split()
        result=runtime.update(False)
        self.assertEqual(result['loss'],1.);self.assertEqual(result['correct'],128)
        self.assertEqual(parameter.grad,1.);self.assertEqual(events.count('forward'),8);self.assertEqual(events.count('backward'),8)
        self.assertEqual(events[-3:],['clip','optimizer',('schedule',128)])
        self.assertEqual(runtime.progress,dict(cursor=128,step=8,epoch_loss=138.,epoch_correct=130,epoch_samples=132))


class ArtifactTests(unittest.TestCase):
    def test_real_clis_help_under_native_import_guard(self):
        code="""import importlib.abc,runpy,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','jax','tensorflow'): raise AssertionError('Forbidden native import: '+name)
sys.meta_path.insert(0,Guard());sys.argv=[sys.argv[1],'--help'];runpy.run_module(sys.argv[0],run_name='__main__')
"""
        for module in ('attention_update_diagnostic','attention_update_native'):
            result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,'experiments.baseline_delivery.'+module],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--output' if module.endswith('native') else '--source-suite',result.stdout)

    def test_dossier_plan_and_trace_candidate_are_bound_without_native_claim(self):
        plan=read(entry.PLAN);self.assertEqual(plan['schema_version'],'research-plugin-architecture.plugin-plan.v2')
        self.assertEqual(plan['recovery_contract']['dossier_sha256'],digest(entry.DOSSIER));self.assertEqual(plan['authorization'],'approved')
        self.assertEqual(plan['trace_readiness']['status'],'blocked')
        source=(ROOT/'experiments/baseline_delivery/attention_update_trace.py').read_text(encoding='utf8')
        self.assertIn('require_complete=True',source);self.assertIn('full_model_replay_exact',source)


if __name__=='__main__': unittest.main()
