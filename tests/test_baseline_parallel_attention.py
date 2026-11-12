"""Stream/gradient ownership and bounded handoff contracts; no native framework."""
from contextlib import contextmanager,nullcontext
import argparse
import ast
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery import attention_diagnostic as diagnostic
from experiments.baseline_delivery.common import atomic_json,read,fingerprint,child_spec
from experiments.baseline_delivery.parallel_attention import parallel_runtime,fixed_merge
from experiments.baseline_delivery.torch_acceleration import policy,torch_runtime,BackendFailure,CaptureCapacity,Executor,admit_recovery,RecoveryMismatch,AccelerationRequired


class Tensor:
    dtype='float32';device='cuda:0';requires_grad=True;grad=None
    def __init__(self,value,shape=(16,10)): self.value=value;self.shape=shape
    def clone(self): return Tensor(self.value,self.shape)
    def detach(self): return self
    def copy_(self,other): self.value=other.value
    def __add__(self,other): return Tensor(self.value+other.value,self.shape)
    def record_stream(self,stream): pass


class Module:
    training=True
    def __call__(self,*a,**k): return self.forward(*a,**k)


class Model(Module):
    config=dict(model='transformer',task='retrieval',dropout=.1,layers=4,heads=4,width=128)
    def __init__(self):
        self.weight=Tensor(2);self.head=Tensor(7);self.position=Tensor(1)
    def train(self,state): self.training=state
    def named_parameters(self): return iter((('encoder.weight',self.weight),('head.weight',self.head)))
    def named_buffers(self): return iter((('position',self.position),))
    def encode(self,x,validated_inputs): return Tensor(x.value*self.weight.value)
    def classify_pair(self,a,b): return Tensor((a.value+b.value)*self.head.value)
    def forward(self,batch,validated_inputs): return Tensor(-1)


class FakeFunction:
    @classmethod
    def apply(cls,*args):
        cls.last_args=args;cls.last_context=NS()
        return cls.forward(cls.last_context,*args)


def framework():
    events=[];current=['main'];rng=[0]
    class Stream:
        def __init__(self,name=None): self.name=name or 'side'+str(len([e for e in events if e[0]=='new'])) ;events.append(('new',self.name))
        def wait_stream(self,other): events.append(('wait',self.name,other.name))
    main=Stream('main')
    @contextmanager
    def stream(s):
        old=current[0];current[0]=s.name
        try: yield
        finally: current[0]=old
    class OOM(RuntimeError): pass
    cuda=NS(Stream=Stream,current_stream=lambda:main,stream=stream,synchronize=Mock(),empty_cache=Mock(),OutOfMemoryError=OOM)
    def graph(adapter,static,**kwargs):
        events.append(('capture',id(adapter.base),kwargs))
        def replay(value): events.append(('forward',current[0],value.value));rng[0]+=1;return adapter(value)
        return replay
    cuda.make_graphed_callables=Mock(side_effect=graph)
    def grad(output,parameters,gradient,allow_unused):
        events.append(('backward',current[0],tuple(id(p) for p in parameters)))
        return tuple(Tensor(output.value*gradient.value) for p in parameters)
    torch=NS(cuda=cuda,nn=NS(Module=Module),float32='float32',compile=Mock(side_effect=lambda call,**kw:call),
        autograd=NS(Function=FakeFunction,grad=Mock(side_effect=grad),function=NS(once_differentiable=lambda f:f)),
        enable_grad=nullcontext,no_grad=nullcontext,ones_like=lambda x:Tensor(1),zeros_like=lambda x:Tensor(0))
    return torch,events,rng


class StreamOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.torch,self.events,self.rng=framework();self.model=Model()
        self.runtime=torch_runtime(self.torch,self.model,lambda:self.rng[0],lambda value:self.rng.__setitem__(0,value),
            policy('compile_graph','eager_parallel','graph'))
        self.batch=dict(inputs1=Tensor(3),inputs2=Tensor(5))
        self.runtime.begin_update(16)

    def test_two_independent_pools_replicas_and_fixed_merge_original_head(self):
        result=self.runtime.forward(self.batch,'compile_graph')
        self.assertEqual(result.value,112)
        self.assertEqual(self.rng[0],2)  # capture warmup cannot advance live RNG
        captures=[e for e in self.events if e[0]=='capture']
        self.assertEqual(len(captures),2);self.assertNotEqual(captures[0][1],captures[1][1])
        for parameters in self.runtime.parameters:
            self.assertEqual(len(parameters),1);self.assertIsNot(parameters[0],self.model.weight)
        bridge=FakeFunction.__subclasses__()[-1]
        self.assertIs(bridge.last_args[-1],self.model.weight)
        self.assertNotIn(self.model.head,bridge.last_args)
        result=bridge.backward(bridge.last_context,Tensor(10),Tensor(20))
        self.assertEqual(result[:3],(None,None,None));self.assertEqual(result[3].value,260)
        self.assertEqual(self.model.weight.value,2);self.assertIsNone(self.model.weight.grad)
        self.assertEqual(self.runtime.status()['backward_graph_replays'],2)

    def test_stream_dependencies_surround_forward_backward_and_original_merge(self):
        self.runtime.forward(self.batch,'compile_graph');self.events.clear()
        self.runtime.forward(self.batch,'compile_graph')
        bridge=FakeFunction.__subclasses__()[-1]
        bridge.backward(bridge.last_context,Tensor(1),Tensor(1))
        names=[s.name for s in self.runtime.streams]
        expected=[('wait',names[0],'main'),('forward',names[0],3),
            ('wait',names[1],'main'),('forward',names[1],5),
            ('wait','main',names[0]),('wait','main',names[1]),
            ('wait',names[1],'main')]
        expected.insert(3,('wait',names[1],names[0]))
        self.assertEqual(self.events[:8],expected)
        backward=[e for e in self.events if e[0]=='backward']
        self.assertEqual([e[1] for e in backward],list(reversed(names)))
        self.assertEqual(self.events[-2:],[('wait','main',n) for n in names])

    def test_default_rng_writes_are_ordered_and_cross_stream_storage_is_recorded(self):
        self.runtime.forward(self.batch,'compile_graph')
        bridge=FakeFunction.__subclasses__()[-1]
        g1,g2=Tensor(1),Tensor(1)
        with patch.object(Tensor,'record_stream',autospec=True) as record:
            self.runtime.forward(self.batch,'compile_graph')
            bridge.backward(bridge.last_context,g1,g2)
        recorded=[(id(call.args[0]),call.args[1].name) for call in record.call_args_list]
        names=[s.name for s in self.runtime.streams]
        self.assertIn(('wait',names[1],names[0]),self.events)
        for index,key in enumerate(('inputs1','inputs2')):
            self.assertIn((id(self.batch[key]),names[index]),recorded)
            self.assertIn((id(bridge.last_context.outputs[index]),'main'),recorded)
        self.assertIn((id(g1),names[0]),recorded);self.assertIn((id(g2),names[1]),recorded)
        self.assertEqual(self.runtime.status()['forward_schedule'],'document1_then_document2_shared_rng')

    def test_original_weight_refresh_and_shape_rebuild_tail_do_not_mutate_checkpoint(self):
        self.runtime.forward(self.batch,'compile_graph');self.model.weight.value=4
        self.assertEqual(self.runtime.forward(self.batch,'compile_graph').value,224)
        self.assertEqual(self.torch.cuda.make_graphed_callables.call_count,2)
        self.runtime.begin_update(32)
        self.assertEqual(self.runtime.forward(self.batch,'compile_graph').value,-1)
        self.assertEqual(self.runtime.status()['eager_tail_calls'],1)
        self.runtime.begin_update(16);self.batch['inputs1'].shape=(16,11)
        self.runtime.forward(self.batch,'compile_graph')
        self.assertEqual(self.runtime.status()['graph_builds'],2)
        self.assertEqual([n for n,_ in self.model.named_parameters()],['encoder.weight','head.weight'])

    def test_rng_capacity_preflight_and_capture_failure_restore_rng(self):
        for value in self.batch.values(): value.shape=(32,4000)
        self.runtime.begin_update(32)
        with self.assertRaises(CaptureCapacity): self.runtime.forward(self.batch,'compile_graph')
        self.torch.cuda.make_graphed_callables.assert_not_called()
        for value in self.batch.values(): value.shape=(16,4000)
        self.runtime.begin_update(16)
        def failure(*args,**kwargs): self.rng[0]=99;raise RuntimeError('capture unavailable')
        self.torch.cuda.make_graphed_callables.side_effect=failure
        with self.assertRaises(BackendFailure): self.runtime.forward(self.batch,'compile_graph')
        self.assertEqual(self.rng[0],0)

    def test_fatal_and_nested_oom_escape_for_existing_finite_microbatch_retry(self):
        oom=self.torch.cuda.OutOfMemoryError('out of memory')
        wrapped=RuntimeError('compiler wrapper');wrapped.__cause__=oom
        for error in (oom,wrapped,RuntimeError('illegal memory access')):
            with self.subTest(error=type(error).__name__):
                self.torch.cuda.make_graphed_callables.side_effect=error
                with self.assertRaises(type(oom) if error is wrapped else type(error)):
                    self.runtime.forward(self.batch,'compile_graph')

    def test_candidate_is_explicit_scope_fp32_and_no_redundant_operator_profile(self):
        for change in (dict(task='text'),dict(model='mega')):
            model=Model();model.config={**model.config,**change}
            with self.assertRaises(BackendFailure): parallel_runtime(self.torch,model,lambda:0,lambda _:None,policy(backend='eager_parallel'))
        model=Model();model.weight.dtype='bfloat16'
        with self.assertRaises(BackendFailure): parallel_runtime(self.torch,model,lambda:0,lambda _:None,policy(backend='eager_parallel'))
        executor=Executor(self.runtime,policy(backend='eager_parallel'),diagnose=Mock(),operators=Mock())
        self.assertIsNone(executor.diagnose);self.assertIsNone(executor.operators)
        with self.assertRaises(BackendFailure): self.runtime.forward(self.batch,'compile')

    def test_gradient_merge_never_aliases_static_graph_buffers(self):
        a,b=Tensor(1),Tensor(2)
        merged=fixed_merge(self.torch,(a,None,a,None),(b,b,None,None))
        self.assertEqual([v.value if v else None for v in merged],[3,2,1,None])
        for value in merged:
            if value: self.assertIsNot(value,a);self.assertIsNot(value,b)


class HandoffTests(unittest.TestCase):
    def test_gpu_preflight_rejects_unknown_inventory_and_attributes_only_selected_gpu(self):
        with patch.object(diagnostic.subprocess,'run',return_value=NS(returncode=0,
                stdout='123, GPU-other\n456, GPU-selected\n')) as query:
            self.assertEqual(diagnostic.gpu_preflight('0, GPU-selected, RTX3090, 24576'),
                             dict(idle=False,compute_pids=[456]))
            self.assertEqual(query.call_args.kwargs['timeout'],5)
        with patch.object(diagnostic.subprocess,'run',return_value=NS(returncode=0,stdout='')):
            self.assertTrue(diagnostic.gpu_preflight('0, GPU-selected, RTX3090, 24576')['idle'])
        for response in (NS(returncode=1,stdout=''),NS(returncode=0,stdout='unknown inventory')):
            with patch.object(diagnostic.subprocess,'run',return_value=response):
                with self.assertRaises(ValueError): diagnostic.gpu_preflight('0, GPU-selected, RTX3090, 24576')

    def candidate_handoff(self,fences=False):
        with tempfile.TemporaryDirectory() as temp:
            suite=Path(temp)/'suite';data=Path(temp)/'data';data.mkdir()
            old=dict(model='transformer',task='retrieval',source_revision='old',performance={'enabled':True},
                adaptive_memory=dict(logical_tiers=[32,64,128]),torch_acceleration=policy())
            atomic_json(suite/'migration-source'/diagnostic.JOB/'config.json',old)
            atomic_json(suite/'migration-source/plan.json',{})
            plan=dict(data=str(data),jobs={diagnostic.JOB:dict(checkpoint=dict(sha256='sha',step=12683)),
                'transformer-listops-s42':dict(completion={'passed':True})})
            def canary(config,data,cache,folder,device,announce,run):
                self.assertEqual(run.__self__.__class__.__name__,'Supervisor')
                self.assertEqual(config['torch_acceleration']['backend'],'eager_parallel')
                self.assertEqual(device,'cuda:0');self.assertEqual(__import__('os').environ['CUDA_VISIBLE_DEVICES'],'2')
                self.assertEqual(__import__('os').environ['QKAN_DIAGNOSTIC_STAGE_FENCES'],'1' if fences else '0')
                announce('Recovery child: simulated smoke')
                return dict(passed=True,fresh_process=True,trace_semantic_and_replay=True,test_access=False,
                    config_sha256=fingerprint(config),source_revision='new',
                    torch_acceleration=dict(state=dict(mode='compile_graph')))
            def calibration(*args):
                config=args[2];record=args[7]
                self.assertEqual(args[-2].__self__.__class__.__name__,'Supervisor')
                self.assertEqual(record['checkpoint']['step'],12683)
                # Use the actual parent builder supplied by the diagnostic.
                atomic_json(suite/'config.json',config)
                spec=args[-1]('--config',suite/'config.json','--output',suite/'calibration',
                    '--data',data,'--cache',args[4],'--device','cuda:0','--acceleration-probe-source',suite)
                self.assertIn('experiments.baseline_delivery.worker',spec['argv'])
                self.assertNotIn('--resume',spec['argv']);self.assertNotIn('--synthetic',spec['argv'])
                # Smoke the dispatched argv through the real parser/main AST,
                # with only the native run body substituted (no Torch import).
                from experiments.baseline_delivery.common import ROOT
                tree=ast.parse((ROOT/'experiments/baseline_delivery/worker.py').read_text(encoding='utf8'))
                node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
                worker=Mock(return_value=0)
                scope=dict(argparse=argparse,Path=Path,read=read,lock=lambda _:nullcontext(),
                    run=worker,signal=__import__('signal'),request_pause=lambda *_:None)
                exec(compile(ast.Module(body=[node],type_ignores=[]),'worker-main','exec'),scope)
                argv=spec['argv'];start=argv.index('experiments.baseline_delivery.worker')+1
                with patch('sys.argv',['worker',*argv[start:]]),patch('signal.signal'):
                    self.assertEqual(scope['main'](),0)
                self.assertEqual(worker.call_args.args[-2],suite)
                self.assertFalse(worker.call_args.args[5]);self.assertFalse(worker.call_args.args[7])
                path=suite/'calibration/acceleration-probe.json';atomic_json(path,dict(passed=True));return path
            report={};args=NS(migrate_from=Path('original'),cache=Path('cache'),gpu='2',stage_fences=fences)
            with patch('experiments.baseline_delivery.readiness.policy_check',return_value=('impl','review')),\
                patch.object(diagnostic,'revision',return_value='new'),\
                patch('experiments.baseline_delivery.migration.snapshot',return_value=plan),\
                patch('experiments.baseline_delivery.migration.compatible') as compatible,\
                patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['2'],{'2':{'uuid':'gpu'}})),\
                patch.object(diagnostic,'gpu_preflight',return_value=dict(idle=True,compute_pids=[])),\
                patch('experiments.baseline_delivery.runner.canary',side_effect=canary) as recovery,\
                patch('experiments.baseline_delivery.acceleration_calibration.calibrate',side_effect=calibration) as real:
                diagnostic.execute(args,suite,report)
            self.assertEqual(report['passed'],not fences)
            recovery.assert_called_once();compatible.assert_called_once()
            if fences:
                real.assert_not_called()
                self.assertEqual(report['status'],'stage_fences_passed');self.assertTrue(report['stage_observation_passed'])
            else: real.assert_called_once()
            self.assertEqual(report['preserved_completed'],['transformer-listops-s42'])
            self.assertFalse(read(suite/'context.json')['formal_release_allowed'])
            self.assertEqual(read(suite/'migration-source'/diagnostic.JOB/'config.json'),old)

    def test_only_one_candidate_recovery_then_actual_calibration_and_no_supervisor(self):
        self.candidate_handoff()

    def test_fenced_recovery_is_observation_without_calibration_or_admission(self):
        import os
        for fences in (False,True):
            with patch.dict(os.environ,{'QKAN_DIAGNOSTIC_STAGE_FENCES':'ambient'}):
                self.candidate_handoff(fences)
                self.assertEqual(os.environ['QKAN_DIAGNOSTIC_STAGE_FENCES'],'ambient')

    def test_generic_backend_sweep_does_not_activate_parallel_candidate(self):
        choices=[]
        def probe(config):
            choices.append(config['torch_acceleration']['backend'])
            raise RecoveryMismatch('candidate rejected')
        with self.assertRaises(RecoveryMismatch): admit_recovery(dict(torch_acceleration=policy()),probe)
        self.assertNotIn('eager_parallel',choices)
        choices.clear()
        with self.assertRaises(AccelerationRequired):
            admit_recovery(dict(torch_acceleration=policy(backend='eager_parallel',required='graph')),probe)
        self.assertEqual(choices,['eager_parallel'])

    def test_failure_export_uses_only_declared_compact_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'preflight'/diagnostic.JOB;folder.mkdir(parents=True)
            (folder/'uninterrupted.log').write_text('private logs and inputs')
            atomic_json(folder/'child-failure.json',dict(returncode=1))
            result=diagnostic.failure_details(Path(temp),ValueError('secret'))
            self.assertEqual(result,{'preflight/'+diagnostic.JOB+'/child-failure.json':{'returncode':1}})

    def test_failed_candidate_is_automatically_returned_without_claiming_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            original=ValueError('private native exception')
            with patch.object(diagnostic,'ROOT',Path(temp)),\
                patch('sys.argv',['attention','--migrate-from','original','--cache','cache']),\
                patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                patch.object(diagnostic,'execute',side_effect=original) as execute,\
                patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload:
                with self.assertRaises(ValueError) as caught: diagnostic.main()
                self.assertIs(caught.exception,original);execute.assert_called_once();upload.assert_called_once()
                suite,namespace,files=upload.call_args.args
                self.assertEqual(namespace,'parallel-attention');self.assertEqual(files,['parallel-attention.json'])
                report=read(suite/files[0]);self.assertFalse(report['passed']);self.assertFalse(report['formal_training_started'])
                self.assertNotIn('samples_committed',report);self.assertNotIn('private',str(report))

    def test_child_timeout_and_uncertain_cleanup_reach_automatic_failure_return(self):
        from experiments.baseline_delivery.diagnostic_watch import DiagnosticFailure
        child=dict(schema='qkan-diagnostic-child.v1',outcome='phase_timeout',
                   termination=dict(parent_reaped=True,group_verified=False,remaining_pids=[123]))
        def timeout(args,suite,report):
            report['child_executions']=[child]
            raise DiagnosticFailure('phase_timeout',child)
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(diagnostic,'ROOT',Path(temp)),\
                patch('sys.argv',['attention','--migrate-from','original','--cache','cache']),\
                patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                patch.object(diagnostic,'execute',side_effect=timeout),\
                patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload:
                with self.assertRaises(DiagnosticFailure): diagnostic.main()
                suite,namespace,files=upload.call_args.args
                report=read(suite/files[0])
                self.assertEqual(namespace,'parallel-attention')
                self.assertEqual(report['child_executions'],[child])
                self.assertEqual(report['diagnostics']['child_execution'],child)
                self.assertFalse(report['passed']);self.assertFalse(report['formal_training_started'])


if __name__=='__main__': unittest.main()
