"""Partition VJPs, stream/pool order and sealed diagnostics using stdlib fakes."""
from contextlib import contextmanager,nullcontext
import argparse
import ast
import copy
import os
import time
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from tests import test_baseline_parallel_attention as fake
from experiments.baseline_delivery import partitioned_diagnostic as diagnostic
from experiments.baseline_delivery.common import ROOT,atomic_json,read,fingerprint
from experiments.baseline_delivery.partitioned_attention import stream_graphed_chain,partitioned_encoder
from experiments.baseline_delivery.partition_comparison import (compare_routes,validate_comparison,
    execution_validation,prepare_route)
from experiments.baseline_delivery.torch_acceleration import policy,torch_runtime,graph_rng_budget,BackendFailure,Executor
from experiments.baseline_delivery.migration import compatible,PARALLEL_REVISION
from experiments.baseline_delivery.logical_batch import policy as logical_policy
from experiments.baseline_delivery.performance import policy as performance_policy


class PartitionCaptureTests(unittest.TestCase):
    def build(self):
        torch,events,rng=fake.framework();graphs=[]
        class Graph:
            def __init__(self): graphs.append(self)
            def replay(self): events.append(('replay',graphs.index(self)))
        @contextmanager
        def capture(graph,pool,stream):
            events.append(('capture',graphs.index(graph),pool,stream))
            yield
        torch.cuda.CUDAGraph=Graph;torch.cuda.graph=capture;torch.cuda.graph_pool_handle=lambda:object()
        torch.empty_like=lambda x:fake.Tensor(0)
        torch.autograd.grad.side_effect=lambda output,surface,gradient,allow_unused:tuple(fake.Tensor(100+i) for i in range(len(surface)))
        token=fake.Tensor(3);token.requires_grad=False
        hidden=fake.Tensor(6);a,b=fake.Tensor(2),fake.Tensor(4);stream=torch.cuda.Stream()
        previous=len(fake.FakeFunction.__subclasses__())
        call=stream_graphed_chain(torch,(lambda x:fake.Tensor(x.value*2),lambda h,x:fake.Tensor(h.value*4)),
            ((token,),(hidden,token.clone())),((a,),(b,)),stream,2)
        return NS(torch=torch,events=events,call=call,token=token,a=a,b=b,stream=stream,
                  wrappers=fake.FakeFunction.__subclasses__()[previous:])

    def test_capture_order_pool_and_hidden_input_vjp(self):
        row=self.build();captures=[e for e in row.events if e[0]=='capture']
        self.assertEqual([e[1] for e in captures],[0,1,3,2])  # F0,F1,B1,B0
        self.assertEqual(len({id(e[2]) for e in captures}),1)
        self.assertTrue(all(e[3] is row.stream for e in captures))
        syncs=row.torch.cuda.synchronize.call_count
        row.call(row.token)
        first,second=row.wrappers
        self.assertIs(first.last_args[-1],row.a);self.assertIs(second.last_args[-1],row.b)
        values=second.backward(second.last_context,fake.Tensor(5))
        self.assertEqual([v.value if v is not None else None for v in values],[100,None,101])
        values=first.backward(first.last_context,values[0])
        self.assertEqual([v.value if v is not None else None for v in values],[None,100])
        self.assertEqual(row.torch.cuda.synchronize.call_count,syncs)
        self.assertEqual([e[1] for e in row.events if e[0]=='replay'],[0,1,3,2])

    def test_documents_get_distinct_pools_and_invalid_surfaces_stop_early(self):
        a,b=self.build(),self.build()
        self.assertIsNot(next(e[2] for e in a.events if e[0]=='capture'),
                         next(e[2] for e in b.events if e[0]=='capture'))
        torch,_,_=fake.framework()
        for inputs in ((),((fake.Tensor(1),),(fake.Tensor(2),))):
            with self.assertRaises(BackendFailure):
                stream_graphed_chain(torch,(lambda x:x,lambda x:x),inputs,((),()),object(),1)

    def test_halves_preserve_mask_position_layer_order_and_parameter_names(self):
        torch,_,_=fake.framework();events=[]
        class Value(fake.Tensor):
            def ne(self,value): events.append(('padding_id',value));return Value(1)
            def __invert__(self): return self
            def unsqueeze(self,*args): return self
            def masked_fill(self,mask,value): events.append(('mask_fill',value));return self
            def __getitem__(self,key): events.append(('position_slice',));return self
            def __add__(self,value): events.append(('position_add',));return self
            def sum(self,*args): return self
            def __truediv__(self,value): return self
            def clone(self): return Value(self.value,self.shape)
            def requires_grad_(self,value): self.requires_grad=value;return self
        torch.zeros_like=lambda value,dtype: (events.append(('mask_dtype',dtype)) or Value(0))
        layers=[Mock(side_effect=lambda hidden,**kw:(events.append(('layer',kw)) or hidden)) for _ in range(4)]
        names=['embedding.weight',*[f'encoder.layers.{i}.weight' for i in range(4)],'norm.weight','head.weight']
        parameters=[fake.Tensor(i) for i in range(7)]
        replica=NS(encoder=NS(layers=layers,norm=None),embedding=lambda x:Value(2),position=Value(1),
                   norm=lambda x:x,named_parameters=lambda:iter(zip(names,parameters)))
        def chain(torch,calls,statics,surfaces,stream,warmup):
            self.assertEqual(surfaces,(tuple(parameters[:3]),tuple(parameters[3:6])))
            self.assertTrue(statics[1][0].requires_grad)
            self.assertIsNot(statics[0][0],statics[1][1])
            return lambda x:calls[1](calls[0](x),x)
        token=Value(3);token.requires_grad=False
        with patch('experiments.baseline_delivery.partitioned_attention.stream_graphed_chain',side_effect=chain):
            call,compiled=partitioned_encoder(torch,replica,token,torch.cuda.Stream(),3)
            events.clear();call(token)
        self.assertEqual(len(compiled),2)
        self.assertEqual([layer.call_count for layer in layers],[2,2,1,1])  # first sample warmup + one full encode
        self.assertEqual(events.count(('position_add',)),1)
        self.assertEqual(events.count(('mask_fill',float('-inf'))),2)
        self.assertEqual(events.count(('mask_dtype','float32')),2)
        self.assertTrue(all(row[1]['is_causal'] is False for row in events if row[0]=='layer'))
        self.assertEqual(names[-1],'head.weight')

    def test_rng_bound_and_sync_once_per_full_update(self):
        shape=fake.Model.config
        self.assertFalse(graph_rng_budget(shape,32,4000)['passed'])
        self.assertTrue(graph_rng_budget({**shape,'layers':2},32,4000)['passed'])
        torch,_,rng=fake.framework();model=fake.Model()
        runtime=torch_runtime(torch,model,lambda:rng[0],lambda value:rng.__setitem__(0,value),
                              policy('compile_graph','eager_partitioned','graph'))
        def encoder(torch,replica,tokens,stream,warmup):
            return (lambda x:replica.encode(x,True)),(object(),object())
        batch=dict(inputs1=fake.Tensor(3,(32,10)),inputs2=fake.Tensor(4,(32,10)))
        with patch('experiments.baseline_delivery.partitioned_attention.partitioned_encoder',side_effect=encoder):
            runtime.begin_update(32);runtime.forward(batch,'compile_graph');runtime.forward(batch,'compile_graph')
            self.assertEqual(runtime.status()['replica_sync_rounds'],1)
            runtime.begin_update(32);runtime.forward(batch,'compile_graph')
            self.assertEqual(runtime.status()['replica_sync_rounds'],2)
            self.assertEqual(runtime.status()['graph_replays'],12)
            runtime.clear();self.assertFalse(runtime.synced)
            runtime.begin_update(32);runtime.forward(batch,'compile_graph')
            self.assertEqual(runtime.status()['replica_sync_rounds'],3)
        self.assertEqual(runtime.status()['capture_partition'],'parallel_retrieval_layers_2')


class ComparisonTests(unittest.TestCase):
    def exercise(self,candidate_seconds=6,fail=None):
        resident=set();clock=[0];events=[];restores=Mock()
        def route(name,seconds):
            def prepare():
                self.assertFalse(resident);resident.add(name);events.append(('prepare',name));clock[0]+=50
                if fail==name: raise ValueError('prepare failed')
            def run():
                self.assertEqual(resident,{name});events.append(('run',name));clock[0]+=seconds
            def clear(): resident.discard(name);events.append(('clear',name))
            return dict(prepare=prepare,run=run,clear=clear,reset_peak=Mock(),
                validation=lambda:dict(mode='compile_graph',generation=1,validation_checks=1,
                    graph_resident=name in resident,validated=True))
        routes=dict(incumbent=route('incumbent',10),candidate=route('candidate',candidate_seconds))
        try: result=compare_routes(routes,restores,Mock(),lambda:dict(allocated_bytes=12),lambda:clock[0])
        finally: self.assertFalse(resident);self.assertGreater(restores.call_count,0)
        return result,events

    def test_balanced_full_update_measurement_excludes_rebuild_and_never_co_resides(self):
        report,events=self.exercise();validate_comparison(report)
        self.assertEqual(report['samples_seconds'],dict(incumbent=[10,10],candidate=[6,6]))
        self.assertEqual([e[1] for e in events if e[0]=='run'],['incumbent','candidate','candidate','incumbent'])
        self.assertEqual(report['throughput_gain_fraction'],10/6-1)
        self.assertTrue(all(row['before']==row['after'] for rows in report['steady_validation'].values() for row in rows))

    def test_no_gain_and_tampered_samples_are_rejected(self):
        for value in (10,12):
            report,_=self.exercise(value)
            with self.assertRaises(ValueError): validate_comparison(report)
        report,_=self.exercise()
        for key,value in (('candidate_seconds',1),('throughput_gain_fraction',10),('samples_committed',1)):
            with self.assertRaises(ValueError): validate_comparison({**report,key:value})
        with self.assertRaises(ValueError): validate_comparison({**report,'schema':'qkan-incumbent-graph-comparison.v1'})
        changed=copy.deepcopy(report);changed['steady_validation']['candidate'][0]['after']['validation_checks']+=1
        with self.assertRaises(ValueError): validate_comparison(changed)
        with self.assertRaises(ValueError): self.exercise(fail='candidate')

    def executor_routes(self,prepare=True,rebuild_in_timing=False):
        """Run the actual Executor, including its generation cache and checks."""
        timer=[0.];state={'weight':0};resident=set();records=[]
        def restore(): state['weight']=0
        class Runtime:
            def __init__(self,name,seconds): self.name=name;self.seconds=seconds;self.builds=1;self.graph=False
            def set_mode(self,mode): self.mode=mode
            def generation(self): return self.builds
            def status(self): return dict(graph_resident=self.graph)
            def clear(self): self.graph=False;resident.discard(self.name)
            def backend_error(self,error): return False
            def forward(self,batch,mode):
                if mode=='compile_graph' and not self.graph:
                    assert not resident
                    resident.add(self.name);self.builds+=1;self.graph=True;timer[0]+=50
                timer[0]+=10 if mode=='eager' else self.seconds
                state['weight']+=1
        routes={}
        for name,micro,seconds in (('incumbent',16,8),('candidate',32,7.8)):
            runtime=Runtime(name,seconds)
            executor=Executor(runtime,policy('compile_graph','eager_partitioned' if micro==32 else 'eager_parallel','graph'),
                saved=dict(mode='compile_graph',benchmarked=True,reason='prior_gain',benchmark={}),
                read_state=lambda:dict(state),compare=lambda a,b:self.assertEqual(a,b),trial=Mock())
            key=(micro,128);executor.validated.add(('compile_graph',key,1));executor.checks=1
            def route(executor,runtime,key):
                def run():
                    return executor.run(lambda:executor.forward({}),{'weight':0},lambda value:state.update(value),
                        lambda:None,lambda:None,lambda _:False,execution_key=key)
                validation=lambda:execution_validation(executor,key)
                def ready():
                    if prepare: prepare_route(run,validation,restore,lambda:None)
                    else: run()
                    if rebuild_in_timing: runtime.clear()
                return dict(run=run,prepare=ready,validation=validation,clear=runtime.clear,reset_peak=Mock())
            routes[name]=route(executor,runtime,key);records.append(executor)
        try: report=compare_routes(routes,restore,lambda:None,lambda:{},lambda:timer[0])
        finally: self.assertFalse(resident);self.assertEqual(state,{'weight':0})
        return report,records

    def test_recaptured_generation_is_revalidated_before_clock_starts(self):
        report,executors=self.executor_routes();validate_comparison(report)
        self.assertEqual(report['samples_seconds']['incumbent'],[8,8])
        for value in report['samples_seconds']['candidate']: self.assertAlmostEqual(value,7.8)
        self.assertGreater(executors[0].checks,1);self.assertGreater(executors[1].checks,1)
        self.assertEqual(len(report['preparation_memory']['incumbent']),2)
        self.assertEqual(len(report['preparation_memory']['candidate']),1)

    def test_old_one_call_prepare_and_forced_recapture_are_blocked_before_timing(self):
        with self.assertRaisesRegex(ValueError,'not resident and validated'):
            self.executor_routes(prepare=False)
        with self.assertRaisesRegex(ValueError,'not resident and validated'):
            self.executor_routes(rebuild_in_timing=True)

    def test_prepare_has_finite_budget_and_timing_cannot_include_new_validation(self):
        run=Mock();restore=Mock()
        value=dict(mode='compile_graph',generation=1,validation_checks=1,graph_resident=False,validated=False)
        with self.assertRaisesRegex(ValueError,'did not reach'):
            prepare_route(run,lambda:value,restore,lambda:None)
        self.assertEqual(run.call_count,3)
        value.update(graph_resident=True,validated=True)
        def changed(): value['validation_checks']+=1
        route=dict(prepare=Mock(),validation=lambda:dict(value),run=changed,clear=Mock(),reset_peak=Mock())
        with self.assertRaisesRegex(ValueError,'changed inside timing'):
            compare_routes(dict(incumbent=route,candidate=route),lambda:None,lambda:None,lambda:{})

    def test_cleanup_failure_still_releases_other_route_and_attempts_rollback(self):
        restore=Mock();clear=Mock()
        routes=dict(incumbent=dict(clear=Mock(side_effect=ValueError('clear failed')),prepare=Mock(side_effect=ValueError('prepare failed'))),
                    candidate=dict(clear=clear))
        routes['incumbent']['reset_peak']=Mock()
        with self.assertRaises(ValueError): compare_routes(routes,restore,Mock(),lambda:{})
        clear.assert_called_once();self.assertGreater(restore.call_count,0)


class DiagnosticTests(unittest.TestCase):
    def source(self):
        return dict(model='transformer',task='retrieval',seed=42,batch_size=32,source_revision=PARALLEL_REVISION,
                    performance=performance_policy(),adaptive_memory=logical_policy(32),
                    torch_acceleration=policy('compile_graph','eager_parallel','graph'))

    def test_only_exact_b07_scientific_contract_can_enter_candidate(self):
        old=self.source();new={**old,'source_revision':'new','torch_acceleration':policy('compile_graph','eager_partitioned','graph')}
        compatible(old,new)
        for changes in ({'batch_size':128},{'seed':43},{'task':'text'},{'performance':{}}):
            with self.assertRaises(ValueError): compatible(old,{**new,**changes})
        self.assertNotEqual(policy('compile_graph','eager_parallel','graph'),new['torch_acceleration'])
        with self.assertRaises(ValueError): policy('compile','eager_partitioned')

    def handoff(self,busy=False,wrong_micro=False):
        with tempfile.TemporaryDirectory() as temp:
            suite=Path(temp)/'suite';data=Path(temp)/'data';data.mkdir()
            old=self.source();atomic_json(suite/'migration-source'/diagnostic.JOB/'config.json',old)
            atomic_json(suite/'migration-source/plan.json',{})
            record=dict(checkpoint=dict(sha256='sha',step=12839))
            plan=dict(data=str(data),jobs={diagnostic.JOB:record,'completed':dict(completion={'passed':True})})
            dispatch=[]
            def canary(config,data,cache,folder,device,announce,run):
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'2')
                self.assertEqual(os.environ['QKAN_DIAGNOSTIC_STAGE_FENCES'],'0')
                run(dict(argv=['worker','--synthetic']),folder/'worker.log')
                micro=int(dispatch[-1]['argv'][-1])
                for name in ('uninterrupted','interrupted'):
                    atomic_json(folder/name/'adaptive-memory.json',dict(last_update=dict(used_micro_batch=16 if wrong_micro else micro)))
                return dict(passed=True,fresh_process=True,trace_semantic_and_replay=True,test_access=False,
                    config_sha256=fingerprint(config),source_revision='new',torch_acceleration=dict(state=dict(mode='compile_graph')))
            def calibration(*args,**kwargs):
                self.assertEqual(kwargs,dict(probe_micro_batch=32,compare_incumbent=True))
                self.assertEqual(args[7],record)
                path=suite/'calibration/result.json';atomic_json(path,dict(passed=True));return path
            with patch('experiments.baseline_delivery.readiness.policy_check',return_value=('impl','policy')), \
                 patch('experiments.baseline_delivery.migration.snapshot',return_value=plan), \
                 patch.object(diagnostic,'revision',return_value='new'), \
                 patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['2'],{'2':'gpu'})), \
                 patch.object(diagnostic,'gpu_preflight',return_value=dict(idle=not busy,compute_pids=[123] if busy else [])), \
                 patch('experiments.baseline_delivery.runner.canary',side_effect=canary) as check, \
                 patch('experiments.baseline_delivery.diagnostic_watch.Supervisor.run',side_effect=lambda spec,*a:dispatch.append(spec) or 0), \
                 patch('experiments.baseline_delivery.acceleration_calibration.calibrate',side_effect=calibration) as measure, \
                 patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'old','QKAN_DIAGNOSTIC_STAGE_FENCES':'1'}):
                report={};args=NS(child_timeout=900,stall_timeout=300,migrate_from='source',cache='cache',gpu='2')
                if busy or wrong_micro:
                    with self.assertRaises(ValueError): diagnostic.execute(args,suite,report)
                    measure.assert_not_called()
                    if busy: check.assert_not_called()
                else:
                    diagnostic.execute(args,suite,report)
                    self.assertTrue(report['passed']);self.assertEqual(set(report['recovery']),{'16','32'})
                    self.assertEqual([spec['argv'][-2:] for spec in dispatch],[['--probe-micro-batch','16'],['--probe-micro-batch','32']])
                    self.assertFalse(read(suite/'context.json')['formal_release_allowed'])
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'old')
                self.assertEqual(os.environ['QKAN_DIAGNOSTIC_STAGE_FENCES'],'1')

    def test_real_handoff_seals_profiles_and_preserves_source_environment(self): self.handoff()
    def test_busy_gpu_stops_without_launch_or_kill(self): self.handoff(busy=True)
    def test_micro_fallback_is_not_reported_as_candidate_success(self): self.handoff(wrong_micro=True)

    def test_worker_parser_retains_new_diagnostic_flags(self):
        tree=ast.parse((ROOT/'experiments/baseline_delivery/worker.py').read_text(encoding='utf8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        worker=Mock(return_value=0)
        with tempfile.TemporaryDirectory() as temp:
            cfg=Path(temp)/'config.json';atomic_json(cfg,{})
            scope=dict(argparse=argparse,Path=Path,read=read,lock=lambda _:nullcontext(),run=worker,
                       signal=__import__('signal'),request_pause=Mock())
            exec(compile(ast.Module(body=[node],type_ignores=[]),'worker-main','exec'),scope)
            with patch('sys.argv',['worker','--config',str(cfg),'--data','data','--cache','cache','--output',temp,
                                  '--device','cuda:0','--synthetic','--probe-micro-batch','32']),patch('signal.signal'):
                scope['main']()
        self.assertEqual(worker.call_args.kwargs,dict(probe_micro_batch=32,compare_incumbent=False))

    def test_worker_rejects_formal_partition_or_nonpartition_fixed_override_before_framework(self):
        tree=ast.parse((ROOT/'experiments/baseline_delivery/worker.py').read_text(encoding='utf8'))
        node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run'))
        # Execute the real entry guards only; stop before the first Torch call.
        index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr)
                   and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.set_num_threads')
        node.body=node.body[:index]
        scope=dict(os=os,time=time,Path=Path)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'worker-entry-guards','exec'),scope)
        with tempfile.TemporaryDirectory() as temp:
            def invoke(config,synthetic=False,micro=None,compare=False):
                return scope['_run'](config,'data','cache',temp,'cuda:0',False,None,synthetic,None,
                                     False,probe_micro_batch=micro,compare_incumbent=compare)
            candidate=dict(torch_acceleration=policy('compile_graph','eager_partitioned','graph'))
            with self.assertRaises(ValueError): invoke(candidate)
            with self.assertRaises(ValueError): invoke({},True,32)
            with self.assertRaises(ValueError): invoke(candidate,True,32,True)
            with patch.dict(os.environ,{'QKAN_DIAGNOSTIC_STAGE_FENCES':'0'}):
                self.assertIsNone(invoke(candidate,True,32))

    def test_calibration_seals_fixed_micro_and_graph_comparison_and_rejects_relabeling(self):
        from experiments.baseline_delivery.acceleration_calibration import calibrate,finish_probe
        from experiments.baseline_delivery.torch_acceleration import RecoveryMismatch
        with tempfile.TemporaryDirectory() as temp:
            config={**self.source(),'source_revision':'new',
                    'torch_acceleration':policy('compile_graph','eager_partitioned','graph')}
            record=dict(checkpoint=dict(sha256='sha',step=12839))
            comparison,_=ComparisonTests().exercise()
            def child(spec,log,env):
                argv=spec['argv'];self.assertEqual(argv[-3:],['--probe-micro-batch','32','--compare-incumbent'])
                folder=Path(argv[argv.index('--output')+1])
                selected=dict(mode='compile_graph',benchmarked=True,reason='benchmark_gain',
                              benchmark=dict(eager_seconds=10,accelerated_seconds=6))
                value=finish_probe(config,dict(step=12839,migration=dict(source_checkpoint_sha256='sha')),
                                   selected,[],lambda:None,dict(graph_replays=32))
                atomic_json(folder/'acceleration-probe.json',dict(value,requested_micro_batch=32,
                    observed_micro_batches=[32],incumbent_comparison=comparison))
                return 0
            suite=Path(temp);run=Mock(side_effect=child)
            build=lambda *args:dict(argv=list(map(str,args)))
            args=(suite,diagnostic.JOB,config,'data','cache','0','gpu',record,run,build)
            path=calibrate(*args,probe_micro_batch=32,compare_incumbent=True)
            self.assertEqual(path,calibrate(*args,probe_micro_batch=32,compare_incumbent=True));self.assertEqual(run.call_count,1)
            value=read(path);value['observed_micro_batches']=[16];atomic_json(path,value)
            with self.assertRaises(RecoveryMismatch): calibrate(*args,probe_micro_batch=32,compare_incumbent=True)
            value['observed_micro_batches']=[32];value['incumbent_comparison']['passed']=False;atomic_json(path,value)
            with self.assertRaises(RecoveryMismatch): calibrate(*args,probe_micro_batch=32,compare_incumbent=True)


if __name__=='__main__': unittest.main()
