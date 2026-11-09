"""Real-checkpoint calibration and required acceleration, with stdlib fakes only."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery.acceleration_calibration import calibrate,finish_probe,validate_context
from experiments.baseline_delivery.common import atomic_json,read,digest
from experiments.baseline_delivery.torch_acceleration import (Executor,policy,compare_update,
    NumericalMismatch,AccelerationRequired,RecoveryMismatch,admit_recovery,torch_runtime)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.suite=Path(self.temp.name)/'runs/suite';self.suite.mkdir(parents=True)
        self.config=dict(model='transformer',source_revision='new',torch_acceleration=policy(required='graph'))
        self.record=dict(checkpoint=dict(file='latest.pt',sha256='sha',step=123))
        self.state=dict(step=123,migration=dict(source_checkpoint_sha256='sha'))
        self.selected=dict(mode='compile_graph',benchmarked=True,reason='benchmark_gain',
            benchmark=dict(eager_seconds=2.,accelerated_seconds=1.,training_updates_committed=0))
        self.gpu=dict(uuid='gpu0',driver='native')
        self.runtime=dict(graph_replays=7,graph_resident=True)

    def invoke(self,run,build=None):
        return calibrate(self.suite,'transformer-retrieval-s42',self.config,Path('data'),Path('cache'),
            '0',self.gpu,self.record,run,build or (lambda *args:dict(argv=list(map(str,args)))))

    def native_child(self,code=0,change=None):
        def child(spec,log,env):
            argv=spec['argv'];folder=Path(argv[argv.index('--output')+1])
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
            self.assertEqual(argv[argv.index('--acceleration-probe-source')+1],str(self.suite))
            self.assertNotIn('--resume',argv);self.assertNotIn('--synthetic',argv)
            self.assertTrue(folder.is_relative_to(self.suite/'calibration'))
            selected=self.selected if code==0 else dict(mode='eager',benchmarked=False,reason='NumericalMismatch')
            value=finish_probe(self.config,self.state,selected,[],lambda:None,self.runtime)
            if change: value.update(change)
            atomic_json(folder/'acceleration-probe.json',value)
            return code
        return child

    def test_real_child_is_hash_bound_and_reused_without_repeating_benchmark(self):
        run=Mock(side_effect=self.native_child());path=self.invoke(run)
        first=path.read_bytes();self.assertEqual(self.invoke(run),path)
        self.assertEqual(path.read_bytes(),first);self.assertEqual(run.call_count,1)
        value=read(path);self.assertTrue(value['passed']);self.assertTrue(value['rollback_exact'])
        self.assertEqual(value['restored_step'],123);self.assertEqual(value['gpu_identity'],self.gpu)

    def test_changed_checkpoint_or_device_requires_new_calibration(self):
        run=Mock(side_effect=self.native_child());first=self.invoke(run)
        self.gpu=dict(uuid='gpu1',driver='native');second=self.invoke(run)
        self.assertNotEqual(first,second)
        self.record['checkpoint']['sha256']='next';self.state['migration']['source_checkpoint_sha256']='next'
        self.assertNotEqual(second,self.invoke(run));self.assertEqual(run.call_count,3)

    def test_failed_real_numerics_returns_diagnostic_for_next_profile(self):
        with self.assertRaises(RecoveryMismatch) as caught: self.invoke(self.native_child(76))
        self.assertEqual(caught.exception.diagnostic['actual']['mode'],'eager')
        self.assertEqual(caught.exception.diagnostic['samples_committed'],0)

    def test_fatal_child_fault_does_not_become_backend_fallback(self):
        for code in (1,75,137):
            with self.subTest(code=code),self.assertRaises(ValueError): self.invoke(lambda *args:code)

    def test_identity_forgery_and_unrolled_update_are_never_admitted(self):
        for change in (dict(samples_committed=1),dict(optimizer_updates_committed=1),dict(rollback_exact=False),
                dict(source_checkpoint_sha256='wrong'),dict(restored_step=0),dict(sample_clock_reset=True),
                dict(backend='wrong')):
            with tempfile.TemporaryDirectory() as temporary,self.subTest(change=change):
                self.suite=Path(temporary)/'suite'
                with self.assertRaises(ValueError): self.invoke(self.native_child(change=change))

    def test_report_requires_real_rollback_and_measured_positive_gain(self):
        rollback=Mock();report=finish_probe(self.config,self.state,self.selected,[],rollback,self.runtime)
        rollback.assert_called_once();self.assertTrue(report['passed'])
        with self.assertRaises(RuntimeError):
            finish_probe(self.config,self.state,self.selected,[],Mock(side_effect=RuntimeError('rollback differs')),self.runtime)
        for selected in (dict(self.selected,mode='compile'),dict(self.selected,benchmarked=False),
                dict(self.selected,benchmark=dict(eager_seconds=1.,accelerated_seconds=2.))):
            self.assertFalse(finish_probe(self.config,self.state,selected,[],lambda:None,self.runtime)['passed'])
        self.assertFalse(finish_probe(self.config,self.state,self.selected,[],lambda:None,dict(graph_replays=0))['passed'])

    def test_context_binds_source_implementation_review_and_migration_snapshot(self):
        atomic_json(self.suite/'migration-source/plan.json',dict(snapshot='native'))
        context=dict(source_revision='new',implementation_sha256='impl',policy_sha256='review',
            migration_sha256=digest(self.suite/'migration-source/plan.json'))
        atomic_json(self.suite/'context.json',context)
        validate_context(self.suite,self.config,'new','impl','review')
        atomic_json(self.suite/'migration-source/plan.json',dict(snapshot='changed'))
        with self.assertRaises(ValueError): validate_context(self.suite,self.config,'new','impl','review')

    def test_candidate_order_can_admit_native_aten_graph_after_inductor_mismatch(self):
        calls=[]
        def probe(config):
            selected=config['torch_acceleration'];calls.append((selected['mode'],selected['backend']))
            if selected['backend']=='inductor': raise RecoveryMismatch('model')
            return [dict(torch_acceleration=dict(state=dict(mode='compile_graph')))]
        chosen,_,attempts=admit_recovery(self.config,probe)
        self.assertEqual(calls,[('compile_graph','inductor'),('compile_graph','aot_eager_default_partitioner')])
        self.assertEqual(chosen['torch_acceleration']['required'],'graph');self.assertEqual(attempts[-1]['outcome'],'passed')

    def test_graph_requirement_never_admits_compile_or_eager(self):
        probe=Mock(side_effect=RecoveryMismatch('differs'))
        with self.assertRaises(AccelerationRequired) as caught: admit_recovery(self.config,probe)
        self.assertEqual(probe.call_count,2);self.assertEqual(len(caught.exception.diagnostic['attempts']),2)

    def test_required_runtime_mismatch_restores_before_lower_backend_or_commit(self):
        runtime=Mock();runtime.backend_error=lambda _:False
        state=dict(weight=4,adam=7,rng=9);original=copy.deepcopy(state);calls=[]
        executor=Executor(runtime,policy(required='graph'),read_state=lambda:copy.deepcopy(state),
            compare=Mock(side_effect=NumericalMismatch(dict(first_differences=[dict(path='state/observations/raw_gradients/w')]))))
        def update():
            calls.append((executor.state['mode'],executor.reference))
            for key in state: state[key]+=1
        def restore(value): state.clear();state.update(value)
        with self.assertRaises(AccelerationRequired):
            executor.run(update,original,restore,lambda:None,lambda:None,lambda _:False)
        self.assertEqual(state,original);self.assertEqual(len(calls),3)
        self.assertEqual(len(executor.diagnostics),1);self.assertFalse(executor.validating)

    def test_required_runtime_no_gain_rolls_back_instead_of_committing_eager(self):
        runtime=Mock();runtime.backend_error=lambda _:False;state={'updates':0};timer=[0.]
        executor=Executor(runtime,policy(required='graph'),clock=lambda:timer[0])
        def update(): state['updates']+=1;timer[0]+=1 if executor.reference else 2
        def restore(value): state.update(value)
        with self.assertRaises(AccelerationRequired):
            executor.run(update,{'updates':0},restore,lambda:None,lambda:None,lambda _:False)
        self.assertEqual(state['updates'],0);self.assertEqual(executor.state['reason'],'benchmark_no_gain')

    def test_diagnostics_cover_all_fields_even_after_detail_cap_and_do_not_export_values(self):
        left=dict(model={str(i):0 for i in range(40)},observations=dict(logits=1,raw_gradients=2),rng=3)
        right=dict(model={str(i):1 for i in range(40)},observations=dict(logits=4,raw_gradients=5),rng=3)
        with self.assertRaises(NumericalMismatch) as caught: compare_update(left,right,lambda *args:None,lambda *args:None)
        report=caught.exception.report
        self.assertEqual(len(report['first_differences']),32);self.assertTrue(report['details_truncated'])
        self.assertEqual(report['differing_leaves'],42);self.assertTrue(report['fields']['rng']['exact'])
        self.assertFalse(report['fields']['observations']['exact']);self.assertFalse(report['threshold_relaxed'])
        self.assertFalse(any('value' in key for item in report['first_differences'] for key in item))

    def test_native_aten_profile_uses_compile_without_inductor_options(self):
        model=Mock();torch=SimpleNamespace(compile=Mock(return_value=model.forward))
        runtime=torch_runtime(torch,model,Mock(),Mock(),policy('compile','aot_eager_default_partitioner'))
        runtime.forward({},'compile')
        self.assertEqual(torch.compile.call_args.kwargs,
            dict(backend='aot_eager_default_partitioner',fullgraph=True,dynamic=False))

    def test_required_graph_oom_retries_smaller_microbatch_and_commits_once(self):
        from experiments.baseline_delivery.adaptive_memory import Controller,default_policy,transaction
        class OOM(RuntimeError): pass
        runtime=Mock();runtime.backend_error=lambda _:False
        timer=[0.]
        executor=Executor(runtime,policy(required='graph'),saved=dict(mode='compile_graph',benchmarked=True,reason='benchmark_gain'),clock=lambda:timer[0])
        state=dict(weight=3,adam=5,rng=7);original=copy.deepcopy(state);tried=[]
        controller=Controller(default_policy(8),8)
        def restore(value): state.clear();state.update(value)
        def update(micro):
            def callback():
                tried.append(micro)
                timer[0]+=1 if executor.reference else .5
                for key in state: state[key]+=1
                if micro>2: raise OOM('graph allocation')
                return 1
            return executor.run(callback,original,restore,lambda:None,lambda:None,lambda error:isinstance(error,OOM),micro)
        result,status=transaction(controller,update,lambda:copy.deepcopy(state),restore,lambda:None,
            lambda error:isinstance(error,OOM),lambda:{},lambda:None)
        self.assertEqual(result,1);self.assertEqual(tried[0],4);self.assertEqual(set(tried[1:]),{2})
        self.assertEqual(controller.state['oom_count'],1)
        self.assertEqual(state,{key:value+1 for key,value in original.items()})
        self.assertEqual(executor.state['mode'],'compile_graph');runtime.clear.assert_called_once()

    def test_small_epoch_tails_preserve_hot_graph_and_actual_replay_counter(self):
        class Tensor:
            dtype='float32';device='cuda:0'
            def __init__(self,rows): self.shape=(rows,4000)
            def detach(self): return self
            def clone(self): return Tensor(self.shape[0])
        class Module:
            def __init__(self): pass
        model=Mock();model.parameters.return_value=[];model.return_value='native-tail'
        graph=Mock(return_value='graph')
        torch=SimpleNamespace(nn=SimpleNamespace(Module=Module),compile=Mock(return_value=model.forward),
            cuda=SimpleNamespace(make_graphed_callables=Mock(return_value=graph),synchronize=Mock(),empty_cache=Mock()))
        runtime=torch_runtime(torch,model,Mock(),Mock(),policy(required='graph'))
        runtime.set_mode('compile_graph');runtime.begin_update(16)
        with patch('experiments.baseline_delivery.torch_acceleration.gc.collect'):
            for _ in range(100):
                self.assertEqual(runtime.forward({'inputs':Tensor(16)},'compile_graph'),'graph')
                self.assertEqual(runtime.forward({'inputs':Tensor(3)},'compile_graph'),'native-tail')
            self.assertEqual(runtime.generation(),1);self.assertEqual(runtime.status()['graph_replays'],100)
            self.assertEqual(runtime.status()['eager_tail_calls'],100)
            # A real microbatch change, unlike a tail, replaces the main graph.
            runtime.begin_update(8);runtime.forward({'inputs':Tensor(8)},'compile_graph')
            self.assertEqual(runtime.generation(),2)

    def test_native_capture_preserves_oom_type_for_controller_retry(self):
        class OOM(RuntimeError): pass
        class Tensor:
            shape=(4,10);dtype='float32';device='cuda:0'
            def detach(self): return self
            def clone(self): return self
        class Module:
            def __init__(self): pass
        model=Mock();torch=SimpleNamespace(nn=SimpleNamespace(Module=Module),compile=Mock(return_value=model.forward),
            cuda=SimpleNamespace(OutOfMemoryError=OOM,make_graphed_callables=Mock(side_effect=OOM('allocation')),
                synchronize=Mock(),empty_cache=Mock()))
        runtime=torch_runtime(torch,model,Mock(),Mock(),policy(required='graph'));runtime.set_mode('compile_graph')
        with patch('experiments.baseline_delivery.torch_acceleration.gc.collect'),self.assertRaises(OOM):
            runtime.forward({'inputs':Tensor()},'compile_graph')

    def test_graph_rebuild_invalidates_agreement_but_cached_epoch_tail_does_not(self):
        runtime=Mock();runtime.backend_error=lambda _:False;generation=[1];runtime.generation=lambda:generation[0]
        comparison=Mock();executor=Executor(runtime,policy(required='graph'),synthetic=True,read_state=lambda:{},compare=comparison)
        for _ in range(100):
            for shape in ((16,128),(16,5)):
                executor.run(lambda:1,{},lambda _:None,lambda:None,lambda:None,lambda _:False,shape)
        self.assertEqual(comparison.call_count,2);self.assertEqual(executor.state['mode'],'compile_graph')
        generation[0]=2
        executor.run(lambda:1,{},lambda _:None,lambda:None,lambda:None,lambda _:False,(16,128))
        self.assertEqual(comparison.call_count,3)


if __name__=='__main__': unittest.main()
