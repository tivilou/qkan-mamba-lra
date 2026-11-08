"""Compiler/graph orchestration tests with injected runtimes; never import Torch."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery.torch_acceleration import Executor,BackendFailure,policy,torch_runtime,admit_recovery,RecoveryMismatch
from experiments.baseline_delivery.migration import compatible,DYNAMIC_REVISION
from experiments.baseline_delivery.performance import policy as performance_policy
from experiments.baseline_delivery.logical_batch import policy as logical_policy


class OOM(RuntimeError): pass
class CompilerError(RuntimeError): pass


class Runtime:
    def __init__(self,failures=()): self.failures=list(failures);self.calls=[];self.clears=0
    def set_mode(self,mode): self.mode=mode
    def clear(self): self.clears+=1
    def backend_error(self,error): return isinstance(error,CompilerError)
    def forward(self,batch,mode):
        self.calls.append(mode)
        if self.failures: raise self.failures.pop(0)
        return mode


class AccelerationTests(unittest.TestCase):
    def setup_executor(self,synthetic=False,candidate=.5,saved=None,failures=()):
        self.clock=0.;self.state=dict(weight=3,optimizer=7,scheduler=9,rng=11,cursor=17)
        self.original=copy.deepcopy(self.state);self.runtime=Runtime(failures);self.events=[];self.trials=[]
        executor=Executor(self.runtime,policy(),saved,synthetic,self.events.append,lambda:self.clock,self.trials.append)
        def update():
            mode=executor.forward({'rows':(17,18)})
            for key in ('weight','optimizer','scheduler','rng'): self.state[key]+=1
            self.clock+=1 if mode=='eager' else candidate
            return mode
        def restore(snapshot): self.state=copy.deepcopy(snapshot)
        self.run=lambda:executor.run(update,self.original,restore,lambda:None,lambda:None,lambda e:isinstance(e,OOM))
        return executor

    def assert_one_commit(self):
        for key,value in self.original.items(): self.assertEqual(self.state[key],value+(key!='cursor'))

    def test_benchmark_restores_every_mutable_state_then_commits_once(self):
        executor=self.setup_executor();self.assertEqual(self.run(),'compile_graph');self.assert_one_commit()
        self.assertEqual(len(self.runtime.calls),7)
        self.assertEqual(executor.state['reason'],'benchmark_gain')
        self.assertEqual(executor.state['benchmark']['training_updates_committed'],0)
        self.assertEqual(self.trials,[True,False])

    def test_slower_candidate_commits_eager_once(self):
        executor=self.setup_executor(candidate=2.);self.assertEqual(self.run(),'eager');self.assert_one_commit()
        self.assertEqual(executor.state['reason'],'benchmark_no_gain')

    def test_any_positive_gain_is_kept(self):
        executor=self.setup_executor(candidate=.999);self.assertEqual(self.run(),'compile_graph')
        self.assertEqual(executor.state['reason'],'benchmark_gain')

    def test_canary_exercises_requested_backend_without_timing_decisions(self):
        executor=self.setup_executor(synthetic=True);self.assertEqual(self.run(),'compile_graph')
        self.assertEqual(len(self.runtime.calls),1);self.assertFalse(executor.state['benchmarked'])

    def test_saved_selection_resumes_without_repeating_optimizer_trials(self):
        saved=dict(mode='compile',benchmarked=True,reason='benchmark_gain',benchmark={})
        executor=self.setup_executor(saved=saved);self.assertEqual(self.run(),'compile')
        self.assertEqual(len(self.runtime.calls),1);self.assertEqual(executor.snapshot(),saved)

    def test_compiler_and_capture_errors_restore_then_fallback_finitely(self):
        executor=self.setup_executor(synthetic=True,failures=[BackendFailure('capture'),CompilerError('unsupported')])
        self.assertEqual(self.run(),'eager');self.assert_one_commit()
        self.assertEqual(self.runtime.calls,['compile_graph','compile','eager'])
        self.assertEqual(executor.snapshot()['mode'],'eager')

    def test_graph_oom_releases_pool_and_retries_same_update(self):
        executor=self.setup_executor(synthetic=True,failures=[OOM('graph pool')])
        self.assertEqual(self.run(),'compile');self.assert_one_commit();self.assertEqual(self.runtime.clears,1)

    def test_compile_oom_propagates_to_existing_batch_controller(self):
        self.setup_executor(synthetic=True,saved=dict(mode='compile',benchmarked=False),failures=[OOM('capacity')])
        with self.assertRaises(OOM): self.run()

    def test_real_model_errors_are_not_disguised_as_backend_failure(self):
        self.setup_executor(synthetic=True,failures=[ValueError('nonfinite')])
        with self.assertRaises(ValueError): self.run()
        self.assertEqual(self.runtime.clears,0)

    def test_poisoned_cuda_context_is_never_retried(self):
        self.setup_executor(synthetic=True,failures=[BackendFailure('CUDA illegal memory access')])
        with self.assertRaises(BackendFailure): self.run()
        self.assertEqual(len(self.runtime.calls),1)

    def test_policy_change_and_invalid_saved_modes_rejected(self):
        with self.assertRaises(ValueError): Executor(Runtime(),dict(policy(),precision='amp'))
        with self.assertRaises(ValueError): Executor(Runtime(),policy(),saved=dict(mode='unknown'))

    def test_9a_migration_accepts_only_declared_execution_fields(self):
        old=dict(source_revision=DYNAMIC_REVISION,model='transformer',batch_size=32,lr=.001,adaptive_memory=logical_policy(32))
        new=dict(old,source_revision='next',performance=performance_policy(),torch_acceleration=policy())
        compatible(old,new)
        for change in (dict(lr=.1),dict(batch_size=64),dict(torch_acceleration={}),dict(performance={})):
            with self.assertRaises(ValueError): compatible(old,dict(new,**change))

    def test_native_binding_captures_one_shape_and_restores_rng(self):
        class Tensor:
            dtype='int64';device='cuda:0'
            def __init__(self,n=4): self.shape=(n,4000)
            def detach(self): return self
            def clone(self): return Tensor(self.shape[0])
        class Module:
            def __init__(self): pass
        model=Mock();model.forward=Mock(return_value='logits')
        model.parameters.return_value=[]
        torch=SimpleNamespace(nn=SimpleNamespace(Module=Module),compile=Mock(return_value=model.forward),
            cuda=SimpleNamespace(make_graphed_callables=Mock(return_value=lambda *args:'graph'),synchronize=Mock(),empty_cache=Mock()))
        get_rng=Mock(return_value={'rng':'saved'});set_rng=Mock()
        runtime=torch_runtime(torch,model,get_rng,set_rng,policy());runtime.set_mode('compile_graph')
        with patch('experiments.baseline_delivery.torch_acceleration.gc.collect'):
            for _ in range(2): self.assertEqual(runtime.forward({'inputs':Tensor()},'compile_graph'),'graph')
            self.assertEqual(torch.cuda.make_graphed_callables.call_count,1)
            self.assertEqual(runtime.forward({'inputs':Tensor(8)},'compile_graph'),'graph')
        self.assertEqual(torch.cuda.make_graphed_callables.call_count,2)
        self.assertEqual(set_rng.call_count,2);set_rng.assert_called_with({'rng':'saved'})
        kwargs=torch.compile.call_args.kwargs
        self.assertTrue(kwargs['options']['fallback_random']);self.assertFalse(kwargs['options']['triton.cudagraphs'])
        self.assertIs(model.forward,torch.compile.call_args.args[0])
        runtime.isolate_gradients();self.assertEqual(runtime.status()['graph_builds'],2)

    def test_parameter_gradient_storage_is_cloned_without_creating_unused_grads(self):
        grad=Mock();parameter=SimpleNamespace(grad=grad);unused=SimpleNamespace(grad=None)
        model=Mock();model.parameters.return_value=[parameter,unused]
        runtime=torch_runtime(Mock(),model,Mock(),Mock(),policy());runtime.isolate_gradients()
        self.assertIs(parameter.grad,grad.clone.return_value);self.assertIsNone(unused.grad)

    def test_native_agreement_check_is_rolled_back_before_real_update(self):
        executor=self.setup_executor(synthetic=True)
        executor.read_state=lambda:copy.deepcopy(self.state)
        executor.compare=lambda a,b:self.assertEqual(a,b)
        self.run();self.assert_one_commit();self.assertEqual(len(self.runtime.calls),4)
        self.assertEqual(executor.state['eager_agreement'],'passed_rtol1e-5_atol1e-6_rng_exact')

    def test_cold_backward_warmup_is_discarded_before_agreement(self):
        executor=self.setup_executor(synthetic=True);reads=[]
        executor.read_state=lambda:(reads.append(len(self.runtime.calls)) or copy.deepcopy(self.state))
        executor.compare=lambda a,b:self.assertEqual(a,b)
        self.run();self.assert_one_commit()
        self.assertEqual(reads,[2,3])
        self.assertEqual(self.runtime.calls,['compile_graph','eager','compile_graph','compile_graph'])

    def test_shape_change_rechecks_once_without_repeating_speed_benchmark(self):
        executor=self.setup_executor(synthetic=True)
        executor.read_state=lambda:copy.deepcopy(self.state);executor.compare=Mock()
        def update(key):
            return executor.run(lambda:executor.forward({}),self.original,lambda _:None,
                Mock(),Mock(),lambda _:False,execution_key=key)
        for key in ((16,128),(16,128),(32,128),(32,128),(16,128)):
            update(key)
        self.assertEqual(executor.compare.call_count,3)  # Graph recaptured on return to16.
        self.assertFalse(any(e.get('phase')=='bounded_benchmark' for e in self.events))

    def test_shape_validation_budget_falls_back_instead_of_endless_trials(self):
        executor=self.setup_executor(synthetic=True);executor.read_state=lambda:{};executor.compare=Mock()
        for i in range(20):
            executor.run(lambda:executor.forward({}),{},lambda _:None,Mock(),Mock(),lambda _:False,execution_key=i)
        self.assertEqual(executor.state['mode'],'eager')
        self.assertLessEqual(executor.compare.call_count,2*policy()['max_builds'])

    def test_restarted_process_warms_again_but_same_shape_does_not(self):
        executor=self.setup_executor(synthetic=True);executor.read_state=lambda:{};executor.compare=Mock()
        self.run();first=len(self.runtime.calls);self.run()
        self.assertEqual(len(self.runtime.calls)-first,1)
        restarted=Executor(Runtime(),policy(),saved=executor.snapshot(),synthetic=True,read_state=lambda:{},compare=Mock())
        restarted.run(lambda:restarted.forward({}),{},lambda _:None,Mock(),Mock(),lambda _:False)
        self.assertEqual(restarted.runtime.calls,['compile_graph','eager','compile_graph','compile_graph'])

    def test_numerical_mismatch_reverts_both_accelerated_modes_without_extra_commit(self):
        executor=self.setup_executor(synthetic=True)
        executor.read_state=lambda:copy.deepcopy(self.state)
        executor.compare=Mock(side_effect=AssertionError('different gradients'))
        self.assertEqual(self.run(),'eager');self.assert_one_commit()
        self.assertEqual(executor.state['reason'],'NumericalMismatch')
        self.assertEqual(executor.compare.call_count,2)

    def test_late_backward_error_restores_partially_modified_optimizer_rng(self):
        runtime=Runtime();executor=Executor(runtime,policy(),synthetic=True)
        state={'optimizer':4,'rng':8};snapshot=copy.deepcopy(state);attempts=[]
        def callback():
            attempts.append(executor.state['mode']);state['optimizer']+=1;state['rng']+=1
            if len(attempts)==1: raise CompilerError('late backward compilation')
            return state['optimizer']
        def restore(value): state.clear();state.update(value)
        self.assertEqual(executor.run(callback,snapshot,restore,Mock(),Mock(),lambda _:False),5)
        self.assertEqual(state,{'optimizer':5,'rng':9});self.assertEqual(attempts,['compile_graph','compile'])

    def test_full_recovery_mismatch_restarts_next_backend_without_relaxing_check(self):
        original=dict(torch_acceleration=policy(),lr=.001);calls=[]
        def probe(config):
            mode=config['torch_acceleration']['mode'];calls.append(mode)
            if mode=='compile_graph': raise RecoveryMismatch('Recovery tensor differs at model.encoder')
            return [dict(torch_acceleration=dict(state=dict(mode=mode)))]
        selected,receipts,attempts=admit_recovery(original,probe)
        self.assertEqual(calls,['compile_graph','compile']);self.assertEqual(selected['torch_acceleration'],policy('compile'))
        self.assertEqual(original['torch_acceleration'],policy());self.assertEqual(attempts[0]['outcome'],'recovery_mismatch')

    def test_failed_attempt_keeps_compact_diagnostic_even_if_fallback_passes(self):
        def probe(config):
            mode=config['torch_acceleration']['mode']
            if mode=='compile_graph':
                error=RecoveryMismatch('Recovery tensor differs');error.diagnostic={'control_and_rng_exact':True}
                raise error
            return [dict(torch_acceleration=dict(state=dict(mode=mode)))]
        _,_,attempts=admit_recovery(dict(torch_acceleration=policy()),probe)
        self.assertEqual(attempts[0]['diagnostic'],{'control_and_rng_exact':True})

    def test_diagnostic_covers_all_fields_caps_detail_and_contains_no_values(self):
        from experiments.baseline_delivery.runner import recovery_diagnostic
        def rows(a,b,key):
            return [dict(path=key+'/'+str(i),exact=key=='rng',max_abs=.001,differing_elements=1) for i in range(40)]
        with patch('experiments.baseline_delivery.recovery_audit.differences',side_effect=rows):
            result=recovery_diagnostic([dict(model={},optimizer={},rng={})]*2,['model','optimizer','rng'])
        self.assertEqual(result['differing_leaves'],80)
        self.assertEqual(len(result['first_differences']),32);self.assertTrue(result['details_truncated'])
        self.assertIsNone(result['control_and_rng_exact']);self.assertFalse(result['fields']['optimizer']['exact'])

    def test_all_modes_fail_recovery_blocks_formal_run(self):
        probe=Mock(side_effect=RecoveryMismatch('differs'))
        with self.assertRaises(RecoveryMismatch): admit_recovery(dict(torch_acceleration=policy()),probe)
        self.assertEqual(probe.call_count,3)

    def test_worker_fallback_is_retested_from_beginning_at_actual_shared_mode(self):
        probe=Mock(side_effect=[
            [dict(torch_acceleration=dict(state=dict(mode='compile'))),dict(torch_acceleration=dict(state=dict(mode='eager')))],
            [dict(torch_acceleration=dict(state=dict(mode='eager')))]])
        selected,_,attempts=admit_recovery(dict(torch_acceleration=policy()),probe)
        self.assertEqual(selected['torch_acceleration'],policy('eager'))
        self.assertEqual(probe.call_count,2);self.assertEqual(attempts[0]['outcome'],'worker_fallback')

    def test_data_trace_and_poisoned_context_errors_never_trigger_admission_fallback(self):
        for error in (ValueError('Trace semantic/replay gate failed'),RuntimeError('CUDA illegal memory access')):
            probe=Mock(side_effect=error)
            with self.assertRaises(type(error)): admit_recovery(dict(torch_acceleration=policy()),probe)
            self.assertEqual(probe.call_count,1)

    def test_admitted_mode_is_formal_worker_start_and_checkpoint_ceiling(self):
        executor=Executor(Runtime(),policy('compile'))
        self.assertEqual(executor.state['mode'],'compile')
        with self.assertRaises(ValueError): Executor(Runtime(),policy('compile'),saved=dict(mode='compile_graph'))
        self.assertEqual(Executor(Runtime(),policy('eager')).state['mode'],'eager')

    def test_9a_migration_allows_only_declared_recovery_mode_selection(self):
        old=dict(source_revision=DYNAMIC_REVISION,model='transformer',batch_size=32,adaptive_memory=logical_policy(32))
        for mode in ('compile_graph','compile','eager'):
            new=dict(old,source_revision='next',performance=performance_policy(),torch_acceleration=policy(mode))
            compatible(old,new)
            with self.assertRaises(ValueError): compatible(old,dict(new,torch_acceleration=dict(policy(mode),precision='bf16')))

    def test_exact_diagnostic_returns_state_path_and_modes_without_tensor_values(self):
        import sys,tempfile
        from pathlib import Path
        from experiments.baseline_delivery.runner import compare_recovery
        from experiments.baseline_delivery.common import read
        class Tensor:
            def __init__(self,value): self.value=value
        fake_torch=SimpleNamespace(Tensor=Tensor,equal=lambda a,b:a.value==b.value)
        fake_numpy=SimpleNamespace(ndarray=type('Array',(),{}))
        states=[dict(model={'weight':Tensor(n)},torch_acceleration={'mode':'compile'}) for n in (1,2)]
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,torch=fake_torch,numpy=fake_numpy):
            with self.assertRaisesRegex(RecoveryMismatch,'model.weight'):
                compare_recovery(states,['model'],Path(folder),dict(model='transformer',task='retrieval',torch_acceleration=policy()))
            report=read(Path(folder)/'recovery-validation.json')
            self.assertFalse(report['passed']);self.assertFalse(report['threshold_relaxed'])
            self.assertEqual(report['torch_acceleration'],[dict(mode='compile')]*2)

    def test_invalid_receipt_cannot_become_recovery_admission(self):
        for receipts in ([],[dict(torch_acceleration=dict(state=dict(mode='unknown')))]):
            with self.assertRaises(ValueError): admit_recovery(dict(torch_acceleration=policy()),lambda _:receipts)

    def test_real_readiness_route_binds_selected_mode_to_config_manifest_and_certificate(self):
        from contextlib import ExitStack
        from pathlib import Path
        import os,tempfile
        from experiments.baseline_delivery import readiness
        from experiments.baseline_delivery.common import atomic_json,read,fingerprint
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary);suite=root/'runs/suite';suite.mkdir(parents=True)
            atomic_json(root/'experiments/baseline_delivery/protocol.json',{})
            atomic_json(suite/'transformer-source-probe.json',dict(passed=True,implementation_sha256='impl'))
            args=SimpleNamespace(migrate_from=None,adaptive_memory=True,dynamic_logical_batch=True,
                performance=True,torch_acceleration=True,cache=root/'cache',gpus='0,1',gates_only=True)
            calls=[]
            def canary(config,data,cache,folder,device):
                mode=config['torch_acceleration']['mode'];gpu=os.environ['CUDA_VISIBLE_DEVICES'];calls.append((mode,gpu))
                if mode=='compile_graph' and gpu=='1': raise RecoveryMismatch('Recovery tensor differs at model.weight')
                for name in ('uninterrupted','interrupted'): atomic_json(folder/name/'trace-validation.json',dict(passed=True))
                return dict(passed=True,trace_semantic_and_replay=True,fresh_process=True,test_access=False,
                    config_sha256=fingerprint(config),torch_acceleration=dict(state=dict(mode=mode)))
            patches={
                'experiments.baseline_delivery.readiness.ROOT':root,
                'experiments.baseline_delivery.readiness.MODELS':('transformer',),
                'experiments.baseline_delivery.readiness.TASKS':('retrieval',),
                'experiments.baseline_delivery.readiness.revision':lambda:'new',
                'experiments.baseline_delivery.readiness.policy_check':lambda:('impl','policy'),
                'experiments.baseline_delivery.readiness.validate_runtime_release':lambda *a:None,
                'experiments.baseline_delivery.readiness.run_child':lambda *a:0,
                'experiments.baseline_delivery.runner.canary':canary,
                'experiments.baseline_delivery.runner.config_for':lambda *a:dict(model='transformer',task='retrieval',seed=42,batch_size=32),
                'experiments.baseline_delivery.runner.discover_gpus':lambda *a:(['0','1'],{'0':'gpu0','1':'gpu1'}),
                'experiments.baseline_delivery.sync.prepare':lambda *a:root/'data',
                'experiments.baseline_delivery.bootstrap.check_core':lambda:None,
                'experiments.baseline_delivery.bootstrap.check_mamba':lambda:None,
                'experiments.baseline_delivery.migration.plan_for':lambda *a:None,
                'experiments.baseline_delivery.migration.admit':lambda *a:None,
                'experiments.baseline_delivery.return_artifacts.upload_declared':lambda *a:None}
            for target,value in patches.items(): stack.enter_context(patch(target,value))
            stack.enter_context(patch.dict(os.environ,{},clear=False))
            readiness.execute(args,suite)
            self.assertEqual(calls,[('compile_graph','0'),('compile_graph','1'),('compile','0'),('compile','1')])
            job=suite/'jobs/transformer-retrieval-s42';config=read(job/'config.json')
            self.assertEqual(config['torch_acceleration'],policy('compile'))
            self.assertEqual(read(suite/'suite.json')['jobs'][0]['config_sha256'],fingerprint(config))
            certificate=read(suite/'transformer-certificate.json')
            self.assertIn((job/'acceleration-admission.json').relative_to(root).as_posix(),certificate['evidence'])
            self.assertFalse(any('compile_graph' in name for name in certificate['evidence']))
            # The real second readiness entry must consume its selected config,
            # retaining original fallback attempts and all certificate-bound bytes.
            original=(job/'acceleration-admission.json').read_bytes()
            sealed=(suite/'gate-evidence.json').read_bytes()
            readiness.execute(args,suite)
            self.assertEqual(len(calls),4)
            self.assertEqual(read(job/'config.json')['torch_acceleration'],policy('compile'))
            self.assertEqual((job/'acceleration-admission.json').read_bytes(),original)
            self.assertEqual((suite/'gate-evidence.json').read_bytes(),sealed)


if __name__=='__main__': unittest.main()
