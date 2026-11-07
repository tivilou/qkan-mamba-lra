"""Compiler/graph orchestration tests with injected runtimes; never import Torch."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery.torch_acceleration import Executor,BackendFailure,policy,torch_runtime
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
        self.run();self.assert_one_commit();self.assertEqual(len(self.runtime.calls),3)
        self.assertEqual(executor.state['eager_agreement'],'passed_rtol1e-5_atol1e-6_rng_exact')

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


if __name__=='__main__': unittest.main()
