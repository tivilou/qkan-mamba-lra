"""Calibration timing, rollback storage and diagnostics with stdlib stand-ins."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from experiments.baseline_delivery.execution import restore_optimizer_in_place
from experiments.baseline_delivery.torch_acceleration import Executor,policy,AccelerationRequired,calibration_summary


class TimingTests(unittest.TestCase):
    def setup_executor(self,candidate=1.,diagnose=None,synthetic=False):
        self.timer=0.;self.queued=0.;self.state={'weight':4,'adam':5,'rng':6};self.original=copy.deepcopy(self.state)
        self.calls=[];runtime=Mock();runtime.generation.return_value=0;runtime.backend_error=lambda _:False
        executor=Executor(runtime,policy(required='graph'),synthetic=synthetic,clock=lambda:self.timer,diagnose=diagnose)
        self.executor=executor
        def restore(value): self.state=copy.deepcopy(value);self.queued+=7
        def synchronize(): self.timer+=self.queued;self.queued=0
        def callback():
            self.calls.append(executor.reference)
            self.timer+=1. if executor.reference else candidate
            for key in self.state: self.state[key]+=1
        self.callback=callback;self.restore=restore;self.synchronize=synchronize
        self.run=lambda:executor.run(callback,self.original,restore,lambda:None,synchronize,lambda _:False)
        return executor

    def test_rollback_cuda_work_is_completed_before_starting_each_timer(self):
        executor=self.setup_executor(candidate=.5);self.run()
        benchmark=executor.state['benchmark']
        self.assertEqual(benchmark['eager_samples_seconds'],[1.,1.])
        self.assertEqual(benchmark['accelerated_samples_seconds'],[.5,.5])
        self.assertEqual(benchmark['gain_fraction'],.5)
        self.assertEqual(self.state,{k:v+1 for k,v in self.original.items()})

    def test_measured_pairs_are_counterbalanced_and_order_is_reported(self):
        executor=self.setup_executor(candidate=.5);self.run()
        self.assertEqual(self.calls,[True,False,True,False,False,True,False])
        self.assertEqual(executor.state['benchmark']['pair_orders'],[['eager','compile_graph'],['compile_graph','eager']])

    def test_equal_speed_is_rejected_without_repeating_until_a_lucky_gain(self):
        self.setup_executor()
        with self.assertRaises(AccelerationRequired): self.run()
        self.assertEqual(len(self.calls),6);self.assertEqual(self.state,self.original)
        self.assertEqual(self.executor.state['reason'],'benchmark_no_gain')

    def test_no_gain_phase_profiles_run_once_each_outside_the_benchmark_and_rollback(self):
        seen=[]
        def diagnose(reference,callback,synchronize):
            self.assertEqual(self.state,self.original)
            seen.append(reference);callback();synchronize()
            self.timer+=1000  # Diagnostic overhead must not enter the benchmark.
            return {'profiled_updates':1,'backend':'eager' if reference else 'compile_graph'}
        executor=self.setup_executor(candidate=2.,diagnose=diagnose)
        with self.assertRaises(AccelerationRequired): self.run()
        self.assertEqual(seen,[True,False]);self.assertEqual(self.state,self.original)
        self.assertEqual(len(self.calls),8)
        b=executor.state['benchmark'];self.assertEqual(b['accelerated_seconds'],2.)
        self.assertEqual(set(b['no_gain_phase_profiles']),{'eager','compile_graph'})
        self.assertEqual(b['training_updates_committed'],0)

    def test_positive_gain_does_not_add_diagnostic_updates(self):
        diagnose=Mock();self.setup_executor(.5,diagnose);self.run();diagnose.assert_not_called()
        self.assertEqual(len(self.calls),7)

    def test_synthetic_recovery_does_not_make_timing_decisions_or_profile(self):
        diagnose=Mock();self.setup_executor(2.,diagnose,synthetic=True);self.run()
        self.assertEqual(len(self.calls),1);diagnose.assert_not_called()

    def test_numeric_rejection_summary_reports_no_gain_without_raw_errors(self):
        report=dict(actual=dict(reason='benchmark_no_gain',eager_agreement='passed_rtol1e-5_atol1e-6_rng_exact',
            benchmark=dict(eager_seconds=8.780320465564728,accelerated_seconds=8.801068268716335)))
        message=calibration_summary(report)
        self.assertIn('数值/RNG已通过',message);self.assertIn('0.236%',message)
        self.assertIn('未启动正式训练',message)
        for candidate in (float('nan'),float('inf'),0,'secret error'):
            report['actual']['benchmark']['accelerated_seconds']=candidate
            self.assertIsNone(calibration_summary(report))


class Tensor:
    def __init__(self,value,device='cuda:0',shape=(1,),dtype='float32'):
        self.value=value;self.device=device;self.shape=shape;self.dtype=dtype
    def copy_(self,other): self.value=other.value;return self


class Optimizer:
    def __init__(self):
        self.state={0:dict(exp_avg=Tensor(9),exp_avg_sq=Tensor(10),step=Tensor(7,'cpu'))}
        self.param_groups=[dict(params=['original-parameter'],lr=.2,weight_decay=.1)]
    def state_dict(self):
        return dict(state=dict(self.state),param_groups=[dict(g,params=[0]) for g in self.param_groups])


class OptimizerRestoreTests(unittest.TestCase):
    def setup_state(self):
        self.optimizer=Optimizer();self.torch=SimpleNamespace(Tensor=Tensor)
        self.saved=copy.deepcopy(self.optimizer.state_dict())
        for value in self.saved['state'][0].values(): value.value=3
        self.saved['param_groups'][0]['lr']=.1

    def test_in_place_restore_preserves_live_storage_devices_and_parameter_identity(self):
        self.setup_state();live=self.optimizer.state[0]['exp_avg'];params=self.optimizer.param_groups[0]['params']
        self.assertTrue(restore_optimizer_in_place(self.torch,self.optimizer,self.saved))
        self.assertIs(self.optimizer.state[0]['exp_avg'],live);self.assertEqual(live.value,3)
        self.assertIs(self.optimizer.param_groups[0]['params'],params)
        self.assertEqual(self.optimizer.param_groups[0]['lr'],.1)
        self.assertEqual(self.optimizer.state[0]['step'].device,'cpu')
        live.value=27;self.assertEqual(self.saved['state'][0]['exp_avg'].value,3)

    def test_cpu_fallback_snapshot_copies_into_existing_gpu_moments(self):
        self.setup_state();self.saved['state'][0]['exp_avg'].device='cpu'
        self.assertTrue(restore_optimizer_in_place(self.torch,self.optimizer,self.saved))
        self.assertEqual(self.optimizer.state[0]['exp_avg'].device,'cuda:0')
        self.assertEqual(self.optimizer.state[0]['exp_avg'].value,3)

    def test_late_metadata_mismatch_is_preflighted_before_any_copy(self):
        for change in ('shape','dtype','missing','parameter_ids'):
            with self.subTest(change=change):
                self.setup_state()
                if change=='shape': self.saved['state'][0]['exp_avg_sq'].shape=(2,)
                elif change=='dtype': self.saved['state'][0]['exp_avg_sq'].dtype='float64'
                elif change=='missing': del self.saved['state'][0]['exp_avg_sq']
                else: self.saved['param_groups'][0]['params']=[1]
                self.assertFalse(restore_optimizer_in_place(self.torch,self.optimizer,self.saved))
                self.assertEqual(self.optimizer.state[0]['exp_avg'].value,9)
                self.assertEqual(self.optimizer.param_groups[0]['lr'],.2)

    def test_lazy_empty_optimizer_uses_original_load_fallback(self):
        self.setup_state();self.saved['state']={}
        self.assertFalse(restore_optimizer_in_place(self.torch,self.optimizer,self.saved))
        self.assertEqual(self.optimizer.state[0]['exp_avg'].value,9)


if __name__=='__main__': unittest.main()
