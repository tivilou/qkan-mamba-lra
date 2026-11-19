"""Operator attribution, isolated child handoff and exact rollback using stdlib."""
from contextlib import contextmanager
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,MagicMock,patch

from experiments.baseline_delivery.common import atomic_json,child_spec,read,ROOT,fingerprint
from experiments.baseline_delivery.operator_profile import OperatorObservation,summarize,PREFIX,MAX_EVENTS
from experiments.baseline_delivery.operator_diagnostic import collect_child,observe_once,execute
from experiments.baseline_delivery.performance import Profile
from experiments.baseline_delivery.torch_acceleration import AccelerationRequired
from tests import test_baseline_calibration_timing as timing


def event(name,device=0,parent=None,kind='CPU',host=3):
    return NS(name=name,self_device_time_total=device,self_cpu_time_total=host,
        cpu_parent=parent,device_type='DeviceType.'+kind,is_async=False)


class AttributionTests(unittest.TestCase):
    def test_backward_self_time_is_not_counted_twice_with_kernel_rows_or_ancestors(self):
        phase=event(PREFIX+'backward');node=event('MmBackward0',2000,phase)
        op=event('aten::mm',7000,node)
        result=summarize([phase,node,op,event('raw_kernel',9000,kind='CUDA')])
        backward=result['phases']['backward']
        self.assertEqual(backward['self_cuda_ms'],9)
        self.assertEqual(backward['top_operators'][0]['operator'],'aten::mm')
        self.assertEqual(result['device_rows_excluded'],1)

    def test_top_bound_preserves_omitted_cost_and_unknown_ancestry(self):
        parent=event(PREFIX+'forward_loss')
        events=[event('aten::op'+str(i),1000*(i+1),parent) for i in range(20)]
        events+=[event('aten::copy_',1000)]
        result=summarize(events)
        phase=result['phases']['forward_loss']
        self.assertEqual(len(phase['top_operators']),12)
        self.assertEqual(phase['omitted_self_cuda_ms'],sum(range(1,9)))
        self.assertEqual(result['phases']['unattributed']['self_cuda_ms'],1)

    def test_missing_cuda_nan_negative_and_cpu_only_do_not_fabricate_gpu_cost(self):
        events=[event('aten::x',v) for v in (None,float('nan'),float('inf'),-1)]
        events+=[event('aten::cpu',0)]
        result=summarize(events)
        self.assertEqual(result['status'],'cuda_attribution_unavailable')
        self.assertEqual(result['invalid_cpu_events'],4)
        json.dumps(result,allow_nan=False)

    def test_private_names_paths_and_newlines_are_not_returned(self):
        result=summarize([event('/secret/path.py token=private\nstack',1000)])
        self.assertNotIn('private',json.dumps(result));self.assertNotIn('/secret',json.dumps(result))
        self.assertEqual(result['phases']['unattributed']['top_operators'][0]['operator'],'<other_cpu_event>')

    def test_event_limit_never_returns_a_biased_partial_ranking(self):
        result=summarize(event('aten::x',1) for _ in range(MAX_EVENTS+1))
        self.assertEqual(result['status'],'event_limit_exceeded');self.assertNotIn('phases',result)


class ObservationTests(unittest.TestCase):
    def setup_runtime(self):
        self.profiler=MagicMock();self.profiler.events.return_value=[event('aten::mm',1000)]
        self.api=NS(ProfilerActivity=NS(CPU='cpu',CUDA='cuda'),profile=Mock(return_value=self.profiler),
            record_function=Mock(side_effect=lambda _:__import__('contextlib').nullcontext()))
        self.observer=OperatorObservation(NS(profiler=self.api),fatal=lambda e:'illegal memory access' in str(e))
        return self.observer

    def test_one_callback_finite_summary_and_private_capture_options(self):
        observer=self.setup_runtime();callback=Mock();sync=Mock()
        result=observer.collect(callback,sync)
        callback.assert_called_once();sync.assert_called_once()
        options=self.api.profile.call_args.kwargs
        self.assertEqual(options['activities'],['cpu','cuda'])
        for key in ('record_shapes','profile_memory','with_stack','with_flops','with_modules'):
            self.assertFalse(options[key])
        self.assertEqual(result['status'],'collected');self.assertFalse(observer.active)
        self.profiler.__exit__.assert_called_once_with(None,None,None)

    def test_update_error_is_never_suppressed_or_retried_even_if_profiler_exit_fails(self):
        observer=self.setup_runtime();error=ValueError('scientific failure')
        self.profiler.__exit__.side_effect=RuntimeError('profiler stop')
        callback=Mock(side_effect=error)
        with self.assertRaises(ValueError) as caught: observer.collect(callback,Mock())
        self.assertIs(caught.exception,error);callback.assert_called_once();self.assertFalse(observer.active)

    def test_start_failure_does_not_execute_update_and_stop_failure_does_not_repeat_it(self):
        observer=self.setup_runtime();callback=Mock()
        self.api.profile.side_effect=RuntimeError('secret host context')
        result=observer.collect(callback,Mock());callback.assert_not_called()
        self.assertEqual(result['stage'],'start');self.assertNotIn('secret',json.dumps(result))
        observer=self.setup_runtime();self.profiler.__exit__.side_effect=RuntimeError('stop error')
        result=observer.collect(callback,Mock());callback.assert_called_once()
        self.assertEqual(result['stage'],'collect');self.assertFalse(observer.active)

    def test_fatal_profiler_cuda_error_escapes(self):
        observer=self.setup_runtime();self.profiler.events.side_effect=RuntimeError('illegal memory access')
        with self.assertRaises(RuntimeError): observer.collect(Mock(),Mock())

    def test_phase_ranges_reach_real_profile_context_and_end_on_error(self):
        seen=[]
        @contextmanager
        def scope(name):
            seen.append(('enter',name))
            try: yield
            finally: seen.append(('exit',name))
        profile=Profile(limit=1,scope=scope);profile.begin()
        with self.assertRaises(ValueError):
            with profile.phase('backward'): raise ValueError('update')
        self.assertEqual(seen,[('enter','backward'),('exit','backward')])


class TransactionTests(unittest.TestCase):
    def test_observation_and_instrumentation_failure_both_restore_before_declaration(self):
        for fail in (False,True):
            state={'weight':4,'adam':6,'rng':7,'cursor':9};saved=copy.deepcopy(state);order=[]
            def observe():
                order.append('observe');state['weight']+=1
                if fail: raise ValueError('update error')
                return {'status':'collected'}
            def rollback(): order.append('rollback');state.update(saved)
            call=lambda:observe_once(observe,lambda:order.append('capture'),rollback,lambda:order.append('sync'))
            if fail:
                with self.assertRaises(ValueError): call()
            else: self.assertEqual(call()['status'],'collected')
            self.assertEqual(state,saved);self.assertEqual(order,['capture','sync','observe','rollback'])

    def test_bad_rollback_is_not_reported_as_success(self):
        with self.assertRaises(ValueError):
            observe_once(lambda:{'status':'collected'},Mock(),Mock(side_effect=ValueError('inexact')),Mock())

    def test_no_gain_hook_is_eager_once_and_excluded_from_admission_times(self):
        case=timing.TimingTests();executor=case.setup_executor(candidate=2.)
        def observe(callback,sync):
            self.assertTrue(executor.reference);self.assertEqual(case.state,case.original)
            callback();sync();case.timer+=500
            return {'status':'collected'}
        executor.operators=Mock(side_effect=observe)
        with self.assertRaises(AccelerationRequired): case.run()
        self.assertEqual(case.state,case.original);self.assertEqual(len(case.calls),7)
        benchmark=executor.state['benchmark']
        self.assertEqual(benchmark['accelerated_seconds'],2.)
        self.assertEqual(benchmark['no_gain_operator_profile']['status'],'collected')
        executor.operators.assert_called_once()

    def test_healthy_training_and_synthetic_recovery_never_enable_operator_profiler(self):
        for synthetic in (False,True):
            case=timing.TimingTests();executor=case.setup_executor(candidate=.5,synthetic=synthetic)
            executor.operators=Mock();case.run();executor.operators.assert_not_called()


class ChildTests(unittest.TestCase):
    def test_real_worker_entry_parser_dispatches_operator_mode_without_framework_import(self):
        # Execute the actual parser/main AST; native module imports stay on junior.
        import argparse
        from contextlib import nullcontext
        tree=ast.parse((ROOT/'experiments/baseline_delivery/worker.py').read_text(encoding='utf8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        run=Mock(return_value=0)
        scope=dict(argparse=argparse,Path=Path,read=lambda _: {'model':'transformer'},
            lock=lambda _:nullcontext(),run=run,signal=__import__('signal'),request_pause=lambda *_:None)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'worker-main','exec'),scope)
        args=['worker','--config','config.json','--data','data','--cache','cache','--output','out',
            '--device','cuda:0','--acceleration-probe-source','runs/profile','--operator-profile-only']
        with patch('sys.argv',args),patch('signal.signal'):
            self.assertEqual(scope['main'](),0)
        self.assertTrue(run.call_args.args[-1]);self.assertEqual(run.call_args.args[-2],Path('runs/profile'))

    def test_targeted_parent_preserves_old_config_and_skips_cohort_admission(self):
        from experiments.baseline_delivery.logical_batch import policy as logical_policy
        from experiments.baseline_delivery.performance import policy as performance_policy
        from experiments.baseline_delivery.migration import WARMUP_REVISION
        from experiments.baseline_delivery.torch_acceleration import legacy_policy
        with tempfile.TemporaryDirectory() as temporary:
            suite=Path(temporary)/'suite';data=Path(temporary)/'data';data.mkdir()
            job='transformer-retrieval-s42'
            old=dict(model='transformer',batch_size=32,source_revision=WARMUP_REVISION,
                performance=performance_policy(),adaptive_memory=logical_policy(32),
                torch_acceleration=legacy_policy('compile_graph',2))
            atomic_json(suite/'migration-source'/job/'config.json',old)
            atomic_json(suite/'migration-source/plan.json',{})
            args=NS(migrate_from=Path('original'),cache=Path('cache'),job=job,gpu='0')
            record=dict(checkpoint={'sha256':'sha','step':123})
            with patch('experiments.baseline_delivery.operator_diagnostic.revision',return_value='new'),\
                    patch('experiments.baseline_delivery.readiness.policy_check',return_value=('impl','policy')),\
                    patch('experiments.baseline_delivery.migration.snapshot',return_value=dict(data=str(data),jobs={job:record})),\
                    patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'gpu'})),\
                    patch('experiments.baseline_delivery.operator_diagnostic.collect_child',return_value={}) as launch:
                execute(args,suite)
                config=launch.call_args.args[2]
                self.assertEqual(config['torch_acceleration']['mode'],'eager')
                self.assertEqual(config['adaptive_memory'],old['adaptive_memory'])
                self.assertEqual(read(suite/'migration-source'/job/'config.json'),old)
                self.assertFalse(read(suite/'context.json')['formal_release_allowed'])
                launch.assert_called_once()

    def invoke(self,change=None,code=0):
        with tempfile.TemporaryDirectory() as temporary:
            suite=Path(temporary);config={'model':'transformer','source_revision':'new'}
            def child(spec,log,env):
                self.assertEqual(spec['target'],'experiments.baseline_delivery.worker')
                self.assertEqual(spec['cwd'],str(ROOT));argv=spec['argv']
                self.assertTrue((ROOT/Path(*spec['target'].split('.'))).with_suffix('.py').is_file())
                self.assertIn('--operator-profile-only',argv);self.assertNotIn('--resume',argv)
                self.assertEqual(argv[argv.index('--acceleration-probe-source')+1],str(suite))
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
                report=dict(schema='qkan-operator-profile.v1',mode='eager',config_sha256=fingerprint(config),
                    source_revision='new',source_checkpoint_sha256='sha',restored_step=123,
                    samples_committed=0,optimizer_updates_committed=0,sample_clock_reset=False,
                    rollback_exact=True,maximum_updates=1,status='collected',formal_release_allowed=False)
                report.update(change or {})
                atomic_json(Path(argv[argv.index('--output')+1])/'operator-profile.json',report)
                return code
            return collect_child(suite,'transformer-retrieval-s42',config,Path('data'),Path('cache'),
                '0',{'uuid':'gpu'},dict(checkpoint={'sha256':'sha','step':123}),run=child,build=child_spec)

    def test_actual_parent_worker_contract_binds_original_snapshot_without_resume(self):
        self.assertEqual(self.invoke()['gpu_identity'],{'uuid':'gpu'})

    def test_bad_identity_commit_or_missing_exact_rollback_stops_return(self):
        for change in ({'samples_committed':1},{'rollback_exact':False},{'restored_step':0},
                {'source_checkpoint_sha256':'other'},{'formal_release_allowed':True}):
            with self.subTest(change=change),self.assertRaises(ValueError): self.invoke(change)
        with self.assertRaises(ValueError): self.invoke(code=1)


if __name__=='__main__': unittest.main()
