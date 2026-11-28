"""Owner-side stdlib/fake controls; these are not native Graph speed evidence."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import bf16_graph_diagnostic as d, bf16_graph_native as n
from experiments.baseline_delivery import bf16_graph_runtime as g, bootstrap, return_artifacts
from experiments.baseline_delivery.common import atomic_json, digest, fingerprint, read
from tests.test_baseline_key_bias_formal import fixture
from tests.test_baseline_key_bias_recovery import Runtime


class Graph:
    def __init__(self): self.f = 0; self.b = 0; self.s = 0; self.builds = 1; self.resident = True
    def status(self):
        return dict(graph_resident=self.resident, graph_builds=self.builds,
                    forward_replays=self.f, backward_replays=self.b, replica_sync_rounds=self.s)


class FakeRuntime(Runtime):
    def __init__(self, fail=False):
        super().__init__(fail); self.t.cuda = NS(synchronize=lambda: None)
        self.t.is_tensor = lambda v: False; self.np = NS(ndarray=bytes)
    def update(self, enabled, *, executor=None, micro_batch=16, retain=False):
        assert micro_batch == 16
        if executor is not None:
            assert not enabled
            executor.f += 16; executor.b += 16; executor.s += 8
        result = Runtime.update(self, True)
        if retain: result.update(logits=[1., 2.], gradients={'w': [0., 1.]}, deltas={'w': .01})
        if enabled: result['attention'] = {'different_host_telemetry': True}
        return result


class OperationTests(unittest.TestCase):
    def test_parity_includes_values_and_full_state_then_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = FakeRuntime(); graph = Graph()
            result = n.operation(rt, graph, tmp, 'parity', 'request')
            self.assertTrue(result['parity']['passed']); self.assertEqual(rt.snapshot(), rt.base)
            self.assertEqual(graph.f, 16); self.assertEqual(result['rollback'], d.ROLLBACK)
            self.assertTrue(all(not p._backward_hooks for p in rt.parameters.values()))

    def test_parity_mismatch_retains_compact_path_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = FakeRuntime(); original = rt.update
            def changed(*a, **kw):
                result = original(*a, **kw)
                if kw.get('executor'): result['logits'][0] += .01
                return result
            rt.update = changed
            with self.assertRaises(n.ParityMismatch): n.operation(rt, Graph(), tmp, 'parity', 'r')
            partial = read(Path(tmp)/'partial.json')
            self.assertEqual(partial['parity_failure']['field'], 'update/logits/0')
            self.assertEqual(rt.snapshot(), rt.base)
            self.assertNotIn('tensor_values', partial)

    def test_independent_recovery2_vs1plus1_and_hash_checked_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); results = {}
            for mode in ('continuous', 'interrupt', 'resume'):
                output = folder/mode; output.mkdir(); rt = FakeRuntime()
                results[mode] = n.operation(rt, Graph(), output, mode, 'r')
                self.assertEqual(rt.snapshot(), rt.base)
            self.assertTrue(d.agreement(results)['fresh_process'])
            receipt = results['interrupt']['diagnostic_checkpoint']
            self.assertTrue(receipt['complete_values_reopened_exact'])
            self.assertTrue(d.remove_midpoint(folder, results['interrupt'])['removed'])
            self.assertFalse((folder/'interrupt/diagnostic-state.pt').exists())

    def test_recovery_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/'interrupt').mkdir(); (folder/'resume').mkdir()
            r = n.operation(FakeRuntime(), Graph(), folder/'interrupt', 'interrupt', 'r')
            (folder/'interrupt/diagnostic-state.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError): n.operation(FakeRuntime(), Graph(), folder/'resume', 'resume', 'r')
            with self.assertRaises(ValueError): d.remove_midpoint(folder, r)

    def test_abba_full_updates_and_same_route_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = []; now = 0
            for route in n.ORDER:
                ticks += [now, now+(2 if route == 'incumbent' else 1)]; now += 3
            rt = FakeRuntime(); graph = Graph()
            with patch.object(n.time, 'perf_counter', side_effect=ticks):
                result = n.operation(rt, graph, tmp, 'timing', 'r')
            self.assertTrue(result['passed']); self.assertEqual(result['timing']['throughput_gain'], 1.)
            self.assertEqual([v['route'] for v in result['timing_records']], list(n.ORDER))
            self.assertEqual(rt.snapshot(), rt.base)

    def test_no_gain_is_a_rejection_not_a_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = []; now = 0
            for route in n.ORDER:
                ticks += [now, now+(1 if route == 'incumbent' else 2)]; now += 3
            with patch.object(n.time, 'perf_counter', side_effect=ticks):
                result = n.operation(FakeRuntime(), Graph(), tmp, 'timing', 'r')
            self.assertFalse(result['passed']); self.assertLess(result['timing']['throughput_gain'], 0)

    def test_exception_rolls_back_and_removes_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = FakeRuntime(fail=True)
            with self.assertRaises(RuntimeError): n.operation(rt, Graph(), tmp, 'continuous', 'r')
            self.assertEqual(rt.snapshot(), rt.base)
            self.assertTrue(all(not p._backward_hooks for p in rt.parameters.values()))
            self.assertEqual(read(Path(tmp)/'partial.json')['rollback'], d.ROLLBACK)

    def test_graph_replay_proof_required(self):
        rt = FakeRuntime(); graph = Graph(); graph.builds = 2
        with self.assertRaises(ValueError): n.update(rt, graph)

    def test_recovery_agreement_rejects_changed_gradient_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); results = {}
            for mode in ('continuous', 'interrupt', 'resume'):
                output = root/mode; output.mkdir()
                results[mode] = n.operation(FakeRuntime(), Graph(), output, mode, 'r')
            results['resume']['component_sha256']['gradients'] = 'changed'
            with self.assertRaises(ValueError): d.agreement(results)

    def test_rng_budget_is_per_document_not_two_document_capture(self):
        config = dict(model='transformer', layers=4, heads=4, width=128, dropout=.1)
        result = g.budgets(config, (NS(shape=(16, 4000)),)*2)
        self.assertEqual(result[0]['estimated_increment'], 4292608000)
        self.assertTrue(all(v['passed'] for v in result))
        self.assertFalse(g.graph_rng_budget(config, 16, 4000, documents=2)['passed'])
        for shape in ((32, 4000), (16, 3999)):
            with self.assertRaises(g.BackendFailure): g.budgets(config, (NS(shape=shape),)*2)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.suite, path, r, jobs = fixture(self.root)
        for name in d.strategy.FILES:
            code = self.root/'experiments/baseline_delivery'/name
            code.parent.mkdir(parents=True, exist_ok=True); code.write_bytes(b'original bound code fixture')
        self.cache = self.root/'cache'; self.data = self.root/'data'; self.data.mkdir()
        atomic_json(self.cache/'arrays/train/complete.json', {})
        original_path = self.suite/'migration-source'/d.JOB/'config.json'
        original = read(original_path); original['data_hashes'] = {'train': 'train'}; atomic_json(original_path, original)
        target = d.strategy.target_config(original); target['source_revision'] = 'new'
        atomic_json(self.suite/'jobs'/d.JOB/'config.json', target)
        manifest = read(self.suite/'suite.json'); manifest['data'] = str(self.data)
        for job in manifest['jobs']:
            if job['id'] == d.JOB: job['config_sha256'] = fingerprint(target)
        atomic_json(self.suite/'suite.json', manifest)
        r.update(data=str(self.data), cache=str(self.cache), target_config=target,
                 suite_manifest_sha256=digest(self.suite/'suite.json'))
        atomic_json(path, r); atomic_json(self.suite/'formal-ready.json',
            dict(passed=True, input_sha256=digest(path), selection_sha256=digest(self.suite/'selection/result.json')))
        status = read(self.suite/'status.json'); status[d.JOB]['state'] = 'paused'; atomic_json(self.suite/'status.json', status)
        generation = self.suite/'jobs'/d.JOB/'checkpoints/latest.pt'; generation.parent.mkdir(); generation.write_bytes(b'fullstate')
        atomic_json(generation.parent/'latest.json', dict(schema='qkan-training-checkpoint.v1', file='latest.pt',
            sha256=digest(generation), step=15048, contract='contract'))
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(d.subprocess, 'check_output', side_effect=lambda argv, **kw:
            '' if 'status' in argv else 'new'))
        self.stack.enter_context(patch.object(d.strategy, 'binding', return_value=r['implementation']))
        self.stack.enter_context(patch('experiments.baseline_delivery.key_bias_formal.entry_binding', return_value={}))
        self.stack.enter_context(patch('experiments.baseline_delivery.migration.completed_files', return_value={'files': {}}))

    def test_latest_candidate_not_old_fp32_checkpoint_and_metadata_immutable(self):
        bound = d.bind_source(self.suite, self.cache)
        self.assertEqual(bound['checkpoint']['step'], 15048); d.unchanged(bound)
        self.assertEqual(bound['source_config']['training_strategy'], d.strategy.POLICY)
        (self.suite/'jobs'/d.JOB/'checkpoints/latest.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError): d.unchanged(bound)

    def test_running_source_rejected_without_querying_gpu(self):
        status = read(self.suite/'status.json'); status[d.JOB]['state'] = 'running'; atomic_json(self.suite/'status.json', status)
        with self.assertRaisesRegex(ValueError, 'paused'): d.bind_source(self.suite, self.cache)

    def test_dirty_source_config_checkpoint_cache_and_readiness_reject(self):
        with patch.object(d.subprocess, 'check_output', side_effect=lambda argv, **kw: 'M code' if 'status' in argv else 'new'):
            with self.assertRaises(ValueError): d.bind_source(self.suite, self.cache)
        with self.assertRaises(ValueError): d.bind_source(self.suite, self.root/'other-cache')
        config = self.suite/'jobs'/d.JOB/'config.json'; old = config.read_bytes()
        config.write_text('{}')
        with self.assertRaises(ValueError): d.bind_source(self.suite, self.cache)
        config.write_bytes(old)
        ready = self.suite/'formal-ready.json'; ready.write_text('{}')
        with self.assertRaises(ValueError): d.bind_source(self.suite, self.cache)


class ParentTests(unittest.TestCase):
    def test_real_worker_bootstrap_accepts_supervisor_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root/'worker'; (output/'diagnostics/run').mkdir(parents=True)
            request = root/'input.json'; atomic_json(request, dict(schema=d.SCHEMA, implementation={}, source_revision='new'))
            def native(bound, path, folder, mode): atomic_json(folder/'result.json', {'bootstrap': 'passed'})
            with patch.object(d, 'binding', return_value={}), patch.object(d, 'revision', return_value='new'), \
                 patch.object(d, 'unchanged'), patch.object(n, 'native', side_effect=native), \
                 patch.object(sys, 'argv', ['entry', '--worker', '--request', str(request), '--output', str(output), '--mode', 'parity']):
                self.assertEqual(d.main(), 0)
            self.assertEqual(read(output/'result.json')['bootstrap'], 'passed')

    def test_busy_gpu_and_failed_child_never_read_missing_result(self):
        from experiments.baseline_delivery import runner, attention_diagnostic, diagnostic_watch
        for busy in (True, False):
            with self.subTest(busy=busy), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root/'source'; source.mkdir(); atomic_json(source/'formal-input.json', {})
                output = root/'diagnostic'; output.mkdir(); report = {}
                args = NS(source_suite=source, cache=root/'cache', gpu='0', child_timeout=30, stall_timeout=15)
                def failed(spec, logfile, env):
                    folder = Path(spec['argv'][spec['argv'].index('--output')+1]); folder.mkdir()
                    atomic_json(folder/'partial.json', {'stage': 'capture'})
                    atomic_json(folder/'worker-error.json', {'error_type': 'ControlledFailure'})
                    return 1
                with patch.object(d, 'bind_source', return_value={'checkpoint': {}, 'source_executor_revision': 'new', 'source_metadata_sha256': {}}), \
                     patch.object(d, 'unchanged') as source_check, patch.object(d, 'revision', return_value='new'), \
                     patch.object(runner, 'discover_gpus', return_value=(['0'], {'0': 'fixture'})), \
                     patch.object(attention_diagnostic, 'gpu_preflight', return_value={'idle': not busy}), \
                     patch.object(diagnostic_watch.Supervisor, 'run', side_effect=failed) as child:
                    with self.assertRaises(ValueError if busy else RuntimeError): d.execute(args, output, report)
                    source_check.assert_called_once(); self.assertEqual(child.call_count, 0 if busy else 1)
                self.assertTrue(report['original_checkpoint_and_metadata_unchanged'])
                if not busy: self.assertEqual(report['diagnostics']['parity/partial.json']['stage'], 'capture')

    def test_result_identity_rollback_and_false_admission(self):
        request = {'checkpoint': {'sha256': 'hash'}, 'implementation': {'x': 1}}
        result = dict(schema=d.SCHEMA, mode='parity', passed=True, request_sha256='r', checkpoint_sha256='hash',
                      implementation={'x': 1}, rollback=d.ROLLBACK, **d.CLOSED)
        d.accept(result, request, 'r', 'parity', 0)
        for field, value in [('passed', False), ('candidate_admitted', True), ('training_samples_committed', False),
                             ('implementation', {}), ('rollback', {}), ('request_sha256', 'other')]:
            changed = dict(result); changed[field] = value
            with self.assertRaises(ValueError): d.accept(changed, request, 'r', 'parity', 0)

    def test_actual_supervised_child_target_help_and_cleanup(self):
        from experiments.baseline_delivery.diagnostic_watch import Supervisor
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); spec = d.child_spec(folder/'request.json', folder/'worker', 'parity')
            spec['argv'].append('--help')
            supervisor = Supervisor(30, 15)
            code = supervisor.run(spec, folder/'child.log', dict(__import__('os').environ, PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(code, 0); self.assertTrue(supervisor.records)
            self.assertIn('paused formal source', (folder/'child.log').read_text())

    def test_failure_still_returns_compact_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); seen = []
            def upload(folder, kind, files):
                self.assertEqual(kind, 'bf16-graph'); seen.append(read(folder/files[0]))
            with patch.object(d, 'ROOT', root), patch.object(d, 'revision', return_value='diagnostic'), \
                 patch.object(d, 'binding', return_value={}), patch.object(d, 'execute', side_effect=RuntimeError('controlled')), \
                 patch.object(bootstrap, 'check_exchange'), patch.object(return_artifacts, 'upload_declared', side_effect=upload), \
                 patch.object(sys, 'argv', ['entry', '--source-suite', 'source', '--cache', 'cache']):
                with self.assertRaises(RuntimeError): d.main()
            self.assertEqual(len(seen), 1); self.assertFalse(seen[0]['passed'])
            self.assertFalse(seen[0]['formal_training_started'])

    def test_unified_entry_has_only_one_new_diagnostic(self):
        import runpy
        entry = runpy.run_path(str(d.ROOT/'scripts/experiment.py'))
        self.assertEqual(entry['DIAGNOSTICS']['bf16-graph'], 'bf16_graph_diagnostic')
        with patch.object(subprocess, 'call', return_value=0) as child:
            self.assertEqual(entry['main'](['diagnose', 'bf16-graph', '--', '--help']), 0)
            self.assertIn(d.TARGET, child.call_args.args[0])


if __name__ == '__main__': unittest.main()
