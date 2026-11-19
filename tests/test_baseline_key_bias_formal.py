"""Formal control path and real no-data child launch; no native imports/models."""
import copy
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import key_bias_formal as f
from experiments.baseline_delivery.common import atomic_json, digest, fingerprint, read

REAL_ROOT = Path(__file__).resolve().parents[1]


def old_config(model='transformer', task='retrieval'):
    return dict(model=model, task=task, source_revision=f.PARALLEL_REVISION,
        layers=4, width=128, heads=4, dropout=.1,
        adaptive_memory=dict(execution=dict(precision='fp32')),
        torch_acceleration=dict(mode='compile_graph'))


def fixture(root):
    suite = root/'runs/key-bias-formal-fixture'; suite.mkdir(parents=True)
    source = root/'old/runs/source'; source.mkdir(parents=True)
    checkpoint = source/'source.pt'; checkpoint.write_bytes(b'original full-state fixture')
    cp = dict(path=str(checkpoint), sha256=digest(checkpoint), step=5)
    target = f.strategy.target_config(old_config()); target['source_revision'] = 'new'
    jobs = []; records = {}
    for m in f.MODELS:
        for t in f.TASKS:
            name = m+'-'+t+'-s42'; config = target if name == f.JOB else old_config(m, t)
            atomic_json(suite/'jobs'/name/'config.json', config)
            jobs.append(dict(id=name, config_sha256=fingerprint(config)))
            records[name] = dict(checkpoint=None)
    atomic_json(suite/'migration-source'/f.JOB/'config.json', old_config())
    atomic_json(suite/'migration-source/plan.json', dict(jobs=records))
    atomic_json(suite/'suite.json', dict(jobs=jobs, models=list(f.MODELS), tasks=list(f.TASKS), seeds=[42], data='data'))
    atomic_json(suite/'status.json', {j['id']: dict(state='pending', started_training=False) for j in jobs})
    atomic_json(suite/'key-bias-worker-migration/worker-input.json', dict(fixture=True))
    r = dict(schema=f.SCHEMA, source_revision='new', entry_binding={}, implementation={'native': 'reviewed'},
        checkpoint=cp, target_config=target, source_suite=str(source), original_root=str(root/'old'),
        worker_from='reviewed-folder', data='data', cache='cache', original_metadata_sha256={},
        suite_manifest_sha256=digest(suite/'suite.json'), snapshot_sha256=digest(suite/'migration-source/plan.json'),
        migration_input_sha256=digest(suite/'key-bias-worker-migration/worker-input.json'))
    path = suite/'formal-input.json'; atomic_json(path, r)
    atomic_json(suite/'selection/result.json', dict(passed=True))
    atomic_json(suite/'formal-ready.json', dict(passed=True, input_sha256=digest(path), selection_sha256=digest(suite/'selection/result.json')))
    return suite, path, r, jobs


class SelectionTests(unittest.TestCase):
    def state(self):
        return dict(best_accuracy=.8, best_epoch=7, best_weights={'w': [1, 2]},
            migration={'previous': 'retained'}, model={'w': [3, 4]}, optimizer={'moments': [5, 6], 'step': 123},
            rng={'state': [9]}, strategy_gradients={'w': [7, 8]}, sample_clock={'progress': 456},
            history=[dict(epoch=7, validation={'accuracy': .8})], step=123, cursor=24)

    def test_recertifies_same_best_and_preserves_all_training_fields(self):
        old = self.state(); original = copy.deepcopy(old); seen = []
        def evaluate(weights):
            seen.append(copy.deepcopy(weights)); return dict(loss=.4, accuracy=.75, samples=1804)
        new, record = f.recertify_selection(old, evaluate, fingerprint, 'val-hash', 1804)
        self.assertEqual(old, original); self.assertEqual(seen, [old['best_weights']])
        self.assertEqual(new['best_accuracy'], .75); self.assertEqual(new['best_epoch'], 7)
        for key in old:
            if key not in ('best_accuracy', 'migration'): self.assertEqual(new[key], old[key], key)
        self.assertEqual(record['source_best_accuracy'], .8)
        self.assertFalse(record['test_access']); self.assertEqual(record['training_updates'], 0)
        self.assertEqual(new['migration']['previous'], 'retained')

    def test_missing_best_does_not_invent_a_model_or_read_validation(self):
        state = self.state(); state.update(best_weights=None, best_accuracy=-1, best_epoch=None)
        with patch('builtins.print'):
            new, record = f.recertify_selection(state, lambda _: self.fail('No selected weights'), fingerprint, 'v', 2)
        self.assertEqual(new['best_accuracy'], -1); self.assertIsNone(record['candidate_validation'])

    def test_invalid_validation_never_mutates_original(self):
        for metric in (dict(loss=float('nan'), accuracy=.5, samples=2),
                       dict(loss=.2, accuracy=1.1, samples=2), dict(loss=.2, accuracy=.5, samples=1)):
            state = self.state(); old = copy.deepcopy(state)
            with self.assertRaises(ValueError): f.recertify_selection(state, lambda _: metric, fingerprint, 'v', 2)
            self.assertEqual(state, old)

    def test_no_weights_with_existing_score_is_rejected(self):
        state = self.state(); state['best_weights'] = None
        with self.assertRaises(ValueError): f.recertify_selection(state, lambda _: None, fingerprint, 'v', 2)


class FormalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.suite, self.path, self.r, self.jobs = fixture(self.root)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for obj, name, value in ((f, 'ROOT', self.root), (f, 'entry_binding', lambda: {}),
                                  (f, 'revision', lambda: 'new'), (f.strategy, 'binding', lambda: self.r['implementation'])):
            self.stack.enter_context(patch.object(obj, name, value))

    def install_pointer(self, name=f.JOB):
        folder = self.suite/'jobs'/name/'checkpoints'; folder.mkdir()
        generation = folder/'step-5-state.pt'; generation.write_bytes(b'full fixture')
        atomic_json(folder/'latest.json', dict(schema='qkan-training-checkpoint.v1', file=generation.name,
            sha256=digest(generation), step=5, contract='contract', previous=[]))

    def test_formal_capability_is_exact_and_ordinary_finite_gate_stays_closed(self):
        gate = f.FormalAdmission(self.path); output = self.suite/'jobs'/f.JOB
        gate.validate(self.r['target_config'], output, 'cuda:0', True, None, False, False, None)
        self.assertFalse(gate.trace); self.assertFalse(gate.check_only)
        for i, value in ((1, output.parent/'other'), (2, 'cpu'), (3, False), (4, 7), (5, True), (6, True), (7, 'probe')):
            args = [self.r['target_config'], output, 'cuda:0', True, None, False, False, None]; args[i] = value
            with self.subTest(i=i), self.assertRaises(ValueError): gate.validate(*args)
        finite = f.strategy.Admission(dict(target_config=self.r['target_config'], checkpoint=self.r['checkpoint'],
            implementation=self.r['implementation']), self.path, output, 'continuous')
        with self.assertRaises(ValueError): finite.validate(self.r['target_config'], output, 'cuda:0', True, None, False, False, None)

    def test_manifest_config_and_selection_receipt_tampering_reject(self):
        for target in (self.suite/'jobs'/f.JOB/'config.json', self.suite/'selection/result.json', self.suite/'suite.json'):
            original = target.read_bytes(); atomic_json(target, {'changed': True})
            try:
                with self.assertRaises(ValueError): f.verify_input(self.path, ready=True)
            finally: target.write_bytes(original)

    def test_missing_started_checkpoint_and_corrupt_generation_reject(self):
        other = self.jobs[-1]
        f.checkpoint_required(self.suite, other, {'started_training': False})
        with self.assertRaises(ValueError): f.checkpoint_required(self.suite, other, {'started_training': True})
        with self.assertRaises(ValueError): f.checkpoint_required(self.suite, {'id': f.JOB}, {})
        self.install_pointer(); f.checkpoint_required(self.suite, {'id': f.JOB}, {})
        (self.suite/'jobs'/f.JOB/'checkpoints/step-5-state.pt').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): f.checkpoint_required(self.suite, {'id': f.JOB}, {})

    def test_dispatch_uses_original_cwd_and_isolated_s5_mega_interpreters(self):
        for model in ('transformer', 'mamba1', 'mega', 's5', 's4d_minimal'):
            job = dict(id=model+'-image-s42'); spec = f.job_spec(self.suite, self.r, job)
            self.assertEqual(spec['cwd'], self.r['original_root'])
            self.assertEqual(spec['argv'][spec['argv'].index('-m')+1], spec['target'])
            if model in ('mega', 's5'): self.assertIn('.venv-'+model, spec['argv'][0])
            self.assertEqual(spec['target'].endswith('s5_worker'), model == 's5')
        spec = f.job_spec(self.suite, self.r, {'id': f.JOB})
        self.assertEqual(spec['cwd'], str(self.root)); self.assertEqual(spec['target'], f.TARGET)
        self.assertIn('--worker', spec['argv'])

    def test_changed_legacy_policy_is_rejected(self):
        name = 'transformer-image-s42'; path = self.suite/'jobs'/name/'config.json'
        config = read(path); config['training_strategy'] = f.strategy.POLICY; atomic_json(path, config)
        with self.assertRaises(ValueError): f.job_spec(self.suite, self.r, {'id': name})

    def test_real_parent_launches_exact_new_module_help_and_records_completion(self):
        self.install_pointer(); job = {'id': f.JOB}; statuses = read(self.suite/'status.json')
        spec = f.job_spec(self.suite, self.r, job)
        # Same module/interpreter as the real dispatch; no-data startup replaces only worker arguments.
        spec.update(cwd=str(REAL_ROOT), argv=spec['argv'][:5]+['--help'])
        with patch.object(f, 'job_spec', return_value=spec), patch.object(f, 'validate_job_result', return_value={}), \
                patch('scripts.formal_progress.Terminal') as terminal, patch('builtins.print'):
            self.assertEqual(f.run_job(self.suite, self.r, job, '0', statuses), 0)
        frame = terminal.return_value.show.call_args.args[0]
        self.assertIn('任务 25', frame); self.assertNotIn('"pending": [', frame)
        terminal.return_value.close.assert_called_once()
        self.assertIsInstance(statuses[f.JOB]['pid'], int)
        self.assertGreater(statuses[f.JOB]['started'], 0)
        self.assertEqual(read(self.suite/'heartbeat.json')['job'], f.JOB)
        self.assertEqual(read(self.suite/'status.json')[f.JOB]['state'], 'computed')
        self.assertIn('Explicit mixed-executor', (self.suite/'jobs'/f.JOB/'worker.log').read_text())
        self.assertTrue(list((self.suite/'jobs'/f.JOB).glob('formal-child-*.json')))

    def test_paused_child_does_not_become_computed(self):
        self.install_pointer(); statuses = read(self.suite/'status.json')
        spec = dict(target='stdlib.control', cwd=str(REAL_ROOT), argv=[sys.executable, '-B', '-c', 'raise SystemExit(75)'])
        with patch.object(f, 'job_spec', return_value=spec), patch.object(f, 'completed_files') as complete, patch('builtins.print'):
            self.assertEqual(f.run_job(self.suite, self.r, {'id': f.JOB}, '0', statuses), 75)
        complete.assert_not_called(); self.assertEqual(statuses[f.JOB]['state'], 'paused')

    def execution_mocks(self, busy=False):
        self.stack.enter_context(patch.object(f, 'original_executor'))
        self.stack.enter_context(patch.object(f, 'reviewed_worker', return_value={'gpu_identity': 'gpu'}))
        self.stack.enter_context(patch('experiments.baseline_delivery.runner.discover_gpus', return_value=(['0'], {'0': 'gpu'})))
        self.stack.enter_context(patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight', return_value={'idle': not busy}))

    def args(self, **kw):
        return SimpleNamespace(cache=None, gpu='0', upload_only=False, prepare_only=False, **kw)

    def test_ready_return_failure_prevents_first_long_update(self):
        self.execution_mocks()
        with patch.object(f, 'publish_status', side_effect=OSError('offline')), patch.object(f, 'run_job') as run:
            with self.assertRaises(OSError): f.execute(self.args(), self.suite)
        run.assert_not_called()

    def test_busy_gpu_prevents_preparation_and_training(self):
        self.execution_mocks(busy=True)
        with patch.object(f, 'zero_preparation') as zero, patch.object(f, 'run_job') as run:
            with self.assertRaises(ValueError): f.execute(self.args(), self.suite)
        zero.assert_not_called(); run.assert_not_called()

    def test_queue_skips_completed_preserves_config_and_stops_on_pause(self):
        self.execution_mocks()
        first = self.jobs[0]['id']; atomic_json(self.suite/'jobs'/first/'summary.json', {'original': True})
        old = (self.suite/'jobs'/first/'config.json').read_bytes()
        with patch.object(f, 'validate_job_result', return_value={}), patch.object(f, 'publish_status'), \
                patch.object(f, 'run_job', return_value=75) as run:
            self.assertEqual(f.execute(self.args(), self.suite), 75)
        self.assertEqual(run.call_count, 1); self.assertEqual(run.call_args.args[2]['id'], self.jobs[1]['id'])
        self.assertEqual((self.suite/'jobs'/first/'config.json').read_bytes(), old)

    def test_upload_only_avoids_gpu_and_source_execution(self):
        args = self.args(); args.upload_only = True
        event = self.suite/'events/failed-123'; event.mkdir(parents=True)
        atomic_json(event/'artifact-delivery.json', {'pending': True})
        with patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload, \
                patch.object(f, 'original_executor') as source, patch.object(f, 'run_job') as run:
            self.assertEqual(f.execute(args, self.suite), 0)
        upload.assert_called_once(); source.assert_not_called(); run.assert_not_called()

    def test_compact_failure_retains_native_locations_without_raw_logs(self):
        atomic_json(self.suite/'selection/worker-error.json', dict(error_type='ValueError', project_frames=[{'file': 'module.py', 'line': 42}]))
        (self.suite/'selection/private.log').write_text('private input and token, never return')
        with patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload:
            f.publish_status(self.suite, 'failed')
        folder, experiment, files = upload.call_args.args
        self.assertEqual(experiment, 'key-bias-formal'); self.assertEqual(files, ['formal-status.json'])
        value = read(folder/files[0]); self.assertIn('selection', value['worker_failure_locations'])
        self.assertNotIn('private input', (folder/files[0]).read_text())
        self.assertFalse(value['formal_training_started'])

    def test_upload_only_before_first_event_does_not_start_preparation(self):
        args = self.args(); args.upload_only = True
        with patch.object(f, 'original_executor') as source, patch.object(f, 'zero_preparation') as zero:
            self.assertEqual(f.execute(args, self.suite), 0)
        source.assert_not_called(); zero.assert_not_called()

    def test_incomplete_zero_preparation_never_publishes_formal_readiness(self):
        (self.suite/'formal-ready.json').unlink()
        (self.suite/'selection/result.json').unlink()
        with patch('experiments.baseline_delivery.diagnostic_watch.Supervisor.run', return_value=1):
            with self.assertRaises(ValueError): f.zero_preparation(self.suite, self.r, '0')
        self.assertFalse((self.suite/'formal-ready.json').exists())
        self.assertTrue((self.suite/'preparation-children.json').exists())

    def test_candidate_completion_requires_strategy_binding(self):
        folder = self.suite/'jobs'/f.JOB
        atomic_json(folder/'summary.json', dict(training_strategy='unbound'))
        with patch.object(f, 'completed_files', return_value={}):
            with self.assertRaises(ValueError): f.validate_job_result(self.suite, self.r, {'id': f.JOB})
            atomic_json(folder/'summary.json', dict(training_strategy=f.strategy.envelope(
                self.r['target_config'], self.r['checkpoint']['sha256'], self.r['implementation'])))
            f.validate_job_result(self.suite, self.r, {'id': f.JOB})

    def test_preparation_only_publishes_ready_and_launches_no_long_job(self):
        self.execution_mocks(); args = self.args(); args.prepare_only = True
        with patch.object(f, 'publish_status') as publish, patch.object(f, 'run_job') as run:
            self.assertEqual(f.execute(args, self.suite), 0)
        publish.assert_called_once_with(self.suite, 'ready'); run.assert_not_called()

    def test_reviewed_fixture_rejects_report_hash_and_code_changes(self):
        from experiments.baseline_delivery.key_bias_worker_admission import agreement
        first = dict(step=6, samples=128, micro_batch=16); second = dict(step=7, samples=128, micro_batch=16)
        a = dict(scientific_state_sha256='final', component_sha256={'rng': 'same'}, updates=[first, second], loaded_state_sha256='base')
        b = dict(scientific_state_sha256='mid', updates=[first], loaded_state_sha256='base')
        c = dict(**{k: v for k, v in a.items() if k != 'loaded_state_sha256'}, loaded_state_sha256='mid', trace=dict(validation=dict(passed=True)))
        results = dict(migration=dict(scientific_state_sha256='base'), continuous=a, interrupt=b, resume=c)
        folder = self.root/'reviewed'; folder.mkdir()
        path = folder/'key-bias-worker-admission.json'
        report = dict(passed=True, candidate_admitted=False, formal_training_started=False,
            original_checkpoint_and_metadata_unchanged=True, implementation=self.r['implementation'],
            results=results, recovery=agreement(results))
        atomic_json(path, report)
        review = self.root/'review.json'; atomic_json(review, dict(native_passed=True,
            source_report=dict(sha256=digest(path), bytes=path.stat().st_size)))
        with patch.object(f, 'REVIEW', review):
            self.assertEqual(f.reviewed_worker(folder), report)
            with patch.object(f.strategy, 'binding', return_value={'changed': True}):
                with self.assertRaises(ValueError): f.reviewed_worker(folder)
            path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaises(ValueError): f.reviewed_worker(folder)


if __name__ == '__main__': unittest.main()
