"""Repeated diagnostic controls and real bounded parent-to-child route."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import recovery_replicates as audit
from experiments.baseline_delivery.common import atomic_json, read, revision
from test_baseline_recovery_audit import state


class ReplicateTests(unittest.TestCase):
    def test_rows_scheduler_and_selection_are_exact_controls(self):
        for change in ('row', 'scheduler', 'selection'):
            left = state(); right = copy.deepcopy(left)
            left['update_trace'][0]['rows'] = [0, 1]
            right['update_trace'][0]['rows'] = [0, 1]
            if change == 'row': right['update_trace'][0]['rows'][0] = 2
            if change == 'scheduler': right['scheduler']['lr'] += 1e-12
            if change == 'selection': right['best_epoch'] = 1
            self.assertFalse(audit.comparison(left, right)['control_state_exact'])

    def test_fields_do_not_hide_optimizer_noise(self):
        left = state(); right = copy.deepcopy(left)
        right['optimizer']['moment'][0] += .001
        result = audit.comparison(left, right)
        self.assertTrue(result['control_state_exact'])
        self.assertFalse(result['fields']['optimizer']['exact'])
        self.assertTrue(result['fields']['model']['exact'])

    def fixture(self, folder):
        atomic_json(folder/'config.json', dict(epochs=2, model='mamba1', task='listops',
                                               seed=42, source_revision='producer'))
        atomic_json(folder/'data/synthetic-canary.json', dict(bounded=True, test_included=False))

    def test_real_launch_contract_and_pair_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); self.fixture(folder); output = folder/'audit'; output.mkdir()
            commands = []
            def child(spec, log, env):
                args = spec['argv']; commands.append(args)
                self.assertEqual(args[1:4], ['-B', '-m', 'experiments.baseline_delivery.worker'])
                target = Path(args[args.index('--output')+1])
                atomic_json(target/'checkpoints/latest.json', dict(contract='same-source-and-data', sha256='hash'))
                atomic_json(target/'trace-validation.json', dict(passed=True))
                atomic_json(target/'environment.json', dict(torch='same'))
                return 75 if '--stop-after' in args else 0
            with patch.object(audit, 'check_source', return_value={}), \
                 patch.object(audit, 'run_child', side_effect=child), \
                 patch('experiments.baseline_delivery.worker.load_checkpoint', return_value=state()):
                audit.execute(folder, folder/'cache', 'cpu', 3, output, 'audit')
            self.assertEqual(len(commands), 9)
            self.assertEqual(sum('--resume' in c for c in commands), 3)
            report = read(output/'report.json')
            self.assertEqual(report['summary']['paired_recovery']['comparisons'], 3)
            self.assertEqual(report['summary']['uninterrupted_repeat']['comparisons'], 3)
            self.assertFalse(report['recovery_gate_changed'])
            self.assertIsNone(report['numerical_acceptance_criterion'])
            self.assertNotIn('passed', report)
            self.assertFalse((folder/'canary.json').exists())

    def test_failed_diagnostic_is_returned_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.object(audit, 'revision', return_value='audit'), \
                 patch('experiments.baseline_delivery.bootstrap.check_exchange'), \
                 patch.object(audit, 'execute', side_effect=ValueError('private details')), \
                 patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload, \
                 patch('sys.argv', ['audit', '--canary', str(folder), '--cache', str(folder)]):
                with self.assertRaisesRegex(SystemExit, 'failure report uploaded'): audit.main()
            self.assertEqual(upload.call_args.args[1], 'recovery-replicates')
            report = read(next(folder.glob('recovery-replicates-*/report.json')))
            self.assertEqual(report['status'], 'failed')
            self.assertNotIn('private details', json.dumps(report))

    def test_real_cpu_subprocess_replicates(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); data = folder/'data'; (data/'image').mkdir(parents=True)
            for split, n in (('train', 6), ('val', 2)):
                np.savez(data/'image'/(split+'.npz'), inputs=np.arange(n*1024, dtype=np.uint8).reshape(n, 1024),
                         labels=np.arange(n) % 2)
            atomic_json(data/'synthetic-canary.json', dict(bounded=True, test_included=False))
            config = dict(task='image', model='transformer', width=8, layers=1, heads=2,
                          state_size=4, dropout=0., batch_size=2, micro_batch=1, epochs=2,
                          lr=.001, weight_decay=0., seed=42, classes=2, vocabulary=None,
                          warmup_steps=2, clip_norm=1., protocol_sha256='0'*64,
                          checkpoint_steps=1, source_revision=revision(), data_hashes={})
            atomic_json(folder/'config.json', config)
            output = folder/'audit'; output.mkdir()
            try:
                audit.execute(folder, folder/'cache', 'cpu', 3, output, revision())
            except ValueError as error:
                self.fail(str(error)+'\n'+'\n'.join(log.read_text()[-6000:] for log in output.glob('*.log')))
            report = read(output/'report.json')
            self.assertTrue(all(c['exact'] for c in report['comparisons'].values()))
            self.assertEqual(len(list(output.glob('*.command.json'))), 9)


if __name__ == '__main__':
    unittest.main()
