"""Only scheduler views change; source config/checkpoint/results stay intact."""
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from scripts import experiment as e
from scripts.formal_progress import Dashboard
from experiments.baseline_delivery.common import atomic_json


class Tests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();self.suite=self.root/'runs/original'
        self.jobs=[dict(id=n) for n in ('slow-listops','fast-retrieval','finished')]
        atomic_json(self.suite/'suite.json',dict(jobs=self.jobs))
        atomic_json(self.suite/'status.json',{'slow-listops':dict(state='pending'),'fast-retrieval':dict(state='paused'),'finished':dict(state='failed')})
        atomic_json(self.suite/'jobs/finished/summary.json',dict(computed=True))
        atomic_json(self.suite/'jobs/finished/config.json',dict(original=True))
        self.record=dict(schema='qkan-task-order.v1',policy='fastest-first',manifest_sha256='bound',estimates_sha256='verified',
            remaining_seconds={'slow-listops':1000,'fast-retrieval':100})
        atomic_json(self.suite/'task-order.json',self.record)

    def test_short_retrieval_precedes_long_other_task_and_completed_is_not_relaunched(self):
        before={str(p):p.read_bytes() for p in self.suite.rglob('*.json')}
        with patch.object(e,'queue_record',return_value=self.record), \
                patch('experiments.baseline_delivery.migration.completed_files',return_value={}) as validate:
            view=e.queue_read(self.suite,e.read)
            self.assertEqual([j['id'] for j in view(self.suite/'suite.json')['jobs']],['finished','fast-retrieval','slow-listops'])
            self.assertEqual(view(self.suite/'status.json')['finished']['state'],'computed')
            self.assertEqual(view(self.suite/'status.json')['fast-retrieval']['state'],'paused')
        validate.assert_called_once();self.assertEqual(before,{str(p):p.read_bytes() for p in self.suite.rglob('*.json')})

    def test_bad_completed_result_and_missing_time_stop_instead_of_rerun(self):
        with patch.object(e,'queue_record',return_value=self.record), \
                patch('experiments.baseline_delivery.migration.completed_files',side_effect=ValueError('invalid')):
            with self.assertRaises(ValueError):e.queue_read(self.suite,e.read)
        (self.suite/'jobs/finished/summary.json').unlink()
        with patch.object(e,'queue_record',return_value=self.record):
            with self.assertRaises(ValueError):e.queue_read(self.suite,e.read)

    def test_resumes_keep_recorded_policy_and_dashboard_matches(self):
        atomic_json(self.suite/'mamba-input.json',{})
        with patch.object(e,'resume_command',return_value=['dispatcher']),patch.object(e,'display_module'), \
                patch.object(e,'supervise',return_value=0) as run:
            self.assertEqual(e.main(['--root',str(self.root),'resume','--suite',str(self.suite)]),0)
        self.assertEqual(run.call_args.kwargs['task_order'],'fastest-first')
        with patch('scripts.formal_progress.subprocess.run',side_effect=OSError('No GPU')):
            frame=Dashboard(self.root,self.suite).render()
        self.assertIn('从短到长',frame)
        order=next(line for line in frame.splitlines() if line.startswith('续跑次序：'))
        self.assertLess(order.index('fast-retrieval'),order.index('slow-listops'))

    def test_review_covers_every_estimated_job_and_sort_order_is_known(self):
        record=e.read(Path(e.__file__).resolve().parents[1]/'docs/baselines/runtime-order-20261128.json')
        ordered=sorted(record['jobs'],key=lambda n:record['jobs'][n]['remaining_seconds'])
        self.assertEqual(len(ordered),14)
        self.assertEqual(ordered[0],'s4d_minimal-image-s42')
        self.assertEqual(ordered[-1],'mega-retrieval-s42')
        self.assertLess(ordered.index('s4d_minimal-retrieval-s42'),ordered.index('s5-listops-s42'))

    def test_estimate_record_binds_manifest_and_each_scientific_config(self):
        code=self.root/'controller/scripts/experiment.py';path=code.parent.parent/'docs/baselines/runtime-order-20261128.json'
        file=self.suite/'jobs/slow-listops/config.json';atomic_json(file,dict(untouched=True))
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        evidence=dict(manifest_sha256=digest(self.suite/'suite.json'),jobs={
            'slow-listops':dict(remaining_seconds=1000,config_sha256=digest(file))})
        atomic_json(path,evidence)
        with patch.object(e,'__file__',str(code)):
            record=e.queue_record(self.suite,'fastest-first')
            self.assertEqual(record['estimates_sha256'],digest(path))
            atomic_json(file,dict(changed=True))
            with self.assertRaises(ValueError):e.queue_record(self.suite,'fastest-first')


if __name__=='__main__':unittest.main()
