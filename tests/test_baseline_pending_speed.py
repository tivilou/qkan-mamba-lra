"""All-pending inventory, actual strategy reuse and multi-epoch cap contracts."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from experiments.baseline_delivery import roster_speed as r
from experiments.baseline_delivery.common import atomic_json,read,digest


CONFIG=dict(model='s4d_minimal',task='listops',seed=42,batch_size=32,micro_batch=4,epochs=100,
    adaptive_memory=dict(logical_tiers=[32,64,128]))


class Tests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();self.source=self.root/'origin/runs/formal'
        self.jobs=[dict(id=n) for n in ('mega-text-s42','s5-text-s42','s4d_minimal-image-s42','mamba1-retrieval-s42')]
        atomic_json(self.source/'suite.json',dict(jobs=self.jobs))
        atomic_json(self.source/'mamba-input.json',dict(original_root='pinned-science',source_suite='release'))
        atomic_json(self.source/'status.json',{j['id']:dict(state='pending') for j in self.jobs})
        for job in self.jobs:
            model,task,_=job['id'].split('-');atomic_json(self.source/'jobs'/job['id']/'config.json',{**CONFIG,'model':model,'task':task})

    def test_cap_allows_requested_windows_across_first_epoch_without_budget_change(self):
        before=dict(CONFIG)
        cap=r.final_training_limit(96000,CONFIG)
        self.assertEqual(cap,750*99);self.assertGreater(cap,749*90)
        # At ANY admitted tier, cap is reached no later than99epochs, before test.
        for logical in (32,64,128):self.assertLessEqual(cap,100*((96000+logical-1)//logical)-1)
        self.assertEqual(CONFIG,before)
        meter=r.Windows(CONFIG,warmup=30,seconds=300)
        for step,stamp,epoch,samples in ((1,0,0,32),(6,30,0,192),(400,330,1,96128),
            (800,630,2,192256),(1200,930,3,288384)):
            meter.observe(dict(optimizer_step=step,updated_at=stamp,epoch=epoch,samples_seen=samples))
        self.assertEqual(len(meter.windows),3);self.assertTrue(meter.result(96000,100)['stable'])

    def test_summary_validated_even_when_status_failed_and_retrieval_is_last(self):
        atomic_json(self.source/'jobs/mega-text-s42/summary.json',dict(done=True))
        with patch('experiments.baseline_delivery.migration.completed_files',return_value=dict(files={})) as validate:
            jobs,done=r.select_pending(self.source)
        validate.assert_called_once();self.assertEqual(done,['mega-text-s42'])
        self.assertEqual([j['id'] for j in jobs],['s5-text-s42','s4d_minimal-image-s42','mamba1-retrieval-s42'])
        with patch('experiments.baseline_delivery.migration.completed_files',side_effect=ValueError('bad result')):
            with self.assertRaises(ValueError):r.select_pending(self.source)

    def test_saved_mamba_fused_observation_uses_same_job_not_unfused_probe(self):
        job='mamba1-retrieval-s42';config=read(self.source/'jobs'/job/'config.json')
        folder=self.source/'jobs'/job
        atomic_json(folder/'heartbeat.json',dict(samples_seen=40000,run_samples_start=10000,
            run_started_at=100,updated_at=400))
        atomic_json(folder/'mamba-fused-execution.json',dict(observed_fused_calls=100))
        result=r.observed_source(self.source,job,100000,config)
        self.assertEqual(result['samples_per_second'],100);self.assertFalse(result['stable'])
        self.assertEqual(result['remaining']['seconds'],99600)
        self.assertIn('no new probe',result['provenance'])
        atomic_json(folder/'mamba-fused-execution.json',dict(observed_fused_calls=0))
        self.assertIsNone(r.observed_source(self.source,job,100000,config))

    def test_unknown_checkpoint_progress_does_not_guess_zero_remaining(self):
        job='s5-text-s42';folder=self.source/'jobs'/job
        atomic_json(folder/'checkpoints/latest.json',dict(step=100))
        result=dict(samples_per_second=100)
        value=r.remaining_estimate(result,self.source,job,10000,CONFIG)
        self.assertIsNone(value['remaining']['seconds'])
        report=dict(selected_jobs=[job,'missing-job'],results={job:dict(estimates=dict(full=dict(seconds=100)),**value)})
        total=r.totals(report);self.assertFalse(total['coverage_complete']);self.assertEqual(total['known_remaining_seconds'],0)

    def test_no_window_result_is_reported_and_does_not_break_aggregate_return(self):
        report=dict(all_pending=True,selected_jobs=['failed-job'],results={
            'failed-job':dict(passed=False,stable=False,estimates=None,remaining=dict(seconds=None))})
        total=r.totals(report)
        self.assertFalse(total['coverage_complete']);self.assertEqual(total['estimated_jobs'],0)
        self.assertIn('unknown',r.summary(report))

    def test_all_pending_campaign_probes_each_task_returns_once_preserves_source(self):
        controller=self.root/'controller'
        atomic_json(controller/'docs/evidence/mega-document-v3-build.json',dict(tasks=[
            dict(task=t,splits=dict(train=dict(arrays=dict(labels=dict(shape=[n])))))
            for t,n in (('text',19877),('image',50000),('retrieval',117790))]))
        atomic_json(self.source/'jobs/mega-text-s42/summary.json',dict(done=True))
        args=SimpleNamespace(source_suite=self.source,gpu='0',display='append',warmup_seconds=30,window_seconds=300,timeout=2400)
        before={str(p):digest(p) for p in self.source.rglob('*.json')}
        seen=[]
        def probe(source,suite,model,options,rows,terminal,completed):
            seen.append((options.probe_job,rows,options.window_seconds))
            return dict(passed=True,stable=True,samples_per_second=100,estimates=dict(full=dict(seconds=rows),tenth=dict(seconds=rows/10)))
        def existing(source,name,rows,config):
            if config['model']=='mamba1':return dict(passed=True,estimates=dict(full=dict(seconds=100)),remaining=dict(seconds=50))
        with patch.object(r,'ROOT',controller),patch.object(r,'require_linux'),patch.object(r,'revision',return_value='controller'), \
            patch.object(r,'lock',side_effect=lambda _:nullcontext()),patch.object(r,'probe',side_effect=probe), \
            patch.object(r,'observed_source',side_effect=existing),patch('experiments.baseline_delivery.bootstrap.check_exchange'), \
            patch('scripts.experiment.validate_executor',return_value='mamba-fused'),patch('scripts.formal_progress.Terminal'), \
            patch('experiments.baseline_delivery.migration.completed_files',return_value=dict(files={})), \
            patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'identity'})), \
            patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)), \
            patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload,patch('builtins.print'):
            self.assertEqual(r.campaign_pending(args),0)
        self.assertEqual(seen,[('s5-text-s42',19877,300),('s4d_minimal-image-s42',50000,300)])
        upload.assert_called_once();report=read(upload.call_args.args[0]/'report.json')
        self.assertEqual(report['completed_skipped'],['mega-text-s42']);self.assertTrue(report['totals']['coverage_complete'])
        self.assertFalse(report['original_queue_resumed']);self.assertTrue(report['source_unchanged'])
        self.assertEqual(before,{str(p):digest(p) for p in self.source.rglob('*.json')})


if __name__=='__main__':unittest.main()
