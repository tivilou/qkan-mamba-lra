"""Runtime-probe boundaries; stdlib/fakes only, never import model frameworks."""
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery import roster_speed as r
from experiments.baseline_delivery.common import atomic_json,read,digest
from scripts import experiment as entry


CONFIG=dict(model='s5',task='listops',batch_size=32,micro_batch=4,epochs=100,
    adaptive_memory=dict(logical_tiers=[32,64,128]))


def heartbeat(step,at,seen=None,micro=4,logical=32,oom=0):
    return dict(optimizer_step=step,updated_at=at,samples_seen=seen if seen is not None else step*32,
        epoch=0,adaptive_memory=dict(used_micro_batch=micro,used_logical_batch=logical,oom_retries=oom))


class Tests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();self.source=self.root/'original/runs/formal'
        self.suite=self.root/'controller/runs/probe'
        for model in r.MODELS:atomic_json(self.source/'jobs'/(model+'-listops-s42')/'config.json',{**CONFIG,'model':model})
        atomic_json(self.source/'suite.json',dict(jobs=[]))
        atomic_json(self.source/'status.json',dict(mega=dict(state='paused')))
        atomic_json(self.source/'mamba-input.json',dict(original_root='pinned-science',source_suite='old-release'))
        self.args=SimpleNamespace(gpu='0',warmup_seconds=30,window_seconds=60,timeout=1200,display='append',source_suite=self.source)

    def test_three_windows_use_worker_samples_and_do_not_count_startup(self):
        meter=r.Windows(CONFIG)
        for step,at in ((1,100),(6,130),(66,190),(126,250),(186,310)):
            meter.observe(heartbeat(step,at))
            meter.observe(heartbeat(step,at))  # repeated display reads are not samples
        result=meter.result(96000,100)
        self.assertTrue(result['stable']);self.assertEqual(result['samples_per_second'],32)
        self.assertEqual(result['estimates']['full']['seconds'],300000)
        self.assertEqual(result['estimates']['tenth']['seconds'],30000)
        self.assertEqual(sum(w['samples'] for w in result['windows']),180*32)

    def test_tier_or_oom_change_restarts_measurement_and_short_run_is_provisional(self):
        meter=r.Windows(CONFIG)
        for step,at in ((1,0),(6,30),(66,90)):meter.observe(heartbeat(step,at))
        self.assertEqual(len(meter.windows),1)
        meter.observe(heartbeat(67,95,micro=8));self.assertEqual(meter.windows,[])
        meter.observe(heartbeat(72,125,micro=8));meter.observe(heartbeat(92,145,micro=8))
        short=meter.result(96000,100)
        self.assertFalse(short['stable']);self.assertEqual(short['samples_per_second'],32)
        self.assertTrue(short['windows'][0]['partial'])
        meter.observe(heartbeat(93,150,micro=8,oom=1));self.assertEqual(meter.resets,2)
        self.assertIsNone(meter.result(96000,100)['estimates'])

    def test_faster_and_slower_windows_are_range_not_confidence_interval(self):
        meter=r.Windows(CONFIG)
        for step,at in ((1,0),(6,30),(66,90),(186,150),(246,210)):meter.observe(heartbeat(step,at))
        value=meter.result(96000,100)
        self.assertFalse(value['stable']);self.assertGreater(value['relative_window_spread'],.15)
        self.assertIn('not a confidence interval',value['estimate_scope'])

    def test_original_spec_s5_environment_identity_and_stop_before_validation(self):
        old=self.source.parent.parent/'.venv-s5/bin/python';new=self.root/'controller/.venv-s5/bin/python'
        old.parent.mkdir(parents=True);old.write_bytes(b'interpreter')
        new.parent.mkdir(parents=True);os.link(old,new)
        spec=dict(target='experiments.baseline_delivery.s5_worker',cwd='pinned-science',argv=[str(old),'-B','-m','experiments.baseline_delivery.s5_worker','--output','owned'])
        with patch.object(r,'ROOT',self.root/'controller'),patch.object(r.subprocess,'check_output',return_value=json.dumps(spec)) as call:
            value=r.make_spec(self.source,self.suite,'s5-listops-s42',749)
        self.assertEqual(value['argv'][0],str(new));self.assertEqual(value['argv'][-2:],['--stop-after','749'])
        self.assertEqual(value['cwd'],'pinned-science')
        self.assertEqual(call.call_args.kwargs['env']['PYTHONPATH'],str(self.source.parent.parent))
        self.assertNotIn('--synthetic',value['argv']);self.assertLess(749*128,96000)
        with patch.object(r.subprocess,'check_output',return_value=json.dumps({**spec,'argv':spec['argv']+['--resume']})):
            with self.assertRaises(ValueError):r.make_spec(self.source,self.suite,'s5-listops-s42',749)

    def test_probe_uses_owned_output_safe_pause_and_deletes_only_disposable_checkpoint(self):
        child=Mock(pid=123);child.poll.side_effect=[None,None,None,None,None,None,75];child.returncode=75
        series=iter([heartbeat(1,100),heartbeat(6,130),heartbeat(66,190),heartbeat(126,250),heartbeat(186,310),heartbeat(186,310)])
        def launch(spec,log,env):
            out=self.suite/'jobs/s5-listops-s42';atomic_json(out/'checkpoints/latest.json',dict(step=186))
            self.assertEqual(env['PYTHONPATH'],'pinned-science');self.assertIn('old-release',env['QKAN_RUNTIME_RELEASE'])
            self.assertEqual(env['QKAN_MEGA_SOURCE'],str(self.source.parent.parent/'.bootstrap/strong-upstream/mega'))
            return child
        spec=dict(argv=['python','-m','old-worker'],cwd='pinned-science',target='old-worker')
        with patch.object(r,'make_spec',return_value=spec),patch.object(r,'spawn',side_effect=launch), \
                patch.object(r,'peek',side_effect=lambda _:next(series)),patch.object(r,'board',return_value=dict(board_used_mib=1024)), \
                patch.object(r.time,'sleep'),patch.object(r,'stop_child') as stop, \
                patch('experiments.baseline_delivery.diagnostic_watch.terminate',return_value=dict(group_verified=True)):
            value=r.probe(self.source,self.suite,'s5',self.args,96000,Mock(),{})
        stop.assert_called_once();self.assertTrue(value['passed']);self.assertTrue(value['stable'])
        self.assertFalse((self.suite/'jobs/s5-listops-s42/checkpoints/latest.json').exists())
        self.assertEqual(read(self.source/'status.json'),dict(mega=dict(state='paused')))

    def test_two_models_one_return_and_original_files_remain_identical(self):
        root=self.root/'controller';root.mkdir(exist_ok=True)
        atomic_json(root/'docs/evidence/mega-document-v3-build.json',dict(tasks=[dict(task='listops',splits=dict(train=dict(arrays=dict(labels=dict(shape=[96000])))))]))
        before={str(p):digest(p) for p in self.source.rglob('*.json')}
        def probe(source,suite,model,args,rows,terminal,completed):
            self.assertEqual(source,self.source);self.assertEqual(rows,96000)
            if model=='s5':raise RuntimeError('private worker failure')
            return dict(passed=True,stable=False,estimates=None)
        with patch.object(r,'ROOT',root),patch.object(r,'revision',return_value='current'),patch.object(r,'require_linux'), \
                patch.object(r,'lock',side_effect=lambda _:nullcontext()),patch.object(r,'probe',side_effect=probe) as run, \
                patch('experiments.baseline_delivery.bootstrap.check_exchange'), \
                patch('scripts.experiment.validate_executor',return_value='mamba-fused'), \
                patch('scripts.formal_progress.Terminal'),patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'identity'})), \
                patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)), \
                patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload,patch('builtins.print'):
            self.assertEqual(r.campaign(self.args),2)
        self.assertEqual(run.call_count,2);upload.assert_called_once()
        output=upload.call_args.args[0];report=read(output/'report.json')
        self.assertTrue(report['source_unchanged']);self.assertFalse(report['original_queue_resumed'])
        self.assertIn('s5',report['failures']);self.assertIn('s4d_minimal',report['results'])
        self.assertEqual({str(p):digest(p) for p in self.source.rglob('*.json')},before)
        self.assertNotIn('private worker failure',(output/'report.json').read_text())

    def test_unified_entry_routes_existing_diagnostic_without_native_import(self):
        with patch.object(entry.subprocess,'call',return_value=0) as call:
            self.assertEqual(entry.main(['diagnose','roster-speed','--','--source-suite',str(self.source)]),0)
        self.assertIn('experiments.baseline_delivery.roster_speed',call.call_args.args[0])

    def test_unverified_owned_group_blocks_second_model_even_if_gpu_appears_idle(self):
        root=self.root/'controller';root.mkdir(exist_ok=True)
        atomic_json(root/'docs/evidence/mega-document-v3-build.json',dict(tasks=[dict(task='listops',splits=dict(train=dict(arrays=dict(labels=dict(shape=[96000])))))]))
        def failed(source,suite,model,*args):
            out=suite/'jobs'/(model+'-listops-s42')
            atomic_json(out/'child-start.json',dict(pid=123))
            atomic_json(out/'child-execution.json',dict(cleanup=dict(group_verified=False)))
            raise RuntimeError('Unverified cleanup')
        with patch.object(r,'ROOT',root),patch.object(r,'revision',return_value='current'),patch.object(r,'require_linux'), \
                patch.object(r,'lock',side_effect=lambda _:nullcontext()),patch.object(r,'probe',side_effect=failed) as run, \
                patch('experiments.baseline_delivery.bootstrap.check_exchange'), \
                patch('scripts.experiment.validate_executor',return_value='mamba-fused'),patch('scripts.formal_progress.Terminal'), \
                patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'identity'})), \
                patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)), \
                patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload,patch('builtins.print'):
            self.assertEqual(r.campaign(self.args),2)
        run.assert_called_once();upload.assert_called_once()
        self.assertTrue(read(upload.call_args.args[0]/'report.json')['unsafe_remaining_processes'])


if __name__=='__main__':unittest.main()
