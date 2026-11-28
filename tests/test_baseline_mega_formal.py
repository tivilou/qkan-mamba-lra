"""Affected amendment boundaries only: stdlib/fakes, no framework or GPU."""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery import mega_formal as m
from experiments.baseline_delivery.common import atomic_json,read,digest,fingerprint
from scripts import experiment as e
from scripts.formal_progress import Dashboard


class Tests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();self.suite=self.root/'runs/existing';self.folder=self.suite/'jobs'/m.JOB
        self.config=dict(model='mega',task='text',batch_size=128,epochs=100,source_revision='original')
        atomic_json(self.folder/'config.json',self.config)
        cp=self.folder/'checkpoints/step-6200.pt';cp.parent.mkdir();cp.write_bytes(b'full-state fixture')
        self.cp=dict(schema='qkan-training-checkpoint.v1',file=cp.name,sha256=digest(cp),step=6200,contract='contract')
        atomic_json(self.folder/'checkpoints/latest.json',self.cp)
        atomic_json(self.suite/'suite.json',dict(jobs=[dict(id=m.JOB),dict(id='mamba1-retrieval-s42')]))
        atomic_json(self.suite/'status.json',{m.JOB:dict(state='paused'),'mamba1-retrieval-s42':dict(state='paused')})
        atomic_json(self.suite/'mamba-input.json',dict(data='data',cache='cache',original_root='original-root'))
        self.review=self.root/'review.json'
        atomic_json(self.review,dict(config_sha256=fingerprint(self.config),contract='contract',
            qualified_source_step=6016,qualified_binding={},quality_scope='Limited val256; prospective variant'))

    def activate(self,mode='fft-tail-bf16'):
        with patch.object(m,'REVIEW',self.review),patch.object(m,'binding',return_value={'qualified':'sha'}), \
                patch.object(m,'check_code'),patch.object(m,'revision',return_value='new-controller'), \
                patch('builtins.print'):
            return m.activate(self.root,self.suite,mode)

    def test_latest_switch_and_rollback_preserve_configs_roster_status_and_completed_results(self):
        complete=self.suite/'jobs/completed/summary.json';atomic_json(complete,dict(computed=True))
        paths=[self.folder/'config.json',self.suite/'suite.json',self.suite/'status.json',complete]
        before={p:p.read_bytes() for p in paths}
        record=self.activate()
        self.assertEqual(record['transitions'][0]['checkpoint']['step'],6200)
        baseline=self.suite/'events'/record['amendment']/'baseline/checkpoints'/self.cp['file']
        self.assertEqual(digest(baseline),self.cp['sha256'])
        # A later latest full generation must be used on rollback, not source6016.
        cp=self.folder/'checkpoints/step-6210.pt';cp.write_bytes(b'new full state')
        later=dict(self.cp,file=cp.name,sha256=digest(cp),step=6210)
        atomic_json(self.folder/'checkpoints/latest.json',later)
        rolled=self.activate('original')
        self.assertEqual(rolled['amendment'],record['amendment'])
        self.assertEqual(rolled['transitions'][-1]['checkpoint']['step'],6210)
        self.assertEqual(read(self.folder/'checkpoints/latest.json'),later)
        self.assertEqual({p:p.read_bytes() for p in paths},before)
        self.assertIsNone(m.lineage(rolled)['selection'])

    def test_completed_job_and_wrong_config_fail_without_amendment(self):
        atomic_json(self.folder/'summary.json',dict(computed=True))
        with self.assertRaises(ValueError):self.activate()
        (self.folder/'summary.json').unlink()
        atomic_json(self.folder/'config.json',{**self.config,'task':'image'})
        with self.assertRaises(ValueError):self.activate()
        self.assertFalse((self.suite/m.RECORD).exists())

    def test_dispatch_uses_private_seams_and_delegates_other_jobs(self):
        record=self.activate()
        def execute(args,suite):
            return job_spec(suite,{},args),validate_result(suite,{},args),read(suite/'suite.json')
        target=SimpleNamespace(execute=execute,job_spec=Mock(return_value='old-worker'),
            validate_result=Mock(return_value='old-validation'))
        ordered=Mock(return_value='retrieval-last-view')
        with patch.object(m,'check_record',return_value=record):function=m.attach(target,self.root,self.suite,ordered)
        other=dict(id='mamba1-retrieval-s42')
        self.assertEqual(function(other,self.suite),('old-worker','old-validation','retrieval-last-view'))
        atomic_json(self.folder/'summary.json',dict(execution_amendment=m.lineage(record)))
        spec,value,_=function(dict(id=m.JOB),self.suite)
        self.assertEqual(spec['cwd'],record['code_root']);self.assertIn('--worker',spec['argv'])
        self.assertIn('.venv-mega',spec['argv'][0]);self.assertEqual(value,'old-validation')
        self.assertNotIn('job_spec',execute.__globals__)
        atomic_json(self.folder/'summary.json',dict(execution_amendment={}))
        with self.assertRaises(ValueError):function(dict(id=m.JOB),self.suite)

    def test_cli_one_entry_routes_switch_with_dashboard_and_retrieval_last(self):
        with patch.object(e,'resume_command',return_value=['pinned-dispatch']), \
                patch.object(e,'display_module',return_value='beautified-ui'),patch.object(e,'supervise',return_value=0) as run:
            self.assertEqual(e.main(['--root',str(self.root),'resume','--suite',str(self.suite),
                '--mega-policy','fft-tail-bf16']),0)
        self.assertEqual(run.call_args.kwargs,dict(task_order='retrieval-last',mega_policy='fft-tail-bf16'))
        self.assertEqual(run.call_args.args[3],'beautified-ui')

    def test_runpy_helper_keeps_original_package_unimported(self):
        source=(m.ROOT/'experiments/baseline_delivery/mega_formal.py').read_text()
        import ast
        tree=ast.parse(source)
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.level for n in imports))
        self.assertEqual(e.mega_control()['JOB'],m.JOB)

    def test_controller_revision_does_not_use_already_imported_old_common_root(self):
        record=dict(code_root=str(m.ROOT),code_revision='new-controller',binding={'plugin':'qualified'})
        with patch.object(m,'REVIEW',self.review),patch.object(m,'binding',return_value=record['binding']), \
                patch('experiments.baseline_delivery.common.revision',return_value='old-parent'), \
                patch.object(m.subprocess,'check_output',side_effect=['new-controller\n','']) as git:
            m.check_code(record)
        self.assertTrue(all(call.args[0][2]==str(m.ROOT) for call in git.call_args_list))

    def test_dashboard_requires_current_pid_and_observed_calls(self):
        record=self.activate()
        atomic_json(self.suite/'status.json',{m.JOB:dict(state='running',pid=321,started=100,physical_gpu='0')})
        atomic_json(self.folder/'heartbeat.json',dict(updated_at=110,run_started_at=100,optimizer_step=6201,epoch=40))
        proof=dict(pid=321,updated_at=110,amendment=record['amendment'],mode=record['mode'],
            training_calls=1,fft_calls=12,island_calls=1)
        with patch('scripts.formal_progress.subprocess.run',side_effect=OSError('No GPU in owner test')):
            for changes,observed in (({},True),({'pid':320},False),({'fft_calls':0},False),({'updated_at':99},False)):
                atomic_json(self.folder/'mega-execution.json',{**proof,**changes})
                frame=Dashboard(self.root,self.suite).render(now=111)
                self.assertIn('FFT补齐＋尾层 FFN BF16',frame)
                self.assertEqual('实际调用已确认' in frame,observed)

    def test_native_hooks_annotate_latest_save_summary_and_return_without_model_imports(self):
        record=self.activate();w=ModuleType('experiments.baseline_delivery.worker')
        w.request_pause=Mock();controller=SimpleNamespace(state=dict(samples=[1]),policy=dict(warmup=2))
        state=dict(step=6200)
        policy=SimpleNamespace(status=lambda:dict(plugin='mega.fft_tail_bf16',graph=False,
            training_calls=1,fft_calls=12,island_calls=1,precision='tail BF16'))
        # The mock worker exercises the real wrapper hooks, not framework math.
        def run(config,data,cache,output,device,**kwargs):
            hooks=kwargs['execution'];hooks.model_context(object())
            hooks.configure_controller(controller,state)
            self.assertEqual(state['step'],6200);self.assertEqual(controller.state['samples'],[])
            atomic_json(output/'environment.json',dict(precision='old'))
            hooks.before_save(state);hooks.after_save(state);hooks.on_progress(dict(optimizer_step=6201))
            atomic_json(output/'summary.json',dict(optimizer_steps=6201))
            return 0
        w.run=run
        from experiments.baseline_delivery import execution_plugins as plugins
        with patch.object(m,'check_record',return_value=record),patch.object(m,'ROOT',self.root), \
                patch.dict(sys.modules,{'torch':ModuleType('torch'),'experiments.baseline_delivery.worker':w}), \
                patch.object(plugins,'create',return_value=policy),patch.object(m.subprocess,'check_call') as certificate, \
                patch('experiments.baseline_delivery.common.lock',side_effect=lambda _:nullcontext()), \
                patch.object(m.signal,'signal'),patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload:
            self.assertEqual(m.native(self.suite),0)
        self.assertEqual(read(self.folder/'summary.json')['execution_amendment']['amendment'],record['amendment'])
        self.assertEqual(read(self.folder/'mega-execution.json')['fft_calls'],12)
        self.assertEqual(certificate.call_args.kwargs['cwd'],'original-root')
        self.assertEqual(certificate.call_args.kwargs['env']['PYTHONPATH'],'original-root')
        upload.assert_called_once();self.assertNotIn('numpy',sys.modules)


if __name__=='__main__':unittest.main()
