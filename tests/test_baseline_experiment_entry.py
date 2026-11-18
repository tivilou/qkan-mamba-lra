"""Unified user operations; stdlib and fakes only, never start model workers."""
from contextlib import nullcontext
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import experiment as e
from experiments.baseline_delivery.common import atomic_json


class UnifiedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.suite=self.root/'runs/key-bias-formal-fixture'
        self.suite.mkdir(parents=True)
        atomic_json(self.suite/'formal-input.json',dict(schema='qkan-key-bias-formal-input.v1',source_revision='pinned'))

    def validated(self):
        return patch.object(e.subprocess,'check_output',side_effect=['pinned\n',''])

    def test_candidate_resume_uses_original_interpreter_and_original_module(self):
        with self.validated(): command=e.resume_command(self.root,self.suite)
        self.assertEqual(command,[sys.executable,'-B','-m','experiments.baseline_delivery.key_bias_formal','--resume',str(self.suite)])
        self.assertNotIn('--migrate-from',command);self.assertNotIn('--prepare-only',command)

    def test_upload_only_has_no_training_supervisor(self):
        with self.validated(),patch.object(e,'supervise') as supervisor,patch.object(e.subprocess,'call',return_value=0) as call:
            self.assertEqual(e.main(['--root',str(self.root),'upload','--suite',str(self.suite)]),0)
        self.assertIn('--upload-only',call.call_args.args[0]);supervisor.assert_not_called()

    def test_changed_revision_and_dirty_checkout_reject(self):
        for values in (['different',''],['pinned',' M worker.py']):
            with patch.object(e.subprocess,'check_output',side_effect=values),self.assertRaises(ValueError):
                e.resume_command(self.root,self.suite)

    def test_missing_git_head_exits_cleanly_without_launch(self):
        with patch.object(e.subprocess,'check_output',side_effect=subprocess.CalledProcessError(128,['git','rev-parse','HEAD'])), \
                patch.object(e,'supervise') as train,patch('sys.stderr',io.StringIO()):
            self.assertEqual(e.main(['--root',str(self.root),'resume','--suite',str(self.suite)]),2)
        train.assert_not_called()

    def test_unknown_formal_schema_rejects(self):
        atomic_json(self.suite/'formal-input.json',dict(schema='other'))
        with self.assertRaises(ValueError):e.validate_executor(self.root,self.suite)

    def legacy(self):
        (self.suite/'formal-input.json').unlink()
        for name in ('context.json','suite.json','gate-report.json'):
            atomic_json(self.suite/name,dict(source_revision='pinned',status='passed'))

    def test_legacy_resume_uses_its_admitted_entry_not_new_readiness(self):
        self.legacy()
        with self.validated():command=e.resume_command(self.root,self.suite)
        self.assertEqual(command[2],str(self.root/'scripts/run_admitted.py'))
        self.assertNotIn('experiments.baseline_delivery.readiness',command)

    def test_legacy_unadmitted_suite_rejects(self):
        self.legacy();atomic_json(self.suite/'gate-report.json',dict(source_revision='pinned',status='failed'))
        with self.assertRaises(ValueError):e.resume_command(self.root,self.suite)

    def test_legacy_upload_uses_existing_whitelist(self):
        self.legacy()
        with self.validated():command=e.resume_command(self.root,self.suite,True)
        self.assertIn('experiments.baseline_delivery.runner',command)
        self.assertIn('upload',command)

    def args(self,**kw):
        return SimpleNamespace(**dict(dict(profile='formal',cache=self.root/'cache',gpu='0',
            migrate_from=None,worker_from=None,output=None),**kw))

    def test_six_profiles_flatten_wrapper_chain_into_explicit_flags(self):
        for profile,flags in e.PROFILES.items():
            if profile in ('key-bias','bf16-graph'):continue
            command=e.start_command(self.args(profile=profile),self.root,self.suite)
            self.assertIn('--gates-only',command)
            self.assertIn('experiments.baseline_delivery.readiness',command)
            self.assertEqual(command[-len(flags):] if flags else [],flags)
            self.assertNotIn('bash',command)

    def test_candidate_start_prepares_before_any_long_training(self):
        args=self.args(profile='key-bias',migrate_from=self.root/'old',worker_from=self.root/'reviewed')
        command=e.start_command(args,self.root,None)
        self.assertIn('--prepare-only',command);self.assertIn('--worker-from',command)
        self.assertNotIn('--resume',command)

    def test_invalid_start_combinations_reject(self):
        for args in (self.args(profile='key-bias'),self.args(worker_from='unexpected'),
                     self.args(migrate_from='original'),self.args(profile='key-bias',migrate_from='x',worker_from='y',output='x')):
            with self.assertRaises(ValueError):e.start_command(args,self.root,self.suite)

    def test_path_confinement_and_existing_requirement(self):
        self.assertEqual(e.confined_suite(self.root,self.suite),self.suite)
        for value in (self.root/'outside',self.root/'runs/missing',self.suite/'nested'):
            with self.assertRaises(ValueError):e.confined_suite(self.root,value)

    def test_existing_suite_start_never_falls_back_to_new_training(self):
        with patch.object(e,'supervise') as run,patch('sys.stderr',io.StringIO()):
            code=e.main(['--root',str(self.root),'start','--cache','cache','--output',str(self.suite)])
        self.assertEqual(code,2);run.assert_not_called()

    def test_resume_default_is_integrated_ui(self):
        with self.validated(),patch.object(e,'display_module',return_value='original-ui'),patch.object(e,'supervise',return_value=75) as run:
            code=e.main(['--root',str(self.root),'resume','--suite',str(self.suite)])
        self.assertEqual(code,75);self.assertEqual(run.call_args.args[3],'original-ui')

    def test_readonly_watch_does_not_validate_launch_or_write(self):
        before={p.name:p.read_bytes() for p in self.suite.iterdir()}
        ui={'Dashboard':Mock(return_value=Mock(render=Mock(return_value='fixture progress'))),'Terminal':Mock()}
        with patch.object(e,'display_module',return_value=ui),patch.object(e,'resume_command') as resume,patch.object(e.subprocess,'call') as launch,patch('builtins.print'):
            self.assertEqual(e.main(['--root',str(self.root),'watch','--suite',str(self.suite),'--once']),0)
        resume.assert_not_called();launch.assert_not_called()
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.suite.iterdir()})

    def test_diagnostic_is_explicit_and_maps_exact_module(self):
        with patch.object(e.subprocess,'call',return_value=0) as call,patch.object(e,'supervise') as train:
            self.assertEqual(e.main(['--root',str(self.root),'diagnose','padding','--','--help']),0)
        self.assertEqual(call.call_args.args[0][-2:],['experiments.baseline_delivery.padding_profile','--help'])
        train.assert_not_called()

    def test_created_suite_parser_reads_only_literal_prefix(self):
        log=self.root/'parent.log';log.write_text('private raw log\nFormal continuation Suite: '+str(self.suite)+'\n',encoding='utf8')
        self.assertEqual(e.discover_created(self.root,log,0),self.suite)
        self.assertIsNone(e.discover_created(self.root,log,log.stat().st_size))

    def test_no_native_framework_import_from_entry_and_display(self):
        ui=e.display_module();self.assertIn('Dashboard',ui);self.assertIn('Terminal',ui)
        self.assertFalse(any(n.split('.')[0] in ('torch','numpy','jax','tensorflow') for n in sys.modules))

    def test_actual_cli_help_and_bad_operation(self):
        script=Path(e.__file__).resolve()
        for tail,expected in ((['--help'],0),(['resume','--help'],0),(['not-an-operation'],2)):
            result=subprocess.run([sys.executable,'-B','-X','utf8',str(script),*tail],capture_output=True,text=True,encoding='utf8')
            self.assertEqual(result.returncode,expected,result.stderr)

    def run_supervisor(self,children,callback=None,after=False):
        ui={'Terminal':Mock(return_value=Mock()),'Dashboard':Mock(return_value=Mock(render=Mock(return_value='original dashboard')))}
        def signal_handler(sig,handler):
            if sig==signal.SIGTERM:self.pause=handler
            return signal.SIG_DFL
        def spawn(*a,**kw):
            child=children.pop(0)
            if callback:callback(child)
            return child
        with patch.object(e,'require_linux'),patch.object(e.signal,'SIGHUP',getattr(signal,'SIGHUP',999),create=True), \
                patch.object(e.signal,'signal',side_effect=signal_handler),patch('experiments.baseline_delivery.common.lock',side_effect=lambda p:nullcontext()), \
                patch.object(e.subprocess,'Popen',side_effect=spawn),patch.object(e.time,'sleep'), \
                patch.object(e,'resume_command',return_value=['original-resume']) as resume,patch('builtins.print'):
            code=e.supervise(self.root,self.suite,['original-entry'],ui,after_prepare=after)
        return code,ui,resume

    def test_supervisor_renders_original_ui_captures_logs_and_propagates_exit(self):
        child=Mock(pid=123,returncode=75);child.poll.side_effect=[None,75,75]
        code,ui,_=self.run_supervisor([child])
        self.assertEqual(code,75);self.assertIn('original dashboard',ui['Terminal'].return_value.show.call_args.args[0])
        ui['Terminal'].return_value.close.assert_called_once()
        self.assertEqual(json.loads((self.suite/'display-heartbeat.json').read_text())['runner_pid'],123)

    def test_failed_preparation_never_dispatches_longrun(self):
        child=Mock(pid=123,returncode=2);child.poll.return_value=2
        code,_,resume=self.run_supervisor([child],after=True)
        self.assertEqual(code,2);resume.assert_not_called()

    def test_successful_preparation_continues_without_second_user_command(self):
        children=[Mock(pid=123,returncode=0),Mock(pid=124,returncode=75)]
        children[0].poll.return_value=0;children[1].poll.return_value=75
        code,_,resume=self.run_supervisor(children,after=True)
        self.assertEqual(code,75);resume.assert_called_once_with(self.root,self.suite)

    def test_safe_pause_forwards_to_owned_parent_and_never_starts_next_stage(self):
        child=Mock(pid=123,returncode=75);child.poll.side_effect=[None,None,75,75]
        def callback(child):self.pause(signal.SIGTERM,None)
        code,_,resume=self.run_supervisor([child],callback,after=True)
        self.assertEqual(code,75);child.send_signal.assert_called_once_with(signal.SIGTERM);resume.assert_not_called()

    def test_busy_supervisor_lock_never_launches_a_second_parent(self):
        with patch.object(e,'require_linux'),patch.object(e.signal,'SIGHUP',getattr(signal,'SIGHUP',999),create=True), \
                patch.object(e.signal,'signal',return_value=signal.SIG_DFL), \
                patch('experiments.baseline_delivery.common.lock',side_effect=OSError('already owned')), \
                patch.object(e.subprocess,'Popen') as launch,patch('builtins.print'):
            with self.assertRaises(OSError):e.supervise(self.root,self.suite,['original-entry'],{'Terminal':Mock()})
        launch.assert_not_called()

    def test_prepare_only_never_auto_continues(self):
        with patch.object(e,'display_module',return_value='ui'),patch.object(e,'supervise',return_value=0) as run:
            self.assertEqual(e.main(['--root',str(self.root),'start','--cache','cache','--prepare-only']),0)
        self.assertFalse(run.call_args.args[-1])

    def test_old_executor_upload_does_not_load_ui(self):
        with self.validated(),patch.object(e,'display_module') as ui,patch.object(e.subprocess,'call',return_value=0):
            e.main(['--root',str(self.root),'upload','--suite',str(self.suite)])
        ui.assert_not_called()

    def test_error_in_ui_preserves_cleanup_and_waits_owned_child(self):
        child=Mock(pid=321,returncode=75);child.poll.return_value=None
        ui={'Terminal':Mock(return_value=Mock(show=Mock(side_effect=ValueError('display fixture')))),
            'Dashboard':Mock(return_value=Mock(render=Mock(return_value='frame')))}
        with patch.object(e,'require_linux'),patch.object(e.signal,'SIGHUP',getattr(signal,'SIGHUP',999),create=True), \
                patch.object(e.signal,'signal',return_value=signal.SIG_DFL), \
                patch('experiments.baseline_delivery.common.lock',side_effect=lambda p:nullcontext()), \
                patch.object(e.subprocess,'Popen',return_value=child),patch('builtins.print'):
            with self.assertRaises(ValueError):e.supervise(self.root,self.suite,['entry'],ui)
        child.send_signal.assert_called_once_with(signal.SIGTERM);child.wait.assert_called_once()
        ui['Terminal'].return_value.close.assert_called_once()

    def test_actual_standalone_watch_cli_is_readonly_and_native_free(self):
        launcher=self.root/'experiment.py';launcher.write_bytes(Path(e.__file__).read_bytes())
        display=Path(e.__file__).with_name('formal_progress.py')
        (self.root/'formal_progress.py').write_bytes(display.read_bytes())
        atomic_json(self.suite/'suite.json',dict(jobs=[dict(id='transformer-retrieval-s42')]))
        atomic_json(self.suite/'status.json',{'transformer-retrieval-s42':dict(state='paused')})
        before={p.name:p.read_bytes() for p in self.suite.iterdir()}
        code='''import importlib.abc,runpy,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','numpy','jax','tensorflow'):raise AssertionError(name)
sys.meta_path.insert(0,Guard())
import subprocess
def blocked(*a,**k):raise AssertionError('No child process in paused watch fixture')
subprocess.Popen=blocked
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
        result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(launcher),
            '--root',str(self.root),'watch','--suite',str(self.suite),'--once'],capture_output=True,text=True,encoding='utf8')
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('暂停 1',result.stdout)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.suite.iterdir()})


if __name__=='__main__':unittest.main()
