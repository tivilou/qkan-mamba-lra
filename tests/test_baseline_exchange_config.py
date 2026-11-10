"""Private configuration persistence and the actual gates-only shell handoff."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import exchange_config as c
from experiments.baseline_delivery.bootstrap import check_exchange


class ExchangeConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.home=self.root/'account';self.home.mkdir()
        self.repo=self.root/'checkout';self.repo.mkdir();self.ca=self.root/'ca.crt';self.ca.write_text('certificate fixture')
        self.values=dict(zip(c.KEYS,('https://example.invalid:18084','private-fixture',str(self.ca.resolve()))))
        stack=contextlib.ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ,{},clear=True))
        stack.enter_context(patch.object(c.Path,'home',return_value=self.home))
        stack.enter_context(patch.object(c,'ROOT',self.repo))

    def write(self,path,values=None):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(values or self.values));path.chmod(0o600)

    def test_complete_environment_is_kept_and_does_not_touch_any_profile(self):
        os.environ.update(self.values)
        with patch.object(c,'read_private',side_effect=AssertionError('must not read')):
            check_exchange()
        self.assertEqual(c.load(),self.values);self.assertFalse(c.profile_path().exists())

    def test_saved_profile_loads_after_a_fresh_process_environment_and_across_worktrees(self):
        path=c.save(self.values);self.assertEqual(path,self.home/'qkan-private/exchange.json')
        check_exchange();self.assertEqual({key:os.environ[key] for key in c.KEYS},self.values)
        os.environ.clear()
        with patch.object(c,'ROOT',self.root/'separate-worktree'): check_exchange()
        self.assertEqual(os.environ[c.KEYS[1]],self.values[c.KEYS[1]])

    def test_configuration_save_uses_owner_only_mode_and_does_not_print_token(self):
        os.environ.update(self.values)
        output=io.StringIO()
        with patch.object(sys,'argv',['configure','--configure']),contextlib.redirect_stdout(output): c.main()
        path=c.profile_path();self.assertEqual(json.loads(path.read_text()),self.values)
        self.assertNotIn(self.values[c.KEYS[1]],output.getvalue())
        if os.name=='posix': self.assertEqual(path.stat().st_mode&0o777,0o600)

    def test_existing_profile_is_idempotent_and_different_values_are_not_overwritten(self):
        path=c.save(self.values);before=path.read_bytes();c.save(self.values)
        with self.assertRaisesRegex(ValueError,'differs'): c.save(dict(self.values,PROJECT_EXCHANGE_TOKEN='another-token'))
        self.assertEqual(path.read_bytes(),before)

    def test_partial_environment_may_only_complete_a_matching_profile(self):
        c.save(self.values);os.environ[c.KEYS[0]]=self.values[c.KEYS[0]]+'/'
        self.assertEqual(c.load(),self.values)
        os.environ.clear();os.environ[c.KEYS[0]]='https://other.invalid'
        with self.assertRaisesRegex(ValueError,'conflicts'): c.load()
        self.assertNotIn(c.KEYS[1],os.environ)

    def test_explicit_profile_has_no_silent_fallback_to_another_account(self):
        c.save(self.values);os.environ['QKAN_EXCHANGE_CONFIG']=str(self.root/'absent.json')
        with self.assertRaisesRegex(ValueError,'missing'): c.load()
        os.environ['QKAN_EXCHANGE_CONFIG']='relative.json'
        with self.assertRaisesRegex(ValueError,'absolute'): c.load()

    def test_shared_bootstrap_profile_is_accepted_without_copying_secrets_to_source(self):
        self.write(self.repo/'.bootstrap/exchange.json')
        check_exchange();self.assertEqual(os.environ[c.KEYS[1]],self.values[c.KEYS[1]])
        self.assertFalse(c.profile_path().exists())

    def test_legacy_env_assignments_are_literal_and_never_execute_shell(self):
        path=self.home/'qkan-private/exchange.env';path.parent.mkdir()
        values=dict(self.values,PROJECT_EXCHANGE_TOKEN='$(touch marker)')
        path.write_text('\n'.join('export '+key+'='+json.dumps(value) for key,value in values.items()));path.chmod(0o600)
        self.assertEqual(c.load()[c.KEYS[1]],'$(touch marker)');self.assertFalse((self.root/'marker').exists())

    def test_invalid_profiles_do_not_expose_raw_documents(self):
        path=c.profile_path();path.parent.mkdir()
        for text in ('{"token":"DO-NOT-PRINT",','["DO-NOT-PRINT"]','{"PROJECT_EXCHANGE_URL": "https://DO-NOT-PRINT@invalid"}'):
            with self.subTest(text=text):
                path.write_text(text);path.chmod(0o600)
                with self.assertRaisesRegex(ValueError,'Invalid private') as error: c.load()
                self.assertNotIn('DO-NOT-PRINT',str(error.exception));self.assertNotIn(c.KEYS[1],os.environ)

    def test_missing_settings_give_one_time_configuration_command(self):
        with self.assertRaisesRegex(ValueError,'exchange_config --configure'): check_exchange()

    def test_invalid_origins_tokens_and_certificates_are_rejected_before_env_mutation(self):
        for origin in ('http://example.invalid','https://user:secret@example.invalid','https://example.invalid/path',
                'https://example.invalid?token=secret','https://example.invalid/#secret','https://[bad','https://bad host'):
            with self.subTest(origin=origin),self.assertRaises(ValueError): c.validate(dict(self.values,PROJECT_EXCHANGE_URL=origin))
        for token in ('', 'secret\nheader', 'secret\rheader','secret\x00'):
            with self.assertRaises(ValueError): c.validate(dict(self.values,PROJECT_EXCHANGE_TOKEN=token))
        with self.assertRaisesRegex(ValueError,'certificate'): c.validate(dict(self.values,PROJECT_EXCHANGE_CA_FILE=str(self.root/'absent.crt')))
        self.assertNotIn(c.KEYS[1],os.environ)

    @unittest.skipUnless(os.name=='posix','POSIX file ownership/mode')
    def test_readable_to_other_users_profile_is_rejected(self):
        self.write(c.profile_path());c.profile_path().chmod(0o644)
        with self.assertRaisesRegex(ValueError,'owner-only'): c.load()

    def test_noninteractive_missing_configuration_fails_without_getpass(self):
        with patch.object(sys,'argv',['configure','--configure']),patch.object(sys.stdin,'isatty',return_value=False),\
                patch.object(c.getpass,'getpass') as hidden,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error: c.main()
        self.assertEqual(error.exception.code,2);hidden.assert_not_called()


class GatesOnlyShellTests(unittest.TestCase):
    def run_entry(self,gates_only,readiness_code=0,supervisor_code=0):
        bash=(Path('D:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path(shutil.which('bash') or '/missing'))
        if not bash.is_file(): self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripts').mkdir();(root/'.venv-baselines/bin').mkdir(parents=True)
            source=Path(__file__).resolve().parents[1]
            shutil.copyfile(source/'scripts/migrate_fixed_to_adaptive.sh',root/'scripts/migrate_fixed_to_adaptive.sh')
            (root/'scripts/run_adaptive_readiness.sh').write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > readiness-args\nexit '+str(readiness_code)+'\n',newline='\n')
            python=root/'.venv-baselines/bin/python'
            python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > supervisor-args\nexit '+str(supervisor_code)+'\n',newline='\n');python.chmod(0o700)
            args=[str(bash),'scripts/migrate_fixed_to_adaptive.sh','--cache','/fixture/cache','--gpus','0']
            if gates_only: args+=['--gates-only']
            result=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=15)
            return result,(root/'readiness-args').read_text(),(root/'supervisor-args').read_text() if (root/'supervisor-args').exists() else None

    def test_gates_only_never_launches_the_training_supervisor(self):
        result,readiness,supervisor=self.run_entry(True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--gates-only',readiness)
        self.assertIsNone(supervisor);self.assertIn('No formal training started',result.stdout)

    def test_normal_migration_still_launches_supervisor_and_preserves_exit_status(self):
        result,_,supervisor=self.run_entry(False,supervisor_code=17)
        self.assertEqual(result.returncode,17,result.stderr);self.assertIn('scripts/formal_progress.py',supervisor)

    def test_readiness_failure_never_starts_training(self):
        result,_,supervisor=self.run_entry(False,readiness_code=9)
        self.assertEqual(result.returncode,9);self.assertIsNone(supervisor)


if __name__=='__main__': unittest.main()
