"""Transport-only regression: upstream scripts namespace vs pinned client."""
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock,patch

from scripts import experiment as e,pinned_mega_worker as helper
from experiments.baseline_delivery.common import atomic_json


ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_foreign_scripts_reproduces_error_then_exact_client_cache_fixes_it(self):
        foreign=ModuleType('scripts');foreign.__path__=[]
        with patch.dict(sys.modules,{'scripts':foreign}):
            original=sys.modules.pop('scripts.exchange_artifacts',None)
            try:
                with self.assertRaises(ModuleNotFoundError):
                    exec('from scripts.exchange_artifacts import Transport',{})
                client=helper.preload(ROOT)
                namespace={};exec('from scripts.exchange_artifacts import Transport, atomic_json, prepare, deliver, safe_path',namespace)
                self.assertIs(namespace['Transport'],client.Transport)
                self.assertIs(sys.modules['scripts'],foreign)
                self.assertEqual(Path(client.__file__).resolve(),ROOT/'scripts/exchange_artifacts.py')
            finally:
                sys.modules.pop('scripts.exchange_artifacts',None)
                if original is not None:sys.modules['scripts.exchange_artifacts']=original

    def test_worker_entry_preserves_target_arguments_and_own_pinned_root(self):
        with patch.object(helper,'preload') as preload,patch.object(helper.runpy,'run_module') as run:
            before=sys.argv
            helper.main(['--root',str(ROOT),'--','--worker','--suite','/existing-suite'])
            preload.assert_called_once_with(ROOT)
            run.assert_called_once_with(helper.TARGET,run_name='__main__',alter_sys=True)
            self.assertIs(sys.argv,before)

    def test_transport_wrapper_keeps_interpreter_cwd_and_scientific_arguments(self):
        spec=dict(target=helper.TARGET,cwd='/pinned-5f96729',argv=['/pinned/.venv-mega/bin/python','-B','-m',helper.TARGET,'--worker','--suite','/same-suite'])
        builder=Mock(return_value=spec)
        actual=e.transport_spec(builder)('suite','record','job')
        self.assertEqual(actual['cwd'],spec['cwd']);self.assertEqual(actual['argv'][:2],spec['argv'][:2])
        self.assertEqual(actual['argv'][-3:],spec['argv'][-3:])
        self.assertIn('/pinned-5f96729',actual['argv'])
        self.assertNotIn('stop-after',str(actual));self.assertEqual(len(actual['transport_wrapper']['sha256']),64)
        self.assertEqual(spec['argv'][2],'-m')

    def test_other_workers_are_identical_and_no_module_globals_replaced(self):
        spec=dict(target='experiments.baseline_delivery.worker',argv=['python','-m','experiments.baseline_delivery.worker'])
        self.assertIs(e.transport_spec(lambda *args:spec)(None,None,None),spec)

    def test_new_controller_loads_recorded_science_helper_not_its_own_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);suite=root/'runs/suite';pinned=root/'pinned'
            target=pinned/'experiments/baseline_delivery/mega_formal.py'
            target.parent.mkdir(parents=True);target.write_text('PINNED = "5f96729"\n')
            atomic_json(suite/'mega-policy.json',dict(code_root=str(pinned)))
            self.assertEqual(e.mega_control(suite)['PINNED'],'5f96729')
        # Default initial activation still uses the current implementation.
        self.assertEqual(e.mega_control()['JOB'],'mega-text-s42')

    def test_conflicting_client_fails_before_worker_import(self):
        wrong=ModuleType('scripts.exchange_artifacts');wrong.__file__='/different-client.py'
        with patch.dict(sys.modules,{'scripts.exchange_artifacts':wrong}):
            with self.assertRaises(ValueError):helper.preload(ROOT)


if __name__=='__main__':unittest.main()
