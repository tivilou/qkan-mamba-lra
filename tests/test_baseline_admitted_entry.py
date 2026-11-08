"""Admission-to-scheduler launch regressions, no framework or GPU use."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from experiments.baseline_delivery.common import atomic_json,digest,fingerprint

def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

entry=module('run_admitted');display=module('formal_progress')

class AdmittedEntryTests(unittest.TestCase):
    def test_supervisor_uses_admitted_entry_instead_of_preparation(self):
        command=display.resume_command(Path('/project'),Path('/project/runs/suite'),{'gpus':'0'},Path('/cache'))
        self.assertEqual(Path(command[2]).name,'run_admitted.py')
        self.assertNotIn('run_formal_readiness.sh',' '.join(command))

    def fixture(self,root):
        suite=root/'runs/suite';suite.mkdir(parents=True)
        context=dict(source_revision='source',cache=str(root/'cache'),gpus='0')
        config=dict(source_revision='source',torch_acceleration={'mode':'eager'})
        atomic_json(suite/'context.json',context)
        atomic_json(suite/'gate-evidence.json',dict(context=context))
        atomic_json(suite/'gate-report.json',dict(status='passed',source_revision='source',evidence_sha256=digest(suite/'gate-evidence.json')))
        atomic_json(suite/'runtime-release.json',dict(status='released'))
        atomic_json(suite/'jobs/job/config.json',config)
        atomic_json(suite/'suite.json',dict(source_revision='source',data=str(root/'data'),gpu_inventory={'0':'device'},
            jobs=[dict(id='job',config_sha256=fingerprint(config))]))
        return suite,config

    def test_real_admitted_entry_launches_scheduler_with_persisted_eager_without_reselecting(self):
        self.run_case(None)

    def test_changed_config_rejected_before_scheduler(self): self.run_case('config')
    def test_changed_gpu_rejected_before_scheduler(self): self.run_case('gpu')
    def test_changed_evidence_rejected_before_scheduler(self): self.run_case('evidence')
    def test_dirty_executor_rejected_before_scheduler(self): self.run_case('dirty')
    def test_invalid_runtime_certificate_rejected_before_scheduler(self): self.run_case('certificate')
    def test_missing_migration_receipt_cannot_reexecute_restore(self): self.run_case('migration')

    def run_case(self,corrupt):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);suite,config=self.fixture(root);before=(suite/'jobs/job/config.json').read_bytes()
            if corrupt=='config': atomic_json(suite/'jobs/job/config.json',dict(config,torch_acceleration={'mode':'compile_graph'}))
            if corrupt=='evidence': atomic_json(suite/'gate-evidence.json',{})
            execute=Mock();admit=Mock();validate=Mock(side_effect=ValueError('bad certificate') if corrupt=='certificate' else None)
            replacements={
                'experiments.baseline_delivery.common.revision':lambda:'source',
                'experiments.baseline_delivery.readiness.validate_runtime_release':validate,
                'experiments.baseline_delivery.runner.execute':execute,
                'experiments.baseline_delivery.runner.discover_gpus':lambda _:(['0'],{'0':'changed' if corrupt=='gpu' else 'device'}),
                'experiments.baseline_delivery.sync.verify_data':Mock(),
                'experiments.baseline_delivery.bootstrap.check_exchange':Mock(),
                'experiments.baseline_delivery.migration.admit':admit}
            replacements['experiments.baseline_delivery.migration.plan_for']=lambda _: (
                dict(jobs={'job':dict(checkpoint={'sha256':'source'})}) if corrupt=='migration' else None)
            for target,value in replacements.items(): stack.enter_context(patch(target,value))
            stack.enter_context(patch.object(entry.subprocess,'check_output',return_value='dirty' if corrupt=='dirty' else ''))
            stack.enter_context(patch.dict(entry.os.environ,{},clear=False))
            if corrupt:
                with self.assertRaises(ValueError): entry.run(root,suite)
                execute.assert_not_called()
            else:
                entry.run(root,suite);execute.assert_called_once_with(suite,['0'],root/'cache',root/'data')
                self.assertEqual(admit.call_args.args[1]['job']['torch_acceleration']['mode'],'eager')
                self.assertEqual((suite/'jobs/job/config.json').read_bytes(),before)

if __name__=='__main__': unittest.main()
