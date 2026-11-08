"""Admission-to-scheduler launch regressions, no framework or GPU use."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from experiments.baseline_delivery.common import atomic_json,digest,fingerprint,read

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

    def sealed_fixture(self,root):
        from experiments.baseline_delivery.readiness import MODELS
        suite,config=self.fixture(root)
        context=dict(read(suite/'context.json'),implementation_sha256='implementation',policy_sha256='policy')
        atomic_json(suite/'context.json',context)
        admission=suite/'jobs/job/acceleration-admission.json'
        report=dict(schema='qkan-acceleration-admission.v1',attempts=[{'mode':'compile_graph','passed':False}],
            selected_mode='eager',config_sha256=fingerprint(config),passed=True)
        atomic_json(admission,report);original=admission.read_bytes()
        other=suite/'preflight/trace-validation.json';atomic_json(other,{'passed':True})
        certificates={};entries={}
        for model in MODELS:
            certificate=dict(model=model,policy_sha256='policy',evidence={})
            if model=='transformer': certificate['evidence']={
                admission.relative_to(root).as_posix():digest(admission),other.relative_to(root).as_posix():digest(other)}
            path=suite/(model+'-certificate.json');atomic_json(path,certificate)
            certificates[model]=certificate;entries[model]={'path':path.relative_to(root).as_posix(),'sha256':digest(path)}
        atomic_json(suite/'gate-evidence.json',dict(context=context,certificates=certificates,
            acceleration_admissions=[dict(job='job',report=report)]))
        atomic_json(suite/'gate-report.json',dict(status='passed',source_revision='source',evidence_sha256=digest(suite/'gate-evidence.json')))
        atomic_json(suite/'runtime-release.json',dict(status='released',source_revision='source',
            implementation_sha256='implementation',policy_sha256='policy',certificates=entries))
        return suite,admission,original,other

    def sealed_patches(self,stack,root):
        replacements={
            'experiments.baseline_delivery.common.revision':lambda:'source',
            'experiments.baseline_delivery.readiness.revision':lambda:'source',
            'experiments.baseline_delivery.readiness.policy_check':lambda:('implementation','policy'),
            'experiments.baseline_delivery.readiness.ROOT':root,
            'experiments.baseline_delivery.runner.discover_gpus':lambda _:(['0'],{'0':'device'}),
            'experiments.baseline_delivery.sync.verify_data':Mock(),
            'experiments.baseline_delivery.bootstrap.check_exchange':Mock(),
            'experiments.baseline_delivery.migration.admit':Mock(),
            'experiments.baseline_delivery.migration.plan_for':lambda _:None}
        for target,value in replacements.items(): stack.enter_context(patch(target,value))
        stack.enter_context(patch.object(entry.subprocess,'check_output',return_value=''))
        stack.enter_context(patch.dict(entry.os.environ,{},clear=False))

    def test_overwritten_admission_restored_then_real_certificate_validator_allows_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);suite,admission,original,_=self.sealed_fixture(root)
            checkpoint=suite/'jobs/job/latest.pt';checkpoint.write_bytes(b'full checkpoint sentinel')
            protected=[p for p in suite.rglob('*') if p.is_file() and p!=admission]
            before={p:p.read_bytes() for p in protected}
            atomic_json(admission,dict(read(admission),selected_mode='compile_graph'))
            overwritten=admission.read_bytes()
            self.sealed_patches(stack,root)
            execute=stack.enter_context(patch('experiments.baseline_delivery.runner.execute'))
            returned=stack.enter_context(patch.object(entry,'return_repair'))
            entry.run(root,suite)
            execute.assert_called_once_with(suite,['0'],root/'cache',root/'data')
            self.assertEqual(admission.read_bytes(),original)
            for path,value in before.items(): self.assertEqual(path.read_bytes(),value)
            folder=returned.call_args.args[0]
            self.assertEqual((folder/'job.before.json').read_bytes(),overwritten)
            receipt=read(folder/'entry-repair.json')
            self.assertEqual(receipt['records'][0]['restored_sha256'],digest(admission))
            # A second resume is idempotent: no new backup/return or reselection.
            returned.reset_mock();entry.run(root,suite);returned.assert_not_called()
            self.assertEqual(len(list(suite.glob('entry-repair-*'))),1)

    def test_damaged_non_admission_evidence_blocks_before_any_repair(self): self.repair_rejected('other')
    def test_damaged_second_certificate_blocks_before_any_repair(self): self.repair_rejected('certificate')
    def test_damaged_sealed_gate_evidence_cannot_supply_repair(self): self.repair_rejected('gate')
    def test_reconstruction_hash_must_equal_original_certificate(self): self.repair_rejected('candidate')
    def test_admission_repair_cannot_change_recorded_config(self): self.repair_rejected('config')
    def test_admission_repair_checks_original_policy(self): self.repair_rejected('policy')
    def test_missing_non_admission_evidence_blocks_before_any_repair(self): self.repair_rejected('missing')

    def repair_rejected(self,corrupt):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);suite,admission,_,other=self.sealed_fixture(root)
            atomic_json(admission,dict(read(admission),selected_mode='compile_graph'));overwritten=admission.read_bytes()
            if corrupt=='other': atomic_json(other,{'passed':False})
            if corrupt=='missing': other.unlink()
            if corrupt=='certificate': atomic_json(suite/'s5-certificate.json',{})
            if corrupt in ('gate','candidate'):
                evidence=read(suite/'gate-evidence.json');evidence['acceleration_admissions'][0]['report']['passed']=False
                atomic_json(suite/'gate-evidence.json',evidence)
                if corrupt=='candidate':
                    gate=read(suite/'gate-report.json');gate['evidence_sha256']=digest(suite/'gate-evidence.json')
                    atomic_json(suite/'gate-report.json',gate)
            if corrupt=='config': atomic_json(suite/'jobs/job/config.json',{'changed':True})
            self.sealed_patches(stack,root)
            if corrupt=='policy': stack.enter_context(patch('experiments.baseline_delivery.readiness.policy_check',return_value=('changed','policy')))
            execute=stack.enter_context(patch('experiments.baseline_delivery.runner.execute'))
            with self.assertRaises((ValueError,FileNotFoundError)): entry.run(root,suite)
            execute.assert_not_called();self.assertEqual(admission.read_bytes(),overwritten)
            self.assertEqual(list(suite.glob('entry-repair-*')),[])

    def test_failure_diagnostic_excludes_exception_values_and_locals(self):
        secret='private-token-and-array-values'
        try: raise ValueError(secret)
        except ValueError as error: report=entry.failure_report(error,'certificate')
        self.assertNotIn(secret,json.dumps(report));self.assertEqual(report['stage'],'certificate')
        self.assertEqual(report['error_type'],'ValueError')

    def test_failed_entry_automatically_returns_only_compact_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp,patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload:
            root=Path(tmp);suite=root/'runs/suite'
            report=dict(schema='qkan-admitted-entry-failure.v1',stage='certificate',error_type='ValueError',project_frames=[])
            entry.return_failure(root,suite,report)
            folder,experiment,files=upload.call_args.args
            self.assertEqual(experiment,'admitted-entry');self.assertEqual(files,['entry-failure.json'])
            self.assertEqual(read(folder/'entry-failure.json'),report)
            status=read(suite/'entry-status.json')
            self.assertEqual(status['state'],'failed');self.assertEqual(status['stage'],'certificate')

    def test_repair_automatically_returns_only_compact_receipt(self):
        with patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload:
            entry.return_repair(Path('repair'))
            upload.assert_called_once_with(Path('repair'),'admitted-entry',['entry-repair.json'])

if __name__=='__main__': unittest.main()
