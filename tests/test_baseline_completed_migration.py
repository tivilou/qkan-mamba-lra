"""Mixed completed/unfinished migration, pure stdlib fixtures and mocked children."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery.common import atomic_json,digest,fingerprint,read
from experiments.baseline_delivery.migration import SOURCE_REVISION,ACCELERATION_REVISION,snapshot,preserve_completed,verify_completed,admit,validate_source_admission


class CompletedMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'old/runs/source';self.target=self.root/'new/runs/target'
        self.source.mkdir(parents=True);self.target.mkdir(parents=True)
        self.cache=self.root/'cache';self.data=self.root/'data'
        self.config=dict(source_revision=SOURCE_REVISION,model='transformer',task='listops',seed=42,batch_size=32,data_hashes={})
        self.done='transformer-listops-s42';self.waiting='transformer-text-s42'
        self.folders={}
        jobs=[]
        for name in (self.done,self.waiting):
            folder=self.source/'jobs'/name;folder.mkdir(parents=True);self.folders[name]=folder
            cfg={**self.config,'task':'listops' if name==self.done else 'text'}
            atomic_json(folder/'config.json',cfg);jobs.append(dict(id=name,config_sha256=fingerprint(cfg)))
        atomic_json(self.source/'suite.json',dict(source_revision=SOURCE_REVISION,data=str(self.data),jobs=jobs))
        atomic_json(self.source/'context.json',dict(source_revision=SOURCE_REVISION,cache=str(self.cache.resolve())))
        atomic_json(self.source/'gate-report.json',dict(status='passed',source_revision=SOURCE_REVISION))
        atomic_json(self.source/'status.json',{self.done:dict(state='computed'),self.waiting:dict(state='pending')})
        self.complete(self.folders[self.done],self.config)

    def complete(self,folder,config):
        weights='best-model.msgpack' if config['model']=='s5' else 'best-model.pt'
        (folder/weights).write_bytes(b'opaque selected weights')
        atomic_json(folder/'summary.json',dict(formal=True,synthetic=False,
            **{k:config[k] for k in ('source_revision','task','model','seed')},data_hashes={},
            config_sha256=fingerprint(config),best_weights_sha256=digest(folder/weights),
            contract='old-contract',optimizer_steps=100))
        atomic_json(folder/'trace-validation.json',dict(passed=True,checkpoint_sha256=digest(folder/weights)))
        atomic_json(folder/'environment.json',dict(runtime='original'))
        cp=folder/'checkpoints';cp.mkdir();(cp/'state').write_bytes(b'opaque full checkpoint')
        atomic_json(cp/'latest.json',dict(schema='qkan-training-checkpoint.v1',file='state',sha256=digest(cp/'state'),step=100,contract='old-contract'))

    def stage(self):
        with patch('experiments.baseline_delivery.migration.subprocess.check_output',side_effect=[SOURCE_REVISION,'']):
            return snapshot(self.source,self.target,self.cache)

    def latest_admission(self):
        root=self.source.parent.parent
        policy_path=root/'experiments/baseline_delivery/fidelity-policy.json'
        atomic_json(policy_path,dict(models=['transformer'],reviewed_artifacts={}))
        policy_sha=digest(policy_path)
        context=dict(source_revision=ACCELERATION_REVISION,cache=str(self.cache.resolve()),
            policy_sha256=policy_sha,implementation_sha256='impl')
        atomic_json(self.source/'context.json',context)
        manifest=read(self.source/'suite.json');manifest['source_revision']=ACCELERATION_REVISION
        # First preserve the9a-style original completion inside an existing fork.
        from experiments.baseline_delivery.migration import completed_files
        entry=completed_files(self.folders[self.done],self.config)
        manifest['jobs'][0]['preserved_completion']=entry
        atomic_json(self.source/'migration-source/plan.json',dict(jobs={self.done:dict(completion=entry)}))
        cfg=read(self.folders[self.waiting]/'config.json');cfg['source_revision']=ACCELERATION_REVISION
        atomic_json(self.folders[self.waiting]/'config.json',cfg)
        manifest['jobs'][1]['config_sha256']=fingerprint(cfg)
        atomic_json(self.source/'suite.json',manifest)
        artifact=self.source/'proof.json';atomic_json(artifact,dict(passed=True))
        cert=dict(model='transformer',policy_sha256=policy_sha,
            evidence={artifact.relative_to(root).as_posix():digest(artifact)})
        certificate=self.source/'transformer-certificate.json';atomic_json(certificate,cert)
        atomic_json(self.source/'runtime-release.json',dict(source_revision=ACCELERATION_REVISION,
            formal_run_allowed=True,policy_sha256=policy_sha,implementation_sha256='impl',
            certificates=dict(transformer=dict(path=certificate.relative_to(root).as_posix(),sha256=digest(certificate)))))
        atomic_json(self.source/'gate-evidence.json',dict(context=context,certificates=dict(transformer=cert)))
        atomic_json(self.source/'gate-report.json',dict(status='passed',source_revision=ACCELERATION_REVISION,
            evidence_sha256=digest(self.source/'gate-evidence.json')))
        return entry

    def test_latest9ce_can_preserve_an_inherited_original_completion_again(self):
        expected=self.latest_admission()
        with patch('experiments.baseline_delivery.migration.subprocess.check_output',side_effect=[ACCELERATION_REVISION,'']):
            plan=snapshot(self.source,self.target,self.cache)
        self.assertEqual(plan['source_revision'],ACCELERATION_REVISION)
        job=preserve_completed(self.target,self.done)
        self.assertEqual(job['preserved_completion'],expected)
        self.assertEqual(read(self.target/'jobs'/self.done/'config.json')['source_revision'],SOURCE_REVISION)

    def test_latest9ce_changed_evidence_is_rejected_without_source_repair(self):
        self.latest_admission();artifact=self.source/'proof.json';artifact.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'runtime evidence changed'):
            validate_source_admission(self.source,ACCELERATION_REVISION)
        self.assertEqual(artifact.read_bytes(),b'changed')

    def test_latest9ce_inherited_completion_requires_original_provenance(self):
        self.latest_admission();path=self.source/'migration-source/plan.json'
        atomic_json(path,dict(jobs={self.done:{}}))
        with patch('experiments.baseline_delivery.migration.subprocess.check_output',side_effect=[ACCELERATION_REVISION,'']):
            with self.assertRaises(ValueError): snapshot(self.source,self.target,self.cache)
        self.assertFalse((self.target/'migration-source').exists())

    def test_preserves_exact_bytes_original_identity_and_idempotency(self):
        import json
        # Even harmless source JSON formatting must be preserved byte-for-byte.
        path=self.folders[self.done]/'config.json';path.write_text(json.dumps(self.config,separators=(',',':')),encoding='utf8')
        plan=self.stage();record=plan['jobs'][self.done]
        self.assertIsNone(record['checkpoint']);self.assertIn('completion',record)
        job=preserve_completed(self.target,self.done)
        self.assertEqual(job,preserve_completed(self.target,self.done))
        for name in record['completion']['files']:
            self.assertEqual((self.target/'jobs'/self.done/name).read_bytes(),(self.folders[self.done]/name).read_bytes())
        self.assertEqual(read(self.target/'jobs'/self.done/'config.json')['source_revision'],SOURCE_REVISION)
        self.assertIsNone(plan['jobs'][self.waiting]['checkpoint'])
        self.assertIn('gate-report.json',plan['source_admission'])

    def test_missing_or_corrupt_completion_never_becomes_pending(self):
        for name in ('trace-validation.json','best-model.pt','checkpoints/state'):
            p=self.folders[self.done]/name;original=p.read_bytes();p.write_bytes(b'corrupt')
            with self.assertRaises((ValueError,KeyError)): self.stage()
            self.assertFalse((self.target/'migration-source').exists());p.write_bytes(original)
        (self.folders[self.done]/'summary.json').unlink()
        with self.assertRaises(ValueError): self.stage()

    def test_changed_copied_summary_or_provenance_blocks_resume(self):
        self.stage();job=preserve_completed(self.target,self.done)
        bad=copy.deepcopy(job);bad['preserved_completion']['source_revision']='new'
        with self.assertRaises(ValueError): verify_completed(self.target,bad)
        atomic_json(self.target/'jobs'/self.done/'summary.json',{})
        with self.assertRaises(ValueError): verify_completed(self.target,job)
        with self.assertRaises(ValueError): preserve_completed(self.target,self.done)

    def test_inconsistent_formal_identity_or_weight_reference_is_rejected(self):
        folder=self.folders[self.done];path=folder/'summary.json';original=read(path)
        for change in (dict(source_revision='new'),dict(config_sha256='wrong'),dict(formal=False),
                       dict(best_weights_sha256='wrong'),dict(optimizer_steps=99)):
            atomic_json(path,{**original,**change})
            with self.assertRaises(ValueError): self.stage()
        atomic_json(path,original)
        atomic_json(folder/'trace-validation.json',dict(passed=True,checkpoint_sha256='wrong'))
        with self.assertRaises(ValueError): self.stage()

    def test_previous_generation_and_source_admission_remain_immutable(self):
        folder=self.folders[self.done];previous=folder/'checkpoints/previous';previous.write_bytes(b'previous full state')
        pointer=read(folder/'checkpoints/latest.json');pointer['previous']=[dict(file=previous.name,sha256=digest(previous))]
        atomic_json(folder/'checkpoints/latest.json',pointer)
        plan=self.stage();preserve_completed(self.target,self.done)
        self.assertEqual((self.target/'jobs'/self.done/'checkpoints/previous').read_bytes(),b'previous full state')
        self.assertEqual(digest(self.target/'migration-source/source-admission/gate-report.json'),plan['source_admission']['gate-report.json'])

    def test_all_completed_admission_launches_no_native_process(self):
        self.complete(self.folders[self.waiting],read(self.folders[self.waiting]/'config.json'))
        self.stage()
        configs={}
        for name in (self.done,self.waiting):
            preserve_completed(self.target,name);configs[name]=read(self.target/'jobs'/name/'config.json')
        run=Mock(side_effect=AssertionError('completed jobs must not launch'))
        report=admit(self.target,configs,self.data,self.cache,['0'],run,Mock(),{'0':'gpu'})
        run.assert_not_called();self.assertEqual(len(report['preserved_completions']),2)
        self.assertEqual(report['restores'],{})

    def test_completed_native_s5_uses_msgpack_without_loading_framework(self):
        config={**self.config,'model':'s5'};folder=self.root/'s5';folder.mkdir();atomic_json(folder/'config.json',config)
        self.complete(folder,config)
        from experiments.baseline_delivery.migration import completed_files
        files=completed_files(folder,config)['files']
        self.assertIn('best-model.msgpack',files);self.assertNotIn('best-model.pt',files)

    def test_admission_skips_completed_and_only_restores_started_job(self):
        folder=self.folders[self.waiting];self.complete(folder,read(folder/'config.json'))
        (folder/'summary.json').unlink()
        statuses=read(self.source/'status.json');statuses[self.waiting]={'state':'paused'};atomic_json(self.source/'status.json',statuses)
        self.stage();job=preserve_completed(self.target,self.done)
        config={**read(folder/'config.json'),'source_revision':'new','adaptive_memory':{'enabled':True}}
        out=self.target/'jobs'/self.waiting;out.mkdir(parents=True);atomic_json(out/'config.json',config)
        run=Mock(return_value=1)
        with self.assertRaises(ValueError):
            admit(self.target,{self.done:self.config,self.waiting:config},self.data,self.cache,['0'],run,lambda *args:args,{'0':'gpu'})
        self.assertEqual(run.call_count,1)
        self.assertIn(str(out/'config.json'),list(map(str,run.call_args.args[0])))

    def test_real_parent_never_launches_completed_job(self):
        from experiments.baseline_delivery import runner
        self.stage();done=preserve_completed(self.target,self.done)
        cfg={**read(self.folders[self.waiting]/'config.json'),'source_revision':'new','adaptive_memory':{'enabled':True}}
        folder=self.target/'jobs'/self.waiting;folder.mkdir(parents=True);atomic_json(folder/'config.json',cfg)
        waiting=dict(id=self.waiting,config_sha256=fingerprint(cfg))
        atomic_json(self.target/'suite.json',dict(jobs=[done,waiting]));atomic_json(self.target/'status.json',{self.done:dict(state='computed'),self.waiting:dict(state='pending')})
        def spawn(spec,log,env):
            self.assertEqual(spec['target'],'mock-native-worker')
            atomic_json(folder/'summary.json',dict(config_sha256=fingerprint(cfg)))
            atomic_json(folder/'trace-validation.json',dict(passed=True))
            child=Mock();child.pid=111;child.poll.return_value=0;return child
        with patch.object(runner,'check_release') as release,patch.object(runner,'child_spec',return_value=dict(target='mock-native-worker')) as build,patch.object(runner,'spawn',side_effect=spawn) as launch,patch.object(runner,'deliver') as deliver,patch.object(runner.time,'sleep'):
            runner.execute(self.target,['0'],self.cache,self.data)
        self.assertEqual(launch.call_count,1);release.assert_called_once_with(cfg);deliver.assert_called_once()
        self.assertIn(str(folder/'config.json'),list(map(str,build.call_args.args)))
        self.assertEqual(read(self.target/'status.json')[self.done],dict(state='computed',preserved=True))

    def test_final_delivery_marks_mixed_revisions_and_keeps_original_result(self):
        from experiments.baseline_delivery import runner
        self.stage();done=preserve_completed(self.target,self.done)
        config={**read(self.folders[self.waiting]/'config.json'),'source_revision':'new'}
        folder=self.target/'jobs'/self.waiting;folder.mkdir(parents=True);atomic_json(folder/'config.json',config)
        self.complete(folder,config)
        atomic_json(self.target/'suite.json',dict(source_revision='new',jobs=[done,dict(id=self.waiting,config_sha256=fingerprint(config))]))
        atomic_json(self.target/'status.json',{name:dict(state='computed') for name in (self.done,self.waiting)})
        atomic_json(self.target/'compute-complete.json',dict(state='computed',jobs=2))
        original=(self.target/'jobs'/self.done/'summary.json').read_bytes()
        with patch.object(runner,'ROOT',self.target.parent.parent),patch('scripts.exchange_artifacts.Transport'),patch('scripts.exchange_artifacts.prepare',return_value=dict(source_revision='new')) as inventory,patch('scripts.exchange_artifacts.deliver',return_value=dict(status='uploaded',files=[])):
            runner.deliver(self.target)
        result=read(self.target/'delivery.json')
        self.assertTrue(result['mixed_source_revisions'])
        self.assertEqual(result['result_source_revisions'],{self.done:SOURCE_REVISION,self.waiting:'new'})
        mappings=inventory.call_args.args[3]
        self.assertTrue(any('jobs/'+self.done+'/summary.json' in name for name in mappings))
        self.assertTrue(any('source-admission/gate-report.json' in name for name in mappings))
        self.assertEqual(original,(self.target/'jobs'/self.done/'summary.json').read_bytes())


if __name__=='__main__': unittest.main()
