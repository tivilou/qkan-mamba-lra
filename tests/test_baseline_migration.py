"""Migration policy/filesystem tests only; no native checkpoint deserialization."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.baseline_delivery.common import atomic_json,digest,fingerprint,read
from experiments.baseline_delivery.migration import SOURCE_REVISION,ACCELERATION_REVISION,WARMUP_REVISION,compatible,snapshot,load_state,admit


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.old=self.base/'old/runs/suite';self.old.mkdir(parents=True)
        self.new=self.base/'new/runs/suite';self.new.mkdir(parents=True)
        self.cache=self.base/'cache';self.data=self.base/'data'
        self.config=dict(source_revision=SOURCE_REVISION,model='transformer',task='listops',
                         batch_size=32,seed=42,lr=.001,data_hashes={'train':'a'})
        self.target={**self.config,'source_revision':'new-revision','adaptive_memory':{'enabled':True}}
        self.name='transformer-listops-s42';folder=self.old/'jobs'/self.name;folder.mkdir(parents=True)
        atomic_json(folder/'config.json',self.config)
        self.checkpoint(folder,'original',12)
        atomic_json(self.old/'context.json',dict(source_revision=SOURCE_REVISION,cache=str(self.cache.resolve())))
        atomic_json(self.old/'suite.json',dict(source_revision=SOURCE_REVISION,data=str(self.data),
            jobs=[dict(id=self.name,config_sha256=fingerprint(self.config))]))
        atomic_json(self.old/'gate-report.json',dict(status='passed',source_revision=SOURCE_REVISION))
        atomic_json(self.old/'status.json',{self.name:dict(state='paused')})

    def checkpoint(self,folder,contract,step):
        cp=folder/'checkpoints';cp.mkdir(exist_ok=True);(cp/'state.pt').write_bytes(b'opaque native state')
        atomic_json(cp/'latest.json',dict(schema='qkan-training-checkpoint.v1',file='state.pt',
            sha256=digest(cp/'state.pt'),contract=contract,step=step))

    def stage(self):
        with patch('experiments.baseline_delivery.migration.subprocess.check_output',side_effect=[SOURCE_REVISION,'']):
            return snapshot(self.old,self.new,self.cache)

    def test_snapshot_preserves_original_and_hashes(self):
        before=(self.old/'jobs'/self.name/'checkpoints/latest.json').read_bytes()
        plan=self.stage()
        self.assertEqual(plan['jobs'][self.name]['checkpoint']['step'],12)
        self.assertEqual(before,(self.old/'jobs'/self.name/'checkpoints/latest.json').read_bytes())
        self.assertEqual((self.new/'migration-source'/self.name/'checkpoints/state.pt').read_bytes(),b'opaque native state')

    def test_scientific_config_change_rejected(self):
        compatible(self.config,self.target)
        for change in ({'lr':.002},{'batch_size':64},{'seed':43},{'data_hashes':{}},{'adaptive_memory':None}):
            with self.assertRaises(ValueError): compatible(self.config,{**self.target,**change})

    def test_wrong_source_revision_rejected(self):
        with self.assertRaises(ValueError): compatible({**self.config,'source_revision':'other'},self.target)

    def compiler_configs(self,model='transformer',revision=ACCELERATION_REVISION):
        from experiments.baseline_delivery.torch_acceleration import policy,legacy_policy
        from experiments.baseline_delivery.performance import policy as performance
        from experiments.baseline_delivery.logical_batch import policy as logical
        old=dict(self.config,source_revision=revision,model=model,
            performance=performance(),adaptive_memory=logical(32))
        new=dict(old,source_revision='next')
        if model!='s5':
            old['torch_acceleration']=legacy_policy('eager',1 if revision==ACCELERATION_REVISION else 2)
            new['torch_acceleration']=policy()
        return old,new

    def test_latest9ce_migration_preserves_scientific_fields_for_torch_and_s5(self):
        for model in ('transformer','s5'):
            old,new=self.compiler_configs(model);compatible(old,new)
            for changed in (dict(lr=.2),dict(batch_size=64),dict(performance={}),dict(adaptive_memory={})):
                with self.assertRaises(ValueError): compatible(old,dict(new,**changed))
        old,new=self.compiler_configs();old['torch_acceleration']['max_builds']=99
        with self.assertRaises(ValueError): compatible(old,new)

    def test_latest9ce_backend_selection_is_recertified_without_training_reset(self):
        old,new=self.compiler_configs();self.stage()
        folder=self.new/'migration-source'/self.name;atomic_json(folder/'config.json',old)
        plan=read(self.new/'migration-source/plan.json');plan['source_revision']=ACCELERATION_REVISION
        plan['jobs'][self.name]['config_sha256']=fingerprint(old);atomic_json(self.new/'migration-source/plan.json',plan)
        previous=dict(mode='eager',benchmarked=True,reason='NumericalMismatch')
        original=dict(contract='original',step=12,cursor=100,epoch=3,
            model={'weight':9},optimizer={'moment':8},rng=[7],scheduler={'progress':1000},
            sample_clock={'progress':1000},adaptive_memory={'oom_count':5,'failed_ceiling':16},
            migration={'source':'9a'},torch_acceleration=previous)
        loaded=load_state(self.new/'jobs'/self.name,new,'new',lambda _:'original',lambda *_:copy.deepcopy(original))
        for key in ('step','cursor','epoch','model','optimizer','rng','scheduler','sample_clock','adaptive_memory'):
            self.assertEqual(loaded[key],original[key])
        self.assertNotIn('torch_acceleration',loaded)
        self.assertEqual(loaded['migration']['previous_torch_acceleration'],previous)
        self.assertEqual(loaded['migration']['previous_migration'],{'source':'9a'})

    def test_current840_migration_preserves_full_state_and_recertifies_backend_only(self):
        from experiments.baseline_delivery.torch_acceleration import policy
        old,new=self.compiler_configs(revision=WARMUP_REVISION)
        new['torch_acceleration']=policy('compile_graph','aot_eager_default_partitioner','graph')
        compatible(old,new);self.stage()
        folder=self.new/'migration-source'/self.name;atomic_json(folder/'config.json',old)
        plan=read(self.new/'migration-source/plan.json');plan['source_revision']=WARMUP_REVISION
        plan['jobs'][self.name]['config_sha256']=fingerprint(old);atomic_json(self.new/'migration-source/plan.json',plan)
        original=dict(contract='original',step=12,cursor=100,epoch=3,permutation=[4,3,2,1],
            model={'weight':9},optimizer={'moment':8},rng=[7],scheduler={'progress':1000},
            sample_clock={'progress':1000},adaptive_memory={'oom_count':3,'micro_batch':16},
            migration={'source':'9ce'},torch_acceleration={'mode':'eager','reason':'NumericalMismatch'})
        before=(folder/'checkpoints/state.pt').read_bytes()
        # Calibration lives outside jobs/, but reads the exact same source suite.
        loaded=load_state(self.new/'calibration/key'/self.name,new,'new',lambda _:'original',
            lambda *_:copy.deepcopy(original),migration_suite=self.new)
        for key in original.keys()-{'contract','migration','torch_acceleration'}:
            self.assertEqual(loaded[key],original[key])
        self.assertEqual(loaded['migration']['previous_torch_acceleration'],original['torch_acceleration'])
        self.assertNotIn('torch_acceleration',loaded)
        self.assertEqual((folder/'checkpoints/state.pt').read_bytes(),before)
        old['torch_acceleration']['update_warmup']=2
        with self.assertRaises(ValueError): compatible(old,new)

    def test_corrupt_checkpoint_never_staged(self):
        (self.old/'jobs'/self.name/'checkpoints/state.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.stage()
        self.assertFalse((self.new/'migration-source').exists())
        self.assertEqual(list(self.new.glob('migration-pending-*')),[])

    def test_started_job_without_checkpoint_never_restarts_fresh(self):
        cp=self.old/'jobs'/self.name/'checkpoints'
        for p in cp.iterdir(): p.unlink()
        with self.assertRaises(ValueError): self.stage()

    def test_pending_job_is_recorded_without_checkpoint(self):
        cp=self.old/'jobs'/self.name/'checkpoints'
        for p in cp.iterdir(): p.unlink()
        atomic_json(self.old/'status.json',{self.name:dict(state='pending')})
        self.assertIsNone(self.stage()['jobs'][self.name]['checkpoint'])

    def test_completed_job_not_relabeled(self):
        atomic_json(self.old/'jobs'/self.name/'summary.json',{})
        with self.assertRaises(ValueError): self.stage()

    def test_path_traversal_rejected(self):
        pointer=self.old/'jobs'/self.name/'checkpoints/latest.json';v=read(pointer)
        v['file']='../outside.pt';atomic_json(pointer,v)
        with self.assertRaises(ValueError): self.stage()

    def test_rebinding_preserves_all_training_values(self):
        self.stage();out=self.new/'jobs'/self.name;out.mkdir(parents=True)
        original=dict(contract='original',step=12,weights=[1,2],optimizer={'moment':8},
            scheduler=12,rng=[7],cursor=384,history=[.4],best_weights=[1],training_seconds=100)
        loaded=load_state(out,self.target,'new-contract',lambda old:'original',lambda p,c:copy.deepcopy(original))
        for key,value in original.items():
            self.assertEqual(loaded[key], 'new-contract' if key=='contract' else value)
        self.assertEqual(loaded['migration']['optimizer_step'],12)

    def test_original_contract_must_match_real_data(self):
        self.stage();out=self.new/'jobs'/self.name
        with self.assertRaises(ValueError):
            load_state(out,self.target,'new',lambda old:'wrong-data',lambda *_:self.fail('must not deserialize'))

    def test_fresh_child_contract_and_failed_check_block_admission(self):
        self.stage();out=self.new/'jobs'/self.name;out.mkdir(parents=True)
        atomic_json(out/'config.json',self.target);calls=[]
        def build(*args): return dict(argv=list(map(str,args)),target='native-worker')
        def fail(spec,log,env): calls.append(spec);return 1
        with self.assertRaises(ValueError):
            admit(self.new,{self.name:self.target},self.data,self.cache,['0'],fail,build,{'0':'gpu'})
        self.assertIn('--migration-check-only',calls[0]['argv'])
        self.assertNotIn('--synthetic',calls[0]['argv'])
        self.assertFalse((self.new/'migration-report.json').exists())

    def test_changed_snapshot_pointer_rejected_before_loading(self):
        self.stage();folder=self.new/'migration-source'/self.name
        pointer=read(folder/'checkpoints/latest.json');pointer['step']=13
        atomic_json(folder/'checkpoints/latest.json',pointer)
        with self.assertRaises(ValueError):
            load_state(self.new/'jobs'/self.name,self.target,'new',lambda _: 'original',lambda *_:self.fail('changed snapshot'))

    def test_successful_admission_requires_matching_native_receipt_and_checkpoint(self):
        plan=self.stage();out=self.new/'jobs'/self.name;out.mkdir(parents=True)
        atomic_json(out/'config.json',self.target)
        def run(spec,log,env):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
            self.checkpoint(out,'new',12)
            atomic_json(out/'migration-validation.json',dict(passed=True,
                target_config_sha256=fingerprint(self.target),
                source_checkpoint_sha256=plan['jobs'][self.name]['checkpoint']['sha256']))
            return 0
        result=admit(self.new,{self.name:self.target},self.data,self.cache,['0'],run,
            lambda *args:{'argv':args},{'0':'gpu'})
        self.assertTrue(result['passed'])
        self.assertEqual(result['restores'][self.name]['target_config_sha256'],fingerprint(self.target))

    def test_parent_selects_real_native_target_for_each_backend(self):
        from experiments.baseline_delivery.common import child_spec
        for model,module in [('transformer','worker'),('mega','worker'),('s5','s5_worker')]:
            cfg=self.base/(model+'.json');atomic_json(cfg,{**self.target,'model':model})
            spec=child_spec('--config',cfg,'--migration-check-only')
            self.assertEqual(spec['target'],'experiments.baseline_delivery.'+module)
            self.assertEqual(spec['argv'][spec['argv'].index('-m')+1],spec['target'])
            self.assertIn('--migration-check-only',spec['argv'])


if __name__=='__main__': unittest.main()
