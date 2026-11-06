"""Joint batch and sample schedule checks, no framework imports/devices."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.baseline_delivery.logical_batch import JointController,SampleClock,TorchSampleSchedule,policy,canary_rows,canary_tier
from experiments.baseline_delivery.adaptive_memory import transaction
from experiments.baseline_delivery.common import atomic_json,fingerprint,read
from experiments.baseline_delivery.migration import ADAPTIVE_REVISION,compatible,load_state


class LogicalBatchTests(unittest.TestCase):
    def make(self):
        c=JointController({**policy(32),'warmup':1,'window':2,'cooldown':2},32)
        c.state['index']=c.policy['tiers'].index(32)
        return c
    def memory(self): return dict(total_bytes=24*1024**3,free_bytes=20*1024**3,peak_bytes=1024**3)
    def feed(self,c,count,seconds,n):
        for _ in range(n): c.used_samples=count;c.observe(seconds,self.memory())

    def test_tiers_expand_only_at_batch_boundary(self):
        c=self.make();self.feed(c,32,1,3)
        self.assertEqual((c.logical_batch,c.micro_batch),(64,64))
        self.assertEqual(c.state['previous_logical'],0)
    def test_equal_step_rate_is_double_sample_throughput(self):
        c=self.make();self.feed(c,32,1,3);self.feed(c,64,1,7)
        self.assertEqual(c.logical_batch,64)
        self.assertEqual(c.state['last_event'],'throughput_upgrade_accepted')
    def test_slower_samples_revert_logical_and_physical(self):
        c=self.make();self.feed(c,32,1,3);self.feed(c,64,3,7)
        self.assertEqual((c.logical_batch,c.micro_batch),(32,32))
    def test_small_positive_gain_retained(self):
        c=self.make();self.feed(c,32,1,3);self.feed(c,64,1.98,7)
        self.assertEqual(c.state['last_event'],'throughput_upgrade_accepted')
    def test_one_slow_window_no_revert(self):
        c=self.make();self.feed(c,32,1,3);self.feed(c,64,3,3)
        self.assertEqual(c.logical_batch,64)
    def test_oom_keeps_failed_logical_rows(self):
        c=self.make();self.feed(c,32,1,3);c.oom()
        self.assertEqual((c.logical_batch,c.micro_batch),(64,32))
    def test_epoch_tail_never_promotes_from_small_remainder(self):
        c=self.make();self.feed(c,3,.01,20)
        self.assertEqual((c.logical_batch,c.micro_batch),(32,32))
    def test_no_telemetry_no_speculation(self):
        c=self.make()
        for _ in range(5): c.used_samples=32;c.observe(1,{})
        self.assertEqual(c.logical_batch,32)
    def test_logical_growth_can_reuse_same_physical_slice(self):
        c=self.make();memory=dict(self.memory(),peak_bytes=12*1024**3,free_bytes=12*1024**3)
        for _ in range(3): c.used_samples=32;c.observe(1,memory)
        self.assertEqual((c.logical_batch,c.micro_batch),(64,32))
        self.assertEqual(c.state['last_event'],'logical_upgrade_accumulation_trial')
    def test_controller_partial_confirmation_restores(self):
        c=self.make();self.feed(c,32,1,3);self.feed(c,64,1,3)
        d=JointController(c.policy,32,json.loads(json.dumps(c.snapshot())))
        self.feed(c,64,1,4);self.feed(d,64,1,4)
        self.assertEqual(c.snapshot(),d.snapshot())
    def test_logical_growth_not_blocked_by_smaller_physical_slice(self):
        c=self.make();c.state.update(logical_index=1,warmup=0)
        memory=dict(self.memory(),peak_bytes=12*1024**3,free_bytes=12*1024**3)
        for _ in range(2): c.used_samples=64;c.observe(1,memory)
        self.assertEqual((c.logical_batch,c.micro_batch),(128,32))
    def test_maximum_is_finite(self):
        c=self.make();c.state.update(logical_index=2,index=len(c.policy['tiers'])-1)
        self.feed(c,128,1,20);self.assertEqual((c.logical_batch,c.micro_batch),(128,128))
    def test_transaction_rolls_back_clock_and_optimizer(self):
        c=self.make();self.feed(c,32,1,3)
        state={'rows':0,'adam_updates':0};clock=SampleClock(1000,0,[.001])
        scheduler=TorchSampleSchedule(SimpleNamespace(param_groups=[{'lr':.001}]),clock)
        def capture(): return copy.deepcopy(state),scheduler.state_dict()
        def restore(v): state.update(v[0]);scheduler.load_state_dict(v[1])
        class OOM(Exception): pass
        def update(micro):
            state['adam_updates']+=1;scheduler.step(64)
            if micro>32: raise OOM()
            state['rows']+=64;return 64
        value,meta=transaction(c,update,capture,restore,lambda:None,lambda e:isinstance(e,OOM),
            self.memory,lambda:None,samples=64)
        self.assertEqual(value,64);self.assertEqual(state,{'rows':64,'adam_updates':1})
        self.assertEqual(clock.state['progress'],64);self.assertEqual(meta['used_logical_batch'],64)
        self.assertEqual(meta['logical_batch'],32)
        self.assertEqual(c.state['last_event'],'logical_oom_cooldown')
    def test_exhausted_oom_still_publishes(self):
        from experiments.baseline_delivery.adaptive_memory import Exhausted
        c=self.make();c.state['index']=0;events=[];c.on_event=events.append
        with self.assertRaises(Exhausted): c.oom()
        self.assertEqual(events[-1]['event'],'oom_exhausted')
    def test_event_names_both_source_and_destination(self):
        c=self.make();events=[];c.on_event=events.append
        self.feed(c,32,1,3);c.publish()
        self.assertEqual((events[-1]['from_logical_batch'],events[-1]['to_logical_batch']),(32,64))
    def test_samples_are_neither_skipped_nor_repeated_across_tiers(self):
        c=self.make();cursor=0;seen=[]
        while cursor<357:
            end=min(cursor+c.logical_batch,357);seen+=list(range(cursor,end))
            c.used_samples=end-cursor;c.observe(.01,self.memory());cursor=end
        self.assertEqual(seen,list(range(357)))


class SampleClockTests(unittest.TestCase):
    def test_same_samples_same_lr_regardless_batch_partition(self):
        a=SampleClock(10000,320,[.001]);b=SampleClock(10000,320,[.001])
        for _ in range(4): a.advance(32)
        b.advance(128);self.assertEqual(a.rates(),b.rates())
    def test_migration_anchor_preserves_current_lr(self):
        c=SampleClock(10000,320,[.001],progress=2000,anchor_rates=[.0007])
        self.assertEqual(c.rates(),[.0007]);c.advance(64)
        self.assertLess(c.rates()[0],.0007)
    def test_warmup_anchor_no_restart(self):
        c=SampleClock(10000,320,[.001],progress=160,anchor_rates=[.0005])
        self.assertEqual(c.rates(),[.0005]);c.advance(160);self.assertAlmostEqual(c.rates()[0],.001)
    def test_resume_preserves_exact_anchor_and_progress(self):
        c=SampleClock(1000,32,[.001],128,[.0008]);c.advance(64)
        d=SampleClock(1000,32,[.001],saved=json.loads(json.dumps(c.snapshot())))
        self.assertEqual(c.rates(),d.rates());self.assertEqual(c.snapshot(),d.snapshot())
    def test_parameter_group_ratios_preserved(self):
        c=SampleClock(1000,32,[.001,.0001]);c.advance(64)
        self.assertAlmostEqual(c.rates()[0]/c.rates()[1],10)
    def test_fixed_sample_budget_and_final_partial_batch(self):
        c=SampleClock(97,0,[.001]);c.advance(64);c.advance(33)
        self.assertEqual(c.rates(),[0.])
        with self.assertRaises(ValueError): c.advance(1)
    def test_changed_schedule_identity_rejected(self):
        c=SampleClock(1000,32,[.001])
        with self.assertRaises(ValueError): SampleClock(2000,32,[.001],saved=c.snapshot())
    def test_optimizer_counter_not_rewritten_by_clock(self):
        opt=SimpleNamespace(param_groups=[{'lr':.001}],actual_adam_count=127)
        s=TorchSampleSchedule(opt,SampleClock(1000,0,[.001]));s.step(128)
        self.assertEqual(opt.actual_adam_count,127)
    def test_invalid_lr_anchor_rejected(self):
        for rates in ([float('nan')],[-1.],[],[.001,.002]):
            with self.assertRaises(ValueError): SampleClock(1000,0,[.001],32,rates)
    def test_nondefault_reference_batch_warmup(self):
        c=SampleClock(1000,160,[.001],reference_batch=16)
        self.assertAlmostEqual(c.rates()[0],.0001)
        with self.assertRaises(ValueError): SampleClock(1000,160,[.001],saved=c.snapshot())


class NativeCanaryPolicyTests(unittest.TestCase):
    def test_real_formal_jobs_cannot_force_tiers(self):
        config=dict(batch_size=32,epochs=2,adaptive_memory=policy(32),canary_logical_tiers=[32,64,128])
        self.assertEqual(canary_rows(config,True),224)
        with self.assertRaises(ValueError): canary_rows(config,False)
    def test_bounded_canary_exercises_all_tiers_after_restore(self):
        config=dict(batch_size=32,epochs=2,adaptive_memory=policy(32),canary_logical_tiers=[32,64,128])
        c=JointController(config['adaptive_memory'],32);sizes=[]
        for step in range(6):
            if step==2: c=JointController(c.policy,32,json.loads(json.dumps(c.snapshot())))
            canary_tier(c,config,step);sizes.append(c.logical_batch)
        self.assertEqual(sizes,[32,64,128]*2)
    def test_no_canary_flag_preserves_formal_controller(self):
        c=JointController(policy(32),32);before=c.snapshot()
        canary_tier(c,{'batch_size':32},4)
        self.assertEqual(c.snapshot(),before)


class LogicalDashboardTests(unittest.TestCase):
    def test_progress_uses_samples_not_legacy_optimizer_total(self):
        from scripts.formal_progress import Dashboard
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);suite=root/'runs/example';job=suite/'jobs/transformer-listops-s42';job.mkdir(parents=True)
            name=job.name
            atomic_json(suite/'suite.json',dict(jobs=[dict(id=name)]))
            atomic_json(suite/'status.json',{name:dict(state='running',started=10,pid=99999)})
            atomic_json(job/'config.json',dict(task='listops',epochs=100,batch_size=32))
            hb=dict(epoch=10,optimizer_step=121584,samples_seen=4000,sample_budget=10000,
                loss=1.2,updated_at=20,adaptive_memory=dict(used_micro_batch=32,micro_batch=64,
                used_logical_batch=64,logical_batch=128,oom_retries=0,event='headroom_upgrade_trial'))
            atomic_json(job/'heartbeat.json',hb)
            dashboard=Dashboard(root,suite)
            with patch('scripts.formal_progress.gpu_usage',side_effect=AssertionError('Device query forbidden')):
                first=dashboard.render(20)
                atomic_json(job/'heartbeat.json',{**hb,'samples_seen':4256,'updated_at':22})
                second=dashboard.render(22)
            self.assertIn('样本 4000/10000 (40.00%)',first)
            self.assertIn('128.00 samples/s',second)
            self.assertIn('逻辑 batch：本步 64 | 下一步 128',second)
            self.assertNotIn('step/s',second)


class LogicalMigrationTests(unittest.TestCase):
    def test_current_adaptive_source_supported_and_config_locked(self):
        old=dict(source_revision=ADAPTIVE_REVISION,batch_size=32,lr=.001,adaptive_memory={'old':True})
        new={**old,'source_revision':'new','adaptive_memory':policy(32)}
        compatible(old,new)
        with self.assertRaises(ValueError): compatible(old,{**new,'lr':.002})
    def test_current_migration_preserves_lineage_and_resets_timing_controller(self):
        old=dict(source_revision=ADAPTIVE_REVISION,batch_size=32,lr=.001,adaptive_memory={'old':True})
        new={**old,'source_revision':'new','adaptive_memory':policy(32)}
        with tempfile.TemporaryDirectory() as temp:
            suite=Path(temp);folder=suite/'migration-source/job';folder.mkdir(parents=True)
            atomic_json(folder/'config.json',old)
            record=dict(config_sha256=fingerprint(old),checkpoint=dict(file='state',sha256='hash',contract='old',step=121584))
            atomic_json(suite/'migration-source/plan.json',dict(source_revision=ADAPTIVE_REVISION,jobs={'job':record},trajectory='old'))
            original=dict(contract='old',step=121584,cursor=50688,adaptive_memory={'index':5},migration={'source':'original'})
            with patch('experiments.baseline_delivery.migration.checked_pointer',return_value=(record['checkpoint'],folder/'state')):
                result=load_state(suite/'jobs/job',new,'new',lambda _: 'old',lambda *_:copy.deepcopy(original))
            self.assertEqual(result['step'],121584);self.assertEqual(result['cursor'],50688)
            self.assertEqual(result['migration']['previous_migration'],{'source':'original'})
            self.assertNotIn('adaptive_memory',result)


if __name__=='__main__': unittest.main()
