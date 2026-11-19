"""Data order, staging lifetime and bounded timing; no framework imports."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery.performance import Lookahead,PinnedStaging,Profile,ProfileReturn,policy
from experiments.baseline_delivery.migration import compatible,DYNAMIC_REVISION,load_state
from experiments.baseline_delivery.logical_batch import policy as logical_policy
from experiments.baseline_delivery.common import atomic_json,fingerprint


class PerformanceTests(unittest.TestCase):
    def test_lookahead_prefix_survives_tier_changes_and_tail(self):
        def load(ids): return dict(labels=list(ids),inputs=[v*3 for v in ids])
        with Lookahead(load) as p:
            self.assertEqual(p.get([0,1]),load([0,1]))
            for ids,wanted in ((range(2,130),range(2,66)),(range(66,194),range(66,194)),(range(194,197),range(194,197))):
                p.submit(ids)
                self.assertEqual(p.get(wanted),load(wanted))
    def test_lookahead_never_accepts_shifted_sampler(self):
        with Lookahead(lambda ids:dict(labels=ids)) as p:
            p.submit([1,2,3])
            with self.assertRaises(ValueError): p.get([2,3])
    def test_profile_uses_events_only_until_bound(self):
        event=lambda:Mock(elapsed_time=Mock(return_value=3.))
        p=Profile(limit=2,event=event)
        for _ in range(5):
            p.begin()
            with p.phase('forward'): pass
            p.finish()
        r=p.report()
        self.assertEqual(r['sampled_updates'],2)
        self.assertEqual(r['metrics']['forward_stream_ms']['mean'],3.)
        self.assertEqual(r['metrics']['update_wall_ms']['count'],2)
    def test_oom_events_not_read(self):
        event=Mock(side_effect=lambda:Mock(elapsed_time=Mock(side_effect=AssertionError)))
        p=Profile(limit=1,event=event);p.begin()
        with p.phase('failed'): pass
        self.assertTrue(p.finish(retries=1));self.assertNotIn('failed_stream_ms',p.report()['metrics'])
    def test_pinning_failure_falls_back_all_arrays(self):
        runtime=Mock();runtime.as_tensor.side_effect=lambda x:x
        class Source(list):
            shape=(2,);dtype='int'
        runtime.empty.side_effect=RuntimeError('not supported')
        p=PinnedStaging(runtime,128);a=dict(labels=Source([1,2]))
        self.assertEqual(p.batch(a),a);self.assertFalse(p.enabled)
        self.assertEqual(p.batch(a),a);self.assertEqual(runtime.empty.call_count,1)
    def test_reusable_host_storage_copies_only_live_prefix(self):
        class Tensor:
            dtype='int'
            def __init__(self,n): self.shape=(n,);self.n=n;self.copies=[]
            def __len__(self): return self.n
            def __getitem__(self,k): return self
            def copy_(self,source): self.copies.append(source)
        runtime=Mock();runtime.as_tensor.side_effect=lambda x:x;runtime.empty.side_effect=lambda shape,**_:Tensor(shape[0])
        p=PinnedStaging(runtime,128);x=Tensor(32);y=Tensor(64)
        a=p.batch(dict(labels=x));b=p.batch(dict(labels=y))
        self.assertIs(a['labels'],b['labels']);self.assertEqual(runtime.empty.call_count,1)
        self.assertEqual(b['labels'].copies,[x,y])
    def test_migration_locks_batch_lr_and_optimizer_config(self):
        old=dict(source_revision=DYNAMIC_REVISION,batch_size=32,lr=.001,adaptive_memory=logical_policy(32))
        new=dict(old,source_revision='next',performance=policy())
        compatible(old,new)
        for change in (dict(lr=.1),dict(batch_size=64),dict(performance={}),dict(adaptive_memory={})):
            with self.assertRaises(ValueError): compatible(old,dict(new,**change))
    def test_dynamic_migration_preserves_clock_and_controller(self):
        old=dict(source_revision=DYNAMIC_REVISION,batch_size=32,adaptive_memory=logical_policy(32))
        new=dict(old,source_revision='next',performance=policy())
        pointer=dict(file='state',sha256='hash',contract='old',step=99)
        plan=dict(source_revision=DYNAMIC_REVISION,trajectory='continuation',jobs={'job':dict(config_sha256=fingerprint(old),checkpoint=pointer)})
        state=dict(contract='old',step=99,sample_clock={'progress':531776},adaptive_memory={'oom_count':1,'index':6},migration={'prefix':'original'},optimizer={'counter':99})
        with patch('experiments.baseline_delivery.migration.plan_for',return_value=plan),patch('experiments.baseline_delivery.migration.read',return_value=old),patch('experiments.baseline_delivery.migration.checked_pointer',return_value=(pointer,None)):
            out=load_state(Path('suite/jobs/job'),new,'new',lambda _: 'old',lambda *_:copy.deepcopy(state))
            for key in ('sample_clock','adaptive_memory','optimizer'): self.assertEqual(out[key],state[key])
            self.assertEqual(out['migration']['previous_migration'],state['migration'])
            del state['sample_clock']
            with self.assertRaises(ValueError): load_state(Path('suite/jobs/job'),new,'new',lambda _: 'old',lambda *_:copy.deepcopy(state))
    def test_profile_return_is_one_process_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'jobs/job';folder.mkdir(parents=True)
            atomic_json(folder/'performance-profile.json',{})
            child=Mock();child.poll.return_value=None
            with patch('experiments.baseline_delivery.performance.subprocess.Popen',return_value=child) as popen:
                p=ProfileReturn(root);p.tick(root,[{'id':'job'}]);p.tick(root,[{'id':'job'}])
                self.assertEqual(popen.call_count,1)
                self.assertIn(root.name+'--job',popen.call_args.args[0])
                child.poll.return_value=0;p.tick(root,[{'id':'job'}]);self.assertEqual(popen.call_count,1)
            with patch('experiments.baseline_delivery.performance.subprocess.Popen',side_effect=OSError):
                p=ProfileReturn(root);p.tick(root,[{'id':'job'}]);self.assertIsNone(p.child)


if __name__=='__main__': unittest.main()
