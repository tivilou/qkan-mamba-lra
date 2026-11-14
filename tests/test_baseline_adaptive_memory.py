"""Policy/transaction tests only: stdlib, no model/framework/device execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.baseline_delivery.adaptive_memory import (
    Controller,Exhausted,attach_journal,board_memory,default_policy,evaluation,jax_oom,transaction)


class AllocationOOM(Exception): pass


class AdaptiveMemoryTests(unittest.TestCase):
    def policy(self):
        return {**default_policy(32),'warmup':1,'window':2,'cooldown':2}

    def memory(self):
        return dict(total_bytes=24*1024**3,free_bytes=20*1024**3,peak_bytes=1024**3)

    def feed(self,controller,seconds=10,memory=None):
        for _ in range(3): controller.observe(seconds,self.memory() if memory is None else memory)

    def confirm(self,controller,seconds):
        for _ in range(7): controller.observe(seconds,self.memory())

    def test_headroom_promotes_but_slow_trial_reverts(self):
        c=Controller(self.policy(),32);self.assertEqual(c.micro_batch,4)
        self.feed(c);self.assertEqual(c.micro_batch,8)
        self.confirm(c,11);self.assertEqual(c.micro_batch,4)
        self.assertEqual(c.state['last_event'],'throughput_revert')

    def test_headroom_alone_does_not_claim_speedup(self):
        c=Controller(self.policy(),32);self.feed(c);self.confirm(c,10)
        self.assertEqual(c.micro_batch,8)
        self.assertNotEqual(c.state['last_event'],'throughput_upgrade_accepted')

    def test_fast_trial_is_accepted(self):
        c=Controller(self.policy(),32);self.feed(c);self.confirm(c,8)
        self.assertGreaterEqual(c.micro_batch,8)

    def test_one_percent_gain_is_retained(self):
        c=Controller(self.policy(),32);self.feed(c);self.confirm(c,9.9)
        self.assertEqual(c.micro_batch,8)
        self.assertEqual(c.state['last_event'],'throughput_upgrade_accepted')

    def test_one_slow_window_does_not_revert(self):
        c=Controller(self.policy(),32);self.feed(c);self.feed(c,11)
        self.assertEqual(c.micro_batch,8)
        for _ in range(8): c.observe(8,self.memory())
        self.assertEqual(c.micro_batch,8)
        self.assertEqual(c.state['last_event'],'throughput_upgrade_accepted')

    def test_equal_trial_is_bounded_and_retained_without_gain_claim(self):
        c=Controller(self.policy(),32);self.feed(c)
        for _ in range(17): c.observe(10,self.memory())
        self.assertEqual(c.micro_batch,8)
        self.assertIsNone(c.state['previous'])
        self.assertEqual(c.state['last_event'],'throughput_trial_inconclusive')

    def test_partial_confirmation_windows_restore(self):
        c=Controller(self.policy(),32);self.feed(c);self.feed(c,9.9)
        self.assertEqual(len(c.state['trial_windows']),1)
        restored=Controller(self.policy(),32,json.loads(json.dumps(c.snapshot())))
        for _ in range(4):
            c.observe(9.9,self.memory());restored.observe(9.9,self.memory())
        self.assertEqual(restored.snapshot(),c.snapshot())
        self.assertEqual(c.state['last_event'],'throughput_upgrade_accepted')

    def test_missing_or_low_headroom_blocks_promotion(self):
        for memory in ({},dict(self.memory(),free_bytes=1),dict(self.memory(),peak_bytes=20*1024**3)):
            c=Controller(self.policy(),32);self.feed(c,memory=memory)
            self.assertEqual(c.micro_batch,4)

    def test_cooldown_allows_later_recovery_of_capacity(self):
        c=Controller(self.policy(),32);c.oom();self.assertEqual(c.micro_batch,2)
        self.feed(c);self.assertEqual(c.micro_batch,2)
        self.feed(c);self.assertEqual(c.micro_batch,4)

    def test_saved_policy_restores_pending_trial(self):
        c=Controller(self.policy(),32);self.feed(c)
        restored=Controller(self.policy(),32,json.loads(json.dumps(c.snapshot())))
        self.confirm(c,11);self.confirm(restored,11)
        self.assertEqual(restored.snapshot(),c.snapshot())

    def test_controller_restores_in_a_fresh_stdlib_process(self):
        c=Controller(self.policy(),32);self.feed(c)
        payload=json.dumps(dict(policy=c.policy,saved=c.snapshot()))
        script=('import json,sys; from experiments.baseline_delivery.adaptive_memory import Controller; '
                'v=json.loads(sys.stdin.read()); c=Controller(v["policy"],32,v["saved"]); '
                'print(json.dumps(c.snapshot()))')
        result=subprocess.run([sys.executable,'-B','-c',script],input=payload,
            capture_output=True,text=True,check=True,timeout=10)
        self.assertEqual(json.loads(result.stdout),c.snapshot())

    def test_no_shared_controller_between_workers(self):
        a=Controller(self.policy(),32);b=Controller(self.policy(),32);a.oom()
        self.assertEqual(b.micro_batch,4);self.assertEqual(a.micro_batch,2)

    def test_invalid_policy_rejected(self):
        for change in ({'tiers':[1,3,4]},{'tiers':[4,2,1]},{'initial':7},{'window':0},{'gain':float('nan')}):
            with self.assertRaises(ValueError): Controller({**self.policy(),**change},32)

    def test_partial_optimizer_rng_and_cursor_rollback(self):
        c=Controller(self.policy(),32)
        state=dict(weights=0,moments=0,scheduler=0,rng=7,cursor=0)
        attempts=[];cleanups=[]
        def capture(): return copy.deepcopy(state)
        def restore(snapshot): state.clear();state.update(snapshot)
        def update(micro):
            attempts.append((micro,state.copy()))
            state.update(weights=state['weights']+1,moments=state['moments']+1,
                         scheduler=state['scheduler']+1,rng=state['rng']+1)
            if micro>2: raise AllocationOOM()
            return 3.5
        result,telemetry=transaction(c,update,capture,restore,lambda:cleanups.append(True),
            lambda e:isinstance(e,AllocationOOM),self.memory,lambda:None)
        state['cursor']+=32  # Only the caller commits a successful logical batch.
        self.assertEqual(result,3.5);self.assertEqual(len(attempts),2)
        self.assertEqual(attempts[0][1],attempts[1][1])
        self.assertEqual(state,dict(weights=1,moments=1,scheduler=1,rng=8,cursor=32))
        self.assertEqual(telemetry['used_micro_batch'],2);self.assertEqual(len(cleanups),1)

    def test_async_oom_rolls_back_update(self):
        c=Controller(self.policy(),32);state=[0];sync_count=[0]
        def update(micro): state[0]+=1;return micro
        def synchronize():
            sync_count[0]+=1
            if sync_count[0]==1: raise AllocationOOM()
        result,_=transaction(c,update,lambda:state[:],lambda value:state.__setitem__(slice(None),value),
            lambda:None,lambda e:isinstance(e,AllocationOOM),self.memory,synchronize)
        self.assertEqual(state,[1]);self.assertEqual(result,2)

    def test_exhaustion_has_finite_attempts_and_no_success(self):
        c=Controller(self.policy(),32);attempts=[]
        def fail(micro): attempts.append(micro);raise AllocationOOM()
        with self.assertRaises(Exhausted):
            transaction(c,fail,lambda:None,lambda _:None,lambda:None,
                lambda e:isinstance(e,AllocationOOM),self.memory,lambda:None)
        self.assertEqual(attempts,[4,2,1]);self.assertEqual(c.state['last_event'],'oom_exhausted')

    def test_non_oom_is_not_retried(self):
        c=Controller(self.policy(),32);attempts=[]
        def fail(micro): attempts.append(micro);raise ValueError('data error')
        with self.assertRaises(ValueError):
            transaction(c,fail,lambda:None,lambda _:None,lambda:None,
                lambda e:isinstance(e,AllocationOOM),self.memory,lambda:None)
        self.assertEqual(attempts,[4]);self.assertEqual(c.state['oom_count'],0)

    def test_evaluation_discards_partial_pass(self):
        c=Controller(self.policy(),32);attempts=[]
        def run(micro):
            attempts.append(micro)
            if micro>1: raise AllocationOOM()
            return dict(samples=32)
        result=evaluation(c,run,lambda:None,lambda e:isinstance(e,AllocationOOM))
        self.assertEqual(result,dict(samples=32));self.assertEqual(attempts,[4,2,1])

    def test_journal_and_status_are_compact(self):
        c=Controller(self.policy(),32)
        with tempfile.TemporaryDirectory() as tmp:
            attach_journal(c,tmp,lambda:12);c.oom()
            records=[json.loads(line) for line in (Path(tmp)/'adaptive-events.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),2);self.assertEqual(records[-1]['from_micro_batch'],4)
            self.assertEqual(records[-1]['to_micro_batch'],2)
            self.assertEqual(json.loads((Path(tmp)/'adaptive-memory.json').read_text())['step'],12)

    def test_jax_oom_classification_is_specific(self):
        XlaRuntimeError=type('XlaRuntimeError',(Exception,),{'__module__':'jaxlib.xla_extension'})
        self.assertTrue(jax_oom(XlaRuntimeError('RESOURCE_EXHAUSTED: Out of memory allocating buffer')))
        self.assertFalse(jax_oom(RuntimeError('RESOURCE_EXHAUSTED: Out of memory')))
        self.assertFalse(jax_oom(XlaRuntimeError('INVALID_ARGUMENT: bad dimensions')))

    def test_board_query_uses_physical_mapping_and_timeout(self):
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'2'}),patch(
                'subprocess.run',return_value=subprocess.CompletedProcess([],0,'1024, 24576\n')) as run:
            self.assertEqual(board_memory()['free_bytes'],1024**3)
            self.assertIn('--id=2',run.call_args.args[0]);self.assertEqual(run.call_args.kwargs['timeout'],2)
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'2'}),patch(
                'subprocess.run',side_effect=subprocess.TimeoutExpired('nvidia-smi',2)):
            self.assertEqual(board_memory(),{})


if __name__=='__main__': unittest.main()
