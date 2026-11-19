"""Control tests with deterministic fake updates, never native frameworks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from contextlib import contextmanager
from experiments.baseline_delivery import key_bias_benchmark as b
from experiments.baseline_delivery import bootstrap,return_artifacts
from experiments.baseline_delivery.common import atomic_json
from tests.test_baseline_key_bias_recovery import Runtime


@contextmanager
def profile(**kwargs):
    yield NS(key_averages=lambda:[NS(key='_scaled_dot_product_efficient_attention'),
        NS(key='_scaled_dot_product_efficient_attention_backward')])


class BenchRuntime(Runtime):
    def __init__(self,fail_at=None):
        super().__init__();self.events=[];self.fail_at=fail_at
        self.t.cuda=NS(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,
            memory_allocated=lambda:10,max_memory_allocated=lambda:20,max_memory_reserved=lambda:30)
        self.t.profiler=NS(profile=profile,ProfilerActivity=NS(CPU=1,CUDA=2))
    def update(self,enabled,**kwargs):
        self.events.append((enabled,kwargs.get('dropout_off',False)))
        if len(self.events)==self.fail_at: raise RuntimeError('controlled failure')
        for p in self.parameters.values():
            for _ in range(8): p.emit(__import__('tests.test_baseline_key_bias_consistency',fromlist=['Tensor']).Tensor(range(384)))
        self.progress['cursor']+=128;self.progress['step']+=1
        return dict(loss=1.,row_ids_sha256=str(self.progress['cursor']),rows=128,correct=64)


class Tests(unittest.TestCase):
    def test_complete_order_gate_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=BenchRuntime()
            # Each baseline takes2s, each BF161s; all4paired gains positive.
            durations=[2 if n=='fp32' else 1 for n in b.ORDER];ticks=[];now=0
            for d in durations: ticks.extend([now,now+d]);now+=d+1
            with patch.object(b.time,'perf_counter',side_effect=ticks): result=b.benchmark(rt,Path(tmp))
            self.assertTrue(result['passed']);self.assertEqual(rt.snapshot(),rt.base)
            self.assertEqual([v['route'] for v in result['timing_records']],list(b.ORDER))
            self.assertEqual(result['timing']['throughput_gain'],1)
            self.assertEqual(len(rt.events),26)
            self.assertTrue(all(drop for _,drop in rt.events[:12]))
            self.assertTrue(all(not drop for _,drop in rt.events[12:]))
            self.assertTrue(all(not p._backward_hooks for p in rt.parameters.values()))
    def test_trajectory_failure_stops_before_profile_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=BenchRuntime()
            with patch.object(b,'trajectory_gate',return_value=dict(passed=False)):
                result=b.benchmark(rt,Path(tmp))
            self.assertFalse(result['passed']);self.assertEqual(len(rt.events),12)
            self.assertEqual(rt.snapshot(),rt.base)
    def test_failure_mid_candidate_cleans_hooks_and_writes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=BenchRuntime(fail_at=10)
            with self.assertRaisesRegex(RuntimeError,'controlled'): b.benchmark(rt,Path(tmp))
            self.assertEqual(rt.snapshot(),rt.base)
            self.assertTrue(all(not p._backward_hooks for p in rt.parameters.values()))
            self.assertTrue(json.loads((Path(tmp)/'benchmark.json').read_text())['rollback']['full_state_exact'])
    def test_loss_gate_fixed_boundary_nonfinite_and_row_identity(self):
        values={n:[dict(loss=1.,row_ids_sha256=str(i)) for i in range(4)] for n in b.ROUTES}
        self.assertTrue(b.trajectory_gate(values)['passed'])
        values['bf16_analytic'][1]['loss']=1.101
        self.assertFalse(b.trajectory_gate(values)['passed'])
        values['bf16_analytic'][1]['loss']=float('nan')
        with self.assertRaises(ValueError): b.trajectory_gate(values)
        values['bf16_analytic'][1]=dict(loss=1.,row_ids_sha256='changed')
        with self.assertRaises(ValueError): b.trajectory_gate(values)
    def test_kernel_gate_rejects_missing_backward_and_fallback(self):
        f='_scaled_dot_product_efficient_attention';back=f+'_backward'
        self.assertTrue(b.kernel_gate([f,back])['efficient_backward'])
        for names in ([f],[back],[f,back,'_scaled_dot_product_flash_attention']):
            with self.assertRaises(ValueError): b.kernel_gate(names)
    def test_preflight_hash_rejects_before_screen_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'key-bias-recovery.json';p.write_text('{}')
            with self.assertRaises(ValueError): b.bind_preflight(Path(tmp),{})
    def test_unknown_route_rejected(self):
        with self.assertRaises(ValueError): b.update(None,'unreviewed')
    def test_failed_child_returns_partial_and_failed_upload_preserves_report(self):
        import sys
        for child_failure,upload_failure in ((True,False),(False,True)):
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);seen=[]
                def execute(args,out,report):
                    report['partial_benchmark']={'stage':'profile'}
                    if child_failure: raise RuntimeError('controlled child failure')
                    report.update(passed=True,status='passed')
                def upload(out,kind,files):
                    self.assertEqual(kind,'key-bias-benchmark')
                    seen.append(json.loads((out/files[0]).read_text()))
                    if upload_failure: raise RuntimeError('controlled upload failure')
                argv=['entry','--source-suite','source','--cache','cache','--screen-from','screen','--preflight-from','preflight']
                with patch.object(b,'ROOT',folder),patch.object(sys,'argv',argv),patch.object(b,'execute',side_effect=execute),\
                        patch.object(b,'revision',return_value='revision'),patch.object(bootstrap,'check_exchange'),\
                        patch.object(return_artifacts,'upload_declared',side_effect=upload):
                    with self.assertRaises(RuntimeError): b.main()
                self.assertEqual(len(seen),1);self.assertFalse(seen[0]['candidate_admitted'])
                self.assertEqual(seen[0]['partial_benchmark']['stage'],'profile')
                self.assertEqual(len(list((folder/'runs').glob('*/key-bias-benchmark.json'))),1)
    def test_nonpositive_timing_gain_rejects_without_formal_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=BenchRuntime();ticks=list(range(16))
            with patch.object(b.time,'perf_counter',side_effect=ticks): result=b.benchmark(rt,Path(tmp))
            self.assertFalse(result['passed']);self.assertEqual(result['rollback']['committed_updates'],0)


if __name__=='__main__': unittest.main()
