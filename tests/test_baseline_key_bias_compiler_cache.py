"""Partition diagnostic cache lifetime. No native framework imports."""
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from tests.test_baseline_key_bias_stability import Runtime
from experiments.baseline_delivery import key_bias_stability_native as n,torch_acceleration as a
from experiments.baseline_delivery.common import read


class CacheGraph:
    def __init__(self): self.resident=False;self.builds=0;self.variants=0;self.resets=[];self.mode=None
    def set_mode(self,mode): self.mode=mode
    def clear(self): self.resident=False
    def generation(self): return self.builds
    def status(self): return dict(graph_resident=self.resident)
    def backend_error(self,error): return False
    def reset(self):
        assert not self.resident
        self.resets.append(self.builds);self.variants=0
    def forward(self,batch,mode):
        if mode=='compile_graph' and not self.resident:
            self.builds+=1;self.variants+=4
            if self.variants>8:
                raise a.BackendFailure('FailOnRecompileLimitHit',a.backend_diagnostic(RuntimeError('recompile_limit'),'parallel_graph_capture','eager_partitioned'))
            self.resident=True


def fixture():
    rt=Runtime();graph=CacheGraph();rt.t.compiler=NS(reset=graph.reset)
    rt.np=NS(ndarray=type('Unused',(),{}));rt.opt=NS(zero_grad=lambda **k:None)
    rt.t.is_tensor=lambda x:False;rt.t.cuda.OutOfMemoryError=MemoryError
    old=rt.update
    def update(enabled,**kwargs):
        if kwargs.get('executor'): kwargs['executor'].forward(None)
        return old(enabled,**kwargs)
    rt.update=update
    return rt,graph


class Tests(unittest.TestCase):
    def test_full_abba_cache_reset_after_release_before_each_capture_and_outside_timing(self):
        rt,graph=fixture();ticks=[];clock=0
        for name in ('incumbent','candidate','candidate','incumbent')*2:
            ticks.extend([clock,clock+(2 if name=='incumbent' else 1)]);clock+=3
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'torch_runtime',return_value=graph),patch.object(n.time,'perf_counter',side_effect=ticks):
            report=n.comparison(rt,Path(tmp),'partition32')
            self.assertTrue(report['passed']);self.assertEqual(graph.resets,[0,1,2]);self.assertEqual(graph.builds,3)
            self.assertEqual([v['before_timing_index'] for v in report['compiler_cache_resets']],[0,3,7])
            self.assertEqual(len(report['timing_records']),8);self.assertEqual(rt.snapshot(),rt.base);self.assertFalse(graph.resident)
            self.assertTrue(all(v['outside_timing'] and v['full_state_unchanged'] and not v['graph_resident'] for v in report['compiler_cache_resets']))
            proofs=[v for v in report['timing_records'] if v['route']=='incumbent']
            self.assertEqual([v['steady_before']['generation'] for v in proofs],[1,2,2,3])
            self.assertTrue(all(v['steady_before']==v['steady_after'] for v in proofs))
    def test_original_cache_accumulation_rejects_third_build_with_partial_retained(self):
        rt,graph=fixture();rt.t.compiler.reset=lambda:None
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'torch_runtime',return_value=graph):
            with self.assertRaises(a.AccelerationRequired): n.comparison(rt,Path(tmp),'partition32')
            report=read(Path(tmp)/'comparison.json')
            self.assertEqual(len(report['timing_records']),7)
            self.assertEqual(report['failure']['executor_diagnostics'][0]['error']['exceptions'][0]['category'],'recompile_budget')
            self.assertEqual(rt.snapshot(),rt.base);self.assertFalse(graph.resident)
    def test_reset_rejects_live_graph_and_changed_complete_state(self):
        rt,graph=fixture();graph.resident=True
        with self.assertRaisesRegex(ValueError,'Release'): n.reset_partition_compiler(rt,graph,{},0)
        self.assertEqual(graph.resets,[])
        graph.resident=False;rt.t.compiler.reset=lambda:rt.state['rng'].update(cuda=99)
        with self.assertRaisesRegex(ValueError,'training/RNG'): n.reset_partition_compiler(rt,graph,{},0)
    def test_reset_failure_is_returned_and_full_state_restored(self):
        rt,graph=fixture()
        def fail(): rt.state['rng']['cuda']=99;raise RuntimeError('fixture reset failed')
        rt.t.compiler.reset=fail
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'torch_runtime',return_value=graph):
            with self.assertRaisesRegex(RuntimeError,'fixture reset'): n.comparison(rt,Path(tmp),'partition32')
            report=read(Path(tmp)/'comparison.json')
            self.assertEqual(report['failure']['completed_timing_records'],0)
            self.assertEqual(report['rollback'],dict(full_state_exact=True,committed_samples=0,committed_updates=0))
            self.assertEqual(rt.snapshot(),rt.base)


if __name__=='__main__': unittest.main()
