"""MEGA FFT handoff tests: stdlib only, no framework/device execution."""
import cmath
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from experiments.baseline_delivery import mega_fft as plugin,mega_diagnostic as diagnostic
from experiments.baseline_delivery.common import atomic_json,digest,read


class Tests(unittest.TestCase):
    def test_power_two_lengths_and_invalid_inputs(self):
        for n,wanted in ((1,1),(8190,8192),(4094,4096),(15998,16384),(8192,8192)):
            self.assertEqual(plugin.fft_length(n),wanted)
        for n in (0,-1,True,1.5):
            with self.assertRaises(ValueError):plugin.fft_length(n)

    def test_zero_padding_preserves_convolution_slice_and_linear_derivative(self):
        def convolution(x,k,n):
            fft=lambda a:[sum(v*cmath.exp(-2j*math.pi*j*i/n) for i,v in enumerate(a)) for j in range(n)]
            a,b=fft(x),fft(k)
            return [sum(a[j]*b[j]*cmath.exp(2j*math.pi*j*i/n) for j in range(n)).real/n for i in range(n)]
        for length in (2,3,5):
            k=[.3*i-.2 for i in range(2*length-1)];x=[0.]*(length-1)+[.2+i for i in range(length)]
            original=2*(2*length-1);padded=plugin.fft_length(original);start=2*length-2
            for values in (x,[float(i==length) for i in range(len(x))]):
                left=convolution(values,k,original)[start:start+length]
                right=convolution(values,k,padded)[start:start+length]
                direct=[sum(values[j]*k[i-j] for j in range(len(values)) if 0<=i-j<len(k)) for i in range(start,start+length)]
                for a,b,c in zip(left,right,direct):
                    self.assertAlmostEqual(a,b,places=10);self.assertAlmostEqual(b,c,places=10)

    def test_instance_toggle_shared_class_unchanged_and_close(self):
        source='''class MultiHeadEMA:
    def forward(self, fft_len):
        a = self.compute(n=2 * fft_len)
        b = self.compute(n=2 * fft_len)
        c = self.compute(n=2 * fft_len)
        return a, b, c
    def compute(self, n): return n
'''
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fake_ema.py';path.write_text(source)
            spec=importlib.util.spec_from_file_location('fixture_ema',path);module=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=module;self.addCleanup(sys.modules.pop,spec.name,None);spec.loader.exec_module(module)
            first,second=module.MultiHeadEMA(),module.MultiHeadEMA();original=module.MultiHeadEMA.forward
            class Model:
                def modules(self):return [first]
            with patch.object(plugin,'SOURCE_SHA',digest(path)):
                candidate=plugin.PowerTwoEMA(Model())
            self.assertEqual(first.forward(4095),(8190,)*3)
            with candidate.activate():
                self.assertEqual(first.forward(4095),(8192,)*3)
                self.assertEqual(second.forward(4095),(8190,)*3)
            self.assertEqual(first.forward(4095),(8190,)*3)
            self.assertIs(module.MultiHeadEMA.forward,original)
            self.assertEqual(candidate.calls,3);candidate.close()
            self.assertNotIn('forward',first.__dict__)

    def test_snapshot_preserves_original_bytes_and_requires_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original';suite=root/'diagnostic';job='mega-listops-s42';folder=source/'jobs'/job
            cache=root/'cache';cache.mkdir();data=root/'data';data.mkdir()
            atomic_json(source/'status.json',{job:dict(state='paused')})
            atomic_json(source/'suite.json',dict(data=str(data)))
            atomic_json(folder/'config.json',dict(model='mega',task='listops',adaptive_memory=dict(enabled=True)))
            checkpoint=folder/'checkpoints/step-10.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'fixture checkpoint')
            pointer=dict(schema='qkan-training-checkpoint.v1',file=checkpoint.name,sha256=digest(checkpoint),step=10,contract='bound')
            atomic_json(folder/'checkpoints/latest.json',pointer)
            originals={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            with patch.object(diagnostic,'revision',return_value='pinned'):
                request=diagnostic.snapshot(source,suite,job,cache)
            self.assertEqual(request['checkpoint'],pointer)
            self.assertTrue(all(Path(p).read_bytes()==b for p,b in originals.items()))
            self.assertEqual((suite/'source/checkpoints'/checkpoint.name).read_bytes(),checkpoint.read_bytes())
            atomic_json(source/'status.json',{job:dict(state='running')})
            with self.assertRaisesRegex(ValueError,'Pause'):diagnostic.snapshot(source,root/'second',job,cache)

    def test_child_targets_existing_mega_env_and_owned_output(self):
        suite=Path('/fixture/runs/mega-fft');spec=diagnostic.child_spec(suite);argv=spec['argv']
        self.assertEqual(Path(argv[0]).parts[-3:],('.venv-mega','bin','python'))
        self.assertEqual(argv[argv.index('-m')+1],spec['target'])
        self.assertEqual(argv.count('--output'),1)
        self.assertEqual(Path(argv[argv.index('--output')+1]),suite/'child')
        self.assertEqual(len(diagnostic.ORDER),8);self.assertEqual(sum(diagnostic.ORDER),4)

    def test_failed_child_returns_report_removes_copy_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original';folder=source/'jobs/mega-listops-s42'
            cache=root/'cache';cache.mkdir();data=root/'data';data.mkdir()
            atomic_json(source/'status.json',{'mega-listops-s42':dict(state='paused')})
            atomic_json(source/'suite.json',dict(data=str(data)))
            atomic_json(folder/'config.json',dict(model='mega',task='listops',adaptive_memory=dict(enabled=True)))
            checkpoint=folder/'checkpoints/step-10.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'original checkpoint')
            atomic_json(folder/'checkpoints/latest.json',dict(schema='qkan-training-checkpoint.v1',file=checkpoint.name,
                sha256=digest(checkpoint),step=10,contract='bound'))
            before=digest(checkpoint);returned=[]
            class FailedChild:
                def __init__(self,*args):self.records=[]
                def run(self,spec,log,env):
                    self.records.append(dict(termination=dict(group_verified=True)))
                    return 77
            args=SimpleNamespace(source_suite=source,cache=cache,gpu='0',job='mega-listops-s42')
            with patch.object(diagnostic,'ROOT',root),patch.object(diagnostic,'revision',return_value='pinned'),\
                    patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                    patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})),\
                    patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)),\
                    patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',FailedChild),\
                    patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:returned.append(a)):
                self.assertEqual(diagnostic.campaign(args),76)
            suite=returned[0][0];report=read(suite/'mega-fft.json')
            self.assertEqual(returned[0][1: ],('mega-fft',['mega-fft.json']))
            self.assertFalse(report['measurement_completed']);self.assertFalse(report['formal_release_allowed'])
            self.assertTrue(report['source_unchanged']);self.assertTrue(report['copied_checkpoint_removed'])
            self.assertIn('fft',report['failures']);self.assertEqual(digest(checkpoint),before)
            args.all_methods=True
            with patch.object(diagnostic,'ROOT',root),patch.object(diagnostic,'revision',return_value='pinned'),\
                    patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                    patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})),\
                    patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)),\
                    patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',FailedChild),\
                    patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:returned.append(a)):
                self.assertEqual(diagnostic.campaign(args),0)
            report=read(returned[-1][0]/'mega-fft.json')
            self.assertEqual(report['attempted'],list(diagnostic.METHODS+diagnostic.STACKS))
            self.assertEqual(len(returned),2);self.assertEqual(len(report['failures']),11)
            self.assertTrue(report['campaign_completed']);self.assertEqual(digest(checkpoint),before)
            class MixedChild(FailedChild):
                def run(self,spec,log,env):
                    super().run(spec,log,env)
                    route=spec['argv'][-1]
                    if route=='tf32':return 77
                    request=read(Path(spec['argv'][spec['argv'].index('--request')+1]))
                    output=Path(spec['argv'][spec['argv'].index('--output')+1])
                    atomic_json(output/'result.json',dict(route=route,
                        request_sha256=digest(output.parent/'request.json'),rollback_exact=True,
                        disposable_updates=11,formal_updates_committed=0,
                        timing=dict(timing_promising=route in ('fft','micro-up','resident'),throughput_gain=.1)))
                    return 0
            with patch.object(diagnostic,'ROOT',root),patch.object(diagnostic,'revision',return_value='pinned'),\
                    patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                    patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})),\
                    patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)),\
                    patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',MixedChild),\
                    patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:returned.append(a)):
                self.assertEqual(diagnostic.campaign(args),0)
            report=read(returned[-1][0]/'mega-fft.json')
            self.assertEqual(report['attempted'][-1],'fft+micro-up+resident')
            self.assertEqual(len(report['attempted']),12);self.assertEqual(len(returned),3)
            self.assertEqual(list(report['failures']),['tf32']);self.assertTrue(report['source_unchanged'])
            args.refine=True;args.all_methods=False
            with patch.object(diagnostic,'ROOT',root),patch.object(diagnostic,'revision',return_value='pinned'),\
                    patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                    patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})),\
                    patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)),\
                    patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',FailedChild),\
                    patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:returned.append(a)):
                self.assertEqual(diagnostic.campaign(args),0)
            report=read(returned[-1][0]/'mega-fft.json')
            self.assertEqual(report['attempted'],list(diagnostic.REFINEMENTS))
            self.assertEqual(len(returned),4);self.assertTrue(report['source_unchanged'])
            args.graph_repair=True;args.refine=False
            with patch.object(diagnostic,'ROOT',root),patch.object(diagnostic,'revision',return_value='pinned'),\
                    patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                    patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})),\
                    patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)),\
                    patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',FailedChild),\
                    patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:returned.append(a)):
                self.assertEqual(diagnostic.campaign(args),0)
            report=read(returned[-1][0]/'mega-fft.json')
            self.assertEqual(report['attempted'],list(diagnostic.GRAPH_REPAIR))
            self.assertEqual(len(returned),5);self.assertTrue(report['source_unchanged'])
            args.graph_diagnose=True;args.graph_repair=False
            with patch.object(diagnostic,'ROOT',root),patch.object(diagnostic,'revision',return_value='pinned'),\
                    patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                    patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})),\
                    patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)),\
                    patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',FailedChild),\
                    patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:returned.append(a)):
                self.assertEqual(diagnostic.campaign(args),0)
            report=read(returned[-1][0]/'mega-fft.json')
            self.assertEqual(report['attempted'],['graph-probe'])
            self.assertEqual(len(report['skipped']),2);self.assertFalse(report['measurement_completed'])
            self.assertEqual(len(returned),6);self.assertTrue(report['source_unchanged'])
            class PassingProbe(MixedChild):
                def run(self,spec,log,env):
                    if spec['argv'][-1]!='graph-probe':return super().run(spec,log,env)
                    FailedChild.run(self,spec,log,env)
                    output=Path(spec['argv'][spec['argv'].index('--output')+1])
                    atomic_json(output/'result.json',dict(route='graph-probe',request_sha256=digest(output.parent/'request.json'),
                        formal_updates_committed=0,disposable_updates=0,probe_passed=True))
                    return 0
            with patch.object(diagnostic,'ROOT',root),patch.object(diagnostic,'revision',return_value='pinned'),\
                    patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                    patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})),\
                    patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value=dict(idle=True)),\
                    patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',PassingProbe),\
                    patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:returned.append(a)):
                self.assertEqual(diagnostic.campaign(args),0)
            report=read(returned[-1][0]/'mega-fft.json')
            self.assertEqual(report['attempted'],list(diagnostic.GRAPH_DIAGNOSE))
            self.assertTrue(report['measurement_completed']);self.assertEqual(len(returned),7)





    def test_combination_resolves_conflicting_tiers_and_precision(self):
        results={n:dict(timing=dict(timing_promising=True,throughput_gain=.1)) for n in diagnostic.METHODS}
        results['micro-up']['timing']['throughput_gain']=.2
        self.assertEqual(diagnostic.combined(results),'fft+bf16+graph+micro-up+resident')
        self.assertIsNone(diagnostic.combined({'fft':results['fft']}))
        for route in ('fft+unknown','micro-up+micro-down','fft+fft','../graph'):
            with self.assertRaises(ValueError):diagnostic.settings(route)
        for route in diagnostic.METHODS+diagnostic.STACKS:
            spec=diagnostic.child_spec(Path('/fixture'),route)
            self.assertEqual(spec['argv'][-2:],['--route',route])
            self.assertEqual(Path(spec['argv'][spec['argv'].index('--output')+1]),Path('/fixture')/route)

    def test_cli_help_routes_without_native_import(self):
        proc=subprocess.run([sys.executable,'-B','-c',"import sys,runpy;sys.modules['torch']=None;sys.modules['numpy']=None;sys.argv=['mega','--help'];runpy.run_module('experiments.baseline_delivery.mega_diagnostic',run_name='__main__')"],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        from scripts.experiment import DIAGNOSTICS
        self.assertEqual(DIAGNOSTICS['mega-fft'],'mega_diagnostic')


if __name__=='__main__':unittest.main()
