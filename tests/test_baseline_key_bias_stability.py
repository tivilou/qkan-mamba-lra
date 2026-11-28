"""Bounded pilot controls, injected fake updates and actual stdlib supervisor."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from experiments.baseline_delivery import key_bias_stability as entry,key_bias_stability_native as n
from experiments.baseline_delivery import bootstrap,return_artifacts,diagnostic_watch
from experiments.baseline_delivery.common import ROOT,read,atomic_json,digest
from tests.test_baseline_key_bias_benchmark import BenchRuntime


class Runtime(BenchRuntime):
    def __init__(self):
        super().__init__();self.split=list(range(20000));self.config['dropout']=.1
        self.rng_state=lambda:self.state['rng'];self.restore_rng=lambda value:None
        self.t.compiler=NS(reset=lambda:None)
    def update(self,enabled,**kwargs):
        value=super().update(enabled,**kwargs)
        self.state['scheduler']['progress']+=128;self.state['model']['weight']-=.001
        return value


class Tests(unittest.TestCase):
    def test_64versus32plus32_complete_state_and_checkpoint_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);results={}
            for mode in ('bf16_analytic','interrupt','resume'):
                out=folder/mode;out.mkdir();rt=Runtime()
                results[mode]=n.stability(rt,out,mode,'request')
                self.assertEqual(rt.snapshot(),rt.base)
                self.assertTrue(all(not p._backward_hooks for p in rt.parameters.values()))
            self.assertTrue(n.agreement(results)['passed'])
            self.assertEqual(results['bf16_analytic']['step'],14332)
            self.assertTrue(results['interrupt']['diagnostic_checkpoint']['complete_values_reopened_exact'])
            broken=copy.deepcopy(results);broken['resume']['component_sha256']['rng']='other'
            with self.assertRaises(ValueError): n.agreement(broken)
    def test_epoch_capacity_stops_before_first_update(self):
        rt=Runtime();rt.split=list(range(1000))
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'8192'):
            n.stability(rt,Path(tmp),'fp32','req')
        self.assertEqual(rt.events,[])
    def test_online_deterioration_stops_at_first_block(self):
        rt=Runtime();original=rt.update
        def bad(*args,**kwargs):
            value=original(*args,**kwargs);value['loss']=2.;return value
        rt.update=bad
        reference=[dict(loss=1.,row_ids_sha256=str(rt.base['progress']['cursor']+128*(i+1))) for i in range(64)]
        with tempfile.TemporaryDirectory() as tmp:
            result=n.stability(rt,Path(tmp),'bf16_analytic','req',reference=reference)
            self.assertFalse(result['passed']);self.assertEqual(len(result['updates']),8)
            self.assertEqual(rt.snapshot(),rt.base)
    def test_failure_retains_partial_and_rolls_back(self):
        rt=Runtime();rt.fail_at=10
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError): n.stability(rt,Path(tmp),'bf16_analytic','req')
            result=read(Path(tmp)/'stability.json');self.assertEqual(len(result['updates']),9)
            self.assertTrue(result['rollback']['full_state_exact']);self.assertEqual(rt.snapshot(),rt.base)
            self.assertTrue(all(not p._backward_hooks for p in rt.parameters.values()))
    def test_midpoint_payload_receipt_and_source_rejection(self):
        for field,value in [('strategy',{}),('request_sha256','other'),('source_checkpoint_sha256','x')]:
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);(folder/'interrupt').mkdir();(folder/'resume').mkdir()
                n.stability(Runtime(),folder/'interrupt','interrupt','request')
                receipt=read(folder/'interrupt/diagnostic-state.json');receipt[field]=value
                atomic_json(folder/'interrupt/diagnostic-state.json',receipt)
                with self.assertRaises(ValueError): n.load_midpoint(Runtime(),folder/'interrupt/diagnostic-state.pt',folder/'resume','request')
    def test_block_gate_same_population_finite_and_deterioration(self):
        reference=[dict(loss=.4,row_ids_sha256=str(i)) for i in range(64)]
        rows={name:dict(updates=copy.deepcopy(reference)) for name in ('fp32','fp32_analytic','bf16_analytic')}
        self.assertTrue(n.gate(rows)['passed'])
        for row in rows['bf16_analytic']['updates'][8:16]: row['loss']=.501
        self.assertFalse(n.gate(rows)['passed'])
        rows['bf16_analytic']['updates'][0]['loss']=float('nan')
        with self.assertRaises(ValueError): n.gate(rows)
        rows['bf16_analytic']['updates'][0]=dict(loss=.4,row_ids_sha256='other')
        with self.assertRaises(ValueError): n.gate(rows)
    def test_summary_rejects_replay_changes_and_nonpositive_pair(self):
        records=[dict(route=name,seconds=2. if name=='incumbent' else 1.,update=dict(row_ids_sha256='rows'),full_state_sha256=name) for name in ('incumbent','candidate','candidate','incumbent')*2]
        self.assertTrue(n.compare_summary(records)['passed'])
        changed=copy.deepcopy(records);changed[4]['full_state_sha256']='other'
        with self.assertRaises(ValueError): n.compare_summary(changed)
        records[1]['seconds']=3.
        self.assertFalse(n.compare_summary(records)['passed'])
    def test_scalar_compare_preserves_rng_exactness(self):
        fake=NS(t=NS(is_tensor=lambda x:False),np=NS(ndarray=type('Unused',(),{})))
        n.compare_values(fake,{'loss':1.},{'loss':1.000001})
        with self.assertRaises(ValueError): n.compare_values(fake,{'rng':{'x':1.}},{'rng':{'x':1.000001}})
    def test_comparison_rebuilds_and_validates_outside_timing(self):
        from experiments.baseline_delivery import torch_acceleration as a
        class Graph:
            def __init__(self): self.resident=False;self.builds=0;self.clears=0
            def clear(self): self.resident=False;self.clears+=1
            def generation(self): return self.builds
            def status(self): return dict(graph_resident=self.resident)
        class Executor:
            def __init__(self,graph,*args,**kwargs):
                self.runtime=graph;self.state=dict(mode='compile_graph');self.checks=0;self.validated=set();self.validating=False
            def run(self,callback,*args,execution_key=None):
                if not self.runtime.resident:
                    self.runtime.builds+=1;self.runtime.resident=True;self.checks+=1
                    self.validated.add(('compile_graph',execution_key,self.runtime.builds))
                return callback()
        with tempfile.TemporaryDirectory() as tmp:
            rt=Runtime();rt.t.cuda.OutOfMemoryError=RuntimeError;graph=Graph();ticks=[];clock=0
            for name in ('incumbent','candidate','candidate','incumbent')*2:
                d=2 if name=='incumbent' else 1;ticks.extend([clock,clock+d]);clock+=d+1
            with patch.object(a,'torch_runtime',return_value=graph),patch.object(a,'Executor',Executor),\
                    patch.object(n.time,'perf_counter',side_effect=ticks): result=n.comparison(rt,Path(tmp),'graph16')
            self.assertTrue(result['passed']);self.assertEqual(graph.builds,3)
            self.assertEqual(rt.snapshot(),rt.base);self.assertFalse(graph.resident)
            proof=[v for v in result['timing_records'] if v['route']=='incumbent']
            self.assertTrue(all(v['steady_before']==v['steady_after'] for v in proof))
    def test_parent_dispatches_seven_owned_children_and_retains_failure(self):
        from experiments.baseline_delivery import attention_precision_diagnostic as source,runner,attention_diagnostic
        for failed in (None,'graph16'):
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);origin=folder/'runs/original';(origin/'jobs'/source.JOB).mkdir(parents=True)
                output=folder/'pilot';output.mkdir();modes=[]
                bound=dict(source_suite=str(origin),source_config={'seed':42},source_config_sha256='config',
                    checkpoint=dict(sha256='a'*64,step=14268),source_metadata_sha256={},data='data',cache='cache')
                class Supervisor:
                    def __init__(self,*args): self.records=[]
                    def run(self,spec,log,env):
                        argv=spec['argv'];mode=argv[argv.index('--mode')+1];modes.append(mode)
                        out=Path(argv[argv.index('--output')+1]);out.mkdir()
                        self.records.append(dict(exit_code=1 if mode==failed else 0))
                        if mode==failed: atomic_json(out/'worker-error.json',dict(stage=mode));return 1
                        request=read(output/'stability-input.json');request_sha=digest(output/'stability-input.json')
                        value=n.stability(Runtime(),out,mode,request_sha) if mode not in ('graph16','partition32') else dict(passed=True,rollback=entry.recovery.ROLLBACK)
                        value.update(schema=entry.recovery.SCHEMA,mode=mode,request_sha256=request_sha,key_bias_manifest=entry.recovery.manifest(),
                            checkpoint_sha256='a'*64,checkpoint_step=14268,plugin_tree_sha256=entry.fingerprint(request['plugin_files']),
                            candidate_admitted=False,formal_training_started=False,training_samples_committed=0)
                        atomic_json(out/'result.json',value);return 0
                args=NS(source_suite=origin,cache=Path('cache'),gpu='0',screen_from=Path('screen'),benchmark_from=Path('bench'),child_timeout=900,stall_timeout=300)
                report={}
                with patch.object(source,'bind_source',return_value=bound),patch.object(source,'unchanged'),patch.object(entry,'revision',return_value='rev'),\
                        patch.object(entry.recovery,'bind_screen',return_value={'suite':'screen'}),patch.object(entry,'bind_benchmark',return_value={'suite':'bench'}),\
                        patch.object(runner,'discover_gpus',return_value=(['0'],{'0':'GPU'})),patch.object(attention_diagnostic,'gpu_preflight',return_value={'idle':True}),\
                        patch.object(diagnostic_watch,'Supervisor',Supervisor):
                    if failed:
                        with self.assertRaises(RuntimeError): entry.execute(args,output,report)
                    else: entry.execute(args,output,report);self.assertTrue(report['passed'])
                self.assertEqual(modes,list(n.MODES[:6] if failed else n.MODES))
                self.assertTrue(report['recovery']['passed']);self.assertTrue(report['disposable_checkpoint']['removed'])
                self.assertTrue(report['original_checkpoint_and_metadata_unchanged'])
    def test_actual_supervisor_worker_reaches_validated_operation(self):
        child='''import importlib.abc,sys
from pathlib import Path
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','numpy','jax','tensorflow'): raise AssertionError(name)
sys.meta_path.insert(0,Guard())
from experiments.baseline_delivery import key_bias_stability as e
e.bind_benchmark=lambda folder,source:source['benchmark_evidence']
def native(request,output,mode,**kwargs):
 e.atomic_json(Path(output)/'result.json',{'mode':mode,'reached':callable(kwargs['operation']),'native_imports':0})
e.recovery.native=native
raise SystemExit(e.main())
'''
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'pilot_child.py').write_text(child,encoding='utf8')
            req=folder/'request.json';atomic_json(req,dict(pilot_binding=entry.binding(),benchmark_evidence=dict(suite='prior')))
            out=folder/'child';supervisor=diagnostic_watch.Supervisor(15,10,announce=lambda *a,**k:None)
            code=supervisor.run(dict(target='pilot_child',cwd=str(ROOT),argv=[sys.executable,'-B','-m','pilot_child',
                '--worker','--request',str(req),'--output',str(out),'--mode','fp32']),folder/'child.log',
                dict(os.environ,PYTHONPATH=os.pathsep.join([str(folder),str(ROOT)]),PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(code,0,(folder/'child.log').read_text())
            self.assertTrue(read(out/'result.json')['reached']);self.assertTrue((out/'diagnostics').is_dir())
    def test_failed_child_and_upload_failure_preserve_report(self):
        for child,upload in ((True,False),(False,True)):
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);seen=[]
                def execute(args,out,report):
                    report['results']={'fp32':{'partial':True}}
                    if child: raise RuntimeError('controlled child failure')
                def send(out,kind,files):
                    self.assertEqual(kind,'key-bias-stability');seen.append(read(out/files[0]))
                    if upload: raise RuntimeError('controlled upload failure')
                with patch.object(entry,'ROOT',folder),patch.object(entry,'revision',return_value='rev'),patch.object(entry,'execute',side_effect=execute),\
                        patch.object(bootstrap,'check_exchange'),patch.object(return_artifacts,'upload_declared',side_effect=send),\
                        patch.object(sys,'argv',['entry','--source-suite','source','--cache','cache','--screen-from','screen','--benchmark-from','bench']):
                    with self.assertRaises(RuntimeError): entry.main()
                self.assertEqual(len(seen),1);self.assertFalse(seen[0]['candidate_admitted'])
                self.assertEqual(len(list((folder/'runs').glob('*/key-bias-stability.json'))),1)


if __name__=='__main__': unittest.main()
