"""Authenticated prefix reuse and diagnostic preservation, framework-free."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from experiments.baseline_delivery import key_bias_stability as e,key_bias_stability_native as n,torch_acceleration as a
from experiments.baseline_delivery import attention_precision_diagnostic as source,runner,attention_diagnostic,diagnostic_watch
from experiments.baseline_delivery.common import read,atomic_json,digest,fingerprint
from tests.test_baseline_key_bias_stability import Runtime


def prefix(folder):
    origin=folder/'source';(origin/'jobs'/source.JOB).mkdir(parents=True)
    request=dict(source_suite=str(origin),source_config={'seed':42},source_config_sha256='config',
        checkpoint=dict(sha256='a'*64,step=14268),source_metadata_sha256={},data='data',cache='cache',
        plugin_files=e.recovery.code_binding(),pilot_binding=e.binding(),source_revision='old')
    old=copy.deepcopy(request);old['pilot_binding'].pop('reuse_review_sha256')
    old['pilot_binding']['files']['key_bias_stability.py']='old-launcher'
    old['pilot_binding']['files']['key_bias_stability_native.py']='old-observation'
    atomic_json(folder/'stability-input.json',old);request_sha=digest(folder/'stability-input.json');results={}
    for mode in n.MODES[:-2]:
        out=folder/mode;out.mkdir();value=n.stability(Runtime(),out,mode,request_sha)
        value.update(schema=e.recovery.SCHEMA,mode=mode,request_sha256=request_sha,key_bias_manifest=e.recovery.manifest(),
            checkpoint_sha256='a'*64,checkpoint_step=14268,plugin_tree_sha256=fingerprint(old['plugin_files']),
            candidate_admitted=False,formal_training_started=False,training_samples_committed=0)
        results[mode]=value
    steady=dict(mode='compile_graph',generation=1,validation_checks=1,graph_resident=True,validated=True)
    rows=[dict(route=name,seconds=2. if name=='incumbent' else 1.,update=dict(row_ids_sha256='rows'),
        full_state_sha256=name,steady_before=steady if name=='incumbent' else None,
        steady_after=steady if name=='incumbent' else None) for name in ('incumbent','candidate','candidate','incumbent')*2]
    value=copy.deepcopy(results['fp32']);value.update(mode='graph16',timing_records=rows,timing=n.compare_summary(rows))
    results['graph16']=value
    report=dict(source_revision='old',pilot_binding=old['pilot_binding'],plugin_files=old['plugin_files'],results=results,
        passed=False,candidate_admitted=False,formal_training_started=False,training_samples_committed=0,
        original_checkpoint_and_metadata_unchanged=True,stability_gate=n.gate(results),recovery=n.agreement(results),
        disposable_checkpoint=dict(removed=True),child_executions=[{'exit_code':0}]*6+[{'exit_code':1}])
    path=folder/'key-bias-stability.json';atomic_json(path,report)
    review=dict(source_revision='old',pilot_binding=old['pilot_binding'],request_sha256=request_sha,
        source_report=dict(sha256=digest(path),bytes=path.stat().st_size))
    review_path=folder/'review.json';atomic_json(review_path,review)
    return request,report,review_path


class Tests(unittest.TestCase):
    def test_reuse_authenticates_request_report_source_and_unchanged_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);request,prior,review=prefix(folder)
            with patch.object(e,'REUSE_REVIEW',review):
                reused,binding=e.bind_reuse(folder,request)
                self.assertEqual(reused,prior);self.assertEqual(binding['rerun_modes'],['partition32'])
                for key in ('checkpoint','source_metadata_sha256','plugin_files','source_config'):
                    bad=copy.deepcopy(request);bad[key]={'changed':True}
                    with self.subTest(key=key),self.assertRaises(ValueError): e.bind_reuse(folder,bad)
                bad=copy.deepcopy(request);bad['pilot_binding']['files']['partitioned_attention.py']='changed'
                with self.assertRaises(ValueError): e.bind_reuse(folder,bad)
                p=folder/'stability-input.json';original=p.read_bytes();p.write_bytes(original+b' ')
                with self.assertRaises(ValueError): e.bind_reuse(folder,request)
                p.write_bytes(original)
                p=folder/'key-bias-stability.json';p.write_bytes(p.read_bytes()+b' ')
                with self.assertRaises(ValueError): e.bind_reuse(folder,request)

    def test_reuse_recomputes_gates_and_rejects_missing_or_unvalidated_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);request,prior,review=prefix(folder);pin=read(review)
            for change in ('missing','recovery','timing','resident','failed','rollback'):
                bad=copy.deepcopy(prior)
                if change=='missing': del bad['results']['resume']
                if change=='recovery': bad['results']['resume']['component_sha256']['rng']='changed'
                if change=='timing': bad['results']['graph16']['timing_records'][0]['seconds']=.5
                if change=='resident': bad['results']['graph16']['timing_records'][0]['steady_before']['graph_resident']=False
                if change=='failed': bad['results']['bf16_analytic']['passed']=False
                if change=='rollback': bad['results']['fp32']['rollback']['full_state_exact']=False
                path=folder/'key-bias-stability.json';atomic_json(path,bad)
                pin['source_report']=dict(sha256=digest(path),bytes=path.stat().st_size);atomic_json(review,pin)
                with self.subTest(change=change),patch.object(e,'REUSE_REVIEW',review),self.assertRaises(ValueError): e.bind_reuse(folder,request)

    def test_parent_retry_launches_only_partition32_and_preserves_both_error_files(self):
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);request,prior,review=prefix(folder);out=folder/'retry';out.mkdir();modes=[]
                class Supervisor:
                    def __init__(self,*args): self.records=[]
                    def run(self,spec,log,env):
                        argv=spec['argv'];mode=argv[argv.index('--mode')+1];modes.append(mode)
                        target=Path(argv[argv.index('--output')+1]);target.mkdir();self.records.append({'exit_code':int(fail)})
                        atomic_json(target/'comparison.json',dict(passed=False,timing_records=[{'route':'incumbent'}],failure={'underlying':'retained'}))
                        if fail:
                            atomic_json(target/'worker-error.json',dict(error_type='AccelerationRequired'));return 1
                        req=read(out/'stability-input.json');value=copy.deepcopy(prior['results']['graph16'])
                        value.update(mode=mode,request_sha256=digest(out/'stability-input.json'),plugin_tree_sha256=fingerprint(req['plugin_files']))
                        atomic_json(target/'result.json',value);return 0
                args=NS(source_suite=Path(request['source_suite']),cache=Path('cache'),gpu='0',screen_from=Path('screen'),benchmark_from=Path('bench'),reuse_from=folder,child_timeout=900,stall_timeout=300)
                with ExitStack() as held:
                    bound={k:request[k] for k in e.recovery.BINDING_KEYS}
                    for obj,name,value in [(source,'bind_source',bound),(source,'unchanged',None),
                        (e.recovery,'bind_screen',{'suite':'screen'}),(e,'bind_benchmark',{'suite':'bench'}),
                        (runner,'discover_gpus',(['0'],{'0':'GPU'})),(attention_diagnostic,'gpu_preflight',{'idle':True})]:
                        held.enter_context(patch.object(obj,name,return_value=value))
                    held.enter_context(patch.object(e,'bind_reuse',return_value=(prior,{'suite':str(folder),'rerun_modes':['partition32']})))
                    held.enter_context(patch.object(e,'revision',return_value='retry-revision'))
                    held.enter_context(patch.object(diagnostic_watch,'Supervisor',Supervisor))
                    report={}
                    if fail:
                        with self.assertRaises(RuntimeError): e.execute(args,out,report)
                        partial=report['partial_children']['partition32']
                        self.assertEqual(partial['comparison.json']['failure']['underlying'],'retained')
                        self.assertEqual(partial['worker-error.json']['error_type'],'AccelerationRequired')
                        self.assertFalse((out/'interrupt').exists())
                    else:
                        e.execute(args,out,report);self.assertTrue(report['passed'])
                    self.assertEqual(modes,['partition32']);self.assertEqual(len(report['reused_child_executions']),6)
                    self.assertEqual(report['recovery'],prior['recovery']);self.assertTrue(report['original_checkpoint_and_metadata_unchanged'])

    def test_real_executor_rejection_retains_native_diagnostic_and_seven_completed_timings(self):
        class Graph:
            def __init__(self): self.resident=False;self.builds=0
            def set_mode(self,mode): pass
            def clear(self): self.resident=False
            def generation(self): return self.builds
            def status(self): return dict(graph_resident=self.resident)
            def backend_error(self,error): return False
            def forward(self,batch,mode):
                if mode=='compile_graph' and not self.resident:
                    self.builds+=1
                    if self.builds==3:
                        error=RuntimeError('recompile_limit PRIVATE_VALUE')
                        raise a.BackendFailure('private native message',a.backend_diagnostic(error,'partition_compile_warmup','eager_partitioned'))
                    self.resident=True
        with tempfile.TemporaryDirectory() as tmp:
            rt=Runtime();rt.np=NS(ndarray=type('Unused',(),{}));rt.opt=NS(zero_grad=lambda **k:None)
            rt.t.is_tensor=lambda x:False
            rt.t.cuda.OutOfMemoryError=MemoryError;original=rt.update
            def update(enabled,**kwargs):
                if kwargs.get('executor'): kwargs['executor'].forward(None)
                return original(enabled,**kwargs)
            rt.update=update;graph=Graph()
            with patch.object(a,'torch_runtime',return_value=graph),self.assertRaises(a.AccelerationRequired):
                n.comparison(rt,Path(tmp),'partition32')
            report=read(Path(tmp)/'comparison.json');failure=report['failure']
            self.assertEqual(failure['completed_timing_records'],7);self.assertEqual(len(report['timing_records']),7)
            self.assertEqual(failure['executor_diagnostics'][0]['error']['exceptions'][0]['category'],'recompile_budget')
            self.assertEqual(failure['executor_state']['mode'],'compile');self.assertNotIn('private',str(report).lower())
            self.assertEqual(report['rollback'],e.recovery.ROLLBACK);self.assertEqual(rt.snapshot(),rt.base);self.assertFalse(graph.resident)

    def test_real_executor_preserves_structured_parity_field(self):
        class Graph:
            def set_mode(self,mode): pass
            def clear(self): pass
            def generation(self): return 1
            def status(self): return dict(graph_resident=True)
            def backend_error(self,error): return False
        rt=Runtime();rt.np=NS(ndarray=type('Unused',(),{}));rt.opt=NS(zero_grad=lambda **k:None);rt.t.cuda.OutOfMemoryError=MemoryError
        graph=Graph()
        def changed(runtime,left,right,path='state'):
            raise n.ParityMismatch('control','state/rng/counter')
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'torch_runtime',return_value=graph),patch.object(n,'compare_values',side_effect=changed):
            with self.assertRaises(a.AccelerationRequired): n.comparison(rt,Path(tmp),'partition32')
            report=read(Path(tmp)/'comparison.json');proof=report['failure']['executor_diagnostics'][0]['report']
            self.assertEqual(proof['field'],'state/rng/counter');self.assertFalse(proof['tensor_values_included'])
            self.assertEqual(proof['kind'],'control');self.assertEqual(rt.snapshot(),rt.base)


if __name__=='__main__': unittest.main()
