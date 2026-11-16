"""Stdlib/fake control checks. No native framework or claimed real fixture."""
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import key_bias_recovery as r,key_bias_trace as trace
from experiments.baseline_delivery import bootstrap,diagnostic_watch,return_artifacts
from experiments.baseline_delivery.common import ROOT,atomic_json,digest,read,fingerprint
from experiments.baseline_delivery.key_bias_consistency import manifest,NAMES
from tests.test_baseline_key_bias_consistency import TORCH,Tensor,hook_fixture


class Runtime:
    def __init__(self,fail=False):
        self.model,self.parameters=hook_fixture();self.fail=fail;self.calls=0
        self.t=NS(**vars(TORCH),save=lambda value,path:Path(path).write_bytes(pickle.dumps(value)),
            load=lambda path,**kwargs:pickle.loads(Path(path).read_bytes()))
        self.base=dict(progress=dict(step=14268,cursor=80,epoch=3),model={'weight':3.},
            optimizer={'m':.02,'v':.0003,'step':14268},scheduler={'progress':80},rng={'python':17,'cuda':29},gradients=None)
        self.request=dict(checkpoint=dict(sha256='a'*64));self.config=dict(dropout=.1)
        self.restore(self.base);self.schedule=NS(state_dict=lambda:dict(self.state['scheduler']))
    def snapshot(self): return copy.deepcopy(self.state)
    def restore(self,value): self.state=copy.deepcopy(value);self.progress=self.state['progress']
    def state_hash(self,value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    def update(self,enabled):
        assert enabled;self.calls+=1
        if self.fail: raise RuntimeError('controlled update failure')
        for p in self.parameters.values():
            for _ in range(8): p.emit(Tensor(range(384)))
        s=self.state;s['progress']['step']+=1;s['progress']['cursor']+=128
        s['scheduler']['progress']+=128;s['optimizer']['step']+=1
        s['optimizer']['m']*=.9;s['optimizer']['v']*=.999
        s['model']['weight']-=s['optimizer']['m'];s['rng']['cuda']+=7;s['rng']['python']+=3
        s['gradients']=[0,2]
        return dict(loss=s['model']['weight'],rows=128,row_ids_sha256=str(s['progress']['cursor']))


class RecoveryTests(unittest.TestCase):
    def test_independent_runtime_resume_equals_continuous_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);results={}
            for mode in r.MODES[:3]:
                output=folder/mode;output.mkdir();rt=Runtime()
                results[mode]=r.recovery_run(rt,output,mode,'request',folder/'interrupt/diagnostic-state.pt' if mode=='resume' else None)
                self.assertEqual(rt.snapshot(),rt.base)
                self.assertTrue(all(not p._backward_hooks for p in rt.parameters.values()))
            proof=r.agreement(results)
            self.assertTrue(proof['fresh_process']);self.assertEqual(proof['dropout'],.1)
            receipt=results['interrupt']['diagnostic_checkpoint']
            self.assertEqual(receipt['strategy'],manifest());self.assertTrue(receipt['complete_values_reopened_exact'])
            saved=pickle.loads((folder/'interrupt/diagnostic-state.pt').read_bytes())
            self.assertAlmostEqual(saved['state']['optimizer']['m'],.018)
            self.assertAlmostEqual(saved['state']['optimizer']['v'],.0002997)
            self.assertEqual(saved['state']['optimizer']['step'],14269)

    def test_load_rejects_legacy_strategy_wrong_source_request_or_modified_state(self):
        for field,value in [('strategy',{}),('source_checkpoint_sha256','b'*64),('request_sha256','other'),('baseline_state_sha256','bad')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);interrupt=folder/'interrupt';interrupt.mkdir();output=folder/'resume';output.mkdir()
                rt=Runtime();rt.update=lambda enabled:None
                # Build expected one-step progress without framework hooks.
                rt.state['progress']['step']+=1;rt.state['progress']['cursor']+=128
                path=interrupt/'diagnostic-state.pt';r.save_state(rt,path,'request')
                receipt=read(path.with_suffix('.json'));receipt[field]=value;atomic_json(path.with_suffix('.json'),receipt)
                with self.assertRaises(ValueError): r.load_state(Runtime(),path,output,'request')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'interrupt').mkdir();(folder/'resume').mkdir();path=folder/'interrupt/diagnostic-state.pt'
            r.recovery_run(Runtime(),folder/'interrupt','interrupt','request');path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'receipt'): r.load_state(Runtime(),path,folder/'resume','request')

    def test_payload_strategy_cannot_be_bypassed_by_matching_file_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'interrupt').mkdir();(folder/'resume').mkdir()
            r.recovery_run(Runtime(),folder/'interrupt','interrupt','request');path=folder/'interrupt/diagnostic-state.pt'
            payload=pickle.loads(path.read_bytes());payload['strategy']={};path.write_bytes(pickle.dumps(payload))
            receipt=read(path.with_suffix('.json'));receipt.update(sha256=digest(path),bytes=path.stat().st_size);atomic_json(path.with_suffix('.json'),receipt)
            with self.assertRaisesRegex(ValueError,'payload'): r.load_state(Runtime(),path,folder/'resume','request')

    def test_reopened_state_tamper_and_update_failure_restore_state_and_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=Runtime(fail=True)
            with self.assertRaises(RuntimeError): r.recovery_run(rt,Path(tmp),'uninterrupted','request')
            self.assertEqual(rt.snapshot(),rt.base)
            self.assertTrue(all(not p._backward_hooks for p in rt.parameters.values()))
            self.assertEqual(read(Path(tmp)/'recovery.json')['rollback'],r.ROLLBACK)
        with tempfile.TemporaryDirectory() as tmp:
            rt=Runtime();rt.t.load=lambda *args,**kwargs:dict(corrupted=True)
            with self.assertRaisesRegex(ValueError,'Reopened'): r.save_state(rt,Path(tmp)/'state.pt','request')

    def test_agreement_rejects_rng_component_output_and_loaded_hash_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);results={}
            for mode in r.MODES[:3]:
                (folder/mode).mkdir();results[mode]=r.recovery_run(Runtime(),folder/mode,mode,'request',folder/'interrupt/diagnostic-state.pt' if mode=='resume' else None)
            for key,value in [('full_state_sha256','other'),('component_sha256',{}),('updates',[]),('loaded_state_sha256','other')]:
                broken=copy.deepcopy(results);broken['resume'][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError): r.agreement(broken)

    def test_cleanup_removes_only_verified_disposable_checkpoint_after_agreement(self):
        from experiments.baseline_delivery.attention_update_diagnostic import remove_disposable_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);result=r.recovery_run(Runtime(),folder,'interrupt','request')
            receipt=dict(result['diagnostic_checkpoint']);receipt.pop('complete_values_reopened_exact')
            proof=remove_disposable_checkpoint(folder,dict(diagnostic_checkpoint=receipt))
            self.assertTrue(proof['removed']);self.assertFalse((folder/'diagnostic-state.pt').exists())


class BindingTests(unittest.TestCase):
    def test_current_config_code_and_dossier_binding_and_mutation_rejection(self):
        from experiments.baseline_delivery.attention_update import POLICY
        request=dict(schema='qkan-attention-update-input.v1',source_revision='rev',policy=POLICY,
            source_config={'seed':42},source_config_sha256=fingerprint({'seed':42}),optimization_config=read(r.BF16),
            optimization_config_sha256=digest(r.BF16),key_bias_config=read(r.PRESET),key_bias_config_sha256=digest(r.PRESET),
            key_bias_manifest=manifest(),dossier_sha256=digest(r.DOSSIER),plugin_plan_sha256=digest(r.PLAN),plugin_files=r.code_binding())
        self.assertTrue(r.validate(request,'rev'))
        for key,value in [('key_bias_manifest',{}),('plugin_files',{}),('plugin_plan_sha256','bad'),('key_bias_config_sha256','bad')]:
            broken=dict(request);broken[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): r.validate(broken,'rev')

    def test_child_acceptance_requires_new_manifest_replay_and_full_values(self):
        request=dict(checkpoint=dict(sha256='a'*64,step=14268),plugin_files={'x':'hash'})
        route=dict(passed=True,full_model_replay_exact=True,structural_errors=0,structural_warnings=0,samples=2,complete_values_reopened_exact=True)
        value=dict(schema=r.SCHEMA,mode='trace',passed=True,request_sha256='request',key_bias_manifest=manifest(),
            checkpoint_sha256='a'*64,checkpoint_step=14268,plugin_tree_sha256=fingerprint(request['plugin_files']),
            rollback=r.ROLLBACK,candidate_admitted=False,formal_training_started=False,training_samples_committed=0,
            trace=dict(passed=True,schema='sample-trace.v2',routes=dict(fp32=route,bf16_analytic=route),same_samples=True,target_boundary_passed=True))
        self.assertEqual(r.accept(value,request,'request','trace',0),value)
        for key in ('complete_values_reopened_exact','full_model_replay_exact'):
            broken=copy.deepcopy(value);broken['trace']['routes']['fp32'][key]=False
            with self.assertRaises(ValueError): r.accept(broken,request,'request','trace',0)
        for field,new in [('key_bias_manifest',{}),('rollback',{}),('candidate_admitted',True)]:
            broken=dict(value);broken[field]=new
            with self.assertRaises(ValueError): r.accept(broken,request,'request','trace',0)

    def test_reviewed_screen_requires_same_original_binding_and_preserved_request(self):
        binding={k:k for k in r.BINDING_KEYS};evidence={'suite':'prior','hash':'priorhash'}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);atomic_json(folder/'screen-input.json',binding)
            report=dict(source_revision='rev',screen_complete=True,candidate_admitted=False,
                numerical_passed=dict(fp32_analytic=True,bf16_analytic=True),result=dict(request_sha256=digest(folder/'screen-input.json')),evidence_binding=evidence)
            atomic_json(folder/'key-bias.json',report)
            review=folder/'review.json';atomic_json(review,dict(source_revision='rev',source_report=dict(sha256=digest(folder/'key-bias.json'),bytes=(folder/'key-bias.json').stat().st_size)))
            with patch.object(r,'REVIEW',review),patch('experiments.baseline_delivery.key_bias_diagnostic.bind_evidence',return_value=(evidence,None)):
                identity=r.bind_screen(folder,binding);self.assertEqual(identity['report_sha256'],digest(folder/'key-bias.json'))
                with self.assertRaises(ValueError): r.bind_screen(folder,dict(binding,cache='changed'))
                atomic_json(folder/'screen-input.json',dict(binding,cache='changed'))
                with self.assertRaises(ValueError): r.bind_screen(folder,binding)


class TraceLogicTests(unittest.TestCase):
    def test_scalar_adam_check_uses_actual_gradient_and_retained_history(self):
        before=dict(m=[.02]*384,v=[.003]*384,step=14268);g=[.0]*384
        after=dict(m=[.018]*384,v=[.002997]*384,step=14269)
        self.assertTrue(trace.adam_equations(before,after,g,.9,.999))
        for key,value in [('m',[0.]*384),('v',[0.]*384),('step',1)]:
            broken=dict(after);broken[key]=value
            with self.assertRaises(ValueError): trace.adam_equations(before,broken,g,.9,.999)
        with self.assertRaises(ValueError): trace.adam_equations(before,after,[0.]*128,.9,.999)

    def test_selected_samples_are_first_two_original_cursor_rows(self):
        class IDs(list):
            def numpy(self): return self
            def copy(self): return IDs(self)
        class Perm(list):
            def __getitem__(self,key): return IDs(super().__getitem__(key))
        rt=NS(base=dict(progress=dict(cursor=2),permutation=Perm([100,101,709,231,999])))
        self.assertEqual(trace.selected_ids(rt),[709,231])

    def test_complete_value_replay_rejects_state_dtype_sample_and_value_change(self):
        fake=NS(np=NS(array_equal=lambda a,b:a==b))
        value=dict(dtypes={'x':'fp32'},post_update_state_sha256='state',ids=[709,231],labels=[1,0],values={'x':[2,4]})
        self.assertTrue(trace.replay_equal(fake,value,copy.deepcopy(value)))
        for key,new in [('dtypes',{}),('post_update_state_sha256','other'),('ids',[0,1]),('labels',[0,0]),('values',{'x':[2,5]})]:
            broken=copy.deepcopy(value);broken[key]=new
            with self.subTest(key=key),self.assertRaises(ValueError): trace.replay_equal(fake,value,broken)

    def test_collector_rejects_baseline_adapter_mismatch_and_rolls_back(self):
        rt=Runtime();rt.np=NS(array_equal=lambda a,b:a==b)
        rt.metrics=lambda a,b:dict(logits=.01)
        observed=dict(comparison={})
        with tempfile.TemporaryDirectory() as tmp,patch.object(trace,'produce',return_value=observed):
            with self.assertRaisesRegex(ValueError,'adapter'): trace.collect_and_validate(rt,Path(tmp))
            self.assertEqual(rt.snapshot(),rt.base)


class EntryTests(unittest.TestCase):
    def invoke(self,folder,fail=False,upload_fail=False):
        seen=[]
        def execute(args,output,report):
            report['results']={'uninterrupted':{'passed':True}}
            if fail: raise RuntimeError('controlled missing child result')
            report.update(passed=True,status='recovery_trace_preflight_passed',native_trace_ready=True)
        def upload(output,kind,files):
            seen.append(read(output/files[0]));self.assertEqual(kind,'key-bias-recovery')
            if upload_fail: raise RuntimeError('controlled return failure')
        argv=['entry','--source-suite',str(folder/'source'),'--cache',str(folder/'cache'),
            '--screen-from',str(folder/'completed'),'--optimizations',str(r.PRESET)]
        with patch.object(r,'ROOT',folder),patch.object(sys,'argv',argv),patch.object(r,'execute',side_effect=execute),\
                patch.object(r,'revision',return_value='revision'),patch.object(bootstrap,'check_exchange'),\
                patch.object(return_artifacts,'upload_declared',side_effect=upload):
            if fail or upload_fail:
                with self.assertRaises(RuntimeError): r.main()
            else: self.assertEqual(r.main(),0)
        return seen

    def test_success_never_opens_formal_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen=self.invoke(Path(tmp));self.assertTrue(seen[0]['passed']);self.assertFalse(seen[0]['candidate_admitted'])
            self.assertFalse(seen[0]['formal_training_started']);self.assertEqual(seen[0]['training_samples_committed'],0)

    def test_child_failure_and_upload_failure_preserve_compact_partial_evidence(self):
        for child,upload in ((True,False),(False,True)):
            with tempfile.TemporaryDirectory() as tmp:
                seen=self.invoke(Path(tmp),child,upload);self.assertEqual(len(seen),1)
                self.assertIn('uninterrupted',seen[0]['results'])
                self.assertEqual(seen[0]['passed'],not child)
                if child: self.assertIn('failure_locations',seen[0])
                paths=list((Path(tmp)/'runs').glob('*/key-bias-recovery.json'));self.assertEqual(len(paths),1)


class OrchestrationTests(unittest.TestCase):
    def test_four_distinct_children_resume_exact_sibling_and_failure_keeps_partial_results(self):
        from experiments.baseline_delivery import attention_precision_diagnostic as source_module,runner,attention_diagnostic
        for fail_mode in (None,'resume'):
            with self.subTest(fail_mode=fail_mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'source/runs/original';(source/'jobs'/source_module.JOB).mkdir(parents=True)
                output=root/'result';output.mkdir();specs=[]
                binding=dict(source_suite=str(source),source_config={'seed':42},source_config_sha256=fingerprint({'seed':42}),
                    checkpoint=dict(sha256='a'*64,step=14268),source_metadata_sha256={},data='data',cache='cache')
                args=NS(source_suite=source,cache=Path('cache'),gpu='0',screen_from=Path('screen'),optimizations=r.PRESET,
                    child_timeout=900,stall_timeout=300)
                class Supervisor:
                    def __init__(self,*unused): self.records=[]
                    def run(self,spec,log,env):
                        argv=spec['argv'];mode=argv[argv.index('--mode')+1];specs.append(spec)
                        self.records.append(dict(exit_code=1 if mode==fail_mode else 0))
                        folder=Path(argv[argv.index('--output')+1]);folder.mkdir()
                        if mode==fail_mode:
                            atomic_json(folder/'worker-error.json',dict(stage=mode));return 1
                        if mode=='trace':
                            lane=dict(passed=True,full_model_replay_exact=True,structural_errors=0,structural_warnings=0,
                                samples=2,complete_values_reopened_exact=True)
                            result=dict(passed=True,rollback=r.ROLLBACK,trace=dict(passed=True,schema='sample-trace.v2',
                                routes=dict(fp32=lane,bf16_analytic=lane),same_samples=True,target_boundary_passed=True))
                        else:
                            checkpoint=Path(argv[argv.index('--recovery-state')+1]) if mode=='resume' else None
                            result=r.recovery_run(Runtime(),folder,mode,digest(output/'recovery-input.json'),checkpoint)
                        result.update(schema=r.SCHEMA,mode=mode,request_sha256=digest(output/'recovery-input.json'),
                            key_bias_manifest=manifest(),checkpoint_sha256='a'*64,checkpoint_step=14268,
                            plugin_tree_sha256=fingerprint(r.code_binding()),candidate_admitted=False,
                            formal_training_started=False,training_samples_committed=0)
                        atomic_json(folder/'result.json',result);return 0
                report=dict(source_revision='rev',plugin_files=r.code_binding())
                with patch.object(source_module,'bind_source',return_value=binding),patch.object(source_module,'unchanged'),\
                        patch.object(r,'bind_screen',return_value={'suite':'screen'}),patch.object(runner,'discover_gpus',return_value=(['0'],{'0':{}})),\
                        patch.object(attention_diagnostic,'gpu_preflight',return_value=dict(idle=True,compute_pids=[])),\
                        patch.object(diagnostic_watch,'Supervisor',Supervisor):
                    if fail_mode:
                        with self.assertRaises(RuntimeError): r.execute(args,output,report)
                        self.assertEqual(set(report['results']),{'uninterrupted','interrupt'})
                        self.assertTrue((output/'interrupt/diagnostic-state.pt').exists())
                        self.assertEqual(report['native_failure_locations'],{'stage':'resume'})
                    else:
                        r.execute(args,output,report);self.assertTrue(report['passed']);self.assertTrue(report['native_trace_ready'])
                        self.assertFalse((output/'interrupt/diagnostic-state.pt').exists())
                self.assertEqual([s['argv'][s['argv'].index('--mode')+1] for s in specs],list(r.MODES[:len(specs)]))
                self.assertTrue(all(s['target']==r.TARGET and '--worker' in s['argv'] for s in specs))


class Array:
    """Tiny injected array for artifact-control checks; never a native fixture."""
    def __init__(self,data,dtype=None):
        self.data=copy.deepcopy(data.data if isinstance(data,Array) else data)
        self.shape=self.dimensions(self.data);self.ndim=len(self.shape)
        flat=self.flatten(self.data);self.size=len(flat);self.nbytes=self.size*8
        self.dtype=dtype or ('bool' if all(type(v) is bool for v in flat) else 'int64' if all(type(v) is int for v in flat) else 'float32')
    @staticmethod
    def dimensions(data): return (len(data),*Array.dimensions(data[0])) if isinstance(data,list) and data else ()
    @staticmethod
    def flatten(data): return [v for child in data for v in Array.flatten(child)] if isinstance(data,list) else [data]
    def __getitem__(self,key):
        if isinstance(key,tuple):
            v=self.data
            for i in key: v=v[i]
            return Scalar(v)
        v=self.data[key];return Array(v,self.dtype) if isinstance(v,list) else v
    def __iter__(self): return iter(self.data)
    def __len__(self): return len(self.data)
    def copy(self): return Array(self.data,self.dtype)
    def numpy(self): return self
    def all(self): return all(self.flatten(self.data))
    def any(self): return any(self.flatten(self.data))
    def tolist(self): return copy.deepcopy(self.data)
    def argmax(self): return max(range(len(self.data)),key=self.data.__getitem__)
    def __ne__(self,other): return Array(self.apply(self.data,other,lambda a,b:a!=b))
    def __sub__(self,other): return Array(self.apply(self.data,other.data if isinstance(other,Array) else other,lambda a,b:a-b),self.dtype)
    @staticmethod
    def apply(a,b,fn):
        if isinstance(a,list): return [Array.apply(v,b[i] if isinstance(b,list) else b,fn) for i,v in enumerate(a)]
        return fn(a,b)


class Scalar(float):
    def item(self): return float(self)


class FakeNP:
    @staticmethod
    def array_equal(a,b): return (a.data if isinstance(a,Array) else a)==(b.data if isinstance(b,Array) else b)
    @staticmethod
    def ascontiguousarray(a): return a.copy()
    @staticmethod
    def isfinite(a): return Array([math.isfinite(v) for v in Array.flatten(a.data)])
    @staticmethod
    def arange(n): return Array(list(range(n)))
    @staticmethod
    def save(path,value,allow_pickle=False): Path(path).write_bytes(pickle.dumps(value))
    @staticmethod
    def load(path,**kwargs): return pickle.loads(Path(path).read_bytes())


class TraceArtifactTests(unittest.TestCase):
    def fixture(self):
        ids=Array([1,0]);tokens=[[4]*4000,[7]*4000];mask=[[True]*4000]*2;positions=[list(range(4000))]*2
        values={prefix+key+str(side):Array([[.2,.3],[.4,.5]]) for prefix in ('projected_','sdpa_','gradient_') for key in ('q','k','v') for side in (0,1)}
        for side in (0,1):
            values.update({k+str(side):Array(v) for k,v in dict(tokens=tokens,mask=mask,positions=positions,
                encoder=[[.4],[.6]],attention=[[.2],[.3]]).items()})
        values.update(logits=Array([[.2,.8],[.9,.1]]),loss=Array([[.4],[.7]]),pooled=Array([[.3],[.5]]),gradient_batch_projection=Array([[.1],[.1]]))
        groups={str(layer):dict(betas=[.9,.999]) for layer in range(4)}
        for layer in range(4):
            for kind in ('raw','corrected','clipped'): values['gradient_batch_bias_'+kind+'_'+str(layer)]=Array([[0.]*384]*2)
            for when,bias,m,v,step in [('before',.5,.02,.003,14268.),('after',.4,.018,.002997,14269.)]:
                values['bias_'+str(layer)+'_'+when]=Array([[bias]*384]*2)
                for key,data in [('exp_avg',[m]*384),('exp_avg_sq',[v]*384),('step',[step])]: values['adam_'+str(layer)+'_'+key+'_'+when]=Array([data]*2)
            values['bias_'+str(layer)+'_delta']=values['bias_'+str(layer)+'_after']-values['bias_'+str(layer)+'_before']
        split=NS(identity='d'*64,arrays=dict(inputs1=Array(tokens[::-1]),inputs2=Array(tokens[::-1]),labels=Array([0,1])))
        runtime=NS(np=FakeNP,split=split,request=dict(source_revision='c'*40,source_config_sha256='b'*64,checkpoint=dict(sha256='a'*64),
            source_config=dict(seed=42),plugin_files=dict(module='e'*64),key_bias_config_sha256='f'*64))
        produced=dict(values=values,ids=ids,labels=Array([1,0]),dtypes={},adam_groups=groups,
            post_update_state_sha256='e'*64,shared_values_scope='Joint2row mean CE;not per-sample gradients',attention={})
        return runtime,produced

    def reader_module(self):
        # Execute the actual coordinate reader against injected fake arrays.
        # Loading no NumPy module and making no claim about native .npy files.
        source=(ROOT/'experiments/baseline_delivery/trace.py').read_text(encoding='utf8').replace('import numpy as np','np = injected_np')
        module=NS();ns=dict(__name__='experiments.baseline_delivery.trace',__package__='experiments.baseline_delivery',injected_np=FakeNP)
        exec(compile(source,'injected-trace-reader','exec'),ns)
        module.display=ns['display'];module.coordinate=ns['coordinate'];return module

    def test_emitter_reopens_complete_fake_values_validates_schema_and_actual_coordinate_reader(self):
        runtime,produced=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'experiments.baseline_delivery.trace':self.reader_module()}):
            result=trace.emit(runtime,Path(tmp)/'trace','fp32',produced)
            self.assertTrue(result['complete_values_reopened_exact']);self.assertTrue(result['adam_moment_equations_passed'])
            self.assertEqual(result['sample_ids'],['train:1','train:0']);self.assertEqual(result['structural_warnings'],0)
            decoded=read(result['path']);self.assertEqual(decoded['schema_version'],'sample-trace.v2')
            self.assertTrue(all('not a per-sample' in rep['meaning'] for rep in decoded['samples'][0]['representations'] if rep['id'].startswith('gradient_batch_')))

    def test_emitter_rejects_wrong_coordinates_source_labels_missing_values_and_moment_tamper(self):
        for broken in ('reader','target','missing','moment','artifact'):
            runtime,produced=self.fixture();reader=self.reader_module();original=FakeNP.load
            if broken=='reader': reader.coordinate=lambda *a:999
            if broken=='target': produced['labels']=Array([0,0])
            if broken=='missing': del produced['values']['adam_0_exp_avg_after']
            if broken=='moment': produced['values']['adam_0_exp_avg_after']=Array([[0.]*384]*2)
            def altered(path,**kwargs):
                array=original(path,**kwargs)
                if broken=='artifact': array.data[0]=999
                return array
            with self.subTest(broken=broken),tempfile.TemporaryDirectory() as tmp,\
                    patch.dict(sys.modules,{'experiments.baseline_delivery.trace':reader}),patch.object(FakeNP,'load',side_effect=altered):
                with self.assertRaises((ValueError,KeyError)): trace.emit(runtime,Path(tmp)/'trace','fp32',produced)


if __name__=='__main__': unittest.main()
