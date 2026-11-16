"""Injected leaf hooks and scalar controls; real model tests stay on collaborator."""
import copy
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import key_bias_consistency as k,key_bias_diagnostic as e
from experiments.baseline_delivery import attention_attribution as a
from experiments.baseline_delivery import bootstrap,diagnostic_watch,return_artifacts
from experiments.baseline_delivery.common import ROOT,atomic_json,digest,read
from tests.test_baseline_attention_attribution import Vector


class Tensor(Vector):
    dtype='fp32'
    def __init__(self,values,declared_numel=None):
        super().__init__(values);self.declared_numel=declared_numel
    def clone(self): return copy.deepcopy(self)
    def __getitem__(self,key):
        return Tensor(self.values[key]) if isinstance(key,slice) else self.values[key]
    def __setitem__(self,key,value):
        if isinstance(key,slice):
            indices=list(range(len(self.values)))[key]
            for i in indices: self.values[i]=value
        else: self.values[key]=value
    def __ne__(self,value): return Tensor([v!=value for v in self.values])
    def any(self): return any(self.values)
    def numel(self): return self.declared_numel or len(self.values)


class Parameter(Tensor):
    requires_grad=True
    def __init__(self): super().__init__([0.]*384);self._backward_hooks={};self.fail_registration=False
    def register_hook(self,callback):
        if self.fail_registration: raise RuntimeError('fixture registration failure')
        index=len(self._backward_hooks);self._backward_hooks[index]=callback
        return NS(remove=lambda:self._backward_hooks.pop(index,None))
    def emit(self,gradient):
        for callback in self._backward_hooks.values(): gradient=callback(gradient)
        return gradient


def hook_fixture():
    params={n:Parameter() for n in k.NAMES}
    layers=[NS(self_attn=NS(in_proj_bias=params[n],embed_dim=128,num_heads=4,
        bias_k=None,bias_v=None,add_zero_attn=False)) for n in k.NAMES]
    return NS(named_parameters=lambda:list(params.items()),encoder=NS(layers=layers)),params


TORCH=NS(float32='fp32',equal=lambda x,y:x.values==y.values,
    isfinite=lambda v:Vector(math.isfinite(x) for x in v.values))


def routes():
    def make(bf16):
        result={f:{} for f in a.FIELDS}
        result.update(logits=Tensor([1.0001 if bf16 else 1.,0.]),loss=1.001 if bf16 else 1.,correct=1,
            rows=128,row_ids_sha256='fixed128',optimizer_groups=[dict(parameters=list(k.NAMES),lr=.001,eps=1e-8,betas=[.9,.999],weight_decay=.01,amsgrad=False,maximize=False)],
            optimizer_steps_before={n:14268. for n in k.NAMES},optimizer_steps_after={n:14269. for n in k.NAMES})
        for n in k.NAMES:
            values=dict(gradients=[.1]*128+[1e-5 if bf16 else 1e-9]*128+[.2]*128,
                clipped_gradients=[.1]*128+[1e-5 if bf16 else 1e-9]*128+[.2]*128,
                deltas=[.01]*128+[.002 if bf16 else 1e-6]*128+[.02]*128,
                exp_avg_before=[.005]*384,exp_avg_sq_before=[.006]*384,
                exp_avg_after=[.0045]*128+[.004501 if bf16 else .0045000001]*128+[.0045]*128,
                exp_avg_sq_after=[.005994]*128+[.0059940000001 if bf16 else .005994]*128+[.005994]*128)
            for f,v in values.items(): result[f][n]=Tensor(v)
        for i in range(49):
            for f in a.FIELDS: result[f]['other.'+str(i)]=Tensor([.03,.04],805538 if i==0 else 2)
        return result
    return dict(fp32=make(False),bf16=make(True))


class HookTests(unittest.TestCase):
    def test_disabled_path_does_not_inspect_model_or_runtime(self):
        with k.analytic_key_bias(object(),object()) as r: self.assertFalse(r['enabled'])
        self.assertFalse(k.resolve());self.assertFalse(k.resolve(dict(schema=k.SCHEMA,enabled=False,gradient_policy=k.GRADIENT_POLICY)))

    def test_config_is_explicit_and_strict(self):
        value=read(e.PRESET);self.assertTrue(k.resolve(value))
        for broken in [dict(value,enabled=1),dict(value,gradient_policy='reset_adam'),dict(value,extra=True)]:
            with self.assertRaises(ValueError): k.resolve(broken)

    def test_zero_key_preserves_qv_raw_input_identity_and_parameter_objects(self):
        model,params=hook_fixture();identity={n:id(p) for n,p in params.items()};raw=Tensor(range(384))
        with k.analytic_key_bias(TORCH,model,True) as r:
            for p in params.values():
                for _ in range(8):
                    out=p.emit(raw)
                    self.assertEqual(out.values[:128],list(range(128)))
                    self.assertEqual(out.values[128:256],[0]*128)
                    self.assertEqual(out.values[256:],list(range(256,384)))
                    self.assertEqual(raw.values,list(range(384)))
        self.assertTrue(k.validate_observation(r));self.assertTrue(all(not p._backward_hooks for p in params.values()))
        self.assertEqual({n:id(p) for n,p in params.items()},identity)

    def test_registration_failure_and_backward_exception_remove_owned_handles(self):
        for registration in (True,False):
            model,params=hook_fixture()
            if registration: params[k.NAMES[2]].fail_registration=True
            with self.assertRaises(RuntimeError):
                with k.analytic_key_bias(TORCH,model,True):
                    raise RuntimeError('fixture backward failure')
            self.assertTrue(all(not p._backward_hooks for p in params.values()))

    def test_existing_hook_other_instance_and_invalid_scope_are_preserved(self):
        model,params=hook_fixture();other,other_params=hook_fixture()
        params[k.NAMES[0]]._backward_hooks[99]=lambda g:g
        with self.assertRaises(ValueError):
            with k.analytic_key_bias(TORCH,model,True): pass
        self.assertIn(99,params[k.NAMES[0]]._backward_hooks)
        self.assertTrue(all(not p._backward_hooks for p in other_params.values()))
        other.encoder.layers[0].self_attn.bias_k='extra-key'
        with self.assertRaises(ValueError):
            with k.analytic_key_bias(TORCH,other,True): pass

    def test_nonfinite_gradient_excess_or_missing_callbacks_fail(self):
        model,params=hook_fixture()
        with self.assertRaises(ValueError):
            with k.analytic_key_bias(TORCH,model,True): params[k.NAMES[0]].emit(Tensor([float('nan')]*384))
        with self.assertRaises(ValueError):
            with k.analytic_key_bias(TORCH,model,True):
                for _ in range(9): params[k.NAMES[0]].emit(Tensor([1.]*384))
        with k.analytic_key_bias(TORCH,model,True) as r: pass
        with self.assertRaises(ValueError): k.validate_observation(r)

    def test_exact_attention_common_key_bias_has_zero_derivative_even_with_mask(self):
        # Independent scalar softmax identity, including a fixed excluded key.
        scores=[.2,-.3,1.1];upstream=[.4,-.2,.8];visible=[True,False,True]
        def probabilities(shift):
            weights=[math.exp(s+shift) if ok else 0 for s,ok in zip(scores,visible)]
            return [w/sum(weights) for w in weights]
        p=probabilities(0);shifted=probabilities(3.)
        for x,y in zip(p,shifted): self.assertAlmostEqual(x,y,places=14)
        mean=sum(x*y for x,y in zip(p,upstream))
        derivative=sum(x*(u-mean) for x,u in zip(p,upstream))
        self.assertAlmostEqual(derivative,0.,places=14)

    def test_zero_gradient_keeps_momentum_variance_step_and_weight_decay_active(self):
        # Independent AdamW scalar: g=0 still carries m/v and updates the weight.
        weight,m,v,step,lr,wd=2.,.03,.004,14268,.001,.01
        next_m=.9*m;next_v=.999*v;next_step=step+1
        update=-lr*(next_m/(1-.9**next_step))/(math.sqrt(next_v/(1-.999**next_step))+1e-8)
        after=weight*(1-lr*wd)+update
        self.assertNotEqual(after,weight);self.assertNotEqual(next_m,0);self.assertNotEqual(next_v,0)
        self.assertEqual(next_step,14269)


class ComparisonTests(unittest.TestCase):
    def test_saved_request_is_bound_to_original_source_and_checkpoint(self):
        binding={key:'original-'+key for key in e.BINDING_KEYS}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);path=folder/'update-input.json';atomic_json(path,binding)
            prior={'results':{'attribution':{'request_sha256':digest(path)}}}
            with patch('experiments.baseline_delivery.attention_bias_postprocess.bind',return_value=(prior,Path('private.pt'),{'verified':True})):
                identity,values=e.bind_evidence(folder,binding)
                self.assertTrue(identity['verified']);self.assertEqual(values,Path('private.pt'))
                changed=dict(binding,checkpoint='advanced')
                with self.assertRaises(ValueError): e.bind_evidence(folder,changed)

    def test_saved_forward_initial_history_and_all_parameter_population_are_bound(self):
        previous=routes()['fp32'];same=copy.deepcopy(previous)
        self.assertTrue(e.compare_unchanged(TORCH,previous,same,corrected=False)['forward_exact'])
        for kind in ('logits','history','population','non_k'):
            same=copy.deepcopy(previous)
            if kind=='logits': same['logits'].values[0]+=1
            if kind=='history': same['exp_avg_before'][k.NAMES[0]].values[128]+=1
            if kind=='population': del same['gradients']['other.0']
            if kind=='non_k': same['gradients'][k.NAMES[0]].values[0]+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError): e.compare_unchanged(TORCH,previous,same,corrected=True)

    def test_dossier_plan_and_gradient_policy_are_bound_before_native_import(self):
        request=dict(schema='qkan-attention-update-input.v1',source_revision='fixture',policy=e.POLICY,
            source_config={},source_config_sha256=e.fingerprint({}),optimization_config=read(e.BF16),
            key_bias_config=read(e.PRESET),key_bias_manifest=k.manifest(),dossier_sha256=digest(e.DOSSIER),
            plugin_plan_sha256=digest(e.PLAN),plugin_files={n:digest(Path(e.__file__).parent/n) for n in e.FILES})
        self.assertTrue(e.validate_screen(request,'fixture'))
        for key in ('dossier_sha256','key_bias_manifest','plugin_files'):
            broken=copy.deepcopy(request);broken[key]='changed'
            with self.assertRaises(ValueError): e.validate_screen(broken,'fixture')


class ScreenTests(unittest.TestCase):
    def run_screen(self,folder,fail_update=False,corrupt_reopen=False):
        previous=routes();model,params=hook_fixture();saved={};calls=[]
        def save(value,path): saved['pairs']=copy.deepcopy(value);Path(path).write_bytes(pickle.dumps(value))
        def load(path,**kwargs):
            self.assertEqual(kwargs,dict(map_location='cpu',weights_only=True))
            value=pickle.loads(Path(path).read_bytes())
            if corrupt_reopen: value['routes']['fp32']['loss']=999
            return value
        torch=NS(**vars(TORCH),save=save,load=load)
        class Runtime:
            t=torch;base={'cursor':14268}
            def __init__(self): self.model=model;self.state=copy.deepcopy(self.base)
            def restore(self,value): self.state=copy.deepcopy(value)
            def snapshot(self): return self.state
            def clone(self,value): return copy.deepcopy(value)
            def state_hash(self,value): return hashlib.sha256(pickle.dumps(value)).hexdigest()
            def metrics(self,left,right):
                s=a.compare(TORCH,left,right)
                return dict(logits_relative_l2=.0001,loss_absolute=abs(left['loss']-right['loss']),
                    gradient_relative_l2=s['fields']['gradients']['total']['relative_l2'],
                    adam_delta_relative_l2=s['fields']['deltas']['total']['relative_l2'])
        runtime=Runtime()
        def capture(rt,enabled):
            calls.append(enabled);rt.state={'cursor':14269}
            if fail_update and len(calls)==2: raise RuntimeError('fixture native update error')
            result=copy.deepcopy(previous['bf16' if enabled else 'fp32'])
            if any(p._backward_hooks for p in params.values()):
                for name,p in params.items():
                    for _ in range(8): p.emit(result['gradients'][name])
                    for f in ('gradients','clipped_gradients'): result[f][name][128:256]=0
                    result['exp_avg_after'][name][128:256]=.0045
                    result['exp_avg_sq_after'][name][128:256]=.005994
                    result['deltas'][name][128:256]=2e-7
            return result
        with patch.object(a,'capture_update',side_effect=capture):
            try: saved['result']=e.screen(runtime,previous,folder)
            except Exception as error: saved['error']=type(error).__name__
        saved.update(runtime=runtime,params=params,calls=calls)
        return saved

    def test_three_routes_retain_history_full_metrics_private_reopen_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=self.run_screen(Path(tmp));r=s['result']
            self.assertEqual(s['calls'],[False,False,True]);self.assertTrue(r['screen_complete'])
            self.assertTrue(all(r['numerical_passed'].values()));self.assertFalse(r['candidate_admitted'])
            self.assertTrue(r['private_values']['complete_values_reopened_exact'])
            self.assertEqual(r['global_fields']['bf16_analytic']['deltas']['parameter_count'],53)
            self.assertEqual(r['global_fields']['bf16_analytic']['deltas']['numel'],807170)
            self.assertEqual(s['runtime'].snapshot(),s['runtime'].base)
            self.assertTrue(all(not p._backward_hooks for p in s['params'].values()))

    def test_update_exception_and_reopen_corruption_restore_exact_state_and_hooks(self):
        for update,reopen in [(True,False),(False,True)]:
            with tempfile.TemporaryDirectory() as tmp:
                s=self.run_screen(Path(tmp),update,reopen)
                self.assertIn(s['error'],('RuntimeError','ValueError'))
                r=read(Path(tmp)/'screen.json');self.assertFalse(r['screen_complete'])
                self.assertEqual(s['runtime'].snapshot(),s['runtime'].base)
                self.assertTrue(all(not p._backward_hooks for p in s['params'].values()))
                self.assertEqual(r['rollback']['committed_updates'],0)

    def test_accept_recomputes_global_gate_and_never_grants_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=self.run_screen(Path(tmp))['result']
            r.update(schema=e.SCHEMA,request_sha256='request',checkpoint_sha256='checkpoint',key_bias_manifest=k.manifest(),
                formal_training_started=False,training_samples_committed=0,native_trace_ready=False)
            self.assertEqual(e.accept(r,dict(checkpoint={'sha256':'checkpoint'}),'request',0),r)
            r['numerical']['bf16_analytic']['adam_delta_relative_l2']=.21
            with self.assertRaises(ValueError): e.accept(r,dict(checkpoint={'sha256':'checkpoint'}),'request',0)
            r['numerical_passed']['bf16_analytic']=False
            r['global_fields']['bf16_analytic']['deltas']['relative_l2']=.21
            self.assertIs(e.accept(r,dict(checkpoint={'sha256':'checkpoint'}),'request',0)['numerical_passed']['bf16_analytic'],False)

    def test_gpu_busy_rejects_before_source_or_native_child(self):
        with patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{'uuid':'fixture'}})),\
                patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':False}),\
                patch.object(diagnostic_watch.Supervisor,'run',side_effect=AssertionError('Unexpected launch')):
            with self.assertRaisesRegex(ValueError,'GPU busy'): e.execute(NS(gpu='0'),Path('unused'),{})

    def parent(self,folder,fail=False,upload_fail=False):
        def execute(args,suite,report):
            if fail: raise RuntimeError('fixture child failure')
            report.update(screen_complete=True,numerical_passed={'fp32_analytic':True,'bf16_analytic':False})
        def upload(suite,experiment,files):
            self.assertEqual(experiment,'key-bias');self.assertEqual(files,['key-bias.json'])
            if upload_fail:
                atomic_json(suite/'artifact-delivery.json',{'status':'pending'});raise OSError('fixture transport error')
        with patch.object(e,'ROOT',folder),patch.object(e,'execute',side_effect=execute),\
                patch.object(e,'revision',return_value='new'),patch.object(bootstrap,'check_exchange'),\
                patch.object(return_artifacts,'upload_declared',side_effect=upload),\
                patch.object(sys,'argv',['screen','--source-suite','original','--cache','cache','--attribution-from','prior','--optimizations',str(e.PRESET)]):
            try: result=e.main()
            except Exception as error: result=type(error).__name__
        suite=next((folder/'runs').iterdir())
        return result,read(suite/'key-bias.json'),suite

    def test_parent_returns_compact_failed_and_rejected_results_without_false_admission(self):
        for failed in (True,False):
            with tempfile.TemporaryDirectory() as tmp:
                status,r,_=self.parent(Path(tmp),failed)
                self.assertFalse(r['candidate_admitted']);self.assertFalse(r['native_trace_ready'])
                self.assertEqual(status,'RuntimeError' if failed else 0)
                self.assertEqual(r['screen_complete'],not failed)

    def test_upload_failure_retains_completed_screen_and_pending_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            status,r,suite=self.parent(Path(tmp),upload_fail=True)
            self.assertEqual(status,'OSError');self.assertTrue(r['screen_complete'])
            self.assertTrue((suite/'artifact-delivery.json').is_file())


if __name__=='__main__': unittest.main()
