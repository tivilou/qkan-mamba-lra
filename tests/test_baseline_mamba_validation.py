"""Finite validation controller checks; no native frameworks or model runs."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import mamba_validation as v
from experiments.baseline_delivery.common import atomic_json,read


def result(mode):
    n=2 if mode in v.FP32_MODES+v.NOISE_MODES+v.PROJECTION_MODES+v.CONV_MODES else 4 if mode.endswith('_interrupt') else 8
    indices=range(4,8) if mode.endswith('_resume') else range(n)
    return dict(schema='qkan-mamba-validation-child.v1',mode=mode,request_sha256='hash',
        completed_updates=n,source_restored_exact=True,formal_updates_committed=0,
        formal_release_allowed=False,finite=True,state_sha256='mid' if n==4 else 'end',
        components={'model':'x'},starting_state_sha256='mid' if mode.endswith('_resume') else 'start',
        updates=[dict(index=i,loss=.1,row_ids_sha256='rows'+str(i),components={'model':'x'}) for i in indices])


class Tests(unittest.TestCase):
    def test_actual_child_help(self):
        s=v.spec(Path('runs/fixture'),'bf16_graph_resume')
        self.assertEqual(s['argv'][3],s['target']);self.assertIn('--output',s['argv'])
        r=subprocess.run(s['argv'][:4]+['--help'],cwd=s['cwd'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--source-suite',r.stdout)
        with self.assertRaises(ValueError):v.spec(Path('runs/fixture'),'formal')

    def test_receipt_rejects_wrong_identity_commit_or_incomplete(self):
        x=result('reference');v.accept(x,'hash','reference')
        for field,value in [('request_sha256','wrong'),('formal_updates_committed',1),
                ('completed_updates',7),('finite',False),('source_restored_exact',False)]:
            with self.assertRaises(ValueError):v.accept(dict(x,**{field:value}),'hash','reference')

    def test_recovery_checks_cursor_hash_components_and_update_sequence(self):
        c=result('bf16_eager');i=result('bf16_eager_interrupt');r=result('bf16_eager_resume')
        self.assertTrue(v.recovery_agreement(c,i,r)['exact'])
        for key,value in [('starting_state_sha256','wrong'),('components',{'model':'wrong'}),
                          ('updates',[]),('state_sha256','wrong')]:
            self.assertFalse(v.recovery_agreement(c,i,dict(r,**{key:value}))['exact'])

    def test_cleanup_leaves_original_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original=root/'original.pt';original.write_bytes(b'keep')
            suite=root/'suite';suite.mkdir();(suite/'reference-values.pt').write_bytes(b'discard')
            (suite/'bf16_graph-noise-values.pt').write_bytes(b'discard')
            (suite/'unrelated.pt').write_bytes(b'keep')
            record=v.cleanup(suite,{})
            self.assertFalse(record['remaining_owned_binaries']);self.assertTrue(original.exists())
            self.assertTrue((suite/'unrelated.pt').exists())
            self.assertFalse((suite/'bf16_graph-noise-values.pt').exists())

    def test_parent_orders_seven_children_returns_only_report_and_cleans(self):
        self.run_parent(False)

    def test_parent_failure_stops_and_still_cleans_and_returns(self):
        self.run_parent(True)

    def test_fp32_repeat_modes_and_native_route(self):
        for mode in v.FP32_MODES:
            self.assertIn(mode,v.spec(Path('suite'),mode)['argv'])
            v.accept(result(mode),'hash',mode)
            with self.assertRaises(ValueError):v.accept(dict(result(mode),completed_updates=8),'hash',mode)
            with patch.object(v.speed,'native') as native,patch.object(sys,'argv',[
                    'validate','--request','suite/request.json','--mode',mode,'--output','suite/'+mode]):
                self.assertEqual(v.main(),0)
                self.assertEqual(native.call_args.args,(Path('suite/request.json'),'fp32_eager'))

    def test_fp32_repeat_detects_state_difference_with_equal_loss(self):
        import copy
        a=result(v.FP32_MODES[0]);b=result(v.FP32_MODES[1])
        self.assertTrue(v.repeat_agreement(a,b)['exact'])
        b=copy.deepcopy(b);b['updates'][0]['components']['model']='different'
        difference=v.repeat_agreement(a,b)
        self.assertFalse(difference['exact']);self.assertEqual(difference['first_different_update_1based'],1)
        self.assertEqual(difference['steps'][0]['differing_components'],['model'])
        self.assertTrue(difference['steps'][0]['loss_equal'])
        b=result(v.FP32_MODES[1]);b['starting_state_sha256']='wrong'
        self.assertFalse(v.repeat_agreement(a,b)['exact'])
        b=result(v.FP32_MODES[1]);b['updates'][0]['step']=42
        self.assertFalse(v.repeat_agreement(a,b)['exact'])

    def test_fp32_parent_only_two_children_and_checkpoint_binding(self):
        self.run_parent(False,repeat=True)
        self.run_parent(False,repeat=True,differ=True)
        self.run_parent(True,repeat=True)
        self.run_parent(False,repeat=True,mismatch=True)

    def test_noise_parent_six_children_measurement_not_admission(self):
        self.run_parent(False,noise=True)
        self.run_parent(False,noise=True,differ=True)
        self.run_parent(True,noise=True)
        self.run_parent(False,noise=True,mismatch=True)
        for mode in v.NOISE_MODES:
            v.accept(result(mode),'hash',mode)
            with patch.object(v.speed,'native') as native,patch.object(sys,'argv',[
                    'validate','--request','suite/request.json','--mode',mode,'--output','suite/'+mode]):
                self.assertEqual(v.main(),0)
                expected='fp32_eager' if mode.startswith('fp32') else 'bf16_graph' if 'graph' in mode else 'bf16_eager'
                self.assertEqual(native.call_args.args,(Path('suite/request.json'),expected))

    def test_projection_four_children_failure_cleanup_and_native_binding(self):
        self.run_parent(False,projection=True)
        self.run_parent(False,projection=True,differ=True)
        self.run_parent(True,projection=True)
        self.run_parent(False,projection=True,mismatch=True)
        for mode in v.PROJECTION_MODES[2:]:
            with patch.object(v.speed,'native') as native,patch.object(sys,'argv',[
                    'validate','--request','suite/request.json','--mode',mode,'--output','suite/'+mode]):
                self.assertEqual(v.main(),0)
                self.assertEqual(native.call_args.args,(Path('suite/request.json'),'projection_bf16_eager'))
        command=[sys.executable,'-B','-m','experiments.baseline_delivery.mamba_validation',
            '--projection-bf16','--measure-noise']
        self.assertEqual(subprocess.run(command,capture_output=True).returncode,2)

    def test_projection_scope_dtype_original_forward_and_exception_restore(self):
        from types import SimpleNamespace
        from experiments.baseline_delivery import mamba_diagnostic as d
        enabled=[False];inner_calls=[]
        @contextlib.contextmanager
        def autocast(device,enabled=True,**kwargs):
            prior=active[0];active[0]=enabled
            try:yield
            finally:active[0]=prior
        active=enabled
        class Tensor:
            shape=(2,8,4)
            def __init__(self,dtype='fp32'):self.dtype=dtype
            def float(self):return Tensor()
            def __matmul__(self,other):return Tensor('bf16' if active[0] else 'fp32')
            def __neg__(self):return self
        def inner(value,path):
            self.assertFalse(active[0])
            self.assertEqual(value.dtype,'fp32')
            inner_calls.append(path);return Tensor()
        class Mamba:
            use_fast_path=True
            def __init__(self):
                for name in ('in_proj','conv1d','x_proj','dt_proj','out_proj'):
                    setattr(self,name,SimpleNamespace(weight=Tensor(),bias=None if name!='dt_proj' else Tensor()))
                self.A_log=Tensor();self.D=Tensor()
            def parameters(self):return [self.in_proj.weight]
            def forward(self,hidden_states,inference_params=None):return 'original'
        module=SimpleNamespace(__file__='fixture.py',Mamba=Mamba,causal_conv1d_fn=True,
            rearrange=lambda value,*args,**kwargs:value,original_inner=inner,
            mamba_inner_fn=lambda *args,**kwargs:inner(args[0],'fused'))
        source='''def forward(self,hidden_states,inference_params=None):
    xz=self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)")
    path='fused' if self.use_fast_path and causal_conv1d_fn is not None else 'original_unfused'
    return original_inner(xz,path)
'''
        torch=SimpleNamespace(float32='fp32',bfloat16='bf16',exp=lambda x:x,autocast=autocast)
        mixer=Mamba();model=SimpleNamespace(blocks=[SimpleNamespace(mixer=mixer)])
        weight=mixer.in_proj.weight
        with patch('importlib.import_module',return_value=module),\
                patch('inspect.getsource',return_value=source),\
                patch.object(d,'digest',return_value=d.InputProjectionBF16.source_sha256):
            policy=d.InputProjectionBF16(torch,model)
            self.assertEqual(mixer.forward(Tensor()),'original')
            with policy.activate():self.assertEqual(mixer.forward(Tensor()).dtype,'fp32')
            self.assertEqual(policy.calls,1);self.assertEqual(len(inner_calls),1)
            self.assertIs(mixer.in_proj.weight,weight)
            with self.assertRaisesRegex(ValueError,'FP32 training inputs'):
                with policy.activate():mixer.forward(Tensor('bf16'))
            self.assertFalse(policy.active);self.assertEqual(mixer.forward(Tensor()),'original')
            policy.close();self.assertNotIn('forward',mixer.__dict__)
            self.assertEqual(inner_calls,['fused'])
            for flag,convolution in ((False,True),(True,None)):
                mixer.use_fast_path=flag;module.causal_conv1d_fn=convolution
                policy=d.InputProjectionBF16(torch,model)
                with policy.activate():self.assertEqual(mixer.forward(Tensor()).dtype,'fp32')
                self.assertEqual(inner_calls[-1],'original_unfused')
                self.assertEqual(policy.mechanism['inner_paths'],['original_unfused'])
                self.assertEqual(mixer.use_fast_path,flag);self.assertIs(module.causal_conv1d_fn,convolution)
                policy.close();self.assertNotIn('forward',mixer.__dict__)
            mixer.in_proj.weight=Tensor('bf16')
            with self.assertRaisesRegex(ValueError,'FP32 parameters'):d.InputProjectionBF16(torch,model)
            mixer.in_proj.weight=weight
            with patch('inspect.getsource',return_value=source.replace(' @ ',' * ')):
                with self.assertRaisesRegex(ValueError,'expression boundary'):d.InputProjectionBF16(torch,model)
        with patch('importlib.import_module',return_value=module),patch.object(d,'digest',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'pinned Mamba source'):d.InputProjectionBF16(torch,model)

    def test_noise_tensor_metrics_known_vectors_and_zero_reference(self):
        import math
        from types import SimpleNamespace
        class Vector:
            dtype='float32'
            def __init__(self,values):self.values=list(values);self.shape=(len(self.values),)
            def is_floating_point(self):return True
            def double(self):return self
            def flatten(self):return self
            def norm(self):return math.sqrt(sum(x*x for x in self.values))
            def sum(self):return sum(self.values)
            def numel(self):return len(self.values)
            def abs(self):return Vector(abs(x) for x in self.values)
            def max(self):return max(self.values)
            def __sub__(self,other):return Vector(x-y for x,y in zip(self.values,other.values))
            def __mul__(self,other):return Vector(x*y for x,y in zip(self.values,other.values))
        torch=SimpleNamespace(is_tensor=lambda x:isinstance(x,Vector),
            isfinite=lambda x:SimpleNamespace(all=lambda:all(math.isfinite(v) for v in x.values)))
        a={'weight':Vector([3,4]),'none':None,'group':{'lr':.1}}
        b={'weight':Vector([0,4]),'none':None,'group':{'lr':.1}}
        metric=v.tensor_errors(torch,a,b)
        self.assertEqual(metric['difference_norm'],3);self.assertEqual(metric['relative_l2'],.6)
        self.assertEqual(metric['max_absolute'],3);self.assertEqual(metric['cosine'],.8)
        self.assertTrue(metric['nonfloating_controls_equal'])
        zero=v.tensor_errors(torch,Vector([0,0]),Vector([1,0]))
        self.assertIsNone(zero['relative_l2']);self.assertTrue(zero['zero_reference'])
        self.assertEqual(zero['difference_norm'],1)
        b['group']['lr']=.2;self.assertFalse(v.tensor_errors(torch,a,b)['nonfloating_controls_equal'])
        with self.assertRaises(ValueError):v.tensor_errors(torch,a,{})
        with self.assertRaises(ValueError):v.tensor_errors(torch,Vector([float('inf')]),Vector([0]))

    def test_noise_completion_requires_each_comparison_and_invariants(self):
        children={}
        for mode in v.NOISE_MODES:
            names=({'repeat'} if mode.endswith('_b') else set())|({'vs_fp32_a'} if mode.startswith('bf16') else set())
            children[mode]={'numerical_comparisons':{k:dict(starting_scientific_state_exact=True,
                invariants_exact=True,steps=[{},{}]) for k in names}}
        self.assertTrue(v.measurements_complete(children))
        children['bf16_graph_noise_b']['numerical_comparisons']['repeat']['invariants_exact']=False
        self.assertFalse(v.measurements_complete(children))
        children['bf16_graph_noise_b']['numerical_comparisons'].pop('repeat')
        self.assertFalse(v.measurements_complete(children))
        suite=Path('suite')
        with patch.object(sys,'argv',['validate','--fp32-repeat','--measure-noise']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:v.main()
        self.assertEqual(e.exception.code,2)

    def run_parent(self,fail,repeat=False,differ=False,mismatch=False,noise=False,projection=False,conv=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';cache=root/'cache';calls=[];exports=[]
            def snapshot(source,suite,job,cache):
                request={'source_files':{},'checkpoint':{'sha256':'checkpoint','file':'fixture.pt'}}
                atomic_json(suite/'request.json',request)
                (suite/'reference-values.pt').write_bytes(b'private')
                return request
            class Supervisor:
                records=[]
                def __init__(self,*args):pass
                def run(self,spec,log,env):
                    self_outer.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
                    mode=spec['argv'][spec['argv'].index('--mode')+1];calls.append(mode)
                    if fail:return 1
                    receipt=result(mode);receipt['request_sha256']=v.digest(log.parent/'request.json')
                    if differ and mode==v.FP32_MODES[1]:receipt['updates'][0]['components']={'model':'different'}
                    if noise or projection or conv:
                        names=({'repeat'} if mode.endswith('_b') else set())|({'vs_fp32_a'} if v.native_route(mode)!='fp32_eager' else set())
                        receipt['numerical_comparisons']={k:dict(starting_scientific_state_exact=True,
                            invariants_exact=not differ,steps=[{},{}]) for k in names}
                    if mode.startswith('projection_'):
                        receipt['projection_mechanism']={'observed_projection_calls':1}
                        receipt['paired_timing']={'summary':{}} if mode.endswith('_b') else None
                    if mode.startswith('conv_'):
                        receipt['convolution_mechanism']={'observed_fused_calls':1}
                        receipt['paired_timing']={'summary':{}} if mode.endswith('_b') else None
                    atomic_json(log.parent/mode/'result.json',receipt);return 0
            self_outer=self
            review=root/'review.json';atomic_json(review,{'routes':{'bf16_eager':{'timing':{'timing_promising':True}}}})
            repeat_review=root/'repeat-review.json';atomic_json(repeat_review,{'evidence':{
                'source_checkpoint_sha256':'wrong' if mismatch else 'checkpoint'}})
            with patch.object(v,'ROOT',root),patch.object(v,'REVIEW',review),\
                 patch.object(v,'REPEAT_REVIEW',repeat_review),\
                 patch('experiments.baseline_delivery.mamba_convolution.stage',return_value={'root':'isolated-fixture'}),\
                 patch.object(v.speed,'snapshot',side_effect=snapshot),\
                 patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                 patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'gpu'})),\
                 patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}),\
                 patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',Supervisor),\
                 patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:exports.append(a)),\
                 patch.object(sys,'argv',['validate','--source-suite',str(source),'--cache',str(cache)]+(['--fused-fp32'] if conv else ['--projection-bf16'] if projection else ['--measure-noise'] if noise else ['--fp32-repeat'] if repeat else [])),\
                 contextlib.redirect_stdout(io.StringIO()):
                if mismatch:
                    with self.assertRaisesRegex(ValueError,'checkpoint differs'):v.main()
                else:self.assertEqual(v.main(),76 if fail or differ else 0)
            modes=v.CONV_MODES if conv else v.PROJECTION_MODES if projection else v.NOISE_MODES if noise else v.FP32_MODES if repeat else v.MODES
            self.assertEqual(calls,[] if mismatch else [modes[0]] if fail else list(modes))
            filename='mamba-convolution.json' if conv else 'mamba-projection.json' if projection else 'mamba-noise.json' if noise else 'mamba-repeat.json' if repeat else 'mamba-validation.json'
            self.assertEqual(exports[0][2],[filename])
            report=read(exports[0][0]/filename)
            self.assertFalse((exports[0][0]/'reference-values.pt').exists())
            self.assertFalse(report['formal_release_allowed'])
            if not mismatch:self.assertTrue(report['source_unchanged'])
            if repeat or noise or projection or conv:self.assertEqual(report['prefix_steps'],2)
            if (noise or projection or conv) and not mismatch:
                self.assertEqual(report['measurement_completed'],not fail and not differ)
                self.assertIn('not_evaluated',report['numeric_acceptance'])


if __name__=='__main__':unittest.main()
