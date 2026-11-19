"""CPU postprocessor controls and independent scalar fixtures; no native imports."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import attention_bias_postprocess as b
from experiments.baseline_delivery import bootstrap,diagnostic_watch,return_artifacts
from experiments.baseline_delivery.common import atomic_json,read


class Values:
    def __init__(self,values):
        self.values=list(values);self.shape=(len(self.values),)
        self.device=NS(type='cpu');self.dtype='torch.float32'
    def detach(self): return self
    def double(self): return self
    def reshape(self,*args): return self
    def tolist(self): return self.values.copy()


def fixture():
    policy=dict(parameters=list(b.NAMES),lr=.001,eps=1e-8,betas=[.9,.999],weight_decay=.01,amsgrad=False,maximize=False)
    pairs={}
    for route in ('fp32','bf16'):
        r={f:{} for f in b.FIELDS}
        r.update(row_ids_sha256='same-rows',optimizer_groups=[copy.deepcopy(policy)],
            optimizer_steps_before={n:1. for n in b.NAMES},optimizer_steps_after={n:2. for n in b.NAMES})
        for name in b.NAMES:
            g=[.1]*128+([0.]*128 if route=='fp32' else [1e-6]*128)+[.2]*128
            m=[.1*x for x in g];v=[.001*x*x for x in g]
            u=[-.001*(x/(1-.9**2))/(math.sqrt(y/(1-.999**2))+1e-8)-1e-6 for x,y in zip(m,v)]
            for f,vals in [('gradients',g),('clipped_gradients',g),('deltas',u),
                           ('exp_avg_before',[0.]*384),('exp_avg_after',m),
                           ('exp_avg_sq_before',[0.]*384),('exp_avg_sq_after',v)]:
                r[f][name]=Values(vals)
        pairs[route]=r
    fields={}
    for f in b.FIELDS:
        rows={}
        for name in b.NAMES:
            a=pairs['fp32'][f][name].values;c=pairs['bf16'][f][name].values
            rows[name]=dict(reference_squared_l2=math.fsum(x*x for x in a),
                candidate_squared_l2=math.fsum(x*x for x in c),difference_squared_l2=math.fsum((x-y)**2 for x,y in zip(a,c)))
        fields[f]=dict(parameters=rows,total={k:math.fsum(v[k] for v in rows.values()) for k in rows[b.NAMES[0]]})
    summary=dict(fields=fields,optimizer_groups=[policy],
        optimizer_steps_before=pairs['fp32']['optimizer_steps_before'],optimizer_steps_after=pairs['fp32']['optimizer_steps_after'])
    return pairs,dict(results=dict(attribution=dict(summary=summary,row_ids_sha256='same-rows')))


class MathTests(unittest.TestCase):
    def test_qkv_cover_exactly_384_disjoint_elements(self):
        indices=[i for lo,hi in b.SLICES.values() for i in range(lo,hi)]
        self.assertEqual(indices,list(range(384)))
        self.assertEqual([hi-lo for lo,hi in b.SLICES.values()],[128]*3)

    def test_known_adam_bias_correction_and_moment_equations(self):
        p=dict(betas=[.9,.999],step_after=2,eps=1e-8,lr=.001)
        v=dict(exp_avg_before=[.1],exp_avg_after=[.19],exp_avg_sq_before=[.001],
            exp_avg_sq_after=[.001999],clipped_gradients=[1.])
        c=b.components(v,p)
        self.assertAlmostEqual(c['denominator'][0],1.+1e-8,places=12)
        self.assertAlmostEqual(c['adaptive_update'][0],-.001/(1+1e-8),places=14)
        self.assertLess(c['first_moment_equation']['difference']['l2'],1e-15)
        self.assertLess(c['second_moment_equation']['difference']['l2'],1e-15)

    def test_known_key_only_error_is_reconstructed_by_layer_and_pool(self):
        pairs,prior=fixture();r=b.analyze(pairs,prior)
        self.assertEqual(r['packed_bias_error_energy_share'],1.)
        self.assertEqual(r['coverage']['packed_elements'],1536)
        for part,share in [('Q',0.),('K',1.),('V',0.)]:
            row=r['pooled_qkv'][part]
            self.assertAlmostEqual(row['global_error_energy_share'],share)
            self.assertEqual(row['fields']['deltas']['reference']['numel'],512)
            self.assertLess(row['observed_vs_adaptive_route_difference']['difference']['l2'],1e-15)
        for layer in r['layers'].values():
            self.assertAlmostEqual(layer['slices']['K']['global_error_energy_share'],.25)
        self.assertLess(len(json.dumps(r,allow_nan=False)),512*1024)

    def test_reference_bins_cover_all_coordinates_even_when_candidate_moves(self):
        pairs,prior=fixture();r=b.analyze(pairs,prior)['pooled_qkv']['K']
        bands=r['reference_denominator_bands']
        self.assertEqual([v['count'] for v in bands],[512,0,0,0])
        self.assertAlmostEqual(sum(v['global_error_energy_share'] for v in bands),1.)
        self.assertGreater(r['denominator_candidate_over_reference']['min'],2.)

    def test_zero_reference_flag_and_empty_bin_null_quantiles(self):
        self.assertTrue(b.paired([0],[1])['denominator_clamped'])
        self.assertEqual(b.stats([])['absolute_quantiles'],dict.fromkeys(['min','p10','p50','p90','max']))
        with self.assertRaises(ValueError): b.paired([float('nan')],[0])

    def test_changed_population_policy_counter_and_initial_summary_stop(self):
        for kind in ('rows','policy','steps','values'):
            pairs,prior=fixture();r=pairs['bf16']
            if kind=='rows': r['row_ids_sha256']='changed'
            if kind=='policy': r['optimizer_groups'][0]['eps']=1e-6
            if kind=='steps': r['optimizer_steps_after'][b.NAMES[0]]=3
            if kind=='values': r['exp_avg_before'][b.NAMES[0]].values[0]=1
            with self.subTest(kind=kind),self.assertRaises(ValueError): b.analyze(pairs,prior)

    def test_bad_variance_dtype_shape_and_missing_values_stop(self):
        for kind in ('negative','dtype','shape','missing','device'):
            pairs,prior=fixture();t=pairs['bf16']['exp_avg_sq_after'][b.NAMES[0]]
            if kind=='negative': t.values[0]=-1
            if kind=='dtype': t.dtype='torch.float64'
            if kind=='shape': t.shape=(128,3)
            if kind=='missing': del pairs['bf16']['deltas'][b.NAMES[0]]
            if kind=='device': t.device.type='cuda'
            with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)): b.analyze(pairs,prior)

    def test_unsupported_amsgrad_maximize_or_nonintegral_step_stop(self):
        for kind in ('amsgrad','maximize','step'):
            pairs,_=fixture();r=pairs['fp32']
            if kind=='step': r['optimizer_steps_after'][b.NAMES[0]]=2.5
            else: r['optimizer_groups'][0][kind]=True
            with self.subTest(kind=kind),self.assertRaises(ValueError): b.group_for(r,b.NAMES[0])


class ControlTests(unittest.TestCase):
    def test_pinned_wrong_file_stops_before_native_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'attribution').mkdir()
            (root/'attention-attribution.json').write_text('{}')
            (root/'attribution/paired-parameters.pt').write_bytes(b'wrong')
            with self.assertRaises(ValueError): b.execute_cpu(root,root)
            self.assertFalse((root/'result.json').exists())

    def test_cpu_worker_load_is_restricted_and_does_no_gpu_work(self):
        pairs,prior=fixture();binding={'hash':'pinned'};calls=[]
        def load(path,**kwargs):
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(kwargs,dict(map_location='cpu',weights_only=True))
            calls.append(path);return pairs
        fake=NS(load=load,set_num_threads=lambda n:self.assertEqual(n,1),__version__='fixture')
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,torch=fake),patch.dict(os.environ),\
                patch.object(b,'bind',return_value=(prior,Path('private.pt'),binding)):
            r=b.execute_cpu(Path(tmp),Path(tmp));saved=read(Path(tmp)/'result.json')
            self.assertEqual(saved,r);self.assertEqual(len(calls),1)
            self.assertEqual(r['execution']['optimizer_steps'],0)
            self.assertEqual(r['input_integrity']['source_files_written'],0)
            self.assertFalse(r['candidate_admitted'])
            self.assertNotIn('values',json.dumps(r))

    def invoke_parent(self,root,*,code=0,upload_error=False,changed=False):
        binding={'hash':'same'};calls=[];snapshot={}
        class Supervisor:
            def __init__(self,*args): self.records=[]
            def run(self,spec,log,env):
                self.records.append({'exit_code':code,'termination':{'group_verified':True}})
                self_spec=spec;snapshot.update(env=env,spec=self_spec)
                args=spec['argv'];output=Path(args[args.index('--output')+1])
                atomic_json(output/'result.json',dict(schema=b.SCHEMA,binding=binding,
                    postprocess_complete=True,candidate_admitted=False))
                return code
        def upload(folder,experiment,files):
            calls.append((folder,experiment,files))
            self.assertEqual(files,['attention-bias.json'])
            if upload_error:
                atomic_json(folder/'artifact-delivery.json',{'status':'pending'})
                raise OSError('fixture transport failure')
        effects=[({},Path('private.pt'),binding),ValueError('changed')] if changed else None
        with patch.object(b,'ROOT',root),patch.object(b,'revision',return_value='new'),\
                patch.object(b,'bind',side_effect=effects,return_value=({},Path('private.pt'),binding)),\
                patch.object(bootstrap,'check_exchange'),\
                patch.object(diagnostic_watch,'Supervisor',Supervisor),\
                patch.object(diagnostic_watch,'spawn',side_effect=AssertionError('Unexpected real child launch')),\
                patch.object(return_artifacts,'upload_declared',side_effect=upload),\
                patch.object(sys,'argv',['cpu','--attribution-suite',str(root/'prior')]):
            try: snapshot['status']=b.main()
            except Exception as e: snapshot['error']=type(e).__name__
        snapshot['calls']=calls
        snapshot['report']=read(next((root/'runs').glob('*/attention-bias.json')))
        return snapshot

    def test_parent_supervision_env_success_and_only_compact_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=self.invoke_parent(Path(tmp))
            self.assertEqual(s['status'],0);self.assertEqual(len(s['calls']),1)
            self.assertEqual(s['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(s['env']['OMP_NUM_THREADS'],'1')
            self.assertIn('--worker',s['spec']['argv'])
            self.assertTrue(s['report']['postprocess_complete'])
            self.assertFalse(s['report']['formal_training_started'])

    def test_failed_child_report_is_returned_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=self.invoke_parent(Path(tmp),code=2)
            self.assertEqual(s['error'],'ValueError');self.assertEqual(len(s['calls']),1)
            self.assertFalse(s['report']['postprocess_complete'])
            self.assertEqual(s['report']['child_executions'][0]['exit_code'],2)

    def test_upload_failure_retains_result_and_pending_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=self.invoke_parent(Path(tmp),upload_error=True)
            self.assertEqual(s['error'],'OSError')
            self.assertTrue(s['report']['postprocess_complete'])
            self.assertTrue((s['calls'][0][0]/'artifact-delivery.json').exists())

    def test_evidence_changed_after_worker_invalidates_completed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=self.invoke_parent(Path(tmp),changed=True)
            self.assertEqual(s['error'],'ValueError')
            self.assertFalse(s['report']['postprocess_complete'])
            self.assertFalse(s['report']['input_integrity']['before_after_sha256_equal'])


if __name__=='__main__': unittest.main()
