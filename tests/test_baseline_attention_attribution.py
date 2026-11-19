"""Stdlib contract/control tests; native measurements stay on collaborator GPU."""
import copy
from contextlib import contextmanager
import json
import math
import pickle
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import attention_attribution as a
from experiments.baseline_delivery import attention_update_diagnostic as entry
from experiments.baseline_delivery.common import atomic_json,digest,read
from tests.test_baseline_attention_update import child,PRESET


class Vector:
    def __init__(self,values): self.values=list(values);self.shape=(len(self.values),)
    def detach(self): return self
    def cpu(self): return self
    def double(self): return self
    def reshape(self,*args): return self
    def numel(self): return len(self.values)
    def op(self,other,f):
        vals=other.values if isinstance(other,Vector) else [other]*len(self.values)
        return Vector(f(x,y) for x,y in zip(self.values,vals))
    def __sub__(self,other): return self.op(other,lambda x,y:x-y)
    def __mul__(self,other): return self.op(other,lambda x,y:x*y)
    def __lt__(self,other): return self.op(other,lambda x,y:x<y)
    def __eq__(self,other): return self.op(other,lambda x,y:x==y)
    def sum(self): return sum(self.values)
    def all(self): return all(self.values)
    def abs(self): return Vector(abs(v) for v in self.values)
    def max(self): return max(self.values)


TORCH=NS(isfinite=lambda v:Vector(math.isfinite(x) for x in v.values))


def pair():
    def make(tiny,large):
        values={'encoder.layers.0.weight':Vector([large,0]),'head.bias':Vector([tiny,0])}
        row={field:copy.deepcopy(values) for field in a.FIELDS}
        row.update(row_ids_sha256='same128rows',optimizer_groups=[{'lr':.001}],optimizer_steps_before={'p':7},optimizer_steps_after={'p':8})
        return row
    x=make(1e-10,10);y=make(1e-8,11)
    for field in ('exp_avg_before','exp_avg_sq_before'): y[field]=copy.deepcopy(x[field])
    return x,y


class SummaryTests(unittest.TestCase):
    def test_known_vector_absolute_relative_sign_and_zero_statistics(self):
        value=a.scalar_pair(TORCH,Vector([-1,0,2]),Vector([1,1,2]))
        self.assertAlmostEqual(value['relative_l2'],1.)
        self.assertEqual(value['difference_squared_l2'],5)
        self.assertEqual(value['sign_disagreements'],1)
        self.assertEqual(value['reference_zero_count'],1)
        self.assertEqual(value['max_absolute_difference'],2)

    def test_zero_reference_is_explicit_and_nonfinite_is_rejected(self):
        value=a.scalar_pair(TORCH,Vector([0]),Vector([1]))
        self.assertTrue(value['denominator_clamped']);self.assertIsNone(value['cosine'])
        self.assertEqual(value['relative_l2'],1e12)
        with self.assertRaisesRegex(ValueError,'Nonfinite'): a.scalar_pair(TORCH,Vector([1]),Vector([math.nan]))
        with self.assertRaisesRegex(ValueError,'shape'): a.scalar_pair(TORCH,Vector([1]),Vector([1,2]))

    def test_error_energy_ranking_distinguishes_tiny_denominator_from_global_contribution(self):
        x,y=pair();summary=a.compare(TORCH,x,y);delta=summary['fields']['deltas']
        self.assertEqual(delta['descending_error_contribution'][0],'encoder.layers.0.weight')
        self.assertGreater(delta['parameters']['head.bias']['relative_l2'],90)
        self.assertLess(delta['parameters']['head.bias']['global_error_energy_share'],1e-12)
        self.assertAlmostEqual(sum(row['global_error_energy_share'] for row in delta['parameters'].values()),1)
        self.assertAlmostEqual(delta['total']['relative_l2'],.1)
        self.assertEqual(set(delta['groups']),{'encoder.layers.0','head'})
        self.assertTrue(summary['preexisting_moments_equal'])

    def test_population_policy_counter_coverage_and_initial_moment_changes_reject(self):
        for mutate in (lambda y:y.update(row_ids_sha256='other'),lambda y:y.update(optimizer_groups=[]),
                lambda y:y.update(optimizer_steps_before={}),lambda y:y['deltas'].pop('head.bias'),
                lambda y:y['exp_avg_before'].update({'head.bias':Vector([1,0])})):
            x,y=pair();mutate(y)
            with self.assertRaises(ValueError): a.compare(TORCH,x,y)

    def test_missing_initial_slots_are_reported_without_fabricated_values(self):
        x,y=pair()
        for row in (x,y): row['exp_avg_before']={};row['exp_avg_sq_before']={}
        value=a.compare(TORCH,x,y)
        self.assertEqual(value['fields']['exp_avg_before']['total']['parameter_count'],0)
        self.assertEqual(len(value['missing_before_moments']['fp32']['exp_avg']),2)


class CaptureTests(unittest.TestCase):
    def fixture(self,error=False):
        class Parameter: pass
        p=Parameter();p.grad=Vector([.5])
        class Opt:
            def step(self):
                self.state[p]={'exp_avg':Vector([.2]),'exp_avg_sq':Vector([.3]),'step':8}
        opt=Opt();opt.state={p:{'exp_avg':Vector([.1]),'exp_avg_sq':Vector([.2]),'step':7}}
        opt.param_groups=[dict(params=[p],lr=.01,eps=1e-8,betas=(.9,.999),weight_decay=.01)]
        runtime=NS(opt=opt,model=NS(named_parameters=lambda:[('p',p)]),clone=copy.deepcopy)
        def update(*args,**kwargs):
            opt.step()
            if error: raise RuntimeError('injected')
            opt.param_groups[0]['lr']=.005
            return {'gradients':{'p':Vector([1])}}
        runtime.update=update;return runtime

    def test_real_step_called_once_and_used_lr_and_pre_post_moments_observed(self):
        runtime=self.fixture();value=a.capture_update(runtime,True)
        self.assertEqual(value['optimizer_groups'][0]['lr'],.01)
        self.assertEqual(runtime.opt.param_groups[0]['lr'],.005)
        self.assertEqual(value['exp_avg_before']['p'].values,[.1])
        self.assertEqual(value['exp_avg_after']['p'].values,[.2])
        self.assertEqual(value['optimizer_steps_after']['p'],8)
        self.assertNotIn('step',runtime.opt.__dict__)

    def test_instance_hook_restores_on_failure_and_preserves_owned_override(self):
        runtime=self.fixture(error=True)
        with self.assertRaisesRegex(RuntimeError,'injected'): a.capture_update(runtime,True)
        self.assertNotIn('step',runtime.opt.__dict__)
        runtime=self.fixture();override=runtime.opt.step;runtime.opt.step=override
        a.capture_update(runtime,False);self.assertIs(runtime.opt.step,override)


class BindingTests(unittest.TestCase):
    def test_main_returns_only_compact_attribution_report_with_no_candidate_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            argv=['diagnostic','--source-suite',str(root/'source'),'--cache',str(root/'cache'),
                '--optimizations',str(PRESET),'--attribution-from',str(root/'prior')]
            def execute(args,folder,report):
                report.update(attribution_complete=True,candidate_admitted=False,passed=False,status='attribution_complete')
            with patch.object(sys,'argv',argv),patch.object(entry,'ROOT',root),patch.object(entry,'revision',return_value='rev'),\
                 patch.object(entry,'execute',side_effect=execute),patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                 patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload:
                self.assertEqual(entry.main(),0)
            folder,namespace,files=upload.call_args.args
            self.assertEqual(namespace,'attention-attribution');self.assertEqual(files,['attention-attribution.json'])
            report=read(folder/files[0]);self.assertEqual(report['schema'],'qkan-attention-attribution.v1')
            self.assertFalse(report['passed']);self.assertFalse(report['candidate_admitted'])

    def test_hash_bound_prior_rejects_changed_checkpoint_request_or_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);binding={'checkpoint':{'sha256':'original'},'cache':'cache'}
            atomic_json(folder/'update-input.json',binding)
            report=dict(status='diagnostic_rejected',passed=False,recovery={'passed':True},results={'benchmark':dict(request_sha256=digest(folder/'update-input.json'),numerical={})})
            atomic_json(folder/'attention-update.json',report)
            contract=folder/'contract.json';atomic_json(contract,{'prior_report_sha256':digest(folder/'attention-update.json')})
            with patch.object(a,'CONTRACT',contract):
                self.assertEqual(a.bind_prior(folder,binding)['prior_report_sha256'],digest(folder/'attention-update.json'))
                with self.assertRaisesRegex(ValueError,'changed'): a.bind_prior(folder,dict(binding,checkpoint={}))
                atomic_json(folder/'attention-update.json',dict(report,passed=True))
                with self.assertRaisesRegex(ValueError,'immutable'): a.bind_prior(folder,binding)

    def test_attribution_controller_starts_only_one_child_and_keeps_candidate_unadmitted(self):
        from tests.test_baseline_attention_precision import EntryTests
        with tempfile.TemporaryDirectory() as tmp:
            args=EntryTests().fixture(Path(tmp));args.optimizations=PRESET;args.attribution_from=Path(tmp)/'prior'
            binding=EntryTests().binding(args);folder=Path(tmp)/'out';folder.mkdir();modes=[]
            class Supervisor:
                def __init__(self,*args): self.records=[]
                def run(self,spec,log,env):
                    argv=spec['argv'];mode=argv[argv.index('--mode')+1];modes.append(mode)
                    out=Path(argv[argv.index('--output')+1]);out.mkdir()
                    request=read(folder/'update-input.json');result=child(request,mode)
                    result.update(checkpoint_sha256=request['checkpoint']['sha256'],request_sha256=digest(folder/'update-input.json'),
                        attribution_complete=True,candidate_admitted=False,updates_per_route=1,routes=['fp32','bf16'])
                    atomic_json(out/'result.json',result);return 0
            report={'source_revision':'new'}
            with patch.object(entry,'bind_source',return_value=binding),patch.object(a,'bind_prior',return_value={'contract_sha256':'contract'}),\
                 patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})),\
                 patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}),\
                 patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',Supervisor):
                entry.execute(args,folder,report)
            self.assertEqual(modes,['attribution']);self.assertTrue(report['attribution_complete'])
            self.assertFalse(report['passed']);self.assertFalse(report['candidate_admitted'])
            self.assertTrue(report['original_checkpoint_and_metadata_unchanged']);self.assertNotIn('recovery',report)


class RunTests(unittest.TestCase):
    def fixture(self,corrupt=False):
        x,y=pair();y['deltas']['encoder.layers.0.weight']=Vector([13,0])
        x['logits']=Vector([1]);y['logits']=Vector([1]);x['loss']=y['loss']=.5
        def save(value,path): Path(path).write_bytes(pickle.dumps(value))
        def load(path,**kwargs): return {'corrupt':True} if corrupt else pickle.loads(Path(path).read_bytes())
        rt=NS(t=NS(isfinite=TORCH.isfinite,save=save,load=load),base={'position':0},position=0)
        rt.restore=lambda s:setattr(rt,'position',s['position'])
        rt.snapshot=lambda:{'position':rt.position}
        rt.state_hash=lambda s:json.dumps(s,sort_keys=True,default=lambda v:v.values)
        rt.metrics=lambda l,r:dict(logits_relative_l2=0.,loss_absolute=0.,
            gradient_relative_l2=a.compare(TORCH,l,r)['fields']['gradients']['total']['relative_l2'],
            adam_delta_relative_l2=a.compare(TORCH,l,r)['fields']['deltas']['total']['relative_l2'])
        def capture(runtime,enabled): runtime.position+=1;return copy.deepcopy(y if enabled else x)
        return rt,capture

    def test_rejected_numerical_gate_still_gets_complete_diagnostics_and_exact_rollback(self):
        runtime,capture=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'capture_update',side_effect=capture):
            result=a.run(runtime,Path(tmp))
            self.assertTrue(result['attribution_complete']);self.assertFalse(result['numerical_passed'])
            self.assertFalse(result['candidate_admitted']);self.assertEqual(runtime.position,0)
            self.assertEqual(result['updates_per_route'],1)
            self.assertTrue(result['private_values']['complete_values_reopened_exact'])
            self.assertTrue(read(Path(tmp)/'attribution.json')['rollback']['full_state_exact'])

    def test_private_full_value_reopen_failure_cannot_complete_and_still_rolls_back(self):
        runtime,capture=self.fixture(corrupt=True)
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'capture_update',side_effect=capture):
            with self.assertRaisesRegex(ValueError,'Reopened'): a.run(runtime,Path(tmp))
            self.assertEqual(runtime.position,0)
            self.assertFalse(read(Path(tmp)/'attribution.json')['attribution_complete'])


if __name__=='__main__': unittest.main()
