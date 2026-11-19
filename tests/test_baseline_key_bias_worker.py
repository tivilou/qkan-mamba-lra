"""Final-worker controls, no native framework imports or model execution."""
import ast
import copy
from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import key_bias_training as strategy
from experiments.baseline_delivery import key_bias_worker_admission as admission
from experiments.baseline_delivery.common import atomic_json


def config():
    return dict(model='transformer',task='retrieval',layers=4,width=128,heads=4,dropout=.1,
        adaptive_memory=dict(execution=dict(precision='fp32')),torch_acceleration=dict(mode='compile_graph'))


class CandidateTests(unittest.TestCase):
    def test_all_off_identity_and_source_copy(self):
        self.assertFalse(strategy.select({}))
        old=config();original=copy.deepcopy(old);new=strategy.target_config(old)
        self.assertEqual(old,original);self.assertTrue(strategy.select(new))
        self.assertNotIn('torch_acceleration',new)
        self.assertEqual(new['adaptive_memory']['execution']['precision'],strategy.POLICY['precision'])

    def test_unknown_combinations_closed(self):
        for key,value in (('task','text'),('model','mega'),('dropout',0),('layers',2),
                          ('torch_acceleration',{'mode':'compile_graph'}),('canary_logical_tiers',[32,64,128])):
            with self.subTest(key=key):
                c=strategy.target_config(config());c[key]=value
                with self.assertRaises(ValueError): strategy.select(c)
        c=strategy.target_config(config());c['training_strategy']['gradient_policy']='old'
        with self.assertRaises(ValueError): strategy.select(c)

    def test_checkpoint_strategy_and_all_off_resume_reject(self):
        c=strategy.target_config(config());expected=strategy.envelope(c,'source',{'files':{'worker':'current'}})
        state=dict(training_strategy=expected,strategy_gradients={})
        strategy.check_checkpoint(state,c,expected)
        for changed in ('policy','config_sha256','source_checkpoint_sha256','implementation','transition'):
            bad=copy.deepcopy(state);bad['training_strategy'][changed]='stale'
            with self.assertRaises(ValueError): strategy.check_checkpoint(bad,c,expected)
        with self.assertRaises(ValueError): strategy.check_checkpoint(state,config(),None)
        with self.assertRaises(ValueError): strategy.check_checkpoint({},c,expected)
        strategy.check_checkpoint({},config(),None)

    def test_fixed_policy_recertifies_ceiling_without_touching_oom_count(self):
        controller=SimpleNamespace(policy=dict(logical_tiers=[32,64,128],tiers=[1,2,4,8,16,32,64,128]),
            state=dict(index=2,logical_index=0,ceiling=3,capacity_ceiling=3,oom_count=7))
        strategy.pin(controller)
        self.assertEqual((controller.state['index'],controller.state['logical_index'],controller.state['ceiling']),(4,2,7))
        self.assertEqual(controller.state['oom_count'],7);self.assertNotIn('capacity_ceiling',controller.state)

    def test_process_global_flag_restored_on_failure(self):
        calls=[];mha=SimpleNamespace(get_fastpath_enabled=lambda:True,set_fastpath_enabled=calls.append)
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with strategy.process(SimpleNamespace(backends=SimpleNamespace(mha=mha))): raise RuntimeError('stop')
        self.assertEqual(calls,[False,True])

    def test_update_coverage_and_exception_cleanup(self):
        closed=[]
        @contextmanager
        def forward(*a,**kw):
            try: yield dict(layer_calls=[16]*4,qkv_dtypes=['torch.bfloat16'],output_dtypes=['torch.bfloat16'])
            finally: closed.append('forward')
        @contextmanager
        def backward(*a,**kw):
            value=dict(hook_calls={n:8 for n in strategy.NAMES},handles_removed=False)
            try: yield value
            finally: value['handles_removed']=True;closed.append('gradient')
        with patch.object(strategy,'attention_route',forward),patch.object(strategy,'analytic_key_bias',backward):
            with strategy.update(None,None,16,128): pass
            self.assertEqual(closed,['gradient','forward']);closed.clear()
            with self.assertRaisesRegex(RuntimeError,'native'):
                with strategy.update(None,None,16,128): raise RuntimeError('native')
            self.assertEqual(closed,['gradient','forward'])
            with self.assertRaises(ValueError):
                with strategy.update(None,None,8,128): pass
            with self.assertRaises(ValueError):
                with strategy.update(None,None,16,64): pass

    def test_worker_numerical_update_body_identical_to_preintegration(self):
        root=Path(__file__).resolve().parents[1];name='experiments/baseline_delivery/worker.py'
        before=subprocess.check_output(['git','show','7e17f8fa6fd530580d559c424385fad1f98d11f7:'+name],cwd=root,text=True)
        after=(root/name).read_text(encoding='utf8')
        def get(text,name):
            node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name)
            node.name='original_update';return ast.dump(node,include_attributes=False)
        self.assertEqual(get(before,'update_impl'),get(after,'update_body'))

    def test_admission_requires_finite_owned_current_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);suite=root/'runs/key-bias-worker-fixture';suite.mkdir(parents=True)
            path=suite/'worker-input.json';c=strategy.target_config(config())
            r=dict(schema='qkan-key-bias-worker-input.v1',target_config=c,checkpoint=dict(step=100,sha256='x'),implementation={'current':True})
            atomic_json(path,r);gate=strategy.Admission(r,path,suite/'continuous','continuous')
            with patch.object(strategy,'ROOT',root),patch.object(strategy,'binding',return_value=r['implementation']), \
                    patch('experiments.baseline_delivery.attention_precision_diagnostic.unchanged') as unchanged:
                gate.validate(c,gate.output,'cuda:0',True,102,False,False,None)
                self.assertEqual(unchanged.call_count,1)
                for limit in (None,101,103,100000):
                    with self.assertRaises(ValueError): gate.validate(c,gate.output,'cuda:0',True,limit,False,False,None)
                with patch.object(strategy,'binding',return_value={'stale':True}):
                    with self.assertRaises(ValueError): gate.validate(c,gate.output,'cuda:0',True,102,False,False,None)

    def test_scientific_state_never_excludes_gradients_rng_or_progress(self):
        value=dict(model=1,optimizer=2,scheduler=3,rng=4,permutation=5,step=6,cursor=7,epoch_loss=8,
                   strategy_gradients=9,training_strategy=10,adaptive_memory='timings',training_seconds=2)
        actual=admission.scientific(value)
        self.assertEqual(set(actual),set(value)-{'adaptive_memory','training_seconds'})

    def test_child_identity_budget_and_kernel_coverage_reject(self):
        r=dict(implementation={'worker':'new'},checkpoint=dict(step=100,sha256='source'))
        value=dict(schema=admission.SCHEMA,mode='continuous',request_sha256='request',implementation=r['implementation'],
            source_checkpoint_sha256='source',passed=True,formal_training_started=False,candidate_admitted=False,
            step=102,updates=[dict(samples=128,micro_batch=16)]*2)
        self.assertIs(admission.accept(value,r,'request','continuous'),value)
        for key,bad in (('step',103),('implementation',{'worker':'old'}),('candidate_admitted',True),('updates',[])):
            with self.subTest(key=key),self.assertRaises(ValueError):
                admission.accept(dict(value,**{key:bad}),r,'request','continuous')
        bad=copy.deepcopy(value);bad['updates'][0]['micro_batch']=32
        with self.assertRaises(ValueError): admission.accept(bad,r,'request','continuous')

    def test_recovery_gate_rejects_scientific_or_trace_difference(self):
        first=dict(step=101,samples=128,micro_batch=16);second=dict(step=102,samples=128,micro_batch=16)
        a=dict(scientific_state_sha256='final',component_sha256={'rng':'same'},updates=[first,second])
        b=dict(scientific_state_sha256='mid',updates=[first])
        c=dict(**a,loaded_state_sha256='mid',trace=dict(validation=dict(passed=True)))
        results=dict(continuous=a,interrupt=b,resume=c)
        self.assertTrue(admission.agreement(results)['passed'])
        for field,value in (('scientific_state_sha256','bad'),('component_sha256',{}),('loaded_state_sha256','wrong'),('trace',{})):
            bad=copy.deepcopy(results);bad['resume'][field]=value
            with self.assertRaises(ValueError): admission.agreement(bad)

    def test_parent_always_declares_compact_autoreturn_on_pass_and_failure(self):
        for failure in (False,True):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                def execute(args,folder,report):
                    report.update(passed=True,status='final_worker_preflight_passed_review_pending')
                    if failure: raise RuntimeError('simulated native failure after provisional pass')
                with patch.object(admission,'ROOT',root),patch.object(admission,'revision',return_value='pinned'),patch.object(strategy,'binding',return_value={'code':'pinned'}), \
                        patch.object(admission,'execute',side_effect=execute),patch('experiments.baseline_delivery.bootstrap.check_exchange'), \
                        patch('experiments.baseline_delivery.return_artifacts.upload_declared') as upload, \
                        patch('sys.argv',['admission','--source-suite','source','--cache','cache','--pilot-from','pilot']):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError,'simulated'): admission.main()
                    else: self.assertEqual(admission.main(),0)
                folder,namespace,files=upload.call_args.args
                self.assertEqual(namespace,'key-bias-worker-admission');self.assertEqual(files,['key-bias-worker-admission.json'])
                report=__import__('json').loads((folder/files[0]).read_text(encoding='utf8'))
                self.assertEqual(report['passed'],not failure)
                self.assertIs(report['candidate_admitted'],False);self.assertIs(report['formal_training_started'],False)
                self.assertEqual(report['training_samples_committed'],0)


if __name__=='__main__': unittest.main()
