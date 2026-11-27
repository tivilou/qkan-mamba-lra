"""Production hook routing and finite qualification contracts; stdlib only."""
from contextlib import contextmanager,ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.baseline_delivery.common import atomic_json,read,digest
from experiments.baseline_delivery import execution_plugins as plugins,mega_execution as execution,mega_qualification as q


class Tests(unittest.TestCase):
    def test_even_pending_window_does_not_cross_epoch(self):
        for remaining,expected in ((2048,8),(1024,8),(896,6),(512,4),(384,2),(256,2),(255,0),(128,0),(0,0)):
            state=dict(training_complete=False,cursor=10000-remaining)
            self.assertEqual(q.window(state,10000,128),expected)
            self.assertLessEqual(state['cursor']+expected*128,10000)
        self.assertEqual(q.window(dict(training_complete=True,cursor=0),10000,128),0)

    def test_pre_worker_exception_keeps_original_frame_and_error_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            try:raise ValueError('private unknown content')
            except ValueError as error:q.failure_detail(out,error)
            detail=read(out/'failure-detail.json')
            self.assertEqual(detail['native_error']['error_type'],'ValueError')
            self.assertEqual(detail['chain'][0]['error_type'],'ValueError')
            self.assertNotIn('private',detail['chain'][0]['template'])
            self.assertTrue(detail['native_error']['project_frames'])

    def test_registry_graph_is_independent_boolean_default_off(self):
        cfg=dict(model='mega',task='text');value=dict(plugin='mega.fft_tail_bf16',version=1)
        self.assertEqual(plugins.resolve(value,cfg)['backend'],'eager')
        self.assertEqual(plugins.resolve(dict(value,graph=True),cfg)['backend'],'ffn_graph')
        for bad,config in ((dict(value,graph=1),cfg),(dict(value,unknown=True),cfg),(value,dict(model='mamba1',task='text'))):
            with self.assertRaises(ValueError):plugins.resolve(bad,config)
        with patch.object(execution,'Policy') as policy:
            plugins.create(value,cfg,'torch','model');policy.assert_called_once_with('torch','model',graph=False)

    def test_cli_captures_failure_before_native_worker_try(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'reference'
            argv=['qualify','--request',str(root/'request.json'),'--mode','reference','--output',str(out)]
            with patch.object(sys,'argv',argv),patch.object(q,'native',side_effect=ValueError('private preparation error')):
                with self.assertRaises(ValueError):q.main()
            detail=read(out/'failure-detail.json')
            self.assertEqual(detail['native_error']['error_type'],'ValueError')
            self.assertTrue(detail['native_error']['project_frames'])
            self.assertNotIn('private',detail['chain'][0]['template'])

    def test_training_only_context_evaluation_original_and_cleanup(self):
        active={};closed=[]
        class Subpolicy:
            def __init__(self,*args):self.calls=0;self.captures=0
            @contextmanager
            def activate(self,enabled=True):
                active[id(self)]=enabled
                try:yield self
                finally:active.pop(id(self))
            def close(self):closed.append(self)
        class Model:
            training=True
            def forward(self,x):return x,tuple(active.values())
        model=Model();original=Model.forward
        with patch('experiments.baseline_delivery.mega_fft.PowerTwoEMA',Subpolicy),\
                patch('experiments.baseline_delivery.mega_refinement.PrecisionIslands',Subpolicy),\
                patch('experiments.baseline_delivery.mega_refinement.FFNGraph',Subpolicy):
            with execution.Policy('torch',model,graph=False) as policy:
                self.assertEqual(model.forward(1),(1,(True,True)))
                model.training=False;self.assertEqual(model.forward(2),(2,(False,False)))
                self.assertEqual(policy.calls,1);self.assertFalse(policy.graph)
            self.assertIs(Model.forward,original);self.assertNotIn('forward',model.__dict__)
            self.assertEqual(len(closed),2);self.assertFalse(active)

    def test_modes_use_existing_native_env_and_single_owned_output(self):
        suite=Path('/fixture')
        self.assertEqual(q.STEPS,8);self.assertEqual(len(q.MODES),7)
        for mode in q.MODES:
            spec=q.child_spec(suite,mode);a=spec['argv']
            self.assertEqual(Path(a[0]).parts[-3:],('.venv-mega','bin','python'))
            self.assertEqual(a.count('--output'),1);self.assertEqual(Path(a[a.index('--output')+1]),suite/mode)
        with self.assertRaises(ValueError):q.child_spec(suite,'formal')

    def test_technical_recovery_does_not_imply_quality(self):
        results={m:dict(passed=True,recovery_passed=True) for m in q.MODES}
        self.assertTrue(q.technical(results))
        results['graph-resume']['recovery_passed']=False;self.assertFalse(q.technical(results))
        results.pop('reference');self.assertFalse(q.technical(results))

    def test_campaign_reuses_snapshot_one_return_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original';job='mega-text-s42';folder=source/'jobs'/job
            cache=root/'cache';cache.mkdir();data=root/'data';data.mkdir()
            atomic_json(source/'status.json',{job:dict(state='paused')})
            atomic_json(source/'suite.json',dict(data=str(data)))
            atomic_json(folder/'config.json',dict(model='mega',task='text',adaptive_memory={'enabled':True}))
            cp=folder/'checkpoints/step-10.pt';cp.parent.mkdir();cp.write_bytes(b'checkpoint')
            atomic_json(folder/'checkpoints/latest.json',dict(schema='qkan-training-checkpoint.v1',file=cp.name,sha256=digest(cp),step=10,contract='bound'))
            original=cp.read_bytes();returned=[]
            class Worker:
                def __init__(self,*a):self.records=[]
                def run(self,spec,log,env):
                    self.records.append(dict(exit_code=0,termination=dict(group_verified=True)))
                    a=spec['argv'];out=Path(a[a.index('--output')+1]);mode=a[a.index('--mode')+1]
                    request=Path(a[a.index('--request')+1])
                    atomic_json(out/'result.json',dict(passed=True,mode=mode,planned_steps=8,request_sha256=digest(request),
                        temporary_updates=4 if mode.endswith(('interrupt','resume')) else 8,formal_updates=0,
                        step=14 if mode.endswith('interrupt') else 18,recovery_passed=True,row_ids_sha256='same',updates=[dict(loss=.5)],training_seconds=1.,
                        validation=dict(loss=.5,accuracy=.5,samples=256)))
                    return 0
            args=SimpleNamespace(source_suite=source,cache=cache,job='auto',gpu='0')
            with ExitStack() as stack:
                stack.enter_context(patch.object(q,'ROOT',root));stack.enter_context(patch.object(q,'revision',return_value='pinned'))
                stack.enter_context(patch.object(q,'binding',return_value={'worker':'same'}))
                stack.enter_context(patch('experiments.baseline_delivery.mega_diagnostic.revision',return_value='pinned'))
                stack.enter_context(patch('experiments.baseline_delivery.bootstrap.check_exchange'))
                stack.enter_context(patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})))
                stack.enter_context(patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}))
                stack.enter_context(patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',Worker))
                stack.enter_context(patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:returned.append(a)))
                self.assertEqual(q.campaign(args),0)
            self.assertEqual(len(returned),1);report=read(returned[0][0]/'qualification.json')
            self.assertTrue(report['technical_passed']);self.assertFalse(report['quality_admitted'])
            self.assertTrue(report['owned_checkpoints_removed']);self.assertTrue(report['source_unchanged'])
            self.assertEqual(len(report['child_executions']),7);self.assertEqual(cp.read_bytes(),original)
            def run_case(worker):
                returned.clear()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(q,'ROOT',root));stack.enter_context(patch.object(q,'revision',return_value='pinned'))
                    stack.enter_context(patch.object(q,'binding',return_value={'worker':'same'}))
                    stack.enter_context(patch('experiments.baseline_delivery.mega_diagnostic.revision',return_value='pinned'))
                    stack.enter_context(patch('experiments.baseline_delivery.bootstrap.check_exchange'))
                    stack.enter_context(patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})))
                    stack.enter_context(patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}))
                    stack.enter_context(patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',worker))
                    stack.enter_context(patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:returned.append(a)))
                    self.assertEqual(q.campaign(args),0)
                self.assertEqual(len(returned),1);self.assertEqual(cp.read_bytes(),original)
                return returned[0][0],read(returned[0][0]/'qualification.json')
            class ShortWorker(Worker):
                def run(self,spec,log,env):
                    super().run(spec,log,env);a=spec['argv'];mode=a[a.index('--mode')+1];out=Path(a[a.index('--output')+1])
                    value=read(out/'result.json');value.update(planned_steps=2,
                        temporary_updates=1 if mode.endswith(('interrupt','resume')) else 2,
                        step=11 if mode.endswith('interrupt') else 12)
                    atomic_json(out/'result.json',value);return 0
            _,report=run_case(ShortWorker)
            self.assertTrue(report['technical_passed']);self.assertEqual(report['planned_steps'],2)
            self.assertEqual(len(report['child_executions']),7)
            class EarlyFailure(Worker):
                def run(self,spec,log,env):
                    self.records.append(dict(exit_code=1,termination=dict(group_verified=True)))
                    out=Path(spec['argv'][spec['argv'].index('--output')+1])
                    try:raise ValueError('private_setup_error')
                    except ValueError as error:q.failure_detail(out,error)
                    return 1
            suite,report=run_case(EarlyFailure)
            self.assertEqual(len(report['child_executions']),1);self.assertFalse(report['technical_passed'])
            self.assertEqual(report['failures']['reference']['error']['error_type'],'RuntimeError')
            self.assertEqual(report['failures']['reference']['native_error']['error_type'],'ValueError')
            self.assertEqual(read(suite/'reference/worker-error.json')['error_type'],'ValueError')
            class NoWindow(Worker):
                def run(self,spec,log,env):
                    self.records.append(dict(exit_code=0,termination=dict(group_verified=True)))
                    out=Path(spec['argv'][spec['argv'].index('--output')+1]);request=Path(spec['argv'][spec['argv'].index('--request')+1])
                    atomic_json(out/'result.json',dict(mode='reference',skipped=True,request_sha256=digest(request),temporary_updates=0,formal_updates=0))
                    return 0
            _,report=run_case(NoWindow)
            self.assertEqual(len(report['child_executions']),1);self.assertTrue(report['skipped']['skipped'])
            self.assertFalse(report['technical_passed']);self.assertTrue(report['owned_checkpoints_removed'])


    def test_cli_no_native_import_and_unified_route(self):
        text="import sys,runpy;sys.modules['torch']=None;sys.modules['numpy']=None;sys.argv=['qualify','--help'];runpy.run_module('experiments.baseline_delivery.mega_qualification',run_name='__main__')"
        p=subprocess.run([sys.executable,'-B','-c',text],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        from scripts.experiment import DIAGNOSTICS
        self.assertEqual(DIAGNOSTICS['mega-qualify'],'mega_qualification')

if __name__=='__main__':unittest.main()
