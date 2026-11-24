"""Focused handoff/recovery fixtures; no framework, GPU or dependency imports."""
from contextlib import ExitStack,contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.baseline_delivery import mamba_fused_formal as f,mamba_fused_worker as w
from experiments.baseline_delivery.common import atomic_json,read,digest,fingerprint
from experiments.baseline_delivery import mamba_convolution as conv
from experiments.baseline_delivery import attention_update_native as update,key_bias_stability_native as values

spec=importlib.util.spec_from_file_location('unified_fused_entry',Path('scripts/experiment.py'))
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class Tests(unittest.TestCase):
    def test_unified_profile_and_resume_target(self):
        args=SimpleNamespace(profile='mamba-fused',migrate_from=Path('source'),cache=Path('cache'),
            gpu='0',graph_from=None,worker_from=None,output=None)
        cmd=entry.start_command(args,Path.cwd(),None)
        self.assertEqual(cmd[3],'experiments.baseline_delivery.mamba_fused_formal')
        self.assertIn('--prepare-only',cmd)
        with patch.object(entry,'validate_executor',return_value='mamba-fused'):
            self.assertEqual(entry.resume_command(Path.cwd(),Path('suite'))[3],cmd[3])
        args.graph_from=Path('unexpected')
        with self.assertRaises(ValueError):entry.start_command(args,Path.cwd(),None)
        spec=f.child_spec(Path('suite'),f.JOB,'resume')
        self.assertEqual(spec['target'],spec['argv'][3]);self.assertEqual(spec['argv'][-1],'resume')

    def test_supervisor_launch_requires_exact_output_for_every_mode(self):
        from experiments.baseline_delivery.diagnostic_watch import Supervisor
        suite=Path('runs/fused-fixture')
        class Accepted(Exception):pass
        for mode in (*f.MODES,'formal'):
            launch=f.child_spec(suite,f.JOB,mode)
            argv=launch['argv'];self.assertEqual(argv.count('--output'),1)
            output=Path(argv[argv.index('--output')+1])
            expected=suite/('jobs/'+f.JOB if mode=='formal' else 'preflight/'+mode)
            self.assertEqual(output,expected)
            self.assertEqual(w.checked_output(suite/'mamba-input.json',f.JOB,mode,output),expected)
            with self.assertRaisesRegex(ValueError,'output differs'):
                w.checked_output(suite/'mamba-input.json',f.JOB,mode,Path('wrong'))
            # Exercise the real Supervisor argument parser, stopping immediately
            # before filesystem/process creation. No model worker is launched.
            with patch.object(Path,'mkdir',side_effect=Accepted):
                with self.assertRaises(Accepted):Supervisor().run(launch,Path('fixture.log'),{})
        with patch.object(sys,'argv',['worker','--request','suite/mamba-input.json','--job',f.JOB,
                    '--output','wrong','--mode','resume']),patch.object(w,'native') as native,                    patch.object(w,'worker_failure'):
            with self.assertRaisesRegex(ValueError,'output differs'):w.main()
            native.assert_not_called()

    def test_actual_targets_help_without_framework_import(self):
        for target in (f.__name__,w.__name__):
            proc=subprocess.run([sys.executable,'-B','-c',
                "import sys,runpy;sys.argv=['worker','--help'];"+
                "sys.modules['torch']=None;sys.modules['numpy']=None;runpy.run_module('"+target+"',run_name='__main__')"],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)

    def test_original_relative_import_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'experiments/baseline_delivery';p.mkdir(parents=True)
            (p/'worker.py').write_text('from .fixture import token\n')
            (p/'fixture.py').write_text('token="original"\n')
            try:self.assertEqual(w.original_worker(tmp).token,'original')
            finally:
                for key in list(sys.modules):
                    if key.startswith('_qkan_mamba_original'):sys.modules.pop(key)

    def fixture(self,suite):
        source=suite/'jobs'/f.JOB;source.mkdir(parents=True)
        config=dict(model='mamba1',task='retrieval');atomic_json(source/'config.json',config)
        state=dict(step=24981,epoch=6,cursor=100,sample_clock=dict(progress=100),model=dict(weight=1.0),
            rng=dict(token=123),training_seconds=0.)
        def save(folder,state):
            folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
            path=folder/'checkpoints'/('step-'+str(state['step'])+'.pt');atomic_json(path,state)
            atomic_json(folder/'checkpoints/latest.json',dict(schema='qkan-training-checkpoint.v1',file=path.name,
                sha256=digest(path),contract='original',step=state['step'],previous=[]))
        save(source,state);base=read(source/'checkpoints/latest.json')
        r=dict(candidates={f.JOB:dict(checkpoint=base,execution_root='original',execution_revision='old')},
            data='data',cache='cache',binding=dict(code='bound'),source_files={})
        atomic_json(suite/'mamba-input.json',r)
        class Model:
            training=True
            def forward(self):return self.training
        class Policy:
            def __init__(self,torch,model):self.calls=0;self.mechanism=dict(policy='fused');self.closed=False
            @contextmanager
            def activate(self):self.calls+=1;yield
            def close(self):self.closed=True
        fake=SimpleNamespace(Classifier=lambda *a,**k:Model(),save_checkpoint=save,atomic_json=atomic_json,
            request_pause=lambda *a:None,load_checkpoint=lambda folder,contract:read(Path(folder)/'checkpoints'/read(Path(folder)/'checkpoints/latest.json')['file']))
        def run(config,data,cache,folder,device,resume,stop_after):
            state=fake.load_checkpoint(folder,'original');model=fake.Classifier(config,cache)
            for i in range(stop_after-state['step']):
                model.training=True;model.forward();model.training=False;model.forward()
                state['step']+=1;state['cursor']+=32;state['sample_clock']['progress']+=32
                state['model']['weight']+=.01
                fake.atomic_json(Path(folder)/'heartbeat.json',dict(optimizer_step=state['step']))
            fake.save_checkpoint(folder,state);return 75
        fake.run=run
        return r,fake,Policy

    def test_actual_worker_continuous_and_fresh_resume_fixture(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            suite=Path(tmp);r,fake,policy=self.fixture(suite);original=fake.Classifier
            stack.enter_context(patch.object(f,'verify',return_value=r))
            stack.enter_context(patch.object(w,'original_worker',return_value=fake))
            stack.enter_context(patch.dict(sys.modules,{'torch':SimpleNamespace(),'numpy':SimpleNamespace()}))
            stack.enter_context(patch.object(conv,'prepare_native'))
            stack.enter_context(patch.object(conv,'FusedFP32',policy))
            stack.enter_context(patch.object(update.NativeUpdate,'state_hash',side_effect=lambda _,v:fingerprint(v)))
            stack.enter_context(patch.object(values,'compare_values',side_effect=lambda _,a,b:self.assertEqual(a,b)))
            for mode in f.MODES:self.assertEqual(w.native(suite/'mamba-input.json',f.JOB,mode),0)
            result=read(suite/'preflight/resume/result.json')
            self.assertTrue(result['recovery']['passed']);self.assertEqual(result['calls'],1)
            self.assertEqual(read(suite/'preflight/continuous/result.json')['calls'],2)
            self.assertIs(fake.Classifier,original)
            self.assertEqual(read(suite/'jobs'/f.JOB/'checkpoints/latest.json')['step'],24981)

    def test_preflight_only_three_children_cleanup_return_and_failure(self):
        from experiments.baseline_delivery import diagnostic_watch,return_artifacts
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                suite=Path(tmp);r,_,_=self.fixture(suite)
                calls=[]
                class Supervisor:
                    def __init__(self,*a):self.records=[]
                    def run(self,spec,log,env):
                        mode=spec['argv'][-1];calls.append(mode)
                        self_output=Path(spec['argv'][spec['argv'].index('--output')+1])
                        assert self_output==suite/'preflight'/mode
                        self.records.append(dict(exit_code=1 if fail else 0,termination=dict(group_verified=True)))
                        atomic_json(suite/'preflight'/mode/'checkpoints/step-1.pt',dict(disposable=True))
                        atomic_json(suite/'preflight'/mode/'result.json',dict(passed=True,mode=mode,job=f.JOB,
                            input_sha256=digest(suite/'mamba-input.json'),step=24982 if mode=='interrupt' else 24983,
                            calls=1,copy_roundtrip_exact=True,formal_updates_committed=0,recovery=dict(passed=True)))
                        return 1 if fail else 0
                with patch.object(diagnostic_watch,'Supervisor',Supervisor),patch.object(return_artifacts,'upload_declared') as upload:
                    if fail:
                        with self.assertRaises(RuntimeError):f.preflight(suite,r,'0')
                        self.assertFalse((suite/'mamba-ready.json').exists())
                    else:
                        f.preflight(suite,r,'0');self.assertEqual(calls,list(f.MODES))
                        self.assertTrue((suite/'mamba-ready.json').exists())
                    self.assertFalse(list((suite/'preflight').glob('*/checkpoints/step-*.pt')))
                    upload.assert_called_once()

    def test_prepare_copies_seven_completed_and_keeps_configs(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as held:
            root=Path(tmp);source=root/'origin/runs/source';suite=root/'new/runs/mamba-fused-formal-fixture'
            suite.mkdir(parents=True);jobs=[];states={}
            done={n for n in f.original.JOBS if n.startswith('transformer-')}|{'mamba1-listops-s42','mamba1-text-s42'}
            for name in sorted(f.original.JOBS):
                folder=source/'jobs'/name;folder.mkdir(parents=True)
                model,task,_=name.split('-');config=dict(model=model,task=task,source_revision='old')
                atomic_json(folder/'config.json',config)
                jobs.append(dict(id=name,config_sha256=fingerprint(config),execution_root='pinned-original',execution_revision='old'))
                states[name]=dict(state='computed' if name in done else 'paused' if name==f.JOB else 'pending')
            folder=source/'jobs'/f.JOB;atomic_json(folder/'checkpoints/step-24981.pt',dict(fixture=True))
            pointer=dict(schema='qkan-training-checkpoint.v1',file='step-24981.pt',sha256=digest(folder/'checkpoints/step-24981.pt'),contract='fixed',step=24981,previous=[])
            atomic_json(folder/'checkpoints/latest.json',pointer)
            atomic_json(source/'suite.json',dict(jobs=jobs,data='fixed-data'))
            atomic_json(source/'status.json',states)
            atomic_json(source/'graph-input.json',dict(source_revision='old',cache=str((root/'cache').resolve()),source_suite='legacy',original_root='original'))
            review=root/'review.json';atomic_json(review,dict(timing=dict(summary=dict(timing_promising=True)),evidence=dict(source_checkpoint_sha256=pointer['sha256'])))
            held.enter_context(patch.object(f,'ROOT',suite.parent.parent));held.enter_context(patch.object(f,'REVIEW',review))
            held.enter_context(patch.object(f,'revision',return_value='new'));held.enter_context(patch.object(f,'binding',return_value={'fixed':'binding'}))
            held.enter_context(patch.object(f,'checked_executor',side_effect=lambda root,rev:str(root)))
            held.enter_context(patch.object(conv,'stage',return_value=dict(root='offline')))
            held.enter_context(patch.object(f,'completed_files',side_effect=lambda folder,config:dict(files={'config.json':digest(folder/'config.json')})))
            held.enter_context(patch.object(f,'verify_completed'))
            r=f.prepare(SimpleNamespace(source_suite=source,cache=root/'cache'),suite)
            self.assertEqual(len(r['candidates']),3)
            self.assertEqual(sum(s['state']=='computed' for s in read(suite/'status.json').values()),7)
            self.assertEqual(read(suite/'jobs'/f.JOB/'checkpoints/latest.json'),pointer)
            self.assertFalse((suite/'jobs/mamba1-image-s42/checkpoints/latest.json').exists())
            self.assertEqual(read(suite/'jobs'/f.JOB/'config.json'),read(source/'jobs'/f.JOB/'config.json'))
            self.assertTrue(f.verify(suite/'mamba-input.json'))
            with self.assertRaises(FileNotFoundError):f.verify(suite/'mamba-input.json',ready=True)
            atomic_json(source/'status.json',{})
            with self.assertRaisesRegex(ValueError,'source changed'):f.unchanged(r)

    def test_binding_tamper_and_missing_ready_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp)/'runs/suite';suite.mkdir(parents=True)
            atomic_json(suite/'suite.json',dict(jobs=[]));atomic_json(suite/'migration-source/plan.json',{})
            r=dict(schema=f.SCHEMA,source_revision='pinned',binding=f.binding(),manifest_sha256=digest(suite/'suite.json'),plan_sha256=digest(suite/'migration-source/plan.json'))
            atomic_json(suite/'mamba-input.json',r)
            with patch.object(f,'ROOT',Path(tmp)),patch.object(f,'revision',return_value='pinned'),patch.object(f,'binding',return_value=r['binding']):
                with self.assertRaisesRegex(ValueError,'Roster'):f.verify(suite/'mamba-input.json')
                r['manifest_sha256']='changed';atomic_json(suite/'mamba-input.json',r)
                with self.assertRaisesRegex(ValueError,'Bound'):f.verify(suite/'mamba-input.json')


if __name__=='__main__':unittest.main()
