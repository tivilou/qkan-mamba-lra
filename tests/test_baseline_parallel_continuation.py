"""Exercise real readiness selection/admission/preservation with stdlib fixtures."""
from contextlib import ExitStack
import copy
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery import readiness
from experiments.baseline_delivery.common import ROOT,atomic_json,digest,fingerprint,read
from experiments.baseline_delivery.migration import completed_files,WARMUP_REVISION
from experiments.baseline_delivery.torch_acceleration import policy,legacy_policy,AccelerationRequired,RecoveryMismatch
from experiments.baseline_delivery.logical_batch import policy as logical_policy
from experiments.baseline_delivery.performance import policy as performance_policy


class ParallelContinuationTests(unittest.TestCase):
    def args(self,root):
        return NS(migrate_from=None,resume=root/'runs/suite',parallel_retrieval_graph=True,
            adaptive_memory=False,dynamic_logical_batch=False,performance=False,torch_acceleration=False,
            cache=root/'cache',gpus='0',gates_only=True,require_graph=[],require_acceleration=[])

    def test_explicit_selection_is_job_scoped_and_rejects_changed_saved_backend(self):
        args=self.args(Path('.'));readiness.configure_parallel_continuation(args)
        self.assertTrue(args.adaptive_memory and args.performance and args.torch_acceleration)
        self.assertEqual(args.require_graph,[readiness.PARALLEL_JOB])
        for model in readiness.MODELS:
            for task in readiness.TASKS:
                job=model+'-'+task+'-s42';value=readiness.acceleration_profile(args,job)
                self.assertEqual(value,policy('compile_graph','eager_parallel','graph')
                    if job==readiness.PARALLEL_JOB else policy('eager'))
        with self.assertRaisesRegex(ValueError,'backend'):
            readiness.acceleration_profile(args,readiness.PARALLEL_JOB,policy('eager'))

    def test_explicit_entry_rejects_fresh_run_or_other_required_jobs(self):
        args=self.args(Path('.'));args.resume=None
        with self.assertRaises(ValueError): readiness.configure_parallel_continuation(args)
        args=self.args(Path('.'));args.require_graph=['mamba1-retrieval-s42']
        with self.assertRaises(ValueError): readiness.configure_parallel_continuation(args)

    def complete(self,suite,task):
        job='transformer-'+task+'-s42';folder=suite/'migration-source'/job
        config=dict(model='transformer',task=task,seed=42,source_revision='original-completed',data_hashes={})
        atomic_json(folder/'config.json',config)
        (folder/'best-model.pt').write_bytes(b'original selected model')
        atomic_json(folder/'summary.json',dict(config,formal=True,synthetic=False,
            config_sha256=fingerprint(config),best_weights_sha256=digest(folder/'best-model.pt'),
            contract='original-contract',optimizer_steps=100))
        atomic_json(folder/'environment.json',dict(runtime='original'))
        atomic_json(folder/'trace-validation.json',dict(passed=True,checkpoint_sha256=digest(folder/'best-model.pt')))
        (folder/'checkpoints').mkdir();(folder/'checkpoints/state').write_bytes(b'original full state')
        atomic_json(folder/'checkpoints/latest.json',dict(schema='qkan-training-checkpoint.v1',
            file='state',sha256=digest(folder/'checkpoints/state'),step=100,contract='original-contract'))
        return completed_files(folder,config)

    def route(self,failed_calibration=False):
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary);suite=root/'runs/suite';suite.mkdir(parents=True)
            args=self.args(root);tasks=('listops','text','retrieval','image')
            atomic_json(root/'experiments/baseline_delivery/protocol.json',{})
            atomic_json(suite/'transformer-source-probe.json',dict(passed=True,implementation_sha256='impl'))
            plan=dict(jobs={});completed={}
            for task in tasks:
                job='transformer-'+task+'-s42'
                if task in ('listops','text'):
                    entry=self.complete(suite,task);plan['jobs'][job]=dict(completion=entry)
                    completed[job]=entry
                else:
                    old=dict(model='transformer',task=task,seed=42,batch_size=32,source_revision=WARMUP_REVISION,
                        adaptive_memory=logical_policy(32),performance=performance_policy(),torch_acceleration=legacy_policy('eager',2))
                    atomic_json(suite/'migration-source'/job/'config.json',old)
                    plan['jobs'][job]=dict(checkpoint=dict(sha256='original-checkpoint',step=12683)
                        if task=='retrieval' else None)
            atomic_json(suite/'migration-source/plan.json',plan)
            calls=[];supervised=[];launches=[]
            def canary(config,data,cache,folder,device,**kwargs):
                calls.append((config['task'],copy.deepcopy(config['torch_acceleration'])))
                if config['task']=='retrieval':
                    self.assertEqual(kwargs['run'].__self__.__class__.__name__,'Supervisor')
                    self.assertEqual(kwargs['run'].__self__.child_timeout,900)
                    self.assertEqual(kwargs['run'].__self__.stall_timeout,300)
                else: self.assertEqual(kwargs,{})
                for name in ('uninterrupted','interrupted'):
                    atomic_json(folder/name/'trace-validation.json',dict(passed=True))
                return dict(passed=True,trace_semantic_and_replay=True,fresh_process=True,test_access=False,
                    config_sha256=fingerprint(config),torch_acceleration=dict(state=dict(mode=config['torch_acceleration']['mode'])))
            def child(spec,log,env=None):
                launches.append(spec)
                argv=spec['argv']
                if '--acceleration-probe-source' in argv:
                    cfg=read(argv[argv.index('--config')+1]);output=Path(argv[argv.index('--output')+1])
                    from experiments.baseline_delivery.acceleration_calibration import finish_probe
                    selected=dict(mode='eager' if failed_calibration else 'compile_graph',benchmarked=True,
                        reason='benchmark_no_gain' if failed_calibration else 'benchmark_gain',
                        benchmark=dict(eager_seconds=8.9,accelerated_seconds=9.0 if failed_calibration else 7.9))
                    value=finish_probe(cfg,dict(step=12683,migration=dict(source_checkpoint_sha256='original-checkpoint')),
                        selected,[],lambda:None,dict(graph_replays=96))
                    atomic_json(output/'acceleration-probe.json',value)
                    return 76 if failed_calibration else 0
                return 0
            def bounded(self_,spec,log,env=None):
                supervised.append(spec);return child(spec,log,env)
            patches={
                'experiments.baseline_delivery.readiness.ROOT':root,
                'experiments.baseline_delivery.readiness.MODELS':('transformer',),
                'experiments.baseline_delivery.readiness.TASKS':tasks,
                'experiments.baseline_delivery.readiness.revision':lambda:'new',
                'experiments.baseline_delivery.readiness.policy_check':lambda:('impl','policy'),
                'experiments.baseline_delivery.readiness.validate_runtime_release':lambda *a:None,
                'experiments.baseline_delivery.readiness.run_child':child,
                'experiments.baseline_delivery.runner.canary':canary,
                'experiments.baseline_delivery.runner.config_for':lambda task,model,seed,*a:dict(
                    model=model,task=task,seed=seed,batch_size=32,source_revision='new'),
                'experiments.baseline_delivery.runner.discover_gpus':lambda *a:(['0'],{'0':'same-device'}),
                'experiments.baseline_delivery.sync.prepare':lambda *a:root/'data',
                'experiments.baseline_delivery.bootstrap.check_core':lambda:None,
                'experiments.baseline_delivery.bootstrap.check_mamba':lambda:None,
                'experiments.baseline_delivery.migration.admit':lambda *a:dict(passed=True),
                'experiments.baseline_delivery.return_artifacts.upload_declared':Mock(),
                'experiments.baseline_delivery.diagnostic_watch.Supervisor.run':bounded}
            for target,value in patches.items(): stack.enter_context(patch(target,value))
            stack.enter_context(patch.dict(os.environ,{},clear=False))
            train=stack.enter_context(patch('experiments.baseline_delivery.runner.execute'))
            if failed_calibration:
                with self.assertRaises(AccelerationRequired): readiness.execute(args,suite)
                self.assertFalse((suite/'runtime-release.json').exists());train.assert_not_called()
                self.assertEqual([task for task,_ in calls],['retrieval'])
                return
            readiness.execute(args,suite)
            self.assertEqual([task for task,_ in calls],['retrieval','image'])
            self.assertEqual(calls[0][1],policy('compile_graph','eager_parallel','graph'))
            self.assertEqual(calls[1][1],policy('eager'))
            self.assertEqual(len(supervised),1)
            self.assertIn('experiments.baseline_delivery.worker',supervised[0]['argv'])
            self.assertNotIn('--synthetic',supervised[0]['argv'])
            self.assertIn('--acceleration-probe-source',supervised[0]['argv'])
            for job,entry in completed.items():
                for name,sha in entry['files'].items(): self.assertEqual(digest(suite/'jobs'/job/name),sha)
            context=read(suite/'context.json');self.assertTrue(context['parallel_retrieval_graph'])
            admission=suite/'jobs'/readiness.PARALLEL_JOB/'acceleration-admission.json'
            self.assertEqual(read(admission)['selected_backend'],'eager_parallel')
            self.assertEqual(read(admission)['required'],'graph')
            evidence=read(suite/'transformer-certificate.json')['evidence']
            self.assertIn(admission.relative_to(root).as_posix(),evidence)
            self.assertTrue(any('/calibration/' in path for path in evidence))
            self.assertEqual(len(read(suite/'suite.json')['jobs']),4)
            sealed=(suite/'gate-evidence.json').read_bytes();admitted=admission.read_bytes()
            readiness.execute(args,suite)
            self.assertEqual(len(calls),2);self.assertEqual(len(supervised),1)
            self.assertEqual((suite/'gate-evidence.json').read_bytes(),sealed)
            self.assertEqual(admission.read_bytes(),admitted)
            args.gates_only=False;readiness.execute(args,suite)
            train.assert_called_once_with(suite,['0'],args.cache,root/'data')
            self.assertEqual(len(calls),2)
            changed=self.args(root);changed.parallel_retrieval_graph=False
            changed.torch_acceleration=changed.performance=changed.dynamic_logical_batch=changed.adaptive_memory=True
            with self.assertRaisesRegex(ValueError,'resume identity'): readiness.execute(changed,suite)

    def test_real_readiness_binds_graph_preserves_completed_and_trains_only_after_gates(self):
        self.route()

    def test_nonpositive_calibration_stops_without_release_or_training(self):
        self.route(failed_calibration=True)

    def test_completed_or_missing_checkpoint_rejected_before_setup(self):
        for record in ({},{'completion':{'passed':True}},{'checkpoint':None}):
            with tempfile.TemporaryDirectory() as temp:
                args=self.args(Path(temp));suite=Path(temp)/'runs/suite'
                atomic_json(suite/'migration-source/plan.json',dict(jobs={readiness.PARALLEL_JOB:record}))
                with patch.object(readiness,'policy_check',return_value=('impl','policy')),patch.object(readiness,'revision',return_value='new'),patch.object(readiness,'run_child') as child:
                    with self.assertRaisesRegex(ValueError,'unfinished full checkpoint'): readiness.execute(args,suite)
                    child.assert_not_called()

    def test_exact_public_entry_parser_exposes_selection_without_framework_import(self):
        result=subprocess.run([sys.executable,'-B','-m','experiments.baseline_delivery.readiness','--help'],
            cwd=ROOT,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--parallel-retrieval-graph',result.stdout)

    def test_real_shell_chain_preserves_arguments_gate_only_and_failure_exit(self):
        bash=shutil.which('bash')
        if not bash:
            git=shutil.which('git')
            candidate=Path(git).parent.parent/'bin/bash.exe' if git else Path('missing')
            if candidate.is_file(): bash=str(candidate)
        if not bash: self.skipTest('Bash unavailable')
        for gates_only,gate_exit,supervisor_exit in ((True,0,0),(False,0,17),(False,9,0)):
            with self.subTest(gates_only=gates_only,gate_exit=gate_exit),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'scripts').mkdir();(root/'.venv-baselines/bin').mkdir(parents=True)
                for name in ('migrate_parallel_retrieval.sh','migrate_torch_acceleration.sh',
                        'migrate_performance.sh','migrate_dynamic_logical.sh','migrate_fixed_to_adaptive.sh'):
                    shutil.copyfile(ROOT/'scripts'/name,root/'scripts'/name)
                (root/'scripts/run_adaptive_readiness.sh').write_text(
                    '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > readiness-args\nexit '+str(gate_exit)+'\n',newline='\n')
                python=root/'.venv-baselines/bin/python'
                python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > supervisor-args\nexit '+str(supervisor_exit)+'\n',newline='\n')
                python.chmod(0o700)
                command=[bash,'scripts/migrate_parallel_retrieval.sh','--migrate-from','/fixture/original',
                    '--cache','/fixture/cache','--gpus','0']+(['--gates-only'] if gates_only else [])
                result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,gate_exit or supervisor_exit,result.stderr)
                arguments=(root/'readiness-args').read_text().splitlines()
                for flag in ('--parallel-retrieval-graph','--torch-acceleration','--performance','--dynamic-logical-batch','--gates-only'):
                    self.assertIn(flag,arguments)
                self.assertEqual(arguments[arguments.index('--migrate-from')+1],'/fixture/original')
                self.assertEqual((root/'supervisor-args').exists(),not gates_only and gate_exit==0)
                if (root/'supervisor-args').exists():
                    self.assertIn('scripts/formal_progress.py',(root/'supervisor-args').read_text())


if __name__=='__main__': unittest.main()
