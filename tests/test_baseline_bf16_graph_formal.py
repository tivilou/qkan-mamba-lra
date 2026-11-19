"""Owner stdlib/fakes only. These controls do not certify native Graph execution."""
from contextlib import contextmanager, ExitStack, nullcontext
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch, Mock

from experiments import baseline_delivery as package
from experiments.baseline_delivery import bf16_graph_formal as f, bf16_graph_worker as w
from experiments.baseline_delivery import bf16_graph_training as a, key_bias_formal as old
from experiments.baseline_delivery import return_artifacts, diagnostic_watch
from experiments.baseline_delivery.common import atomic_json, digest, read, fingerprint
from scripts import experiment as e, formal_progress as ui
from tests.test_baseline_key_bias_formal import fixture


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name);self.exec=a.Execution(self.output,{'fixture':True})
        self.graph=NS(f=0,b=0,s=0,cleared=False,begin_update=lambda micro:None)
        self.graph.status=lambda:dict(graph_resident=not self.graph.cleared,graph_builds=1,
            forward_replays=self.graph.f,backward_replays=self.graph.b,replica_sync_rounds=self.graph.s)
        self.graph.clear=lambda:setattr(self.graph,'cleared',True)
        def replay(batch):
            self.graph.f+=2;self.graph.b+=2;self.graph.s+=1;return 'graph'
        self.graph.forward=replay
        self.model=NS(training=True,forward=lambda *args,**kw:'eager')
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        def prepare(*args):self.exec.graph=self.graph
        self.stack.enter_context(patch.object(self.exec,'prepare',side_effect=prepare))
        self.attention=dict(layer_calls=[0]*4,qkv_dtypes=[],output_dtypes=[])
        self.gradient=dict(hook_calls={n:0 for n in a.NAMES},handles_removed=False,qv_exact=True,key_zero=True)
        @contextmanager
        def attention(*args):yield self.attention
        @contextmanager
        def analytic(*args):
            try:yield self.gradient
            finally:self.gradient['handles_removed']=True
        self.stack.enter_context(patch.object(a.strategy,'attention_route',attention))
        self.stack.enter_context(patch.object(a,'analytic_key_bias',analytic))

    def run_update(self,count):
        original=self.model.forward
        with self.exec.install(),a.strategy.update(None,self.model,16,count):
            for start in range(0,count,16):
                n=min(16,count-start)
                result=self.model.forward({'inputs1':NS(shape=(n,4000))},validated_inputs=True)
                self.assertEqual(result,'graph' if n==16 else 'eager')
                if n<16:
                    self.attention.update(layer_calls=[2]*4,qkv_dtypes=['torch.bfloat16'],output_dtypes=['torch.bfloat16'])
                for name in a.NAMES:self.gradient['hook_calls'][name]+=1
        self.assertIs(self.model.forward,original)

    def test_full_update_replays_and_canonical_forward_restore(self):
        original=a.strategy.update;self.run_update(128)
        self.assertIs(a.strategy.update,original);self.assertTrue(self.graph.cleared)
        self.assertEqual((self.exec.full_micros,self.exec.tail_micros,self.exec.updates),(8,0,1))

    def test_epoch_tail_is_explicit_bf16_eager(self):
        self.run_update(117)
        self.assertEqual((self.exec.full_micros,self.exec.tail_micros),(7,1))

    def test_less_than_one_micro_is_eager_without_graph_replay(self):
        self.run_update(3);self.assertEqual(self.graph.f,0)

    def test_bad_shape_and_exception_restore_every_adapter(self):
        original=self.model.forward;update=a.strategy.update
        with self.assertRaises(ValueError),self.exec.install(),a.strategy.update(None,self.model,16,128):
            self.model.forward({'inputs1':NS(shape=(32,4000))},validated_inputs=True)
        self.assertIs(self.model.forward,original);self.assertIs(a.strategy.update,update)
        self.assertTrue(self.graph.cleared)

    def test_missing_backward_coverage_rejects(self):
        with self.assertRaises(ValueError),self.exec.install(),a.strategy.update(None,self.model,16,16):pass

    def test_nested_install_rejected_and_original_restored(self):
        original=a.strategy.update
        with self.exec.install():
            with self.assertRaises(ValueError),a.Execution(self.output,{}).install():pass
        self.assertIs(a.strategy.update,original)

    def test_all_off_delegates_original_without_preparation(self):
        seen=[]
        @contextmanager
        def original(*args):seen.append(args);yield
        with patch.object(a.strategy,'update',original),a.Execution(self.output,{},False).install() as execution:
            with a.strategy.update('t','m',16,128):pass
        self.assertEqual(seen,[('t','m',16,128)]);self.assertIsNone(execution.graph)

    def test_lineage_is_separate_and_publish_has_current_pid(self):
        state={'step':10,'training_strategy':{'graph':False}}
        self.exec.publish(state);value=read(self.output/'bf16-graph-execution.json')
        self.assertEqual(state['training_strategy'],{'graph':False})
        self.assertEqual(value['step'],10);self.assertIsInstance(value['pid'],int)
        self.assertNotIn('bf16_graph_execution',w.scientific(state))


class FakeExecution:
    def __init__(self,output,implementation,enabled=True):
        self.output=output;self.implementation=implementation;self.enabled=enabled;self.n=0
    def install(self):return nullcontext(self)
    def publish(self,state):
        self.n+=1
        state['bf16_graph_execution']=dict(policy=a.POLICY,implementation=self.implementation,enabled=self.enabled)
        atomic_json(self.output/'bf16-graph-execution.json',dict(step=state['step'],worker_updates=self.n,
            enabled=self.enabled,
            full_graph_micros=8*self.n if self.enabled else 0,eager_tail_micros=0,
            runtime=dict(graph_builds=1 if self.enabled else 0,graph_resident=self.enabled,
                forward_replays=16*self.n if self.enabled else 0,backward_replays=16*self.n if self.enabled else 0)))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.suite=Path(self.temp.name)/'runs/bf16-graph-formal-fixture';self.suite.mkdir(parents=True)
        source=self.suite/'jobs'/f.JOB;source.mkdir(parents=True)
        target=old.strategy.target_config(dict(model='transformer',task='retrieval',layers=4,width=128,heads=4,
            dropout=.1,adaptive_memory=dict(execution=dict(precision='fp32'))))
        self.r=dict(source_checkpoint=dict(step=20325,contract='c'),expected_strategy={'fixture':'unchanged'},
            execution_binding={'fixture':'code'},target_config=target,data='data',cache='cache')
        self.state=dict(schema='qkan-training-state.v1',contract='c',step=20325,epoch=5,cursor=128,
            permutation=list(range(1024)),sample_clock={'progress':128},model={'w':[1.,2.]},
            optimizer={'adam':.25},rng={'torch':[3,4]},strategy_gradients={'w':[.1,.2]},
            training_strategy=self.r['expected_strategy'],strategy_updates=[{'previous':True}],
            history=[{'epoch':4}],best_weights={'w':[1.,1.]},best_accuracy=.8)
        def load(folder,contract):return read(folder/'checkpoints'/read(folder/'checkpoints/latest.json')['file'])
        def save(folder,state):
            generation=folder/'checkpoints'/('step-'+str(state['step'])+'-fixture.pt');atomic_json(generation,state)
            atomic_json(folder/'checkpoints/latest.json',dict(schema='qkan-training-checkpoint.v1',
                file=generation.name,sha256=digest(generation),step=state['step'],contract='c',previous=[]))
        save(source,self.state);self.r['source_checkpoint']['sha256']=read(source/'checkpoints/latest.json')['sha256']
        self.path=self.suite/'graph-input.json';atomic_json(self.path,self.r)
        fake=ModuleType('experiments.baseline_delivery.worker');fake.load_checkpoint=load
        fake.request_pause=lambda *_:None
        def run(config,data,cache,output,device,**kw):
            admission=kw['strategy_admission'];admission.validate(config,output,device,kw['resume'],kw['stop_after'],False,False,None)
            state=load(output,'c')
            while state['step']<kw['stop_after']:
                state['step']+=1;state['cursor']+=128;state['sample_clock']['progress']+=128
                state['model']['w'][0]+=.1
                admission.record(state,list(range(state['cursor']-128,state['cursor'])),64.,64)
            save(output,state)
            if admission.trace:
                atomic_json(output/'strategy-trace.json',dict(validation={'passed':True},model_gradients_rng_unchanged=True))
            return 75
        fake.run=run;self.fake=fake
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for obj,name,value in ((f,'verify',lambda *a,**kw:self.r),(f,'unchanged',lambda r:None),
            (old,'native_runtime',lambda:(NS(is_tensor=lambda _:False),NS(ndarray=bytes))),
            (w,'Execution',FakeExecution),(package,'worker',fake)):
            self.stack.enter_context(patch.object(obj,name,value,create=True))
        self.stack.enter_context(patch.dict(sys.modules,{'experiments.baseline_delivery.worker':fake}))

    def receipts(self):
        results={}
        for mode in f.MODES:
            output=self.suite/'preflight'/mode
            self.assertEqual(w.native(self.path,output,mode),0)
            results[mode]=read(output/'result.json')
        return results

    def test_actual_wrapper_fakes_exercise_four_modes_and_exact_recovery(self):
        source_before=read(self.suite/'jobs'/f.JOB/'checkpoints/latest.json')
        values=self.receipts();self.assertTrue(f.agreement(values)['passed'])
        self.assertEqual(values['continuous']['step'],20327)
        self.assertEqual(source_before,read(self.suite/'jobs'/f.JOB/'checkpoints/latest.json'))
        self.assertEqual(values['resume']['sample_clock']['progress'],384)

    def test_full_state_recovery_difference_is_rejected(self):
        values=self.receipts();values['resume']['component_sha256']['optimizer']='changed'
        with self.assertRaises(ValueError):f.agreement(values)

    def test_unexecuted_graph_is_not_admitted(self):
        values=self.receipts();values['continuous']['execution']['full_graph_micros']=0
        with self.assertRaises(ValueError):f.agreement(values)

    def test_invocation_paths_stop_limit_and_nonfinite_access_rejected(self):
        output=self.suite/'preflight/continuous';gate=w.Admission(self.r,self.path,output,'continuous')
        args=[self.r['target_config'],output,'cuda:0',True,20327,False,False,None]
        gate.validate(*args)
        for index,value in ((1,self.suite/'outside'),(2,'cpu'),(3,False),(4,None),(5,True),(6,True),(7,'probe')):
            changed=list(args);changed[index]=value
            with self.assertRaises(ValueError):gate.validate(*changed)

    def test_formal_gate_requires_ready_and_never_has_finite_stop(self):
        gate=w.Admission(self.r,self.path,self.suite/'jobs'/f.JOB,'formal')
        with patch.object(f,'verify',return_value=self.r) as verify:
            gate.validate(self.r['target_config'],gate.output,'cuda:0',True,None,False,False,None)
        verify.assert_called_once_with(self.path,ready=True)
        self.assertFalse(gate.trace)

    def test_advanced_unmarked_or_wrong_binding_checkpoint_rejects(self):
        w.check_lineage({},self.r,self.r['source_checkpoint']['sha256'])
        with self.assertRaises(ValueError):w.check_lineage({},self.r,'advanced')
        good=dict(bf16_graph_execution=dict(policy=a.POLICY,implementation=self.r['execution_binding'],enabled=True))
        w.check_lineage(good,self.r,'advanced')
        good['bf16_graph_execution']['enabled']=False
        with self.assertRaises(ValueError):w.check_lineage(good,self.r,'advanced')

    def test_preflight_cleanup_upload_and_no_formal_progress(self):
        values=self.receipts()
        # Exercise parent orchestration from a fresh directory using owned fake children.
        for mode in f.MODES:
            for p in (self.suite/'preflight'/mode/'checkpoints').glob('*'):p.unlink()
            (self.suite/'preflight'/mode/'checkpoints').rmdir()
        class Supervisor:
            def __init__(self,*args):self.records=[]
            def run(inner,spec,log,env):
                mode=spec['argv'][-1];output=self.suite/'preflight'/mode
                atomic_json(output/'result.json',values[mode])
                gen=output/'checkpoints/step-1-fixture.pt';gen.parent.mkdir(exist_ok=True);gen.write_bytes(b'private checkpoint')
                inner.records.append(dict(exit_code=0,outcome='exited',termination=dict(parent_reaped=True,group_verified=True,remaining_pids=[])))
                return 0
        with patch.object(diagnostic_watch,'Supervisor',Supervisor),patch.object(return_artifacts,'upload_declared') as upload:
            f.preflight(self.suite,self.r,'0')
        self.assertTrue(read(self.suite/'graph-ready.json')['passed'])
        self.assertEqual(upload.call_count,1)
        self.assertEqual(list((self.suite/'preflight').glob('*/checkpoints/step-*.pt')),[])
        self.assertEqual(read(self.suite/'jobs'/f.JOB/'checkpoints/latest.json')['step'],20325)

    def test_failed_child_auto_returns_partial_before_any_formal_start(self):
        class Supervisor:
            def __init__(self,*args):self.records=[]
            def run(inner,spec,log,env):
                output=Path(spec['argv'][spec['argv'].index('--output')+1])
                atomic_json(output/'worker-error.json',dict(exception='FixtureFailure'))
                return 2
        with patch.object(diagnostic_watch,'Supervisor',Supervisor),patch.object(return_artifacts,'upload_declared') as upload:
            with self.assertRaises(RuntimeError):f.preflight(self.suite,self.r,'0')
        self.assertFalse((self.suite/'graph-ready.json').exists());upload.assert_called_once()
        report=read(self.suite/'preflight/preflight.json')
        self.assertFalse(report['passed']);self.assertIn('reference',report['failure_locations'])


class DispatchTests(unittest.TestCase):
    def test_explicit_profile_and_existing_suite_resume(self):
        args=NS(profile='bf16-graph',migrate_from='original',graph_from='reviewed',worker_from=None,output=None,cache='cache',gpu='0')
        command=e.start_command(args,Path('.'),None)
        self.assertIn('--prepare-only',command);self.assertIn('--diagnostic-from',command)
        self.assertIn('experiments.baseline_delivery.bf16_graph_formal',command)
        with patch.object(e,'validate_executor',return_value='bf16-graph'):
            self.assertNotIn('--prepare-only',e.resume_command(Path('.'),Path('suite')))
            self.assertIn('--upload-only',e.resume_command(Path('.'),Path('suite'),True))

    def test_invalid_graph_profile_never_falls_back(self):
        args=NS(profile='bf16-graph',migrate_from='source',graph_from=None,worker_from=None,output=None,cache='cache',gpu='0')
        with self.assertRaises(ValueError):e.start_command(args,Path('.'),None)
        args.graph_from='reviewed';args.profile='formal'
        with self.assertRaises(ValueError):e.start_command(args,Path('.'),None)

    def test_other22_use_original_dispatch_and_candidate_new_child(self):
        with patch.object(old,'job_spec',return_value='original') as job:
            self.assertEqual(f.job_spec(Path('suite'),{},dict(id='s5-image-s42')),'original')
            spec=f.job_spec(Path('suite'),{},dict(id=f.JOB))
        job.assert_called_once();self.assertEqual(spec['target'],f.WORKER);self.assertEqual(spec['argv'][-1],'formal')

    def test_native_diagnostic_29_file_bindings_are_byte_identical(self):
        review=read(f.REVIEW)
        self.assertEqual(len(review['executed_bindings']),29)
        for name,sha in review['executed_bindings'].items():self.assertEqual(digest(f.ROOT/name),sha,name)
        self.assertEqual(f.diagnostic.binding(),review['implementation'])

    def test_reviewed_native_report_is_reused_without_gpu(self):
        path=f.ROOT/'.runtime/received-20261118/bf16-graph/bf16-graph-20261118T104322Z-8870aa95'
        if not path.exists():self.skipTest('Private received report is local only')
        self.assertEqual(f.bind_diagnostic(path)['sha256'],read(f.REVIEW)['source_report']['sha256'])

    def test_native_children_actual_help_is_stdlib_only(self):
        for module in (f.TARGET,f.WORKER):
            result=subprocess.run([sys.executable,'-B','-X','utf8','-m',module,'--help'],capture_output=True,text=True,encoding='utf8')
            self.assertEqual(result.returncode,0,result.stderr)

    def test_graph_start_uses_integrated_supervisor_and_automatic_after_prepare(self):
        with patch.object(e,'supervise',return_value=75) as run,patch.object(e,'display_module',return_value='original-dashboard'):
            code=e.main(['start','--profile','bf16-graph','--migrate-from','original','--graph-from','reviewed','--cache','cache'])
        self.assertEqual(code,75);self.assertIsNone(run.call_args.args[1])
        self.assertEqual(run.call_args.args[3],'original-dashboard');self.assertTrue(run.call_args.args[-1])

    def test_real_parent_supervises_new_no_data_child_and_original_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);suite,path,r,jobs=fixture(root)
            gen=suite/'jobs'/f.JOB/'checkpoints/step-5-fixture.pt';gen.parent.mkdir();gen.write_bytes(b'fixture')
            atomic_json(gen.parent/'latest.json',dict(schema='qkan-training-checkpoint.v1',file=gen.name,
                sha256=digest(gen),step=5,contract='c',previous=[]))
            spec=f.child_spec(suite,'formal');spec['argv']=spec['argv'][:4]+['--help']
            statuses=read(suite/'status.json')
            with patch('scripts.formal_progress.Terminal') as terminal,patch('builtins.print'):
                code=old.run_job(suite,r,{'id':f.JOB},'0',statuses,
                    spec_builder=lambda *args:spec,result_validator=lambda *args:None)
            self.assertEqual(code,0);self.assertEqual(read(suite/'status.json')[f.JOB]['state'],'computed')
            self.assertIn('任务 25',terminal.return_value.show.call_args.args[0])


class DisplayTests(unittest.TestCase):
    def test_only_current_process_execution_proof_displays_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);suite=root/'runs/bf16-graph-formal-fixture';folder=suite/'jobs'/f.JOB
            atomic_json(suite/'graph-input.json',{})
            atomic_json(suite/'suite.json',dict(jobs=[{'id':f.JOB}]))
            atomic_json(suite/'status.json',{f.JOB:dict(state='running',pid=123,started=10)})
            atomic_json(folder/'config.json',dict(training_strategy=a.strategy.POLICY,epochs=100))
            atomic_json(folder/'heartbeat.json',dict(epoch=1,optimizer_step=20326,updated_at=20,
                samples_seen=256,sample_budget=10000,run_started_at=10,run_samples_start=128))
            proof=dict(enabled=True,pid=123,updated_at=20,full_graph_micros=8,eager_tail_micros=1,
                runtime=dict(graph_builds=1,forward_replays=16,backward_replays=16))
            atomic_json(folder/'bf16-graph-execution.json',proof)
            before={p.relative_to(suite):p.read_bytes() for p in suite.rglob('*') if p.is_file()}
            text=ui.Dashboard(root,suite).render(20)
            self.assertIn('CUDA Graph（按文档',text);self.assertIn('短尾 eager 1',text)
            self.assertIn('16/16',text)
            self.assertEqual(before,{p.relative_to(suite):p.read_bytes() for p in suite.rglob('*') if p.is_file()})
            atomic_json(folder/'bf16-graph-execution.json',dict(proof,pid=99))
            self.assertIn('eager（无 Graph）',ui.Dashboard(root,suite).render(20))
            self.assertEqual(ui.latest_formal(root),suite)

    def test_preflight_is_visible_without_calling_it_training_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);atomic_json(suite/'graph-input.json',{})
            atomic_json(suite/'preflight/progress.json',dict(mode='resume',completed=3))
            lines='\n'.join(ui.startup_lines(suite,{},20))
            self.assertIn('已完成 3/4',lines);self.assertIn('尚未推进正式训练',lines)


class ForkTests(unittest.TestCase):
    def test_new_fork_copies_latest_checkpoint_configs_and_completed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);source,path,inherited,jobs=fixture(root)
            target=root/'runs/bf16-graph-formal-fixture';target.mkdir()
            completed=('transformer-listops-s42','transformer-text-s42')
            statuses={j['id']:dict(state='computed' if j['id'] in completed else 'paused' if j['id']==f.JOB else 'pending') for j in jobs}
            atomic_json(source/'status.json',statuses)
            generation=source/'jobs'/f.JOB/'checkpoints/step-20325-fixture.pt'
            generation.parent.mkdir();generation.write_bytes(b'actual latest full-state values fixture')
            cp=dict(schema='qkan-training-checkpoint.v1',file=generation.name,sha256=digest(generation),
                step=20325,contract='contract',previous=[])
            atomic_json(generation.parent/'latest.json',cp)
            bound=dict(source_executor_revision=read(f.REVIEW)['source_executor_revision'],
                checkpoint=dict(cp,path=str(generation)),source_metadata_sha256={},
                source_config=inherited['target_config'],data='data',cache='cache')
            native=dict(reviewed_source_checkpoint=dict(step=20325,contract='contract'),suite='reviewed')
            def complete(folder,config):
                return dict(config_sha256=fingerprint(config),files={'config.json':digest(folder/'config.json')})
            prior={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            execution=f.binding()
            for obj,name,value in ((f,'ROOT',root),(f,'revision',lambda:'new'),(f,'binding',lambda:execution),
                (f.diagnostic,'bind_source',lambda *args:bound),(f,'bind_diagnostic',lambda *args:native),
                (old,'original_executor',lambda r:None),(f,'completed_files',complete)):
                stack.enter_context(patch.object(obj,name,value))
            request=f.prepare(NS(source_suite=source,cache=Path('cache'),diagnostic_from=Path('reviewed')),target)
            self.assertEqual(request['target_config'],inherited['target_config'])
            self.assertEqual(request['checkpoint'],inherited['checkpoint'])
            self.assertEqual(digest(target/'jobs'/f.JOB/'checkpoints'/generation.name),cp['sha256'])
            self.assertEqual(len([j for j in read(target/'suite.json')['jobs'] if j.get('preserved_completion')]),2)
            for job in jobs:
                self.assertEqual((source/'jobs'/job['id']/'config.json').read_bytes(),(target/'jobs'/job['id']/'config.json').read_bytes())
            self.assertEqual(prior,{p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()})
            self.assertFalse((target/'selection').exists());self.assertFalse((target/'graph-ready.json').exists())


if __name__=='__main__':unittest.main()
