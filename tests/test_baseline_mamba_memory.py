"""Focused memory-for-speed interfaces and scheduling, without native imports."""
from contextlib import ExitStack,contextmanager,redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import mamba_memory_benchmark as m,mamba_diagnostic as d
from experiments.baseline_delivery.common import atomic_json,read,digest


class Module:
    def __call__(self,*args):return self.forward(*args)


class Ids(list):
    def __getitem__(self,key):
        value=super().__getitem__(key);return Ids(value) if isinstance(key,slice) else value
    def numpy(self):return self
    def copy(self):return Ids(self)
    def tolist(self):return list(self)


class Tests(unittest.TestCase):
    def test_graph_wrapper_never_replaces_eager_reference(self):
        class Ordinary(Module):
            def forward(self,value):return value+1
        ordinary=Ordinary();c=dict(torch=SimpleNamespace(nn=SimpleNamespace(Module=Module)),forward=ordinary)
        adapter=m.graph_adapter(c);adapter.forward=lambda value:value+10
        self.assertEqual(adapter(1),11);self.assertEqual(ordinary(1),2)

    def test_explicit_routes_and_only_independent_successes_combine(self):
        self.assertEqual(m.options('memory_retain',{}),('retain',))
        for value in ([],['graph'],['graph','graph'],['graph','unknown']):
            with self.assertRaises(ValueError):m.options(m.COMBINED,dict(memory_combination=value))
        good=dict(measurement_completed=True,timing=dict(timing_promising=True),numerics=dict(rng_and_scheduler_exact=True))
        results={m.METHODS[0]:good,m.METHODS[1]:good,m.METHODS[2]:{**good,'status':'failed'}}
        self.assertEqual(m.combination(results),['retain','prefetch'])
        spec=d.child_spec(Path('runs/fixture'),m.COMBINED)
        self.assertIn(str(Path('runs/fixture/combined-request.json')),spec['argv'])

    def test_gpu_double_buffer_waits_for_host_copy_and_compute_before_reuse(self):
        events=[];serial={'event':0}
        class Tensor:
            dtype='int64'
            def __init__(self,ids,pinned=False):self.ids=list(ids);self.shape=(len(ids),2);self.pinned=pinned
            def __len__(self):return len(self.ids)
            def __getitem__(self,key):return Tensor(self.ids[key],self.pinned)
            def copy_(self,other,**kwargs):self.ids=list(other.ids);events.append(('copy',self.pinned,tuple(self.ids)));return self
            def is_pinned(self):return self.pinned
            def record_stream(self,stream):events.append(('record_stream',stream.name))
        class Event:
            def __init__(self):serial['event']+=1;self.id=serial['event']
            def record(self,stream):events.append(('record_event',self.id,stream.name))
            def synchronize(self):events.append(('sync_event',self.id))
        class Stream:
            def __init__(self,name='copy'):self.name=name
            def wait_event(self,event):events.append(('wait_event',self.name,event.id))
        current=Stream('compute')
        @contextmanager
        def stream(value):yield
        cuda=SimpleNamespace(Stream=Stream,Event=Event,current_stream=lambda:current,stream=stream,synchronize=lambda:None)
        torch=SimpleNamespace(cuda=cuda,as_tensor=lambda v:v,
            empty=lambda shape,**kw:Tensor([0]*shape[0],kw.get('pin_memory',False)),empty_like=lambda v,**kw:Tensor([0]*len(v)))
        seen=[]
        def load(ids,validate):
            self.assertIsInstance(ids,list);seen.append(tuple(ids));return {'inputs':Tensor(ids)}
        pipeline=m.InputPipeline(torch,SimpleNamespace(numpy_batch=load),[Ids([i]*32) for i in range(3)],True)
        try:
            pipeline.begin()
            for i in range(3):
                self.assertEqual(pipeline.get(i)['inputs'].ids,[i]*32)
                pipeline.after_forward(i);pipeline.consumed(i)
        finally:pipeline.close()
        self.assertEqual(seen,[tuple([i]*32) for i in range(3)])
        self.assertEqual(pipeline.copies,3);self.assertEqual(pipeline.overlapped_submissions,2)
        self.assertEqual(len(pipeline.buffers),2)
        first_done=next(e[1] for e in events if e[0]=='record_event' and e[2]=='compute')
        self.assertIn(('wait_event','copy',first_done),events)
        self.assertIn(('sync_event',1),events)
        self.assertIsNone(pipeline.prefetch)

    def test_all_method_drivers_restore_bound_blocks_and_report_real_calls(self):
        for route in (*m.METHODS,m.COMBINED):
            with self.subTest(route=route),tempfile.TemporaryDirectory() as tmp:
                request=Path(tmp)/'request.json';atomic_json(request,dict(checkpoint=dict(sha256='latest'),memory_combination=['retain','graph']))
                clock={'time':0.};state={'dirty':False};calls=[];restores=[]
                class Policy:
                    calls=0;level_calls={0:0,1:0};mechanism={};level=1
                    def set_checkpoint_level(self,value):self.level=value;self.mechanism['checkpoint_lvl']=value
                policy=Policy()
                class Pipe:
                    def __init__(self,t,train,batches,overlap):self.overlap=overlap;self.copies=0;self.overlapped_submissions=0
                    def begin(self):pass
                    def get(self,i):self.copies+=1;return {'slot':i,'overlap':self.overlap}
                    def after_forward(self,i):self.overlapped_submissions+=int(self.overlap and i<3)
                    def consumed(self,i):pass
                    def end(self):pass
                    def close(self):pass
                cuda=SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,
                    max_memory_allocated=lambda:100,max_memory_reserved=lambda:200,
                    make_graphed_callables=lambda adapter,args,**kw:adapter)
                class Forward(Module):
                    def forward(self,*args):return 1
                class Train:
                    def __len__(self):return 1000
                    def batch(self,ids,device):return {'inputs':1}
                def restore():state['dirty']=False;restores.append(True)
                def exact():self.assertFalse(state['dirty'])
                def update(candidate,**kw):
                    calls.append(kw);state['dirty']=True;policy.calls+=4;policy.level_calls[policy.level]+=4
                    self.assertEqual(kw['micro_batch'],8);self.assertEqual(len(kw['batch_ids']),32)
                    kw['after_forward']()
                    if kw['forward_override'] is not None:
                        for _ in range(4):kw['forward_override'](1)
                    faster=policy.level==0 or kw['staged_batch']['overlap'] or kw['graph_gradients']
                    clock['time']+=.5 if faster else 1.;return .1
                c=dict(torch=SimpleNamespace(cuda=cuda,nn=SimpleNamespace(Module=Module)),projection=policy,
                    logical=32,state=dict(cursor=0,permutation=Ids(range(1000)),model={}),train=Train(),
                    restore=restore,exact=exact,update=update,output=lambda:dict(model={},optimizer={},rng={},scheduler={}),
                    host=lambda v:v,model=SimpleNamespace(named_parameters=lambda:[]),forward=Forward(),inputs=('inputs',))
                with patch.object(m,'InputPipeline',Pipe),patch.object(m,'numeric_summary',return_value={'rng_and_scheduler_exact':True}),\
                        patch.object(m.time,'perf_counter',side_effect=lambda:clock['time']):
                    result=m.operation(c,request,route)
                self.assertEqual(len(calls),44);self.assertEqual(result['disposable_updates'],44)
                self.assertEqual(result['samples_per_block'],128);self.assertTrue(result['timing']['timing_promising'])
                self.assertEqual(result['timing']['positive_pairs'],4);self.assertEqual(policy.level,1);exact()
                self.assertEqual(len(result['reference_seconds']),4)
                self.assertEqual(result['formal_updates_committed'],0)
                self.assertEqual(read(request.parent/(route+'.json')),result)
                if route in ('memory_graph',m.COMBINED):
                    self.assertGreater(result['graph_forward_replays'],0)
                    self.assertTrue(any(kw['graph_gradients'] for kw in calls))

    def test_parent_isolates_failure_keeps_request_binding_and_returns_all_three(self):
        for combine in (False,True):
            with tempfile.TemporaryDirectory() as tmp,ExitStack() as held:
                root=Path(tmp);calls=[];uploads=[];initial=[]
                def snapshot(source,suite,job,cache):
                    atomic_json(suite/'source/checkpoints/step-1.pt',{'disposable':True})
                    request=dict(source_files={},checkpoint=dict(sha256='latest'));atomic_json(suite/'request.json',request);return request
                class Supervisor:
                    def __init__(self,*args):self.records=[]
                    def run(self,spec,log,env):
                        route=spec['argv'][spec['argv'].index('--route')+1];calls.append(route)
                        source=Path(spec['argv'][spec['argv'].index('--request')+1])
                        if not initial:initial.append(digest(source))
                        if route==m.COMBINED:
                            self_outer.assertEqual(read(source)['memory_combination'],['retain','prefetch'])
                            self_outer.assertEqual(digest(log.parent/'request.json'),initial[0])
                        failed=route==m.METHODS[2]
                        atomic_json(log.parent/(route+'.json'),dict(measurement_completed=not failed,
                            request_sha256=digest(source),timing=dict(throughput_gain=.1,timing_promising=combine),
                            numerics=dict(rng_and_scheduler_exact=True)))
                        self.records.append(dict(termination=dict(group_verified=True),exit_code=1 if failed else 0))
                        return 1 if failed else 0
                self_outer=self
                held.enter_context(patch.object(d,'ROOT',root));held.enter_context(patch.object(d,'snapshot',side_effect=snapshot))
                held.enter_context(patch('experiments.baseline_delivery.bootstrap.check_exchange'))
                held.enter_context(patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'GPU'})))
                held.enter_context(patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}))
                held.enter_context(patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',Supervisor))
                held.enter_context(patch('experiments.baseline_delivery.mamba_convolution.stage',return_value={'fixture':True}))
                held.enter_context(patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *args:uploads.append(args)))
                held.enter_context(patch.object(sys,'argv',['mamba','--source-suite',str(root/'source'),'--cache',str(root/'cache'),'--memory-benchmark']))
                held.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(d.main(),0)
                self.assertEqual(calls,list(m.METHODS)+([m.COMBINED] if combine else []))
                report=read(uploads[0][0]/'mamba-memory-benchmark.json')
                self.assertTrue(report['source_unchanged']);self.assertTrue(report['copied_checkpoint_removed'])
                self.assertTrue(report['campaign_completed']);self.assertFalse(report['all_methods_measured'])
                self.assertEqual(report['routes']['memory_graph']['status'],'failed')
                self.assertEqual(uploads[0][1:3],('mamba-memory-benchmark',['mamba-memory-benchmark.json']))


if __name__=='__main__':unittest.main()
