"""Graph partition and retry checks using Python stand-ins, never native Torch."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from experiments.baseline_delivery.adaptive_memory import (
    Controller,Exhausted,default_policy,transaction,validate_capacity_recovery)
from experiments.baseline_delivery.logical_batch import JointController,policy as logical_policy
from experiments.baseline_delivery.torch_acceleration import (
    CaptureCapacity,Executor,backend_diagnostic,graph_rng_budget,policy,torch_runtime)


CONFIG=dict(model='transformer',task='retrieval',width=128,heads=4,layers=4,dropout=.1)


class Tensor:
    def __init__(self,rows=16,length=4000,name='a'):
        self.shape=(rows,length);self.dtype='int64';self.device='cuda:0';self.name=name
    def detach(self): return self
    def clone(self): return Tensor(*self.shape,self.name)


class Module:
    def __call__(self,*args,**kwargs): return self.forward(*args,**kwargs)


class PartitionTests(unittest.TestCase):
    def setup_runtime(self):
        rng=[41];events=[];captures=[];restores=[]
        class Model:
            config=dict(CONFIG)
            def parameters(self): return []
            def encode(self,x,validated_inputs=False):
                assert validated_inputs
                rng[0]+=1;events.append(x.name)
                return (x.name,rng[0])
            def classify_pair(self,a,b): return (a,b)
            def __call__(self,batch,validated_inputs=False):
                return self.classify_pair(self.encode(batch['inputs1'],validated_inputs),
                                          self.encode(batch['inputs2'],validated_inputs))
        def capture(adapter,static,**kwargs):
            self.assertEqual(kwargs,dict(num_warmup_iters=3,allow_unused_input=True))
            self.assertIsInstance(adapter,Module)  # Separate calls/pools, never tuple pooling.
            captures.append((adapter,static));adapter(*static)
            return lambda *values:adapter(*values)
        torch=SimpleNamespace(nn=SimpleNamespace(Module=Module),
            compile=Mock(side_effect=lambda forward,**kwargs:forward),
            cuda=SimpleNamespace(make_graphed_callables=Mock(side_effect=capture),
                synchronize=Mock(),empty_cache=Mock(),OutOfMemoryError=type('OOM',(RuntimeError,),{})))
        def restore(value): restores.append(value);rng[0]=value
        model=Model();runtime=torch_runtime(torch,model,lambda:rng[0],restore,policy(backend='eager'))
        runtime.set_mode('compile_graph');runtime.begin_update(16)
        self.runtime=runtime;self.torch=torch;self.model=model;self.rng=rng
        self.events=events;self.captures=captures;self.restores=restores

    def batch(self,rows=16): return dict(inputs1=Tensor(rows,name='a'),inputs2=Tensor(rows,name='b'))

    def test_whole_retrieval_exceeds_uint32_but_two_document_graphs_fit(self):
        self.assertEqual(graph_rng_budget(CONFIG,16,4000,2)['estimated_increment'],8585216000)
        self.assertFalse(graph_rng_budget(CONFIG,16,4000,2)['passed'])
        single=graph_rng_budget(CONFIG,16,4000)
        self.assertEqual(single['estimated_increment'],4292608000)
        self.assertTrue(single['passed']);self.assertEqual(single['limit'],4294967295)
        self.assertTrue(graph_rng_budget(CONFIG,4,4000,2)['passed'])
        self.assertFalse(graph_rng_budget(CONFIG,32,4000)['passed'])

    def test_budget_applies_only_to_native_transformer_dropout(self):
        for config in (dict(CONFIG,dropout=0),dict(CONFIG,model='mamba1'),None):
            self.assertIsNone(graph_rng_budget(config,16,4000))

    def test_two_independent_graphs_replay_in_native_document_order_with_rng_restored(self):
        self.setup_runtime();expected=self.model(self.batch(),validated_inputs=True);self.rng[0]=41
        self.assertEqual(self.runtime.forward(self.batch(),'compile_graph'),expected)
        self.assertEqual(len(self.captures),2);self.assertEqual(self.restores,[41])
        self.assertIsNot(self.captures[0][0],self.captures[1][0])
        self.assertIsNot(self.captures[0][1][0],self.captures[1][1][0])
        self.assertEqual(self.events[-2:],['a','b']);self.assertEqual(self.rng[0],43)
        self.torch.compile.assert_called_once_with(self.model.encode,backend='eager',fullgraph=True,dynamic=False)
        status=self.runtime.status()
        self.assertEqual(status['graph_callables'],2);self.assertEqual(status['graph_replays'],2)
        self.assertEqual(status['capture_partition'],'retrieval_documents')
        self.assertEqual(status['compiler_warmups'],2);self.assertEqual(status['graph_builds'],1)
        self.assertTrue(status['graph_resident']);self.assertTrue(status['capture_budget']['passed'])

    def test_same_pair_and_rare_tail_keep_resident_graphs(self):
        self.setup_runtime();self.runtime.forward(self.batch(),'compile_graph');graphs=self.runtime.graph
        for rows in (16,3,16): self.runtime.forward(self.batch(rows),'compile_graph')
        self.assertIs(self.runtime.graph,graphs);self.assertEqual(len(self.captures),2)
        self.assertEqual(self.runtime.status()['eager_tail_calls'],1)
        self.assertEqual(self.runtime.status()['graph_replays'],6)

    def test_shape_replaces_pair_and_keeps_generation_budget(self):
        self.setup_runtime();self.runtime.forward(self.batch(),'compile_graph')
        self.runtime.begin_update(8);self.runtime.forward(self.batch(8),'compile_graph')
        self.assertEqual(len(self.captures),4);self.assertEqual(self.runtime.status()['graph_callables'],2)
        self.assertEqual(self.runtime.generation(),2)

    def test_oversize_is_refused_before_compile_capture_or_rng_mutation(self):
        self.setup_runtime();self.runtime.begin_update(32)
        with self.assertRaises(CaptureCapacity) as failure:
            self.runtime.forward(self.batch(32),'compile_graph')
        self.assertFalse(failure.exception.diagnostic['passed'])
        self.assertEqual(failure.exception.diagnostic['stage'],'capture_budget')
        self.torch.compile.assert_not_called();self.torch.cuda.make_graphed_callables.assert_not_called()
        self.assertEqual(self.rng[0],41);self.assertEqual(self.restores,[])

    def test_second_capture_failure_restores_rng_and_preserves_native_oom(self):
        self.setup_runtime();native=self.torch.cuda.OutOfMemoryError('allocation')
        original=self.torch.cuda.make_graphed_callables.side_effect
        def fail_second(*args,**kwargs):
            if self.captures: raise native
            return original(*args,**kwargs)
        self.torch.cuda.make_graphed_callables.side_effect=fail_second
        with self.assertRaises(self.torch.cuda.OutOfMemoryError) as failure:
            self.runtime.forward(self.batch(),'compile_graph')
        self.assertIs(failure.exception,native);self.assertEqual(self.rng[0],41)
        self.assertIsNone(self.runtime.graph)
        self.assertEqual(native.diagnostic['stage'],'graph_capture')

    def test_second_document_budget_is_checked_before_any_capture(self):
        self.setup_runtime();batch=self.batch();batch['inputs2']=Tensor(16,5000,'b')
        with self.assertRaises(CaptureCapacity): self.runtime.forward(batch,'compile_graph')
        self.torch.compile.assert_not_called();self.assertEqual(self.captures,[])
        self.assertEqual(self.runtime.capture_budget['length'],5000)

    def test_rng_offset_diagnostic_is_specific_without_raw_message(self):
        for message,category in [('Increment causes overflow in the offset value.','rng_offset_overflow'),
                                 ('RNG offset must be a multiple of 4.','rng_offset_alignment')]:
            value=backend_diagnostic(RuntimeError(message+' private contents'),'graph_capture','eager')
            self.assertEqual(value['exceptions'][0]['category'],category)
            self.assertNotIn('private contents',str(value))


class CapacityRetryTests(unittest.TestCase):
    def controller(self):
        c=JointController(logical_policy(32),32)
        c.state.update(index=c.policy['tiers'].index(32),logical_index=2)
        return c

    def test_declared_capacity_rolls_back_entire_update_without_lowering_logical_batch(self):
        controller=self.controller();state=dict(model=1,adam=2,rng=3,scheduler=4,cursor=5)
        before=copy.deepcopy(state);attempts=[];cleanups=[]
        def update(micro):
            attempts.append((micro,copy.deepcopy(state)))
            for key in ('model','adam','rng','scheduler'): state[key]+=1
            if micro>16: raise CaptureCapacity(graph_rng_budget(CONFIG,micro,4000))
            return 'committed once'
        def restore(value): state.clear();state.update(value)
        result,status=transaction(controller,update,lambda:copy.deepcopy(state),restore,
            lambda:cleanups.append(True),lambda _:False,lambda:{},lambda:None,samples=128,
            is_capacity=lambda error:isinstance(error,CaptureCapacity))
        self.assertEqual(result,'committed once');self.assertEqual([a[0] for a in attempts],[32,16])
        self.assertEqual(attempts[0][1],attempts[1][1]);self.assertEqual(len(cleanups),1)
        self.assertEqual(state,{key:value+(key!='cursor') for key,value in before.items()})
        self.assertEqual(status['used_logical_batch'],128);self.assertEqual(controller.logical_batch,128)
        self.assertEqual(status['used_micro_batch'],16);self.assertEqual(status['oom_retries'],0)
        self.assertEqual(status['graph_capacity_retries'],1)

    def test_known_capacity_ceiling_survives_cooldown_and_checkpoint_restore(self):
        controller=self.controller();controller.capacity()
        saved=controller.snapshot();restored=JointController(controller.policy,32,saved)
        memory=dict(total_bytes=24*1024**3,free_bytes=20*1024**3,peak_bytes=1024**3)
        for c in (controller,restored):
            c.used_samples=128
            for _ in range(1200): c.observe(1,memory)
            self.assertEqual(c.micro_batch,16);self.assertEqual(c.logical_batch,128)
            self.assertEqual(c.state['capacity_count'],1)
        self.assertEqual(controller.snapshot(),restored.snapshot())

    def test_smallest_capacity_tier_exhausts_finitely_without_counting_oom(self):
        controller=Controller(default_policy(32),32);attempts=[]
        def fail(micro): attempts.append(micro);raise CaptureCapacity({})
        with self.assertRaises(Exhausted):
            transaction(controller,fail,lambda:None,lambda _:None,lambda:None,lambda _:False,
                lambda:{},lambda:None,is_capacity=lambda error:isinstance(error,CaptureCapacity))
        self.assertEqual(attempts,[4,2,1]);self.assertEqual(controller.state['oom_count'],0)

    def test_ordinary_oom_cooldown_still_respects_known_graph_capacity(self):
        controller=self.controller();controller.capacity();controller.oom()
        self.assertEqual(controller.micro_batch,8);controller.used_samples=128
        memory=dict(total_bytes=24*1024**3,free_bytes=20*1024**3,peak_bytes=1024**3)
        for _ in range(1200): controller.observe(1,memory)
        self.assertLessEqual(controller.micro_batch,16)
        self.assertEqual(controller.state['oom_count'],1);self.assertEqual(controller.state['capacity_count'],1)

    def test_fresh_worker_capacity_mismatch_cannot_pass_recovery(self):
        validate_capacity_recovery({},{});validate_capacity_recovery({'capacity_count':1},{'capacity_count':1})
        for key in ('capacity_count','capacity_ceiling'):
            for left,right in (({key:1},{key:2}),({},{key:1})):
                with self.assertRaisesRegex(ValueError,key): validate_capacity_recovery(left,right)

    def test_capacity_bubbles_out_of_executor_without_backend_fallback(self):
        runtime=Mock();runtime.generation.return_value=0
        executor=Executor(runtime,policy(backend='eager',required='graph'),synthetic=True)
        error=CaptureCapacity(graph_rng_budget(CONFIG,32,4000))
        with self.assertRaises(CaptureCapacity):
            executor.run(Mock(side_effect=error),{},Mock(),Mock(),Mock(),lambda _:False)
        self.assertEqual(executor.state['mode'],'compile_graph');runtime.clear.assert_called_once()
        self.assertEqual(executor.state['reason'],'graph_rng_capacity_retry_smaller_microbatch')

    def test_unknown_runtime_error_never_uses_capacity_retry(self):
        controller=self.controller();update=Mock(side_effect=RuntimeError('generator capture failure'))
        with self.assertRaises(RuntimeError):
            transaction(controller,update,lambda:None,lambda _:None,lambda:None,lambda _:False,
                lambda:{},lambda:None,is_capacity=lambda error:isinstance(error,CaptureCapacity))
        self.assertEqual(update.call_count,1);self.assertNotIn('capacity_count',controller.state)


if __name__=='__main__': unittest.main()
