"""Stdlib fixtures for scheduling, throttling and reusable tensor-copy contracts."""
import unittest
from types import SimpleNamespace
from experiments.baseline_delivery.execution import BatchPrefetch,RollbackBuffer,StatusWriter


class Tensor:
    def __init__(self,value,device='cuda:0'):
        self.value=value;self.device=device;self.shape=(1,);self.dtype='float32'
    def detach(self): return self
    def to(self,device,copy): return Tensor(self.value,device)
    def copy_(self,source): self.value=source.value;return self


class ExecutionTests(unittest.TestCase):
    def test_one_batch_ahead_preserves_order(self):
        loaded=[]
        def load(ids): loaded.append(ids);return list(ids)
        with BatchPrefetch(load) as prefetch:
            self.assertEqual(prefetch.get([3,1]),[3,1])
            prefetch.submit([9,2]);self.assertEqual(prefetch.get([9,2]),[9,2])
        self.assertEqual(loaded,[(3,1),(9,2)])
        self.assertTrue(prefetch.pool._shutdown)

    def test_mismatched_sampler_ids_fail(self):
        with BatchPrefetch(list) as prefetch:
            prefetch.submit([1,2])
            with self.assertRaises(ValueError): prefetch.get([2,1])

    def test_prefetch_failure_propagates_and_thread_closes(self):
        def fail(ids): raise ValueError('injected read failure')
        with self.assertRaises(ValueError):
            with BatchPrefetch(fail) as prefetch:
                prefetch.submit([1]);prefetch.get([1])
        self.assertTrue(prefetch.pool._shutdown)

    def test_no_unbounded_prefetch_queue(self):
        with BatchPrefetch(list) as prefetch:
            prefetch.submit([1])
            with self.assertRaises(RuntimeError): prefetch.submit([2])

    def test_status_interval_transition_and_pause(self):
        now=[0.];writes=[]
        writer=StatusWriter(writes.append,5,lambda:now[0])
        emit=lambda write:write('snapshot')
        self.assertTrue(writer.emit(emit,token=0))
        now[0]=1;self.assertFalse(writer.emit(emit,token=0))
        self.assertTrue(writer.emit(emit,token=1))
        self.assertFalse(writer.emit(emit,token=1))
        now[0]=6;self.assertTrue(writer.emit(emit,token=1))
        self.assertTrue(writer.emit(emit,force=True,token=1))
        self.assertEqual(len(writes),4)

    def test_gpu_snapshot_reuses_storage_without_aliasing_live_state(self):
        buffer=RollbackBuffer(SimpleNamespace(Tensor=Tensor))
        live=Tensor(1);first=buffer.capture({'weight':live,'optimizer':[Tensor(3)]})
        backup=first['weight'];self.assertIsNot(live,backup)
        live.value=2;self.assertEqual(backup.value,1)
        second=buffer.capture({'weight':live,'optimizer':[Tensor(4)]})
        self.assertIs(second['weight'],backup);self.assertEqual(backup.value,2)
        self.assertEqual(second['optimizer'][0].device,'cuda:0')

    def test_cpu_fallback_and_new_optimizer_fields(self):
        buffer=RollbackBuffer(SimpleNamespace(Tensor=Tensor),cpu=True)
        first=buffer.capture({'state':{}})
        second=buffer.capture({'state':{1:Tensor(7)}})
        self.assertEqual(second['state'][1].device,'cpu')
        self.assertEqual(first,{'state':{}})
        buffer.release();self.assertIsNone(buffer.buffer)


if __name__=='__main__': unittest.main()
