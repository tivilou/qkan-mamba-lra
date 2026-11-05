"""Long-sequence replay retains batch shape, dtype, masks and pair semantics."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from experiments.baseline_delivery.trace import replay_classifier


class ReplayTests(unittest.TestCase):
    def test_full_length_retrieval_replay_matches_batched_producer(self):
        torch.manual_seed(17)
        h=[torch.randn(2,4000,128) for _ in range(2)]
        masks=[torch.ones(2,4000,dtype=torch.bool) for _ in range(2)]
        masks[0][0,3001:]=False; masks[1][1,2107:]=False
        for side in range(2): h[side][~masks[side]]=0
        a,b=[x.sum(1)/m.sum(1).unsqueeze(-1) for x,m in zip(h,masks)]
        expected=torch.cat(((a-b).abs(),a*b),dim=-1)
        head=torch.nn.Linear(256,2)
        rows=[{**{'encoder'+str(s):h[s][i].numpy() for s in range(2)},
               **{'mask'+str(s):masks[s][i].numpy() for s in range(2)}} for i in range(2)]
        pooled,logits=replay_classifier(SimpleNamespace(head=head),rows,'retrieval','cpu')
        np.testing.assert_array_equal(pooled,expected.numpy())
        np.testing.assert_array_equal(logits,head(expected).detach().numpy())
        self.assertEqual(pooled.dtype,np.float32)
        # A real changed representation must remain detectable at the old tolerance.
        corrupted=pooled.copy();corrupted[0,0]+=.01
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(corrupted,expected.numpy(),rtol=1e-5,atol=1e-6)

    def test_single_document_and_invalid_padding(self):
        h=np.ones((16,4),dtype=np.float32);mask=np.ones(16,dtype=bool)
        mask[-3:]=False;h[-3:]=0
        head=torch.nn.Linear(4,2)
        row={'encoder0':h,'mask0':mask}
        pooled,_=replay_classifier(SimpleNamespace(head=head),[row],'text','cpu')
        np.testing.assert_array_equal(pooled,np.ones((1,4),dtype=np.float32))
        h[-1]=1
        with self.assertRaisesRegex(ValueError,'nonzero padding'):
            replay_classifier(SimpleNamespace(head=head),[row],'text','cpu')
        h[-1]=0;mask[:]=False
        with self.assertRaisesRegex(ValueError,'padding mask'):
            replay_classifier(SimpleNamespace(head=head),[row],'text','cpu')
