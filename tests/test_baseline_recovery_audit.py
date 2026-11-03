"""Recovery diagnostics quantify noise without accepting approximate recovery."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.baseline_delivery import recovery_audit as audit


def state():
    result={key:None for key in audit.FIELDS}
    result.update(contract='same-source-and-data',runtime={'torch':'pinned'},
                  model={'weight':torch.tensor([1.,2.])},
                  optimizer={'moment':torch.tensor([.5])},scheduler={'step':6,'lr':.001},
                  rng={'torch':torch.tensor([1,2],dtype=torch.uint8),'cuda':[]},
                  permutation=np.array([2,0,1]),cursor=0,step=6,epoch=2,
                  best_weights={'weight':torch.tensor([1.,2.])},best_accuracy=.5,best_epoch=0,
                  history=[{'loss':1.}],update_trace=[{'loss':1.,'step':6}],
                  epoch_loss=0.,epoch_correct=0,epoch_samples=0,training_complete=True,
                  amp={'enabled':False,'scaler':None})
    return result


class AuditTests(unittest.TestCase):
    def test_identical_full_state(self):
        a=state(); report=audit.compare(a,copy.deepcopy(a))
        self.assertTrue(report['exact']);self.assertTrue(report['control_state_exact'])
        paths={r['path'] for r in report['leaves']}
        for name in ('model/weight','optimizer/moment','scheduler/lr','rng/torch',
                     'history/0/loss','update_trace/0/loss','best_weights/weight','step'):
            self.assertIn(name,paths)

    def test_tiny_float_noise_is_measured_never_passed(self):
        a=state();b=copy.deepcopy(a)
        b['model']['weight'][0]=torch.nextafter(b['model']['weight'][0],torch.tensor(2.))
        b['optimizer']['moment'][0]+=.0001
        report=audit.compare(a,b)
        self.assertFalse(report['exact']);self.assertTrue(report['control_state_exact'])
        self.assertEqual(report['differing_leaves'],2)
        leaf=next(r for r in report['leaves'] if r['path']=='model/weight')
        self.assertEqual(leaf['differing_elements'],1);self.assertGreater(leaf['max_abs'],0)
        self.assertNotIn('passed',report)

    def test_control_state_mismatch_and_nonfinite(self):
        a=state();b=copy.deepcopy(a);b['rng']['torch'][0]=3
        report=audit.compare(a,b)
        self.assertFalse(report['control_state_exact'])
        b=copy.deepcopy(a);b['model']['weight'][0]=float('nan')
        report=audit.compare(a,b)
        self.assertFalse(report['all_finite']);self.assertFalse(report['exact'])

    def test_structural_mismatch(self):
        for a,b in ((torch.zeros(1),torch.zeros(2)),({'x':1},{'y':1}),([1],[1,2]),(1,1.)):
            self.assertFalse(audit.differences(a,b)[0]['exact'])

    def test_scientific_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'experiments/baseline_delivery';directory.mkdir(parents=True)
            (directory/'worker.py').write_bytes(b'changed')
            with patch.object(audit,'ROOT',root),patch.object(audit.subprocess,'check_output',return_value=b'pinned'):
                with self.assertRaisesRegex(ValueError,'Original scientific source changed'):
                    audit.check_source({'source_revision':'old'})
