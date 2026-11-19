"""Execute the worker's whole zero-update migration branch with stdlib state fakes."""
import ast
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery import runner
from experiments.baseline_delivery.common import ROOT,atomic_json,read


def invoke(output,*,candidate=True,dynamic=True,corrupt=False,lineage=None):
    """Retain the real branch, including roundtrip comparisons and receipt call.

    Only injected model/optimizer/checkpoint runtimes are faked. No native import,
    checkpoint deserialization, training update or model computation occurs.
    """
    tree=ast.parse((ROOT/'experiments/baseline_delivery/worker.py').read_text(encoding='utf8'))
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run')
    branch=next(n for n in run.body if isinstance(n,ast.If)
                and ast.unparse(n.test)=='migration_check_only or (candidate and strategy_admission.check_only)')
    function=ast.FunctionDef(name='migration_boundary',args=ast.arguments(posonlyargs=[],args=[],
        kwonlyargs=[],kw_defaults=[],defaults=[]),body=[copy.deepcopy(branch)],decorator_list=[])
    state=dict(contract='target',step=14268,epoch=2,cursor=32384,epoch_loss=17.5,epoch_correct=15,
        epoch_samples=32384,permutation=[3,7,1,5],model={'weight':[1.,2.]},
        optimizer={'moments':[.1,.2],'steps':14268},rng={'python':[4,5],'cuda':[6,7]},
        scheduler={'progress':1799234,'schema':'qkan-sample-clock.v1'},
        sample_clock={'progress':1799234,'schema':'qkan-sample-clock.v1'},
        strategy_gradients={'weight':None},training_strategy={'gradient_policy':'analytic'},
        migration=copy.deepcopy(lineage if lineage is not None else dict(
            source_contract='source',target_contract='target',source_checkpoint_sha256='original',
            optimizer_updates_performed=0,previous_migration={'kept':True})))
    original=copy.deepcopy(state);stored=[];comparisons=[]
    def compare(left,right):
        if left!=right: raise ValueError('State roundtrip differs')
        comparisons.append(copy.deepcopy(left))
    def save(folder,payload): stored.append(copy.deepcopy(payload))
    def load(folder,contract):
        result=copy.deepcopy(stored[-1])
        if corrupt: result['model']['weight'][0]=99
        return result
    no_update=Mock(side_effect=AssertionError('Zero-update migration performed an optimizer operation'))
    clock_snapshot=copy.deepcopy(state['sample_clock']);schedule=NS(state_dict=lambda:copy.deepcopy(clock_snapshot),step=no_update)
    scope=dict(__package__='experiments.baseline_delivery',migration_check_only=not candidate,candidate=candidate,
        strategy_admission=NS(check_only=True),adaptive=NS(snapshot=lambda:{'retained_controller':True}),dynamic=dynamic,
        model=NS(state_dict=lambda:copy.deepcopy(original['model'])),
        opt=NS(state_dict=lambda:copy.deepcopy(original['optimizer']),step=no_update),schedule=schedule,
        rng_state=lambda:copy.deepcopy(original['rng']),host=copy.deepcopy,state=state,
        clock=NS(snapshot=lambda:copy.deepcopy(clock_snapshot),state={'total':11779000},rates=lambda:[.001]),
        position=1799234,save_checkpoint=save,load_checkpoint=load,atomic_json=atomic_json,output=Path(output),contract='target')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
        'production-worker-migration-branch','exec'),scope)
    with patch.object(runner,'compare_states',side_effect=compare): result=scope['migration_boundary']()
    no_update.assert_not_called()
    return NS(result=result,state=state,original=original,stored=stored,comparisons=comparisons)


class MigrationReceiptTests(unittest.TestCase):
    def test_candidate_actual_branch_roundtrip_and_overlapping_contract_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            actual=invoke(temp);receipt=read(Path(temp)/'migration-validation.json')
            self.assertEqual(actual.result,0);self.assertTrue(receipt['passed'])
            self.assertEqual(receipt['target_contract'],'target');self.assertEqual(receipt['source_contract'],'source')
            self.assertEqual(receipt['optimizer_steps_performed'],0)
            self.assertEqual(receipt['source_checkpoint_sha256'],'original')
            self.assertEqual(receipt['previous_migration'],{'kept':True})
            self.assertEqual(len(actual.stored),1);self.assertEqual(len(actual.comparisons),4)
            for key in ('model','optimizer','rng','permutation','step','epoch','cursor','epoch_loss',
                        'epoch_correct','epoch_samples','strategy_gradients','training_strategy','sample_clock','scheduler'):
                self.assertEqual(actual.state[key],actual.original[key],key)

    def test_legacy_branch_receipt_and_zero_update_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            actual=invoke(temp,candidate=False,dynamic=False,lineage=dict(source_contract='old'))
            self.assertEqual(actual.result,0);self.assertEqual(len(actual.comparisons),5)
            self.assertEqual(read(Path(temp)/'migration-validation.json'),dict(passed=True,source_contract='old',
                state_roundtrip='exact before first adaptive optimizer update',target_contract='target',optimizer_steps_performed=0))

    def test_corrupt_roundtrip_never_writes_pass_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'roundtrip'): invoke(temp,corrupt=True)
            self.assertFalse((Path(temp)/'migration-validation.json').exists())

    def test_verified_fields_take_precedence_over_lineage_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            invoke(temp,lineage=dict(source_contract='source',target_contract='target',passed=False,
                                    optimizer_steps_performed=9,state_roundtrip='old',unrelated='preserved'))
            receipt=read(Path(temp)/'migration-validation.json')
            self.assertTrue(receipt['passed']);self.assertEqual(receipt['optimizer_steps_performed'],0)
            self.assertEqual(receipt['state_roundtrip'],'exact before first adaptive optimizer update')
            self.assertEqual(receipt['unrelated'],'preserved')


if __name__=='__main__': unittest.main()
