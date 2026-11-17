"""Real supervised bootstrap against the actual native-entry preparation statements.

No framework import or native job: stop immediately after output preparation.
"""
import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest

from experiments.baseline_delivery import diagnostic_watch as watch
from experiments.baseline_delivery import key_bias_worker_admission as admission
from experiments.baseline_delivery.common import ROOT,atomic_json,read


def entry_output_preparation(output):
    """Execute the production statements between input validation and hashing.

    This also reproduces the previous bug against a baseline checkout: the
    supervisor has created diagnostics before native() sees the output path.
    """
    tree=ast.parse(Path(admission.__file__).read_text(encoding='utf8'))
    native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native')
    start=next(i for i,n in enumerate(native.body) if isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='admission.validate')+1
    stop=next(i for i,n in enumerate(native.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='hasher' for t in n.targets))
    scope=dict(output=Path(output),prepare_worker_output=admission.prepare_worker_output)
    exec(compile(ast.Module(body=native.body[start:stop],type_ignores=[]),'actual-native-output-preparation','exec'),scope)


class BootstrapTests(unittest.TestCase):
    def launch(self,root,stale=False):
        output=root/'migration'
        if stale: atomic_json(output/'checkpoints/latest.json',{'preserve':'stale-checkpoint'})
        supervisor=watch.Supervisor(15,10,announce=lambda *a,**k:None)
        module='tests.test_baseline_key_bias_worker_bootstrap'
        code=supervisor.run(dict(target=module,cwd=str(ROOT),argv=[sys.executable,'-B','-m',module,
            '--fixture','--output',str(output)]),root/'worker-private.log',dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        return output,supervisor.records,code

    def test_actual_supervisor_diagnostics_enters_migration_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            output,records,code=self.launch(Path(temp))
            self.assertEqual(code,0)
            proof=read(output/'bootstrap-boundary.json')
            self.assertTrue(proof['supervisor_diagnostics_present']);self.assertEqual(proof['framework_imports'],0)
            self.assertEqual(proof['reached'],'checkpoint-preparation-boundary')
            self.assertEqual(len(records),1);self.assertTrue(records[0]['termination']['parent_reaped'])
            self.assertEqual(len(list((output/'diagnostics').iterdir())),1)

    def test_actual_supervisor_stale_checkpoint_rejected_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            output,records,code=self.launch(Path(temp),stale=True)
            self.assertEqual(code,1);self.assertFalse((output/'bootstrap-boundary.json').exists())
            self.assertEqual(read(output/'checkpoints/latest.json'),{'preserve':'stale-checkpoint'})
            self.assertTrue(records[0]['termination']['parent_reaped'])

    def test_existing_shared_guard_rejects_unknown_entries_and_invalid_diagnostics(self):
        for name in ('config.json','result.json','worker-error.json','trace-values','checkpoints'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                output=Path(temp)/'migration';output.mkdir()
                (output/'diagnostics').mkdir();entry=output/name
                entry.write_text('preserve',encoding='utf8')
                with self.assertRaises(ValueError): entry_output_preparation(output)
                self.assertEqual(entry.read_text(encoding='utf8'),'preserve')
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'fresh'
            entry_output_preparation(output);self.assertTrue(output.is_dir())
            (output/'diagnostics').write_text('invalid directory',encoding='utf8')
            with self.assertRaises(ValueError): entry_output_preparation(output)


def fixture():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    entry_output_preparation(args.output)
    assert (args.output/'diagnostics').is_dir()
    assert not any(n.split('.')[0] in ('torch','numpy','jax','tensorflow') for n in sys.modules)
    atomic_json(args.output/'bootstrap-boundary.json',dict(reached='checkpoint-preparation-boundary',
        supervisor_diagnostics_present=True,framework_imports=0))


if __name__=='__main__':
    if '--fixture' in sys.argv: fixture()
    else: unittest.main()
