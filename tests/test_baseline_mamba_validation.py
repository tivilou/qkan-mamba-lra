"""Finite validation controller checks; no native frameworks or model runs."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import mamba_validation as v
from experiments.baseline_delivery.common import atomic_json,read


def result(mode):
    n=4 if mode.endswith('_interrupt') else 8
    indices=range(4,8) if mode.endswith('_resume') else range(n)
    return dict(schema='qkan-mamba-validation-child.v1',mode=mode,request_sha256='hash',
        completed_updates=n,source_restored_exact=True,formal_updates_committed=0,
        formal_release_allowed=False,finite=True,state_sha256='mid' if n==4 else 'end',
        components={'model':'x'},starting_state_sha256='mid' if mode.endswith('_resume') else 'start',
        updates=[dict(index=i,loss=.1) for i in indices])


class Tests(unittest.TestCase):
    def test_actual_child_help(self):
        s=v.spec(Path('runs/fixture'),'bf16_graph_resume')
        self.assertEqual(s['argv'][3],s['target']);self.assertIn('--output',s['argv'])
        r=subprocess.run(s['argv'][:4]+['--help'],cwd=s['cwd'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--source-suite',r.stdout)
        with self.assertRaises(ValueError):v.spec(Path('runs/fixture'),'formal')

    def test_receipt_rejects_wrong_identity_commit_or_incomplete(self):
        x=result('reference');v.accept(x,'hash','reference')
        for field,value in [('request_sha256','wrong'),('formal_updates_committed',1),
                ('completed_updates',7),('finite',False),('source_restored_exact',False)]:
            with self.assertRaises(ValueError):v.accept(dict(x,**{field:value}),'hash','reference')

    def test_recovery_checks_cursor_hash_components_and_update_sequence(self):
        c=result('bf16_eager');i=result('bf16_eager_interrupt');r=result('bf16_eager_resume')
        self.assertTrue(v.recovery_agreement(c,i,r)['exact'])
        for key,value in [('starting_state_sha256','wrong'),('components',{'model':'wrong'}),
                          ('updates',[]),('state_sha256','wrong')]:
            self.assertFalse(v.recovery_agreement(c,i,dict(r,**{key:value}))['exact'])

    def test_cleanup_leaves_original_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original=root/'original.pt';original.write_bytes(b'keep')
            suite=root/'suite';suite.mkdir();(suite/'reference-values.pt').write_bytes(b'discard')
            (suite/'unrelated.pt').write_bytes(b'keep')
            record=v.cleanup(suite,{})
            self.assertFalse(record['remaining_owned_binaries']);self.assertTrue(original.exists())
            self.assertTrue((suite/'unrelated.pt').exists())

    def test_parent_orders_seven_children_returns_only_report_and_cleans(self):
        self.run_parent(False)

    def test_parent_failure_stops_and_still_cleans_and_returns(self):
        self.run_parent(True)

    def run_parent(self,fail):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';cache=root/'cache';calls=[];exports=[]
            def snapshot(source,suite,job,cache):
                request={'source_files':{}}
                atomic_json(suite/'request.json',request)
                (suite/'reference-values.pt').write_bytes(b'private')
                return request
            class Supervisor:
                records=[]
                def __init__(self,*args):pass
                def run(self,spec,log,env):
                    self_outer.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
                    mode=spec['argv'][spec['argv'].index('--mode')+1];calls.append(mode)
                    if fail:return 1
                    receipt=result(mode);receipt['request_sha256']=v.digest(log.parent/'request.json')
                    atomic_json(log.parent/mode/'result.json',receipt);return 0
            self_outer=self
            review=root/'review.json';atomic_json(review,{'routes':{'bf16_eager':{'timing':{'timing_promising':True}}}})
            with patch.object(v,'ROOT',root),patch.object(v,'REVIEW',review),\
                 patch.object(v.speed,'snapshot',side_effect=snapshot),\
                 patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                 patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'gpu'})),\
                 patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}),\
                 patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',Supervisor),\
                 patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:exports.append(a)),\
                 patch.object(sys,'argv',['validate','--source-suite',str(source),'--cache',str(cache)]),\
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(v.main(),76 if fail else 0)
            self.assertEqual(calls,['reference'] if fail else list(v.MODES))
            self.assertEqual(exports[0][2],['mamba-validation.json'])
            report=read(exports[0][0]/'mamba-validation.json')
            self.assertFalse((exports[0][0]/'reference-values.pt').exists())
            self.assertFalse(report['formal_release_allowed']);self.assertTrue(report['source_unchanged'])


if __name__=='__main__':unittest.main()
