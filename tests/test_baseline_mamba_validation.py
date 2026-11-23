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
    n=2 if mode in v.FP32_MODES else 4 if mode.endswith('_interrupt') else 8
    indices=range(4,8) if mode.endswith('_resume') else range(n)
    return dict(schema='qkan-mamba-validation-child.v1',mode=mode,request_sha256='hash',
        completed_updates=n,source_restored_exact=True,formal_updates_committed=0,
        formal_release_allowed=False,finite=True,state_sha256='mid' if n==4 else 'end',
        components={'model':'x'},starting_state_sha256='mid' if mode.endswith('_resume') else 'start',
        updates=[dict(index=i,loss=.1,row_ids_sha256='rows'+str(i),components={'model':'x'}) for i in indices])


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

    def test_fp32_repeat_modes_and_native_route(self):
        for mode in v.FP32_MODES:
            self.assertIn(mode,v.spec(Path('suite'),mode)['argv'])
            v.accept(result(mode),'hash',mode)
            with self.assertRaises(ValueError):v.accept(dict(result(mode),completed_updates=8),'hash',mode)
            with patch.object(v.speed,'native') as native,patch.object(sys,'argv',[
                    'validate','--request','suite/request.json','--mode',mode,'--output','suite/'+mode]):
                self.assertEqual(v.main(),0)
                self.assertEqual(native.call_args.args,(Path('suite/request.json'),'fp32_eager'))

    def test_fp32_repeat_detects_state_difference_with_equal_loss(self):
        import copy
        a=result(v.FP32_MODES[0]);b=result(v.FP32_MODES[1])
        self.assertTrue(v.repeat_agreement(a,b)['exact'])
        b=copy.deepcopy(b);b['updates'][0]['components']['model']='different'
        difference=v.repeat_agreement(a,b)
        self.assertFalse(difference['exact']);self.assertEqual(difference['first_different_update_1based'],1)
        self.assertEqual(difference['steps'][0]['differing_components'],['model'])
        self.assertTrue(difference['steps'][0]['loss_equal'])
        b=result(v.FP32_MODES[1]);b['starting_state_sha256']='wrong'
        self.assertFalse(v.repeat_agreement(a,b)['exact'])
        b=result(v.FP32_MODES[1]);b['updates'][0]['step']=42
        self.assertFalse(v.repeat_agreement(a,b)['exact'])

    def test_fp32_parent_only_two_children_and_checkpoint_binding(self):
        self.run_parent(False,repeat=True)
        self.run_parent(False,repeat=True,differ=True)
        self.run_parent(True,repeat=True)
        self.run_parent(False,repeat=True,mismatch=True)

    def run_parent(self,fail,repeat=False,differ=False,mismatch=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';cache=root/'cache';calls=[];exports=[]
            def snapshot(source,suite,job,cache):
                request={'source_files':{},'checkpoint':{'sha256':'checkpoint','file':'fixture.pt'}}
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
                    if differ and mode==v.FP32_MODES[1]:receipt['updates'][0]['components']={'model':'different'}
                    atomic_json(log.parent/mode/'result.json',receipt);return 0
            self_outer=self
            review=root/'review.json';atomic_json(review,{'routes':{'bf16_eager':{'timing':{'timing_promising':True}}}})
            repeat_review=root/'repeat-review.json';atomic_json(repeat_review,{'evidence':{
                'source_checkpoint_sha256':'wrong' if mismatch else 'checkpoint'}})
            with patch.object(v,'ROOT',root),patch.object(v,'REVIEW',review),\
                 patch.object(v,'REPEAT_REVIEW',repeat_review),\
                 patch.object(v.speed,'snapshot',side_effect=snapshot),\
                 patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                 patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'gpu'})),\
                 patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}),\
                 patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',Supervisor),\
                 patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a:exports.append(a)),\
                 patch.object(sys,'argv',['validate','--source-suite',str(source),'--cache',str(cache)]+(['--fp32-repeat'] if repeat else [])),\
                 contextlib.redirect_stdout(io.StringIO()):
                if mismatch:
                    with self.assertRaisesRegex(ValueError,'checkpoint differs'):v.main()
                else:self.assertEqual(v.main(),76 if fail or differ else 0)
            modes=v.FP32_MODES if repeat else v.MODES
            self.assertEqual(calls,[] if mismatch else [modes[0]] if fail else list(modes))
            filename='mamba-repeat.json' if repeat else 'mamba-validation.json'
            self.assertEqual(exports[0][2],[filename])
            report=read(exports[0][0]/filename)
            self.assertFalse((exports[0][0]/'reference-values.pt').exists())
            self.assertFalse(report['formal_release_allowed'])
            if not mismatch:self.assertTrue(report['source_unchanged'])
            if repeat:self.assertEqual(report['prefix_steps'],2)


if __name__=='__main__':unittest.main()
