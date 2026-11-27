"""Stdlib-only tests for MEGA refinement boundaries; no native imports."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.baseline_delivery import mega_refinement as mod,mega_diagnostic as diag
from experiments.baseline_delivery.common import digest,atomic_json


class Tests(unittest.TestCase):
    def test_auto_job_skips_finished_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);states={job:dict(state='pending') for job in diag.JOBS}
            states['mega-listops-s42']['state']='computed'
            states['mega-retrieval-s42']['state']='paused'
            for task in ('listops','image','retrieval'):
                atomic_json(root/'jobs'/('mega-'+task+'-s42')/'checkpoints/latest.json',{})
            atomic_json(root/'status.json',states)
            self.assertEqual(diag.select_job(root),'mega-image-s42')
            atomic_json(root/'jobs/mega-image-s42/summary.json',{})
            self.assertEqual(diag.select_job(root),'mega-retrieval-s42')
            self.assertEqual(diag.select_job(root,'mega-retrieval-s42'),'mega-retrieval-s42')
            with self.assertRaisesRegex(ValueError,'auto'):diag.select_job(root,'mega-listops-s42')
            states['mega-retrieval-s42']['state']='computed';atomic_json(root/'status.json',states)
            self.assertIsNone(diag.select_job(root))
            states['mega-text-s42']['state']='running';atomic_json(root/'status.json',states)
            with self.assertRaisesRegex(ValueError,'Pause'):diag.select_job(root)

    def test_no_eligible_job_skips_native_and_upload_and_releases_locks(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/'status.json',{j:dict(state='computed') for j in diag.JOBS})
            args=SimpleNamespace(source_suite=root,job='auto',cache=root,gpu='0')
            with ExitStack() as stack:
                stack.enter_context(patch.object(diag,'ROOT',root));stack.enter_context(patch.object(diag,'revision',return_value='pinned'))
                stack.enter_context(patch('experiments.baseline_delivery.bootstrap.check_exchange'))
                stack.enter_context(patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':{}})))
                stack.enter_context(patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}))
                run=stack.enter_context(patch('experiments.baseline_delivery.diagnostic_watch.Supervisor.run'))
                upload=stack.enter_context(patch('experiments.baseline_delivery.return_artifacts.upload_declared'))
                self.assertEqual(diag.campaign(args),0);run.assert_not_called();upload.assert_not_called()
            from experiments.baseline_delivery.common import lock
            with lock(root/'suite.lock'):pass

    def test_all_routes_use_real_supervisor_phase_contract(self):
        from experiments.baseline_delivery.diagnostic_watch import Recorder
        with tempfile.TemporaryDirectory() as tmp:
            recorder=Recorder(tmp,'fixture')
            for route in diag.METHODS+diag.STACKS+diag.REFINEMENTS:
                for stage in ('reference_a','reference_b','candidate','graph_capture','graph_replay','timing_0'):
                    name=diag.phase_name(route,stage)
                    recorder.mark(name,flush=True)
                    self.assertEqual(recorder.phase,name)
            with self.assertRaises(ValueError):diag.phase_name('fft','invalid-stage')

    def test_precision_targets_and_restore(self):
        class Value:
            def float(self):return 'fp32-output'
        class Linear:
            def forward(self,*a,**k):return Value()
        names=['encoder.layers.0.nffn.fc1','encoder.layers.1.nffn.fc2','encoder.layers.0.mega_layer.h_proj','head']
        modules={n:Linear() for n in names}
        model=SimpleNamespace(named_modules=lambda:list(modules.items()))
        self.assertEqual(mod.linear_names(model,'tail'),[names[1]])
        self.assertEqual(mod.linear_names(model,'ffn'),names[:2])
        self.assertEqual(mod.linear_names(model,'projections'),names[:3])
        calls=[]
        torch=SimpleNamespace(bfloat16='bf16',autocast=lambda *a,**k:(calls.append(k) or nullcontext()))
        plugin=mod.PrecisionIslands(model,'ffn',torch)
        self.assertIsInstance(modules[names[0]].forward(),Value)
        with plugin.activate():
            self.assertEqual(modules[names[0]].forward(),'fp32-output')
            self.assertIsInstance(modules['head'].forward(),Value)
        self.assertEqual(plugin.calls,1);self.assertFalse(calls[0]['cache_enabled'])
        plugin.close();self.assertTrue(all('forward' not in m.__dict__ for m in modules.values()))

    def test_padding_fix_has_no_scalar_read_and_preserves_return_shape(self):
        source='''class Encoder:
    traceable=False
    tpu=False
    def forward(self,padding_mask):
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask=None
        return [padding_mask], 'pooled'
'''
        class Mask:
            def any(self):raise RuntimeError('GPU scalar branch')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'encoder.py';path.write_text(source,encoding='utf8')
            spec=importlib.util.spec_from_file_location('fake_refine_encoder',path);m=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=m;self.addCleanup(sys.modules.pop,spec.name,None);spec.loader.exec_module(m)
            encoder=m.Encoder();other=m.Encoder();original=m.Encoder.forward;mask=Mask()
            with patch.object(mod,'ENCODER_SHA',digest(path)):plugin=mod.GraphPadding(encoder)
            with self.assertRaises(RuntimeError):encoder.forward(mask)
            with plugin.activate():
                out=encoder.forward(mask);self.assertIs(out[0][0],mask);self.assertEqual(out[1],'pooled')
                with self.assertRaises(RuntimeError):other.forward(mask)
            self.assertIs(m.Encoder.forward,original);self.assertFalse(encoder.traceable)
            plugin.close();self.assertNotIn('forward',encoder.__dict__)

    def test_refinement_routes_and_allowlisted_failure(self):
        for route in diag.REFINEMENTS:
            diag.settings(route);spec=diag.child_spec(Path('/fixture'),route)
            self.assertEqual(spec['argv'][-1],route)
        for route in ('bf16+island-ffn','island-tail+island-ffn'):
            with self.assertRaises(ValueError):diag.settings(route)
        self.assertEqual(mod.error_category(RuntimeError('operation not permitted when stream is capturing')),'capture_forbidden_operation')
        self.assertEqual(mod.error_category(RuntimeError('secret arbitrary content')),'unclassified_native_error')
        self.assertEqual(mod.error_category(RuntimeError('tensor appears to not have been used in graph')),'unused_autograd_input')

if __name__=='__main__':unittest.main()
