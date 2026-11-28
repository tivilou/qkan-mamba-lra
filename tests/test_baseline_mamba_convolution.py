"""Bounded offline asset/ABI dispatch tests; no native packages or model runs."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from experiments.baseline_delivery import mamba_convolution as c, mamba_validation as v
from experiments.baseline_delivery.common import atomic_json,read,digest
from tests.test_baseline_mamba_validation import Tests as ParentFixture


class Tests(unittest.TestCase):
    def test_parent_modes_binding_failure_and_cleanup(self):
        helper=ParentFixture()
        for kwargs in ({},{'fail':True},{'differ':True},{'mismatch':True}):
            helper.run_parent(kwargs.pop('fail',False),conv=True,**kwargs)
        for mode in v.CONV_MODES[2:]:
            with patch.object(v.speed,'native') as native,patch.object(sys,'argv',[
                    'validate','--request','suite/request.json','--mode',mode,'--output','suite/'+mode]):
                self.assertEqual(v.main(),0)
                self.assertEqual(native.call_args.args,(Path('suite/request.json'),'conv_fused_fp32'))

    def test_stage_reuses_fetch_and_rejects_changed_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';cache.mkdir();wheel=cache/'fixture.whl'
            with zipfile.ZipFile(wheel,'w') as z:
                z.writestr('causal_conv1d/__init__.py','# fixture')
                z.writestr('causal_conv1d_cuda.cpython-311-x86_64-linux-gnu.so','not executable fixture')
            manifest=root/'manifest.json';atomic_json(manifest,dict(objects=[dict(filename=wheel.name,sha256=digest(wheel))]))
            with patch.object(c,'ROOT',root),patch.object(c,'MANIFEST',manifest),\
                    patch('experiments.baseline_delivery.install_mirror.fetch',return_value=cache) as fetch:
                first=c.stage();self.assertEqual(first,c.stage());self.assertEqual(fetch.call_count,2)
                target=Path(first['root']);self.assertFalse(target.is_relative_to(cache))
                (target/'causal_conv1d/__init__.py').write_text('# changed')
                with self.assertRaisesRegex(ValueError,'package changed'):c.stage()

    def test_return_value_bridge_and_fused_baseline_switch_restore(self):
        calls=[]
        functions=SimpleNamespace(causal_conv1d_fwd_function=lambda *x:('fwd',x),
            causal_conv1d_bwd_function=lambda *x:('bwd',x))
        bridge=c.legacy_cuda_api(functions)
        self.assertEqual(bridge.causal_conv1d_fwd(*range(7)),('fwd',tuple(range(7))))
        self.assertEqual(bridge.causal_conv1d_bwd(*range(10)),('bwd',tuple(range(10))))
        class Mamba:
            use_fast_path=True
            def parameters(self):return [SimpleNamespace(dtype='float32')]
        function=object();inner=lambda *a,**k:calls.append((a,k))
        module=SimpleNamespace(__file__='mamba.py',Mamba=Mamba,causal_conv1d_fn=function,mamba_inner_fn=inner)
        old_cuda=object();scan=SimpleNamespace(__file__='scan.py',causal_conv1d_cuda=old_cuda)
        modules={'mamba_ssm.modules.mamba_simple':module,'mamba_ssm.ops.selective_scan_interface':scan,
            'causal_conv1d.cpp_functions':functions}
        hashes={'mamba.py':v.speed.InputProjectionBF16.source_sha256,
            'scan.py':'c79f4432536402fe4d6c1026ccf8d5fc0aaa62b790e9a622295aee14ac45dbcc'}
        with patch.object(c.importlib,'import_module',side_effect=lambda name:modules[name]),\
                patch.object(c,'digest',side_effect=lambda p:hashes[str(p)]):
            model=SimpleNamespace(blocks=[SimpleNamespace(mixer=Mamba())])
            policy=c.FusedFP32(SimpleNamespace(float32='float32'),model)
            self.assertIsNone(module.causal_conv1d_fn)
            with self.assertRaisesRegex(ValueError,'reference'):module.mamba_inner_fn('baseline')
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with policy.activate():
                    self.assertIs(module.causal_conv1d_fn,function);module.mamba_inner_fn('candidate')
                    self.assertEqual(scan.causal_conv1d_cuda.causal_conv1d_fwd(*range(7))[0],'fwd')
                    raise RuntimeError('fixture')
            self.assertIsNone(module.causal_conv1d_fn);self.assertEqual(policy.calls,1)
            policy.close();self.assertIs(module.causal_conv1d_fn,function)
            self.assertIs(module.mamba_inner_fn,inner);self.assertIs(scan.causal_conv1d_cuda,old_cuda)
            self.assertEqual(len(calls),1)

    def test_abi_mismatch_stops_before_import_or_package_access(self):
        torch=SimpleNamespace(__version__='2.7.1+cu126',_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False))
        with self.assertRaisesRegex(ValueError,'ABI true'):c.prepare_native(torch,{})


if __name__=='__main__':unittest.main()
