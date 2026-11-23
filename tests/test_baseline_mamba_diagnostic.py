"""Mamba diagnostic control checks; stdlib only, no native model execution."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import mamba_diagnostic as d
from experiments.baseline_delivery.common import atomic_json,digest


class Tests(unittest.TestCase):
    def test_paired_wall_times(self):
        value=d.summarize([10]*4,[8]*4)
        self.assertEqual(value['throughput_gain'],.25)
        self.assertTrue(value['timing_promising'])
        self.assertFalse(d.summarize([10]*4,[9,11,11,9])['timing_promising'])
        for invalid in ([0]*4,[float('nan')]*4,[1]):
            with self.assertRaises(ValueError):d.summarize([1]*4,invalid)

    def test_real_child_spec_help(self):
        spec=d.child_spec(Path('runs/example'),'bf16_graph')
        self.assertEqual(spec['argv'][3],spec['target'])
        self.assertIn('--output',spec['argv'])
        result=subprocess.run(spec['argv'][:4]+['--help'],cwd=spec['cwd'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--source-suite',result.stdout)

    def test_selected_snapshot_preserves_source_and_blocks_live_or_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'formal';suite=root/'diagnostic';folder=source/'jobs'/d.JOBS[0]
            cache=root/'cache';cache.mkdir();data=root/'data';data.mkdir()
            atomic_json(source/'suite.json',dict(data=str(data)))
            atomic_json(source/'status.json',{d.JOBS[0]:dict(state='paused')})
            atomic_json(folder/'config.json',dict(model='mamba1',task='retrieval',adaptive_memory={'initial':8}))
            checkpoint=folder/'checkpoints/step.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'fixture')
            atomic_json(folder/'checkpoints/latest.json',dict(schema='qkan-training-checkpoint.v1',
                file=checkpoint.name,sha256=digest(checkpoint),step=20085,contract='x'))
            before={p:digest(p) for p in source.rglob('*') if p.is_file()}
            with patch.object(d,'revision',return_value='revision'):
                request=d.snapshot(source,suite,d.JOBS[0],cache)
            self.assertEqual(request['checkpoint']['step'],20085)
            self.assertFalse(request['formal_release_allowed'])
            self.assertTrue(all(digest(p)==h for p,h in before.items()))
            self.assertEqual((suite/'source/checkpoints/step.pt').read_bytes(),b'fixture')
            for state in ('running','computed'):
                atomic_json(source/'status.json',{d.JOBS[0]:dict(state=state)})
                with self.assertRaises(ValueError):d.snapshot(source,root/state,d.JOBS[0],cache)

    def test_parent_uses_all_routes_and_returns_only_compact_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original';cache=root/'cache';request={'source_files':{}}
            specs=[];uploads=[]
            def snapshot(source,suite,job,cache):
                atomic_json(suite/'request.json',request);return request
            class Supervisor:
                records=[]
                def __init__(self,*args):pass
                def run(self,spec,log,env):
                    specs.append(spec);route=spec['argv'][spec['argv'].index('--route')+1]
                    atomic_json(log.parent/(route+'.json'),dict(timing={'fixture':True}));return 0
            with patch.object(d,'ROOT',root),patch.object(d,'snapshot',side_effect=snapshot),\
                 patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                 patch('experiments.baseline_delivery.runner.discover_gpus',return_value=(['0'],{'0':'GPU, uuid'})),\
                 patch('experiments.baseline_delivery.attention_diagnostic.gpu_preflight',return_value={'idle':True}),\
                 patch('experiments.baseline_delivery.diagnostic_watch.Supervisor',Supervisor),\
                 patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=lambda *a,**k:uploads.append(a)),\
                 patch.object(sys,'argv',['mamba','--source-suite',str(source),'--cache',str(cache)]),\
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(d.main(),0)
            self.assertEqual(len(specs),4);self.assertEqual(uploads[0][2],['mamba-speed.json'])
            self.assertTrue(all('--output' in s['argv'] for s in specs))


if __name__=='__main__':unittest.main()
