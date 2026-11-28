"""CPU plugin/control-plane tests with stdlib arrays; no native framework imports."""
import argparse
import ast
import copy
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery import optimization_plugins as plugins,padding_profile as profile
from experiments.baseline_delivery.common import ROOT,atomic_json,read


class Array:
    """Small explicit 2-D indexing fake for the real NumPy adapter's logic."""
    ndim=2
    dtype=NS(kind='i')
    def __init__(self,rows): self.rows=copy.deepcopy(rows)
    @property
    def shape(self): return (len(self.rows),len(self.rows[0]) if self.rows else 0)
    def __getitem__(self,key):
        if isinstance(key,slice): return Array(self.rows[key])
        a,b=key
        return Array([row[b] for row in self.rows[a]])
    def __ne__(self,value): return Array([[n!=value for n in row] for row in self.rows])
    def __eq__(self,value): return Array([[n==value for n in row] for row in self.rows])
    def any(self,axis): return Vector([any(row) for row in self.rows])
    def all(self): return all(all(row) for row in self.rows)


class Vector(list):
    def all(self): return all(self)
    def __rsub__(self,value): return Vector([value-n for n in self])


NP=NS(__version__='stdlib-fake',argmax=lambda a,axis:Vector([row.index(max(row)) for row in a.rows]),
      array_equal=lambda a,b:a.rows==b.rows)


def config(**options): return dict(schema=plugins.SCHEMA,optimizations=options)
def selection(alignment=4): return plugins.resolve(config(padding_trim=dict(enabled=True,length_alignment=alignment)))
def pair(rows): return dict(inputs1=Array(rows),inputs2=Array(rows))


class PluginTests(unittest.TestCase):
    def test_presets_and_zero_touch_null(self):
        for value in (None,config(),read(ROOT/'experiments/baseline_delivery/optimization-presets/baseline.json')):
            selected=plugins.resolve(value);ops=Mock();tokens=object()
            result,observations=plugins.transform_batch(tokens,selected,ops)
            self.assertIs(result,tokens);self.assertEqual(observations,{});self.assertEqual(ops.mock_calls,[])
            self.assertTrue(selected.manifest()['baseline_adapter'])
        self.assertTrue(plugins.resolve(read(ROOT/'experiments/baseline_delivery/optimization-presets/padding-trim.json')).padding_trim)

    def test_invalid_fields_fail_early(self):
        invalid=[{},[],dict(schema=plugins.SCHEMA,optimizations={},typo=True),config(typo={'enabled':True}),
                 config(padding_trim={'enabled':1}),config(padding_trim={'enabled':False,'typo':4})]
        invalid += [config(padding_trim={'enabled':True,'length_alignment':n}) for n in (True,0,-1,4001,1.5)]
        for value in invalid:
            with self.subTest(value=value),self.assertRaises(ValueError): plugins.resolve(value)

    def test_dependency_conflicts_and_unimplemented_flags(self):
        with self.assertRaisesRegex(ValueError,'dependency'): plugins.resolve(config(flash_attention={'enabled':True}))
        with self.assertRaisesRegex(ValueError,'conflict'):
            plugins.resolve(config(bf16_amp={'enabled':True},fp32_attention_tuned={'enabled':True}))
        for value in (config(bf16_amp={'enabled':True}),config(fp32_attention_tuned={'enabled':True}),
                      config(bf16_amp={'enabled':True},flash_attention={'enabled':True})):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'pending'): plugins.resolve(value)

    def test_preserves_interior_pad_and_original_token_coordinates(self):
        tokens=dict(inputs1=Array([[2,1,3,1,1,1,1,1],[7,8,1,1,1,1,1,1]]),
                    inputs2=Array([[2,1,1,1,6,1,1,1],[3,1,1,1,1,1,1,1]]))
        original=copy.deepcopy(tokens);result,obs=plugins.transform_batch(tokens,selection(),profile.NumpyTokenOps(NP))
        self.assertEqual(result['inputs1'].rows,[[2,1,3,1],[7,8,1,1]])
        self.assertEqual(result['inputs2'].shape,(2,8))
        self.assertEqual(obs['inputs1']['last_valid_positions'],[3,2])
        self.assertEqual(obs['inputs1']['attention_cells_after'],32)
        self.assertEqual({k:a.rows for k,a in tokens.items()},{k:a.rows for k,a in original.items()})

    def test_alignment_caps_at_original_width(self):
        tokens={'inputs':Array([[2,1,1,1,3]])}
        out,obs=plugins.transform_batch(tokens,selection(4),profile.NumpyTokenOps(NP))
        self.assertEqual(out['inputs'].shape,(1,5));self.assertEqual(obs['inputs']['execution_width'],5)

    def test_bad_dimensions_empty_inputs_labels_and_row_mismatch(self):
        ops=profile.NumpyTokenOps(NP)
        for tokens in ({'inputs':Array([])},{'inputs':Array([[1,1]])},{'labels':Array([[0]])},
                       dict(inputs1=Array([[2]]),inputs2=Array([[3],[4]]))):
            with self.subTest(tokens=list(tokens)),self.assertRaises(ValueError):
                plugins.transform_batch(tokens,selection(),ops)
        for shape in ((0,8,[]),(1,0,[1]),(1,8,[9])):
            with self.subTest(shape=shape),self.assertRaises(ValueError):
                plugins.transform_batch({'inputs':object()},selection(),NS(inspect=lambda x:shape))

    def test_adapter_rejects_dtype_and_changed_suffix_or_prefix(self):
        ops=profile.NumpyTokenOps(NP);bad=Array([[2]]);bad.dtype=NS(kind='f')
        with self.assertRaises(ValueError): ops.inspect(bad)
        with self.assertRaises(ValueError): ops.verify(Array([[2,3]]),Array([[2]]),1)
        with self.assertRaises(ValueError): ops.verify(Array([[2,1]]),Array([[9]]),1)

    def test_bounded_summary_covers_tail_and_reports_work_not_speed(self):
        rows=[[2,1,1,1,1,1,1,1],[2,3,1,1,1,1,1,1],[2,1,3,1,1,1,1,1]];calls=[]
        def load(start,count): calls.append((start,count));return pair(rows[start:start+count])
        result=profile.summarize_batches(load,3,2,selection(),profile.NumpyTokenOps(NP))
        self.assertEqual(calls,[(0,2),(2,1)]);self.assertEqual(result['batches'],2)
        self.assertEqual(result['attention_cells_before'],384);self.assertEqual(result['attention_cells_after'],96)
        self.assertEqual(result['attention_cells_reduction_fraction'],.75)
        self.assertEqual(result['documents']['inputs1']['span_quantiles'],dict(p50=2,p90=3,p99=3))
        self.assertNotIn('last_valid_positions',json.dumps(result));self.assertNotIn('speedup',result)

    def test_full_length_has_no_opportunity(self):
        result=profile.summarize_batches(lambda a,b:pair([[2,1,3]]),1,1,selection(),profile.NumpyTokenOps(NP))
        self.assertEqual(result['attention_cells_reduction_fraction'],0)

    def test_disabled_summary_and_wrong_coverage(self):
        result=profile.summarize_batches(lambda a,b:pair([[2,1,1,1]]),1,1,plugins.resolve(),profile.NumpyTokenOps(NP))
        self.assertEqual(result['attention_cells_reduction_fraction'],0)
        for rows,micro in ((0,16),(1,0),(1,129),(True,16)):
            with self.assertRaises(ValueError): profile.summarize_batches(Mock(),rows,micro,selection(),Mock())
        with self.assertRaisesRegex(ValueError,'coverage'):
            profile.summarize_batches(lambda a,b:pair([[2],[3]]),1,1,selection(),profile.NumpyTokenOps(NP))


class EntryTests(unittest.TestCase):
    def test_bash_dispatch_preserves_exact_argv_and_environment(self):
        bash=shutil.which('bash') if os.name!='nt' else 'D:/Program Files/Git/bin/bash.exe'
        if not bash or not Path(bash).is_file(): self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);script=folder/'scripts/profile_optional_padding.sh';script.parent.mkdir()
            script.write_bytes((ROOT/'scripts/profile_optional_padding.sh').read_bytes())
            python=folder/'.venv-baselines/bin/python';python.parent.mkdir(parents=True)
            python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$PWD" "$PYTHONDONTWRITEBYTECODE" "$PYTHONUNBUFFERED" "$@"\n',encoding='utf8',newline='\n')
            python.chmod(0o755)
            argv=['--source-suite','source with spaces','--cache','cache','--optimizations','padding.json','--micro-batch','32','--max-rows','4096']
            result=subprocess.run([bash,str(script),*argv],cwd=folder,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            lines=result.stdout.splitlines();self.assertTrue(lines[0].endswith(folder.name))
            self.assertEqual(lines[1:],['1','1','-B','-m','experiments.baseline_delivery.padding_profile',*argv])

    def test_actual_help_and_invalid_cli_do_not_import_frameworks(self):
        guard="""import importlib.abc,runpy,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','numpy','jax','tensorflow'): raise AssertionError('framework import '+name)
sys.meta_path.insert(0,Guard())
sys.argv=['padding_profile',*sys.argv[1:]]
runpy.run_module('experiments.baseline_delivery.padding_profile',run_name='__main__')
"""
        for args,code,text in ((['--help'],0,'--max-rows'),(['--unknown'],2,'required')):
            proc=subprocess.run([sys.executable,'-B','-c',guard,*args],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(proc.returncode,code,proc.stderr);self.assertIn(text,proc.stdout+proc.stderr)

    def source_fixture(self,folder):
        source=folder/'source';data=folder/'data';data.mkdir()
        atomic_json(source/'suite.json',dict(data=str(data)))
        atomic_json(source/'jobs'/profile.JOB/'config.json',dict(source_revision=profile.SOURCE_REVISION,
            model='transformer',task='retrieval',data_hashes={'train':'frozen-train-sha'}))
        atomic_json(folder/'cache/arrays/frozen-train-sha/complete.json',{})
        return NS(source_suite=source,cache=folder/'cache',max_rows=3,micro_batch=2,
                  optimizations=ROOT/'experiments/baseline_delivery/optimization-presets/padding-trim.json')

    def test_collect_train_only_exact_hash_binding_and_no_label_values(self):
        class Labels:
            def __getitem__(self,key): raise AssertionError('Label values accessed')
        arrays=pair([[2,1,1,1],[3,1,1,1],[2,3,1,1]])
        split=NS(arrays=dict(arrays,labels=Labels()),identity='frozen-train-sha')
        class Split:
            def __init__(self,*args): calls.append(args)
            arrays=split.arrays;identity=split.identity
            def __len__(self): return 3
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            args=self.source_fixture(Path(tmp));before={p:p.read_bytes() for p in args.source_suite.rglob('*.json')}
            with patch.dict(sys.modules,{'numpy':NP,'experiments.baseline_delivery.data':NS(Split=Split)}),patch.object(profile,'revision',return_value='candidate'):
                report={};profile.collect(args,report)
            self.assertEqual(calls,[(Path(tmp)/'data','retrieval','train',args.cache,'frozen-train-sha')])
            self.assertFalse(report['dataset']['label_values_used']);self.assertFalse(report['dataset']['test_access'])
            self.assertIsNone(report['actual_gpu_speedup']);self.assertEqual(report['measurement']['rows'],3)
            self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_wrong_source_rejected_before_numpy_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.source_fixture(Path(tmp));path=args.source_suite/'jobs'/profile.JOB/'config.json'
            value=read(path);value['source_revision']='wrong';atomic_json(path,value)
            with self.assertRaisesRegex(ValueError,'b07'): profile.collect(args,{})

    def test_missing_cache_does_not_extract_duplicate_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.source_fixture(Path(tmp));args.cache=Path(tmp)/'missing-cache'
            with self.assertRaisesRegex(ValueError,'existing training cache'): profile.collect(args,{})
            self.assertFalse(args.cache.exists())

    def test_main_success_and_failure_both_return_only_declared_compact_report(self):
        for error in (None,ValueError('fixture')):
            with self.subTest(error=error),tempfile.TemporaryDirectory() as tmp:
                args=['padding_profile','--source-suite','fixture','--cache','cache','--optimizations',
                      str(ROOT/'experiments/baseline_delivery/optimization-presets/padding-trim.json')]
                def collect(args,report):
                    if error: raise error
                    report.update(passed=True,status='opportunity_measured')
                upload=Mock()
                with patch.object(sys,'argv',args),patch.object(profile,'ROOT',Path(tmp)),patch.object(profile,'revision',return_value='fixture'),\
                     patch.object(profile,'collect',side_effect=collect),patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                     patch('experiments.baseline_delivery.return_artifacts.upload_declared',upload):
                    if error:
                        with self.assertRaises(ValueError): profile.main()
                    else: self.assertEqual(profile.main(),0)
                folder,kind,files=upload.call_args.args
                self.assertEqual(kind,'padding-profile');self.assertEqual(files,['padding-profile.json'])
                report=read(folder/files[0]);self.assertFalse(report['model_executed']);self.assertFalse(report['gpu_accessed'])
                self.assertEqual(report['passed'],error is None);self.assertEqual(report['training_samples_committed'],0)
                self.assertNotIn('fixture',report.get('error_message',''))

    def test_no_top_level_native_imports(self):
        for name in ('optimization_plugins.py','padding_profile.py'):
            tree=ast.parse((ROOT/'experiments/baseline_delivery'/name).read_text(encoding='utf8'))
            for node in tree.body:
                if isinstance(node,ast.Import):
                    self.assertFalse({x.name.split('.')[0] for x in node.names}&{'numpy','torch','jax','tensorflow'})

    def test_failed_upload_preserves_local_report_without_rerunning_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=['padding_profile','--source-suite','fixture','--cache','cache','--optimizations',
                  str(ROOT/'experiments/baseline_delivery/optimization-presets/padding-trim.json')]
            collect=Mock(side_effect=lambda args,report:report.update(passed=True))
            with patch.object(sys,'argv',args),patch.object(profile,'ROOT',Path(tmp)),patch.object(profile,'revision',return_value='fixture'),\
                 patch.object(profile,'collect',collect),patch('experiments.baseline_delivery.bootstrap.check_exchange'),\
                 patch('experiments.baseline_delivery.return_artifacts.upload_declared',side_effect=OSError('offline')):
                with self.assertRaises(OSError): profile.main()
            files=list(Path(tmp).rglob('padding-profile.json'))
            self.assertEqual(len(files),1);self.assertTrue(read(files[0])['passed']);self.assertEqual(collect.call_count,1)


if __name__=='__main__': unittest.main()
