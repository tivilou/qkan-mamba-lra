import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_lra_data as checker
import rebuild_mega_lra as rebuild
import curate_lra_pairs as curate
import run_data_qualification as scheduled


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, split, x, y=None, task="text", **extra):
        path = self.root / task
        path.mkdir(exist_ok=True)
        np.savez_compressed(path/(split+".npz"), inputs=np.asarray(x),
                            labels=np.asarray([0,1] if y is None else y,dtype=np.int64), **extra)

    def fill(self):
        for n, split in enumerate(checker.SPLITS):
            self.write(split, np.arange(8).reshape(2,4)+n*10)

    def test_clean_numeric_data_never_auto_qualifies(self):
        self.fill()
        report = checker.audit(self.root,["text"])
        self.assertEqual(report["status"],"insufficient")
        self.assertFalse(report["formal_baseline_allowed"])
        self.assertEqual(report["tasks"][0]["failures"],[])
        self.assertEqual(report["tasks"][0]["splits"]["train"]["class_counts"],{"0":1,"1":1})

    def test_missing_real_files_are_insufficient(self):
        report = checker.audit(self.root)
        self.assertEqual(len(report["tasks"]),5)
        self.assertTrue(all(len(t["missing"])==3 for t in report["tasks"]))

    def test_duplicate_validation_test_detected_across_dtype_and_shape(self):
        self.fill()
        x=np.arange(8).reshape(2,4)+20
        self.write("val",x.astype(np.float32)[...,None])
        report=checker.audit(self.root,["text"])["tasks"][0]
        self.assertEqual(report["status"],"revise")
        self.assertEqual(report["overlaps"]["val:test"],2)

    def test_nonfinite_and_out_of_range_labels_rejected(self):
        self.fill()
        self.write("train",[[float('nan'),0,1,2],[1,2,3,4]])
        self.write("test",[[1,2,3,4],[5,6,7,8]],y=[0,2])
        task=checker.audit(self.root,["text"])["tasks"][0]
        self.assertEqual(len(task["failures"]),2)
        self.assertNotIn("train",task["splits"])
        self.assertNotIn("test",task["splits"])

    def test_object_payload_is_rejected_without_unpickling(self):
        self.fill()
        self.write("train",np.asarray([[object()],[object()]],dtype=object))
        task=checker.audit(self.root,["text"])["tasks"][0]
        self.assertEqual(task["status"],"revise")

    def test_large_integer_identity_cannot_silently_round(self):
        self.fill()
        self.write("train",np.full((2,4),2**54,dtype=np.uint64))
        record=checker.audit(self.root,["text"])["tasks"][0]
        self.assertEqual(record['status'],'revise')
        self.assertNotIn('train',record['splits'])

    def test_streaming_multiple_blocks_and_conflicting_labels(self):
        old=checker.BLOCK_BYTES
        checker.BLOCK_BYTES=128
        try:
            self.fill()
            self.write("train",np.tile([[0,0,0,0]],(30,1)),y=[0]*29+[1])
            record=checker.audit(self.root,["text"])["tasks"][0]
            self.assertEqual(record["splits"]["train"]["samples"],30)
            self.assertEqual(record["conflicting_input_labels"],1)
            self.assertEqual(record["splits"]["train"]["duplicate_model_inputs_within_split"],29)
        finally:
            checker.BLOCK_BYTES=old

    def test_retrieval_swapped_pair_overlap(self):
        path=self.root/"retrieval";path.mkdir()
        for split,a,b in [("train",1,2),("val",2,1),("test",3,4)]:
            np.savez(path/(split+".npz"),inputs1=np.full((1,4),a),inputs2=np.full((1,4),b),labels=np.asarray([0]))
        task=checker.audit(self.root,["retrieval"])["tasks"][0]
        self.assertEqual(task["overlaps"]["train:val"],1)

    def test_index_dtype_and_dictionary_label_semantics(self):
        buffer=io.BytesIO()
        rows=[np.asarray([4,2],dtype='<i4'),np.asarray([5,2],dtype='<i4')]
        sizes=np.asarray([2,2],dtype='<i4');pointers=np.asarray([0,8],dtype='<i8')
        header=b'MMIDIDX\0\0'+struct.pack('<QBQ',1,4,2)+sizes.tobytes()+pointers.tobytes()
        with zipfile.ZipFile(buffer,'w') as z:
            z.writestr('label/train.idx',header)
            z.writestr('label/train.bin',b''.join(r.tobytes() for r in rows))
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as z:
            index=rebuild.Indexed(z,'label','train')
            self.assertEqual(index.dtype,np.dtype('<i4'))
            # Frequency order need not equal numeric label order.
            np.testing.assert_array_equal(rebuild.labels_from_index(index,rebuild.SPECIAL+['1','0'],2),[1,0])

    def test_text_partition_is_grouped_deterministic_and_independent(self):
        class Fixture:
            def __init__(self,rows):self.data=rows
            def rows(self):return iter(self.data)
        train=Fixture([np.asarray([i,2]) for i in [4,5,6,7,8,8,9]])
        test=Fixture([np.asarray([9,2])])
        y=np.asarray([0,0,1,1,0,0,1])
        parts,proof=rebuild.independent_text_split(train,y,test,4000,val_per_class=1)
        self.assertFalse(set(parts['train'])&set(parts['val']))
        self.assertNotIn(6,set(parts['train'])|set(parts['val']))
        self.assertEqual(4 in parts['train'],5 in parts['train'])
        self.assertEqual(proof['excluded_train_test_overlap_rows'],1)
        again,_=rebuild.independent_text_split(train,y,test,4000,val_per_class=1)
        np.testing.assert_array_equal(parts['val'],again['val'])

    def test_legacy_placeholder_and_wrong_text_paths_fail_before_writes(self):
        for filename,names in [('download_lra.py',['download_pathfinder','download_retrieval']),('prepare_lra_data.py',['process_text'])]:
            spec=importlib.util.spec_from_file_location('legacy',ROOT/'scripts'/filename)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            for name in names:
                args=[self.root/'output'] if name.startswith('download') else [self.root,self.root/'output']
                with self.assertRaises(RuntimeError):getattr(module,name)(*args)
            self.assertFalse((self.root/'output').exists())

    def test_pair_curation_preserves_source_and_creates_independent_groups(self):
        source=self.root/'source';source.mkdir()
        for folder in ['listops','text','image','pathfinder','metadata','retrieval']:
            (source/folder).mkdir()
        tasks=[]
        original={
            'train':[(1,2,1),(3,4,0),(4,3,1),(6,7,0),(6,7,0)],
            'val':[(2,1,1),(4,5,0)],'test':[(1,2,1),(7,8,0)]}
        splits={}
        for split,rows in original.items():
            path=source/'retrieval'/(split+'.npz')
            np.savez_compressed(path,inputs1=np.asarray([[a]*4 for a,b,y in rows],dtype=np.uint16),inputs2=np.asarray([[b]*4 for a,b,y in rows],dtype=np.uint16),labels=np.asarray([y for a,b,y in rows]))
            splits[split]={'sha256':checker.sha256_file(path)}
        build={'tasks':[{'task':'retrieval','splits':splits,'padded_length':4}]}
        (source/'build-manifest.json').write_text(json.dumps(build))
        out=self.root/'clean'
        report=curate.curate(source,out)
        self.assertEqual(report['unordered_conflicting_groups'],1)
        self.assertEqual(report['ordered_conflicting_groups'],0)
        self.assertEqual([report['splits'][s]['kept_rows'] for s in ['train','val','test']],[1,1,2])
        result=checker.audit(out,['retrieval'])['tasks'][0]
        self.assertEqual(result['failures'],[])
        self.assertEqual(result['conflicting_input_labels'],0)
        self.assertEqual(checker.sha256_file(source/'retrieval/train.npz'),splits['train']['sha256'])

    def test_scheduled_assessment_never_releases_incomplete_qualification(self):
        curated=self.root/'curated';curated.mkdir()
        (curated/'build-manifest.json').write_text('{}')
        (curated/'pair-curation.json').write_text('{}')
        diagnostics={'tasks':[{'task':task,'missing':[],'failures':[]} for task in checker.TASKS]}
        record=scheduled.assessment(curated,diagnostics)
        self.assertEqual(record['status'],'insufficient')
        self.assertFalse(record['formal_baseline_allowed'])
        self.assertEqual(record['release_gate'],'stop')
        self.assertFalse(record['gates']['per_task_data_license_and_use_boundary'])


if __name__ == '__main__':
    unittest.main()
