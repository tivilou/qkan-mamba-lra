import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_lra_relations import audit_relations, content, shingles, signature, similarity
from curate_lra_documents import curate_documents, partition
from audit_lra_data import sha256_file
import hashlib
from qualification_pilot import evaluate_expression, features, fit_reference, sample_split


class QualificationTests(unittest.TestCase):
    def test_pair_disjoint_can_still_share_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"retrieval").mkdir()
            (root/"build-manifest.json").write_text("{}")
            scratch = root/"scratch"
            scratch.mkdir()
            for split,a,b in [("train",4,5),("val",4,6),("test",7,8)]:
                np.savez(root/"retrieval"/(split+".npz"),inputs1=np.asarray([[a,2,1]],dtype=np.uint16),inputs2=np.asarray([[b,2,1]],dtype=np.uint16),labels=np.asarray([0]))
            report = audit_relations(root,scratch)
            self.assertEqual(report["unique_documents"],5)
            self.assertEqual(report["exact_shared_documents"]["train:val"]["shared_unique_documents"],1)
            self.assertEqual(report["exact_shared_documents"]["train:val"]["affected_rows"],{"train":1,"val":1})
            self.assertFalse(report["formal_baseline_allowed"])
            self.assertFalse(report["near_duplicate_search"]["exhaustive"])

    def test_padding_specials_and_minhash_candidate(self):
        np.testing.assert_array_equal(content(np.asarray([4,5,2,1,1])),[4,5,2])
        with self.assertRaises(ValueError):
            content(np.asarray([4,1,2,1]))
        tokens=np.arange(4,1004,dtype=np.uint16)
        other=tokens.copy();other[500]=4000
        a,b=shingles(tokens),shingles(other)
        self.assertGreater(similarity(a,b),0.98)
        self.assertTrue(any(np.array_equal(signature(a)[i:i+4],signature(b)[i:i+4]) for i in range(0,32,4)))

    def test_pilot_never_opens_test_and_sample_identity_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"text").mkdir()
            x=np.arange(600,dtype=np.uint16).reshape(100,6)+4
            np.savez_compressed(root/"text/train.npz",inputs=x,labels=np.arange(100)%2)
            a,ap=sample_split(root,"text","train",17,9)
            b,bp=sample_split(root,"text","train",17,9)
            np.testing.assert_array_equal(a["inputs"],b["inputs"])
            self.assertEqual(ap,bp)
            with self.assertRaises(ValueError):
                sample_split(root,"text","test",1,0)

    def test_pair_features_are_symmetric_and_token_bags_ignore_special_ids(self):
        a=np.asarray([[4,5,2,1]])
        b=np.asarray([[6,4,2,1]])
        one=features("retrieval",dict(inputs1=a,inputs2=b,labels=np.asarray([0])))
        two=features("retrieval",dict(inputs1=b,inputs2=a,labels=np.asarray([0])))
        np.testing.assert_array_equal(one,two)
        np.testing.assert_array_equal(features("text",dict(inputs=a,labels=np.asarray([0]))),features("text",dict(inputs=np.asarray([[4,5,1,1]]),labels=np.asarray([0]))))

    def test_reference_learns_known_signal_and_shuffled_labels_do_not(self):
        x=np.tile(np.asarray([[-1.,0.],[1.,0.]],dtype=np.float32),(256,1))
        y=np.tile([0,1],256)
        w,b=fit_reference(x,y,2)
        self.assertTrue(np.all((x@w+b).argmax(1)==y))
        shuffled=np.random.default_rng(1).permutation(y)
        w,b=fit_reference(x,shuffled,2)
        self.assertLessEqual(np.mean((x@w+b).argmax(1)==shuffled),0.56)

    def test_listops_nested_reference_and_incomplete_input(self):
        symbols=["<s>","<pad>","</s>","<unk>","[SM","[MAX","1","9","]"]
        self.assertEqual(evaluate_expression([4,6,5,6,7,8,8,2,1],symbols),0)
        with self.assertRaises(ValueError):
            evaluate_expression([4,6,2],symbols)

    def test_document_partition_reassigns_pairs_without_cross_split_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"source";source.mkdir()
            for name in ("retrieval","listops","text","image","pathfinder","metadata"):
                (source/name).mkdir()
            docs=[[] for _ in range(3)]
            for token in range(4,300):
                digest=hashlib.sha256(np.asarray([token,2],dtype="<u2").tobytes()).hexdigest()
                docs[partition(digest)].append(token)
            self.assertTrue(all(len(values)>=2 for values in docs))
            rows={"train":[(docs[2][0],docs[2][1],0),(docs[0][0],docs[1][0],1)],
                  "val":[(docs[0][0],docs[0][1],1)],"test":[(docs[1][0],docs[1][1],0)]}
            splits={}
            for split,items in rows.items():
                path=source/"retrieval"/(split+".npz")
                np.savez(path,inputs1=np.asarray([[a,2,1] for a,b,y in items],dtype=np.uint16),inputs2=np.asarray([[b,2,1] for a,b,y in items],dtype=np.uint16),labels=np.asarray([y for a,b,y in items]))
                splits[split]={"sha256":sha256_file(path)}
            (source/"build-manifest.json").write_text(json.dumps({"profile":"fixture","tasks":[{"task":"retrieval","padded_length":3,"splits":splits}]}))
            output=root/"clean"
            report=curate_documents(source,output,root/"scratch")
            self.assertEqual(report["excluded_cross_partition_pairs"],1)
            self.assertEqual([report["splits"][s]["kept_rows"] for s in ("train","val","test")],[1,1,1])
            scratch=root/"recheck";scratch.mkdir()
            audit=audit_relations(output,scratch)
            self.assertTrue(all(v["shared_unique_documents"]==0 for v in audit["exact_shared_documents"].values()))
            for split in rows:
                self.assertEqual(sha256_file(source/"retrieval"/(split+".npz")),splits[split]["sha256"])
            with np.load(output/"metadata/retrieval-train-docsplit-rows.npz",allow_pickle=False) as mapping:
                np.testing.assert_array_equal(mapping["parent_split"],[1])
                np.testing.assert_array_equal(mapping["parent_row"],[0])
            with self.assertRaises(ValueError):
                curate_documents(source,output,root/"scratch")


if __name__ == "__main__":
    unittest.main()
