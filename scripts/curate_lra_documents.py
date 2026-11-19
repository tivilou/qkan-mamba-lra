#!/usr/bin/env python3
"""Create a new document-group-disjoint Retrieval population; preserve every parent byte."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from audit_lra_data import SPLITS, audit, sha256_file
from audit_lra_relations import audit_relations, content
from curate_lra_pairs import blocks
from qualification_pilot import pilot
from rebuild_mega_lra import save_arrays

SALT = b"qkan-document-split-20261028-v1:"


def partition(representative):
    bucket = int.from_bytes(hashlib.sha256(SALT+bytes.fromhex(representative)).digest()[:8], "big") % 100
    return 0 if bucket < 80 else (1 if bucket < 90 else 2)


def curate_documents(source, output, scratch):
    source,output,scratch = source.resolve(),output.resolve(),scratch.resolve()
    if output.exists() or output == source or output.is_relative_to(source):
        raise ValueError("Require a new sibling version; never overwrite source")
    build = json.loads((source/"build-manifest.json").read_text())
    for task in build["tasks"]:
        for split,record in task["splits"].items():
            if sha256_file(source/task["task"]/(split+".npz")) != record["sha256"]:
                raise ValueError("Source build identity changed")
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="doc-group-index-",dir=scratch) as work:
        relation_report,groups = audit_relations(source,Path(work),return_groups=True)
    if relation_report["near_duplicate_search"]["truncated"]:
        raise ValueError("Near-duplicate candidate budget exhausted; refuse an incomplete grouping")
    selections = [[] for _ in SPLITS]
    excluded = 0
    for source_id,split in enumerate(SPLITS):
        for start,block in blocks(source/"retrieval"/(split+".npz")):
            for offset,(a,b,label) in enumerate(zip(block["inputs1"],block["inputs2"],block["labels"])):
                hashes = [hashlib.sha256(content(row).tobytes()).hexdigest() for row in (a,b)]
                lane = [partition(groups[digest]) for digest in hashes]
                if lane[0] == lane[1]:
                    selections[lane[0]].append((source_id,start+offset,int(label)))
                else:
                    excluded += 1
    if any(not rows for rows in selections):
        raise ValueError("Document split has an empty partition")
    output.mkdir(parents=True)
    # Copy immutable parent bytes for the other four tasks and all existing provenance.
    for name in ("listops","text","image","pathfinder","metadata"):
        shutil.copytree(source/name,output/name)
    report = {"schema": "qkan-document-curation.v1", "profile": "mega-document-disjoint-v3",
              "status": "curated_not_qualified", "parent_build_sha256": sha256_file(source/"build-manifest.json"),
              "code_sha256": sha256_file(Path(__file__)), "source_relations": relation_report,
              "policy": {"sampling_unit": "connected component of exact encoded documents and verified near-duplicate edges",
                         "seed_salt": SALT.decode(), "document_group_ratios": [0.8,0.1,0.1],
                         "label_stratification": False, "rule": "hash groups before examining labels; keep pairs only when both groups select the same partition",
                         "original_partitions": "pooled parent pairs; original test is no longer this version's test; provenance retained"},
              "excluded_cross_partition_pairs": excluded, "splits": {},
              "limitations": ["Encoded-document holdout, not author holdout; source IDs unavailable.",
                              "Near-duplicate search is approximate; all verified edges are grouped, undiscovered similarities may remain.",
                              "This is a new adapted evaluation population, not directly comparable with official/historical LRA scores."]}
    retrieval = next(t for t in build["tasks"] if t["task"] == "retrieval")
    length = retrieval["padded_length"]
    for target_id,split in enumerate(SPLITS):
        selected = np.asarray(selections[target_id],dtype=np.int64)
        with tempfile.TemporaryDirectory(prefix="doc-output-",dir=output) as work:
            arrays = {key:np.lib.format.open_memmap(Path(work)/(key+".npy"),mode="w+",dtype="<u2",shape=(len(selected),length)) for key in ("inputs1","inputs2")}
            arrays["labels"] = np.empty(len(selected),dtype=np.int64)
            dest = 0
            for source_id,source_split in enumerate(SPLITS):
                rows = selected[selected[:,0] == source_id,1]
                for start,block in blocks(source/"retrieval"/(source_split+".npz")):
                    offsets = rows[(rows >= start) & (rows < start+len(block["labels"]))]-start
                    count = len(offsets)
                    for key in arrays:
                        arrays[key][dest:dest+count] = block[key][offsets]
                    dest += count
            if dest != len(selected):
                raise ValueError("Incomplete document split")
            result = save_arrays(output/"retrieval"/(split+".npz"),arrays)
            class_counts = np.bincount(arrays["labels"],minlength=2).tolist()
            for value in arrays.values():
                if isinstance(value,np.memmap):
                    value._mmap.close()
        save_arrays(output/"metadata"/("retrieval-"+split+"-docsplit-rows.npz"),{"parent_split":selected[:,0].astype(np.uint8),"parent_row":selected[:,1]})
        report["splits"][split] = {"kept_rows":len(selected),"class_counts":class_counts,"output":result}
    identities = sorted(groups)
    save_arrays(output/"metadata/document-group-assignment.npz", {
        "document_sha256": np.asarray([list(bytes.fromhex(key)) for key in identities],dtype=np.uint8),
        "group_sha256": np.asarray([list(bytes.fromhex(groups[key])) for key in identities],dtype=np.uint8),
        "split_id": np.asarray([partition(groups[key]) for key in identities],dtype=np.uint8)})
    new_build = json.loads(json.dumps(build))
    new_build.update(profile=report["profile"],status="curated_not_qualified",parent_build_sha256=report["parent_build_sha256"],curation_code_sha256=report["code_sha256"])
    for task in new_build["tasks"]:
        if task["task"] == "retrieval":
            task["splits"] = {split:item["output"] for split,item in report["splits"].items()}
            task["curation"] = "document-curation.json; encoded document-group-disjoint partitions"
    (output/"build-manifest.json").write_text(json.dumps(new_build,indent=2)+"\n")
    (output/"document-curation.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--curated",type=Path,required=True)
    parser.add_argument("--scratch",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("Require an empty scheduler output directory")
    report = curate_documents(args.source,args.curated,args.scratch)
    diagnostics = audit(args.curated)
    with tempfile.TemporaryDirectory(prefix="doc-recheck-",dir=args.scratch) as work:
        relations = audit_relations(args.curated,Path(work))
    checks = {"source_preserved": report["parent_build_sha256"] == sha256_file(args.source/"build-manifest.json"),
              "all_numeric_arrays_and_exact_pairs": all(not t["failures"] and not t["missing"] for t in diagnostics["tasks"]),
              "document_disjoint": all(v["shared_unique_documents"] == 0 for v in relations["exact_shared_documents"].values()),
              "verified_near_duplicate_disjoint": not relations["near_duplicate_search"]["truncated"] and not any(relations["near_duplicate_search"]["cross_split_near_duplicate_pairs"].values()),
              "both_retrieval_classes": all(all(count>0 for count in item["class_counts"]) for item in report["splits"].values())}
    for name,value in (("document-curation.json",report),("diagnostics.json",diagnostics),("relations.json",relations)):
        (args.output_dir/name).write_text(json.dumps(value,indent=2)+"\n")
    shutil.copyfile(args.curated/"build-manifest.json",args.output_dir/"build-manifest.json")
    (args.output_dir/"checks.json").write_text(json.dumps({"checks":checks,"formal_baseline_allowed":False,"status":"insufficient" if all(checks.values()) else "revise"},indent=2)+"\n")
    if all(checks.values()):
        pilot_report = pilot(args.curated,tasks=["retrieval"])
        (args.output_dir/"pilot.json").write_text(json.dumps(pilot_report,indent=2)+"\n")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
