#!/usr/bin/env python3
"""Audit encoded document relations, including a bounded approximate near-duplicate search."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import tempfile
import zipfile

import numpy as np

from audit_lra_data import ArrayStream, BLOCK_BYTES, SPLITS, sha256_file

MASK = (1 << 64) - 1
# Frozen, deterministic min-hash family; approximate candidate recall, not a proof of absence.
RNG = np.random.default_rng(20261028)
A = RNG.integers(1, 2**63, size=32, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
B = RNG.integers(0, 2**63, size=32, dtype=np.uint64)


def content(row):
    row = np.asarray(row).reshape(-1)
    if row.dtype.kind not in "iu" or np.any(row < 0) or np.any(row > 65535):
        raise ValueError("Require frozen uint16-compatible token IDs")
    # The builder appends EOS and then right pads. Interior padding is invalid.
    nonpad = np.flatnonzero(row != 1)
    if len(nonpad) == 0 or np.any(row[:nonpad[-1] + 1] == 1):
        raise ValueError("Require nonempty right-padded token sequences")
    return np.asarray(row[:nonpad[-1] + 1], dtype="<u2")


def shingles(tokens):
    tokens = np.asarray(tokens, dtype=np.uint64)
    if len(tokens) < 5:
        return np.asarray([int.from_bytes(hashlib.sha256(tokens.tobytes()).digest()[:8], "little")], dtype=np.uint64)
    values = np.zeros(len(tokens) - 4, dtype=np.uint64)
    # Arithmetic intentionally wraps at 64 bits; use set identity, not counts.
    with np.errstate(over="ignore"):
        for i in range(5):
            values = values * np.uint64(1000003) + tokens[i:i + len(values)] + np.uint64(1)
    return np.unique(values)


def signature(values):
    with np.errstate(over="ignore"):
        return (values[:, None] * A + B).min(axis=0).astype("<u8")


def similarity(left, right):
    common = np.intersect1d(left, right, assume_unique=True).size
    return common / (len(left) + len(right) - common)


def audit_relations(root: Path, scratch: Path, candidate_limit=200000):
    connection = sqlite3.connect(scratch / "relations.sqlite")
    connection.executescript("""
        PRAGMA cache_size=-8192;
        CREATE TABLE docs(id INTEGER PRIMARY KEY, digest BLOB UNIQUE, tokens BLOB, splitmask INTEGER);
        CREATE TABLE occurrences(split TEXT, rowid_in_split INTEGER, side INTEGER, doc INTEGER);
        CREATE TABLE bands(band INTEGER, key BLOB, doc INTEGER);
        CREATE INDEX bands_lookup ON bands(band,key);
        CREATE INDEX occurrences_lookup ON occurrences(doc,split);
    """)
    totals = {}
    for split_index, split in enumerate(SPLITS):
        path = root / "retrieval" / (split + ".npz")
        with ExitStack() as stack:
            archive = stack.enter_context(zipfile.ZipFile(path))
            streams = [ArrayStream(archive, f"inputs{side}.npy", stack) for side in (1, 2)]
            if streams[0].shape != streams[1].shape:
                raise ValueError("Paired shapes differ")
            n = streams[0].shape[0]
            batch = max(1, BLOCK_BYTES // sum(s.row_bytes for s in streams))
            for start in range(0, n, batch):
                blocks = [s.read(min(batch, n-start)) for s in streams]
                for offset in range(len(blocks[0])):
                    for side, block in enumerate(blocks, 1):
                        tokens = content(block[offset])
                        encoded = tokens.tobytes()
                        digest = hashlib.sha256(encoded).digest()
                        found = connection.execute("SELECT id,tokens FROM docs WHERE digest=?", (digest,)).fetchone()
                        if found:
                            doc_id = found[0]
                            if found[1] != encoded:
                                raise ValueError("Digest collision")
                            connection.execute("UPDATE docs SET splitmask=splitmask|? WHERE id=?", (1 << split_index, doc_id))
                        else:
                            doc_id = connection.execute("INSERT INTO docs(digest,tokens,splitmask) VALUES(?,?,?)", (digest,encoded,1 << split_index)).lastrowid
                        connection.execute("INSERT INTO occurrences VALUES(?,?,?,?)", (split,start+offset,side,doc_id))
                connection.commit()
        totals[split] = {"rows": n, "input_sha256": sha256_file(path)}
        print(json.dumps({"stage": "document_index", "split": split, "rows": n}), flush=True)
    shared = {}
    for i, j in itertools.combinations(range(3), 2):
        key = f"{SPLITS[i]}:{SPLITS[j]}"
        mask = (1 << i) | (1 << j)
        count = connection.execute("SELECT COUNT(*) FROM docs WHERE splitmask&?=?", (mask,mask)).fetchone()[0]
        shared[key] = {"shared_unique_documents": count, "affected_rows": {}}
        for split in (SPLITS[i], SPLITS[j]):
            shared[key]["affected_rows"][split] = connection.execute("SELECT COUNT(DISTINCT o.rowid_in_split) FROM occurrences o JOIN docs d ON o.doc=d.id WHERE o.split=? AND d.splitmask&?=?", (split,mask,mask)).fetchone()[0]
    # Only unique documents are shingled. No raw text or document hashes enter the compact report.
    for doc_id, encoded in connection.execute("SELECT id,tokens FROM docs"):
        sig = signature(shingles(np.frombuffer(encoded, dtype="<u2")))
        connection.executemany("INSERT INTO bands VALUES(?,?,?)", [(band,sig[band*4:(band+1)*4].tobytes(),doc_id) for band in range(8)])
    connection.commit()
    # SQL deduplication is disk-backed. The explicit limit bounds verification work.
    cursor = connection.execute("SELECT DISTINCT a.doc,b.doc FROM bands a JOIN bands b ON a.band=b.band AND a.key=b.key AND a.doc<b.doc ORDER BY a.doc,b.doc LIMIT ?", (candidate_limit+1,))
    verified = near_pairs = 0
    near_cross = {f"{a}:{b}": 0 for a,b in itertools.combinations(SPLITS,2)}
    truncated = False
    for left_id,right_id in cursor:
        if verified == candidate_limit:
            truncated = True
            break
        left,lmask = connection.execute("SELECT tokens,splitmask FROM docs WHERE id=?", (left_id,)).fetchone()
        right,rmask = connection.execute("SELECT tokens,splitmask FROM docs WHERE id=?", (right_id,)).fetchone()
        score = similarity(shingles(np.frombuffer(left,dtype="<u2")), shingles(np.frombuffer(right,dtype="<u2")))
        verified += 1
        if score >= 0.90:
            near_pairs += 1
            for i,j in itertools.combinations(range(3),2):
                if ((lmask & (1 << i) and rmask & (1 << j)) or
                        (lmask & (1 << j) and rmask & (1 << i))):
                    near_cross[f"{SPLITS[i]}:{SPLITS[j]}"] += 1
    unique = connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    connection.close()
    return {"schema": "qkan-retrieval-relations.v1", "profile": "mega-independent-pairclean-v2",
            "build_manifest_sha256": sha256_file(root / "build-manifest.json"), "splits": totals,
            "unit": "encoded document after 4000-token truncation; EOS retained; right padding removed",
            "unique_documents": unique, "exact_shared_documents": shared,
            "near_duplicate_search": {"method": "32 deterministic affine uint64 minhashes; 8 bands of 4; exact 5-token shingle Jaccard verification",
                "threshold": 0.90, "candidate_limit": candidate_limit, "candidates_verified": verified,
                "truncated": truncated, "verified_near_duplicate_pairs": near_pairs,
                "cross_split_near_duplicate_pairs": near_cross, "exhaustive": False,
                "limitations": "Approximate candidates can miss pairs; repeated encoded documents are counted separately above. Hash collisions and repetitive text can affect shingle identity."},
            "source_document_and_author_ids_available": False,
            "formal_baseline_allowed": False,
            "claim_boundary": "Pair disjointness does not establish unseen-document or unseen-author generalization. Do not certify document/author independence from encoded pairs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite an audit identity")
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="relations-", dir=args.scratch) as temp:
        report = audit_relations(args.data, Path(temp))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
