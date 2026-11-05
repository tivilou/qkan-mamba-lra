#!/usr/bin/env python3
"""Stream NPZ diagnostics without training; passing diagnostics never qualifies data."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import zipfile

import numpy as np
from numpy.lib import format as npformat

TASKS = {"listops": 10, "text": 2, "retrieval": 2, "image": 10, "pathfinder": 2}
SPLITS = ("train", "val", "test")
BLOCK_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024**3


class AuditError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArrayStream:
    """Read C-order numeric NPY members in bounded blocks; never unpickle."""
    def __init__(self, archive, member, stack):
        info = archive.getinfo(member)
        if info.file_size > MAX_MEMBER_BYTES:
            raise AuditError("Member exceeds the explicit 8 GiB audit limit")
        self.handle = stack.enter_context(archive.open(member))
        version = npformat.read_magic(self.handle)
        if version == (1, 0):
            shape, fortran, dtype = npformat.read_array_header_1_0(self.handle)
        elif version == (2, 0):
            shape, fortran, dtype = npformat.read_array_header_2_0(self.handle)
        else:
            raise AuditError("Unsupported NPY version")
        if (fortran or dtype.kind not in "iuf" or not shape or
                any(not isinstance(v, int) or v <= 0 for v in shape)):
            raise AuditError("Require nonempty C-order numeric arrays without objects")
        self.shape, self.dtype = shape, dtype
        self.row_values = math.prod(shape[1:])
        self.row_bytes = self.row_values * dtype.itemsize
        if self.row_bytes > BLOCK_BYTES:
            raise AuditError("One sample exceeds the bounded block size")
        expected = math.prod(shape) * dtype.itemsize
        if self.handle.tell() + expected != info.file_size:
            raise AuditError("NPY header/payload size mismatch")
        self.remaining = shape[0]

    def read(self, rows):
        if rows > self.remaining:
            raise AuditError("Array row count mismatch")
        size = rows * self.row_bytes
        raw = self.handle.read(size)
        if len(raw) != size:
            raise AuditError("Truncated numeric payload")
        self.remaining -= rows
        return np.frombuffer(raw, dtype=self.dtype).reshape((rows, *self.shape[1:]))


def audit_split(path, task, split, connection):
    inputs = ("inputs1", "inputs2") if task == "retrieval" else ("inputs",)
    keys = (*inputs, "labels")
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(path))
        if sorted(archive.namelist()) != sorted(key + ".npy" for key in keys):
            raise AuditError("Require exactly the declared inputs and labels; reject duplicate/extra members")
        arrays = {key: ArrayStream(archive, key + ".npy", stack) for key in keys}
        labels = arrays["labels"]
        n = labels.shape[0]
        if len(labels.shape) != 1 or labels.dtype.kind not in "iu":
            raise AuditError("Labels must be a one-dimensional integer array")
        for key in inputs:
            shape = arrays[key].shape
            if (shape[0] != n or len(shape) not in (2, 3) or
                    (len(shape) == 3 and shape[2] != 1)):
                raise AuditError("Inputs must have matching N and shape (N,L) or (N,L,1)")
        if task == "retrieval" and arrays["inputs1"].shape[1] != arrays["inputs2"].shape[1]:
            raise AuditError("Retrieval documents must have equal declared padded lengths")
        block_rows = max(1, BLOCK_BYTES // sum(a.row_values * 8 for a in arrays.values()))
        counts = np.zeros(TASKS[task], dtype=np.int64)
        stats = {key: {"min": None, "max": None, "values": 0, "zeros": 0,
                       "non_integer_values": 0} for key in inputs}
        nonfinite = 0
        for start in range(0, n, block_rows):
            take = min(block_rows, n - start)
            block = {key: array.read(take) for key, array in arrays.items()}
            y = block["labels"]
            if np.any(y < 0) or np.any(y >= TASKS[task]):
                raise AuditError("Label outside the task class range")
            counts += np.bincount(y.astype(np.int64), minlength=TASKS[task])
            for key in inputs:
                value = block[key]
                finite = np.isfinite(value)
                nonfinite += int(value.size - np.count_nonzero(finite))
                if not np.all(finite):
                    raise AuditError("Model inputs contain NaN or infinity")
                s = stats[key]
                lo, hi = float(value.min()), float(value.max())
                if value.dtype.kind in "iu" and (value.min() < -(2**53) or value.max() > 2**53):
                    raise AuditError("Integer input exceeds exact canonical identity range")
                s["min"] = lo if s["min"] is None else min(lo, s["min"])
                s["max"] = hi if s["max"] is None else max(hi, s["max"])
                s["values"] += int(value.size)
                s["zeros"] += int(np.count_nonzero(value == 0))
                s["non_integer_values"] += int(np.count_nonzero(value != np.floor(value)))
            for i, label in enumerate(y):
                parts = []
                for key in inputs:
                    # Canonical numeric identity: no dtype/endian or trailing singleton differences.
                    value = np.asarray(block[key][i], dtype="<f8").reshape(-1).copy()
                    value[value == 0] = 0  # Normalize signed zero.
                    parts.append(hashlib.sha256(str(value.size).encode() + b":" + value.tobytes()).digest())
                # For pair matching, swapped documents count as the same independent pair.
                if task == "retrieval":
                    parts.sort()
                digest = hashlib.sha256(b"".join(parts)).digest()
                connection.execute("INSERT OR IGNORE INTO seen VALUES (?,?,?,?)",
                                   (task, digest, split, int(label)))
            connection.commit()
        for array in arrays.values():
            if array.remaining or array.handle.read(1):
                raise AuditError("Unread or trailing payload")
        unique = connection.execute("SELECT COUNT(DISTINCT digest) FROM seen WHERE task=? AND split=?", (task, split)).fetchone()[0]
        return {"path": f"{task}/{split}.npz", "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path), "samples": n,
                "arrays": {key: {"shape": list(a.shape), "dtype": a.dtype.str} for key, a in arrays.items()},
                "class_counts": {str(i): int(v) for i, v in enumerate(counts)},
                "majority_class_accuracy": int(counts.max()) / n,
                "input_statistics": stats, "nonfinite_values": nonfinite,
                "unique_model_inputs": unique, "duplicate_model_inputs_within_split": n - unique}


def audit(root: Path, tasks=None):
    root = root.resolve()
    tasks = list(TASKS) if tasks is None else list(tasks)
    if not tasks or len(tasks) != len(set(tasks)) or any(t not in TASKS for t in tasks):
        raise AuditError("Choose distinct known tasks")
    report = {"schema_version": "qkan-data-diagnostics.v1", "scope": tasks,
              "status": "insufficient", "formal_baseline_allowed": False,
              "mode": "complete declared arrays; no row sampling or model training",
              "tasks": [], "limitations": [
                  "Passing numeric checks is not dataset qualification.",
                  "Source, license, conversion provenance, split identity, tokenizer/mask semantics and learnability require separate evidence.",
                  "Input hashes detect exact encoded duplicates; semantic/near duplicates and shared Retrieval documents require source IDs.",
                  "Raw samples and per-sample hashes are not exported."]}
    with tempfile.TemporaryDirectory(prefix="qkan-audit-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "identity.sqlite"))
        try:
            connection.execute("CREATE TABLE seen (task TEXT, digest BLOB, split TEXT, label INTEGER, PRIMARY KEY(task,digest,split,label)) WITHOUT ROWID")
            for task in tasks:
                record = {"task": task, "status": "insufficient", "splits": {}, "failures": [], "missing": [], "overlaps": {}}
                for split in SPLITS:
                    path = root / task / (split + ".npz")
                    if path.is_symlink() or path.parent.is_symlink():
                        record["failures"].append(split + ": symlink rejected")
                    elif not path.is_file():
                        record["missing"].append(split)
                    else:
                        try:
                            record["splits"][split] = audit_split(path, task, split, connection)
                        except (AuditError, OSError, ValueError, EOFError, zipfile.BadZipFile, RuntimeError) as error:
                            # Remove partial identities so a corrupt split cannot affect another split's counts.
                            connection.execute("DELETE FROM seen WHERE task=? AND split=?", (task, split))
                            connection.commit()
                            record["failures"].append(split + ": " + type(error).__name__ + ": " + str(error))
                lengths = {tuple(v["arrays"][key]["shape"][1] for key in (("inputs1", "inputs2") if task == "retrieval" else ("inputs",))) for v in record["splits"].values()}
                if len(lengths) > 1:
                    record["failures"].append("Padded input lengths differ across splits")
                for a, b in itertools.combinations(SPLITS, 2):
                    count = connection.execute("SELECT COUNT(DISTINCT a.digest) FROM seen a JOIN seen b ON a.task=b.task AND a.digest=b.digest WHERE a.task=? AND a.split=? AND b.split=?", (task, a, b)).fetchone()[0]
                    record["overlaps"][a + ":" + b] = count
                    if count:
                        record["failures"].append(f"{a}:{b}: {count} shared encoded inputs")
                conflicts = connection.execute("SELECT COUNT(*) FROM (SELECT digest FROM seen WHERE task=? GROUP BY digest HAVING COUNT(DISTINCT label)>1)", (task,)).fetchone()[0]
                record["conflicting_input_labels"] = conflicts
                if conflicts:
                    record["failures"].append(f"{conflicts} encoded inputs have conflicting labels")
                record["status"] = "revise" if record["failures"] else "insufficient"
                report["tasks"].append(record)
        finally:
            connection.close()
    if any(t["status"] == "revise" for t in report["tasks"]):
        report["status"] = "revise"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tasks", choices=tuple(TASKS), nargs="+", default=list(TASKS))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    if report_path.exists() or report_path.is_relative_to(args.data_dir.resolve()):
        parser.error("Report must be new and outside the data directory")
    report = audit(args.data_dir, args.tasks)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".pending")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({"status": report["status"], "tasks": len(report["tasks"]), "formal_baseline_allowed": False}))
    return 2  # This diagnostic cannot unlock formal baselines by itself.


if __name__ == "__main__":
    raise SystemExit(main())
