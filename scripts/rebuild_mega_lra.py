#!/usr/bin/env python3
"""Rebuild five LRA tasks from pinned MEGA bytes; no pickle execution or training."""
from contextlib import ExitStack
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import zipfile

import numpy as np

from audit_lra_data import sha256_file

ARCHIVE_SHA256 = "b141aacd5537ad6108862117597c41d9d2a67d6420c54f74c76df615073fee13"
SPLITS = {"train": "train", "valid": "val", "test": "test"}
TEXT_TASKS = {"listops": ("listops", 2048, 10), "text": ("imdb-4000", 4000, 2),
              "retrieval": ("aan", 4000, 2)}
SPECIAL = ["<s>", "<pad>", "</s>", "<unk>"]
DTYPES = {1: "u1", 2: "i1", 3: "<i2", 4: "<i4", 5: "<i8", 6: "<f4", 7: "<f8", 8: "<u2"}


def dictionary(archive, prefix):
    words = archive.read(prefix + "/dict.txt").decode("utf-8").splitlines()
    result = SPECIAL + [line.rsplit(" ", 1)[0] for line in words]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate dictionary symbol")
    return result


class Indexed:
    """Sequential MMIDIDX reader for the pinned MEGA fairseq format."""
    def __init__(self, archive, prefix, split):
        self.archive, self.member = archive, prefix + "/" + split + ".bin"
        raw = archive.read(prefix + "/" + split + ".idx")
        if raw[:9] != b"MMIDIDX\0\0" or struct.unpack_from("<Q", raw, 9)[0] != 1:
            raise ValueError("Unexpected index magic/version")
        self.dtype = np.dtype(DTYPES[raw[17]])
        self.n = struct.unpack_from("<Q", raw, 18)[0]
        if len(raw) != 26 + self.n * 12:
            raise ValueError("Unexpected index length")
        self.sizes = np.frombuffer(raw, dtype="<i4", count=self.n, offset=26)
        self.pointers = np.frombuffer(raw, dtype="<i8", count=self.n, offset=26 + self.n * 4)
        expected = np.zeros(self.n, dtype=np.int64)
        if self.n > 1:
            expected[1:] = np.cumsum(self.sizes[:-1].astype(np.int64)) * self.dtype.itemsize
        if (not self.n or np.any(self.sizes < 0) or not np.array_equal(expected, self.pointers) or
                int(np.sum(self.sizes, dtype=np.int64)) * self.dtype.itemsize != archive.getinfo(self.member).file_size):
            raise ValueError("Invalid index pointers, sizes or binary extent")

    def rows(self):
        with self.archive.open(self.member) as handle:
            for size in self.sizes:
                length = int(size) * self.dtype.itemsize
                raw = handle.read(length)
                if len(raw) != length:
                    raise ValueError("Truncated indexed data")
                yield np.frombuffer(raw, dtype=self.dtype)
            if handle.read(1):
                raise ValueError("Trailing indexed bytes")


def labels_from_index(index, symbols, classes):
    result = []
    for row in index.rows():
        ids = [int(i) for i in row if i != 2]
        if len(ids) != 1 or not 4 <= ids[0] < len(symbols):
            raise ValueError("Expected one label token plus optional EOS")
        label = int(symbols[ids[0]])
        if not 0 <= label < classes:
            raise ValueError("Label outside semantic class range")
        result.append(label)
    return np.asarray(result, dtype=np.int64)


def encoded_hash(row, max_len):
    value = np.asarray(row[:max_len], dtype="<u2")
    return hashlib.sha256(len(value).to_bytes(8, "little") + value.tobytes()).digest()


def independent_text_split(train_index, labels, test_index, max_len, val_per_class=2500):
    """Group identical encoded reviews; validation comes only from original train."""
    test_ids = {encoded_hash(row, max_len) for row in test_index.rows()}
    groups = {}
    excluded = []
    for i, row in enumerate(train_index.rows()):
        identity = encoded_hash(row, max_len)
        if identity in test_ids:
            excluded.append(i)
        else:
            groups.setdefault(identity, []).append(i)
    train, val = [], []
    counts = {0: 0, 1: 0}
    conflicts = []
    for identity, indices in sorted(groups.items(), key=lambda item: hashlib.sha256(b"qkan-text-split-v1:" + item[0]).digest()):
        classes = {int(labels[i]) for i in indices}
        if len(classes) != 1:
            conflicts.extend(indices)
            continue
        label = next(iter(classes))
        if counts[label] < val_per_class:
            val.extend(indices)
            counts[label] += len(indices)
        else:
            train.extend(indices)
    if not train or any(c < val_per_class for c in counts.values()):
        raise ValueError("Insufficient independent training examples for declared validation quota")
    return {"train": np.asarray(sorted(train)), "val": np.asarray(sorted(val))}, {
        "policy": "group identical encoded input; SHA-256 ordering with qkan-text-split-v1 salt; 2500 per class validation target",
        "validation_class_counts": counts, "excluded_train_test_overlap_rows": len(excluded),
        "excluded_conflicting_training_rows": len(conflicts),
        "test_access": "input identities only for exclusion; no test labels or performance used to choose validation",
        "comparability": "adapted independent validation protocol; not the original LRA/MEGA validation protocol"}


def save_arrays(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("Refuse existing output")
    temp = path.with_name(path.name + ".pending")
    with temp.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    temp.rename(path)
    return {"file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}}


def build_indexed(archive, out, task):
    folder, max_len, classes = TEXT_TASKS[task]
    prefix = "lra/" + folder
    symbols = dictionary(archive, prefix + "/src-bin")
    label_symbols = dictionary(archive, prefix + "/label-bin")
    result = {"task": task, "source_prefix": prefix, "padded_length": max_len,
              "pad_id": 1, "eos_id": 2, "unk_id": 3, "input_token_ids": "unchanged MEGA dictionary IDs; EOS retained; right pad=1",
              "label_mapping": {str(i): word for i, word in enumerate(label_symbols) if i >= 4 and word.isdigit()}, "splits": {}}
    plans = {s: {s2: None} for s, s2 in SPLITS.items()}
    if task == "text":
        val_bin = archive.read(prefix + "/src-bin/valid.bin")
        test_bin = archive.read(prefix + "/src-bin/test.bin")
        result["original_valid_equals_test_inputs"] = val_bin == test_bin
        del val_bin, test_bin
        train_index = Indexed(archive, prefix + "/src-bin", "train")
        y = labels_from_index(Indexed(archive, prefix + "/label-bin", "train"), label_symbols, classes)
        partitions, result["split_repair"] = independent_text_split(train_index, y, Indexed(archive, prefix + "/src-bin", "test"), max_len)
        plans = {"train": partitions, "test": {"test": None}}
    for source_split, destinations in plans.items():
        y = labels_from_index(Indexed(archive, prefix + "/label-bin", source_split), label_symbols, classes)
        indices = [Indexed(archive, prefix + "/src-bin", source_split)]
        if task == "retrieval":
            indices.append(Indexed(archive, prefix + "/src1-bin", source_split))
        if any(index.n != len(y) for index in indices):
            raise ValueError("Input and label count mismatch")
        for split, selected in destinations.items():
            if selected is None:
                selected = np.arange(len(y), dtype=np.int64)
            with tempfile.TemporaryDirectory(prefix="rebuild-", dir=out) as temp:
                arrays, lengths = {}, {}
                for side in range(len(indices)):
                    name = ("inputs1", "inputs2")[side] if task == "retrieval" else "inputs"
                    arrays[name] = np.lib.format.open_memmap(Path(temp)/(name+".npy"), mode="w+", dtype="<u2", shape=(len(selected), max_len))
                    arrays[name][:] = 1
                    lengths[name] = np.empty(len(selected), dtype=np.int32)
                    cursor = 0
                    for source_id, row in enumerate(indices[side].rows()):
                        if cursor == len(selected):
                            break
                        if source_id != selected[cursor]:
                            continue
                        if np.any(row < 0) or np.any(row >= len(symbols)):
                            raise ValueError("Input token outside dictionary")
                        take = min(len(row), max_len)
                        arrays[name][cursor, :take] = row[:take]
                        lengths[name][cursor] = take
                        cursor += 1
                    if cursor != len(selected):
                        raise ValueError("Incomplete selected rows")
                    arrays[name].flush()
                arrays["labels"] = y[selected]
                result["splits"][split] = save_arrays(out/task/(split+".npz"), arrays)
                save_arrays(out/"metadata"/(task+"-"+split+"-rows.npz"), {"source_row": selected, **{k+"_length": v for k,v in lengths.items()}})
                for value in arrays.values():
                    if isinstance(value, np.memmap):
                        value._mmap.close()
            print(json.dumps({"task": task, "split": split, "samples": len(selected)}), flush=True)
    (out/"metadata"/(task+"-dictionary.json")).write_text(json.dumps({"input_symbols": symbols, "label_symbols": label_symbols}, ensure_ascii=False)+"\n", encoding="utf-8")
    return result


def build_pixels(archive, out, task):
    folder = "cifar10" if task == "image" else "pathfinder"
    result = {"task": task, "source_prefix": "lra/"+folder,
              "representation": "unchanged 0..255 grayscale uint8 pixels in row-major order; normalization deferred to frozen runner", "splits": {}}
    for source_split, split in SPLITS.items():
        prefix = "lra/" + folder
        raw_labels = archive.read(prefix+"/label/"+source_split+".label").decode("ascii").splitlines()
        labels = np.asarray([int(v) for v in raw_labels], dtype=np.int64)
        if np.any(labels < 0) or np.any(labels >= (10 if task == "image" else 2)):
            raise ValueError("Invalid pixel label")
        with tempfile.TemporaryDirectory(prefix="rebuild-", dir=out) as temp:
            inputs = np.lib.format.open_memmap(Path(temp)/"inputs.npy", mode="w+", dtype="u1", shape=(len(labels),1024))
            with archive.open(prefix+"/input/"+source_split+".src") as handle:
                n = 0
                for line in handle:
                    values = np.fromstring(line.decode("ascii"), sep=" ", dtype=np.int64)
                    if n >= len(labels) or len(values) != 1024 or np.any(values<0) or np.any(values>255):
                        raise ValueError("Invalid pixel sequence length/range/count")
                    inputs[n] = values
                    n += 1
            if n != len(labels):
                raise ValueError("Pixel/label count mismatch")
            inputs.flush()
            result["splits"][split] = save_arrays(out/task/(split+".npz"), {"inputs":inputs,"labels":labels})
            inputs._mmap.close()
        print(json.dumps({"task":task,"split":split,"samples":len(labels)}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TEXT_TASKS)+("image","pathfinder"), default=list(TEXT_TASKS)+["image","pathfinder"])
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error("Output must be a new version directory; preserve all existing data")
    if sha256_file(args.archive) != ARCHIVE_SHA256:
        parser.error("Archive does not match reviewed MEGA bytes")
    if shutil.disk_usage(args.archive.parent).free < 8*1024**3:
        parser.error("Require 8 GiB free for bounded rebuild")
    out.mkdir(parents=True)
    (out/"metadata").mkdir()
    report = {"schema_version":"qkan-data-build.v1", "profile":"mega-independent-v1", "status":"built_not_qualified", "archive_sha256":ARCHIVE_SHA256,
              "code_sha256":sha256_file(Path(__file__)), "numpy_version":np.__version__, "tasks":[],
              "claim_boundary":"An adapted five-task protocol; historical scores cannot be mixed with this data version."}
    with zipfile.ZipFile(args.archive) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate archive member")
        for task in args.tasks:
            report["tasks"].append(build_indexed(archive,out,task) if task in TEXT_TASKS else build_pixels(archive,out,task))
            (out/"build-manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"built_not_qualified","tasks":len(report['tasks'])}),flush=True)


if __name__ == "__main__":
    main()
