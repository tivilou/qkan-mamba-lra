#!/usr/bin/env python3
"""Bounded train/validation diagnostics; never a formal baseline or test-set score."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from audit_lra_data import ArrayStream, TASKS, sha256_file


def sample_split(root, task, split, limit, seed):
    if split not in ("train", "val"):
        raise ValueError("Qualification pilot must never open the test split")
    path = root / task / (split + ".npz")
    keys = ("inputs1", "inputs2", "labels") if task == "retrieval" else ("inputs", "labels")
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(path))
        streams = {key: ArrayStream(archive, key + ".npy", stack) for key in keys}
        n = streams["labels"].shape[0]
        if any(array.shape[0] != n for array in streams.values()):
            raise ValueError("Mismatched sample counts")
        ids = np.sort(np.random.default_rng(seed).choice(n, min(n, limit), replace=False))
        output = {key: np.empty((len(ids), *array.shape[1:]), dtype=array.dtype) for key,array in streams.items()}
        cursor = 0
        for start in range(0, n, 128):
            end = min(n, start+128)
            right = int(np.searchsorted(ids, end))
            block = {key: array.read(end-start) for key,array in streams.items()}
            for key in keys:
                output[key][cursor:right] = block[key][ids[cursor:right]-start]
            cursor = right
    return output, {"split": split, "samples": len(ids), "population": n,
                    "row_ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                    "npz_sha256": sha256_file(path)}


def bag(rows, width=512):
    output = np.zeros((len(rows), width), dtype=np.float32)
    for i,row in enumerate(rows):
        tokens = np.asarray(row).reshape(-1)
        # IDs 0..3 are special symbols; no numeric distance between token IDs is used.
        tokens = tokens[tokens >= 4].astype(np.int64)
        output[i] = np.bincount(tokens % width, minlength=width)[:width]
    output /= np.maximum(1, np.linalg.norm(output, axis=1, keepdims=True))
    return output


def features(task, data):
    if task in ("image", "pathfinder"):
        return data["inputs"].reshape(len(data["labels"]), -1).astype(np.float32) / 255.0 - 0.5
    if task == "retrieval":
        a,b = bag(data["inputs1"]), bag(data["inputs2"])
        # Symmetric pair features preserve separate documents; no concatenated sequence.
        return np.concatenate((np.abs(a-b), a*b), axis=1)
    return bag(data["inputs"])


def softmax(logits):
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def fit_reference(x, y, classes, seed=20261028, epochs=12):
    rng = np.random.default_rng(seed)
    # Fixed linear reference: zero initialization, minibatch SGD, no validation tuning.
    weights = np.zeros((x.shape[1], classes), dtype=np.float32)
    bias = np.zeros(classes, dtype=np.float32)
    for _ in range(epochs):
        for rows in np.array_split(rng.permutation(len(y)), max(1, (len(y)+63)//64)):
            probability = softmax(x[rows] @ weights + bias)
            probability[np.arange(len(rows)), y[rows]] -= 1
            probability /= len(rows)
            weights -= 0.25 * (x[rows].T @ probability + 1e-4 * weights)
            bias -= 0.25 * probability.sum(axis=0)
    return weights,bias


def wilson_lower(correct, total):
    z = 1.96
    p = correct / total
    return (p + z*z/(2*total) - z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))) / (1+z*z/total)


def evaluate_expression(tokens, symbols):
    stack = []
    answer = None
    for token in tokens:
        word = symbols[int(token)]
        if word in ("<s>", "<pad>", "</s>"):
            continue
        if word.startswith("["):
            stack.append([word, []])
        elif word.isdigit() and len(word) == 1:
            if not stack:
                raise ValueError("Digit outside expression")
            stack[-1][1].append(int(word))
        elif word == "]":
            operator,values = stack.pop()
            if not values:
                raise ValueError("Empty ListOps operation")
            operations = {"[MIN": min, "[MAX": max, "[MED": lambda x: int(np.median(x)), "[SM": lambda x: sum(x)%10}
            result = operations[operator](values)
            if stack:
                stack[-1][1].append(result)
            elif answer is None:
                answer = result
            else:
                raise ValueError("Multiple root expressions")
        else:
            raise ValueError("Unknown ListOps symbol")
    if stack or answer is None:
        raise ValueError("Incomplete/truncated ListOps expression")
    return answer


def pilot(root, train_limit=4096, val_limit=1024, tasks=None):
    report = {"schema": "qkan-qualification-pilot.v1", "formal_baseline_allowed": False,
              "build_manifest_sha256": sha256_file(root/"build-manifest.json"),
              "contract": {"seed": 20261028, "train_limit": train_limit, "val_limit": val_limit,
                           "test_access": False, "epochs": 12, "batch_size": 64,
                           "reference": "fixed linear softmax SGD; paired token bags or fixed grayscale pixels",
                           "selection": "none; fixed final epoch", "signal_gate": "Wilson 95% lower accuracy > validation accuracy of training-majority predictor + 0.02",
                           "null": "same training features, deterministically shuffled training labels"}, "tasks": []}
    for task in (TASKS if tasks is None else tasks):
        classes = TASKS[task]
        train,tproof = sample_split(root,task,"train",train_limit,20261028)
        val,vproof = sample_split(root,task,"val",val_limit,20261029)
        x,vx = features(task,train),features(task,val)
        y,vy = train["labels"],val["labels"]
        majority = int(np.bincount(y,minlength=classes).argmax())
        majority_accuracy = float(np.mean(vy == majority))
        predictions = []
        for labels in (y, np.random.default_rng(20261030).permutation(y)):
            weights,bias = fit_reference(x,labels,classes)
            predictions.append((vx@weights+bias).argmax(axis=1))
        correct = int(np.count_nonzero(predictions[0] == vy))
        record = {"task": task, "train": tproof, "validation": vproof,
                  "accuracy": correct/len(vy), "majority_accuracy": majority_accuracy,
                  "shuffled_label_accuracy": float(np.mean(predictions[1] == vy)),
                  "wilson_lower": float(wilson_lower(correct,len(vy))),
                  "signal_detected": bool(wilson_lower(correct,len(vy)) > majority_accuracy+0.02)}
        if task == "listops":
            symbols = json.loads((root/"metadata/source-v1/listops-dictionary.json").read_text())["input_symbols"]
            mismatches = invalid = 0
            for row,label in zip(val["inputs"],vy):
                try:
                    mismatches += evaluate_expression(row,symbols) != int(label)
                except (ValueError,KeyError,IndexError):
                    invalid += 1
            record["input_only_reference"] = {"samples": len(vy), "mismatches": int(mismatches), "invalid_expressions": invalid,
                                              "role": "symbol/label alignment; deterministic parser is not a learned model"}
        report["tasks"].append(record)
        print(json.dumps(record), flush=True)
        del train,val,x,vx
    report["status"] = "diagnostic_only"
    report["limitations"] = "A negative weak-reference pilot is inconclusive for a difficult sequence task. It cannot reject learnability or certify neural-model correctness. Sampled row outcomes are not formal method results."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite a pilot identity")
    report = pilot(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
