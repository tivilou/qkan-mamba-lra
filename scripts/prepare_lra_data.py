"""Preprocess raw LRA data into .npz format for training.

LRA dataset source: https://github.com/google-research/long-range-arena

Usage:
    python scripts/prepare_lra_data.py --raw_dir /path/to/lra_release --out_dir data/lra

Expected raw directory structure (after downloading from LRA repo):
    lra_release/
        listops-1000/
            basic_train.tsv
            basic_val.tsv
            basic_test.tsv
        tsv_data/
            imdb_reviews.train.tsv  (or similar)
        ...

Output format (per task):
    data/lra/<task>/train.npz  (keys: inputs, labels)
    data/lra/<task>/val.npz
    data/lra/<task>/test.npz
"""
import argparse
import os
from pathlib import Path

import numpy as np


def tokenize_chars(text: str, max_len: int) -> np.ndarray:
    tokens = [ord(c) for c in text[:max_len]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.float32)


def process_listops(raw_dir: Path, out_dir: Path):
    print("Processing ListOps...")
    task_dir = raw_dir / "listops-1000"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": "basic_train.tsv",
        "val": "basic_val.tsv",
        "test": "basic_test.tsv",
    }
    max_len = 2048
    vocab = {}

    for split, fname in splits.items():
        path = task_dir / fname
        if not path.exists():
            print(f"  SKIP {split}: {path} not found")
            continue

        inputs_list, labels_list = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                label = int(parts[1])
                tokens_str = parts[0].split()
                token_ids = []
                for t in tokens_str[:max_len]:
                    if t not in vocab:
                        vocab[t] = len(vocab) + 1
                    token_ids.append(vocab[t])
                if len(token_ids) < max_len:
                    token_ids += [0] * (max_len - len(token_ids))
                inputs_list.append(token_ids)
                labels_list.append(label)

        inputs = np.array(inputs_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)
        np.savez(out_dir / f"{split}.npz", inputs=inputs, labels=labels)
        print(f"  {split}: {len(labels)} samples")


def process_text(raw_dir: Path, out_dir: Path, max_len: int = 4096):
    print("Processing Text (IMDB)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": "new_aan_pairs.train.tsv",
        "val": "new_aan_pairs.eval.tsv",
        "test": "new_aan_pairs.test.tsv",
    }
    text_dir = raw_dir / "tsv_data"

    for split, fname in splits.items():
        path = text_dir / fname
        if not path.exists():
            path = raw_dir / "lra_release" / fname
        if not path.exists():
            print(f"  SKIP {split}: file not found")
            continue

        inputs_list, labels_list = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                text = parts[0]
                label = int(parts[1])
                tokens = tokenize_chars(text, max_len)
                inputs_list.append(tokens)
                labels_list.append(label)

        inputs = np.array(inputs_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)
        np.savez(out_dir / f"{split}.npz", inputs=inputs, labels=labels)
        print(f"  {split}: {len(labels)} samples")


def process_retrieval(raw_dir: Path, out_dir: Path, max_len: int = 4000):
    print("Processing Retrieval...")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": "new_aan_pairs.train.tsv",
        "val": "new_aan_pairs.eval.tsv",
        "test": "new_aan_pairs.test.tsv",
    }
    retrieval_dir = raw_dir / "tsv_data"

    for split, fname in splits.items():
        path = retrieval_dir / fname
        if not path.exists():
            print(f"  SKIP {split}: file not found")
            continue

        inputs1_list, inputs2_list, labels_list = [], [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                text1, text2 = parts[0], parts[1]
                label = int(parts[2])
                t1 = tokenize_chars(text1, max_len // 2)
                t2 = tokenize_chars(text2, max_len // 2)
                inputs1_list.append(t1)
                inputs2_list.append(t2)
                labels_list.append(label)

        inputs1 = np.array(inputs1_list, dtype=np.float32)
        inputs2 = np.array(inputs2_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)
        np.savez(out_dir / f"{split}.npz",
                 inputs1=inputs1, inputs2=inputs2, labels=labels)
        print(f"  {split}: {len(labels)} samples")


def process_image(raw_dir: Path, out_dir: Path):
    print("Processing Image (CIFAR-10)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": "train", "val": "val", "test": "test"}
    img_dir = raw_dir / "image"

    for split, prefix in splits.items():
        path = img_dir / f"{prefix}.npz"
        if not path.exists():
            print(f"  SKIP {split}: {path} not found")
            print(f"  Tip: convert CIFAR-10 to grayscale, flatten to (N, 1024)")
            continue
        data = np.load(path)
        inputs = data["inputs"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        if inputs.ndim == 3:
            inputs = inputs.reshape(inputs.shape[0], -1)
        np.savez(out_dir / f"{split}.npz", inputs=inputs, labels=labels)
        print(f"  {split}: {len(labels)} samples")


def process_pathfinder(raw_dir: Path, out_dir: Path):
    print("Processing Pathfinder...")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": "train", "val": "val", "test": "test"}
    pf_dir = raw_dir / "pathfinder"

    for split, prefix in splits.items():
        path = pf_dir / f"{prefix}.npz"
        if not path.exists():
            print(f"  SKIP {split}: {path} not found")
            print(f"  Tip: flatten pathfinder images to (N, 1024)")
            continue
        data = np.load(path)
        inputs = data["inputs"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        if inputs.ndim == 3:
            inputs = inputs.reshape(inputs.shape[0], -1)
        np.savez(out_dir / f"{split}.npz", inputs=inputs, labels=labels)
        print(f"  {split}: {len(labels)} samples")


def main():
    parser = argparse.ArgumentParser(description="Preprocess LRA data to .npz")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Path to raw LRA data (lra_release/)")
    parser.add_argument("--out_dir", type=str, default="data/lra",
                        help="Output directory (default: data/lra)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    process_listops(raw_dir, out_dir / "listops")
    process_text(raw_dir, out_dir / "text")
    process_retrieval(raw_dir, out_dir / "retrieval")
    process_image(raw_dir, out_dir / "image")
    process_pathfinder(raw_dir, out_dir / "pathfinder")

    print("\nDone! Data saved to:", out_dir)


if __name__ == "__main__":
    main()
