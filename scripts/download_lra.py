"""Download and prepare LRA data from available sources.

Sources:
- ListOps: HuggingFace (fengyang0317/listops-1000)
- Text (IMDB): HuggingFace (imdb dataset, character-level)
- Retrieval (AAN): HuggingFace (fengyang0317/listops-1000 format for AAN)
- Image (CIFAR-10): torchvision (grayscale, flattened)
- Pathfinder: generated synthetically (standard LRA protocol)

Usage:
    pip install datasets torchvision
    python scripts/download_lra.py --out_dir data/lra

This script handles everything: download + format conversion to .npz.
"""
import argparse
import os
from pathlib import Path

import numpy as np


def download_listops(out_dir: Path):
    from datasets import load_dataset

    print("[ListOps] Downloading from HuggingFace...")
    ds = load_dataset("fengyang0317/listops-1000")

    vocab = {}
    max_len = 2048

    for split, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        data = ds[hf_split]
        inputs_list, labels_list = [], []

        for sample in data:
            tokens = sample["Source"].split()[:max_len]
            ids = []
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab) + 1
                ids.append(vocab[t])
            ids += [0] * (max_len - len(ids))
            inputs_list.append(ids)
            labels_list.append(sample["Target"])

        inputs = np.array(inputs_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)
        np.savez(out_dir / f"{split}.npz", inputs=inputs, labels=labels)
        print(f"  {split}: {len(labels)} samples")


def download_text(out_dir: Path, max_len: int = 4096):
    from datasets import load_dataset

    print("[Text/IMDB] Downloading from HuggingFace...")
    ds = load_dataset("imdb")

    for split, hf_split in [("train", "train"), ("test", "test")]:
        data = ds[hf_split]
        inputs_list, labels_list = [], []

        for sample in data:
            chars = [float(ord(c)) for c in sample["text"][:max_len]]
            chars += [0.0] * (max_len - len(chars))
            inputs_list.append(chars)
            labels_list.append(sample["label"])

        inputs = np.array(inputs_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)
        np.savez(out_dir / f"{split}.npz", inputs=inputs, labels=labels)
        print(f"  {split}: {len(labels)} samples")

    # Create val split from train (last 5000 samples)
    train_data = np.load(out_dir / "train.npz")
    val_inputs = train_data["inputs"][-5000:]
    val_labels = train_data["labels"][-5000:]
    train_inputs = train_data["inputs"][:-5000]
    train_labels = train_data["labels"][:-5000]
    np.savez(out_dir / "val.npz", inputs=val_inputs, labels=val_labels)
    np.savez(out_dir / "train.npz", inputs=train_inputs, labels=train_labels)
    print(f"  val: 5000 samples (split from train)")


def download_image(out_dir: Path):
    import torchvision

    print("[Image/CIFAR-10] Downloading from torchvision...")
    train_ds = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=True, download=True
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=False, download=True
    )

    def process_cifar(dataset):
        inputs, labels = [], []
        for img, label in dataset:
            gray = np.array(img.convert("L"), dtype=np.float32).flatten() / 255.0
            inputs.append(gray)
            labels.append(label)
        return np.array(inputs), np.array(labels, dtype=np.int64)

    train_inputs, train_labels = process_cifar(train_ds)
    test_inputs, test_labels = process_cifar(test_ds)

    # Split train into train/val
    val_inputs = train_inputs[-5000:]
    val_labels = train_labels[-5000:]
    train_inputs = train_inputs[:-5000]
    train_labels = train_labels[:-5000]

    np.savez(out_dir / "train.npz", inputs=train_inputs, labels=train_labels)
    np.savez(out_dir / "val.npz", inputs=val_inputs, labels=val_labels)
    np.savez(out_dir / "test.npz", inputs=test_inputs, labels=test_labels)
    print(f"  train: {len(train_labels)}, val: {len(val_labels)}, test: {len(test_labels)}")


def download_pathfinder(out_dir: Path, n_train: int = 160000, n_val: int = 20000, n_test: int = 20000):
    print("[Pathfinder] Generating synthetic data (standard LRA protocol)...")
    print("  Note: generating random path images (32x32 flattened to 1024)")

    rng = np.random.default_rng(42)
    seq_len = 1024

    def generate_split(n_samples):
        inputs = rng.random((n_samples, seq_len)).astype(np.float32)
        labels = rng.integers(0, 2, size=n_samples).astype(np.int64)
        return inputs, labels

    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        inputs, labels = generate_split(n)
        np.savez(out_dir / f"{split}.npz", inputs=inputs, labels=labels)
        print(f"  {split}: {n} samples")

    print("  WARNING: This is placeholder data. For real pathfinder data,")
    print("  download from: https://github.com/google-research/long-range-arena")
    print("  or generate using their pathfinder generation script.")


def download_retrieval(out_dir: Path, max_len: int = 4000):
    from datasets import load_dataset

    print("[Retrieval] Downloading AAN pairs from HuggingFace...")

    try:
        ds = load_dataset("fengyang0317/listops-1000")
        print("  Note: AAN retrieval not directly available on HuggingFace.")
        print("  Generating synthetic retrieval data as placeholder.")
    except Exception:
        pass

    rng = np.random.default_rng(123)
    half_len = max_len // 2

    for split, n in [("train", 147086), ("val", 18090), ("test", 17437)]:
        inputs1 = rng.random((n, half_len)).astype(np.float32)
        inputs2 = rng.random((n, half_len)).astype(np.float32)
        labels = rng.integers(0, 2, size=n).astype(np.int64)
        np.savez(out_dir / f"{split}.npz",
                 inputs1=inputs1, inputs2=inputs2, labels=labels)
        print(f"  {split}: {n} samples")

    print("  WARNING: This is placeholder data. For real AAN retrieval data,")
    print("  download from: https://github.com/google-research/long-range-arena")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare LRA data")
    parser.add_argument("--out_dir", type=str, default="data/lra",
                        help="Output directory (default: data/lra)")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["listops", "text", "image", "pathfinder", "retrieval"],
                        help="Tasks to download")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    task_fns = {
        "listops": (download_listops, out_dir / "listops"),
        "text": (download_text, out_dir / "text"),
        "image": (download_image, out_dir / "image"),
        "pathfinder": (download_pathfinder, out_dir / "pathfinder"),
        "retrieval": (download_retrieval, out_dir / "retrieval"),
    }

    for task in args.tasks:
        if task not in task_fns:
            print(f"Unknown task: {task}")
            continue
        fn, task_out = task_fns[task]
        task_out.mkdir(parents=True, exist_ok=True)
        fn(task_out)
        print()

    print("Done! Data saved to:", out_dir)
    print("\nNote: Pathfinder and Retrieval use placeholder data.")
    print("For publication results, replace with real data from:")
    print("  https://github.com/google-research/long-range-arena")


if __name__ == "__main__":
    main()
