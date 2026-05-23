"""Convert MEGA's processed LRA data to .npz format.

MEGA provides pre-processed LRA data at:
    https://dl.fbaipublicfiles.com/mega/data/lra.zip

After downloading and unzipping, run this script:
    python scripts/convert_mega_lra.py --mega_dir /path/to/lra --out_dir data/lra

MEGA data has two formats:
- Text tasks (listops, imdb-4000, aan): fairseq binary (src-bin/*.bin + *.idx)
- Vision tasks (cifar10, pathfinder): plain text (input/*.src + label/*.label)
"""
import argparse
import struct
from pathlib import Path

import numpy as np


def read_fairseq_bin(src_dir: Path, split: str, max_len: int = None):
    """Read fairseq MMapIndexedDataset binary format."""
    idx_path = src_dir / f"{split}.idx"
    bin_path = src_dir / f"{split}.bin"

    with open(idx_path, "rb") as f:
        _magic = f.read(9)
        _version = struct.unpack("<Q", f.read(8))[0]
        _dtype_code = struct.unpack("<B", f.read(1))[0]
        length = struct.unpack("<Q", f.read(8))[0]
        sizes = np.frombuffer(f.read(length * 4), dtype=np.int32)
        pointers = np.frombuffer(f.read(length * 8), dtype=np.int64)

    with open(bin_path, "rb") as f:
        raw = f.read()

    seq_len = max_len if max_len else int(sizes.max())
    inputs = np.zeros((length, seq_len), dtype=np.float32)

    for i in range(length):
        start = int(pointers[i])
        n = int(sizes[i])
        sample = np.frombuffer(raw[start:start + n * 2], dtype=np.int16)
        actual_len = min(len(sample), seq_len)
        inputs[i, :actual_len] = sample[:actual_len].astype(np.float32)

    return inputs


def convert_fairseq_task(mega_dir: Path, out_dir: Path, task_name: str,
                         src_subdir: str, max_len: int):
    """Convert a fairseq binary format task."""
    print(f"[{task_name}] Converting fairseq binary format...")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = mega_dir / src_subdir / "src-bin"
    label_dir = mega_dir / src_subdir / "label-bin"

    split_map = {"train": "train", "valid": "val", "test": "test"}

    for mega_split, our_split in split_map.items():
        if not (src_dir / f"{mega_split}.idx").exists():
            print(f"  SKIP {mega_split}: not found")
            continue

        inputs = read_fairseq_bin(src_dir, mega_split, max_len)
        label_inputs = read_fairseq_bin(label_dir, mega_split, max_len=1)
        labels = label_inputs[:, 0].astype(np.int64)

        np.savez(out_dir / f"{our_split}.npz", inputs=inputs, labels=labels)
        print(f"  {our_split}: {len(labels)} samples, seq_len={inputs.shape[1]}")


def convert_retrieval_task(mega_dir: Path, out_dir: Path, max_len: int = 4000):
    """Convert AAN retrieval task (two source inputs)."""
    print("[retrieval] Converting fairseq binary format...")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = mega_dir / "aan" / "src-bin"
    src1_dir = mega_dir / "aan" / "src1-bin"
    label_dir = mega_dir / "aan" / "label-bin"
    half_len = max_len // 2

    split_map = {"train": "train", "valid": "val", "test": "test"}

    for mega_split, our_split in split_map.items():
        if not (src_dir / f"{mega_split}.idx").exists():
            print(f"  SKIP {mega_split}: not found")
            continue

        inputs1 = read_fairseq_bin(src_dir, mega_split, half_len)
        inputs2 = read_fairseq_bin(src1_dir, mega_split, half_len)
        label_inputs = read_fairseq_bin(label_dir, mega_split, max_len=1)
        labels = label_inputs[:, 0].astype(np.int64)

        np.savez(out_dir / f"{our_split}.npz",
                 inputs1=inputs1, inputs2=inputs2, labels=labels)
        print(f"  {our_split}: {len(labels)} samples, seq_len={half_len}x2")


def convert_text_task(src_dir: Path, out_dir: Path, task_name: str):
    """Convert plain text format task (cifar10, pathfinder)."""
    print(f"[{task_name}] Converting plain text format...")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train", "valid": "val", "test": "test"}

    for mega_split, our_split in split_map.items():
        src_file = src_dir / "input" / f"{mega_split}.src"
        label_file = src_dir / "label" / f"{mega_split}.label"

        if not src_file.exists():
            print(f"  SKIP {mega_split}: {src_file} not found")
            continue

        labels = np.loadtxt(label_file, dtype=np.int64)
        inputs_list = []
        with open(src_file) as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                inputs_list.append(values)

        inputs = np.array(inputs_list, dtype=np.float32)
        np.savez(out_dir / f"{our_split}.npz", inputs=inputs, labels=labels)
        print(f"  {our_split}: {len(labels)} samples, seq_len={inputs.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description="Convert MEGA LRA data to .npz")
    parser.add_argument("--mega_dir", type=str, required=True,
                        help="Path to unzipped MEGA LRA data directory")
    parser.add_argument("--out_dir", type=str, default="data/lra",
                        help="Output directory (default: data/lra)")
    args = parser.parse_args()

    mega_dir = Path(args.mega_dir)
    out_dir = Path(args.out_dir)

    convert_fairseq_task(mega_dir, out_dir / "listops", "listops", "listops", max_len=2048)
    convert_fairseq_task(mega_dir, out_dir / "text", "text", "imdb-4000", max_len=4000)
    convert_retrieval_task(mega_dir, out_dir / "retrieval", max_len=4000)

    cifar_dir = mega_dir / "cifar10"
    if cifar_dir.exists():
        convert_text_task(cifar_dir, out_dir / "image", "image")
    else:
        print("[image] SKIP: cifar10/ not found")

    pf_dir = mega_dir / "pathfinder"
    if pf_dir.exists():
        convert_text_task(pf_dir, out_dir / "pathfinder", "pathfinder")
    else:
        print("[pathfinder] SKIP: pathfinder/ not found")

    print("\nDone! Data saved to:", out_dir)


if __name__ == "__main__":
    main()
