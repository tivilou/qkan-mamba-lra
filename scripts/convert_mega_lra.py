"""Convert MEGA's processed LRA data to .npz format.

MEGA provides pre-processed LRA data at:
    https://dl.fbaipublicfiles.com/mega/data/lra.zip

After downloading and unzipping, run this script:
    python scripts/convert_mega_lra.py --mega_dir /path/to/lra --out_dir data/lra

MEGA data format:
    lra/
      listops/     input/{train,valid,test}.src + label/{train,valid,test}.label
      imdb/        input/{train,valid,test}.src + label/{train,valid,test}.label
      aan/         input/{train,valid,test}.src + label/{train,valid,test}.label
      cifar10/     input/{train,valid,test}.src + label/{train,valid,test}.label
      pathfinder32/curv_contour_length_14/
                   input/{train,valid,test}.src + label/{train,valid,test}.label
"""
import argparse
from pathlib import Path

import numpy as np


def convert_task(src_dir: Path, out_dir: Path, task_name: str, is_retrieval: bool = False):
    print(f"[{task_name}] Converting...")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train", "valid": "val", "test": "test"}

    for mega_split, our_split in split_map.items():
        src_file = src_dir / "input" / f"{mega_split}.src"
        label_file = src_dir / "label" / f"{mega_split}.label"

        if not src_file.exists():
            print(f"  SKIP {mega_split}: {src_file} not found")
            continue

        labels = []
        with open(label_file) as f:
            for line in f:
                labels.append(int(line.strip()))
        labels = np.array(labels, dtype=np.int64)

        inputs_list = []
        with open(src_file) as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                inputs_list.append(values)

        if is_retrieval:
            half = len(inputs_list[0]) // 2
            inputs1 = np.array([row[:half] for row in inputs_list], dtype=np.float32)
            inputs2 = np.array([row[half:] for row in inputs_list], dtype=np.float32)
            np.savez(out_dir / f"{our_split}.npz",
                     inputs1=inputs1, inputs2=inputs2, labels=labels)
        else:
            inputs = np.array(inputs_list, dtype=np.float32)
            np.savez(out_dir / f"{our_split}.npz", inputs=inputs, labels=labels)

        print(f"  {our_split}: {len(labels)} samples")


def main():
    parser = argparse.ArgumentParser(description="Convert MEGA LRA data to .npz")
    parser.add_argument("--mega_dir", type=str, required=True,
                        help="Path to unzipped MEGA LRA data directory")
    parser.add_argument("--out_dir", type=str, default="data/lra",
                        help="Output directory (default: data/lra)")
    args = parser.parse_args()

    mega_dir = Path(args.mega_dir)
    out_dir = Path(args.out_dir)

    tasks = [
        ("listops", mega_dir / "listops", False),
        ("text", mega_dir / "imdb", False),
        ("retrieval", mega_dir / "aan", True),
        ("image", mega_dir / "cifar10", False),
        ("pathfinder", mega_dir / "pathfinder32" / "curv_contour_length_14", False),
    ]

    for task_name, src_dir, is_retrieval in tasks:
        if src_dir.exists():
            convert_task(src_dir, out_dir / task_name, task_name, is_retrieval)
        else:
            print(f"[{task_name}] SKIP: {src_dir} not found")
        print()

    print("Done! Data saved to:", out_dir)


if __name__ == "__main__":
    main()
