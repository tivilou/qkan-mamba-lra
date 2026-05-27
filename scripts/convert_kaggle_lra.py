"""Convert Kaggle LRA pickle data to .npz format.

Kaggle dataset: https://www.kaggle.com/datasets/a24998667/long-range-arena-processed

Usage:
    python scripts/convert_kaggle_lra.py --archive data/archive.zip --out_dir data/lra
    python scripts/convert_kaggle_lra.py --archive data/archive.zip --out_dir data/lra --tasks listops image

Processes one file at a time to minimize memory usage.
"""
import argparse
import gc
import pickle
import tempfile
import zipfile
from pathlib import Path

import numpy as np


TASK_FILES = {
    "listops": ("lra-listops", False),
    "text": ("lra-text", False),
    "retrieval": ("lra-retrieval", True),
    "image": ("lra-image", False),
    "pathfinder": ("lra-pathfinder32-curv_contour_length_14", False),
}

SPLIT_MAP = {"train": "train", "dev": "val", "test": "test"}


def convert_one_file(pickle_path: str, out_path: Path, is_retrieval: bool):
    """Load pickle, convert to .npz, free memory."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    n = len(data)
    seq_len = data[0]["input_ids_0"].shape[0]

    if is_retrieval and "input_ids_1" in data[0]:
        inputs1 = np.empty((n, seq_len), dtype=np.float32)
        inputs2 = np.empty((n, seq_len), dtype=np.float32)
        labels = np.empty(n, dtype=np.int64)
        for i, s in enumerate(data):
            inputs1[i] = s["input_ids_0"]
            inputs2[i] = s["input_ids_1"]
            labels[i] = int(s["label"])
        del data
        gc.collect()
        np.savez(out_path, inputs1=inputs1, inputs2=inputs2, labels=labels)
    else:
        inputs = np.empty((n, seq_len), dtype=np.float32)
        labels = np.empty(n, dtype=np.int64)
        for i, s in enumerate(data):
            inputs[i] = s["input_ids_0"]
            labels[i] = int(s["label"])
        del data
        gc.collect()
        np.savez(out_path, inputs=inputs, labels=labels)

    print(f"    -> {n} samples, seq_len={seq_len}")


def main():
    parser = argparse.ArgumentParser(description="Convert Kaggle LRA pickles to .npz")
    parser.add_argument("--archive", type=str, default=None,
                        help="Path to archive.zip")
    parser.add_argument("--pickle_dir", type=str, default=None,
                        help="Path to extracted pickle files")
    parser.add_argument("--out_dir", type=str, default="data/lra",
                        help="Output directory (default: data/lra)")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=list(TASK_FILES.keys()),
                        help="Tasks to convert")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.archive:
        zf = zipfile.ZipFile(args.archive)
        tmp_dir = Path(tempfile.mkdtemp())

        for task in args.tasks:
            prefix, is_retrieval = TASK_FILES[task]
            task_out = out_dir / task
            task_out.mkdir(parents=True, exist_ok=True)
            print(f"[{task}] Converting...")

            for raw_split, our_split in SPLIT_MAP.items():
                fname = f"{prefix}.{raw_split}.pickle"
                if fname not in zf.namelist():
                    print(f"  SKIP {raw_split}: {fname} not in archive")
                    continue
                tmp_path = tmp_dir / fname
                print(f"  Extracting {fname}...")
                zf.extract(fname, tmp_dir)
                convert_one_file(str(tmp_path), task_out / f"{our_split}.npz", is_retrieval)
                tmp_path.unlink()
            print()

        zf.close()

    elif args.pickle_dir:
        pickle_dir = Path(args.pickle_dir)
        for task in args.tasks:
            prefix, is_retrieval = TASK_FILES[task]
            task_out = out_dir / task
            task_out.mkdir(parents=True, exist_ok=True)
            print(f"[{task}] Converting...")

            for raw_split, our_split in SPLIT_MAP.items():
                fname = f"{prefix}.{raw_split}.pickle"
                pickle_path = pickle_dir / fname
                if not pickle_path.exists():
                    print(f"  SKIP {raw_split}: {fname} not found")
                    continue
                convert_one_file(str(pickle_path), task_out / f"{our_split}.npz", is_retrieval)
            print()
    else:
        print("Error: provide --archive or --pickle_dir")
        return

    print("Done! Data saved to:", out_dir)


if __name__ == "__main__":
    main()
