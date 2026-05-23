"""Convert LRA pickle files to the .npz layout expected by this project.

Expected input examples:
  lra-text.train.pickle
  lra-text.dev.pickle
  lra-text.test.pickle
  lra-listops.train.pickle
  lra-retrieval.train.pickle
  lra-image.train.pickle
  lra-pathfinder32-curv_contour_length_14.train.pickle

Output layout:
  data/lra/text/train.npz
  data/lra/text/val.npz
  data/lra/text/test.npz
"""
import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


SPLIT_MAP = {
    "train": "train",
    "dev": "val",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def to_array(values: Any) -> np.ndarray:
    return np.asarray(values)


def get_first(data: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"Cannot find any of keys {names}; available keys: {list(data)}")


def extract_arrays(data: Any, task: str) -> Dict[str, np.ndarray]:
    """Extract arrays from common LRA pickle formats."""
    if isinstance(data, dict):
        if task == "retrieval":
            inputs1 = get_first(data, ("inputs1", "input1", "input_ids_0", "x1", "sentence1", "doc1"))
            inputs2 = get_first(data, ("inputs2", "input2", "input_ids_1", "x2", "sentence2", "doc2"))
            labels = get_first(data, ("labels", "label", "targets", "y"))
            return {
                "inputs1": to_array(inputs1),
                "inputs2": to_array(inputs2),
                "labels": to_array(labels),
            }

        inputs = get_first(data, ("inputs", "input", "input_ids_0", "x", "tokens", "features"))
        labels = get_first(data, ("labels", "label", "targets", "y"))
        return {"inputs": to_array(inputs), "labels": to_array(labels)}

    if isinstance(data, tuple) and len(data) == 2:
        inputs, labels = data
        return {"inputs": to_array(inputs), "labels": to_array(labels)}

    if isinstance(data, tuple) and len(data) == 3 and task == "retrieval":
        inputs1, inputs2, labels = data
        return {
            "inputs1": to_array(inputs1),
            "inputs2": to_array(inputs2),
            "labels": to_array(labels),
        }

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            labels = [get_first(item, ("labels", "label", "targets", "y")) for item in data]
            if task == "retrieval":
                inputs1 = [
                    get_first(item, ("inputs1", "input1", "input_ids_0", "x1", "sentence1", "doc1"))
                    for item in data
                ]
                inputs2 = [
                    get_first(item, ("inputs2", "input2", "input_ids_1", "x2", "sentence2", "doc2"))
                    for item in data
                ]
                return {
                    "inputs1": to_array(inputs1),
                    "inputs2": to_array(inputs2),
                    "labels": to_array(labels),
                }

            inputs = [
                get_first(item, ("inputs", "input", "input_ids_0", "x", "tokens", "features"))
                for item in data
            ]
            return {"inputs": to_array(inputs), "labels": to_array(labels)}

        if isinstance(first, tuple) and len(first) == 2:
            inputs, labels = zip(*data)
            return {"inputs": to_array(inputs), "labels": to_array(labels)}

        if isinstance(first, tuple) and len(first) == 3 and task == "retrieval":
            inputs1, inputs2, labels = zip(*data)
            return {
                "inputs1": to_array(inputs1),
                "inputs2": to_array(inputs2),
                "labels": to_array(labels),
            }

    raise TypeError(
        "Unsupported pickle structure. Inspect it with "
        "`python scripts/convert_lra_pickle_to_npz.py --inspect <file>`."
    )


def infer_task_and_split(path: Path) -> Tuple[str, str]:
    name = path.name
    parts = name.split(".")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer split from filename: {name}")

    split = SPLIT_MAP.get(parts[-2])
    if split is None:
        raise ValueError(f"Unknown split `{parts[-2]}` in filename: {name}")

    stem = ".".join(parts[:-2])
    if stem.startswith("lra-"):
        stem = stem[4:]

    if stem.startswith("pathfinder"):
        task = "pathfinder"
    else:
        task = stem.split("-")[0]

    return task, split


def inspect_pickle(path: Path) -> None:
    data = load_pickle(path)
    print(f"path: {path}")
    print(f"type: {type(data)}")
    if isinstance(data, dict):
        print(f"keys: {list(data.keys())}")
        for key, value in data.items():
            arr = np.asarray(value)
            print(f"  {key}: type={type(value)}, shape={arr.shape}, dtype={arr.dtype}")
    elif isinstance(data, (list, tuple)):
        print(f"len: {len(data)}")
        if len(data) > 0:
            print(f"first type: {type(data[0])}")
            print(f"first value preview: {repr(data[0])[:500]}")
    else:
        print(f"preview: {repr(data)[:500]}")


def should_skip_pathfinder(path: Path, pathfinder_size: str) -> bool:
    name = path.name
    return name.startswith("lra-pathfinder") and f"pathfinder{pathfinder_size}" not in name


def convert_file(path: Path, output_dir: Path, skip_existing: bool = False) -> None:
    task, split = infer_task_and_split(path)
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = task_dir / f"{split}.npz"

    if skip_existing and out_path.exists():
        print(f"skip existing: {out_path}")
        return

    arrays = extract_arrays(load_pickle(path), task)
    np.savez_compressed(out_path, **arrays)

    shapes = ", ".join(f"{name}={value.shape}" for name, value in arrays.items())
    print(f"{path.name} -> {out_path} ({shapes})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LRA .pickle files to .npz files.")
    parser.add_argument("input", type=Path, help="Input pickle file or directory")
    parser.add_argument("--output", type=Path, default=Path("data/lra"), help="Output LRA data directory")
    parser.add_argument("--inspect", action="store_true", help="Inspect one pickle file and exit")
    parser.add_argument(
        "--pathfinder-size",
        default="32",
        choices=("32", "64", "128"),
        help="Pathfinder variant to convert when processing a directory",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_pickle(args.input)
        return

    if args.input.is_file():
        paths = [args.input]
    else:
        paths = sorted(args.input.glob("*.pickle")) + sorted(args.input.glob("*.pkl"))
        paths = [
            path for path in paths
            if not should_skip_pathfinder(path, args.pathfinder_size)
        ]

    if not paths:
        raise FileNotFoundError(f"No .pickle or .pkl files found under {args.input}")

    for path in paths:
        convert_file(path, args.output, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
