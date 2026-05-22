"""LRA (Long Range Arena) data loading for all 5 tasks."""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path(os.environ.get("LRA_DATA_DIR", "./data/lra"))


class LRADataset(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        self.inputs = torch.from_numpy(inputs).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class RetrievalDataset(Dataset):
    def __init__(self, inputs1: np.ndarray, inputs2: np.ndarray, labels: np.ndarray):
        self.inputs1 = torch.from_numpy(inputs1).float()
        self.inputs2 = torch.from_numpy(inputs2).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs1[idx], self.inputs2[idx], self.labels[idx]


def load_listops(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / "listops" / f"{split}.npz"
    data = np.load(path)
    return data["inputs"], data["labels"]


def load_text(split: str = "train", max_len: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / "text" / f"{split}.npz"
    data = np.load(path)
    inputs = data["inputs"][:, :max_len]
    return inputs, data["labels"]


def load_retrieval(split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = DATA_DIR / "retrieval" / f"{split}.npz"
    data = np.load(path)
    return data["inputs1"], data["inputs2"], data["labels"]


def load_image(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / "image" / f"{split}.npz"
    data = np.load(path)
    return data["inputs"], data["labels"]


def load_pathfinder(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / "pathfinder" / f"{split}.npz"
    data = np.load(path)
    return data["inputs"], data["labels"]


TASK_LOADERS = {
    "listops": load_listops,
    "text": load_text,
    "retrieval": load_retrieval,
    "image": load_image,
    "pathfinder": load_pathfinder,
}

TASK_META = {
    "listops": {"d_input": 1, "n_classes": 10, "seq_len": 2048},
    "text": {"d_input": 1, "n_classes": 2, "seq_len": 4096},
    "retrieval": {"d_input": 1, "n_classes": 2, "seq_len": 4000},
    "image": {"d_input": 1, "n_classes": 10, "seq_len": 1024},
    "pathfinder": {"d_input": 1, "n_classes": 2, "seq_len": 1024},
}


def get_dataloaders(
    task: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    meta = TASK_META[task]

    if task == "retrieval":
        splits = {}
        for s in ["train", "val", "test"]:
            in1, in2, labels = load_retrieval(s)
            in1 = in1[..., np.newaxis] if in1.ndim == 2 else in1
            in2 = in2[..., np.newaxis] if in2.ndim == 2 else in2
            splits[s] = RetrievalDataset(in1, in2, labels)
    else:
        loader_fn = TASK_LOADERS[task]
        splits = {}
        for s in ["train", "val", "test"]:
            inputs, labels = loader_fn(s)
            if inputs.ndim == 2:
                inputs = inputs[..., np.newaxis]
            splits[s] = LRADataset(inputs, labels)

    loaders = {}
    for s in ["train", "val", "test"]:
        loaders[s] = DataLoader(
            splits[s],
            batch_size=batch_size,
            shuffle=(s == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders["train"], loaders["val"], loaders["test"]
