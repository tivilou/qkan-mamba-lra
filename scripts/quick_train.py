"""Quick training script for single-task testing.

Usage:
    python scripts/quick_train.py --task text --data_dir data/lra --epochs 10
    python scripts/quick_train.py --task listops --data_dir /tmp/lra_fixed --epochs 50

Unlike experiments/lra/train.py which uses YAML configs, this script
takes all parameters via command line for fast iteration.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.optim as optim

from baselines.mamba_only import MambaOnlyModel
from experiments.lra.data_loader import get_dataloaders, TASK_META


def main():
    parser = argparse.ArgumentParser(description="Quick LRA training")
    parser.add_argument("--task", type=str, required=True,
                        choices=["listops", "text", "retrieval", "image", "pathfinder"])
    parser.add_argument("--data_dir", type=str, default="data/lra")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["LRA_DATA_DIR"] = args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    meta = TASK_META[args.task]
    n_classes = meta["n_classes"]

    train_loader, val_loader, test_loader = get_dataloaders(
        args.task, batch_size=args.batch_size, num_workers=2
    )
    print(f"Task: {args.task} | Classes: {n_classes}")
    print(f"Data: train={len(train_loader)} batches, val={len(val_loader)}, test={len(test_loader)}")

    model = MambaOnlyModel(
        d_input=1, d_model=args.d_model, n_layers=args.n_layers,
        d_state=16, d_conv=4, expand=2,
        n_classes=n_classes, pooling="mean"
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Mamba-only: {n_params:,} params")
    print()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(-1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"[{epoch:03d}] train={train_acc:.4f} val={val_acc:.4f} loss={total_loss/total:.4f} time={elapsed:.1f}s")

    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_correct += (logits.argmax(-1) == y).sum().item()
            test_total += y.size(0)
    test_acc = test_correct / test_total

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
