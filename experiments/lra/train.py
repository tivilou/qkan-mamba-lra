"""Training loop for LRA experiments."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mamba.model import QKANMambaModel
from baselines.transformer import TransformerModel
from baselines.s4 import S4Model
from baselines.mamba_only import MambaOnlyModel
from experiments.lra.data_loader import get_dataloaders, TASK_META


MODEL_REGISTRY = {
    "qkan_mamba": QKANMambaModel,
    "transformer": TransformerModel,
    "s4": S4Model,
    "mamba_only": MambaOnlyModel,
}


def build_model(config: dict) -> nn.Module:
    model_name = config["model"]["name"]
    model_cls = MODEL_REGISTRY[model_name]
    task_meta = TASK_META[config["task"]]
    params = {
        "d_input": task_meta["d_input"],
        "n_classes": task_meta["n_classes"],
        **config["model"]["params"],
    }
    return model_cls(**params)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if len(batch) == 3:
            x1, x2, y = batch
            x = torch.cat([x1, x2], dim=1).to(device)
        else:
            x, y = batch
            x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if len(batch) == 3:
            x1, x2, y = batch
            x = torch.cat([x1, x2], dim=1).to(device)
        else:
            x, y = batch
            x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def run(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        task=config["task"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
    )

    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model']['name']} | Params: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time": elapsed,
        }
        history.append(record)
        print(
            f"[{epoch:03d}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"loss={train_loss:.4f} time={elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), results_dir / "best_model.pt")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    summary = {
        "task": config["task"],
        "model": config["model"]["name"],
        "seed": seed,
        "n_params": n_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "history": history,
    }
    out_path = results_dir / f"{config['task']}_{config['model']['name']}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train on LRA benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed
    run(config)


if __name__ == "__main__":
    main()

