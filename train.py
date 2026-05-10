#!/usr/bin/env python3
"""
Handwritten digit recognition on MNIST using a convolutional neural network (PyTorch).
Matches the internship brief (CNN on MNIST); EMNIST letters can use the same architecture with adjusted classes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_model import MnistCNN


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
TORCH_DATA = Path(__file__).resolve().parent / ".torch_data"


def main():
    parser = argparse.ArgumentParser(description="Train MNIST handwritten digit CNN (PyTorch).")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    TORCH_DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(
        root=str(TORCH_DATA),
        train=True,
        download=True,
        transform=tfm,
    )
    test_ds = datasets.MNIST(
        root=str(TORCH_DATA),
        train=False,
        download=True,
        transform=tfm,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = MnistCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} — train loss: {train_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print(f"\nTest accuracy: {test_acc:.4f}")

    ckpt = ARTIFACTS / "mnist_cnn.pt"
    torch.save({"model_state": model.state_dict(), "arch": "MnistCNN"}, ckpt)
    print(f"Saved weights to {ckpt}")


if __name__ == "__main__":
    main()
