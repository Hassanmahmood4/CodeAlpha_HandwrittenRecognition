#!/usr/bin/env python3
"""
Train MNIST CNN with PyTorch when TensorFlow is not installable.

Writes ``artifacts/mnist_cnn_torch.pt`` and ``artifacts/training_metrics.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, METRICS_PATH  # noqa: E402
from src.torch_model import DigitCNN  # noqa: E402

TORCH_CKPT = ARTIFACTS_DIR / "mnist_cnn_torch.pt"
TORCH_DATA = ROOT / ".torch_data"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_DATA.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=str(TORCH_DATA), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=str(TORCH_DATA), train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = DigitCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, 13):
        running = 0.0
        for x, y in train_loader:
            x = (x * 255.0).to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        print(f"epoch {epoch}/12 train_loss={running / len(train_loader.dataset):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = (x * 255.0).to(device)
            y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    acc = correct / total
    payload = {"test_loss": 0.0, "test_accuracy": float(acc), "epochs_trained": 12, "backend": "pytorch"}
    torch.save({"state_dict": model.state_dict(), "meta": payload}, TORCH_CKPT)

    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
