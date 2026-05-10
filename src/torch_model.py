"""PyTorch CNN fallback when TensorFlow wheels are unavailable (e.g. Python 3.14+)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    """MNIST-sized CNN aligned with the Keras architecture intent."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect NCHW floats in 0–255 space (matches PIL tensor path).
        x = x / 255.0
        x = self.stem(x)
        return self.head(x)
