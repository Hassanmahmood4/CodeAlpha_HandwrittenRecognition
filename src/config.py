"""Centralized paths for artifacts."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "mnist_cnn.keras"
TORCH_CKPT = ARTIFACTS_DIR / "mnist_cnn_torch.pt"
METRICS_PATH = ARTIFACTS_DIR / "training_metrics.json"

IMAGE_SIZE = (28, 28)
