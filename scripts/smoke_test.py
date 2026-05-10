#!/usr/bin/env python3
"""Ensure artifacts exist and inference backend returns valid softmax outputs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import METRICS_PATH, MODEL_PATH, TORCH_CKPT  # noqa: E402
from src.inference import load_digit_predictor  # noqa: E402


def main() -> None:
    if not MODEL_PATH.is_file() and not TORCH_CKPT.is_file():
        subprocess.check_call([sys.executable, str(ROOT / "scripts/train_model.py")])

    assert METRICS_PATH.is_file(), "metrics missing after training"

    predictor, backend = load_digit_predictor()
    assert predictor is not None, f"predictor failed: {backend}"

    zeros = np.zeros((1, 28, 28, 1), dtype=np.float32)
    probs = predictor(zeros)
    assert probs.shape == (1, 10)
    assert abs(float(probs.sum()) - 1.0) < 5e-3
    print(f"smoke_ok backend={backend} sum={float(probs.sum()):.4f}")


if __name__ == "__main__":
    main()
